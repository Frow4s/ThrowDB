#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "throwdb/server/collection_engine.hpp"

#include "../src/core/sha256.hpp"
#include "../src/core/native_collection_test_hooks.hpp"
#include "../src/server/idempotency_journal.hpp"

namespace {

namespace server = throwdb::server;
namespace journal_detail = throwdb::server::detail;

constexpr std::uint32_t kDimension = 4;
constexpr std::uint64_t kRecordSize = kDimension * 2;
constexpr std::string_view kPendingFenceName = "pending-mutation-v1";

std::atomic<journal_detail::IdempotencyJournalTestHookPoint> fault_point{
    journal_detail::IdempotencyJournalTestHookPoint::PrepareSynced};

void throw_at_journal_sync(
    const journal_detail::IdempotencyJournalTestHookPoint point) {
    if (point == fault_point.load(std::memory_order_acquire)) {
        throw throwdb::StorageError("simulated post-fsync journal failure");
    }
}

class JournalHookGuard {
public:
    explicit JournalHookGuard(
        const journal_detail::IdempotencyJournalTestHookPoint point) {
        fault_point.store(point, std::memory_order_release);
        journal_detail::set_idempotency_journal_test_hook(
            &throw_at_journal_sync);
    }
    ~JournalHookGuard() {
        journal_detail::set_idempotency_journal_test_hook(nullptr);
    }
    JournalHookGuard(const JournalHookGuard&) = delete;
    JournalHookGuard& operator=(const JournalHookGuard&) = delete;
};

std::atomic<bool> lifecycle_hook_entered{false};
std::atomic<bool> release_lifecycle_hook{false};
std::atomic<bool> lifecycle_fence_hook_entered{false};
std::atomic<bool> release_lifecycle_fence_hook{false};
std::atomic<bool> orphan_plan_hook_entered{false};
std::atomic<bool> release_orphan_plan_hook{false};

void blocking_materialization_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation
            != NativeCollectionTestOperation::MaterializeFrozenWals
        || context.point
               != NativeCollectionTestHookPoint::LifecycleBuildUnlocked) {
        return;
    }
    lifecycle_hook_entered.store(true, std::memory_order_release);
    while (!release_lifecycle_hook.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void blocking_before_lifecycle_fence_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation
            != NativeCollectionTestOperation::MaterializeFrozenWals
        || context.point
               != NativeCollectionTestHookPoint::BeforeLifecycleFenceCheck) {
        return;
    }
    lifecycle_fence_hook_entered.store(true, std::memory_order_release);
    while (!release_lifecycle_fence_hook.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void blocking_orphan_plan_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation != NativeCollectionTestOperation::CollectOrphans
        || context.point
               != NativeCollectionTestHookPoint::BeforeOrphanRevalidation) {
        return;
    }
    orphan_plan_hook_entered.store(true, std::memory_order_release);
    while (!release_orphan_plan_hook.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

class BlockingNativeHookThreadGuard {
public:
    BlockingNativeHookThreadGuard(
        std::atomic<bool>& release,
        const throwdb::detail::NativeCollectionTestHook hook,
        std::thread& worker) noexcept
        : release_(release), worker_(worker) {
        release_.store(false, std::memory_order_relaxed);
        throwdb::detail::set_native_collection_test_hook(hook);
    }

    ~BlockingNativeHookThreadGuard() {
        release_and_join();
    }

    BlockingNativeHookThreadGuard(const BlockingNativeHookThreadGuard&) = delete;
    BlockingNativeHookThreadGuard& operator=(
        const BlockingNativeHookThreadGuard&) = delete;

    void release_and_join() noexcept {
        release_.store(true, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
        if (active_) {
            throwdb::detail::set_native_collection_test_hook(nullptr);
            active_ = false;
        }
    }

private:
    std::atomic<bool>& release_;
    std::thread& worker_;
    bool active_{true};
};

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-idempotency-test-"
                   + std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count())
                   + "-" + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Error, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
    try {
        std::forward<Callback>(callback)();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

[[nodiscard]] std::vector<std::byte> row(const std::uint8_t salt) {
    std::vector<std::byte> output(kRecordSize);
    for (std::size_t index = 0; index != output.size(); ++index) {
        output[index] = static_cast<std::byte>(
            salt + static_cast<std::uint8_t>(17 * index));
    }
    return output;
}

void test_sha256_matches_standard_known_vector() {
    const std::array<std::byte, 3> input{
        std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    constexpr std::array<std::byte, 32> expected{
        std::byte{0xba}, std::byte{0x78}, std::byte{0x16}, std::byte{0xbf},
        std::byte{0x8f}, std::byte{0x01}, std::byte{0xcf}, std::byte{0xea},
        std::byte{0x41}, std::byte{0x41}, std::byte{0x40}, std::byte{0xde},
        std::byte{0x5d}, std::byte{0xae}, std::byte{0x22}, std::byte{0x23},
        std::byte{0xb0}, std::byte{0x03}, std::byte{0x61}, std::byte{0xa3},
        std::byte{0x96}, std::byte{0x17}, std::byte{0x7a}, std::byte{0x9c},
        std::byte{0xb4}, std::byte{0x10}, std::byte{0xff}, std::byte{0x61},
        std::byte{0xf2}, std::byte{0x00}, std::byte{0x15}, std::byte{0xad}};
    require(
        throwdb::detail::sha256(input) == expected,
        "SHA-256 one-shot result differs from the standard abc vector");

    throwdb::detail::Sha256 incremental;
    incremental.update(std::span(input).first(1));
    incremental.update(std::span(input).subspan(1, 1));
    incremental.update(std::span(input).last(1));
    require(
        incremental.finish() == expected,
        "SHA-256 incremental result differs from the standard abc vector");
}

[[nodiscard]] std::unique_ptr<throwdb::NativeCollection> make_store(
    const std::filesystem::path& store) {
    throwdb::NativeBuilderOptions options;
    options.format_version = 7;
    options.dimension = kDimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 2 * kRecordSize;

    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    const std::array<std::uint64_t, 1> ids{10};
    const auto embeddings = row(1);
    builder->append(ids, embeddings);
    builder->finish();
    return throwdb::NativeCollection::open(store, {}, true);
}

[[nodiscard]] journal_detail::JournalPreparedOperation prepare_insert(
    const throwdb::NativeCollection& collection,
    const server::OperationId operation_id,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    journal_detail::JournalPreparedOperation prepared;
    prepared.operation_id = operation_id;
    prepared.request_digest = journal_detail::mutation_digest(
        server::MutationKind::Insert,
        collection.dimension(),
        collection.dtype(),
        ids,
        embeddings);
    prepared.wal_fingerprint =
        throwdb::NativeCollection::fingerprint_mutation(
            throwdb::NativeMutationOperation::Put,
            ids,
            embeddings,
            collection.record_size());
    prepared.kind = server::MutationKind::Insert;
    prepared.expected_sequence =
        collection.identity().committed_sequence + 1;
    prepared.row_count = ids.size();
    return prepared;
}

void test_replay_survives_restart_and_conflicting_digest_is_rejected() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x11, 0x22};
    const std::array<std::uint64_t, 1> ids{20};
    const auto embeddings = row(23);

    std::uint64_t sequence = 0;
    {
        server::SingleCollectionEngine engine(*collection, store);
        const auto first = engine.mutate(
            server::MutationKind::Insert, operation, ids, embeddings);
        require(!first.replayed, "first mutation was reported as replayed");
        require(first.commit_sequence != 0, "first mutation has zero sequence");
        sequence = first.commit_sequence;

        const auto replay = engine.mutate(
            server::MutationKind::Insert, operation, ids, embeddings);
        require(replay.replayed, "same operation was not replayed");
        require(
            replay.commit_sequence == sequence,
            "in-process replay changed commit sequence");
        require(
            collection->identity().committed_sequence == sequence,
            "in-process replay committed another WAL transaction");
    }

    server::SingleCollectionEngine restarted(*collection, store);
    const auto replay = restarted.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(replay.replayed, "restarted engine forgot completed operation");
    require(
        replay.commit_sequence == sequence
            && collection->identity().committed_sequence == sequence,
        "restart replay changed commit sequence");

    const auto changed_embeddings = row(99);
    require_throws<server::IdempotencyConflictError>(
        [&] {
            static_cast<void>(restarted.mutate(
                server::MutationKind::Insert,
                operation,
                ids,
                changed_embeddings));
        },
        "same operation_id accepted a different request digest");
}

void test_prepare_without_wal_is_aborted_and_retryable() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x31, 0x32};
    const std::array<std::uint64_t, 1> ids{30};
    const auto embeddings = row(31);

    {
        const auto identity = collection->identity();
        auto journal = journal_detail::IdempotencyJournal::open(
            store,
            identity.collection_id_low,
            identity.collection_id_high);
        journal->prepare(prepare_insert(
            *collection, operation, ids, embeddings));
    }

    const auto before = collection->identity().committed_sequence;
    server::SingleCollectionEngine recovered(*collection, store);
    const auto result = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(!result.replayed, "PREPARE without WAL was falsely replayed");
    require(
        result.commit_sequence == before + 1,
        "retry after PREPARE-only recovery did not commit once");
}

void test_prepare_and_matching_wal_is_completed_during_recovery() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x41, 0x42};
    const std::array<std::uint64_t, 1> ids{40};
    const auto embeddings = row(41);

    std::uint64_t committed_sequence = 0;
    {
        const auto identity = collection->identity();
        auto journal = journal_detail::IdempotencyJournal::open(
            store,
            identity.collection_id_low,
            identity.collection_id_high);
        journal->prepare(prepare_insert(
            *collection, operation, ids, embeddings));
        collection->insert(ids, embeddings);
        committed_sequence = collection->identity().committed_sequence;
    }

    server::SingleCollectionEngine recovered(*collection, store);
    const auto result = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(result.replayed, "matching WAL was not synthesized as COMPLETE");
    require(
        result.commit_sequence == committed_sequence,
        "recovered COMPLETE returned the wrong sequence");
    require(
        collection->identity().committed_sequence == committed_sequence,
        "matching-WAL replay committed another transaction");
}

void test_prepare_sequence_occupied_by_another_mutation_fails_closed() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x51, 0x52};
    const std::array<std::uint64_t, 1> intended_ids{50};
    const auto intended_embeddings = row(51);

    {
        const auto identity = collection->identity();
        auto journal = journal_detail::IdempotencyJournal::open(
            store,
            identity.collection_id_low,
            identity.collection_id_high);
        journal->prepare(prepare_insert(
            *collection, operation, intended_ids, intended_embeddings));

        const std::array<std::uint64_t, 1> other_ids{500};
        const auto other_embeddings = row(52);
        collection->insert(other_ids, other_embeddings);
    }

    require_throws<throwdb::StorageError>(
        [&] {
            server::SingleCollectionEngine recovered(*collection, store);
        },
        "sequence occupied by another mutation did not fail closed");
}

void test_pending_prepare_blocks_wal_materialization_until_recovery() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x71, 0x72};
    const std::array<std::uint64_t, 1> ids{70};
    const auto embeddings = row(71);

    {
        const auto identity = collection->identity();
        auto journal = journal_detail::IdempotencyJournal::open(
            store,
            identity.collection_id_low,
            identity.collection_id_high);
        journal->prepare(prepare_insert(
            *collection, operation, ids, embeddings));
        collection->insert(ids, embeddings);
        require(
            collection->rotate_wal().rotated,
            "matching WAL did not rotate before materialization");
        require(
            std::filesystem::exists(
                store / "server" / kPendingFenceName),
            "durable PREPARE did not publish its lifecycle fence");
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(collection->materialize_frozen_wals());
            },
            "pending PREPARE did not block WAL materialization");
    }

    server::SingleCollectionEngine recovered(*collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "recovered COMPLETE did not remove the lifecycle fence");
    const auto replay = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(replay.replayed, "recovered pending mutation was not replayable");
    require(
        collection->materialize_frozen_wals().materialized,
        "materialization stayed blocked after pending recovery");
}

void test_prepare_post_sync_failure_keeps_fence_until_restart() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x81, 0x82};
    const std::array<std::uint64_t, 1> ids{80};
    const auto embeddings = row(81);
    const auto before = collection->identity().committed_sequence;

    {
        server::SingleCollectionEngine engine(*collection, store);
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::PrepareSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "post-sync PREPARE fault was not surfaced");
        require(
            collection->identity().committed_sequence == before,
            "PREPARE fault unexpectedly committed a WAL transaction");
        require(
            std::filesystem::exists(
                store / "server" / kPendingFenceName),
            "ambiguous PREPARE append removed its lifecycle fence");
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "poisoned journal accepted another live mutation");
    }

    server::SingleCollectionEngine recovered(*collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "restart did not abort PREPARE-only operation and clear its fence");
    const auto committed = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(
        !committed.replayed && committed.commit_sequence == before + 1,
        "retry after PREPARE-only recovery did not commit exactly once");
}

void test_complete_post_sync_failure_recovers_without_duplicate_complete() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x91, 0x92};
    const std::array<std::uint64_t, 1> ids{90};
    const auto embeddings = row(91);
    std::uint64_t committed_sequence = 0;

    {
        server::SingleCollectionEngine engine(*collection, store);
        const auto before = collection->identity().committed_sequence;
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::CompleteSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "post-sync COMPLETE fault was not surfaced");
        committed_sequence = collection->identity().committed_sequence;
        require(
            committed_sequence == before + 1,
            "COMPLETE fault lost the already-durable WAL transaction");
        require(
            std::filesystem::exists(
                store / "server" / kPendingFenceName),
            "ambiguous COMPLETE append removed its lifecycle fence");
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "poisoned COMPLETE path attempted a duplicate journal append");
    }

    server::SingleCollectionEngine recovered(*collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "restart did not clear the stale post-COMPLETE fence");
    const auto replay = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(
        replay.replayed && replay.commit_sequence == committed_sequence,
        "durable COMPLETE was not replayed exactly once after restart");
}

void test_failed_fence_cleanup_is_retried_before_completed_replay() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x93, 0x94};
    const std::array<std::uint64_t, 1> ids{93};
    const auto embeddings = row(94);
    server::SingleCollectionEngine engine(*collection, store);
    const auto before = collection->identity().committed_sequence;

    {
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::FenceRemove);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "simulated fence-unlink failure was not surfaced");
    }
    const auto committed_sequence =
        collection->identity().committed_sequence;
    require(
        committed_sequence == before + 1,
        "fence cleanup failure lost the committed WAL transaction");
    require(
        std::filesystem::exists(store / "server" / kPendingFenceName),
        "fence cleanup fault did not leave a fail-closed marker");

    const auto replay = engine.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(
        replay.replayed && replay.commit_sequence == committed_sequence,
        "completed replay was not preserved after fence reconciliation");
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "completed replay bypassed stale-fence reconciliation");
}

void test_operation_history_cap_preserves_existing_replays() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::EngineOptions options;
    options.idempotency_journal_max_operations = 1;
    server::SingleCollectionEngine engine(*collection, store, options);

    constexpr server::OperationId first_operation{0xb1, 0xb2};
    const std::array<std::uint64_t, 1> first_ids{110};
    const auto first_embeddings = row(111);
    const auto first = engine.mutate(
        server::MutationKind::Insert,
        first_operation,
        first_ids,
        first_embeddings);
    const auto stats = engine.idempotency_stats();
    require(
        stats.enabled && stats.completed_operations == 1
            && stats.maximum_operations == 1 && !stats.pending,
        "idempotency cap stats do not describe the completed history");

    constexpr server::OperationId second_operation{0xb3, 0xb4};
    const std::array<std::uint64_t, 1> second_ids{120};
    const auto second_embeddings = row(121);
    require_throws<server::IdempotencyCapacityError>(
        [&] {
            static_cast<void>(engine.mutate(
                server::MutationKind::Insert,
                second_operation,
                second_ids,
                second_embeddings));
        },
        "new operation ID was accepted after the history cap");
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "history-cap rejection left a pending fence");

    const auto replay = engine.mutate(
        server::MutationKind::Insert,
        first_operation,
        first_ids,
        first_embeddings);
    require(
        replay.replayed && replay.commit_sequence == first.commit_sequence,
        "history cap prevented replay of an existing operation ID");
}

void test_journal_checkpoint_rotates_replay_domain_atomically() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    const auto identity = collection->identity();
    auto journal = journal_detail::IdempotencyJournal::open(
        store, identity.collection_id_low, identity.collection_id_high);
    const auto original_domain = journal->domain();
    require(!original_domain.empty(), "journal created an empty replay domain");

    constexpr server::OperationId operation{0xc1, 0xc2};
    const std::array<std::uint64_t, 1> ids{130};
    const auto embeddings = row(131);
    auto prepared = prepare_insert(
        *collection, operation, ids, embeddings);
    journal->prepare(prepared);
    static_cast<void>(journal->complete(
        operation, prepared.expected_sequence));
    require(
        journal->completed(operation) != nullptr,
        "checkpoint fixture did not retain completed operation");

    const auto next_domain = journal->checkpoint();
    require(
        !next_domain.empty() && next_domain != original_domain,
        "checkpoint did not rotate the replay domain");
    const auto stats = journal->stats();
    require(
        stats.domain == next_domain && stats.generation == 2
            && stats.bytes == 4096 && stats.completed_operations == 0
            && stats.aborted_operations == 0 && !stats.pending,
        "checkpoint did not publish one empty bounded replay generation");
    journal.reset();

    auto reopened = journal_detail::IdempotencyJournal::open(
        store, identity.collection_id_low, identity.collection_id_high);
    const auto reopened_stats = reopened->stats();
    require(
        reopened->domain() == next_domain
            && reopened_stats.generation == stats.generation
            && reopened->completed(operation) == nullptr,
        "reopen did not select the checkpointed replay domain");
}

void test_engine_checkpoint_rejects_delayed_domain_and_accepts_current() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::EngineOptions engine_options;
    engine_options.idempotency_journal_max_operations = 2;
    server::SingleCollectionEngine engine(
        *collection, store, engine_options);
    const server::OperationId original_domain =
        engine.idempotency_domain();
    constexpr server::OperationId first_operation{0xc3, 0xc4};
    const std::array<std::uint64_t, 1> first_ids{135};
    const auto first_embeddings = row(136);
    static_cast<void>(engine.mutate_in_domain(
        server::MutationKind::Insert, first_operation, original_domain,
        first_ids, first_embeddings));

    server::EngineMaintenanceOptions maintenance;
    maintenance.max_actions = 1;
    maintenance.idempotency_checkpoint_trigger_percent = 50;
    maintenance.clean_segments = false;
    maintenance.reclaim_orphans = false;
    const auto checkpoint = engine.maintain_once(maintenance);
    const server::OperationId current_domain =
        engine.idempotency_domain();
    require(
        checkpoint.idempotency_checkpointed
            && checkpoint.actions_attempted == 1
            && checkpoint.actions_performed == 1
            && current_domain != original_domain
            && checkpoint.idempotency_after.generation == 2,
        "automatic maintenance did not checkpoint the replay history");

    constexpr server::OperationId delayed_operation{0xc5, 0xc6};
    const std::array<std::uint64_t, 1> delayed_ids{145};
    const auto delayed_embeddings = row(146);
    require_throws<server::IdempotencyDomainError>(
        [&] {
            static_cast<void>(engine.mutate_in_domain(
                server::MutationKind::Insert, delayed_operation,
                original_domain, delayed_ids, delayed_embeddings));
        },
        "a delayed mutation from the retired replay domain was accepted");
    require(
        collection->contains(delayed_ids) == std::vector<std::uint8_t>{0},
        "replay-domain rejection happened after the mutation commit");

    const auto committed = engine.mutate_in_domain(
        server::MutationKind::Insert, delayed_operation, current_domain,
        delayed_ids, delayed_embeddings);
    require(
        committed.commit_sequence != 0 && !committed.replayed,
        "the current replay domain was not accepted");
}

void test_engine_maintenance_clean_store_is_noop() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::SingleCollectionEngine engine(*collection, store);
    server::EngineMaintenanceOptions options;
    options.rotate_wal_bytes = 1;
    options.rotate_wal_transactions = 1;
    options.fold_min_runs = 1;
    options.clean_segments = false;
    options.reclaim_orphans = false;

    const auto result = engine.maintain_once(options);
    require(
        result.actions_attempted == 0 && result.actions_performed == 0
            && !result.rotated && !result.materialized && !result.folded
            && !result.action_budget_reached,
        "maintenance changed a clean collection");
    require(
        result.before.committed_sequence == result.after.committed_sequence
            && result.after.active_wal_transaction_count == 0,
        "no-op maintenance changed mutation-log state");
}

void test_cleaner_zero_thresholds_dry_run_and_interval() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::SingleCollectionEngine engine(*collection, store);
    server::EngineMaintenanceOptions options;
    options.max_actions = 1;
    options.cleaner_min_interval_seconds = 3'600;
    options.cleaner_min_garbage_rows = 0;
    options.cleaner_min_reclaimable_bytes = 0;
    options.cleaner_min_garbage_ratio_ppm = 0;
    options.reclaim_orphans = false;

    const auto first = engine.maintain_once(options);
    require(
        first.actions_attempted == 1 && first.cleaner_attempted
            && first.cleaner.dry_run && !first.cleaner.candidate_found,
        "zero cleaner thresholds did not produce a safe dry-run pass");

    const auto deferred = engine.maintain_once(options);
    require(
        deferred.actions_attempted == 0 && !deferred.cleaner_attempted,
        "cleaner minimum interval did not suppress an immediate rescan");

    options.cleaner_min_interval_seconds = 0;
    const auto immediate = engine.maintain_once(options);
    require(
        immediate.actions_attempted == 1 && immediate.cleaner_attempted
            && immediate.cleaner.dry_run,
        "zero cleaner interval did not permit an explicit immediate pass");
}

void test_operational_stats_are_nonblocking_during_maintenance() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::SingleCollectionEngine engine(*collection, store);
    const std::array<std::uint64_t, 1> ids{131};
    const auto embeddings = row(132);
    static_cast<void>(engine.mutate(
        server::MutationKind::Insert,
        server::OperationId{0xf1, 0xf2}, ids, embeddings));
    require(collection->rotate_wal().rotated,
            "nonblocking-stats fixture did not freeze a WAL");

    lifecycle_hook_entered.store(false, std::memory_order_relaxed);
    std::exception_ptr maintenance_failure;
    std::thread maintenance;
    BlockingNativeHookThreadGuard hook_guard(
        release_lifecycle_hook, &blocking_materialization_hook, maintenance);
    maintenance = std::thread([&] {
        try {
            server::EngineMaintenanceOptions options;
            options.max_actions = 1;
            options.clean_segments = false;
            options.reclaim_orphans = false;
            static_cast<void>(engine.maintain_once(options));
        } catch (...) {
            maintenance_failure = std::current_exception();
        }
    });

    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(5);
    while (!lifecycle_hook_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!lifecycle_hook_entered.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "maintenance did not reach the blocking lifecycle hook");
    }

    auto stats_future = std::async(std::launch::async, [&] {
        return engine.operational_stats();
    });
    const bool returned_promptly =
        stats_future.wait_for(std::chrono::milliseconds(500))
        == std::future_status::ready;
    hook_guard.release_and_join();
    const auto stats = stats_future.get();
    if (maintenance_failure != nullptr) {
        std::rethrow_exception(maintenance_failure);
    }
    require(
        returned_promptly,
        "operational_stats blocked behind the lifecycle mutation mutex");
    require(
        stats.maintenance_active,
        "cached operational stats hid active maintenance");
}

void test_mutations_continue_during_materialization_and_pending_survives_reopen() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    auto engine = std::make_unique<server::SingleCollectionEngine>(
        *collection, store);

    const std::array<std::uint64_t, 1> frozen_ids{201};
    const auto frozen_embeddings = row(201);
    static_cast<void>(engine->mutate(
        server::MutationKind::Insert,
        server::OperationId{0x201, 0x202},
        frozen_ids,
        frozen_embeddings));
    require(
        collection->rotate_wal().rotated,
        "concurrent-maintenance fixture did not freeze its source WAL");

    lifecycle_hook_entered.store(false, std::memory_order_relaxed);
    server::EngineMaintenanceResult maintenance_result;
    std::exception_ptr maintenance_failure;
    std::thread maintenance;
    BlockingNativeHookThreadGuard hook_guard(
        release_lifecycle_hook, &blocking_materialization_hook, maintenance);
    maintenance = std::thread([&] {
        try {
            server::EngineMaintenanceOptions options;
            options.max_actions = 1;
            options.clean_segments = false;
            options.reclaim_orphans = false;
            maintenance_result = engine->maintain_once(options);
        } catch (...) {
            maintenance_failure = std::current_exception();
        }
    });

    const auto hook_deadline = std::chrono::steady_clock::now()
                               + std::chrono::seconds(5);
    while (!lifecycle_hook_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < hook_deadline) {
        std::this_thread::yield();
    }
    if (!lifecycle_hook_entered.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "materialization did not reach its durable reservation hook");
    }

    const std::array<std::uint64_t, 1> live_ids{202};
    const auto live_embeddings = row(202);
    std::atomic<bool> live_mutation_started{false};
    std::atomic<bool> live_mutation_completed{false};
    server::MutationResult live_mutation_result;
    std::exception_ptr live_mutation_failure;
    std::thread live_mutation([&] {
        live_mutation_started.store(true, std::memory_order_release);
        try {
            live_mutation_result = engine->mutate(
                server::MutationKind::Insert,
                server::OperationId{0x203, 0x204},
                live_ids,
                live_embeddings);
        } catch (...) {
            live_mutation_failure = std::current_exception();
        }
        live_mutation_completed.store(true, std::memory_order_release);
    });
    const auto mutation_deadline = std::chrono::steady_clock::now()
                                   + std::chrono::seconds(5);
    while (!live_mutation_completed.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < mutation_deadline) {
        std::this_thread::yield();
    }
    const bool mutation_completed_while_paused =
        live_mutation_completed.load(std::memory_order_acquire);
    if (!mutation_completed_while_paused) {
        release_lifecycle_hook.store(true, std::memory_order_release);
    }
    live_mutation.join();
    if (!mutation_completed_while_paused || live_mutation_failure != nullptr) {
        if (live_mutation_failure != nullptr) {
            std::rethrow_exception(live_mutation_failure);
        }
        throw std::runtime_error(
            "mutation blocked behind paused materialization");
    }
    require(
        live_mutation_started.load(std::memory_order_acquire)
            && !live_mutation_result.replayed
            && live_mutation_result.commit_sequence != 0,
        "mutation concurrent with materialization was not committed");
    require(
        collection->get_bytes(live_ids) == live_embeddings,
        "concurrent mutation was not visible before materialization publish");

    // Leave a durable COMPLETE whose fence cleanup is ambiguous while the
    // older frozen prefix is still being materialized.  The selected prefix
    // must remain independent of this active-WAL transaction, and recovery
    // must still be able to verify it after a full native reopen.
    constexpr server::OperationId pending_operation{0x205, 0x206};
    const std::array<std::uint64_t, 1> pending_ids{203};
    const auto pending_embeddings = row(203);
    {
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::CompleteSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine->mutate(
                    server::MutationKind::Insert,
                    pending_operation,
                    pending_ids,
                    pending_embeddings));
            },
            "pending fixture did not fail after its durable COMPLETE");
    }
    require(
        std::filesystem::exists(store / "server" / kPendingFenceName),
        "ambiguous concurrent mutation did not retain its lifecycle fence");
    require(
        collection->get_bytes(pending_ids) == pending_embeddings,
        "ambiguous concurrent mutation lost its durable WAL value");

    hook_guard.release_and_join();
    if (maintenance_failure != nullptr) {
        std::rethrow_exception(maintenance_failure);
    }
    require(
        maintenance_result.materialized,
        "paused maintenance did not publish its selected frozen WAL");

    engine.reset();
    collection.reset();
    throwdb::NativeOpenOptions recovery_options;
    recovery_options.allow_pending_mutation_recovery = true;
    auto reopened_collection = throwdb::NativeCollection::open(
        store, {}, true, false, recovery_options);
    server::SingleCollectionEngine recovered(*reopened_collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "reopen did not reconcile the concurrent pending mutation fence");
    const auto replay = recovered.mutate(
        server::MutationKind::Insert,
        pending_operation,
        pending_ids,
        pending_embeddings);
    require(
        replay.replayed,
        "concurrent pending mutation was not exactly-once replayable after reopen");
    require(
        reopened_collection->get_bytes(frozen_ids) == frozen_embeddings
            && reopened_collection->get_bytes(live_ids) == live_embeddings
            && reopened_collection->get_bytes(pending_ids)
                   == pending_embeddings,
        "concurrent mutations did not survive native collection reopen");
}

void test_mutation_continues_while_orphan_inventory_is_paused() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    auto engine = std::make_unique<server::SingleCollectionEngine>(
        *collection, store);
    const auto orphan =
        store / "index" / "runs"
        / "run-00000000000000010000000000000002.idx";
    {
        std::ofstream output(orphan, std::ios::binary | std::ios::trunc);
        output << "orphan";
        require(output.good(), "cannot create orphan inventory fixture");
    }

    orphan_plan_hook_entered.store(false, std::memory_order_relaxed);
    server::EngineMaintenanceResult maintenance_result;
    std::exception_ptr maintenance_failure;
    std::thread maintenance;
    BlockingNativeHookThreadGuard hook_guard(
        release_orphan_plan_hook, &blocking_orphan_plan_hook, maintenance);
    maintenance = std::thread([&] {
        try {
            server::EngineMaintenanceOptions options;
            options.max_actions = 1;
            options.checkpoint_idempotency = false;
            options.clean_segments = false;
            options.reclaim_orphans = true;
            options.orphan_min_age_ns = 0;
            maintenance_result = engine->maintain_once(options);
        } catch (...) {
            maintenance_failure = std::current_exception();
        }
    });

    const auto hook_deadline = std::chrono::steady_clock::now()
                               + std::chrono::seconds(5);
    while (!orphan_plan_hook_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < hook_deadline) {
        std::this_thread::yield();
    }
    if (!orphan_plan_hook_entered.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "orphan maintenance did not pause after its unlocked inventory");
    }

    const std::array<std::uint64_t, 1> ids{204};
    const auto embeddings = row(204);
    server::MutationResult mutation_result;
    std::exception_ptr mutation_failure;
    std::atomic<bool> mutation_completed{false};
    std::thread mutation([&] {
        try {
            mutation_result = engine->mutate(
                server::MutationKind::Insert,
                server::OperationId{0x207, 0x208}, ids, embeddings);
        } catch (...) {
            mutation_failure = std::current_exception();
        }
        mutation_completed.store(true, std::memory_order_release);
    });
    const auto mutation_deadline = std::chrono::steady_clock::now()
                                   + std::chrono::seconds(5);
    while (!mutation_completed.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < mutation_deadline) {
        std::this_thread::yield();
    }
    const bool completed_while_inventory_paused =
        mutation_completed.load(std::memory_order_acquire);
    if (!completed_while_inventory_paused) {
        release_orphan_plan_hook.store(true, std::memory_order_release);
    }
    mutation.join();
    if (!completed_while_inventory_paused || mutation_failure != nullptr) {
        if (mutation_failure != nullptr) {
            std::rethrow_exception(mutation_failure);
        }
        throw std::runtime_error(
            "mutation blocked behind unlocked orphan inventory");
    }
    require(
        mutation_result.commit_sequence != 0 && !mutation_result.replayed,
        "mutation concurrent with orphan inventory did not commit");

    hook_guard.release_and_join();
    if (maintenance_failure != nullptr) {
        std::rethrow_exception(maintenance_failure);
    }
    require(
        maintenance_result.orphan_reclaim_attempted
            && maintenance_result.orphan_removed_files == 1
            && !std::filesystem::exists(orphan),
        "two-phase orphan maintenance did not reclaim its planned file");

    engine.reset();
    collection.reset();
    auto reopened = throwdb::NativeCollection::open(store, {}, true);
    require(
        reopened->get_bytes(ids) == embeddings,
        "mutation concurrent with orphan inventory did not survive reopen");
}

void test_orphan_apply_defers_when_prepare_appears_after_plan() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    auto engine = std::make_unique<server::SingleCollectionEngine>(
        *collection, store);
    const auto orphan =
        store / "index" / "runs"
        / "run-00000000000000030000000000000004.idx";
    {
        std::ofstream output(orphan, std::ios::binary | std::ios::trunc);
        output << "orphan-fence-race";
        require(output.good(), "cannot create orphan fence-race fixture");
    }

    orphan_plan_hook_entered.store(false, std::memory_order_relaxed);
    server::EngineMaintenanceResult maintenance_result;
    std::exception_ptr maintenance_failure;
    std::thread maintenance;
    BlockingNativeHookThreadGuard hook_guard(
        release_orphan_plan_hook, &blocking_orphan_plan_hook, maintenance);
    maintenance = std::thread([&] {
        try {
            server::EngineMaintenanceOptions options;
            options.max_actions = 1;
            options.checkpoint_idempotency = false;
            options.clean_segments = false;
            options.reclaim_orphans = true;
            options.orphan_min_age_ns = 0;
            maintenance_result = engine->maintain_once(options);
        } catch (...) {
            maintenance_failure = std::current_exception();
        }
    });

    const auto hook_deadline = std::chrono::steady_clock::now()
                               + std::chrono::seconds(5);
    while (!orphan_plan_hook_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < hook_deadline) {
        std::this_thread::yield();
    }
    require(
        orphan_plan_hook_entered.load(std::memory_order_acquire),
        "orphan maintenance did not pause before its second fence check");

    constexpr server::OperationId pending_operation{0x209, 0x20a};
    const std::array<std::uint64_t, 1> pending_ids{205};
    const auto pending_embeddings = row(205);
    {
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::PrepareSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine->mutate(
                    server::MutationKind::Insert,
                    pending_operation,
                    pending_ids,
                    pending_embeddings));
            },
            "orphan fence-race fixture did not retain a durable PREPARE");
    }
    require(
        std::filesystem::exists(store / "server" / kPendingFenceName),
        "durable PREPARE fence was not visible before orphan apply");

    hook_guard.release_and_join();
    if (maintenance_failure != nullptr) {
        std::rethrow_exception(maintenance_failure);
    }
    require(
        maintenance_result.pending_mutation_deferred
            && maintenance_result.orphan_reclaim_attempted
            && maintenance_result.actions_attempted == 1
            && maintenance_result.actions_performed == 0
            && maintenance_result.orphan_removed_files == 0
            && std::filesystem::exists(orphan),
        "orphan apply crossed a PREPARE created after its unlocked plan");

    engine.reset();
    collection.reset();
    throwdb::NativeOpenOptions recovery_options;
    recovery_options.allow_pending_mutation_recovery = true;
    auto reopened_collection = throwdb::NativeCollection::open(
        store, {}, true, false, recovery_options);
    server::SingleCollectionEngine recovered(*reopened_collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "reopen did not resolve the PREPARE that deferred orphan apply");

    server::EngineMaintenanceOptions retry_options;
    retry_options.max_actions = 1;
    retry_options.checkpoint_idempotency = false;
    retry_options.clean_segments = false;
    retry_options.reclaim_orphans = true;
    retry_options.orphan_min_age_ns = 0;
    const auto retry = recovered.maintain_once(retry_options);
    require(
        !retry.pending_mutation_deferred
            && retry.orphan_reclaim_attempted
            && retry.actions_performed == 1
            && retry.orphan_removed_files == 1
            && !std::filesystem::exists(orphan),
        "orphan maintenance did not reclaim the file after fence recovery");
}

void test_pending_fence_admission_race_defers_maintenance() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    auto engine = std::make_unique<server::SingleCollectionEngine>(
        *collection, store);

    const std::array<std::uint64_t, 1> frozen_ids{211};
    const auto frozen_embeddings = row(211);
    static_cast<void>(engine->mutate(
        server::MutationKind::Insert,
        server::OperationId{0x211, 0x212},
        frozen_ids,
        frozen_embeddings));
    require(
        collection->rotate_wal().rotated,
        "fence-admission fixture did not freeze a WAL");
    const auto before = collection->mutation_log_stats();

    lifecycle_fence_hook_entered.store(false, std::memory_order_relaxed);
    server::EngineMaintenanceResult maintenance_result;
    std::exception_ptr maintenance_failure;
    std::thread maintenance;
    BlockingNativeHookThreadGuard hook_guard(
        release_lifecycle_fence_hook,
        &blocking_before_lifecycle_fence_hook,
        maintenance);
    maintenance = std::thread([&] {
        try {
            server::EngineMaintenanceOptions options;
            options.max_actions = 1;
            options.clean_segments = false;
            options.reclaim_orphans = false;
            maintenance_result = engine->maintain_once(options);
        } catch (...) {
            maintenance_failure = std::current_exception();
        }
    });

    const auto hook_deadline = std::chrono::steady_clock::now()
                               + std::chrono::seconds(5);
    while (!lifecycle_fence_hook_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < hook_deadline) {
        std::this_thread::yield();
    }
    if (!lifecycle_fence_hook_entered.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "maintenance did not pause before its lifecycle fence check");
    }

    constexpr server::OperationId pending_operation{0x213, 0x214};
    const std::array<std::uint64_t, 1> pending_ids{212};
    const auto pending_embeddings = row(212);
    {
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::PrepareSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine->mutate(
                    server::MutationKind::Insert,
                    pending_operation,
                    pending_ids,
                    pending_embeddings));
            },
            "fence-admission fixture did not retain an ambiguous PREPARE");
    }
    require(
        std::filesystem::exists(store / "server" / kPendingFenceName),
        "ambiguous PREPARE did not appear in the lifecycle admission window");

    hook_guard.release_and_join();
    if (maintenance_failure != nullptr) {
        std::rethrow_exception(maintenance_failure);
    }
    const auto after = collection->mutation_log_stats();
    require(
        maintenance_result.pending_mutation_deferred
            && maintenance_result.actions_attempted == 1
            && maintenance_result.actions_performed == 0
            && !maintenance_result.materialized,
        "pending fence admission was not reported as a transient deferral");
    require(
        after.generation == before.generation
            && after.frozen_wal_count == before.frozen_wal_count,
        "deferred lifecycle action changed the frozen WAL snapshot");

    engine.reset();
    collection.reset();
    throwdb::NativeOpenOptions recovery_options;
    recovery_options.allow_pending_mutation_recovery = true;
    auto reopened_collection = throwdb::NativeCollection::open(
        store, {}, true, false, recovery_options);
    server::SingleCollectionEngine recovered(*reopened_collection, store);
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "reopen did not resolve the PREPARE that deferred maintenance");

    server::EngineMaintenanceOptions retry_options;
    retry_options.max_actions = 1;
    retry_options.clean_segments = false;
    retry_options.reclaim_orphans = false;
    const auto retry = recovered.maintain_once(retry_options);
    require(
        retry.materialized && !retry.pending_mutation_deferred,
        "maintenance did not make progress after pending-fence recovery");
    const auto committed = recovered.mutate(
        server::MutationKind::Insert,
        pending_operation,
        pending_ids,
        pending_embeddings);
    require(
        !committed.replayed
            && reopened_collection->get_bytes(pending_ids)
                   == pending_embeddings,
        "PREPARE-only operation was not retryable after deferred maintenance");
}

void test_engine_maintenance_actions_are_bounded_across_passes() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::SingleCollectionEngine engine(*collection, store);
    constexpr server::OperationId operation{0xc1, 0xc2};
    const std::array<std::uint64_t, 1> ids{130};
    const auto embeddings = row(131);
    static_cast<void>(engine.mutate(
        server::MutationKind::Insert, operation, ids, embeddings));

    server::EngineMaintenanceOptions one_action;
    one_action.max_actions = 1;
    one_action.rotate_wal_bytes = 1;
    one_action.rotate_wal_transactions = 1;
    one_action.materialize_max_wals = 1;
    one_action.materialize_max_input_bytes = 1ULL << 20;
    one_action.fold_min_runs = 1;
    one_action.fold_max_runs = 1;
    one_action.fold_max_input_bytes = 1ULL << 20;
    one_action.fold_max_input_entries = 1024;
    one_action.clean_segments = false;
    one_action.reclaim_orphans = false;

    const auto rotated = engine.maintain_once(one_action);
    require(
        rotated.actions_attempted == 1 && rotated.actions_performed == 1
            && rotated.rotated && !rotated.materialized
            && rotated.after.frozen_wal_count == 1
            && rotated.action_budget_reached,
        "one-action maintenance did not stop after WAL rotation");

    const auto materialized = engine.maintain_once(one_action);
    require(
        materialized.actions_attempted == 1
            && materialized.actions_performed == 1
            && !materialized.rotated && materialized.materialized
            && materialized.after.frozen_wal_count == 0
            && materialized.after.index_run_count >= 1
            && materialized.action_budget_reached,
        "second bounded pass did not stop after materialization");

    const auto folded = engine.maintain_once(one_action);
    require(
        folded.actions_attempted == 1 && folded.actions_performed == 1
            && folded.folded,
        "third bounded pass did not fold the format-7 run");

    server::EngineMaintenanceOptions orphan_pass = one_action;
    orphan_pass.reclaim_orphans = true;
    orphan_pass.fold_min_runs = 4096;
    orphan_pass.fold_max_runs = 4096;
    orphan_pass.orphan_min_age_ns = 0;
    const auto reclaimed = engine.maintain_once(orphan_pass);
    require(
        reclaimed.actions_attempted == 1
            && reclaimed.orphan_reclaim_attempted
            && reclaimed.orphan_removed_files > 0,
        "bounded orphan pass did not reclaim retired lifecycle files");
}

void test_engine_maintenance_refuses_poisoned_pending_prepare() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::SingleCollectionEngine engine(*collection, store);
    constexpr server::OperationId committed_operation{0xd1, 0xd2};
    const std::array<std::uint64_t, 1> committed_ids{140};
    const auto committed_embeddings = row(141);
    static_cast<void>(engine.mutate(
        server::MutationKind::Insert,
        committed_operation,
        committed_ids,
        committed_embeddings));
    const auto before = collection->mutation_log_stats();

    constexpr server::OperationId pending_operation{0xd3, 0xd4};
    const std::array<std::uint64_t, 1> pending_ids{150};
    const auto pending_embeddings = row(151);
    {
        JournalHookGuard fault(
            journal_detail::IdempotencyJournalTestHookPoint::PrepareSynced);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    pending_operation,
                    pending_ids,
                    pending_embeddings));
            },
            "pending-safety fixture did not poison journal after PREPARE");

        server::EngineMaintenanceOptions options;
        options.max_actions = 1;
        options.rotate_wal_bytes = 1;
        options.rotate_wal_transactions = 1;
        options.reclaim_orphans = false;
        require_throws<throwdb::StorageError>(
            [&] { static_cast<void>(engine.maintain_once(options)); },
            "maintenance crossed an ambiguous PREPARE boundary");
    }
    const auto after = collection->mutation_log_stats();
    require(
        after.generation == before.generation
            && after.active_wal_transaction_count
                   == before.active_wal_transaction_count
            && after.frozen_wal_count == before.frozen_wal_count
            && std::filesystem::exists(
                store / "server" / kPendingFenceName),
        "blocked maintenance changed lifecycle state or removed its fence");
}

void test_disk_reserve_rejects_before_prepare_and_is_retryable() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0xa1, 0xa2};
    const std::array<std::uint64_t, 1> ids{100};
    const auto embeddings = row(101);
    const auto before = collection->identity().committed_sequence;

    server::EngineOptions constrained;
    constrained.mutation_disk_reserve_bytes =
        std::filesystem::space(store).available;
    {
        server::SingleCollectionEngine engine(
            *collection, store, constrained);
        require_throws<server::InsufficientStorageError>(
            [&] {
                static_cast<void>(engine.mutate(
                    server::MutationKind::Insert,
                    operation,
                    ids,
                    embeddings));
            },
            "disk reserve did not reject mutation before PREPARE");
    }
    require(
        collection->identity().committed_sequence == before,
        "disk-admission rejection changed the WAL sequence");
    require(
        !std::filesystem::exists(store / "server" / kPendingFenceName),
        "disk-admission rejection left a pending fence");
    require(
        collection->contains(ids) == std::vector<std::uint8_t>{0},
        "disk-admission rejection changed collection contents");

    server::EngineOptions permitted;
    permitted.mutation_disk_reserve_bytes = 0;
    server::SingleCollectionEngine recovered(*collection, store, permitted);
    const auto result = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(
        !result.replayed && result.commit_sequence == before + 1,
        "disk-admission rejection was not safely retryable");
}

void test_automatic_rotation_keeps_active_wal_below_polling_budget() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::EngineOptions options;
    options.mutation_disk_reserve_bytes = 0;
    options.automatic_rotate_wal_bytes = 1;
    options.automatic_materialize_max_input_bytes = 1ULL << 20;
    server::SingleCollectionEngine engine(*collection, store, options);
    constexpr server::OperationId operation{0xe1, 0xe2};
    const std::array<std::uint64_t, 1> ids{160};
    const auto embeddings = row(161);

    const auto result = engine.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    const auto stats = collection->mutation_log_stats();
    require(
        result.commit_sequence != 0 && stats.frozen_wal_count == 1
            && stats.active_wal_transaction_count == 0,
        "post-commit safety valve did not rotate an oversized active WAL");
}

void test_automatic_rotation_rejects_one_unmaterializable_frame() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::EngineOptions options;
    options.mutation_disk_reserve_bytes = 0;
    options.automatic_rotate_wal_bytes = 4200;
    options.automatic_materialize_max_input_bytes = 4200;
    server::SingleCollectionEngine engine(*collection, store, options);
    constexpr server::OperationId operation{0xe5, 0xe6};
    const std::array<std::uint64_t, 1> ids{180};
    const auto embeddings = row(181);

    require_throws<server::MutationCapacityError>(
        [&] {
            static_cast<void>(engine.mutate(
                server::MutationKind::Insert,
                operation,
                ids,
                embeddings));
        },
        "one WAL frame larger than the materialization budget was accepted");
    require(
        collection->contains(ids) == std::vector<std::uint8_t>{0}
            && !std::filesystem::exists(
                store / "server" / kPendingFenceName),
        "oversized WAL-frame rejection happened after PREPARE or commit");
}

void test_automatic_rotation_happens_before_budget_crossing_commit() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    server::EngineOptions options;
    options.mutation_disk_reserve_bytes = 0;
    options.automatic_rotate_wal_bytes = 4500;
    options.automatic_materialize_max_input_bytes = 4500;
    server::SingleCollectionEngine engine(*collection, store, options);
    const std::array<std::uint64_t, 1> first_ids{190};
    const std::array<std::uint64_t, 1> second_ids{191};
    const auto first_row = row(191);
    const auto second_row = row(192);

    static_cast<void>(engine.mutate(
        server::MutationKind::Insert,
        server::OperationId{0xe7, 0xe8},
        first_ids,
        first_row));
    static_cast<void>(engine.mutate(
        server::MutationKind::Insert,
        server::OperationId{0xe9, 0xea},
        second_ids,
        second_row));
    const auto stats = collection->mutation_log_stats();
    require(
        stats.frozen_wal_count == 1
            && stats.active_wal_transaction_count == 1
            && stats.total_wal_bytes - stats.active_wal_bytes <= 4500,
        "budget-crossing commit was not moved into a fresh WAL before PREPARE");
}

void test_low_disk_materialization_defers_to_orphan_recovery() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0xe3, 0xe4};
    const std::array<std::uint64_t, 1> ids{170};
    const auto embeddings = row(171);
    {
        server::EngineOptions options;
        options.mutation_disk_reserve_bytes = 0;
        server::SingleCollectionEngine engine(*collection, store, options);
        static_cast<void>(engine.mutate(
            server::MutationKind::Insert, operation, ids, embeddings));
    }
    const auto rotated = collection->rotate_wal();
    require(rotated.rotated && rotated.frozen_wal_count == 1,
            "maintenance reserve fixture did not freeze a WAL");

    server::EngineOptions constrained;
    constrained.mutation_disk_reserve_bytes =
        std::filesystem::space(store).available;
    server::SingleCollectionEngine engine(*collection, store, constrained);
    server::EngineMaintenanceOptions maintenance;
    maintenance.max_actions = 1;
    maintenance.clean_segments = false;
    maintenance.reclaim_orphans = true;
    maintenance.orphan_min_age_ns = 0;
    const auto result = engine.maintain_once(maintenance);
    require(
        result.materialization_disk_budget_blocked
            && result.orphan_reclaim_attempted
            && result.orphan_removed_files > 0
            && result.actions_attempted == 1,
        "low-disk materialization did not yield its bounded slot to orphan "
        "recovery");
    require(
        collection->mutation_log_stats().frozen_wal_count == 1,
        "deferred low-disk materialization changed frozen WAL state");
}

void test_low_disk_cleaner_defers_to_orphan_recovery() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    const std::array<std::uint64_t, 1> ids{10};
    const auto embeddings = row(211);
    {
        server::SingleCollectionEngine engine(*collection, store);
        static_cast<void>(engine.mutate(
            server::MutationKind::Update,
            server::OperationId{0xfa, 0xfb}, ids, embeddings));
    }
    require(collection->rotate_wal().rotated,
            "cleaner low-disk fixture did not rotate its WAL");
    require(collection->materialize_frozen_wals().materialized,
            "cleaner low-disk fixture did not materialize its update");

    server::EngineOptions constrained;
    constrained.mutation_disk_reserve_bytes =
        std::filesystem::space(store).available;
    server::SingleCollectionEngine engine(*collection, store, constrained);
    server::EngineMaintenanceOptions options;
    options.max_actions = 1;
    options.fold_min_runs = 4096;
    options.cleaner_min_interval_seconds = 0;
    options.cleaner_min_garbage_rows = 1;
    options.cleaner_min_reclaimable_bytes = 0;
    options.cleaner_min_garbage_ratio_ppm = 0;
    options.reclaim_orphans = true;
    options.orphan_min_age_ns = 0;
    const auto result = engine.maintain_once(options);
    require(
        result.cleaner_attempted && result.cleaner.candidate_found
            && result.cleaner_disk_budget_blocked
            && result.orphan_reclaim_attempted,
        "low-disk cleaner did not report deferral and run orphan recovery");
}

void test_torn_journal_tail_is_truncated_without_losing_result() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    auto collection = make_store(store);
    constexpr server::OperationId operation{0x61, 0x62};
    const std::array<std::uint64_t, 1> ids{60};
    const auto embeddings = row(61);

    std::uint64_t committed_sequence = 0;
    {
        server::SingleCollectionEngine engine(*collection, store);
        committed_sequence = engine.mutate(
            server::MutationKind::Insert,
            operation,
            ids,
            embeddings).commit_sequence;
    }

    const auto journal_path = store / "server" / "idempotency-v1.log";
    const auto valid_size = std::filesystem::file_size(journal_path);
    {
        std::ofstream tail(journal_path, std::ios::binary | std::ios::app);
        require(tail.good(), "cannot open journal to simulate torn tail");
        const std::array<char, 17> partial{
            'T', 'D', 'B', 'I', 'D', 'R', '0', '1',
            '\1', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'};
        tail.write(partial.data(), partial.size());
        tail.flush();
        require(tail.good(), "cannot append simulated torn journal tail");
    }
    require(
        std::filesystem::file_size(journal_path) == valid_size + 17,
        "simulated torn tail has the wrong size");

    server::EngineOptions bounded;
    bounded.idempotency_journal_max_bytes = valid_size;
    server::SingleCollectionEngine recovered(*collection, store, bounded);
    require(
        std::filesystem::file_size(journal_path) == valid_size,
        "journal recovery did not truncate torn tail");
    const auto replay = recovered.mutate(
        server::MutationKind::Insert, operation, ids, embeddings);
    require(replay.replayed, "torn-tail recovery lost completed operation");
    require(
        replay.commit_sequence == committed_sequence
            && collection->identity().committed_sequence == committed_sequence,
        "torn-tail replay changed commit sequence");
}

}  // namespace

int main() {
    const std::array tests{
        std::pair{
            "sha256_matches_standard_known_vector",
            &test_sha256_matches_standard_known_vector},
        std::pair{
            "replay_survives_restart_and_conflicting_digest_is_rejected",
            &test_replay_survives_restart_and_conflicting_digest_is_rejected},
        std::pair{
            "prepare_without_wal_is_aborted_and_retryable",
            &test_prepare_without_wal_is_aborted_and_retryable},
        std::pair{
            "prepare_and_matching_wal_is_completed_during_recovery",
            &test_prepare_and_matching_wal_is_completed_during_recovery},
        std::pair{
            "prepare_sequence_occupied_by_another_mutation_fails_closed",
            &test_prepare_sequence_occupied_by_another_mutation_fails_closed},
        std::pair{
            "pending_prepare_blocks_wal_materialization_until_recovery",
            &test_pending_prepare_blocks_wal_materialization_until_recovery},
        std::pair{
            "prepare_post_sync_failure_keeps_fence_until_restart",
            &test_prepare_post_sync_failure_keeps_fence_until_restart},
        std::pair{
            "complete_post_sync_failure_recovers_without_duplicate_complete",
            &test_complete_post_sync_failure_recovers_without_duplicate_complete},
        std::pair{
            "failed_fence_cleanup_is_retried_before_completed_replay",
            &test_failed_fence_cleanup_is_retried_before_completed_replay},
        std::pair{
            "operation_history_cap_preserves_existing_replays",
            &test_operation_history_cap_preserves_existing_replays},
        std::pair{
            "journal_checkpoint_rotates_replay_domain_atomically",
            &test_journal_checkpoint_rotates_replay_domain_atomically},
        std::pair{
            "engine_checkpoint_rejects_delayed_domain_and_accepts_current",
            &test_engine_checkpoint_rejects_delayed_domain_and_accepts_current},
        std::pair{
            "engine_maintenance_clean_store_is_noop",
            &test_engine_maintenance_clean_store_is_noop},
        std::pair{
            "cleaner_zero_thresholds_dry_run_and_interval",
            &test_cleaner_zero_thresholds_dry_run_and_interval},
        std::pair{
            "operational_stats_are_nonblocking_during_maintenance",
            &test_operational_stats_are_nonblocking_during_maintenance},
        std::pair{
            "mutations_continue_during_materialization_and_pending_survives_reopen",
            &test_mutations_continue_during_materialization_and_pending_survives_reopen},
        std::pair{
            "mutation_continues_while_orphan_inventory_is_paused",
            &test_mutation_continues_while_orphan_inventory_is_paused},
        std::pair{
            "orphan_apply_defers_when_prepare_appears_after_plan",
            &test_orphan_apply_defers_when_prepare_appears_after_plan},
        std::pair{
            "pending_fence_admission_race_defers_maintenance",
            &test_pending_fence_admission_race_defers_maintenance},
        std::pair{
            "engine_maintenance_actions_are_bounded_across_passes",
            &test_engine_maintenance_actions_are_bounded_across_passes},
        std::pair{
            "engine_maintenance_refuses_poisoned_pending_prepare",
            &test_engine_maintenance_refuses_poisoned_pending_prepare},
        std::pair{
            "disk_reserve_rejects_before_prepare_and_is_retryable",
            &test_disk_reserve_rejects_before_prepare_and_is_retryable},
        std::pair{
            "automatic_rotation_keeps_active_wal_below_polling_budget",
            &test_automatic_rotation_keeps_active_wal_below_polling_budget},
        std::pair{
            "automatic_rotation_rejects_one_unmaterializable_frame",
            &test_automatic_rotation_rejects_one_unmaterializable_frame},
        std::pair{
            "automatic_rotation_happens_before_budget_crossing_commit",
            &test_automatic_rotation_happens_before_budget_crossing_commit},
        std::pair{
            "low_disk_materialization_defers_to_orphan_recovery",
            &test_low_disk_materialization_defers_to_orphan_recovery},
        std::pair{
            "low_disk_cleaner_defers_to_orphan_recovery",
            &test_low_disk_cleaner_defers_to_orphan_recovery},
        std::pair{
            "torn_journal_tail_is_truncated_without_losing_result",
            &test_torn_journal_tail_is_truncated_without_losing_result},
    };
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
