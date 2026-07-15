#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "throwdb/server/collection_engine.hpp"

#include "../src/core/sha256.hpp"
#include "../src/server/idempotency_journal.hpp"

namespace {

namespace server = throwdb::server;
namespace journal_detail = throwdb::server::detail;

constexpr std::uint32_t kDimension = 4;
constexpr std::uint64_t kRecordSize = kDimension * 2;

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

void test_prepare_with_unretained_matching_wal_fails_closed() {
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
            collection->materialize_frozen_wals().materialized,
            "matching WAL was not materialized out of retained WALs");
    }

    require_throws<throwdb::StorageError>(
        [&] {
            server::SingleCollectionEngine recovered(*collection, store);
        },
        "unverifiable PREPARE was falsely accepted after WAL reclamation");
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

    server::SingleCollectionEngine recovered(*collection, store);
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
            "prepare_with_unretained_matching_wal_fails_closed",
            &test_prepare_with_unretained_matching_wal_fails_closed},
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
