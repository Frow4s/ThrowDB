#include "throwdb/server/collection_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <sys/statvfs.h>
#include <unordered_set>
#include <utility>

#include "idempotency_journal.hpp"
#include "throwdb/errors.hpp"

namespace throwdb::server {
namespace {

void validate_distinct_ids(const std::span<const std::uint64_t> ids) {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(ids.size());
    for (const auto id : ids) {
        if (!seen.insert(id).second) {
            throw DuplicateIdError(id);
        }
    }
}

[[nodiscard]] std::uint64_t checked_payload_size(
    const std::size_t count,
    const std::uint64_t record_size) {
    if (count > std::numeric_limits<std::uint64_t>::max()
                    / record_size) {
        throw InvalidArgumentError("mutation payload size overflows uint64");
    }
    return static_cast<std::uint64_t>(count) * record_size;
}

[[nodiscard]] MutationResult public_result(
    const detail::JournalCompletedOperation& operation,
    const bool replayed) {
    return MutationResult{
        operation.commit_sequence,
        operation.prepared.row_count,
        operation.prepared.deleted,
        replayed};
}

[[nodiscard]] std::uint64_t checked_add_bytes(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const description) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw InvalidArgumentError(
            std::string(description) + " overflows uint64");
    }
    return left + right;
}

[[nodiscard]] std::uint64_t estimated_wal_transaction_bytes(
    const std::uint64_t row_count,
    const std::uint64_t embedding_bytes) {
    constexpr std::uint64_t wal_envelope = 128 + 7 + 64;
    if (row_count > std::numeric_limits<std::uint64_t>::max() / 16) {
        throw InvalidArgumentError("mutation WAL operation bytes overflow");
    }
    auto estimate = checked_add_bytes(
        wal_envelope, row_count * 16, "mutation WAL frame estimate");
    return checked_add_bytes(
        estimate, embedding_bytes, "mutation WAL frame estimate");
}

[[nodiscard]] std::uint64_t estimated_mutation_write_bytes(
    const MutationKind kind,
    const std::uint64_t row_count,
    const std::uint64_t embedding_bytes,
    const bool journalled) {
    // The current WAL uses a 128-byte frame header, a 16-byte operation
    // entry per ID, at most seven bytes of alignment, and a 64-byte COMMIT.
    // Journal PREPARE/COMPLETE records each use a 192-byte header and DELETE
    // repeats the result mask. Add filesystem-allocation headroom so a check
    // is not balanced exactly on the last metadata block.
    constexpr std::uint64_t journal_record_header = 192;
    constexpr std::uint64_t pending_fence_bytes = 4096;
    constexpr std::uint64_t allocation_headroom = 64ULL << 10;
    std::uint64_t estimate = estimated_wal_transaction_bytes(
        row_count, embedding_bytes);
    if (journalled) {
        const std::uint64_t result_bytes =
            kind == MutationKind::Delete ? row_count : 0;
        const auto one_record = checked_add_bytes(
            journal_record_header,
            result_bytes,
            "idempotency record estimate");
        estimate = checked_add_bytes(
            estimate, one_record, "mutation write estimate");
        estimate = checked_add_bytes(
            estimate, one_record, "mutation write estimate");
        estimate = checked_add_bytes(
            estimate, pending_fence_bytes, "mutation write estimate");
    }
    return checked_add_bytes(
        estimate, allocation_headroom, "mutation write estimate");
}

[[nodiscard]] std::uint64_t filesystem_available_bytes(
    const std::filesystem::path& path) {
    struct statvfs status {};
    while (::statvfs(path.c_str(), &status) != 0) {
        if (errno == EINTR) {
            continue;
        }
        throw InsufficientStorageError(
            "cannot inspect collection filesystem free space before mutation: "
            + std::string(std::strerror(errno)));
    }
    const std::uint64_t fragment_size =
        status.f_frsize != 0 ? status.f_frsize : status.f_bsize;
    if (fragment_size != 0
        && status.f_bavail
               > std::numeric_limits<std::uint64_t>::max() / fragment_size) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(status.f_bavail) * fragment_size;
}

void require_mutation_disk_capacity(
    const std::filesystem::path& storage_path,
    const std::uint64_t reserve_bytes,
    const std::uint64_t estimated_write_bytes,
    const std::string_view operation = "mutation before idempotency PREPARE") {
    if (storage_path.empty()) {
        return;
    }
    const auto required = checked_add_bytes(
        reserve_bytes,
        estimated_write_bytes,
        "mutation disk admission requirement");
    const auto available = filesystem_available_bytes(storage_path);
    if (available < required) {
        throw InsufficientStorageError(
            std::string(operation) + " rejected: collection "
            "filesystem has "
            + std::to_string(available) + " available bytes, but "
            + std::to_string(required) + " are required ("
            + std::to_string(estimated_write_bytes)
            + " estimated durable-write bytes plus "
            + std::to_string(reserve_bytes) + " reserve bytes)");
    }
}

[[nodiscard]] std::uint64_t estimated_lifecycle_write_bytes(
    const std::uint64_t bounded_input_bytes) {
    // Materialization and folding may temporarily retain both their selected
    // immutable inputs and newly published output. Twice the input plus fixed
    // catalog/filesystem headroom is intentionally conservative; the reserve
    // remains untouched if the action succeeds within this estimate.
    constexpr std::uint64_t publication_headroom = 64ULL << 20;
    if (bounded_input_bytes
        > (std::numeric_limits<std::uint64_t>::max()
           - publication_headroom) / 2) {
        throw InvalidArgumentError(
            "maintenance durable-write estimate overflows uint64");
    }
    return bounded_input_bytes * 2 + publication_headroom;
}

[[nodiscard]] std::uint64_t estimated_overlay_bytes(
    const std::uint64_t entries) noexcept {
    constexpr std::uint64_t bytes_per_entry = 64;
    return entries > std::numeric_limits<std::uint64_t>::max()
                         / bytes_per_entry
               ? std::numeric_limits<std::uint64_t>::max()
               : entries * bytes_per_entry;
}

void require_backlog_within_limits(
    const NativeMutationLogStats& stats,
    const EngineOptions& options,
    const std::string_view context) {
    if (stats.total_wal_bytes > options.max_wal_backlog_bytes
        || stats.total_wal_operation_count
               > options.max_wal_backlog_operations
        || stats.overlay_entry_count > options.max_overlay_entries
        || stats.estimated_overlay_bytes > options.max_overlay_bytes) {
        throw MutationCapacityError(
            std::string(context)
            + " exceeds configured retained-mutation limits: wal_bytes="
            + std::to_string(stats.total_wal_bytes) + "/"
            + std::to_string(options.max_wal_backlog_bytes)
            + ", wal_operations="
            + std::to_string(stats.total_wal_operation_count) + "/"
            + std::to_string(options.max_wal_backlog_operations)
            + ", overlay_entries="
            + std::to_string(stats.overlay_entry_count) + "/"
            + std::to_string(options.max_overlay_entries)
            + ", estimated_overlay_bytes="
            + std::to_string(stats.estimated_overlay_bytes) + "/"
            + std::to_string(options.max_overlay_bytes));
    }
}

void require_projected_mutation_capacity(
    const NativeMutationLogStats& stats,
    const EngineOptions& options,
    const std::uint64_t wal_bytes,
    const std::uint64_t wal_operations,
    const std::uint64_t maximum_new_overlay_entries,
    const bool may_rotate) {
    constexpr std::uint64_t wal_header_bytes = 4096;
    std::uint64_t projected_wal_bytes = checked_add_bytes(
        stats.total_wal_bytes, wal_bytes,
        "projected retained WAL bytes");
    if (may_rotate && wal_operations != 0) {
        projected_wal_bytes = checked_add_bytes(
            projected_wal_bytes, 2 * wal_header_bytes,
            "projected retained WAL bytes");
    }
    const std::uint64_t projected_wal_operations = checked_add_bytes(
        stats.total_wal_operation_count, wal_operations,
        "projected retained WAL operations");
    const std::uint64_t projected_overlay_entries = checked_add_bytes(
        stats.overlay_entry_count, maximum_new_overlay_entries,
        "projected mutation overlay entries");
    NativeMutationLogStats projected = stats;
    projected.total_wal_bytes = projected_wal_bytes;
    projected.total_wal_operation_count = projected_wal_operations;
    projected.overlay_entry_count = projected_overlay_entries;
    projected.estimated_overlay_bytes =
        estimated_overlay_bytes(projected_overlay_entries);
    require_backlog_within_limits(
        projected, options, "mutation admission");
}

void require_rotation_capacity(
    const NativeMutationLogStats& stats,
    const EngineOptions& options) {
    NativeMutationLogStats projected = stats;
    projected.total_wal_bytes = checked_add_bytes(
        stats.total_wal_bytes, 4096, "projected WAL rotation bytes");
    require_backlog_within_limits(
        projected, options, "WAL rotation");
}

void validate_engine_capacity_options(const EngineOptions& options) {
    if (options.max_wal_backlog_bytes == 0
        || options.max_wal_backlog_operations == 0
        || options.max_overlay_entries == 0
        || options.max_overlay_bytes == 0) {
        throw InvalidArgumentError(
            "engine retained-mutation limits must be positive");
    }
}

void validate_maintenance_options(const EngineMaintenanceOptions& options) {
    if (options.max_actions > 6) {
        throw InvalidArgumentError(
            "engine maintenance max_actions must not exceed 6");
    }
    if (options.idempotency_checkpoint_trigger_percent == 0
        || options.idempotency_checkpoint_trigger_percent >= 100
        || options.rotate_wal_bytes == 0
        || options.rotate_wal_transactions == 0
        || options.materialize_max_wals == 0
        || options.materialize_max_input_bytes == 0
        || options.materialize_max_input_entries == 0
        || options.materialize_max_memory_bytes == 0
        || options.fold_min_runs == 0
        || options.fold_max_runs == 0
        || options.fold_max_runs > 4096
        || options.fold_min_runs > options.fold_max_runs
        || options.fold_max_input_bytes == 0
        || options.fold_max_input_entries == 0) {
        throw InvalidArgumentError(
            "engine maintenance rotation, materialization, memory, and fold bounds "
            "must be positive and internally consistent");
    }
    if (options.clean_segments
        && (options.cleaner_max_candidate_segments == 0
            || options.cleaner_min_interval_seconds
                   > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int64_t>::max())
            || options.cleaner_max_sidecar_scan_bytes == 0
            || options.cleaner_max_source_bytes == 0
            || options.cleaner_max_output_bytes == 0
            || options.cleaner_max_memory_bytes == 0
            || options.cleaner_min_garbage_ratio_ppm > 1'000'000)) {
        throw InvalidArgumentError(
            "engine cleaner budgets must be positive and its ratio must not "
            "exceed 1000000 ppm");
    }
    if (options.reclaim_orphans
        && (options.orphan_max_files == 0
            || options.orphan_max_bytes == 0)) {
        throw InvalidArgumentError(
            "engine maintenance orphan budgets must be positive");
    }
}

[[nodiscard]] bool percentage_reached(
    const std::uint64_t value,
    const std::uint64_t maximum,
    const std::uint32_t percent) noexcept {
    if (maximum == 0) {
        return false;
    }
    const std::uint64_t quotient = maximum / 100;
    const std::uint64_t remainder = maximum % 100;
    const std::uint64_t threshold =
        quotient * percent + (remainder * percent + 99) / 100;
    return value >= threshold;
}

class MaintenanceActivityGuard final {
public:
    explicit MaintenanceActivityGuard(std::atomic<bool>& active)
        : active_(active) {
        bool expected = false;
        if (!active_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            throw PendingOperationError(
                "another engine maintenance pass is already active");
        }
    }
    ~MaintenanceActivityGuard() {
        active_.store(false, std::memory_order_release);
    }

    MaintenanceActivityGuard(const MaintenanceActivityGuard&) = delete;
    MaintenanceActivityGuard& operator=(const MaintenanceActivityGuard&) = delete;

private:
    std::atomic<bool>& active_;
};

}  // namespace

class SingleCollectionEngine::Impl {
public:
    Impl(
        NativeCollection& collection,
        const std::filesystem::path& storage_path,
        const EngineOptions options)
        : collection_(collection),
          storage_path_(storage_path),
          mutation_disk_reserve_bytes_(
              options.mutation_disk_reserve_bytes),
          automatic_rotate_wal_bytes_(
              options.automatic_rotate_wal_bytes),
          automatic_materialize_max_input_bytes_(
              options.automatic_materialize_max_input_bytes),
          automatic_materialize_max_input_entries_(
              options.automatic_materialize_max_input_entries),
          capacity_options_(options),
          reader_(options.reader),
          tuning_(options.tuning) {
        validate_engine_capacity_options(options);
        if (options.durable_idempotency && collection_.writable()
            && !storage_path.empty()) {
            const auto identity = collection_.identity();
            journal_ = detail::IdempotencyJournal::open(
                storage_path,
                identity.collection_id_low,
                identity.collection_id_high,
                detail::IdempotencyJournalLimits{
                    options.idempotency_journal_max_bytes,
                    options.idempotency_journal_max_operations});
            resolve_recovered_pending(identity.committed_sequence);
        }
        if (collection_.writable()
            && (automatic_materialize_max_input_bytes_ != 0
                || automatic_materialize_max_input_entries_ != 0)) {
            const auto stats = collection_.mutation_log_stats();
            if (stats.active_wal_transaction_count != 0
                && ((automatic_materialize_max_input_bytes_ != 0
                     && stats.active_wal_bytes
                            > automatic_materialize_max_input_bytes_)
                    || (automatic_materialize_max_input_entries_ != 0
                        && stats.active_wal_operation_count
                               > automatic_materialize_max_input_entries_))) {
                throw InvalidArgumentError(
                    "active WAL already exceeds the automatic "
                    "materialization byte/entry budget; restart with larger "
                    "materialization limits and materialize it");
            }
        }
        if (collection_.writable()) {
            require_backlog_within_limits(
                collection_.mutation_log_stats(), options,
                "server startup backlog");
        }
        operational_cache_ = fresh_operational_stats_unlocked();
    }

    [[nodiscard]] CollectionDescription describe() const {
        return CollectionDescription{
            collection_.format_version(),
            collection_.dimension(),
            collection_.dtype(),
            collection_.record_size(),
            collection_.count(),
            collection_.writable(),
            journal_ != nullptr,
            reader_,
            tuning_};
    }

    [[nodiscard]] GetResult get(
        const std::span<const std::uint64_t> ids,
        const std::uint32_t requested_prefix) const {
        const auto dimension = requested_prefix == 0
                                   ? collection_.dimension()
                                   : requested_prefix;
        if (dimension == 0 || dimension > collection_.dimension()) {
            throw InvalidArgumentError("prefix dimension is out of range");
        }
        auto full = collection_.get_bytes(ids);
        return project_get_result(ids.size(), dimension, std::move(full));
    }

    [[nodiscard]] ProfiledGetResult get_profiled(
        const std::span<const std::uint64_t> ids,
        const std::uint32_t requested_prefix) const {
        const auto dimension = requested_prefix == 0
                                   ? collection_.dimension()
                                   : requested_prefix;
        if (dimension == 0 || dimension > collection_.dimension()) {
            throw InvalidArgumentError("prefix dimension is out of range");
        }
        const std::uint64_t full_size =
            checked_payload_size(ids.size(), collection_.record_size());
        if (full_size > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError(
                "profiled get output does not fit address space");
        }
        std::vector<std::byte> full(
            static_cast<std::size_t>(full_size));
        NativeReadStats stats =
            collection_.get_into_profiled(ids, full);
        return ProfiledGetResult{
            .result = project_get_result(
                ids.size(), dimension, std::move(full)),
            .read = stats,
        };
    }

    [[nodiscard]] GetResult project_get_result(
        const std::size_t row_count,
        const std::uint32_t dimension,
        std::vector<std::byte> full) const {
        GetResult result;
        result.dimension = dimension;
        result.dtype = collection_.dtype();
        result.row_count = row_count;
        if (dimension == collection_.dimension()) {
            result.embeddings = std::move(full);
            return result;
        }

        const auto item_size = collection_.record_size()
                               / collection_.dimension();
        const auto prefix_bytes = static_cast<std::size_t>(
            static_cast<std::uint64_t>(dimension) * item_size);
        const auto full_bytes = static_cast<std::size_t>(collection_.record_size());
        if (row_count
            > std::numeric_limits<std::size_t>::max() / prefix_bytes) {
            throw InvalidArgumentError("prefix result does not fit memory");
        }
        result.embeddings.resize(row_count * prefix_bytes);
        for (std::size_t row = 0; row != row_count; ++row) {
            std::memcpy(
                result.embeddings.data() + row * prefix_bytes,
                full.data() + row * full_bytes,
                prefix_bytes);
        }
        return result;
    }

    [[nodiscard]] MutationResult mutate(
        const MutationKind kind,
        const OperationId operation_id,
        const OperationId replay_domain,
        const bool require_replay_domain,
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) {
        // The native lifecycle is two-phase: it reserves an immutable input
        // prefix under its mutation lane, performs the expensive build without
        // that lane, and publishes under the lane again.  Serializing the
        // complete lifecycle pass here would therefore turn a bounded native
        // publication barrier into a seconds- or minutes-long write blackout.
        // Keep only the journal PREPARE -> native commit -> COMPLETE sequence
        // serialized at the server layer.  Maintenance serializes its journal
        // checkpoint and WAL rotation with this same mutex, while long
        // materialization/fold/clean/orphan work runs outside it.
        std::lock_guard mutation(mutation_mutex_);
        if (require_replay_domain) {
            if (journal_ == nullptr) {
                if (!replay_domain.empty()) {
                    throw IdempotencyDomainError(
                        "non-durable server does not accept an idempotency "
                        "replay domain");
                }
            } else if (replay_domain.empty()
                       || replay_domain != journal_->domain()) {
                throw IdempotencyDomainError(
                    "idempotency replay domain expired or does not belong "
                    "to this server; refresh Describe before creating a new "
                    "operation, and never rebind an ambiguous old operation");
            }
        }
        resolve_live_pending();
        validate_request(kind, ids, embeddings);
        const auto request_digest = detail::mutation_digest(
            kind,
            collection_.dimension(),
            collection_.dtype(),
            ids,
            embeddings);

        if (journal_ != nullptr && !operation_id.empty()) {
            if (const auto* completed = journal_->completed(operation_id)) {
                if (completed->prepared.request_digest != request_digest) {
                    throw IdempotencyConflictError(
                        "operation_id was already committed with another payload");
                }
                return public_result(*completed, true);
            }
            if (const auto* aborted = journal_->aborted(operation_id);
                aborted != nullptr
                && aborted->request_digest != request_digest) {
                throw IdempotencyConflictError(
                    "operation_id was already used with another payload");
            }
        }

        const auto existed = collection_.contains(ids);
        for (std::size_t index = 0; index != ids.size(); ++index) {
            if (kind == MutationKind::Insert && existed[index] != 0) {
                throw ExistingIdError(ids[index]);
            }
            if (kind == MutationKind::Update && existed[index] == 0) {
                throw MissingIdError(ids[index]);
            }
        }

        std::uint64_t write_estimate = estimated_mutation_write_bytes(
            kind,
            ids.size(),
            kind == MutationKind::Delete ? 0 : embeddings.size(),
            journal_ != nullptr && !operation_id.empty());
        if (automatic_rotate_wal_bytes_ != 0
            || automatic_materialize_max_input_bytes_ != 0) {
            // A budget-boundary rotation followed by a threshold rotation may
            // publish two new WAL/catalog pairs around one commit.
            write_estimate = checked_add_bytes(
                write_estimate, 8ULL << 20,
                "mutation plus automatic rotation write estimate");
        }
        require_mutation_disk_capacity(
            storage_path_, mutation_disk_reserve_bytes_, write_estimate);

        const std::uint64_t wal_row_count =
            kind == MutationKind::Delete
                ? static_cast<std::uint64_t>(std::count(
                      existed.begin(), existed.end(), std::uint8_t{1}))
                : static_cast<std::uint64_t>(ids.size());
        const std::uint64_t wal_transaction_bytes =
            wal_row_count == 0
                ? 0
                : estimated_wal_transaction_bytes(
                      wal_row_count,
                      kind == MutationKind::Delete ? 0 : embeddings.size());
        require_projected_mutation_capacity(
            collection_.mutation_log_stats(),
            capacity_options_,
            wal_transaction_bytes,
            wal_row_count,
            // Exact unique growth is enforced once more by NativeCollection
            // under its state lock. This pessimistic bound keeps admission
            // independent of private overlay contents.
            wal_row_count,
            automatic_rotate_wal_bytes_ != 0
                || automatic_materialize_max_input_bytes_ != 0
                || automatic_materialize_max_input_entries_ != 0);

        if ((automatic_materialize_max_input_bytes_ != 0
             || automatic_materialize_max_input_entries_ != 0)
            && wal_row_count != 0) {
            constexpr std::uint64_t wal_header_bytes = 4096;
            const std::uint64_t new_wal_bytes = checked_add_bytes(
                wal_header_bytes, wal_transaction_bytes,
                "new WAL materialization bound");
            if ((automatic_materialize_max_input_bytes_ != 0
                 && new_wal_bytes
                        > automatic_materialize_max_input_bytes_)
                || (automatic_materialize_max_input_entries_ != 0
                    && wal_row_count
                           > automatic_materialize_max_input_entries_)) {
                throw MutationCapacityError(
                    "mutation WAL frame exceeds the automatic "
                    "materialization byte/entry budget; split the request or "
                    "raise the bounded materialization limits");
            }
            const auto stats = collection_.mutation_log_stats();
            if (stats.active_wal_transaction_count != 0) {
                if ((automatic_materialize_max_input_bytes_ != 0
                     && stats.active_wal_bytes
                            > automatic_materialize_max_input_bytes_)
                    || (automatic_materialize_max_input_entries_ != 0
                        && stats.active_wal_operation_count
                               > automatic_materialize_max_input_entries_)) {
                    throw InvalidArgumentError(
                        "active WAL exceeds the automatic materialization "
                        "byte/entry budget; restart with larger limits");
                }
                const bool next_frame_fits_bytes =
                    automatic_materialize_max_input_bytes_ == 0
                    || wal_transaction_bytes
                           <= automatic_materialize_max_input_bytes_
                                  - stats.active_wal_bytes;
                const bool next_frame_fits_entries =
                    automatic_materialize_max_input_entries_ == 0
                    || wal_row_count
                           <= automatic_materialize_max_input_entries_
                                  - stats.active_wal_operation_count;
                const bool next_frame_fits =
                    next_frame_fits_bytes && next_frame_fits_entries;
                if (!next_frame_fits) {
                    const auto rotated = collection_.rotate_wal();
                    if (!rotated.rotated) {
                        throw StorageError(
                            "automatic pre-commit WAL rotation made no "
                            "progress");
                    }
                }
            }
        }

        std::vector<std::uint64_t> committed_delete_ids;
        if (kind == MutationKind::Delete) {
            committed_delete_ids.reserve(ids.size());
            for (std::size_t index = 0; index != ids.size(); ++index) {
                if (existed[index] != 0) {
                    committed_delete_ids.push_back(ids[index]);
                }
            }
        }
        const auto wal_fingerprint = NativeCollection::fingerprint_mutation(
            kind == MutationKind::Delete
                ? NativeMutationOperation::Delete
                : NativeMutationOperation::Put,
            kind == MutationKind::Delete
                ? std::span<const std::uint64_t>(committed_delete_ids)
                : ids,
            kind == MutationKind::Delete
                ? std::span<const std::byte>{}
                : embeddings,
            collection_.record_size());

        const auto before = collection_.identity().committed_sequence;
        if (before == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("mutation commit sequence is exhausted");
        }
        detail::JournalPreparedOperation prepared;
        prepared.operation_id = operation_id;
        prepared.request_digest = request_digest;
        prepared.wal_fingerprint = wal_fingerprint;
        prepared.kind = kind;
        prepared.expected_sequence = before + 1;
        prepared.row_count = ids.size();
        if (kind == MutationKind::Delete) {
            prepared.deleted = existed;
        }
        const bool journalled = journal_ != nullptr && !operation_id.empty();
        if (journalled) {
            journal_->prepare(prepared);
        }

        std::vector<std::uint8_t> actual_deleted;
        switch (kind) {
            case MutationKind::Insert:
                collection_.insert(ids, embeddings);
                break;
            case MutationKind::Update:
                collection_.update(ids, embeddings);
                break;
            case MutationKind::Upsert:
                collection_.upsert(ids, embeddings);
                break;
            case MutationKind::Delete:
                actual_deleted = collection_.erase(ids);
                if (actual_deleted != existed) {
                    throw StorageError(
                        "delete preflight changed before the sole-writer commit");
                }
                break;
        }

        const auto after = collection_.identity().committed_sequence;
        const bool committed = kind == MutationKind::Delete
                                   ? !committed_delete_ids.empty()
                                   : !ids.empty();
        const auto expected_after = committed ? prepared.expected_sequence : before;
        if (after != expected_after) {
            throw StorageError(
                "mutation commit sequence changed outside the server lane");
        }

        MutationResult result;
        if (journalled) {
            result = public_result(
                journal_->complete(operation_id, after), false);
        } else {
            result = MutationResult{
                after,
                static_cast<std::uint64_t>(ids.size()),
                std::move(actual_deleted),
                false};
        }
        if (automatic_rotate_wal_bytes_ != 0
            || automatic_materialize_max_input_entries_ != 0) {
            const auto stats = collection_.mutation_log_stats();
            if (stats.active_wal_transaction_count != 0
                && ((automatic_rotate_wal_bytes_ != 0
                     && stats.active_wal_bytes >= automatic_rotate_wal_bytes_)
                    || (automatic_materialize_max_input_entries_ != 0
                        && stats.active_wal_operation_count
                               >= automatic_materialize_max_input_entries_))) {
                static_cast<void>(collection_.rotate_wal());
            }
        }
        refresh_operational_cache_unlocked_noexcept();
        return result;
    }

    [[nodiscard]] std::unique_ptr<NativeIdScan> open_scan(
        const std::uint64_t max_snapshot_bytes) const {
        return collection_.scan_ids(max_snapshot_bytes);
    }

    [[nodiscard]] std::unique_ptr<NativeReadView> open_read_view(
        const std::uint32_t prefix_dimension,
        const std::uint64_t max_snapshot_bytes) const {
        return collection_.open_read_view(
            prefix_dimension, max_snapshot_bytes);
    }

    [[nodiscard]] std::unique_ptr<NativeRecordScan> open_record_scan(
        const std::uint32_t prefix_dimension,
        const std::uint64_t max_snapshot_bytes) const {
        return collection_.scan_records(
            prefix_dimension, max_snapshot_bytes);
    }

    [[nodiscard]] bool durable_idempotency() const noexcept {
        return journal_ != nullptr;
    }

    [[nodiscard]] OperationId idempotency_domain() const {
        std::lock_guard mutation(mutation_mutex_);
        return journal_ == nullptr ? OperationId{} : journal_->domain();
    }

    [[nodiscard]] IdempotencyStats idempotency_stats() const {
        std::lock_guard mutation(mutation_mutex_);
        if (journal_ == nullptr) {
            return {};
        }
        const auto stats = journal_->stats();
        return IdempotencyStats{
            true,
            stats.pending,
            stats.bytes,
            stats.completed_operations,
            stats.aborted_operations,
            stats.max_bytes,
            stats.max_operations,
            stats.domain,
            stats.generation};
    }

    [[nodiscard]] EngineOperationalStats operational_stats() const {
        if (maintenance_active_.load(std::memory_order_acquire)) {
            // A native lifecycle reservation/publication may briefly own its
            // internal mutation lane even though the server journal lane is
            // intentionally free. Never turn diagnostics into a waiter on
            // that native barrier; mutations refresh this cache on commit.
            std::lock_guard cache(operational_cache_mutex_);
            EngineOperationalStats result = operational_cache_;
            result.maintenance_active = true;
            return result;
        }
        std::unique_lock mutation(mutation_mutex_, std::try_to_lock);
        if (!mutation.owns_lock()) {
            std::lock_guard cache(operational_cache_mutex_);
            EngineOperationalStats result = operational_cache_;
            result.maintenance_active =
                maintenance_active_.load(std::memory_order_acquire);
            return result;
        }
        if (maintenance_active_.load(std::memory_order_acquire)) {
            std::lock_guard cache(operational_cache_mutex_);
            EngineOperationalStats result = operational_cache_;
            result.maintenance_active = true;
            return result;
        }
        EngineOperationalStats result = fresh_operational_stats_unlocked();
        {
            // Lock order is always mutation_mutex_ -> operational_cache_mutex_.
            std::lock_guard cache(operational_cache_mutex_);
            operational_cache_ = result;
        }
        result.maintenance_active =
            maintenance_active_.load(std::memory_order_acquire);
        return result;
    }

    [[nodiscard]] EngineOperationalStats
    fresh_operational_stats_unlocked() const {
        EngineOperationalStats result;
        result.mutation = collection_.mutation_log_stats();
        if (journal_ != nullptr) {
            const auto stats = journal_->stats();
            result.idempotency = IdempotencyStats{
                true,
                stats.pending,
                stats.bytes,
                stats.completed_operations,
                stats.aborted_operations,
                stats.max_bytes,
                stats.max_operations,
                stats.domain,
                stats.generation};
        }
        result.filesystem_available_bytes = storage_path_.empty()
            ? 0
            : filesystem_available_bytes(storage_path_);
        result.mutation_disk_reserve_bytes = mutation_disk_reserve_bytes_;
        result.maintenance_active =
            maintenance_active_.load(std::memory_order_acquire);
        return result;
    }

    [[nodiscard]] EngineMaintenanceResult maintain_once(
        const EngineMaintenanceOptions& options) {
        validate_maintenance_options(options);
        if (!collection_.writable()) {
            throw InvalidArgumentError(
                "engine maintenance requires a writable collection");
        }
        MaintenanceActivityGuard activity(maintenance_active_);

        EngineMaintenanceResult result;
        bool lifecycle_disk_budget_blocked = false;
        const auto may_attempt = [&]() {
            if (result.actions_attempted >= options.max_actions) {
                result.action_budget_reached = true;
                return false;
            }
            ++result.actions_attempted;
            return true;
        };

        bool checkpoint_due = false;
        {
            // Journal reconciliation and stats are protected by the same lane
            // as PREPARE/COMPLETE.  No native lifecycle work is performed in
            // this critical section.
            std::lock_guard mutation(mutation_mutex_);
            resolve_live_pending();
            result.before = collection_.mutation_log_stats();
            if (journal_ != nullptr) {
                const auto stats = journal_->stats();
                result.idempotency_before = IdempotencyStats{
                    true, stats.pending, stats.bytes,
                    stats.completed_operations, stats.aborted_operations,
                    stats.max_bytes, stats.max_operations, stats.domain,
                    stats.generation};
            }
            if (journal_ != nullptr && options.checkpoint_idempotency) {
                const auto stats = journal_->stats();
                const std::uint64_t operations = checked_add_bytes(
                    stats.completed_operations, stats.aborted_operations,
                    "idempotency operation count");
                checkpoint_due = percentage_reached(
                        stats.bytes, stats.max_bytes,
                        options.idempotency_checkpoint_trigger_percent)
                    || percentage_reached(
                        operations, stats.max_operations,
                        options.idempotency_checkpoint_trigger_percent);
            }
        }
        NativeMutationLogStats current = result.before;

        if (checkpoint_due && may_attempt()) {
            // Checkpoint replaces the advertised replay domain and must never
            // cross a PREPARE/COMPLETE sequence.  It is bounded journal I/O,
            // so holding the server mutation lane here is intentional.
            std::lock_guard mutation(mutation_mutex_);
            resolve_live_pending();
            if (journal_ == nullptr) {
                throw StorageError(
                    "idempotency journal disappeared during maintenance");
            }
            const auto stats = journal_->stats();
            const std::uint64_t operations = checked_add_bytes(
                stats.completed_operations, stats.aborted_operations,
                "idempotency operation count");
            const bool still_due = percentage_reached(
                    stats.bytes, stats.max_bytes,
                    options.idempotency_checkpoint_trigger_percent)
                || percentage_reached(
                    operations, stats.max_operations,
                    options.idempotency_checkpoint_trigger_percent);
            if (still_due) {
                static_cast<void>(journal_->checkpoint());
                result.idempotency_checkpointed = true;
                ++result.actions_performed;
            }
        }

        const bool rotation_due =
            current.active_wal_transaction_count != 0
            && (current.active_wal_bytes >= options.rotate_wal_bytes
                || current.active_wal_transaction_count
                       >= options.rotate_wal_transactions);
        if (rotation_due && may_attempt()) {
            // Rotation can freeze the WAL needed to resolve an ambiguous
            // PREPARE.  Keep this short publication serialized with the full
            // journal transaction; long lifecycle consumers of already-frozen
            // inputs run below without the server mutex.
            std::lock_guard mutation(mutation_mutex_);
            resolve_live_pending();
            current = collection_.mutation_log_stats();
            const bool still_due =
                current.active_wal_transaction_count != 0
                && (current.active_wal_bytes >= options.rotate_wal_bytes
                    || current.active_wal_transaction_count
                           >= options.rotate_wal_transactions);
            if (still_due) {
                require_rotation_capacity(current, capacity_options_);
                current = collection_.rotate_wal();
                result.rotated = current.rotated;
                result.actions_performed += static_cast<std::uint64_t>(
                    result.rotated);
            }
        }

        if (current.frozen_wal_count != 0 && may_attempt()) {
            try {
                const std::uint64_t frozen_bytes =
                    current.total_wal_bytes >= current.active_wal_bytes
                        ? current.total_wal_bytes - current.active_wal_bytes
                        : current.total_wal_bytes;
                require_mutation_disk_capacity(
                    storage_path_,
                    mutation_disk_reserve_bytes_,
                    estimated_lifecycle_write_bytes(
                        std::min(
                            frozen_bytes,
                            options.materialize_max_input_bytes)),
                    "automatic WAL materialization");
                current = collection_.materialize_frozen_wals(
                    options.materialize_max_wals,
                    options.materialize_max_input_bytes,
                    options.materialize_max_input_entries,
                    options.materialize_max_memory_bytes);
                result.materialized = current.materialized;
                result.materialization_budget_blocked =
                    current.materialization_budget_blocked;
                result.materialization_entry_budget_blocked =
                    current.materialization_entry_budget_blocked;
                result.materialization_memory_budget_blocked =
                    current.materialization_memory_budget_blocked;
                result.actions_performed += static_cast<std::uint64_t>(
                    result.materialized);
            } catch (const PendingMutationFenceError&) {
                // PREPARE won the admission race after the short journal
                // snapshot above. Native lifecycle refused to cross it before
                // selecting destructive inputs, so this is a safe transient
                // deferral rather than a terminal storage fault.
                result.pending_mutation_deferred = true;
            } catch (const InsufficientStorageError&) {
                result.materialization_disk_budget_blocked = true;
                lifecycle_disk_budget_blocked = true;
            }
        }

        const bool fold_due =
            collection_.format_version() == 7
            && current.index_run_count >= options.fold_min_runs;
        if (!result.pending_mutation_deferred
            && !lifecycle_disk_budget_blocked && fold_due && may_attempt()) {
            try {
                require_mutation_disk_capacity(
                    storage_path_,
                    mutation_disk_reserve_bytes_,
                    estimated_lifecycle_write_bytes(
                        options.fold_max_input_bytes),
                    "automatic index folding");
                current = collection_.fold_partitions(
                    options.fold_max_runs,
                    options.fold_max_input_bytes,
                    options.fold_max_input_entries);
                result.folded = current.folded;
                result.fold_budget_blocked =
                    current.fold_budget_blocked;
                result.actions_performed += static_cast<std::uint64_t>(
                    result.folded);
            } catch (const PendingMutationFenceError&) {
                result.pending_mutation_deferred = true;
            } catch (const InsufficientStorageError&) {
                result.fold_disk_budget_blocked = true;
                lifecycle_disk_budget_blocked = true;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const bool cleaner_interval_due =
            last_cleaner_attempt_ == std::chrono::steady_clock::time_point{}
            || options.cleaner_min_interval_seconds == 0
            || now - last_cleaner_attempt_
                   >= std::chrono::seconds(
                       options.cleaner_min_interval_seconds);
        if (!result.pending_mutation_deferred
            && !lifecycle_disk_budget_blocked && options.clean_segments
            && collection_.format_version() == 7
            && cleaner_interval_due && may_attempt()) {
            result.cleaner_attempted = true;
            last_cleaner_attempt_ = now;
            NativeSegmentCleaningOptions cleaner;
            cleaner.dry_run = true;
            cleaner.max_candidate_segments =
                options.cleaner_max_candidate_segments;
            cleaner.max_sidecar_scan_bytes =
                options.cleaner_max_sidecar_scan_bytes;
            cleaner.max_source_bytes = options.cleaner_max_source_bytes;
            cleaner.max_output_bytes = options.cleaner_max_output_bytes;
            cleaner.max_memory_bytes = options.cleaner_max_memory_bytes;
            cleaner.min_garbage_rows = options.cleaner_min_garbage_rows;
            cleaner.min_reclaimable_bytes =
                options.cleaner_min_reclaimable_bytes;
            cleaner.min_garbage_ratio_ppm =
                options.cleaner_min_garbage_ratio_ppm;
            cleaner.start_after_segment_id = cleaner_cursor_;
            NativeSegmentCleaningStats plan =
                collection_.clean_segments(cleaner);
            result.cleaner = plan;
            if (!plan.candidate_found) {
                cleaner_cursor_ = plan.cycle_complete
                    ? 0
                    : plan.next_start_after_segment_id;
            } else {
                const std::uint64_t output_bytes = checked_add_bytes(
                    plan.output_segment_bytes, plan.output_run_bytes,
                    "automatic cleaner output bytes");
                try {
                    require_mutation_disk_capacity(
                        storage_path_, mutation_disk_reserve_bytes_,
                        estimated_lifecycle_write_bytes(output_bytes),
                        "automatic segment cleaning");
                    cleaner.dry_run = false;
                    result.cleaner = collection_.clean_segments(cleaner);
                    cleaner_cursor_ = result.cleaner.cycle_complete
                        ? 0
                        : result.cleaner.next_start_after_segment_id;
                    result.actions_performed +=
                        static_cast<std::uint64_t>(result.cleaner.cleaned);
                    if (result.cleaner.cleaned) {
                        current = collection_.mutation_log_stats();
                    }
                } catch (const PendingMutationFenceError&) {
                    result.pending_mutation_deferred = true;
                    last_cleaner_attempt_ = {};
                } catch (const InsufficientStorageError&) {
                    // Low free space is an observable policy outcome, not a
                    // terminal storage fault. Keep the cursor on this
                    // candidate so the next bounded pass retries it after
                    // operators free headroom or orphan GC makes progress.
                    result.cleaner_disk_budget_blocked = true;
                    lifecycle_disk_budget_blocked = true;
                    last_cleaner_attempt_ = {};
                }
            }
        }

        bool orphan_slot = false;
        if (!result.pending_mutation_deferred && options.reclaim_orphans) {
            if (result.actions_attempted < options.max_actions) {
                orphan_slot = may_attempt();
            } else if (lifecycle_disk_budget_blocked
                       && options.max_actions != 0) {
                // A failed disk preflight did not start a lifecycle write.
                // Reuse that bounded slot for the recovery action which may
                // free the space needed by the deferred output.
                orphan_slot = true;
                result.action_budget_reached = true;
            } else {
                result.action_budget_reached = true;
            }
        }
        if (orphan_slot) {
            result.orphan_reclaim_attempted = true;
            try {
                NativeOrphanCollectionOptions orphan_options;
                orphan_options.min_age_ns = options.orphan_min_age_ns;
                orphan_options.max_files = options.orphan_max_files;
                orphan_options.max_bytes = options.orphan_max_bytes;
                const auto reclaimed =
                    collection_.reclaim_orphans(orphan_options);
                result.orphan_durability_complete =
                    reclaimed.durability_complete;
                for (const auto& entry : reclaimed.entries) {
                    if (!entry.removed) {
                        if (entry.selected
                            && entry.disposition
                                   == NativeOrphanDisposition::UnlinkFailed) {
                            ++result.orphan_failed_files;
                        }
                        continue;
                    }
                    if (result.orphan_removed_files
                            == std::numeric_limits<std::uint64_t>::max()
                        || entry.file.bytes
                               > std::numeric_limits<std::uint64_t>::max()
                                     - result.orphan_removed_bytes) {
                        throw CorruptionError(
                            "engine maintenance orphan result overflows uint64");
                    }
                    ++result.orphan_removed_files;
                    result.orphan_removed_bytes += entry.file.bytes;
                }
                result.actions_performed += static_cast<std::uint64_t>(
                    result.orphan_removed_files != 0);
                if (!result.orphan_durability_complete
                    || result.orphan_failed_files != 0) {
                    throw StorageError(
                        "engine maintenance orphan reclamation was incomplete: "
                        + std::to_string(result.orphan_failed_files)
                        + " selected files failed and durability_complete="
                        + (result.orphan_durability_complete
                               ? "true" : "false"));
                }
            } catch (const PendingMutationFenceError&) {
                result.pending_mutation_deferred = true;
            }
        }

        {
            // Return one coherent post-pass journal/native snapshot and refresh
            // the nonblocking operational cache.  A mutation may have completed
            // during the long build and is deliberately included here.
            std::lock_guard mutation(mutation_mutex_);
            result.after = collection_.mutation_log_stats();
            if (journal_ != nullptr) {
                const auto stats = journal_->stats();
                result.idempotency_after = IdempotencyStats{
                    true, stats.pending, stats.bytes,
                    stats.completed_operations, stats.aborted_operations,
                    stats.max_bytes, stats.max_operations, stats.domain,
                    stats.generation};
            }
            refresh_operational_cache_unlocked_noexcept();
        }
        return result;
    }

    [[nodiscard]] NativeCollection& collection() noexcept {
        return collection_;
    }
    [[nodiscard]] const NativeCollection& collection() const noexcept {
        return collection_;
    }
    [[nodiscard]] const std::filesystem::path& storage_path() const noexcept {
        return storage_path_;
    }

private:
    void refresh_operational_cache_unlocked_noexcept() const noexcept {
        try {
            EngineOperationalStats fresh = fresh_operational_stats_unlocked();
            std::lock_guard cache(operational_cache_mutex_);
            operational_cache_ = std::move(fresh);
        } catch (...) {
            // Diagnostics must never change the outcome of a durable write.
        }
    }

    void validate_request(
        const MutationKind kind,
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) const {
        if (kind != MutationKind::Insert && kind != MutationKind::Update
            && kind != MutationKind::Upsert && kind != MutationKind::Delete) {
            throw InvalidArgumentError("unknown mutation kind");
        }
        validate_distinct_ids(ids);
        if (kind == MutationKind::Delete) {
            if (!embeddings.empty()) {
                throw InvalidArgumentError(
                    "delete request unexpectedly contains embeddings");
            }
            return;
        }
        const auto expected = checked_payload_size(
            ids.size(), collection_.record_size());
        if (expected > std::numeric_limits<std::size_t>::max()
            || embeddings.size() != static_cast<std::size_t>(expected)) {
            throw InvalidArgumentError(
                "mutation embedding bytes do not match collection schema");
        }
    }

    void resolve_recovered_pending(const std::uint64_t current_sequence) {
        if (journal_ == nullptr || journal_->pending() == nullptr) {
            return;
        }
        const auto pending = *journal_->pending();
        if (current_sequence < pending.expected_sequence) {
            journal_->abort(pending.operation_id);
            return;
        }
        const auto verification = collection_.verify_committed_mutation(
            pending.expected_sequence, pending.wal_fingerprint);
        if (verification == NativeMutationVerification::Match) {
            static_cast<void>(journal_->complete(
                pending.operation_id, pending.expected_sequence));
            return;
        }
        if (verification == NativeMutationVerification::Mismatch) {
            throw StorageError(
                "ambiguous idempotency PREPARE sequence belongs to another mutation; "
                "refusing to start");
        }
        throw StorageError(
            "ambiguous idempotency PREPARE cannot be verified because its WAL "
            "transaction is no longer retained; refusing to start");
    }

    void resolve_live_pending() {
        if (journal_ == nullptr) {
            return;
        }
        // COMPLETE/ABORT is durable before its fail-closed marker is removed.
        // A failed unlink must be retried even when this request is a replay
        // that would otherwise return before entering journal_->prepare().
        journal_->reconcile_fence();
        if (journal_->pending() == nullptr) {
            return;
        }
        const auto sequence = collection_.identity().committed_sequence;
        resolve_recovered_pending(sequence);
        if (journal_->pending() != nullptr) {
            throw MutationAbortedError(
                "an ambiguous operation must be resolved before another mutation");
        }
    }

    NativeCollection& collection_;
    std::unique_ptr<detail::IdempotencyJournal> journal_;
    std::filesystem::path storage_path_;
    std::uint64_t mutation_disk_reserve_bytes_ = 0;
    std::uint64_t automatic_rotate_wal_bytes_ = 0;
    std::uint64_t automatic_materialize_max_input_bytes_ = 0;
    std::uint64_t automatic_materialize_max_input_entries_ = 0;
    EngineOptions capacity_options_;
    ReaderOptions reader_;
    ReaderTuningDescription tuning_;
    mutable std::mutex mutation_mutex_;
    mutable std::mutex operational_cache_mutex_;
    mutable EngineOperationalStats operational_cache_;
    std::atomic<bool> maintenance_active_{false};
    std::uint32_t cleaner_cursor_ = 0;
    std::chrono::steady_clock::time_point last_cleaner_attempt_{};
};

SingleCollectionEngine::SingleCollectionEngine(
    NativeCollection& collection,
    std::filesystem::path storage_path,
    const EngineOptions options)
    : impl_(std::make_unique<Impl>(collection, storage_path, options)) {}
SingleCollectionEngine::~SingleCollectionEngine() = default;

CollectionDescription SingleCollectionEngine::describe() const {
    return impl_->describe();
}
GetResult SingleCollectionEngine::get(
    const std::span<const std::uint64_t> ids,
    const std::uint32_t prefix_dimension) const {
    return impl_->get(ids, prefix_dimension);
}
ProfiledGetResult SingleCollectionEngine::get_profiled(
    const std::span<const std::uint64_t> ids,
    const std::uint32_t prefix_dimension) const {
    return impl_->get_profiled(ids, prefix_dimension);
}
MutationResult SingleCollectionEngine::mutate(
    const MutationKind kind,
    const OperationId operation_id,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    return impl_->mutate(
        kind, operation_id, {}, false, ids, embeddings);
}
MutationResult SingleCollectionEngine::mutate_in_domain(
    const MutationKind kind,
    const OperationId operation_id,
    const OperationId replay_domain,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    return impl_->mutate(
        kind, operation_id, replay_domain, true, ids, embeddings);
}
std::unique_ptr<NativeIdScan> SingleCollectionEngine::open_scan(
    const std::uint64_t max_snapshot_bytes) const {
    return impl_->open_scan(max_snapshot_bytes);
}
std::unique_ptr<NativeReadView> SingleCollectionEngine::open_read_view(
    const std::uint32_t prefix_dimension,
    const std::uint64_t max_snapshot_bytes) const {
    return impl_->open_read_view(prefix_dimension, max_snapshot_bytes);
}
std::unique_ptr<NativeRecordScan> SingleCollectionEngine::open_record_scan(
    const std::uint32_t prefix_dimension,
    const std::uint64_t max_snapshot_bytes) const {
    return impl_->open_record_scan(prefix_dimension, max_snapshot_bytes);
}
NativeCollection& SingleCollectionEngine::collection() noexcept {
    return impl_->collection();
}
const NativeCollection& SingleCollectionEngine::collection() const noexcept {
    return impl_->collection();
}
const std::filesystem::path&
SingleCollectionEngine::storage_path() const noexcept {
    return impl_->storage_path();
}
bool SingleCollectionEngine::durable_idempotency() const noexcept {
    return impl_->durable_idempotency();
}
OperationId SingleCollectionEngine::idempotency_domain() const {
    return impl_->idempotency_domain();
}
IdempotencyStats SingleCollectionEngine::idempotency_stats() const {
    return impl_->idempotency_stats();
}
EngineOperationalStats SingleCollectionEngine::operational_stats() const {
    return impl_->operational_stats();
}
EngineMaintenanceResult SingleCollectionEngine::maintain_once(
    const EngineMaintenanceOptions options) {
    return impl_->maintain_once(options);
}

}  // namespace throwdb::server
