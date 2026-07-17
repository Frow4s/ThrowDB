#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"

namespace throwdb::server {

struct OperationId {
    std::uint64_t high = 0;
    std::uint64_t low = 0;

    [[nodiscard]] bool empty() const noexcept { return high == 0 && low == 0; }
    bool operator==(const OperationId&) const = default;
};

enum class MutationKind : std::uint32_t {
    Insert = 1,
    Update = 2,
    Upsert = 3,
    Delete = 4,
};

struct ReaderTuningDescription {
    std::string mode = "conservative";
    std::string source = "fallback";
    std::string status = "native_defaults";
    std::string reuse_level = "fallback";
    bool explicit_overrides = false;
};

struct CollectionDescription {
    std::uint32_t format_version = 0;
    std::uint32_t dimension = 0;
    DType dtype = DType::Float16;
    std::uint64_t record_size = 0;
    std::uint64_t count = 0;
    bool writable = false;
    bool durable_idempotency = false;
    ReaderOptions reader;
    ReaderTuningDescription tuning;
};

struct GetResult {
    std::uint32_t dimension = 0;
    DType dtype = DType::Float16;
    std::uint64_t row_count = 0;
    std::vector<std::byte> embeddings;
};

struct MutationResult {
    std::uint64_t commit_sequence = 0;
    std::uint64_t row_count = 0;
    std::vector<std::uint8_t> deleted;
    bool replayed = false;
};

struct EngineOptions {
    bool durable_idempotency = true;
    ReaderOptions reader;
    ReaderTuningDescription tuning;
    // A mutation is admitted only when the collection filesystem can retain
    // this many bytes after the conservatively estimated WAL/journal append.
    // This is a safety reserve, not a quota: concurrent users of the same
    // filesystem can still consume space after the check.
    std::uint64_t mutation_disk_reserve_bytes = 1ULL << 30;
    // Hard fail-safe bounds. Existing operation IDs remain replayable after a
    // limit is reached, but a new operation ID is rejected until a bounded
    // maintenance checkpoint publishes a new advertised replay domain.
    std::uint64_t idempotency_journal_max_bytes = 4ULL << 30;
    std::uint64_t idempotency_journal_max_operations = 1'000'000;
    // Writable servers may rotate immediately after a commit crosses this
    // byte threshold. When automatic_materialize_max_input_bytes is nonzero,
    // the engine also rotates before PREPARE when the next frame would cross
    // that bound, and rejects one frame that cannot fit by itself. Zero leaves
    // rotation entirely to explicit/background maintenance.
    std::uint64_t automatic_rotate_wal_bytes = 0;
    std::uint64_t automatic_materialize_max_input_bytes = 0;
    std::uint64_t automatic_materialize_max_input_entries = 0;
    // Aggregate retained-mutation bounds. These are deliberately independent
    // from a single materialization pass: startup and every new mutation fail
    // closed before WAL replay/backlog growth can require unbounded RAM.
    std::uint64_t max_wal_backlog_bytes = 16ULL << 30;
    std::uint64_t max_wal_backlog_operations = 4'000'000;
    std::uint64_t max_overlay_entries = 4'000'000;
    std::uint64_t max_overlay_bytes = 256ULL << 20;
};

class IdempotencyConflictError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IdempotencyDomainError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PendingOperationError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InsufficientStorageError final : public StorageError {
public:
    using StorageError::StorageError;
};

class IdempotencyCapacityError final : public StorageError {
public:
    using StorageError::StorageError;
};

class MutationCapacityError final : public StorageError {
public:
    using StorageError::StorageError;
};

struct IdempotencyStats {
    bool enabled = false;
    bool pending = false;
    std::uint64_t journal_bytes = 0;
    std::uint64_t completed_operations = 0;
    std::uint64_t aborted_operations = 0;
    std::uint64_t maximum_journal_bytes = 0;
    std::uint64_t maximum_operations = 0;
    OperationId domain;
    std::uint64_t generation = 0;
};

struct EngineOperationalStats {
    NativeMutationLogStats mutation;
    IdempotencyStats idempotency;
    std::uint64_t filesystem_available_bytes = 0;
    std::uint64_t mutation_disk_reserve_bytes = 0;
    bool maintenance_active = false;
};

struct EngineMaintenanceOptions {
    // At most one journal checkpoint, rotate, materialize, format-7 fold,
    // selective clean, and orphan pass is considered in that order.
    std::uint64_t max_actions = 6;
    bool checkpoint_idempotency = true;
    std::uint32_t idempotency_checkpoint_trigger_percent = 75;
    std::uint64_t rotate_wal_bytes = 1ULL << 30;
    std::uint64_t rotate_wal_transactions = 100'000;
    std::uint64_t materialize_max_wals = 8;
    std::uint64_t materialize_max_input_bytes = 8ULL << 30;
    std::uint64_t materialize_max_input_entries = 2'000'000;
    std::uint64_t materialize_max_memory_bytes = 1ULL << 30;
    std::uint64_t fold_min_runs = 8;
    std::uint64_t fold_max_runs = 4096;
    std::uint64_t fold_max_input_bytes = 8ULL << 30;
    std::uint64_t fold_max_input_entries = 1'000'000;
    bool clean_segments = true;
    // Zero permits every maintenance pass; the server default scans at most
    // once per hour so sidecar inspection cannot become background I/O noise.
    std::uint64_t cleaner_min_interval_seconds = 3'600;
    std::uint64_t cleaner_max_candidate_segments = 8;
    std::uint64_t cleaner_max_sidecar_scan_bytes = 1ULL << 30;
    std::uint64_t cleaner_max_source_bytes = 32ULL << 30;
    std::uint64_t cleaner_max_output_bytes = 32ULL << 30;
    std::uint64_t cleaner_max_memory_bytes = 2ULL << 30;
    std::uint64_t cleaner_min_garbage_rows = 1;
    std::uint64_t cleaner_min_reclaimable_bytes = 512ULL << 20;
    std::uint32_t cleaner_min_garbage_ratio_ppm = 300'000;
    bool reclaim_orphans = true;
    std::uint64_t orphan_min_age_ns = 3'600ULL * 1'000'000'000;
    std::uint64_t orphan_max_files = 1024;
    std::uint64_t orphan_max_bytes = 8ULL << 30;
};

struct EngineMaintenanceResult {
    NativeMutationLogStats before;
    NativeMutationLogStats after;
    IdempotencyStats idempotency_before;
    IdempotencyStats idempotency_after;
    std::uint64_t actions_attempted = 0;
    std::uint64_t actions_performed = 0;
    bool rotated = false;
    bool materialized = false;
    bool materialization_budget_blocked = false;
    bool materialization_entry_budget_blocked = false;
    bool materialization_memory_budget_blocked = false;
    bool materialization_disk_budget_blocked = false;
    bool folded = false;
    bool compaction_budget_blocked = false;
    bool fold_disk_budget_blocked = false;
    bool idempotency_checkpointed = false;
    bool cleaner_attempted = false;
    bool cleaner_disk_budget_blocked = false;
    NativeSegmentCleaningStats cleaner;
    bool orphan_reclaim_attempted = false;
    std::uint64_t orphan_removed_files = 0;
    std::uint64_t orphan_removed_bytes = 0;
    bool orphan_durability_complete = true;
    std::uint64_t orphan_failed_files = 0;
    bool action_budget_reached = false;
};

// The sole transport-neutral owner of remote operation semantics. TDBP and
// gRPC adapters must share one instance so mutation ordering, preconditions,
// durable idempotency, and returned commit sequences cannot diverge.
class SingleCollectionEngine {
public:
    SingleCollectionEngine(
        NativeCollection& collection,
        std::filesystem::path storage_path = {},
        EngineOptions options = {});
    ~SingleCollectionEngine();

    SingleCollectionEngine(const SingleCollectionEngine&) = delete;
    SingleCollectionEngine& operator=(const SingleCollectionEngine&) = delete;

    [[nodiscard]] CollectionDescription describe() const;
    [[nodiscard]] GetResult get(
        std::span<const std::uint64_t> ids,
        std::uint32_t prefix_dimension = 0) const;
    [[nodiscard]] MutationResult mutate(
        MutationKind kind,
        OperationId operation_id,
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings = {});
    // Wire-facing mutation entry point. A durable request is admitted only
    // when replay_domain exactly matches the domain returned by Describe.
    [[nodiscard]] MutationResult mutate_in_domain(
        MutationKind kind,
        OperationId operation_id,
        OperationId replay_domain,
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings = {});
    [[nodiscard]] std::unique_ptr<NativeIdScan> open_scan(
        std::uint64_t max_snapshot_bytes =
            std::numeric_limits<std::uint64_t>::max()) const;

    [[nodiscard]] NativeCollection& collection() noexcept;
    [[nodiscard]] const NativeCollection& collection() const noexcept;
    [[nodiscard]] bool durable_idempotency() const noexcept;
    [[nodiscard]] OperationId idempotency_domain() const;
    [[nodiscard]] IdempotencyStats idempotency_stats() const;
    [[nodiscard]] EngineOperationalStats operational_stats() const;
    [[nodiscard]] EngineMaintenanceResult maintain_once(
        EngineMaintenanceOptions options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server
