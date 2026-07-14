#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "throwdb/storage.hpp"

namespace throwdb {

struct NativeBuilderOptions {
    std::uint32_t dimension = 1024;
    DType dtype = DType::Float16;
    // Target payload bytes per immutable base segment. Headers/ID sidecars are
    // additional. The builder also caps every segment at 2^32 rows.
    std::uint64_t segment_size_bytes = 16ULL << 30;
    // Format 6 remains the stable default. Format 7 is an explicit opt-in to
    // the partitioned RowRef layout while its mutation lifecycle is built out.
    std::uint32_t format_version = 6;
    // Format-7 builders use 4096 authoritative partitions by default. Kept at
    // the end to preserve aggregate initialization of the stable v6 fields.
    std::uint32_t partition_bits = 12;
};

struct NativeOpenStats {
    double total_seconds = 0.0;
    double manifest_seconds = 0.0;
    double index_seconds = 0.0;
    double index_header_seconds = 0.0;
    double index_allocation_seconds = 0.0;
    double index_controls_seconds = 0.0;
    double index_slots_seconds = 0.0;
    double index_checksum_seconds = 0.0;
    double index_structural_validation_seconds = 0.0;
    double base_segments_seconds = 0.0;
    double wal_replay_seconds = 0.0;
    std::uint64_t index_file_bytes = 0;
    bool index_body_checksum_present = false;
    bool index_body_checksum_verified = false;
    bool index_structural_validation_performed = false;
};

// Lightweight immutable layout metadata used to select reader policies before
// loading the potentially multi-gigabyte flat index. inspect_metadata() reads
// only the checksummed manifest and index header.
struct NativeCollectionMetadata {
    std::uint32_t format_version = 0;
    std::uint32_t dimension = 0;
    DType dtype = DType::Float16;
    std::uint64_t record_size = 0;
    std::uint64_t segment_size_bytes = 0;
    std::string layout;
    std::string feature_discriminator;
    std::uint64_t required_feature_bits = 0;
    std::uint64_t optional_feature_bits = 0;
};

// Process/container capability probe used before selecting a persisted reader
// backend. It creates the requested ring topology, completes harmless reads
// through every ring, and then destroys it without opening collection data or
// loading an index.
struct NativeReaderCapabilities {
    bool io_uring_available = false;
    std::uint32_t io_uring_probe_queue_depth = 0;
    std::uint32_t io_uring_probe_workers = 0;
    std::string io_uring_reason;
};

// Per-call diagnostics collected only by get_into_profiled(). The ordinary
// get/get_into path does not read clocks or update telemetry counters.
struct NativeReadStats {
    double total_seconds = 0.0;
    double lookup_seconds = 0.0;
    double physical_plan_seconds = 0.0;
    // Queue/admission wait, physical I/O, and scatter copies currently share
    // one wall-clock phase because those operations overlap in parallel
    // backends. More detailed backend hooks can extend this later.
    double physical_backend_seconds = 0.0;
    std::uint64_t requested_rows = 0;
    std::uint64_t base_rows = 0;
    std::uint64_t mutation_rows = 0;
    std::uint64_t planned_ranges = 0;
    std::uint64_t planned_read_bytes = 0;
    std::uint64_t backend_tasks = 0;
    bool zero_delta_fast_path = false;
};

// A deterministic, bounded sample from the immutable base index. Sampling
// never scans more than inspected_slots and complete is true only when the
// requested number of currently-live IDs was collected.
struct NativeIdSample {
    std::vector<std::uint64_t> ids;
    std::uint64_t inspected_slots = 0;
    bool complete = false;
};

// Durable mutation-log state captured under the collection's mutation lock.
// rotate_wal() returns the same snapshot with rotated=true when it published a
// successor WAL. An empty active WAL is a successful no-op.
struct NativeMutationLogStats {
    bool catalog_managed = false;
    bool delta_managed = false;
    std::uint64_t generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t materialized_sequence = 0;
    std::uint64_t frozen_wal_count = 0;
    std::uint64_t active_wal_bytes = 0;
    std::uint64_t active_wal_put_rows = 0;
    std::uint64_t total_wal_bytes = 0;
    std::uint64_t delta_run_count = 0;
    std::uint64_t delta_entry_count = 0;
    std::uint64_t delta_put_rows = 0;
    std::uint64_t total_delta_bytes = 0;
    std::uint64_t reclaimed_wal_count = 0;
    std::uint64_t reclaimed_wal_bytes = 0;
    std::uint64_t materialized_wal_count = 0;
    std::uint64_t materialized_wal_input_bytes = 0;
    std::uint64_t compacted_delta_run_count = 0;
    std::uint64_t compacted_delta_input_entries = 0;
    std::uint64_t compacted_delta_input_bytes = 0;
    std::uint64_t reclaimed_delta_run_count = 0;
    std::uint64_t reclaimed_delta_bytes = 0;
    bool rotated = false;
    bool materialized = false;
    bool materialization_budget_blocked = false;
    bool compacted = false;
    bool reclamation_complete = true;
};

// Read-only classification of files in the mutable-resource namespaces. The
// class is diagnostic only: no value authorizes unlinking a path without the
// executor's fresh locked scan and no-follow identity revalidation.
enum class NativeOrphanClass : std::uint8_t {
    Reachable,
    SupersededCatalog,
    TempPublication,
    UnreferencedWal,
    CompleteUnreferencedDeltaPair,
    PartialUnreferencedDeltaPair,
    UnknownDoNotTouch,
};

struct NativeOrphanEntry {
    // Portable relative path below the collection root.
    std::string path;
    NativeOrphanClass classification{NativeOrphanClass::UnknownDoNotTouch};
    // Exact size for a regular file. Non-regular entries are never followed
    // and report zero bytes.
    std::uint64_t bytes = 0;
    // Modification time reported by lstat, normalized to Unix nanoseconds.
    // Negative pre-epoch timestamps are retained rather than silently
    // clamped so policy code can conservatively reject them.
    std::int64_t modified_unix_ns = 0;
};

struct NativeOrphanInventory {
    std::uint64_t current_generation = 0;
    // The authoritative catalog UUID as 32 lowercase hexadecimal digits,
    // high word first, matching the on-disk catalog filename token.
    std::string current_catalog_id;
    // Deterministically sorted by path.
    std::vector<NativeOrphanEntry> entries;
};

enum class NativeOrphanDisposition : std::uint8_t {
    NotCandidate,
    InvalidModificationTime,
    MinimumAgeNotReached,
    MaximumFilesReached,
    MaximumBytesReached,
    Removed,
    NoLongerPresent,
    IdentityChanged,
    UnlinkFailed,
};

struct NativeOrphanCollectionOptions {
    std::uint64_t snapshot_unix_ns = 0;
    std::uint64_t min_age_ns = 0;
    std::uint64_t max_files = 1024;
    std::uint64_t max_bytes = 8ULL << 30;
};

struct NativeOrphanCollectionEntry {
    NativeOrphanEntry file;
    NativeOrphanDisposition disposition{
        NativeOrphanDisposition::NotCandidate};
    bool eligible = false;
    bool selected = false;
    bool removed = false;
    bool durable = false;
    std::string error;
};

struct NativeOrphanCollectionResult {
    std::uint64_t current_generation = 0;
    std::string current_catalog_id;
    std::uint64_t snapshot_unix_ns = 0;
    std::uint64_t min_age_ns = 0;
    std::uint64_t max_files = 0;
    std::uint64_t max_bytes = 0;
    std::vector<std::string> synced_directories;
    std::vector<NativeOrphanCollectionEntry> entries;
    bool durability_complete = true;
};

class NativeCollectionBuilder {
public:
    static std::unique_ptr<NativeCollectionBuilder> create(
        const std::filesystem::path& path,
        NativeBuilderOptions options = {});

    ~NativeCollectionBuilder();

    NativeCollectionBuilder(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder& operator=(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder(NativeCollectionBuilder&&) = delete;
    NativeCollectionBuilder& operator=(NativeCollectionBuilder&&) = delete;

    void append(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void finish();

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool finished() const;

private:
    class Impl;
    explicit NativeCollectionBuilder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class NativeCollection;

// One persistent physical-reader candidate used to compare reader policies
// against an already-open collection snapshot. It owns only executor/backend
// state; the collection continues to own the loaded index and file handles.
class NativeReaderTrial {
public:
    ~NativeReaderTrial();

    NativeReaderTrial(const NativeReaderTrial&) = delete;
    NativeReaderTrial& operator=(const NativeReaderTrial&) = delete;
    NativeReaderTrial(NativeReaderTrial&&) = delete;
    NativeReaderTrial& operator=(NativeReaderTrial&&) = delete;

private:
    class Impl;
    explicit NativeReaderTrial(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class NativeCollection;
};

class NativeCollection {
public:
    [[nodiscard]] static NativeReaderCapabilities reader_capabilities(
        std::uint32_t io_uring_queue_depth = 32,
        std::uint32_t io_uring_workers = 1);
    [[nodiscard]] static NativeCollectionMetadata inspect_metadata(
        const std::filesystem::path& path);
    // Lightweight maintenance inventory: acquires the writer lease and
    // mutation publication barrier without loading the base index, opening
    // base segments, or replaying mutation data.
    [[nodiscard]] static NativeOrphanInventory inspect_orphans(
        const std::filesystem::path& path);
    // Lightweight destructive counterpart to inspect_orphans(). It always
    // performs a fresh locked scan and never accepts a previous plan as an
    // authorization token.
    [[nodiscard]] static NativeOrphanCollectionResult reclaim_orphans(
        const std::filesystem::path& path,
        NativeOrphanCollectionOptions options = {});

    static std::unique_ptr<NativeCollection> open(
        const std::filesystem::path& path,
        ReaderOptions reader_options = {},
        bool writable = false,
        bool calibration = false);

    ~NativeCollection();

    NativeCollection(const NativeCollection&) = delete;
    NativeCollection& operator=(const NativeCollection&) = delete;
    NativeCollection(NativeCollection&&) = delete;
    NativeCollection& operator=(NativeCollection&&) = delete;

    void get_into(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeReadStats get_into_profiled(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] std::unique_ptr<NativeReaderTrial> create_reader_trial(
        ReaderOptions reader_options = {}) const;
    void get_into_with_reader(
        const NativeReaderTrial& reader,
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeReadStats get_into_with_reader_profiled(
        const NativeReaderTrial& reader,
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeIdSample sample_ids(
        std::uint64_t requested,
        std::uint64_t seed,
        std::uint64_t max_inspected_slots) const;
    [[nodiscard]] std::vector<std::byte> get_bytes(
        std::span<const std::uint64_t> ids) const;

    void insert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void update(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void upsert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    [[nodiscard]] std::vector<std::uint8_t> erase(
        std::span<const std::uint64_t> ids);

    [[nodiscard]] NativeMutationLogStats mutation_log_stats() const;
    [[nodiscard]] NativeMutationLogStats rotate_wal();
    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals(
        std::uint64_t max_wals =
            std::numeric_limits<std::uint64_t>::max(),
        std::uint64_t max_input_bytes =
            std::numeric_limits<std::uint64_t>::max());
    // Merge the oldest contiguous DeltaV1 prefix when at least two complete
    // runs fit every budget. Original sequences and tombstones are retained;
    // base segments and newer delta/WAL resources are never rewritten.
    [[nodiscard]] NativeMutationLogStats compact_delta_runs(
        std::uint64_t max_runs = 8,
        std::uint64_t max_input_bytes = 8ULL << 30,
        std::uint64_t max_input_entries = 1'000'000);
    // Build the same read-only inventory through an already-open writable
    // handle. A writable handle is required because its lifetime writer lease
    // excludes unpublished materialization/compaction output.
    [[nodiscard]] NativeOrphanInventory collect_orphans();
    [[nodiscard]] NativeOrphanCollectionResult reclaim_orphans(
        NativeOrphanCollectionOptions options = {});

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool writable() const;
    [[nodiscard]] NativeOpenStats open_stats() const;

private:
    class Impl;
    explicit NativeCollection(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb
