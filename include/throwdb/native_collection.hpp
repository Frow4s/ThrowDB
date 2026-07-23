#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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
    // Format 7 is the stable local-storage format. Format 6 remains readable
    // and writable for existing collections.
    std::uint32_t format_version = 7;
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

// Native-only safety bounds applied while opening and replaying mutable state.
// Format 6 also applies finite values to live overlay growth, committed WAL
// operations/bytes, and the extra header retained by rotation. Defaults
// preserve the historical unbounded embedded-open behaviour. Servers should
// provide finite values so a valid but unexpectedly large WAL backlog fails
// closed before rebuilding an unbounded in-memory overlay.
struct NativeOpenOptions {
    std::uint64_t max_wal_bytes =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_wal_operations =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_overlay_entries =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_overlay_bytes =
        std::numeric_limits<std::uint64_t>::max();
    // A durable server must open the native writer before it can reconcile its
    // PREPARE journal. Ordinary embedded writers leave this false and refuse to
    // cross an existing server/pending-mutation-v1 fence.
    bool allow_pending_mutation_recovery = false;
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

// A deterministic, bounded sample within one open collection snapshot and
// runtime. Format 7 proportionally inspects active snapshot slots, active run
// records, and WAL overlays, then keeps one seed-ordered bottom-K of currently
// live IDs. inspected_slots counts ID-bearing and raw snapshot slots and never
// exceeds max_inspected_slots; bounded run-bitmap metadata traversal is not an
// ID slot. complete is true only when the requested number of live IDs was
// collected.
struct NativeIdSample {
    std::vector<std::uint64_t> ids;
    std::uint64_t inspected_slots = 0;
    bool complete = false;
};

// Stable physical coordinates within one captured collection view. Segment
// IDs are monotonically allocated inside a collection and row_index is the
// zero-based record position in that segment. A location is a hint outside
// the NativeIdScan that produced it: lifecycle operations may publish a new
// view which relocates an otherwise unchanged logical ID.
struct NativeIdLocation {
    std::uint32_t segment_id = 0;
    std::uint32_t row_index = 0;

    bool operator==(const NativeIdLocation&) const = default;
};

static_assert(sizeof(NativeIdLocation) == sizeof(std::uint64_t));

// Identity and exact logical size of the immutable view captured by an ID
// scan. catalog_generation changes for lifecycle publications, while
// committed_sequence changes for WAL commits; together with collection_id
// they identify the view in which returned physical locations are valid.
struct NativeIdScanMetadata {
    std::uint64_t collection_id_low = 0;
    std::uint64_t collection_id_high = 0;
    std::uint64_t catalog_generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t live_count = 0;
    std::uint64_t snapshot_bytes = 0;
};

// Identity, exact logical size, and admitted heap estimate of an immutable
// format-7 read view. The estimate excludes already-open immutable index
// mappings and data files, which are shared with the collection and merely
// pinned for the lifetime of the view.
struct NativeReadViewMetadata {
    std::uint64_t collection_id_low = 0;
    std::uint64_t collection_id_high = 0;
    std::uint64_t catalog_generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t live_count = 0;
    std::uint64_t snapshot_bytes = 0;
};

// Lightweight identity of the currently published logical view. Unlike an ID
// scan this does not pin resources or copy the WAL overlay.
struct NativeCollectionIdentity {
    std::uint64_t collection_id_low = 0;
    std::uint64_t collection_id_high = 0;
    std::uint64_t catalog_generation = 0;
    std::uint64_t committed_sequence = 0;
};

enum class NativeMutationOperation : std::uint8_t {
    Put = 1,
    Delete = 2,
};

struct NativeMutationFingerprint {
    std::array<std::byte, 32> bytes{};
    bool operator==(const NativeMutationFingerprint&) const = default;
};

enum class NativeMutationVerification : std::uint8_t {
    Match = 1,
    Mismatch = 2,
    Unavailable = 3,
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
    // New diagnostics are appended to preserve the layout of the original
    // format-6 stats prefix for native callers.
    std::uint64_t total_delta_run_bytes = 0;
    std::uint64_t total_delta_segment_bytes = 0;
    std::uint64_t compacted_partition_count = 0;
    std::uint64_t compacted_run_section_count = 0;
    std::uint64_t compacted_snapshot_input_entries = 0;
    std::uint64_t compacted_snapshot_input_bytes = 0;
    std::uint64_t compacted_output_entries = 0;
    std::uint64_t compacted_output_bytes = 0;
    // Committed transaction records in the active WAL. Unlike put_rows, this
    // is non-zero for a delete-only transaction stream.
    std::uint64_t active_wal_transaction_count = 0;
    bool compaction_budget_blocked = false;
    // Format-6/7 retained-memory/backlog diagnostics. Appended for native ABI
    // compatibility with callers compiled against the original stats prefix.
    std::uint64_t overlay_entry_count = 0;
    std::uint64_t estimated_overlay_bytes = 0;
    std::uint64_t active_wal_operation_count = 0;
    std::uint64_t frozen_wal_operation_count = 0;
    std::uint64_t total_wal_operation_count = 0;
    std::uint64_t frozen_wal_bytes = 0;
    std::uint64_t materialized_wal_input_entries = 0;
    std::uint64_t materialization_estimated_peak_memory_bytes = 0;
    bool materialization_entry_budget_blocked = false;
    bool materialization_memory_budget_blocked = false;
};

// One bounded, deterministic selective-segment cleaning pass. The cleaner
// scans complete immutable sidecars only and retires at most one segment per
// call. start_after_segment_id is a performance cursor, not correctness
// state: callers may persist the returned next_start_after_segment_id, while
// losing it merely causes an earlier segment to be inspected again.
struct NativeSegmentCleaningOptions {
    bool dry_run = true;
    std::uint64_t max_candidate_segments = 8;
    std::uint64_t max_sidecar_scan_bytes = 1ULL << 30;
    std::uint64_t max_source_bytes = 32ULL << 30;
    std::uint64_t max_output_bytes = 32ULL << 30;
    std::uint64_t max_memory_bytes = 2ULL << 30;
    std::uint64_t min_garbage_rows = 1;
    std::uint64_t min_reclaimable_bytes = 512ULL << 20;
    // Integer parts per million avoids an architecture-dependent floating
    // point decision at the catalog publication boundary.
    std::uint32_t min_garbage_ratio_ppm = 300'000;
    std::uint32_t start_after_segment_id = 0;
};

struct NativeSegmentCleaningStats {
    std::uint64_t generation = 0;
    std::uint64_t scanned_segment_count = 0;
    std::uint64_t scanned_row_count = 0;
    std::uint64_t scanned_sidecar_bytes = 0;
    std::uint64_t skipped_budget_segment_count = 0;
    std::uint32_t next_start_after_segment_id = 0;
    std::uint32_t candidate_segment_id = 0;
    std::uint64_t source_rows = 0;
    std::uint64_t relocated_rows = 0;
    std::uint64_t garbage_rows = 0;
    std::uint64_t source_bytes = 0;
    std::uint64_t output_segment_count = 0;
    std::uint64_t output_segment_bytes = 0;
    std::uint64_t output_run_bytes = 0;
    std::uint64_t reclaimable_bytes = 0;
    bool dry_run = true;
    bool candidate_found = false;
    bool cleaned = false;
    bool budget_blocked = false;
    bool cycle_complete = false;
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
    // Format-7 values are append-only so existing numeric values remain
    // stable for native callers that persist diagnostics.
    CompleteUnreferencedSegmentPair,
    PartialUnreferencedSegmentPair,
    UnreferencedSnapshot,
    UnreferencedRun,
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
    // True when this storage format's destructive collector may consider the
    // class after its own fresh locked scan, age/budget checks, and no-follow
    // identity revalidation. This remains diagnostic, not unlink authority.
    // Kept append-only so the layout of the pre-existing fields stays stable.
    bool candidate = false;
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

// Immutable format-7 point-read snapshot. Input order and duplicates are
// preserved. Every ID in one get_into() batch is resolved before physical I/O
// starts, so a missing ID leaves the complete output batch untouched.
class NativeReadView {
public:
    ~NativeReadView();

    NativeReadView(const NativeReadView&) = delete;
    NativeReadView& operator=(const NativeReadView&) = delete;
    NativeReadView(NativeReadView&&) = delete;
    NativeReadView& operator=(NativeReadView&&) = delete;

    [[nodiscard]] const NativeReadViewMetadata& metadata() const noexcept;
    [[nodiscard]] std::uint32_t output_dimension() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    void get_into(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;

private:
    class Impl;
    explicit NativeReadView(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class NativeCollection;
};

// Snapshot cursor over authoritative records in physical RowRef order. A
// non-empty batch never spans physical segments. embeddings contains the
// configured full row or direct on-disk prefix for each returned ID. The IDs,
// embeddings, and optional locations output spans must be byte-disjoint.
class NativeRecordScan {
public:
    using StopPredicate = std::function<bool()>;

    ~NativeRecordScan();

    NativeRecordScan(const NativeRecordScan&) = delete;
    NativeRecordScan& operator=(const NativeRecordScan&) = delete;
    NativeRecordScan(NativeRecordScan&&) = delete;
    NativeRecordScan& operator=(NativeRecordScan&&) = delete;

    [[nodiscard]] const NativeReadViewMetadata& metadata() const noexcept;
    [[nodiscard]] std::uint32_t output_dimension() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] std::uint64_t read_into(
        std::span<std::uint64_t> ids,
        std::span<std::byte> embeddings,
        std::span<NativeIdLocation> locations = {});
    [[nodiscard]] std::uint64_t read_into(
        std::span<std::uint64_t> ids,
        std::span<std::byte> embeddings,
        std::span<NativeIdLocation> locations,
        const StopPredicate& stop_requested);
    [[nodiscard]] bool done() const noexcept;

private:
    class Impl;
    explicit NativeRecordScan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class NativeCollection;
};

// Snapshot cursor over authoritative live IDs in physical RowRef order:
// (segment_id, row_index). read_into() never mixes two physical segments in
// one non-empty result. Passing an empty locations span disables location
// output; otherwise it must have exactly the same capacity as ids and be
// byte-disjoint from the IDs output span.
class NativeIdScan {
public:
    using StopPredicate = std::function<bool()>;

    ~NativeIdScan();

    NativeIdScan(const NativeIdScan&) = delete;
    NativeIdScan& operator=(const NativeIdScan&) = delete;
    NativeIdScan(NativeIdScan&&) = delete;
    NativeIdScan& operator=(NativeIdScan&&) = delete;

    [[nodiscard]] const NativeIdScanMetadata& metadata() const noexcept;
    [[nodiscard]] std::uint64_t read_into(
        std::span<std::uint64_t> ids,
        std::span<NativeIdLocation> locations = {});
    // Cooperative server-side variant. A true predicate returns the rows
    // already produced by this call (possibly zero) without marking the
    // snapshot done. Candidate traversal is interruptible at a fixed bounded
    // cadence. The original overload deliberately remains non-cancellable.
    [[nodiscard]] std::uint64_t read_into(
        std::span<std::uint64_t> ids,
        std::span<NativeIdLocation> locations,
        const StopPredicate& stop_requested);
    [[nodiscard]] bool done() const noexcept;

private:
    class Impl;
    explicit NativeIdScan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class NativeCollection;
};

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
    // Canonical SHA-256 over the exact logical operations written to one WAL
    // transaction. Server-side crash recovery uses this as a witness without
    // changing the stable local mutation API or WAL layout.
    [[nodiscard]] static NativeMutationFingerprint fingerprint_mutation(
        NativeMutationOperation operation,
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings,
        std::uint64_t record_size);
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
        bool calibration = false,
        NativeOpenOptions open_options = {});

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
    // Format 7 only. The cursor pins immutable resources and copies the WAL
    // overlay while briefly holding the collection state lock; iteration
    // itself does not block mutation publication. max_snapshot_bytes is
    // checked under that same state lock before any overlay-sized allocation,
    // allowing servers to bind cursor memory to their admission lease.
    [[nodiscard]] std::unique_ptr<NativeIdScan> scan_ids(
        std::uint64_t max_snapshot_bytes =
            std::numeric_limits<std::uint64_t>::max()) const;
    // Format 7 only. A zero prefix_dimensions selects the stored dimension;
    // otherwise the leading prefix is passed to the physical reader as the
    // requested record width. Range coalescing, filesystem blocks, and page
    // cache may still amplify device traffic across unused suffix bytes.
    [[nodiscard]] std::unique_ptr<NativeReadView> open_read_view(
        std::uint32_t prefix_dimensions = 0,
        std::uint64_t max_snapshot_bytes =
            std::numeric_limits<std::uint64_t>::max()) const;
    [[nodiscard]] std::unique_ptr<NativeRecordScan> scan_records(
        std::uint32_t prefix_dimensions = 0,
        std::uint64_t max_snapshot_bytes =
            std::numeric_limits<std::uint64_t>::max()) const;
    [[nodiscard]] std::vector<std::byte> get_bytes(
        std::span<const std::uint64_t> ids) const;
    // Batch membership in one logical collection view. The output preserves
    // input order and duplicates. Unlike get(), missing IDs are ordinary false
    // values and no embedding payload is read from disk.
    [[nodiscard]] std::vector<std::uint8_t> contains(
        std::span<const std::uint64_t> ids) const;
    [[nodiscard]] NativeCollectionIdentity identity() const;
    // Match is returned only while the exact sequence remains in a retained
    // WAL and its CRC-validated operations hash to expected. Lifecycle may
    // make an older witness Unavailable; callers must fail closed.
    [[nodiscard]] NativeMutationVerification verify_committed_mutation(
        std::uint64_t sequence,
        const NativeMutationFingerprint& expected) const;

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
            std::numeric_limits<std::uint64_t>::max(),
        std::uint64_t max_input_entries =
            std::numeric_limits<std::uint64_t>::max(),
        std::uint64_t max_memory_bytes =
            std::numeric_limits<std::uint64_t>::max());
    // Format 6 merges the oldest contiguous DeltaV1 prefix. Format 7 folds
    // complete active run sections into selected authoritative partitions.
    // Embedding segments and newer WAL resources are never rewritten here.
    // The no-argument overload uses the format-aware complete bound: 8 runs
    // for format 6 and all 4096 possible active runs for format 7.
    [[nodiscard]] NativeMutationLogStats compact_delta_runs();
    [[nodiscard]] NativeMutationLogStats compact_delta_runs(
        std::uint64_t max_runs,
        std::uint64_t max_input_bytes = 8ULL << 30,
        std::uint64_t max_input_entries = 1'000'000);
    // Selectively retire one garbage-heavy immutable segment in format 7.
    // The source pathname is deliberately left for the orphan collector so
    // readers that pinned the old inode remain isolated from publication.
    [[nodiscard]] NativeSegmentCleaningStats clean_segments(
        NativeSegmentCleaningOptions options = {});
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
