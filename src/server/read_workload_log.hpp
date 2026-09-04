#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace throwdb::server::detail {

// Read logging is opt-in at the server boundary. The logger owns only bounded
// preallocated memory and never makes a data-plane request wait for telemetry.
struct ReadWorkloadLogOptions {
    std::filesystem::path directory;
    std::uint32_t id_chunk_rows = 8'192;
    std::uint32_t max_active_requests = 64;
    std::uint64_t buffer_bytes = 256ULL << 20;
    std::uint64_t segment_bytes = 256ULL << 20;
    std::uint64_t rotate_seconds = 3'600;
    std::uint64_t retention_bytes = 50ULL << 30;
    std::uint64_t retention_seconds = 7ULL * 24 * 60 * 60;
    std::uint64_t min_free_bytes = 10ULL << 30;
    std::uint64_t max_write_bytes_per_second = 16ULL << 20;
    std::uint64_t max_retained_files = 4'096;
};

struct ReadWorkloadCollectionInfo {
    std::uint64_t collection_id_low = 0;
    std::uint64_t collection_id_high = 0;
    std::uint32_t format = 0;
    std::uint32_t dtype = 0;
    std::uint32_t dimension = 0;
    std::uint32_t record_size_bytes = 0;
    std::uint32_t reader_backend = 0;
    std::uint32_t io_workers = 0;
    std::uint32_t io_queue_capacity = 0;
    std::uint32_t io_ranges_per_task = 0;
    std::uint64_t max_read_range_bytes = 0;
    std::uint64_t max_gap_bytes = 0;
    std::uint32_t io_uring_queue_depth = 0;
    std::uint32_t io_uring_workers = 0;
    std::uint32_t metric_page_bytes = 4'096;
    std::uint32_t metric_chunk_bytes = 1U << 20;
    std::uint32_t tuning_explicit_overrides = 0;
};

enum class ReadWorkloadMethod : std::uint32_t {
    Get = 1,
    StreamGet = 2,
};

enum class ReadWorkloadBackend : std::uint32_t {
    Unknown = 0,
    ThreadPool = 1,
    IoUring = 2,
};

enum class ReadWorkloadRequestStatus : std::uint32_t {
    Success = 1,
    Cancelled = 2,
    DeadlineExceeded = 3,
    Failed = 4,
    ClientClosed = 5,
    Shutdown = 6,
};

struct ReadWorkloadSnapshot {
    std::uint64_t catalog_generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t row_count = 0;
    bool exact = false;
};

struct ReadWorkloadRequestInfo {
    ReadWorkloadMethod method = ReadWorkloadMethod::Get;
    ReadWorkloadSnapshot snapshot;
    std::uint32_t prefix_dimensions = 0;
    // Zero means that the total is not known at begin time (normal for a
    // bidirectional StreamGet). A nonzero declaration is verified at end.
    std::uint64_t declared_ids = 0;
    // A caller may provide its transport request/stream token. When both are
    // zero the logger assigns a process-unique ID.
    std::uint64_t request_id_low = 0;
    std::uint64_t request_id_high = 0;
};

struct ReadWorkloadIdChunkInfo {
    std::uint64_t logical_offset = 0;
    std::uint64_t batch_sequence = 0;
};

struct ReadWorkloadBatchMetrics {
    ReadWorkloadBackend backend = ReadWorkloadBackend::Unknown;
    std::uint64_t batch_sequence = 0;
    // Global successful StreamGet delivery ordinal. Zero means that the
    // transport cannot observe delivery (currently unary Get).
    std::uint64_t delivery_sequence = 0;
    std::uint64_t logical_offset = 0;
    std::uint64_t rows = 0;
    std::uint64_t response_bytes = 0;

    std::uint64_t total_ns = 0;
    std::uint64_t lookup_ns = 0;
    std::uint64_t sort_ns = 0;
    std::uint64_t plan_ns = 0;
    std::uint64_t scheduler_wait_ns = 0;
    std::uint64_t io_ns = 0;
    std::uint64_t scatter_ns = 0;
    std::uint64_t encode_ns = 0;
    std::uint64_t write_ns = 0;
    std::uint64_t logger_queue_ns = 0;

    std::uint64_t unique_storage_segments = 0;
    std::uint64_t physical_requests = 0;
    std::uint64_t merged_ranges = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t useful_read_bytes = 0;
    std::uint64_t planned_bytes = 0;
    std::uint64_t completed_bytes = 0;
    std::uint64_t gap_bytes = 0;
    std::uint64_t overread_bytes = 0;
    std::uint64_t pages_touched = 0;
    std::uint64_t page_runs = 0;
    std::uint64_t layout_chunks_touched = 0;
    std::uint64_t layout_chunk_runs = 0;
    std::uint64_t discontinuities = 0;
    std::uint64_t inversions = 0;
    std::uint64_t max_range_bytes = 0;
    std::uint64_t mean_range_bytes = 0;
    std::uint64_t backend_tasks = 0;
};

struct ReadWorkloadRequestEnd {
    ReadWorkloadRequestStatus status = ReadWorkloadRequestStatus::Failed;
    std::uint32_t error_code = 0;
    std::uint64_t delivered_ids = 0;
    std::uint64_t response_bytes = 0;
};

struct ReadWorkloadLogStats {
    std::uint64_t observed_requests = 0;
    std::uint64_t begun_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t complete_requests = 0;
    std::uint64_t incomplete_requests = 0;

    std::uint64_t observed_records = 0;
    std::uint64_t enqueued_records = 0;
    std::uint64_t written_records = 0;
    std::uint64_t observed_id_chunks = 0;
    std::uint64_t enqueued_id_chunks = 0;
    std::uint64_t written_id_chunks = 0;
    std::uint64_t observed_batches = 0;
    std::uint64_t enqueued_batches = 0;
    std::uint64_t written_batches = 0;
    std::uint64_t observed_ids = 0;
    std::uint64_t enqueued_ids = 0;
    std::uint64_t written_ids = 0;

    std::uint64_t dropped_full_records = 0;
    std::uint64_t dropped_full_ids = 0;
    std::uint64_t dropped_runtime_failure_records = 0;
    std::uint64_t dropped_runtime_failure_ids = 0;
    std::uint64_t dropped_min_free_records = 0;
    std::uint64_t dropped_min_free_ids = 0;
    std::uint64_t dropped_io_budget_records = 0;
    std::uint64_t dropped_io_budget_ids = 0;

    std::uint64_t write_errors = 0;
    std::uint64_t bytes_written = 0;
    std::uint64_t segments_created = 0;
    std::uint64_t rotated_segments = 0;
    std::uint64_t retained_bytes = 0;
    std::uint64_t retained_segments = 0;
    std::uint64_t retention_deleted_bytes = 0;
    std::uint64_t retention_deleted_segments = 0;
    std::uint64_t queued_records = 0;
    std::uint64_t queued_bytes = 0;
    std::uint64_t peak_queued_bytes = 0;
    std::uint64_t buffer_capacity_bytes = 0;
    std::uint64_t active_requests = 0;
    int last_error_number = 0;
    bool paused_for_min_free = false;
    bool runtime_failed = false;
    bool shutting_down = false;
};

inline constexpr std::size_t kReadWorkloadLogHeaderBytes = 4'096;
inline constexpr std::size_t kReadWorkloadRecordHeaderBytes = 256;
inline constexpr std::size_t kReadWorkloadBatchRecordBytes = 384;
inline constexpr std::size_t kReadWorkloadRecordCrcOffset = 24;
inline constexpr std::uint32_t kReadWorkloadMaximumIdChunkRows = 8'192;
inline constexpr std::uint32_t kReadWorkloadMaximumActiveRequests = 65'536;

class ReadWorkloadLogger;
class ReadWorkloadLogImpl;

// Move-only request-local bookkeeping. The handle must not outlive its logger
// and must be used by one request thread at a time. Its destructor emits a
// best-effort Shutdown terminal record when the caller missed end_request().
class ReadWorkloadRequestHandle final {
public:
    ReadWorkloadRequestHandle() noexcept = default;
    ~ReadWorkloadRequestHandle();
    ReadWorkloadRequestHandle(ReadWorkloadRequestHandle&&) noexcept;
    ReadWorkloadRequestHandle& operator=(
        ReadWorkloadRequestHandle&&) noexcept;
    ReadWorkloadRequestHandle(const ReadWorkloadRequestHandle&) = delete;
    ReadWorkloadRequestHandle& operator=(
        const ReadWorkloadRequestHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept {
        return logger_ != nullptr;
    }
    [[nodiscard]] std::uint64_t request_id_low() const noexcept {
        return request_id_low_;
    }
    [[nodiscard]] std::uint64_t request_id_high() const noexcept {
        return request_id_high_;
    }

private:
    friend class ReadWorkloadLogger;
    friend class ReadWorkloadLogImpl;
    ReadWorkloadLogger* logger_ = nullptr;
    std::uint64_t request_id_low_ = 0;
    std::uint64_t request_id_high_ = 0;
    std::uint64_t started_steady_ns_ = 0;
    std::uint64_t declared_ids_ = 0;
    std::uint64_t submitted_ids_ = 0;
    std::uint64_t enqueued_ids_ = 0;
    std::uint64_t id_chunks_ = 0;
    std::uint64_t batches_ = 0;
    std::uint64_t response_bytes_ = 0;
    std::uint64_t dropped_records_ = 0;
    std::uint64_t dropped_ids_ = 0;
    std::uint64_t dropped_full_records_ = 0;
    std::uint64_t dropped_full_ids_ = 0;
    std::uint64_t dropped_runtime_records_ = 0;
    std::uint64_t dropped_runtime_ids_ = 0;
    std::uint64_t dropped_min_free_records_ = 0;
    std::uint64_t dropped_min_free_ids_ = 0;
    std::uint64_t dropped_io_budget_records_ = 0;
    std::uint64_t dropped_io_budget_ids_ = 0;
    std::uint64_t digest_low_ = 0xcbf29ce484222325ULL;
    std::uint64_t digest_high_ = 0x84222325cbf29ce4ULL;
    std::uint64_t next_chunk_sequence_ = 0;
    bool logical_offsets_contiguous_ = true;
};

class ReadWorkloadLogger final {
public:
    static std::unique_ptr<ReadWorkloadLogger> open(
        ReadWorkloadLogOptions options,
        ReadWorkloadCollectionInfo collection);

    ~ReadWorkloadLogger();
    ReadWorkloadLogger(const ReadWorkloadLogger&) = delete;
    ReadWorkloadLogger& operator=(const ReadWorkloadLogger&) = delete;

    [[nodiscard]] ReadWorkloadRequestHandle begin_request(
        const ReadWorkloadRequestInfo& info) noexcept;

    // Splits an arbitrary input span into exact ordered records no larger than
    // id_chunk_rows. false means at least one complete chunk was dropped; no
    // prefix within a chunk is ever sampled or truncated.
    [[nodiscard]] bool try_append_ids(
        ReadWorkloadRequestHandle& request,
        std::span<const std::uint64_t> ids,
        const ReadWorkloadIdChunkInfo& info) noexcept;

    [[nodiscard]] bool try_append_batch(
        ReadWorkloadRequestHandle& request,
        const ReadWorkloadBatchMetrics& metrics) noexcept;

    // Always releases the bounded active-request admission. Terminal logging
    // remains fail-open and may itself be dropped.
    void end_request(
        ReadWorkloadRequestHandle& request,
        const ReadWorkloadRequestEnd& end) noexcept;

    [[nodiscard]] ReadWorkloadLogStats stats() const noexcept;
    void shutdown() noexcept;

private:
    friend class ReadWorkloadRequestHandle;
    explicit ReadWorkloadLogger(std::unique_ptr<ReadWorkloadLogImpl> impl);
    void abandon_request(ReadWorkloadRequestHandle& request) noexcept;
    std::unique_ptr<ReadWorkloadLogImpl> impl_;
};

}  // namespace throwdb::server::detail
