#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace throwdb::server::detail {

// Read logging is opt-in at the server boundary. An empty/disabled option is
// deliberately not represented here: callers must not construct a logger
// unless the operator supplied a directory.
struct ReadWorkloadLogOptions {
    std::filesystem::path directory;
    std::uint32_t sample_rate_ppm = 10'000;
    std::uint32_t max_ids_per_event = 4'096;
    std::uint64_t buffer_bytes = 64ULL << 20;
    std::uint64_t segment_bytes = 256ULL << 20;
    std::uint64_t rotate_seconds = 3'600;
    std::uint64_t retention_bytes = 10ULL << 30;
    std::uint64_t retention_seconds = 7ULL * 24 * 60 * 60;
    std::uint64_t min_free_bytes = 2ULL << 30;
    std::uint64_t max_write_bytes_per_second = 16ULL << 20;
    std::uint64_t max_retained_files = 4'096;
};

// Stable schema identity copied into every segment header. Numeric dtype and
// format values are intentionally transport-independent.
struct ReadWorkloadCollectionInfo {
    std::uint64_t collection_id_low = 0;
    std::uint64_t collection_id_high = 0;
    std::uint32_t format = 0;
    std::uint32_t dtype = 0;
    std::uint32_t dimension = 0;
    std::uint32_t record_size_bytes = 0;
};

enum class ReadWorkloadMethod : std::uint32_t {
    Get = 1,
    StreamGet = 2,
};

struct ReadWorkloadSnapshot {
    std::uint64_t catalog_generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t row_count = 0;
    std::uint64_t stream_id_low = 0;
    std::uint64_t stream_id_high = 0;
    std::uint64_t batch_sequence = 0;
    bool exact = false;
};

struct ReadWorkloadLogStats {
    std::uint64_t observed_events = 0;
    std::uint64_t observed_ids = 0;
    std::uint64_t unsampled_events = 0;
    std::uint64_t sampled_events = 0;
    std::uint64_t enqueued_events = 0;
    std::uint64_t written_events = 0;
    std::uint64_t captured_ids = 0;
    std::uint64_t truncated_events = 0;
    std::uint64_t sampled_ids = 0;
    std::uint64_t enqueued_ids = 0;
    std::uint64_t written_ids = 0;
    std::uint64_t dropped_busy = 0;
    std::uint64_t dropped_busy_ids = 0;
    std::uint64_t dropped_full = 0;
    std::uint64_t dropped_full_ids = 0;
    std::uint64_t dropped_runtime_failure = 0;
    std::uint64_t dropped_runtime_failure_ids = 0;
    std::uint64_t dropped_min_free = 0;
    std::uint64_t dropped_min_free_ids = 0;
    std::uint64_t dropped_io_budget = 0;
    std::uint64_t dropped_io_budget_ids = 0;
    std::uint64_t write_errors = 0;
    std::uint64_t bytes_written = 0;
    std::uint64_t segments_created = 0;
    std::uint64_t rotated_segments = 0;
    std::uint64_t retained_bytes = 0;
    std::uint64_t retained_segments = 0;
    std::uint64_t retention_deleted_bytes = 0;
    std::uint64_t retention_deleted_segments = 0;
    std::uint64_t queued_events = 0;
    std::uint64_t queued_bytes = 0;
    std::uint64_t peak_queued_bytes = 0;
    std::uint64_t buffer_capacity_bytes = 0;
    int last_error_number = 0;
    bool paused_for_min_free = false;
    bool runtime_failed = false;
    bool shutting_down = false;
};

// On-disk v1 constants are public only inside the server detail namespace so
// tooling/tests can parse the append-only format without duplicating magic
// sizes. All integer fields are little-endian.
inline constexpr std::size_t kReadWorkloadLogHeaderBytes = 4'096;
inline constexpr std::size_t kReadWorkloadRecordHeaderBytes = 256;
inline constexpr std::size_t kReadWorkloadRecordCrcOffset = 24;

class ReadWorkloadLogger final {
public:
    // Explicit configuration errors (including an invalid or unwritable
    // directory) throw. Once open succeeds, all request-path operations are
    // noexcept and storage failures disable only the logger.
    static std::unique_ptr<ReadWorkloadLogger> open(
        ReadWorkloadLogOptions options,
        ReadWorkloadCollectionInfo collection);

    ~ReadWorkloadLogger();
    ReadWorkloadLogger(const ReadWorkloadLogger&) = delete;
    ReadWorkloadLogger& operator=(const ReadWorkloadLogger&) = delete;

    // Whole-event sampling happens before queue admission. A sampled oversized
    // request contributes a bounded, near-uniform systematic sample of its
    // IDs with deterministic worst-case O(max_ids_per_event) CPU.
    // true means the event was queued, not that it is already durable.
    [[nodiscard]] bool try_log(
        ReadWorkloadMethod method,
        std::span<const std::uint64_t> ids,
        ReadWorkloadSnapshot snapshot,
        std::uint32_t prefix_dimensions) noexcept;

    [[nodiscard]] ReadWorkloadLogStats stats() const noexcept;

    // Drains accepted events, writes a final counter checkpoint and finalizes
    // the active .open segment. Safe to call repeatedly.
    void shutdown() noexcept;

private:
    class Impl;
    explicit ReadWorkloadLogger(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server::detail
