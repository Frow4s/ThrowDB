#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

#include "throwdb/storage.hpp"

namespace throwdb::detail {

// One already-resolved fixed-width row. The caller owns the file descriptor
// and must keep it open until execute() returns. output_index identifies the
// row-sized slot in the caller-provided output buffer.
struct PhysicalReadRequest {
    int fd{-1};
    std::uint64_t absolute_offset{0};
    std::uint64_t output_index{0};
};

// Per-call diagnostics for an explicitly profiled physical read. Planning
// covers request validation, sorting, range merging, and task partitioning.
// Backend time deliberately covers the complete synchronous execution phase:
// queue/admission wait, physical I/O, and scatter copies. Those operations can
// overlap when a parallel backend is active, so this first diagnostic surface
// does not pretend that they are independent wall-clock phases.
struct PhysicalReadStats {
    double plan_seconds{0.0};
    double backend_seconds{0.0};
    std::uint64_t request_count{0};
    std::uint64_t range_count{0};
    std::uint64_t task_count{0};
    std::uint64_t planned_read_bytes{0};
};

// Private deterministic test seam for the thread-pool coordinator. Production
// readers leave it null, so the physical task fast path has only one pointer
// check. Hooks must be thread-safe and must not call back into the reader.
struct PhysicalReaderTestHooks {
    std::function<void(std::size_t task_count)> before_admission_wait;
    std::function<void(std::size_t task_count)> after_job_submitted;
    std::function<void(std::size_t task_count, std::size_t task_index)>
        before_task;
    // Runs under the coordinator lock after every dispatch returned by one
    // pump has been handed to the executor. Intended for drain/error tests.
    std::function<void()> after_dispatch_batch_submitted;
};

// Executes physical reads after logical ID lookup and RowRef resolution have
// completed. One instance owns a bounded backend shared by all concurrent
// execute() calls; execution is synchronous from each caller's perspective.
class PhysicalReader {
public:
    explicit PhysicalReader(ReaderOptions options = {});
    PhysicalReader(
        ReaderOptions options,
        std::shared_ptr<const PhysicalReaderTestHooks> test_hooks);
    ~PhysicalReader();

    PhysicalReader(const PhysicalReader&) = delete;
    PhysicalReader& operator=(const PhysicalReader&) = delete;
    PhysicalReader(PhysicalReader&&) = delete;
    PhysicalReader& operator=(PhysicalReader&&) = delete;

    // Reads one record for every request. Geometry and every request are
    // validated before output is touched. output_index values must form an
    // exact permutation of [0, requests.size()).
    void execute(
        std::uint64_t record_size,
        std::span<const PhysicalReadRequest> requests,
        std::span<std::byte> output) const;

    // Execute the same read while collecting diagnostics for this call only.
    // The ordinary execute() path is a separate compile-time specialization
    // and performs no telemetry clock reads, atomics, or locking.
    [[nodiscard]] PhysicalReadStats execute_profiled(
        std::uint64_t record_size,
        std::span<const PhysicalReadRequest> requests,
        std::span<std::byte> output) const;

    // Drop the child-process wrapper around inherited executor thread state.
    // This intentionally leaks only that process-local state until exit.
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
