#pragma once

#include "io_uring_engine.hpp"
#include "read_scheduler.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace throwdb::detail {

// A single persistent ring (or a deterministic fake in tests). The
// coordinator guarantees that at most one thread calls a lane at a time.
class IoUringLane {
public:
    virtual ~IoUringLane() = default;

    virtual void read_exact(
        std::span<const IoUringReadOperation> operations) = 0;

    // Ordinary per-read errors leave a lane usable. False is reserved for a
    // fatal ring/protocol failure and poisons the complete coordinator domain.
    [[nodiscard]] virtual bool usable() const noexcept = 0;
};

using IoUringLaneFactory = std::function<std::unique_ptr<IoUringLane>(
    std::size_t lane_index,
    std::uint32_t lane_queue_depth)>;

struct IoUringCoordinatorSnapshot {
    ReadSchedulerSnapshot scheduler;
    std::uint64_t admitted_jobs{0};
    std::uint64_t admission_waiters{0};
    bool domain_failed{false};
};

// Fair synchronous facade over one or more persistent io_uring lanes. One
// caller drives the single-lane topology. Multi-lane topologies have exactly
// one permanent owner thread per lane. ReadScheduler bounds aggregate
// dispatch, range and byte credits and applies DRR between concurrent calls.
class IoUringReadCoordinator {
public:
    using Task = std::function<void(
        std::uint64_t task_index,
        IoUringLane& lane)>;

    IoUringReadCoordinator(
        std::uint32_t worker_count,
        std::uint32_t total_queue_depth,
        std::uint64_t max_read_range_bytes,
        IoUringLaneFactory lane_factory = {});
    ~IoUringReadCoordinator();

    IoUringReadCoordinator(const IoUringReadCoordinator&) = delete;
    IoUringReadCoordinator& operator=(const IoUringReadCoordinator&) = delete;
    IoUringReadCoordinator(IoUringReadCoordinator&&) = delete;
    IoUringReadCoordinator& operator=(IoUringReadCoordinator&&) = delete;

    // The task may retain references owned by this caller. execute() does not
    // return until every task already issued for this job has drained, even
    // when another task fails or the coordinator shuts down.
    void execute(std::vector<ReadWorkUnit> work, Task task);

    // Stops admission and drains already-issued lane calls. Primarily exposed
    // for deterministic lifecycle tests; normal owners use the destructor.
    void shutdown() noexcept;

    [[nodiscard]] std::uint32_t worker_count() const noexcept;
    [[nodiscard]] std::uint32_t total_queue_depth() const noexcept;
    [[nodiscard]] IoUringCoordinatorSnapshot snapshot() const;

    // A child process must reopen the collection instead of touching inherited
    // ring/thread state. Abandoning intentionally leaks only the child copy.
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
