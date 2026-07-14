#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace throwdb::detail {

inline constexpr std::uint64_t kReadSchedulerCostFloorBytes = 4096;
inline constexpr std::uint32_t kMaximumReadSchedulerWeight = 16;

// Charge both transferred bytes and random-I/O pressure. Every physical range
// costs at least 4 KiB, while larger contiguous reads are charged by their
// 4-KiB-rounded byte count. Overflow saturates instead of wrapping.
[[nodiscard]] std::uint64_t read_scheduler_cost(
    std::uint64_t range_count,
    std::uint64_t physical_bytes) noexcept;

struct ReadSchedulerBudget {
    std::uint64_t max_inflight_dispatches{1};
    std::uint64_t max_inflight_ranges{1};
    std::uint64_t max_inflight_bytes{kReadSchedulerCostFloorBytes};
};

struct ReadSchedulerOptions {
    ReadSchedulerBudget budget;
    std::uint64_t quantum_cost{256ULL << 10};
    std::uint32_t max_weight{kMaximumReadSchedulerWeight};
};

// One indivisible backend dispatch. A later planner is responsible for
// bounding these units to the desired fairness granularity.
struct ReadWorkUnit {
    std::uint64_t user_token{0};
    std::uint64_t range_count{0};
    std::uint64_t physical_bytes{0};
};

struct ReadJobSpec {
    std::vector<ReadWorkUnit> work;
    std::uint32_t weight{1};
    // Opaque monotonic tick in the same domain as submit()/pump() now_tick.
    // Equality means expired. No wall clock is read by the scheduler.
    std::optional<std::uint64_t> deadline_tick;
};

using ReadJobId = std::uint64_t;
using ReadDispatchId = std::uint64_t;

struct ReadDispatch {
    ReadDispatchId dispatch_id{0};
    ReadJobId job_id{0};
    std::uint64_t user_token{0};
    std::uint64_t range_count{0};
    std::uint64_t physical_bytes{0};
    std::uint64_t cost{0};
    // An individually over-budget unit may run only as the sole global
    // in-flight dispatch. This escape prevents an oversized row from deadlock.
    bool oversized_single{false};
};

enum class ReadDispatchStatus : std::uint8_t {
    Succeeded = 1,
    Failed = 2,
};

enum class ReadJobOutcome : std::uint8_t {
    Succeeded = 1,
    Cancelled = 2,
    DeadlineExceeded = 3,
    Failed = 4,
    Shutdown = 5,
};

struct ReadJobCompletion {
    ReadJobId job_id{0};
    ReadJobOutcome outcome{ReadJobOutcome::Succeeded};
    std::uint64_t failure_code{0};
    std::uint64_t total_dispatches{0};
    std::uint64_t completed_dispatches{0};
};

struct ReadSchedulerSnapshot {
    std::uint64_t runnable_jobs{0};
    std::uint64_t draining_jobs{0};
    std::uint64_t inflight_dispatches{0};
    std::uint64_t inflight_ranges{0};
    std::uint64_t inflight_bytes{0};
    std::uint64_t pending_completion_events{0};
    bool stopping{false};
};

// A thread-confined deterministic DRR state machine. It owns no threads and
// reads no clock. The future runtime coordinator will serialize calls, submit
// pump() results to a backend, and feed completions back explicitly.
class ReadScheduler {
public:
    explicit ReadScheduler(ReadSchedulerOptions options);
    ~ReadScheduler();

    ReadScheduler(const ReadScheduler&) = delete;
    ReadScheduler& operator=(const ReadScheduler&) = delete;
    ReadScheduler(ReadScheduler&&) = delete;
    ReadScheduler& operator=(ReadScheduler&&) = delete;

    [[nodiscard]] ReadJobId submit(
        ReadJobSpec spec,
        std::uint64_t now_tick = 0);

    // Returns false for an unknown or already-terminal job. Cancellation is
    // cooperative: no new work is issued, but completion is emitted only after
    // every already-issued dispatch has drained.
    [[nodiscard]] bool cancel(ReadJobId job_id);

    // Advances DRR while credits permit. With one runnable job it bypasses the
    // deficit and fills available credits (work-conserving solo burst). With
    // multiple jobs it issues at most one work unit per visit and retains
    // deficit across calls. Calling pump again advances another empty DRR
    // round when every head unit costs more than its current deficit.
    [[nodiscard]] std::vector<ReadDispatch> pump(std::uint64_t now_tick);

    // failure_code is opaque and only retained for the first terminal failure.
    // A dispatch ID is accepted exactly once; duplicate/unknown completion is
    // an InvalidArgumentError.
    void complete(
        ReadDispatchId dispatch_id,
        ReadDispatchStatus status,
        std::uint64_t failure_code = 0);

    [[nodiscard]] std::vector<ReadJobCompletion> take_completions();

    // Stops admission deterministically. Queued jobs complete immediately as
    // Shutdown; issued work must still drain before its one completion event.
    void shutdown();

    [[nodiscard]] ReadSchedulerSnapshot snapshot() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
