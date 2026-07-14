#include "read_scheduler.hpp"

#include "throwdb/errors.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace throwdb::detail {
namespace {

[[nodiscard]] constexpr std::uint64_t saturating_add(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return left > std::numeric_limits<std::uint64_t>::max() - right
        ? std::numeric_limits<std::uint64_t>::max()
        : left + right;
}

[[nodiscard]] constexpr std::uint64_t saturating_multiply(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return right != 0
               && left > std::numeric_limits<std::uint64_t>::max() / right
        ? std::numeric_limits<std::uint64_t>::max()
        : left * right;
}

[[nodiscard]] constexpr std::uint64_t rounded_cost_bytes(
    const std::uint64_t bytes) noexcept {
    if (bytes == 0) {
        return 0;
    }
    const std::uint64_t blocks =
        bytes / kReadSchedulerCostFloorBytes
        + (bytes % kReadSchedulerCostFloorBytes != 0 ? 1 : 0);
    return saturating_multiply(blocks, kReadSchedulerCostFloorBytes);
}

void validate_options(const ReadSchedulerOptions& options) {
    if (options.budget.max_inflight_dispatches == 0
        || options.budget.max_inflight_ranges == 0
        || options.budget.max_inflight_bytes == 0) {
        throw InvalidArgumentError(
            "read scheduler budgets must all be positive");
    }
    if (options.quantum_cost == 0) {
        throw InvalidArgumentError(
            "read scheduler quantum must be positive");
    }
    if (options.max_weight == 0
        || options.max_weight > kMaximumReadSchedulerWeight) {
        throw InvalidArgumentError(
            "read scheduler max_weight must be between 1 and 16");
    }
}

}  // namespace

std::uint64_t read_scheduler_cost(
    const std::uint64_t range_count,
    const std::uint64_t physical_bytes) noexcept {
    const std::uint64_t byte_cost = rounded_cost_bytes(physical_bytes);
    const std::uint64_t iops_cost = saturating_multiply(
        range_count, kReadSchedulerCostFloorBytes);
    return std::max(byte_cost, iops_cost);
}

class ReadScheduler::Impl {
private:
    struct Unit {
        ReadWorkUnit work;
        std::uint64_t cost{0};
    };

    struct Job {
        ReadJobId id{0};
        std::deque<Unit> pending;
        std::uint32_t weight{1};
        std::optional<std::uint64_t> deadline_tick;
        std::uint64_t deficit{0};
        std::uint64_t inflight{0};
        std::uint64_t total_dispatches{0};
        std::uint64_t completed_dispatches{0};
        std::uint64_t failure_code{0};
        std::optional<ReadJobOutcome> terminal_cause;
        bool queued{false};
        bool completion_emitted{false};
    };

    struct Inflight {
        ReadJobId job_id{0};
        std::uint64_t range_count{0};
        std::uint64_t physical_bytes{0};
    };

public:
    explicit Impl(const ReadSchedulerOptions options) : options_(options) {
        validate_options(options_);
    }

    [[nodiscard]] ReadJobId submit(
        ReadJobSpec spec,
        const std::uint64_t now_tick) {
        if (stopping_) {
            throw StorageError("read scheduler is stopping");
        }
        if (spec.weight == 0 || spec.weight > options_.max_weight) {
            throw InvalidArgumentError(
                "read job weight is outside the configured bound");
        }
        if (next_job_id_ == 0) {
            throw StorageError("read scheduler job ID space is exhausted");
        }

        Job job;
        job.id = next_job_id_++;
        job.weight = spec.weight;
        job.deadline_tick = spec.deadline_tick;
        for (const ReadWorkUnit& work : spec.work) {
            if (work.range_count == 0 || work.physical_bytes == 0) {
                throw InvalidArgumentError(
                    "read work ranges and bytes must be positive");
            }
            job.pending.push_back(Unit{
                work,
                read_scheduler_cost(work.range_count, work.physical_bytes)});
        }

        const ReadJobId id = job.id;
        auto [position, inserted] = jobs_.emplace(id, std::move(job));
        if (!inserted) {
            throw StorageError("read scheduler reused a live job ID");
        }
        Job& stored = position->second;
        if (stored.deadline_tick.has_value()
            && now_tick >= *stored.deadline_tick) {
            mark_terminal(stored, ReadJobOutcome::DeadlineExceeded);
        } else if (stored.pending.empty()) {
            mark_terminal(stored, ReadJobOutcome::Succeeded);
        } else {
            enqueue(stored);
        }
        emit_if_drained(stored);
        return id;
    }

    [[nodiscard]] bool cancel(const ReadJobId job_id) {
        const auto position = jobs_.find(job_id);
        if (position == jobs_.end()
            || position->second.completion_emitted
            || position->second.terminal_cause.has_value()) {
            return false;
        }
        Job& job = position->second;
        mark_terminal(job, ReadJobOutcome::Cancelled);
        emit_if_drained(job);
        return true;
    }

    [[nodiscard]] std::vector<ReadDispatch> pump(
        const std::uint64_t now_tick) {
        expire_deadlines(now_tick);
        if (stopping_) {
            return {};
        }

        std::vector<ReadDispatch> result;
        const std::uint64_t available_dispatches =
            options_.budget.max_inflight_dispatches
            - inflight_dispatches_;
        const std::size_t available_for_reserve =
            available_dispatches
                    > static_cast<std::uint64_t>(
                          std::numeric_limits<std::size_t>::max())
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(available_dispatches);
        result.reserve(std::min(
            available_for_reserve, pending_unit_count()));
        std::size_t visits_without_dispatch = 0;
        while (true) {
            const std::size_t runnable = runnable_job_count();
            if (runnable == 0) {
                break;
            }
            const bool solo = runnable == 1;
            const std::optional<ReadJobId> selected = pop_runnable();
            if (!selected.has_value()) {
                break;
            }
            Job& job = jobs_.at(*selected);
            if (!solo) {
                job.deficit = saturating_add(
                    job.deficit,
                    saturating_multiply(
                        options_.quantum_cost, job.weight));
            }

            bool dispatched = false;
            do {
                const Unit& unit = job.pending.front();
                const bool oversized = individually_oversized(unit);
                const bool deficit_ready = solo || unit.cost <= job.deficit;
                if (!deficit_ready || !can_admit(unit, oversized)) {
                    break;
                }
                result.push_back(issue(job, oversized, solo));
                dispatched = true;
                // Fair mode emits one indivisible unit per visit. Solo mode
                // keeps bursting until credits are exhausted.
            } while (solo && !job.pending.empty());

            if (!job.pending.empty()
                && !job.terminal_cause.has_value()) {
                enqueue(job);
            } else {
                emit_if_drained(job);
            }

            if (dispatched) {
                visits_without_dispatch = 0;
            } else {
                ++visits_without_dispatch;
                if (visits_without_dispatch >= runnable_job_count()) {
                    break;
                }
            }
        }
        return result;
    }

    void complete(
        const ReadDispatchId dispatch_id,
        const ReadDispatchStatus status,
        const std::uint64_t failure_code) {
        if (status != ReadDispatchStatus::Succeeded
            && status != ReadDispatchStatus::Failed) {
            throw InvalidArgumentError(
                "read dispatch completion has an unknown status");
        }
        const auto position = inflight_.find(dispatch_id);
        if (position == inflight_.end()) {
            throw InvalidArgumentError(
                "read dispatch completion is unknown or duplicated");
        }
        const Inflight record = position->second;
        inflight_.erase(position);
        if (inflight_dispatches_ == 0
            || inflight_ranges_ < record.range_count
            || inflight_bytes_ < record.physical_bytes) {
            throw StorageError("read scheduler credit accounting underflow");
        }
        --inflight_dispatches_;
        inflight_ranges_ -= record.range_count;
        inflight_bytes_ -= record.physical_bytes;

        const auto job_position = jobs_.find(record.job_id);
        if (job_position == jobs_.end()) {
            throw StorageError("read dispatch lost its owning job");
        }
        Job& job = job_position->second;
        if (job.inflight == 0) {
            throw StorageError("read job in-flight accounting underflow");
        }
        --job.inflight;
        ++job.completed_dispatches;
        if (status == ReadDispatchStatus::Failed
            && !job.terminal_cause.has_value()) {
            job.failure_code = failure_code;
            mark_terminal(job, ReadJobOutcome::Failed);
        }
        if (!job.terminal_cause.has_value()
            && job.pending.empty() && job.inflight == 0) {
            mark_terminal(job, ReadJobOutcome::Succeeded);
        }
        emit_if_drained(job);
    }

    [[nodiscard]] std::vector<ReadJobCompletion> take_completions() {
        std::vector<ReadJobCompletion> result;
        result.reserve(completions_.size());
        while (!completions_.empty()) {
            result.push_back(completions_.front());
            completions_.pop_front();
        }
        for (const ReadJobCompletion& completion : result) {
            const auto position = jobs_.find(completion.job_id);
            if (position != jobs_.end()
                && position->second.completion_emitted
                && position->second.inflight == 0) {
                jobs_.erase(position);
            }
        }
        return result;
    }

    void shutdown() {
        if (stopping_) {
            return;
        }
        stopping_ = true;
        active_.clear();
        for (auto& [id, job] : jobs_) {
            static_cast<void>(id);
            job.queued = false;
            if (job.completion_emitted) {
                continue;
            }
            if (!job.terminal_cause.has_value()) {
                mark_terminal(job, ReadJobOutcome::Shutdown);
            }
            emit_if_drained(job);
        }
    }

    [[nodiscard]] ReadSchedulerSnapshot snapshot() const noexcept {
        ReadSchedulerSnapshot result;
        for (const auto& [id, job] : jobs_) {
            static_cast<void>(id);
            if (job.completion_emitted) {
                continue;
            }
            if (!job.terminal_cause.has_value() && !job.pending.empty()) {
                ++result.runnable_jobs;
            } else if (job.terminal_cause.has_value()
                       && job.inflight != 0) {
                ++result.draining_jobs;
            }
        }
        result.inflight_dispatches = inflight_dispatches_;
        result.inflight_ranges = inflight_ranges_;
        result.inflight_bytes = inflight_bytes_;
        result.pending_completion_events = completions_.size();
        result.stopping = stopping_;
        return result;
    }

private:
    void enqueue(Job& job) {
        if (job.queued || job.pending.empty()
            || job.terminal_cause.has_value()) {
            return;
        }
        active_.push_back(job.id);
        job.queued = true;
    }

    [[nodiscard]] std::optional<ReadJobId> pop_runnable() {
        while (!active_.empty()) {
            const ReadJobId id = active_.front();
            active_.pop_front();
            const auto position = jobs_.find(id);
            if (position == jobs_.end()) {
                continue;
            }
            Job& job = position->second;
            job.queued = false;
            if (!job.completion_emitted
                && !job.terminal_cause.has_value()
                && !job.pending.empty()) {
                return id;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t runnable_job_count() const noexcept {
        std::size_t count = 0;
        for (const auto& [id, job] : jobs_) {
            static_cast<void>(id);
            if (!job.completion_emitted
                && !job.terminal_cause.has_value()
                && !job.pending.empty()) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] std::size_t pending_unit_count() const noexcept {
        std::size_t count = 0;
        for (const auto& [id, job] : jobs_) {
            static_cast<void>(id);
            if (job.completion_emitted
                || job.terminal_cause.has_value()) {
                continue;
            }
            if (job.pending.size()
                > std::numeric_limits<std::size_t>::max() - count) {
                return std::numeric_limits<std::size_t>::max();
            }
            count += job.pending.size();
        }
        return count;
    }

    [[nodiscard]] bool individually_oversized(
        const Unit& unit) const noexcept {
        return unit.work.range_count
                   > options_.budget.max_inflight_ranges
               || unit.work.physical_bytes
                      > options_.budget.max_inflight_bytes;
    }

    [[nodiscard]] bool can_admit(
        const Unit& unit,
        const bool oversized) const noexcept {
        if (inflight_dispatches_
            >= options_.budget.max_inflight_dispatches) {
            return false;
        }
        if (oversized) {
            return inflight_dispatches_ == 0
                   && inflight_ranges_ == 0 && inflight_bytes_ == 0;
        }
        if (inflight_ranges_ > options_.budget.max_inflight_ranges
            || inflight_bytes_ > options_.budget.max_inflight_bytes) {
            // An oversized-single dispatch owns the domain exclusively until
            // it drains; unsigned remaining-credit arithmetic must not wrap.
            return false;
        }
        return unit.work.range_count
                   <= options_.budget.max_inflight_ranges - inflight_ranges_
               && unit.work.physical_bytes
                      <= options_.budget.max_inflight_bytes - inflight_bytes_;
    }

    [[nodiscard]] ReadDispatch issue(
        Job& job,
        const bool oversized,
        const bool solo) {
        if (next_dispatch_id_ == 0) {
            throw StorageError(
                "read scheduler dispatch ID space is exhausted");
        }
        const Unit& unit = job.pending.front();
        const ReadDispatchId dispatch_id = next_dispatch_id_;
        const auto [position, inserted] = inflight_.emplace(
            dispatch_id,
            Inflight{
                job.id,
                unit.work.range_count,
                unit.work.physical_bytes});
        static_cast<void>(position);
        if (!inserted) {
            throw StorageError(
                "read scheduler reused an in-flight dispatch ID");
        }
        const ReadDispatch result{
            dispatch_id,
            job.id,
            unit.work.user_token,
            unit.work.range_count,
            unit.work.physical_bytes,
            unit.cost,
            oversized};

        ++next_dispatch_id_;
        if (!solo) {
            job.deficit -= unit.cost;
        }
        ++job.inflight;
        ++job.total_dispatches;
        ++inflight_dispatches_;
        inflight_ranges_ = saturating_add(
            inflight_ranges_, unit.work.range_count);
        inflight_bytes_ = saturating_add(
            inflight_bytes_, unit.work.physical_bytes);
        job.pending.pop_front();
        return result;
    }

    void expire_deadlines(const std::uint64_t now_tick) {
        for (auto& [id, job] : jobs_) {
            static_cast<void>(id);
            if (!job.completion_emitted
                && !job.terminal_cause.has_value()
                && job.deadline_tick.has_value()
                && now_tick >= *job.deadline_tick) {
                mark_terminal(job, ReadJobOutcome::DeadlineExceeded);
                emit_if_drained(job);
            }
        }
    }

    static void mark_terminal(
        Job& job,
        const ReadJobOutcome outcome) {
        if (!job.terminal_cause.has_value()) {
            job.terminal_cause = outcome;
        }
        job.pending.clear();
        job.queued = false;
    }

    void emit_if_drained(Job& job) {
        if (job.completion_emitted || job.inflight != 0
            || !job.terminal_cause.has_value()) {
            return;
        }
        job.completion_emitted = true;
        job.queued = false;
        completions_.push_back(ReadJobCompletion{
            job.id,
            *job.terminal_cause,
            job.failure_code,
            job.total_dispatches,
            job.completed_dispatches});
    }

    ReadSchedulerOptions options_;
    std::map<ReadJobId, Job> jobs_;
    std::deque<ReadJobId> active_;
    std::map<ReadDispatchId, Inflight> inflight_;
    std::deque<ReadJobCompletion> completions_;
    ReadJobId next_job_id_{1};
    ReadDispatchId next_dispatch_id_{1};
    std::uint64_t inflight_dispatches_{0};
    std::uint64_t inflight_ranges_{0};
    std::uint64_t inflight_bytes_{0};
    bool stopping_{false};
};

ReadScheduler::ReadScheduler(const ReadSchedulerOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

ReadScheduler::~ReadScheduler() = default;

ReadJobId ReadScheduler::submit(
    ReadJobSpec spec,
    const std::uint64_t now_tick) {
    return impl_->submit(std::move(spec), now_tick);
}

bool ReadScheduler::cancel(const ReadJobId job_id) {
    return impl_->cancel(job_id);
}

std::vector<ReadDispatch> ReadScheduler::pump(
    const std::uint64_t now_tick) {
    return impl_->pump(now_tick);
}

void ReadScheduler::complete(
    const ReadDispatchId dispatch_id,
    const ReadDispatchStatus status,
    const std::uint64_t failure_code) {
    impl_->complete(dispatch_id, status, failure_code);
}

std::vector<ReadJobCompletion> ReadScheduler::take_completions() {
    return impl_->take_completions();
}

void ReadScheduler::shutdown() { impl_->shutdown(); }

ReadSchedulerSnapshot ReadScheduler::snapshot() const noexcept {
    return impl_->snapshot();
}

}  // namespace throwdb::detail
