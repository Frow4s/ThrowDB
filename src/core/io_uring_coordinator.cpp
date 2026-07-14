#include "io_uring_coordinator.hpp"

#include "io_uring_pool.hpp"
#include "throwdb/errors.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace throwdb::detail {
namespace {

[[nodiscard]] constexpr std::uint64_t saturating_multiply(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return right != 0
               && left > std::numeric_limits<std::uint64_t>::max() / right
        ? std::numeric_limits<std::uint64_t>::max()
        : left * right;
}

[[nodiscard]] ReadSchedulerOptions coordinator_scheduler_options(
    const std::uint32_t worker_count,
    const std::uint32_t total_queue_depth,
    const std::uint64_t max_read_range_bytes) {
    return ReadSchedulerOptions{
        ReadSchedulerBudget{
            worker_count,
            total_queue_depth,
            saturating_multiply(
                total_queue_depth, max_read_range_bytes)},
        256ULL << 10,
        kMaximumReadSchedulerWeight};
}

[[nodiscard]] std::size_t admitted_job_capacity(
    const std::uint32_t worker_count,
    const std::uint32_t total_queue_depth) noexcept {
    const std::size_t workers = worker_count;
    const std::size_t depth = total_queue_depth;
    return depth > std::numeric_limits<std::size_t>::max() - workers
        ? std::numeric_limits<std::size_t>::max()
        : workers + depth;
}

[[nodiscard]] std::exception_ptr make_storage_error(
    const char* const message) noexcept {
    try {
        throw StorageError(message);
    } catch (...) {
        return std::current_exception();
    }
}

class EngineLane final : public IoUringLane {
public:
    explicit EngineLane(const std::uint32_t queue_depth)
        : engine_(queue_depth) {}

    void read_exact(
        const std::span<const IoUringReadOperation> operations) override {
        engine_.read_exact(operations);
    }

    [[nodiscard]] bool usable() const noexcept override {
        return engine_.usable();
    }

private:
    IoUringEngine engine_;
};

[[nodiscard]] IoUringLaneFactory production_lane_factory() {
    return [](const std::size_t, const std::uint32_t queue_depth) {
        return std::make_unique<EngineLane>(queue_depth);
    };
}

class CoordinatorJob {
public:
    using Task = IoUringReadCoordinator::Task;

    explicit CoordinatorJob(Task task) : task_(std::move(task)) {}

    void run(const std::uint64_t task_index, IoUringLane& lane) const {
        task_(task_index, lane);
    }

    // These three methods are called with the coordinator mutex held.
    void remember_task_error(std::exception_ptr error) noexcept {
        if (first_task_error_ == nullptr) {
            first_task_error_ = std::move(error);
        }
    }

    void remember_terminal_error(std::exception_ptr error) noexcept {
        if (terminal_error_ == nullptr) {
            terminal_error_ = std::move(error);
        }
    }

    void finish(const ReadJobOutcome outcome) noexcept {
        std::exception_ptr result_error;
        switch (outcome) {
            case ReadJobOutcome::Succeeded:
                break;
            case ReadJobOutcome::Failed:
                result_error = first_task_error_ != nullptr
                    ? first_task_error_
                    : make_storage_error("scheduled io_uring read failed");
                break;
            case ReadJobOutcome::Cancelled:
                result_error = terminal_error_ != nullptr
                    ? terminal_error_
                    : make_storage_error(
                          "scheduled io_uring read was cancelled");
                break;
            case ReadJobOutcome::DeadlineExceeded:
                result_error = make_storage_error(
                    "scheduled io_uring read exceeded its deadline");
                break;
            case ReadJobOutcome::Shutdown:
                result_error = terminal_error_ != nullptr
                    ? terminal_error_
                    : make_storage_error(
                          "io_uring read coordinator is stopping");
                break;
        }

        {
            std::lock_guard lock(wait_mutex_);
            if (finished_) {
                return;
            }
            result_error_ = std::move(result_error);
            finished_ = true;
        }
        finished_condition_.notify_all();
    }

    [[nodiscard]] std::exception_ptr wait() {
        std::unique_lock lock(wait_mutex_);
        finished_condition_.wait(lock, [this] { return finished_; });
        return result_error_;
    }

private:
    Task task_;
    std::exception_ptr first_task_error_;
    std::exception_ptr terminal_error_;
    std::mutex wait_mutex_;
    std::condition_variable finished_condition_;
    std::exception_ptr result_error_;
    bool finished_{false};
};

}  // namespace

class IoUringReadCoordinator::Impl {
private:
    struct Assignment {
        ReadDispatch dispatch;
        std::shared_ptr<CoordinatorJob> job;
    };

    struct LaneState {
        explicit LaneState(std::unique_ptr<IoUringLane> input_lane)
            : lane(std::move(input_lane)) {}

        std::unique_ptr<IoUringLane> lane;
        std::condition_variable ready;
        std::optional<Assignment> assignment;
        std::thread owner;
        bool busy{false};
    };

public:
    Impl(
        const std::uint32_t worker_count,
        const std::uint32_t total_queue_depth,
        const std::uint64_t max_read_range_bytes,
        IoUringLaneFactory lane_factory)
        : worker_count_(worker_count),
          total_queue_depth_(total_queue_depth),
          max_admitted_jobs_(admitted_job_capacity(
              worker_count, total_queue_depth)),
          scheduler_(coordinator_scheduler_options(
              worker_count, total_queue_depth, max_read_range_bytes)) {
        if (worker_count == 0) {
            throw InvalidArgumentError(
                "io_uring coordinator worker count must be positive");
        }
        if (total_queue_depth < worker_count) {
            throw InvalidArgumentError(
                "io_uring queue depth must be at least its worker count");
        }
        if (max_read_range_bytes == 0) {
            throw InvalidArgumentError(
                "io_uring maximum read range must be positive");
        }
        if (!lane_factory) {
            lane_factory = production_lane_factory();
        }

        const std::vector<std::uint32_t> lane_depths =
            split_io_uring_queue_depth(worker_count, total_queue_depth);
        lanes_.reserve(worker_count);
        for (std::size_t lane_index = 0;
             lane_index != lane_depths.size(); ++lane_index) {
            std::unique_ptr<IoUringLane> lane = lane_factory(
                lane_index, lane_depths[lane_index]);
            if (lane == nullptr) {
                throw StorageError(
                    "io_uring lane factory returned a null lane");
            }
            if (!lane->usable()) {
                throw StorageError(
                    "io_uring lane factory returned an unusable lane");
            }
            lanes_.push_back(
                std::make_unique<LaneState>(std::move(lane)));
        }

        if (worker_count_ > 1) {
            try {
                for (std::size_t lane_index = 0;
                     lane_index != lanes_.size(); ++lane_index) {
                    lanes_[lane_index]->owner = std::thread(
                        [this, lane_index] { lane_loop(lane_index); });
                }
            } catch (...) {
                request_worker_stop_after_construction_failure();
                join_lane_threads();
                throw;
            }
        }
    }

    ~Impl() { shutdown(); }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    [[nodiscard]] std::uint32_t worker_count() const noexcept {
        return worker_count_;
    }

    [[nodiscard]] std::uint32_t total_queue_depth() const noexcept {
        return total_queue_depth_;
    }

    [[nodiscard]] IoUringCoordinatorSnapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return IoUringCoordinatorSnapshot{
            scheduler_.snapshot(),
            jobs_.size(),
            admission_waiters_.size(),
            domain_error_ != nullptr};
    }

    void execute(std::vector<ReadWorkUnit> work, Task task) {
        if (!owned_by_current_process()) {
            throw StorageError(
                "io_uring coordinator was inherited across fork; reopen "
                "the collection in the child process");
        }
        if (!task) {
            throw InvalidArgumentError(
                "io_uring coordinator task must be callable");
        }
        auto job = std::make_shared<CoordinatorJob>(std::move(task));

        ReadJobId job_id = 0;
        bool registered = false;
        {
            std::unique_lock lock(mutex_);
            wait_for_admission_locked(lock);
            try {
                job_id = scheduler_.submit(ReadJobSpec{
                    std::move(work), 1, std::nullopt});
                const auto [position, inserted] = jobs_.emplace(job_id, job);
                static_cast<void>(position);
                if (!inserted) {
                    throw StorageError(
                        "io_uring coordinator reused a live job ID");
                }
                registered = true;
                deliver_completions_locked();
                if (worker_count_ > 1) {
                    pump_multi_locked();
                }
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                if (registered) {
                    job->remember_terminal_error(error);
                    try {
                        static_cast<void>(scheduler_.cancel(job_id));
                        deliver_completions_locked();
                    } catch (...) {
                        fail_domain_locked(std::current_exception());
                    }
                } else {
                    if (job_id != 0) {
                        static_cast<void>(scheduler_.cancel(job_id));
                        deliver_completions_locked();
                    }
                    admission_available_.notify_all();
                    std::rethrow_exception(error);
                }
            }
        }

        if (worker_count_ == 1) {
            drive_single_lane_until_finished(job_id);
        }
        if (std::exception_ptr error = job->wait(); error != nullptr) {
            std::rethrow_exception(error);
        }
    }

    void shutdown() noexcept {
        if (!owned_by_current_process()) {
            return;
        }
        {
            std::lock_guard lock(mutex_);
            if (!stopping_) {
                stopping_ = true;
                const std::exception_ptr error = make_storage_error(
                    "io_uring read coordinator is stopping");
                for (auto& [job_id, job] : jobs_) {
                    static_cast<void>(job_id);
                    job->remember_terminal_error(error);
                }
                try {
                    scheduler_.shutdown();
                    deliver_completions_locked();
                } catch (...) {
                    // Lane completions retry delivery after issued work drains.
                }
                admission_available_.notify_all();
                driver_changed_.notify_all();
                notify_all_lanes_locked();
            }
        }
        if (worker_count_ == 1) {
            try {
                std::unique_lock lock(mutex_);
                driver_changed_.wait(lock, [this] {
                    return !single_driver_active_;
                });
                deliver_completions_locked();
            } catch (...) {
            }
        }
        join_lane_threads();
        try {
            std::lock_guard lock(mutex_);
            deliver_completions_locked();
        } catch (...) {
            // Destructors and explicit shutdown are noexcept.
        }
    }

private:
    void wait_for_admission_locked(std::unique_lock<std::mutex>& lock) {
        if (stopping_) {
            rethrow_domain_or_stopping_locked();
        }
        if (jobs_.size() < max_admitted_jobs_
            && admission_waiters_.empty()) {
            return;
        }
        if (next_admission_ticket_ == 0) {
            throw StorageError(
                "io_uring admission ticket space is exhausted");
        }
        const std::uint64_t ticket = next_admission_ticket_++;
        admission_waiters_.push_back(ticket);
        try {
            admission_available_.wait(lock, [this, ticket] {
                return stopping_
                    || (!admission_waiters_.empty()
                        && admission_waiters_.front() == ticket
                        && jobs_.size() < max_admitted_jobs_);
            });
        } catch (...) {
            erase_admission_waiter(ticket);
            admission_available_.notify_all();
            throw;
        }
        if (stopping_) {
            erase_admission_waiter(ticket);
            admission_available_.notify_all();
            rethrow_domain_or_stopping_locked();
        }
        if (admission_waiters_.empty()
            || admission_waiters_.front() != ticket) {
            throw StorageError(
                "io_uring admission FIFO lost its next ticket");
        }
        admission_waiters_.pop_front();
        admission_available_.notify_all();
    }

    [[noreturn]] void rethrow_domain_or_stopping_locked() const {
        if (domain_error_ != nullptr) {
            std::rethrow_exception(domain_error_);
        }
        throw StorageError("io_uring read coordinator is stopping");
    }

    void erase_admission_waiter(const std::uint64_t ticket) noexcept {
        const auto position = std::find(
            admission_waiters_.begin(), admission_waiters_.end(), ticket);
        if (position != admission_waiters_.end()) {
            admission_waiters_.erase(position);
        }
    }

    void drive_single_lane_until_finished(const ReadJobId caller_job_id) {
        while (true) {
            std::optional<Assignment> assignment;
            {
                std::unique_lock lock(mutex_);
                driver_changed_.wait(lock, [this, caller_job_id] {
                    return jobs_.find(caller_job_id) == jobs_.end()
                        || !single_driver_active_;
                });
                if (jobs_.find(caller_job_id) == jobs_.end()) {
                    return;
                }
                single_driver_active_ = true;
                try {
                    std::vector<ReadDispatch> dispatches =
                        scheduler_.pump_until_blocked(0);
                    if (dispatches.size() > 1) {
                        throw StorageError(
                            "single io_uring lane received multiple dispatches");
                    }
                    if (!dispatches.empty()) {
                        const ReadDispatch dispatch = dispatches.front();
                        const auto position = jobs_.find(dispatch.job_id);
                        if (position == jobs_.end()) {
                            throw StorageError(
                                "io_uring dispatch lost its read job");
                        }
                        assignment = Assignment{dispatch, position->second};
                    }
                } catch (...) {
                    fail_domain_locked(std::current_exception());
                }

                if (!assignment.has_value()) {
                    single_driver_active_ = false;
                    driver_changed_.notify_all();
                    if (jobs_.find(caller_job_id) == jobs_.end()) {
                        return;
                    }
                    // A live job with no dispatch can only be draining. In a
                    // single-lane topology there is no other issued lane call,
                    // so reaching this state signals an internal failure.
                    fail_domain_locked(make_storage_error(
                        "single io_uring coordinator made no progress"));
                    continue;
                }
            }

            std::exception_ptr error;
            try {
                assignment->job->run(
                    assignment->dispatch.user_token, *lanes_[0]->lane);
            } catch (...) {
                error = std::current_exception();
            }
            const bool lane_usable = lanes_[0]->lane->usable();

            {
                std::lock_guard lock(mutex_);
                try {
                    complete_assignment_locked(
                        *assignment, std::move(error), lane_usable);
                } catch (...) {
                    fail_domain_locked(std::current_exception());
                }
                single_driver_active_ = false;
                driver_changed_.notify_all();
            }
        }
    }

    void pump_multi_locked() {
        if (stopping_) {
            return;
        }
        std::vector<ReadDispatch> dispatches =
            scheduler_.pump_until_blocked(0);
        std::size_t dispatch_index = 0;
        for (const std::unique_ptr<LaneState>& lane : lanes_) {
            if (dispatch_index == dispatches.size()) {
                break;
            }
            if (lane->busy) {
                continue;
            }
            const ReadDispatch dispatch = dispatches[dispatch_index++];
            const auto position = jobs_.find(dispatch.job_id);
            if (position == jobs_.end()) {
                throw StorageError("io_uring dispatch lost its read job");
            }
            lane->assignment.emplace(Assignment{dispatch, position->second});
            lane->busy = true;
            lane->ready.notify_one();
        }
        if (dispatch_index != dispatches.size()) {
            throw StorageError(
                "io_uring scheduler exceeded the number of idle lanes");
        }
    }

    void lane_loop(const std::size_t lane_index) noexcept {
        LaneState& lane = *lanes_[lane_index];
        while (true) {
            std::optional<Assignment> assignment;
            {
                std::unique_lock lock(mutex_);
                lane.ready.wait(lock, [this, &lane] {
                    return stopping_ || lane.assignment.has_value();
                });
                if (!lane.assignment.has_value()) {
                    return;
                }
                assignment = std::move(lane.assignment);
                lane.assignment.reset();
            }

            std::exception_ptr error;
            try {
                assignment->job->run(
                    assignment->dispatch.user_token, *lane.lane);
            } catch (...) {
                error = std::current_exception();
            }
            const bool lane_usable = lane.lane->usable();

            std::lock_guard lock(mutex_);
            try {
                lane.busy = false;
                complete_assignment_locked(
                    *assignment, std::move(error), lane_usable);
                pump_multi_locked();
            } catch (...) {
                fail_domain_locked(std::current_exception());
            }
        }
    }

    void complete_assignment_locked(
        const Assignment& assignment,
        std::exception_ptr error,
        const bool lane_usable) {
        if (!lane_usable) {
            if (error == nullptr) {
                error = make_storage_error(
                    "io_uring lane became unusable");
            }
            fail_domain_locked(error);
        } else if (error != nullptr) {
            assignment.job->remember_task_error(error);
        }

        scheduler_.complete(
            assignment.dispatch.dispatch_id,
            error == nullptr
                ? ReadDispatchStatus::Succeeded
                : ReadDispatchStatus::Failed,
            error == nullptr ? 0 : 1);
        deliver_completions_locked();
    }

    void fail_domain_locked(std::exception_ptr error) noexcept {
        if (domain_error_ == nullptr) {
            domain_error_ = error != nullptr
                ? std::move(error)
                : make_storage_error("io_uring coordinator failed");
        }
        const bool first_stop = !stopping_;
        stopping_ = true;
        if (first_stop) {
            for (auto& [job_id, job] : jobs_) {
                static_cast<void>(job_id);
                job->remember_terminal_error(domain_error_);
            }
        }
        try {
            scheduler_.shutdown();
            deliver_completions_locked();
        } catch (...) {
        }
        admission_available_.notify_all();
        driver_changed_.notify_all();
        notify_all_lanes_locked();
    }

    void deliver_completions_locked() {
        std::vector<ReadJobCompletion> completions =
            scheduler_.take_completions();
        bool released_admission = false;
        for (const ReadJobCompletion& completion : completions) {
            const auto position = jobs_.find(completion.job_id);
            if (position == jobs_.end()) {
                continue;
            }
            const std::shared_ptr<CoordinatorJob> job = position->second;
            jobs_.erase(position);
            released_admission = true;
            job->finish(completion.outcome);
        }
        if (released_admission) {
            admission_available_.notify_all();
            driver_changed_.notify_all();
        }
    }

    void notify_all_lanes_locked() noexcept {
        for (const std::unique_ptr<LaneState>& lane : lanes_) {
            lane->ready.notify_all();
        }
    }

    void request_worker_stop_after_construction_failure() noexcept {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        notify_all_lanes_locked();
    }

    void join_lane_threads() noexcept {
        if (worker_count_ <= 1) {
            return;
        }
        for (const std::unique_ptr<LaneState>& lane : lanes_) {
            if (lane->owner.joinable()) {
                lane->owner.join();
            }
        }
    }

    using Task = IoUringReadCoordinator::Task;

    const std::uint32_t worker_count_;
    const std::uint32_t total_queue_depth_;
    const std::size_t max_admitted_jobs_;
    mutable std::mutex mutex_;
    std::condition_variable admission_available_;
    std::condition_variable driver_changed_;
    ReadScheduler scheduler_;
    std::map<ReadJobId, std::shared_ptr<CoordinatorJob>> jobs_;
    std::deque<std::uint64_t> admission_waiters_;
    std::vector<std::unique_ptr<LaneState>> lanes_;
    std::exception_ptr domain_error_;
    std::uint64_t next_admission_ticket_{1};
    bool single_driver_active_{false};
    bool stopping_{false};
    const pid_t owner_pid_{::getpid()};
};

IoUringReadCoordinator::IoUringReadCoordinator(
    const std::uint32_t worker_count,
    const std::uint32_t total_queue_depth,
    const std::uint64_t max_read_range_bytes,
    IoUringLaneFactory lane_factory)
    : impl_(std::make_unique<Impl>(
          worker_count,
          total_queue_depth,
          max_read_range_bytes,
          std::move(lane_factory))) {}

IoUringReadCoordinator::~IoUringReadCoordinator() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        static_cast<void>(impl_.release());
    }
}

void IoUringReadCoordinator::execute(
    std::vector<ReadWorkUnit> work,
    Task task) {
    impl_->execute(std::move(work), std::move(task));
}

void IoUringReadCoordinator::shutdown() noexcept {
    if (impl_ != nullptr) {
        impl_->shutdown();
    }
}

std::uint32_t IoUringReadCoordinator::worker_count() const noexcept {
    return impl_->worker_count();
}

std::uint32_t IoUringReadCoordinator::total_queue_depth() const noexcept {
    return impl_->total_queue_depth();
}

IoUringCoordinatorSnapshot IoUringReadCoordinator::snapshot() const {
    return impl_->snapshot();
}

void IoUringReadCoordinator::abandon_after_fork() noexcept {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        static_cast<void>(impl_.release());
    }
}

}  // namespace throwdb::detail
