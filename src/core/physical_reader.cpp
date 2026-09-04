#include "physical_reader.hpp"

#include "internal.hpp"
#include "read_scheduler.hpp"
#include "task_partition.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace throwdb::detail {
namespace {

constexpr std::uint32_t kMaxIoWorkers = 1024;
constexpr std::uint32_t kMaxIoQueueCapacity = 1U << 20;
constexpr std::uint32_t kMaxIoRangesPerTask = 1U << 20;
constexpr std::uint32_t kMaxIoUringWorkers = 64;
constexpr std::uint32_t kMaxIoUringQueueDepth = 4096;

[[nodiscard]] std::size_t effective_queue_capacity(
    const ReaderOptions& options) {
    if (options.io_queue_capacity != 0) {
        return options.io_queue_capacity;
    }
    return static_cast<std::size_t>(options.io_workers) * 2;
}

void validate_reader_options(const ReaderOptions& options) {
    if (options.io_ranges_per_task == 0
        || options.io_ranges_per_task > kMaxIoRangesPerTask) {
        throw InvalidArgumentError(
            "io_ranges_per_task must be between 1 and 1048576");
    }
    if (options.io_uring_queue_depth == 0
        || options.io_uring_queue_depth > kMaxIoUringQueueDepth) {
        throw InvalidArgumentError(
            "io_uring_queue_depth must be between 1 and 4096");
    }
    if (options.io_uring_workers == 0
        || options.io_uring_workers > kMaxIoUringWorkers) {
        throw InvalidArgumentError(
            "io_uring_workers must be between 1 and 64");
    }
    if (options.io_backend == IoBackend::IoUring) {
        if (options.io_workers != 0 || options.io_queue_capacity != 0) {
            throw InvalidArgumentError(
                "io_workers and io_queue_capacity must both be zero for "
                "the io_uring backend");
        }
        if (options.io_uring_workers > options.io_uring_queue_depth) {
            throw InvalidArgumentError(
                "io_uring_queue_depth must be at least io_uring_workers");
        }
        return;
    }
    if (options.io_backend != IoBackend::ThreadPool) {
        throw InvalidArgumentError("reader contains an unknown I/O backend");
    }
    if (options.io_workers == 0) {
        if (options.io_queue_capacity != 0) {
            throw InvalidArgumentError(
                "io_queue_capacity must be zero when io_workers is zero");
        }
        return;
    }
    if (options.io_workers > kMaxIoWorkers) {
        throw InvalidArgumentError("io_workers must not exceed 1024");
    }
    if (options.io_queue_capacity > kMaxIoQueueCapacity) {
        throw InvalidArgumentError(
            "io_queue_capacity must not exceed 1048576");
    }
}

class BoundedExecutor {
public:
    BoundedExecutor(
        const std::size_t worker_count,
        const std::size_t queue_capacity)
        : queue_capacity_(queue_capacity) {
        workers_.reserve(worker_count);
        try {
            for (std::size_t index = 0; index != worker_count; ++index) {
                workers_.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    ~BoundedExecutor() { stop_and_join(); }

    BoundedExecutor(const BoundedExecutor&) = delete;
    BoundedExecutor& operator=(const BoundedExecutor&) = delete;

    void submit(std::function<void()> task) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] {
            return stopping_ || queue_.size() < queue_capacity_;
        });
        if (stopping_) {
            throw StorageError("I/O executor is stopping");
        }
        queue_.push_back(std::move(task));
        lock.unlock();
        not_empty_.notify_one();
    }

private:
    void stop_and_join() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    void worker_loop() noexcept {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                not_empty_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                if (queue_.empty()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            not_full_.notify_one();
            try {
                task();
            } catch (...) {
                // PhysicalReader tasks capture and report their own errors.
                // Keep the executor alive if an unexpected task is submitted.
            }
        }
    }

    const std::size_t queue_capacity_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
};

class ScheduledReadJob {
public:
    using Task = std::function<void(std::uint64_t)>;

    ScheduledReadJob(
        Task task,
        const std::size_t task_count,
        std::shared_ptr<const PhysicalReaderTestHooks> test_hooks)
        : task_(std::move(task)),
          task_count_(task_count),
          test_hooks_(std::move(test_hooks)) {}

    void run(const std::uint64_t task_index) const {
        if (test_hooks_ != nullptr && test_hooks_->before_task) {
            test_hooks_->before_task(
                task_count_, static_cast<std::size_t>(task_index));
        }
        task_(task_index);
    }

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

    void finish(const ReadJobOutcome outcome) {
        std::exception_ptr result_error;
        switch (outcome) {
            case ReadJobOutcome::Succeeded:
                break;
            case ReadJobOutcome::Failed:
                result_error = first_task_error_;
                if (result_error == nullptr) {
                    result_error = make_storage_error(
                        "scheduled physical read failed");
                }
                break;
            case ReadJobOutcome::Cancelled:
                result_error = terminal_error_ != nullptr
                    ? terminal_error_
                    : make_storage_error(
                          "scheduled physical read was cancelled");
                break;
            case ReadJobOutcome::DeadlineExceeded:
                result_error = make_storage_error(
                    "scheduled physical read exceeded its deadline");
                break;
            case ReadJobOutcome::Shutdown:
                result_error = terminal_error_ != nullptr
                    ? terminal_error_
                    : make_storage_error(
                          "physical read coordinator is stopping");
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
    [[nodiscard]] static std::exception_ptr make_storage_error(
        const char* const message) {
        try {
            throw StorageError(message);
        } catch (...) {
            return std::current_exception();
        }
    }

    Task task_;
    std::size_t task_count_{0};
    std::shared_ptr<const PhysicalReaderTestHooks> test_hooks_;
    // Protected by ThreadPoolReadCoordinator::mutex_.
    std::exception_ptr first_task_error_;
    std::exception_ptr terminal_error_;
    std::mutex wait_mutex_;
    std::condition_variable finished_condition_;
    std::exception_ptr result_error_;
    bool finished_{false};
};

[[nodiscard]] ReadSchedulerOptions thread_pool_scheduler_options(
    const std::size_t worker_count) {
    return ReadSchedulerOptions{
        ReadSchedulerBudget{
            static_cast<std::uint64_t>(worker_count),
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max()},
        256ULL << 10,
        kMaximumReadSchedulerWeight};
}

[[nodiscard]] std::size_t admitted_job_capacity(
    const std::size_t worker_count,
    const std::size_t queue_capacity) noexcept {
    return queue_capacity
               > std::numeric_limits<std::size_t>::max() - worker_count
        ? std::numeric_limits<std::size_t>::max()
        : worker_count + queue_capacity;
}

class ThreadPoolReadCoordinator {
public:
    ThreadPoolReadCoordinator(
        const std::size_t worker_count,
        const std::size_t queue_capacity,
        std::shared_ptr<const PhysicalReaderTestHooks> test_hooks)
        : worker_count_(worker_count),
          max_admitted_jobs_(
              admitted_job_capacity(worker_count, queue_capacity)),
          scheduler_(thread_pool_scheduler_options(worker_count)),
          executor_(std::make_unique<BoundedExecutor>(
              worker_count, queue_capacity)),
          test_hooks_(std::move(test_hooks)) {}

    ~ThreadPoolReadCoordinator() { shutdown_and_join(); }

    ThreadPoolReadCoordinator(const ThreadPoolReadCoordinator&) = delete;
    ThreadPoolReadCoordinator& operator=(
        const ThreadPoolReadCoordinator&) = delete;

    void execute(
        std::vector<ReadWorkUnit> work,
        ScheduledReadJob::Task task) {
        const std::size_t task_count = work.size();
        auto job = std::make_shared<ScheduledReadJob>(
            std::move(task), task_count, test_hooks_);

        {
            std::unique_lock lock(mutex_);
            wait_for_admission_locked(lock, task_count);
            ReadJobId job_id = 0;
            try {
                job_id = scheduler_.submit(ReadJobSpec{
                    std::move(work), 1, std::nullopt});
            } catch (...) {
                admission_available_.notify_all();
                throw;
            }
            try {
                const auto [position, inserted] = jobs_.emplace(job_id, job);
                static_cast<void>(position);
                if (!inserted) {
                    throw StorageError(
                        "physical read coordinator reused a live job ID");
                }
            } catch (...) {
                static_cast<void>(scheduler_.cancel(job_id));
                deliver_completions_locked();
                admission_available_.notify_all();
                throw;
            }

            try {
                if (test_hooks_ != nullptr
                    && test_hooks_->after_job_submitted) {
                    test_hooks_->after_job_submitted(task_count);
                }
                pump_until_stable_locked();
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                job->remember_terminal_error(error);
                try {
                    static_cast<void>(scheduler_.cancel(job_id));
                    deliver_completions_locked();
                } catch (...) {
                    fail_coordinator_locked(std::current_exception());
                }
                // Do not rethrow here: pump may already have handed closures
                // to workers, and they capture the caller-owned read plan and
                // output. The terminal scheduler event is emitted only after
                // every such closure drains, so waiting below is the lifetime
                // barrier for those references.
            }
        }

        if (std::exception_ptr error = job->wait(); error != nullptr) {
            std::rethrow_exception(error);
        }
    }

private:
    void wait_for_admission_locked(
        std::unique_lock<std::mutex>& lock,
        const std::size_t task_count) {
        if (stopping_) {
            throw StorageError("physical read coordinator is stopping");
        }
        if (jobs_.size() < max_admitted_jobs_
            && admission_waiters_.empty()) {
            return;
        }
        if (next_admission_ticket_ == 0) {
            throw StorageError(
                "physical read admission ticket space is exhausted");
        }

        const std::uint64_t ticket = next_admission_ticket_++;
        admission_waiters_.push_back(ticket);
        try {
            if (test_hooks_ != nullptr
                && test_hooks_->before_admission_wait) {
                test_hooks_->before_admission_wait(task_count);
            }
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
            throw StorageError("physical read coordinator is stopping");
        }
        if (admission_waiters_.empty()
            || admission_waiters_.front() != ticket) {
            throw StorageError(
                "physical read admission FIFO lost its next ticket");
        }
        admission_waiters_.pop_front();
        admission_available_.notify_all();
    }

    void erase_admission_waiter(const std::uint64_t ticket) noexcept {
        const auto position = std::find(
            admission_waiters_.begin(), admission_waiters_.end(), ticket);
        if (position != admission_waiters_.end()) {
            admission_waiters_.erase(position);
        }
    }

    void pump_until_stable_locked() {
        while (!stopping_) {
            const ReadSchedulerSnapshot before = scheduler_.snapshot();
            if (before.runnable_jobs == 0
                || before.inflight_dispatches >= worker_count_) {
                return;
            }

            std::vector<ReadDispatch> dispatches =
                scheduler_.pump_until_blocked(0);
            if (dispatches.empty()) {
                return;
            }

            for (const ReadDispatch& dispatch : dispatches) {
                const auto position = jobs_.find(dispatch.job_id);
                if (position == jobs_.end()) {
                    throw StorageError(
                        "scheduled dispatch lost its physical read job");
                }
                const std::shared_ptr<ScheduledReadJob> job =
                    position->second;
                try {
                    executor_->submit([this, job, dispatch] {
                        std::exception_ptr error;
                        try {
                            job->run(dispatch.user_token);
                        } catch (...) {
                            error = std::current_exception();
                        }
                        task_completed(
                            dispatch.dispatch_id, job, std::move(error));
                    });
                } catch (...) {
                    const std::exception_ptr error =
                        std::current_exception();
                    job->remember_task_error(error);
                    scheduler_.complete(
                        dispatch.dispatch_id,
                        ReadDispatchStatus::Failed,
                        1);
                }
            }
            if (test_hooks_ != nullptr
                && test_hooks_->after_dispatch_batch_submitted) {
                test_hooks_->after_dispatch_batch_submitted();
            }
            deliver_completions_locked();
        }
    }

    void task_completed(
        const ReadDispatchId dispatch_id,
        const std::shared_ptr<ScheduledReadJob>& job,
        std::exception_ptr error) noexcept {
        std::unique_lock lock(mutex_);
        try {
            const ReadDispatchStatus status = error == nullptr
                ? ReadDispatchStatus::Succeeded
                : ReadDispatchStatus::Failed;
            if (error != nullptr) {
                job->remember_task_error(std::move(error));
            }
            scheduler_.complete(
                dispatch_id,
                status,
                status == ReadDispatchStatus::Failed ? 1 : 0);
            // Publish the just-finished synchronous job before potentially
            // queuing replacement work for peers.
            deliver_completions_locked();
            pump_until_stable_locked();
        } catch (...) {
            fail_coordinator_locked(std::current_exception());
        }
    }

    void fail_coordinator_locked(std::exception_ptr error) noexcept {
        const bool first_stop = !stopping_;
        stopping_ = true;
        admission_available_.notify_all();
        if (first_stop) {
            for (auto& [job_id, job] : jobs_) {
                static_cast<void>(job_id);
                job->remember_terminal_error(error);
            }
        }
        try {
            scheduler_.shutdown();
        } catch (...) {
        }
        try {
            deliver_completions_locked();
        } catch (...) {
            // Issued closures still own their job records. Later callbacks
            // retry scheduler completion delivery while stopping_ prevents
            // any new physical work from being dispatched.
        }
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
            const std::shared_ptr<ScheduledReadJob> job = position->second;
            jobs_.erase(position);
            released_admission = true;
            job->finish(completion.outcome);
        }
        if (released_admission) {
            admission_available_.notify_all();
        }
    }

    void shutdown_and_join() noexcept {
        try {
            std::unique_lock lock(mutex_);
            if (!stopping_) {
                stopping_ = true;
                admission_available_.notify_all();
                scheduler_.shutdown();
                deliver_completions_locked();
            }
        } catch (...) {
            // Continue into the mandatory executor join below. A callback may
            // still be the only owner of caller-referenced plan data.
        }

        // BoundedExecutor drains every already-issued closure. Their callbacks
        // return scheduler credits and publish terminal jobs; stopping_
        // prevents replacement dispatches during the join. This reset must run
        // even if scheduler shutdown delivery encountered an allocation error.
        executor_.reset();

        try {
            std::unique_lock lock(mutex_);
            deliver_completions_locked();
        } catch (...) {
            // Destructors must not throw.
        }
    }

    const std::size_t worker_count_;
    const std::size_t max_admitted_jobs_;
    std::mutex mutex_;
    std::condition_variable admission_available_;
    ReadScheduler scheduler_;
    std::map<ReadJobId, std::shared_ptr<ScheduledReadJob>> jobs_;
    std::deque<std::uint64_t> admission_waiters_;
    std::unique_ptr<BoundedExecutor> executor_;
    std::shared_ptr<const PhysicalReaderTestHooks> test_hooks_;
    std::uint64_t next_admission_ticket_{1};
    bool stopping_{false};
};

[[nodiscard]] std::filesystem::path descriptor_description(const int fd) {
    return std::filesystem::path("<file-descriptor-" + std::to_string(fd) + ">");
}

}  // namespace

class PhysicalReader::Impl {
public:
    struct Request {
        int fd{-1};
        std::uint64_t absolute_offset{0};
        std::uint64_t output_index{0};
    };

    struct PlannedRange {
        std::size_t request_begin{0};
        std::size_t request_end{0};
        int fd{-1};
        std::uint64_t first_offset{0};
        std::uint64_t end_offset{0};
    };

    explicit Impl(
        const ReaderOptions options,
        std::shared_ptr<const PhysicalReaderTestHooks> test_hooks = nullptr)
        : options_(options) {
        validate_reader_options(options_);
        try {
            if (options_.io_backend == IoBackend::IoUring) {
                IoUringLaneFactory lane_factory;
                if (test_hooks != nullptr) {
                    lane_factory = test_hooks->io_uring_lane_factory;
                }
                io_uring_coordinator_ =
                    std::make_unique<IoUringReadCoordinator>(
                        options_.io_uring_workers,
                        options_.io_uring_queue_depth,
                        options_.max_read_range_bytes,
                        std::move(lane_factory));
            } else if (options_.io_workers != 0) {
                thread_pool_coordinator_ =
                    std::make_unique<ThreadPoolReadCoordinator>(
                        options_.io_workers,
                        effective_queue_capacity(options_),
                        std::move(test_hooks));
            }
        } catch (const std::exception& error) {
            throw ReaderBackendError(
                "reader backend initialization failed: "
                + std::string(error.what()));
        }
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    void execute(
        const std::uint64_t record_size,
        const std::span<const PhysicalReadRequest> input,
        const std::span<std::byte> output) const {
        execute_impl<false>(record_size, input, output, nullptr);
    }

    [[nodiscard]] PhysicalReadStats execute_profiled(
        const std::uint64_t record_size,
        const std::span<const PhysicalReadRequest> input,
        const std::span<std::byte> output) const {
        PhysicalReadStats stats;
        execute_impl<true>(record_size, input, output, &stats);
        return stats;
    }

private:
    template <bool Profiled>
    void execute_impl(
        const std::uint64_t record_size,
        const std::span<const PhysicalReadRequest> input,
        const std::span<std::byte> output,
        PhysicalReadStats* const stats) const {
        using Clock = std::chrono::steady_clock;
        Clock::time_point plan_started;
        if constexpr (Profiled) {
            plan_started = Clock::now();
        }
        if (!owned_by_current_process()) {
            throw StorageError(
                "physical reader was inherited across fork; reopen its "
                "collection in the child process");
        }
        if (record_size == 0) {
            throw InvalidArgumentError("record_size must be greater than zero");
        }
        if (record_size > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("record_size does not fit in address space");
        }
        if (record_size > options_.max_read_range_bytes) {
            throw InvalidArgumentError(
                "max_read_range_bytes must fit at least one record");
        }
        const std::uint64_t expected_bytes = checked_multiply(
            input.size(), record_size, "physical read output");
        if (output.size() != expected_bytes) {
            throw InvalidArgumentError(
                "output byte count does not equal requests.size() * record_size");
        }
        if (input.empty()) {
            if constexpr (Profiled) {
                stats->requested_bytes = expected_bytes;
                stats->plan_seconds = std::chrono::duration<double>(
                    Clock::now() - plan_started).count();
            }
            return;
        }

        std::vector<std::uint8_t> seen_output_indices(input.size(), 0);
        std::vector<Request> requests;
        requests.reserve(input.size());
        for (const PhysicalReadRequest& request : input) {
            if (request.fd < 0) {
                throw InvalidArgumentError(
                    "physical read request contains an invalid file descriptor");
            }
            if (request.absolute_offset
                > std::numeric_limits<std::uint64_t>::max() - record_size) {
                throw InvalidArgumentError(
                    "physical read request offset overflows uint64");
            }
            if (request.output_index >= input.size()) {
                throw InvalidArgumentError(
                    "physical read output index is out of range");
            }
            const auto output_index = static_cast<std::size_t>(
                request.output_index);
            if (seen_output_indices[output_index] != 0) {
                throw InvalidArgumentError(
                    "physical read output indices must be unique");
            }
            seen_output_indices[output_index] = 1;
            requests.push_back(Request{
                request.fd,
                request.absolute_offset,
                request.output_index});
        }

        if constexpr (Profiled) {
            stats->requested_bytes = expected_bytes;
            for (std::size_t index = 1; index != input.size(); ++index) {
                const PhysicalReadRequest& previous = input[index - 1];
                const PhysicalReadRequest& current = input[index];
                const bool contiguous =
                    current.fd == previous.fd
                    && previous.absolute_offset
                           <= std::numeric_limits<std::uint64_t>::max()
                                  - record_size
                    && current.absolute_offset
                           == previous.absolute_offset + record_size;
                if (!contiguous) {
                    ++stats->input_discontinuities;
                }
                if (current.fd < previous.fd
                    || (current.fd == previous.fd
                        && current.absolute_offset
                               < previous.absolute_offset)) {
                    ++stats->input_inversions;
                }
            }
        }

        Clock::time_point sort_started;
        if constexpr (Profiled) {
            sort_started = Clock::now();
        }
        std::sort(
            requests.begin(), requests.end(),
            [](const Request& left, const Request& right) {
                if (left.fd != right.fd) {
                    return left.fd < right.fd;
                }
                if (left.absolute_offset != right.absolute_offset) {
                    return left.absolute_offset < right.absolute_offset;
                }
                return left.output_index < right.output_index;
            });
        if constexpr (Profiled) {
            stats->sort_seconds = std::chrono::duration<double>(
                Clock::now() - sort_started).count();
        }

        std::vector<PlannedRange> ranges;
        ranges.reserve(requests.size());
        std::size_t begin = 0;
        while (begin != requests.size()) {
            const Request& first = requests[begin];
            const std::uint64_t first_offset = first.absolute_offset;
            std::uint64_t end_offset = first_offset + record_size;
            std::size_t end = begin + 1;
            while (end != requests.size() && requests[end].fd == first.fd) {
                const std::uint64_t next_offset =
                    requests[end].absolute_offset;
                const std::uint64_t next_end = next_offset + record_size;
                const std::uint64_t candidate_end =
                    std::max(end_offset, next_end);
                const std::uint64_t candidate_bytes =
                    candidate_end - first_offset;
                const std::uint64_t gap = next_offset > end_offset
                    ? next_offset - end_offset
                    : 0;
                if (gap > options_.max_gap_bytes
                    || candidate_bytes > options_.max_read_range_bytes) {
                    break;
                }
                end_offset = candidate_end;
                ++end;
            }
            ranges.push_back(PlannedRange{
                begin, end, first.fd, first_offset, end_offset});
            begin = end;
        }

        if constexpr (Profiled) {
            std::size_t task_count = 0;
            if (io_uring_coordinator_ != nullptr) {
                const std::size_t max_ranges_per_task =
                    options_.io_uring_queue_depth
                        / options_.io_uring_workers
                    + (options_.io_uring_queue_depth
                           % options_.io_uring_workers
                       != 0
                       ? 1
                       : 0);
                task_count = plan_range_tasks(
                    ranges.size(),
                    options_.io_uring_workers,
                    max_ranges_per_task).task_count;
            } else if (ranges.size() == 1) {
                task_count = 1;
            } else if (thread_pool_coordinator_ == nullptr) {
                task_count = ranges.size();
            } else {
                task_count = plan_range_tasks(
                    ranges.size(), options_.io_workers,
                    options_.io_ranges_per_task).task_count;
            }
            stats->request_count = input.size();
            stats->range_count = ranges.size();
            stats->task_count = task_count;
            int previous_file = -1;
            int useful_file = -1;
            std::uint64_t useful_end = 0;
            for (const Request& request : requests) {
                if (request.fd != previous_file) {
                    ++stats->unique_file_count;
                    previous_file = request.fd;
                }
                const std::uint64_t request_end =
                    request.absolute_offset + record_size;
                if (request.fd != useful_file
                    || request.absolute_offset >= useful_end) {
                    stats->useful_read_bytes += record_size;
                    useful_file = request.fd;
                    useful_end = request_end;
                } else if (request_end > useful_end) {
                    stats->useful_read_bytes += request_end - useful_end;
                    useful_end = request_end;
                }
            }
            for (const PlannedRange& range : ranges) {
                const std::uint64_t range_bytes =
                    range.end_offset - range.first_offset;
                if (range_bytes
                    > std::numeric_limits<std::uint64_t>::max()
                          - stats->planned_read_bytes) {
                    stats->planned_read_bytes =
                        std::numeric_limits<std::uint64_t>::max();
                    break;
                }
                stats->planned_read_bytes += range_bytes;
                stats->maximum_range_bytes =
                    std::max(stats->maximum_range_bytes, range_bytes);
            }
            if (!ranges.empty()) {
                stats->mean_range_bytes =
                    stats->planned_read_bytes / ranges.size();
            }
            stats->gap_bytes =
                stats->planned_read_bytes > stats->useful_read_bytes
                    ? stats->planned_read_bytes
                          - stats->useful_read_bytes
                    : 0;
            // PhysicalReader issues exact byte ranges. Consequently all
            // overread in this layer is merge-gap amplification; filesystem
            // page-cache/device amplification is modelled separately.
            stats->overread_bytes = stats->gap_bytes;
            const auto measure_units =
                [&ranges](const std::uint64_t unit_bytes) {
                    std::pair<std::uint64_t, std::uint64_t> result{0, 0};
                    int current_file = -1;
                    std::uint64_t current_end = 0;
                    for (const PlannedRange& range : ranges) {
                        const std::uint64_t first =
                            range.first_offset / unit_bytes;
                        const std::uint64_t end =
                            (range.end_offset - 1) / unit_bytes + 1;
                        if (range.fd != current_file
                            || first > current_end) {
                            ++result.second;
                            result.first += end - first;
                            current_file = range.fd;
                            current_end = end;
                        } else if (end > current_end) {
                            result.first += end - current_end;
                            current_end = end;
                        }
                    }
                    return result;
                };
            constexpr std::uint64_t page_bytes = 4ULL << 10;
            constexpr std::uint64_t chunk_bytes = 1ULL << 20;
            const auto [pages, page_runs] = measure_units(page_bytes);
            const auto [chunks, chunk_runs] = measure_units(chunk_bytes);
            stats->pages_touched = pages;
            stats->page_runs = page_runs;
            stats->chunks_touched = chunks;
            stats->chunk_runs = chunk_runs;
            stats->plan_seconds = std::chrono::duration<double>(
                Clock::now() - plan_started).count();
        }

        Clock::time_point backend_started;
        if constexpr (Profiled) {
            backend_started = Clock::now();
        }
        const auto finish_backend = [&] {
            if constexpr (Profiled) {
                stats->backend_seconds = std::chrono::duration<double>(
                    Clock::now() - backend_started).count();
                stats->completed_read_bytes = stats->planned_read_bytes;
            }
        };

        if (thread_pool_coordinator_ != nullptr) {
            read_ranges_thread_pool(
                requests, ranges, record_size, output);
            finish_backend();
            return;
        }
        if (io_uring_coordinator_ != nullptr) {
            read_ranges_io_uring(requests, ranges, record_size, output);
            finish_backend();
            return;
        }
        if (ranges.size() == 1) {
            read_range(requests, ranges.front(), record_size, output);
            finish_backend();
            return;
        }
        for (const PlannedRange& range : ranges) {
            read_range(requests, range, record_size, output);
        }
        finish_backend();
    }

    void read_ranges_thread_pool(
        const std::vector<Request>& requests,
        const std::vector<PlannedRange>& ranges,
        const std::uint64_t record_size,
        const std::span<std::byte> output) const {
        const RangeTaskPartition partition = plan_range_tasks(
            ranges.size(), options_.io_workers, options_.io_ranges_per_task);
        std::vector<ReadWorkUnit> work;
        work.reserve(partition.task_count);
        for (std::size_t task_index = 0;
             task_index != partition.task_count; ++task_index) {
            const RangeTaskBounds bounds = partition.bounds(task_index);
            const std::size_t range_count = bounds.end - bounds.begin;
            if (range_count > std::numeric_limits<std::uint64_t>::max()) {
                throw InvalidArgumentError(
                    "scheduled range-task count does not fit uint64");
            }
            std::uint64_t physical_bytes = 0;
            for (std::size_t range_index = bounds.begin;
                 range_index != bounds.end; ++range_index) {
                const PlannedRange& range = ranges[range_index];
                const std::uint64_t range_bytes =
                    range.end_offset - range.first_offset;
                if (range_bytes
                    > std::numeric_limits<std::uint64_t>::max()
                          - physical_bytes) {
                    throw InvalidArgumentError(
                        "scheduled range-task byte count overflows uint64");
                }
                physical_bytes += range_bytes;
            }
            work.push_back(ReadWorkUnit{
                static_cast<std::uint64_t>(task_index),
                static_cast<std::uint64_t>(range_count),
                physical_bytes});
        }

        thread_pool_coordinator_->execute(
            std::move(work),
            [&requests, &ranges, partition, record_size, output](
                const std::uint64_t task_index) {
                if (task_index >= partition.task_count) {
                    throw StorageError(
                        "scheduled physical task index is out of range");
                }
                const RangeTaskBounds bounds = partition.bounds(
                    static_cast<std::size_t>(task_index));
                for (std::size_t range_index = bounds.begin;
                     range_index != bounds.end; ++range_index) {
                    read_range(
                        requests, ranges[range_index], record_size, output);
                }
            });
    }

    static void read_range(
        const std::vector<Request>& requests,
        const PlannedRange& range,
        const std::uint64_t record_size,
        const std::span<std::byte> output) {
        if (range.request_end == range.request_begin + 1) {
            const Request& request = requests[range.request_begin];
            const std::uint64_t destination_offset =
                request.output_index * record_size;
            pread_exact(
                request.fd,
                output.data() + static_cast<std::size_t>(destination_offset),
                static_cast<std::size_t>(record_size),
                request.absolute_offset,
                descriptor_description(request.fd));
            return;
        }

        const std::uint64_t range_bytes_u64 =
            range.end_offset - range.first_offset;
        if (range_bytes_u64 > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError(
                "planned physical read range does not fit in address space");
        }
        std::vector<std::byte> buffer(
            static_cast<std::size_t>(range_bytes_u64));
        pread_exact(
            range.fd,
            buffer.data(),
            buffer.size(),
            range.first_offset,
            descriptor_description(range.fd));

        for (std::size_t request_index = range.request_begin;
             request_index != range.request_end; ++request_index) {
            const Request& request = requests[request_index];
            const std::uint64_t source_offset =
                request.absolute_offset - range.first_offset;
            const std::uint64_t destination_offset =
                request.output_index * record_size;
            std::memcpy(
                output.data() + static_cast<std::size_t>(destination_offset),
                buffer.data() + static_cast<std::size_t>(source_offset),
                static_cast<std::size_t>(record_size));
        }
    }

    void read_ranges_io_uring(
        const std::vector<Request>& requests,
        const std::vector<PlannedRange>& ranges,
        const std::uint64_t record_size,
        const std::span<std::byte> output) const {
        const std::size_t max_ranges_per_task =
            options_.io_uring_queue_depth / options_.io_uring_workers
            + (options_.io_uring_queue_depth % options_.io_uring_workers != 0
                   ? 1
                   : 0);
        const RangeTaskPartition partition = plan_range_tasks(
            ranges.size(),
            options_.io_uring_workers,
            max_ranges_per_task);
        std::vector<ReadWorkUnit> work;
        work.reserve(partition.task_count);
        for (std::size_t task_index = 0;
             task_index != partition.task_count; ++task_index) {
            const RangeTaskBounds bounds = partition.bounds(task_index);
            std::uint64_t physical_bytes = 0;
            for (std::size_t range_index = bounds.begin;
                 range_index != bounds.end; ++range_index) {
                const PlannedRange& range = ranges[range_index];
                const std::uint64_t range_bytes =
                    range.end_offset - range.first_offset;
                if (range_bytes
                    > std::numeric_limits<std::uint64_t>::max()
                          - physical_bytes) {
                    throw InvalidArgumentError(
                        "scheduled io_uring task byte count overflows uint64");
                }
                physical_bytes += range_bytes;
            }
            work.push_back(ReadWorkUnit{
                static_cast<std::uint64_t>(task_index),
                static_cast<std::uint64_t>(bounds.end - bounds.begin),
                physical_bytes});
        }

        io_uring_coordinator_->execute(
            std::move(work),
            [&requests, &ranges, partition, record_size, output](
                const std::uint64_t task_index,
                IoUringLane& lane) {
                if (task_index >= partition.task_count) {
                    throw StorageError(
                        "scheduled io_uring task index is out of range");
                }
                const RangeTaskBounds bounds = partition.bounds(
                    static_cast<std::size_t>(task_index));

                struct PreparedRange {
                    std::size_t range_index{0};
                    std::vector<std::byte> buffer;
                };
                std::vector<PreparedRange> prepared;
                prepared.reserve(bounds.end - bounds.begin);
                std::vector<IoUringReadOperation> operations;
                operations.reserve(bounds.end - bounds.begin);
                for (std::size_t range_index = bounds.begin;
                     range_index != bounds.end; ++range_index) {
                    const PlannedRange& range = ranges[range_index];
                    void* destination = nullptr;
                    std::size_t size = 0;
                    std::uint64_t offset = range.first_offset;
                    if (range.request_end == range.request_begin + 1) {
                        const Request& request =
                            requests[range.request_begin];
                        const std::uint64_t destination_offset =
                            request.output_index * record_size;
                        destination = output.data()
                            + static_cast<std::size_t>(destination_offset);
                        size = static_cast<std::size_t>(record_size);
                        offset = request.absolute_offset;
                    } else {
                        const std::uint64_t range_bytes =
                            range.end_offset - range.first_offset;
                        if (range_bytes
                            > std::numeric_limits<std::size_t>::max()) {
                            throw InvalidArgumentError(
                                "planned io_uring range does not fit in "
                                "address space");
                        }
                        prepared.push_back(PreparedRange{
                            range_index,
                            std::vector<std::byte>(
                                static_cast<std::size_t>(range_bytes))});
                        destination = prepared.back().buffer.data();
                        size = prepared.back().buffer.size();
                    }
                    operations.push_back(IoUringReadOperation{
                        range.fd,
                        destination,
                        size,
                        offset,
                        static_cast<std::uint64_t>(range_index)});
                }

                lane.read_exact(operations);

                for (const PreparedRange& item : prepared) {
                    const PlannedRange& range = ranges[item.range_index];
                    for (std::size_t request_index = range.request_begin;
                         request_index != range.request_end;
                         ++request_index) {
                        const Request& request = requests[request_index];
                        const std::uint64_t source_offset =
                            request.absolute_offset - range.first_offset;
                        const std::uint64_t destination_offset =
                            request.output_index * record_size;
                        std::memcpy(
                            output.data() + static_cast<std::size_t>(
                                destination_offset),
                            item.buffer.data() + static_cast<std::size_t>(
                                source_offset),
                            static_cast<std::size_t>(record_size));
                    }
                }
            });
    }

    ReaderOptions options_;
    std::unique_ptr<ThreadPoolReadCoordinator> thread_pool_coordinator_;
    std::unique_ptr<IoUringReadCoordinator> io_uring_coordinator_;
    const pid_t owner_pid_{::getpid()};
};

PhysicalReader::PhysicalReader(const ReaderOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

PhysicalReader::PhysicalReader(
    const ReaderOptions options,
    std::shared_ptr<const PhysicalReaderTestHooks> test_hooks)
    : impl_(std::make_unique<Impl>(options, std::move(test_hooks))) {}

PhysicalReader::~PhysicalReader() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        // std::thread objects copied by fork still look joinable in the child,
        // although their threads no longer exist. Do not run executor/ring
        // teardown there; process exit reclaims this process-local copy.
        static_cast<void>(impl_.release());
    }
}

void PhysicalReader::execute(
    const std::uint64_t record_size,
    const std::span<const PhysicalReadRequest> requests,
    const std::span<std::byte> output) const {
    impl_->execute(record_size, requests, output);
}

PhysicalReadStats PhysicalReader::execute_profiled(
    const std::uint64_t record_size,
    const std::span<const PhysicalReadRequest> requests,
    const std::span<std::byte> output) const {
    return impl_->execute_profiled(record_size, requests, output);
}

void PhysicalReader::abandon_after_fork() noexcept {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        static_cast<void>(impl_.release());
    }
}

}  // namespace throwdb::detail
