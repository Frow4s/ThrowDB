#include "io_uring_pool.hpp"

#include "throwdb/errors.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace throwdb::detail {

std::vector<std::uint32_t> split_io_uring_queue_depth(
    const std::uint32_t worker_count,
    const std::uint32_t total_queue_depth) {
    if (worker_count == 0 || total_queue_depth < worker_count) {
        throw InvalidArgumentError(
            "io_uring_queue_depth must be at least io_uring_workers");
    }
    std::vector<std::uint32_t> result(worker_count);
    const std::uint32_t base_depth = total_queue_depth / worker_count;
    const std::uint32_t remainder = total_queue_depth % worker_count;
    for (std::uint32_t index = 0; index != worker_count; ++index) {
        result[index] = base_depth + (index < remainder ? 1U : 0U);
    }
    return result;
}

std::vector<std::size_t> plan_io_uring_lanes(
    const std::span<const IoUringReadOperation> operations,
    const std::span<const std::uint32_t> lane_queue_depths) {
    if (lane_queue_depths.empty()
        || std::any_of(
            lane_queue_depths.begin(), lane_queue_depths.end(),
            [](const std::uint32_t depth) { return depth == 0; })) {
        throw InvalidArgumentError(
            "io_uring lane queue depths must be positive");
    }

    std::vector<long double> normalized_load(lane_queue_depths.size(), 0);
    std::vector<std::size_t> result;
    result.reserve(operations.size());
    for (const IoUringReadOperation& operation : operations) {
        const auto least_loaded = std::min_element(
            normalized_load.begin(), normalized_load.end());
        const std::size_t lane = static_cast<std::size_t>(
            std::distance(normalized_load.begin(), least_loaded));
        result.push_back(lane);
        const std::uint64_t cost = std::max<std::uint64_t>(
            static_cast<std::uint64_t>(operation.size), 4096);
        *least_loaded += static_cast<long double>(cost)
                         / lane_queue_depths[lane];
    }
    return result;
}

class IoUringPool::Impl {
private:
    class FailureState {
    public:
        void record(std::exception_ptr error) noexcept {
            std::lock_guard lock(mutex_);
            if (error_ == nullptr) {
                error_ = std::move(error);
            }
        }

        [[nodiscard]] std::exception_ptr get() const noexcept {
            std::lock_guard lock(mutex_);
            return error_;
        }

    private:
        mutable std::mutex mutex_;
        std::exception_ptr error_;
    };

    class BatchState {
    public:
        explicit BatchState(const std::size_t task_count)
            : remaining_(task_count) {}

        void complete(std::exception_ptr error = nullptr) noexcept {
            std::lock_guard lock(mutex_);
            if (error != nullptr && error_ == nullptr) {
                error_ = std::move(error);
            }
            if (remaining_ != 0) {
                --remaining_;
            }
            if (remaining_ == 0) {
                completed_.notify_all();
            }
        }

        [[nodiscard]] std::exception_ptr wait() {
            std::unique_lock lock(mutex_);
            completed_.wait(lock, [this] { return remaining_ == 0; });
            return error_;
        }

    private:
        std::mutex mutex_;
        std::condition_variable completed_;
        std::size_t remaining_;
        std::exception_ptr error_;
    };

    struct WorkItem {
        std::vector<IoUringReadOperation> operations;
        std::shared_ptr<BatchState> batch;
    };

    class Worker {
    public:
        Worker(const std::uint32_t queue_depth,
               std::shared_ptr<FailureState> failure)
            : engine_(queue_depth), failure_(std::move(failure)) {}

        ~Worker() {
            request_stop();
            join();
        }

        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;

        void start() {
            thread_ = std::thread([this] { run(); });
        }

        void enqueue(std::shared_ptr<WorkItem> item) {
            {
                std::lock_guard lock(mutex_);
                if (stopping_) {
                    throw StorageError("io_uring worker is stopping");
                }
                queue_.push_back(std::move(item));
            }
            ready_.notify_one();
        }

        void request_stop() noexcept {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
        }

        void join() noexcept {
            if (thread_.joinable()) {
                thread_.join();
            }
        }

    private:
        void run() noexcept {
            while (true) {
                std::shared_ptr<WorkItem> item;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [this] {
                        return stopping_ || !queue_.empty();
                    });
                    if (queue_.empty()) {
                        return;
                    }
                    item = std::move(queue_.front());
                    queue_.pop_front();
                }

                std::exception_ptr error;
                if (std::exception_ptr fatal = failure_->get();
                    fatal != nullptr) {
                    error = std::move(fatal);
                } else {
                    try {
                        engine_.read_exact(item->operations);
                    } catch (...) {
                        error = std::current_exception();
                        if (!engine_.usable()) {
                            failure_->record(error);
                        }
                    }
                }
                item->batch->complete(std::move(error));
            }
        }

        IoUringEngine engine_;
        std::shared_ptr<FailureState> failure_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::deque<std::shared_ptr<WorkItem>> queue_;
        std::thread thread_;
        bool stopping_{false};
    };

    class Admission {
    public:
        explicit Admission(Impl& owner) : owner_(&owner) {
            owner_->acquire_admission();
        }

        ~Admission() {
            if (owner_ != nullptr) {
                owner_->release_admission();
            }
        }

        Admission(const Admission&) = delete;
        Admission& operator=(const Admission&) = delete;

    private:
        Impl* owner_;
    };

public:
    Impl(const std::uint32_t worker_count,
         const std::uint32_t total_queue_depth)
        : worker_count_(worker_count),
          total_queue_depth_(total_queue_depth),
          admission_capacity_(static_cast<std::size_t>(worker_count) * 2) {
        if (worker_count < 2) {
            throw InvalidArgumentError(
                "io_uring pool requires at least two workers");
        }
        queue_depths_ = split_io_uring_queue_depth(
            worker_count, total_queue_depth);

        workers_.reserve(worker_count);
        try {
            for (std::uint32_t index = 0; index != worker_count; ++index) {
                workers_.push_back(std::make_unique<Worker>(
                    queue_depths_[index], failure_));
            }
            for (const std::unique_ptr<Worker>& worker : workers_) {
                worker->start();
            }
        } catch (...) {
            stop_workers();
            throw;
        }
    }

    ~Impl() {
        {
            std::unique_lock lock(admission_mutex_);
            stopping_ = true;
            admission_changed_.notify_all();
            admission_changed_.wait(lock, [this] {
                return active_batches_ == 0;
            });
        }
        stop_workers();
    }

    [[nodiscard]] std::uint32_t worker_count() const noexcept {
        return worker_count_;
    }

    [[nodiscard]] std::uint32_t total_queue_depth() const noexcept {
        return total_queue_depth_;
    }

    void read_exact(const std::span<const IoUringReadOperation> operations) {
        if (operations.empty()) {
            return;
        }
        if (std::exception_ptr fatal = failure_->get(); fatal != nullptr) {
            std::rethrow_exception(fatal);
        }

        const std::size_t task_count =
            std::min<std::size_t>(workers_.size(), operations.size());
        auto batch = std::make_shared<BatchState>(task_count);
        std::vector<std::shared_ptr<WorkItem>> items;
        items.reserve(task_count);
        for (std::size_t task_index = 0; task_index != task_count;
             ++task_index) {
            auto item = std::make_shared<WorkItem>();
            item->operations.reserve(
                (operations.size() + task_count - 1) / task_count);
            item->batch = batch;
            items.push_back(std::move(item));
        }

        // The planner preserves original physical order inside each lane and
        // normalizes accumulated cost by that lane's share of total QD.
        const std::vector<std::size_t> lanes = plan_io_uring_lanes(
            operations,
            std::span<const std::uint32_t>(queue_depths_.data(), task_count));
        for (std::size_t operation_index = 0;
             operation_index != operations.size(); ++operation_index) {
            items[lanes[operation_index]]->operations.push_back(
                operations[operation_index]);
        }

        Admission admission(*this);
        if (std::exception_ptr fatal = failure_->get(); fatal != nullptr) {
            std::rethrow_exception(fatal);
        }
        std::size_t submitted = 0;
        {
            // Keep each worker's FIFO in the same global batch order, even
            // when multiple callers submit concurrently.
            std::lock_guard submission_lock(submission_mutex_);
            try {
                for (; submitted != task_count; ++submitted) {
                    workers_[submitted]->enqueue(items[submitted]);
                }
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                for (; submitted != task_count; ++submitted) {
                    batch->complete(error);
                }
            }
        }

        if (std::exception_ptr error = batch->wait(); error != nullptr) {
            std::rethrow_exception(error);
        }
    }

private:
    void acquire_admission() {
        std::unique_lock lock(admission_mutex_);
        const std::uint64_t ticket = next_admission_ticket_++;
        admission_changed_.wait(lock, [this, ticket] {
            return stopping_
                   || (ticket == admitting_ticket_
                       && active_batches_ < admission_capacity_);
        });
        if (stopping_) {
            throw StorageError("io_uring pool is stopping");
        }
        ++admitting_ticket_;
        ++active_batches_;
        lock.unlock();
        // Capacity may still be available for the next ticket. Waking all is
        // safe because only the unique admitting_ticket_ can pass the guard.
        admission_changed_.notify_all();
    }

    void release_admission() noexcept {
        {
            std::lock_guard lock(admission_mutex_);
            if (active_batches_ != 0) {
                --active_batches_;
            }
        }
        admission_changed_.notify_all();
    }

    void stop_workers() noexcept {
        for (const std::unique_ptr<Worker>& worker : workers_) {
            worker->request_stop();
        }
        for (const std::unique_ptr<Worker>& worker : workers_) {
            worker->join();
        }
    }

    const std::uint32_t worker_count_;
    const std::uint32_t total_queue_depth_;
    const std::size_t admission_capacity_;
    std::vector<std::uint32_t> queue_depths_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::shared_ptr<FailureState> failure_ =
        std::make_shared<FailureState>();
    std::mutex submission_mutex_;
    std::mutex admission_mutex_;
    std::condition_variable admission_changed_;
    std::size_t active_batches_{0};
    std::uint64_t next_admission_ticket_{0};
    std::uint64_t admitting_ticket_{0};
    bool stopping_{false};
};

IoUringPool::IoUringPool(const std::uint32_t worker_count,
                         const std::uint32_t total_queue_depth)
    : impl_(std::make_unique<Impl>(worker_count, total_queue_depth)) {}

IoUringPool::~IoUringPool() = default;

void IoUringPool::read_exact(
    const std::span<const IoUringReadOperation> operations) {
    impl_->read_exact(operations);
}

std::uint32_t IoUringPool::worker_count() const noexcept {
    return impl_->worker_count();
}

std::uint32_t IoUringPool::total_queue_depth() const noexcept {
    return impl_->total_queue_depth();
}

}  // namespace throwdb::detail
