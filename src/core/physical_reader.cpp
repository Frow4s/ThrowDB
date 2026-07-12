#include "physical_reader.hpp"

#include "internal.hpp"
#include "io_uring_engine.hpp"
#include "io_uring_pool.hpp"
#include "task_partition.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
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

class ReadBatchState {
public:
    explicit ReadBatchState(const std::size_t task_count)
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
            complete_.notify_all();
        }
    }

    [[nodiscard]] std::exception_ptr wait() {
        std::unique_lock lock(mutex_);
        complete_.wait(lock, [this] { return remaining_ == 0; });
        return error_;
    }

private:
    std::mutex mutex_;
    std::condition_variable complete_;
    std::size_t remaining_;
    std::exception_ptr error_;
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

    explicit Impl(const ReaderOptions options) : options_(options) {
        validate_reader_options(options_);
        if (options_.io_backend == IoBackend::IoUring) {
            if (options_.io_uring_workers == 1) {
                io_uring_ = std::make_unique<IoUringEngine>(
                    options_.io_uring_queue_depth);
            } else {
                io_uring_pool_ = std::make_unique<IoUringPool>(
                    options_.io_uring_workers,
                    options_.io_uring_queue_depth);
            }
        } else if (options_.io_workers != 0) {
            executor_ = std::make_unique<BoundedExecutor>(
                options_.io_workers, effective_queue_capacity(options_));
        }
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    void execute(
        const std::uint64_t record_size,
        const std::span<const PhysicalReadRequest> input,
        const std::span<std::byte> output) const {
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

        if (ranges.size() == 1) {
            read_range(requests, ranges.front(), record_size, output);
            return;
        }
        if (io_uring_ != nullptr || io_uring_pool_ != nullptr) {
            read_ranges_io_uring(requests, ranges, record_size, output);
            return;
        }
        if (executor_ == nullptr) {
            for (const PlannedRange& range : ranges) {
                read_range(requests, range, record_size, output);
            }
            return;
        }

        const RangeTaskPartition partition = plan_range_tasks(
            ranges.size(), options_.io_workers, options_.io_ranges_per_task);
        auto state = std::make_shared<ReadBatchState>(partition.task_count);
        std::size_t submitted = 0;
        try {
            for (; submitted != partition.task_count; ++submitted) {
                const RangeTaskBounds task_bounds = partition.bounds(submitted);
                executor_->submit(
                    [&requests, &ranges, task_bounds, record_size, output,
                     state] {
                        std::exception_ptr error;
                        for (std::size_t range_index = task_bounds.begin;
                             range_index != task_bounds.end; ++range_index) {
                            try {
                                read_range(
                                    requests, ranges[range_index], record_size,
                                    output);
                            } catch (...) {
                                if (error == nullptr) {
                                    error = std::current_exception();
                                }
                            }
                        }
                        state->complete(std::move(error));
                    });
            }
        } catch (...) {
            state->complete(std::current_exception());
            ++submitted;
            while (submitted != partition.task_count) {
                state->complete();
                ++submitted;
            }
        }
        if (std::exception_ptr error = state->wait(); error != nullptr) {
            std::rethrow_exception(error);
        }
    }

private:
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
        struct PreparedRange {
            std::vector<std::byte> buffer;
        };

        std::vector<PreparedRange> prepared(ranges.size());
        std::vector<IoUringReadOperation> operations;
        operations.reserve(ranges.size());
        for (std::size_t range_index = 0; range_index != ranges.size();
             ++range_index) {
            const PlannedRange& range = ranges[range_index];
            void* destination = nullptr;
            std::size_t size = 0;
            std::uint64_t offset = range.first_offset;
            if (range.request_end == range.request_begin + 1) {
                const Request& request = requests[range.request_begin];
                const std::uint64_t destination_offset =
                    request.output_index * record_size;
                destination = output.data()
                              + static_cast<std::size_t>(destination_offset);
                size = static_cast<std::size_t>(record_size);
                offset = request.absolute_offset;
            } else {
                const std::uint64_t range_bytes =
                    range.end_offset - range.first_offset;
                if (range_bytes > std::numeric_limits<std::size_t>::max()) {
                    throw InvalidArgumentError(
                        "planned io_uring range does not fit in address space");
                }
                prepared[range_index].buffer.resize(
                    static_cast<std::size_t>(range_bytes));
                destination = prepared[range_index].buffer.data();
                size = prepared[range_index].buffer.size();
            }
            operations.push_back(IoUringReadOperation{
                range.fd, destination, size, offset, range_index});
        }

        if (io_uring_pool_ != nullptr) {
            io_uring_pool_->read_exact(operations);
        } else {
            io_uring_->read_exact(operations);
        }

        for (std::size_t range_index = 0; range_index != ranges.size();
             ++range_index) {
            const PlannedRange& range = ranges[range_index];
            if (range.request_end == range.request_begin + 1) {
                continue;
            }
            const std::vector<std::byte>& buffer =
                prepared[range_index].buffer;
            for (std::size_t request_index = range.request_begin;
                 request_index != range.request_end; ++request_index) {
                const Request& request = requests[request_index];
                const std::uint64_t source_offset =
                    request.absolute_offset - range.first_offset;
                const std::uint64_t destination_offset =
                    request.output_index * record_size;
                std::memcpy(
                    output.data()
                        + static_cast<std::size_t>(destination_offset),
                    buffer.data() + static_cast<std::size_t>(source_offset),
                    static_cast<std::size_t>(record_size));
            }
        }
    }

    ReaderOptions options_;
    std::unique_ptr<BoundedExecutor> executor_;
    std::unique_ptr<IoUringEngine> io_uring_;
    std::unique_ptr<IoUringPool> io_uring_pool_;
    const pid_t owner_pid_{::getpid()};
};

PhysicalReader::PhysicalReader(const ReaderOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

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

void PhysicalReader::abandon_after_fork() noexcept {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        static_cast<void>(impl_.release());
    }
}

}  // namespace throwdb::detail
