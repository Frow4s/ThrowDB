#include "throwdb/storage.hpp"

#include "internal.hpp"
#include "task_partition.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace throwdb {
namespace {

constexpr std::uint32_t kMaxIoWorkers = 1024;
constexpr std::uint32_t kMaxIoQueueCapacity = 1U << 20;
constexpr std::uint32_t kMaxIoRangesPerTask = 1U << 20;

[[nodiscard]] std::size_t effective_queue_capacity(
    const ReaderOptions& options) {
    if (options.io_queue_capacity != 0) {
        return options.io_queue_capacity;
    }
    return static_cast<std::size_t>(options.io_workers) * 2;
}

void validate_executor_options(const ReaderOptions& options) {
    if (options.io_ranges_per_task == 0
        || options.io_ranges_per_task > kMaxIoRangesPerTask) {
        throw InvalidArgumentError(
            "io_ranges_per_task must be between 1 and 1048576");
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
        throw InvalidArgumentError("io_queue_capacity must not exceed 1048576");
    }
}

class BoundedExecutor {
public:
    BoundedExecutor(const std::size_t worker_count,
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
                // Tasks submitted by Storage capture their own exceptions. This
                // final guard keeps an unexpected task from killing the process.
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

}  // namespace

class Storage::Impl {
public:
    struct Shard {
        std::filesystem::path path;
        detail::FileDescriptor fd;
        std::uint64_t record_count{0};
    };

    struct Request {
        std::uint64_t shard{0};
        std::uint64_t local_record{0};
        std::uint64_t output_index{0};
    };

    struct PlannedRange {
        std::size_t request_begin{0};
        std::size_t request_end{0};
        std::uint64_t first_record{0};
        std::uint64_t end_record{0};
    };

    Impl(std::filesystem::path path, const ReaderOptions options)
        : path_(std::move(path)),
          options_(options),
          manifest_(detail::read_manifest(path_)) {
        validate_manifest();
        open_and_validate_shards();
        index_ = detail::read_index(path_, manifest_);
        if (options_.io_workers != 0) {
            executor_ = std::make_unique<BoundedExecutor>(
                options_.io_workers, effective_queue_capacity(options_));
        }
    }

    void get_into(const std::span<const std::uint64_t> ids,
                  const std::span<std::byte> output) const {
        const std::uint64_t expected_bytes = detail::checked_multiply(
            ids.size(), manifest_.record_size, "read output");
        if (output.size() != expected_bytes) {
            throw InvalidArgumentError(
                "output byte count does not equal ids.size() * record_size");
        }
        if (ids.empty()) {
            return;
        }

        // Resolve the entire request before touching output. This gives strict
        // missing-ID errors without partially filled caller-owned buffers.
        std::vector<Request> requests;
        requests.reserve(ids.size());
        for (std::size_t output_index = 0; output_index != ids.size();
             ++output_index) {
            std::uint64_t ordinal = 0;
            if (!detail::index_lookup(index_, ids[output_index], ordinal)) {
                throw MissingIdError(ids[output_index]);
            }
            requests.push_back(Request{
                ordinal / manifest_.records_per_shard,
                ordinal % manifest_.records_per_shard,
                output_index});
        }

        std::sort(requests.begin(), requests.end(),
                  [](const Request& left, const Request& right) {
                      if (left.shard != right.shard) {
                          return left.shard < right.shard;
                      }
                      if (left.local_record != right.local_record) {
                          return left.local_record < right.local_record;
                      }
                      return left.output_index < right.output_index;
                  });

        std::vector<PlannedRange> ranges;
        ranges.reserve(requests.size());
        std::size_t begin = 0;
        while (begin != requests.size()) {
            const std::uint64_t shard_id = requests[begin].shard;
            const std::uint64_t first_record = requests[begin].local_record;
            std::uint64_t end_record = first_record + 1;
            std::size_t end = begin + 1;
            while (end != requests.size() && requests[end].shard == shard_id) {
                const std::uint64_t next_record = requests[end].local_record;
                const std::uint64_t candidate_end = std::max(end_record, next_record + 1);
                const std::uint64_t candidate_bytes =
                    (candidate_end - first_record) * manifest_.record_size;
                const std::uint64_t current_end_bytes =
                    end_record * manifest_.record_size;
                const std::uint64_t next_start_bytes =
                    next_record * manifest_.record_size;
                const std::uint64_t gap = next_start_bytes > current_end_bytes
                                              ? next_start_bytes - current_end_bytes
                                              : 0;
                if (gap > options_.max_gap_bytes
                    || candidate_bytes > options_.max_read_range_bytes) {
                    break;
                }
                end_record = candidate_end;
                ++end;
            }
            ranges.push_back(PlannedRange{begin, end, first_record, end_record});
            begin = end;
        }

        if (executor_ == nullptr || ranges.size() == 1) {
            for (const PlannedRange& range : ranges) {
                read_range(requests, range.request_begin, range.request_end,
                           range.first_record, range.end_record, output);
            }
            return;
        }

        const detail::RangeTaskPartition partition = detail::plan_range_tasks(
            ranges.size(), options_.io_workers, options_.io_ranges_per_task);

        auto state = std::make_shared<ReadBatchState>(partition.task_count);
        std::size_t submitted = 0;
        try {
            for (; submitted != partition.task_count; ++submitted) {
                const detail::RangeTaskBounds task_bounds =
                    partition.bounds(submitted);
                executor_->submit(
                    [this, &requests, &ranges, task_bounds, output, state] {
                        std::exception_ptr error;
                        for (std::size_t range_index = task_bounds.begin;
                             range_index != task_bounds.end; ++range_index) {
                            const PlannedRange& range = ranges[range_index];
                            try {
                                read_range(
                                    requests, range.request_begin,
                                    range.request_end, range.first_record,
                                    range.end_record, output);
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
            // Account for every chunk that was never queued, then still wait
            // for all submitted tasks before stack/output captures expire.
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

    [[nodiscard]] std::uint64_t count() const noexcept {
        return manifest_.record_count;
    }
    [[nodiscard]] std::uint32_t dimension() const noexcept {
        return manifest_.dimension;
    }
    [[nodiscard]] DType dtype() const noexcept {
        return static_cast<DType>(manifest_.dtype);
    }
    [[nodiscard]] std::uint64_t record_size() const noexcept {
        return manifest_.record_size;
    }

private:
    void validate_manifest() const {
        if (manifest_.dtype != static_cast<std::uint32_t>(DType::Float16)) {
            throw CorruptionError("manifest contains an unsupported dtype");
        }
        const std::uint64_t expected_record_size = detail::checked_multiply(
            manifest_.dimension, 2, "manifest record size");
        if (manifest_.record_size != expected_record_size) {
            throw CorruptionError("manifest record size does not match dimension");
        }
        if (manifest_.shard_size_bytes < manifest_.record_size
            || manifest_.records_per_shard
                   != manifest_.shard_size_bytes / manifest_.record_size) {
            throw CorruptionError("manifest contains invalid shard sizing");
        }
        const std::uint64_t expected_shards =
            manifest_.record_count / manifest_.records_per_shard
            + (manifest_.record_count % manifest_.records_per_shard != 0 ? 1 : 0);
        if (manifest_.shard_count != expected_shards) {
            throw CorruptionError("manifest shard count does not match record count");
        }
        if (options_.max_read_range_bytes < manifest_.record_size) {
            throw InvalidArgumentError(
                "max_read_range_bytes must fit at least one record");
        }
    }

    void open_and_validate_shards() {
        if (manifest_.shard_count > std::numeric_limits<std::size_t>::max()) {
            throw CorruptionError("shard count does not fit in address space");
        }
        shards_.reserve(static_cast<std::size_t>(manifest_.shard_count));
        for (std::uint64_t shard_id = 0; shard_id != manifest_.shard_count;
             ++shard_id) {
            auto path = detail::shard_path(path_, shard_id);
            auto fd = detail::open_read_only(path);
            std::array<std::byte, detail::kHeaderSize> encoded{};
            detail::pread_exact(fd.get(), encoded.data(), encoded.size(), 0, path);
            const detail::ShardHeader header = detail::decode_shard_header(encoded);
            const std::uint64_t first_ordinal =
                shard_id * manifest_.records_per_shard;
            const std::uint64_t expected_count = std::min(
                manifest_.records_per_shard,
                manifest_.record_count - first_ordinal);
            if (header.shard_id != shard_id
                || header.first_ordinal != first_ordinal
                || header.record_count != expected_count
                || header.record_size != manifest_.record_size
                || header.dimension != manifest_.dimension
                || header.dtype != manifest_.dtype) {
                throw CorruptionError(
                    "shard header does not match manifest for '" + path.string() + "'");
            }
            const std::uint64_t payload_size = detail::checked_multiply(
                expected_count, manifest_.record_size, "shard payload");
            if (detail::file_size(fd.get(), path) != detail::kHeaderSize + payload_size) {
                throw CorruptionError(
                    "shard file size does not match header for '" + path.string() + "'");
            }
            shards_.push_back(Shard{std::move(path), std::move(fd), expected_count});
        }
    }

    void read_range(const std::vector<Request>& requests,
                    const std::size_t begin, const std::size_t end,
                    const std::uint64_t first_record,
                    const std::uint64_t end_record,
                    const std::span<std::byte> output) const {
        const std::uint64_t shard_id = requests[begin].shard;
        if (shard_id >= shards_.size() || end_record > shards_[shard_id].record_count) {
            throw CorruptionError("index points outside a data shard");
        }
        if (end == begin + 1) {
            const std::uint64_t destination_offset =
                requests[begin].output_index * manifest_.record_size;
            detail::pread_exact(
                shards_[shard_id].fd.get(),
                output.data() + static_cast<std::size_t>(destination_offset),
                static_cast<std::size_t>(manifest_.record_size),
                detail::kHeaderSize
                    + requests[begin].local_record * manifest_.record_size,
                shards_[shard_id].path);
            return;
        }
        const std::uint64_t range_bytes_u64 =
            (end_record - first_record) * manifest_.record_size;
        const auto range_bytes = static_cast<std::size_t>(range_bytes_u64);
        std::vector<std::byte> buffer(range_bytes);
        detail::pread_exact(
            shards_[shard_id].fd.get(), buffer.data(), buffer.size(),
            detail::kHeaderSize + first_record * manifest_.record_size,
            shards_[shard_id].path);

        for (std::size_t request_index = begin; request_index != end;
             ++request_index) {
            const Request& request = requests[request_index];
            const std::uint64_t source_offset =
                (request.local_record - first_record) * manifest_.record_size;
            const std::uint64_t destination_offset =
                request.output_index * manifest_.record_size;
            std::memcpy(output.data() + static_cast<std::size_t>(destination_offset),
                        buffer.data() + static_cast<std::size_t>(source_offset),
                        static_cast<std::size_t>(manifest_.record_size));
        }
    }

    std::filesystem::path path_;
    ReaderOptions options_;
    detail::Manifest manifest_;
    detail::IndexData index_;
    std::vector<Shard> shards_;
    std::unique_ptr<BoundedExecutor> executor_;
};

std::unique_ptr<Storage> Storage::open(const std::filesystem::path& path,
                                       const ReaderOptions options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    validate_executor_options(options);
    return std::unique_ptr<Storage>(
        new Storage(std::make_unique<Impl>(path, options)));
}

Storage::Storage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Storage::~Storage() = default;

void Storage::get_into(const std::span<const std::uint64_t> ids,
                       const std::span<std::byte> output) const {
    impl_->get_into(ids, output);
}

void Storage::get_into(const std::uint64_t* ids, const std::size_t count,
                       void* output, const std::size_t output_bytes) const {
    if (count != 0 && ids == nullptr) {
        throw InvalidArgumentError("ids pointer must not be null");
    }
    if (output_bytes != 0 && output == nullptr) {
        throw InvalidArgumentError("output pointer must not be null");
    }
    impl_->get_into(
        std::span<const std::uint64_t>(ids, count),
        std::span<std::byte>(static_cast<std::byte*>(output), output_bytes));
}

std::vector<std::uint16_t> Storage::get(
    const std::span<const std::uint64_t> ids) const {
    const std::uint64_t element_count = detail::checked_multiply(
        ids.size(), dimension(), "read result");
    if (element_count > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("read result does not fit in address space");
    }
    std::vector<std::uint16_t> result(static_cast<std::size_t>(element_count));
    get_into(ids, std::as_writable_bytes(std::span<std::uint16_t>(result)));
    return result;
}

std::uint64_t Storage::count() const noexcept { return impl_->count(); }
std::uint32_t Storage::dimension() const noexcept { return impl_->dimension(); }
DType Storage::dtype() const noexcept { return impl_->dtype(); }
std::uint64_t Storage::record_size() const noexcept { return impl_->record_size(); }

}  // namespace throwdb
