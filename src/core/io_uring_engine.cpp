#include "io_uring_engine.hpp"

#include "throwdb/errors.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace throwdb::detail {

#if defined(__linux__)
namespace {

[[nodiscard]] std::string system_error(
    const std::string& operation, const int error_number) {
    return operation + ": " + std::strerror(error_number)
           + " (errno=" + std::to_string(error_number) + ")";
}

template <typename Value>
[[nodiscard]] Value* at_offset(void* base, const std::uint32_t offset) {
    return reinterpret_cast<Value*>(
        static_cast<std::byte*>(base) + offset);
}

[[nodiscard]] std::uint32_t load_acquire(const std::uint32_t* value) noexcept {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

[[nodiscard]] std::uint32_t load_relaxed(const std::uint32_t* value) noexcept {
    return __atomic_load_n(value, __ATOMIC_RELAXED);
}

void store_release(std::uint32_t* target, const std::uint32_t value) noexcept {
    __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

}  // namespace

class IoUringEngine::Impl {
public:
    explicit Impl(const std::uint32_t requested_queue_depth) {
        io_uring_params parameters{};
        const long setup_result = ::syscall(
            __NR_io_uring_setup, requested_queue_depth, &parameters);
        if (setup_result < 0) {
            const int error_number = errno;
            throw StorageError(system_error("io_uring_setup failed", error_number));
        }
        ring_fd_ = static_cast<int>(setup_result);

        try {
            if ((parameters.features & IORING_FEAT_NODROP) == 0) {
                throw StorageError(
                    "io_uring backend requires IORING_FEAT_NODROP");
            }
            map_rings(parameters);
            queue_depth_ = std::min(
                requested_queue_depth,
                static_cast<std::uint32_t>(parameters.sq_entries));
            if (queue_depth_ == 0 || parameters.cq_entries < queue_depth_) {
                throw StorageError("io_uring returned an invalid queue size");
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    [[nodiscard]] std::uint32_t queue_depth() const noexcept {
        return queue_depth_;
    }

    void read_exact(const std::span<const IoUringReadOperation> operations) {
        if (operations.empty()) {
            return;
        }

        struct OperationState {
            IoUringReadOperation operation;
            std::size_t completed{0};
            bool in_flight{false};
        };
        enum class ErrorKind { None, UnexpectedEof, System, Protocol };
        struct FirstError {
            ErrorKind kind{ErrorKind::None};
            std::uint64_t user_index{0};
            int error_number{0};
            std::string message;
        };

        std::vector<OperationState> states;
        states.reserve(operations.size());
        for (const IoUringReadOperation& operation : operations) {
            if (operation.fd < 0 || operation.buffer == nullptr
                || operation.size == 0) {
                throw InvalidArgumentError(
                    "io_uring read operation contains an invalid fd or buffer");
            }
            if (operation.size > std::numeric_limits<std::uint32_t>::max()) {
                throw InvalidArgumentError(
                    "io_uring read operation exceeds the 32-bit SQE length");
            }
            if (operation.offset
                > std::numeric_limits<std::uint64_t>::max() - operation.size) {
                throw InvalidArgumentError("io_uring read offset overflows");
            }
            states.push_back(OperationState{operation});
        }

        std::lock_guard lock(mutex_);
        if (poisoned_) {
            throw StorageError(
                "io_uring ring is unavailable after a previous protocol error");
        }
        std::deque<std::size_t> pending;
        for (std::size_t index = 0; index != states.size(); ++index) {
            pending.push_back(index);
        }
        FirstError first_error;
        bool protocol_failure = false;
        const std::uint32_t initial_sq_dropped = load_relaxed(sq_dropped_);
        const std::uint32_t initial_cq_overflow = load_relaxed(cq_overflow_);

        std::size_t outstanding = 0;
        while (!pending.empty() || outstanding != 0) {
            if (!pending.empty() && outstanding < queue_depth_) {
                const std::size_t submission_count = std::min<std::size_t>(
                    queue_depth_ - outstanding, pending.size());
                std::vector<std::size_t> submission;
                submission.reserve(submission_count);
                for (std::size_t index = 0; index != submission_count;
                     ++index) {
                    submission.push_back(pending.front());
                    pending.pop_front();
                }

                const std::uint32_t submission_head =
                    publish(submission, states);
                try {
                    submit_all(static_cast<std::uint32_t>(submission.size()));
                } catch (...) {
                    // Discard SQEs the kernel did not consume and wait in
                    // userspace for every older or newly consumed request to
                    // post a CQE. Caller buffers remain alive during the drain.
                    poisoned_ = true;
                    discard_unsubmitted_and_drain(
                        submission_head, outstanding);
                    throw;
                }
                outstanding += submission.size();
            }

            if (outstanding != 0) {
                const std::size_t completed = reap_available_or_wait_one(
                    outstanding, states, pending, first_error,
                    protocol_failure);
                outstanding -= completed;
            }
        }

        if (load_relaxed(sq_dropped_) != initial_sq_dropped
            || load_relaxed(cq_overflow_) != initial_cq_overflow) {
            poisoned_ = true;
            throw StorageError(
                "io_uring reported a dropped submission or completion");
        }
        if (protocol_failure) {
            poisoned_ = true;
        }
        if (first_error.kind == ErrorKind::UnexpectedEof) {
            throw CorruptionError(
                "unexpected EOF in io_uring read operation "
                + std::to_string(first_error.user_index));
        }
        if (first_error.kind == ErrorKind::System) {
            throw StorageError(
                system_error(
                    "io_uring read operation "
                        + std::to_string(first_error.user_index) + " failed",
                    first_error.error_number));
        }
        if (first_error.kind == ErrorKind::Protocol) {
            throw StorageError(first_error.message);
        }
    }

private:
    template <typename OperationState>
    [[nodiscard]] std::uint32_t publish(
        const std::vector<std::size_t>& window,
        std::vector<OperationState>& states) {
        const std::uint32_t head = load_acquire(sq_head_);
        std::uint32_t tail = load_relaxed(sq_tail_);
        const std::uint32_t capacity = *sq_ring_entries_;
        if (capacity - (tail - head) < window.size()) {
            throw StorageError("io_uring submission queue unexpectedly became full");
        }

        for (const std::size_t state_index : window) {
            OperationState& state = states[state_index];
            const std::uint32_t ring_index = tail & *sq_ring_mask_;
            io_uring_sqe& sqe = sqes_[ring_index];
            std::memset(&sqe, 0, sizeof(sqe));
            const std::size_t remaining =
                state.operation.size - state.completed;
            sqe.opcode = IORING_OP_READ;
            sqe.fd = state.operation.fd;
            sqe.off = state.operation.offset + state.completed;
            sqe.addr = reinterpret_cast<std::uint64_t>(
                static_cast<std::byte*>(state.operation.buffer)
                + state.completed);
            sqe.len = static_cast<std::uint32_t>(remaining);
            sqe.user_data = state_index + 1;
            sq_array_[ring_index] = ring_index;
            state.in_flight = true;
            ++tail;
        }
        store_release(sq_tail_, tail);
        return head;
    }

    void submit_all(std::uint32_t remaining) {
        while (remaining != 0) {
            const long result = ::syscall(
                __NR_io_uring_enter, ring_fd_, remaining, 0, 0, nullptr, 0);
            if (result > 0) {
                remaining -= std::min(
                    remaining, static_cast<std::uint32_t>(result));
                continue;
            }
            if (result == 0 || errno == EINTR || errno == EAGAIN
                || errno == EBUSY) {
                std::this_thread::yield();
                continue;
            }
            const int error_number = errno;
            throw StorageError(
                system_error("io_uring_enter submit failed", error_number));
        }
    }

    template <typename OperationState, typename FirstError>
    [[nodiscard]] std::size_t reap_available_or_wait_one(
        const std::size_t outstanding,
        std::vector<OperationState>& states,
        std::deque<std::size_t>& pending,
        FirstError& first_error,
        bool& protocol_failure) {
        std::uint32_t head = load_relaxed(cq_head_);
        std::uint32_t tail = load_acquire(cq_tail_);
        while (head == tail) {
            const long result = ::syscall(
                __NR_io_uring_enter, ring_fd_, 0, 1,
                IORING_ENTER_GETEVENTS, nullptr, 0);
            if (result < 0 && errno != EINTR && errno != EAGAIN) {
                const int error_number = errno;
                poisoned_ = true;
                drain_completions_without_enter(outstanding);
                throw StorageError(system_error(
                    "io_uring_enter wait failed", error_number));
            }
            head = load_relaxed(cq_head_);
            tail = load_acquire(cq_tail_);
        }

        std::size_t reaped = 0;
        while (head != tail && reaped != outstanding) {
            const io_uring_cqe& cqe = cqes_[head & *cq_ring_mask_];
            const std::uint64_t encoded_index = cqe.user_data;
            if (encoded_index == 0 || encoded_index > states.size()) {
                protocol_failure = true;
                if (first_error.kind == decltype(first_error.kind)::None) {
                    first_error.kind = decltype(first_error.kind)::Protocol;
                    first_error.message =
                        "io_uring returned an invalid completion token";
                }
            } else {
                const std::size_t state_index =
                    static_cast<std::size_t>(encoded_index - 1);
                OperationState& state = states[state_index];
                if (!state.in_flight) {
                    protocol_failure = true;
                    if (first_error.kind
                        == decltype(first_error.kind)::None) {
                        first_error.kind =
                            decltype(first_error.kind)::Protocol;
                        first_error.message =
                            "io_uring returned a duplicate completion token";
                    }
                } else if (cqe.res == -EINTR) {
                    state.in_flight = false;
                    pending.push_back(state_index);
                } else if (cqe.res < 0) {
                    state.in_flight = false;
                    if (first_error.kind
                        == decltype(first_error.kind)::None) {
                        first_error.kind =
                            decltype(first_error.kind)::System;
                        first_error.user_index =
                            state.operation.user_index;
                        first_error.error_number = -cqe.res;
                    }
                } else if (cqe.res == 0) {
                    state.in_flight = false;
                    if (first_error.kind
                        == decltype(first_error.kind)::None) {
                        first_error.kind =
                            decltype(first_error.kind)::UnexpectedEof;
                        first_error.user_index =
                            state.operation.user_index;
                    }
                } else if (
                    static_cast<std::size_t>(cqe.res)
                    > state.operation.size - state.completed) {
                    state.in_flight = false;
                    protocol_failure = true;
                    if (first_error.kind
                        == decltype(first_error.kind)::None) {
                        first_error.kind =
                            decltype(first_error.kind)::Protocol;
                        first_error.message =
                            "io_uring returned more bytes than requested";
                    }
                } else {
                    state.in_flight = false;
                    state.completed += static_cast<std::size_t>(cqe.res);
                    if (state.completed != state.operation.size) {
                        pending.push_back(state_index);
                    }
                }
            }
            ++head;
            ++reaped;
        }
        store_release(cq_head_, head);
        return reaped;
    }

    void drain_completions_without_enter(std::size_t expected) noexcept {
        while (expected != 0) {
            std::uint32_t head = load_relaxed(cq_head_);
            const std::uint32_t tail = load_acquire(cq_tail_);
            if (head == tail) {
                std::this_thread::yield();
                continue;
            }
            const std::size_t available = tail - head;
            const std::size_t consumed = std::min(expected, available);
            head += static_cast<std::uint32_t>(consumed);
            store_release(cq_head_, head);
            expected -= consumed;
        }
    }

    void discard_unsubmitted_and_drain(
        const std::uint32_t submission_head,
        const std::size_t previously_outstanding) noexcept {
        const std::uint32_t consumed_head = load_acquire(sq_head_);
        const std::size_t consumed = consumed_head - submission_head;
        store_release(sq_tail_, consumed_head);
        drain_completions_without_enter(previously_outstanding + consumed);
    }

    void map_rings(const io_uring_params& parameters) {
        sq_ring_size_ = parameters.sq_off.array
                        + parameters.sq_entries * sizeof(std::uint32_t);
        cq_ring_size_ = parameters.cq_off.cqes
                        + parameters.cq_entries * sizeof(io_uring_cqe);
        single_mapping_ =
            (parameters.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single_mapping_) {
            sq_ring_size_ = std::max(sq_ring_size_, cq_ring_size_);
            cq_ring_size_ = sq_ring_size_;
            sq_ring_ = map_region(sq_ring_size_, IORING_OFF_SQ_RING);
            cq_ring_ = sq_ring_;
        } else {
            sq_ring_ = map_region(sq_ring_size_, IORING_OFF_SQ_RING);
            cq_ring_ = map_region(cq_ring_size_, IORING_OFF_CQ_RING);
        }
        sqes_size_ = parameters.sq_entries * sizeof(io_uring_sqe);
        sqes_mapping_ = map_region(sqes_size_, IORING_OFF_SQES);

        sq_head_ = at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.head);
        sq_tail_ = at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.tail);
        sq_ring_mask_ =
            at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.ring_mask);
        sq_ring_entries_ =
            at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.ring_entries);
        sq_dropped_ =
            at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.dropped);
        sq_array_ =
            at_offset<std::uint32_t>(sq_ring_, parameters.sq_off.array);
        sqes_ = static_cast<io_uring_sqe*>(sqes_mapping_);

        cq_head_ = at_offset<std::uint32_t>(cq_ring_, parameters.cq_off.head);
        cq_tail_ = at_offset<std::uint32_t>(cq_ring_, parameters.cq_off.tail);
        cq_ring_mask_ =
            at_offset<std::uint32_t>(cq_ring_, parameters.cq_off.ring_mask);
        cq_overflow_ =
            at_offset<std::uint32_t>(cq_ring_, parameters.cq_off.overflow);
        cqes_ = at_offset<io_uring_cqe>(cq_ring_, parameters.cq_off.cqes);
    }

    [[nodiscard]] void* map_region(
        const std::size_t size, const std::uint64_t offset) const {
        void* mapping = ::mmap(
            nullptr, size, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, ring_fd_, static_cast<off_t>(offset));
        if (mapping == MAP_FAILED) {
            const int error_number = errno;
            throw StorageError(system_error("io_uring mmap failed", error_number));
        }
        return mapping;
    }

    void cleanup() noexcept {
        if (sqes_mapping_ != nullptr) {
            ::munmap(sqes_mapping_, sqes_size_);
            sqes_mapping_ = nullptr;
        }
        if (single_mapping_) {
            if (sq_ring_ != nullptr) {
                ::munmap(sq_ring_, sq_ring_size_);
            }
        } else {
            if (sq_ring_ != nullptr) {
                ::munmap(sq_ring_, sq_ring_size_);
            }
            if (cq_ring_ != nullptr) {
                ::munmap(cq_ring_, cq_ring_size_);
            }
        }
        sq_ring_ = nullptr;
        cq_ring_ = nullptr;
        if (ring_fd_ >= 0) {
            ::close(ring_fd_);
            ring_fd_ = -1;
        }
    }

    int ring_fd_{-1};
    std::uint32_t queue_depth_{0};
    bool single_mapping_{false};
    void* sq_ring_{nullptr};
    void* cq_ring_{nullptr};
    void* sqes_mapping_{nullptr};
    std::size_t sq_ring_size_{0};
    std::size_t cq_ring_size_{0};
    std::size_t sqes_size_{0};
    std::uint32_t* sq_head_{nullptr};
    std::uint32_t* sq_tail_{nullptr};
    std::uint32_t* sq_ring_mask_{nullptr};
    std::uint32_t* sq_ring_entries_{nullptr};
    std::uint32_t* sq_dropped_{nullptr};
    std::uint32_t* sq_array_{nullptr};
    io_uring_sqe* sqes_{nullptr};
    std::uint32_t* cq_head_{nullptr};
    std::uint32_t* cq_tail_{nullptr};
    std::uint32_t* cq_ring_mask_{nullptr};
    std::uint32_t* cq_overflow_{nullptr};
    io_uring_cqe* cqes_{nullptr};
    std::mutex mutex_;
    bool poisoned_{false};
};

#else

class IoUringEngine::Impl {
public:
    explicit Impl(const std::uint32_t) {
        throw StorageError("io_uring backend is available only on Linux");
    }

    [[nodiscard]] std::uint32_t queue_depth() const noexcept { return 0; }
    void read_exact(std::span<const IoUringReadOperation>) {}
};

#endif

IoUringEngine::IoUringEngine(const std::uint32_t queue_depth)
    : impl_(std::make_unique<Impl>(queue_depth)) {}

IoUringEngine::~IoUringEngine() = default;

void IoUringEngine::read_exact(
    const std::span<const IoUringReadOperation> operations) {
    impl_->read_exact(operations);
}

std::uint32_t IoUringEngine::queue_depth() const noexcept {
    return impl_->queue_depth();
}

}  // namespace throwdb::detail
