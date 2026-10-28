#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "../src/core/internal.hpp"
#include "../src/core/io_uring_pool.hpp"
#include "../src/core/task_partition.hpp"

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Actual, typename Expected>
void require_equal(const Actual& actual, const Expected& expected,
                   const std::string& message) {
    require(actual == expected, message);
}



void test_range_task_partition_is_balanced_and_bounded() {
    const auto verify = [](
                            const std::size_t range_count,
                            const std::size_t worker_count,
                            const std::size_t max_ranges_per_task,
                            const std::size_t expected_task_count) {
        const auto partition = throwdb::detail::plan_range_tasks(
            range_count, worker_count, max_ranges_per_task);
        require_equal(
            partition.task_count, expected_task_count,
            "range partition produced an unexpected task count");

        std::size_t next_range = 0;
        std::size_t smallest_task = range_count;
        std::size_t largest_task = 0;
        for (std::size_t task = 0; task != partition.task_count; ++task) {
            const auto bounds = partition.bounds(task);
            require_equal(
                bounds.begin, next_range,
                "range partition left a gap or overlap");
            require(bounds.end > bounds.begin, "range partition created an empty task");
            const std::size_t task_size = bounds.end - bounds.begin;
            require(
                task_size <= max_ranges_per_task,
                "range partition exceeded io_ranges_per_task");
            smallest_task = std::min(smallest_task, task_size);
            largest_task = std::max(largest_task, task_size);
            next_range = bounds.end;
        }
        require_equal(next_range, range_count, "range partition lost ranges");
        require(
            largest_task - smallest_task <= 1,
            "range partition was not balanced");
        require(
            partition.task_count >= std::min(range_count, worker_count),
            "range partition did not expose all useful workers");
    };

    verify(17, 16, 8, 16);
    verify(100, 16, 8, 16);
    verify(160, 16, 8, 20);
    verify(7, 16, 8, 7);
    verify(100, 1, 8, 13);
    verify(32, 4, 1, 32);
}

void test_io_uring_pool_planning_is_depth_aware() {
    require_equal(
        throwdb::detail::split_io_uring_queue_depth(2, 32),
        std::vector<std::uint32_t>{16, 16},
        "even io_uring queue depth split changed");
    require_equal(
        throwdb::detail::split_io_uring_queue_depth(2, 3),
        std::vector<std::uint32_t>{2, 1},
        "odd io_uring queue depth split changed");

    std::array<std::byte, 1> placeholder{};
    std::vector<throwdb::detail::IoUringReadOperation> operations(6);
    for (auto& operation : operations) {
        operation.buffer = placeholder.data();
        operation.size = 2048;
    }
    const std::array<std::uint32_t, 2> even_depths{16, 16};
    require_equal(
        throwdb::detail::plan_io_uring_lanes(operations, even_depths),
        std::vector<std::size_t>{0, 1, 0, 1, 0, 1},
        "equal-depth io_uring lanes were not balanced");

    operations.resize(3);
    const std::array<std::uint32_t, 2> uneven_depths{2, 1};
    require_equal(
        throwdb::detail::plan_io_uring_lanes(operations, uneven_depths),
        std::vector<std::size_t>{0, 1, 0},
        "io_uring lane planner ignored unequal queue depths");
}

int interrupted_close_calls = 0;
int interrupted_close_replacement = -1;

int interrupted_descriptor_close(const int fd) noexcept {
    ++interrupted_close_calls;
    if (interrupted_close_calls == 1) {
#if defined(__linux__)
        // Model Linux releasing the fd before reporting an interrupted close,
        // followed by another thread opening a file under the reused number.
        static_cast<void>(::close(fd));
        interrupted_close_replacement = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
#endif
        errno = EINTR;
        return -1;
    }
    return ::close(fd);
}

void test_file_descriptor_interrupted_close_ownership() {
    const int original = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    require(original >= 0, "cannot open descriptor-lifecycle fixture");
    throwdb::detail::FileDescriptor descriptor(original);
    interrupted_close_calls = 0;
    interrupted_close_replacement = -1;
    throwdb::detail::set_file_descriptor_close_test_hook(
        interrupted_descriptor_close);
    descriptor.reset();
    throwdb::detail::set_file_descriptor_close_test_hook(nullptr);
    require_equal(descriptor.get(), -1, "reset retained ownership of closed fd");
#if defined(__linux__)
    const int replacement = interrupted_close_replacement;
    const bool replacement_open =
        replacement >= 0 && ::fcntl(replacement, F_GETFD) >= 0;
    if (replacement_open) {
        static_cast<void>(::close(replacement));
    }
    require_equal(replacement, original, "fixture did not reuse the closed fd");
    require_equal(interrupted_close_calls, 1,
                  "Linux close(EINTR) retried a released descriptor");
    require(replacement_open, "close retry closed an unrelated reused fd");
#else
    require_equal(interrupted_close_calls, 2,
                  "non-Linux interrupted-close retry behavior changed");
    require(::fcntl(original, F_GETFD) < 0 && errno == EBADF,
            "non-Linux interrupted close leaked its descriptor");
#endif
}



}  // namespace

int main() {
    try {
        test_range_task_partition_is_balanced_and_bounded();
        test_io_uring_pool_planning_is_depth_aware();
        test_file_descriptor_interrupted_close_ownership();
        std::cout << "core utility tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core utility test failure: " << error.what() << '\n';
        return 1;
    }
}
