#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"
#include "../src/core/physical_reader.hpp"

namespace {

constexpr std::uint64_t kRecordSize = 4;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-physical-reader-test-"
                   + std::to_string(timestamp) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class ScopedFd {
public:
    explicit ScopedFd(const std::filesystem::path& path)
        : value_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (value_ < 0) {
            throw std::runtime_error("cannot open test data file");
        }
    }

    ~ScopedFd() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    [[nodiscard]] int get() const noexcept { return value_; }

private:
    int value_{-1};
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Exception, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
    try {
        std::forward<Callback>(callback)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        fail(message + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        fail(message + " (wrong non-standard exception)");
    }
    fail(message + " (no exception)");
}

[[nodiscard]] std::vector<std::byte> pattern(
    const std::uint8_t base,
    const std::size_t row_count) {
    std::vector<std::byte> result(row_count * kRecordSize);
    for (std::size_t row = 0; row != row_count; ++row) {
        for (std::size_t column = 0; column != kRecordSize; ++column) {
            result[row * kRecordSize + column] = static_cast<std::byte>(
                base + static_cast<std::uint8_t>(row * 8 + column));
        }
    }
    return result;
}

void write_bytes(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("cannot write test data file");
    }
}

void require_record(
    const std::span<const std::byte> output,
    const std::size_t output_index,
    const std::span<const std::byte> source,
    const std::size_t source_row,
    const std::string& message) {
    const auto actual = output.subspan(
        output_index * kRecordSize, kRecordSize);
    const auto expected = source.subspan(
        source_row * kRecordSize, kRecordSize);
    require(std::equal(actual.begin(), actual.end(), expected.begin()), message);
}

void test_sequential_cross_source_and_repeated_rows() {
    TempDirectory temporary;
    const std::vector<std::byte> first_data = pattern(10, 8);
    const std::vector<std::byte> second_data = pattern(110, 8);
    const auto first_path = temporary.path() / "first.bin";
    const auto second_path = temporary.path() / "second.bin";
    write_bytes(first_path, first_data);
    write_bytes(second_path, second_data);
    const ScopedFd first(first_path);
    const ScopedFd second(second_path);

    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {second.get(), 2 * kRecordSize, 2},
        {first.get(), 3 * kRecordSize, 0},
        {first.get(), 1 * kRecordSize, 1},
        {first.get(), 3 * kRecordSize, 3},
    };
    std::vector<std::byte> output(requests.size() * kRecordSize);
    throwdb::detail::PhysicalReader reader;
    reader.execute(kRecordSize, requests, output);

    require_record(output, 0, first_data, 3, "first repeated row is wrong");
    require_record(output, 1, first_data, 1, "first-source row is wrong");
    require_record(output, 2, second_data, 2, "second-source row is wrong");
    require_record(output, 3, first_data, 3, "second repeated row is wrong");
}

void test_gap_merging_scatters_to_original_output_order() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(30, 8);
    const auto path = temporary.path() / "merged.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_gap_bytes = 2 * kRecordSize;
    options.max_read_range_bytes = 64;
    throwdb::detail::PhysicalReader reader(options);
    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {fd.get(), 4 * kRecordSize, 0},
        {fd.get(), 2 * kRecordSize, 1},
        {fd.get(), 1 * kRecordSize, 2},
    };
    std::vector<std::byte> output(requests.size() * kRecordSize);
    reader.execute(kRecordSize, requests, output);

    require_record(output, 0, data, 4, "merged range lost output row 0");
    require_record(output, 1, data, 2, "merged range lost output row 1");
    require_record(output, 2, data, 1, "merged range lost output row 2");
}

void test_profiled_execute_reports_plan_and_matches_default_output() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(35, 8);
    const auto path = temporary.path() / "profiled-merged.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_gap_bytes = 2 * kRecordSize;
    options.max_read_range_bytes = 64;
    throwdb::detail::PhysicalReader reader(options);
    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {fd.get(), 4 * kRecordSize, 0},
        {fd.get(), 2 * kRecordSize, 1},
        {fd.get(), 1 * kRecordSize, 2},
    };
    std::vector<std::byte> expected(requests.size() * kRecordSize);
    std::vector<std::byte> actual(requests.size() * kRecordSize);
    reader.execute(kRecordSize, requests, expected);

    const throwdb::detail::PhysicalReadStats stats =
        reader.execute_profiled(kRecordSize, requests, actual);

    require(actual == expected, "profiled execute changed output bytes");
    require(stats.request_count == 3, "profiled request count is wrong");
    require(stats.range_count == 1, "merged range count is wrong");
    require(stats.task_count == 1, "inline backend task count is wrong");
    require(
        stats.planned_read_bytes == 4 * kRecordSize,
        "merged planned byte count is wrong");
    require(
        stats.requested_bytes == 3 * kRecordSize
            && stats.completed_read_bytes == 4 * kRecordSize
            && stats.useful_read_bytes == 3 * kRecordSize
            && stats.gap_bytes == kRecordSize
            && stats.overread_bytes == kRecordSize,
        "profiled byte amplification metrics are wrong");
    require(
        stats.unique_file_count == 1
            && stats.pages_touched == 1
            && stats.page_runs == 1
            && stats.chunks_touched == 1
            && stats.chunk_runs == 1,
        "profiled file/page/chunk metrics are wrong");
    require(
        stats.input_discontinuities == 2
            && stats.input_inversions == 2
            && stats.maximum_range_bytes == 4 * kRecordSize
            && stats.mean_range_bytes == 4 * kRecordSize,
        "profiled input-order/range metrics are wrong");
    require(stats.plan_seconds >= 0.0, "plan duration is negative");
    require(stats.sort_seconds >= 0.0, "sort duration is negative");
    require(stats.backend_seconds >= 0.0, "backend duration is negative");
}

void test_concurrent_profiled_thread_pool_calls_keep_per_call_stats() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(65, 8);
    const auto path = temporary.path() / "profiled-pool.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 2;
    options.io_queue_capacity = 2;
    options.io_ranges_per_task = 2;
    throwdb::detail::PhysicalReader reader(options);
    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {fd.get(), 0 * kRecordSize, 0},
        {fd.get(), 2 * kRecordSize, 1},
        {fd.get(), 4 * kRecordSize, 2},
        {fd.get(), 6 * kRecordSize, 3},
    };

    constexpr std::size_t kCallerCount = 4;
    std::array<std::vector<std::byte>, kCallerCount> outputs;
    std::array<throwdb::detail::PhysicalReadStats, kCallerCount> stats;
    std::vector<std::thread> callers;
    callers.reserve(kCallerCount);
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (std::size_t caller = 0; caller != kCallerCount; ++caller) {
        outputs[caller].resize(requests.size() * kRecordSize);
        callers.emplace_back([&, caller] {
            try {
                stats[caller] = reader.execute_profiled(
                    kRecordSize, requests, outputs[caller]);
            } catch (...) {
                std::lock_guard lock(error_mutex);
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    for (std::size_t caller = 0; caller != kCallerCount; ++caller) {
        require_record(outputs[caller], 0, data, 0, "profiled pool row 0 is wrong");
        require_record(outputs[caller], 1, data, 2, "profiled pool row 1 is wrong");
        require_record(outputs[caller], 2, data, 4, "profiled pool row 2 is wrong");
        require_record(outputs[caller], 3, data, 6, "profiled pool row 3 is wrong");
        require(
            stats[caller].request_count == requests.size(),
            "concurrent profiled request count is wrong");
        require(
            stats[caller].range_count == requests.size(),
            "concurrent profiled range count is wrong");
        require(
            stats[caller].task_count == 2,
            "concurrent profiled task count is wrong");
        require(
            stats[caller].planned_read_bytes
                == requests.size() * kRecordSize,
            "concurrent profiled planned byte count is wrong");
    }
}

void test_thread_pool_and_concurrent_execute() {
    TempDirectory temporary;
    const std::vector<std::byte> first_data = pattern(50, 8);
    const std::vector<std::byte> second_data = pattern(150, 8);
    const auto first_path = temporary.path() / "pool-first.bin";
    const auto second_path = temporary.path() / "pool-second.bin";
    write_bytes(first_path, first_data);
    write_bytes(second_path, second_data);
    const ScopedFd first(first_path);
    const ScopedFd second(second_path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 3;
    options.io_queue_capacity = 2;
    options.io_ranges_per_task = 1;
    throwdb::detail::PhysicalReader reader(options);
    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {first.get(), 0 * kRecordSize, 0},
        {first.get(), 2 * kRecordSize, 1},
        {second.get(), 1 * kRecordSize, 2},
        {second.get(), 3 * kRecordSize, 3},
        {first.get(), 0 * kRecordSize, 4},
    };

    constexpr std::size_t kCallerCount = 8;
    std::array<std::vector<std::byte>, kCallerCount> outputs;
    std::vector<std::thread> callers;
    callers.reserve(kCallerCount);
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (std::size_t caller = 0; caller != kCallerCount; ++caller) {
        outputs[caller].resize(requests.size() * kRecordSize);
        callers.emplace_back([&, caller] {
            try {
                reader.execute(kRecordSize, requests, outputs[caller]);
            } catch (...) {
                std::lock_guard lock(error_mutex);
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    for (const std::vector<std::byte>& output : outputs) {
        require_record(output, 0, first_data, 0, "pool row 0 is wrong");
        require_record(output, 1, first_data, 2, "pool row 1 is wrong");
        require_record(output, 2, second_data, 1, "pool row 2 is wrong");
        require_record(output, 3, second_data, 3, "pool row 3 is wrong");
        require_record(output, 4, first_data, 0, "pool repeated row is wrong");
    }
}

void test_thread_pool_drr_small_job_overtakes_large_job() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(90, 16);
    const auto path = temporary.path() / "fairness.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 2;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 1;

    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    std::size_t initial_large_tasks = 0;
    bool small_submitted = false;
    bool release_initial_large_tasks = false;
    bool release_remaining_large_tasks = false;
    bool small_finished = false;

    auto hooks = std::make_shared<
        throwdb::detail::PhysicalReaderTestHooks>();
    hooks->after_job_submitted = [&](const std::size_t task_count) {
        if (task_count != 1) {
            return;
        }
        {
            std::lock_guard lock(gate_mutex);
            small_submitted = true;
        }
        gate_condition.notify_all();
    };
    hooks->before_task = [&](
        const std::size_t task_count,
        const std::size_t task_index) {
        if (task_count != 8) {
            return;
        }
        std::unique_lock lock(gate_mutex);
        if (task_index < 2) {
            ++initial_large_tasks;
            gate_condition.notify_all();
            gate_condition.wait(lock, [&] {
                return release_initial_large_tasks;
            });
            return;
        }
        gate_condition.wait(lock, [&] {
            return release_remaining_large_tasks;
        });
    };

    throwdb::detail::PhysicalReader reader(options, hooks);
    std::vector<throwdb::detail::PhysicalReadRequest> large_requests;
    large_requests.reserve(8);
    for (std::size_t index = 0; index != 8; ++index) {
        large_requests.push_back(throwdb::detail::PhysicalReadRequest{
            fd.get(), index * 2 * kRecordSize, index});
    }
    const std::array<throwdb::detail::PhysicalReadRequest, 1> small_requests{
        throwdb::detail::PhysicalReadRequest{
            fd.get(), 15 * kRecordSize, 0}};
    std::vector<std::byte> large_output(
        large_requests.size() * kRecordSize);
    std::vector<std::byte> small_output(kRecordSize);
    std::exception_ptr large_error;
    std::exception_ptr small_error;
    std::atomic<bool> large_finished{false};

    std::thread large_caller([&] {
        try {
            reader.execute(kRecordSize, large_requests, large_output);
        } catch (...) {
            large_error = std::current_exception();
        }
        large_finished.store(true, std::memory_order_release);
    });

    bool initial_tasks_started = false;
    {
        std::unique_lock lock(gate_mutex);
        initial_tasks_started = gate_condition.wait_for(
            lock, std::chrono::seconds(5), [&] {
                return initial_large_tasks == 2;
            });
    }
    if (!initial_tasks_started) {
        {
            std::lock_guard lock(gate_mutex);
            release_initial_large_tasks = true;
            release_remaining_large_tasks = true;
        }
        gate_condition.notify_all();
        large_caller.join();
        fail("large request did not occupy both worker slots");
    }

    std::thread small_caller([&] {
        try {
            reader.execute(kRecordSize, small_requests, small_output);
        } catch (...) {
            small_error = std::current_exception();
        }
        {
            std::lock_guard lock(gate_mutex);
            small_finished = true;
        }
        gate_condition.notify_all();
    });

    bool observed_small_submission = false;
    {
        std::unique_lock lock(gate_mutex);
        observed_small_submission = gate_condition.wait_for(
            lock, std::chrono::seconds(5), [&] {
                return small_submitted;
            });
        release_initial_large_tasks = true;
    }
    gate_condition.notify_all();

    bool observed_small_completion = false;
    if (observed_small_submission) {
        std::unique_lock lock(gate_mutex);
        observed_small_completion = gate_condition.wait_for(
            lock, std::chrono::seconds(5), [&] {
                return small_finished;
            });
    }
    const bool large_was_still_running =
        !large_finished.load(std::memory_order_acquire);
    {
        std::lock_guard lock(gate_mutex);
        release_remaining_large_tasks = true;
    }
    gate_condition.notify_all();
    small_caller.join();
    large_caller.join();

    require(observed_small_submission,
            "small request was not admitted by the shared coordinator");
    require(observed_small_completion,
            "small request starved behind the large request");
    require(large_was_still_running,
            "large request completed before the later small request");
    if (small_error != nullptr) {
        std::rethrow_exception(small_error);
    }
    if (large_error != nullptr) {
        std::rethrow_exception(large_error);
    }
    require_record(small_output, 0, data, 15,
                   "fairly scheduled small row is wrong");
    for (std::size_t index = 0; index != large_requests.size(); ++index) {
        require_record(
            large_output, index, data, index * 2,
            "fairly scheduled large row is wrong");
    }
}

void test_thread_pool_admission_is_bounded_and_fifo() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(160, 8);
    const auto path = temporary.path() / "admission.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 1;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 1;

    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    std::vector<std::size_t> admitted_order;
    std::vector<std::size_t> waiting_order;
    std::size_t tasks_started = 0;
    bool release_tasks = false;
    auto hooks = std::make_shared<
        throwdb::detail::PhysicalReaderTestHooks>();
    hooks->after_job_submitted = [&](const std::size_t task_count) {
        {
            std::lock_guard lock(gate_mutex);
            admitted_order.push_back(task_count);
        }
        gate_condition.notify_all();
    };
    hooks->before_admission_wait = [&](const std::size_t task_count) {
        {
            std::lock_guard lock(gate_mutex);
            waiting_order.push_back(task_count);
        }
        gate_condition.notify_all();
    };
    hooks->before_task = [&](
        const std::size_t,
        const std::size_t) {
        std::unique_lock lock(gate_mutex);
        ++tasks_started;
        gate_condition.notify_all();
        gate_condition.wait(lock, [&] { return release_tasks; });
    };

    throwdb::detail::PhysicalReader reader(options, hooks);
    const std::array<std::size_t, 4> query_sizes{1, 1, 2, 3};
    std::array<std::vector<throwdb::detail::PhysicalReadRequest>, 4> queries;
    std::array<std::vector<std::byte>, 4> outputs;
    for (std::size_t caller = 0; caller != queries.size(); ++caller) {
        for (std::size_t index = 0; index != query_sizes[caller]; ++index) {
            queries[caller].push_back(
                throwdb::detail::PhysicalReadRequest{
                    fd.get(),
                    (caller + index) * kRecordSize,
                    index});
        }
        outputs[caller].resize(query_sizes[caller] * kRecordSize);
    }

    std::array<std::exception_ptr, 4> errors{};
    std::vector<std::thread> callers;
    callers.reserve(4);
    const auto start_caller = [&](const std::size_t caller) {
        callers.emplace_back([&, caller] {
            try {
                reader.execute(
                    kRecordSize, queries[caller], outputs[caller]);
            } catch (...) {
                errors[caller] = std::current_exception();
            }
        });
    };
    const auto wait_for_state = [&](auto&& predicate) {
        std::unique_lock lock(gate_mutex);
        return gate_condition.wait_for(
            lock,
            std::chrono::seconds(5),
            std::forward<decltype(predicate)>(predicate));
    };

    bool fixture_ready = true;
    start_caller(0);
    fixture_ready = wait_for_state([&] {
        return admitted_order.size() == 1 && tasks_started == 1;
    });
    if (fixture_ready) {
        start_caller(1);
        fixture_ready = wait_for_state([&] {
            return admitted_order.size() == 2;
        });
    }
    if (fixture_ready) {
        start_caller(2);
        fixture_ready = wait_for_state([&] {
            return waiting_order.size() == 1;
        });
    }
    if (fixture_ready) {
        start_caller(3);
        fixture_ready = wait_for_state([&] {
            return waiting_order.size() == 2;
        });
    }
    bool bounded_before_release = false;
    {
        std::lock_guard lock(gate_mutex);
        bounded_before_release = admitted_order.size() == 2;
        release_tasks = true;
    }
    gate_condition.notify_all();
    for (std::thread& caller : callers) {
        caller.join();
    }

    require(fixture_ready,
            "admission fixture did not reach both FIFO waiters");
    require(bounded_before_release,
            "coordinator admitted more than workers + queue capacity");
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
    require(
        admitted_order == std::vector<std::size_t>({1, 1, 2, 3}),
        "coordinator admission did not preserve ticket FIFO");
    require(
        waiting_order == std::vector<std::size_t>({2, 3}),
        "coordinator wait tickets were observed out of order");
    for (std::size_t caller = 0; caller != outputs.size(); ++caller) {
        for (std::size_t index = 0; index != query_sizes[caller]; ++index) {
            require_record(
                outputs[caller], index, data, caller + index,
                "admission-controlled output row is wrong");
        }
    }
}

void test_post_submit_failure_waits_for_issued_task_drain() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(120, 4);
    const auto path = temporary.path() / "post-submit-failure.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 1;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 1;

    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    bool issued_task_started = false;
    bool release_issued_task = false;
    bool fault_injected = false;
    bool caller_finished = false;

    auto hooks = std::make_shared<
        throwdb::detail::PhysicalReaderTestHooks>();
    hooks->before_task = [&](
        const std::size_t task_count,
        const std::size_t task_index) {
        if (task_count != 2 || task_index != 0) {
            return;
        }
        std::unique_lock lock(gate_mutex);
        issued_task_started = true;
        gate_condition.notify_all();
        gate_condition.wait(lock, [&] { return release_issued_task; });
    };
    hooks->after_dispatch_batch_submitted = [&] {
        {
            std::lock_guard lock(gate_mutex);
            if (fault_injected) {
                return;
            }
            fault_injected = true;
        }
        gate_condition.notify_all();
        throw std::runtime_error("injected post-submit failure");
    };

    throwdb::detail::PhysicalReader reader(options, hooks);
    const std::array<throwdb::detail::PhysicalReadRequest, 2> requests{
        throwdb::detail::PhysicalReadRequest{fd.get(), 0, 0},
        throwdb::detail::PhysicalReadRequest{fd.get(), 2 * kRecordSize, 1},
    };
    std::vector<std::byte> output(requests.size() * kRecordSize);
    std::exception_ptr caller_error;
    std::thread caller([&] {
        try {
            reader.execute(kRecordSize, requests, output);
        } catch (...) {
            caller_error = std::current_exception();
        }
        {
            std::lock_guard lock(gate_mutex);
            caller_finished = true;
        }
        gate_condition.notify_all();
    });

    bool observed_fault_and_task = false;
    {
        std::unique_lock lock(gate_mutex);
        observed_fault_and_task = gate_condition.wait_for(
            lock, std::chrono::seconds(5), [&] {
                return fault_injected && issued_task_started;
            });
    }
    bool returned_before_drain = false;
    {
        std::lock_guard lock(gate_mutex);
        returned_before_drain = caller_finished;
        release_issued_task = true;
    }
    gate_condition.notify_all();
    caller.join();

    require(observed_fault_and_task,
            "post-submit fault fixture did not reach issued work");
    require(!returned_before_drain,
            "execute returned while an issued task still held caller data");
    require(caller_error != nullptr,
            "post-submit coordinator failure was not propagated");
    try {
        std::rethrow_exception(caller_error);
    } catch (const std::runtime_error& error) {
        require(
            std::string(error.what()).find("injected post-submit failure")
                != std::string::npos,
            "post-submit failure message changed");
    }

    // The fault is one-shot. A one-range request must use the same coordinator
    // and prove that it remains usable after the drained cancellation.
    const std::array<throwdb::detail::PhysicalReadRequest, 1> recovery_request{
        throwdb::detail::PhysicalReadRequest{fd.get(), kRecordSize, 0}};
    std::vector<std::byte> recovery_output(kRecordSize);
    reader.execute(kRecordSize, recovery_request, recovery_output);
    require_record(recovery_output, 0, data, 1,
                   "coordinator recovery row is wrong");
}

void test_parallel_read_error_drains_and_coordinator_recovers() {
    TempDirectory temporary;
    const auto bad_path = temporary.path() / "bad.bin";
    const auto good_path = temporary.path() / "good.bin";
    write_bytes(bad_path, std::span<const std::byte>());
    const std::vector<std::byte> good_data = pattern(140, 4);
    write_bytes(good_path, good_data);
    const ScopedFd bad(bad_path);
    const ScopedFd good(good_path);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = kRecordSize;
    options.io_workers = 2;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 1;
    throwdb::detail::PhysicalReader reader(options);
    const std::array<throwdb::detail::PhysicalReadRequest, 2> failing{
        throwdb::detail::PhysicalReadRequest{bad.get(), 0, 0},
        throwdb::detail::PhysicalReadRequest{good.get(), 0, 1},
    };
    std::vector<std::byte> failing_output(failing.size() * kRecordSize);
    require_throws<throwdb::StorageError>(
        [&] { reader.execute(kRecordSize, failing, failing_output); },
        "parallel physical read failure was not propagated");

    const std::array<throwdb::detail::PhysicalReadRequest, 1> recovery{
        throwdb::detail::PhysicalReadRequest{good.get(), kRecordSize, 0}};
    std::vector<std::byte> recovery_output(kRecordSize);
    reader.execute(kRecordSize, recovery, recovery_output);
    require_record(recovery_output, 0, good_data, 1,
                   "coordinator failed after a backend read error");
}

void test_geometry_is_validated_before_output_is_touched() {
    TempDirectory temporary;
    const std::vector<std::byte> data = pattern(70, 2);
    const auto path = temporary.path() / "geometry.bin";
    write_bytes(path, data);
    const ScopedFd fd(path);
    throwdb::detail::PhysicalReader reader;

    std::vector<std::byte> output(2 * kRecordSize, std::byte{0x5a});
    const auto require_untouched = [&output] {
        require(
            std::all_of(
                output.begin(), output.end(),
                [](const std::byte value) { return value == std::byte{0x5a}; }),
            "invalid geometry modified output");
    };

    const std::vector<throwdb::detail::PhysicalReadRequest> duplicate_output{
        {fd.get(), 0, 0},
        {fd.get(), kRecordSize, 0},
    };
    require_throws<throwdb::InvalidArgumentError>(
        [&] { reader.execute(kRecordSize, duplicate_output, output); },
        "duplicate output index was accepted");
    require_untouched();

    const std::vector<throwdb::detail::PhysicalReadRequest> bad_offset{
        {fd.get(), std::numeric_limits<std::uint64_t>::max() - 1, 0},
        {fd.get(), kRecordSize, 1},
    };
    require_throws<throwdb::InvalidArgumentError>(
        [&] { reader.execute(kRecordSize, bad_offset, output); },
        "overflowing source offset was accepted");
    require_untouched();

    const std::vector<throwdb::detail::PhysicalReadRequest> bad_fd{
        {-1, 0, 0},
        {fd.get(), kRecordSize, 1},
    };
    require_throws<throwdb::InvalidArgumentError>(
        [&] { reader.execute(kRecordSize, bad_fd, output); },
        "negative source descriptor was accepted");
    require_untouched();

    const std::array<throwdb::detail::PhysicalReadRequest, 1> one_request{
        throwdb::detail::PhysicalReadRequest{fd.get(), 0, 0}};
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            reader.execute(
                kRecordSize,
                one_request,
                std::span<std::byte>(output.data(), output.size()));
        },
        "wrong output byte count was accepted");
    require_untouched();
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*callback)();
    };
    const std::array tests{
        TestCase{
            "sequential_cross_source_and_repeated_rows",
            test_sequential_cross_source_and_repeated_rows},
        TestCase{
            "gap_merging_scatters_to_original_output_order",
            test_gap_merging_scatters_to_original_output_order},
        TestCase{
            "profiled_execute_reports_plan_and_matches_default_output",
            test_profiled_execute_reports_plan_and_matches_default_output},
        TestCase{
            "concurrent_profiled_thread_pool_calls_keep_per_call_stats",
            test_concurrent_profiled_thread_pool_calls_keep_per_call_stats},
        TestCase{
            "thread_pool_and_concurrent_execute",
            test_thread_pool_and_concurrent_execute},
        TestCase{
            "thread_pool_drr_small_job_overtakes_large_job",
            test_thread_pool_drr_small_job_overtakes_large_job},
        TestCase{
            "thread_pool_admission_is_bounded_and_fifo",
            test_thread_pool_admission_is_bounded_and_fifo},
        TestCase{
            "post_submit_failure_waits_for_issued_task_drain",
            test_post_submit_failure_waits_for_issued_task_drain},
        TestCase{
            "parallel_read_error_drains_and_coordinator_recovers",
            test_parallel_read_error_drains_and_coordinator_recovers},
        TestCase{
            "geometry_is_validated_before_output_is_touched",
            test_geometry_is_validated_before_output_is_touched},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.callback();
            ++passed;
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
            return 1;
        }
    }
    std::cout << passed << " physical-reader tests passed\n";
    return 0;
}
