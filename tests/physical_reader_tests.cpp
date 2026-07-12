#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
            "thread_pool_and_concurrent_execute",
            test_thread_pool_and_concurrent_execute},
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
