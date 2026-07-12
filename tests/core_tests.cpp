#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"
#include "../src/core/task_partition.hpp"

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-test-" + std::to_string(timestamp) + "-"
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

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Actual, typename Expected>
void require_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string& message) {
    if (!(actual == expected)) {
        fail(message);
    }
}

std::vector<std::uint16_t> make_payload(
    const std::size_t rows,
    const std::size_t dimension) {
    std::vector<std::uint16_t> result(rows * dimension);
    for (std::size_t index = 0; index < result.size(); ++index) {
        // ThrowDB must preserve every binary16 payload, including patterns that
        // represent NaNs. No floating-point comparison is involved.
        result[index] = static_cast<std::uint16_t>(
            (index * 40503U + 17U) & std::numeric_limits<std::uint16_t>::max());
    }
    return result;
}

void build_store(
    const std::filesystem::path& path,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::uint16_t> payload,
    const std::uint32_t dimension,
    const std::uint64_t shard_size_bytes) {
    throwdb::WriterOptions options;
    options.dimension = dimension;
    options.shard_size_bytes = shard_size_bytes;
    auto writer = throwdb::Writer::create(path, options);
    writer->append(ids, std::as_bytes(payload));
    writer->finish();
}

std::filesystem::path find_regular_file(
    const std::filesystem::path& root,
    const std::string_view name_fragment) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()
            && entry.path().filename().string().find(name_fragment)
                   != std::string::npos) {
            return entry.path();
        }
    }
    fail("could not find a file containing '" + std::string(name_fragment) + "'");
}

std::size_t count_regular_files(
    const std::filesystem::path& root,
    const std::string_view name_fragment) {
    std::size_t result = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()
            && entry.path().filename().string().find(name_fragment)
                   != std::string::npos) {
            ++result;
        }
    }
    return result;
}

void test_round_trip_arbitrary_ids_and_order() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 13;
    const std::vector<std::uint64_t> ids{
        0,
        std::numeric_limits<std::uint64_t>::max(),
        42,
        9'000'000'000'000ULL,
        7,
    };
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    auto storage = throwdb::Storage::open(store_path);
    require_equal(storage->dimension(), dimension, "dimension did not round-trip");
    require_equal(storage->count(), std::uint64_t{ids.size()}, "count is wrong");
    require_equal(
        storage->record_size(),
        std::uint64_t{dimension * sizeof(std::uint16_t)},
        "record size is wrong");

    const std::array<std::size_t, 6> positions{3, 0, 4, 1, 3, 2};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    query.reserve(positions.size());
    expected.reserve(positions.size() * dimension);
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }

    require_equal(storage->get(query), expected, "query order or payload changed");
}

void test_multiple_append_batches_and_shards() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::size_t row_count = 19;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 104729U + 13U);
    }
    const auto payload = make_payload(row_count, dimension);

    throwdb::WriterOptions options;
    options.dimension = dimension;
    options.shard_size_bytes = 2U * dimension * sizeof(std::uint16_t);
    auto writer = throwdb::Writer::create(store_path, options);
    constexpr std::array<std::size_t, 4> batch_sizes{3, 1, 9, 6};
    std::size_t offset = 0;
    for (const auto batch_size : batch_sizes) {
        const auto id_batch = std::span<const std::uint64_t>(ids).subspan(
            offset, batch_size);
        const auto value_batch = std::span<const std::uint16_t>(payload).subspan(
            offset * dimension, batch_size * dimension);
        writer->append(id_batch, std::as_bytes(value_batch));
        offset += batch_size;
    }
    require_equal(writer->count(), std::uint64_t{row_count}, "writer count is wrong");
    writer->finish();

    require(
        count_regular_files(store_path, "shard-") >= 2,
        "tiny shard capacity did not produce several shard files");

    auto storage = throwdb::Storage::open(store_path);
    const std::array<std::size_t, 9> positions{18, 0, 10, 4, 17, 5, 8, 1, 13};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }
    require_equal(storage->get(query), expected, "cross-shard read is wrong");
}

void test_duplicate_id_rejects_publication() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    throwdb::WriterOptions options;
    options.dimension = 3;
    options.shard_size_bytes = 1U << 20;
    auto writer = throwdb::Writer::create(store_path, options);

    const std::array<std::uint64_t, 3> first_ids{10, 20, 30};
    const std::array<std::uint16_t, 9> first_payload{};
    writer->append(first_ids, std::as_bytes(std::span(first_payload)));
    const std::array<std::uint64_t, 2> second_ids{40, 20};
    const std::array<std::uint16_t, 6> second_payload{};
    writer->append(second_ids, std::as_bytes(std::span(second_payload)));

    bool rejected = false;
    try {
        writer->finish();
    } catch (const throwdb::DuplicateIdError& error) {
        rejected = true;
        require_equal(error.id(), std::uint64_t{20}, "wrong duplicate ID reported");
    }
    require(rejected, "duplicate ID did not fail finish()");
    writer.reset();

    bool opened = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
        opened = true;
    } catch (const throwdb::StorageError&) {
    }
    require(!opened, "failed duplicate build became readable");
}

void test_missing_id_is_atomic_for_output() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 5;
    const std::array<std::uint64_t, 3> ids{11, 22, 33};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);
    auto storage = throwdb::Storage::open(store_path);

    const std::array<std::uint64_t, 3> query{11, 999, 22};
    std::vector<std::uint16_t> output(
        query.size() * dimension, static_cast<std::uint16_t>(0xA55A));
    const auto before = output;
    bool rejected = false;
    try {
        storage->get_into(
            query.data(),
            query.size(),
            output.data(),
            output.size() * sizeof(std::uint16_t));
    } catch (const throwdb::MissingIdError& error) {
        rejected = true;
        require_equal(error.id(), std::uint64_t{999}, "wrong missing ID reported");
    }
    require(rejected, "missing ID did not fail get_into()");
    require_equal(output, before, "failed lookup partially modified output");
}

void test_append_rejects_mismatched_payload_size() {
    TempDirectory temporary;
    throwdb::WriterOptions options;
    options.dimension = 4;
    auto writer = throwdb::Writer::create(temporary.path() / "store", options);
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::array<std::uint16_t, 4> one_record{};

    bool rejected = false;
    try {
        writer->append(ids, std::as_bytes(std::span(one_record)));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "append accepted a payload with the wrong byte count");
    require_equal(writer->count(), std::uint64_t{0}, "failed append changed count");
}

void test_truncated_shard_is_reported_as_corruption() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    const std::array<std::uint64_t, 3> ids{101, 202, 303};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto shard = find_regular_file(store_path, "shard-");
    const auto original_size = std::filesystem::file_size(shard);
    require(original_size > 0, "shard is unexpectedly empty");
    std::filesystem::resize_file(shard, original_size - 1);

    bool rejected = false;
    try {
        auto storage = throwdb::Storage::open(store_path);
        static_cast<void>(storage->get(ids));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "truncated shard was silently accepted");
}

void test_corrupt_manifest_magic_is_rejected() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 2;
    const std::array<std::uint64_t, 1> ids{77};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto manifest = find_regular_file(store_path, "manifest");
    std::fstream stream(manifest, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "could not open manifest for corruption test");
    const char invalid = '\0';
    stream.write(&invalid, 1);
    stream.flush();
    stream.close();

    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "manifest with invalid magic was silently accepted");
}

void test_truncated_index_is_reported_as_corruption() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    const std::array<std::uint64_t, 3> ids{41, 42, 43};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto index = find_regular_file(store_path, "index.bin");
    const auto original_size = std::filesystem::file_size(index);
    require(original_size > 4096, "index is unexpectedly small");
    std::filesystem::resize_file(index, original_size - 8);

    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "truncated index was silently accepted");
}

void test_internal_io_worker_counts_with_repeated_cross_shard_ids() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 17;
    constexpr std::size_t row_count = 67;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 1'000'003ULL + 97ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 3 * record_size);

    const std::array<std::size_t, 16> positions{
        66, 0, 31, 4, 65, 5, 17, 1, 48, 31, 2, 64, 17, 9, 40, 0};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }

    for (const std::uint32_t workers : {0U, 1U, 2U, 4U, 8U}) {
        throwdb::ReaderOptions options;
        options.max_read_range_bytes = record_size;
        options.io_workers = workers;
        // An explicit queue smaller than some worker counts exercises
        // submit-side backpressure. Zero workers is the sequential baseline.
        options.io_queue_capacity = workers == 0 ? 0 : 2;
        options.io_ranges_per_task = 8;
        auto storage = throwdb::Storage::open(store_path, options);
        require_equal(
            storage->get(query), expected,
            "I/O worker count changed cross-shard query results");
    }

    for (const std::uint32_t ranges_per_task : {1U, 4U, 8U, 16U}) {
        throwdb::ReaderOptions options;
        options.max_read_range_bytes = record_size;
        options.io_workers = 4;
        options.io_queue_capacity = 2;
        options.io_ranges_per_task = ranges_per_task;
        auto storage = throwdb::Storage::open(store_path, options);
        require_equal(
            storage->get(query), expected,
            "ranges-per-task changed repeated cross-shard query results");
    }
}

void test_chunked_executor_adapts_to_small_batches() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 11;
    constexpr std::size_t row_count = 37;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 999'983ULL + 53ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 2 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    // One row exercises the public result of the direct fast path. The other
    // sizes straddle both the task-width and worker-count partition boundaries.
    for (const std::size_t query_size : {1U, 2U, 3U, 4U, 7U, 8U, 9U, 15U, 16U, 17U}) {
        std::vector<std::uint64_t> query;
        std::vector<std::uint16_t> expected;
        query.reserve(query_size);
        expected.reserve(query_size * dimension);
        for (std::size_t index = 0; index != query_size; ++index) {
            const auto position = (index * 5 + 3) % row_count;
            query.push_back(ids[position]);
            const auto begin = payload.begin()
                               + static_cast<std::ptrdiff_t>(position * dimension);
            expected.insert(expected.end(), begin, begin + dimension);
        }
        require_equal(
            storage->get(query), expected,
            "adaptive chunking changed a small-batch result");
    }
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

void test_simultaneous_get_calls_share_internal_executor() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 16;
    constexpr std::size_t row_count = 64;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 65'537ULL + 11ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 4 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 2;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    constexpr std::size_t client_count = 8;
    constexpr std::size_t calls_per_client = 20;
    std::array<std::exception_ptr, client_count> errors{};
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> clients;
    clients.reserve(client_count);
    for (std::size_t client = 0; client != client_count; ++client) {
        clients.emplace_back([&, client] {
            try {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (std::size_t call = 0; call != calls_per_client; ++call) {
                    const std::array<std::size_t, 8> positions{
                        (client + call) % row_count,
                        (63 - client + call) % row_count,
                        (client * 7 + call * 3) % row_count,
                        (client + call) % row_count,
                        (32 + client + call) % row_count,
                        (5 + client * 11 + call) % row_count,
                        (48 + call) % row_count,
                        (client * 7 + call * 3) % row_count,
                    };
                    std::vector<std::uint64_t> query;
                    std::vector<std::uint16_t> expected;
                    for (const auto position : positions) {
                        query.push_back(ids[position]);
                        const auto begin = payload.begin()
                                           + static_cast<std::ptrdiff_t>(
                                               position * dimension);
                        expected.insert(expected.end(), begin, begin + dimension);
                    }
                    require_equal(
                        storage->get(query), expected,
                        "simultaneous get returned incorrect rows");
                }
            } catch (...) {
                errors[client] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != client_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& client : clients) {
        client.join();
    }
    for (const auto& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
}

void test_parallel_read_error_is_propagated_and_executor_survives() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::size_t row_count = 12;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(500 + row * 13);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 2 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    const auto last_shard = store_path / "data" / "shard-000005.dat";
    std::filesystem::resize_file(last_shard, 4096);
    bool rejected = false;
    try {
        static_cast<void>(storage->get(ids));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "parallel shard read error was not propagated");

    // Two rows in different healthy shards cannot coalesce and therefore
    // force the recovery read through the executor instead of the one-range
    // direct fast path.
    const std::array<std::uint64_t, 2> healthy_query{ids[0], ids[2]};
    const auto healthy = storage->get(healthy_query);
    std::vector<std::uint16_t> healthy_expected;
    healthy_expected.insert(
        healthy_expected.end(), payload.begin(), payload.begin() + dimension);
    healthy_expected.insert(
        healthy_expected.end(), payload.begin() + 2 * dimension,
        payload.begin() + 3 * dimension);
    require_equal(
        healthy,
        healthy_expected,
        "executor was unusable after a task error");
}

void test_invalid_io_executor_options_are_rejected() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 2;
    const std::array<std::uint64_t, 1> ids{1};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    throwdb::ReaderOptions options;
    options.io_workers = 0;
    options.io_queue_capacity = 1;
    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "queue capacity without I/O workers was accepted");

    options = {};
    options.io_ranges_per_task = 0;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "zero ranges per executor task was accepted");

    options = {};
    options.io_ranges_per_task = (1U << 20) + 1U;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "excessive ranges per executor task was accepted");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 15> tests{{
        {"round_trip_arbitrary_ids_and_order", test_round_trip_arbitrary_ids_and_order},
        {"multiple_append_batches_and_shards", test_multiple_append_batches_and_shards},
        {"duplicate_id_rejects_publication", test_duplicate_id_rejects_publication},
        {"missing_id_is_atomic_for_output", test_missing_id_is_atomic_for_output},
        {"append_rejects_mismatched_payload_size", test_append_rejects_mismatched_payload_size},
        {"truncated_shard_is_reported_as_corruption", test_truncated_shard_is_reported_as_corruption},
        {"corrupt_manifest_magic_is_rejected", test_corrupt_manifest_magic_is_rejected},
        {"truncated_index_is_reported_as_corruption", test_truncated_index_is_reported_as_corruption},
        {"internal_io_worker_counts_with_repeated_cross_shard_ids", test_internal_io_worker_counts_with_repeated_cross_shard_ids},
        {"chunked_executor_adapts_to_small_batches", test_chunked_executor_adapts_to_small_batches},
        {"range_task_partition_is_balanced_and_bounded", test_range_task_partition_is_balanced_and_bounded},
        {"simultaneous_get_calls_share_internal_executor", test_simultaneous_get_calls_share_internal_executor},
        {"parallel_read_error_is_propagated_and_executor_survives", test_parallel_read_error_is_propagated_and_executor_survives},
        {"invalid_io_executor_options_are_rejected", test_invalid_io_executor_options_are_rejected},
        {"empty_query_is_valid", [] {
             TempDirectory temporary;
             constexpr std::uint32_t dimension = 3;
             const std::array<std::uint64_t, 1> ids{1};
             const auto payload = make_payload(ids.size(), dimension);
             build_store(temporary.path() / "store", ids, payload, dimension, 1U << 20);
             auto storage = throwdb::Storage::open(temporary.path() / "store");
             const std::span<const std::uint64_t> empty;
             require(storage->get(empty).empty(), "empty query returned data");
         }},
    }};

    std::size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
