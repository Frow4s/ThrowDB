#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/native_format.hpp"

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-native-collection-test-"
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

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t row_count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> result(row_count * record_size);
    for (std::size_t row = 0; row != row_count; ++row) {
        for (std::size_t column = 0; column != record_size; ++column) {
            result[row * record_size + column] = static_cast<std::byte>(
                salt + static_cast<std::uint8_t>(row * 29 + column * 7));
        }
    }
    return result;
}

void require_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::byte> expected,
    const std::string& message) {
    require(
        actual.size() == expected.size()
            && std::equal(actual.begin(), actual.end(), expected.begin()),
        message);
}

void remove_index_body_checksum(const std::filesystem::path& store) {
    using namespace throwdb::detail;
    const auto path = native_index_path(store);
    std::array<std::byte, kNativeHeaderSize> encoded{};
    {
        std::ifstream stream(path, std::ios::binary);
        require(stream.good(), "cannot open index header for reading");
        stream.read(
            reinterpret_cast<char*>(encoded.data()),
            static_cast<std::streamsize>(encoded.size()));
        require(
            stream.gcount() == static_cast<std::streamsize>(encoded.size()),
            "cannot read complete index header");
    }
    NativeIndexHeader header = decode_native_index_header(encoded);
    header.body_checksum_algorithm = 0;
    header.body_checksum = 0;
    encoded = encode_native_index_header(header);
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open index header for legacy rewrite");
    stream.write(
        reinterpret_cast<const char*>(encoded.data()),
        static_cast<std::streamsize>(encoded.size()));
    stream.close();
    require(stream.good(), "cannot rewrite legacy index header");
}

void test_multisegment_crud_snapshot_and_reopen() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 5> base_ids{10, 20, 30, 40, 50};
    const std::vector<std::byte> base = rows(base_ids.size(), record_size, 3);

    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 2 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(
        std::span(base_ids).first(3),
        std::span(base).first(3 * record_size));
    builder->append(
        std::span(base_ids).subspan(3),
        std::span(base).subspan(3 * record_size));
    require(builder->count() == base_ids.size(), "builder count is wrong");
    require(builder->format_version() == 6, "native public version is wrong");
    builder->finish();

    auto old_snapshot = throwdb::NativeCollection::open(store);
    const std::array<std::uint64_t, 4> base_query{50, 10, 30, 10};
    const std::vector<std::byte> base_output = old_snapshot->get_bytes(base_query);
    std::vector<std::byte> expected_base;
    for (const std::size_t row : {4U, 0U, 2U, 0U}) {
        const auto source = std::span(base).subspan(row * record_size, record_size);
        expected_base.insert(expected_base.end(), source.begin(), source.end());
    }
    require_equal(base_output, expected_base, "base lookup is not bit exact");

    std::array<std::byte, 2 * record_size> untouched{};
    untouched.fill(std::byte{0x5a});
    const auto before = untouched;
    const std::array<std::uint64_t, 2> missing_query{10, 999};
    require_throws<throwdb::MissingIdError>(
        [&] { old_snapshot->get_into(missing_query, untouched); },
        "missing base ID was accepted");
    require(untouched == before, "missing lookup modified caller output");

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require(writer->writable(), "read-write handle is not writable");
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(
                throwdb::NativeCollection::open(store, {}, true));
        },
        "a second writer lease was acquired");

    const std::array<std::uint64_t, 2> insert_ids{60, 70};
    const std::vector<std::byte> inserted = rows(2, record_size, 71);
    writer->insert(insert_ids, inserted);
    require(writer->count() == 7, "insert did not increase count");

    require_throws<throwdb::MissingIdError>(
        [&] { static_cast<void>(old_snapshot->get_bytes(insert_ids)); },
        "old snapshot observed a later commit");

    const std::array<std::uint64_t, 2> update_ids{20, 40};
    const std::vector<std::byte> updated = rows(2, record_size, 111);
    writer->update(update_ids, updated);
    require(writer->count() == 7, "update changed count");

    const std::array<std::uint64_t, 2> upsert_ids{10, 80};
    const std::vector<std::byte> upserted = rows(2, record_size, 151);
    writer->upsert(upsert_ids, upserted);
    require(writer->count() == 8, "upsert count is wrong");

    const std::array<std::uint64_t, 3> delete_ids{30, 999, 70};
    const std::vector<std::uint8_t> deleted = writer->erase(delete_ids);
    require(
        deleted == std::vector<std::uint8_t>({1, 0, 1}),
        "delete existence results are wrong");
    require(writer->count() == 6, "delete count is wrong");
    require(
        writer->erase(delete_ids) == std::vector<std::uint8_t>({0, 0, 0}),
        "delete is not idempotent");

    const std::array<std::uint64_t, 2> duplicate_ids{90, 90};
    const std::vector<std::byte> duplicate_rows = rows(2, record_size, 191);
    require_throws<throwdb::DuplicateIdError>(
        [&] { writer->upsert(duplicate_ids, duplicate_rows); },
        "duplicate upsert batch was accepted");
    require(writer->count() == 6, "failed duplicate batch changed count");

    const std::array<std::uint64_t, 2> existing_insert{90, 20};
    require_throws<throwdb::ExistingIdError>(
        [&] { writer->insert(existing_insert, duplicate_rows); },
        "insert accepted an existing ID");
    require_throws<throwdb::MissingIdError>(
        [&] { static_cast<void>(writer->get_bytes(std::array<std::uint64_t, 1>{90})); },
        "failed insert partially committed");

    const std::array<std::uint64_t, 2> missing_update{20, 999};
    require_throws<throwdb::MissingIdError>(
        [&] { writer->update(missing_update, duplicate_rows); },
        "update accepted a missing ID");
    writer.reset();
    old_snapshot.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require(reopened->count() == 6, "recovered count is wrong");
    const std::array<std::uint64_t, 6> final_ids{80, 10, 60, 40, 20, 50};
    const std::vector<std::byte> actual = reopened->get_bytes(final_ids);
    std::vector<std::byte> expected;
    const auto append_row = [&expected](
                                const std::vector<std::byte>& source,
                                const std::size_t row) {
        const auto bytes = std::span(source).subspan(
            row * record_size, record_size);
        expected.insert(expected.end(), bytes.begin(), bytes.end());
    };
    append_row(upserted, 1);
    append_row(upserted, 0);
    append_row(inserted, 0);
    append_row(updated, 1);
    append_row(updated, 0);
    append_row(base, 4);
    require_equal(actual, expected, "reopened CRUD state is wrong");
}

void test_open_stats_fast_forced_and_legacy_validation() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    throwdb::NativeBuilderOptions options;
    options.dimension = 2;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    const std::array<std::uint64_t, 2> ids{7, 11};
    const std::vector<std::byte> embeddings = rows(2, 4, 29);
    builder->append(ids, embeddings);
    builder->finish();

    {
        auto reader = throwdb::NativeCollection::open(store);
        require(reader->format_version() == 6,
                "delta collection did not report version 6");
        const throwdb::NativeOpenStats stats = reader->open_stats();
        require(stats.total_seconds > 0.0,
                "open diagnostics have no total duration");
        require(stats.index_file_bytes == std::filesystem::file_size(
                    throwdb::detail::native_index_path(store)),
                "open diagnostics report the wrong index size");
        require(stats.index_body_checksum_present,
                "normal open did not report the index checksum");
        require(stats.index_body_checksum_verified,
                "normal open did not verify the index checksum");
        require(!stats.index_structural_validation_performed,
                "normal open unexpectedly ran the full structural scan");
    }

    {
        throwdb::ReaderOptions reader_options;
        reader_options.verify_native_index = true;
        auto reader = throwdb::NativeCollection::open(
            store, reader_options);
        const throwdb::NativeOpenStats stats = reader->open_stats();
        require(stats.index_body_checksum_verified,
                "forced open did not verify the checksum");
        require(stats.index_structural_validation_performed,
                "forced open did not perform structural verification");
    }

    remove_index_body_checksum(store);
    auto legacy = throwdb::NativeCollection::open(store);
    require(legacy->format_version() == 6,
            "catalog collection changed version with index validation mode");
    const throwdb::NativeOpenStats legacy_stats = legacy->open_stats();
    require(!legacy_stats.index_body_checksum_present,
            "legacy open invented an index checksum");
    require(legacy_stats.index_structural_validation_performed,
            "legacy open skipped its full-validation fallback");
}

void test_active_wal_tail_is_ignored_and_writer_repairs_it() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 3;
    constexpr std::uint64_t record_size = dimension * 4;

    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float32;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    const std::array<std::uint64_t, 1> base_id{1};
    const std::vector<std::byte> base = rows(1, record_size, 17);
    builder->append(base_id, base);
    builder->finish();

    {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        const std::array<std::uint64_t, 1> inserted_id{2};
        const std::vector<std::byte> inserted = rows(1, record_size, 99);
        writer->insert(inserted_id, inserted);
    }

    std::vector<std::filesystem::path> wal_paths;
    for (const auto& entry : std::filesystem::directory_iterator(store / "wal")) {
        if (entry.is_regular_file() && entry.path().extension() == ".wal") {
            wal_paths.push_back(entry.path());
        }
    }
    require(wal_paths.size() == 1, "fresh collection has an unexpected WAL set");
    const auto wal = wal_paths.front();
    const auto committed_size = std::filesystem::file_size(wal);
    {
        std::ofstream stream(wal, std::ios::binary | std::ios::app);
        const std::array<char, 13> torn_tail{
            't', 'o', 'r', 'n', '-', 'w', 'a', 'l', '-', 't', 'a', 'i', 'l'};
        stream.write(torn_tail.data(), torn_tail.size());
        if (!stream) {
            fail("cannot append synthetic WAL tail");
        }
    }
    require(
        std::filesystem::file_size(wal) == committed_size + 13,
        "synthetic WAL tail was not appended");

    {
        auto reader = throwdb::NativeCollection::open(store);
        require(reader->count() == 2, "reader did not ignore active WAL tail");
        require(
            std::filesystem::file_size(wal) == committed_size + 13,
            "read-only open unexpectedly repaired WAL");
    }
    {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        require(writer->count() == 2, "writer recovery lost committed state");
        require(
            std::filesystem::file_size(wal) == committed_size,
            "writer did not truncate ignored active WAL tail");
    }
}

void test_zero_delta_read_path_transitions_on_first_commit() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::size_t row_count = 512;

    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = 10'000 + index;
    }
    const std::vector<std::byte> base = rows(row_count, record_size, 13);
    const std::vector<std::byte> updated = rows(row_count, record_size, 149);

    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.segment_size_bytes = 64 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(ids, base);
    builder->finish();

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require_equal(
        writer->get_bytes(ids), base,
        "zero-delta base-only read is not bit exact");

    std::atomic<bool> start{false};
    std::atomic<bool> finished{false};
    std::exception_ptr update_error;
    std::thread updater([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            writer->update(ids, updated);
        } catch (...) {
            update_error = std::current_exception();
        }
        finished.store(true, std::memory_order_release);
    });

    start.store(true, std::memory_order_release);
    std::size_t concurrent_reads = 0;
    bool valid_concurrent_snapshot = true;
    std::exception_ptr read_error;
    try {
        do {
            const std::vector<std::byte> actual = writer->get_bytes(ids);
            const bool complete_base =
                std::equal(actual.begin(), actual.end(), base.begin());
            const bool complete_update =
                std::equal(actual.begin(), actual.end(), updated.begin());
            if (!complete_base && !complete_update) {
                valid_concurrent_snapshot = false;
                break;
            }
            ++concurrent_reads;
        } while (!finished.load(std::memory_order_acquire));
    } catch (...) {
        read_error = std::current_exception();
    }
    updater.join();
    if (read_error != nullptr) {
        std::rethrow_exception(read_error);
    }
    if (update_error != nullptr) {
        std::rethrow_exception(update_error);
    }
    require(concurrent_reads != 0, "concurrent read loop did not execute");
    require(
        valid_concurrent_snapshot,
        "first commit produced a mixed or torn read snapshot");
    require_equal(
        writer->get_bytes(ids), updated,
        "read path did not switch to the committed mutation overlay");

    writer.reset();
    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(
        reopened->get_bytes(ids), updated,
        "WAL recovery did not activate the mutation read path");
}

void test_empty_collection_and_empty_batches() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    throwdb::NativeBuilderOptions options;
    options.dimension = 2;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append({}, {});
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    collection->insert({}, {});
    collection->update({}, {});
    collection->upsert({}, {});
    require(collection->erase({}).empty(), "empty delete result is not empty");
    require(collection->get_bytes({}).empty(), "empty get result is not empty");
    require(collection->count() == 0, "empty batches changed count");
}

void test_read_only_open_does_not_require_writable_lock_metadata() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    throwdb::NativeBuilderOptions options;
    options.dimension = 2;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->finish();

    const auto locks = store / "locks";
    const auto writer_lock = locks / "writer.lock";
    const auto commit_lock = locks / "commit.lock";
    require(::chmod(writer_lock.c_str(), 0444) == 0, "cannot chmod writer lock");
    require(::chmod(commit_lock.c_str(), 0444) == 0, "cannot chmod commit lock");
    require(::chmod(locks.c_str(), 0555) == 0, "cannot chmod lock directory");
    {
        auto reader = throwdb::NativeCollection::open(store);
        require(reader->count() == 0, "read-only metadata open changed state");
    }
    require(::chmod(locks.c_str(), 0755) == 0, "cannot restore lock directory");
    require(::chmod(writer_lock.c_str(), 0644) == 0, "cannot restore writer lock");
    require(::chmod(commit_lock.c_str(), 0644) == 0, "cannot restore commit lock");

    require(std::filesystem::remove(writer_lock), "cannot remove writer lock");
    require_throws<throwdb::StorageError>(
        [&] { static_cast<void>(throwdb::NativeCollection::open(store)); },
        "open recreated a missing lock file");
    require(
        !std::filesystem::exists(writer_lock),
        "read-only open recreated missing metadata");

    const auto second_store = temporary.path() / "non-regular";
    auto second_builder =
        throwdb::NativeCollectionBuilder::create(second_store, options);
    second_builder->finish();
    const auto second_writer_lock = second_store / "locks" / "writer.lock";
    require(
        std::filesystem::remove(second_writer_lock),
        "cannot remove second writer lock");
    require(
        std::filesystem::create_directory(second_writer_lock),
        "cannot replace lock with a directory");
    require_throws<throwdb::StorageError>(
        [&] { static_cast<void>(throwdb::NativeCollection::open(second_store)); },
        "open accepted a non-regular lock path");
}

void test_forked_builder_and_writer_handles_are_rejected() {
    TempDirectory temporary;
    constexpr std::uint32_t dimension = 2;
    constexpr std::uint64_t record_size = dimension * 2;

    const auto builder_store = temporary.path() / "builder";
    throwdb::NativeBuilderOptions builder_options;
    builder_options.dimension = dimension;
    auto builder = throwdb::NativeCollectionBuilder::create(
        builder_store, builder_options);
    const std::array<std::uint64_t, 1> builder_id{41};
    const std::vector<std::byte> builder_row = rows(1, record_size, 7);

    pid_t child = ::fork();
    require(child >= 0, "fork failed for builder regression");
    if (child == 0) {
        int result = 10;
        try {
            builder->append(builder_id, builder_row);
        } catch (const throwdb::StorageError& error) {
            result = std::string(error.what()).find("fork") != std::string::npos
                ? 0
                : 11;
        } catch (...) {
            result = 12;
        }
        builder.reset();
        ::_exit(result);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(
        waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "forked builder was not rejected or did not tear down safely");
    bool parent_builder_lease_preserved = false;
    try {
        static_cast<void>(
            throwdb::NativeCollection::inspect_orphans(builder_store));
    } catch (const throwdb::StorageError& error) {
        parent_builder_lease_preserved =
            std::string(error.what()).find("another writer")
            != std::string::npos;
    }
    require(
        parent_builder_lease_preserved,
        "forked child teardown released the parent builder lease");
    builder->append(builder_id, builder_row);
    builder->finish();

    const auto store = temporary.path() / "writer";
    auto finished_builder =
        throwdb::NativeCollectionBuilder::create(store, builder_options);
    finished_builder->finish();
    throwdb::ReaderOptions reader_options;
    reader_options.io_workers = 2;
    auto writer = throwdb::NativeCollection::open(
        store, reader_options, true);
    const std::array<std::uint64_t, 1> child_id{2};
    const std::array<std::uint64_t, 1> parent_id{3};
    const std::vector<std::byte> child_row = rows(1, record_size, 31);
    const std::vector<std::byte> parent_row = rows(1, record_size, 63);

    int child_ready[2]{};
    int child_release[2]{};
    require(::pipe(child_ready) == 0, "cannot create child-ready pipe");
    require(::pipe(child_release) == 0, "cannot create child-release pipe");
    child = ::fork();
    require(child >= 0, "fork failed for writer regression");
    if (child == 0) {
        static_cast<void>(::close(child_ready[0]));
        static_cast<void>(::close(child_release[1]));
        int result = 20;
        try {
            writer->insert(child_id, child_row);
        } catch (const throwdb::StorageError& error) {
            result = std::string(error.what()).find("fork") != std::string::npos
                ? 0
                : 21;
        } catch (...) {
            result = 22;
        }
        // This used to risk unlocking the parent's shared flock description
        // and destroying inherited joinable std::thread objects.
        writer.reset();
        const char ready = 'R';
        if (::write(child_ready[1], &ready, 1) != 1) {
            result = 23;
        }
        char release = 0;
        if (::read(child_release[0], &release, 1) != 1 || release != 'X') {
            result = 24;
        }
        static_cast<void>(::close(child_ready[1]));
        static_cast<void>(::close(child_release[0]));
        ::_exit(result);
    }
    static_cast<void>(::close(child_ready[1]));
    static_cast<void>(::close(child_release[0]));
    char ready = 0;
    const bool child_closed_handle =
        ::read(child_ready[0], &ready, 1) == 1 && ready == 'R';

    bool parent_lease_preserved = false;
    bool successor_opened_while_child_alive = false;
    std::exception_ptr parent_error;
    try {
        try {
            static_cast<void>(
                throwdb::NativeCollection::open(store, {}, true));
        } catch (const throwdb::StorageError&) {
            parent_lease_preserved = true;
        }
        writer->insert(parent_id, parent_row);
        writer.reset();

        // The child is deliberately still alive. Its close must have dropped
        // inherited fd references so it cannot prolong the parent's lease.
        auto successor = throwdb::NativeCollection::open(store, {}, true);
        successor_opened_while_child_alive = true;
        successor.reset();
    } catch (...) {
        parent_error = std::current_exception();
    }

    if (child_closed_handle) {
        const char release = 'X';
        static_cast<void>(::write(child_release[1], &release, 1));
    }
    static_cast<void>(::close(child_ready[0]));
    static_cast<void>(::close(child_release[1]));
    status = 0;
    waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (parent_error != nullptr) {
        std::rethrow_exception(parent_error);
    }
    require(child_closed_handle, "forked child did not close its handle");
    require(
        waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "forked writer was not rejected or did not tear down safely");
    require(
        parent_lease_preserved,
        "child teardown released the still-open parent's writer lease");
    require(
        successor_opened_while_child_alive,
        "closed child handle retained the writer lease until child exit");

    auto reopened = throwdb::NativeCollection::open(store);
    require(reopened->count() == 1, "fork regression changed committed count");
    require_equal(
        reopened->get_bytes(parent_id), parent_row,
        "parent commit did not survive fork regression");
    require_throws<throwdb::MissingIdError>(
        [&] { static_cast<void>(reopened->get_bytes(child_id)); },
        "rejected child mutation reached the WAL");
}

void test_native_open_rejects_big_endian_hosts() {
    if constexpr (std::endian::native != std::endian::little) {
        require_throws<throwdb::InvalidArgumentError>(
            [] {
                static_cast<void>(throwdb::NativeCollection::open(
                    "/path-does-not-matter-on-big-endian"));
            },
            "native collection open accepted a big-endian host");
    }
}

void test_lightweight_metadata_inspection() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    throwdb::NativeBuilderOptions options;
    options.dimension = 7;
    options.dtype = throwdb::DType::Float32;
    options.segment_size_bytes = 32 * 1024;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->finish();

    const throwdb::NativeCollectionMetadata metadata =
        throwdb::NativeCollection::inspect_metadata(store);
    require(metadata.format_version == 6, "metadata format version is wrong");
    require(metadata.dimension == options.dimension,
            "metadata dimension is wrong");
    require(metadata.dtype == options.dtype, "metadata dtype is wrong");
    require(metadata.record_size == 7 * sizeof(float),
            "metadata record size is wrong");
    require(metadata.segment_size_bytes
                == (options.segment_size_bytes / metadata.record_size)
                       * metadata.record_size,
            "metadata segment size is wrong");
    require(metadata.layout == "uint64_dense_row_major_flat_v1",
            "metadata layout discriminator is wrong");
    require(metadata.feature_discriminator == "catalog_v1+delta_v1",
            "metadata feature discriminator is wrong");
    require(metadata.required_feature_bits == 3,
            "metadata required features are wrong");
    require(metadata.optional_feature_bits == 0,
            "metadata optional features are wrong");
}

void test_reader_trials_and_bounded_id_sampling() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    std::vector<std::uint64_t> ids;
    ids.reserve(32);
    for (std::uint64_t index = 0; index != 32; ++index) {
        ids.push_back(1000 + index * 17);
    }
    const std::vector<std::byte> embeddings =
        rows(ids.size(), record_size, 23);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.dimension = dimension;
    builder_options.segment_size_bytes = 7 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    throwdb::ReaderOptions trial_options;
    trial_options.io_workers = 2;
    trial_options.io_queue_capacity = 4;
    auto trial = collection->create_reader_trial(trial_options);
    const std::array<std::uint64_t, 6> query{
        ids[17], ids[2], ids[29], ids[2], ids[0], ids[11]};
    const std::vector<std::byte> expected = collection->get_bytes(query);
    std::vector<std::byte> actual(expected.size());
    collection->get_into_with_reader(*trial, query, actual);
    require_equal(actual, expected, "reader trial is not bit exact");
    std::fill(actual.begin(), actual.end(), std::byte{0});
    const throwdb::NativeReadStats stats =
        collection->get_into_with_reader_profiled(*trial, query, actual);
    require_equal(actual, expected, "profiled reader trial is not bit exact");
    require(stats.requested_rows == query.size(),
            "reader trial profiled the wrong row count");
    require(stats.base_rows == query.size(),
            "reader trial did not resolve base rows");

    throwdb::ReaderOptions invalid_options;
    invalid_options.max_read_range_bytes = record_size - 1;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                collection->create_reader_trial(invalid_options));
        },
        "reader trial accepted a range smaller than one row");

    const throwdb::NativeIdSample first =
        collection->sample_ids(12, 0x12345678ULL, 128);
    const throwdb::NativeIdSample repeated =
        collection->sample_ids(12, 0x12345678ULL, 128);
    require(first.complete, "bounded sample did not collect requested IDs");
    require(first.ids == repeated.ids,
            "bounded sample is not deterministic");
    require(first.inspected_slots == repeated.inspected_slots,
            "bounded sample inspection count is not deterministic");
    require(first.inspected_slots <= 128,
            "bounded sample exceeded its inspection limit");

    const throwdb::NativeIdSample capped =
        collection->sample_ids(12, 0x12345678ULL, 3);
    require(capped.inspected_slots == 3,
            "incomplete sample did not honor its hard slot cap");
    require(!capped.complete,
            "slot-capped sample unexpectedly reported completion");
    require(capped.ids.size() <= capped.inspected_slots,
            "sample returned more IDs than inspected slots");
    const throwdb::NativeIdSample empty =
        collection->sample_ids(0, 0x12345678ULL, 0);
    require(empty.complete && empty.ids.empty()
                && empty.inspected_slots == 0,
            "zero-sized sample has incorrect semantics");

    const std::array<std::uint64_t, 1> deleted_id{first.ids.front()};
    require(collection->erase(deleted_id) == std::vector<std::uint8_t>{1},
            "sample fixture delete failed");
    const throwdb::NativeIdSample after_delete =
        collection->sample_ids(12, 0x12345678ULL, 128);
    require(after_delete.complete,
            "sample did not replace a deleted base ID");
    require(std::find(
                after_delete.ids.begin(), after_delete.ids.end(),
                deleted_id.front()) == after_delete.ids.end(),
            "sample returned a logically deleted base ID");
}

void test_calibration_handle_owns_exclusive_read_only_lease() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint64_t record_size = 4;
    throwdb::NativeBuilderOptions options;
    options.dimension = 2;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    const std::array<std::uint64_t, 1> ids{7};
    const std::vector<std::byte> embeddings = rows(1, record_size, 5);
    builder->append(ids, embeddings);
    builder->finish();

    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                store, {}, true, true));
        },
        "writable calibration handle was accepted");

    auto calibration = throwdb::NativeCollection::open(
        store, {}, false, true);
    require(!calibration->writable(),
            "calibration handle unexpectedly permits writes");
    require_equal(
        calibration->get_bytes(ids), embeddings,
        "calibration snapshot cannot read base data");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { calibration->upsert(ids, embeddings); },
        "calibration snapshot accepted a mutation");
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(
                throwdb::NativeCollection::open(store, {}, true));
        },
        "writer opened during calibration");
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                store, {}, false, true));
        },
        "second calibrator opened during calibration");

    // Ordinary immutable snapshots do not participate in the lifecycle
    // lease and remain available while tuning is in progress.
    auto reader = throwdb::NativeCollection::open(store);
    require_equal(reader->get_bytes(ids), embeddings,
                  "ordinary reader was disrupted by calibration");
    reader.reset();
    calibration.reset();

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require(writer->writable(),
            "calibration did not release its writer lease");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*callback)();
    };
    const std::array tests{
        TestCase{
            "multisegment_crud_snapshot_and_reopen",
            test_multisegment_crud_snapshot_and_reopen},
        TestCase{
            "open_stats_fast_forced_and_legacy_validation",
            test_open_stats_fast_forced_and_legacy_validation},
        TestCase{
            "active_wal_tail_is_ignored_and_writer_repairs_it",
            test_active_wal_tail_is_ignored_and_writer_repairs_it},
        TestCase{
            "zero_delta_read_path_transitions_on_first_commit",
            test_zero_delta_read_path_transitions_on_first_commit},
        TestCase{
            "empty_collection_and_empty_batches",
            test_empty_collection_and_empty_batches},
        TestCase{
            "read_only_open_does_not_require_writable_lock_metadata",
            test_read_only_open_does_not_require_writable_lock_metadata},
        TestCase{
            "forked_builder_and_writer_handles_are_rejected",
            test_forked_builder_and_writer_handles_are_rejected},
        TestCase{
            "native_open_rejects_big_endian_hosts",
            test_native_open_rejects_big_endian_hosts},
        TestCase{
            "lightweight_metadata_inspection",
            test_lightweight_metadata_inspection},
        TestCase{
            "reader_trials_and_bounded_id_sampling",
            test_reader_trials_and_bounded_id_sampling},
        TestCase{
            "calibration_handle_owns_exclusive_read_only_lease",
            test_calibration_handle_owns_exclusive_read_only_lease},
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
    std::cout << passed << " native-collection tests passed\n";
    return 0;
}
