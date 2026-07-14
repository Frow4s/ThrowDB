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
#include <thread>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/mutation_catalog.hpp"
#include "../src/core/mutation_delta.hpp"
#include "../src/core/mutation_wal.hpp"
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
                / ("throwdb-native-materialization-test-"
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

template <typename Actual, typename Expected>
void require_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string& message) {
    if (!(actual == expected)) {
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

void append_row(
    std::vector<std::byte>& destination,
    const std::vector<std::byte>& source,
    const std::size_t source_row,
    const std::size_t record_size) {
    const auto bytes = std::span(source).subspan(
        source_row * record_size, record_size);
    destination.insert(destination.end(), bytes.begin(), bytes.end());
}

void require_bytes_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::byte> expected,
    const std::string& message) {
    require(
        actual.size() == expected.size()
            && std::equal(actual.begin(), actual.end(), expected.begin()),
        message);
}

void build_store(
    const std::filesystem::path& store,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension,
    const std::uint64_t segment_size_bytes = 1ULL << 20) {
    throwdb::NativeBuilderOptions options;
    options.format_version = 6;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = segment_size_bytes;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    require_equal(
        builder->format_version(), std::uint32_t{6},
        "builder did not report native format 6");
    builder->append(ids, embeddings);
    builder->finish();
}

[[nodiscard]] throwdb::detail::NativeManifest manifest_for(
    const std::filesystem::path& store) {
    return throwdb::detail::read_native_manifest(store);
}

[[nodiscard]] throwdb::detail::MutationCatalog catalog_for(
    const std::filesystem::path& store) {
    const throwdb::detail::NativeManifest manifest = manifest_for(store);
    return throwdb::detail::read_current_mutation_catalog(store, manifest);
}

[[nodiscard]] std::filesystem::path wal_path_for(
    const std::filesystem::path& store,
    const throwdb::detail::MutationWalEntry& entry) {
    return throwdb::detail::mutation_wal_path(store, entry.wal_id);
}

void require_missing(
    throwdb::NativeCollection& collection,
    const std::span<const std::uint64_t> ids,
    const std::string& context) {
    for (const std::uint64_t id : ids) {
        require_throws<throwdb::MissingIdError>(
            [&collection, id] {
                static_cast<void>(collection.get_bytes(
                    std::array<std::uint64_t, 1>{id}));
            },
            context + ": deleted ID was found");
    }
}

void flip_file_byte(
    const std::filesystem::path& path,
    const std::uint64_t offset) {
    require(
        std::filesystem::file_size(path) > offset,
        "corruption offset is outside the fixture file");
    std::fstream stream(
        path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open fixture for corruption");
    stream.seekg(static_cast<std::streamoff>(offset));
    char value = 0;
    stream.read(&value, 1);
    require(stream.gcount() == 1, "cannot read fixture corruption byte");
    value = static_cast<char>(static_cast<unsigned char>(value) ^ 0x5aU);
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(&value, 1);
    stream.close();
    require(stream.good(), "cannot write fixture corruption byte");
}

void test_mixed_lifecycle_materializes_and_reopens_bit_exactly() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::uint64_t maximum_id =
        std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t almost_maximum_id = maximum_id - 1;

    const std::array<std::uint64_t, 6> base_ids{
        0, 1, 42, maximum_id, 7, 99};
    const std::vector<std::byte> base = rows(6, record_size, 1);
    build_store(store, base_ids, base, dimension);

    const std::array<std::uint64_t, 2> first_update_ids{0, 42};
    const std::vector<std::byte> first_updates = rows(2, record_size, 31);
    const std::array<std::uint64_t, 2> first_delete_ids{1, maximum_id};
    const std::array<std::uint64_t, 2> first_insert_ids{maximum_id, 100};
    const std::vector<std::byte> first_inserts = rows(2, record_size, 51);

    const std::array<std::uint64_t, 2> second_update_ids{0, 100};
    const std::vector<std::byte> second_updates = rows(2, record_size, 71);
    const std::array<std::uint64_t, 2> second_delete_ids{maximum_id, 7};
    const std::array<std::uint64_t, 2> second_upsert_ids{
        42, almost_maximum_id};
    const std::vector<std::byte> second_upserts = rows(2, record_size, 91);
    const std::vector<std::byte> frozen_reinsert_42 =
        rows(1, record_size, 111);

    const std::array<std::uint64_t, 2> active_update_ids{0, 42};
    const std::vector<std::byte> active_updates = rows(2, record_size, 131);
    const std::vector<std::byte> active_insert_1 =
        rows(1, record_size, 151);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    writer->update(first_update_ids, first_updates);
    require_equal(
        writer->erase(first_delete_ids),
        std::vector<std::uint8_t>({1, 1}),
        "first frozen WAL deletes returned the wrong result");
    writer->insert(first_insert_ids, first_inserts);
    require(writer->rotate_wal().rotated,
            "first non-empty WAL did not rotate");

    writer->update(second_update_ids, second_updates);
    require_equal(
        writer->erase(second_delete_ids),
        std::vector<std::uint8_t>({1, 1}),
        "second frozen WAL deletes returned the wrong result");
    writer->upsert(second_upsert_ids, second_upserts);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{42}),
        std::vector<std::uint8_t>({1}),
        "second frozen WAL did not delete ID 42");
    writer->insert(
        std::array<std::uint64_t, 1>{42}, frozen_reinsert_42);
    require(writer->rotate_wal().rotated,
            "second non-empty WAL did not rotate");

    // These mutations stay in the active WAL. They must remain logically
    // newer than the delta produced from the two frozen predecessors.
    writer->update(active_update_ids, active_updates);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{100}),
        std::vector<std::uint8_t>({1}),
        "active WAL did not delete ID 100");
    writer->insert(std::array<std::uint64_t, 1>{1}, active_insert_1);

    const std::array<std::uint64_t, 5> live_ids{
        0, 1, 42, 99, almost_maximum_id};
    std::vector<std::byte> expected;
    append_row(expected, active_updates, 0, record_size);
    append_row(expected, active_insert_1, 0, record_size);
    append_row(expected, active_updates, 1, record_size);
    append_row(expected, base, 5, record_size);
    append_row(expected, second_upserts, 1, record_size);
    const std::array<std::uint64_t, 3> missing_ids{7, 100, maximum_id};

    require_equal(writer->count(), std::uint64_t{5},
                  "mixed fixture has the wrong live count");
    require_bytes_equal(
        writer->get_bytes(live_ids), expected,
        "mixed fixture is not bit exact before materialization");
    require_missing(*writer, missing_ids, "before materialization");

    const throwdb::NativeMutationLogStats before =
        writer->mutation_log_stats();
    require(before.catalog_managed && before.delta_managed,
            "format 6 stats do not report catalog and delta management");
    require_equal(before.generation, std::uint64_t{3},
                  "two rotations produced the wrong generation");
    require_equal(before.committed_sequence, std::uint64_t{11},
                  "mixed fixture has the wrong committed sequence");
    require_equal(before.materialized_sequence, std::uint64_t{0},
                  "fresh fixture invented a materialized sequence");
    require_equal(before.frozen_wal_count, std::uint64_t{2},
                  "mixed fixture has the wrong frozen WAL count");
    require_equal(before.delta_run_count, std::uint64_t{0},
                  "mixed fixture contains a delta before materialization");
    require_equal(before.active_wal_put_rows, std::uint64_t{3},
                  "mixed fixture active WAL PUT count is wrong");

    const throwdb::detail::MutationCatalog before_catalog =
        catalog_for(store);
    require_equal(before_catalog.wal_entries.size(), std::size_t{3},
                  "mixed fixture catalog has the wrong WAL count");
    std::vector<std::filesystem::path> frozen_paths;
    std::uint64_t frozen_bytes = 0;
    for (std::size_t index = 0; index != 2; ++index) {
        require_equal(
            before_catalog.wal_entries[index].state,
            throwdb::detail::MutationWalState::Frozen,
            "mixed fixture WAL prefix is not frozen");
        frozen_paths.push_back(
            wal_path_for(store, before_catalog.wal_entries[index]));
        require(std::filesystem::exists(frozen_paths.back()),
                "referenced frozen WAL is missing before materialization");
        frozen_bytes += before_catalog.wal_entries[index].sealed_file_size;
    }
    const std::filesystem::path active_path =
        wal_path_for(store, before_catalog.wal_entries.back());

    const throwdb::NativeMutationLogStats materialized =
        writer->materialize_frozen_wals();
    require(materialized.materialized,
            "non-empty frozen prefix was treated as a no-op");
    require_equal(materialized.generation, before.generation + 2,
                  "materialization did not publish reservation and final generations");
    require_equal(materialized.committed_sequence, std::uint64_t{11},
                  "materialization changed the committed sequence");
    require_equal(materialized.materialized_sequence, std::uint64_t{8},
                  "materialization stopped at the wrong sequence");
    require_equal(materialized.frozen_wal_count, std::uint64_t{0},
                  "materialization retained a selected frozen WAL");
    require_equal(materialized.delta_run_count, std::uint64_t{1},
                  "materialization did not create one delta run");
    require_equal(materialized.delta_entry_count, std::uint64_t{7},
                  "multiple frozen mutations were not collapsed per ID");
    require_equal(materialized.delta_put_rows, std::uint64_t{4},
                  "collapsed delta has the wrong PUT count");
    require(materialized.total_delta_bytes > 0,
            "materialized delta reports zero bytes");
    require_equal(materialized.reclaimed_wal_count, std::uint64_t{2},
                  "materialization reclaimed the wrong WAL count");
    require_equal(materialized.reclaimed_wal_bytes, frozen_bytes,
                  "materialization reclaimed the wrong WAL byte count");
    require(materialized.reclamation_complete,
            "ordinary local WAL reclamation was incomplete");
    require_equal(materialized.total_wal_bytes, before.active_wal_bytes,
                  "materialization retained bytes outside the active WAL");

    for (const auto& path : frozen_paths) {
        require(!std::filesystem::exists(path),
                "reclaimed frozen WAL still exists");
    }
    require(std::filesystem::exists(active_path),
            "materialization removed the active WAL");

    const throwdb::detail::MutationCatalog first_catalog =
        catalog_for(store);
    require_equal(first_catalog.generation, materialized.generation,
                  "CURRENT does not reference the returned generation");
    require_equal(first_catalog.delta_entries.size(), std::size_t{1},
                  "catalog does not reference the materialized delta");
    require_equal(first_catalog.wal_entries.size(), std::size_t{1},
                  "catalog retained selected frozen WAL descriptors");
    const throwdb::detail::MutationDeltaDescriptor first_delta =
        first_catalog.delta_entries.front();
    require_equal(first_delta.source_wal_count, std::uint32_t{2},
                  "delta descriptor has the wrong source WAL count");
    require_equal(first_delta.entry_count, std::uint64_t{7},
                  "delta descriptor has the wrong collapsed entry count");
    require_equal(first_delta.put_count, std::uint64_t{4},
                  "delta descriptor has the wrong collapsed PUT count");
    const auto first_data_path =
        throwdb::detail::mutation_delta_data_path(store, first_delta.delta_id);
    const auto first_index_path =
        throwdb::detail::mutation_delta_index_path(store, first_delta.delta_id);
    require(std::filesystem::exists(first_data_path)
                && std::filesystem::exists(first_index_path),
            "catalog references a missing delta file pair");
    require_equal(std::filesystem::file_size(first_data_path),
                  first_delta.data_file_size,
                  "delta data file size differs from its descriptor");
    require_equal(std::filesystem::file_size(first_index_path),
                  first_delta.index_file_size,
                  "delta index file size differs from its descriptor");

    require_bytes_equal(
        writer->get_bytes(live_ids), expected,
        "active WAL did not shadow the new delta after materialization");
    require_missing(*writer, missing_ids, "after materialization");
    writer.reset();

    {
        auto reader = throwdb::NativeCollection::open(store);
        require_equal(reader->format_version(), std::uint32_t{6},
                      "reopen lost native format 6");
        require_equal(reader->count(), std::uint64_t{5},
                      "reopen changed the mixed live count");
        require_bytes_equal(
            reader->get_bytes(live_ids), expected,
            "delta plus active WAL is not bit exact after reopen");
        require_missing(*reader, missing_ids, "after first reopen");
        const auto stats = reader->mutation_log_stats();
        require_equal(stats.generation, materialized.generation,
                      "reopen changed the materialized generation");
        require_equal(stats.materialized_sequence, std::uint64_t{8},
                      "reopen changed the materialized sequence");
        require_equal(stats.delta_run_count, std::uint64_t{1},
                      "reopen lost the materialized delta");
    }

    // Exercise ordinary CRUD, rotation, a second materialization and a newer
    // active-WAL shadow after recovering the first delta from disk.
    writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> updated_99 = rows(1, record_size, 171);
    const std::vector<std::byte> inserted_555 = rows(1, record_size, 191);
    const std::vector<std::byte> active_555 = rows(1, record_size, 211);
    writer->update(std::array<std::uint64_t, 1>{99}, updated_99);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{0}),
        std::vector<std::uint8_t>({1}),
        "post-reopen delete returned the wrong result");
    writer->insert(std::array<std::uint64_t, 1>{555}, inserted_555);
    const auto second_rotation = writer->rotate_wal();
    require(second_rotation.rotated,
            "post-reopen non-empty WAL did not rotate");
    writer->update(std::array<std::uint64_t, 1>{555}, active_555);

    const auto second_materialized = writer->materialize_frozen_wals();
    require(second_materialized.materialized,
            "second frozen prefix was not materialized");
    require_equal(
        second_materialized.generation, second_rotation.generation + 2,
        "second materialization published the wrong generation");
    require_equal(second_materialized.committed_sequence, std::uint64_t{15},
                  "second lifecycle has the wrong committed sequence");
    require_equal(second_materialized.materialized_sequence,
                  std::uint64_t{14},
                  "second delta ended at the wrong sequence");
    require_equal(second_materialized.delta_run_count, std::uint64_t{2},
                  "second materialization did not retain two delta runs");
    require_equal(second_materialized.delta_entry_count, std::uint64_t{13},
                  "second delta has the wrong cumulative entry count");
    require_equal(second_materialized.delta_put_rows, std::uint64_t{8},
                  "second delta has the wrong cumulative PUT count");
    require_equal(second_materialized.frozen_wal_count, std::uint64_t{0},
                  "second materialization retained its frozen WAL");
    require_equal(second_materialized.reclaimed_wal_count, std::uint64_t{1},
                  "second materialization reclaimed the wrong WAL count");
    require_equal(second_materialized.active_wal_put_rows, std::uint64_t{1},
                  "new active WAL has the wrong PUT count");

    const std::array<std::uint64_t, 5> final_live_ids{
        1, 42, 99, almost_maximum_id, 555};
    std::vector<std::byte> final_expected;
    append_row(final_expected, active_insert_1, 0, record_size);
    append_row(final_expected, active_updates, 1, record_size);
    append_row(final_expected, updated_99, 0, record_size);
    append_row(final_expected, second_upserts, 1, record_size);
    append_row(final_expected, active_555, 0, record_size);
    const std::array<std::uint64_t, 4> final_missing_ids{
        0, 7, 100, maximum_id};
    require_equal(writer->count(), std::uint64_t{5},
                  "second lifecycle has the wrong live count");
    require_bytes_equal(
        writer->get_bytes(final_live_ids), final_expected,
        "second delta plus active shadow is not bit exact");
    require_missing(*writer, final_missing_ids,
                    "after second materialization");
    writer.reset();

    auto final_reader = throwdb::NativeCollection::open(store);
    require_equal(final_reader->count(), std::uint64_t{5},
                  "final reopen changed the live count");
    require_bytes_equal(
        final_reader->get_bytes(final_live_ids), final_expected,
        "second materialization is not bit exact after reopen");
    require_missing(*final_reader, final_missing_ids, "after final reopen");
    const auto final_stats = final_reader->mutation_log_stats();
    require_equal(final_stats.generation, second_materialized.generation,
                  "final reopen changed catalog generation");
    require_equal(final_stats.committed_sequence, std::uint64_t{15},
                  "final reopen changed committed sequence");
    require_equal(final_stats.materialized_sequence, std::uint64_t{14},
                  "final reopen changed materialized sequence");
    require_equal(final_stats.delta_run_count, std::uint64_t{2},
                  "final reopen lost a delta run");
}

void test_no_frozen_noop_and_readonly_legacy_rejections() {
    TempDirectory temporary;
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> ids{77};
    const std::vector<std::byte> base = rows(1, record_size, 9);

    const auto no_op_store = temporary.path() / "no-op";
    build_store(no_op_store, ids, base, dimension);
    {
        auto writer = throwdb::NativeCollection::open(
            no_op_store, {}, true);
        const auto before = writer->mutation_log_stats();
        const auto after = writer->materialize_frozen_wals();
        require(!after.materialized,
                "no-frozen materialization did not report a no-op");
        require_equal(after.generation, before.generation,
                      "no-op materialization advanced generation");
        require_equal(after.committed_sequence, before.committed_sequence,
                      "no-op materialization changed commit sequence");
        require_equal(after.delta_run_count, std::uint64_t{0},
                      "no-op materialization created a delta");
        require_equal(after.reclaimed_wal_count, std::uint64_t{0},
                      "no-op materialization reclaimed a WAL");
        require(after.reclamation_complete,
                "no-op materialization reports incomplete reclamation");
    }
    {
        auto reader = throwdb::NativeCollection::open(no_op_store);
        require_throws<throwdb::InvalidArgumentError>(
            [&reader] {
                static_cast<void>(reader->materialize_frozen_wals());
            },
            "read-only collection accepted materialization");
    }

    const auto legacy_store = temporary.path() / "legacy";
    build_store(legacy_store, ids, base, dimension);
    throwdb::detail::NativeManifest manifest = manifest_for(legacy_store);
    const throwdb::detail::MutationCatalog catalog = catalog_for(legacy_store);
    require_equal(catalog.wal_entries.size(), std::size_t{1},
                  "legacy fixture source has the wrong WAL count");
    const auto catalog_wal = wal_path_for(
        legacy_store, catalog.wal_entries.front());
    const auto legacy_wal =
        throwdb::detail::native_active_wal_path(legacy_store);
    std::error_code rename_error;
    std::filesystem::rename(catalog_wal, legacy_wal, rename_error);
    require(!rename_error,
            "cannot move active WAL into the legacy v4 location");
    manifest.required_feature_bits &=
        ~(throwdb::detail::kNativeRequiredFeatureCatalogV1
          | throwdb::detail::kNativeRequiredFeatureDeltaV1);
    throwdb::detail::write_native_manifest_atomic(legacy_store, manifest);

    auto legacy = throwdb::NativeCollection::open(legacy_store, {}, true);
    require_equal(legacy->format_version(), std::uint32_t{4},
                  "legacy fixture did not report native format 4");
    require(!legacy->mutation_log_stats().delta_managed,
            "legacy fixture reports delta management");
    require_throws<throwdb::InvalidArgumentError>(
        [&legacy] {
            static_cast<void>(legacy->materialize_frozen_wals());
        },
        "legacy v4 collection accepted materialization");
}

void make_single_delta_fixture(const std::filesystem::path& store) {
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> base = rows(2, record_size, 13);
    const std::vector<std::byte> updated = rows(2, record_size, 73);
    build_store(store, ids, base, dimension);
    auto writer = throwdb::NativeCollection::open(store, {}, true);
    writer->update(ids, updated);
    require(writer->rotate_wal().rotated,
            "delta failure fixture did not rotate");
    const auto materialized = writer->materialize_frozen_wals();
    require(materialized.materialized,
            "delta failure fixture did not materialize");
    require_equal(materialized.delta_run_count, std::uint64_t{1},
                  "delta failure fixture has the wrong run count");
}

void test_missing_and_corrupt_delta_fail_closed() {
    TempDirectory temporary;

    const auto missing_store = temporary.path() / "missing";
    make_single_delta_fixture(missing_store);
    const throwdb::detail::MutationCatalog missing_catalog =
        catalog_for(missing_store);
    require_equal(missing_catalog.delta_entries.size(), std::size_t{1},
                  "missing fixture does not reference one delta");
    const auto missing_data = throwdb::detail::mutation_delta_data_path(
        missing_store, missing_catalog.delta_entries.front().delta_id);
    require(std::filesystem::remove(missing_data),
            "cannot remove referenced delta data fixture");
    require_throws<throwdb::StorageError>(
        [&missing_store] {
            static_cast<void>(throwdb::NativeCollection::open(missing_store));
        },
        "collection opened without referenced delta data");

    const auto corrupt_store = temporary.path() / "corrupt";
    make_single_delta_fixture(corrupt_store);
    const throwdb::detail::MutationCatalog corrupt_catalog =
        catalog_for(corrupt_store);
    require_equal(corrupt_catalog.delta_entries.size(), std::size_t{1},
                  "corrupt fixture does not reference one delta");
    const auto corrupt_index = throwdb::detail::mutation_delta_index_path(
        corrupt_store, corrupt_catalog.delta_entries.front().delta_id);
    flip_file_byte(
        corrupt_index, throwdb::detail::kMutationDeltaHeaderSize);
    require_throws<throwdb::CorruptionError>(
        [&corrupt_store] {
            static_cast<void>(throwdb::NativeCollection::open(corrupt_store));
        },
        "collection opened with a corrupted delta index body");
}

void test_concurrent_reads_during_materialization() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 128;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::size_t row_count = 65'536;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = 1'000'000 + index;
    }
    const std::vector<std::byte> base = rows(row_count, record_size, 17);
    const std::vector<std::byte> updated = rows(row_count, record_size, 117);
    build_store(store, ids, base, dimension, 4ULL << 20);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    writer->update(ids, updated);
    require(writer->rotate_wal().rotated,
            "concurrent fixture did not rotate its WAL");

    std::atomic<bool> start{false};
    std::atomic<bool> finished{false};
    std::exception_ptr materialization_error;
    throwdb::NativeMutationLogStats materialized;
    std::thread materializer([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            materialized = writer->materialize_frozen_wals();
        } catch (...) {
            materialization_error = std::current_exception();
        }
        finished.store(true, std::memory_order_release);
    });

    start.store(true, std::memory_order_release);
    std::size_t completed_reads = 0;
    std::exception_ptr read_error;
    try {
        do {
            require_bytes_equal(
                writer->get_bytes(ids), updated,
                "read concurrent with materialization returned torn data");
            ++completed_reads;
        } while (!finished.load(std::memory_order_acquire));
    } catch (...) {
        read_error = std::current_exception();
    }
    materializer.join();
    if (read_error != nullptr) {
        std::rethrow_exception(read_error);
    }
    if (materialization_error != nullptr) {
        std::rethrow_exception(materialization_error);
    }
    require(completed_reads != 0,
            "concurrent materialization loop completed no reads");
    require(materialized.materialized,
            "concurrent materialization was a no-op");
    require_equal(materialized.delta_run_count, std::uint64_t{1},
                  "concurrent materialization created the wrong run count");
    require_equal(materialized.delta_entry_count,
                  static_cast<std::uint64_t>(row_count),
                  "concurrent materialization lost collapsed IDs");
    require_bytes_equal(
        writer->get_bytes(ids), updated,
        "post-materialization data is not bit exact");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_bytes_equal(
        reopened->get_bytes(ids), updated,
        "concurrent materialization is not bit exact after reopen");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 4> tests{{
        {"mixed_lifecycle_materializes_and_reopens_bit_exactly",
         test_mixed_lifecycle_materializes_and_reopens_bit_exactly},
        {"no_frozen_noop_and_readonly_legacy_rejections",
         test_no_frozen_noop_and_readonly_legacy_rejections},
        {"missing_and_corrupt_delta_fail_closed",
         test_missing_and_corrupt_delta_fail_closed},
        {"concurrent_reads_during_materialization",
         test_concurrent_reads_during_materialization},
    }};

    std::size_t failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name
                      << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " materialization test(s) failed\n";
        return 1;
    }
    return 0;
}
