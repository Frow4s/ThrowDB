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
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/mutation_catalog.hpp"
#include "../src/core/mutation_delta.hpp"
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
                / ("throwdb-native-compaction-test-"
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

void build_store(
    const std::filesystem::path& store,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension) {
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
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

[[nodiscard]] std::uint64_t delta_bytes(
    const throwdb::detail::MutationDeltaDescriptor& descriptor) {
    require(
        descriptor.data_file_size
            <= std::numeric_limits<std::uint64_t>::max()
                   - descriptor.index_file_size,
        "fixture delta byte count overflows uint64");
    return descriptor.data_file_size + descriptor.index_file_size;
}

[[nodiscard]] std::vector<std::filesystem::path> delta_paths(
    const std::filesystem::path& store,
    const std::span<const throwdb::detail::MutationDeltaDescriptor> deltas) {
    std::vector<std::filesystem::path> result;
    result.reserve(deltas.size() * 2);
    for (const auto& descriptor : deltas) {
        result.push_back(throwdb::detail::mutation_delta_data_path(
            store, descriptor.delta_id));
        result.push_back(throwdb::detail::mutation_delta_index_path(
            store, descriptor.delta_id));
    }
    return result;
}

[[nodiscard]] std::vector<std::string> delta_directory_entries(
    const std::filesystem::path& store) {
    std::vector<std::string> result;
    const auto directory =
        throwdb::detail::mutation_deltas_directory(store);
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        result.push_back(entry.path().filename().string());
    }
    std::sort(result.begin(), result.end());
    return result;
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

[[nodiscard]] throwdb::NativeMutationLogStats materialize_active_wal(
    throwdb::NativeCollection& writer,
    const std::string& context) {
    require(writer.rotate_wal().rotated,
            context + ": non-empty active WAL did not rotate");
    const throwdb::NativeMutationLogStats stats =
        writer.materialize_frozen_wals();
    require(stats.materialized,
            context + ": frozen WAL did not materialize");
    return stats;
}

void test_noop_validation_and_budgets() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> base = rows(2, record_size, 3);
    build_store(store, ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const auto fresh_before = writer->mutation_log_stats();
    const auto fresh_noop = writer->compact_delta_runs();
    require(!fresh_noop.compacted,
            "zero-run compaction did not report a no-op");
    require_equal(fresh_noop.generation, fresh_before.generation,
                  "zero-run compaction advanced the catalog generation");
    require_equal(fresh_noop.delta_run_count, std::uint64_t{0},
                  "zero-run compaction created a delta");

    const std::vector<std::byte> first_updates =
        rows(2, record_size, 23);
    writer->update(ids, first_updates);
    static_cast<void>(materialize_active_wal(*writer, "first run"));
    const auto one_run_before = writer->mutation_log_stats();
    const auto one_run_noop = writer->compact_delta_runs();
    require(!one_run_noop.compacted,
            "one-run compaction did not report a no-op");
    require_equal(one_run_noop.generation, one_run_before.generation,
                  "one-run compaction advanced the catalog generation");
    require_equal(one_run_noop.delta_run_count, std::uint64_t{1},
                  "one-run compaction changed the delta count");

    require_throws<throwdb::InvalidArgumentError>(
        [&writer] {
            static_cast<void>(writer->compact_delta_runs(1, 1, 1));
        },
        "compaction accepted max_runs below two");
    require_throws<throwdb::InvalidArgumentError>(
        [&writer] {
            static_cast<void>(writer->compact_delta_runs(65, 1, 1));
        },
        "compaction accepted max_runs above the fan-in limit");
    require_throws<throwdb::InvalidArgumentError>(
        [&writer] {
            static_cast<void>(writer->compact_delta_runs(2, 0, 1));
        },
        "compaction accepted a zero byte budget");
    require_throws<throwdb::InvalidArgumentError>(
        [&writer] {
            static_cast<void>(writer->compact_delta_runs(2, 1, 0));
        },
        "compaction accepted a zero entry budget");

    const std::vector<std::byte> second_update =
        rows(1, record_size, 43);
    writer->update(std::array<std::uint64_t, 1>{1}, second_update);
    static_cast<void>(materialize_active_wal(*writer, "second run"));
    const throwdb::detail::MutationCatalog before = catalog_for(store);
    require_equal(before.delta_entries.size(), std::size_t{2},
                  "budget fixture does not contain two delta runs");
    const std::uint64_t first_bytes = delta_bytes(
        before.delta_entries.front());
    const std::uint64_t first_entries =
        before.delta_entries.front().entry_count;

    const auto byte_limited = writer->compact_delta_runs(
        2, first_bytes, std::numeric_limits<std::uint64_t>::max());
    require(!byte_limited.compacted,
            "single-admitted-run byte budget unexpectedly compacted");
    require_equal(byte_limited.generation, before.generation,
                  "byte-budget no-op advanced the generation");
    require_equal(catalog_for(store), before,
                  "byte-budget no-op changed the current catalog");

    const auto entry_limited = writer->compact_delta_runs(
        2, std::numeric_limits<std::uint64_t>::max(), first_entries);
    require(!entry_limited.compacted,
            "single-admitted-run entry budget unexpectedly compacted");
    require_equal(entry_limited.generation, before.generation,
                  "entry-budget no-op advanced the generation");
    require_equal(catalog_for(store), before,
                  "entry-budget no-op changed the current catalog");
    writer.reset();

    auto reader = throwdb::NativeCollection::open(store);
    require_throws<throwdb::InvalidArgumentError>(
        [&reader] {
            static_cast<void>(reader->compact_delta_runs());
        },
        "read-only collection accepted delta compaction");
}

void test_overlapping_versions_tombstones_and_newer_wal_survive() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 5> base_ids{10, 20, 30, 40, 50};
    const std::vector<std::byte> base = rows(5, record_size, 5);
    build_store(store, base_ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::array<std::uint64_t, 2> first_update_ids{10, 20};
    const std::vector<std::byte> first_updates =
        rows(2, record_size, 25);
    const std::vector<std::byte> inserted_60 =
        rows(1, record_size, 45);
    writer->update(first_update_ids, first_updates);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{30}),
        std::vector<std::uint8_t>({1}),
        "first run delete returned the wrong result");
    writer->insert(std::array<std::uint64_t, 1>{60}, inserted_60);
    static_cast<void>(materialize_active_wal(*writer, "first run"));

    const std::array<std::uint64_t, 2> second_update_ids{10, 40};
    const std::vector<std::byte> second_updates =
        rows(2, record_size, 65);
    const std::vector<std::byte> inserted_70 =
        rows(1, record_size, 85);
    writer->update(second_update_ids, second_updates);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{20}),
        std::vector<std::uint8_t>({1}),
        "second run delete returned the wrong result");
    writer->insert(std::array<std::uint64_t, 1>{70}, inserted_70);
    static_cast<void>(materialize_active_wal(*writer, "second run"));

    // Leave logically newer mutations in a retained frozen WAL. Compaction
    // may rewrite only the selected delta prefix and must not replace these
    // overlay entries with the older merged versions.
    const std::vector<std::byte> newest_10 = rows(1, record_size, 105);
    const std::vector<std::byte> resurrected_20 =
        rows(1, record_size, 125);
    writer->update(std::array<std::uint64_t, 1>{10}, newest_10);
    writer->upsert(std::array<std::uint64_t, 1>{20}, resurrected_20);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{40}),
        std::vector<std::uint8_t>({1}),
        "newer WAL delete returned the wrong result");
    require(writer->rotate_wal().rotated,
            "newer WAL did not rotate into a retained frozen run");

    const std::array<std::uint64_t, 5> live_ids{10, 20, 50, 60, 70};
    std::vector<std::byte> expected;
    append_row(expected, newest_10, 0, record_size);
    append_row(expected, resurrected_20, 0, record_size);
    append_row(expected, base, 4, record_size);
    append_row(expected, inserted_60, 0, record_size);
    append_row(expected, inserted_70, 0, record_size);
    const std::array<std::uint64_t, 2> missing_ids{30, 40};
    require_equal(writer->count(), std::uint64_t{5},
                  "pre-compaction fixture has the wrong live count");
    require_bytes_equal(
        writer->get_bytes(live_ids), expected,
        "pre-compaction fixture is not bit exact");
    require_missing(*writer, missing_ids, "before compaction");

    const throwdb::NativeMutationLogStats before_stats =
        writer->mutation_log_stats();
    const throwdb::detail::MutationCatalog before = catalog_for(store);
    require_equal(before.delta_entries.size(), std::size_t{2},
                  "overlap fixture does not contain two delta runs");
    require_equal(before_stats.frozen_wal_count, std::uint64_t{1},
                  "newer shadow is not in one retained frozen WAL");
    const auto source_paths = delta_paths(store, before.delta_entries);
    std::uint64_t input_bytes = 0;
    std::uint64_t input_entries = 0;
    std::uint64_t input_source_wals = 0;
    for (const auto& descriptor : before.delta_entries) {
        input_bytes += delta_bytes(descriptor);
        input_entries += descriptor.entry_count;
        input_source_wals += descriptor.source_wal_count;
    }

    const throwdb::NativeMutationLogStats compacted =
        writer->compact_delta_runs(
            2, std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max());
    require(compacted.compacted,
            "two-run overlap fixture did not compact");
    require_equal(compacted.generation, before_stats.generation + 2,
                  "compaction did not publish reservation and final generations");
    require_equal(compacted.committed_sequence,
                  before_stats.committed_sequence,
                  "compaction changed the committed sequence");
    require_equal(compacted.materialized_sequence,
                  before_stats.materialized_sequence,
                  "compaction changed the materialized sequence");
    require_equal(compacted.frozen_wal_count,
                  before_stats.frozen_wal_count,
                  "compaction changed the retained frozen WAL count");
    require_equal(compacted.delta_run_count, std::uint64_t{1},
                  "compaction did not replace two runs with one");
    require_equal(compacted.delta_entry_count, std::uint64_t{6},
                  "compaction did not collapse overlapping IDs");
    require_equal(compacted.delta_put_rows, std::uint64_t{4},
                  "compaction retained the wrong materialized PUT count");
    require_equal(compacted.compacted_delta_run_count, std::uint64_t{2},
                  "compaction reports the wrong input run count");
    require_equal(compacted.compacted_delta_input_entries, input_entries,
                  "compaction reports the wrong input entry count");
    require_equal(compacted.compacted_delta_input_bytes, input_bytes,
                  "compaction reports the wrong input byte count");
    require_equal(compacted.reclaimed_delta_run_count, std::uint64_t{2},
                  "compaction reclaimed the wrong source run count");
    require_equal(compacted.reclaimed_delta_bytes, input_bytes,
                  "compaction reclaimed the wrong source byte count");
    require(compacted.reclamation_complete,
            "ordinary local delta reclamation was incomplete");

    const throwdb::detail::MutationCatalog after = catalog_for(store);
    require_equal(after.generation, before.generation + 2,
                  "CURRENT references the wrong post-compaction generation");
    require_equal(after.next_segment_id, before.next_segment_id + 1,
                  "compaction reserved the wrong number of segment IDs");
    require_equal(after.materialized_sequence, before.materialized_sequence,
                  "catalog compaction changed the materialized anchor");
    require_equal(after.terminal_wal_id, before.terminal_wal_id,
                  "catalog compaction changed the materialized WAL anchor");
    require_equal(after.wal_entries, before.wal_entries,
                  "catalog compaction changed retained WAL descriptors");
    require_equal(after.delta_entries.size(), std::size_t{1},
                  "catalog compaction did not publish one replacement run");
    const auto& output = after.delta_entries.front();
    require(output.delta_id != before.delta_entries[0].delta_id
                && output.delta_id != before.delta_entries[1].delta_id,
            "compaction reused a source delta ID");
    require_equal(output.first_sequence,
                  before.delta_entries.front().first_sequence,
                  "compaction changed the lineage start sequence");
    require_equal(output.last_sequence,
                  before.delta_entries.back().last_sequence,
                  "compaction changed the lineage end sequence");
    require_equal(output.source_predecessor_wal_id,
                  before.delta_entries.front().source_predecessor_wal_id,
                  "compaction changed the lineage predecessor");
    require_equal(output.source_terminal_wal_id,
                  before.delta_entries.back().source_terminal_wal_id,
                  "compaction changed the lineage terminal WAL");
    require_equal(output.source_wal_count,
                  static_cast<std::uint32_t>(input_source_wals),
                  "compaction changed the lineage source WAL count");
    for (const auto& path : source_paths) {
        require(!std::filesystem::exists(path),
                "reclaimed source delta file still exists");
    }

    require_equal(writer->count(), std::uint64_t{5},
                  "compaction changed the live count");
    require_bytes_equal(
        writer->get_bytes(live_ids), expected,
        "newer frozen WAL did not shadow the compacted delta");
    require_missing(*writer, missing_ids, "after compaction");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(reopened->count(), std::uint64_t{5},
                  "reopen changed the post-compaction live count");
    require_bytes_equal(
        reopened->get_bytes(live_ids), expected,
        "compacted delta plus newer WAL is not bit exact after reopen");
    require_missing(*reopened, missing_ids, "after compaction reopen");
}

void test_old_reader_survives_source_unlink() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> base = rows(2, record_size, 7);
    build_store(store, ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> delta_1 = rows(1, record_size, 27);
    const std::vector<std::byte> delta_2 = rows(1, record_size, 47);
    writer->update(std::array<std::uint64_t, 1>{1}, delta_1);
    static_cast<void>(materialize_active_wal(*writer, "first reader run"));
    writer->update(std::array<std::uint64_t, 1>{2}, delta_2);
    static_cast<void>(materialize_active_wal(*writer, "second reader run"));

    const throwdb::detail::MutationCatalog before = catalog_for(store);
    require_equal(before.delta_entries.size(), std::size_t{2},
                  "reader fixture does not contain two delta runs");
    const auto source_paths = delta_paths(store, before.delta_entries);
    auto old_reader = throwdb::NativeCollection::open(store);

    const std::vector<std::byte> newest_1 = rows(1, record_size, 67);
    writer->update(std::array<std::uint64_t, 1>{1}, newest_1);
    const auto compacted = writer->compact_delta_runs(
        2, std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max());
    require(compacted.compacted,
            "reader fixture did not compact");
    require(compacted.reclamation_complete,
            "reader fixture could not unlink source runs");
    for (const auto& path : source_paths) {
        require(!std::filesystem::exists(path),
                "old-reader source path still exists after compaction");
    }

    std::vector<std::byte> old_expected;
    append_row(old_expected, delta_1, 0, record_size);
    append_row(old_expected, delta_2, 0, record_size);
    require_bytes_equal(
        old_reader->get_bytes(ids), old_expected,
        "old reader lost its snapshot after source unlink");

    std::vector<std::byte> current_expected;
    append_row(current_expected, newest_1, 0, record_size);
    append_row(current_expected, delta_2, 0, record_size);
    require_bytes_equal(
        writer->get_bytes(ids), current_expected,
        "writer did not retain its newer active-WAL snapshot");
    old_reader.reset();
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_bytes_equal(
        reopened->get_bytes(ids), current_expected,
        "reopen lost active-WAL data after delta source unlink");
}

void test_unselected_delta_and_active_wal_survive() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> ids{1};
    const std::vector<std::byte> base = rows(1, record_size, 11);
    build_store(store, ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> first = rows(1, record_size, 31);
    const std::vector<std::byte> second = rows(1, record_size, 51);
    const std::vector<std::byte> third = rows(1, record_size, 71);
    const std::vector<std::byte> active = rows(1, record_size, 91);
    writer->update(ids, first);
    static_cast<void>(materialize_active_wal(*writer, "first prefix run"));
    writer->update(ids, second);
    static_cast<void>(materialize_active_wal(*writer, "second prefix run"));
    writer->update(ids, third);
    static_cast<void>(materialize_active_wal(*writer, "unselected run"));
    writer->update(ids, active);

    const throwdb::detail::MutationCatalog before = catalog_for(store);
    const auto before_stats = writer->mutation_log_stats();
    require_equal(before.delta_entries.size(), std::size_t{3},
                  "unselected fixture does not contain three delta runs");
    const auto selected_paths = delta_paths(
        store,
        std::span<const throwdb::detail::MutationDeltaDescriptor>(
            before.delta_entries.data(), 2));
    const auto retained_paths = delta_paths(
        store,
        std::span<const throwdb::detail::MutationDeltaDescriptor>(
            before.delta_entries.data() + 2, 1));

    const auto compacted = writer->compact_delta_runs(
        2, std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max());
    require(compacted.compacted,
            "oldest two of three delta runs were not compacted");
    require_equal(compacted.delta_run_count, std::uint64_t{2},
                  "compaction did not retain the unselected delta");
    require_equal(compacted.active_wal_put_rows, std::uint64_t{1},
                  "compaction changed the newer active WAL");
    require_equal(compacted.committed_sequence,
                  before_stats.committed_sequence,
                  "compaction changed the latest commit sequence");

    const throwdb::detail::MutationCatalog after = catalog_for(store);
    require_equal(after.delta_entries.size(), std::size_t{2},
                  "post-compaction catalog has the wrong delta count");
    require_equal(after.delta_entries.back(), before.delta_entries.back(),
                  "compaction changed the unselected delta descriptor");
    require_equal(after.wal_entries, before.wal_entries,
                  "compaction changed the active WAL descriptor");
    for (const auto& path : selected_paths) {
        require(!std::filesystem::exists(path),
                "selected delta pathname survived reclamation");
    }
    for (const auto& path : retained_paths) {
        require(std::filesystem::exists(path),
                "compaction reclaimed an unselected delta file");
    }
    require_bytes_equal(
        writer->get_bytes(ids), active,
        "active WAL did not shadow retained and compacted deltas");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_bytes_equal(
        reopened->get_bytes(ids), active,
        "reopen lost the active-WAL shadow over an unselected delta");
}

void test_corrupt_source_payload_refuses_publication() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> base = rows(2, record_size, 9);
    build_store(store, ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> first = rows(1, record_size, 29);
    const std::vector<std::byte> second = rows(1, record_size, 49);
    writer->update(std::array<std::uint64_t, 1>{1}, first);
    static_cast<void>(materialize_active_wal(*writer, "first corrupt run"));
    writer->update(std::array<std::uint64_t, 1>{2}, second);
    static_cast<void>(materialize_active_wal(*writer, "second corrupt run"));

    const throwdb::detail::MutationCatalog before = catalog_for(store);
    require_equal(before.delta_entries.size(), std::size_t{2},
                  "corruption fixture does not contain two delta runs");
    const auto source_paths = delta_paths(store, before.delta_entries);
    const auto directory_before = delta_directory_entries(store);
    const auto stats_before = writer->mutation_log_stats();
    const auto corrupt_data =
        throwdb::detail::mutation_delta_data_path(
            store, before.delta_entries.front().delta_id);
    flip_file_byte(corrupt_data, throwdb::detail::kMutationDeltaHeaderSize);

    require_throws<throwdb::CorruptionError>(
        [&writer] {
            static_cast<void>(writer->compact_delta_runs(
                2, std::numeric_limits<std::uint64_t>::max(),
                std::numeric_limits<std::uint64_t>::max()));
        },
        "compaction published data copied from a corrupted source payload");

    const throwdb::detail::MutationCatalog after = catalog_for(store);
    require_equal(after, before,
                  "failed source validation changed the current catalog");
    require_equal(delta_directory_entries(store), directory_before,
                  "failed source validation created or removed delta files");
    for (const auto& path : source_paths) {
        require(std::filesystem::exists(path),
                "failed source validation reclaimed a referenced delta file");
    }
    const auto stats_after = writer->mutation_log_stats();
    require_equal(stats_after.generation, stats_before.generation,
                  "failed source validation advanced the generation");
    require_equal(stats_after.delta_run_count, stats_before.delta_run_count,
                  "failed source validation changed the run count");
    require_equal(stats_after.committed_sequence,
                  stats_before.committed_sequence,
                  "failed source validation changed the commit sequence");
}

void test_all_tombstone_output_survives_reopen() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> base = rows(2, record_size, 13);
    build_store(store, ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{1}),
        std::vector<std::uint8_t>({1}),
        "first tombstone fixture delete failed");
    static_cast<void>(materialize_active_wal(*writer, "first tombstone run"));
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{2}),
        std::vector<std::uint8_t>({1}),
        "second tombstone fixture delete failed");
    static_cast<void>(materialize_active_wal(*writer, "second tombstone run"));

    const auto compacted = writer->compact_delta_runs(
        2, std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max());
    require(compacted.compacted,
            "all-tombstone input runs were not compacted");
    require_equal(compacted.delta_run_count, std::uint64_t{1},
                  "all-tombstone compaction created the wrong run count");
    require_equal(compacted.delta_entry_count, std::uint64_t{2},
                  "all-tombstone compaction lost version records");
    require_equal(compacted.delta_put_rows, std::uint64_t{0},
                  "all-tombstone compaction invented payload rows");
    require_equal(writer->count(), std::uint64_t{0},
                  "all-tombstone compaction resurrected base rows");
    require_missing(*writer, ids, "after all-tombstone compaction");

    const auto catalog = catalog_for(store);
    require_equal(catalog.delta_entries.size(), std::size_t{1},
                  "all-tombstone catalog has the wrong delta count");
    require_equal(catalog.delta_entries.front().put_count, std::uint64_t{0},
                  "all-tombstone descriptor has PUT rows");
    require_equal(
        catalog.delta_entries.front().data_file_size,
        throwdb::detail::kMutationDeltaHeaderSize,
        "all-tombstone data file is not header-only");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(reopened->count(), std::uint64_t{0},
                  "reopen resurrected an all-tombstone base row");
    require_missing(*reopened, ids, "after all-tombstone reopen");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 6> tests{{
        {"noop_validation_and_budgets",
         test_noop_validation_and_budgets},
        {"overlapping_versions_tombstones_and_newer_wal_survive",
         test_overlapping_versions_tombstones_and_newer_wal_survive},
        {"old_reader_survives_source_unlink",
         test_old_reader_survives_source_unlink},
        {"unselected_delta_and_active_wal_survive",
         test_unselected_delta_and_active_wal_survive},
        {"corrupt_source_payload_refuses_publication",
         test_corrupt_source_payload_refuses_publication},
        {"all_tombstone_output_survives_reopen",
         test_all_tombstone_output_survives_reopen},
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
        std::cerr << failures << " compaction test(s) failed\n";
        return 1;
    }
    return 0;
}
