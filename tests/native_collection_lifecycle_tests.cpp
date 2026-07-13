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
                / ("throwdb-native-lifecycle-test-"
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

void require_bytes_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::byte> expected,
    const std::string& message) {
    require(
        actual.size() == expected.size()
            && std::equal(actual.begin(), actual.end(), expected.begin()),
        message);
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

void build_store(
    const std::filesystem::path& store,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension,
    const std::uint64_t segment_size_bytes = 1ULL << 20) {
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = segment_size_bytes;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    require_equal(
        builder->format_version(), std::uint32_t{6},
        "delta builder did not report native format 6");
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

[[nodiscard]] throwdb::detail::MutationWalEntry active_entry_for(
    const std::filesystem::path& store) {
    const throwdb::detail::MutationCatalog catalog = catalog_for(store);
    const auto active = std::find_if(
        catalog.wal_entries.begin(), catalog.wal_entries.end(),
        [](const throwdb::detail::MutationWalEntry& entry) {
            return entry.state == throwdb::detail::MutationWalState::Active;
        });
    require(active != catalog.wal_entries.end(),
            "catalog does not contain an active WAL");
    return *active;
}

[[nodiscard]] std::vector<throwdb::detail::MutationWalEntry>
frozen_entries_for(const std::filesystem::path& store) {
    const throwdb::detail::MutationCatalog catalog = catalog_for(store);
    std::vector<throwdb::detail::MutationWalEntry> frozen;
    for (const throwdb::detail::MutationWalEntry& entry : catalog.wal_entries) {
        if (entry.state == throwdb::detail::MutationWalState::Frozen) {
            frozen.push_back(entry);
        }
    }
    return frozen;
}

[[nodiscard]] std::filesystem::path wal_path_for(
    const std::filesystem::path& store,
    const throwdb::detail::MutationWalEntry& entry) {
    return throwdb::detail::mutation_wal_path(store, entry.wal_id);
}

void append_tail(
    const std::filesystem::path& path,
    const std::span<const char> tail) {
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    require(stream.good(), "cannot open WAL for synthetic tail append");
    stream.write(tail.data(), static_cast<std::streamsize>(tail.size()));
    stream.close();
    require(stream.good(), "cannot append complete synthetic WAL tail");
}

void require_fresh_catalog_stats(
    const throwdb::NativeMutationLogStats& stats,
    const std::string& context) {
    require(stats.catalog_managed, context + ": catalog flag is false");
    require_equal(stats.generation, std::uint64_t{1},
                  context + ": initial generation is wrong");
    require_equal(stats.committed_sequence, std::uint64_t{0},
                  context + ": fresh WAL has a commit sequence");
    require_equal(stats.frozen_wal_count, std::uint64_t{0},
                  context + ": fresh catalog has a frozen WAL");
    require_equal(
        stats.active_wal_bytes,
        throwdb::detail::kWalFileHeaderSize,
        context + ": fresh active WAL size is wrong");
    require_equal(stats.active_wal_put_rows, std::uint64_t{0},
                  context + ": fresh active WAL has PUT rows");
    require_equal(
        stats.total_wal_bytes,
        throwdb::detail::kWalFileHeaderSize,
        context + ": fresh total WAL size is wrong");
    require(!stats.rotated, context + ": ordinary stats claim a rotation");
}

void test_builder_stats_empty_and_delete_only_rotation() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> ids{42};
    const std::vector<std::byte> base = rows(1, record_size, 3);
    build_store(store, ids, base, dimension);

    {
        auto reader = throwdb::NativeCollection::open(store);
        require_equal(reader->format_version(), std::uint32_t{6},
                      "delta reader did not report native format 6");
        require_fresh_catalog_stats(
            reader->mutation_log_stats(), "fresh read-only open");
        require_throws<throwdb::InvalidArgumentError>(
            [&reader] { static_cast<void>(reader->rotate_wal()); },
            "read-only handle rotated its WAL");
    }

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require_fresh_catalog_stats(
        writer->mutation_log_stats(), "fresh writer open");
    require_throws<throwdb::StorageError>(
        [&store] {
            static_cast<void>(
                throwdb::NativeCollection::open(store, {}, true));
        },
        "a second writer acquired the lifecycle lease");

    const throwdb::NativeMutationLogStats empty_rotation =
        writer->rotate_wal();
    require_fresh_catalog_stats(empty_rotation, "empty rotation");

    require_equal(
        writer->erase(ids), std::vector<std::uint8_t>({1}),
        "delete-only active WAL returned the wrong result");
    const throwdb::NativeMutationLogStats before_rotation =
        writer->mutation_log_stats();
    require_equal(before_rotation.committed_sequence, std::uint64_t{1},
                  "delete did not advance the commit sequence");
    require_equal(before_rotation.active_wal_put_rows, std::uint64_t{0},
                  "delete consumed a WAL PUT row");
    require(
        before_rotation.active_wal_bytes
            > throwdb::detail::kWalFileHeaderSize,
        "delete did not append a WAL transaction");

    const throwdb::NativeMutationLogStats rotated = writer->rotate_wal();
    require(rotated.rotated, "delete-only WAL rotation was treated as empty");
    require_equal(rotated.generation, std::uint64_t{2},
                  "delete-only rotation did not advance generation");
    require_equal(rotated.committed_sequence, std::uint64_t{1},
                  "rotation changed the committed sequence");
    require_equal(rotated.frozen_wal_count, std::uint64_t{1},
                  "rotation did not retain one frozen WAL");
    require_equal(
        rotated.active_wal_bytes,
        throwdb::detail::kWalFileHeaderSize,
        "successor WAL is not header-only");
    require_equal(rotated.active_wal_put_rows, std::uint64_t{0},
                  "successor WAL inherited PUT rows");
    require_equal(
        rotated.total_wal_bytes,
        before_rotation.active_wal_bytes
            + throwdb::detail::kWalFileHeaderSize,
        "rotation reports the wrong total WAL bytes");

    const throwdb::NativeMutationLogStats second_empty = writer->rotate_wal();
    require(!second_empty.rotated,
            "header-only successor was rotated a second time");
    require_equal(second_empty.generation, rotated.generation,
                  "empty rotation advanced catalog generation");
    require_equal(second_empty.total_wal_bytes, rotated.total_wal_bytes,
                  "empty rotation changed total WAL bytes");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(reopened->count(), std::uint64_t{0},
                  "delete-only rotation resurrected a base row");
    require_throws<throwdb::MissingIdError>(
        [&reopened, &ids] {
            static_cast<void>(reopened->get_bytes(ids));
        },
        "reopened collection found a deleted base ID");
}

void test_crud_across_multiple_rotations_and_reopen() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::uint64_t id_a = 0;
    constexpr std::uint64_t id_b = 0xffffffffULL;
    constexpr std::uint64_t id_c = 0x100000000ULL;
    constexpr std::uint64_t id_d = std::uint64_t{1} << 63;
    constexpr std::uint64_t id_e =
        std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t id_f = 17;
    constexpr std::uint64_t id_g =
        std::numeric_limits<std::uint64_t>::max() - 1;
    const std::array<std::uint64_t, 5> base_ids{
        id_a, id_b, id_c, id_d, id_e};
    const std::vector<std::byte> base = rows(base_ids.size(), record_size, 5);
    build_store(
        store, base_ids, base, dimension, 2 * record_size);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> first_update = rows(2, record_size, 31);
    const std::vector<std::byte> first_insert = rows(1, record_size, 61);
    writer->update(
        std::array<std::uint64_t, 2>{id_a, id_e}, first_update);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{id_b}),
        std::vector<std::uint8_t>({1}),
        "first-round delete returned the wrong result");
    writer->insert(
        std::array<std::uint64_t, 1>{id_f}, first_insert);

    const throwdb::NativeMutationLogStats first_active =
        writer->mutation_log_stats();
    require_equal(first_active.committed_sequence, std::uint64_t{3},
                  "first active WAL has the wrong sequence");
    require_equal(first_active.active_wal_put_rows, std::uint64_t{3},
                  "first active WAL has the wrong PUT-row count");
    const throwdb::NativeMutationLogStats first_rotation =
        writer->rotate_wal();
    require(first_rotation.rotated, "first non-empty WAL did not rotate");
    require_equal(first_rotation.generation, std::uint64_t{2},
                  "first rotation has the wrong generation");
    require_equal(first_rotation.frozen_wal_count, std::uint64_t{1},
                  "first rotation has the wrong frozen count");
    require_equal(
        first_rotation.total_wal_bytes,
        first_active.active_wal_bytes
            + throwdb::detail::kWalFileHeaderSize,
        "first rotation has the wrong total bytes");

    const std::vector<std::byte> second_update = rows(2, record_size, 91);
    const std::vector<std::byte> reinserted_b = rows(1, record_size, 121);
    writer->update(
        std::array<std::uint64_t, 2>{id_a, id_c}, second_update);
    writer->insert(
        std::array<std::uint64_t, 1>{id_b}, reinserted_b);
    require_equal(
        writer->erase(std::array<std::uint64_t, 2>{id_f, id_d}),
        std::vector<std::uint8_t>({1, 1}),
        "second-round deletes returned the wrong results");

    const throwdb::NativeMutationLogStats second_active =
        writer->mutation_log_stats();
    require_equal(second_active.committed_sequence, std::uint64_t{6},
                  "second active WAL has the wrong sequence");
    require_equal(second_active.active_wal_put_rows, std::uint64_t{3},
                  "second active WAL has the wrong PUT-row count");
    require_equal(
        second_active.total_wal_bytes,
        first_active.active_wal_bytes + second_active.active_wal_bytes,
        "second active WAL reports the wrong total bytes");
    const throwdb::NativeMutationLogStats second_rotation =
        writer->rotate_wal();
    require(second_rotation.rotated, "second non-empty WAL did not rotate");
    require_equal(second_rotation.generation, std::uint64_t{3},
                  "second rotation has the wrong generation");
    require_equal(second_rotation.frozen_wal_count, std::uint64_t{2},
                  "second rotation has the wrong frozen count");
    require_equal(
        second_rotation.total_wal_bytes,
        first_active.active_wal_bytes + second_active.active_wal_bytes
            + throwdb::detail::kWalFileHeaderSize,
        "second rotation has the wrong total bytes");

    const std::vector<std::byte> third_upsert = rows(2, record_size, 151);
    const std::vector<std::byte> reinserted_e = rows(1, record_size, 181);
    writer->upsert(
        std::array<std::uint64_t, 2>{id_e, id_g}, third_upsert);
    require_equal(
        writer->erase(std::array<std::uint64_t, 1>{id_e}),
        std::vector<std::uint8_t>({1}),
        "third-round delete returned the wrong result");
    writer->insert(
        std::array<std::uint64_t, 1>{id_e}, reinserted_e);

    const throwdb::NativeMutationLogStats final_stats =
        writer->mutation_log_stats();
    require_equal(final_stats.generation, std::uint64_t{3},
                  "ordinary commits changed catalog generation");
    require_equal(final_stats.committed_sequence, std::uint64_t{9},
                  "final committed sequence is wrong");
    require_equal(final_stats.frozen_wal_count, std::uint64_t{2},
                  "final frozen WAL count is wrong");
    require_equal(final_stats.active_wal_put_rows, std::uint64_t{3},
                  "final active WAL PUT-row count is wrong");
    require_equal(
        final_stats.total_wal_bytes,
        first_active.active_wal_bytes + second_active.active_wal_bytes
            + final_stats.active_wal_bytes,
        "final total WAL bytes are wrong");
    require_equal(writer->count(), std::uint64_t{5},
                  "multi-rotation CRUD produced the wrong live count");

    const std::array<std::uint64_t, 6> query{
        id_g, id_a, id_e, id_b, id_c, id_a};
    std::vector<std::byte> expected;
    append_row(expected, third_upsert, 1, record_size);
    append_row(expected, second_update, 0, record_size);
    append_row(expected, reinserted_e, 0, record_size);
    append_row(expected, reinserted_b, 0, record_size);
    append_row(expected, second_update, 1, record_size);
    append_row(expected, second_update, 0, record_size);
    require_bytes_equal(
        writer->get_bytes(query), expected,
        "multi-WAL read is not bit exact before reopen");
    for (const std::uint64_t missing : {id_d, id_f}) {
        require_throws<throwdb::MissingIdError>(
            [&writer, missing] {
                static_cast<void>(writer->get_bytes(
                    std::array<std::uint64_t, 1>{missing}));
            },
            "multi-WAL writer found a deleted ID");
    }
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(reopened->format_version(), std::uint32_t{6},
                  "reopened delta store lost format 6");
    require_equal(reopened->count(), std::uint64_t{5},
                  "reopened multi-WAL count is wrong");
    require_bytes_equal(
        reopened->get_bytes(query), expected,
        "multi-WAL recovery is not bit exact");
    const throwdb::NativeMutationLogStats recovered_stats =
        reopened->mutation_log_stats();
    require_equal(recovered_stats.generation, final_stats.generation,
                  "reopen changed catalog generation");
    require_equal(
        recovered_stats.committed_sequence, final_stats.committed_sequence,
        "reopen changed committed sequence");
    require_equal(
        recovered_stats.frozen_wal_count, final_stats.frozen_wal_count,
        "reopen changed frozen WAL count");
    require_equal(
        recovered_stats.active_wal_put_rows,
        final_stats.active_wal_put_rows,
        "reopen changed active WAL PUT-row count");
    require_equal(
        recovered_stats.total_wal_bytes, final_stats.total_wal_bytes,
        "reopen changed total WAL bytes");
}

void test_active_wal_torn_tail_recovery_uses_catalog_path() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 3;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> base_id{1};
    const std::vector<std::byte> base = rows(1, record_size, 7);
    const std::vector<std::byte> inserted = rows(1, record_size, 37);
    const std::vector<std::byte> after_repair = rows(1, record_size, 67);
    build_store(store, base_id, base, dimension);
    {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        writer->insert(std::array<std::uint64_t, 1>{2}, inserted);
    }

    const throwdb::detail::MutationWalEntry active = active_entry_for(store);
    const std::filesystem::path wal_path = wal_path_for(store, active);
    const std::uint64_t committed_size = std::filesystem::file_size(wal_path);
    constexpr std::array<char, 13> tail{
        't', 'o', 'r', 'n', '-', 'w', 'a', 'l', '-', 't', 'a', 'i', 'l'};
    append_tail(wal_path, tail);
    require_equal(
        std::filesystem::file_size(wal_path), committed_size + tail.size(),
        "synthetic active-WAL tail has the wrong size");

    {
        auto reader = throwdb::NativeCollection::open(store);
        require_equal(reader->count(), std::uint64_t{2},
                      "read-only recovery lost committed active-WAL state");
        require_equal(
            std::filesystem::file_size(wal_path), committed_size + tail.size(),
            "read-only open repaired the active WAL");
    }
    {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        require_equal(std::filesystem::file_size(wal_path), committed_size,
                      "writer did not repair the catalog active WAL");
        writer->insert(
            std::array<std::uint64_t, 1>{3}, after_repair);
        require_equal(
            writer->mutation_log_stats().committed_sequence,
            std::uint64_t{2},
            "tail repair consumed or duplicated a commit sequence");
    }

    auto reopened = throwdb::NativeCollection::open(store);
    const std::array<std::uint64_t, 3> query{1, 2, 3};
    std::vector<std::byte> expected;
    append_row(expected, base, 0, record_size);
    append_row(expected, inserted, 0, record_size);
    append_row(expected, after_repair, 0, record_size);
    require_bytes_equal(
        reopened->get_bytes(query), expected,
        "commit after catalog active-WAL repair is not bit exact");
}

void test_referenced_wal_corruption_and_orphan_catalog_handling() {
    TempDirectory temporary;
    constexpr std::uint32_t dimension = 2;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> base_id{11};
    const std::vector<std::byte> base = rows(1, record_size, 9);
    const std::vector<std::byte> inserted = rows(1, record_size, 39);

    const auto frozen_store = temporary.path() / "frozen-tail";
    build_store(frozen_store, base_id, base, dimension);
    {
        auto writer = throwdb::NativeCollection::open(
            frozen_store, {}, true);
        writer->insert(
            std::array<std::uint64_t, 1>{12}, inserted);
        require(writer->rotate_wal().rotated,
                "corruption fixture did not create a frozen WAL");
    }
    const auto frozen_entries = frozen_entries_for(frozen_store);
    require_equal(frozen_entries.size(), std::size_t{1},
                  "corruption fixture has the wrong frozen count");
    const std::filesystem::path frozen_path =
        wal_path_for(frozen_store, frozen_entries.front());
    constexpr std::array<char, 1> extra_byte{'x'};
    append_tail(frozen_path, extra_byte);
    require_throws<throwdb::CorruptionError>(
        [&frozen_store] {
            static_cast<void>(
                throwdb::NativeCollection::open(frozen_store));
        },
        "frozen WAL accepted a synthetic tail");

    const auto missing_store = temporary.path() / "missing-wal";
    build_store(missing_store, base_id, base, dimension);
    const std::filesystem::path missing_path =
        wal_path_for(missing_store, active_entry_for(missing_store));
    require(std::filesystem::remove(missing_path),
            "cannot remove referenced active WAL fixture");
    require_throws<throwdb::StorageError>(
        [&missing_store] {
            static_cast<void>(
                throwdb::NativeCollection::open(missing_store));
        },
        "collection opened without its referenced active WAL");

    const auto orphan_store = temporary.path() / "orphan-catalog";
    build_store(orphan_store, base_id, base, dimension);
    const throwdb::detail::NativeManifest manifest = manifest_for(orphan_store);
    const throwdb::detail::MutationCurrent current =
        throwdb::detail::read_mutation_current(orphan_store, manifest);
    const std::filesystem::path current_catalog =
        throwdb::detail::mutation_catalog_path(
            orphan_store, current.catalog_id);
    throwdb::detail::Uuid128 orphan_id{
        current.catalog_id.low ^ 0xa5a5a5a5a5a5a5a5ULL,
        current.catalog_id.high ^ 0x5a5a5a5a5a5a5a5aULL};
    if (orphan_id.empty()) {
        orphan_id.low = 1;
    }
    const std::filesystem::path orphan_catalog =
        throwdb::detail::mutation_catalog_path(orphan_store, orphan_id);
    std::error_code copy_error;
    const bool copied = std::filesystem::copy_file(
        current_catalog, orphan_catalog,
        std::filesystem::copy_options::none, copy_error);
    require(copied && !copy_error,
            "cannot create an unreferenced catalog fixture");

    auto reader = throwdb::NativeCollection::open(orphan_store);
    require_equal(reader->count(), std::uint64_t{1},
                  "orphan catalog changed the selected collection view");
    require_equal(
        reader->mutation_log_stats().generation, current.generation,
        "open selected an orphan catalog instead of CURRENT");
    require_bytes_equal(
        reader->get_bytes(base_id), base,
        "orphan catalog changed base data");
}

void test_legacy_v4_reports_uncatalogued_and_rejects_rotation() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 2;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 1> ids{77};
    const std::vector<std::byte> base = rows(1, record_size, 19);
    build_store(store, ids, base, dimension);

    // Reproduce the exact pre-catalog v4 layout from freshly written resources:
    // the immutable base/index stay unchanged, the same empty WAL moves back to
    // its legacy fixed path, and the catalog/delta manifest bits are cleared.
    // CURRENT and catalog orphans must be ignored by the legacy open.
    throwdb::detail::NativeManifest manifest = manifest_for(store);
    const throwdb::detail::MutationWalEntry active = active_entry_for(store);
    const std::filesystem::path catalog_wal = wal_path_for(store, active);
    const std::filesystem::path legacy_wal =
        throwdb::detail::native_active_wal_path(store);
    std::error_code rename_error;
    std::filesystem::rename(catalog_wal, legacy_wal, rename_error);
    require(!rename_error,
            "cannot move catalog WAL into the legacy v4 location");
    manifest.required_feature_bits &=
        ~(throwdb::detail::kNativeRequiredFeatureCatalogV1
          | throwdb::detail::kNativeRequiredFeatureDeltaV1);
    throwdb::detail::write_native_manifest_atomic(store, manifest);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    require_equal(writer->format_version(), std::uint32_t{4},
                  "legacy fixture did not report native format 4");
    const throwdb::NativeMutationLogStats stats =
        writer->mutation_log_stats();
    require(!stats.catalog_managed,
            "legacy v4 stats claim catalog management");
    require_equal(stats.generation, std::uint64_t{0},
                  "legacy v4 stats invented a catalog generation");
    require_equal(stats.committed_sequence, std::uint64_t{0},
                  "legacy v4 empty WAL has a commit sequence");
    require_equal(stats.frozen_wal_count, std::uint64_t{0},
                  "legacy v4 stats invented a frozen WAL");
    require_equal(
        stats.active_wal_bytes,
        throwdb::detail::kWalFileHeaderSize,
        "legacy v4 active WAL size is wrong");
    require_equal(
        stats.total_wal_bytes,
        throwdb::detail::kWalFileHeaderSize,
        "legacy v4 total WAL size is wrong");
    require_bytes_equal(
        writer->get_bytes(ids), base,
        "legacy v4 fixture changed logical data");
    require_throws<throwdb::InvalidArgumentError>(
        [&writer] { static_cast<void>(writer->rotate_wal()); },
        "legacy v4 writer accepted WAL rotation");

    const std::array<std::uint64_t, 2> mutation_ids{77, 88};
    const std::vector<std::byte> mutations = rows(2, record_size, 59);
    writer->upsert(mutation_ids, mutations);
    const throwdb::NativeMutationLogStats mutated_stats =
        writer->mutation_log_stats();
    require_equal(mutated_stats.committed_sequence, std::uint64_t{1},
                  "legacy v4 commit sequence did not advance");
    require_equal(mutated_stats.active_wal_put_rows, std::uint64_t{2},
                  "legacy v4 PUT rows were not indexed");
    require_bytes_equal(
        writer->get_bytes(mutation_ids), mutations,
        "legacy v4 mutations are not bit exact");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_equal(reopened->format_version(), std::uint32_t{4},
                  "mutated legacy fixture changed format version");
    require_equal(reopened->count(), std::uint64_t{2},
                  "legacy v4 mutation count did not recover");
    require_bytes_equal(
        reopened->get_bytes(mutation_ids), mutations,
        "legacy v4 mutations did not survive reopen");
}

void test_concurrent_reads_and_writer_lease_during_rotation() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 64;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::size_t row_count = 4096;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = 10'000 + index;
    }
    const std::vector<std::byte> base = rows(row_count, record_size, 13);
    const std::vector<std::byte> updated = rows(row_count, record_size, 113);
    build_store(store, ids, base, dimension, 1ULL << 20);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    writer->update(ids, updated);
    require_throws<throwdb::StorageError>(
        [&store] {
            static_cast<void>(
                throwdb::NativeCollection::open(store, {}, true));
        },
        "concurrent lifecycle test acquired a second writer lease");

    std::atomic<bool> start{false};
    std::atomic<bool> finished{false};
    std::exception_ptr rotation_error;
    throwdb::NativeMutationLogStats rotation_stats;
    std::thread rotator([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            rotation_stats = writer->rotate_wal();
        } catch (...) {
            rotation_error = std::current_exception();
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
                "read concurrent with rotation returned torn logical state");
            ++completed_reads;
        } while (!finished.load(std::memory_order_acquire));
    } catch (...) {
        read_error = std::current_exception();
    }
    rotator.join();
    if (read_error != nullptr) {
        std::rethrow_exception(read_error);
    }
    if (rotation_error != nullptr) {
        std::rethrow_exception(rotation_error);
    }
    require(completed_reads != 0,
            "concurrent read loop did not complete a read");
    require(rotation_stats.rotated,
            "concurrent non-empty WAL did not rotate");
    require_equal(rotation_stats.generation, std::uint64_t{2},
                  "concurrent rotation published the wrong generation");
    require_equal(rotation_stats.committed_sequence, std::uint64_t{1},
                  "concurrent rotation changed the committed sequence");
    require_bytes_equal(
        writer->get_bytes(ids), updated,
        "post-rotation writer read is not bit exact");
    writer.reset();

    auto reopened = throwdb::NativeCollection::open(store);
    require_bytes_equal(
        reopened->get_bytes(ids), updated,
        "concurrent rotation did not survive reopen");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 6> tests{{
        {"builder_stats_empty_and_delete_only_rotation",
         test_builder_stats_empty_and_delete_only_rotation},
        {"crud_across_multiple_rotations_and_reopen",
         test_crud_across_multiple_rotations_and_reopen},
        {"active_wal_torn_tail_recovery_uses_catalog_path",
         test_active_wal_torn_tail_recovery_uses_catalog_path},
        {"referenced_wal_corruption_and_orphan_catalog_handling",
         test_referenced_wal_corruption_and_orphan_catalog_handling},
        {"legacy_v4_reports_uncatalogued_and_rejects_rotation",
         test_legacy_v4_reports_uncatalogued_and_rejects_rotation},
        {"concurrent_reads_and_writer_lease_during_rotation",
         test_concurrent_reads_and_writer_lease_during_rotation},
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
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " lifecycle test(s) failed\n";
        return 1;
    }
    return 0;
}
