#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/mutation_catalog.hpp"
#include "../src/core/mutation_delta.hpp"
#include "../src/core/mutation_inventory.hpp"
#include "../src/core/native_format.hpp"

namespace {

using throwdb::detail::MutationInventoryClass;
using throwdb::detail::MutationInventoryEntry;
using throwdb::detail::MutationResourceInventory;
using throwdb::detail::Uuid128;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-mutation-inventory-test-"
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

void build_store(const std::filesystem::path& store) {
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> embeddings = rows(2, record_size, 3);
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(ids, embeddings);
    builder->finish();
}

void add_reachable_delta(const std::filesystem::path& store) {
    constexpr std::uint64_t record_size = 8;
    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> updated = rows(1, record_size, 33);
    writer->update(std::array<std::uint64_t, 1>{1}, updated);
    require(writer->rotate_wal().rotated,
            "inventory fixture WAL did not rotate");
    const auto materialized = writer->materialize_frozen_wals();
    require(materialized.materialized,
            "inventory fixture WAL did not materialize");
    require_equal(materialized.delta_run_count, std::uint64_t{1},
                  "inventory fixture has the wrong delta count");
}

void write_sized_file(
    const std::filesystem::path& path,
    const std::size_t size,
    const char fill) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot create inventory fixture file");
    const std::vector<char> bytes(size, fill);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    require(output.good(), "cannot write inventory fixture file");
}

void flip_file_byte(
    const std::filesystem::path& path,
    const std::uint64_t offset) {
    require(std::filesystem::file_size(path) > offset,
            "inventory corruption offset is outside the file");
    std::fstream stream(
        path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open inventory fixture for corruption");
    stream.seekg(static_cast<std::streamoff>(offset));
    char value = 0;
    stream.read(&value, 1);
    require(stream.gcount() == 1,
            "cannot read inventory fixture corruption byte");
    value = static_cast<char>(static_cast<unsigned char>(value) ^ 0x5aU);
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(&value, 1);
    stream.close();
    require(stream.good(), "cannot write inventory fixture corruption byte");
}

[[nodiscard]] std::string relative_path(
    const std::filesystem::path& store,
    const std::filesystem::path& path) {
    return path.lexically_relative(store).generic_string();
}

[[nodiscard]] std::string uuid_token(const Uuid128 id) {
    const std::string filename =
        throwdb::detail::mutation_wal_path({}, id).filename().string();
    require(filename.size() == 4 + 32 + 4,
            "UUID-derived WAL fixture name has the wrong size");
    return filename.substr(4, 32);
}

[[nodiscard]] const MutationInventoryEntry& entry_for(
    const MutationResourceInventory& inventory,
    const std::string& path) {
    const auto found = std::find_if(
        inventory.entries.begin(), inventory.entries.end(),
        [&path](const MutationInventoryEntry& entry) {
            return entry.relative_path == path;
        });
    if (found == inventory.entries.end()) {
        fail("inventory is missing entry '" + path + "'");
    }
    return *found;
}

void require_classification(
    const MutationResourceInventory& inventory,
    const std::filesystem::path& store,
    const std::filesystem::path& path,
    const MutationInventoryClass expected,
    const std::uint64_t expected_bytes,
    const std::string& context) {
    const MutationInventoryEntry& entry =
        entry_for(inventory, relative_path(store, path));
    require_equal(entry.classification, expected,
                  context + " '" + relative_path(store, path)
                      + "': wrong classification");
    require_equal(entry.bytes, expected_bytes,
                  context + " '" + relative_path(store, path)
                      + "': wrong byte count");
}

void test_classifies_current_history_orphans_and_unknowns() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    require_equal(
        uuid_token(Uuid128{0x101, 0x201}),
        std::string("00000000000002010000000000000101"),
        "UUID filename token changed high/low word order");
    build_store(store);
    add_reachable_delta(store);

    const throwdb::detail::NativeManifest manifest =
        throwdb::detail::read_native_manifest(store);
    const throwdb::detail::MutationCatalog catalog =
        throwdb::detail::read_current_mutation_catalog(store, manifest);
    require_equal(catalog.delta_entries.size(), std::size_t{1},
                  "inventory fixture does not reference one delta");
    require_equal(catalog.wal_entries.size(), std::size_t{1},
                  "inventory fixture does not reference one active WAL");
    const auto current_catalog_path =
        throwdb::detail::mutation_catalog_path(store, catalog.catalog_id);

    // Corrupting a historical catalog must not affect reachability. Only the
    // catalog selected by validated CURRENT is authoritative.
    std::filesystem::path historical_catalog;
    for (const auto& entry :
         std::filesystem::directory_iterator(store / "catalogs")) {
        if (entry.path() != current_catalog_path
            && entry.path().extension() == ".bin") {
            historical_catalog = entry.path();
            break;
        }
    }
    require(!historical_catalog.empty(),
            "inventory fixture has no historical catalog");
    flip_file_byte(historical_catalog, 0);

    const Uuid128 orphan_wal_id{0x101, 0x201};
    const auto orphan_wal =
        throwdb::detail::mutation_wal_path(store, orphan_wal_id);
    write_sized_file(orphan_wal, 17, 'w');

    const Uuid128 complete_delta_id{0x102, 0x202};
    const auto complete_data =
        throwdb::detail::mutation_delta_data_path(store, complete_delta_id);
    const auto complete_index =
        throwdb::detail::mutation_delta_index_path(store, complete_delta_id);
    write_sized_file(complete_data, 19, 'd');
    write_sized_file(complete_index, 23, 'i');

    const Uuid128 partial_delta_id{0x103, 0x203};
    const auto partial_data =
        throwdb::detail::mutation_delta_data_path(store, partial_delta_id);
    write_sized_file(partial_data, 29, 'p');

    const Uuid128 temp_id{0x104, 0x204};
    const auto current_temp = store
        / ("CURRENT." + uuid_token(temp_id) + ".tmp");
    const auto catalog_temp = std::filesystem::path(
        throwdb::detail::mutation_catalog_path(store, temp_id).string()
        + ".tmp");
    const auto delta_data_temp = std::filesystem::path(
        throwdb::detail::mutation_delta_data_path(store, temp_id).string()
        + ".tmp");
    const auto delta_index_temp = std::filesystem::path(
        throwdb::detail::mutation_delta_index_path(store, temp_id).string()
        + ".tmp");
    write_sized_file(current_temp, 31, 'c');
    write_sized_file(catalog_temp, 37, 't');
    write_sized_file(delta_data_temp, 41, 'x');
    write_sized_file(delta_index_temp, 43, 'y');

    const auto malformed = store / "wal" / "wal-not-a-uuid.wal";
    const auto notes = store / "wal" / "notes.txt";
    write_sized_file(malformed, 47, 'm');
    write_sized_file(notes, 53, 'n');

    const Uuid128 symlink_wal_id{0x105, 0x205};
    const auto symlink_wal =
        throwdb::detail::mutation_wal_path(store, symlink_wal_id);
    std::error_code symlink_error;
    std::filesystem::create_symlink("notes.txt", symlink_wal, symlink_error);
    require(!symlink_error, "cannot create inventory symlink fixture");

    const Uuid128 directory_delta_id{0x106, 0x206};
    const auto directory_delta =
        throwdb::detail::mutation_delta_index_path(store, directory_delta_id);
    std::error_code directory_error;
    const bool made_directory =
        std::filesystem::create_directory(directory_delta, directory_error);
    require(made_directory && !directory_error,
            "cannot create inventory non-regular fixture");

    std::string uppercase_token = uuid_token(
        Uuid128{0xabcdef0123456107ULL, 0xfedcba9876543207ULL});
    std::transform(
        uppercase_token.begin(), uppercase_token.end(),
        uppercase_token.begin(), [](const unsigned char character) {
            return character >= 'a' && character <= 'f'
                       ? static_cast<char>(character - 'a' + 'A')
                       : static_cast<char>(character);
        });
    const auto uppercase_wal =
        store / "wal" / ("wal-" + uppercase_token + ".wal");
    write_sized_file(uppercase_wal, 59, 'u');
    const auto zero_uuid_wal = store / "wal"
        / "wal-00000000000000000000000000000000.wal";
    write_sized_file(zero_uuid_wal, 61, 'z');
    const auto extra_suffix_delta = std::filesystem::path(
        throwdb::detail::mutation_delta_data_path(
            store, Uuid128{0x108, 0x208}).string()
        + ".extra");
    write_sized_file(extra_suffix_delta, 67, 'e');

    const MutationResourceInventory inventory =
        throwdb::detail::inspect_mutation_resources(store, manifest);
    const MutationResourceInventory repeated =
        throwdb::detail::inspect_mutation_resources(store, manifest);
    require_equal(repeated, inventory,
                  "repeated inventory is not deterministic");
    require_equal(inventory.current_generation, catalog.generation,
                  "inventory reports the wrong current generation");
    require_equal(inventory.current_catalog_id, catalog.catalog_id,
                  "inventory reports the wrong current catalog ID");

    require(std::is_sorted(
                inventory.entries.begin(), inventory.entries.end(),
                [](const auto& left, const auto& right) {
                    return left.relative_path < right.relative_path;
                }),
            "inventory paths are not sorted");
    for (std::size_t index = 0; index != inventory.entries.size(); ++index) {
        const auto& entry = inventory.entries[index];
        require(std::filesystem::path(entry.relative_path).is_relative(),
                "inventory emitted an absolute path");
        if (index != 0) {
            require(inventory.entries[index - 1].relative_path
                        != entry.relative_path,
                    "inventory emitted a duplicate path");
        }
        const auto status = std::filesystem::symlink_status(
            store / entry.relative_path);
        require(status.type() != std::filesystem::file_type::not_found,
                "inventory synthesized a path absent from the filesystem");
        if (std::filesystem::is_regular_file(status)) {
            require_equal(
                entry.bytes,
                static_cast<std::uint64_t>(std::filesystem::file_size(
                    store / entry.relative_path)),
                "inventory regular-file byte count is not exact");
        } else {
            require_equal(entry.bytes, std::uint64_t{0},
                          "inventory followed a non-regular entry");
        }
    }

    require_classification(
        inventory, store, store / "CURRENT",
        MutationInventoryClass::Reachable,
        throwdb::detail::kMutationCurrentFileSize, "CURRENT");
    require_classification(
        inventory, store, current_catalog_path,
        MutationInventoryClass::Reachable,
        std::filesystem::file_size(current_catalog_path), "current catalog");
    require_classification(
        inventory, store, historical_catalog,
        MutationInventoryClass::SupersededCatalog,
        std::filesystem::file_size(historical_catalog),
        "historical catalog");
    for (const auto& entry :
         std::filesystem::directory_iterator(store / "catalogs")) {
        if (entry.path().extension() == ".bin"
            && entry.path() != current_catalog_path) {
            require_classification(
                inventory, store, entry.path(),
                MutationInventoryClass::SupersededCatalog,
                std::filesystem::file_size(entry.path()),
                "superseded catalog set");
        }
    }

    const auto reachable_wal = throwdb::detail::mutation_wal_path(
        store, catalog.wal_entries.front().wal_id);
    require_classification(
        inventory, store, reachable_wal,
        MutationInventoryClass::Reachable,
        std::filesystem::file_size(reachable_wal), "reachable WAL");
    const auto& reachable_delta = catalog.delta_entries.front();
    const auto reachable_data =
        throwdb::detail::mutation_delta_data_path(
            store, reachable_delta.delta_id);
    const auto reachable_index =
        throwdb::detail::mutation_delta_index_path(
            store, reachable_delta.delta_id);
    require_classification(
        inventory, store, reachable_data,
        MutationInventoryClass::Reachable,
        std::filesystem::file_size(reachable_data), "reachable delta data");
    require_classification(
        inventory, store, reachable_index,
        MutationInventoryClass::Reachable,
        std::filesystem::file_size(reachable_index), "reachable delta index");

    require_classification(
        inventory, store, orphan_wal,
        MutationInventoryClass::UnreferencedWal, 17,
        "unreferenced WAL");
    require_classification(
        inventory, store, complete_data,
        MutationInventoryClass::CompleteUnreferencedDeltaPair, 19,
        "complete delta data");
    require_classification(
        inventory, store, complete_index,
        MutationInventoryClass::CompleteUnreferencedDeltaPair, 23,
        "complete delta index");
    require_classification(
        inventory, store, partial_data,
        MutationInventoryClass::PartialUnreferencedDeltaPair, 29,
        "partial delta data");

    for (const auto& [path, bytes] :
         std::array<std::pair<std::filesystem::path, std::uint64_t>, 4>{{
             {current_temp, 31},
             {catalog_temp, 37},
             {delta_data_temp, 41},
             {delta_index_temp, 43},
         }}) {
        require_classification(
            inventory, store, path,
            MutationInventoryClass::TempPublication, bytes,
            "temporary publication");
    }
    for (const auto& [path, bytes] :
         std::array<std::pair<std::filesystem::path, std::uint64_t>, 7>{{
             {malformed, 47},
             {notes, 53},
             {symlink_wal, 0},
             {directory_delta, 0},
             {uppercase_wal, 59},
             {zero_uuid_wal, 61},
             {extra_suffix_delta, 67},
         }}) {
        require_classification(
            inventory, store, path,
            MutationInventoryClass::UnknownDoNotTouch, bytes,
            "unknown do-not-touch entry");
    }
}

void test_invalid_current_or_current_catalog_fails_closed() {
    TempDirectory temporary;
    const auto store = temporary.path() / "corrupt-store";
    build_store(store);
    const throwdb::detail::NativeManifest manifest =
        throwdb::detail::read_native_manifest(store);
    const throwdb::detail::MutationCatalog catalog =
        throwdb::detail::read_current_mutation_catalog(store, manifest);
    const auto current = throwdb::detail::mutation_current_path(store);
    const auto current_catalog =
        throwdb::detail::mutation_catalog_path(store, catalog.catalog_id);

    static_cast<void>(
        throwdb::detail::inspect_mutation_resources(store, manifest));
    flip_file_byte(current, 0);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::detail::inspect_mutation_resources(store, manifest));
        },
        "inventory derived reachability from corrupted CURRENT");
    flip_file_byte(current, 0);

    flip_file_byte(current_catalog, 0);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::detail::inspect_mutation_resources(store, manifest));
        },
        "inventory derived reachability from corrupted current catalog");

    const auto symlink_store = temporary.path() / "symlink-current-store";
    build_store(symlink_store);
    const throwdb::detail::NativeManifest symlink_manifest =
        throwdb::detail::read_native_manifest(symlink_store);
    const auto symlink_current =
        throwdb::detail::mutation_current_path(symlink_store);
    const auto current_target = symlink_store / "CURRENT.real";
    std::error_code rename_error;
    std::filesystem::rename(
        symlink_current, current_target, rename_error);
    require(!rename_error, "cannot move authoritative CURRENT fixture");
    std::error_code symlink_error;
    std::filesystem::create_symlink(
        "CURRENT.real", symlink_current, symlink_error);
    require(!symlink_error, "cannot create authoritative CURRENT symlink");
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::inspect_mutation_resources(
                symlink_store, symlink_manifest));
        },
        "inventory followed an authoritative CURRENT symlink");

    const auto directory_store = temporary.path() / "directory-catalog-store";
    build_store(directory_store);
    const throwdb::detail::NativeManifest directory_manifest =
        throwdb::detail::read_native_manifest(directory_store);
    const throwdb::detail::MutationCatalog directory_catalog =
        throwdb::detail::read_current_mutation_catalog(
            directory_store, directory_manifest);
    const auto directory_catalog_path =
        throwdb::detail::mutation_catalog_path(
            directory_store, directory_catalog.catalog_id);
    const auto saved_catalog = std::filesystem::path(
        directory_catalog_path.string() + ".real");
    rename_error.clear();
    std::filesystem::rename(
        directory_catalog_path, saved_catalog, rename_error);
    require(!rename_error, "cannot move authoritative catalog fixture");
    std::error_code directory_error;
    const bool made_directory = std::filesystem::create_directory(
        directory_catalog_path, directory_error);
    require(made_directory && !directory_error,
            "cannot create authoritative catalog directory");
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::inspect_mutation_resources(
                directory_store, directory_manifest));
        },
        "inventory treated a non-regular current catalog as authoritative");
}

[[nodiscard]] const throwdb::NativeOrphanEntry& public_entry_for(
    const throwdb::NativeOrphanInventory& inventory,
    const std::string& path) {
    const auto found = std::find_if(
        inventory.entries.begin(), inventory.entries.end(),
        [&path](const throwdb::NativeOrphanEntry& entry) {
            return entry.path == path;
        });
    if (found == inventory.entries.end()) {
        fail("public inventory is missing entry '" + path + "'");
    }
    return *found;
}

void require_public_inventories_equal(
    const throwdb::NativeOrphanInventory& actual,
    const throwdb::NativeOrphanInventory& expected,
    const std::string& context) {
    require_equal(actual.current_generation, expected.current_generation,
                  context + ": generation changed");
    require_equal(actual.current_catalog_id, expected.current_catalog_id,
                  context + ": catalog ID changed");
    require_equal(actual.entries.size(), expected.entries.size(),
                  context + ": entry count changed");
    for (std::size_t index = 0; index != actual.entries.size(); ++index) {
        const auto& left = actual.entries[index];
        const auto& right = expected.entries[index];
        require_equal(left.path, right.path,
                      context + ": path changed");
        require_equal(left.classification, right.classification,
                      context + ": classification changed");
        require_equal(left.bytes, right.bytes,
                      context + ": byte count changed");
        require_equal(left.modified_unix_ns, right.modified_unix_ns,
                      context + ": modification time changed");
    }
}

void test_public_inventory_is_lightweight_stable_and_writable_only() {
    TempDirectory temporary;
    const auto store = temporary.path() / "public-store";
    build_store(store);
    const throwdb::detail::NativeManifest manifest =
        throwdb::detail::read_native_manifest(store);
    const throwdb::detail::MutationCatalog catalog =
        throwdb::detail::read_current_mutation_catalog(store, manifest);

    const Uuid128 orphan_id{0x1122334455667788ULL,
                            0x99aabbccddeeff00ULL};
    const auto orphan_path =
        throwdb::detail::mutation_wal_path(store, orphan_id);
    write_sized_file(orphan_path, 71, 'o');

    // Static inventory must not load or validate the potentially huge flat
    // base index. Temporarily hiding it makes an accidental index read fail.
    const auto index_path = throwdb::detail::native_index_path(store);
    const auto hidden_index = store / "index" / "index.hidden";
    std::error_code error;
    std::filesystem::rename(index_path, hidden_index, error);
    require(!error, "cannot hide index for lightweight inventory test");
    const throwdb::NativeOrphanInventory lightweight =
        throwdb::NativeCollection::inspect_orphans(store);
    error.clear();
    std::filesystem::rename(hidden_index, index_path, error);
    require(!error, "cannot restore index after lightweight inventory test");

    require_equal(lightweight.current_generation, catalog.generation,
                  "public inventory reports the wrong generation");
    require_equal(lightweight.current_catalog_id,
                  uuid_token(catalog.catalog_id),
                  "public inventory reports the wrong catalog token");
    require(std::is_sorted(
                lightweight.entries.begin(), lightweight.entries.end(),
                [](const auto& left, const auto& right) {
                    return left.path < right.path;
                }),
            "public inventory entries are not path-sorted");
    const auto& orphan = public_entry_for(
        lightweight, relative_path(store, orphan_path));
    require_equal(orphan.classification,
                  throwdb::NativeOrphanClass::UnreferencedWal,
                  "public inventory maps orphan WAL class incorrectly");
    require_equal(orphan.bytes, std::uint64_t{71},
                  "public inventory reports the wrong orphan size");
    require(orphan.modified_unix_ns > 0,
            "public inventory did not expose the lstat modification time");

    {
        auto reader = throwdb::NativeCollection::open(store);
        require_throws<throwdb::InvalidArgumentError>(
            [&] { static_cast<void>(reader->collect_orphans()); },
            "read-only handle collected an unstable orphan inventory");
    }

    {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::inspect_orphans(store));
            },
            "lightweight inventory ignored an existing writer lease");
        const throwdb::NativeOrphanInventory through_writer =
            writer->collect_orphans();
        require_public_inventories_equal(
            through_writer, lightweight,
            "writable-handle and lightweight inventories differ");
    }

    const throwdb::NativeOrphanInventory repeated =
        throwdb::NativeCollection::inspect_orphans(store);
    require_public_inventories_equal(
        repeated, lightweight,
        "repeated public inventory is not deterministic");
}

void test_lightweight_inventory_excludes_builder_and_ignores_format5_deltas() {
    TempDirectory temporary;
    const auto building_store = temporary.path() / "building-store";
    throwdb::NativeBuilderOptions options;
    options.dimension = 4;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    auto builder =
        throwdb::NativeCollectionBuilder::create(building_store, options);
    bool builder_lease_busy = false;
    try {
        static_cast<void>(
            throwdb::NativeCollection::inspect_orphans(building_store));
    } catch (const throwdb::StorageError& error) {
        builder_lease_busy =
            std::string(error.what()).find("another writer")
            != std::string::npos;
    }
    require(builder_lease_busy,
            "lightweight inventory did not observe the active builder lease");

    // Successful publication releases the lease before finish() returns even
    // while the builder object itself remains alive.
    builder->finish();
    const throwdb::NativeOrphanInventory finished_builder_inventory =
        throwdb::NativeCollection::inspect_orphans(building_store);
    require_equal(finished_builder_inventory.current_generation,
                  std::uint64_t{1},
                  "finished builder retained its writer lease");

    const auto failed_store = temporary.path() / "failed-builder-store";
    auto failed_builder =
        throwdb::NativeCollectionBuilder::create(failed_store, options);
    std::error_code failure_error;
    std::filesystem::remove_all(failed_store / "index", failure_error);
    require(!failure_error,
            "cannot remove failed-builder index directory fixture");
    write_sized_file(failed_store / "index", 1, 'x');
    require_throws<throwdb::StorageError>(
        [&] { failed_builder->finish(); },
        "builder failure fixture unexpectedly published");
    bool failed_builder_lease_busy = false;
    try {
        static_cast<void>(
            throwdb::NativeCollection::inspect_orphans(failed_store));
    } catch (const throwdb::StorageError& error) {
        failed_builder_lease_busy =
            std::string(error.what()).find("another writer")
            != std::string::npos;
    }
    require(failed_builder_lease_busy,
            "failed builder released its lease before destruction");
    failed_builder.reset();
    bool lease_released_after_destruction = false;
    try {
        static_cast<void>(
            throwdb::NativeCollection::inspect_orphans(failed_store));
    } catch (const throwdb::StorageError& error) {
        lease_released_after_destruction =
            std::string(error.what()).find("another writer")
            == std::string::npos;
    }
    require(lease_released_after_destruction,
            "failed builder destructor retained its writer lease");

    const auto format5_store = temporary.path() / "format5-store";
    build_store(format5_store);
    throwdb::detail::NativeManifest format5_manifest =
        throwdb::detail::read_native_manifest(format5_store);
    format5_manifest.required_feature_bits &=
        ~throwdb::detail::kNativeRequiredFeatureDeltaV1;
    const Uuid128 format5_catalog_id{
        0x3141592653589793ULL, 0x2384626433832795ULL};
    const throwdb::detail::MutationCatalog format5_catalog =
        throwdb::detail::make_initial_mutation_catalog(
            format5_manifest, format5_catalog_id);
    static_cast<void>(throwdb::detail::publish_mutation_catalog(
        format5_store, format5_manifest, format5_catalog));
    throwdb::detail::write_native_manifest_atomic(
        format5_store, format5_manifest);

    std::error_code error;
    std::filesystem::remove_all(format5_store / "deltas", error);
    require(!error, "cannot remove format-5 deltas fixture directory");
    write_sized_file(format5_store / "deltas", 79, 'x');
    const throwdb::NativeOrphanInventory inventory =
        throwdb::NativeCollection::inspect_orphans(format5_store);
    require_equal(inventory.current_generation, std::uint64_t{1},
                  "format-5 inventory reports the wrong generation");
    require_equal(inventory.current_catalog_id,
                  uuid_token(format5_catalog_id),
                  "format-5 inventory reports the wrong catalog");
    require(std::none_of(
                inventory.entries.begin(), inventory.entries.end(),
                [](const throwdb::NativeOrphanEntry& entry) {
                    return entry.path == "deltas"
                           || entry.path.starts_with("deltas/");
                }),
            "format-5 inventory assigned semantics to the deltas namespace");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 4> tests{{
        {"classifies_current_history_orphans_and_unknowns",
         test_classifies_current_history_orphans_and_unknowns},
        {"invalid_current_or_current_catalog_fails_closed",
         test_invalid_current_or_current_catalog_fails_closed},
        {"public_inventory_is_lightweight_stable_and_writable_only",
         test_public_inventory_is_lightweight_stable_and_writable_only},
        {"lightweight_inventory_excludes_builder_and_ignores_format5_deltas",
         test_lightweight_inventory_excludes_builder_and_ignores_format5_deltas},
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
        std::cerr << failures << " mutation inventory test(s) failed\n";
        return 1;
    }
    return 0;
}
