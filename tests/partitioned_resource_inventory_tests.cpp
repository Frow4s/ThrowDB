#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partitioned_index_format.hpp"
#include "../src/core/partitioned_resource_inventory.hpp"

namespace {

using throwdb::detail::PartitionedInventoryClass;
using throwdb::detail::PartitionedInventoryEntry;
using throwdb::detail::PartitionedOrphanCollectionHooks;
using throwdb::detail::PartitionedOrphanCollectionOptions;
using throwdb::detail::PartitionedOrphanDisposition;
using throwdb::detail::PartitionedResourceInventory;
using throwdb::detail::Uuid128;

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
    }
    fail(message + " (no exception)");
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& label) {
        static std::atomic<std::uint64_t> ordinal{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-format7-inventory-" + label + "-"
                   + std::to_string(::getpid()) + "-"
                   + std::to_string(ordinal.fetch_add(1)));
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[nodiscard]] Uuid128 uuid(const std::uint64_t value) {
    return Uuid128{value, value ^ 0x9e3779b97f4a7c15ULL};
}

[[nodiscard]] std::string uuid_token(const Uuid128 value) {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> token{};
    const auto put = [&token](const std::uint64_t word, const std::size_t at) {
        for (std::size_t index = 0; index != 16; ++index) {
            const unsigned shift = static_cast<unsigned>((15 - index) * 4);
            token[at + index] = hex[(word >> shift) & 0xfU];
        }
    };
    put(value.high, 0);
    put(value.low, 16);
    return std::string(token.begin(), token.end());
}

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t count,
    const std::size_t record_size) {
    std::vector<std::byte> result(count * record_size);
    for (std::size_t index = 0; index != result.size(); ++index) {
        result[index] = static_cast<std::byte>((index * 37 + 11) & 0xffU);
    }
    return result;
}

void build_store(const std::filesystem::path& store) {
    throwdb::NativeBuilderOptions options;
    options.dimension = 4;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    options.format_version = 7;
    options.partition_bits = 2;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    const std::array<std::uint64_t, 3> ids{10, 20, 30};
    const std::vector<std::byte> embeddings = rows(ids.size(), 8);
    builder->append(ids, embeddings);
    builder->finish();
}

void write_sized_file(
    const std::filesystem::path& path,
    const std::size_t bytes,
    const char value = 'x') {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("cannot create fixture '" + path.string() + "'");
    }
    const std::string payload(bytes, value);
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!output) {
        fail("cannot write fixture '" + path.string() + "'");
    }
}

[[nodiscard]] std::vector<std::byte> read_file(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        fail("cannot read fixture '" + path.string() + "'");
    }
    const std::streamoff end = input.tellg();
    if (end < 0) {
        fail("cannot size fixture '" + path.string() + "'");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        fail("cannot read complete fixture '" + path.string() + "'");
    }
    return bytes;
}

void write_file(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        fail("cannot rewrite fixture '" + path.string() + "'");
    }
}

[[nodiscard]] const PartitionedInventoryEntry& entry_for(
    const PartitionedResourceInventory& inventory,
    const std::string& relative_path) {
    const auto position = std::find_if(
        inventory.entries.begin(), inventory.entries.end(),
        [&relative_path](const PartitionedInventoryEntry& entry) {
            return entry.relative_path == relative_path;
        });
    if (position == inventory.entries.end()) {
        fail("inventory omitted '" + relative_path + "'");
    }
    return *position;
}

void require_class(
    const PartitionedResourceInventory& inventory,
    const std::string& path,
    const PartitionedInventoryClass expected) {
    require(
        entry_for(inventory, path).classification == expected,
        "wrong classification for '" + path + "'");
}

[[nodiscard]] const throwdb::NativeOrphanEntry& public_entry_for(
    const throwdb::NativeOrphanInventory& inventory,
    const std::string& relative_path) {
    const auto position = std::find_if(
        inventory.entries.begin(), inventory.entries.end(),
        [&relative_path](const throwdb::NativeOrphanEntry& entry) {
            return entry.path == relative_path;
        });
    if (position == inventory.entries.end()) {
        fail("public inventory omitted '" + relative_path + "'");
    }
    return *position;
}

[[nodiscard]] std::uint64_t now_ns() {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return value < 0 ? 0 : static_cast<std::uint64_t>(value);
}

[[nodiscard]] PartitionedOrphanCollectionOptions permissive_options() {
    PartitionedOrphanCollectionOptions options;
    options.snapshot_unix_ns = now_ns() + 1'000'000'000ULL;
    options.min_age_ns = 0;
    options.max_files = 1024;
    options.max_bytes = 1ULL << 30;
    return options;
}

void test_reachability_and_strict_filename_classification() {
    TemporaryDirectory temporary("classify");
    build_store(temporary.path());
    const Uuid128 catalog_id = uuid(100);
    const Uuid128 wal_id = uuid(101);
    const Uuid128 segment_id = uuid(102);
    const Uuid128 partial_id = uuid(103);
    const Uuid128 snapshot_id = uuid(104);
    const Uuid128 run_id = uuid(105);
    const Uuid128 temp_id = uuid(106);

    write_sized_file(
        temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(catalog_id) + ".bin"),
        3);
    write_sized_file(
        temporary.path() / "wal" / ("wal-" + uuid_token(wal_id) + ".wal"),
        5);
    write_sized_file(
        temporary.path() / "segments"
            / ("segment-" + uuid_token(segment_id) + ".dat"),
        7);
    write_sized_file(
        temporary.path() / "segments"
            / ("segment-" + uuid_token(segment_id) + ".ids"),
        9);
    write_sized_file(
        temporary.path() / "segments"
            / ("segment-" + uuid_token(partial_id) + ".dat"),
        11);
    write_sized_file(
        temporary.path() / "index" / "partitions"
            / ("partition-1-" + uuid_token(snapshot_id) + ".idx"),
        13);
    write_sized_file(
        temporary.path() / "index" / "runs"
            / ("run-" + uuid_token(run_id) + ".idx"),
        15);
    write_sized_file(
        temporary.path()
            / ("CURRENT." + uuid_token(temp_id) + ".tmp"),
        17);
    write_sized_file(
        temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(temp_id) + ".bin.tmp"),
        19);
    write_sized_file(
        temporary.path() / "segments"
            / (".segment-" + uuid_token(temp_id) + ".dat.staging"),
        21);

    std::string uppercase = uuid_token(uuid(107));
    uppercase[0] = 'A';
    write_sized_file(
        temporary.path() / "wal" / ("wal-" + uppercase + ".wal"), 1);
    write_sized_file(
        temporary.path() / "index" / "partitions"
            / ("partition-01-" + uuid_token(uuid(108)) + ".idx"),
        1);
    write_sized_file(
        temporary.path() / "segments"
            / ("segment-" + uuid_token(uuid(109)) + ".DAT"),
        1);

    const PartitionedResourceInventory inventory =
        throwdb::detail::inspect_partitioned_resources(temporary.path());
    require(
        std::is_sorted(
            inventory.entries.begin(), inventory.entries.end(),
            [](const auto& left, const auto& right) {
                return left.relative_path < right.relative_path;
            }),
        "inventory report is not deterministic/sorted");
    require_class(inventory, "CURRENT", PartitionedInventoryClass::Reachable);
    require_class(
        inventory,
        "catalogs/catalog-" + uuid_token(catalog_id) + ".bin",
        PartitionedInventoryClass::SupersededCatalog);
    require_class(
        inventory, "wal/wal-" + uuid_token(wal_id) + ".wal",
        PartitionedInventoryClass::UnreferencedWal);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(segment_id) + ".dat",
        PartitionedInventoryClass::CompleteUnreferencedSegmentPair);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(segment_id) + ".ids",
        PartitionedInventoryClass::CompleteUnreferencedSegmentPair);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(partial_id) + ".dat",
        PartitionedInventoryClass::PartialUnreferencedSegmentPair);
    require_class(
        inventory,
        "index/partitions/partition-1-" + uuid_token(snapshot_id) + ".idx",
        PartitionedInventoryClass::UnreferencedSnapshot);
    require_class(
        inventory, "index/runs/run-" + uuid_token(run_id) + ".idx",
        PartitionedInventoryClass::UnreferencedRun);
    require_class(
        inventory, "CURRENT." + uuid_token(temp_id) + ".tmp",
        PartitionedInventoryClass::TempPublication);
    require_class(
        inventory,
        "catalogs/catalog-" + uuid_token(temp_id) + ".bin.tmp",
        PartitionedInventoryClass::TempPublication);
    require_class(
        inventory,
        "segments/.segment-" + uuid_token(temp_id) + ".dat.staging",
        PartitionedInventoryClass::TempPublication);
    require_class(
        inventory, "wal/wal-" + uppercase + ".wal",
        PartitionedInventoryClass::UnknownDoNotTouch);
    require_class(
        inventory,
        "index/partitions/partition-01-" + uuid_token(uuid(108)) + ".idx",
        PartitionedInventoryClass::UnknownDoNotTouch);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(uuid(109)) + ".DAT",
        PartitionedInventoryClass::UnknownDoNotTouch);

    const bool has_reachable_segment = std::any_of(
        inventory.entries.begin(), inventory.entries.end(),
        [](const PartitionedInventoryEntry& entry) {
            return entry.relative_path.starts_with("segments/segment-")
                && entry.classification == PartitionedInventoryClass::Reachable;
        });
    const bool has_reachable_snapshot = std::any_of(
        inventory.entries.begin(), inventory.entries.end(),
        [](const PartitionedInventoryEntry& entry) {
            return entry.relative_path.starts_with("index/partitions/")
                && entry.classification == PartitionedInventoryClass::Reachable;
        });
    require(has_reachable_segment, "catalog segments were not reachable");
    require(has_reachable_snapshot, "catalog snapshots were not reachable");
}

void test_authoritative_validation_and_catalog_cap() {
    {
        TemporaryDirectory temporary("current-corrupt");
        build_store(temporary.path());
        std::vector<std::byte> bytes = read_file(temporary.path() / "CURRENT");
        bytes[300] ^= std::byte{1};
        write_file(temporary.path() / "CURRENT", bytes);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "corrupt CURRENT was accepted");
    }
    {
        TemporaryDirectory temporary("current-cap");
        build_store(temporary.path());
        std::vector<std::byte> bytes = read_file(temporary.path() / "CURRENT");
        throwdb::detail::put_u64(
            bytes, 80,
            throwdb::detail::kPartitionedMaximumCatalogFileSize + 1);
        throwdb::detail::put_u32(bytes, 20, 0);
        throwdb::detail::put_u32(bytes, 20, throwdb::detail::crc32c(bytes));
        write_file(temporary.path() / "CURRENT", bytes);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "CURRENT catalog allocation cap was not enforced");
    }
    {
        TemporaryDirectory temporary("current-symlink");
        build_store(temporary.path());
        std::filesystem::rename(
            temporary.path() / "CURRENT", temporary.path() / "CURRENT.saved");
        std::filesystem::create_symlink(
            "CURRENT.saved", temporary.path() / "CURRENT");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "symlink CURRENT was followed");
    }
    {
        TemporaryDirectory temporary("catalog-corrupt");
        build_store(temporary.path());
        const PartitionedResourceInventory before =
            throwdb::detail::inspect_partitioned_resources(temporary.path());
        const std::filesystem::path catalog =
            temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(before.current_catalog_id) + ".bin");
        std::vector<std::byte> bytes = read_file(catalog);
        bytes.back() ^= std::byte{1};
        write_file(catalog, bytes);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "catalog corruption was accepted");
    }
    {
        TemporaryDirectory temporary("catalog-symlink");
        build_store(temporary.path());
        const PartitionedResourceInventory before =
            throwdb::detail::inspect_partitioned_resources(temporary.path());
        const std::filesystem::path catalog =
            temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(before.current_catalog_id) + ".bin");
        const std::filesystem::path saved = catalog.string() + ".saved";
        std::filesystem::rename(catalog, saved);
        std::filesystem::create_symlink(saved.filename(), catalog);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "symlink current catalog was followed");
    }
    {
        TemporaryDirectory temporary("manifest-symlink");
        build_store(temporary.path());
        std::filesystem::rename(
            temporary.path() / "manifest.bin",
            temporary.path() / "manifest.saved");
        std::filesystem::create_symlink(
            "manifest.saved", temporary.path() / "manifest.bin");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::inspect_partitioned_resources(
                        temporary.path()));
            },
            "symlink manifest was followed");
    }
}

void test_nonregular_entries_are_safe_and_partial_pairs_are_retried() {
    TemporaryDirectory temporary("nonregular");
    build_store(temporary.path());
    const Uuid128 symlink_id = uuid(200);
    const Uuid128 fifo_id = uuid(201);
    const Uuid128 directory_id = uuid(202);
    const Uuid128 partial_id = uuid(203);
    const std::filesystem::path symlink_path =
        temporary.path() / "wal"
        / ("wal-" + uuid_token(symlink_id) + ".wal");
    const std::filesystem::path fifo_path =
        temporary.path() / "index" / "runs"
        / ("run-" + uuid_token(fifo_id) + ".idx");
    const std::filesystem::path directory_path =
        temporary.path() / "index" / "partitions"
        / ("partition-0-" + uuid_token(directory_id) + ".idx");
    const std::filesystem::path partial_data =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(partial_id) + ".dat");
    const Uuid128 interrupted_id = uuid(204);
    const std::filesystem::path interrupted_data =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(interrupted_id) + ".dat");
    const std::filesystem::path interrupted_ids =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(interrupted_id) + ".ids");
    std::filesystem::create_symlink(
        temporary.path() / "manifest.bin", symlink_path);
    require(::mkfifo(fifo_path.c_str(), 0600) == 0, "cannot create FIFO fixture");
    std::filesystem::create_directory(directory_path);
    write_sized_file(partial_data, 23);
    write_sized_file(interrupted_data, 29);
    write_sized_file(interrupted_ids, 31);
    // Model an earlier complete-pair pass that unlinked its first member and
    // then stopped before the second unlink. The next fresh scan must safely
    // converge the remaining singleton rather than leak it forever.
    require(
        std::filesystem::remove(interrupted_data),
        "cannot create interrupted segment-pair fixture");

    const PartitionedResourceInventory inventory =
        throwdb::detail::inspect_partitioned_resources(temporary.path());
    require_class(
        inventory,
        "wal/wal-" + uuid_token(symlink_id) + ".wal",
        PartitionedInventoryClass::UnknownDoNotTouch);
    require_class(
        inventory,
        "index/runs/run-" + uuid_token(fifo_id) + ".idx",
        PartitionedInventoryClass::UnknownDoNotTouch);
    require_class(
        inventory,
        "index/partitions/partition-0-" + uuid_token(directory_id) + ".idx",
        PartitionedInventoryClass::UnknownDoNotTouch);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(partial_id) + ".dat",
        PartitionedInventoryClass::PartialUnreferencedSegmentPair);
    require_class(
        inventory,
        "segments/segment-" + uuid_token(interrupted_id) + ".ids",
        PartitionedInventoryClass::PartialUnreferencedSegmentPair);

    const auto result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), inventory.current_generation,
        inventory.current_catalog_id, permissive_options());
    require(result.durability_complete, "empty collector was not durable");
    require(std::filesystem::is_symlink(symlink_path), "collector removed symlink");
    struct stat status {};
    require(::lstat(fifo_path.c_str(), &status) == 0 && S_ISFIFO(status.st_mode),
            "collector removed or followed FIFO");
    require(std::filesystem::is_directory(directory_path),
            "collector removed strict-name directory");
    require(!std::filesystem::exists(partial_data),
            "collector retained a partial data singleton");
    require(!std::filesystem::exists(interrupted_ids),
            "collector did not retry the remainder of an interrupted pair");
    require(
        result.synced_directories
            == std::vector{
                throwdb::detail::PartitionedInventoryArea::Segments},
        "partial-pair retry synced the wrong namespace");
    const auto removed_partial_count = std::count_if(
        result.entries.begin(), result.entries.end(),
        [](const auto& entry) {
            return entry.file.classification
                       == PartitionedInventoryClass::
                           PartialUnreferencedSegmentPair
                && entry.eligible && entry.selected && entry.removed
                && entry.durable
                && entry.disposition
                       == PartitionedOrphanDisposition::Removed;
        });
    require(removed_partial_count == 2,
            "partial segment singletons were not reported durably removed");
}

void test_current_referenced_segment_half_is_never_a_candidate() {
    TemporaryDirectory temporary("reachable-half");
    build_store(temporary.path());

    const PartitionedResourceInventory complete =
        throwdb::detail::inspect_partitioned_resources(temporary.path());
    std::vector<std::string> reachable_segment_paths;
    for (const PartitionedInventoryEntry& entry : complete.entries) {
        if (entry.area == throwdb::detail::PartitionedInventoryArea::Segments
            && entry.classification == PartitionedInventoryClass::Reachable) {
            reachable_segment_paths.push_back(entry.relative_path);
        }
    }
    require(reachable_segment_paths.size() == 2,
            "fixture did not publish one reachable segment pair");
    require(
        std::filesystem::remove(
            temporary.path() / reachable_segment_paths.front()),
        "cannot remove one reachable segment member");
    const std::string remaining = reachable_segment_paths.back();

    const PartitionedResourceInventory expected =
        throwdb::detail::inspect_partitioned_resources(temporary.path());
    require_class(
        expected, remaining, PartitionedInventoryClass::Reachable);
    const auto result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, permissive_options());
    require(std::filesystem::exists(temporary.path() / remaining),
            "collector removed a CURRENT-referenced segment half");
    const auto& entry = entry_for(expected, remaining);
    const auto collected = std::find_if(
        result.entries.begin(), result.entries.end(),
        [&entry](const auto& candidate) {
            return candidate.file.relative_path == entry.relative_path;
        });
    require(
        collected != result.entries.end()
            && collected->disposition
                   == PartitionedOrphanDisposition::NotCandidate
            && !collected->eligible && !collected->selected
            && !collected->removed,
        "CURRENT-referenced segment half gained deletion authority");
}

void test_pair_budgets_age_fsync_and_idempotence() {
    TemporaryDirectory temporary("budgets");
    build_store(temporary.path());
    const Uuid128 pair_id = uuid(300);
    const std::filesystem::path data =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(pair_id) + ".dat");
    const std::filesystem::path ids =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(pair_id) + ".ids");
    write_sized_file(data, 11);
    write_sized_file(ids, 13);
    const PartitionedResourceInventory expected =
        throwdb::detail::inspect_partitioned_resources(temporary.path());

    PartitionedOrphanCollectionOptions options = permissive_options();
    options.max_files = 1;
    auto result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, options);
    require(std::filesystem::exists(data) && std::filesystem::exists(ids),
            "file budget split a complete segment pair");
    require(
        std::count_if(
            result.entries.begin(), result.entries.end(),
            [](const auto& entry) {
                return entry.disposition
                    == PartitionedOrphanDisposition::MaximumFilesReached;
            }) == 2,
        "pair file budget did not classify both members together");

    options = permissive_options();
    options.max_bytes = 23;
    result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, options);
    require(std::filesystem::exists(data) && std::filesystem::exists(ids),
            "byte budget split a complete segment pair");
    require(
        std::count_if(
            result.entries.begin(), result.entries.end(),
            [](const auto& entry) {
                return entry.disposition
                    == PartitionedOrphanDisposition::MaximumBytesReached;
            }) == 2,
        "pair byte budget did not classify both members together");

    options = permissive_options();
    options.min_age_ns = 10ULL * 1'000'000'000ULL;
    result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, options);
    require(std::filesystem::exists(data) && std::filesystem::exists(ids),
            "minimum age did not protect a complete pair");

    options = permissive_options();
    result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, options);
    require(!std::filesystem::exists(data) && !std::filesystem::exists(ids),
            "eligible complete pair was not removed");
    require(result.durability_complete, "directory fsync was not successful");
    require(
        result.synced_directories
            == std::vector{throwdb::detail::PartitionedInventoryArea::Segments},
        "collector fsynced the wrong namespace");
    require(
        std::count_if(
            result.entries.begin(), result.entries.end(),
            [](const auto& entry) {
                return entry.removed && entry.durable;
            }) == 2,
        "removed pair was not reported durable");

    const auto second = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, options);
    require(
        std::none_of(
            second.entries.begin(), second.entries.end(),
            [](const auto& entry) { return entry.removed; }),
        "idempotent collector removed a second resource");
    require(second.synced_directories.empty(),
            "idempotent collector performed an unnecessary fsync");
}

void test_expected_current_and_identity_revalidation() {
    TemporaryDirectory temporary("identity");
    build_store(temporary.path());
    const Uuid128 run_id = uuid(400);
    const std::filesystem::path run =
        temporary.path() / "index" / "runs"
        / ("run-" + uuid_token(run_id) + ".idx");
    write_sized_file(run, 31, 'a');
    const PartitionedResourceInventory expected =
        throwdb::detail::inspect_partitioned_resources(temporary.path());

    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::detail::collect_partitioned_orphans(
                temporary.path(), expected.current_generation + 1,
                expected.current_catalog_id, permissive_options()));
        },
        "expected CURRENT mismatch was accepted");
    require(std::filesystem::exists(run),
            "expected CURRENT mismatch removed a file");

    bool replaced = false;
    PartitionedOrphanCollectionHooks hooks;
    hooks.before_revalidate =
        [&run, &replaced](
            const std::size_t,
            const PartitionedInventoryEntry& entry) {
            if (replaced || entry.relative_path
                                != "index/runs/" + run.filename().string()) {
                return;
            }
            replaced = true;
            std::filesystem::remove(run);
            write_sized_file(run, 37, 'b');
        };
    const auto result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, permissive_options(), hooks);
    require(replaced, "identity replacement hook did not run");
    require(std::filesystem::file_size(run) == 37,
            "collector unlinked an identity replacement");
    const auto position = std::find_if(
        result.entries.begin(), result.entries.end(),
        [&run](const auto& entry) {
            return entry.file.relative_path
                == "index/runs/" + run.filename().string();
        });
    require(position != result.entries.end(), "identity result omitted run");
    require(
        position->disposition == PartitionedOrphanDisposition::IdentityChanged
            && !position->removed,
        "identity replacement was not rejected");
}

void test_all_owned_candidate_namespaces_are_collected() {
    TemporaryDirectory temporary("namespaces");
    build_store(temporary.path());
    const Uuid128 temp_id = uuid(500);
    const Uuid128 wal_id = uuid(501);
    const Uuid128 snapshot_id = uuid(502);
    const Uuid128 run_id = uuid(503);
    const Uuid128 catalog_id = uuid(504);
    const std::array<std::filesystem::path, 5> candidates{
        temporary.path()
            / ("CURRENT." + uuid_token(temp_id) + ".tmp"),
        temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(catalog_id) + ".bin"),
        temporary.path() / "wal"
            / ("wal-" + uuid_token(wal_id) + ".wal"),
        temporary.path() / "index" / "partitions"
            / ("partition-3-" + uuid_token(snapshot_id) + ".idx"),
        temporary.path() / "index" / "runs"
            / ("run-" + uuid_token(run_id) + ".idx")};
    for (std::size_t index = 0; index != candidates.size(); ++index) {
        write_sized_file(candidates[index], index + 1);
    }
    const PartitionedResourceInventory expected =
        throwdb::detail::inspect_partitioned_resources(temporary.path());
    const auto result = throwdb::detail::collect_partitioned_orphans(
        temporary.path(), expected.current_generation,
        expected.current_catalog_id, permissive_options());
    for (const std::filesystem::path& candidate : candidates) {
        require(!std::filesystem::exists(candidate),
                "collector retained owned candidate '" + candidate.string()
                    + "'");
    }
    require(result.durability_complete, "multi-namespace fsync failed");
    require(result.synced_directories.size() == candidates.size(),
            "collector did not fsync every affected namespace");
}

void test_inventory_scan_limit_is_fail_closed() {
    TemporaryDirectory temporary("scan-limit");
    build_store(temporary.path());
    const PartitionedResourceInventory expected =
        throwdb::detail::inspect_partitioned_resources(temporary.path());

    std::array<std::filesystem::path, 3> candidates;
    for (std::size_t index = 0; index != candidates.size(); ++index) {
        candidates[index] =
            temporary.path() / "index" / "runs"
            / ("run-" + uuid_token(uuid(550 + index)) + ".idx");
        write_sized_file(candidates[index], index + 1);
    }

    PartitionedOrphanCollectionHooks hooks;
    hooks.maximum_uncataloged_scan_entries = candidates.size() - 1;
    bool rejected = false;
    try {
        static_cast<void>(throwdb::detail::collect_partitioned_orphans(
            temporary.path(), expected.current_generation,
            expected.current_catalog_id, permissive_options(), hooks));
    } catch (const throwdb::StorageError& error) {
        rejected = true;
        require(
            std::string(error.what()).find("hard safety limit")
                != std::string::npos,
            "scan-limit rejection did not explain the safety bound");
    }
    require(rejected, "inventory accepted one entry beyond its scan limit");
    for (const std::filesystem::path& candidate : candidates) {
        require(
            std::filesystem::exists(candidate),
            "scan-limit rejection removed an orphan before completing the "
            "inventory");
    }
}

void test_native_collection_public_surface() {
    TemporaryDirectory temporary("public");
    build_store(temporary.path());
    const Uuid128 temp_id = uuid(600);
    const Uuid128 wal_id = uuid(601);
    const Uuid128 segment_id = uuid(602);
    const Uuid128 partial_id = uuid(603);
    const Uuid128 snapshot_id = uuid(604);
    const Uuid128 run_id = uuid(605);
    const Uuid128 catalog_id = uuid(606);
    const std::filesystem::path root_temp =
        temporary.path() / ("CURRENT." + uuid_token(temp_id) + ".tmp");
    const std::filesystem::path wal =
        temporary.path() / "wal"
        / ("wal-" + uuid_token(wal_id) + ".wal");
    const std::filesystem::path segment_data =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(segment_id) + ".dat");
    const std::filesystem::path segment_ids =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(segment_id) + ".ids");
    const std::filesystem::path partial =
        temporary.path() / "segments"
        / ("segment-" + uuid_token(partial_id) + ".dat");
    const std::filesystem::path superseded_catalog =
        temporary.path() / "catalogs"
        / ("catalog-" + uuid_token(catalog_id) + ".bin");
    const std::filesystem::path snapshot =
        temporary.path() / "index" / "partitions"
        / ("partition-2-" + uuid_token(snapshot_id) + ".idx");
    const std::filesystem::path run =
        temporary.path() / "index" / "runs"
        / ("run-" + uuid_token(run_id) + ".idx");
    for (const std::filesystem::path& path : {
             root_temp, superseded_catalog, wal, segment_data, segment_ids,
             partial, snapshot, run}) {
        write_sized_file(path, 7);
    }

    const throwdb::NativeOrphanInventory inspected =
        throwdb::NativeCollection::inspect_orphans(temporary.path());
    require(
        !public_entry_for(inspected, "CURRENT").candidate,
        "public inventory marked CURRENT as a deletion candidate");
    require(
        public_entry_for(
            inspected,
            "catalogs/catalog-" + uuid_token(catalog_id) + ".bin")
                .candidate,
        "public inventory did not mark a format-7 superseded catalog as a "
        "candidate");
    require(
        public_entry_for(
            inspected,
            "segments/segment-" + uuid_token(segment_id) + ".dat")
                .classification
            == throwdb::NativeOrphanClass::
                CompleteUnreferencedSegmentPair,
        "public inventory lost complete segment-pair classification");
    require(
        public_entry_for(
            inspected,
            "segments/segment-" + uuid_token(segment_id) + ".dat")
            .candidate,
        "public inventory did not mark a complete segment pair as a "
        "candidate");
    require(
        public_entry_for(
            inspected,
            "segments/segment-" + uuid_token(partial_id) + ".dat")
                .classification
            == throwdb::NativeOrphanClass::PartialUnreferencedSegmentPair,
        "public inventory lost partial segment-pair classification");
    require(
        public_entry_for(
            inspected,
            "segments/segment-" + uuid_token(partial_id) + ".dat")
            .candidate,
        "public inventory did not mark a partial segment singleton as a "
        "candidate");
    require(
        public_entry_for(
            inspected,
            "index/partitions/partition-2-" + uuid_token(snapshot_id)
                + ".idx")
                .classification
            == throwdb::NativeOrphanClass::UnreferencedSnapshot,
        "public inventory lost snapshot classification");
    require(
        public_entry_for(
            inspected,
            "index/runs/run-" + uuid_token(run_id) + ".idx")
                .classification
            == throwdb::NativeOrphanClass::UnreferencedRun,
        "public inventory lost run classification");

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    const throwdb::NativeOrphanInventory through_writer =
        writer->collect_orphans();
    require(
        through_writer.current_generation == inspected.current_generation
            && through_writer.current_catalog_id
                   == inspected.current_catalog_id,
        "writer inventory selected a different CURRENT view");
    throwdb::NativeOrphanCollectionOptions options;
    options.snapshot_unix_ns = now_ns() + 1'000'000'000ULL;
    options.min_age_ns = 0;
    options.max_files = 1024;
    options.max_bytes = 1ULL << 30;
    const throwdb::NativeOrphanCollectionResult reclaimed =
        writer->reclaim_orphans(options);
    require(
        reclaimed.synced_directories
            == std::vector<std::string>{
                ".", "catalogs", "segments", "index/partitions",
                "index/runs", "wal"},
        "public collector mapped synced format-7 directories incorrectly");
    require(
        !std::filesystem::exists(root_temp)
            && !std::filesystem::exists(superseded_catalog)
            && !std::filesystem::exists(wal)
            && !std::filesystem::exists(segment_data)
            && !std::filesystem::exists(segment_ids)
            && !std::filesystem::exists(partial)
            && !std::filesystem::exists(snapshot)
            && !std::filesystem::exists(run),
        "writer public collector retained an eligible format-7 orphan");
    const auto partial_result = std::find_if(
        reclaimed.entries.begin(), reclaimed.entries.end(),
        [&partial](const throwdb::NativeOrphanCollectionEntry& entry) {
            return entry.file.path
                == "segments/" + partial.filename().string();
        });
    require(
        partial_result != reclaimed.entries.end()
            && partial_result->disposition
                   == throwdb::NativeOrphanDisposition::Removed
            && partial_result->file.candidate
            && partial_result->eligible && partial_result->selected
            && partial_result->removed && partial_result->durable,
        "partial segment singleton was not a public deletion candidate");
    writer.reset();

    const Uuid128 second_wal_id = uuid(607);
    const std::filesystem::path second_wal =
        temporary.path() / "wal"
        / ("wal-" + uuid_token(second_wal_id) + ".wal");
    write_sized_file(second_wal, 9);
    const throwdb::NativeOrphanCollectionResult static_reclaimed =
        throwdb::NativeCollection::reclaim_orphans(
            temporary.path(), options);
    require(
        !std::filesystem::exists(second_wal)
            && static_reclaimed.synced_directories
                   == std::vector<std::string>{"wal"},
        "lightweight static collector did not reclaim a format-7 WAL");

    auto reader = throwdb::NativeCollection::open(temporary.path());
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(reader->collect_orphans()); },
        "read-only format-7 handle collected orphans");
}

}  // namespace

int main() {
    using Test = std::pair<const char*, void (*)()>;
    const std::array<Test, 9> tests{
        Test{"reachability_and_strict_filename_classification",
             &test_reachability_and_strict_filename_classification},
        Test{"authoritative_validation_and_catalog_cap",
             &test_authoritative_validation_and_catalog_cap},
        Test{"nonregular_entries_are_safe_and_partial_pairs_are_retried",
             &test_nonregular_entries_are_safe_and_partial_pairs_are_retried},
        Test{"current_referenced_segment_half_is_never_a_candidate",
             &test_current_referenced_segment_half_is_never_a_candidate},
        Test{"pair_budgets_age_fsync_and_idempotence",
             &test_pair_budgets_age_fsync_and_idempotence},
        Test{"expected_current_and_identity_revalidation",
             &test_expected_current_and_identity_revalidation},
        Test{"all_owned_candidate_namespaces_are_collected",
             &test_all_owned_candidate_namespaces_are_collected},
        Test{"inventory_scan_limit_is_fail_closed",
             &test_inventory_scan_limit_is_fail_closed},
        Test{"native_collection_public_surface",
             &test_native_collection_public_surface},
    };
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
