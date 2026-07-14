#include "partitioned_resource_inventory.hpp"

#include "crc32c.hpp"
#include "internal.hpp"
#include "partitioned_index_format.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include <sys/stat.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

constexpr std::size_t kAreaCount = 6;

struct ScannedEntry {
    PartitionedInventoryArea area{PartitionedInventoryArea::Root};
    std::string name;
    std::string relative_path;
    bool regular{false};
    std::uint64_t bytes{0};
    std::int64_t modified_unix_ns{0};
    PartitionedFileIdentity identity;
};

struct ParsedSnapshotName {
    std::uint32_t partition_id{0};
    Uuid128 snapshot_id;
};

struct WorkItem {
    std::vector<std::size_t> entries;
};

[[nodiscard]] std::string uuid_token(const Uuid128 value) {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> token{};
    const auto put_word = [&token](
                              const std::uint64_t word,
                              const std::size_t offset) {
        for (std::size_t nibble = 0; nibble != 16; ++nibble) {
            const unsigned shift = static_cast<unsigned>((15 - nibble) * 4);
            token[offset + nibble] = hex[(word >> shift) & 0xfU];
        }
    };
    put_word(value.high, 0);
    put_word(value.low, 16);
    return std::string(token.begin(), token.end());
}

[[nodiscard]] std::optional<std::uint64_t> parse_hex_word(
    const std::string_view token) noexcept {
    if (token.size() != 16) {
        return std::nullopt;
    }
    std::uint64_t result = 0;
    for (const char character : token) {
        std::uint8_t digit = 0;
        if (character >= '0' && character <= '9') {
            digit = static_cast<std::uint8_t>(character - '0');
        } else if (character >= 'a' && character <= 'f') {
            digit = static_cast<std::uint8_t>(character - 'a' + 10);
        } else {
            return std::nullopt;
        }
        result = (result << 4U) | digit;
    }
    return result;
}

[[nodiscard]] std::optional<Uuid128> parse_uuid_filename(
    const std::string_view name,
    const std::string_view prefix,
    const std::string_view suffix) noexcept {
    constexpr std::size_t token_size = 32;
    if (name.size() != prefix.size() + token_size + suffix.size()
        || !name.starts_with(prefix) || !name.ends_with(suffix)) {
        return std::nullopt;
    }
    const std::string_view token = name.substr(prefix.size(), token_size);
    const auto high = parse_hex_word(token.substr(0, 16));
    const auto low = parse_hex_word(token.substr(16));
    if (!high.has_value() || !low.has_value()) {
        return std::nullopt;
    }
    const Uuid128 result{*low, *high};
    return result.empty() ? std::nullopt : std::optional<Uuid128>{result};
}

[[nodiscard]] std::optional<std::uint32_t> parse_decimal_partition(
    const std::string_view token) noexcept {
    if (token.empty() || (token.size() != 1 && token.front() == '0')) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (const char character : token) {
        if (character < '0' || character > '9') {
            return std::nullopt;
        }
        const std::uint8_t digit =
            static_cast<std::uint8_t>(character - '0');
        if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        value = value * 10 + digit;
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::optional<ParsedSnapshotName> parse_snapshot_filename(
    const std::string_view name,
    const std::string_view suffix = ".idx") noexcept {
    constexpr std::string_view prefix = "partition-";
    constexpr std::size_t uuid_size = 32;
    if (!name.starts_with(prefix) || !name.ends_with(suffix)
        || name.size() <= prefix.size() + 1 + uuid_size + suffix.size()) {
        return std::nullopt;
    }
    const std::size_t uuid_begin = name.size() - suffix.size() - uuid_size;
    if (uuid_begin <= prefix.size() || name[uuid_begin - 1] != '-') {
        return std::nullopt;
    }
    const auto partition = parse_decimal_partition(
        name.substr(prefix.size(), uuid_begin - prefix.size() - 1));
    const auto high = parse_hex_word(name.substr(uuid_begin, 16));
    const auto low = parse_hex_word(name.substr(uuid_begin + 16, 16));
    if (!partition.has_value() || !high.has_value() || !low.has_value()) {
        return std::nullopt;
    }
    const Uuid128 id{*low, *high};
    if (id.empty()) {
        return std::nullopt;
    }
    return ParsedSnapshotName{*partition, id};
}

[[nodiscard]] std::int64_t unix_timestamp_ns(
    const std::int64_t seconds,
    const std::int64_t nanoseconds,
    const std::filesystem::path& path,
    const std::string_view kind) {
    constexpr std::int64_t billion = 1'000'000'000;
    if (nanoseconds < 0 || nanoseconds >= billion
        || seconds > std::numeric_limits<std::int64_t>::max() / billion
        || seconds < std::numeric_limits<std::int64_t>::min() / billion) {
        throw StorageError(
            "format-7 inventory " + std::string(kind)
            + " is invalid for '" + path.string() + "'");
    }
    const std::int64_t scaled = seconds * billion;
    if (seconds >= 0
        && nanoseconds
               > std::numeric_limits<std::int64_t>::max() - scaled) {
        throw StorageError(
            "format-7 inventory " + std::string(kind)
            + " overflows for '" + path.string() + "'");
    }
    return scaled + nanoseconds;
}

[[nodiscard]] std::int64_t unix_mtime_ns(
    const struct stat& status,
    const std::filesystem::path& path) {
#if defined(__APPLE__)
    const auto seconds = status.st_mtimespec.tv_sec;
    const auto nanoseconds = status.st_mtimespec.tv_nsec;
#else
    const auto seconds = status.st_mtim.tv_sec;
    const auto nanoseconds = status.st_mtim.tv_nsec;
#endif
    if (!std::in_range<std::int64_t>(seconds)
        || !std::in_range<std::int64_t>(nanoseconds)) {
        throw StorageError(
            "format-7 inventory mtime does not fit int64: '"
            + path.string() + "'");
    }
    return unix_timestamp_ns(
        static_cast<std::int64_t>(seconds),
        static_cast<std::int64_t>(nanoseconds), path, "mtime");
}

[[nodiscard]] std::int64_t unix_ctime_ns(
    const struct stat& status,
    const std::filesystem::path& path) {
#if defined(__APPLE__)
    const auto seconds = status.st_ctimespec.tv_sec;
    const auto nanoseconds = status.st_ctimespec.tv_nsec;
#else
    const auto seconds = status.st_ctim.tv_sec;
    const auto nanoseconds = status.st_ctim.tv_nsec;
#endif
    if (!std::in_range<std::int64_t>(seconds)
        || !std::in_range<std::int64_t>(nanoseconds)) {
        throw StorageError(
            "format-7 inventory ctime does not fit int64: '"
            + path.string() + "'");
    }
    return unix_timestamp_ns(
        static_cast<std::int64_t>(seconds),
        static_cast<std::int64_t>(nanoseconds), path, "ctime");
}

[[nodiscard]] std::uint64_t status_word(
    const auto value,
    const std::filesystem::path& path,
    const std::string_view kind) {
    if (!std::in_range<std::uint64_t>(value)) {
        throw StorageError(
            "format-7 inventory " + std::string(kind)
            + " does not fit uint64: '" + path.string() + "'");
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] PartitionedFileIdentity file_identity(
    const struct stat& status,
    const std::filesystem::path& path,
    const bool regular) {
    std::uint64_t bytes = 0;
    if (regular) {
        if (status.st_size < 0
            || !std::in_range<std::uint64_t>(status.st_size)) {
            throw StorageError(
                "format-7 inventory file size exceeds uint64: '"
                + path.string() + "'");
        }
        bytes = static_cast<std::uint64_t>(status.st_size);
    }
    return PartitionedFileIdentity{
        status_word(status.st_dev, path, "device"),
        status_word(status.st_ino, path, "inode"),
        status_word(status.st_mode, path, "mode"),
        bytes,
        unix_mtime_ns(status, path),
        unix_ctime_ns(status, path)};
}

void require_real_directory(
    const std::filesystem::path& path,
    const std::string_view description) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect " + std::string(description), path));
    }
    if (!S_ISDIR(status.st_mode)) {
        throw CorruptionError(
            std::string(description) + " is not a real directory: '"
            + path.string() + "'");
    }
}

[[nodiscard]] FileDescriptor open_regular_nofollow(
    const std::filesystem::path& path,
    const std::string_view description) {
    struct stat before {};
    if (::lstat(path.c_str(), &before) < 0) {
        throw StorageError(
            system_error_message("cannot inspect " + std::string(description), path));
    }
    if (!S_ISREG(before.st_mode)) {
        throw CorruptionError(
            std::string(description) + " is not a regular file: '"
            + path.string() + "'");
    }
    int flags = O_RDONLY | O_CLOEXEC | O_NONBLOCK;
#if defined(O_NOFOLLOW)
    flags |= O_NOFOLLOW;
#endif
    const int raw = ::open(path.c_str(), flags);
    if (raw < 0) {
        throw StorageError(
            system_error_message("cannot open " + std::string(description), path));
    }
    FileDescriptor result(raw);
    struct stat after {};
    if (::fstat(result.get(), &after) < 0) {
        throw StorageError(
            system_error_message("cannot inspect " + std::string(description), path));
    }
    if (!S_ISREG(after.st_mode)) {
        throw CorruptionError(
            std::string(description) + " is not a regular file: '"
            + path.string() + "'");
    }
    return result;
}

[[nodiscard]] std::vector<std::byte> read_exact_regular(
    const std::filesystem::path& path,
    const std::uint64_t expected_size,
    const std::string_view description) {
    if (expected_size > std::numeric_limits<std::size_t>::max()) {
        throw CorruptionError(
            std::string(description) + " does not fit the process address space");
    }
    FileDescriptor fd = open_regular_nofollow(path, description);
    if (file_size(fd.get(), path) != expected_size) {
        throw CorruptionError(
            std::string(description) + " has unexpected size: '"
            + path.string() + "'");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(expected_size));
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    return bytes;
}

[[nodiscard]] std::filesystem::path area_path(
    const std::filesystem::path& root,
    const PartitionedInventoryArea area) {
    switch (area) {
        case PartitionedInventoryArea::Root:
            return root;
        case PartitionedInventoryArea::Catalogs:
            return root / "catalogs";
        case PartitionedInventoryArea::Segments:
            return root / "segments";
        case PartitionedInventoryArea::Partitions:
            return root / "index" / "partitions";
        case PartitionedInventoryArea::Runs:
            return root / "index" / "runs";
        case PartitionedInventoryArea::Wal:
            return root / "wal";
    }
    throw StorageError("format-7 inventory contains an unknown area");
}

[[nodiscard]] std::filesystem::path relative_area_path(
    const PartitionedInventoryArea area) {
    switch (area) {
        case PartitionedInventoryArea::Root:
            return {};
        case PartitionedInventoryArea::Catalogs:
            return "catalogs";
        case PartitionedInventoryArea::Segments:
            return "segments";
        case PartitionedInventoryArea::Partitions:
            return std::filesystem::path("index") / "partitions";
        case PartitionedInventoryArea::Runs:
            return std::filesystem::path("index") / "runs";
        case PartitionedInventoryArea::Wal:
            return "wal";
    }
    throw StorageError("format-7 inventory contains an unknown area");
}

[[nodiscard]] std::size_t area_index(
    const PartitionedInventoryArea area) noexcept {
    return static_cast<std::size_t>(area);
}

[[nodiscard]] ScannedEntry inspect_entry(
    const std::filesystem::directory_entry& entry,
    const PartitionedInventoryArea area) {
    ScannedEntry result;
    result.area = area;
    result.name = entry.path().filename().string();
    result.relative_path =
        (relative_area_path(area) / result.name).generic_string();
    struct stat status {};
    if (::lstat(entry.path().c_str(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect format-7 inventory entry", entry.path()));
    }
    result.regular = S_ISREG(status.st_mode);
    result.identity = file_identity(status, entry.path(), result.regular);
    result.bytes = result.identity.bytes;
    result.modified_unix_ns = result.identity.modified_unix_ns;
    return result;
}

void scan_directory(
    const std::filesystem::path& root,
    const PartitionedInventoryArea area,
    const bool root_filter,
    const std::size_t maximum_entries,
    std::vector<ScannedEntry>& output) {
    const std::filesystem::path directory = area_path(root, area);
    require_real_directory(directory, "format-7 inventory namespace");
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error) {
        throw StorageError(
            "cannot scan format-7 inventory namespace '" + directory.string()
            + "': " + error.message());
    }
    const std::filesystem::directory_iterator end;
    while (iterator != end) {
        const std::string name = iterator->path().filename().string();
        if (!root_filter || name.starts_with("CURRENT")
            || name == "manifest.bin.tmp") {
            if (output.size() >= maximum_entries) {
                throw StorageError(
                    "format-7 inventory scan exceeds its hard safety limit "
                    "of "
                    + std::to_string(maximum_entries)
                    + " entries; reduce stale or unexpected files in the "
                      "collection namespaces before retrying");
            }
            output.push_back(inspect_entry(*iterator, area));
        }
        iterator.increment(error);
        if (error) {
            throw StorageError(
                "cannot continue format-7 inventory scan in '"
                + directory.string() + "': " + error.message());
        }
    }
}

[[nodiscard]] PartitionedManifestV2 read_manifest(
    const std::filesystem::path& root) {
    const std::vector<std::byte> bytes = read_exact_regular(
        root / "manifest.bin", kPartitionedHeaderSize,
        "format-7 manifest");
    return decode_partitioned_manifest_v2(bytes);
}

struct AuthoritativeView {
    PartitionedManifestV2 manifest;
    PartitionedCurrentV1 current;
    PartitionedCatalogV3 catalog;
};

[[nodiscard]] AuthoritativeView read_authoritative_view(
    const std::filesystem::path& root) {
    AuthoritativeView result;
    result.manifest = read_manifest(root);
    const std::vector<std::byte> current_bytes = read_exact_regular(
        root / "CURRENT", kPartitionedHeaderSize, "format-7 CURRENT");
    result.current = decode_partitioned_current_v1(
        current_bytes, result.manifest);
    if (result.current.catalog_file_size > kPartitionedMaximumCatalogFileSize) {
        throw CorruptionError("format-7 CURRENT catalog size exceeds 512 MiB");
    }
    const std::filesystem::path catalog_path =
        root / "catalogs"
        / ("catalog-" + uuid_token(result.current.catalog_id) + ".bin");
    const std::vector<std::byte> catalog_bytes = read_exact_regular(
        catalog_path, result.current.catalog_file_size,
        "current format-7 catalog");
    if (crc32c(catalog_bytes) != result.current.catalog_crc32c) {
        throw CorruptionError(
            "current format-7 catalog checksum does not match CURRENT");
    }
    result.catalog = decode_partitioned_catalog_v3(catalog_bytes);
    if (result.catalog.collection_id != result.manifest.collection_id
        || result.catalog.schema_id != result.manifest.schema_id
        || result.catalog.catalog_id != result.current.catalog_id
        || result.catalog.generation != result.current.generation
        || result.catalog.partition_bits != result.manifest.partition_bits
        || result.catalog.hash_seed != result.manifest.hash_seed) {
        throw CorruptionError(
            "current format-7 catalog identity does not match manifest/CURRENT");
    }
    return result;
}

[[nodiscard]] std::set<std::string> reachable_paths(
    const AuthoritativeView& view) {
    std::set<std::string> result;
    result.insert("CURRENT");
    result.insert(
        (std::filesystem::path("catalogs")
         / ("catalog-" + uuid_token(view.catalog.catalog_id) + ".bin"))
            .generic_string());
    for (const PartitionedSegmentDescriptor& descriptor
         : view.catalog.segments) {
        const std::string stem =
            "segment-" + uuid_token(descriptor.segment_id);
        result.insert(
            (std::filesystem::path("segments") / (stem + ".dat"))
                .generic_string());
        result.insert(
            (std::filesystem::path("segments") / (stem + ".ids"))
                .generic_string());
    }
    for (const PartitionSnapshotDescriptor& descriptor
         : view.catalog.snapshots) {
        result.insert(
            (std::filesystem::path("index") / "partitions"
             / ("partition-" + std::to_string(descriptor.partition_id) + "-"
                + uuid_token(descriptor.snapshot_id) + ".idx"))
                .generic_string());
    }
    for (const PartitionIndexRunDescriptor& descriptor : view.catalog.runs) {
        result.insert(
            (std::filesystem::path("index") / "runs"
             / ("run-" + uuid_token(descriptor.run_id) + ".idx"))
                .generic_string());
    }
    for (const MutationWalEntry& entry : view.catalog.wal_entries) {
        result.insert(
            (std::filesystem::path("wal")
             / ("wal-" + uuid_token(entry.wal_id) + ".wal"))
                .generic_string());
    }
    return result;
}

[[nodiscard]] bool segment_staging_name(
    const std::string_view name) noexcept {
    return parse_uuid_filename(name, ".segment-", ".dat.staging").has_value()
        || parse_uuid_filename(name, ".segment-", ".ids.staging").has_value();
}

[[nodiscard]] bool snapshot_temp_name(const std::string_view name) noexcept {
    return parse_snapshot_filename(name, ".idx.tmp").has_value();
}

[[nodiscard]] PartitionedInventoryClass classify_entry(
    const ScannedEntry& entry,
    const Uuid128 current_catalog,
    const std::set<std::string>& reachable,
    const std::map<Uuid128, std::uint8_t>& unreferenced_segment_parts) {
    if (!entry.regular) {
        return PartitionedInventoryClass::UnknownDoNotTouch;
    }
    if (reachable.contains(entry.relative_path)) {
        return PartitionedInventoryClass::Reachable;
    }
    switch (entry.area) {
        case PartitionedInventoryArea::Root:
            if (entry.name == "manifest.bin.tmp"
                || parse_uuid_filename(entry.name, "CURRENT.", ".tmp")
                       .has_value()) {
                return PartitionedInventoryClass::TempPublication;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
        case PartitionedInventoryArea::Catalogs:
            if (const auto id = parse_uuid_filename(
                    entry.name, "catalog-", ".bin");
                id.has_value()) {
                return *id == current_catalog
                           ? PartitionedInventoryClass::Reachable
                           : PartitionedInventoryClass::SupersededCatalog;
            }
            if (parse_uuid_filename(entry.name, "catalog-", ".bin.tmp")
                    .has_value()) {
                return PartitionedInventoryClass::TempPublication;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
        case PartitionedInventoryArea::Segments: {
            std::optional<Uuid128> id = parse_uuid_filename(
                entry.name, "segment-", ".dat");
            if (!id.has_value()) {
                id = parse_uuid_filename(entry.name, "segment-", ".ids");
            }
            if (id.has_value()) {
                const auto parts = unreferenced_segment_parts.find(*id);
                if (parts == unreferenced_segment_parts.end()) {
                    throw StorageError(
                        "format-7 inventory lost a parsed segment candidate");
                }
                return parts->second == 0x3U
                           ? PartitionedInventoryClass::
                                 CompleteUnreferencedSegmentPair
                           : PartitionedInventoryClass::
                                 PartialUnreferencedSegmentPair;
            }
            if (segment_staging_name(entry.name)) {
                return PartitionedInventoryClass::TempPublication;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
        }
        case PartitionedInventoryArea::Partitions:
            if (parse_snapshot_filename(entry.name).has_value()) {
                return PartitionedInventoryClass::UnreferencedSnapshot;
            }
            if (snapshot_temp_name(entry.name)) {
                return PartitionedInventoryClass::TempPublication;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
        case PartitionedInventoryArea::Runs:
            if (parse_uuid_filename(entry.name, "run-", ".idx").has_value()) {
                return PartitionedInventoryClass::UnreferencedRun;
            }
            if (parse_uuid_filename(entry.name, "run-", ".idx.tmp")
                    .has_value()) {
                return PartitionedInventoryClass::TempPublication;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
        case PartitionedInventoryArea::Wal:
            if (parse_uuid_filename(entry.name, "wal-", ".wal").has_value()) {
                return PartitionedInventoryClass::UnreferencedWal;
            }
            return PartitionedInventoryClass::UnknownDoNotTouch;
    }
    return PartitionedInventoryClass::UnknownDoNotTouch;
}

[[nodiscard]] bool ordinary_orphan_candidate(
    const PartitionedInventoryClass classification) noexcept {
    return classification == PartitionedInventoryClass::SupersededCatalog
        || classification == PartitionedInventoryClass::TempPublication
        || classification == PartitionedInventoryClass::UnreferencedWal
        || classification
               == PartitionedInventoryClass::PartialUnreferencedSegmentPair
        || classification == PartitionedInventoryClass::UnreferencedSnapshot
        || classification == PartitionedInventoryClass::UnreferencedRun;
}

[[nodiscard]] std::optional<Uuid128> segment_id_for_entry(
    const PartitionedInventoryEntry& entry) noexcept {
    std::optional<Uuid128> result = parse_uuid_filename(
        entry.name, "segment-", ".dat");
    if (!result.has_value()) {
        result = parse_uuid_filename(entry.name, "segment-", ".ids");
    }
    return result;
}

[[nodiscard]] FileDescriptor open_area_directory(
    const std::filesystem::path& path) {
    int flags = O_RDONLY | O_CLOEXEC;
#if defined(O_DIRECTORY)
    flags |= O_DIRECTORY;
#endif
#if defined(O_NOFOLLOW)
    flags |= O_NOFOLLOW;
#endif
    const int raw = ::open(path.c_str(), flags);
    if (raw < 0) {
        throw StorageError(
            system_error_message("cannot open format-7 orphan namespace", path));
    }
    FileDescriptor result(raw);
    struct stat status {};
    if (::fstat(result.get(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect format-7 orphan namespace", path));
    }
    if (!S_ISDIR(status.st_mode)) {
        throw CorruptionError(
            "format-7 orphan namespace is not a directory: '"
            + path.string() + "'");
    }
    return result;
}

[[nodiscard]] bool same_identity(
    const PartitionedFileIdentity& expected,
    const struct stat& actual,
    const std::filesystem::path& diagnostic) {
    return S_ISREG(actual.st_mode)
        && file_identity(actual, diagnostic, true) == expected;
}

void set_work_disposition(
    PartitionedOrphanCollectionResult& result,
    const WorkItem& work,
    const PartitionedOrphanDisposition disposition,
    const std::string& error = {}) {
    for (const std::size_t index : work.entries) {
        result.entries[index].disposition = disposition;
        if (!error.empty()) {
            result.entries[index].error = error;
        }
    }
}

}  // namespace

PartitionedResourceInventory inspect_partitioned_resources(
    const std::filesystem::path& store_path,
    const std::uint64_t maximum_uncataloged_entries) {
    require_real_directory(store_path, "format-7 collection root");
    require_real_directory(store_path / "index", "format-7 index namespace");
    const AuthoritativeView view = read_authoritative_view(store_path);
    const std::set<std::string> reachable = reachable_paths(view);
    if (maximum_uncataloged_entries
        > std::numeric_limits<std::size_t>::max() - reachable.size()) {
        throw InvalidArgumentError(
            "format-7 inventory scan limit exceeds the process address "
            "space");
    }
    const std::size_t maximum_entries =
        reachable.size()
        + static_cast<std::size_t>(maximum_uncataloged_entries);

    std::vector<ScannedEntry> scanned;
    scan_directory(
        store_path, PartitionedInventoryArea::Root, true, maximum_entries,
        scanned);
    scan_directory(
        store_path, PartitionedInventoryArea::Catalogs, false,
        maximum_entries, scanned);
    scan_directory(
        store_path, PartitionedInventoryArea::Segments, false,
        maximum_entries, scanned);
    scan_directory(
        store_path, PartitionedInventoryArea::Partitions, false,
        maximum_entries, scanned);
    scan_directory(
        store_path, PartitionedInventoryArea::Runs, false, maximum_entries,
        scanned);
    scan_directory(
        store_path, PartitionedInventoryArea::Wal, false, maximum_entries,
        scanned);

    std::map<Uuid128, std::uint8_t> unreferenced_segment_parts;
    for (const ScannedEntry& entry : scanned) {
        if (!entry.regular || entry.area != PartitionedInventoryArea::Segments
            || reachable.contains(entry.relative_path)) {
            continue;
        }
        if (const auto id = parse_uuid_filename(
                entry.name, "segment-", ".dat");
            id.has_value()) {
            unreferenced_segment_parts[*id] |= 0x1U;
        } else if (const auto id = parse_uuid_filename(
                       entry.name, "segment-", ".ids");
                   id.has_value()) {
            unreferenced_segment_parts[*id] |= 0x2U;
        }
    }

    PartitionedResourceInventory inventory;
    inventory.current_generation = view.catalog.generation;
    inventory.current_catalog_id = view.catalog.catalog_id;
    inventory.entries.reserve(scanned.size());
    for (ScannedEntry& entry : scanned) {
        const PartitionedInventoryClass classification = classify_entry(
            entry, view.catalog.catalog_id, reachable,
            unreferenced_segment_parts);
        inventory.entries.push_back(PartitionedInventoryEntry{
            std::move(entry.relative_path),
            classification,
            entry.bytes,
            entry.modified_unix_ns,
            entry.area,
            std::move(entry.name),
            entry.identity});
    }
    std::sort(
        inventory.entries.begin(), inventory.entries.end(),
        [](const PartitionedInventoryEntry& left,
           const PartitionedInventoryEntry& right) {
            return left.relative_path < right.relative_path;
        });
    return inventory;
}

PartitionedOrphanCollectionResult collect_partitioned_orphans(
    const std::filesystem::path& store_path,
    const std::uint64_t expected_generation,
    const Uuid128 expected_catalog_id,
    const PartitionedOrphanCollectionOptions options,
    const PartitionedOrphanCollectionHooks& hooks) {
    const bool use_scanned_view =
        expected_generation == 0 && expected_catalog_id.empty();
    if ((expected_generation == 0) != expected_catalog_id.empty()) {
        throw InvalidArgumentError(
            "format-7 orphan collection expected CURRENT generation and ID "
            "must either both be present or both be omitted");
    }
    if (options.max_files == 0 || options.max_bytes == 0) {
        throw InvalidArgumentError(
            "format-7 orphan collection file and byte budgets must be positive");
    }

    PartitionedResourceInventory inventory =
        inspect_partitioned_resources(
            store_path, hooks.maximum_uncataloged_scan_entries);
    if (!use_scanned_view
        && (inventory.current_generation != expected_generation
            || inventory.current_catalog_id != expected_catalog_id)) {
        throw StorageError(
            "format-7 orphan collection CURRENT differs from the expected "
            "writer view; reopen the collection");
    }

    PartitionedOrphanCollectionResult result;
    result.current_generation = inventory.current_generation;
    result.current_catalog_id = inventory.current_catalog_id;
    result.options = options;
    result.entries.reserve(inventory.entries.size());
    for (PartitionedInventoryEntry& file : inventory.entries) {
        PartitionedOrphanCollectionEntry entry;
        entry.file = std::move(file);
        result.entries.push_back(std::move(entry));
    }

    std::vector<WorkItem> work;
    std::map<Uuid128, std::vector<std::size_t>> complete_segment_pairs;
    for (std::size_t index = 0; index != result.entries.size(); ++index) {
        const PartitionedInventoryEntry& entry = result.entries[index].file;
        if (ordinary_orphan_candidate(entry.classification)) {
            work.push_back(WorkItem{{index}});
        } else if (entry.classification
                   == PartitionedInventoryClass::
                       CompleteUnreferencedSegmentPair) {
            const auto id = segment_id_for_entry(entry);
            if (!id.has_value()) {
                throw StorageError(
                    "format-7 inventory lost a complete segment pair ID");
            }
            complete_segment_pairs[*id].push_back(index);
        }
    }
    for (auto& [id, members] : complete_segment_pairs) {
        static_cast<void>(id);
        if (members.size() != 2) {
            throw StorageError(
                "format-7 inventory complete segment pair has wrong cardinality");
        }
        work.push_back(WorkItem{std::move(members)});
    }
    for (WorkItem& item : work) {
        std::sort(item.entries.begin(), item.entries.end());
    }
    std::sort(
        work.begin(), work.end(),
        [&result](const WorkItem& left, const WorkItem& right) {
            return result.entries[left.entries.front()].file.relative_path
                 < result.entries[right.entries.front()].file.relative_path;
        });

    std::uint64_t selected_files = 0;
    std::uint64_t selected_bytes = 0;
    std::array<bool, kAreaCount> selected_areas{};
    for (const WorkItem& item : work) {
        bool invalid_time = false;
        bool too_young = false;
        std::uint64_t item_bytes = 0;
        for (const std::size_t index : item.entries) {
            const PartitionedInventoryEntry& file = result.entries[index].file;
            if (file.modified_unix_ns < 0
                || static_cast<std::uint64_t>(file.modified_unix_ns)
                       > options.snapshot_unix_ns) {
                invalid_time = true;
            } else if (
                options.snapshot_unix_ns
                        - static_cast<std::uint64_t>(file.modified_unix_ns)
                    < options.min_age_ns) {
                too_young = true;
            }
            if (file.bytes > std::numeric_limits<std::uint64_t>::max()
                                 - item_bytes) {
                throw StorageError(
                    "format-7 orphan work item byte count overflows uint64");
            }
            item_bytes += file.bytes;
        }
        if (invalid_time) {
            set_work_disposition(
                result, item,
                PartitionedOrphanDisposition::InvalidModificationTime);
            continue;
        }
        if (too_young) {
            set_work_disposition(
                result, item,
                PartitionedOrphanDisposition::MinimumAgeNotReached);
            continue;
        }
        for (const std::size_t index : item.entries) {
            result.entries[index].eligible = true;
        }
        if (item.entries.size() > options.max_files - selected_files) {
            set_work_disposition(
                result, item,
                PartitionedOrphanDisposition::MaximumFilesReached);
            continue;
        }
        if (item_bytes > options.max_bytes - selected_bytes) {
            set_work_disposition(
                result, item,
                PartitionedOrphanDisposition::MaximumBytesReached);
            continue;
        }
        selected_files += item.entries.size();
        selected_bytes += item_bytes;
        for (const std::size_t index : item.entries) {
            PartitionedOrphanCollectionEntry& entry = result.entries[index];
            entry.selected = true;
            selected_areas[area_index(entry.file.area)] = true;
        }
    }

    std::array<std::optional<FileDescriptor>, kAreaCount> directories;
    for (std::size_t index = 0; index != kAreaCount; ++index) {
        if (!selected_areas[index]) {
            continue;
        }
        const auto area = static_cast<PartitionedInventoryArea>(index);
        directories[index].emplace(
            open_area_directory(area_path(store_path, area)));
    }

    std::array<bool, kAreaCount> removed_areas{};
    std::size_t hook_ordinal = 0;
    for (const WorkItem& item : work) {
        if (!result.entries[item.entries.front()].selected) {
            continue;
        }
        if (hooks.before_revalidate) {
            for (const std::size_t index : item.entries) {
                hooks.before_revalidate(
                    hook_ordinal++, result.entries[index].file);
            }
        }

        bool valid = true;
        PartitionedOrphanDisposition failure =
            PartitionedOrphanDisposition::IdentityChanged;
        std::string failure_error;
        for (const std::size_t index : item.entries) {
            const PartitionedInventoryEntry& file = result.entries[index].file;
            const int directory_fd = directories[area_index(file.area)]->get();
            struct stat status {};
            if (::fstatat(
                    directory_fd, file.name.c_str(), &status,
                    AT_SYMLINK_NOFOLLOW)
                < 0) {
                const int error = errno;
                valid = false;
                failure = error == ENOENT
                              ? PartitionedOrphanDisposition::NoLongerPresent
                              : PartitionedOrphanDisposition::UnlinkFailed;
                if (error != ENOENT) {
                    failure_error = std::system_category().message(error);
                }
                break;
            }
            const std::filesystem::path diagnostic =
                area_path(store_path, file.area) / file.name;
            try {
                if (!same_identity(file.identity, status, diagnostic)) {
                    valid = false;
                    failure = PartitionedOrphanDisposition::IdentityChanged;
                    break;
                }
            } catch (const std::exception& error) {
                valid = false;
                failure = PartitionedOrphanDisposition::IdentityChanged;
                failure_error = error.what();
                break;
            }
        }
        if (!valid) {
            set_work_disposition(result, item, failure, failure_error);
            continue;
        }

        for (const std::size_t index : item.entries) {
            PartitionedOrphanCollectionEntry& entry = result.entries[index];
            const std::size_t area = area_index(entry.file.area);
            if (::unlinkat(
                    directories[area]->get(), entry.file.name.c_str(), 0)
                < 0) {
                const int error = errno;
                entry.disposition =
                    error == ENOENT
                        ? PartitionedOrphanDisposition::NoLongerPresent
                        : PartitionedOrphanDisposition::UnlinkFailed;
                if (error != ENOENT) {
                    entry.error = std::system_category().message(error);
                }
                continue;
            }
            entry.disposition = PartitionedOrphanDisposition::Removed;
            entry.removed = true;
            removed_areas[area] = true;
        }
    }

    for (std::size_t index = 0; index != kAreaCount; ++index) {
        if (!removed_areas[index]) {
            continue;
        }
        const auto area = static_cast<PartitionedInventoryArea>(index);
        try {
            sync_file(
                directories[index]->get(), area_path(store_path, area));
            result.synced_directories.push_back(area);
            for (PartitionedOrphanCollectionEntry& entry : result.entries) {
                if (entry.removed && entry.file.area == area) {
                    entry.durable = true;
                }
            }
        } catch (const std::exception& error) {
            result.durability_complete = false;
            for (PartitionedOrphanCollectionEntry& entry : result.entries) {
                if (entry.removed && entry.file.area == area) {
                    entry.error = error.what();
                }
            }
        }
    }
    return result;
}

PartitionedOrphanCollectionResult collect_partitioned_orphans(
    const std::filesystem::path& store_path,
    const PartitionedOrphanCollectionOptions options,
    const PartitionedOrphanCollectionHooks& hooks) {
    return collect_partitioned_orphans(
        store_path, 0, {}, options, hooks);
}

}  // namespace throwdb::detail
