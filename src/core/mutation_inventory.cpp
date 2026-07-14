#include "mutation_inventory.hpp"

#include "mutation_catalog.hpp"
#include "mutation_delta.hpp"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>
#include <utility>

#include <sys/stat.h>

namespace throwdb::detail {
namespace {

enum class InventoryArea {
    Root,
    Catalogs,
    Wal,
    Deltas,
};

struct ScannedEntry {
    InventoryArea area{InventoryArea::Root};
    std::string name;
    std::string relative_path;
    bool regular{false};
    std::uint64_t bytes{0};
    std::int64_t modified_unix_ns{0};
};

[[nodiscard]] std::int64_t unix_mtime_ns(
    const struct stat& status,
    const std::filesystem::path& path) {
#if defined(__APPLE__)
    const auto seconds_raw = status.st_mtimespec.tv_sec;
    const auto nanoseconds_raw = status.st_mtimespec.tv_nsec;
#else
    const auto seconds_raw = status.st_mtim.tv_sec;
    const auto nanoseconds_raw = status.st_mtim.tv_nsec;
#endif
    if (!std::in_range<std::int64_t>(seconds_raw)
        || !std::in_range<std::int64_t>(nanoseconds_raw)) {
        throw StorageError(
            "mutation inventory timestamp does not fit int64: '"
            + path.string() + "'");
    }
    const std::int64_t seconds =
        static_cast<std::int64_t>(seconds_raw);
    const std::int64_t nanoseconds =
        static_cast<std::int64_t>(nanoseconds_raw);
    constexpr std::int64_t billion = 1'000'000'000;
    if (nanoseconds < 0 || nanoseconds >= billion) {
        throw StorageError(
            "mutation inventory timestamp has invalid nanoseconds: '"
            + path.string() + "'");
    }
    constexpr std::int64_t maximum =
        std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t minimum =
        std::numeric_limits<std::int64_t>::min();
    if (seconds > maximum / billion || seconds < minimum / billion) {
        throw StorageError(
            "mutation inventory timestamp overflows Unix nanoseconds: '"
            + path.string() + "'");
    }
    const std::int64_t scaled = seconds * billion;
    if (seconds >= 0 && nanoseconds > maximum - scaled) {
        throw StorageError(
            "mutation inventory timestamp overflows Unix nanoseconds: '"
            + path.string() + "'");
    }
    return scaled + nanoseconds;
}

[[nodiscard]] std::optional<std::uint64_t> parse_hex_word(
    const std::string_view token) noexcept {
    if (token.size() != 16) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (const char character : token) {
        std::uint8_t digit = 0;
        if (character >= '0' && character <= '9') {
            digit = static_cast<std::uint8_t>(character - '0');
        } else if (character >= 'a' && character <= 'f') {
            digit = static_cast<std::uint8_t>(character - 'a' + 10);
        } else {
            return std::nullopt;
        }
        value = (value << 4U) | digit;
    }
    return value;
}

[[nodiscard]] std::optional<Uuid128> parse_uuid_filename(
    const std::string_view name,
    const std::string_view prefix,
    const std::string_view suffix) noexcept {
    constexpr std::size_t uuid_token_size = 32;
    if (name.size() != prefix.size() + uuid_token_size + suffix.size()
        || !name.starts_with(prefix) || !name.ends_with(suffix)) {
        return std::nullopt;
    }
    const std::string_view token = name.substr(prefix.size(), uuid_token_size);
    const auto high = parse_hex_word(token.substr(0, 16));
    const auto low = parse_hex_word(token.substr(16));
    if (!high.has_value() || !low.has_value()) {
        return std::nullopt;
    }
    const Uuid128 result{*low, *high};
    return result.empty() ? std::nullopt : std::optional<Uuid128>{result};
}

[[nodiscard]] std::filesystem::file_status symlink_status_checked(
    const std::filesystem::path& path,
    const std::string_view description) {
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error) {
        throw StorageError(
            "cannot inspect " + std::string(description) + " '"
            + path.string() + "': " + error.message());
    }
    return status;
}

void require_authoritative_regular_file(
    const std::filesystem::path& path,
    const std::string_view description) {
    const std::filesystem::file_status status =
        symlink_status_checked(path, description);
    if (!std::filesystem::is_regular_file(status)) {
        throw CorruptionError(
            std::string(description) + " is not a regular file: '"
            + path.string() + "'");
    }
}

void require_real_directory(
    const std::filesystem::path& path,
    const std::string_view description) {
    const std::filesystem::file_status status =
        symlink_status_checked(path, description);
    if (!std::filesystem::is_directory(status)) {
        throw CorruptionError(
            std::string(description) + " is not a directory: '"
            + path.string() + "'");
    }
}

[[nodiscard]] ScannedEntry inspect_entry(
    const std::filesystem::directory_entry& entry,
    const InventoryArea area,
    const std::filesystem::path& relative_directory) {
    ScannedEntry result;
    result.area = area;
    result.name = entry.path().filename().string();
    result.relative_path =
        (relative_directory / result.name).generic_string();

    struct stat status {};
    if (::lstat(entry.path().c_str(), &status) < 0) {
        const int error = errno;
        throw StorageError(
            "cannot inspect mutation inventory entry '"
            + entry.path().string() + "': "
            + std::system_category().message(error));
    }
    result.modified_unix_ns = unix_mtime_ns(status, entry.path());
    result.regular = S_ISREG(status.st_mode);
    if (!result.regular) {
        return result;
    }
    if (status.st_size < 0
        || !std::in_range<std::uint64_t>(status.st_size)) {
        throw StorageError(
            "mutation inventory file size exceeds uint64: '"
            + entry.path().string() + "'");
    }
    result.bytes = static_cast<std::uint64_t>(status.st_size);
    return result;
}

void scan_directory(
    const std::filesystem::path& directory,
    const std::filesystem::path& relative_directory,
    const InventoryArea area,
    const bool current_names_only,
    std::vector<ScannedEntry>& output) {
    require_real_directory(directory, "mutation inventory directory");
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error) {
        throw StorageError(
            "cannot scan mutation inventory directory '"
            + directory.string() + "': " + error.message());
    }
    const std::filesystem::directory_iterator end;
    while (iterator != end) {
        const std::string name = iterator->path().filename().string();
        if (!current_names_only || name.starts_with("CURRENT")) {
            output.push_back(inspect_entry(
                *iterator, area, relative_directory));
        }
        iterator.increment(error);
        if (error) {
            throw StorageError(
                "cannot continue mutation inventory scan in '"
                + directory.string() + "': " + error.message());
        }
    }
}

[[nodiscard]] MutationInventoryClass classify_entry(
    const ScannedEntry& entry,
    const MutationCatalog& catalog,
    const std::set<Uuid128>& reachable_wals,
    const std::set<Uuid128>& reachable_deltas,
    const std::map<Uuid128, std::uint8_t>& unreferenced_delta_parts) {
    if (!entry.regular) {
        return MutationInventoryClass::UnknownDoNotTouch;
    }

    if (entry.area == InventoryArea::Root) {
        if (entry.name == "CURRENT") {
            return MutationInventoryClass::Reachable;
        }
        if (parse_uuid_filename(entry.name, "CURRENT.", ".tmp")
                .has_value()) {
            return MutationInventoryClass::TempPublication;
        }
        return MutationInventoryClass::UnknownDoNotTouch;
    }

    if (entry.area == InventoryArea::Catalogs) {
        if (const auto id = parse_uuid_filename(
                entry.name, "catalog-", ".bin"); id.has_value()) {
            return *id == catalog.catalog_id
                       ? MutationInventoryClass::Reachable
                       : MutationInventoryClass::SupersededCatalog;
        }
        if (parse_uuid_filename(entry.name, "catalog-", ".bin.tmp")
                .has_value()) {
            return MutationInventoryClass::TempPublication;
        }
        return MutationInventoryClass::UnknownDoNotTouch;
    }

    if (entry.area == InventoryArea::Wal) {
        if (const auto id = parse_uuid_filename(
                entry.name, "wal-", ".wal"); id.has_value()) {
            return reachable_wals.contains(*id)
                       ? MutationInventoryClass::Reachable
                       : MutationInventoryClass::UnreferencedWal;
        }
        return MutationInventoryClass::UnknownDoNotTouch;
    }

    std::optional<Uuid128> final_delta =
        parse_uuid_filename(entry.name, "delta-", ".dat");
    if (!final_delta.has_value()) {
        final_delta = parse_uuid_filename(entry.name, "delta-", ".idx");
    }
    if (final_delta.has_value()) {
        if (reachable_deltas.contains(*final_delta)) {
            return MutationInventoryClass::Reachable;
        }
        const auto parts = unreferenced_delta_parts.find(*final_delta);
        if (parts == unreferenced_delta_parts.end()) {
            throw StorageError(
                "mutation inventory lost a parsed delta candidate");
        }
        return parts->second == 0x3U
                   ? MutationInventoryClass::CompleteUnreferencedDeltaPair
                   : MutationInventoryClass::PartialUnreferencedDeltaPair;
    }
    if (parse_uuid_filename(entry.name, "delta-", ".dat.tmp").has_value()
        || parse_uuid_filename(entry.name, "delta-", ".idx.tmp")
               .has_value()) {
        return MutationInventoryClass::TempPublication;
    }
    return MutationInventoryClass::UnknownDoNotTouch;
}

}  // namespace

MutationResourceInventory inspect_mutation_resources(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    require_authoritative_regular_file(
        mutation_current_path(store_path), "mutation CURRENT");
    const MutationCurrent current =
        read_mutation_current(store_path, manifest);
    const std::filesystem::path current_catalog_path =
        mutation_catalog_path(store_path, current.catalog_id);
    require_authoritative_regular_file(
        current_catalog_path, "current mutation catalog");
    const MutationCatalog catalog =
        read_mutation_catalog(store_path, manifest, current);

    std::set<Uuid128> reachable_wals;
    for (const MutationWalEntry& entry : catalog.wal_entries) {
        reachable_wals.insert(entry.wal_id);
    }
    std::set<Uuid128> reachable_deltas;
    for (const MutationDeltaDescriptor& descriptor : catalog.delta_entries) {
        reachable_deltas.insert(descriptor.delta_id);
    }

    std::vector<ScannedEntry> scanned;
    scan_directory(
        store_path, {}, InventoryArea::Root, true, scanned);
    scan_directory(
        mutation_catalogs_directory(store_path), "catalogs",
        InventoryArea::Catalogs, false, scanned);
    scan_directory(
        store_path / "wal", "wal", InventoryArea::Wal, false, scanned);
    // Format 5 does not assign semantics to the deltas namespace. Ignoring it
    // is both backwards-compatible and fail-closed: arbitrary user content
    // cannot become a deletion candidate merely because of a filename.
    if (mutation_delta_managed(manifest)) {
        scan_directory(
            mutation_deltas_directory(store_path), "deltas",
            InventoryArea::Deltas, false, scanned);
    }

    std::map<Uuid128, std::uint8_t> unreferenced_delta_parts;
    for (const ScannedEntry& entry : scanned) {
        if (!entry.regular || entry.area != InventoryArea::Deltas) {
            continue;
        }
        if (const auto id = parse_uuid_filename(
                entry.name, "delta-", ".dat");
            id.has_value() && !reachable_deltas.contains(*id)) {
            unreferenced_delta_parts[*id] |= 0x1U;
        } else if (const auto id = parse_uuid_filename(
                       entry.name, "delta-", ".idx");
                   id.has_value() && !reachable_deltas.contains(*id)) {
            unreferenced_delta_parts[*id] |= 0x2U;
        }
    }

    MutationResourceInventory inventory;
    inventory.current_generation = catalog.generation;
    inventory.current_catalog_id = catalog.catalog_id;
    inventory.entries.reserve(scanned.size());
    for (ScannedEntry& entry : scanned) {
        inventory.entries.push_back(MutationInventoryEntry{
            std::move(entry.relative_path),
            classify_entry(
                entry, catalog, reachable_wals, reachable_deltas,
                unreferenced_delta_parts),
            entry.bytes,
            entry.modified_unix_ns});
    }
    std::sort(
        inventory.entries.begin(), inventory.entries.end(),
        [](const MutationInventoryEntry& left,
           const MutationInventoryEntry& right) {
            return left.relative_path < right.relative_path;
        });
    return inventory;
}

}  // namespace throwdb::detail
