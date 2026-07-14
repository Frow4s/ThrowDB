#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mutation_types.hpp"
#include "native_format.hpp"

namespace throwdb::detail {

// Private, read-only filesystem classification used as the foundation for a
// future mutation-resource collector. SupersededCatalog means only that the
// file is not the catalog selected by validated CURRENT; predecessor catalogs
// are not runtime-reachable snapshots. Classification never implies that an
// entry may be removed: a collector must apply its own locking, age and
// durability policy before acting on a later inventory.
enum class MutationInventoryClass {
    Reachable,
    SupersededCatalog,
    TempPublication,
    UnreferencedWal,
    CompleteUnreferencedDeltaPair,
    PartialUnreferencedDeltaPair,
    UnknownDoNotTouch,
};

struct MutationInventoryEntry {
    // Portable, generic relative path below the collection root. Entries are
    // sorted lexicographically by this field.
    std::string relative_path;
    MutationInventoryClass classification{
        MutationInventoryClass::UnknownDoNotTouch};
    // Exact regular-file size. Symlinks and other non-regular entries are
    // deliberately not followed and report zero bytes.
    std::uint64_t bytes{0};
    // lstat timestamp normalized to Unix nanoseconds. Keeping the signed
    // value lets callers conservatively reject pre-epoch or otherwise unusual
    // metadata instead of making it appear newer or older through clamping.
    std::int64_t modified_unix_ns{0};

    bool operator==(const MutationInventoryEntry&) const = default;
};

struct MutationResourceInventory {
    std::uint64_t current_generation{0};
    Uuid128 current_catalog_id;
    std::vector<MutationInventoryEntry> entries;

    bool operator==(const MutationResourceInventory&) const = default;
};

// Reads and validates CURRENT and exactly the catalog named by it before
// deriving reachability. It scans only CURRENT-like root entries and the
// catalogs/, wal/ and deltas/ namespaces. No files are created, modified or
// removed. Reserved segment-ID gaps are metadata only and never synthesize
// inventory entries.
//
// A future collector must call this while holding the collection writer lease
// and the mutation publication barrier exclusively. The helper intentionally
// does not acquire locks itself so it can be composed into that larger scope.
// The result is diagnostic input, not an authorization token: a future deleter
// must revalidate type and identity without following symlinks (for example
// with openat/fstat) immediately before unlinking each path, while it still
// holds both leases. This helper performs no deletion.
[[nodiscard]] MutationResourceInventory inspect_mutation_resources(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);

}  // namespace throwdb::detail
