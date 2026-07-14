#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mutation_types.hpp"
#include "native_format.hpp"

namespace throwdb::detail {

// Private, read-only filesystem classification used by the mutation-resource
// collector. SupersededCatalog means only that the
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

enum class MutationInventoryArea {
    Root,
    Catalogs,
    Wal,
    Deltas,
};

struct MutationFileIdentity {
    std::uint64_t device{0};
    std::uint64_t inode{0};
    std::uint64_t mode{0};
    std::uint64_t bytes{0};
    std::int64_t modified_unix_ns{0};
    std::int64_t changed_unix_ns{0};

    bool operator==(const MutationFileIdentity&) const = default;
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
    // Private deletion coordinates and lstat identity. A collector uses the
    // area/name pair with *at syscalls; relative_path is diagnostic only.
    MutationInventoryArea area{MutationInventoryArea::Root};
    std::string name;
    MutationFileIdentity identity;

    bool operator==(const MutationInventoryEntry&) const = default;
};

enum class MutationOrphanDisposition {
    NotCandidate,
    InvalidModificationTime,
    MinimumAgeNotReached,
    MaximumFilesReached,
    MaximumBytesReached,
    Removed,
    NoLongerPresent,
    IdentityChanged,
    UnlinkFailed,
};

struct MutationOrphanCollectionOptions {
    std::uint64_t snapshot_unix_ns{0};
    std::uint64_t min_age_ns{0};
    std::uint64_t max_files{1024};
    std::uint64_t max_bytes{8ULL << 30};
    bool require_expected_current{false};
    std::uint64_t expected_generation{0};
    Uuid128 expected_catalog_id;
};

struct MutationOrphanCollectionEntry {
    MutationInventoryEntry file;
    MutationOrphanDisposition disposition{
        MutationOrphanDisposition::NotCandidate};
    bool eligible{false};
    bool selected{false};
    bool removed{false};
    bool durable{false};
    std::string error;
};

struct MutationOrphanCollectionResult {
    std::uint64_t current_generation{0};
    Uuid128 current_catalog_id;
    MutationOrphanCollectionOptions options;
    std::vector<MutationInventoryArea> synced_directories;
    std::vector<MutationOrphanCollectionEntry> entries;
    bool durability_complete{true};
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
// A collector must call this while holding the collection writer lease
// and the mutation publication barrier exclusively. The helper intentionally
// does not acquire locks itself so it can be composed into that larger scope.
// The result is diagnostic input, not an authorization token. The destructive
// helper below performs its own fresh call and revalidates type and identity
// without following symlinks immediately before each unlinkat.
[[nodiscard]] MutationResourceInventory inspect_mutation_resources(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);

// Destructively collect only fresh-scan candidates. The caller must hold the
// writer lease and commit barrier exclusively. Each selected pathname is
// revalidated with AT_SYMLINK_NOFOLLOW immediately before unlinkat, and every
// affected namespace directory is fsynced before success is reported.
[[nodiscard]] MutationOrphanCollectionResult collect_mutation_orphans(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    MutationOrphanCollectionOptions options);

}  // namespace throwdb::detail
