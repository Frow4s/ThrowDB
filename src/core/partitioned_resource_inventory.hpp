#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "mutation_types.hpp"

namespace throwdb::detail {

// The validated catalog already bounds authoritative resource metadata. This
// additional headroom bounds filesystem entries not accounted for by that
// catalog (retired resources, interrupted publications, and unknown files)
// while the first local MVP still builds a complete diagnostic inventory.
// A later streaming collector can replace this fail-closed guard with a
// resumable cursor.
inline constexpr std::uint64_t
    kPartitionedInventoryMaximumUncatalogedEntries = 1ULL << 16;

// Private format-7 lifecycle inventory. Classification is deliberately
// separate from deletion authorization: only the destructive helper below may
// turn a fresh, lock-protected scan into unlinkat operations.
enum class PartitionedInventoryClass {
    Reachable,
    SupersededCatalog,
    TempPublication,
    UnreferencedWal,
    CompleteUnreferencedSegmentPair,
    PartialUnreferencedSegmentPair,
    UnreferencedSnapshot,
    UnreferencedRun,
    UnknownDoNotTouch,
};

enum class PartitionedInventoryArea {
    Root,
    Catalogs,
    Segments,
    Partitions,
    Runs,
    Wal,
};

struct PartitionedFileIdentity {
    std::uint64_t device{0};
    std::uint64_t inode{0};
    std::uint64_t mode{0};
    std::uint64_t bytes{0};
    std::int64_t modified_unix_ns{0};
    std::int64_t changed_unix_ns{0};

    bool operator==(const PartitionedFileIdentity&) const = default;
};

struct PartitionedInventoryEntry {
    std::string relative_path;
    PartitionedInventoryClass classification{
        PartitionedInventoryClass::UnknownDoNotTouch};
    std::uint64_t bytes{0};
    std::int64_t modified_unix_ns{0};
    PartitionedInventoryArea area{PartitionedInventoryArea::Root};
    std::string name;
    PartitionedFileIdentity identity;

    bool operator==(const PartitionedInventoryEntry&) const = default;
};

struct PartitionedResourceInventory {
    std::uint64_t current_generation{0};
    Uuid128 current_catalog_id;
    std::vector<PartitionedInventoryEntry> entries;

    bool operator==(const PartitionedResourceInventory&) const = default;
};

enum class PartitionedOrphanDisposition {
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

struct PartitionedOrphanCollectionOptions {
    std::uint64_t snapshot_unix_ns{0};
    std::uint64_t min_age_ns{0};
    std::uint64_t max_files{1024};
    std::uint64_t max_bytes{8ULL << 30};
};

struct PartitionedOrphanCollectionEntry {
    PartitionedInventoryEntry file;
    PartitionedOrphanDisposition disposition{
        PartitionedOrphanDisposition::NotCandidate};
    bool eligible{false};
    bool selected{false};
    bool removed{false};
    bool durable{false};
    std::string error;
};

struct PartitionedOrphanCollectionResult {
    std::uint64_t current_generation{0};
    Uuid128 current_catalog_id;
    PartitionedOrphanCollectionOptions options;
    std::vector<PartitionedInventoryArea> synced_directories;
    std::vector<PartitionedOrphanCollectionEntry> entries;
    bool durability_complete{true};
};

// Test/fault-injection seam invoked after the fresh scan and selection but
// before fstatat identity validation. Production callers leave it empty.
struct PartitionedOrphanCollectionHooks {
    std::function<void(
        std::size_t ordinal,
        const PartitionedInventoryEntry& entry)> before_revalidate;
    // Test-only override for exercising the scan guard without creating the
    // production number of directory entries.
    std::uint64_t maximum_uncataloged_scan_entries{
        kPartitionedInventoryMaximumUncatalogedEntries};
};

// Phase-one result for an open writer. Inventory, classification, age checks,
// and budget selection are intentionally advisory: CURRENT may advance while
// this plan is built. apply_partitioned_orphan_plan() must re-read the current
// authoritative view and reject every selected path which became reachable.
struct PartitionedOrphanCollectionPlan {
    PartitionedOrphanCollectionResult result;
};

// Validates manifest.bin, CURRENT and exactly the catalog selected by CURRENT
// (including its size and CRC) before scanning the product-owned mutable
// namespaces. Directory entries are inspected with lstat and never followed.
// The scan fails before returning a partial inventory if filesystem entries
// exceed the catalog-derived path count plus the uncataloged-entry headroom.
[[nodiscard]] PartitionedResourceInventory inspect_partitioned_resources(
    const std::filesystem::path& store_path,
    std::uint64_t maximum_uncataloged_entries =
        kPartitionedInventoryMaximumUncatalogedEntries);

// Performs the unbounded-by-selected-count directory scan and bounded
// candidate selection without deleting files. An open writer may call this
// while holding only its lifecycle lease, leaving the mutation lane available.
[[nodiscard]] PartitionedOrphanCollectionPlan plan_partitioned_orphans(
    const std::filesystem::path& store_path,
    PartitionedOrphanCollectionOptions options,
    const PartitionedOrphanCollectionHooks& hooks = {});

// Re-reads CURRENT/catalog and applies a phase-one plan. The caller must hold
// writer.lock, commit.lock and view.lock exclusively and must prevent its
// in-memory catalog from changing for the duration. Only selected identities
// still unreachable from this fresh authoritative view can be unlinked.
[[nodiscard]] PartitionedOrphanCollectionResult
apply_partitioned_orphan_plan(
    const std::filesystem::path& store_path,
    std::uint64_t expected_generation,
    Uuid128 expected_catalog_id,
    PartitionedOrphanCollectionPlan plan,
    const PartitionedOrphanCollectionHooks& hooks = {});

// The caller must already hold writer.lock, commit.lock and view.lock
// exclusively. The executor always performs one fresh inventory, checks any
// supplied expected view, revalidates every selected regular file with
// fstatat(AT_SYMLINK_NOFOLLOW), unlinks by directory FD, and fsyncs every
// affected namespace. Complete unreferenced segment pairs are selected and
// prevalidated as one unit. A partial unreferenced pair is a singleton
// candidate: CURRENT-referenced halves classify as Reachable, while a fresh
// unreferenced half can safely converge an interrupted earlier pair deletion.
[[nodiscard]] PartitionedOrphanCollectionResult collect_partitioned_orphans(
    const std::filesystem::path& store_path,
    std::uint64_t expected_generation,
    Uuid128 expected_catalog_id,
    PartitionedOrphanCollectionOptions options,
    const PartitionedOrphanCollectionHooks& hooks = {});

// Static reclaimers that do not already own an open writer view use the same
// single fresh scan as both the authoritative view selection and deletion
// plan. The caller still must hold all three locks named above.
[[nodiscard]] PartitionedOrphanCollectionResult collect_partitioned_orphans(
    const std::filesystem::path& store_path,
    PartitionedOrphanCollectionOptions options,
    const PartitionedOrphanCollectionHooks& hooks = {});

}  // namespace throwdb::detail
