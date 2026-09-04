#pragma once

#include <cstdint>
#include <type_traits>

namespace throwdb::detail {

// Private deterministic synchronization points for native lifecycle tests.
// Compiled out when native tests are disabled; this header is not installed
// as part of the public ThrowDB API.
enum class NativeCollectionTestHookPoint : std::uint8_t {
    ReaderOpenCommitLockAcquired = 1,
    MutationCatalogDurableBeforeCurrent = 2,
    MutationCurrentDurable = 3,
    MutationReservationDurable = 4,
    LifecycleOutputDurable = 5,
    BeforeOrphanRevalidation = 8,
    AfterOrphanUnlink = 9,
    BeforeOrphanDirectorySync = 10,
    AfterOrphanDirectorySync = 11,
    CleanerSegmentOutputDurable = 12,
    CleanerRunOutputDurable = 13,
    BeforeLifecycleFenceCheck = 14,
    LifecycleBuildUnlocked = 15,
};

enum class NativeCollectionTestOperation : std::uint8_t {
    None = 0,
    MaterializeFrozenWals = 1,
    CollectOrphans = 3,
    FoldPartitions = 4,
    CleanSegment = 5,
};

enum class NativeCollectionTestResourceKind : std::uint8_t {
    None = 0,
    OrphanFile = 5,
    OrphanDirectory = 6,
    PartitionSnapshot = 7,
    PartitionedSegmentPair = 8,
    PartitionIndexRun = 9,
};

// Keep the callback payload a plain value so fork-based crash tests can
// inspect it and call _exit without allocating memory or acquiring a lock.
struct NativeCollectionTestHookContext {
    NativeCollectionTestHookPoint point;
    NativeCollectionTestOperation operation{
        NativeCollectionTestOperation::None};
    NativeCollectionTestResourceKind resource_kind{
        NativeCollectionTestResourceKind::None};
    std::uint64_t generation{0};
    std::uint64_t ordinal{0};
};

static_assert(std::is_standard_layout_v<NativeCollectionTestHookContext>);
static_assert(std::is_trivially_copyable_v<NativeCollectionTestHookContext>);

using NativeCollectionTestHook = void (*)(
    NativeCollectionTestHookContext context) noexcept;

#ifdef THROWDB_NATIVE_TEST_HOOKS
void set_native_collection_test_hook(
    NativeCollectionTestHook hook) noexcept;
void invoke_native_collection_test_hook(
    NativeCollectionTestHookContext context) noexcept;

#else
inline void invoke_native_collection_test_hook(
    NativeCollectionTestHookContext) noexcept {}
#endif

}  // namespace throwdb::detail
