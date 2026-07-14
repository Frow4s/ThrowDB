#pragma once

#include <cstdint>
#include <type_traits>

namespace throwdb::detail {

// Private deterministic synchronization points for native lifecycle tests.
// Production code pays only one null atomic load at an instrumented point;
// this header is not installed as part of the public ThrowDB API.
enum class NativeCollectionTestHookPoint : std::uint8_t {
    ReaderOpenCommitLockAcquired = 1,
    MutationCatalogDurableBeforeCurrent = 2,
    MutationCurrentDurable = 3,
    MutationReservationDurable = 4,
    MutationDeltaOutputDurable = 5,
    BeforeSourceUnlink = 6,
    AfterSourceUnlink = 7,
};

enum class NativeCollectionTestOperation : std::uint8_t {
    None = 0,
    MaterializeFrozenWals = 1,
    CompactDeltaRuns = 2,
};

enum class NativeCollectionTestResourceKind : std::uint8_t {
    None = 0,
    MutationWal = 1,
    MutationDeltaPair = 2,
    MutationDeltaData = 3,
    MutationDeltaIndex = 4,
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

void set_native_collection_test_hook(
    NativeCollectionTestHook hook) noexcept;
void invoke_native_collection_test_hook(
    NativeCollectionTestHookContext context) noexcept;

}  // namespace throwdb::detail
