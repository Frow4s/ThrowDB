#include "native_collection_test_hooks.hpp"

#include <atomic>

namespace throwdb::detail {
namespace {

std::atomic<NativeCollectionTestHook> active_hook{nullptr};
static_assert(
    std::atomic<NativeCollectionTestHook>::is_always_lock_free,
    "native collection crash-test callbacks must be fork-safe");

}  // namespace

void set_native_collection_test_hook(
    const NativeCollectionTestHook hook) noexcept {
    active_hook.store(hook, std::memory_order_release);
}

void invoke_native_collection_test_hook(
    const NativeCollectionTestHookContext context) noexcept {
    const NativeCollectionTestHook hook =
        active_hook.load(std::memory_order_acquire);
    if (hook != nullptr) {
        hook(context);
    }
}

}  // namespace throwdb::detail
