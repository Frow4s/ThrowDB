#pragma once

#include <algorithm>
#include <cstdint>

namespace throwdb::server::detail {

// Refill a one-second bucket exactly, retaining fractional-byte credit.
// The caller maintains tokens <= rate and remainder < nanoseconds_per_second.
constexpr void refill_read_workload_write_budget(
    const std::uint64_t rate,
    const std::uint64_t elapsed_ns,
    std::uint64_t& tokens,
    std::uint64_t& remainder) noexcept {
    constexpr std::uint64_t nanoseconds_per_second = 1'000'000'000;
    const auto partial_ns = elapsed_ns % nanoseconds_per_second;
    // Both factors are below 1e9, so this sum is below 1e18 and fits uint64.
    const auto fractional =
        (rate % nanoseconds_per_second) * partial_ns + remainder;
    remainder = fractional % nanoseconds_per_second;
    if (elapsed_ns >= nanoseconds_per_second) {
        // A full second earns at least the bucket capacity. Avoid computing
        // the uncapped credit, which can exceed uint64 after a long idle.
        tokens = rate;
        return;
    }
    // elapsed_ns < 1e9 makes the complete credit <= rate, without a wide
    // intermediate product. Add only the credit that fits the bucket.
    const auto added = (rate / nanoseconds_per_second) * partial_ns
                       + fractional / nanoseconds_per_second;
    tokens += std::min(added, rate - tokens);
}

}  // namespace throwdb::server::detail
