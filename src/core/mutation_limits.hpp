#pragma once

#include <cstdint>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"

namespace throwdb::detail {

// Charge every retained WAL, including frozen predecessors and empty headers.
// Subtracting from the limits also avoids overflow when summing a large chain.
// Call before creating files or priming publication nodes: rejection must leave
// both the durable log and the in-memory mutation budget unchanged.
template <typename Sources>
void ensure_wal_growth(
    const Sources& sources,
    const NativeOpenOptions& limits,
    const std::uint64_t additional_bytes,
    const std::uint64_t additional_operations) {
    std::uint64_t remaining_bytes = limits.max_wal_bytes;
    std::uint64_t remaining_operations = limits.max_wal_operations;
    for (const auto& source : sources) {
        if (source->valid_end > remaining_bytes) {
            throw StorageError("retained WAL bytes exceed the configured limit");
        }
        remaining_bytes -= source->valid_end;
        if (source->operation_count > remaining_operations) {
            throw StorageError(
                "retained WAL operations exceed the configured limit");
        }
        remaining_operations -= source->operation_count;
    }
    if (additional_bytes > remaining_bytes) {
        throw StorageError(
            "mutation would exceed the configured total WAL byte limit before "
            "durable append; run bounded maintenance or raise the explicit limit");
    }
    if (additional_operations > remaining_operations) {
        throw StorageError(
            "mutation would exceed the configured total WAL operation limit "
            "before durable append; run bounded maintenance or raise the "
            "explicit limit");
    }
}

}  // namespace throwdb::detail
