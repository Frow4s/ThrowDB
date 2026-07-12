#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "throwdb/storage.hpp"

namespace throwdb::detail {

// One already-resolved fixed-width row. The caller owns the file descriptor
// and must keep it open until execute() returns. output_index identifies the
// row-sized slot in the caller-provided output buffer.
struct PhysicalReadRequest {
    int fd{-1};
    std::uint64_t absolute_offset{0};
    std::uint64_t output_index{0};
};

// Executes physical reads after logical ID lookup and RowRef resolution have
// completed. One instance owns a bounded backend shared by all concurrent
// execute() calls; execution is synchronous from each caller's perspective.
class PhysicalReader {
public:
    explicit PhysicalReader(ReaderOptions options = {});
    ~PhysicalReader();

    PhysicalReader(const PhysicalReader&) = delete;
    PhysicalReader& operator=(const PhysicalReader&) = delete;
    PhysicalReader(PhysicalReader&&) = delete;
    PhysicalReader& operator=(PhysicalReader&&) = delete;

    // Reads one record for every request. Geometry and every request are
    // validated before output is touched. output_index values must form an
    // exact permutation of [0, requests.size()).
    void execute(
        std::uint64_t record_size,
        std::span<const PhysicalReadRequest> requests,
        std::span<std::byte> output) const;

    // Drop the child-process wrapper around inherited executor thread state.
    // This intentionally leaks only that process-local state until exit.
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
