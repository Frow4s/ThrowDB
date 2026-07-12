#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace throwdb::detail {

// One exact positional read submitted through IoUringEngine. user_index is
// opaque to the engine except that it is included in errors, allowing callers
// to associate a failed physical read with their own range plan.
struct IoUringReadOperation {
    int fd{-1};
    void* buffer{nullptr};
    std::size_t size{0};
    std::uint64_t offset{0};
    std::uint64_t user_index{0};
};

// A process-local, buffered-I/O io_uring service. The first experimental
// implementation serializes complete multi-range calls with one mutex, so one
// caller owns the SQ/CQ until its operations have completed or failed. This is
// deliberate: it tests whether queue depth can replace pread worker threads
// before adding a more complex multi-producer scheduler.
//
// Construction performs the runtime io_uring capability check. On non-Linux
// platforms construction throws StorageError, but the stub remains buildable.
class IoUringEngine {
public:
    explicit IoUringEngine(std::uint32_t queue_depth);
    ~IoUringEngine();

    IoUringEngine(const IoUringEngine&) = delete;
    IoUringEngine& operator=(const IoUringEngine&) = delete;
    IoUringEngine(IoUringEngine&&) = delete;
    IoUringEngine& operator=(IoUringEngine&&) = delete;

    void read_exact(std::span<const IoUringReadOperation> operations);
    void read_exact(const std::vector<IoUringReadOperation>& operations) {
        read_exact(std::span<const IoUringReadOperation>(operations));
    }

    [[nodiscard]] std::uint32_t queue_depth() const noexcept;
    // False means a fatal SQ/CQ or io_uring_enter failure poisoned this ring.
    // Ordinary per-read errors and unexpected EOF do not poison it.
    [[nodiscard]] bool usable() const noexcept;
    [[nodiscard]] static constexpr bool platform_supported() noexcept {
#if defined(__linux__)
        return true;
#else
        return false;
#endif
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
