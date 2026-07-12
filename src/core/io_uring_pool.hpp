#pragma once

#include "io_uring_engine.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace throwdb::detail {

[[nodiscard]] std::vector<std::uint32_t> split_io_uring_queue_depth(
    std::uint32_t worker_count,
    std::uint32_t total_queue_depth);

[[nodiscard]] std::vector<std::size_t> plan_io_uring_lanes(
    std::span<const IoUringReadOperation> operations,
    std::span<const std::uint32_t> lane_queue_depths);

// A synchronous facade over multiple dedicated io_uring owner threads. The
// total queue depth is divided between workers, and every multi-range request
// is partitioned across the workers so a single large call can use every ring.
class IoUringPool {
public:
    IoUringPool(std::uint32_t worker_count,
                std::uint32_t total_queue_depth);
    ~IoUringPool();

    IoUringPool(const IoUringPool&) = delete;
    IoUringPool& operator=(const IoUringPool&) = delete;
    IoUringPool(IoUringPool&&) = delete;
    IoUringPool& operator=(IoUringPool&&) = delete;

    void read_exact(std::span<const IoUringReadOperation> operations);
    void read_exact(const std::vector<IoUringReadOperation>& operations) {
        read_exact(std::span<const IoUringReadOperation>(operations));
    }

    [[nodiscard]] std::uint32_t worker_count() const noexcept;
    [[nodiscard]] std::uint32_t total_queue_depth() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
