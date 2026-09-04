#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace throwdb {

enum class DType : std::uint32_t {
    Float16 = 1,
    Float32 = 2,
};

enum class IoBackend : std::uint32_t {
    ThreadPool = 0,
    IoUring = 1,
};

struct ReaderOptions {
    // Adjacent requested records are read as one range up to this size.
    std::uint64_t max_read_range_bytes = 1ULL << 20;
    // A range may include this many unrequested bytes between records.
    std::uint64_t max_gap_bytes = 0;
    // Zero keeps the deterministic sequential pread baseline. A positive
    // value creates one executor shared by every get() call on this collection.
    std::uint32_t io_workers = 0;
    // Maximum number of tasks waiting for a worker. With workers enabled,
    // zero selects 2 * io_workers. submit() applies backpressure at the bound.
    // The fair coordinator additionally admits at most io_workers plus this
    // effective capacity execute() jobs; later callers wait in ticket order.
    std::uint32_t io_queue_capacity = 0;
    // Maximum number of planned read ranges grouped into one executor task.
    // The effective value may be smaller so a batch can use every I/O worker.
    std::uint32_t io_ranges_per_task = 8;
    // ThreadPool uses synchronous pread, optionally through io_workers.
    // IoUring is an experimental Linux-only buffered-read backend.
    IoBackend io_backend = IoBackend::ThreadPool;
    // Maximum total number of io_uring reads in flight across all workers.
    // With multiple workers this depth is divided as evenly as possible.
    std::uint32_t io_uring_queue_depth = 32;
    // Number of independent io_uring lanes. One preserves the caller-driven
    // baseline; two or more create dedicated ring-owner worker threads.
    std::uint32_t io_uring_workers = 1;
    // Collection opens always verify persisted index checksums. Set this to
    // additionally verify complete index structure and segment payloads.
    bool verify_native_index = false;
};

}  // namespace throwdb
