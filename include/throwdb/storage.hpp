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
};

enum class IoBackend : std::uint32_t {
    ThreadPool = 0,
    IoUring = 1,
};

struct WriterOptions {
    std::uint32_t dimension = 1024;
    DType dtype = DType::Float16;
    // The payload capacity of one shard. The 4 KiB shard header is additional.
    std::uint64_t shard_size_bytes = 16ULL << 30;
};

struct ReaderOptions {
    // Adjacent requested records are read as one range up to this size.
    std::uint64_t max_read_range_bytes = 1ULL << 20;
    // A range may include this many unrequested bytes between records.
    std::uint64_t max_gap_bytes = 0;
    // Zero keeps the deterministic sequential pread baseline. A positive
    // value creates one executor shared by every get() call on this Storage.
    std::uint32_t io_workers = 0;
    // Maximum number of tasks waiting for a worker. With workers enabled,
    // zero selects 2 * io_workers. submit() applies backpressure at the bound.
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
};

class Writer {
public:
    static std::unique_ptr<Writer> create(
        const std::filesystem::path& path,
        WriterOptions options = {});

    ~Writer();

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&&) = delete;
    Writer& operator=(Writer&&) = delete;

    // append() is synchronous. The input buffers may be reused when it returns.
    void append(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);

    void append(
        const std::uint64_t* ids,
        std::size_t count,
        const void* embeddings,
        std::size_t embedding_bytes);

    // Makes all shard and index files durable, then atomically publishes the
    // manifest. Readers cannot open a build until finish() succeeds.
    void finish();

    [[nodiscard]] std::uint64_t count() const noexcept;
    [[nodiscard]] std::uint32_t dimension() const noexcept;
    [[nodiscard]] std::uint64_t record_size() const noexcept;
    [[nodiscard]] bool finished() const noexcept;

private:
    class Impl;
    explicit Writer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Storage {
public:
    static std::unique_ptr<Storage> open(
        const std::filesystem::path& path,
        ReaderOptions options = {});

    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = delete;
    Storage& operator=(Storage&&) = delete;

    // Missing IDs are a strict error. Lookup of every ID completes before any
    // output bytes are modified.
    void get_into(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;

    void get_into(
        const std::uint64_t* ids,
        std::size_t count,
        void* output,
        std::size_t output_bytes) const;

    // float16 values are returned as their raw IEEE-754 binary16 bit patterns.
    [[nodiscard]] std::vector<std::uint16_t> get(
        std::span<const std::uint64_t> ids) const;

    [[nodiscard]] std::uint64_t count() const noexcept;
    [[nodiscard]] std::uint32_t dimension() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] std::uint64_t record_size() const noexcept;

private:
    class Impl;
    explicit Storage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

using StorageWriter = Writer;
using StorageReader = Storage;

}  // namespace throwdb
