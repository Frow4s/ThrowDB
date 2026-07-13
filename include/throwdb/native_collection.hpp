#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "throwdb/storage.hpp"

namespace throwdb {

struct NativeBuilderOptions {
    std::uint32_t dimension = 1024;
    DType dtype = DType::Float16;
    // Target payload bytes per immutable base segment. Headers/ID sidecars are
    // additional. The builder also caps every segment at 2^32 rows.
    std::uint64_t segment_size_bytes = 16ULL << 30;
};

struct NativeOpenStats {
    double total_seconds = 0.0;
    double manifest_seconds = 0.0;
    double index_seconds = 0.0;
    double index_header_seconds = 0.0;
    double index_allocation_seconds = 0.0;
    double index_controls_seconds = 0.0;
    double index_slots_seconds = 0.0;
    double index_checksum_seconds = 0.0;
    double index_structural_validation_seconds = 0.0;
    double base_segments_seconds = 0.0;
    double wal_replay_seconds = 0.0;
    std::uint64_t index_file_bytes = 0;
    bool index_body_checksum_present = false;
    bool index_body_checksum_verified = false;
    bool index_structural_validation_performed = false;
};

// Per-call diagnostics collected only by get_into_profiled(). The ordinary
// get/get_into path does not read clocks or update telemetry counters.
struct NativeReadStats {
    double total_seconds = 0.0;
    double lookup_seconds = 0.0;
    double physical_plan_seconds = 0.0;
    // Queue/admission wait, physical I/O, and scatter copies currently share
    // one wall-clock phase because those operations overlap in parallel
    // backends. More detailed backend hooks can extend this later.
    double physical_backend_seconds = 0.0;
    std::uint64_t requested_rows = 0;
    std::uint64_t base_rows = 0;
    std::uint64_t mutation_rows = 0;
    std::uint64_t planned_ranges = 0;
    std::uint64_t planned_read_bytes = 0;
    std::uint64_t backend_tasks = 0;
    bool zero_delta_fast_path = false;
};

// Durable mutation-log state captured under the collection's mutation lock.
// rotate_wal() returns the same snapshot with rotated=true when it published a
// successor WAL. An empty active WAL is a successful no-op.
struct NativeMutationLogStats {
    bool catalog_managed = false;
    bool delta_managed = false;
    std::uint64_t generation = 0;
    std::uint64_t committed_sequence = 0;
    std::uint64_t materialized_sequence = 0;
    std::uint64_t frozen_wal_count = 0;
    std::uint64_t active_wal_bytes = 0;
    std::uint64_t active_wal_put_rows = 0;
    std::uint64_t total_wal_bytes = 0;
    std::uint64_t delta_run_count = 0;
    std::uint64_t delta_entry_count = 0;
    std::uint64_t delta_put_rows = 0;
    std::uint64_t total_delta_bytes = 0;
    std::uint64_t reclaimed_wal_count = 0;
    std::uint64_t reclaimed_wal_bytes = 0;
    bool rotated = false;
    bool materialized = false;
    bool reclamation_complete = true;
};

class NativeCollectionBuilder {
public:
    static std::unique_ptr<NativeCollectionBuilder> create(
        const std::filesystem::path& path,
        NativeBuilderOptions options = {});

    ~NativeCollectionBuilder();

    NativeCollectionBuilder(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder& operator=(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder(NativeCollectionBuilder&&) = delete;
    NativeCollectionBuilder& operator=(NativeCollectionBuilder&&) = delete;

    void append(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void finish();

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool finished() const;

private:
    class Impl;
    explicit NativeCollectionBuilder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class NativeCollection {
public:
    static std::unique_ptr<NativeCollection> open(
        const std::filesystem::path& path,
        ReaderOptions reader_options = {},
        bool writable = false);

    ~NativeCollection();

    NativeCollection(const NativeCollection&) = delete;
    NativeCollection& operator=(const NativeCollection&) = delete;
    NativeCollection(NativeCollection&&) = delete;
    NativeCollection& operator=(NativeCollection&&) = delete;

    void get_into(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeReadStats get_into_profiled(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] std::vector<std::byte> get_bytes(
        std::span<const std::uint64_t> ids) const;

    void insert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void update(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void upsert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    [[nodiscard]] std::vector<std::uint8_t> erase(
        std::span<const std::uint64_t> ids);

    [[nodiscard]] NativeMutationLogStats mutation_log_stats() const;
    [[nodiscard]] NativeMutationLogStats rotate_wal();
    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals();

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool writable() const;
    [[nodiscard]] NativeOpenStats open_stats() const;

private:
    class Impl;
    explicit NativeCollection(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb
