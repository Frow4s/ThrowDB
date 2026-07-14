#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "partitioned_resource_inventory.hpp"
#include "physical_reader.hpp"
#include "throwdb/native_collection.hpp"

namespace throwdb::detail {

enum class PartitionedPutMode : std::uint8_t {
    Insert,
    Update,
    Upsert,
};

// Reads only the manifest discriminator, without following a substituted
// symlink or blocking on a non-regular file. A false result means that the
// existing format-6 path should perform its normal validation.
[[nodiscard]] bool is_partitioned_index_store(
    const std::filesystem::path& store_path);

[[nodiscard]] NativeCollectionMetadata inspect_partitioned_index_metadata(
    const std::filesystem::path& store_path);

class PartitionedIndexBuilder {
public:
    PartitionedIndexBuilder(
        std::filesystem::path store_path,
        NativeBuilderOptions options);
    ~PartitionedIndexBuilder();

    PartitionedIndexBuilder(const PartitionedIndexBuilder&) = delete;
    PartitionedIndexBuilder& operator=(const PartitionedIndexBuilder&) = delete;

    void append(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void finish();

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool finished() const;
    [[nodiscard]] bool owned_by_current_process() const noexcept;
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class PartitionedIndexReader {
public:
    PartitionedIndexReader(
        std::filesystem::path store_path,
        const ReaderOptions& reader_options,
        bool writable,
        bool calibration = false);
    ~PartitionedIndexReader();

    PartitionedIndexReader(const PartitionedIndexReader&) = delete;
    PartitionedIndexReader& operator=(const PartitionedIndexReader&) = delete;

    void get_into(
        const PhysicalReader& physical_reader,
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeReadStats get_into_profiled(
        const PhysicalReader& physical_reader,
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] NativeIdSample sample_ids(
        std::uint64_t requested,
        std::uint64_t seed,
        std::uint64_t max_inspected_slots) const;
    void put(
        PartitionedPutMode mode,
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    [[nodiscard]] std::vector<std::uint8_t> erase(
        std::span<const std::uint64_t> ids);
    [[nodiscard]] NativeMutationLogStats mutation_log_stats() const;
    [[nodiscard]] NativeMutationLogStats rotate_wal();
    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals(
        const PhysicalReader& physical_reader,
        std::uint64_t max_wals,
        std::uint64_t max_input_bytes);
    [[nodiscard]] NativeMutationLogStats fold_partitions(
        std::uint64_t max_runs,
        std::uint64_t max_input_bytes,
        std::uint64_t max_input_entries);
    [[nodiscard]] NativeSegmentCleaningStats clean_segments(
        const PhysicalReader& physical_reader,
        NativeSegmentCleaningOptions options);
    [[nodiscard]] PartitionedResourceInventory collect_orphans();
    [[nodiscard]] PartitionedOrphanCollectionResult reclaim_orphans(
        PartitionedOrphanCollectionOptions options);

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool writable() const;
    [[nodiscard]] NativeOpenStats open_stats() const;
    [[nodiscard]] bool owned_by_current_process() const noexcept;
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
