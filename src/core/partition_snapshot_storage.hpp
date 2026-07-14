#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "partitioned_index_format.hpp"

namespace throwdb::detail {

struct PartitionSnapshotBuildSpec {
    Uuid128 collection_id;
    Uuid128 schema_id;
    Uuid128 snapshot_id;
    std::uint32_t partition_bits{12};
    std::uint32_t partition_id{0};
    std::uint64_t hash_seed{kHashSeed};
    // Selective folds always publish a new authoritative epoch. Epoch zero is
    // reserved for snapshots constructed with the initial collection.
    std::uint64_t epoch{0};
};

struct PartitionSnapshotRecord {
    std::uint64_t id;
    std::uint64_t sequence;
    RowRef row_ref;

    PartitionSnapshotRecord(
        const std::uint64_t id_value,
        const std::uint64_t sequence_value,
        const RowRef row_value)
        : id(id_value), sequence(sequence_value), row_ref(row_value) {}

    bool operator==(const PartitionSnapshotRecord&) const = default;
};

[[nodiscard]] std::filesystem::path partitioned_partition_snapshots_directory(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path partitioned_partition_snapshot_path(
    const std::filesystem::path& store_path,
    const PartitionSnapshotDescriptor& descriptor);

// Builds one complete authoritative partition. The destination namespace must
// already exist. The UUID-derived file is created O_EXCL and fsynced; catalog
// publication remains responsible for syncing index/partitions first.
[[nodiscard]] PartitionSnapshotDescriptor write_partition_snapshot(
    const std::filesystem::path& store_path,
    const PartitionSnapshotBuildSpec& spec,
    std::span<const PartitionRunRecord> records);

// Pins one immutable snapshot with mmap and immediately closes the file
// descriptor. Every slot is structurally validated during construction.
class PartitionSnapshotReader {
public:
    PartitionSnapshotReader(
        std::filesystem::path store_path,
        Uuid128 collection_id,
        Uuid128 schema_id,
        std::uint32_t partition_bits,
        std::uint64_t hash_seed,
        PartitionSnapshotDescriptor expected_descriptor,
        bool verify_structure = true);
    ~PartitionSnapshotReader();

    PartitionSnapshotReader(const PartitionSnapshotReader&) = delete;
    PartitionSnapshotReader& operator=(const PartitionSnapshotReader&) = delete;
    PartitionSnapshotReader(PartitionSnapshotReader&&) noexcept;
    PartitionSnapshotReader& operator=(PartitionSnapshotReader&&) noexcept;

    [[nodiscard]] const PartitionSnapshotDescriptor& descriptor() const noexcept;
    [[nodiscard]] std::uint64_t record_count() const noexcept;
    [[nodiscard]] std::uint64_t slot_count() const noexcept;
    // Allocation-free lifecycle iteration: scan [0, slot_count()) and consume
    // every engaged value.
    [[nodiscard]] std::optional<PartitionSnapshotRecord> slot_record(
        std::uint64_t slot) const;
    [[nodiscard]] std::optional<PartitionSnapshotRecord> lookup(
        std::uint64_t id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
