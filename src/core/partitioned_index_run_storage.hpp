#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "partitioned_index_format.hpp"

namespace throwdb::detail {

// Scalar catalog identity supplied by a lifecycle build.  entry_count,
// file_size, body_crc32c and both partition bitmaps are derived by the writer
// and must be left empty/zero by the caller.
struct PartitionIndexRunBuildSpec {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint32_t partition_bits{12};
    std::uint64_t hash_seed{kHashSeed};
    PartitionIndexRunDescriptor descriptor;
};

[[nodiscard]] std::filesystem::path partitioned_index_runs_directory(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path partitioned_index_run_path(
    const std::filesystem::path& store_path, Uuid128 run_id);

// Writes one UUID-addressed immutable run directly into index/runs.  The
// destination namespace must already exist.  The file is created O_EXCL and
// fully fsynced; publication code remains responsible for fsyncing the
// directory before making a catalog generation visible.
[[nodiscard]] PartitionIndexRunDescriptor write_partitioned_index_run(
    const std::filesystem::path& store_path,
    const PartitionIndexRunBuildSpec& spec,
    std::span<const PartitionRunRecord> records);

// Owns an mmap of an immutable run.  Construction validates the complete body
// and then closes the source descriptor, so opening many runs does not retain
// one FD per run.  Indexed access is intentionally allocation-free and gives
// lifecycle jobs a bounded way to iterate records.
class PartitionIndexRunReader {
public:
    PartitionIndexRunReader(
        std::filesystem::path store_path,
        Uuid128 collection_id,
        Uuid128 schema_id,
        std::uint32_t partition_bits,
        std::uint64_t hash_seed,
        PartitionIndexRunDescriptor expected_descriptor);
    ~PartitionIndexRunReader();

    PartitionIndexRunReader(const PartitionIndexRunReader&) = delete;
    PartitionIndexRunReader& operator=(const PartitionIndexRunReader&) = delete;
    PartitionIndexRunReader(PartitionIndexRunReader&&) noexcept;
    PartitionIndexRunReader& operator=(PartitionIndexRunReader&&) noexcept;

    [[nodiscard]] const PartitionIndexRunDescriptor& descriptor() const noexcept;
    [[nodiscard]] std::uint64_t record_count() const noexcept;
    [[nodiscard]] std::uint64_t partition_count() const noexcept;
    [[nodiscard]] PartitionRunTocEntry partition_records(
        std::uint32_t partition) const;
    [[nodiscard]] PartitionRunRecord record_at(std::uint64_t index) const;
    [[nodiscard]] std::optional<PartitionRunRecord> lookup(
        std::uint32_t partition, std::uint64_t id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::detail
