#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "partitioned_index_format.hpp"
#include "physical_reader.hpp"

namespace throwdb::detail {

struct PartitionedSegmentSetBuildSpec {
    PartitionedManifestV2 manifest;
    PartitionedSegmentKind kind{PartitionedSegmentKind::Materialized};
    std::uint32_t first_row_ref_segment_id{0};
    std::uint64_t creation_epoch{0};
    // Optional deterministic allocation seam. Production callers leave this
    // empty and receive process-unique UUIDs.
    std::function<Uuid128()> segment_id_factory;
};

struct PartitionedSegmentIdEntry {
    std::uint64_t id{0};
    std::uint64_t sequence{0};

    bool operator==(const PartitionedSegmentIdEntry&) const = default;
};

[[nodiscard]] std::filesystem::path partitioned_segments_directory(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path partitioned_segment_data_path(
    const std::filesystem::path& store_path,
    Uuid128 segment_id);
[[nodiscard]] std::filesystem::path partitioned_segment_ids_path(
    const std::filesystem::path& store_path,
    Uuid128 segment_id);

// Number of rows placed in one segment. The data-payload target from the
// manifest is always additionally bounded by the 32-bit RowRef row field.
[[nodiscard]] std::uint64_t partitioned_segment_row_capacity(
    const PartitionedManifestV2& manifest);

// Streaming immutable segment-set builder used by WAL materialization and
// selective cleaning. Final UUID resources are published only by finish();
// until then all output names are private staging files.
class PartitionedSegmentSetWriter {
public:
    PartitionedSegmentSetWriter(
        std::filesystem::path store_path,
        PartitionedSegmentSetBuildSpec spec);
    ~PartitionedSegmentSetWriter();

    PartitionedSegmentSetWriter(const PartitionedSegmentSetWriter&) = delete;
    PartitionedSegmentSetWriter& operator=(
        const PartitionedSegmentSetWriter&) = delete;
    PartitionedSegmentSetWriter(PartitionedSegmentSetWriter&&) = delete;
    PartitionedSegmentSetWriter& operator=(
        PartitionedSegmentSetWriter&&) = delete;

    // Appends exactly one complete opaque row and its durable sidecar tuple.
    // The returned RowRef is safe to publish only after finish() succeeds.
    [[nodiscard]] RowRef append(
        std::uint64_t id,
        std::uint64_t sequence,
        std::span<const std::byte> row);

    // Idempotent after success. Every data/IDs file and the segments directory
    // are durable before descriptors become observable to the caller.
    [[nodiscard]] const std::vector<PartitionedSegmentDescriptor>& finish();

    [[nodiscard]] const std::vector<PartitionedSegmentDescriptor>&
    descriptors() const;
    [[nodiscard]] std::uint64_t row_count() const;
    [[nodiscard]] bool finished() const;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool owned_by_current_process() const noexcept;
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Validated open view over one catalog-referenced immutable segment. The data
// FD remains alive so resolve() requests are valid until this object is
// destroyed or abandoned. Sidecar reads are allocation-free.
class PartitionedSegmentReader {
public:
    PartitionedSegmentReader(
        std::filesystem::path store_path,
        PartitionedManifestV2 manifest,
        PartitionedSegmentDescriptor expected_descriptor,
        bool verify_payload_crc);
    ~PartitionedSegmentReader();

    PartitionedSegmentReader(const PartitionedSegmentReader&) = delete;
    PartitionedSegmentReader& operator=(const PartitionedSegmentReader&) = delete;
    PartitionedSegmentReader(PartitionedSegmentReader&&) noexcept;
    PartitionedSegmentReader& operator=(PartitionedSegmentReader&&) noexcept;

    [[nodiscard]] const PartitionedSegmentDescriptor& descriptor() const;
    [[nodiscard]] std::uint64_t row_count() const;
    [[nodiscard]] PartitionedSegmentIdEntry sidecar_at(
        std::uint64_t row_index) const;
    [[nodiscard]] PhysicalReadRequest resolve(
        RowRef row_ref,
        std::uint64_t output_index) const;

    [[nodiscard]] bool owned_by_current_process() const noexcept;
    void abandon_after_fork() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::unique_ptr<PartitionedSegmentReader>
open_partitioned_segment(
    std::filesystem::path store_path,
    PartitionedManifestV2 manifest,
    PartitionedSegmentDescriptor expected_descriptor,
    bool verify_payload_crc = false);

}  // namespace throwdb::detail
