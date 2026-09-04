#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "internal.hpp"
#include "mutation_types.hpp"
#include "native_collection_test_hooks.hpp"

namespace throwdb::detail {

// Durable format-7 wire primitives. Individual resource wire versions are
// independent of the public collection format number.
inline constexpr std::uint32_t kPartitionedPublicFormatVersion = 7;
inline constexpr std::uint32_t kPartitionedManifestWireVersion = 2;
inline constexpr std::uint32_t kPartitionedCatalogWireVersion = 3;
inline constexpr std::uint32_t kPartitionedResourceWireVersion = 1;
inline constexpr std::uint64_t kPartitionedHeaderSize = 4096;
inline constexpr std::uint32_t kPartitionedChecksumAlgorithmCrc32c = 1;

inline constexpr std::uint64_t kRequiredFeatureCatalogV1 = 1ULL << 0;
inline constexpr std::uint64_t kRequiredFeatureSegmentCatalogV1 = 1ULL << 2;
inline constexpr std::uint64_t kRequiredFeaturePartitionedRowRefV1 = 1ULL << 3;
inline constexpr std::uint64_t kPartitionedRequiredFeatureBits =
    kRequiredFeatureCatalogV1
    | kRequiredFeatureSegmentCatalogV1
    | kRequiredFeaturePartitionedRowRefV1;
inline constexpr std::uint64_t kPartitionedSupportedRequiredFeatureBits =
    kPartitionedRequiredFeatureBits;

inline constexpr std::uint32_t kPartitionedMaximumPartitionBits = 16;
inline constexpr std::uint64_t kPartitionedMaximumResourceEntries = 1'000'000;
inline constexpr std::uint64_t kPartitionedMaximumRuns = 4096;
// Covers the worst supported v3 catalog geometry (segments, snapshots, runs,
// WALs, and two run bitmaps) with headroom, while preventing corrupt CURRENT
// metadata from driving an unbounded allocation before catalog validation.
inline constexpr std::uint64_t kPartitionedMaximumCatalogFileSize =
    512ULL << 20;
inline constexpr std::uint64_t kPartitionedSegmentIdEntrySize = 16;
inline constexpr std::uint64_t kPartitionSnapshotLookupSlotSize = 16;
inline constexpr std::uint64_t kPartitionSnapshotSequenceSlotSize = 8;
inline constexpr std::uint64_t kPartitionRunRecordSize = 24;
inline constexpr std::uint64_t kPartitionRunTocEntrySize = 16;

enum class PartitionedSegmentKind : std::uint32_t {
    Initial = 1,
    Materialized = 2,
    Cleaner = 3,
};

enum class PartitionedRunKind : std::uint32_t {
    Mutation = 1,
    Cleaner = 2,
};

struct PartitionedManifestV2 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    Uuid128 initial_active_wal_id;
    CollectionSchema schema;
    std::uint64_t record_size{0};
    std::uint32_t partition_bits{12};
    std::uint64_t partition_count{4096};
    std::uint64_t hash_seed{kHashSeed};
    std::uint64_t target_segment_size{4ULL << 30};
    std::uint64_t initial_record_count{0};
    std::uint32_t initial_active_wal_segment_id{1};
    std::uint32_t initial_next_segment_id{2};
    std::uint64_t required_feature_bits{kPartitionedRequiredFeatureBits};
    std::uint64_t optional_feature_bits{0};
    std::uint64_t creation_time_unix_ns{0};

    bool operator==(const PartitionedManifestV2&) const = default;
};

struct PartitionedSegmentDescriptor {
    Uuid128 segment_id;
    std::uint32_t row_ref_segment_id{0};
    PartitionedSegmentKind kind{PartitionedSegmentKind::Initial};
    std::uint64_t row_count{0};
    // Initial rows use sequence zero. UINT64_MAX is always reserved.
    std::uint64_t minimum_sequence{0};
    std::uint64_t maximum_sequence{0};
    std::uint64_t data_file_size{0};
    std::uint64_t ids_file_size{0};
    std::uint32_t data_payload_crc32c{0};
    std::uint32_t ids_payload_crc32c{0};
    std::uint64_t creation_epoch{0};

    bool operator==(const PartitionedSegmentDescriptor&) const = default;
};

struct PartitionSnapshotDescriptor {
    Uuid128 snapshot_id;
    std::uint32_t partition_id{0};
    std::uint64_t epoch{0};
    std::uint64_t entry_count{0};
    std::uint64_t capacity{0};
    std::uint64_t file_size{0};
    std::uint32_t body_crc32c{0};

    bool operator==(const PartitionSnapshotDescriptor&) const = default;
};

struct PartitionIndexRunDescriptor {
    Uuid128 run_id;
    std::uint64_t epoch{0};
    PartitionedRunKind kind{PartitionedRunKind::Mutation};
    std::uint64_t first_sequence{0};
    std::uint64_t last_sequence{0};
    std::uint64_t entry_count{0};
    std::uint64_t file_size{0};
    std::uint32_t body_crc32c{0};
    // Immutable partition membership derived from nonempty TOC sections.
    // active_partitions is a catalog-generation subset of this bitmap.
    std::vector<std::byte> covered_partitions;
    // Exactly ceil(partition_count / 8) bytes. A set bit means that the run's
    // section still participates in lookup for that partition.
    std::vector<std::byte> active_partitions;
    // Cleaner runs bind the exact source whose lower mappings they fence.
    std::uint32_t retired_row_ref_segment_id{0};
    Uuid128 retired_segment_id;
    std::uint64_t retired_row_count{0};
    std::uint32_t retired_ids_payload_crc32c{0};

    bool operator==(const PartitionIndexRunDescriptor&) const = default;
};

struct PartitionedCatalogV3 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint64_t generation{0};
    Uuid128 catalog_id;
    Uuid128 previous_catalog_id;
    std::uint32_t next_segment_id{0};
    std::uint64_t next_index_epoch{1};
    std::uint64_t indexed_sequence{0};
    std::uint64_t indexed_live_count{0};
    Uuid128 terminal_wal_id;
    std::uint32_t partition_bits{12};
    std::uint64_t hash_seed{kHashSeed};
    std::vector<PartitionedSegmentDescriptor> segments;
    // Exactly one descriptor per partition, ordered by partition_id.
    std::vector<PartitionSnapshotDescriptor> snapshots;
    // Strictly increasing epoch order.
    std::vector<PartitionIndexRunDescriptor> runs;
    // Oldest frozen first and exactly one active WAL last.
    std::vector<MutationWalEntry> wal_entries;

    bool operator==(const PartitionedCatalogV3&) const = default;
};

struct PartitionedCurrentV1 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint64_t generation{0};
    Uuid128 catalog_id;
    std::uint64_t catalog_file_size{0};
    std::uint32_t catalog_crc32c{0};

    bool operator==(const PartitionedCurrentV1&) const = default;
};

struct PartitionedSegmentDataHeaderV1 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    PartitionedSegmentDescriptor descriptor;
    CollectionSchema schema;
    std::uint64_t record_size{0};
    std::uint64_t payload_offset{kPartitionedHeaderSize};
    std::uint64_t payload_size{0};

    bool operator==(const PartitionedSegmentDataHeaderV1&) const = default;
};

struct PartitionedSegmentIdsHeaderV1 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    PartitionedSegmentDescriptor descriptor;
    IdType id_type{IdType::UInt64};
    std::uint64_t entry_size{kPartitionedSegmentIdEntrySize};
    std::uint64_t payload_offset{kPartitionedHeaderSize};
    std::uint64_t payload_size{0};

    bool operator==(const PartitionedSegmentIdsHeaderV1&) const = default;
};

struct PartitionSnapshotHeaderV1 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    PartitionSnapshotDescriptor descriptor;
    std::uint32_t partition_bits{12};
    std::uint64_t hash_seed{kHashSeed};
    std::uint64_t controls_offset{kPartitionedHeaderSize};
    std::uint64_t lookup_slots_offset{kPartitionedHeaderSize};
    std::uint64_t sequence_slots_offset{kPartitionedHeaderSize};
    std::uint64_t lookup_slot_size{kPartitionSnapshotLookupSlotSize};
    std::uint64_t sequence_slot_size{kPartitionSnapshotSequenceSlotSize};

    bool operator==(const PartitionSnapshotHeaderV1&) const = default;
};

struct PartitionIndexRunHeaderV1 {
    Uuid128 collection_id;
    Uuid128 schema_id;
    PartitionIndexRunDescriptor descriptor;
    std::uint32_t partition_bits{12};
    std::uint64_t hash_seed{kHashSeed};
    std::uint64_t toc_offset{kPartitionedHeaderSize};
    std::uint64_t toc_size{0};
    std::uint64_t records_offset{kPartitionedHeaderSize};
    std::uint64_t records_size{0};
    std::uint64_t toc_entry_size{kPartitionRunTocEntrySize};
    std::uint64_t record_size{kPartitionRunRecordSize};

    bool operator==(const PartitionIndexRunHeaderV1&) const = default;
};

struct PartitionRunTocEntry {
    std::uint64_t first_record{0};
    std::uint64_t record_count{0};

    bool operator==(const PartitionRunTocEntry&) const = default;
};

struct PartitionRunRecord {
    std::uint64_t id{0};
    std::uint64_t sequence{0};
    // Zero is a tombstone. Every nonzero value must decode as RowRef.
    std::uint64_t value_raw{0};

    [[nodiscard]] bool tombstone() const noexcept { return value_raw == 0; }
    bool operator==(const PartitionRunRecord&) const = default;
};

[[nodiscard]] std::uint64_t partition_bitmap_size(std::uint32_t partition_bits);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_manifest_v2(const PartitionedManifestV2& manifest);
[[nodiscard]] PartitionedManifestV2 decode_partitioned_manifest_v2(
    std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_segment_data_header_v1(
    const PartitionedSegmentDataHeaderV1& header);
[[nodiscard]] PartitionedSegmentDataHeaderV1
decode_partitioned_segment_data_header_v1(std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_segment_ids_header_v1(
    const PartitionedSegmentIdsHeaderV1& header);
[[nodiscard]] PartitionedSegmentIdsHeaderV1
decode_partitioned_segment_ids_header_v1(std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partition_snapshot_header_v1(const PartitionSnapshotHeaderV1& header);
[[nodiscard]] PartitionSnapshotHeaderV1 decode_partition_snapshot_header_v1(
    std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partition_index_run_header_v1(
    const PartitionIndexRunHeaderV1& header);
[[nodiscard]] PartitionIndexRunHeaderV1 decode_partition_index_run_header_v1(
    std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionRunTocEntrySize>
encode_partition_run_toc_entry(const PartitionRunTocEntry& entry);
[[nodiscard]] PartitionRunTocEntry decode_partition_run_toc_entry(
    std::span<const std::byte> bytes);
[[nodiscard]] std::array<std::byte, kPartitionRunRecordSize>
encode_partition_run_record(const PartitionRunRecord& record);
[[nodiscard]] PartitionRunRecord decode_partition_run_record(
    std::span<const std::byte> bytes);

[[nodiscard]] std::vector<std::byte> encode_partitioned_catalog_v3(
    const PartitionedCatalogV3& catalog);
[[nodiscard]] PartitionedCatalogV3 decode_partitioned_catalog_v3(
    std::span<const std::byte> bytes);

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_current_v1(
    const PartitionedManifestV2& manifest,
    const PartitionedCurrentV1& current);
[[nodiscard]] PartitionedCurrentV1 decode_partitioned_current_v1(
    std::span<const std::byte> bytes,
    const PartitionedManifestV2& manifest);

}  // namespace throwdb::detail
