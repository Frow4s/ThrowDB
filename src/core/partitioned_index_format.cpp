#include "partitioned_index_format.hpp"

#include "crc32c.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace throwdb::detail {
namespace {

constexpr std::array<char, 8> kPartitionedManifestMagic{
    'T', 'D', 'B', 'N', 'M', 'N', 'F', '2'};
constexpr std::array<char, 8> kPartitionedSegmentDataMagic{
    'T', 'D', 'B', 'S', 'D', 'A', 'T', '1'};
constexpr std::array<char, 8> kPartitionedSegmentIdsMagic{
    'T', 'D', 'B', 'S', 'I', 'D', 'S', '1'};
constexpr std::array<char, 8> kPartitionedSnapshotMagic{
    'T', 'D', 'B', 'P', 'S', 'N', 'P', '1'};
constexpr std::array<char, 8> kPartitionedRunMagic{
    'T', 'D', 'B', 'P', 'R', 'U', 'N', '1'};
constexpr std::array<char, 8> kPartitionedCatalogMagic{
    'T', 'D', 'B', 'M', 'C', 'A', 'T', '3'};
constexpr std::array<char, 8> kPartitionedCurrentMagic{
    'T', 'D', 'B', 'M', 'C', 'U', 'R', '1'};

constexpr std::size_t kChecksumOffset = 20;
constexpr std::size_t kManifestKnownBytes = 176;
constexpr std::size_t kSegmentDataKnownBytes = 184;
constexpr std::size_t kSegmentIdsKnownBytes = 168;
constexpr std::size_t kSnapshotKnownBytes = 168;
constexpr std::size_t kRunKnownBytes = 224;
constexpr std::size_t kCatalogKnownHeaderBytes = 264;
constexpr std::size_t kCurrentKnownBytes = 96;

constexpr std::uint64_t kCatalogSegmentEntrySize = 128;
constexpr std::uint64_t kCatalogSnapshotEntrySize = 128;
constexpr std::uint64_t kCatalogRunEntrySize = 192;
constexpr std::uint64_t kCatalogWalEntrySize = 128;

constexpr std::uint64_t kMaximumSequence =
    std::numeric_limits<std::uint64_t>::max() - 1;

template <typename Error>
void require(const bool condition, const std::string_view message) {
    static_assert(std::is_base_of_v<std::exception, Error>);
    if (!condition) {
        throw Error(std::string(message));
    }
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left + right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left * right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_align(
    const std::uint64_t value,
    const std::uint64_t alignment,
    const std::string_view description) {
    require<Error>(
        alignment != 0 && (alignment & (alignment - 1)) == 0,
        std::string(description) + " has invalid alignment");
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0
               ? value
               : checked_sum<Error>(
                     value, alignment - remainder, description);
}

void put_uuid(
    const std::span<std::byte> bytes,
    const std::size_t offset,
    const Uuid128 value) {
    put_u64(bytes, offset, value.low);
    put_u64(bytes, offset + 8, value.high);
}

[[nodiscard]] Uuid128 get_uuid(
    const std::span<const std::byte> bytes,
    const std::size_t offset) {
    return Uuid128{get_u64(bytes, offset), get_u64(bytes, offset + 8)};
}

void put_magic(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    std::memcpy(bytes.data(), magic.data(), magic.size());
}

template <typename Error>
void require_zero_range(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::size_t end,
    const std::string_view description) {
    require<Error>(
        begin <= end && end <= bytes.size(),
        std::string(description) + " has invalid reserved geometry");
    for (std::size_t offset = begin; offset != end; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw Error(
                std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

[[nodiscard]] std::uint32_t checksum_with_zeroed_field(
    const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes.first(kChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(kChecksumOffset + sizeof(std::uint32_t)));
    return checksum.value();
}

void encode_common(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic,
    const std::uint32_t version,
    const std::uint32_t header_size) {
    put_magic(bytes, magic);
    put_u32(bytes, 8, version);
    put_u32(bytes, 12, header_size);
    put_u32(bytes, 16, kPartitionedChecksumAlgorithmCrc32c);
}

void finish_checksum(const std::span<std::byte> bytes) {
    put_u32(bytes, kChecksumOffset, 0);
    put_u32(bytes, kChecksumOffset, checksum_with_zeroed_field(bytes));
}

template <typename Error>
void decode_common(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic,
    const std::uint32_t version,
    const std::uint32_t header_size,
    const std::string_view description) {
    require<Error>(
        bytes.size() >= header_size,
        std::string(description) + " is truncated");
    require<Error>(
        std::memcmp(bytes.data(), magic.data(), magic.size()) == 0,
        std::string(description) + " has invalid magic");
    require<Error>(
        get_u32(bytes, 8) == version,
        std::string(description) + " has unsupported version");
    require<Error>(
        get_u32(bytes, 12) == header_size,
        std::string(description) + " has invalid header size");
    require<Error>(
        get_u32(bytes, 16) == kPartitionedChecksumAlgorithmCrc32c,
        std::string(description) + " has unsupported checksum algorithm");
    require<Error>(
        get_u32(bytes, kChecksumOffset) == checksum_with_zeroed_field(bytes),
        std::string(description) + " checksum mismatch");
}

template <typename Error>
void validate_schema(
    const CollectionSchema& schema,
    const std::uint64_t record_size) {
    require<Error>(
        schema.id_type == IdType::UInt64,
        "partitioned format requires uint64 IDs");
    require<Error>(
        schema.dtype == DTypeCode::Float16
            || schema.dtype == DTypeCode::Float32,
        "partitioned format has unsupported dtype");
    require<Error>(schema.dimension != 0, "partitioned schema has zero dimension");
    require<Error>(
        schema.item_size == dtype_item_size(schema.dtype),
        "partitioned schema dtype and item size disagree");
    require<Error>(
        schema.row_layout == RowLayout::DenseRowMajor,
        "partitioned format has unsupported row layout");
    require<Error>(
        schema.index_kind == IndexKind::UInt64PartitionedRowRef,
        "partitioned format has unsupported index kind discriminator");
    require<Error>(
        schema.hash_algorithm == HashAlgorithm::SplitMix64V1,
        "partitioned format has unsupported hash algorithm");
    require<Error>(
        record_size
            == checked_product<Error>(
                schema.dimension, schema.item_size,
                "partitioned record size"),
        "partitioned record size does not match schema");
}

template <typename Error>
[[nodiscard]] std::uint64_t validate_partition_bits(
    const std::uint32_t partition_bits) {
    require<Error>(
        partition_bits <= kPartitionedMaximumPartitionBits,
        "partitioned format exceeds the maximum partition bits");
    return std::uint64_t{1} << partition_bits;
}

template <typename Error>
void validate_sequence_range(
    const std::uint64_t minimum,
    const std::uint64_t maximum,
    const std::string_view description) {
    require<Error>(
        minimum <= maximum && maximum <= kMaximumSequence,
        std::string(description) + " has invalid sequence range");
}

template <typename Error>
void validate_segment_id(
    const std::uint32_t value,
    const std::string_view description) {
    require<Error>(
        value >= SegmentId::kMinimum && value <= SegmentId::kMaximum,
        std::string(description) + " is invalid or reserved");
}

template <typename Error>
void validate_segment_kind(const PartitionedSegmentKind kind) {
    require<Error>(
        kind == PartitionedSegmentKind::Initial
            || kind == PartitionedSegmentKind::Materialized
            || kind == PartitionedSegmentKind::Cleaner
            || kind == PartitionedSegmentKind::LegacyDeltaV1,
        "partitioned segment has unknown kind");
}

template <typename Error>
void validate_run_kind(const PartitionedRunKind kind) {
    require<Error>(
        kind == PartitionedRunKind::Mutation
            || kind == PartitionedRunKind::Cleaner
            || kind == PartitionedRunKind::LegacyDeltaV1,
        "partitioned index run has unknown kind");
}

template <typename Error>
void validate_segment_descriptor(
    const PartitionedSegmentDescriptor& descriptor) {
    require<Error>(!descriptor.segment_id.empty(), "segment UUID is empty");
    validate_segment_id<Error>(
        descriptor.row_ref_segment_id, "segment RowRef segment ID");
    validate_segment_kind<Error>(descriptor.kind);
    require<Error>(descriptor.row_count != 0, "segment row count is zero");
    require<Error>(
        descriptor.row_count <= RowRef::kMaximumRowsPerSegment,
        "segment row count exceeds RowRef capacity");
    validate_sequence_range<Error>(
        descriptor.minimum_sequence, descriptor.maximum_sequence,
        "segment descriptor");
    if (descriptor.kind == PartitionedSegmentKind::Initial) {
        require<Error>(
            descriptor.minimum_sequence == 0
                && descriptor.maximum_sequence == 0
                && descriptor.creation_epoch == 0,
            "initial segment has mutation sequence or creation epoch");
    } else {
        require<Error>(
            descriptor.creation_epoch != 0,
            "non-initial segment creation epoch is zero");
        if (descriptor.kind != PartitionedSegmentKind::Cleaner) {
            require<Error>(
                descriptor.minimum_sequence != 0,
                "materialized segment uses initial sequence zero");
        }
    }
    require<Error>(
        descriptor.data_file_size >= kPartitionedHeaderSize
            && descriptor.ids_file_size >= kPartitionedHeaderSize,
        "segment file size is smaller than its header");
}

template <typename Error>
void validate_snapshot_descriptor(
    const PartitionSnapshotDescriptor& descriptor,
    const std::uint32_t partition_bits) {
    const std::uint64_t partition_count =
        validate_partition_bits<Error>(partition_bits);
    require<Error>(!descriptor.snapshot_id.empty(), "snapshot UUID is empty");
    require<Error>(
        descriptor.partition_id < partition_count,
        "snapshot partition ID is outside the layout");
    require<Error>(
        descriptor.capacity >= 8
            && (descriptor.capacity & (descriptor.capacity - 1)) == 0,
        "snapshot capacity is not a power of two of at least eight");
    require<Error>(
        descriptor.entry_count
            <= (descriptor.capacity / 10) * 7
                + ((descriptor.capacity % 10) * 7) / 10,
        "snapshot exceeds the maximum load factor");
    const std::uint32_t local_bits = static_cast<std::uint32_t>(
        std::bit_width(descriptor.capacity) - 1);
    require<Error>(
        local_bits <= 64 - partition_bits,
        "snapshot local hash bits overlap partition bits");
    require<Error>(
        descriptor.file_size >= kPartitionedHeaderSize,
        "snapshot file size is smaller than its header");
}

[[nodiscard]] bool bitmap_has_any(const std::span<const std::byte> bitmap) {
    return std::any_of(bitmap.begin(), bitmap.end(), [](const std::byte value) {
        return value != std::byte{0};
    });
}

[[nodiscard]] bool bitmap_contains(
    const std::span<const std::byte> bitmap,
    const std::uint64_t partition) {
    const std::size_t byte_index = static_cast<std::size_t>(partition / 8);
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    return (std::to_integer<std::uint8_t>(bitmap[byte_index]) & mask) != 0;
}

template <typename Error>
void validate_bitmap(
    const std::span<const std::byte> bitmap,
    const std::uint32_t partition_bits) {
    const std::uint64_t partition_count =
        validate_partition_bits<Error>(partition_bits);
    const std::uint64_t expected = (partition_count + 7) / 8;
    require<Error>(
        expected <= std::numeric_limits<std::size_t>::max()
            && bitmap.size() == static_cast<std::size_t>(expected),
        "run active-partition bitmap has invalid size");
    const std::uint32_t used_tail_bits =
        static_cast<std::uint32_t>(partition_count % 8);
    if (used_tail_bits != 0) {
        const std::uint8_t allowed = static_cast<std::uint8_t>(
            (std::uint32_t{1} << used_tail_bits) - 1);
        require<Error>(
            (std::to_integer<std::uint8_t>(bitmap.back()) & ~allowed) == 0,
            "run active-partition bitmap has nonzero padding bits");
    }
    require<Error>(bitmap_has_any(bitmap), "run has no active partitions");
}

template <typename Error>
void validate_run_descriptor(
    const PartitionIndexRunDescriptor& descriptor,
    const std::uint32_t partition_bits) {
    require<Error>(!descriptor.run_id.empty(), "index run UUID is empty");
    require<Error>(descriptor.epoch != 0, "index run epoch is zero");
    validate_run_kind<Error>(descriptor.kind);
    validate_sequence_range<Error>(
        descriptor.first_sequence, descriptor.last_sequence,
        "index run descriptor");
    if (descriptor.kind != PartitionedRunKind::Cleaner) {
        require<Error>(
            descriptor.first_sequence != 0,
            "logical mutation run uses initial sequence zero");
    }
    require<Error>(descriptor.entry_count != 0, "index run entry count is zero");
    require<Error>(
        descriptor.file_size >= kPartitionedHeaderSize,
        "index run file size is smaller than its header");
    validate_bitmap<Error>(descriptor.covered_partitions, partition_bits);
    validate_bitmap<Error>(descriptor.active_partitions, partition_bits);
    require<Error>(
        std::equal(
            descriptor.active_partitions.begin(),
            descriptor.active_partitions.end(),
            descriptor.covered_partitions.begin(),
            [](const std::byte active, const std::byte covered) {
                return (active & ~covered) == std::byte{0};
            }),
        "run active partitions are not a subset of covered partitions");
    if (descriptor.kind == PartitionedRunKind::Cleaner) {
        validate_segment_id<Error>(
            descriptor.retired_row_ref_segment_id,
            "cleaner run retired segment ID");
        require<Error>(
            !descriptor.retired_segment_id.empty(),
            "cleaner run retired segment UUID is empty");
        require<Error>(
            descriptor.retired_row_count != 0
                && descriptor.retired_row_count
                       <= RowRef::kMaximumRowsPerSegment,
            "cleaner run retired row count is invalid");
        require<Error>(
            descriptor.entry_count == descriptor.retired_row_count,
            "cleaner run does not cover every retired source row");
    } else {
        require<Error>(
            descriptor.retired_row_ref_segment_id == 0
                && descriptor.retired_segment_id.empty()
                && descriptor.retired_row_count == 0
                && descriptor.retired_ids_payload_crc32c == 0,
            "non-cleaner run contains retirement metadata");
    }
}

template <typename Error>
void validate_run_header_descriptor(
    const PartitionIndexRunDescriptor& descriptor,
    const std::uint32_t partition_bits) {
    require<Error>(
        descriptor.covered_partitions.empty()
            && descriptor.active_partitions.empty(),
        "partition run file header must not contain catalog-only bitmaps");
    PartitionIndexRunDescriptor scalar = descriptor;
    const std::uint64_t partition_count =
        validate_partition_bits<Error>(partition_bits);
    const std::uint64_t bitmap_size = (partition_count + 7) / 8;
    require<Error>(
        bitmap_size <= std::numeric_limits<std::size_t>::max(),
        "partition run bitmap does not fit memory");
    scalar.covered_partitions.assign(
        static_cast<std::size_t>(bitmap_size), std::byte{0xff});
    scalar.active_partitions = scalar.covered_partitions;
    if (partition_count % 8 != 0) {
        const std::byte tail = static_cast<std::byte>(
            (std::uint32_t{1} << (partition_count % 8)) - 1);
        scalar.covered_partitions.back() = tail;
        scalar.active_partitions.back() = tail;
    }
    validate_run_descriptor<Error>(scalar, partition_bits);
}

struct UuidHash {
    [[nodiscard]] std::size_t operator()(const Uuid128 value) const noexcept {
        std::uint64_t mixed = value.low
                              ^ (value.high + 0x9e3779b97f4a7c15ULL
                                 + (value.low << 6) + (value.low >> 2));
        mixed ^= mixed >> 30;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27;
        mixed *= 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31;
        return static_cast<std::size_t>(mixed);
    }
};

template <typename Error>
void insert_unique_uuid(
    std::unordered_set<Uuid128, UuidHash>& values,
    const Uuid128 value,
    const std::string_view description) {
    require<Error>(!value.empty(), std::string(description) + " UUID is empty");
    require<Error>(
        values.insert(value).second,
        std::string(description) + " UUID is duplicated or aliases a resource");
}

template <typename Error>
void validate_manifest(const PartitionedManifestV2& manifest) {
    require<Error>(!manifest.collection_id.empty(), "manifest collection UUID is empty");
    require<Error>(!manifest.schema_id.empty(), "manifest schema UUID is empty");
    require<Error>(
        !manifest.initial_active_wal_id.empty(),
        "manifest initial WAL UUID is empty");
    validate_schema<Error>(manifest.schema, manifest.record_size);
    const std::uint64_t partition_count =
        validate_partition_bits<Error>(manifest.partition_bits);
    require<Error>(
        manifest.partition_count == partition_count,
        "manifest partition count does not match partition bits");
    require<Error>(
        manifest.target_segment_size >= manifest.record_size,
        "manifest target segment cannot fit one record");
    validate_segment_id<Error>(
        manifest.initial_active_wal_segment_id,
        "manifest initial active WAL segment ID");
    validate_segment_id<Error>(
        manifest.initial_next_segment_id,
        "manifest initial next segment ID");
    require<Error>(
        manifest.initial_next_segment_id
            == manifest.initial_active_wal_segment_id + 1,
        "manifest initial next segment ID is not contiguous");
    require<Error>(
        (manifest.required_feature_bits
         & ~kPartitionedSupportedRequiredFeatureBits) == 0,
        "manifest requires an unknown format-7 feature");
    require<Error>(
        (manifest.required_feature_bits & kPartitionedRequiredFeatureBits)
            == kPartitionedRequiredFeatureBits,
        "manifest lacks a required format-7 feature");
    require<Error>(
        (manifest.required_feature_bits & kNativeRequiredFeatureDeltaV1) == 0
            || (manifest.required_feature_bits
                & kNativeRequiredFeatureCatalogV1) != 0,
        "manifest requires DeltaV1 without a catalog");
}

template <typename Error>
void validate_wal_chain(
    const PartitionedCatalogV3& catalog,
    std::unordered_set<Uuid128, UuidHash>& resource_uuids,
    std::unordered_set<std::uint32_t>& segment_ids,
    std::uint32_t& maximum_segment_id) {
    require<Error>(!catalog.wal_entries.empty(), "catalog WAL chain is empty");
    Uuid128 expected_predecessor = catalog.terminal_wal_id;
    std::uint64_t previous_sequence = catalog.indexed_sequence;
    bool saw_active = false;
    for (std::size_t index = 0; index != catalog.wal_entries.size(); ++index) {
        const MutationWalEntry& entry = catalog.wal_entries[index];
        insert_unique_uuid<Error>(resource_uuids, entry.wal_id, "WAL");
        require<Error>(
            segment_ids.insert(entry.segment_id).second,
            "WAL segment ID is duplicated or aliases a segment");
        validate_segment_id<Error>(entry.segment_id, "WAL segment ID");
        maximum_segment_id = std::max(maximum_segment_id, entry.segment_id);
        require<Error>(
            entry.predecessor_wal_id == expected_predecessor,
            "catalog WAL predecessor chain is discontinuous");
        require<Error>(
            entry.previous_sequence == previous_sequence
                && previous_sequence < kMaximumSequence
                && entry.first_sequence == previous_sequence + 1,
            "catalog WAL sequence chain is discontinuous");
        if (entry.state == MutationWalState::Frozen) {
            require<Error>(!saw_active, "frozen WAL follows active WAL");
            require<Error>(
                entry.last_sequence >= entry.first_sequence
                    && entry.last_sequence <= kMaximumSequence,
                "frozen WAL sequence range is invalid");
            require<Error>(
                entry.put_count <= RowRef::kMaximumRowsPerSegment,
                "frozen WAL PUT count exceeds RowRef capacity");
            require<Error>(
                entry.sealed_file_size >= kPartitionedHeaderSize,
                "frozen WAL file size is invalid");
            previous_sequence = entry.last_sequence;
        } else {
            require<Error>(
                entry.state == MutationWalState::Active,
                "catalog WAL has unknown state");
            require<Error>(
                index + 1 == catalog.wal_entries.size() && !saw_active,
                "active WAL is not the unique final WAL");
            require<Error>(
                entry.last_sequence == 0 && entry.put_count == 0
                    && entry.sealed_file_size == 0,
                "active WAL contains frozen-only metadata");
            saw_active = true;
        }
        expected_predecessor = entry.wal_id;
    }
    require<Error>(saw_active, "catalog lacks an active WAL");
}

template <typename Error>
void validate_catalog(const PartitionedCatalogV3& catalog) {
    require<Error>(!catalog.collection_id.empty(), "catalog collection UUID is empty");
    require<Error>(!catalog.schema_id.empty(), "catalog schema UUID is empty");
    require<Error>(catalog.generation != 0, "catalog generation is zero");
    require<Error>(!catalog.catalog_id.empty(), "catalog UUID is empty");
    require<Error>(
        catalog.generation == 1 ? catalog.previous_catalog_id.empty()
                                : !catalog.previous_catalog_id.empty(),
        "catalog predecessor does not match generation");
    require<Error>(
        catalog.indexed_sequence <= kMaximumSequence,
        "catalog indexed sequence is reserved");
    require<Error>(
        (catalog.indexed_sequence == 0) == catalog.terminal_wal_id.empty(),
        "catalog indexed sequence and terminal WAL lineage disagree");
    const std::uint64_t partition_count =
        validate_partition_bits<Error>(catalog.partition_bits);
    require<Error>(
        catalog.segments.size() <= kPartitionedMaximumResourceEntries,
        "catalog has too many segments");
    require<Error>(
        catalog.snapshots.size() == partition_count,
        "catalog does not contain exactly one snapshot per partition");
    require<Error>(
        catalog.runs.size() <= kPartitionedMaximumRuns,
        "catalog has too many index runs");
    require<Error>(
        catalog.wal_entries.size() <= kPartitionedMaximumResourceEntries,
        "catalog has too many WALs");

    std::unordered_set<Uuid128, UuidHash> resource_uuids;
    resource_uuids.reserve(
        catalog.segments.size() + catalog.snapshots.size()
        + catalog.runs.size() + catalog.wal_entries.size());
    std::unordered_set<std::uint32_t> segment_ids;
    segment_ids.reserve(catalog.segments.size() + catalog.wal_entries.size());
    std::uint32_t maximum_segment_id = 0;
    std::uint64_t physical_rows = 0;
    for (const PartitionedSegmentDescriptor& descriptor : catalog.segments) {
        validate_segment_descriptor<Error>(descriptor);
        insert_unique_uuid<Error>(resource_uuids, descriptor.segment_id, "segment");
        require<Error>(
            segment_ids.insert(descriptor.row_ref_segment_id).second,
            "segment ID is duplicated");
        maximum_segment_id =
            std::max(maximum_segment_id, descriptor.row_ref_segment_id);
        physical_rows = checked_sum<Error>(
            physical_rows, descriptor.row_count, "catalog physical rows");
        require<Error>(
            descriptor.maximum_sequence <= catalog.indexed_sequence,
            "segment sequence exceeds catalog indexed sequence");
    }
    require<Error>(
        catalog.indexed_live_count <= physical_rows,
        "catalog indexed live count exceeds physical rows");

    std::uint64_t maximum_epoch = 0;
    for (std::size_t index = 0; index != catalog.snapshots.size(); ++index) {
        const PartitionSnapshotDescriptor& descriptor = catalog.snapshots[index];
        validate_snapshot_descriptor<Error>(descriptor, catalog.partition_bits);
        require<Error>(
            descriptor.partition_id == index,
            "catalog snapshots are not in exact partition order");
        insert_unique_uuid<Error>(
            resource_uuids, descriptor.snapshot_id, "partition snapshot");
        maximum_epoch = std::max(maximum_epoch, descriptor.epoch);
    }
    std::uint64_t previous_epoch = 0;
    std::unordered_set<Uuid128, UuidHash> retired_uuids;
    std::unordered_set<std::uint32_t> retired_segment_ids;
    for (const PartitionIndexRunDescriptor& descriptor : catalog.runs) {
        validate_run_descriptor<Error>(descriptor, catalog.partition_bits);
        require<Error>(
            descriptor.epoch > previous_epoch,
            "catalog index run epochs are not strictly increasing");
        require<Error>(
            descriptor.last_sequence <= catalog.indexed_sequence,
            "index run sequence exceeds catalog indexed sequence");
        insert_unique_uuid<Error>(resource_uuids, descriptor.run_id, "index run");
        if (descriptor.kind == PartitionedRunKind::Cleaner) {
            require<Error>(
                resource_uuids.find(descriptor.retired_segment_id)
                    == resource_uuids.end()
                    && segment_ids.find(descriptor.retired_row_ref_segment_id)
                           == segment_ids.end(),
                "cleaner run retires a still-active segment");
            require<Error>(
                retired_uuids.insert(descriptor.retired_segment_id).second
                    && retired_segment_ids
                           .insert(descriptor.retired_row_ref_segment_id)
                           .second,
                "retired segment has more than one cleaner fence");
        }
        for (std::uint64_t partition = 0;
             partition != partition_count; ++partition) {
            if (bitmap_contains(descriptor.covered_partitions, partition)
                && !bitmap_contains(descriptor.active_partitions, partition)) {
                require<Error>(
                    catalog.snapshots[static_cast<std::size_t>(partition)].epoch
                        >= descriptor.epoch,
                    "folded run partition lacks a covering snapshot epoch");
            }
        }
        previous_epoch = descriptor.epoch;
        maximum_epoch = std::max(maximum_epoch, descriptor.epoch);
    }
    require<Error>(
        catalog.next_index_epoch > maximum_epoch,
        "catalog next index epoch is not above every published epoch");

    validate_wal_chain<Error>(
        catalog, resource_uuids, segment_ids, maximum_segment_id);
    for (const Uuid128 retired : retired_uuids) {
        require<Error>(
            resource_uuids.find(retired) == resource_uuids.end(),
            "cleaner run retired UUID aliases an active resource");
    }
    for (const std::uint32_t retired : retired_segment_ids) {
        require<Error>(
            segment_ids.find(retired) == segment_ids.end(),
            "cleaner run retired segment ID aliases an active resource");
    }
    validate_segment_id<Error>(catalog.next_segment_id, "catalog next segment ID");
    require<Error>(
        catalog.next_segment_id > maximum_segment_id,
        "catalog next segment ID is not above every published segment");
}

void put_segment_descriptor(
    const std::span<std::byte> bytes,
    const PartitionedSegmentDescriptor& descriptor) {
    put_uuid(bytes, 0, descriptor.segment_id);
    put_u32(bytes, 16, descriptor.row_ref_segment_id);
    put_u32(bytes, 20, static_cast<std::uint32_t>(descriptor.kind));
    put_u64(bytes, 24, descriptor.row_count);
    put_u64(bytes, 32, descriptor.minimum_sequence);
    put_u64(bytes, 40, descriptor.maximum_sequence);
    put_u64(bytes, 48, descriptor.data_file_size);
    put_u64(bytes, 56, descriptor.ids_file_size);
    put_u32(bytes, 64, descriptor.data_payload_crc32c);
    put_u32(bytes, 68, descriptor.ids_payload_crc32c);
    put_u64(bytes, 72, descriptor.creation_epoch);
}

[[nodiscard]] PartitionedSegmentDescriptor get_segment_descriptor(
    const std::span<const std::byte> bytes) {
    PartitionedSegmentDescriptor descriptor;
    descriptor.segment_id = get_uuid(bytes, 0);
    descriptor.row_ref_segment_id = get_u32(bytes, 16);
    descriptor.kind = static_cast<PartitionedSegmentKind>(get_u32(bytes, 20));
    descriptor.row_count = get_u64(bytes, 24);
    descriptor.minimum_sequence = get_u64(bytes, 32);
    descriptor.maximum_sequence = get_u64(bytes, 40);
    descriptor.data_file_size = get_u64(bytes, 48);
    descriptor.ids_file_size = get_u64(bytes, 56);
    descriptor.data_payload_crc32c = get_u32(bytes, 64);
    descriptor.ids_payload_crc32c = get_u32(bytes, 68);
    descriptor.creation_epoch = get_u64(bytes, 72);
    return descriptor;
}

void put_snapshot_descriptor(
    const std::span<std::byte> bytes,
    const PartitionSnapshotDescriptor& descriptor) {
    put_uuid(bytes, 0, descriptor.snapshot_id);
    put_u32(bytes, 16, descriptor.partition_id);
    put_u32(bytes, 20, 0);
    put_u64(bytes, 24, descriptor.epoch);
    put_u64(bytes, 32, descriptor.entry_count);
    put_u64(bytes, 40, descriptor.capacity);
    put_u64(bytes, 48, descriptor.file_size);
    put_u32(bytes, 56, descriptor.body_crc32c);
    put_u32(bytes, 60, 0);
}

[[nodiscard]] PartitionSnapshotDescriptor get_snapshot_descriptor(
    const std::span<const std::byte> bytes) {
    PartitionSnapshotDescriptor descriptor;
    descriptor.snapshot_id = get_uuid(bytes, 0);
    descriptor.partition_id = get_u32(bytes, 16);
    descriptor.epoch = get_u64(bytes, 24);
    descriptor.entry_count = get_u64(bytes, 32);
    descriptor.capacity = get_u64(bytes, 40);
    descriptor.file_size = get_u64(bytes, 48);
    descriptor.body_crc32c = get_u32(bytes, 56);
    return descriptor;
}

void put_wal_entry(
    const std::span<std::byte> bytes,
    const MutationWalEntry& entry) {
    put_uuid(bytes, 0, entry.wal_id);
    put_uuid(bytes, 16, entry.predecessor_wal_id);
    put_u32(bytes, 32, entry.segment_id);
    put_u32(bytes, 36, static_cast<std::uint32_t>(entry.state));
    put_u64(bytes, 40, entry.previous_sequence);
    put_u64(bytes, 48, entry.first_sequence);
    put_u64(bytes, 56, entry.last_sequence);
    put_u64(bytes, 64, entry.put_count);
    put_u64(bytes, 72, entry.sealed_file_size);
}

[[nodiscard]] MutationWalEntry get_wal_entry(
    const std::span<const std::byte> bytes) {
    MutationWalEntry entry;
    entry.wal_id = get_uuid(bytes, 0);
    entry.predecessor_wal_id = get_uuid(bytes, 16);
    entry.segment_id = get_u32(bytes, 32);
    entry.state = static_cast<MutationWalState>(get_u32(bytes, 36));
    entry.previous_sequence = get_u64(bytes, 40);
    entry.first_sequence = get_u64(bytes, 48);
    entry.last_sequence = get_u64(bytes, 56);
    entry.put_count = get_u64(bytes, 64);
    entry.sealed_file_size = get_u64(bytes, 72);
    return entry;
}

}  // namespace

std::uint64_t partition_bitmap_size(const std::uint32_t partition_bits) {
    const std::uint64_t count =
        validate_partition_bits<InvalidArgumentError>(partition_bits);
    return (count + 7) / 8;
}

std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_manifest_v2(const PartitionedManifestV2& manifest) {
    validate_manifest<InvalidArgumentError>(manifest);
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedManifestMagic, kPartitionedManifestWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, manifest.collection_id);
    put_uuid(bytes, 40, manifest.schema_id);
    put_uuid(bytes, 56, manifest.initial_active_wal_id);
    put_u32(bytes, 72, static_cast<std::uint32_t>(manifest.schema.dtype));
    put_u32(bytes, 76, manifest.schema.dimension);
    put_u32(bytes, 80, manifest.schema.item_size);
    put_u32(bytes, 84, static_cast<std::uint32_t>(manifest.schema.id_type));
    put_u32(bytes, 88, static_cast<std::uint32_t>(manifest.schema.row_layout));
    put_u32(bytes, 92, static_cast<std::uint32_t>(manifest.schema.index_kind));
    put_u32(bytes, 96, static_cast<std::uint32_t>(manifest.schema.hash_algorithm));
    put_u32(bytes, 100, manifest.partition_bits);
    put_u64(bytes, 104, manifest.record_size);
    put_u64(bytes, 112, manifest.partition_count);
    put_u64(bytes, 120, manifest.hash_seed);
    put_u64(bytes, 128, manifest.target_segment_size);
    put_u64(bytes, 136, manifest.initial_record_count);
    put_u64(bytes, 144, manifest.required_feature_bits);
    put_u64(bytes, 152, manifest.optional_feature_bits);
    put_u32(bytes, 160, manifest.initial_active_wal_segment_id);
    put_u32(bytes, 164, manifest.initial_next_segment_id);
    put_u64(bytes, 168, manifest.creation_time_unix_ns);
    finish_checksum(bytes);
    return bytes;
}

PartitionedManifestV2 decode_partitioned_manifest_v2(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "partitioned manifest is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedManifestMagic, kPartitionedManifestWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "partitioned manifest");
    require_zero_range<CorruptionError>(
        bytes, kManifestKnownBytes, bytes.size(), "partitioned manifest");
    PartitionedManifestV2 manifest;
    manifest.collection_id = get_uuid(bytes, 24);
    manifest.schema_id = get_uuid(bytes, 40);
    manifest.initial_active_wal_id = get_uuid(bytes, 56);
    manifest.schema.dtype = static_cast<DTypeCode>(get_u32(bytes, 72));
    manifest.schema.dimension = get_u32(bytes, 76);
    manifest.schema.item_size = get_u32(bytes, 80);
    manifest.schema.id_type = static_cast<IdType>(get_u32(bytes, 84));
    manifest.schema.row_layout = static_cast<RowLayout>(get_u32(bytes, 88));
    manifest.schema.index_kind = static_cast<IndexKind>(get_u32(bytes, 92));
    manifest.schema.hash_algorithm =
        static_cast<HashAlgorithm>(get_u32(bytes, 96));
    manifest.partition_bits = get_u32(bytes, 100);
    manifest.record_size = get_u64(bytes, 104);
    manifest.partition_count = get_u64(bytes, 112);
    manifest.hash_seed = get_u64(bytes, 120);
    manifest.target_segment_size = get_u64(bytes, 128);
    manifest.initial_record_count = get_u64(bytes, 136);
    manifest.required_feature_bits = get_u64(bytes, 144);
    manifest.optional_feature_bits = get_u64(bytes, 152);
    manifest.initial_active_wal_segment_id = get_u32(bytes, 160);
    manifest.initial_next_segment_id = get_u32(bytes, 164);
    manifest.creation_time_unix_ns = get_u64(bytes, 168);
    validate_manifest<CorruptionError>(manifest);
    return manifest;
}

std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_segment_data_header_v1(
    const PartitionedSegmentDataHeaderV1& header) {
    validate_segment_descriptor<InvalidArgumentError>(header.descriptor);
    require<InvalidArgumentError>(
        !header.collection_id.empty() && !header.schema_id.empty(),
        "segment data identity is empty");
    validate_schema<InvalidArgumentError>(header.schema, header.record_size);
    const std::uint64_t expected_payload = checked_product<InvalidArgumentError>(
        header.descriptor.row_count, header.record_size,
        "segment data payload");
    require<InvalidArgumentError>(
        header.payload_offset == kPartitionedHeaderSize
            && header.payload_size == expected_payload
            && header.descriptor.data_file_size
                   == checked_sum<InvalidArgumentError>(
                       header.payload_offset, header.payload_size,
                       "segment data file size"),
        "segment data geometry does not match descriptor");
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedSegmentDataMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, header.collection_id);
    put_uuid(bytes, 40, header.schema_id);
    put_segment_descriptor(std::span<std::byte>(bytes).subspan(56, 80),
                           header.descriptor);
    put_u32(bytes, 136, static_cast<std::uint32_t>(header.schema.dtype));
    put_u32(bytes, 140, header.schema.dimension);
    put_u32(bytes, 144, header.schema.item_size);
    put_u32(bytes, 148, static_cast<std::uint32_t>(header.schema.id_type));
    put_u32(bytes, 152, static_cast<std::uint32_t>(header.schema.row_layout));
    put_u32(bytes, 156, static_cast<std::uint32_t>(header.schema.hash_algorithm));
    put_u64(bytes, 160, header.record_size);
    put_u64(bytes, 168, header.payload_offset);
    put_u64(bytes, 176, header.payload_size);
    finish_checksum(bytes);
    return bytes;
}

PartitionedSegmentDataHeaderV1 decode_partitioned_segment_data_header_v1(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "segment data header is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedSegmentDataMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "segment data header");
    require_zero_range<CorruptionError>(
        bytes, kSegmentDataKnownBytes, bytes.size(), "segment data header");
    PartitionedSegmentDataHeaderV1 header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor = get_segment_descriptor(bytes.subspan(56, 80));
    header.schema.dtype = static_cast<DTypeCode>(get_u32(bytes, 136));
    header.schema.dimension = get_u32(bytes, 140);
    header.schema.item_size = get_u32(bytes, 144);
    header.schema.id_type = static_cast<IdType>(get_u32(bytes, 148));
    header.schema.row_layout = static_cast<RowLayout>(get_u32(bytes, 152));
    header.schema.index_kind = IndexKind::UInt64PartitionedRowRef;
    header.schema.hash_algorithm =
        static_cast<HashAlgorithm>(get_u32(bytes, 156));
    header.record_size = get_u64(bytes, 160);
    header.payload_offset = get_u64(bytes, 168);
    header.payload_size = get_u64(bytes, 176);
    // Reuse the encoder's complete semantic validation after the wire-level
    // checksum/reserved-byte checks, translating caller errors to corruption.
    try {
        static_cast<void>(encode_partitioned_segment_data_header_v1(header));
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
    return header;
}

std::array<std::byte, kPartitionedHeaderSize>
encode_partitioned_segment_ids_header_v1(
    const PartitionedSegmentIdsHeaderV1& header) {
    validate_segment_descriptor<InvalidArgumentError>(header.descriptor);
    require<InvalidArgumentError>(
        !header.collection_id.empty() && !header.schema_id.empty(),
        "segment IDs identity is empty");
    require<InvalidArgumentError>(
        header.id_type == IdType::UInt64
            && header.entry_size == kPartitionedSegmentIdEntrySize,
        "segment IDs encoding is unsupported");
    const std::uint64_t expected_payload = checked_product<InvalidArgumentError>(
        header.descriptor.row_count, header.entry_size,
        "segment IDs payload");
    require<InvalidArgumentError>(
        header.payload_offset == kPartitionedHeaderSize
            && header.payload_size == expected_payload
            && header.descriptor.ids_file_size
                   == checked_sum<InvalidArgumentError>(
                       header.payload_offset, header.payload_size,
                       "segment IDs file size"),
        "segment IDs geometry does not match descriptor");
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedSegmentIdsMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, header.collection_id);
    put_uuid(bytes, 40, header.schema_id);
    put_segment_descriptor(std::span<std::byte>(bytes).subspan(56, 80),
                           header.descriptor);
    put_u32(bytes, 136, static_cast<std::uint32_t>(header.id_type));
    put_u32(bytes, 140, 0);
    put_u64(bytes, 144, header.entry_size);
    put_u64(bytes, 152, header.payload_offset);
    put_u64(bytes, 160, header.payload_size);
    finish_checksum(bytes);
    return bytes;
}

PartitionedSegmentIdsHeaderV1 decode_partitioned_segment_ids_header_v1(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "segment IDs header is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedSegmentIdsMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "segment IDs header");
    require_zero_range<CorruptionError>(
        bytes, kSegmentIdsKnownBytes, bytes.size(), "segment IDs header");
    require<CorruptionError>(
        get_u32(bytes, 140) == 0,
        "segment IDs header has unsupported flags");
    PartitionedSegmentIdsHeaderV1 header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor = get_segment_descriptor(bytes.subspan(56, 80));
    header.id_type = static_cast<IdType>(get_u32(bytes, 136));
    header.entry_size = get_u64(bytes, 144);
    header.payload_offset = get_u64(bytes, 152);
    header.payload_size = get_u64(bytes, 160);
    try {
        static_cast<void>(encode_partitioned_segment_ids_header_v1(header));
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
    return header;
}

std::array<std::byte, kPartitionedHeaderSize>
encode_partition_snapshot_header_v1(
    const PartitionSnapshotHeaderV1& header) {
    require<InvalidArgumentError>(
        !header.collection_id.empty() && !header.schema_id.empty(),
        "partition snapshot identity is empty");
    validate_snapshot_descriptor<InvalidArgumentError>(
        header.descriptor, header.partition_bits);
    const std::uint64_t controls_end = checked_sum<InvalidArgumentError>(
        kPartitionedHeaderSize, header.descriptor.capacity,
        "partition snapshot controls");
    const std::uint64_t expected_lookup_offset = checked_align<InvalidArgumentError>(
        controls_end, kPartitionedHeaderSize,
        "partition snapshot lookup slots");
    const std::uint64_t lookup_end = checked_sum<InvalidArgumentError>(
        expected_lookup_offset,
        checked_product<InvalidArgumentError>(
            header.descriptor.capacity, kPartitionSnapshotLookupSlotSize,
            "partition snapshot lookup bytes"),
        "partition snapshot lookup end");
    const std::uint64_t expected_sequence_offset =
        checked_align<InvalidArgumentError>(
            lookup_end, kPartitionedHeaderSize,
            "partition snapshot sequence slots");
    const std::uint64_t expected_file_size = checked_sum<InvalidArgumentError>(
        expected_sequence_offset,
        checked_product<InvalidArgumentError>(
            header.descriptor.capacity, kPartitionSnapshotSequenceSlotSize,
            "partition snapshot sequence bytes"),
        "partition snapshot file size");
    require<InvalidArgumentError>(
        header.controls_offset == kPartitionedHeaderSize
            && header.lookup_slots_offset == expected_lookup_offset
            && header.sequence_slots_offset == expected_sequence_offset
            && header.lookup_slot_size == kPartitionSnapshotLookupSlotSize
            && header.sequence_slot_size == kPartitionSnapshotSequenceSlotSize
            && header.descriptor.file_size == expected_file_size,
        "partition snapshot geometry is invalid");

    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedSnapshotMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, header.collection_id);
    put_uuid(bytes, 40, header.schema_id);
    put_uuid(bytes, 56, header.descriptor.snapshot_id);
    put_u32(bytes, 72, header.descriptor.partition_id);
    put_u32(bytes, 76, header.partition_bits);
    put_u64(bytes, 80, header.descriptor.epoch);
    put_u64(bytes, 88, header.descriptor.entry_count);
    put_u64(bytes, 96, header.descriptor.capacity);
    put_u64(bytes, 104, header.descriptor.file_size);
    put_u32(bytes, 112, header.descriptor.body_crc32c);
    put_u32(bytes, 116, 0);
    put_u64(bytes, 120, header.hash_seed);
    put_u64(bytes, 128, header.controls_offset);
    put_u64(bytes, 136, header.lookup_slots_offset);
    put_u64(bytes, 144, header.sequence_slots_offset);
    put_u64(bytes, 152, header.lookup_slot_size);
    put_u64(bytes, 160, header.sequence_slot_size);
    finish_checksum(bytes);
    return bytes;
}

PartitionSnapshotHeaderV1 decode_partition_snapshot_header_v1(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "partition snapshot header is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedSnapshotMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "partition snapshot header");
    require_zero_range<CorruptionError>(
        bytes, kSnapshotKnownBytes, bytes.size(), "partition snapshot header");
    require<CorruptionError>(
        get_u32(bytes, 116) == 0,
        "partition snapshot header has unsupported flags");
    PartitionSnapshotHeaderV1 header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor.snapshot_id = get_uuid(bytes, 56);
    header.descriptor.partition_id = get_u32(bytes, 72);
    header.partition_bits = get_u32(bytes, 76);
    header.descriptor.epoch = get_u64(bytes, 80);
    header.descriptor.entry_count = get_u64(bytes, 88);
    header.descriptor.capacity = get_u64(bytes, 96);
    header.descriptor.file_size = get_u64(bytes, 104);
    header.descriptor.body_crc32c = get_u32(bytes, 112);
    header.hash_seed = get_u64(bytes, 120);
    header.controls_offset = get_u64(bytes, 128);
    header.lookup_slots_offset = get_u64(bytes, 136);
    header.sequence_slots_offset = get_u64(bytes, 144);
    header.lookup_slot_size = get_u64(bytes, 152);
    header.sequence_slot_size = get_u64(bytes, 160);
    try {
        static_cast<void>(encode_partition_snapshot_header_v1(header));
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
    return header;
}

std::array<std::byte, kPartitionedHeaderSize>
encode_partition_index_run_header_v1(
    const PartitionIndexRunHeaderV1& header) {
    require<InvalidArgumentError>(
        !header.collection_id.empty() && !header.schema_id.empty(),
        "partition run identity is empty");
    validate_run_header_descriptor<InvalidArgumentError>(
        header.descriptor, header.partition_bits);
    const std::uint64_t partition_count =
        validate_partition_bits<InvalidArgumentError>(header.partition_bits);
    const std::uint64_t expected_toc_size = checked_product<InvalidArgumentError>(
        partition_count, kPartitionRunTocEntrySize,
        "partition run TOC bytes");
    const std::uint64_t expected_records_offset = checked_align<InvalidArgumentError>(
        checked_sum<InvalidArgumentError>(
            kPartitionedHeaderSize, expected_toc_size,
            "partition run TOC end"),
        kPartitionedHeaderSize, "partition run records");
    const std::uint64_t expected_records_size = checked_product<InvalidArgumentError>(
        header.descriptor.entry_count, kPartitionRunRecordSize,
        "partition run record bytes");
    const std::uint64_t expected_file_size = checked_sum<InvalidArgumentError>(
        expected_records_offset, expected_records_size,
        "partition run file size");
    require<InvalidArgumentError>(
        header.toc_offset == kPartitionedHeaderSize
            && header.toc_size == expected_toc_size
            && header.records_offset == expected_records_offset
            && header.records_size == expected_records_size
            && header.toc_entry_size == kPartitionRunTocEntrySize
            && header.record_size == kPartitionRunRecordSize
            && header.descriptor.file_size == expected_file_size,
        "partition run geometry is invalid");

    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedRunMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, header.collection_id);
    put_uuid(bytes, 40, header.schema_id);
    put_uuid(bytes, 56, header.descriptor.run_id);
    put_u64(bytes, 72, header.descriptor.epoch);
    put_u32(bytes, 80, static_cast<std::uint32_t>(header.descriptor.kind));
    put_u32(bytes, 84, header.partition_bits);
    put_u64(bytes, 88, header.descriptor.first_sequence);
    put_u64(bytes, 96, header.descriptor.last_sequence);
    put_u64(bytes, 104, header.descriptor.entry_count);
    put_u64(bytes, 112, header.descriptor.file_size);
    put_u32(bytes, 120, header.descriptor.body_crc32c);
    put_u32(bytes, 124, 0);
    put_u64(bytes, 128, header.hash_seed);
    put_u64(bytes, 136, header.toc_offset);
    put_u64(bytes, 144, header.toc_size);
    put_u64(bytes, 152, header.records_offset);
    put_u64(bytes, 160, header.records_size);
    put_u64(bytes, 168, header.toc_entry_size);
    put_u64(bytes, 176, header.record_size);
    put_u32(bytes, 184, header.descriptor.retired_row_ref_segment_id);
    put_u32(bytes, 188, 0);
    put_uuid(bytes, 192, header.descriptor.retired_segment_id);
    put_u64(bytes, 208, header.descriptor.retired_row_count);
    put_u32(bytes, 216, header.descriptor.retired_ids_payload_crc32c);
    put_u32(bytes, 220, 0);
    finish_checksum(bytes);
    return bytes;
}

PartitionIndexRunHeaderV1 decode_partition_index_run_header_v1(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "partition run header is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedRunMagic, kPartitionedResourceWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "partition run header");
    require_zero_range<CorruptionError>(
        bytes, kRunKnownBytes, bytes.size(), "partition run header");
    require<CorruptionError>(
        get_u32(bytes, 124) == 0 && get_u32(bytes, 188) == 0
            && get_u32(bytes, 220) == 0,
        "partition run header has unsupported flags");
    PartitionIndexRunHeaderV1 header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor.run_id = get_uuid(bytes, 56);
    header.descriptor.epoch = get_u64(bytes, 72);
    header.descriptor.kind =
        static_cast<PartitionedRunKind>(get_u32(bytes, 80));
    header.partition_bits = get_u32(bytes, 84);
    header.descriptor.first_sequence = get_u64(bytes, 88);
    header.descriptor.last_sequence = get_u64(bytes, 96);
    header.descriptor.entry_count = get_u64(bytes, 104);
    header.descriptor.file_size = get_u64(bytes, 112);
    header.descriptor.body_crc32c = get_u32(bytes, 120);
    header.hash_seed = get_u64(bytes, 128);
    header.toc_offset = get_u64(bytes, 136);
    header.toc_size = get_u64(bytes, 144);
    header.records_offset = get_u64(bytes, 152);
    header.records_size = get_u64(bytes, 160);
    header.toc_entry_size = get_u64(bytes, 168);
    header.record_size = get_u64(bytes, 176);
    header.descriptor.retired_row_ref_segment_id = get_u32(bytes, 184);
    header.descriptor.retired_segment_id = get_uuid(bytes, 192);
    header.descriptor.retired_row_count = get_u64(bytes, 208);
    header.descriptor.retired_ids_payload_crc32c = get_u32(bytes, 216);
    try {
        static_cast<void>(encode_partition_index_run_header_v1(header));
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
    return header;
}

std::array<std::byte, kPartitionRunTocEntrySize>
encode_partition_run_toc_entry(const PartitionRunTocEntry& entry) {
    std::array<std::byte, kPartitionRunTocEntrySize> bytes{};
    put_u64(bytes, 0, entry.first_record);
    put_u64(bytes, 8, entry.record_count);
    return bytes;
}

PartitionRunTocEntry decode_partition_run_toc_entry(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionRunTocEntrySize,
        "partition run TOC entry has invalid size");
    return PartitionRunTocEntry{get_u64(bytes, 0), get_u64(bytes, 8)};
}

std::array<std::byte, kPartitionRunRecordSize>
encode_partition_run_record(const PartitionRunRecord& record) {
    if (record.sequence > kMaximumSequence) {
        throw InvalidArgumentError("partition run record uses reserved sequence");
    }
    if (!record.tombstone()) {
        // Persisted RowRef validation reports corruption. Translate it at the
        // construction boundary to a caller argument error.
        try {
            static_cast<void>(RowRef::decode(record.value_raw));
        } catch (const CorruptionError& error) {
            throw InvalidArgumentError(error.what());
        }
    }
    std::array<std::byte, kPartitionRunRecordSize> bytes{};
    put_u64(bytes, 0, record.id);
    put_u64(bytes, 8, record.sequence);
    put_u64(bytes, 16, record.value_raw);
    return bytes;
}

PartitionRunRecord decode_partition_run_record(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() == kPartitionRunRecordSize,
        "partition run record has invalid size");
    PartitionRunRecord record{
        get_u64(bytes, 0), get_u64(bytes, 8), get_u64(bytes, 16)};
    require<CorruptionError>(
        record.sequence <= kMaximumSequence,
        "partition run record uses reserved sequence");
    if (!record.tombstone()) {
        static_cast<void>(RowRef::decode(record.value_raw));
    }
    return record;
}

std::vector<std::byte> encode_partitioned_catalog_v3(
    const PartitionedCatalogV3& catalog) {
    validate_catalog<InvalidArgumentError>(catalog);
    const std::uint64_t segment_count = catalog.segments.size();
    const std::uint64_t snapshot_count = catalog.snapshots.size();
    const std::uint64_t run_count = catalog.runs.size();
    const std::uint64_t wal_count = catalog.wal_entries.size();
    const std::uint64_t segment_offset = kPartitionedHeaderSize;
    const std::uint64_t snapshot_offset = checked_sum<InvalidArgumentError>(
        segment_offset,
        checked_product<InvalidArgumentError>(
            segment_count, kCatalogSegmentEntrySize,
            "catalog segment entries"),
        "catalog snapshot offset");
    const std::uint64_t run_offset = checked_sum<InvalidArgumentError>(
        snapshot_offset,
        checked_product<InvalidArgumentError>(
            snapshot_count, kCatalogSnapshotEntrySize,
            "catalog snapshot entries"),
        "catalog run offset");
    const std::uint64_t wal_offset = checked_sum<InvalidArgumentError>(
        run_offset,
        checked_product<InvalidArgumentError>(
            run_count, kCatalogRunEntrySize,
            "catalog run entries"),
        "catalog WAL offset");
    const std::uint64_t bitmap_offset = checked_sum<InvalidArgumentError>(
        wal_offset,
        checked_product<InvalidArgumentError>(
            wal_count, kCatalogWalEntrySize,
            "catalog WAL entries"),
        "catalog bitmap offset");
    const std::uint64_t one_bitmap_size =
        partition_bitmap_size(catalog.partition_bits);
    const std::uint64_t bitmap_size = checked_product<InvalidArgumentError>(
        checked_product<InvalidArgumentError>(
            run_count, 2, "catalog run bitmap count"),
        one_bitmap_size, "catalog run bitmaps");
    const std::uint64_t file_size = checked_sum<InvalidArgumentError>(
        bitmap_offset, bitmap_size, "catalog file size");
    require<InvalidArgumentError>(
        file_size <= std::numeric_limits<std::size_t>::max(),
        "partitioned catalog does not fit address space");

    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    encode_common(
        bytes, kPartitionedCatalogMagic, kPartitionedCatalogWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, catalog.collection_id);
    put_uuid(bytes, 40, catalog.schema_id);
    put_uuid(bytes, 56, catalog.catalog_id);
    put_uuid(bytes, 72, catalog.previous_catalog_id);
    put_u64(bytes, 88, catalog.generation);
    put_u32(bytes, 96, catalog.next_segment_id);
    put_u32(bytes, 100, catalog.partition_bits);
    put_u64(bytes, 104, catalog.next_index_epoch);
    put_u64(bytes, 112, catalog.indexed_sequence);
    put_u64(bytes, 120, catalog.indexed_live_count);
    put_u64(bytes, 128, catalog.hash_seed);
    put_u64(bytes, 136, segment_count);
    put_u64(bytes, 144, segment_offset);
    put_u32(bytes, 152, static_cast<std::uint32_t>(kCatalogSegmentEntrySize));
    put_u32(bytes, 156, static_cast<std::uint32_t>(kCatalogSnapshotEntrySize));
    put_u64(bytes, 160, snapshot_count);
    put_u64(bytes, 168, snapshot_offset);
    put_u32(bytes, 176, static_cast<std::uint32_t>(kCatalogRunEntrySize));
    put_u32(bytes, 180, static_cast<std::uint32_t>(kCatalogWalEntrySize));
    put_u64(bytes, 184, run_count);
    put_u64(bytes, 192, run_offset);
    put_u64(bytes, 200, wal_count);
    put_u64(bytes, 208, wal_offset);
    put_u64(bytes, 216, bitmap_offset);
    put_u64(bytes, 224, bitmap_size);
    put_u64(bytes, 232, file_size);
    put_uuid(bytes, 240, catalog.terminal_wal_id);
    put_u64(bytes, 256, 0);

    for (std::size_t index = 0; index != catalog.segments.size(); ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            segment_offset + index * kCatalogSegmentEntrySize);
        const std::span<std::byte> encoded = std::span<std::byte>(bytes).subspan(
            offset, static_cast<std::size_t>(kCatalogSegmentEntrySize));
        put_segment_descriptor(encoded, catalog.segments[index]);
    }
    for (std::size_t index = 0; index != catalog.snapshots.size(); ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            snapshot_offset + index * kCatalogSnapshotEntrySize);
        const std::span<std::byte> encoded = std::span<std::byte>(bytes).subspan(
            offset, static_cast<std::size_t>(kCatalogSnapshotEntrySize));
        put_snapshot_descriptor(encoded, catalog.snapshots[index]);
    }
    for (std::size_t index = 0; index != catalog.runs.size(); ++index) {
        const PartitionIndexRunDescriptor& descriptor = catalog.runs[index];
        const std::size_t offset = static_cast<std::size_t>(
            run_offset + index * kCatalogRunEntrySize);
        const std::span<std::byte> encoded = std::span<std::byte>(bytes).subspan(
            offset, static_cast<std::size_t>(kCatalogRunEntrySize));
        put_uuid(encoded, 0, descriptor.run_id);
        put_u64(encoded, 16, descriptor.epoch);
        put_u32(encoded, 24, static_cast<std::uint32_t>(descriptor.kind));
        put_u32(encoded, 28, 0);
        put_u64(encoded, 32, descriptor.first_sequence);
        put_u64(encoded, 40, descriptor.last_sequence);
        put_u64(encoded, 48, descriptor.entry_count);
        put_u64(encoded, 56, descriptor.file_size);
        put_u32(encoded, 64, descriptor.body_crc32c);
        put_u32(encoded, 68, 0);
        const std::uint64_t run_bitmap_offset = checked_sum<InvalidArgumentError>(
            bitmap_offset,
            checked_product<InvalidArgumentError>(
                index * 2, one_bitmap_size, "catalog run bitmap offset"),
            "catalog run bitmap offset");
        put_u64(encoded, 72, run_bitmap_offset);
        put_u64(encoded, 80, one_bitmap_size);
        put_u32(encoded, 88, descriptor.retired_row_ref_segment_id);
        put_u32(encoded, 92, 0);
        put_uuid(encoded, 96, descriptor.retired_segment_id);
        put_u64(encoded, 112, descriptor.retired_row_count);
        put_u32(encoded, 120, descriptor.retired_ids_payload_crc32c);
        put_u32(encoded, 124, 0);
        const std::uint64_t covered_bitmap_offset =
            checked_sum<InvalidArgumentError>(
                run_bitmap_offset, one_bitmap_size,
                "catalog covered bitmap offset");
        put_u64(encoded, 128, covered_bitmap_offset);
        put_u64(encoded, 136, one_bitmap_size);
        std::copy(
            descriptor.active_partitions.begin(),
            descriptor.active_partitions.end(),
            bytes.begin() + static_cast<std::ptrdiff_t>(run_bitmap_offset));
        std::copy(
            descriptor.covered_partitions.begin(),
            descriptor.covered_partitions.end(),
            bytes.begin()
                + static_cast<std::ptrdiff_t>(covered_bitmap_offset));
    }
    for (std::size_t index = 0; index != catalog.wal_entries.size(); ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            wal_offset + index * kCatalogWalEntrySize);
        const std::span<std::byte> encoded = std::span<std::byte>(bytes).subspan(
            offset, static_cast<std::size_t>(kCatalogWalEntrySize));
        put_wal_entry(encoded, catalog.wal_entries[index]);
    }
    finish_checksum(bytes);
    return bytes;
}

PartitionedCatalogV3 decode_partitioned_catalog_v3(
    const std::span<const std::byte> bytes) {
    require<CorruptionError>(
        bytes.size() >= kPartitionedHeaderSize,
        "partitioned catalog is truncated");
    decode_common<CorruptionError>(
        bytes, kPartitionedCatalogMagic, kPartitionedCatalogWireVersion,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "partitioned catalog");
    require_zero_range<CorruptionError>(
        bytes, kCatalogKnownHeaderBytes,
        static_cast<std::size_t>(kPartitionedHeaderSize),
        "partitioned catalog header");
    require<CorruptionError>(
        get_u64(bytes, 256) == 0,
        "partitioned catalog has unsupported flags");

    const std::uint32_t partition_bits = get_u32(bytes, 100);
    const std::uint64_t partition_count =
        validate_partition_bits<CorruptionError>(partition_bits);
    const std::uint64_t segment_count = get_u64(bytes, 136);
    const std::uint64_t snapshot_count = get_u64(bytes, 160);
    const std::uint64_t run_count = get_u64(bytes, 184);
    const std::uint64_t wal_count = get_u64(bytes, 200);
    require<CorruptionError>(
        segment_count <= kPartitionedMaximumResourceEntries
            && snapshot_count == partition_count
            && run_count <= kPartitionedMaximumRuns
            && wal_count != 0
            && wal_count <= kPartitionedMaximumResourceEntries,
        "partitioned catalog resource counts are invalid");
    require<CorruptionError>(
        get_u32(bytes, 152) == kCatalogSegmentEntrySize
            && get_u32(bytes, 156) == kCatalogSnapshotEntrySize
            && get_u32(bytes, 176) == kCatalogRunEntrySize
            && get_u32(bytes, 180) == kCatalogWalEntrySize,
        "partitioned catalog entry size is invalid");

    const std::uint64_t expected_segment_offset = kPartitionedHeaderSize;
    const std::uint64_t expected_snapshot_offset = checked_sum<CorruptionError>(
        expected_segment_offset,
        checked_product<CorruptionError>(
            segment_count, kCatalogSegmentEntrySize,
            "catalog segment entries"),
        "catalog snapshot offset");
    const std::uint64_t expected_run_offset = checked_sum<CorruptionError>(
        expected_snapshot_offset,
        checked_product<CorruptionError>(
            snapshot_count, kCatalogSnapshotEntrySize,
            "catalog snapshot entries"),
        "catalog run offset");
    const std::uint64_t expected_wal_offset = checked_sum<CorruptionError>(
        expected_run_offset,
        checked_product<CorruptionError>(
            run_count, kCatalogRunEntrySize,
            "catalog run entries"),
        "catalog WAL offset");
    const std::uint64_t expected_bitmap_offset = checked_sum<CorruptionError>(
        expected_wal_offset,
        checked_product<CorruptionError>(
            wal_count, kCatalogWalEntrySize,
            "catalog WAL entries"),
        "catalog bitmap offset");
    const std::uint64_t one_bitmap_size = (partition_count + 7) / 8;
    const std::uint64_t expected_bitmap_size = checked_product<CorruptionError>(
        checked_product<CorruptionError>(
            run_count, 2, "catalog run bitmap count"),
        one_bitmap_size, "catalog run bitmaps");
    const std::uint64_t expected_file_size = checked_sum<CorruptionError>(
        expected_bitmap_offset, expected_bitmap_size,
        "partitioned catalog file size");
    require<CorruptionError>(
        get_u64(bytes, 144) == expected_segment_offset
            && get_u64(bytes, 168) == expected_snapshot_offset
            && get_u64(bytes, 192) == expected_run_offset
            && get_u64(bytes, 208) == expected_wal_offset
            && get_u64(bytes, 216) == expected_bitmap_offset
            && get_u64(bytes, 224) == expected_bitmap_size
            && get_u64(bytes, 232) == expected_file_size
            && bytes.size() == expected_file_size,
        "partitioned catalog geometry is invalid");

    PartitionedCatalogV3 catalog;
    catalog.collection_id = get_uuid(bytes, 24);
    catalog.schema_id = get_uuid(bytes, 40);
    catalog.catalog_id = get_uuid(bytes, 56);
    catalog.previous_catalog_id = get_uuid(bytes, 72);
    catalog.generation = get_u64(bytes, 88);
    catalog.next_segment_id = get_u32(bytes, 96);
    catalog.partition_bits = partition_bits;
    catalog.next_index_epoch = get_u64(bytes, 104);
    catalog.indexed_sequence = get_u64(bytes, 112);
    catalog.indexed_live_count = get_u64(bytes, 120);
    catalog.hash_seed = get_u64(bytes, 128);
    catalog.terminal_wal_id = get_uuid(bytes, 240);

    catalog.segments.reserve(static_cast<std::size_t>(segment_count));
    for (std::uint64_t index = 0; index != segment_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            expected_segment_offset + index * kCatalogSegmentEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset, static_cast<std::size_t>(kCatalogSegmentEntrySize));
        require_zero_range<CorruptionError>(
            encoded, 80, encoded.size(), "catalog segment entry");
        catalog.segments.push_back(get_segment_descriptor(encoded));
    }
    catalog.snapshots.reserve(static_cast<std::size_t>(snapshot_count));
    for (std::uint64_t index = 0; index != snapshot_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            expected_snapshot_offset + index * kCatalogSnapshotEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset, static_cast<std::size_t>(kCatalogSnapshotEntrySize));
        require<CorruptionError>(
            get_u32(encoded, 20) == 0 && get_u32(encoded, 60) == 0,
            "catalog snapshot entry has unsupported flags");
        require_zero_range<CorruptionError>(
            encoded, 64, encoded.size(), "catalog snapshot entry");
        catalog.snapshots.push_back(get_snapshot_descriptor(encoded));
    }
    catalog.runs.reserve(static_cast<std::size_t>(run_count));
    for (std::uint64_t index = 0; index != run_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            expected_run_offset + index * kCatalogRunEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset, static_cast<std::size_t>(kCatalogRunEntrySize));
        require<CorruptionError>(
            get_u32(encoded, 28) == 0 && get_u32(encoded, 68) == 0
                && get_u32(encoded, 92) == 0
                && get_u32(encoded, 124) == 0,
            "catalog run entry has unsupported flags");
        require_zero_range<CorruptionError>(
            encoded, 144, encoded.size(), "catalog run entry");
        const std::uint64_t expected_one_offset = checked_sum<CorruptionError>(
            expected_bitmap_offset,
            checked_product<CorruptionError>(
                index * 2, one_bitmap_size, "catalog run bitmap offset"),
            "catalog run bitmap offset");
        const std::uint64_t expected_covered_offset =
            checked_sum<CorruptionError>(
                expected_one_offset, one_bitmap_size,
                "catalog covered bitmap offset");
        require<CorruptionError>(
            get_u64(encoded, 72) == expected_one_offset
                && get_u64(encoded, 80) == one_bitmap_size
                && get_u64(encoded, 128) == expected_covered_offset
                && get_u64(encoded, 136) == one_bitmap_size,
            "catalog run bitmap geometry is invalid");
        PartitionIndexRunDescriptor descriptor;
        descriptor.run_id = get_uuid(encoded, 0);
        descriptor.epoch = get_u64(encoded, 16);
        descriptor.kind =
            static_cast<PartitionedRunKind>(get_u32(encoded, 24));
        descriptor.first_sequence = get_u64(encoded, 32);
        descriptor.last_sequence = get_u64(encoded, 40);
        descriptor.entry_count = get_u64(encoded, 48);
        descriptor.file_size = get_u64(encoded, 56);
        descriptor.body_crc32c = get_u32(encoded, 64);
        descriptor.retired_row_ref_segment_id = get_u32(encoded, 88);
        descriptor.retired_segment_id = get_uuid(encoded, 96);
        descriptor.retired_row_count = get_u64(encoded, 112);
        descriptor.retired_ids_payload_crc32c = get_u32(encoded, 120);
        descriptor.active_partitions.assign(
            bytes.begin() + static_cast<std::ptrdiff_t>(expected_one_offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(
                                expected_one_offset + one_bitmap_size));
        descriptor.covered_partitions.assign(
            bytes.begin()
                + static_cast<std::ptrdiff_t>(expected_covered_offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(
                                expected_covered_offset + one_bitmap_size));
        catalog.runs.push_back(std::move(descriptor));
    }
    catalog.wal_entries.reserve(static_cast<std::size_t>(wal_count));
    for (std::uint64_t index = 0; index != wal_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            expected_wal_offset + index * kCatalogWalEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset, static_cast<std::size_t>(kCatalogWalEntrySize));
        require_zero_range<CorruptionError>(
            encoded, 80, encoded.size(), "catalog WAL entry");
        catalog.wal_entries.push_back(get_wal_entry(encoded));
    }
    validate_catalog<CorruptionError>(catalog);
    return catalog;
}

std::array<std::byte, kPartitionedHeaderSize> encode_partitioned_current_v1(
    const PartitionedManifestV2& manifest,
    const PartitionedCurrentV1& current) {
    validate_manifest<InvalidArgumentError>(manifest);
    require<InvalidArgumentError>(
        current.collection_id == manifest.collection_id
            && current.schema_id == manifest.schema_id,
        "partitioned CURRENT identity does not match manifest");
    require<InvalidArgumentError>(
        current.generation != 0 && !current.catalog_id.empty()
            && current.catalog_file_size >= kPartitionedHeaderSize
            && current.catalog_file_size
                   <= kPartitionedMaximumCatalogFileSize,
        "partitioned CURRENT metadata is invalid");
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    encode_common(
        bytes, kPartitionedCurrentMagic, 1,
        static_cast<std::uint32_t>(kPartitionedHeaderSize));
    put_uuid(bytes, 24, current.collection_id);
    put_uuid(bytes, 40, current.schema_id);
    put_u64(bytes, 56, current.generation);
    put_uuid(bytes, 64, current.catalog_id);
    put_u64(bytes, 80, current.catalog_file_size);
    put_u32(bytes, 88, current.catalog_crc32c);
    put_u32(bytes, 92, 0);
    finish_checksum(bytes);
    return bytes;
}

PartitionedCurrentV1 decode_partitioned_current_v1(
    const std::span<const std::byte> bytes,
    const PartitionedManifestV2& manifest) {
    require<CorruptionError>(
        bytes.size() == kPartitionedHeaderSize,
        "partitioned CURRENT is not exactly 4096 bytes");
    decode_common<CorruptionError>(
        bytes, kPartitionedCurrentMagic, 1,
        static_cast<std::uint32_t>(kPartitionedHeaderSize),
        "partitioned CURRENT");
    require_zero_range<CorruptionError>(
        bytes, kCurrentKnownBytes, bytes.size(), "partitioned CURRENT");
    require<CorruptionError>(
        get_u32(bytes, 92) == 0,
        "partitioned CURRENT has unsupported flags");
    PartitionedCurrentV1 current;
    current.collection_id = get_uuid(bytes, 24);
    current.schema_id = get_uuid(bytes, 40);
    current.generation = get_u64(bytes, 56);
    current.catalog_id = get_uuid(bytes, 64);
    current.catalog_file_size = get_u64(bytes, 80);
    current.catalog_crc32c = get_u32(bytes, 88);
    try {
        static_cast<void>(encode_partitioned_current_v1(manifest, current));
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
    return current;
}

}  // namespace throwdb::detail
