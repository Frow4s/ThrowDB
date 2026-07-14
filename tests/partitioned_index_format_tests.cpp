#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partitioned_index_format.hpp"

namespace {

using namespace throwdb::detail;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

template <typename Actual, typename Expected>
void require_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string& message) {
    if (!(actual == expected)) {
        fail(message);
    }
}

template <typename Exception, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
    try {
        std::forward<Callback>(callback)();
    } catch (const Exception&) {
        return;
    } catch (...) {
        fail(message + " (wrong exception type)");
    }
    fail(message + " (no exception)");
}

template <typename Bytes>
void refresh_checksum(Bytes& bytes) {
    const std::span<std::byte> span(bytes);
    put_u32(span, 20, 0);
    put_u32(span, 20, crc32c(span));
}

[[nodiscard]] Uuid128 uuid(const std::uint64_t value) {
    return Uuid128{value, value + 0x1000};
}

[[nodiscard]] CollectionSchema schema() {
    CollectionSchema result;
    result.id_type = IdType::UInt64;
    result.dtype = DTypeCode::Float16;
    result.dimension = 2;
    result.item_size = 2;
    result.row_layout = RowLayout::DenseRowMajor;
    result.index_kind = IndexKind::UInt64PartitionedRowRef;
    result.hash_algorithm = HashAlgorithm::SplitMix64V1;
    return result;
}

[[nodiscard]] PartitionedManifestV2 manifest() {
    PartitionedManifestV2 result;
    result.collection_id = uuid(1);
    result.schema_id = uuid(2);
    result.initial_active_wal_id = uuid(3);
    result.schema = schema();
    result.record_size = 4;
    result.partition_bits = 2;
    result.partition_count = 4;
    result.hash_seed = 0x123456789abcdef0ULL;
    result.target_segment_size = 1ULL << 20;
    result.initial_record_count = 2;
    result.initial_active_wal_segment_id = 2;
    result.initial_next_segment_id = 3;
    result.creation_time_unix_ns = 123456;
    return result;
}

[[nodiscard]] PartitionedSegmentDescriptor segment_descriptor() {
    PartitionedSegmentDescriptor descriptor;
    descriptor.segment_id = uuid(10);
    descriptor.row_ref_segment_id = 1;
    descriptor.kind = PartitionedSegmentKind::Initial;
    descriptor.row_count = 2;
    descriptor.minimum_sequence = 0;
    descriptor.maximum_sequence = 0;
    descriptor.data_file_size = kPartitionedHeaderSize + 8;
    descriptor.ids_file_size = kPartitionedHeaderSize + 32;
    descriptor.data_payload_crc32c = 0x11111111U;
    descriptor.ids_payload_crc32c = 0x22222222U;
    descriptor.creation_epoch = 0;
    return descriptor;
}

[[nodiscard]] std::uint64_t snapshot_file_size(const std::uint64_t capacity) {
    const std::uint64_t lookup = align_up(
        kPartitionedHeaderSize + capacity, kPartitionedHeaderSize);
    const std::uint64_t sequences = align_up(
        lookup + capacity * kPartitionSnapshotLookupSlotSize,
        kPartitionedHeaderSize);
    return sequences + capacity * kPartitionSnapshotSequenceSlotSize;
}

[[nodiscard]] PartitionSnapshotDescriptor snapshot_descriptor(
    const std::uint32_t partition) {
    PartitionSnapshotDescriptor descriptor;
    descriptor.snapshot_id = uuid(100 + partition);
    descriptor.partition_id = partition;
    descriptor.epoch = 0;
    descriptor.entry_count = partition == 0 ? 2 : 0;
    descriptor.capacity = 8;
    descriptor.file_size = snapshot_file_size(descriptor.capacity);
    descriptor.body_crc32c = 0x30000000U + partition;
    return descriptor;
}

[[nodiscard]] std::uint64_t run_file_size(
    const std::uint32_t partition_bits,
    const std::uint64_t entries) {
    const std::uint64_t partition_count = std::uint64_t{1} << partition_bits;
    const std::uint64_t records = align_up(
        kPartitionedHeaderSize
            + partition_count * kPartitionRunTocEntrySize,
        kPartitionedHeaderSize);
    return records + entries * kPartitionRunRecordSize;
}

[[nodiscard]] PartitionIndexRunDescriptor run_descriptor() {
    PartitionIndexRunDescriptor descriptor;
    descriptor.run_id = uuid(200);
    descriptor.epoch = 1;
    descriptor.kind = PartitionedRunKind::Mutation;
    descriptor.first_sequence = 1;
    descriptor.last_sequence = 1;
    descriptor.entry_count = 1;
    descriptor.file_size = run_file_size(2, 1);
    descriptor.body_crc32c = 0x44444444U;
    descriptor.covered_partitions = {std::byte{0x0f}};
    descriptor.active_partitions = {std::byte{0x0f}};
    return descriptor;
}

[[nodiscard]] MutationWalEntry active_wal() {
    MutationWalEntry entry;
    entry.wal_id = uuid(300);
    entry.predecessor_wal_id = uuid(299);
    entry.segment_id = 2;
    entry.state = MutationWalState::Active;
    entry.previous_sequence = 1;
    entry.first_sequence = 2;
    return entry;
}

[[nodiscard]] PartitionedCatalogV3 catalog() {
    PartitionedCatalogV3 result;
    result.collection_id = uuid(1);
    result.schema_id = uuid(2);
    result.generation = 1;
    result.catalog_id = uuid(400);
    result.next_segment_id = 3;
    result.next_index_epoch = 2;
    result.indexed_sequence = 1;
    result.indexed_live_count = 2;
    result.terminal_wal_id = uuid(299);
    result.partition_bits = 2;
    result.hash_seed = 0x123456789abcdef0ULL;
    result.segments = {segment_descriptor()};
    for (std::uint32_t partition = 0; partition != 4; ++partition) {
        result.snapshots.push_back(snapshot_descriptor(partition));
    }
    result.runs = {run_descriptor()};
    result.wal_entries = {active_wal()};
    return result;
}

void test_manifest_v2_round_trip_and_feature_rejection() {
    const PartitionedManifestV2 expected = manifest();
    const auto encoded = encode_partitioned_manifest_v2(expected);
    require_equal(
        decode_partitioned_manifest_v2(encoded), expected,
        "partitioned manifest did not round trip");

    auto checksum_damage = encoded;
    checksum_damage.back() ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_manifest_v2(checksum_damage)); },
        "manifest body corruption was accepted");

    auto unknown_feature = encoded;
    put_u64(
        unknown_feature, 144,
        get_u64(unknown_feature, 144) | (std::uint64_t{1} << 63));
    refresh_checksum(unknown_feature);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_manifest_v2(unknown_feature)); },
        "manifest unknown required feature was accepted");

    auto reserved = encoded;
    reserved[300] = std::byte{1};
    refresh_checksum(reserved);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_manifest_v2(reserved)); },
        "manifest nonzero reserved byte was accepted");

    PartitionedManifestV2 invalid = expected;
    invalid.initial_next_segment_id = 0xffffffffU;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_manifest_v2(invalid)); },
        "manifest encode did not report invalid segment ID as argument error");
    auto corrupted_segment_id = encoded;
    put_u32(corrupted_segment_id, 164, 0xffffffffU);
    refresh_checksum(corrupted_segment_id);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                decode_partitioned_manifest_v2(corrupted_segment_id));
        },
        "manifest decode did not report invalid segment ID as corruption");
}

void test_segment_headers_round_trip_and_strict_geometry() {
    PartitionedSegmentDataHeaderV1 data;
    data.collection_id = uuid(1);
    data.schema_id = uuid(2);
    data.descriptor = segment_descriptor();
    data.schema = schema();
    data.record_size = 4;
    data.payload_size = 8;
    const auto encoded_data = encode_partitioned_segment_data_header_v1(data);
    require_equal(
        decode_partitioned_segment_data_header_v1(encoded_data), data,
        "segment data header did not round trip");

    PartitionedSegmentIdsHeaderV1 ids;
    ids.collection_id = uuid(1);
    ids.schema_id = uuid(2);
    ids.descriptor = segment_descriptor();
    ids.payload_size = 32;
    const auto encoded_ids = encode_partitioned_segment_ids_header_v1(ids);
    require_equal(
        decode_partitioned_segment_ids_header_v1(encoded_ids), ids,
        "segment IDs header did not round trip");

    data.descriptor.data_file_size += 1;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_segment_data_header_v1(data)); },
        "segment data size mismatch was accepted");

    auto reserved = encoded_ids;
    reserved[300] = std::byte{1};
    refresh_checksum(reserved);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_segment_ids_header_v1(reserved)); },
        "segment IDs reserved bytes were accepted");
}

void test_snapshot_header_round_trip_and_bounds() {
    PartitionSnapshotHeaderV1 header;
    header.collection_id = uuid(1);
    header.schema_id = uuid(2);
    header.descriptor = snapshot_descriptor(0);
    header.partition_bits = 2;
    header.hash_seed = 0x123456789abcdef0ULL;
    header.lookup_slots_offset = 8192;
    header.sequence_slots_offset = 12288;
    const auto encoded = encode_partition_snapshot_header_v1(header);
    require_equal(
        decode_partition_snapshot_header_v1(encoded), header,
        "partition snapshot header did not round trip");

    auto bad_offset = encoded;
    put_u64(bad_offset, 136, 4096);
    refresh_checksum(bad_offset);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partition_snapshot_header_v1(bad_offset)); },
        "snapshot overlapping geometry was accepted");

    header.descriptor.entry_count = 7;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partition_snapshot_header_v1(header)); },
        "snapshot above 70 percent load was accepted");
}

void test_run_header_records_and_retirement_contract() {
    PartitionIndexRunHeaderV1 header;
    header.collection_id = uuid(1);
    header.schema_id = uuid(2);
    header.descriptor = run_descriptor();
    header.partition_bits = 2;
    header.hash_seed = 0x123456789abcdef0ULL;
    header.toc_size = 64;
    header.records_offset = 8192;
    header.records_size = 24;
    const auto encoded = encode_partition_index_run_header_v1(header);
    require_equal(
        decode_partition_index_run_header_v1(encoded), header,
        "partition run header did not round trip");

    const PartitionRunRecord put{
        0, 0, RowRef::create(1, 1).raw()};
    require_equal(
        decode_partition_run_record(encode_partition_run_record(put)), put,
        "partition run PUT did not round trip");
    const PartitionRunRecord tombstone{
        std::numeric_limits<std::uint64_t>::max(), 1, 0};
    require_equal(
        decode_partition_run_record(encode_partition_run_record(tombstone)),
        tombstone, "partition run tombstone did not round trip");
    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(encode_partition_run_record(
                {1, std::numeric_limits<std::uint64_t>::max(), 0}));
        },
        "run record reserved sequence was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(encode_partition_run_record({1, 0, 7})); },
        "run record invalid RowRef had wrong encode error semantics");
    auto invalid_persisted_row = encode_partition_run_record(tombstone);
    put_u64(invalid_persisted_row, 16, 7);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                decode_partition_run_record(invalid_persisted_row));
        },
        "run record invalid RowRef had wrong decode error semantics");

    PartitionIndexRunDescriptor cleaner = run_descriptor();
    cleaner.kind = PartitionedRunKind::Cleaner;
    cleaner.retired_row_ref_segment_id = 1;
    cleaner.retired_segment_id = uuid(10);
    cleaner.retired_row_count = 2;
    cleaner.retired_ids_payload_crc32c = 77;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            PartitionIndexRunHeaderV1 invalid = header;
            invalid.descriptor = cleaner;
            static_cast<void>(encode_partition_index_run_header_v1(invalid));
        },
        "cleaner run without exact source-row coverage was accepted");
}

void test_catalog_v3_round_trip() {
    const PartitionedCatalogV3 expected = catalog();
    const std::vector<std::byte> encoded =
        encode_partitioned_catalog_v3(expected);
    require_equal(
        decode_partitioned_catalog_v3(encoded), expected,
        "partitioned catalog did not round trip");
    require_equal(
        get_u32(encoded, 8), kPartitionedCatalogWireVersion,
        "catalog wire version changed");
}

void test_catalog_rejects_corruption_reserved_fields_and_bounds() {
    const std::vector<std::byte> encoded =
        encode_partitioned_catalog_v3(catalog());

    auto checksum_damage = encoded;
    checksum_damage.back() ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_catalog_v3(checksum_damage)); },
        "catalog checksum damage was accepted");

    auto reserved = encoded;
    put_u64(reserved, 256, 1);
    refresh_checksum(reserved);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_catalog_v3(reserved)); },
        "catalog unsupported header flags were accepted");

    auto wrong_partition = encoded;
    const std::uint64_t snapshot_offset = get_u64(wrong_partition, 168);
    put_u32(
        wrong_partition, static_cast<std::size_t>(snapshot_offset + 16), 1);
    refresh_checksum(wrong_partition);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_catalog_v3(wrong_partition)); },
        "catalog out-of-order snapshots were accepted");

    auto bad_count = encoded;
    put_u64(bad_count, 184, kPartitionedMaximumRuns + 1);
    refresh_checksum(bad_count);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_catalog_v3(bad_count)); },
        "catalog excessive run count was accepted");

    auto bitmap_padding = encoded;
    const std::uint64_t bitmap_offset = get_u64(bitmap_padding, 216);
    bitmap_padding[static_cast<std::size_t>(bitmap_offset)] = std::byte{0x8f};
    refresh_checksum(bitmap_padding);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(decode_partitioned_catalog_v3(bitmap_padding)); },
        "catalog bitmap padding bits were accepted");
}

void test_catalog_rejects_resource_aliases_and_lineage_gaps() {
    PartitionedCatalogV3 invalid = catalog();
    invalid.snapshots[0].snapshot_id = invalid.segments[0].segment_id;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_catalog_v3(invalid)); },
        "catalog resource UUID alias was accepted");

    invalid = catalog();
    invalid.wal_entries[0].previous_sequence = 0;
    invalid.wal_entries[0].first_sequence = 1;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_catalog_v3(invalid)); },
        "catalog WAL lineage gap was accepted");

    invalid = catalog();
    invalid.next_index_epoch = 1;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_catalog_v3(invalid)); },
        "catalog reused an index epoch");

    invalid = catalog();
    invalid.runs[0].active_partitions = {std::byte{0x0e}};
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_catalog_v3(invalid)); },
        "catalog cleared a run partition without a covering snapshot");
    invalid.snapshots[0].epoch = invalid.runs[0].epoch;
    static_cast<void>(encode_partitioned_catalog_v3(invalid));

    invalid = catalog();
    invalid.runs[0].covered_partitions = {std::byte{0x07}};
    invalid.runs[0].active_partitions = {std::byte{0x08}};
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_partitioned_catalog_v3(invalid)); },
        "catalog active bitmap escaped immutable run coverage");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 7> tests{{
        {"manifest_v2_round_trip_and_feature_rejection",
         test_manifest_v2_round_trip_and_feature_rejection},
        {"segment_headers_round_trip_and_strict_geometry",
         test_segment_headers_round_trip_and_strict_geometry},
        {"snapshot_header_round_trip_and_bounds",
         test_snapshot_header_round_trip_and_bounds},
        {"run_header_records_and_retirement_contract",
         test_run_header_records_and_retirement_contract},
        {"catalog_v3_round_trip", test_catalog_v3_round_trip},
        {"catalog_rejects_corruption_reserved_fields_and_bounds",
         test_catalog_rejects_corruption_reserved_fields_and_bounds},
        {"catalog_rejects_resource_aliases_and_lineage_gaps",
         test_catalog_rejects_resource_aliases_and_lineage_gaps},
    }};

    std::size_t failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
