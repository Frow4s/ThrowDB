#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include "native_format.hpp"
#include "mutation_types.hpp"

namespace throwdb::detail {

inline constexpr std::uint64_t kMutationDeltaHeaderSize = 4096;
inline constexpr std::uint64_t kMutationDeltaVersionRecordSize = 32;
inline constexpr std::uint64_t kMutationDeltaReverseRecordSize = 16;
inline constexpr std::uint64_t kMutationDeltaIndexAlignment = 4096;
inline constexpr std::uint64_t kMutationDeltaTombstonePayloadRow =
    std::numeric_limits<std::uint64_t>::max();

// This descriptor is copied verbatim into the immutable catalog and is also
// redundantly bound by both files in the delta pair. The body checksums are
// deliberately included: a catalog generation identifies exact immutable
// file contents, not merely UUID-derived names.
struct MutationDeltaDescriptor {
    Uuid128 delta_id;
    Uuid128 source_predecessor_wal_id;
    Uuid128 source_terminal_wal_id;
    std::uint64_t first_sequence{0};
    std::uint64_t last_sequence{0};
    std::uint32_t segment_id{0};
    std::uint32_t source_wal_count{0};
    std::uint64_t entry_count{0};
    std::uint64_t put_count{0};
    std::uint64_t data_file_size{0};
    std::uint64_t index_file_size{0};
    std::uint32_t data_payload_crc32c{0};
    std::uint32_t index_body_crc32c{0};

    bool operator==(const MutationDeltaDescriptor&) const = default;
};

struct MutationDeltaBuildSpec {
    Uuid128 delta_id;
    Uuid128 source_predecessor_wal_id;
    Uuid128 source_terminal_wal_id;
    std::uint64_t first_sequence{0};
    std::uint64_t last_sequence{0};
    std::uint32_t segment_id{0};
    std::uint32_t source_wal_count{0};
};

// For a PUT, value.row_ref() is the source WAL/delta row whose payload must be
// copied. The writer replaces it with a row in spec.segment_id in the on-disk
// version index. A tombstone has no source row.
struct MutationDeltaBuildEntry {
    std::uint64_t id{0};
    VersionedMutationValue value;
};

struct MutationDeltaVersionRecord {
    std::uint64_t id{0};
    VersionedMutationValue value;

    bool operator==(const MutationDeltaVersionRecord&) const = default;
};

struct MutationDeltaDataHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    MutationDeltaDescriptor descriptor;
    std::uint64_t record_size{0};
    std::uint32_t dimension{0};
    DTypeCode dtype{DTypeCode::Invalid};
    std::uint32_t item_size{0};
    IdType id_type{IdType::Invalid};
    RowLayout row_layout{RowLayout::Invalid};
    std::uint64_t payload_offset{kMutationDeltaHeaderSize};
    std::uint64_t payload_size{0};
};

struct MutationDeltaIndexHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    MutationDeltaDescriptor descriptor;
    std::uint64_t records_offset{kMutationDeltaHeaderSize};
    std::uint64_t records_size{0};
    std::uint64_t reverse_offset{kMutationDeltaHeaderSize};
    std::uint64_t reverse_size{0};
};

// destination contains source_rows.size() consecutive record-sized slots.
// The callback must fill all bytes or throw. Calls are bounded by
// payload_batch_bytes and occur in output payload-row order.
using MutationDeltaPayloadReader = std::function<void(
    std::span<const RowRef> source_rows,
    std::span<std::byte> destination)>;

struct OpenMutationDelta {
    MutationDeltaDescriptor descriptor;
    MutationDeltaDataHeader data_header;
    FileDescriptor data_fd;
    // Strictly increasing by ID. PUT RowRefs resolve to descriptor.segment_id.
    std::vector<MutationDeltaVersionRecord> records;
};

[[nodiscard]] std::filesystem::path mutation_deltas_directory(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path mutation_delta_data_path(
    const std::filesystem::path& store_path, Uuid128 delta_id);
[[nodiscard]] std::filesystem::path mutation_delta_index_path(
    const std::filesystem::path& store_path, Uuid128 delta_id);

// Produces an immutable data/index pair. Payload is streamed in bounded
// batches; only version metadata is sorted and retained in memory. Both temp
// files are fully synced before they are renamed, followed by one directory
// fsync. Unreferenced partial publications are safe crash orphans. As with
// catalog publication, the caller holds the collection's exclusive writer
// lease, which also serializes publishers of a UUID-derived destination.
[[nodiscard]] MutationDeltaDescriptor write_mutation_delta(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationDeltaBuildSpec& spec,
    std::span<const MutationDeltaBuildEntry> entries,
    const MutationDeltaPayloadReader& payload_reader,
    std::size_t payload_batch_bytes = 8U * 1024U * 1024U);

// Normal recovery validates the complete index body but deliberately avoids a
// multi-gigabyte payload scan. Set verify_payload_checksum for scrub/tests.
[[nodiscard]] OpenMutationDelta open_mutation_delta(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor,
    bool verify_payload_checksum = false);

}  // namespace throwdb::detail
