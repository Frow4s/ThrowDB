#include "native_format.hpp"

#include "crc32c.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace throwdb::detail {
namespace {

constexpr std::size_t kIndexIoEntries = 1U << 16;
constexpr std::size_t kManifestKnownBytes = 200;
constexpr std::size_t kDataKnownBytes = 120;
constexpr std::size_t kIdsKnownBytes = 104;
constexpr std::size_t kIndexKnownBytes = 136;

static_assert(std::is_trivially_copyable_v<IndexSlot>);
static_assert(sizeof(IndexSlot) == kNativeIndexSlotSize);
static_assert(offsetof(IndexSlot, id) == 0);
static_assert(offsetof(IndexSlot, ordinal) == sizeof(std::uint64_t));

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(
    const SteadyClock::time_point started) noexcept {
    return std::chrono::duration<double>(SteadyClock::now() - started).count();
}

template <typename Error>
void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
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
[[nodiscard]] std::uint64_t checked_align(
    const std::uint64_t value,
    const std::uint64_t alignment,
    const std::string_view description) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw Error(std::string(description) + " has invalid alignment");
    }
    const std::uint64_t remainder = value & (alignment - 1);
    if (remainder == 0) {
        return value;
    }
    return checked_sum<Error>(
        value, alignment - remainder, description);
}

template <typename Error>
void require_nonempty_uuid(
    const Uuid128& uuid, const std::string_view description) {
    require<Error>(!uuid.empty(), std::string(description) + " is empty");
}

template <typename Error>
void require_valid_segment_id(const std::uint32_t segment_id) {
    require<Error>(
        segment_id >= SegmentId::kMinimum
            && segment_id <= SegmentId::kMaximum,
        "segment ID is invalid or reserved");
}

[[nodiscard]] std::uint32_t header_crc32c(
    const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes.first(kNativeHeaderChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(
        kNativeHeaderChecksumOffset + sizeof(std::uint32_t),
        kNativeHeaderSize
            - kNativeHeaderChecksumOffset - sizeof(std::uint32_t)));
    return checksum.value();
}

void put_magic(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    std::memcpy(bytes.data(), magic.data(), magic.size());
}

void encode_common(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    put_magic(bytes, magic);
    put_u32(bytes, 8, kNativeFormatVersion);
    put_u32(bytes, 12, static_cast<std::uint32_t>(kNativeHeaderSize));
    put_u32(bytes, 16, kNativeCrc32cAlgorithm);
}

void finish_header_crc(const std::span<std::byte> bytes) {
    put_u32(bytes, kNativeHeaderChecksumOffset, 0);
    put_u32(
        bytes, kNativeHeaderChecksumOffset,
        header_crc32c(std::span<const std::byte>(bytes)));
}

void decode_common(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic,
    const std::string_view description) {
    if (bytes.size() < kNativeHeaderSize) {
        throw CorruptionError(std::string(description) + " header is truncated");
    }
    if (std::memcmp(bytes.data(), magic.data(), magic.size()) != 0) {
        throw CorruptionError(std::string(description) + " has invalid magic");
    }
    if (get_u32(bytes, 8) != kNativeFormatVersion) {
        throw CorruptionError(
            std::string(description) + " has unsupported format version");
    }
    if (get_u32(bytes, 12) != kNativeHeaderSize) {
        throw CorruptionError(std::string(description) + " has invalid header size");
    }
    if (get_u32(bytes, 16) != kNativeCrc32cAlgorithm) {
        throw CorruptionError(
            std::string(description) + " has unsupported checksum algorithm");
    }
    if (get_u32(bytes, kNativeHeaderChecksumOffset)
        != header_crc32c(bytes.first(kNativeHeaderSize))) {
        throw CorruptionError(
            std::string(description) + " header checksum does not match");
    }
}

void require_reserved_zero(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::string_view description) {
    for (std::size_t offset = begin; offset != kNativeHeaderSize; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw CorruptionError(
                std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

template <typename Error>
void validate_schema(
    const CollectionSchema& schema,
    const std::uint64_t record_size,
    const std::string_view description) {
    require<Error>(
        schema.dtype == DTypeCode::Float16
            || schema.dtype == DTypeCode::Float32,
        std::string(description) + " has unsupported dtype");
    require<Error>(
        schema.dimension != 0,
        std::string(description) + " has zero dimension");
    require<Error>(
        schema.item_size == dtype_item_size(schema.dtype),
        std::string(description) + " dtype and item size disagree");
    require<Error>(
        schema.id_type == IdType::UInt64,
        std::string(description) + " has unsupported ID type");
    require<Error>(
        schema.row_layout == RowLayout::DenseRowMajor,
        std::string(description) + " has unsupported row layout");
    require<Error>(
        schema.index_kind == IndexKind::UInt64Flat,
        std::string(description) + " has unsupported index kind");
    require<Error>(
        schema.hash_algorithm == HashAlgorithm::SplitMix64V1,
        std::string(description) + " has unsupported hash algorithm");
    require<Error>(
        record_size
            == checked_product<Error>(
                schema.dimension, schema.item_size,
                std::string(description) + " record size"),
        std::string(description) + " record size does not match schema");
}

template <typename Error>
void validate_manifest(const NativeManifest& manifest) {
    require_nonempty_uuid<Error>(manifest.collection_id, "collection UUID");
    require_nonempty_uuid<Error>(manifest.schema_id, "schema UUID");
    require_nonempty_uuid<Error>(manifest.active_wal_id, "active WAL UUID");
    validate_schema<Error>(manifest.schema, manifest.record_size, "manifest");
    require<Error>(
        manifest.segment_payload_capacity >= manifest.record_size,
        "segment payload capacity cannot fit one row");
    require<Error>(
        manifest.records_per_segment
            == manifest.segment_payload_capacity / manifest.record_size,
        "manifest records-per-segment geometry is invalid");
    require<Error>(
        manifest.records_per_segment != 0
            && manifest.records_per_segment <= RowRef::kMaximumRowsPerSegment,
        "manifest segment row count exceeds RowRef capacity");

    const std::uint64_t expected_segment_count =
        manifest.base_record_count / manifest.records_per_segment
        + (manifest.base_record_count % manifest.records_per_segment == 0 ? 0 : 1);
    require<Error>(
        manifest.base_segment_count == expected_segment_count,
        "manifest base segment count does not match record geometry");
    require<Error>(
        manifest.base_segment_count <= 0xfffffffcULL,
        "manifest has no space for active and next segment IDs");
    require<Error>(
        manifest.active_wal_segment_id == manifest.base_segment_count + 1,
        "manifest active WAL segment ID is not contiguous");
    require<Error>(
        manifest.next_segment_id
            == static_cast<std::uint64_t>(manifest.active_wal_segment_id) + 1,
        "manifest next segment ID is not contiguous");
    require_valid_segment_id<Error>(manifest.active_wal_segment_id);
    require_valid_segment_id<Error>(manifest.next_segment_id);

    require<Error>(
        manifest.index_capacity >= 8
            && (manifest.index_capacity & (manifest.index_capacity - 1)) == 0
            && manifest.base_record_count < manifest.index_capacity,
        "manifest has invalid flat-index capacity");
    require<Error>(
        manifest.index_control_offset == kNativeHeaderSize,
        "manifest has invalid index control offset");
    const std::uint64_t controls_end = checked_sum<Error>(
        manifest.index_control_offset, manifest.index_capacity,
        "native index controls");
    const std::uint64_t expected_slots_offset = checked_align<Error>(
        controls_end, kNativeHeaderSize, "native index slots");
    require<Error>(
        manifest.index_slots_offset == expected_slots_offset,
        "manifest has invalid index slots offset");
    require<Error>(
        manifest.index_slot_size == kNativeIndexSlotSize,
        "manifest has invalid index slot size");
    const std::uint64_t slot_bytes = checked_product<Error>(
        manifest.index_capacity, manifest.index_slot_size,
        "native index slots");
    require<Error>(
        manifest.index_file_size
            == checked_sum<Error>(
                manifest.index_slots_offset, slot_bytes,
                "native index file size"),
        "manifest has invalid index file size");
}

template <typename Error>
void validate_data_header(const NativeDataHeader& header) {
    require_nonempty_uuid<Error>(header.collection_id, "data collection UUID");
    require_nonempty_uuid<Error>(header.schema_id, "data schema UUID");
    require_valid_segment_id<Error>(header.segment_id);
    CollectionSchema schema;
    schema.dtype = header.dtype;
    schema.dimension = header.dimension;
    schema.item_size = header.item_size;
    schema.id_type = header.id_type;
    schema.row_layout = header.row_layout;
    schema.index_kind = IndexKind::UInt64Flat;
    schema.hash_algorithm = HashAlgorithm::SplitMix64V1;
    validate_schema<Error>(schema, header.record_size, "data segment");
    require<Error>(
        header.row_count <= RowRef::kMaximumRowsPerSegment,
        "data segment row count exceeds RowRef capacity");
    require<Error>(
        header.payload_offset == kNativeHeaderSize,
        "data segment has invalid payload offset");
    const std::uint64_t expected_payload = checked_product<Error>(
        header.row_count, header.record_size, "data segment payload");
    require<Error>(
        header.payload_size == expected_payload,
        "data segment payload size does not match row geometry");
    require<Error>(
        header.file_size
            == checked_sum<Error>(
                header.payload_offset, header.payload_size,
                "data segment file size"),
        "data segment has invalid file size");
}

template <typename Error>
void validate_ids_header(const NativeIdsHeader& header) {
    require_nonempty_uuid<Error>(header.collection_id, "IDs collection UUID");
    require_nonempty_uuid<Error>(header.schema_id, "IDs schema UUID");
    require_valid_segment_id<Error>(header.segment_id);
    require<Error>(
        header.id_type == IdType::UInt64,
        "IDs sidecar has unsupported ID type");
    require<Error>(
        header.row_count <= RowRef::kMaximumRowsPerSegment,
        "IDs sidecar row count exceeds RowRef capacity");
    require<Error>(
        header.entry_size == kNativeIdEntrySize,
        "IDs sidecar has invalid entry size");
    require<Error>(
        header.payload_offset == kNativeHeaderSize,
        "IDs sidecar has invalid payload offset");
    const std::uint64_t expected_payload = checked_product<Error>(
        header.row_count, header.entry_size, "IDs sidecar payload");
    require<Error>(
        header.payload_size == expected_payload,
        "IDs sidecar payload size does not match row geometry");
    require<Error>(
        header.file_size
            == checked_sum<Error>(
                header.payload_offset, header.payload_size,
                "IDs sidecar file size"),
        "IDs sidecar has invalid file size");
}

template <typename Error>
void validate_index_header_fields(const NativeIndexHeader& header) {
    require_nonempty_uuid<Error>(header.collection_id, "index collection UUID");
    require_nonempty_uuid<Error>(header.schema_id, "index schema UUID");
    require<Error>(
        header.capacity >= 8
            && (header.capacity & (header.capacity - 1)) == 0
            && header.record_count < header.capacity,
        "native index has invalid capacity");
    require<Error>(
        header.control_offset == kNativeHeaderSize,
        "native index has invalid control offset");
    const std::uint64_t expected_slots = checked_align<Error>(
        checked_sum<Error>(
            header.control_offset, header.capacity,
            "native index controls"),
        kNativeHeaderSize, "native index slots");
    require<Error>(
        header.slots_offset == expected_slots,
        "native index has invalid slots offset");
    require<Error>(
        header.slot_size == kNativeIndexSlotSize,
        "native index has invalid slot size");
    const std::uint64_t slot_bytes = checked_product<Error>(
        header.capacity, header.slot_size, "native index slots");
    require<Error>(
        header.file_size
            == checked_sum<Error>(
                header.slots_offset, slot_bytes,
                "native index file size"),
        "native index has invalid file size");
    require<Error>(
        header.control_encoding == kNativeFlatControlEncoding,
        "native index has unsupported control encoding");
    require<Error>(
        header.value_encoding == kNativeRowRefValueEncoding,
        "native index has unsupported value encoding");
    require<Error>(
        header.base_segment_count <= 0xfffffffcULL,
        "native index has invalid base segment count");
    require<Error>(
        header.body_checksum_algorithm == 0
            || header.body_checksum_algorithm
                   == kNativeIndexBodyCrc32cAlgorithm,
        "native index has unsupported body checksum algorithm");
    require<Error>(
        header.body_checksum_algorithm != 0 || header.body_checksum == 0,
        "native index without a checksum algorithm has a checksum value");
}

void write_header_atomic(
    const std::filesystem::path& directory,
    const std::filesystem::path& temporary,
    const std::filesystem::path& final_path,
    const std::span<const std::byte> bytes,
    const std::string_view description) {
    auto fd = create_exclusive(temporary);
    try {
        write_all(fd.get(), bytes.data(), bytes.size(), temporary);
        sync_file(fd.get(), temporary);
        fd.reset();
        std::error_code error;
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            throw StorageError(
                "cannot publish " + std::string(description) + ": "
                + error.message());
        }
        sync_directory(directory);
    } catch (...) {
        fd.reset();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void require_manifest_match(
    const NativeManifest& manifest,
    const Uuid128& collection_id,
    const Uuid128& schema_id,
    const std::string_view description) {
    if (collection_id != manifest.collection_id
        || schema_id != manifest.schema_id) {
        throw CorruptionError(
            std::string(description) + " identity does not match manifest");
    }
}

void insert_native_index(
    IndexData& index,
    const std::uint64_t id,
    const RowRef row_ref) {
    const std::uint64_t mask = index.capacity - 1;
    std::uint64_t position = hash_id(id, index.hash_seed) & mask;
    for (std::uint64_t probe = 0; probe != index.capacity; ++probe) {
        const auto slot = static_cast<std::size_t>(position);
        if (index.controls[slot] == 0) {
            index.controls[slot] = 1;
            index.slots[slot] = IndexSlot{id, row_ref.raw()};
            return;
        }
        if (index.slots[slot].id == id) {
            throw DuplicateIdError(id);
        }
        position = (position + 1) & mask;
    }
    throw StorageError("native flat index is unexpectedly full");
}

void validate_index_data_for_write(
    const NativeManifest& manifest, const IndexData& index) {
    if (index.count != manifest.base_record_count
        || index.capacity != manifest.index_capacity
        || index.hash_seed != manifest.hash_seed
        || index.controls.size() != index.capacity
        || index.slots.size() != index.capacity) {
        throw InvalidArgumentError("native index data does not match manifest");
    }
}

void validate_native_index_body(
    const NativeManifest& manifest, const IndexData& index) {
    std::uint64_t occupied = 0;
    const std::uint64_t bitmap_words = manifest.base_record_count / 64
        + (manifest.base_record_count % 64 == 0 ? 0 : 1);
    if (bitmap_words > std::numeric_limits<std::size_t>::max()) {
        throw CorruptionError("native index RowRef bitmap does not fit memory");
    }
    std::vector<std::uint64_t> seen_rows(
        static_cast<std::size_t>(bitmap_words), 0);
    const std::uint64_t mask = index.capacity - 1;

    for (std::uint64_t position = 0; position != index.capacity; ++position) {
        const auto slot_index = static_cast<std::size_t>(position);
        const std::uint8_t control = index.controls[slot_index];
        const IndexSlot& slot = index.slots[slot_index];
        if (control > 1) {
            throw CorruptionError("native index contains an invalid control byte");
        }
        if (control == 0) {
            if (slot.id != 0 || slot.ordinal != 0) {
                throw CorruptionError("native index empty slot is not zero-filled");
            }
            continue;
        }
        ++occupied;

        const RowRef row_ref = RowRef::decode(slot.ordinal);
        const std::uint32_t segment = row_ref.segment_id().value();
        if (segment > manifest.base_segment_count) {
            throw CorruptionError("native base index references a non-base segment");
        }
        const std::uint64_t segment_rows =
            native_base_segment_row_count(manifest, segment);
        if (row_ref.row_index() >= segment_rows) {
            throw CorruptionError("native base index RowRef is outside its segment");
        }
        const std::uint64_t ordinal = checked_sum<CorruptionError>(
            checked_product<CorruptionError>(
                static_cast<std::uint64_t>(segment - 1),
                manifest.records_per_segment,
                "native RowRef ordinal"),
            row_ref.row_index(), "native RowRef ordinal");
        if (ordinal >= manifest.base_record_count) {
            throw CorruptionError("native base index RowRef is outside base rows");
        }
        const auto word = static_cast<std::size_t>(ordinal / 64);
        const std::uint64_t bit = std::uint64_t{1} << (ordinal % 64);
        if ((seen_rows[word] & bit) != 0) {
            throw CorruptionError("native index RowRefs are not a bijection");
        }
        seen_rows[word] |= bit;

        std::uint64_t probe_position = hash_id(slot.id, index.hash_seed) & mask;
        bool reached = false;
        for (std::uint64_t probe = 0; probe != index.capacity; ++probe) {
            const auto candidate = static_cast<std::size_t>(probe_position);
            if (index.controls[candidate] == 0) {
                break;
            }
            if (probe_position == position) {
                reached = true;
                break;
            }
            if (index.slots[candidate].id == slot.id) {
                throw CorruptionError("native index contains a duplicate ID");
            }
            probe_position = (probe_position + 1) & mask;
        }
        if (!reached) {
            throw CorruptionError("native index contains a probe-unreachable slot");
        }
    }
    if (occupied != manifest.base_record_count) {
        throw CorruptionError("native index occupancy does not match manifest");
    }
}

[[nodiscard]] std::uint32_t native_index_body_checksum(
    const NativeManifest& manifest,
    const IndexData& index,
    const std::span<const std::byte> alignment_padding = {}) {
    Crc32c checksum;
    checksum.update(std::as_bytes(std::span(index.controls)));
    const std::uint64_t controls_end = checked_sum<InvalidArgumentError>(
        manifest.index_control_offset, manifest.index_capacity,
        "native index controls");
    const auto padding_size = static_cast<std::size_t>(
        manifest.index_slots_offset - controls_end);
    if (alignment_padding.empty()) {
        checksum.update_zeroes(padding_size);
    } else {
        if (alignment_padding.size() != padding_size) {
            throw InvalidArgumentError(
                "native index checksum padding has the wrong size");
        }
        checksum.update(alignment_padding);
    }

    if constexpr (std::endian::native == std::endian::little) {
        checksum.update(std::as_bytes(std::span(index.slots)));
    } else {
        std::vector<std::byte> encoded(
            kIndexIoEntries * kNativeIndexSlotSize);
        std::uint64_t position = 0;
        while (position != index.capacity) {
            const std::uint64_t batch_count = std::min<std::uint64_t>(
                kIndexIoEntries, index.capacity - position);
            const std::size_t batch_bytes = static_cast<std::size_t>(
                batch_count * kNativeIndexSlotSize);
            std::span<std::byte> bytes(encoded.data(), batch_bytes);
            for (std::uint64_t item = 0; item != batch_count; ++item) {
                const IndexSlot& slot = index.slots[
                    static_cast<std::size_t>(position + item)];
                const std::size_t item_offset = static_cast<std::size_t>(
                    item * kNativeIndexSlotSize);
                put_u64(bytes, item_offset, slot.id);
                put_u64(bytes, item_offset + 8, slot.ordinal);
            }
            checksum.update(bytes);
            position += batch_count;
        }
    }
    return checksum.value();
}

}  // namespace

std::uint64_t choose_native_index_capacity(const std::uint64_t record_count) {
    std::uint64_t capacity = 8;
    while (record_count
           > (capacity / 10) * 7 + ((capacity % 10) * 7) / 10) {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / 2) {
            throw InvalidArgumentError("native index is too large");
        }
        capacity *= 2;
    }
    if (capacity > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("native index does not fit address space");
    }
    return capacity;
}

void derive_native_manifest_geometry(NativeManifest& manifest) {
    require_nonempty_uuid<InvalidArgumentError>(
        manifest.collection_id, "collection UUID");
    require_nonempty_uuid<InvalidArgumentError>(manifest.schema_id, "schema UUID");
    require_nonempty_uuid<InvalidArgumentError>(
        manifest.active_wal_id, "active WAL UUID");
    if (manifest.schema.dtype != DTypeCode::Float16
        && manifest.schema.dtype != DTypeCode::Float32) {
        throw InvalidArgumentError("native manifest has unsupported dtype");
    }
    if (manifest.schema.dimension == 0) {
        throw InvalidArgumentError("native manifest has zero dimension");
    }
    manifest.schema.item_size = dtype_item_size(manifest.schema.dtype);
    manifest.schema.id_type = IdType::UInt64;
    manifest.schema.row_layout = RowLayout::DenseRowMajor;
    manifest.schema.index_kind = IndexKind::UInt64Flat;
    manifest.schema.hash_algorithm = HashAlgorithm::SplitMix64V1;
    manifest.record_size = checked_product<InvalidArgumentError>(
        manifest.schema.dimension, manifest.schema.item_size,
        "native record size");
    if (manifest.segment_payload_capacity < manifest.record_size) {
        throw InvalidArgumentError("segment payload capacity cannot fit one row");
    }
    manifest.records_per_segment =
        manifest.segment_payload_capacity / manifest.record_size;
    if (manifest.records_per_segment > RowRef::kMaximumRowsPerSegment) {
        throw InvalidArgumentError("segment contains more than 2^32 rows");
    }
    manifest.base_segment_count =
        manifest.base_record_count / manifest.records_per_segment
        + (manifest.base_record_count % manifest.records_per_segment == 0 ? 0 : 1);
    if (manifest.base_segment_count > 0xfffffffcULL) {
        throw InvalidArgumentError(
            "base segments exhaust active/next segment ID space");
    }
    manifest.index_capacity =
        choose_native_index_capacity(manifest.base_record_count);
    manifest.index_control_offset = kNativeHeaderSize;
    manifest.index_slots_offset = checked_align<InvalidArgumentError>(
        checked_sum<InvalidArgumentError>(
            manifest.index_control_offset, manifest.index_capacity,
            "native index controls"),
        kNativeHeaderSize, "native index slots");
    manifest.index_slot_size = kNativeIndexSlotSize;
    manifest.index_file_size = checked_sum<InvalidArgumentError>(
        manifest.index_slots_offset,
        checked_product<InvalidArgumentError>(
            manifest.index_capacity, manifest.index_slot_size,
            "native index slots"),
        "native index file size");
    manifest.active_wal_segment_id =
        static_cast<std::uint32_t>(manifest.base_segment_count + 1);
    manifest.next_segment_id = manifest.active_wal_segment_id + 1;
    validate_manifest<InvalidArgumentError>(manifest);
}

std::uint64_t native_base_segment_row_count(
    const NativeManifest& manifest, const std::uint32_t segment_id) {
    validate_manifest<InvalidArgumentError>(manifest);
    if (segment_id == 0 || segment_id > manifest.base_segment_count) {
        throw InvalidArgumentError("segment ID is not a base segment");
    }
    const std::uint64_t first = checked_product<InvalidArgumentError>(
        static_cast<std::uint64_t>(segment_id - 1),
        manifest.records_per_segment, "base segment first row");
    return std::min(
        manifest.records_per_segment, manifest.base_record_count - first);
}

std::array<std::byte, kNativeHeaderSize> encode_native_manifest(
    const NativeManifest& manifest) {
    validate_manifest<InvalidArgumentError>(manifest);
    std::array<std::byte, kNativeHeaderSize> output{};
    encode_common(output, kNativeManifestMagic);
    put_u64(output, 24, manifest.collection_id.low);
    put_u64(output, 32, manifest.collection_id.high);
    put_u64(output, 40, manifest.schema_id.low);
    put_u64(output, 48, manifest.schema_id.high);
    put_u64(output, 56, manifest.active_wal_id.low);
    put_u64(output, 64, manifest.active_wal_id.high);
    put_u32(output, 72, static_cast<std::uint32_t>(manifest.schema.dtype));
    put_u32(output, 76, manifest.schema.dimension);
    put_u32(output, 80, manifest.schema.item_size);
    put_u32(output, 84, static_cast<std::uint32_t>(manifest.schema.id_type));
    put_u32(output, 88, static_cast<std::uint32_t>(manifest.schema.row_layout));
    put_u32(output, 92, static_cast<std::uint32_t>(manifest.schema.index_kind));
    put_u32(
        output, 96,
        static_cast<std::uint32_t>(manifest.schema.hash_algorithm));
    put_u64(output, 104, manifest.record_size);
    put_u64(output, 112, manifest.base_record_count);
    put_u64(output, 120, manifest.segment_payload_capacity);
    put_u64(output, 128, manifest.records_per_segment);
    put_u64(output, 136, manifest.base_segment_count);
    put_u64(output, 144, manifest.index_capacity);
    put_u64(output, 152, manifest.index_control_offset);
    put_u64(output, 160, manifest.index_slots_offset);
    put_u64(output, 168, manifest.index_slot_size);
    put_u64(output, 176, manifest.index_file_size);
    put_u64(output, 184, manifest.hash_seed);
    put_u32(output, 192, manifest.active_wal_segment_id);
    put_u32(output, 196, manifest.next_segment_id);
    finish_header_crc(output);
    return output;
}

NativeManifest decode_native_manifest(const std::span<const std::byte> bytes) {
    decode_common(bytes, kNativeManifestMagic, "native manifest");
    NativeManifest result;
    result.collection_id = {get_u64(bytes, 24), get_u64(bytes, 32)};
    result.schema_id = {get_u64(bytes, 40), get_u64(bytes, 48)};
    result.active_wal_id = {get_u64(bytes, 56), get_u64(bytes, 64)};
    result.schema.dtype = static_cast<DTypeCode>(get_u32(bytes, 72));
    result.schema.dimension = get_u32(bytes, 76);
    result.schema.item_size = get_u32(bytes, 80);
    result.schema.id_type = static_cast<IdType>(get_u32(bytes, 84));
    result.schema.row_layout = static_cast<RowLayout>(get_u32(bytes, 88));
    result.schema.index_kind = static_cast<IndexKind>(get_u32(bytes, 92));
    result.schema.hash_algorithm =
        static_cast<HashAlgorithm>(get_u32(bytes, 96));
    if (get_u32(bytes, 100) != 0) {
        throw CorruptionError("native manifest has nonzero reserved schema flags");
    }
    result.record_size = get_u64(bytes, 104);
    result.base_record_count = get_u64(bytes, 112);
    result.segment_payload_capacity = get_u64(bytes, 120);
    result.records_per_segment = get_u64(bytes, 128);
    result.base_segment_count = get_u64(bytes, 136);
    result.index_capacity = get_u64(bytes, 144);
    result.index_control_offset = get_u64(bytes, 152);
    result.index_slots_offset = get_u64(bytes, 160);
    result.index_slot_size = get_u64(bytes, 168);
    result.index_file_size = get_u64(bytes, 176);
    result.hash_seed = get_u64(bytes, 184);
    result.active_wal_segment_id = get_u32(bytes, 192);
    result.next_segment_id = get_u32(bytes, 196);
    validate_manifest<CorruptionError>(result);
    require_reserved_zero(bytes, kManifestKnownBytes, "native manifest");
    return result;
}

NativeDataHeader make_native_data_header(
    const NativeManifest& manifest, const std::uint32_t segment_id) {
    const std::uint64_t row_count =
        native_base_segment_row_count(manifest, segment_id);
    NativeDataHeader result;
    result.collection_id = manifest.collection_id;
    result.schema_id = manifest.schema_id;
    result.segment_id = segment_id;
    result.dtype = manifest.schema.dtype;
    result.dimension = manifest.schema.dimension;
    result.item_size = manifest.schema.item_size;
    result.id_type = manifest.schema.id_type;
    result.row_layout = manifest.schema.row_layout;
    result.record_size = manifest.record_size;
    result.row_count = row_count;
    result.payload_size = checked_product<InvalidArgumentError>(
        row_count, result.record_size, "data segment payload");
    result.file_size = checked_sum<InvalidArgumentError>(
        result.payload_offset, result.payload_size, "data segment file size");
    validate_data_header<InvalidArgumentError>(result);
    return result;
}

std::array<std::byte, kNativeHeaderSize> encode_native_data_header(
    const NativeDataHeader& header) {
    validate_data_header<InvalidArgumentError>(header);
    std::array<std::byte, kNativeHeaderSize> output{};
    encode_common(output, kNativeDataMagic);
    put_u64(output, 24, header.collection_id.low);
    put_u64(output, 32, header.collection_id.high);
    put_u64(output, 40, header.schema_id.low);
    put_u64(output, 48, header.schema_id.high);
    put_u32(output, 56, header.segment_id);
    put_u32(output, 60, static_cast<std::uint32_t>(header.dtype));
    put_u32(output, 64, header.dimension);
    put_u32(output, 68, header.item_size);
    put_u32(output, 72, static_cast<std::uint32_t>(header.id_type));
    put_u32(output, 76, static_cast<std::uint32_t>(header.row_layout));
    put_u64(output, 80, header.record_size);
    put_u64(output, 88, header.row_count);
    put_u64(output, 96, header.payload_offset);
    put_u64(output, 104, header.payload_size);
    put_u64(output, 112, header.file_size);
    finish_header_crc(output);
    return output;
}

NativeDataHeader decode_native_data_header(
    const std::span<const std::byte> bytes) {
    decode_common(bytes, kNativeDataMagic, "native data segment");
    NativeDataHeader result;
    result.collection_id = {get_u64(bytes, 24), get_u64(bytes, 32)};
    result.schema_id = {get_u64(bytes, 40), get_u64(bytes, 48)};
    result.segment_id = get_u32(bytes, 56);
    result.dtype = static_cast<DTypeCode>(get_u32(bytes, 60));
    result.dimension = get_u32(bytes, 64);
    result.item_size = get_u32(bytes, 68);
    result.id_type = static_cast<IdType>(get_u32(bytes, 72));
    result.row_layout = static_cast<RowLayout>(get_u32(bytes, 76));
    result.record_size = get_u64(bytes, 80);
    result.row_count = get_u64(bytes, 88);
    result.payload_offset = get_u64(bytes, 96);
    result.payload_size = get_u64(bytes, 104);
    result.file_size = get_u64(bytes, 112);
    validate_data_header<CorruptionError>(result);
    require_reserved_zero(bytes, kDataKnownBytes, "native data segment");
    return result;
}

NativeIdsHeader make_native_ids_header(
    const NativeManifest& manifest, const std::uint32_t segment_id) {
    const std::uint64_t row_count =
        native_base_segment_row_count(manifest, segment_id);
    NativeIdsHeader result;
    result.collection_id = manifest.collection_id;
    result.schema_id = manifest.schema_id;
    result.segment_id = segment_id;
    result.id_type = manifest.schema.id_type;
    result.row_count = row_count;
    result.payload_size = checked_product<InvalidArgumentError>(
        row_count, result.entry_size, "IDs sidecar payload");
    result.file_size = checked_sum<InvalidArgumentError>(
        result.payload_offset, result.payload_size, "IDs sidecar file size");
    validate_ids_header<InvalidArgumentError>(result);
    return result;
}

std::array<std::byte, kNativeHeaderSize> encode_native_ids_header(
    const NativeIdsHeader& header) {
    validate_ids_header<InvalidArgumentError>(header);
    std::array<std::byte, kNativeHeaderSize> output{};
    encode_common(output, kNativeIdsMagic);
    put_u64(output, 24, header.collection_id.low);
    put_u64(output, 32, header.collection_id.high);
    put_u64(output, 40, header.schema_id.low);
    put_u64(output, 48, header.schema_id.high);
    put_u32(output, 56, header.segment_id);
    put_u32(output, 60, static_cast<std::uint32_t>(header.id_type));
    put_u64(output, 64, header.row_count);
    put_u64(output, 72, header.payload_offset);
    put_u64(output, 80, header.payload_size);
    put_u64(output, 88, header.file_size);
    put_u32(output, 96, header.entry_size);
    finish_header_crc(output);
    return output;
}

NativeIdsHeader decode_native_ids_header(
    const std::span<const std::byte> bytes) {
    decode_common(bytes, kNativeIdsMagic, "native IDs sidecar");
    NativeIdsHeader result;
    result.collection_id = {get_u64(bytes, 24), get_u64(bytes, 32)};
    result.schema_id = {get_u64(bytes, 40), get_u64(bytes, 48)};
    result.segment_id = get_u32(bytes, 56);
    result.id_type = static_cast<IdType>(get_u32(bytes, 60));
    result.row_count = get_u64(bytes, 64);
    result.payload_offset = get_u64(bytes, 72);
    result.payload_size = get_u64(bytes, 80);
    result.file_size = get_u64(bytes, 88);
    result.entry_size = get_u32(bytes, 96);
    if (get_u32(bytes, 100) != 0) {
        throw CorruptionError("native IDs sidecar has nonzero reserved flags");
    }
    validate_ids_header<CorruptionError>(result);
    require_reserved_zero(bytes, kIdsKnownBytes, "native IDs sidecar");
    return result;
}

NativeIndexHeader make_native_index_header(const NativeManifest& manifest) {
    validate_manifest<InvalidArgumentError>(manifest);
    NativeIndexHeader result;
    result.collection_id = manifest.collection_id;
    result.schema_id = manifest.schema_id;
    result.record_count = manifest.base_record_count;
    result.capacity = manifest.index_capacity;
    result.hash_seed = manifest.hash_seed;
    result.control_offset = manifest.index_control_offset;
    result.slots_offset = manifest.index_slots_offset;
    result.slot_size = manifest.index_slot_size;
    result.file_size = manifest.index_file_size;
    result.base_segment_count = manifest.base_segment_count;
    validate_index_header_fields<InvalidArgumentError>(result);
    return result;
}

std::array<std::byte, kNativeHeaderSize> encode_native_index_header(
    const NativeIndexHeader& header) {
    validate_index_header_fields<InvalidArgumentError>(header);
    std::array<std::byte, kNativeHeaderSize> output{};
    encode_common(output, kNativeIndexMagic);
    put_u64(output, 24, header.collection_id.low);
    put_u64(output, 32, header.collection_id.high);
    put_u64(output, 40, header.schema_id.low);
    put_u64(output, 48, header.schema_id.high);
    put_u64(output, 56, header.record_count);
    put_u64(output, 64, header.capacity);
    put_u64(output, 72, header.hash_seed);
    put_u64(output, 80, header.control_offset);
    put_u64(output, 88, header.slots_offset);
    put_u64(output, 96, header.slot_size);
    put_u64(output, 104, header.file_size);
    put_u32(output, 112, header.control_encoding);
    put_u32(output, 116, header.value_encoding);
    put_u64(output, 120, header.base_segment_count);
    put_u32(output, 128, header.body_checksum_algorithm);
    put_u32(output, 132, header.body_checksum);
    finish_header_crc(output);
    return output;
}

NativeIndexHeader decode_native_index_header(
    const std::span<const std::byte> bytes) {
    decode_common(bytes, kNativeIndexMagic, "native index");
    NativeIndexHeader result;
    result.collection_id = {get_u64(bytes, 24), get_u64(bytes, 32)};
    result.schema_id = {get_u64(bytes, 40), get_u64(bytes, 48)};
    result.record_count = get_u64(bytes, 56);
    result.capacity = get_u64(bytes, 64);
    result.hash_seed = get_u64(bytes, 72);
    result.control_offset = get_u64(bytes, 80);
    result.slots_offset = get_u64(bytes, 88);
    result.slot_size = get_u64(bytes, 96);
    result.file_size = get_u64(bytes, 104);
    result.control_encoding = get_u32(bytes, 112);
    result.value_encoding = get_u32(bytes, 116);
    result.base_segment_count = get_u64(bytes, 120);
    result.body_checksum_algorithm = get_u32(bytes, 128);
    result.body_checksum = get_u32(bytes, 132);
    validate_index_header_fields<CorruptionError>(result);
    require_reserved_zero(bytes, kIndexKnownBytes, "native index");
    return result;
}

std::filesystem::path native_manifest_path(
    const std::filesystem::path& store_path) {
    return store_path / "manifest.bin";
}

std::filesystem::path native_data_segment_path(
    const std::filesystem::path& store_path, const std::uint32_t segment_id) {
    static_cast<void>(SegmentId::create(segment_id));
    return store_path / "data"
           / ("segment-" + std::to_string(segment_id) + ".dat");
}

std::filesystem::path native_ids_segment_path(
    const std::filesystem::path& store_path, const std::uint32_t segment_id) {
    static_cast<void>(SegmentId::create(segment_id));
    return store_path / "ids"
           / ("segment-" + std::to_string(segment_id) + ".ids");
}

std::filesystem::path native_index_path(
    const std::filesystem::path& store_path) {
    return store_path / "index" / "index.bin";
}

std::filesystem::path native_active_wal_path(
    const std::filesystem::path& store_path) {
    return store_path / "wal" / "active.wal";
}

NativeManifest read_native_manifest(
    const std::filesystem::path& store_path) {
    const auto path = native_manifest_path(store_path);
    auto fd = open_read_only(path);
    if (file_size(fd.get(), path) != kNativeHeaderSize) {
        throw CorruptionError("native manifest file size is not 4096 bytes");
    }
    std::array<std::byte, kNativeHeaderSize> bytes{};
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    return decode_native_manifest(bytes);
}

void write_native_manifest_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    const auto bytes = encode_native_manifest(manifest);
    write_header_atomic(
        store_path, store_path / "manifest.bin.tmp",
        native_manifest_path(store_path), bytes, "native manifest");
}

NativeDataHeader read_native_data_header(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const std::uint32_t segment_id) {
    const auto path = native_data_segment_path(store_path, segment_id);
    auto fd = open_read_only(path);
    const std::uint64_t actual_file_size = file_size(fd.get(), path);
    std::array<std::byte, kNativeHeaderSize> bytes{};
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    const NativeDataHeader result = decode_native_data_header(bytes);
    require_manifest_match(
        manifest, result.collection_id, result.schema_id, "data segment");
    const NativeDataHeader expected = make_native_data_header(manifest, segment_id);
    if (result.segment_id != expected.segment_id
        || result.dtype != expected.dtype
        || result.dimension != expected.dimension
        || result.item_size != expected.item_size
        || result.id_type != expected.id_type
        || result.row_layout != expected.row_layout
        || result.record_size != expected.record_size
        || result.row_count != expected.row_count
        || result.payload_offset != expected.payload_offset
        || result.payload_size != expected.payload_size
        || result.file_size != expected.file_size) {
        throw CorruptionError("data segment geometry does not match manifest");
    }
    if (actual_file_size != result.file_size) {
        throw CorruptionError("data segment file size does not match its header");
    }
    return result;
}

NativeIdsHeader read_native_ids_header(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const std::uint32_t segment_id) {
    const auto path = native_ids_segment_path(store_path, segment_id);
    auto fd = open_read_only(path);
    const std::uint64_t actual_file_size = file_size(fd.get(), path);
    std::array<std::byte, kNativeHeaderSize> bytes{};
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    const NativeIdsHeader result = decode_native_ids_header(bytes);
    require_manifest_match(
        manifest, result.collection_id, result.schema_id, "IDs sidecar");
    const NativeIdsHeader expected = make_native_ids_header(manifest, segment_id);
    if (result.segment_id != expected.segment_id
        || result.id_type != expected.id_type
        || result.row_count != expected.row_count
        || result.payload_offset != expected.payload_offset
        || result.payload_size != expected.payload_size
        || result.file_size != expected.file_size
        || result.entry_size != expected.entry_size) {
        throw CorruptionError("IDs sidecar geometry does not match manifest");
    }
    if (actual_file_size != result.file_size) {
        throw CorruptionError("IDs sidecar file size does not match its header");
    }
    return result;
}

IndexData build_native_index(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    validate_manifest<InvalidArgumentError>(manifest);
    if (manifest.index_capacity > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("native index does not fit address space");
    }
    IndexData index;
    index.count = manifest.base_record_count;
    index.capacity = manifest.index_capacity;
    index.hash_seed = manifest.hash_seed;
    index.controls.resize(static_cast<std::size_t>(index.capacity), 0);
    index.slots.resize(static_cast<std::size_t>(index.capacity));

    std::vector<std::byte> encoded(kIndexIoEntries * kNativeIdEntrySize);
    std::uint64_t total_rows = 0;
    for (std::uint64_t raw_segment = 1;
         raw_segment <= manifest.base_segment_count; ++raw_segment) {
        const auto segment_id = static_cast<std::uint32_t>(raw_segment);
        const NativeDataHeader data =
            read_native_data_header(store_path, manifest, segment_id);
        const NativeIdsHeader ids =
            read_native_ids_header(store_path, manifest, segment_id);
        if (data.row_count != ids.row_count) {
            throw CorruptionError("data and IDs segment row counts differ");
        }
        const auto path = native_ids_segment_path(store_path, segment_id);
        auto fd = open_read_only(path);
        std::uint64_t row = 0;
        while (row != ids.row_count) {
            const std::uint64_t batch_count = std::min<std::uint64_t>(
                kIndexIoEntries, ids.row_count - row);
            const std::size_t batch_bytes = static_cast<std::size_t>(
                batch_count * kNativeIdEntrySize);
            pread_exact(
                fd.get(), encoded.data(), batch_bytes,
                ids.payload_offset + row * kNativeIdEntrySize, path);
            const std::span<const std::byte> bytes(encoded.data(), batch_bytes);
            for (std::uint64_t item = 0; item != batch_count; ++item) {
                const std::uint64_t id = get_u64(
                    bytes, static_cast<std::size_t>(item * kNativeIdEntrySize));
                insert_native_index(
                    index, id, RowRef::create(segment_id, row + item));
            }
            row += batch_count;
        }
        total_rows += ids.row_count;
    }
    if (total_rows != manifest.base_record_count) {
        throw CorruptionError("base segment rows do not match manifest count");
    }
    return index;
}

void write_native_index_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const IndexData& index) {
    validate_manifest<InvalidArgumentError>(manifest);
    validate_index_data_for_write(manifest, index);
    // Do not make it possible for a caller of the lower-level atomic helper to
    // publish a table that the strict reader would immediately reject.
    validate_native_index_body(manifest, index);
    NativeIndexHeader index_header = make_native_index_header(manifest);
    index_header.body_checksum_algorithm =
        kNativeIndexBodyCrc32cAlgorithm;
    index_header.body_checksum = native_index_body_checksum(manifest, index);
    const auto directory = store_path / "index";
    const auto temporary = directory / "index.bin.tmp";
    const auto final_path = native_index_path(store_path);
    auto fd = create_exclusive(temporary);
    try {
        const auto header = encode_native_index_header(index_header);
        write_all(fd.get(), header.data(), header.size(), temporary);
        write_all(
            fd.get(), index.controls.data(), index.controls.size(), temporary);

        std::array<std::byte, kNativeHeaderSize> zeroes{};
        std::uint64_t offset =
            manifest.index_control_offset + manifest.index_capacity;
        while (offset != manifest.index_slots_offset) {
            const std::size_t amount = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    zeroes.size(), manifest.index_slots_offset - offset));
            write_all(fd.get(), zeroes.data(), amount, temporary);
            offset += amount;
        }

        if constexpr (std::endian::native == std::endian::little) {
            static_assert(sizeof(IndexSlot) == kNativeIndexSlotSize);
            write_all(
                fd.get(), index.slots.data(),
                index.slots.size() * sizeof(IndexSlot), temporary);
        } else {
            std::vector<std::byte> encoded(
                kIndexIoEntries * kNativeIndexSlotSize);
            std::uint64_t position = 0;
            while (position != index.capacity) {
                const std::uint64_t batch_count = std::min<std::uint64_t>(
                    kIndexIoEntries, index.capacity - position);
                const std::size_t batch_bytes = static_cast<std::size_t>(
                    batch_count * kNativeIndexSlotSize);
                std::span<std::byte> bytes(encoded.data(), batch_bytes);
                for (std::uint64_t item = 0; item != batch_count; ++item) {
                    const IndexSlot& slot = index.slots[
                        static_cast<std::size_t>(position + item)];
                    const std::size_t item_offset = static_cast<std::size_t>(
                        item * kNativeIndexSlotSize);
                    put_u64(bytes, item_offset, slot.id);
                    put_u64(bytes, item_offset + 8, slot.ordinal);
                }
                write_all(fd.get(), encoded.data(), batch_bytes, temporary);
                position += batch_count;
            }
        }
        sync_file(fd.get(), temporary);
        fd.reset();
        std::error_code error;
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            throw StorageError("cannot publish native index: " + error.message());
        }
        sync_directory(directory);
    } catch (...) {
        fd.reset();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void build_and_write_native_index_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    IndexData index = build_native_index(store_path, manifest);
    write_native_index_atomic(store_path, manifest, index);
}

IndexData read_native_index(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const NativeIndexValidationMode validation_mode,
    NativeIndexReadStats* const stats) {
    NativeIndexReadStats local_stats;
    NativeIndexReadStats& read_stats = stats == nullptr ? local_stats : *stats;
    read_stats = {};
    validate_manifest<CorruptionError>(manifest);
    if (manifest.index_capacity > std::numeric_limits<std::size_t>::max()
        || manifest.index_capacity
               > std::numeric_limits<std::size_t>::max() / sizeof(IndexSlot)) {
        throw CorruptionError("native index does not fit address space");
    }
    auto phase_started = SteadyClock::now();
    const auto path = native_index_path(store_path);
    auto fd = open_read_only(path);
    if (file_size(fd.get(), path) != manifest.index_file_size) {
        throw CorruptionError("native index file size does not match manifest");
    }
    std::array<std::byte, kNativeHeaderSize> header_bytes{};
    pread_exact(fd.get(), header_bytes.data(), header_bytes.size(), 0, path);
    const NativeIndexHeader header =
        decode_native_index_header(header_bytes);
    require_manifest_match(
        manifest, header.collection_id, header.schema_id, "native index");
    const NativeIndexHeader expected = make_native_index_header(manifest);
    if (header.record_count != expected.record_count
        || header.capacity != expected.capacity
        || header.hash_seed != expected.hash_seed
        || header.control_offset != expected.control_offset
        || header.slots_offset != expected.slots_offset
        || header.slot_size != expected.slot_size
        || header.file_size != expected.file_size
        || header.control_encoding != expected.control_encoding
        || header.value_encoding != expected.value_encoding
        || header.base_segment_count != expected.base_segment_count) {
        throw CorruptionError("native index header does not match manifest");
    }
    read_stats.header_seconds = elapsed_seconds(phase_started);
    read_stats.body_checksum_present = header.body_checksum_algorithm != 0;

    IndexData index;
    index.count = manifest.base_record_count;
    index.capacity = manifest.index_capacity;
    index.hash_seed = manifest.hash_seed;
    phase_started = SteadyClock::now();
    index.controls.resize(static_cast<std::size_t>(index.capacity));
    index.slots.resize(static_cast<std::size_t>(index.capacity));
    read_stats.allocation_seconds = elapsed_seconds(phase_started);
    phase_started = SteadyClock::now();
    pread_exact(
        fd.get(), index.controls.data(), index.controls.size(),
        manifest.index_control_offset, path);
    const std::uint64_t controls_end = manifest.index_control_offset
        + manifest.index_capacity;
    std::vector<std::byte> alignment_padding(static_cast<std::size_t>(
        manifest.index_slots_offset - controls_end));
    if (!alignment_padding.empty()) {
        pread_exact(
            fd.get(), alignment_padding.data(), alignment_padding.size(),
            controls_end, path);
        if (std::any_of(
                alignment_padding.begin(), alignment_padding.end(),
                [](const std::byte value) { return value != std::byte{0}; })) {
            throw CorruptionError(
                "native index alignment padding is not zero-filled");
        }
    }
    read_stats.controls_seconds = elapsed_seconds(phase_started);

    phase_started = SteadyClock::now();
    if constexpr (std::endian::native == std::endian::little) {
        static_assert(sizeof(IndexSlot) == kNativeIndexSlotSize);
        pread_exact(
            fd.get(), index.slots.data(),
            index.slots.size() * sizeof(IndexSlot),
            manifest.index_slots_offset, path);
    } else {
        std::vector<std::byte> encoded(
            kIndexIoEntries * kNativeIndexSlotSize);
        std::uint64_t position = 0;
        while (position != index.capacity) {
            const std::uint64_t batch_count = std::min<std::uint64_t>(
                kIndexIoEntries, index.capacity - position);
            const std::size_t batch_bytes = static_cast<std::size_t>(
                batch_count * kNativeIndexSlotSize);
            pread_exact(
                fd.get(), encoded.data(), batch_bytes,
                manifest.index_slots_offset
                    + position * kNativeIndexSlotSize,
                path);
            const std::span<const std::byte> bytes(encoded.data(), batch_bytes);
            for (std::uint64_t item = 0; item != batch_count; ++item) {
                const std::size_t item_offset = static_cast<std::size_t>(
                    item * kNativeIndexSlotSize);
                index.slots[static_cast<std::size_t>(position + item)] =
                    IndexSlot{
                        get_u64(bytes, item_offset),
                        get_u64(bytes, item_offset + 8)};
            }
            position += batch_count;
        }
    }
    read_stats.slots_seconds = elapsed_seconds(phase_started);

    if (header.body_checksum_algorithm != 0) {
        phase_started = SteadyClock::now();
        const std::uint32_t actual_checksum =
            native_index_body_checksum(
                manifest, index, alignment_padding);
        read_stats.checksum_seconds = elapsed_seconds(phase_started);
        if (actual_checksum != header.body_checksum) {
            throw CorruptionError("native index body checksum mismatch");
        }
        read_stats.body_checksum_verified = true;
    }

    const bool perform_structural_validation =
        validation_mode == NativeIndexValidationMode::Full
        || header.body_checksum_algorithm == 0;
    if (perform_structural_validation) {
        phase_started = SteadyClock::now();
        validate_native_index_body(manifest, index);
        read_stats.structural_validation_seconds =
            elapsed_seconds(phase_started);
        read_stats.structural_validation_performed = true;
    }
    return index;
}

std::optional<RowRef> native_index_lookup(
    const IndexData& index, const std::uint64_t id) {
    const std::uint64_t mask = index.capacity - 1;
    std::uint64_t position = hash_id(id, index.hash_seed) & mask;
    for (std::uint64_t probe = 0; probe != index.capacity; ++probe) {
        const auto slot = static_cast<std::size_t>(position);
        if (index.controls[slot] == 0) {
            return std::nullopt;
        }
        if (index.slots[slot].id == id) {
            return RowRef::decode(index.slots[slot].ordinal);
        }
        position = (position + 1) & mask;
    }
    return std::nullopt;
}

}  // namespace throwdb::detail
