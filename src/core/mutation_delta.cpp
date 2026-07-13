#include "mutation_delta.hpp"

#include "crc32c.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace throwdb::detail {
namespace {

constexpr std::array<char, 8> kDeltaDataMagic{
    'T', 'D', 'B', 'M', 'D', 'A', 'T', '1'};
constexpr std::array<char, 8> kDeltaIndexMagic{
    'T', 'D', 'B', 'M', 'D', 'I', 'X', '1'};
constexpr std::uint32_t kDeltaFormatVersion = 1;
constexpr std::uint32_t kChecksumAlgorithmCrc32c = 1;
constexpr std::size_t kHeaderChecksumOffset = 20;
constexpr std::size_t kDescriptorOffset = 56;
constexpr std::size_t kDataKnownHeaderBytes = 224;
constexpr std::size_t kIndexKnownHeaderBytes = 232;
constexpr std::size_t kIoBytes = 2U * 1024U * 1024U;

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

void put_magic(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    std::memcpy(bytes.data(), magic.data(), magic.size());
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

[[nodiscard]] std::uint32_t header_crc32c(
    const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes.first(kHeaderChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(
        kHeaderChecksumOffset + sizeof(std::uint32_t),
        static_cast<std::size_t>(kMutationDeltaHeaderSize)
            - kHeaderChecksumOffset - sizeof(std::uint32_t)));
    return checksum.value();
}

void encode_common(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    put_magic(bytes, magic);
    put_u32(bytes, 8, kDeltaFormatVersion);
    put_u32(bytes, 12, static_cast<std::uint32_t>(kMutationDeltaHeaderSize));
    put_u32(bytes, 16, kChecksumAlgorithmCrc32c);
}

void finish_header(const std::span<std::byte> bytes) {
    put_u32(bytes, kHeaderChecksumOffset, 0);
    put_u32(bytes, kHeaderChecksumOffset, header_crc32c(bytes));
}

void decode_common(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic,
    const std::string_view description) {
    require<CorruptionError>(
        bytes.size() == kMutationDeltaHeaderSize,
        std::string(description) + " header is not exactly 4096 bytes");
    require<CorruptionError>(
        std::memcmp(bytes.data(), magic.data(), magic.size()) == 0,
        std::string(description) + " has invalid magic");
    require<CorruptionError>(
        get_u32(bytes, 8) == kDeltaFormatVersion,
        std::string(description) + " has an unsupported format version");
    require<CorruptionError>(
        get_u32(bytes, 12) == kMutationDeltaHeaderSize,
        std::string(description) + " has an invalid header size");
    require<CorruptionError>(
        get_u32(bytes, 16) == kChecksumAlgorithmCrc32c,
        std::string(description) + " has an unsupported checksum algorithm");
    require<CorruptionError>(
        get_u32(bytes, kHeaderChecksumOffset) == header_crc32c(bytes),
        std::string(description) + " header checksum mismatch");
}

template <typename Error>
void require_zero_range(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::size_t end,
    const std::string_view description) {
    if (begin > end || end > bytes.size()) {
        throw Error(
            std::string(description) + " has invalid reserved geometry");
    }
    for (std::size_t offset = begin; offset != end; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw Error(
                std::string(description)
                + " contains nonzero reserved bytes");
        }
    }
}

void put_descriptor(
    const std::span<std::byte> bytes,
    const std::size_t offset,
    const MutationDeltaDescriptor& descriptor) {
    put_uuid(bytes, offset, descriptor.delta_id);
    put_uuid(bytes, offset + 16, descriptor.source_predecessor_wal_id);
    put_uuid(bytes, offset + 32, descriptor.source_terminal_wal_id);
    put_u64(bytes, offset + 48, descriptor.first_sequence);
    put_u64(bytes, offset + 56, descriptor.last_sequence);
    put_u32(bytes, offset + 64, descriptor.segment_id);
    put_u32(bytes, offset + 68, descriptor.source_wal_count);
    put_u64(bytes, offset + 72, descriptor.entry_count);
    put_u64(bytes, offset + 80, descriptor.put_count);
    put_u64(bytes, offset + 88, descriptor.data_file_size);
    put_u64(bytes, offset + 96, descriptor.index_file_size);
    put_u32(bytes, offset + 104, descriptor.data_payload_crc32c);
    put_u32(bytes, offset + 108, descriptor.index_body_crc32c);
}

[[nodiscard]] MutationDeltaDescriptor get_descriptor(
    const std::span<const std::byte> bytes,
    const std::size_t offset) {
    MutationDeltaDescriptor descriptor;
    descriptor.delta_id = get_uuid(bytes, offset);
    descriptor.source_predecessor_wal_id = get_uuid(bytes, offset + 16);
    descriptor.source_terminal_wal_id = get_uuid(bytes, offset + 32);
    descriptor.first_sequence = get_u64(bytes, offset + 48);
    descriptor.last_sequence = get_u64(bytes, offset + 56);
    descriptor.segment_id = get_u32(bytes, offset + 64);
    descriptor.source_wal_count = get_u32(bytes, offset + 68);
    descriptor.entry_count = get_u64(bytes, offset + 72);
    descriptor.put_count = get_u64(bytes, offset + 80);
    descriptor.data_file_size = get_u64(bytes, offset + 88);
    descriptor.index_file_size = get_u64(bytes, offset + 96);
    descriptor.data_payload_crc32c = get_u32(bytes, offset + 104);
    descriptor.index_body_crc32c = get_u32(bytes, offset + 108);
    return descriptor;
}

template <typename Error>
void validate_manifest_schema(const NativeManifest& manifest) {
    require<Error>(
        !manifest.collection_id.empty(), "manifest collection UUID is empty");
    require<Error>(
        !manifest.schema_id.empty(), "manifest schema UUID is empty");
    require<Error>(
        manifest.schema.dtype == DTypeCode::Float16
            || manifest.schema.dtype == DTypeCode::Float32,
        "manifest dtype is unsupported by mutation deltas");
    require<Error>(manifest.schema.dimension != 0, "manifest dimension is zero");
    require<Error>(
        manifest.schema.item_size == dtype_item_size(manifest.schema.dtype),
        "manifest dtype and item size disagree");
    require<Error>(
        manifest.schema.id_type == IdType::UInt64,
        "manifest ID type is unsupported by mutation deltas");
    require<Error>(
        manifest.schema.row_layout == RowLayout::DenseRowMajor,
        "manifest row layout is unsupported by mutation deltas");
    require<Error>(
        manifest.record_size
            == checked_product<Error>(
                manifest.schema.dimension, manifest.schema.item_size,
                "manifest record size"),
        "manifest record size does not match schema");
}

template <typename Error>
void validate_descriptor(
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    validate_manifest_schema<Error>(manifest);
    require<Error>(!descriptor.delta_id.empty(), "delta UUID is empty");
    require<Error>(
        !descriptor.source_terminal_wal_id.empty(),
        "delta source terminal WAL UUID is empty");
    require<Error>(
        descriptor.source_predecessor_wal_id
            != descriptor.source_terminal_wal_id,
        "delta source predecessor and terminal WAL are identical");
    require<Error>(
        descriptor.delta_id != descriptor.source_predecessor_wal_id
            && descriptor.delta_id != descriptor.source_terminal_wal_id,
        "delta UUID aliases its source WAL lineage");
    require<Error>(
        descriptor.first_sequence
                >= VersionedMutationValue::kMinimumCommitSequence
            && descriptor.first_sequence <= descriptor.last_sequence
            && descriptor.last_sequence
                   <= VersionedMutationValue::kMaximumCommitSequence,
        "delta source sequence interval is invalid");
    require<Error>(
        descriptor.segment_id > manifest.base_segment_count
            && descriptor.segment_id >= SegmentId::kMinimum
            && descriptor.segment_id <= SegmentId::kMaximum,
        "delta segment ID is invalid, reserved, or overlaps base data");
    require<Error>(
        descriptor.source_wal_count != 0,
        "delta source WAL count is zero");
    require<Error>(
        descriptor.put_count <= descriptor.entry_count,
        "delta PUT count exceeds entry count");
    require<Error>(
        descriptor.put_count <= RowRef::kMaximumRowsPerSegment,
        "delta PUT count exceeds RowRef capacity");

    const std::uint64_t payload_size = checked_product<Error>(
        descriptor.put_count, manifest.record_size, "delta payload size");
    require<Error>(
        descriptor.data_file_size
            == checked_sum<Error>(
                kMutationDeltaHeaderSize, payload_size,
                "delta data file size"),
        "delta data file size does not match row geometry");
    const std::uint64_t record_bytes = checked_product<Error>(
        descriptor.entry_count, kMutationDeltaVersionRecordSize,
        "delta version records");
    const std::uint64_t records_end = checked_sum<Error>(
        kMutationDeltaHeaderSize, record_bytes, "delta version records end");
    const std::uint64_t reverse_offset = checked_align<Error>(
        records_end, kMutationDeltaIndexAlignment,
        "delta reverse table offset");
    const std::uint64_t reverse_bytes = checked_product<Error>(
        descriptor.put_count, kMutationDeltaReverseRecordSize,
        "delta reverse records");
    require<Error>(
        descriptor.index_file_size
            == checked_sum<Error>(
                reverse_offset, reverse_bytes, "delta index file size"),
        "delta index file size does not match entry geometry");
}

[[nodiscard]] std::array<std::byte, kMutationDeltaHeaderSize>
encode_data_header(
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    validate_descriptor<InvalidArgumentError>(manifest, descriptor);
    std::array<std::byte, kMutationDeltaHeaderSize> bytes{};
    encode_common(bytes, kDeltaDataMagic);
    put_uuid(bytes, 24, manifest.collection_id);
    put_uuid(bytes, 40, manifest.schema_id);
    put_descriptor(bytes, kDescriptorOffset, descriptor);
    put_u64(bytes, 168, manifest.record_size);
    put_u32(bytes, 176, manifest.schema.dimension);
    put_u32(bytes, 180, static_cast<std::uint32_t>(manifest.schema.dtype));
    put_u32(bytes, 184, manifest.schema.item_size);
    put_u32(bytes, 188, static_cast<std::uint32_t>(manifest.schema.id_type));
    put_u32(bytes, 192, static_cast<std::uint32_t>(manifest.schema.row_layout));
    put_u32(bytes, 196, kChecksumAlgorithmCrc32c);
    put_u64(bytes, 200, kMutationDeltaHeaderSize);
    put_u64(
        bytes, 208, descriptor.data_file_size - kMutationDeltaHeaderSize);
    put_u64(bytes, 216, descriptor.data_file_size);
    finish_header(bytes);
    return bytes;
}

[[nodiscard]] std::array<std::byte, kMutationDeltaHeaderSize>
encode_index_header(
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    validate_descriptor<InvalidArgumentError>(manifest, descriptor);
    const std::uint64_t records_size = checked_product<InvalidArgumentError>(
        descriptor.entry_count, kMutationDeltaVersionRecordSize,
        "delta version records");
    const std::uint64_t reverse_offset = checked_align<InvalidArgumentError>(
        checked_sum<InvalidArgumentError>(
            kMutationDeltaHeaderSize, records_size,
            "delta version records end"),
        kMutationDeltaIndexAlignment, "delta reverse table offset");
    const std::uint64_t reverse_size = checked_product<InvalidArgumentError>(
        descriptor.put_count, kMutationDeltaReverseRecordSize,
        "delta reverse records");

    std::array<std::byte, kMutationDeltaHeaderSize> bytes{};
    encode_common(bytes, kDeltaIndexMagic);
    put_uuid(bytes, 24, manifest.collection_id);
    put_uuid(bytes, 40, manifest.schema_id);
    put_descriptor(bytes, kDescriptorOffset, descriptor);
    put_u32(bytes, 168, static_cast<std::uint32_t>(kMutationDeltaVersionRecordSize));
    put_u32(bytes, 172, static_cast<std::uint32_t>(kMutationDeltaReverseRecordSize));
    put_u32(bytes, 176, kChecksumAlgorithmCrc32c);
    put_u32(bytes, 180, descriptor.index_body_crc32c);
    put_u64(bytes, 184, kMutationDeltaHeaderSize);
    put_u64(bytes, 192, records_size);
    put_u64(bytes, 200, reverse_offset);
    put_u64(bytes, 208, reverse_size);
    put_u64(bytes, 216, descriptor.index_file_size);
    put_u64(bytes, 224, kMutationDeltaIndexAlignment);
    finish_header(bytes);
    return bytes;
}

[[nodiscard]] MutationDeltaDataHeader decode_data_header(
    const std::span<const std::byte> bytes) {
    decode_common(bytes, kDeltaDataMagic, "mutation delta data");
    require_zero_range<CorruptionError>(
        bytes, kDataKnownHeaderBytes, bytes.size(),
        "mutation delta data header");
    require<CorruptionError>(
        get_u32(bytes, 196) == kChecksumAlgorithmCrc32c,
        "mutation delta data has an unsupported payload checksum algorithm");

    MutationDeltaDataHeader header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor = get_descriptor(bytes, kDescriptorOffset);
    header.record_size = get_u64(bytes, 168);
    header.dimension = get_u32(bytes, 176);
    header.dtype = static_cast<DTypeCode>(get_u32(bytes, 180));
    header.item_size = get_u32(bytes, 184);
    header.id_type = static_cast<IdType>(get_u32(bytes, 188));
    header.row_layout = static_cast<RowLayout>(get_u32(bytes, 192));
    header.payload_offset = get_u64(bytes, 200);
    header.payload_size = get_u64(bytes, 208);
    require<CorruptionError>(
        get_u64(bytes, 216) == header.descriptor.data_file_size,
        "mutation delta data repeats a different file size");
    return header;
}

[[nodiscard]] MutationDeltaIndexHeader decode_index_header(
    const std::span<const std::byte> bytes) {
    decode_common(bytes, kDeltaIndexMagic, "mutation delta index");
    require_zero_range<CorruptionError>(
        bytes, kIndexKnownHeaderBytes, bytes.size(),
        "mutation delta index header");
    require<CorruptionError>(
        get_u32(bytes, 168) == kMutationDeltaVersionRecordSize,
        "mutation delta index has an invalid version-record size");
    require<CorruptionError>(
        get_u32(bytes, 172) == kMutationDeltaReverseRecordSize,
        "mutation delta index has an invalid reverse-record size");
    require<CorruptionError>(
        get_u32(bytes, 176) == kChecksumAlgorithmCrc32c,
        "mutation delta index has an unsupported body checksum algorithm");
    require<CorruptionError>(
        get_u64(bytes, 224) == kMutationDeltaIndexAlignment,
        "mutation delta index has an invalid reverse-table alignment");

    MutationDeltaIndexHeader header;
    header.collection_id = get_uuid(bytes, 24);
    header.schema_id = get_uuid(bytes, 40);
    header.descriptor = get_descriptor(bytes, kDescriptorOffset);
    require<CorruptionError>(
        get_u32(bytes, 180) == header.descriptor.index_body_crc32c,
        "mutation delta index repeats a different body checksum");
    header.records_offset = get_u64(bytes, 184);
    header.records_size = get_u64(bytes, 192);
    header.reverse_offset = get_u64(bytes, 200);
    header.reverse_size = get_u64(bytes, 208);
    require<CorruptionError>(
        get_u64(bytes, 216) == header.descriptor.index_file_size,
        "mutation delta index repeats a different file size");
    return header;
}

void validate_data_binding(
    const MutationDeltaDataHeader& header,
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    require<CorruptionError>(
        header.collection_id == manifest.collection_id,
        "mutation delta data collection UUID does not match manifest");
    require<CorruptionError>(
        header.schema_id == manifest.schema_id,
        "mutation delta data schema UUID does not match manifest");
    require<CorruptionError>(
        header.descriptor == descriptor,
        "mutation delta data descriptor does not match catalog");
    require<CorruptionError>(
        header.record_size == manifest.record_size
            && header.dimension == manifest.schema.dimension
            && header.dtype == manifest.schema.dtype
            && header.item_size == manifest.schema.item_size
            && header.id_type == manifest.schema.id_type
            && header.row_layout == manifest.schema.row_layout,
        "mutation delta data schema does not match manifest");
    require<CorruptionError>(
        header.payload_offset == kMutationDeltaHeaderSize
            && header.payload_size
                   == descriptor.data_file_size - kMutationDeltaHeaderSize,
        "mutation delta data payload geometry is invalid");
}

void validate_index_binding(
    const MutationDeltaIndexHeader& header,
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    require<CorruptionError>(
        header.collection_id == manifest.collection_id,
        "mutation delta index collection UUID does not match manifest");
    require<CorruptionError>(
        header.schema_id == manifest.schema_id,
        "mutation delta index schema UUID does not match manifest");
    require<CorruptionError>(
        header.descriptor == descriptor,
        "mutation delta index descriptor does not match catalog");
    const std::uint64_t expected_records = checked_product<CorruptionError>(
        descriptor.entry_count, kMutationDeltaVersionRecordSize,
        "delta version records");
    const std::uint64_t expected_reverse_offset =
        checked_align<CorruptionError>(
            checked_sum<CorruptionError>(
                kMutationDeltaHeaderSize, expected_records,
                "delta version records end"),
            kMutationDeltaIndexAlignment, "delta reverse table offset");
    const std::uint64_t expected_reverse = checked_product<CorruptionError>(
        descriptor.put_count, kMutationDeltaReverseRecordSize,
        "delta reverse records");
    require<CorruptionError>(
        header.records_offset == kMutationDeltaHeaderSize
            && header.records_size == expected_records
            && header.reverse_offset == expected_reverse_offset
            && header.reverse_size == expected_reverse,
        "mutation delta index body geometry is invalid");
}

[[nodiscard]] std::string uuid_filename_token(const Uuid128 id) {
    if (id.empty()) {
        throw InvalidArgumentError("delta file UUID is empty");
    }
    std::array<char, 33> token{};
    const int count = std::snprintf(
        token.data(), token.size(), "%016llx%016llx",
        static_cast<unsigned long long>(id.high),
        static_cast<unsigned long long>(id.low));
    if (count != 32) {
        throw StorageError("cannot format delta UUID-derived file name");
    }
    return std::string(token.data(), 32);
}

void ensure_delta_directory(const std::filesystem::path& store_path) {
    std::error_code error;
    const bool store_is_directory =
        std::filesystem::is_directory(store_path, error);
    if (error) {
        throw StorageError(
            "cannot inspect mutation delta store path: " + error.message());
    }
    require<InvalidArgumentError>(
        store_is_directory,
        "mutation delta store path is not a directory");
    const auto directory = mutation_deltas_directory(store_path);
    error.clear();
    const bool exists = std::filesystem::exists(directory, error);
    if (error) {
        throw StorageError(
            "cannot inspect mutation delta directory: " + error.message());
    }
    if (exists) {
        error.clear();
        require<StorageError>(
            std::filesystem::is_directory(directory, error) && !error,
            "mutation delta path exists but is not a directory");
        return;
    }
    error.clear();
    if (!std::filesystem::create_directory(directory, error) || error) {
        throw StorageError(
            "cannot create mutation delta directory: " + error.message());
    }
    sync_directory(store_path);
}

void require_destination_absent(const std::filesystem::path& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw StorageError(
            "cannot inspect immutable delta destination: " + error.message());
    }
    require<InvalidArgumentError>(
        !exists, "immutable mutation delta UUID already exists");
}

void rename_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    const std::string_view description) {
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error) {
        throw StorageError(
            "cannot publish " + std::string(description) + ": "
            + error.message());
    }
}

[[nodiscard]] std::array<std::byte, kMutationDeltaVersionRecordSize>
encode_version_record(
    const MutationDeltaBuildEntry& entry,
    const std::uint64_t payload_row) {
    std::array<std::byte, kMutationDeltaVersionRecordSize> bytes{};
    put_u64(bytes, 0, entry.id);
    put_u64(bytes, 8, entry.value.sequence());
    put_u64(
        bytes, 16,
        entry.value.is_put()
            ? payload_row
            : kMutationDeltaTombstonePayloadRow);
    bytes[24] = static_cast<std::byte>(entry.value.encoded_state());
    return bytes;
}

[[nodiscard]] std::array<std::byte, kMutationDeltaReverseRecordSize>
encode_reverse_record(const MutationDeltaBuildEntry& entry) {
    std::array<std::byte, kMutationDeltaReverseRecordSize> bytes{};
    put_u64(bytes, 0, entry.id);
    put_u64(bytes, 8, entry.value.sequence());
    return bytes;
}

void remove_if_present(const std::filesystem::path& path) noexcept {
    std::error_code ignored;
    static_cast<void>(std::filesystem::remove(path, ignored));
}

[[nodiscard]] std::array<std::byte, kMutationDeltaHeaderSize> read_header(
    const int fd,
    const std::filesystem::path& path) {
    std::array<std::byte, kMutationDeltaHeaderSize> bytes{};
    pread_exact(fd, bytes.data(), bytes.size(), 0, path);
    return bytes;
}

void verify_payload_crc(
    const int fd,
    const std::filesystem::path& path,
    const MutationDeltaDataHeader& header) {
    Crc32c checksum;
    std::vector<std::byte> bytes(kIoBytes);
    std::uint64_t offset = header.payload_offset;
    std::uint64_t remaining = header.payload_size;
    while (remaining != 0) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, bytes.size()));
        pread_exact(fd, bytes.data(), count, offset, path);
        checksum.update(std::span<const std::byte>(bytes).first(count));
        offset += count;
        remaining -= count;
    }
    require<CorruptionError>(
        checksum.value() == header.descriptor.data_payload_crc32c,
        "mutation delta payload checksum mismatch");
}

}  // namespace

std::filesystem::path mutation_deltas_directory(
    const std::filesystem::path& store_path) {
    return store_path / "deltas";
}

std::filesystem::path mutation_delta_data_path(
    const std::filesystem::path& store_path,
    const Uuid128 delta_id) {
    return mutation_deltas_directory(store_path)
           / ("delta-" + uuid_filename_token(delta_id) + ".dat");
}

std::filesystem::path mutation_delta_index_path(
    const std::filesystem::path& store_path,
    const Uuid128 delta_id) {
    return mutation_deltas_directory(store_path)
           / ("delta-" + uuid_filename_token(delta_id) + ".idx");
}

MutationDeltaDescriptor write_mutation_delta(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationDeltaBuildSpec& spec,
    const std::span<const MutationDeltaBuildEntry> entries,
    const MutationDeltaPayloadReader& payload_reader,
    const std::size_t payload_batch_bytes) {
    validate_manifest_schema<InvalidArgumentError>(manifest);
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));

    std::vector<MutationDeltaBuildEntry> sorted(entries.begin(), entries.end());
    std::sort(
        sorted.begin(), sorted.end(),
        [](const auto& left, const auto& right) { return left.id < right.id; });
    std::uint64_t put_count = 0;
    for (std::size_t index = 0; index != sorted.size(); ++index) {
        const MutationDeltaBuildEntry& entry = sorted[index];
        require<InvalidArgumentError>(
            index == 0 || sorted[index - 1].id != entry.id,
            "mutation delta contains a duplicate ID");
        require<InvalidArgumentError>(
            entry.value.sequence() >= spec.first_sequence
                && entry.value.sequence() <= spec.last_sequence,
            "mutation delta entry sequence is outside its source interval");
        if (entry.value.is_put()) {
            ++put_count;
            // Decode now, before creating any files.
            static_cast<void>(entry.value.row_ref().value());
        }
    }

    MutationDeltaDescriptor descriptor;
    descriptor.delta_id = spec.delta_id;
    descriptor.source_predecessor_wal_id = spec.source_predecessor_wal_id;
    descriptor.source_terminal_wal_id = spec.source_terminal_wal_id;
    descriptor.first_sequence = spec.first_sequence;
    descriptor.last_sequence = spec.last_sequence;
    descriptor.segment_id = spec.segment_id;
    descriptor.source_wal_count = spec.source_wal_count;
    descriptor.entry_count = static_cast<std::uint64_t>(sorted.size());
    descriptor.put_count = put_count;
    descriptor.data_file_size = checked_sum<InvalidArgumentError>(
        kMutationDeltaHeaderSize,
        checked_product<InvalidArgumentError>(
            put_count, manifest.record_size, "delta payload size"),
        "delta data file size");
    const std::uint64_t records_end = checked_sum<InvalidArgumentError>(
        kMutationDeltaHeaderSize,
        checked_product<InvalidArgumentError>(
            descriptor.entry_count, kMutationDeltaVersionRecordSize,
            "delta version records"),
        "delta version records end");
    const std::uint64_t reverse_offset = checked_align<InvalidArgumentError>(
        records_end, kMutationDeltaIndexAlignment,
        "delta reverse table offset");
    descriptor.index_file_size = checked_sum<InvalidArgumentError>(
        reverse_offset,
        checked_product<InvalidArgumentError>(
            put_count, kMutationDeltaReverseRecordSize,
            "delta reverse records"),
        "delta index file size");
    validate_descriptor<InvalidArgumentError>(manifest, descriptor);

    if (put_count != 0) {
        require<InvalidArgumentError>(
            static_cast<bool>(payload_reader),
            "mutation delta PUTs require a payload reader");
        if constexpr (
            std::numeric_limits<std::size_t>::max()
            < std::numeric_limits<std::uint64_t>::max()) {
            require<InvalidArgumentError>(
                manifest.record_size
                    <= std::numeric_limits<std::size_t>::max(),
                "mutation delta record size does not fit address space");
        }
        require<InvalidArgumentError>(
            payload_batch_bytes >= manifest.record_size,
            "mutation delta payload batch cannot fit one row");
    }

    ensure_delta_directory(store_path);
    const auto data_final = mutation_delta_data_path(store_path, spec.delta_id);
    const auto index_final = mutation_delta_index_path(store_path, spec.delta_id);
    const auto data_temporary =
        std::filesystem::path(data_final.string() + ".tmp");
    const auto index_temporary =
        std::filesystem::path(index_final.string() + ".tmp");
    require_destination_absent(data_final);
    require_destination_absent(index_final);

    FileDescriptor data_fd;
    FileDescriptor index_fd;
    bool owns_data_temporary = false;
    bool owns_index_temporary = false;
    try {
        data_fd = create_exclusive(data_temporary);
        owns_data_temporary = true;
        index_fd = create_exclusive(index_temporary);
        owns_index_temporary = true;
        const std::array<std::byte, kMutationDeltaHeaderSize> zero_header{};
        write_all(
            data_fd.get(), zero_header.data(), zero_header.size(),
            data_temporary);
        write_all(
            index_fd.get(), zero_header.data(), zero_header.size(),
            index_temporary);

        Crc32c data_checksum;
        if (put_count != 0) {
            const std::size_t record_size =
                static_cast<std::size_t>(manifest.record_size);
            const std::size_t rows_per_batch = std::max<std::size_t>(
                1, payload_batch_bytes / record_size);
            std::vector<RowRef> source_rows;
            source_rows.reserve(rows_per_batch);
            std::vector<std::byte> payload;
            for (const MutationDeltaBuildEntry& entry : sorted) {
                if (!entry.value.is_put()) {
                    continue;
                }
                source_rows.push_back(entry.value.row_ref().value());
                if (source_rows.size() != rows_per_batch) {
                    continue;
                }
                payload.resize(source_rows.size() * record_size);
                payload_reader(source_rows, payload);
                data_checksum.update(payload);
                write_all(
                    data_fd.get(), payload.data(), payload.size(),
                    data_temporary);
                source_rows.clear();
            }
            if (!source_rows.empty()) {
                payload.resize(source_rows.size() * record_size);
                payload_reader(source_rows, payload);
                data_checksum.update(payload);
                write_all(
                    data_fd.get(), payload.data(), payload.size(),
                    data_temporary);
            }
        }
        descriptor.data_payload_crc32c = data_checksum.value();

        Crc32c index_checksum;
        std::uint64_t payload_row = 0;
        constexpr std::size_t versions_per_batch =
            kIoBytes / kMutationDeltaVersionRecordSize;
        std::vector<std::byte> index_buffer;
        for (std::size_t begin = 0; begin != sorted.size();) {
            const std::size_t count = std::min(
                versions_per_batch, sorted.size() - begin);
            index_buffer.resize(
                count
                * static_cast<std::size_t>(
                    kMutationDeltaVersionRecordSize));
            for (std::size_t local = 0; local != count; ++local) {
                const MutationDeltaBuildEntry& entry = sorted[begin + local];
                const auto encoded =
                    encode_version_record(entry, payload_row);
                std::copy(
                    encoded.begin(), encoded.end(),
                    index_buffer.begin()
                        + static_cast<std::ptrdiff_t>(
                              local * kMutationDeltaVersionRecordSize));
                if (entry.value.is_put()) {
                    ++payload_row;
                }
            }
            index_checksum.update(index_buffer);
            write_all(
                index_fd.get(), index_buffer.data(), index_buffer.size(),
                index_temporary);
            begin += count;
        }
        require<StorageError>(
            payload_row == descriptor.put_count,
            "mutation delta PUT count changed while writing index");

        const std::uint64_t padding_size = reverse_offset - records_end;
        std::array<std::byte, kMutationDeltaIndexAlignment> zeroes{};
        std::uint64_t padding_remaining = padding_size;
        while (padding_remaining != 0) {
            const std::size_t count = static_cast<std::size_t>(
                std::min<std::uint64_t>(padding_remaining, zeroes.size()));
            const auto chunk = std::span<const std::byte>(zeroes).first(count);
            index_checksum.update(chunk);
            write_all(
                index_fd.get(), chunk.data(), chunk.size(), index_temporary);
            padding_remaining -= count;
        }
        constexpr std::size_t reverse_per_batch =
            kIoBytes / kMutationDeltaReverseRecordSize;
        std::size_t source_index = 0;
        while (source_index != sorted.size()) {
            index_buffer.clear();
            index_buffer.reserve(
                reverse_per_batch
                * static_cast<std::size_t>(
                    kMutationDeltaReverseRecordSize));
            while (source_index != sorted.size()
                   && index_buffer.size()
                          < reverse_per_batch
                                * kMutationDeltaReverseRecordSize) {
                const MutationDeltaBuildEntry& entry = sorted[source_index++];
                if (!entry.value.is_put()) {
                    continue;
                }
                const auto encoded = encode_reverse_record(entry);
                index_buffer.insert(
                    index_buffer.end(), encoded.begin(), encoded.end());
            }
            if (index_buffer.empty()) {
                continue;
            }
            index_checksum.update(index_buffer);
            write_all(
                index_fd.get(), index_buffer.data(), index_buffer.size(),
                index_temporary);
        }
        descriptor.index_body_crc32c = index_checksum.value();

        const auto data_header = encode_data_header(manifest, descriptor);
        const auto index_header = encode_index_header(manifest, descriptor);
        pwrite_all(
            data_fd.get(), data_header.data(), data_header.size(), 0,
            data_temporary);
        pwrite_all(
            index_fd.get(), index_header.data(), index_header.size(), 0,
            index_temporary);
        require<StorageError>(
            file_size(data_fd.get(), data_temporary)
                == descriptor.data_file_size,
            "mutation delta data writer produced an unexpected file size");
        require<StorageError>(
            file_size(index_fd.get(), index_temporary)
                == descriptor.index_file_size,
            "mutation delta index writer produced an unexpected file size");
        sync_file(data_fd.get(), data_temporary);
        sync_file(index_fd.get(), index_temporary);
        data_fd.reset();
        index_fd.reset();
        rename_file(data_temporary, data_final, "mutation delta data");
        owns_data_temporary = false;
        rename_file(index_temporary, index_final, "mutation delta index");
        owns_index_temporary = false;
        sync_directory(mutation_deltas_directory(store_path));
    } catch (...) {
        data_fd.reset();
        index_fd.reset();
        if (owns_data_temporary) {
            remove_if_present(data_temporary);
        }
        if (owns_index_temporary) {
            remove_if_present(index_temporary);
        }
        throw;
    }
    return descriptor;
}

OpenMutationDelta open_mutation_delta(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor,
    const bool verify_payload_checksum) {
    validate_descriptor<CorruptionError>(manifest, descriptor);
    const auto data_path = mutation_delta_data_path(store_path, descriptor.delta_id);
    const auto index_path = mutation_delta_index_path(store_path, descriptor.delta_id);
    FileDescriptor data_fd = open_read_only(data_path);
    require<CorruptionError>(
        file_size(data_fd.get(), data_path) == descriptor.data_file_size,
        "mutation delta data file size does not match catalog");
    const MutationDeltaDataHeader data_header =
        decode_data_header(read_header(data_fd.get(), data_path));
    validate_data_binding(data_header, manifest, descriptor);
    if (verify_payload_checksum) {
        verify_payload_crc(data_fd.get(), data_path, data_header);
    }

    FileDescriptor index_fd = open_read_only(index_path);
    require<CorruptionError>(
        file_size(index_fd.get(), index_path) == descriptor.index_file_size,
        "mutation delta index file size does not match catalog");
    const MutationDeltaIndexHeader index_header =
        decode_index_header(read_header(index_fd.get(), index_path));
    validate_index_binding(index_header, manifest, descriptor);
    if constexpr (
        std::numeric_limits<std::size_t>::max()
        < std::numeric_limits<std::uint64_t>::max()) {
        require<CorruptionError>(
            descriptor.entry_count
                <= std::numeric_limits<std::size_t>::max(),
            "mutation delta entry count does not fit address space");
    }

    std::vector<MutationDeltaVersionRecord> records;
    records.reserve(static_cast<std::size_t>(descriptor.entry_count));
    Crc32c body_checksum;
    constexpr std::size_t records_per_io =
        kIoBytes / kMutationDeltaVersionRecordSize;
    std::vector<std::byte> buffer(
        records_per_io * kMutationDeltaVersionRecordSize);
    std::uint64_t record_index = 0;
    std::uint64_t expected_put_row = 0;
    std::uint64_t offset = index_header.records_offset;
    while (record_index != descriptor.entry_count) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                descriptor.entry_count - record_index, records_per_io));
        const std::size_t byte_count =
            count * static_cast<std::size_t>(kMutationDeltaVersionRecordSize);
        pread_exact(index_fd.get(), buffer.data(), byte_count, offset, index_path);
        body_checksum.update(
            std::span<const std::byte>(buffer).first(byte_count));
        for (std::size_t local = 0; local != count; ++local) {
            const auto encoded = std::span<const std::byte>(buffer).subspan(
                local * kMutationDeltaVersionRecordSize,
                kMutationDeltaVersionRecordSize);
            const std::uint64_t id = get_u64(encoded, 0);
            const std::uint64_t sequence = get_u64(encoded, 8);
            const std::uint64_t payload_row = get_u64(encoded, 16);
            const std::uint8_t state = std::to_integer<std::uint8_t>(encoded[24]);
            require_zero_range<CorruptionError>(
                encoded, 25, encoded.size(), "mutation delta version record");
            require<CorruptionError>(
                records.empty() || records.back().id < id,
                "mutation delta version IDs are not strictly increasing");
            require<CorruptionError>(
                sequence >= descriptor.first_sequence
                    && sequence <= descriptor.last_sequence,
                "mutation delta version sequence is outside source interval");

            VersionedMutationValue value =
                VersionedMutationValue::tombstone(sequence);
            if (state == static_cast<std::uint8_t>(MutationValueState::Put)) {
                require<CorruptionError>(
                    payload_row == expected_put_row
                        && payload_row < descriptor.put_count,
                    "mutation delta payload rows are not dense and bijective");
                const RowRef output_row = RowRef::create(
                    SegmentId::decode(descriptor.segment_id), payload_row);
                value = VersionedMutationValue::decode(
                    sequence, state, output_row.raw());
                ++expected_put_row;
            } else if (
                state
                == static_cast<std::uint8_t>(MutationValueState::Tombstone)) {
                require<CorruptionError>(
                    payload_row == kMutationDeltaTombstonePayloadRow,
                    "mutation delta tombstone has a payload row");
                value = VersionedMutationValue::decode(sequence, state, 0);
            } else {
                throw CorruptionError(
                    "mutation delta version record has an invalid state");
            }
            records.push_back(MutationDeltaVersionRecord{id, value});
        }
        record_index += count;
        offset += byte_count;
    }
    require<CorruptionError>(
        expected_put_row == descriptor.put_count,
        "mutation delta PUT count does not match version records");

    std::uint64_t padding_remaining =
        index_header.reverse_offset
        - (index_header.records_offset + index_header.records_size);
    buffer.resize(kIoBytes);
    while (padding_remaining != 0) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(padding_remaining, buffer.size()));
        pread_exact(index_fd.get(), buffer.data(), count, offset, index_path);
        const auto chunk = std::span<const std::byte>(buffer).first(count);
        body_checksum.update(chunk);
        require_zero_range<CorruptionError>(
            chunk, 0, chunk.size(), "mutation delta index alignment padding");
        padding_remaining -= count;
        offset += count;
    }

    constexpr std::size_t reverse_per_io =
        kIoBytes / kMutationDeltaReverseRecordSize;
    buffer.resize(reverse_per_io * kMutationDeltaReverseRecordSize);
    std::uint64_t reverse_index = 0;
    std::size_t version_index = 0;
    while (reverse_index != descriptor.put_count) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                descriptor.put_count - reverse_index, reverse_per_io));
        const std::size_t byte_count =
            count * static_cast<std::size_t>(kMutationDeltaReverseRecordSize);
        pread_exact(index_fd.get(), buffer.data(), byte_count, offset, index_path);
        body_checksum.update(
            std::span<const std::byte>(buffer).first(byte_count));
        for (std::size_t local = 0; local != count; ++local) {
            while (version_index != records.size()
                   && records[version_index].value.is_tombstone()) {
                ++version_index;
            }
            require<CorruptionError>(
                version_index != records.size(),
                "mutation delta reverse table has too many rows");
            const auto encoded = std::span<const std::byte>(buffer).subspan(
                local * kMutationDeltaReverseRecordSize,
                kMutationDeltaReverseRecordSize);
            require<CorruptionError>(
                get_u64(encoded, 0) == records[version_index].id
                    && get_u64(encoded, 8)
                           == records[version_index].value.sequence(),
                "mutation delta reverse metadata does not match version index");
            ++version_index;
        }
        reverse_index += count;
        offset += byte_count;
    }
    while (version_index != records.size()
           && records[version_index].value.is_tombstone()) {
        ++version_index;
    }
    require<CorruptionError>(
        version_index == records.size(),
        "mutation delta reverse table is missing PUT rows");
    require<CorruptionError>(
        offset == descriptor.index_file_size,
        "mutation delta index decoder did not consume the exact file");
    require<CorruptionError>(
        body_checksum.value() == descriptor.index_body_crc32c,
        "mutation delta index body checksum mismatch");

    OpenMutationDelta result;
    result.descriptor = descriptor;
    result.data_header = data_header;
    result.data_fd = std::move(data_fd);
    result.records = std::move(records);
    return result;
}

}  // namespace throwdb::detail
