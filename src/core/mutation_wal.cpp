#include "mutation_wal.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#include "crc32c.hpp"

namespace throwdb::detail {
namespace {

constexpr std::array<char, 8> kWalMagic{
    'T', 'H', 'R', 'O', 'W', 'D', 'B', 'W'};
constexpr std::array<char, 8> kBatchMagic{
    'T', 'D', 'B', 'W', 'B', 'A', 'T', '1'};
constexpr std::array<char, 8> kCommitMagic{
    'T', 'D', 'B', 'W', 'C', 'O', 'M', '1'};
constexpr std::uint32_t kWalFormatVersion = 1;
constexpr std::uint32_t kBatchFrameVersion = 1;
constexpr std::uint32_t kCommitMarkerVersion = 1;
constexpr std::uint64_t kMaximumCommitSequence =
    std::numeric_limits<std::uint64_t>::max() - 1;

constexpr std::size_t kWalChecksumAlgorithmOffset = 124;
constexpr std::size_t kWalFlagsOffset = 128;
constexpr std::size_t kWalHeaderCrcOffset = 132;
constexpr std::size_t kWalReservedOffset = 160;

constexpr std::size_t kBatchHeaderCrcOffset = 108;
constexpr std::size_t kBatchFlagsOffset = 112;
constexpr std::size_t kBatchReservedOffset = 116;

constexpr std::size_t kCommitCrcOffset = 48;
constexpr std::size_t kCommitFlagsOffset = 52;
constexpr std::size_t kCommitReservedOffset = 56;

template <typename Error>
void require_value(const bool condition, const std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_add(
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
[[nodiscard]] std::uint64_t checked_align_up(
    const std::uint64_t value,
    const std::uint64_t alignment,
    const std::string_view description) {
    static_assert(std::is_base_of_v<std::exception, Error>);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw Error(std::string(description) + " has an invalid alignment");
    }
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

void put_magic(
    const std::span<std::byte> output,
    const std::array<char, 8>& magic) {
    std::memcpy(output.data(), magic.data(), magic.size());
}

template <typename Error>
void require_magic(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& expected,
    const std::string_view description) {
    require_value<Error>(
        bytes.size() >= expected.size()
            && std::memcmp(bytes.data(), expected.data(), expected.size()) == 0,
        std::string(description) + " has invalid magic");
}

void put_uuid(
    const std::span<std::byte> output,
    const std::size_t offset,
    const Uuid128 value) {
    put_u64(output, offset, value.low);
    put_u64(output, offset + 8, value.high);
}

[[nodiscard]] Uuid128 get_uuid(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    return Uuid128{get_u64(input, offset), get_u64(input, offset + 8)};
}

template <typename Error>
void require_zero_range(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::size_t end,
    const std::string_view description) {
    require_value<Error>(
        begin <= end && end <= bytes.size(),
        std::string(description) + " has an invalid reserved range");
    for (std::size_t offset = begin; offset != end; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw Error(std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

[[nodiscard]] std::uint32_t crc32c_with_zeroed_field(
    const std::span<const std::byte> bytes,
    const std::size_t field_offset,
    const std::size_t field_size) noexcept {
    Crc32c checksum;
    checksum.update(bytes.first(field_offset));
    checksum.update_zeroes(field_size);
    checksum.update(bytes.subspan(field_offset + field_size));
    return checksum.value();
}

template <typename Error>
void validate_limits(const WalLimits& limits) {
    require_value<Error>(
        limits.max_operations != 0,
        "WAL max_operations must be nonzero");
    require_value<Error>(
        limits.max_frame_size >= kWalBatchHeaderSize + kWalOperationEntrySize,
        "WAL max_frame_size is too small for one operation");
    require_value<Error>(
        limits.max_operations <= kWalFormatMaxOperations,
        "WAL max_operations exceeds the immutable format maximum");
    require_value<Error>(
        limits.max_frame_size <= kWalFormatMaxFrameSize,
        "WAL max_frame_size exceeds the immutable format maximum");
}

template <typename Error>
void validate_wal_header_value(const WalHeader& header) {
    require_value<Error>(
        !header.collection_id.empty(), "WAL collection ID is empty");
    require_value<Error>(!header.schema_id.empty(), "WAL schema ID is empty");
    require_value<Error>(!header.wal_id.empty(), "WAL file ID is empty");
    require_value<Error>(
        header.predecessor_wal_id.empty()
            || header.predecessor_wal_id != header.wal_id,
        "WAL file names itself as its predecessor");
    require_value<Error>(
        header.previous_sequence < kMaximumCommitSequence,
        "WAL previous sequence leaves no usable commit sequence");
    require_value<Error>(
        header.first_sequence == header.previous_sequence + 1,
        "WAL first sequence does not follow previous sequence");
    require_value<Error>(
        header.dimension != 0, "WAL schema has zero dimension");
    require_value<Error>(
        header.dtype == DTypeCode::Float16
            || header.dtype == DTypeCode::Float32,
        "WAL schema has an unsupported dtype");
    require_value<Error>(
        header.item_size == dtype_item_size(header.dtype),
        "WAL dtype and item size do not agree");
    require_value<Error>(
        header.id_type == IdType::UInt64,
        "WAL has an unsupported ID type");
    require_value<Error>(
        header.row_layout == RowLayout::DenseRowMajor,
        "WAL has an unsupported row layout");
    const std::uint64_t expected_record_size = checked_product<Error>(
        header.dimension, header.item_size, "WAL record size");
    require_value<Error>(
        header.record_size == expected_record_size,
        "WAL record size does not match its schema");
    require_value<Error>(
        header.required_feature_bits == 0
            && header.optional_feature_bits == 0,
        "WAL contains unsupported feature bits");
}

struct BatchHeader {
    std::uint64_t sequence{0};
    std::uint64_t operation_count{0};
    std::uint64_t put_count{0};
    std::uint64_t record_size{0};
    std::uint64_t operations_offset{0};
    std::uint64_t operations_size{0};
    std::uint64_t payload_offset{0};
    std::uint64_t payload_size{0};
    std::uint64_t frame_size{0};
    Uuid128 wal_id;
};

[[nodiscard]] BatchHeader decode_batch_header(
    const std::span<const std::byte> bytes,
    const WalHeader& wal_header) {
    require_value<CorruptionError>(
        bytes.size() == kWalBatchHeaderSize,
        "WAL batch header has the wrong size");
    require_magic<CorruptionError>(bytes, kBatchMagic, "WAL batch");
    require_value<CorruptionError>(
        get_u32(bytes, 8) == kBatchFrameVersion,
        "WAL batch has an unsupported frame version");
    require_value<CorruptionError>(
        get_u32(bytes, 12) == kWalBatchHeaderSize,
        "WAL batch has an invalid header size");
    require_value<CorruptionError>(
        get_u32(bytes, 104)
            == static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c),
        "WAL batch has an unsupported checksum algorithm");
    const std::uint32_t stored_header_crc = get_u32(bytes, kBatchHeaderCrcOffset);
    require_value<CorruptionError>(
        stored_header_crc
            == crc32c_with_zeroed_field(bytes, kBatchHeaderCrcOffset, 4),
        "WAL batch header checksum mismatch");
    require_value<CorruptionError>(
        get_u32(bytes, kBatchFlagsOffset) == 0,
        "WAL batch has unsupported flags");
    require_zero_range<CorruptionError>(
        bytes, kBatchReservedOffset, bytes.size(), "WAL batch header");

    BatchHeader result;
    result.sequence = get_u64(bytes, 16);
    result.operation_count = get_u64(bytes, 24);
    result.put_count = get_u64(bytes, 32);
    result.record_size = get_u64(bytes, 40);
    result.operations_offset = get_u64(bytes, 48);
    result.operations_size = get_u64(bytes, 56);
    result.payload_offset = get_u64(bytes, 64);
    result.payload_size = get_u64(bytes, 72);
    result.frame_size = get_u64(bytes, 80);
    result.wal_id = get_uuid(bytes, 88);

    require_value<CorruptionError>(
        result.sequence != 0 && result.sequence <= kMaximumCommitSequence,
        "WAL batch has an invalid commit sequence");
    require_value<CorruptionError>(
        result.operation_count != 0
            && result.operation_count <= kWalFormatMaxOperations,
        "WAL batch operation count exceeds the format maximum");
    require_value<CorruptionError>(
        result.put_count <= result.operation_count,
        "WAL batch has more PUTs than operations");
    require_value<CorruptionError>(
        result.record_size == wal_header.record_size,
        "WAL batch record size does not match the WAL schema");
    require_value<CorruptionError>(
        result.wal_id == wal_header.wal_id,
        "WAL batch belongs to a different WAL file");
    require_value<CorruptionError>(
        result.operations_offset == kWalBatchHeaderSize,
        "WAL batch has an invalid operation-table offset");

    const std::uint64_t expected_operations_size =
        checked_product<CorruptionError>(
            result.operation_count, kWalOperationEntrySize,
            "WAL operation table size");
    require_value<CorruptionError>(
        result.operations_size == expected_operations_size,
        "WAL batch has an invalid operation-table size");
    const std::uint64_t expected_payload_offset =
        checked_add<CorruptionError>(
            kWalBatchHeaderSize, expected_operations_size,
            "WAL payload offset");
    require_value<CorruptionError>(
        result.payload_offset == expected_payload_offset,
        "WAL batch has an invalid payload offset");
    const std::uint64_t expected_payload_size =
        checked_product<CorruptionError>(
            result.put_count, result.record_size, "WAL payload size");
    require_value<CorruptionError>(
        result.payload_size == expected_payload_size,
        "WAL batch has an invalid payload size");
    const std::uint64_t unaligned_frame_size = checked_add<CorruptionError>(
        expected_payload_offset, expected_payload_size, "WAL frame size");
    const std::uint64_t expected_frame_size = checked_align_up<CorruptionError>(
        unaligned_frame_size, 8, "WAL frame size");
    require_value<CorruptionError>(
        result.frame_size == expected_frame_size,
        "WAL batch has an invalid frame size");
    require_value<CorruptionError>(
        result.frame_size <= kWalFormatMaxFrameSize,
        "WAL batch exceeds the format frame-size maximum");
    return result;
}

[[nodiscard]] bool has_valid_commit_marker_crc_at(
    const std::span<const std::byte> image,
    const std::uint64_t marker_offset) noexcept {
    if (marker_offset > image.size()
        || image.size() - marker_offset < kWalCommitMarkerSize) {
        return false;
    }
    const auto marker = image.subspan(
        static_cast<std::size_t>(marker_offset),
        static_cast<std::size_t>(kWalCommitMarkerSize));
    return get_u32(marker, kCommitCrcOffset)
           == crc32c_with_zeroed_field(marker, kCommitCrcOffset, 4);
}

[[nodiscard]] bool is_intact_commit_marker_at(
    const std::span<const std::byte> image,
    const std::uint64_t marker_offset) noexcept {
    if (!has_valid_commit_marker_crc_at(image, marker_offset)) {
        return false;
    }
    const auto marker = image.subspan(
        static_cast<std::size_t>(marker_offset),
        static_cast<std::size_t>(kWalCommitMarkerSize));
    if (std::memcmp(marker.data(), kCommitMagic.data(), kCommitMagic.size()) != 0
        || get_u32(marker, 8) != kCommitMarkerVersion
        || get_u32(marker, 12) != kWalCommitMarkerSize
        || get_u32(marker, 44)
               != static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c)
        || get_u32(marker, kCommitFlagsOffset) != 0
        || get_u64(marker, kCommitReservedOffset) != 0) {
        return false;
    }
    const std::uint64_t sequence = get_u64(marker, 16);
    return sequence != 0 && sequence <= kMaximumCommitSequence;
}

[[nodiscard]] bool is_bound_commit_marker_at(
    const std::span<const std::byte> image,
    const std::uint64_t marker_offset) noexcept {
    if (!is_intact_commit_marker_at(image, marker_offset)) {
        return false;
    }
    const auto marker = image.subspan(
        static_cast<std::size_t>(marker_offset),
        static_cast<std::size_t>(kWalCommitMarkerSize));
    const std::uint64_t batch_offset = get_u64(marker, 24);
    const std::uint64_t batch_size = get_u64(marker, 32);
    if (batch_offset < kWalFileHeaderSize
        || batch_offset % 8 != 0
        || batch_size < kWalBatchHeaderSize
        || batch_size > kWalFormatMaxFrameSize
        || batch_size % 8 != 0
        || batch_offset > std::numeric_limits<std::uint64_t>::max() - batch_size) {
        return false;
    }
    return batch_offset + batch_size == marker_offset;
}

[[nodiscard]] bool tail_contains_valid_commit(
    const std::span<const std::byte> image,
    const std::uint64_t tail_offset) noexcept {
    auto cursor = image.begin() + static_cast<std::ptrdiff_t>(tail_offset);
    while (cursor != image.end()) {
        const auto found = std::search(
            cursor, image.end(), kCommitMagic.begin(), kCommitMagic.end(),
            [](const std::byte byte, const char expected) {
                return byte == static_cast<std::byte>(expected);
            });
        if (found == image.end()) {
            return false;
        }
        const auto marker_offset = static_cast<std::uint64_t>(
            std::distance(image.begin(), found));
        if (marker_offset % 8 == 0
            && is_bound_commit_marker_at(image, marker_offset)) {
            return true;
        }
        cursor = found + 1;
    }
    return false;
}

[[nodiscard]] bool can_discard_active_tail(
    const std::span<const std::byte> image,
    const std::uint64_t tail_offset,
    const WalScanMode mode) noexcept {
    return mode == WalScanMode::Active
           && !tail_contains_valid_commit(image, tail_offset);
}

void validate_expected_identity(
    const WalHeader& actual,
    const ExpectedWalIdentity& expected) {
    require_value<CorruptionError>(
        actual.collection_id == expected.collection_id
            && actual.schema_id == expected.schema_id
            && actual.wal_id == expected.wal_id
            && actual.predecessor_wal_id == expected.predecessor_wal_id
            && actual.previous_sequence == expected.previous_sequence
            && actual.first_sequence == expected.first_sequence
            && actual.record_size == expected.record_size
            && actual.dimension == expected.dimension
            && actual.dtype == expected.dtype
            && actual.item_size == expected.item_size
            && actual.id_type == expected.id_type
            && actual.row_layout == expected.row_layout,
        "WAL header does not match the expected catalog identity or schema");
}

void validate_commit_marker(
    const std::span<const std::byte> bytes,
    const BatchHeader& batch,
    const std::uint64_t batch_offset,
    const std::uint32_t batch_crc) {
    require_value<CorruptionError>(
        bytes.size() == kWalCommitMarkerSize,
        "WAL commit marker has the wrong size");
    require_magic<CorruptionError>(bytes, kCommitMagic, "WAL commit marker");
    require_value<CorruptionError>(
        get_u32(bytes, 8) == kCommitMarkerVersion,
        "WAL commit marker has an unsupported version");
    require_value<CorruptionError>(
        get_u32(bytes, 12) == kWalCommitMarkerSize,
        "WAL commit marker has an invalid size");
    require_value<CorruptionError>(
        get_u32(bytes, 44)
            == static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c),
        "WAL commit marker has an unsupported checksum algorithm");
    require_value<CorruptionError>(
        get_u32(bytes, kCommitCrcOffset)
            == crc32c_with_zeroed_field(bytes, kCommitCrcOffset, 4),
        "WAL commit marker checksum mismatch");
    require_value<CorruptionError>(
        get_u32(bytes, kCommitFlagsOffset) == 0,
        "WAL commit marker has unsupported flags");
    require_value<CorruptionError>(
        get_u64(bytes, kCommitReservedOffset) == 0,
        "WAL commit marker contains nonzero reserved bytes");
    require_value<CorruptionError>(
        get_u64(bytes, 16) == batch.sequence,
        "WAL commit marker sequence does not match its batch");
    require_value<CorruptionError>(
        get_u64(bytes, 24) == batch_offset,
        "WAL commit marker offset does not match its batch");
    require_value<CorruptionError>(
        get_u64(bytes, 32) == batch.frame_size,
        "WAL commit marker size does not match its batch");
    require_value<CorruptionError>(
        get_u32(bytes, 40) == batch_crc,
        "WAL batch checksum mismatch");
}

[[nodiscard]] std::vector<RecoveredWalOperation> decode_operations(
    const std::span<const std::byte> batch_bytes,
    const std::uint64_t batch_offset,
    const BatchHeader& batch) {
    std::vector<RecoveredWalOperation> result;
    result.reserve(static_cast<std::size_t>(batch.operation_count));
    std::vector<std::uint64_t> ids;
    ids.reserve(static_cast<std::size_t>(batch.operation_count));
    std::uint64_t put_ordinal = 0;

    for (std::uint64_t index = 0; index != batch.operation_count; ++index) {
        const std::uint64_t entry_offset = batch.operations_offset
                                           + index * kWalOperationEntrySize;
        const auto entry = batch_bytes.subspan(
            static_cast<std::size_t>(entry_offset),
            static_cast<std::size_t>(kWalOperationEntrySize));
        require_zero_range<CorruptionError>(
            entry, 1, 8, "WAL operation entry");
        const std::uint8_t opcode = std::to_integer<std::uint8_t>(entry[0]);
        const std::uint64_t id = get_u64(entry, 8);
        ids.push_back(id);

        if (opcode == static_cast<std::uint8_t>(WalOperationKind::Put)) {
            const std::uint64_t relative_payload_offset =
                batch.payload_offset + put_ordinal * batch.record_size;
            result.push_back(RecoveredWalOperation{
                id,
                WalOperationKind::Put,
                batch_offset + relative_payload_offset,
                batch.record_size});
            ++put_ordinal;
        } else if (opcode
                   == static_cast<std::uint8_t>(WalOperationKind::Delete)) {
            result.push_back(RecoveredWalOperation{
                id, WalOperationKind::Delete, 0, 0});
        } else {
            throw CorruptionError("WAL operation entry has an unknown opcode");
        }
    }

    require_value<CorruptionError>(
        put_ordinal == batch.put_count,
        "WAL operation table PUT count does not match its header");
    std::sort(ids.begin(), ids.end());
    require_value<CorruptionError>(
        std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
        "WAL batch contains duplicate IDs");

    const std::uint64_t padding_offset = batch.payload_offset + batch.payload_size;
    require_zero_range<CorruptionError>(
        batch_bytes,
        static_cast<std::size_t>(padding_offset),
        static_cast<std::size_t>(batch.frame_size),
        "WAL batch padding");
    return result;
}

}  // namespace

WalMutationFingerprintBuilder::WalMutationFingerprintBuilder(
    const std::uint64_t record_size,
    const std::uint64_t operation_count)
    : record_size_(record_size), expected_count_(operation_count) {
    require_value<InvalidArgumentError>(
        record_size != 0, "WAL fingerprint record size is zero");
    constexpr std::array<std::byte, 24> domain{
        std::byte{'T'}, std::byte{'h'}, std::byte{'r'}, std::byte{'o'},
        std::byte{'w'}, std::byte{'D'}, std::byte{'B'}, std::byte{'-'},
        std::byte{'W'}, std::byte{'A'}, std::byte{'L'}, std::byte{'-'},
        std::byte{'M'}, std::byte{'u'}, std::byte{'t'}, std::byte{'a'},
        std::byte{'t'}, std::byte{'i'}, std::byte{'o'}, std::byte{'n'},
        std::byte{'-'}, std::byte{'V'}, std::byte{'1'}, std::byte{0}};
    hash_.update(domain);
    std::array<std::byte, 16> geometry{};
    put_u64(geometry, 0, record_size);
    put_u64(geometry, 8, operation_count);
    hash_.update(geometry);
}

void WalMutationFingerprintBuilder::add(
    const WalOperationKind kind,
    const std::uint64_t id,
    const std::span<const std::byte> value) {
    require_value<InvalidArgumentError>(
        added_count_ < expected_count_,
        "WAL fingerprint has too many operations");
    require_value<InvalidArgumentError>(
        (kind == WalOperationKind::Put && value.size() == record_size_)
            || (kind == WalOperationKind::Delete && value.empty()),
        "WAL fingerprint operation has invalid value geometry");
    std::array<std::byte, 16> entry{};
    entry[0] = static_cast<std::byte>(kind);
    put_u64(entry, 8, id);
    hash_.update(entry);
    hash_.update(value);
    ++added_count_;
}

WalMutationFingerprint WalMutationFingerprintBuilder::finish() {
    require_value<InvalidArgumentError>(
        added_count_ == expected_count_,
        "WAL fingerprint operation count is incomplete");
    return hash_.finish();
}

WalMutationFingerprint fingerprint_wal_mutations(
    const std::uint64_t record_size,
    const std::span<const WalMutation> operations) {
    WalMutationFingerprintBuilder builder(record_size, operations.size());
    for (const WalMutation& operation : operations) {
        builder.add(operation.kind, operation.id, operation.value);
    }
    return builder.finish();
}

ExpectedWalIdentity ExpectedWalIdentity::from_trusted_header(
    const WalHeader& header) noexcept {
    return ExpectedWalIdentity{
        header.collection_id,
        header.schema_id,
        header.wal_id,
        header.predecessor_wal_id,
        header.previous_sequence,
        header.first_sequence,
        header.record_size,
        header.dimension,
        header.dtype,
        header.item_size,
        header.id_type,
        header.row_layout};
}

std::array<std::byte, kWalFileHeaderSize> encode_wal_header(
    const WalHeader& header) {
    validate_wal_header_value<InvalidArgumentError>(header);

    std::array<std::byte, kWalFileHeaderSize> output{};
    put_magic(output, kWalMagic);
    put_u32(output, 8, kWalFormatVersion);
    put_u32(output, 12, static_cast<std::uint32_t>(kWalFileHeaderSize));
    put_uuid(output, 16, header.collection_id);
    put_uuid(output, 32, header.schema_id);
    put_uuid(output, 48, header.wal_id);
    put_uuid(output, 64, header.predecessor_wal_id);
    put_u64(output, 80, header.previous_sequence);
    put_u64(output, 88, header.first_sequence);
    put_u64(output, 96, header.record_size);
    put_u32(output, 104, header.dimension);
    put_u32(output, 108, static_cast<std::uint32_t>(header.dtype));
    put_u32(output, 112, header.item_size);
    put_u32(output, 116, static_cast<std::uint32_t>(header.id_type));
    put_u32(output, 120, static_cast<std::uint32_t>(header.row_layout));
    put_u32(
        output, kWalChecksumAlgorithmOffset,
        static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c));
    put_u32(output, kWalFlagsOffset, 0);
    put_u64(output, 136, header.required_feature_bits);
    put_u64(output, 144, header.optional_feature_bits);
    put_u64(output, 152, header.creation_time_unix_ns);
    put_u32(
        output, kWalHeaderCrcOffset,
        crc32c_with_zeroed_field(output, kWalHeaderCrcOffset, 4));
    return output;
}

WalHeader decode_wal_header(const std::span<const std::byte> bytes) {
    require_value<CorruptionError>(
        bytes.size() == kWalFileHeaderSize,
        "WAL file header is truncated or has the wrong size");
    require_magic<CorruptionError>(bytes, kWalMagic, "WAL file");
    require_value<CorruptionError>(
        get_u32(bytes, 8) == kWalFormatVersion,
        "WAL file has an unsupported format version");
    require_value<CorruptionError>(
        get_u32(bytes, 12) == kWalFileHeaderSize,
        "WAL file has an invalid header size");
    require_value<CorruptionError>(
        get_u32(bytes, kWalChecksumAlgorithmOffset)
            == static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c),
        "WAL file has an unsupported checksum algorithm");
    require_value<CorruptionError>(
        get_u32(bytes, kWalHeaderCrcOffset)
            == crc32c_with_zeroed_field(bytes, kWalHeaderCrcOffset, 4),
        "WAL file header checksum mismatch");
    require_value<CorruptionError>(
        get_u32(bytes, kWalFlagsOffset) == 0,
        "WAL file has unsupported flags");
    require_zero_range<CorruptionError>(
        bytes, kWalReservedOffset, bytes.size(), "WAL file header");

    WalHeader result;
    result.collection_id = get_uuid(bytes, 16);
    result.schema_id = get_uuid(bytes, 32);
    result.wal_id = get_uuid(bytes, 48);
    result.predecessor_wal_id = get_uuid(bytes, 64);
    result.previous_sequence = get_u64(bytes, 80);
    result.first_sequence = get_u64(bytes, 88);
    result.record_size = get_u64(bytes, 96);
    result.dimension = get_u32(bytes, 104);
    result.dtype = static_cast<DTypeCode>(get_u32(bytes, 108));
    result.item_size = get_u32(bytes, 112);
    result.id_type = static_cast<IdType>(get_u32(bytes, 116));
    result.row_layout = static_cast<RowLayout>(get_u32(bytes, 120));
    result.required_feature_bits = get_u64(bytes, 136);
    result.optional_feature_bits = get_u64(bytes, 144);
    result.creation_time_unix_ns = get_u64(bytes, 152);
    validate_wal_header_value<CorruptionError>(result);
    return result;
}

EncodedWalTransaction encode_wal_transaction(
    const WalHeader& header,
    const std::uint64_t sequence,
    const std::uint64_t batch_offset,
    const std::span<const WalMutation> operations,
    const WalLimits& limits) {
    validate_wal_header_value<InvalidArgumentError>(header);
    validate_limits<InvalidArgumentError>(limits);
    require_value<InvalidArgumentError>(
        batch_offset >= kWalFileHeaderSize && batch_offset % 8 == 0,
        "WAL batch offset must be 8-byte aligned and follow the file header");
    require_value<InvalidArgumentError>(
        sequence >= header.first_sequence
            && sequence <= kMaximumCommitSequence,
        "WAL commit sequence is outside the usable range");
    require_value<InvalidArgumentError>(
        batch_offset != kWalFileHeaderSize
            || sequence == header.first_sequence,
        "first WAL batch does not use the header's first sequence");
    require_value<InvalidArgumentError>(
        !operations.empty() && operations.size() <= limits.max_operations,
        "WAL mutation batch is empty or exceeds max_operations");

    std::uint64_t put_count = 0;
    std::vector<std::uint64_t> ids;
    ids.reserve(operations.size());
    for (const WalMutation& operation : operations) {
        ids.push_back(operation.id);
        switch (operation.kind) {
            case WalOperationKind::Put:
                require_value<InvalidArgumentError>(
                    operation.value.size() == header.record_size,
                    "WAL PUT payload does not match the collection record size");
                ++put_count;
                break;
            case WalOperationKind::Delete:
                require_value<InvalidArgumentError>(
                    operation.value.empty(),
                    "WAL DELETE operation unexpectedly contains a payload");
                break;
            default:
                throw InvalidArgumentError("WAL mutation has an unknown opcode");
        }
    }
    std::sort(ids.begin(), ids.end());
    require_value<InvalidArgumentError>(
        std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
        "WAL mutation batch contains duplicate IDs");

    const std::uint64_t operation_count = operations.size();
    const std::uint64_t operations_size = checked_product<InvalidArgumentError>(
        operation_count, kWalOperationEntrySize, "WAL operation table size");
    const std::uint64_t payload_offset = checked_add<InvalidArgumentError>(
        kWalBatchHeaderSize, operations_size, "WAL payload offset");
    const std::uint64_t payload_size = checked_product<InvalidArgumentError>(
        put_count, header.record_size, "WAL payload size");
    const std::uint64_t unaligned_frame_size = checked_add<InvalidArgumentError>(
        payload_offset, payload_size, "WAL frame size");
    const std::uint64_t frame_size = checked_align_up<InvalidArgumentError>(
        unaligned_frame_size, 8, "WAL frame size");
    require_value<InvalidArgumentError>(
        frame_size <= limits.max_frame_size,
        "WAL mutation batch exceeds max_frame_size");
    const std::uint64_t transaction_size = checked_add<InvalidArgumentError>(
        frame_size, kWalCommitMarkerSize, "WAL transaction size");
    static_cast<void>(checked_add<InvalidArgumentError>(
        batch_offset, transaction_size, "WAL append end"));
    require_value<InvalidArgumentError>(
        transaction_size <= std::numeric_limits<std::size_t>::max(),
        "WAL transaction cannot fit in process address space");

    EncodedWalTransaction result;
    result.bytes.resize(static_cast<std::size_t>(transaction_size));
    const std::span<std::byte> output(result.bytes);
    put_magic(output, kBatchMagic);
    put_u32(output, 8, kBatchFrameVersion);
    put_u32(output, 12, static_cast<std::uint32_t>(kWalBatchHeaderSize));
    put_u64(output, 16, sequence);
    put_u64(output, 24, operation_count);
    put_u64(output, 32, put_count);
    put_u64(output, 40, header.record_size);
    put_u64(output, 48, kWalBatchHeaderSize);
    put_u64(output, 56, operations_size);
    put_u64(output, 64, payload_offset);
    put_u64(output, 72, payload_size);
    put_u64(output, 80, frame_size);
    put_uuid(output, 88, header.wal_id);
    put_u32(
        output, 104,
        static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c));
    put_u32(output, kBatchFlagsOffset, 0);

    std::uint64_t operation_offset = kWalBatchHeaderSize;
    std::uint64_t payload_cursor = payload_offset;
    for (const WalMutation& operation : operations) {
        output[static_cast<std::size_t>(operation_offset)] =
            static_cast<std::byte>(operation.kind);
        put_u64(output, static_cast<std::size_t>(operation_offset + 8), operation.id);
        operation_offset += kWalOperationEntrySize;
        if (operation.kind == WalOperationKind::Put) {
            std::memcpy(
                output.data() + static_cast<std::size_t>(payload_cursor),
                operation.value.data(), operation.value.size());
            payload_cursor += header.record_size;
        }
    }
    put_u32(
        output, kBatchHeaderCrcOffset,
        crc32c_with_zeroed_field(
            output.first(static_cast<std::size_t>(kWalBatchHeaderSize)),
            kBatchHeaderCrcOffset, 4));

    result.batch_size = frame_size;
    result.batch_crc32c =
        crc32c(output.first(static_cast<std::size_t>(frame_size)));
    const std::size_t commit_offset = static_cast<std::size_t>(frame_size);
    const auto commit = output.subspan(
        commit_offset, static_cast<std::size_t>(kWalCommitMarkerSize));
    put_magic(commit, kCommitMagic);
    put_u32(commit, 8, kCommitMarkerVersion);
    put_u32(commit, 12, static_cast<std::uint32_t>(kWalCommitMarkerSize));
    put_u64(commit, 16, sequence);
    put_u64(commit, 24, batch_offset);
    put_u64(commit, 32, frame_size);
    put_u32(commit, 40, result.batch_crc32c);
    put_u32(
        commit, 44,
        static_cast<std::uint32_t>(ChecksumAlgorithm::Crc32c));
    put_u32(commit, kCommitFlagsOffset, 0);
    put_u32(
        commit, kCommitCrcOffset,
        crc32c_with_zeroed_field(commit, kCommitCrcOffset, 4));
    return result;
}

WalScanResult scan_wal(
    const std::span<const std::byte> image,
    const ExpectedWalIdentity& expected,
    const WalScanMode mode,
    const WalRecoveryVisitor& visitor,
    const WalScanLimits limits) {
    require_value<InvalidArgumentError>(
        mode == WalScanMode::Active || mode == WalScanMode::Frozen,
        "WAL scan mode is invalid");
    require_value<CorruptionError>(
        image.size() >= kWalFileHeaderSize,
        "WAL file is shorter than its file header");
    require_value<InvalidArgumentError>(
        image.size()
            <= static_cast<std::size_t>(
                std::numeric_limits<std::ptrdiff_t>::max()),
        "WAL image is too large for safe pointer arithmetic");

    WalScanResult result;
    result.header = decode_wal_header(
        image.first(static_cast<std::size_t>(kWalFileHeaderSize)));
    validate_expected_identity(result.header, expected);
    result.valid_end = kWalFileHeaderSize;
    result.next_sequence = result.header.first_sequence;
    std::uint64_t offset = kWalFileHeaderSize;

    while (offset < image.size()) {
        const std::uint64_t remaining = image.size() - offset;
        if (remaining < kWalBatchHeaderSize) {
            if (!can_discard_active_tail(image, offset, mode)) {
                throw CorruptionError(
                    "frozen or internally committed WAL has a short tail");
            }
            break;
        }
        const auto batch_header_bytes = image.subspan(
            static_cast<std::size_t>(offset),
            static_cast<std::size_t>(kWalBatchHeaderSize));
        BatchHeader batch;
        try {
            batch = decode_batch_header(batch_header_bytes, result.header);
        } catch (const CorruptionError&) {
            if (can_discard_active_tail(image, offset, mode)) {
                break;
            }
            throw;
        }
        const std::uint64_t frame_end = checked_add<CorruptionError>(
            offset, batch.frame_size, "WAL frame end");
        if (frame_end > image.size()) {
            if (!can_discard_active_tail(image, offset, mode)) {
                throw CorruptionError(
                    "frozen or internally committed WAL has a truncated batch");
            }
            break;
        }
        if (image.size() - frame_end < kWalCommitMarkerSize) {
            if (!can_discard_active_tail(image, offset, mode)) {
                throw CorruptionError(
                    "frozen or internally committed WAL has a truncated commit marker");
            }
            break;
        }

        const auto batch_bytes = image.subspan(
            static_cast<std::size_t>(offset),
            static_cast<std::size_t>(batch.frame_size));
        const std::uint32_t batch_crc = crc32c(batch_bytes);
        const auto commit_bytes = image.subspan(
            static_cast<std::size_t>(frame_end),
            static_cast<std::size_t>(kWalCommitMarkerSize));
        try {
            validate_commit_marker(commit_bytes, batch, offset, batch_crc);
        } catch (const CorruptionError&) {
            // A marker that passes its own CRC is a completed commit claim.
            // Bad binding or batch CRC is corruption, never an active tail.
            if (has_valid_commit_marker_crc_at(image, frame_end)) {
                throw;
            }
            if (can_discard_active_tail(image, offset, mode)) {
                break;
            }
            throw;
        }
        require_value<CorruptionError>(
            batch.sequence == result.next_sequence,
            "WAL commit sequence is duplicated, missing, or out of order");

        if (batch.operation_count
            > limits.max_committed_operations
                  - result.committed_operation_count) {
            throw StorageError(
                "WAL committed operation count exceeds the configured "
                "replay limit");
        }
        result.committed_operation_count += batch.operation_count;

        // Validate operation semantics now, but defer callbacks until every
        // committed transaction in this stable image has passed validation.
        static_cast<void>(decode_operations(batch_bytes, offset, batch));
        result.valid_end = checked_add<CorruptionError>(
            frame_end, kWalCommitMarkerSize, "WAL commit end");
        result.next_sequence = batch.sequence + 1;
        ++result.committed_batch_count;
        offset = result.valid_end;
    }

    result.ignored_tail_bytes = image.size() - result.valid_end;

    if (visitor) {
        offset = kWalFileHeaderSize;
        while (offset < result.valid_end) {
            const auto batch_header_bytes = image.subspan(
                static_cast<std::size_t>(offset),
                static_cast<std::size_t>(kWalBatchHeaderSize));
            const BatchHeader batch =
                decode_batch_header(batch_header_bytes, result.header);
            const auto batch_bytes = image.subspan(
                static_cast<std::size_t>(offset),
                static_cast<std::size_t>(batch.frame_size));
            const auto operations = decode_operations(batch_bytes, offset, batch);
            visitor(RecoveredWalBatch{
                batch.sequence, offset, batch.frame_size, operations});
            offset += batch.frame_size + kWalCommitMarkerSize;
        }
    }
    return result;
}

}  // namespace throwdb::detail
