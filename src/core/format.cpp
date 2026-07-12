#include "internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace throwdb {

MissingIdError::MissingIdError(const std::uint64_t id)
    : std::out_of_range("embedding ID not found: " + std::to_string(id)), id_(id) {}

DuplicateIdError::DuplicateIdError(const std::uint64_t id)
    : std::invalid_argument("duplicate embedding ID: " + std::to_string(id)), id_(id) {}

ExistingIdError::ExistingIdError(const std::uint64_t id)
    : std::invalid_argument("embedding ID already exists: " + std::to_string(id)),
      id_(id) {}

}  // namespace throwdb

namespace throwdb::detail {
namespace {

constexpr std::size_t kIoChunk = 1U << 30;

void require_magic(std::span<const std::byte> bytes,
                   const std::array<char, 8>& expected,
                   std::string_view description) {
    if (bytes.size() < expected.size()
        || std::memcmp(bytes.data(), expected.data(), expected.size()) != 0) {
        throw CorruptionError(std::string(description) + " has invalid magic");
    }
}

void put_magic(std::span<std::byte> bytes, const std::array<char, 8>& magic) {
    std::memcpy(bytes.data(), magic.data(), magic.size());
}

[[nodiscard]] std::string unsupported_version_message(
    const std::string_view description, const std::uint32_t version) {
    return std::string(description) + " has unsupported format version "
           + std::to_string(version);
}

template <typename Error>
void require_value(const bool condition, const std::string_view message) {
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
void validate_numeric_v2_manifest(const Manifest& manifest) {
    require_value<Error>(
        manifest.format_version == FormatVersion::V2,
        "format-v2 manifest has the wrong format version");
    require_value<Error>(
        manifest.schema.id_type == IdType::UInt64,
        "format-v2 manifest has an unsupported ID type");
    require_value<Error>(
        manifest.schema.dtype == DTypeCode::Float16
            || manifest.schema.dtype == DTypeCode::Float32,
        "format-v2 manifest has an unsupported dtype");
    require_value<Error>(
        manifest.schema.dimension != 0,
        "format-v2 manifest has a zero dimension");
    require_value<Error>(
        manifest.schema.item_size == dtype_item_size(manifest.schema.dtype),
        "format-v2 manifest dtype and item size do not agree");
    require_value<Error>(
        manifest.schema.row_layout == RowLayout::DenseRowMajor,
        "format-v2 manifest has an unsupported row layout");
    require_value<Error>(
        manifest.schema.index_kind == IndexKind::UInt64Flat,
        "format-v2 manifest has an unsupported index kind");
    require_value<Error>(
        manifest.schema.hash_algorithm == HashAlgorithm::SplitMix64V1,
        "format-v2 manifest has an unsupported hash algorithm");
    require_value<Error>(
        manifest.header_checksum_algorithm == ChecksumAlgorithm::None
            && manifest.header_checksum == 0,
        "format-v2 manifest uses an unsupported header checksum");

    const std::uint64_t expected_record_size = checked_product<Error>(
        manifest.schema.dimension, manifest.schema.item_size,
        "format-v2 manifest record size");
    require_value<Error>(
        manifest.record_size == expected_record_size,
        "format-v2 manifest record size does not match its schema");
    require_value<Error>(
        manifest.shard_size_bytes >= manifest.record_size,
        "format-v2 manifest shard capacity cannot fit one row");
    require_value<Error>(
        manifest.shard_size_bytes
            <= std::numeric_limits<std::uint64_t>::max() - kHeaderSize,
        "format-v2 manifest shard file size overflows uint64");
    require_value<Error>(
        manifest.records_per_shard
            == manifest.shard_size_bytes / manifest.record_size,
        "format-v2 manifest has invalid records-per-shard geometry");
    const std::uint64_t expected_shard_count =
        manifest.record_count / manifest.records_per_shard
        + (manifest.record_count % manifest.records_per_shard == 0 ? 0 : 1);
    require_value<Error>(
        manifest.shard_count == expected_shard_count,
        "format-v2 manifest shard count does not match its record count");

    require_value<Error>(
        manifest.index_capacity != 0
            && (manifest.index_capacity & (manifest.index_capacity - 1)) == 0
            && manifest.record_count <= manifest.index_capacity,
        "format-v2 manifest has an invalid index capacity");
    require_value<Error>(
        manifest.index_slot_size == kIndexSlotSize,
        "format-v2 manifest has an invalid numeric index slot size");
    require_value<Error>(
        manifest.index_control_offset == kHeaderSize,
        "format-v2 manifest has an invalid index control offset");
    const std::uint64_t controls_end = checked_sum<Error>(
        manifest.index_control_offset, manifest.index_capacity,
        "format-v2 index controls");
    std::uint64_t expected_slots_offset = controls_end;
    const std::uint64_t remainder = expected_slots_offset % kHeaderSize;
    if (remainder != 0) {
        expected_slots_offset = checked_sum<Error>(
            expected_slots_offset, kHeaderSize - remainder,
            "format-v2 index slot alignment");
    }
    require_value<Error>(
        manifest.index_slots_offset == expected_slots_offset,
        "format-v2 manifest has an invalid index slot offset");
    const std::uint64_t slot_bytes = checked_product<Error>(
        manifest.index_capacity, manifest.index_slot_size,
        "format-v2 index slots");
    const std::uint64_t expected_index_size = checked_sum<Error>(
        manifest.index_slots_offset, slot_bytes,
        "format-v2 index file size");
    require_value<Error>(
        manifest.index_file_size == expected_index_size,
        "format-v2 manifest has an invalid index file size");

    require_value<Error>(
        manifest.expected_string_arena_size == 0
            && manifest.string_arena_payload_offset == 0,
        "format-v2 numeric manifest contains string-arena metadata");
    require_value<Error>(
        manifest.required_feature_bits == 0
            && manifest.optional_feature_bits == 0,
        "format-v2 manifest contains unsupported feature bits");
    require_value<Error>(
        !manifest.snapshot_id.empty(),
        "format-v2 manifest has an empty snapshot ID");
    require_value<Error>(
        manifest.hash_seed_high == 0,
        "format-v2 numeric manifest has a nonzero hash-key high word");
}

template <typename Error>
void validate_v2_shard_header(const ShardHeader& header) {
    require_value<Error>(
        header.format_version == FormatVersion::V2,
        "format-v2 shard has the wrong format version");
    require_value<Error>(
        !header.snapshot_id.empty(),
        "format-v2 shard has an empty snapshot ID");
    require_value<Error>(
        header.dtype == DTypeCode::Float16
            || header.dtype == DTypeCode::Float32,
        "format-v2 shard has an unsupported dtype");
    require_value<Error>(
        header.dimension != 0,
        "format-v2 shard has a zero dimension");
    require_value<Error>(
        header.row_layout == RowLayout::DenseRowMajor,
        "format-v2 shard has an unsupported row layout");
    const std::uint64_t expected_record_size = checked_product<Error>(
        header.dimension, dtype_item_size(header.dtype),
        "format-v2 shard record size");
    require_value<Error>(
        header.record_size == expected_record_size,
        "format-v2 shard record size does not match its schema");
    const std::uint64_t expected_payload_size = checked_product<Error>(
        header.record_count, header.record_size,
        "format-v2 shard payload size");
    require_value<Error>(
        header.payload_size == expected_payload_size,
        "format-v2 shard payload size does not match its rows");
    require_value<Error>(
        header.payload_checksum_algorithm == ChecksumAlgorithm::None
            && header.payload_checksum == 0,
        "format-v2 shard uses an unsupported payload checksum");
    require_value<Error>(
        header.header_checksum == 0,
        "format-v2 shard has a checksum without checksum support");
}

void require_reserved_zero(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::string_view description) {
    for (std::size_t offset = begin; offset != kHeaderSize; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw CorruptionError(
                std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

[[nodiscard]] std::uint64_t mix_snapshot_word(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

FileDescriptor::~FileDescriptor() {
    reset();
}

FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
        reset(std::exchange(other.value, -1));
    }
    return *this;
}

void FileDescriptor::reset(const int fd) noexcept {
    if (value >= 0) {
        while (::close(value) < 0 && errno == EINTR) {
        }
    }
    value = fd;
}

std::string system_error_message(const std::string_view operation,
                                 const std::filesystem::path& path,
                                 const int error_number) {
    std::ostringstream stream;
    stream << operation << " '" << path.string() << "': "
           << std::strerror(error_number);
    return stream.str();
}

FileDescriptor open_read_only(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw StorageError(system_error_message("cannot open", path));
    }
    return FileDescriptor(fd);
}

FileDescriptor open_read_write(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        throw StorageError(system_error_message("cannot open read-write", path));
    }
    return FileDescriptor(fd);
}

FileDescriptor create_exclusive(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw StorageError(system_error_message("cannot create", path));
    }
    return FileDescriptor(fd);
}

void write_all(const int fd, const void* data, const std::size_t size,
               const std::filesystem::path& path) {
    const auto* cursor = static_cast<const std::byte*>(data);
    std::size_t remaining = size;
    while (remaining != 0) {
        const std::size_t requested = std::min(remaining, kIoChunk);
        const ssize_t written = ::write(fd, cursor, requested);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw StorageError(system_error_message("write failed for", path));
        }
        if (written == 0) {
            throw StorageError("write made no progress for '" + path.string() + "'");
        }
        cursor += static_cast<std::size_t>(written);
        remaining -= static_cast<std::size_t>(written);
    }
}

void pwrite_all(const int fd, const void* data, const std::size_t size,
                std::uint64_t offset, const std::filesystem::path& path) {
    const auto maximum_offset = static_cast<std::uint64_t>(
        std::numeric_limits<off_t>::max());
    if (offset > maximum_offset || size > maximum_offset - offset) {
        throw InvalidArgumentError(
            "pwrite range is too large for '" + path.string() + "'");
    }
    const auto* cursor = static_cast<const std::byte*>(data);
    std::size_t remaining = size;
    while (remaining != 0) {
        const std::size_t requested = std::min(remaining, kIoChunk);
        const ssize_t written = ::pwrite(fd, cursor, requested,
                                         static_cast<off_t>(offset));
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw StorageError(system_error_message("pwrite failed for", path));
        }
        if (written == 0) {
            throw StorageError("pwrite made no progress for '" + path.string() + "'");
        }
        cursor += static_cast<std::size_t>(written);
        remaining -= static_cast<std::size_t>(written);
        offset += static_cast<std::uint64_t>(written);
    }
}

void pread_exact(const int fd, void* data, const std::size_t size,
                 std::uint64_t offset, const std::filesystem::path& path) {
    const auto maximum_offset = static_cast<std::uint64_t>(
        std::numeric_limits<off_t>::max());
    if (offset > maximum_offset || size > maximum_offset - offset) {
        throw CorruptionError(
            "pread range is too large for '" + path.string() + "'");
    }
    auto* cursor = static_cast<std::byte*>(data);
    std::size_t remaining = size;
    while (remaining != 0) {
        const std::size_t requested = std::min(remaining, kIoChunk);
        const ssize_t received = ::pread(fd, cursor, requested,
                                         static_cast<off_t>(offset));
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw StorageError(system_error_message("pread failed for", path));
        }
        if (received == 0) {
            throw CorruptionError("unexpected end of file in '" + path.string() + "'");
        }
        cursor += static_cast<std::size_t>(received);
        remaining -= static_cast<std::size_t>(received);
        offset += static_cast<std::uint64_t>(received);
    }
}

void sync_file(const int fd, const std::filesystem::path& path) {
    while (::fsync(fd) < 0) {
        if (errno == EINTR) {
            continue;
        }
        throw StorageError(system_error_message("fsync failed for", path));
    }
}

void sync_directory(const std::filesystem::path& path) {
    const int raw = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (raw < 0) {
        throw StorageError(system_error_message("cannot open directory", path));
    }
    FileDescriptor fd(raw);
    sync_file(fd.get(), path);
}

std::uint64_t file_size(const int fd, const std::filesystem::path& path) {
    struct stat status {};
    if (::fstat(fd, &status) < 0) {
        throw StorageError(system_error_message("fstat failed for", path));
    }
    if (status.st_size < 0) {
        throw CorruptionError("negative file size for '" + path.string() + "'");
    }
    return static_cast<std::uint64_t>(status.st_size);
}

void truncate_file(const int fd, const std::uint64_t size,
                   const std::filesystem::path& path) {
    if (size > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
        throw InvalidArgumentError(
            "truncate size is too large for '" + path.string() + "'");
    }
    if (::ftruncate(fd, static_cast<off_t>(size)) < 0) {
        throw StorageError(system_error_message("ftruncate failed for", path));
    }
}

std::uint64_t checked_multiply(const std::uint64_t left,
                               const std::uint64_t right,
                               const std::string_view description) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw InvalidArgumentError(std::string(description) + " is too large");
    }
    return left * right;
}

std::uint64_t align_up(const std::uint64_t value, const std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw InvalidArgumentError("alignment must be a power of two");
    }
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw InvalidArgumentError("aligned value is too large");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint32_t dtype_item_size(const DTypeCode dtype) noexcept {
    switch (dtype) {
        case DTypeCode::Invalid:
            return 0;
        case DTypeCode::Float16:
        case DTypeCode::BFloat16:
            return 2;
        case DTypeCode::Float32:
            return 4;
    }
    return 0;
}

SnapshotId make_snapshot_id() {
    static std::atomic<std::uint64_t> sequence{0};
    std::random_device random;
    const auto random_word = [&random] {
        const std::uint64_t high = static_cast<std::uint64_t>(random());
        const std::uint64_t low = static_cast<std::uint64_t>(random());
        return (high << 32) ^ low;
    };
    const auto now = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::uint64_t serial = sequence.fetch_add(1, std::memory_order_relaxed);
    SnapshotId result{
        mix_snapshot_word(random_word() ^ now ^ serial),
        mix_snapshot_word(
            random_word() ^ (now + 0x9e3779b97f4a7c15ULL)
            ^ (serial << 1))};
    if (result.empty()) {
        result.high = 1;
    }
    return result;
}

void put_u32(const std::span<std::byte> output, const std::size_t offset,
             const std::uint32_t value) {
    if (offset > output.size() || output.size() - offset < 4) {
        throw InvalidArgumentError("serialization output is too small");
    }
    for (std::size_t index = 0; index != 4; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8 * index));
    }
}

void put_u64(const std::span<std::byte> output, const std::size_t offset,
             const std::uint64_t value) {
    if (offset > output.size() || output.size() - offset < 8) {
        throw InvalidArgumentError("serialization output is too small");
    }
    for (std::size_t index = 0; index != 8; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8 * index));
    }
}

std::uint32_t get_u32(const std::span<const std::byte> input,
                      const std::size_t offset) {
    if (offset > input.size() || input.size() - offset < 4) {
        throw CorruptionError("serialized uint32 is truncated");
    }
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index) {
        value |= std::uint32_t{std::to_integer<std::uint8_t>(input[offset + index])}
                 << (8 * index);
    }
    return value;
}

std::uint64_t get_u64(const std::span<const std::byte> input,
                      const std::size_t offset) {
    if (offset > input.size() || input.size() - offset < 8) {
        throw CorruptionError("serialized uint64 is truncated");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != 8; ++index) {
        value |= std::uint64_t{std::to_integer<std::uint8_t>(input[offset + index])}
                 << (8 * index);
    }
    return value;
}

CommonHeader decode_common_header(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& expected_magic,
    const std::string_view description) {
    require_magic(bytes, expected_magic, description);
    if (bytes.size() < kHeaderSize) {
        throw CorruptionError(std::string(description) + " header is truncated");
    }
    if (get_u32(bytes, 12) != kHeaderSize) {
        throw CorruptionError(std::string(description) + " has invalid header size");
    }
    const std::uint32_t encoded_version = get_u32(bytes, 8);
    FormatVersion format_version;
    switch (encoded_version) {
        case static_cast<std::uint32_t>(FormatVersion::V1):
            format_version = FormatVersion::V1;
            break;
        case static_cast<std::uint32_t>(FormatVersion::V2):
            format_version = FormatVersion::V2;
            break;
        default:
            throw CorruptionError(
                unsupported_version_message(description, encoded_version));
    }
    return CommonHeader{
        format_version, static_cast<std::uint32_t>(kHeaderSize)};
}

std::array<std::byte, kHeaderSize> encode_manifest(const Manifest& manifest) {
    std::array<std::byte, kHeaderSize> output{};
    switch (manifest.format_version) {
        case FormatVersion::V1:
            if (manifest.schema.id_type != IdType::UInt64
                || manifest.schema.dtype != DTypeCode::Float16
                || manifest.schema.item_size != 2
                || manifest.schema.row_layout != RowLayout::DenseRowMajor
                || manifest.schema.index_kind != IndexKind::UInt64Flat
                || manifest.schema.hash_algorithm != HashAlgorithm::SplitMix64V1) {
                throw InvalidArgumentError(
                    "format-v1 manifest contains an incompatible collection schema");
            }
            put_magic(output, kManifestMagic);
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V1));
            put_u32(output, 12, kHeaderSize);
            put_u32(output, 16, static_cast<std::uint32_t>(manifest.schema.dtype));
            put_u32(output, 20, manifest.schema.dimension);
            put_u64(output, 24, manifest.record_size);
            put_u64(output, 32, manifest.record_count);
            put_u64(output, 40, manifest.shard_size_bytes);
            put_u64(output, 48, manifest.records_per_shard);
            put_u64(output, 56, manifest.shard_count);
            put_u64(output, 64, manifest.index_capacity);
            put_u64(output, 72, manifest.hash_seed);
            put_u64(output, 80, manifest.index_control_offset);
            put_u64(output, 88, manifest.index_slots_offset);
            put_u64(output, 96, manifest.index_file_size);
            return output;

        case FormatVersion::V2:
            validate_numeric_v2_manifest<InvalidArgumentError>(manifest);
            put_magic(output, kManifestMagic);
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V2));
            put_u32(output, 12, kHeaderSize);
            put_u32(output, 16, static_cast<std::uint32_t>(manifest.schema.dtype));
            put_u32(output, 20, manifest.schema.dimension);
            put_u32(output, 24, manifest.schema.item_size);
            put_u32(output, 28, static_cast<std::uint32_t>(manifest.schema.id_type));
            put_u32(output, 32, static_cast<std::uint32_t>(manifest.schema.row_layout));
            put_u32(output, 36, static_cast<std::uint32_t>(manifest.schema.index_kind));
            put_u32(
                output, 40,
                static_cast<std::uint32_t>(manifest.schema.hash_algorithm));
            put_u32(
                output, 44,
                static_cast<std::uint32_t>(manifest.header_checksum_algorithm));
            put_u64(output, 48, manifest.record_size);
            put_u64(output, 56, manifest.record_count);
            put_u64(output, 64, manifest.shard_size_bytes);
            put_u64(output, 72, manifest.records_per_shard);
            put_u64(output, 80, manifest.shard_count);
            put_u64(output, 88, manifest.index_capacity);
            put_u64(output, 96, manifest.index_file_size);
            put_u64(output, 104, manifest.expected_string_arena_size);
            put_u64(output, 112, manifest.required_feature_bits);
            put_u64(output, 120, manifest.optional_feature_bits);
            put_u64(output, 128, manifest.snapshot_id.low);
            put_u64(output, 136, manifest.snapshot_id.high);
            put_u64(output, 144, manifest.hash_seed);
            put_u64(output, 152, manifest.hash_seed_high);
            put_u64(output, 160, manifest.index_control_offset);
            put_u64(output, 168, manifest.index_slots_offset);
            put_u64(output, 176, manifest.index_slot_size);
            put_u64(output, 184, manifest.string_arena_payload_offset);
            put_u64(output, 192, manifest.creation_time_unix_ns);
            put_u64(output, 200, manifest.header_checksum);
            return output;

        case FormatVersion::Invalid:
            break;
    }
    throw InvalidArgumentError("manifest has an invalid format version");
}

Manifest decode_manifest(const std::span<const std::byte> bytes) {
    const CommonHeader common = decode_common_header(
        bytes, kManifestMagic, "manifest");
    Manifest result;
    result.format_version = common.format_version;
    switch (common.format_version) {
        case FormatVersion::V1: {
            result.schema.id_type = IdType::UInt64;
            const std::uint32_t encoded_dtype = get_u32(bytes, 16);
            if (encoded_dtype != static_cast<std::uint32_t>(DTypeCode::Float16)) {
                throw CorruptionError("format-v1 manifest has unsupported dtype code "
                                      + std::to_string(encoded_dtype));
            }
            result.schema.dtype = DTypeCode::Float16;
            result.schema.dimension = get_u32(bytes, 20);
            result.schema.item_size = 2;
            result.schema.row_layout = RowLayout::DenseRowMajor;
            result.schema.index_kind = IndexKind::UInt64Flat;
            result.schema.hash_algorithm = HashAlgorithm::SplitMix64V1;
            result.record_size = get_u64(bytes, 24);
            result.record_count = get_u64(bytes, 32);
            result.shard_size_bytes = get_u64(bytes, 40);
            result.records_per_shard = get_u64(bytes, 48);
            result.shard_count = get_u64(bytes, 56);
            result.index_capacity = get_u64(bytes, 64);
            result.hash_seed = get_u64(bytes, 72);
            result.index_control_offset = get_u64(bytes, 80);
            result.index_slots_offset = get_u64(bytes, 88);
            result.index_file_size = get_u64(bytes, 96);
            if (result.schema.dimension == 0 || result.record_size == 0
                || result.records_per_shard == 0 || result.index_capacity == 0) {
                throw CorruptionError("manifest contains zero-valued required fields");
            }
            return result;
        }

        case FormatVersion::V2:
            result.schema.dtype = static_cast<DTypeCode>(get_u32(bytes, 16));
            result.schema.dimension = get_u32(bytes, 20);
            result.schema.item_size = get_u32(bytes, 24);
            result.schema.id_type = static_cast<IdType>(get_u32(bytes, 28));
            result.schema.row_layout = static_cast<RowLayout>(get_u32(bytes, 32));
            result.schema.index_kind = static_cast<IndexKind>(get_u32(bytes, 36));
            result.schema.hash_algorithm =
                static_cast<HashAlgorithm>(get_u32(bytes, 40));
            result.header_checksum_algorithm =
                static_cast<ChecksumAlgorithm>(get_u32(bytes, 44));
            result.record_size = get_u64(bytes, 48);
            result.record_count = get_u64(bytes, 56);
            result.shard_size_bytes = get_u64(bytes, 64);
            result.records_per_shard = get_u64(bytes, 72);
            result.shard_count = get_u64(bytes, 80);
            result.index_capacity = get_u64(bytes, 88);
            result.index_file_size = get_u64(bytes, 96);
            result.expected_string_arena_size = get_u64(bytes, 104);
            result.required_feature_bits = get_u64(bytes, 112);
            result.optional_feature_bits = get_u64(bytes, 120);
            result.snapshot_id.low = get_u64(bytes, 128);
            result.snapshot_id.high = get_u64(bytes, 136);
            result.hash_seed = get_u64(bytes, 144);
            result.hash_seed_high = get_u64(bytes, 152);
            result.index_control_offset = get_u64(bytes, 160);
            result.index_slots_offset = get_u64(bytes, 168);
            result.index_slot_size = get_u64(bytes, 176);
            result.string_arena_payload_offset = get_u64(bytes, 184);
            result.creation_time_unix_ns = get_u64(bytes, 192);
            result.header_checksum = get_u64(bytes, 200);
            validate_numeric_v2_manifest<CorruptionError>(result);
            require_reserved_zero(bytes, 208, "format-v2 manifest");
            return result;

        case FormatVersion::Invalid:
            break;
    }
    throw CorruptionError("manifest has an invalid format version");
}

std::array<std::byte, kHeaderSize> encode_shard_header(const ShardHeader& header) {
    std::array<std::byte, kHeaderSize> output{};
    switch (header.format_version) {
        case FormatVersion::V1:
            if (header.dtype != DTypeCode::Float16
                || header.row_layout != RowLayout::DenseRowMajor) {
                throw InvalidArgumentError(
                    "format-v1 shard contains an incompatible row schema");
            }
            put_magic(output, kShardMagic);
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V1));
            put_u32(output, 12, kHeaderSize);
            put_u64(output, 16, header.shard_id);
            put_u64(output, 24, header.first_ordinal);
            put_u64(output, 32, header.record_count);
            put_u64(output, 40, header.record_size);
            put_u32(output, 48, header.dimension);
            put_u32(output, 52, static_cast<std::uint32_t>(header.dtype));
            return output;

        case FormatVersion::V2:
            validate_v2_shard_header<InvalidArgumentError>(header);
            put_magic(output, kShardMagic);
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V2));
            put_u32(output, 12, kHeaderSize);
            put_u64(output, 16, header.snapshot_id.low);
            put_u64(output, 24, header.snapshot_id.high);
            put_u64(output, 32, header.shard_id);
            put_u64(output, 40, header.first_ordinal);
            put_u64(output, 48, header.record_count);
            put_u64(output, 56, header.record_size);
            put_u32(output, 64, header.dimension);
            put_u32(output, 68, static_cast<std::uint32_t>(header.dtype));
            put_u32(output, 72, static_cast<std::uint32_t>(header.row_layout));
            put_u32(
                output, 76,
                static_cast<std::uint32_t>(header.payload_checksum_algorithm));
            put_u64(output, 80, header.payload_size);
            put_u64(output, 88, header.payload_checksum);
            put_u64(output, 96, header.header_checksum);
            return output;

        case FormatVersion::Invalid:
            break;
    }
    throw InvalidArgumentError("shard has an invalid format version");
}

ShardHeader decode_shard_header(
    const std::span<const std::byte> bytes,
    const FormatVersion expected_format_version) {
    const CommonHeader common = decode_common_header(bytes, kShardMagic, "shard");
    if (common.format_version != expected_format_version) {
        throw CorruptionError("shard format version does not match manifest");
    }
    ShardHeader result;
    result.format_version = common.format_version;
    switch (expected_format_version) {
        case FormatVersion::V1: {
            result.shard_id = get_u64(bytes, 16);
            result.first_ordinal = get_u64(bytes, 24);
            result.record_count = get_u64(bytes, 32);
            result.record_size = get_u64(bytes, 40);
            result.dimension = get_u32(bytes, 48);
            const std::uint32_t encoded_dtype = get_u32(bytes, 52);
            if (encoded_dtype != static_cast<std::uint32_t>(DTypeCode::Float16)) {
                throw CorruptionError("format-v1 shard has unsupported dtype code "
                                      + std::to_string(encoded_dtype));
            }
            result.dtype = DTypeCode::Float16;
            result.row_layout = RowLayout::DenseRowMajor;
            return result;
        }

        case FormatVersion::V2:
            result.snapshot_id.low = get_u64(bytes, 16);
            result.snapshot_id.high = get_u64(bytes, 24);
            result.shard_id = get_u64(bytes, 32);
            result.first_ordinal = get_u64(bytes, 40);
            result.record_count = get_u64(bytes, 48);
            result.record_size = get_u64(bytes, 56);
            result.dimension = get_u32(bytes, 64);
            result.dtype = static_cast<DTypeCode>(get_u32(bytes, 68));
            result.row_layout = static_cast<RowLayout>(get_u32(bytes, 72));
            result.payload_checksum_algorithm =
                static_cast<ChecksumAlgorithm>(get_u32(bytes, 76));
            result.payload_size = get_u64(bytes, 80);
            result.payload_checksum = get_u64(bytes, 88);
            result.header_checksum = get_u64(bytes, 96);
            validate_v2_shard_header<CorruptionError>(result);
            require_reserved_zero(bytes, 104, "format-v2 shard");
            return result;

        case FormatVersion::Invalid:
            break;
    }
    throw CorruptionError("shard has an invalid format version");
}

std::array<std::byte, kHeaderSize> encode_staging_header(const std::uint64_t count) {
    std::array<std::byte, kHeaderSize> output{};
    put_magic(output, kStagingMagic);
    put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V1));
    put_u32(output, 12, kHeaderSize);
    put_u64(output, 16, count);
    put_u64(output, 24, kStagingEntrySize);
    return output;
}

std::uint64_t decode_staging_header(const std::span<const std::byte> bytes) {
    const CommonHeader common = decode_common_header(
        bytes, kStagingMagic, "staging log");
    if (common.format_version != FormatVersion::V1) {
        throw CorruptionError(unsupported_version_message(
            "staging log", static_cast<std::uint32_t>(common.format_version)));
    }
    if (get_u64(bytes, 24) != kStagingEntrySize) {
        throw CorruptionError("staging log has invalid entry size");
    }
    return get_u64(bytes, 16);
}

void write_manifest_atomic(const std::filesystem::path& store_path,
                           const Manifest& manifest) {
    const auto temporary = store_path / "manifest.bin.tmp";
    const auto final_path = store_path / "manifest.bin";
    auto file = create_exclusive(temporary);
    const auto bytes = encode_manifest(manifest);
    write_all(file.get(), bytes.data(), bytes.size(), temporary);
    sync_file(file.get(), temporary);
    file.reset();
    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if (error) {
        throw StorageError("cannot publish manifest: " + error.message());
    }
    sync_directory(store_path);
}

Manifest read_manifest(const std::filesystem::path& store_path) {
    const auto path = store_path / "manifest.bin";
    auto fd = open_read_only(path);
    if (file_size(fd.get(), path) != kHeaderSize) {
        throw CorruptionError("manifest has invalid file size");
    }
    std::array<std::byte, kHeaderSize> bytes{};
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    return decode_manifest(bytes);
}

std::filesystem::path shard_path(const std::filesystem::path& store_path,
                                 const std::uint64_t shard_id) {
    std::array<char, 64> name{};
    std::snprintf(name.data(), name.size(), "shard-%06llu.dat",
                  static_cast<unsigned long long>(shard_id));
    return store_path / "data" / name.data();
}

}  // namespace throwdb::detail
