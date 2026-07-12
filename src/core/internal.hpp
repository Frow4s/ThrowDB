#pragma once

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"

namespace throwdb::detail {

enum class FormatVersion : std::uint32_t {
    Invalid = 0,
    V1 = 1,
    V2 = 2,
};

enum class DTypeCode : std::uint32_t {
    Invalid = 0,
    Float16 = 1,
    Float32 = 2,
    BFloat16 = 3,
};

enum class IdType : std::uint32_t {
    Invalid = 0,
    UInt64 = 1,
    Utf8 = 2,
};

enum class RowLayout : std::uint32_t {
    Invalid = 0,
    DenseRowMajor = 1,
};

enum class IndexKind : std::uint32_t {
    Invalid = 0,
    UInt64Flat = 1,
    Utf8FingerprintFlat = 2,
};

enum class HashAlgorithm : std::uint32_t {
    Invalid = 0,
    SplitMix64V1 = 1,
    SipHash24 = 2,
};

enum class ChecksumAlgorithm : std::uint32_t {
    None = 0,
    Crc32c = 1,
};

inline constexpr std::uint64_t kHeaderSize = 4096;
inline constexpr std::uint64_t kIndexSlotSize = 16;
inline constexpr std::uint64_t kStagingEntrySize = 16;
inline constexpr std::uint64_t kHashSeed = 0x6a09e667f3bcc909ULL;

inline constexpr std::array<char, 8> kManifestMagic{
    'T', 'H', 'R', 'O', 'W', 'D', 'B', 'M'};
inline constexpr std::array<char, 8> kShardMagic{
    'T', 'H', 'R', 'O', 'W', 'D', 'B', 'S'};
inline constexpr std::array<char, 8> kIndexMagic{
    'T', 'H', 'R', 'O', 'W', 'D', 'B', 'I'};
inline constexpr std::array<char, 8> kStagingMagic{
    'T', 'H', 'R', 'O', 'W', 'D', 'B', 'L'};

struct FileDescriptor {
    int value{-1};

    FileDescriptor() = default;
    explicit FileDescriptor(int fd) noexcept : value(fd) {}
    ~FileDescriptor();

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    FileDescriptor(FileDescriptor&& other) noexcept
        : value(std::exchange(other.value, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept;

    [[nodiscard]] int get() const noexcept { return value; }
    [[nodiscard]] explicit operator bool() const noexcept { return value >= 0; }
    int release() noexcept { return std::exchange(value, -1); }
    void reset(int fd = -1) noexcept;
};

struct CommonHeader {
    FormatVersion format_version{FormatVersion::Invalid};
    std::uint32_t header_size{0};
};

struct SnapshotId {
    std::uint64_t low{0};
    std::uint64_t high{0};

    [[nodiscard]] bool empty() const noexcept { return low == 0 && high == 0; }
    bool operator==(const SnapshotId&) const = default;
};

struct CollectionSchema {
    IdType id_type{IdType::Invalid};
    DTypeCode dtype{DTypeCode::Invalid};
    std::uint32_t dimension{0};
    std::uint32_t item_size{0};
    RowLayout row_layout{RowLayout::Invalid};
    IndexKind index_kind{IndexKind::Invalid};
    HashAlgorithm hash_algorithm{HashAlgorithm::Invalid};
};

struct Manifest {
    FormatVersion format_version{FormatVersion::Invalid};
    CollectionSchema schema;
    ChecksumAlgorithm header_checksum_algorithm{ChecksumAlgorithm::None};
    std::uint64_t record_size{0};
    std::uint64_t record_count{0};
    std::uint64_t shard_size_bytes{0};
    std::uint64_t records_per_shard{0};
    std::uint64_t shard_count{0};
    std::uint64_t index_capacity{0};
    std::uint64_t expected_string_arena_size{0};
    std::uint64_t required_feature_bits{0};
    std::uint64_t optional_feature_bits{0};
    SnapshotId snapshot_id;
    std::uint64_t hash_seed{kHashSeed};
    std::uint64_t hash_seed_high{0};
    std::uint64_t index_control_offset{kHeaderSize};
    std::uint64_t index_slots_offset{kHeaderSize};
    std::uint64_t index_slot_size{kIndexSlotSize};
    std::uint64_t index_file_size{kHeaderSize};
    std::uint64_t string_arena_payload_offset{0};
    std::uint64_t creation_time_unix_ns{0};
    std::uint64_t header_checksum{0};
};

struct ShardHeader {
    FormatVersion format_version{FormatVersion::Invalid};
    SnapshotId snapshot_id;
    std::uint64_t shard_id{0};
    std::uint64_t first_ordinal{0};
    std::uint64_t record_count{0};
    std::uint64_t record_size{0};
    std::uint32_t dimension{0};
    DTypeCode dtype{DTypeCode::Invalid};
    RowLayout row_layout{RowLayout::Invalid};
    ChecksumAlgorithm payload_checksum_algorithm{ChecksumAlgorithm::None};
    std::uint64_t payload_size{0};
    std::uint64_t payload_checksum{0};
    std::uint64_t header_checksum{0};
};

struct IndexSlot {
    std::uint64_t id{0};
    std::uint64_t ordinal{0};
};

struct IndexData {
    std::uint64_t count{0};
    std::uint64_t capacity{0};
    std::uint64_t hash_seed{kHashSeed};
    std::vector<std::uint8_t> controls;
    std::vector<IndexSlot> slots;
};

[[nodiscard]] std::string system_error_message(
    std::string_view operation,
    const std::filesystem::path& path,
    int error_number = errno);

[[nodiscard]] FileDescriptor open_read_only(const std::filesystem::path& path);
[[nodiscard]] FileDescriptor create_exclusive(const std::filesystem::path& path);
void write_all(int fd, const void* data, std::size_t size,
               const std::filesystem::path& path);
void pwrite_all(int fd, const void* data, std::size_t size,
                std::uint64_t offset, const std::filesystem::path& path);
void pread_exact(int fd, void* data, std::size_t size,
                 std::uint64_t offset, const std::filesystem::path& path);
void sync_file(int fd, const std::filesystem::path& path);
void sync_directory(const std::filesystem::path& path);
[[nodiscard]] std::uint64_t file_size(int fd, const std::filesystem::path& path);

[[nodiscard]] std::uint64_t checked_multiply(
    std::uint64_t left, std::uint64_t right, std::string_view description);
[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment);
[[nodiscard]] std::uint32_t dtype_item_size(DTypeCode dtype) noexcept;
[[nodiscard]] SnapshotId make_snapshot_id();

void put_u32(std::span<std::byte> output, std::size_t offset, std::uint32_t value);
void put_u64(std::span<std::byte> output, std::size_t offset, std::uint64_t value);
[[nodiscard]] std::uint32_t get_u32(
    std::span<const std::byte> input, std::size_t offset);
[[nodiscard]] std::uint64_t get_u64(
    std::span<const std::byte> input, std::size_t offset);

[[nodiscard]] CommonHeader decode_common_header(
    std::span<const std::byte> bytes,
    const std::array<char, 8>& expected_magic,
    std::string_view description);

[[nodiscard]] std::array<std::byte, kHeaderSize> encode_manifest(
    const Manifest& manifest);
[[nodiscard]] Manifest decode_manifest(std::span<const std::byte> bytes);
[[nodiscard]] std::array<std::byte, kHeaderSize> encode_shard_header(
    const ShardHeader& header);
[[nodiscard]] ShardHeader decode_shard_header(
    std::span<const std::byte> bytes,
    FormatVersion expected_format_version);
[[nodiscard]] std::array<std::byte, kHeaderSize> encode_staging_header(
    std::uint64_t count);
[[nodiscard]] std::uint64_t decode_staging_header(
    std::span<const std::byte> bytes);

void write_manifest_atomic(
    const std::filesystem::path& store_path, const Manifest& manifest);
[[nodiscard]] Manifest read_manifest(const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path shard_path(
    const std::filesystem::path& store_path, std::uint64_t shard_id);

[[nodiscard]] std::uint64_t hash_id(std::uint64_t id, std::uint64_t seed) noexcept;
[[nodiscard]] IndexData build_index(
    int staging_fd, const std::filesystem::path& staging_path,
    std::uint64_t count, std::uint64_t seed);
void write_index_atomic(
    const std::filesystem::path& store_path, const IndexData& index,
    Manifest& manifest);
[[nodiscard]] IndexData read_index(
    const std::filesystem::path& store_path, const Manifest& manifest);
[[nodiscard]] bool index_lookup(
    const IndexData& index, std::uint64_t id, std::uint64_t& ordinal) noexcept;

}  // namespace throwdb::detail
