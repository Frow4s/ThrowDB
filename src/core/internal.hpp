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

enum class DTypeCode : std::uint32_t {
    Invalid = 0,
    Float16 = 1,
    Float32 = 2,
};

enum class IdType : std::uint32_t {
    Invalid = 0,
    UInt64 = 1,
};

enum class RowLayout : std::uint32_t {
    Invalid = 0,
    DenseRowMajor = 1,
};

enum class IndexKind : std::uint32_t {
    Invalid = 0,
    UInt64PartitionedRowRef = 3,
};

enum class HashAlgorithm : std::uint32_t {
    Invalid = 0,
    SplitMix64V1 = 1,
};

enum class ChecksumAlgorithm : std::uint32_t {
    None = 0,
    Crc32c = 1,
};

inline constexpr std::uint64_t kHashSeed = 0x6a09e667f3bcc909ULL;

#ifdef THROWDB_NATIVE_TEST_HOOKS
// Private syscall seam for deterministic descriptor-lifecycle tests. Absent
// from production builds and the installed public headers.
using FileDescriptorCloseTestHook = int (*)(int fd) noexcept;
void set_file_descriptor_close_test_hook(
    FileDescriptorCloseTestHook hook) noexcept;
#endif

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

    bool operator==(const CollectionSchema&) const = default;
};

[[nodiscard]] std::string system_error_message(
    std::string_view operation,
    const std::filesystem::path& path,
    int error_number = errno);

[[nodiscard]] FileDescriptor open_read_only(const std::filesystem::path& path);
[[nodiscard]] FileDescriptor open_read_write(const std::filesystem::path& path);
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
void truncate_file(int fd, std::uint64_t size,
                   const std::filesystem::path& path);

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

[[nodiscard]] std::uint64_t hash_id(std::uint64_t id, std::uint64_t seed) noexcept;

}  // namespace throwdb::detail
