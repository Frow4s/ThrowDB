#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace throwdb {

MissingIdError::MissingIdError(const std::uint64_t id)
    : std::out_of_range("embedding ID not found: " + std::to_string(id)), id_(id) {}

DuplicateIdError::DuplicateIdError(const std::uint64_t id)
    : std::invalid_argument("duplicate embedding ID: " + std::to_string(id)), id_(id) {}

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

void validate_common_header(std::span<const std::byte> bytes,
                            std::string_view description) {
    if (bytes.size() < kHeaderSize) {
        throw CorruptionError(std::string(description) + " header is truncated");
    }
    if (get_u32(bytes, 8) != kFormatVersion) {
        throw CorruptionError(std::string(description) + " has unsupported format version");
    }
    if (get_u32(bytes, 12) != kHeaderSize) {
        throw CorruptionError(std::string(description) + " has invalid header size");
    }
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

std::array<std::byte, kHeaderSize> encode_manifest(const Manifest& manifest) {
    std::array<std::byte, kHeaderSize> output{};
    put_magic(output, kManifestMagic);
    put_u32(output, 8, kFormatVersion);
    put_u32(output, 12, kHeaderSize);
    put_u32(output, 16, manifest.dtype);
    put_u32(output, 20, manifest.dimension);
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
}

Manifest decode_manifest(const std::span<const std::byte> bytes) {
    require_magic(bytes, kManifestMagic, "manifest");
    validate_common_header(bytes, "manifest");
    Manifest result;
    result.dtype = get_u32(bytes, 16);
    result.dimension = get_u32(bytes, 20);
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
    if (result.dimension == 0 || result.record_size == 0
        || result.records_per_shard == 0 || result.index_capacity == 0) {
        throw CorruptionError("manifest contains zero-valued required fields");
    }
    return result;
}

std::array<std::byte, kHeaderSize> encode_shard_header(const ShardHeader& header) {
    std::array<std::byte, kHeaderSize> output{};
    put_magic(output, kShardMagic);
    put_u32(output, 8, kFormatVersion);
    put_u32(output, 12, kHeaderSize);
    put_u64(output, 16, header.shard_id);
    put_u64(output, 24, header.first_ordinal);
    put_u64(output, 32, header.record_count);
    put_u64(output, 40, header.record_size);
    put_u32(output, 48, header.dimension);
    put_u32(output, 52, header.dtype);
    return output;
}

ShardHeader decode_shard_header(const std::span<const std::byte> bytes) {
    require_magic(bytes, kShardMagic, "shard");
    validate_common_header(bytes, "shard");
    ShardHeader result;
    result.shard_id = get_u64(bytes, 16);
    result.first_ordinal = get_u64(bytes, 24);
    result.record_count = get_u64(bytes, 32);
    result.record_size = get_u64(bytes, 40);
    result.dimension = get_u32(bytes, 48);
    result.dtype = get_u32(bytes, 52);
    return result;
}

std::array<std::byte, kHeaderSize> encode_staging_header(const std::uint64_t count) {
    std::array<std::byte, kHeaderSize> output{};
    put_magic(output, kStagingMagic);
    put_u32(output, 8, kFormatVersion);
    put_u32(output, 12, kHeaderSize);
    put_u64(output, 16, count);
    put_u64(output, 24, kStagingEntrySize);
    return output;
}

std::uint64_t decode_staging_header(const std::span<const std::byte> bytes) {
    require_magic(bytes, kStagingMagic, "staging log");
    validate_common_header(bytes, "staging log");
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
