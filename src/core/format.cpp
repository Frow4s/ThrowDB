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

#ifdef THROWDB_NATIVE_TEST_HOOKS
std::atomic<FileDescriptorCloseTestHook> descriptor_close_hook{nullptr};
#endif

int close_descriptor(const int fd) noexcept {
#ifdef THROWDB_NATIVE_TEST_HOOKS
    const auto hook = descriptor_close_hook.load(std::memory_order_acquire);
    if (hook != nullptr) {
        return hook(fd);
    }
#endif
    return ::close(fd);
}

[[nodiscard]] std::uint64_t mix_snapshot_word(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

#ifdef THROWDB_NATIVE_TEST_HOOKS
void set_file_descriptor_close_test_hook(
    const FileDescriptorCloseTestHook hook) noexcept {
    descriptor_close_hook.store(hook, std::memory_order_release);
}
#endif

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
    const int previous = std::exchange(value, fd);
    if (previous >= 0) {
#if defined(__linux__)
        // Linux releases the descriptor even when close reports EINTR. A
        // retry could close an unrelated file that another thread just opened
        // with the reused number. Durability is established by explicit fsync.
        // https://man7.org/linux/man-pages/man2/close.2.html
        static_cast<void>(close_descriptor(previous));
#else
        while (close_descriptor(previous) < 0 && errno == EINTR) {
        }
#endif
    }
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

std::uint64_t hash_id(std::uint64_t id, const std::uint64_t seed) noexcept {
    id += seed + 0x9e3779b97f4a7c15ULL;
    id = (id ^ (id >> 30)) * 0xbf58476d1ce4e5b9ULL;
    id = (id ^ (id >> 27)) * 0x94d049bb133111ebULL;
    return id ^ (id >> 31);
}

}  // namespace throwdb::detail
