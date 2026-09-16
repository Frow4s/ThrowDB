#include "read_workload_log.hpp"
#include "read_workload_write_budget.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "crc32c.hpp"
#include "throwdb/errors.hpp"

namespace throwdb::server::detail {
namespace {

using namespace std::chrono_literals;

constexpr std::array<char, 8> kLogMagic{
    'T', 'D', 'B', 'R', 'L', 'O', 'G', '2'};
constexpr std::array<char, 8> kRecordMagic{
    'T', 'D', 'B', 'R', 'R', 'E', 'C', '2'};
constexpr std::uint32_t kLogVersion = 2;
constexpr std::uint32_t kLogHeaderUsedBytes = 232;
constexpr std::uint32_t kRecordVersion = 2;
constexpr std::uint32_t kRequestBeginRecordKind = 1;
constexpr std::uint32_t kIdChunkRecordKind = 2;
constexpr std::uint32_t kBatchMetricRecordKind = 3;
constexpr std::uint32_t kRequestEndRecordKind = 4;
constexpr std::uint32_t kCheckpointRecordKind = 5;
constexpr std::uint32_t kCheckpointExtensionVersion = 1;
constexpr std::uint32_t kCheckpointFlagOpeningBaseline = 1U << 0U;
constexpr std::uint32_t kRecordFlagExactSnapshot = 1U << 0U;
constexpr std::uint32_t kRecordFlagDeclaredIdsKnown = 1U << 1U;
constexpr std::uint32_t kRecordFlagCompleteForTraining = 1U << 0U;
constexpr std::size_t kLogHeaderCrcOffset = 24;
constexpr std::uint64_t kCheckpointEventInterval = 1'024;
constexpr std::uint64_t kMaximumScannedDirectoryEntries = 65'536;
constexpr std::uint64_t kMaximumRetainedFiles =
    kMaximumScannedDirectoryEntries - 3;
constexpr std::uint64_t kMaximumRetentionSweepSeconds = 300;
constexpr std::string_view kWriterLockName = ".read-workload.writer.lock";
constexpr std::string_view kDirectoryMarkerName = ".read-workload.owner";
constexpr std::string_view kFinalizedExtension = ".tdbrl";
constexpr std::string_view kOpenExtension = ".open";
constexpr std::array<char, 8> kDirectoryMarkerMagic{
    'T', 'D', 'B', 'R', 'D', 'I', 'R', '1'};
constexpr std::size_t kDirectoryMarkerBytes = 128;

class FileDescriptor {
public:
    explicit FileDescriptor(const int value = -1) noexcept : value_(value) {}
    ~FileDescriptor() {
        reset();
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept
        : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept {
        return value_ >= 0;
    }
    int release() noexcept { return std::exchange(value_, -1); }
    void reset(const int value = -1) noexcept {
        if (value_ >= 0) {
            // close(EINTR) leaves descriptor state platform-dependent; never
            // risk closing a descriptor another thread reused.
            ::close(value_);
        }
        value_ = value;
    }

private:
    int value_ = -1;
};

class DirectoryStream {
public:
    explicit DirectoryStream(DIR* value = nullptr) noexcept : value_(value) {}
    ~DirectoryStream() {
        if (value_ != nullptr) {
            ::closedir(value_);
        }
    }
    DirectoryStream(const DirectoryStream&) = delete;
    DirectoryStream& operator=(const DirectoryStream&) = delete;
    [[nodiscard]] DIR* get() const noexcept { return value_; }

private:
    DIR* value_ = nullptr;
};

[[nodiscard]] std::uint64_t unix_time_ns() noexcept {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto count =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    return count <= 0 ? 0 : static_cast<std::uint64_t>(count);
}

[[nodiscard]] std::uint64_t steady_time_ns() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto count =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    return count <= 0 ? 0 : static_cast<std::uint64_t>(count);
}

[[nodiscard]] constexpr std::uint64_t steady_clock_resolution_ns() noexcept {
    constexpr auto numerator =
        std::uint64_t{1'000'000'000}
        * static_cast<std::uint64_t>(std::chrono::steady_clock::period::num);
    constexpr auto denominator =
        static_cast<std::uint64_t>(
            std::chrono::steady_clock::period::den);
    return std::max<std::uint64_t>(
        1, (numerator + denominator - 1) / denominator);
}

[[nodiscard]] std::uint64_t saturating_add_value(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

void atomic_add_saturating(
    std::atomic<std::uint64_t>& value,
    const std::uint64_t amount = 1) noexcept {
    auto current = value.load(std::memory_order_relaxed);
    while (true) {
        const auto next = saturating_add_value(current, amount);
        if (value.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

void atomic_max(
    std::atomic<std::uint64_t>& value,
    const std::uint64_t candidate) noexcept {
    auto current = value.load(std::memory_order_relaxed);
    while (candidate > current
           && !value.compare_exchange_weak(
               current,
               candidate,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

void put_u32(
    const std::span<std::byte> output,
    const std::size_t offset,
    const std::uint32_t value) noexcept {
    for (std::size_t index = 0; index != 4; ++index) {
        output[offset + index] =
            static_cast<std::byte>(value >> (8U * index));
    }
}

void put_u64(
    const std::span<std::byte> output,
    const std::size_t offset,
    const std::uint64_t value) noexcept {
    for (std::size_t index = 0; index != 8; ++index) {
        output[offset + index] =
            static_cast<std::byte>(value >> (8U * index));
    }
}

[[nodiscard]] std::uint32_t get_u32(
    const std::span<const std::byte> input,
    const std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index) {
        value |= std::uint32_t(
                     std::to_integer<std::uint8_t>(input[offset + index]))
                 << (8U * index);
    }
    return value;
}

[[nodiscard]] std::uint64_t get_u64(
    const std::span<const std::byte> input,
    const std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != 8; ++index) {
        value |= std::uint64_t(
                     std::to_integer<std::uint8_t>(input[offset + index]))
                 << (8U * index);
    }
    return value;
}

[[nodiscard]] std::uint32_t crc32c_with_zeroed_field(
    const std::span<const std::byte> bytes,
    const std::size_t offset) noexcept {
    throwdb::detail::Crc32c crc;
    crc.update(bytes.first(offset));
    crc.update_zeroes(4);
    crc.update(bytes.subspan(offset + 4));
    return crc.value();
}

[[nodiscard]] bool has_magic(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic) noexcept {
    return bytes.size() >= magic.size()
           && std::memcmp(bytes.data(), magic.data(), magic.size()) == 0;
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] std::string hex_u64(const std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

[[nodiscard]] std::string collection_token(
    const ReadWorkloadCollectionInfo& collection) {
    return hex_u64(collection.collection_id_high)
           + hex_u64(collection.collection_id_low);
}

[[nodiscard]] std::string instance_token(
    const std::uint64_t low,
    const std::uint64_t high) {
    return hex_u64(high) + hex_u64(low);
}

[[nodiscard]] StorageError io_error(
    const std::string& operation,
    const std::filesystem::path& path,
    const int error_number = errno) {
    return StorageError(
        operation + " '" + path.string() + "': "
        + std::strerror(error_number));
}

void write_all(
    const int fd,
    const std::span<const std::byte> bytes,
    const std::filesystem::path& path) {
    std::size_t cursor = 0;
    while (cursor != bytes.size()) {
        const auto amount =
            ::write(fd, bytes.data() + cursor, bytes.size() - cursor);
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw io_error("cannot write read-workload log", path);
        }
        if (amount == 0) {
            throw StorageError(
                "read-workload log append made no progress: "
                + path.string());
        }
        cursor += static_cast<std::size_t>(amount);
    }
}

[[nodiscard]] bool pread_all(
    const int fd,
    const std::span<std::byte> bytes,
    const std::uint64_t offset) noexcept {
    std::size_t cursor = 0;
    while (cursor != bytes.size()) {
        const auto amount = ::pread(
            fd,
            bytes.data() + cursor,
            bytes.size() - cursor,
            static_cast<off_t>(offset + cursor));
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (amount == 0) {
            return false;
        }
        cursor += static_cast<std::size_t>(amount);
    }
    return true;
}

void sync_file(const int fd, const std::filesystem::path& path) {
    while (::fsync(fd) != 0) {
        if (errno == EINTR) {
            continue;
        }
        throw io_error("cannot sync read-workload log", path);
    }
}

void evict_file_cache_best_effort(const int fd) noexcept {
#if defined(POSIX_FADV_DONTNEED)
    // Telemetry is consumed offline and should not displace collection data
    // from the page cache. The hint is deliberately non-fatal and is issued
    // only after fsync has made the segment durable.
    static_cast<void>(
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
#else
    static_cast<void>(fd);
#endif
}

void sync_directory(
    const int fd,
    const std::filesystem::path& path) {
    while (::fsync(fd) != 0) {
        if (errno == EINTR) {
            continue;
        }
        throw io_error("cannot sync read-workload log directory", path);
    }
}

void sync_directory_best_effort(const int fd) noexcept {
    while (::fsync(fd) != 0 && errno == EINTR) {
    }
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> random_instance_id() {
    FileDescriptor random_fd(
        ::open("/dev/urandom", O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!random_fd) {
        throw io_error(
            "cannot open read-workload entropy source", "/dev/urandom");
    }
    std::array<std::byte, 16> bytes{};
    std::size_t cursor = 0;
    while (cursor != bytes.size()) {
        const auto amount = ::read(
            random_fd.get(),
            bytes.data() + cursor,
            bytes.size() - cursor);
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw io_error(
                "cannot read read-workload entropy source", "/dev/urandom");
        }
        if (amount == 0) {
            throw StorageError(
                "read-workload entropy source ended unexpectedly");
        }
        cursor += static_cast<std::size_t>(amount);
    }
    auto low = get_u64(bytes, 0);
    auto high = get_u64(bytes, 8);
    if (low == 0 && high == 0) {
        low = 1;
    }
    return {low, high};
}

void validate_options(
    const ReadWorkloadLogOptions& options,
    const ReadWorkloadCollectionInfo& collection) {
    if (options.directory.empty()) {
        throw InvalidArgumentError(
            "read-workload log directory must not be empty");
    }
    if (options.id_chunk_rows == 0
        || options.id_chunk_rows > kReadWorkloadMaximumIdChunkRows) {
        throw InvalidArgumentError(
            "read-workload ID chunk rows must be in [1, 8192]");
    }
    if (options.max_active_requests == 0
        || options.max_active_requests
               > kReadWorkloadMaximumActiveRequests) {
        throw InvalidArgumentError(
            "read-workload max active requests must be in [1, 65536]");
    }
    const auto maximum_frame = std::max<std::uint64_t>(
        kReadWorkloadBatchRecordBytes,
        kReadWorkloadRecordHeaderBytes
            + std::uint64_t{options.id_chunk_rows}
                  * sizeof(std::uint64_t));
    if (maximum_frame
            > std::numeric_limits<std::uint64_t>::max() / 2
        || options.buffer_bytes < maximum_frame * 2
        || options.buffer_bytes
               > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
        throw InvalidArgumentError(
            "read-workload buffer must hold at least two maximum records");
    }
    if (options.segment_bytes
        < kReadWorkloadLogHeaderBytes
              + 2 * kReadWorkloadRecordHeaderBytes + maximum_frame) {
        throw InvalidArgumentError(
            "read-workload segment must hold its header, opening checkpoint, "
            "one maximum record, and terminal checkpoint");
    }
    if (options.rotate_seconds == 0
        || options.rotate_seconds
               > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())
        || options.retention_bytes == 0
        || options.retention_seconds == 0) {
        throw InvalidArgumentError(
            "read-workload rotation and retention bounds must be nonzero");
    }
    if (options.retention_bytes < options.segment_bytes) {
        throw InvalidArgumentError(
            "read-workload retention bytes must be at least one segment");
    }
    const auto maximum_write_transaction =
        kReadWorkloadLogHeaderBytes + maximum_frame
        + 2 * kReadWorkloadRecordHeaderBytes;
    if (options.max_write_bytes_per_second < maximum_write_transaction) {
        throw InvalidArgumentError(
            "read-workload write budget must cover one maximum record, a "
            "segment header, opening checkpoint, and terminal checkpoint "
            "per second");
    }
    if (options.max_retained_files == 0
        || options.max_retained_files > kMaximumRetainedFiles
        || options.max_retained_files
               > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
        throw InvalidArgumentError(
            "read-workload retained-file bound must be in [1, 65533]");
    }
    if ((collection.collection_id_low == 0
         && collection.collection_id_high == 0)
        || collection.format == 0 || collection.dimension == 0
        || (collection.dtype != 1 && collection.dtype != 2)) {
        throw InvalidArgumentError(
            "read-workload collection identity and schema are invalid");
    }
    const std::uint64_t item_size = collection.dtype == 1 ? 2 : 4;
    if (collection.record_size_bytes
        != std::uint64_t{collection.dimension} * item_size) {
        throw InvalidArgumentError(
            "read-workload record size does not match dimension and dtype");
    }
    if (collection.reader_backend > 2
        || collection.metric_page_bytes == 0
        || (collection.metric_page_bytes
            & (collection.metric_page_bytes - 1))
               != 0
        || collection.metric_chunk_bytes < collection.metric_page_bytes
        || (collection.metric_chunk_bytes
            & (collection.metric_chunk_bytes - 1))
               != 0
        || collection.tuning_explicit_overrides > 0xffU) {
        throw InvalidArgumentError(
            "read-workload reader context is invalid");
    }
    constexpr std::uint32_t maximum_io_workers = 1'024;
    constexpr std::uint32_t maximum_io_queue_capacity = 1U << 20;
    constexpr std::uint32_t maximum_io_ranges_per_task = 1U << 20;
    constexpr std::uint32_t maximum_io_uring_queue_depth = 4'096;
    constexpr std::uint32_t maximum_io_uring_workers = 64;
    if ((collection.reader_backend == 1 || collection.reader_backend == 2)
        && (collection.io_ranges_per_task == 0
            || collection.io_ranges_per_task
                   > maximum_io_ranges_per_task
            || collection.max_read_range_bytes
                   < collection.record_size_bytes
            || collection.io_uring_queue_depth == 0
            || collection.io_uring_queue_depth
                   > maximum_io_uring_queue_depth
            || collection.io_uring_workers == 0
            || collection.io_uring_workers > maximum_io_uring_workers
            || collection.io_uring_queue_depth
                   < collection.io_uring_workers)) {
        throw InvalidArgumentError(
            "read-workload reader context is invalid");
    }
    if (collection.reader_backend == 1
        && (collection.io_workers > maximum_io_workers
            || collection.io_queue_capacity > maximum_io_queue_capacity
            || (collection.io_workers == 0
                && collection.io_queue_capacity != 0))) {
        throw InvalidArgumentError(
            "read-workload thread-pool context is invalid");
    }
    if (collection.reader_backend == 2
        && (collection.io_workers != 0
            || collection.io_queue_capacity != 0)) {
        throw InvalidArgumentError(
            "read-workload io_uring context is invalid");
    }
}

[[nodiscard]] ReadWorkloadLogOptions validated_options(
    ReadWorkloadLogOptions options,
    const ReadWorkloadCollectionInfo& collection) {
    validate_options(options, collection);
    return options;
}

[[nodiscard]] std::array<std::byte, kReadWorkloadLogHeaderBytes>
encode_log_header(
    const ReadWorkloadLogOptions& options,
    const ReadWorkloadCollectionInfo& collection,
    const std::uint64_t instance_low,
    const std::uint64_t instance_high,
    const std::uint64_t created_ns,
    const std::uint64_t segment_sequence) noexcept {
    std::array<std::byte, kReadWorkloadLogHeaderBytes> header{};
    std::memcpy(header.data(), kLogMagic.data(), kLogMagic.size());
    put_u32(header, 8, kLogVersion);
    put_u32(
        header, 12, static_cast<std::uint32_t>(header.size()));
    put_u32(header, 16, kLogHeaderUsedBytes);
    put_u32(header, 20, 0);
    put_u64(header, 32, collection.collection_id_low);
    put_u64(header, 40, collection.collection_id_high);
    put_u32(header, 48, collection.format);
    put_u32(header, 52, collection.dtype);
    put_u32(header, 56, collection.dimension);
    put_u32(header, 60, collection.record_size_bytes);
    put_u64(header, 64, instance_low);
    put_u64(header, 72, instance_high);
    put_u64(header, 80, created_ns);
    put_u32(header, 88, options.id_chunk_rows);
    put_u32(header, 92, options.max_active_requests);
    put_u64(header, 96, options.buffer_bytes);
    put_u64(header, 104, options.segment_bytes);
    put_u64(header, 112, options.rotate_seconds);
    put_u64(header, 120, options.retention_bytes);
    put_u64(header, 128, options.retention_seconds);
    put_u64(header, 136, options.min_free_bytes);
    put_u64(header, 144, static_cast<std::uint64_t>(::getpid()));
    put_u64(header, 152, segment_sequence);
    put_u64(header, 160, options.max_write_bytes_per_second);
    put_u64(header, 168, options.max_retained_files);
    put_u32(header, 176, collection.reader_backend);
    put_u32(header, 180, collection.io_workers);
    put_u32(header, 184, collection.io_queue_capacity);
    put_u32(header, 188, collection.io_ranges_per_task);
    put_u64(header, 192, collection.max_read_range_bytes);
    put_u64(header, 200, collection.max_gap_bytes);
    put_u32(header, 208, collection.io_uring_queue_depth);
    put_u32(header, 212, collection.io_uring_workers);
    put_u32(header, 216, collection.metric_page_bytes);
    put_u32(header, 220, collection.metric_chunk_bytes);
    put_u32(header, 224, collection.tuning_explicit_overrides);
    put_u32(
        header,
        kLogHeaderCrcOffset,
        crc32c_with_zeroed_field(header, kLogHeaderCrcOffset));
    return header;
}

[[nodiscard]] std::array<std::byte, kDirectoryMarkerBytes>
encode_directory_marker(
    const ReadWorkloadCollectionInfo& collection) noexcept {
    std::array<std::byte, kDirectoryMarkerBytes> marker{};
    std::memcpy(
        marker.data(),
        kDirectoryMarkerMagic.data(),
        kDirectoryMarkerMagic.size());
    put_u32(marker, 8, 1);
    put_u32(marker, 12, static_cast<std::uint32_t>(marker.size()));
    put_u32(marker, 16, 72);
    put_u32(marker, 20, 0);
    put_u64(marker, 32, collection.collection_id_low);
    put_u64(marker, 40, collection.collection_id_high);
    put_u32(marker, 48, collection.format);
    put_u32(marker, 52, collection.dtype);
    put_u32(marker, 56, collection.dimension);
    put_u32(marker, 60, collection.record_size_bytes);
    put_u64(marker, 64, static_cast<std::uint64_t>(::geteuid()));
    put_u32(
        marker,
        kLogHeaderCrcOffset,
        crc32c_with_zeroed_field(marker, kLogHeaderCrcOffset));
    return marker;
}

[[nodiscard]] bool valid_directory_marker(
    const std::span<const std::byte> marker,
    const ReadWorkloadCollectionInfo& collection) noexcept {
    return marker.size() == kDirectoryMarkerBytes
           && has_magic(marker, kDirectoryMarkerMagic)
           && get_u32(marker, 8) == 1
           && get_u32(marker, 12) == kDirectoryMarkerBytes
           && get_u32(marker, 16) == 72
           && get_u32(marker, 20) == 0
           && get_u32(marker, kLogHeaderCrcOffset)
                  == crc32c_with_zeroed_field(
                      marker, kLogHeaderCrcOffset)
           && get_u64(marker, 32) == collection.collection_id_low
           && get_u64(marker, 40) == collection.collection_id_high
           && get_u32(marker, 48) == collection.format
           && get_u32(marker, 52) == collection.dtype
           && get_u32(marker, 56) == collection.dimension
           && get_u32(marker, 60) == collection.record_size_bytes
           && get_u64(marker, 64)
                  == static_cast<std::uint64_t>(::geteuid());
}

struct AtomicStats {
    std::atomic<std::uint64_t> observed_requests{0};
    std::atomic<std::uint64_t> begun_requests{0};
    std::atomic<std::uint64_t> completed_requests{0};
    std::atomic<std::uint64_t> complete_requests{0};
    std::atomic<std::uint64_t> incomplete_requests{0};
    std::atomic<std::uint64_t> observed_records{0};
    std::atomic<std::uint64_t> enqueued_records{0};
    std::atomic<std::uint64_t> written_records{0};
    std::atomic<std::uint64_t> observed_id_chunks{0};
    std::atomic<std::uint64_t> enqueued_id_chunks{0};
    std::atomic<std::uint64_t> written_id_chunks{0};
    std::atomic<std::uint64_t> observed_batches{0};
    std::atomic<std::uint64_t> enqueued_batches{0};
    std::atomic<std::uint64_t> written_batches{0};
    std::atomic<std::uint64_t> observed_ids{0};
    std::atomic<std::uint64_t> enqueued_ids{0};
    std::atomic<std::uint64_t> written_ids{0};
    std::atomic<std::uint64_t> dropped_full_records{0};
    std::atomic<std::uint64_t> dropped_full_ids{0};
    std::atomic<std::uint64_t> dropped_runtime_failure_records{0};
    std::atomic<std::uint64_t> dropped_runtime_failure_ids{0};
    std::atomic<std::uint64_t> dropped_min_free_records{0};
    std::atomic<std::uint64_t> dropped_min_free_ids{0};
    std::atomic<std::uint64_t> dropped_io_budget_records{0};
    std::atomic<std::uint64_t> dropped_io_budget_ids{0};
    std::atomic<std::uint64_t> write_errors{0};
    std::atomic<std::uint64_t> bytes_written{0};
    std::atomic<std::uint64_t> segments_created{0};
    std::atomic<std::uint64_t> rotated_segments{0};
    std::atomic<std::uint64_t> retained_bytes{0};
    std::atomic<std::uint64_t> retained_segments{0};
    std::atomic<std::uint64_t> retention_deleted_bytes{0};
    std::atomic<std::uint64_t> retention_deleted_segments{0};
    std::atomic<std::uint64_t> queued_records{0};
    std::atomic<std::uint64_t> queued_bytes{0};
    std::atomic<std::uint64_t> peak_queued_bytes{0};
    std::atomic<std::uint64_t> active_requests{0};
    std::atomic<int> last_error_number{0};
    std::atomic<bool> paused_for_min_free{false};
    std::atomic<bool> runtime_failed{false};
    std::atomic<bool> shutting_down{false};
};

struct CounterSnapshot {
    std::uint64_t observed_requests = 0;
    std::uint64_t begun_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t complete_requests = 0;
    std::uint64_t incomplete_requests = 0;
    std::uint64_t observed_records = 0;
    std::uint64_t enqueued_records = 0;
    std::uint64_t written_records = 0;
    std::uint64_t observed_ids = 0;
    std::uint64_t enqueued_ids = 0;
    std::uint64_t written_ids = 0;
    std::uint64_t observed_id_chunks = 0;
    std::uint64_t enqueued_id_chunks = 0;
    std::uint64_t written_id_chunks = 0;
    std::uint64_t observed_batches = 0;
    std::uint64_t enqueued_batches = 0;
    std::uint64_t written_batches = 0;
    std::uint64_t dropped_full_records = 0;
    std::uint64_t dropped_full_ids = 0;
    std::uint64_t dropped_runtime_failure_records = 0;
    std::uint64_t dropped_runtime_failure_ids = 0;
    std::uint64_t dropped_min_free_records = 0;
    std::uint64_t dropped_min_free_ids = 0;
    std::uint64_t dropped_io_budget_records = 0;
    std::uint64_t dropped_io_budget_ids = 0;
    std::uint64_t write_errors = 0;
    std::uint64_t bytes_written = 0;
    std::uint64_t segments_created = 0;
    std::uint64_t rotated_segments = 0;

    bool operator==(const CounterSnapshot&) const = default;
};

[[nodiscard]] CounterSnapshot counter_snapshot(
    const AtomicStats& stats) noexcept {
    CounterSnapshot result;
    result.observed_requests =
        stats.observed_requests.load(std::memory_order_relaxed);
    result.begun_requests =
        stats.begun_requests.load(std::memory_order_relaxed);
    result.completed_requests =
        stats.completed_requests.load(std::memory_order_relaxed);
    result.complete_requests =
        stats.complete_requests.load(std::memory_order_relaxed);
    result.incomplete_requests =
        stats.incomplete_requests.load(std::memory_order_relaxed);
    result.observed_records =
        stats.observed_records.load(std::memory_order_relaxed);
    result.enqueued_records =
        stats.enqueued_records.load(std::memory_order_relaxed);
    result.written_records =
        stats.written_records.load(std::memory_order_relaxed);
    result.observed_ids =
        stats.observed_ids.load(std::memory_order_relaxed);
    result.enqueued_ids =
        stats.enqueued_ids.load(std::memory_order_relaxed);
    result.written_ids =
        stats.written_ids.load(std::memory_order_relaxed);
    result.observed_id_chunks =
        stats.observed_id_chunks.load(std::memory_order_relaxed);
    result.enqueued_id_chunks =
        stats.enqueued_id_chunks.load(std::memory_order_relaxed);
    result.written_id_chunks =
        stats.written_id_chunks.load(std::memory_order_relaxed);
    result.observed_batches =
        stats.observed_batches.load(std::memory_order_relaxed);
    result.enqueued_batches =
        stats.enqueued_batches.load(std::memory_order_relaxed);
    result.written_batches =
        stats.written_batches.load(std::memory_order_relaxed);
    result.dropped_full_records =
        stats.dropped_full_records.load(std::memory_order_relaxed);
    result.dropped_full_ids =
        stats.dropped_full_ids.load(std::memory_order_relaxed);
    result.dropped_runtime_failure_records =
        stats.dropped_runtime_failure_records.load(
            std::memory_order_relaxed);
    result.dropped_runtime_failure_ids =
        stats.dropped_runtime_failure_ids.load(
            std::memory_order_relaxed);
    result.dropped_min_free_records =
        stats.dropped_min_free_records.load(std::memory_order_relaxed);
    result.dropped_min_free_ids =
        stats.dropped_min_free_ids.load(std::memory_order_relaxed);
    result.dropped_io_budget_records =
        stats.dropped_io_budget_records.load(std::memory_order_relaxed);
    result.dropped_io_budget_ids =
        stats.dropped_io_budget_ids.load(std::memory_order_relaxed);
    result.write_errors =
        stats.write_errors.load(std::memory_order_relaxed);
    result.bytes_written =
        stats.bytes_written.load(std::memory_order_relaxed);
    result.segments_created =
        stats.segments_created.load(std::memory_order_relaxed);
    result.rotated_segments =
        stats.rotated_segments.load(std::memory_order_relaxed);

    result.completed_requests = saturating_add_value(
        result.complete_requests, result.incomplete_requests);
    result.observed_requests = std::max(
        {result.observed_requests,
         result.begun_requests,
         result.completed_requests});
    result.enqueued_records =
        std::max(result.enqueued_records, result.written_records);
    result.observed_records =
        std::max(result.observed_records, result.enqueued_records);
    result.enqueued_ids =
        std::max(result.enqueued_ids, result.written_ids);
    result.observed_ids =
        std::max(result.observed_ids, result.enqueued_ids);
    result.enqueued_id_chunks =
        std::max(result.enqueued_id_chunks, result.written_id_chunks);
    result.observed_id_chunks =
        std::max(result.observed_id_chunks, result.enqueued_id_chunks);
    result.enqueued_batches =
        std::max(result.enqueued_batches, result.written_batches);
    result.observed_batches =
        std::max(result.observed_batches, result.enqueued_batches);
    result.segments_created =
        std::max(result.segments_created, result.rotated_segments);
    return result;
}

void encode_counter_snapshot(
    const std::span<std::byte> output,
    const CounterSnapshot& counters) noexcept {
    put_u64(output, 56, counters.observed_requests);
    put_u64(output, 64, counters.begun_requests);
    put_u64(output, 72, counters.completed_requests);
    put_u64(output, 80, counters.complete_requests);
    put_u64(output, 88, counters.incomplete_requests);
    put_u64(output, 96, counters.observed_records);
    put_u64(output, 104, counters.enqueued_records);
    put_u64(output, 112, counters.written_records);
    put_u64(output, 120, counters.observed_ids);
    put_u64(output, 128, counters.enqueued_ids);
    put_u64(output, 136, counters.written_ids);
    put_u64(output, 144, counters.observed_id_chunks);
    put_u64(output, 152, counters.enqueued_id_chunks);
    put_u64(output, 160, counters.written_id_chunks);
    put_u64(output, 168, counters.observed_batches);
    put_u64(output, 176, counters.enqueued_batches);
    put_u64(output, 184, counters.written_batches);
    put_u64(output, 192, counters.dropped_full_records);
    put_u64(output, 200, counters.dropped_full_ids);
    put_u64(output, 208, counters.dropped_runtime_failure_records);
    put_u64(output, 216, counters.dropped_runtime_failure_ids);
    put_u64(output, 224, counters.dropped_min_free_records);
    put_u64(output, 232, counters.dropped_min_free_ids);
    put_u64(output, 240, counters.dropped_io_budget_records);
    put_u64(output, 248, counters.dropped_io_budget_ids);
}

[[nodiscard]] bool counters_equal(
    const CounterSnapshot& left,
    const CounterSnapshot& right) noexcept {
    // Storage bookkeeping changes when a checkpoint is appended. It must not
    // make that same checkpoint look dirty forever during an idle period.
    return left.observed_requests == right.observed_requests
           && left.begun_requests == right.begun_requests
           && left.completed_requests == right.completed_requests
           && left.complete_requests == right.complete_requests
           && left.incomplete_requests == right.incomplete_requests
           && left.observed_records == right.observed_records
           && left.enqueued_records == right.enqueued_records
           && left.written_records == right.written_records
           && left.observed_ids == right.observed_ids
           && left.enqueued_ids == right.enqueued_ids
           && left.written_ids == right.written_ids
           && left.observed_id_chunks == right.observed_id_chunks
           && left.enqueued_id_chunks == right.enqueued_id_chunks
           && left.written_id_chunks == right.written_id_chunks
           && left.observed_batches == right.observed_batches
           && left.enqueued_batches == right.enqueued_batches
           && left.written_batches == right.written_batches
           && left.dropped_full_records
                  == right.dropped_full_records
           && left.dropped_full_ids == right.dropped_full_ids
           && left.dropped_runtime_failure_records
                  == right.dropped_runtime_failure_records
           && left.dropped_runtime_failure_ids
                  == right.dropped_runtime_failure_ids
           && left.dropped_min_free_records
                  == right.dropped_min_free_records
           && left.dropped_min_free_ids == right.dropped_min_free_ids
           && left.dropped_io_budget_records
                  == right.dropped_io_budget_records
           && left.dropped_io_budget_ids == right.dropped_io_budget_ids
           && left.write_errors == right.write_errors;
}

struct RetainedFile {
    std::string name;
    std::uint64_t bytes = 0;
    std::int64_t modified_ns = 0;
    dev_t device = 0;
    ino_t inode = 0;
    bool finalized = false;
    bool recognized = false;
};

enum class RemovalResult {
    Removed,
    Missing,
    Failed,
};

enum class PrepareEventResult {
    Ready,
    MinimumFreeSpace,
    WriteBudget,
};

[[nodiscard]] std::int64_t stat_modified_ns(const struct stat& status) noexcept {
#if defined(__APPLE__)
    return std::int64_t{status.st_mtimespec.tv_sec} * 1'000'000'000
           + status.st_mtimespec.tv_nsec;
#else
    return std::int64_t{status.st_mtim.tv_sec} * 1'000'000'000
           + status.st_mtim.tv_nsec;
#endif
}

}  // namespace

class ReadWorkloadLogImpl {
public:
    ReadWorkloadLogImpl(
        ReadWorkloadLogOptions options,
        ReadWorkloadCollectionInfo collection)
        : options_(validated_options(std::move(options), collection)),
          collection_(collection),
          maximum_frame_bytes_(std::max<std::size_t>(
              kReadWorkloadBatchRecordBytes,
              kReadWorkloadRecordHeaderBytes
                  + std::size_t{options_.id_chunk_rows}
                        * sizeof(std::uint64_t))),
          slot_count_(
              static_cast<std::size_t>(
                  options_.buffer_bytes / maximum_frame_bytes_)),
          queue_storage_(slot_count_ * maximum_frame_bytes_),
          queue_slots_(std::make_unique<QueueSlot[]>(slot_count_)),
          retention_sweep_interval_(std::chrono::seconds(
              std::min(
                  options_.retention_seconds,
                  kMaximumRetentionSweepSeconds))) {
        for (std::size_t index = 0; index != slot_count_; ++index) {
            queue_slots_[index].sequence.store(
                index, std::memory_order_relaxed);
        }
        const auto ids = random_instance_id();
        instance_low_ = ids.first;
        instance_high_ = ids.second;
        prepare_directory_and_lock();
        prune_retention();
        constexpr std::uint64_t segment_bootstrap_bytes =
            kReadWorkloadLogHeaderBytes
            + kReadWorkloadRecordHeaderBytes;
        if (update_min_free_state(segment_bootstrap_bytes)
            && consume_write_budget(segment_bootstrap_bytes)) {
            create_segment();
        }
    }

    ~ReadWorkloadLogImpl() {
        shutdown();
    }

    void start() {
        worker_ = std::thread([this] { run(); });
    }

    [[nodiscard]] bool begin_request(
        ReadWorkloadRequestHandle& request,
        const ReadWorkloadRequestInfo& info) noexcept {
        if ((info.method != ReadWorkloadMethod::Get
             && info.method != ReadWorkloadMethod::StreamGet)
            || info.prefix_dimensions == 0
            || info.prefix_dimensions > collection_.dimension) {
            return false;
        }
        ProducerGuard producer_guard(producers_inflight_);
        bool active_admitted = false;
        bool lifecycle_begun = false;
        const auto reject_before_begin =
            [this]() noexcept {
                atomic_add_saturating(stats_.incomplete_requests);
                atomic_add_saturating(stats_.completed_requests);
                atomic_add_saturating(stats_.observed_requests);
            };
        try {
            if (stats_.runtime_failed.load(std::memory_order_acquire)
                || stats_.shutting_down.load(std::memory_order_acquire)) {
                reject_before_begin();
                atomic_add_saturating(stats_.observed_records);
                atomic_add_saturating(
                    stats_.dropped_runtime_failure_records);
                return false;
            }
            if (stats_.paused_for_min_free.load(
                std::memory_order_acquire)) {
                reject_before_begin();
                atomic_add_saturating(stats_.observed_records);
                atomic_add_saturating(stats_.dropped_min_free_records);
                return false;
            }
            auto active =
                stats_.active_requests.load(std::memory_order_relaxed);
            while (true) {
                if (active >= options_.max_active_requests) {
                    reject_before_begin();
                    atomic_add_saturating(stats_.observed_records);
                    atomic_add_saturating(stats_.dropped_full_records);
                    return false;
                }
                if (stats_.active_requests.compare_exchange_weak(
                        active,
                        active + 1,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    active_admitted = true;
                    break;
                }
            }
            // Publish lifecycle admission before the BEGIN queue slot becomes
            // visible to the writer. A rotation checkpoint can therefore
            // always derive the correct number of requests whose BEGIN lies
            // just before the retained window, without waiting for every
            // producer to become quiescent.
            atomic_add_saturating(stats_.begun_requests);
            atomic_add_saturating(stats_.observed_requests);
            lifecycle_begun = true;

            request.request_id_low_ = info.request_id_low;
            request.request_id_high_ = info.request_id_high;
            if (request.request_id_low_ == 0
                && request.request_id_high_ == 0) {
                const auto number = request_sequence_.fetch_add(
                                        1, std::memory_order_relaxed)
                                    + 1;
                request.request_id_low_ =
                    splitmix64(instance_low_ ^ number);
                request.request_id_high_ =
                    splitmix64(instance_high_ + number);
                if (request.request_id_low_ == 0
                    && request.request_id_high_ == 0) {
                    request.request_id_low_ = number;
                }
            }
            request.started_steady_ns_ = steady_time_ns();
            request.declared_ids_ = info.declared_ids;

            std::array<std::byte, kReadWorkloadRecordHeaderBytes> frame{};
            initialize_record(frame, kRequestBeginRecordKind);
            std::uint32_t flags = 0;
            if (info.snapshot.exact) {
                flags |= kRecordFlagExactSnapshot;
            }
            if (info.declared_ids != 0) {
                flags |= kRecordFlagDeclaredIdsKnown;
            }
            put_u32(frame, 28, flags);
            put_u32(
                frame, 32, static_cast<std::uint32_t>(info.method));
            put_u32(frame, 36, info.prefix_dimensions);
            put_u64(frame, 56, request.started_steady_ns_);
            put_u64(frame, 64, request.request_id_low_);
            put_u64(frame, 72, request.request_id_high_);
            put_u64(frame, 80, info.snapshot.catalog_generation);
            put_u64(frame, 88, info.snapshot.committed_sequence);
            put_u64(frame, 96, info.snapshot.row_count);
            put_u64(frame, 104, info.declared_ids);
            put_u64(frame, 112, steady_clock_resolution_ns());
            const auto result = enqueue_record(
                frame, kRequestBeginRecordKind, 0);
            if (result != EnqueueResult::Enqueued) {
                note_global_drop(result, 1, 0);
                atomic_add_saturating(stats_.incomplete_requests);
                atomic_add_saturating(stats_.completed_requests);
                stats_.active_requests.fetch_sub(
                    1, std::memory_order_release);
                active_admitted = false;
                return false;
            }
            return true;
        } catch (...) {
            if (active_admitted) {
                stats_.active_requests.fetch_sub(
                    1, std::memory_order_release);
            }
            atomic_add_saturating(stats_.incomplete_requests);
            atomic_add_saturating(stats_.completed_requests);
            if (!lifecycle_begun) {
                atomic_add_saturating(stats_.observed_requests);
            }
            atomic_add_saturating(stats_.observed_records);
            atomic_add_saturating(
                stats_.dropped_runtime_failure_records);
            return false;
        }
    }

    [[nodiscard]] bool append_ids(
        ReadWorkloadRequestHandle& request,
        const std::span<const std::uint64_t> ids,
        const ReadWorkloadIdChunkInfo& info) noexcept {
        if (ids.empty()) {
            return true;
        }
        ProducerGuard producer_guard(producers_inflight_);
        try {
            if (info.logical_offset != request.submitted_ids_) {
                request.logical_offsets_contiguous_ = false;
            }
            update_digest(request.digest_low_, ids);
            update_digest(request.digest_high_, ids);
            request.submitted_ids_ = saturating_add_value(
                request.submitted_ids_,
                static_cast<std::uint64_t>(ids.size()));

            std::array<
                std::byte,
                kReadWorkloadRecordHeaderBytes
                    + std::size_t{kReadWorkloadMaximumIdChunkRows}
                          * sizeof(std::uint64_t)>
                frame{};
            bool complete = true;
            std::size_t cursor = 0;
            while (cursor != ids.size()) {
                const auto count = std::min<std::size_t>(
                    options_.id_chunk_rows, ids.size() - cursor);
                const auto frame_bytes =
                    kReadWorkloadRecordHeaderBytes
                    + count * sizeof(std::uint64_t);
                auto output =
                    std::span<std::byte>(frame).first(frame_bytes);
                std::fill(output.begin(), output.end(), std::byte{0});
                initialize_record(output, kIdChunkRecordKind);
                put_u32(output, 32, static_cast<std::uint32_t>(count));
                put_u64(output, 64, request.request_id_low_);
                put_u64(output, 72, request.request_id_high_);
                put_u64(
                    output,
                    80,
                    saturating_add_value(
                        info.logical_offset,
                        static_cast<std::uint64_t>(cursor)));
                put_u64(output, 88, info.batch_sequence);
                // Delivery is only known after the response write succeeds
                // and is therefore carried by BATCH_METRICS, not by the
                // pre-read ID chunk.
                put_u64(output, 96, 0);
                put_u64(
                    output, 104, request.next_chunk_sequence_++);
                for (std::size_t index = 0; index != count; ++index) {
                    put_u64(
                        output,
                        kReadWorkloadRecordHeaderBytes
                            + index * sizeof(std::uint64_t),
                        ids[cursor + index]);
                }
                ++request.id_chunks_;
                const auto result = enqueue_record(
                    output,
                    kIdChunkRecordKind,
                    static_cast<std::uint64_t>(count));
                if (result == EnqueueResult::Enqueued) {
                    request.enqueued_ids_ = saturating_add_value(
                        request.enqueued_ids_,
                        static_cast<std::uint64_t>(count));
                } else {
                    complete = false;
                    note_request_drop(
                        request,
                        result,
                        1,
                        static_cast<std::uint64_t>(count));
                }
                cursor += count;
            }
            return complete;
        } catch (...) {
            note_request_drop(
                request,
                EnqueueResult::RuntimeFailure,
                1,
                static_cast<std::uint64_t>(ids.size()));
            return false;
        }
    }

    [[nodiscard]] bool append_batch(
        ReadWorkloadRequestHandle& request,
        const ReadWorkloadBatchMetrics& metrics) noexcept {
        ProducerGuard producer_guard(producers_inflight_);
        try {
            std::array<std::byte, kReadWorkloadBatchRecordBytes> frame{};
            initialize_record(frame, kBatchMetricRecordKind);
            put_u32(
                frame,
                32,
                static_cast<std::uint32_t>(metrics.backend));
            put_u64(frame, 64, request.request_id_low_);
            put_u64(frame, 72, request.request_id_high_);
            put_u64(frame, 80, metrics.batch_sequence);
            put_u64(frame, 88, metrics.logical_offset);
            put_u64(frame, 96, metrics.rows);
            put_u64(frame, 104, metrics.response_bytes);
            put_u64(frame, 112, metrics.total_ns);
            put_u64(frame, 120, metrics.lookup_ns);
            put_u64(frame, 128, metrics.sort_ns);
            put_u64(frame, 136, metrics.plan_ns);
            put_u64(frame, 144, metrics.scheduler_wait_ns);
            put_u64(frame, 152, metrics.io_ns);
            put_u64(frame, 160, metrics.scatter_ns);
            put_u64(frame, 168, metrics.encode_ns);
            put_u64(frame, 176, metrics.write_ns);
            put_u64(frame, 184, metrics.logger_queue_ns);
            put_u64(frame, 224, metrics.delivery_sequence);
            put_u64(frame, 232, metrics.useful_read_bytes);
            put_u64(frame, 240, metrics.backend_tasks);
            put_u64(frame, 256, metrics.unique_storage_segments);
            put_u64(frame, 264, metrics.physical_requests);
            put_u64(frame, 272, metrics.merged_ranges);
            put_u64(frame, 280, metrics.requested_bytes);
            put_u64(frame, 288, metrics.planned_bytes);
            put_u64(frame, 296, metrics.completed_bytes);
            put_u64(frame, 304, metrics.gap_bytes);
            put_u64(frame, 312, metrics.overread_bytes);
            put_u64(frame, 320, metrics.pages_touched);
            put_u64(frame, 328, metrics.page_runs);
            put_u64(frame, 336, metrics.layout_chunks_touched);
            put_u64(frame, 344, metrics.layout_chunk_runs);
            put_u64(frame, 352, metrics.discontinuities);
            put_u64(frame, 360, metrics.inversions);
            put_u64(frame, 368, metrics.max_range_bytes);
            put_u64(frame, 376, metrics.mean_range_bytes);
            ++request.batches_;
            request.response_bytes_ = saturating_add_value(
                request.response_bytes_, metrics.response_bytes);
            const auto result =
                enqueue_record(frame, kBatchMetricRecordKind, 0);
            if (result != EnqueueResult::Enqueued) {
                note_request_drop(request, result, 1, 0);
                return false;
            }
            return true;
        } catch (...) {
            note_request_drop(
                request, EnqueueResult::RuntimeFailure, 1, 0);
            return false;
        }
    }

    void end_request(
        ReadWorkloadRequestHandle& request,
        const ReadWorkloadRequestEnd& end) noexcept {
        ProducerGuard producer_guard(producers_inflight_);
        try {
            const bool complete =
                end.status == ReadWorkloadRequestStatus::Success
                && end.delivered_ids == request.submitted_ids_
                && (request.declared_ids_ == 0
                    || request.declared_ids_ == request.submitted_ids_)
                && request.dropped_records_ == 0
                && request.logical_offsets_contiguous_;
            std::array<std::byte, kReadWorkloadRecordHeaderBytes> frame{};
            initialize_record(frame, kRequestEndRecordKind);
            put_u32(
                frame,
                28,
                complete ? kRecordFlagCompleteForTraining : 0);
            put_u32(
                frame, 32, static_cast<std::uint32_t>(end.status));
            put_u32(frame, 36, end.error_code);
            put_u64(frame, 64, request.request_id_low_);
            put_u64(frame, 72, request.request_id_high_);
            put_u64(frame, 80, request.submitted_ids_);
            put_u64(frame, 88, end.delivered_ids);
            put_u64(frame, 96, request.id_chunks_);
            put_u64(frame, 104, request.batches_);
            put_u64(
                frame,
                112,
                end.response_bytes != 0 ? end.response_bytes
                                        : request.response_bytes_);
            put_u64(frame, 120, request.dropped_records_);
            put_u64(frame, 128, request.dropped_ids_);
            const auto finished = steady_time_ns();
            put_u64(
                frame,
                136,
                finished >= request.started_steady_ns_
                    ? finished - request.started_steady_ns_
                    : 0);
            put_u64(frame, 144, request.dropped_full_records_);
            put_u64(frame, 152, request.dropped_full_ids_);
            put_u64(frame, 160, request.dropped_runtime_records_);
            put_u64(frame, 168, request.dropped_runtime_ids_);
            put_u64(frame, 176, request.dropped_min_free_records_);
            put_u64(frame, 184, request.dropped_min_free_ids_);
            put_u64(frame, 192, request.dropped_io_budget_records_);
            put_u64(frame, 200, request.dropped_io_budget_ids_);
            put_u64(frame, 208, request.digest_low_);
            put_u64(frame, 216, request.digest_high_);

            atomic_add_saturating(stats_.completed_requests);
            const auto result =
                enqueue_record(frame, kRequestEndRecordKind, 0);
            if (result == EnqueueResult::Enqueued && complete) {
                atomic_add_saturating(stats_.complete_requests);
            } else {
                if (result != EnqueueResult::Enqueued) {
                    note_global_drop(result, 1, 0);
                }
                atomic_add_saturating(stats_.incomplete_requests);
            }
        } catch (...) {
            atomic_add_saturating(stats_.incomplete_requests);
            atomic_add_saturating(stats_.observed_records);
            atomic_add_saturating(
                stats_.dropped_runtime_failure_records);
        }
        release_request(request);
    }

    [[nodiscard]] ReadWorkloadLogStats stats() const noexcept {
        ReadWorkloadLogStats result;
        result.observed_requests =
            stats_.observed_requests.load(std::memory_order_relaxed);
        result.begun_requests =
            stats_.begun_requests.load(std::memory_order_relaxed);
        result.completed_requests =
            stats_.completed_requests.load(std::memory_order_relaxed);
        result.complete_requests =
            stats_.complete_requests.load(std::memory_order_relaxed);
        result.incomplete_requests =
            stats_.incomplete_requests.load(std::memory_order_relaxed);
        result.observed_records =
            stats_.observed_records.load(std::memory_order_relaxed);
        result.enqueued_records =
            stats_.enqueued_records.load(std::memory_order_relaxed);
        result.written_records =
            stats_.written_records.load(std::memory_order_relaxed);
        result.observed_ids =
            stats_.observed_ids.load(std::memory_order_relaxed);
        result.enqueued_ids =
            stats_.enqueued_ids.load(std::memory_order_relaxed);
        result.written_ids =
            stats_.written_ids.load(std::memory_order_relaxed);
        result.observed_id_chunks =
            stats_.observed_id_chunks.load(std::memory_order_relaxed);
        result.enqueued_id_chunks =
            stats_.enqueued_id_chunks.load(std::memory_order_relaxed);
        result.written_id_chunks =
            stats_.written_id_chunks.load(std::memory_order_relaxed);
        result.observed_batches =
            stats_.observed_batches.load(std::memory_order_relaxed);
        result.enqueued_batches =
            stats_.enqueued_batches.load(std::memory_order_relaxed);
        result.written_batches =
            stats_.written_batches.load(std::memory_order_relaxed);
        result.dropped_full_records =
            stats_.dropped_full_records.load(std::memory_order_relaxed);
        result.dropped_full_ids =
            stats_.dropped_full_ids.load(std::memory_order_relaxed);
        result.dropped_runtime_failure_records =
            stats_.dropped_runtime_failure_records.load(
                std::memory_order_relaxed);
        result.dropped_runtime_failure_ids =
            stats_.dropped_runtime_failure_ids.load(
                std::memory_order_relaxed);
        result.dropped_min_free_records =
            stats_.dropped_min_free_records.load(
                std::memory_order_relaxed);
        result.dropped_min_free_ids =
            stats_.dropped_min_free_ids.load(std::memory_order_relaxed);
        result.dropped_io_budget_records =
            stats_.dropped_io_budget_records.load(
                std::memory_order_relaxed);
        result.dropped_io_budget_ids =
            stats_.dropped_io_budget_ids.load(std::memory_order_relaxed);
        result.write_errors =
            stats_.write_errors.load(std::memory_order_relaxed);
        result.bytes_written =
            stats_.bytes_written.load(std::memory_order_relaxed);
        result.segments_created =
            stats_.segments_created.load(std::memory_order_relaxed);
        result.rotated_segments =
            stats_.rotated_segments.load(std::memory_order_relaxed);
        result.retained_bytes =
            stats_.retained_bytes.load(std::memory_order_relaxed);
        result.retained_segments =
            stats_.retained_segments.load(std::memory_order_relaxed);
        result.retention_deleted_bytes =
            stats_.retention_deleted_bytes.load(std::memory_order_relaxed);
        result.retention_deleted_segments =
            stats_.retention_deleted_segments.load(
                std::memory_order_relaxed);
        result.queued_records =
            stats_.queued_records.load(std::memory_order_relaxed);
        result.queued_bytes =
            stats_.queued_bytes.load(std::memory_order_relaxed);
        result.peak_queued_bytes =
            stats_.peak_queued_bytes.load(std::memory_order_relaxed);
        result.buffer_capacity_bytes =
            slot_count_ * maximum_frame_bytes_;
        result.active_requests =
            stats_.active_requests.load(std::memory_order_relaxed);
        result.last_error_number =
            stats_.last_error_number.load(std::memory_order_relaxed);
        result.paused_for_min_free =
            stats_.paused_for_min_free.load(std::memory_order_relaxed);
        result.runtime_failed =
            stats_.runtime_failed.load(std::memory_order_relaxed);
        result.shutting_down =
            stats_.shutting_down.load(std::memory_order_relaxed);
        const auto normalized = counter_snapshot(stats_);
        result.observed_requests = normalized.observed_requests;
        result.begun_requests = normalized.begun_requests;
        result.completed_requests = normalized.completed_requests;
        result.complete_requests = normalized.complete_requests;
        result.incomplete_requests = normalized.incomplete_requests;
        result.observed_records = normalized.observed_records;
        result.enqueued_records = normalized.enqueued_records;
        result.written_records = normalized.written_records;
        result.observed_ids = normalized.observed_ids;
        result.enqueued_ids = normalized.enqueued_ids;
        result.written_ids = normalized.written_ids;
        result.observed_id_chunks = normalized.observed_id_chunks;
        result.enqueued_id_chunks = normalized.enqueued_id_chunks;
        result.written_id_chunks = normalized.written_id_chunks;
        result.observed_batches = normalized.observed_batches;
        result.enqueued_batches = normalized.enqueued_batches;
        result.written_batches = normalized.written_batches;
        return result;
    }

    void shutdown() noexcept {
        bool expected = false;
        if (!stats_.shutting_down.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            if (worker_.joinable()
                && worker_.get_id() != std::this_thread::get_id()) {
                worker_.join();
            }
            return;
        }
        ready_records_.release();
        if (worker_.joinable()
            && worker_.get_id() != std::this_thread::get_id()) {
            worker_.join();
        }
    }

private:
    enum class EnqueueResult {
        Enqueued,
        Full,
        RuntimeFailure,
        MinimumFreeSpace,
        IoBudget,
    };

    struct QueueSlot {
        // A producer owns ticket T when sequence == T and publishes it by
        // storing T + 1. The sole consumer releases it for the next ring lap
        // by storing T + slot_count. This preserves reservation order even
        // when a later producer finishes its memcpy first.
        std::atomic<std::uint64_t> sequence{0};
        std::uint32_t size = 0;
        std::uint32_t kind = 0;
        std::uint64_t ids = 0;
    };

    class ProducerGuard {
    public:
        explicit ProducerGuard(
            std::atomic<std::uint64_t>& producers) noexcept
            : producers_(producers) {
            producers_.fetch_add(1, std::memory_order_acq_rel);
        }
        ~ProducerGuard() {
            producers_.fetch_sub(1, std::memory_order_release);
        }
        ProducerGuard(const ProducerGuard&) = delete;
        ProducerGuard& operator=(const ProducerGuard&) = delete;

    private:
        std::atomic<std::uint64_t>& producers_;
    };

    static void initialize_record(
        const std::span<std::byte> frame,
        const std::uint32_t kind) noexcept {
        std::memcpy(
            frame.data(), kRecordMagic.data(), kRecordMagic.size());
        put_u32(frame, 8, kRecordVersion);
        put_u32(
            frame,
            12,
            static_cast<std::uint32_t>(
                kReadWorkloadRecordHeaderBytes));
        put_u32(
            frame, 16, static_cast<std::uint32_t>(frame.size()));
        put_u32(frame, 20, kind);
    }

    static void update_digest(
        std::uint64_t& state,
        const std::span<const std::uint64_t> ids) noexcept {
        constexpr std::uint64_t prime = 0x100000001b3ULL;
        for (const auto id : ids) {
            for (unsigned int shift = 0; shift != 64; shift += 8) {
                state ^= (id >> shift) & 0xffU;
                state *= prime;
            }
        }
    }

    [[nodiscard]] EnqueueResult enqueue_record(
        const std::span<std::byte> frame,
        const std::uint32_t kind,
        const std::uint64_t ids) noexcept {
        atomic_add_saturating(stats_.observed_records);
        if (kind == kIdChunkRecordKind) {
            atomic_add_saturating(stats_.observed_id_chunks);
            atomic_add_saturating(stats_.observed_ids, ids);
        } else if (kind == kBatchMetricRecordKind) {
            atomic_add_saturating(stats_.observed_batches);
        }
        if (stats_.runtime_failed.load(std::memory_order_acquire)
            || stats_.shutting_down.load(std::memory_order_acquire)) {
            return EnqueueResult::RuntimeFailure;
        }
        if (stats_.paused_for_min_free.load(
                std::memory_order_acquire)) {
            return EnqueueResult::MinimumFreeSpace;
        }

        put_u64(
            frame, 40, 0);
        put_u64(frame, 48, unix_time_ns());
        if (get_u64(frame, 56) == 0) {
            put_u64(frame, 56, steady_time_ns());
        }
        put_u32(frame, kReadWorkloadRecordCrcOffset, 0);

        auto ticket = enqueue_ticket_.load(std::memory_order_relaxed);
        while (true) {
            const auto consumed =
                dequeue_ticket_.load(std::memory_order_acquire);
            // A relaxed producer load may lag the consumer's acquired
            // position. Do not let unsigned subtraction turn that harmless
            // stale observation into a false queue-full result; the CAS will
            // refresh `ticket` from the enqueue modification order.
            if (ticket >= consumed
                && ticket - consumed
                       >= static_cast<std::uint64_t>(slot_count_)) {
                return EnqueueResult::Full;
            }
            if (enqueue_ticket_.compare_exchange_weak(
                    ticket,
                    ticket + 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                break;
            }
        }
        const auto slot_index =
            static_cast<std::size_t>(ticket % slot_count_);
        auto& slot = queue_slots_[slot_index];
        while (slot.sequence.load(std::memory_order_acquire) != ticket) {
            // Capacity was checked before reservation. The previous lap has
            // already advanced dequeue_ticket_, so this loop is only a
            // memory-publication handoff and never waits for disk I/O.
            std::this_thread::yield();
        }

        std::memcpy(
            queue_storage_.data()
                + slot_index * maximum_frame_bytes_,
            frame.data(),
            frame.size());
        slot.size = static_cast<std::uint32_t>(frame.size());
        slot.kind = kind;
        slot.ids = ids;
        const auto queued_records =
            queued_records_.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto queued_bytes =
            queued_bytes_.fetch_add(
                frame.size(), std::memory_order_relaxed)
            + frame.size();
        stats_.queued_records.store(
            queued_records, std::memory_order_relaxed);
        stats_.queued_bytes.store(
            queued_bytes, std::memory_order_relaxed);
        atomic_max(stats_.peak_queued_bytes, queued_bytes);
        atomic_add_saturating(stats_.enqueued_records);
        if (kind == kIdChunkRecordKind) {
            atomic_add_saturating(stats_.enqueued_id_chunks);
            atomic_add_saturating(stats_.enqueued_ids, ids);
        } else if (kind == kBatchMetricRecordKind) {
            atomic_add_saturating(stats_.enqueued_batches);
        }
        slot.sequence.store(ticket + 1, std::memory_order_release);
        ready_records_.release();
        return EnqueueResult::Enqueued;
    }

    static void add_request_drop(
        std::uint64_t& destination,
        const std::uint64_t amount) noexcept {
        destination = saturating_add_value(destination, amount);
    }

    void note_global_drop(
        const EnqueueResult result,
        const std::uint64_t records,
        const std::uint64_t ids) noexcept {
        switch (result) {
            case EnqueueResult::Full:
                atomic_add_saturating(
                    stats_.dropped_full_records, records);
                atomic_add_saturating(stats_.dropped_full_ids, ids);
                return;
            case EnqueueResult::MinimumFreeSpace:
                atomic_add_saturating(
                    stats_.dropped_min_free_records, records);
                atomic_add_saturating(stats_.dropped_min_free_ids, ids);
                return;
            case EnqueueResult::IoBudget:
                atomic_add_saturating(
                    stats_.dropped_io_budget_records, records);
                atomic_add_saturating(stats_.dropped_io_budget_ids, ids);
                return;
            case EnqueueResult::RuntimeFailure:
                atomic_add_saturating(
                    stats_.dropped_runtime_failure_records, records);
                atomic_add_saturating(
                    stats_.dropped_runtime_failure_ids, ids);
                return;
            case EnqueueResult::Enqueued:
                return;
        }
    }

    void note_request_drop(
        ReadWorkloadRequestHandle& request,
        const EnqueueResult result,
        const std::uint64_t records,
        const std::uint64_t ids) noexcept {
        add_request_drop(request.dropped_records_, records);
        add_request_drop(request.dropped_ids_, ids);
        switch (result) {
            case EnqueueResult::Full:
                add_request_drop(request.dropped_full_records_, records);
                add_request_drop(request.dropped_full_ids_, ids);
                break;
            case EnqueueResult::RuntimeFailure:
                add_request_drop(
                    request.dropped_runtime_records_, records);
                add_request_drop(request.dropped_runtime_ids_, ids);
                break;
            case EnqueueResult::MinimumFreeSpace:
                add_request_drop(
                    request.dropped_min_free_records_, records);
                add_request_drop(request.dropped_min_free_ids_, ids);
                break;
            case EnqueueResult::IoBudget:
                add_request_drop(
                    request.dropped_io_budget_records_, records);
                add_request_drop(request.dropped_io_budget_ids_, ids);
                break;
            case EnqueueResult::Enqueued:
                break;
        }
        note_global_drop(result, records, ids);
    }

    void release_request(ReadWorkloadRequestHandle& request) noexcept {
        request.logger_ = nullptr;
        const auto active =
            stats_.active_requests.load(std::memory_order_relaxed);
        if (active != 0) {
            stats_.active_requests.fetch_sub(
                1, std::memory_order_release);
        }
    }

    [[nodiscard]] bool directory_is_empty() const {
        FileDescriptor enumeration_fd(::openat(
            directory_fd_.get(),
            ".",
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
        if (!enumeration_fd) {
            throw io_error(
                "cannot open read-workload log directory for enumeration",
                options_.directory);
        }
        DirectoryStream stream(::fdopendir(enumeration_fd.release()));
        if (stream.get() == nullptr) {
            throw io_error(
                "cannot enumerate read-workload log directory",
                options_.directory);
        }
        errno = 0;
        while (const dirent* entry = ::readdir(stream.get())) {
            const std::string_view name(entry->d_name);
            if (name != "." && name != "..") {
                return false;
            }
            errno = 0;
        }
        if (errno != 0) {
            throw io_error(
                "cannot enumerate read-workload log directory",
                options_.directory);
        }
        return true;
    }

    [[nodiscard]] bool validate_existing_marker() const {
        FileDescriptor marker_fd(::openat(
            directory_fd_.get(),
            kDirectoryMarkerName.data(),
            O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (!marker_fd) {
            if (errno == ENOENT) {
                return false;
            }
            throw io_error(
                "cannot open read-workload ownership marker",
                options_.directory / kDirectoryMarkerName);
        }
        struct stat status {};
        std::array<std::byte, kDirectoryMarkerBytes> marker{};
        if (::fstat(marker_fd.get(), &status) != 0
            || !S_ISREG(status.st_mode)
            || status.st_size
                   != static_cast<off_t>(marker.size())
            || status.st_uid != ::geteuid()
            || !pread_all(marker_fd.get(), marker, 0)
            || !valid_directory_marker(marker, collection_)) {
            throw InvalidArgumentError(
                "read-workload log directory has an invalid, foreign, or "
                "incompatible ownership marker: "
                + options_.directory.string());
        }
        return true;
    }

    void ensure_directory_marker() {
        if (validate_existing_marker()) {
            return;
        }
        if (!directory_is_empty()) {
            throw InvalidArgumentError(
                "existing read-workload log directory must be empty or carry "
                "a valid ownership marker for this collection: "
                + options_.directory.string());
        }
        const auto marker_path =
            options_.directory / kDirectoryMarkerName;
        FileDescriptor marker_fd(::openat(
            directory_fd_.get(),
            kDirectoryMarkerName.data(),
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            0600));
        if (!marker_fd) {
            if (errno == EEXIST && validate_existing_marker()) {
                return;
            }
            throw io_error(
                "cannot create read-workload ownership marker",
                marker_path);
        }
        bool complete = false;
        try {
            if (::fchmod(marker_fd.get(), 0600) != 0) {
                throw io_error(
                    "cannot protect read-workload ownership marker",
                    marker_path);
            }
            const auto marker = encode_directory_marker(collection_);
            if (!consume_write_budget(marker.size())) {
                throw StorageError(
                    "read-workload ownership marker exceeds the configured "
                    "write budget");
            }
            write_all(marker_fd.get(), marker, marker_path);
            sync_file(marker_fd.get(), marker_path);
            complete = true;
        } catch (...) {
            marker_fd.reset();
            if (::unlinkat(
                    directory_fd_.get(),
                    kDirectoryMarkerName.data(),
                    0)
                == 0) {
                sync_directory_best_effort(directory_fd_.get());
            }
            throw;
        }
        if (complete) {
            sync_directory(directory_fd_.get(), options_.directory);
        }
    }

    void prepare_directory_and_lock() {
        std::error_code error;
        const auto status =
            std::filesystem::symlink_status(options_.directory, error);
        if (error && error != std::errc::no_such_file_or_directory) {
            throw StorageError(
                "cannot inspect read-workload log directory '"
                + options_.directory.string() + "': " + error.message());
        }
        if (error == std::errc::no_such_file_or_directory
            || status.type() == std::filesystem::file_type::not_found) {
            error.clear();
            if (!std::filesystem::create_directories(
                    options_.directory, error)
                && error) {
                throw StorageError(
                    "cannot create read-workload log directory '"
                    + options_.directory.string() + "': "
                    + error.message());
            }
        } else if (status.type()
                   != std::filesystem::file_type::directory) {
            throw InvalidArgumentError(
                "read-workload log path is not a real directory: "
                + options_.directory.string());
        }
        directory_fd_.reset(::open(
            options_.directory.c_str(),
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
        if (!directory_fd_) {
            throw io_error(
                "cannot open read-workload log directory",
                options_.directory);
        }
        struct stat directory_status {};
        if (::fstat(directory_fd_.get(), &directory_status) != 0
            || !S_ISDIR(directory_status.st_mode)) {
            throw InvalidArgumentError(
                "read-workload log path is not a real directory: "
                + options_.directory.string());
        }
        ensure_directory_marker();
        if (::fchmod(directory_fd_.get(), 0700) != 0) {
            throw io_error(
                "cannot protect read-workload log directory",
                options_.directory);
        }

        const auto lock_path = options_.directory / kWriterLockName;
        writer_lock_.reset(::openat(
            directory_fd_.get(),
            kWriterLockName.data(),
            O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
            0600));
        if (!writer_lock_) {
            throw io_error("cannot open read-workload writer lock", lock_path);
        }
        if (::fchmod(writer_lock_.get(), 0600) != 0) {
            throw io_error(
                "cannot protect read-workload writer lock", lock_path);
        }
        if (::flock(writer_lock_.get(), LOCK_EX | LOCK_NB) != 0) {
            const auto error_number = errno;
            if (error_number == EWOULDBLOCK || error_number == EAGAIN) {
                throw StorageError(
                    "read-workload log directory already has an active "
                    "writer: "
                    + options_.directory.string());
            }
            throw io_error(
                "cannot lock read-workload log directory",
                lock_path,
                error_number);
        }
        sync_directory(directory_fd_.get(), options_.directory);
    }

    [[nodiscard]] bool recognized_header(
        const int fd,
        const std::uint64_t file_bytes) const noexcept {
        if (file_bytes < kReadWorkloadLogHeaderBytes) {
            return false;
        }
        std::array<std::byte, kReadWorkloadLogHeaderBytes> header{};
        if (!pread_all(fd, header, 0)
            || !has_magic(header, kLogMagic)
            || get_u32(header, 8) != kLogVersion
            || get_u32(header, 12) != kReadWorkloadLogHeaderBytes
            || get_u32(header, 16) != kLogHeaderUsedBytes
            || get_u32(header, kLogHeaderCrcOffset)
                   != crc32c_with_zeroed_field(
                       header, kLogHeaderCrcOffset)
            || get_u64(header, 32) != collection_.collection_id_low
            || get_u64(header, 40) != collection_.collection_id_high
            || get_u32(header, 48) != collection_.format
            || get_u32(header, 52) != collection_.dtype
            || get_u32(header, 56) != collection_.dimension
            || get_u32(header, 60) != collection_.record_size_bytes) {
            return false;
        }
        return true;
    }

    [[nodiscard]] RemovalResult remove_unchanged(
        const std::string& name,
        const struct stat& expected) noexcept {
        struct stat current {};
        if (::fstatat(
                directory_fd_.get(),
                name.c_str(),
                &current,
                AT_SYMLINK_NOFOLLOW)
            != 0) {
            return errno == ENOENT ? RemovalResult::Missing
                                   : RemovalResult::Failed;
        }
        if (!S_ISREG(current.st_mode)
            || current.st_dev != expected.st_dev
            || current.st_ino != expected.st_ino) {
            return RemovalResult::Failed;
        }
        if (::unlinkat(directory_fd_.get(), name.c_str(), 0) == 0) {
            return RemovalResult::Removed;
        }
        return errno == ENOENT ? RemovalResult::Missing
                               : RemovalResult::Failed;
    }

    [[nodiscard]] RemovalResult remove_unchanged(
        const RetainedFile& file) noexcept {
        struct stat expected {};
        expected.st_dev = file.device;
        expected.st_ino = file.inode;
        expected.st_mode = S_IFREG;
        return remove_unchanged(file.name, expected);
    }

    [[nodiscard]] std::vector<RetainedFile> retained_files() {
        std::vector<RetainedFile> files;
        files.reserve(static_cast<std::size_t>(
            std::min<std::uint64_t>(
                options_.max_retained_files,
                kMaximumScannedDirectoryEntries)));
        FileDescriptor enumeration_fd(::openat(
            directory_fd_.get(),
            ".",
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
        if (!enumeration_fd) {
            throw io_error(
                "cannot open read-workload log directory for enumeration",
                options_.directory);
        }
        DirectoryStream stream(::fdopendir(enumeration_fd.release()));
        if (stream.get() == nullptr) {
            throw io_error(
                "cannot enumerate read-workload log directory",
                options_.directory);
        }

        const auto prefix =
            std::string("read-workload-")
            + collection_token(collection_) + "-";
        std::uint64_t scanned_entries = 0;
        while (true) {
            errno = 0;
            const dirent* entry = ::readdir(stream.get());
            if (entry == nullptr) {
                if (errno != 0) {
                    throw io_error(
                        "cannot enumerate read-workload log directory",
                        options_.directory);
                }
                break;
            }
            const std::string name(entry->d_name);
            if (name == "." || name == "..") {
                continue;
            }
            ++scanned_entries;
            if (scanned_entries > kMaximumScannedDirectoryEntries) {
                throw StorageError(
                    "read-workload log directory exceeds the bounded "
                    "inventory limit of "
                    + std::to_string(kMaximumScannedDirectoryEntries)
                    + " entries: " + options_.directory.string());
            }
            if (name == active_name_ || !name.starts_with(prefix)) {
                continue;
            }
            const bool finalized = name.ends_with(kFinalizedExtension);
            const bool open = name.ends_with(kOpenExtension);
            if (!finalized && !open) {
                continue;
            }

            struct stat status {};
            if (::fstatat(
                    directory_fd_.get(),
                    name.c_str(),
                    &status,
                    AT_SYMLINK_NOFOLLOW)
                != 0) {
                if (errno == ENOENT) {
                    continue;
                }
                throw io_error(
                    "cannot stat owned read-workload segment",
                    options_.directory / name);
            }
            if (!S_ISREG(status.st_mode)) {
                continue;
            }
            if (status.st_size < 0) {
                throw StorageError(
                    "owned read-workload segment has an invalid size: "
                    + (options_.directory / name).string());
            }
            FileDescriptor fd(::openat(
                directory_fd_.get(),
                name.c_str(),
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
            if (!fd) {
                if (errno == ENOENT) {
                    continue;
                }
                throw io_error(
                    "cannot open owned read-workload segment",
                    options_.directory / name);
            }
            struct stat opened_status {};
            const bool stable_regular =
                ::fstat(fd.get(), &opened_status) == 0
                && S_ISREG(opened_status.st_mode)
                && opened_status.st_dev == status.st_dev
                && opened_status.st_ino == status.st_ino
                && opened_status.st_size >= 0
                && opened_status.st_uid == ::geteuid();
            const bool recognized =
                stable_regular
                && recognized_header(
                    fd.get(),
                    static_cast<std::uint64_t>(opened_status.st_size));
            if (!stable_regular) {
                throw StorageError(
                    "cannot safely inventory owned read-workload segment: "
                    + (options_.directory / name).string());
            }
            files.push_back(RetainedFile{
                name,
                static_cast<std::uint64_t>(opened_status.st_size),
                stat_modified_ns(opened_status),
                opened_status.st_dev,
                opened_status.st_ino,
                finalized,
                recognized});
        }
        return files;
    }

    void refresh_retained_stats() noexcept {
        stats_.retained_bytes.store(
            saturating_add_value(
                retained_historical_bytes_, active_bytes_),
            std::memory_order_relaxed);
        stats_.retained_segments.store(
            saturating_add_value(
                retained_historical_segments_,
                active_fd_ ? 1 : 0),
            std::memory_order_relaxed);
    }

    void prune_retention() {
        auto files = retained_files();
        std::sort(
            files.begin(),
            files.end(),
            [](const RetainedFile& left, const RetainedFile& right) {
                const bool left_invalid_open =
                    !left.finalized && !left.recognized;
                const bool right_invalid_open =
                    !right.finalized && !right.recognized;
                if (left_invalid_open != right_invalid_open) {
                    return left_invalid_open;
                }
                if (left.modified_ns != right.modified_ns) {
                    return left.modified_ns < right.modified_ns;
                }
                return left.name < right.name;
            });
        std::uint64_t total_bytes = 0;
        for (const auto& file : files) {
            total_bytes = saturating_add_value(total_bytes, file.bytes);
        }
        std::uint64_t total_segments = files.size();
        const auto now = static_cast<std::int64_t>(
            std::min<std::uint64_t>(
                unix_time_ns(),
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())));
        const auto retention_ns =
            options_.retention_seconds
                    > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()
                        / 1'000'000'000)
                ? std::numeric_limits<std::int64_t>::max()
                : static_cast<std::int64_t>(
                      options_.retention_seconds * 1'000'000'000);
        // Reserve one complete successor while running, and retain that
        // reserve for the current active file during shutdown. Only after
        // finalization leaves no active writer may the full configured
        // budget remain as finalized history.
        const bool reserve_active =
            active_fd_
            || !stats_.shutting_down.load(std::memory_order_relaxed);
        const auto historical_byte_limit =
            options_.retention_bytes
            - (reserve_active ? options_.segment_bytes : 0);
        const auto retained_file_limit =
            options_.max_retained_files
            - (reserve_active ? 1 : 0);

        for (const auto& file : files) {
            const bool aged =
                file.modified_ns <= now
                && now - file.modified_ns >= retention_ns;
            const bool over_budget =
                total_bytes > historical_byte_limit;
            const bool over_file_limit =
                total_segments > retained_file_limit;
            // The exclusive directory lock proves every non-active .open is
            // stale. Invalid stale tails cannot yield a recoverable prefix
            // and are removed immediately. Corrupt finalized files remain
            // available for diagnosis only while normal privacy-retention
            // bounds allow them.
            const bool invalid_open =
                !file.finalized && !file.recognized;
            if (!invalid_open && !aged && !over_budget
                && !over_file_limit) {
                continue;
            }
            const RemovalResult removed = remove_unchanged(file);
            if (removed == RemovalResult::Failed) {
                throw StorageError(
                    "cannot enforce read-workload retention for: "
                    + (options_.directory / file.name).string());
            }
            total_bytes -= file.bytes;
            --total_segments;
            if (removed == RemovalResult::Removed) {
                atomic_add_saturating(
                    stats_.retention_deleted_bytes, file.bytes);
                atomic_add_saturating(
                    stats_.retention_deleted_segments);
                retention_removed_any_ = true;
            }
        }
        if (total_bytes > historical_byte_limit
            || total_segments > retained_file_limit) {
            throw StorageError(
                "read-workload retention bounds remain violated after "
                "pruning: "
                + options_.directory.string());
        }
        if (retention_removed_any_) {
            sync_directory(directory_fd_.get(), options_.directory);
            retention_removed_any_ = false;
        }
        retained_historical_bytes_ = total_bytes;
        retained_historical_segments_ = total_segments;
        refresh_retained_stats();
        next_retention_sweep_ =
            std::chrono::steady_clock::now()
            + retention_sweep_interval_;
    }

    void create_segment() {
        const auto created = unix_time_ns();
        for (std::uint64_t collision = 0; collision != 1'024; ++collision) {
            const auto suffix =
                std::to_string(created) + "-"
                + std::to_string(segment_sequence_) + "-"
                + std::to_string(collision);
            active_name_ =
                "read-workload-" + collection_token(collection_) + "-"
                + instance_token(instance_low_, instance_high_) + "-"
                + suffix + std::string(kOpenExtension);
            active_path_ = options_.directory / active_name_;
            active_fd_.reset(::openat(
                directory_fd_.get(),
                active_name_.c_str(),
                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                0600));
            if (active_fd_) {
                break;
            }
            if (errno != EEXIST) {
                throw io_error(
                    "cannot create read-workload log segment",
                    active_path_);
            }
        }
        if (!active_fd_) {
            throw StorageError(
                "cannot allocate a unique read-workload log segment name");
        }
        try {
            if (::fchmod(active_fd_.get(), 0600) != 0) {
                throw io_error(
                    "cannot protect read-workload log segment", active_path_);
            }
            const auto header = encode_log_header(
                options_,
                collection_,
                instance_low_,
                instance_high_,
                created,
                segment_sequence_);
            const auto opening_checkpoint =
                checkpoint_frame(last_checkpoint_, true);
            write_all(active_fd_.get(), header, active_path_);
            write_all(
                active_fd_.get(), opening_checkpoint, active_path_);
            sync_file(active_fd_.get(), active_path_);
            sync_directory(directory_fd_.get(), options_.directory);
        } catch (...) {
            active_fd_.reset();
            if (::unlinkat(
                    directory_fd_.get(), active_name_.c_str(), 0)
                == 0) {
                sync_directory_best_effort(directory_fd_.get());
            }
            active_name_.clear();
            active_path_.clear();
            throw;
        }
        active_bytes_ =
            kReadWorkloadLogHeaderBytes
            + kReadWorkloadRecordHeaderBytes;
        refresh_retained_stats();
        active_started_ = std::chrono::steady_clock::now();
        active_has_records_ = false;
        active_terminal_checkpoint_ = true;
        ++segment_sequence_;
        atomic_add_saturating(stats_.segments_created);
        atomic_add_saturating(
            stats_.bytes_written,
            kReadWorkloadLogHeaderBytes
                + kReadWorkloadRecordHeaderBytes);
    }

    void finalize_segment() {
        if (!active_fd_) {
            return;
        }
        if (!active_has_records_ || !active_terminal_checkpoint_) {
            throw StorageError(
                "refusing to finalize a read-workload segment without a "
                "terminal checkpoint");
        }
        sync_file(active_fd_.get(), active_path_);
        evict_file_cache_best_effort(active_fd_.get());
        active_fd_.reset();
        std::filesystem::path finalized_name = active_name_;
        finalized_name.replace_extension(kFinalizedExtension);
        if (::renameat(
                directory_fd_.get(),
                active_name_.c_str(),
                directory_fd_.get(),
                finalized_name.c_str())
            != 0) {
            throw StorageError(
                "cannot finalize read-workload log segment '"
                + active_path_.string() + "': "
                + std::strerror(errno));
        }
        sync_directory(directory_fd_.get(), options_.directory);
        active_path_.clear();
        active_name_.clear();
        active_bytes_ = 0;
        refresh_retained_stats();
        active_has_records_ = false;
        active_terminal_checkpoint_ = false;
        atomic_add_saturating(stats_.rotated_segments);
    }

    void discard_active_segment() {
        if (!active_fd_) {
            return;
        }
        active_fd_.reset();
        if (!active_name_.empty()) {
            const int result = ::unlinkat(
                directory_fd_.get(), active_name_.c_str(), 0);
            if (result != 0 && errno != ENOENT) {
                throw io_error(
                    "cannot discard unused read-workload log segment",
                    active_path_);
            }
            if (result == 0) {
                sync_directory(directory_fd_.get(), options_.directory);
            }
            active_name_.clear();
            active_path_.clear();
        }
        active_bytes_ = 0;
        refresh_retained_stats();
        active_has_records_ = false;
        active_terminal_checkpoint_ = false;
    }

    [[nodiscard]] bool update_min_free_state(
        const std::uint64_t prospective_bytes) noexcept {
        if (options_.min_free_bytes == 0) {
            stats_.paused_for_min_free.store(
                false, std::memory_order_release);
            return true;
        }
        struct statvfs space {};
        if (::fstatvfs(directory_fd_.get(), &space) != 0) {
            mark_runtime_failure(errno);
            return false;
        }
        const auto block_size = static_cast<std::uint64_t>(
            space.f_frsize != 0 ? space.f_frsize : space.f_bsize);
        const auto blocks = static_cast<std::uint64_t>(space.f_bavail);
        const auto available =
            block_size != 0
                    && blocks
                           > std::numeric_limits<std::uint64_t>::max()
                                 / block_size
                ? std::numeric_limits<std::uint64_t>::max()
                : block_size * blocks;
        const bool enough =
            available >= options_.min_free_bytes
            && prospective_bytes
                   <= available - options_.min_free_bytes;
        stats_.paused_for_min_free.store(
            !enough, std::memory_order_release);
        return enough;
    }

    void append_bytes(const std::span<const std::byte> bytes) {
        write_all(active_fd_.get(), bytes, active_path_);
        active_bytes_ += bytes.size();
        refresh_retained_stats();
        atomic_add_saturating(stats_.bytes_written, bytes.size());
    }

    [[nodiscard]] std::array<
        std::byte,
        kReadWorkloadRecordHeaderBytes>
    checkpoint_frame(
        const CounterSnapshot& counters,
        const bool opening_baseline = false) const noexcept {
        std::array<std::byte, kReadWorkloadRecordHeaderBytes> frame{};
        std::memcpy(
            frame.data(), kRecordMagic.data(), kRecordMagic.size());
        put_u32(frame, 8, kRecordVersion);
        put_u32(
            frame,
            12,
            static_cast<std::uint32_t>(
                kReadWorkloadRecordHeaderBytes));
        put_u32(
            frame,
            16,
            static_cast<std::uint32_t>(frame.size()));
        put_u32(frame, 20, kCheckpointRecordKind);
        put_u32(frame, 28, kCheckpointExtensionVersion);
        put_u32(
            frame,
            32,
            opening_baseline ? kCheckpointFlagOpeningBaseline : 0);
        put_u64(frame, 48, unix_time_ns());
        encode_counter_snapshot(frame, counters);
        put_u32(
            frame,
            kReadWorkloadRecordCrcOffset,
            crc32c_with_zeroed_field(
                frame, kReadWorkloadRecordCrcOffset));
        return frame;
    }

    [[nodiscard]] bool record_fits(
        const std::uint64_t bytes) const noexcept {
        return active_fd_
               && active_bytes_ <= options_.segment_bytes
               && bytes <= options_.segment_bytes - active_bytes_;
    }

    [[nodiscard]] bool active_time_due() const noexcept {
        return active_fd_ && active_has_records_
               && std::chrono::steady_clock::now() - active_started_
                      >= std::chrono::seconds(options_.rotate_seconds);
    }

    [[nodiscard]] bool retention_sweep_due() const noexcept {
        return std::chrono::steady_clock::now()
               >= next_retention_sweep_;
    }

    void append_checkpoint_record(const CounterSnapshot& counters) {
        const auto frame = checkpoint_frame(counters, false);
        append_bytes(frame);
        last_checkpoint_ = counters;
        events_since_checkpoint_ = 0;
        active_has_records_ = true;
        active_terminal_checkpoint_ = true;
    }

    [[nodiscard]] bool write_checkpoint_if_changed(const bool force) {
        const CounterSnapshot current = counter_snapshot(stats_);
        if (!force && counters_equal(current, last_checkpoint_)
            && !active_time_due()) {
            return true;
        }
        constexpr std::uint64_t checkpoint_bytes =
            kReadWorkloadRecordHeaderBytes;
        constexpr std::uint64_t segment_bootstrap_bytes =
            kReadWorkloadLogHeaderBytes + checkpoint_bytes;
        const bool rotate =
            active_fd_ && active_has_records_
            && (active_time_due() || !record_fits(checkpoint_bytes));
        const bool append_to_old =
            rotate
            && (!active_terminal_checkpoint_
                || record_fits(checkpoint_bytes));

        std::uint64_t prospective = checkpoint_bytes;
        if (!active_fd_) {
            prospective = saturating_add_value(
                segment_bootstrap_bytes, checkpoint_bytes);
        } else if (rotate) {
            prospective = saturating_add_value(
                append_to_old ? checkpoint_bytes : 0,
                append_to_old && force
                    ? 0
                    : segment_bootstrap_bytes);
            if (!append_to_old) {
                prospective = saturating_add_value(
                    prospective, checkpoint_bytes);
            }
        }
        if (!update_min_free_state(prospective)) {
            return false;
        }
        const bool has_write_budget =
            force ? wait_for_write_budget(prospective)
                  : consume_write_budget(prospective);
        if (!has_write_budget) {
            return false;
        }

        if (!active_fd_) {
            create_segment();
        } else if (rotate) {
            if (append_to_old) {
                if (!record_fits(checkpoint_bytes)) {
                    throw StorageError(
                        "read-workload terminal checkpoint no longer fits "
                        "its reserved segment space");
                }
                append_checkpoint_record(current);
                finalize_segment();
                prune_retention();
                if (!force) {
                    create_segment();
                }
                return true;
            }
            finalize_segment();
            prune_retention();
            create_segment();
        }
        append_checkpoint_record(current);
        return true;
    }

    [[nodiscard]] PrepareEventResult prepare_event_append(
        const std::uint64_t frame_bytes) {
        constexpr std::uint64_t checkpoint_bytes =
            kReadWorkloadRecordHeaderBytes;
        constexpr std::uint64_t segment_bootstrap_bytes =
            kReadWorkloadLogHeaderBytes + checkpoint_bytes;
        const auto event_and_checkpoint = saturating_add_value(
            frame_bytes, checkpoint_bytes);
        const bool rotate =
            active_fd_ && active_has_records_
            && (active_time_due() || !record_fits(event_and_checkpoint));
        // Reserve both the event append and the terminal checkpoint that
        // makes the resulting segment eligible for finalization. This keeps
        // the min-free invariant true across the complete write sequence,
        // rather than only for the immediately visible event payload.
        std::uint64_t prospective = event_and_checkpoint;
        if (!active_fd_) {
            prospective = saturating_add_value(
                segment_bootstrap_bytes, event_and_checkpoint);
        } else if (rotate) {
            prospective = saturating_add_value(
                active_terminal_checkpoint_ ? 0 : checkpoint_bytes,
                saturating_add_value(
                    segment_bootstrap_bytes, event_and_checkpoint));
        }
        if (!update_min_free_state(prospective)) {
            return PrepareEventResult::MinimumFreeSpace;
        }
        std::uint64_t immediate_write_bytes = frame_bytes;
        if (!active_fd_) {
            immediate_write_bytes = saturating_add_value(
                segment_bootstrap_bytes, frame_bytes);
        } else if (rotate) {
            immediate_write_bytes = saturating_add_value(
                segment_bootstrap_bytes, frame_bytes);
            if (!active_terminal_checkpoint_) {
                immediate_write_bytes = saturating_add_value(
                    immediate_write_bytes, checkpoint_bytes);
            }
        }
        // Debit the complete immediate write transaction before rotating or
        // creating any file. A budget-dropped event therefore cannot cause
        // uncharged segment churn.
        if (!wait_for_write_budget(immediate_write_bytes)) {
            return PrepareEventResult::WriteBudget;
        }
        if (!active_fd_) {
            create_segment();
        } else if (rotate) {
            if (!active_terminal_checkpoint_) {
                if (!record_fits(checkpoint_bytes)) {
                    throw StorageError(
                        "read-workload terminal checkpoint no longer fits "
                        "its reserved segment space");
                }
                append_checkpoint_record(counter_snapshot(stats_));
            }
            finalize_segment();
            prune_retention();
            create_segment();
        }
        if (!record_fits(event_and_checkpoint)) {
            throw StorageError(
                "read-workload event would consume terminal checkpoint "
                "reserve");
        }
        return PrepareEventResult::Ready;
    }

    [[nodiscard]] std::size_t take_ready_slot() noexcept {
        const auto ticket =
            dequeue_ticket_.load(std::memory_order_relaxed);
        const auto index =
            static_cast<std::size_t>(ticket % slot_count_);
        return queue_slots_[index].sequence.load(
                   std::memory_order_acquire)
                       == ticket + 1
                   ? index
                   : slot_count_;
    }

    void release_slot(const std::size_t index) noexcept {
        auto& slot = queue_slots_[index];
        const auto ticket =
            dequeue_ticket_.load(std::memory_order_relaxed);
        const auto bytes = slot.size;
        slot.size = 0;
        slot.kind = 0;
        slot.ids = 0;
        const auto records =
            queued_records_.fetch_sub(1, std::memory_order_relaxed) - 1;
        const auto queued_bytes =
            queued_bytes_.fetch_sub(bytes, std::memory_order_relaxed)
            - bytes;
        stats_.queued_records.store(records, std::memory_order_relaxed);
        stats_.queued_bytes.store(
            queued_bytes, std::memory_order_relaxed);
        slot.sequence.store(
            ticket + static_cast<std::uint64_t>(slot_count_),
            std::memory_order_release);
        dequeue_ticket_.store(ticket + 1, std::memory_order_release);
    }

    void drop_slot(
        const std::size_t index,
        const EnqueueResult reason) noexcept {
        const auto& slot = queue_slots_[index];
        note_global_drop(reason, 1, slot.ids);
        release_slot(index);
    }

    void wait_for_producers() noexcept {
        while (producers_inflight_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }

    void drop_queue(const EnqueueResult reason) noexcept {
        wait_for_producers();
        while (dequeue_ticket_.load(std::memory_order_acquire)
               != enqueue_ticket_.load(std::memory_order_acquire)) {
            const auto index = take_ready_slot();
            if (index == slot_count_) {
                std::this_thread::yield();
                continue;
            }
            drop_slot(index, reason);
        }
    }

    void mark_runtime_failure(const int error_number) noexcept {
        stats_.last_error_number.store(
            error_number != 0 ? error_number : EIO,
            std::memory_order_relaxed);
        atomic_add_saturating(stats_.write_errors);
        stats_.runtime_failed.store(true, std::memory_order_release);
        stats_.paused_for_min_free.store(false, std::memory_order_release);
        ready_records_.release();
    }

    [[nodiscard]] bool consume_write_budget(
        const std::uint64_t bytes) noexcept {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(now - write_budget_updated_);
        if (elapsed.count() > 0) {
            refill_read_workload_write_budget(
                options_.max_write_bytes_per_second,
                static_cast<std::uint64_t>(elapsed.count()),
                write_budget_tokens_, write_budget_remainder_);
            write_budget_updated_ = now;
        }
        if (bytes > write_budget_tokens_) {
            return false;
        }
        write_budget_tokens_ -= bytes;
        return true;
    }

    [[nodiscard]] bool wait_for_write_budget(
        const std::uint64_t bytes) noexcept {
        const auto deadline =
            std::chrono::steady_clock::now() + 1'100ms;
        while (!consume_write_budget(bytes)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(1ms);
        }
        return true;
    }

    void run() noexcept {
        try {
            while (true) {
                static_cast<void>(
                    ready_records_.try_acquire_for(100ms));
                if (stats_.runtime_failed.load(
                        std::memory_order_acquire)) {
                    break;
                }
                const bool stopping =
                    stats_.shutting_down.load(std::memory_order_acquire);
                auto slot_index = take_ready_slot();
                while (slot_index == slot_count_
                       && dequeue_ticket_.load(std::memory_order_acquire)
                              != enqueue_ticket_.load(
                                  std::memory_order_acquire)) {
                    if (producers_inflight_.load(
                            std::memory_order_acquire)
                        == 0) {
                        // The last producer may have published the head and
                        // decremented producers_inflight_ between the first
                        // slot probe and this load. Its release plus this
                        // acquire makes a second probe authoritative.
                        slot_index = take_ready_slot();
                        if (slot_index != slot_count_) {
                            break;
                        }
                        if (dequeue_ticket_.load(
                                std::memory_order_acquire)
                            == enqueue_ticket_.load(
                                std::memory_order_acquire)) {
                            break;
                        }
                        throw StorageError(
                            "read-workload FIFO head was reserved but never "
                            "published");
                    }
                    std::this_thread::yield();
                    slot_index = take_ready_slot();
                }
                if (slot_index == slot_count_) {
                    if (stopping
                        && dequeue_ticket_.load(
                               std::memory_order_acquire)
                               == enqueue_ticket_.load(
                                   std::memory_order_acquire)
                        && producers_inflight_.load(
                               std::memory_order_acquire)
                               == 0) {
                        break;
                    }
                    if (retention_sweep_due()) {
                        prune_retention();
                    }
                    if (stats_.paused_for_min_free.load(
                            std::memory_order_acquire)) {
                        (void)update_min_free_state(
                            kReadWorkloadLogHeaderBytes
                            + kReadWorkloadRecordHeaderBytes);
                    } else {
                        static_cast<void>(
                            write_checkpoint_if_changed(false));
                    }
                    continue;
                }
                const auto frame_size =
                    queue_slots_[slot_index].size;
                auto frame = std::span<std::byte>(
                    queue_storage_.data()
                        + slot_index * maximum_frame_bytes_,
                    frame_size);
                if (retention_sweep_due()) {
                    prune_retention();
                }

                const PrepareEventResult prepared =
                    prepare_event_append(frame_size);
                if (prepared == PrepareEventResult::WriteBudget) {
                    drop_slot(slot_index, EnqueueResult::IoBudget);
                    continue;
                }
                if (prepared == PrepareEventResult::MinimumFreeSpace) {
                    if (stats_.runtime_failed.load(
                            std::memory_order_acquire)) {
                        break;
                    }
                    if (stopping) {
                        drop_slot(
                            slot_index,
                            EnqueueResult::MinimumFreeSpace);
                        drop_queue(EnqueueResult::MinimumFreeSpace);
                        break;
                    }
                    std::this_thread::sleep_for(100ms);
                    ready_records_.release();
                    continue;
                }
                if (record_sequence_
                    == std::numeric_limits<std::uint64_t>::max()) {
                    throw StorageError(
                        "read-workload physical record sequence overflow");
                }
                put_u64(frame, 40, ++record_sequence_);
                put_u32(frame, kReadWorkloadRecordCrcOffset, 0);
                put_u32(
                    frame,
                    kReadWorkloadRecordCrcOffset,
                    crc32c_with_zeroed_field(
                        frame, kReadWorkloadRecordCrcOffset));
                append_bytes(frame);
                active_has_records_ = true;
                active_terminal_checkpoint_ = false;
                atomic_add_saturating(stats_.written_records);
                if (queue_slots_[slot_index].kind
                    == kIdChunkRecordKind) {
                    atomic_add_saturating(stats_.written_id_chunks);
                    atomic_add_saturating(
                        stats_.written_ids,
                        queue_slots_[slot_index].ids);
                } else if (
                    queue_slots_[slot_index].kind
                    == kBatchMetricRecordKind) {
                    atomic_add_saturating(stats_.written_batches);
                }
                ++events_since_checkpoint_;
                release_slot(slot_index);
                if (events_since_checkpoint_
                    >= kCheckpointEventInterval) {
                    static_cast<void>(
                        write_checkpoint_if_changed(false));
                }
            }

            if (!stats_.runtime_failed.load(std::memory_order_acquire)) {
                const CounterSnapshot current = counter_snapshot(stats_);
                if (!active_has_records_
                    && counters_equal(current, last_checkpoint_)) {
                    discard_active_segment();
                    prune_retention();
                } else if (
                    active_terminal_checkpoint_
                    && counters_equal(current, last_checkpoint_)) {
                    // The most recent periodic checkpoint is already a
                    // complete terminal record. Avoid publishing a
                    // zero-delta successor merely because the full segment
                    // has no room for a duplicate checkpoint.
                    finalize_segment();
                    prune_retention();
                } else if (write_checkpoint_if_changed(true)) {
                    finalize_segment();
                    prune_retention();
                } else {
                    // Disk pressure may prevent a terminal checkpoint, but
                    // deleting expired history can still restore space and
                    // must not be skipped during shutdown.
                    prune_retention();
                }
            }
        } catch (...) {
            const auto error_number = errno;
            mark_runtime_failure(
                error_number != 0 ? error_number : EIO);
        }
        if (stats_.runtime_failed.load(std::memory_order_acquire)) {
            drop_queue(EnqueueResult::RuntimeFailure);
            active_fd_.reset();
        }
    }

    ReadWorkloadLogOptions options_;
    ReadWorkloadCollectionInfo collection_;
    const std::size_t maximum_frame_bytes_;
    const std::size_t slot_count_;
    std::vector<std::byte> queue_storage_;
    std::unique_ptr<QueueSlot[]> queue_slots_;
    std::counting_semaphore<> ready_records_{0};
    std::atomic<std::uint64_t> enqueue_ticket_{0};
    std::atomic<std::uint64_t> dequeue_ticket_{0};
    std::atomic<std::uint64_t> queued_records_{0};
    std::atomic<std::uint64_t> queued_bytes_{0};
    std::atomic<std::uint64_t> producers_inflight_{0};
    std::thread worker_;
    AtomicStats stats_;
    FileDescriptor writer_lock_;
    FileDescriptor directory_fd_;
    FileDescriptor active_fd_;
    std::string active_name_;
    std::filesystem::path active_path_;
    std::uint64_t active_bytes_ = 0;
    std::uint64_t retained_historical_bytes_ = 0;
    std::uint64_t retained_historical_segments_ = 0;
    std::chrono::steady_clock::time_point active_started_{};
    bool active_has_records_ = false;
    bool active_terminal_checkpoint_ = false;
    std::uint64_t segment_sequence_ = 0;
    std::uint64_t instance_low_ = 0;
    std::uint64_t instance_high_ = 0;
    std::uint64_t write_budget_tokens_ =
        options_.max_write_bytes_per_second;
    std::uint64_t write_budget_remainder_ = 0;
    std::chrono::steady_clock::time_point write_budget_updated_ =
        std::chrono::steady_clock::now();
    std::atomic<std::uint64_t> request_sequence_{0};
    std::uint64_t record_sequence_ = 0;
    std::uint64_t events_since_checkpoint_ = 0;
    CounterSnapshot last_checkpoint_;
    bool retention_removed_any_ = false;
    const std::chrono::seconds retention_sweep_interval_;
    std::chrono::steady_clock::time_point next_retention_sweep_{};
};

std::unique_ptr<ReadWorkloadLogger> ReadWorkloadLogger::open(
    ReadWorkloadLogOptions options,
    const ReadWorkloadCollectionInfo collection) {
    auto impl =
        std::make_unique<ReadWorkloadLogImpl>(
            std::move(options), collection);
    impl->start();
    return std::unique_ptr<ReadWorkloadLogger>(
        new ReadWorkloadLogger(std::move(impl)));
}

ReadWorkloadLogger::ReadWorkloadLogger(
    std::unique_ptr<ReadWorkloadLogImpl> impl)
    : impl_(std::move(impl)) {}

ReadWorkloadLogger::~ReadWorkloadLogger() = default;

ReadWorkloadRequestHandle ReadWorkloadLogger::begin_request(
    const ReadWorkloadRequestInfo& info) noexcept {
    ReadWorkloadRequestHandle result;
    if (impl_->begin_request(result, info)) {
        result.logger_ = this;
    }
    return result;
}

bool ReadWorkloadLogger::try_append_ids(
    ReadWorkloadRequestHandle& request,
    const std::span<const std::uint64_t> ids,
    const ReadWorkloadIdChunkInfo& info) noexcept {
    return request.logger_ == this
           && impl_->append_ids(request, ids, info);
}

bool ReadWorkloadLogger::try_append_batch(
    ReadWorkloadRequestHandle& request,
    const ReadWorkloadBatchMetrics& metrics) noexcept {
    return request.logger_ == this
           && impl_->append_batch(request, metrics);
}

void ReadWorkloadLogger::end_request(
    ReadWorkloadRequestHandle& request,
    const ReadWorkloadRequestEnd& end) noexcept {
    if (request.logger_ == this) {
        impl_->end_request(request, end);
    }
}

ReadWorkloadLogStats ReadWorkloadLogger::stats() const noexcept {
    return impl_->stats();
}

void ReadWorkloadLogger::shutdown() noexcept {
    impl_->shutdown();
}

void ReadWorkloadLogger::abandon_request(
    ReadWorkloadRequestHandle& request) noexcept {
    end_request(
        request,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Shutdown,
        });
}

ReadWorkloadRequestHandle::~ReadWorkloadRequestHandle() {
    if (logger_ != nullptr) {
        logger_->abandon_request(*this);
    }
}

ReadWorkloadRequestHandle::ReadWorkloadRequestHandle(
    ReadWorkloadRequestHandle&& other) noexcept {
    *this = std::move(other);
}

ReadWorkloadRequestHandle& ReadWorkloadRequestHandle::operator=(
    ReadWorkloadRequestHandle&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (logger_ != nullptr) {
        logger_->abandon_request(*this);
    }
    logger_ = std::exchange(other.logger_, nullptr);
    request_id_low_ = std::exchange(other.request_id_low_, 0);
    request_id_high_ = std::exchange(other.request_id_high_, 0);
    started_steady_ns_ = std::exchange(other.started_steady_ns_, 0);
    declared_ids_ = std::exchange(other.declared_ids_, 0);
    submitted_ids_ = std::exchange(other.submitted_ids_, 0);
    enqueued_ids_ = std::exchange(other.enqueued_ids_, 0);
    id_chunks_ = std::exchange(other.id_chunks_, 0);
    batches_ = std::exchange(other.batches_, 0);
    response_bytes_ = std::exchange(other.response_bytes_, 0);
    dropped_records_ = std::exchange(other.dropped_records_, 0);
    dropped_ids_ = std::exchange(other.dropped_ids_, 0);
    dropped_full_records_ =
        std::exchange(other.dropped_full_records_, 0);
    dropped_full_ids_ = std::exchange(other.dropped_full_ids_, 0);
    dropped_runtime_records_ =
        std::exchange(other.dropped_runtime_records_, 0);
    dropped_runtime_ids_ =
        std::exchange(other.dropped_runtime_ids_, 0);
    dropped_min_free_records_ =
        std::exchange(other.dropped_min_free_records_, 0);
    dropped_min_free_ids_ =
        std::exchange(other.dropped_min_free_ids_, 0);
    dropped_io_budget_records_ =
        std::exchange(other.dropped_io_budget_records_, 0);
    dropped_io_budget_ids_ =
        std::exchange(other.dropped_io_budget_ids_, 0);
    digest_low_ = std::exchange(
        other.digest_low_, 0xcbf29ce484222325ULL);
    digest_high_ = std::exchange(
        other.digest_high_, 0x84222325cbf29ce4ULL);
    next_chunk_sequence_ =
        std::exchange(other.next_chunk_sequence_, 0);
    logical_offsets_contiguous_ =
        std::exchange(other.logical_offsets_contiguous_, true);
    return *this;
}

}  // namespace throwdb::server::detail
