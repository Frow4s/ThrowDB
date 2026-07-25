#include "read_workload_log.hpp"

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
    'T', 'D', 'B', 'R', 'L', 'O', 'G', '1'};
constexpr std::array<char, 8> kRecordMagic{
    'T', 'D', 'B', 'R', 'R', 'E', 'C', '1'};
constexpr std::uint32_t kLogVersion = 1;
constexpr std::uint32_t kLogHeaderUsedBytes = 176;
constexpr std::uint32_t kRecordVersion = 1;
constexpr std::uint32_t kEventRecordKind = 1;
constexpr std::uint32_t kCheckpointRecordKind = 2;
constexpr std::uint32_t kCheckpointExtensionVersion = 2;
constexpr std::uint32_t kCheckpointFlagOpeningBaseline = 1U << 0U;
constexpr std::uint32_t kRecordFlagExactSnapshot = 1U << 0U;
constexpr std::uint32_t kRecordFlagTruncatedIds = 1U << 1U;
constexpr std::size_t kLogHeaderCrcOffset = 24;
constexpr std::uint32_t kMaximumIdsPerEvent = 65'536;
constexpr std::uint64_t kSampleDenominator = 1'000'000;
constexpr std::uint64_t kEventSamplingDomain =
    0x45564e5453414d50ULL;
constexpr std::uint64_t kIdSelectionDomain =
    0x494453454c454354ULL;
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

[[nodiscard]] std::uint64_t uniform_below(
    std::uint64_t value,
    const std::uint64_t bound) noexcept {
    // At most two rejection attempts keep request-path CPU strictly bounded.
    // The multiply-high fallback has per-bucket bias below 1 / 2^64 and is
    // reached only through the already tiny double-rejection tail.
    const auto threshold = (std::uint64_t{0} - bound) % bound;
    for (unsigned int attempt = 0; attempt != 2; ++attempt) {
        if (value >= threshold) {
            return value % bound;
        }
        value = splitmix64(value);
    }
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(value) * bound) >> 64U);
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
    if (options.sample_rate_ppm == 0
        || options.sample_rate_ppm > kSampleDenominator) {
        throw InvalidArgumentError(
            "read-workload sample rate must be in [1, 1000000] ppm");
    }
    if (options.max_ids_per_event == 0
        || options.max_ids_per_event > kMaximumIdsPerEvent) {
        throw InvalidArgumentError(
            "read-workload max IDs per event must be in [1, 65536]");
    }
    const auto maximum_frame =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{options.max_ids_per_event} * sizeof(std::uint64_t);
    if (maximum_frame
            > std::numeric_limits<std::uint64_t>::max() / 2
        || options.buffer_bytes < maximum_frame * 2
        || options.buffer_bytes
               > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
        throw InvalidArgumentError(
            "read-workload buffer must hold at least two maximum events");
    }
    if (options.segment_bytes
        < kReadWorkloadLogHeaderBytes
              + 2 * kReadWorkloadRecordHeaderBytes + maximum_frame) {
        throw InvalidArgumentError(
            "read-workload segment must hold its header, opening checkpoint, "
            "one maximum event, and terminal checkpoint");
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
            "read-workload write budget must cover one maximum event, a "
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
    put_u32(header, 88, options.sample_rate_ppm);
    put_u32(header, 92, options.max_ids_per_event);
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
    std::atomic<std::uint64_t> observed_events{0};
    std::atomic<std::uint64_t> observed_ids{0};
    std::atomic<std::uint64_t> unsampled_events{0};
    std::atomic<std::uint64_t> sampled_events{0};
    std::atomic<std::uint64_t> enqueued_events{0};
    std::atomic<std::uint64_t> written_events{0};
    std::atomic<std::uint64_t> captured_ids{0};
    std::atomic<std::uint64_t> truncated_events{0};
    std::atomic<std::uint64_t> sampled_ids{0};
    std::atomic<std::uint64_t> enqueued_ids{0};
    std::atomic<std::uint64_t> written_ids{0};
    std::atomic<std::uint64_t> dropped_busy{0};
    std::atomic<std::uint64_t> dropped_busy_ids{0};
    std::atomic<std::uint64_t> dropped_full{0};
    std::atomic<std::uint64_t> dropped_full_ids{0};
    std::atomic<std::uint64_t> dropped_runtime_failure{0};
    std::atomic<std::uint64_t> dropped_runtime_failure_ids{0};
    std::atomic<std::uint64_t> dropped_min_free{0};
    std::atomic<std::uint64_t> dropped_min_free_ids{0};
    std::atomic<std::uint64_t> dropped_io_budget{0};
    std::atomic<std::uint64_t> dropped_io_budget_ids{0};
    std::atomic<std::uint64_t> write_errors{0};
    std::atomic<std::uint64_t> bytes_written{0};
    std::atomic<std::uint64_t> segments_created{0};
    std::atomic<std::uint64_t> rotated_segments{0};
    std::atomic<std::uint64_t> retained_bytes{0};
    std::atomic<std::uint64_t> retained_segments{0};
    std::atomic<std::uint64_t> retention_deleted_bytes{0};
    std::atomic<std::uint64_t> retention_deleted_segments{0};
    std::atomic<std::uint64_t> queued_events{0};
    std::atomic<std::uint64_t> queued_bytes{0};
    std::atomic<std::uint64_t> peak_queued_bytes{0};
    std::atomic<int> last_error_number{0};
    std::atomic<bool> paused_for_min_free{false};
    std::atomic<bool> runtime_failed{false};
    std::atomic<bool> shutting_down{false};
};

struct CounterSnapshot {
    std::uint64_t observed_events = 0;
    std::uint64_t observed_ids = 0;
    std::uint64_t unsampled_events = 0;
    std::uint64_t sampled_events = 0;
    std::uint64_t enqueued_events = 0;
    std::uint64_t written_events = 0;
    std::uint64_t captured_ids = 0;
    std::uint64_t truncated_events = 0;
    std::uint64_t sampled_ids = 0;
    std::uint64_t enqueued_ids = 0;
    std::uint64_t written_ids = 0;
    std::uint64_t dropped_busy = 0;
    std::uint64_t dropped_busy_ids = 0;
    std::uint64_t dropped_full = 0;
    std::uint64_t dropped_full_ids = 0;
    std::uint64_t dropped_runtime_failure = 0;
    std::uint64_t dropped_runtime_failure_ids = 0;
    std::uint64_t dropped_min_free = 0;
    std::uint64_t dropped_min_free_ids = 0;
    std::uint64_t dropped_io_budget = 0;
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
    result.observed_events =
        stats.observed_events.load(std::memory_order_relaxed);
    result.observed_ids =
        stats.observed_ids.load(std::memory_order_relaxed);
    result.unsampled_events =
        stats.unsampled_events.load(std::memory_order_relaxed);
    result.sampled_events =
        stats.sampled_events.load(std::memory_order_relaxed);
    result.enqueued_events =
        stats.enqueued_events.load(std::memory_order_relaxed);
    result.written_events =
        stats.written_events.load(std::memory_order_relaxed);
    result.captured_ids =
        stats.captured_ids.load(std::memory_order_relaxed);
    result.truncated_events =
        stats.truncated_events.load(std::memory_order_relaxed);
    result.sampled_ids =
        stats.sampled_ids.load(std::memory_order_relaxed);
    result.enqueued_ids =
        stats.enqueued_ids.load(std::memory_order_relaxed);
    result.written_ids =
        stats.written_ids.load(std::memory_order_relaxed);
    result.dropped_busy =
        stats.dropped_busy.load(std::memory_order_relaxed);
    result.dropped_busy_ids =
        stats.dropped_busy_ids.load(std::memory_order_relaxed);
    result.dropped_full =
        stats.dropped_full.load(std::memory_order_relaxed);
    result.dropped_full_ids =
        stats.dropped_full_ids.load(std::memory_order_relaxed);
    result.dropped_runtime_failure =
        stats.dropped_runtime_failure.load(std::memory_order_relaxed);
    result.dropped_runtime_failure_ids =
        stats.dropped_runtime_failure_ids.load(
            std::memory_order_relaxed);
    result.dropped_min_free =
        stats.dropped_min_free.load(std::memory_order_relaxed);
    result.dropped_min_free_ids =
        stats.dropped_min_free_ids.load(std::memory_order_relaxed);
    result.dropped_io_budget =
        stats.dropped_io_budget.load(std::memory_order_relaxed);
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

    // Producers update several independent saturating atomics without a
    // global lock. A raw relaxed snapshot may therefore observe a child
    // counter after its parent was loaded (for example enqueued=11 while
    // sampled=10). Preserve the hot path and normalize only the serialized
    // snapshot into conservative monotonic lower bounds. This also makes
    // periodic checkpoints semantically valid while requests are active.
    const auto dropped = saturating_add_value(
        saturating_add_value(
            result.dropped_busy, result.dropped_full),
        saturating_add_value(
            saturating_add_value(
                result.dropped_runtime_failure,
                result.dropped_min_free),
            result.dropped_io_budget));
    const auto dropped_ids = saturating_add_value(
        saturating_add_value(
            result.dropped_busy_ids, result.dropped_full_ids),
        saturating_add_value(
            saturating_add_value(
                result.dropped_runtime_failure_ids,
                result.dropped_min_free_ids),
            result.dropped_io_budget_ids));
    result.enqueued_events = std::max(
        {result.enqueued_events,
         result.written_events,
         result.truncated_events});
    result.sampled_events = std::max(
        {result.sampled_events, result.enqueued_events, dropped});
    result.enqueued_ids = std::max(
        {result.enqueued_ids,
         result.written_ids,
         result.captured_ids});
    result.sampled_ids = std::max(
        {result.sampled_ids, result.enqueued_ids, dropped_ids});
    result.observed_events = std::max(
        result.observed_events,
        saturating_add_value(
            result.sampled_events, result.unsampled_events));
    result.observed_ids = std::max(
        {result.observed_ids, result.captured_ids, result.sampled_ids});
    result.segments_created =
        std::max(result.segments_created, result.rotated_segments);
    return result;
}

void encode_counter_snapshot(
    const std::span<std::byte> output,
    const CounterSnapshot& counters) noexcept {
    put_u64(output, 128, counters.observed_events);
    put_u64(output, 136, counters.observed_ids);
    put_u64(output, 144, counters.unsampled_events);
    put_u64(output, 152, counters.sampled_events);
    put_u64(output, 160, counters.enqueued_events);
    put_u64(output, 168, counters.written_events);
    put_u64(output, 176, counters.captured_ids);
    put_u64(output, 184, counters.truncated_events);
    put_u64(output, 192, counters.dropped_busy);
    put_u64(output, 200, counters.dropped_full);
    put_u64(output, 208, counters.dropped_runtime_failure);
    put_u64(output, 216, counters.dropped_min_free);
    put_u64(output, 224, counters.write_errors);
    put_u64(output, 232, counters.bytes_written);
    put_u64(output, 240, counters.segments_created);
    put_u64(output, 248, counters.rotated_segments);
}

void encode_checkpoint_extension(
    const std::span<std::byte> output,
    const CounterSnapshot& counters,
    const bool opening_baseline) noexcept {
    put_u32(output, 28, kCheckpointExtensionVersion);
    put_u32(
        output,
        32,
        opening_baseline ? kCheckpointFlagOpeningBaseline : 0);
    put_u64(output, 56, counters.sampled_ids);
    put_u64(output, 64, counters.enqueued_ids);
    put_u64(output, 72, counters.written_ids);
    put_u64(output, 80, counters.dropped_busy_ids);
    put_u64(output, 88, counters.dropped_full_ids);
    put_u64(output, 96, counters.dropped_runtime_failure_ids);
    put_u64(output, 104, counters.dropped_min_free_ids);
    put_u64(output, 112, counters.dropped_io_budget);
    put_u64(output, 120, counters.dropped_io_budget_ids);
}

[[nodiscard]] bool counters_equal(
    const CounterSnapshot& left,
    const CounterSnapshot& right) noexcept {
    // Storage bookkeeping changes when a checkpoint is appended. It must not
    // make that same checkpoint look dirty forever during an idle period.
    return left.observed_events == right.observed_events
           && left.observed_ids == right.observed_ids
           && left.unsampled_events == right.unsampled_events
           && left.sampled_events == right.sampled_events
           && left.enqueued_events == right.enqueued_events
           && left.written_events == right.written_events
           && left.captured_ids == right.captured_ids
           && left.truncated_events == right.truncated_events
           && left.sampled_ids == right.sampled_ids
           && left.enqueued_ids == right.enqueued_ids
           && left.written_ids == right.written_ids
           && left.dropped_busy == right.dropped_busy
           && left.dropped_busy_ids == right.dropped_busy_ids
           && left.dropped_full == right.dropped_full
           && left.dropped_full_ids == right.dropped_full_ids
           && left.dropped_runtime_failure
                  == right.dropped_runtime_failure
           && left.dropped_runtime_failure_ids
                  == right.dropped_runtime_failure_ids
           && left.dropped_min_free == right.dropped_min_free
           && left.dropped_min_free_ids == right.dropped_min_free_ids
           && left.dropped_io_budget == right.dropped_io_budget
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

class ReadWorkloadLogger::Impl {
public:
    Impl(
        ReadWorkloadLogOptions options,
        ReadWorkloadCollectionInfo collection)
        : options_(validated_options(std::move(options), collection)),
          collection_(collection),
          maximum_frame_bytes_(
              kReadWorkloadRecordHeaderBytes
              + std::size_t{options_.max_ids_per_event}
                    * sizeof(std::uint64_t)),
          slot_count_(
              static_cast<std::size_t>(
                  options_.buffer_bytes / maximum_frame_bytes_)),
          queue_storage_(slot_count_ * maximum_frame_bytes_),
          slot_sizes_(slot_count_, 0),
          retention_sweep_interval_(std::chrono::seconds(
              std::min(
                  options_.retention_seconds,
                  kMaximumRetentionSweepSeconds))) {
        const auto ids = random_instance_id();
        instance_low_ = ids.first;
        instance_high_ = ids.second;
        random_seed_ = splitmix64(
            instance_low_ ^ splitmix64(instance_high_) ^ unix_time_ns());
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

    ~Impl() {
        shutdown();
    }

    void start() {
        worker_ = std::thread([this] { run(); });
    }

    [[nodiscard]] bool try_log(
        const ReadWorkloadMethod method,
        const std::span<const std::uint64_t> ids,
        const ReadWorkloadSnapshot snapshot,
        const std::uint32_t prefix_dimensions) noexcept {
        try {
            // Empty Gets carry no locality signal and the v1 record contract
            // reserves event frames for non-empty requests.
            if (ids.empty() || prefix_dimensions == 0
                || prefix_dimensions > collection_.dimension) {
                return false;
            }
            if (stats_.runtime_failed.load(std::memory_order_acquire)
                || stats_.shutting_down.load(std::memory_order_acquire)) {
                return false;
            }
            const auto original_ids =
                static_cast<std::uint64_t>(ids.size());
            const auto sequence =
                event_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
            atomic_add_saturating(stats_.observed_events);
            atomic_add_saturating(
                stats_.observed_ids,
                original_ids);
            const auto event_sample_seed = splitmix64(
                random_seed_
                ^ splitmix64(sequence ^ kEventSamplingDomain));
            const auto selection_seed = splitmix64(
                random_seed_
                ^ splitmix64(sequence ^ kIdSelectionDomain));
            if (!sampled(event_sample_seed)) {
                atomic_add_saturating(stats_.unsampled_events);
                return false;
            }
            atomic_add_saturating(stats_.sampled_events);
            atomic_add_saturating(stats_.sampled_ids, original_ids);

            if (stats_.shutting_down.load(std::memory_order_acquire)
                || stats_.runtime_failed.load(std::memory_order_acquire)) {
                atomic_add_saturating(stats_.dropped_runtime_failure);
                atomic_add_saturating(
                    stats_.dropped_runtime_failure_ids, original_ids);
                return false;
            }
            if (stats_.paused_for_min_free.load(
                    std::memory_order_acquire)) {
                atomic_add_saturating(stats_.dropped_min_free);
                atomic_add_saturating(
                    stats_.dropped_min_free_ids, original_ids);
                return false;
            }

            const auto captured = std::min<std::size_t>(
                ids.size(), options_.max_ids_per_event);
            const bool truncated = captured != ids.size();
            const auto frame_bytes =
                kReadWorkloadRecordHeaderBytes
                + captured * sizeof(std::uint64_t);

            if (!queue_mutex_.try_lock()) {
                atomic_add_saturating(stats_.dropped_busy);
                atomic_add_saturating(stats_.dropped_busy_ids, original_ids);
                return false;
            }
            std::unique_lock queue_lock(queue_mutex_, std::adopt_lock);
            if (queued_count_ == slot_count_) {
                atomic_add_saturating(stats_.dropped_full);
                atomic_add_saturating(stats_.dropped_full_ids, original_ids);
                return false;
            }
            if (stats_.shutting_down.load(std::memory_order_relaxed)
                || stats_.runtime_failed.load(std::memory_order_relaxed)) {
                atomic_add_saturating(stats_.dropped_runtime_failure);
                atomic_add_saturating(
                    stats_.dropped_runtime_failure_ids, original_ids);
                return false;
            }
            if (stats_.paused_for_min_free.load(
                    std::memory_order_relaxed)) {
                atomic_add_saturating(stats_.dropped_min_free);
                atomic_add_saturating(
                    stats_.dropped_min_free_ids, original_ids);
                return false;
            }

            auto frame = std::span<std::byte>(
                queue_storage_.data() + queue_tail_ * maximum_frame_bytes_,
                frame_bytes);
            std::fill(frame.begin(), frame.end(), std::byte{0});
            std::memcpy(
                frame.data(), kRecordMagic.data(), kRecordMagic.size());
            put_u32(frame, 8, kRecordVersion);
            put_u32(
                frame,
                12,
                static_cast<std::uint32_t>(
                    kReadWorkloadRecordHeaderBytes));
            put_u32(frame, 16, static_cast<std::uint32_t>(frame_bytes));
            put_u32(frame, 20, kEventRecordKind);
            put_u32(frame, 28, static_cast<std::uint32_t>(method));
            std::uint32_t flags = 0;
            if (snapshot.exact) {
                flags |= kRecordFlagExactSnapshot;
            }
            if (truncated) {
                flags |= kRecordFlagTruncatedIds;
            }
            put_u32(frame, 32, flags);
            put_u32(frame, 36, prefix_dimensions);
            put_u64(frame, 40, sequence);
            put_u64(frame, 48, unix_time_ns());
            put_u64(frame, 56, snapshot.catalog_generation);
            put_u64(frame, 64, snapshot.committed_sequence);
            put_u64(frame, 72, snapshot.row_count);
            put_u64(frame, 80, snapshot.stream_id_low);
            put_u64(frame, 88, snapshot.stream_id_high);
            put_u64(frame, 96, snapshot.batch_sequence);
            put_u64(frame, 104, static_cast<std::uint64_t>(ids.size()));
            put_u32(frame, 112, static_cast<std::uint32_t>(captured));
            put_u32(frame, 116, options_.sample_rate_ppm);
            put_u64(frame, 120, selection_seed);

            if (truncated) {
                atomic_add_saturating(stats_.truncated_events);
            }
            atomic_add_saturating(stats_.enqueued_events);
            atomic_add_saturating(stats_.enqueued_ids, original_ids);
            atomic_add_saturating(
                stats_.captured_ids,
                static_cast<std::uint64_t>(captured));
            encode_counter_snapshot(frame, counter_snapshot(stats_));
            encode_ids(frame, ids, captured, selection_seed);
            put_u32(
                frame,
                kReadWorkloadRecordCrcOffset,
                crc32c_with_zeroed_field(
                    frame, kReadWorkloadRecordCrcOffset));

            slot_sizes_[queue_tail_] = frame_bytes;
            queue_tail_ = (queue_tail_ + 1) % slot_count_;
            ++queued_count_;
            queued_bytes_ += frame_bytes;
            stats_.queued_events.store(
                queued_count_, std::memory_order_relaxed);
            stats_.queued_bytes.store(
                queued_bytes_, std::memory_order_relaxed);
            atomic_max(stats_.peak_queued_bytes, queued_bytes_);
            queue_lock.unlock();
            queue_condition_.notify_one();
            return true;
        } catch (...) {
            // No exception from optional telemetry may escape into a request.
            atomic_add_saturating(stats_.dropped_runtime_failure);
            atomic_add_saturating(
                stats_.dropped_runtime_failure_ids,
                static_cast<std::uint64_t>(ids.size()));
            return false;
        }
    }

    [[nodiscard]] ReadWorkloadLogStats stats() const noexcept {
        ReadWorkloadLogStats result;
        result.observed_events =
            stats_.observed_events.load(std::memory_order_relaxed);
        result.observed_ids =
            stats_.observed_ids.load(std::memory_order_relaxed);
        result.unsampled_events =
            stats_.unsampled_events.load(std::memory_order_relaxed);
        result.sampled_events =
            stats_.sampled_events.load(std::memory_order_relaxed);
        result.enqueued_events =
            stats_.enqueued_events.load(std::memory_order_relaxed);
        result.written_events =
            stats_.written_events.load(std::memory_order_relaxed);
        result.captured_ids =
            stats_.captured_ids.load(std::memory_order_relaxed);
        result.truncated_events =
            stats_.truncated_events.load(std::memory_order_relaxed);
        result.sampled_ids =
            stats_.sampled_ids.load(std::memory_order_relaxed);
        result.enqueued_ids =
            stats_.enqueued_ids.load(std::memory_order_relaxed);
        result.written_ids =
            stats_.written_ids.load(std::memory_order_relaxed);
        result.dropped_busy =
            stats_.dropped_busy.load(std::memory_order_relaxed);
        result.dropped_busy_ids =
            stats_.dropped_busy_ids.load(std::memory_order_relaxed);
        result.dropped_full =
            stats_.dropped_full.load(std::memory_order_relaxed);
        result.dropped_full_ids =
            stats_.dropped_full_ids.load(std::memory_order_relaxed);
        result.dropped_runtime_failure =
            stats_.dropped_runtime_failure.load(std::memory_order_relaxed);
        result.dropped_runtime_failure_ids =
            stats_.dropped_runtime_failure_ids.load(
                std::memory_order_relaxed);
        result.dropped_min_free =
            stats_.dropped_min_free.load(std::memory_order_relaxed);
        result.dropped_min_free_ids =
            stats_.dropped_min_free_ids.load(std::memory_order_relaxed);
        result.dropped_io_budget =
            stats_.dropped_io_budget.load(std::memory_order_relaxed);
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
        result.queued_events =
            stats_.queued_events.load(std::memory_order_relaxed);
        result.queued_bytes =
            stats_.queued_bytes.load(std::memory_order_relaxed);
        result.peak_queued_bytes =
            stats_.peak_queued_bytes.load(std::memory_order_relaxed);
        result.buffer_capacity_bytes =
            slot_count_ * maximum_frame_bytes_;
        result.last_error_number =
            stats_.last_error_number.load(std::memory_order_relaxed);
        result.paused_for_min_free =
            stats_.paused_for_min_free.load(std::memory_order_relaxed);
        result.runtime_failed =
            stats_.runtime_failed.load(std::memory_order_relaxed);
        result.shutting_down =
            stats_.shutting_down.load(std::memory_order_relaxed);
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
        queue_condition_.notify_all();
        if (worker_.joinable()
            && worker_.get_id() != std::this_thread::get_id()) {
            worker_.join();
        }
    }

private:
    [[nodiscard]] bool sampled(const std::uint64_t seed) const noexcept {
        if (options_.sample_rate_ppm == kSampleDenominator) {
            return true;
        }
        const auto bucket = uniform_below(seed, kSampleDenominator);
        return bucket < options_.sample_rate_ppm;
    }

    static void encode_ids(
        const std::span<std::byte> frame,
        const std::span<const std::uint64_t> ids,
        const std::size_t captured,
        const std::uint64_t seed) noexcept {
        if (captured == ids.size()) {
            for (std::size_t index = 0; index != captured; ++index) {
                put_u64(
                    frame,
                    kReadWorkloadRecordHeaderBytes
                        + index * sizeof(std::uint64_t),
                    ids[index]);
            }
            return;
        }

        // A random circular translation of K nearly-even base points has
        // exact inclusion probability K/N for every input position, avoids
        // O(N) work and retains coverage across the complete request.
        const auto count = ids.size();
        const auto quotient = count / captured;
        const auto remainder = count % captured;
        const auto offset = static_cast<std::size_t>(uniform_below(
            seed, static_cast<std::uint64_t>(count)));
        for (std::size_t index = 0; index != captured; ++index) {
            const auto base =
                index * quotient + (index * remainder) / captured;
            const auto position =
                base >= count - offset ? base - (count - offset)
                                       : base + offset;
            put_u64(
                frame,
                kReadWorkloadRecordHeaderBytes
                    + index * sizeof(std::uint64_t),
                ids[position]);
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
        // Keep one slot available for the active segment while the logger is
        // running. During final shutdown there is no successor segment, so
        // the full configured count may remain finalized.
        const auto retained_file_limit =
            options_.max_retained_files
            - (stats_.shutting_down.load(std::memory_order_relaxed) ? 0 : 1);

        for (const auto& file : files) {
            const bool aged =
                file.modified_ns <= now
                && now - file.modified_ns >= retention_ns;
            const bool over_budget =
                total_bytes > options_.retention_bytes;
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
        if (total_bytes > options_.retention_bytes
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
        stats_.retained_bytes.store(
            total_bytes, std::memory_order_relaxed);
        stats_.retained_segments.store(
            total_segments, std::memory_order_relaxed);
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
        put_u64(frame, 48, unix_time_ns());
        encode_counter_snapshot(frame, counters);
        encode_checkpoint_extension(
            frame, counters, opening_baseline);
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
        if (!consume_write_budget(immediate_write_bytes)) {
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

    void remove_head_locked() noexcept {
        const auto bytes = slot_sizes_[queue_head_];
        slot_sizes_[queue_head_] = 0;
        queue_head_ = (queue_head_ + 1) % slot_count_;
        --queued_count_;
        queued_bytes_ -= bytes;
        stats_.queued_events.store(
            queued_count_, std::memory_order_relaxed);
        stats_.queued_bytes.store(
            queued_bytes_, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t head_original_ids_locked() const noexcept {
        const auto frame = std::span<const std::byte>(
            queue_storage_.data()
                + queue_head_ * maximum_frame_bytes_,
            slot_sizes_[queue_head_]);
        return get_u64(frame, 104);
    }

    void drop_head_locked(
        std::atomic<std::uint64_t>& event_destination,
        std::atomic<std::uint64_t>& id_destination) noexcept {
        atomic_add_saturating(event_destination);
        atomic_add_saturating(
            id_destination, head_original_ids_locked());
        remove_head_locked();
    }

    void drop_queue(
        std::atomic<std::uint64_t>& event_destination,
        std::atomic<std::uint64_t>& id_destination) noexcept {
        const std::lock_guard lock(queue_mutex_);
        while (queued_count_ != 0) {
            drop_head_locked(event_destination, id_destination);
        }
    }

    void mark_runtime_failure(const int error_number) noexcept {
        stats_.last_error_number.store(
            error_number != 0 ? error_number : EIO,
            std::memory_order_relaxed);
        atomic_add_saturating(stats_.write_errors);
        stats_.runtime_failed.store(true, std::memory_order_release);
        stats_.paused_for_min_free.store(false, std::memory_order_release);
        queue_condition_.notify_all();
    }

    [[nodiscard]] bool consume_write_budget(
        const std::uint64_t bytes) noexcept {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(now - write_budget_updated_);
        if (elapsed.count() > 0) {
            const auto numerator =
                static_cast<unsigned __int128>(
                    options_.max_write_bytes_per_second)
                    * static_cast<std::uint64_t>(elapsed.count())
                + write_budget_remainder_;
            const auto added = static_cast<std::uint64_t>(
                numerator / 1'000'000'000ULL);
            write_budget_remainder_ = static_cast<std::uint64_t>(
                numerator % 1'000'000'000ULL);
            write_budget_tokens_ = std::min(
                options_.max_write_bytes_per_second,
                saturating_add_value(write_budget_tokens_, added));
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
                std::unique_lock lock(queue_mutex_);
                queue_condition_.wait_for(lock, 1s, [&] {
                    return queued_count_ != 0
                           || stats_.runtime_failed.load(
                               std::memory_order_acquire)
                           || stats_.shutting_down.load(
                               std::memory_order_acquire);
                });
                if (stats_.runtime_failed.load(
                        std::memory_order_acquire)) {
                    lock.unlock();
                    break;
                }
                const bool stopping =
                    stats_.shutting_down.load(std::memory_order_acquire);
                if (queued_count_ == 0) {
                    lock.unlock();
                    if (stopping) {
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
                const auto frame_size = slot_sizes_[queue_head_];
                const auto frame = std::span<const std::byte>(
                    queue_storage_.data()
                        + queue_head_ * maximum_frame_bytes_,
                    frame_size);
                lock.unlock();
                if (retention_sweep_due()) {
                    prune_retention();
                }

                const PrepareEventResult prepared =
                    prepare_event_append(frame_size);
                if (prepared == PrepareEventResult::WriteBudget) {
                    const std::lock_guard pop_lock(queue_mutex_);
                    drop_head_locked(
                        stats_.dropped_io_budget,
                        stats_.dropped_io_budget_ids);
                    continue;
                }
                if (prepared == PrepareEventResult::MinimumFreeSpace) {
                    if (stats_.runtime_failed.load(
                            std::memory_order_acquire)) {
                        break;
                    }
                    if (stopping) {
                        drop_queue(
                            stats_.dropped_min_free,
                            stats_.dropped_min_free_ids);
                        break;
                    }
                    std::unique_lock wait_lock(queue_mutex_);
                    queue_condition_.wait_for(wait_lock, 1s);
                    continue;
                }
                append_bytes(frame);
                active_has_records_ = true;
                active_terminal_checkpoint_ = false;
                atomic_add_saturating(stats_.written_events);
                atomic_add_saturating(
                    stats_.written_ids, get_u64(frame, 104));
                ++events_since_checkpoint_;
                {
                    const std::lock_guard pop_lock(queue_mutex_);
                    remove_head_locked();
                }
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
            drop_queue(
                stats_.dropped_runtime_failure,
                stats_.dropped_runtime_failure_ids);
            active_fd_.reset();
        }
    }

    ReadWorkloadLogOptions options_;
    ReadWorkloadCollectionInfo collection_;
    const std::size_t maximum_frame_bytes_;
    const std::size_t slot_count_;
    std::vector<std::byte> queue_storage_;
    std::vector<std::size_t> slot_sizes_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::size_t queue_head_ = 0;
    std::size_t queue_tail_ = 0;
    std::size_t queued_count_ = 0;
    std::uint64_t queued_bytes_ = 0;
    std::thread worker_;
    AtomicStats stats_;
    FileDescriptor writer_lock_;
    FileDescriptor directory_fd_;
    FileDescriptor active_fd_;
    std::string active_name_;
    std::filesystem::path active_path_;
    std::uint64_t active_bytes_ = 0;
    std::chrono::steady_clock::time_point active_started_{};
    bool active_has_records_ = false;
    bool active_terminal_checkpoint_ = false;
    std::uint64_t segment_sequence_ = 0;
    std::uint64_t instance_low_ = 0;
    std::uint64_t instance_high_ = 0;
    std::uint64_t random_seed_ = 0;
    std::uint64_t write_budget_tokens_ =
        options_.max_write_bytes_per_second;
    std::uint64_t write_budget_remainder_ = 0;
    std::chrono::steady_clock::time_point write_budget_updated_ =
        std::chrono::steady_clock::now();
    std::atomic<std::uint64_t> event_sequence_{0};
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
        std::make_unique<Impl>(std::move(options), collection);
    impl->start();
    return std::unique_ptr<ReadWorkloadLogger>(
        new ReadWorkloadLogger(std::move(impl)));
}

ReadWorkloadLogger::ReadWorkloadLogger(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ReadWorkloadLogger::~ReadWorkloadLogger() = default;

bool ReadWorkloadLogger::try_log(
    const ReadWorkloadMethod method,
    const std::span<const std::uint64_t> ids,
    const ReadWorkloadSnapshot snapshot,
    const std::uint32_t prefix_dimensions) noexcept {
    return impl_->try_log(method, ids, snapshot, prefix_dimensions);
}

ReadWorkloadLogStats ReadWorkloadLogger::stats() const noexcept {
    return impl_->stats();
}

void ReadWorkloadLogger::shutdown() noexcept {
    impl_->shutdown();
}

}  // namespace throwdb::server::detail
