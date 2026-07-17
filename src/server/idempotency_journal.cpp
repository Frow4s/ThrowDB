#include "idempotency_journal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "sha256.hpp"
#include "throwdb/errors.hpp"

namespace throwdb::server::detail {
namespace {

constexpr std::array<char, 8> kJournalMagic{
    'T', 'D', 'B', 'I', 'D', 'E', 'M', '1'};
constexpr std::array<char, 8> kRecordMagic{
    'T', 'D', 'B', 'I', 'D', 'R', '0', '1'};
constexpr std::array<char, 8> kPendingFenceMagic{
    'T', 'D', 'B', 'I', 'P', 'N', 'D', '1'};
constexpr std::uint32_t kLegacyJournalVersion = 1;
constexpr std::uint32_t kJournalVersion = 2;
constexpr std::uint32_t kJournalHeaderBytes = 4096;
constexpr std::uint32_t kRecordHeaderBytes = 192;
constexpr std::uint32_t kPendingFenceBytes = 4096;
constexpr std::uint64_t kMaximumRecordBytes = 64ULL << 20;
constexpr std::size_t kJournalCrcOffset = 32;
constexpr std::size_t kRecordCrcOffset = 144;
constexpr std::size_t kPendingFenceCrcOffset = 40;
constexpr std::size_t kJournalDomainLowOffset = 40;
constexpr std::size_t kJournalDomainHighOffset = 48;
constexpr std::size_t kJournalGenerationOffset = 56;
constexpr std::size_t kJournalHeaderUsedBytes = 64;
constexpr std::string_view kPendingFenceName = "pending-mutation-v1";
constexpr std::string_view kCheckpointTempPrefix =
    ".idempotency-checkpoint-";

enum class RecordKind : std::uint32_t {
    Prepare = 1,
    Complete = 2,
    AbortNotApplied = 3,
};

std::atomic<IdempotencyJournalTestHook> journal_test_hook{nullptr};

void invoke_journal_test_hook(const RecordKind kind) {
    const auto hook = journal_test_hook.load(std::memory_order_acquire);
    if (hook == nullptr) {
        return;
    }
    switch (kind) {
        case RecordKind::Prepare:
            hook(IdempotencyJournalTestHookPoint::PrepareSynced);
            return;
        case RecordKind::Complete:
            hook(IdempotencyJournalTestHookPoint::CompleteSynced);
            return;
        case RecordKind::AbortNotApplied:
            hook(IdempotencyJournalTestHookPoint::AbortSynced);
            return;
    }
}

class FileDescriptor {
public:
    explicit FileDescriptor(const int value = -1) noexcept : value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept
        : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (value_ >= 0) {
            ::close(value_);
        }
        value_ = std::exchange(other.value_, -1);
        return *this;
    }
    [[nodiscard]] int get() const noexcept { return value_; }

private:
    int value_ = -1;
};

[[nodiscard]] std::filesystem::path pending_fence_path(
    const std::filesystem::path& journal_path) {
    return journal_path.parent_path() / kPendingFenceName;
}

[[nodiscard]] StorageError io_error(
    const std::string& operation,
    const std::filesystem::path& path) {
    return StorageError(
        operation + " '" + path.string() + "': " + std::strerror(errno));
}

void put_u32(
    const std::span<std::byte> output,
    const std::size_t offset,
    const std::uint32_t value) {
    if (offset > output.size() || 4 > output.size() - offset) {
        throw StorageError("idempotency journal uint32 write overflow");
    }
    for (std::size_t index = 0; index != 4; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8U * index));
    }
}

void put_u64(
    const std::span<std::byte> output,
    const std::size_t offset,
    const std::uint64_t value) {
    if (offset > output.size() || 8 > output.size() - offset) {
        throw StorageError("idempotency journal uint64 write overflow");
    }
    for (std::size_t index = 0; index != 8; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8U * index));
    }
}

[[nodiscard]] std::uint32_t get_u32(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    if (offset > input.size() || 4 > input.size() - offset) {
        throw CorruptionError("idempotency journal uint32 is truncated");
    }
    std::uint32_t result = 0;
    for (std::size_t index = 0; index != 4; ++index) {
        result |= std::uint32_t(std::to_integer<std::uint8_t>(
                      input[offset + index]))
                  << (8U * index);
    }
    return result;
}

[[nodiscard]] std::uint64_t get_u64(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    if (offset > input.size() || 8 > input.size() - offset) {
        throw CorruptionError("idempotency journal uint64 is truncated");
    }
    std::uint64_t result = 0;
    for (std::size_t index = 0; index != 8; ++index) {
        result |= std::uint64_t(std::to_integer<std::uint8_t>(
                      input[offset + index]))
                  << (8U * index);
    }
    return result;
}

[[nodiscard]] std::uint32_t crc32c(
    const std::span<const std::byte> bytes,
    const std::size_t zero_offset,
    const std::size_t zero_size) noexcept {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t index = 0; index != bytes.size(); ++index) {
        const auto byte = index >= zero_offset && index < zero_offset + zero_size
                              ? 0U
                              : std::to_integer<std::uint8_t>(bytes[index]);
        crc ^= byte;
        for (unsigned int bit = 0; bit != 8; ++bit) {
            crc = (crc >> 1U) ^ (0x82f63b78U & (0U - (crc & 1U)));
        }
    }
    return ~crc;
}

[[nodiscard]] OperationId legacy_domain(
    const std::uint64_t collection_id_low,
    const std::uint64_t collection_id_high) noexcept {
    std::array<std::byte, 40> material{};
    constexpr std::array<char, 24> label{
        'T', 'h', 'r', 'o', 'w', 'D', 'B', '-', 'I', 'd', 'e', 'm',
        'p', 'o', 't', 'e', 'n', 'c', 'y', '-', 'V', '1', 0, 0};
    for (std::size_t index = 0; index != label.size(); ++index) {
        material[index] = static_cast<std::byte>(label[index]);
    }
    put_u64(material, 24, collection_id_low);
    put_u64(material, 32, collection_id_high);
    const auto digest = throwdb::detail::sha256(material);
    OperationId result{get_u64(digest, 8), get_u64(digest, 0)};
    if (result.empty()) {
        result.low = 1;
    }
    return result;
}

[[nodiscard]] OperationId random_domain() {
    FileDescriptor random_fd(
        ::open("/dev/urandom", O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (random_fd.get() < 0) {
        throw io_error("cannot open idempotency-domain entropy source",
                       "/dev/urandom");
    }
    std::array<std::byte, 16> bytes{};
    std::size_t cursor = 0;
    while (cursor != bytes.size()) {
        const auto amount = ::read(
            random_fd.get(), bytes.data() + cursor, bytes.size() - cursor);
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw io_error("cannot read idempotency-domain entropy source",
                           "/dev/urandom");
        }
        if (amount == 0) {
            throw StorageError(
                "idempotency-domain entropy source ended unexpectedly");
        }
        cursor += static_cast<std::size_t>(amount);
    }
    OperationId result{get_u64(bytes, 8), get_u64(bytes, 0)};
    if (result.empty()) {
        result.low = 1;
    }
    return result;
}

[[nodiscard]] std::array<std::byte, kJournalHeaderBytes> encode_header(
    const std::uint64_t collection_id_low,
    const std::uint64_t collection_id_high,
    const OperationId domain,
    const std::uint64_t generation) {
    if (domain.empty() || generation == 0) {
        throw InvalidArgumentError(
            "idempotency checkpoint domain and generation must be nonzero");
    }
    std::array<std::byte, kJournalHeaderBytes> header{};
    std::memcpy(header.data(), kJournalMagic.data(), kJournalMagic.size());
    put_u32(header, 8, kJournalVersion);
    put_u32(header, 12, kJournalHeaderBytes);
    put_u64(header, 16, collection_id_low);
    put_u64(header, 24, collection_id_high);
    put_u64(header, kJournalDomainLowOffset, domain.low);
    put_u64(header, kJournalDomainHighOffset, domain.high);
    put_u64(header, kJournalGenerationOffset, generation);
    put_u32(header, kJournalCrcOffset,
            crc32c(header, kJournalCrcOffset, 4));
    return header;
}

void pread_exact(
    const int fd,
    const std::span<std::byte> output,
    const std::uint64_t offset,
    const std::filesystem::path& path) {
    std::size_t cursor = 0;
    while (cursor != output.size()) {
        const auto amount = ::pread(
            fd,
            output.data() + cursor,
            output.size() - cursor,
            static_cast<off_t>(offset + cursor));
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw io_error("cannot read idempotency journal", path);
        }
        if (amount == 0) {
            throw CorruptionError(
                "idempotency journal ended during a checked read");
        }
        cursor += static_cast<std::size_t>(amount);
    }
}

void pwrite_all(
    const int fd,
    const std::span<const std::byte> input,
    const std::uint64_t offset,
    const std::filesystem::path& path) {
    std::size_t cursor = 0;
    while (cursor != input.size()) {
        const auto amount = ::pwrite(
            fd,
            input.data() + cursor,
            input.size() - cursor,
            static_cast<off_t>(offset + cursor));
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw io_error("cannot append idempotency journal", path);
        }
        if (amount == 0) {
            throw StorageError("idempotency journal append made no progress");
        }
        cursor += static_cast<std::size_t>(amount);
    }
}

void sync_fd(
    const int fd,
    const std::filesystem::path& path,
    const std::string& description) {
    while (::fsync(fd) != 0) {
        if (errno == EINTR) {
            continue;
        }
        throw io_error("cannot sync " + description, path);
    }
}

void sync_directory(const std::filesystem::path& path) {
    FileDescriptor directory(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (directory.get() < 0) {
        throw io_error("cannot open directory for sync", path);
    }
    sync_fd(directory.get(), path, "directory");
}

[[nodiscard]] std::uint64_t file_size(
    int fd,
    const std::filesystem::path& path);
[[nodiscard]] bool same_magic(
    std::span<const std::byte> bytes,
    const std::array<char, 8>& magic) noexcept;
[[nodiscard]] bool path_is_missing(const std::error_code& error) noexcept;

struct PendingFenceIdentity {
    OperationId operation_id;
    std::uint64_t expected_sequence = 0;
};

[[nodiscard]] std::array<std::byte, kPendingFenceBytes>
encode_pending_fence(const JournalPreparedOperation& operation) {
    std::array<std::byte, kPendingFenceBytes> encoded{};
    std::memcpy(
        encoded.data(), kPendingFenceMagic.data(), kPendingFenceMagic.size());
    put_u32(encoded, 8, kJournalVersion);
    put_u32(encoded, 12, kPendingFenceBytes);
    put_u64(encoded, 16, operation.operation_id.low);
    put_u64(encoded, 24, operation.operation_id.high);
    put_u64(encoded, 32, operation.expected_sequence);
    put_u32(
        encoded,
        kPendingFenceCrcOffset,
        crc32c(encoded, kPendingFenceCrcOffset, 4));
    return encoded;
}

[[nodiscard]] std::optional<PendingFenceIdentity> read_pending_fence(
    const std::filesystem::path& journal_path) {
    const auto path = pending_fence_path(journal_path);
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error) {
        if (path_is_missing(error)) {
            return std::nullopt;
        }
        throw StorageError(
            "cannot inspect pending-mutation fence: " + error.message());
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        return std::nullopt;
    }
    if (status.type() != std::filesystem::file_type::regular) {
        throw CorruptionError(
            "pending-mutation fence is not a regular file");
    }

    FileDescriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (fd.get() < 0) {
        throw io_error("cannot open pending-mutation fence", path);
    }
    if (file_size(fd.get(), path) != kPendingFenceBytes) {
        throw CorruptionError("pending-mutation fence has an invalid size");
    }
    std::array<std::byte, kPendingFenceBytes> encoded{};
    pread_exact(fd.get(), encoded, 0, path);
    const auto version = get_u32(encoded, 8);
    if (!same_magic(encoded, kPendingFenceMagic)
        || (version != kLegacyJournalVersion && version != kJournalVersion)
        || get_u32(encoded, 12) != kPendingFenceBytes
        || get_u32(encoded, kPendingFenceCrcOffset)
               != crc32c(encoded, kPendingFenceCrcOffset, 4)) {
        throw CorruptionError("pending-mutation fence is corrupt");
    }
    for (std::size_t offset = 44; offset != encoded.size(); ++offset) {
        if (encoded[offset] != std::byte{0}) {
            throw CorruptionError(
                "pending-mutation fence contains nonzero reserved bytes");
        }
    }
    PendingFenceIdentity identity{
        OperationId{get_u64(encoded, 24), get_u64(encoded, 16)},
        get_u64(encoded, 32)};
    if (identity.operation_id.empty() || identity.expected_sequence == 0) {
        throw CorruptionError(
            "pending-mutation fence has invalid operation identity");
    }
    return identity;
}

void create_pending_fence(
    const std::filesystem::path& journal_path,
    const JournalPreparedOperation& operation) {
    const auto path = pending_fence_path(journal_path);
    const int raw_fd = ::open(
        path.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        S_IRUSR | S_IWUSR);
    if (raw_fd < 0) {
        throw io_error("cannot create pending-mutation fence", path);
    }
    FileDescriptor fd(raw_fd);
    try {
        const auto encoded = encode_pending_fence(operation);
        pwrite_all(fd.get(), encoded, 0, path);
        sync_fd(fd.get(), path, "pending-mutation fence");
        sync_directory(path.parent_path());
    } catch (...) {
        // No WAL append can happen until this function returns. Best-effort
        // cleanup is therefore safe; a cleanup failure merely leaves a
        // fail-closed stale fence for recovery to reconcile.
        static_cast<void>(::unlink(path.c_str()));
        try {
            sync_directory(path.parent_path());
        } catch (...) {
        }
        throw;
    }
}

void remove_pending_fence(
    const std::filesystem::path& journal_path,
    const JournalPreparedOperation& operation) {
    const auto identity = read_pending_fence(journal_path);
    if (!identity.has_value()) {
        throw CorruptionError(
            "durable PREPARE has no pending-mutation fence");
    }
    if (identity->operation_id != operation.operation_id
        || identity->expected_sequence != operation.expected_sequence) {
        throw CorruptionError(
            "pending-mutation fence disagrees with the journal PREPARE");
    }
    if (const auto hook = journal_test_hook.load(std::memory_order_acquire);
        hook != nullptr) {
        hook(IdempotencyJournalTestHookPoint::FenceRemove);
    }
    const auto path = pending_fence_path(journal_path);
    if (::unlink(path.c_str()) != 0) {
        throw io_error("cannot remove pending-mutation fence", path);
    }
    sync_directory(path.parent_path());
}

[[nodiscard]] std::uint64_t file_size(
    const int fd,
    const std::filesystem::path& path) {
    struct stat status {};
    if (::fstat(fd, &status) != 0) {
        throw io_error("cannot stat idempotency journal", path);
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        throw CorruptionError("idempotency journal is not a regular file");
    }
    return static_cast<std::uint64_t>(status.st_size);
}

void truncate_tail(
    const int fd,
    const std::uint64_t size,
    const std::filesystem::path& path) {
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        throw io_error("cannot truncate torn idempotency journal tail", path);
    }
    sync_fd(fd, path, "repaired idempotency journal");
}

[[nodiscard]] bool path_is_missing(const std::error_code& error) noexcept {
    return error == std::make_error_code(std::errc::no_such_file_or_directory);
}

[[nodiscard]] bool same_magic(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic) noexcept {
    return bytes.size() >= magic.size()
           && std::memcmp(bytes.data(), magic.data(), magic.size()) == 0;
}

void copy_digest(
    const std::span<std::byte> output,
    const std::size_t offset,
    const std::span<const std::byte, 32> digest) {
    std::copy(digest.begin(), digest.end(), output.begin() + offset);
}

[[nodiscard]] PayloadDigest read_digest(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    if (offset > input.size() || 32 > input.size() - offset) {
        throw CorruptionError("idempotency journal digest is truncated");
    }
    PayloadDigest result{};
    std::copy_n(input.begin() + offset, result.size(), result.begin());
    return result;
}

[[nodiscard]] bool valid_mutation_kind(const MutationKind kind) noexcept {
    return kind == MutationKind::Insert || kind == MutationKind::Update
           || kind == MutationKind::Upsert || kind == MutationKind::Delete;
}

[[nodiscard]] bool valid_result_shape(
    const JournalPreparedOperation& operation) noexcept {
    if (operation.kind == MutationKind::Delete) {
        return operation.deleted.size() == operation.row_count
               && std::all_of(
                   operation.deleted.begin(),
                   operation.deleted.end(),
                   [](const std::uint8_t value) { return value <= 1; });
    }
    return operation.deleted.empty();
}

[[nodiscard]] bool same_prepared_operation(
    const JournalPreparedOperation& left,
    const JournalPreparedOperation& right) noexcept {
    return left.operation_id == right.operation_id
           && left.request_digest == right.request_digest
           && left.wal_fingerprint == right.wal_fingerprint
           && left.kind == right.kind
           && left.expected_sequence == right.expected_sequence
           && left.row_count == right.row_count
           && left.deleted == right.deleted;
}

[[nodiscard]] std::uint64_t expected_commit_sequence(
    const JournalPreparedOperation& operation) noexcept {
    const bool commits = operation.kind == MutationKind::Delete
                             ? std::any_of(
                                   operation.deleted.begin(),
                                   operation.deleted.end(),
                                   [](const std::uint8_t value) {
                                       return value != 0;
                                   })
                             : operation.row_count != 0;
    return commits ? operation.expected_sequence
                   : operation.expected_sequence - 1;
}

[[nodiscard]] std::filesystem::path prepare_server_directory(
    const std::filesystem::path& collection_path) {
    const auto server_path = collection_path / "server";
    std::error_code error;
    const auto status = std::filesystem::symlink_status(server_path, error);
    if (error
        && error != std::make_error_code(
                        std::errc::no_such_file_or_directory)) {
        throw StorageError(
            "cannot inspect server metadata directory: " + error.message());
    }
    if (error) {
        error.clear();
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        if (!std::filesystem::create_directory(server_path, error) || error) {
            throw StorageError(
                "cannot create server metadata directory: " + error.message());
        }
        sync_directory(collection_path);
    } else if (status.type() != std::filesystem::file_type::directory) {
        throw CorruptionError(
            "server metadata path is not a real directory");
    }
    return server_path;
}

}  // namespace

void set_idempotency_journal_test_hook(
    const IdempotencyJournalTestHook hook) noexcept {
    journal_test_hook.store(hook, std::memory_order_release);
}

std::size_t OperationIdHash::operator()(const OperationId value) const noexcept {
    auto mixed = value.low ^ std::rotl(value.high, 23);
    mixed ^= mixed >> 30U;
    mixed *= 0xbf58476d1ce4e5b9ULL;
    mixed ^= mixed >> 27U;
    return static_cast<std::size_t>(mixed);
}

class IdempotencyJournal::Impl {
public:
    Impl(
        std::filesystem::path path,
        FileDescriptor fd,
        const std::uint64_t collection_id_low,
        const std::uint64_t collection_id_high,
        const std::uint64_t valid_end,
        const IdempotencyJournalLimits limits)
        : path_(std::move(path)),
          fd_(std::move(fd)),
          collection_id_low_(collection_id_low),
          collection_id_high_(collection_id_high),
          valid_end_(valid_end),
          limits_(limits) {}

    void recover() {
        const auto size = file_size(fd_.get(), path_);
        if (size < kJournalHeaderBytes) {
            throw CorruptionError("idempotency journal header is truncated");
        }
        std::array<std::byte, kJournalHeaderBytes> header{};
        pread_exact(fd_.get(), header, 0, path_);
        const std::uint32_t version = get_u32(header, 8);
        if (!same_magic(header, kJournalMagic)
            || (version != kLegacyJournalVersion
                && version != kJournalVersion)
            || get_u32(header, 12) != kJournalHeaderBytes
            || get_u64(header, 16) != collection_id_low_
            || get_u64(header, 24) != collection_id_high_
            || get_u32(header, kJournalCrcOffset)
                   != crc32c(header, kJournalCrcOffset, 4)) {
            throw CorruptionError(
                "idempotency journal header is corrupt or belongs to another collection");
        }
        journal_version_ = version;
        if (version == kLegacyJournalVersion) {
            domain_ = legacy_domain(
                collection_id_low_, collection_id_high_);
            generation_ = 1;
        } else {
            for (std::size_t offset = 36;
                 offset != kJournalDomainLowOffset; ++offset) {
                if (header[offset] != std::byte{0}) {
                    throw CorruptionError(
                        "idempotency journal header contains nonzero flags");
                }
            }
            domain_ = OperationId{
                get_u64(header, kJournalDomainHighOffset),
                get_u64(header, kJournalDomainLowOffset)};
            generation_ = get_u64(header, kJournalGenerationOffset);
            if (domain_.empty() || generation_ == 0) {
                throw CorruptionError(
                    "idempotency journal has an invalid replay domain");
            }
        }
        const std::size_t reserved_begin =
            version == kLegacyJournalVersion ? 36 : kJournalHeaderUsedBytes;
        for (std::size_t offset = reserved_begin;
             offset != header.size(); ++offset) {
            if (header[offset] != std::byte{0}) {
                throw CorruptionError(
                    "idempotency journal header contains nonzero reserved bytes");
            }
        }

        valid_end_ = kJournalHeaderBytes;
        while (valid_end_ != size) {
            if (size - valid_end_ < kRecordHeaderBytes) {
                truncate_tail(fd_.get(), valid_end_, path_);
                return;
            }
            std::array<std::byte, kRecordHeaderBytes> encoded_header{};
            pread_exact(fd_.get(), encoded_header, valid_end_, path_);
            if (!same_magic(encoded_header, kRecordMagic)
                || get_u32(encoded_header, 8) != journal_version_
                || get_u32(encoded_header, 12) != kRecordHeaderBytes) {
                throw CorruptionError(
                    "idempotency journal record header is corrupt");
            }
            const auto record_size = get_u64(encoded_header, 24);
            const auto payload_size = get_u64(encoded_header, 136);
            if (record_size < kRecordHeaderBytes
                || record_size > kMaximumRecordBytes
                || payload_size != record_size - kRecordHeaderBytes) {
                throw CorruptionError(
                    "idempotency journal record geometry is corrupt");
            }
            if (record_size > size - valid_end_) {
                truncate_tail(fd_.get(), valid_end_, path_);
                return;
            }
            if (record_size > limits_.max_bytes - valid_end_) {
                throw IdempotencyCapacityError(
                    "idempotency journal exceeds the configured startup byte "
                    "limit; raise idempotency_journal_max_bytes or restore "
                    "from a bounded checkpoint");
            }
            std::vector<std::byte> record(static_cast<std::size_t>(record_size));
            pread_exact(fd_.get(), record, valid_end_, path_);
            if (get_u32(record, kRecordCrcOffset)
                != crc32c(record, kRecordCrcOffset, 4)) {
                throw CorruptionError(
                    "idempotency journal record checksum mismatch");
            }
            for (std::size_t offset = 148; offset != kRecordHeaderBytes; ++offset) {
                if (record[offset] != std::byte{0}) {
                    throw CorruptionError(
                        "idempotency journal record has nonzero reserved bytes");
                }
            }
            apply_record(record);
            valid_end_ += record_size;
        }
    }

    [[nodiscard]] const JournalCompletedOperation* completed(
        const OperationId id) const noexcept {
        const auto found = completed_.find(id);
        return found == completed_.end() ? nullptr : &found->second;
    }
    [[nodiscard]] const JournalPreparedOperation* pending() const noexcept {
        return pending_.has_value() ? &*pending_ : nullptr;
    }
    [[nodiscard]] const JournalPreparedOperation* aborted(
        const OperationId id) const noexcept {
        const auto found = aborted_.find(id);
        return found == aborted_.end() ? nullptr : &found->second;
    }

    [[nodiscard]] OperationId domain() const noexcept { return domain_; }

    [[nodiscard]] IdempotencyJournalStats stats() const noexcept {
        return IdempotencyJournalStats{
            pending_.has_value(),
            valid_end_,
            static_cast<std::uint64_t>(completed_.size()),
            static_cast<std::uint64_t>(aborted_.size()),
            limits_.max_bytes,
            limits_.max_operations,
            domain_,
            generation_};
    }

    [[nodiscard]] OperationId checkpoint() {
        ensure_healthy();
        reconcile_pending_fence();
        if (pending_.has_value()) {
            throw PendingOperationError(
                "idempotency journal cannot checkpoint a pending PREPARE");
        }
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError(
                "idempotency replay-domain generation is exhausted");
        }

        OperationId next_domain;
        do {
            next_domain = random_domain();
        } while (next_domain == domain_);
        const std::uint64_t next_generation = generation_ + 1;
        const auto directory = path_.parent_path();
        const auto temporary = directory
            / (std::string(kCheckpointTempPrefix)
               + std::to_string(static_cast<unsigned long long>(::getpid()))
               + "-" + std::to_string(next_generation) + "-"
               + std::to_string(next_domain.low));
        const int raw_fd = ::open(
            temporary.c_str(),
            O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR);
        if (raw_fd < 0) {
            throw io_error(
                "cannot create idempotency checkpoint", temporary);
        }
        FileDescriptor next_fd(raw_fd);
        bool published = false;
        try {
            const auto header = encode_header(
                collection_id_low_, collection_id_high_, next_domain,
                next_generation);
            pwrite_all(next_fd.get(), header, 0, temporary);
            sync_fd(next_fd.get(), temporary, "idempotency checkpoint");
            if (::rename(temporary.c_str(), path_.c_str()) != 0) {
                throw io_error(
                    "cannot publish idempotency checkpoint", path_);
            }
            published = true;
            sync_directory(directory);
        } catch (...) {
            if (!published) {
                static_cast<void>(::unlink(temporary.c_str()));
            } else {
                // The pathname may already select the fresh domain while the
                // old descriptor remains active. No further mutation is safe
                // until reopen resolves which header survived.
                poisoned_ = true;
            }
            throw;
        }

        fd_ = std::move(next_fd);
        valid_end_ = kJournalHeaderBytes;
        journal_version_ = kJournalVersion;
        domain_ = next_domain;
        generation_ = next_generation;
        completed_.clear();
        aborted_.clear();
        if (const auto hook =
                journal_test_hook.load(std::memory_order_acquire);
            hook != nullptr) {
            hook(IdempotencyJournalTestHookPoint::CheckpointPublished);
        }
        return domain_;
    }

    void reconcile_pending_fence() {
        ensure_healthy();
        const auto identity = read_pending_fence(path_);
        if (pending_.has_value()) {
            if (!identity.has_value()) {
                throw CorruptionError(
                    "durable idempotency PREPARE has no lifecycle fence");
            }
            if (identity->operation_id != pending_->operation_id
                || identity->expected_sequence != pending_->expected_sequence) {
                throw CorruptionError(
                    "pending-mutation fence disagrees with journal recovery");
            }
            return;
        }
        if (identity.has_value()) {
            // The marker is made durable before PREPARE and removed only
            // after COMPLETE/ABORT. Therefore a marker without a recovered
            // PREPARE is a safe crash remnant, never evidence of an applied
            // mutation.
            JournalPreparedOperation stale;
            stale.operation_id = identity->operation_id;
            stale.expected_sequence = identity->expected_sequence;
            remove_pending_fence(path_, stale);
        }
    }

    void prepare(const JournalPreparedOperation& operation) {
        ensure_healthy();
        if (operation.operation_id.empty() || operation.expected_sequence == 0
            || pending_.has_value() || completed_.contains(operation.operation_id)) {
            throw IdempotencyConflictError(
                "operation_id cannot enter PREPARE state");
        }
        if (!valid_mutation_kind(operation.kind)
            || !valid_result_shape(operation)) {
            throw InvalidArgumentError(
                "idempotency PREPARE result shape is invalid");
        }
        if (const auto* prior = aborted(operation.operation_id);
            prior != nullptr && prior->request_digest != operation.request_digest) {
            throw IdempotencyConflictError(
                "operation_id was already used with a different payload");
        }
        const bool reuses_aborted =
            aborted_.contains(operation.operation_id);
        if (!reuses_aborted
            && operation_count() >= limits_.max_operations) {
            throw IdempotencyCapacityError(
                "idempotency operation-history limit reached; existing "
                "operation IDs remain replayable, but new mutations require "
                "a bounded maintenance checkpoint and a refreshed replay "
                "domain");
        }
        const auto record_bytes = checked_record_bytes(operation);
        if (record_bytes > limits_.max_bytes - valid_end_
            || record_bytes
                   > limits_.max_bytes - valid_end_ - record_bytes) {
            throw IdempotencyCapacityError(
                "idempotency journal byte limit cannot retain both PREPARE "
                "and its terminal COMPLETE/ABORT record");
        }
        reconcile_pending_fence();
        JournalPreparedOperation staged = operation;
        create_pending_fence(path_, operation);
        try {
            append(RecordKind::Prepare, operation, 0);
        } catch (...) {
            // pwrite/fsync can report failure after bytes reached stable
            // storage. Keep the fence and refuse every further write until
            // reopen determines whether PREPARE is present.
            poisoned_ = true;
            throw;
        }
        aborted_.erase(operation.operation_id);
        pending_.emplace(std::move(staged));
    }

    JournalCompletedOperation complete(
        const OperationId operation_id,
        const std::uint64_t commit_sequence) {
        ensure_healthy();
        if (!pending_.has_value() || pending_->operation_id != operation_id) {
            throw IdempotencyConflictError(
                "operation_id has no matching PREPARE record");
        }
        if (commit_sequence != expected_commit_sequence(*pending_)) {
            throw IdempotencyConflictError(
                "operation_id COMPLETE sequence does not match PREPARE");
        }
        const JournalPreparedOperation prepared = *pending_;
        JournalCompletedOperation result{prepared, commit_sequence};
        std::unordered_map<
            OperationId, JournalCompletedOperation, OperationIdHash> staged;
        staged.emplace(operation_id, result);
        auto node = staged.extract(operation_id);
        completed_.reserve(completed_.size() + 1);
        try {
            append(RecordKind::Complete, prepared, commit_sequence);
        } catch (...) {
            poisoned_ = true;
            throw;
        }
        try {
            static_cast<void>(completed_.insert(std::move(node)));
        } catch (...) {
            // COMPLETE is already durable. Do not risk appending it twice;
            // reopen will reconstruct the completed map from the journal.
            poisoned_ = true;
            throw;
        }
        pending_.reset();
        remove_pending_fence(path_, prepared);
        return result;
    }

    void abort(const OperationId operation_id) {
        ensure_healthy();
        if (!pending_.has_value() || pending_->operation_id != operation_id) {
            throw IdempotencyConflictError(
                "operation_id has no matching PREPARE record");
        }
        const JournalPreparedOperation prepared = *pending_;
        std::unordered_map<
            OperationId, JournalPreparedOperation, OperationIdHash> staged;
        staged.emplace(operation_id, prepared);
        auto node = staged.extract(operation_id);
        aborted_.reserve(aborted_.size() + 1);
        try {
            append(RecordKind::AbortNotApplied, prepared, 0);
        } catch (...) {
            poisoned_ = true;
            throw;
        }
        try {
            static_cast<void>(aborted_.insert(std::move(node)));
        } catch (...) {
            poisoned_ = true;
            throw;
        }
        pending_.reset();
        remove_pending_fence(path_, prepared);
    }

private:
    void apply_record(const std::span<const std::byte> record) {
        const auto record_kind = static_cast<RecordKind>(get_u32(record, 16));
        JournalPreparedOperation operation;
        operation.kind = static_cast<MutationKind>(get_u32(record, 20));
        operation.operation_id = OperationId{
            get_u64(record, 40), get_u64(record, 32)};
        operation.request_digest = read_digest(record, 48);
        operation.wal_fingerprint.bytes = read_digest(record, 80);
        operation.expected_sequence = get_u64(record, 112);
        const auto commit_sequence = get_u64(record, 120);
        operation.row_count = get_u64(record, 128);
        const auto payload_size = get_u64(record, 136);
        if (payload_size > std::numeric_limits<std::size_t>::max()) {
            throw CorruptionError("idempotency result mask does not fit memory");
        }
        operation.deleted.resize(static_cast<std::size_t>(payload_size));
        std::transform(
            record.begin() + kRecordHeaderBytes,
            record.end(),
            operation.deleted.begin(),
            [](const std::byte value) {
                return std::to_integer<std::uint8_t>(value);
            });
        if (operation.operation_id.empty() || operation.expected_sequence == 0
            || !valid_mutation_kind(operation.kind)
            || !valid_result_shape(operation)) {
            throw CorruptionError(
                "idempotency journal record semantics are corrupt");
        }

        switch (record_kind) {
            case RecordKind::Prepare:
                if (commit_sequence != 0 || pending_.has_value()
                    || completed_.contains(operation.operation_id)) {
                    throw CorruptionError(
                        "idempotency journal has an invalid PREPARE transition");
                }
                if (!aborted_.contains(operation.operation_id)
                    && operation_count() >= limits_.max_operations) {
                    throw IdempotencyCapacityError(
                        "idempotency journal operation-history limit was "
                        "exceeded during recovery");
                }
                if (const auto found = aborted_.find(operation.operation_id);
                    found != aborted_.end()
                    && found->second.request_digest != operation.request_digest) {
                    throw CorruptionError(
                        "aborted operation_id was reused with another digest");
                }
                aborted_.erase(operation.operation_id);
                pending_ = std::move(operation);
                return;
            case RecordKind::Complete:
                if (!pending_.has_value()
                    || !same_prepared_operation(*pending_, operation)
                    || commit_sequence
                           != expected_commit_sequence(operation)) {
                    throw CorruptionError(
                        "idempotency journal has an invalid COMPLETE transition");
                }
                completed_.insert_or_assign(
                    operation.operation_id,
                    JournalCompletedOperation{*pending_, commit_sequence});
                pending_.reset();
                return;
            case RecordKind::AbortNotApplied:
                if (commit_sequence != 0 || !pending_.has_value()
                    || !same_prepared_operation(*pending_, operation)) {
                    throw CorruptionError(
                        "idempotency journal has an invalid ABORT transition");
                }
                aborted_.insert_or_assign(operation.operation_id, *pending_);
                pending_.reset();
                return;
        }
        throw CorruptionError("idempotency journal record kind is invalid");
    }

    void append(
        const RecordKind record_kind,
        const JournalPreparedOperation& operation,
        const std::uint64_t commit_sequence) {
        const auto payload_size = operation.deleted.size();
        if (payload_size > kMaximumRecordBytes - kRecordHeaderBytes) {
            throw InvalidArgumentError(
                "idempotency delete mask exceeds journal record limit");
        }
        std::vector<std::byte> record(kRecordHeaderBytes + payload_size);
        std::memcpy(record.data(), kRecordMagic.data(), kRecordMagic.size());
        put_u32(record, 8, journal_version_);
        put_u32(record, 12, kRecordHeaderBytes);
        put_u32(record, 16, static_cast<std::uint32_t>(record_kind));
        put_u32(record, 20, static_cast<std::uint32_t>(operation.kind));
        put_u64(record, 24, record.size());
        put_u64(record, 32, operation.operation_id.low);
        put_u64(record, 40, operation.operation_id.high);
        copy_digest(record, 48, operation.request_digest);
        copy_digest(record, 80, operation.wal_fingerprint.bytes);
        put_u64(record, 112, operation.expected_sequence);
        put_u64(record, 120, commit_sequence);
        put_u64(record, 128, operation.row_count);
        put_u64(record, 136, payload_size);
        for (std::size_t index = 0; index != payload_size; ++index) {
            record[kRecordHeaderBytes + index] =
                static_cast<std::byte>(operation.deleted[index]);
        }
        put_u32(record, kRecordCrcOffset, crc32c(record, kRecordCrcOffset, 4));
        if (record.size() > limits_.max_bytes - valid_end_) {
            throw IdempotencyCapacityError(
                "idempotency journal byte limit reached before terminal "
                "record publication");
        }
        pwrite_all(fd_.get(), record, valid_end_, path_);
        sync_fd(fd_.get(), path_, "idempotency journal record");
        invoke_journal_test_hook(record_kind);
        valid_end_ += record.size();
    }

    void ensure_healthy() const {
        if (poisoned_) {
            throw StorageError(
                "idempotency journal write outcome is ambiguous; reopen the "
                "writable server before another mutation");
        }
    }

    [[nodiscard]] std::uint64_t operation_count() const noexcept {
        return static_cast<std::uint64_t>(completed_.size())
               + static_cast<std::uint64_t>(aborted_.size())
               + static_cast<std::uint64_t>(pending_.has_value());
    }

    [[nodiscard]] static std::uint64_t checked_record_bytes(
        const JournalPreparedOperation& operation) {
        if (operation.deleted.size()
            > std::numeric_limits<std::uint64_t>::max()
                  - kRecordHeaderBytes) {
            throw InvalidArgumentError(
                "idempotency record byte size overflows uint64");
        }
        return kRecordHeaderBytes
               + static_cast<std::uint64_t>(operation.deleted.size());
    }

    std::filesystem::path path_;
    FileDescriptor fd_;
    std::uint64_t collection_id_low_ = 0;
    std::uint64_t collection_id_high_ = 0;
    std::uint64_t valid_end_ = kJournalHeaderBytes;
    std::unordered_map<
        OperationId, JournalCompletedOperation, OperationIdHash> completed_;
    std::unordered_map<
        OperationId, JournalPreparedOperation, OperationIdHash> aborted_;
    std::optional<JournalPreparedOperation> pending_;
    bool poisoned_ = false;
    std::uint32_t journal_version_ = kLegacyJournalVersion;
    OperationId domain_;
    std::uint64_t generation_ = 0;
    IdempotencyJournalLimits limits_;
};

std::unique_ptr<IdempotencyJournal> IdempotencyJournal::open(
    const std::filesystem::path& collection_path,
    const std::uint64_t collection_id_low,
    const std::uint64_t collection_id_high,
    const IdempotencyJournalLimits limits) {
    if (collection_path.empty()
        || (collection_id_low == 0 && collection_id_high == 0)) {
        throw InvalidArgumentError(
            "durable idempotency requires a collection path and identity");
    }
    if (limits.max_bytes < kJournalHeaderBytes + 2 * kRecordHeaderBytes
        || limits.max_operations == 0) {
        throw InvalidArgumentError(
            "idempotency journal limits must fit the header, PREPARE, "
            "terminal record, and at least one operation");
    }
    const auto directory = prepare_server_directory(collection_path);
    const auto path = directory / "idempotency-v1.log";

    int raw_fd = ::open(
        path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    bool created = false;
    if (raw_fd < 0 && errno == ENOENT) {
        raw_fd = ::open(
            path.c_str(),
            O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR);
        created = raw_fd >= 0;
    }
    if (raw_fd < 0) {
        throw io_error("cannot open idempotency journal", path);
    }
    FileDescriptor fd(raw_fd);

    if (created) {
        const auto header = encode_header(
            collection_id_low, collection_id_high, random_domain(), 1);
        pwrite_all(fd.get(), header, 0, path);
        sync_fd(fd.get(), path, "idempotency journal header");
        sync_directory(directory);
    }

    auto impl = std::make_unique<Impl>(
        path,
        std::move(fd),
        collection_id_low,
        collection_id_high,
        kJournalHeaderBytes,
        limits);
    impl->recover();
    impl->reconcile_pending_fence();
    return std::unique_ptr<IdempotencyJournal>(
        new IdempotencyJournal(std::move(impl)));
}

IdempotencyJournal::IdempotencyJournal(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
IdempotencyJournal::~IdempotencyJournal() = default;

const JournalCompletedOperation* IdempotencyJournal::completed(
    const OperationId operation_id) const noexcept {
    return impl_->completed(operation_id);
}
const JournalPreparedOperation* IdempotencyJournal::pending() const noexcept {
    return impl_->pending();
}
const JournalPreparedOperation* IdempotencyJournal::aborted(
    const OperationId operation_id) const noexcept {
    return impl_->aborted(operation_id);
}
OperationId IdempotencyJournal::domain() const noexcept {
    return impl_->domain();
}
IdempotencyJournalStats IdempotencyJournal::stats() const noexcept {
    return impl_->stats();
}
void IdempotencyJournal::reconcile_fence() {
    impl_->reconcile_pending_fence();
}
OperationId IdempotencyJournal::checkpoint() {
    return impl_->checkpoint();
}
void IdempotencyJournal::prepare(const JournalPreparedOperation& operation) {
    impl_->prepare(operation);
}
JournalCompletedOperation IdempotencyJournal::complete(
    const OperationId operation_id,
    const std::uint64_t commit_sequence) {
    return impl_->complete(operation_id, commit_sequence);
}
void IdempotencyJournal::abort(const OperationId operation_id) {
    impl_->abort(operation_id);
}

PayloadDigest mutation_digest(
    const MutationKind kind,
    const std::uint32_t dimension,
    const DType dtype,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    throwdb::detail::Sha256 hash;
    constexpr std::array<std::byte, 24> domain{
        std::byte{'T'}, std::byte{'h'}, std::byte{'r'}, std::byte{'o'},
        std::byte{'w'}, std::byte{'D'}, std::byte{'B'}, std::byte{'-'},
        std::byte{'R'}, std::byte{'e'}, std::byte{'q'}, std::byte{'u'},
        std::byte{'e'}, std::byte{'s'}, std::byte{'t'}, std::byte{'-'},
        std::byte{'V'}, std::byte{'1'}, std::byte{0}, std::byte{0},
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};
    hash.update(domain);
    std::array<std::byte, 32> header{};
    put_u32(header, 0, static_cast<std::uint32_t>(kind));
    put_u32(header, 4, dimension);
    put_u32(header, 8, static_cast<std::uint32_t>(dtype));
    put_u64(header, 16, ids.size());
    put_u64(header, 24, embeddings.size());
    hash.update(header);
    std::array<std::byte, 8> encoded_id{};
    for (const auto id : ids) {
        put_u64(encoded_id, 0, id);
        hash.update(encoded_id);
    }
    hash.update(embeddings);
    return hash.finish();
}

}  // namespace throwdb::server::detail
