#include "idempotency_journal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
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
constexpr std::uint32_t kJournalVersion = 1;
constexpr std::uint32_t kJournalHeaderBytes = 4096;
constexpr std::uint32_t kRecordHeaderBytes = 192;
constexpr std::uint64_t kMaximumRecordBytes = 64ULL << 20;
constexpr std::size_t kJournalCrcOffset = 32;
constexpr std::size_t kRecordCrcOffset = 144;

enum class RecordKind : std::uint32_t {
    Prepare = 1,
    Complete = 2,
    AbortNotApplied = 3,
};

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
    [[nodiscard]] int get() const noexcept { return value_; }

private:
    int value_ = -1;
};

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
        const std::uint64_t valid_end)
        : path_(std::move(path)),
          fd_(std::move(fd)),
          collection_id_low_(collection_id_low),
          collection_id_high_(collection_id_high),
          valid_end_(valid_end) {}

    void recover() {
        const auto size = file_size(fd_.get(), path_);
        if (size < kJournalHeaderBytes) {
            throw CorruptionError("idempotency journal header is truncated");
        }
        std::array<std::byte, kJournalHeaderBytes> header{};
        pread_exact(fd_.get(), header, 0, path_);
        if (!same_magic(header, kJournalMagic)
            || get_u32(header, 8) != kJournalVersion
            || get_u32(header, 12) != kJournalHeaderBytes
            || get_u64(header, 16) != collection_id_low_
            || get_u64(header, 24) != collection_id_high_
            || get_u32(header, kJournalCrcOffset)
                   != crc32c(header, kJournalCrcOffset, 4)) {
            throw CorruptionError(
                "idempotency journal header is corrupt or belongs to another collection");
        }
        for (std::size_t offset = 36; offset != header.size(); ++offset) {
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
                || get_u32(encoded_header, 8) != kJournalVersion
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

    void prepare(const JournalPreparedOperation& operation) {
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
        append(RecordKind::Prepare, operation, 0);
        aborted_.erase(operation.operation_id);
        pending_ = operation;
    }

    JournalCompletedOperation complete(
        const OperationId operation_id,
        const std::uint64_t commit_sequence) {
        if (!pending_.has_value() || pending_->operation_id != operation_id) {
            throw IdempotencyConflictError(
                "operation_id has no matching PREPARE record");
        }
        if (commit_sequence != expected_commit_sequence(*pending_)) {
            throw IdempotencyConflictError(
                "operation_id COMPLETE sequence does not match PREPARE");
        }
        append(RecordKind::Complete, *pending_, commit_sequence);
        JournalCompletedOperation result{*pending_, commit_sequence};
        completed_.insert_or_assign(operation_id, result);
        pending_.reset();
        return result;
    }

    void abort(const OperationId operation_id) {
        if (!pending_.has_value() || pending_->operation_id != operation_id) {
            throw IdempotencyConflictError(
                "operation_id has no matching PREPARE record");
        }
        append(RecordKind::AbortNotApplied, *pending_, 0);
        aborted_.insert_or_assign(operation_id, *pending_);
        pending_.reset();
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
        put_u32(record, 8, kJournalVersion);
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
        pwrite_all(fd_.get(), record, valid_end_, path_);
        sync_fd(fd_.get(), path_, "idempotency journal record");
        valid_end_ += record.size();
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
};

std::unique_ptr<IdempotencyJournal> IdempotencyJournal::open(
    const std::filesystem::path& collection_path,
    const std::uint64_t collection_id_low,
    const std::uint64_t collection_id_high) {
    if (collection_path.empty()
        || (collection_id_low == 0 && collection_id_high == 0)) {
        throw InvalidArgumentError(
            "durable idempotency requires a collection path and identity");
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
        std::array<std::byte, kJournalHeaderBytes> header{};
        std::memcpy(header.data(), kJournalMagic.data(), kJournalMagic.size());
        put_u32(header, 8, kJournalVersion);
        put_u32(header, 12, kJournalHeaderBytes);
        put_u64(header, 16, collection_id_low);
        put_u64(header, 24, collection_id_high);
        put_u32(header, kJournalCrcOffset, crc32c(header, kJournalCrcOffset, 4));
        pwrite_all(fd.get(), header, 0, path);
        sync_fd(fd.get(), path, "idempotency journal header");
        sync_directory(directory);
    }

    auto impl = std::make_unique<Impl>(
        path,
        std::move(fd),
        collection_id_low,
        collection_id_high,
        kJournalHeaderBytes);
    impl->recover();
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
