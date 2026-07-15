#include "wal_file.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sys/mman.h>

#include "internal.hpp"

namespace throwdb::detail {
namespace {

class ReadOnlyMapping {
public:
    ReadOnlyMapping(
        const int fd,
        const std::uint64_t size,
        const std::filesystem::path& path)
        : size_(size) {
        if (size == 0 || size > std::numeric_limits<std::size_t>::max()) {
            throw CorruptionError(
                "cannot map WAL with invalid size: '" + path.string() + "'");
        }
        address_ = ::mmap(
            nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_SHARED,
            fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw StorageError(system_error_message("mmap failed for", path));
        }
    }

    ~ReadOnlyMapping() {
        if (address_ != nullptr) {
            static_cast<void>(
                ::munmap(address_, static_cast<std::size_t>(size_)));
        }
    }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {
            static_cast<const std::byte*>(address_),
            static_cast<std::size_t>(size_)};
    }

private:
    void* address_{nullptr};
    std::uint64_t size_{0};
};

}  // namespace

void create_wal_file(
    const std::filesystem::path& path,
    const WalHeader& header) {
    auto file = create_exclusive(path);
    const auto encoded = encode_wal_header(header);
    write_all(file.get(), encoded.data(), encoded.size(), path);
    sync_file(file.get(), path);
}

WalScanResult scan_wal_file(
    const int fd,
    const std::filesystem::path& path,
    const ExpectedWalIdentity& expected,
    const WalScanMode mode,
    const WalRecoveryVisitor& visitor) {
    const std::uint64_t size = file_size(fd, path);
    ReadOnlyMapping mapping(fd, size, path);
    return scan_wal(mapping.bytes(), expected, mode, visitor);
}

WalMutationFingerprint fingerprint_recovered_wal_batch(
    const int fd,
    const std::filesystem::path& path,
    const RecoveredWalBatch& batch,
    const std::uint64_t record_size) {
    WalMutationFingerprintBuilder builder(record_size, batch.operations.size());
    std::vector<std::byte> value(static_cast<std::size_t>(record_size));
    for (const RecoveredWalOperation& operation : batch.operations) {
        if (operation.kind == WalOperationKind::Put) {
            if (operation.payload_size != record_size) {
                throw CorruptionError(
                    "recovered WAL PUT has unexpected payload size");
            }
            pread_exact(
                fd,
                value.data(),
                value.size(),
                operation.payload_offset,
                path);
            builder.add(operation.kind, operation.id, value);
        } else {
            builder.add(operation.kind, operation.id, {});
        }
    }
    return builder.finish();
}

std::uint64_t append_wal_transaction_durable(
    const int fd,
    const std::filesystem::path& path,
    const std::uint64_t batch_offset,
    const EncodedWalTransaction& transaction) {
    if (transaction.batch_size < kWalBatchHeaderSize
        || transaction.batch_size > transaction.bytes.size()
        || transaction.bytes.size() - transaction.batch_size
               != kWalCommitMarkerSize) {
        throw InvalidArgumentError(
            "encoded WAL transaction has invalid BATCH/COMMIT geometry");
    }
    if (batch_offset > std::numeric_limits<std::uint64_t>::max()
                           - transaction.bytes.size()) {
        throw InvalidArgumentError("WAL append offset overflows uint64");
    }

    pwrite_all(
        fd, transaction.bytes.data(),
        static_cast<std::size_t>(transaction.batch_size), batch_offset, path);
    sync_file(fd, path);

    const auto* commit = transaction.bytes.data()
                         + static_cast<std::size_t>(transaction.batch_size);
    pwrite_all(
        fd, commit, static_cast<std::size_t>(kWalCommitMarkerSize),
        batch_offset + transaction.batch_size, path);
    sync_file(fd, path);
    return batch_offset + transaction.bytes.size();
}

}  // namespace throwdb::detail
