#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "throwdb/server/collection_engine.hpp"

namespace throwdb::server::detail {

using PayloadDigest = std::array<std::byte, 32>;

struct JournalPreparedOperation {
    OperationId operation_id;
    PayloadDigest request_digest{};
    NativeMutationFingerprint wal_fingerprint{};
    MutationKind kind{MutationKind::Insert};
    std::uint64_t expected_sequence = 0;
    std::uint64_t row_count = 0;
    std::vector<std::uint8_t> deleted;
};

struct JournalCompletedOperation {
    JournalPreparedOperation prepared;
    std::uint64_t commit_sequence = 0;
};

struct OperationIdHash {
    [[nodiscard]] std::size_t operator()(OperationId value) const noexcept;
};

class IdempotencyJournal {
public:
    static std::unique_ptr<IdempotencyJournal> open(
        const std::filesystem::path& collection_path,
        std::uint64_t collection_id_low,
        std::uint64_t collection_id_high);

    ~IdempotencyJournal();
    IdempotencyJournal(const IdempotencyJournal&) = delete;
    IdempotencyJournal& operator=(const IdempotencyJournal&) = delete;

    [[nodiscard]] const JournalCompletedOperation* completed(
        OperationId operation_id) const noexcept;
    [[nodiscard]] const JournalPreparedOperation* pending() const noexcept;
    [[nodiscard]] const JournalPreparedOperation* aborted(
        OperationId operation_id) const noexcept;

    void prepare(const JournalPreparedOperation& operation);
    JournalCompletedOperation complete(
        OperationId operation_id,
        std::uint64_t commit_sequence);
    void abort(OperationId operation_id);

private:
    class Impl;
    explicit IdempotencyJournal(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] PayloadDigest mutation_digest(
    MutationKind kind,
    std::uint32_t dimension,
    DType dtype,
    std::span<const std::uint64_t> ids,
    std::span<const std::byte> embeddings);

}  // namespace throwdb::server::detail
