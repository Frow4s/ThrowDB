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

enum class IdempotencyJournalTestHookPoint : std::uint8_t {
    PrepareSynced = 1,
    CompleteSynced = 2,
    AbortSynced = 3,
    FenceRemove = 4,
    CheckpointPublished = 5,
};

using IdempotencyJournalTestHook = void (*)(
    IdempotencyJournalTestHookPoint point);

// Internal deterministic fault seam. Production code leaves the hook null.
void set_idempotency_journal_test_hook(
    IdempotencyJournalTestHook hook) noexcept;

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

struct IdempotencyJournalLimits {
    std::uint64_t max_bytes = 4ULL << 30;
    std::uint64_t max_operations = 1'000'000;
};

struct IdempotencyJournalStats {
    bool pending = false;
    std::uint64_t bytes = 0;
    std::uint64_t completed_operations = 0;
    std::uint64_t aborted_operations = 0;
    std::uint64_t max_bytes = 0;
    std::uint64_t max_operations = 0;
    OperationId domain;
    std::uint64_t generation = 0;
};

class IdempotencyJournal {
public:
    static std::unique_ptr<IdempotencyJournal> open(
        const std::filesystem::path& collection_path,
        std::uint64_t collection_id_low,
        std::uint64_t collection_id_high,
        IdempotencyJournalLimits limits = {});

    ~IdempotencyJournal();
    IdempotencyJournal(const IdempotencyJournal&) = delete;
    IdempotencyJournal& operator=(const IdempotencyJournal&) = delete;

    [[nodiscard]] const JournalCompletedOperation* completed(
        OperationId operation_id) const noexcept;
    [[nodiscard]] const JournalPreparedOperation* pending() const noexcept;
    [[nodiscard]] const JournalPreparedOperation* aborted(
        OperationId operation_id) const noexcept;
    [[nodiscard]] OperationId domain() const noexcept;
    [[nodiscard]] IdempotencyJournalStats stats() const noexcept;
    void reconcile_fence();

    // Atomically starts a fresh replay domain and retires the complete prior
    // operation history. Requests carrying the old domain are rejected by
    // the transport before mutation, so an expired operation ID can never be
    // applied a second time merely because its detailed result was retired.
    // A checkpoint is legal only with no PREPARE in flight.
    [[nodiscard]] OperationId checkpoint();

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
