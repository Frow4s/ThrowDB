#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include "throwdb/native_collection.hpp"

namespace throwdb::server {

struct OperationId {
    std::uint64_t high = 0;
    std::uint64_t low = 0;

    [[nodiscard]] bool empty() const noexcept { return high == 0 && low == 0; }
    bool operator==(const OperationId&) const = default;
};

enum class MutationKind : std::uint32_t {
    Insert = 1,
    Update = 2,
    Upsert = 3,
    Delete = 4,
};

struct CollectionDescription {
    std::uint32_t format_version = 0;
    std::uint32_t dimension = 0;
    DType dtype = DType::Float16;
    std::uint64_t record_size = 0;
    std::uint64_t count = 0;
    bool writable = false;
    bool durable_idempotency = false;
};

struct GetResult {
    std::uint32_t dimension = 0;
    DType dtype = DType::Float16;
    std::uint64_t row_count = 0;
    std::vector<std::byte> embeddings;
};

struct MutationResult {
    std::uint64_t commit_sequence = 0;
    std::uint64_t row_count = 0;
    std::vector<std::uint8_t> deleted;
    bool replayed = false;
};

struct EngineOptions {
    bool durable_idempotency = true;
};

class IdempotencyConflictError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PendingOperationError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The sole transport-neutral owner of remote operation semantics. TDBP and
// gRPC adapters must share one instance so mutation ordering, preconditions,
// durable idempotency, and returned commit sequences cannot diverge.
class SingleCollectionEngine {
public:
    SingleCollectionEngine(
        NativeCollection& collection,
        std::filesystem::path storage_path = {},
        EngineOptions options = {});
    ~SingleCollectionEngine();

    SingleCollectionEngine(const SingleCollectionEngine&) = delete;
    SingleCollectionEngine& operator=(const SingleCollectionEngine&) = delete;

    [[nodiscard]] CollectionDescription describe() const;
    [[nodiscard]] GetResult get(
        std::span<const std::uint64_t> ids,
        std::uint32_t prefix_dimension = 0) const;
    [[nodiscard]] MutationResult mutate(
        MutationKind kind,
        OperationId operation_id,
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings = {});
    [[nodiscard]] std::unique_ptr<NativeIdScan> open_scan() const;

    [[nodiscard]] NativeCollection& collection() noexcept;
    [[nodiscard]] const NativeCollection& collection() const noexcept;
    [[nodiscard]] bool durable_idempotency() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server
