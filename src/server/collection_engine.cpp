#include "throwdb/server/collection_engine.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_set>
#include <utility>

#include "idempotency_journal.hpp"
#include "throwdb/errors.hpp"

namespace throwdb::server {
namespace {

void validate_distinct_ids(const std::span<const std::uint64_t> ids) {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(ids.size());
    for (const auto id : ids) {
        if (!seen.insert(id).second) {
            throw DuplicateIdError(id);
        }
    }
}

[[nodiscard]] std::uint64_t checked_payload_size(
    const std::size_t count,
    const std::uint64_t record_size) {
    if (count > std::numeric_limits<std::uint64_t>::max()
                    / record_size) {
        throw InvalidArgumentError("mutation payload size overflows uint64");
    }
    return static_cast<std::uint64_t>(count) * record_size;
}

[[nodiscard]] MutationResult public_result(
    const detail::JournalCompletedOperation& operation,
    const bool replayed) {
    return MutationResult{
        operation.commit_sequence,
        operation.prepared.row_count,
        operation.prepared.deleted,
        replayed};
}

}  // namespace

class SingleCollectionEngine::Impl {
public:
    Impl(
        NativeCollection& collection,
        const std::filesystem::path& storage_path,
        const EngineOptions options)
        : collection_(collection) {
        if (options.durable_idempotency && collection_.writable()
            && !storage_path.empty()) {
            const auto identity = collection_.identity();
            journal_ = detail::IdempotencyJournal::open(
                storage_path,
                identity.collection_id_low,
                identity.collection_id_high);
            resolve_recovered_pending(identity.committed_sequence);
        }
    }

    [[nodiscard]] CollectionDescription describe() const {
        return CollectionDescription{
            collection_.format_version(),
            collection_.dimension(),
            collection_.dtype(),
            collection_.record_size(),
            collection_.count(),
            collection_.writable(),
            journal_ != nullptr};
    }

    [[nodiscard]] GetResult get(
        const std::span<const std::uint64_t> ids,
        const std::uint32_t requested_prefix) const {
        const auto dimension = requested_prefix == 0
                                   ? collection_.dimension()
                                   : requested_prefix;
        if (dimension == 0 || dimension > collection_.dimension()) {
            throw InvalidArgumentError("prefix dimension is out of range");
        }
        auto full = collection_.get_bytes(ids);
        GetResult result;
        result.dimension = dimension;
        result.dtype = collection_.dtype();
        result.row_count = ids.size();
        if (dimension == collection_.dimension()) {
            result.embeddings = std::move(full);
            return result;
        }

        const auto item_size = collection_.record_size()
                               / collection_.dimension();
        const auto prefix_bytes = static_cast<std::size_t>(
            static_cast<std::uint64_t>(dimension) * item_size);
        const auto full_bytes = static_cast<std::size_t>(collection_.record_size());
        if (ids.size() > std::numeric_limits<std::size_t>::max() / prefix_bytes) {
            throw InvalidArgumentError("prefix result does not fit memory");
        }
        result.embeddings.resize(ids.size() * prefix_bytes);
        for (std::size_t row = 0; row != ids.size(); ++row) {
            std::memcpy(
                result.embeddings.data() + row * prefix_bytes,
                full.data() + row * full_bytes,
                prefix_bytes);
        }
        return result;
    }

    [[nodiscard]] MutationResult mutate(
        const MutationKind kind,
        const OperationId operation_id,
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) {
        std::lock_guard mutation(mutation_mutex_);
        resolve_live_pending();
        validate_request(kind, ids, embeddings);
        const auto request_digest = detail::mutation_digest(
            kind,
            collection_.dimension(),
            collection_.dtype(),
            ids,
            embeddings);

        if (journal_ != nullptr && !operation_id.empty()) {
            if (const auto* completed = journal_->completed(operation_id)) {
                if (completed->prepared.request_digest != request_digest) {
                    throw IdempotencyConflictError(
                        "operation_id was already committed with another payload");
                }
                return public_result(*completed, true);
            }
            if (const auto* aborted = journal_->aborted(operation_id);
                aborted != nullptr
                && aborted->request_digest != request_digest) {
                throw IdempotencyConflictError(
                    "operation_id was already used with another payload");
            }
        }

        const auto existed = collection_.contains(ids);
        for (std::size_t index = 0; index != ids.size(); ++index) {
            if (kind == MutationKind::Insert && existed[index] != 0) {
                throw ExistingIdError(ids[index]);
            }
            if (kind == MutationKind::Update && existed[index] == 0) {
                throw MissingIdError(ids[index]);
            }
        }

        std::vector<std::uint64_t> committed_delete_ids;
        if (kind == MutationKind::Delete) {
            committed_delete_ids.reserve(ids.size());
            for (std::size_t index = 0; index != ids.size(); ++index) {
                if (existed[index] != 0) {
                    committed_delete_ids.push_back(ids[index]);
                }
            }
        }
        const auto wal_fingerprint = NativeCollection::fingerprint_mutation(
            kind == MutationKind::Delete
                ? NativeMutationOperation::Delete
                : NativeMutationOperation::Put,
            kind == MutationKind::Delete
                ? std::span<const std::uint64_t>(committed_delete_ids)
                : ids,
            kind == MutationKind::Delete
                ? std::span<const std::byte>{}
                : embeddings,
            collection_.record_size());

        const auto before = collection_.identity().committed_sequence;
        if (before == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("mutation commit sequence is exhausted");
        }
        detail::JournalPreparedOperation prepared;
        prepared.operation_id = operation_id;
        prepared.request_digest = request_digest;
        prepared.wal_fingerprint = wal_fingerprint;
        prepared.kind = kind;
        prepared.expected_sequence = before + 1;
        prepared.row_count = ids.size();
        if (kind == MutationKind::Delete) {
            prepared.deleted = existed;
        }
        const bool journalled = journal_ != nullptr && !operation_id.empty();
        if (journalled) {
            journal_->prepare(prepared);
        }

        std::vector<std::uint8_t> actual_deleted;
        switch (kind) {
            case MutationKind::Insert:
                collection_.insert(ids, embeddings);
                break;
            case MutationKind::Update:
                collection_.update(ids, embeddings);
                break;
            case MutationKind::Upsert:
                collection_.upsert(ids, embeddings);
                break;
            case MutationKind::Delete:
                actual_deleted = collection_.erase(ids);
                if (actual_deleted != existed) {
                    throw StorageError(
                        "delete preflight changed before the sole-writer commit");
                }
                break;
        }

        const auto after = collection_.identity().committed_sequence;
        const bool committed = kind == MutationKind::Delete
                                   ? !committed_delete_ids.empty()
                                   : !ids.empty();
        const auto expected_after = committed ? prepared.expected_sequence : before;
        if (after != expected_after) {
            throw StorageError(
                "mutation commit sequence changed outside the server lane");
        }

        if (journalled) {
            return public_result(journal_->complete(operation_id, after), false);
        }
        return MutationResult{
            after,
            static_cast<std::uint64_t>(ids.size()),
            std::move(actual_deleted),
            false};
    }

    [[nodiscard]] std::unique_ptr<NativeIdScan> open_scan() const {
        return collection_.scan_ids();
    }

    [[nodiscard]] bool durable_idempotency() const noexcept {
        return journal_ != nullptr;
    }

    [[nodiscard]] NativeCollection& collection() noexcept {
        return collection_;
    }
    [[nodiscard]] const NativeCollection& collection() const noexcept {
        return collection_;
    }

private:
    void validate_request(
        const MutationKind kind,
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) const {
        if (kind != MutationKind::Insert && kind != MutationKind::Update
            && kind != MutationKind::Upsert && kind != MutationKind::Delete) {
            throw InvalidArgumentError("unknown mutation kind");
        }
        validate_distinct_ids(ids);
        if (kind == MutationKind::Delete) {
            if (!embeddings.empty()) {
                throw InvalidArgumentError(
                    "delete request unexpectedly contains embeddings");
            }
            return;
        }
        const auto expected = checked_payload_size(
            ids.size(), collection_.record_size());
        if (expected > std::numeric_limits<std::size_t>::max()
            || embeddings.size() != static_cast<std::size_t>(expected)) {
            throw InvalidArgumentError(
                "mutation embedding bytes do not match collection schema");
        }
    }

    void resolve_recovered_pending(const std::uint64_t current_sequence) {
        if (journal_ == nullptr || journal_->pending() == nullptr) {
            return;
        }
        const auto pending = *journal_->pending();
        if (current_sequence < pending.expected_sequence) {
            journal_->abort(pending.operation_id);
            return;
        }
        const auto verification = collection_.verify_committed_mutation(
            pending.expected_sequence, pending.wal_fingerprint);
        if (verification == NativeMutationVerification::Match) {
            static_cast<void>(journal_->complete(
                pending.operation_id, pending.expected_sequence));
            return;
        }
        if (verification == NativeMutationVerification::Mismatch) {
            throw StorageError(
                "ambiguous idempotency PREPARE sequence belongs to another mutation; "
                "refusing to start");
        }
        throw StorageError(
            "ambiguous idempotency PREPARE cannot be verified because its WAL "
            "transaction is no longer retained; refusing to start");
    }

    void resolve_live_pending() {
        if (journal_ == nullptr || journal_->pending() == nullptr) {
            return;
        }
        const auto sequence = collection_.identity().committed_sequence;
        resolve_recovered_pending(sequence);
        if (journal_->pending() != nullptr) {
            throw PendingOperationError(
                "an ambiguous operation must be resolved before another mutation");
        }
    }

    NativeCollection& collection_;
    std::unique_ptr<detail::IdempotencyJournal> journal_;
    std::mutex mutation_mutex_;
};

SingleCollectionEngine::SingleCollectionEngine(
    NativeCollection& collection,
    std::filesystem::path storage_path,
    const EngineOptions options)
    : impl_(std::make_unique<Impl>(collection, storage_path, options)) {}
SingleCollectionEngine::~SingleCollectionEngine() = default;

CollectionDescription SingleCollectionEngine::describe() const {
    return impl_->describe();
}
GetResult SingleCollectionEngine::get(
    const std::span<const std::uint64_t> ids,
    const std::uint32_t prefix_dimension) const {
    return impl_->get(ids, prefix_dimension);
}
MutationResult SingleCollectionEngine::mutate(
    const MutationKind kind,
    const OperationId operation_id,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    return impl_->mutate(kind, operation_id, ids, embeddings);
}
std::unique_ptr<NativeIdScan> SingleCollectionEngine::open_scan() const {
    return impl_->open_scan();
}
NativeCollection& SingleCollectionEngine::collection() noexcept {
    return impl_->collection();
}
const NativeCollection& SingleCollectionEngine::collection() const noexcept {
    return impl_->collection();
}
bool SingleCollectionEngine::durable_idempotency() const noexcept {
    return impl_->durable_idempotency();
}

}  // namespace throwdb::server
