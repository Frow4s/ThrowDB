#include "throwdb/native_collection.hpp"

#include "file_lock.hpp"
#include "internal.hpp"
#include "mutation_catalog.hpp"
#include "mutation_delta.hpp"
#include "mutation_types.hpp"
#include "mutation_wal.hpp"
#include "native_format.hpp"
#include "physical_reader.hpp"
#include "wal_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace throwdb {
namespace {

constexpr std::uint32_t kDeltaPublicNativeFormatVersion = 6;
constexpr std::uint32_t kCatalogPublicNativeFormatVersion = 5;
constexpr std::uint32_t kCurrentPublicNativeFormatVersion = 4;
constexpr std::uint32_t kLegacyPublicNativeFormatVersion = 3;
using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(
    const SteadyClock::time_point started) noexcept {
    return std::chrono::duration<double>(SteadyClock::now() - started).count();
}

[[nodiscard]] detail::NativeManifest read_manifest_timed(
    const std::filesystem::path& path,
    NativeOpenStats& stats) {
    const auto started = SteadyClock::now();
    detail::NativeManifest manifest = detail::read_native_manifest(path);
    stats.manifest_seconds = elapsed_seconds(started);
    return manifest;
}

enum class PutMode {
    Insert,
    Update,
    Upsert,
};

using OverlayValue = std::optional<detail::VersionedMutationValue>;

[[nodiscard]] detail::DTypeCode dtype_code(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return detail::DTypeCode::Float16;
        case DType::Float32:
            return detail::DTypeCode::Float32;
    }
    throw InvalidArgumentError("unsupported embedding dtype");
}

[[nodiscard]] DType public_dtype(const detail::DTypeCode dtype) noexcept {
    return dtype == detail::DTypeCode::Float32
               ? DType::Float32
               : DType::Float16;
}

[[nodiscard]] detail::Uuid128 make_uuid() {
    const detail::SnapshotId value = detail::make_snapshot_id();
    return detail::Uuid128{value.low, value.high};
}

[[nodiscard]] std::uint64_t current_unix_time_ns() noexcept {
    const auto raw = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return raw > 0 ? static_cast<std::uint64_t>(raw) : 0;
}

[[nodiscard]] std::filesystem::path writer_lock_path(
    const std::filesystem::path& path) {
    return path / "locks" / "writer.lock";
}

[[nodiscard]] std::filesystem::path commit_lock_path(
    const std::filesystem::path& path) {
    return path / "locks" / "commit.lock";
}

void create_native_store_directories(const std::filesystem::path& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw StorageError("cannot inspect storage path: " + error.message());
    }
    if (exists) {
        if (!std::filesystem::is_directory(path, error) || error) {
            throw InvalidArgumentError(
                "storage path exists and is not a directory");
        }
        if (!std::filesystem::is_empty(path, error) || error) {
            throw InvalidArgumentError("storage path must be absent or empty");
        }
    } else if (!std::filesystem::create_directories(path, error) || error) {
        throw StorageError(
            "cannot create storage directory: " + error.message());
    }

    for (const char* child : {
             "data", "ids", "index", "wal", "catalogs", "deltas", "locks"}) {
        if (!std::filesystem::create_directory(path / child, error) || error) {
            throw StorageError(
                "cannot create storage subdirectory '" + std::string(child)
                + "': " + error.message());
        }
    }

    // Publish the lock inodes before the manifest. Read-only opens therefore
    // never need to create metadata inside an otherwise immutable collection.
    {
        auto writer = detail::AdvisoryLockFile::create(writer_lock_path(path));
        auto commit = detail::AdvisoryLockFile::create(commit_lock_path(path));
        static_cast<void>(writer);
        static_cast<void>(commit);
    }
    detail::sync_directory(path / "locks");
    detail::sync_directory(path);
}

void validate_embedding_batch(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint64_t record_size) {
    const std::uint64_t expected = detail::checked_multiply(
        ids.size(), record_size, "embedding batch");
    if (expected > std::numeric_limits<std::size_t>::max()
        || embeddings.size() != static_cast<std::size_t>(expected)) {
        throw InvalidArgumentError(
            "embedding byte count does not equal ids.size() * record_size");
    }
}

void validate_distinct_ids(const std::span<const std::uint64_t> ids) {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(ids.size());
    for (const std::uint64_t id : ids) {
        if (!seen.insert(id).second) {
            throw DuplicateIdError(id);
        }
    }
}

[[nodiscard]] detail::WalHeader wal_header_for(
    const detail::NativeManifest& manifest,
    const detail::Uuid128 wal_id = {},
    const detail::Uuid128 predecessor_wal_id = {},
    const std::uint64_t previous_sequence = 0) {
    detail::WalHeader header;
    header.collection_id = manifest.collection_id;
    header.schema_id = manifest.schema_id;
    header.wal_id = wal_id.empty() ? manifest.active_wal_id : wal_id;
    header.predecessor_wal_id = predecessor_wal_id;
    header.previous_sequence = previous_sequence;
    header.first_sequence = previous_sequence + 1;
    header.record_size = manifest.record_size;
    header.dimension = manifest.schema.dimension;
    header.dtype = manifest.schema.dtype;
    header.item_size = manifest.schema.item_size;
    header.id_type = manifest.schema.id_type;
    header.row_layout = manifest.schema.row_layout;
    header.creation_time_unix_ns = current_unix_time_ns();
    return header;
}

}  // namespace

class NativeCollectionBuilder::Impl {
public:
    Impl(std::filesystem::path path, const NativeBuilderOptions options)
        : path_(std::move(path)),
          options_(options),
          dtype_code_(dtype_code(options.dtype)),
          item_size_(detail::dtype_item_size(dtype_code_)),
          record_size_(detail::checked_multiply(
              options.dimension, item_size_, "embedding record size")),
          records_per_segment_(std::min<std::uint64_t>(
              options.segment_size_bytes / record_size_,
              detail::RowRef::kMaximumRowsPerSegment)),
          segment_payload_capacity_(detail::checked_multiply(
              records_per_segment_, record_size_, "segment payload capacity")),
          collection_id_(make_uuid()),
          schema_id_(make_uuid()),
          wal_id_(make_uuid()) {
        create_native_store_directories(path_);
    }

    void append(
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) {
        ensure_building();
        validate_embedding_batch(ids, embeddings, record_size_);
        if (ids.empty()) {
            return;
        }
        if (ids.size() > std::numeric_limits<std::uint64_t>::max() - count_) {
            throw InvalidArgumentError("record count is too large");
        }

        try {
            std::uint64_t input_row = 0;
            while (input_row != ids.size()) {
                if (!active_data_fd_) {
                    open_next_segment();
                }
                const std::uint64_t available =
                    records_per_segment_ - active_rows_;
                const std::uint64_t rows = std::min<std::uint64_t>(
                    available, ids.size() - input_row);
                const std::uint64_t bytes = detail::checked_multiply(
                    rows, record_size_, "embedding segment write");
                const std::size_t byte_offset = static_cast<std::size_t>(
                    detail::checked_multiply(
                        input_row, record_size_, "embedding input offset"));

                detail::write_all(
                    active_data_fd_.get(), embeddings.data() + byte_offset,
                    static_cast<std::size_t>(bytes), active_data_path_);
                detail::write_all(
                    active_ids_fd_.get(), ids.data() + input_row,
                    static_cast<std::size_t>(rows * sizeof(std::uint64_t)),
                    active_ids_path_);

                input_row += rows;
                active_rows_ += rows;
                count_ += rows;
                if (active_rows_ == records_per_segment_) {
                    close_active_segment();
                }
            }
        } catch (...) {
            poisoned_ = true;
            throw;
        }
    }

    void finish() {
        ensure_owner_process();
        if (finished_) {
            return;
        }
        ensure_building();
        try {
            close_active_segment();

            detail::NativeManifest manifest;
            manifest.collection_id = collection_id_;
            manifest.schema_id = schema_id_;
            manifest.active_wal_id = wal_id_;
            manifest.schema.dtype = dtype_code_;
            manifest.schema.dimension = options_.dimension;
            manifest.base_record_count = count_;
            manifest.segment_payload_capacity = segment_payload_capacity_;
            manifest.required_feature_bits |=
                detail::kNativeRequiredFeatureCatalogV1;
            manifest.required_feature_bits |=
                detail::kNativeRequiredFeatureDeltaV1;
            detail::derive_native_manifest_geometry(manifest);
            if (manifest.base_segment_count != segment_count_
                || manifest.records_per_segment != records_per_segment_) {
                throw StorageError(
                    "native builder segment geometry changed unexpectedly");
            }

            finalize_segment_headers(manifest);
            detail::sync_directory(path_ / "data");
            detail::sync_directory(path_ / "ids");

            detail::build_and_write_native_index_atomic(path_, manifest);

            const detail::WalHeader wal_header = wal_header_for(manifest);
            detail::create_wal_file(
                detail::mutation_wal_path(path_, manifest.active_wal_id),
                wal_header);
            detail::sync_directory(path_ / "wal");

            const detail::MutationCatalog catalog =
                detail::make_initial_mutation_catalog(manifest, make_uuid());
            static_cast<void>(detail::publish_mutation_catalog(
                path_, manifest, catalog));

            // This is the only initial publication point. Everything it names
            // has already reached stable storage.
            detail::write_native_manifest_atomic(path_, manifest);
            finished_ = true;
        } catch (...) {
            poisoned_ = true;
            throw;
        }
    }

    [[nodiscard]] std::uint64_t count() const {
        ensure_owner_process();
        return count_;
    }
    [[nodiscard]] std::uint32_t dimension() const {
        ensure_owner_process();
        return options_.dimension;
    }
    [[nodiscard]] DType dtype() const {
        ensure_owner_process();
        return options_.dtype;
    }
    [[nodiscard]] std::uint32_t format_version() const {
        ensure_owner_process();
        return kDeltaPublicNativeFormatVersion;
    }
    [[nodiscard]] std::uint64_t record_size() const {
        ensure_owner_process();
        return record_size_;
    }
    [[nodiscard]] bool finished() const {
        ensure_owner_process();
        return finished_;
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        active_data_fd_.reset();
        active_ids_fd_.reset();
    }

private:
    void ensure_owner_process() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "native collection builder was inherited across fork; "
                "create a new builder in the child process");
        }
    }

    void ensure_building() const {
        ensure_owner_process();
        if (finished_) {
            throw InvalidArgumentError("cannot modify a finished builder");
        }
        if (poisoned_) {
            throw StorageError("native builder is unusable after a failed write");
        }
    }

    void open_next_segment() {
        if (segment_count_ >= 0xfffffffcULL) {
            throw InvalidArgumentError(
                "base segments exhaust active WAL segment ID space");
        }
        const auto segment_id = static_cast<std::uint32_t>(segment_count_ + 1);
        active_data_path_ =
            detail::native_data_segment_path(path_, segment_id);
        active_ids_path_ =
            detail::native_ids_segment_path(path_, segment_id);
        active_data_fd_ = detail::create_exclusive(active_data_path_);
        active_ids_fd_ = detail::create_exclusive(active_ids_path_);
        std::array<std::byte, detail::kNativeHeaderSize> empty_header{};
        detail::write_all(
            active_data_fd_.get(), empty_header.data(), empty_header.size(),
            active_data_path_);
        detail::write_all(
            active_ids_fd_.get(), empty_header.data(), empty_header.size(),
            active_ids_path_);
        active_rows_ = 0;
        ++segment_count_;
    }

    void close_active_segment() {
        if (!active_data_fd_) {
            return;
        }
        detail::sync_file(active_data_fd_.get(), active_data_path_);
        detail::sync_file(active_ids_fd_.get(), active_ids_path_);
        active_data_fd_.reset();
        active_ids_fd_.reset();
        active_rows_ = 0;
    }

    void finalize_segment_headers(const detail::NativeManifest& manifest) {
        for (std::uint64_t raw_segment = 1;
             raw_segment <= manifest.base_segment_count; ++raw_segment) {
            const auto segment = static_cast<std::uint32_t>(raw_segment);
            const auto data_path =
                detail::native_data_segment_path(path_, segment);
            const auto ids_path =
                detail::native_ids_segment_path(path_, segment);
            auto data_fd = detail::open_read_write(data_path);
            auto ids_fd = detail::open_read_write(ids_path);
            const detail::NativeDataHeader data_header =
                detail::make_native_data_header(manifest, segment);
            const detail::NativeIdsHeader ids_header =
                detail::make_native_ids_header(manifest, segment);
            if (detail::file_size(data_fd.get(), data_path)
                    != data_header.file_size
                || detail::file_size(ids_fd.get(), ids_path)
                       != ids_header.file_size) {
                throw StorageError(
                    "native builder segment payload has unexpected size");
            }
            const auto encoded_data =
                detail::encode_native_data_header(data_header);
            const auto encoded_ids =
                detail::encode_native_ids_header(ids_header);
            detail::pwrite_all(
                data_fd.get(), encoded_data.data(), encoded_data.size(), 0,
                data_path);
            detail::pwrite_all(
                ids_fd.get(), encoded_ids.data(), encoded_ids.size(), 0,
                ids_path);
            detail::sync_file(data_fd.get(), data_path);
            detail::sync_file(ids_fd.get(), ids_path);
        }
    }

    std::filesystem::path path_;
    NativeBuilderOptions options_;
    detail::DTypeCode dtype_code_{detail::DTypeCode::Invalid};
    std::uint32_t item_size_{0};
    std::uint64_t record_size_{0};
    std::uint64_t records_per_segment_{0};
    std::uint64_t segment_payload_capacity_{0};
    detail::Uuid128 collection_id_;
    detail::Uuid128 schema_id_;
    detail::Uuid128 wal_id_;

    detail::FileDescriptor active_data_fd_;
    detail::FileDescriptor active_ids_fd_;
    std::filesystem::path active_data_path_;
    std::filesystem::path active_ids_path_;
    std::uint64_t active_rows_{0};
    std::uint64_t segment_count_{0};
    std::uint64_t count_{0};
    bool finished_{false};
    bool poisoned_{false};
    const pid_t owner_pid_{::getpid()};
};

class NativeCollection::Impl {
public:
    Impl(
        std::filesystem::path path,
        const ReaderOptions& reader_options,
        const bool writable,
        const SteadyClock::time_point open_started)
        : path_(std::move(path)),
          manifest_(read_manifest_timed(path_, open_stats_)),
          physical_reader_(reader_options),
          writer_lock_(detail::AdvisoryLockFile::open_existing(
              writer_lock_path(path_), writable)),
          commit_lock_(detail::AdvisoryLockFile::open_existing(
              commit_lock_path(path_), writable)),
          writable_(writable),
          catalog_managed_(detail::mutation_catalog_managed(manifest_)),
          delta_managed_(detail::mutation_delta_managed(manifest_)),
          live_count_(manifest_.base_record_count) {
        if (writable_) {
            // Fail a competing writer before loading a potentially very large
            // base index into a second process.
            writer_lock_.lock(detail::FileLockMode::Exclusive, true);
        }
        detail::NativeIndexReadStats index_stats;
        auto phase_started = SteadyClock::now();
        index_ = detail::read_native_index(
            path_, manifest_,
            reader_options.verify_native_index
                ? detail::NativeIndexValidationMode::Full
                : detail::NativeIndexValidationMode::ChecksumOrFull,
            &index_stats);
        open_stats_.index_seconds = elapsed_seconds(phase_started);
        open_stats_.index_header_seconds = index_stats.header_seconds;
        open_stats_.index_allocation_seconds = index_stats.allocation_seconds;
        open_stats_.index_controls_seconds = index_stats.controls_seconds;
        open_stats_.index_slots_seconds = index_stats.slots_seconds;
        open_stats_.index_checksum_seconds = index_stats.checksum_seconds;
        open_stats_.index_structural_validation_seconds =
            index_stats.structural_validation_seconds;
        open_stats_.index_file_bytes = manifest_.index_file_size;
        open_stats_.index_body_checksum_present =
            index_stats.body_checksum_present;
        open_stats_.index_body_checksum_verified =
            index_stats.body_checksum_verified;
        open_stats_.index_structural_validation_performed =
            index_stats.structural_validation_performed;

        phase_started = SteadyClock::now();
        open_base_segments();
        open_stats_.base_segments_seconds = elapsed_seconds(phase_started);

        phase_started = SteadyClock::now();
        detail::ScopedAdvisoryLock stable(
            commit_lock_, writable_ ? detail::FileLockMode::Exclusive
                                    : detail::FileLockMode::Shared);
        open_mutation_logs();
        // This flag is monotonic for the lifetime of the current collection
        // implementation. A clean collection can therefore plan reads from
        // its immutable base index without touching state_mutex_ or overlay_.
        // Recovered mutations are fully built before the handle is published.
        mutation_read_path_active_.store(
            !overlay_.empty(), std::memory_order_relaxed);
        open_stats_.wal_replay_seconds = elapsed_seconds(phase_started);
        open_stats_.total_seconds = elapsed_seconds(open_started);
    }

    ~Impl() {
        if (owned_by_current_process() && writer_lock_.locked()) {
            try {
                writer_lock_.unlock();
            } catch (...) {
                // Closing the descriptor releases the lease.
            }
        }
    }

    void get_into(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        get_into_impl<false>(ids, output, nullptr);
    }

    [[nodiscard]] NativeReadStats get_into_profiled(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        NativeReadStats stats;
        get_into_impl<true>(ids, output, &stats);
        return stats;
    }

    [[nodiscard]] std::vector<std::byte> get_bytes(
        const std::span<const std::uint64_t> ids) const {
        ensure_owner_process();
        const std::uint64_t bytes = detail::checked_multiply(
            ids.size(), manifest_.record_size, "get output");
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("get output does not fit address space");
        }
        std::vector<std::byte> output(static_cast<std::size_t>(bytes));
        get_into(ids, output);
        return output;
    }

    void put(
        const PutMode mode,
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) {
        ensure_mutable();
        validate_embedding_batch(ids, embeddings, manifest_.record_size);
        validate_distinct_ids(ids);
        if (ids.empty()) {
            return;
        }

        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        std::vector<bool> existed(ids.size(), false);
        std::uint64_t resulting_count = live_count_.load();
        {
            std::unique_lock lock(state_mutex_);
            for (std::size_t index = 0; index != ids.size(); ++index) {
                existed[index] = logical_lookup_unlocked(ids[index]).has_value();
                if (mode == PutMode::Insert && existed[index]) {
                    throw ExistingIdError(ids[index]);
                }
                if (mode == PutMode::Update && !existed[index]) {
                    throw MissingIdError(ids[index]);
                }
                if (!existed[index]) {
                    if (resulting_count == std::numeric_limits<std::uint64_t>::max()) {
                        throw InvalidArgumentError("live record count overflows uint64");
                    }
                    ++resulting_count;
                }
            }
        }

        if (ids.size() > detail::RowRef::kMaximumRowsPerSegment
                             - active_wal().payload_offsets.size()) {
            if (!catalog_managed_ || !rotate_wal_locked()) {
                throw InvalidArgumentError(
                    "active WAL exceeds the 2^32 PUT-row limit");
            }
        }
        const std::uint64_t original_wal_rows =
            active_wal().payload_offsets.size();
        {
            std::unique_lock lock(state_mutex_);
            prepare_publication_capacity(ids.size(), ids.size());
            for (const std::uint64_t id : ids) {
                prime_overlay_node_unlocked(id);
            }
        }

        std::vector<detail::WalMutation> operations;
        operations.reserve(ids.size());
        for (std::size_t index = 0; index != ids.size(); ++index) {
            operations.push_back(detail::WalMutation{
                ids[index], detail::WalOperationKind::Put,
                embeddings.subspan(
                    index * static_cast<std::size_t>(manifest_.record_size),
                    static_cast<std::size_t>(manifest_.record_size))});
        }
        commit_operations(operations, original_wal_rows, resulting_count);
    }

    [[nodiscard]] std::vector<std::uint8_t> erase(
        const std::span<const std::uint64_t> ids) {
        ensure_mutable();
        validate_distinct_ids(ids);
        std::vector<std::uint8_t> deleted(ids.size(), 0);
        if (ids.empty()) {
            return deleted;
        }

        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        std::vector<detail::WalMutation> operations;
        operations.reserve(ids.size());
        std::uint64_t resulting_count = live_count_.load();
        {
            std::unique_lock lock(state_mutex_);
            for (std::size_t index = 0; index != ids.size(); ++index) {
                if (!logical_lookup_unlocked(ids[index]).has_value()) {
                    continue;
                }
                deleted[index] = 1;
                --resulting_count;
                operations.push_back(detail::WalMutation{
                    ids[index], detail::WalOperationKind::Delete, {}});
            }
            if (!operations.empty()) {
                prepare_publication_capacity(operations.size(), 0);
                for (const detail::WalMutation& operation : operations) {
                    prime_overlay_node_unlocked(operation.id);
                }
            }
        }
        if (operations.empty()) {
            return deleted;
        }
        commit_operations(
            operations, active_wal().payload_offsets.size(), resulting_count);
        return deleted;
    }

    [[nodiscard]] NativeMutationLogStats mutation_log_stats() const {
        ensure_healthy();
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        return mutation_log_stats_locked(false);
    }

    [[nodiscard]] NativeMutationLogStats rotate_wal() {
        ensure_mutable();
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        if (!catalog_managed_) {
            throw InvalidArgumentError(
                "WAL rotation requires a catalog-managed format-5 collection");
        }
        const bool rotated = rotate_wal_locked();
        return mutation_log_stats_locked(rotated);
    }

    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals() {
        ensure_mutable();
        std::unique_lock lifecycle(materialization_mutex_);

        std::shared_ptr<WalResourceSet> source_resources;
        std::vector<detail::MutationWalEntry> selected_wals;
        detail::MutationDeltaBuildSpec build_spec;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            if (!delta_managed_) {
                throw InvalidArgumentError(
                    "WAL materialization requires a delta-managed "
                    "format-6 collection");
            }

            std::size_t frozen_count = 0;
            while (frozen_count < catalog_.wal_entries.size()
                   && catalog_.wal_entries[frozen_count].state
                          == detail::MutationWalState::Frozen) {
                ++frozen_count;
            }
            if (frozen_count == 0) {
                return mutation_log_stats_locked(false);
            }
            if (catalog_.generation
                > std::numeric_limits<std::uint64_t>::max() - 2) {
                throw StorageError(
                    "mutation catalog lacks two generations for materialization");
            }
            if (catalog_.next_segment_id >= detail::SegmentId::kMaximum) {
                throw StorageError("mutation segment ID space is exhausted");
            }

            selected_wals.assign(
                catalog_.wal_entries.begin(),
                catalog_.wal_entries.begin()
                    + static_cast<std::ptrdiff_t>(frozen_count));
            source_resources = wal_resources_;
            build_spec.delta_id = make_uuid();
            build_spec.source_predecessor_wal_id =
                selected_wals.front().predecessor_wal_id;
            build_spec.source_terminal_wal_id =
                selected_wals.back().wal_id;
            build_spec.first_sequence =
                selected_wals.front().first_sequence;
            build_spec.last_sequence =
                selected_wals.back().last_sequence;
            build_spec.segment_id = catalog_.next_segment_id;
            if (frozen_count
                > std::numeric_limits<std::uint32_t>::max()) {
                throw StorageError(
                    "materialized WAL prefix count exceeds uint32");
            }
            build_spec.source_wal_count =
                static_cast<std::uint32_t>(frozen_count);

            detail::MutationCatalog reservation = catalog_;
            reservation.generation = catalog_.generation + 1;
            reservation.catalog_id = make_uuid();
            reservation.previous_catalog_id = catalog_.catalog_id;
            reservation.next_segment_id = build_spec.segment_id + 1;
            try {
                detail::ScopedAdvisoryLock commit(
                    commit_lock_, detail::FileLockMode::Exclusive);
                static_cast<void>(detail::publish_mutation_catalog(
                    path_, manifest_, reservation));
                std::swap(catalog_, reservation);
            } catch (...) {
                poisoned_.store(true, std::memory_order_release);
                throw;
            }
        }

        std::unordered_map<std::uint64_t, detail::VersionedMutationValue>
            collapsed;
        for (std::size_t source_index = 0;
             source_index != selected_wals.size(); ++source_index) {
            const detail::MutationWalEntry& entry =
                selected_wals[source_index];
            const std::shared_ptr<WalSource>& source =
                source_resources->sources[source_index];
            std::uint64_t put_index = 0;
            const detail::ExpectedWalIdentity expected =
                detail::ExpectedWalIdentity::from_trusted_header(
                    source->header);
            const detail::WalScanResult scan = detail::scan_wal_file(
                source->fd.get(), source->path, expected,
                detail::WalScanMode::Frozen,
                [&collapsed, &entry, &put_index](
                    const detail::RecoveredWalBatch& batch) {
                    for (const detail::RecoveredWalOperation& operation :
                         batch.operations) {
                        if (operation.kind
                            == detail::WalOperationKind::Put) {
                            if (put_index
                                >= detail::RowRef::kMaximumRowsPerSegment) {
                                throw CorruptionError(
                                    "frozen WAL PUT count exceeds RowRef capacity");
                            }
                            collapsed.insert_or_assign(
                                operation.id,
                                detail::VersionedMutationValue::put(
                                    batch.sequence,
                                    detail::RowRef::create(
                                        entry.segment_id, put_index)));
                            ++put_index;
                        } else {
                            collapsed.insert_or_assign(
                                operation.id,
                                detail::VersionedMutationValue::tombstone(
                                    batch.sequence));
                        }
                    }
                });
            if (scan.valid_end != entry.sealed_file_size
                || scan.ignored_tail_bytes != 0
                || scan.next_sequence == 0
                || scan.next_sequence - 1 != entry.last_sequence
                || put_index != entry.put_count) {
                throw CorruptionError(
                    "frozen WAL changed during materialization");
            }
        }

        std::vector<detail::MutationDeltaBuildEntry> build_entries;
        build_entries.reserve(collapsed.size());
        for (auto& [id, value] : collapsed) {
            build_entries.push_back(
                detail::MutationDeltaBuildEntry{id, value});
        }
        collapsed.clear();
        collapsed.rehash(0);

        const detail::MutationDeltaDescriptor descriptor =
            detail::write_mutation_delta(
                path_, manifest_, build_spec, build_entries,
                [this, source_resources](
                    const std::span<const detail::RowRef> source_rows,
                    const std::span<std::byte> destination) {
                    std::vector<detail::PhysicalReadRequest> requests;
                    requests.reserve(source_rows.size());
                    for (std::size_t index = 0;
                         index != source_rows.size(); ++index) {
                        requests.push_back(resolve_row_unlocked(
                            source_rows[index], index, *source_resources));
                    }
                    physical_reader_.execute(
                        manifest_.record_size, requests, destination);
                });
        detail::OpenMutationDelta opened = detail::open_mutation_delta(
            path_, manifest_, descriptor);

        std::uint64_t reclaimed_wal_count = 0;
        std::uint64_t reclaimed_wal_bytes = 0;
        bool reclamation_complete = true;
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();

        if (catalog_.materialized_sequence
                != build_spec.first_sequence - 1
            || catalog_.terminal_wal_id
                   != build_spec.source_predecessor_wal_id
            || catalog_.wal_entries.size() <= selected_wals.size()) {
            throw StorageError(
                "mutation catalog changed incompatibly during materialization");
        }
        for (std::size_t index = 0; index != selected_wals.size(); ++index) {
            if (catalog_.wal_entries[index] != selected_wals[index]) {
                throw StorageError(
                    "materialized WAL prefix changed while delta was built");
            }
        }
        if (catalog_.generation
            == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("mutation catalog generation is exhausted");
        }

        detail::MutationCatalog next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        next_catalog.materialized_sequence = descriptor.last_sequence;
        next_catalog.terminal_wal_id = descriptor.source_terminal_wal_id;
        next_catalog.delta_entries.push_back(descriptor);
        next_catalog.wal_entries.erase(
            next_catalog.wal_entries.begin(),
            next_catalog.wal_entries.begin()
                + static_cast<std::ptrdiff_t>(selected_wals.size()));

        auto delta_source = std::make_shared<DeltaSource>();
        delta_source->descriptor = descriptor;
        delta_source->header = opened.data_header;
        delta_source->fd = std::move(opened.data_fd);
        delta_source->path = detail::mutation_delta_data_path(
            path_, descriptor.delta_id);

        auto next_resources = std::make_shared<WalResourceSet>();
        {
            std::shared_lock state(state_mutex_);
            next_resources->delta_sources = wal_resources_->delta_sources;
            next_resources->delta_source_indices =
                wal_resources_->delta_source_indices;
            next_resources->sources.assign(
                wal_resources_->sources.begin()
                    + static_cast<std::ptrdiff_t>(selected_wals.size()),
                wal_resources_->sources.end());
        }
        const std::size_t delta_index =
            next_resources->delta_sources.size();
        next_resources->delta_sources.push_back(std::move(delta_source));
        if (!next_resources->delta_source_indices.emplace(
                 descriptor.segment_id, delta_index).second) {
            throw CorruptionError(
                "materialized delta reuses a mutation segment ID");
        }
        next_resources->source_indices.reserve(
            next_resources->sources.size());
        for (std::size_t index = 0;
             index != next_resources->sources.size(); ++index) {
            if (!next_resources->source_indices.emplace(
                     next_resources->sources[index]->entry.segment_id,
                     index).second) {
                throw CorruptionError(
                    "retained WAL segment ID is duplicated");
            }
        }
        if (next_resources->sources.empty()) {
            throw CorruptionError(
                "materialization removed the active WAL");
        }
        next_resources->active_index = next_resources->sources.size() - 1;

        struct OverlayPatch {
            OverlayValue* destination;
            detail::VersionedMutationValue replacement;
        };
        static_assert(std::is_nothrow_copy_assignable_v<OverlayValue>);
        std::vector<OverlayPatch> overlay_patches;
        overlay_patches.reserve(opened.records.size());
        {
            std::shared_lock state(state_mutex_);
            for (const detail::MutationDeltaVersionRecord& record :
                 opened.records) {
                const auto current = overlay_.find(record.id);
                if (current == overlay_.end()
                    || !current->second.has_value()
                    || current->second->sequence()
                           < record.value.sequence()) {
                    throw CorruptionError(
                        "materialized delta is newer than the live overlay");
                }
                if (current->second->sequence() == record.value.sequence()) {
                    overlay_patches.push_back(
                        OverlayPatch{&current->second, record.value});
                }
            }
        }

        try {
            detail::ScopedAdvisoryLock commit(
                commit_lock_, detail::FileLockMode::Exclusive);
            static_cast<void>(detail::publish_mutation_catalog(
                path_, manifest_, next_catalog));
            {
                std::unique_lock state(state_mutex_);
                for (const OverlayPatch& patch : overlay_patches) {
                    *patch.destination = patch.replacement;
                }
                wal_resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
            }

            for (const detail::MutationWalEntry& entry : selected_wals) {
                std::error_code error;
                const bool removed = std::filesystem::remove(
                    detail::mutation_wal_path(path_, entry.wal_id), error);
                if (error) {
                    reclamation_complete = false;
                } else if (removed) {
                    ++reclaimed_wal_count;
                    if (reclaimed_wal_bytes
                        > std::numeric_limits<std::uint64_t>::max()
                              - entry.sealed_file_size) {
                        reclamation_complete = false;
                    } else {
                        reclaimed_wal_bytes += entry.sealed_file_size;
                    }
                } else {
                    reclamation_complete = false;
                }
            }
            if (reclaimed_wal_count != 0) {
                try {
                    detail::sync_directory(path_ / "wal");
                } catch (...) {
                    reclamation_complete = false;
                }
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        return mutation_log_stats_locked(
            false, true, reclaimed_wal_count, reclaimed_wal_bytes,
            reclamation_complete);
    }

    [[nodiscard]] std::uint64_t count() const {
        ensure_owner_process();
        return live_count_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint32_t dimension() const {
        ensure_owner_process();
        return manifest_.schema.dimension;
    }
    [[nodiscard]] DType dtype() const {
        ensure_owner_process();
        return public_dtype(manifest_.schema.dtype);
    }
    [[nodiscard]] std::uint32_t format_version() const {
        ensure_owner_process();
        if (delta_managed_) {
            return kDeltaPublicNativeFormatVersion;
        }
        if (catalog_managed_) {
            return kCatalogPublicNativeFormatVersion;
        }
        return open_stats_.index_body_checksum_present
                   ? kCurrentPublicNativeFormatVersion
                   : kLegacyPublicNativeFormatVersion;
    }
    [[nodiscard]] std::uint64_t record_size() const {
        ensure_owner_process();
        return manifest_.record_size;
    }
    [[nodiscard]] bool writable() const {
        ensure_owner_process();
        return writable_;
    }
    [[nodiscard]] NativeOpenStats open_stats() const {
        ensure_owner_process();
        return open_stats_;
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        // Close every inherited descriptor reference without explicitly
        // unlocking the parent's shared flock open description. The mutexes
        // and copied thread objects remain intentionally undestroyed below.
        for (const auto& source : wal_resources_->sources) {
            source->fd.reset();
        }
        for (const auto& source : wal_resources_->delta_sources) {
            source->fd.reset();
        }
        for (BaseSegment& segment : base_segments_) {
            segment.fd.reset();
        }
        writer_lock_.abandon_after_fork();
        commit_lock_.abandon_after_fork();
        physical_reader_.abandon_after_fork();
    }

private:
    struct BaseSegment {
        detail::NativeDataHeader header;
        detail::FileDescriptor fd;
    };

    struct WalSource {
        detail::MutationWalEntry entry;
        detail::WalHeader header;
        detail::FileDescriptor fd;
        std::filesystem::path path;
        std::vector<std::uint64_t> payload_offsets;
        std::uint64_t valid_end{detail::kWalFileHeaderSize};
        std::uint64_t next_sequence{1};
    };

    struct DeltaSource {
        detail::MutationDeltaDescriptor descriptor;
        detail::MutationDeltaDataHeader header;
        detail::FileDescriptor fd;
        std::filesystem::path path;
    };

    struct WalResourceSet {
        std::vector<std::shared_ptr<WalSource>> sources;
        std::unordered_map<std::uint32_t, std::size_t> source_indices;
        std::vector<std::shared_ptr<DeltaSource>> delta_sources;
        std::unordered_map<std::uint32_t, std::size_t> delta_source_indices;
        std::size_t active_index{0};
    };

    template <bool Profiled>
    void get_into_impl(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output,
        NativeReadStats* const stats) const {
        SteadyClock::time_point total_started;
        if constexpr (Profiled) {
            total_started = SteadyClock::now();
        }

        ensure_healthy();
        const std::uint64_t expected = detail::checked_multiply(
            ids.size(), manifest_.record_size, "get output");
        if (expected > std::numeric_limits<std::size_t>::max()
            || output.size() != static_cast<std::size_t>(expected)) {
            throw InvalidArgumentError(
                "output byte count does not equal ids.size() * record_size");
        }
        if constexpr (Profiled) {
            stats->requested_rows = ids.size();
        }
        if (ids.empty()) {
            if constexpr (Profiled) {
                stats->zero_delta_fast_path =
                    !mutation_read_path_active_.load(
                        std::memory_order_acquire);
                stats->total_seconds = elapsed_seconds(total_started);
            }
            return;
        }

        SteadyClock::time_point lookup_started;
        if constexpr (Profiled) {
            lookup_started = SteadyClock::now();
        }
        std::vector<detail::PhysicalReadRequest> requests;
        requests.reserve(ids.size());
        // Catalog generations will eventually reclaim retired resources.
        // Retain every resolver used by this captured state until physical I/O
        // has completed, rather than relying only on its integer descriptor.
        std::shared_ptr<WalResourceSet> wal_pin;
        const bool zero_delta_fast_path =
            !mutation_read_path_active_.load(std::memory_order_acquire);
        if constexpr (Profiled) {
            stats->zero_delta_fast_path = zero_delta_fast_path;
        }
        if (zero_delta_fast_path) {
            // index_ and base_segments_ never change after open. If the first
            // mutation is published concurrently after the acquire load, this
            // read is a valid snapshot immediately before that publication and
            // does not race with any mutable object.
            for (std::size_t output_index = 0;
                 output_index != ids.size(); ++output_index) {
                const auto row = detail::native_index_lookup(
                    index_, ids[output_index]);
                if (!row.has_value()) {
                    throw MissingIdError(ids[output_index]);
                }
                requests.push_back(resolve_base_row(
                    *row, static_cast<std::uint64_t>(output_index)));
            }
            if constexpr (Profiled) {
                stats->base_rows = ids.size();
            }
        } else {
            std::shared_lock lock(state_mutex_);
            wal_pin = wal_resources_;
            for (std::size_t output_index = 0;
                 output_index != ids.size(); ++output_index) {
                const auto row = logical_lookup_unlocked(ids[output_index]);
                if (!row.has_value()) {
                    throw MissingIdError(ids[output_index]);
                }
                if constexpr (Profiled) {
                    if (row->segment_id().value()
                        <= manifest_.base_segment_count) {
                        ++stats->base_rows;
                    } else {
                        ++stats->mutation_rows;
                    }
                }
                requests.push_back(resolve_row_unlocked(
                    *row, static_cast<std::uint64_t>(output_index),
                    *wal_pin));
            }
        }
        if constexpr (Profiled) {
            stats->lookup_seconds = elapsed_seconds(lookup_started);
            const detail::PhysicalReadStats physical =
                physical_reader_.execute_profiled(
                    manifest_.record_size, requests, output);
            stats->physical_plan_seconds = physical.plan_seconds;
            stats->physical_backend_seconds = physical.backend_seconds;
            stats->planned_ranges = physical.range_count;
            stats->planned_read_bytes = physical.planned_read_bytes;
            stats->backend_tasks = physical.task_count;
            stats->total_seconds = elapsed_seconds(total_started);
        } else {
            physical_reader_.execute(
                manifest_.record_size, requests, output);
        }
    }

    void ensure_healthy() const {
        ensure_owner_process();
        if (poisoned_.load(std::memory_order_acquire)) {
            throw StorageError(
                "native collection handle requires reopen after a failed WAL write");
        }
    }

    void ensure_owner_process() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "native collection handle was inherited across fork; "
                "close it without use and reopen the collection in the child process");
        }
    }

    void ensure_mutable() const {
        ensure_owner_process();
        if (!writable_) {
            throw InvalidArgumentError(
                "collection was opened read-only; use writable mode for mutations");
        }
        ensure_healthy();
    }

    void open_base_segments() {
        if (manifest_.base_segment_count
            > std::numeric_limits<std::size_t>::max()) {
            throw CorruptionError("base segment list does not fit address space");
        }
        base_segments_.reserve(
            static_cast<std::size_t>(manifest_.base_segment_count));
        for (std::uint64_t raw_segment = 1;
             raw_segment <= manifest_.base_segment_count; ++raw_segment) {
            const auto segment = static_cast<std::uint32_t>(raw_segment);
            const detail::NativeDataHeader header =
                detail::read_native_data_header(path_, manifest_, segment);
            static_cast<void>(
                detail::read_native_ids_header(path_, manifest_, segment));
            const auto data_path =
                detail::native_data_segment_path(path_, segment);
            auto fd = detail::open_read_only(data_path);
            if (detail::file_size(fd.get(), data_path) != header.file_size) {
                throw CorruptionError(
                    "data segment changed while collection was opening");
            }
            base_segments_.push_back(BaseSegment{header, std::move(fd)});
        }
    }

    void open_mutation_logs() {
        wal_resources_ = std::make_shared<WalResourceSet>();
        if (catalog_managed_) {
            catalog_ = detail::read_current_mutation_catalog(path_, manifest_);
            if (!delta_managed_ && !catalog_.delta_entries.empty()) {
                throw CorruptionError(
                    "catalog references deltas without the required feature");
            }
            wal_resources_->delta_sources.reserve(
                catalog_.delta_entries.size());
            wal_resources_->delta_source_indices.reserve(
                catalog_.delta_entries.size());
            for (const detail::MutationDeltaDescriptor& descriptor :
                 catalog_.delta_entries) {
                open_delta_source(descriptor);
            }
            wal_resources_->sources.reserve(catalog_.wal_entries.size());
            wal_resources_->source_indices.reserve(
                catalog_.wal_entries.size());
            for (const detail::MutationWalEntry& entry : catalog_.wal_entries) {
                open_catalog_wal(entry);
            }
        } else {
            detail::MutationWalEntry entry;
            entry.wal_id = manifest_.active_wal_id;
            entry.segment_id = manifest_.active_wal_segment_id;
            entry.state = detail::MutationWalState::Active;
            entry.previous_sequence = 0;
            entry.first_sequence = 1;
            open_wal_source(
                entry, detail::native_active_wal_path(path_), true);
        }

        if (wal_resources_->sources.empty()
            || wal_resources_->sources.back()->entry.state
                   != detail::MutationWalState::Active) {
            throw CorruptionError(
                "mutation log does not contain a final active WAL");
        }
        wal_resources_->active_index = wal_resources_->sources.size() - 1;
        append_offset_ = active_wal().valid_end;
        next_sequence_ = active_wal().next_sequence;
    }

    void open_delta_source(
        const detail::MutationDeltaDescriptor& descriptor) {
        detail::OpenMutationDelta opened = detail::open_mutation_delta(
            path_, manifest_, descriptor);
        auto source = std::make_shared<DeltaSource>();
        source->descriptor = descriptor;
        source->header = opened.data_header;
        source->fd = std::move(opened.data_fd);
        source->path = detail::mutation_delta_data_path(
            path_, descriptor.delta_id);

        const std::size_t source_index =
            wal_resources_->delta_sources.size();
        wal_resources_->delta_sources.push_back(std::move(source));
        if (!wal_resources_->delta_source_indices.emplace(
                 descriptor.segment_id, source_index).second) {
            throw CorruptionError("mutation delta segment ID is duplicated");
        }
        for (const detail::MutationDeltaVersionRecord& record :
             opened.records) {
            apply_recovered_delta(record);
        }
    }

    void open_catalog_wal(const detail::MutationWalEntry& entry) {
        open_wal_source(
            entry, detail::mutation_wal_path(path_, entry.wal_id),
            entry.state == detail::MutationWalState::Active);
    }

    void open_wal_source(
        const detail::MutationWalEntry& entry,
        const std::filesystem::path& wal_path,
        const bool active) {
        auto source = std::make_shared<WalSource>();
        source->entry = entry;
        source->header = wal_header_for(
            manifest_, entry.wal_id, entry.predecessor_wal_id,
            entry.previous_sequence);
        source->path = wal_path;
        source->fd = writable_ && active
                         ? detail::open_read_write(wal_path)
                         : detail::open_read_only(wal_path);

        if (!active) {
            const std::uint64_t actual_size = detail::file_size(
                source->fd.get(), source->path);
            if (actual_size != entry.sealed_file_size) {
                throw CorruptionError(
                    "frozen WAL size does not match its catalog entry");
            }
        }

        const detail::ExpectedWalIdentity expected =
            detail::ExpectedWalIdentity::from_trusted_header(source->header);
        const detail::WalScanResult scan = detail::scan_wal_file(
            source->fd.get(), source->path, expected,
            active ? detail::WalScanMode::Active
                   : detail::WalScanMode::Frozen,
            [this, &source](const detail::RecoveredWalBatch& batch) {
                apply_recovered_batch(*source, batch);
            });
        source->valid_end = scan.valid_end;
        source->next_sequence = scan.next_sequence;

        if (!active) {
            if (scan.ignored_tail_bytes != 0
                || scan.valid_end != entry.sealed_file_size
                || scan.next_sequence == 0
                || scan.next_sequence - 1 != entry.last_sequence
                || source->payload_offsets.size() != entry.put_count) {
                throw CorruptionError(
                    "frozen WAL contents do not match its catalog entry");
            }
        } else if (writable_ && scan.ignored_tail_bytes != 0) {
            detail::truncate_file(
                source->fd.get(), scan.valid_end, source->path);
            detail::sync_file(source->fd.get(), source->path);
        }

        wal_resources_->sources.push_back(std::move(source));
        const std::size_t source_index = wal_resources_->sources.size() - 1;
        if (!wal_resources_->source_indices.emplace(
                 entry.segment_id, source_index).second) {
            throw CorruptionError("mutation WAL segment ID is duplicated");
        }
    }

    [[nodiscard]] WalSource& active_wal() {
        return *wal_resources_->sources.at(wal_resources_->active_index);
    }

    [[nodiscard]] const WalSource& active_wal() const {
        return *wal_resources_->sources.at(wal_resources_->active_index);
    }

    [[nodiscard]] bool base_contains(const std::uint64_t id) const noexcept {
        return detail::native_index_lookup(index_, id).has_value();
    }

    [[nodiscard]] std::optional<detail::RowRef> logical_lookup_unlocked(
        const std::uint64_t id) const {
        const auto overlay = overlay_.find(id);
        if (overlay != overlay_.end() && overlay->second.has_value()) {
            return overlay->second->row_ref();
        }
        return detail::native_index_lookup(index_, id);
    }

    [[nodiscard]] detail::PhysicalReadRequest resolve_base_row(
        const detail::RowRef row,
        const std::uint64_t output_index) const {
        const std::uint32_t segment = row.segment_id().value();
        const std::uint64_t row_index = row.row_index();
        if (segment == 0 || segment > manifest_.base_segment_count) {
            throw CorruptionError(
                "base index RowRef does not resolve to a base segment");
        }
        const BaseSegment& base =
            base_segments_[static_cast<std::size_t>(segment - 1)];
        if (row_index >= base.header.row_count) {
            throw CorruptionError("base RowRef exceeds segment row count");
        }
        return detail::PhysicalReadRequest{
            base.fd.get(),
            base.header.payload_offset + row_index * manifest_.record_size,
            output_index};
    }

    [[nodiscard]] detail::PhysicalReadRequest resolve_row_unlocked(
        const detail::RowRef row,
        const std::uint64_t output_index,
        const WalResourceSet& resources) const {
        const std::uint32_t segment = row.segment_id().value();
        const std::uint64_t row_index = row.row_index();
        if (segment <= manifest_.base_segment_count) {
            return resolve_base_row(row, output_index);
        }
        const auto delta_position =
            resources.delta_source_indices.find(segment);
        if (delta_position != resources.delta_source_indices.end()) {
            const DeltaSource& source =
                *resources.delta_sources[delta_position->second];
            if (row_index >= source.descriptor.put_count) {
                throw CorruptionError(
                    "overlay RowRef exceeds mutation delta row count");
            }
            return detail::PhysicalReadRequest{
                source.fd.get(),
                source.header.payload_offset
                    + row_index * manifest_.record_size,
                output_index};
        }
        if (segment != manifest_.active_wal_segment_id
            && !catalog_managed_) {
            throw CorruptionError("overlay RowRef does not resolve to active WAL");
        }
        const auto source_position = resources.source_indices.find(segment);
        if (source_position == resources.source_indices.end()) {
            throw CorruptionError("overlay RowRef does not resolve to a catalog WAL");
        }
        const WalSource& source =
            *resources.sources[source_position->second];
        if (row_index >= source.payload_offsets.size()) {
            throw CorruptionError("overlay RowRef exceeds WAL PUT-row count");
        }
        return detail::PhysicalReadRequest{
            source.fd.get(),
            source.payload_offsets[static_cast<std::size_t>(row_index)],
            output_index};
    }

    void apply_recovered_delta(
        const detail::MutationDeltaVersionRecord& record) {
        const auto existing = overlay_.find(record.id);
        if (existing != overlay_.end() && existing->second.has_value()
            && existing->second->sequence() >= record.value.sequence()) {
            throw CorruptionError(
                "mutation delta versions are not strictly chronological");
        }
        const bool existed =
            existing != overlay_.end() && existing->second.has_value()
                ? existing->second->is_put()
                : base_contains(record.id);
        overlay_.insert_or_assign(record.id, record.value);
        if (record.value.is_put() && !existed) {
            live_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (record.value.is_tombstone() && existed) {
            live_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    void apply_recovered_batch(
        WalSource& source,
        const detail::RecoveredWalBatch& batch) {
        for (const detail::RecoveredWalOperation& operation : batch.operations) {
            const auto prior = overlay_.find(operation.id);
            if (prior != overlay_.end() && prior->second.has_value()
                && prior->second->sequence() >= batch.sequence) {
                throw CorruptionError(
                    "WAL versions are not strictly chronological");
            }
            const bool existed = [&] {
                const auto overlay = overlay_.find(operation.id);
                if (overlay != overlay_.end() && overlay->second.has_value()) {
                    return overlay->second->is_put();
                }
                return base_contains(operation.id);
            }();
            if (operation.kind == detail::WalOperationKind::Put) {
                if (operation.payload_size != manifest_.record_size) {
                    throw CorruptionError("WAL PUT has unexpected payload size");
                }
                if (source.payload_offsets.size()
                    >= detail::RowRef::kMaximumRowsPerSegment) {
                    throw CorruptionError("WAL contains more than 2^32 PUTs");
                }
                const detail::RowRef row = detail::RowRef::create(
                    source.entry.segment_id,
                    source.payload_offsets.size());
                source.payload_offsets.push_back(operation.payload_offset);
                overlay_.insert_or_assign(
                    operation.id,
                    detail::VersionedMutationValue::put(batch.sequence, row));
                if (!existed) {
                    live_count_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                overlay_.insert_or_assign(
                    operation.id,
                    detail::VersionedMutationValue::tombstone(batch.sequence));
                if (existed) {
                    live_count_.fetch_sub(1, std::memory_order_relaxed);
                }
            }
        }
    }

    [[nodiscard]] NativeMutationLogStats mutation_log_stats_locked(
        const bool rotated,
        const bool materialized = false,
        const std::uint64_t reclaimed_wal_count = 0,
        const std::uint64_t reclaimed_wal_bytes = 0,
        const bool reclamation_complete = true) const {
        NativeMutationLogStats stats;
        stats.catalog_managed = catalog_managed_;
        stats.delta_managed = delta_managed_;
        stats.generation = catalog_managed_ ? catalog_.generation : 0;
        stats.committed_sequence = next_sequence_ - 1;
        stats.materialized_sequence =
            catalog_managed_ ? catalog_.materialized_sequence : 0;
        stats.active_wal_bytes = append_offset_;
        stats.active_wal_put_rows = active_wal().payload_offsets.size();
        stats.rotated = rotated;
        stats.materialized = materialized;
        stats.reclaimed_wal_count = reclaimed_wal_count;
        stats.reclaimed_wal_bytes = reclaimed_wal_bytes;
        stats.reclamation_complete = reclamation_complete;
        for (const auto& source : wal_resources_->sources) {
            if (source->entry.state == detail::MutationWalState::Frozen) {
                ++stats.frozen_wal_count;
            }
            if (stats.total_wal_bytes
                > std::numeric_limits<std::uint64_t>::max()
                      - source->valid_end) {
                throw CorruptionError("mutation WAL byte count overflows uint64");
            }
            stats.total_wal_bytes += source->valid_end;
        }
        for (const auto& source : wal_resources_->delta_sources) {
            ++stats.delta_run_count;
            if (stats.delta_entry_count
                > std::numeric_limits<std::uint64_t>::max()
                      - source->descriptor.entry_count
                || stats.delta_put_rows
                       > std::numeric_limits<std::uint64_t>::max()
                             - source->descriptor.put_count) {
                throw CorruptionError("mutation delta row count overflows uint64");
            }
            stats.delta_entry_count += source->descriptor.entry_count;
            stats.delta_put_rows += source->descriptor.put_count;
            if (source->descriptor.data_file_size
                > std::numeric_limits<std::uint64_t>::max()
                      - source->descriptor.index_file_size) {
                throw CorruptionError(
                    "mutation delta byte count overflows uint64");
            }
            const std::uint64_t bytes =
                source->descriptor.data_file_size
                + source->descriptor.index_file_size;
            if (stats.total_delta_bytes
                > std::numeric_limits<std::uint64_t>::max() - bytes) {
                throw CorruptionError(
                    "mutation delta byte count overflows uint64");
            }
            stats.total_delta_bytes += bytes;
        }
        return stats;
    }

    [[nodiscard]] bool rotate_wal_locked() {
        if (!catalog_managed_) {
            throw InvalidArgumentError(
                "WAL rotation requires a catalog-managed format-5 collection");
        }
        WalSource& old_active = active_wal();
        if (next_sequence_ == old_active.header.first_sequence) {
            return false;
        }
        if (catalog_.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("mutation catalog generation is exhausted");
        }
        if (catalog_.wal_entries.size()
            >= detail::kMutationCatalogMaxEntries) {
            throw StorageError("mutation catalog WAL entry limit is exhausted");
        }
        if (catalog_.next_segment_id >= detail::SegmentId::kMaximum) {
            throw StorageError("mutation segment ID space is exhausted");
        }
        const std::uint64_t previous_sequence = next_sequence_ - 1;
        if (previous_sequence
            >= detail::VersionedMutationValue::kMaximumCommitSequence) {
            throw StorageError("mutation commit sequence space is exhausted");
        }

        detail::MutationWalEntry frozen = old_active.entry;
        frozen.state = detail::MutationWalState::Frozen;
        frozen.last_sequence = previous_sequence;
        frozen.put_count = old_active.payload_offsets.size();
        frozen.sealed_file_size = append_offset_;

        detail::MutationWalEntry successor;
        successor.wal_id = make_uuid();
        successor.predecessor_wal_id = old_active.entry.wal_id;
        successor.segment_id = catalog_.next_segment_id;
        successor.state = detail::MutationWalState::Active;
        successor.previous_sequence = previous_sequence;
        successor.first_sequence = next_sequence_;

        detail::MutationCatalog next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        next_catalog.next_segment_id = successor.segment_id + 1;
        next_catalog.wal_entries.back() = frozen;
        next_catalog.wal_entries.push_back(successor);

        const detail::WalHeader successor_header = wal_header_for(
            manifest_, successor.wal_id, successor.predecessor_wal_id,
            successor.previous_sequence);
        const std::filesystem::path successor_path =
            detail::mutation_wal_path(path_, successor.wal_id);

        try {
            detail::ScopedAdvisoryLock commit(
                commit_lock_, detail::FileLockMode::Exclusive);
            if (detail::file_size(old_active.fd.get(), old_active.path)
                != append_offset_) {
                throw CorruptionError(
                    "active WAL size changed before rotation");
            }
            // Every acknowledged transaction was already synced, but this
            // barrier also establishes an explicit immutable freeze boundary.
            detail::sync_file(old_active.fd.get(), old_active.path);
            detail::create_wal_file(successor_path, successor_header);
            detail::sync_directory(path_ / "wal");

            auto successor_source = std::make_shared<WalSource>();
            successor_source->entry = successor;
            successor_source->header = successor_header;
            successor_source->fd = detail::open_read_write(successor_path);
            successor_source->path = successor_path;
            successor_source->valid_end = detail::kWalFileHeaderSize;
            successor_source->next_sequence = successor.first_sequence;

            auto next_resources = std::make_shared<WalResourceSet>();
            {
                std::shared_lock state(state_mutex_);
                next_resources->sources = wal_resources_->sources;
                next_resources->source_indices =
                    wal_resources_->source_indices;
                next_resources->delta_sources =
                    wal_resources_->delta_sources;
                next_resources->delta_source_indices =
                    wal_resources_->delta_source_indices;
                next_resources->active_index = wal_resources_->active_index;
            }
            next_resources->sources.push_back(std::move(successor_source));
            if (!next_resources->source_indices.emplace(
                     successor.segment_id,
                     next_resources->sources.size() - 1).second) {
                throw CorruptionError(
                    "successor WAL reuses a mutation segment ID");
            }

            // CURRENT is the only durable switch. Everything needed for the
            // no-throw in-memory publication below has already been allocated.
            static_cast<void>(detail::publish_mutation_catalog(
                path_, manifest_, next_catalog));

            {
                std::unique_lock state(state_mutex_);
                old_active.entry = frozen;
                wal_resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
                wal_resources_->active_index =
                    wal_resources_->sources.size() - 1;
                append_offset_ = detail::kWalFileHeaderSize;
                active_wal().valid_end = append_offset_;
                active_wal().next_sequence = next_sequence_;
            }
        } catch (...) {
            // A publication error can be reported after CURRENT was renamed
            // but before its directory fsync completed. Never risk appending
            // to what may now be a catalog-frozen predecessor.
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        return true;
    }

    void prepare_publication_capacity(
        const std::size_t overlay_entries,
        const std::size_t put_count) {
        if (overlay_entries
            > std::numeric_limits<std::size_t>::max() - overlay_.size()) {
            throw InvalidArgumentError("mutation overlay is too large");
        }
        if (put_count
            > std::numeric_limits<std::size_t>::max()
                  - active_wal().payload_offsets.size()) {
            throw InvalidArgumentError("WAL RowRef table is too large");
        }
        overlay_.reserve(overlay_.size() + overlay_entries);
        active_wal().payload_offsets.reserve(
            active_wal().payload_offsets.size() + put_count);
    }

    void prime_overlay_node_unlocked(const std::uint64_t id) {
        if (overlay_.find(id) != overlay_.end()) {
            return;
        }
        // A disengaged optional is a preallocated publication node. Until its
        // durable commit is installed, lookup deliberately falls through to
        // the immutable base and therefore preserves the pre-commit view.
        overlay_.emplace(id, std::nullopt);
    }

    void commit_operations(
        const std::span<const detail::WalMutation> operations,
        const std::uint64_t original_wal_rows,
        const std::uint64_t resulting_count) {
        WalSource& active = active_wal();
        const detail::EncodedWalTransaction transaction =
            detail::encode_wal_transaction(
                active.header, next_sequence_, append_offset_, operations);
        const std::uint64_t relative_payload_offset = detail::get_u64(
            std::span<const std::byte>(transaction.bytes), 64);
        std::uint64_t payload_cursor = append_offset_ + relative_payload_offset;

        std::vector<std::uint64_t> put_offsets;
        put_offsets.reserve(operations.size());
        for (const detail::WalMutation& operation : operations) {
            if (operation.kind == detail::WalOperationKind::Put) {
                put_offsets.push_back(payload_cursor);
                payload_cursor += manifest_.record_size;
            }
        }

        std::uint64_t new_end = 0;
        try {
            detail::ScopedAdvisoryLock commit(
                commit_lock_, detail::FileLockMode::Exclusive);
            new_end = detail::append_wal_transaction_durable(
                active.fd.get(), active.path, append_offset_, transaction);

            try {
                std::unique_lock lock(state_mutex_);
                std::size_t put_index = 0;
                for (const detail::WalMutation& operation : operations) {
                    if (operation.kind == detail::WalOperationKind::Put) {
                        const std::uint64_t row_index =
                            original_wal_rows + put_index;
                        const detail::RowRef row = detail::RowRef::create(
                            active.entry.segment_id, row_index);
                        active.payload_offsets.push_back(put_offsets[put_index]);
                        overlay_.insert_or_assign(
                            operation.id,
                            detail::VersionedMutationValue::put(
                                next_sequence_, row));
                        ++put_index;
                    } else {
                        overlay_.insert_or_assign(
                            operation.id,
                            detail::VersionedMutationValue::tombstone(
                                next_sequence_));
                    }
                }
                live_count_.store(resulting_count, std::memory_order_release);
                append_offset_ = new_end;
                ++next_sequence_;
                active.valid_end = append_offset_;
                active.next_sequence = next_sequence_;
                // Publish only after every overlay and WAL RowRef update is
                // complete. A reader that still observes false is concurrent
                // with this first commit and may linearize immediately before
                // it; a reader that observes true takes state_mutex_.
                mutation_read_path_active_.store(
                    true, std::memory_order_release);
            } catch (...) {
                poisoned_.store(true, std::memory_order_release);
                throw;
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
    }

    std::filesystem::path path_;
    NativeOpenStats open_stats_;
    detail::NativeManifest manifest_;
    detail::IndexData index_;
    detail::PhysicalReader physical_reader_;
    detail::AdvisoryLockFile writer_lock_;
    detail::AdvisoryLockFile commit_lock_;
    bool writable_{false};

    std::vector<BaseSegment> base_segments_;
    bool catalog_managed_{false};
    bool delta_managed_{false};
    detail::MutationCatalog catalog_;
    std::shared_ptr<WalResourceSet> wal_resources_;

    mutable std::shared_mutex state_mutex_;
    std::unordered_map<std::uint64_t, OverlayValue> overlay_;
    std::atomic<bool> mutation_read_path_active_{false};
    std::atomic<std::uint64_t> live_count_{0};
    std::uint64_t append_offset_{detail::kWalFileHeaderSize};
    std::uint64_t next_sequence_{1};
    mutable std::mutex mutation_mutex_;
    mutable std::mutex materialization_mutex_;
    std::atomic<bool> poisoned_{false};
    const pid_t owner_pid_{::getpid()};
};

std::unique_ptr<NativeCollectionBuilder> NativeCollectionBuilder::create(
    const std::filesystem::path& path,
    const NativeBuilderOptions options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection writing requires a little-endian host");
    }
    if (options.dimension == 0) {
        throw InvalidArgumentError("dimension must be greater than zero");
    }
    const detail::DTypeCode code = dtype_code(options.dtype);
    const std::uint64_t record_size = detail::checked_multiply(
        options.dimension, detail::dtype_item_size(code),
        "embedding record size");
    if (options.segment_size_bytes < record_size) {
        throw InvalidArgumentError(
            "segment_size_bytes must fit at least one record");
    }
    return std::unique_ptr<NativeCollectionBuilder>(
        new NativeCollectionBuilder(std::make_unique<Impl>(path, options)));
}

NativeCollectionBuilder::NativeCollectionBuilder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeCollectionBuilder::~NativeCollectionBuilder() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        // After fork, inherited mutex state is not safe to tear down
        // explicitly. Close the child's descriptor references without an
        // explicit unlock, then leave only process-local heap state for exit.
        impl_->abandon_after_fork();
        static_cast<void>(impl_.release());
    }
}

void NativeCollectionBuilder::append(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->append(ids, embeddings);
}
void NativeCollectionBuilder::finish() { impl_->finish(); }
std::uint64_t NativeCollectionBuilder::count() const {
    return impl_->count();
}
std::uint32_t NativeCollectionBuilder::dimension() const {
    return impl_->dimension();
}
DType NativeCollectionBuilder::dtype() const { return impl_->dtype(); }
std::uint32_t NativeCollectionBuilder::format_version() const {
    return impl_->format_version();
}
std::uint64_t NativeCollectionBuilder::record_size() const {
    return impl_->record_size();
}
bool NativeCollectionBuilder::finished() const {
    return impl_->finished();
}

std::unique_ptr<NativeCollection> NativeCollection::open(
    const std::filesystem::path& path,
    const ReaderOptions reader_options,
    const bool writable) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection access requires a little-endian host");
    }
    return std::unique_ptr<NativeCollection>(new NativeCollection(
        std::make_unique<Impl>(
            path, reader_options, writable, SteadyClock::now())));
}

NativeCollection::NativeCollection(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeCollection::~NativeCollection() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        // See the builder destructor above. In particular, do not explicitly
        // LOCK_UN an inherited flock: it belongs to the same open file
        // description as the still-running parent writer.
        impl_->abandon_after_fork();
        static_cast<void>(impl_.release());
    }
}

void NativeCollection::get_into(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    impl_->get_into(ids, output);
}
NativeReadStats NativeCollection::get_into_profiled(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    return impl_->get_into_profiled(ids, output);
}
std::vector<std::byte> NativeCollection::get_bytes(
    const std::span<const std::uint64_t> ids) const {
    return impl_->get_bytes(ids);
}
void NativeCollection::insert(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Insert, ids, embeddings);
}
void NativeCollection::update(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Update, ids, embeddings);
}
void NativeCollection::upsert(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Upsert, ids, embeddings);
}
std::vector<std::uint8_t> NativeCollection::erase(
    const std::span<const std::uint64_t> ids) {
    return impl_->erase(ids);
}
NativeMutationLogStats NativeCollection::mutation_log_stats() const {
    return impl_->mutation_log_stats();
}
NativeMutationLogStats NativeCollection::rotate_wal() {
    return impl_->rotate_wal();
}
NativeMutationLogStats NativeCollection::materialize_frozen_wals() {
    return impl_->materialize_frozen_wals();
}
std::uint64_t NativeCollection::count() const { return impl_->count(); }
std::uint32_t NativeCollection::dimension() const {
    return impl_->dimension();
}
DType NativeCollection::dtype() const { return impl_->dtype(); }
std::uint32_t NativeCollection::format_version() const {
    return impl_->format_version();
}
std::uint64_t NativeCollection::record_size() const {
    return impl_->record_size();
}
bool NativeCollection::writable() const { return impl_->writable(); }
NativeOpenStats NativeCollection::open_stats() const {
    return impl_->open_stats();
}

}  // namespace throwdb
