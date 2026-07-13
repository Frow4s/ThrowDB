#include "throwdb/native_collection.hpp"

#include "file_lock.hpp"
#include "internal.hpp"
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
#include <shared_mutex>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace throwdb {
namespace {

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

    for (const char* child : {"data", "ids", "index", "wal", "locks"}) {
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
    const detail::NativeManifest& manifest) {
    detail::WalHeader header;
    header.collection_id = manifest.collection_id;
    header.schema_id = manifest.schema_id;
    header.wal_id = manifest.active_wal_id;
    header.previous_sequence = 0;
    header.first_sequence = 1;
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

            detail::WalHeader wal_header = wal_header_for(manifest);
            detail::create_wal_file(
                detail::native_active_wal_path(path_), wal_header);
            detail::sync_directory(path_ / "wal");

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
        return kCurrentPublicNativeFormatVersion;
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
          wal_path_(detail::native_active_wal_path(path_)),
          wal_header_(wal_header_for(manifest_)),
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
        wal_fd_ = writable_ ? detail::open_read_write(wal_path_)
                            : detail::open_read_only(wal_path_);
        const detail::ExpectedWalIdentity expected =
            detail::ExpectedWalIdentity::from_trusted_header(wal_header_);
        const detail::WalScanResult scan = detail::scan_wal_file(
            wal_fd_.get(), wal_path_, expected, detail::WalScanMode::Active,
            [this](const detail::RecoveredWalBatch& batch) {
                apply_recovered_batch(batch);
            });
        append_offset_ = scan.valid_end;
        next_sequence_ = scan.next_sequence;
        if (writable_ && scan.ignored_tail_bytes != 0) {
            detail::truncate_file(wal_fd_.get(), scan.valid_end, wal_path_);
            detail::sync_file(wal_fd_.get(), wal_path_);
        }
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
        const std::uint64_t original_wal_rows = wal_payload_offsets_.size();
        if (ids.size() > detail::RowRef::kMaximumRowsPerSegment
                             - original_wal_rows) {
            throw InvalidArgumentError(
                "active WAL exceeds the 2^32 PUT-row limit");
        }
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
            operations, wal_payload_offsets_.size(), resulting_count);
        return deleted;
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
        wal_fd_.reset();
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
                    *row, static_cast<std::uint64_t>(output_index)));
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

    [[nodiscard]] bool base_contains(const std::uint64_t id) const noexcept {
        return detail::native_index_lookup(index_, id).has_value();
    }

    [[nodiscard]] std::optional<detail::RowRef> logical_lookup_unlocked(
        const std::uint64_t id) const {
        const auto overlay = overlay_.find(id);
        if (overlay != overlay_.end()) {
            return overlay->second.row_ref();
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
        const std::uint64_t output_index) const {
        const std::uint32_t segment = row.segment_id().value();
        const std::uint64_t row_index = row.row_index();
        if (segment <= manifest_.base_segment_count) {
            return resolve_base_row(row, output_index);
        }
        if (segment != manifest_.active_wal_segment_id
            || row_index >= wal_payload_offsets_.size()) {
            throw CorruptionError("overlay RowRef does not resolve to active WAL");
        }
        return detail::PhysicalReadRequest{
            wal_fd_.get(),
            wal_payload_offsets_[static_cast<std::size_t>(row_index)],
            output_index};
    }

    void apply_recovered_batch(const detail::RecoveredWalBatch& batch) {
        for (const detail::RecoveredWalOperation& operation : batch.operations) {
            const bool existed = [&] {
                const auto overlay = overlay_.find(operation.id);
                if (overlay != overlay_.end()) {
                    return overlay->second.is_put();
                }
                return base_contains(operation.id);
            }();
            if (operation.kind == detail::WalOperationKind::Put) {
                if (operation.payload_size != manifest_.record_size) {
                    throw CorruptionError("WAL PUT has unexpected payload size");
                }
                if (wal_payload_offsets_.size()
                    >= detail::RowRef::kMaximumRowsPerSegment) {
                    throw CorruptionError("active WAL contains more than 2^32 PUTs");
                }
                const detail::RowRef row = detail::RowRef::create(
                    manifest_.active_wal_segment_id,
                    wal_payload_offsets_.size());
                wal_payload_offsets_.push_back(operation.payload_offset);
                overlay_.insert_or_assign(
                    operation.id, detail::MutationValue::put(row));
                if (!existed) {
                    live_count_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                overlay_.insert_or_assign(
                    operation.id, detail::MutationValue::tombstone());
                if (existed) {
                    live_count_.fetch_sub(1, std::memory_order_relaxed);
                }
            }
        }
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
                  - wal_payload_offsets_.size()) {
            throw InvalidArgumentError("WAL RowRef table is too large");
        }
        overlay_.reserve(overlay_.size() + overlay_entries);
        wal_payload_offsets_.reserve(wal_payload_offsets_.size() + put_count);
    }

    void prime_overlay_node_unlocked(const std::uint64_t id) {
        if (overlay_.find(id) != overlay_.end()) {
            return;
        }
        const auto base = detail::native_index_lookup(index_, id);
        if (base.has_value()) {
            overlay_.emplace(id, detail::MutationValue::put(*base));
        } else {
            overlay_.emplace(id, detail::MutationValue::tombstone());
        }
    }

    void commit_operations(
        const std::span<const detail::WalMutation> operations,
        const std::uint64_t original_wal_rows,
        const std::uint64_t resulting_count) {
        const detail::EncodedWalTransaction transaction =
            detail::encode_wal_transaction(
                wal_header_, next_sequence_, append_offset_, operations);
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
                wal_fd_.get(), wal_path_, append_offset_, transaction);

            try {
                std::unique_lock lock(state_mutex_);
                std::size_t put_index = 0;
                for (const detail::WalMutation& operation : operations) {
                    if (operation.kind == detail::WalOperationKind::Put) {
                        const std::uint64_t row_index =
                            original_wal_rows + put_index;
                        const detail::RowRef row = detail::RowRef::create(
                            manifest_.active_wal_segment_id, row_index);
                        wal_payload_offsets_.push_back(put_offsets[put_index]);
                        overlay_.insert_or_assign(
                            operation.id, detail::MutationValue::put(row));
                        ++put_index;
                    } else {
                        overlay_.insert_or_assign(
                            operation.id,
                            detail::MutationValue::tombstone());
                    }
                }
                live_count_.store(resulting_count, std::memory_order_release);
                append_offset_ = new_end;
                ++next_sequence_;
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
    detail::FileDescriptor wal_fd_;
    std::filesystem::path wal_path_;
    detail::WalHeader wal_header_;

    mutable std::shared_mutex state_mutex_;
    std::unordered_map<std::uint64_t, detail::MutationValue> overlay_;
    std::vector<std::uint64_t> wal_payload_offsets_;
    std::atomic<bool> mutation_read_path_active_{false};
    std::atomic<std::uint64_t> live_count_{0};
    std::uint64_t append_offset_{detail::kWalFileHeaderSize};
    std::uint64_t next_sequence_{1};
    std::mutex mutation_mutex_;
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
