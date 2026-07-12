#include "throwdb/storage.hpp"

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <limits>
#include <system_error>

namespace throwdb {
namespace {

constexpr std::size_t kStagingBatchEntries = 1U << 16;

[[nodiscard]] detail::DTypeCode dtype_code(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return detail::DTypeCode::Float16;
        case DType::Float32:
            return detail::DTypeCode::Float32;
    }
    throw InvalidArgumentError("unsupported embedding dtype");
}

[[nodiscard]] std::uint32_t dtype_item_size(const DType dtype) {
    return detail::dtype_item_size(dtype_code(dtype));
}

[[nodiscard]] detail::FormatVersion write_format_version(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return detail::FormatVersion::V1;
        case DType::Float32:
            return detail::FormatVersion::V2;
    }
    throw InvalidArgumentError("unsupported embedding dtype");
}

[[nodiscard]] std::uint64_t current_unix_time_ns() {
    const auto raw = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return raw > 0 ? static_cast<std::uint64_t>(raw) : 0;
}

void create_store_directories(const std::filesystem::path& path) {
    std::error_code error;
    if (std::filesystem::exists(path, error)) {
        if (error) {
            throw StorageError("cannot inspect storage path: " + error.message());
        }
        if (!std::filesystem::is_directory(path, error) || error) {
            throw InvalidArgumentError("storage path exists and is not a directory");
        }
        if (!std::filesystem::is_empty(path, error) || error) {
            throw InvalidArgumentError("storage path must be absent or empty");
        }
    } else {
        if (error) {
            throw StorageError("cannot inspect storage path: " + error.message());
        }
        if (!std::filesystem::create_directories(path, error) || error) {
            throw StorageError("cannot create storage directory: " + error.message());
        }
    }
    for (const char* child : {"data", "index", "staging"}) {
        if (!std::filesystem::create_directory(path / child, error) || error) {
            throw StorageError(
                "cannot create storage subdirectory '" + std::string(child)
                + "': " + error.message());
        }
    }
    detail::sync_directory(path);
}

}  // namespace

class Writer::Impl {
public:
    Impl(std::filesystem::path path, const WriterOptions options)
        : path_(std::move(path)),
          options_(options),
          dtype_code_(dtype_code(options.dtype)),
          item_size_(dtype_item_size(options.dtype)),
          format_version_(write_format_version(options.dtype)),
          snapshot_id_(format_version_ == detail::FormatVersion::V2
                           ? detail::make_snapshot_id()
                           : detail::SnapshotId{}),
          record_size_(detail::checked_multiply(
              options.dimension, item_size_, "embedding record size")),
          records_per_shard_(options.shard_size_bytes / record_size_),
          staging_path_(path_ / "staging" / "id_ordinal.log") {
        create_store_directories(path_);
        staging_fd_ = detail::create_exclusive(staging_path_);
        const auto header = detail::encode_staging_header(0);
        detail::write_all(staging_fd_.get(), header.data(), header.size(), staging_path_);
    }

    ~Impl() = default;

    void append(const std::span<const std::uint64_t> ids,
                const std::span<const std::byte> embeddings) {
        if (finished_) {
            throw InvalidArgumentError("cannot append to a finished writer");
        }
        const std::uint64_t expected_bytes = detail::checked_multiply(
            ids.size(), record_size_, "embedding batch");
        if (embeddings.size() != expected_bytes) {
            throw InvalidArgumentError(
                "embedding byte count does not equal ids.size() * record_size");
        }
        if (ids.empty()) {
            return;
        }
        if (ids.size() > std::numeric_limits<std::uint64_t>::max() - count_) {
            throw InvalidArgumentError("record count is too large");
        }

        write_embeddings(embeddings, ids.size());
        append_staging_entries(ids);
        count_ += ids.size();
    }

    void finish() {
        if (finished_) {
            return;
        }
        close_active_shard();
        detail::sync_directory(path_ / "data");

        const auto staging_header = detail::encode_staging_header(count_);
        detail::pwrite_all(staging_fd_.get(), staging_header.data(),
                           staging_header.size(), 0, staging_path_);
        detail::sync_file(staging_fd_.get(), staging_path_);
        staging_fd_.reset();
        detail::sync_directory(path_ / "staging");

        auto staging_reader = detail::open_read_only(staging_path_);
        detail::IndexData index = detail::build_index(
            staging_reader.get(), staging_path_, count_, detail::kHashSeed);
        staging_reader.reset();

        detail::Manifest manifest;
        manifest.format_version = format_version_;
        manifest.schema.id_type = detail::IdType::UInt64;
        manifest.schema.dtype = dtype_code_;
        manifest.schema.dimension = options_.dimension;
        manifest.schema.item_size = item_size_;
        manifest.schema.row_layout = detail::RowLayout::DenseRowMajor;
        manifest.schema.index_kind = detail::IndexKind::UInt64Flat;
        manifest.schema.hash_algorithm = detail::HashAlgorithm::SplitMix64V1;
        manifest.record_size = record_size_;
        manifest.record_count = count_;
        manifest.shard_size_bytes = options_.shard_size_bytes;
        manifest.records_per_shard = records_per_shard_;
        manifest.shard_count = shard_count_;
        if (format_version_ == detail::FormatVersion::V2) {
            manifest.header_checksum_algorithm =
                detail::ChecksumAlgorithm::None;
            manifest.expected_string_arena_size = 0;
            manifest.required_feature_bits = 0;
            manifest.optional_feature_bits = 0;
            manifest.snapshot_id = snapshot_id_;
            manifest.hash_seed_high = 0;
            manifest.index_slot_size = detail::kIndexSlotSize;
            manifest.string_arena_payload_offset = 0;
            manifest.creation_time_unix_ns = current_unix_time_ns();
            manifest.header_checksum = 0;
        }
        detail::write_index_atomic(path_, index, manifest);
        detail::write_manifest_atomic(path_, manifest);
        finished_ = true;

        // The durable index is now authoritative; the journal is build-only.
        std::error_code error;
        std::filesystem::remove(staging_path_, error);
        if (!error) {
            detail::sync_directory(path_ / "staging");
        }
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint32_t dimension() const noexcept {
        return options_.dimension;
    }
    [[nodiscard]] DType dtype() const noexcept { return options_.dtype; }
    [[nodiscard]] std::uint32_t format_version() const noexcept {
        return static_cast<std::uint32_t>(format_version_);
    }
    [[nodiscard]] std::uint64_t record_size() const noexcept {
        return record_size_;
    }
    [[nodiscard]] bool finished() const noexcept { return finished_; }

private:
    void open_next_shard() {
        active_shard_id_ = shard_count_;
        active_shard_path_ = detail::shard_path(path_, active_shard_id_);
        active_shard_fd_ = detail::create_exclusive(active_shard_path_);
        active_shard_first_ordinal_ = count_ + pending_batch_records_;
        active_shard_records_ = 0;
        detail::ShardHeader header;
        header.format_version = format_version_;
        header.shard_id = active_shard_id_;
        header.first_ordinal = active_shard_first_ordinal_;
        header.record_size = record_size_;
        header.dimension = options_.dimension;
        header.dtype = dtype_code_;
        header.row_layout = detail::RowLayout::DenseRowMajor;
        if (format_version_ == detail::FormatVersion::V2) {
            header.snapshot_id = snapshot_id_;
            header.payload_checksum_algorithm =
                detail::ChecksumAlgorithm::None;
            header.payload_size = 0;
            header.payload_checksum = 0;
            header.header_checksum = 0;
        }
        const auto encoded = detail::encode_shard_header(header);
        detail::write_all(active_shard_fd_.get(), encoded.data(), encoded.size(),
                          active_shard_path_);
        ++shard_count_;
    }

    void close_active_shard() {
        if (!active_shard_fd_) {
            return;
        }
        detail::ShardHeader header;
        header.format_version = format_version_;
        header.shard_id = active_shard_id_;
        header.first_ordinal = active_shard_first_ordinal_;
        header.record_count = active_shard_records_;
        header.record_size = record_size_;
        header.dimension = options_.dimension;
        header.dtype = dtype_code_;
        header.row_layout = detail::RowLayout::DenseRowMajor;
        if (format_version_ == detail::FormatVersion::V2) {
            header.snapshot_id = snapshot_id_;
            header.payload_checksum_algorithm =
                detail::ChecksumAlgorithm::None;
            header.payload_size = detail::checked_multiply(
                active_shard_records_, record_size_, "shard payload size");
            header.payload_checksum = 0;
            header.header_checksum = 0;
        }
        const auto encoded = detail::encode_shard_header(header);
        detail::pwrite_all(active_shard_fd_.get(), encoded.data(), encoded.size(),
                           0, active_shard_path_);
        detail::sync_file(active_shard_fd_.get(), active_shard_path_);
        active_shard_fd_.reset();
        active_shard_records_ = 0;
    }

    void write_embeddings(const std::span<const std::byte> embeddings,
                          const std::uint64_t record_count) {
        const std::byte* cursor = embeddings.data();
        std::uint64_t remaining_records = record_count;
        pending_batch_records_ = 0;
        while (remaining_records != 0) {
            if (!active_shard_fd_) {
                open_next_shard();
            }
            const std::uint64_t available =
                records_per_shard_ - active_shard_records_;
            const std::uint64_t records = std::min(available, remaining_records);
            const std::uint64_t bytes = records * record_size_;
            detail::write_all(active_shard_fd_.get(), cursor,
                              static_cast<std::size_t>(bytes), active_shard_path_);
            cursor += static_cast<std::size_t>(bytes);
            remaining_records -= records;
            active_shard_records_ += records;
            pending_batch_records_ += records;
            if (active_shard_records_ == records_per_shard_) {
                close_active_shard();
            }
        }
        pending_batch_records_ = 0;
    }

    void append_staging_entries(const std::span<const std::uint64_t> ids) {
        std::vector<std::byte> encoded(kStagingBatchEntries * detail::kStagingEntrySize);
        std::uint64_t processed = 0;
        while (processed != ids.size()) {
            const std::uint64_t batch_count = std::min<std::uint64_t>(
                kStagingBatchEntries, ids.size() - processed);
            const std::size_t batch_bytes = static_cast<std::size_t>(
                batch_count * detail::kStagingEntrySize);
            std::span<std::byte> output(encoded.data(), batch_bytes);
            for (std::uint64_t item = 0; item != batch_count; ++item) {
                const std::size_t offset = static_cast<std::size_t>(
                    item * detail::kStagingEntrySize);
                detail::put_u64(output, offset,
                                ids[static_cast<std::size_t>(processed + item)]);
                detail::put_u64(output, offset + 8, count_ + processed + item);
            }
            detail::write_all(staging_fd_.get(), encoded.data(), batch_bytes,
                              staging_path_);
            processed += batch_count;
        }
    }

    std::filesystem::path path_;
    WriterOptions options_;
    detail::DTypeCode dtype_code_{detail::DTypeCode::Invalid};
    std::uint32_t item_size_{0};
    detail::FormatVersion format_version_{detail::FormatVersion::Invalid};
    detail::SnapshotId snapshot_id_;
    std::uint64_t record_size_{0};
    std::uint64_t records_per_shard_{0};
    std::filesystem::path staging_path_;
    detail::FileDescriptor staging_fd_;

    detail::FileDescriptor active_shard_fd_;
    std::filesystem::path active_shard_path_;
    std::uint64_t active_shard_id_{0};
    std::uint64_t active_shard_first_ordinal_{0};
    std::uint64_t active_shard_records_{0};
    std::uint64_t pending_batch_records_{0};
    std::uint64_t shard_count_{0};
    std::uint64_t count_{0};
    bool finished_{false};
};

std::unique_ptr<Writer> Writer::create(const std::filesystem::path& path,
                                       const WriterOptions options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (options.dimension == 0) {
        throw InvalidArgumentError("dimension must be greater than zero");
    }
    const std::uint32_t item_size = dtype_item_size(options.dtype);
    const detail::FormatVersion format_version =
        write_format_version(options.dtype);
    if (format_version == detail::FormatVersion::V2
        && std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "format-v2 writing requires a little-endian host");
    }
    const std::uint64_t record_size = detail::checked_multiply(
        options.dimension, item_size, "embedding record size");
    if (options.shard_size_bytes < record_size) {
        throw InvalidArgumentError("shard_size_bytes must fit at least one record");
    }
    return std::unique_ptr<Writer>(
        new Writer(std::make_unique<Impl>(path, options)));
}

Writer::Writer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Writer::~Writer() = default;

void Writer::append(const std::span<const std::uint64_t> ids,
                    const std::span<const std::byte> embeddings) {
    impl_->append(ids, embeddings);
}

void Writer::append(const std::uint64_t* ids, const std::size_t count,
                    const void* embeddings, const std::size_t embedding_bytes) {
    if (count != 0 && ids == nullptr) {
        throw InvalidArgumentError("ids pointer must not be null");
    }
    if (embedding_bytes != 0 && embeddings == nullptr) {
        throw InvalidArgumentError("embeddings pointer must not be null");
    }
    impl_->append(
        std::span<const std::uint64_t>(ids, count),
        std::span<const std::byte>(static_cast<const std::byte*>(embeddings),
                                  embedding_bytes));
}

void Writer::finish() { impl_->finish(); }
std::uint64_t Writer::count() const noexcept { return impl_->count(); }
std::uint32_t Writer::dimension() const noexcept { return impl_->dimension(); }
DType Writer::dtype() const noexcept { return impl_->dtype(); }
std::uint32_t Writer::format_version() const noexcept {
    return impl_->format_version();
}
std::uint64_t Writer::record_size() const noexcept { return impl_->record_size(); }
bool Writer::finished() const noexcept { return impl_->finished(); }

}  // namespace throwdb
