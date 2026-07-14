#include "partitioned_index_storage.hpp"

#include "crc32c.hpp"
#include "file_lock.hpp"
#include "mutation_wal.hpp"
#include "partitioned_index_format.hpp"
#include "wal_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

constexpr std::array<char, 8> kPartitionedManifestDiscriminator{
    'T', 'D', 'B', 'N', 'M', 'N', 'F', '2'};
constexpr std::size_t kChecksumBufferBytes = 1U << 20;
using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(
    const SteadyClock::time_point started) noexcept {
    return std::chrono::duration<double>(SteadyClock::now() - started).count();
}

[[nodiscard]] int nofollow_flag() noexcept {
#if defined(O_NOFOLLOW)
    return O_NOFOLLOW;
#else
    return 0;
#endif
}

[[nodiscard]] FileDescriptor open_regular(
    const std::filesystem::path& path,
    const bool writable = false) {
    const int fd = ::open(
        path.c_str(),
        (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NONBLOCK
            | nofollow_flag());
    if (fd < 0) {
        throw StorageError(system_error_message("cannot open file", path));
    }
    FileDescriptor result(fd);
    struct stat status {};
    if (::fstat(result.get(), &status) < 0) {
        throw StorageError(system_error_message("cannot inspect file", path));
    }
    if (!S_ISREG(status.st_mode)) {
        throw CorruptionError(
            "collection resource is not a regular file: '" + path.string()
            + "'");
    }
    return result;
}

void require_regular_directory(const std::filesystem::path& path) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) < 0) {
        throw StorageError(system_error_message("cannot inspect directory", path));
    }
    if (!S_ISDIR(status.st_mode)) {
        throw CorruptionError(
            "collection namespace is not a directory: '" + path.string()
            + "'");
    }
}

[[nodiscard]] std::vector<std::byte> read_exact_file(
    const std::filesystem::path& path,
    const std::uint64_t expected_size) {
    if (expected_size > std::numeric_limits<std::size_t>::max()) {
        throw CorruptionError("collection resource does not fit address space");
    }
    FileDescriptor fd = open_regular(path);
    if (file_size(fd.get(), path) != expected_size) {
        throw CorruptionError(
            "collection resource has unexpected size: '" + path.string()
            + "'");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(expected_size));
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    return bytes;
}

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize> read_header(
    const int fd,
    const std::filesystem::path& path) {
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    pread_exact(fd, bytes.data(), bytes.size(), 0, path);
    return bytes;
}

[[nodiscard]] Uuid128 make_uuid() {
    const SnapshotId value = make_snapshot_id();
    return Uuid128{value.low, value.high};
}

[[nodiscard]] std::uint64_t current_unix_time_ns() noexcept {
    const auto raw = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return raw > 0 ? static_cast<std::uint64_t>(raw) : 0;
}

[[nodiscard]] DTypeCode private_dtype(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return DTypeCode::Float16;
        case DType::Float32:
            return DTypeCode::Float32;
    }
    throw InvalidArgumentError("unsupported embedding dtype");
}

[[nodiscard]] DType public_dtype(const DTypeCode dtype) {
    if (dtype == DTypeCode::Float16) {
        return DType::Float16;
    }
    if (dtype == DTypeCode::Float32) {
        return DType::Float32;
    }
    throw CorruptionError("partitioned collection has unsupported dtype");
}

[[nodiscard]] std::uint64_t validated_record_size(
    const NativeBuilderOptions& options) {
    if (options.dimension == 0) {
        throw InvalidArgumentError("dimension must be greater than zero");
    }
    const DTypeCode dtype = private_dtype(options.dtype);
    const std::uint64_t size = checked_multiply(
        options.dimension, dtype_item_size(dtype), "embedding record size");
    if (options.segment_size_bytes < size) {
        throw InvalidArgumentError(
            "segment_size_bytes must fit at least one record");
    }
    return size;
}

[[nodiscard]] std::uint64_t validated_partition_count(
    const std::uint32_t partition_bits) {
    if (partition_bits > kPartitionedMaximumPartitionBits) {
        throw InvalidArgumentError(
            "format-7 partition_bits must be between 0 and 16");
    }
    return std::uint64_t{1} << partition_bits;
}

[[nodiscard]] std::string uuid_token(const Uuid128 value) {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> token{};
    const auto put_word = [&token](const std::uint64_t word,
                                   const std::size_t begin) {
        for (std::size_t nibble = 0; nibble != 16; ++nibble) {
            const unsigned shift = static_cast<unsigned>((15 - nibble) * 4);
            token[begin + nibble] = hex[(word >> shift) & 0xfU];
        }
    };
    put_word(value.high, 0);
    put_word(value.low, 16);
    return std::string(token.begin(), token.end());
}

[[nodiscard]] std::filesystem::path manifest_path(
    const std::filesystem::path& root) {
    return root / "manifest.bin";
}
[[nodiscard]] std::filesystem::path partitioned_current_path(
    const std::filesystem::path& root) {
    return root / "CURRENT";
}
[[nodiscard]] std::filesystem::path writer_lock_path(
    const std::filesystem::path& root) {
    return root / "locks" / "writer.lock";
}
[[nodiscard]] std::filesystem::path commit_lock_path(
    const std::filesystem::path& root) {
    return root / "locks" / "commit.lock";
}
[[nodiscard]] std::filesystem::path view_lock_path(
    const std::filesystem::path& root) {
    return root / "locks" / "view.lock";
}
[[nodiscard]] std::filesystem::path segment_data_path(
    const std::filesystem::path& root,
    const Uuid128 id) {
    return root / "segments" / ("segment-" + uuid_token(id) + ".dat");
}
[[nodiscard]] std::filesystem::path segment_ids_path(
    const std::filesystem::path& root,
    const Uuid128 id) {
    return root / "segments" / ("segment-" + uuid_token(id) + ".ids");
}
[[nodiscard]] std::filesystem::path snapshot_path(
    const std::filesystem::path& root,
    const PartitionSnapshotDescriptor& descriptor) {
    return root / "index" / "partitions"
           / ("partition-" + std::to_string(descriptor.partition_id) + "-"
              + uuid_token(descriptor.snapshot_id) + ".idx");
}
[[nodiscard]] std::filesystem::path catalog_path(
    const std::filesystem::path& root,
    const Uuid128 id) {
    return root / "catalogs" / ("catalog-" + uuid_token(id) + ".bin");
}
[[nodiscard]] std::filesystem::path wal_path(
    const std::filesystem::path& root,
    const Uuid128 id) {
    return root / "wal" / ("wal-" + uuid_token(id) + ".wal");
}

void rename_file(
    const std::filesystem::path& from,
    const std::filesystem::path& to,
    const std::string_view description) {
    std::error_code error;
    std::filesystem::rename(from, to, error);
    if (error) {
        throw StorageError(
            "cannot publish " + std::string(description) + ": "
            + error.message());
    }
}

void write_atomic(
    const std::filesystem::path& root,
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination,
    const std::span<const std::byte> bytes,
    const std::string_view description) {
    FileDescriptor fd = create_exclusive(temporary);
    write_all(fd.get(), bytes.data(), bytes.size(), temporary);
    sync_file(fd.get(), temporary);
    fd.reset();
    rename_file(temporary, destination, description);
    sync_directory(destination.parent_path());
    if (destination.parent_path() != root) {
        // The parent directory entry is durable above; callers decide when a
        // root-level pointer should become visible.
        return;
    }
}

[[nodiscard]] WalHeader partitioned_wal_header_for(
    const PartitionedManifestV2& manifest,
    const MutationWalEntry& entry) {
    WalHeader header;
    header.collection_id = manifest.collection_id;
    header.schema_id = manifest.schema_id;
    header.wal_id = entry.wal_id;
    header.predecessor_wal_id = entry.predecessor_wal_id;
    header.previous_sequence = entry.previous_sequence;
    header.first_sequence = entry.first_sequence;
    header.record_size = manifest.record_size;
    header.dimension = manifest.schema.dimension;
    header.dtype = manifest.schema.dtype;
    header.item_size = manifest.schema.item_size;
    header.id_type = manifest.schema.id_type;
    header.row_layout = manifest.schema.row_layout;
    header.creation_time_unix_ns = current_unix_time_ns();
    return header;
}

void publish_partitioned_catalog(
    const std::filesystem::path& root,
    const PartitionedManifestV2& manifest,
    const PartitionedCatalogV3& catalog) {
    const std::vector<std::byte> bytes =
        encode_partitioned_catalog_v3(catalog);
    const std::filesystem::path destination =
        catalog_path(root, catalog.catalog_id);
    write_atomic(
        root, destination.string() + ".tmp", destination, bytes,
        "partitioned catalog");

    PartitionedCurrentV1 current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = bytes.size();
    current.catalog_crc32c = crc32c(bytes);
    const auto encoded = encode_partitioned_current_v1(manifest, current);
    write_atomic(
        root,
        root / ("CURRENT." + uuid_token(catalog.catalog_id) + ".tmp"),
        partitioned_current_path(root), encoded, "partitioned CURRENT");
}

void validate_embedding_batch(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint64_t record_size) {
    const std::uint64_t expected = checked_multiply(
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

[[nodiscard]] std::uint32_t checksum_payload(
    const int fd,
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint64_t size) {
    Crc32c checksum;
    std::vector<std::byte> buffer(kChecksumBufferBytes);
    std::uint64_t cursor = 0;
    while (cursor != size) {
        const std::size_t amount = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), size - cursor));
        pread_exact(fd, buffer.data(), amount, offset + cursor, path);
        checksum.update(std::span<const std::byte>(buffer.data(), amount));
        cursor += amount;
    }
    return checksum.value();
}

[[nodiscard]] std::uint64_t partition_for_hash(
    const std::uint64_t hash,
    const std::uint32_t partition_bits) noexcept {
    return partition_bits == 0 ? 0 : hash >> (64 - partition_bits);
}

[[nodiscard]] std::uint64_t choose_capacity(const std::uint64_t count) {
    std::uint64_t capacity = 8;
    while (count
           > (capacity / 10) * 7 + ((capacity % 10) * 7) / 10) {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / 2) {
            throw InvalidArgumentError("partition snapshot is too large");
        }
        capacity *= 2;
    }
    if (capacity > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("partition snapshot does not fit memory");
    }
    return capacity;
}

[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw InvalidArgumentError(std::string(description) + " overflows uint64");
    }
    return left + right;
}

[[nodiscard]] std::uint64_t snapshot_lookup_offset(
    const std::uint64_t capacity) {
    return align_up(
        checked_sum(kPartitionedHeaderSize, capacity, "snapshot controls"),
        kPartitionedHeaderSize);
}

[[nodiscard]] std::uint64_t snapshot_sequence_offset(
    const std::uint64_t capacity) {
    return align_up(
        checked_sum(
            snapshot_lookup_offset(capacity),
            checked_multiply(
                capacity, kPartitionSnapshotLookupSlotSize,
                "snapshot lookup slots"),
            "snapshot lookup end"),
        kPartitionedHeaderSize);
}

[[nodiscard]] std::uint64_t snapshot_file_size(
    const std::uint64_t capacity) {
    return checked_sum(
        snapshot_sequence_offset(capacity),
        checked_multiply(
            capacity, kPartitionSnapshotSequenceSlotSize,
            "snapshot sequence slots"),
        "snapshot file size");
}

[[nodiscard]] PartitionedManifestV2 read_partitioned_manifest(
    const std::filesystem::path& root) {
    const std::vector<std::byte> bytes =
        read_exact_file(manifest_path(root), kPartitionedHeaderSize);
    return decode_partitioned_manifest_v2(bytes);
}

class ReadOnlyMapping {
public:
    ReadOnlyMapping() = default;
    ReadOnlyMapping(
        const int fd,
        const std::uint64_t size,
        const std::filesystem::path& path)
        : size_(size) {
        if (size == 0 || size > std::numeric_limits<std::size_t>::max()) {
            throw CorruptionError("mapped resource has invalid size");
        }
        address_ = ::mmap(
            nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE,
            fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw StorageError(system_error_message("cannot mmap file", path));
        }
    }
    ~ReadOnlyMapping() { reset(); }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping(ReadOnlyMapping&& other) noexcept
        : address_(std::exchange(other.address_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    ReadOnlyMapping& operator=(ReadOnlyMapping&& other) noexcept {
        if (this != &other) {
            reset();
            address_ = std::exchange(other.address_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {static_cast<const std::byte*>(address_),
                static_cast<std::size_t>(size_)};
    }

private:
    void reset() noexcept {
        if (address_ != nullptr) {
            static_cast<void>(
                ::munmap(address_, static_cast<std::size_t>(size_)));
            address_ = nullptr;
            size_ = 0;
        }
    }

    void* address_{nullptr};
    std::uint64_t size_{0};
};

}  // namespace

bool is_partitioned_index_store(const std::filesystem::path& store_path) {
    const std::filesystem::path path = manifest_path(store_path);
    FileDescriptor fd = open_regular(path);
    if (file_size(fd.get(), path) < kPartitionedManifestDiscriminator.size()) {
        return false;
    }
    std::array<char, 8> magic{};
    pread_exact(fd.get(), magic.data(), magic.size(), 0, path);
    return magic == kPartitionedManifestDiscriminator;
}

NativeCollectionMetadata inspect_partitioned_index_metadata(
    const std::filesystem::path& store_path) {
    const PartitionedManifestV2 manifest =
        read_partitioned_manifest(store_path);
    NativeCollectionMetadata metadata;
    metadata.format_version = kPartitionedPublicFormatVersion;
    metadata.dimension = manifest.schema.dimension;
    metadata.dtype = public_dtype(manifest.schema.dtype);
    metadata.record_size = manifest.record_size;
    metadata.segment_size_bytes = manifest.target_segment_size;
    metadata.layout = "uint64_dense_row_major_partitioned_rowref_v1";
    metadata.feature_discriminator =
        "segment_catalog_v1+partitioned_rowref_v1";
    metadata.required_feature_bits = manifest.required_feature_bits;
    metadata.optional_feature_bits = manifest.optional_feature_bits;
    return metadata;
}

class PartitionedIndexBuilder::Impl {
public:
    Impl(std::filesystem::path root, NativeBuilderOptions options)
        : root_(std::move(root)),
          options_(options),
          dtype_(private_dtype(options.dtype)),
          item_size_(dtype_item_size(dtype_)),
          record_size_(validated_record_size(options)),
          rows_per_segment_(std::min<std::uint64_t>(
              options.segment_size_bytes / record_size_,
              RowRef::kMaximumRowsPerSegment)),
          collection_id_(make_uuid()),
          schema_id_(make_uuid()),
          wal_id_(make_uuid()),
          partition_count_(validated_partition_count(options.partition_bits)),
          buckets_(static_cast<std::size_t>(partition_count_)) {
        create_layout();
    }

    ~Impl() {
        if (owned_by_current_process() && writer_lock_.locked()) {
            try {
                writer_lock_.unlock();
            } catch (...) {
            }
        }
    }

    void append(
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings) {
        ensure_building();
        const std::uint64_t expected = checked_multiply(
            ids.size(), record_size_, "embedding batch");
        if (expected > std::numeric_limits<std::size_t>::max()
            || embeddings.size() != static_cast<std::size_t>(expected)) {
            throw InvalidArgumentError(
                "embedding byte count does not equal ids.size() * record_size");
        }
        if (ids.empty()) {
            return;
        }
        if (ids.size() > std::numeric_limits<std::uint64_t>::max() - count_) {
            throw InvalidArgumentError("record count is too large");
        }
        try {
            std::uint64_t input_row = 0;
            while (input_row != ids.size()) {
                if (!data_fd_) {
                    open_segment();
                }
                const std::uint64_t rows = std::min<std::uint64_t>(
                    rows_per_segment_ - active_rows_, ids.size() - input_row);
                append_rows(ids, embeddings, input_row, rows);
                input_row += rows;
                if (active_rows_ == rows_per_segment_) {
                    close_segment();
                }
            }
        } catch (...) {
            poisoned_ = true;
            throw;
        }
    }

    void finish() {
        ensure_owner();
        if (finished_) {
            return;
        }
        ensure_building();
        try {
            close_segment();
            sync_directory(root_ / "segments");

            std::vector<PartitionSnapshotDescriptor> snapshots;
            snapshots.reserve(buckets_.size());
            for (std::uint32_t partition = 0;
                 partition != partition_count_; ++partition) {
                snapshots.push_back(write_snapshot(partition));
                std::vector<IndexSlot>().swap(buckets_[partition]);
            }
            sync_directory(root_ / "index" / "partitions");

            const std::uint32_t wal_segment =
                static_cast<std::uint32_t>(segments_.size() + 1);
            WalHeader wal_header;
            wal_header.collection_id = collection_id_;
            wal_header.schema_id = schema_id_;
            wal_header.wal_id = wal_id_;
            wal_header.first_sequence = 1;
            wal_header.record_size = record_size_;
            wal_header.dimension = options_.dimension;
            wal_header.dtype = dtype_;
            wal_header.item_size = item_size_;
            wal_header.id_type = IdType::UInt64;
            wal_header.row_layout = RowLayout::DenseRowMajor;
            wal_header.creation_time_unix_ns = current_unix_time_ns();
            create_wal_file(wal_path(root_, wal_id_), wal_header);
            sync_directory(root_ / "wal");

            PartitionedCatalogV3 catalog;
            catalog.collection_id = collection_id_;
            catalog.schema_id = schema_id_;
            catalog.generation = 1;
            catalog.catalog_id = make_uuid();
            catalog.next_segment_id = wal_segment + 1;
            catalog.next_index_epoch = 1;
            catalog.indexed_live_count = count_;
            catalog.partition_bits = options_.partition_bits;
            catalog.hash_seed = kHashSeed;
            catalog.segments = segments_;
            catalog.snapshots = std::move(snapshots);
            catalog.wal_entries.push_back(MutationWalEntry{
                wal_id_, {}, wal_segment, MutationWalState::Active,
                0, 1, 0, 0, 0});
            const std::vector<std::byte> catalog_bytes =
                encode_partitioned_catalog_v3(catalog);
            const std::filesystem::path catalog_final =
                catalog_path(root_, catalog.catalog_id);
            write_atomic(
                root_,
                catalog_final.string() + ".tmp",
                catalog_final, catalog_bytes, "partitioned catalog");

            PartitionedManifestV2 manifest;
            manifest.collection_id = collection_id_;
            manifest.schema_id = schema_id_;
            manifest.initial_active_wal_id = wal_id_;
            manifest.schema.id_type = IdType::UInt64;
            manifest.schema.dtype = dtype_;
            manifest.schema.dimension = options_.dimension;
            manifest.schema.item_size = item_size_;
            manifest.schema.row_layout = RowLayout::DenseRowMajor;
            manifest.schema.index_kind = IndexKind::UInt64PartitionedRowRef;
            manifest.schema.hash_algorithm = HashAlgorithm::SplitMix64V1;
            manifest.record_size = record_size_;
            manifest.partition_bits = options_.partition_bits;
            manifest.partition_count = partition_count_;
            manifest.hash_seed = kHashSeed;
            manifest.target_segment_size = options_.segment_size_bytes;
            manifest.initial_record_count = count_;
            manifest.initial_active_wal_segment_id = wal_segment;
            manifest.initial_next_segment_id = wal_segment + 1;
            manifest.creation_time_unix_ns = current_unix_time_ns();

            PartitionedCurrentV1 current;
            current.collection_id = collection_id_;
            current.schema_id = schema_id_;
            current.generation = catalog.generation;
            current.catalog_id = catalog.catalog_id;
            current.catalog_file_size = catalog_bytes.size();
            current.catalog_crc32c = crc32c(catalog_bytes);
            const auto current_bytes =
                encode_partitioned_current_v1(manifest, current);
            write_atomic(
                root_, root_ / ("CURRENT." + uuid_token(catalog.catalog_id)
                                + ".tmp"),
                partitioned_current_path(root_), current_bytes,
                "partitioned CURRENT");

            // The manifest is the only initial visibility point. It is
            // published after every referenced inode and CURRENT are durable.
            const auto manifest_bytes = encode_partitioned_manifest_v2(manifest);
            write_atomic(
                root_, root_ / "manifest.bin.tmp", manifest_path(root_),
                manifest_bytes, "partitioned manifest");
            sync_directory(root_);
            writer_lock_.unlock();
            finished_ = true;
        } catch (...) {
            poisoned_ = true;
            throw;
        }
    }

    [[nodiscard]] std::uint64_t count() const { ensure_owner(); return count_; }
    [[nodiscard]] std::uint32_t dimension() const {
        ensure_owner(); return options_.dimension;
    }
    [[nodiscard]] DType dtype() const { ensure_owner(); return options_.dtype; }
    [[nodiscard]] std::uint64_t record_size() const {
        ensure_owner(); return record_size_;
    }
    [[nodiscard]] bool finished() const { ensure_owner(); return finished_; }
    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }
    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        data_fd_.reset();
        ids_fd_.reset();
        writer_lock_.abandon_after_fork();
    }

private:
    void create_layout() {
        std::error_code error;
        const bool exists = std::filesystem::exists(root_, error);
        if (error) {
            throw StorageError("cannot inspect storage path: " + error.message());
        }
        if (exists) {
            if (!std::filesystem::is_directory(root_, error) || error) {
                throw InvalidArgumentError(
                    "storage path exists and is not a directory");
            }
            if (!std::filesystem::is_empty(root_, error) || error) {
                throw InvalidArgumentError("storage path must be absent or empty");
            }
        } else if (!std::filesystem::create_directories(root_, error) || error) {
            throw StorageError(
                "cannot create storage directory: " + error.message());
        }
        for (const std::filesystem::path& child : {
                 std::filesystem::path("segments"),
                 std::filesystem::path("index"),
                 std::filesystem::path("index/partitions"),
                 std::filesystem::path("wal"),
                 std::filesystem::path("catalogs"),
                 std::filesystem::path("locks")}) {
            if (!std::filesystem::create_directory(root_ / child, error)
                || error) {
                throw StorageError(
                    "cannot create storage subdirectory '" + child.string()
                    + "': " + error.message());
            }
        }
        writer_lock_ = AdvisoryLockFile::create(writer_lock_path(root_));
        writer_lock_.lock(FileLockMode::Exclusive);
        auto commit = AdvisoryLockFile::create(commit_lock_path(root_));
        auto view = AdvisoryLockFile::create(view_lock_path(root_));
        static_cast<void>(commit);
        static_cast<void>(view);
        sync_directory(root_ / "locks");
        // `partitions` is a directory entry in `index`, not in the root.
        // Persist that intermediate namespace before manifest publication.
        sync_directory(root_ / "index");
        sync_directory(root_);
    }

    void ensure_owner() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "partitioned builder was inherited across fork; create a new "
                "builder in the child process");
        }
    }
    void ensure_building() const {
        ensure_owner();
        if (finished_) {
            throw InvalidArgumentError("cannot modify a finished builder");
        }
        if (poisoned_) {
            throw StorageError("partitioned builder is unusable after failure");
        }
    }

    void open_segment() {
        if (segments_.size() >= SegmentId::kMaximum - 1ULL) {
            throw InvalidArgumentError("segment ID space is exhausted");
        }
        active_segment_id_ =
            static_cast<std::uint32_t>(segments_.size() + 1);
        active_segment_uuid_ = make_uuid();
        data_path_ = segment_data_path(root_, active_segment_uuid_);
        ids_path_ = segment_ids_path(root_, active_segment_uuid_);
        data_fd_ = create_exclusive(data_path_);
        ids_fd_ = create_exclusive(ids_path_);
        std::array<std::byte, kPartitionedHeaderSize> empty{};
        write_all(data_fd_.get(), empty.data(), empty.size(), data_path_);
        write_all(ids_fd_.get(), empty.data(), empty.size(), ids_path_);
        active_rows_ = 0;
        data_crc_ = Crc32c{};
        ids_crc_ = Crc32c{};
    }

    void append_rows(
        const std::span<const std::uint64_t> ids,
        const std::span<const std::byte> embeddings,
        const std::uint64_t input_row,
        const std::uint64_t rows) {
        const std::uint64_t byte_count = checked_multiply(
            rows, record_size_, "segment append bytes");
        const std::uint64_t byte_offset = checked_multiply(
            input_row, record_size_, "embedding input offset");
        if (byte_count > std::numeric_limits<std::size_t>::max()
            || byte_offset > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("embedding batch does not fit memory");
        }
        const auto payload = embeddings.subspan(
            static_cast<std::size_t>(byte_offset),
            static_cast<std::size_t>(byte_count));
        write_all(data_fd_.get(), payload.data(), payload.size(), data_path_);
        data_crc_.update(payload);

        if (rows > std::numeric_limits<std::size_t>::max()
                       / kPartitionedSegmentIdEntrySize) {
            throw InvalidArgumentError("IDs sidecar batch does not fit memory");
        }
        std::vector<std::byte> encoded_ids(
            static_cast<std::size_t>(rows * kPartitionedSegmentIdEntrySize));
        for (std::uint64_t row = 0; row != rows; ++row) {
            const std::uint64_t id = ids[static_cast<std::size_t>(input_row + row)];
            const std::size_t offset = static_cast<std::size_t>(
                row * kPartitionedSegmentIdEntrySize);
            put_u64(encoded_ids, offset, id);
            put_u64(encoded_ids, offset + 8, 0);
            const std::uint64_t hash = hash_id(id, kHashSeed);
            const std::uint64_t partition = partition_for_hash(
                hash, options_.partition_bits);
            buckets_[static_cast<std::size_t>(partition)].push_back(IndexSlot{
                id, RowRef::create(active_segment_id_, active_rows_ + row).raw()});
        }
        write_all(
            ids_fd_.get(), encoded_ids.data(), encoded_ids.size(), ids_path_);
        ids_crc_.update(encoded_ids);
        active_rows_ += rows;
        count_ += rows;
    }

    void close_segment() {
        if (!data_fd_) {
            return;
        }
        PartitionedSegmentDescriptor descriptor;
        descriptor.segment_id = active_segment_uuid_;
        descriptor.row_ref_segment_id = active_segment_id_;
        descriptor.kind = PartitionedSegmentKind::Initial;
        descriptor.row_count = active_rows_;
        descriptor.data_file_size = checked_sum(
            kPartitionedHeaderSize,
            checked_multiply(active_rows_, record_size_, "segment data size"),
            "segment data file size");
        descriptor.ids_file_size = checked_sum(
            kPartitionedHeaderSize,
            checked_multiply(
                active_rows_, kPartitionedSegmentIdEntrySize,
                "segment IDs size"),
            "segment IDs file size");
        descriptor.data_payload_crc32c = data_crc_.value();
        descriptor.ids_payload_crc32c = ids_crc_.value();

        PartitionedSegmentDataHeaderV1 data_header;
        data_header.collection_id = collection_id_;
        data_header.schema_id = schema_id_;
        data_header.descriptor = descriptor;
        data_header.schema = CollectionSchema{
            IdType::UInt64, dtype_, options_.dimension, item_size_,
            RowLayout::DenseRowMajor, IndexKind::UInt64PartitionedRowRef,
            HashAlgorithm::SplitMix64V1};
        data_header.record_size = record_size_;
        data_header.payload_size = descriptor.data_file_size
                                   - kPartitionedHeaderSize;
        PartitionedSegmentIdsHeaderV1 ids_header;
        ids_header.collection_id = collection_id_;
        ids_header.schema_id = schema_id_;
        ids_header.descriptor = descriptor;
        ids_header.payload_size = descriptor.ids_file_size
                                  - kPartitionedHeaderSize;
        const auto encoded_data =
            encode_partitioned_segment_data_header_v1(data_header);
        const auto encoded_ids =
            encode_partitioned_segment_ids_header_v1(ids_header);
        pwrite_all(
            data_fd_.get(), encoded_data.data(), encoded_data.size(), 0,
            data_path_);
        pwrite_all(
            ids_fd_.get(), encoded_ids.data(), encoded_ids.size(), 0,
            ids_path_);
        if (file_size(data_fd_.get(), data_path_) != descriptor.data_file_size
            || file_size(ids_fd_.get(), ids_path_) != descriptor.ids_file_size) {
            throw StorageError("partitioned builder wrote unexpected segment size");
        }
        sync_file(data_fd_.get(), data_path_);
        sync_file(ids_fd_.get(), ids_path_);
        data_fd_.reset();
        ids_fd_.reset();
        segments_.push_back(descriptor);
        active_rows_ = 0;
    }

    [[nodiscard]] PartitionSnapshotDescriptor write_snapshot(
        const std::uint32_t partition) {
        const std::vector<IndexSlot>& input = buckets_[partition];
        const std::uint64_t capacity = choose_capacity(input.size());
        const std::uint64_t lookup_offset = snapshot_lookup_offset(capacity);
        const std::uint64_t sequence_offset = snapshot_sequence_offset(capacity);
        const std::uint64_t total_size = snapshot_file_size(capacity);
        if (total_size > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("partition snapshot does not fit memory");
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(total_size));
        const std::size_t controls_offset =
            static_cast<std::size_t>(kPartitionedHeaderSize);
        const std::uint64_t mask = capacity - 1;
        for (const IndexSlot& entry : input) {
            std::uint64_t slot = hash_id(entry.id, kHashSeed) & mask;
            while (bytes[controls_offset + static_cast<std::size_t>(slot)]
                   != std::byte{0}) {
                const std::size_t existing_offset = static_cast<std::size_t>(
                    lookup_offset + slot * kPartitionSnapshotLookupSlotSize);
                if (get_u64(bytes, existing_offset) == entry.id) {
                    throw DuplicateIdError(entry.id);
                }
                slot = (slot + 1) & mask;
            }
            bytes[controls_offset + static_cast<std::size_t>(slot)] =
                std::byte{1};
            const std::size_t offset = static_cast<std::size_t>(
                lookup_offset + slot * kPartitionSnapshotLookupSlotSize);
            put_u64(bytes, offset, entry.id);
            put_u64(bytes, offset + 8, entry.ordinal);
        }

        PartitionSnapshotDescriptor descriptor;
        descriptor.snapshot_id = make_uuid();
        descriptor.partition_id = partition;
        descriptor.entry_count = input.size();
        descriptor.capacity = capacity;
        descriptor.file_size = total_size;
        descriptor.body_crc32c = crc32c(
            std::span<const std::byte>(bytes).subspan(
                static_cast<std::size_t>(kPartitionedHeaderSize)));
        PartitionSnapshotHeaderV1 header;
        header.collection_id = collection_id_;
        header.schema_id = schema_id_;
        header.descriptor = descriptor;
        header.partition_bits = options_.partition_bits;
        header.hash_seed = kHashSeed;
        header.lookup_slots_offset = lookup_offset;
        header.sequence_slots_offset = sequence_offset;
        const auto encoded = encode_partition_snapshot_header_v1(header);
        std::copy(encoded.begin(), encoded.end(), bytes.begin());

        const std::filesystem::path path = snapshot_path(root_, descriptor);
        FileDescriptor fd = create_exclusive(path);
        write_all(fd.get(), bytes.data(), bytes.size(), path);
        sync_file(fd.get(), path);
        return descriptor;
    }

    std::filesystem::path root_;
    NativeBuilderOptions options_;
    DTypeCode dtype_{DTypeCode::Invalid};
    std::uint32_t item_size_{0};
    std::uint64_t record_size_{0};
    std::uint64_t rows_per_segment_{0};
    Uuid128 collection_id_;
    Uuid128 schema_id_;
    Uuid128 wal_id_;
    std::uint64_t partition_count_{0};
    std::vector<std::vector<IndexSlot>> buckets_;
    std::vector<PartitionedSegmentDescriptor> segments_;
    AdvisoryLockFile writer_lock_;

    FileDescriptor data_fd_;
    FileDescriptor ids_fd_;
    std::filesystem::path data_path_;
    std::filesystem::path ids_path_;
    Uuid128 active_segment_uuid_;
    std::uint32_t active_segment_id_{0};
    std::uint64_t active_rows_{0};
    Crc32c data_crc_;
    Crc32c ids_crc_;
    std::uint64_t count_{0};
    bool finished_{false};
    bool poisoned_{false};
    const pid_t owner_pid_{::getpid()};
};

class PartitionedIndexReader::Impl {
public:
    Impl(
        std::filesystem::path root,
        const ReaderOptions& options,
        const bool writable,
        const bool calibration)
        : root_(std::move(root)), writable_(writable) {
        const auto total_started = SteadyClock::now();
        auto phase_started = SteadyClock::now();
        manifest_ = read_partitioned_manifest(root_);
        open_stats_.manifest_seconds = elapsed_seconds(phase_started);
        if (options.max_read_range_bytes < manifest_.record_size) {
            throw InvalidArgumentError(
                "max_read_range_bytes must fit at least one record");
        }
        if (writable_ || calibration) {
            writer_lock_ = AdvisoryLockFile::open_existing(
                writer_lock_path(root_), true);
            writer_lock_.lock(FileLockMode::Exclusive, true);
        }
        for (const char* child : {
                 "segments", "index", "index/partitions", "wal", "catalogs",
                 "locks"}) {
            require_regular_directory(root_ / child);
        }

        commit_lock_ = AdvisoryLockFile::open_existing(
            commit_lock_path(root_), writable_);
        view_lock_ = AdvisoryLockFile::open_existing(view_lock_path(root_), false);
        commit_lock_.lock(
            writable_ ? FileLockMode::Exclusive : FileLockMode::Shared);
        try {
            view_lock_.lock(FileLockMode::Shared);
            open_catalog();

            phase_started = SteadyClock::now();
            // view.lock protects pathnames only while immutable inodes are
            // pinned. Data descriptors remain open; every snapshot is mmaped
            // and its descriptor is closed immediately. Potentially large
            // checksums and structural scans happen after releasing both
            // publication locks.
            pin_segments(options.verify_native_index);
            pin_snapshots();
            open_and_replay_wals();
            view_lock_.unlock();
            commit_lock_.unlock();
            open_stats_.index_header_seconds = elapsed_seconds(phase_started);

            if (options.verify_native_index) {
                verify_segment_payloads();
            }
            auto validation_started = SteadyClock::now();
            verify_snapshot_checksums();
            open_stats_.index_checksum_seconds =
                elapsed_seconds(validation_started);
            if (options.verify_native_index) {
                validation_started = SteadyClock::now();
                validate_snapshot_structure();
                open_stats_.index_structural_validation_seconds =
                    elapsed_seconds(validation_started);
                open_stats_.index_structural_validation_performed = true;
            }
            open_stats_.index_seconds = elapsed_seconds(phase_started);
            open_stats_.index_body_checksum_present = true;
            open_stats_.index_body_checksum_verified = true;
        } catch (...) {
            if (commit_lock_.locked()) {
                try { commit_lock_.unlock(); } catch (...) {}
            }
            if (view_lock_.locked()) {
                try { view_lock_.unlock(); } catch (...) {}
            }
            throw;
        }
        open_stats_.total_seconds = elapsed_seconds(total_started);
    }

    template <bool Profiled>
    void get_into_impl(
        const PhysicalReader& reader,
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output,
        NativeReadStats* stats) const {
        SteadyClock::time_point total_started;
        if constexpr (Profiled) {
            total_started = SteadyClock::now();
            stats->requested_rows = ids.size();
        }
        ensure_healthy();
        const std::uint64_t expected = checked_multiply(
            ids.size(), manifest_.record_size, "get output");
        if (expected > std::numeric_limits<std::size_t>::max()
            || output.size() != static_cast<std::size_t>(expected)) {
            throw InvalidArgumentError(
                "output byte count does not equal ids.size() * record_size");
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
        std::vector<PhysicalReadRequest> requests;
        requests.reserve(ids.size());
        std::shared_ptr<WalResourceSet> wal_pin;
        const bool zero_delta_fast_path =
            !mutation_read_path_active_.load(std::memory_order_acquire);
        if constexpr (Profiled) {
            stats->zero_delta_fast_path = zero_delta_fast_path;
        }
        if (zero_delta_fast_path) {
            for (std::size_t output_index = 0;
                 output_index != ids.size(); ++output_index) {
                const std::optional<RowRef> row =
                    snapshot_lookup(ids[output_index]);
                if (!row.has_value()) {
                    throw MissingIdError(ids[output_index]);
                }
                requests.push_back(resolve_base(
                    *row, static_cast<std::uint64_t>(output_index)));
            }
            if constexpr (Profiled) {
                stats->base_rows = ids.size();
            }
        } else {
            std::shared_lock state(state_mutex_);
            wal_pin = wal_resources_;
            for (std::size_t output_index = 0;
                 output_index != ids.size(); ++output_index) {
                const std::optional<RowRef> row =
                    logical_lookup_unlocked(ids[output_index]);
                if (!row.has_value()) {
                    throw MissingIdError(ids[output_index]);
                }
                const bool mutation =
                    wal_pin->source_indices.contains(
                        row->segment_id().value());
                if constexpr (Profiled) {
                    mutation ? ++stats->mutation_rows : ++stats->base_rows;
                }
                requests.push_back(resolve_unlocked(
                    *row, static_cast<std::uint64_t>(output_index),
                    *wal_pin));
            }
        }
        if constexpr (Profiled) {
            stats->lookup_seconds = elapsed_seconds(lookup_started);
            const PhysicalReadStats physical = reader.execute_profiled(
                manifest_.record_size, requests, output);
            stats->physical_plan_seconds = physical.plan_seconds;
            stats->physical_backend_seconds = physical.backend_seconds;
            stats->planned_ranges = physical.range_count;
            stats->planned_read_bytes = physical.planned_read_bytes;
            stats->backend_tasks = physical.task_count;
            stats->total_seconds = elapsed_seconds(total_started);
        } else {
            reader.execute(manifest_.record_size, requests, output);
        }
    }

    [[nodiscard]] NativeIdSample sample_ids(
        const std::uint64_t requested,
        const std::uint64_t seed,
        const std::uint64_t max_inspected_slots) const {
        ensure_healthy();
        NativeIdSample result;
        if (requested == 0) {
            result.complete = true;
            return result;
        }
        if (requested > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("requested sample does not fit memory");
        }
        result.ids.reserve(static_cast<std::size_t>(requested));
        const bool mutation_path =
            mutation_read_path_active_.load(std::memory_order_acquire);
        std::shared_lock state(state_mutex_, std::defer_lock);
        if (mutation_path) {
            state.lock();
        }
        std::vector<std::pair<std::uint64_t, const VersionedMutationValue*>>
            overlay_entries;
        std::unordered_set<std::uint64_t> shadowed;
        if (mutation_path) {
            overlay_entries.reserve(overlay_.size());
            shadowed.reserve(overlay_.size());
            for (const auto& [id, value] : overlay_) {
                if (!value.has_value()) {
                    continue;
                }
                overlay_entries.emplace_back(id, &*value);
                shadowed.insert(id);
            }
            std::sort(
                overlay_entries.begin(), overlay_entries.end(),
                [seed](const auto& left, const auto& right) {
                    const std::uint64_t left_hash = hash_id(left.first, seed);
                    const std::uint64_t right_hash = hash_id(right.first, seed);
                    return left_hash != right_hash
                               ? left_hash < right_hash
                               : left.first < right.first;
                });
            for (const auto& [id, value] : overlay_entries) {
                if (result.inspected_slots == max_inspected_slots
                    || result.ids.size() == requested) {
                    break;
                }
                ++result.inspected_slots;
                if (value->is_put()) {
                    result.ids.push_back(id);
                }
            }
            if (result.ids.size() == requested) {
                result.complete = true;
                return result;
            }
        }
        const std::size_t partitions = snapshots_.size();
        std::vector<std::uint64_t> inspected(partitions, 0);
        std::vector<std::uint64_t> starts(partitions, 0);
        std::uint64_t total_capacity = 0;
        for (std::size_t index = 0; index != partitions; ++index) {
            const std::uint64_t capacity =
                snapshots_[index].header.descriptor.capacity;
            total_capacity = checked_sum(
                total_capacity, capacity, "partition sample capacity");
            starts[index] = hash_id(
                seed ^ static_cast<std::uint64_t>(index),
                0xd1b54a32d192ed03ULL) & (capacity - 1);
        }
        const std::uint64_t remaining_limit =
            max_inspected_slots - result.inspected_slots;
        const std::uint64_t limit = std::min(remaining_limit, total_capacity);
        std::size_t partition = static_cast<std::size_t>(seed % partitions);
        std::uint64_t base_inspected = 0;
        while (base_inspected != limit
               && result.ids.size() != requested) {
            while (inspected[partition]
                   == snapshots_[partition].header.descriptor.capacity) {
                partition = (partition + 1) % partitions;
            }
            const Snapshot& snapshot = snapshots_[partition];
            const std::uint64_t slot =
                (starts[partition] + inspected[partition])
                & (snapshot.header.descriptor.capacity - 1);
            ++inspected[partition];
            ++base_inspected;
            ++result.inspected_slots;
            if (control(snapshot, slot) == 1) {
                const std::uint64_t id = slot_id(snapshot, slot);
                if (!mutation_path || !shadowed.contains(id)) {
                    result.ids.push_back(id);
                }
            }
            partition = (partition + 1) % partitions;
        }
        result.complete = result.ids.size() == requested;
        return result;
    }

    void put(
        const PartitionedPutMode mode,
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
        std::uint64_t resulting_count =
            live_count_.load(std::memory_order_acquire);
        {
            std::unique_lock state(state_mutex_);
            for (const std::uint64_t id : ids) {
                const bool exists = logical_lookup_unlocked(id).has_value();
                if (mode == PartitionedPutMode::Insert && exists) {
                    throw ExistingIdError(id);
                }
                if (mode == PartitionedPutMode::Update && !exists) {
                    throw MissingIdError(id);
                }
                if (!exists) {
                    if (resulting_count
                        == std::numeric_limits<std::uint64_t>::max()) {
                        throw InvalidArgumentError(
                            "live record count overflows uint64");
                    }
                    ++resulting_count;
                }
            }
        }
        if (ids.size() > RowRef::kMaximumRowsPerSegment
                             - active_wal().payload_offsets.size()) {
            if (!rotate_wal_locked()) {
                throw InvalidArgumentError(
                    "active WAL exceeds the 2^32 PUT-row limit");
            }
        }
        const std::uint64_t original_rows =
            active_wal().payload_offsets.size();
        {
            std::unique_lock state(state_mutex_);
            prepare_publication_capacity(ids.size(), ids.size());
            for (const std::uint64_t id : ids) {
                prime_overlay_node_unlocked(id);
            }
        }

        std::vector<WalMutation> operations;
        operations.reserve(ids.size());
        for (std::size_t index = 0; index != ids.size(); ++index) {
            operations.push_back(WalMutation{
                ids[index], WalOperationKind::Put,
                embeddings.subspan(
                    index * static_cast<std::size_t>(manifest_.record_size),
                    static_cast<std::size_t>(manifest_.record_size))});
        }
        commit_operations(operations, original_rows, resulting_count);
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
        std::vector<WalMutation> operations;
        operations.reserve(ids.size());
        std::uint64_t resulting_count =
            live_count_.load(std::memory_order_acquire);
        {
            std::unique_lock state(state_mutex_);
            for (std::size_t index = 0; index != ids.size(); ++index) {
                if (!logical_lookup_unlocked(ids[index]).has_value()) {
                    continue;
                }
                deleted[index] = 1;
                --resulting_count;
                operations.push_back(WalMutation{
                    ids[index], WalOperationKind::Delete, {}});
            }
            if (!operations.empty()) {
                prepare_publication_capacity(operations.size(), 0);
                for (const WalMutation& operation : operations) {
                    prime_overlay_node_unlocked(operation.id);
                }
            }
        }
        if (!operations.empty()) {
            commit_operations(
                operations, active_wal().payload_offsets.size(),
                resulting_count);
        }
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
        return mutation_log_stats_locked(rotate_wal_locked());
    }

    [[nodiscard]] std::uint64_t count() const {
        ensure_healthy();
        return live_count_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint32_t dimension() const {
        ensure_owner(); return manifest_.schema.dimension;
    }
    [[nodiscard]] DType dtype() const {
        ensure_owner(); return public_dtype(manifest_.schema.dtype);
    }
    [[nodiscard]] std::uint64_t record_size() const {
        ensure_owner(); return manifest_.record_size;
    }
    [[nodiscard]] bool writable() const {
        ensure_owner(); return writable_;
    }
    [[nodiscard]] NativeOpenStats open_stats() const {
        ensure_owner(); return open_stats_;
    }
    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }
    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        for (Segment& segment : segments_) {
            segment.fd.reset();
        }
        if (wal_resources_ != nullptr) {
            for (const auto& source : wal_resources_->sources) {
                source->fd.reset();
            }
        }
        writer_lock_.abandon_after_fork();
        commit_lock_.abandon_after_fork();
        view_lock_.abandon_after_fork();
    }

private:
    struct Segment {
        PartitionedSegmentDataHeaderV1 header;
        FileDescriptor fd;
        std::filesystem::path data_path;
        PartitionedSegmentIdsHeaderV1 ids_header;
        ReadOnlyMapping ids_mapping;
        std::filesystem::path ids_path;
    };
    struct Snapshot {
        PartitionSnapshotHeaderV1 header;
        ReadOnlyMapping mapping;
    };
    struct WalSource {
        MutationWalEntry entry;
        WalHeader header;
        FileDescriptor fd;
        std::filesystem::path path;
        std::vector<std::uint64_t> payload_offsets;
        std::uint64_t valid_end{kWalFileHeaderSize};
        std::uint64_t next_sequence{1};
    };
    struct WalResourceSet {
        std::vector<std::shared_ptr<WalSource>> sources;
        std::unordered_map<std::uint32_t, std::size_t> source_indices;
        std::size_t active_index{0};
    };
    using OverlayValue = std::optional<VersionedMutationValue>;

    void ensure_owner() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "partitioned collection was inherited across fork; reopen it "
                "in the child process");
        }
    }

    void ensure_healthy() const {
        ensure_owner();
        if (poisoned_.load(std::memory_order_acquire)) {
            throw StorageError(
                "partitioned collection requires reopen after a failed WAL "
                "write or catalog publication");
        }
    }

    void ensure_mutable() const {
        ensure_owner();
        if (!writable_) {
            throw InvalidArgumentError(
                "collection was opened read-only; use writable mode for "
                "mutations");
        }
        ensure_healthy();
    }

    void open_catalog() {
        const std::vector<std::byte> current_bytes =
            read_exact_file(
                partitioned_current_path(root_), kPartitionedHeaderSize);
        const PartitionedCurrentV1 current =
            decode_partitioned_current_v1(current_bytes, manifest_);
        if (current.catalog_file_size > kPartitionedMaximumCatalogFileSize) {
            throw CorruptionError(
                "partitioned CURRENT catalog size exceeds the hard limit");
        }
        const std::vector<std::byte> catalog_bytes = read_exact_file(
            catalog_path(root_, current.catalog_id), current.catalog_file_size);
        if (crc32c(catalog_bytes) != current.catalog_crc32c) {
            throw CorruptionError(
                "partitioned catalog checksum does not match CURRENT");
        }
        catalog_ = decode_partitioned_catalog_v3(catalog_bytes);
        if (catalog_.collection_id != manifest_.collection_id
            || catalog_.schema_id != manifest_.schema_id
            || catalog_.generation != current.generation
            || catalog_.catalog_id != current.catalog_id
            || catalog_.partition_bits != manifest_.partition_bits
            || catalog_.hash_seed != manifest_.hash_seed
            || catalog_.indexed_live_count != manifest_.initial_record_count) {
            throw CorruptionError(
                "partitioned catalog does not match manifest or CURRENT");
        }
        if (!catalog_.runs.empty()
            || catalog_.indexed_sequence != 0
            || !catalog_.terminal_wal_id.empty()
            || catalog_.next_index_epoch != 1
            || (manifest_.required_feature_bits
                & kNativeRequiredFeatureDeltaV1) != 0) {
            throw CorruptionError(
                "format-7 WAL slice cannot interpret immutable index runs");
        }
        std::uint64_t initial_rows = 0;
        for (std::size_t index = 0; index != catalog_.segments.size(); ++index) {
            const PartitionedSegmentDescriptor& descriptor =
                catalog_.segments[index];
            if (descriptor.kind != PartitionedSegmentKind::Initial
                || descriptor.minimum_sequence != 0
                || descriptor.maximum_sequence != 0
                || descriptor.creation_epoch != 0
                || descriptor.row_ref_segment_id != index + 1) {
                throw CorruptionError(
                    "format-7 initial catalog contains a non-initial segment");
            }
            initial_rows = checked_sum(
                initial_rows, descriptor.row_count,
                "format-7 initial segment rows");
        }
        for (const PartitionSnapshotDescriptor& descriptor :
             catalog_.snapshots) {
            if (descriptor.epoch != 0) {
                throw CorruptionError(
                    "format-7 initial catalog contains a folded snapshot");
            }
        }
        if (catalog_.wal_entries.empty()) {
            throw CorruptionError("format-7 catalog has no active WAL");
        }
        const MutationWalEntry& entry = catalog_.wal_entries.front();
        if (initial_rows != manifest_.initial_record_count
            || entry.wal_id != manifest_.initial_active_wal_id
            || !entry.predecessor_wal_id.empty()
            || entry.segment_id != manifest_.initial_active_wal_segment_id
            || entry.previous_sequence != 0 || entry.first_sequence != 1
            || catalog_.next_segment_id < manifest_.initial_next_segment_id) {
            throw CorruptionError(
                "format-7 initial catalog lineage does not match manifest");
        }
    }

    void open_and_replay_wals() {
        const auto started = SteadyClock::now();
        live_count_.store(
            manifest_.initial_record_count, std::memory_order_relaxed);
        wal_resources_ = std::make_shared<WalResourceSet>();
        wal_resources_->sources.reserve(catalog_.wal_entries.size());
        wal_resources_->source_indices.reserve(catalog_.wal_entries.size());
        overlay_.reserve(catalog_.wal_entries.size());

        for (const MutationWalEntry& entry : catalog_.wal_entries) {
            const bool active = entry.state == MutationWalState::Active;
            auto source = std::make_shared<WalSource>();
            source->entry = entry;
            source->header = partitioned_wal_header_for(manifest_, entry);
            source->path = wal_path(root_, entry.wal_id);
            source->fd = open_regular(source->path, writable_ && active);
            if (!active
                && file_size(source->fd.get(), source->path)
                       != entry.sealed_file_size) {
                throw CorruptionError(
                    "frozen partitioned WAL size disagrees with catalog");
            }
            const WalScanResult scan = scan_wal_file(
                source->fd.get(), source->path,
                ExpectedWalIdentity::from_trusted_header(source->header),
                active ? WalScanMode::Active : WalScanMode::Frozen,
                [this, &source](const RecoveredWalBatch& batch) {
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
                        "frozen partitioned WAL contents disagree with catalog");
                }
            } else if (writable_ && scan.ignored_tail_bytes != 0) {
                truncate_file(source->fd.get(), scan.valid_end, source->path);
                sync_file(source->fd.get(), source->path);
            }
            const std::size_t source_index =
                wal_resources_->sources.size();
            wal_resources_->sources.push_back(std::move(source));
            if (!wal_resources_->source_indices.emplace(
                     entry.segment_id, source_index).second) {
                throw CorruptionError(
                    "partitioned WAL segment ID is duplicated");
            }
        }
        if (wal_resources_->sources.empty()
            || wal_resources_->sources.back()->entry.state
                   != MutationWalState::Active) {
            throw CorruptionError(
                "partitioned WAL chain lacks a final active WAL");
        }
        wal_resources_->active_index = wal_resources_->sources.size() - 1;
        append_offset_ = active_wal().valid_end;
        next_sequence_ = active_wal().next_sequence;
        mutation_read_path_active_.store(
            !overlay_.empty(), std::memory_order_relaxed);
        open_stats_.wal_replay_seconds = elapsed_seconds(started);
    }

    void pin_segments(const bool retain_ids_for_verification) {
        const auto started = SteadyClock::now();
        segments_.reserve(catalog_.segments.size());
        for (const PartitionedSegmentDescriptor& descriptor :
             catalog_.segments) {
            const std::filesystem::path data_path =
                segment_data_path(root_, descriptor.segment_id);
            FileDescriptor data_fd = open_regular(data_path);
            if (file_size(data_fd.get(), data_path)
                != descriptor.data_file_size) {
                throw CorruptionError("partitioned data segment size changed");
            }
            const PartitionedSegmentDataHeaderV1 data_header =
                decode_partitioned_segment_data_header_v1(
                    read_header(data_fd.get(), data_path));
            if (data_header.collection_id != manifest_.collection_id
                || data_header.schema_id != manifest_.schema_id
                || data_header.descriptor != descriptor
                || data_header.schema != manifest_.schema
                || data_header.record_size != manifest_.record_size) {
                throw CorruptionError(
                    "partitioned data segment does not match catalog");
            }

            const std::filesystem::path ids_path =
                segment_ids_path(root_, descriptor.segment_id);
            FileDescriptor ids_fd = open_regular(ids_path);
            if (file_size(ids_fd.get(), ids_path) != descriptor.ids_file_size) {
                throw CorruptionError("partitioned IDs segment size changed");
            }
            const PartitionedSegmentIdsHeaderV1 ids_header =
                decode_partitioned_segment_ids_header_v1(
                    read_header(ids_fd.get(), ids_path));
            if (ids_header.collection_id != manifest_.collection_id
                || ids_header.schema_id != manifest_.schema_id
                || ids_header.descriptor != descriptor) {
                throw CorruptionError(
                    "partitioned IDs segment does not match catalog");
            }
            if (!segment_indices_.emplace(
                    descriptor.row_ref_segment_id, segments_.size()).second) {
                throw CorruptionError("partitioned segment ID is duplicated");
            }
            ReadOnlyMapping ids_mapping;
            if (retain_ids_for_verification) {
                ids_mapping = ReadOnlyMapping(
                    ids_fd.get(), descriptor.ids_file_size, ids_path);
            }
            segments_.push_back(Segment{
                data_header,
                std::move(data_fd),
                data_path,
                ids_header,
                std::move(ids_mapping),
                ids_path});
        }
        open_stats_.base_segments_seconds = elapsed_seconds(started);
    }

    void pin_snapshots() {
        snapshots_.reserve(catalog_.snapshots.size());
        for (const PartitionSnapshotDescriptor& descriptor :
             catalog_.snapshots) {
            const std::filesystem::path path = snapshot_path(root_, descriptor);
            FileDescriptor fd = open_regular(path);
            if (file_size(fd.get(), path) != descriptor.file_size) {
                throw CorruptionError("partition snapshot size changed");
            }
            ReadOnlyMapping mapping(fd.get(), descriptor.file_size, path);
            // Close immediately: the mapping pins this exact immutable inode,
            // keeping descriptor usage independent of partition count.
            fd.reset();
            const std::span<const std::byte> bytes = mapping.bytes();
            const PartitionSnapshotHeaderV1 header =
                decode_partition_snapshot_header_v1(
                    bytes.first(static_cast<std::size_t>(
                        kPartitionedHeaderSize)));
            if (header.collection_id != manifest_.collection_id
                || header.schema_id != manifest_.schema_id
                || header.descriptor != descriptor
                || header.partition_bits != manifest_.partition_bits
                || header.hash_seed != manifest_.hash_seed) {
                throw CorruptionError(
                    "partition snapshot does not match catalog");
            }
            open_stats_.index_file_bytes = checked_sum(
                open_stats_.index_file_bytes, descriptor.file_size,
                "partition snapshot bytes");
            snapshots_.push_back(Snapshot{header, std::move(mapping)});
        }
    }

    void verify_segment_payloads() const {
        for (const Segment& segment : segments_) {
            if (segment.ids_mapping.bytes().empty()) {
                throw CorruptionError(
                    "partitioned IDs sidecar was not pinned for verification");
            }
            if (checksum_payload(
                    segment.fd.get(), segment.data_path,
                    segment.header.payload_offset,
                    segment.header.payload_size)
                    != segment.header.descriptor.data_payload_crc32c
                || crc32c(segment.ids_mapping.bytes().subspan(
                       static_cast<std::size_t>(
                           segment.ids_header.payload_offset),
                       static_cast<std::size_t>(
                           segment.ids_header.payload_size)))
                       != segment.header.descriptor.ids_payload_crc32c) {
                throw CorruptionError(
                    "partitioned segment payload checksum mismatch");
            }
        }
    }

    [[nodiscard]] static std::uint8_t control(
        const Snapshot& snapshot,
        const std::uint64_t slot) noexcept {
        return std::to_integer<std::uint8_t>(snapshot.mapping.bytes()[
            static_cast<std::size_t>(
                snapshot.header.controls_offset + slot)]);
    }
    [[nodiscard]] static std::uint64_t slot_id(
        const Snapshot& snapshot,
        const std::uint64_t slot) noexcept {
        return get_u64(
            snapshot.mapping.bytes(), static_cast<std::size_t>(
                snapshot.header.lookup_slots_offset
                + slot * kPartitionSnapshotLookupSlotSize));
    }
    [[nodiscard]] static std::uint64_t slot_value(
        const Snapshot& snapshot,
        const std::uint64_t slot) noexcept {
        return get_u64(
            snapshot.mapping.bytes(), static_cast<std::size_t>(
                snapshot.header.lookup_slots_offset
                + slot * kPartitionSnapshotLookupSlotSize + 8));
    }
    [[nodiscard]] static std::uint64_t slot_sequence(
        const Snapshot& snapshot,
        const std::uint64_t slot) noexcept {
        return get_u64(
            snapshot.mapping.bytes(), static_cast<std::size_t>(
                snapshot.header.sequence_slots_offset
                + slot * kPartitionSnapshotSequenceSlotSize));
    }

    void verify_snapshot_checksums() const {
        for (const Snapshot& snapshot : snapshots_) {
            if (crc32c(snapshot.mapping.bytes().subspan(
                    static_cast<std::size_t>(kPartitionedHeaderSize)))
                != snapshot.header.descriptor.body_crc32c) {
                throw CorruptionError("partition snapshot checksum mismatch");
            }
        }
    }

    void validate_snapshot_structure() {
        std::vector<std::vector<std::uint8_t>> seen_rows;
        seen_rows.reserve(segments_.size());
        for (const Segment& segment : segments_) {
            const std::uint64_t bytes =
                (segment.header.descriptor.row_count + 7) / 8;
            if (bytes > std::numeric_limits<std::size_t>::max()) {
                throw CorruptionError("segment RowRef bitmap does not fit memory");
            }
            seen_rows.emplace_back(static_cast<std::size_t>(bytes));
        }
        std::uint64_t occupied_total = 0;
        for (const Snapshot& snapshot : snapshots_) {
            const std::uint64_t capacity = snapshot.header.descriptor.capacity;
            const std::uint64_t mask = capacity - 1;
            std::uint64_t occupied = 0;
            for (std::uint64_t slot = 0; slot != capacity; ++slot) {
                const std::uint8_t state = control(snapshot, slot);
                if (state > 1) {
                    throw CorruptionError(
                        "partition snapshot has invalid control byte");
                }
                const std::uint64_t id = slot_id(snapshot, slot);
                const std::uint64_t raw = slot_value(snapshot, slot);
                const std::uint64_t sequence = slot_sequence(snapshot, slot);
                if (state == 0) {
                    if (id != 0 || raw != 0 || sequence != 0) {
                        throw CorruptionError(
                            "empty partition slot contains nonzero data");
                    }
                    continue;
                }
                ++occupied;
                ++occupied_total;
                if (sequence != 0) {
                    throw CorruptionError(
                        "initial partition snapshot contains mutation sequence");
                }
                const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
                if (partition_for_hash(hash, manifest_.partition_bits)
                    != snapshot.header.descriptor.partition_id) {
                    throw CorruptionError(
                        "partition snapshot contains an ID in the wrong partition");
                }
                const RowRef row = RowRef::decode(raw);
                const auto segment_position = segment_indices_.find(
                    row.segment_id().value());
                if (segment_position == segment_indices_.end()) {
                    throw CorruptionError(
                        "partition snapshot RowRef has no segment");
                }
                const std::size_t segment_index = segment_position->second;
                const Segment& source_segment = segments_[segment_index];
                if (row.row_index()
                    >= source_segment.header.descriptor.row_count) {
                    throw CorruptionError(
                        "partition snapshot RowRef exceeds its segment");
                }
                if (source_segment.ids_mapping.bytes().empty()) {
                    throw CorruptionError(
                        "full verification lacks a pinned IDs sidecar");
                }
                const std::size_t ids_offset = static_cast<std::size_t>(
                    source_segment.ids_header.payload_offset
                    + static_cast<std::uint64_t>(row.row_index())
                          * kPartitionedSegmentIdEntrySize);
                if (get_u64(
                        source_segment.ids_mapping.bytes(), ids_offset) != id
                    || get_u64(
                           source_segment.ids_mapping.bytes(), ids_offset + 8)
                           != sequence) {
                    throw CorruptionError(
                        "partition snapshot entry disagrees with its IDs sidecar");
                }
                std::vector<std::uint8_t>& bitmap = seen_rows[segment_index];
                const std::size_t byte = row.row_index() / 8;
                const std::uint8_t bit = static_cast<std::uint8_t>(
                    std::uint32_t{1} << (row.row_index() % 8));
                if ((bitmap[byte] & bit) != 0) {
                    throw CorruptionError(
                        "partition snapshot duplicates a physical RowRef");
                }
                bitmap[byte] |= bit;

                std::uint64_t probe = hash & mask;
                bool reached = false;
                while (control(snapshot, probe) != 0) {
                    if (probe == slot) {
                        reached = true;
                        break;
                    }
                    if (slot_id(snapshot, probe) == id) {
                        throw CorruptionError(
                            "partition snapshot contains duplicate ID");
                    }
                    probe = (probe + 1) & mask;
                }
                if (!reached) {
                    throw CorruptionError(
                        "partition snapshot contains probe-unreachable slot");
                }
            }
            if (occupied != snapshot.header.descriptor.entry_count) {
                throw CorruptionError(
                    "partition snapshot occupancy does not match catalog");
            }
        }
        if (occupied_total != catalog_.indexed_live_count
            || occupied_total != manifest_.initial_record_count) {
            throw CorruptionError(
                "partitioned index occupancy does not match collection count");
        }
        for (std::size_t segment_index = 0;
             segment_index != segments_.size(); ++segment_index) {
            const std::uint64_t rows =
                segments_[segment_index].header.descriptor.row_count;
            for (std::uint64_t row = 0; row != rows; ++row) {
                if ((seen_rows[segment_index][static_cast<std::size_t>(row / 8)]
                     & static_cast<std::uint8_t>(
                         std::uint32_t{1} << (row % 8))) == 0) {
                    throw CorruptionError(
                        "partitioned index does not reference every base row");
                }
            }
        }
        // Full verification no longer needs the sidecar inode pins. Releasing
        // the mappings here also guarantees that verification does not leave
        // one extra descriptor or mapping per data segment for handle life.
        for (Segment& segment : segments_) {
            segment.ids_mapping = ReadOnlyMapping{};
        }
    }

    [[nodiscard]] WalSource& active_wal() {
        return *wal_resources_->sources.at(wal_resources_->active_index);
    }

    [[nodiscard]] const WalSource& active_wal() const {
        return *wal_resources_->sources.at(wal_resources_->active_index);
    }

    [[nodiscard]] std::optional<RowRef> snapshot_lookup(
        const std::uint64_t id) const {
        const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
        const std::uint64_t partition = partition_for_hash(
            hash, manifest_.partition_bits);
        const Snapshot& snapshot =
            snapshots_[static_cast<std::size_t>(partition)];
        const std::uint64_t mask = snapshot.header.descriptor.capacity - 1;
        std::uint64_t slot = hash & mask;
        while (control(snapshot, slot) != 0) {
            if (slot_id(snapshot, slot) == id) {
                return RowRef::decode(slot_value(snapshot, slot));
            }
            slot = (slot + 1) & mask;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<RowRef> logical_lookup_unlocked(
        const std::uint64_t id) const {
        const auto position = overlay_.find(id);
        if (position != overlay_.end() && position->second.has_value()) {
            return position->second->row_ref();
        }
        return snapshot_lookup(id);
    }

    [[nodiscard]] PhysicalReadRequest resolve_base(
        const RowRef row,
        const std::uint64_t output_index) const {
        const auto position = segment_indices_.find(row.segment_id().value());
        if (position == segment_indices_.end()) {
            throw CorruptionError("partition RowRef has no pinned segment");
        }
        const Segment& segment = segments_[position->second];
        if (row.row_index() >= segment.header.descriptor.row_count) {
            throw CorruptionError("partition RowRef exceeds pinned segment");
        }
        return PhysicalReadRequest{
            segment.fd.get(),
            segment.header.payload_offset
                + static_cast<std::uint64_t>(row.row_index())
                      * manifest_.record_size,
            output_index};
    }

    [[nodiscard]] PhysicalReadRequest resolve_unlocked(
        const RowRef row,
        const std::uint64_t output_index,
        const WalResourceSet& resources) const {
        if (segment_indices_.contains(row.segment_id().value())) {
            return resolve_base(row, output_index);
        }
        const auto source_position = resources.source_indices.find(
            row.segment_id().value());
        if (source_position == resources.source_indices.end()) {
            throw CorruptionError(
                "partitioned mutation RowRef has no pinned WAL");
        }
        const WalSource& source =
            *resources.sources[source_position->second];
        if (row.row_index() >= source.payload_offsets.size()) {
            throw CorruptionError(
                "partitioned mutation RowRef exceeds WAL PUT rows");
        }
        return PhysicalReadRequest{
            source.fd.get(),
            source.payload_offsets[static_cast<std::size_t>(row.row_index())],
            output_index};
    }

    void apply_recovered_batch(
        WalSource& source,
        const RecoveredWalBatch& batch) {
        for (const RecoveredWalOperation& operation : batch.operations) {
            const auto prior = overlay_.find(operation.id);
            if (prior != overlay_.end() && prior->second.has_value()
                && prior->second->sequence() >= batch.sequence) {
                throw CorruptionError(
                    "partitioned WAL versions are not strictly chronological");
            }
            const bool existed =
                prior != overlay_.end() && prior->second.has_value()
                    ? prior->second->is_put()
                    : snapshot_lookup(operation.id).has_value();
            if (operation.kind == WalOperationKind::Put) {
                if (operation.payload_size != manifest_.record_size) {
                    throw CorruptionError(
                        "partitioned WAL PUT has unexpected payload size");
                }
                if (source.payload_offsets.size()
                    >= RowRef::kMaximumRowsPerSegment) {
                    throw CorruptionError(
                        "partitioned WAL contains more than 2^32 PUT rows");
                }
                const RowRef row = RowRef::create(
                    source.entry.segment_id,
                    source.payload_offsets.size());
                source.payload_offsets.push_back(operation.payload_offset);
                overlay_.insert_or_assign(
                    operation.id,
                    VersionedMutationValue::put(batch.sequence, row));
                if (!existed) {
                    const std::uint64_t current =
                        live_count_.load(std::memory_order_relaxed);
                    if (current == std::numeric_limits<std::uint64_t>::max()) {
                        throw CorruptionError(
                            "partitioned WAL live count overflows uint64");
                    }
                    live_count_.store(current + 1, std::memory_order_relaxed);
                }
            } else {
                overlay_.insert_or_assign(
                    operation.id,
                    VersionedMutationValue::tombstone(batch.sequence));
                if (existed) {
                    const std::uint64_t current =
                        live_count_.load(std::memory_order_relaxed);
                    if (current == 0) {
                        throw CorruptionError(
                            "partitioned WAL live count underflows");
                    }
                    live_count_.store(current - 1, std::memory_order_relaxed);
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
        if (put_count > std::numeric_limits<std::size_t>::max()
                            - active_wal().payload_offsets.size()) {
            throw InvalidArgumentError("WAL RowRef table is too large");
        }
        overlay_.reserve(overlay_.size() + overlay_entries);
        active_wal().payload_offsets.reserve(
            active_wal().payload_offsets.size() + put_count);
    }

    void prime_overlay_node_unlocked(const std::uint64_t id) {
        if (!overlay_.contains(id)) {
            overlay_.emplace(id, std::nullopt);
        }
    }

    void commit_operations(
        const std::span<const WalMutation> operations,
        const std::uint64_t original_wal_rows,
        const std::uint64_t resulting_count) {
        if (next_sequence_
            > VersionedMutationValue::kMaximumCommitSequence) {
            throw StorageError("mutation commit sequence space is exhausted");
        }
        WalSource& active = active_wal();
        const EncodedWalTransaction transaction = encode_wal_transaction(
            active.header, next_sequence_, append_offset_, operations);
        const std::uint64_t relative_payload_offset = get_u64(
            std::span<const std::byte>(transaction.bytes), 64);
        std::uint64_t payload_cursor = append_offset_ + relative_payload_offset;
        std::vector<std::uint64_t> put_offsets;
        put_offsets.reserve(operations.size());
        for (const WalMutation& operation : operations) {
            if (operation.kind == WalOperationKind::Put) {
                put_offsets.push_back(payload_cursor);
                payload_cursor = checked_sum(
                    payload_cursor, manifest_.record_size,
                    "partitioned WAL payload offset");
            }
        }

        try {
            ScopedAdvisoryLock commit(commit_lock_, FileLockMode::Exclusive);
            const std::uint64_t new_end = append_wal_transaction_durable(
                active.fd.get(), active.path, append_offset_, transaction);
            try {
                std::unique_lock state(state_mutex_);
                static_assert(
                    std::is_nothrow_copy_assignable_v<OverlayValue>);
                std::size_t put_index = 0;
                for (const WalMutation& operation : operations) {
                    if (operation.kind == WalOperationKind::Put) {
                        const RowRef row = RowRef::create(
                            active.entry.segment_id,
                            original_wal_rows + put_index);
                        active.payload_offsets.push_back(
                            put_offsets[put_index]);
                        overlay_.insert_or_assign(
                            operation.id,
                            VersionedMutationValue::put(next_sequence_, row));
                        ++put_index;
                    } else {
                        overlay_.insert_or_assign(
                            operation.id,
                            VersionedMutationValue::tombstone(next_sequence_));
                    }
                }
                live_count_.store(resulting_count, std::memory_order_release);
                append_offset_ = new_end;
                ++next_sequence_;
                active.valid_end = append_offset_;
                active.next_sequence = next_sequence_;
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

    [[nodiscard]] NativeMutationLogStats mutation_log_stats_locked(
        const bool rotated) const {
        NativeMutationLogStats stats;
        stats.catalog_managed = true;
        stats.delta_managed = false;
        stats.generation = catalog_.generation;
        stats.committed_sequence = next_sequence_ - 1;
        stats.materialized_sequence = catalog_.indexed_sequence;
        stats.active_wal_bytes = append_offset_;
        stats.active_wal_put_rows = active_wal().payload_offsets.size();
        stats.rotated = rotated;
        for (const auto& source : wal_resources_->sources) {
            if (source->entry.state == MutationWalState::Frozen) {
                ++stats.frozen_wal_count;
            }
            stats.total_wal_bytes = checked_sum(
                stats.total_wal_bytes, source->valid_end,
                "partitioned WAL total bytes");
        }
        return stats;
    }

    [[nodiscard]] bool rotate_wal_locked() {
        WalSource& old_active = active_wal();
        if (next_sequence_ == old_active.header.first_sequence) {
            return false;
        }
        if (catalog_.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("partitioned catalog generation is exhausted");
        }
        if (catalog_.wal_entries.size()
            >= kPartitionedMaximumResourceEntries) {
            throw StorageError("partitioned catalog WAL limit is exhausted");
        }
        if (catalog_.next_segment_id >= SegmentId::kMaximum) {
            throw StorageError("mutation segment ID space is exhausted");
        }
        const std::uint64_t previous_sequence = next_sequence_ - 1;
        if (previous_sequence
            >= VersionedMutationValue::kMaximumCommitSequence) {
            throw StorageError("mutation commit sequence space is exhausted");
        }

        MutationWalEntry frozen = old_active.entry;
        frozen.state = MutationWalState::Frozen;
        frozen.last_sequence = previous_sequence;
        frozen.put_count = old_active.payload_offsets.size();
        frozen.sealed_file_size = append_offset_;

        MutationWalEntry successor;
        successor.wal_id = make_uuid();
        successor.predecessor_wal_id = old_active.entry.wal_id;
        successor.segment_id = catalog_.next_segment_id;
        successor.state = MutationWalState::Active;
        successor.previous_sequence = previous_sequence;
        successor.first_sequence = next_sequence_;

        PartitionedCatalogV3 next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        next_catalog.next_segment_id = successor.segment_id + 1;
        next_catalog.wal_entries.back() = frozen;
        next_catalog.wal_entries.push_back(successor);

        const WalHeader successor_header =
            partitioned_wal_header_for(manifest_, successor);
        const std::filesystem::path successor_path =
            wal_path(root_, successor.wal_id);

        try {
            ScopedAdvisoryLock commit(commit_lock_, FileLockMode::Exclusive);
            if (file_size(old_active.fd.get(), old_active.path)
                != append_offset_) {
                throw CorruptionError(
                    "active partitioned WAL size changed before rotation");
            }
            sync_file(old_active.fd.get(), old_active.path);
            create_wal_file(successor_path, successor_header);
            sync_directory(root_ / "wal");

            auto successor_source = std::make_shared<WalSource>();
            successor_source->entry = successor;
            successor_source->header = successor_header;
            successor_source->fd = open_regular(successor_path, true);
            successor_source->path = successor_path;
            successor_source->valid_end = kWalFileHeaderSize;
            successor_source->next_sequence = successor.first_sequence;

            auto next_resources = std::make_shared<WalResourceSet>();
            {
                std::shared_lock state(state_mutex_);
                next_resources->sources = wal_resources_->sources;
                next_resources->source_indices =
                    wal_resources_->source_indices;
            }
            next_resources->sources.push_back(std::move(successor_source));
            if (!next_resources->source_indices.emplace(
                     successor.segment_id,
                     next_resources->sources.size() - 1).second) {
                throw CorruptionError(
                    "successor partitioned WAL reuses a segment ID");
            }
            next_resources->active_index = next_resources->sources.size() - 1;

            publish_partitioned_catalog(root_, manifest_, next_catalog);
            {
                std::unique_lock state(state_mutex_);
                old_active.entry = frozen;
                wal_resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
                append_offset_ = kWalFileHeaderSize;
                active_wal().valid_end = append_offset_;
                active_wal().next_sequence = next_sequence_;
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        return true;
    }

    std::filesystem::path root_;
    PartitionedManifestV2 manifest_;
    PartitionedCatalogV3 catalog_;
    NativeOpenStats open_stats_;
    AdvisoryLockFile writer_lock_;
    AdvisoryLockFile commit_lock_;
    AdvisoryLockFile view_lock_;
    std::vector<Segment> segments_;
    std::unordered_map<std::uint32_t, std::size_t> segment_indices_;
    std::vector<Snapshot> snapshots_;
    std::shared_ptr<WalResourceSet> wal_resources_;
    mutable std::shared_mutex state_mutex_;
    std::unordered_map<std::uint64_t, OverlayValue> overlay_;
    std::atomic<bool> mutation_read_path_active_{false};
    std::atomic<std::uint64_t> live_count_{0};
    std::uint64_t append_offset_{kWalFileHeaderSize};
    std::uint64_t next_sequence_{1};
    mutable std::mutex mutation_mutex_;
    std::atomic<bool> poisoned_{false};
    bool writable_{false};
    const pid_t owner_pid_{::getpid()};
};

PartitionedIndexBuilder::PartitionedIndexBuilder(
    std::filesystem::path store_path,
    NativeBuilderOptions options)
    : impl_(std::make_unique<Impl>(std::move(store_path), options)) {}
PartitionedIndexBuilder::~PartitionedIndexBuilder() = default;
void PartitionedIndexBuilder::append(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->append(ids, embeddings);
}
void PartitionedIndexBuilder::finish() { impl_->finish(); }
std::uint64_t PartitionedIndexBuilder::count() const { return impl_->count(); }
std::uint32_t PartitionedIndexBuilder::dimension() const {
    return impl_->dimension();
}
DType PartitionedIndexBuilder::dtype() const { return impl_->dtype(); }
std::uint64_t PartitionedIndexBuilder::record_size() const {
    return impl_->record_size();
}
bool PartitionedIndexBuilder::finished() const { return impl_->finished(); }
bool PartitionedIndexBuilder::owned_by_current_process() const noexcept {
    return impl_->owned_by_current_process();
}
void PartitionedIndexBuilder::abandon_after_fork() noexcept {
    impl_->abandon_after_fork();
}

PartitionedIndexReader::PartitionedIndexReader(
    std::filesystem::path store_path,
    const ReaderOptions& reader_options,
    const bool writable,
    const bool calibration)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), reader_options, writable, calibration)) {}
PartitionedIndexReader::~PartitionedIndexReader() = default;
void PartitionedIndexReader::get_into(
    const PhysicalReader& reader,
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    impl_->get_into_impl<false>(reader, ids, output, nullptr);
}
NativeReadStats PartitionedIndexReader::get_into_profiled(
    const PhysicalReader& reader,
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    NativeReadStats stats;
    impl_->get_into_impl<true>(reader, ids, output, &stats);
    return stats;
}
NativeIdSample PartitionedIndexReader::sample_ids(
    const std::uint64_t requested,
    const std::uint64_t seed,
    const std::uint64_t max_inspected_slots) const {
    return impl_->sample_ids(requested, seed, max_inspected_slots);
}
void PartitionedIndexReader::put(
    const PartitionedPutMode mode,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(mode, ids, embeddings);
}
std::vector<std::uint8_t> PartitionedIndexReader::erase(
    const std::span<const std::uint64_t> ids) {
    return impl_->erase(ids);
}
NativeMutationLogStats PartitionedIndexReader::mutation_log_stats() const {
    return impl_->mutation_log_stats();
}
NativeMutationLogStats PartitionedIndexReader::rotate_wal() {
    return impl_->rotate_wal();
}
std::uint64_t PartitionedIndexReader::count() const { return impl_->count(); }
std::uint32_t PartitionedIndexReader::dimension() const {
    return impl_->dimension();
}
DType PartitionedIndexReader::dtype() const { return impl_->dtype(); }
std::uint64_t PartitionedIndexReader::record_size() const {
    return impl_->record_size();
}
bool PartitionedIndexReader::writable() const { return impl_->writable(); }
NativeOpenStats PartitionedIndexReader::open_stats() const {
    return impl_->open_stats();
}
bool PartitionedIndexReader::owned_by_current_process() const noexcept {
    return impl_->owned_by_current_process();
}
void PartitionedIndexReader::abandon_after_fork() noexcept {
    impl_->abandon_after_fork();
}

}  // namespace throwdb::detail
