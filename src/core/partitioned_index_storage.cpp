#include "partitioned_index_storage.hpp"

#include "crc32c.hpp"
#include "file_lock.hpp"
#include "mutation_wal.hpp"
#include "native_collection_test_hooks.hpp"
#include "partition_snapshot_storage.hpp"
#include "partitioned_index_format.hpp"
#include "partitioned_index_merge.hpp"
#include "partitioned_index_run_storage.hpp"
#include "pending_mutation_fence.hpp"
#include "partitioned_segment_storage.hpp"
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
#include <queue>
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
constexpr std::uint64_t kCleanerReadBatchBytes = 8ULL << 20;
constexpr std::uint64_t kCleanerRatioScale = 1'000'000;
// Covers two candidate plans during deterministic comparison and the later
// plan + run writer sort peak, including conservative allocator headroom.
constexpr std::uint64_t kCleanerEstimatedBytesPerRow = 128;
// The run writer holds a decoded TOC and its serialized form at the same
// time, while descriptor/header construction and the reopened reader add
// bitmap and TOC scratch.  Charging 64 bytes per partition is deliberately
// above those simultaneous allocations (two 16-byte TOCs plus bitmaps), and
// the fixed allowance covers vector/control-object allocator overhead.
constexpr std::uint64_t kCleanerEstimatedBytesPerPartition = 64;
constexpr std::uint64_t kCleanerFixedMemoryOverhead = 64ULL << 10;
// A format-7 overlay node contains the ID, optional version/RowRef, hash-node
// linkage and bucket load. This intentionally exceeds the supported LP64
// implementation's measured retained bytes per entry.
constexpr std::uint64_t kEstimatedOverlayBytesPerEntry = 64;
// During materialization one input operation can simultaneously contribute to
// the collapsed hash map, decoded WAL batch, CollapsedRecord vector, pointer
// ordering, run output, and publication erase list. The fixed allowance covers
// the bounded physical-read batch and container/control overhead.
constexpr std::uint64_t kMaterializationEstimatedBytesPerInputEntry = 256;
constexpr std::uint64_t kMaterializationFixedMemoryOverhead = 16ULL << 20;
// ID scans may spend most of a call rejecting stale/deleted physical rows.
// Poll cancellation before the first candidate and then at least this often.
constexpr std::uint64_t kIdScanStopCheckCandidates = 256;
using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t saturating_product(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return left != 0
                   && right > std::numeric_limits<std::uint64_t>::max() / left
               ? std::numeric_limits<std::uint64_t>::max()
               : left * right;
}

[[nodiscard]] std::uint64_t saturating_sum(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

[[nodiscard]] std::uint64_t estimated_overlay_bytes(
    const std::uint64_t entries) noexcept {
    return saturating_product(entries, kEstimatedOverlayBytesPerEntry);
}

[[nodiscard]] std::uint64_t estimated_materialization_peak_memory(
    const std::uint64_t overlay_entries,
    const std::uint64_t input_entries) noexcept {
    return saturating_sum(
        kMaterializationFixedMemoryOverhead,
        saturating_sum(
            estimated_overlay_bytes(overlay_entries),
            saturating_product(
                input_entries,
                kMaterializationEstimatedBytesPerInputEntry)));
}

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
    static constexpr char hex[] = "0123456789abcdef";
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
    const PartitionedCatalogV3& catalog,
    const NativeCollectionTestOperation operation =
        NativeCollectionTestOperation::None) {
    const std::vector<std::byte> bytes =
        encode_partitioned_catalog_v3(catalog);
    const std::filesystem::path destination =
        catalog_path(root, catalog.catalog_id);
    write_atomic(
        root, destination.string() + ".tmp", destination, bytes,
        "partitioned catalog");
    invoke_native_collection_test_hook({
        NativeCollectionTestHookPoint::MutationCatalogDurableBeforeCurrent,
        operation,
        NativeCollectionTestResourceKind::None,
        catalog.generation,
        0});

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
    invoke_native_collection_test_hook({
        NativeCollectionTestHookPoint::MutationCurrentDurable,
        operation,
        NativeCollectionTestResourceKind::None,
        catalog.generation,
        0});
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

[[nodiscard]] bool partition_bitmap_contains(
    const std::span<const std::byte> bitmap,
    const std::uint32_t partition) noexcept {
    const std::size_t byte = partition / 8;
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    return byte < bitmap.size()
           && (std::to_integer<std::uint8_t>(bitmap[byte]) & mask) != 0;
}

[[nodiscard]] std::uint64_t checked_sum(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view description);

[[nodiscard]] std::vector<std::uint64_t> proportional_sample_quotas(
    const std::span<const std::uint64_t> slots,
    const std::uint64_t limit,
    const std::uint64_t seed) {
    std::uint64_t total = 0;
    for (const std::uint64_t count : slots) {
        total = checked_sum(total, count, "ID sample quota slots");
    }
    if (limit > total) {
        throw CorruptionError("ID sample quota exceeds available slots");
    }
    std::vector<std::uint64_t> quotas(slots.size(), 0);
    if (limit == 0) {
        return quotas;
    }

    struct Remainder {
        std::size_t index{0};
        std::uint64_t value{0};
        std::uint64_t tie_breaker{0};
    };
    std::vector<Remainder> remainders;
    remainders.reserve(slots.size());
    std::uint64_t allocated = 0;
    for (std::size_t index = 0; index != slots.size(); ++index) {
        using Wide = __uint128_t;
        const Wide product = static_cast<Wide>(limit)
                             * static_cast<Wide>(slots[index]);
        quotas[index] = static_cast<std::uint64_t>(
            product / static_cast<Wide>(total));
        const std::uint64_t remainder = static_cast<std::uint64_t>(
            product % static_cast<Wide>(total));
        allocated = checked_sum(
            allocated, quotas[index], "ID sample allocated quota");
        remainders.push_back(Remainder{
            index,
            remainder,
            hash_id(
                static_cast<std::uint64_t>(index),
                seed ^ 0x9e3779b97f4a7c15ULL)});
    }
    std::sort(
        remainders.begin(), remainders.end(),
        [](const Remainder& left, const Remainder& right) {
            if (left.value != right.value) {
                return left.value > right.value;
            }
            if (left.tie_breaker != right.tie_breaker) {
                return left.tie_breaker < right.tie_breaker;
            }
            return left.index < right.index;
        });
    std::uint64_t remaining = limit - allocated;
    for (const Remainder& remainder : remainders) {
        if (remaining == 0) {
            break;
        }
        if (quotas[remainder.index] == slots[remainder.index]) {
            continue;
        }
        ++quotas[remainder.index];
        --remaining;
    }
    if (remaining != 0) {
        throw CorruptionError("ID sample quota remainder was not allocated");
    }
    return quotas;
}

template <typename Visitor>
void visit_stratified_sample_positions(
    const std::uint64_t total,
    const std::uint64_t count,
    const std::uint64_t seed,
    Visitor&& visitor) {
    if (count > total) {
        throw CorruptionError("stratified ID sample exceeds its source");
    }
    if (count == 0) {
        return;
    }
    using Wide = __uint128_t;
    for (std::uint64_t index = 0; index != count; ++index) {
        const std::uint64_t begin = static_cast<std::uint64_t>(
            static_cast<Wide>(index) * static_cast<Wide>(total)
            / static_cast<Wide>(count));
        const std::uint64_t end = static_cast<std::uint64_t>(
            static_cast<Wide>(index + 1) * static_cast<Wide>(total)
            / static_cast<Wide>(count));
        const std::uint64_t width = end - begin;
        if (width == 0) {
            throw CorruptionError("stratified ID sample cell is empty");
        }
        const std::uint64_t offset = hash_id(
            index, seed ^ 0xd1b54a32d192ed03ULL) % width;
        visitor(begin + offset);
    }
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
                 std::filesystem::path("index/runs"),
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
        const bool calibration,
        const NativeOpenOptions open_options)
        : root_(std::move(root)),
          open_options_(open_options),
          writable_(writable) {
        const auto total_started = SteadyClock::now();
        auto phase_started = SteadyClock::now();
        if (open_options_.max_wal_bytes == 0
            || open_options_.max_wal_operations == 0
            || open_options_.max_overlay_entries == 0
            || open_options_.max_overlay_bytes == 0) {
            throw InvalidArgumentError(
                "native mutable-open safety limits must be positive");
        }
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
        if (writable_ && !open_options_.allow_pending_mutation_recovery) {
            ensure_pending_mutation_fence_absent(
                root_, "writable collection open");
        }
        for (const char* child : {
                 "segments", "index", "index/partitions", "index/runs",
                 "wal", "catalogs", "locks"}) {
            require_regular_directory(root_ / child);
        }

        commit_lock_ = AdvisoryLockFile::open_existing(
            commit_lock_path(root_), writable_);
        // Writable handles later take view.lock exclusively while unlinking
        // orphaned immutable resources; keep that portable by opening the
        // lock inode read-write just like commit.lock.
        view_lock_ = AdvisoryLockFile::open_existing(
            view_lock_path(root_), writable_);
        commit_lock_.lock(
            writable_ ? FileLockMode::Exclusive : FileLockMode::Shared);
        try {
            view_lock_.lock(FileLockMode::Shared);
            open_catalog();
            resources_ = std::make_shared<ResourceSet>();

            phase_started = SteadyClock::now();
            // view.lock protects pathnames only while immutable inodes are
            // pinned. Data descriptors remain open; every snapshot is mmaped
            // and its descriptor is closed immediately. Potentially large
            // checksums and structural scans happen after releasing both
            // publication locks.
            pin_segments(options.verify_native_index);
            pin_snapshots();
            pin_runs();
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
        std::shared_ptr<ResourceSet> resource_pin;
        {
            std::shared_lock state(state_mutex_);
            resource_pin = resources_;
            const bool zero_delta_fast_path =
                resource_pin->runs.empty()
                && !mutation_read_path_active_.load(
                    std::memory_order_relaxed);
            if constexpr (Profiled) {
                stats->zero_delta_fast_path = zero_delta_fast_path;
            }
            for (std::size_t output_index = 0;
                 output_index != ids.size(); ++output_index) {
                const std::optional<RowRef> row = zero_delta_fast_path
                    ? [&]() -> std::optional<RowRef> {
                          const std::optional<IndexedVersion> indexed =
                              snapshot_version(
                                  ids[output_index], *resource_pin);
                          return indexed.has_value()
                                     ? indexed->row_ref()
                                     : std::nullopt;
                      }()
                    : logical_lookup_unlocked(
                          ids[output_index], *resource_pin);
                if (!row.has_value()) {
                    throw MissingIdError(ids[output_index]);
                }
                const bool mutation =
                    !resource_pin->initial_segment_indices.contains(
                        row->segment_id().value());
                if constexpr (Profiled) {
                    mutation ? ++stats->mutation_rows : ++stats->base_rows;
                }
                requests.push_back(resolve_unlocked(
                    *row, static_cast<std::uint64_t>(output_index),
                    *resource_pin));
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
        std::shared_lock state(state_mutex_);
        const std::shared_ptr<ResourceSet> resource_pin = resources_;
        std::vector<std::uint64_t> run_slot_counts;
        run_slot_counts.reserve(resource_pin->runs.size());
        std::uint64_t run_slots = 0;
        for (const auto& run : resource_pin->runs) {
            run_slots = checked_sum(
                run_slots,
                run.active_entry_count,
                "partition run sample slots");
            run_slot_counts.push_back(run.active_entry_count);
        }

        const std::size_t partitions = resource_pin->snapshots.size();
        std::uint64_t snapshot_slots = 0;
        for (std::size_t index = 0; index != partitions; ++index) {
            const PartitionSnapshotDescriptor& descriptor =
                resource_pin->snapshots[index]->header.descriptor;
            snapshot_slots = checked_sum(
                snapshot_slots, descriptor.capacity,
                "partition sample capacity");
        }

        const std::array<std::uint64_t, 3> source_slots{
            static_cast<std::uint64_t>(overlay_.size()),
            run_slots,
            snapshot_slots};
        std::uint64_t total_slots = 0;
        for (const std::uint64_t slots : source_slots) {
            total_slots = checked_sum(
                total_slots, slots, "logical ID sample source slots");
        }
        const std::uint64_t inspection_limit =
            std::min(max_inspected_slots, total_slots);
        std::array<std::uint64_t, 3> reserved_source_slots{};
        std::array<std::uint64_t, 3> residual_source_slots = source_slots;
        std::uint64_t residual_limit = inspection_limit;
        // A very small calibration sample must still see the mutable layers
        // before falling back to the immutable snapshot.  For larger samples
        // this one-slot reservation is negligible, while it prevents a small
        // delta from disappearing behind sparse snapshot capacity.
        for (const std::size_t index : std::array<std::size_t, 3>{0, 1, 2}) {
            if (residual_limit == 0 || residual_source_slots[index] == 0) {
                continue;
            }
            reserved_source_slots[index] = 1;
            --residual_source_slots[index];
            --residual_limit;
        }
        const std::vector<std::uint64_t> residual_quotas =
            proportional_sample_quotas(
                std::span<const std::uint64_t>(residual_source_slots),
                residual_limit, seed);
        std::array<std::uint64_t, 3> quotas{};
        for (std::size_t index = 0; index != quotas.size(); ++index) {
            quotas[index] = reserved_source_slots[index]
                            + residual_quotas[index];
        }
        const std::size_t unique_limit = static_cast<std::size_t>(
            std::min(requested, inspection_limit));

        using HashedId = std::pair<std::uint64_t, std::uint64_t>;
        std::priority_queue<HashedId> selected;
        std::unordered_set<std::uint64_t> selected_ids;
        selected_ids.reserve(unique_limit);
        const auto inspect_id = [&](const std::uint64_t id) {
            ++result.inspected_slots;
            if (!logical_lookup_unlocked(id, *resource_pin).has_value()
                || selected_ids.contains(id)) {
                return;
            }
            const HashedId candidate{hash_id(id, seed), id};
            if (selected.size() != unique_limit) {
                selected.push(candidate);
                selected_ids.insert(id);
            } else if (unique_limit != 0 && candidate < selected.top()) {
                selected_ids.erase(selected.top().second);
                selected.pop();
                selected.push(candidate);
                selected_ids.insert(id);
            }
        };

        std::uint64_t overlay_inspected = 0;
        for (const auto& [id, value] : overlay_) {
            static_cast<void>(value);
            if (overlay_inspected == quotas[0]) {
                break;
            }
            inspect_id(id);
            ++overlay_inspected;
        }

        const std::vector<std::uint64_t> run_quotas =
            proportional_sample_quotas(
                run_slot_counts, quotas[1],
                seed ^ 0x94d049bb133111ebULL);
        std::vector<PartitionRunTocEntry> active_sections;
        active_sections.reserve(manifest_.partition_count);
        for (std::size_t run_index = 0;
             run_index != resource_pin->runs.size(); ++run_index) {
            const RunView& run = resource_pin->runs[run_index];
            const std::uint64_t run_quota = run_quotas[run_index];
            if (run_quota == 0) {
                continue;
            }

            active_sections.clear();
            std::uint64_t active_entries = 0;
            for (std::size_t byte_index = 0;
                 byte_index != run.catalog.active_partitions.size();
                 ++byte_index) {
                unsigned int bits = std::to_integer<std::uint8_t>(
                    run.catalog.active_partitions[byte_index]);
                while (bits != 0) {
                    const unsigned int bit = std::countr_zero(bits);
                    const std::uint64_t partition =
                        static_cast<std::uint64_t>(byte_index) * 8 + bit;
                    if (partition >= manifest_.partition_count) {
                        throw CorruptionError(
                            "partition run active bitmap uses padding bits");
                    }
                    const PartitionRunTocEntry section =
                        run.file->partition_records(
                            static_cast<std::uint32_t>(partition));
                    if (section.record_count == 0) {
                        throw CorruptionError(
                            "active partition run section is empty");
                    }
                    active_entries = checked_sum(
                        active_entries, section.record_count,
                        "sampled active partition run entries");
                    active_sections.push_back(section);
                    bits &= bits - 1;
                }
            }
            if (active_entries != run.active_entry_count) {
                throw CorruptionError(
                    "partition run active entry count changed");
            }

            std::size_t section_index = 0;
            std::uint64_t section_begin = 0;
            std::uint64_t section_end = active_sections.front().record_count;
            visit_stratified_sample_positions(
                active_entries, run_quota,
                hash_id(
                    run.catalog.epoch,
                    seed ^ static_cast<std::uint64_t>(run_index)),
                [&](const std::uint64_t position) {
                    while (position >= section_end) {
                        section_begin = section_end;
                        ++section_index;
                        if (section_index == active_sections.size()) {
                            throw CorruptionError(
                                "run sample position exceeds active sections");
                        }
                        section_end = checked_sum(
                            section_end,
                            active_sections[section_index].record_count,
                            "run sample section end");
                    }
                    const PartitionRunTocEntry& section =
                        active_sections[section_index];
                    inspect_id(run.file
                                   ->record_at(
                                       section.first_record
                                       + (position - section_begin))
                                   .id);
                });
        }

        if (quotas[2] != 0) {
            if (partitions == 0) {
                throw CorruptionError(
                    "snapshot sample has no partition snapshots");
            }
            std::size_t partition = 0;
            std::uint64_t partition_begin = 0;
            std::uint64_t partition_end = resource_pin->snapshots.front()
                                                      ->header.descriptor
                                                      .capacity;
            visit_stratified_sample_positions(
                snapshot_slots, quotas[2],
                seed ^ 0x369dea0f31a53f85ULL,
                [&](const std::uint64_t position) {
                    while (position >= partition_end) {
                        partition_begin = partition_end;
                        ++partition;
                        if (partition == partitions) {
                            throw CorruptionError(
                                "snapshot sample position exceeds capacity");
                        }
                        partition_end = checked_sum(
                            partition_end,
                            resource_pin->snapshots[partition]
                                ->header.descriptor.capacity,
                            "snapshot sample partition end");
                    }
                    const Snapshot& snapshot =
                        *resource_pin->snapshots[partition];
                    const std::uint64_t slot = position - partition_begin;
                    if (control(snapshot, slot) == 1) {
                        inspect_id(slot_id(snapshot, slot));
                    } else {
                        ++result.inspected_slots;
                    }
                });
        }
        if (result.inspected_slots != inspection_limit) {
            throw CorruptionError("ID sample inspection accounting changed");
        }
        selected_ids.clear();
        selected_ids.rehash(0);
        result.ids.resize(selected.size());
        while (!selected.empty()) {
            result.ids[selected.size() - 1] = selected.top().second;
            selected.pop();
        }
        result.complete = result.ids.size() == requested;
        return result;
    }

    [[nodiscard]] std::unique_ptr<PartitionedIdScan> scan_ids(
        std::uint64_t max_snapshot_bytes) const;

    [[nodiscard]] std::vector<std::uint8_t> contains(
        const std::span<const std::uint64_t> ids) const {
        ensure_healthy();
        std::vector<std::uint8_t> result(ids.size(), 0);
        std::shared_lock state(state_mutex_);
        const std::shared_ptr<ResourceSet> resource_pin = resources_;
        const bool zero_delta_fast_path =
            resource_pin->runs.empty()
            && !mutation_read_path_active_.load(std::memory_order_relaxed);
        for (std::size_t index = 0; index != ids.size(); ++index) {
            result[index] = static_cast<std::uint8_t>(
                zero_delta_fast_path
                    ? snapshot_version(ids[index], *resource_pin).has_value()
                    : logical_lookup_unlocked(
                          ids[index], *resource_pin).has_value());
        }
        return result;
    }

    [[nodiscard]] NativeCollectionIdentity identity() const {
        ensure_healthy();
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        return NativeCollectionIdentity{
            manifest_.collection_id.low,
            manifest_.collection_id.high,
            catalog_.generation,
            next_sequence_ - 1};
    }

    [[nodiscard]] NativeMutationVerification verify_committed_mutation(
        const std::uint64_t sequence,
        const NativeMutationFingerprint& expected) const {
        ensure_healthy();
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        for (const auto& source : resources_->sources) {
            if (sequence < source->header.first_sequence
                || sequence >= source->next_sequence) {
                continue;
            }
            std::optional<WalMutationFingerprint> actual;
            static_cast<void>(scan_wal_file(
                source->fd.get(),
                source->path,
                ExpectedWalIdentity::from_trusted_header(source->header),
                source->entry.state == MutationWalState::Active
                    ? WalScanMode::Active
                    : WalScanMode::Frozen,
                [&](const RecoveredWalBatch& batch) {
                    if (batch.sequence == sequence) {
                        actual = fingerprint_recovered_wal_batch(
                            source->fd.get(),
                            source->path,
                            batch,
                            source->header.record_size);
                    }
                }));
            if (!actual.has_value()) {
                throw CorruptionError(
                    "retained partitioned WAL sequence disappeared during verification");
            }
            return *actual == expected.bytes
                       ? NativeMutationVerification::Match
                       : NativeMutationVerification::Mismatch;
        }
        return NativeMutationVerification::Unavailable;
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
                const bool exists =
                    logical_lookup_unlocked(id, *resources_).has_value();
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
            std::size_t new_overlay_entries = 0;
            for (const std::uint64_t id : ids) {
                new_overlay_entries += static_cast<std::size_t>(
                    !overlay_.contains(id));
            }
            ensure_overlay_growth_unlocked(new_overlay_entries);
            prepare_publication_capacity(new_overlay_entries, ids.size());
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
                if (!logical_lookup_unlocked(ids[index], *resources_)
                         .has_value()) {
                    continue;
                }
                deleted[index] = 1;
                --resulting_count;
                operations.push_back(WalMutation{
                    ids[index], WalOperationKind::Delete, {}});
            }
            if (!operations.empty()) {
                std::size_t new_overlay_entries = 0;
                for (const WalMutation& operation : operations) {
                    new_overlay_entries += static_cast<std::size_t>(
                        !overlay_.contains(operation.id));
                }
                ensure_overlay_growth_unlocked(new_overlay_entries);
                prepare_publication_capacity(new_overlay_entries, 0);
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

    [[nodiscard]] PartitionedResourceInventory collect_orphans() {
        ensure_mutable();
        std::unique_lock lifecycle(lifecycle_mutex_);
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        ScopedAdvisoryLock commit(
            commit_lock_, FileLockMode::Exclusive);
        ScopedAdvisoryLock view(
            view_lock_, FileLockMode::Exclusive);
        PartitionedResourceInventory inventory =
            inspect_partitioned_resources(root_);
        if (inventory.current_generation != catalog_.generation
            || inventory.current_catalog_id != catalog_.catalog_id) {
            throw StorageError(
                "format-7 orphan inventory CURRENT differs from the open "
                "writer snapshot; reopen the collection");
        }
        return inventory;
    }

    [[nodiscard]] PartitionedOrphanCollectionResult reclaim_orphans(
        const PartitionedOrphanCollectionOptions options) {
        ensure_mutable();
        std::unique_lock lifecycle(lifecycle_mutex_);
        // Directory inventory and candidate selection can scale with retained
        // and stray files. Build that advisory plan without the mutation or
        // publication barriers; the apply phase below re-reads CURRENT and
        // revokes every path that became reachable in the meantime.
        PartitionedOrphanCollectionPlan plan = plan_partitioned_orphans(
            root_, options);
        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::BeforeOrphanRevalidation,
            NativeCollectionTestOperation::CollectOrphans});
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        // PREPARE may have appeared while the unlocked inventory walked the
        // filesystem. Recheck after entering the native mutation lane so an
        // already-durable ambiguous operation defers this apply phase.
        ensure_pending_mutation_fence_absent(
            root_, "format-7 orphan reclamation");
        ScopedAdvisoryLock commit(
            commit_lock_, FileLockMode::Exclusive);
        ScopedAdvisoryLock view(
            view_lock_, FileLockMode::Exclusive);
        return apply_partitioned_orphan_plan(
            root_, catalog_.generation, catalog_.catalog_id,
            std::move(plan));
    }

    [[nodiscard]] NativeMutationLogStats rotate_wal() {
        ensure_mutable();
        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        return mutation_log_stats_locked(rotate_wal_locked());
    }

    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals(
        const PhysicalReader& physical_reader,
        const std::uint64_t max_wals,
        const std::uint64_t max_input_bytes,
        const std::uint64_t max_input_entries,
        const std::uint64_t max_memory_bytes) {
        ensure_mutable();
        if (max_wals == 0 || max_input_bytes == 0
            || max_input_entries == 0 || max_memory_bytes == 0) {
            throw InvalidArgumentError(
                "WAL materialization file, byte, entry, and memory budgets "
                "must be positive");
        }
        std::unique_lock lifecycle(lifecycle_mutex_);

        std::shared_ptr<ResourceSet> source_resources;
        std::vector<MutationWalEntry> selected_wals;
        std::uint64_t selected_input_bytes = 0;
        std::uint64_t selected_input_entries = 0;
        std::uint64_t selected_peak_memory = 0;
        bool entry_budget_blocked = false;
        bool memory_budget_blocked = false;
        std::uint64_t indexed_sequence = 0;
        std::uint64_t indexed_live_count = 0;
        Uuid128 terminal_wal_id;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            {
                std::shared_lock state(state_mutex_);
                source_resources = resources_;
            }
            if (source_resources->sources.size()
                < catalog_.wal_entries.size()) {
                throw CorruptionError(
                    "partitioned WAL resources do not cover the catalog");
            }
            std::size_t frozen_count = 0;
            while (frozen_count < catalog_.wal_entries.size()
                   && frozen_count < max_wals
                   && catalog_.wal_entries[frozen_count].state
                          == MutationWalState::Frozen) {
                const std::uint64_t bytes =
                    catalog_.wal_entries[frozen_count].sealed_file_size;
                if (bytes > max_input_bytes - selected_input_bytes) {
                    break;
                }
                const std::uint64_t entries =
                    source_resources->sources[frozen_count]->operation_count;
                if (entries > max_input_entries - selected_input_entries) {
                    entry_budget_blocked = true;
                    break;
                }
                const std::uint64_t candidate_entries =
                    selected_input_entries + entries;
                const std::uint64_t candidate_peak =
                    estimated_materialization_peak_memory(
                        overlay_.size(), candidate_entries);
                if (candidate_peak > max_memory_bytes) {
                    memory_budget_blocked = true;
                    selected_peak_memory = candidate_peak;
                    break;
                }
                selected_input_bytes += bytes;
                selected_input_entries = candidate_entries;
                selected_peak_memory = candidate_peak;
                ++frozen_count;
            }
            if (frozen_count == 0) {
                NativeMutationLogStats stats = mutation_log_stats_locked(
                    false, false, 0, 0,
                    !catalog_.wal_entries.empty()
                        && catalog_.wal_entries.front().state
                               == MutationWalState::Frozen);
                stats.materialization_entry_budget_blocked =
                    entry_budget_blocked;
                stats.materialization_memory_budget_blocked =
                    memory_budget_blocked;
                stats.materialization_estimated_peak_memory_bytes =
                    selected_peak_memory;
                return stats;
            }
            selected_wals.assign(
                catalog_.wal_entries.begin(),
                catalog_.wal_entries.begin()
                    + static_cast<std::ptrdiff_t>(frozen_count));
            if (source_resources->sources.size() < selected_wals.size()) {
                throw CorruptionError(
                    "partitioned WAL resources do not cover frozen prefix");
            }
            indexed_sequence = catalog_.indexed_sequence;
            indexed_live_count = catalog_.indexed_live_count;
            terminal_wal_id = catalog_.terminal_wal_id;
        }

        struct CollapsedRecord {
            std::uint64_t id{0};
            VersionedMutationValue value =
                VersionedMutationValue::tombstone(1);
            std::uint64_t materialized_value_raw{0};
        };
        std::unordered_map<std::uint64_t, VersionedMutationValue> collapsed;
        for (std::size_t source_index = 0;
             source_index != selected_wals.size(); ++source_index) {
            const MutationWalEntry& entry = selected_wals[source_index];
            const std::shared_ptr<WalSource>& source =
                source_resources->sources[source_index];
            if (source->entry != entry) {
                throw CorruptionError(
                    "partitioned WAL resource does not match selected prefix");
            }
            std::uint64_t put_index = 0;
            const WalScanResult scan = scan_wal_file(
                source->fd.get(), source->path,
                ExpectedWalIdentity::from_trusted_header(source->header),
                WalScanMode::Frozen,
                [&collapsed, &entry, &put_index](
                    const RecoveredWalBatch& batch) {
                    for (const RecoveredWalOperation& operation :
                         batch.operations) {
                        if (operation.kind == WalOperationKind::Put) {
                            if (put_index
                                >= RowRef::kMaximumRowsPerSegment) {
                                throw CorruptionError(
                                    "frozen partitioned WAL exceeds RowRef "
                                    "capacity");
                            }
                            collapsed.insert_or_assign(
                                operation.id,
                                VersionedMutationValue::put(
                                    batch.sequence,
                                    RowRef::create(
                                        entry.segment_id, put_index)));
                            ++put_index;
                        } else {
                            collapsed.insert_or_assign(
                                operation.id,
                                VersionedMutationValue::tombstone(
                                    batch.sequence));
                        }
                    }
                });
            if (scan.valid_end != entry.sealed_file_size
                || scan.ignored_tail_bytes != 0
                || scan.next_sequence == 0
                || scan.next_sequence - 1 != entry.last_sequence
                || scan.committed_operation_count
                       != source->operation_count
                || put_index != entry.put_count) {
                throw CorruptionError(
                    "frozen partitioned WAL changed during materialization");
            }
        }
        if (collapsed.empty()) {
            throw CorruptionError(
                "nonempty frozen partitioned WAL prefix has no operations");
        }

        std::vector<CollapsedRecord> records;
        records.reserve(collapsed.size());
        std::uint64_t final_put_count = 0;
        std::uint64_t resulting_indexed_live_count = indexed_live_count;
        for (auto& [id, value] : collapsed) {
            const std::optional<IndexedVersion> previous =
                indexed_version(id, *source_resources);
            const bool previously_live =
                previous.has_value() && !previous->tombstone();
            if (value.is_put()) {
                ++final_put_count;
                if (!previously_live) {
                    resulting_indexed_live_count = checked_sum(
                        resulting_indexed_live_count, 1,
                        "partitioned indexed live count");
                }
            } else if (previously_live) {
                if (resulting_indexed_live_count == 0) {
                    throw CorruptionError(
                        "partitioned indexed live count underflows");
                }
                --resulting_indexed_live_count;
            }
            records.push_back(CollapsedRecord{id, value, 0});
        }
        collapsed.clear();
        collapsed.rehash(0);

        const std::uint64_t rows_per_segment =
            partitioned_segment_row_capacity(manifest_);
        const std::uint64_t required_segment_count =
            final_put_count / rows_per_segment
            + (final_put_count % rows_per_segment == 0 ? 0 : 1);
        std::uint32_t first_segment_id = 0;
        std::uint64_t reserved_epoch = 0;
        std::uint64_t reservation_generation = 0;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            validate_materialization_anchor(
                selected_wals, indexed_sequence, terminal_wal_id);
            if (catalog_.generation
                > std::numeric_limits<std::uint64_t>::max() - 2) {
                throw StorageError(
                    "partitioned catalog lacks two materialization "
                    "generations");
            }
            if (catalog_.runs.size() >= kPartitionedMaximumRuns) {
                throw StorageError("partitioned index run limit is exhausted");
            }
            if (required_segment_count
                > kPartitionedMaximumResourceEntries
                      - catalog_.segments.size()) {
                throw StorageError(
                    "partitioned segment catalog limit is exhausted");
            }
            if (required_segment_count
                > static_cast<std::uint64_t>(SegmentId::kMaximum)
                      - catalog_.next_segment_id) {
                throw StorageError(
                    "partitioned materialization segment ID space is "
                    "exhausted");
            }
            if (catalog_.next_index_epoch
                == std::numeric_limits<std::uint64_t>::max()) {
                throw StorageError(
                    "partitioned index epoch space is exhausted");
            }
            first_segment_id = catalog_.next_segment_id;
            reserved_epoch = catalog_.next_index_epoch;

            PartitionedCatalogV3 reservation = catalog_;
            reservation.generation = catalog_.generation + 1;
            reservation.catalog_id = make_uuid();
            reservation.previous_catalog_id = catalog_.catalog_id;
            reservation.next_segment_id = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(catalog_.next_segment_id)
                + required_segment_count);
            reservation.next_index_epoch = reserved_epoch + 1;
            reservation_generation = reservation.generation;
            try {
                ScopedAdvisoryLock commit(
                    commit_lock_, FileLockMode::Exclusive);
                publish_partitioned_catalog(
                    root_, manifest_, reservation,
                    NativeCollectionTestOperation::MaterializeFrozenWals);
                invoke_native_collection_test_hook({
                    NativeCollectionTestHookPoint::
                        MutationReservationDurable,
                    NativeCollectionTestOperation::MaterializeFrozenWals,
                    NativeCollectionTestResourceKind::None,
                    reservation.generation,
                    0});
                std::unique_lock state(state_mutex_);
                std::swap(catalog_, reservation);
            } catch (...) {
                poisoned_.store(true, std::memory_order_release);
                throw;
            }
        }

        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::LifecycleBuildUnlocked,
            NativeCollectionTestOperation::MaterializeFrozenWals,
            NativeCollectionTestResourceKind::None,
            reservation_generation,
            0});

        std::vector<CollapsedRecord*> put_records;
        put_records.reserve(static_cast<std::size_t>(final_put_count));
        for (CollapsedRecord& record : records) {
            if (record.value.is_put()) {
                put_records.push_back(&record);
            }
        }
        std::sort(
            put_records.begin(), put_records.end(),
            [](const CollapsedRecord* left, const CollapsedRecord* right) {
                const RowRef left_row = *left->value.row_ref();
                const RowRef right_row = *right->value.row_ref();
                return left_row < right_row;
            });

        std::vector<PartitionedSegmentDescriptor> segment_descriptors;
        if (!put_records.empty()) {
            PartitionedSegmentSetBuildSpec segment_spec;
            segment_spec.manifest = manifest_;
            segment_spec.kind = PartitionedSegmentKind::Materialized;
            segment_spec.first_row_ref_segment_id = first_segment_id;
            segment_spec.creation_epoch = reserved_epoch;
            PartitionedSegmentSetWriter writer(root_, std::move(segment_spec));
            const std::uint64_t batch_rows = std::max<std::uint64_t>(
                1, (8ULL << 20) / manifest_.record_size);
            for (std::size_t begin = 0; begin != put_records.size();) {
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(
                        batch_rows, put_records.size() - begin));
                const std::uint64_t byte_count = checked_multiply(
                    count, manifest_.record_size,
                    "partitioned materialization read batch");
                if (byte_count > std::numeric_limits<std::size_t>::max()) {
                    throw StorageError(
                        "partitioned materialization batch exceeds address "
                        "space");
                }
                std::vector<std::byte> batch(
                    static_cast<std::size_t>(byte_count));
                std::vector<PhysicalReadRequest> requests;
                requests.reserve(count);
                for (std::size_t offset = 0; offset != count; ++offset) {
                    requests.push_back(resolve_unlocked(
                        *put_records[begin + offset]->value.row_ref(),
                        offset, *source_resources));
                }
                physical_reader.execute(
                    manifest_.record_size, requests, batch);
                for (std::size_t offset = 0; offset != count; ++offset) {
                    CollapsedRecord& record =
                        *put_records[begin + offset];
                    const RowRef output_row = writer.append(
                        record.id, record.value.sequence(),
                        std::span<const std::byte>(batch).subspan(
                            offset * static_cast<std::size_t>(
                                manifest_.record_size),
                            static_cast<std::size_t>(manifest_.record_size)));
                    record.materialized_value_raw = output_row.raw();
                }
                begin += count;
            }
            segment_descriptors = writer.finish();
        }
        if (segment_descriptors.size() != required_segment_count) {
            throw StorageError(
                "partitioned segment writer did not honor reserved range");
        }

        std::vector<PartitionRunRecord> run_records;
        run_records.reserve(records.size());
        for (const CollapsedRecord& record : records) {
            run_records.push_back(PartitionRunRecord{
                record.id, record.value.sequence(),
                record.materialized_value_raw});
        }
        PartitionIndexRunBuildSpec run_spec;
        run_spec.collection_id = manifest_.collection_id;
        run_spec.schema_id = manifest_.schema_id;
        run_spec.partition_bits = manifest_.partition_bits;
        run_spec.hash_seed = manifest_.hash_seed;
        run_spec.descriptor.run_id = make_uuid();
        run_spec.descriptor.epoch = reserved_epoch;
        run_spec.descriptor.kind = PartitionedRunKind::Mutation;
        run_spec.descriptor.first_sequence =
            selected_wals.front().first_sequence;
        run_spec.descriptor.last_sequence =
            selected_wals.back().last_sequence;
        const PartitionIndexRunDescriptor run_descriptor =
            write_partitioned_index_run(root_, run_spec, run_records);
        sync_directory(partitioned_index_runs_directory(root_));
        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::MutationDeltaOutputDurable,
            NativeCollectionTestOperation::MaterializeFrozenWals,
            NativeCollectionTestResourceKind::MutationDeltaPair,
            reservation_generation,
            0});

        std::vector<std::shared_ptr<PartitionedSegmentReader>>
            opened_segments;
        opened_segments.reserve(segment_descriptors.size());
        for (const PartitionedSegmentDescriptor& descriptor :
             segment_descriptors) {
            opened_segments.push_back(
                std::make_shared<PartitionedSegmentReader>(
                    root_, manifest_, descriptor, false));
        }
        auto opened_run = std::make_shared<PartitionIndexRunReader>(
            root_, manifest_.collection_id, manifest_.schema_id,
            manifest_.partition_bits, manifest_.hash_seed,
            run_descriptor);

        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        validate_materialization_anchor(
            selected_wals, indexed_sequence, terminal_wal_id);
        if (catalog_.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("partitioned catalog generation is exhausted");
        }

        PartitionedCatalogV3 next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        next_catalog.indexed_sequence = run_descriptor.last_sequence;
        next_catalog.indexed_live_count = resulting_indexed_live_count;
        next_catalog.terminal_wal_id = selected_wals.back().wal_id;
        next_catalog.segments.insert(
            next_catalog.segments.end(), segment_descriptors.begin(),
            segment_descriptors.end());
        next_catalog.runs.push_back(run_descriptor);
        next_catalog.wal_entries.erase(
            next_catalog.wal_entries.begin(),
            next_catalog.wal_entries.begin()
                + static_cast<std::ptrdiff_t>(selected_wals.size()));

        auto next_resources = std::make_shared<ResourceSet>();
        std::vector<decltype(overlay_)::iterator> overlay_erases;
        overlay_erases.reserve(records.size());
        {
            std::shared_lock state(state_mutex_);
            next_resources->initial_segments = resources_->initial_segments;
            next_resources->initial_segment_indices =
                resources_->initial_segment_indices;
            next_resources->materialized_segments =
                resources_->materialized_segments;
            next_resources->materialized_segment_indices =
                resources_->materialized_segment_indices;
            next_resources->snapshots = resources_->snapshots;
            next_resources->runs = resources_->runs;
            next_resources->sources.assign(
                resources_->sources.begin()
                    + static_cast<std::ptrdiff_t>(selected_wals.size()),
                resources_->sources.end());
            for (const CollapsedRecord& record : records) {
                const auto current = overlay_.find(record.id);
                if (current == overlay_.end()
                    || !current->second.has_value()
                    || current->second->sequence()
                           < record.value.sequence()) {
                    throw CorruptionError(
                        "materialized run is newer than the live overlay");
                }
                if (current->second->sequence()
                    == record.value.sequence()) {
                    overlay_erases.push_back(current);
                }
            }
        }
        for (std::shared_ptr<PartitionedSegmentReader>& segment :
             opened_segments) {
            const std::uint32_t segment_id =
                segment->descriptor().row_ref_segment_id;
            const std::size_t index =
                next_resources->materialized_segments.size();
            next_resources->materialized_segments.push_back(
                std::move(segment));
            if (!next_resources->materialized_segment_indices.emplace(
                     segment_id, index).second) {
                throw CorruptionError(
                    "materialized segment reuses a RowRef segment ID");
            }
        }
        next_resources->runs.push_back(
            RunView{
                std::move(opened_run), run_descriptor,
                run_descriptor.entry_count});
        next_resources->source_indices.reserve(
            next_resources->sources.size());
        for (std::size_t index = 0;
             index != next_resources->sources.size(); ++index) {
            if (!next_resources->source_indices.emplace(
                     next_resources->sources[index]->entry.segment_id,
                     index).second) {
                throw CorruptionError(
                    "retained partitioned WAL segment ID is duplicated");
            }
        }
        if (next_resources->sources.empty()
            || next_resources->sources.back()->entry.state
                   != MutationWalState::Active) {
            throw CorruptionError(
                "materialization removed the active partitioned WAL");
        }
        next_resources->active_index = next_resources->sources.size() - 1;

        try {
            ScopedAdvisoryLock commit(
                commit_lock_, FileLockMode::Exclusive);
            publish_partitioned_catalog(
                root_, manifest_, next_catalog,
                NativeCollectionTestOperation::MaterializeFrozenWals);
            {
                std::unique_lock state(state_mutex_);
                for (const auto position : overlay_erases) {
                    overlay_.erase(position);
                }
                resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
                mutation_read_path_active_.store(
                    true, std::memory_order_release);
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        NativeMutationLogStats stats = mutation_log_stats_locked(
            false, true,
            static_cast<std::uint64_t>(selected_wals.size()),
            selected_input_bytes, false);
        stats.materialized_wal_input_entries = selected_input_entries;
        stats.materialization_estimated_peak_memory_bytes =
            selected_peak_memory;
        return stats;
    }

    [[nodiscard]] NativeMutationLogStats fold_partitions(
        const std::uint64_t max_runs,
        const std::uint64_t max_input_bytes,
        const std::uint64_t max_input_entries) {
        ensure_mutable();
        if (max_runs < 2 || max_runs > kPartitionedMaximumRuns) {
            throw InvalidArgumentError(
                "partition fold max_runs must be between 2 and 4096");
        }
        if (max_input_bytes == 0 || max_input_entries == 0) {
            throw InvalidArgumentError(
                "partition fold byte and entry budgets must be positive");
        }
        std::unique_lock lifecycle(lifecycle_mutex_);

        std::shared_ptr<ResourceSet> source_resources;
        std::vector<PartitionSnapshotDescriptor> source_snapshots;
        std::vector<PartitionIndexRunDescriptor> source_runs;
        std::uint64_t indexed_sequence = 0;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            std::shared_lock state(state_mutex_);
            source_resources = resources_;
            source_snapshots = catalog_.snapshots;
            source_runs = catalog_.runs;
            indexed_sequence = catalog_.indexed_sequence;
        }

        const auto run_views_for = [source_resources](
                                       const std::vector<
                                           PartitionIndexRunDescriptor>&
                                           descriptors) {
            std::vector<PartitionRunView> views;
            views.reserve(descriptors.size());
            for (const PartitionIndexRunDescriptor& descriptor :
                 descriptors) {
                const auto source = std::find_if(
                    source_resources->runs.begin(),
                    source_resources->runs.end(),
                    [&descriptor](const RunView& run) {
                        return run.catalog.run_id == descriptor.run_id;
                    });
                if (source == source_resources->runs.end()) {
                    throw CorruptionError(
                        "partition fold run has no pinned immutable view");
                }
                views.push_back(PartitionRunView{
                    source->file.get(), &descriptor});
            }
            return views;
        };

        std::vector<PartitionIndexRunDescriptor> remaining_runs = source_runs;
        std::vector<PartitionIndexRunDescriptor> selection_runs = source_runs;
        std::vector<PartitionFoldCandidate> selected;
        std::uint64_t remaining_entries = max_input_entries;
        std::uint64_t remaining_bytes = max_input_bytes;
        while (!selection_runs.empty()) {
            const std::vector<PartitionRunView> views =
                run_views_for(selection_runs);
            const std::optional<PartitionFoldCandidate> candidate =
                select_partition_for_fold(
                    manifest_.partition_bits, source_snapshots, views,
                    PartitionFoldBudget{
                        1, max_runs, remaining_entries, remaining_bytes});
            if (!candidate.has_value()) {
                break;
            }
            const std::uint64_t worst_output_bytes = snapshot_file_size(
                choose_capacity(candidate->input_entries));
            const std::uint64_t charged_bytes =
                std::max(candidate->input_bytes, worst_output_bytes);
            if (charged_bytes > remaining_bytes) {
                // This complete partition cannot fit later because the
                // aggregate budget only decreases. Exclude it from this
                // selection pass without clearing its catalog-active masks.
                clear_folded_partition(
                    selection_runs, manifest_.partition_bits,
                    candidate->partition_id);
                continue;
            }
            selected.push_back(*candidate);
            remaining_entries -= candidate->input_entries;
            remaining_bytes -= charged_bytes;
            clear_folded_partition(
                selection_runs, manifest_.partition_bits,
                candidate->partition_id);
            clear_folded_partition(
                remaining_runs, manifest_.partition_bits,
                candidate->partition_id);
        }
        const bool compaction_budget_blocked = !remaining_runs.empty();
        if (selected.empty()) {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            return mutation_log_stats_locked(
                false, false, 0, 0, false, false,
                compaction_budget_blocked);
        }

        std::uint64_t compacted_run_sections = 0;
        std::uint64_t run_input_entries = 0;
        std::uint64_t run_input_bytes = 0;
        std::uint64_t snapshot_input_entries = 0;
        std::uint64_t snapshot_input_bytes = 0;
        for (const PartitionFoldCandidate& candidate : selected) {
            compacted_run_sections = checked_sum(
                compacted_run_sections, candidate.active_run_count,
                "partition fold run sections");
            run_input_entries = checked_sum(
                run_input_entries, candidate.run_entries,
                "partition fold run input entries");
            run_input_bytes = checked_sum(
                run_input_bytes, candidate.run_bytes,
                "partition fold run input bytes");
            snapshot_input_entries = checked_sum(
                snapshot_input_entries, candidate.snapshot_entries,
                "partition fold snapshot input entries");
            snapshot_input_bytes = checked_sum(
                snapshot_input_bytes, candidate.snapshot_bytes,
                "partition fold snapshot input bytes");
        }

        std::uint64_t reserved_epoch = 0;
        std::uint64_t reservation_generation = 0;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            validate_fold_anchor(
                indexed_sequence, source_snapshots, source_runs);
            if (catalog_.generation
                > std::numeric_limits<std::uint64_t>::max() - 2) {
                throw StorageError(
                    "partitioned catalog lacks two fold generations");
            }
            if (catalog_.next_index_epoch
                == std::numeric_limits<std::uint64_t>::max()) {
                throw StorageError(
                    "partitioned index epoch space is exhausted");
            }
            reserved_epoch = catalog_.next_index_epoch;
            PartitionedCatalogV3 reservation = catalog_;
            reservation.generation = catalog_.generation + 1;
            reservation.catalog_id = make_uuid();
            reservation.previous_catalog_id = catalog_.catalog_id;
            reservation.next_index_epoch = reserved_epoch + 1;
            reservation_generation = reservation.generation;
            try {
                ScopedAdvisoryLock commit(
                    commit_lock_, FileLockMode::Exclusive);
                publish_partitioned_catalog(
                    root_, manifest_, reservation,
                    NativeCollectionTestOperation::FoldPartitions);
                invoke_native_collection_test_hook({
                    NativeCollectionTestHookPoint::
                        MutationReservationDurable,
                    NativeCollectionTestOperation::FoldPartitions,
                    NativeCollectionTestResourceKind::None,
                    reservation.generation,
                    0});
                std::unique_lock state(state_mutex_);
                std::swap(catalog_, reservation);
            } catch (...) {
                poisoned_.store(true, std::memory_order_release);
                throw;
            }
        }

        const std::vector<PartitionRunView> source_run_views =
            run_views_for(source_runs);
        std::vector<PartitionSnapshotDescriptor> output_descriptors;
        std::vector<std::shared_ptr<Snapshot>> output_snapshots;
        output_descriptors.reserve(selected.size());
        output_snapshots.reserve(selected.size());
        std::uint64_t output_entries = 0;
        std::uint64_t output_bytes = 0;
        for (const PartitionFoldCandidate& candidate : selected) {
            const PartitionSnapshotDescriptor& source_descriptor =
                source_snapshots[candidate.partition_id];
            PartitionSnapshotReader source_snapshot(
                root_, manifest_.collection_id, manifest_.schema_id,
                manifest_.partition_bits, manifest_.hash_seed,
                source_descriptor, true);
            const std::vector<PartitionRunRecord> records =
                merge_partition_for_fold(
                    manifest_.partition_bits, candidate.partition_id,
                    source_snapshot, source_run_views);
            const std::uint64_t expected_output_bytes = snapshot_file_size(
                choose_capacity(static_cast<std::uint64_t>(records.size())));
            const std::uint64_t worst_output_bytes = snapshot_file_size(
                choose_capacity(candidate.input_entries));
            if (expected_output_bytes > worst_output_bytes) {
                throw CorruptionError(
                    "partition fold output exceeds its pre-reserved worst "
                    "case geometry");
            }
            PartitionSnapshotBuildSpec spec;
            spec.collection_id = manifest_.collection_id;
            spec.schema_id = manifest_.schema_id;
            spec.snapshot_id = make_uuid();
            spec.partition_bits = manifest_.partition_bits;
            spec.partition_id = candidate.partition_id;
            spec.hash_seed = manifest_.hash_seed;
            spec.epoch = reserved_epoch;
            const PartitionSnapshotDescriptor output =
                write_partition_snapshot(root_, spec, records);
            output_entries = checked_sum(
                output_entries, output.entry_count,
                "partition fold output entries");
            output_bytes = checked_sum(
                output_bytes, output.file_size,
                "partition fold output bytes");
            output_descriptors.push_back(output);
        }
        sync_directory(partitioned_partition_snapshots_directory(root_));
        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::MutationDeltaOutputDurable,
            NativeCollectionTestOperation::FoldPartitions,
            NativeCollectionTestResourceKind::PartitionSnapshot,
            reservation_generation,
            static_cast<std::uint64_t>(output_descriptors.size())});
        for (const PartitionSnapshotDescriptor& descriptor :
             output_descriptors) {
            output_snapshots.push_back(open_snapshot_view(descriptor));
        }

        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        validate_fold_anchor(
            indexed_sequence, source_snapshots, source_runs);
        if (catalog_.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError("partitioned catalog generation is exhausted");
        }
        PartitionedCatalogV3 next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        for (std::size_t index = 0; index != selected.size(); ++index) {
            const std::uint32_t partition = selected[index].partition_id;
            next_catalog.snapshots[partition] = output_descriptors[index];
            clear_folded_partition(
                next_catalog.runs, manifest_.partition_bits, partition);
        }

        auto next_resources = std::make_shared<ResourceSet>();
        {
            std::shared_lock state(state_mutex_);
            next_resources->initial_segments = resources_->initial_segments;
            next_resources->initial_segment_indices =
                resources_->initial_segment_indices;
            next_resources->materialized_segments =
                resources_->materialized_segments;
            next_resources->materialized_segment_indices =
                resources_->materialized_segment_indices;
            next_resources->snapshots = resources_->snapshots;
            next_resources->sources = resources_->sources;
            next_resources->source_indices = resources_->source_indices;
            next_resources->active_index = resources_->active_index;
            for (std::size_t index = 0;
                 index != selected.size(); ++index) {
                next_resources->snapshots[selected[index].partition_id] =
                    output_snapshots[index];
            }
            next_resources->runs.reserve(next_catalog.runs.size());
            for (const PartitionIndexRunDescriptor& descriptor :
                 next_catalog.runs) {
                const auto source = std::find_if(
                    resources_->runs.begin(), resources_->runs.end(),
                    [&descriptor](const RunView& run) {
                        return run.catalog.run_id == descriptor.run_id;
                    });
                if (source == resources_->runs.end()) {
                    throw CorruptionError(
                        "folded catalog run has no pinned immutable view");
                }
                std::uint64_t active_entry_count =
                    source->active_entry_count;
                for (const PartitionFoldCandidate& candidate : selected) {
                    const std::uint32_t partition = candidate.partition_id;
                    if (partition_bitmap_contains(
                            source->catalog.active_partitions, partition)
                        && !partition_bitmap_contains(
                            descriptor.active_partitions, partition)) {
                        const std::uint64_t folded_entries =
                            source->file->partition_records(partition)
                                .record_count;
                        if (folded_entries > active_entry_count) {
                            throw CorruptionError(
                                "folded run active entry count underflowed");
                        }
                        active_entry_count -= folded_entries;
                    }
                }
                if (active_entry_count == 0) {
                    throw CorruptionError(
                        "retained folded run has no active entries");
                }
                next_resources->runs.push_back(RunView{
                    source->file, descriptor, active_entry_count});
            }
        }
        const bool mutation_path_active =
            !overlay_.empty() || !next_resources->runs.empty();
        try {
            ScopedAdvisoryLock commit(
                commit_lock_, FileLockMode::Exclusive);
            publish_partitioned_catalog(
                root_, manifest_, next_catalog,
                NativeCollectionTestOperation::FoldPartitions);
            {
                std::unique_lock state(state_mutex_);
                resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
                mutation_read_path_active_.store(
                    mutation_path_active, std::memory_order_release);
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        return mutation_log_stats_locked(
            false, false, 0, 0, false, true,
            compaction_budget_blocked,
            static_cast<std::uint64_t>(selected.size()),
            compacted_run_sections, snapshot_input_entries,
            snapshot_input_bytes, output_entries, output_bytes,
            run_input_entries, run_input_bytes);
    }

    [[nodiscard]] NativeSegmentCleaningStats clean_segments(
        const PhysicalReader& physical_reader,
        const NativeSegmentCleaningOptions options) {
        ensure_mutable();
        if (options.max_candidate_segments == 0
            || options.max_sidecar_scan_bytes == 0
            || options.max_source_bytes == 0
            || options.max_output_bytes == 0
            || options.max_memory_bytes == 0) {
            throw InvalidArgumentError(
                "segment cleaning budgets must be positive");
        }
        if (options.min_garbage_ratio_ppm > kCleanerRatioScale) {
            throw InvalidArgumentError(
                "segment cleaning garbage ratio exceeds 1000000 ppm");
        }
        if (options.start_after_segment_id > SegmentId::kMaximum) {
            throw InvalidArgumentError(
                "segment cleaning cursor exceeds the RowRef ID space");
        }
        std::unique_lock lifecycle(lifecycle_mutex_);

        NativeSegmentCleaningStats stats;
        stats.dry_run = options.dry_run;
        std::shared_ptr<ResourceSet> source_resources;
        std::vector<PartitionedSegmentDescriptor> candidates;
        std::uint64_t indexed_sequence = 0;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            std::shared_lock state(state_mutex_);
            source_resources = resources_;
            candidates = catalog_.segments;
            indexed_sequence = catalog_.indexed_sequence;
            stats.generation = catalog_.generation;
        }
        std::sort(
            candidates.begin(), candidates.end(),
            [](const PartitionedSegmentDescriptor& left,
               const PartitionedSegmentDescriptor& right) {
                return left.row_ref_segment_id
                       < right.row_ref_segment_id;
            });

        const auto sidecar_for = [&source_resources](
                                     const PartitionedSegmentDescriptor&
                                         descriptor)
            -> std::span<const std::byte> {
            const auto initial =
                source_resources->initial_segment_indices.find(
                    descriptor.row_ref_segment_id);
            if (initial
                != source_resources->initial_segment_indices.end()) {
                const Segment& segment =
                    *source_resources->initial_segments[initial->second];
                if (segment.header.descriptor != descriptor) {
                    throw CorruptionError(
                        "cleaner initial segment view disagrees with catalog");
                }
                return segment.ids_mapping.bytes().subspan(
                    static_cast<std::size_t>(
                        segment.ids_header.payload_offset),
                    static_cast<std::size_t>(
                        segment.ids_header.payload_size));
            }
            const auto derived =
                source_resources->materialized_segment_indices.find(
                    descriptor.row_ref_segment_id);
            if (derived
                == source_resources->materialized_segment_indices.end()) {
                throw CorruptionError(
                    "cleaner candidate has no pinned immutable segment");
            }
            const auto& segment =
                source_resources->materialized_segments[derived->second];
            if (segment->descriptor() != descriptor) {
                throw CorruptionError(
                    "cleaner derived segment view disagrees with catalog");
            }
            return segment->sidecar_payload();
        };
        const auto fixed_working_memory = [this](const std::uint64_t rows) {
            const std::uint64_t row_bytes = checked_multiply(
                rows, kCleanerEstimatedBytesPerRow,
                "segment cleaner memory estimate");
            const std::uint64_t partition_bytes = checked_multiply(
                manifest_.partition_count,
                kCleanerEstimatedBytesPerPartition,
                "segment cleaner partition scratch estimate");
            return checked_sum(
                checked_sum(
                    row_bytes, partition_bytes,
                    "segment cleaner memory estimate"),
                kCleanerFixedMemoryOverhead,
                "segment cleaner memory estimate");
        };
        const auto required_memory = [this, &fixed_working_memory](
                                         const std::uint64_t rows) {
            return checked_sum(
                fixed_working_memory(rows), manifest_.record_size,
                "segment cleaner memory estimate");
        };
        const auto replacement_bytes = [this](
                                           const std::uint64_t rows,
                                           const std::uint64_t segments) {
            if (rows == 0) {
                return std::uint64_t{0};
            }
            return checked_sum(
                checked_multiply(
                    checked_multiply(
                        segments, 2,
                        "segment cleaner output header count"),
                    kPartitionedHeaderSize,
                    "segment cleaner output headers"),
                checked_multiply(
                    rows,
                    checked_sum(
                        manifest_.record_size,
                        kPartitionedSegmentIdEntrySize,
                        "segment cleaner output row size"),
                    "segment cleaner output rows"),
                "segment cleaner output bytes");
        };
        const auto ratio_ppm = [](const std::uint64_t garbage,
                                  const std::uint64_t rows) {
            return garbage == rows
                       ? kCleanerRatioScale
                       : (garbage * kCleanerRatioScale) / rows;
        };
        const auto minimum_ratio_rows = [](const std::uint64_t rows,
                                           const std::uint32_t ppm) {
            const std::uint64_t whole =
                (rows / kCleanerRatioScale) * ppm;
            const std::uint64_t remainder =
                (rows % kCleanerRatioScale) * ppm;
            return whole + remainder / kCleanerRatioScale
                   + (remainder % kCleanerRatioScale == 0 ? 0 : 1);
        };
        const auto better_candidate = [](const CleanerCandidate& left,
                                         const CleanerCandidate& right) {
            if (left.reclaimable_bytes != right.reclaimable_bytes) {
                return left.reclaimable_bytes > right.reclaimable_bytes;
            }
            if (left.garbage_ratio_ppm != right.garbage_ratio_ppm) {
                return left.garbage_ratio_ppm > right.garbage_ratio_ppm;
            }
            if (left.garbage_rows != right.garbage_rows) {
                return left.garbage_rows > right.garbage_rows;
            }
            return left.descriptor.row_ref_segment_id
                   < right.descriptor.row_ref_segment_id;
        };

        std::optional<CleanerCandidate> selected;
        const auto first = std::upper_bound(
            candidates.begin(), candidates.end(),
            options.start_after_segment_id,
            [](const std::uint32_t cursor,
               const PartitionedSegmentDescriptor& descriptor) {
                return cursor < descriptor.row_ref_segment_id;
            });
        bool reached_end = first == candidates.end();
        std::uint64_t considered = 0;
        std::uint32_t last_considered = options.start_after_segment_id;
        for (auto position = first; position != candidates.end(); ++position) {
            if (considered == options.max_candidate_segments) {
                stats.budget_blocked = true;
                reached_end = false;
                break;
            }
            const PartitionedSegmentDescriptor& descriptor = *position;
            const std::uint64_t sidecar_bytes = checked_multiply(
                descriptor.row_count, kPartitionedSegmentIdEntrySize,
                "segment cleaner sidecar bytes");
            if (sidecar_bytes
                > options.max_sidecar_scan_bytes
                      - stats.scanned_sidecar_bytes) {
                stats.budget_blocked = true;
                if (stats.scanned_sidecar_bytes != 0
                    && sidecar_bytes <= options.max_sidecar_scan_bytes) {
                    reached_end = false;
                    break;
                }
                ++stats.skipped_budget_segment_count;
                ++considered;
                last_considered = descriptor.row_ref_segment_id;
                reached_end = std::next(position) == candidates.end();
                continue;
            }
            const std::uint64_t source_bytes = checked_sum(
                descriptor.data_file_size, descriptor.ids_file_size,
                "segment cleaner source bytes");
            if (source_bytes > options.max_source_bytes
                || required_memory(descriptor.row_count)
                       > options.max_memory_bytes) {
                stats.budget_blocked = true;
                ++stats.skipped_budget_segment_count;
                ++considered;
                last_considered = descriptor.row_ref_segment_id;
                reached_end = std::next(position) == candidates.end();
                continue;
            }

            const std::span<const std::byte> sidecar =
                sidecar_for(descriptor);
            if (sidecar.size() != sidecar_bytes
                || crc32c(sidecar) != descriptor.ids_payload_crc32c) {
                throw CorruptionError(
                    "segment cleaner IDs payload checksum mismatch");
            }
            CleanerCandidate candidate;
            candidate.descriptor = descriptor;
            candidate.source_bytes = source_bytes;
            candidate.records.reserve(
                static_cast<std::size_t>(descriptor.row_count));
            std::vector<std::uint64_t> unique_ids;
            unique_ids.reserve(
                static_cast<std::size_t>(descriptor.row_count));
            for (std::uint64_t row_index = 0;
                 row_index != descriptor.row_count; ++row_index) {
                const std::size_t offset = static_cast<std::size_t>(
                    row_index * kPartitionedSegmentIdEntrySize);
                const std::uint64_t id = get_u64(sidecar, offset);
                const std::uint64_t source_sequence =
                    get_u64(sidecar, offset + 8);
                if (source_sequence < descriptor.minimum_sequence
                    || source_sequence > descriptor.maximum_sequence
                    || source_sequence > indexed_sequence
                    || (descriptor.kind == PartitionedSegmentKind::Initial
                        && source_sequence != 0)
                    || (descriptor.kind
                            == PartitionedSegmentKind::Materialized
                        && source_sequence == 0)) {
                    throw CorruptionError(
                        "segment cleaner sidecar sequence is invalid");
                }
                unique_ids.push_back(id);
                const RowRef source_row = RowRef::create(
                    descriptor.row_ref_segment_id, row_index);
                const std::optional<IndexedVersion> winner =
                    indexed_version(id, *source_resources);
                const bool exact = winner.has_value()
                                   && !winner->tombstone()
                                   && winner->sequence == source_sequence
                                   && winner->value_raw == source_row.raw();
                if (exact) {
                    ++candidate.relocated_rows;
                    candidate.records.push_back(CleanerPlanRecord{
                        id, source_sequence, source_row.raw(), true});
                    continue;
                }
                ++candidate.garbage_rows;
                if (!winner.has_value()) {
                    candidate.records.push_back(CleanerPlanRecord{
                        id, source_sequence, 0, false});
                    continue;
                }
                if (winner->sequence < source_sequence
                    || winner->sequence == source_sequence) {
                    throw CorruptionError(
                        "segment cleaner found a conflicting indexed "
                        "version for a source row");
                }
                candidate.records.push_back(CleanerPlanRecord{
                    id, winner->sequence, winner->value_raw, false});
            }
            std::sort(unique_ids.begin(), unique_ids.end());
            if (std::adjacent_find(unique_ids.begin(), unique_ids.end())
                != unique_ids.end()) {
                throw CorruptionError(
                    "segment cleaner source sidecar contains duplicate IDs");
            }

            ++considered;
            ++stats.scanned_segment_count;
            stats.scanned_row_count = checked_sum(
                stats.scanned_row_count, descriptor.row_count,
                "segment cleaner scanned rows");
            stats.scanned_sidecar_bytes = checked_sum(
                stats.scanned_sidecar_bytes, sidecar_bytes,
                "segment cleaner scanned sidecar bytes");
            last_considered = descriptor.row_ref_segment_id;
            reached_end = std::next(position) == candidates.end();

            const std::uint64_t required_garbage = std::max(
                options.min_garbage_rows,
                minimum_ratio_rows(
                    descriptor.row_count,
                    options.min_garbage_ratio_ppm));
            if (candidate.garbage_rows == 0
                || candidate.garbage_rows < required_garbage) {
                continue;
            }
            const std::uint64_t segment_capacity =
                partitioned_segment_row_capacity(manifest_);
            candidate.output_segment_count =
                candidate.relocated_rows / segment_capacity
                + (candidate.relocated_rows % segment_capacity == 0
                       ? 0
                       : 1);
            candidate.output_segment_bytes = replacement_bytes(
                candidate.relocated_rows,
                candidate.output_segment_count);
            candidate.output_run_bytes =
                partitioned_index_run_file_size(
                    manifest_.partition_bits, descriptor.row_count);
            const std::uint64_t output_bytes = checked_sum(
                candidate.output_segment_bytes,
                candidate.output_run_bytes,
                "segment cleaner total output bytes");
            if (output_bytes > options.max_output_bytes
                || (output_bytes >= candidate.source_bytes
                    && options.min_reclaimable_bytes != 0)) {
                stats.budget_blocked = true;
                ++stats.skipped_budget_segment_count;
                continue;
            }
            candidate.reclaimable_bytes =
                output_bytes < candidate.source_bytes
                    ? candidate.source_bytes - output_bytes
                    : 0;
            if (candidate.reclaimable_bytes
                < options.min_reclaimable_bytes) {
                continue;
            }
            candidate.garbage_ratio_ppm = ratio_ppm(
                candidate.garbage_rows, descriptor.row_count);
            if (!selected.has_value()
                || better_candidate(candidate, *selected)) {
                selected = std::move(candidate);
            }
        }
        stats.cycle_complete = reached_end;
        stats.next_start_after_segment_id =
            reached_end ? 0 : last_considered;
        if (!selected.has_value()) {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            stats.generation = catalog_.generation;
            return stats;
        }

        CleanerCandidate& candidate = *selected;
        stats.candidate_found = true;
        stats.candidate_segment_id =
            candidate.descriptor.row_ref_segment_id;
        stats.source_rows = candidate.descriptor.row_count;
        stats.relocated_rows = candidate.relocated_rows;
        stats.garbage_rows = candidate.garbage_rows;
        stats.source_bytes = candidate.source_bytes;
        stats.output_segment_count = candidate.output_segment_count;
        stats.output_segment_bytes = candidate.output_segment_bytes;
        stats.output_run_bytes = candidate.output_run_bytes;
        stats.reclaimable_bytes = candidate.reclaimable_bytes;
        if (options.dry_run) {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            stats.generation = catalog_.generation;
            return stats;
        }

        std::uint32_t first_segment_id = 0;
        std::uint64_t reserved_epoch = 0;
        std::uint64_t reservation_generation = 0;
        {
            std::lock_guard mutation(mutation_mutex_);
            ensure_healthy();
            const auto source = std::find(
                catalog_.segments.begin(), catalog_.segments.end(),
                candidate.descriptor);
            if (source == catalog_.segments.end()
                || catalog_.indexed_sequence != indexed_sequence) {
                throw StorageError(
                    "segment cleaner indexed source changed before "
                    "reservation");
            }
            if (catalog_.generation
                > std::numeric_limits<std::uint64_t>::max() - 2) {
                throw StorageError(
                    "partitioned catalog lacks two cleaner generations");
            }
            if (catalog_.runs.size() >= kPartitionedMaximumRuns) {
                throw StorageError(
                    "partitioned index run limit is exhausted");
            }
            if (candidate.output_segment_count
                > kPartitionedMaximumResourceEntries
                      - (catalog_.segments.size() - 1)) {
                throw StorageError(
                    "partitioned segment catalog limit is exhausted");
            }
            if (catalog_.next_index_epoch
                == std::numeric_limits<std::uint64_t>::max()) {
                throw StorageError(
                    "partitioned index epoch space is exhausted");
            }
            if (candidate.output_segment_count
                > static_cast<std::uint64_t>(SegmentId::kMaximum)
                      - catalog_.next_segment_id) {
                throw StorageError(
                    "partitioned cleaner segment ID space is exhausted");
            }
            first_segment_id = catalog_.next_segment_id;
            reserved_epoch = catalog_.next_index_epoch;
            PartitionedCatalogV3 reservation = catalog_;
            reservation.generation = catalog_.generation + 1;
            reservation.catalog_id = make_uuid();
            reservation.previous_catalog_id = catalog_.catalog_id;
            reservation.next_index_epoch = reserved_epoch + 1;
            if (candidate.output_segment_count != 0) {
                reservation.next_segment_id = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(first_segment_id)
                    + candidate.output_segment_count);
            }
            reservation_generation = reservation.generation;
            try {
                ScopedAdvisoryLock commit(
                    commit_lock_, FileLockMode::Exclusive);
                publish_partitioned_catalog(
                    root_, manifest_, reservation,
                    NativeCollectionTestOperation::CleanSegment);
                invoke_native_collection_test_hook({
                    NativeCollectionTestHookPoint::MutationReservationDurable,
                    NativeCollectionTestOperation::CleanSegment,
                    NativeCollectionTestResourceKind::None,
                    reservation.generation,
                    0});
                std::unique_lock state(state_mutex_);
                std::swap(catalog_, reservation);
            } catch (...) {
                poisoned_.store(true, std::memory_order_release);
                throw;
            }
        }

        std::unique_ptr<PartitionedSegmentSetWriter> segment_writer;
        if (candidate.relocated_rows != 0) {
            PartitionedSegmentSetBuildSpec spec;
            spec.manifest = manifest_;
            spec.kind = PartitionedSegmentKind::Cleaner;
            spec.first_row_ref_segment_id = first_segment_id;
            spec.creation_epoch = reserved_epoch;
            segment_writer = std::make_unique<PartitionedSegmentSetWriter>(
                root_, std::move(spec));
        }
        const std::uint64_t fixed_memory =
            fixed_working_memory(candidate.descriptor.row_count);
        const std::uint64_t batch_byte_budget = std::min(
            kCleanerReadBatchBytes,
            options.max_memory_bytes - fixed_memory);
        const std::uint64_t batch_rows = std::max<std::uint64_t>(
            1, batch_byte_budget / manifest_.record_size);
        Crc32c source_crc;
        for (std::uint64_t begin = 0;
             begin != candidate.descriptor.row_count;) {
            const std::uint64_t count = std::min(
                batch_rows, candidate.descriptor.row_count - begin);
            const std::uint64_t byte_count = checked_multiply(
                count, manifest_.record_size,
                "segment cleaner read batch");
            if (byte_count > std::numeric_limits<std::size_t>::max()) {
                throw StorageError(
                    "segment cleaner read batch exceeds address space");
            }
            std::vector<std::byte> batch(
                static_cast<std::size_t>(byte_count));
            std::vector<PhysicalReadRequest> requests;
            requests.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t local = 0; local != count; ++local) {
                requests.push_back(resolve_unlocked(
                    RowRef::create(
                        candidate.descriptor.row_ref_segment_id,
                        begin + local),
                    local, *source_resources));
            }
            physical_reader.execute(
                manifest_.record_size, requests, batch);
            source_crc.update(batch);
            if (segment_writer != nullptr) {
                for (std::uint64_t local = 0; local != count; ++local) {
                    CleanerPlanRecord& record = candidate.records[
                        static_cast<std::size_t>(begin + local)];
                    if (!record.relocate) {
                        continue;
                    }
                    const RowRef output = segment_writer->append(
                        record.id, record.sequence,
                        std::span<const std::byte>(batch).subspan(
                            static_cast<std::size_t>(
                                local * manifest_.record_size),
                            static_cast<std::size_t>(
                                manifest_.record_size)));
                    record.value_raw = output.raw();
                }
            }
            begin += count;
        }
        if (source_crc.value()
            != candidate.descriptor.data_payload_crc32c) {
            throw CorruptionError(
                "segment cleaner data payload checksum mismatch");
        }

        std::vector<PartitionedSegmentDescriptor> output_segments;
        if (segment_writer != nullptr) {
            output_segments = segment_writer->finish();
        }
        if (output_segments.size() != stats.output_segment_count) {
            throw StorageError(
                "segment cleaner writer violated reserved geometry");
        }
        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::CleanerSegmentOutputDurable,
            NativeCollectionTestOperation::CleanSegment,
            NativeCollectionTestResourceKind::PartitionedSegmentPair,
            reservation_generation,
            static_cast<std::uint64_t>(output_segments.size())});

        std::vector<PartitionRunRecord> run_records;
        run_records.reserve(candidate.records.size());
        std::uint64_t first_sequence =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t last_sequence = 0;
        for (const CleanerPlanRecord& record : candidate.records) {
            first_sequence = std::min(first_sequence, record.sequence);
            last_sequence = std::max(last_sequence, record.sequence);
            run_records.push_back(PartitionRunRecord{
                record.id, record.sequence, record.value_raw});
        }
        PartitionIndexRunBuildSpec run_spec;
        run_spec.collection_id = manifest_.collection_id;
        run_spec.schema_id = manifest_.schema_id;
        run_spec.partition_bits = manifest_.partition_bits;
        run_spec.hash_seed = manifest_.hash_seed;
        run_spec.descriptor.run_id = make_uuid();
        run_spec.descriptor.epoch = reserved_epoch;
        run_spec.descriptor.kind = PartitionedRunKind::Cleaner;
        run_spec.descriptor.first_sequence = first_sequence;
        run_spec.descriptor.last_sequence = last_sequence;
        run_spec.descriptor.retired_row_ref_segment_id =
            candidate.descriptor.row_ref_segment_id;
        run_spec.descriptor.retired_segment_id =
            candidate.descriptor.segment_id;
        run_spec.descriptor.retired_row_count =
            candidate.descriptor.row_count;
        run_spec.descriptor.retired_ids_payload_crc32c =
            candidate.descriptor.ids_payload_crc32c;
        const PartitionIndexRunDescriptor output_run =
            write_partitioned_index_run(root_, run_spec, run_records);
        sync_directory(partitioned_index_runs_directory(root_));
        if (output_run.file_size != stats.output_run_bytes) {
            throw StorageError(
                "segment cleaner run violated predicted geometry");
        }
        invoke_native_collection_test_hook({
            NativeCollectionTestHookPoint::CleanerRunOutputDurable,
            NativeCollectionTestOperation::CleanSegment,
            NativeCollectionTestResourceKind::PartitionIndexRun,
            reservation_generation,
            1});

        std::vector<std::shared_ptr<PartitionedSegmentReader>>
            opened_segments;
        for (const PartitionedSegmentDescriptor& descriptor :
             output_segments) {
            opened_segments.push_back(
                std::make_shared<PartitionedSegmentReader>(
                    root_, manifest_, descriptor, false));
        }
        auto opened_run = std::make_shared<PartitionIndexRunReader>(
            root_, manifest_.collection_id, manifest_.schema_id,
            manifest_.partition_bits, manifest_.hash_seed, output_run);

        std::lock_guard mutation(mutation_mutex_);
        ensure_healthy();
        const auto source = std::find(
            catalog_.segments.begin(), catalog_.segments.end(),
            candidate.descriptor);
        if (source == catalog_.segments.end()
            || catalog_.indexed_sequence != indexed_sequence
            || catalog_.next_index_epoch != reserved_epoch + 1
            || static_cast<std::uint64_t>(catalog_.next_segment_id)
                   < static_cast<std::uint64_t>(first_segment_id)
                         + output_segments.size()) {
            throw StorageError(
                "segment cleaner source changed during immutable build");
        }
        if (catalog_.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StorageError(
                "partitioned catalog generation is exhausted");
        }
        PartitionedCatalogV3 next_catalog = catalog_;
        next_catalog.generation = catalog_.generation + 1;
        next_catalog.catalog_id = make_uuid();
        next_catalog.previous_catalog_id = catalog_.catalog_id;
        next_catalog.segments.erase(
            next_catalog.segments.begin()
            + std::distance(catalog_.segments.begin(), source));
        next_catalog.segments.insert(
            next_catalog.segments.end(), output_segments.begin(),
            output_segments.end());
        next_catalog.runs.push_back(output_run);

        auto next_resources = std::make_shared<ResourceSet>();
        {
            std::shared_lock state(state_mutex_);
            next_resources->snapshots = resources_->snapshots;
            next_resources->runs = resources_->runs;
            next_resources->sources = resources_->sources;
            next_resources->source_indices = resources_->source_indices;
            next_resources->active_index = resources_->active_index;
            for (const auto& segment : resources_->initial_segments) {
                if (segment->header.descriptor.row_ref_segment_id
                    != candidate.descriptor.row_ref_segment_id) {
                    next_resources->initial_segments.push_back(segment);
                }
            }
            for (std::size_t index = 0;
                 index != next_resources->initial_segments.size(); ++index) {
                const std::uint32_t id = next_resources
                                             ->initial_segments[index]
                                             ->header.descriptor
                                             .row_ref_segment_id;
                if (!next_resources->initial_segment_indices.emplace(
                         id, index).second) {
                    throw CorruptionError(
                        "segment cleaner rebuilt duplicate initial IDs");
                }
            }
            for (const auto& segment : resources_->materialized_segments) {
                if (segment->descriptor().row_ref_segment_id
                    != candidate.descriptor.row_ref_segment_id) {
                    next_resources->materialized_segments.push_back(segment);
                }
            }
        }
        for (auto& segment : opened_segments) {
            next_resources->materialized_segments.push_back(
                std::move(segment));
        }
        for (std::size_t index = 0;
             index != next_resources->materialized_segments.size(); ++index) {
            const std::uint32_t id = next_resources
                                         ->materialized_segments[index]
                                         ->descriptor()
                                         .row_ref_segment_id;
            if (!next_resources->materialized_segment_indices.emplace(
                     id, index).second) {
                throw CorruptionError(
                    "segment cleaner rebuilt duplicate derived IDs");
            }
        }
        next_resources->runs.push_back(
            RunView{
                std::move(opened_run), output_run,
                output_run.entry_count});
        try {
            ScopedAdvisoryLock commit(
                commit_lock_, FileLockMode::Exclusive);
            publish_partitioned_catalog(
                root_, manifest_, next_catalog,
                NativeCollectionTestOperation::CleanSegment);
            {
                std::unique_lock state(state_mutex_);
                resources_.swap(next_resources);
                std::swap(catalog_, next_catalog);
                mutation_read_path_active_.store(
                    true, std::memory_order_release);
            }
        } catch (...) {
            poisoned_.store(true, std::memory_order_release);
            throw;
        }
        stats.generation = catalog_.generation;
        stats.cleaned = true;
        stats.reclamation_complete = false;
        return stats;
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
        if (resources_ != nullptr) {
            for (const auto& segment : resources_->initial_segments) {
                segment->fd.reset();
            }
            for (const auto& segment : resources_->materialized_segments) {
                segment->abandon_after_fork();
            }
            for (const auto& source : resources_->sources) {
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
        std::uint64_t operation_count{0};
    };
    struct RunView {
        std::shared_ptr<PartitionIndexRunReader> file;
        PartitionIndexRunDescriptor catalog;
        std::uint64_t active_entry_count{0};
    };
    struct ResourceSet {
        std::vector<std::shared_ptr<Segment>> initial_segments;
        std::unordered_map<std::uint32_t, std::size_t>
            initial_segment_indices;
        std::vector<std::shared_ptr<PartitionedSegmentReader>>
            materialized_segments;
        std::unordered_map<std::uint32_t, std::size_t>
            materialized_segment_indices;
        std::vector<std::shared_ptr<Snapshot>> snapshots;
        std::vector<RunView> runs;
        std::vector<std::shared_ptr<WalSource>> sources;
        std::unordered_map<std::uint32_t, std::size_t> source_indices;
        std::size_t active_index{0};
    };
    struct IndexedVersion {
        std::uint64_t sequence{0};
        std::uint64_t value_raw{0};
        std::uint64_t source_epoch{0};

        [[nodiscard]] bool tombstone() const noexcept {
            return value_raw == 0;
        }
        [[nodiscard]] std::optional<RowRef> row_ref() const {
            return tombstone()
                       ? std::nullopt
                       : std::optional<RowRef>(RowRef::decode(value_raw));
        }
    };
    class IdScanState final : public PartitionedIdScanState {
    public:
        struct OverlayEntry {
            std::uint64_t id;
            VersionedMutationValue value;
        };

        IdScanState(
            PartitionedManifestV2 manifest,
            std::shared_ptr<ResourceSet> resources,
            std::vector<OverlayEntry> overlay,
            const NativeIdScanMetadata metadata)
            : manifest_(std::move(manifest)),
              resources_(std::move(resources)),
              overlay_(std::move(overlay)),
              metadata_(metadata) {
            std::sort(
                overlay_.begin(), overlay_.end(),
                [](const OverlayEntry& left, const OverlayEntry& right) {
                    return left.id < right.id;
                });
            overlay_put_order_.reserve(overlay_.size());
            for (std::size_t index = 0; index != overlay_.size(); ++index) {
                if (overlay_[index].value.is_put()) {
                    overlay_put_order_.push_back(index);
                }
            }
            std::sort(
                overlay_put_order_.begin(), overlay_put_order_.end(),
                [this](const std::size_t left, const std::size_t right) {
                    const RowRef left_row =
                        *overlay_[left].value.row_ref();
                    const RowRef right_row =
                        *overlay_[right].value.row_ref();
                    return left_row.raw() < right_row.raw();
                });

            const std::uint64_t segment_capacity = checked_sum(
                checked_sum(
                    resources_->initial_segments.size(),
                    resources_->materialized_segments.size(),
                    "ID scan segment capacity"),
                checked_multiply(
                    resources_->sources.size(), 2,
                    "ID scan segment capacity"),
                "ID scan segment capacity");
            if (segment_capacity > std::numeric_limits<std::size_t>::max()) {
                throw std::length_error(
                    "ID scan segment capacity does not fit this process");
            }
            segments_.reserve(static_cast<std::size_t>(segment_capacity));
            for (std::size_t index = 0;
                 index != resources_->initial_segments.size(); ++index) {
                const Segment& segment =
                    *resources_->initial_segments[index];
                segments_.push_back(SegmentView{
                    segment.header.descriptor.row_ref_segment_id,
                    SegmentKind::Initial,
                    index,
                    0,
                    segment.header.descriptor.row_count});
            }
            for (std::size_t index = 0;
                 index != resources_->materialized_segments.size(); ++index) {
                const PartitionedSegmentDescriptor& descriptor =
                    resources_->materialized_segments[index]->descriptor();
                segments_.push_back(SegmentView{
                    descriptor.row_ref_segment_id,
                    SegmentKind::Materialized,
                    index,
                    0,
                    descriptor.row_count});
            }
            std::size_t begin = 0;
            while (begin != overlay_put_order_.size()) {
                const std::uint32_t segment_id = overlay_row(begin)
                                                     .segment_id()
                                                     .value();
                std::size_t end = begin + 1;
                while (end != overlay_put_order_.size()
                       && overlay_row(end).segment_id().value()
                              == segment_id) {
                    ++end;
                }
                segments_.push_back(SegmentView{
                    segment_id,
                    SegmentKind::Overlay,
                    0,
                    begin,
                    static_cast<std::uint64_t>(end - begin)});
                begin = end;
            }
            std::sort(
                segments_.begin(), segments_.end(),
                [](const SegmentView& left, const SegmentView& right) {
                    return left.segment_id < right.segment_id;
                });
            for (std::size_t index = 1; index < segments_.size(); ++index) {
                if (segments_[index - 1].segment_id
                    == segments_[index].segment_id) {
                    throw CorruptionError(
                        "physical ID scan found a duplicated segment ID");
                }
            }
            done_ = metadata_.live_count == 0;
        }

        [[nodiscard]] const NativeIdScanMetadata& metadata()
            const noexcept override {
            return metadata_;
        }

        [[nodiscard]] std::uint64_t read_into(
            const std::span<std::uint64_t> ids,
            const std::span<NativeIdLocation> locations,
            const NativeIdScan::StopPredicate& stop_requested) override {
            if (::getpid() != owner_pid_) {
                throw StorageError(
                    "ID scan was inherited across fork; create a new scan "
                    "in the child process");
            }
            if (!locations.empty() && locations.size() != ids.size()) {
                throw InvalidArgumentError(
                    "ID scan locations capacity must equal IDs capacity");
            }
            if (ids.empty() || done_) {
                return 0;
            }

            std::uint64_t output_count = 0;
            std::uint64_t candidates_until_stop_check = 0;
            while (segment_position_ != segments_.size()) {
                const SegmentView& segment = segments_[segment_position_];
                while (row_position_ != segment.row_count) {
                    if (candidates_until_stop_check == 0) {
                        if (stop_requested && stop_requested()) {
                            return output_count;
                        }
                        candidates_until_stop_check =
                            kIdScanStopCheckCandidates;
                    }
                    const Candidate candidate = candidate_at(
                        segment, row_position_++);
                    --candidates_until_stop_check;
                    if (!is_authoritative(candidate)) {
                        continue;
                    }
                    ids[static_cast<std::size_t>(output_count)] =
                        candidate.id;
                    if (!locations.empty()) {
                        locations[static_cast<std::size_t>(output_count)] =
                            NativeIdLocation{
                                candidate.row.segment_id().value(),
                                candidate.row.row_index()};
                    }
                    ++output_count;
                    ++emitted_count_;
                    if (emitted_count_ > metadata_.live_count) {
                        throw CorruptionError(
                            "physical ID scan exceeded snapshot live count");
                    }
                    if (emitted_count_ == metadata_.live_count) {
                        done_ = true;
                        return output_count;
                    }
                    if (output_count == ids.size()) {
                        return output_count;
                    }
                }
                ++segment_position_;
                row_position_ = 0;
                // Preserve physical locality: a non-empty batch never spans
                // two segment IDs, even if caller capacity remains.
                if (output_count != 0) {
                    return output_count;
                }
            }
            if (emitted_count_ != metadata_.live_count) {
                throw CorruptionError(
                    "physical ID scan ended before snapshot live count");
            }
            done_ = true;
            return 0;
        }

        [[nodiscard]] bool done() const noexcept override {
            return done_;
        }

    private:
        enum class SegmentKind : std::uint8_t {
            Initial,
            Materialized,
            Overlay,
        };
        struct SegmentView {
            std::uint32_t segment_id;
            SegmentKind kind;
            std::size_t resource_index;
            std::size_t overlay_begin;
            std::uint64_t row_count;
        };
        struct Candidate {
            std::uint64_t id;
            std::uint64_t sequence;
            RowRef row;
        };

        [[nodiscard]] RowRef overlay_row(
            const std::size_t ordered_position) const {
            return *overlay_[overlay_put_order_[ordered_position]]
                        .value.row_ref();
        }

        [[nodiscard]] Candidate candidate_at(
            const SegmentView& segment,
            const std::uint64_t row_position) const {
            const RowRef row = RowRef::create(
                segment.segment_id,
                segment.kind == SegmentKind::Overlay
                    ? overlay_row(
                          segment.overlay_begin
                          + static_cast<std::size_t>(row_position))
                          .row_index()
                    : row_position);
            if (segment.kind == SegmentKind::Overlay) {
                const OverlayEntry& entry = overlay_[overlay_put_order_[
                    segment.overlay_begin
                    + static_cast<std::size_t>(row_position)]];
                return Candidate{entry.id, entry.value.sequence(), row};
            }
            if (segment.kind == SegmentKind::Materialized) {
                const PartitionedSegmentIdEntry entry =
                    resources_->materialized_segments[
                        segment.resource_index]
                        ->sidecar_at(row_position);
                return Candidate{entry.id, entry.sequence, row};
            }
            const Segment& initial =
                *resources_->initial_segments[segment.resource_index];
            const std::size_t offset = static_cast<std::size_t>(
                initial.ids_header.payload_offset
                + row_position * kPartitionedSegmentIdEntrySize);
            return Candidate{
                get_u64(initial.ids_mapping.bytes(), offset),
                get_u64(initial.ids_mapping.bytes(), offset + 8),
                row};
        }

        [[nodiscard]] const OverlayEntry* overlay_entry(
            const std::uint64_t id) const {
            const auto position = std::lower_bound(
                overlay_.begin(), overlay_.end(), id,
                [](const OverlayEntry& entry, const std::uint64_t value) {
                    return entry.id < value;
                });
            return position != overlay_.end() && position->id == id
                       ? &*position
                       : nullptr;
        }

        [[nodiscard]] std::optional<IndexedVersion> snapshot_version(
            const std::uint64_t id) const {
            const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
            const std::uint64_t partition = partition_for_hash(
                hash, manifest_.partition_bits);
            const Snapshot& snapshot = *resources_->snapshots[
                static_cast<std::size_t>(partition)];
            const std::uint64_t mask =
                snapshot.header.descriptor.capacity - 1;
            std::uint64_t slot = hash & mask;
            for (std::uint64_t probes = 0;
                 probes != snapshot.header.descriptor.capacity; ++probes) {
                if (Impl::control(snapshot, slot) == 0) {
                    return std::nullopt;
                }
                if (Impl::slot_id(snapshot, slot) == id) {
                    return IndexedVersion{
                        Impl::slot_sequence(snapshot, slot),
                        Impl::slot_value(snapshot, slot),
                        snapshot.header.descriptor.epoch};
                }
                slot = (slot + 1) & mask;
            }
            throw CorruptionError(
                "partition snapshot lookup has no empty slot");
        }

        [[nodiscard]] std::optional<IndexedVersion> indexed_version(
            const std::uint64_t id) const {
            std::optional<IndexedVersion> winner = snapshot_version(id);
            const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
            const auto partition = static_cast<std::uint32_t>(
                partition_for_hash(hash, manifest_.partition_bits));
            for (const RunView& run : resources_->runs) {
                if (!partition_bitmap_contains(
                        run.catalog.active_partitions, partition)) {
                    continue;
                }
                if (run.catalog.kind == PartitionedRunKind::LegacyDeltaV1
                    || run.catalog.epoch
                           <= resources_->snapshots[partition]
                                  ->header.descriptor.epoch) {
                    throw CorruptionError(
                        "active partition run is incompatible with its "
                        "authoritative snapshot");
                }
                const std::optional<PartitionRunRecord> record =
                    run.file->lookup(partition, id);
                if (!record.has_value()) {
                    continue;
                }
                const IndexedVersion candidate{
                    record->sequence, record->value_raw,
                    run.catalog.epoch};
                if (!winner.has_value()
                    || candidate.sequence > winner->sequence) {
                    winner = candidate;
                    continue;
                }
                if (candidate.sequence < winner->sequence) {
                    continue;
                }
                if (candidate.value_raw == winner->value_raw) {
                    winner->source_epoch = std::max(
                        winner->source_epoch, candidate.source_epoch);
                    continue;
                }
                bool valid_cleaner_relocation = false;
                if (run.catalog.kind == PartitionedRunKind::Cleaner
                    && run.catalog.epoch > winner->source_epoch
                    && candidate.value_raw != 0
                    && winner->value_raw != 0) {
                    const RowRef previous =
                        RowRef::decode(winner->value_raw);
                    valid_cleaner_relocation =
                        previous.segment_id().value()
                        == run.catalog.retired_row_ref_segment_id;
                }
                if (!valid_cleaner_relocation) {
                    throw CorruptionError(
                        "partitioned index has conflicting values at one "
                        "sequence");
                }
                winner = candidate;
            }
            return winner;
        }

        [[nodiscard]] bool is_authoritative(
            const Candidate& candidate) const {
            const OverlayEntry* overlay = overlay_entry(candidate.id);
            if (overlay != nullptr) {
                return overlay->value.is_put()
                       && overlay->value.sequence() == candidate.sequence
                       && overlay->value.encoded_row_ref()
                              == candidate.row.raw();
            }
            const std::optional<IndexedVersion> winner =
                indexed_version(candidate.id);
            return winner.has_value() && !winner->tombstone()
                   && winner->sequence == candidate.sequence
                   && winner->value_raw == candidate.row.raw();
        }

        PartitionedManifestV2 manifest_;
        std::shared_ptr<ResourceSet> resources_;
        std::vector<OverlayEntry> overlay_;
        std::vector<std::size_t> overlay_put_order_;
        std::vector<SegmentView> segments_;
        NativeIdScanMetadata metadata_;
        std::size_t segment_position_{0};
        std::uint64_t row_position_{0};
        std::uint64_t emitted_count_{0};
        bool done_{false};
        const pid_t owner_pid_{::getpid()};
    };
    struct CleanerPlanRecord {
        std::uint64_t id{0};
        std::uint64_t sequence{0};
        std::uint64_t value_raw{0};
        bool relocate{false};
    };
    struct CleanerCandidate {
        PartitionedSegmentDescriptor descriptor;
        std::vector<CleanerPlanRecord> records;
        std::uint64_t relocated_rows{0};
        std::uint64_t garbage_rows{0};
        std::uint64_t source_bytes{0};
        std::uint64_t output_segment_count{0};
        std::uint64_t output_segment_bytes{0};
        std::uint64_t output_run_bytes{0};
        std::uint64_t reclaimable_bytes{0};
        std::uint64_t garbage_ratio_ppm{0};
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
            || catalog_.hash_seed != manifest_.hash_seed) {
            throw CorruptionError(
                "partitioned catalog does not match manifest or CURRENT");
        }
        std::uint64_t initial_rows = 0;
        const std::uint32_t initial_segment_count =
            manifest_.initial_active_wal_segment_id - 1;
        const std::uint64_t rows_per_segment =
            partitioned_segment_row_capacity(manifest_);
        const std::uint64_t expected_initial_segment_count =
            manifest_.initial_record_count / rows_per_segment
            + (manifest_.initial_record_count % rows_per_segment == 0
                   ? 0
                   : 1);
        if (expected_initial_segment_count != initial_segment_count) {
            throw CorruptionError(
                "format-7 manifest initial segment geometry is invalid");
        }
        std::unordered_set<std::uint32_t> saw_initial;
        saw_initial.reserve(std::min<std::size_t>(
            catalog_.segments.size(), initial_segment_count));
        for (const PartitionedSegmentDescriptor& descriptor :
             catalog_.segments) {
            if (descriptor.kind != PartitionedSegmentKind::Initial) {
                if (descriptor.row_ref_segment_id
                    < manifest_.initial_next_segment_id) {
                    throw CorruptionError(
                        "format-7 derived segment reuses the initial ID "
                        "range");
                }
                continue;
            }
            if (descriptor.row_ref_segment_id == 0
                || descriptor.row_ref_segment_id > initial_segment_count
                || !saw_initial.insert(
                        descriptor.row_ref_segment_id).second) {
                throw CorruptionError(
                    "format-7 initial segment topology disagrees with manifest");
            }
            const std::uint64_t first_row = checked_multiply(
                descriptor.row_ref_segment_id - 1, rows_per_segment,
                "format-7 initial segment first row");
            const std::uint64_t expected_rows = std::min(
                rows_per_segment,
                manifest_.initial_record_count - first_row);
            const std::uint64_t expected_data_size = checked_sum(
                kPartitionedHeaderSize,
                checked_multiply(
                    expected_rows, manifest_.record_size,
                    "format-7 initial data geometry"),
                "format-7 initial data geometry");
            const std::uint64_t expected_ids_size = checked_sum(
                kPartitionedHeaderSize,
                checked_multiply(
                    expected_rows, kPartitionedSegmentIdEntrySize,
                    "format-7 initial sidecar geometry"),
                "format-7 initial sidecar geometry");
            if (descriptor.row_count != expected_rows
                || descriptor.data_file_size != expected_data_size
                || descriptor.ids_file_size != expected_ids_size) {
                throw CorruptionError(
                    "format-7 active initial segment has unexpected "
                    "original geometry");
            }
            initial_rows = checked_sum(
                initial_rows, descriptor.row_count,
                "format-7 initial segment rows");
        }
        if (initial_rows > manifest_.initial_record_count
            || saw_initial.size() > initial_segment_count
            || catalog_.next_segment_id < manifest_.initial_next_segment_id) {
            throw CorruptionError(
                "format-7 initial segment set does not match manifest");
        }
        if (catalog_.indexed_sequence == 0) {
            const MutationWalEntry& entry = catalog_.wal_entries.front();
            if (entry.wal_id != manifest_.initial_active_wal_id
                || !entry.predecessor_wal_id.empty()
                || entry.segment_id
                       != manifest_.initial_active_wal_segment_id
                || entry.previous_sequence != 0 || entry.first_sequence != 1) {
                throw CorruptionError(
                    "format-7 unmaterialized WAL lineage disagrees with "
                    "manifest");
            }
        }
    }

    void open_and_replay_wals() {
        const auto started = SteadyClock::now();
        live_count_.store(
            catalog_.indexed_live_count, std::memory_order_relaxed);
        resources_->sources.reserve(catalog_.wal_entries.size());
        resources_->source_indices.reserve(catalog_.wal_entries.size());
        overlay_.reserve(catalog_.wal_entries.size());

        std::uint64_t replayed_wal_bytes = 0;
        std::uint64_t replayed_wal_operations = 0;

        for (const MutationWalEntry& entry : catalog_.wal_entries) {
            const bool active = entry.state == MutationWalState::Active;
            auto source = std::make_shared<WalSource>();
            source->entry = entry;
            source->header = partitioned_wal_header_for(manifest_, entry);
            source->path = wal_path(root_, entry.wal_id);
            source->fd = open_regular(source->path, writable_ && active);
            const std::uint64_t source_file_size =
                file_size(source->fd.get(), source->path);
            if (!active && source_file_size != entry.sealed_file_size) {
                throw CorruptionError(
                    "frozen partitioned WAL size disagrees with catalog");
            }
            if (source_file_size
                > open_options_.max_wal_bytes - replayed_wal_bytes) {
                throw StorageError(
                    "format-7 WAL backlog exceeds the configured open byte "
                    "limit; run bounded offline maintenance or raise the "
                    "explicit limit");
            }
            replayed_wal_bytes += source_file_size;
            const std::uint64_t remaining_operations =
                open_options_.max_wal_operations - replayed_wal_operations;
            const WalScanResult scan = scan_wal_file(
                source->fd.get(), source->path,
                ExpectedWalIdentity::from_trusted_header(source->header),
                active ? WalScanMode::Active : WalScanMode::Frozen,
                [this, &source](const RecoveredWalBatch& batch) {
                    apply_recovered_batch(*source, batch);
                },
                WalScanLimits{remaining_operations});
            source->valid_end = scan.valid_end;
            source->next_sequence = scan.next_sequence;
            source->operation_count = scan.committed_operation_count;
            replayed_wal_operations = checked_sum(
                replayed_wal_operations,
                scan.committed_operation_count,
                "format-7 replayed WAL operation count");
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
                resources_->sources.size();
            resources_->sources.push_back(std::move(source));
            if (!resources_->source_indices.emplace(
                     entry.segment_id, source_index).second) {
                throw CorruptionError(
                    "partitioned WAL segment ID is duplicated");
            }
        }
        if (resources_->sources.empty()
            || resources_->sources.back()->entry.state
                   != MutationWalState::Active) {
            throw CorruptionError(
                "partitioned WAL chain lacks a final active WAL");
        }
        resources_->active_index = resources_->sources.size() - 1;
        append_offset_ = active_wal().valid_end;
        next_sequence_ = active_wal().next_sequence;
        mutation_read_path_active_.store(
            !overlay_.empty() || !resources_->runs.empty(),
            std::memory_order_relaxed);
        open_stats_.wal_replay_seconds = elapsed_seconds(started);
    }

    void pin_segments(const bool retain_ids_for_verification) {
        const auto started = SteadyClock::now();
        resources_->initial_segments.reserve(catalog_.segments.size());
        resources_->materialized_segments.reserve(catalog_.segments.size());
        resources_->initial_segment_indices.reserve(catalog_.segments.size());
        resources_->materialized_segment_indices.reserve(
            catalog_.segments.size());
        for (const PartitionedSegmentDescriptor& descriptor :
             catalog_.segments) {
            if (descriptor.kind != PartitionedSegmentKind::Initial) {
                const std::size_t index =
                    resources_->materialized_segments.size();
                auto reader = std::make_shared<PartitionedSegmentReader>(
                    root_, manifest_, descriptor,
                    retain_ids_for_verification);
                resources_->materialized_segments.push_back(
                    std::move(reader));
                if (!resources_->materialized_segment_indices.emplace(
                         descriptor.row_ref_segment_id, index).second) {
                    throw CorruptionError(
                        "partitioned materialized segment ID is duplicated");
                }
                continue;
            }
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
            if (!resources_->initial_segment_indices.emplace(
                    descriptor.row_ref_segment_id,
                    resources_->initial_segments.size()).second) {
                throw CorruptionError("partitioned segment ID is duplicated");
            }
            ReadOnlyMapping ids_mapping(
                ids_fd.get(), descriptor.ids_file_size, ids_path);
            resources_->initial_segments.push_back(std::make_shared<Segment>(Segment{
                data_header,
                std::move(data_fd),
                data_path,
                ids_header,
                std::move(ids_mapping),
                ids_path}));
        }
        open_stats_.base_segments_seconds = elapsed_seconds(started);
    }

    [[nodiscard]] std::shared_ptr<Snapshot> open_snapshot_view(
        const PartitionSnapshotDescriptor& descriptor) const {
        const std::filesystem::path path = snapshot_path(root_, descriptor);
        FileDescriptor fd = open_regular(path);
        if (file_size(fd.get(), path) != descriptor.file_size) {
            throw CorruptionError("partition snapshot size changed");
        }
        ReadOnlyMapping mapping(fd.get(), descriptor.file_size, path);
        // The mapping pins this exact immutable inode, keeping descriptor
        // usage independent of partition count and catalog generations.
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
        return std::make_shared<Snapshot>(
            Snapshot{header, std::move(mapping)});
    }

    void pin_snapshots() {
        resources_->snapshots.reserve(catalog_.snapshots.size());
        for (const PartitionSnapshotDescriptor& descriptor :
             catalog_.snapshots) {
            open_stats_.index_file_bytes = checked_sum(
                open_stats_.index_file_bytes, descriptor.file_size,
                "partition snapshot bytes");
            resources_->snapshots.push_back(open_snapshot_view(descriptor));
        }
    }

    void pin_runs() {
        resources_->runs.reserve(catalog_.runs.size());
        for (const PartitionIndexRunDescriptor& descriptor : catalog_.runs) {
            if (descriptor.kind == PartitionedRunKind::LegacyDeltaV1) {
                throw CorruptionError(
                    "format-7 reader does not support legacy delta runs");
            }
            auto run = std::make_shared<PartitionIndexRunReader>(
                root_, manifest_.collection_id, manifest_.schema_id,
                manifest_.partition_bits, manifest_.hash_seed, descriptor);
            std::uint64_t active_entry_count = 0;
            for (std::uint32_t partition = 0;
                 partition != manifest_.partition_count; ++partition) {
                if (!partition_bitmap_contains(
                        descriptor.active_partitions, partition)) {
                    continue;
                }
                if (descriptor.epoch
                    <= catalog_.snapshots[partition].epoch) {
                    throw CorruptionError(
                        "active partition run is not newer than its "
                        "snapshot");
                }
                active_entry_count = checked_sum(
                    active_entry_count,
                    run->partition_records(partition).record_count,
                    "active partition run entries");
            }
            if (active_entry_count == 0
                || active_entry_count > descriptor.entry_count) {
                throw CorruptionError(
                    "partition run active entry count is invalid");
            }
            open_stats_.index_file_bytes = checked_sum(
                open_stats_.index_file_bytes, descriptor.file_size,
                "partition index run bytes");
            resources_->runs.push_back(RunView{
                std::move(run), descriptor, active_entry_count});
        }
    }

    void verify_segment_payloads() const {
        for (const auto& segment_pointer : resources_->initial_segments) {
            const Segment& segment = *segment_pointer;
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
        for (const auto& snapshot_pointer : resources_->snapshots) {
            const Snapshot& snapshot = *snapshot_pointer;
            if (crc32c(snapshot.mapping.bytes().subspan(
                    static_cast<std::size_t>(kPartitionedHeaderSize)))
                != snapshot.header.descriptor.body_crc32c) {
                throw CorruptionError("partition snapshot checksum mismatch");
            }
        }
    }

    void validate_snapshot_structure() {
        std::uint64_t occupied_total = 0;
        for (const auto& snapshot_pointer : resources_->snapshots) {
            const Snapshot& snapshot = *snapshot_pointer;
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
                if (sequence > catalog_.indexed_sequence) {
                    throw CorruptionError(
                        "partition snapshot sequence exceeds indexed state");
                }
                const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
                if (partition_for_hash(hash, manifest_.partition_bits)
                    != snapshot.header.descriptor.partition_id) {
                    throw CorruptionError(
                        "partition snapshot contains an ID in the wrong partition");
                }
                static_cast<void>(RowRef::decode(raw));

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
        std::uint64_t described_snapshot_rows = 0;
        for (const PartitionSnapshotDescriptor& descriptor :
             catalog_.snapshots) {
            described_snapshot_rows = checked_sum(
                described_snapshot_rows, descriptor.entry_count,
                "partition snapshot entry count");
        }
        if (occupied_total != described_snapshot_rows) {
            throw CorruptionError(
                "partition snapshot occupancy disagrees with catalog");
        }

        std::uint64_t authoritative_live = 0;
        const auto audit_candidate = [this, &authoritative_live](
                                         const std::uint64_t id,
                                         const std::uint64_t sequence,
                                         const std::uint64_t value_raw,
                                         const std::uint64_t source_epoch) {
            const std::optional<IndexedVersion> winner =
                indexed_version(id, *resources_);
            if (!winner.has_value()) {
                throw CorruptionError(
                    "partitioned index candidate has no authoritative winner");
            }
            if (winner->source_epoch != source_epoch) {
                return;
            }
            if (winner->sequence != sequence
                || winner->value_raw != value_raw) {
                throw CorruptionError(
                    "partitioned index winner disagrees with its source");
            }
            if (winner->tombstone()) {
                return;
            }
            validate_authoritative_row(id, *winner);
            authoritative_live = checked_sum(
                authoritative_live, 1,
                "partitioned authoritative live count");
        };
        for (const auto& snapshot_pointer : resources_->snapshots) {
            const Snapshot& snapshot = *snapshot_pointer;
            for (std::uint64_t slot = 0;
                 slot != snapshot.header.descriptor.capacity; ++slot) {
                if (control(snapshot, slot) == 1) {
                    audit_candidate(
                        slot_id(snapshot, slot), slot_sequence(snapshot, slot),
                        slot_value(snapshot, slot),
                        snapshot.header.descriptor.epoch);
                }
            }
        }
        for (const RunView& run : resources_->runs) {
            for (std::uint32_t partition = 0;
                 partition != manifest_.partition_count; ++partition) {
                if (!partition_bitmap_contains(
                        run.catalog.active_partitions, partition)) {
                    continue;
                }
                const PartitionRunTocEntry section =
                    run.file->partition_records(partition);
                for (std::uint64_t offset = 0;
                     offset != section.record_count; ++offset) {
                    const PartitionRunRecord record = run.file->record_at(
                        section.first_record + offset);
                    audit_candidate(
                        record.id, record.sequence, record.value_raw,
                        run.catalog.epoch);
                }
            }
        }
        if (authoritative_live != catalog_.indexed_live_count) {
            throw CorruptionError(
                "authoritative partitioned index live count disagrees with "
                "catalog");
        }
    }

    void validate_authoritative_row(
        const std::uint64_t id,
        const IndexedVersion& winner) const {
        const RowRef row = *winner.row_ref();
        const auto initial = resources_->initial_segment_indices.find(
            row.segment_id().value());
        if (initial != resources_->initial_segment_indices.end()) {
            const Segment& segment =
                *resources_->initial_segments[initial->second];
            if (row.row_index() >= segment.header.descriptor.row_count) {
                throw CorruptionError(
                    "authoritative RowRef exceeds its initial segment");
            }
            const std::size_t offset = static_cast<std::size_t>(
                segment.ids_header.payload_offset
                + static_cast<std::uint64_t>(row.row_index())
                      * kPartitionedSegmentIdEntrySize);
            if (get_u64(segment.ids_mapping.bytes(), offset) != id
                || get_u64(segment.ids_mapping.bytes(), offset + 8)
                       != winner.sequence) {
                throw CorruptionError(
                    "authoritative index winner disagrees with initial "
                    "sidecar");
            }
            return;
        }
        const auto derived = resources_->materialized_segment_indices.find(
            row.segment_id().value());
        if (derived == resources_->materialized_segment_indices.end()) {
            throw CorruptionError(
                "authoritative RowRef has no active immutable segment");
        }
        const PartitionedSegmentIdEntry sidecar =
            resources_->materialized_segments[derived->second]->sidecar_at(
                row.row_index());
        if (sidecar.id != id || sidecar.sequence != winner.sequence) {
            throw CorruptionError(
                "authoritative index winner disagrees with materialized "
                "sidecar");
        }
    }

    [[nodiscard]] WalSource& active_wal() {
        return *resources_->sources.at(resources_->active_index);
    }

    [[nodiscard]] const WalSource& active_wal() const {
        return *resources_->sources.at(resources_->active_index);
    }

    [[nodiscard]] std::optional<IndexedVersion> snapshot_version(
        const std::uint64_t id,
        const ResourceSet& resources) const {
        const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
        const std::uint64_t partition = partition_for_hash(
            hash, manifest_.partition_bits);
        const Snapshot& snapshot =
            *resources.snapshots[static_cast<std::size_t>(partition)];
        const std::uint64_t mask = snapshot.header.descriptor.capacity - 1;
        std::uint64_t slot = hash & mask;
        for (std::uint64_t probes = 0;
             probes != snapshot.header.descriptor.capacity; ++probes) {
            if (control(snapshot, slot) == 0) {
                return std::nullopt;
            }
            if (slot_id(snapshot, slot) == id) {
                return IndexedVersion{
                    slot_sequence(snapshot, slot), slot_value(snapshot, slot),
                    snapshot.header.descriptor.epoch};
            }
            slot = (slot + 1) & mask;
        }
        throw CorruptionError("partition snapshot lookup has no empty slot");
    }

    [[nodiscard]] std::optional<IndexedVersion> indexed_version(
        const std::uint64_t id,
        const ResourceSet& resources) const {
        std::optional<IndexedVersion> winner = snapshot_version(id, resources);
        const std::uint64_t hash = hash_id(id, manifest_.hash_seed);
        const auto partition = static_cast<std::uint32_t>(partition_for_hash(
            hash, manifest_.partition_bits));
        for (const RunView& run : resources.runs) {
            if (!partition_bitmap_contains(
                    run.catalog.active_partitions, partition)) {
                continue;
            }
            if (run.catalog.kind == PartitionedRunKind::LegacyDeltaV1
                || run.catalog.epoch
                       <= resources.snapshots[partition]
                              ->header.descriptor.epoch) {
                throw CorruptionError(
                    "active partition run is incompatible with its "
                    "authoritative snapshot");
            }
            const std::optional<PartitionRunRecord> record =
                run.file->lookup(partition, id);
            if (!record.has_value()) {
                continue;
            }
            const PartitionIndexRunDescriptor& descriptor = run.catalog;
            const IndexedVersion candidate{
                record->sequence, record->value_raw, descriptor.epoch};
            if (!winner.has_value()
                || candidate.sequence > winner->sequence) {
                winner = candidate;
                continue;
            }
            if (candidate.sequence < winner->sequence) {
                continue;
            }
            if (candidate.value_raw == winner->value_raw) {
                winner->source_epoch = std::max(
                    winner->source_epoch, candidate.source_epoch);
                continue;
            }
            bool valid_cleaner_relocation = false;
            if (descriptor.kind == PartitionedRunKind::Cleaner
                && descriptor.epoch > winner->source_epoch
                && candidate.value_raw != 0 && winner->value_raw != 0) {
                const RowRef previous = RowRef::decode(winner->value_raw);
                valid_cleaner_relocation =
                    previous.segment_id().value()
                    == descriptor.retired_row_ref_segment_id;
            }
            if (!valid_cleaner_relocation) {
                throw CorruptionError(
                    "partitioned index has conflicting values at one "
                    "sequence");
            }
            winner = candidate;
        }
        return winner;
    }

    [[nodiscard]] std::optional<RowRef> logical_lookup_unlocked(
        const std::uint64_t id,
        const ResourceSet& resources) const {
        const auto position = overlay_.find(id);
        if (position != overlay_.end() && position->second.has_value()) {
            return position->second->row_ref();
        }
        const std::optional<IndexedVersion> indexed =
            indexed_version(id, resources);
        return indexed.has_value() ? indexed->row_ref() : std::nullopt;
    }

    [[nodiscard]] PhysicalReadRequest resolve_base(
        const RowRef row,
        const std::uint64_t output_index,
        const ResourceSet& resources) const {
        const auto position = resources.initial_segment_indices.find(
            row.segment_id().value());
        if (position == resources.initial_segment_indices.end()) {
            throw CorruptionError("partition RowRef has no pinned segment");
        }
        const Segment& segment = *resources.initial_segments[position->second];
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
        const ResourceSet& resources) const {
        if (resources.initial_segment_indices.contains(
                row.segment_id().value())) {
            return resolve_base(row, output_index, resources);
        }
        const auto materialized = resources.materialized_segment_indices.find(
            row.segment_id().value());
        if (materialized != resources.materialized_segment_indices.end()) {
            return resources.materialized_segments[materialized->second]
                ->resolve(row, output_index);
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
            if (prior == overlay_.end()) {
                if (overlay_.size() >= open_options_.max_overlay_entries
                    || estimated_overlay_bytes(overlay_.size() + 1)
                           > open_options_.max_overlay_bytes) {
                    throw StorageError(
                        "format-7 mutation overlay exceeds the configured "
                        "open memory limit; run bounded offline maintenance "
                        "or raise the explicit limit");
                }
            }
            const bool existed =
                prior != overlay_.end() && prior->second.has_value()
                    ? prior->second->is_put()
                    : [&] {
                          const auto indexed = indexed_version(
                              operation.id, *resources_);
                          return indexed.has_value()
                                 && !indexed->tombstone();
                      }();
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

    void ensure_overlay_growth_unlocked(
        const std::size_t additional_entries) const {
        if (additional_entries
            > std::numeric_limits<std::size_t>::max() - overlay_.size()) {
            throw InvalidArgumentError("mutation overlay size overflows");
        }
        const std::size_t prospective =
            overlay_.size() + additional_entries;
        if (prospective > open_options_.max_overlay_entries
            || estimated_overlay_bytes(prospective)
                   > open_options_.max_overlay_bytes) {
            throw StorageError(
                "format-7 mutation would exceed the configured overlay "
                "memory limit before WAL append; run bounded maintenance or "
                "raise the explicit limit");
        }
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
        const std::uint64_t new_operation_count = checked_sum(
            active.operation_count,
            static_cast<std::uint64_t>(operations.size()),
            "active WAL operation count");
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
                active.operation_count = new_operation_count;
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

    void validate_materialization_anchor(
        const std::span<const MutationWalEntry> selected_wals,
        const std::uint64_t indexed_sequence,
        const Uuid128 terminal_wal_id) const {
        if (selected_wals.empty()
            || catalog_.indexed_sequence != indexed_sequence
            || catalog_.terminal_wal_id != terminal_wal_id
            || catalog_.wal_entries.size() <= selected_wals.size()) {
            throw StorageError(
                "partitioned catalog changed incompatibly during "
                "materialization");
        }
        for (std::size_t index = 0; index != selected_wals.size(); ++index) {
            if (catalog_.wal_entries[index] != selected_wals[index]) {
                throw StorageError(
                    "partitioned frozen WAL prefix changed during "
                    "materialization");
            }
        }
    }

    void validate_fold_anchor(
        const std::uint64_t indexed_sequence,
        const std::vector<PartitionSnapshotDescriptor>& snapshots,
        const std::vector<PartitionIndexRunDescriptor>& runs) const {
        if (catalog_.indexed_sequence != indexed_sequence
            || catalog_.snapshots != snapshots || catalog_.runs != runs) {
            throw StorageError(
                "partitioned indexed view changed during selective fold");
        }
    }

    [[nodiscard]] NativeMutationLogStats mutation_log_stats_locked(
        const bool rotated,
        const bool materialized = false,
        const std::uint64_t materialized_wal_count = 0,
        const std::uint64_t materialized_wal_input_bytes = 0,
        const bool materialization_budget_blocked = false,
        const bool compacted = false,
        const bool compaction_budget_blocked = false,
        const std::uint64_t compacted_partition_count = 0,
        const std::uint64_t compacted_run_section_count = 0,
        const std::uint64_t compacted_snapshot_input_entries = 0,
        const std::uint64_t compacted_snapshot_input_bytes = 0,
        const std::uint64_t compacted_output_entries = 0,
        const std::uint64_t compacted_output_bytes = 0,
        const std::uint64_t compacted_delta_input_entries = 0,
        const std::uint64_t compacted_delta_input_bytes = 0) const {
        NativeMutationLogStats stats;
        stats.catalog_managed = true;
        stats.delta_managed = true;
        stats.generation = catalog_.generation;
        stats.committed_sequence = next_sequence_ - 1;
        stats.materialized_sequence = catalog_.indexed_sequence;
        stats.active_wal_bytes = append_offset_;
        stats.active_wal_put_rows = active_wal().payload_offsets.size();
        stats.active_wal_operation_count = active_wal().operation_count;
        stats.active_wal_transaction_count =
            next_sequence_ - active_wal().header.first_sequence;
        stats.overlay_entry_count = overlay_.size();
        stats.estimated_overlay_bytes =
            estimated_overlay_bytes(stats.overlay_entry_count);
        stats.rotated = rotated;
        stats.materialized = materialized;
        stats.materialized_wal_count = materialized_wal_count;
        stats.materialized_wal_input_bytes =
            materialized_wal_input_bytes;
        stats.materialization_budget_blocked =
            materialization_budget_blocked;
        stats.compacted = compacted;
        stats.compaction_budget_blocked = compaction_budget_blocked;
        stats.compacted_partition_count = compacted_partition_count;
        stats.compacted_run_section_count = compacted_run_section_count;
        stats.compacted_snapshot_input_entries =
            compacted_snapshot_input_entries;
        stats.compacted_snapshot_input_bytes =
            compacted_snapshot_input_bytes;
        stats.compacted_output_entries = compacted_output_entries;
        stats.compacted_output_bytes = compacted_output_bytes;
        stats.compacted_delta_input_entries =
            compacted_delta_input_entries;
        stats.compacted_delta_input_bytes = compacted_delta_input_bytes;
        // Materialization intentionally leaves selected WAL pathnames for the
        // orphan collector. Readers already hold exact inode pins, so unlink
        // and namespace reclamation are a separate lifecycle transaction.
        stats.reclamation_complete = !materialized && !compacted;
        for (const auto& source : resources_->sources) {
            if (source->entry.state == MutationWalState::Frozen) {
                ++stats.frozen_wal_count;
                stats.frozen_wal_bytes = checked_sum(
                    stats.frozen_wal_bytes, source->valid_end,
                    "partitioned frozen WAL bytes");
                stats.frozen_wal_operation_count = checked_sum(
                    stats.frozen_wal_operation_count,
                    source->operation_count,
                    "partitioned frozen WAL operations");
            }
            stats.total_wal_bytes = checked_sum(
                stats.total_wal_bytes, source->valid_end,
                "partitioned WAL total bytes");
            stats.total_wal_operation_count = checked_sum(
                stats.total_wal_operation_count,
                source->operation_count,
                "partitioned WAL total operations");
        }
        for (const RunView& run : resources_->runs) {
            ++stats.delta_run_count;
            stats.delta_entry_count = checked_sum(
                stats.delta_entry_count, run.catalog.entry_count,
                "partitioned run entries");
            stats.total_delta_run_bytes = checked_sum(
                stats.total_delta_run_bytes, run.catalog.file_size,
                "partitioned run bytes");
        }
        for (const auto& segment : resources_->materialized_segments) {
            const PartitionedSegmentDescriptor& descriptor =
                segment->descriptor();
            stats.delta_put_rows = checked_sum(
                stats.delta_put_rows, descriptor.row_count,
                "partitioned materialized rows");
            const std::uint64_t bytes = checked_sum(
                descriptor.data_file_size, descriptor.ids_file_size,
                "partitioned materialized segment bytes");
            stats.total_delta_segment_bytes = checked_sum(
                stats.total_delta_segment_bytes, bytes,
                "partitioned materialized segment bytes");
        }
        stats.total_delta_bytes = checked_sum(
            stats.total_delta_run_bytes,
            stats.total_delta_segment_bytes,
            "partitioned mutation resource bytes");
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

            auto next_resources = std::make_shared<ResourceSet>();
            {
                std::shared_lock state(state_mutex_);
                next_resources->initial_segments = resources_->initial_segments;
                next_resources->initial_segment_indices =
                    resources_->initial_segment_indices;
                next_resources->materialized_segments =
                    resources_->materialized_segments;
                next_resources->materialized_segment_indices =
                    resources_->materialized_segment_indices;
                next_resources->snapshots = resources_->snapshots;
                next_resources->runs = resources_->runs;
                next_resources->sources = resources_->sources;
                next_resources->source_indices =
                    resources_->source_indices;
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
                resources_.swap(next_resources);
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
    NativeOpenOptions open_options_;
    NativeOpenStats open_stats_;
    AdvisoryLockFile writer_lock_;
    AdvisoryLockFile commit_lock_;
    AdvisoryLockFile view_lock_;
    std::shared_ptr<ResourceSet> resources_;
    mutable std::shared_mutex state_mutex_;
    std::unordered_map<std::uint64_t, OverlayValue> overlay_;
    std::atomic<bool> mutation_read_path_active_{false};
    std::atomic<std::uint64_t> live_count_{0};
    std::uint64_t append_offset_{kWalFileHeaderSize};
    std::uint64_t next_sequence_{1};
    mutable std::mutex mutation_mutex_;
    // Serializes every future index lifecycle transaction (materialization,
    // selective fold, and cleaner publication) while still allowing ordinary
    // WAL commits and rotations during long immutable-output builds.
    std::mutex lifecycle_mutex_;
    std::atomic<bool> poisoned_{false};
    bool writable_{false};
    const pid_t owner_pid_{::getpid()};
};

std::unique_ptr<PartitionedIdScan>
PartitionedIndexReader::Impl::scan_ids(
    const std::uint64_t max_snapshot_bytes) const {
    ensure_healthy();
    std::shared_ptr<ResourceSet> resource_pin;
    std::vector<IdScanState::OverlayEntry> overlay;
    NativeIdScanMetadata metadata;
    {
        std::shared_lock state(state_mutex_);
        ensure_healthy();
        resource_pin = resources_;
        // OverlayEntry is 32 bytes on the supported LP64 targets and the scan
        // additionally retains a size_t ordering vector. Sixty-four bytes per
        // map entry covers both vectors, allocator slack, and null publication
        // nodes for which reserve() still allocates capacity. SegmentView is
        // bounded separately because base-only scans may have an empty overlay.
        constexpr std::uint64_t kOverlaySnapshotBytesPerEntry = 64;
        constexpr std::uint64_t kSegmentSnapshotBytesPerEntry = 64;
        constexpr std::uint64_t kSnapshotFixedAllowance = 4096;
        const std::uint64_t overlay_entries = overlay_.size();
        const std::uint64_t segment_entries = checked_sum(
            checked_sum(
                resources_->initial_segments.size(),
                resources_->materialized_segments.size(),
                "ID scan segment memory estimate"),
            checked_multiply(
                resources_->sources.size(), 2,
                "ID scan segment memory estimate"),
            "ID scan segment memory estimate");
        std::uint64_t snapshot_bytes = kSnapshotFixedAllowance;
        if (overlay_entries
                > (std::numeric_limits<std::uint64_t>::max()
                   - snapshot_bytes)
                      / kOverlaySnapshotBytesPerEntry) {
            throw std::length_error("ID scan overlay memory estimate overflow");
        }
        snapshot_bytes +=
            overlay_entries * kOverlaySnapshotBytesPerEntry;
        if (segment_entries
                > (std::numeric_limits<std::uint64_t>::max()
                   - snapshot_bytes)
                      / kSegmentSnapshotBytesPerEntry) {
            throw std::length_error("ID scan segment memory estimate overflow");
        }
        snapshot_bytes +=
            segment_entries * kSegmentSnapshotBytesPerEntry;
        if (snapshot_bytes > max_snapshot_bytes) {
            throw std::length_error(
                "ID scan snapshot exceeds its admitted memory budget; "
                "materialize WALs or raise the server scan budget");
        }
        overlay.reserve(overlay_.size());
        for (const auto& [id, value] : overlay_) {
            // A null value is a preallocated publication node. It falls
            // through to the indexed view and is not part of this snapshot.
            if (value.has_value()) {
                overlay.push_back(IdScanState::OverlayEntry{id, *value});
            }
        }
        metadata.collection_id_low = manifest_.collection_id.low;
        metadata.collection_id_high = manifest_.collection_id.high;
        metadata.catalog_generation = catalog_.generation;
        metadata.committed_sequence = next_sequence_ - 1;
        metadata.live_count = live_count_.load(std::memory_order_relaxed);
    }
    return std::make_unique<PartitionedIdScan>(
        std::make_unique<IdScanState>(
            manifest_, std::move(resource_pin), std::move(overlay),
            metadata));
}

PartitionedIdScan::PartitionedIdScan(
    std::unique_ptr<PartitionedIdScanState> state)
    : state_(std::move(state)) {
    if (state_ == nullptr) {
        throw InvalidArgumentError("partitioned ID scan state is null");
    }
}

PartitionedIdScan::~PartitionedIdScan() = default;

const NativeIdScanMetadata& PartitionedIdScan::metadata() const noexcept {
    return state_->metadata();
}

std::uint64_t PartitionedIdScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<NativeIdLocation> locations) {
    return state_->read_into(
        ids, locations, NativeIdScan::StopPredicate{});
}

std::uint64_t PartitionedIdScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<NativeIdLocation> locations,
    const NativeIdScan::StopPredicate& stop_requested) {
    return state_->read_into(ids, locations, stop_requested);
}

bool PartitionedIdScan::done() const noexcept {
    return state_->done();
}

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
    const bool calibration,
    const NativeOpenOptions open_options)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), reader_options, writable, calibration,
          open_options)) {}
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
std::unique_ptr<PartitionedIdScan> PartitionedIndexReader::scan_ids(
    const std::uint64_t max_snapshot_bytes) const {
    return impl_->scan_ids(max_snapshot_bytes);
}
std::vector<std::uint8_t> PartitionedIndexReader::contains(
    const std::span<const std::uint64_t> ids) const {
    return impl_->contains(ids);
}
NativeCollectionIdentity PartitionedIndexReader::identity() const {
    return impl_->identity();
}
NativeMutationVerification PartitionedIndexReader::verify_committed_mutation(
    const std::uint64_t sequence,
    const NativeMutationFingerprint& expected) const {
    return impl_->verify_committed_mutation(sequence, expected);
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
NativeMutationLogStats PartitionedIndexReader::materialize_frozen_wals(
    const PhysicalReader& physical_reader,
    const std::uint64_t max_wals,
    const std::uint64_t max_input_bytes,
    const std::uint64_t max_input_entries,
    const std::uint64_t max_memory_bytes) {
    return impl_->materialize_frozen_wals(
        physical_reader, max_wals, max_input_bytes, max_input_entries,
        max_memory_bytes);
}
NativeMutationLogStats PartitionedIndexReader::fold_partitions(
    const std::uint64_t max_runs,
    const std::uint64_t max_input_bytes,
    const std::uint64_t max_input_entries) {
    return impl_->fold_partitions(
        max_runs, max_input_bytes, max_input_entries);
}
NativeSegmentCleaningStats PartitionedIndexReader::clean_segments(
    const PhysicalReader& physical_reader,
    const NativeSegmentCleaningOptions options) {
    return impl_->clean_segments(physical_reader, options);
}
PartitionedResourceInventory PartitionedIndexReader::collect_orphans() {
    return impl_->collect_orphans();
}
PartitionedOrphanCollectionResult PartitionedIndexReader::reclaim_orphans(
    const PartitionedOrphanCollectionOptions options) {
    return impl_->reclaim_orphans(options);
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
