#include "partitioned_index_run_storage.hpp"

#include "crc32c.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

constexpr std::size_t kWriteBufferBytes = 1U << 20;
constexpr std::uint64_t kMaximumSequence =
    std::numeric_limits<std::uint64_t>::max() - 1;

template <typename Error>
void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left + right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left * right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_align(
    const std::uint64_t value,
    const std::uint64_t alignment,
    const std::string_view description) {
    require<Error>(
        alignment != 0 && (alignment & (alignment - 1)) == 0,
        "partition run alignment is invalid");
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0
               ? value
               : checked_sum<Error>(
                     value, alignment - remainder, description);
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_partition_count(
    const std::uint32_t partition_bits) {
    require<Error>(
        partition_bits <= kPartitionedMaximumPartitionBits,
        "partition run exceeds the maximum partition bits");
    return std::uint64_t{1} << partition_bits;
}

[[nodiscard]] std::uint64_t partition_for_id(
    const std::uint64_t id,
    const std::uint64_t hash_seed,
    const std::uint32_t partition_bits) noexcept {
    if (partition_bits == 0) {
        return 0;
    }
    return hash_id(id, hash_seed) >> (64 - partition_bits);
}

[[nodiscard]] std::string uuid_token(const Uuid128 value) {
    constexpr std::array<char, 16> hex{
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::array<char, 32> result{};
    const auto append = [&result, &hex](
                            const std::uint64_t word,
                            const std::size_t offset) {
        for (std::size_t index = 0; index != 16; ++index) {
            const unsigned shift = static_cast<unsigned>((15 - index) * 4);
            result[offset + index] = hex[(word >> shift) & 0xfU];
        }
    };
    append(value.high, 0);
    append(value.low, 16);
    return std::string(result.begin(), result.end());
}

[[nodiscard]] int nofollow_flag() noexcept {
#if defined(O_NOFOLLOW)
    return O_NOFOLLOW;
#else
    return 0;
#endif
}

[[nodiscard]] FileDescriptor open_regular_nofollow(
    const std::filesystem::path& path) {
    const int fd = ::open(
        path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | nofollow_flag());
    if (fd < 0) {
#if defined(ELOOP)
        if (errno == ELOOP) {
            throw CorruptionError(
                "partition run path is a symlink: '" + path.string() + "'");
        }
#endif
        throw StorageError(system_error_message("cannot open partition run", path));
    }
    FileDescriptor result(fd);
    struct stat status {};
    if (::fstat(result.get(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect partition run", path));
    }
    require<CorruptionError>(
        S_ISREG(status.st_mode),
        "partition run resource is not a regular file");
    return result;
}

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize> read_header(
    const int fd,
    const std::filesystem::path& path) {
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    pread_exact(fd, bytes.data(), bytes.size(), 0, path);
    return bytes;
}

void bitmap_insert(
    const std::span<std::byte> bitmap,
    const std::uint64_t partition) noexcept {
    const std::size_t byte_index = static_cast<std::size_t>(partition / 8);
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    bitmap[byte_index] |= static_cast<std::byte>(mask);
}

[[nodiscard]] bool bitmap_contains(
    const std::span<const std::byte> bitmap,
    const std::uint64_t partition) noexcept {
    const std::size_t byte_index = static_cast<std::size_t>(partition / 8);
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    return (std::to_integer<std::uint8_t>(bitmap[byte_index]) & mask) != 0;
}

template <typename Error>
void validate_catalog_bitmaps(
    const PartitionIndexRunDescriptor& descriptor,
    const std::uint32_t partition_bits) {
    const std::uint64_t partition_count =
        checked_partition_count<Error>(partition_bits);
    const std::size_t expected_size = static_cast<std::size_t>(
        (partition_count + 7) / 8);
    require<Error>(
        descriptor.covered_partitions.size() == expected_size
            && descriptor.active_partitions.size() == expected_size,
        "partition run catalog bitmap has invalid size");
    bool covered_any = false;
    bool active_any = false;
    for (std::size_t index = 0; index != expected_size; ++index) {
        const std::uint8_t covered = std::to_integer<std::uint8_t>(
            descriptor.covered_partitions[index]);
        const std::uint8_t active = std::to_integer<std::uint8_t>(
            descriptor.active_partitions[index]);
        require<Error>(
            (active & static_cast<std::uint8_t>(~covered)) == 0,
            "partition run active partitions are not a subset of coverage");
        covered_any = covered_any || covered != 0;
        active_any = active_any || active != 0;
    }
    if (partition_count % 8 != 0) {
        const std::uint8_t allowed = static_cast<std::uint8_t>(
            (std::uint32_t{1} << (partition_count % 8)) - 1);
        require<Error>(
            (std::to_integer<std::uint8_t>(
                 descriptor.covered_partitions.back())
             & static_cast<std::uint8_t>(~allowed)) == 0
                && (std::to_integer<std::uint8_t>(
                        descriptor.active_partitions.back())
                    & static_cast<std::uint8_t>(~allowed)) == 0,
            "partition run catalog bitmap has nonzero padding bits");
    }
    require<Error>(covered_any, "partition run catalog coverage is empty");
    require<Error>(active_any, "partition run catalog active set is empty");
}

struct RunGeometry {
    std::uint64_t partition_count{0};
    std::uint64_t toc_size{0};
    std::uint64_t records_offset{0};
    std::uint64_t records_size{0};
    std::uint64_t file_size{0};
};

template <typename Error>
[[nodiscard]] RunGeometry run_geometry(
    const std::uint32_t partition_bits,
    const std::uint64_t entry_count) {
    RunGeometry result;
    result.partition_count =
        checked_partition_count<Error>(partition_bits);
    result.toc_size = checked_product<Error>(
        result.partition_count, kPartitionRunTocEntrySize,
        "partition run TOC size");
    result.records_offset = checked_align<Error>(
        checked_sum<Error>(
            kPartitionedHeaderSize, result.toc_size,
            "partition run TOC end"),
        kPartitionedHeaderSize, "partition run records offset");
    result.records_size = checked_product<Error>(
        entry_count, kPartitionRunRecordSize,
        "partition run records size");
    result.file_size = checked_sum<Error>(
        result.records_offset, result.records_size,
        "partition run file size");
    return result;
}

[[nodiscard]] PartitionIndexRunDescriptor scalar_descriptor(
    PartitionIndexRunDescriptor descriptor) {
    descriptor.covered_partitions.clear();
    descriptor.active_partitions.clear();
    return descriptor;
}

void validate_build_spec_is_scalar(
    const PartitionIndexRunBuildSpec& spec) {
    require<InvalidArgumentError>(
        !spec.collection_id.empty() && !spec.schema_id.empty(),
        "partition run collection or schema identity is empty");
    require<InvalidArgumentError>(
        spec.descriptor.entry_count == 0
            && spec.descriptor.file_size == 0
            && spec.descriptor.body_crc32c == 0
            && spec.descriptor.covered_partitions.empty()
            && spec.descriptor.active_partitions.empty(),
        "partition run build spec contains writer-derived metadata");
}

class ReadOnlyMapping {
public:
    ReadOnlyMapping() = default;
    ~ReadOnlyMapping() { reset(); }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

    void map(
        const int fd,
        const std::uint64_t size,
        const std::filesystem::path& path) {
        require<CorruptionError>(
            size != 0 && size <= std::numeric_limits<std::size_t>::max(),
            "partition run does not fit the process address space");
        address_ = ::mmap(
            nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE,
            fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw StorageError(
                system_error_message("cannot mmap partition run", path));
        }
        size_ = static_cast<std::size_t>(size);
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {
            static_cast<const std::byte*>(address_),
            size_};
    }

private:
    void reset() noexcept {
        if (address_ != nullptr) {
            static_cast<void>(::munmap(address_, size_));
            address_ = nullptr;
            size_ = 0;
        }
    }

    void* address_{nullptr};
    std::size_t size_{0};
};

struct SortableRecord {
    std::uint32_t partition{0};
    PartitionRunRecord record;
};

}  // namespace

std::filesystem::path partitioned_index_runs_directory(
    const std::filesystem::path& store_path) {
    return store_path / "index" / "runs";
}

std::filesystem::path partitioned_index_run_path(
    const std::filesystem::path& store_path,
    const Uuid128 run_id) {
    return partitioned_index_runs_directory(store_path)
           / ("run-" + uuid_token(run_id) + ".idx");
}

PartitionIndexRunDescriptor write_partitioned_index_run(
    const std::filesystem::path& store_path,
    const PartitionIndexRunBuildSpec& spec,
    const std::span<const PartitionRunRecord> records) {
    validate_build_spec_is_scalar(spec);
    const RunGeometry geometry = run_geometry<InvalidArgumentError>(
        spec.partition_bits, static_cast<std::uint64_t>(records.size()));
    require<InvalidArgumentError>(
        !records.empty(), "partition run cannot be empty");

    std::vector<SortableRecord> sorted;
    sorted.reserve(records.size());
    for (const PartitionRunRecord& record : records) {
        require<InvalidArgumentError>(
            record.sequence >= spec.descriptor.first_sequence
                && record.sequence <= spec.descriptor.last_sequence
                && record.sequence <= kMaximumSequence,
            "partition run record sequence is outside the run interval");
        // Validate a persisted RowRef (or the canonical zero tombstone) before
        // any destination is created.
        static_cast<void>(encode_partition_run_record(record));
        sorted.push_back(SortableRecord{
            static_cast<std::uint32_t>(partition_for_id(
                record.id, spec.hash_seed, spec.partition_bits)),
            record});
    }
    std::sort(
        sorted.begin(), sorted.end(),
        [](const SortableRecord& left, const SortableRecord& right) {
            if (left.partition != right.partition) {
                return left.partition < right.partition;
            }
            return left.record.id < right.record.id;
        });
    for (std::size_t index = 1; index != sorted.size(); ++index) {
        require<InvalidArgumentError>(
            sorted[index - 1].record.id != sorted[index].record.id,
            "partition run contains a duplicate ID");
    }

    std::vector<PartitionRunTocEntry> toc(
        static_cast<std::size_t>(geometry.partition_count));
    std::uint64_t cursor = 0;
    for (std::uint64_t partition = 0;
         partition != geometry.partition_count; ++partition) {
        const std::uint64_t first = cursor;
        while (cursor != sorted.size()
               && sorted[static_cast<std::size_t>(cursor)].partition
                      == partition) {
            ++cursor;
        }
        toc[static_cast<std::size_t>(partition)] =
            PartitionRunTocEntry{first, cursor - first};
    }
    require<StorageError>(
        cursor == sorted.size(),
        "partition run writer failed to route every record");

    PartitionIndexRunDescriptor descriptor = spec.descriptor;
    descriptor.entry_count = static_cast<std::uint64_t>(records.size());
    descriptor.file_size = geometry.file_size;
    descriptor.covered_partitions.assign(
        static_cast<std::size_t>((geometry.partition_count + 7) / 8),
        std::byte{0});
    for (std::uint64_t partition = 0;
         partition != geometry.partition_count; ++partition) {
        if (toc[static_cast<std::size_t>(partition)].record_count != 0) {
            bitmap_insert(descriptor.covered_partitions, partition);
        }
    }
    descriptor.active_partitions = descriptor.covered_partitions;

    PartitionIndexRunHeaderV1 header;
    header.collection_id = spec.collection_id;
    header.schema_id = spec.schema_id;
    header.descriptor = scalar_descriptor(descriptor);
    header.partition_bits = spec.partition_bits;
    header.hash_seed = spec.hash_seed;
    header.toc_offset = kPartitionedHeaderSize;
    header.toc_size = geometry.toc_size;
    header.records_offset = geometry.records_offset;
    header.records_size = geometry.records_size;
    // Validate all scalar descriptor rules and exact geometry before creating
    // the immutable destination.
    static_cast<void>(encode_partition_index_run_header_v1(header));

    const std::filesystem::path destination =
        partitioned_index_run_path(store_path, descriptor.run_id);
    FileDescriptor fd;
    bool owns_destination = false;
    try {
        fd = create_exclusive(destination);
        owns_destination = true;
        const std::array<std::byte, kPartitionedHeaderSize> zero_header{};
        write_all(
            fd.get(), zero_header.data(), zero_header.size(), destination);

        Crc32c body_checksum;
        std::vector<std::byte> toc_bytes(
            static_cast<std::size_t>(geometry.toc_size));
        for (std::size_t index = 0; index != toc.size(); ++index) {
            const auto encoded = encode_partition_run_toc_entry(toc[index]);
            std::copy(
                encoded.begin(), encoded.end(),
                toc_bytes.begin()
                    + static_cast<std::ptrdiff_t>(
                          index * kPartitionRunTocEntrySize));
        }
        body_checksum.update(toc_bytes);
        write_all(
            fd.get(), toc_bytes.data(), toc_bytes.size(), destination);

        const std::uint64_t toc_end =
            kPartitionedHeaderSize + geometry.toc_size;
        std::uint64_t padding = geometry.records_offset - toc_end;
        const std::array<std::byte, kPartitionedHeaderSize> zeroes{};
        while (padding != 0) {
            const std::size_t count = static_cast<std::size_t>(
                std::min<std::uint64_t>(padding, zeroes.size()));
            const auto chunk = std::span<const std::byte>(zeroes).first(count);
            body_checksum.update(chunk);
            write_all(fd.get(), chunk.data(), chunk.size(), destination);
            padding -= count;
        }

        constexpr std::size_t records_per_batch =
            kWriteBufferBytes / kPartitionRunRecordSize;
        std::vector<std::byte> record_bytes;
        for (std::size_t begin = 0; begin != sorted.size();) {
            const std::size_t count = std::min(
                records_per_batch, sorted.size() - begin);
            record_bytes.resize(
                count * static_cast<std::size_t>(kPartitionRunRecordSize));
            for (std::size_t local = 0; local != count; ++local) {
                const auto encoded = encode_partition_run_record(
                    sorted[begin + local].record);
                std::copy(
                    encoded.begin(), encoded.end(),
                    record_bytes.begin()
                        + static_cast<std::ptrdiff_t>(
                              local * kPartitionRunRecordSize));
            }
            body_checksum.update(record_bytes);
            write_all(
                fd.get(), record_bytes.data(), record_bytes.size(),
                destination);
            begin += count;
        }

        descriptor.body_crc32c = body_checksum.value();
        header.descriptor = scalar_descriptor(descriptor);
        const auto header_bytes =
            encode_partition_index_run_header_v1(header);
        pwrite_all(
            fd.get(), header_bytes.data(), header_bytes.size(), 0,
            destination);
        require<StorageError>(
            file_size(fd.get(), destination) == descriptor.file_size,
            "partition run writer produced an unexpected file size");
        sync_file(fd.get(), destination);
        fd.reset();
        owns_destination = false;
    } catch (...) {
        fd.reset();
        if (owns_destination) {
            std::error_code ignored;
            static_cast<void>(std::filesystem::remove(destination, ignored));
        }
        throw;
    }
    return descriptor;
}

class PartitionIndexRunReader::Impl {
public:
    Impl(
        std::filesystem::path store_path,
        const Uuid128 collection_id,
        const Uuid128 schema_id,
        const std::uint32_t partition_bits,
        const std::uint64_t hash_seed,
        PartitionIndexRunDescriptor expected_descriptor)
        : descriptor_(std::move(expected_descriptor)),
          partition_bits_(partition_bits),
          hash_seed_(hash_seed) {
        validate_catalog_bitmaps<CorruptionError>(
            descriptor_, partition_bits_);
        const RunGeometry geometry = run_geometry<CorruptionError>(
            partition_bits_, descriptor_.entry_count);
        records_offset_ = geometry.records_offset;
        require<CorruptionError>(
            descriptor_.file_size == geometry.file_size,
            "partition run catalog file geometry is invalid");

        PartitionIndexRunHeaderV1 expected_header;
        expected_header.collection_id = collection_id;
        expected_header.schema_id = schema_id;
        expected_header.descriptor = scalar_descriptor(descriptor_);
        expected_header.partition_bits = partition_bits_;
        expected_header.hash_seed = hash_seed_;
        expected_header.toc_offset = kPartitionedHeaderSize;
        expected_header.toc_size = geometry.toc_size;
        expected_header.records_offset = geometry.records_offset;
        expected_header.records_size = geometry.records_size;
        try {
            static_cast<void>(
                encode_partition_index_run_header_v1(expected_header));
        } catch (const InvalidArgumentError& error) {
            throw CorruptionError(error.what());
        }

        path_ = partitioned_index_run_path(store_path, descriptor_.run_id);
        FileDescriptor fd = open_regular_nofollow(path_);
        require<CorruptionError>(
            file_size(fd.get(), path_) == descriptor_.file_size,
            "partition run file size does not match catalog");
        const PartitionIndexRunHeaderV1 actual_header =
            decode_partition_index_run_header_v1(read_header(fd.get(), path_));
        require<CorruptionError>(
            actual_header.collection_id == collection_id
                && actual_header.schema_id == schema_id
                && actual_header.partition_bits == partition_bits_
                && actual_header.hash_seed == hash_seed_
                && actual_header.descriptor
                       == scalar_descriptor(descriptor_),
            "partition run header does not match catalog identity");
        mapping_.map(fd.get(), descriptor_.file_size, path_);
        // POSIX mappings remain valid after close.  Do this before any O(N)
        // checksum or structural validation so no descriptor is retained.
        fd.reset();

        const std::span<const std::byte> bytes = mapping_.bytes();
        require<CorruptionError>(
            crc32c(bytes.subspan(kPartitionedHeaderSize))
                == descriptor_.body_crc32c,
            "partition run body checksum mismatch");
        const std::uint64_t toc_end =
            kPartitionedHeaderSize + geometry.toc_size;
        for (std::uint64_t offset = toc_end;
             offset != geometry.records_offset; ++offset) {
            require<CorruptionError>(
                bytes[static_cast<std::size_t>(offset)] == std::byte{0},
                "partition run alignment padding is nonzero");
        }

        toc_.reserve(static_cast<std::size_t>(geometry.partition_count));
        std::vector<std::byte> derived_coverage(
            descriptor_.covered_partitions.size(), std::byte{0});
        std::uint64_t expected_first = 0;
        for (std::uint64_t partition = 0;
             partition != geometry.partition_count; ++partition) {
            const std::uint64_t offset =
                kPartitionedHeaderSize
                + partition * kPartitionRunTocEntrySize;
            const PartitionRunTocEntry entry =
                decode_partition_run_toc_entry(bytes.subspan(
                    static_cast<std::size_t>(offset),
                    static_cast<std::size_t>(kPartitionRunTocEntrySize)));
            require<CorruptionError>(
                entry.first_record == expected_first
                    && entry.record_count
                           <= descriptor_.entry_count - expected_first,
                "partition run TOC is not contiguous or is out of bounds");
            expected_first += entry.record_count;
            if (entry.record_count != 0) {
                bitmap_insert(derived_coverage, partition);
            }
            toc_.push_back(entry);
        }
        require<CorruptionError>(
            expected_first == descriptor_.entry_count,
            "partition run TOC does not cover every record");
        require<CorruptionError>(
            derived_coverage == descriptor_.covered_partitions,
            "partition run TOC coverage does not match catalog");

        for (std::uint64_t partition = 0;
             partition != geometry.partition_count; ++partition) {
            const PartitionRunTocEntry entry =
                toc_[static_cast<std::size_t>(partition)];
            std::uint64_t previous_id = 0;
            bool have_previous = false;
            for (std::uint64_t local = 0;
                 local != entry.record_count; ++local) {
                const PartitionRunRecord record =
                    decode_record(entry.first_record + local);
                require<CorruptionError>(
                    !have_previous || previous_id < record.id,
                    "partition run IDs are duplicated or not strictly sorted");
                require<CorruptionError>(
                    partition_for_id(record.id, hash_seed_, partition_bits_)
                        == partition,
                    "partition run record is stored in the wrong partition");
                require<CorruptionError>(
                    record.sequence >= descriptor_.first_sequence
                        && record.sequence <= descriptor_.last_sequence,
                    "partition run record sequence is outside the run interval");
                previous_id = record.id;
                have_previous = true;
            }
        }
    }

    [[nodiscard]] const PartitionIndexRunDescriptor& descriptor() const noexcept {
        return descriptor_;
    }

    [[nodiscard]] std::uint64_t partition_count() const noexcept {
        return toc_.size();
    }

    [[nodiscard]] PartitionRunTocEntry partition_records(
        const std::uint32_t partition) const {
        if (partition >= toc_.size()) {
            throw InvalidArgumentError("partition run partition is out of range");
        }
        return toc_[partition];
    }

    [[nodiscard]] PartitionRunRecord record_at(
        const std::uint64_t index) const {
        if (index >= descriptor_.entry_count) {
            throw InvalidArgumentError("partition run record index is out of range");
        }
        return decode_record(index);
    }

    [[nodiscard]] std::optional<PartitionRunRecord> lookup(
        const std::uint32_t partition,
        const std::uint64_t id) const {
        const PartitionRunTocEntry entry = partition_records(partition);
        if (!bitmap_contains(descriptor_.active_partitions, partition)) {
            return std::nullopt;
        }
        if (partition_for_id(id, hash_seed_, partition_bits_) != partition) {
            return std::nullopt;
        }
        std::uint64_t first = entry.first_record;
        std::uint64_t last = entry.first_record + entry.record_count;
        while (first != last) {
            const std::uint64_t middle = first + (last - first) / 2;
            const PartitionRunRecord record = decode_record(middle);
            if (record.id < id) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        if (first == entry.first_record + entry.record_count) {
            return std::nullopt;
        }
        const PartitionRunRecord record = decode_record(first);
        return record.id == id
                   ? std::optional<PartitionRunRecord>(record)
                   : std::nullopt;
    }

private:
    [[nodiscard]] PartitionRunRecord decode_record(
        const std::uint64_t index) const {
        const std::uint64_t offset =
            records_offset() + index * kPartitionRunRecordSize;
        return decode_partition_run_record(mapping_.bytes().subspan(
            static_cast<std::size_t>(offset),
            static_cast<std::size_t>(kPartitionRunRecordSize)));
    }

    [[nodiscard]] std::uint64_t records_offset() const noexcept {
        return records_offset_;
    }

    PartitionIndexRunDescriptor descriptor_;
    std::uint32_t partition_bits_{0};
    std::uint64_t hash_seed_{0};
    std::uint64_t records_offset_{0};
    std::filesystem::path path_;
    ReadOnlyMapping mapping_;
    std::vector<PartitionRunTocEntry> toc_;
};

PartitionIndexRunReader::PartitionIndexRunReader(
    std::filesystem::path store_path,
    const Uuid128 collection_id,
    const Uuid128 schema_id,
    const std::uint32_t partition_bits,
    const std::uint64_t hash_seed,
    PartitionIndexRunDescriptor expected_descriptor)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), collection_id, schema_id, partition_bits,
          hash_seed, std::move(expected_descriptor))) {}

PartitionIndexRunReader::~PartitionIndexRunReader() = default;
PartitionIndexRunReader::PartitionIndexRunReader(
    PartitionIndexRunReader&&) noexcept = default;
PartitionIndexRunReader& PartitionIndexRunReader::operator=(
    PartitionIndexRunReader&&) noexcept = default;

const PartitionIndexRunDescriptor&
PartitionIndexRunReader::descriptor() const noexcept {
    return impl_->descriptor();
}

std::uint64_t PartitionIndexRunReader::record_count() const noexcept {
    return impl_->descriptor().entry_count;
}

std::uint64_t PartitionIndexRunReader::partition_count() const noexcept {
    return impl_->partition_count();
}

PartitionRunTocEntry PartitionIndexRunReader::partition_records(
    const std::uint32_t partition) const {
    return impl_->partition_records(partition);
}

PartitionRunRecord PartitionIndexRunReader::record_at(
    const std::uint64_t index) const {
    return impl_->record_at(index);
}

std::optional<PartitionRunRecord> PartitionIndexRunReader::lookup(
    const std::uint32_t partition,
    const std::uint64_t id) const {
    return impl_->lookup(partition, id);
}

}  // namespace throwdb::detail
