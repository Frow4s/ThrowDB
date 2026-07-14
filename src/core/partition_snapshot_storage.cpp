#include "partition_snapshot_storage.hpp"

#include "crc32c.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

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
        "partition snapshot alignment is invalid");
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0
               ? value
               : checked_sum<Error>(
                     value, alignment - remainder, description);
}

template <typename Error>
[[nodiscard]] std::uint64_t partition_count(
    const std::uint32_t partition_bits) {
    require<Error>(
        partition_bits <= kPartitionedMaximumPartitionBits,
        "partition snapshot exceeds the maximum partition bits");
    return std::uint64_t{1} << partition_bits;
}

[[nodiscard]] std::uint32_t partition_for_id(
    const std::uint64_t id,
    const std::uint64_t hash_seed,
    const std::uint32_t partition_bits) noexcept {
    if (partition_bits == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(
        hash_id(id, hash_seed) >> (64 - partition_bits));
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

struct SnapshotGeometry {
    std::uint64_t lookup_offset{0};
    std::uint64_t sequence_offset{0};
    std::uint64_t file_size{0};
};

template <typename Error>
[[nodiscard]] SnapshotGeometry snapshot_geometry(
    const std::uint64_t capacity) {
    require<Error>(
        capacity >= 8 && (capacity & (capacity - 1)) == 0,
        "partition snapshot capacity is not a power of two of at least eight");
    const std::uint64_t controls_end = checked_sum<Error>(
        kPartitionedHeaderSize, capacity,
        "partition snapshot controls end");
    SnapshotGeometry result;
    result.lookup_offset = checked_align<Error>(
        controls_end, kPartitionedHeaderSize,
        "partition snapshot lookup offset");
    const std::uint64_t lookup_end = checked_sum<Error>(
        result.lookup_offset,
        checked_product<Error>(
            capacity, kPartitionSnapshotLookupSlotSize,
            "partition snapshot lookup bytes"),
        "partition snapshot lookup end");
    result.sequence_offset = checked_align<Error>(
        lookup_end, kPartitionedHeaderSize,
        "partition snapshot sequence offset");
    result.file_size = checked_sum<Error>(
        result.sequence_offset,
        checked_product<Error>(
            capacity, kPartitionSnapshotSequenceSlotSize,
            "partition snapshot sequence bytes"),
        "partition snapshot file size");
    return result;
}

[[nodiscard]] std::uint64_t choose_capacity(const std::uint64_t count) {
    std::uint64_t capacity = 8;
    while (count
           > (capacity / 10) * 7 + ((capacity % 10) * 7) / 10) {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / 2) {
            throw InvalidArgumentError(
                "partition snapshot capacity overflows uint64");
        }
        capacity *= 2;
    }
    return capacity;
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
                "partition snapshot path is a symlink: '" + path.string()
                + "'");
        }
#endif
        throw StorageError(
            system_error_message("cannot open partition snapshot", path));
    }
    FileDescriptor result(fd);
    struct stat status {};
    if (::fstat(result.get(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect partition snapshot", path));
    }
    require<CorruptionError>(
        S_ISREG(status.st_mode),
        "partition snapshot resource is not a regular file");
    return result;
}

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize> read_header(
    const int fd,
    const std::filesystem::path& path) {
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    pread_exact(fd, bytes.data(), bytes.size(), 0, path);
    return bytes;
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
            "partition snapshot does not fit the process address space");
        address_ = ::mmap(
            nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE,
            fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw StorageError(
                system_error_message("cannot mmap partition snapshot", path));
        }
        size_ = static_cast<std::size_t>(size);
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {static_cast<const std::byte*>(address_), size_};
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

}  // namespace

std::filesystem::path partitioned_partition_snapshots_directory(
    const std::filesystem::path& store_path) {
    return store_path / "index" / "partitions";
}

std::filesystem::path partitioned_partition_snapshot_path(
    const std::filesystem::path& store_path,
    const PartitionSnapshotDescriptor& descriptor) {
    return partitioned_partition_snapshots_directory(store_path)
           / ("partition-" + std::to_string(descriptor.partition_id) + "-"
              + uuid_token(descriptor.snapshot_id) + ".idx");
}

PartitionSnapshotDescriptor write_partition_snapshot(
    const std::filesystem::path& store_path,
    const PartitionSnapshotBuildSpec& spec,
    const std::span<const PartitionRunRecord> records) {
    require<InvalidArgumentError>(
        !spec.collection_id.empty() && !spec.schema_id.empty()
            && !spec.snapshot_id.empty(),
        "partition snapshot identity is empty");
    const std::uint64_t partitions =
        partition_count<InvalidArgumentError>(spec.partition_bits);
    require<InvalidArgumentError>(
        spec.partition_id < partitions,
        "partition snapshot partition ID is outside the layout");
    require<InvalidArgumentError>(
        spec.epoch != 0, "selective partition snapshot epoch is zero");

    const std::uint64_t capacity = choose_capacity(records.size());
    const std::uint32_t local_bits = static_cast<std::uint32_t>(
        std::bit_width(capacity) - 1);
    require<InvalidArgumentError>(
        local_bits <= 64 - spec.partition_bits,
        "partition snapshot capacity overlaps partition hash bits");
    const SnapshotGeometry geometry =
        snapshot_geometry<InvalidArgumentError>(capacity);
    require<InvalidArgumentError>(
        geometry.file_size <= std::numeric_limits<std::size_t>::max(),
        "partition snapshot does not fit memory");
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(geometry.file_size));
    const std::uint64_t mask = capacity - 1;
    for (const PartitionRunRecord& record : records) {
        require<InvalidArgumentError>(
            !record.tombstone(),
            "authoritative partition snapshot contains a tombstone");
        require<InvalidArgumentError>(
            record.sequence <= kMaximumSequence,
            "partition snapshot record uses reserved sequence");
        require<InvalidArgumentError>(
            partition_for_id(record.id, spec.hash_seed, spec.partition_bits)
                == spec.partition_id,
            "partition snapshot record routes to a different partition");
        try {
            static_cast<void>(RowRef::decode(record.value_raw));
        } catch (const CorruptionError& error) {
            throw InvalidArgumentError(error.what());
        }

        std::uint64_t slot = hash_id(record.id, spec.hash_seed) & mask;
        while (bytes[static_cast<std::size_t>(
                   kPartitionedHeaderSize + slot)] != std::byte{0}) {
            const std::size_t existing_offset = static_cast<std::size_t>(
                geometry.lookup_offset
                + slot * kPartitionSnapshotLookupSlotSize);
            require<InvalidArgumentError>(
                get_u64(bytes, existing_offset) != record.id,
                "partition snapshot contains a duplicate ID");
            slot = (slot + 1) & mask;
        }
        bytes[static_cast<std::size_t>(kPartitionedHeaderSize + slot)] =
            std::byte{1};
        const std::size_t lookup_offset = static_cast<std::size_t>(
            geometry.lookup_offset
            + slot * kPartitionSnapshotLookupSlotSize);
        put_u64(bytes, lookup_offset, record.id);
        put_u64(bytes, lookup_offset + 8, record.value_raw);
        put_u64(
            bytes,
            static_cast<std::size_t>(
                geometry.sequence_offset
                + slot * kPartitionSnapshotSequenceSlotSize),
            record.sequence);
    }

    PartitionSnapshotDescriptor descriptor;
    descriptor.snapshot_id = spec.snapshot_id;
    descriptor.partition_id = spec.partition_id;
    descriptor.epoch = spec.epoch;
    descriptor.entry_count = records.size();
    descriptor.capacity = capacity;
    descriptor.file_size = geometry.file_size;
    descriptor.body_crc32c = crc32c(
        std::span<const std::byte>(bytes).subspan(kPartitionedHeaderSize));
    PartitionSnapshotHeaderV1 header;
    header.collection_id = spec.collection_id;
    header.schema_id = spec.schema_id;
    header.descriptor = descriptor;
    header.partition_bits = spec.partition_bits;
    header.hash_seed = spec.hash_seed;
    header.lookup_slots_offset = geometry.lookup_offset;
    header.sequence_slots_offset = geometry.sequence_offset;
    const auto encoded = encode_partition_snapshot_header_v1(header);
    std::copy(encoded.begin(), encoded.end(), bytes.begin());

    const std::filesystem::path destination =
        partitioned_partition_snapshot_path(store_path, descriptor);
    FileDescriptor fd;
    bool owns_destination = false;
    try {
        fd = create_exclusive(destination);
        owns_destination = true;
        write_all(fd.get(), bytes.data(), bytes.size(), destination);
        require<StorageError>(
            file_size(fd.get(), destination) == descriptor.file_size,
            "partition snapshot writer produced an unexpected file size");
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

class PartitionSnapshotReader::Impl {
public:
    Impl(
        std::filesystem::path store_path,
        const Uuid128 collection_id,
        const Uuid128 schema_id,
        const std::uint32_t partition_bits,
        const std::uint64_t hash_seed,
        PartitionSnapshotDescriptor expected_descriptor)
        : descriptor_(std::move(expected_descriptor)),
          partition_bits_(partition_bits),
          hash_seed_(hash_seed) {
        const std::uint64_t partitions =
            partition_count<CorruptionError>(partition_bits_);
        require<CorruptionError>(
            descriptor_.partition_id < partitions,
            "partition snapshot catalog partition is outside the layout");
        geometry_ = snapshot_geometry<CorruptionError>(descriptor_.capacity);
        require<CorruptionError>(
            descriptor_.file_size == geometry_.file_size,
            "partition snapshot catalog geometry is invalid");

        PartitionSnapshotHeaderV1 expected_header;
        expected_header.collection_id = collection_id;
        expected_header.schema_id = schema_id;
        expected_header.descriptor = descriptor_;
        expected_header.partition_bits = partition_bits_;
        expected_header.hash_seed = hash_seed_;
        expected_header.lookup_slots_offset = geometry_.lookup_offset;
        expected_header.sequence_slots_offset = geometry_.sequence_offset;
        try {
            static_cast<void>(encode_partition_snapshot_header_v1(
                expected_header));
        } catch (const InvalidArgumentError& error) {
            throw CorruptionError(error.what());
        }

        path_ = partitioned_partition_snapshot_path(store_path, descriptor_);
        FileDescriptor fd = open_regular_nofollow(path_);
        require<CorruptionError>(
            file_size(fd.get(), path_) == descriptor_.file_size,
            "partition snapshot file size does not match catalog");
        const PartitionSnapshotHeaderV1 actual_header =
            decode_partition_snapshot_header_v1(read_header(fd.get(), path_));
        require<CorruptionError>(
            actual_header.collection_id == collection_id
                && actual_header.schema_id == schema_id
                && actual_header.descriptor == descriptor_
                && actual_header.partition_bits == partition_bits_
                && actual_header.hash_seed == hash_seed_,
            "partition snapshot header does not match catalog identity");
        mapping_.map(fd.get(), descriptor_.file_size, path_);
        fd.reset();

        const std::span<const std::byte> bytes = mapping_.bytes();
        require<CorruptionError>(
            crc32c(bytes.subspan(kPartitionedHeaderSize))
                == descriptor_.body_crc32c,
            "partition snapshot body checksum mismatch");
        const std::uint64_t controls_end =
            kPartitionedHeaderSize + descriptor_.capacity;
        validate_zero_range(controls_end, geometry_.lookup_offset);
        const std::uint64_t lookup_end =
            geometry_.lookup_offset
            + descriptor_.capacity * kPartitionSnapshotLookupSlotSize;
        validate_zero_range(lookup_end, geometry_.sequence_offset);

        std::uint64_t occupied = 0;
        for (std::uint64_t slot = 0; slot != descriptor_.capacity; ++slot) {
            const std::uint8_t state = control(slot);
            require<CorruptionError>(
                state <= 1,
                "partition snapshot contains an invalid control byte");
            if (state == 0) {
                require<CorruptionError>(
                    slot_id(slot) == 0 && slot_value(slot) == 0
                        && slot_sequence(slot) == 0,
                    "empty partition snapshot slot contains nonzero data");
                continue;
            }
            ++occupied;
            const std::uint64_t id = slot_id(slot);
            const std::uint64_t sequence = slot_sequence(slot);
            require<CorruptionError>(
                sequence <= kMaximumSequence,
                "partition snapshot slot uses reserved sequence");
            require<CorruptionError>(
                partition_for_id(id, hash_seed_, partition_bits_)
                    == descriptor_.partition_id,
                "partition snapshot ID routes to a different partition");
            static_cast<void>(RowRef::decode(slot_value(slot)));
            validate_probe_reachability(slot, id);
        }
        require<CorruptionError>(
            occupied == descriptor_.entry_count,
            "partition snapshot occupancy does not match catalog");
    }

    [[nodiscard]] const PartitionSnapshotDescriptor& descriptor() const noexcept {
        return descriptor_;
    }

    [[nodiscard]] std::optional<PartitionSnapshotRecord> slot_record(
        const std::uint64_t slot) const {
        if (slot >= descriptor_.capacity) {
            throw InvalidArgumentError(
                "partition snapshot slot is out of range");
        }
        if (control(slot) == 0) {
            return std::nullopt;
        }
        return decoded_record(slot);
    }

    [[nodiscard]] std::optional<PartitionSnapshotRecord> lookup(
        const std::uint64_t id) const {
        if (partition_for_id(id, hash_seed_, partition_bits_)
            != descriptor_.partition_id) {
            return std::nullopt;
        }
        const std::uint64_t mask = descriptor_.capacity - 1;
        std::uint64_t slot = hash_id(id, hash_seed_) & mask;
        for (std::uint64_t probes = 0;
             probes != descriptor_.capacity; ++probes) {
            if (control(slot) == 0) {
                return std::nullopt;
            }
            if (slot_id(slot) == id) {
                return decoded_record(slot);
            }
            slot = (slot + 1) & mask;
        }
        throw CorruptionError("partition snapshot lookup has no empty slot");
    }

private:
    void validate_zero_range(
        const std::uint64_t begin,
        const std::uint64_t end) const {
        const auto bytes = mapping_.bytes();
        for (std::uint64_t offset = begin; offset != end; ++offset) {
            require<CorruptionError>(
                bytes[static_cast<std::size_t>(offset)] == std::byte{0},
                "partition snapshot alignment padding is nonzero");
        }
    }

    [[nodiscard]] std::uint8_t control(
        const std::uint64_t slot) const noexcept {
        return std::to_integer<std::uint8_t>(mapping_.bytes()[
            static_cast<std::size_t>(kPartitionedHeaderSize + slot)]);
    }

    [[nodiscard]] std::uint64_t slot_id(
        const std::uint64_t slot) const noexcept {
        return get_u64(mapping_.bytes(), static_cast<std::size_t>(
            geometry_.lookup_offset
            + slot * kPartitionSnapshotLookupSlotSize));
    }

    [[nodiscard]] std::uint64_t slot_value(
        const std::uint64_t slot) const noexcept {
        return get_u64(mapping_.bytes(), static_cast<std::size_t>(
            geometry_.lookup_offset
            + slot * kPartitionSnapshotLookupSlotSize + 8));
    }

    [[nodiscard]] std::uint64_t slot_sequence(
        const std::uint64_t slot) const noexcept {
        return get_u64(mapping_.bytes(), static_cast<std::size_t>(
            geometry_.sequence_offset
            + slot * kPartitionSnapshotSequenceSlotSize));
    }

    [[nodiscard]] PartitionSnapshotRecord decoded_record(
        const std::uint64_t slot) const {
        return PartitionSnapshotRecord{
            slot_id(slot), slot_sequence(slot),
            RowRef::decode(slot_value(slot))};
    }

    void validate_probe_reachability(
        const std::uint64_t physical_slot,
        const std::uint64_t id) const {
        const std::uint64_t mask = descriptor_.capacity - 1;
        std::uint64_t probe = hash_id(id, hash_seed_) & mask;
        for (std::uint64_t inspected = 0;
             inspected != descriptor_.capacity; ++inspected) {
            if (control(probe) == 0) {
                break;
            }
            if (probe == physical_slot) {
                return;
            }
            require<CorruptionError>(
                slot_id(probe) != id,
                "partition snapshot contains a duplicate ID");
            probe = (probe + 1) & mask;
        }
        throw CorruptionError(
            "partition snapshot contains a probe-unreachable slot");
    }

    PartitionSnapshotDescriptor descriptor_;
    std::uint32_t partition_bits_{0};
    std::uint64_t hash_seed_{0};
    SnapshotGeometry geometry_;
    std::filesystem::path path_;
    ReadOnlyMapping mapping_;
};

PartitionSnapshotReader::PartitionSnapshotReader(
    std::filesystem::path store_path,
    const Uuid128 collection_id,
    const Uuid128 schema_id,
    const std::uint32_t partition_bits,
    const std::uint64_t hash_seed,
    PartitionSnapshotDescriptor expected_descriptor)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), collection_id, schema_id, partition_bits,
          hash_seed, std::move(expected_descriptor))) {}

PartitionSnapshotReader::~PartitionSnapshotReader() = default;
PartitionSnapshotReader::PartitionSnapshotReader(
    PartitionSnapshotReader&&) noexcept = default;
PartitionSnapshotReader& PartitionSnapshotReader::operator=(
    PartitionSnapshotReader&&) noexcept = default;

const PartitionSnapshotDescriptor&
PartitionSnapshotReader::descriptor() const noexcept {
    return impl_->descriptor();
}

std::uint64_t PartitionSnapshotReader::record_count() const noexcept {
    return impl_->descriptor().entry_count;
}

std::uint64_t PartitionSnapshotReader::slot_count() const noexcept {
    return impl_->descriptor().capacity;
}

std::optional<PartitionSnapshotRecord> PartitionSnapshotReader::slot_record(
    const std::uint64_t slot) const {
    return impl_->slot_record(slot);
}

std::optional<PartitionSnapshotRecord> PartitionSnapshotReader::lookup(
    const std::uint64_t id) const {
    return impl_->lookup(id);
}

}  // namespace throwdb::detail
