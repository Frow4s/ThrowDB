#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>

namespace throwdb::detail {
namespace {

constexpr std::size_t kEntryBatch = 1U << 16;

[[nodiscard]] std::uint64_t choose_capacity(const std::uint64_t count) {
    std::uint64_t capacity = 8;
    // Keep the linear-probing table at or below a 70% load factor.
    while (count > (capacity / 10) * 7 + ((capacity % 10) * 7) / 10) {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / 2) {
            throw InvalidArgumentError("index is too large");
        }
        capacity *= 2;
    }
    if (capacity > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("index does not fit in address space");
    }
    return capacity;
}

void insert(IndexData& index, const std::uint64_t id,
            const std::uint64_t ordinal) {
    const std::uint64_t mask = index.capacity - 1;
    std::uint64_t position = hash_id(id, index.hash_seed) & mask;
    for (std::uint64_t probe = 0; probe != index.capacity; ++probe) {
        if (index.controls[static_cast<std::size_t>(position)] == 0) {
            index.controls[static_cast<std::size_t>(position)] = 1;
            index.slots[static_cast<std::size_t>(position)] = IndexSlot{id, ordinal};
            return;
        }
        if (index.slots[static_cast<std::size_t>(position)].id == id) {
            throw DuplicateIdError(id);
        }
        position = (position + 1) & mask;
    }
    throw StorageError("immutable hash index is unexpectedly full");
}

[[nodiscard]] std::array<std::byte, kHeaderSize> encode_index_header(
    const IndexData& index, const std::uint64_t controls_offset,
    const std::uint64_t slots_offset, const std::uint64_t total_size) {
    std::array<std::byte, kHeaderSize> output{};
    std::memcpy(output.data(), kIndexMagic.data(), kIndexMagic.size());
    put_u32(output, 8, static_cast<std::uint32_t>(kWriteFormatVersion));
    put_u32(output, 12, kHeaderSize);
    put_u64(output, 16, index.count);
    put_u64(output, 24, index.capacity);
    put_u64(output, 32, index.hash_seed);
    put_u64(output, 40, controls_offset);
    put_u64(output, 48, slots_offset);
    put_u64(output, 56, total_size);
    put_u64(output, 64, kIndexSlotSize);
    return output;
}

void validate_index_header(const std::span<const std::byte> bytes,
                           const Manifest& manifest) {
    const CommonHeader common = decode_common_header(bytes, kIndexMagic, "index");
    if (common.format_version != FormatVersion::V1
        || manifest.format_version != FormatVersion::V1) {
        throw CorruptionError(
            "index has unsupported format version "
            + std::to_string(static_cast<std::uint32_t>(
                common.format_version)));
    }
    if (get_u64(bytes, 16) != manifest.record_count
        || get_u64(bytes, 24) != manifest.index_capacity
        || get_u64(bytes, 32) != manifest.hash_seed
        || get_u64(bytes, 40) != manifest.index_control_offset
        || get_u64(bytes, 48) != manifest.index_slots_offset
        || get_u64(bytes, 56) != manifest.index_file_size
        || get_u64(bytes, 64) != kIndexSlotSize) {
        throw CorruptionError("index header does not match manifest");
    }
}

}  // namespace

std::uint64_t hash_id(std::uint64_t id, const std::uint64_t seed) noexcept {
    id += seed + 0x9e3779b97f4a7c15ULL;
    id = (id ^ (id >> 30)) * 0xbf58476d1ce4e5b9ULL;
    id = (id ^ (id >> 27)) * 0x94d049bb133111ebULL;
    return id ^ (id >> 31);
}

IndexData build_index(const int staging_fd,
                      const std::filesystem::path& staging_path,
                      const std::uint64_t count,
                      const std::uint64_t seed) {
    const std::uint64_t log_bytes = checked_multiply(count, kStagingEntrySize,
                                                     "staging log");
    if (file_size(staging_fd, staging_path) != kHeaderSize + log_bytes) {
        throw CorruptionError("staging log size does not match record count");
    }
    std::array<std::byte, kHeaderSize> staging_header{};
    pread_exact(staging_fd, staging_header.data(), staging_header.size(), 0,
                staging_path);
    if (decode_staging_header(staging_header) != count) {
        throw CorruptionError("staging log count does not match writer count");
    }

    IndexData index;
    index.count = count;
    index.capacity = choose_capacity(count);
    index.hash_seed = seed;
    index.controls.resize(static_cast<std::size_t>(index.capacity), 0);
    index.slots.resize(static_cast<std::size_t>(index.capacity));

    std::vector<std::byte> buffer(kEntryBatch * kStagingEntrySize);
    std::uint64_t processed = 0;
    while (processed != count) {
        const std::uint64_t batch_count =
            std::min<std::uint64_t>(kEntryBatch, count - processed);
        const std::size_t batch_bytes = static_cast<std::size_t>(
            batch_count * kStagingEntrySize);
        pread_exact(staging_fd, buffer.data(), batch_bytes,
                    kHeaderSize + processed * kStagingEntrySize, staging_path);
        const std::span<const std::byte> bytes(buffer.data(), batch_bytes);
        for (std::uint64_t index_in_batch = 0; index_in_batch != batch_count;
             ++index_in_batch) {
            const std::size_t offset = static_cast<std::size_t>(
                index_in_batch * kStagingEntrySize);
            const std::uint64_t id = get_u64(bytes, offset);
            const std::uint64_t ordinal = get_u64(bytes, offset + 8);
            if (ordinal != processed + index_in_batch) {
                throw CorruptionError("staging log contains a non-sequential ordinal");
            }
            insert(index, id, ordinal);
        }
        processed += batch_count;
    }
    return index;
}

void write_index_atomic(const std::filesystem::path& store_path,
                        const IndexData& index, Manifest& manifest) {
    const auto index_directory = store_path / "index";
    const auto temporary = index_directory / "index.bin.tmp";
    const auto final_path = index_directory / "index.bin";

    const std::uint64_t controls_offset = kHeaderSize;
    const std::uint64_t slots_offset = align_up(
        controls_offset + index.capacity, kHeaderSize);
    const std::uint64_t slot_bytes = checked_multiply(
        index.capacity, kIndexSlotSize, "index slot file");
    if (slots_offset > std::numeric_limits<std::uint64_t>::max() - slot_bytes) {
        throw InvalidArgumentError("index file is too large");
    }
    const std::uint64_t total_size = slots_offset + slot_bytes;

    auto fd = create_exclusive(temporary);
    const auto header = encode_index_header(
        index, controls_offset, slots_offset, total_size);
    write_all(fd.get(), header.data(), header.size(), temporary);
    write_all(fd.get(), index.controls.data(), index.controls.size(), temporary);

    std::array<std::byte, kHeaderSize> zeroes{};
    std::uint64_t current_offset = controls_offset + index.capacity;
    while (current_offset != slots_offset) {
        const std::size_t amount = static_cast<std::size_t>(
            std::min<std::uint64_t>(zeroes.size(), slots_offset - current_offset));
        write_all(fd.get(), zeroes.data(), amount, temporary);
        current_offset += amount;
    }

    std::vector<std::byte> encoded(kEntryBatch * kIndexSlotSize);
    std::uint64_t position = 0;
    while (position != index.capacity) {
        const std::uint64_t batch_count =
            std::min<std::uint64_t>(kEntryBatch, index.capacity - position);
        const std::size_t batch_bytes = static_cast<std::size_t>(
            batch_count * kIndexSlotSize);
        std::span<std::byte> output(encoded.data(), batch_bytes);
        for (std::uint64_t item = 0; item != batch_count; ++item) {
            const IndexSlot& slot = index.slots[static_cast<std::size_t>(position + item)];
            const std::size_t offset = static_cast<std::size_t>(item * kIndexSlotSize);
            put_u64(output, offset, slot.id);
            put_u64(output, offset + 8, slot.ordinal);
        }
        write_all(fd.get(), encoded.data(), batch_bytes, temporary);
        position += batch_count;
    }
    sync_file(fd.get(), temporary);
    fd.reset();

    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if (error) {
        throw StorageError("cannot publish index: " + error.message());
    }
    sync_directory(index_directory);

    manifest.index_capacity = index.capacity;
    manifest.hash_seed = index.hash_seed;
    manifest.index_control_offset = controls_offset;
    manifest.index_slots_offset = slots_offset;
    manifest.index_file_size = total_size;
}

IndexData read_index(const std::filesystem::path& store_path,
                     const Manifest& manifest) {
    if ((manifest.index_capacity & (manifest.index_capacity - 1)) != 0
        || manifest.record_count > manifest.index_capacity) {
        throw CorruptionError("manifest contains an invalid index capacity");
    }
    const std::uint64_t expected_slots_bytes = checked_multiply(
        manifest.index_capacity, kIndexSlotSize, "index slots");
    if (manifest.index_slots_offset
        > std::numeric_limits<std::uint64_t>::max() - expected_slots_bytes) {
        throw CorruptionError("manifest index file size overflows uint64");
    }
    if (manifest.index_control_offset != kHeaderSize
        || manifest.index_slots_offset < kHeaderSize + manifest.index_capacity
        || manifest.index_file_size != manifest.index_slots_offset + expected_slots_bytes) {
        throw CorruptionError("manifest contains invalid index offsets");
    }
    if (manifest.index_capacity > std::numeric_limits<std::size_t>::max()) {
        throw CorruptionError("index does not fit in address space");
    }

    const auto path = store_path / "index" / "index.bin";
    auto fd = open_read_only(path);
    if (file_size(fd.get(), path) != manifest.index_file_size) {
        throw CorruptionError("index file size does not match manifest");
    }
    std::array<std::byte, kHeaderSize> header{};
    pread_exact(fd.get(), header.data(), header.size(), 0, path);
    validate_index_header(header, manifest);

    IndexData index;
    index.count = manifest.record_count;
    index.capacity = manifest.index_capacity;
    index.hash_seed = manifest.hash_seed;
    index.controls.resize(static_cast<std::size_t>(index.capacity));
    index.slots.resize(static_cast<std::size_t>(index.capacity));
    pread_exact(fd.get(), index.controls.data(), index.controls.size(),
                manifest.index_control_offset, path);

    std::vector<std::byte> encoded(kEntryBatch * kIndexSlotSize);
    std::uint64_t position = 0;
    std::uint64_t occupied = 0;
    while (position != index.capacity) {
        const std::uint64_t batch_count =
            std::min<std::uint64_t>(kEntryBatch, index.capacity - position);
        const std::size_t batch_bytes = static_cast<std::size_t>(
            batch_count * kIndexSlotSize);
        pread_exact(fd.get(), encoded.data(), batch_bytes,
                    manifest.index_slots_offset + position * kIndexSlotSize, path);
        const std::span<const std::byte> bytes(encoded.data(), batch_bytes);
        for (std::uint64_t item = 0; item != batch_count; ++item) {
            const auto slot_index = static_cast<std::size_t>(position + item);
            if (index.controls[slot_index] > 1) {
                throw CorruptionError("index contains an invalid control byte");
            }
            occupied += index.controls[slot_index];
            const std::size_t offset = static_cast<std::size_t>(item * kIndexSlotSize);
            index.slots[slot_index] = IndexSlot{
                get_u64(bytes, offset), get_u64(bytes, offset + 8)};
            if (index.controls[slot_index] != 0
                && index.slots[slot_index].ordinal >= manifest.record_count) {
                throw CorruptionError("index contains an out-of-range ordinal");
            }
        }
        position += batch_count;
    }
    if (occupied != manifest.record_count) {
        throw CorruptionError("index occupancy does not match manifest count");
    }
    return index;
}

bool index_lookup(const IndexData& index, const std::uint64_t id,
                  std::uint64_t& ordinal) noexcept {
    const std::uint64_t mask = index.capacity - 1;
    std::uint64_t position = hash_id(id, index.hash_seed) & mask;
    for (std::uint64_t probe = 0; probe != index.capacity; ++probe) {
        const auto slot_index = static_cast<std::size_t>(position);
        if (index.controls[slot_index] == 0) {
            return false;
        }
        if (index.slots[slot_index].id == id) {
            ordinal = index.slots[slot_index].ordinal;
            return true;
        }
        position = (position + 1) & mask;
    }
    return false;
}

}  // namespace throwdb::detail
