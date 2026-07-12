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
    const IndexData& index, const Manifest& manifest) {
    std::array<std::byte, kHeaderSize> output{};
    std::memcpy(output.data(), kIndexMagic.data(), kIndexMagic.size());
    put_u32(output, 12, kHeaderSize);
    switch (manifest.format_version) {
        case FormatVersion::V1:
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V1));
            put_u64(output, 16, index.count);
            put_u64(output, 24, index.capacity);
            put_u64(output, 32, index.hash_seed);
            put_u64(output, 40, manifest.index_control_offset);
            put_u64(output, 48, manifest.index_slots_offset);
            put_u64(output, 56, manifest.index_file_size);
            put_u64(output, 64, kIndexSlotSize);
            return output;

        case FormatVersion::V2:
            if (manifest.snapshot_id.empty()
                || manifest.schema.id_type != IdType::UInt64
                || manifest.schema.index_kind != IndexKind::UInt64Flat
                || manifest.schema.hash_algorithm != HashAlgorithm::SplitMix64V1
                || manifest.hash_seed_high != 0
                || manifest.index_slot_size != kIndexSlotSize
                || manifest.expected_string_arena_size != 0
                || manifest.header_checksum_algorithm != ChecksumAlgorithm::None
                || index.count != manifest.record_count
                || index.capacity != manifest.index_capacity
                || index.hash_seed != manifest.hash_seed) {
                throw InvalidArgumentError(
                    "format-v2 index does not match the numeric manifest");
            }
            put_u32(output, 8, static_cast<std::uint32_t>(FormatVersion::V2));
            put_u64(output, 16, manifest.snapshot_id.low);
            put_u64(output, 24, manifest.snapshot_id.high);
            put_u32(output, 32, static_cast<std::uint32_t>(manifest.schema.id_type));
            put_u32(
                output, 36,
                static_cast<std::uint32_t>(manifest.schema.index_kind));
            put_u32(
                output, 40,
                static_cast<std::uint32_t>(manifest.schema.hash_algorithm));
            put_u32(output, 44, 1);  // one-byte empty/occupied controls
            put_u64(output, 48, index.count);
            put_u64(output, 56, index.capacity);
            put_u64(output, 64, index.hash_seed);
            put_u64(output, 72, manifest.hash_seed_high);
            put_u64(output, 80, manifest.index_control_offset);
            put_u64(output, 88, manifest.index_slots_offset);
            put_u64(output, 96, manifest.index_slot_size);
            put_u64(output, 104, manifest.index_file_size);
            put_u64(output, 112, manifest.expected_string_arena_size);
            put_u64(output, 120, 0);  // checksums are disabled for this slice
            return output;

        case FormatVersion::Invalid:
            break;
    }
    throw InvalidArgumentError("index has an invalid format version");
}

void require_reserved_zero(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::string_view description) {
    for (std::size_t offset = begin; offset != kHeaderSize; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw CorruptionError(
                std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

void validate_index_header(const std::span<const std::byte> bytes,
                           const Manifest& manifest) {
    const CommonHeader common = decode_common_header(bytes, kIndexMagic, "index");
    if (common.format_version != manifest.format_version) {
        throw CorruptionError("index format version does not match manifest");
    }
    switch (common.format_version) {
        case FormatVersion::V1:
            if (get_u64(bytes, 16) != manifest.record_count
                || get_u64(bytes, 24) != manifest.index_capacity
                || get_u64(bytes, 32) != manifest.hash_seed
                || get_u64(bytes, 40) != manifest.index_control_offset
                || get_u64(bytes, 48) != manifest.index_slots_offset
                || get_u64(bytes, 56) != manifest.index_file_size
                || get_u64(bytes, 64) != kIndexSlotSize) {
                throw CorruptionError("index header does not match manifest");
            }
            return;

        case FormatVersion::V2:
            if (get_u64(bytes, 16) != manifest.snapshot_id.low
                || get_u64(bytes, 24) != manifest.snapshot_id.high
                || get_u32(bytes, 32)
                       != static_cast<std::uint32_t>(manifest.schema.id_type)
                || get_u32(bytes, 36)
                       != static_cast<std::uint32_t>(manifest.schema.index_kind)
                || get_u32(bytes, 40)
                       != static_cast<std::uint32_t>(
                           manifest.schema.hash_algorithm)
                || get_u32(bytes, 44) != 1
                || get_u64(bytes, 48) != manifest.record_count
                || get_u64(bytes, 56) != manifest.index_capacity
                || get_u64(bytes, 64) != manifest.hash_seed
                || get_u64(bytes, 72) != manifest.hash_seed_high
                || get_u64(bytes, 80) != manifest.index_control_offset
                || get_u64(bytes, 88) != manifest.index_slots_offset
                || get_u64(bytes, 96) != manifest.index_slot_size
                || get_u64(bytes, 104) != manifest.index_file_size
                || get_u64(bytes, 112) != manifest.expected_string_arena_size
                || get_u64(bytes, 120) != 0) {
                throw CorruptionError(
                    "format-v2 index header does not match manifest");
            }
            require_reserved_zero(bytes, 128, "format-v2 index");
            return;

        case FormatVersion::Invalid:
            break;
    }
    throw CorruptionError("index has an invalid format version");
}

[[nodiscard]] std::uint64_t corruption_checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw CorruptionError(std::string(description) + " overflows uint64");
    }
    return left * right;
}

void validate_v2_probe_reachability(const IndexData& index) {
    const auto empty = std::find(
        index.controls.begin(), index.controls.end(), std::uint8_t{0});
    if (empty == index.controls.end()) {
        throw CorruptionError(
            "format-v2 index does not contain an empty control slot");
    }

    // Rotate the table so the scan starts immediately after a known empty
    // slot. Every occupied cluster is then a non-wrapping interval. A slot is
    // reachable by linear probing exactly when its home bucket lies between
    // the start of its cluster and the slot itself.
    const std::uint64_t mask = index.capacity - 1;
    const std::uint64_t empty_position = static_cast<std::uint64_t>(
        std::distance(index.controls.begin(), empty));
    const std::uint64_t origin = (empty_position + 1) & mask;
    std::uint64_t cluster_start = 0;

    for (std::uint64_t logical_position = 0;
         logical_position != index.capacity; ++logical_position) {
        const std::uint64_t position = (origin + logical_position) & mask;
        const auto slot_index = static_cast<std::size_t>(position);
        if (index.controls[slot_index] == 0) {
            cluster_start = logical_position + 1;
            continue;
        }

        const std::uint64_t home =
            hash_id(index.slots[slot_index].id, index.hash_seed) & mask;
        const std::uint64_t logical_home = (home - origin) & mask;
        if (logical_home < cluster_start
            || logical_home > logical_position) {
            throw CorruptionError(
                "format-v2 index contains a probe-unreachable slot");
        }
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

    manifest.index_capacity = index.capacity;
    manifest.hash_seed = index.hash_seed;
    manifest.index_control_offset = controls_offset;
    manifest.index_slots_offset = slots_offset;
    manifest.index_slot_size = kIndexSlotSize;
    manifest.index_file_size = total_size;

    auto fd = create_exclusive(temporary);
    const auto header = encode_index_header(index, manifest);
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

}

IndexData read_index(const std::filesystem::path& store_path,
                     const Manifest& manifest) {
    if (manifest.index_capacity == 0
        || (manifest.index_capacity & (manifest.index_capacity - 1)) != 0
        || manifest.record_count > manifest.index_capacity) {
        throw CorruptionError("manifest contains an invalid index capacity");
    }
    if (manifest.index_slot_size != kIndexSlotSize) {
        throw CorruptionError("manifest contains an invalid index slot size");
    }
    const std::uint64_t expected_slots_bytes = corruption_checked_multiply(
        manifest.index_capacity, manifest.index_slot_size, "index slots");
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

    std::vector<std::uint64_t> seen_ordinals;
    if (manifest.format_version == FormatVersion::V2) {
        const std::uint64_t word_count = manifest.record_count / 64
            + (manifest.record_count % 64 == 0 ? 0 : 1);
        seen_ordinals.resize(static_cast<std::size_t>(word_count), 0);
    }

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
            if (index.controls[slot_index] != 0) {
                const std::uint64_t ordinal = index.slots[slot_index].ordinal;
                if (ordinal >= manifest.record_count) {
                    throw CorruptionError(
                        "index contains an out-of-range ordinal");
                }
                if (manifest.format_version == FormatVersion::V2) {
                    const std::size_t word =
                        static_cast<std::size_t>(ordinal / 64);
                    const std::uint64_t bit =
                        std::uint64_t{1} << (ordinal % 64);
                    if ((seen_ordinals[word] & bit) != 0) {
                        throw CorruptionError(
                            "format-v2 index ordinals are not a bijection");
                    }
                    seen_ordinals[word] |= bit;
                }
            }
        }
        position += batch_count;
    }
    if (occupied != manifest.record_count) {
        throw CorruptionError("index occupancy does not match manifest count");
    }
    if (manifest.format_version == FormatVersion::V2) {
        validate_v2_probe_reachability(index);
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
