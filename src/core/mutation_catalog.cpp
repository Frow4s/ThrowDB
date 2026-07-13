#include "mutation_catalog.hpp"

#include "crc32c.hpp"
#include "mutation_wal.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace throwdb::detail {
namespace {

constexpr std::array<char, 8> kCatalogMagic{
    'T', 'D', 'B', 'M', 'C', 'A', 'T', '1'};
constexpr std::array<char, 8> kCurrentMagic{
    'T', 'D', 'B', 'M', 'C', 'U', 'R', '1'};
constexpr std::uint32_t kCatalogFormatVersionV1 = 1;
constexpr std::uint32_t kCatalogFormatVersionV2 = 2;
constexpr std::uint32_t kCurrentFormatVersion = 1;
constexpr std::uint32_t kChecksumAlgorithmCrc32c = 1;
constexpr std::uint64_t kMaximumCommitSequence =
    std::numeric_limits<std::uint64_t>::max() - 1;
constexpr std::size_t kChecksumOffset = 20;
constexpr std::size_t kCatalogV1KnownHeaderBytes = 152;
constexpr std::size_t kCatalogV2KnownHeaderBytes = 176;
constexpr std::size_t kCurrentKnownBytes = 96;
constexpr std::size_t kEntryKnownBytes = 80;
constexpr std::size_t kDeltaEntryKnownBytes = 120;

template <typename Error>
void require(const bool condition, const std::string_view message) {
    static_assert(std::is_base_of_v<std::exception, Error>);
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
        std::string(description) + " has invalid alignment");
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0
               ? value
               : checked_sum<Error>(
                     value, alignment - remainder, description);
}

void put_magic(
    const std::span<std::byte> bytes,
    const std::array<char, 8>& magic) {
    std::memcpy(bytes.data(), magic.data(), magic.size());
}

template <typename Error>
void require_magic(
    const std::span<const std::byte> bytes,
    const std::array<char, 8>& magic,
    const std::string_view description) {
    require<Error>(
        bytes.size() >= magic.size()
            && std::memcmp(bytes.data(), magic.data(), magic.size()) == 0,
        std::string(description) + " has invalid magic");
}

void put_uuid(
    const std::span<std::byte> bytes,
    const std::size_t offset,
    const Uuid128 value) {
    put_u64(bytes, offset, value.low);
    put_u64(bytes, offset + 8, value.high);
}

[[nodiscard]] Uuid128 get_uuid(
    const std::span<const std::byte> bytes,
    const std::size_t offset) {
    return Uuid128{get_u64(bytes, offset), get_u64(bytes, offset + 8)};
}

template <typename Error>
void require_zero_range(
    const std::span<const std::byte> bytes,
    const std::size_t begin,
    const std::size_t end,
    const std::string_view description) {
    require<Error>(
        begin <= end && end <= bytes.size(),
        std::string(description) + " has an invalid reserved range");
    for (std::size_t offset = begin; offset != end; ++offset) {
        if (bytes[offset] != std::byte{0}) {
            throw Error(
                std::string(description) + " contains nonzero reserved bytes");
        }
    }
}

[[nodiscard]] std::uint32_t crc32c_with_zeroed_checksum(
    const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes.first(kChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(kChecksumOffset + sizeof(std::uint32_t)));
    return checksum.value();
}

[[nodiscard]] std::string uuid_filename_token(const Uuid128 id) {
    if (id.empty()) {
        throw InvalidArgumentError("file UUID is empty");
    }
    std::array<char, 33> token{};
    const int count = std::snprintf(
        token.data(), token.size(), "%016llx%016llx",
        static_cast<unsigned long long>(id.high),
        static_cast<unsigned long long>(id.low));
    if (count != 32) {
        throw StorageError("cannot format UUID-derived file name");
    }
    return std::string(token.data(), 32);
}

struct UuidHash {
    [[nodiscard]] std::size_t operator()(const Uuid128 value) const noexcept {
        std::uint64_t mixed = value.low
                              ^ (value.high + 0x9e3779b97f4a7c15ULL
                                 + (value.low << 6) + (value.low >> 2));
        mixed ^= mixed >> 30;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27;
        mixed *= 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31;
        return static_cast<std::size_t>(mixed);
    }
};

template <typename Error>
void validate_manifest_for_catalog(const NativeManifest& manifest) {
    require<Error>(
        mutation_catalog_managed(manifest),
        "manifest does not require a mutation catalog");
    require<Error>(
        !mutation_delta_managed(manifest)
            || mutation_catalog_managed(manifest),
        "manifest requires mutation deltas without a mutation catalog");
    require<Error>(
        !manifest.collection_id.empty(), "manifest collection UUID is empty");
    require<Error>(
        !manifest.schema_id.empty(), "manifest schema UUID is empty");
    require<Error>(
        !manifest.active_wal_id.empty(), "manifest WAL anchor UUID is empty");
    require<Error>(
        manifest.active_wal_segment_id >= SegmentId::kMinimum
            && manifest.active_wal_segment_id <= SegmentId::kMaximum,
        "manifest WAL anchor segment ID is invalid");
    require<Error>(
        manifest.base_segment_count < manifest.active_wal_segment_id,
        "manifest WAL anchor overlaps base segments");
}

template <typename Error>
void validate_delta_descriptor(
    const NativeManifest& manifest,
    const MutationDeltaDescriptor& descriptor) {
    require<Error>(
        !descriptor.delta_id.empty(), "catalog delta UUID is empty");
    require<Error>(
        !descriptor.source_terminal_wal_id.empty(),
        "catalog delta terminal WAL UUID is empty");
    require<Error>(
        descriptor.delta_id != descriptor.source_terminal_wal_id
            && descriptor.delta_id
                   != descriptor.source_predecessor_wal_id,
        "catalog delta UUID aliases its source WAL lineage");
    require<Error>(
        descriptor.source_predecessor_wal_id
            != descriptor.source_terminal_wal_id,
        "catalog delta source WAL lineage is cyclic");
    require<Error>(
        descriptor.first_sequence != 0
            && descriptor.last_sequence >= descriptor.first_sequence
            && descriptor.last_sequence <= kMaximumCommitSequence,
        "catalog delta has an invalid sequence interval");
    require<Error>(
        descriptor.segment_id > manifest.base_segment_count
            && descriptor.segment_id >= SegmentId::kMinimum
            && descriptor.segment_id <= SegmentId::kMaximum,
        "catalog delta segment ID is invalid or overlaps base segments");
    require<Error>(
        descriptor.source_wal_count != 0
            && descriptor.source_wal_count
                   <= descriptor.last_sequence - descriptor.first_sequence + 1,
        "catalog delta source WAL count is invalid");
    require<Error>(
        descriptor.entry_count != 0
            && descriptor.entry_count
                   <= std::numeric_limits<std::uint64_t>::max()
                          / kMutationDeltaVersionRecordSize,
        "catalog delta entry count is invalid");
    require<Error>(
        descriptor.put_count <= descriptor.entry_count
            && descriptor.put_count <= RowRef::kMaximumRowsPerSegment,
        "catalog delta PUT count is invalid");

    const std::uint64_t expected_data_size = checked_sum<Error>(
        kMutationDeltaHeaderSize,
        checked_product<Error>(
            descriptor.put_count, manifest.record_size,
            "catalog delta data payload"),
        "catalog delta data file size");
    require<Error>(
        descriptor.data_file_size == expected_data_size,
        "catalog delta data file geometry is invalid");

    const std::uint64_t records_end = checked_sum<Error>(
        kMutationDeltaHeaderSize,
        checked_product<Error>(
            descriptor.entry_count, kMutationDeltaVersionRecordSize,
            "catalog delta index records"),
        "catalog delta index records end");
    const std::uint64_t reverse_offset = checked_align<Error>(
        records_end, kMutationDeltaIndexAlignment,
        "catalog delta reverse index");
    const std::uint64_t expected_index_size = checked_sum<Error>(
        reverse_offset,
        checked_product<Error>(
            descriptor.put_count, kMutationDeltaReverseRecordSize,
            "catalog delta reverse records"),
        "catalog delta index file size");
    require<Error>(
        descriptor.index_file_size == expected_index_size,
        "catalog delta index file geometry is invalid");
}

template <typename Error>
void validate_catalog(
    const NativeManifest& manifest,
    const MutationCatalog& catalog) {
    validate_manifest_for_catalog<Error>(manifest);
    require<Error>(
        catalog.collection_id == manifest.collection_id,
        "mutation catalog collection UUID does not match manifest");
    require<Error>(
        catalog.schema_id == manifest.schema_id,
        "mutation catalog schema UUID does not match manifest");
    require<Error>(catalog.generation != 0, "catalog generation is zero");
    require<Error>(!catalog.catalog_id.empty(), "catalog UUID is empty");
    require<Error>(
        catalog.previous_catalog_id != catalog.catalog_id,
        "catalog names itself as its predecessor");
    require<Error>(
        (catalog.generation == 1 && catalog.previous_catalog_id.empty())
            || (catalog.generation > 1
                && !catalog.previous_catalog_id.empty()),
        "catalog predecessor does not match its generation");
    require<Error>(
        catalog.next_segment_id >= SegmentId::kMinimum
            && catalog.next_segment_id <= SegmentId::kMaximum,
        "catalog next segment ID is invalid");
    require<Error>(
        !catalog.wal_entries.empty(), "catalog WAL chain is empty");
    require<Error>(
        catalog.wal_entries.size() <= kMutationCatalogMaxEntries
            && catalog.delta_entries.size()
                   <= kMutationCatalogMaxEntries
                          - catalog.wal_entries.size(),
        "catalog resource count exceeds the format maximum");
    require<Error>(
        (catalog.materialized_sequence == 0
         && catalog.terminal_wal_id.empty())
            || (catalog.materialized_sequence != 0
                && !catalog.terminal_wal_id.empty()),
        "catalog materialized sequence and terminal WAL disagree");

    if (!mutation_delta_managed(manifest)) {
        require<Error>(
            catalog.delta_entries.empty(),
            "catalog v1 cannot contain immutable deltas");
    } else {
        require<Error>(
            catalog.delta_entries.empty()
                == (catalog.materialized_sequence == 0),
            "catalog deltas do not match the materialized anchor");
    }

    std::unordered_set<Uuid128, UuidHash> resource_ids;
    resource_ids.reserve(
        catalog.delta_entries.size() + catalog.wal_entries.size());
    std::unordered_set<Uuid128, UuidHash> materialized_wal_ids;
    materialized_wal_ids.reserve(catalog.delta_entries.size());
    std::unordered_set<std::uint32_t> resource_segment_ids;
    resource_segment_ids.reserve(
        catalog.delta_entries.size() + catalog.wal_entries.size());
    std::uint32_t maximum_resource_segment_id =
        static_cast<std::uint32_t>(manifest.base_segment_count);

    for (std::size_t index = 0;
         index != catalog.delta_entries.size(); ++index) {
        const MutationDeltaDescriptor& descriptor =
            catalog.delta_entries[index];
        validate_delta_descriptor<Error>(manifest, descriptor);
        require<Error>(
            resource_ids.insert(descriptor.delta_id).second,
            "catalog contains a duplicate delta UUID");
        require<Error>(
            materialized_wal_ids
                .insert(descriptor.source_terminal_wal_id)
                .second,
            "catalog contains a duplicate materialized terminal WAL UUID");
        require<Error>(
            resource_segment_ids.insert(descriptor.segment_id).second,
            "catalog contains a duplicate mutation segment ID");
        maximum_resource_segment_id =
            std::max(maximum_resource_segment_id, descriptor.segment_id);

        if (index == 0) {
            require<Error>(
                descriptor.first_sequence == 1,
                "first catalog delta does not begin at sequence 1");
            require<Error>(
                descriptor.source_predecessor_wal_id.empty(),
                "first catalog delta has a source predecessor");
        } else {
            const MutationDeltaDescriptor& predecessor =
                catalog.delta_entries[index - 1];
            require<Error>(
                predecessor.last_sequence < kMaximumCommitSequence
                    && descriptor.first_sequence
                           == predecessor.last_sequence + 1,
                "catalog delta sequence intervals are not contiguous");
            require<Error>(
                descriptor.source_predecessor_wal_id
                    == predecessor.source_terminal_wal_id,
                "catalog delta source WAL lineage is broken");
        }
    }

    for (const Uuid128 terminal_id : materialized_wal_ids) {
        require<Error>(
            !resource_ids.contains(terminal_id),
            "catalog delta UUID aliases a materialized WAL UUID");
    }
    if (!catalog.delta_entries.empty()) {
        const MutationDeltaDescriptor& last = catalog.delta_entries.back();
        require<Error>(
            last.last_sequence == catalog.materialized_sequence,
            "last catalog delta does not reach materialized sequence");
        require<Error>(
            last.source_terminal_wal_id == catalog.terminal_wal_id,
            "last catalog delta does not reach terminal WAL");
    }

    const MutationWalEntry& first = catalog.wal_entries.front();
    require<Error>(
        first.predecessor_wal_id == catalog.terminal_wal_id,
        "first retained WAL does not follow the catalog terminal WAL");
    require<Error>(
        first.previous_sequence == catalog.materialized_sequence,
        "first retained WAL does not follow materialized sequence");
    if (catalog.materialized_sequence == 0) {
        require<Error>(
            first.wal_id == manifest.active_wal_id,
            "initial WAL chain does not use the manifest WAL anchor");
        require<Error>(
            first.segment_id == manifest.active_wal_segment_id,
            "initial WAL chain does not use the manifest segment anchor");
    }

    std::unordered_set<Uuid128, UuidHash> seen_wal_ids =
        materialized_wal_ids;
    seen_wal_ids.reserve(
        materialized_wal_ids.size() + catalog.wal_entries.size());
    if (!catalog.terminal_wal_id.empty()) {
        static_cast<void>(seen_wal_ids.insert(catalog.terminal_wal_id));
    }
    std::uint32_t previous_segment_id = 0;
    for (std::size_t index = 0; index != catalog.wal_entries.size(); ++index) {
        const MutationWalEntry& entry = catalog.wal_entries[index];
        require<Error>(!entry.wal_id.empty(), "catalog WAL UUID is empty");
        require<Error>(
            entry.predecessor_wal_id != entry.wal_id,
            "catalog WAL names itself as predecessor");
        require<Error>(
            seen_wal_ids.insert(entry.wal_id).second,
            "catalog contains a duplicate WAL UUID");
        require<Error>(
            resource_ids.insert(entry.wal_id).second,
            "catalog WAL UUID aliases another mutation resource");
        require<Error>(
            entry.segment_id > manifest.base_segment_count
                && entry.segment_id >= SegmentId::kMinimum
                && entry.segment_id <= SegmentId::kMaximum,
            "catalog WAL segment ID is invalid or overlaps base segments");
        require<Error>(
            entry.segment_id > previous_segment_id,
            "catalog WAL segment IDs are not strictly increasing");
        previous_segment_id = entry.segment_id;
        require<Error>(
            resource_segment_ids.insert(entry.segment_id).second,
            "catalog contains a duplicate mutation segment ID");
        maximum_resource_segment_id =
            std::max(maximum_resource_segment_id, entry.segment_id);
        require<Error>(
            entry.previous_sequence < kMaximumCommitSequence,
            "catalog WAL previous sequence cannot be advanced");
        require<Error>(
            entry.first_sequence == entry.previous_sequence + 1,
            "catalog WAL first sequence does not follow previous sequence");

        const bool is_last = index + 1 == catalog.wal_entries.size();
        if (is_last) {
            require<Error>(
                entry.state == MutationWalState::Active,
                "last catalog WAL is not active");
            require<Error>(
                entry.last_sequence == 0,
                "active catalog WAL has a nonzero last-sequence sentinel");
            require<Error>(
                entry.put_count == 0,
                "active catalog WAL has a nonzero PUT-count sentinel");
            require<Error>(
                entry.sealed_file_size == 0,
                "active catalog WAL has a nonzero sealed-size sentinel");
        } else {
            require<Error>(
                entry.state == MutationWalState::Frozen,
                "non-final catalog WAL is not frozen");
            require<Error>(
                entry.last_sequence >= entry.first_sequence
                    && entry.last_sequence <= kMaximumCommitSequence,
                "frozen catalog WAL has an invalid final sequence");
            require<Error>(
                entry.put_count <= RowRef::kMaximumRowsPerSegment,
                "frozen catalog WAL PUT count exceeds RowRef capacity");
            require<Error>(
                entry.sealed_file_size >= kWalFileHeaderSize,
                "frozen catalog WAL is smaller than its file header");
        }

        if (index != 0) {
            const MutationWalEntry& predecessor =
                catalog.wal_entries[index - 1];
            require<Error>(
                entry.predecessor_wal_id == predecessor.wal_id,
                "catalog WAL predecessor chain is broken");
            require<Error>(
                entry.previous_sequence == predecessor.last_sequence,
                "catalog WAL sequence chain is broken");
        }
    }
    require<Error>(
        catalog.next_segment_id > maximum_resource_segment_id,
        "catalog next segment ID is not after every mutation resource");
}

template <typename Error>
void validate_current(
    const NativeManifest& manifest,
    const MutationCurrent& current) {
    validate_manifest_for_catalog<Error>(manifest);
    require<Error>(
        current.collection_id == manifest.collection_id,
        "CURRENT collection UUID does not match manifest");
    require<Error>(
        current.schema_id == manifest.schema_id,
        "CURRENT schema UUID does not match manifest");
    require<Error>(current.generation != 0, "CURRENT generation is zero");
    require<Error>(!current.catalog_id.empty(), "CURRENT catalog UUID is empty");
    const bool delta_managed = mutation_delta_managed(manifest);
    const std::uint64_t maximum_entry_size =
        delta_managed ? kMutationCatalogDeltaEntrySize
                      : kMutationCatalogEntrySize;
    const std::uint64_t maximum_file_size = checked_sum<Error>(
        kMutationCatalogHeaderSize,
        checked_product<Error>(
            kMutationCatalogMaxEntries, maximum_entry_size,
            "maximum catalog entry bytes"),
        "maximum catalog file size");
    require<Error>(
        current.catalog_file_size
                >= kMutationCatalogHeaderSize + kMutationCatalogEntrySize
            && current.catalog_file_size <= maximum_file_size,
        "CURRENT has invalid catalog file size");
    const std::uint64_t entry_bytes =
        current.catalog_file_size - kMutationCatalogHeaderSize;
    if (!delta_managed) {
        require<Error>(
            entry_bytes % kMutationCatalogEntrySize == 0,
            "CURRENT has invalid catalog v1 geometry");
        return;
    }

    // V2 geometry is 192 * delta_count + 128 * wal_count, with at
    // least one WAL and at most kMutationCatalogMaxEntries resources.
    constexpr std::uint64_t geometry_unit = 64;
    require<Error>(
        entry_bytes % geometry_unit == 0,
        "CURRENT has invalid catalog v2 geometry");
    const std::uint64_t geometry_units = entry_bytes / geometry_unit;
    require<Error>(
        geometry_units >= 2,
        "CURRENT catalog v2 geometry has no WAL entry");
    std::uint64_t minimum_delta_count = 0;
    const std::uint64_t twice_maximum_count =
        kMutationCatalogMaxEntries * 2;
    if (geometry_units > twice_maximum_count) {
        minimum_delta_count = geometry_units - twice_maximum_count;
    }
    const std::uint64_t maximum_delta_count =
        (geometry_units - 2) / 3;
    if ((minimum_delta_count & 1U) != (geometry_units & 1U)) {
        ++minimum_delta_count;
    }
    require<Error>(
        minimum_delta_count <= maximum_delta_count,
        "CURRENT catalog v2 geometry is not representable");
}

[[nodiscard]] std::filesystem::path mutation_catalog_temporary_path(
    const std::filesystem::path& store_path,
    const Uuid128 catalog_id) {
    return std::filesystem::path(
        mutation_catalog_path(store_path, catalog_id).string() + ".tmp");
}

[[nodiscard]] std::filesystem::path mutation_current_temporary_path(
    const std::filesystem::path& store_path,
    const Uuid128 catalog_id) {
    return store_path
           / ("CURRENT." + uuid_filename_token(catalog_id) + ".tmp");
}

void rename_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    const std::string_view description) {
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error) {
        throw StorageError(
            "cannot publish " + std::string(description) + ": "
            + error.message());
    }
}

[[nodiscard]] std::vector<std::byte> read_exact_file(
    const std::filesystem::path& path,
    const std::uint64_t expected_size) {
    if (expected_size > std::numeric_limits<std::size_t>::max()) {
        throw CorruptionError("file does not fit address space: '"
                              + path.string() + "'");
    }
    auto fd = open_read_only(path);
    if (file_size(fd.get(), path) != expected_size) {
        throw CorruptionError("file size does not match metadata: '"
                              + path.string() + "'");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(expected_size));
    pread_exact(fd.get(), bytes.data(), bytes.size(), 0, path);
    return bytes;
}

}  // namespace

bool mutation_catalog_managed(const NativeManifest& manifest) noexcept {
    return (manifest.required_feature_bits
            & kNativeRequiredFeatureCatalogV1) != 0;
}

bool mutation_delta_managed(const NativeManifest& manifest) noexcept {
    return (manifest.required_feature_bits
            & kNativeRequiredFeatureDeltaV1) != 0;
}

std::filesystem::path mutation_catalogs_directory(
    const std::filesystem::path& store_path) {
    return store_path / "catalogs";
}

std::filesystem::path mutation_catalog_path(
    const std::filesystem::path& store_path,
    const Uuid128 catalog_id) {
    return mutation_catalogs_directory(store_path)
           / ("catalog-" + uuid_filename_token(catalog_id) + ".bin");
}

std::filesystem::path mutation_wal_path(
    const std::filesystem::path& store_path,
    const Uuid128 wal_id) {
    return store_path / "wal"
           / ("wal-" + uuid_filename_token(wal_id) + ".wal");
}

std::filesystem::path mutation_current_path(
    const std::filesystem::path& store_path) {
    return store_path / "CURRENT";
}

std::vector<std::byte> encode_mutation_catalog(
    const NativeManifest& manifest,
    const MutationCatalog& catalog) {
    validate_catalog<InvalidArgumentError>(manifest, catalog);
    const bool encode_v2 = mutation_delta_managed(manifest);
    const std::uint64_t delta_count = catalog.delta_entries.size();
    const std::uint64_t wal_count = catalog.wal_entries.size();
    const std::uint64_t wal_entries_offset = checked_sum<InvalidArgumentError>(
        kMutationCatalogHeaderSize,
        checked_product<InvalidArgumentError>(
            delta_count, kMutationCatalogDeltaEntrySize,
            "catalog delta entry bytes"),
        "catalog WAL entries offset");
    const std::uint64_t file_size = checked_sum<InvalidArgumentError>(
        wal_entries_offset,
        checked_product<InvalidArgumentError>(
            wal_count, kMutationCatalogEntrySize,
            "catalog WAL entry bytes"),
        "catalog file size");
    if (file_size > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("catalog does not fit address space");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    put_magic(bytes, kCatalogMagic);
    put_u32(
        bytes, 8,
        encode_v2 ? kCatalogFormatVersionV2 : kCatalogFormatVersionV1);
    put_u32(bytes, 12, static_cast<std::uint32_t>(kMutationCatalogHeaderSize));
    put_u32(bytes, 16, kChecksumAlgorithmCrc32c);
    put_uuid(bytes, 24, catalog.collection_id);
    put_uuid(bytes, 40, catalog.schema_id);
    put_uuid(bytes, 56, catalog.catalog_id);
    put_uuid(bytes, 72, catalog.previous_catalog_id);
    put_u64(bytes, 88, catalog.generation);
    put_u32(bytes, 96, catalog.next_segment_id);
    put_u32(bytes, 100, static_cast<std::uint32_t>(kMutationCatalogEntrySize));
    put_u64(bytes, 104, wal_count);
    put_u64(bytes, 112, wal_entries_offset);
    put_u64(bytes, 120, file_size);
    put_u64(bytes, 128, catalog.materialized_sequence);
    put_uuid(bytes, 136, catalog.terminal_wal_id);
    if (encode_v2) {
        put_u32(
            bytes, 152,
            static_cast<std::uint32_t>(kMutationCatalogDeltaEntrySize));
        put_u32(bytes, 156, 0);
        put_u64(bytes, 160, delta_count);
        put_u64(bytes, 168, kMutationCatalogHeaderSize);
    }

    for (std::size_t index = 0;
         index != catalog.delta_entries.size(); ++index) {
        const MutationDeltaDescriptor& entry = catalog.delta_entries[index];
        const std::size_t offset = static_cast<std::size_t>(
            kMutationCatalogHeaderSize
            + index * kMutationCatalogDeltaEntrySize);
        const std::span<std::byte> encoded =
            std::span<std::byte>(bytes).subspan(
                offset,
                static_cast<std::size_t>(kMutationCatalogDeltaEntrySize));
        put_uuid(encoded, 0, entry.delta_id);
        put_uuid(encoded, 16, entry.source_predecessor_wal_id);
        put_uuid(encoded, 32, entry.source_terminal_wal_id);
        put_u64(encoded, 48, entry.first_sequence);
        put_u64(encoded, 56, entry.last_sequence);
        put_u32(encoded, 64, entry.segment_id);
        put_u32(encoded, 68, entry.source_wal_count);
        put_u64(encoded, 72, entry.entry_count);
        put_u64(encoded, 80, entry.put_count);
        put_u64(encoded, 88, entry.data_file_size);
        put_u64(encoded, 96, entry.index_file_size);
        put_u32(encoded, 104, entry.data_payload_crc32c);
        put_u32(encoded, 108, entry.index_body_crc32c);
        // Reserved descriptor flags are fixed at zero in DeltaV1.
        put_u64(encoded, 112, 0);
    }

    for (std::size_t index = 0; index != catalog.wal_entries.size(); ++index) {
        const MutationWalEntry& entry = catalog.wal_entries[index];
        const std::size_t offset = static_cast<std::size_t>(
            wal_entries_offset
            + index * kMutationCatalogEntrySize);
        const std::span<std::byte> encoded =
            std::span<std::byte>(bytes).subspan(
                offset, static_cast<std::size_t>(kMutationCatalogEntrySize));
        put_uuid(encoded, 0, entry.wal_id);
        put_uuid(encoded, 16, entry.predecessor_wal_id);
        put_u32(encoded, 32, entry.segment_id);
        put_u32(encoded, 36, static_cast<std::uint32_t>(entry.state));
        put_u64(encoded, 40, entry.previous_sequence);
        put_u64(encoded, 48, entry.first_sequence);
        put_u64(encoded, 56, entry.last_sequence);
        put_u64(encoded, 64, entry.put_count);
        put_u64(encoded, 72, entry.sealed_file_size);
    }
    put_u32(bytes, kChecksumOffset, crc32c_with_zeroed_checksum(bytes));
    return bytes;
}

MutationCatalog decode_mutation_catalog(
    const std::span<const std::byte> bytes,
    const NativeManifest& manifest) {
    validate_manifest_for_catalog<CorruptionError>(manifest);
    require<CorruptionError>(
        bytes.size() >= kMutationCatalogHeaderSize,
        "mutation catalog is truncated");
    require_magic<CorruptionError>(bytes, kCatalogMagic, "mutation catalog");
    const std::uint32_t format_version = get_u32(bytes, 8);
    const bool decode_v2 = mutation_delta_managed(manifest);
    require<CorruptionError>(
        format_version
            == (decode_v2 ? kCatalogFormatVersionV2
                          : kCatalogFormatVersionV1),
        "mutation catalog version does not match required manifest features");
    require<CorruptionError>(
        get_u32(bytes, 12) == kMutationCatalogHeaderSize,
        "mutation catalog has an invalid header size");
    require<CorruptionError>(
        get_u32(bytes, 16) == kChecksumAlgorithmCrc32c,
        "mutation catalog has an unsupported checksum algorithm");
    require<CorruptionError>(
        get_u32(bytes, kChecksumOffset)
            == crc32c_with_zeroed_checksum(bytes),
        "mutation catalog checksum mismatch");
    require_zero_range<CorruptionError>(
        bytes,
        decode_v2 ? kCatalogV2KnownHeaderBytes
                  : kCatalogV1KnownHeaderBytes,
        static_cast<std::size_t>(kMutationCatalogHeaderSize),
        "mutation catalog header");

    const std::uint64_t wal_count = get_u64(bytes, 104);
    require<CorruptionError>(
        wal_count != 0 && wal_count <= kMutationCatalogMaxEntries,
        "mutation catalog WAL entry count is invalid");
    require<CorruptionError>(
        get_u32(bytes, 100) == kMutationCatalogEntrySize,
        "mutation catalog has an invalid WAL entry size");

    std::uint64_t delta_count = 0;
    if (decode_v2) {
        require<CorruptionError>(
            get_u32(bytes, 152) == kMutationCatalogDeltaEntrySize,
            "mutation catalog has an invalid delta entry size");
        require<CorruptionError>(
            get_u32(bytes, 156) == 0,
            "mutation catalog has unsupported flags");
        delta_count = get_u64(bytes, 160);
        require<CorruptionError>(
            get_u64(bytes, 168) == kMutationCatalogHeaderSize,
            "mutation catalog has an invalid delta entries offset");
    }
    require<CorruptionError>(
        delta_count <= kMutationCatalogMaxEntries - wal_count,
        "mutation catalog resource count exceeds the format maximum");
    const std::uint64_t expected_wal_offset = checked_sum<CorruptionError>(
        kMutationCatalogHeaderSize,
        checked_product<CorruptionError>(
            delta_count, kMutationCatalogDeltaEntrySize,
            "catalog delta entry bytes"),
        "catalog WAL entries offset");
    require<CorruptionError>(
        get_u64(bytes, 112) == expected_wal_offset,
        "mutation catalog has an invalid WAL entries offset");
    const std::uint64_t expected_size = checked_sum<CorruptionError>(
        expected_wal_offset,
        checked_product<CorruptionError>(
            wal_count, kMutationCatalogEntrySize,
            "catalog WAL entry bytes"),
        "catalog file size");
    require<CorruptionError>(
        get_u64(bytes, 120) == expected_size
            && bytes.size() == expected_size,
        "mutation catalog file size is invalid");

    MutationCatalog catalog;
    catalog.collection_id = get_uuid(bytes, 24);
    catalog.schema_id = get_uuid(bytes, 40);
    catalog.catalog_id = get_uuid(bytes, 56);
    catalog.previous_catalog_id = get_uuid(bytes, 72);
    catalog.generation = get_u64(bytes, 88);
    catalog.next_segment_id = get_u32(bytes, 96);
    catalog.materialized_sequence = get_u64(bytes, 128);
    catalog.terminal_wal_id = get_uuid(bytes, 136);
    catalog.delta_entries.reserve(static_cast<std::size_t>(delta_count));
    for (std::uint64_t index = 0; index != delta_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            kMutationCatalogHeaderSize
            + index * kMutationCatalogDeltaEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset,
            static_cast<std::size_t>(kMutationCatalogDeltaEntrySize));
        require<CorruptionError>(
            get_u64(encoded, 112) == 0,
            "mutation catalog delta entry has unsupported flags");
        require_zero_range<CorruptionError>(
            encoded, kDeltaEntryKnownBytes, encoded.size(),
            "mutation catalog delta entry");
        MutationDeltaDescriptor entry;
        entry.delta_id = get_uuid(encoded, 0);
        entry.source_predecessor_wal_id = get_uuid(encoded, 16);
        entry.source_terminal_wal_id = get_uuid(encoded, 32);
        entry.first_sequence = get_u64(encoded, 48);
        entry.last_sequence = get_u64(encoded, 56);
        entry.segment_id = get_u32(encoded, 64);
        entry.source_wal_count = get_u32(encoded, 68);
        entry.entry_count = get_u64(encoded, 72);
        entry.put_count = get_u64(encoded, 80);
        entry.data_file_size = get_u64(encoded, 88);
        entry.index_file_size = get_u64(encoded, 96);
        entry.data_payload_crc32c = get_u32(encoded, 104);
        entry.index_body_crc32c = get_u32(encoded, 108);
        catalog.delta_entries.push_back(entry);
    }
    catalog.wal_entries.reserve(static_cast<std::size_t>(wal_count));
    for (std::uint64_t index = 0; index != wal_count; ++index) {
        const std::size_t offset = static_cast<std::size_t>(
            expected_wal_offset + index * kMutationCatalogEntrySize);
        const std::span<const std::byte> encoded = bytes.subspan(
            offset, static_cast<std::size_t>(kMutationCatalogEntrySize));
        require_zero_range<CorruptionError>(
            encoded, kEntryKnownBytes, encoded.size(),
            "mutation catalog WAL entry");
        MutationWalEntry entry;
        entry.wal_id = get_uuid(encoded, 0);
        entry.predecessor_wal_id = get_uuid(encoded, 16);
        entry.segment_id = get_u32(encoded, 32);
        entry.state = static_cast<MutationWalState>(get_u32(encoded, 36));
        entry.previous_sequence = get_u64(encoded, 40);
        entry.first_sequence = get_u64(encoded, 48);
        entry.last_sequence = get_u64(encoded, 56);
        entry.put_count = get_u64(encoded, 64);
        entry.sealed_file_size = get_u64(encoded, 72);
        catalog.wal_entries.push_back(entry);
    }
    validate_catalog<CorruptionError>(manifest, catalog);
    return catalog;
}

std::array<std::byte, kMutationCurrentFileSize> encode_mutation_current(
    const NativeManifest& manifest,
    const MutationCurrent& current) {
    validate_current<InvalidArgumentError>(manifest, current);
    std::array<std::byte, kMutationCurrentFileSize> bytes{};
    put_magic(bytes, kCurrentMagic);
    put_u32(bytes, 8, kCurrentFormatVersion);
    put_u32(bytes, 12, static_cast<std::uint32_t>(kMutationCurrentFileSize));
    put_u32(bytes, 16, kChecksumAlgorithmCrc32c);
    put_uuid(bytes, 24, current.collection_id);
    put_uuid(bytes, 40, current.schema_id);
    put_u64(bytes, 56, current.generation);
    put_uuid(bytes, 64, current.catalog_id);
    put_u64(bytes, 80, current.catalog_file_size);
    put_u32(bytes, 88, current.catalog_crc32c);
    put_u32(bytes, 92, 0);
    put_u32(bytes, kChecksumOffset, crc32c_with_zeroed_checksum(bytes));
    return bytes;
}

MutationCurrent decode_mutation_current(
    const std::span<const std::byte> bytes,
    const NativeManifest& manifest) {
    validate_manifest_for_catalog<CorruptionError>(manifest);
    require<CorruptionError>(
        bytes.size() == kMutationCurrentFileSize,
        "CURRENT file size is not 4096 bytes");
    require_magic<CorruptionError>(bytes, kCurrentMagic, "CURRENT");
    require<CorruptionError>(
        get_u32(bytes, 8) == kCurrentFormatVersion,
        "CURRENT has an unsupported format version");
    require<CorruptionError>(
        get_u32(bytes, 12) == kMutationCurrentFileSize,
        "CURRENT has an invalid header size");
    require<CorruptionError>(
        get_u32(bytes, 16) == kChecksumAlgorithmCrc32c,
        "CURRENT has an unsupported checksum algorithm");
    require<CorruptionError>(
        get_u32(bytes, kChecksumOffset)
            == crc32c_with_zeroed_checksum(bytes),
        "CURRENT checksum mismatch");
    require<CorruptionError>(
        get_u32(bytes, 92) == 0,
        "CURRENT has unsupported flags");
    require_zero_range<CorruptionError>(
        bytes, kCurrentKnownBytes, bytes.size(), "CURRENT");

    MutationCurrent current;
    current.collection_id = get_uuid(bytes, 24);
    current.schema_id = get_uuid(bytes, 40);
    current.generation = get_u64(bytes, 56);
    current.catalog_id = get_uuid(bytes, 64);
    current.catalog_file_size = get_u64(bytes, 80);
    current.catalog_crc32c = get_u32(bytes, 88);
    validate_current<CorruptionError>(manifest, current);
    return current;
}

MutationCatalog make_initial_mutation_catalog(
    const NativeManifest& manifest,
    const Uuid128 catalog_id) {
    validate_manifest_for_catalog<InvalidArgumentError>(manifest);
    MutationCatalog catalog;
    catalog.collection_id = manifest.collection_id;
    catalog.schema_id = manifest.schema_id;
    catalog.generation = 1;
    catalog.catalog_id = catalog_id;
    catalog.next_segment_id = manifest.next_segment_id;
    catalog.wal_entries.push_back(MutationWalEntry{
        manifest.active_wal_id,
        {},
        manifest.active_wal_segment_id,
        MutationWalState::Active,
        0,
        1,
        0,
        0,
        0});
    validate_catalog<InvalidArgumentError>(manifest, catalog);
    return catalog;
}

MutationCurrent publish_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationCatalog& catalog) {
    const std::vector<std::byte> catalog_bytes =
        encode_mutation_catalog(manifest, catalog);
    MutationCurrent current;
    current.collection_id = catalog.collection_id;
    current.schema_id = catalog.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = catalog_bytes.size();
    current.catalog_crc32c = crc32c(catalog_bytes);
    const auto current_bytes = encode_mutation_current(manifest, current);

    const auto catalog_directory = mutation_catalogs_directory(store_path);
    const auto catalog_temporary =
        mutation_catalog_temporary_path(store_path, catalog.catalog_id);
    const auto catalog_final =
        mutation_catalog_path(store_path, catalog.catalog_id);
    std::error_code exists_error;
    const bool catalog_exists =
        std::filesystem::exists(catalog_final, exists_error);
    if (exists_error) {
        throw StorageError(
            "cannot inspect immutable catalog destination: "
            + exists_error.message());
    }
    if (catalog_exists) {
        throw InvalidArgumentError(
            "immutable catalog UUID already exists: '"
            + catalog_final.string() + "'");
    }
    auto catalog_fd = create_exclusive(catalog_temporary);
    try {
        write_all(
            catalog_fd.get(), catalog_bytes.data(), catalog_bytes.size(),
            catalog_temporary);
        sync_file(catalog_fd.get(), catalog_temporary);
        catalog_fd.reset();
        rename_file(catalog_temporary, catalog_final, "mutation catalog");
        sync_directory(catalog_directory);
    } catch (...) {
        catalog_fd.reset();
        throw;
    }

    const auto current_temporary =
        mutation_current_temporary_path(store_path, catalog.catalog_id);
    auto current_fd = create_exclusive(current_temporary);
    try {
        write_all(
            current_fd.get(), current_bytes.data(), current_bytes.size(),
            current_temporary);
        sync_file(current_fd.get(), current_temporary);
        current_fd.reset();
        rename_file(
            current_temporary, mutation_current_path(store_path), "CURRENT");
        sync_directory(store_path);
    } catch (...) {
        current_fd.reset();
        throw;
    }
    return current;
}

MutationCurrent read_mutation_current(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    const auto path = mutation_current_path(store_path);
    const std::vector<std::byte> bytes =
        read_exact_file(path, kMutationCurrentFileSize);
    return decode_mutation_current(bytes, manifest);
}

MutationCatalog read_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationCurrent& current) {
    validate_current<CorruptionError>(manifest, current);
    const auto path = mutation_catalog_path(store_path, current.catalog_id);
    const std::vector<std::byte> bytes =
        read_exact_file(path, current.catalog_file_size);
    require<CorruptionError>(
        crc32c(bytes) == current.catalog_crc32c,
        "catalog whole-file checksum does not match CURRENT");
    MutationCatalog catalog = decode_mutation_catalog(bytes, manifest);
    require<CorruptionError>(
        catalog.generation == current.generation,
        "catalog generation does not match CURRENT");
    require<CorruptionError>(
        catalog.catalog_id == current.catalog_id,
        "catalog UUID does not match CURRENT");
    return catalog;
}

MutationCatalog read_current_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest) {
    const MutationCurrent current = read_mutation_current(store_path, manifest);
    return read_mutation_catalog(store_path, manifest, current);
}

}  // namespace throwdb::detail
