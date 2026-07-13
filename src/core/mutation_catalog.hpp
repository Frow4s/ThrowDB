#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "native_format.hpp"
#include "mutation_delta.hpp"

namespace throwdb::detail {

inline constexpr std::uint64_t kMutationCatalogHeaderSize = 4096;
inline constexpr std::uint64_t kMutationCatalogEntrySize = 128;
inline constexpr std::uint64_t kMutationCatalogDeltaEntrySize = 192;
inline constexpr std::uint64_t kMutationCurrentFileSize = 4096;
inline constexpr std::uint64_t kMutationCatalogMaxEntries = 1'000'000;

enum class MutationWalState : std::uint32_t {
    Frozen = 1,
    Active = 2,
};

struct MutationWalEntry {
    Uuid128 wal_id;
    Uuid128 predecessor_wal_id;
    std::uint32_t segment_id{0};
    MutationWalState state{MutationWalState::Active};
    std::uint64_t previous_sequence{0};
    std::uint64_t first_sequence{1};
    // Frozen WALs store their inclusive final sequence. Active WALs use zero.
    std::uint64_t last_sequence{0};
    // Frozen WALs store their immutable PUT-row count. Active WALs use zero.
    std::uint64_t put_count{0};
    // Frozen WALs store their exact immutable size. Active WALs use zero.
    std::uint64_t sealed_file_size{0};

    bool operator==(const MutationWalEntry&) const = default;
};

struct MutationCatalog {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint64_t generation{0};
    Uuid128 catalog_id;
    // Empty only for generation 1; later generations name their direct
    // immutable predecessor.
    Uuid128 previous_catalog_id;
    std::uint32_t next_segment_id{0};
    // Lineage anchor for WALs already incorporated into materialized data.
    // terminal_wal_id is the last WAL incorporated into that state. Initial
    // catalogs use sequence zero and an empty terminal UUID.
    std::uint64_t materialized_sequence{0};
    Uuid128 terminal_wal_id;
    // Oldest materialized sequence interval first. Catalog wire v2 stores
    // these immutable delta descriptors before the retained WAL entries.
    std::vector<MutationDeltaDescriptor> delta_entries;
    // Oldest retained frozen WAL first; exactly one active WAL is last.
    std::vector<MutationWalEntry> wal_entries;

    bool operator==(const MutationCatalog&) const = default;
};

struct MutationCurrent {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint64_t generation{0};
    Uuid128 catalog_id;
    std::uint64_t catalog_file_size{0};
    std::uint32_t catalog_crc32c{0};

    bool operator==(const MutationCurrent&) const = default;
};

[[nodiscard]] bool mutation_catalog_managed(
    const NativeManifest& manifest) noexcept;
[[nodiscard]] bool mutation_delta_managed(
    const NativeManifest& manifest) noexcept;

[[nodiscard]] std::filesystem::path mutation_catalogs_directory(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path mutation_catalog_path(
    const std::filesystem::path& store_path, Uuid128 catalog_id);
[[nodiscard]] std::filesystem::path mutation_wal_path(
    const std::filesystem::path& store_path, Uuid128 wal_id);
[[nodiscard]] std::filesystem::path mutation_current_path(
    const std::filesystem::path& store_path);

[[nodiscard]] std::vector<std::byte> encode_mutation_catalog(
    const NativeManifest& manifest,
    const MutationCatalog& catalog);
[[nodiscard]] MutationCatalog decode_mutation_catalog(
    std::span<const std::byte> bytes,
    const NativeManifest& manifest);

[[nodiscard]] std::array<std::byte, kMutationCurrentFileSize>
encode_mutation_current(
    const NativeManifest& manifest,
    const MutationCurrent& current);
[[nodiscard]] MutationCurrent decode_mutation_current(
    std::span<const std::byte> bytes,
    const NativeManifest& manifest);

[[nodiscard]] MutationCatalog make_initial_mutation_catalog(
    const NativeManifest& manifest,
    Uuid128 catalog_id);

// The caller serializes publishers with the collection writer lease. Durable
// publication order uses UUID-suffixed temporary files so crash orphans never
// block another generation:
// catalog.<uuid>.tmp write+sync -> immutable catalog rename+catalogs fsync ->
// CURRENT.<uuid>.tmp write+sync -> CURRENT rename+store-root fsync.
[[nodiscard]] MutationCurrent publish_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationCatalog& catalog);

[[nodiscard]] MutationCurrent read_mutation_current(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);
[[nodiscard]] MutationCatalog read_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const MutationCurrent& current);
[[nodiscard]] MutationCatalog read_current_mutation_catalog(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);

}  // namespace throwdb::detail
