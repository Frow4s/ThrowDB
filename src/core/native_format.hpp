#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

#include "internal.hpp"
#include "mutation_types.hpp"

namespace throwdb::detail {

// Native mutable format v1 is deliberately independent of immutable
// FormatVersion::V1/V2. Every file uses a 4 KiB checksummed header and
// little-endian integers.
inline constexpr std::uint32_t kNativeFormatVersion = 1;
inline constexpr std::uint64_t kNativeHeaderSize = 4096;
inline constexpr std::uint64_t kNativeIndexSlotSize = 16;
inline constexpr std::uint64_t kNativeIdEntrySize = 8;
inline constexpr std::uint32_t kNativeCrc32cAlgorithm = 1;
inline constexpr std::uint32_t kNativeFlatControlEncoding = 1;
inline constexpr std::uint32_t kNativeRowRefValueEncoding = 1;
inline constexpr std::uint32_t kNativeIndexBodyCrc32cAlgorithm = 1;
inline constexpr std::size_t kNativeHeaderChecksumOffset = 20;
inline constexpr std::uint64_t kNativeRequiredFeatureCatalogV1 = 1ULL << 0;
inline constexpr std::uint64_t kNativeRequiredFeatureDeltaV1 = 1ULL << 1;
inline constexpr std::uint64_t kNativeSupportedRequiredFeatureBits =
    kNativeRequiredFeatureCatalogV1 | kNativeRequiredFeatureDeltaV1;

inline constexpr std::array<char, 8> kNativeManifestMagic{
    'T', 'D', 'B', 'N', 'M', 'N', 'F', '1'};
inline constexpr std::array<char, 8> kNativeDataMagic{
    'T', 'D', 'B', 'N', 'D', 'A', 'T', '1'};
inline constexpr std::array<char, 8> kNativeIdsMagic{
    'T', 'D', 'B', 'N', 'I', 'D', 'S', '1'};
inline constexpr std::array<char, 8> kNativeIndexMagic{
    'T', 'D', 'B', 'N', 'I', 'D', 'X', '1'};

struct NativeManifest {
    Uuid128 collection_id;
    Uuid128 schema_id;
    Uuid128 active_wal_id;

    CollectionSchema schema;
    std::uint64_t record_size{0};
    std::uint64_t base_record_count{0};
    std::uint64_t segment_payload_capacity{0};
    std::uint64_t records_per_segment{0};
    std::uint64_t base_segment_count{0};

    std::uint64_t index_capacity{0};
    std::uint64_t index_control_offset{kNativeHeaderSize};
    std::uint64_t index_slots_offset{kNativeHeaderSize};
    std::uint64_t index_slot_size{kNativeIndexSlotSize};
    std::uint64_t index_file_size{kNativeHeaderSize};
    std::uint64_t hash_seed{kHashSeed};

    std::uint32_t active_wal_segment_id{1};
    std::uint32_t next_segment_id{2};
    // Catalog-managed collections set CatalogV1. Collections whose catalog
    // may reference immutable mutation deltas additionally set DeltaV1. A
    // reader must reject an unknown required bit instead of silently opening
    // an incomplete view.
    std::uint64_t required_feature_bits{0};
    std::uint64_t optional_feature_bits{0};
};

struct NativeDataHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint32_t segment_id{0};
    DTypeCode dtype{DTypeCode::Invalid};
    std::uint32_t dimension{0};
    std::uint32_t item_size{0};
    IdType id_type{IdType::Invalid};
    RowLayout row_layout{RowLayout::Invalid};
    std::uint64_t record_size{0};
    std::uint64_t row_count{0};
    std::uint64_t payload_offset{kNativeHeaderSize};
    std::uint64_t payload_size{0};
    std::uint64_t file_size{kNativeHeaderSize};
};

struct NativeIdsHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint32_t segment_id{0};
    IdType id_type{IdType::Invalid};
    std::uint64_t row_count{0};
    std::uint64_t payload_offset{kNativeHeaderSize};
    std::uint64_t payload_size{0};
    std::uint64_t file_size{kNativeHeaderSize};
    std::uint32_t entry_size{kNativeIdEntrySize};
};

struct NativeIndexHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    std::uint64_t record_count{0};
    std::uint64_t capacity{0};
    std::uint64_t hash_seed{kHashSeed};
    std::uint64_t control_offset{kNativeHeaderSize};
    std::uint64_t slots_offset{kNativeHeaderSize};
    std::uint64_t slot_size{kNativeIndexSlotSize};
    std::uint64_t file_size{kNativeHeaderSize};
    std::uint32_t control_encoding{kNativeFlatControlEncoding};
    std::uint32_t value_encoding{kNativeRowRefValueEncoding};
    std::uint64_t base_segment_count{0};
    // Zero denotes a legacy index without a body checksum. New writers store
    // CRC32C over every byte after the header (controls, alignment padding,
    // and slots). The header checksum protects both fields themselves.
    std::uint32_t body_checksum_algorithm{0};
    std::uint32_t body_checksum{0};
};

enum class NativeIndexValidationMode {
    // Verify a persisted body checksum when available. Legacy indexes fall
    // back to the full structural scan rather than silently skipping it.
    ChecksumOrFull,
    // Always run the complete slot/RowRef/probe validation after loading.
    Full,
};

struct NativeIndexReadStats {
    double header_seconds{0.0};
    double allocation_seconds{0.0};
    double controls_seconds{0.0};
    double slots_seconds{0.0};
    double checksum_seconds{0.0};
    double structural_validation_seconds{0.0};
    bool body_checksum_present{false};
    bool body_checksum_verified{false};
    bool structural_validation_performed{false};
};

// Fills every derived schema, segment, index, and reserved-segment field.
// The caller supplies nonzero UUIDs, dtype, dimension, base_record_count,
// segment_payload_capacity, and optionally hash_seed before calling it.
void derive_native_manifest_geometry(NativeManifest& manifest);

[[nodiscard]] std::uint64_t choose_native_index_capacity(
    std::uint64_t record_count);
[[nodiscard]] std::uint64_t native_base_segment_row_count(
    const NativeManifest& manifest, std::uint32_t segment_id);

[[nodiscard]] std::array<std::byte, kNativeHeaderSize>
encode_native_manifest(const NativeManifest& manifest);
[[nodiscard]] NativeManifest decode_native_manifest(
    std::span<const std::byte> bytes);

[[nodiscard]] NativeDataHeader make_native_data_header(
    const NativeManifest& manifest, std::uint32_t segment_id);
[[nodiscard]] std::array<std::byte, kNativeHeaderSize>
encode_native_data_header(const NativeDataHeader& header);
[[nodiscard]] NativeDataHeader decode_native_data_header(
    std::span<const std::byte> bytes);

[[nodiscard]] NativeIdsHeader make_native_ids_header(
    const NativeManifest& manifest, std::uint32_t segment_id);
[[nodiscard]] std::array<std::byte, kNativeHeaderSize>
encode_native_ids_header(const NativeIdsHeader& header);
[[nodiscard]] NativeIdsHeader decode_native_ids_header(
    std::span<const std::byte> bytes);

[[nodiscard]] NativeIndexHeader make_native_index_header(
    const NativeManifest& manifest);
[[nodiscard]] std::array<std::byte, kNativeHeaderSize>
encode_native_index_header(const NativeIndexHeader& header);
[[nodiscard]] NativeIndexHeader decode_native_index_header(
    std::span<const std::byte> bytes);

[[nodiscard]] std::filesystem::path native_manifest_path(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path native_data_segment_path(
    const std::filesystem::path& store_path, std::uint32_t segment_id);
[[nodiscard]] std::filesystem::path native_ids_segment_path(
    const std::filesystem::path& store_path, std::uint32_t segment_id);
[[nodiscard]] std::filesystem::path native_index_path(
    const std::filesystem::path& store_path);
[[nodiscard]] std::filesystem::path native_active_wal_path(
    const std::filesystem::path& store_path);

[[nodiscard]] NativeManifest read_native_manifest(
    const std::filesystem::path& store_path);
void write_native_manifest_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);

[[nodiscard]] NativeDataHeader read_native_data_header(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    std::uint32_t segment_id);
[[nodiscard]] NativeIdsHeader read_native_ids_header(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    std::uint32_t segment_id);

// IndexData::slots[].ordinal contains a raw RowRef in these private APIs.
// build_and_write scans the authoritative per-row ID sidecars; no global
// ordinal staging file is involved.
[[nodiscard]] IndexData build_native_index(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);
void write_native_index_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    const IndexData& index);
void build_and_write_native_index_atomic(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest);
[[nodiscard]] IndexData read_native_index(
    const std::filesystem::path& store_path,
    const NativeManifest& manifest,
    NativeIndexValidationMode validation_mode =
        NativeIndexValidationMode::Full,
    NativeIndexReadStats* stats = nullptr);
[[nodiscard]] std::optional<RowRef> native_index_lookup(
    const IndexData& index, std::uint64_t id);

}  // namespace throwdb::detail
