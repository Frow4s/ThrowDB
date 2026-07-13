#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/mutation_catalog.hpp"

namespace {

using throwdb::detail::MutationCatalog;
using throwdb::detail::MutationCurrent;
using throwdb::detail::MutationDeltaDescriptor;
using throwdb::detail::MutationWalEntry;
using throwdb::detail::MutationWalState;
using throwdb::detail::NativeManifest;
using throwdb::detail::Uuid128;

constexpr std::size_t kChecksumOffset = 20;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-mutation-catalog-test-"
                   + std::to_string(
                       std::chrono::steady_clock::now()
                           .time_since_epoch().count())
                   + "-" + std::to_string(sequence++));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Actual, typename Expected>
void require_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string& message) {
    if (!(actual == expected)) {
        fail(message);
    }
}

template <typename Exception, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
    try {
        std::forward<Callback>(callback)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        fail(message + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        fail(message + " (wrong non-standard exception)");
    }
    fail(message + " (no exception)");
}

[[nodiscard]] NativeManifest make_manifest() {
    NativeManifest manifest;
    manifest.collection_id = {
        0x0102030405060708ULL, 0x1112131415161718ULL};
    manifest.schema_id = {
        0x2122232425262728ULL, 0x3132333435363738ULL};
    manifest.active_wal_id = {
        0x4142434445464748ULL, 0x5152535455565758ULL};
    manifest.schema.dtype = throwdb::detail::DTypeCode::Float16;
    manifest.schema.dimension = 4;
    manifest.base_record_count = 0;
    manifest.segment_payload_capacity = 4096;
    manifest.hash_seed = 0x6162636465666768ULL;
    manifest.required_feature_bits =
        throwdb::detail::kNativeRequiredFeatureCatalogV1;
    throwdb::detail::derive_native_manifest_geometry(manifest);
    return manifest;
}

[[nodiscard]] NativeManifest make_delta_manifest() {
    NativeManifest manifest = make_manifest();
    manifest.required_feature_bits |=
        throwdb::detail::kNativeRequiredFeatureDeltaV1;
    return manifest;
}

[[nodiscard]] MutationDeltaDescriptor make_delta_descriptor(
    const NativeManifest& manifest,
    const Uuid128 delta_id,
    const Uuid128 predecessor_wal_id,
    const Uuid128 terminal_wal_id,
    const std::uint64_t first_sequence,
    const std::uint64_t last_sequence,
    const std::uint32_t segment_id,
    const std::uint64_t entry_count,
    const std::uint64_t put_count) {
    MutationDeltaDescriptor descriptor;
    descriptor.delta_id = delta_id;
    descriptor.source_predecessor_wal_id = predecessor_wal_id;
    descriptor.source_terminal_wal_id = terminal_wal_id;
    descriptor.first_sequence = first_sequence;
    descriptor.last_sequence = last_sequence;
    descriptor.segment_id = segment_id;
    descriptor.source_wal_count = 1;
    descriptor.entry_count = entry_count;
    descriptor.put_count = put_count;
    descriptor.data_file_size =
        throwdb::detail::kMutationDeltaHeaderSize
        + put_count * manifest.record_size;
    const std::uint64_t records_end =
        throwdb::detail::kMutationDeltaHeaderSize
        + entry_count * throwdb::detail::kMutationDeltaVersionRecordSize;
    const std::uint64_t reverse_offset =
        (records_end + throwdb::detail::kMutationDeltaIndexAlignment - 1)
        & ~(throwdb::detail::kMutationDeltaIndexAlignment - 1);
    descriptor.index_file_size =
        reverse_offset
        + put_count * throwdb::detail::kMutationDeltaReverseRecordSize;
    descriptor.data_payload_crc32c = 0x12345678U;
    descriptor.index_body_crc32c = 0x90abcdefU;
    return descriptor;
}

[[nodiscard]] MutationCatalog make_rotated_catalog(
    const NativeManifest& manifest,
    const Uuid128 catalog_id,
    const Uuid128 previous_catalog_id) {
    MutationCatalog catalog = throwdb::detail::make_initial_mutation_catalog(
        manifest, catalog_id);
    catalog.generation = 2;
    catalog.previous_catalog_id = previous_catalog_id;
    MutationWalEntry& frozen = catalog.wal_entries.front();
    frozen.state = MutationWalState::Frozen;
    frozen.last_sequence = 7;
    frozen.put_count = 3;
    frozen.sealed_file_size = 8192;
    const Uuid128 active_id{
        0x7172737475767778ULL, 0x8182838485868788ULL};
    catalog.wal_entries.push_back(MutationWalEntry{
        active_id,
        frozen.wal_id,
        manifest.next_segment_id,
        MutationWalState::Active,
        frozen.last_sequence,
        frozen.last_sequence + 1,
        0,
        0,
        0});
    catalog.next_segment_id = manifest.next_segment_id + 1;
    return catalog;
}

[[nodiscard]] MutationCatalog make_delta_catalog(
    const NativeManifest& manifest) {
    MutationCatalog catalog = make_rotated_catalog(
        manifest, Uuid128{3, 4}, Uuid128{1, 2});
    const MutationWalEntry frozen = catalog.wal_entries.front();
    catalog.generation = 3;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = {5, 6};
    catalog.materialized_sequence = frozen.last_sequence;
    catalog.terminal_wal_id = frozen.wal_id;
    catalog.delta_entries.push_back(make_delta_descriptor(
        manifest, Uuid128{0xd1, 0xd2}, frozen.predecessor_wal_id,
        frozen.wal_id, frozen.first_sequence, frozen.last_sequence,
        catalog.next_segment_id, 3, 2));
    ++catalog.next_segment_id;
    catalog.wal_entries.erase(catalog.wal_entries.begin());
    return catalog;
}

[[nodiscard]] MutationCatalog make_two_delta_catalog(
    const NativeManifest& manifest) {
    MutationCatalog catalog = make_delta_catalog(manifest);
    const MutationWalEntry second_source = catalog.wal_entries.front();
    const Uuid128 final_active_id{0xc1, 0xc2};
    catalog.delta_entries.push_back(make_delta_descriptor(
        manifest, Uuid128{0xe1, 0xe2}, catalog.terminal_wal_id,
        second_source.wal_id, second_source.first_sequence, 11,
        catalog.next_segment_id + 1, 2, 1));
    catalog.materialized_sequence = 11;
    catalog.terminal_wal_id = second_source.wal_id;
    catalog.wal_entries.front() = MutationWalEntry{
        final_active_id,
        second_source.wal_id,
        catalog.next_segment_id,
        MutationWalState::Active,
        11,
        12,
        0,
        0,
        0};
    catalog.next_segment_id += 2;
    catalog.generation = 4;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = {7, 8};
    return catalog;
}

[[nodiscard]] std::uint32_t checksum_with_zeroed_field(
    const std::span<const std::byte> bytes) {
    throwdb::detail::Crc32c checksum;
    checksum.update(bytes.first(kChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(kChecksumOffset + sizeof(std::uint32_t)));
    return checksum.value();
}

void reseal(std::span<std::byte> bytes) {
    throwdb::detail::put_u32(bytes, kChecksumOffset, 0);
    throwdb::detail::put_u32(
        bytes, kChecksumOffset,
        checksum_with_zeroed_field(std::span<const std::byte>(bytes)));
}

void write_bytes(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good(), "cannot create test file");
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "cannot write test file");
}

void test_initial_roundtrip_and_uuid_paths() {
    const NativeManifest manifest = make_manifest();
    const Uuid128 catalog_id{
        0x9192939495969798ULL, 0xa1a2a3a4a5a6a7a8ULL};
    const MutationCatalog catalog =
        throwdb::detail::make_initial_mutation_catalog(manifest, catalog_id);
    require_equal(catalog.generation, std::uint64_t{1},
                  "initial generation is wrong");
    require(catalog.previous_catalog_id.empty(),
            "initial catalog has a predecessor");
    require_equal(catalog.wal_entries.size(), std::size_t{1},
                  "initial catalog does not have one WAL");
    require_equal(catalog.wal_entries.front().state, MutationWalState::Active,
                  "initial WAL is not active");

    const std::vector<std::byte> encoded =
        throwdb::detail::encode_mutation_catalog(manifest, catalog);
    require_equal(
        encoded.size(),
        static_cast<std::size_t>(
            throwdb::detail::kMutationCatalogHeaderSize
            + throwdb::detail::kMutationCatalogEntrySize),
        "initial encoded catalog has wrong size");
    require_equal(
        throwdb::detail::decode_mutation_catalog(encoded, manifest), catalog,
        "initial catalog roundtrip changed fields");

    MutationCurrent current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = encoded.size();
    current.catalog_crc32c = throwdb::detail::crc32c(encoded);
    const auto encoded_current =
        throwdb::detail::encode_mutation_current(manifest, current);
    require_equal(
        throwdb::detail::decode_mutation_current(encoded_current, manifest),
        current, "CURRENT roundtrip changed fields");

    const std::filesystem::path root = "/tmp/example";
    require_equal(
        throwdb::detail::mutation_catalog_path(root, catalog_id).filename(),
        std::filesystem::path(
            "catalog-a1a2a3a4a5a6a7a89192939495969798.bin"),
        "catalog UUID path is unstable");
    require_equal(
        throwdb::detail::mutation_wal_path(
            root, manifest.active_wal_id).filename(),
        std::filesystem::path(
            "wal-51525354555657584142434445464748.wal"),
        "WAL UUID path is unstable");
    require_equal(
        throwdb::detail::mutation_current_path(root), root / "CURRENT",
        "CURRENT path is wrong");
}

void test_rotated_and_materialized_chain_roundtrip() {
    const NativeManifest manifest = make_manifest();
    const Uuid128 first_catalog_id{1, 2};
    const MutationCatalog rotated = make_rotated_catalog(
        manifest, Uuid128{3, 4}, first_catalog_id);
    const std::vector<std::byte> encoded =
        throwdb::detail::encode_mutation_catalog(manifest, rotated);
    require_equal(
        throwdb::detail::decode_mutation_catalog(encoded, manifest), rotated,
        "rotated catalog roundtrip changed fields");

    MutationCatalog materialized = rotated;
    materialized.generation = 3;
    materialized.previous_catalog_id = rotated.catalog_id;
    materialized.catalog_id = {5, 6};
    materialized.materialized_sequence =
        materialized.wal_entries.front().last_sequence;
    materialized.terminal_wal_id =
        materialized.wal_entries.front().wal_id;
    materialized.wal_entries.erase(materialized.wal_entries.begin());
    require_equal(
        throwdb::detail::decode_mutation_catalog(
            throwdb::detail::encode_mutation_catalog(manifest, materialized),
            manifest),
        materialized, "materialized-prefix catalog is not representable");
}

void test_catalog_v1_v2_compatibility() {
    const NativeManifest v1_manifest = make_manifest();
    const MutationCatalog initial =
        throwdb::detail::make_initial_mutation_catalog(
            v1_manifest, Uuid128{3, 4});
    const std::vector<std::byte> v1_bytes =
        throwdb::detail::encode_mutation_catalog(v1_manifest, initial);
    require_equal(
        throwdb::detail::get_u32(v1_bytes, 8), std::uint32_t{1},
        "catalog without DeltaV1 was not encoded as wire v1");
    require_equal(
        throwdb::detail::decode_mutation_catalog(v1_bytes, v1_manifest),
        initial, "catalog wire v1 no longer decodes");

    const NativeManifest v2_manifest = make_delta_manifest();
    const std::vector<std::byte> v2_bytes =
        throwdb::detail::encode_mutation_catalog(v2_manifest, initial);
    require_equal(
        throwdb::detail::get_u32(v2_bytes, 8), std::uint32_t{2},
        "DeltaV1 catalog was not encoded as wire v2");
    require_equal(
        throwdb::detail::get_u64(v2_bytes, 160), std::uint64_t{0},
        "initial wire v2 catalog unexpectedly has deltas");
    require_equal(
        throwdb::detail::get_u64(v2_bytes, 168),
        throwdb::detail::kMutationCatalogHeaderSize,
        "wire v2 delta offset is wrong");
    require_equal(
        throwdb::detail::decode_mutation_catalog(v2_bytes, v2_manifest),
        initial, "empty catalog wire v2 does not roundtrip");

    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::decode_mutation_catalog(
                v1_bytes, v2_manifest));
        },
        "wire v1 was accepted when DeltaV1 requires wire v2");
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::decode_mutation_catalog(
                v2_bytes, v1_manifest));
        },
        "wire v2 was accepted without the DeltaV1 required feature");
}

void test_delta_v2_roundtrip_and_geometry() {
    const NativeManifest manifest = make_delta_manifest();
    const MutationCatalog catalog = make_delta_catalog(manifest);
    const std::vector<std::byte> encoded =
        throwdb::detail::encode_mutation_catalog(manifest, catalog);
    const std::uint64_t expected_wal_offset =
        throwdb::detail::kMutationCatalogHeaderSize
        + throwdb::detail::kMutationCatalogDeltaEntrySize;
    const std::uint64_t expected_size =
        expected_wal_offset + throwdb::detail::kMutationCatalogEntrySize;
    require_equal(
        encoded.size(), static_cast<std::size_t>(expected_size),
        "wire v2 catalog size is wrong");
    require_equal(
        throwdb::detail::get_u32(encoded, 8), std::uint32_t{2},
        "delta catalog has the wrong wire version");
    require_equal(
        throwdb::detail::get_u32(encoded, 152),
        static_cast<std::uint32_t>(
            throwdb::detail::kMutationCatalogDeltaEntrySize),
        "wire v2 delta entry size is wrong");
    require_equal(
        throwdb::detail::get_u64(encoded, 160), std::uint64_t{1},
        "wire v2 delta count is wrong");
    require_equal(
        throwdb::detail::get_u64(encoded, 112), expected_wal_offset,
        "wire v2 does not place deltas before WAL entries");
    require_equal(
        throwdb::detail::get_u64(encoded, 120), expected_size,
        "wire v2 file size is wrong");
    require_equal(
        throwdb::detail::decode_mutation_catalog(encoded, manifest), catalog,
        "delta catalog roundtrip changed fields");

    const MutationCatalog two_deltas = make_two_delta_catalog(manifest);
    require_equal(
        throwdb::detail::decode_mutation_catalog(
            throwdb::detail::encode_mutation_catalog(manifest, two_deltas),
            manifest),
        two_deltas, "two-delta catalog roundtrip changed fields");

    MutationCurrent current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = encoded.size();
    current.catalog_crc32c = throwdb::detail::crc32c(encoded);
    require_equal(
        throwdb::detail::decode_mutation_current(
            throwdb::detail::encode_mutation_current(manifest, current),
            manifest),
        current, "CURRENT does not accept wire v2 catalog geometry");

    MutationCurrent invalid = current;
    ++invalid.catalog_file_size;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                throwdb::detail::encode_mutation_current(manifest, invalid));
        },
        "CURRENT accepted non-integral wire v2 geometry");
    invalid = current;
    invalid.catalog_file_size =
        throwdb::detail::kMutationCatalogHeaderSize
        + throwdb::detail::kMutationCatalogDeltaEntrySize;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                throwdb::detail::encode_mutation_current(manifest, invalid));
        },
        "CURRENT accepted wire v2 geometry without a WAL entry");
    invalid = current;
    invalid.catalog_file_size =
        throwdb::detail::kMutationCatalogHeaderSize
        + throwdb::detail::kMutationCatalogMaxEntries
              * throwdb::detail::kMutationCatalogDeltaEntrySize
        + 64;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                throwdb::detail::encode_mutation_current(manifest, invalid));
        },
        "CURRENT accepted a catalog larger than the wire v2 maximum");
}

void test_delta_v2_lineage_and_resource_invariants() {
    const NativeManifest manifest = make_delta_manifest();
    const MutationCatalog valid = make_two_delta_catalog(manifest);
    const auto rejected = [&](MutationCatalog catalog, const std::string& name) {
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::encode_mutation_catalog(manifest, catalog));
            },
            name);
    };

    MutationCatalog broken = valid;
    broken.delta_entries.clear();
    rejected(broken, "materialized anchor without deltas was accepted");
    broken = valid;
    broken.materialized_sequence = 0;
    broken.terminal_wal_id = {};
    rejected(broken, "deltas without a materialized anchor were accepted");
    broken = valid;
    broken.delta_entries.front().first_sequence = 2;
    rejected(broken, "delta prefix not starting at sequence 1 was accepted");
    broken = valid;
    broken.delta_entries.front().source_predecessor_wal_id = {9, 10};
    rejected(broken, "first delta source predecessor was accepted");
    broken = valid;
    ++broken.delta_entries.back().first_sequence;
    rejected(broken, "gap between delta sequence intervals was accepted");
    broken = valid;
    broken.delta_entries.back().source_predecessor_wal_id = {9, 10};
    rejected(broken, "broken delta WAL lineage was accepted");
    broken = valid;
    --broken.materialized_sequence;
    rejected(broken, "delta prefix not reaching materialized sequence was accepted");
    broken = valid;
    broken.terminal_wal_id = {9, 10};
    rejected(broken, "delta prefix not reaching terminal WAL was accepted");
    broken = valid;
    broken.delta_entries.back().delta_id =
        broken.delta_entries.front().delta_id;
    rejected(broken, "duplicate delta UUID was accepted");
    broken = valid;
    broken.delta_entries.front().delta_id = broken.wal_entries.front().wal_id;
    rejected(broken, "delta UUID aliasing a retained WAL was accepted");
    broken = valid;
    broken.delta_entries.front().delta_id =
        broken.delta_entries.back().source_terminal_wal_id;
    rejected(broken, "delta UUID aliasing a materialized WAL was accepted");
    broken = valid;
    broken.delta_entries.front().source_wal_count = 0;
    rejected(broken, "delta with zero source WALs was accepted");
    broken = valid;
    broken.delta_entries.front().source_wal_count =
        static_cast<std::uint32_t>(
            broken.delta_entries.front().last_sequence
            - broken.delta_entries.front().first_sequence + 2);
    rejected(broken, "delta with too many source WALs was accepted");
    broken = valid;
    broken.delta_entries.front().entry_count = 0;
    rejected(broken, "empty catalog delta was accepted");
    broken = valid;
    broken.delta_entries.front().segment_id =
        broken.wal_entries.front().segment_id;
    rejected(broken, "duplicate delta/WAL segment ID was accepted");
    broken = valid;
    broken.next_segment_id = broken.delta_entries.back().segment_id;
    rejected(broken, "next segment ID not after every resource was accepted");
    broken = valid;
    ++broken.delta_entries.front().data_file_size;
    rejected(broken, "invalid delta data geometry was accepted");
    broken = valid;
    ++broken.delta_entries.front().index_file_size;
    rejected(broken, "invalid delta index geometry was accepted");
    broken = valid;
    broken.delta_entries.front().put_count =
        broken.delta_entries.front().entry_count + 1;
    rejected(broken, "delta with more PUTs than index entries was accepted");
}

void test_delta_v2_wire_corruption_is_rejected() {
    const NativeManifest manifest = make_delta_manifest();
    const MutationCatalog catalog = make_delta_catalog(manifest);
    const std::vector<std::byte> valid =
        throwdb::detail::encode_mutation_catalog(manifest, catalog);
    const auto rejected = [&](
        std::vector<std::byte> corrupted, const std::string& name) {
        reseal(corrupted);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::decode_mutation_catalog(
                    corrupted, manifest));
            },
            name);
    };

    std::vector<std::byte> corrupted = valid;
    corrupted[throwdb::detail::kMutationCatalogHeaderSize + 120] =
        std::byte{1};
    rejected(std::move(corrupted), "nonzero delta reserved byte was accepted");
    corrupted = valid;
    throwdb::detail::put_u64(
        corrupted, throwdb::detail::kMutationCatalogHeaderSize + 112, 1);
    rejected(std::move(corrupted), "unsupported delta flags were accepted");
    corrupted = valid;
    throwdb::detail::put_u32(corrupted, 152, 128);
    rejected(std::move(corrupted), "wrong delta entry size was accepted");
    corrupted = valid;
    throwdb::detail::put_u64(corrupted, 168, 0);
    rejected(std::move(corrupted), "wrong delta entries offset was accepted");
    corrupted = valid;
    throwdb::detail::put_u64(
        corrupted, 112,
        throwdb::detail::get_u64(corrupted, 112) + 64);
    rejected(std::move(corrupted), "wrong WAL entries offset was accepted");
    corrupted = valid;
    throwdb::detail::put_u64(
        corrupted, throwdb::detail::kMutationCatalogHeaderSize + 88,
        catalog.delta_entries.front().data_file_size + 1);
    rejected(std::move(corrupted), "corrupt delta geometry was accepted");
    corrupted = valid;
    throwdb::detail::put_u64(
        corrupted, 160, throwdb::detail::kMutationCatalogMaxEntries);
    rejected(std::move(corrupted), "unbounded delta entry count was accepted");
}

void test_strict_catalog_invariants() {
    const NativeManifest manifest = make_manifest();
    const MutationCatalog valid = make_rotated_catalog(
        manifest, Uuid128{3, 4}, Uuid128{1, 2});
    const auto rejected = [&](MutationCatalog catalog, const std::string& name) {
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(
                    throwdb::detail::encode_mutation_catalog(manifest, catalog));
            },
            name);
    };

    MutationCatalog broken = valid;
    broken.wal_entries.front().state = MutationWalState::Active;
    rejected(broken, "non-final active WAL was accepted");
    broken = valid;
    broken.wal_entries.back().state = MutationWalState::Frozen;
    rejected(broken, "final frozen WAL was accepted");
    broken = valid;
    broken.wal_entries.back().predecessor_wal_id = {99, 100};
    rejected(broken, "broken predecessor chain was accepted");
    broken = valid;
    ++broken.wal_entries.back().first_sequence;
    rejected(broken, "broken sequence chain was accepted");
    broken = valid;
    broken.wal_entries.back().put_count = 1;
    rejected(broken, "active PUT-count sentinel was accepted");
    broken = valid;
    broken.wal_entries.back().sealed_file_size = 4096;
    rejected(broken, "active sealed-size sentinel was accepted");
    broken = valid;
    broken.next_segment_id = broken.wal_entries.back().segment_id;
    rejected(broken, "reused next segment ID was accepted");
    broken = valid;
    broken.previous_catalog_id = {};
    rejected(broken, "generation without predecessor was accepted");
    broken = valid;
    broken.wal_entries.back().wal_id = broken.wal_entries.front().wal_id;
    rejected(broken, "duplicate WAL UUID was accepted");
    broken = valid;
    broken.wal_entries.front().last_sequence =
        std::numeric_limits<std::uint64_t>::max() - 1;
    broken.wal_entries.back().previous_sequence =
        broken.wal_entries.front().last_sequence;
    broken.wal_entries.back().first_sequence =
        std::numeric_limits<std::uint64_t>::max();
    rejected(broken, "unusable maximum commit sequence was accepted");
}

void test_checksums_reserved_bytes_and_manifest_binding() {
    const NativeManifest manifest = make_manifest();
    const MutationCatalog catalog =
        throwdb::detail::make_initial_mutation_catalog(
            manifest, Uuid128{3, 4});
    std::vector<std::byte> encoded =
        throwdb::detail::encode_mutation_catalog(manifest, catalog);
    encoded.back() ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::detail::decode_mutation_catalog(encoded, manifest));
        },
        "catalog checksum corruption was accepted");

    encoded = throwdb::detail::encode_mutation_catalog(manifest, catalog);
    encoded[152] = std::byte{1};
    reseal(encoded);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::detail::decode_mutation_catalog(encoded, manifest));
        },
        "catalog reserved bytes were accepted after reseal");

    NativeManifest wrong_manifest = manifest;
    wrong_manifest.collection_id.low ^= 1;
    encoded = throwdb::detail::encode_mutation_catalog(manifest, catalog);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::decode_mutation_catalog(
                encoded, wrong_manifest));
        },
        "catalog from another collection was accepted");

    MutationCurrent current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = encoded.size();
    current.catalog_crc32c = throwdb::detail::crc32c(encoded);
    auto encoded_current =
        throwdb::detail::encode_mutation_current(manifest, current);
    encoded_current.back() ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::decode_mutation_current(
                encoded_current, manifest));
        },
        "CURRENT checksum corruption was accepted");
    encoded_current =
        throwdb::detail::encode_mutation_current(manifest, current);
    encoded_current[96] = std::byte{1};
    reseal(encoded_current);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::decode_mutation_current(
                encoded_current, manifest));
        },
        "CURRENT reserved bytes were accepted after reseal");

    NativeManifest unmanaged = manifest;
    unmanaged.required_feature_bits = 0;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                throwdb::detail::encode_mutation_catalog(unmanaged, catalog));
        },
        "catalog was encoded for an unmanaged manifest");
}

void test_durable_publish_current_and_orphans() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    std::filesystem::create_directories(store / "catalogs");
    std::filesystem::create_directories(store / "wal");
    const NativeManifest manifest = make_manifest();
    const MutationCatalog initial =
        throwdb::detail::make_initial_mutation_catalog(
            manifest, Uuid128{1, 2});
    const MutationCurrent first_current =
        throwdb::detail::publish_mutation_catalog(store, manifest, initial);
    require(std::filesystem::exists(
                throwdb::detail::mutation_catalog_path(
                    store, initial.catalog_id)),
            "published immutable catalog is missing");
    require(std::filesystem::exists(
                throwdb::detail::mutation_current_path(store)),
            "published CURRENT is missing");
    require_equal(
        throwdb::detail::read_mutation_current(store, manifest), first_current,
        "published CURRENT does not roundtrip");
    require_equal(
        throwdb::detail::read_current_mutation_catalog(store, manifest),
        initial, "published catalog does not roundtrip through CURRENT");

    const MutationCatalog orphan = make_rotated_catalog(
        manifest, Uuid128{3, 4}, initial.catalog_id);
    const std::vector<std::byte> orphan_bytes =
        throwdb::detail::encode_mutation_catalog(manifest, orphan);
    write_bytes(
        throwdb::detail::mutation_catalog_path(store, orphan.catalog_id),
        orphan_bytes);
    MutationCurrent orphan_current;
    orphan_current.collection_id = manifest.collection_id;
    orphan_current.schema_id = manifest.schema_id;
    orphan_current.generation = orphan.generation;
    orphan_current.catalog_id = orphan.catalog_id;
    orphan_current.catalog_file_size = orphan_bytes.size();
    orphan_current.catalog_crc32c = throwdb::detail::crc32c(orphan_bytes);
    const auto orphan_current_bytes =
        throwdb::detail::encode_mutation_current(manifest, orphan_current);
    write_bytes(
        store / "CURRENT.00000000000000040000000000000003.tmp",
        orphan_current_bytes);
    require_equal(
        throwdb::detail::read_current_mutation_catalog(store, manifest),
        initial, "orphan newer catalog or temp pointer replaced CURRENT");

    const MutationCatalog rotated = make_rotated_catalog(
        manifest, Uuid128{5, 6}, initial.catalog_id);
    const MutationCurrent second_current =
        throwdb::detail::publish_mutation_catalog(store, manifest, rotated);
    require_equal(second_current.generation, std::uint64_t{2},
                  "published rotated generation is wrong");
    require_equal(
        throwdb::detail::read_current_mutation_catalog(store, manifest),
        rotated, "CURRENT did not atomically advance to rotated catalog");
    require(std::filesystem::exists(
                throwdb::detail::mutation_catalog_path(
                    store, initial.catalog_id)),
            "new publication removed the preceding immutable catalog");
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(throwdb::detail::publish_mutation_catalog(
                store, manifest, rotated));
        },
        "immutable catalog UUID was overwritten");
    require_equal(
        throwdb::detail::read_current_mutation_catalog(store, manifest),
        rotated, "failed duplicate publication changed CURRENT");

    MutationCurrent wrong_checksum = second_current;
    ++wrong_checksum.catalog_crc32c;
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::read_mutation_catalog(
                store, manifest, wrong_checksum));
        },
        "CURRENT-to-catalog whole-file checksum mismatch was accepted");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*callback)();
    };
    const std::array tests{
        TestCase{
            "initial_roundtrip_and_uuid_paths",
            test_initial_roundtrip_and_uuid_paths},
        TestCase{
            "rotated_and_materialized_chain_roundtrip",
            test_rotated_and_materialized_chain_roundtrip},
        TestCase{
            "catalog_v1_v2_compatibility",
            test_catalog_v1_v2_compatibility},
        TestCase{
            "delta_v2_roundtrip_and_geometry",
            test_delta_v2_roundtrip_and_geometry},
        TestCase{
            "delta_v2_lineage_and_resource_invariants",
            test_delta_v2_lineage_and_resource_invariants},
        TestCase{
            "delta_v2_wire_corruption_is_rejected",
            test_delta_v2_wire_corruption_is_rejected},
        TestCase{
            "strict_catalog_invariants", test_strict_catalog_invariants},
        TestCase{
            "checksums_reserved_bytes_and_manifest_binding",
            test_checksums_reserved_bytes_and_manifest_binding},
        TestCase{
            "durable_publish_current_and_orphans",
            test_durable_publish_current_and_orphans},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.callback();
            ++passed;
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
            return 1;
        }
    }
    std::cout << passed << " mutation-catalog tests passed\n";
    return 0;
}
