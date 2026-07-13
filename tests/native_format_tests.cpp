#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "../src/core/native_format.hpp"

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-native-format-test-"
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
    } catch (...) {
        fail(message + " (wrong exception type)");
    }
    fail(message + " (no exception)");
}

throwdb::detail::NativeManifest make_manifest(
    const throwdb::detail::DTypeCode dtype =
        throwdb::detail::DTypeCode::Float16,
    const std::uint64_t record_count = 5) {
    throwdb::detail::NativeManifest manifest;
    manifest.collection_id = {0x1111222233334444ULL, 0x5555666677778888ULL};
    manifest.schema_id = {0x9999aaaabbbbccccULL, 0xddddeeeeffff0001ULL};
    manifest.active_wal_id = {0x123456789abcdef0ULL, 0xfedcba9876543210ULL};
    manifest.schema.dtype = dtype;
    manifest.schema.dimension = 2;
    manifest.base_record_count = record_count;
    // float16: two rows/segment; float32: one row/segment.
    manifest.segment_payload_capacity = 8;
    manifest.hash_seed = 0x3141592653589793ULL;
    throwdb::detail::derive_native_manifest_geometry(manifest);
    return manifest;
}

void create_layout(const std::filesystem::path& path) {
    std::filesystem::create_directories(path / "data");
    std::filesystem::create_directories(path / "ids");
    std::filesystem::create_directories(path / "index");
    std::filesystem::create_directories(path / "wal");
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

void write_u8_at(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint8_t value) {
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open test file for byte mutation");
    stream.seekp(static_cast<std::streamoff>(offset));
    const char byte = static_cast<char>(value);
    stream.write(&byte, 1);
    stream.close();
    require(stream.good(), "cannot mutate test byte");
}

void write_u64_at(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint64_t value) {
    std::array<std::byte, 8> bytes{};
    throwdb::detail::put_u64(bytes, 0, value);
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open test file for uint64 mutation");
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "cannot mutate test uint64");
}

void write_at(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::span<const std::byte> bytes) {
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "cannot open test file for byte-range mutation");
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "cannot mutate test byte range");
}

[[nodiscard]] std::array<std::byte, throwdb::detail::kNativeHeaderSize>
read_native_header(const std::filesystem::path& path) {
    std::array<std::byte, throwdb::detail::kNativeHeaderSize> bytes{};
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open native header for reading");
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(
        stream.gcount() == static_cast<std::streamsize>(bytes.size()),
        "cannot read complete native header");
    return bytes;
}

void remove_index_body_checksum(const std::filesystem::path& index_path) {
    using namespace throwdb::detail;
    NativeIndexHeader header =
        decode_native_index_header(read_native_header(index_path));
    header.body_checksum_algorithm = 0;
    header.body_checksum = 0;
    const auto encoded = encode_native_index_header(header);
    write_at(index_path, 0, encoded);
}

void materialize_base(
    const std::filesystem::path& path,
    const throwdb::detail::NativeManifest& manifest,
    const std::span<const std::uint64_t> ids) {
    require_equal(
        ids.size(), static_cast<std::size_t>(manifest.base_record_count),
        "test IDs do not match manifest");
    create_layout(path);
    std::size_t source_row = 0;
    for (std::uint64_t raw_segment = 1;
         raw_segment <= manifest.base_segment_count; ++raw_segment) {
        const auto segment_id = static_cast<std::uint32_t>(raw_segment);
        const auto data_header =
            throwdb::detail::make_native_data_header(manifest, segment_id);
        const auto ids_header =
            throwdb::detail::make_native_ids_header(manifest, segment_id);

        std::vector<std::byte> data(
            static_cast<std::size_t>(data_header.file_size), std::byte{0});
        const auto encoded_data =
            throwdb::detail::encode_native_data_header(data_header);
        std::copy(encoded_data.begin(), encoded_data.end(), data.begin());
        write_bytes(
            throwdb::detail::native_data_segment_path(path, segment_id), data);

        std::vector<std::byte> id_bytes(
            static_cast<std::size_t>(ids_header.file_size), std::byte{0});
        const auto encoded_ids =
            throwdb::detail::encode_native_ids_header(ids_header);
        std::copy(encoded_ids.begin(), encoded_ids.end(), id_bytes.begin());
        for (std::uint64_t row = 0; row != ids_header.row_count; ++row) {
            throwdb::detail::put_u64(
                id_bytes,
                static_cast<std::size_t>(
                    ids_header.payload_offset
                    + row * throwdb::detail::kNativeIdEntrySize),
                ids[source_row++]);
        }
        write_bytes(
            throwdb::detail::native_ids_segment_path(path, segment_id),
            id_bytes);
    }
    require_equal(source_row, ids.size(), "not all test IDs were written");
}

struct BuiltStore {
    TempDirectory temporary;
    std::filesystem::path store;
    throwdb::detail::NativeManifest manifest;
    std::array<std::uint64_t, 5> ids{
        0,
        std::numeric_limits<std::uint64_t>::max(),
        42,
        7,
        8'000'000'000ULL};

    BuiltStore()
        : store(temporary.path() / "store"), manifest(make_manifest()) {
        materialize_base(store, manifest, ids);
        throwdb::detail::write_native_manifest_atomic(store, manifest);
        throwdb::detail::build_and_write_native_index_atomic(store, manifest);
    }
};

void test_manifest_is_independent_checksummed_format() {
    using namespace throwdb::detail;
    NativeManifest manifest = make_manifest();
    require_equal(manifest.record_size, std::uint64_t{4},
                  "float16 native record size is wrong");
    require_equal(manifest.records_per_segment, std::uint64_t{2},
                  "records-per-segment is wrong");
    require_equal(manifest.base_segment_count, std::uint64_t{3},
                  "base segment count is wrong");
    require_equal(manifest.active_wal_segment_id, std::uint32_t{4},
                  "active WAL segment is wrong");
    require_equal(manifest.next_segment_id, std::uint32_t{5},
                  "next segment is wrong");

    const auto encoded = encode_native_manifest(manifest);
    require(
        std::memcmp(encoded.data(), kNativeManifestMagic.data(), 8) == 0,
        "native manifest magic is wrong");
    require_equal(get_u32(encoded, 8), std::uint32_t{1},
                  "native format version is wrong");
    require_equal(get_u32(encoded, 12), std::uint32_t{4096},
                  "native header size is wrong");
    require(get_u32(encoded, kNativeHeaderChecksumOffset) != 0,
            "native manifest checksum is zero");
    const NativeManifest decoded = decode_native_manifest(encoded);
    require_equal(decoded.collection_id, manifest.collection_id,
                  "collection UUID did not round trip");
    require_equal(decoded.schema_id, manifest.schema_id,
                  "schema UUID did not round trip");
    require_equal(decoded.active_wal_id, manifest.active_wal_id,
                  "WAL UUID did not round trip");
    require_equal(decoded.index_file_size, manifest.index_file_size,
                  "index geometry did not round trip");

    auto corrupted = encoded;
    corrupted[120] ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&corrupted] { static_cast<void>(decode_native_manifest(corrupted)); },
        "manifest corruption was not caught by CRC32C");

    NativeManifest float32 = make_manifest(DTypeCode::Float32, 2);
    require_equal(float32.record_size, std::uint64_t{8},
                  "float32 native record size is wrong");
    require_equal(float32.records_per_segment, std::uint64_t{1},
                  "float32 geometry is wrong");

    NativeManifest too_many_rows = make_manifest(DTypeCode::Float16, 0);
    too_many_rows.segment_payload_capacity =
        (RowRef::kMaximumRowsPerSegment + 1) * 4;
    require_throws<throwdb::InvalidArgumentError>(
        [&too_many_rows] {
            derive_native_manifest_geometry(too_many_rows);
        },
        "more than 2^32 rows per segment were accepted");
}

void test_paths_and_atomic_manifest_are_deterministic() {
    using namespace throwdb::detail;
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    create_layout(store);
    const NativeManifest manifest = make_manifest();
    write_native_manifest_atomic(store, manifest);
    const NativeManifest reopened = read_native_manifest(store);
    require_equal(reopened.base_record_count, manifest.base_record_count,
                  "atomic manifest did not reopen");
    require_equal(
        native_manifest_path(store), store / "manifest.bin",
        "manifest path changed");
    require_equal(
        native_data_segment_path(store, 12), store / "data/segment-12.dat",
        "data segment path changed");
    require_equal(
        native_ids_segment_path(store, 12), store / "ids/segment-12.ids",
        "IDs segment path changed");
    require_equal(
        native_index_path(store), store / "index/index.bin",
        "index path changed");
    require_equal(
        native_active_wal_path(store), store / "wal/active.wal",
        "active WAL path changed");
}

void test_data_and_ids_headers_are_bound_and_exact() {
    using namespace throwdb::detail;
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    const NativeManifest manifest = make_manifest();
    const std::array<std::uint64_t, 5> ids{1, 2, 3, 4, 5};
    materialize_base(store, manifest, ids);

    for (std::uint32_t segment = 1;
         segment <= manifest.base_segment_count; ++segment) {
        const NativeDataHeader data =
            read_native_data_header(store, manifest, segment);
        const NativeIdsHeader sidecar =
            read_native_ids_header(store, manifest, segment);
        require_equal(data.row_count, sidecar.row_count,
                      "data/IDs rows differ");
        require_equal(data.row_count,
                      native_base_segment_row_count(manifest, segment),
                      "segment row geometry differs from manifest");
    }

    const auto last = native_data_segment_path(store, 3);
    std::filesystem::resize_file(last, std::filesystem::file_size(last) - 1);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_data_header(store, manifest, 3)); },
        "truncated data segment file was accepted");

    auto wrong_manifest = manifest;
    wrong_manifest.collection_id.low ^= 1;
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(read_native_ids_header(store, wrong_manifest, 1));
        },
        "IDs sidecar from another collection was accepted");
}

void test_index_builds_from_sidecars_and_round_trips_rowrefs() {
    using namespace throwdb::detail;
    BuiltStore built;
    const NativeManifest reopened_manifest = read_native_manifest(built.store);
    const IndexData index = read_native_index(built.store, reopened_manifest);
    require_equal(index.count, std::uint64_t{built.ids.size()},
                  "native index count is wrong");
    for (std::size_t ordinal = 0; ordinal != built.ids.size(); ++ordinal) {
        const auto found = native_index_lookup(index, built.ids[ordinal]);
        require(found.has_value(), "native index lost an ID");
        const std::uint64_t expected_segment =
            ordinal / built.manifest.records_per_segment + 1;
        const std::uint64_t expected_row =
            ordinal % built.manifest.records_per_segment;
        require_equal(found->segment_id().value(),
                      static_cast<std::uint32_t>(expected_segment),
                      "native index returned the wrong segment");
        require_equal(found->row_index(),
                      static_cast<std::uint32_t>(expected_row),
                      "native index returned the wrong row");
    }
    require(!native_index_lookup(index, 999'999).has_value(),
            "native index found a missing ID");

    TempDirectory duplicate_directory;
    const auto duplicate_store = duplicate_directory.path() / "store";
    const NativeManifest duplicate_manifest = make_manifest();
    const std::array<std::uint64_t, 5> duplicate_ids{1, 2, 3, 2, 5};
    materialize_base(duplicate_store, duplicate_manifest, duplicate_ids);
    require_throws<throwdb::DuplicateIdError>(
        [&] {
            static_cast<void>(
                build_native_index(duplicate_store, duplicate_manifest));
        },
        "duplicate sidecar ID was accepted");
}

void test_index_checksum_fast_path_legacy_fallback_and_forced_verify() {
    using namespace throwdb::detail;
    BuiltStore built;
    const auto index_path = native_index_path(built.store);
    const NativeIndexHeader header =
        decode_native_index_header(read_native_header(index_path));
    require_equal(
        header.body_checksum_algorithm,
        kNativeIndexBodyCrc32cAlgorithm,
        "new native index has no body checksum");
    NativeIndexHeader invalid_header = header;
    invalid_header.body_checksum_algorithm = 0;
    invalid_header.body_checksum = 1;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_native_index_header(invalid_header)); },
        "index header accepted a checksum without an algorithm");
    invalid_header = header;
    invalid_header.body_checksum_algorithm = 99;
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(encode_native_index_header(invalid_header)); },
        "index header accepted an unknown checksum algorithm");

    NativeIndexReadStats fast_stats;
    const IndexData valid = read_native_index(
        built.store, built.manifest,
        NativeIndexValidationMode::ChecksumOrFull, &fast_stats);
    require(fast_stats.body_checksum_present,
            "fast open did not discover the body checksum");
    require(fast_stats.body_checksum_verified,
            "fast open did not verify the body checksum");
    require(!fast_stats.structural_validation_performed,
            "fast open unexpectedly ran the full structural scan");

    NativeIndexReadStats full_stats;
    static_cast<void>(read_native_index(
        built.store, built.manifest,
        NativeIndexValidationMode::Full, &full_stats));
    require(full_stats.body_checksum_verified,
            "forced verification skipped the body checksum");
    require(full_stats.structural_validation_performed,
            "forced verification skipped the structural scan");

    std::size_t occupied = 0;
    while (valid.controls[occupied] == 0) {
        ++occupied;
    }
    write_u8_at(
        index_path, built.manifest.index_control_offset + occupied, 0);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(read_native_index(
                built.store, built.manifest,
                NativeIndexValidationMode::ChecksumOrFull));
        },
        "checksummed fast open accepted body corruption");

    write_native_index_atomic(built.store, built.manifest, valid);
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset
            + occupied * kNativeIndexSlotSize,
        valid.slots[occupied].id ^ 1U);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(read_native_index(
                built.store, built.manifest,
                NativeIndexValidationMode::ChecksumOrFull));
        },
        "checksummed fast open accepted slot corruption");

    write_native_index_atomic(built.store, built.manifest, valid);
    const std::uint64_t controls_end = built.manifest.index_control_offset
        + built.manifest.index_capacity;
    require(
        controls_end < built.manifest.index_slots_offset,
        "test index unexpectedly has no alignment padding");
    write_u8_at(index_path, controls_end, 1);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(read_native_index(
                built.store, built.manifest,
                NativeIndexValidationMode::ChecksumOrFull));
        },
        "fast open accepted nonzero index alignment padding");

    write_native_index_atomic(built.store, built.manifest, valid);
    remove_index_body_checksum(index_path);
    NativeIndexReadStats legacy_stats;
    static_cast<void>(read_native_index(
        built.store, built.manifest,
        NativeIndexValidationMode::ChecksumOrFull, &legacy_stats));
    require(!legacy_stats.body_checksum_present,
            "legacy index unexpectedly reports a body checksum");
    require(!legacy_stats.body_checksum_verified,
            "legacy index reports a verified absent checksum");
    require(legacy_stats.structural_validation_performed,
            "legacy index did not fall back to full validation");

    write_u8_at(
        index_path, built.manifest.index_control_offset + occupied, 2);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(read_native_index(
                built.store, built.manifest,
                NativeIndexValidationMode::ChecksumOrFull));
        },
        "legacy fallback accepted structurally invalid body corruption");
}

void test_index_reader_rejects_header_and_body_corruption() {
    using namespace throwdb::detail;
    BuiltStore built;
    const auto index_path = native_index_path(built.store);
    const IndexData valid = read_native_index(built.store, built.manifest);
    const auto restore_legacy_index = [&] {
        write_native_index_atomic(built.store, built.manifest, valid);
        remove_index_body_checksum(index_path);
    };

    write_u8_at(index_path, 24, 0xff);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "index header corruption was accepted");

    restore_legacy_index();
    std::size_t occupied = 0;
    while (valid.controls[occupied] == 0) {
        ++occupied;
    }
    std::size_t second_occupied = occupied + 1;
    while (second_occupied != valid.controls.size()
           && valid.controls[second_occupied] == 0) {
        ++second_occupied;
    }
    if (second_occupied == valid.controls.size()) {
        second_occupied = 0;
        while (second_occupied == occupied
               || valid.controls[second_occupied] == 0) {
            ++second_occupied;
        }
    }
    std::size_t empty = 0;
    while (valid.controls[empty] != 0) {
        ++empty;
    }

    write_u8_at(
        index_path, built.manifest.index_control_offset + occupied, 2);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "invalid index control byte was accepted");

    restore_legacy_index();
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset
            + occupied * kNativeIndexSlotSize + 8,
        RowRef::create(built.manifest.active_wal_segment_id, 0).raw());
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "base index RowRef into active WAL was accepted");

    restore_legacy_index();
    std::size_t last_segment_slot = 0;
    while (last_segment_slot != valid.slots.size()) {
        if (valid.controls[last_segment_slot] != 0
            && RowRef::decode(valid.slots[last_segment_slot].ordinal)
                       .segment_id().value()
                   == built.manifest.base_segment_count) {
            break;
        }
        ++last_segment_slot;
    }
    require(last_segment_slot != valid.slots.size(),
            "could not find last-segment RowRef for corruption test");
    const std::uint64_t last_segment_rows = native_base_segment_row_count(
        built.manifest,
        static_cast<std::uint32_t>(built.manifest.base_segment_count));
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset
            + last_segment_slot * kNativeIndexSlotSize + 8,
        RowRef::create(
            built.manifest.base_segment_count, last_segment_rows).raw());
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "out-of-range base segment row was accepted");

    restore_legacy_index();
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset
            + second_occupied * kNativeIndexSlotSize + 8,
        valid.slots[occupied].ordinal);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "duplicate RowRef was accepted");

    restore_legacy_index();
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset
            + second_occupied * kNativeIndexSlotSize,
        valid.slots[occupied].id);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "duplicate occupied ID was accepted");

    restore_legacy_index();
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset + empty * kNativeIndexSlotSize,
        123);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "nonzero empty index slot was accepted");

    restore_legacy_index();
    std::uint64_t unreachable_id = 0;
    bool found_unreachable = false;
    for (std::uint64_t candidate = 1; candidate != 1'000'000; ++candidate) {
        const std::uint64_t home =
            hash_id(candidate, valid.hash_seed) & (valid.capacity - 1);
        if (home != occupied && valid.controls[static_cast<std::size_t>(home)] == 0) {
            unreachable_id = candidate;
            found_unreachable = true;
            break;
        }
    }
    require(found_unreachable, "could not synthesize unreachable test ID");
    write_u64_at(
        index_path,
        built.manifest.index_slots_offset + occupied * kNativeIndexSlotSize,
        unreachable_id);
    require_throws<throwdb::CorruptionError>(
        [&] { static_cast<void>(read_native_index(built.store, built.manifest)); },
        "probe-unreachable index slot was accepted");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 6> tests{{
        {"manifest_is_independent_checksummed_format",
         test_manifest_is_independent_checksummed_format},
        {"paths_and_atomic_manifest_are_deterministic",
         test_paths_and_atomic_manifest_are_deterministic},
        {"data_and_ids_headers_are_bound_and_exact",
         test_data_and_ids_headers_are_bound_and_exact},
        {"index_builds_from_sidecars_and_round_trips_rowrefs",
         test_index_builds_from_sidecars_and_round_trips_rowrefs},
        {"index_checksum_fast_path_legacy_fallback_and_forced_verify",
         test_index_checksum_fast_path_legacy_fallback_and_forced_verify},
        {"index_reader_rejects_header_and_body_corruption",
         test_index_reader_rejects_header_and_body_corruption},
    }};

    std::size_t failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
