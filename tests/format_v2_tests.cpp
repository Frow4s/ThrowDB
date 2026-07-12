#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"
#include "../src/core/internal.hpp"

namespace {

constexpr std::uint64_t kHeaderSize = 4096;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-v2-test-" + std::to_string(timestamp) + "-"
                   + std::to_string(sequence++));
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
void require_equal(const Actual& actual, const Expected& expected,
                   const std::string& message) {
    if (!(actual == expected)) {
        fail(message);
    }
}

std::uint32_t read_u32_at(const std::filesystem::path& path,
                          const std::uint64_t offset) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "could not open file for uint32 read");
    stream.seekg(static_cast<std::streamoff>(offset));
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    require(stream.gcount() == static_cast<std::streamsize>(bytes.size()),
            "could not read serialized uint32");
    std::uint32_t result = 0;
    for (std::size_t index = 0; index != bytes.size(); ++index) {
        result |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
    }
    return result;
}

std::uint64_t read_u64_at(const std::filesystem::path& path,
                          const std::uint64_t offset) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "could not open file for uint64 read");
    stream.seekg(static_cast<std::streamoff>(offset));
    std::array<unsigned char, 8> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    require(stream.gcount() == static_cast<std::streamsize>(bytes.size()),
            "could not read serialized uint64");
    std::uint64_t result = 0;
    for (std::size_t index = 0; index != bytes.size(); ++index) {
        result |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
    }
    return result;
}

void write_u32_at(const std::filesystem::path& path,
                  const std::uint64_t offset,
                  const std::uint32_t value) {
    std::fstream stream(
        path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "could not open file for uint32 write");
    stream.seekp(static_cast<std::streamoff>(offset));
    std::array<unsigned char, 4> bytes{};
    for (std::size_t index = 0; index != bytes.size(); ++index) {
        bytes[index] = static_cast<unsigned char>(value >> (8 * index));
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    stream.flush();
    require(stream.good(), "could not write serialized uint32");
}

void write_u8_at(const std::filesystem::path& path,
                 const std::uint64_t offset,
                 const std::uint8_t value) {
    std::fstream stream(
        path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "could not open file for uint8 write");
    stream.seekp(static_cast<std::streamoff>(offset));
    const char byte = static_cast<char>(value);
    stream.write(&byte, 1);
    stream.flush();
    require(stream.good(), "could not write serialized uint8");
}

void write_u64_at(const std::filesystem::path& path,
                  const std::uint64_t offset,
                  const std::uint64_t value) {
    std::fstream stream(
        path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "could not open file for uint64 write");
    stream.seekp(static_cast<std::streamoff>(offset));
    std::array<unsigned char, 8> bytes{};
    for (std::size_t index = 0; index != bytes.size(); ++index) {
        bytes[index] = static_cast<unsigned char>(value >> (8 * index));
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    stream.flush();
    require(stream.good(), "could not write serialized uint64");
}

std::vector<std::uint32_t> make_float32_payload(
    const std::size_t rows, const std::size_t dimension) {
    constexpr std::array<std::uint32_t, 10> special{
        0x00000000U, 0x80000000U, 0x3f800000U, 0xbf800000U,
        0x7f800000U, 0xff800000U, 0x7fc00001U, 0x7fa12345U,
        0x00000001U, 0x807fffffU};
    std::vector<std::uint32_t> result(rows * dimension);
    for (std::size_t index = 0; index != result.size(); ++index) {
        if (index < special.size()) {
            result[index] = special[index];
        } else {
            result[index] = static_cast<std::uint32_t>(
                index * 2'654'435'761ULL + 0x9e3779b9ULL);
        }
    }
    return result;
}

std::unique_ptr<throwdb::Writer> create_float32_writer(
    const std::filesystem::path& path,
    const std::uint32_t dimension,
    const std::uint64_t shard_size_bytes) {
    throwdb::WriterOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float32;
    options.shard_size_bytes = shard_size_bytes;
    return throwdb::Writer::create(path, options);
}

void build_float32_store(
    const std::filesystem::path& path,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::uint32_t> payload,
    const std::uint32_t dimension,
    const std::uint64_t shard_size_bytes) {
    auto writer = create_float32_writer(path, dimension, shard_size_bytes);
    const std::size_t split = ids.size() / 2;
    writer->append(
        ids.first(split),
        std::as_bytes(payload.first(split * dimension)));
    writer->append(
        ids.subspan(split),
        std::as_bytes(payload.subspan(split * dimension)));
    writer->finish();
}

void require_raw_u32_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::uint32_t> expected,
    const std::string& message) {
    const auto expected_bytes = std::as_bytes(expected);
    if (actual.size() != expected_bytes.size()
        || (actual.size() != 0
            && std::memcmp(
                   actual.data(), expected_bytes.data(), actual.size()) != 0)) {
        fail(message);
    }
}

bool open_is_rejected(const std::filesystem::path& path) {
    try {
        static_cast<void>(throwdb::Storage::open(path));
    } catch (const throwdb::StorageError&) {
        return true;
    }
    return false;
}

void test_float32_v2_round_trip_is_bit_exact() {
    if constexpr (std::endian::native != std::endian::little) {
        return;
    }

    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 9;
    const std::array<std::uint64_t, 7> ids{
        0, 17, 9001, std::numeric_limits<std::uint64_t>::max(),
        42, 8'000'000'000ULL, 3};
    const auto payload = make_float32_payload(ids.size(), dimension);
    auto writer = create_float32_writer(
        store_path, dimension, 2ULL * dimension * sizeof(std::uint32_t));
    require(writer->dtype() == throwdb::DType::Float32,
            "float32 writer reports the wrong dtype");
    require_equal(writer->format_version(), std::uint32_t{2},
                  "float32 writer did not select format v2");
    require_equal(
        writer->record_size(),
        std::uint64_t{dimension * sizeof(std::uint32_t)},
        "float32 writer has the wrong record size");
    writer->append(
        std::span(ids).first(3),
        std::as_bytes(std::span(payload).first(3 * dimension)));
    writer->append(
        std::span(ids).subspan(3),
        std::as_bytes(std::span(payload).subspan(3 * dimension)));
    writer->finish();

    auto storage = throwdb::Storage::open(store_path);
    require(storage->dtype() == throwdb::DType::Float32,
            "float32 reader reports the wrong dtype");
    require_equal(storage->format_version(), std::uint32_t{2},
                  "float32 reader reports the wrong format version");
    require_equal(storage->record_size(), writer->record_size(),
                  "reader and writer record size differ");

    constexpr std::array<std::size_t, 8> positions{3, 0, 6, 2, 3, 5, 1, 4};
    std::vector<std::uint64_t> query;
    std::vector<std::uint32_t> expected;
    for (const std::size_t position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }

    const std::vector<std::byte> bytes = storage->get_bytes(query);
    require_raw_u32_equal(bytes, expected, "get_bytes changed float32 bits");
    std::vector<std::uint32_t> output(expected.size(), 0xdeadbeefU);
    storage->get_into(
        query.data(), query.size(), output.data(),
        output.size() * sizeof(std::uint32_t));
    require_equal(output, expected, "get_into changed float32 bits");

    bool typed_get_rejected = false;
    try {
        static_cast<void>(storage->get(query));
    } catch (const throwdb::InvalidArgumentError&) {
        typed_get_rejected = true;
    }
    require(typed_get_rejected,
            "legacy uint16 get() accepted a float32 collection");

    throwdb::ReaderOptions parallel_options;
    parallel_options.io_workers = 3;
    parallel_options.io_queue_capacity = 2;
    auto parallel = throwdb::Storage::open(store_path, parallel_options);
    require_raw_u32_equal(
        parallel->get_bytes(query), expected,
        "parallel reader changed float32 bits");
}

void test_v2_headers_match_the_numeric_spec() {
    if constexpr (std::endian::native != std::endian::little) {
        return;
    }

    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 5;
    const std::array<std::uint64_t, 3> ids{101, 202, 303};
    const auto payload = make_float32_payload(ids.size(), dimension);
    const std::uint64_t record_size = dimension * sizeof(std::uint32_t);
    build_float32_store(
        store_path, ids, payload, dimension, 2 * record_size);

    const auto manifest = store_path / "manifest.bin";
    const auto index = store_path / "index" / "index.bin";
    const auto shard0 = store_path / "data" / "shard-000000.dat";
    const auto shard1 = store_path / "data" / "shard-000001.dat";

    require_equal(read_u32_at(manifest, 8), std::uint32_t{2},
                  "manifest is not v2");
    require_equal(read_u32_at(manifest, 12), std::uint32_t{4096},
                  "manifest header size changed");
    require_equal(read_u32_at(manifest, 16), std::uint32_t{2},
                  "manifest float32 dtype code is wrong");
    require_equal(read_u32_at(manifest, 20), dimension,
                  "manifest dimension is wrong");
    require_equal(read_u32_at(manifest, 24), std::uint32_t{4},
                  "manifest item size is wrong");
    require_equal(read_u32_at(manifest, 28), std::uint32_t{1},
                  "manifest ID type is wrong");
    require_equal(read_u32_at(manifest, 32), std::uint32_t{1},
                  "manifest row layout is wrong");
    require_equal(read_u32_at(manifest, 36), std::uint32_t{1},
                  "manifest index kind is wrong");
    require_equal(read_u32_at(manifest, 40), std::uint32_t{1},
                  "manifest hash algorithm is wrong");
    require_equal(read_u32_at(manifest, 44), std::uint32_t{0},
                  "manifest unexpectedly enabled checksums");
    require_equal(read_u64_at(manifest, 48), record_size,
                  "manifest record size is wrong");
    require_equal(read_u64_at(manifest, 56), std::uint64_t{ids.size()},
                  "manifest record count is wrong");
    require_equal(read_u64_at(manifest, 104), std::uint64_t{0},
                  "numeric manifest has a string arena");
    require_equal(read_u64_at(manifest, 112), std::uint64_t{0},
                  "manifest has unknown required features");
    require_equal(read_u64_at(manifest, 120), std::uint64_t{0},
                  "manifest has unexpected optional features");
    const std::uint64_t snapshot_low = read_u64_at(manifest, 128);
    const std::uint64_t snapshot_high = read_u64_at(manifest, 136);
    require(snapshot_low != 0 || snapshot_high != 0,
            "format-v2 snapshot ID is zero");
    require_equal(read_u64_at(manifest, 152), std::uint64_t{0},
                  "numeric hash high word is not zero");
    require_equal(read_u64_at(manifest, 176), std::uint64_t{16},
                  "manifest index slot size is wrong");
    require_equal(read_u64_at(manifest, 184), std::uint64_t{0},
                  "numeric manifest has a string payload offset");
    require(read_u64_at(manifest, 192) != 0,
            "manifest creation timestamp is zero");
    require_equal(read_u64_at(manifest, 200), std::uint64_t{0},
                  "manifest checksum is nonzero when disabled");

    require_equal(read_u32_at(index, 8), std::uint32_t{2},
                  "index is not v2");
    require_equal(read_u64_at(index, 16), snapshot_low,
                  "index snapshot low word differs");
    require_equal(read_u64_at(index, 24), snapshot_high,
                  "index snapshot high word differs");
    require_equal(read_u32_at(index, 32), std::uint32_t{1},
                  "index ID type is wrong");
    require_equal(read_u32_at(index, 36), std::uint32_t{1},
                  "index kind is wrong");
    require_equal(read_u32_at(index, 40), std::uint32_t{1},
                  "index hash algorithm is wrong");
    require_equal(read_u32_at(index, 44), std::uint32_t{1},
                  "index control encoding is wrong");
    require_equal(read_u64_at(index, 96), std::uint64_t{16},
                  "index slot size is wrong");
    require_equal(read_u64_at(index, 112), std::uint64_t{0},
                  "numeric index references string bytes");
    require_equal(read_u64_at(index, 120), std::uint64_t{0},
                  "index checksum is nonzero when disabled");

    for (const auto& shard : {shard0, shard1}) {
        require_equal(read_u32_at(shard, 8), std::uint32_t{2},
                      "data shard is not v2");
        require_equal(read_u64_at(shard, 16), snapshot_low,
                      "shard snapshot low word differs");
        require_equal(read_u64_at(shard, 24), snapshot_high,
                      "shard snapshot high word differs");
        require_equal(read_u64_at(shard, 56), record_size,
                      "shard record size is wrong");
        require_equal(read_u32_at(shard, 64), dimension,
                      "shard dimension is wrong");
        require_equal(read_u32_at(shard, 68), std::uint32_t{2},
                      "shard dtype is wrong");
        require_equal(read_u32_at(shard, 72), std::uint32_t{1},
                      "shard row layout is wrong");
        require_equal(read_u32_at(shard, 76), std::uint32_t{0},
                      "shard unexpectedly enabled checksums");
        require_equal(read_u64_at(shard, 88), std::uint64_t{0},
                      "shard payload checksum is nonzero when disabled");
        require_equal(read_u64_at(shard, 96), std::uint64_t{0},
                      "shard header checksum is nonzero when disabled");
    }
    require_equal(read_u32_at(shard0, kHeaderSize), payload[0],
                  "float32 payload was not stored little-endian bit-exactly");
}

enum class CorruptionCase {
    ItemSize,
    DType,
    RequiredFeatures,
    OptionalFeatures,
    ManifestReserved,
    ManifestChecksumAlgorithm,
    ManifestChecksum,
    StringArenaSize,
    HashHighWord,
    SnapshotZero,
    RowLayout,
    ShardCapacityOverflow,
    ShardSnapshot,
    ShardPayloadSize,
    IndexSnapshot,
    IndexControlEncoding,
    IndexReserved,
    ShardVersion,
    IndexVersion,
};

void test_v2_schema_snapshot_and_reserved_corruption_is_rejected() {
    if constexpr (std::endian::native != std::endian::little) {
        return;
    }

    constexpr std::array cases{
        CorruptionCase::ItemSize,
        CorruptionCase::DType,
        CorruptionCase::RequiredFeatures,
        CorruptionCase::OptionalFeatures,
        CorruptionCase::ManifestReserved,
        CorruptionCase::ManifestChecksumAlgorithm,
        CorruptionCase::ManifestChecksum,
        CorruptionCase::StringArenaSize,
        CorruptionCase::HashHighWord,
        CorruptionCase::SnapshotZero,
        CorruptionCase::RowLayout,
        CorruptionCase::ShardCapacityOverflow,
        CorruptionCase::ShardSnapshot,
        CorruptionCase::ShardPayloadSize,
        CorruptionCase::IndexSnapshot,
        CorruptionCase::IndexControlEncoding,
        CorruptionCase::IndexReserved,
        CorruptionCase::ShardVersion,
        CorruptionCase::IndexVersion,
    };

    for (const CorruptionCase corruption : cases) {
        TempDirectory temporary;
        const auto store_path = temporary.path() / "store";
        constexpr std::uint32_t dimension = 3;
        const std::array<std::uint64_t, 2> ids{11, 29};
        const auto payload = make_float32_payload(ids.size(), dimension);
        build_float32_store(
            store_path, ids, payload, dimension,
            dimension * sizeof(std::uint32_t));
        const auto manifest = store_path / "manifest.bin";
        const auto index = store_path / "index" / "index.bin";
        const auto shard = store_path / "data" / "shard-000000.dat";

        switch (corruption) {
        case CorruptionCase::ItemSize:
            write_u32_at(manifest, 24, 2);
            break;
        case CorruptionCase::DType:
            write_u32_at(manifest, 16, 3);
            break;
        case CorruptionCase::RequiredFeatures:
            write_u64_at(manifest, 112, 1);
            break;
        case CorruptionCase::OptionalFeatures:
            write_u64_at(manifest, 120, 1);
            break;
        case CorruptionCase::ManifestReserved:
            write_u32_at(manifest, 208, 1);
            break;
        case CorruptionCase::ManifestChecksumAlgorithm:
            write_u32_at(manifest, 44, 1);
            break;
        case CorruptionCase::ManifestChecksum:
            write_u64_at(manifest, 200, 1);
            break;
        case CorruptionCase::StringArenaSize:
            write_u64_at(manifest, 104, 1);
            break;
        case CorruptionCase::HashHighWord:
            write_u64_at(manifest, 152, 1);
            break;
        case CorruptionCase::SnapshotZero:
            write_u64_at(manifest, 128, 0);
            write_u64_at(manifest, 136, 0);
            break;
        case CorruptionCase::RowLayout:
            write_u32_at(manifest, 32, 99);
            break;
        case CorruptionCase::ShardCapacityOverflow:
            write_u64_at(
                manifest, 64, std::numeric_limits<std::uint64_t>::max());
            break;
        case CorruptionCase::ShardSnapshot:
            write_u64_at(shard, 16, read_u64_at(shard, 16) ^ 1);
            break;
        case CorruptionCase::ShardPayloadSize:
            write_u64_at(shard, 80, read_u64_at(shard, 80) + 1);
            break;
        case CorruptionCase::IndexSnapshot:
            write_u64_at(index, 16, read_u64_at(index, 16) ^ 1);
            break;
        case CorruptionCase::IndexControlEncoding:
            write_u32_at(index, 44, 2);
            break;
        case CorruptionCase::IndexReserved:
            write_u32_at(index, 128, 1);
            break;
        case CorruptionCase::ShardVersion:
            write_u32_at(shard, 8, 1);
            break;
        case CorruptionCase::IndexVersion:
            write_u32_at(index, 8, 1);
            break;
        }
        require(open_is_rejected(store_path),
                "corrupt format-v2 metadata was accepted");
    }
}

void test_float32_validation_and_missing_id_are_atomic() {
    if constexpr (std::endian::native != std::endian::little) {
        return;
    }

    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    const std::array<std::uint64_t, 3> ids{7, 8, 9};
    const auto payload = make_float32_payload(ids.size(), dimension);
    auto writer = create_float32_writer(
        store_path, dimension, 2ULL * dimension * sizeof(std::uint32_t));
    const auto bytes = std::as_bytes(std::span(payload));
    bool append_rejected = false;
    try {
        writer->append(ids, bytes.first(bytes.size() - 1));
    } catch (const throwdb::InvalidArgumentError&) {
        append_rejected = true;
    }
    require(append_rejected, "float32 writer accepted a short payload");
    require_equal(writer->count(), std::uint64_t{0},
                  "failed float32 append changed writer count");
    writer->append(ids, bytes);
    writer->finish();

    auto storage = throwdb::Storage::open(store_path);
    const std::array<std::uint64_t, 3> query{7, 999, 8};
    std::vector<std::uint32_t> output(
        query.size() * dimension, 0xa5a5a5a5U);
    const auto before = output;
    bool missing_rejected = false;
    try {
        storage->get_into(
            query.data(), query.size(), output.data(),
            output.size() * sizeof(std::uint32_t));
    } catch (const throwdb::MissingIdError& error) {
        missing_rejected = true;
        require_equal(error.id(), std::uint64_t{999},
                      "wrong missing float32 ID reported");
    }
    require(missing_rejected, "missing float32 ID was accepted");
    require_equal(output, before,
                  "missing float32 ID partially modified output");
}

void test_float16_writer_remains_format_v1() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    throwdb::WriterOptions options;
    options.dimension = 2;
    options.shard_size_bytes = 1U << 20;
    auto writer = throwdb::Writer::create(store_path, options);
    require(writer->dtype() == throwdb::DType::Float16,
            "default writer dtype changed");
    require_equal(writer->format_version(), std::uint32_t{1},
                  "float16 writer stopped selecting format v1");
    const std::array<std::uint64_t, 1> ids{1};
    const std::array<std::uint16_t, 2> payload{0x8000U, 0x7e01U};
    writer->append(ids, std::as_bytes(std::span(payload)));
    writer->finish();

    require_equal(
        read_u32_at(store_path / "manifest.bin", 8), std::uint32_t{1},
        "float16 manifest version changed");
    auto storage = throwdb::Storage::open(store_path);
    require_equal(storage->format_version(), std::uint32_t{1},
                  "float16 reader format metadata is wrong");
    require_equal(storage->get(ids), std::vector<std::uint16_t>(payload.begin(), payload.end()),
                  "format-v1 compatibility read changed bits");
}

void test_v2_float16_codec_and_empty_float32_store() {
    throwdb::detail::Manifest manifest;
    manifest.format_version = throwdb::detail::FormatVersion::V2;
    manifest.schema.id_type = throwdb::detail::IdType::UInt64;
    manifest.schema.dtype = throwdb::detail::DTypeCode::Float16;
    manifest.schema.dimension = 3;
    manifest.schema.item_size = 2;
    manifest.schema.row_layout = throwdb::detail::RowLayout::DenseRowMajor;
    manifest.schema.index_kind = throwdb::detail::IndexKind::UInt64Flat;
    manifest.schema.hash_algorithm =
        throwdb::detail::HashAlgorithm::SplitMix64V1;
    manifest.record_size = 6;
    manifest.record_count = 0;
    manifest.shard_size_bytes = 60;
    manifest.records_per_shard = 10;
    manifest.shard_count = 0;
    manifest.index_capacity = 8;
    manifest.index_control_offset = kHeaderSize;
    manifest.index_slots_offset = 8192;
    manifest.index_slot_size = 16;
    manifest.index_file_size = 8192 + 8 * 16;
    manifest.snapshot_id = throwdb::detail::SnapshotId{123, 456};

    const auto encoded_manifest =
        throwdb::detail::encode_manifest(manifest);
    const auto decoded_manifest =
        throwdb::detail::decode_manifest(encoded_manifest);
    require(
        decoded_manifest.schema.dtype
            == throwdb::detail::DTypeCode::Float16,
        "numeric v2 decoder rejected float16 schema");
    require_equal(
        decoded_manifest.schema.item_size, std::uint32_t{2},
        "numeric v2 float16 item size changed");

    throwdb::detail::ShardHeader shard;
    shard.format_version = throwdb::detail::FormatVersion::V2;
    shard.snapshot_id = manifest.snapshot_id;
    shard.record_count = 2;
    shard.record_size = 6;
    shard.dimension = 3;
    shard.dtype = throwdb::detail::DTypeCode::Float16;
    shard.row_layout = throwdb::detail::RowLayout::DenseRowMajor;
    shard.payload_size = 12;
    const auto encoded_shard = throwdb::detail::encode_shard_header(shard);
    const auto decoded_shard = throwdb::detail::decode_shard_header(
        encoded_shard, throwdb::detail::FormatVersion::V2);
    require(
        decoded_shard.dtype == throwdb::detail::DTypeCode::Float16,
        "numeric v2 shard decoder rejected float16");
    require_equal(decoded_shard.payload_size, std::uint64_t{12},
                  "numeric v2 float16 shard payload size changed");

    if constexpr (std::endian::native == std::endian::little) {
        TempDirectory temporary;
        const auto store_path = temporary.path() / "empty-float32";
        auto writer = create_float32_writer(store_path, 4, 1U << 20);
        writer->finish();
        auto storage = throwdb::Storage::open(store_path);
        require_equal(storage->count(), std::uint64_t{0},
                      "empty float32 store has records");
        require_equal(storage->format_version(), std::uint32_t{2},
                      "empty float32 store is not v2");
        require(storage->get_bytes({}).empty(),
                "empty float32 query returned bytes");
    }
}

void test_v2_index_bijection_and_probe_reachability_are_validated() {
    if constexpr (std::endian::native != std::endian::little) {
        return;
    }

    {
        TempDirectory temporary;
        const auto store_path = temporary.path() / "duplicate-ordinal";
        constexpr std::uint32_t dimension = 2;
        const std::array<std::uint64_t, 3> ids{5, 17, 29};
        const auto payload = make_float32_payload(ids.size(), dimension);
        build_float32_store(
            store_path, ids, payload, dimension, 1U << 20);

        const auto manifest = store_path / "manifest.bin";
        const auto index = store_path / "index" / "index.bin";
        const std::uint64_t capacity = read_u64_at(manifest, 88);
        const std::uint64_t controls_offset = read_u64_at(manifest, 160);
        const std::uint64_t slots_offset = read_u64_at(manifest, 168);
        std::vector<std::uint64_t> occupied;
        for (std::uint64_t slot = 0; slot != capacity; ++slot) {
            std::ifstream stream(index, std::ios::binary);
            stream.seekg(static_cast<std::streamoff>(controls_offset + slot));
            char control = 0;
            stream.read(&control, 1);
            require(stream.good(), "could not read index control byte");
            if (control != 0) {
                occupied.push_back(slot);
            }
        }
        require(occupied.size() >= 2,
                "test index does not contain two occupied slots");
        const std::uint64_t first_ordinal = read_u64_at(
            index, slots_offset + occupied[0] * 16 + 8);
        write_u64_at(
            index, slots_offset + occupied[1] * 16 + 8, first_ordinal);
        require(open_is_rejected(store_path),
                "v2 index with duplicate ordinals was accepted");
    }

    {
        TempDirectory temporary;
        const auto store_path = temporary.path() / "unreachable-probe";
        constexpr std::uint64_t capacity = 8;
        std::array<std::uint64_t, capacity> first_by_home{};
        std::array<bool, capacity> occupied_home{};
        std::array<std::uint64_t, 2> ids{};
        std::uint64_t home = 0;
        for (std::uint64_t candidate = 1; ids[1] == 0; ++candidate) {
            const std::uint64_t candidate_home =
                throwdb::detail::hash_id(
                    candidate, throwdb::detail::kHashSeed)
                & (capacity - 1);
            const auto position = static_cast<std::size_t>(candidate_home);
            if (occupied_home[position]) {
                ids = {first_by_home[position], candidate};
                home = candidate_home;
            } else {
                occupied_home[position] = true;
                first_by_home[position] = candidate;
            }
        }

        constexpr std::uint32_t dimension = 2;
        const auto payload = make_float32_payload(ids.size(), dimension);
        build_float32_store(
            store_path, ids, payload, dimension, 1U << 20);
        const auto manifest = store_path / "manifest.bin";
        const auto index = store_path / "index" / "index.bin";
        const std::uint64_t controls_offset = read_u64_at(manifest, 160);
        const std::uint64_t slots_offset = read_u64_at(manifest, 168);
        const std::uint64_t target = (home + 2) & (capacity - 1);

        const std::uint64_t moved_id =
            read_u64_at(index, slots_offset + home * 16);
        const std::uint64_t moved_ordinal =
            read_u64_at(index, slots_offset + home * 16 + 8);
        write_u8_at(index, controls_offset + home, 0);
        write_u8_at(index, controls_offset + target, 1);
        write_u64_at(index, slots_offset + target * 16, moved_id);
        write_u64_at(
            index, slots_offset + target * 16 + 8, moved_ordinal);
        require(open_is_rejected(store_path),
                "v2 index with a probe-unreachable slot was accepted");
    }
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 7> tests{{
        {"float32_v2_round_trip_is_bit_exact",
         test_float32_v2_round_trip_is_bit_exact},
        {"v2_headers_match_the_numeric_spec",
         test_v2_headers_match_the_numeric_spec},
        {"v2_schema_snapshot_and_reserved_corruption_is_rejected",
         test_v2_schema_snapshot_and_reserved_corruption_is_rejected},
        {"float32_validation_and_missing_id_are_atomic",
         test_float32_validation_and_missing_id_are_atomic},
        {"float16_writer_remains_format_v1",
         test_float16_writer_remains_format_v1},
        {"v2_float16_codec_and_empty_float32_store",
         test_v2_float16_codec_and_empty_float32_store},
        {"v2_index_bijection_and_probe_reachability_are_validated",
         test_v2_index_bijection_and_probe_reachability_are_validated},
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
