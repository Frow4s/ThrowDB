#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
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
#include <thread>
#include <vector>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/io_uring_pool.hpp"
#include "../src/core/task_partition.hpp"

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-test-" + std::to_string(timestamp) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

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

std::vector<std::uint16_t> make_payload(
    const std::size_t rows,
    const std::size_t dimension) {
    std::vector<std::uint16_t> result(rows * dimension);
    for (std::size_t index = 0; index < result.size(); ++index) {
        // ThrowDB must preserve every binary16 payload, including patterns that
        // represent NaNs. No floating-point comparison is involved.
        result[index] = static_cast<std::uint16_t>(
            (index * 40503U + 17U) & std::numeric_limits<std::uint16_t>::max());
    }
    return result;
}

void build_store(
    const std::filesystem::path& path,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::uint16_t> payload,
    const std::uint32_t dimension,
    const std::uint64_t shard_size_bytes) {
    throwdb::WriterOptions options;
    options.dimension = dimension;
    options.shard_size_bytes = shard_size_bytes;
    auto writer = throwdb::Writer::create(path, options);
    writer->append(ids, std::as_bytes(payload));
    writer->finish();
}

std::filesystem::path find_regular_file(
    const std::filesystem::path& root,
    const std::string_view name_fragment) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()
            && entry.path().filename().string().find(name_fragment)
                   != std::string::npos) {
            return entry.path();
        }
    }
    fail("could not find a file containing '" + std::string(name_fragment) + "'");
}

std::size_t count_regular_files(
    const std::filesystem::path& root,
    const std::string_view name_fragment) {
    std::size_t result = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()
            && entry.path().filename().string().find(name_fragment)
                   != std::string::npos) {
            ++result;
        }
    }
    return result;
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

void golden_put_u32(const std::span<std::byte> output,
                    const std::size_t offset,
                    const std::uint32_t value) {
    require(offset <= output.size() && output.size() - offset >= 4,
            "golden uint32 write exceeds buffer");
    for (std::size_t index = 0; index != 4; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8 * index));
    }
}

void golden_put_u16(const std::span<std::byte> output,
                    const std::size_t offset,
                    const std::uint16_t value) {
    require(offset <= output.size() && output.size() - offset >= 2,
            "golden uint16 write exceeds buffer");
    output[offset] = static_cast<std::byte>(value);
    output[offset + 1] = static_cast<std::byte>(value >> 8);
}

void golden_put_u64(const std::span<std::byte> output,
                    const std::size_t offset,
                    const std::uint64_t value) {
    require(offset <= output.size() && output.size() - offset >= 8,
            "golden uint64 write exceeds buffer");
    for (std::size_t index = 0; index != 8; ++index) {
        output[offset + index] = static_cast<std::byte>(value >> (8 * index));
    }
}

void golden_put_magic(const std::span<std::byte> output,
                      const std::string_view magic) {
    require(output.size() >= magic.size(), "golden magic exceeds buffer");
    std::memcpy(output.data(), magic.data(), magic.size());
}

std::vector<std::byte> read_file_bytes(const std::filesystem::path& path) {
    const std::uintmax_t raw_size = std::filesystem::file_size(path);
    require(raw_size <= std::numeric_limits<std::size_t>::max(),
            "test fixture file is too large");
    std::vector<std::byte> result(static_cast<std::size_t>(raw_size));
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "could not open fixture file");
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    require(stream.gcount() == static_cast<std::streamsize>(result.size()),
            "could not read complete fixture file");
    return result;
}

std::uint64_t golden_hash_v1(std::uint64_t id,
                             const std::uint64_t seed) noexcept {
    id += seed + 0x9e3779b97f4a7c15ULL;
    id = (id ^ (id >> 30)) * 0xbf58476d1ce4e5b9ULL;
    id = (id ^ (id >> 27)) * 0x94d049bb133111ebULL;
    return id ^ (id >> 31);
}

void test_round_trip_arbitrary_ids_and_order() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 13;
    const std::vector<std::uint64_t> ids{
        0,
        std::numeric_limits<std::uint64_t>::max(),
        42,
        9'000'000'000'000ULL,
        7,
    };
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    auto storage = throwdb::Storage::open(store_path);
    require_equal(storage->dimension(), dimension, "dimension did not round-trip");
    require_equal(storage->count(), std::uint64_t{ids.size()}, "count is wrong");
    require_equal(
        storage->record_size(),
        std::uint64_t{dimension * sizeof(std::uint16_t)},
        "record size is wrong");

    const std::array<std::size_t, 6> positions{3, 0, 4, 1, 3, 2};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    query.reserve(positions.size());
    expected.reserve(positions.size() * dimension);
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }

    require_equal(storage->get(query), expected, "query order or payload changed");
}

void test_multiple_append_batches_and_shards() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::size_t row_count = 19;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 104729U + 13U);
    }
    const auto payload = make_payload(row_count, dimension);

    throwdb::WriterOptions options;
    options.dimension = dimension;
    options.shard_size_bytes = 2U * dimension * sizeof(std::uint16_t);
    auto writer = throwdb::Writer::create(store_path, options);
    constexpr std::array<std::size_t, 4> batch_sizes{3, 1, 9, 6};
    std::size_t offset = 0;
    for (const auto batch_size : batch_sizes) {
        const auto id_batch = std::span<const std::uint64_t>(ids).subspan(
            offset, batch_size);
        const auto value_batch = std::span<const std::uint16_t>(payload).subspan(
            offset * dimension, batch_size * dimension);
        writer->append(id_batch, std::as_bytes(value_batch));
        offset += batch_size;
    }
    require_equal(writer->count(), std::uint64_t{row_count}, "writer count is wrong");
    writer->finish();

    require(
        count_regular_files(store_path, "shard-") >= 2,
        "tiny shard capacity did not produce several shard files");

    auto storage = throwdb::Storage::open(store_path);
    const std::array<std::size_t, 9> positions{18, 0, 10, 4, 17, 5, 8, 1, 13};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }
    require_equal(storage->get(query), expected, "cross-shard read is wrong");
}

void test_duplicate_id_rejects_publication() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    throwdb::WriterOptions options;
    options.dimension = 3;
    options.shard_size_bytes = 1U << 20;
    auto writer = throwdb::Writer::create(store_path, options);

    const std::array<std::uint64_t, 3> first_ids{10, 20, 30};
    const std::array<std::uint16_t, 9> first_payload{};
    writer->append(first_ids, std::as_bytes(std::span(first_payload)));
    const std::array<std::uint64_t, 2> second_ids{40, 20};
    const std::array<std::uint16_t, 6> second_payload{};
    writer->append(second_ids, std::as_bytes(std::span(second_payload)));

    bool rejected = false;
    try {
        writer->finish();
    } catch (const throwdb::DuplicateIdError& error) {
        rejected = true;
        require_equal(error.id(), std::uint64_t{20}, "wrong duplicate ID reported");
    }
    require(rejected, "duplicate ID did not fail finish()");
    writer.reset();

    bool opened = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
        opened = true;
    } catch (const throwdb::StorageError&) {
    }
    require(!opened, "failed duplicate build became readable");
}

void test_missing_id_is_atomic_for_output() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 5;
    const std::array<std::uint64_t, 3> ids{11, 22, 33};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);
    auto storage = throwdb::Storage::open(store_path);

    const std::array<std::uint64_t, 3> query{11, 999, 22};
    std::vector<std::uint16_t> output(
        query.size() * dimension, static_cast<std::uint16_t>(0xA55A));
    const auto before = output;
    bool rejected = false;
    try {
        storage->get_into(
            query.data(),
            query.size(),
            output.data(),
            output.size() * sizeof(std::uint16_t));
    } catch (const throwdb::MissingIdError& error) {
        rejected = true;
        require_equal(error.id(), std::uint64_t{999}, "wrong missing ID reported");
    }
    require(rejected, "missing ID did not fail get_into()");
    require_equal(output, before, "failed lookup partially modified output");
}

void test_append_rejects_mismatched_payload_size() {
    TempDirectory temporary;
    throwdb::WriterOptions options;
    options.dimension = 4;
    auto writer = throwdb::Writer::create(temporary.path() / "store", options);
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::array<std::uint16_t, 4> one_record{};

    bool rejected = false;
    try {
        writer->append(ids, std::as_bytes(std::span(one_record)));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "append accepted a payload with the wrong byte count");
    require_equal(writer->count(), std::uint64_t{0}, "failed append changed count");
}

void test_truncated_shard_is_reported_as_corruption() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    const std::array<std::uint64_t, 3> ids{101, 202, 303};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto shard = find_regular_file(store_path, "shard-");
    const auto original_size = std::filesystem::file_size(shard);
    require(original_size > 0, "shard is unexpectedly empty");
    std::filesystem::resize_file(shard, original_size - 1);

    bool rejected = false;
    try {
        auto storage = throwdb::Storage::open(store_path);
        static_cast<void>(storage->get(ids));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "truncated shard was silently accepted");
}

void test_corrupt_manifest_magic_is_rejected() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 2;
    const std::array<std::uint64_t, 1> ids{77};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto manifest = find_regular_file(store_path, "manifest");
    std::fstream stream(manifest, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.good(), "could not open manifest for corruption test");
    const char invalid = '\0';
    stream.write(&invalid, 1);
    stream.flush();
    stream.close();

    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "manifest with invalid magic was silently accepted");
}

void test_format_v1_writer_projects_collection_schema() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 7;
    const std::array<std::uint64_t, 2> ids{12, 34};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(
        store_path, ids, payload, dimension,
        dimension * sizeof(std::uint16_t));

    const auto manifest_path = store_path / "manifest.bin";
    const auto index_path = store_path / "index" / "index.bin";
    const auto shard_path = find_regular_file(store_path, "shard-");
    for (const auto& path : {manifest_path, index_path, shard_path}) {
        require_equal(
            read_u32_at(path, 8), std::uint32_t{1},
            "writer changed a format-v1 file version");
        require_equal(
            read_u32_at(path, 12), std::uint32_t{4096},
            "writer changed the format-v1 header size");
    }
    require_equal(
        count_regular_files(store_path / "data", "shard-"),
        std::size_t{ids.size()},
        "format-v1 compatibility test did not force multiple shards");
    for (const auto& entry :
         std::filesystem::directory_iterator(store_path / "data")) {
        require_equal(
            read_u32_at(entry.path(), 8), std::uint32_t{1},
            "writer changed a child shard format version");
    }
    require_equal(
        read_u32_at(manifest_path, 16), std::uint32_t{1},
        "writer changed the format-v1 dtype code");
    require_equal(
        read_u32_at(manifest_path, 20), dimension,
        "writer changed the format-v1 manifest dimension offset");
    require_equal(
        read_u64_at(manifest_path, 24),
        std::uint64_t{dimension * sizeof(std::uint16_t)},
        "writer changed the format-v1 record-size offset");
    require_equal(
        read_u64_at(index_path, 64), std::uint64_t{16},
        "writer changed the format-v1 numeric index slot size");
    require_equal(
        read_u32_at(shard_path, 48), dimension,
        "writer changed the format-v1 shard dimension offset");
    require_equal(
        read_u32_at(shard_path, 52), std::uint32_t{1},
        "writer changed the format-v1 shard dtype offset");

    constexpr std::uint64_t header_size = 4096;
    constexpr std::uint64_t record_size =
        dimension * sizeof(std::uint16_t);
    constexpr std::uint64_t index_capacity = 8;
    constexpr std::uint64_t hash_seed = 0x6a09e667f3bcc909ULL;
    constexpr std::uint64_t controls_offset = header_size;
    constexpr std::uint64_t slots_offset = 8192;
    constexpr std::uint64_t slot_size = 16;
    constexpr std::uint64_t index_file_size =
        slots_offset + index_capacity * slot_size;

    std::vector<std::byte> golden_manifest(header_size);
    golden_put_magic(golden_manifest, "THROWDBM");
    golden_put_u32(golden_manifest, 8, 1);
    golden_put_u32(golden_manifest, 12, header_size);
    golden_put_u32(golden_manifest, 16, 1);
    golden_put_u32(golden_manifest, 20, dimension);
    golden_put_u64(golden_manifest, 24, record_size);
    golden_put_u64(golden_manifest, 32, ids.size());
    golden_put_u64(golden_manifest, 40, record_size);
    golden_put_u64(golden_manifest, 48, 1);
    golden_put_u64(golden_manifest, 56, ids.size());
    golden_put_u64(golden_manifest, 64, index_capacity);
    golden_put_u64(golden_manifest, 72, hash_seed);
    golden_put_u64(golden_manifest, 80, controls_offset);
    golden_put_u64(golden_manifest, 88, slots_offset);
    golden_put_u64(golden_manifest, 96, index_file_size);
    require_equal(
        read_file_bytes(manifest_path), golden_manifest,
        "writer no longer emits the golden format-v1 manifest");

    std::vector<std::byte> golden_index(index_file_size);
    golden_put_magic(golden_index, "THROWDBI");
    golden_put_u32(golden_index, 8, 1);
    golden_put_u32(golden_index, 12, header_size);
    golden_put_u64(golden_index, 16, ids.size());
    golden_put_u64(golden_index, 24, index_capacity);
    golden_put_u64(golden_index, 32, hash_seed);
    golden_put_u64(golden_index, 40, controls_offset);
    golden_put_u64(golden_index, 48, slots_offset);
    golden_put_u64(golden_index, 56, index_file_size);
    golden_put_u64(golden_index, 64, slot_size);
    for (std::size_t ordinal = 0; ordinal != ids.size(); ++ordinal) {
        std::uint64_t slot =
            golden_hash_v1(ids[ordinal], hash_seed) & (index_capacity - 1);
        while (golden_index[controls_offset + slot] != std::byte{0}) {
            slot = (slot + 1) & (index_capacity - 1);
        }
        golden_index[controls_offset + slot] = std::byte{1};
        golden_put_u64(
            golden_index, slots_offset + slot * slot_size, ids[ordinal]);
        golden_put_u64(
            golden_index, slots_offset + slot * slot_size + 8, ordinal);
    }
    require_equal(
        read_file_bytes(index_path), golden_index,
        "writer no longer emits the golden format-v1 index");

    for (std::size_t shard = 0; shard != ids.size(); ++shard) {
        std::vector<std::byte> golden_shard(header_size + record_size);
        golden_put_magic(golden_shard, "THROWDBS");
        golden_put_u32(golden_shard, 8, 1);
        golden_put_u32(golden_shard, 12, header_size);
        golden_put_u64(golden_shard, 16, shard);
        golden_put_u64(golden_shard, 24, shard);
        golden_put_u64(golden_shard, 32, 1);
        golden_put_u64(golden_shard, 40, record_size);
        golden_put_u32(golden_shard, 48, dimension);
        golden_put_u32(golden_shard, 52, 1);
        for (std::size_t value = 0; value != dimension; ++value) {
            golden_put_u16(
                golden_shard, header_size + value * sizeof(std::uint16_t),
                payload[shard * dimension + value]);
        }
        char shard_name[32]{};
        std::snprintf(
            shard_name, sizeof(shard_name), "shard-%06zu.dat", shard);
        require_equal(
            read_file_bytes(store_path / "data" / shard_name), golden_shard,
            "writer no longer emits a golden format-v1 shard");
    }

    const throwdb::detail::Manifest manifest =
        throwdb::detail::read_manifest(store_path);
    require(
        manifest.format_version == throwdb::detail::FormatVersion::V1,
        "v1 manifest was not dispatched as format v1");
    require(
        manifest.schema.id_type == throwdb::detail::IdType::UInt64,
        "v1 manifest did not project the uint64 ID domain");
    require(
        manifest.schema.dtype == throwdb::detail::DTypeCode::Float16,
        "v1 manifest did not project float16");
    require_equal(
        manifest.schema.dimension, dimension,
        "v1 manifest did not project dimension");
    require_equal(
        manifest.schema.item_size, std::uint32_t{2},
        "v1 manifest did not project float16 item size");
    require(
        manifest.schema.row_layout
            == throwdb::detail::RowLayout::DenseRowMajor,
        "v1 manifest did not project row-major layout");
    require(
        manifest.schema.index_kind == throwdb::detail::IndexKind::UInt64Flat,
        "v1 manifest did not project numeric index kind");
    require(
        manifest.schema.hash_algorithm
            == throwdb::detail::HashAlgorithm::SplitMix64V1,
        "v1 manifest did not project hash algorithm");
}

void test_version_dispatch_rejects_unimplemented_and_unknown_versions() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 5;
    const std::array<std::uint64_t, 2> ids{101, 202};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto manifest_path = store_path / "manifest.bin";
    const auto index_path = store_path / "index" / "index.bin";
    const auto shard_path = find_regular_file(store_path, "shard-");
    const auto open_is_rejected = [&] {
        try {
            static_cast<void>(throwdb::Storage::open(store_path));
        } catch (const throwdb::CorruptionError&) {
            return true;
        }
        return false;
    };

    write_u32_at(manifest_path, 8, 2);
    require(open_is_rejected(), "unimplemented v2 manifest was accepted");
    write_u32_at(manifest_path, 8, 1);

    write_u32_at(shard_path, 8, 2);
    require(open_is_rejected(), "unimplemented v2 shard was accepted");
    write_u32_at(shard_path, 8, 1);

    write_u32_at(index_path, 8, 2);
    require(open_is_rejected(), "unimplemented v2 index was accepted");
    write_u32_at(index_path, 8, 1);

    write_u32_at(manifest_path, 16, 2);
    require(open_is_rejected(), "corrupt v1 dtype was canonicalized as float16");
    write_u32_at(manifest_path, 16, 1);

    auto staging_header = throwdb::detail::encode_staging_header(3);
    throwdb::detail::put_u32(staging_header, 8, 2);
    bool staging_rejected = false;
    try {
        static_cast<void>(
            throwdb::detail::decode_staging_header(staging_header));
    } catch (const throwdb::CorruptionError&) {
        staging_rejected = true;
    }
    require(staging_rejected, "unimplemented v2 staging header was accepted");

    write_u32_at(manifest_path, 8, 99);
    write_u32_at(manifest_path, 20, 0);
    std::string unknown_version_error;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
    } catch (const throwdb::CorruptionError& error) {
        unknown_version_error = error.what();
    }
    require(
        unknown_version_error.find("99") != std::string::npos,
        "unknown version was not rejected before v1 field decoding");
}

void test_truncated_index_is_reported_as_corruption() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    const std::array<std::uint64_t, 3> ids{41, 42, 43};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    const auto index = find_regular_file(store_path, "index.bin");
    const auto original_size = std::filesystem::file_size(index);
    require(original_size > 4096, "index is unexpectedly small");
    std::filesystem::resize_file(index, original_size - 8);

    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "truncated index was silently accepted");
}

void test_internal_io_worker_counts_with_repeated_cross_shard_ids() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 17;
    constexpr std::size_t row_count = 67;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 1'000'003ULL + 97ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 3 * record_size);

    const std::array<std::size_t, 16> positions{
        66, 0, 31, 4, 65, 5, 17, 1, 48, 31, 2, 64, 17, 9, 40, 0};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    for (const auto position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }

    for (const std::uint32_t workers : {0U, 1U, 2U, 4U, 8U}) {
        throwdb::ReaderOptions options;
        options.max_read_range_bytes = record_size;
        options.io_workers = workers;
        // An explicit queue smaller than some worker counts exercises
        // submit-side backpressure. Zero workers is the sequential baseline.
        options.io_queue_capacity = workers == 0 ? 0 : 2;
        options.io_ranges_per_task = 8;
        auto storage = throwdb::Storage::open(store_path, options);
        require_equal(
            storage->get(query), expected,
            "I/O worker count changed cross-shard query results");
    }

    for (const std::uint32_t ranges_per_task : {1U, 4U, 8U, 16U}) {
        throwdb::ReaderOptions options;
        options.max_read_range_bytes = record_size;
        options.io_workers = 4;
        options.io_queue_capacity = 2;
        options.io_ranges_per_task = ranges_per_task;
        auto storage = throwdb::Storage::open(store_path, options);
        require_equal(
            storage->get(query), expected,
            "ranges-per-task changed repeated cross-shard query results");
    }
}

void test_chunked_executor_adapts_to_small_batches() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 11;
    constexpr std::size_t row_count = 37;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 999'983ULL + 53ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 2 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    // One row exercises the public result of the direct fast path. The other
    // sizes straddle both the task-width and worker-count partition boundaries.
    for (const std::size_t query_size : {1U, 2U, 3U, 4U, 7U, 8U, 9U, 15U, 16U, 17U}) {
        std::vector<std::uint64_t> query;
        std::vector<std::uint16_t> expected;
        query.reserve(query_size);
        expected.reserve(query_size * dimension);
        for (std::size_t index = 0; index != query_size; ++index) {
            const auto position = (index * 5 + 3) % row_count;
            query.push_back(ids[position]);
            const auto begin = payload.begin()
                               + static_cast<std::ptrdiff_t>(position * dimension);
            expected.insert(expected.end(), begin, begin + dimension);
        }
        require_equal(
            storage->get(query), expected,
            "adaptive chunking changed a small-batch result");
    }
}

void test_range_task_partition_is_balanced_and_bounded() {
    const auto verify = [](
                            const std::size_t range_count,
                            const std::size_t worker_count,
                            const std::size_t max_ranges_per_task,
                            const std::size_t expected_task_count) {
        const auto partition = throwdb::detail::plan_range_tasks(
            range_count, worker_count, max_ranges_per_task);
        require_equal(
            partition.task_count, expected_task_count,
            "range partition produced an unexpected task count");

        std::size_t next_range = 0;
        std::size_t smallest_task = range_count;
        std::size_t largest_task = 0;
        for (std::size_t task = 0; task != partition.task_count; ++task) {
            const auto bounds = partition.bounds(task);
            require_equal(
                bounds.begin, next_range,
                "range partition left a gap or overlap");
            require(bounds.end > bounds.begin, "range partition created an empty task");
            const std::size_t task_size = bounds.end - bounds.begin;
            require(
                task_size <= max_ranges_per_task,
                "range partition exceeded io_ranges_per_task");
            smallest_task = std::min(smallest_task, task_size);
            largest_task = std::max(largest_task, task_size);
            next_range = bounds.end;
        }
        require_equal(next_range, range_count, "range partition lost ranges");
        require(
            largest_task - smallest_task <= 1,
            "range partition was not balanced");
        require(
            partition.task_count >= std::min(range_count, worker_count),
            "range partition did not expose all useful workers");
    };

    verify(17, 16, 8, 16);
    verify(100, 16, 8, 16);
    verify(160, 16, 8, 20);
    verify(7, 16, 8, 7);
    verify(100, 1, 8, 13);
    verify(32, 4, 1, 32);
}

void test_simultaneous_get_calls_share_internal_executor() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 16;
    constexpr std::size_t row_count = 64;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 65'537ULL + 11ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 4 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 2;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    constexpr std::size_t client_count = 8;
    constexpr std::size_t calls_per_client = 20;
    std::array<std::exception_ptr, client_count> errors{};
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> clients;
    clients.reserve(client_count);
    for (std::size_t client = 0; client != client_count; ++client) {
        clients.emplace_back([&, client] {
            try {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (std::size_t call = 0; call != calls_per_client; ++call) {
                    const std::array<std::size_t, 8> positions{
                        (client + call) % row_count,
                        (63 - client + call) % row_count,
                        (client * 7 + call * 3) % row_count,
                        (client + call) % row_count,
                        (32 + client + call) % row_count,
                        (5 + client * 11 + call) % row_count,
                        (48 + call) % row_count,
                        (client * 7 + call * 3) % row_count,
                    };
                    std::vector<std::uint64_t> query;
                    std::vector<std::uint16_t> expected;
                    for (const auto position : positions) {
                        query.push_back(ids[position]);
                        const auto begin = payload.begin()
                                           + static_cast<std::ptrdiff_t>(
                                               position * dimension);
                        expected.insert(expected.end(), begin, begin + dimension);
                    }
                    require_equal(
                        storage->get(query), expected,
                        "simultaneous get returned incorrect rows");
                }
            } catch (...) {
                errors[client] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != client_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& client : clients) {
        client.join();
    }
    for (const auto& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }
}

void test_parallel_read_error_is_propagated_and_executor_survives() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::size_t row_count = 12;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(500 + row * 13);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 2 * record_size);

    throwdb::ReaderOptions options;
    options.max_read_range_bytes = record_size;
    options.io_workers = 4;
    options.io_queue_capacity = 1;
    options.io_ranges_per_task = 8;
    auto storage = throwdb::Storage::open(store_path, options);

    const auto last_shard = store_path / "data" / "shard-000005.dat";
    std::filesystem::resize_file(last_shard, 4096);
    bool rejected = false;
    try {
        static_cast<void>(storage->get(ids));
    } catch (const throwdb::CorruptionError&) {
        rejected = true;
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "parallel shard read error was not propagated");

    // Two rows in different healthy shards cannot coalesce, so the recovery
    // read verifies that the coordinator can issue multiple healthy ranges
    // after a task error.
    const std::array<std::uint64_t, 2> healthy_query{ids[0], ids[2]};
    const auto healthy = storage->get(healthy_query);
    std::vector<std::uint16_t> healthy_expected;
    healthy_expected.insert(
        healthy_expected.end(), payload.begin(), payload.begin() + dimension);
    healthy_expected.insert(
        healthy_expected.end(), payload.begin() + 2 * dimension,
        payload.begin() + 3 * dimension);
    require_equal(
        healthy,
        healthy_expected,
        "executor was unusable after a task error");
}

void test_invalid_io_executor_options_are_rejected() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 2;
    const std::array<std::uint64_t, 1> ids{1};
    const auto payload = make_payload(ids.size(), dimension);
    build_store(store_path, ids, payload, dimension, 1U << 20);

    throwdb::ReaderOptions options;
    options.io_workers = 0;
    options.io_queue_capacity = 1;
    bool rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "queue capacity without I/O workers was accepted");

    options = {};
    options.io_ranges_per_task = 0;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "zero ranges per executor task was accepted");

    options = {};
    options.io_ranges_per_task = (1U << 20) + 1U;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "excessive ranges per executor task was accepted");

    options = {};
    options.io_uring_queue_depth = 0;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "zero io_uring queue depth was accepted");

    options = {};
    options.io_uring_queue_depth = 4097;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "excessive io_uring queue depth was accepted");

    options = {};
    options.io_uring_workers = 0;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "zero io_uring workers was accepted");

    options = {};
    options.io_uring_workers = 65;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "excessive io_uring workers was accepted");

    options = {};
    options.io_backend = throwdb::IoBackend::IoUring;
    options.io_uring_workers = 2;
    options.io_uring_queue_depth = 1;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "io_uring accepted fewer queue entries than workers");

    options = {};
    options.io_backend = throwdb::IoBackend::IoUring;
    options.io_workers = 1;
    rejected = false;
    try {
        static_cast<void>(throwdb::Storage::open(store_path, options));
    } catch (const throwdb::InvalidArgumentError&) {
        rejected = true;
    }
    require(rejected, "io_uring accepted thread-pool workers");
}

void test_io_uring_backend_round_trip_when_runtime_allows_it() {
    TempDirectory temporary;
    const auto store_path = temporary.path() / "store";
    constexpr std::uint32_t dimension = 9;
    constexpr std::size_t row_count = 31;
    const std::uint64_t record_size = dimension * sizeof(std::uint16_t);
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t row = 0; row != row_count; ++row) {
        ids[row] = static_cast<std::uint64_t>(row * 1'000'003ULL + 71ULL);
    }
    const auto payload = make_payload(row_count, dimension);
    build_store(store_path, ids, payload, dimension, 3 * record_size);

    const std::array<std::size_t, 10> positions{
        30, 0, 14, 3, 29, 14, 1, 20, 0, 27};
    std::vector<std::uint64_t> query;
    std::vector<std::uint16_t> expected;
    for (const std::size_t position : positions) {
        query.push_back(ids[position]);
        const auto begin = payload.begin()
                           + static_cast<std::ptrdiff_t>(position * dimension);
        expected.insert(expected.end(), begin, begin + dimension);
    }
    std::unique_ptr<throwdb::Storage> storage;
    struct RuntimeConfiguration {
        std::uint32_t workers;
        std::uint32_t queue_depth;
    };
    const std::array<RuntimeConfiguration, 6> configurations{{
        {1, 1}, {1, 2}, {1, 4}, {2, 2}, {2, 3}, {2, 4}}};
    for (const RuntimeConfiguration configuration : configurations) {
        throwdb::ReaderOptions options;
        options.max_read_range_bytes = record_size;
        options.io_backend = throwdb::IoBackend::IoUring;
        options.io_uring_workers = configuration.workers;
        options.io_uring_queue_depth = configuration.queue_depth;
        try {
            storage = throwdb::Storage::open(store_path, options);
        } catch (const throwdb::StorageError& error) {
#if defined(__linux__)
            const std::string message = error.what();
            if (message.find("(errno=1)") == std::string::npos
                && message.find("(errno=13)") == std::string::npos
                && message.find("(errno=38)") == std::string::npos) {
                throw;
            }
#endif
            // Non-Linux builds and Linux CI restricted by EPERM, EACCES, or
            // ENOSYS skip the runtime path. Other Linux setup/mmap failures
            // are implementation failures and must not become false greens.
            return;
        }
        require_equal(
            storage->get(query), expected,
            "io_uring changed query order or embedding bits");
    }

    constexpr std::size_t client_count = 8;
    std::array<std::exception_ptr, client_count> errors{};
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> clients;
    for (std::size_t client = 0; client != client_count; ++client) {
        clients.emplace_back([&, client] {
            try {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (std::size_t call = 0; call != 20; ++call) {
                    require_equal(
                        storage->get(query), expected,
                        "concurrent io_uring caller returned wrong data");
                }
            } catch (...) {
                errors[client] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != client_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread& client : clients) {
        client.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }

    const auto last_shard = store_path / "data" / "shard-000010.dat";
    std::filesystem::resize_file(last_shard, 4096);
    bool rejected = false;
    const std::array<std::uint64_t, 2> broken_query{ids[30], ids[0]};
    try {
        static_cast<void>(storage->get(broken_query));
    } catch (const throwdb::StorageError&) {
        rejected = true;
    }
    require(rejected, "io_uring silently accepted an unexpected shard EOF");

    const std::array<std::uint64_t, 2> healthy_query{ids[0], ids[3]};
    std::vector<std::uint16_t> healthy_expected;
    healthy_expected.insert(
        healthy_expected.end(), payload.begin(), payload.begin() + dimension);
    healthy_expected.insert(
        healthy_expected.end(), payload.begin() + 3 * dimension,
        payload.begin() + 4 * dimension);
    require_equal(
        storage->get(healthy_query), healthy_expected,
        "io_uring ring was unusable after a read error");
}

void test_io_uring_pool_planning_is_depth_aware() {
    require_equal(
        throwdb::detail::split_io_uring_queue_depth(2, 32),
        std::vector<std::uint32_t>{16, 16},
        "even io_uring queue depth split changed");
    require_equal(
        throwdb::detail::split_io_uring_queue_depth(2, 3),
        std::vector<std::uint32_t>{2, 1},
        "odd io_uring queue depth split changed");

    std::array<std::byte, 1> placeholder{};
    std::vector<throwdb::detail::IoUringReadOperation> operations(6);
    for (auto& operation : operations) {
        operation.buffer = placeholder.data();
        operation.size = 2048;
    }
    const std::array<std::uint32_t, 2> even_depths{16, 16};
    require_equal(
        throwdb::detail::plan_io_uring_lanes(operations, even_depths),
        std::vector<std::size_t>{0, 1, 0, 1, 0, 1},
        "equal-depth io_uring lanes were not balanced");

    operations.resize(3);
    const std::array<std::uint32_t, 2> uneven_depths{2, 1};
    require_equal(
        throwdb::detail::plan_io_uring_lanes(operations, uneven_depths),
        std::vector<std::size_t>{0, 1, 0},
        "io_uring lane planner ignored unequal queue depths");
}

int interrupted_close_calls = 0;
int interrupted_close_replacement = -1;

int interrupted_descriptor_close(const int fd) noexcept {
    ++interrupted_close_calls;
    if (interrupted_close_calls == 1) {
#if defined(__linux__)
        // Model Linux releasing the fd before reporting an interrupted close,
        // followed by another thread opening a file under the reused number.
        static_cast<void>(::close(fd));
        interrupted_close_replacement = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
#endif
        errno = EINTR;
        return -1;
    }
    return ::close(fd);
}

void test_file_descriptor_interrupted_close_ownership() {
    const int original = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    require(original >= 0, "cannot open descriptor-lifecycle fixture");
    throwdb::detail::FileDescriptor descriptor(original);
    interrupted_close_calls = 0;
    interrupted_close_replacement = -1;
    throwdb::detail::set_file_descriptor_close_test_hook(
        interrupted_descriptor_close);
    descriptor.reset();
    throwdb::detail::set_file_descriptor_close_test_hook(nullptr);
    require_equal(descriptor.get(), -1, "reset retained ownership of closed fd");
#if defined(__linux__)
    const int replacement = interrupted_close_replacement;
    const bool replacement_open =
        replacement >= 0 && ::fcntl(replacement, F_GETFD) >= 0;
    if (replacement_open) {
        static_cast<void>(::close(replacement));
    }
    require_equal(replacement, original, "fixture did not reuse the closed fd");
    require_equal(interrupted_close_calls, 1,
                  "Linux close(EINTR) retried a released descriptor");
    require(replacement_open, "close retry closed an unrelated reused fd");
#else
    require_equal(interrupted_close_calls, 2,
                  "non-Linux interrupted-close retry behavior changed");
    require(::fcntl(original, F_GETFD) < 0 && errno == EBADF,
            "non-Linux interrupted close leaked its descriptor");
#endif
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 20> tests{{
        {"file_descriptor_interrupted_close_ownership",
         test_file_descriptor_interrupted_close_ownership},
        {"round_trip_arbitrary_ids_and_order", test_round_trip_arbitrary_ids_and_order},
        {"multiple_append_batches_and_shards", test_multiple_append_batches_and_shards},
        {"duplicate_id_rejects_publication", test_duplicate_id_rejects_publication},
        {"missing_id_is_atomic_for_output", test_missing_id_is_atomic_for_output},
        {"append_rejects_mismatched_payload_size", test_append_rejects_mismatched_payload_size},
        {"truncated_shard_is_reported_as_corruption", test_truncated_shard_is_reported_as_corruption},
        {"corrupt_manifest_magic_is_rejected", test_corrupt_manifest_magic_is_rejected},
        {"format_v1_writer_projects_collection_schema", test_format_v1_writer_projects_collection_schema},
        {"version_dispatch_rejects_unimplemented_and_unknown_versions", test_version_dispatch_rejects_unimplemented_and_unknown_versions},
        {"truncated_index_is_reported_as_corruption", test_truncated_index_is_reported_as_corruption},
        {"internal_io_worker_counts_with_repeated_cross_shard_ids", test_internal_io_worker_counts_with_repeated_cross_shard_ids},
        {"chunked_executor_adapts_to_small_batches", test_chunked_executor_adapts_to_small_batches},
        {"range_task_partition_is_balanced_and_bounded", test_range_task_partition_is_balanced_and_bounded},
        {"simultaneous_get_calls_share_internal_executor", test_simultaneous_get_calls_share_internal_executor},
        {"parallel_read_error_is_propagated_and_executor_survives", test_parallel_read_error_is_propagated_and_executor_survives},
        {"invalid_io_executor_options_are_rejected", test_invalid_io_executor_options_are_rejected},
        {"io_uring_pool_planning_is_depth_aware", test_io_uring_pool_planning_is_depth_aware},
        {"io_uring_backend_round_trip_when_runtime_allows_it", test_io_uring_backend_round_trip_when_runtime_allows_it},
        {"empty_query_is_valid", [] {
             TempDirectory temporary;
             constexpr std::uint32_t dimension = 3;
             const std::array<std::uint64_t, 1> ids{1};
             const auto payload = make_payload(ids.size(), dimension);
             build_store(temporary.path() / "store", ids, payload, dimension, 1U << 20);
             auto storage = throwdb::Storage::open(temporary.path() / "store");
             const std::span<const std::uint64_t> empty;
             require(storage->get(empty).empty(), "empty query returned data");
         }},
    }};

    std::size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
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
