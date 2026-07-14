#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <sys/resource.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partitioned_index_format.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
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

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& label) {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-format7-" + label + "-"
                   + std::to_string(::getpid()) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t count,
    const std::size_t record_size) {
    std::vector<std::byte> result(count * record_size);
    for (std::size_t row = 0; row != count; ++row) {
        for (std::size_t byte = 0; byte != record_size; ++byte) {
            result[row * record_size + byte] = static_cast<std::byte>(
                (row * 37 + byte * 11 + 3) & 0xffU);
        }
    }
    return result;
}

[[nodiscard]] std::vector<std::byte> expected_rows(
    const std::span<const std::uint64_t> stored_ids,
    const std::span<const std::byte> stored_rows,
    const std::size_t record_size,
    const std::span<const std::uint64_t> requested) {
    std::vector<std::byte> result(requested.size() * record_size);
    for (std::size_t output = 0; output != requested.size(); ++output) {
        const auto position = std::find(
            stored_ids.begin(), stored_ids.end(), requested[output]);
        if (position == stored_ids.end()) {
            fail("test requested an unstored ID");
        }
        const std::size_t input = static_cast<std::size_t>(
            position - stored_ids.begin());
        std::copy_n(
            stored_rows.begin()
                + static_cast<std::ptrdiff_t>(input * record_size),
            record_size,
            result.begin()
                + static_cast<std::ptrdiff_t>(output * record_size));
    }
    return result;
}

void build_collection(
    const std::filesystem::path& path,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const throwdb::DType dtype,
    const std::uint32_t dimension,
    const std::uint64_t segment_size,
    const std::uint32_t partition_bits) {
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = dtype;
    options.segment_size_bytes = segment_size;
    options.format_version = 7;
    options.partition_bits = partition_bits;
    auto builder = throwdb::NativeCollectionBuilder::create(path, options);
    const std::size_t first = std::min<std::size_t>(2, ids.size());
    const std::size_t record_size =
        dimension * (dtype == throwdb::DType::Float16 ? 2U : 4U);
    builder->append(
        ids.first(first), embeddings.first(first * record_size));
    builder->append(
        ids.subspan(first), embeddings.subspan(first * record_size));
    builder->append({}, {});
    require(builder->format_version() == 7, "builder did not report format 7");
    builder->finish();
    builder->finish();
    require(builder->finished(), "format-7 builder did not finish");
}

[[nodiscard]] std::filesystem::path first_file_with_extension(
    const std::filesystem::path& directory,
    const std::string& extension) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == extension) {
            return entry.path();
        }
    }
    fail("test resource file was not found");
}

void flip_byte(const std::filesystem::path& path, const std::uint64_t offset) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        fail("cannot open test resource for corruption");
    }
    file.seekg(static_cast<std::streamoff>(offset));
    char value = 0;
    file.read(&value, 1);
    if (!file) {
        fail("cannot read test corruption byte");
    }
    value ^= 0x5a;
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(&value, 1);
    file.flush();
    if (!file) {
        fail("cannot write test corruption byte");
    }
}

[[nodiscard]] std::vector<std::byte> read_file(
    const std::filesystem::path& path) {
    const std::uint64_t size = std::filesystem::file_size(path);
    if (size > std::numeric_limits<std::size_t>::max()) {
        fail("test file does not fit memory");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    file.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        fail("cannot read test file");
    }
    return bytes;
}

void write_file(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) {
        fail("cannot write test file");
    }
}

[[nodiscard]] std::string uuid_token(
    const throwdb::detail::Uuid128 value) {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> token{};
    const auto put_word = [&token](const std::uint64_t word,
                                   const std::size_t begin) {
        for (std::size_t nibble = 0; nibble != 16; ++nibble) {
            token[begin + nibble] = hex[
                (word >> static_cast<unsigned>((15 - nibble) * 4)) & 0xfU];
        }
    };
    put_word(value.high, 0);
    put_word(value.low, 16);
    return std::string(token.begin(), token.end());
}

[[nodiscard]] throwdb::detail::PartitionedCatalogV3 current_catalog(
    const std::filesystem::path& root) {
    using namespace throwdb::detail;
    const PartitionedManifestV2 manifest = decode_partitioned_manifest_v2(
        read_file(root / "manifest.bin"));
    const PartitionedCurrentV1 current = decode_partitioned_current_v1(
        read_file(root / "CURRENT"), manifest);
    return decode_partitioned_catalog_v3(read_file(
        root / "catalogs"
        / ("catalog-" + uuid_token(current.catalog_id) + ".bin")));
}

[[nodiscard]] std::filesystem::path test_wal_path(
    const std::filesystem::path& root,
    const throwdb::detail::Uuid128 wal_id) {
    return root / "wal" / ("wal-" + uuid_token(wal_id) + ".wal");
}

void test_float16_end_to_end_and_read_surface() {
    TemporaryDirectory temporary("f16");
    const std::array<std::uint64_t, 6> ids{
        0, std::numeric_limits<std::uint64_t>::max(), 7, 42, 9, 1001};
    const std::vector<std::byte> embeddings = rows(ids.size(), 8);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        4, 16, 2);

    const throwdb::NativeCollectionMetadata metadata =
        throwdb::NativeCollection::inspect_metadata(temporary.path());
    require(metadata.format_version == 7, "metadata format is not 7");
    require(metadata.dimension == 4, "metadata dimension changed");
    require(metadata.dtype == throwdb::DType::Float16, "metadata dtype changed");
    require(
        metadata.layout == "uint64_dense_row_major_partitioned_rowref_v1",
        "metadata layout discriminator changed");

    auto collection = throwdb::NativeCollection::open(temporary.path());
    require(collection->format_version() == 7, "reader format is not 7");
    require(collection->count() == ids.size(), "reader count changed");
    require(collection->dimension() == 4, "reader dimension changed");
    require(collection->record_size() == 8, "reader record size changed");
    require(!collection->writable(), "format-7 reader unexpectedly writable");
    const throwdb::NativeOpenStats open_stats = collection->open_stats();
    require(
        open_stats.index_body_checksum_present
            && open_stats.index_body_checksum_verified
            && !open_stats.index_structural_validation_performed,
        "ordinary format-7 open did not use checksum-or-full semantics");

    const std::array<std::uint64_t, 6> requested{
        42, 0, 42, std::numeric_limits<std::uint64_t>::max(), 1001, 9};
    const std::vector<std::byte> expected = expected_rows(
        ids, embeddings, 8, requested);
    require(
        collection->get_bytes(requested) == expected,
        "random/repeated format-7 lookup returned wrong rows");
    std::vector<std::byte> output(expected.size());
    const throwdb::NativeReadStats stats =
        collection->get_into_profiled(requested, output);
    require(output == expected, "profiled format-7 lookup changed output");
    require(
        stats.zero_delta_fast_path && stats.base_rows == requested.size()
            && stats.mutation_rows == 0,
        "profiled format-7 lookup reported wrong source");

    std::vector<std::byte> unchanged(8, std::byte{0x6a});
    require_throws<throwdb::MissingIdError>(
        [&] {
            const std::array<std::uint64_t, 1> missing{1234567};
            collection->get_into(missing, unchanged);
        },
        "missing format-7 ID was accepted");
    require(
        std::all_of(unchanged.begin(), unchanged.end(), [](const std::byte value) {
            return value == std::byte{0x6a};
        }),
        "missing lookup modified output");
    collection->get_into({}, {});

    const throwdb::NativeIdSample sample =
        collection->sample_ids(ids.size(), 99, 1024);
    require(sample.complete, "format-7 ID sample was incomplete");
    const std::set<std::uint64_t> sampled(sample.ids.begin(), sample.ids.end());
    require(
        sampled == std::set<std::uint64_t>(ids.begin(), ids.end()),
        "format-7 ID sample was not authoritative");

    auto reopened = throwdb::NativeCollection::open(temporary.path());
    require(
        reopened->get_bytes(requested) == expected,
        "format-7 reopen changed lookup output");
    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto verified = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        verified->open_stats().index_structural_validation_performed,
        "explicit format-7 verification skipped the structural audit");
    auto writable = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    require(writable->writable(), "format-7 writable mode was not reported");
    const throwdb::NativeMutationLogStats clean_stats =
        writable->rotate_wal();
    require(
        !clean_stats.rotated && clean_stats.catalog_managed
            && !clean_stats.delta_managed && clean_stats.generation == 1,
        "empty format-7 WAL rotation was not a clean no-op");
    writable.reset();
    auto calibration = throwdb::NativeCollection::open(
        temporary.path(), {}, false, true);
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, false, true));
        },
        "two format-7 calibration handles acquired the writer lease");
    calibration.reset();
    static_cast<void>(throwdb::NativeCollection::open(
        temporary.path(), {}, false, true));
}

void test_float32_and_reader_trial() {
    TemporaryDirectory temporary("f32");
    const std::array<std::uint64_t, 4> ids{11, 5, 90, 1};
    const std::vector<std::byte> embeddings = rows(ids.size(), 12);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float32,
        3, 24, 1);
    auto collection = throwdb::NativeCollection::open(temporary.path());
    require(collection->dtype() == throwdb::DType::Float32, "float32 lost");
    throwdb::ReaderOptions options;
    options.max_read_range_bytes = 12;
    auto trial = collection->create_reader_trial(options);
    const std::array<std::uint64_t, 4> requested{90, 1, 11, 5};
    std::vector<std::byte> output(48);
    collection->get_into_with_reader(*trial, requested, output);
    require(
        output == expected_rows(ids, embeddings, 12, requested),
        "format-7 reader trial returned wrong rows");
}

void test_writable_crud_reopen_rotation_and_snapshot_isolation() {
    TemporaryDirectory temporary("crud");
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    const std::array<std::uint64_t, 4> base_ids{0, maximum, 7, 42};
    const std::vector<std::byte> base_rows = rows(base_ids.size(), 4);
    build_collection(
        temporary.path(), base_ids, base_rows, throwdb::DType::Float16,
        2, 8, 1);

    const auto bytes = [](const std::uint8_t seed, const std::size_t count) {
        std::vector<std::byte> value(count * 4);
        for (std::size_t index = 0; index != value.size(); ++index) {
            value[index] = static_cast<std::byte>(seed + index);
        }
        return value;
    };
    auto old_reader = throwdb::NativeCollection::open(temporary.path());
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    require(writer->writable(), "format-7 writer did not report writable");
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, true));
        },
        "a second format-7 writer acquired writer.lock");
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, false, true));
        },
        "a calibrator acquired the live format-7 writer lease");
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            const std::array<std::uint64_t, 1> id{99};
            old_reader->insert(id, bytes(1, 1));
        },
        "read-only format-7 handle accepted a mutation");

    const std::uint64_t initial_count = writer->count();
    require_throws<throwdb::DuplicateIdError>(
        [&] {
            const std::array<std::uint64_t, 2> ids{5, 5};
            writer->upsert(ids, bytes(10, 2));
        },
        "format-7 mutation accepted duplicate request IDs");
    require_throws<throwdb::ExistingIdError>(
        [&] {
            const std::array<std::uint64_t, 2> ids{12, 7};
            writer->insert(ids, bytes(20, 2));
        },
        "format-7 insert precondition was not atomic");
    require_throws<throwdb::MissingIdError>(
        [&] {
            const std::array<std::uint64_t, 2> ids{0, 999};
            writer->update(ids, bytes(30, 2));
        },
        "format-7 update precondition was not atomic");
    require(writer->count() == initial_count, "failed mutations changed count");
    require_throws<throwdb::MissingIdError>(
        [&] {
            const std::array<std::uint64_t, 1> id{12};
            static_cast<void>(writer->get_bytes(id));
        },
        "atomic insert failure partially published a leading ID");
    require(
        writer->get_bytes(std::array<std::uint64_t, 1>{0})
            == expected_rows(
                base_ids, base_rows, 4,
                std::array<std::uint64_t, 1>{0}),
        "atomic update failure changed its leading ID");

    const std::vector<std::byte> inserted = bytes(40, 2);
    const std::array<std::uint64_t, 2> inserted_ids{9, 10};
    writer->insert(inserted_ids, inserted);
    const std::vector<std::byte> updated_extremes = bytes(60, 2);
    const std::array<std::uint64_t, 2> extreme_ids{0, maximum};
    writer->update(extreme_ids, updated_extremes);
    const std::vector<std::byte> upserted = bytes(80, 2);
    const std::array<std::uint64_t, 2> upserted_ids{7, 11};
    writer->upsert(upserted_ids, upserted);

    const std::array<std::uint64_t, 2> erased_ids{maximum, 123456};
    require(
        writer->erase(erased_ids) == std::vector<std::uint8_t>({1, 0}),
        "format-7 delete result bitmap changed");
    require(
        writer->erase(std::array<std::uint64_t, 1>{maximum})
            == std::vector<std::uint8_t>({0}),
        "repeated format-7 delete was not idempotent");
    const std::vector<std::byte> maximum_reinserted = bytes(100, 1);
    writer->insert(
        std::array<std::uint64_t, 1>{maximum}, maximum_reinserted);

    const throwdb::NativeMutationLogStats before_rotation =
        writer->mutation_log_stats();
    require(
        before_rotation.committed_sequence == 5
            && before_rotation.frozen_wal_count == 0
            && before_rotation.active_wal_put_rows == 7,
        "format-7 active WAL stats changed before rotation");
    const throwdb::NativeMutationLogStats rotated = writer->rotate_wal();
    require(
        rotated.rotated && rotated.generation == 2
            && rotated.frozen_wal_count == 1
            && rotated.committed_sequence == 5
            && rotated.active_wal_put_rows == 0,
        "format-7 WAL rotation did not publish its successor");

    auto middle_reader = throwdb::NativeCollection::open(temporary.path());
    require(
        middle_reader->get_bytes(std::array<std::uint64_t, 1>{0})
            == std::vector<std::byte>(
                updated_extremes.begin(), updated_extremes.begin() + 4),
        "post-rotation reader missed the frozen WAL");
    require(
        writer->erase(std::array<std::uint64_t, 1>{0})
            == std::vector<std::uint8_t>({1}),
        "format-7 failed to delete ID zero");
    const std::vector<std::byte> zero_reinserted = bytes(120, 1);
    writer->upsert(std::array<std::uint64_t, 1>{0}, zero_reinserted);

    require(writer->count() == 7, "format-7 live count after CRUD changed");
    const std::array<std::uint64_t, 3> profiled_ids{42, 0, 9};
    std::vector<std::byte> profiled_output(profiled_ids.size() * 4);
    const throwdb::NativeReadStats profile = writer->get_into_profiled(
        profiled_ids, profiled_output);
    require(
        !profile.zero_delta_fast_path && profile.base_rows == 1
            && profile.mutation_rows == 2,
        "format-7 mixed read source accounting changed");

    require(
        old_reader->count() == base_ids.size()
            && old_reader->get_bytes(base_ids) == base_rows,
        "pre-mutation reader did not preserve its open snapshot");
    require_throws<throwdb::MissingIdError>(
        [&] {
            static_cast<void>(old_reader->get_bytes(
                std::array<std::uint64_t, 1>{9}));
        },
        "pre-mutation reader observed a later WAL append");
    require(
        middle_reader->get_bytes(std::array<std::uint64_t, 1>{0})
            == std::vector<std::byte>(
                updated_extremes.begin(), updated_extremes.begin() + 4),
        "reader open snapshot changed after a successor-WAL append");

    const throwdb::NativeIdSample sample = writer->sample_ids(7, 19, 1024);
    require(sample.complete, "format-7 mutation-aware sample was incomplete");
    require(
        std::set<std::uint64_t>(sample.ids.begin(), sample.ids.end())
            == std::set<std::uint64_t>({0, maximum, 7, 42, 9, 10, 11}),
        "format-7 mutation-aware sample was not authoritative");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(writer->materialize_frozen_wals()); },
        "format-7 materialization did not fail explicitly");

    writer.reset();
    auto reopened = throwdb::NativeCollection::open(temporary.path());
    require(reopened->count() == 7, "format-7 replay changed live count");
    require(
        reopened->get_bytes(std::array<std::uint64_t, 1>{0})
            == zero_reinserted,
        "format-7 replay lost ID-zero reinsertion");
    require(
        reopened->get_bytes(std::array<std::uint64_t, 1>{maximum})
            == maximum_reinserted,
        "format-7 replay lost UINT64_MAX reinsertion");
    const throwdb::NativeMutationLogStats replayed =
        reopened->mutation_log_stats();
    require(
        replayed.generation == 2 && replayed.committed_sequence == 7
            && replayed.frozen_wal_count == 1
            && replayed.active_wal_put_rows == 1,
        "format-7 reopen mutation stats changed");
}

void test_active_wal_tail_recovery_and_frozen_corruption() {
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    {
        TemporaryDirectory temporary("wal-tail");
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        const std::vector<std::byte> replacement{
            std::byte{0xa1}, std::byte{0xa2}, std::byte{0xa3},
            std::byte{0xa4}};
        {
            auto writer = throwdb::NativeCollection::open(
                temporary.path(), {}, true);
            writer->update(std::array<std::uint64_t, 1>{1}, replacement);
        }
        const throwdb::detail::PartitionedCatalogV3 catalog =
            current_catalog(temporary.path());
        const std::filesystem::path active_path = test_wal_path(
            temporary.path(), catalog.wal_entries.back().wal_id);
        const std::uint64_t committed_size =
            std::filesystem::file_size(active_path);
        {
            std::ofstream tail(
                active_path, std::ios::binary | std::ios::app);
            const std::array<char, 37> garbage{
                't', 'o', 'r', 'n', '-', 'u', 'n', 'c', 'o', 'm', 'm', 'i',
                't', 't', 'e', 'd', '-', 't', 'a', 'i', 'l'};
            tail.write(garbage.data(), garbage.size());
            tail.close();
            require(static_cast<bool>(tail), "cannot append torn WAL tail");
        }
        require(
            std::filesystem::file_size(active_path)
                == committed_size + 37,
            "test did not append its torn WAL tail");
        {
            auto reader = throwdb::NativeCollection::open(temporary.path());
            require(
                reader->get_bytes(std::array<std::uint64_t, 1>{1})
                    == replacement,
                "read-only format-7 recovery lost a committed transaction");
            require(
                std::filesystem::file_size(active_path)
                    == committed_size + 37,
                "read-only format-7 recovery modified the active WAL");
        }
        {
            auto writer = throwdb::NativeCollection::open(
                temporary.path(), {}, true);
            require(
                std::filesystem::file_size(active_path) == committed_size,
                "writable format-7 recovery did not truncate a safe tail");
            require(
                writer->get_bytes(std::array<std::uint64_t, 1>{1})
                    == replacement,
                "writable tail recovery changed committed state");
        }
    }
    {
        TemporaryDirectory temporary("frozen-corrupt");
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        {
            auto writer = throwdb::NativeCollection::open(
                temporary.path(), {}, true);
            const std::array<std::byte, 4> replacement{
                std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
            writer->update(std::array<std::uint64_t, 1>{1}, replacement);
            require(writer->rotate_wal().rotated, "test WAL did not rotate");
        }
        const throwdb::detail::PartitionedCatalogV3 catalog =
            current_catalog(temporary.path());
        require(
            catalog.wal_entries.size() == 2
                && catalog.wal_entries.front().state
                       == throwdb::detail::MutationWalState::Frozen,
            "test catalog did not publish a frozen WAL");
        const std::filesystem::path frozen_path = test_wal_path(
            temporary.path(), catalog.wal_entries.front().wal_id);
        flip_byte(frozen_path, throwdb::detail::kPartitionedHeaderSize + 9);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::open(temporary.path()));
            },
            "format-7 recovery accepted a corrupted frozen WAL");
    }
}

void test_duplicate_and_abandoned_build_are_not_published() {
    {
        TemporaryDirectory temporary("duplicate");
        const std::array<std::uint64_t, 3> ids{1, 2, 1};
        const std::vector<std::byte> embeddings = rows(ids.size(), 4);
        throwdb::NativeBuilderOptions options;
        options.dimension = 2;
        options.segment_size_bytes = 8;
        options.format_version = 7;
        options.partition_bits = 1;
        auto builder = throwdb::NativeCollectionBuilder::create(
            temporary.path(), options);
        builder->append(ids, embeddings);
        require_throws<throwdb::DuplicateIdError>(
            [&] { builder->finish(); },
            "format-7 builder accepted a duplicate ID");
        require(
            !std::filesystem::exists(temporary.path() / "manifest.bin"),
            "duplicate build published a manifest");
    }
    {
        TemporaryDirectory temporary("abandon");
        const std::array<std::uint64_t, 1> ids{7};
        const std::vector<std::byte> embeddings = rows(1, 4);
        {
            throwdb::NativeBuilderOptions options;
            options.dimension = 2;
            options.segment_size_bytes = 8;
            options.format_version = 7;
            options.partition_bits = 1;
            auto builder = throwdb::NativeCollectionBuilder::create(
                temporary.path(), options);
            builder->append(ids, embeddings);
        }
        require(
            !std::filesystem::exists(temporary.path() / "manifest.bin"),
            "abandoned build published a manifest");
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::open(temporary.path()));
            },
            "abandoned build could be opened");
    }
}

void test_snapshot_and_current_corruption_and_symlink_rejected() {
    const std::array<std::uint64_t, 3> ids{1, 4, 8};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    {
        TemporaryDirectory temporary("snapshot-corrupt");
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        const std::filesystem::path snapshot = first_file_with_extension(
            temporary.path() / "index" / "partitions", ".idx");
        flip_byte(snapshot, std::filesystem::file_size(snapshot) - 1);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::open(temporary.path()));
            },
            "snapshot body corruption was accepted");
    }
    {
        TemporaryDirectory temporary("current-corrupt");
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        flip_byte(temporary.path() / "CURRENT", 300);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::open(temporary.path()));
            },
            "CURRENT corruption was accepted");
    }
    {
        TemporaryDirectory temporary("snapshot-symlink");
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        const std::filesystem::path snapshot = first_file_with_extension(
            temporary.path() / "index" / "partitions", ".idx");
        const std::filesystem::path saved = snapshot.string() + ".saved";
        std::filesystem::rename(snapshot, saved);
        std::filesystem::create_symlink(saved.filename(), snapshot);
        require_throws<throwdb::StorageError>(
            [&] {
                static_cast<void>(
                    throwdb::NativeCollection::open(temporary.path()));
            },
            "snapshot symlink was followed");
    }
}

void test_segment_body_checksums_in_full_validation() {
    TemporaryDirectory temporary("segment-crc");
    const std::array<std::uint64_t, 2> ids{5, 6};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 8, 1);
    const std::filesystem::path data = first_file_with_extension(
        temporary.path() / "segments", ".dat");
    flip_byte(data, 4096);
    static_cast<void>(throwdb::NativeCollection::open(temporary.path()));
    throwdb::ReaderOptions options;
    options.verify_native_index = true;
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), options));
        },
        "full validation accepted segment payload corruption");
}

class RlimitGuard {
public:
    RlimitGuard() {
        if (::getrlimit(RLIMIT_NOFILE, &original_) != 0) {
            fail("getrlimit failed");
        }
    }
    ~RlimitGuard() { static_cast<void>(::setrlimit(RLIMIT_NOFILE, &original_)); }
    void lower(const rlim_t target) {
        struct rlimit value = original_;
        value.rlim_cur = std::min(original_.rlim_cur, target);
        if (::setrlimit(RLIMIT_NOFILE, &value) != 0) {
            fail("setrlimit failed");
        }
    }

private:
    struct rlimit original_ {};
};

void test_4096_partitions_open_under_low_fd_limit() {
    TemporaryDirectory temporary("rlimit");
    const std::array<std::uint64_t, 1> ids{77};
    const std::vector<std::byte> embeddings = rows(1, 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 8, 12);
    std::uint64_t snapshots = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
             temporary.path() / "index" / "partitions")) {
        snapshots += entry.path().extension() == ".idx" ? 1 : 0;
    }
    require(snapshots == 4096, "default partition topology is not 4096");

    RlimitGuard limit;
    limit.lower(48);
    auto collection = throwdb::NativeCollection::open(temporary.path());
    require(collection->get_bytes(ids) == embeddings, "low-fd lookup failed");
}

void test_empty_collection_surface() {
    TemporaryDirectory temporary("empty");
    build_collection(
        temporary.path(), {}, {}, throwdb::DType::Float16, 2, 8, 1);
    auto collection = throwdb::NativeCollection::open(temporary.path());
    require(collection->count() == 0, "empty format-7 count changed");
    require(collection->get_bytes({}).empty(), "empty get returned bytes");
    std::vector<std::byte> output;
    const throwdb::NativeReadStats stats =
        collection->get_into_profiled({}, output);
    require(
        stats.zero_delta_fast_path && stats.requested_rows == 0,
        "empty profiled get did not use the immutable fast path");
    const throwdb::NativeIdSample zero = collection->sample_ids(0, 1, 0);
    require(zero.complete && zero.ids.empty(), "zero-sized sample changed");
    const throwdb::NativeIdSample unavailable =
        collection->sample_ids(1, 1, 100);
    require(
        !unavailable.complete && unavailable.ids.empty()
            && unavailable.inspected_slots == 16,
        "empty collection sample did not terminate at snapshot capacity");
    require_throws<throwdb::MissingIdError>(
        [&] {
            const std::array<std::uint64_t, 1> missing{1};
            static_cast<void>(collection->get_bytes(missing));
        },
        "empty format-7 collection found an ID");

    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto verified = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(verified->count() == 0, "verified empty collection changed count");
}

void test_forked_builder_and_reader_lifecycle() {
    TemporaryDirectory temporary("fork");
    const std::filesystem::path builder_path = temporary.path() / "builder";
    const std::array<std::uint64_t, 1> ids{41};
    const std::vector<std::byte> embeddings = rows(1, 4);
    throwdb::NativeBuilderOptions options;
    options.dimension = 2;
    options.segment_size_bytes = 8;
    options.format_version = 7;
    options.partition_bits = 1;
    auto builder = throwdb::NativeCollectionBuilder::create(
        builder_path, options);

    pid_t child = ::fork();
    require(child >= 0, "fork failed for format-7 builder");
    if (child == 0) {
        int result = 10;
        try {
            builder->append(ids, embeddings);
        } catch (const throwdb::StorageError& error) {
            result = std::string(error.what()).find("fork")
                         != std::string::npos
                     ? 0
                     : 11;
        } catch (...) {
            result = 12;
        }
        builder.reset();
        ::_exit(result);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(
        waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "forked format-7 builder was not rejected safely");

    const std::filesystem::path writer_lock =
        builder_path / "locks" / "writer.lock";
    const int probe_fd = ::open(writer_lock.c_str(), O_RDONLY | O_CLOEXEC);
    require(probe_fd >= 0, "cannot open writer-lock probe");
    errno = 0;
    const int probe_result = ::flock(probe_fd, LOCK_EX | LOCK_NB);
    const int probe_error = errno;
    static_cast<void>(::close(probe_fd));
    require(
        probe_result < 0
            && (probe_error == EWOULDBLOCK || probe_error == EAGAIN),
        "forked child teardown released the parent builder lease");
    builder->append(ids, embeddings);
    builder->finish();

    auto reader = throwdb::NativeCollection::open(builder_path);
    child = ::fork();
    require(child >= 0, "fork failed for format-7 reader");
    if (child == 0) {
        int result = 20;
        try {
            static_cast<void>(reader->get_bytes(ids));
        } catch (const throwdb::StorageError& error) {
            result = std::string(error.what()).find("fork")
                         != std::string::npos
                     ? 0
                     : 21;
        } catch (...) {
            result = 22;
        }
        reader.reset();
        if (result == 0) {
            try {
                auto reopened = throwdb::NativeCollection::open(builder_path);
                if (reopened->get_bytes(ids) != embeddings) {
                    result = 23;
                }
            } catch (...) {
                result = 24;
            }
        }
        ::_exit(result);
    }
    status = 0;
    waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(
        waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "forked format-7 reader was not rejected/reopened safely");
    require(
        reader->get_bytes(ids) == embeddings,
        "child teardown damaged the parent format-7 reader");

    auto writer = throwdb::NativeCollection::open(
        builder_path, {}, true);
    child = ::fork();
    require(child >= 0, "fork failed for format-7 writer");
    if (child == 0) {
        int result = 30;
        try {
            const std::array<std::byte, 4> replacement{
                std::byte{7}, std::byte{8}, std::byte{9}, std::byte{10}};
            writer->upsert(ids, replacement);
        } catch (const throwdb::StorageError& error) {
            result = std::string(error.what()).find("fork")
                         != std::string::npos
                     ? 0
                     : 31;
        } catch (...) {
            result = 32;
        }
        writer.reset();
        ::_exit(result);
    }
    status = 0;
    waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(
        waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "forked format-7 writer was not rejected safely");

    const int writer_probe_fd = ::open(
        writer_lock.c_str(), O_RDONLY | O_CLOEXEC);
    require(writer_probe_fd >= 0, "cannot open writer-lock probe for writer");
    errno = 0;
    const int writer_probe_result =
        ::flock(writer_probe_fd, LOCK_EX | LOCK_NB);
    const int writer_probe_error = errno;
    static_cast<void>(::close(writer_probe_fd));
    require(
        writer_probe_result < 0
            && (writer_probe_error == EWOULDBLOCK
                || writer_probe_error == EAGAIN),
        "forked child teardown released the parent writer lease");
    const std::array<std::byte, 4> parent_replacement{
        std::byte{11}, std::byte{12}, std::byte{13}, std::byte{14}};
    writer->update(ids, parent_replacement);
    require(
        writer->get_bytes(ids)
            == std::vector<std::byte>(
                parent_replacement.begin(), parent_replacement.end()),
        "parent writer stopped working after child teardown");
}

void test_checksummed_post_initial_catalog_fails_closed() {
    TemporaryDirectory temporary("post-initial");
    const std::array<std::uint64_t, 2> ids{1, 2};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 8, 1);

    using namespace throwdb::detail;
    const PartitionedManifestV2 manifest = decode_partitioned_manifest_v2(
        read_file(temporary.path() / "manifest.bin"));
    const PartitionedCurrentV1 original_current =
        decode_partitioned_current_v1(
            read_file(temporary.path() / "CURRENT"), manifest);
    PartitionedCatalogV3 catalog = decode_partitioned_catalog_v3(read_file(
        temporary.path() / "catalogs"
        / ("catalog-" + uuid_token(original_current.catalog_id) + ".bin")));

    catalog.generation = 2;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xf001, 0xf002};
    catalog.indexed_sequence = 1;
    catalog.terminal_wal_id = Uuid128{0xe001, 0xe002};
    catalog.next_index_epoch = 2;
    MutationWalEntry& active = catalog.wal_entries.front();
    active.predecessor_wal_id = catalog.terminal_wal_id;
    active.previous_sequence = 1;
    active.first_sequence = 2;
    PartitionIndexRunDescriptor run;
    run.run_id = Uuid128{0xd001, 0xd002};
    run.epoch = 1;
    run.kind = PartitionedRunKind::Mutation;
    run.first_sequence = 1;
    run.last_sequence = 1;
    run.entry_count = 1;
    run.file_size = 8192 + kPartitionRunRecordSize;
    run.body_crc32c = 0x12345678U;
    run.covered_partitions = {std::byte{0x03}};
    run.active_partitions = run.covered_partitions;
    catalog.runs.push_back(run);
    const std::vector<std::byte> catalog_bytes =
        encode_partitioned_catalog_v3(catalog);
    write_file(
        temporary.path() / "catalogs"
            / ("catalog-" + uuid_token(catalog.catalog_id) + ".bin"),
        catalog_bytes);

    PartitionedCurrentV1 current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = catalog_bytes.size();
    current.catalog_crc32c = crc32c(catalog_bytes);
    const auto current_bytes =
        encode_partitioned_current_v1(manifest, current);
    write_file(temporary.path() / "CURRENT", current_bytes);

    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::NativeCollection::open(temporary.path()));
        },
        "read-only slice accepted a checksummed post-initial run catalog");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 11> tests{{
        {"float16_end_to_end_and_read_surface",
         test_float16_end_to_end_and_read_surface},
        {"float32_and_reader_trial", test_float32_and_reader_trial},
        {"writable_crud_reopen_rotation_and_snapshot_isolation",
         test_writable_crud_reopen_rotation_and_snapshot_isolation},
        {"active_wal_tail_recovery_and_frozen_corruption",
         test_active_wal_tail_recovery_and_frozen_corruption},
        {"duplicate_and_abandoned_build_are_not_published",
         test_duplicate_and_abandoned_build_are_not_published},
        {"snapshot_and_current_corruption_and_symlink_rejected",
         test_snapshot_and_current_corruption_and_symlink_rejected},
        {"segment_body_checksums_in_full_validation",
         test_segment_body_checksums_in_full_validation},
        {"4096_partitions_open_under_low_fd_limit",
         test_4096_partitions_open_under_low_fd_limit},
        {"empty_collection_surface", test_empty_collection_surface},
        {"forked_builder_and_reader_lifecycle",
         test_forked_builder_and_reader_lifecycle},
        {"checksummed_post_initial_catalog_fails_closed",
         test_checksummed_post_initial_catalog_fails_closed},
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
