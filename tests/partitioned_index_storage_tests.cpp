#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
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
#include "../src/core/mutation_wal.hpp"
#include "../src/core/native_collection_test_hooks.hpp"
#include "../src/core/partition_snapshot_storage.hpp"
#include "../src/core/partitioned_index_format.hpp"
#include "../src/core/partitioned_index_merge.hpp"
#include "../src/core/partitioned_index_run_storage.hpp"
#include "../src/core/partitioned_segment_storage.hpp"

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
    static constexpr char hex[] = "0123456789abcdef";
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

[[nodiscard]] throwdb::detail::PartitionedManifestV2 current_manifest(
    const std::filesystem::path& root) {
    return throwdb::detail::decode_partitioned_manifest_v2(
        read_file(root / "manifest.bin"));
}

void publish_test_catalog(
    const std::filesystem::path& root,
    const throwdb::detail::PartitionedManifestV2& manifest,
    const throwdb::detail::PartitionedCatalogV3& catalog) {
    using namespace throwdb::detail;
    const std::vector<std::byte> catalog_bytes =
        encode_partitioned_catalog_v3(catalog);
    write_file(
        root / "catalogs"
            / ("catalog-" + uuid_token(catalog.catalog_id) + ".bin"),
        catalog_bytes);
    PartitionedCurrentV1 current;
    current.collection_id = manifest.collection_id;
    current.schema_id = manifest.schema_id;
    current.generation = catalog.generation;
    current.catalog_id = catalog.catalog_id;
    current.catalog_file_size = catalog_bytes.size();
    current.catalog_crc32c = crc32c(catalog_bytes);
    write_file(root / "CURRENT", encode_partitioned_current_v1(manifest, current));
}

[[nodiscard]] std::filesystem::path test_wal_path(
    const std::filesystem::path& root,
    const throwdb::detail::Uuid128 wal_id) {
    return root / "wal" / ("wal-" + uuid_token(wal_id) + ".wal");
}

std::atomic<bool> materialization_output_entered{false};
std::atomic<bool> release_materialization_output{false};

void blocking_materialization_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation
            != NativeCollectionTestOperation::MaterializeFrozenWals
        || context.point
               != NativeCollectionTestHookPoint::LifecycleOutputDurable) {
        return;
    }
    materialization_output_entered.store(true, std::memory_order_release);
    while (!release_materialization_output.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

std::atomic<bool> fold_output_entered{false};
std::atomic<bool> release_fold_output{false};

void blocking_fold_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation != NativeCollectionTestOperation::FoldPartitions
        || context.point
               != NativeCollectionTestHookPoint::LifecycleOutputDurable) {
        return;
    }
    fold_output_entered.store(true, std::memory_order_release);
    while (!release_fold_output.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

throwdb::detail::NativeCollectionTestHookPoint crash_target_point =
    throwdb::detail::NativeCollectionTestHookPoint::
        MutationReservationDurable;
std::uint64_t crash_target_generation = 0;
throwdb::detail::NativeCollectionTestOperation crash_target_operation =
    throwdb::detail::NativeCollectionTestOperation::MaterializeFrozenWals;

void crashing_materialization_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    if (context.operation == crash_target_operation
        && context.point == crash_target_point
        && context.generation == crash_target_generation) {
        ::_exit(73);
    }
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
    require(
        metadata.collection_id_low != 0
            || metadata.collection_id_high != 0,
        "metadata collection identity is empty");
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
            && clean_stats.delta_managed && clean_stats.generation == 1,
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

void test_physical_id_scan_order_locations_and_snapshot_isolation() {
    TemporaryDirectory temporary("physical-id-scan");
    const std::array<std::uint64_t, 5> base_ids{50, 10, 40, 20, 30};
    build_collection(
        temporary.path(), base_ids, rows(base_ids.size(), 4),
        throwdb::DType::Float16, 2, 8, 1);

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    writer->update(
        std::array<std::uint64_t, 1>{10}, rows(1, 4));
    writer->insert(
        std::array<std::uint64_t, 1>{60}, rows(1, 4));
    require(
        writer->erase(std::array<std::uint64_t, 1>{40})
            == std::vector<std::uint8_t>({1}),
        "physical ID scan setup delete failed");
    writer->upsert(
        std::array<std::uint64_t, 1>{50}, rows(1, 4));

    require_throws<std::length_error>(
        [&] { static_cast<void>(writer->scan_ids(4500)); },
        "bounded physical ID scan allocated an overlay beyond its budget");
    auto old_scan = writer->scan_ids();
    const throwdb::NativeIdScanMetadata old_metadata = old_scan->metadata();
    require(
        old_metadata.live_count == 5
            && old_metadata.catalog_generation == 1
            && old_metadata.committed_sequence == 4
            && (old_metadata.collection_id_low != 0
                || old_metadata.collection_id_high != 0),
        "physical ID scan captured wrong metadata");
    std::array<std::uint64_t, 2> invalid_ids{};
    std::array<throwdb::NativeIdLocation, 1> invalid_locations{};
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                old_scan->read_into(invalid_ids, invalid_locations));
        },
        "physical ID scan accepted mismatched buffer capacities");
    std::array<std::uint64_t, 2> overlapping_outputs{};
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(old_scan->read_into(
                overlapping_outputs,
                std::span<throwdb::NativeIdLocation>(
                    reinterpret_cast<throwdb::NativeIdLocation*>(
                        overlapping_outputs.data()),
                    overlapping_outputs.size())));
        },
        "physical ID scan accepted overlapping ID and location outputs");

    // Mutate the live writer after the cursor captured its resource set and
    // overlay. The old cursor must retain both its logical contents and the
    // exact physical coordinates from sequence 4.
    writer->update(
        std::array<std::uint64_t, 1>{20}, rows(1, 4));
    require(
        writer->erase(std::array<std::uint64_t, 1>{30})
            == std::vector<std::uint8_t>({1}),
        "physical ID scan snapshot setup delete failed");
    writer->insert(
        std::array<std::uint64_t, 1>{70}, rows(1, 4));
    require(
        old_scan->metadata().committed_sequence == 4
            && old_scan->metadata().live_count == 5,
        "physical ID scan metadata changed after later commits");

    std::array<std::uint64_t, 8> id_buffer{};
    std::array<throwdb::NativeIdLocation, 8> location_buffer{};
    std::vector<std::uint64_t> old_ids;
    std::vector<throwdb::NativeIdLocation> old_locations;
    const std::array<std::uint64_t, 3> expected_batch_sizes{1, 1, 3};
    for (const std::uint64_t expected_size : expected_batch_sizes) {
        const std::uint64_t count = old_scan->read_into(
            id_buffer, location_buffer);
        require(
            count == expected_size,
            "physical ID scan crossed a segment boundary");
        old_ids.insert(
            old_ids.end(), id_buffer.begin(),
            id_buffer.begin() + static_cast<std::ptrdiff_t>(count));
        old_locations.insert(
            old_locations.end(), location_buffer.begin(),
            location_buffer.begin() + static_cast<std::ptrdiff_t>(count));
    }
    require(old_scan->done(), "physical ID scan did not finish at live_count");
    require(
        old_scan->read_into(id_buffer, location_buffer) == 0,
        "completed physical ID scan returned another row");
    require(
        old_ids == std::vector<std::uint64_t>({20, 30, 10, 60, 50}),
        "physical ID scan did not return authoritative RowRef order");
    require(
        old_locations
            == std::vector<throwdb::NativeIdLocation>({
                {2, 1}, {3, 0}, {4, 0}, {4, 1}, {4, 2}}),
        "physical ID scan returned wrong exact locations");

    auto current_scan = writer->scan_ids();
    require(
        current_scan->metadata().live_count == 5
            && current_scan->metadata().committed_sequence == 7,
        "new physical ID scan missed later commits");
    std::array<std::uint64_t, 2> small_buffer{};
    std::vector<std::uint64_t> current_ids;
    while (!current_scan->done()) {
        const std::uint64_t count = current_scan->read_into(small_buffer);
        current_ids.insert(
            current_ids.end(), small_buffer.begin(),
            small_buffer.begin() + static_cast<std::ptrdiff_t>(count));
    }
    require(
        current_ids == std::vector<std::uint64_t>({10, 60, 50, 20, 70}),
        "new physical ID scan returned stale or non-physical order");

    // A cursor owns the pinned collection view rather than borrowing the
    // NativeCollection object itself.
    auto detached_scan = writer->scan_ids();
    writer.reset();
    std::vector<std::uint64_t> detached_ids;
    while (!detached_scan->done()) {
        const std::uint64_t count = detached_scan->read_into(small_buffer);
        detached_ids.insert(
            detached_ids.end(), small_buffer.begin(),
            small_buffer.begin() + static_cast<std::ptrdiff_t>(count));
    }
    require(
        detached_ids == current_ids,
        "physical ID scan borrowed a destroyed collection handle");
}

void test_physical_id_scan_cooperative_cancellation() {
    TemporaryDirectory temporary("physical-id-scan-cancellation");
    constexpr std::size_t row_count = 8193;
    constexpr std::size_t first_live_row = 256;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = 10'000 + static_cast<std::uint64_t>(index);
    }
    build_collection(
        temporary.path(), ids, rows(ids.size(), 4),
        throwdb::DType::Float16, 2, ids.size() * 4, 1);

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    std::vector<std::uint64_t> deleted;
    deleted.reserve(ids.size() - 2);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        if (index != first_live_row && index + 1 != ids.size()) {
            deleted.push_back(ids[index]);
        }
    }
    const std::vector<std::uint8_t> erased = writer->erase(deleted);
    require(
        std::all_of(erased.begin(), erased.end(), [](const std::uint8_t value) {
            return value == 1;
        }),
        "cooperative ID scan fixture did not delete its stale rows");

    auto scan = writer->scan_ids();
    require(scan->metadata().live_count == 2,
            "cooperative ID scan captured the wrong live count");
    std::array<std::uint64_t, 2> output{};
    std::array<throwdb::NativeIdLocation, 2> locations{};
    std::uint64_t stop_polls = 0;
    const std::uint64_t produced = scan->read_into(
        output, locations,
        [&stop_polls] { return ++stop_polls == 10; });
    require(stop_polls == 10,
            "physical ID scan did not poll cancellation at bounded cadence");
    require(produced == 1 && !scan->done(),
            "cooperative cancellation did not return its partial batch");
    require(output[0] == ids[first_live_row]
                && locations[0]
                       == throwdb::NativeIdLocation{1, first_live_row},
            "cooperative cancellation changed physical scan order/location");

    // The original embedded overload remains non-cancellable and resumes the
    // same snapshot after the cooperatively stopped call. It must traverse the
    // remaining large stale/deleted physical tail and find the final live row.
    const std::uint64_t resumed = scan->read_into(output, locations);
    require(resumed == 1 && scan->done(),
            "embedded ID scan did not resume after cooperative cancellation");
    require(output[0] == ids.back()
                && locations[0]
                       == throwdb::NativeIdLocation{
                           1,
                           static_cast<std::uint32_t>(ids.size() - 1)},
            "resumed physical ID scan returned the wrong final row");
}

void test_read_view_and_record_scan_prefix_budget_and_concurrency() {
    TemporaryDirectory temporary("read-view-record-scan");
    constexpr std::uint32_t dimension = 4;
    constexpr std::size_t record_size = 8;
    constexpr std::size_t prefix_size = 4;
    const std::array<std::uint64_t, 5> base_ids{1, 2, 3, 4, 5};
    const std::vector<std::byte> base_rows = rows(base_ids.size(), record_size);
    build_collection(
        temporary.path(), base_ids, base_rows,
        throwdb::DType::Float16, dimension, 2 * record_size, 1);

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    const std::vector<std::byte> updated_two(record_size, std::byte{0xa2});
    const std::vector<std::byte> inserted_six(record_size, std::byte{0xb6});
    writer->update(std::array<std::uint64_t, 1>{2}, updated_two);
    require(
        writer->erase(std::array<std::uint64_t, 1>{3})
            == std::vector<std::uint8_t>({1}),
        "read view fixture delete failed");
    writer->insert(std::array<std::uint64_t, 1>{6}, inserted_six);

    require_throws<std::length_error>(
        [&] { static_cast<void>(writer->open_read_view(2, 4095)); },
        "read view ignored its pre-allocation snapshot budget");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(writer->open_read_view(dimension + 1)); },
        "read view accepted an oversized prefix");

    auto view = writer->open_read_view(2);
    auto scan = writer->scan_records(2);
    require(
        view->output_dimension() == 2
            && scan->output_dimension() == 2
            && view->dtype() == throwdb::DType::Float16
            && scan->dtype() == throwdb::DType::Float16,
        "prefix read surfaces reported the wrong output schema");
    require(
        view->metadata().committed_sequence == 3
            && view->metadata().live_count == 5
            && view->metadata().snapshot_bytes >= 4096
            && scan->metadata().committed_sequence == 3
            && scan->metadata().snapshot_bytes
                   == view->metadata().snapshot_bytes,
        "prefix read surfaces captured the wrong snapshot metadata");
    require_throws<std::length_error>(
        [&] {
            static_cast<void>(writer->scan_records(
                2, scan->metadata().snapshot_bytes - 1));
        },
        "record scan ignored its snapshot budget");

    std::array<std::uint64_t, 2> overlap_words{};
    std::array<std::uint64_t, 2> separate_ids{};
    std::array<throwdb::NativeIdLocation, 2> separate_locations{};
    std::vector<std::byte> separate_embeddings(
        separate_ids.size() * prefix_size);
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(scan->read_into(
                overlap_words,
                std::span<std::byte>(
                    reinterpret_cast<std::byte*>(overlap_words.data()),
                    separate_embeddings.size()),
                separate_locations));
        },
        "record scan accepted overlapping ID and embedding outputs");
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(scan->read_into(
                overlap_words,
                separate_embeddings,
                std::span<throwdb::NativeIdLocation>(
                    reinterpret_cast<throwdb::NativeIdLocation*>(
                        overlap_words.data()),
                    overlap_words.size())));
        },
        "record scan accepted overlapping ID and location outputs");
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(scan->read_into(
                separate_ids,
                std::span<std::byte>(
                    reinterpret_cast<std::byte*>(
                        separate_locations.data()),
                    separate_embeddings.size()),
                separate_locations));
        },
        "record scan accepted overlapping embedding and location outputs");
    require(!scan->done(),
            "overlap validation advanced the record scan cursor");

    const auto expected_prefix = [&](const std::uint64_t id) {
        std::array<std::byte, prefix_size> result{};
        if (id == 2) {
            std::copy_n(updated_two.begin(), prefix_size, result.begin());
            return result;
        }
        if (id == 6) {
            std::copy_n(inserted_six.begin(), prefix_size, result.begin());
            return result;
        }
        const auto position = std::find(base_ids.begin(), base_ids.end(), id);
        if (position == base_ids.end() || id == 3) {
            fail("unexpected ID in read-view expected-prefix helper");
        }
        const std::size_t row = static_cast<std::size_t>(
            position - base_ids.begin());
        std::copy_n(
            base_rows.begin()
                + static_cast<std::ptrdiff_t>(row * record_size),
            prefix_size, result.begin());
        return result;
    };
    const auto expected_batch = [&](const auto& requested) {
        std::vector<std::byte> expected(requested.size() * prefix_size);
        for (std::size_t index = 0; index != requested.size(); ++index) {
            const auto prefix = expected_prefix(requested[index]);
            std::copy(
                prefix.begin(), prefix.end(),
                expected.begin()
                    + static_cast<std::ptrdiff_t>(index * prefix_size));
        }
        return expected;
    };

    const std::array<std::uint64_t, 4> requested{6, 1, 2, 6};
    const std::vector<std::byte> expected = expected_batch(requested);
    std::vector<std::byte> output(expected.size());
    view->get_into(requested, output);
    require(
        output == expected,
        "read view prefix was not bit-exact or lost order/duplicates");
    std::fill(output.begin(), output.end(), std::byte{0});
    const throwdb::NativeReadStats view_stats =
        view->get_into_profiled(requested, output);
    require(
        output == expected,
        "profiled read view prefix was not bit-exact");
    require(
        view_stats.requested_rows == requested.size()
            && view_stats.base_rows == 1
            && view_stats.mutation_rows == 3
            && view_stats.planned_ranges > 0
            && view_stats.planned_read_bytes
                   >= 3 * prefix_size
            && view_stats.backend_tasks > 0
            && !view_stats.zero_delta_fast_path
            && view_stats.total_seconds
                   >= view_stats.lookup_seconds,
        "profiled read view did not expose its logical/physical plan: "
            + std::to_string(view_stats.requested_rows) + "/"
            + std::to_string(view_stats.base_rows) + "/"
            + std::to_string(view_stats.mutation_rows) + " rows, "
            + std::to_string(view_stats.planned_ranges) + " ranges, "
            + std::to_string(view_stats.planned_read_bytes) + " bytes, "
            + std::to_string(view_stats.backend_tasks) + " tasks");

    const std::array<std::uint64_t, 2> with_missing{1, 999};
    std::vector<std::byte> untouched(
        with_missing.size() * prefix_size, std::byte{0x5d});
    const std::vector<std::byte> sentinel = untouched;
    require_throws<throwdb::MissingIdError>(
        [&] { view->get_into(with_missing, untouched); },
        "read view did not reject a missing ID");
    require(
        untouched == sentinel,
        "missing-ID validation modified the current output batch");
    view->get_into(requested, output);
    require(
        output == expected,
        "read view was not reusable after a missing-ID exception");

    std::atomic<bool> read_started{false};
    std::atomic<bool> stop_reader{false};
    std::exception_ptr reader_error;
    std::thread reader([&] {
        try {
            std::vector<std::byte> concurrent_output(expected.size());
            do {
                view->get_into(requested, concurrent_output);
                if (concurrent_output != expected) {
                    fail("concurrent read view changed its snapshot bytes");
                }
                read_started.store(true, std::memory_order_release);
            } while (!stop_reader.load(std::memory_order_acquire));
        } catch (...) {
            reader_error = std::current_exception();
        }
    });
    while (!read_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const std::vector<std::byte> newer_one(record_size, std::byte{0xc1});
    const std::vector<std::byte> newer_two(record_size, std::byte{0xc2});
    writer->update(std::array<std::uint64_t, 1>{1}, newer_one);
    writer->update(std::array<std::uint64_t, 1>{2}, newer_two);
    require(
        writer->erase(std::array<std::uint64_t, 1>{4})
            == std::vector<std::uint8_t>({1}),
        "concurrent snapshot fixture delete failed");
    static_cast<void>(writer->rotate_wal());
    static_cast<void>(writer->materialize_frozen_wals());
    stop_reader.store(true, std::memory_order_release);
    reader.join();
    if (reader_error != nullptr) {
        std::rethrow_exception(reader_error);
    }
    view->get_into(requested, output);
    require(
        output == expected
            && view->metadata().committed_sequence == 3,
        "read view observed later mutation/maintenance publication");

    std::array<std::uint64_t, 3> scan_ids{};
    std::array<throwdb::NativeIdLocation, 3> scan_locations{};
    std::vector<std::byte> scan_embeddings(scan_ids.size() * prefix_size);
    std::vector<std::uint64_t> all_ids;
    std::vector<std::byte> all_embeddings;
    std::vector<throwdb::NativeIdLocation> all_locations;
    while (!scan->done()) {
        std::fill(scan_ids.begin(), scan_ids.end(), 0);
        std::fill(
            scan_embeddings.begin(), scan_embeddings.end(),
            std::byte{0xee});
        const std::uint64_t count = scan->read_into(
            scan_ids, scan_embeddings, scan_locations);
        require(count != 0, "record scan stalled before EOF");
        for (std::size_t index = 0; index != count; ++index) {
            all_ids.push_back(scan_ids[index]);
            all_locations.push_back(scan_locations[index]);
            all_embeddings.insert(
                all_embeddings.end(),
                scan_embeddings.begin()
                    + static_cast<std::ptrdiff_t>(index * prefix_size),
                scan_embeddings.begin()
                    + static_cast<std::ptrdiff_t>(
                        (index + 1) * prefix_size));
        }
        for (std::size_t index = 1; index < count; ++index) {
            require(
                scan_locations[index - 1].segment_id
                    == scan_locations[index].segment_id,
                "record scan crossed a physical segment boundary");
        }
    }
    require(
        all_ids.size() == scan->metadata().live_count,
        "record scan returned the wrong captured live count");
    require(
        all_embeddings == expected_batch(all_ids),
        "record scan prefixes were not bit-exact for its old snapshot");
    for (std::size_t index = 1; index < all_locations.size(); ++index) {
        const std::uint64_t previous =
            (static_cast<std::uint64_t>(
                 all_locations[index - 1].segment_id)
             << 32)
            | all_locations[index - 1].row_index;
        const std::uint64_t current =
            (static_cast<std::uint64_t>(all_locations[index].segment_id)
             << 32)
            | all_locations[index].row_index;
        require(previous < current, "record scan locations are not ordered");
    }
    require(
        scan->read_into(scan_ids, scan_embeddings, scan_locations) == 0,
        "record scan returned rows after done");

    // Both handles own the reader executor and pinned file resources.
    writer.reset();
    view->get_into(requested, output);
    require(output == expected,
            "detached read view borrowed the collection lifetime");
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
            && before_rotation.active_wal_put_rows == 7
            && before_rotation.active_wal_transaction_count == 5,
        "format-7 active WAL stats changed before rotation");
    const throwdb::NativeMutationLogStats rotated = writer->rotate_wal();
    require(
        rotated.rotated && rotated.generation == 2
            && rotated.frozen_wal_count == 1
            && rotated.committed_sequence == 5
            && rotated.active_wal_put_rows == 0
            && rotated.active_wal_transaction_count == 0,
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
    const throwdb::NativeIdSample bounded = writer->sample_ids(2, 19, 3);
    const throwdb::NativeIdSample bounded_repeat =
        writer->sample_ids(2, 19, 3);
    require(
        bounded.ids.size() <= 2 && bounded.inspected_slots == 3,
        "format-7 overlay sample did not honor its bounded scan");
    require(
        bounded.ids == bounded_repeat.ids
            && bounded.inspected_slots == bounded_repeat.inspected_slots,
        "format-7 bounded overlay sample is not deterministic");
    const throwdb::NativeMutationLogStats materialized =
        writer->materialize_frozen_wals();
    require(
        materialized.materialized && materialized.generation == 4
            && materialized.materialized_sequence == 5
            && materialized.materialized_wal_count == 1
            && materialized.frozen_wal_count == 0
            && materialized.delta_run_count == 1
            && materialized.delta_entry_count == 6
            && materialized.delta_put_rows == 6
            && materialized.total_delta_run_bytes != 0
            && materialized.total_delta_segment_bytes != 0
            && !materialized.reclamation_complete,
        "format-7 materialization stats changed");
    require(
        writer->get_bytes(std::array<std::uint64_t, 1>{0})
            == zero_reinserted,
        "format-7 materialization hid a newer active-WAL value");
    require(
        middle_reader->get_bytes(std::array<std::uint64_t, 1>{0})
            == std::vector<std::byte>(
                updated_extremes.begin(), updated_extremes.begin() + 4),
        "format-7 materialization changed an older reader view");

    auto materialized_scan = writer->scan_ids();
    std::array<std::uint64_t, 16> materialized_ids_buffer{};
    std::array<throwdb::NativeIdLocation, 16> materialized_locations_buffer{};
    std::vector<std::uint64_t> materialized_ids;
    std::vector<throwdb::NativeIdLocation> materialized_locations;
    while (!materialized_scan->done()) {
        const std::uint64_t count = materialized_scan->read_into(
            materialized_ids_buffer, materialized_locations_buffer);
        materialized_ids.insert(
            materialized_ids.end(), materialized_ids_buffer.begin(),
            materialized_ids_buffer.begin()
                + static_cast<std::ptrdiff_t>(count));
        materialized_locations.insert(
            materialized_locations.end(),
            materialized_locations_buffer.begin(),
            materialized_locations_buffer.begin()
                + static_cast<std::ptrdiff_t>(count));
    }
    require(
        std::set<std::uint64_t>(
            materialized_ids.begin(), materialized_ids.end())
            == std::set<std::uint64_t>(
                {0, maximum, 7, 42, 9, 10, 11}),
        "physical ID scan exposed stale materialized-run rows");
    for (std::size_t index = 1;
         index < materialized_locations.size(); ++index) {
        const auto packed = [](const throwdb::NativeIdLocation location) {
            return (static_cast<std::uint64_t>(location.segment_id) << 32)
                   | location.row_index;
        };
        require(
            packed(materialized_locations[index - 1])
                < packed(materialized_locations[index]),
            "physical ID scan lost RowRef order after materialization");
    }

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
        replayed.generation == 4 && replayed.committed_sequence == 7
            && replayed.materialized_sequence == 5
            && replayed.frozen_wal_count == 0
            && replayed.active_wal_put_rows == 1
            && replayed.delta_run_count == 1,
        "format-7 reopen mutation stats changed");
}

void test_sampler_sparse_snapshot_and_overlay_quotas() {
    TemporaryDirectory temporary("sample-sparse-snapshot");
    std::vector<std::uint64_t> base_ids(100);
    std::vector<std::uint64_t> overlay_ids(100);
    for (std::size_t index = 0; index != base_ids.size(); ++index) {
        base_ids[index] = static_cast<std::uint64_t>(index + 1);
        overlay_ids[index] = static_cast<std::uint64_t>(10'000 + index);
    }
    build_collection(
        temporary.path(), base_ids, rows(base_ids.size(), 4),
        throwdb::DType::Float16, 2, 4, 0);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    writer->insert(overlay_ids, rows(overlay_ids.size(), 4));

    // A 100-row single-partition snapshot has 256 raw hash-table slots.
    // Inspecting exactly half of the 100 overlay + 256 snapshot slots must
    // therefore allocate 50 overlay records and 128 snapshot slots. Keeping K
    // equal to the inspection limit prevents any accepted overlay ID from
    // being evicted by the bottom-K heap.
    const throwdb::NativeIdSample sample = writer->sample_ids(178, 2, 178);
    const std::size_t sampled_overlay = static_cast<std::size_t>(std::count_if(
        sample.ids.begin(), sample.ids.end(), [](const std::uint64_t id) {
            return id >= 10'000 && id < 10'100;
        }));
    require(
        sample.inspected_slots == 178 && !sample.complete,
        "sparse snapshot sample did not honor its bounded raw-slot plan");
    require(
        sampled_overlay == 50,
        "sparse snapshot density biased the overlay source quota");
}

void test_sampler_distributes_quota_between_materialized_runs() {
    TemporaryDirectory temporary("sample-multiple-runs");
    build_collection(
        temporary.path(), {}, {}, throwdb::DType::Float16,
        2, 4, 0);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    std::vector<std::uint64_t> older_ids(100);
    std::vector<std::uint64_t> newer_ids(100);
    for (std::size_t index = 0; index != older_ids.size(); ++index) {
        older_ids[index] = static_cast<std::uint64_t>(10'000 + index);
        newer_ids[index] = static_cast<std::uint64_t>(20'000 + index);
    }

    writer->insert(older_ids, rows(older_ids.size(), 4));
    require(writer->rotate_wal().rotated, "older sample WAL did not rotate");
    require(
        writer->materialize_frozen_wals().materialized,
        "older sample WAL did not materialize");
    writer->insert(newer_ids, rows(newer_ids.size(), 4));
    require(writer->rotate_wal().rotated, "newer sample WAL did not rotate");
    const throwdb::NativeMutationLogStats materialized =
        writer->materialize_frozen_wals();
    require(
        materialized.materialized && materialized.delta_run_count == 2,
        "multiple-run sample fixture did not retain two runs");

    // The empty snapshot contributes eight raw slots. With a 100-slot limit,
    // proportional source quotas are 96 run records and four snapshot slots;
    // two equal runs must then contribute 48 records each.
    const throwdb::NativeIdSample sample = writer->sample_ids(100, 2, 100);
    const std::size_t sampled_older = static_cast<std::size_t>(std::count_if(
        sample.ids.begin(), sample.ids.end(), [](const std::uint64_t id) {
            return id >= 10'000 && id < 10'100;
        }));
    const std::size_t sampled_newer = static_cast<std::size_t>(std::count_if(
        sample.ids.begin(), sample.ids.end(), [](const std::uint64_t id) {
            return id >= 20'000 && id < 20'100;
        }));
    require(
        sample.inspected_slots == 100 && !sample.complete
            && sample.ids.size() == 96,
        "multiple-run sample did not honor its bounded source quotas");
    require(
        sampled_older == 48 && sampled_newer == 48,
        "multiple-run sample quota favored run recency");
}

void test_sampler_counts_only_active_run_sections() {
    TemporaryDirectory temporary("sample-active-run-sections");
    using namespace throwdb::detail;
    constexpr std::uint32_t partition_bits = 1;
    std::array<std::uint64_t, 2> ids{};
    std::array<bool, 2> found{};
    for (std::uint64_t candidate = 1;
         candidate != 10'000 && (!found[0] || !found[1]); ++candidate) {
        const std::size_t partition = static_cast<std::size_t>(
            hash_id(candidate, kHashSeed) >> 63);
        if (!found[partition]) {
            ids[partition] = candidate;
            found[partition] = true;
        }
    }
    require(
        found[0] && found[1],
        "active-run sample fixture could not cover both partitions");
    const std::vector<std::byte> base = rows(ids.size(), 4);
    const std::vector<std::byte> replacement{
        std::byte{0x31}, std::byte{0x32}, std::byte{0x33}, std::byte{0x34},
        std::byte{0x41}, std::byte{0x42}, std::byte{0x43}, std::byte{0x44}};
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, partition_bits);
    {
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        writer->update(ids, replacement);
        require(writer->rotate_wal().rotated, "active-run WAL did not rotate");
        require(
            writer->materialize_frozen_wals().materialized,
            "active-run WAL did not materialize");
    }

    const PartitionedManifestV2 manifest = current_manifest(temporary.path());
    PartitionedCatalogV3 catalog = current_catalog(temporary.path());
    require(
        catalog.runs.size() == 1,
        "active-run sample fixture did not create one run");
    PartitionIndexRunReader mutation_run(
        temporary.path(), manifest.collection_id, manifest.schema_id,
        manifest.partition_bits, manifest.hash_seed, catalog.runs.front());
    const PartitionRunRecord folded_record = *mutation_run.lookup(0, ids[0]);

    PartitionSnapshotBuildSpec snapshot_spec;
    snapshot_spec.collection_id = manifest.collection_id;
    snapshot_spec.schema_id = manifest.schema_id;
    snapshot_spec.snapshot_id = Uuid128{0xd001, 0xd002};
    snapshot_spec.partition_bits = manifest.partition_bits;
    snapshot_spec.partition_id = 0;
    snapshot_spec.hash_seed = manifest.hash_seed;
    snapshot_spec.epoch = catalog.runs.front().epoch;
    const std::array<PartitionRunRecord, 1> folded_records{folded_record};
    catalog.snapshots[0] = write_partition_snapshot(
        temporary.path(), snapshot_spec, folded_records);
    sync_directory(partitioned_partition_snapshots_directory(temporary.path()));
    catalog.generation += 1;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xd003, 0xd004};
    clear_folded_partition(catalog.runs, manifest.partition_bits, 0);
    require(
        catalog.runs.size() == 1
            && std::to_integer<std::uint8_t>(
                   catalog.runs.front().active_partitions.front())
                   == 0x02,
        "active-run sample fixture did not retire one run section");
    publish_test_catalog(temporary.path(), manifest, catalog);

    auto reader = throwdb::NativeCollection::open(temporary.path());
    const throwdb::NativeIdSample sample = reader->sample_ids(2, 17, 100);
    require(
        sample.complete && sample.inspected_slots == 17
            && std::set<std::uint64_t>(sample.ids.begin(), sample.ids.end())
                   == std::set<std::uint64_t>(ids.begin(), ids.end()),
        "sample counted a covered-but-inactive run section");
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

void test_materialization_budget_and_tombstone_only_run() {
    TemporaryDirectory temporary("materialize-tombstone");
    const std::array<std::uint64_t, 3> ids{17, 29, 41};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 4, 1);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    require(
        writer->erase(std::array<std::uint64_t, 2>{17, 29})
            == std::vector<std::uint8_t>({1, 1}),
        "format-7 tombstone setup failed");
    static_cast<void>(writer->rotate_wal());
    const throwdb::detail::PartitionedCatalogV3 before =
        current_catalog(temporary.path());
    const std::uint64_t sealed = before.wal_entries.front().sealed_file_size;
    const throwdb::NativeMutationLogStats blocked =
        writer->materialize_frozen_wals(1, sealed - 1);
    require(
        !blocked.materialized && blocked.materialization_budget_blocked
            && blocked.generation == before.generation
            && blocked.frozen_wal_count == 1,
        "format-7 materialization did not honor its input-byte budget");

    const throwdb::NativeMutationLogStats entry_blocked =
        writer->materialize_frozen_wals(1, sealed, 1);
    require(
        !entry_blocked.materialized
            && entry_blocked.materialization_budget_blocked
            && entry_blocked.materialization_entry_budget_blocked
            && !entry_blocked.materialization_memory_budget_blocked,
        "format-7 materialization did not honor its operation budget");
    const throwdb::NativeMutationLogStats memory_blocked =
        writer->materialize_frozen_wals(1, sealed, 2, 1);
    require(
        !memory_blocked.materialized
            && memory_blocked.materialization_budget_blocked
            && !memory_blocked.materialization_entry_budget_blocked
            && memory_blocked.materialization_memory_budget_blocked,
        "format-7 materialization did not honor its peak-memory budget");

    const throwdb::NativeMutationLogStats materialized =
        writer->materialize_frozen_wals(1, sealed, 2, 1ULL << 30);
    require(
        materialized.materialized
            && !materialized.materialization_budget_blocked
            && materialized.materialized_sequence == 1
            && materialized.delta_run_count == 1
            && materialized.delta_entry_count == 2
            && materialized.delta_put_rows == 0
            && materialized.materialized_wal_input_entries == 2
            && materialized.materialization_estimated_peak_memory_bytes != 0
            && materialized.materialization_estimated_peak_memory_bytes
                   <= (1ULL << 30)
            && materialized.total_delta_segment_bytes == 0
            && materialized.total_delta_run_bytes != 0,
        "format-7 tombstone-only materialization stats changed");
    require(writer->count() == 1, "tombstone materialization changed count");
    require_throws<throwdb::MissingIdError>(
        [&] {
            static_cast<void>(writer->get_bytes(
                std::array<std::uint64_t, 1>{17}));
        },
        "materialized tombstone fell through to the initial snapshot");
    require(
        writer->get_bytes(std::array<std::uint64_t, 1>{41})
            == expected_rows(
                ids, embeddings, 4,
                std::array<std::uint64_t, 1>{41}),
        "tombstone-only run changed an unrelated ID");
    writer.reset();

    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto reopened = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(reopened->count() == 1, "tombstone run reopen changed count");
    require_throws<throwdb::MissingIdError>(
        [&] {
            static_cast<void>(reopened->get_bytes(
                std::array<std::uint64_t, 1>{17}));
        },
        "verified reopen lost the authoritative tombstone");
    reopened.reset();
    const std::filesystem::path run_file = first_file_with_extension(
        temporary.path() / "index" / "runs", ".idx");
    flip_byte(run_file, std::filesystem::file_size(run_file) - 1);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                throwdb::NativeCollection::open(temporary.path()));
        },
        "ordinary reopen accepted a corrupt partitioned run body");
}

void test_materialization_rebases_over_newer_rotation() {
    TemporaryDirectory temporary("materialize-rebase");
    const std::array<std::uint64_t, 1> ids{41};
    const std::vector<std::byte> base = rows(1, 4);
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, 1);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    const std::vector<std::byte> first{
        std::byte{0x11}, std::byte{0x12}, std::byte{0x13}, std::byte{0x14}};
    const std::vector<std::byte> second{
        std::byte{0x21}, std::byte{0x22}, std::byte{0x23}, std::byte{0x24}};
    writer->update(ids, first);
    static_cast<void>(writer->rotate_wal());
    auto middle_reader = throwdb::NativeCollection::open(temporary.path());

    materialization_output_entered.store(false, std::memory_order_release);
    release_materialization_output.store(false, std::memory_order_release);
    throwdb::detail::set_native_collection_test_hook(
        blocking_materialization_hook);
    std::exception_ptr background_error;
    throwdb::NativeMutationLogStats first_stats;
    std::thread background([&] {
        try {
            first_stats = writer->materialize_frozen_wals();
        } catch (...) {
            background_error = std::current_exception();
        }
    });
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(10);
    while (!materialization_output_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!materialization_output_entered.load(std::memory_order_acquire)) {
        release_materialization_output.store(true, std::memory_order_release);
        background.join();
        throwdb::detail::set_native_collection_test_hook(nullptr);
        fail("format-7 materialization did not reach its output barrier");
    }

    writer->update(ids, second);
    const throwdb::NativeMutationLogStats concurrent_rotation =
        writer->rotate_wal();
    require(
        concurrent_rotation.rotated
            && concurrent_rotation.frozen_wal_count == 2,
        "concurrent format-7 rotation did not preserve both WALs");
    release_materialization_output.store(true, std::memory_order_release);
    background.join();
    throwdb::detail::set_native_collection_test_hook(nullptr);
    if (background_error != nullptr) {
        std::rethrow_exception(background_error);
    }
    require(
        first_stats.materialized && first_stats.materialized_wal_count == 1
            && first_stats.materialized_sequence == 1
            && first_stats.committed_sequence == 2
            && first_stats.frozen_wal_count == 1,
        "format-7 final rebase lost a concurrent frozen WAL");
    require(
        writer->get_bytes(ids) == second,
        "materialization overrode a newer retained WAL version");
    require(
        middle_reader->get_bytes(ids) == first,
        "materialization changed a pre-existing reader resource set");

    const throwdb::NativeMutationLogStats second_stats =
        writer->materialize_frozen_wals();
    require(
        second_stats.materialized && second_stats.delta_run_count == 2
            && second_stats.materialized_sequence == 2
            && second_stats.frozen_wal_count == 0,
        "second format-7 materialization did not advance run precedence");
    require(writer->get_bytes(ids) == second, "second run changed newest row");
    writer.reset();

    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto reopened = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        reopened->get_bytes(ids) == second,
        "verified reopen did not choose the newest materialized run");
}

void test_catalog_mask_subset_and_cleaner_relocation() {
    TemporaryDirectory temporary("cleaner-relocation");
    using namespace throwdb::detail;
    std::array<std::uint64_t, 2> ids{};
    std::array<bool, 2> found{};
    for (std::uint64_t candidate = 1;
         candidate != 10000 && (!found[0] || !found[1]); ++candidate) {
        const std::size_t partition = static_cast<std::size_t>(
            hash_id(candidate, kHashSeed) >> 63);
        if (!found[partition]) {
            ids[partition] = candidate;
            found[partition] = true;
        }
    }
    require(found[0] && found[1], "test could not find IDs in both partitions");
    const std::vector<std::byte> base = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, 1);
    const std::vector<std::byte> replacement{
        std::byte{0x31}, std::byte{0x32}, std::byte{0x33}, std::byte{0x34},
        std::byte{0x41}, std::byte{0x42}, std::byte{0x43}, std::byte{0x44}};
    {
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        writer->update(ids, replacement);
        static_cast<void>(writer->rotate_wal());
        static_cast<void>(writer->materialize_frozen_wals());
    }

    const PartitionedManifestV2 manifest = current_manifest(temporary.path());
    PartitionedCatalogV3 catalog = current_catalog(temporary.path());
    require(
        catalog.runs.size() == 1
            && std::to_integer<std::uint8_t>(
                   catalog.runs.front().covered_partitions.front())
                   == 0x03
            && catalog.runs.front().active_partitions
                   == catalog.runs.front().covered_partitions,
        "materialized fixture did not cover both partitions");
    PartitionIndexRunReader mutation_run(
        temporary.path(), manifest.collection_id, manifest.schema_id,
        manifest.partition_bits, manifest.hash_seed, catalog.runs.front());
    const PartitionRunRecord partition_zero = *mutation_run.lookup(0, ids[0]);
    const PartitionRunRecord partition_one = *mutation_run.lookup(1, ids[1]);

    PartitionSnapshotBuildSpec snapshot_spec;
    snapshot_spec.collection_id = manifest.collection_id;
    snapshot_spec.schema_id = manifest.schema_id;
    snapshot_spec.snapshot_id = Uuid128{0xa001, 0xa002};
    snapshot_spec.partition_bits = manifest.partition_bits;
    snapshot_spec.partition_id = 0;
    snapshot_spec.hash_seed = manifest.hash_seed;
    snapshot_spec.epoch = catalog.runs.front().epoch;
    const std::array<PartitionRunRecord, 1> folded_records{partition_zero};
    const PartitionSnapshotDescriptor folded = write_partition_snapshot(
        temporary.path(), snapshot_spec, folded_records);
    sync_directory(partitioned_partition_snapshots_directory(temporary.path()));
    catalog.generation += 1;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xa003, 0xa004};
    catalog.snapshots[0] = folded;
    clear_folded_partition(catalog.runs, manifest.partition_bits, 0);
    require(
        catalog.runs.size() == 1
            && std::to_integer<std::uint8_t>(
                   catalog.runs.front().covered_partitions.front())
                   == 0x03
            && std::to_integer<std::uint8_t>(
                   catalog.runs.front().active_partitions.front())
                   == 0x02,
        "fold did not preserve a covered-but-inactive run partition");
    publish_test_catalog(temporary.path(), manifest, catalog);
    {
        auto folded_reader = throwdb::NativeCollection::open(temporary.path());
        require(
            folded_reader->get_bytes(ids) == replacement,
            "catalog-active mask subset changed online lookup");
    }

    catalog = current_catalog(temporary.path());
    const RowRef retired_row = RowRef::decode(partition_one.value_raw);
    const auto retired_position = std::find_if(
        catalog.segments.begin(), catalog.segments.end(),
        [retired_row](const PartitionedSegmentDescriptor& descriptor) {
            return descriptor.row_ref_segment_id
                   == retired_row.segment_id().value();
        });
    require(
        retired_position != catalog.segments.end(),
        "cleaner fixture could not find its retired segment");
    const PartitionedSegmentDescriptor retired = *retired_position;

    PartitionedSegmentSetBuildSpec segment_spec;
    segment_spec.manifest = manifest;
    segment_spec.kind = PartitionedSegmentKind::Cleaner;
    segment_spec.first_row_ref_segment_id = catalog.next_segment_id;
    segment_spec.creation_epoch = catalog.next_index_epoch;
    PartitionedSegmentSetWriter cleaner_writer(
        temporary.path(), std::move(segment_spec));
    const RowRef relocated = cleaner_writer.append(
        ids[1], partition_one.sequence,
        std::span<const std::byte>(replacement).subspan(4, 4));
    const std::vector<PartitionedSegmentDescriptor> cleaner_segments =
        cleaner_writer.finish();
    require(cleaner_segments.size() == 1, "cleaner emitted extra segments");

    PartitionIndexRunBuildSpec cleaner_spec;
    cleaner_spec.collection_id = manifest.collection_id;
    cleaner_spec.schema_id = manifest.schema_id;
    cleaner_spec.partition_bits = manifest.partition_bits;
    cleaner_spec.hash_seed = manifest.hash_seed;
    cleaner_spec.descriptor.run_id = Uuid128{0xb001, 0xb002};
    cleaner_spec.descriptor.epoch = catalog.next_index_epoch;
    cleaner_spec.descriptor.kind = PartitionedRunKind::Cleaner;
    cleaner_spec.descriptor.first_sequence = partition_one.sequence;
    cleaner_spec.descriptor.last_sequence = partition_one.sequence;
    cleaner_spec.descriptor.retired_row_ref_segment_id =
        retired.row_ref_segment_id;
    cleaner_spec.descriptor.retired_segment_id = retired.segment_id;
    cleaner_spec.descriptor.retired_row_count = retired.row_count;
    cleaner_spec.descriptor.retired_ids_payload_crc32c =
        retired.ids_payload_crc32c;
    const std::array<PartitionRunRecord, 1> cleaner_records{
        PartitionRunRecord{ids[1], partition_one.sequence, relocated.raw()}};
    const PartitionIndexRunDescriptor cleaner_run =
        write_partitioned_index_run(
            temporary.path(), cleaner_spec, cleaner_records);
    sync_directory(partitioned_index_runs_directory(temporary.path()));

    catalog.generation += 1;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xb003, 0xb004};
    catalog.next_segment_id += 1;
    catalog.next_index_epoch += 1;
    catalog.segments.erase(retired_position);
    catalog.segments.push_back(cleaner_segments.front());
    catalog.runs.push_back(cleaner_run);
    publish_test_catalog(temporary.path(), manifest, catalog);

    auto relocated_reader = throwdb::NativeCollection::open(temporary.path());
    require(
        relocated_reader->get_bytes(ids) == replacement,
        "online reducer rejected a valid cleaner relocation");
    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto verified = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        verified->get_bytes(ids) == replacement,
        "full verification rejected a valid cleaner relocation");
}

void test_materialization_crash_checkpoints_reopen() {
    using HookPoint =
        throwdb::detail::NativeCollectionTestHookPoint;
    struct CrashCase {
        HookPoint point;
        std::uint64_t generation;
        bool final_current_visible;
    };
    const std::array<CrashCase, 4> cases{{
        {HookPoint::MutationReservationDurable, 3, false},
        {HookPoint::LifecycleOutputDurable, 3, false},
        {HookPoint::MutationCatalogDurableBeforeCurrent, 4, false},
        {HookPoint::MutationCurrentDurable, 4, true},
    }};
    crash_target_operation =
        throwdb::detail::NativeCollectionTestOperation::
            MaterializeFrozenWals;
    for (std::size_t case_index = 0; case_index != cases.size(); ++case_index) {
        TemporaryDirectory temporary(
            "materialize-crash-" + std::to_string(case_index));
        const std::array<std::uint64_t, 1> ids{71};
        const std::vector<std::byte> base = rows(1, 4);
        const std::vector<std::byte> replacement{
            std::byte{0x51}, std::byte{0x52},
            std::byte{0x53}, std::byte{0x54}};
        build_collection(
            temporary.path(), ids, base, throwdb::DType::Float16,
            2, 4, 1);
        {
            auto writer = throwdb::NativeCollection::open(
                temporary.path(), {}, true);
            writer->update(ids, replacement);
            static_cast<void>(writer->rotate_wal());
        }

        crash_target_point = cases[case_index].point;
        crash_target_generation = cases[case_index].generation;
        const pid_t child = ::fork();
        require(child >= 0, "fork failed for materialization crash test");
        if (child == 0) {
            throwdb::detail::set_native_collection_test_hook(
                crashing_materialization_hook);
            try {
                auto writer = throwdb::NativeCollection::open(
                    temporary.path(), {}, true);
                static_cast<void>(writer->materialize_frozen_wals());
            } catch (...) {
                ::_exit(74);
            }
            ::_exit(75);
        }
        int status = 0;
        pid_t waited = -1;
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        require(
            waited == child && WIFEXITED(status)
                && WEXITSTATUS(status) == 73,
            "materialization crash hook did not terminate at its checkpoint");

        throwdb::ReaderOptions verified_options;
        verified_options.verify_native_index = true;
        auto reopened = throwdb::NativeCollection::open(
            temporary.path(), verified_options);
        require(
            reopened->get_bytes(ids) == replacement,
            "crash checkpoint lost the committed logical value");
        const throwdb::NativeMutationLogStats stats =
            reopened->mutation_log_stats();
        if (cases[case_index].final_current_visible) {
            require(
                stats.materialized_sequence == 1
                    && stats.frozen_wal_count == 0
                    && stats.delta_run_count == 1,
                "durable final CURRENT did not expose materialized state");
        } else {
            require(
                stats.materialized_sequence == 0
                    && stats.frozen_wal_count == 1
                    && stats.delta_run_count == 0,
                "pre-CURRENT crash exposed unpublished materialization");
        }
    }
}

void test_fully_folded_materialized_snapshot_uses_fast_path() {
    TemporaryDirectory temporary("folded-fast-path");
    using namespace throwdb::detail;
    const std::array<std::uint64_t, 1> ids{83};
    const std::vector<std::byte> base = rows(1, 4);
    const std::vector<std::byte> replacement{
        std::byte{0x61}, std::byte{0x62},
        std::byte{0x63}, std::byte{0x64}};
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, 0);
    {
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        writer->update(ids, replacement);
        static_cast<void>(writer->rotate_wal());
        static_cast<void>(writer->materialize_frozen_wals());
    }
    const PartitionedManifestV2 manifest = current_manifest(temporary.path());
    PartitionedCatalogV3 catalog = current_catalog(temporary.path());
    PartitionIndexRunReader run(
        temporary.path(), manifest.collection_id, manifest.schema_id,
        manifest.partition_bits, manifest.hash_seed, catalog.runs.front());
    const PartitionRunRecord winner = *run.lookup(0, ids.front());
    PartitionSnapshotBuildSpec snapshot_spec;
    snapshot_spec.collection_id = manifest.collection_id;
    snapshot_spec.schema_id = manifest.schema_id;
    snapshot_spec.snapshot_id = Uuid128{0xc001, 0xc002};
    snapshot_spec.partition_bits = manifest.partition_bits;
    snapshot_spec.partition_id = 0;
    snapshot_spec.hash_seed = manifest.hash_seed;
    snapshot_spec.epoch = catalog.runs.front().epoch;
    const std::array<PartitionRunRecord, 1> records{winner};
    catalog.snapshots[0] = write_partition_snapshot(
        temporary.path(), snapshot_spec, records);
    sync_directory(partitioned_partition_snapshots_directory(temporary.path()));
    clear_folded_partition(catalog.runs, manifest.partition_bits, 0);
    require(catalog.runs.empty(), "full fold retained an inactive run");
    catalog.generation += 1;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xc003, 0xc004};
    publish_test_catalog(temporary.path(), manifest, catalog);

    auto reader = throwdb::NativeCollection::open(temporary.path());
    std::vector<std::byte> output(4);
    const throwdb::NativeReadStats profile =
        reader->get_into_profiled(ids, output);
    require(
        output == replacement && profile.zero_delta_fast_path
            && profile.base_rows == 0 && profile.mutation_rows == 1,
        "fully folded materialized snapshot missed the direct fast path");
}

void test_selective_fold_is_bounded_and_rebases_newer_wal() {
    TemporaryDirectory temporary("selective-fold");
    constexpr std::uint32_t partition_bits = 2;
    std::array<std::uint64_t, 4> ids{};
    std::array<bool, 4> found{};
    for (std::uint64_t candidate = 1;
         candidate != 100000
         && std::find(found.begin(), found.end(), false) != found.end();
         ++candidate) {
        const std::size_t partition = static_cast<std::size_t>(
            throwdb::detail::hash_id(candidate, throwdb::detail::kHashSeed)
            >> (64 - partition_bits));
        if (!found[partition]) {
            ids[partition] = candidate;
            found[partition] = true;
        }
    }
    require(
        std::find(found.begin(), found.end(), false) == found.end(),
        "test could not route one ID to every partition");
    const auto tagged_rows = [](const std::uint8_t tag,
                                const std::size_t count) {
        std::vector<std::byte> result(count * 4);
        for (std::size_t index = 0; index != result.size(); ++index) {
            result[index] = static_cast<std::byte>(tag + index);
        }
        return result;
    };
    const std::vector<std::byte> base = tagged_rows(1, ids.size());
    const std::vector<std::byte> first = tagged_rows(31, ids.size());
    const std::vector<std::byte> second = tagged_rows(61, ids.size());
    const std::vector<std::byte> third = tagged_rows(91, 1);
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, partition_bits);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    const throwdb::NativeMutationLogStats fresh =
        writer->compact_delta_runs();
    require(
        !fresh.compacted && !fresh.compaction_budget_blocked
            && fresh.generation == 1,
        "selective fold fresh-store no-op changed");

    writer->update(ids, first);
    static_cast<void>(writer->rotate_wal());
    static_cast<void>(writer->materialize_frozen_wals());
    writer->update(ids, second);
    static_cast<void>(writer->rotate_wal());
    static_cast<void>(writer->materialize_frozen_wals());
    auto old_reader = throwdb::NativeCollection::open(temporary.path());

    const throwdb::NativeMutationLogStats before =
        writer->mutation_log_stats();
    const throwdb::NativeMutationLogStats blocked =
        writer->compact_delta_runs(
            2, 1, std::numeric_limits<std::uint64_t>::max());
    require(
        !blocked.compacted && blocked.compaction_budget_blocked
            && blocked.generation == before.generation,
        "selective fold did not report a budget-blocked no-op");

    fold_output_entered.store(false, std::memory_order_release);
    release_fold_output.store(false, std::memory_order_release);
    throwdb::detail::set_native_collection_test_hook(blocking_fold_hook);
    std::exception_ptr background_error;
    throwdb::NativeMutationLogStats first_fold;
    std::thread background([&] {
        try {
            first_fold = writer->compact_delta_runs(
                2, std::numeric_limits<std::uint64_t>::max(), 6);
        } catch (...) {
            background_error = std::current_exception();
        }
    });
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(10);
    while (!fold_output_entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!fold_output_entered.load(std::memory_order_acquire)) {
        release_fold_output.store(true, std::memory_order_release);
        background.join();
        throwdb::detail::set_native_collection_test_hook(nullptr);
        fail("selective fold did not reach its output barrier");
    }
    writer->update(
        std::array<std::uint64_t, 1>{ids[0]}, third);
    const throwdb::NativeMutationLogStats concurrent_rotation =
        writer->rotate_wal();
    require(
        concurrent_rotation.rotated
            && concurrent_rotation.frozen_wal_count == 1,
        "selective fold blocked a concurrent WAL rotation");
    release_fold_output.store(true, std::memory_order_release);
    background.join();
    throwdb::detail::set_native_collection_test_hook(nullptr);
    if (background_error != nullptr) {
        std::rethrow_exception(background_error);
    }

    require(
        first_fold.compacted && first_fold.generation == 10
            && first_fold.committed_sequence == 3
            && first_fold.compacted_partition_count == 2
            && first_fold.compacted_run_section_count == 4
            && first_fold.compacted_snapshot_input_entries == 2
            && first_fold.compacted_delta_input_entries == 4
            && first_fold.compacted_delta_input_bytes
                   == 4 * throwdb::detail::kPartitionRunRecordSize
            && first_fold.compacted_output_entries == 2
            && first_fold.compacted_snapshot_input_bytes != 0
            && first_fold.compacted_output_bytes != 0
            && first_fold.compaction_budget_blocked
            && !first_fold.reclamation_complete
            && first_fold.delta_run_count == 2
            && first_fold.frozen_wal_count == 1,
        "bounded selective fold diagnostics changed");
    std::vector<std::byte> expected = second;
    std::copy(third.begin(), third.end(), expected.begin());
    require(
        writer->get_bytes(ids) == expected,
        "selective fold overrode a newer WAL value");
    require(
        old_reader->get_bytes(ids) == second,
        "selective fold changed an already-open reader view");

    const throwdb::NativeMutationLogStats second_fold =
        writer->compact_delta_runs(
            2, std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max());
    require(
        second_fold.compacted && second_fold.generation == 12
            && second_fold.compacted_partition_count == 2
            && second_fold.compacted_run_section_count == 4
            && !second_fold.compaction_budget_blocked
            && second_fold.delta_run_count == 0,
        "second selective fold did not clear all active run sections");
    require(writer->get_bytes(ids) == expected, "second fold changed output");
    writer.reset();

    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto reopened = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        reopened->get_bytes(ids) == expected,
        "verified reopen changed selectively folded state");
    require(
        old_reader->get_bytes(ids) == second,
        "old reader pins did not survive all selective folds");
}

void test_format_aware_default_fold_covers_all_active_runs() {
    TemporaryDirectory temporary("format-aware-default-fold");
    const std::array<std::uint64_t, 1> ids{17};
    const std::vector<std::byte> base{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 4, 0);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);

    std::vector<std::byte> expected(4);
    for (std::uint8_t round = 0; round != 9; ++round) {
        for (std::size_t index = 0; index != expected.size(); ++index) {
            expected[index] = static_cast<std::byte>(31 + round + index);
        }
        writer->update(ids, expected);
        require(writer->rotate_wal().rotated,
                "default-fold fixture did not rotate");
        require(writer->materialize_frozen_wals().materialized,
                "default-fold fixture did not materialize");
    }
    require(writer->mutation_log_stats().delta_run_count == 9,
            "default-fold fixture did not create nine active runs");

    const throwdb::NativeMutationLogStats folded =
        writer->compact_delta_runs();
    require(
        folded.compacted && folded.compacted_partition_count == 1
            && folded.compacted_run_section_count == 9
            && folded.delta_run_count == 0
            && !folded.compaction_budget_blocked,
        "format-7 native default did not fold every active run section");
    require(writer->get_bytes(ids) == expected,
            "format-aware default fold changed the newest value");
}

void test_fold_budget_charges_worst_case_output_geometry() {
    TemporaryDirectory temporary("fold-output-budget");
    build_collection(
        temporary.path(), {}, {}, throwdb::DType::Float16,
        2, 4, 0);
    constexpr std::size_t row_count = 1000;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = static_cast<std::uint64_t>(index + 1);
    }
    const std::vector<std::byte> embeddings = rows(row_count, 4);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    writer->insert(ids, embeddings);
    static_cast<void>(writer->rotate_wal());
    static_cast<void>(writer->materialize_frozen_wals());
    const throwdb::detail::PartitionedCatalogV3 catalog =
        current_catalog(temporary.path());
    require(
        catalog.runs.size() == 1
            && catalog.runs.front().entry_count == row_count
            && catalog.snapshots.front().entry_count == 0,
        "worst-case fold fixture geometry changed");
    const std::uint64_t raw_input_bytes =
        catalog.snapshots.front().file_size
        + catalog.runs.front().entry_count
              * throwdb::detail::kPartitionRunRecordSize;
    const throwdb::NativeMutationLogStats blocked =
        writer->compact_delta_runs(2, raw_input_bytes, row_count);
    require(
        !blocked.compacted && blocked.compaction_budget_blocked
            && blocked.generation == catalog.generation,
        "fold admitted output growth that exceeded the byte budget");

    const throwdb::NativeMutationLogStats folded =
        writer->compact_delta_runs(2, 100000, row_count);
    require(
        folded.compacted && !folded.compaction_budget_blocked
            && folded.compacted_partition_count == 1
            && folded.compacted_run_section_count == 1
            && folded.compacted_delta_input_entries == row_count
            && folded.compacted_snapshot_input_entries == 0
            && folded.compacted_output_entries == row_count
            && folded.compacted_output_bytes <= 100000
            && folded.delta_run_count == 0,
        "fold did not honor its bounded worst-case output geometry");
    require(
        writer->get_bytes(ids) == embeddings,
        "worst-case geometry fold changed stored rows");
}

void test_selective_fold_crash_checkpoints_reopen() {
    using HookPoint =
        throwdb::detail::NativeCollectionTestHookPoint;
    struct CrashCase {
        HookPoint point;
        std::uint64_t generation;
        bool final_current_visible;
    };
    const std::array<CrashCase, 4> cases{{
        {HookPoint::MutationReservationDurable, 5, false},
        {HookPoint::LifecycleOutputDurable, 5, false},
        {HookPoint::MutationCatalogDurableBeforeCurrent, 6, false},
        {HookPoint::MutationCurrentDurable, 6, true},
    }};
    crash_target_operation =
        throwdb::detail::NativeCollectionTestOperation::FoldPartitions;
    for (std::size_t case_index = 0; case_index != cases.size(); ++case_index) {
        TemporaryDirectory temporary(
            "fold-crash-" + std::to_string(case_index));
        const std::array<std::uint64_t, 1> ids{101};
        const std::vector<std::byte> base = rows(1, 4);
        const std::vector<std::byte> replacement{
            std::byte{0x71}, std::byte{0x72},
            std::byte{0x73}, std::byte{0x74}};
        build_collection(
            temporary.path(), ids, base, throwdb::DType::Float16,
            2, 4, 0);
        {
            auto writer = throwdb::NativeCollection::open(
                temporary.path(), {}, true);
            writer->update(ids, replacement);
            static_cast<void>(writer->rotate_wal());
            static_cast<void>(writer->materialize_frozen_wals());
        }
        crash_target_point = cases[case_index].point;
        crash_target_generation = cases[case_index].generation;
        const pid_t child = ::fork();
        require(child >= 0, "fork failed for selective-fold crash test");
        if (child == 0) {
            throwdb::detail::set_native_collection_test_hook(
                crashing_materialization_hook);
            try {
                auto writer = throwdb::NativeCollection::open(
                    temporary.path(), {}, true);
                static_cast<void>(writer->compact_delta_runs(
                    2, std::numeric_limits<std::uint64_t>::max(),
                    std::numeric_limits<std::uint64_t>::max()));
            } catch (...) {
                ::_exit(76);
            }
            ::_exit(77);
        }
        int status = 0;
        pid_t waited = -1;
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        require(
            waited == child && WIFEXITED(status)
                && WEXITSTATUS(status) == 73,
            "selective-fold crash hook missed its checkpoint");

        throwdb::ReaderOptions verified_options;
        verified_options.verify_native_index = true;
        auto reopened = throwdb::NativeCollection::open(
            temporary.path(), verified_options);
        require(
            reopened->get_bytes(ids) == replacement,
            "selective-fold crash changed the logical value");
        const throwdb::detail::PartitionedCatalogV3 catalog =
            current_catalog(temporary.path());
        require(
            catalog.runs.empty()
                == cases[case_index].final_current_visible,
            "selective-fold crash exposed the wrong catalog generation");
    }
    crash_target_operation =
        throwdb::detail::NativeCollectionTestOperation::
            MaterializeFrozenWals;
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

void test_selective_fold_rejects_checksum_valid_snapshot_corruption() {
    using namespace throwdb::detail;
    TemporaryDirectory temporary("fold-structural-corruption");
    const std::array<std::uint64_t, 1> ids{701};
    const std::vector<std::byte> base = rows(1, 4);
    const std::vector<std::byte> replacement{
        std::byte{0x61}, std::byte{0x62},
        std::byte{0x63}, std::byte{0x64}};
    build_collection(
        temporary.path(), ids, base, throwdb::DType::Float16,
        2, 8, 0);
    {
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        writer->update(ids, replacement);
        require(writer->rotate_wal().rotated,
                "structural-corruption fixture did not rotate");
        require(writer->materialize_frozen_wals().materialized,
                "structural-corruption fixture did not materialize");
    }

    const PartitionedManifestV2 manifest = current_manifest(temporary.path());
    PartitionedCatalogV3 catalog = current_catalog(temporary.path());
    require(catalog.runs.size() == 1,
            "structural-corruption fixture lacks an active run");
    PartitionSnapshotDescriptor descriptor = catalog.snapshots.front();
    const std::filesystem::path snapshot_path =
        partitioned_partition_snapshot_path(temporary.path(), descriptor);
    std::vector<std::byte> snapshot = read_file(snapshot_path);
    PartitionSnapshotHeaderV1 header =
        decode_partition_snapshot_header_v1(
            std::span<const std::byte>(snapshot).first(
                static_cast<std::size_t>(kPartitionedHeaderSize)));
    const std::uint64_t controls_end =
        kPartitionedHeaderSize + descriptor.capacity;
    require(controls_end < header.lookup_slots_offset,
            "snapshot fixture has no alignment padding to corrupt");
    snapshot[static_cast<std::size_t>(controls_end)] = std::byte{1};
    descriptor.body_crc32c = crc32c(
        std::span<const std::byte>(snapshot).subspan(
            static_cast<std::size_t>(kPartitionedHeaderSize)));
    header.descriptor = descriptor;
    const auto encoded_header = encode_partition_snapshot_header_v1(header);
    std::copy(
        encoded_header.begin(), encoded_header.end(), snapshot.begin());
    write_file(snapshot_path, snapshot);

    catalog.generation += 1;
    catalog.previous_catalog_id = catalog.catalog_id;
    catalog.catalog_id = Uuid128{0xf001, 0xf002};
    catalog.snapshots.front() = descriptor;
    publish_test_catalog(temporary.path(), manifest, catalog);

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(
                writer->compact_delta_runs(2, 1ULL << 20, 100));
        },
        "selective fold normalized checksum-valid snapshot corruption");
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
    auto empty_scan = collection->scan_ids();
    require(
        empty_scan->metadata().live_count == 0 && empty_scan->done(),
        "empty physical ID scan did not finish immediately");
    std::array<std::uint64_t, 1> empty_output{};
    require(
        empty_scan->read_into(empty_output) == 0,
        "empty physical ID scan returned a row");
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
        const auto require_inherited_stream_rejected =
            [&result](auto&& open_stream, const int not_rejected,
                      const int wrong_error) {
                if (result != 0) {
                    return;
                }
                try {
                    static_cast<void>(open_stream());
                    result = not_rejected;
                } catch (const throwdb::StorageError& error) {
                    if (std::string(error.what()).find("fork")
                        == std::string::npos) {
                        result = wrong_error;
                    }
                } catch (...) {
                    result = wrong_error;
                }
            };
        require_inherited_stream_rejected(
            [&] { return reader->scan_ids(); }, 25, 26);
        require_inherited_stream_rejected(
            [&] { return reader->open_read_view(); }, 27, 28);
        require_inherited_stream_rejected(
            [&] { return reader->scan_records(); }, 29, 33);
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

    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(
                throwdb::NativeCollection::open(temporary.path()));
        },
        "format-7 reader accepted a catalog whose run resource is missing");
}

void test_mutable_open_limits_and_pending_fence_fail_closed() {
    {
        TemporaryDirectory live("bounded-live-write");
        const std::array<std::uint64_t, 2> live_ids{3, 5};
        const std::vector<std::byte> live_embeddings =
            rows(live_ids.size(), 4);
        build_collection(
            live.path(), live_ids, live_embeddings,
            throwdb::DType::Float16, 2, 8, 1);
        throwdb::NativeOpenOptions live_bound;
        live_bound.max_overlay_entries = 1;
        auto bounded_writer = throwdb::NativeCollection::open(
            live.path(), {}, true, false, live_bound);
        require_throws<throwdb::StorageError>(
            [&] { bounded_writer->update(live_ids, live_embeddings); },
            "live mutation exceeded its exact overlay entry bound");
        const auto after = bounded_writer->mutation_log_stats();
        require(
            after.committed_sequence == 0
                && after.total_wal_operation_count == 0
                && after.overlay_entry_count == 0,
            "live overlay rejection happened after WAL publication");
    }

    TemporaryDirectory temporary("bounded-open");
    const std::array<std::uint64_t, 2> ids{7, 11};
    const std::vector<std::byte> embeddings = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 8, 1);
    {
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        const std::vector<std::byte> replacements = rows(ids.size(), 4);
        writer->update(ids, replacements);
        const auto stats = writer->mutation_log_stats();
        require(
            stats.total_wal_operation_count == 2
                && stats.overlay_entry_count == 2
                && stats.estimated_overlay_bytes >= 2 * 64,
            "format-7 retained-mutation diagnostics are incomplete");
    }

    throwdb::NativeOpenOptions operation_bound;
    operation_bound.max_wal_operations = 1;
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, true, false, operation_bound));
        },
        "writable open replay exceeded its WAL operation bound");

    throwdb::NativeOpenOptions overlay_bound;
    overlay_bound.max_overlay_entries = 1;
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, true, false, overlay_bound));
        },
        "writable open replay exceeded its overlay entry bound");

    std::filesystem::create_directories(temporary.path() / "server");
    {
        std::ofstream fence(
            temporary.path() / "server" / "pending-mutation-v1",
            std::ios::binary | std::ios::trunc);
        require(fence.good(), "cannot create pending-mutation test fence");
        fence << "prepared\n";
    }
    // Read-only diagnostics remain available, but an ordinary writer cannot
    // bypass the durable server's unresolved PREPARE marker.
    static_cast<void>(throwdb::NativeCollection::open(temporary.path()));
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(throwdb::NativeCollection::open(
                temporary.path(), {}, true));
        },
        "ordinary writable open crossed a pending-mutation fence");
    throwdb::NativeOpenOptions recovery;
    recovery.allow_pending_mutation_recovery = true;
    static_cast<void>(throwdb::NativeCollection::open(
        temporary.path(), {}, true, false, recovery));
}

void test_live_wal_limits_preserve_state_and_recover_after_materialization() {
    // Exercise byte and operation bounds independently, including the frozen
    // prefix: an empty active WAL must not bypass retained-backlog limits.
    for (const bool bound_bytes : {false, true}) {
        TemporaryDirectory temporary(
            bound_bytes ? "live-wal-bytes" : "live-wal-operations");
        const std::array<std::uint64_t, 2> ids{3, 5};
        auto embeddings = rows(ids.size(), 4);
        build_collection(
            temporary.path(), ids, embeddings, throwdb::DType::Float16,
            2, 8, 1);
        const std::array<std::uint64_t, 1> changed{ids[0]};
        const std::array<std::uint64_t, 1> untouched{ids[1]};
        const std::array<std::uint64_t, 1> inserted{7};
        const std::vector<std::byte> replacement(4, std::byte{0x7f});
        auto writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true);
        writer->update(changed, replacement);
        std::copy(replacement.begin(), replacement.end(), embeddings.begin());
        const auto baseline = writer->rotate_wal();
        require(baseline.frozen_wal_count == 1
                    && baseline.active_wal_operation_count == 0,
                "live-bound fixture does not retain a frozen WAL");
        writer.reset();

        throwdb::NativeOpenOptions limits;
        if (bound_bytes) {
            limits.max_wal_bytes = baseline.total_wal_bytes;
        } else {
            limits.max_wal_operations = baseline.total_wal_operation_count;
        }
        writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true, false, limits);
        const auto require_unchanged = [&] {
            const auto after = writer->mutation_log_stats();
            require(after.generation == baseline.generation
                        && after.committed_sequence == baseline.committed_sequence
                        && after.total_wal_bytes == baseline.total_wal_bytes
                        && after.total_wal_operation_count
                               == baseline.total_wal_operation_count
                        && after.overlay_entry_count == baseline.overlay_entry_count
                        && writer->count() == ids.size(),
                    "rejected mutation changed WAL or overlay state");
            require(writer->get_bytes(ids) == embeddings,
                    "rejected mutation changed visible rows");
        };
        require_throws<throwdb::StorageError>(
            [&] { writer->insert(inserted, replacement); },
            "live insert exceeded the retained WAL limit");
        require_unchanged();
        require_throws<throwdb::StorageError>(
            [&] { writer->upsert(inserted, replacement); },
            "live upsert exceeded the retained WAL limit");
        require_unchanged();
        require_throws<throwdb::StorageError>(
            [&] { writer->update(changed, replacement); },
            "live update exceeded the retained WAL limit");
        require_unchanged();
        require_throws<throwdb::StorageError>(
            [&] { static_cast<void>(writer->erase(untouched)); },
            "live erase exceeded the retained WAL limit");
        require_unchanged();

        // A no-op consumes no durable budget, even exactly at the limit.
        require(writer->erase(inserted) == std::vector<std::uint8_t>{0},
                "missing-ID erase changed state at the WAL limit");
        require(!writer->rotate_wal().rotated,
                "empty active WAL rotated at the WAL limit");
        require_unchanged();
        writer.reset();
        writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true, false, limits);
        require_unchanged();

        const auto maintained = writer->materialize_frozen_wals();
        require(maintained.materialized
                    && maintained.total_wal_operation_count == 0,
                "materialization did not release retained WAL budget");
        writer->insert(inserted, replacement);
        require(writer->get_bytes(inserted) == replacement,
                "released WAL budget did not admit a new mutation");
        writer.reset();
        writer = throwdb::NativeCollection::open(
            temporary.path(), {}, true, false, limits);
        require(writer->count() == ids.size() + 1
                    && writer->get_bytes(inserted) == replacement,
                "bounded writer could not reopen its accepted mutation");
    }
}

void test_live_wal_rotation_and_append_accept_exact_byte_limits() {
    TemporaryDirectory temporary("rotation-byte-budget");
    const std::array<std::uint64_t, 1> ids{9};
    const auto embeddings = rows(ids.size(), 4);
    build_collection(
        temporary.path(), ids, embeddings, throwdb::DType::Float16,
        2, 8, 1);
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    writer->update(ids, embeddings);
    const auto baseline = writer->mutation_log_stats();
    writer.reset();

    throwdb::NativeOpenOptions limits;
    limits.max_wal_bytes =
        baseline.total_wal_bytes + throwdb::detail::kWalFileHeaderSize - 1;
    writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true, false, limits);
    require_throws<throwdb::StorageError>(
        [&] { static_cast<void>(writer->rotate_wal()); },
        "rotation failed to charge its complete retained WAL header");
    const auto rejected = writer->mutation_log_stats();
    require(rejected.generation == baseline.generation
                && rejected.total_wal_bytes == baseline.total_wal_bytes
                && rejected.frozen_wal_count == 0,
            "rejected rotation published a catalog or successor WAL");
    writer.reset();

    ++limits.max_wal_bytes;
    writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true, false, limits);
    const auto rotated = writer->rotate_wal();
    require(rotated.rotated
                && rotated.total_wal_bytes == limits.max_wal_bytes,
            "rotation rejected an exact header-byte budget");
    writer.reset();

    limits.max_wal_bytes +=
        baseline.total_wal_bytes - throwdb::detail::kWalFileHeaderSize;
    limits.max_wal_operations = 2;
    writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true, false, limits);
    writer->update(ids, embeddings);
    const auto appended = writer->mutation_log_stats();
    require(appended.total_wal_bytes == limits.max_wal_bytes
                && appended.total_wal_operation_count == limits.max_wal_operations,
            "append rejected an exact byte/operation budget");
    require_throws<throwdb::StorageError>(
        [&] { writer->update(ids, embeddings); },
        "append exceeded its exact retained WAL budget");
    writer.reset();
    writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true, false, limits);
    require(writer->get_bytes(ids) == embeddings,
            "exact-budget mutations could not reopen under the same limits");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 30> tests{{
        {"float16_end_to_end_and_read_surface",
         test_float16_end_to_end_and_read_surface},
        {"float32_and_reader_trial", test_float32_and_reader_trial},
        {"physical_id_scan_order_locations_and_snapshot_isolation",
         test_physical_id_scan_order_locations_and_snapshot_isolation},
        {"physical_id_scan_cooperative_cancellation",
         test_physical_id_scan_cooperative_cancellation},
        {"read_view_and_record_scan_prefix_budget_and_concurrency",
         test_read_view_and_record_scan_prefix_budget_and_concurrency},
        {"writable_crud_reopen_rotation_and_snapshot_isolation",
         test_writable_crud_reopen_rotation_and_snapshot_isolation},
        {"sampler_sparse_snapshot_and_overlay_quotas",
         test_sampler_sparse_snapshot_and_overlay_quotas},
        {"sampler_distributes_quota_between_materialized_runs",
         test_sampler_distributes_quota_between_materialized_runs},
        {"sampler_counts_only_active_run_sections",
         test_sampler_counts_only_active_run_sections},
        {"active_wal_tail_recovery_and_frozen_corruption",
         test_active_wal_tail_recovery_and_frozen_corruption},
        {"materialization_budget_and_tombstone_only_run",
         test_materialization_budget_and_tombstone_only_run},
        {"materialization_rebases_over_newer_rotation",
         test_materialization_rebases_over_newer_rotation},
        {"catalog_mask_subset_and_cleaner_relocation",
         test_catalog_mask_subset_and_cleaner_relocation},
        {"materialization_crash_checkpoints_reopen",
         test_materialization_crash_checkpoints_reopen},
        {"fully_folded_materialized_snapshot_uses_fast_path",
         test_fully_folded_materialized_snapshot_uses_fast_path},
        {"selective_fold_is_bounded_and_rebases_newer_wal",
         test_selective_fold_is_bounded_and_rebases_newer_wal},
        {"format_aware_default_fold_covers_all_active_runs",
         test_format_aware_default_fold_covers_all_active_runs},
        {"fold_budget_charges_worst_case_output_geometry",
         test_fold_budget_charges_worst_case_output_geometry},
        {"selective_fold_crash_checkpoints_reopen",
         test_selective_fold_crash_checkpoints_reopen},
        {"selective_fold_rejects_checksum_valid_snapshot_corruption",
         test_selective_fold_rejects_checksum_valid_snapshot_corruption},
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
        {"mutable_open_limits_and_pending_fence_fail_closed",
         test_mutable_open_limits_and_pending_fence_fail_closed},
        {"live_wal_limits_preserve_state_and_recover_after_materialization",
         test_live_wal_limits_preserve_state_and_recover_after_materialization},
        {"live_wal_rotation_and_append_accept_exact_byte_limits",
         test_live_wal_rotation_and_append_accept_exact_byte_limits},
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
