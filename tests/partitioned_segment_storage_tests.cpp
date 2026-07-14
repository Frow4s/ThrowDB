#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partitioned_segment_storage.hpp"

namespace {

using throwdb::detail::PartitionedManifestV2;
using throwdb::detail::PartitionedSegmentDescriptor;
using throwdb::detail::PartitionedSegmentKind;
using throwdb::detail::PartitionedSegmentReader;
using throwdb::detail::PartitionedSegmentSetBuildSpec;
using throwdb::detail::PartitionedSegmentSetWriter;
using throwdb::detail::RowRef;
using throwdb::detail::SegmentId;

constexpr std::uint64_t kRecordSize = 8;

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
    } catch (const std::exception& error) {
        fail(message + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        fail(message + " (wrong non-standard exception)");
    }
    fail(message + " (no exception)");
}

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
            / ("throwdb-segment-set-test-" + std::to_string(::getpid()) + "-"
               + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_ / "segments");
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

[[nodiscard]] PartitionedManifestV2 manifest(
    const std::uint64_t target_segment_size = 2 * kRecordSize) {
    PartitionedManifestV2 result;
    result.collection_id = {11, 12};
    result.schema_id = {21, 22};
    result.initial_active_wal_id = {31, 32};
    result.schema.id_type = throwdb::detail::IdType::UInt64;
    result.schema.dtype = throwdb::detail::DTypeCode::Float16;
    result.schema.dimension = 4;
    result.schema.item_size = 2;
    result.schema.row_layout = throwdb::detail::RowLayout::DenseRowMajor;
    result.schema.index_kind =
        throwdb::detail::IndexKind::UInt64PartitionedRowRef;
    result.schema.hash_algorithm =
        throwdb::detail::HashAlgorithm::SplitMix64V1;
    result.record_size = kRecordSize;
    result.partition_bits = 2;
    result.partition_count = 4;
    result.target_segment_size = target_segment_size;
    result.initial_active_wal_segment_id = 1;
    result.initial_next_segment_id = 2;
    result.creation_time_unix_ns = 123;
    return result;
}

[[nodiscard]] PartitionedSegmentSetBuildSpec spec(
    const PartitionedManifestV2& input_manifest,
    const PartitionedSegmentKind kind = PartitionedSegmentKind::Materialized,
    const std::uint32_t first_segment_id = 100) {
    return PartitionedSegmentSetBuildSpec{
        input_manifest, kind, first_segment_id, 9, {}};
}

[[nodiscard]] std::array<std::byte, kRecordSize> row(
    const std::uint8_t seed) {
    std::array<std::byte, kRecordSize> result{};
    for (std::size_t index = 0; index != result.size(); ++index) {
        result[index] = static_cast<std::byte>(
            static_cast<std::uint8_t>(seed + index * 29));
    }
    return result;
}

[[nodiscard]] std::size_t final_resource_count(
    const std::filesystem::path& store) {
    std::size_t result = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
             store / "segments")) {
        const auto extension = entry.path().extension();
        if (extension == ".dat" || extension == ".ids") {
            ++result;
        }
    }
    return result;
}

[[nodiscard]] std::size_t all_resource_count(
    const std::filesystem::path& store) {
    return static_cast<std::size_t>(std::distance(
        std::filesystem::directory_iterator(store / "segments"),
        std::filesystem::directory_iterator{}));
}

void test_multi_segment_stream_and_bit_exact_resolution() {
    TempDirectory temporary;
    const PartitionedManifestV2 input_manifest = manifest();
    PartitionedSegmentSetWriter writer(
        temporary.path(), spec(input_manifest));
    const std::array<std::uint64_t, 5> ids{
        0,
        std::numeric_limits<std::uint64_t>::max(),
        42,
        0xfeedfacecafebeefULL,
        7};
    const std::array<std::uint64_t, 5> sequences{5, 3, 9, 4, 7};
    std::array<std::array<std::byte, kRecordSize>, 5> rows{
        row(1), row(51), row(101), row(151), row(201)};
    std::vector<RowRef> refs;
    for (std::size_t index = 0; index != ids.size(); ++index) {
        refs.push_back(writer.append(ids[index], sequences[index], rows[index]));
    }
    require(writer.row_count() == 5, "streamed row count is wrong");
    require(
        refs == std::vector<RowRef>({
                    RowRef::create(100, 0), RowRef::create(100, 1),
                    RowRef::create(101, 0), RowRef::create(101, 1),
                    RowRef::create(102, 0)}),
        "RowRefs were not allocated contiguously across segment splits");
    require(
        final_resource_count(temporary.path()) == 0,
        "unfinished segment resources became visible under final names");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(writer.descriptors()); },
        "descriptors became visible before finish");

    const std::vector<PartitionedSegmentDescriptor> descriptors =
        writer.finish();
    require(writer.finished(), "finished writer did not report completion");
    require(
        writer.finish() == descriptors,
        "successful finish was not idempotent");
    require(descriptors.size() == 3, "target size did not split three segments");
    require(
        final_resource_count(temporary.path()) == 6,
        "finish did not publish both resources for every segment");
    require(
        descriptors[0].row_ref_segment_id == 100
            && descriptors[0].row_count == 2
            && descriptors[0].minimum_sequence == 3
            && descriptors[0].maximum_sequence == 5
            && descriptors[1].row_ref_segment_id == 101
            && descriptors[1].row_count == 2
            && descriptors[1].minimum_sequence == 4
            && descriptors[1].maximum_sequence == 9
            && descriptors[2].row_ref_segment_id == 102
            && descriptors[2].row_count == 1
            && descriptors[2].minimum_sequence == 7
            && descriptors[2].maximum_sequence == 7,
        "segment descriptors lost row or sequence bounds");

    std::vector<std::unique_ptr<PartitionedSegmentReader>> readers;
    std::vector<throwdb::detail::PhysicalReadRequest> requests;
    std::size_t global_row = 0;
    for (const PartitionedSegmentDescriptor& descriptor : descriptors) {
        readers.push_back(throwdb::detail::open_partitioned_segment(
            temporary.path(), input_manifest, descriptor, true));
        PartitionedSegmentReader& reader = *readers.back();
        require(reader.descriptor() == descriptor, "opened descriptor changed");
        for (std::uint64_t local = 0; local != descriptor.row_count; ++local) {
            require(
                reader.sidecar_at(local)
                    == throwdb::detail::PartitionedSegmentIdEntry{
                        ids[global_row], sequences[global_row]},
                "sidecar did not preserve exact uint64 ID and sequence");
            requests.push_back(reader.resolve(
                RowRef::create(descriptor.row_ref_segment_id, local),
                global_row));
            ++global_row;
        }
    }
    require(global_row == rows.size(), "not every row was reopened");
    std::vector<std::byte> output(rows.size() * kRecordSize);
    throwdb::detail::PhysicalReader physical_reader;
    physical_reader.execute(kRecordSize, requests, output);
    for (std::size_t index = 0; index != rows.size(); ++index) {
        require(
            std::equal(
                rows[index].begin(), rows[index].end(),
                output.begin() + index * kRecordSize),
            "opaque row bits changed through PhysicalReadRequest resolution");
    }
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(readers.front()->resolve(
                RowRef::create(101, 0), 0));
        },
        "reader accepted a RowRef owned by another segment");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(readers.front()->sidecar_at(2)); },
        "reader accepted a sidecar row outside exact geometry");
}

void test_cleaner_zero_sequence_and_empty_finish() {
    TempDirectory temporary;
    const PartitionedManifestV2 input_manifest = manifest(8 * kRecordSize);
    PartitionedSegmentSetWriter cleaner(
        temporary.path(),
        spec(input_manifest, PartitionedSegmentKind::Cleaner, 200));
    const auto first = row(9);
    const auto second = row(19);
    static_cast<void>(cleaner.append(10, 0, first));
    static_cast<void>(cleaner.append(11, 6, second));
    const auto descriptors = cleaner.finish();
    require(
        descriptors.size() == 1
            && descriptors[0].kind == PartitionedSegmentKind::Cleaner
            && descriptors[0].minimum_sequence == 0
            && descriptors[0].maximum_sequence == 6,
        "cleaner segment rejected or lost sequence zero");
    auto reader = throwdb::detail::open_partitioned_segment(
        temporary.path(), input_manifest, descriptors[0], true);
    require(
        reader->sidecar_at(0).sequence == 0
            && reader->sidecar_at(1).sequence == 6,
        "cleaner sidecar sequences changed");

    TempDirectory empty_directory;
    PartitionedSegmentSetWriter empty(
        empty_directory.path(), spec(input_manifest));
    require(empty.finish().empty(), "empty segment-set finish created a segment");
    require(
        all_resource_count(empty_directory.path()) == 0,
        "empty segment-set finish left resources");
}

void test_invalid_inputs_poison_and_rowref_capacity() {
    {
        TempDirectory temporary;
        PartitionedManifestV2 invalid = manifest();
        invalid.target_segment_size = invalid.record_size - 1;
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(PartitionedSegmentSetWriter(
                    temporary.path(), spec(invalid)));
            },
            "target smaller than one row was accepted");
    }
    for (const PartitionedSegmentKind kind : {
             PartitionedSegmentKind::Initial,
             PartitionedSegmentKind::LegacyDeltaV1}) {
        TempDirectory temporary;
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(PartitionedSegmentSetWriter(
                    temporary.path(), spec(manifest(), kind)));
            },
            "unsupported segment kind was accepted");
    }
    {
        TempDirectory temporary;
        auto invalid = spec(manifest());
        invalid.creation_epoch = 0;
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(PartitionedSegmentSetWriter(
                    temporary.path(), invalid));
            },
            "zero non-initial creation epoch was accepted");
    }
    for (const std::uint32_t invalid_id : {
             SegmentId::kInvalid, SegmentId::kReserved}) {
        TempDirectory temporary;
        require_throws<throwdb::InvalidArgumentError>(
            [&] {
                static_cast<void>(PartitionedSegmentSetWriter(
                    temporary.path(), spec(
                        manifest(), PartitionedSegmentKind::Materialized,
                        invalid_id)));
            },
            "invalid first RowRef segment ID was accepted");
    }

    PartitionedManifestV2 huge = manifest();
    huge.target_segment_size = kRecordSize
        * (RowRef::kMaximumRowsPerSegment + 17);
    require(
        throwdb::detail::partitioned_segment_row_capacity(huge)
            == RowRef::kMaximumRowsPerSegment,
        "manifest target bypassed the 2^32 RowRef row cap");

    {
        TempDirectory temporary;
        PartitionedSegmentSetWriter writer(
            temporary.path(), spec(manifest()));
        const auto bytes = row(1);
        require_throws<throwdb::InvalidArgumentError>(
            [&] { static_cast<void>(writer.append(1, 0, bytes)); },
            "materialized segment accepted sequence zero");
        require(writer.poisoned(), "append failure did not poison writer");
        require_throws<throwdb::StorageError>(
            [&] { static_cast<void>(writer.append(1, 1, bytes)); },
            "poisoned writer accepted more rows");
        require(
            all_resource_count(temporary.path()) == 0,
            "poisoned writer left visible or staging output");
    }
    {
        TempDirectory temporary;
        PartitionedSegmentSetWriter writer(
            temporary.path(), spec(manifest()));
        const std::array<std::byte, 7> short_row{};
        require_throws<throwdb::InvalidArgumentError>(
            [&] { static_cast<void>(writer.append(1, 1, short_row)); },
            "short opaque row was accepted");
        require(writer.poisoned(), "row-size failure did not poison writer");
    }
    {
        TempDirectory temporary;
        const PartitionedManifestV2 one_row_manifest = manifest(kRecordSize);
        PartitionedSegmentSetWriter writer(
            temporary.path(), spec(
                one_row_manifest, PartitionedSegmentKind::Materialized,
                SegmentId::kMaximum));
        const auto bytes = row(2);
        static_cast<void>(writer.append(1, 1, bytes));
        require_throws<throwdb::InvalidArgumentError>(
            [&] { static_cast<void>(writer.append(2, 2, bytes)); },
            "contiguous segment allocation crossed reserved ID");
        require(writer.poisoned(), "segment-ID exhaustion did not poison writer");
        require(
            all_resource_count(temporary.path()) == 0,
            "ID-cap failure exposed a previously staged segment");
    }
}

[[nodiscard]] PartitionedSegmentDescriptor build_one(
    const std::filesystem::path& store,
    const PartitionedManifestV2& input_manifest) {
    PartitionedSegmentSetWriter writer(store, spec(input_manifest));
    const auto bytes = row(77);
    static_cast<void>(writer.append(123, 5, bytes));
    return writer.finish().front();
}

void flip_byte(const std::filesystem::path& path, const std::uint64_t offset) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot open corruption fixture");
    }
    std::byte value{};
    if (::pread(fd, &value, 1, static_cast<off_t>(offset)) != 1) {
        ::close(fd);
        throw std::runtime_error("cannot read corruption fixture");
    }
    value ^= std::byte{0x80};
    if (::pwrite(fd, &value, 1, static_cast<off_t>(offset)) != 1) {
        ::close(fd);
        throw std::runtime_error("cannot write corruption fixture");
    }
    ::close(fd);
}

void write_u64_at(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint64_t value) {
    std::array<std::byte, 8> bytes{};
    throwdb::detail::put_u64(bytes, 0, value);
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0
        || ::pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset))
               != static_cast<ssize_t>(bytes.size())) {
        if (fd >= 0) {
            ::close(fd);
        }
        throw std::runtime_error("cannot overwrite sidecar fixture");
    }
    ::close(fd);
}

void test_open_crc_geometry_catalog_and_symlink_validation() {
    const PartitionedManifestV2 input_manifest = manifest(4 * kRecordSize);
    {
        TempDirectory temporary;
        const auto descriptor = build_one(temporary.path(), input_manifest);
        const auto data_path = throwdb::detail::partitioned_segment_data_path(
            temporary.path(), descriptor.segment_id);
        flip_byte(data_path, throwdb::detail::kPartitionedHeaderSize);
        static_cast<void>(throwdb::detail::open_partitioned_segment(
            temporary.path(), input_manifest, descriptor, false));
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_partitioned_segment(
                    temporary.path(), input_manifest, descriptor, true));
            },
            "optional payload CRC did not detect changed row bits");
    }
    {
        TempDirectory temporary;
        const auto descriptor = build_one(temporary.path(), input_manifest);
        const auto ids_path = throwdb::detail::partitioned_segment_ids_path(
            temporary.path(), descriptor.segment_id);
        write_u64_at(
            ids_path,
            throwdb::detail::kPartitionedHeaderSize + 8,
            0);
        auto unchecked = throwdb::detail::open_partitioned_segment(
            temporary.path(), input_manifest, descriptor, false);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(unchecked->sidecar_at(0)); },
            "sidecar access accepted sequence outside descriptor bounds");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_partitioned_segment(
                    temporary.path(), input_manifest, descriptor, true));
            },
            "IDs payload CRC did not detect sidecar corruption");
    }
    {
        TempDirectory temporary;
        const auto descriptor = build_one(temporary.path(), input_manifest);
        PartitionedSegmentDescriptor wrong_catalog = descriptor;
        wrong_catalog.data_payload_crc32c ^= 1;
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_partitioned_segment(
                    temporary.path(), input_manifest, wrong_catalog, false));
            },
            "header/catalog descriptor mismatch was accepted");

        const auto data_path = throwdb::detail::partitioned_segment_data_path(
            temporary.path(), descriptor.segment_id);
        require(
            ::truncate(
                data_path.c_str(),
                static_cast<off_t>(descriptor.data_file_size - 1)) == 0,
            "cannot truncate geometry fixture");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_partitioned_segment(
                    temporary.path(), input_manifest, descriptor, false));
            },
            "truncated segment geometry was accepted");
    }
    {
        TempDirectory temporary;
        const auto descriptor = build_one(temporary.path(), input_manifest);
        const auto data_path = throwdb::detail::partitioned_segment_data_path(
            temporary.path(), descriptor.segment_id);
        const auto backup = data_path.string() + ".real";
        std::filesystem::rename(data_path, backup);
        require(
            ::symlink(backup.c_str(), data_path.c_str()) == 0,
            "cannot create segment symlink fixture");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_partitioned_segment(
                    temporary.path(), input_manifest, descriptor, false));
            },
            "O_NOFOLLOW segment open accepted a symlink");
    }
}

void test_unfinished_cleanup_and_fork_abandon() {
    const PartitionedManifestV2 input_manifest = manifest();
    {
        TempDirectory temporary;
        {
            PartitionedSegmentSetWriter unfinished(
                temporary.path(), spec(input_manifest));
            const auto bytes = row(1);
            static_cast<void>(unfinished.append(1, 1, bytes));
            require(
                final_resource_count(temporary.path()) == 0,
                "unfinished writer exposed final resources");
        }
        require(
            all_resource_count(temporary.path()) == 0,
            "unfinished writer destructor left staging resources");
    }

    TempDirectory temporary;
    PartitionedSegmentSetWriter writer(
        temporary.path(), spec(input_manifest));
    const auto first = row(2);
    const auto second = row(3);
    static_cast<void>(writer.append(10, 1, first));
    const pid_t writer_child = ::fork();
    if (writer_child < 0) {
        throw std::runtime_error("writer fork failed");
    }
    if (writer_child == 0) {
        bool rejected = false;
        try {
            static_cast<void>(writer.append(11, 2, second));
        } catch (const throwdb::StorageError& error) {
            rejected = std::string(error.what()).find("inherited across fork")
                != std::string::npos;
        } catch (...) {
        }
        writer.abandon_after_fork();
        ::_exit(rejected ? 0 : 1);
    }
    int writer_status = 0;
    require(
        ::waitpid(writer_child, &writer_status, 0) == writer_child
            && WIFEXITED(writer_status)
            && WEXITSTATUS(writer_status) == 0,
        "child used or cleaned inherited segment writer state");
    static_cast<void>(writer.append(11, 2, second));
    const PartitionedSegmentDescriptor descriptor = writer.finish().front();

    auto reader = throwdb::detail::open_partitioned_segment(
        temporary.path(), input_manifest, descriptor, true);
    const pid_t reader_child = ::fork();
    if (reader_child < 0) {
        throw std::runtime_error("reader fork failed");
    }
    if (reader_child == 0) {
        bool rejected = false;
        try {
            static_cast<void>(reader->sidecar_at(0));
        } catch (const throwdb::StorageError& error) {
            rejected = std::string(error.what()).find("inherited across fork")
                != std::string::npos;
        } catch (...) {
        }
        reader->abandon_after_fork();
        ::_exit(rejected ? 0 : 1);
    }
    int reader_status = 0;
    require(
        ::waitpid(reader_child, &reader_status, 0) == reader_child
            && WIFEXITED(reader_status)
            && WEXITSTATUS(reader_status) == 0,
        "child used inherited segment reader descriptors");
    require(
        reader->sidecar_at(0).id == 10
            && reader->sidecar_at(1).id == 11,
        "child abandon changed parent segment reader resources");
}

[[nodiscard]] std::filesystem::path staging_for(
    const std::filesystem::path& final_path) {
    return final_path.parent_path()
        / ("." + final_path.filename().string() + ".staging");
}

void write_text(
    const std::filesystem::path& path,
    const std::string& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) {
        throw std::runtime_error("cannot write ownership collision fixture");
    }
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void test_uuid_collisions_preserve_unowned_paths() {
    constexpr throwdb::detail::Uuid128 fixed_uuid{901, 902};
    const PartitionedManifestV2 input_manifest = manifest();
    {
        TempDirectory temporary;
        auto build_spec = spec(input_manifest);
        build_spec.segment_id_factory = [fixed_uuid] { return fixed_uuid; };
        PartitionedSegmentSetWriter writer(temporary.path(), build_spec);
        const auto bytes = row(40);
        static_cast<void>(writer.append(1, 1, bytes));
        const auto authoritative =
            throwdb::detail::partitioned_segment_data_path(
                temporary.path(), fixed_uuid);
        write_text(authoritative, "authoritative-final");
        require_throws<throwdb::StorageError>(
            [&] { static_cast<void>(writer.finish()); },
            "preexisting final UUID resource was overwritten");
        require(writer.poisoned(), "final UUID collision did not poison writer");
        require(
            read_text(authoritative) == "authoritative-final",
            "collision cleanup deleted or changed authoritative final bytes");
    }
    {
        TempDirectory temporary;
        auto build_spec = spec(input_manifest);
        build_spec.segment_id_factory = [fixed_uuid] { return fixed_uuid; };
        const auto ids_final =
            throwdb::detail::partitioned_segment_ids_path(
                temporary.path(), fixed_uuid);
        const auto foreign_staging = staging_for(ids_final);
        write_text(foreign_staging, "foreign-staging");
        {
            PartitionedSegmentSetWriter writer(temporary.path(), build_spec);
            const auto bytes = row(41);
            require_throws<throwdb::StorageError>(
                [&] { static_cast<void>(writer.append(2, 2, bytes)); },
                "preexisting staging resource was overwritten");
            require(
                writer.poisoned(),
                "staging UUID collision did not poison writer");
        }
        require(
            read_text(foreign_staging) == "foreign-staging",
            "collision cleanup deleted or changed unowned staging bytes");
        require(
            final_resource_count(temporary.path()) == 0,
            "staging collision published a partial final resource");
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"multi_segment_stream_and_bit_exact_resolution",
         test_multi_segment_stream_and_bit_exact_resolution},
        {"cleaner_zero_sequence_and_empty_finish",
         test_cleaner_zero_sequence_and_empty_finish},
        {"invalid_inputs_poison_and_rowref_capacity",
         test_invalid_inputs_poison_and_rowref_capacity},
        {"open_crc_geometry_catalog_and_symlink_validation",
         test_open_crc_geometry_catalog_and_symlink_validation},
        {"unfinished_cleanup_and_fork_abandon",
         test_unfinished_cleanup_and_fork_abandon},
        {"uuid_collisions_preserve_unowned_paths",
         test_uuid_collisions_preserve_unowned_paths},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " tests passed\n";
    return 0;
}
