#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sys/resource.h>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partitioned_index_run_storage.hpp"

namespace {

using namespace throwdb::detail;

constexpr std::uint32_t kPartitionBits = 2;
constexpr std::uint64_t kHashSeed = 0x123456789abcdef0ULL;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-partition-run-test-"
                   + std::to_string(
                       std::chrono::steady_clock::now()
                           .time_since_epoch().count())
                   + "-" + std::to_string(sequence++));
        std::filesystem::create_directories(
            partitioned_index_runs_directory(path_));
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

[[nodiscard]] Uuid128 uuid(const std::uint64_t value) {
    return Uuid128{value, value + 0x1000};
}

[[nodiscard]] PartitionIndexRunBuildSpec build_spec(
    const Uuid128 run_id = uuid(10)) {
    PartitionIndexRunBuildSpec spec;
    spec.collection_id = uuid(1);
    spec.schema_id = uuid(2);
    spec.partition_bits = kPartitionBits;
    spec.hash_seed = kHashSeed;
    spec.descriptor.run_id = run_id;
    spec.descriptor.epoch = 7;
    spec.descriptor.kind = PartitionedRunKind::Mutation;
    spec.descriptor.first_sequence = 3;
    spec.descriptor.last_sequence = 9;
    return spec;
}

[[nodiscard]] std::uint32_t partition_for(const std::uint64_t id) {
    return static_cast<std::uint32_t>(
        hash_id(id, kHashSeed) >> (64 - kPartitionBits));
}

[[nodiscard]] std::uint64_t id_for_partition(
    const std::uint32_t partition,
    const std::uint64_t occurrence = 0) {
    std::uint64_t seen = 0;
    for (std::uint64_t id = 0;; ++id) {
        if (partition_for(id) != partition) {
            continue;
        }
        if (seen++ == occurrence) {
            return id;
        }
    }
}

[[nodiscard]] std::vector<PartitionRunRecord> fixture_records() {
    return {
        {id_for_partition(3), 9, RowRef::create(12, 3).raw()},
        {id_for_partition(0, 1), 4, RowRef::create(11, 2).raw()},
        {id_for_partition(1), 5, 0},
        {id_for_partition(0), 3, RowRef::create(11, 1).raw()},
    };
}

[[nodiscard]] std::vector<std::byte> read_file(
    const std::filesystem::path& path) {
    const std::uint64_t size = std::filesystem::file_size(path);
    require(
        size <= std::numeric_limits<std::size_t>::max(),
        "test file does not fit memory");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open test file");
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(
        stream.gcount() == static_cast<std::streamsize>(bytes.size()),
        "cannot read complete test file");
    return bytes;
}

void write_file(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good(), "cannot open test file for writing");
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "cannot write complete test file");
}

void reseal_body(
    const std::filesystem::path& path,
    PartitionIndexRunDescriptor& descriptor) {
    std::vector<std::byte> bytes = read_file(path);
    PartitionIndexRunHeaderV1 header =
        decode_partition_index_run_header_v1(
            std::span<const std::byte>(bytes).first(kPartitionedHeaderSize));
    descriptor.body_crc32c = crc32c(
        std::span<const std::byte>(bytes).subspan(kPartitionedHeaderSize));
    header.descriptor.body_crc32c = descriptor.body_crc32c;
    const auto encoded = encode_partition_index_run_header_v1(header);
    std::copy(encoded.begin(), encoded.end(), bytes.begin());
    write_file(path, bytes);
}

void reseal_raw_header(const std::span<std::byte> header) {
    require_equal(
        header.size(), static_cast<std::size_t>(kPartitionedHeaderSize),
        "test header has invalid size");
    put_u32(header, 20, 0);
    put_u32(header, 20, crc32c(header));
}

[[nodiscard]] PartitionIndexRunDescriptor build_fixture(
    const std::filesystem::path& root,
    const Uuid128 run_id = uuid(10)) {
    return write_partitioned_index_run(
        root, build_spec(run_id), fixture_records());
}

[[nodiscard]] PartitionIndexRunReader open_fixture(
    const std::filesystem::path& root,
    const PartitionIndexRunDescriptor& descriptor) {
    const PartitionIndexRunBuildSpec spec = build_spec(descriptor.run_id);
    return PartitionIndexRunReader(
        root, spec.collection_id, spec.schema_id, spec.partition_bits,
        spec.hash_seed, descriptor);
}

void test_roundtrip_grouping_lookup_iteration_and_path() {
    TempDirectory temporary;
    const PartitionIndexRunDescriptor descriptor =
        build_fixture(temporary.path());
    require_equal(descriptor.entry_count, std::uint64_t{4},
                  "writer returned wrong entry count");
    require_equal(descriptor.covered_partitions,
                  std::vector<std::byte>{std::byte{0x0b}},
                  "writer derived wrong coverage");
    require_equal(descriptor.active_partitions,
                  descriptor.covered_partitions,
                  "new run did not activate every covered partition");
    require_equal(
        partitioned_index_run_path(temporary.path(), descriptor.run_id)
            .filename(),
        std::filesystem::path(
            "run-000000000000100a000000000000000a.idx"),
        "run UUID path is unstable");

    PartitionIndexRunReader reader =
        open_fixture(temporary.path(), descriptor);
    require_equal(reader.record_count(), std::uint64_t{4},
                  "reader returned wrong record count");
    require_equal(reader.partition_count(), std::uint64_t{4},
                  "reader returned wrong partition count");
    require_equal(reader.partition_records(2).record_count, std::uint64_t{0},
                  "empty partition is not empty");

    std::uint64_t previous_partition = 0;
    std::uint64_t previous_id = 0;
    bool first = true;
    for (std::uint64_t index = 0; index != reader.record_count(); ++index) {
        const PartitionRunRecord record = reader.record_at(index);
        const std::uint64_t partition = partition_for(record.id);
        require(
            first || partition > previous_partition
                || (partition == previous_partition && record.id > previous_id),
            "record iteration is not partition-grouped and ID-sorted");
        previous_partition = partition;
        previous_id = record.id;
        first = false;
    }
    const std::uint64_t tombstone_id = id_for_partition(1);
    const auto tombstone = reader.lookup(1, tombstone_id);
    require(tombstone.has_value() && tombstone->tombstone(),
            "canonical tombstone did not roundtrip");
    require(!reader.lookup(0, tombstone_id).has_value(),
            "lookup searched a partition to which the ID does not route");
    require(!reader.lookup(3, id_for_partition(3, 99)).has_value(),
            "lookup found an absent ID");

    PartitionIndexRunDescriptor retired_partition = descriptor;
    retired_partition.active_partitions[0] &= std::byte{0x09};
    PartitionIndexRunReader with_retired =
        open_fixture(temporary.path(), retired_partition);
    require(
        !with_retired.lookup(1, tombstone_id).has_value(),
        "lookup exposed a catalog-inactive partition");
    const PartitionRunTocEntry inactive = with_retired.partition_records(1);
    require_equal(
        with_retired.record_at(inactive.first_record).id, tombstone_id,
        "lifecycle iteration cannot inspect an inactive partition");

    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(reader.record_at(4)); },
        "out-of-range record index was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&] { static_cast<void>(reader.partition_records(4)); },
        "out-of-range partition was accepted");
}

void test_writer_rejects_duplicate_invalid_and_existing_destination() {
    TempDirectory temporary;
    const auto records = fixture_records();
    auto duplicate = records;
    duplicate.push_back(records.front());
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partitioned_index_run(
                temporary.path(), build_spec(uuid(20)), duplicate));
        },
        "duplicate ID was accepted");
    require(
        !std::filesystem::exists(
            partitioned_index_run_path(temporary.path(), uuid(20))),
        "invalid input created an immutable destination");

    auto invalid_sequence = records;
    invalid_sequence[0].sequence = 10;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partitioned_index_run(
                temporary.path(), build_spec(uuid(21)), invalid_sequence));
        },
        "out-of-range sequence was accepted");
    auto invalid_row_ref = records;
    invalid_row_ref[0].value_raw = 0xffffffff00000000ULL;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partitioned_index_run(
                temporary.path(), build_spec(uuid(22)), invalid_row_ref));
        },
        "invalid RowRef was accepted");

    const Uuid128 run_id = uuid(23);
    static_cast<void>(build_fixture(temporary.path(), run_id));
    require_throws<throwdb::StorageError>(
        [&] { static_cast<void>(build_fixture(temporary.path(), run_id)); },
        "immutable destination was overwritten");
}

void test_checksum_header_reserved_padding_and_size_corruption() {
    {
        TempDirectory temporary;
        const PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        bytes[kPartitionedHeaderSize] ^= std::byte{0x80};
        write_file(path, bytes);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "body checksum corruption was accepted");
    }
    {
        TempDirectory temporary;
        const PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        bytes[300] = std::byte{1};
        reseal_raw_header(
            std::span<std::byte>(bytes).first(kPartitionedHeaderSize));
        write_file(path, bytes);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "resealed nonzero header-reserved byte was accepted");
    }
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        const std::uint64_t toc_end =
            kPartitionedHeaderSize
            + (std::uint64_t{1} << kPartitionBits)
                  * kPartitionRunTocEntrySize;
        bytes[static_cast<std::size_t>(toc_end)] = std::byte{1};
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "resealed nonzero alignment padding was accepted");
    }
    {
        TempDirectory temporary;
        const PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::filesystem::resize_file(path, descriptor.file_size - 1);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "truncated run was accepted");
    }
}

void test_toc_coverage_and_catalog_subset_validation() {
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        put_u64(bytes, kPartitionedHeaderSize, 1);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "non-contiguous TOC was accepted");
    }
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        descriptor.covered_partitions[0] &= std::byte{0x09};
        descriptor.active_partitions = descriptor.covered_partitions;
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "catalog coverage mismatch was accepted");
    }
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        descriptor.active_partitions[0] |= std::byte{0x04};
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "active partition outside immutable coverage was accepted");
    }
}

void test_resealed_duplicate_wrong_partition_and_invalid_rowref() {
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        PartitionIndexRunReader reader =
            open_fixture(temporary.path(), descriptor);
        const PartitionRunTocEntry section = reader.partition_records(0);
        require_equal(section.record_count, std::uint64_t{2},
                      "duplicate test needs two records");
        const std::uint64_t records_offset =
            descriptor.file_size
            - descriptor.entry_count * kPartitionRunRecordSize;
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        const std::uint64_t first_id = get_u64(
            bytes, static_cast<std::size_t>(
                       records_offset
                       + section.first_record * kPartitionRunRecordSize));
        put_u64(
            bytes,
            static_cast<std::size_t>(
                records_offset
                + (section.first_record + 1) * kPartitionRunRecordSize),
            first_id);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "resealed duplicate ID was accepted");
    }
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        PartitionIndexRunReader reader =
            open_fixture(temporary.path(), descriptor);
        const PartitionRunTocEntry section = reader.partition_records(3);
        const std::uint64_t records_offset =
            descriptor.file_size
            - descriptor.entry_count * kPartitionRunRecordSize;
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        put_u64(
            bytes,
            static_cast<std::size_t>(
                records_offset
                + section.first_record * kPartitionRunRecordSize),
            id_for_partition(2));
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "record routed to the wrong partition was accepted");
    }
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        PartitionIndexRunReader reader =
            open_fixture(temporary.path(), descriptor);
        const PartitionRunTocEntry section = reader.partition_records(1);
        const std::uint64_t records_offset =
            descriptor.file_size
            - descriptor.entry_count * kPartitionRunRecordSize;
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        std::vector<std::byte> bytes = read_file(path);
        put_u64(
            bytes,
            static_cast<std::size_t>(
                records_offset
                + section.first_record * kPartitionRunRecordSize + 16),
            0xffffffff00000000ULL);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "invalid persisted RowRef was accepted as a tombstone");
    }
}

void test_scalar_binding_and_symlink_rejected() {
    {
        TempDirectory temporary;
        PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        ++descriptor.epoch;
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "catalog/header scalar mismatch was accepted");
    }
#if defined(O_NOFOLLOW)
    {
        TempDirectory temporary;
        const PartitionIndexRunDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_index_run_path(
            temporary.path(), descriptor.run_id);
        const auto target = path.string() + ".target";
        std::filesystem::rename(path, target);
        std::filesystem::create_symlink(target, path);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "symlinked run resource was followed");
    }
#endif
}

class RlimitGuard {
public:
    RlimitGuard() {
        if (::getrlimit(RLIMIT_NOFILE, &original_) != 0) {
            fail("getrlimit failed");
        }
    }

    ~RlimitGuard() {
        static_cast<void>(::setrlimit(RLIMIT_NOFILE, &original_));
    }

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

void test_mmap_readers_do_not_retain_file_descriptors() {
    TempDirectory temporary;
    const PartitionIndexRunDescriptor descriptor =
        build_fixture(temporary.path());
    RlimitGuard limit;
    limit.lower(32);
    std::vector<std::unique_ptr<PartitionIndexRunReader>> readers;
    for (std::size_t index = 0; index != 64; ++index) {
        const PartitionIndexRunBuildSpec spec = build_spec(descriptor.run_id);
        readers.push_back(std::make_unique<PartitionIndexRunReader>(
            temporary.path(), spec.collection_id, spec.schema_id,
            spec.partition_bits, spec.hash_seed, descriptor));
    }
    require_equal(readers.size(), std::size_t{64},
                  "low-FD reader construction stopped early");
    require(
        readers.back()->lookup(1, id_for_partition(1)).has_value(),
        "last low-FD reader cannot access its mapping");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 7> tests{{
        {"roundtrip_grouping_lookup_iteration_and_path",
         test_roundtrip_grouping_lookup_iteration_and_path},
        {"writer_rejects_duplicate_invalid_and_existing_destination",
         test_writer_rejects_duplicate_invalid_and_existing_destination},
        {"checksum_header_reserved_padding_and_size_corruption",
         test_checksum_header_reserved_padding_and_size_corruption},
        {"toc_coverage_and_catalog_subset_validation",
         test_toc_coverage_and_catalog_subset_validation},
        {"resealed_duplicate_wrong_partition_and_invalid_rowref",
         test_resealed_duplicate_wrong_partition_and_invalid_rowref},
        {"scalar_binding_and_symlink_rejected",
         test_scalar_binding_and_symlink_rejected},
        {"mmap_readers_do_not_retain_file_descriptors",
         test_mmap_readers_do_not_retain_file_descriptors},
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
