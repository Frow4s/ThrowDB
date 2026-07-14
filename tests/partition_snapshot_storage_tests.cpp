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
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partition_snapshot_storage.hpp"

namespace {

using namespace throwdb::detail;

constexpr std::uint32_t kPartitionBits = 2;
constexpr std::uint32_t kPartition = 1;
constexpr std::uint64_t kHashSeed = 0x123456789abcdef0ULL;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-partition-snapshot-test-"
                   + std::to_string(
                       std::chrono::steady_clock::now()
                           .time_since_epoch().count())
                   + "-" + std::to_string(sequence++));
        std::filesystem::create_directories(
            partitioned_partition_snapshots_directory(path_));
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

[[nodiscard]] PartitionSnapshotBuildSpec build_spec(
    const Uuid128 snapshot_id = uuid(10)) {
    PartitionSnapshotBuildSpec spec;
    spec.collection_id = uuid(1);
    spec.schema_id = uuid(2);
    spec.snapshot_id = snapshot_id;
    spec.partition_bits = kPartitionBits;
    spec.partition_id = kPartition;
    spec.hash_seed = kHashSeed;
    spec.epoch = 7;
    return spec;
}

[[nodiscard]] std::uint32_t partition_for(const std::uint64_t id) {
    return static_cast<std::uint32_t>(
        hash_id(id, kHashSeed) >> (64 - kPartitionBits));
}

[[nodiscard]] std::uint64_t id_for(
    const std::uint32_t partition,
    const std::uint64_t occurrence = 0) {
    std::uint64_t seen = 0;
    for (std::uint64_t id = 0;; ++id) {
        if (partition_for(id) == partition && seen++ == occurrence) {
            return id;
        }
    }
}

[[nodiscard]] std::vector<std::uint64_t> colliding_ids(
    const std::size_t count,
    const std::uint64_t home = 3) {
    std::vector<std::uint64_t> ids;
    for (std::uint64_t id = 0; ids.size() != count; ++id) {
        const std::uint64_t hash = hash_id(id, kHashSeed);
        if (partition_for(id) == kPartition && (hash & 7) == home) {
            ids.push_back(id);
        }
    }
    return ids;
}

[[nodiscard]] std::vector<PartitionRunRecord> fixture_records() {
    const auto ids = colliding_ids(4);
    return {
        {ids[2], 5, RowRef::create(12, 2).raw()},
        {ids[0], 0, RowRef::create(11, 0).raw()},
        {ids[3], 9, RowRef::create(12, 3).raw()},
        {ids[1], 3, RowRef::create(11, 1).raw()},
    };
}

[[nodiscard]] std::vector<PartitionRunRecord> records_for_capacity(
    const std::size_t count) {
    std::vector<PartitionRunRecord> records;
    for (std::size_t index = 0; index != count; ++index) {
        records.push_back(PartitionRunRecord{
            id_for(kPartition, index),
            static_cast<std::uint64_t>(index),
            RowRef::create(20, index).raw()});
    }
    return records;
}

[[nodiscard]] std::vector<std::byte> read_file(
    const std::filesystem::path& path) {
    const std::uint64_t size = std::filesystem::file_size(path);
    require(size <= std::numeric_limits<std::size_t>::max(),
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
    PartitionSnapshotDescriptor& descriptor) {
    std::vector<std::byte> bytes = read_file(path);
    PartitionSnapshotHeaderV1 header = decode_partition_snapshot_header_v1(
        std::span<const std::byte>(bytes).first(kPartitionedHeaderSize));
    descriptor.body_crc32c = crc32c(
        std::span<const std::byte>(bytes).subspan(kPartitionedHeaderSize));
    header.descriptor.body_crc32c = descriptor.body_crc32c;
    const auto encoded = encode_partition_snapshot_header_v1(header);
    std::copy(encoded.begin(), encoded.end(), bytes.begin());
    write_file(path, bytes);
}

void reseal_raw_header(const std::span<std::byte> header) {
    put_u32(header, 20, 0);
    put_u32(header, 20, crc32c(header));
}

[[nodiscard]] PartitionSnapshotDescriptor build_fixture(
    const std::filesystem::path& root,
    const Uuid128 snapshot_id = uuid(10)) {
    return write_partition_snapshot(
        root, build_spec(snapshot_id), fixture_records());
}

[[nodiscard]] PartitionSnapshotReader open_fixture(
    const std::filesystem::path& root,
    const PartitionSnapshotDescriptor& descriptor) {
    const PartitionSnapshotBuildSpec spec = build_spec(descriptor.snapshot_id);
    return PartitionSnapshotReader(
        root, spec.collection_id, spec.schema_id, spec.partition_bits,
        spec.hash_seed, descriptor);
}

[[nodiscard]] std::vector<std::uint64_t> occupied_slots(
    const std::vector<std::byte>& bytes,
    const PartitionSnapshotDescriptor& descriptor) {
    std::vector<std::uint64_t> result;
    for (std::uint64_t slot = 0; slot != descriptor.capacity; ++slot) {
        if (bytes[static_cast<std::size_t>(kPartitionedHeaderSize + slot)]
            == std::byte{1}) {
            result.push_back(slot);
        }
    }
    return result;
}

void test_empty_and_mixed_sequence_roundtrip() {
    {
        TempDirectory temporary;
        const PartitionSnapshotBuildSpec spec = build_spec(uuid(30));
        const std::vector<PartitionRunRecord> empty;
        const PartitionSnapshotDescriptor descriptor =
            write_partition_snapshot(temporary.path(), spec, empty);
        require_equal(descriptor.entry_count, std::uint64_t{0},
                      "empty snapshot has records");
        require_equal(descriptor.capacity, std::uint64_t{8},
                      "empty snapshot has noncanonical capacity");
        PartitionSnapshotReader reader(
            temporary.path(), spec.collection_id, spec.schema_id,
            spec.partition_bits, spec.hash_seed, descriptor);
        require_equal(reader.record_count(), std::uint64_t{0},
                      "empty reader has records");
        for (std::uint64_t slot = 0; slot != reader.slot_count(); ++slot) {
            require(!reader.slot_record(slot).has_value(),
                    "empty snapshot contains an occupied slot");
        }
        require(!reader.lookup(id_for(kPartition)).has_value(),
                "empty snapshot found an ID");

        // The writer reserves epoch zero for initial construction, but the
        // reusable reader must be able to consume an initial snapshot as the
        // first selective-fold input.
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        PartitionSnapshotHeaderV1 initial_header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        PartitionSnapshotDescriptor initial_descriptor = descriptor;
        initial_descriptor.epoch = 0;
        initial_header.descriptor.epoch = 0;
        const auto initial_bytes =
            encode_partition_snapshot_header_v1(initial_header);
        std::copy(initial_bytes.begin(), initial_bytes.end(), bytes.begin());
        write_file(path, bytes);
        PartitionSnapshotReader initial_reader(
            temporary.path(), spec.collection_id, spec.schema_id,
            spec.partition_bits, spec.hash_seed, initial_descriptor);
        require_equal(initial_reader.descriptor().epoch, std::uint64_t{0},
                      "reader rejected an initial epoch-zero snapshot");
    }
    {
        TempDirectory temporary;
        const auto input = fixture_records();
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        require_equal(descriptor.entry_count, std::uint64_t{4},
                      "mixed snapshot has wrong count");
        require_equal(descriptor.capacity, std::uint64_t{8},
                      "mixed snapshot chose wrong capacity");
        require_equal(
            partitioned_partition_snapshot_path(
                temporary.path(), descriptor).filename(),
            std::filesystem::path(
                "partition-1-000000000000100a000000000000000a.idx"),
            "snapshot UUID path is unstable");
        PartitionSnapshotReader reader =
            open_fixture(temporary.path(), descriptor);
        std::size_t iterated = 0;
        for (std::uint64_t slot = 0; slot != reader.slot_count(); ++slot) {
            if (reader.slot_record(slot).has_value()) {
                ++iterated;
            }
        }
        require_equal(iterated, input.size(),
                      "slot iteration lost records");
        for (const PartitionRunRecord& expected : input) {
            const auto actual = reader.lookup(expected.id);
            require(actual.has_value(), "lookup missed a live record");
            require_equal(actual->id, expected.id, "lookup changed ID");
            require_equal(actual->sequence, expected.sequence,
                          "lookup changed sequence");
            require_equal(actual->row_ref.raw(), expected.value_raw,
                          "lookup changed RowRef");
        }
        require(!reader.lookup(id_for(2)).has_value(),
                "lookup searched an ID routed to another partition");
        require_throws<throwdb::InvalidArgumentError>(
            [&] { static_cast<void>(reader.slot_record(reader.slot_count())); },
            "out-of-range slot was accepted");
    }
}

void test_capacity_and_collision_probe_chain() {
    TempDirectory temporary;
    const auto records = records_for_capacity(6);
    const PartitionSnapshotBuildSpec spec = build_spec(uuid(31));
    const PartitionSnapshotDescriptor descriptor = write_partition_snapshot(
        temporary.path(), spec, records);
    require_equal(descriptor.capacity, std::uint64_t{16},
                  "70-percent capacity boundary is wrong");

    TempDirectory collision_directory;
    const PartitionSnapshotDescriptor collision =
        build_fixture(collision_directory.path(), uuid(32));
    PartitionSnapshotReader reader =
        open_fixture(collision_directory.path(), collision);
    for (const PartitionRunRecord& record : fixture_records()) {
        require(reader.lookup(record.id).has_value(),
                "linear probe lookup missed a collision-chain record");
    }
}

void test_writer_rejects_invalid_input_and_immutable_destination() {
    TempDirectory temporary;
    const auto valid = fixture_records();
    auto duplicate = valid;
    duplicate.push_back(valid.front());
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), build_spec(uuid(40)), duplicate));
        },
        "duplicate snapshot ID was accepted");
    auto wrong_partition = valid;
    wrong_partition[0].id = id_for(2);
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), build_spec(uuid(41)), wrong_partition));
        },
        "wrong-partition snapshot ID was accepted");
    auto tombstone = valid;
    tombstone[0].value_raw = 0;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), build_spec(uuid(42)), tombstone));
        },
        "snapshot tombstone was accepted");
    auto bad_sequence = valid;
    bad_sequence[0].sequence = std::numeric_limits<std::uint64_t>::max();
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), build_spec(uuid(43)), bad_sequence));
        },
        "reserved snapshot sequence was accepted");
    auto invalid_rowref = valid;
    invalid_rowref[0].value_raw = 0xffffffff00000000ULL;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), build_spec(uuid(44)), invalid_rowref));
        },
        "invalid snapshot RowRef was accepted");
    PartitionSnapshotBuildSpec zero_epoch = build_spec(uuid(45));
    zero_epoch.epoch = 0;
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(write_partition_snapshot(
                temporary.path(), zero_epoch, valid));
        },
        "selective snapshot epoch zero was accepted");

    const Uuid128 snapshot_id = uuid(46);
    static_cast<void>(build_fixture(temporary.path(), snapshot_id));
    require_throws<throwdb::StorageError>(
        [&] { static_cast<void>(build_fixture(temporary.path(), snapshot_id)); },
        "immutable snapshot was overwritten");
}

void test_checksum_header_padding_and_size_corruption() {
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        bytes[kPartitionedHeaderSize] ^= std::byte{0x80};
        write_file(path, bytes);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "snapshot body checksum damage was accepted");
    }
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        bytes[300] = std::byte{1};
        reseal_raw_header(
            std::span<std::byte>(bytes).first(kPartitionedHeaderSize));
        write_file(path, bytes);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "resealed reserved header byte was accepted");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        bytes[static_cast<std::size_t>(
            kPartitionedHeaderSize + descriptor.capacity)] = std::byte{1};
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "resealed nonzero alignment padding was accepted");
    }
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::filesystem::resize_file(path, descriptor.file_size - 1);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "truncated snapshot was accepted");
    }
}

void test_resealed_control_empty_duplicate_routing_and_probe_corruption() {
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const auto occupied = occupied_slots(bytes, descriptor);
        bytes[static_cast<std::size_t>(
            kPartitionedHeaderSize + occupied.front())] = std::byte{2};
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "invalid snapshot control byte was accepted");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        std::uint64_t empty = 0;
        while (bytes[static_cast<std::size_t>(
                   kPartitionedHeaderSize + empty)] != std::byte{0}) {
            ++empty;
        }
        put_u64(bytes, static_cast<std::size_t>(
            header.lookup_slots_offset
            + empty * kPartitionSnapshotLookupSlotSize), 99);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "nonzero empty snapshot slot was accepted");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        const auto occupied = occupied_slots(bytes, descriptor);
        const std::uint64_t first_id = get_u64(bytes,
            static_cast<std::size_t>(header.lookup_slots_offset
                + occupied[0] * kPartitionSnapshotLookupSlotSize));
        put_u64(bytes, static_cast<std::size_t>(header.lookup_slots_offset
            + occupied[1] * kPartitionSnapshotLookupSlotSize), first_id);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "duplicate snapshot ID was accepted after resealing");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        const auto occupied = occupied_slots(bytes, descriptor);
        put_u64(bytes, static_cast<std::size_t>(header.lookup_slots_offset
            + occupied.front() * kPartitionSnapshotLookupSlotSize), id_for(2));
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "wrong-partition snapshot ID was accepted after resealing");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        const auto occupied = occupied_slots(bytes, descriptor);
        const std::uint64_t home = occupied.front();
        bytes[static_cast<std::size_t>(kPartitionedHeaderSize + home)] =
            std::byte{0};
        const std::size_t lookup = static_cast<std::size_t>(
            header.lookup_slots_offset
            + home * kPartitionSnapshotLookupSlotSize);
        put_u64(bytes, lookup, 0);
        put_u64(bytes, lookup + 8, 0);
        put_u64(bytes, static_cast<std::size_t>(
            header.sequence_slots_offset
            + home * kPartitionSnapshotSequenceSlotSize), 0);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "probe-unreachable snapshot slot was accepted");
    }
}

void test_resealed_sequence_and_rowref_corruption() {
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        const auto occupied = occupied_slots(bytes, descriptor);
        put_u64(bytes, static_cast<std::size_t>(
            header.sequence_slots_offset
            + occupied.front() * kPartitionSnapshotSequenceSlotSize),
            std::numeric_limits<std::uint64_t>::max());
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "reserved snapshot sequence was accepted after resealing");
    }
    {
        TempDirectory temporary;
        PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        std::vector<std::byte> bytes = read_file(path);
        const PartitionSnapshotHeaderV1 header =
            decode_partition_snapshot_header_v1(
                std::span<const std::byte>(bytes).first(
                    kPartitionedHeaderSize));
        const auto occupied = occupied_slots(bytes, descriptor);
        put_u64(bytes, static_cast<std::size_t>(header.lookup_slots_offset
            + occupied.front() * kPartitionSnapshotLookupSlotSize + 8),
            0xffffffff00000000ULL);
        write_file(path, bytes);
        reseal_body(path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "invalid snapshot RowRef was accepted after resealing");
    }
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

void test_symlink_and_low_fd_behavior() {
#if defined(O_NOFOLLOW)
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        const auto path = partitioned_partition_snapshot_path(
            temporary.path(), descriptor);
        const auto target = path.string() + ".target";
        std::filesystem::rename(path, target);
        std::filesystem::create_symlink(target, path);
        require_throws<throwdb::CorruptionError>(
            [&] { static_cast<void>(open_fixture(temporary.path(), descriptor)); },
            "symlinked snapshot was followed");
    }
#endif
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        RlimitGuard limit;
        limit.lower(32);
        std::vector<std::unique_ptr<PartitionSnapshotReader>> readers;
        for (std::size_t index = 0; index != 64; ++index) {
            const auto spec = build_spec(descriptor.snapshot_id);
            readers.push_back(std::make_unique<PartitionSnapshotReader>(
                temporary.path(), spec.collection_id, spec.schema_id,
                spec.partition_bits, spec.hash_seed, descriptor));
        }
        require(
            readers.back()->lookup(fixture_records().front().id).has_value(),
            "low-FD snapshot mapping is unusable");
    }
    {
        TempDirectory temporary;
        const PartitionSnapshotDescriptor descriptor =
            build_fixture(temporary.path());
        PartitionSnapshotReader reader =
            open_fixture(temporary.path(), descriptor);
        const std::uint64_t id = fixture_records().front().id;
        const pid_t child = ::fork();
        require(child >= 0, "fork failed");
        if (child == 0) {
            const bool found = reader.lookup(id).has_value();
            ::_exit(found ? 0 : 1);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child,
                "waitpid failed");
        require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "forked child could not use the immutable mapping");
        require(reader.lookup(id).has_value(),
                "child teardown damaged the parent mapping");
    }
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 7> tests{{
        {"empty_and_mixed_sequence_roundtrip",
         test_empty_and_mixed_sequence_roundtrip},
        {"capacity_and_collision_probe_chain",
         test_capacity_and_collision_probe_chain},
        {"writer_rejects_invalid_input_and_immutable_destination",
         test_writer_rejects_invalid_input_and_immutable_destination},
        {"checksum_header_padding_and_size_corruption",
         test_checksum_header_padding_and_size_corruption},
        {"resealed_control_empty_duplicate_routing_and_probe_corruption",
         test_resealed_control_empty_duplicate_routing_and_probe_corruption},
        {"resealed_sequence_and_rowref_corruption",
         test_resealed_sequence_and_rowref_corruption},
        {"symlink_and_low_fd_behavior", test_symlink_and_low_fd_behavior},
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
