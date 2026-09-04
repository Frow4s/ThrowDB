#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/partition_snapshot_storage.hpp"
#include "../src/core/partitioned_index_merge.hpp"
#include "../src/core/partitioned_index_run_storage.hpp"

namespace {

using namespace throwdb::detail;

constexpr std::uint64_t kHashSeed = 0x123456789abcdef0ULL;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-partition-merge-test-"
                   + std::to_string(
                       std::chrono::steady_clock::now()
                           .time_since_epoch().count())
                   + "-" + std::to_string(sequence++));
        std::filesystem::create_directories(
            partitioned_index_runs_directory(path_));
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

[[nodiscard]] Uuid128 collection_id() { return uuid(1); }
[[nodiscard]] Uuid128 schema_id() { return uuid(2); }

[[nodiscard]] std::uint32_t partition_for(
    const std::uint64_t id,
    const std::uint32_t bits) {
    return bits == 0
               ? 0
               : static_cast<std::uint32_t>(
                     hash_id(id, kHashSeed) >> (64 - bits));
}

[[nodiscard]] std::uint64_t id_for(
    const std::uint32_t bits,
    const std::uint32_t partition,
    const std::uint64_t occurrence = 0) {
    std::uint64_t seen = 0;
    for (std::uint64_t id = 0;; ++id) {
        if (partition_for(id, bits) == partition && seen++ == occurrence) {
            return id;
        }
    }
}

[[nodiscard]] std::vector<PartitionRunRecord> partition_records(
    const std::uint32_t bits,
    const std::uint32_t partition,
    const std::size_t count,
    const std::uint64_t sequence,
    const std::uint32_t segment,
    const std::uint64_t occurrence_offset = 0) {
    std::vector<PartitionRunRecord> result;
    for (std::size_t index = 0; index != count; ++index) {
        result.push_back(PartitionRunRecord{
            id_for(bits, partition, occurrence_offset + index), sequence,
            RowRef::create(segment, index).raw()});
    }
    return result;
}

void append(
    std::vector<PartitionRunRecord>& destination,
    const std::vector<PartitionRunRecord>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

struct RunHandle {
    PartitionIndexRunDescriptor descriptor;
    std::unique_ptr<PartitionIndexRunReader> reader;
};

[[nodiscard]] RunHandle build_run(
    const std::filesystem::path& root,
    const std::uint32_t bits,
    const std::uint64_t token,
    const std::uint64_t epoch,
    const PartitionedRunKind kind,
    const std::span<const PartitionRunRecord> records,
    const std::uint32_t retired_segment = 0) {
    require(!records.empty(), "test run cannot be empty");
    PartitionIndexRunBuildSpec spec;
    spec.collection_id = collection_id();
    spec.schema_id = schema_id();
    spec.partition_bits = bits;
    spec.hash_seed = kHashSeed;
    spec.descriptor.run_id = uuid(token);
    spec.descriptor.epoch = epoch;
    spec.descriptor.kind = kind;
    spec.descriptor.first_sequence = records.front().sequence;
    spec.descriptor.last_sequence = records.front().sequence;
    for (const PartitionRunRecord& record : records) {
        spec.descriptor.first_sequence = std::min(
            spec.descriptor.first_sequence, record.sequence);
        spec.descriptor.last_sequence = std::max(
            spec.descriptor.last_sequence, record.sequence);
    }
    if (kind == PartitionedRunKind::Cleaner) {
        spec.descriptor.retired_row_ref_segment_id = retired_segment;
        spec.descriptor.retired_segment_id = uuid(token + 0x10000);
        spec.descriptor.retired_row_count = records.size();
        spec.descriptor.retired_ids_payload_crc32c =
            static_cast<std::uint32_t>(token);
    }
    RunHandle result;
    result.descriptor = write_partitioned_index_run(root, spec, records);
    result.reader = std::make_unique<PartitionIndexRunReader>(
        root, spec.collection_id, spec.schema_id, bits, spec.hash_seed,
        result.descriptor);
    return result;
}

[[nodiscard]] PartitionSnapshotDescriptor build_snapshot(
    const std::filesystem::path& root,
    const std::uint32_t bits,
    const std::uint32_t partition,
    const std::uint64_t token,
    const std::uint64_t epoch,
    const std::span<const PartitionRunRecord> records) {
    PartitionSnapshotBuildSpec spec;
    spec.collection_id = collection_id();
    spec.schema_id = schema_id();
    spec.snapshot_id = uuid(token);
    spec.partition_bits = bits;
    spec.partition_id = partition;
    spec.hash_seed = kHashSeed;
    spec.epoch = epoch;
    return write_partition_snapshot(root, spec, records);
}

[[nodiscard]] std::unique_ptr<PartitionSnapshotReader> open_snapshot(
    const std::filesystem::path& root,
    const std::uint32_t bits,
    const PartitionSnapshotDescriptor& descriptor) {
    return std::make_unique<PartitionSnapshotReader>(
        root, collection_id(), schema_id(), bits, kHashSeed, descriptor);
}

[[nodiscard]] std::vector<PartitionRunView> views(
    const std::vector<RunHandle>& runs) {
    std::vector<PartitionRunView> result;
    result.reserve(runs.size());
    for (const RunHandle& run : runs) {
        result.push_back(
            PartitionRunView{run.reader.get(), &run.descriptor});
    }
    return result;
}

[[nodiscard]] std::vector<PartitionSnapshotDescriptor> empty_snapshots(
    const std::filesystem::path& root,
    const std::uint32_t bits,
    const std::uint64_t epoch = 1) {
    const std::uint64_t count = std::uint64_t{1} << bits;
    std::vector<PartitionSnapshotDescriptor> result;
    result.reserve(static_cast<std::size_t>(count));
    const std::vector<PartitionRunRecord> empty;
    for (std::uint64_t partition = 0; partition != count; ++partition) {
        result.push_back(build_snapshot(
            root, bits, static_cast<std::uint32_t>(partition),
            1000 + partition, epoch, empty));
    }
    return result;
}

void test_selector_uses_all_sections_and_total_budget() {
    constexpr std::uint32_t bits = 2;
    TempDirectory temporary;
    const auto snapshots = empty_snapshots(temporary.path(), bits);
    std::vector<RunHandle> runs;
    {
        std::vector<PartitionRunRecord> records =
            partition_records(bits, 0, 1, 2, 10);
        append(records, partition_records(bits, 1, 3, 2, 10));
        runs.push_back(build_run(
            temporary.path(), bits, 200, 2,
            PartitionedRunKind::Mutation, records));
    }
    {
        std::vector<PartitionRunRecord> records =
            partition_records(bits, 0, 2, 3, 11, 10);
        append(records, partition_records(bits, 1, 1, 3, 11, 10));
        runs.push_back(build_run(
            temporary.path(), bits, 201, 3,
            PartitionedRunKind::Mutation, records));
    }
    runs.push_back(build_run(
        temporary.path(), bits, 202, 4,
        PartitionedRunKind::Mutation,
        partition_records(bits, 0, 4, 4, 12, 20)));
    const auto run_views = views(runs);

    PartitionFoldBudget budget{
        2, 2, 100,
        std::numeric_limits<std::uint64_t>::max()};
    auto selected = select_partition_for_fold(
        bits, snapshots, run_views, budget);
    require(selected.has_value() && selected->partition_id == 1,
            "selector partially folded an over-budget three-run partition");
    require_equal(selected->active_run_count, std::uint64_t{2},
                  "selector counted wrong active sections");
    require_equal(selected->run_entries, std::uint64_t{4},
                  "selector counted wrong run entries");
    require_equal(selected->input_entries, std::uint64_t{4},
                  "selector counted wrong total entries");
    require_equal(
        selected->input_bytes,
        snapshots[1].file_size + 4 * kPartitionRunRecordSize,
        "selector did not charge snapshot plus run bytes");

    budget.max_runs = 3;
    selected = select_partition_for_fold(bits, snapshots, run_views, budget);
    require(selected.has_value() && selected->partition_id == 0,
            "selector did not prioritize more active sections");

    budget.max_input_entries = 6;
    selected = select_partition_for_fold(bits, snapshots, run_views, budget);
    require(selected.has_value() && selected->partition_id == 1,
            "entry budget admitted an oversized complete partition");
    budget.max_input_entries = 100;
    budget.max_input_bytes = snapshots[1].file_size
                             + 4 * kPartitionRunRecordSize;
    selected = select_partition_for_fold(bits, snapshots, run_views, budget);
    require(selected.has_value() && selected->partition_id == 1,
            "byte budget did not use exact total input cost");

    std::vector<PartitionSnapshotDescriptor> charged = snapshots;
    charged[1].entry_count = 2;
    budget.max_input_bytes = std::numeric_limits<std::uint64_t>::max();
    budget.max_input_entries = 5;
    selected = select_partition_for_fold(bits, charged, run_views, budget);
    require(!selected.has_value(),
            "selector omitted current snapshot entries from its budget");

    std::vector<PartitionSnapshotDescriptor> stale = snapshots;
    stale[1].epoch = runs[1].descriptor.epoch;
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(select_partition_for_fold(
                bits, stale, run_views, PartitionFoldBudget{
                    1, 3, 100,
                    std::numeric_limits<std::uint64_t>::max()}));
        },
        "active run no newer than its snapshot was accepted");
}

void test_selector_deterministic_ties_and_zero_bit_layout() {
    constexpr std::uint32_t bits = 2;
    {
        TempDirectory temporary;
        const auto snapshots = empty_snapshots(temporary.path(), bits);
        std::vector<RunHandle> runs;
        for (std::uint64_t epoch = 2; epoch != 4; ++epoch) {
            std::vector<PartitionRunRecord> records = partition_records(
                bits, 1, 1, epoch, static_cast<std::uint32_t>(20 + epoch),
                epoch * 10);
            append(records, partition_records(
                bits, 2, epoch == 2 ? 3 : 2, epoch,
                static_cast<std::uint32_t>(20 + epoch), epoch * 10));
            runs.push_back(build_run(
                temporary.path(), bits, 300 + epoch, epoch,
                PartitionedRunKind::Mutation, records));
        }
        const auto run_views = views(runs);
        const auto selected = select_partition_for_fold(
            bits, snapshots, run_views,
            PartitionFoldBudget{
                2, 2, 100,
                std::numeric_limits<std::uint64_t>::max()});
        require(selected.has_value() && selected->partition_id == 2,
                "selector did not use entry count as second priority");
    }
    {
        TempDirectory temporary;
        const auto snapshots = empty_snapshots(temporary.path(), bits);
        std::vector<RunHandle> runs;
        for (std::uint64_t epoch = 2; epoch != 4; ++epoch) {
            std::vector<PartitionRunRecord> records = partition_records(
                bits, 1, 1, epoch, static_cast<std::uint32_t>(30 + epoch),
                epoch * 10);
            append(records, partition_records(
                bits, 2, 1, epoch,
                static_cast<std::uint32_t>(30 + epoch), epoch * 10));
            runs.push_back(build_run(
                temporary.path(), bits, 400 + epoch, epoch,
                PartitionedRunKind::Mutation, records));
        }
        const auto run_views = views(runs);
        const auto selected = select_partition_for_fold(
            bits, snapshots, run_views,
            PartitionFoldBudget{
                2, 2, 100,
                std::numeric_limits<std::uint64_t>::max()});
        require(selected.has_value() && selected->partition_id == 1,
                "selector did not use lower partition ID as final tie-break");
    }
    {
        TempDirectory temporary;
        const auto snapshots = empty_snapshots(temporary.path(), 0);
        std::vector<RunHandle> runs;
        runs.push_back(build_run(
            temporary.path(), 0, 500, 2,
            PartitionedRunKind::Mutation,
            partition_records(0, 0, 1, 2, 50)));
        const auto run_views = views(runs);
        const auto selected = select_partition_for_fold(
            0, snapshots, run_views,
            PartitionFoldBudget{
                1, 1, 10,
                std::numeric_limits<std::uint64_t>::max()});
        require(selected.has_value() && selected->partition_id == 0,
                "zero-bit partition layout was not selectable");
    }
}

struct MergeFixture {
    TempDirectory temporary;
    PartitionSnapshotDescriptor snapshot_descriptor;
    std::unique_ptr<PartitionSnapshotReader> snapshot;
    std::vector<RunHandle> runs;
    std::vector<PartitionRunView> run_views;
};

[[nodiscard]] MergeFixture build_relocation_fixture() {
    MergeFixture fixture;
    const std::array<PartitionRunRecord, 4> base{{
        {0, 1, RowRef::create(10, 0).raw()},
        {std::numeric_limits<std::uint64_t>::max(), 2,
         RowRef::create(20, 0).raw()},
        {42, 5, RowRef::create(30, 0).raw()},
        {100, 5, RowRef::create(31, 0).raw()},
    }};
    fixture.snapshot_descriptor = build_snapshot(
        fixture.temporary.path(), 0, 0, 600, 1, base);
    fixture.snapshot = open_snapshot(
        fixture.temporary.path(), 0, fixture.snapshot_descriptor);

    const std::array<PartitionRunRecord, 5> mutation{{
        {0, 5, RowRef::create(11, 0).raw()},
        {std::numeric_limits<std::uint64_t>::max(), 3, 0},
        {42, 6, 0},
        {100, 5, RowRef::create(31, 0).raw()},
        {7, 4, RowRef::create(40, 0).raw()},
    }};
    fixture.runs.push_back(build_run(
        fixture.temporary.path(), 0, 601, 2,
        PartitionedRunKind::Mutation, mutation));
    const std::array<PartitionRunRecord, 1> cleaner_one{{
        {0, 5, RowRef::create(12, 0).raw()},
    }};
    fixture.runs.push_back(build_run(
        fixture.temporary.path(), 0, 602, 3,
        PartitionedRunKind::Cleaner, cleaner_one, 11));
    const std::array<PartitionRunRecord, 1> idempotent_layer{{
        {0, 5, RowRef::create(12, 0).raw()},
    }};
    fixture.runs.push_back(build_run(
        fixture.temporary.path(), 0, 603, 4,
        PartitionedRunKind::Mutation, idempotent_layer));
    const std::array<PartitionRunRecord, 1> cleaner_two{{
        {0, 5, RowRef::create(13, 0).raw()},
    }};
    fixture.runs.push_back(build_run(
        fixture.temporary.path(), 0, 604, 5,
        PartitionedRunKind::Cleaner, cleaner_two, 12));
    fixture.run_views = views(fixture.runs);
    return fixture;
}

void test_merge_lookup_tombstones_extremes_and_cleaner_chain() {
    MergeFixture fixture = build_relocation_fixture();
    const auto zero = lookup_partition_version(
        0, 0, 0, *fixture.snapshot, fixture.run_views);
    require(
        zero.has_value() && !zero->tombstone()
            && zero->sequence == 5
            && RowRef::decode(zero->value_raw).segment_id().value() == 13,
        "point lookup did not follow repeated cleaner relocations");
    const auto maximum = lookup_partition_version(
        0, 0, std::numeric_limits<std::uint64_t>::max(),
        *fixture.snapshot, fixture.run_views);
    require(maximum.has_value() && maximum->tombstone()
                && maximum->sequence == 3,
            "point lookup lost a winning tombstone for extreme uint64 ID");
    require(
        !lookup_partition_version(
             0, 0, 999, *fixture.snapshot, fixture.run_views).has_value(),
        "point lookup invented an absent version");

    const std::vector<PartitionRunRecord> merged = merge_partition_for_fold(
        0, 0, *fixture.snapshot, fixture.run_views);
    require_equal(merged.size(), std::size_t{3},
                  "fold did not remove winning tombstones");
    require_equal(merged[0].id, std::uint64_t{0},
                  "fold output is not sorted by ID");
    require_equal(merged[1].id, std::uint64_t{7},
                  "fold output lost inserted live ID");
    require_equal(merged[2].id, std::uint64_t{100},
                  "fold output lost idempotent live ID");
    require_equal(
        RowRef::decode(merged[0].value_raw).segment_id().value(),
        std::uint32_t{13},
        "fold output did not retain final cleaner relocation");
}

[[nodiscard]] MergeFixture single_tie_fixture(
    const PartitionedRunKind kind,
    const std::uint64_t value_raw,
    const std::uint32_t retired_segment,
    const std::uint64_t token) {
    MergeFixture fixture;
    const std::array<PartitionRunRecord, 1> base{{
        {0, 5, RowRef::create(11, 0).raw()},
    }};
    fixture.snapshot_descriptor = build_snapshot(
        fixture.temporary.path(), 0, 0, token, 1, base);
    fixture.snapshot = open_snapshot(
        fixture.temporary.path(), 0, fixture.snapshot_descriptor);
    const std::array<PartitionRunRecord, 1> run{{{0, 5, value_raw}}};
    fixture.runs.push_back(build_run(
        fixture.temporary.path(), 0, token + 1, 2, kind, run,
        retired_segment));
    fixture.run_views = views(fixture.runs);
    return fixture;
}

void test_equal_sequence_conflicts_rejected() {
    {
        MergeFixture fixture = single_tie_fixture(
            PartitionedRunKind::Mutation,
            RowRef::create(12, 0).raw(), 0, 700);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(merge_partition_for_fold(
                    0, 0, *fixture.snapshot, fixture.run_views));
            },
            "mutation changed RowRef at equal sequence");
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(lookup_partition_version(
                    0, 0, 0, *fixture.snapshot, fixture.run_views));
            },
            "point reducer disagrees with fold reducer on a tie");
    }
    {
        MergeFixture fixture = single_tie_fixture(
            PartitionedRunKind::Cleaner,
            RowRef::create(12, 0).raw(), 99, 710);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(merge_partition_for_fold(
                    0, 0, *fixture.snapshot, fixture.run_views));
            },
            "cleaner relocated a RowRef from the wrong retired segment");
    }
    {
        MergeFixture fixture = single_tie_fixture(
            PartitionedRunKind::Cleaner, 0, 11, 720);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(merge_partition_for_fold(
                    0, 0, *fixture.snapshot, fixture.run_views));
            },
            "cleaner replaced a live value with a tombstone at equal sequence");
    }

}

void test_catalog_view_mask_is_independent_of_reused_reader() {
    constexpr std::uint32_t bits = 1;
    TempDirectory temporary;
    const auto base_records = partition_records(bits, 0, 1, 1, 80);
    const PartitionSnapshotDescriptor snapshot_descriptor = build_snapshot(
        temporary.path(), bits, 0, 800, 10, base_records);
    auto snapshot = open_snapshot(
        temporary.path(), bits, snapshot_descriptor);
    std::vector<PartitionRunRecord> records = partition_records(
        bits, 0, 1, 20, 81);
    append(records, partition_records(bits, 1, 1, 20, 81));
    RunHandle run = build_run(
        temporary.path(), bits, 801, 11,
        PartitionedRunKind::Mutation, records);
    PartitionIndexRunDescriptor narrowed = run.descriptor;
    narrowed.active_partitions[0] = std::byte{0x02};
    const std::array<PartitionRunView, 1> run_views{{
        {run.reader.get(), &narrowed},
    }};
    const auto merged = merge_partition_for_fold(
        bits, 0, *snapshot, run_views);
    require_equal(merged.size(), std::size_t{1},
                  "catalog-cleared section leaked from reused run reader");
    require_equal(merged.front().value_raw, base_records.front().value_raw,
                  "inactive reused section changed snapshot state");

    PartitionIndexRunDescriptor widened = run.descriptor;
    auto reader_with_narrow_mask = std::make_unique<PartitionIndexRunReader>(
        temporary.path(), collection_id(), schema_id(), bits, kHashSeed,
        narrowed);
    const std::array<PartitionRunView, 1> widened_view{{
        {reader_with_narrow_mask.get(), &widened},
    }};
    // The catalog view is authoritative. Raw TOC lookup must not consult the
    // narrower mask captured by the reused reader.
    const auto winning = lookup_partition_version(
        bits, 0, records.front().id, *snapshot, widened_view);
    require(winning.has_value() && winning->sequence == 20,
            "catalog-active section was hidden by reused reader mask");

    PartitionIndexRunDescriptor mismatched = widened;
    mismatched.body_crc32c ^= 1U;
    const std::array<PartitionRunView, 1> invalid_view{{
        {run.reader.get(), &mismatched},
    }};
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(merge_partition_for_fold(
                bits, 0, *snapshot, invalid_view));
        },
        "catalog descriptor was not bound to its reused reader resource");
}

[[nodiscard]] PartitionIndexRunDescriptor descriptor_with_bits(
    const std::uint64_t epoch,
    const std::vector<std::byte>& covered,
    const std::vector<std::byte>& active) {
    PartitionIndexRunDescriptor descriptor;
    descriptor.run_id = uuid(900 + epoch);
    descriptor.epoch = epoch;
    descriptor.kind = PartitionedRunKind::Mutation;
    descriptor.covered_partitions = covered;
    descriptor.active_partitions = active;
    return descriptor;
}

void test_transactional_bitmap_cleanup_and_extreme_layouts() {
    {
        std::vector<PartitionIndexRunDescriptor> runs{
            descriptor_with_bits(1, {std::byte{0x03}}, {std::byte{0x03}}),
            descriptor_with_bits(2, {std::byte{0x01}}, {std::byte{0x01}}),
            descriptor_with_bits(3, {std::byte{0x06}}, {std::byte{0x04}}),
        };
        const auto original_covered = runs.front().covered_partitions;
        clear_folded_partition(runs, 2, 0);
        require_equal(runs.size(), std::size_t{2},
                      "fully inactive run descriptor was not removed");
        require_equal(runs[0].epoch, std::uint64_t{1},
                      "cleanup changed descriptor order");
        require_equal(runs[0].covered_partitions, original_covered,
                      "cleanup changed immutable covered bitmap");
        require_equal(runs[0].active_partitions,
                      std::vector<std::byte>{std::byte{0x02}},
                      "cleanup did not clear selected active bit");
        require_equal(runs[1].epoch, std::uint64_t{3},
                      "cleanup reordered surviving descriptors");
    }
    {
        std::vector<PartitionIndexRunDescriptor> runs{
            descriptor_with_bits(1, {std::byte{0x01}}, {std::byte{0x01}}),
        };
        clear_folded_partition(runs, 0, 0);
        require(runs.empty(), "zero-bit cleanup retained an inactive run");
    }
    {
        std::vector<std::byte> bitmap(8192, std::byte{0});
        bitmap[0] = std::byte{0x01};
        bitmap.back() = std::byte{0x80};
        std::vector<PartitionIndexRunDescriptor> runs{
            descriptor_with_bits(1, bitmap, bitmap),
        };
        clear_folded_partition(runs, 16, 65535);
        require_equal(runs.size(), std::size_t{1},
                      "16-bit cleanup removed a still-active run");
        require_equal(runs.front().active_partitions.front(), std::byte{0x01},
                      "16-bit cleanup changed low partition bit");
        require_equal(runs.front().active_partitions.back(), std::byte{0x00},
                      "16-bit cleanup did not clear partition 65535");
    }
    {
        std::vector<PartitionIndexRunDescriptor> invalid{
            descriptor_with_bits(2, {std::byte{0x01}}, {std::byte{0x01}}),
            descriptor_with_bits(1, {std::byte{0x01}}, {std::byte{0x01}}),
        };
        const auto original = invalid;
        require_throws<throwdb::CorruptionError>(
            [&] { clear_folded_partition(invalid, 2, 0); },
            "out-of-order run descriptors were cleaned");
        require_equal(invalid, original,
                      "failed cleanup modified its input descriptors");

        invalid = {
            descriptor_with_bits(1, {std::byte{0x01}}, {std::byte{0x02}}),
        };
        const auto subset_original = invalid;
        require_throws<throwdb::CorruptionError>(
            [&] { clear_folded_partition(invalid, 2, 0); },
            "active bitmap outside coverage was cleaned");
        require_equal(invalid, subset_original,
                      "subset validation failure modified input descriptors");
    }
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 7> tests{{
        {"selector_uses_all_sections_and_total_budget",
         test_selector_uses_all_sections_and_total_budget},
        {"selector_deterministic_ties_and_zero_bit_layout",
         test_selector_deterministic_ties_and_zero_bit_layout},
        {"merge_lookup_tombstones_extremes_and_cleaner_chain",
         test_merge_lookup_tombstones_extremes_and_cleaner_chain},
        {"equal_sequence_conflicts_rejected",
         test_equal_sequence_conflicts_rejected},
        {"catalog_view_mask_is_independent_of_reused_reader",
         test_catalog_view_mask_is_independent_of_reused_reader},
        {"transactional_bitmap_cleanup_and_extreme_layouts",
         test_transactional_bitmap_cleanup_and_extreme_layouts},
        {"repeat_selector_is_deterministic", [] {
             constexpr std::uint32_t bits = 0;
             TempDirectory temporary;
             const auto snapshots = empty_snapshots(temporary.path(), bits);
             std::vector<RunHandle> runs;
             runs.push_back(build_run(
                 temporary.path(), bits, 999, 2,
                 PartitionedRunKind::Mutation,
                 partition_records(bits, 0, 1, 2, 99)));
             const auto run_views = views(runs);
             const PartitionFoldBudget budget{
                 1, 1, 10,
                 std::numeric_limits<std::uint64_t>::max()};
             require_equal(
                 select_partition_for_fold(
                     bits, snapshots, run_views, budget),
                 select_partition_for_fold(
                     bits, snapshots, run_views, budget),
                 "selector result changed across identical calls");
         }},
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
