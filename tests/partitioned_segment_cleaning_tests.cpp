#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/native_collection_test_hooks.hpp"
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
                / ("throwdb-cleaner-" + label + "-"
                   + std::to_string(::getpid()) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

constexpr std::uint32_t kDimension = 128;
constexpr std::uint64_t kRecordSize = kDimension * 2ULL;
constexpr std::uint64_t kSegmentRows = 4;

[[nodiscard]] std::vector<std::byte> make_rows(
    const std::size_t count,
    const std::uint8_t salt = 0) {
    std::vector<std::byte> result(count * kRecordSize);
    for (std::size_t row = 0; row != count; ++row) {
        for (std::size_t byte = 0; byte != kRecordSize; ++byte) {
            result[row * kRecordSize + byte] = static_cast<std::byte>(
                (row * 37 + byte * 11 + salt) & 0xffU);
        }
    }
    return result;
}

void build_collection(const std::filesystem::path& path) {
    const std::array<std::uint64_t, 8> ids{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::byte> embeddings = make_rows(ids.size(), 3);
    throwdb::NativeBuilderOptions options;
    options.dimension = kDimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = kSegmentRows * kRecordSize;
    options.format_version = 7;
    options.partition_bits = 1;
    auto builder = throwdb::NativeCollectionBuilder::create(path, options);
    builder->append(ids, embeddings);
    builder->finish();
}

[[nodiscard]] throwdb::NativeSegmentCleaningOptions clean_options(
    const bool dry_run) {
    throwdb::NativeSegmentCleaningOptions options;
    options.dry_run = dry_run;
    options.max_candidate_segments = 16;
    options.max_sidecar_scan_bytes = 1ULL << 20;
    options.max_source_bytes = 1ULL << 20;
    options.max_output_bytes = 1ULL << 20;
    options.max_memory_bytes = 1ULL << 20;
    options.min_garbage_rows = 1;
    options.min_reclaimable_bytes = 0;
    options.min_garbage_ratio_ppm = 0;
    options.start_after_segment_id = 0;
    return options;
}

void create_garbage(
    const std::filesystem::path& path,
    std::unique_ptr<throwdb::NativeCollection>& writer) {
    writer = throwdb::NativeCollection::open(path, {}, true);
    const std::array<std::uint64_t, 3> deleted_ids{1, 2, 3};
    const std::vector<std::uint8_t> deleted = writer->erase(deleted_ids);
    require(
        deleted == std::vector<std::uint8_t>({1, 1, 1}),
        "fixture delete did not remove three base IDs");
    require(writer->rotate_wal().rotated, "fixture WAL did not rotate");
    require(
        writer->materialize_frozen_wals().materialized,
        "fixture WAL did not materialize");
}

[[nodiscard]] std::vector<std::byte> read_file(
    const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    require(static_cast<bool>(input), "cannot read test metadata");
    return result;
}

[[nodiscard]] std::string uuid_token(
    const throwdb::detail::Uuid128 value) {
    constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> result{};
    const auto put = [&result](const std::uint64_t word,
                               const std::size_t begin) {
        for (std::size_t nibble = 0; nibble != 16; ++nibble) {
            result[begin + nibble] = hex[
                (word >> static_cast<unsigned>((15 - nibble) * 4)) & 0xfU];
        }
    };
    put(value.high, 0);
    put(value.low, 16);
    return std::string(result.begin(), result.end());
}

[[nodiscard]] throwdb::detail::PartitionedCatalogV3 current_catalog(
    const std::filesystem::path& path) {
    using namespace throwdb::detail;
    const PartitionedManifestV2 manifest = decode_partitioned_manifest_v2(
        read_file(path / "manifest.bin"));
    const PartitionedCurrentV1 current = decode_partitioned_current_v1(
        read_file(path / "CURRENT"), manifest);
    return decode_partitioned_catalog_v3(read_file(
        path / "catalogs"
        / ("catalog-" + uuid_token(current.catalog_id) + ".bin")));
}

[[nodiscard]] std::vector<std::byte> row_four() {
    const std::vector<std::byte> base = make_rows(8, 3);
    return std::vector<std::byte>(
        base.begin() + static_cast<std::ptrdiff_t>(3 * kRecordSize),
        base.begin() + static_cast<std::ptrdiff_t>(4 * kRecordSize));
}

void test_dry_run_execute_reopen_pin_and_cleaner_chain() {
    TemporaryDirectory temporary("end-to-end");
    build_collection(temporary.path());
    std::unique_ptr<throwdb::NativeCollection> writer;
    create_garbage(temporary.path(), writer);
    const std::uint64_t before_generation =
        writer->mutation_log_stats().generation;

    throwdb::NativeSegmentCleaningOptions blocked = clean_options(true);
    blocked.max_sidecar_scan_bytes = 1;
    const throwdb::NativeSegmentCleaningStats blocked_stats =
        writer->clean_segments(blocked);
    require(
        !blocked_stats.candidate_found && blocked_stats.budget_blocked
            && writer->mutation_log_stats().generation == before_generation,
        "bounded dry-run changed the catalog or ignored its scan budget");

    const throwdb::NativeSegmentCleaningStats plan =
        writer->clean_segments(clean_options(true));
    require(
        plan.dry_run && plan.candidate_found && !plan.cleaned
            && plan.candidate_segment_id == 1 && plan.source_rows == 4
            && plan.relocated_rows == 1 && plan.garbage_rows == 3
            && plan.output_segment_count == 1
            && writer->mutation_log_stats().generation == before_generation,
        "cleaner dry-run chose the wrong deterministic candidate");

    auto old_reader = throwdb::NativeCollection::open(temporary.path());
    const std::array<std::uint64_t, 1> id_four{4};
    require(
        old_reader->get_bytes(id_four) == row_four(),
        "old reader fixture returned the wrong row");

    const throwdb::NativeSegmentCleaningStats cleaned =
        writer->clean_segments(clean_options(false));
    require(
        cleaned.cleaned && !cleaned.reclamation_complete
            && cleaned.candidate_segment_id == 1,
        "cleaner did not publish the selected source retirement");
    const throwdb::detail::PartitionedCatalogV3 catalog =
        current_catalog(temporary.path());
    require(
        std::none_of(
            catalog.segments.begin(), catalog.segments.end(),
            [](const auto& descriptor) {
                return descriptor.row_ref_segment_id == 1;
            }),
        "cleaner left the retired initial segment authoritative");
    const auto replacement = std::find_if(
        catalog.segments.begin(), catalog.segments.end(),
        [](const auto& descriptor) {
            return descriptor.kind
                   == throwdb::detail::PartitionedSegmentKind::Cleaner;
        });
    require(
        replacement != catalog.segments.end()
            && catalog.runs.back().kind
                   == throwdb::detail::PartitionedRunKind::Cleaner
            && catalog.runs.back().retired_row_ref_segment_id == 1
            && catalog.runs.back().entry_count == 4,
        "cleaner catalog lacks its replacement/fence binding");

    throwdb::ReaderOptions verified_options;
    verified_options.verify_native_index = true;
    auto verified = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        verified->count() == 5
            && verified->get_bytes(id_four) == row_four(),
        "verified reopen changed cleaned logical state");
    require_throws<throwdb::MissingIdError>(
        [&] {
            const std::array<std::uint64_t, 1> missing{1};
            static_cast<void>(verified->get_bytes(missing));
        },
        "verified cleaner reopen resurrected a deleted ID");

    throwdb::NativeOrphanCollectionOptions orphan_options;
    orphan_options.min_age_ns = 0;
    orphan_options.max_files = 1024;
    orphan_options.max_bytes = 1ULL << 30;
    const throwdb::NativeOrphanCollectionResult reclaimed =
        writer->reclaim_orphans(orphan_options);
    require(
        std::any_of(
            reclaimed.entries.begin(), reclaimed.entries.end(),
            [](const auto& entry) {
                return entry.removed
                       && entry.file.classification
                              == throwdb::NativeOrphanClass::
                                  CompleteUnreferencedSegmentPair;
            }),
        "orphan collector did not reclaim the retired segment pair");
    require(
        old_reader->get_bytes(id_four) == row_four(),
        "old reader lost its pinned source after orphan unlink");

    const std::vector<std::byte> newest = make_rows(1, 91);
    writer->update(id_four, newest);
    require(writer->rotate_wal().rotated, "second fixture WAL did not rotate");
    require(
        writer->materialize_frozen_wals().materialized,
        "second fixture WAL did not materialize");
    const std::uint32_t first_cleaner_id =
        replacement->row_ref_segment_id;
    const throwdb::NativeSegmentCleaningStats second =
        writer->clean_segments(clean_options(false));
    require(
        second.cleaned && second.candidate_segment_id == first_cleaner_id
            && second.source_rows == 1 && second.relocated_rows == 0
            && second.garbage_rows == 1
            && second.output_segment_count == 0,
        "cleaning an already-Cleaner segment did not emit an all-dead fence");
    writer.reset();
    verified.reset();
    auto reopened = throwdb::NativeCollection::open(
        temporary.path(), verified_options);
    require(
        reopened->get_bytes(id_four) == newest,
        "Cleaner-chain reopen returned a stale embedding");
}

std::atomic<bool> cleaner_output_entered{false};
std::atomic<bool> release_cleaner_output{false};

void blocking_cleaner_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    using namespace throwdb::detail;
    if (context.operation != NativeCollectionTestOperation::CleanSegment
        || context.point
               != NativeCollectionTestHookPoint::CleanerSegmentOutputDurable) {
        return;
    }
    cleaner_output_entered.store(true, std::memory_order_release);
    while (!release_cleaner_output.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void test_concurrent_wal_commit_and_rotation_are_preserved() {
    TemporaryDirectory temporary("concurrent-wal");
    build_collection(temporary.path());
    std::unique_ptr<throwdb::NativeCollection> writer;
    create_garbage(temporary.path(), writer);
    cleaner_output_entered.store(false, std::memory_order_relaxed);
    release_cleaner_output.store(false, std::memory_order_relaxed);
    throwdb::detail::set_native_collection_test_hook(blocking_cleaner_hook);
    std::exception_ptr failure;
    std::thread cleaner([&] {
        try {
            static_cast<void>(writer->clean_segments(clean_options(false)));
        } catch (...) {
            failure = std::current_exception();
        }
    });
    while (!cleaner_output_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    const std::array<std::uint64_t, 1> id_four{4};
    const std::vector<std::byte> newest = make_rows(1, 117);
    writer->update(id_four, newest);
    require(
        writer->rotate_wal().rotated,
        "concurrent WAL rotation was not published");
    release_cleaner_output.store(true, std::memory_order_release);
    cleaner.join();
    throwdb::detail::set_native_collection_test_hook(nullptr);
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    require(
        writer->get_bytes(id_four) == newest,
        "cleaner final rebase discarded a concurrent WAL update");
    require(
        writer->materialize_frozen_wals().materialized,
        "concurrent WAL lineage was not materializable after cleaning");
    writer.reset();
    auto reopened = throwdb::NativeCollection::open(temporary.path());
    require(
        reopened->get_bytes(id_four) == newest,
        "reopen lost a concurrent WAL update preserved by cleaner rebase");
}

void test_memory_budget_includes_run_toc_and_bitmap_scratch() {
    TemporaryDirectory temporary("memory-boundary");
    constexpr std::uint32_t partition_bits = 8;
    constexpr std::uint64_t partition_count = 1ULL << partition_bits;
    const std::array<std::uint64_t, 1> ids{901};
    const std::vector<std::byte> embeddings = make_rows(1, 29);
    throwdb::NativeBuilderOptions builder_options;
    builder_options.dimension = kDimension;
    builder_options.dtype = throwdb::DType::Float16;
    builder_options.segment_size_bytes = kRecordSize;
    builder_options.format_version = 7;
    builder_options.partition_bits = partition_bits;
    auto builder = throwdb::NativeCollectionBuilder::create(
        temporary.path(), builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    require(
        writer->erase(ids) == std::vector<std::uint8_t>{1},
        "memory-boundary fixture delete failed");
    require(writer->rotate_wal().rotated,
            "memory-boundary fixture did not rotate");
    require(writer->materialize_frozen_wals().materialized,
            "memory-boundary fixture did not materialize");

    // Keep this boundary synchronized with the conservative native planner:
    // one 128-byte row plan, 64 bytes per partition for the simultaneous
    // decoded/serialized TOCs and bitmaps, 64 KiB fixed headroom, and one
    // record-sized read batch.
    constexpr std::uint64_t required_memory =
        128 + partition_count * 64 + (64ULL << 10) + kRecordSize;
    throwdb::NativeSegmentCleaningOptions below = clean_options(true);
    below.max_memory_bytes = required_memory - 1;
    const throwdb::NativeSegmentCleaningStats blocked =
        writer->clean_segments(below);
    require(
        !blocked.candidate_found && blocked.budget_blocked
            && blocked.skipped_budget_segment_count == 1,
        "cleaner admitted a candidate below its TOC-aware memory bound");

    throwdb::NativeSegmentCleaningOptions exact = clean_options(true);
    exact.max_memory_bytes = required_memory;
    const throwdb::NativeSegmentCleaningStats admitted =
        writer->clean_segments(exact);
    require(
        admitted.candidate_found && admitted.candidate_segment_id == 1
            && admitted.garbage_rows == 1,
        "cleaner rejected the exact conservative memory boundary");
}

throwdb::detail::NativeCollectionTestHookPoint crash_point =
    throwdb::detail::NativeCollectionTestHookPoint::MutationReservationDurable;
std::uint64_t crash_generation = 0;

void crashing_cleaner_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    if (context.operation
            == throwdb::detail::NativeCollectionTestOperation::CleanSegment
        && context.point == crash_point
        && context.generation == crash_generation) {
        ::_exit(79);
    }
}

void test_cleaner_crash_checkpoints_reopen() {
    using HookPoint = throwdb::detail::NativeCollectionTestHookPoint;
    const std::array<HookPoint, 5> points{
        HookPoint::MutationReservationDurable,
        HookPoint::CleanerSegmentOutputDurable,
        HookPoint::CleanerRunOutputDurable,
        HookPoint::MutationCatalogDurableBeforeCurrent,
        HookPoint::MutationCurrentDurable};
    for (std::size_t index = 0; index != points.size(); ++index) {
        TemporaryDirectory temporary("crash-" + std::to_string(index));
        build_collection(temporary.path());
        std::unique_ptr<throwdb::NativeCollection> writer;
        create_garbage(temporary.path(), writer);
        const std::uint64_t base_generation =
            writer->mutation_log_stats().generation;
        writer.reset();
        crash_point = points[index];
        crash_generation = base_generation
                           + ((points[index]
                               == HookPoint::MutationCatalogDurableBeforeCurrent
                              || points[index]
                                     == HookPoint::MutationCurrentDurable)
                                  ? 2
                                  : 1);
        const pid_t child = ::fork();
        require(child >= 0, "cannot fork cleaner crash child");
        if (child == 0) {
            throwdb::detail::set_native_collection_test_hook(
                crashing_cleaner_hook);
            try {
                auto child_writer = throwdb::NativeCollection::open(
                    temporary.path(), {}, true);
                static_cast<void>(
                    child_writer->clean_segments(clean_options(false)));
            } catch (...) {
                ::_exit(80);
            }
            ::_exit(81);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child, "waitpid failed");
        require(
            WIFEXITED(status) && WEXITSTATUS(status) == 79,
            "cleaner crash child missed its durable checkpoint");
        auto reopened = throwdb::NativeCollection::open(temporary.path());
        const std::array<std::uint64_t, 1> id_four{4};
        require(
            reopened->count() == 5
                && reopened->get_bytes(id_four) == row_four(),
            "cleaner crash exposed a partial logical state");
    }
    throwdb::detail::set_native_collection_test_hook(nullptr);
}

void test_sample_skips_catalog_inactive_run_sections() {
    TemporaryDirectory temporary("sample-active-sections");
    std::array<std::uint64_t, 2> ids{};
    std::array<bool, 2> found{};
    for (std::uint64_t id = 1; id != 10'000 && (!found[0] || !found[1]); ++id) {
        const std::size_t partition = static_cast<std::size_t>(
            throwdb::detail::hash_id(id, throwdb::detail::kHashSeed) >> 63);
        if (!found[partition]) {
            ids[partition] = id;
            found[partition] = true;
        }
    }
    require(found[0] && found[1], "cannot find IDs in both partitions");
    const std::vector<std::byte> base = make_rows(2, 5);
    throwdb::NativeBuilderOptions options;
    options.dimension = kDimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 2 * kRecordSize;
    options.format_version = 7;
    options.partition_bits = 1;
    auto builder = throwdb::NativeCollectionBuilder::create(
        temporary.path(), options);
    builder->append(ids, base);
    builder->finish();
    auto writer = throwdb::NativeCollection::open(
        temporary.path(), {}, true);
    const std::vector<std::byte> replacement = make_rows(2, 71);
    writer->update(ids, replacement);
    static_cast<void>(writer->rotate_wal());
    static_cast<void>(writer->materialize_frozen_wals());
    const throwdb::NativeMutationLogStats folded =
        writer->compact_delta_runs(2, 1ULL << 20, 2);
    require(
        folded.compacted && folded.compacted_partition_count == 1,
        "sample fixture did not fold exactly one run section");
    const auto catalog = current_catalog(temporary.path());
    require(catalog.runs.size() == 1, "sample fixture removed the whole run");
    std::size_t active_partition = 0;
    const std::uint8_t active = std::to_integer<std::uint8_t>(
        catalog.runs.front().active_partitions.front());
    if ((active & 0x01U) != 0) {
        active_partition = 0;
    } else if ((active & 0x02U) != 0) {
        active_partition = 1;
    } else {
        fail("sample fixture lacks an active run section");
    }
    const throwdb::NativeIdSample sample = writer->sample_ids(1, 17, 1);
    require(
        sample.complete && sample.inspected_slots == 1
            && sample.ids == std::vector<std::uint64_t>{ids[active_partition]},
        "bounded sample enumerated a catalog-inactive run section");
}

}  // namespace

int main() {
    const std::array tests{
        std::pair{"dry_run_execute_reopen_pin_and_cleaner_chain",
                  &test_dry_run_execute_reopen_pin_and_cleaner_chain},
        std::pair{"concurrent_wal_commit_and_rotation_are_preserved",
                  &test_concurrent_wal_commit_and_rotation_are_preserved},
        std::pair{"memory_budget_includes_run_toc_and_bitmap_scratch",
                  &test_memory_budget_includes_run_toc_and_bitmap_scratch},
        std::pair{"cleaner_crash_checkpoints_reopen",
                  &test_cleaner_crash_checkpoints_reopen},
        std::pair{"sample_skips_catalog_inactive_run_sections",
                  &test_sample_skips_catalog_inactive_run_sections},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " cleaner tests passed\n";
    return 0;
}
