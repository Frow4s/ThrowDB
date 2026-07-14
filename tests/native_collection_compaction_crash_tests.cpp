#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "throwdb/native_collection.hpp"
#include "../src/core/mutation_catalog.hpp"
#include "../src/core/mutation_delta.hpp"
#include "../src/core/mutation_inventory.hpp"
#include "../src/core/native_collection_test_hooks.hpp"
#include "../src/core/native_format.hpp"

namespace {

using HookContext = throwdb::detail::NativeCollectionTestHookContext;
using HookPoint = throwdb::detail::NativeCollectionTestHookPoint;
using InventoryClass = throwdb::detail::MutationInventoryClass;
using Operation = throwdb::detail::NativeCollectionTestOperation;
using ResourceKind = throwdb::detail::NativeCollectionTestResourceKind;

constexpr int kCrashExitCode = 86;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-compaction-crash-test-"
                   + std::to_string(timestamp) + "-"
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

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t row_count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> result(row_count * record_size);
    for (std::size_t row = 0; row != row_count; ++row) {
        for (std::size_t column = 0; column != record_size; ++column) {
            result[row * record_size + column] = static_cast<std::byte>(
                salt + static_cast<std::uint8_t>(row * 29 + column * 7));
        }
    }
    return result;
}

void require_bytes_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::byte> expected,
    const std::string& message) {
    require(
        actual.size() == expected.size()
            && std::equal(actual.begin(), actual.end(), expected.begin()),
        message);
}

void build_store(
    const std::filesystem::path& store,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension) {
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(ids, embeddings);
    builder->finish();
}

[[nodiscard]] throwdb::detail::NativeManifest manifest_for(
    const std::filesystem::path& store) {
    return throwdb::detail::read_native_manifest(store);
}

[[nodiscard]] throwdb::detail::MutationCatalog catalog_for(
    const std::filesystem::path& store) {
    const auto manifest = manifest_for(store);
    return throwdb::detail::read_current_mutation_catalog(store, manifest);
}

void materialize_active_wal(
    throwdb::NativeCollection& writer,
    const std::string& context) {
    require(writer.rotate_wal().rotated,
            context + ": active WAL did not rotate");
    require(writer.materialize_frozen_wals().materialized,
            context + ": frozen WAL did not materialize");
}

struct Fixture {
    std::array<std::uint64_t, 2> ids{1, 2};
    std::vector<std::byte> expected;
    throwdb::detail::MutationCatalog before;
};

[[nodiscard]] Fixture make_fixture(const std::filesystem::path& store) {
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    Fixture fixture;
    const std::vector<std::byte> base = rows(2, record_size, 3);
    build_store(store, fixture.ids, base, dimension);

    auto writer = throwdb::NativeCollection::open(store, {}, true);
    const std::vector<std::byte> first = rows(1, record_size, 31);
    writer->update(std::array<std::uint64_t, 1>{1}, first);
    materialize_active_wal(*writer, "first fixture run");

    fixture.expected = rows(2, record_size, 61);
    writer->update(fixture.ids, fixture.expected);
    materialize_active_wal(*writer, "second fixture run");
    writer.reset();

    fixture.before = catalog_for(store);
    require_equal(fixture.before.delta_entries.size(), std::size_t{2},
                  "crash fixture does not contain two delta runs");
    return fixture;
}

[[nodiscard]] std::filesystem::path delta_data_path(
    const std::filesystem::path& store,
    const throwdb::detail::MutationDeltaDescriptor& descriptor) {
    return throwdb::detail::mutation_delta_data_path(
        store, descriptor.delta_id);
}

[[nodiscard]] std::filesystem::path delta_index_path(
    const std::filesystem::path& store,
    const throwdb::detail::MutationDeltaDescriptor& descriptor) {
    return throwdb::detail::mutation_delta_index_path(
        store, descriptor.delta_id);
}

void require_logical_view(
    const std::filesystem::path& store,
    const Fixture& fixture,
    const std::string& context) {
    auto reader = throwdb::NativeCollection::open(store);
    require_equal(reader->count(), std::uint64_t{2},
                  context + ": live count changed");
    require_bytes_equal(
        reader->get_bytes(fixture.ids), fixture.expected,
        context + ": logical rows are not bit exact");
}

void require_lineage_unchanged(
    const throwdb::detail::MutationCatalog& actual,
    const throwdb::detail::MutationCatalog& before,
    const std::string& context) {
    require_equal(actual.wal_entries, before.wal_entries,
                  context + ": retained WAL catalog changed");
    require_equal(actual.materialized_sequence, before.materialized_sequence,
                  context + ": materialized sequence changed");
    require_equal(actual.terminal_wal_id, before.terminal_wal_id,
                  context + ": materialized WAL lineage changed");
}

enum class CrashCase {
    ReservationDurable,
    OutputDurable,
    FinalCatalogDurable,
    FinalCurrentDurable,
    FirstDataUnlinked,
    FirstPairUnlinked,
};

[[nodiscard]] const char* crash_case_name(const CrashCase crash_case) {
    switch (crash_case) {
        case CrashCase::ReservationDurable:
            return "reservation_durable";
        case CrashCase::OutputDurable:
            return "output_durable";
        case CrashCase::FinalCatalogDurable:
            return "final_catalog_durable_before_current";
        case CrashCase::FinalCurrentDurable:
            return "final_current_durable_before_unlink";
        case CrashCase::FirstDataUnlinked:
            return "after_first_data_unlink";
        case CrashCase::FirstPairUnlinked:
            return "after_first_pair_unlink";
    }
    return "unknown";
}

struct CrashTarget {
    HookPoint point{HookPoint::MutationReservationDurable};
    ResourceKind resource_kind{ResourceKind::None};
    std::uint64_t generation{0};
    std::uint64_t ordinal{0};
};

CrashTarget active_target;

void crash_hook(const HookContext context) noexcept {
    if (context.operation == Operation::CompactDeltaRuns
        && context.point == active_target.point
        && context.resource_kind == active_target.resource_kind
        && context.generation == active_target.generation
        && context.ordinal == active_target.ordinal) {
        ::_exit(kCrashExitCode);
    }
}

[[nodiscard]] CrashTarget target_for(
    const CrashCase crash_case,
    const std::uint64_t generation) {
    switch (crash_case) {
        case CrashCase::ReservationDurable:
            return {HookPoint::MutationReservationDurable,
                    ResourceKind::None, generation + 1, 0};
        case CrashCase::OutputDurable:
            return {HookPoint::MutationDeltaOutputDurable,
                    ResourceKind::MutationDeltaPair, generation + 1, 0};
        case CrashCase::FinalCatalogDurable:
            return {HookPoint::MutationCatalogDurableBeforeCurrent,
                    ResourceKind::None, generation + 2, 0};
        case CrashCase::FinalCurrentDurable:
            return {HookPoint::MutationCurrentDurable,
                    ResourceKind::None, generation + 2, 0};
        case CrashCase::FirstDataUnlinked:
            return {HookPoint::AfterSourceUnlink,
                    ResourceKind::MutationDeltaData, generation + 2, 0};
        case CrashCase::FirstPairUnlinked:
            return {HookPoint::AfterSourceUnlink,
                    ResourceKind::MutationDeltaIndex, generation + 2, 0};
    }
    fail("unknown crash case");
}

void crash_child_at(
    const std::filesystem::path& store,
    const CrashTarget target) {
    const pid_t child = ::fork();
    if (child < 0) {
        fail("fork failed for compaction crash test");
    }
    if (child == 0) {
        active_target = target;
        throwdb::detail::set_native_collection_test_hook(crash_hook);
        try {
            auto writer = throwdb::NativeCollection::open(store, {}, true);
            static_cast<void>(writer->compact_delta_runs(
                2, std::numeric_limits<std::uint64_t>::max(),
                std::numeric_limits<std::uint64_t>::max()));
            ::_exit(87);
        } catch (...) {
            ::_exit(88);
        }
    }

    int status = 0;
    pid_t waited = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (true) {
        waited = ::waitpid(child, &status, WNOHANG);
        if (waited == child) {
            break;
        }
        if (waited < 0 && errno != EINTR) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            static_cast<void>(::kill(child, SIGKILL));
            do {
                waited = ::waitpid(child, &status, 0);
            } while (waited < 0 && errno == EINTR);
            fail("compaction crash child timed out");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(waited == child, "waitpid failed for compaction crash child");
    require(WIFEXITED(status), "compaction crash child did not exit normally");
    require_equal(WEXITSTATUS(status), kCrashExitCode,
                  "compaction child missed its requested crash checkpoint");
}

[[nodiscard]] std::size_t count_delta_class(
    const throwdb::detail::MutationResourceInventory& inventory,
    const InventoryClass classification) {
    return static_cast<std::size_t>(std::count_if(
        inventory.entries.begin(), inventory.entries.end(),
        [classification](const auto& entry) {
            return entry.relative_path.starts_with("deltas/")
                   && entry.classification == classification;
        }));
}

[[nodiscard]] std::size_t count_class(
    const throwdb::detail::MutationResourceInventory& inventory,
    const InventoryClass classification) {
    return static_cast<std::size_t>(std::count_if(
        inventory.entries.begin(), inventory.entries.end(),
        [classification](const auto& entry) {
            return entry.classification == classification;
        }));
}

[[nodiscard]] const throwdb::detail::MutationInventoryEntry& inventory_entry(
    const throwdb::detail::MutationResourceInventory& inventory,
    const std::string& relative_path) {
    const auto found = std::find_if(
        inventory.entries.begin(), inventory.entries.end(),
        [&relative_path](const auto& entry) {
            return entry.relative_path == relative_path;
        });
    if (found == inventory.entries.end()) {
        fail("inventory is missing '" + relative_path + "'");
    }
    return *found;
}

[[nodiscard]] std::string relative_path(
    const std::filesystem::path& store,
    const std::filesystem::path& path) {
    return path.lexically_relative(store).generic_string();
}

void require_path_class(
    const throwdb::detail::MutationResourceInventory& inventory,
    const std::filesystem::path& store,
    const std::filesystem::path& path,
    const InventoryClass expected,
    const std::string& context) {
    require_equal(
        inventory_entry(inventory, relative_path(store, path)).classification,
        expected, context + ": wrong inventory class for "
                      + relative_path(store, path));
}

[[nodiscard]] std::vector<std::string> delta_paths_with_class(
    const throwdb::detail::MutationResourceInventory& inventory,
    const InventoryClass classification) {
    std::vector<std::string> result;
    for (const auto& entry : inventory.entries) {
        if (entry.relative_path.starts_with("deltas/")
            && entry.classification == classification) {
            result.push_back(entry.relative_path);
        }
    }
    return result;
}

void require_descriptor_class(
    const throwdb::detail::MutationResourceInventory& inventory,
    const std::filesystem::path& store,
    const throwdb::detail::MutationDeltaDescriptor& descriptor,
    const InventoryClass expected,
    const std::string& context) {
    require_path_class(
        inventory, store, delta_data_path(store, descriptor), expected,
        context);
    require_path_class(
        inventory, store, delta_index_path(store, descriptor), expected,
        context);
}

void require_source_file_state(
    const std::filesystem::path& store,
    const Fixture& fixture,
    const bool first_data,
    const bool first_index,
    const bool second_data,
    const bool second_index,
    const std::string& context) {
    const auto& first = fixture.before.delta_entries[0];
    const auto& second = fixture.before.delta_entries[1];
    require_equal(std::filesystem::exists(delta_data_path(store, first)),
                  first_data, context + ": wrong first data state");
    require_equal(std::filesystem::exists(delta_index_path(store, first)),
                  first_index, context + ": wrong first index state");
    require_equal(std::filesystem::exists(delta_data_path(store, second)),
                  second_data, context + ": wrong second data state");
    require_equal(std::filesystem::exists(delta_index_path(store, second)),
                  second_index, context + ": wrong second index state");
}

[[nodiscard]] bool is_pre_final_case(const CrashCase crash_case) {
    return crash_case == CrashCase::ReservationDurable
           || crash_case == CrashCase::OutputDurable
           || crash_case == CrashCase::FinalCatalogDurable;
}

void run_crash_case(const CrashCase crash_case) {
    TempDirectory temporary;
    const auto store = temporary.path() / crash_case_name(crash_case);
    const Fixture fixture = make_fixture(store);
    const std::uint64_t original_generation = fixture.before.generation;
    const std::uint32_t original_next_segment =
        fixture.before.next_segment_id;
    const auto baseline_inventory =
        throwdb::detail::inspect_mutation_resources(
            store, manifest_for(store));

    crash_child_at(
        store, target_for(crash_case, original_generation));

    const auto current = catalog_for(store);
    const auto manifest = manifest_for(store);
    const auto inventory =
        throwdb::detail::inspect_mutation_resources(store, manifest);
    const std::string context = crash_case_name(crash_case);
    require_logical_view(store, fixture, context + " after crash");
    const std::size_t new_catalog_count =
        crash_case == CrashCase::ReservationDurable
                || crash_case == CrashCase::OutputDurable
            ? 1
            : 2;
    require_equal(
        count_class(inventory, InventoryClass::SupersededCatalog),
        count_class(baseline_inventory, InventoryClass::SupersededCatalog)
            + new_catalog_count,
        context + ": durable catalog checkpoint left the wrong history");
    require_equal(
        count_class(inventory, InventoryClass::TempPublication),
        std::size_t{0},
        context + ": durable checkpoint left a temporary publication");
    require_lineage_unchanged(current, fixture.before, context);

    for (const auto& source : fixture.before.delta_entries) {
        if (std::filesystem::exists(delta_data_path(store, source))) {
            require_path_class(
                inventory, store, delta_data_path(store, source),
                is_pre_final_case(crash_case)
                    ? InventoryClass::Reachable
                    : crash_case == CrashCase::FirstDataUnlinked
                              && source == fixture.before.delta_entries.front()
                          ? InventoryClass::PartialUnreferencedDeltaPair
                          : InventoryClass::CompleteUnreferencedDeltaPair,
                context);
        }
        if (std::filesystem::exists(delta_index_path(store, source))) {
            require_path_class(
                inventory, store, delta_index_path(store, source),
                is_pre_final_case(crash_case)
                    ? InventoryClass::Reachable
                    : crash_case == CrashCase::FirstDataUnlinked
                              && source == fixture.before.delta_entries.front()
                          ? InventoryClass::PartialUnreferencedDeltaPair
                          : InventoryClass::CompleteUnreferencedDeltaPair,
                context);
        }
    }

    if (is_pre_final_case(crash_case)) {
        require_equal(current.generation, original_generation + 1,
                      context + ": CURRENT did not retain reservation");
        require_equal(current.delta_entries, fixture.before.delta_entries,
                      context + ": reservation changed the delta view");
        require_equal(current.next_segment_id,
                      static_cast<std::uint32_t>(original_next_segment + 1),
                      context + ": segment ID was not durably reserved");
        require_source_file_state(
            store, fixture, true, true, true, true, context);
        const std::size_t expected_orphan_parts =
            crash_case == CrashCase::ReservationDurable ? 0 : 2;
        require_equal(
            count_delta_class(
                inventory, InventoryClass::CompleteUnreferencedDeltaPair),
            expected_orphan_parts,
            context + ": wrong complete orphan-output classification");
        require_equal(
            count_delta_class(
                inventory, InventoryClass::PartialUnreferencedDeltaPair),
            std::size_t{0},
            context + ": pre-final crash created a partial delta pair");
        const std::vector<std::string> abandoned_output_paths =
            delta_paths_with_class(
                inventory,
                InventoryClass::CompleteUnreferencedDeltaPair);

        auto writer = throwdb::NativeCollection::open(store, {}, true);
        const auto retried = writer->compact_delta_runs(
            2, std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max());
        require(retried.compacted,
                context + ": retry did not compact the authoritative inputs");
        writer.reset();
        const auto after_retry = catalog_for(store);
        require_equal(after_retry.generation, original_generation + 3,
                      context + ": retry published the wrong generations");
        require_equal(after_retry.next_segment_id,
                      static_cast<std::uint32_t>(original_next_segment + 2),
                      context + ": retry reused a reserved segment ID");
        require_equal(after_retry.delta_entries.size(), std::size_t{1},
                      context + ": retry did not publish one merged run");
        require_equal(after_retry.delta_entries.front().segment_id,
                      static_cast<std::uint32_t>(original_next_segment + 1),
                      context + ": retry output used the abandoned segment ID");
        require_lineage_unchanged(
            after_retry, fixture.before, context + " after retry");
        require_source_file_state(
            store, fixture, false, false, false, false,
            context + " after retry");
        const auto retry_inventory =
            throwdb::detail::inspect_mutation_resources(
                store, manifest_for(store));
        require_descriptor_class(
            retry_inventory, store, after_retry.delta_entries.front(),
            InventoryClass::Reachable, context + " retry output");
        require_equal(
            delta_paths_with_class(
                retry_inventory,
                InventoryClass::CompleteUnreferencedDeltaPair),
            abandoned_output_paths,
            context + ": retry changed the abandoned output set");
        require_equal(
            count_delta_class(
                retry_inventory,
                InventoryClass::PartialUnreferencedDeltaPair),
            std::size_t{0},
            context + ": retry left a partial orphan pair");
    } else {
        require_equal(current.generation, original_generation + 2,
                      context + ": CURRENT did not select final compaction");
        require_equal(current.next_segment_id,
                      static_cast<std::uint32_t>(original_next_segment + 1),
                      context + ": final view has the wrong segment cursor");
        require_equal(current.delta_entries.size(), std::size_t{1},
                      context + ": final view lacks the merged run");
        require_equal(current.delta_entries.front().segment_id,
                      original_next_segment,
                      context + ": final view selected the wrong output");
        require_descriptor_class(
            inventory, store, current.delta_entries.front(),
            InventoryClass::Reachable, context + " current output");

        if (crash_case == CrashCase::FinalCurrentDurable) {
            require_source_file_state(
                store, fixture, true, true, true, true, context);
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::CompleteUnreferencedDeltaPair),
                std::size_t{4},
                context + ": retained source pairs were not classified");
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::PartialUnreferencedDeltaPair),
                std::size_t{0},
                context + ": intact source pairs were marked partial");
        } else if (crash_case == CrashCase::FirstDataUnlinked) {
            require_source_file_state(
                store, fixture, false, true, true, true, context);
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::CompleteUnreferencedDeltaPair),
                std::size_t{2},
                context + ": intact second pair was not classified");
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::PartialUnreferencedDeltaPair),
                std::size_t{1},
                context + ": partial first pair was not classified");
        } else {
            require_source_file_state(
                store, fixture, false, false, true, true, context);
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::CompleteUnreferencedDeltaPair),
                std::size_t{2},
                context + ": remaining second pair was not classified");
            require_equal(
                count_delta_class(
                    inventory,
                    InventoryClass::PartialUnreferencedDeltaPair),
                std::size_t{0},
                context + ": removed first pair left a partial orphan");
        }

        auto writer = throwdb::NativeCollection::open(store, {}, true);
        const auto retry = writer->compact_delta_runs(
            2, std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max());
        require(!retry.compacted,
                context + ": retry rewrote an already compacted view");
        require_equal(retry.generation, original_generation + 2,
                      context + ": no-op retry advanced CURRENT");
    }

    require_logical_view(store, fixture, context + " after retry/reopen");
}

struct TestCase {
    CrashCase crash_case;
};

}  // namespace

int main() {
    const std::array<TestCase, 6> tests{{
        {CrashCase::ReservationDurable},
        {CrashCase::OutputDurable},
        {CrashCase::FinalCatalogDurable},
        {CrashCase::FinalCurrentDurable},
        {CrashCase::FirstDataUnlinked},
        {CrashCase::FirstPairUnlinked},
    }};

    std::size_t failures = 0;
    for (const TestCase& test : tests) {
        try {
            run_crash_case(test.crash_case);
            std::cout << "[PASS] " << crash_case_name(test.crash_case)
                      << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << crash_case_name(test.crash_case)
                      << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << crash_case_name(test.crash_case)
                      << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " compaction crash test(s) failed\n";
        return 1;
    }
    return 0;
}
