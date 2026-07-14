#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "partition_snapshot_storage.hpp"
#include "partitioned_index_run_storage.hpp"

namespace throwdb::detail {

struct PartitionFoldBudget {
    std::uint64_t min_runs{1};
    std::uint64_t max_runs{1};
    std::uint64_t max_input_entries{0};
    std::uint64_t max_input_bytes{0};
};

struct PartitionFoldCandidate {
    std::uint32_t partition_id{0};
    std::uint64_t active_run_count{0};
    std::uint64_t snapshot_entries{0};
    std::uint64_t run_entries{0};
    std::uint64_t input_entries{0};
    std::uint64_t snapshot_bytes{0};
    std::uint64_t run_bytes{0};
    std::uint64_t input_bytes{0};

    bool operator==(const PartitionFoldCandidate&) const = default;
};

struct PartitionRunView {
    const PartitionIndexRunReader* reader{nullptr};
    // Immutable descriptor owned by this catalog generation. It may carry a
    // narrower active bitmap than the descriptor used when reader was opened.
    const PartitionIndexRunDescriptor* catalog_descriptor{nullptr};
};

using PartitionRunViews = std::span<const PartitionRunView>;

// Every active section for a candidate partition is included in its cost.
// There is deliberately no "first N runs" fallback: a partition that exceeds
// any budget is ineligible as a whole.
[[nodiscard]] std::optional<PartitionFoldCandidate>
select_partition_for_fold(
    std::uint32_t partition_bits,
    std::span<const PartitionSnapshotDescriptor> snapshots,
    PartitionRunViews runs,
    const PartitionFoldBudget& budget);

// Merges the current authoritative snapshot and every catalog-active run
// section for one partition. The returned live records are strictly ID-sorted
// and directly consumable by write_partition_snapshot().
[[nodiscard]] std::vector<PartitionRunRecord> merge_partition_for_fold(
    std::uint32_t partition_bits,
    std::uint32_t partition_id,
    const PartitionSnapshotReader& snapshot,
    PartitionRunViews runs);

// Online point lookup uses the exact same winner reducer as a materialized
// fold. A returned tombstone is intentional and shadows lower state; nullopt
// means that no version of the ID exists in the supplied view.
[[nodiscard]] std::optional<PartitionRunRecord> lookup_partition_version(
    std::uint32_t partition_bits,
    std::uint32_t partition_id,
    std::uint64_t id,
    const PartitionSnapshotReader& snapshot,
    PartitionRunViews runs);

// Transactional catalog transform. On validation/allocation failure `runs`
// remains byte-for-byte equivalent. Successful completion clears the selected
// active bit, removes descriptors with no remaining active section, and does
// not alter covered membership, epochs, or relative order.
void clear_folded_partition(
    std::vector<PartitionIndexRunDescriptor>& runs,
    std::uint32_t partition_bits,
    std::uint32_t partition_id);

}  // namespace throwdb::detail
