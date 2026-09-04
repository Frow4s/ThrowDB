#include "partitioned_index_merge.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace throwdb::detail {
namespace {

template <typename Error>
void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left + right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left * right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_partition_count(
    const std::uint32_t partition_bits) {
    require<Error>(
        partition_bits <= kPartitionedMaximumPartitionBits,
        "partition fold exceeds the maximum partition bits");
    return std::uint64_t{1} << partition_bits;
}

[[nodiscard]] std::size_t bitmap_size(
    const std::uint64_t partition_count) {
    return static_cast<std::size_t>((partition_count + 7) / 8);
}

[[nodiscard]] bool bitmap_contains(
    const std::span<const std::byte> bitmap,
    const std::uint64_t partition) noexcept {
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    return (std::to_integer<std::uint8_t>(
                bitmap[static_cast<std::size_t>(partition / 8)])
            & mask) != 0;
}

void bitmap_clear(
    const std::span<std::byte> bitmap,
    const std::uint64_t partition) noexcept {
    const std::uint8_t mask = static_cast<std::uint8_t>(
        std::uint32_t{1} << (partition % 8));
    std::byte& value = bitmap[static_cast<std::size_t>(partition / 8)];
    value &= static_cast<std::byte>(static_cast<std::uint8_t>(~mask));
}

[[nodiscard]] bool bitmap_any(
    const std::span<const std::byte> bitmap) noexcept {
    return std::any_of(
        bitmap.begin(), bitmap.end(),
        [](const std::byte value) { return value != std::byte{0}; });
}

template <typename Error>
void validate_run_bitmap(
    const PartitionIndexRunDescriptor& descriptor,
    const std::uint32_t partition_bits) {
    const std::uint64_t partitions =
        checked_partition_count<Error>(partition_bits);
    const std::size_t expected = bitmap_size(partitions);
    require<Error>(
        descriptor.covered_partitions.size() == expected
            && descriptor.active_partitions.size() == expected,
        "partition fold run bitmap has invalid size");
    require<Error>(
        bitmap_any(descriptor.covered_partitions)
            && bitmap_any(descriptor.active_partitions),
        "partition fold run has an empty covered or active bitmap");
    for (std::size_t index = 0; index != expected; ++index) {
        const std::uint8_t covered = std::to_integer<std::uint8_t>(
            descriptor.covered_partitions[index]);
        const std::uint8_t active = std::to_integer<std::uint8_t>(
            descriptor.active_partitions[index]);
        require<Error>(
            (active & static_cast<std::uint8_t>(~covered)) == 0,
            "partition fold active bitmap is not a subset of coverage");
    }
    if (partitions % 8 != 0) {
        const std::uint8_t allowed = static_cast<std::uint8_t>(
            (std::uint32_t{1} << (partitions % 8)) - 1);
        require<Error>(
            (std::to_integer<std::uint8_t>(
                 descriptor.covered_partitions.back())
             & static_cast<std::uint8_t>(~allowed)) == 0
                && (std::to_integer<std::uint8_t>(
                        descriptor.active_partitions.back())
                    & static_cast<std::uint8_t>(~allowed)) == 0,
            "partition fold run bitmap has nonzero padding bits");
    }
}

template <typename Error>
void validate_run_descriptor_order(
    const std::span<const PartitionIndexRunDescriptor> runs,
    const std::uint32_t partition_bits) {
    std::uint64_t previous_epoch = 0;
    for (const PartitionIndexRunDescriptor& descriptor : runs) {
        validate_run_bitmap<Error>(descriptor, partition_bits);
        require<Error>(
            descriptor.epoch > previous_epoch,
            "partition fold runs are not in strict epoch order");
        previous_epoch = descriptor.epoch;
    }
}

[[nodiscard]] bool same_run_resource(
    const PartitionIndexRunDescriptor& left,
    const PartitionIndexRunDescriptor& right) {
    return left.run_id == right.run_id
           && left.epoch == right.epoch
           && left.kind == right.kind
           && left.first_sequence == right.first_sequence
           && left.last_sequence == right.last_sequence
           && left.entry_count == right.entry_count
           && left.file_size == right.file_size
           && left.body_crc32c == right.body_crc32c
           && left.covered_partitions == right.covered_partitions
           && left.retired_row_ref_segment_id
                  == right.retired_row_ref_segment_id
           && left.retired_segment_id == right.retired_segment_id
           && left.retired_row_count == right.retired_row_count
           && left.retired_ids_payload_crc32c
                  == right.retired_ids_payload_crc32c;
}

void validate_reader_set(
    const PartitionRunViews runs,
    const std::uint32_t partition_bits,
    const bool require_foldable_kind) {
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);
    std::uint64_t previous_epoch = 0;
    for (const PartitionRunView& view : runs) {
        require<InvalidArgumentError>(
            view.reader != nullptr && view.catalog_descriptor != nullptr,
            "partition fold contains an incomplete run view");
        const PartitionIndexRunDescriptor& descriptor =
            *view.catalog_descriptor;
        require<CorruptionError>(
            same_run_resource(view.reader->descriptor(), descriptor),
            "partition fold catalog descriptor does not bind its run reader");
        validate_run_bitmap<CorruptionError>(descriptor, partition_bits);
        require<CorruptionError>(
            view.reader->partition_count() == partitions,
            "partition fold run uses a different partition layout");
        require<CorruptionError>(
            descriptor.epoch > previous_epoch,
            "partition fold readers are not in strict catalog epoch order");
        if (require_foldable_kind) {
            require<CorruptionError>(
                descriptor.kind == PartitionedRunKind::Mutation
                    || descriptor.kind == PartitionedRunKind::Cleaner,
                "partition fold cannot consume an unknown run kind");
        }
        previous_epoch = descriptor.epoch;
    }
}

[[nodiscard]] std::optional<PartitionRunRecord> raw_run_lookup(
    const PartitionIndexRunReader& reader,
    const std::uint32_t partition,
    const std::uint64_t id) {
    const PartitionRunTocEntry section =
        reader.partition_records(partition);
    std::uint64_t first = section.first_record;
    std::uint64_t last = section.first_record + section.record_count;
    while (first != last) {
        const std::uint64_t middle = first + (last - first) / 2;
        const PartitionRunRecord record = reader.record_at(middle);
        if (record.id < id) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    if (first == section.first_record + section.record_count) {
        return std::nullopt;
    }
    const PartitionRunRecord record = reader.record_at(first);
    return record.id == id
               ? std::optional<PartitionRunRecord>(record)
               : std::nullopt;
}

[[nodiscard]] std::uint64_t snapshot_file_size(
    const std::uint64_t capacity) {
    require<CorruptionError>(
        capacity >= 8 && (capacity & (capacity - 1)) == 0,
        "partition fold snapshot capacity is invalid");
    const auto align = [](const std::uint64_t value) {
        const std::uint64_t remainder = value & (kPartitionedHeaderSize - 1);
        return remainder == 0
                   ? value
                   : checked_sum<CorruptionError>(
                         value, kPartitionedHeaderSize - remainder,
                         "partition fold snapshot alignment");
    };
    const std::uint64_t lookup_offset = align(
        checked_sum<CorruptionError>(
            kPartitionedHeaderSize, capacity,
            "partition fold snapshot controls"));
    const std::uint64_t sequence_offset = align(
        checked_sum<CorruptionError>(
            lookup_offset,
            checked_product<CorruptionError>(
                capacity, kPartitionSnapshotLookupSlotSize,
                "partition fold snapshot lookup bytes"),
            "partition fold snapshot lookup end"));
    return checked_sum<CorruptionError>(
        sequence_offset,
        checked_product<CorruptionError>(
            capacity, kPartitionSnapshotSequenceSlotSize,
            "partition fold snapshot sequence bytes"),
        "partition fold snapshot file size");
}

void validate_snapshots(
    const std::span<const PartitionSnapshotDescriptor> snapshots,
    const std::uint32_t partition_bits) {
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);
    require<InvalidArgumentError>(
        partitions <= std::numeric_limits<std::size_t>::max()
            && snapshots.size() == static_cast<std::size_t>(partitions),
        "partition fold requires exactly one snapshot per partition");
    for (std::uint64_t partition = 0; partition != partitions; ++partition) {
        const PartitionSnapshotDescriptor& descriptor =
            snapshots[static_cast<std::size_t>(partition)];
        require<CorruptionError>(
            !descriptor.snapshot_id.empty()
                && descriptor.partition_id == partition,
            "partition fold snapshots are missing or out of order");
        const std::uint64_t maximum_entries =
            (descriptor.capacity / 10) * 7
            + ((descriptor.capacity % 10) * 7) / 10;
        require<CorruptionError>(
            descriptor.entry_count <= maximum_entries
                && descriptor.file_size
                       == snapshot_file_size(descriptor.capacity),
            "partition fold snapshot descriptor has invalid geometry");
    }
}

struct WinningValue {
    std::uint64_t sequence{0};
    std::uint64_t value_raw{0};
    std::uint64_t source_epoch{0};
};

[[nodiscard]] WinningValue reduce_winner(
    WinningValue current,
    const std::uint64_t sequence,
    const std::uint64_t value_raw,
    const std::uint64_t source_epoch,
    const PartitionIndexRunDescriptor* source) {
    if (sequence > current.sequence) {
        return WinningValue{sequence, value_raw, source_epoch};
    }
    if (sequence < current.sequence) {
        return current;
    }
    if (value_raw == current.value_raw) {
        current.source_epoch = std::max(current.source_epoch, source_epoch);
        return current;
    }

    bool valid_relocation = false;
    if (source != nullptr
        && source->kind == PartitionedRunKind::Cleaner
        && source_epoch > current.source_epoch
        && current.value_raw != 0 && value_raw != 0) {
        const RowRef previous = RowRef::decode(current.value_raw);
        valid_relocation =
            previous.segment_id().value()
            == source->retired_row_ref_segment_id;
    }
    if (!valid_relocation) {
        throw CorruptionError(
            "partition fold has conflicting values at one sequence");
    }
    return WinningValue{sequence, value_raw, source_epoch};
}

void validate_active_epoch(
    const PartitionSnapshotDescriptor& snapshot,
    const PartitionIndexRunDescriptor& run) {
    require<CorruptionError>(
        snapshot.epoch < run.epoch,
        "partition fold has an active run no newer than its snapshot");
}

}  // namespace

std::optional<PartitionFoldCandidate> select_partition_for_fold(
    const std::uint32_t partition_bits,
    const std::span<const PartitionSnapshotDescriptor> snapshots,
    const PartitionRunViews runs,
    const PartitionFoldBudget& budget) {
    require<InvalidArgumentError>(
        budget.min_runs != 0 && budget.max_runs >= budget.min_runs,
        "partition fold run budget is invalid");
    validate_snapshots(snapshots, partition_bits);
    validate_reader_set(runs, partition_bits, true);
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);

    std::vector<PartitionFoldCandidate> candidates(
        static_cast<std::size_t>(partitions));
    for (std::uint64_t partition = 0; partition != partitions; ++partition) {
        PartitionFoldCandidate& candidate =
            candidates[static_cast<std::size_t>(partition)];
        const PartitionSnapshotDescriptor& snapshot =
            snapshots[static_cast<std::size_t>(partition)];
        candidate.partition_id = static_cast<std::uint32_t>(partition);
        candidate.snapshot_entries = snapshot.entry_count;
        candidate.snapshot_bytes = snapshot.file_size;
        candidate.input_entries = candidate.snapshot_entries;
        candidate.input_bytes = candidate.snapshot_bytes;
    }

    for (const PartitionRunView& view : runs) {
        const PartitionIndexRunDescriptor& descriptor =
            *view.catalog_descriptor;
        for (std::size_t byte_index = 0;
             byte_index != descriptor.active_partitions.size(); ++byte_index) {
            std::uint8_t bits = std::to_integer<std::uint8_t>(
                descriptor.active_partitions[byte_index]);
            while (bits != 0) {
                const unsigned local = std::countr_zero(bits);
                const std::uint64_t partition =
                    static_cast<std::uint64_t>(byte_index) * 8 + local;
                bits &= static_cast<std::uint8_t>(bits - 1);
                require<CorruptionError>(
                    partition < partitions,
                    "partition fold active bit is outside the layout");
                validate_active_epoch(
                    snapshots[static_cast<std::size_t>(partition)],
                    descriptor);
                const PartitionRunTocEntry section =
                    view.reader->partition_records(
                        static_cast<std::uint32_t>(partition));
                require<CorruptionError>(
                    section.record_count != 0,
                    "partition fold active bit names an empty run section");
                PartitionFoldCandidate& candidate =
                    candidates[static_cast<std::size_t>(partition)];
                candidate.active_run_count = checked_sum<CorruptionError>(
                    candidate.active_run_count, 1,
                    "partition fold active run count");
                candidate.run_entries = checked_sum<CorruptionError>(
                    candidate.run_entries, section.record_count,
                    "partition fold run entries");
                const std::uint64_t section_bytes =
                    checked_product<CorruptionError>(
                        section.record_count, kPartitionRunRecordSize,
                        "partition fold run section bytes");
                candidate.run_bytes = checked_sum<CorruptionError>(
                    candidate.run_bytes, section_bytes,
                    "partition fold run bytes");
                candidate.input_entries = checked_sum<CorruptionError>(
                    candidate.input_entries, section.record_count,
                    "partition fold total entries");
                candidate.input_bytes = checked_sum<CorruptionError>(
                    candidate.input_bytes, section_bytes,
                    "partition fold total bytes");
            }
        }
    }

    std::optional<PartitionFoldCandidate> selected;
    for (const PartitionFoldCandidate& candidate : candidates) {
        if (candidate.active_run_count < budget.min_runs
            || candidate.active_run_count > budget.max_runs
            || candidate.input_entries > budget.max_input_entries
            || candidate.input_bytes > budget.max_input_bytes) {
            continue;
        }
        if (!selected.has_value()
            || candidate.active_run_count > selected->active_run_count
            || (candidate.active_run_count == selected->active_run_count
                && candidate.run_entries > selected->run_entries)
            || (candidate.active_run_count == selected->active_run_count
                && candidate.run_entries == selected->run_entries
                && candidate.partition_id < selected->partition_id)) {
            selected = candidate;
        }
    }
    return selected;
}

std::vector<PartitionRunRecord> merge_partition_for_fold(
    const std::uint32_t partition_bits,
    const std::uint32_t partition_id,
    const PartitionSnapshotReader& snapshot,
    const PartitionRunViews runs) {
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);
    require<InvalidArgumentError>(
        partition_id < partitions,
        "partition fold partition ID is outside the layout");
    require<InvalidArgumentError>(
        snapshot.descriptor().partition_id == partition_id,
        "partition fold snapshot belongs to a different partition");
    validate_reader_set(runs, partition_bits, true);

    std::uint64_t maximum_versions = snapshot.record_count();
    for (const PartitionRunView& view : runs) {
        if (!bitmap_contains(
                view.catalog_descriptor->active_partitions, partition_id)) {
            continue;
        }
        validate_active_epoch(
            snapshot.descriptor(), *view.catalog_descriptor);
        maximum_versions = checked_sum<CorruptionError>(
            maximum_versions,
            view.reader->partition_records(partition_id).record_count,
            "partition fold merge versions");
    }
    require<InvalidArgumentError>(
        maximum_versions <= std::numeric_limits<std::size_t>::max(),
        "partition fold merge does not fit memory");

    std::unordered_map<std::uint64_t, WinningValue> winners;
    winners.reserve(static_cast<std::size_t>(maximum_versions));
    const auto merge_value = [&winners](
                                 const std::uint64_t id,
                                 const std::uint64_t sequence,
                                 const std::uint64_t value_raw,
                                 const std::uint64_t source_epoch,
                                 const PartitionIndexRunDescriptor* source) {
        const auto [position, inserted] = winners.emplace(
            id, WinningValue{sequence, value_raw, source_epoch});
        if (inserted) {
            return;
        }
        position->second = reduce_winner(
            position->second, sequence, value_raw, source_epoch, source);
    };

    for (std::uint64_t slot = 0; slot != snapshot.slot_count(); ++slot) {
        const auto record = snapshot.slot_record(slot);
        if (!record.has_value()) {
            continue;
        }
        merge_value(
            record->id, record->sequence, record->row_ref.raw(),
            snapshot.descriptor().epoch, nullptr);
    }
    for (const PartitionRunView& view : runs) {
        const PartitionIndexRunDescriptor& descriptor =
            *view.catalog_descriptor;
        if (!bitmap_contains(descriptor.active_partitions, partition_id)) {
            continue;
        }
        const PartitionRunTocEntry section =
            view.reader->partition_records(partition_id);
        for (std::uint64_t local = 0;
             local != section.record_count; ++local) {
            const PartitionRunRecord record =
                view.reader->record_at(section.first_record + local);
            merge_value(
                record.id, record.sequence, record.value_raw,
                descriptor.epoch, &descriptor);
        }
    }

    std::vector<PartitionRunRecord> result;
    result.reserve(winners.size());
    for (const auto& [id, value] : winners) {
        if (value.value_raw != 0) {
            result.push_back(
                PartitionRunRecord{id, value.sequence, value.value_raw});
        }
    }
    std::sort(
        result.begin(), result.end(),
        [](const PartitionRunRecord& left, const PartitionRunRecord& right) {
            return left.id < right.id;
        });
    return result;
}

std::optional<PartitionRunRecord> lookup_partition_version(
    const std::uint32_t partition_bits,
    const std::uint32_t partition_id,
    const std::uint64_t id,
    const PartitionSnapshotReader& snapshot,
    const PartitionRunViews runs) {
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);
    require<InvalidArgumentError>(
        partition_id < partitions,
        "partition lookup partition ID is outside the layout");
    require<InvalidArgumentError>(
        snapshot.descriptor().partition_id == partition_id,
        "partition lookup snapshot belongs to a different partition");
    validate_reader_set(runs, partition_bits, true);

    std::optional<WinningValue> winner;
    if (const auto base = snapshot.lookup(id); base.has_value()) {
        winner = WinningValue{
            base->sequence, base->row_ref.raw(),
            snapshot.descriptor().epoch};
    }
    for (const PartitionRunView& view : runs) {
        const PartitionIndexRunDescriptor& descriptor =
            *view.catalog_descriptor;
        if (!bitmap_contains(descriptor.active_partitions, partition_id)) {
            continue;
        }
        validate_active_epoch(snapshot.descriptor(), descriptor);
        const auto record = raw_run_lookup(
            *view.reader, partition_id, id);
        if (!record.has_value()) {
            continue;
        }
        if (!winner.has_value()) {
            winner = WinningValue{
                record->sequence, record->value_raw, descriptor.epoch};
        } else {
            winner = reduce_winner(
                *winner, record->sequence, record->value_raw,
                descriptor.epoch, &descriptor);
        }
    }
    if (!winner.has_value()) {
        return std::nullopt;
    }
    return PartitionRunRecord{id, winner->sequence, winner->value_raw};
}

void clear_folded_partition(
    std::vector<PartitionIndexRunDescriptor>& runs,
    const std::uint32_t partition_bits,
    const std::uint32_t partition_id) {
    const std::uint64_t partitions =
        checked_partition_count<InvalidArgumentError>(partition_bits);
    require<InvalidArgumentError>(
        partition_id < partitions,
        "folded partition ID is outside the layout");
    validate_run_descriptor_order<CorruptionError>(runs, partition_bits);

    std::vector<PartitionIndexRunDescriptor> updated;
    updated.reserve(runs.size());
    for (const PartitionIndexRunDescriptor& original : runs) {
        PartitionIndexRunDescriptor descriptor = original;
        bitmap_clear(descriptor.active_partitions, partition_id);
        if (bitmap_any(descriptor.active_partitions)) {
            updated.push_back(std::move(descriptor));
        }
    }
    runs.swap(updated);
}

}  // namespace throwdb::detail
