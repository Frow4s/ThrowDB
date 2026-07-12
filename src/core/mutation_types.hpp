#pragma once

#include <bit>
#include <compare>
#include <cstdint>
#include <limits>
#include <optional>

#include "throwdb/errors.hpp"

namespace throwdb::detail {

struct Uuid128 {
    std::uint64_t low{0};
    std::uint64_t high{0};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return low == 0 && high == 0;
    }

    bool operator==(const Uuid128&) const = default;
    auto operator<=>(const Uuid128&) const = default;
};

// Declared in internal.hpp and implemented by the existing immutable index.
// Keeping the declaration here lets this private foundation header remain
// independent of the complete snapshot-format implementation.
[[nodiscard]] std::uint64_t hash_id(
    std::uint64_t id, std::uint64_t seed) noexcept;

// Segment IDs are scoped to one logical collection. Published IDs are never
// reused: a captured CollectionView can therefore resolve a RowRef without an
// ABA-prone generation field in the 64-bit address itself.
class SegmentId {
public:
    static constexpr std::uint32_t kMinimum = 1;
    static constexpr std::uint32_t kMaximum = 0xfffffffeU;
    static constexpr std::uint32_t kInvalid = 0;
    static constexpr std::uint32_t kReserved = 0xffffffffU;

    [[nodiscard]] static SegmentId create(const std::uint64_t value) {
        if (value < kMinimum || value > kMaximum) {
            throw InvalidArgumentError(
                "segment ID must be between 1 and 4294967294");
        }
        return SegmentId(static_cast<std::uint32_t>(value));
    }

    // decode() is for persisted or otherwise untrusted metadata. Invalid
    // values are corruption rather than caller argument errors.
    [[nodiscard]] static SegmentId decode(const std::uint32_t value) {
        if (value < kMinimum || value > kMaximum) {
            throw CorruptionError(
                "row reference contains an invalid or reserved segment ID");
        }
        return SegmentId(value);
    }

    [[nodiscard]] constexpr std::uint32_t value() const noexcept {
        return value_;
    }

    bool operator==(const SegmentId&) const = default;
    auto operator<=>(const SegmentId&) const = default;

private:
    explicit constexpr SegmentId(const std::uint32_t value) noexcept
        : value_(value) {}

    std::uint32_t value_;
};

// A RowRef is a stable physical address within one immutable CollectionView:
//
//   high 32 bits: nonzero, non-reserved SegmentId
//   low  32 bits: zero-based row within that segment
//
// Tombstones are deliberately represented by MutationValueState, not by a
// special RowRef bit pattern.
class RowRef {
public:
    static constexpr std::uint64_t kInvalidRaw = 0;
    static constexpr std::uint64_t kMaximumRowsPerSegment =
        std::uint64_t{1} << 32;

    [[nodiscard]] static RowRef create(
        const SegmentId segment_id, const std::uint64_t row_index) {
        if (row_index >= kMaximumRowsPerSegment) {
            throw InvalidArgumentError(
                "row index must fit in an unsigned 32-bit value");
        }
        return RowRef(
            (static_cast<std::uint64_t>(segment_id.value()) << 32)
            | row_index);
    }

    [[nodiscard]] static RowRef create(
        const std::uint64_t segment_id, const std::uint64_t row_index) {
        return create(SegmentId::create(segment_id), row_index);
    }

    [[nodiscard]] static RowRef decode(const std::uint64_t raw) {
        if (raw == kInvalidRaw) {
            throw CorruptionError("row reference zero is invalid");
        }
        const auto segment = static_cast<std::uint32_t>(raw >> 32);
        // Validate both segment zero and the reserved all-ones segment. The
        // lower 32 bits are intrinsically a representable row index.
        static_cast<void>(SegmentId::decode(segment));
        return RowRef(raw);
    }

    [[nodiscard]] constexpr std::uint64_t raw() const noexcept {
        return raw_;
    }

    [[nodiscard]] SegmentId segment_id() const {
        return SegmentId::decode(static_cast<std::uint32_t>(raw_ >> 32));
    }

    [[nodiscard]] constexpr std::uint32_t row_index() const noexcept {
        return static_cast<std::uint32_t>(raw_);
    }

    bool operator==(const RowRef&) const = default;
    auto operator<=>(const RowRef&) const = default;

private:
    explicit constexpr RowRef(const std::uint64_t raw) noexcept : raw_(raw) {}

    std::uint64_t raw_;
};

enum class MutationValueState : std::uint8_t {
    Put = 1,
    Tombstone = 2,
};

// The state byte is separate from RowRef so every valid RowRef remains a live
// physical address. In an overlay hash table, control 0 means no entry/fall
// through, Put resolves a row, and Tombstone stops lookup as missing.
class MutationValue {
public:
    [[nodiscard]] static constexpr MutationValue put(
        const RowRef row_ref) noexcept {
        return MutationValue(MutationValueState::Put, row_ref.raw());
    }

    [[nodiscard]] static constexpr MutationValue tombstone() noexcept {
        return MutationValue(MutationValueState::Tombstone, 0);
    }

    [[nodiscard]] static MutationValue decode(
        const std::uint8_t encoded_state,
        const std::uint64_t encoded_row_ref) {
        switch (static_cast<MutationValueState>(encoded_state)) {
            case MutationValueState::Put:
                return put(RowRef::decode(encoded_row_ref));
            case MutationValueState::Tombstone:
                if (encoded_row_ref != 0) {
                    throw CorruptionError(
                        "mutation tombstone contains a nonzero row reference");
                }
                return tombstone();
        }
        throw CorruptionError("mutation value contains an invalid state");
    }

    [[nodiscard]] constexpr MutationValueState state() const noexcept {
        return state_;
    }

    [[nodiscard]] constexpr std::uint8_t encoded_state() const noexcept {
        return static_cast<std::uint8_t>(state_);
    }

    [[nodiscard]] constexpr std::uint64_t encoded_row_ref() const noexcept {
        return raw_row_ref_;
    }

    [[nodiscard]] constexpr bool is_put() const noexcept {
        return state_ == MutationValueState::Put;
    }

    [[nodiscard]] constexpr bool is_tombstone() const noexcept {
        return state_ == MutationValueState::Tombstone;
    }

    [[nodiscard]] std::optional<RowRef> row_ref() const {
        if (is_tombstone()) {
            return std::nullopt;
        }
        return RowRef::decode(raw_row_ref_);
    }

    bool operator==(const MutationValue&) const = default;

private:
    explicit constexpr MutationValue(
        const MutationValueState state,
        const std::uint64_t raw_row_ref) noexcept
        : state_(state), raw_row_ref_(raw_row_ref) {}

    MutationValueState state_;
    std::uint64_t raw_row_ref_;
};

class PartitionLayout {
public:
    // A partition ID is uint32. Allowing 32 bits is safe: partition_count() is
    // uint64 and partition extraction never shifts by 64.
    static constexpr std::uint32_t kMaximumPartitionBits = 32;

    [[nodiscard]] static PartitionLayout create(
        const std::uint64_t partition_bits) {
        if (partition_bits > kMaximumPartitionBits) {
            throw InvalidArgumentError(
                "partition_bits must be between 0 and 32");
        }
        return PartitionLayout(static_cast<std::uint32_t>(partition_bits));
    }

    [[nodiscard]] static PartitionLayout decode(
        const std::uint32_t partition_bits) {
        if (partition_bits > kMaximumPartitionBits) {
            throw CorruptionError(
                "partition layout contains more than 32 partition bits");
        }
        return PartitionLayout(partition_bits);
    }

    [[nodiscard]] constexpr std::uint32_t partition_bits() const noexcept {
        return partition_bits_;
    }

    [[nodiscard]] constexpr std::uint64_t partition_count() const noexcept {
        return std::uint64_t{1} << partition_bits_;
    }

    [[nodiscard]] constexpr std::uint32_t partition_for_hash(
        const std::uint64_t hash) const noexcept {
        if (partition_bits_ == 0) {
            return 0;
        }
        return static_cast<std::uint32_t>(
            hash >> (64 - partition_bits_));
    }

    [[nodiscard]] std::uint32_t partition_for(
        const std::uint64_t id, const std::uint64_t seed) const noexcept {
        return partition_for_hash(hash_id(id, seed));
    }

    // The local linear-probing table uses low hash bits, disjoint from the high
    // partition bits. Reject layouts whose local table would overlap them.
    [[nodiscard]] std::uint64_t local_home_slot_for_hash(
        const std::uint64_t hash, const std::uint64_t local_capacity) const {
        if (local_capacity == 0
            || (local_capacity & (local_capacity - 1)) != 0) {
            throw InvalidArgumentError(
                "local index capacity must be a nonzero power of two");
        }
        const std::uint32_t local_bits = static_cast<std::uint32_t>(
            std::bit_width(local_capacity) - 1);
        if (local_bits > 64 - partition_bits_) {
            throw InvalidArgumentError(
                "local index capacity overlaps the partition hash bits");
        }
        return hash & (local_capacity - 1);
    }

    [[nodiscard]] std::uint64_t local_home_slot(
        const std::uint64_t id,
        const std::uint64_t seed,
        const std::uint64_t local_capacity) const {
        return local_home_slot_for_hash(
            hash_id(id, seed), local_capacity);
    }

    bool operator==(const PartitionLayout&) const = default;

private:
    explicit constexpr PartitionLayout(
        const std::uint32_t partition_bits) noexcept
        : partition_bits_(partition_bits) {}

    std::uint32_t partition_bits_;
};

}  // namespace throwdb::detail
