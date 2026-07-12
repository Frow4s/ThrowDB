#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "throwdb/errors.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/mutation_types.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Actual, typename Expected>
void require_equal(const Actual& actual, const Expected& expected,
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
    } catch (...) {
        fail(message + " (wrong exception type)");
    }
    fail(message + " (no exception)");
}

void test_segment_id_boundaries_are_checked() {
    using throwdb::detail::SegmentId;

    require_equal(SegmentId::create(1).value(), std::uint32_t{1},
                  "minimum segment ID was not preserved");
    require_equal(
        SegmentId::create(0xfffffffeULL).value(), std::uint32_t{0xfffffffeU},
        "maximum segment ID was not preserved");

    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(SegmentId::create(0)); },
        "segment ID zero was accepted for construction");
    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(SegmentId::create(0xffffffffULL)); },
        "reserved segment ID was accepted for construction");
    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(SegmentId::create(0x100000000ULL)); },
        "overflowing segment ID was accepted for construction");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(SegmentId::decode(0)); },
        "persisted segment ID zero was not corruption");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(SegmentId::decode(0xffffffffU)); },
        "persisted reserved segment ID was not corruption");
}

void test_row_ref_round_trip_and_boundaries() {
    using throwdb::detail::RowRef;

    const RowRef first = RowRef::create(1, 0);
    require_equal(first.raw(), std::uint64_t{1} << 32,
                  "first RowRef has the wrong encoding");
    require_equal(first.segment_id().value(), std::uint32_t{1},
                  "first RowRef lost its segment ID");
    require_equal(first.row_index(), std::uint32_t{0},
                  "first RowRef lost its row index");

    const RowRef last = RowRef::create(0xfffffffeULL, 0xffffffffULL);
    require_equal(last.raw(), 0xfffffffeffffffffULL,
                  "maximum RowRef has the wrong encoding");
    require_equal(RowRef::decode(last.raw()), last,
                  "maximum RowRef did not round trip");

    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(RowRef::create(1, 0x100000000ULL)); },
        "overflowing row index was accepted");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(RowRef::decode(0)); },
        "zero RowRef was accepted");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(RowRef::decode(123)); },
        "RowRef with segment zero was accepted");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(RowRef::decode(0xffffffff00000000ULL)); },
        "RowRef with the reserved segment was accepted");
}

void test_tombstone_is_out_of_band() {
    using throwdb::detail::MutationValue;
    using throwdb::detail::MutationValueState;
    using throwdb::detail::RowRef;

    const RowRef ref = RowRef::create(17, 99);
    const MutationValue put = MutationValue::put(ref);
    require(put.is_put() && !put.is_tombstone(),
            "PUT mutation has the wrong state");
    require_equal(put.state(), MutationValueState::Put,
                  "PUT state was not preserved");
    require_equal(put.encoded_row_ref(), ref.raw(),
                  "PUT did not preserve RowRef bits");
    require(put.row_ref().has_value(), "PUT does not expose its RowRef");
    require_equal(*put.row_ref(), ref, "PUT exposes the wrong RowRef");

    const MutationValue tombstone = MutationValue::tombstone();
    require(tombstone.is_tombstone() && !tombstone.is_put(),
            "tombstone mutation has the wrong state");
    require_equal(tombstone.state(), MutationValueState::Tombstone,
                  "tombstone state was not preserved");
    require_equal(tombstone.encoded_row_ref(), std::uint64_t{0},
                  "tombstone stored a RowRef sentinel");
    require(!tombstone.row_ref().has_value(),
            "tombstone unexpectedly exposes a RowRef");

    require_equal(
        MutationValue::decode(1, ref.raw()), put,
        "encoded PUT did not round trip");
    require_equal(
        MutationValue::decode(2, 0), tombstone,
        "encoded tombstone did not round trip");
    require_throws<throwdb::CorruptionError>(
        [ref] { static_cast<void>(MutationValue::decode(2, ref.raw())); },
        "tombstone with a nonzero RowRef was accepted");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(MutationValue::decode(0, 0)); },
        "empty hash control was accepted as a mutation value");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(MutationValue::decode(3, 0)); },
        "unknown mutation state was accepted");
}

void test_partition_layout_uses_high_hash_bits() {
    using throwdb::detail::PartitionLayout;

    constexpr std::uint64_t id = 0x0123456789abcdefULL;
    constexpr std::uint64_t seed = 0x6a09e667f3bcc909ULL;
    const std::uint64_t hash = throwdb::detail::hash_id(id, seed);

    const PartitionLayout single = PartitionLayout::create(0);
    require_equal(single.partition_count(), std::uint64_t{1},
                  "zero-bit layout does not have one partition");
    require_equal(single.partition_for(id, seed), std::uint32_t{0},
                  "zero-bit layout routed outside partition zero");

    const PartitionLayout normal = PartitionLayout::create(12);
    require_equal(normal.partition_count(), std::uint64_t{4096},
                  "12-bit layout has the wrong partition count");
    require_equal(
        normal.partition_for(id, seed),
        static_cast<std::uint32_t>(hash >> 52),
        "partition routing does not use the high hash bits");
    require_equal(
        normal.local_home_slot(id, seed, 1U << 18),
        hash & ((std::uint64_t{1} << 18) - 1),
        "local routing does not use the low hash bits");

    const PartitionLayout maximum = PartitionLayout::create(32);
    require_equal(maximum.partition_count(), std::uint64_t{1} << 32,
                  "32-bit layout overflowed its partition count");
    require_equal(
        maximum.partition_for_hash(std::numeric_limits<std::uint64_t>::max()),
        std::numeric_limits<std::uint32_t>::max(),
        "32-bit layout did not preserve the high half of the hash");
}

void test_partition_layout_rejects_unsafe_boundaries() {
    using throwdb::detail::PartitionLayout;

    require_throws<throwdb::InvalidArgumentError>(
        [] { static_cast<void>(PartitionLayout::create(33)); },
        "more than 32 partition bits were accepted");
    require_throws<throwdb::CorruptionError>(
        [] { static_cast<void>(PartitionLayout::decode(33)); },
        "persisted invalid partition width was not corruption");

    const PartitionLayout layout = PartitionLayout::create(32);
    require_throws<throwdb::InvalidArgumentError>(
        [&layout] {
            static_cast<void>(layout.local_home_slot_for_hash(0, 0));
        },
        "zero local index capacity was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&layout] {
            static_cast<void>(layout.local_home_slot_for_hash(0, 3));
        },
        "non-power-of-two local index capacity was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&layout] {
            static_cast<void>(layout.local_home_slot_for_hash(
                0, std::uint64_t{1} << 33));
        },
        "overlapping local and partition hash bits were accepted");

    require_equal(layout.local_home_slot_for_hash(123, 1), std::uint64_t{0},
                  "unit-capacity local table did not route to zero");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 5> tests{{
        {"segment_id_boundaries_are_checked",
         test_segment_id_boundaries_are_checked},
        {"row_ref_round_trip_and_boundaries",
         test_row_ref_round_trip_and_boundaries},
        {"tombstone_is_out_of_band", test_tombstone_is_out_of_band},
        {"partition_layout_uses_high_hash_bits",
         test_partition_layout_uses_high_hash_bits},
        {"partition_layout_rejects_unsafe_boundaries",
         test_partition_layout_rejects_unsafe_boundaries},
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
