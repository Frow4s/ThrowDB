#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/internal.hpp"
#include "../src/core/mutation_wal.hpp"

namespace {

using throwdb::detail::EncodedWalTransaction;
using throwdb::detail::ExpectedWalIdentity;
using throwdb::detail::RecoveredWalBatch;
using throwdb::detail::RecoveredWalOperation;
using throwdb::detail::Uuid128;
using throwdb::detail::WalHeader;
using throwdb::detail::WalLimits;
using throwdb::detail::WalMutation;
using throwdb::detail::WalOperationKind;
using throwdb::detail::WalScanMode;

constexpr std::size_t kWalHeaderCrcOffset = 132;
constexpr std::size_t kBatchHeaderCrcOffset = 108;
constexpr std::size_t kCommitCrcOffset = 48;

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
    } catch (const std::exception& error) {
        fail(message + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        fail(message + " (wrong non-standard exception)");
    }
    fail(message + " (no exception)");
}

[[nodiscard]] WalHeader make_header() {
    WalHeader header;
    header.collection_id = Uuid128{0x0102030405060708ULL,
                                   0x1112131415161718ULL};
    header.schema_id = Uuid128{0x2122232425262728ULL,
                               0x3132333435363738ULL};
    header.wal_id = Uuid128{0x4142434445464748ULL,
                            0x5152535455565758ULL};
    header.predecessor_wal_id = Uuid128{0x6162636465666768ULL,
                                        0x7172737475767778ULL};
    header.previous_sequence = 40;
    header.first_sequence = 41;
    header.record_size = 6;
    header.dimension = 3;
    header.dtype = throwdb::detail::DTypeCode::Float16;
    header.item_size = 2;
    header.id_type = throwdb::detail::IdType::UInt64;
    header.row_layout = throwdb::detail::RowLayout::DenseRowMajor;
    header.creation_time_unix_ns = 0x8182838485868788ULL;
    return header;
}

[[nodiscard]] std::vector<std::byte> make_wal_image(const WalHeader& header) {
    const auto encoded = throwdb::detail::encode_wal_header(header);
    return std::vector<std::byte>(encoded.begin(), encoded.end());
}

std::uint64_t append_transaction(
    std::vector<std::byte>& image,
    const WalHeader& header,
    const std::uint64_t sequence,
    const std::span<const WalMutation> operations,
    const WalLimits& limits = {}) {
    const std::uint64_t offset = image.size();
    const EncodedWalTransaction transaction =
        throwdb::detail::encode_wal_transaction(
            header, sequence, offset, operations, limits);
    image.insert(
        image.end(), transaction.bytes.begin(), transaction.bytes.end());
    return offset;
}

[[nodiscard]] std::uint32_t crc_with_zeroed_u32(
    const std::span<const std::byte> bytes,
    const std::size_t field_offset) {
    require(field_offset + sizeof(std::uint32_t) <= bytes.size(),
            "test checksum field is out of range");
    throwdb::detail::Crc32c checksum;
    checksum.update(bytes.first(field_offset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(field_offset + sizeof(std::uint32_t)));
    return checksum.value();
}

void reseal_batch(
    std::vector<std::byte>& image,
    const std::uint64_t batch_offset,
    const bool header_changed = false) {
    const std::uint64_t frame_size = throwdb::detail::get_u64(
        std::span<const std::byte>(image),
        static_cast<std::size_t>(batch_offset + 80));
    require(frame_size >= throwdb::detail::kWalBatchHeaderSize,
            "test batch frame is unexpectedly short");
    require(batch_offset + frame_size
                + throwdb::detail::kWalCommitMarkerSize
            <= image.size(),
            "test batch does not fit in its image");

    auto batch = std::span<std::byte>(image).subspan(
        static_cast<std::size_t>(batch_offset),
        static_cast<std::size_t>(frame_size));
    if (header_changed) {
        throwdb::detail::put_u32(batch, kBatchHeaderCrcOffset, 0);
        throwdb::detail::put_u32(
            batch, kBatchHeaderCrcOffset,
            crc_with_zeroed_u32(
                std::span<const std::byte>(batch).first(
                    static_cast<std::size_t>(
                        throwdb::detail::kWalBatchHeaderSize)),
                kBatchHeaderCrcOffset));
    }

    auto commit = std::span<std::byte>(image).subspan(
        static_cast<std::size_t>(batch_offset + frame_size),
        static_cast<std::size_t>(throwdb::detail::kWalCommitMarkerSize));
    throwdb::detail::put_u32(
        commit, 40,
        throwdb::detail::crc32c(std::span<const std::byte>(batch)));
    throwdb::detail::put_u32(commit, kCommitCrcOffset, 0);
    throwdb::detail::put_u32(
        commit, kCommitCrcOffset,
        crc_with_zeroed_u32(commit, kCommitCrcOffset));
}

struct CapturedBatch {
    std::uint64_t sequence{0};
    std::uint64_t batch_offset{0};
    std::uint64_t batch_size{0};
    std::vector<RecoveredWalOperation> operations;
};

[[nodiscard]] std::vector<CapturedBatch> scan_and_capture(
    const std::vector<std::byte>& image,
    const WalHeader& header,
    throwdb::detail::WalScanResult* scan_result = nullptr,
    const WalScanMode mode = WalScanMode::Active) {
    std::vector<CapturedBatch> batches;
    const auto result = throwdb::detail::scan_wal(
        image,
        ExpectedWalIdentity::from_trusted_header(header),
        mode,
        [&batches](const RecoveredWalBatch& batch) {
            batches.push_back(CapturedBatch{
                batch.sequence,
                batch.batch_offset,
                batch.batch_size,
                std::vector<RecoveredWalOperation>(
                    batch.operations.begin(), batch.operations.end())});
        });
    if (scan_result != nullptr) {
        *scan_result = result;
    }
    return batches;
}

void require_header_equal(
    const WalHeader& actual,
    const WalHeader& expected,
    const std::string_view context) {
    const std::string prefix(context);
    require_equal(actual.collection_id, expected.collection_id,
                  prefix + ": collection ID changed");
    require_equal(actual.schema_id, expected.schema_id,
                  prefix + ": schema ID changed");
    require_equal(actual.wal_id, expected.wal_id,
                  prefix + ": WAL ID changed");
    require_equal(actual.predecessor_wal_id, expected.predecessor_wal_id,
                  prefix + ": predecessor WAL ID changed");
    require_equal(actual.previous_sequence, expected.previous_sequence,
                  prefix + ": previous sequence changed");
    require_equal(actual.first_sequence, expected.first_sequence,
                  prefix + ": first sequence changed");
    require_equal(actual.record_size, expected.record_size,
                  prefix + ": record size changed");
    require_equal(actual.dimension, expected.dimension,
                  prefix + ": dimension changed");
    require_equal(actual.dtype, expected.dtype,
                  prefix + ": dtype changed");
    require_equal(actual.item_size, expected.item_size,
                  prefix + ": item size changed");
    require_equal(actual.id_type, expected.id_type,
                  prefix + ": ID type changed");
    require_equal(actual.row_layout, expected.row_layout,
                  prefix + ": row layout changed");
    require_equal(actual.required_feature_bits, expected.required_feature_bits,
                  prefix + ": required feature bits changed");
    require_equal(actual.optional_feature_bits, expected.optional_feature_bits,
                  prefix + ": optional feature bits changed");
    require_equal(actual.creation_time_unix_ns, expected.creation_time_unix_ns,
                  prefix + ": creation time changed");
}

void test_crc32c_known_vector_and_incremental_updates() {
    constexpr std::string_view vector = "123456789";
    const auto bytes = std::as_bytes(std::span(vector.data(), vector.size()));
    require_equal(
        throwdb::detail::crc32c(bytes), std::uint32_t{0xe3069283U},
        "CRC32C known vector changed");
    require_equal(
        throwdb::detail::crc32c({}), std::uint32_t{0},
        "CRC32C empty vector changed");

    throwdb::detail::Crc32c incremental;
    incremental.update(bytes.first(2));
    incremental.update(bytes.subspan(2, 3));
    incremental.update(bytes.subspan(5));
    require_equal(
        incremental.value(), std::uint32_t{0xe3069283U},
        "incremental CRC32C differs from the known vector");
}

[[nodiscard]] std::uint32_t reference_crc32c(
    const std::span<const std::byte> bytes) {
    constexpr std::uint32_t polynomial = 0x82f63b78U;
    std::uint32_t state = 0xffffffffU;
    for (const std::byte byte : bytes) {
        state ^= std::to_integer<std::uint8_t>(byte);
        for (unsigned bit = 0; bit != 8; ++bit) {
            state = (state >> 1)
                    ^ ((state & 1U) == 0 ? 0U : polynomial);
        }
    }
    return state ^ 0xffffffffU;
}

void test_crc32c_alignment_fragmentation_and_zero_updates() {
    std::vector<std::byte> data(4104);
    std::uint64_t random = 0x9e3779b97f4a7c15ULL;
    for (std::byte& byte : data) {
        random ^= random >> 12;
        random ^= random << 25;
        random ^= random >> 27;
        byte = static_cast<std::byte>(random);
    }

    for (std::size_t offset = 0; offset != 8; ++offset) {
        for (std::size_t length = 0; length != 257; ++length) {
            const auto bytes = std::span<const std::byte>(data).subspan(
                offset, length);
            require_equal(
                throwdb::detail::crc32c(bytes), reference_crc32c(bytes),
                "CRC32C differs from bitwise reference for an alignment/length");
        }
    }

    const auto large = std::span<const std::byte>(data).subspan(3, 4096);
    throwdb::detail::Crc32c fragmented;
    std::size_t position = 0;
    constexpr std::array<std::size_t, 9> chunk_sizes{
        0, 1, 7, 8, 9, 31, 64, 255, 1024};
    std::size_t chunk_index = 0;
    while (position != large.size()) {
        const std::size_t chunk_size = std::min(
            chunk_sizes[chunk_index++ % chunk_sizes.size()],
            large.size() - position);
        fragmented.update(large.subspan(position, chunk_size));
        position += chunk_size;
    }
    require_equal(
        fragmented.value(), reference_crc32c(large),
        "fragmented CRC32C differs from bitwise reference");

    constexpr std::size_t prefix_size = 13;
    constexpr std::size_t suffix_size = 17;
    for (std::size_t zero_count = 0; zero_count != 130; ++zero_count) {
        std::vector<std::byte> expected;
        expected.insert(
            expected.end(), data.begin(), data.begin() + prefix_size);
        expected.insert(expected.end(), zero_count, std::byte{0});
        expected.insert(
            expected.end(), data.begin() + prefix_size,
            data.begin() + prefix_size + suffix_size);

        throwdb::detail::Crc32c with_zeroes;
        with_zeroes.update(
            std::span<const std::byte>(data).first(prefix_size));
        with_zeroes.update_zeroes(zero_count);
        with_zeroes.update(std::span<const std::byte>(data).subspan(
            prefix_size, suffix_size));
        require_equal(
            with_zeroes.value(), reference_crc32c(expected),
            "update_zeroes differs from explicit zero bytes");
    }
}

void test_wal_header_round_trip_and_corruption() {
    const WalHeader header = make_header();
    auto encoded = throwdb::detail::encode_wal_header(header);

    require(std::memcmp(encoded.data(), "THROWDBW", 8) == 0,
            "WAL header magic changed");
    require_equal(
        throwdb::detail::get_u32(encoded, 8), std::uint32_t{1},
        "WAL header version changed");
    require_equal(
        throwdb::detail::get_u32(encoded, 12),
        static_cast<std::uint32_t>(throwdb::detail::kWalFileHeaderSize),
        "WAL header size field changed");
    require_equal(
        throwdb::detail::get_u64(encoded, 80), header.previous_sequence,
        "WAL previous-sequence offset changed");
    require_equal(
        throwdb::detail::get_u64(encoded, 96), header.record_size,
        "WAL record-size offset changed");
    require_equal(
        throwdb::detail::get_u32(encoded, kWalHeaderCrcOffset),
        crc_with_zeroed_u32(encoded, kWalHeaderCrcOffset),
        "WAL header CRC does not cover the canonical bytes");
    require_header_equal(
        throwdb::detail::decode_wal_header(encoded), header,
        "WAL header round trip");

    auto corrupted = encoded;
    corrupted[16] ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&corrupted] {
            static_cast<void>(throwdb::detail::decode_wal_header(corrupted));
        },
        "WAL header corruption was accepted");
    require_throws<throwdb::CorruptionError>(
        [&encoded] {
            static_cast<void>(throwdb::detail::decode_wal_header(
                std::span<const std::byte>(encoded).first(
                    encoded.size() - 1)));
        },
        "truncated referenced WAL header was accepted");
}

void test_mixed_transaction_has_golden_offsets_and_bit_exact_payloads() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> first_row{
        std::byte{0x00}, std::byte{0x80}, std::byte{0x01},
        std::byte{0x7e}, std::byte{0xff}, std::byte{0x03}};
    const std::array<std::byte, 6> second_row{
        std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc},
        std::byte{0xdd}, std::byte{0xee}, std::byte{0xff}};
    const std::array<WalMutation, 3> operations{{
        WalMutation{0, WalOperationKind::Put, first_row},
        WalMutation{std::numeric_limits<std::uint64_t>::max(),
                    WalOperationKind::Delete, {}},
        WalMutation{42, WalOperationKind::Put, second_row},
    }};

    std::vector<std::byte> image = make_wal_image(header);
    const std::uint64_t batch_offset = append_transaction(
        image, header, header.first_sequence, operations);
    require_equal(batch_offset, throwdb::detail::kWalFileHeaderSize,
                  "first batch did not follow the WAL header");

    const auto bytes = std::span<const std::byte>(image);
    require(std::memcmp(
                bytes.data() + static_cast<std::size_t>(batch_offset),
                "TDBWBAT1", 8)
                == 0,
            "batch magic changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 16),
                  header.first_sequence,
                  "batch sequence offset changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 24),
                  std::uint64_t{3},
                  "operation count offset changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 32),
                  std::uint64_t{2},
                  "PUT count offset changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 48),
                  throwdb::detail::kWalBatchHeaderSize,
                  "operation table offset changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 56),
                  std::uint64_t{48},
                  "operation table size changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 64),
                  std::uint64_t{176},
                  "dense payload offset changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 72),
                  std::uint64_t{12},
                  "dense payload size changed");
    require_equal(throwdb::detail::get_u64(bytes, batch_offset + 80),
                  std::uint64_t{192},
                  "aligned frame size changed");

    const std::size_t table = static_cast<std::size_t>(
        batch_offset + throwdb::detail::kWalBatchHeaderSize);
    require_equal(std::to_integer<std::uint8_t>(image[table]),
                  std::uint8_t{1}, "first operation is not PUT");
    require_equal(throwdb::detail::get_u64(bytes, table + 8),
                  std::uint64_t{0}, "ID zero changed");
    require_equal(std::to_integer<std::uint8_t>(image[table + 16]),
                  std::uint8_t{2}, "second operation is not DELETE");
    require_equal(throwdb::detail::get_u64(bytes, table + 24),
                  std::numeric_limits<std::uint64_t>::max(),
                  "UINT64_MAX ID changed");
    require_equal(std::to_integer<std::uint8_t>(image[table + 32]),
                  std::uint8_t{1}, "third operation is not PUT");
    require_equal(throwdb::detail::get_u64(bytes, table + 40),
                  std::uint64_t{42}, "third ID changed");

    const std::size_t payload = static_cast<std::size_t>(batch_offset + 176);
    require(std::equal(first_row.begin(), first_row.end(),
                       image.begin() + static_cast<std::ptrdiff_t>(payload)),
            "first dense PUT row changed bits");
    require(std::equal(second_row.begin(), second_row.end(),
                       image.begin()
                           + static_cast<std::ptrdiff_t>(payload + first_row.size())),
            "second dense PUT row changed bits");
    require(std::all_of(
                image.begin() + static_cast<std::ptrdiff_t>(payload + 12),
                image.begin() + static_cast<std::ptrdiff_t>(batch_offset + 192),
                [](const std::byte byte) { return byte == std::byte{0}; }),
            "batch alignment padding is not zero");

    const std::uint64_t commit_offset = batch_offset + 192;
    require(std::memcmp(
                bytes.data() + static_cast<std::size_t>(commit_offset),
                "TDBWCOM1", 8)
                == 0,
            "commit magic changed");
    require_equal(throwdb::detail::get_u64(bytes, commit_offset + 16),
                  header.first_sequence,
                  "commit sequence does not bind the batch");
    require_equal(throwdb::detail::get_u64(bytes, commit_offset + 24),
                  batch_offset,
                  "commit offset does not bind the batch");
    require_equal(throwdb::detail::get_u64(bytes, commit_offset + 32),
                  std::uint64_t{192},
                  "commit size does not bind the batch");
    require_equal(
        throwdb::detail::get_u32(bytes, commit_offset + 40),
        throwdb::detail::crc32c(bytes.subspan(
            static_cast<std::size_t>(batch_offset), 192)),
        "commit stores the wrong batch CRC");

    throwdb::detail::WalScanResult result;
    const auto batches = scan_and_capture(image, header, &result);
    require_equal(batches.size(), std::size_t{1},
                  "mixed WAL did not recover one batch");
    require_equal(result.valid_end, static_cast<std::uint64_t>(image.size()),
                  "committed WAL did not consume the complete image");
    require_equal(result.next_sequence, header.first_sequence + 1,
                  "mixed WAL returned the wrong next sequence");
    require_equal(batches[0].operations.size(), std::size_t{3},
                  "mixed WAL recovered the wrong operation count");
    require_equal(batches[0].operations[0].payload_offset,
                  batch_offset + 176,
                  "first PUT points outside the dense payload");
    require_equal(batches[0].operations[0].payload_size,
                  header.record_size,
                  "first PUT has the wrong payload size");
    require_equal(batches[0].operations[1].payload_offset, std::uint64_t{0},
                  "DELETE unexpectedly points at a payload");
    require_equal(batches[0].operations[1].payload_size, std::uint64_t{0},
                  "DELETE unexpectedly has a payload size");
    require_equal(batches[0].operations[2].payload_offset,
                  batch_offset + 182,
                  "second PUT is not densely packed after the first");
}

void test_every_truncation_of_third_transaction_is_ignored_as_tail() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row_a{
        std::byte{1}, std::byte{2}, std::byte{3},
        std::byte{4}, std::byte{5}, std::byte{6}};
    const std::array<std::byte, 6> row_b{
        std::byte{7}, std::byte{8}, std::byte{9},
        std::byte{10}, std::byte{11}, std::byte{12}};
    const std::array<WalMutation, 1> first{{
        WalMutation{1, WalOperationKind::Put, row_a},
    }};
    const std::array<WalMutation, 1> second{{
        WalMutation{2, WalOperationKind::Delete, {}},
    }};
    const std::array<WalMutation, 2> third{{
        WalMutation{3, WalOperationKind::Put, row_b},
        WalMutation{4, WalOperationKind::Delete, {}},
    }};

    std::vector<std::byte> committed_prefix = make_wal_image(header);
    append_transaction(
        committed_prefix, header, header.first_sequence, first);
    append_transaction(
        committed_prefix, header, header.first_sequence + 1, second);
    const std::uint64_t third_offset = committed_prefix.size();
    const EncodedWalTransaction third_transaction =
        throwdb::detail::encode_wal_transaction(
            header, header.first_sequence + 2, third_offset, third);

    for (std::size_t cut = 0; cut != third_transaction.bytes.size(); ++cut) {
        std::vector<std::byte> truncated = committed_prefix;
        truncated.insert(
            truncated.end(), third_transaction.bytes.begin(),
            third_transaction.bytes.begin()
                + static_cast<std::ptrdiff_t>(cut));
        throwdb::detail::WalScanResult result;
        const auto batches = scan_and_capture(truncated, header, &result);
        require_equal(
            batches.size(), std::size_t{2},
            "truncated third transaction became visible at cut "
                + std::to_string(cut));
        require_equal(
            result.valid_end, third_offset,
            "truncated third transaction changed valid_end at cut "
                + std::to_string(cut));
        require_equal(
            result.next_sequence, header.first_sequence + 2,
            "truncated third transaction consumed a sequence at cut "
                + std::to_string(cut));
        require_equal(
            result.ignored_tail_bytes, static_cast<std::uint64_t>(cut),
            "scanner reported the wrong tail length at cut "
                + std::to_string(cut));
    }

    std::vector<std::byte> complete = committed_prefix;
    complete.insert(complete.end(), third_transaction.bytes.begin(),
                    third_transaction.bytes.end());
    throwdb::detail::WalScanResult result;
    const auto batches = scan_and_capture(complete, header, &result);
    require_equal(batches.size(), std::size_t{3},
                  "complete third transaction was not recovered");
    require_equal(result.ignored_tail_bytes, std::uint64_t{0},
                  "complete WAL reported an ignored tail");
}

void test_interior_corruption_is_rejected_in_every_frame_region() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{0x10}, std::byte{0x20}, std::byte{0x30},
        std::byte{0x40}, std::byte{0x50}, std::byte{0x60}};
    const std::array<WalMutation, 1> first{{
        WalMutation{1, WalOperationKind::Put, row},
    }};
    const std::array<WalMutation, 2> middle{{
        WalMutation{2, WalOperationKind::Put, row},
        WalMutation{3, WalOperationKind::Delete, {}},
    }};
    const std::array<WalMutation, 1> last{{
        WalMutation{4, WalOperationKind::Put, row},
    }};

    std::vector<std::byte> valid = make_wal_image(header);
    append_transaction(valid, header, header.first_sequence, first);
    const std::uint64_t middle_offset = append_transaction(
        valid, header, header.first_sequence + 1, middle);
    append_transaction(valid, header, header.first_sequence + 2, last);
    const auto bytes = std::span<const std::byte>(valid);
    const std::uint64_t payload_offset =
        throwdb::detail::get_u64(bytes, middle_offset + 64);
    const std::uint64_t payload_size =
        throwdb::detail::get_u64(bytes, middle_offset + 72);
    const std::uint64_t frame_size =
        throwdb::detail::get_u64(bytes, middle_offset + 80);
    require(middle_offset + payload_offset + payload_size <
                middle_offset + frame_size,
            "middle test frame has no padding");

    const std::array<std::pair<const char*, std::uint64_t>, 5> flips{{
        {"header", middle_offset + 16},
        {"operation table", middle_offset
                                + throwdb::detail::kWalBatchHeaderSize + 8},
        {"payload", middle_offset + payload_offset},
        {"padding", middle_offset + payload_offset + payload_size},
        {"commit", middle_offset + frame_size + 40},
    }};
    for (const auto& [region, offset] : flips) {
        auto corrupted = valid;
        corrupted[static_cast<std::size_t>(offset)] ^= std::byte{1};
        require_throws<throwdb::CorruptionError>(
            [&corrupted, &header] {
                static_cast<void>(throwdb::detail::scan_wal(
                    corrupted, ExpectedWalIdentity::from_trusted_header(header),
                    WalScanMode::Active));
            },
            std::string("interior ") + region
                + " corruption was silently accepted");
    }
}

void test_forged_structural_errors_are_rejected_with_valid_checksums() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{0x21}, std::byte{0x22}, std::byte{0x23},
        std::byte{0x24}, std::byte{0x25}, std::byte{0x26}};
    const std::array<WalMutation, 2> operations{{
        WalMutation{100, WalOperationKind::Put, row},
        WalMutation{200, WalOperationKind::Delete, {}},
    }};

    std::vector<std::byte> valid = make_wal_image(header);
    const std::uint64_t batch_offset = append_transaction(
        valid, header, header.first_sequence, operations);
    const auto valid_bytes = std::span<const std::byte>(valid);
    const std::uint64_t payload_offset =
        throwdb::detail::get_u64(valid_bytes, batch_offset + 64);
    const std::uint64_t payload_size =
        throwdb::detail::get_u64(valid_bytes, batch_offset + 72);
    const std::uint64_t frame_size =
        throwdb::detail::get_u64(valid_bytes, batch_offset + 80);
    require(payload_offset + payload_size < frame_size,
            "forged-error test frame has no padding");

    {
        auto duplicate = valid;
        throwdb::detail::put_u64(
            duplicate,
            static_cast<std::size_t>(batch_offset
                                     + throwdb::detail::kWalBatchHeaderSize
                                     + throwdb::detail::kWalOperationEntrySize
                                     + 8),
            100);
        reseal_batch(duplicate, batch_offset);
        require_throws<throwdb::CorruptionError>(
            [&duplicate, &header] {
                static_cast<void>(throwdb::detail::scan_wal(
                    duplicate, ExpectedWalIdentity::from_trusted_header(header),
                    WalScanMode::Active));
            },
            "duplicate ID with valid checksums was accepted");
    }
    {
        auto unknown_opcode = valid;
        unknown_opcode[static_cast<std::size_t>(
            batch_offset + throwdb::detail::kWalBatchHeaderSize)] =
            std::byte{0x7f};
        reseal_batch(unknown_opcode, batch_offset);
        require_throws<throwdb::CorruptionError>(
            [&unknown_opcode, &header] {
                static_cast<void>(
                    throwdb::detail::scan_wal(
                        unknown_opcode,
                        ExpectedWalIdentity::from_trusted_header(header),
                        WalScanMode::Active));
            },
            "unknown opcode with valid checksums was accepted");
    }
    {
        auto gap = valid;
        throwdb::detail::put_u64(
            gap, static_cast<std::size_t>(batch_offset + 16),
            header.first_sequence + 1);
        throwdb::detail::put_u64(
            gap, static_cast<std::size_t>(batch_offset + frame_size + 16),
            header.first_sequence + 1);
        reseal_batch(gap, batch_offset, true);
        require_throws<throwdb::CorruptionError>(
            [&gap, &header] {
                static_cast<void>(throwdb::detail::scan_wal(
                    gap, ExpectedWalIdentity::from_trusted_header(header),
                    WalScanMode::Active));
            },
            "sequence gap with valid checksums was accepted");
    }
    {
        auto wrong_binding = valid;
        auto commit = std::span<std::byte>(wrong_binding).subspan(
            static_cast<std::size_t>(batch_offset + frame_size),
            static_cast<std::size_t>(throwdb::detail::kWalCommitMarkerSize));
        throwdb::detail::put_u64(commit, 24, batch_offset + 8);
        throwdb::detail::put_u32(commit, kCommitCrcOffset, 0);
        throwdb::detail::put_u32(
            commit, kCommitCrcOffset,
            crc_with_zeroed_u32(commit, kCommitCrcOffset));
        require_throws<throwdb::CorruptionError>(
            [&wrong_binding, &header] {
                static_cast<void>(throwdb::detail::scan_wal(
                    wrong_binding,
                    ExpectedWalIdentity::from_trusted_header(header),
                    WalScanMode::Active));
            },
            "wrong commit binding with a valid marker checksum was accepted");
    }
    const auto require_invalid_resealed_marker_rejected =
        [&valid, batch_offset, frame_size, &header](
            const std::string& description, const auto& mutate) {
            auto forged = valid;
            auto commit = std::span<std::byte>(forged).subspan(
                static_cast<std::size_t>(batch_offset + frame_size),
                static_cast<std::size_t>(
                    throwdb::detail::kWalCommitMarkerSize));
            mutate(commit);
            throwdb::detail::put_u32(commit, kCommitCrcOffset, 0);
            throwdb::detail::put_u32(
                commit, kCommitCrcOffset,
                crc_with_zeroed_u32(commit, kCommitCrcOffset));
            require_throws<throwdb::CorruptionError>(
                [&forged, &header] {
                    static_cast<void>(throwdb::detail::scan_wal(
                        forged,
                        ExpectedWalIdentity::from_trusted_header(header),
                        WalScanMode::Active));
                },
                description);
        };
    require_invalid_resealed_marker_rejected(
        "CRC-valid marker with unknown version was discarded as active tail",
        [](const std::span<std::byte> commit) {
            throwdb::detail::put_u32(commit, 8, 2);
        });
    require_invalid_resealed_marker_rejected(
        "CRC-valid marker with flags was discarded as active tail",
        [](const std::span<std::byte> commit) {
            throwdb::detail::put_u32(commit, 52, 1);
        });
    require_invalid_resealed_marker_rejected(
        "CRC-valid marker with reserved data was discarded as active tail",
        [](const std::span<std::byte> commit) {
            throwdb::detail::put_u64(commit, 56, 1);
        });
    require_invalid_resealed_marker_rejected(
        "CRC-valid marker with reserved sequence was discarded as active tail",
        [](const std::span<std::byte> commit) {
            throwdb::detail::put_u64(commit, 16, 0);
        });
    require_invalid_resealed_marker_rejected(
        "CRC-valid marker with bad magic was discarded as active tail",
        [](const std::span<std::byte> commit) {
            commit[0] ^= std::byte{1};
        });
    {
        auto nonzero_padding = valid;
        nonzero_padding[static_cast<std::size_t>(
            batch_offset + payload_offset + payload_size)] = std::byte{1};
        reseal_batch(nonzero_padding, batch_offset);
        require_throws<throwdb::CorruptionError>(
            [&nonzero_padding, &header] {
                static_cast<void>(
                    throwdb::detail::scan_wal(
                        nonzero_padding,
                        ExpectedWalIdentity::from_trusted_header(header),
                        WalScanMode::Active));
            },
            "nonzero padding with valid checksums was accepted");
    }
}

void test_encoder_rejects_invalid_batches_before_encoding() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{1}, std::byte{2}, std::byte{3},
        std::byte{4}, std::byte{5}, std::byte{6}};
    const std::array<std::byte, 5> short_row{
        std::byte{1}, std::byte{2}, std::byte{3},
        std::byte{4}, std::byte{5}};
    const std::uint64_t offset = throwdb::detail::kWalFileHeaderSize;

    require_throws<throwdb::InvalidArgumentError>(
        [&header] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, {}));
        },
        "empty WAL batch was encoded");

    const std::array<WalMutation, 2> duplicates{{
        WalMutation{9, WalOperationKind::Put, row},
        WalMutation{9, WalOperationKind::Delete, {}},
    }};
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &duplicates] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, duplicates));
        },
        "duplicate IDs were encoded");

    const std::array<WalMutation, 1> short_put{{
        WalMutation{1, WalOperationKind::Put, short_row},
    }};
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &short_put] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, short_put));
        },
        "short PUT payload was encoded");

    const std::array<WalMutation, 1> delete_with_payload{{
        WalMutation{1, WalOperationKind::Delete, row},
    }};
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &delete_with_payload] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset,
                delete_with_payload));
        },
        "DELETE payload was encoded");

    const std::array<WalMutation, 1> unknown_opcode{{
        WalMutation{1, static_cast<WalOperationKind>(99), {}},
    }};
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &unknown_opcode] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, unknown_opcode));
        },
        "unknown operation kind was encoded");

    const std::array<WalMutation, 1> valid_put{{
        WalMutation{1, WalOperationKind::Put, row},
    }};
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &valid_put] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence,
                throwdb::detail::kWalFileHeaderSize + 1, valid_put));
        },
        "unaligned WAL batch offset was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &valid_put] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, valid_put,
                WalLimits{1, 144}));
        },
        "frame larger than max_frame_size was encoded");
    require_throws<throwdb::InvalidArgumentError>(
        [&header, &valid_put] {
            static_cast<void>(throwdb::detail::encode_wal_transaction(
                header, header.first_sequence, offset, valid_put,
                WalLimits{
                    throwdb::detail::kWalFormatMaxOperations + 1,
                    throwdb::detail::kWalFormatMaxFrameSize}));
        },
        "writer limit larger than the persisted format maximum was accepted");
}

void test_tail_repair_allows_one_exact_next_append() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{0xa1}, std::byte{0xa2}, std::byte{0xa3},
        std::byte{0xa4}, std::byte{0xa5}, std::byte{0xa6}};
    const std::array<WalMutation, 1> put{{
        WalMutation{10, WalOperationKind::Put, row},
    }};
    const std::array<WalMutation, 1> erase{{
        WalMutation{20, WalOperationKind::Delete, {}},
    }};

    std::vector<std::byte> image = make_wal_image(header);
    append_transaction(image, header, header.first_sequence, put);
    append_transaction(image, header, header.first_sequence + 1, erase);
    const std::uint64_t durable_end = image.size();

    const EncodedWalTransaction torn =
        throwdb::detail::encode_wal_transaction(
            header, header.first_sequence + 2, durable_end, put);
    const std::size_t partial_size = torn.bytes.size() / 2;
    image.insert(image.end(), torn.bytes.begin(),
                 torn.bytes.begin() + static_cast<std::ptrdiff_t>(partial_size));

    throwdb::detail::WalScanResult before_repair;
    const auto before_batches = scan_and_capture(
        image, header, &before_repair);
    require_equal(before_batches.size(), std::size_t{2},
                  "torn tail became visible before repair");
    require_equal(before_repair.valid_end, durable_end,
                  "scanner returned the wrong truncate boundary");
    require_equal(before_repair.next_sequence, header.first_sequence + 2,
                  "torn tail consumed the next sequence");
    require_equal(before_repair.ignored_tail_bytes,
                  static_cast<std::uint64_t>(partial_size),
                  "scanner returned the wrong torn-tail size");

    image.resize(static_cast<std::size_t>(before_repair.valid_end));
    const std::array<WalMutation, 1> replacement{{
        WalMutation{30, WalOperationKind::Put, row},
    }};
    append_transaction(
        image, header, before_repair.next_sequence, replacement);

    throwdb::detail::WalScanResult after_repair;
    const auto batches = scan_and_capture(image, header, &after_repair);
    require_equal(batches.size(), std::size_t{3},
                  "replacement transaction was not recovered");
    require_equal(after_repair.valid_end,
                  static_cast<std::uint64_t>(image.size()),
                  "repaired WAL still has an ignored tail");
    require_equal(after_repair.ignored_tail_bytes, std::uint64_t{0},
                  "repaired WAL reported tail bytes");
    for (std::size_t index = 0; index != batches.size(); ++index) {
        require_equal(
            batches[index].sequence,
            header.first_sequence + static_cast<std::uint64_t>(index),
            "repaired WAL sequence is duplicated or missing");
    }
    require_equal(batches.back().operations[0].id, std::uint64_t{30},
                  "replacement transaction recovered the torn operation");
}

void test_active_full_length_torn_tail_is_repaired_but_frozen_rejects_it() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{1}, std::byte{2}, std::byte{3},
        std::byte{4}, std::byte{5}, std::byte{6}};
    const std::array<WalMutation, 1> operations{{
        WalMutation{7, WalOperationKind::Put, row},
    }};

    std::vector<std::byte> prefix = make_wal_image(header);
    append_transaction(prefix, header, header.first_sequence, operations);
    const std::uint64_t valid_end = prefix.size();

    const EncodedWalTransaction next =
        throwdb::detail::encode_wal_transaction(
            header, header.first_sequence + 1, valid_end, operations);
    std::vector<std::byte> torn_marker = prefix;
    torn_marker.insert(
        torn_marker.end(), next.bytes.begin(), next.bytes.end());
    torn_marker[static_cast<std::size_t>(
        valid_end + next.batch_size + kCommitCrcOffset)] ^= std::byte{1};

    throwdb::detail::WalScanResult active_result;
    const auto active_batches = scan_and_capture(
        torn_marker, header, &active_result, WalScanMode::Active);
    require_equal(active_batches.size(), std::size_t{1},
                  "full-length torn marker became visible in active WAL");
    require_equal(active_result.valid_end, valid_end,
                  "active WAL did not return the pre-tail truncate point");
    require_equal(
        active_result.ignored_tail_bytes,
        static_cast<std::uint64_t>(next.bytes.size()),
        "active WAL reported the wrong full-length torn tail size");
    require_throws<throwdb::CorruptionError>(
        [&torn_marker, &header] {
            static_cast<void>(throwdb::detail::scan_wal(
                torn_marker, ExpectedWalIdentity::from_trusted_header(header),
                WalScanMode::Frozen));
        },
        "frozen WAL accepted a full-length torn marker");

    std::vector<std::byte> damaged_batch = prefix;
    damaged_batch.insert(
        damaged_batch.end(), next.bytes.begin(), next.bytes.end());
    const std::uint64_t payload_offset = throwdb::detail::get_u64(
        damaged_batch, static_cast<std::size_t>(valid_end + 64));
    damaged_batch[static_cast<std::size_t>(
        valid_end + payload_offset)] ^= std::byte{1};
    require_throws<throwdb::CorruptionError>(
        [&damaged_batch, &header] {
            static_cast<void>(throwdb::detail::scan_wal(
                damaged_batch,
                ExpectedWalIdentity::from_trusted_header(header),
                WalScanMode::Active));
        },
        "active WAL discarded a damaged BATCH with an intact COMMIT");

    std::vector<std::byte> zero_filled_tail = prefix;
    zero_filled_tail.resize(prefix.size() + next.bytes.size(), std::byte{0});
    const auto zero_batches = scan_and_capture(
        zero_filled_tail, header, nullptr, WalScanMode::Active);
    require_equal(zero_batches.size(), std::size_t{1},
                  "zero-filled active tail became visible");
    require_throws<throwdb::CorruptionError>(
        [&zero_filled_tail, &header] {
            static_cast<void>(throwdb::detail::scan_wal(
                zero_filled_tail,
                ExpectedWalIdentity::from_trusted_header(header),
                WalScanMode::Frozen));
        },
        "frozen WAL accepted a zero-filled transaction tail");

    const auto frozen_batches = scan_and_capture(
        prefix, header, nullptr, WalScanMode::Frozen);
    require_equal(frozen_batches.size(), std::size_t{1},
                  "exact frozen WAL did not recover its committed batch");
}

void test_catalog_identity_is_required_and_callbacks_wait_for_full_validation() {
    const WalHeader header = make_header();
    const std::array<std::byte, 6> row{
        std::byte{0x11}, std::byte{0x12}, std::byte{0x13},
        std::byte{0x14}, std::byte{0x15}, std::byte{0x16}};
    const std::array<WalMutation, 1> operation{{
        WalMutation{99, WalOperationKind::Put, row},
    }};

    std::vector<std::byte> image = make_wal_image(header);
    append_transaction(image, header, header.first_sequence, operation);

    ExpectedWalIdentity wrong_identity =
        ExpectedWalIdentity::from_trusted_header(header);
    wrong_identity.wal_id.low ^= 1;
    require_throws<throwdb::CorruptionError>(
        [&image, &wrong_identity] {
            static_cast<void>(throwdb::detail::scan_wal(
                image, wrong_identity, WalScanMode::Active));
        },
        "WAL from a different catalog identity was accepted");

    const std::uint64_t bad_offset = append_transaction(
        image, header, header.first_sequence + 1, operation);
    image[static_cast<std::size_t>(
        bad_offset + throwdb::detail::kWalBatchHeaderSize)] =
        std::byte{0x7f};
    reseal_batch(image, bad_offset);

    std::size_t callbacks = 0;
    require_throws<throwdb::CorruptionError>(
        [&image, &header, &callbacks] {
            static_cast<void>(throwdb::detail::scan_wal(
                image, ExpectedWalIdentity::from_trusted_header(header),
                WalScanMode::Active,
                [&callbacks](const RecoveredWalBatch&) { ++callbacks; }));
        },
        "late committed corruption was accepted");
    require_equal(callbacks, std::size_t{0},
                  "visitor ran before complete WAL validation succeeded");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 11> tests{{
        {"crc32c_known_vector_and_incremental_updates",
         test_crc32c_known_vector_and_incremental_updates},
        {"crc32c_alignment_fragmentation_and_zero_updates",
         test_crc32c_alignment_fragmentation_and_zero_updates},
        {"wal_header_round_trip_and_corruption",
         test_wal_header_round_trip_and_corruption},
        {"mixed_transaction_has_golden_offsets_and_bit_exact_payloads",
         test_mixed_transaction_has_golden_offsets_and_bit_exact_payloads},
        {"every_truncation_of_third_transaction_is_ignored_as_tail",
         test_every_truncation_of_third_transaction_is_ignored_as_tail},
        {"interior_corruption_is_rejected_in_every_frame_region",
         test_interior_corruption_is_rejected_in_every_frame_region},
        {"forged_structural_errors_are_rejected_with_valid_checksums",
         test_forged_structural_errors_are_rejected_with_valid_checksums},
        {"encoder_rejects_invalid_batches_before_encoding",
         test_encoder_rejects_invalid_batches_before_encoding},
        {"tail_repair_allows_one_exact_next_append",
         test_tail_repair_allows_one_exact_next_append},
        {"active_full_length_torn_tail_is_repaired_but_frozen_rejects_it",
         test_active_full_length_torn_tail_is_repaired_but_frozen_rejects_it},
        {"catalog_identity_is_required_and_callbacks_wait_for_full_validation",
         test_catalog_identity_is_required_and_callbacks_wait_for_full_validation},
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
