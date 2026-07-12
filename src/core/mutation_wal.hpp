#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "internal.hpp"
#include "mutation_types.hpp"

namespace throwdb::detail {

inline constexpr std::uint64_t kWalFileHeaderSize = 4096;
inline constexpr std::uint64_t kWalBatchHeaderSize = 128;
inline constexpr std::uint64_t kWalCommitMarkerSize = 64;
inline constexpr std::uint64_t kWalOperationEntrySize = 16;
inline constexpr std::uint64_t kWalFormatMaxOperations = 1'000'000;
inline constexpr std::uint64_t kWalFormatMaxFrameSize = 1ULL << 30;

enum class WalOperationKind : std::uint8_t {
    Put = 1,
    Delete = 2,
};

struct WalHeader {
    Uuid128 collection_id;
    Uuid128 schema_id;
    Uuid128 wal_id;
    Uuid128 predecessor_wal_id;
    std::uint64_t previous_sequence{0};
    std::uint64_t first_sequence{1};
    std::uint64_t record_size{0};
    std::uint32_t dimension{0};
    DTypeCode dtype{DTypeCode::Invalid};
    std::uint32_t item_size{0};
    IdType id_type{IdType::Invalid};
    RowLayout row_layout{RowLayout::Invalid};
    std::uint64_t required_feature_bits{0};
    std::uint64_t optional_feature_bits{0};
    std::uint64_t creation_time_unix_ns{0};
};

struct WalLimits {
    // A writer may choose stricter per-request limits, but never values above
    // the immutable format maxima. Recovery always accepts up to the format
    // maxima, so changing runtime tuning cannot invalidate a durable WAL.
    std::uint64_t max_operations{kWalFormatMaxOperations};
    std::uint64_t max_frame_size{kWalFormatMaxFrameSize};
};

enum class WalScanMode : std::uint8_t {
    // The final uncommitted/torn append may be discarded. A valid later COMMIT
    // is still evidence of interior corruption and prevents tail recovery.
    Active = 1,
    // Every byte after the file header must belong to a committed transaction.
    Frozen = 2,
};

struct ExpectedWalIdentity {
    Uuid128 collection_id;
    Uuid128 schema_id;
    Uuid128 wal_id;
    Uuid128 predecessor_wal_id;
    std::uint64_t previous_sequence{0};
    std::uint64_t first_sequence{1};
    std::uint64_t record_size{0};
    std::uint32_t dimension{0};
    DTypeCode dtype{DTypeCode::Invalid};
    std::uint32_t item_size{0};
    IdType id_type{IdType::Invalid};
    RowLayout row_layout{RowLayout::Invalid};

    // Convenience for a header already trusted by a writer/test. Recovery
    // must populate this value independently from the active catalog.
    [[nodiscard]] static ExpectedWalIdentity from_trusted_header(
        const WalHeader& header) noexcept;
};

struct WalMutation {
    std::uint64_t id{0};
    WalOperationKind kind{WalOperationKind::Put};
    std::span<const std::byte> value;
};

struct EncodedWalTransaction {
    std::vector<std::byte> bytes;
    std::uint64_t batch_size{0};
    std::uint32_t batch_crc32c{0};
};

struct RecoveredWalOperation {
    std::uint64_t id{0};
    WalOperationKind kind{WalOperationKind::Put};
    // Absolute byte offset in the WAL image. Deletes have offset and size 0.
    std::uint64_t payload_offset{0};
    std::uint64_t payload_size{0};
};

struct RecoveredWalBatch {
    std::uint64_t sequence{0};
    std::uint64_t batch_offset{0};
    std::uint64_t batch_size{0};
    std::span<const RecoveredWalOperation> operations;
};

using WalRecoveryVisitor = std::function<void(const RecoveredWalBatch&)>;

struct WalScanResult {
    WalHeader header;
    // Safe ftruncate target before the next append. It always includes the
    // 4096-byte file header and only complete BATCH+COMMIT pairs after it.
    std::uint64_t valid_end{kWalFileHeaderSize};
    std::uint64_t next_sequence{1};
    std::uint64_t ignored_tail_bytes{0};
    std::uint64_t committed_batch_count{0};
};

[[nodiscard]] std::array<std::byte, kWalFileHeaderSize> encode_wal_header(
    const WalHeader& header);
[[nodiscard]] WalHeader decode_wal_header(
    std::span<const std::byte> bytes);

// batch_offset is the absolute append position in its WAL file. The returned
// bytes contain one BATCH immediately followed by its COMMIT marker. A durable
// appender must write+sync the first batch_size bytes before it writes+syncs the
// marker; one combined pre-sync write does not establish the required ordering.
[[nodiscard]] EncodedWalTransaction encode_wal_transaction(
    const WalHeader& header,
    std::uint64_t sequence,
    std::uint64_t batch_offset,
    std::span<const WalMutation> operations,
    const WalLimits& limits = {});

// Scans a stable in-memory WAL image bound to catalog-provided identity. The
// complete image is validated before the first visitor call. The operation span
// is valid only for the duration of that call; copy references that must live
// longer. A rebuilt overlay must be published only after this function returns.
[[nodiscard]] WalScanResult scan_wal(
    std::span<const std::byte> image,
    const ExpectedWalIdentity& expected,
    WalScanMode mode,
    const WalRecoveryVisitor& visitor = {});

}  // namespace throwdb::detail
