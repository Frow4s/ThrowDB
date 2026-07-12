#pragma once

#include <filesystem>

#include "mutation_wal.hpp"

namespace throwdb::detail {

void create_wal_file(
    const std::filesystem::path& path,
    const WalHeader& header);

[[nodiscard]] WalScanResult scan_wal_file(
    int fd,
    const std::filesystem::path& path,
    const ExpectedWalIdentity& expected,
    WalScanMode mode,
    const WalRecoveryVisitor& visitor = {});

// Enforces the durable ordering required by the codec contract:
// BATCH -> sync -> COMMIT -> sync. Returns the first byte after COMMIT.
[[nodiscard]] std::uint64_t append_wal_transaction_durable(
    int fd,
    const std::filesystem::path& path,
    std::uint64_t batch_offset,
    const EncodedWalTransaction& transaction);

}  // namespace throwdb::detail
