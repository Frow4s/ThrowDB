#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#include "throwdb/errors.hpp"

namespace throwdb::detail {

inline constexpr std::string_view kPendingMutationFenceName =
    "pending-mutation-v1";

inline void ensure_pending_mutation_fence_absent(
    const std::filesystem::path& root,
    const std::string_view action) {
    const auto fence = root / "server" / kPendingMutationFenceName;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(fence, error);
    if (error) {
        if (error
            == std::make_error_code(std::errc::no_such_file_or_directory)) {
            return;
        }
        throw StorageError(
            "cannot inspect pending-mutation fence before "
            + std::string(action) + ": " + error.message());
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        return;
    }
    throw PendingMutationFenceError(
        std::string(action)
        + " is blocked by a durable idempotency PREPARE; reopen the durable "
          "writable server to resolve the pending operation first");
}

}  // namespace throwdb::detail
