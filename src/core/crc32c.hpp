#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace throwdb::detail {

// Portable incremental CRC32C (Castagnoli). The state representation is kept
// private so callers cannot accidentally mix finalized and unfinalized CRCs.
class Crc32c {
public:
    void update(std::span<const std::byte> bytes) noexcept;
    void update_zeroes(std::size_t count) noexcept;

    [[nodiscard]] constexpr std::uint32_t value() const noexcept {
        return state_ ^ 0xffffffffU;
    }

private:
    std::uint32_t state_{0xffffffffU};
};

[[nodiscard]] std::uint32_t crc32c(
    std::span<const std::byte> bytes) noexcept;

}  // namespace throwdb::detail
