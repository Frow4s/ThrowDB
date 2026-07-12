#include "crc32c.hpp"

#include <array>

namespace throwdb::detail {
namespace {

constexpr std::uint32_t kReflectedCastagnoliPolynomial = 0x82f63b78U;

[[nodiscard]] constexpr std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t byte = 0; byte != table.size(); ++byte) {
        std::uint32_t crc = byte;
        for (unsigned bit = 0; bit != 8; ++bit) {
            crc = (crc >> 1)
                  ^ ((crc & 1U) == 0
                         ? 0U
                         : kReflectedCastagnoliPolynomial);
        }
        table[byte] = crc;
    }
    return table;
}

constexpr auto kTable = make_table();

}  // namespace

void Crc32c::update(const std::span<const std::byte> bytes) noexcept {
    for (const std::byte byte : bytes) {
        const auto index = static_cast<std::uint8_t>(state_)
                           ^ std::to_integer<std::uint8_t>(byte);
        state_ = kTable[index] ^ (state_ >> 8);
    }
}

void Crc32c::update_zeroes(std::size_t count) noexcept {
    while (count-- != 0) {
        const auto index = static_cast<std::uint8_t>(state_);
        state_ = kTable[index] ^ (state_ >> 8);
    }
}

std::uint32_t crc32c(const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes);
    return checksum.value();
}

}  // namespace throwdb::detail
