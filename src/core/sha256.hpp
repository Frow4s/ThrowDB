#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace throwdb::detail {

class Sha256 {
public:
    Sha256() noexcept;
    void update(std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] std::array<std::byte, 32> finish() noexcept;

private:
    void compress(const std::byte* block) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::array<std::byte, 64> buffer_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t buffered_ = 0;
    bool finished_ = false;
};

[[nodiscard]] std::array<std::byte, 32> sha256(
    std::span<const std::byte> bytes) noexcept;

}  // namespace throwdb::detail
