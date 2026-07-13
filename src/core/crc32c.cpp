#include "crc32c.hpp"

#include <array>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32) \
    && defined(__BYTE_ORDER__) \
    && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#include <arm_acle.h>
#define THROWDB_CRC32C_ARM64 1
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#define THROWDB_CRC32C_X86_64_RUNTIME 1
#endif

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

[[nodiscard, maybe_unused]] std::uint32_t portable_update(
    std::uint32_t state,
    const std::span<const std::byte> bytes) noexcept {
    for (const std::byte byte : bytes) {
        const auto index = static_cast<std::uint8_t>(state)
                           ^ std::to_integer<std::uint8_t>(byte);
        state = kTable[index] ^ (state >> 8);
    }
    return state;
}

[[nodiscard, maybe_unused]] std::uint32_t portable_update_zeroes(
    std::uint32_t state,
    std::size_t count) noexcept {
    while (count-- != 0) {
        const auto index = static_cast<std::uint8_t>(state);
        state = kTable[index] ^ (state >> 8);
    }
    return state;
}

#if defined(THROWDB_CRC32C_ARM64)

[[nodiscard]] std::uint32_t arm64_update(
    std::uint32_t state,
    std::span<const std::byte> bytes) noexcept {
    while (bytes.size() >= sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data(), sizeof(word));
        state = __crc32cd(state, word);
        bytes = bytes.subspan(sizeof(word));
    }
    for (const std::byte byte : bytes) {
        state = __crc32cb(state, std::to_integer<std::uint8_t>(byte));
    }
    return state;
}

[[nodiscard]] std::uint32_t arm64_update_zeroes(
    std::uint32_t state,
    std::size_t count) noexcept {
    while (count >= sizeof(std::uint64_t)) {
        state = __crc32cd(state, 0);
        count -= sizeof(std::uint64_t);
    }
    while (count-- != 0) {
        state = __crc32cb(state, 0);
    }
    return state;
}

#endif

#if defined(THROWDB_CRC32C_X86_64_RUNTIME)

#define THROWDB_CRC32C_SSE42_TARGET __attribute__((target("sse4.2")))

[[nodiscard]] bool has_sse42() noexcept {
    static const bool supported = [] {
#if defined(__GNUC__) && !defined(__clang__)
        __builtin_cpu_init();
#endif
        return __builtin_cpu_supports("sse4.2");
    }();
    return supported;
}

[[nodiscard]] THROWDB_CRC32C_SSE42_TARGET std::uint32_t sse42_update(
    std::uint32_t state,
    std::span<const std::byte> bytes) noexcept {
    std::uint64_t wide_state = state;
    while (bytes.size() >= sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data(), sizeof(word));
        wide_state = _mm_crc32_u64(wide_state, word);
        bytes = bytes.subspan(sizeof(word));
    }
    state = static_cast<std::uint32_t>(wide_state);
    for (const std::byte byte : bytes) {
        state = _mm_crc32_u8(state, std::to_integer<std::uint8_t>(byte));
    }
    return state;
}

[[nodiscard]] THROWDB_CRC32C_SSE42_TARGET std::uint32_t
sse42_update_zeroes(std::uint32_t state, std::size_t count) noexcept {
    std::uint64_t wide_state = state;
    while (count >= sizeof(std::uint64_t)) {
        wide_state = _mm_crc32_u64(wide_state, 0);
        count -= sizeof(std::uint64_t);
    }
    state = static_cast<std::uint32_t>(wide_state);
    while (count-- != 0) {
        state = _mm_crc32_u8(state, 0);
    }
    return state;
}

#undef THROWDB_CRC32C_SSE42_TARGET

#endif

}  // namespace

void Crc32c::update(const std::span<const std::byte> bytes) noexcept {
#if defined(THROWDB_CRC32C_ARM64)
    state_ = arm64_update(state_, bytes);
#elif defined(THROWDB_CRC32C_X86_64_RUNTIME)
    state_ = has_sse42() ? sse42_update(state_, bytes)
                         : portable_update(state_, bytes);
#else
    state_ = portable_update(state_, bytes);
#endif
}

void Crc32c::update_zeroes(std::size_t count) noexcept {
#if defined(THROWDB_CRC32C_ARM64)
    state_ = arm64_update_zeroes(state_, count);
#elif defined(THROWDB_CRC32C_X86_64_RUNTIME)
    state_ = has_sse42() ? sse42_update_zeroes(state_, count)
                         : portable_update_zeroes(state_, count);
#else
    state_ = portable_update_zeroes(state_, count);
#endif
}

std::uint32_t crc32c(const std::span<const std::byte> bytes) noexcept {
    Crc32c checksum;
    checksum.update(bytes);
    return checksum.value();
}

}  // namespace throwdb::detail
