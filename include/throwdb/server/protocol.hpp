#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace throwdb::server::protocol {

// ThrowDB wire protocol v1 is deliberately small and dependency-free. Every
// integer is little-endian and every message starts with this fixed-size
// header. TCP is only a byte transport: payload boundaries come exclusively
// from payload_bytes.
inline constexpr std::array<std::byte, 4> kMagic{
    std::byte{'T'}, std::byte{'D'}, std::byte{'B'}, std::byte{'P'}};
inline constexpr std::uint16_t kProtocolMajor = 1;
inline constexpr std::uint16_t kProtocolMinor = 0;
inline constexpr std::uint16_t kHeaderBytes = 32;

enum class Opcode : std::uint16_t {
    Health = 1,
    Describe = 2,
    Get = 3,
    Insert = 10,
    Update = 11,
    Upsert = 12,
    Delete = 13,
    ScanIds = 20,
};

enum FrameFlag : std::uint32_t {
    Response = 1U << 0,
    Stream = 1U << 1,
    EndStream = 1U << 2,
    Error = 1U << 3,
};

inline constexpr std::uint32_t kKnownRequestFlags = 0;
inline constexpr std::uint32_t kKnownResponseFlags =
    FrameFlag::Response | FrameFlag::Stream | FrameFlag::EndStream
    | FrameFlag::Error;

enum class StatusCode : std::uint32_t {
    Ok = 0,
    InvalidArgument = 1,
    NotFound = 2,
    AlreadyExists = 3,
    ResourceExhausted = 4,
    FailedPrecondition = 5,
    DataLoss = 6,
    Internal = 7,
    Unimplemented = 8,
    Unavailable = 9,
    Cancelled = 10,
    DeadlineExceeded = 11,
    ProtocolError = 12,
};

enum Capability : std::uint32_t {
    Writable = 1U << 0,
    PrefixGet = 1U << 1,
    ScanLocations = 1U << 2,
    DurableIdempotency = 1U << 3,
};

enum ScanFlag : std::uint32_t {
    IncludeLocations = 1U << 0,
};

enum class LocationFormat : std::uint32_t {
    None = 0,
    SegmentRowU32 = 1,
};

struct FrameHeader {
    std::uint16_t protocol_major = kProtocolMajor;
    std::uint16_t protocol_minor = kProtocolMinor;
    std::uint16_t header_bytes = kHeaderBytes;
    Opcode opcode = Opcode::Health;
    std::uint32_t flags = 0;
    std::uint64_t request_id = 0;
    std::uint64_t payload_bytes = 0;
};

struct Frame {
    FrameHeader header;
    std::vector<std::byte> payload;
};

class ProtocolError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] std::array<std::byte, kHeaderBytes> encode_header(
    const FrameHeader& header);
[[nodiscard]] FrameHeader decode_header(
    std::span<const std::byte, kHeaderBytes> bytes);

// Small checked LE codec used by both the transport and the service. Reader
// never returns a view beyond the payload and Writer rejects size_t overflow.
class PayloadReader {
public:
    explicit PayloadReader(std::span<const std::byte> payload) noexcept;

    [[nodiscard]] std::uint8_t read_u8();
    [[nodiscard]] std::uint16_t read_u16();
    [[nodiscard]] std::uint32_t read_u32();
    [[nodiscard]] std::uint64_t read_u64();
    [[nodiscard]] std::span<const std::byte> read_bytes(std::size_t count);
    [[nodiscard]] std::size_t remaining() const noexcept;
    void require_end() const;

private:
    std::span<const std::byte> payload_;
    std::size_t offset_ = 0;
};

class PayloadWriter {
public:
    explicit PayloadWriter(std::size_t reserve_bytes = 0);

    void write_u8(std::uint8_t value);
    void write_u16(std::uint16_t value);
    void write_u32(std::uint32_t value);
    void write_u64(std::uint64_t value);
    void write_bytes(std::span<const std::byte> bytes);
    void write_string(std::string_view value);

    [[nodiscard]] std::vector<std::byte> take() && noexcept;
    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept;

private:
    std::vector<std::byte> bytes_;
};

void write_status(
    PayloadWriter& writer,
    StatusCode status,
    std::string_view message = {});

struct StatusEnvelope {
    StatusCode status = StatusCode::Ok;
    std::string message;
};

[[nodiscard]] StatusEnvelope read_status(PayloadReader& reader);
[[nodiscard]] bool is_known_opcode(Opcode opcode) noexcept;

}  // namespace throwdb::server::protocol
