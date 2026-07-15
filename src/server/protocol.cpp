#include "throwdb/server/protocol.hpp"

#include <algorithm>
#include <limits>

namespace throwdb::server::protocol {
namespace {

template <typename UInt>
void put_le(std::span<std::byte> output, const std::size_t offset, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    if (offset > output.size() || sizeof(UInt) > output.size() - offset) {
        throw ProtocolError("wire header write exceeds its fixed size");
    }
    for (std::size_t index = 0; index != sizeof(UInt); ++index) {
        output[offset + index] =
            static_cast<std::byte>((value >> (8U * index)) & 0xffU);
    }
}

template <typename UInt>
[[nodiscard]] UInt get_le(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    static_assert(std::is_unsigned_v<UInt>);
    if (offset > input.size() || sizeof(UInt) > input.size() - offset) {
        throw ProtocolError("truncated little-endian integer");
    }
    UInt result = 0;
    for (std::size_t index = 0; index != sizeof(UInt); ++index) {
        result |= static_cast<UInt>(std::to_integer<std::uint8_t>(
                      input[offset + index]))
                  << (8U * index);
    }
    return result;
}

template <typename UInt>
void append_le(std::vector<std::byte>& output, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    const auto old_size = output.size();
    if (old_size > output.max_size() - sizeof(UInt)) {
        throw ProtocolError("wire payload size overflow");
    }
    output.resize(old_size + sizeof(UInt));
    put_le<UInt>(output, old_size, value);
}

}  // namespace

std::array<std::byte, kHeaderBytes> encode_header(const FrameHeader& header) {
    if (header.header_bytes != kHeaderBytes) {
        throw ProtocolError("unsupported wire header size");
    }
    std::array<std::byte, kHeaderBytes> result{};
    std::copy(kMagic.begin(), kMagic.end(), result.begin());
    put_le<std::uint16_t>(result, 4, header.protocol_major);
    put_le<std::uint16_t>(result, 6, header.protocol_minor);
    put_le<std::uint16_t>(result, 8, header.header_bytes);
    put_le<std::uint16_t>(
        result, 10, static_cast<std::uint16_t>(header.opcode));
    put_le<std::uint32_t>(result, 12, header.flags);
    put_le<std::uint64_t>(result, 16, header.request_id);
    put_le<std::uint64_t>(result, 24, header.payload_bytes);
    return result;
}

FrameHeader decode_header(
    const std::span<const std::byte, kHeaderBytes> bytes) {
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw ProtocolError("invalid ThrowDB wire magic");
    }

    FrameHeader result;
    result.protocol_major = get_le<std::uint16_t>(bytes, 4);
    result.protocol_minor = get_le<std::uint16_t>(bytes, 6);
    result.header_bytes = get_le<std::uint16_t>(bytes, 8);
    result.opcode = static_cast<Opcode>(get_le<std::uint16_t>(bytes, 10));
    result.flags = get_le<std::uint32_t>(bytes, 12);
    result.request_id = get_le<std::uint64_t>(bytes, 16);
    result.payload_bytes = get_le<std::uint64_t>(bytes, 24);

    if (result.header_bytes != kHeaderBytes) {
        throw ProtocolError("unsupported ThrowDB wire header size");
    }
    return result;
}

PayloadReader::PayloadReader(const std::span<const std::byte> payload) noexcept
    : payload_(payload) {}

std::uint8_t PayloadReader::read_u8() {
    const auto bytes = read_bytes(sizeof(std::uint8_t));
    return get_le<std::uint8_t>(bytes, 0);
}

std::uint16_t PayloadReader::read_u16() {
    const auto bytes = read_bytes(sizeof(std::uint16_t));
    return get_le<std::uint16_t>(bytes, 0);
}

std::uint32_t PayloadReader::read_u32() {
    const auto bytes = read_bytes(sizeof(std::uint32_t));
    return get_le<std::uint32_t>(bytes, 0);
}

std::uint64_t PayloadReader::read_u64() {
    const auto bytes = read_bytes(sizeof(std::uint64_t));
    return get_le<std::uint64_t>(bytes, 0);
}

std::span<const std::byte> PayloadReader::read_bytes(const std::size_t count) {
    if (offset_ > payload_.size() || count > payload_.size() - offset_) {
        throw ProtocolError("truncated ThrowDB wire payload");
    }
    const auto result = payload_.subspan(offset_, count);
    offset_ += count;
    return result;
}

std::size_t PayloadReader::remaining() const noexcept {
    return payload_.size() - offset_;
}

void PayloadReader::require_end() const {
    if (remaining() != 0) {
        throw ProtocolError("unexpected trailing bytes in ThrowDB wire payload");
    }
}

PayloadWriter::PayloadWriter(const std::size_t reserve_bytes) {
    bytes_.reserve(reserve_bytes);
}

void PayloadWriter::write_u8(const std::uint8_t value) {
    append_le(bytes_, value);
}

void PayloadWriter::write_u16(const std::uint16_t value) {
    append_le(bytes_, value);
}

void PayloadWriter::write_u32(const std::uint32_t value) {
    append_le(bytes_, value);
}

void PayloadWriter::write_u64(const std::uint64_t value) {
    append_le(bytes_, value);
}

void PayloadWriter::write_bytes(const std::span<const std::byte> bytes) {
    if (bytes.size() > bytes_.max_size() - bytes_.size()) {
        throw ProtocolError("wire payload size overflow");
    }
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
}

void PayloadWriter::write_string(const std::string_view value) {
    write_bytes(std::as_bytes(std::span(value)));
}

std::vector<std::byte> PayloadWriter::take() && noexcept {
    return std::move(bytes_);
}

const std::vector<std::byte>& PayloadWriter::bytes() const noexcept {
    return bytes_;
}

void write_status(
    PayloadWriter& writer,
    const StatusCode status,
    const std::string_view message) {
    if (message.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw ProtocolError("wire status message is too large");
    }
    writer.write_u32(static_cast<std::uint32_t>(status));
    writer.write_u32(static_cast<std::uint32_t>(message.size()));
    writer.write_string(message);
}

StatusEnvelope read_status(PayloadReader& reader) {
    StatusEnvelope result;
    result.status = static_cast<StatusCode>(reader.read_u32());
    const auto message_size = reader.read_u32();
    const auto bytes = reader.read_bytes(message_size);
    result.message.assign(
        reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return result;
}

bool is_known_opcode(const Opcode opcode) noexcept {
    switch (opcode) {
        case Opcode::Health:
        case Opcode::Describe:
        case Opcode::Get:
        case Opcode::Insert:
        case Opcode::Update:
        case Opcode::Upsert:
        case Opcode::Delete:
        case Opcode::ScanIds:
            return true;
    }
    return false;
}

}  // namespace throwdb::server::protocol
