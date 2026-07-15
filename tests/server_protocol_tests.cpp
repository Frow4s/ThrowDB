#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "throwdb/server/protocol.hpp"

namespace {

using throwdb::server::protocol::FrameFlag;
using throwdb::server::protocol::FrameHeader;
using throwdb::server::protocol::Opcode;
using throwdb::server::protocol::PayloadReader;
using throwdb::server::protocol::PayloadWriter;
using throwdb::server::protocol::ProtocolError;
using throwdb::server::protocol::StatusCode;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_protocol_error(Callable&& callable, const std::string& message) {
    try {
        callable();
    } catch (const ProtocolError&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_header_golden_bytes() {
    FrameHeader header;
    header.opcode = Opcode::ScanIds;
    header.flags = FrameFlag::Response | FrameFlag::Stream;
    header.request_id = 0x0102030405060708ULL;
    header.payload_bytes = 0x1112131415161718ULL;
    const auto encoded = throwdb::server::protocol::encode_header(header);

    const std::array<std::uint8_t, 32> expected{
        'T', 'D', 'B', 'P',
        1, 0, 0, 0,
        32, 0, 20, 0,
        3, 0, 0, 0,
        8, 7, 6, 5, 4, 3, 2, 1,
        0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11};
    for (std::size_t index = 0; index != encoded.size(); ++index) {
        require(
            std::to_integer<std::uint8_t>(encoded[index]) == expected[index],
            "encoded header differs from protocol-v1 golden bytes");
    }

    const auto decoded = throwdb::server::protocol::decode_header(encoded);
    require(decoded.protocol_major == 1, "decoded major version mismatch");
    require(decoded.protocol_minor == 0, "decoded minor version mismatch");
    require(decoded.header_bytes == 32, "decoded header size mismatch");
    require(decoded.opcode == Opcode::ScanIds, "decoded opcode mismatch");
    require(decoded.flags == 3, "decoded flags mismatch");
    require(decoded.request_id == header.request_id, "decoded request ID mismatch");
    require(
        decoded.payload_bytes == header.payload_bytes,
        "decoded payload size mismatch");
}

void test_header_rejects_magic_and_size() {
    FrameHeader header;
    auto bytes = throwdb::server::protocol::encode_header(header);
    bytes[0] = std::byte{'X'};
    require_protocol_error(
        [&] { static_cast<void>(throwdb::server::protocol::decode_header(bytes)); },
        "decoder accepted invalid magic");

    bytes = throwdb::server::protocol::encode_header(header);
    bytes[8] = std::byte{31};
    require_protocol_error(
        [&] { static_cast<void>(throwdb::server::protocol::decode_header(bytes)); },
        "decoder accepted invalid header size");
}

void test_payload_roundtrip_and_bounds() {
    PayloadWriter writer;
    writer.write_u8(0xa5);
    writer.write_u16(0xb6c7);
    writer.write_u32(0xd8e9fa01);
    writer.write_u64(0x1020304050607080ULL);
    writer.write_string("wire");
    auto payload = std::move(writer).take();

    PayloadReader reader(payload);
    require(reader.read_u8() == 0xa5, "u8 roundtrip mismatch");
    require(reader.read_u16() == 0xb6c7, "u16 roundtrip mismatch");
    require(reader.read_u32() == 0xd8e9fa01, "u32 roundtrip mismatch");
    require(
        reader.read_u64() == 0x1020304050607080ULL,
        "u64 roundtrip mismatch");
    const auto tail = reader.read_bytes(4);
    require(
        std::string(reinterpret_cast<const char*>(tail.data()), tail.size())
            == "wire",
        "byte payload roundtrip mismatch");
    reader.require_end();
    require_protocol_error(
        [&] { static_cast<void>(reader.read_u8()); },
        "payload reader accepted a truncated integer");

    PayloadReader trailing(payload);
    static_cast<void>(trailing.read_u8());
    require_protocol_error(
        [&] { trailing.require_end(); },
        "payload reader accepted trailing bytes");
}

void test_status_envelope_roundtrip() {
    PayloadWriter writer;
    throwdb::server::protocol::write_status(
        writer, StatusCode::InvalidArgument, "bad batch");
    writer.write_u64(42);
    auto payload = std::move(writer).take();

    PayloadReader reader(payload);
    const auto status = throwdb::server::protocol::read_status(reader);
    require(
        status.status == StatusCode::InvalidArgument,
        "status code roundtrip mismatch");
    require(status.message == "bad batch", "status message roundtrip mismatch");
    require(reader.read_u64() == 42, "status body boundary mismatch");
    reader.require_end();
}

void run_all() {
    test_header_golden_bytes();
    test_header_rejects_magic_and_size();
    test_payload_roundtrip_and_bounds();
    test_status_envelope_roundtrip();
}

}  // namespace

int main() {
    try {
        run_all();
        std::cout << "server protocol tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "server protocol test failure: " << error.what() << '\n';
        return 1;
    }
}
