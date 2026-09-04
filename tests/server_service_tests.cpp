#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/native_collection.hpp"
#include "throwdb/server/protocol.hpp"
#include "throwdb/server/collection_engine.hpp"
#include "throwdb/server/single_collection_service.hpp"

namespace {

namespace wire = throwdb::server::protocol;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-server-service-test-"
                   + std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count())
                   + "-" + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> output(count * record_size);
    for (std::size_t row = 0; row != count; ++row) {
        for (std::size_t column = 0; column != record_size; ++column) {
            output[row * record_size + column] = static_cast<std::byte>(
                salt + static_cast<std::uint8_t>(31 * row + 7 * column));
        }
    }
    return output;
}

[[nodiscard]] wire::Frame request(
    const wire::Opcode opcode,
    std::vector<std::byte> payload = {},
    const std::uint64_t request_id = 1) {
    wire::Frame result;
    result.header.opcode = opcode;
    result.header.request_id = request_id;
    result.header.payload_bytes = payload.size();
    result.payload = std::move(payload);
    return result;
}

[[nodiscard]] std::vector<wire::Frame> call(
    throwdb::server::SingleCollectionService& service,
    const wire::Frame& input) {
    std::vector<wire::Frame> output;
    service.handle(input, [&](wire::Frame&& frame) {
        output.push_back(std::move(frame));
        return true;
    });
    require(!output.empty(), "service emitted no response");
    return output;
}

void require_status(
    wire::PayloadReader& reader,
    const wire::StatusCode expected,
    const std::string& context) {
    const auto status = wire::read_status(reader);
    require(
        status.status == expected,
        context + ": unexpected status code "
            + std::to_string(static_cast<std::uint32_t>(status.status))
            + " (expected "
            + std::to_string(static_cast<std::uint32_t>(expected))
            + "): " + status.message);
    if (expected == wire::StatusCode::Ok) {
        require(status.message.empty(), context + ": OK has an error message");
    } else {
        require(!status.message.empty(), context + ": error has no message");
    }
}

[[nodiscard]] std::vector<std::byte> encode_get(
    const std::span<const std::uint64_t> ids,
    const std::uint32_t prefix = 0) {
    wire::PayloadWriter writer;
    writer.write_u32(prefix);
    writer.write_u32(0);
    writer.write_u64(ids.size());
    for (const auto id : ids) {
        writer.write_u64(id);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::vector<std::byte> encode_put(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension,
    const std::uint64_t operation_low) {
    wire::PayloadWriter writer;
    writer.write_u64(0x0102030405060708ULL);
    writer.write_u64(operation_low);
    writer.write_u32(dimension);
    writer.write_u32(static_cast<std::uint32_t>(throwdb::DType::Float16));
    writer.write_u64(ids.size());
    for (const auto id : ids) {
        writer.write_u64(id);
    }
    writer.write_bytes(embeddings);
    return std::move(writer).take();
}

[[nodiscard]] std::vector<std::byte> encode_delete(
    const std::span<const std::uint64_t> ids) {
    wire::PayloadWriter writer;
    writer.write_u64(0x2122232425262728ULL);
    writer.write_u64(0x3132333435363738ULL);
    writer.write_u64(ids.size());
    for (const auto id : ids) {
        writer.write_u64(id);
    }
    return std::move(writer).take();
}

void test_service_crud_get_and_scan() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 4> base_ids{10, 20, 30, 40};
    const auto base_rows = rows(base_ids.size(), record_size, 3);

    throwdb::NativeBuilderOptions options;
    options.format_version = 7;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 2 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(base_ids, base_rows);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    throwdb::server::SingleCollectionEngine engine(*collection, store);
    throwdb::server::SingleCollectionService service(engine);

    auto health = call(service, request(wire::Opcode::Health));
    require(health.size() == 1, "Health is not unary");
    wire::PayloadReader health_reader(health.front().payload);
    require_status(health_reader, wire::StatusCode::Ok, "Health");
    require(health_reader.read_u32() == 1, "Health is not serving");

    auto describe = call(service, request(wire::Opcode::Describe));
    wire::PayloadReader describe_reader(describe.front().payload);
    require_status(describe_reader, wire::StatusCode::Ok, "Describe");
    require(describe_reader.read_u32() == 7, "Describe format mismatch");
    require(describe_reader.read_u32() == dimension, "Describe dimension mismatch");
    require(
        describe_reader.read_u32()
            == static_cast<std::uint32_t>(throwdb::DType::Float16),
        "Describe dtype mismatch");
    const auto capabilities = describe_reader.read_u32();
    require(
        (capabilities & wire::Capability::Writable) != 0,
        "Describe omitted Writable");
    require(
        (capabilities & wire::Capability::ScanLocations) != 0,
        "Describe omitted ScanLocations");
    require(
        (capabilities & wire::Capability::DurableIdempotency) != 0,
        "Describe omitted DurableIdempotency");
    require(describe_reader.read_u64() == record_size, "Describe record size mismatch");
    require(describe_reader.read_u64() == base_ids.size(), "Describe count mismatch");
    describe_reader.require_end();

    const std::array<std::uint64_t, 2> get_ids{40, 10};
    auto get = call(service, request(
        wire::Opcode::Get, encode_get(get_ids, 2), 2));
    wire::PayloadReader get_reader(get.front().payload);
    require_status(get_reader, wire::StatusCode::Ok, "Get");
    require(get_reader.read_u32() == 2, "Get prefix mismatch");
    require(
        get_reader.read_u32()
            == static_cast<std::uint32_t>(throwdb::DType::Float16),
        "Get dtype mismatch");
    require(get_reader.read_u64() == 2, "Get count mismatch");
    const auto get_bytes = get_reader.read_bytes(2 * 2 * 2);
    require(
        std::equal(
            get_bytes.begin(),
            get_bytes.begin() + 4,
            base_rows.begin() + 3 * record_size),
        "Get first prefix is not bit exact");
    require(
        std::equal(
            get_bytes.begin() + 4,
            get_bytes.end(),
            base_rows.begin()),
        "Get second prefix is not bit exact");
    const throwdb::server::ProfiledGetResult profiled =
        engine.get_profiled(get_ids, 2);
    require(
        profiled.result.dimension == 2
            && profiled.result.row_count == get_ids.size()
            && std::equal(
                profiled.result.embeddings.begin(),
                profiled.result.embeddings.end(),
                get_bytes.begin(),
                get_bytes.end()),
        "profiled engine Get changed the projected response");
    require(
        profiled.read.requested_rows == get_ids.size()
            && profiled.read.base_rows == get_ids.size()
            && profiled.read.planned_ranges > 0
            && profiled.read.completed_read_bytes > 0,
        "profiled engine Get omitted physical read diagnostics");

    const std::array<std::uint64_t, 2> insert_ids{50, 60};
    const auto inserted_rows = rows(2, record_size, 71);
    auto inserted = call(service, request(
        wire::Opcode::Insert,
        encode_put(insert_ids, inserted_rows, dimension, 3),
        3));
    wire::PayloadReader inserted_reader(inserted.front().payload);
    require_status(inserted_reader, wire::StatusCode::Ok, "Insert");
    require(inserted_reader.read_u64() > 0, "Insert returned zero commit sequence");
    require(inserted_reader.read_u64() == 2, "Insert returned wrong row count");
    inserted_reader.require_end();

    const std::array<std::uint64_t, 1> update_ids{20};
    const auto updated_row = rows(1, record_size, 111);
    auto updated = call(service, request(
        wire::Opcode::Update,
        encode_put(update_ids, updated_row, dimension, 4),
        4));
    wire::PayloadReader updated_reader(updated.front().payload);
    require_status(updated_reader, wire::StatusCode::Ok, "Update");

    const std::array<std::uint64_t, 2> delete_ids{30, 999};
    auto deleted = call(service, request(
        wire::Opcode::Delete, encode_delete(delete_ids), 5));
    wire::PayloadReader deleted_reader(deleted.front().payload);
    require_status(deleted_reader, wire::StatusCode::Ok, "Delete");
    static_cast<void>(deleted_reader.read_u64());
    require(deleted_reader.read_u64() == 2, "Delete count mismatch");
    require(deleted_reader.read_u8() == 1, "Delete existing mask mismatch");
    require(deleted_reader.read_u8() == 0, "Delete missing mask mismatch");
    deleted_reader.require_end();

    const std::array<std::uint64_t, 2> conflicting_ids{70, 10};
    const auto conflicting_rows = rows(2, record_size, 151);
    auto conflicting = call(service, request(
        wire::Opcode::Insert,
        encode_put(conflicting_ids, conflicting_rows, dimension, 6),
        6));
    wire::PayloadReader conflict_reader(conflicting.front().payload);
    require_status(conflict_reader, wire::StatusCode::AlreadyExists, "Insert conflict");
    auto missing = call(service, request(
        wire::Opcode::Get,
        encode_get(std::span(conflicting_ids).first(1)),
        7));
    wire::PayloadReader missing_reader(missing.front().payload);
    require_status(missing_reader, wire::StatusCode::NotFound, "atomic Insert");

    wire::PayloadWriter scan_request;
    scan_request.write_u32(2);
    scan_request.write_u32(wire::ScanFlag::IncludeLocations);
    scan_request.write_u64(0);
    auto scan = call(service, request(
        wire::Opcode::ScanIds, std::move(scan_request).take(), 8));
    require(
        (scan.back().header.flags & wire::FrameFlag::EndStream) != 0,
        "ScanIds has no final frame");

    std::uint64_t expected_offset = 0;
    std::uint64_t snapshot_count = 0;
    std::uint64_t token_high = 0;
    std::uint64_t token_low = 0;
    std::set<std::uint64_t> all_ids;
    std::pair<std::uint32_t, std::uint32_t> previous_location{0, 0};
    bool first_location = true;
    for (const auto& frame : scan) {
        require(
            (frame.header.flags & wire::FrameFlag::Stream) != 0,
            "ScanIds frame is not marked as a stream");
        wire::PayloadReader reader(frame.payload);
        require_status(reader, wire::StatusCode::Ok, "ScanIds");
        const auto collection_low = reader.read_u64();
        const auto collection_high = reader.read_u64();
        require(collection_low != 0 || collection_high != 0, "ScanIds has empty collection ID");
        static_cast<void>(reader.read_u64());  // catalog generation
        static_cast<void>(reader.read_u64());  // committed sequence
        const auto frame_snapshot_count = reader.read_u64();
        const auto frame_token_high = reader.read_u64();
        const auto frame_token_low = reader.read_u64();
        const auto offset = reader.read_u64();
        const auto count = reader.read_u64();
        require(
            reader.read_u32()
                == static_cast<std::uint32_t>(wire::LocationFormat::SegmentRowU32),
            "ScanIds location format mismatch");
        require(reader.read_u32() == 0, "ScanIds reserved field mismatch");
        if (expected_offset == 0) {
            snapshot_count = frame_snapshot_count;
            token_high = frame_token_high;
            token_low = frame_token_low;
        }
        require(frame_snapshot_count == snapshot_count, "ScanIds count changed between frames");
        require(
            frame_token_high == token_high && frame_token_low == token_low,
            "ScanIds token changed between frames");
        require(offset == expected_offset, "ScanIds offset is not contiguous");
        std::vector<std::uint64_t> chunk_ids;
        chunk_ids.reserve(count);
        for (std::uint64_t index = 0; index != count; ++index) {
            chunk_ids.push_back(reader.read_u64());
        }
        std::uint32_t chunk_segment = 0;
        for (std::uint64_t index = 0; index != count; ++index) {
            const auto segment_id = reader.read_u32();
            const auto row_index = reader.read_u32();
            const auto location = std::pair(segment_id, row_index);
            if (index == 0) {
                chunk_segment = location.first;
            }
            require(location.first == chunk_segment, "ScanIds chunk crossed a segment");
            if (!first_location) {
                require(previous_location < location, "ScanIds locations are not increasing");
            }
            first_location = false;
            previous_location = location;
            require(all_ids.insert(chunk_ids[index]).second, "ScanIds duplicated a live ID");
        }
        reader.require_end();
        expected_offset += count;
    }
    require(snapshot_count == 5, "ScanIds snapshot count mismatch");
    require(expected_offset == snapshot_count, "ScanIds emitted count mismatch");
    require(
        all_ids == std::set<std::uint64_t>({10, 20, 40, 50, 60}),
        "ScanIds returned the wrong logical ID set");
}

}  // namespace

int main() {
    try {
        test_service_crud_get_and_scan();
        std::cout << "server service tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "server service test failure: " << error.what() << '\n';
        return 1;
    }
}
