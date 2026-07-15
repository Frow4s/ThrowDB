#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <grpc/impl/channel_arg_names.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/generic/generic_stub.h>

#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "throwdb/server/collection_engine.hpp"
#include "throwdb/server/grpc_server.hpp"
#include "throwdb/v1/throwdb.grpc.pb.h"

namespace {

namespace wire = ::throwdb::v1;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-grpc-test-"
                   + std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count())
                   + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ~TemporaryDirectory() {
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

void append_u64_le(std::string& output, const std::uint64_t value) {
    for (unsigned int index = 0; index != 8; ++index) {
        output.push_back(static_cast<char>(value >> (index * 8U)));
    }
}

[[nodiscard]] std::uint64_t read_u64_le(const char* input) {
    std::uint64_t value = 0;
    for (unsigned int index = 0; index != 8; ++index) {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(input[index]))
                 << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::string flatten(const grpc::ByteBuffer& input) {
    std::vector<grpc::Slice> slices;
    const grpc::Status status = input.Dump(&slices);
    require(status.ok(), "could not flatten a generic gRPC response");
    std::string result;
    result.reserve(input.Length());
    for (const grpc::Slice& slice : slices) {
        result.append(
            reinterpret_cast<const char*>(slice.begin()), slice.size());
    }
    return result;
}

[[nodiscard]] std::pair<grpc::Status, std::string> generic_unary(
    const std::shared_ptr<grpc::Channel>& channel,
    const std::string& method,
    const std::string& payload) {
    grpc::GenericStub stub(channel);
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(5));
    const grpc::Slice request_slice(payload.data(), payload.size());
    const grpc::ByteBuffer request(&request_slice, 1);
    grpc::CompletionQueue queue;
    auto call = stub.PrepareUnaryCall(&context, method, request, &queue);
    require(call != nullptr, "could not prepare a generic unary call");
    grpc::ByteBuffer response;
    grpc::Status status;
    int finish_tag = 0;
    call->StartCall();
    call->Finish(&response, &status, &finish_tag);
    void* observed_tag = nullptr;
    bool ok = false;
    require(
        queue.Next(&observed_tag, &ok) && ok
            && observed_tag == &finish_tag,
        "generic unary gRPC call did not finish cleanly");
    queue.Shutdown();
    return {status, response.Valid() ? flatten(response) : std::string{}};
}

#if defined(THROWDB_TEST_GRPC_REFLECTION)
[[nodiscard]] std::pair<grpc::Status, std::string> reflection_services(
    const std::shared_ptr<grpc::Channel>& channel) {
    grpc::GenericStub stub(channel);
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::CompletionQueue queue;
    auto call = stub.PrepareCall(
        &context,
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
        &queue);
    require(call != nullptr, "could not prepare the reflection call");

    const auto await = [&queue](void* expected, const bool expected_ok) {
        void* observed = nullptr;
        bool ok = false;
        require(
            queue.Next(&observed, &ok) && observed == expected
                && ok == expected_ok,
            "gRPC reflection stream produced an unexpected event");
    };
    int start_tag = 0;
    int write_tag = 0;
    int writes_done_tag = 0;
    int read_tag = 0;
    int eof_tag = 0;
    int finish_tag = 0;
    call->StartCall(&start_tag);
    await(&start_tag, true);

    // grpc.reflection.v1.ServerReflectionRequest.list_services = "".
    const std::array<char, 2> list_services_request{
        static_cast<char>(0x3a), static_cast<char>(0x00)};
    const grpc::Slice request_slice(
        list_services_request.data(), list_services_request.size());
    const grpc::ByteBuffer request(&request_slice, 1);
    call->Write(request, &write_tag);
    await(&write_tag, true);
    call->WritesDone(&writes_done_tag);
    await(&writes_done_tag, true);

    grpc::ByteBuffer response;
    call->Read(&response, &read_tag);
    await(&read_tag, true);
    grpc::ByteBuffer ignored;
    call->Read(&ignored, &eof_tag);
    await(&eof_tag, false);
    grpc::Status status;
    call->Finish(&status, &finish_tag);
    await(&finish_tag, true);
    queue.Shutdown();
    return {status, flatten(response)};
}
#endif

void set_ids(
    wire::IdBatch& output,
    const std::span<const std::uint64_t> ids) {
    wire::UInt64Ids* encoded = output.mutable_uint64_ids();
    encoded->set_count(ids.size());
    std::string bytes;
    bytes.reserve(ids.size() * 8);
    for (const std::uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    *encoded->mutable_values_le() = std::move(bytes);
}

[[nodiscard]] std::vector<std::uint64_t> read_ids(
    const wire::IdBatch& input) {
    require(
        input.encoding_case() == wire::IdBatch::kUint64Ids,
        "gRPC response did not contain uint64 IDs");
    const wire::UInt64Ids& encoded = input.uint64_ids();
    require(
        encoded.values_le().size() == encoded.count() * 8,
        "gRPC response ID geometry changed");
    std::vector<std::uint64_t> result(encoded.count());
    for (std::size_t index = 0; index != result.size(); ++index) {
        result[index] = read_u64_le(
            encoded.values_le().data() + index * 8);
    }
    return result;
}

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> result(count * record_size);
    for (std::size_t index = 0; index != result.size(); ++index) {
        result[index] = static_cast<std::byte>(salt + index);
    }
    return result;
}

void test_grpc_crud_and_scan() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = 8;
    const std::array<std::uint64_t, 4> base_ids{10, 20, 30, 40};
    const std::vector<std::byte> base_rows =
        rows(base_ids.size(), record_size, 3);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = dimension;
    builder_options.dtype = throwdb::DType::Float16;
    builder_options.segment_size_bytes = 2 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(base_ids, base_rows);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    throwdb::server::SingleCollectionEngine engine(*collection, store);
    throwdb::server::ServiceLimits limits;
    limits.max_request_bytes = 64ULL << 20;
    limits.max_response_bytes = 64ULL << 20;
    throwdb::server::GrpcServerOptions grpc_options;
    grpc_options.max_worker_threads = 8;
    throwdb::server::GrpcServer server(engine, limits, grpc_options);
    const auto endpoint = server.start();

    grpc::ChannelArguments channel_options;
    channel_options.SetMaxSendMessageSize(64 << 20);
    channel_options.SetMaxReceiveMessageSize(64 << 20);
    // A developer shell may export HTTP_PROXY. Loopback smoke tests must
    // never leave the host or depend on NO_PROXY spelling.
    channel_options.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    auto channel = grpc::CreateCustomChannel(
        endpoint.to_string(), grpc::InsecureChannelCredentials(),
        channel_options);
    require(
        channel->WaitForConnected(
            std::chrono::system_clock::now() + std::chrono::seconds(5)),
        "gRPC channel did not become ready");
    auto stub = wire::ThrowDB::NewStub(channel);

    const auto [standard_health_status, standard_health_response] =
        generic_unary(
            channel, "/grpc.health.v1.Health/Check", std::string{});
    require(
        standard_health_status.ok()
            && standard_health_response
                   == std::string{"\x08\x01", 2},
        "standard grpc.health.v1.Health did not report SERVING");

#if defined(THROWDB_TEST_GRPC_REFLECTION)
    const auto [reflection_status, reflected_services] =
        reflection_services(channel);
    require(
        reflection_status.ok()
            && reflected_services.find("throwdb.v1.ThrowDB")
                   != std::string::npos
            && reflected_services.find("grpc.health.v1.Health")
                   != std::string::npos,
        "gRPC reflection did not expose ThrowDB and health services");
#else
    const auto [reflection_status, ignored_reflection_response] =
        generic_unary(
            channel,
            "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
            std::string{"\x3a\x00", 2});
    static_cast<void>(ignored_reflection_response);
    require(
        reflection_status.error_code() == grpc::StatusCode::UNIMPLEMENTED,
        "gRPC reflection was registered despite being disabled");
#endif

    grpc::ClientContext health_context;
    wire::HealthRequest health_request;
    wire::HealthResponse health_response;
    grpc::Status status = stub->Health(
        &health_context, health_request, &health_response);
    require(
        status.ok()
            && health_response.status()
                   == wire::HealthResponse::SERVING_STATUS_SERVING,
        "typed gRPC Health failed");

    grpc::ClientContext describe_context;
    wire::DescribeRequest describe_request;
    wire::DescribeResponse describe_response;
    status = stub->Describe(
        &describe_context, describe_request, &describe_response);
    require(
        status.ok() && describe_response.format_version() == 7
            && describe_response.schema().dimension() == dimension
            && describe_response.count() == base_ids.size()
            && describe_response.capabilities().writable()
            && describe_response.capabilities().durable_idempotency(),
        "gRPC Describe returned wrong collection metadata");

    wire::MutationRequest insert_request;
    insert_request.mutable_operation_id()->set_high(1);
    insert_request.mutable_operation_id()->set_low(2);
    const std::array<std::uint64_t, 1> insert_ids{50};
    set_ids(*insert_request.mutable_ids(), insert_ids);
    const auto inserted_row = rows(1, record_size, 91);
    wire::EmbeddingBatch* insert_embeddings =
        insert_request.mutable_embeddings();
    insert_embeddings->set_dtype(wire::EMBEDDING_DTYPE_FLOAT16);
    insert_embeddings->set_dimension(dimension);
    insert_embeddings->set_row_count(1);
    insert_embeddings->set_values_le(
        reinterpret_cast<const char*>(inserted_row.data()),
        inserted_row.size());
    grpc::ClientContext insert_context;
    wire::MutationResponse insert_response;
    status = stub->Insert(
        &insert_context, insert_request, &insert_response);
    require(
        status.ok() && insert_response.committed_sequence() != 0
            && insert_response.affected_count() == 1,
        "gRPC Insert failed");

    wire::DeleteRequest delete_request;
    delete_request.mutable_operation_id()->set_high(3);
    delete_request.mutable_operation_id()->set_low(4);
    const std::array<std::uint64_t, 2> delete_ids{30, 999};
    set_ids(*delete_request.mutable_ids(), delete_ids);
    grpc::ClientContext delete_context;
    wire::DeleteResponse delete_response;
    status = stub->Delete(
        &delete_context, delete_request, &delete_response);
    require(
        status.ok() && delete_response.affected_count() == 2
            && delete_response.deleted_mask().size() == 2
            && delete_response.deleted_mask()[0] == 1
            && delete_response.deleted_mask()[1] == 0,
        "gRPC Delete failed");

    wire::ScanIdsRequest scan_request;
    scan_request.set_batch_rows(2);
    scan_request.set_include_locations(true);
    grpc::ClientContext scan_context;
    std::unique_ptr<grpc::ClientReader<wire::ScanIdsResponse>> reader =
        stub->ScanIds(&scan_context, scan_request);
    wire::ScanIdsResponse chunk;
    std::set<std::uint64_t> scanned;
    std::uint64_t expected_offset = 0;
    std::uint64_t previous_location = 0;
    bool first_location = true;
    while (reader->Read(&chunk)) {
        require(
            chunk.emitted_offset() == expected_offset
                && chunk.location_encoding()
                       == wire::LOCATION_ENCODING_SEGMENT_ROW_U32_LE,
            "gRPC ScanIds chunk metadata changed");
        const auto chunk_ids = read_ids(chunk.ids());
        require(
            chunk.locations_le().size() == chunk_ids.size() * 8,
            "gRPC ScanIds location geometry changed");
        for (std::size_t index = 0; index != chunk_ids.size(); ++index) {
            require(
                scanned.insert(chunk_ids[index]).second,
                "gRPC ScanIds duplicated an ID");
            const std::uint32_t segment = static_cast<std::uint32_t>(
                read_u64_le(chunk.locations_le().data() + index * 8));
            const std::uint32_t row = static_cast<std::uint32_t>(
                read_u64_le(chunk.locations_le().data() + index * 8) >> 32);
            const std::uint64_t location =
                (static_cast<std::uint64_t>(segment) << 32) | row;
            require(
                first_location || previous_location < location,
                "gRPC ScanIds lost physical order");
            first_location = false;
            previous_location = location;
        }
        expected_offset += chunk_ids.size();
    }
    status = reader->Finish();
    require(
        status.ok() && expected_offset == 4
            && scanned == std::set<std::uint64_t>({10, 20, 40, 50}),
        "gRPC ScanIds returned wrong live IDs");

    server.shutdown();
}

void test_grpc_rejects_legacy_storage_format() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "legacy-store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = 8;
    const std::array<std::uint64_t, 1> ids{10};
    const std::vector<std::byte> embeddings = rows(1, record_size, 9);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 6;
    builder_options.dimension = dimension;
    builder_options.dtype = throwdb::DType::Float16;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    throwdb::server::EngineOptions engine_options;
    engine_options.durable_idempotency = false;
    throwdb::server::SingleCollectionEngine engine(
        *collection, store, engine_options);

    bool rejected = false;
    try {
        throwdb::server::GrpcServer server(engine);
    } catch (const throwdb::InvalidArgumentError& error) {
        rejected = std::string(error.what()).find("format 7")
                   != std::string::npos;
    }
    require(
        rejected,
        "gRPC server did not reject a legacy storage format clearly");
}

}  // namespace

int main() {
    try {
        test_grpc_rejects_legacy_storage_format();
        test_grpc_crud_and_scan();
        std::cout << "gRPC server tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gRPC server test failure: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "gRPC server test failure: unknown exception\n";
        return 1;
    }
}
