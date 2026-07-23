#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

[[nodiscard]] std::string read_text_file(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

struct DaemonResult {
    int exit_code = -1;
    std::string stdout_text;
    std::string stderr_text;
};

[[nodiscard]] DaemonResult run_terminal_daemon(
    const std::filesystem::path& store,
    const std::filesystem::path& output_directory,
    const bool block_after_drain) {
    const auto stdout_path = output_directory / "daemon.stdout";
    const auto stderr_path = output_directory / "daemon.stderr";
    const pid_t child = ::fork();
    if (child < 0) {
        throw std::runtime_error("cannot fork terminal-daemon regression");
    }
    if (child == 0) {
        const int stdout_fd = ::open(
            stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        const int stderr_fd = ::open(
            stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (stdout_fd < 0 || stderr_fd < 0
            || ::dup2(stdout_fd, STDOUT_FILENO) < 0
            || ::dup2(stderr_fd, STDERR_FILENO) < 0) {
            ::_exit(126);
        }
        static_cast<void>(::close(stdout_fd));
        static_cast<void>(::close(stderr_fd));
        static_cast<void>(::setenv(
            "THROWDB_TEST_TERMINAL_AFTER_READY", "1", 1));
        if (block_after_drain) {
            static_cast<void>(::setenv(
                "THROWDB_TEST_BLOCK_AFTER_DRAIN", "1", 1));
        } else {
            static_cast<void>(::unsetenv(
                "THROWDB_TEST_BLOCK_AFTER_DRAIN"));
        }
        const std::string daemon = THROWDB_TEST_GRPC_DAEMON_PATH;
        const std::string storage = store.string();
        const char* const hard_timeout = block_after_drain ? "100" : "5000";
        ::execl(
            daemon.c_str(), daemon.c_str(),
            "--storage", storage.c_str(),
            "--listen", "127.0.0.1:0",
            "--read-only", "--no-auto-maintenance",
            "--shutdown-grace-ms", "1",
            "--shutdown-hard-timeout-ms", hard_timeout,
            static_cast<char*>(nullptr));
        ::_exit(127);
    }

    int status = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        const pid_t waited = ::waitpid(child, &status, WNOHANG);
        if (waited == child) {
            break;
        }
        if (waited < 0) {
            static_cast<void>(::kill(child, SIGKILL));
            static_cast<void>(::waitpid(child, &status, 0));
            throw std::runtime_error("cannot wait for terminal daemon");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            static_cast<void>(::kill(child, SIGKILL));
            static_cast<void>(::waitpid(child, &status, 0));
            throw std::runtime_error("terminal daemon exceeded test deadline");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return DaemonResult{
        exit_code,
        read_text_file(stdout_path),
        read_text_file(stderr_path)};
}

void test_daemon_terminal_exit_and_hard_shutdown_watchdog() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    std::filesystem::create_directories(temporary.path());
    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = 1;
    builder_options.dtype = throwdb::DType::Float16;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    const std::array<std::uint64_t, 1> ids{1};
    const std::array<std::byte, 2> embedding{};
    builder->append(ids, embedding);
    builder->finish();

    const auto terminal = run_terminal_daemon(
        store, temporary.path(), false);
    require(
        terminal.exit_code == EXIT_FAILURE
            && terminal.stdout_text.find("\"event\":\"ready\"")
                   != std::string::npos,
        "terminal storage state did not make the daemon exit nonzero after "
        "publishing readiness; exit=" + std::to_string(terminal.exit_code)
            + " stdout=" + terminal.stdout_text
            + " stderr=" + terminal.stderr_text);

    const auto bounded = run_terminal_daemon(
        store, temporary.path(), true);
    require(
        bounded.exit_code == EXIT_FAILURE
            && bounded.stdout_text.find("\"event\":\"ready\"")
                   != std::string::npos
            && bounded.stderr_text.find("test_shutdown_block_entered")
                   != std::string::npos,
        "hard shutdown watchdog did not terminate a daemon blocked after "
        "drain began; exit=" + std::to_string(bounded.exit_code)
            + " stdout=" + bounded.stdout_text
            + " stderr=" + bounded.stderr_text);
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

    throwdb::ReaderOptions reader_options;
    reader_options.max_read_range_bytes = 4ULL << 20;
    reader_options.max_gap_bytes = 64;
    reader_options.io_workers = 2;
    reader_options.io_queue_capacity = 6;
    reader_options.io_ranges_per_task = 3;
    throwdb::server::EngineOptions engine_options;
    engine_options.reader = reader_options;
    engine_options.tuning.mode = "auto";
    engine_options.tuning.source = "persisted";
    engine_options.tuning.status = "selected";
    engine_options.tuning.reuse_level = "compatible";
    auto collection = throwdb::NativeCollection::open(
        store, reader_options, true);
    throwdb::server::SingleCollectionEngine engine(
        *collection, store, engine_options);
    throwdb::server::ServiceLimits limits;
    limits.max_request_bytes = 64ULL << 20;
    limits.max_response_bytes = 64ULL << 20;
    throwdb::server::GrpcServerOptions grpc_options;
    grpc_options.max_worker_threads = 8;
    grpc_options.max_concurrent_calls = 4;
    grpc_options.defer_serving_until_ready = true;
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

    const auto [starting_health_status, starting_health_response] =
        generic_unary(
            channel, "/grpc.health.v1.Health/Check", std::string{});
    require(
        starting_health_status.ok()
            && starting_health_response == std::string{"\x08\x02", 2},
        "deferred gRPC server advertised readiness before its startup barrier");
    wire::GetRequest starting_get_request;
    set_ids(
        *starting_get_request.mutable_ids(),
        std::span<const std::uint64_t>(base_ids).first<1>());
    wire::GetResponse starting_get_response;
    grpc::ClientContext starting_get_context;
    const grpc::Status starting_get_status = stub->Get(
        &starting_get_context, starting_get_request, &starting_get_response);
    require(
        starting_get_status.error_code() == grpc::StatusCode::UNAVAILABLE,
        "deferred gRPC server admitted data work before readiness");
    server.mark_serving();

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
                   == wire::HealthResponse::SERVING_STATUS_SERVING
            && health_response.protocol_minor() == 3,
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
            && describe_response.capabilities().durable_idempotency()
            && describe_response.capabilities().stream_get()
            && describe_response.capabilities().scan_records()
            && (describe_response.idempotency_domain().high() != 0
                || describe_response.idempotency_domain().low() != 0)
            && describe_response.reader().max_read_range_bytes()
                   == reader_options.max_read_range_bytes
            && describe_response.reader().max_gap_bytes()
                   == reader_options.max_gap_bytes
            && describe_response.reader().io_workers()
                   == reader_options.io_workers
            && describe_response.reader().io_queue_capacity()
                   == reader_options.io_queue_capacity
            && describe_response.reader().io_ranges_per_task()
                   == reader_options.io_ranges_per_task
            && describe_response.reader().io_backend()
                   == wire::READER_BACKEND_THREAD_POOL
            && describe_response.tuning().mode() == "auto"
            && describe_response.tuning().source() == "persisted"
            && describe_response.tuning().status() == "selected"
            && describe_response.tuning().reuse_level() == "compatible"
            && !describe_response.tuning().explicit_overrides()
            && describe_response.availability().max_concurrent_calls()
                   == grpc_options.max_concurrent_calls
            && describe_response.availability().max_concurrent_scans()
                   == grpc_options.max_concurrent_scans
            && describe_response.availability().max_unary_inflight_bytes()
                   == grpc_options.max_inflight_bytes
            && describe_response.availability().max_scan_inflight_bytes()
                   == grpc_options.max_scan_inflight_bytes
            && describe_response.availability().scan_max_duration_millis()
                   == grpc_options.scan_max_duration_millis
            && describe_response.availability().scan_idle_timeout_millis()
                   == grpc_options.scan_idle_timeout_millis
            && describe_response.availability().shutdown_hard_timeout_millis()
                   == grpc_options.shutdown_hard_timeout_millis
            && describe_response.admission_stats().active_unary_calls() == 0
            && describe_response.admission_stats().active_scans() == 0
            && describe_response.admission_stats().admitted_unary_calls() == 0
            && describe_response.admission_stats().admitted_scans() == 0
            && !describe_response.admission_stats().admission_closed()
            && !describe_response.admission_stats().draining()
            && !describe_response.admission_stats().terminal_faulted()
            && describe_response.engine_stats().catalog_generation() != 0
            && describe_response.engine_stats().mutation_disk_reserve_bytes()
                   == engine_options.mutation_disk_reserve_bytes,
        "gRPC Describe returned wrong collection metadata");

    wire::MutationRequest insert_request;
    insert_request.mutable_operation_id()->set_high(1);
    insert_request.mutable_operation_id()->set_low(2);
    insert_request.mutable_idempotency_domain()->CopyFrom(
        describe_response.idempotency_domain());
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
    wire::MutationRequest stale_domain_request = insert_request;
    stale_domain_request.mutable_operation_id()->set_high(8);
    stale_domain_request.mutable_operation_id()->set_low(9);
    stale_domain_request.mutable_idempotency_domain()->set_low(
        stale_domain_request.idempotency_domain().low() ^ 1U);
    grpc::ClientContext stale_domain_context;
    wire::MutationResponse stale_domain_response;
    status = stub->Insert(
        &stale_domain_context,
        stale_domain_request,
        &stale_domain_response);
    require(
        status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
        "gRPC Insert accepted an expired idempotency replay domain");
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
    delete_request.mutable_idempotency_domain()->CopyFrom(
        describe_response.idempotency_domain());
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

    // StreamGet captures its view at the open handshake. A later Update must
    // not change any batch returned through the already-open stream.
    grpc::ClientContext stream_get_context;
    auto stream_get = stub->StreamGet(&stream_get_context);
    wire::StreamGetRequest stream_open;
    stream_open.mutable_open()->set_prefix_dimensions(2);
    stream_open.mutable_open()->set_max_batch_rows(4);
    stream_open.mutable_open()->set_max_batch_payload_bytes(64);
    require(
        stream_get->Write(stream_open),
        "gRPC StreamGet rejected its open frame");
    wire::StreamGetResponse stream_opened;
    require(
        stream_get->Read(&stream_opened)
            && stream_opened.payload_case()
                   == wire::StreamGetResponse::kOpened
            && stream_opened.opened().dimension() == 2
            && stream_opened.opened().dtype()
                   == wire::EMBEDDING_DTYPE_FLOAT16
            && stream_opened.opened().request_window() == 1
            && stream_opened.opened().max_batch_rows() <= 4
            && stream_opened.opened().max_batch_payload_bytes() <= 64
            && stream_opened.snapshot().live_count() == 4,
        "gRPC StreamGet returned an invalid open handshake");

    wire::MutationRequest update_request;
    update_request.mutable_operation_id()->set_high(5);
    update_request.mutable_operation_id()->set_low(6);
    update_request.mutable_idempotency_domain()->CopyFrom(
        describe_response.idempotency_domain());
    const std::array<std::uint64_t, 1> update_ids{10};
    set_ids(*update_request.mutable_ids(), update_ids);
    const auto updated_row = rows(1, record_size, 171);
    auto* update_embeddings = update_request.mutable_embeddings();
    update_embeddings->set_dtype(wire::EMBEDDING_DTYPE_FLOAT16);
    update_embeddings->set_dimension(dimension);
    update_embeddings->set_row_count(1);
    update_embeddings->set_values_le(
        reinterpret_cast<const char*>(updated_row.data()),
        updated_row.size());
    grpc::ClientContext update_context;
    wire::MutationResponse update_response;
    status = stub->Update(
        &update_context, update_request, &update_response);
    require(
        status.ok()
            && update_response.committed_sequence()
                   > stream_opened.snapshot().committed_sequence(),
        "gRPC Update did not advance beyond the pinned StreamGet snapshot");

    wire::StreamGetRequest stream_batch;
    stream_batch.mutable_batch()->set_sequence(0);
    const std::array<std::uint64_t, 3> stream_ids{10, 40, 10};
    set_ids(*stream_batch.mutable_batch()->mutable_ids(), stream_ids);
    require(
        stream_get->Write(stream_batch) && stream_get->WritesDone(),
        "gRPC StreamGet could not send its ID batch");
    wire::StreamGetResponse stream_output;
    require(
        stream_get->Read(&stream_output)
            && stream_output.payload_case()
                   == wire::StreamGetResponse::kBatch
            && stream_output.batch().sequence() == 0
            && stream_output.batch().input_offset() == 0
            && stream_output.batch().embeddings().row_count()
                   == stream_ids.size()
            && stream_output.batch().embeddings().dimension() == 2
            && stream_output.snapshot().committed_sequence()
                   == stream_opened.snapshot().committed_sequence()
            && stream_output.snapshot().token().high()
                   == stream_opened.snapshot().token().high()
            && stream_output.snapshot().token().low()
                   == stream_opened.snapshot().token().low(),
        "gRPC StreamGet returned invalid batch metadata");
    std::string expected_stream_values;
    expected_stream_values.append(
        reinterpret_cast<const char*>(base_rows.data()), 4);
    expected_stream_values.append(
        reinterpret_cast<const char*>(base_rows.data() + 3 * record_size),
        4);
    expected_stream_values.append(
        reinterpret_cast<const char*>(base_rows.data()), 4);
    require(
        stream_output.batch().embeddings().values_le()
                == expected_stream_values
            && !stream_get->Read(&stream_output)
            && stream_get->Finish().ok(),
        "gRPC StreamGet did not preserve pinned values/order/duplicates");

    // A half-close immediately after Open is the valid empty-ID request.
    grpc::ClientContext empty_get_context;
    auto empty_get = stub->StreamGet(&empty_get_context);
    wire::StreamGetRequest empty_open;
    empty_open.mutable_open()->set_max_batch_rows(2);
    empty_open.mutable_open()->set_max_batch_payload_bytes(64);
    wire::StreamGetResponse empty_opened;
    require(
        empty_get->Write(empty_open)
            && empty_get->Read(&empty_opened)
            && empty_get->WritesDone()
            && !empty_get->Read(&empty_opened)
            && empty_get->Finish().ok(),
        "gRPC StreamGet rejected an empty ID stream");

    // Missing IDs terminate the stream with the typed NOT_FOUND status after
    // prior batches (if any) have already remained visible to the caller.
    grpc::ClientContext missing_get_context;
    auto missing_get = stub->StreamGet(&missing_get_context);
    wire::StreamGetRequest missing_open;
    missing_open.mutable_open()->set_max_batch_rows(1);
    missing_open.mutable_open()->set_max_batch_payload_bytes(record_size);
    wire::StreamGetResponse missing_response;
    require(
        missing_get->Write(missing_open)
            && missing_get->Read(&missing_response),
        "gRPC missing-ID StreamGet did not complete its handshake");
    wire::StreamGetRequest missing_batch;
    missing_batch.mutable_batch()->set_sequence(0);
    const std::array<std::uint64_t, 1> missing_stream_id{999};
    set_ids(
        *missing_batch.mutable_batch()->mutable_ids(),
        missing_stream_id);
    require(
        missing_get->Write(missing_batch) && missing_get->WritesDone()
            && !missing_get->Read(&missing_response),
        "gRPC missing-ID StreamGet unexpectedly returned a data batch");
    const grpc::Status missing_stream_status = missing_get->Finish();
    require(
        missing_stream_status.error_code() == grpc::StatusCode::NOT_FOUND,
        "gRPC StreamGet did not return NOT_FOUND for a missing ID");

    grpc::ClientContext cancelled_get_context;
    auto cancelled_get = stub->StreamGet(&cancelled_get_context);
    wire::StreamGetRequest cancelled_open;
    cancelled_open.mutable_open()->set_max_batch_rows(1);
    cancelled_open.mutable_open()->set_max_batch_payload_bytes(record_size);
    wire::StreamGetResponse cancelled_opened;
    require(
        cancelled_get->Write(cancelled_open)
            && cancelled_get->Read(&cancelled_opened),
        "gRPC cancellable StreamGet did not complete its handshake");
    cancelled_get_context.TryCancel();
    while (cancelled_get->Read(&cancelled_opened)) {
    }
    require(
        cancelled_get->Finish().error_code()
            == grpc::StatusCode::CANCELLED,
        "gRPC StreamGet cancellation did not reach the server");

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

    wire::ScanRecordsRequest records_request;
    records_request.set_batch_rows(1);
    records_request.set_max_batch_payload_bytes(20);
    records_request.set_prefix_dimensions(2);
    records_request.set_include_locations(true);
    grpc::ClientContext records_context;
    auto records_reader = stub->ScanRecords(
        &records_context, records_request);
    wire::ScanRecordsResponse record_chunk;
    require(
        records_reader->Read(&record_chunk)
            && record_chunk.snapshot().live_count() == 4
            && record_chunk.max_batch_rows() == 1
            && record_chunk.max_batch_payload_bytes() == 20,
        "gRPC ScanRecords did not open its physical snapshot");

    // Publish a new record after the first response. The pinned scan must
    // still end at its original live_count and never expose this ID.
    wire::MutationRequest later_insert_request;
    later_insert_request.mutable_operation_id()->set_high(7);
    later_insert_request.mutable_operation_id()->set_low(8);
    later_insert_request.mutable_idempotency_domain()->CopyFrom(
        describe_response.idempotency_domain());
    const std::array<std::uint64_t, 1> later_insert_ids{60};
    set_ids(*later_insert_request.mutable_ids(), later_insert_ids);
    const auto later_insert_row = rows(1, record_size, 211);
    auto* later_insert_embeddings =
        later_insert_request.mutable_embeddings();
    later_insert_embeddings->set_dtype(wire::EMBEDDING_DTYPE_FLOAT16);
    later_insert_embeddings->set_dimension(dimension);
    later_insert_embeddings->set_row_count(1);
    later_insert_embeddings->set_values_le(
        reinterpret_cast<const char*>(later_insert_row.data()),
        later_insert_row.size());
    grpc::ClientContext later_insert_context;
    wire::MutationResponse later_insert_response;
    status = stub->Insert(
        &later_insert_context,
        later_insert_request,
        &later_insert_response);
    require(
        status.ok()
            && later_insert_response.committed_sequence()
                   > record_chunk.snapshot().committed_sequence(),
        "gRPC Insert did not advance beyond the pinned ScanRecords snapshot");

    std::set<std::uint64_t> scanned_records;
    std::uint64_t records_offset = 0;
    std::uint64_t previous_record_location = 0;
    bool first_record_location = true;
    const std::uint64_t record_snapshot_sequence =
        record_chunk.snapshot().committed_sequence();
    const std::uint64_t record_token_high =
        record_chunk.snapshot().token().high();
    const std::uint64_t record_token_low =
        record_chunk.snapshot().token().low();
    do {
        const auto record_ids = read_ids(record_chunk.ids());
        require(
            record_chunk.emitted_offset() == records_offset
                && record_chunk.snapshot().committed_sequence()
                       == record_snapshot_sequence
                && record_chunk.snapshot().token().high()
                       == record_token_high
                && record_chunk.snapshot().token().low()
                       == record_token_low
                && record_chunk.embeddings().dimension() == 2
                && record_chunk.embeddings().row_count()
                       == record_ids.size()
                && record_chunk.embeddings().values_le().size()
                       == record_ids.size() * 4
                && record_chunk.location_encoding()
                       == wire::LOCATION_ENCODING_SEGMENT_ROW_U32_LE
                && record_chunk.locations_le().size()
                       == record_ids.size() * 8,
            "gRPC ScanRecords returned invalid batch geometry");
        for (std::size_t index = 0; index != record_ids.size(); ++index) {
            require(
                scanned_records.insert(record_ids[index]).second,
                "gRPC ScanRecords duplicated an ID");
            const std::uint32_t segment = static_cast<std::uint32_t>(
                read_u64_le(
                    record_chunk.locations_le().data() + index * 8));
            const std::uint32_t row = static_cast<std::uint32_t>(
                read_u64_le(
                    record_chunk.locations_le().data() + index * 8)
                >> 32);
            const std::uint64_t location =
                (static_cast<std::uint64_t>(segment) << 32) | row;
            require(
                first_record_location
                    || previous_record_location < location,
                "gRPC ScanRecords lost physical order");
            first_record_location = false;
            previous_record_location = location;
        }
        records_offset += record_ids.size();
    } while (records_reader->Read(&record_chunk));
    require(
        records_reader->Finish().ok() && records_offset == 4
            && scanned_records
                   == std::set<std::uint64_t>({10, 20, 40, 50})
            && scanned_records.find(60) == scanned_records.end(),
        "gRPC ScanRecords was not pinned across a later mutation");

    grpc::ClientContext cancelled_records_context;
    auto cancelled_records = stub->ScanRecords(
        &cancelled_records_context, records_request);
    require(
        cancelled_records->Read(&record_chunk),
        "gRPC cancellable ScanRecords did not return its first batch");
    cancelled_records_context.TryCancel();
    while (cancelled_records->Read(&record_chunk)) {
    }
    require(
        cancelled_records->Finish().error_code()
            == grpc::StatusCode::CANCELLED,
        "gRPC ScanRecords cancellation did not reach the client");

    wire::DeleteRequest delete_all_request;
    delete_all_request.mutable_operation_id()->set_high(9);
    delete_all_request.mutable_operation_id()->set_low(10);
    delete_all_request.mutable_idempotency_domain()->CopyFrom(
        describe_response.idempotency_domain());
    const std::array<std::uint64_t, 5> all_live_ids{10, 20, 40, 50, 60};
    set_ids(*delete_all_request.mutable_ids(), all_live_ids);
    grpc::ClientContext delete_all_context;
    wire::DeleteResponse delete_all_response;
    status = stub->Delete(
        &delete_all_context, delete_all_request, &delete_all_response);
    require(
        status.ok()
            && delete_all_response.deleted_mask()
                   == std::string(all_live_ids.size(), '\1'),
        "gRPC Delete could not prepare the empty ScanRecords snapshot");

    grpc::ClientContext empty_records_context;
    auto empty_records = stub->ScanRecords(
        &empty_records_context, records_request);
    wire::ScanRecordsResponse empty_record_chunk;
    require(
        empty_records->Read(&empty_record_chunk)
            && empty_record_chunk.snapshot().live_count() == 0
            && empty_record_chunk.emitted_offset() == 0
            && read_ids(empty_record_chunk.ids()).empty()
            && empty_record_chunk.embeddings().row_count() == 0
            && empty_record_chunk.embeddings().values_le().empty()
            && !empty_records->Read(&empty_record_chunk)
            && empty_records->Finish().ok(),
        "gRPC ScanRecords did not represent an empty snapshot exactly once");

    const throwdb::server::GrpcAdmissionStats running_stats =
        server.admission_stats();
    require(
        running_stats.active_unary_calls == 0
            && running_stats.active_scans == 0
            && running_stats.unary_estimated_bytes == 0
            && running_stats.scan_estimated_bytes == 0
            && running_stats.admitted_unary_calls == 6
            && running_stats.rejected_unary_calls == 0
            && running_stats.admitted_scans == 8
            && running_stats.rejected_scans == 0
            && !running_stats.admission_closed
            && !running_stats.draining
            && !running_stats.terminal_faulted,
        "public gRPC admission statistics were inconsistent after CRUD/scan");

    server.begin_draining();
    const throwdb::server::GrpcAdmissionStats draining_stats =
        server.admission_stats();
    require(
        draining_stats.admission_closed && draining_stats.draining
            && !draining_stats.terminal_faulted,
        "public gRPC admission statistics did not expose draining");
    grpc::ClientContext draining_describe_context;
    wire::DescribeResponse draining_describe_response;
    status = stub->Describe(
        &draining_describe_context,
        describe_request,
        &draining_describe_response);
    require(
        status.ok()
            && draining_describe_response.admission_stats()
                   .admission_closed()
            && draining_describe_response.admission_stats().draining()
            && !draining_describe_response.admission_stats()
                    .terminal_faulted(),
        "gRPC Describe did not expose draining admission state");
    grpc::ClientContext draining_health_context;
    wire::HealthResponse draining_health_response;
    status = stub->Health(
        &draining_health_context,
        health_request,
        &draining_health_response);
    require(
        status.ok()
            && draining_health_response.status()
                   == wire::HealthResponse::SERVING_STATUS_NOT_SERVING,
        "typed gRPC Health did not become NOT_SERVING while draining");
    const auto [draining_standard_status, draining_standard_response] =
        generic_unary(
            channel, "/grpc.health.v1.Health/Check", std::string{});
    require(
        draining_standard_status.ok()
            && draining_standard_response
                   == std::string{"\x08\x02", 2},
        "standard grpc.health.v1.Health did not become NOT_SERVING while "
        "draining");

    grpc::ClientContext draining_insert_context;
    wire::MutationResponse draining_insert_response;
    status = stub->Insert(
        &draining_insert_context, insert_request, &draining_insert_response);
    require(
        status.error_code() == grpc::StatusCode::UNAVAILABLE,
        "gRPC data plane accepted a mutation after draining began");

    server.shutdown();
}

void test_stalled_scan_is_cancelled_and_releases_admission() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint64_t row_count = 524'288;
    std::vector<std::uint64_t> ids(static_cast<std::size_t>(row_count));
    for (std::uint64_t index = 0; index != row_count; ++index) {
        ids[static_cast<std::size_t>(index)] = index + 1;
    }
    std::vector<std::byte> embeddings(
        static_cast<std::size_t>(row_count) * 2);
    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = 1;
    builder_options.dtype = throwdb::DType::Float16;
    builder_options.segment_size_bytes = 8ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store);
    throwdb::server::SingleCollectionEngine engine(*collection, store);
    throwdb::server::ServiceLimits limits;
    limits.max_response_bytes = 2ULL << 20;
    limits.max_scan_batch_rows = 65'536;
    throwdb::server::GrpcServerOptions options;
    options.max_worker_threads = 4;
    options.max_concurrent_calls = 2;
    options.max_concurrent_scans = 1;
    options.max_scan_inflight_bytes = 8ULL << 20;
    options.max_grpc_memory_bytes = 8ULL << 20;
    options.scan_max_duration_millis = 250;
    options.scan_idle_timeout_millis = 250;
    throwdb::server::GrpcServer server(engine, limits, options);
    const auto endpoint = server.start();

    grpc::ChannelArguments channel_options;
    channel_options.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    channel_options.SetMaxReceiveMessageSize(2 << 20);
    auto channel = grpc::CreateCustomChannel(
        endpoint.to_string(), grpc::InsecureChannelCredentials(),
        channel_options);
    require(
        channel->WaitForConnected(
            std::chrono::system_clock::now() + std::chrono::seconds(5)),
        "stalled-scan channel did not become ready");
    auto stub = wire::ThrowDB::NewStub(channel);

    const auto wait_for_stats = [&](const auto& predicate,
                                    const std::string& message) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto stats = server.admission_stats();
            if (predicate(stats)) {
                return stats;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error(message);
    };

    wire::ScanIdsRequest request;
    request.set_batch_rows(65'536);
    request.set_include_locations(true);
    {
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto reader = stub->ScanIds(&context, request);
        static_cast<void>(wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 1
                       && stats.admitted_scans == 1;
            },
            "ScanIds never acquired its admission lease"));
        // Deliberately do not call Read while the server fills the client's
        // bounded transport quota. The server watchdog must cancel the real
        // HTTP/2 stream and unwind the synchronous handler.
        const auto released = wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 0
                       && stats.scan_estimated_bytes == 0;
            },
            "stalled ScanIds retained admission after its server deadline");
        require(
            released.admitted_scans == 1 && released.rejected_scans == 0,
            "stalled ScanIds corrupted admission counters");
        wire::ScanIdsResponse discarded;
        while (reader->Read(&discarded)) {
        }
        const grpc::Status status = reader->Finish();
        require(
            status.error_code() == grpc::StatusCode::CANCELLED
                && status.error_message().find("server side")
                       != std::string::npos,
            "stalled ScanIds was not cancelled by the server watchdog; "
            "code=" + std::to_string(status.error_code())
                + " message=" + status.error_message());
    }

    {
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto reader = stub->ScanIds(&context, request);
        wire::ScanIdsResponse first;
        require(
            reader->Read(&first),
            "a new ScanIds call was not admitted after stalled-stream cleanup");
        context.TryCancel();
        while (reader->Read(&first)) {
        }
        static_cast<void>(reader->Finish());
    }
    const auto replacement_stats = wait_for_stats(
        [](const throwdb::server::GrpcAdmissionStats& stats) {
            return stats.active_scans == 0 && stats.admitted_scans == 2
                   && stats.scan_estimated_bytes == 0;
        },
        "replacement ScanIds did not release its admission lease");
    require(
        replacement_stats.rejected_scans == 0,
        "released ScanIds capacity was not reusable");

    wire::ScanRecordsRequest records_request;
    records_request.set_batch_rows(65'536);
    records_request.set_max_batch_payload_bytes((2ULL << 20) - 1024);
    records_request.set_prefix_dimensions(1);
    records_request.set_include_locations(true);
    {
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto reader = stub->ScanRecords(&context, records_request);
        static_cast<void>(wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 1
                       && stats.admitted_scans == 3;
            },
            "ScanRecords never acquired its admission lease"));
        static_cast<void>(wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 0
                       && stats.scan_estimated_bytes == 0;
            },
            "stalled ScanRecords retained admission after its watchdog"));
        wire::ScanRecordsResponse discarded;
        while (reader->Read(&discarded)) {
        }
        require(
            reader->Finish().error_code() == grpc::StatusCode::CANCELLED,
            "stalled ScanRecords was not cancelled by its watchdog");
    }

    // No request frame at all is the bidi slowloris case. StreamGet must
    // acquire a scan slot and arm its idle watchdog before its first Read.
    {
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto stream = stub->StreamGet(&context);
        static_cast<void>(wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 1
                       && stats.admitted_scans == 4;
            },
            "pre-open StreamGet never acquired its admission lease"));
        static_cast<void>(wait_for_stats(
            [](const throwdb::server::GrpcAdmissionStats& stats) {
                return stats.active_scans == 0
                       && stats.scan_estimated_bytes == 0;
            },
            "pre-open StreamGet retained admission after its idle watchdog"));
        wire::StreamGetResponse discarded;
        while (stream->Read(&discarded)) {
        }
        require(
            stream->Finish().error_code() == grpc::StatusCode::CANCELLED,
            "pre-open StreamGet slowloris was not cancelled by its watchdog");
    }
    const auto final_stats = wait_for_stats(
        [](const throwdb::server::GrpcAdmissionStats& stats) {
            return stats.active_scans == 0 && stats.admitted_scans == 4
                   && stats.scan_estimated_bytes == 0;
        },
        "new streaming RPCs did not release their admission leases");
    require(
        final_stats.rejected_scans == 0,
        "new streaming RPC watchdogs corrupted admission counters");
    server.shutdown();
}

void test_non_loopback_transport_is_fail_closed() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = 8;
    const std::array<std::uint64_t, 1> ids{10};
    const std::vector<std::byte> embeddings = rows(1, record_size, 9);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = dimension;
    builder_options.dtype = throwdb::DType::Float16;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, false);
    throwdb::server::SingleCollectionEngine engine(*collection, store);

    throwdb::server::GrpcServerOptions insecure;
    insecure.listen_address = "0.0.0.0:0";
    bool rejected = false;
    try {
        throwdb::server::GrpcServer server(engine, {}, insecure);
    } catch (const throwdb::InvalidArgumentError& error) {
        rejected = std::string(error.what()).find("non-loopback")
                   != std::string::npos;
    }
    require(
        rejected,
        "gRPC server accepted an implicit insecure non-loopback endpoint");

    insecure.listen_address = "localhost:0";
    rejected = false;
    try {
        throwdb::server::GrpcServer server(engine, {}, insecure);
    } catch (const throwdb::InvalidArgumentError& error) {
        rejected = std::string(error.what()).find("non-loopback")
                   != std::string::npos;
    }
    require(
        rejected,
        "gRPC server trusted a hostname as plaintext loopback");

    insecure.listen_address = "[::1]suffix:0";
    rejected = false;
    try {
        throwdb::server::GrpcServer server(engine, {}, insecure);
    } catch (const throwdb::InvalidArgumentError& error) {
        rejected = std::string(error.what()).find("strict HOST:PORT")
                   != std::string::npos;
    }
    require(rejected, "gRPC server accepted a malformed IPv6 endpoint");

    insecure.listen_address = "0.0.0.0:0";
    insecure.allow_insecure_non_loopback = true;
    throwdb::server::GrpcServer explicitly_allowed(engine, {}, insecure);

    throwdb::server::GrpcServerOptions incomplete_tls;
    incomplete_tls.tls_certificate_path = "server.pem";
    rejected = false;
    try {
        throwdb::server::GrpcServer server(engine, {}, incomplete_tls);
    } catch (const throwdb::InvalidArgumentError& error) {
        rejected = std::string(error.what()).find("--tls-cert")
                   != std::string::npos;
    }
    require(
        rejected,
        "gRPC server accepted a certificate without a private key");

    throwdb::server::GrpcServerOptions mutual_tls;
    mutual_tls.listen_address = "0.0.0.0:0";
    mutual_tls.tls_certificate_path = "server.pem";
    mutual_tls.tls_private_key_path = "server.key";
    mutual_tls.tls_client_ca_path = "clients.pem";
    throwdb::server::GrpcServer mtls_allowed(engine, {}, mutual_tls);

    throwdb::server::GrpcServer health_server(engine);
    const auto endpoint = health_server.start();
    grpc::ChannelArguments channel_options;
    channel_options.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    auto channel = grpc::CreateCustomChannel(
        endpoint.to_string(),
        grpc::InsecureChannelCredentials(),
        channel_options);
    require(
        channel->WaitForConnected(
            std::chrono::system_clock::now() + std::chrono::seconds(5)),
        "terminal-health test channel did not become ready");
    auto stub = wire::ThrowDB::NewStub(channel);
    health_server.report_terminal_fault();
    require(
        health_server.terminal_faulted(),
        "terminal storage fault was not exposed to the process lifecycle");
    const throwdb::server::GrpcAdmissionStats terminal_stats =
        health_server.admission_stats();
    require(
        terminal_stats.admission_closed && !terminal_stats.draining
            && terminal_stats.terminal_faulted,
        "public gRPC admission statistics did not expose terminal fault");
    grpc::ClientContext describe_context;
    wire::DescribeRequest describe_request;
    wire::DescribeResponse describe_response;
    const grpc::Status describe_status = stub->Describe(
        &describe_context, describe_request, &describe_response);
    require(
        describe_status.ok()
            && describe_response.admission_stats().admission_closed()
            && !describe_response.admission_stats().draining()
            && describe_response.admission_stats().terminal_faulted(),
        "gRPC Describe did not expose terminal admission state");
    grpc::ClientContext context;
    wire::HealthRequest request;
    wire::HealthResponse response;
    const grpc::Status health_status = stub->Health(
        &context, request, &response);
    require(
        health_status.ok()
            && response.status()
                   == wire::HealthResponse::SERVING_STATUS_NOT_SERVING,
        "typed Health remained SERVING after a terminal storage fault");
    const auto [standard_status, standard_response] = generic_unary(
        channel, "/grpc.health.v1.Health/Check", std::string{});
    require(
        standard_status.ok()
            && standard_response == std::string{"\x08\x02", 2},
        "standard Health remained SERVING after a terminal storage fault");
    grpc::ClientContext failed_get_context;
    wire::GetRequest failed_get_request;
    set_ids(*failed_get_request.mutable_ids(), ids);
    wire::GetResponse failed_get_response;
    const grpc::Status failed_get_status = stub->Get(
        &failed_get_context, failed_get_request, &failed_get_response);
    require(
        failed_get_status.error_code() == grpc::StatusCode::UNAVAILABLE,
        "gRPC data plane accepted a read after a terminal storage fault");
    health_server.shutdown();
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

void test_get_is_rejected_before_native_read_or_allocation() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = 8;
    const std::array<std::uint64_t, 1> ids{10};
    const std::vector<std::byte> embeddings = rows(1, record_size, 9);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = dimension;
    builder_options.dtype = throwdb::DType::Float16;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, false);
    throwdb::server::SingleCollectionEngine engine(*collection, store);

    const auto call_get = [&](const throwdb::server::ServiceLimits limits,
                              const throwdb::server::GrpcServerOptions options,
                              const std::span<const std::uint64_t> requested) {
        throwdb::server::GrpcServer server(engine, limits, options);
        const auto endpoint = server.start();
        grpc::ChannelArguments channel_options;
        channel_options.SetMaxSendMessageSize(1 << 20);
        channel_options.SetMaxReceiveMessageSize(1 << 20);
        channel_options.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
        auto channel = grpc::CreateCustomChannel(
            endpoint.to_string(), grpc::InsecureChannelCredentials(),
            channel_options);
        require(
            channel->WaitForConnected(
                std::chrono::system_clock::now()
                + std::chrono::seconds(5)),
            "preflight test channel did not become ready");
        auto stub = wire::ThrowDB::NewStub(channel);
        wire::GetRequest request;
        set_ids(*request.mutable_ids(), requested);
        wire::GetResponse response;
        grpc::ClientContext context;
        const grpc::Status status = stub->Get(
            &context, request, &response);
        server.shutdown();
        return status;
    };

    throwdb::server::ServiceLimits response_limits;
    // float16[4] serializes to an 18-byte GetResponse. A missing ID would
    // produce NOT_FOUND if the native lookup happened before the preflight.
    response_limits.max_response_bytes = 17;
    const std::array<std::uint64_t, 1> missing_id{999};
    const grpc::Status response_status = call_get(
        response_limits, throwdb::server::GrpcServerOptions{}, missing_id);
    require(
        response_status.error_code()
                == grpc::StatusCode::RESOURCE_EXHAUSTED
            && response_status.error_message().find("before native read")
                   != std::string::npos,
        "Get response budget was not checked before native lookup");

    throwdb::server::ServiceLimits admission_limits;
    throwdb::server::GrpcServerOptions admission_options;
    admission_options.max_inflight_bytes = 1;
    const std::array<std::uint64_t, 1> present_id{10};
    const grpc::Status admission_status = call_get(
        admission_limits, admission_options, present_id);
    require(
        admission_status.error_code()
                == grpc::StatusCode::RESOURCE_EXHAUSTED
            && admission_status.error_message().find("admission exhausted")
                   != std::string::npos,
        "Get peak-memory estimate bypassed server-wide admission");

    // Tiny rows make the output payload look cheap, but the native lookup and
    // physical-read planner retain several O(ID-count) vectors. This request
    // would fit a payload-only estimate and must still be rejected before the
    // native lookup starts.
    std::vector<std::uint64_t> repeated_ids(10'000, 10);
    throwdb::server::GrpcServerOptions scratch_options;
    scratch_options.max_inflight_bytes = 512ULL << 10;
    const grpc::Status scratch_status = call_get(
        admission_limits, scratch_options, repeated_ids);
    require(
        scratch_status.error_code()
                == grpc::StatusCode::RESOURCE_EXHAUSTED
            && scratch_status.error_message().find("admission exhausted")
                   != std::string::npos,
        "Get native planner scratch bypassed server-wide admission");
}

void test_streaming_negotiation_accounts_for_snapshot_and_backend_buffers() {
    TemporaryDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 1024;
    constexpr std::uint64_t record_size = dimension * 2;
    constexpr std::size_t row_count = 1024;
    std::vector<std::uint64_t> ids(row_count);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = index + 1;
    }
    std::vector<std::byte> embeddings(row_count * record_size);

    throwdb::NativeBuilderOptions builder_options;
    builder_options.format_version = 7;
    builder_options.dimension = dimension;
    builder_options.dtype = throwdb::DType::Float16;
    builder_options.segment_size_bytes = 4ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(
        store, builder_options);
    builder->append(ids, embeddings);
    builder->finish();

    throwdb::ReaderOptions reader_options;
    reader_options.max_read_range_bytes = 1ULL << 20;
    reader_options.max_gap_bytes = 0;
    reader_options.io_workers = 4;
    auto collection = throwdb::NativeCollection::open(
        store, reader_options, true);
    // Keep every base ID in the committed overlay. At 96 estimated bytes per
    // entry this snapshot is deliberately much larger than the slack after a
    // memory-only maximum-batch calculation. The server must open the view
    // with one-row headroom, measure the snapshot, and then reduce the batch;
    // choosing the maximum batch first would reject this otherwise valid RPC.
    collection->update(ids, embeddings);
    {
        auto snapshot_probe = collection->open_read_view(1);
        require(
            snapshot_probe->metadata().snapshot_bytes > (64ULL << 10),
            "streaming memory-negotiation fixture did not create a large "
            "overlay snapshot");
    }
    throwdb::server::EngineOptions engine_options;
    engine_options.reader = reader_options;
    throwdb::server::SingleCollectionEngine engine(
        *collection, store, engine_options);
    throwdb::server::ServiceLimits limits;
    limits.max_request_bytes =
        128 + row_count * sizeof(std::uint64_t);
    limits.max_response_bytes = 4ULL << 20;
    limits.max_scan_batch_rows = row_count;
    throwdb::server::GrpcServerOptions options;
    options.max_worker_threads = 8;
    options.max_concurrent_calls = 4;
    options.max_concurrent_scans = 1;
    options.max_scan_inflight_bytes = 120ULL << 10;
    options.max_grpc_memory_bytes = 8ULL << 20;
    throwdb::server::GrpcServer server(engine, limits, options);
    const auto endpoint = server.start();

    grpc::ChannelArguments channel_options;
    channel_options.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    channel_options.SetMaxReceiveMessageSize(4 << 20);
    auto channel = grpc::CreateCustomChannel(
        endpoint.to_string(), grpc::InsecureChannelCredentials(),
        channel_options);
    require(
        channel->WaitForConnected(
            std::chrono::system_clock::now() + std::chrono::seconds(5)),
        "ScanRecords memory-negotiation channel did not become ready");
    auto stub = wire::ThrowDB::NewStub(channel);
    const auto wait_for_scan_release = [&](const std::string& message) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        auto current = server.admission_stats();
        while ((current.active_scans != 0
                || current.scan_estimated_bytes != 0)
               && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            current = server.admission_stats();
        }
        require(
            current.active_scans == 0
                && current.scan_estimated_bytes == 0,
            message + "; active="
                + std::to_string(current.active_scans)
                + " bytes="
                + std::to_string(current.scan_estimated_bytes));
        return current;
    };

    wire::ScanIdsRequest ids_request;
    ids_request.set_batch_rows(row_count);
    ids_request.set_include_locations(true);
    grpc::ClientContext ids_context;
    auto ids_reader = stub->ScanIds(&ids_context, ids_request);
    wire::ScanIdsResponse ids_first;
    require(
        ids_reader->Read(&ids_first)
            && ids_first.ids().uint64_ids().count() > 0
            && ids_first.ids().uint64_ids().count() < row_count,
        "ScanIds negotiation did not preserve snapshot headroom while "
        "reducing its batch");
    ids_context.TryCancel();
    while (ids_reader->Read(&ids_first)) {
    }
    static_cast<void>(ids_reader->Finish());
    static_cast<void>(wait_for_scan_release(
        "ScanIds memory-negotiation call retained admission"));

    grpc::ClientContext get_context;
    auto get_stream = stub->StreamGet(&get_context);
    wire::StreamGetRequest get_open;
    get_open.mutable_open()->set_prefix_dimensions(1);
    get_open.mutable_open()->set_max_batch_rows(row_count);
    get_open.mutable_open()->set_max_batch_payload_bytes(64ULL << 10);
    wire::StreamGetResponse get_opened;
    require(
        get_stream->Write(get_open)
            && get_stream->Read(&get_opened)
            && get_opened.payload_case()
                   == wire::StreamGetResponse::kOpened
            && get_opened.opened().max_batch_rows() > 0
            && get_opened.opened().max_batch_rows() < row_count,
        "StreamGet negotiation did not preserve snapshot headroom while "
        "reducing its batch");
    get_context.TryCancel();
    while (get_stream->Read(&get_opened)) {
    }
    static_cast<void>(get_stream->Finish());
    static_cast<void>(wait_for_scan_release(
        "StreamGet memory-negotiation call retained admission"));

    wire::ScanRecordsRequest request;
    request.set_batch_rows(row_count);
    request.set_max_batch_payload_bytes(64ULL << 10);
    request.set_prefix_dimensions(1);
    grpc::ClientContext context;
    auto reader = stub->ScanRecords(&context, request);
    wire::ScanRecordsResponse first;
    require(
        reader->Read(&first)
            && first.max_batch_rows() > 0
            && first.max_batch_rows() < row_count
            && first.embeddings().dimension() == 1,
        "ScanRecords negotiation ignored full-row backend buffers for a "
        "prefix scan");
    context.TryCancel();
    while (reader->Read(&first)) {
    }
    static_cast<void>(reader->Finish());
    static_cast<void>(wait_for_scan_release(
        "ScanRecords memory-negotiation call retained admission"));
    server.shutdown();
}

}  // namespace

int main() {
    try {
        // Fork/exec lifecycle coverage must run before this process initializes
        // any gRPC worker threads.
        test_daemon_terminal_exit_and_hard_shutdown_watchdog();
        test_grpc_rejects_legacy_storage_format();
        test_non_loopback_transport_is_fail_closed();
        test_get_is_rejected_before_native_read_or_allocation();
        test_streaming_negotiation_accounts_for_snapshot_and_backend_buffers();
        test_stalled_scan_is_cancelled_and_releases_admission();
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
