#include "throwdb/server/grpc_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <grpc/impl/channel_arg_names.h>
#if defined(THROWDB_GRPC_ENABLE_REFLECTION)
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#endif
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/resource_quota.h>

#include "throwdb/errors.hpp"
#include "throwdb/server/collection_engine.hpp"
#include "throwdb/v1/throwdb.grpc.pb.h"

namespace throwdb::server {
namespace {

namespace wire = ::throwdb::v1;

inline constexpr std::uint32_t kProtocolMajor = 1;
inline constexpr std::uint32_t kProtocolMinor = 0;
inline constexpr std::uint64_t kScanEnvelopeAllowance = 512;

class UnsupportedIdTypeError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] bool checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result) noexcept {
    if (left != 0
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

[[nodiscard]] std::string bounded_message(
    const std::string_view message,
    const ServiceLimits& limits) {
    return std::string(message.substr(0, std::min<std::size_t>(
        message.size(), limits.max_error_message_bytes)));
}

[[nodiscard]] grpc::Status exception_status(
    const ServiceLimits& limits) noexcept {
    try {
        throw;
    } catch (const MissingIdError& error) {
        return {grpc::StatusCode::NOT_FOUND,
                bounded_message(error.what(), limits)};
    } catch (const ExistingIdError& error) {
        return {grpc::StatusCode::ALREADY_EXISTS,
                bounded_message(error.what(), limits)};
    } catch (const IdempotencyConflictError& error) {
        return {grpc::StatusCode::ALREADY_EXISTS,
                bounded_message(error.what(), limits)};
    } catch (const PendingOperationError& error) {
        return {grpc::StatusCode::ABORTED,
                bounded_message(error.what(), limits)};
    } catch (const UnsupportedIdTypeError& error) {
        return {grpc::StatusCode::UNIMPLEMENTED,
                bounded_message(error.what(), limits)};
    } catch (const DuplicateIdError& error) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                bounded_message(error.what(), limits)};
    } catch (const InvalidArgumentError& error) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                bounded_message(error.what(), limits)};
    } catch (const CorruptionError& error) {
        return {grpc::StatusCode::DATA_LOSS,
                bounded_message(error.what(), limits)};
    } catch (const std::length_error& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                bounded_message(error.what(), limits)};
    } catch (const std::bad_alloc& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                bounded_message(error.what(), limits)};
    } catch (const std::invalid_argument& error) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                bounded_message(error.what(), limits)};
    } catch (const std::exception& error) {
        return {grpc::StatusCode::INTERNAL,
                bounded_message(error.what(), limits)};
    } catch (...) {
        return {grpc::StatusCode::INTERNAL,
                "unknown ThrowDB server failure"};
    }
}

template <typename Message>
void require_request_budget(
    const Message& request,
    const ServiceLimits& limits) {
    if (request.ByteSizeLong() > limits.max_request_bytes) {
        throw std::length_error("gRPC request exceeds max_request_bytes");
    }
}

template <typename Message>
void require_response_budget(
    const Message& response,
    const ServiceLimits& limits) {
    if (response.ByteSizeLong() > limits.max_response_bytes) {
        throw std::length_error("gRPC response exceeds max_response_bytes");
    }
}

[[nodiscard]] std::uint64_t read_u64_le(
    const char* bytes) noexcept {
    std::uint64_t value = 0;
    for (unsigned int index = 0; index != 8; ++index) {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(bytes[index]))
                 << (index * 8U);
    }
    return value;
}

void append_u32_le(std::string& output, const std::uint32_t value) {
    for (unsigned int index = 0; index != 4; ++index) {
        output.push_back(static_cast<char>(value >> (index * 8U)));
    }
}

void append_u64_le(std::string& output, const std::uint64_t value) {
    for (unsigned int index = 0; index != 8; ++index) {
        output.push_back(static_cast<char>(value >> (index * 8U)));
    }
}

[[nodiscard]] std::vector<std::uint64_t> decode_ids(
    const wire::IdBatch& batch,
    const ServiceLimits& limits) {
    switch (batch.encoding_case()) {
        case wire::IdBatch::kUint64Ids: {
            const wire::UInt64Ids& encoded = batch.uint64_ids();
            const std::uint64_t count = encoded.count();
            if (count > limits.max_ids_per_request) {
                throw std::length_error(
                    "ID batch exceeds max_ids_per_request");
            }
            std::uint64_t expected_bytes = 0;
            if (!checked_multiply(count, sizeof(std::uint64_t),
                                  expected_bytes)
                || expected_bytes > std::numeric_limits<std::size_t>::max()
                || encoded.values_le().size()
                       != static_cast<std::size_t>(expected_bytes)) {
                throw InvalidArgumentError(
                    "UInt64Ids values_le length does not equal count * 8");
            }
            std::vector<std::uint64_t> result(
                static_cast<std::size_t>(count));
            const char* source = encoded.values_le().data();
            for (std::size_t index = 0; index != result.size(); ++index) {
                result[index] = read_u64_le(source + index * 8);
            }
            return result;
        }
        case wire::IdBatch::kUtf8Ids:
            throw UnsupportedIdTypeError(
                "UTF-8 IDs are reserved but not supported by format 7");
        case wire::IdBatch::ENCODING_NOT_SET:
            throw InvalidArgumentError("IdBatch encoding is required");
    }
    throw InvalidArgumentError("IdBatch contains an unknown encoding");
}

void encode_ids(
    const std::span<const std::uint64_t> ids,
    wire::IdBatch& output) {
    wire::UInt64Ids* encoded = output.mutable_uint64_ids();
    encoded->set_count(ids.size());
    std::string values;
    if (ids.size() > std::numeric_limits<std::size_t>::max() / 8) {
        throw std::length_error("ID response does not fit address space");
    }
    values.reserve(ids.size() * 8);
    for (const std::uint64_t id : ids) {
        append_u64_le(values, id);
    }
    *encoded->mutable_values_le() = std::move(values);
}

[[nodiscard]] wire::EmbeddingDType public_dtype(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return wire::EMBEDDING_DTYPE_FLOAT16;
        case DType::Float32:
            return wire::EMBEDDING_DTYPE_FLOAT32;
    }
    throw CorruptionError("collection has an unknown embedding dtype");
}

[[nodiscard]] DType native_dtype(const wire::EmbeddingDType dtype) {
    switch (dtype) {
        case wire::EMBEDDING_DTYPE_FLOAT16:
            return DType::Float16;
        case wire::EMBEDDING_DTYPE_FLOAT32:
            return DType::Float32;
        case wire::EMBEDDING_DTYPE_UNSPECIFIED:
            break;
        default:
            // Protobuf generates INT_MIN/INT_MAX enum sentinels and permits
            // unknown numeric values on the wire. They are validation
            // failures, not additional ThrowDB dtypes.
            break;
    }
    throw InvalidArgumentError("embedding dtype is invalid");
}

[[nodiscard]] std::span<const std::byte> bytes_view(
    const std::string& bytes) noexcept {
    return {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()};
}

void set_embedding_response(
    const GetResult& result,
    wire::GetResponse& response) {
    wire::EmbeddingBatch* embeddings = response.mutable_embeddings();
    embeddings->set_dtype(public_dtype(result.dtype));
    embeddings->set_dimension(result.dimension);
    embeddings->set_row_count(result.row_count);
    if (result.embeddings.empty()) {
        embeddings->clear_values_le();
    } else {
        embeddings->set_values_le(
            reinterpret_cast<const char*>(result.embeddings.data()),
            result.embeddings.size());
    }
}

[[nodiscard]] OperationId operation_id(const wire::OperationId& value) {
    const OperationId result{value.high(), value.low()};
    if (result.empty()) {
        throw InvalidArgumentError("operation_id must be nonzero");
    }
    return result;
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
random_scan_token() {
    std::random_device random;
    const auto next = [&random]() {
        return (static_cast<std::uint64_t>(random()) << 32U)
               ^ static_cast<std::uint64_t>(random());
    };
    std::uint64_t high = next();
    std::uint64_t low = next();
    if (high == 0 && low == 0) {
        low = 1;
    }
    return {high, low};
}

class GrpcCollectionService final : public wire::ThrowDB::Service {
public:
    GrpcCollectionService(
        SingleCollectionEngine& engine,
        const ServiceLimits limits)
        : engine_(engine), limits_(limits) {}

    grpc::Status Health(
        grpc::ServerContext*,
        const wire::HealthRequest* request,
        wire::HealthResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            response->set_status(
                wire::HealthResponse::SERVING_STATUS_SERVING);
            response->set_protocol_major(kProtocolMajor);
            response->set_protocol_minor(kProtocolMinor);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

    grpc::Status Describe(
        grpc::ServerContext*,
        const wire::DescribeRequest* request,
        wire::DescribeResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            const CollectionDescription description = engine_.describe();
            response->set_format_version(description.format_version);
            response->set_record_size(description.record_size);
            response->set_count(description.count);
            wire::CollectionSchema* schema = response->mutable_schema();
            schema->set_id_type(wire::ID_TYPE_UINT64);
            schema->set_embedding_dtype(public_dtype(description.dtype));
            schema->set_dimension(description.dimension);
            wire::Capabilities* capabilities =
                response->mutable_capabilities();
            capabilities->set_writable(description.writable);
            capabilities->set_prefix_get(true);
            capabilities->set_scan_locations(true);
            capabilities->set_durable_idempotency(
                description.durable_idempotency);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

    grpc::Status Get(
        grpc::ServerContext* context,
        const wire::GetRequest* request,
        wire::GetResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED, "Get was cancelled"};
            }
            const std::vector<std::uint64_t> ids =
                decode_ids(request->ids(), limits_);
            const GetResult result = engine_.get(
                ids, request->prefix_dimensions());
            set_embedding_response(result, *response);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

    grpc::Status Insert(
        grpc::ServerContext* context,
        const wire::MutationRequest* request,
        wire::MutationResponse* response) override {
        return mutate(
            context, *request, *response, MutationKind::Insert);
    }

    grpc::Status Update(
        grpc::ServerContext* context,
        const wire::MutationRequest* request,
        wire::MutationResponse* response) override {
        return mutate(
            context, *request, *response, MutationKind::Update);
    }

    grpc::Status Upsert(
        grpc::ServerContext* context,
        const wire::MutationRequest* request,
        wire::MutationResponse* response) override {
        return mutate(
            context, *request, *response, MutationKind::Upsert);
    }

    grpc::Status Delete(
        grpc::ServerContext* context,
        const wire::DeleteRequest* request,
        wire::DeleteResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            if (!engine_.describe().writable) {
                return {grpc::StatusCode::FAILED_PRECONDITION,
                        "collection was opened read-only"};
            }
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED,
                        "Delete was cancelled before admission"};
            }
            const std::vector<std::uint64_t> ids =
                decode_ids(request->ids(), limits_);
            const MutationResult result = engine_.mutate(
                MutationKind::Delete, operation_id(request->operation_id()),
                ids);
            response->set_committed_sequence(result.commit_sequence);
            response->set_affected_count(result.row_count);
            if (result.deleted.empty()) {
                response->clear_deleted_mask();
            } else {
                response->set_deleted_mask(
                    reinterpret_cast<const char*>(result.deleted.data()),
                    result.deleted.size());
            }
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

    grpc::Status ScanIds(
        grpc::ServerContext* context,
        const wire::ScanIdsRequest* request,
        grpc::ServerWriter<wire::ScanIdsResponse>* writer) override {
        try {
            require_request_budget(*request, limits_);
            if (request->batch_rows() == 0
                || request->batch_rows() > limits_.max_scan_batch_rows) {
                throw InvalidArgumentError(
                    "ScanIds batch_rows is out of range");
            }
            const std::uint64_t row_bytes =
                request->include_locations() ? 16 : 8;
            if (limits_.max_response_bytes
                <= kScanEnvelopeAllowance + row_bytes) {
                throw std::length_error(
                    "max_response_bytes cannot fit one ScanIds row");
            }
            const std::uint64_t response_rows =
                (limits_.max_response_bytes - kScanEnvelopeAllowance)
                / row_bytes;
            const std::uint64_t batch_rows = std::min<std::uint64_t>(
                request->batch_rows(), response_rows);
            if (batch_rows == 0
                || batch_rows > std::numeric_limits<std::size_t>::max()) {
                throw std::length_error(
                    "ScanIds batch does not fit this process");
            }

            auto scan = engine_.open_scan();
            const NativeIdScanMetadata metadata = scan->metadata();
            const auto [token_high, token_low] = random_scan_token();
            std::vector<std::uint64_t> ids(
                static_cast<std::size_t>(batch_rows));
            std::vector<NativeIdLocation> locations;
            if (request->include_locations()) {
                locations.resize(ids.size());
            }

            std::uint64_t emitted = 0;
            do {
                if (context->IsCancelled()) {
                    return {grpc::StatusCode::CANCELLED,
                            "ScanIds was cancelled"};
                }
                const std::uint64_t count =
                    scan->read_into(ids, locations);
                if (count == 0 && !scan->done()) {
                    throw CorruptionError(
                        "native ID scan made no progress");
                }
                if (count > ids.size()
                    || emitted > std::numeric_limits<std::uint64_t>::max()
                                     - count) {
                    throw CorruptionError(
                        "native ID scan returned an invalid row count");
                }
                const bool done = scan->done();
                if ((done && emitted + count != metadata.live_count)
                    || (!done
                        && emitted + count >= metadata.live_count)) {
                    throw CorruptionError(
                        "native ID scan disagrees with snapshot live_count");
                }

                wire::ScanIdsResponse response;
                wire::IdScanSnapshot* snapshot =
                    response.mutable_snapshot();
                snapshot->mutable_collection_id()->set_low(
                    metadata.collection_id_low);
                snapshot->mutable_collection_id()->set_high(
                    metadata.collection_id_high);
                snapshot->set_catalog_generation(
                    metadata.catalog_generation);
                snapshot->set_committed_sequence(
                    metadata.committed_sequence);
                snapshot->set_count(metadata.live_count);
                snapshot->mutable_token()->set_high(token_high);
                snapshot->mutable_token()->set_low(token_low);
                response.set_emitted_offset(emitted);
                encode_ids(
                    std::span<const std::uint64_t>(
                        ids.data(), static_cast<std::size_t>(count)),
                    *response.mutable_ids());
                if (request->include_locations()) {
                    response.set_location_encoding(
                        wire::LOCATION_ENCODING_SEGMENT_ROW_U32_LE);
                    std::string encoded_locations;
                    encoded_locations.reserve(
                        static_cast<std::size_t>(count) * 8);
                    for (std::size_t index = 0;
                         index != static_cast<std::size_t>(count); ++index) {
                        append_u32_le(
                            encoded_locations, locations[index].segment_id);
                        append_u32_le(
                            encoded_locations, locations[index].row_index);
                    }
                    *response.mutable_locations_le() =
                        std::move(encoded_locations);
                } else {
                    response.set_location_encoding(
                        wire::LOCATION_ENCODING_UNSPECIFIED);
                }
                require_response_budget(response, limits_);
                emitted += count;
                if (!writer->Write(response)) {
                    return {grpc::StatusCode::CANCELLED,
                            "ScanIds client stopped receiving"};
                }
                if (done) {
                    break;
                }
            } while (true);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

private:
    grpc::Status mutate(
        grpc::ServerContext* context,
        const wire::MutationRequest& request,
        wire::MutationResponse& response,
        const MutationKind kind) {
        try {
            require_request_budget(request, limits_);
            const CollectionDescription description = engine_.describe();
            if (!description.writable) {
                return {grpc::StatusCode::FAILED_PRECONDITION,
                        "collection was opened read-only"};
            }
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED,
                        "mutation was cancelled before admission"};
            }
            const std::vector<std::uint64_t> ids =
                decode_ids(request.ids(), limits_);
            const wire::EmbeddingBatch& embeddings = request.embeddings();
            if (embeddings.dimension() != description.dimension
                || native_dtype(embeddings.dtype()) != description.dtype
                || embeddings.row_count() != ids.size()) {
                throw InvalidArgumentError(
                    "mutation embedding schema does not match collection");
            }
            std::uint64_t expected_bytes = 0;
            if (!checked_multiply(ids.size(), description.record_size,
                                  expected_bytes)
                || expected_bytes > std::numeric_limits<std::size_t>::max()
                || embeddings.values_le().size()
                       != static_cast<std::size_t>(expected_bytes)) {
                throw InvalidArgumentError(
                    "mutation values_le length does not match row geometry");
            }
            const MutationResult result = engine_.mutate(
                kind, operation_id(request.operation_id()), ids,
                bytes_view(embeddings.values_le()));
            response.set_committed_sequence(result.commit_sequence);
            response.set_affected_count(result.row_count);
            require_response_budget(response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_);
        }
    }

    SingleCollectionEngine& engine_;
    ServiceLimits limits_;
};

[[nodiscard]] int grpc_message_limit(
    const std::uint64_t bytes,
    const std::string_view name) {
    if (bytes == 0
        || bytes > static_cast<std::uint64_t>(
                       std::numeric_limits<int>::max())) {
        throw InvalidArgumentError(
            std::string(name)
            + " must be between 1 and INT_MAX for gRPC");
    }
    return static_cast<int>(bytes);
}

[[nodiscard]] std::string selected_endpoint(
    const std::string& requested,
    const int selected_port) {
    if (selected_port <= 0) {
        throw StorageError("gRPC did not select a listening port");
    }
    const std::size_t colon = requested.rfind(':');
    if (colon == std::string::npos) {
        return requested;
    }
    return requested.substr(0, colon + 1) + std::to_string(selected_port);
}

}  // namespace

class GrpcServer::Impl {
public:
    Impl(
        SingleCollectionEngine& engine,
        const ServiceLimits limits,
        GrpcServerOptions options)
        : service_(engine, limits),
          limits_(limits),
          options_(std::move(options)) {
        const CollectionDescription description = engine.describe();
        if (description.format_version != 7) {
            throw InvalidArgumentError(
                "throwdbd supports storage format 7 only; rebuild the "
                "collection in the current format before starting the "
                "server");
        }
        if (options_.listen_address.empty()
            || options_.max_worker_threads == 0
            || options_.shutdown_grace_millis == 0
            || options_.keepalive_time_millis == 0
            || options_.keepalive_timeout_millis == 0
            || options_.minimum_ping_interval_millis == 0) {
            throw InvalidArgumentError(
                "gRPC server options must be greater than zero");
        }
        const auto fits_grpc_int = [](const std::uint32_t value) {
            return value
                   <= static_cast<std::uint32_t>(
                       std::numeric_limits<int>::max());
        };
        if (!fits_grpc_int(options_.max_worker_threads)
            || !fits_grpc_int(options_.keepalive_time_millis)
            || !fits_grpc_int(options_.keepalive_timeout_millis)
            || !fits_grpc_int(options_.minimum_ping_interval_millis)) {
            throw InvalidArgumentError(
                "gRPC thread and keepalive options must fit int");
        }
        static_cast<void>(grpc_message_limit(
            limits_.max_request_bytes, "max_request_bytes"));
        static_cast<void>(grpc_message_limit(
            limits_.max_response_bytes, "max_response_bytes"));
    }

    [[nodiscard]] GrpcBoundEndpoint start() {
        const std::lock_guard lock(state_mutex_);
        if (server_ != nullptr) {
            throw std::logic_error("gRPC server is already started");
        }

        grpc::EnableDefaultHealthCheckService(true);
#if defined(THROWDB_GRPC_ENABLE_REFLECTION)
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
#endif
        grpc::ServerBuilder builder;
        builder.SetMaxReceiveMessageSize(grpc_message_limit(
            limits_.max_request_bytes, "max_request_bytes"));
        builder.SetMaxSendMessageSize(grpc_message_limit(
            limits_.max_response_bytes, "max_response_bytes"));
        builder.AddChannelArgument(
            GRPC_ARG_KEEPALIVE_TIME_MS,
            static_cast<int>(options_.keepalive_time_millis));
        builder.AddChannelArgument(
            GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
            static_cast<int>(options_.keepalive_timeout_millis));
        builder.AddChannelArgument(
            GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
            options_.permit_keepalive_without_calls ? 1 : 0);
        builder.AddChannelArgument(
            GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
            static_cast<int>(options_.minimum_ping_interval_millis));
        grpc::ResourceQuota quota;
        quota.SetMaxThreads(static_cast<int>(options_.max_worker_threads));
        builder.SetResourceQuota(quota);
        builder.RegisterService(&service_);

        int port = 0;
        builder.AddListeningPort(
            options_.listen_address,
            grpc::InsecureServerCredentials(), &port);
        server_ = builder.BuildAndStart();
        if (server_ == nullptr) {
            throw StorageError("could not start ThrowDB gRPC server");
        }
        endpoint_.address = selected_endpoint(options_.listen_address, port);
        started_.store(true, std::memory_order_release);
        return endpoint_;
    }

    void serve(const std::function<bool()>& should_stop) {
        if (!should_stop) {
            throw InvalidArgumentError("gRPC stop predicate is empty");
        }
        {
            const std::lock_guard lock(state_mutex_);
            if (server_ == nullptr) {
                throw std::logic_error(
                    "gRPC server must be started before serve");
            }
        }
        while (!should_stop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        shutdown();
    }

    void shutdown() {
        grpc::Server* server = nullptr;
        {
            const std::lock_guard lock(state_mutex_);
            if (server_ == nullptr || shutdown_started_) {
                return;
            }
            shutdown_started_ = true;
            server = server_.get();
        }
        started_.store(false, std::memory_order_release);
        const auto deadline = std::chrono::system_clock::now()
                              + std::chrono::milliseconds(
                                  options_.shutdown_grace_millis);
        server->Shutdown(deadline);
        server->Wait();
    }

    [[nodiscard]] bool started() const noexcept {
        return started_.load(std::memory_order_acquire);
    }

private:
    GrpcCollectionService service_;
    ServiceLimits limits_;
    GrpcServerOptions options_;
    std::unique_ptr<grpc::Server> server_;
    GrpcBoundEndpoint endpoint_;
    mutable std::mutex state_mutex_;
    std::atomic<bool> started_{false};
    bool shutdown_started_{false};
};

GrpcServer::GrpcServer(
    SingleCollectionEngine& engine,
    const ServiceLimits limits,
    GrpcServerOptions options)
    : impl_(std::make_unique<Impl>(
          engine, limits, std::move(options))) {}

GrpcServer::~GrpcServer() {
    if (impl_ != nullptr) {
        impl_->shutdown();
    }
}

GrpcBoundEndpoint GrpcServer::start() { return impl_->start(); }

void GrpcServer::serve(const std::function<bool()>& should_stop) {
    impl_->serve(should_stop);
}

void GrpcServer::shutdown() { impl_->shutdown(); }

bool GrpcServer::started() const noexcept { return impl_->started(); }

}  // namespace throwdb::server
