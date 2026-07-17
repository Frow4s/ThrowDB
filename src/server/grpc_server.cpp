#include "throwdb/server/grpc_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
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
#include "grpc_admission.hpp"

namespace throwdb::server {
namespace {

namespace wire = ::throwdb::v1;

inline constexpr std::uint32_t kProtocolMajor = 1;
inline constexpr std::uint32_t kProtocolMinor = 2;
inline constexpr std::uint64_t kScanEnvelopeAllowance = 512;
inline constexpr std::uint64_t kMutationResponseUpperBound = 32;
inline constexpr std::uint64_t kDeleteResponseEnvelopeAllowance = 64;
// Per-row application-memory estimates deliberately include native planning
// containers, not only protobuf payloads. Physical reads can simultaneously
// retain PhysicalReadRequest, sorted Request, PlannedRange, scheduler work,
// and io_uring operation metadata. Keep this bound conservative when those
// private structures evolve.
inline constexpr std::uint64_t kGetNativeScratchBytesPerRow = 256;
inline constexpr std::uint64_t kMutationWorkingBytesPerRow = 256;
inline constexpr std::uint64_t kScanWorkingBytesPerRow = 64;

enum class HealthPhase : std::uint8_t {
    Starting,
    Serving,
    Draining,
    TerminalFault,
};

class GrpcHealthState final {
public:
    [[nodiscard]] bool serving() const noexcept {
        return phase_.load(std::memory_order_acquire)
               == HealthPhase::Serving;
    }

    [[nodiscard]] bool terminal_faulted() const noexcept {
        return phase_.load(std::memory_order_acquire)
               == HealthPhase::TerminalFault;
    }

    [[nodiscard]] bool draining() const noexcept {
        return phase_.load(std::memory_order_acquire)
               == HealthPhase::Draining;
    }

    void attach(grpc::HealthCheckServiceInterface* const standard) {
        const std::lock_guard lock(mutex_);
        standard_ = standard;
        update_standard_locked();
    }

    void attach_admission(
        detail::GrpcAdmissionController* const admission) noexcept {
        admission_.store(admission, std::memory_order_release);
    }

    void mark_serving() {
        HealthPhase expected = HealthPhase::Starting;
        if (!phase_.compare_exchange_strong(
                expected, HealthPhase::Serving,
                std::memory_order_acq_rel)) {
            return;
        }
        update_standard();
    }

    void begin_draining() noexcept {
        HealthPhase current = phase_.load(std::memory_order_acquire);
        while (current == HealthPhase::Starting
               || current == HealthPhase::Serving) {
            if (phase_.compare_exchange_weak(
                    current, HealthPhase::Draining,
                    std::memory_order_acq_rel)) {
                close_admission_noexcept();
                update_standard_noexcept();
                return;
            }
        }
    }

    void mark_terminal_fault() noexcept {
        phase_.store(HealthPhase::TerminalFault, std::memory_order_release);
        close_admission_noexcept();
        update_standard_noexcept();
    }

private:
    void close_admission_noexcept() noexcept {
        if (auto* const admission =
                admission_.load(std::memory_order_acquire);
            admission != nullptr) {
            admission->close();
        }
    }

    void update_standard() {
        const std::lock_guard lock(mutex_);
        update_standard_locked();
    }

    void update_standard_noexcept() noexcept {
        try {
            update_standard();
        } catch (...) {
            // Health reporting must never replace the storage error that made
            // the process unready, nor make signal-driven shutdown throw.
        }
    }

    void update_standard_locked() {
        if (standard_ == nullptr) {
            return;
        }
        const bool ready = serving();
        standard_->SetServingStatus("", ready);
        standard_->SetServingStatus("throwdb.v1.ThrowDB", ready);
    }

    std::atomic<HealthPhase> phase_{HealthPhase::Starting};
    std::mutex mutex_;
    grpc::HealthCheckServiceInterface* standard_ = nullptr;
    std::atomic<detail::GrpcAdmissionController*> admission_{nullptr};
};

[[nodiscard]] std::optional<grpc::Status> scan_expired_status(
    const detail::GrpcScanExpiry expiry) {
    switch (expiry) {
        case detail::GrpcScanExpiry::None:
            return std::nullopt;
        case detail::GrpcScanExpiry::MaximumDuration:
            return grpc::Status{
                grpc::StatusCode::DEADLINE_EXCEEDED,
                "ScanIds exceeded the server maximum stream duration"};
        case detail::GrpcScanExpiry::IdleProgress:
            return grpc::Status{
                grpc::StatusCode::DEADLINE_EXCEEDED,
                "ScanIds made no transport progress before the server idle "
                "timeout"};
    }
    return std::nullopt;
}

class UnsupportedIdTypeError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ServerUnavailableError final : public std::runtime_error {
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

[[nodiscard]] bool checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

[[nodiscard]] std::uint64_t protobuf_varint_size(
    std::uint64_t value) noexcept {
    std::uint64_t bytes = 1;
    while (value >= 0x80U) {
        value >>= 7U;
        ++bytes;
    }
    return bytes;
}

[[nodiscard]] std::string bounded_message(
    const std::string_view message,
    const ServiceLimits& limits) {
    return std::string(message.substr(0, std::min<std::size_t>(
        message.size(), limits.max_error_message_bytes)));
}

[[nodiscard]] grpc::Status exception_status(
    const ServiceLimits& limits,
    GrpcHealthState& health) noexcept {
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
    } catch (const IdempotencyDomainError& error) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                bounded_message(error.what(), limits)};
    } catch (const PendingOperationError& error) {
        return {grpc::StatusCode::ABORTED,
                bounded_message(error.what(), limits)};
    } catch (const UnsupportedIdTypeError& error) {
        return {grpc::StatusCode::UNIMPLEMENTED,
                bounded_message(error.what(), limits)};
    } catch (const ServerUnavailableError& error) {
        return {grpc::StatusCode::UNAVAILABLE,
                bounded_message(error.what(), limits)};
    } catch (const DuplicateIdError& error) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                bounded_message(error.what(), limits)};
    } catch (const InvalidArgumentError& error) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                bounded_message(error.what(), limits)};
    } catch (const InsufficientStorageError& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                bounded_message(error.what(), limits)};
    } catch (const IdempotencyCapacityError& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                bounded_message(error.what(), limits)};
    } catch (const MutationCapacityError& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                bounded_message(error.what(), limits)};
    } catch (const CorruptionError& error) {
        health.mark_terminal_fault();
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
    } catch (const StorageError& error) {
        health.mark_terminal_fault();
        return {grpc::StatusCode::INTERNAL,
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

[[nodiscard]] std::uint64_t validate_id_batch(
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
            return count;
        }
        case wire::IdBatch::kUtf8Ids:
            throw UnsupportedIdTypeError(
                "UTF-8 IDs are reserved but not supported by format 7");
        case wire::IdBatch::ENCODING_NOT_SET:
            throw InvalidArgumentError("IdBatch encoding is required");
    }
    throw InvalidArgumentError("IdBatch contains an unknown encoding");
}

[[nodiscard]] std::vector<std::uint64_t> decode_ids(
    const wire::IdBatch& batch,
    const ServiceLimits& limits) {
    const std::uint64_t count = validate_id_batch(batch, limits);
    switch (batch.encoding_case()) {
        case wire::IdBatch::kUint64Ids: {
            const wire::UInt64Ids& encoded = batch.uint64_ids();
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

struct GetPreflight {
    std::uint32_t dimension = 0;
    std::uint64_t response_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
};

[[nodiscard]] std::uint64_t estimated_read_buffer_bytes(
    const ReaderOptions& reader,
    const std::uint64_t row_count,
    const std::uint64_t full_read_bytes) {
    // A one-row range is read directly into the result. Every temporary range
    // buffer therefore consumes at least two requested rows, and the sum of
    // all buffered spans cannot exceed useful row bytes plus the permitted
    // gaps between adjacent requests.
    if (row_count < 2) {
        return 0;
    }
    std::uint64_t gap_bytes = 0;
    std::uint64_t maximum_planned_bytes = 0;
    if (!checked_multiply(
            row_count - 1, reader.max_gap_bytes, gap_bytes)
        || !checked_add(
            full_read_bytes, gap_bytes, maximum_planned_bytes)) {
        throw std::length_error("Get backend-buffer estimate overflow");
    }

    std::uint64_t parallel_ranges = 1;
    if (reader.io_backend == IoBackend::IoUring) {
        if (reader.io_uring_workers == 0) {
            throw CorruptionError("io_uring reader has zero workers");
        }
        const std::uint64_t ranges_per_worker =
            reader.io_uring_queue_depth / reader.io_uring_workers
            + (reader.io_uring_queue_depth % reader.io_uring_workers != 0
                   ? 1
                   : 0);
        if (!checked_multiply(
                ranges_per_worker,
                reader.io_uring_workers,
                parallel_ranges)) {
            throw std::length_error("Get io_uring buffer estimate overflow");
        }
    } else if (reader.io_workers != 0) {
        parallel_ranges = reader.io_workers;
    }
    parallel_ranges = std::min(parallel_ranges, row_count / 2);

    std::uint64_t concurrent_buffer_cap = 0;
    if (!checked_multiply(
            parallel_ranges,
            reader.max_read_range_bytes,
            concurrent_buffer_cap)) {
        concurrent_buffer_cap = std::numeric_limits<std::uint64_t>::max();
    }
    return std::min(maximum_planned_bytes, concurrent_buffer_cap);
}

[[nodiscard]] std::uint64_t serialized_get_response_size(
    const CollectionDescription& collection,
    const std::uint32_t dimension,
    const std::uint64_t row_count,
    const std::uint64_t payload_bytes) {
    wire::EmbeddingBatch metadata;
    metadata.set_dtype(public_dtype(collection.dtype));
    metadata.set_dimension(dimension);
    metadata.set_row_count(row_count);

    std::uint64_t embedded_bytes = metadata.ByteSizeLong();
    if (payload_bytes != 0) {
        std::uint64_t value_field_bytes = 0;
        if (!checked_add(
                1, protobuf_varint_size(payload_bytes), value_field_bytes)
            || !checked_add(
                value_field_bytes, payload_bytes, value_field_bytes)
            || !checked_add(
                embedded_bytes, value_field_bytes, embedded_bytes)) {
            throw std::length_error("Get protobuf response size overflow");
        }
    }

    std::uint64_t response_bytes = 0;
    if (!checked_add(
            1, protobuf_varint_size(embedded_bytes), response_bytes)
        || !checked_add(
            response_bytes, embedded_bytes, response_bytes)) {
        throw std::length_error("Get protobuf response size overflow");
    }
    return response_bytes;
}

[[nodiscard]] GetPreflight preflight_get(
    const wire::GetRequest& request,
    const CollectionDescription& collection,
    const std::uint64_t row_count,
    const ServiceLimits& limits) {
    const std::uint32_t dimension = request.prefix_dimensions() == 0
                                        ? collection.dimension
                                        : request.prefix_dimensions();
    if (dimension == 0 || dimension > collection.dimension) {
        throw InvalidArgumentError("prefix dimension is out of range");
    }
    if (collection.dimension == 0 || collection.record_size == 0
        || collection.record_size % collection.dimension != 0) {
        throw CorruptionError("collection row geometry is invalid");
    }
    const std::uint64_t item_size =
        collection.record_size / collection.dimension;
    std::uint64_t output_row_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t full_read_bytes = 0;
    if (!checked_multiply(dimension, item_size, output_row_bytes)
        || !checked_multiply(
            row_count, output_row_bytes, output_bytes)
        || !checked_multiply(
            row_count, collection.record_size, full_read_bytes)
        || output_bytes > std::numeric_limits<std::size_t>::max()
        || full_read_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("Get result geometry exceeds this process");
    }

    const std::uint64_t response_bytes = serialized_get_response_size(
        collection, dimension, row_count, output_bytes);
    if (response_bytes > limits.max_response_bytes) {
        throw std::length_error(
            "Get response exceeds max_response_bytes before native read");
    }

    std::uint64_t decoded_id_bytes = 0;
    std::uint64_t native_scratch_bytes = 0;
    std::uint64_t estimated_peak_bytes = request.ByteSizeLong();
    // The current prefix implementation reads complete rows and then compacts
    // them. At the serialization boundary the native result and protobuf bytes
    // coexist. full_read + serialized_response is therefore a conservative
    // peak for both full-row and prefix reads.
    if (!checked_multiply(
            row_count, sizeof(std::uint64_t), decoded_id_bytes)
        || !checked_multiply(
            row_count, kGetNativeScratchBytesPerRow,
            native_scratch_bytes)
        || !checked_add(
            estimated_peak_bytes, decoded_id_bytes,
            estimated_peak_bytes)
        || !checked_add(
            estimated_peak_bytes, native_scratch_bytes,
            estimated_peak_bytes)
        || !checked_add(
            estimated_peak_bytes, full_read_bytes,
            estimated_peak_bytes)
        || !checked_add(
            estimated_peak_bytes, response_bytes,
            estimated_peak_bytes)
        || !checked_add(
            estimated_peak_bytes,
            estimated_read_buffer_bytes(
                collection.reader, row_count, full_read_bytes),
            estimated_peak_bytes)) {
        throw std::length_error("Get peak-memory estimate overflow");
    }
    return {dimension, response_bytes, estimated_peak_bytes};
}

[[nodiscard]] std::uint64_t estimate_data_call_bytes(
    const std::uint64_t request_bytes,
    const std::uint64_t row_count,
    const std::uint64_t working_bytes_per_row,
    const std::uint64_t additional_bytes) {
    std::uint64_t working_bytes = 0;
    std::uint64_t estimated = request_bytes;
    if (!checked_multiply(
            row_count, working_bytes_per_row, working_bytes)
        || !checked_add(estimated, working_bytes, estimated)
        || !checked_add(estimated, additional_bytes, estimated)) {
        throw std::length_error(
            "gRPC data-plane peak-memory estimate overflow");
    }
    return estimated;
}

[[nodiscard]] detail::GrpcAdmissionController::Lease require_admission(
    detail::GrpcAdmissionController& admission,
    const std::uint64_t estimated_bytes,
    const bool scan = false) {
    auto lease = admission.try_acquire(estimated_bytes, scan);
    if (!lease.has_value()) {
        if (admission.snapshot().closed) {
            throw ServerUnavailableError(
                "ThrowDB server is not serving data-plane requests");
        }
        throw std::length_error(
            "gRPC data-plane admission exhausted; retry later or reduce "
            "the request");
    }
    return std::move(*lease);
}

[[nodiscard]] wire::ReaderBackend public_reader_backend(
    const IoBackend backend) {
    switch (backend) {
        case IoBackend::ThreadPool:
            return wire::READER_BACKEND_THREAD_POOL;
        case IoBackend::IoUring:
            return wire::READER_BACKEND_IO_URING;
    }
    throw InvalidArgumentError("unknown native reader backend");
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

[[nodiscard]] OperationId replay_domain(
    const wire::OperationId& value) noexcept {
    return {value.high(), value.low()};
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
        const ServiceLimits limits,
        const GrpcServerOptions& options,
        std::shared_ptr<GrpcHealthState> health)
        : engine_(engine),
          limits_(limits),
          options_(options),
          health_(std::move(health)),
          admission_(
              options.max_concurrent_calls,
              options.max_inflight_bytes,
              options.max_concurrent_scans,
              options.max_scan_inflight_bytes),
          scan_call_budget_(
              std::max<std::uint64_t>(
                  1,
                  options.max_scan_inflight_bytes
                      / options.max_concurrent_scans)),
          scan_max_duration_millis_(options.scan_max_duration_millis),
          scan_idle_timeout_millis_(options.scan_idle_timeout_millis) {
        health_->attach_admission(&admission_);
    }

    [[nodiscard]] detail::GrpcAdmissionSnapshot admission_snapshot() const {
        return admission_.snapshot();
    }

    grpc::Status Health(
        grpc::ServerContext*,
        const wire::HealthRequest* request,
        wire::HealthResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            response->set_status(
                health_->serving()
                    ? wire::HealthResponse::SERVING_STATUS_SERVING
                    : wire::HealthResponse::SERVING_STATUS_NOT_SERVING);
            response->set_protocol_major(kProtocolMajor);
            response->set_protocol_minor(kProtocolMinor);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_, *health_);
        }
    }

    grpc::Status Describe(
        grpc::ServerContext*,
        const wire::DescribeRequest* request,
        wire::DescribeResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            const CollectionDescription description = engine_.describe();
            const EngineOperationalStats operational =
                engine_.operational_stats();
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
            if (description.durable_idempotency) {
                wire::OperationId* public_domain =
                    response->mutable_idempotency_domain();
                public_domain->set_high(
                    operational.idempotency.domain.high);
                public_domain->set_low(
                    operational.idempotency.domain.low);
            }
            wire::ReaderConfiguration* reader = response->mutable_reader();
            reader->set_max_read_range_bytes(
                description.reader.max_read_range_bytes);
            reader->set_max_gap_bytes(description.reader.max_gap_bytes);
            reader->set_io_workers(description.reader.io_workers);
            reader->set_io_queue_capacity(
                description.reader.io_queue_capacity);
            reader->set_io_ranges_per_task(
                description.reader.io_ranges_per_task);
            reader->set_io_backend(
                public_reader_backend(description.reader.io_backend));
            reader->set_io_uring_queue_depth(
                description.reader.io_uring_queue_depth);
            reader->set_io_uring_workers(
                description.reader.io_uring_workers);
            wire::TuningSelection* tuning = response->mutable_tuning();
            tuning->set_mode(description.tuning.mode);
            tuning->set_source(description.tuning.source);
            tuning->set_status(description.tuning.status);
            tuning->set_reuse_level(description.tuning.reuse_level);
            tuning->set_explicit_overrides(
                description.tuning.explicit_overrides);
            wire::AvailabilityConfiguration* availability =
                response->mutable_availability();
            availability->set_max_concurrent_calls(
                options_.max_concurrent_calls);
            availability->set_max_concurrent_scans(
                options_.max_concurrent_scans);
            availability->set_max_unary_inflight_bytes(
                options_.max_inflight_bytes);
            availability->set_max_scan_inflight_bytes(
                options_.max_scan_inflight_bytes);
            availability->set_max_grpc_memory_bytes(
                options_.max_grpc_memory_bytes);
            availability->set_scan_max_duration_millis(
                options_.scan_max_duration_millis);
            availability->set_scan_idle_timeout_millis(
                options_.scan_idle_timeout_millis);
            availability->set_shutdown_grace_millis(
                options_.shutdown_grace_millis);
            availability->set_shutdown_hard_timeout_millis(
                options_.shutdown_hard_timeout_millis);
            const detail::GrpcAdmissionSnapshot admission_snapshot =
                admission_.snapshot();
            wire::AdmissionStatistics* admission_stats =
                response->mutable_admission_stats();
            admission_stats->set_active_unary_calls(
                admission_snapshot.active_calls
                - admission_snapshot.active_scans);
            admission_stats->set_active_scans(
                admission_snapshot.active_scans);
            admission_stats->set_unary_estimated_bytes(
                admission_snapshot.estimated_bytes);
            admission_stats->set_scan_estimated_bytes(
                admission_snapshot.scan_estimated_bytes);
            admission_stats->set_admitted_unary_calls(
                admission_snapshot.admitted_unary_calls);
            admission_stats->set_rejected_unary_calls(
                admission_snapshot.rejected_unary_calls);
            admission_stats->set_admitted_scans(
                admission_snapshot.admitted_scans);
            admission_stats->set_rejected_scans(
                admission_snapshot.rejected_scans);
            admission_stats->set_admission_closed(
                admission_snapshot.closed);
            admission_stats->set_draining(health_->draining());
            admission_stats->set_terminal_faulted(
                health_->terminal_faulted());
            wire::EngineStatistics* engine_stats =
                response->mutable_engine_stats();
            engine_stats->set_catalog_generation(
                operational.mutation.generation);
            engine_stats->set_committed_sequence(
                operational.mutation.committed_sequence);
            engine_stats->set_materialized_sequence(
                operational.mutation.materialized_sequence);
            engine_stats->set_active_wal_bytes(
                operational.mutation.active_wal_bytes);
            engine_stats->set_active_wal_transactions(
                operational.mutation.active_wal_transaction_count);
            engine_stats->set_active_wal_operations(
                operational.mutation.active_wal_operation_count);
            engine_stats->set_frozen_wal_count(
                operational.mutation.frozen_wal_count);
            engine_stats->set_frozen_wal_bytes(
                operational.mutation.frozen_wal_bytes);
            engine_stats->set_total_wal_bytes(
                operational.mutation.total_wal_bytes);
            engine_stats->set_total_wal_operations(
                operational.mutation.total_wal_operation_count);
            engine_stats->set_overlay_entries(
                operational.mutation.overlay_entry_count);
            engine_stats->set_estimated_overlay_bytes(
                operational.mutation.estimated_overlay_bytes);
            engine_stats->set_delta_run_count(
                operational.mutation.delta_run_count);
            engine_stats->set_total_delta_run_bytes(
                operational.mutation.total_delta_run_bytes);
            engine_stats->set_total_delta_segment_bytes(
                operational.mutation.total_delta_segment_bytes);
            engine_stats->set_idempotency_journal_bytes(
                operational.idempotency.journal_bytes);
            engine_stats->set_idempotency_completed_operations(
                operational.idempotency.completed_operations);
            engine_stats->set_idempotency_aborted_operations(
                operational.idempotency.aborted_operations);
            engine_stats->set_idempotency_max_bytes(
                operational.idempotency.maximum_journal_bytes);
            engine_stats->set_idempotency_max_operations(
                operational.idempotency.maximum_operations);
            engine_stats->set_idempotency_generation(
                operational.idempotency.generation);
            engine_stats->set_filesystem_available_bytes(
                operational.filesystem_available_bytes);
            engine_stats->set_mutation_disk_reserve_bytes(
                operational.mutation_disk_reserve_bytes);
            engine_stats->set_maintenance_active(
                operational.maintenance_active);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_, *health_);
        }
    }

    grpc::Status Get(
        grpc::ServerContext* context,
        const wire::GetRequest* request,
        wire::GetResponse* response) override {
        try {
            require_request_budget(*request, limits_);
            if (!health_->serving()) {
                return {grpc::StatusCode::UNAVAILABLE,
                        "ThrowDB server is not serving data-plane requests"};
            }
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED, "Get was cancelled"};
            }
            const std::uint64_t row_count =
                validate_id_batch(request->ids(), limits_);
            const GetPreflight preflight = preflight_get(
                *request, engine_.describe(), row_count, limits_);
            [[maybe_unused]] auto admission = require_admission(
                admission_, preflight.estimated_peak_bytes);
            const std::vector<std::uint64_t> ids = decode_ids(
                request->ids(), limits_);
            const GetResult result = engine_.get(
                ids, preflight.dimension);
            set_embedding_response(result, *response);
            require_response_budget(*response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_, *health_);
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
            if (!health_->serving()) {
                return {grpc::StatusCode::UNAVAILABLE,
                        "ThrowDB server is not serving data-plane requests"};
            }
            if (!engine_.describe().writable) {
                return {grpc::StatusCode::FAILED_PRECONDITION,
                        "collection was opened read-only"};
            }
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED,
                        "Delete was cancelled before admission"};
            }
            const std::uint64_t row_count =
                validate_id_batch(request->ids(), limits_);
            std::uint64_t response_upper_bound = 0;
            if (!checked_add(
                    row_count, kDeleteResponseEnvelopeAllowance,
                    response_upper_bound)
                || response_upper_bound > limits_.max_response_bytes) {
                throw std::length_error(
                    "Delete response exceeds max_response_bytes before "
                    "mutation");
            }
            const std::uint64_t estimated_bytes = estimate_data_call_bytes(
                request->ByteSizeLong(), row_count,
                kMutationWorkingBytesPerRow,
                response_upper_bound);
            std::unique_lock mutation_lane(
                mutation_lane_, std::try_to_lock);
            if (!mutation_lane.owns_lock()) {
                return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "ThrowDB mutation lane is busy; retry with the same "
                        "operation_id"};
            }
            [[maybe_unused]] auto admission = require_admission(
                admission_, estimated_bytes);
            const std::vector<std::uint64_t> ids = decode_ids(
                request->ids(), limits_);
            const MutationResult result = engine_.mutate_in_domain(
                MutationKind::Delete, operation_id(request->operation_id()),
                replay_domain(request->idempotency_domain()), ids);
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
            return exception_status(limits_, *health_);
        }
    }

    grpc::Status ScanIds(
        grpc::ServerContext* context,
        const wire::ScanIdsRequest* request,
        grpc::ServerWriter<wire::ScanIdsResponse>* writer) override {
        try {
            require_request_budget(*request, limits_);
            if (!health_->serving()) {
                return {grpc::StatusCode::UNAVAILABLE,
                        "ThrowDB server is not serving data-plane requests"};
            }
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

            const std::uint64_t estimated_bytes = estimate_data_call_bytes(
                request->ByteSizeLong(), batch_rows,
                kScanWorkingBytesPerRow,
                kScanEnvelopeAllowance);
            if (estimated_bytes > scan_call_budget_) {
                throw std::length_error(
                    "ScanIds batch exceeds its per-scan memory budget");
            }
            [[maybe_unused]] auto admission = require_admission(
                admission_, scan_call_budget_, true);
            // Synchronous ServerWriter::Write may wait on HTTP/2 flow
            // control. The watchdog cancels the context from a bounded helper
            // thread even when the client supplies no deadline.
            detail::GrpcScanWatchdog lifetime(
                scan_max_duration_millis_,
                scan_idle_timeout_millis_,
                [context] { context->TryCancel(); });

            auto scan = engine_.open_scan(
                scan_call_budget_ - estimated_bytes);
            const NativeIdScanMetadata metadata = scan->metadata();
            const auto [token_high, token_low] = random_scan_token();
            std::vector<std::uint64_t> ids(
                static_cast<std::size_t>(batch_rows));
            std::vector<NativeIdLocation> locations;
            if (request->include_locations()) {
                locations.resize(ids.size());
            }
            const auto cancellation_status = [&]()
                -> std::optional<grpc::Status> {
                if (const auto expired =
                        scan_expired_status(lifetime.expiry());
                    expired.has_value()) {
                    return expired;
                }
                if (context->IsCancelled()) {
                    return grpc::Status{
                        grpc::StatusCode::CANCELLED,
                        "ScanIds was cancelled"};
                }
                return std::nullopt;
            };
            const NativeIdScan::StopPredicate stop_requested =
                [context, &lifetime] {
                    return context->IsCancelled()
                           || lifetime.expiry()
                                  != detail::GrpcScanExpiry::None;
                };

            std::uint64_t emitted = 0;
            do {
                if (const auto cancelled = cancellation_status();
                    cancelled.has_value()) {
                    return *cancelled;
                }
                const std::uint64_t count =
                    scan->read_into(ids, locations, stop_requested);
                // A cooperative stop is intentionally zero-progress and does
                // not mean the native cursor is corrupt. Drop any partial
                // batch and release its pinned snapshot/admission lease.
                if (const auto cancelled = cancellation_status();
                    cancelled.has_value()) {
                    return *cancelled;
                }
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
                    if (const auto expired =
                            scan_expired_status(lifetime.expiry());
                        expired.has_value()) {
                        return *expired;
                    }
                    return {grpc::StatusCode::CANCELLED,
                            "ScanIds client stopped receiving"};
                }
                lifetime.mark_progress();
                if (done) {
                    break;
                }
            } while (true);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_, *health_);
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
            if (!health_->serving()) {
                return {grpc::StatusCode::UNAVAILABLE,
                        "ThrowDB server is not serving data-plane requests"};
            }
            const CollectionDescription description = engine_.describe();
            if (!description.writable) {
                return {grpc::StatusCode::FAILED_PRECONDITION,
                        "collection was opened read-only"};
            }
            if (context->IsCancelled()) {
                return {grpc::StatusCode::CANCELLED,
                        "mutation was cancelled before admission"};
            }
            const std::uint64_t row_count =
                validate_id_batch(request.ids(), limits_);
            const wire::EmbeddingBatch& embeddings = request.embeddings();
            if (embeddings.dimension() != description.dimension
                || native_dtype(embeddings.dtype()) != description.dtype
                || embeddings.row_count() != row_count) {
                throw InvalidArgumentError(
                    "mutation embedding schema does not match collection");
            }
            std::uint64_t expected_bytes = 0;
            if (!checked_multiply(row_count, description.record_size,
                                  expected_bytes)
                || expected_bytes > std::numeric_limits<std::size_t>::max()
                || embeddings.values_le().size()
                       != static_cast<std::size_t>(expected_bytes)) {
                throw InvalidArgumentError(
                    "mutation values_le length does not match row geometry");
            }
            if (limits_.max_response_bytes
                < kMutationResponseUpperBound) {
                throw std::length_error(
                    "Mutation response exceeds max_response_bytes before "
                    "mutation");
            }
            std::uint64_t additional_bytes = 0;
            if (!checked_add(
                    expected_bytes, kMutationResponseUpperBound,
                    additional_bytes)) {
                throw std::length_error(
                    "mutation peak-memory estimate overflow");
            }
            const std::uint64_t estimated_bytes = estimate_data_call_bytes(
                request.ByteSizeLong(), row_count,
                kMutationWorkingBytesPerRow, additional_bytes);
            std::unique_lock mutation_lane(
                mutation_lane_, std::try_to_lock);
            if (!mutation_lane.owns_lock()) {
                return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "ThrowDB mutation lane is busy; retry with the same "
                        "operation_id"};
            }
            [[maybe_unused]] auto admission = require_admission(
                admission_, estimated_bytes);
            const std::vector<std::uint64_t> ids = decode_ids(
                request.ids(), limits_);
            const MutationResult result = engine_.mutate_in_domain(
                kind, operation_id(request.operation_id()),
                replay_domain(request.idempotency_domain()), ids,
                bytes_view(embeddings.values_le()));
            response.set_committed_sequence(result.commit_sequence);
            response.set_affected_count(result.row_count);
            require_response_budget(response, limits_);
            return grpc::Status::OK;
        } catch (...) {
            return exception_status(limits_, *health_);
        }
    }

    SingleCollectionEngine& engine_;
    ServiceLimits limits_;
    GrpcServerOptions options_;
    std::shared_ptr<GrpcHealthState> health_;
    detail::GrpcAdmissionController admission_;
    std::uint64_t scan_call_budget_ = 0;
    std::uint32_t scan_max_duration_millis_ = 0;
    std::uint32_t scan_idle_timeout_millis_ = 0;
    std::mutex mutation_lane_;
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

[[nodiscard]] bool tls_configured(
    const GrpcServerOptions& options) noexcept {
    return !options.tls_certificate_path.empty()
           || !options.tls_private_key_path.empty()
           || !options.tls_client_ca_path.empty();
}

[[nodiscard]] bool mutual_tls_configured(
    const GrpcServerOptions& options) noexcept {
    return !options.tls_certificate_path.empty()
           && !options.tls_private_key_path.empty()
           && !options.tls_client_ca_path.empty();
}

struct ParsedTcpEndpoint {
    std::string host;
    std::uint16_t port = 0;
};

[[nodiscard]] std::optional<ParsedTcpEndpoint> parse_tcp_endpoint(
    const std::string& endpoint) {
    if (endpoint.empty()) {
        return std::nullopt;
    }
    std::string host;
    std::string_view port_text;
    if (endpoint.front() == '[') {
        const std::size_t close = endpoint.find(']');
        if (close == std::string::npos || close == 1
            || close + 2 > endpoint.size() || endpoint[close + 1] != ':') {
            return std::nullopt;
        }
        host = endpoint.substr(1, close - 1);
        port_text = std::string_view(endpoint).substr(close + 2);
    } else {
        const std::size_t colon = endpoint.rfind(':');
        if (colon == std::string::npos || colon == 0
            || endpoint.find(':') != colon) {
            return std::nullopt;
        }
        host = endpoint.substr(0, colon);
        port_text = std::string_view(endpoint).substr(colon + 1);
    }
    if (port_text.empty()
        || !std::all_of(
            port_text.begin(), port_text.end(),
            [](const unsigned char byte) {
                return byte >= '0' && byte <= '9';
            })) {
        return std::nullopt;
    }
    std::uint32_t port = 0;
    for (const unsigned char byte : port_text) {
        port = port * 10U + (byte - '0');
        if (port > 65'535U) {
            return std::nullopt;
        }
    }
    if (std::any_of(
            host.begin(), host.end(),
            [](const unsigned char byte) {
                return byte <= 0x20U || byte == 0x7fU;
            })) {
        return std::nullopt;
    }
    return ParsedTcpEndpoint{std::move(host), static_cast<std::uint16_t>(port)};
}

[[nodiscard]] bool is_loopback_endpoint(const std::string& endpoint) {
    const auto parsed = parse_tcp_endpoint(endpoint);
    if (!parsed.has_value()) {
        return false;
    }
    const std::string& host = parsed->host;
    if (host == "::1" || host == "0:0:0:0:0:0:0:1") {
        return true;
    }
    std::size_t begin = 0;
    for (unsigned int component = 0; component != 4; ++component) {
        const std::size_t end = host.find('.', begin);
        if ((component != 3 && end == std::string::npos)
            || (component == 3 && end != std::string::npos)) {
            return false;
        }
        const std::size_t limit = end == std::string::npos
                                      ? host.size()
                                      : end;
        if (limit == begin || limit - begin > 3) {
            return false;
        }
        if (limit - begin > 1 && host[begin] == '0') {
            return false;
        }
        unsigned int value = 0;
        for (std::size_t index = begin; index != limit; ++index) {
            const unsigned char byte =
                static_cast<unsigned char>(host[index]);
            if (byte < '0' || byte > '9') {
                return false;
            }
            value = value * 10U + (byte - '0');
        }
        if (value > 255U || (component == 0 && value != 127U)) {
            return false;
        }
        begin = limit + 1;
    }
    return begin == host.size() + 1;
}

void validate_transport_security(const GrpcServerOptions& options) {
    if (!parse_tcp_endpoint(options.listen_address).has_value()) {
        throw InvalidArgumentError(
            "gRPC listen address must be a strict HOST:PORT or [IPv6]:PORT "
            "endpoint with a numeric port");
    }
    const bool certificate = !options.tls_certificate_path.empty();
    const bool private_key = !options.tls_private_key_path.empty();
    const bool client_ca = !options.tls_client_ca_path.empty();
    if (certificate != private_key) {
        throw InvalidArgumentError(
            "--tls-cert and --tls-key must be configured together");
    }
    if (client_ca && (!certificate || !private_key)) {
        throw InvalidArgumentError(
            "--tls-client-ca requires --tls-cert and --tls-key");
    }
    if (!is_loopback_endpoint(options.listen_address)
        && !mutual_tls_configured(options)
        && !options.allow_insecure_non_loopback) {
        throw InvalidArgumentError(
            "refusing a non-loopback gRPC endpoint without mTLS; configure "
            "--tls-cert, --tls-key, and --tls-client-ca, or explicitly pass "
            "--allow-insecure-non-loopback for a trusted-network deployment");
    }
}

[[nodiscard]] std::string read_pem_file(
    const std::string& path,
    const std::string_view option) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw InvalidArgumentError(
            "cannot read " + std::string(option) + " file: " + path);
    }
    std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (input.bad() || contents.empty()) {
        throw InvalidArgumentError(
            std::string(option) + " file is empty or unreadable: " + path);
    }
    return contents;
}

[[nodiscard]] std::shared_ptr<grpc::ServerCredentials>
server_credentials(const GrpcServerOptions& options) {
    if (!tls_configured(options)) {
        return grpc::InsecureServerCredentials();
    }
    grpc::SslServerCredentialsOptions tls;
    grpc::SslServerCredentialsOptions::PemKeyCertPair key_pair;
    key_pair.private_key = read_pem_file(
        options.tls_private_key_path, "--tls-key");
    key_pair.cert_chain = read_pem_file(
        options.tls_certificate_path, "--tls-cert");
    tls.pem_key_cert_pairs.push_back(std::move(key_pair));
    if (!options.tls_client_ca_path.empty()) {
        tls.pem_root_certs = read_pem_file(
            options.tls_client_ca_path, "--tls-client-ca");
        tls.client_certificate_request =
            GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
    } else {
        tls.client_certificate_request =
            GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE;
    }
    return grpc::SslServerCredentials(tls);
}

}  // namespace

class GrpcServer::Impl {
public:
    Impl(
        SingleCollectionEngine& engine,
        const ServiceLimits limits,
        GrpcServerOptions options)
        : health_(std::make_shared<GrpcHealthState>()),
          service_(engine, limits, options, health_),
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
            || options_.max_concurrent_calls == 0
            || options_.max_inflight_bytes == 0
            || options_.max_scan_inflight_bytes == 0
            || options_.max_grpc_memory_bytes == 0
            || options_.max_concurrent_scans == 0
            || options_.max_concurrent_scans
                   >= options_.max_concurrent_calls
            || options_.max_concurrent_calls
                   >= options_.max_worker_threads
            || options_.scan_max_duration_millis == 0
            || options_.scan_idle_timeout_millis == 0
            || options_.shutdown_grace_millis == 0
            || options_.shutdown_hard_timeout_millis
                   < options_.shutdown_grace_millis
            || options_.keepalive_time_millis == 0
            || options_.keepalive_timeout_millis == 0
            || options_.minimum_ping_interval_millis == 0) {
            throw InvalidArgumentError(
                "gRPC server options must be greater than zero and "
                "must reserve worker/call slots outside scans; the hard "
                "shutdown timeout must not be shorter than graceful drain");
        }
        if (options_.max_grpc_memory_bytes
            > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError(
                "max_grpc_memory_bytes does not fit this process");
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
        validate_transport_security(options_);
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
        quota.Resize(static_cast<std::size_t>(
            options_.max_grpc_memory_bytes));
        quota.SetMaxThreads(static_cast<int>(options_.max_worker_threads));
        builder.SetResourceQuota(quota);
        builder.RegisterService(&service_);

        int port = 0;
        builder.AddListeningPort(
            options_.listen_address,
            server_credentials(options_), &port);
        server_ = builder.BuildAndStart();
        if (server_ == nullptr) {
            throw StorageError("could not start ThrowDB gRPC server");
        }
        endpoint_.address = selected_endpoint(options_.listen_address, port);
        health_->attach(server_->GetHealthCheckService());
        if (!options_.defer_serving_until_ready) {
            health_->mark_serving();
        }
        started_.store(true, std::memory_order_release);
        return endpoint_;
    }

    void mark_serving() {
        {
            const std::lock_guard lock(state_mutex_);
            if (server_ == nullptr || shutdown_started_) {
                throw std::logic_error(
                    "gRPC server must be running before it can become ready");
            }
        }
        health_->mark_serving();
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

    void begin_draining() { health_->begin_draining(); }

    void report_terminal_fault() noexcept {
        health_->mark_terminal_fault();
    }

    [[nodiscard]] bool terminal_faulted() const noexcept {
        return health_->terminal_faulted();
    }

    [[nodiscard]] GrpcAdmissionStats admission_stats() const {
        const detail::GrpcAdmissionSnapshot snapshot =
            service_.admission_snapshot();
        return {
            .active_unary_calls =
                snapshot.active_calls - snapshot.active_scans,
            .active_scans = snapshot.active_scans,
            .unary_estimated_bytes = snapshot.estimated_bytes,
            .scan_estimated_bytes = snapshot.scan_estimated_bytes,
            .admitted_unary_calls = snapshot.admitted_unary_calls,
            .rejected_unary_calls = snapshot.rejected_unary_calls,
            .admitted_scans = snapshot.admitted_scans,
            .rejected_scans = snapshot.rejected_scans,
            .admission_closed = snapshot.closed,
            .draining = health_->draining(),
            .terminal_faulted = health_->terminal_faulted(),
        };
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
        health_->begin_draining();
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
    std::shared_ptr<GrpcHealthState> health_;
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

void GrpcServer::mark_serving() { impl_->mark_serving(); }

void GrpcServer::serve(const std::function<bool()>& should_stop) {
    impl_->serve(should_stop);
}

void GrpcServer::begin_draining() { impl_->begin_draining(); }

void GrpcServer::report_terminal_fault() noexcept {
    impl_->report_terminal_fault();
}

bool GrpcServer::terminal_faulted() const noexcept {
    return impl_->terminal_faulted();
}

GrpcAdmissionStats GrpcServer::admission_stats() const {
    return impl_->admission_stats();
}

void GrpcServer::shutdown() { impl_->shutdown(); }

bool GrpcServer::started() const noexcept { return impl_->started(); }

}  // namespace throwdb::server
