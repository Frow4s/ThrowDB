#include "throwdb/server/single_collection_service.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"

namespace throwdb::server {
namespace {

using protocol::Frame;
using protocol::FrameFlag;
using protocol::Opcode;
using protocol::PayloadReader;
using protocol::PayloadWriter;
using protocol::StatusCode;

[[nodiscard]] bool checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& output) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

[[nodiscard]] std::vector<std::uint64_t> read_ids(
    PayloadReader& reader,
    const std::uint64_t count,
    const ServiceLimits& limits) {
    if (count > limits.max_ids_per_request) {
        throw std::length_error("request exceeds max_ids_per_request");
    }
    if (count > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("ID count exceeds this process address space");
    }
    if (count > reader.remaining() / sizeof(std::uint64_t)) {
        throw protocol::ProtocolError("ID array is truncated");
    }
    std::vector<std::uint64_t> result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index != count; ++index) {
        result.push_back(reader.read_u64());
    }
    return result;
}

[[nodiscard]] std::uint64_t element_size(
    const CollectionDescription& collection) {
    if (collection.dimension == 0
        || collection.record_size % collection.dimension != 0) {
        throw CorruptionError("collection record size is inconsistent with its dimension");
    }
    return collection.record_size / collection.dimension;
}

[[nodiscard]] Frame response_frame(
    const Frame& request,
    std::vector<std::byte> payload,
    const std::uint32_t flags = 0) {
    Frame result;
    result.header.opcode = request.header.opcode;
    result.header.flags = FrameFlag::Response | flags;
    result.header.request_id = request.header.request_id;
    result.header.payload_bytes = payload.size();
    result.payload = std::move(payload);
    return result;
}

void require_empty_request(const Frame& request) {
    if (!request.payload.empty()) {
        throw protocol::ProtocolError("request payload must be empty");
    }
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> random_scan_token() {
    // The token identifies a live server-side cursor, not durable collection
    // state. random_device is sufficient because the token is not an auth
    // secret and is never accepted from a client in protocol v1.
    std::random_device random;
    const auto next = [&random]() {
        return (static_cast<std::uint64_t>(random()) << 32U)
               ^ static_cast<std::uint64_t>(random());
    };
    auto high = next();
    auto low = next();
    if (high == 0 && low == 0) {
        low = 1;
    }
    return {high, low};
}

}  // namespace

SingleCollectionService::SingleCollectionService(
    SingleCollectionEngine& engine,
    const ServiceLimits limits)
    : engine_(engine), limits_(limits) {
    if (limits_.max_request_bytes == 0 || limits_.max_response_bytes == 0
        || limits_.max_ids_per_request == 0
        || limits_.max_scan_batch_rows == 0) {
        throw InvalidArgumentError("server limits must be greater than zero");
    }
}

const ServiceLimits& SingleCollectionService::limits() const noexcept {
    return limits_;
}

Frame SingleCollectionService::make_error(
    const protocol::FrameHeader& request,
    const StatusCode status,
    const std::string& message,
    const std::uint32_t extra_flags) const {
    const auto length = std::min<std::size_t>(
        message.size(), limits_.max_error_message_bytes);
    PayloadWriter writer(8 + length);
    protocol::write_status(writer, status, std::string_view(message).substr(0, length));

    Frame result;
    result.header.opcode = request.opcode;
    result.header.flags = FrameFlag::Response | FrameFlag::Error | extra_flags;
    result.header.request_id = request.request_id;
    result.payload = std::move(writer).take();
    result.header.payload_bytes = result.payload.size();
    return result;
}

void SingleCollectionService::handle(
    const Frame& request,
    const FrameSink& sink) {
    const auto stream_error_flags = request.header.opcode == Opcode::ScanIds
                                        ? FrameFlag::Stream | FrameFlag::EndStream
                                        : 0U;
    const auto emit_error = [&](const StatusCode status, const std::string& message) {
        sink(make_error(
            request.header, status, message, stream_error_flags));
    };
    try {
        handle_impl(request, sink);
    } catch (const MissingIdError& error) {
        emit_error(StatusCode::NotFound, error.what());
    } catch (const ExistingIdError& error) {
        emit_error(StatusCode::AlreadyExists, error.what());
    } catch (const DuplicateIdError& error) {
        emit_error(StatusCode::InvalidArgument, error.what());
    } catch (const protocol::ProtocolError& error) {
        emit_error(StatusCode::ProtocolError, error.what());
    } catch (const IdempotencyConflictError& error) {
        emit_error(StatusCode::FailedPrecondition, error.what());
    } catch (const MutationAbortedError& error) {
        emit_error(StatusCode::FailedPrecondition, error.what());
    } catch (const PendingOperationError& error) {
        emit_error(StatusCode::FailedPrecondition, error.what());
    } catch (const InvalidArgumentError& error) {
        emit_error(StatusCode::InvalidArgument, error.what());
    } catch (const CorruptionError& error) {
        emit_error(StatusCode::DataLoss, error.what());
    } catch (const std::length_error& error) {
        emit_error(StatusCode::ResourceExhausted, error.what());
    } catch (const std::invalid_argument& error) {
        emit_error(StatusCode::InvalidArgument, error.what());
    } catch (const std::exception& error) {
        emit_error(StatusCode::Internal, error.what());
    } catch (...) {
        emit_error(StatusCode::Internal, "unknown server-side failure");
    }
}

void SingleCollectionService::handle_impl(
    const Frame& request,
    const FrameSink& sink) {
    if (request.header.protocol_major != protocol::kProtocolMajor) {
        throw protocol::ProtocolError("unsupported protocol major version");
    }
    if (request.header.header_bytes != protocol::kHeaderBytes) {
        throw protocol::ProtocolError("unsupported wire header size");
    }
    if (request.header.request_id == 0) {
        throw protocol::ProtocolError("request_id must be nonzero");
    }
    if (request.header.flags != protocol::kKnownRequestFlags) {
        throw protocol::ProtocolError("request contains response or unknown flags");
    }
    if (request.header.payload_bytes != request.payload.size()) {
        throw protocol::ProtocolError("payload length does not match frame header");
    }
    if (request.payload.size() > limits_.max_request_bytes) {
        throw std::length_error("request exceeds max_request_bytes");
    }
    if (!protocol::is_known_opcode(request.header.opcode)) {
        throw protocol::ProtocolError("unknown operation code");
    }

    switch (request.header.opcode) {
        case Opcode::Health:
            health(request, sink);
            return;
        case Opcode::Describe:
            describe(request, sink);
            return;
        case Opcode::Get:
            get(request, sink);
            return;
        case Opcode::Insert:
        case Opcode::Update:
        case Opcode::Upsert:
        case Opcode::Delete:
            mutate(request, sink);
            return;
        case Opcode::ScanIds:
            scan_ids(request, sink);
            return;
    }
    throw protocol::ProtocolError("unknown operation code");
}

void SingleCollectionService::health(
    const Frame& request,
    const FrameSink& sink) const {
    require_empty_request(request);
    PayloadWriter writer(24);
    protocol::write_status(writer, StatusCode::Ok);
    writer.write_u32(1);  // SERVING
    writer.write_u32(protocol::kProtocolMinor);
    writer.write_u64(0);  // reserved
    sink(response_frame(request, std::move(writer).take()));
}

void SingleCollectionService::describe(
    const Frame& request,
    const FrameSink& sink) const {
    require_empty_request(request);
    const auto collection = engine_.describe();
    std::uint32_t capabilities =
        protocol::Capability::PrefixGet | protocol::Capability::ScanLocations;
    if (collection.writable) {
        capabilities |= protocol::Capability::Writable;
    }
    if (collection.durable_idempotency) {
        capabilities |= protocol::Capability::DurableIdempotency;
    }

    PayloadWriter writer(40);
    protocol::write_status(writer, StatusCode::Ok);
    writer.write_u32(collection.format_version);
    writer.write_u32(collection.dimension);
    writer.write_u32(static_cast<std::uint32_t>(collection.dtype));
    writer.write_u32(capabilities);
    writer.write_u64(collection.record_size);
    writer.write_u64(collection.count);
    sink(response_frame(request, std::move(writer).take()));
}

void SingleCollectionService::get(
    const Frame& request,
    const FrameSink& sink) const {
    PayloadReader reader(request.payload);
    const auto requested_prefix = reader.read_u32();
    const auto reserved = reader.read_u32();
    const auto count = reader.read_u64();
    if (reserved != 0) {
        throw protocol::ProtocolError("Get reserved field must be zero");
    }
    const auto collection = engine_.describe();
    const auto dimension = requested_prefix == 0
                               ? collection.dimension
                               : requested_prefix;
    if (dimension == 0 || dimension > collection.dimension) {
        throw InvalidArgumentError("Get prefix dimension is out of range");
    }
    auto ids = read_ids(reader, count, limits_);
    reader.require_end();

    std::uint64_t output_row_bytes = 0;
    if (!checked_multiply(dimension, element_size(collection), output_row_bytes)) {
        throw std::length_error("Get row byte size overflow");
    }
    std::uint64_t output_bytes = 0;
    if (!checked_multiply(count, output_row_bytes, output_bytes)
        || output_bytes > limits_.max_response_bytes
        || output_bytes > std::numeric_limits<std::size_t>::max() - 24) {
        throw std::length_error("Get response exceeds max_response_bytes");
    }

    auto result = engine_.get(ids, dimension);

    PayloadWriter writer(24 + result.embeddings.size());
    protocol::write_status(writer, StatusCode::Ok);
    writer.write_u32(dimension);
    writer.write_u32(static_cast<std::uint32_t>(result.dtype));
    writer.write_u64(count);
    writer.write_bytes(result.embeddings);
    sink(response_frame(request, std::move(writer).take()));
}

void SingleCollectionService::mutate(
    const Frame& request,
    const FrameSink& sink) {
    const auto collection = engine_.describe();
    if (!collection.writable) {
        sink(make_error(
            request.header,
            StatusCode::FailedPrecondition,
            "collection was opened read-only"));
        return;
    }

    PayloadReader reader(request.payload);
    // Reserved for durable WAL-integrated idempotency. Both zero and nonzero
    // values are accepted in v1, but Describe does not advertise that a retry
    // is safe until the core can commit this identity atomically with the WAL.
    const auto operation_id_high = reader.read_u64();
    const auto operation_id_low = reader.read_u64();
    static_cast<void>(operation_id_high);
    static_cast<void>(operation_id_low);

    const bool deleting = request.header.opcode == Opcode::Delete;
    std::uint32_t request_dimension = 0;
    std::uint32_t request_dtype = 0;
    std::uint64_t count = 0;
    if (deleting) {
        count = reader.read_u64();
    } else {
        request_dimension = reader.read_u32();
        request_dtype = reader.read_u32();
        count = reader.read_u64();
        if (request_dimension != collection.dimension) {
            throw InvalidArgumentError("mutation dimension does not match collection schema");
        }
        if (request_dtype != static_cast<std::uint32_t>(collection.dtype)) {
            throw InvalidArgumentError("mutation dtype does not match collection schema");
        }
    }

    auto ids = read_ids(reader, count, limits_);
    std::span<const std::byte> embeddings;
    if (!deleting) {
        std::uint64_t embedding_bytes = 0;
        if (!checked_multiply(count, collection.record_size, embedding_bytes)
            || embedding_bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("mutation embedding byte size overflow");
        }
        embeddings = reader.read_bytes(static_cast<std::size_t>(embedding_bytes));
    }
    reader.require_end();

    MutationKind kind;
    switch (request.header.opcode) {
        case Opcode::Insert:
            kind = MutationKind::Insert;
            break;
        case Opcode::Update:
            kind = MutationKind::Update;
            break;
        case Opcode::Upsert:
            kind = MutationKind::Upsert;
            break;
        case Opcode::Delete:
            kind = MutationKind::Delete;
            break;
        default:
            throw protocol::ProtocolError("non-mutation opcode reached mutation handler");
    }
    const auto result = engine_.mutate(
        kind,
        OperationId{operation_id_high, operation_id_low},
        ids,
        embeddings);

    PayloadWriter writer(24 + result.deleted.size());
    protocol::write_status(writer, StatusCode::Ok);
    writer.write_u64(result.commit_sequence);
    writer.write_u64(count);
    writer.write_bytes(std::as_bytes(std::span(result.deleted)));
    sink(response_frame(request, std::move(writer).take()));
}

void SingleCollectionService::scan_ids(
    const Frame& request,
    const FrameSink& sink) const {
    PayloadReader reader(request.payload);
    const auto batch_rows = reader.read_u32();
    const auto flags = reader.read_u32();
    const auto reserved = reader.read_u64();
    reader.require_end();
    if (batch_rows == 0 || batch_rows > limits_.max_scan_batch_rows) {
        throw InvalidArgumentError("ScanIds batch_rows is out of range");
    }
    if ((flags
         & ~static_cast<std::uint32_t>(
             protocol::ScanFlag::IncludeLocations))
            != 0
        || reserved != 0) {
        throw protocol::ProtocolError("ScanIds contains unknown flags or reserved values");
    }

    const bool include_locations =
        (flags & protocol::ScanFlag::IncludeLocations) != 0;
    constexpr std::uint64_t fixed_payload_bytes = 8 + 9 * 8 + 2 * 4;
    const std::uint64_t row_bytes = include_locations ? 16 : 8;
    if (limits_.max_response_bytes < fixed_payload_bytes + row_bytes) {
        throw std::length_error(
            "max_response_bytes cannot fit one ScanIds row");
    }
    const auto response_capacity_rows =
        (limits_.max_response_bytes - fixed_payload_bytes) / row_bytes;
    const auto effective_batch_rows = std::min<std::uint64_t>(
        batch_rows, response_capacity_rows);
    if (effective_batch_rows > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("ScanIds batch exceeds this process address space");
    }

    auto scan = engine_.open_scan();
    const auto metadata = scan->metadata();
    const auto [token_high, token_low] = random_scan_token();
    std::vector<std::uint64_t> ids(
        static_cast<std::size_t>(effective_batch_rows));
    std::vector<NativeIdLocation> locations;
    if (include_locations) {
        locations.resize(ids.size());
    }

    std::uint64_t emitted = 0;
    do {
        const auto count = scan->read_into(ids, locations);
        if (count == 0 && !scan->done()) {
            throw CorruptionError(
                "native ID scan made no progress before completion");
        }
        if (count > ids.size()) {
            throw CorruptionError("native ID scan exceeded output capacity");
        }
        const bool done = scan->done();
        if ((done && emitted + count != metadata.live_count)
            || (!done && emitted + count >= metadata.live_count)) {
            throw CorruptionError(
                "native ID scan count does not match captured live_count");
        }

        PayloadWriter writer(
            static_cast<std::size_t>(fixed_payload_bytes + count * row_bytes));
        protocol::write_status(writer, StatusCode::Ok);
        writer.write_u64(metadata.collection_id_low);
        writer.write_u64(metadata.collection_id_high);
        writer.write_u64(metadata.catalog_generation);
        writer.write_u64(metadata.committed_sequence);
        writer.write_u64(metadata.live_count);
        writer.write_u64(token_high);
        writer.write_u64(token_low);
        writer.write_u64(emitted);
        writer.write_u64(count);
        writer.write_u32(static_cast<std::uint32_t>(
            include_locations ? protocol::LocationFormat::SegmentRowU32
                              : protocol::LocationFormat::None));
        writer.write_u32(0);
        for (std::size_t index = 0; index != count; ++index) {
            writer.write_u64(ids[index]);
        }
        if (include_locations) {
            for (std::size_t index = 0; index != count; ++index) {
                writer.write_u32(locations[index].segment_id);
                writer.write_u32(locations[index].row_index);
            }
        }

        emitted += count;
        const auto response_flags = FrameFlag::Stream
                                    | (done ? FrameFlag::EndStream : 0U);
        if (!sink(response_frame(
                request, std::move(writer).take(), response_flags))) {
            return;
        }
        if (done) {
            break;
        }
    } while (true);
}

}  // namespace throwdb::server
