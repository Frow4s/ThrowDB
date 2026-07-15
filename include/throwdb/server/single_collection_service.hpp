#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

#include "throwdb/server/collection_engine.hpp"
#include "throwdb/server/protocol.hpp"

namespace throwdb::server {

struct ServiceLimits {
    std::uint64_t max_request_bytes = 256ULL << 20;
    std::uint64_t max_response_bytes = 512ULL << 20;
    std::uint64_t max_ids_per_request = 4'000'000;
    std::uint32_t max_scan_batch_rows = 1'000'000;
    std::uint32_t max_error_message_bytes = 4096;
};

// A transport-independent service over one already-open collection. A sink
// returns false after transport cancellation; streaming work then stops at the
// next chunk boundary. V1 emits exactly one response for unary operations and
// one or more RESPONSE|STREAM frames for ScanIds.
class SingleCollectionService {
public:
    using FrameSink = std::function<bool(protocol::Frame&&)>;

    explicit SingleCollectionService(
        SingleCollectionEngine& engine,
        ServiceLimits limits = {});

    SingleCollectionService(const SingleCollectionService&) = delete;
    SingleCollectionService& operator=(const SingleCollectionService&) = delete;

    void handle(const protocol::Frame& request, const FrameSink& sink);

    [[nodiscard]] const ServiceLimits& limits() const noexcept;

private:
    [[nodiscard]] protocol::Frame make_error(
        const protocol::FrameHeader& request,
        protocol::StatusCode status,
        const std::string& message,
        std::uint32_t extra_flags = 0) const;

    void handle_impl(const protocol::Frame& request, const FrameSink& sink);
    void health(const protocol::Frame& request, const FrameSink& sink) const;
    void describe(const protocol::Frame& request, const FrameSink& sink) const;
    void get(const protocol::Frame& request, const FrameSink& sink) const;
    void mutate(const protocol::Frame& request, const FrameSink& sink);
    void scan_ids(const protocol::Frame& request, const FrameSink& sink) const;

    SingleCollectionEngine& engine_;
    ServiceLimits limits_;
};

}  // namespace throwdb::server
