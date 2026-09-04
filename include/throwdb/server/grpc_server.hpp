#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "throwdb/server/single_collection_service.hpp"

namespace throwdb::server {

class SingleCollectionEngine;

struct GrpcServerOptions {
    std::string listen_address = "127.0.0.1:0";
    // Insecure transport is intentionally limited to loopback. A caller must
    // explicitly acknowledge an unauthenticated non-loopback endpoint, or
    // configure a client CA so gRPC requires and verifies client certificates.
    bool allow_insecure_non_loopback = false;
    std::string tls_certificate_path;
    std::string tls_private_key_path;
    std::string tls_client_ca_path;
    // Bounds synchronous gRPC ingress independently of the core I/O
    // scheduler. Calls wait in gRPC rather than creating unbounded threads.
    std::uint32_t max_worker_threads = 64;
    // Data-plane RPCs are admitted without waiting. Keeping this below the
    // worker limit leaves capacity for Health and Describe under overload.
    std::uint32_t max_concurrent_calls = 32;
    // Conservative peak-memory estimates cover the parsed request, decoded
    // IDs, native read/mutation working sets, and protobuf response payload.
    // This pool is reserved exclusively for unary Get/CRUD calls.
    std::uint64_t max_inflight_bytes = 256ULL << 20;
    // Bounds gRPC's own transport/parser buffers, which are allocated before
    // a synchronous handler can acquire the data-plane admission lease.
    std::uint64_t max_grpc_memory_bytes = 512ULL << 20;
    // Streaming scans receive their own call and memory pools so a slow client
    // cannot consume the capacity reserved for unary requests.
    std::uint32_t max_concurrent_scans = 2;
    std::uint64_t max_scan_inflight_bytes = 128ULL << 20;
    // A watchdog cancels a stream that exceeds its total lifetime or makes no
    // successful transport progress for this long. Both bounds are enforced
    // independently of optional client deadlines.
    std::uint32_t scan_max_duration_millis = 3'600'000;
    std::uint32_t scan_idle_timeout_millis = 60'000;
    std::uint32_t shutdown_grace_millis = 30'000;
    // The standalone daemon arms a process-level watchdog while draining.
    // It uses this larger bound to terminate crash-safe but non-cancellable
    // native work rather than holding a writer lease forever.
    std::uint32_t shutdown_hard_timeout_millis = 60'000;
    std::uint32_t keepalive_time_millis = 120'000;
    std::uint32_t keepalive_timeout_millis = 20'000;
    std::uint32_t minimum_ping_interval_millis = 60'000;
    bool permit_keepalive_without_calls = false;
    // The standalone daemon defers readiness until every background worker
    // has started successfully. Embedded servers keep the historical
    // start()-means-SERVING behaviour unless they opt into the same barrier.
    bool defer_serving_until_ready = false;
    // Point-read workload logging is deliberately opt-in. An empty directory
    // leaves the logger completely absent from the read path. Enabled logs
    // must live outside the collection root so backups cannot mistake
    // telemetry for authoritative collection state.
    std::filesystem::path read_log_directory;
    // Every successful logical Get/StreamGet request is represented in full.
    // Large ID sequences are split into bounded records without changing
    // request boundaries or order.
    std::uint32_t read_log_id_chunk_rows = 8'192;
    std::uint64_t read_log_buffer_bytes = 256ULL << 20;
    std::uint64_t read_log_segment_bytes = 256ULL << 20;
    std::uint64_t read_log_rotate_seconds = 3'600;
    std::uint64_t read_log_retention_bytes = 50ULL << 30;
    std::uint64_t read_log_retention_seconds = 7ULL * 24 * 60 * 60;
    std::uint64_t read_log_min_free_bytes = 10ULL << 30;
    std::uint64_t read_log_max_write_bytes_per_second = 16ULL << 20;
    std::uint64_t read_log_max_retained_files = 4'096;
};

struct GrpcBoundEndpoint {
    std::string address;

    [[nodiscard]] const std::string& to_string() const noexcept {
        return address;
    }
};

// A lock-consistent snapshot of the process-wide application admission gate.
// Counters are monotonic and saturate at UINT64_MAX instead of wrapping.
struct GrpcAdmissionStats {
    std::uint32_t active_unary_calls = 0;
    std::uint32_t active_scans = 0;
    std::uint64_t unary_estimated_bytes = 0;
    std::uint64_t scan_estimated_bytes = 0;
    std::uint64_t admitted_unary_calls = 0;
    std::uint64_t rejected_unary_calls = 0;
    std::uint64_t admitted_scans = 0;
    std::uint64_t rejected_scans = 0;
    bool admission_closed = false;
    bool draining = false;
    bool terminal_faulted = false;
};

// Synchronous gRPC transport over the transport-neutral collection engine.
// gRPC owns its bounded request worker pool; native physical reads continue to
// use the core scheduler selected when the collection was opened.
class GrpcServer {
public:
    GrpcServer(
        SingleCollectionEngine& engine,
        ServiceLimits limits = {},
        GrpcServerOptions options = {});
    ~GrpcServer();

    GrpcServer(const GrpcServer&) = delete;
    GrpcServer& operator=(const GrpcServer&) = delete;

    [[nodiscard]] GrpcBoundEndpoint start();
    // Completes a deferred readiness transition after all process components
    // required by the caller have started. Idempotent after start(); it never
    // revives a draining or terminal-faulted server.
    void mark_serving();
    void serve(const std::function<bool()>& should_stop);
    // Stop advertising readiness while leaving the endpoint alive long enough
    // for orchestrators to drain it before shutdown. Idempotent.
    void begin_draining();
    // Publish a fail-closed readiness state after an out-of-band storage
    // component (for example background maintenance) reports a terminal fault.
    void report_terminal_fault() noexcept;
    [[nodiscard]] bool terminal_faulted() const noexcept;
    [[nodiscard]] GrpcAdmissionStats admission_stats() const;
    void shutdown();
    [[nodiscard]] bool started() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server
