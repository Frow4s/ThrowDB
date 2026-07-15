#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "throwdb/server/single_collection_service.hpp"

namespace throwdb::server {

class SingleCollectionEngine;

struct GrpcServerOptions {
    std::string listen_address = "127.0.0.1:0";
    // Bounds synchronous gRPC ingress independently of the core I/O
    // scheduler. Calls wait in gRPC rather than creating unbounded threads.
    std::uint32_t max_worker_threads = 64;
    std::uint32_t shutdown_grace_millis = 30'000;
    std::uint32_t keepalive_time_millis = 120'000;
    std::uint32_t keepalive_timeout_millis = 20'000;
    std::uint32_t minimum_ping_interval_millis = 60'000;
    bool permit_keepalive_without_calls = false;
};

struct GrpcBoundEndpoint {
    std::string address;

    [[nodiscard]] const std::string& to_string() const noexcept {
        return address;
    }
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
    void serve(const std::function<bool()>& should_stop);
    void shutdown();
    [[nodiscard]] bool started() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server
