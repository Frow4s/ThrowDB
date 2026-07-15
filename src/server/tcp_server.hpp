#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "throwdb/server/single_collection_service.hpp"

namespace throwdb::server {

struct TcpServerOptions {
    std::string address = "127.0.0.1";
    std::uint16_t port = 0;
    std::uint32_t backlog = 128;
    std::uint32_t max_connections = 128;
    // Once draining starts, active responses may finish within this bound.
    // Idle connections stop immediately and no new connection is accepted.
    std::uint32_t shutdown_grace_millis = 30'000;
};

struct BoundEndpoint {
    std::string address;
    std::uint16_t port = 0;

    [[nodiscard]] std::string to_string() const;
};

// V1 intentionally keeps socket I/O separate from request execution. Each
// bounded connection owns one framing loop and allows one in-flight request;
// independent connections execute concurrently in the core's fair scheduler.
// A later evented transport can reuse SingleCollectionService unchanged.
class TcpServer {
public:
    TcpServer(SingleCollectionService& service, TcpServerOptions options = {});
    ~TcpServer();

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    [[nodiscard]] BoundEndpoint start();
    void serve(const std::function<bool()>& should_stop);
    void request_stop() noexcept;
    [[nodiscard]] bool started() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb::server
