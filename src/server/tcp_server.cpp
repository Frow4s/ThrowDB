#include "tcp_server.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#error "throwdbd TCP transport currently requires POSIX sockets"
#endif

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "throwdb/errors.hpp"

namespace throwdb::server {
namespace {

using protocol::Frame;
using protocol::FrameFlag;
using protocol::PayloadWriter;
using protocol::StatusCode;

class FileDescriptor {
public:
    FileDescriptor() = default;
    explicit FileDescriptor(const int value) noexcept : value_(value) {}
    ~FileDescriptor() { reset(); }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    FileDescriptor(FileDescriptor&& other) noexcept
        : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.value_, -1));
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept { return value_ >= 0; }

    [[nodiscard]] int release() noexcept { return std::exchange(value_, -1); }

    void reset(const int replacement = -1) noexcept {
        if (value_ >= 0) {
            ::close(value_);
        }
        value_ = replacement;
    }

private:
    int value_ = -1;
};

[[nodiscard]] std::runtime_error socket_error(const std::string& operation) {
    return std::runtime_error(operation + ": " + std::strerror(errno));
}

void set_close_on_exec(const int socket) {
    const auto flags = ::fcntl(socket, F_GETFD);
    if (flags < 0 || ::fcntl(socket, F_SETFD, flags | FD_CLOEXEC) < 0) {
        throw socket_error("could not set FD_CLOEXEC");
    }
}

enum class ReadResult {
    Complete,
    CleanEof,
    Cancelled,
    Truncated,
    Failed,
};

[[nodiscard]] ReadResult read_exact(
    const int socket,
    const std::span<std::byte> output,
    const std::atomic_bool& stopping) noexcept {
    std::size_t offset = 0;
    while (offset != output.size()) {
        if (stopping.load(std::memory_order_relaxed)) {
            return ReadResult::Cancelled;
        }
        pollfd descriptor{};
        descriptor.fd = socket;
        descriptor.events = POLLIN;
        const auto polled = ::poll(&descriptor, 1, 100);
        if (polled == 0) {
            continue;
        }
        if (polled < 0) {
            if (errno == EINTR) {
                continue;
            }
            return ReadResult::Failed;
        }
        if ((descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
            return ReadResult::Failed;
        }
        const auto received = ::recv(
            socket,
            output.data() + offset,
            output.size() - offset,
            0);
        if (received == 0) {
            return offset == 0 ? ReadResult::CleanEof : ReadResult::Truncated;
        }
        if (received < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return ReadResult::Failed;
        }
        offset += static_cast<std::size_t>(received);
    }
    return ReadResult::Complete;
}

[[nodiscard]] bool write_exact(
    const int socket,
    const std::span<const std::byte> input,
    const std::atomic_bool& stopping,
    const std::atomic<std::int64_t>& stop_deadline_ns) noexcept {
    std::size_t offset = 0;
    while (offset != input.size()) {
        if (stopping.load(std::memory_order_relaxed)) {
            const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now()
                                        .time_since_epoch())
                                    .count();
            if (now_ns >= stop_deadline_ns.load(std::memory_order_relaxed)) {
                return false;
            }
        }
        pollfd descriptor{};
        descriptor.fd = socket;
        descriptor.events = POLLOUT;
        const auto polled = ::poll(&descriptor, 1, 100);
        if (polled == 0) {
            continue;
        }
        if (polled < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            return false;
        }
#if defined(MSG_NOSIGNAL)
        constexpr int send_flags = MSG_NOSIGNAL;
#else
        constexpr int send_flags = 0;
#endif
        const auto sent = ::send(
            socket,
            input.data() + offset,
            input.size() - offset,
            send_flags);
        if (sent < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return false;
        }
        if (sent == 0) {
            return false;
        }
        offset += static_cast<std::size_t>(sent);
    }
    return true;
}

[[nodiscard]] bool write_frame(
    const int socket,
    const Frame& frame,
    const std::atomic_bool& stopping,
    const std::atomic<std::int64_t>& stop_deadline_ns) noexcept {
    try {
        auto header = frame.header;
        header.payload_bytes = frame.payload.size();
        const auto encoded = protocol::encode_header(header);
        return write_exact(socket, encoded, stopping, stop_deadline_ns)
               && write_exact(
                   socket, frame.payload, stopping, stop_deadline_ns);
    } catch (...) {
        return false;
    }
}

[[nodiscard]] Frame transport_error(
    const protocol::FrameHeader& request,
    const StatusCode status,
    const std::string_view message) {
    PayloadWriter writer(8 + message.size());
    protocol::write_status(writer, status, message);
    Frame response;
    response.header.opcode = request.opcode;
    response.header.flags = FrameFlag::Response | FrameFlag::Error;
    response.header.request_id = request.request_id;
    response.payload = std::move(writer).take();
    response.header.payload_bytes = response.payload.size();
    return response;
}

}  // namespace

std::string BoundEndpoint::to_string() const {
    return address + ":" + std::to_string(port);
}

class TcpServer::Impl {
public:
    Impl(SingleCollectionService& service, TcpServerOptions options)
        : service_(service), options_(std::move(options)) {
        if (options_.address.empty() || options_.backlog == 0
            || options_.max_connections == 0
            || options_.shutdown_grace_millis == 0) {
            throw InvalidArgumentError("invalid TCP server options");
        }
    }

    ~Impl() {
        request_stop();
        listener_.reset();
        join_workers();
    }

    [[nodiscard]] BoundEndpoint start() {
        if (listener_) {
            throw std::logic_error("TCP server is already started");
        }

        FileDescriptor socket(::socket(AF_INET, SOCK_STREAM, 0));
        if (!socket) {
            throw socket_error("could not create TCP listener");
        }
        set_close_on_exec(socket.get());
        int reuse = 1;
        if (::setsockopt(
                socket.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))
            != 0) {
            throw socket_error("could not set SO_REUSEADDR");
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(options_.port);
        if (options_.address == "localhost") {
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        } else if (::inet_pton(
                       AF_INET, options_.address.c_str(), &address.sin_addr)
                   != 1) {
            throw InvalidArgumentError(
                "--listen currently requires an IPv4 address or localhost");
        }
        if (::bind(
                socket.get(),
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address))
            != 0) {
            throw socket_error("could not bind TCP listener");
        }
        if (::listen(socket.get(), static_cast<int>(options_.backlog)) != 0) {
            throw socket_error("could not listen on TCP socket");
        }

        sockaddr_in bound{};
        socklen_t bound_size = sizeof(bound);
        if (::getsockname(
                socket.get(), reinterpret_cast<sockaddr*>(&bound), &bound_size)
            != 0) {
            throw socket_error("could not inspect bound TCP endpoint");
        }

        listener_ = std::move(socket);
        endpoint_.address = options_.address == "localhost"
                                ? "127.0.0.1"
                                : options_.address;
        endpoint_.port = ntohs(bound.sin_port);
        return endpoint_;
    }

    void serve(const std::function<bool()>& should_stop) {
        if (!listener_) {
            throw std::logic_error("TCP server must be started before serve");
        }
        while (!stopping_.load(std::memory_order_relaxed)) {
            if (should_stop && should_stop()) {
                request_stop();
                break;
            }
            prune_workers();

            pollfd descriptor{};
            descriptor.fd = listener_.get();
            descriptor.events = POLLIN;
            const auto polled = ::poll(&descriptor, 1, 100);
            if (polled == 0) {
                continue;
            }
            if (polled < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw socket_error("TCP listener poll failed");
            }
            if ((descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
                throw std::runtime_error("TCP listener entered an error state");
            }
            if ((descriptor.revents & POLLIN) == 0) {
                continue;
            }

            sockaddr_in peer{};
            socklen_t peer_size = sizeof(peer);
            FileDescriptor client(::accept(
                listener_.get(),
                reinterpret_cast<sockaddr*>(&peer),
                &peer_size));
            if (!client) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                if (stopping_.load(std::memory_order_relaxed)) {
                    break;
                }
                throw socket_error("TCP accept failed");
            }
            try {
                set_close_on_exec(client.get());
            } catch (const std::exception& error) {
                std::cerr << "throwdbd: " << error.what() << '\n';
                continue;
            }

            if (active_connections_.load(std::memory_order_relaxed)
                >= options_.max_connections) {
                // The connection is closed without reading attacker-controlled
                // bytes. Clients surface Unavailable and may use another pool
                // connection; mutations are never accepted before admission.
                continue;
            }
            start_worker(std::move(client));
        }

        listener_.reset();
        join_workers();
    }

    void request_stop() noexcept {
        if (!stopping_.load(std::memory_order_relaxed)) {
            const auto deadline =
                std::chrono::steady_clock::now()
                + std::chrono::milliseconds(options_.shutdown_grace_millis);
            stop_deadline_ns_.store(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    deadline.time_since_epoch())
                    .count(),
                std::memory_order_relaxed);
            stopping_.store(true, std::memory_order_release);
        }
    }

    [[nodiscard]] bool started() const noexcept { return bool(listener_); }

private:
    struct Worker {
        std::shared_ptr<std::atomic_bool> finished;
        std::thread thread;
    };

    void start_worker(FileDescriptor client) {
        auto finished = std::make_shared<std::atomic_bool>(false);
        active_connections_.fetch_add(1, std::memory_order_relaxed);
        workers_.push_back(Worker{
            finished,
            std::thread(
                [this, finished, client = std::move(client)]() mutable {
                    connection_loop(client.get());
                    active_connections_.fetch_sub(1, std::memory_order_relaxed);
                    finished->store(true, std::memory_order_release);
                })});
    }

    void connection_loop(const int client) noexcept {
        try {
            while (!stopping_.load(std::memory_order_relaxed)) {
                std::array<std::byte, protocol::kHeaderBytes> header_bytes{};
                const auto header_result =
                    read_exact(client, header_bytes, stopping_);
                if (header_result == ReadResult::CleanEof
                    || header_result == ReadResult::Cancelled) {
                    return;
                }
                if (header_result != ReadResult::Complete) {
                    return;
                }

                protocol::FrameHeader header;
                try {
                    header = protocol::decode_header(header_bytes);
                } catch (...) {
                    // Without a trusted header/request ID no correlated error
                    // can be sent safely. Close only this connection.
                    return;
                }
                if (header.payload_bytes > service_.limits().max_request_bytes
                    || header.payload_bytes
                           > std::numeric_limits<std::size_t>::max()) {
                    const auto error = transport_error(
                        header,
                        StatusCode::ResourceExhausted,
                        "request exceeds max_request_bytes");
                    static_cast<void>(write_frame(
                        client, error, stopping_, stop_deadline_ns_));
                    return;
                }

                Frame request;
                request.header = header;
                request.payload.resize(
                    static_cast<std::size_t>(header.payload_bytes));
                const auto payload_result =
                    read_exact(client, request.payload, stopping_);
                if (payload_result != ReadResult::Complete) {
                    return;
                }

                bool transport_alive = true;
                service_.handle(
                    request,
                    [&](Frame&& response) {
                        if (!transport_alive) {
                            return false;
                        }
                        transport_alive =
                            write_frame(
                                client,
                                response,
                                stopping_,
                                stop_deadline_ns_);
                        return transport_alive;
                    });
                if (!transport_alive) {
                    return;
                }
            }
        } catch (const std::exception& error) {
            std::lock_guard lock(log_mutex_);
            std::cerr << "throwdbd: connection failed: " << error.what() << '\n';
        } catch (...) {
            std::lock_guard lock(log_mutex_);
            std::cerr << "throwdbd: connection failed with unknown exception\n";
        }
    }

    void prune_workers() {
        auto iterator = workers_.begin();
        while (iterator != workers_.end()) {
            if ((*iterator).finished->load(std::memory_order_acquire)) {
                iterator = workers_.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    void join_workers() {
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
        workers_.clear();
    }

    SingleCollectionService& service_;
    TcpServerOptions options_;
    FileDescriptor listener_;
    BoundEndpoint endpoint_;
    std::atomic_bool stopping_{false};
    std::atomic<std::int64_t> stop_deadline_ns_{
        std::numeric_limits<std::int64_t>::max()};
    std::atomic<std::uint32_t> active_connections_{0};
    std::vector<Worker> workers_;
    std::mutex log_mutex_;
};

TcpServer::TcpServer(
    SingleCollectionService& service,
    TcpServerOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {}

TcpServer::~TcpServer() = default;

BoundEndpoint TcpServer::start() {
    return impl_->start();
}

void TcpServer::serve(const std::function<bool()>& should_stop) {
    impl_->serve(should_stop);
}

void TcpServer::request_stop() noexcept {
    impl_->request_stop();
}

bool TcpServer::started() const noexcept {
    return impl_->started();
}

}  // namespace throwdb::server
