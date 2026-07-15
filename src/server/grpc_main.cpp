#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "throwdb/native_collection.hpp"
#include "throwdb/server/collection_engine.hpp"
#include "throwdb/server/grpc_server.hpp"

namespace {

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void handle_signal(int) { stop_requested = 1; }

struct Options {
    std::filesystem::path storage;
    throwdb::server::GrpcServerOptions grpc;
    throwdb::server::ServiceLimits limits;
    throwdb::server::EngineOptions engine;
    throwdb::ReaderOptions reader;
    bool writable = true;

    Options() {
        // gRPC's 4 MiB defaults are too small even for 10k x 1024 float16.
        // One GiB remains bounded and leaves headroom below gRPC's INT_MAX.
        limits.max_request_bytes = 1ULL << 30;
        limits.max_response_bytes = 1ULL << 30;
    }
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: throwdbd --storage PATH [--listen HOST:PORT] "
          "[--read-only]\n"
          "                [--max-worker-threads N] "
          "[--max-request-mib N]\n"
          "                [--max-response-mib N] "
          "[--max-ids-per-request N]\n"
          "                [--max-scan-batch-rows N] "
          "[--shutdown-grace-ms N]\n"
          "                [--keepalive-time-ms N] "
          "[--keepalive-timeout-ms N]\n"
          "                [--minimum-ping-interval-ms N] "
          "[--permit-keepalive-without-calls]\n"
          "                [--no-durable-idempotency] [reader options]\n"
          "reader options: --io-backend thread-pool|io-uring "
          "--io-workers N\n"
          "                --io-queue-capacity N --io-ranges-per-task N\n"
          "                --max-read-range-bytes N --max-gap-bytes N\n"
          "                --io-uring-queue-depth N "
          "--io-uring-workers N");
}

[[nodiscard]] std::string require_value(
    const int argc,
    char** argv,
    int& index) {
    if (index + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

[[nodiscard]] std::uint64_t parse_u64(
    const std::string& text,
    const std::string_view option,
    const bool allow_zero = false) {
    std::size_t parsed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &parsed, 10);
    } catch (...) {
        usage_error(std::string(option) + " requires an unsigned integer");
    }
    if (parsed != text.size() || (!allow_zero && value == 0)) {
        usage_error(
            std::string(option)
            + (allow_zero ? " requires an unsigned integer"
                          : " requires an integer greater than zero"));
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::uint32_t parse_u32(
    const std::string& text,
    const std::string_view option,
    const bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, option, allow_zero);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        usage_error(std::string(option) + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint64_t mib_to_bytes(
    const std::string& text,
    const std::string_view option) {
    const std::uint64_t mib = parse_u64(text, option);
    if (mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
        usage_error(std::string(option) + " byte size overflow");
    }
    return mib << 20;
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: throwdbd --storage PATH "
                   "[--listen 127.0.0.1:0] [--read-only]\n";
            std::exit(0);
        }
        if (argument == "--storage") {
            result.storage = require_value(argc, argv, index);
        } else if (argument == "--listen") {
            result.grpc.listen_address = require_value(argc, argv, index);
        } else if (argument == "--read-only") {
            result.writable = false;
        } else if (argument == "--max-worker-threads") {
            result.grpc.max_worker_threads = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-request-mib") {
            result.limits.max_request_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-response-mib") {
            result.limits.max_response_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-ids-per-request") {
            result.limits.max_ids_per_request = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-scan-batch-rows") {
            result.limits.max_scan_batch_rows = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--shutdown-grace-ms") {
            result.grpc.shutdown_grace_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--keepalive-time-ms") {
            result.grpc.keepalive_time_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--keepalive-timeout-ms") {
            result.grpc.keepalive_timeout_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--minimum-ping-interval-ms") {
            result.grpc.minimum_ping_interval_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--permit-keepalive-without-calls") {
            result.grpc.permit_keepalive_without_calls = true;
        } else if (argument == "--no-durable-idempotency") {
            result.engine.durable_idempotency = false;
        } else if (argument == "--io-backend") {
            const std::string value = require_value(argc, argv, index);
            if (value == "thread-pool") {
                result.reader.io_backend = throwdb::IoBackend::ThreadPool;
            } else if (value == "io-uring") {
                result.reader.io_backend = throwdb::IoBackend::IoUring;
            } else {
                usage_error(
                    "--io-backend requires thread-pool or io-uring");
            }
        } else if (argument == "--io-workers") {
            result.reader.io_workers = parse_u32(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--io-queue-capacity") {
            result.reader.io_queue_capacity = parse_u32(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--io-ranges-per-task") {
            result.reader.io_ranges_per_task = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-read-range-bytes") {
            result.reader.max_read_range_bytes = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-gap-bytes") {
            result.reader.max_gap_bytes = parse_u64(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--io-uring-queue-depth") {
            result.reader.io_uring_queue_depth = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--io-uring-workers") {
            result.reader.io_uring_workers = parse_u32(
                require_value(argc, argv, index), argument);
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    if (result.storage.empty()) {
        usage_error("--storage is required");
    }
    if (result.grpc.listen_address.find(':') == std::string::npos) {
        usage_error("--listen requires a gRPC HOST:PORT endpoint");
    }
    return result;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        auto collection = throwdb::NativeCollection::open(
            options.storage, options.reader, options.writable);
        throwdb::server::SingleCollectionEngine engine(
            *collection, options.storage, options.engine);
        throwdb::server::GrpcServer server(
            engine, options.limits, options.grpc);
        const throwdb::server::GrpcBoundEndpoint endpoint = server.start();

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#if defined(SIGPIPE)
        std::signal(SIGPIPE, SIG_IGN);
#endif

        // BuildAndStart() has bound the endpoint and registered both ThrowDB
        // Health and standard grpc.health.v1.Health before readiness is
        // announced. Diagnostics remain on stderr.
        std::cout << "{\"event\":\"ready\","
                     "\"protocol\":\"grpc/throwdb.v1\","
                     "\"endpoint\":\""
                  << endpoint.to_string() << "\",\"writable\":"
                  << (collection->writable() ? "true" : "false")
                  << ",\"durable_idempotency\":"
                  << (engine.durable_idempotency() ? "true" : "false")
                  << "}\n"
                  << std::flush;

        server.serve([] { return stop_requested != 0; });
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "throwdbd: " << error.what() << '\n';
        return 1;
    }
}
