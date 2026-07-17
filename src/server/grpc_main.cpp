#include <algorithm>
#include <csignal>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "throwdb/errors.hpp"
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
    throwdb::server::EngineMaintenanceOptions maintenance;
    std::uint64_t maintenance_interval_seconds = 60;
    bool automatic_maintenance = true;
    bool writable = false;
    bool writable_flag_seen = false;
    bool read_only_flag_seen = false;
    bool reader_override_seen = false;
    std::uint32_t resolved_tuning_fields = 0;
};

inline constexpr std::uint32_t kAllResolvedTuningFields = 0x1fU;

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: throwdbd --storage PATH [--listen HOST:PORT] "
          "[--read-only]\n"
          "                [--writable] [--allow-insecure-non-loopback]\n"
          "                [--tls-cert PATH --tls-key PATH "
          "[--tls-client-ca PATH]]\n"
          "                [--max-worker-threads N] "
          "[--max-concurrent-calls N]\n"
          "                [--max-concurrent-scans N] "
          "[--max-inflight-mib N] [--max-scan-inflight-mib N]\n"
          "                [--max-grpc-memory-mib N] "
          "[--max-request-mib N]\n"
          "                [--max-response-mib N] "
          "[--max-ids-per-request N]\n"
          "                [--max-scan-batch-rows N] "
          "[--scan-max-duration-ms N] [--scan-idle-timeout-ms N]\n"
          "                [--shutdown-grace-ms N] "
          "[--shutdown-hard-timeout-ms N]\n"
          "                [--keepalive-time-ms N] "
          "[--keepalive-timeout-ms N]\n"
          "                [--minimum-ping-interval-ms N] "
          "[--permit-keepalive-without-calls]\n"
          "                [--no-durable-idempotency] [reader options]\n"
          "                [--mutation-disk-reserve-mib N]\n"
          "                [--idempotency-journal-max-mib N]\n"
          "                [--idempotency-journal-max-operations N]\n"
          "                [--max-wal-backlog-mib N] "
          "[--max-wal-backlog-operations N]\n"
          "                [--max-overlay-entries N] "
          "[--max-overlay-memory-mib N]\n"
          "                [--maintenance-interval-seconds N] "
          "[--no-auto-maintenance]\n"
          "                [--maintenance-max-actions N] "
          "[--rotate-wal-mib N]\n"
          "                [--no-idempotency-checkpoint] "
          "[--idempotency-checkpoint-trigger-percent N]\n"
          "                [--rotate-wal-transactions N] "
          "[--materialize-max-wals N]\n"
          "                [--materialize-max-input-mib N] "
          "[--materialize-max-input-entries N]\n"
          "                [--materialize-max-memory-mib N] "
          "[--fold-min-runs N]\n"
          "                [--fold-max-runs N] "
          "[--fold-max-input-mib N]\n"
          "                [--fold-max-input-entries N] "
          "[--no-auto-clean]\n"
          "                [--cleaner-min-interval-seconds N] "
          "[--cleaner-max-candidate-segments N]\n"
          "                [--cleaner-max-sidecar-scan-mib N] "
          "[--cleaner-max-source-mib N]\n"
          "                [--cleaner-max-output-mib N] "
          "[--cleaner-max-memory-mib N]\n"
          "                [--cleaner-min-garbage-rows N] "
          "[--cleaner-min-reclaimable-mib N]\n"
          "                [--cleaner-min-garbage-ratio-ppm N] "
          "[--no-orphan-reclaim]\n"
          "                [--orphan-min-age-seconds N] "
          "[--orphan-max-files N] [--orphan-max-mib N]\n"
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
    if (text.empty()
        || !std::all_of(
            text.begin(), text.end(),
            [](const unsigned char byte) {
                return byte >= '0' && byte <= '9';
            })) {
        usage_error(std::string(option) + " requires an unsigned integer");
    }
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
    const std::string_view option,
    const bool allow_zero = false) {
    const std::uint64_t mib = parse_u64(text, option, allow_zero);
    if (mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
        usage_error(std::string(option) + " byte size overflow");
    }
    return mib << 20;
}

[[nodiscard]] std::uint64_t seconds_to_nanoseconds(
    const std::string& text,
    const std::string_view option) {
    constexpr std::uint64_t one_second = 1'000'000'000ULL;
    const std::uint64_t seconds = parse_u64(text, option, true);
    if (seconds > std::numeric_limits<std::uint64_t>::max() / one_second) {
        usage_error(std::string(option) + " nanosecond size overflow");
    }
    return seconds * one_second;
}

[[nodiscard]] std::string require_choice(
    const std::string& value,
    const std::string_view option,
    const std::initializer_list<std::string_view> choices) {
    for (const std::string_view choice : choices) {
        if (value == choice) {
            return value;
        }
    }
    usage_error(std::string(option) + " has an unsupported value");
}

[[nodiscard]] bool parse_boolean_flag(
    const std::string& value,
    const std::string_view option) {
    if (value == "0") {
        return false;
    }
    if (value == "1") {
        return true;
    }
    usage_error(std::string(option) + " requires 0 or 1");
}

[[nodiscard]] const char* reader_backend_name(
    const throwdb::IoBackend backend) {
    switch (backend) {
        case throwdb::IoBackend::ThreadPool:
            return "thread_pool";
        case throwdb::IoBackend::IoUring:
            return "io_uring";
    }
    return "unknown";
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: throwdbd --storage PATH "
                   "[--listen 127.0.0.1:0] [--writable]\n";
            std::exit(0);
        }
        if (argument == "--storage") {
            result.storage = require_value(argc, argv, index);
        } else if (argument == "--listen") {
            result.grpc.listen_address = require_value(argc, argv, index);
        } else if (argument == "--writable") {
            result.writable = true;
            result.writable_flag_seen = true;
        } else if (argument == "--read-only") {
            result.writable = false;
            result.read_only_flag_seen = true;
        } else if (argument == "--allow-insecure-non-loopback") {
            result.grpc.allow_insecure_non_loopback = true;
        } else if (argument == "--tls-cert") {
            result.grpc.tls_certificate_path =
                require_value(argc, argv, index);
        } else if (argument == "--tls-key") {
            result.grpc.tls_private_key_path =
                require_value(argc, argv, index);
        } else if (argument == "--tls-client-ca") {
            result.grpc.tls_client_ca_path =
                require_value(argc, argv, index);
        } else if (argument == "--max-worker-threads") {
            result.grpc.max_worker_threads = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-concurrent-calls") {
            result.grpc.max_concurrent_calls = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-concurrent-scans") {
            result.grpc.max_concurrent_scans = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-inflight-mib") {
            result.grpc.max_inflight_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-scan-inflight-mib") {
            result.grpc.max_scan_inflight_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-grpc-memory-mib") {
            result.grpc.max_grpc_memory_bytes = mib_to_bytes(
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
        } else if (argument == "--scan-max-duration-ms") {
            result.grpc.scan_max_duration_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--scan-idle-timeout-ms") {
            result.grpc.scan_idle_timeout_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--shutdown-grace-ms") {
            result.grpc.shutdown_grace_millis = parse_u32(
                require_value(argc, argv, index), argument);
        } else if (argument == "--shutdown-hard-timeout-ms") {
            result.grpc.shutdown_hard_timeout_millis = parse_u32(
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
        } else if (argument == "--mutation-disk-reserve-mib") {
            result.engine.mutation_disk_reserve_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--idempotency-journal-max-mib") {
            result.engine.idempotency_journal_max_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--idempotency-journal-max-operations") {
            result.engine.idempotency_journal_max_operations = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-wal-backlog-mib") {
            result.engine.max_wal_backlog_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-wal-backlog-operations") {
            result.engine.max_wal_backlog_operations = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-overlay-entries") {
            result.engine.max_overlay_entries = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--max-overlay-memory-mib") {
            result.engine.max_overlay_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--maintenance-interval-seconds") {
            result.maintenance_interval_seconds = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--no-auto-maintenance") {
            result.automatic_maintenance = false;
        } else if (argument == "--maintenance-max-actions") {
            result.maintenance.max_actions = parse_u64(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--no-idempotency-checkpoint") {
            result.maintenance.checkpoint_idempotency = false;
        } else if (argument ==
                   "--idempotency-checkpoint-trigger-percent") {
            result.maintenance.idempotency_checkpoint_trigger_percent =
                parse_u32(require_value(argc, argv, index), argument);
        } else if (argument == "--rotate-wal-mib") {
            result.maintenance.rotate_wal_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--rotate-wal-transactions") {
            result.maintenance.rotate_wal_transactions = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--materialize-max-wals") {
            result.maintenance.materialize_max_wals = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--materialize-max-input-mib") {
            result.maintenance.materialize_max_input_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--materialize-max-input-entries") {
            result.maintenance.materialize_max_input_entries = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--materialize-max-memory-mib") {
            result.maintenance.materialize_max_memory_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--fold-min-runs") {
            result.maintenance.fold_min_runs = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--fold-max-runs") {
            result.maintenance.fold_max_runs = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--fold-max-input-mib") {
            result.maintenance.fold_max_input_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--fold-max-input-entries") {
            result.maintenance.fold_max_input_entries = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--no-auto-clean") {
            result.maintenance.clean_segments = false;
        } else if (argument == "--cleaner-min-interval-seconds") {
            result.maintenance.cleaner_min_interval_seconds = parse_u64(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--cleaner-max-candidate-segments") {
            result.maintenance.cleaner_max_candidate_segments = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--cleaner-max-sidecar-scan-mib") {
            result.maintenance.cleaner_max_sidecar_scan_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--cleaner-max-source-mib") {
            result.maintenance.cleaner_max_source_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--cleaner-max-output-mib") {
            result.maintenance.cleaner_max_output_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--cleaner-max-memory-mib") {
            result.maintenance.cleaner_max_memory_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--cleaner-min-garbage-rows") {
            result.maintenance.cleaner_min_garbage_rows = parse_u64(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--cleaner-min-reclaimable-mib") {
            result.maintenance.cleaner_min_reclaimable_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--cleaner-min-garbage-ratio-ppm") {
            result.maintenance.cleaner_min_garbage_ratio_ppm = parse_u32(
                require_value(argc, argv, index), argument, true);
        } else if (argument == "--no-orphan-reclaim") {
            result.maintenance.reclaim_orphans = false;
        } else if (argument == "--orphan-min-age-seconds") {
            result.maintenance.orphan_min_age_ns = seconds_to_nanoseconds(
                require_value(argc, argv, index), argument);
        } else if (argument == "--orphan-max-files") {
            result.maintenance.orphan_max_files = parse_u64(
                require_value(argc, argv, index), argument);
        } else if (argument == "--orphan-max-mib") {
            result.maintenance.orphan_max_bytes = mib_to_bytes(
                require_value(argc, argv, index), argument);
        } else if (argument == "--resolved-tuning-mode") {
            result.engine.tuning.mode = require_choice(
                require_value(argc, argv, index), argument,
                {"auto", "persisted", "conservative", "manual"});
            result.resolved_tuning_fields |= 0x01U;
        } else if (argument == "--resolved-tuning-source") {
            result.engine.tuning.source = require_choice(
                require_value(argc, argv, index), argument,
                {"persisted", "fallback", "manual"});
            result.resolved_tuning_fields |= 0x02U;
        } else if (argument == "--resolved-tuning-status") {
            result.engine.tuning.status = require_choice(
                require_value(argc, argv, index), argument,
                {"selected", "explicit", "registry_missing",
                 "registry_corrupt", "probe_failed", "device_unrecognized",
                 "no_match", "profile_invalid", "backend_unavailable",
                 "registry_bypassed", "backend_initialization_failed"});
            result.resolved_tuning_fields |= 0x04U;
        } else if (argument == "--resolved-tuning-reuse-level") {
            result.engine.tuning.reuse_level = require_choice(
                require_value(argc, argv, index), argument,
                {"exact", "compatible", "fallback", "manual"});
            result.resolved_tuning_fields |= 0x08U;
        } else if (argument == "--resolved-tuning-explicit-overrides") {
            result.engine.tuning.explicit_overrides = parse_boolean_flag(
                require_value(argc, argv, index), argument);
            result.resolved_tuning_fields |= 0x10U;
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
            result.reader_override_seen = true;
        } else if (argument == "--io-workers") {
            result.reader.io_workers = parse_u32(
                require_value(argc, argv, index), argument, true);
            result.reader_override_seen = true;
        } else if (argument == "--io-queue-capacity") {
            result.reader.io_queue_capacity = parse_u32(
                require_value(argc, argv, index), argument, true);
            result.reader_override_seen = true;
        } else if (argument == "--io-ranges-per-task") {
            result.reader.io_ranges_per_task = parse_u32(
                require_value(argc, argv, index), argument);
            result.reader_override_seen = true;
        } else if (argument == "--max-read-range-bytes") {
            result.reader.max_read_range_bytes = parse_u64(
                require_value(argc, argv, index), argument);
            result.reader_override_seen = true;
        } else if (argument == "--max-gap-bytes") {
            result.reader.max_gap_bytes = parse_u64(
                require_value(argc, argv, index), argument, true);
            result.reader_override_seen = true;
        } else if (argument == "--io-uring-queue-depth") {
            result.reader.io_uring_queue_depth = parse_u32(
                require_value(argc, argv, index), argument);
            result.reader_override_seen = true;
        } else if (argument == "--io-uring-workers") {
            result.reader.io_uring_workers = parse_u32(
                require_value(argc, argv, index), argument);
            result.reader_override_seen = true;
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
    if (result.writable_flag_seen && result.read_only_flag_seen) {
        usage_error("--writable and --read-only are mutually exclusive");
    }
    if (result.maintenance_interval_seconds
        > static_cast<std::uint64_t>(
            std::chrono::seconds::max().count())) {
        usage_error("--maintenance-interval-seconds is too large");
    }
    if (result.maintenance.max_actions > 6
        || result.maintenance.idempotency_checkpoint_trigger_percent == 0
        || result.maintenance.idempotency_checkpoint_trigger_percent >= 100
        || result.maintenance.rotate_wal_bytes == 0
        || result.maintenance.rotate_wal_transactions == 0
        || result.maintenance.materialize_max_wals == 0
        || result.maintenance.materialize_max_input_bytes == 0
        || result.maintenance.materialize_max_input_entries == 0
        || result.maintenance.materialize_max_memory_bytes == 0
        || result.maintenance.fold_min_runs == 0
        || result.maintenance.fold_max_runs < 2
        || result.maintenance.fold_max_runs > 4096
        || result.maintenance.fold_min_runs
               > result.maintenance.fold_max_runs
        || result.maintenance.fold_max_input_bytes == 0
        || result.maintenance.fold_max_input_entries == 0
        || (result.maintenance.clean_segments
            && (result.maintenance.cleaner_max_candidate_segments == 0
                || result.maintenance.cleaner_max_sidecar_scan_bytes == 0
                || result.maintenance.cleaner_max_source_bytes == 0
                || result.maintenance.cleaner_max_output_bytes == 0
                || result.maintenance.cleaner_max_memory_bytes == 0
                || result.maintenance.cleaner_min_garbage_ratio_ppm
                       > 1'000'000))
        || (result.maintenance.reclaim_orphans
            && (result.maintenance.orphan_max_files == 0
                || result.maintenance.orphan_max_bytes == 0))) {
        usage_error(
            "maintenance bounds are invalid or internally inconsistent");
    }
    if (result.automatic_maintenance
        && result.maintenance.max_actions != 0
        && result.maintenance.rotate_wal_bytes
               > result.maintenance.materialize_max_input_bytes) {
        usage_error(
            "--rotate-wal-mib must not exceed --materialize-max-input-mib "
            "when automatic maintenance is enabled");
    }
    result.engine.automatic_rotate_wal_bytes =
        result.automatic_maintenance && result.maintenance.max_actions != 0
            ? result.maintenance.rotate_wal_bytes
            : 0;
    result.engine.automatic_materialize_max_input_bytes =
        result.automatic_maintenance && result.maintenance.max_actions != 0
            ? result.maintenance.materialize_max_input_bytes
            : 0;
    result.engine.automatic_materialize_max_input_entries =
        result.automatic_maintenance && result.maintenance.max_actions != 0
            ? result.maintenance.materialize_max_input_entries
            : 0;
    if (result.resolved_tuning_fields != 0
        && result.resolved_tuning_fields != kAllResolvedTuningFields) {
        usage_error("resolved tuning diagnostics must be complete");
    }
    if (result.resolved_tuning_fields == 0) {
        result.engine.tuning.mode = result.reader_override_seen
                                        ? "manual"
                                        : "conservative";
        result.engine.tuning.source = result.reader_override_seen
                                          ? "manual"
                                          : "fallback";
        result.engine.tuning.status = result.reader_override_seen
                                          ? "explicit"
                                          : "registry_bypassed";
        result.engine.tuning.reuse_level = result.reader_override_seen
                                               ? "manual"
                                               : "fallback";
        result.engine.tuning.explicit_overrides =
            result.reader_override_seen;
    }
    result.engine.reader = result.reader;
    return result;
}

[[nodiscard]] throwdb::NativeOpenOptions native_open_options(
    const Options& options) {
    throwdb::NativeOpenOptions result;
    result.max_wal_bytes = options.engine.max_wal_backlog_bytes;
    result.max_wal_operations = options.engine.max_wal_backlog_operations;
    result.max_overlay_entries = options.engine.max_overlay_entries;
    result.max_overlay_bytes = options.engine.max_overlay_bytes;
    result.allow_pending_mutation_recovery =
        options.writable && options.engine.durable_idempotency;
    return result;
}

[[nodiscard]] std::string json_string(const std::string_view value) {
    std::ostringstream output;
    output << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (byte < 0x20U) {
                    output << "\\u00" << hex[byte >> 4U]
                           << hex[byte & 0x0fU];
                } else {
                    output << static_cast<char>(byte);
                }
                break;
        }
    }
    output << '"';
    return output.str();
}

class MaintenanceWorker final {
public:
    MaintenanceWorker(
        throwdb::server::SingleCollectionEngine& engine,
        throwdb::server::GrpcServer& server,
        const throwdb::server::EngineMaintenanceOptions options,
        const std::uint64_t interval_seconds)
        : engine_(engine),
          server_(server),
          options_(options),
          interval_(static_cast<std::chrono::seconds::rep>(
              interval_seconds)) {}

    ~MaintenanceWorker() { stop(); }

    MaintenanceWorker(const MaintenanceWorker&) = delete;
    MaintenanceWorker& operator=(const MaintenanceWorker&) = delete;

    void start() {
        if (worker_.joinable()) {
            throw std::logic_error("maintenance worker is already started");
        }
        worker_ = std::thread([this] { run(); });
    }

    void stop() noexcept {
        request_stop();
        join();
    }

    void request_stop() noexcept {
        {
            const std::lock_guard lock(mutex_);
            stop_requested_ = true;
        }
        condition_.notify_all();
    }

    void join() noexcept {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    [[nodiscard]] bool stopping() const noexcept {
        const std::lock_guard lock(mutex_);
        return stop_requested_;
    }

    [[nodiscard]] throwdb::server::EngineMaintenanceResult run_cycle() {
        throwdb::server::EngineMaintenanceResult aggregate;
        throwdb::server::EngineMaintenanceOptions one = options_;
        one.max_actions = 1;
        bool initialized = false;
        for (std::uint64_t action = 0;
             action != options_.max_actions && !stopping(); ++action) {
            const auto result = engine_.maintain_once(one);
            if (!initialized) {
                aggregate.before = result.before;
                aggregate.idempotency_before = result.idempotency_before;
                initialized = true;
            }
            aggregate.after = result.after;
            aggregate.idempotency_after = result.idempotency_after;
            aggregate.actions_attempted += result.actions_attempted;
            aggregate.actions_performed += result.actions_performed;
            aggregate.rotated = aggregate.rotated || result.rotated;
            aggregate.materialized =
                aggregate.materialized || result.materialized;
            aggregate.materialization_budget_blocked =
                aggregate.materialization_budget_blocked
                || result.materialization_budget_blocked;
            aggregate.materialization_entry_budget_blocked =
                aggregate.materialization_entry_budget_blocked
                || result.materialization_entry_budget_blocked;
            aggregate.materialization_memory_budget_blocked =
                aggregate.materialization_memory_budget_blocked
                || result.materialization_memory_budget_blocked;
            aggregate.materialization_disk_budget_blocked =
                aggregate.materialization_disk_budget_blocked
                || result.materialization_disk_budget_blocked;
            aggregate.folded = aggregate.folded || result.folded;
            aggregate.compaction_budget_blocked =
                aggregate.compaction_budget_blocked
                || result.compaction_budget_blocked;
            aggregate.fold_disk_budget_blocked =
                aggregate.fold_disk_budget_blocked
                || result.fold_disk_budget_blocked;
            aggregate.idempotency_checkpointed =
                aggregate.idempotency_checkpointed
                || result.idempotency_checkpointed;
            aggregate.cleaner_attempted =
                aggregate.cleaner_attempted || result.cleaner_attempted;
            aggregate.cleaner_disk_budget_blocked =
                aggregate.cleaner_disk_budget_blocked
                || result.cleaner_disk_budget_blocked;
            if (result.cleaner_attempted) {
                aggregate.cleaner = result.cleaner;
            }
            aggregate.orphan_reclaim_attempted =
                aggregate.orphan_reclaim_attempted
                || result.orphan_reclaim_attempted;
            aggregate.orphan_removed_files +=
                result.orphan_removed_files;
            aggregate.orphan_removed_bytes +=
                result.orphan_removed_bytes;
            aggregate.orphan_durability_complete =
                aggregate.orphan_durability_complete
                && result.orphan_durability_complete;
            aggregate.orphan_failed_files += result.orphan_failed_files;

            if (result.actions_attempted == 0
                || result.orphan_reclaim_attempted
                || result.materialization_budget_blocked
                || result.materialization_entry_budget_blocked
                || result.materialization_memory_budget_blocked
                || result.materialization_disk_budget_blocked
                || result.compaction_budget_blocked
                || result.fold_disk_budget_blocked
                || result.cleaner_disk_budget_blocked
                || result.actions_performed == 0) {
                break;
            }
        }
        aggregate.action_budget_reached =
            aggregate.actions_attempted >= options_.max_actions;
        return aggregate;
    }

    void run() noexcept {
        std::unique_lock lock(mutex_);
        while (!condition_.wait_for(
            lock, interval_, [this] { return stop_requested_; })) {
            lock.unlock();
            try {
                const auto result = run_cycle();
                std::cerr
                    << "{\"event\":\"maintenance\",\"status\":\"ok\","
                       "\"actions_attempted\":"
                    << result.actions_attempted
                    << ",\"actions_performed\":"
                    << result.actions_performed
                    << ",\"rotated\":"
                    << (result.rotated ? "true" : "false")
                    << ",\"materialized\":"
                    << (result.materialized ? "true" : "false")
                    << ",\"folded\":"
                    << (result.folded ? "true" : "false")
                    << ",\"materialization_budget_blocked\":"
                    << (result.materialization_budget_blocked
                            ? "true" : "false")
                    << ",\"materialization_entry_budget_blocked\":"
                    << (result.materialization_entry_budget_blocked
                            ? "true" : "false")
                    << ",\"materialization_memory_budget_blocked\":"
                    << (result.materialization_memory_budget_blocked
                            ? "true" : "false")
                    << ",\"materialization_disk_budget_blocked\":"
                    << (result.materialization_disk_budget_blocked
                            ? "true" : "false")
                    << ",\"compaction_budget_blocked\":"
                    << (result.compaction_budget_blocked
                            ? "true" : "false")
                    << ",\"fold_disk_budget_blocked\":"
                    << (result.fold_disk_budget_blocked
                            ? "true" : "false")
                    << ",\"idempotency_checkpointed\":"
                    << (result.idempotency_checkpointed
                            ? "true" : "false")
                    << ",\"idempotency_generation\":"
                    << result.idempotency_after.generation
                    << ",\"idempotency_journal_bytes\":"
                    << result.idempotency_after.journal_bytes
                    << ",\"cleaner_attempted\":"
                    << (result.cleaner_attempted ? "true" : "false")
                    << ",\"cleaner_cleaned\":"
                    << (result.cleaner.cleaned ? "true" : "false")
                    << ",\"cleaner_disk_budget_blocked\":"
                    << (result.cleaner_disk_budget_blocked
                            ? "true" : "false")
                    << ",\"cleaner_garbage_rows\":"
                    << result.cleaner.garbage_rows
                    << ",\"cleaner_reclaimable_bytes\":"
                    << result.cleaner.reclaimable_bytes
                    << ",\"orphan_removed_files\":"
                    << result.orphan_removed_files
                    << ",\"orphan_removed_bytes\":"
                    << result.orphan_removed_bytes
                    << ",\"orphan_durability_complete\":"
                    << (result.orphan_durability_complete
                            ? "true" : "false")
                    << ",\"orphan_failed_files\":"
                    << result.orphan_failed_files
                    << ",\"active_wal_bytes\":"
                    << result.after.active_wal_bytes
                    << ",\"active_wal_transactions\":"
                    << result.after.active_wal_transaction_count
                    << ",\"active_wal_operations\":"
                    << result.after.active_wal_operation_count
                    << ",\"total_wal_bytes\":"
                    << result.after.total_wal_bytes
                    << ",\"total_wal_operations\":"
                    << result.after.total_wal_operation_count
                    << ",\"overlay_entries\":"
                    << result.after.overlay_entry_count
                    << ",\"estimated_overlay_bytes\":"
                    << result.after.estimated_overlay_bytes
                    << ",\"frozen_wals\":"
                    << result.after.frozen_wal_count
                    << ",\"delta_runs\":"
                    << result.after.delta_run_count << "}\n"
                    << std::flush;
            } catch (const throwdb::server::InsufficientStorageError& error) {
                std::cerr
                    << "{\"event\":\"maintenance\","
                       "\"status\":\"deferred_insufficient_storage\","
                       "\"error\":"
                    << json_string(error.what()) << "}\n"
                    << std::flush;
            } catch (const throwdb::server::MutationCapacityError& error) {
                std::cerr
                    << "{\"event\":\"maintenance\","
                       "\"status\":\"deferred_mutation_capacity\","
                       "\"error\":"
                    << json_string(error.what()) << "}\n"
                    << std::flush;
            } catch (const throwdb::StorageError& error) {
                server_.report_terminal_fault();
                std::cerr
                    << "{\"event\":\"maintenance\","
                       "\"status\":\"terminal_storage_fault\","
                       "\"error\":"
                    << json_string(error.what()) << "}\n"
                    << std::flush;
                return;
            } catch (const std::exception& error) {
                std::cerr
                    << "{\"event\":\"maintenance\","
                       "\"status\":\"error\",\"error\":"
                    << json_string(error.what()) << "}\n"
                    << std::flush;
            } catch (...) {
                std::cerr
                    << "{\"event\":\"maintenance\","
                       "\"status\":\"error\","
                       "\"error\":\"unknown maintenance failure\"}\n"
                    << std::flush;
            }
            lock.lock();
        }
    }

    throwdb::server::SingleCollectionEngine& engine_;
    throwdb::server::GrpcServer& server_;
    throwdb::server::EngineMaintenanceOptions options_;
    std::chrono::seconds interval_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_requested_ = false;
    std::thread worker_;
};

// gRPC cancellation and native lifecycle work are cooperative. The daemon is
// nevertheless a lease-owning process, so orchestration needs one absolute
// upper bound after drain begins. Every mutation/lifecycle transition is
// crash-safe; forced _Exit is therefore safer than retaining the writer lease
// indefinitely after the advertised hard deadline.
class ShutdownWatchdog final {
public:
    explicit ShutdownWatchdog(const std::uint32_t timeout_millis)
        : timeout_(timeout_millis), worker_([this] { run(); }) {}

    ~ShutdownWatchdog() {
        {
            const std::lock_guard lock(mutex_);
            completed_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    ShutdownWatchdog(const ShutdownWatchdog&) = delete;
    ShutdownWatchdog& operator=(const ShutdownWatchdog&) = delete;

private:
    void run() noexcept {
        std::unique_lock lock(mutex_);
        if (condition_.wait_for(lock, timeout_, [this] { return completed_; })) {
            return;
        }
        // Do not log through iostream here: another thread may hold the stream
        // lock, which would turn the hard process bound into another
        // unbounded wait. Supervisors observe the nonzero exit directly.
        std::_Exit(EXIT_FAILURE);
    }

    const std::chrono::milliseconds timeout_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool completed_ = false;
    std::thread worker_;
};

}  // namespace

int main(const int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        // The daemon is format-7-only. Reject legacy stores through the
        // lightweight manifest/index-header path before loading a flat index
        // or replaying a potentially large format-6 mutation backlog.
        const throwdb::NativeCollectionMetadata metadata =
            throwdb::NativeCollection::inspect_metadata(options.storage);
        if (metadata.format_version != 7) {
            throw throwdb::InvalidArgumentError(
                "throwdbd supports storage format 7 only; rebuild the "
                "collection in the current format before starting the "
                "server");
        }
        std::unique_ptr<throwdb::NativeCollection> collection;
        try {
            collection = throwdb::NativeCollection::open(
                options.storage, options.reader, options.writable, false,
                native_open_options(options));
        } catch (const throwdb::ReaderBackendError&) {
            if (options.engine.tuning.mode != "auto"
                || options.engine.tuning.source != "persisted"
                || options.engine.tuning.explicit_overrides) {
                throw;
            }
            // Match Collection.open(auto): a persisted backend can become
            // unavailable after capability probing. Retry exactly once with
            // conservative defaults; strict/manual modes remain fail-closed.
            options.reader = throwdb::ReaderOptions{};
            options.engine.reader = options.reader;
            options.engine.tuning.source = "fallback";
            options.engine.tuning.status =
                "backend_initialization_failed";
            options.engine.tuning.reuse_level = "fallback";
            collection = throwdb::NativeCollection::open(
                options.storage, options.reader, options.writable, false,
                native_open_options(options));
        }
        throwdb::server::SingleCollectionEngine engine(
            *collection, options.storage, options.engine);
        // A bound socket is not ready until the optional maintenance thread
        // has also started. This prevents a transient SERVING response or
        // ready record followed by process exit on std::thread construction.
        options.grpc.defer_serving_until_ready = true;
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
        const bool tls_enabled =
            !options.grpc.tls_certificate_path.empty();
        const bool client_certificate_required =
            !options.grpc.tls_client_ca_path.empty();
        const char* const transport_security =
            client_certificate_required ? "mtls"
            : tls_enabled              ? "tls"
                                       : "insecure";
        const bool maintenance_enabled =
            collection->writable() && options.automatic_maintenance
            && options.maintenance.max_actions != 0;
        std::unique_ptr<MaintenanceWorker> maintenance;
        if (maintenance_enabled) {
            maintenance = std::make_unique<MaintenanceWorker>(
                engine,
                server,
                options.maintenance,
                options.maintenance_interval_seconds);
        }
        const auto startup_operational = engine.operational_stats();
        const auto& startup_mutation = startup_operational.mutation;
        if (maintenance != nullptr) {
            maintenance->start();
        }
        server.mark_serving();
        std::cout << "{\"event\":\"ready\","
                     "\"protocol\":\"grpc/throwdb.v1\","
                     "\"endpoint\":\""
                  << endpoint.to_string() << "\",\"writable\":"
                  << (collection->writable() ? "true" : "false")
                  << ",\"access_mode\":\""
                  << (collection->writable() ? "writable" : "read_only")
                  << "\",\"readiness\":{\"status\":\"SERVING\"}"
                  << ",\"security\":{\"transport\":\""
                  << transport_security
                  << "\",\"client_certificate_required\":"
                  << (client_certificate_required ? "true" : "false")
                  << ",\"insecure_non_loopback_override\":"
                  << (options.grpc.allow_insecure_non_loopback
                          ? "true" : "false")
                  << "}"
                  << ",\"durable_idempotency\":"
                  << (engine.durable_idempotency() ? "true" : "false")
                  << ",\"idempotency_domain\":{\"high\":"
                  << startup_operational.idempotency.domain.high
                  << ",\"low\":"
                  << startup_operational.idempotency.domain.low << "}"
                  << ",\"mutation_safety\":{"
                     "\"disk_reserve_bytes\":"
                  << options.engine.mutation_disk_reserve_bytes
                  << ",\"idempotency_journal_max_bytes\":"
                  << options.engine.idempotency_journal_max_bytes
                  << ",\"idempotency_journal_max_operations\":"
                  << options.engine.idempotency_journal_max_operations
                  << ",\"automatic_rotate_wal_bytes\":"
                  << options.engine.automatic_rotate_wal_bytes
                  << ",\"automatic_materialize_max_input_bytes\":"
                  << options.engine.automatic_materialize_max_input_bytes
                  << ",\"automatic_materialize_max_input_entries\":"
                  << options.engine.automatic_materialize_max_input_entries
                  << ",\"max_wal_backlog_bytes\":"
                  << options.engine.max_wal_backlog_bytes
                  << ",\"max_wal_backlog_operations\":"
                  << options.engine.max_wal_backlog_operations
                  << ",\"max_overlay_entries\":"
                  << options.engine.max_overlay_entries
                  << ",\"max_overlay_bytes\":"
                  << options.engine.max_overlay_bytes
                  << ",\"startup_wal_bytes\":"
                  << startup_mutation.total_wal_bytes
                  << ",\"startup_wal_operations\":"
                  << startup_mutation.total_wal_operation_count
                  << ",\"startup_overlay_entries\":"
                  << startup_mutation.overlay_entry_count
                  << ",\"startup_estimated_overlay_bytes\":"
                  << startup_mutation.estimated_overlay_bytes
                  << ",\"startup_filesystem_available_bytes\":"
                  << startup_operational.filesystem_available_bytes
                  << ",\"startup_idempotency_journal_bytes\":"
                  << startup_operational.idempotency.journal_bytes
                  << ",\"startup_idempotency_operations\":"
                  << (startup_operational.idempotency.completed_operations
                      + startup_operational.idempotency.aborted_operations)
                  << "}"
                  << ",\"maintenance\":{\"enabled\":"
                  << (maintenance_enabled ? "true" : "false")
                  << ",\"interval_seconds\":"
                  << options.maintenance_interval_seconds
                  << ",\"max_actions\":"
                  << options.maintenance.max_actions
                  << ",\"checkpoint_idempotency\":"
                  << (options.maintenance.checkpoint_idempotency
                          ? "true" : "false")
                  << ",\"idempotency_checkpoint_trigger_percent\":"
                  << options.maintenance
                         .idempotency_checkpoint_trigger_percent
                  << ",\"rotate_wal_bytes\":"
                  << options.maintenance.rotate_wal_bytes
                  << ",\"rotate_wal_transactions\":"
                  << options.maintenance.rotate_wal_transactions
                  << ",\"materialize_max_wals\":"
                  << options.maintenance.materialize_max_wals
                  << ",\"materialize_max_input_bytes\":"
                  << options.maintenance.materialize_max_input_bytes
                  << ",\"materialize_max_input_entries\":"
                  << options.maintenance.materialize_max_input_entries
                  << ",\"materialize_max_memory_bytes\":"
                  << options.maintenance.materialize_max_memory_bytes
                  << ",\"fold_min_runs\":"
                  << options.maintenance.fold_min_runs
                  << ",\"fold_max_runs\":"
                  << options.maintenance.fold_max_runs
                  << ",\"fold_max_input_bytes\":"
                  << options.maintenance.fold_max_input_bytes
                  << ",\"fold_max_input_entries\":"
                  << options.maintenance.fold_max_input_entries
                  << ",\"clean_segments\":"
                  << (options.maintenance.clean_segments ? "true" : "false")
                  << ",\"cleaner_min_interval_seconds\":"
                  << options.maintenance.cleaner_min_interval_seconds
                  << ",\"cleaner_max_candidate_segments\":"
                  << options.maintenance.cleaner_max_candidate_segments
                  << ",\"cleaner_max_sidecar_scan_bytes\":"
                  << options.maintenance.cleaner_max_sidecar_scan_bytes
                  << ",\"cleaner_max_source_bytes\":"
                  << options.maintenance.cleaner_max_source_bytes
                  << ",\"cleaner_max_output_bytes\":"
                  << options.maintenance.cleaner_max_output_bytes
                  << ",\"cleaner_max_memory_bytes\":"
                  << options.maintenance.cleaner_max_memory_bytes
                  << ",\"cleaner_min_garbage_rows\":"
                  << options.maintenance.cleaner_min_garbage_rows
                  << ",\"cleaner_min_reclaimable_bytes\":"
                  << options.maintenance.cleaner_min_reclaimable_bytes
                  << ",\"cleaner_min_garbage_ratio_ppm\":"
                  << options.maintenance.cleaner_min_garbage_ratio_ppm
                  << ",\"reclaim_orphans\":"
                  << (options.maintenance.reclaim_orphans
                          ? "true" : "false")
                  << ",\"orphan_min_age_seconds\":"
                  << options.maintenance.orphan_min_age_ns / 1'000'000'000ULL
                  << ",\"orphan_max_files\":"
                  << options.maintenance.orphan_max_files
                  << ",\"orphan_max_bytes\":"
                  << options.maintenance.orphan_max_bytes << "}"
                  << ",\"admission\":{\"max_concurrent_calls\":"
                  << options.grpc.max_concurrent_calls
                  << ",\"max_concurrent_scans\":"
                  << options.grpc.max_concurrent_scans
                  << ",\"max_unary_inflight_bytes\":"
                  << options.grpc.max_inflight_bytes
                  << ",\"max_scan_inflight_bytes\":"
                  << options.grpc.max_scan_inflight_bytes
                  << ",\"max_grpc_memory_bytes\":"
                  << options.grpc.max_grpc_memory_bytes
                  << ",\"scan_max_duration_millis\":"
                  << options.grpc.scan_max_duration_millis
                  << ",\"scan_idle_timeout_millis\":"
                  << options.grpc.scan_idle_timeout_millis << "}"
                  << ",\"shutdown\":{\"grace_millis\":"
                  << options.grpc.shutdown_grace_millis
                  << ",\"hard_timeout_millis\":"
                  << options.grpc.shutdown_hard_timeout_millis << "}"
                  << ",\"reader\":{\"max_read_range_bytes\":"
                  << options.reader.max_read_range_bytes
                  << ",\"max_gap_bytes\":" << options.reader.max_gap_bytes
                  << ",\"io_workers\":" << options.reader.io_workers
                  << ",\"io_queue_capacity\":"
                  << options.reader.io_queue_capacity
                  << ",\"io_ranges_per_task\":"
                  << options.reader.io_ranges_per_task
                  << ",\"io_backend\":\""
                  << reader_backend_name(options.reader.io_backend)
                  << "\",\"io_uring_queue_depth\":"
                  << options.reader.io_uring_queue_depth
                  << ",\"io_uring_workers\":"
                  << options.reader.io_uring_workers << "}"
                  << ",\"tuning\":{\"mode\":\""
                  << options.engine.tuning.mode
                  << "\",\"source\":\"" << options.engine.tuning.source
                  << "\",\"status\":\"" << options.engine.tuning.status
                  << "\",\"reuse_level\":\""
                  << options.engine.tuning.reuse_level
                  << "\",\"explicit_overrides\":"
                  << (options.engine.tuning.explicit_overrides
                          ? "true" : "false")
                  << "}"
                  << "}\n"
                  << std::flush;

#if defined(THROWDB_GRPC_TEST_HOOKS)
        if (const char* const terminal =
                std::getenv("THROWDB_TEST_TERMINAL_AFTER_READY");
            terminal != nullptr && std::string_view(terminal) == "1") {
            server.report_terminal_fault();
        }
#endif

        while (stop_requested == 0 && !server.terminal_faulted()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ShutdownWatchdog shutdown_watchdog(
            options.grpc.shutdown_hard_timeout_millis);
        server.begin_draining();
        if (maintenance != nullptr) {
            maintenance->request_stop();
        }
#if defined(THROWDB_GRPC_TEST_HOOKS)
        if (const char* const block =
                std::getenv("THROWDB_TEST_BLOCK_AFTER_DRAIN");
            block != nullptr && std::string_view(block) == "1") {
            std::cerr
                << "{\"event\":\"test_shutdown_block_entered\"}\n"
                << std::flush;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::hours(24));
            }
        }
#endif
        server.shutdown();
        if (maintenance != nullptr) {
            // Native lifecycle actions are crash-safe but currently
            // cooperative: an action already in progress is allowed to finish.
            maintenance->join();
        }
        return server.terminal_faulted() ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "throwdbd: " << error.what() << '\n';
        return 1;
    }
}
