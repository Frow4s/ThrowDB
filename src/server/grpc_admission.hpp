#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace throwdb::server::detail {

struct GrpcAdmissionSnapshot {
    // active_calls includes active_scans; the cumulative counters below keep
    // unary and streaming decisions separate so operators can distinguish a
    // scan-pool bottleneck from pressure on the latency-sensitive data plane.
    std::uint32_t active_calls = 0;
    std::uint32_t active_scans = 0;
    // Unary Get/CRUD and streaming reads have deliberately independent pools.
    // A slow stream must never consume the bytes reserved for unary work.
    std::uint64_t estimated_bytes = 0;
    std::uint64_t scan_estimated_bytes = 0;
    std::uint64_t admitted_unary_calls = 0;
    std::uint64_t rejected_unary_calls = 0;
    std::uint64_t admitted_scans = 0;
    std::uint64_t rejected_scans = 0;
    bool closed = false;
};

// A non-blocking, process-wide gate for data-plane RPCs. Control-plane
// Health/Describe calls intentionally do not acquire this gate. A rejected
// caller must surface RESOURCE_EXHAUSTED rather than wait in an unbounded
// application queue.
class GrpcAdmissionController {
public:
    class Lease {
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&&) = delete;
        ~Lease();

    private:
        friend class GrpcAdmissionController;

        Lease(
            GrpcAdmissionController& owner,
            std::uint64_t estimated_bytes,
            bool scan) noexcept;

        GrpcAdmissionController* owner_ = nullptr;
        std::uint64_t estimated_bytes_ = 0;
        bool scan_ = false;
    };

    GrpcAdmissionController(
        std::uint32_t max_concurrent_calls,
        std::uint64_t max_inflight_bytes,
        std::uint32_t max_concurrent_scans,
        std::uint64_t max_scan_inflight_bytes);

    [[nodiscard]] std::optional<Lease> try_acquire(
        std::uint64_t estimated_bytes,
        bool scan = false);

    [[nodiscard]] GrpcAdmissionSnapshot snapshot() const;

    // Stop admitting new calls without invalidating leases already executing.
    // Used by drain and terminal-fault transitions so data-plane work cannot
    // start after readiness has been withdrawn.
    void close() noexcept;

private:
    void release(std::uint64_t estimated_bytes, bool scan) noexcept;

    const std::uint32_t max_concurrent_calls_;
    const std::uint64_t max_inflight_bytes_;
    const std::uint32_t max_concurrent_scans_;
    const std::uint64_t max_scan_inflight_bytes_;
    mutable std::mutex mutex_;
    GrpcAdmissionSnapshot current_;
};

enum class GrpcScanExpiry : std::uint8_t {
    None,
    MaximumDuration,
    IdleProgress,
};

// Transport-independent timer used by synchronous gRPC streaming handlers.
// The injected callback lets the timer be tested deterministically enough
// without manufacturing a live grpc::ServerContext.
class GrpcScanWatchdog final {
public:
    using CancelCallback = std::function<void()>;

    GrpcScanWatchdog(
        std::uint32_t maximum_duration_millis,
        std::uint32_t idle_timeout_millis,
        CancelCallback cancel);
    ~GrpcScanWatchdog();

    GrpcScanWatchdog(const GrpcScanWatchdog&) = delete;
    GrpcScanWatchdog& operator=(const GrpcScanWatchdog&) = delete;

    void mark_progress() noexcept;
    [[nodiscard]] GrpcScanExpiry expiry() const noexcept;

private:
    using Clock = std::chrono::steady_clock;

    void run() noexcept;

    const std::chrono::milliseconds maximum_duration_;
    const std::chrono::milliseconds idle_timeout_;
    const CancelCallback cancel_;
    const Clock::time_point started_;
    Clock::time_point last_progress_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = false;
    std::atomic<GrpcScanExpiry> expiry_{GrpcScanExpiry::None};
    std::thread worker_;
};

}  // namespace throwdb::server::detail
