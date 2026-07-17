#include "grpc_admission.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "throwdb/errors.hpp"

namespace throwdb::server::detail {
namespace {

void increment_saturating(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) {
        ++value;
    }
}

GrpcScanWatchdog::CancelCallback validated_scan_cancel(
    const std::uint32_t maximum_duration_millis,
    const std::uint32_t idle_timeout_millis,
    GrpcScanWatchdog::CancelCallback cancel) {
    if (maximum_duration_millis == 0 || idle_timeout_millis == 0
        || !cancel) {
        throw InvalidArgumentError(
            "gRPC scan watchdog bounds and callback must be nonzero");
    }
    return cancel;
}

}  // namespace

GrpcAdmissionController::Lease::Lease(
    GrpcAdmissionController& owner,
    const std::uint64_t estimated_bytes,
    const bool scan) noexcept
    : owner_(&owner),
      estimated_bytes_(estimated_bytes),
      scan_(scan) {}

GrpcAdmissionController::Lease::Lease(Lease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      estimated_bytes_(other.estimated_bytes_),
      scan_(other.scan_) {}

GrpcAdmissionController::Lease::~Lease() {
    if (owner_ != nullptr) {
        owner_->release(estimated_bytes_, scan_);
    }
}

GrpcAdmissionController::GrpcAdmissionController(
    const std::uint32_t max_concurrent_calls,
    const std::uint64_t max_inflight_bytes,
    const std::uint32_t max_concurrent_scans,
    const std::uint64_t max_scan_inflight_bytes)
    : max_concurrent_calls_(max_concurrent_calls),
      max_inflight_bytes_(max_inflight_bytes),
      max_concurrent_scans_(max_concurrent_scans),
      max_scan_inflight_bytes_(max_scan_inflight_bytes) {
    if (max_concurrent_calls_ == 0 || max_inflight_bytes_ == 0
        || max_concurrent_scans_ == 0 || max_scan_inflight_bytes_ == 0
        || max_concurrent_scans_ >= max_concurrent_calls_) {
        throw InvalidArgumentError(
            "gRPC admission limits must be nonzero and at least one call "
            "slot must remain outside the scan limit");
    }
}

std::optional<GrpcAdmissionController::Lease>
GrpcAdmissionController::try_acquire(
    const std::uint64_t estimated_bytes,
    const bool scan) {
    const std::lock_guard lock(mutex_);
    const std::uint64_t byte_limit =
        scan ? max_scan_inflight_bytes_ : max_inflight_bytes_;
    const std::uint64_t current_bytes =
        scan ? current_.scan_estimated_bytes : current_.estimated_bytes;
    if (current_.closed || current_.active_calls >= max_concurrent_calls_
        || (scan && current_.active_scans >= max_concurrent_scans_)
        || estimated_bytes > byte_limit
        || current_bytes > byte_limit - estimated_bytes) {
        increment_saturating(
            scan ? current_.rejected_scans
                 : current_.rejected_unary_calls);
        return std::nullopt;
    }
    ++current_.active_calls;
    if (scan) {
        ++current_.active_scans;
        current_.scan_estimated_bytes += estimated_bytes;
        increment_saturating(current_.admitted_scans);
    } else {
        current_.estimated_bytes += estimated_bytes;
        increment_saturating(current_.admitted_unary_calls);
    }
    return Lease(*this, estimated_bytes, scan);
}

GrpcAdmissionSnapshot GrpcAdmissionController::snapshot() const {
    const std::lock_guard lock(mutex_);
    return current_;
}

void GrpcAdmissionController::close() noexcept {
    const std::lock_guard lock(mutex_);
    current_.closed = true;
}

void GrpcAdmissionController::release(
    const std::uint64_t estimated_bytes,
    const bool scan) noexcept {
    const std::lock_guard lock(mutex_);
    --current_.active_calls;
    if (scan) {
        --current_.active_scans;
        current_.scan_estimated_bytes -= estimated_bytes;
    } else {
        current_.estimated_bytes -= estimated_bytes;
    }
}

GrpcScanWatchdog::GrpcScanWatchdog(
    const std::uint32_t maximum_duration_millis,
    const std::uint32_t idle_timeout_millis,
    CancelCallback cancel)
    : maximum_duration_(maximum_duration_millis),
      idle_timeout_(idle_timeout_millis),
      cancel_(validated_scan_cancel(
          maximum_duration_millis,
          idle_timeout_millis,
          std::move(cancel))),
      started_(Clock::now()),
      last_progress_(started_),
      worker_([this] { run(); }) {}

GrpcScanWatchdog::~GrpcScanWatchdog() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void GrpcScanWatchdog::mark_progress() noexcept {
    {
        const std::lock_guard lock(mutex_);
        if (expiry_.load(std::memory_order_acquire)
            != GrpcScanExpiry::None) {
            return;
        }
        last_progress_ = Clock::now();
    }
    condition_.notify_all();
}

GrpcScanExpiry GrpcScanWatchdog::expiry() const noexcept {
    return expiry_.load(std::memory_order_acquire);
}

void GrpcScanWatchdog::run() noexcept {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        const auto maximum_deadline = started_ + maximum_duration_;
        const auto idle_deadline = last_progress_ + idle_timeout_;
        const auto deadline = std::min(maximum_deadline, idle_deadline);
        condition_.wait_until(lock, deadline);
        if (stopping_) {
            return;
        }
        const auto now = Clock::now();
        GrpcScanExpiry result = GrpcScanExpiry::None;
        if (now >= maximum_deadline) {
            result = GrpcScanExpiry::MaximumDuration;
        } else if (now >= last_progress_ + idle_timeout_) {
            result = GrpcScanExpiry::IdleProgress;
        }
        if (result == GrpcScanExpiry::None) {
            continue;
        }
        expiry_.store(result, std::memory_order_release);
        lock.unlock();
        try {
            cancel_();
        } catch (...) {
            // Expiry remains observable even if an injected cancellation hook
            // unexpectedly fails. The real gRPC TryCancel callback is not
            // expected to throw.
        }
        return;
    }
}

}  // namespace throwdb::server::detail
