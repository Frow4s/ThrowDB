#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "grpc_admission.hpp"

namespace {

using throwdb::server::detail::GrpcAdmissionController;
using throwdb::server::detail::GrpcScanExpiry;
using throwdb::server::detail::GrpcScanWatchdog;
using namespace std::chrono_literals;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_call_and_byte_limits_are_fail_fast() {
    GrpcAdmissionController admission(2, 100, 1, 50);
    auto first = admission.try_acquire(40);
    auto second = admission.try_acquire(50);
    require(first.has_value() && second.has_value(),
            "valid calls were not admitted");
    require(!admission.try_acquire(1).has_value(),
            "concurrent-call limit was not enforced");
    require(!admission.try_acquire(101).has_value(),
            "oversized call was admitted");
    require(admission.snapshot().active_calls == 2
                && admission.snapshot().estimated_bytes == 90,
            "admission accounting changed after rejection");
    require(admission.snapshot().admitted_unary_calls == 2
                && admission.snapshot().rejected_unary_calls == 2
                && admission.snapshot().admitted_scans == 0
                && admission.snapshot().rejected_scans == 0,
            "unary admission decisions were not counted separately");

    second.reset();
    require(!admission.try_acquire(61).has_value(),
            "aggregate byte limit was not enforced");
    auto replacement = admission.try_acquire(60);
    require(replacement.has_value(),
            "released bytes did not become available");
}

void test_scans_have_an_independent_limit() {
    GrpcAdmissionController admission(3, 100, 1, 60);
    auto scan = admission.try_acquire(60, true);
    auto get = admission.try_acquire(40, false);
    require(scan.has_value() && get.has_value(),
            "initial scan and data call were not admitted");
    require(!admission.try_acquire(1, true).has_value(),
            "second concurrent scan bypassed its limit");
    auto another_get = admission.try_acquire(60, false);
    require(another_get.has_value(),
            "scan limit incorrectly rejected a non-scan call");

    scan.reset();
    require(admission.snapshot().active_scans == 0,
            "scan lease did not release its scan slot");
    require(admission.snapshot().scan_estimated_bytes == 0
                && admission.snapshot().estimated_bytes == 100,
            "scan bytes leaked into the unary admission pool");
    require(admission.snapshot().admitted_unary_calls == 2
                && admission.snapshot().rejected_unary_calls == 0
                && admission.snapshot().admitted_scans == 1
                && admission.snapshot().rejected_scans == 1,
            "scan admission decisions were not counted separately");
}

void test_moved_lease_releases_once() {
    GrpcAdmissionController admission(2, 10, 1, 10);
    {
        auto original = admission.try_acquire(10);
        require(original.has_value(), "initial lease was rejected");
        GrpcAdmissionController::Lease moved(std::move(*original));
        original.reset();
        require(admission.snapshot().active_calls == 1,
                "moving a lease released it early");
    }
    require(admission.snapshot().active_calls == 0,
            "moved lease did not release its slot");
}

void test_close_rejects_new_calls_and_preserves_existing_leases() {
    GrpcAdmissionController admission(2, 100, 1, 50);
    auto existing = admission.try_acquire(40);
    require(existing.has_value(), "initial lease was rejected");

    admission.close();
    const auto closed = admission.snapshot();
    require(closed.closed, "closed admission did not report its state");
    require(closed.active_calls == 1 && closed.estimated_bytes == 40,
            "closing admission invalidated an executing lease");
    require(!admission.try_acquire(1).has_value(),
            "closed admission accepted a new call");
    require(admission.snapshot().admitted_unary_calls == 1
                && admission.snapshot().rejected_unary_calls == 1,
            "closed admission decisions were not counted");

    existing.reset();
    require(admission.snapshot().active_calls == 0,
            "existing lease did not drain after admission closed");
}

void test_scan_watchdog_enforces_maximum_duration() {
    std::mutex mutex;
    std::condition_variable condition;
    bool cancelled = false;
    GrpcScanWatchdog watchdog(40, 1'000, [&] {
        {
            const std::lock_guard lock(mutex);
            cancelled = true;
        }
        condition.notify_all();
    });

    std::unique_lock lock(mutex);
    require(
        condition.wait_for(lock, 1s, [&] { return cancelled; }),
        "scan maximum-duration watchdog did not cancel");
    lock.unlock();
    require(
        watchdog.expiry() == GrpcScanExpiry::MaximumDuration,
        "scan watchdog reported the wrong maximum-duration reason");
}

void test_scan_watchdog_resets_idle_progress_deadline() {
    std::mutex mutex;
    std::condition_variable condition;
    bool cancelled = false;
    GrpcScanWatchdog watchdog(2'000, 300, [&] {
        {
            const std::lock_guard lock(mutex);
            cancelled = true;
        }
        condition.notify_all();
    });

    std::this_thread::sleep_for(20ms);
    watchdog.mark_progress();
    std::unique_lock lock(mutex);
    require(
        !condition.wait_for(lock, 100ms, [&] { return cancelled; }),
        "scan idle watchdog ignored recent transport progress");
    require(
        condition.wait_for(lock, 1s, [&] { return cancelled; }),
        "scan idle watchdog did not cancel after progress stopped");
    lock.unlock();
    require(
        watchdog.expiry() == GrpcScanExpiry::IdleProgress,
        "scan watchdog reported the wrong idle-progress reason");
}

}  // namespace

int main() {
    try {
        test_call_and_byte_limits_are_fail_fast();
        test_scans_have_an_independent_limit();
        test_moved_lease_releases_once();
        test_close_rejects_new_calls_and_preserves_existing_leases();
        test_scan_watchdog_enforces_maximum_duration();
        test_scan_watchdog_resets_idle_progress_deadline();
        std::cout << "gRPC admission tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gRPC admission test failure: " << error.what() << '\n';
        return 1;
    }
}
