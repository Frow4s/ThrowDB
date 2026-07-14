#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/read_scheduler.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Actual, typename Expected>
void require_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string& message) {
    if (!(actual == expected)) {
        fail(message);
    }
}

template <typename Exception, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
    try {
        std::forward<Callback>(callback)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        fail(message + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        fail(message + " (wrong non-standard exception)");
    }
    fail(message + " (no exception)");
}

[[nodiscard]] throwdb::detail::ReadWorkUnit work(
    const std::uint64_t token,
    const std::uint64_t ranges = 1,
    const std::uint64_t bytes = 4096) {
    return throwdb::detail::ReadWorkUnit{token, ranges, bytes};
}

[[nodiscard]] throwdb::detail::ReadJobSpec job(
    std::vector<throwdb::detail::ReadWorkUnit> units,
    const std::uint32_t weight = 1,
    const std::optional<std::uint64_t> deadline = std::nullopt) {
    return throwdb::detail::ReadJobSpec{
        std::move(units), weight, deadline};
}

[[nodiscard]] throwdb::detail::ReadSchedulerOptions options(
    const std::uint64_t dispatches,
    const std::uint64_t ranges,
    const std::uint64_t bytes,
    const std::uint64_t quantum = 4096,
    const std::uint32_t max_weight = 4) {
    return throwdb::detail::ReadSchedulerOptions{
        throwdb::detail::ReadSchedulerBudget{
            dispatches, ranges, bytes},
        quantum,
        max_weight};
}

class ManualBackend {
public:
    void accept(std::vector<throwdb::detail::ReadDispatch> dispatches) {
        for (throwdb::detail::ReadDispatch& dispatch : dispatches) {
            pending_.push_back(std::move(dispatch));
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return pending_.size();
    }

    [[nodiscard]] const throwdb::detail::ReadDispatch& at(
        const std::size_t index) const {
        return pending_.at(index);
    }

    [[nodiscard]] throwdb::detail::ReadDispatch pop_front() {
        if (pending_.empty()) {
            fail("manual backend has no dispatch to complete");
        }
        throwdb::detail::ReadDispatch result = pending_.front();
        pending_.pop_front();
        return result;
    }

    throwdb::detail::ReadDispatch complete_front(
        throwdb::detail::ReadScheduler& scheduler,
        const throwdb::detail::ReadDispatchStatus status =
            throwdb::detail::ReadDispatchStatus::Succeeded,
        const std::uint64_t failure_code = 0) {
        const throwdb::detail::ReadDispatch dispatch = pop_front();
        scheduler.complete(dispatch.dispatch_id, status, failure_code);
        return dispatch;
    }

private:
    std::deque<throwdb::detail::ReadDispatch> pending_;
};

void require_normal_budget(
    const throwdb::detail::ReadSchedulerSnapshot& snapshot,
    const throwdb::detail::ReadSchedulerOptions& configuration,
    const std::string& context) {
    require(
        snapshot.inflight_dispatches
            <= configuration.budget.max_inflight_dispatches,
        context + ": dispatch credit exceeded");
    require(
        snapshot.inflight_ranges
            <= configuration.budget.max_inflight_ranges,
        context + ": range credit exceeded");
    require(
        snapshot.inflight_bytes
            <= configuration.budget.max_inflight_bytes,
        context + ": byte credit exceeded");
}

void test_cost_model_and_input_bounds() {
    using throwdb::detail::read_scheduler_cost;
    require_equal(read_scheduler_cost(1, 1), std::uint64_t{4096},
                  "one tiny range did not pay the IOPS floor");
    require_equal(read_scheduler_cost(1, 4096), std::uint64_t{4096},
                  "one block cost changed");
    require_equal(read_scheduler_cost(1, 4097), std::uint64_t{8192},
                  "byte cost did not round to a 4-KiB block");
    require_equal(read_scheduler_cost(3, 1), std::uint64_t{12288},
                  "three tiny ranges did not pay three IOPS floors");
    require_equal(read_scheduler_cost(2, 20000), std::uint64_t{20480},
                  "large contiguous byte cost was not dominant");
    require_equal(
        read_scheduler_cost(
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max()),
        std::numeric_limits<std::uint64_t>::max(),
        "cost overflow did not saturate");

    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(throwdb::detail::ReadScheduler(
                options(0, 1, 4096)));
        },
        "zero dispatch budget was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(throwdb::detail::ReadScheduler(
                options(1, 0, 4096)));
        },
        "zero range budget was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(throwdb::detail::ReadScheduler(
                options(1, 1, 0)));
        },
        "zero byte budget was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(throwdb::detail::ReadScheduler(
                options(1, 1, 4096, 0)));
        },
        "zero DRR quantum was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [] {
            static_cast<void>(throwdb::detail::ReadScheduler(
                options(
                    1, 1, 4096, 4096,
                    throwdb::detail::kMaximumReadSchedulerWeight + 1)));
        },
        "unbounded scheduler weight was accepted");

    throwdb::detail::ReadScheduler scheduler(options(1, 1, 4096));
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler] {
            static_cast<void>(scheduler.submit(job({work(1)}, 0)));
        },
        "zero job weight was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler] {
            static_cast<void>(scheduler.submit(job({work(1, 0, 4096)})));
        },
        "zero-range work was accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler] {
            static_cast<void>(scheduler.submit(job({work(1, 1, 0)})));
        },
        "zero-byte work was accepted");

    const auto empty = scheduler.submit(job({}));
    const auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "empty job did not complete immediately");
    require_equal(completions[0].job_id, empty,
                  "empty completion names the wrong job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Succeeded,
        "empty job did not succeed");
    require_equal(completions[0].total_dispatches, std::uint64_t{0},
                  "empty job reported physical work");

    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    throwdb::detail::ReadScheduler large_budget(
        options(maximum, maximum, maximum));
    static_cast<void>(large_budget.submit(job({work(2)})));
    const auto one_dispatch = large_budget.pump(0);
    require_equal(one_dispatch.size(), std::size_t{1},
                  "large configured budget did not stay queue-bounded");
    large_budget.complete(
        one_dispatch[0].dispatch_id,
        throwdb::detail::ReadDispatchStatus::Succeeded);
    require_equal(large_budget.take_completions().size(), std::size_t{1},
                  "large-budget job did not complete");
}

void test_fifo_and_solo_work_conserving_burst() {
    const auto configuration = options(3, 3, 3 * 4096);
    throwdb::detail::ReadScheduler scheduler(configuration);
    ManualBackend backend;
    const auto id = scheduler.submit(job({
        work(10), work(11), work(12), work(13)}));

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{3},
                  "solo request did not fill global dispatch credits");
    require_equal(backend.at(0).user_token, std::uint64_t{10},
                  "solo FIFO token 0 changed");
    require_equal(backend.at(1).user_token, std::uint64_t{11},
                  "solo FIFO token 1 changed");
    require_equal(backend.at(2).user_token, std::uint64_t{12},
                  "solo FIFO token 2 changed");
    require_normal_budget(scheduler.snapshot(), configuration, "solo burst");

    const auto first = backend.complete_front(scheduler);
    const auto second = backend.complete_front(scheduler);
    const auto third = backend.complete_front(scheduler);
    require(scheduler.take_completions().empty(),
            "job completed before its pending FIFO unit ran");

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{1},
                  "final FIFO unit was not issued");
    require_equal(backend.at(0).user_token, std::uint64_t{13},
                  "final FIFO token changed");
    const auto last = backend.complete_front(scheduler);
    const auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "successful job emitted the wrong completion count");
    require_equal(completions[0].job_id, id,
                  "successful completion names the wrong job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Succeeded,
        "successful job has the wrong outcome");
    require_equal(completions[0].total_dispatches, std::uint64_t{4},
                  "successful job reports the wrong dispatch count");
    require_equal(completions[0].completed_dispatches, std::uint64_t{4},
                  "successful job reports the wrong completed count");
    require(scheduler.take_completions().empty(),
            "successful job emitted completion twice");
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler, first] {
            scheduler.complete(
                first.dispatch_id,
                throwdb::detail::ReadDispatchStatus::Succeeded);
        },
        "duplicate dispatch completion was accepted");
    static_cast<void>(second);
    static_cast<void>(third);
    static_cast<void>(last);
}

void test_drr_no_starvation_and_bounded_weights() {
    const auto configuration = options(2, 2, 2 * 4096, 4096, 2);
    throwdb::detail::ReadScheduler scheduler(configuration);
    ManualBackend backend;
    const auto large = scheduler.submit(job({
        work(100), work(101), work(102), work(103)}));
    const auto small = scheduler.submit(job({work(200)}));

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{2},
                  "DRR did not use available credits");
    require_equal(backend.at(0).job_id, large,
                  "first DRR visit changed job order");
    require_equal(backend.at(1).job_id, small,
                  "small job was starved behind the large FIFO");
    const auto large_first = backend.pop_front();
    const auto small_only = backend.pop_front();
    scheduler.complete(
        small_only.dispatch_id,
        throwdb::detail::ReadDispatchStatus::Succeeded);
    const auto small_completion = scheduler.take_completions();
    require_equal(small_completion.size(), std::size_t{1},
                  "small job did not finish independently");
    require_equal(small_completion[0].job_id, small,
                  "wrong job completed ahead of the large request");
    scheduler.complete(
        large_first.dispatch_id,
        throwdb::detail::ReadDispatchStatus::Succeeded);

    while (scheduler.snapshot().runnable_jobs != 0
           || scheduler.snapshot().inflight_dispatches != 0) {
        backend.accept(scheduler.pump(0));
        while (backend.size() != 0) {
            static_cast<void>(backend.complete_front(scheduler));
        }
    }
    const auto large_completion = scheduler.take_completions();
    require_equal(large_completion.size(), std::size_t{1},
                  "large job did not eventually complete");
    require_equal(large_completion[0].job_id, large,
                  "eventual DRR completion names the wrong job");

    throwdb::detail::ReadScheduler weighted(
        options(2, 2, 2 * 8192, 4096, 2));
    const auto weight_one = weighted.submit(job({work(300, 1, 8192)}, 1));
    const auto weight_two = weighted.submit(job({work(400, 1, 8192)}, 2));
    const auto dispatches = weighted.pump(0);
    require_equal(dispatches.size(), std::size_t{2},
                  "weighted DRR did not make bounded progress");
    require_equal(dispatches[0].job_id, weight_two,
                  "weight-two job did not earn its cost first");
    require_equal(dispatches[1].job_id, weight_one,
                  "weight-one job disappeared after a deferred round");
    require_throws<throwdb::InvalidArgumentError>(
        [&weighted] {
            static_cast<void>(weighted.submit(job({work(500)}, 3)));
        },
        "job exceeded configured maximum weight");
}

void test_fast_forward_skips_adversarial_empty_round_count() {
    constexpr std::uint64_t huge_cost = 1ULL << 60;
    throwdb::detail::ReadScheduler scheduler(options(
        2,
        2,
        std::numeric_limits<std::uint64_t>::max(),
        4096));
    const auto first = scheduler.submit(job({work(1, 1, huge_cost)}));
    const auto second = scheduler.submit(job({work(2, 1, huge_cost)}));

    require(scheduler.pump(0).empty(),
            "ordinary pump unexpectedly skipped its documented empty round");
    const auto dispatches = scheduler.pump_until_blocked(0);
    require_equal(dispatches.size(), std::size_t{2},
                  "fast-forward did not fill available dispatch credits");
    require_equal(dispatches[0].job_id, first,
                  "fast-forward changed equal-weight DRR order");
    require_equal(dispatches[1].job_id, second,
                  "fast-forward starved the second equal-weight job");
    for (const auto& dispatch : dispatches) {
        scheduler.complete(
            dispatch.dispatch_id,
            throwdb::detail::ReadDispatchStatus::Succeeded);
    }
    require_equal(scheduler.take_completions().size(), std::size_t{2},
                  "fast-forwarded jobs did not complete");
}

void test_fast_forward_matches_repeated_rounds_across_units() {
    const auto configuration = options(
        1,
        1,
        std::numeric_limits<std::uint64_t>::max(),
        4096,
        2);
    throwdb::detail::ReadScheduler reference(configuration);
    throwdb::detail::ReadScheduler fast(configuration);
    const auto first_spec = job({
        work(10, 1, 8192),
        work(11, 1, 8192),
        work(12, 1, 8192),
    });
    const auto second_spec = job({
        work(20, 1, 12288),
        work(21, 1, 12288),
        work(22, 1, 12288),
    }, 2);
    static_cast<void>(reference.submit(first_spec));
    static_cast<void>(reference.submit(second_spec));
    static_cast<void>(fast.submit(first_spec));
    static_cast<void>(fast.submit(second_spec));

    for (std::size_t completed = 0; completed != 6; ++completed) {
        std::vector<throwdb::detail::ReadDispatch> expected;
        for (std::size_t rounds = 0; rounds != 16 && expected.empty();
             ++rounds) {
            expected = reference.pump(0);
        }
        const auto actual = fast.pump_until_blocked(0);
        require_equal(expected.size(), std::size_t{1},
                      "reference DRR did not make bounded progress");
        require_equal(actual.size(), std::size_t{1},
                      "fast-forward DRR did not make bounded progress");
        require_equal(actual[0].job_id, expected[0].job_id,
                      "fast-forward changed later job order");
        require_equal(actual[0].user_token, expected[0].user_token,
                      "fast-forward changed FIFO/leftover behavior");
        reference.complete(
            expected[0].dispatch_id,
            throwdb::detail::ReadDispatchStatus::Succeeded);
        fast.complete(
            actual[0].dispatch_id,
            throwdb::detail::ReadDispatchStatus::Succeeded);
    }
    require_equal(reference.take_completions().size(), std::size_t{2},
                  "reference multi-unit jobs did not complete");
    require_equal(fast.take_completions().size(), std::size_t{2},
                  "fast-forward multi-unit jobs did not complete");
}

void test_global_credit_dimensions() {
    const auto configuration = options(3, 4, 8192, 8192);
    throwdb::detail::ReadScheduler scheduler(configuration);
    ManualBackend backend;
    static_cast<void>(scheduler.submit(job({work(1, 2, 4096)})));
    static_cast<void>(scheduler.submit(job({work(2, 2, 4096)})));
    static_cast<void>(scheduler.submit(job({work(3, 2, 4096)})));

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{2},
                  "range credits admitted a third dispatch");
    auto snapshot = scheduler.snapshot();
    require_equal(snapshot.inflight_ranges, std::uint64_t{4},
                  "in-flight range accounting is wrong");
    require_equal(snapshot.inflight_bytes, std::uint64_t{8192},
                  "in-flight byte accounting is wrong");
    require_normal_budget(snapshot, configuration, "range-limited pump");

    static_cast<void>(backend.complete_front(scheduler));
    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{2},
                  "returned credits did not admit exactly one replacement");
    snapshot = scheduler.snapshot();
    require_normal_budget(snapshot, configuration, "replacement pump");
    while (backend.size() != 0) {
        static_cast<void>(backend.complete_front(scheduler));
    }
    require_equal(scheduler.take_completions().size(), std::size_t{3},
                  "credit-limited jobs did not all complete");

    const auto byte_configuration = options(3, 3, 10'000, 8192);
    throwdb::detail::ReadScheduler byte_scheduler(byte_configuration);
    static_cast<void>(byte_scheduler.submit(job({work(10, 1, 7000)})));
    static_cast<void>(byte_scheduler.submit(job({work(11, 1, 7000)})));
    const auto byte_dispatches = byte_scheduler.pump(0);
    require_equal(byte_dispatches.size(), std::size_t{1},
                  "byte credits admitted incompatible dispatches");
    require_normal_budget(
        byte_scheduler.snapshot(), byte_configuration, "byte-limited pump");
}

void test_oversized_single_escape_is_exclusive() {
    const auto configuration = options(4, 2, 8192);
    throwdb::detail::ReadScheduler scheduler(configuration);
    ManualBackend backend;
    const auto oversized = scheduler.submit(job({work(1, 3, 12288)}));
    const auto normal = scheduler.submit(job({work(2)}));

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{1},
                  "oversized unit ran before an affordable peer");
    require_equal(backend.at(0).job_id, normal,
                  "wrong job won the first affordable DRR turn");
    require(!backend.at(0).oversized_single,
            "ordinary dispatch was marked oversized");
    static_cast<void>(backend.complete_front(scheduler));
    static_cast<void>(scheduler.take_completions());

    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{1},
                  "oversized escape did not make progress on an empty domain");
    require_equal(backend.at(0).job_id, oversized,
                  "oversized escape names the wrong job");
    require(backend.at(0).oversized_single,
            "over-budget dispatch was not marked as the escape");
    auto snapshot = scheduler.snapshot();
    require_equal(snapshot.inflight_dispatches, std::uint64_t{1},
                  "oversized escape was not a single dispatch");
    require(snapshot.inflight_ranges > configuration.budget.max_inflight_ranges,
            "oversized range count did not exercise the escape");
    require(snapshot.inflight_bytes > configuration.budget.max_inflight_bytes,
            "oversized bytes did not exercise the escape");

    static_cast<void>(scheduler.submit(job({work(3)})));
    require(scheduler.pump(0).empty(),
            "normal work overlapped an oversized-single dispatch");
    static_cast<void>(backend.complete_front(scheduler));
    static_cast<void>(scheduler.take_completions());
    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{1},
                  "normal work did not resume after oversized drain");
    require(!backend.at(0).oversized_single,
            "resumed normal work was marked oversized");
}

void test_cancellation_and_deadline_before_dispatch() {
    throwdb::detail::ReadScheduler scheduler(options(2, 2, 8192));
    const auto cancelled = scheduler.submit(job({work(1)}));
    require(scheduler.cancel(cancelled),
            "queued cancellation was not accepted");
    require(scheduler.pump(0).empty(),
            "cancelled queued job issued physical work");
    auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "queued cancellation emitted the wrong event count");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Cancelled,
        "queued cancellation has the wrong outcome");
    require_equal(completions[0].total_dispatches, std::uint64_t{0},
                  "queued cancellation reports physical work");
    require(!scheduler.cancel(cancelled),
            "collected terminal job accepted cancellation again");

    const auto deadline = scheduler.submit(job({work(2)}, 1, 10), 0);
    require(scheduler.pump(10).empty(),
            "expired queued job issued physical work");
    completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "deadline emitted the wrong event count");
    require_equal(completions[0].job_id, deadline,
                  "deadline completion names the wrong job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::DeadlineExceeded,
        "deadline completion has the wrong outcome");

    static_cast<void>(scheduler.submit(job({work(3)}, 1, 20), 20));
    completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "already-expired submit did not complete immediately");
    require_equal(completions[0].total_dispatches, std::uint64_t{0},
                  "already-expired submit issued work");
}

void test_cancel_and_deadline_drain_inflight_work() {
    throwdb::detail::ReadScheduler scheduler(options(2, 2, 8192));
    ManualBackend backend;
    const auto cancelled = scheduler.submit(job({
        work(1), work(2), work(3)}));
    backend.accept(scheduler.pump(0));
    require_equal(backend.size(), std::size_t{2},
                  "cancel-drain fixture has the wrong in-flight count");
    require(scheduler.cancel(cancelled),
            "in-flight cancellation was not accepted");
    require(!scheduler.cancel(cancelled),
            "draining cancellation replaced the first terminal cause");
    require(scheduler.take_completions().empty(),
            "cancelled job completed before in-flight drain");
    require_equal(scheduler.snapshot().draining_jobs, std::uint64_t{1},
                  "cancelled job was not reported as draining");
    static_cast<void>(backend.complete_front(scheduler));
    require(scheduler.take_completions().empty(),
            "cancelled job completed after only a partial drain");
    static_cast<void>(backend.complete_front(
        scheduler, throwdb::detail::ReadDispatchStatus::Failed, 99));
    auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "cancelled job emitted the wrong completion count");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Cancelled,
        "later backend failure replaced the earlier cancellation");
    require_equal(completions[0].total_dispatches, std::uint64_t{2},
                  "cancelled job issued pending work after cancellation");
    require_equal(completions[0].completed_dispatches, std::uint64_t{2},
                  "cancelled job did not drain every issued dispatch");

    throwdb::detail::ReadScheduler deadline_scheduler(options(2, 2, 8192));
    ManualBackend deadline_backend;
    static_cast<void>(deadline_scheduler.submit(
        job({work(10), work(11), work(12)}, 1, 5), 0));
    deadline_backend.accept(deadline_scheduler.pump(0));
    require(deadline_scheduler.pump(5).empty(),
            "expired in-flight job issued pending work");
    require(deadline_scheduler.take_completions().empty(),
            "deadline completed before in-flight drain");
    while (deadline_backend.size() != 0) {
        static_cast<void>(deadline_backend.complete_front(deadline_scheduler));
    }
    completions = deadline_scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "drained deadline emitted the wrong event count");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::DeadlineExceeded,
        "drained deadline has the wrong outcome");
}

void test_error_stops_pending_and_completes_exactly_once() {
    throwdb::detail::ReadScheduler scheduler(options(2, 2, 8192));
    ManualBackend backend;
    const auto id = scheduler.submit(job({work(1), work(2), work(3)}));
    backend.accept(scheduler.pump(0));
    const auto failed = backend.pop_front();
    const auto draining = backend.pop_front();
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler, failed] {
            scheduler.complete(
                failed.dispatch_id,
                static_cast<throwdb::detail::ReadDispatchStatus>(0));
        },
        "unknown completion status was accepted");
    scheduler.complete(
        failed.dispatch_id,
        throwdb::detail::ReadDispatchStatus::Failed,
        77);
    require(scheduler.pump(0).empty(),
            "failed job issued its remaining pending unit");
    require(scheduler.take_completions().empty(),
            "failed job completed before another in-flight unit drained");
    scheduler.complete(
        draining.dispatch_id,
        throwdb::detail::ReadDispatchStatus::Succeeded);
    const auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "failed job emitted the wrong event count");
    require_equal(completions[0].job_id, id,
                  "failure completion names the wrong job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Failed,
        "backend failure has the wrong job outcome");
    require_equal(completions[0].failure_code, std::uint64_t{77},
                  "first failure code was not retained");
    require_equal(completions[0].total_dispatches, std::uint64_t{2},
                  "failed job dispatched work after its first error");
    require_equal(completions[0].completed_dispatches, std::uint64_t{2},
                  "failed job did not drain every in-flight dispatch");
    require(scheduler.take_completions().empty(),
            "failed job emitted completion twice");
    require_throws<throwdb::InvalidArgumentError>(
        [&scheduler, failed] {
            scheduler.complete(
                failed.dispatch_id,
                throwdb::detail::ReadDispatchStatus::Succeeded);
        },
        "failed dispatch accepted an exactly-twice completion");
}

void test_deterministic_shutdown_drains_only_issued_work() {
    throwdb::detail::ReadScheduler scheduler(options(1, 1, 4096));
    ManualBackend backend;
    const auto issued = scheduler.submit(job({work(1), work(2)}));
    backend.accept(scheduler.pump(0));
    const auto queued = scheduler.submit(job({work(3)}));

    scheduler.shutdown();
    auto snapshot = scheduler.snapshot();
    require(snapshot.stopping, "shutdown did not mark the scheduler stopping");
    require_equal(snapshot.draining_jobs, std::uint64_t{1},
                  "shutdown did not retain exactly one draining job");
    require(scheduler.pump(0).empty(),
            "shutdown scheduler issued new work");
    auto completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "queued shutdown job did not complete immediately");
    require_equal(completions[0].job_id, queued,
                  "shutdown completed the wrong queued job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Shutdown,
        "queued shutdown job has the wrong outcome");
    require_equal(completions[0].total_dispatches, std::uint64_t{0},
                  "queued shutdown job reports issued work");

    static_cast<void>(backend.complete_front(scheduler));
    completions = scheduler.take_completions();
    require_equal(completions.size(), std::size_t{1},
                  "issued shutdown job did not complete after drain");
    require_equal(completions[0].job_id, issued,
                  "drained shutdown completion names the wrong job");
    require_equal(
        completions[0].outcome,
        throwdb::detail::ReadJobOutcome::Shutdown,
        "drained shutdown job has the wrong outcome");
    require_equal(completions[0].total_dispatches, std::uint64_t{1},
                  "shutdown issued pending work after stop");
    require_equal(completions[0].completed_dispatches, std::uint64_t{1},
                  "shutdown did not account for the drained dispatch");

    scheduler.shutdown();
    require(scheduler.take_completions().empty(),
            "idempotent shutdown emitted duplicate completions");
    require_throws<throwdb::StorageError>(
        [&scheduler] {
            static_cast<void>(scheduler.submit(job({work(4)})));
        },
        "stopped scheduler accepted a new job");
    snapshot = scheduler.snapshot();
    require_equal(snapshot.inflight_dispatches, std::uint64_t{0},
                  "shutdown left a dispatch in flight");
    require_equal(snapshot.runnable_jobs, std::uint64_t{0},
                  "shutdown left a runnable job");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 11> tests{{
        {"cost_model_and_input_bounds", test_cost_model_and_input_bounds},
        {"fifo_and_solo_work_conserving_burst",
         test_fifo_and_solo_work_conserving_burst},
        {"drr_no_starvation_and_bounded_weights",
         test_drr_no_starvation_and_bounded_weights},
        {"fast_forward_skips_adversarial_empty_round_count",
         test_fast_forward_skips_adversarial_empty_round_count},
        {"fast_forward_matches_repeated_rounds_across_units",
         test_fast_forward_matches_repeated_rounds_across_units},
        {"global_credit_dimensions", test_global_credit_dimensions},
        {"oversized_single_escape_is_exclusive",
         test_oversized_single_escape_is_exclusive},
        {"cancellation_and_deadline_before_dispatch",
         test_cancellation_and_deadline_before_dispatch},
        {"cancel_and_deadline_drain_inflight_work",
         test_cancel_and_deadline_drain_inflight_work},
        {"error_stops_pending_and_completes_exactly_once",
         test_error_stops_pending_and_completes_exactly_once},
        {"deterministic_shutdown_drains_only_issued_work",
         test_deterministic_shutdown_drains_only_issued_work},
    }};

    std::size_t failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name
                      << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " read scheduler test(s) failed\n";
        return 1;
    }
    return 0;
}
