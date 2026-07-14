#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"
#include "../src/core/io_uring_engine.hpp"
#include "../src/core/io_uring_coordinator.hpp"
#include "../src/core/physical_reader.hpp"

namespace {

using throwdb::detail::IoUringLane;
using throwdb::detail::IoUringLaneFactory;
using throwdb::detail::IoUringReadCoordinator;
using throwdb::detail::IoUringReadOperation;
using throwdb::detail::ReadWorkUnit;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
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

[[nodiscard]] ReadWorkUnit work(
    const std::uint64_t token,
    const std::uint64_t ranges = 1,
    const std::uint64_t bytes = 4096) {
    return ReadWorkUnit{token, ranges, bytes};
}

struct FakeState {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::uint64_t> started_tokens;
    std::vector<std::vector<std::size_t>> operation_sizes;
    std::vector<std::uint32_t> lane_depths;
    std::map<std::size_t, std::thread::id> lane_owners;
    std::set<std::uint64_t> blocked_tokens;
    std::set<std::uint64_t> released_tokens;
    std::set<std::uint64_t> ordinary_error_tokens;
    std::set<std::uint64_t> poison_tokens;
    std::size_t active_calls{0};
    std::size_t max_active_calls{0};
    std::size_t destroyed_lanes{0};
    bool owner_changed{false};
    bool perform_pread{false};
};

class FakeLane final : public IoUringLane {
public:
    FakeLane(std::shared_ptr<FakeState> state, const std::size_t lane_index)
        : state_(std::move(state)), lane_index_(lane_index) {}

    ~FakeLane() override {
        {
            std::lock_guard lock(state_->mutex);
            ++state_->destroyed_lanes;
        }
        state_->changed.notify_all();
    }

    void read_exact(
        const std::span<const IoUringReadOperation> operations) override {
        if (operations.empty()) {
            throw std::runtime_error("fake lane received an empty batch");
        }
        const std::uint64_t token = operations.front().user_index;
        bool ordinary_error = false;
        bool poison = false;
        bool perform_pread = false;
        {
            std::unique_lock lock(state_->mutex);
            state_->started_tokens.push_back(token);
            std::vector<std::size_t> sizes;
            sizes.reserve(operations.size());
            for (const IoUringReadOperation& operation : operations) {
                sizes.push_back(operation.size);
            }
            state_->operation_sizes.push_back(std::move(sizes));
            const auto [position, inserted] = state_->lane_owners.emplace(
                lane_index_, std::this_thread::get_id());
            if (!inserted
                && position->second != std::this_thread::get_id()) {
                state_->owner_changed = true;
            }
            ++state_->active_calls;
            state_->max_active_calls = std::max(
                state_->max_active_calls, state_->active_calls);
            state_->changed.notify_all();
            state_->changed.wait(lock, [this, token] {
                return !state_->blocked_tokens.contains(token)
                    || state_->released_tokens.contains(token);
            });
            ordinary_error = state_->ordinary_error_tokens.contains(token);
            poison = state_->poison_tokens.contains(token);
            perform_pread = state_->perform_pread;
        }

        std::exception_ptr read_error;
        if (perform_pread) {
            try {
                for (const IoUringReadOperation& operation : operations) {
                    std::size_t completed = 0;
                    while (completed != operation.size) {
                        const ssize_t result = ::pread(
                            operation.fd,
                            static_cast<std::byte*>(operation.buffer)
                                + completed,
                            operation.size - completed,
                            static_cast<off_t>(operation.offset + completed));
                        if (result <= 0) {
                            throw throwdb::StorageError(
                                "fake positional read failed");
                        }
                        completed += static_cast<std::size_t>(result);
                    }
                }
            } catch (...) {
                read_error = std::current_exception();
            }
        }

        {
            std::lock_guard lock(state_->mutex);
            --state_->active_calls;
        }
        state_->changed.notify_all();
        if (poison) {
            usable_.store(false, std::memory_order_release);
            throw throwdb::StorageError("fake lane poison");
        }
        if (ordinary_error) {
            throw throwdb::StorageError("fake ordinary read error");
        }
        if (read_error != nullptr) {
            std::rethrow_exception(read_error);
        }
    }

    [[nodiscard]] bool usable() const noexcept override {
        return usable_.load(std::memory_order_acquire);
    }

private:
    std::shared_ptr<FakeState> state_;
    std::size_t lane_index_{0};
    std::atomic<bool> usable_{true};
};

[[nodiscard]] IoUringLaneFactory fake_factory(
    const std::shared_ptr<FakeState>& state) {
    return [state](
               const std::size_t lane_index,
               const std::uint32_t lane_depth) {
        {
            std::lock_guard lock(state->mutex);
            state->lane_depths.push_back(lane_depth);
        }
        return std::make_unique<FakeLane>(state, lane_index);
    };
}

void run_fake_task(const std::uint64_t token, IoUringLane& lane) {
    std::byte destination{};
    const IoUringReadOperation operation{
        -1, &destination, 1, 0, token};
    lane.read_exact(std::span<const IoUringReadOperation>(&operation, 1));
}

template <typename Predicate>
void wait_state(
    const std::shared_ptr<FakeState>& state,
    Predicate predicate,
    const std::string& message) {
    std::unique_lock lock(state->mutex);
    const bool reached = state->changed.wait_for(
        lock, std::chrono::seconds(5), [&] { return predicate(*state); });
    require(reached, message);
}

template <typename Predicate>
void wait_until(Predicate predicate, const std::string& message) {
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail(message);
        }
        std::this_thread::yield();
    }
}

void release_token(
    const std::shared_ptr<FakeState>& state,
    const std::uint64_t token) {
    {
        std::lock_guard lock(state->mutex);
        state->released_tokens.insert(token);
    }
    state->changed.notify_all();
}

[[nodiscard]] std::string exception_message(
    const std::exception_ptr& error) {
    if (error == nullptr) {
        return {};
    }
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& caught) {
        return caught.what();
    } catch (...) {
        return "non-standard exception";
    }
}

struct AsyncCall {
    std::thread thread;
    std::exception_ptr error;
    std::atomic<bool> done{false};
};

template <typename Callback>
void start_async(AsyncCall& call, Callback callback) {
    call.thread = std::thread([&call, callback = std::move(callback)] {
        try {
            callback();
        } catch (...) {
            call.error = std::current_exception();
        }
        call.done.store(true, std::memory_order_release);
    });
}

void join(AsyncCall& call) {
    if (call.thread.joinable()) {
        call.thread.join();
    }
}

void test_validation_empty_and_persistent_lane_reuse() {
    const auto state = std::make_shared<FakeState>();
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(IoUringReadCoordinator(
                0, 1, 4096, fake_factory(state)));
        },
        "zero io_uring workers were accepted");
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(IoUringReadCoordinator(
                2, 1, 4096, fake_factory(state)));
        },
        "queue depth below worker count was accepted");

    {
        IoUringReadCoordinator coordinator(1, 4, 4096, fake_factory(state));
        std::atomic<std::size_t> invoked{0};
        coordinator.execute({}, [&invoked](const std::uint64_t, IoUringLane&) {
            invoked.fetch_add(1, std::memory_order_relaxed);
        });
        require(invoked.load() == 0, "empty work invoked the backend task");
        coordinator.execute(
            {work(1)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
        coordinator.execute(
            {work(2)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
        require(coordinator.worker_count() == 1, "worker getter changed");
        require(
            coordinator.total_queue_depth() == 4,
            "queue-depth getter changed");
    }

    std::lock_guard lock(state->mutex);
    require(
        state->lane_depths == std::vector<std::uint32_t>{4},
        "single persistent lane was reconstructed or got the wrong QD");
    require(
        state->started_tokens == std::vector<std::uint64_t>({1, 2}),
        "one lane did not accept sequential jobs");
    require(state->destroyed_lanes == 1, "persistent lane was not destroyed");
}

void test_single_lane_caller_driver_is_fair_between_jobs() {
    const auto state = std::make_shared<FakeState>();
    state->blocked_tokens.insert(0);
    IoUringReadCoordinator coordinator(1, 4, 4096, fake_factory(state));

    AsyncCall large;
    start_async(large, [&] {
        coordinator.execute(
            {work(0), work(1), work(2)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    });
    wait_state(
        state,
        [](const FakeState& current) {
            return current.started_tokens.size() == 1;
        },
        "large single-lane job did not start");

    AsyncCall small;
    start_async(small, [&] {
        coordinator.execute(
            {work(100)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    });
    wait_until(
        [&] { return coordinator.snapshot().scheduler.runnable_jobs >= 2; },
        "small job was not admitted behind the active large job");
    release_token(state, 0);
    join(large);
    join(small);
    require(large.error == nullptr, "large fair job failed");
    require(small.error == nullptr, "small fair job failed");

    std::lock_guard lock(state->mutex);
    const auto small_position = std::find(
        state->started_tokens.begin(), state->started_tokens.end(), 100);
    const auto last_large = std::find(
        state->started_tokens.begin(), state->started_tokens.end(), 2);
    require(
        small_position != state->started_tokens.end()
            && last_large != state->started_tokens.end()
            && small_position < last_large,
        "small job waited for every chunk of the large job");
}

void test_multi_lane_owners_and_aggregate_credits() {
    {
        const auto state = std::make_shared<FakeState>();
        state->blocked_tokens = {0, 1};
        IoUringReadCoordinator coordinator(2, 8, 4096, fake_factory(state));
        AsyncCall call;
        start_async(call, [&] {
            coordinator.execute(
                {work(0, 4, 4 * 4096), work(1, 4, 4 * 4096)},
                [](const std::uint64_t token, IoUringLane& lane) {
                    run_fake_task(token, lane);
                });
        });
        wait_state(
            state,
            [](const FakeState& current) {
                return current.active_calls == 2;
            },
            "two lanes did not execute concurrently");
        const auto snapshot = coordinator.snapshot().scheduler;
        require(
            snapshot.inflight_dispatches == 2
                && snapshot.inflight_ranges == 8
                && snapshot.inflight_bytes == 8 * 4096,
            "aggregate coordinator credits were accounted incorrectly");
        release_token(state, 0);
        release_token(state, 1);
        join(call);
        require(call.error == nullptr, "two-lane job failed");

        coordinator.execute(
            {work(2), work(3)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
        std::lock_guard lock(state->mutex);
        require(!state->owner_changed, "a ring changed owner threads");
        require(
            state->lane_depths == std::vector<std::uint32_t>({4, 4}),
            "total queue depth was not divided between lanes");
    }

    {
        const auto state = std::make_shared<FakeState>();
        state->blocked_tokens = {10, 11};
        IoUringReadCoordinator coordinator(2, 8, 4096, fake_factory(state));
        AsyncCall call;
        start_async(call, [&] {
            coordinator.execute(
                {work(10, 5, 5 * 4096), work(11, 5, 5 * 4096)},
                [](const std::uint64_t token, IoUringLane& lane) {
                    run_fake_task(token, lane);
                });
        });
        wait_state(
            state,
            [](const FakeState& current) {
                return current.active_calls == 1;
            },
            "first range-heavy chunk did not start");
        const auto snapshot = coordinator.snapshot().scheduler;
        require(
            snapshot.inflight_dispatches == 1
                && snapshot.inflight_ranges == 5
                && snapshot.runnable_jobs == 1,
            "range credits admitted two chunks whose sum exceeded total QD");
        release_token(state, 10);
        release_token(state, 11);
        join(call);
        require(call.error == nullptr, "range-credit test job failed");
        std::lock_guard lock(state->mutex);
        require(
            state->max_active_calls == 1,
            "range budget was exceeded by concurrent lane calls");
    }
}

void test_ordinary_error_does_not_poison_lane_or_domain() {
    const auto state = std::make_shared<FakeState>();
    state->ordinary_error_tokens.insert(7);
    IoUringReadCoordinator coordinator(1, 4, 4096, fake_factory(state));
    require_throws<throwdb::StorageError>(
        [&] {
            coordinator.execute(
                {work(7)},
                [](const std::uint64_t token, IoUringLane& lane) {
                    run_fake_task(token, lane);
                });
        },
        "ordinary read error was not returned to its caller");
    coordinator.execute(
        {work(8)},
        [](const std::uint64_t token, IoUringLane& lane) {
            run_fake_task(token, lane);
        });
    std::lock_guard lock(state->mutex);
    require(
        state->started_tokens == std::vector<std::uint64_t>({7, 8}),
        "ordinary error poisoned a reusable lane");
}

void test_admission_is_bounded_and_fifo() {
    const auto state = std::make_shared<FakeState>();
    state->blocked_tokens = {0, 1, 2, 3};
    IoUringReadCoordinator coordinator(1, 1, 4096, fake_factory(state));
    std::array<AsyncCall, 4> calls;
    const auto start = [&](const std::size_t index) {
        start_async(calls[index], [&, index] {
            coordinator.execute(
                {work(index)},
                [](const std::uint64_t token, IoUringLane& lane) {
                    run_fake_task(token, lane);
                });
        });
    };

    start(0);
    wait_state(
        state,
        [](const FakeState& current) { return current.active_calls == 1; },
        "first bounded-admission job did not start");
    start(1);
    wait_until(
        [&] {
            const auto snapshot = coordinator.snapshot();
            return snapshot.admitted_jobs == 2
                && snapshot.scheduler.runnable_jobs == 1;
        },
        "second job did not fill the workers+QD admission cap");
    start(2);
    wait_until(
        [&] { return coordinator.snapshot().admission_waiters == 1; },
        "third job bypassed or did not reach bounded admission");
    start(3);
    wait_until(
        [&] { return coordinator.snapshot().admission_waiters == 2; },
        "fourth job bypassed or did not reach bounded admission");

    release_token(state, 0);
    wait_state(
        state,
        [](const FakeState& current) {
            return std::find(
                       current.started_tokens.begin(),
                       current.started_tokens.end(),
                       std::uint64_t{1})
                != current.started_tokens.end();
        },
        "second admitted job did not run next");
    wait_until(
        [&] {
            const auto snapshot = coordinator.snapshot();
            return snapshot.admitted_jobs == 2
                && snapshot.admission_waiters == 1;
        },
        "first FIFO waiter was not admitted when one slot opened");
    release_token(state, 1);
    wait_state(
        state,
        [](const FakeState& current) {
            return std::find(
                       current.started_tokens.begin(),
                       current.started_tokens.end(),
                       std::uint64_t{2})
                != current.started_tokens.end();
        },
        "first FIFO waiter did not run before the second");
    release_token(state, 2);
    wait_state(
        state,
        [](const FakeState& current) {
            return std::find(
                       current.started_tokens.begin(),
                       current.started_tokens.end(),
                       std::uint64_t{3})
                != current.started_tokens.end();
        },
        "second FIFO waiter did not run after the first");
    release_token(state, 3);
    for (AsyncCall& call : calls) {
        join(call);
        require(call.error == nullptr, "bounded-admission caller failed");
    }
    std::lock_guard lock(state->mutex);
    require(
        state->started_tokens == std::vector<std::uint64_t>({0, 1, 2, 3}),
        "admission order was not FIFO");
}

void test_poison_propagates_and_waits_for_issued_drain() {
    const auto state = std::make_shared<FakeState>();
    state->blocked_tokens = {1, 2};
    state->poison_tokens.insert(1);
    IoUringReadCoordinator coordinator(2, 8, 4096, fake_factory(state));

    AsyncCall issued;
    start_async(issued, [&] {
        coordinator.execute(
            {work(1), work(2)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    });
    wait_state(
        state,
        [](const FakeState& current) { return current.active_calls == 2; },
        "poison/drain job did not occupy both lanes");

    AsyncCall queued;
    start_async(queued, [&] {
        coordinator.execute(
            {work(9)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    });
    wait_until(
        [&] { return coordinator.snapshot().scheduler.runnable_jobs >= 1; },
        "peer job was not admitted before poison");

    release_token(state, 1);
    wait_until(
        [&] { return queued.done.load(std::memory_order_acquire); },
        "queued peer did not receive the fatal lane error");
    require(
        !issued.done.load(std::memory_order_acquire),
        "issued job returned before its other lane drained");
    release_token(state, 2);
    join(issued);
    join(queued);
    require(
        exception_message(issued.error).find("fake lane poison")
            != std::string::npos,
        "issued job lost the original poison error");
    require(
        exception_message(queued.error).find("fake lane poison")
            != std::string::npos,
        "queued peer did not receive the original poison error");

    std::string future_error;
    try {
        coordinator.execute(
            {work(10)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    } catch (const std::exception& error) {
        future_error = error.what();
    }
    require(
        future_error.find("fake lane poison") != std::string::npos,
        "future admission did not retain the domain's original fatal error");
}

void test_shutdown_waits_for_single_caller_driver() {
    const auto state = std::make_shared<FakeState>();
    state->blocked_tokens.insert(5);
    IoUringReadCoordinator coordinator(1, 4, 4096, fake_factory(state));
    AsyncCall read;
    start_async(read, [&] {
        coordinator.execute(
            {work(5)},
            [](const std::uint64_t token, IoUringLane& lane) {
                run_fake_task(token, lane);
            });
    });
    wait_state(
        state,
        [](const FakeState& current) { return current.active_calls == 1; },
        "single caller-driver did not start");

    AsyncCall stop;
    start_async(stop, [&] { coordinator.shutdown(); });
    wait_until(
        [&] {
            return coordinator.snapshot().scheduler.stopping;
        },
        "shutdown did not stop scheduler admission");
    require(
        !stop.done.load(std::memory_order_acquire),
        "shutdown returned before the caller-driven lane drained");
    require(
        !read.done.load(std::memory_order_acquire),
        "read returned while its lane still held caller references");
    release_token(state, 5);
    join(stop);
    join(read);
    require(stop.error == nullptr, "explicit shutdown threw");
    require(
        exception_message(read.error).find("stopping") != std::string::npos,
        "drained caller did not receive shutdown outcome");
}

void test_construction_failure_releases_created_lanes() {
    const auto state = std::make_shared<FakeState>();
    IoUringLaneFactory factory = [state](
                                      const std::size_t lane_index,
                                      const std::uint32_t) {
        if (lane_index == 1) {
            throw throwdb::StorageError("factory failure");
        }
        return std::make_unique<FakeLane>(state, lane_index);
    };
    require_throws<throwdb::StorageError>(
        [&] {
            static_cast<void>(IoUringReadCoordinator(
                2, 8, 4096, std::move(factory)));
        },
        "lane construction failure was hidden");
    std::lock_guard lock(state->mutex);
    require(
        state->destroyed_lanes == 1,
        "partially constructed lane topology leaked its first lane");
}

void test_fork_rejects_and_abandons_inherited_domain() {
    const auto state = std::make_shared<FakeState>();
    IoUringReadCoordinator coordinator(2, 8, 4096, fake_factory(state));
    const pid_t child = ::fork();
    if (child < 0) {
        throw std::runtime_error("fork failed");
    }
    if (child == 0) {
        bool rejected = false;
        try {
            coordinator.execute(
                {work(1)},
                [](const std::uint64_t token, IoUringLane& lane) {
                    run_fake_task(token, lane);
                });
        } catch (const throwdb::StorageError& error) {
            rejected = std::string(error.what()).find("inherited across fork")
                != std::string::npos;
        } catch (...) {
        }
        coordinator.abandon_after_fork();
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child, "waitpid failed");
    require(
        WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "child used inherited ring/thread state");
    coordinator.execute(
        {work(2)},
        [](const std::uint64_t token, IoUringLane& lane) {
            run_fake_task(token, lane);
        });
}

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
            / ("throwdb-uring-coordinator-test-"
               + std::to_string(::getpid()) + "-"
               + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class ScopedFd {
public:
    explicit ScopedFd(const std::filesystem::path& path)
        : fd_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (fd_ < 0) {
            throw std::runtime_error("cannot open physical-reader fixture");
        }
    }
    ~ScopedFd() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    [[nodiscard]] int get() const noexcept { return fd_; }

private:
    int fd_{-1};
};

void test_physical_reader_direct_and_merged_scatter() {
    constexpr std::uint64_t record_size = 4;
    TempDirectory temporary;
    std::vector<std::byte> source(10 * record_size);
    for (std::size_t index = 0; index != source.size(); ++index) {
        source[index] = static_cast<std::byte>(index + 1);
    }
    const auto path = temporary.path() / "rows.bin";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(
        reinterpret_cast<const char*>(source.data()),
        static_cast<std::streamsize>(source.size()));
    file.close();
    const ScopedFd fd(path);

    const auto state = std::make_shared<FakeState>();
    state->perform_pread = true;
    auto hooks = std::make_shared<throwdb::detail::PhysicalReaderTestHooks>();
    hooks->io_uring_lane_factory = fake_factory(state);
    throwdb::ReaderOptions options;
    options.io_backend = throwdb::IoBackend::IoUring;
    options.io_uring_workers = 1;
    options.io_uring_queue_depth = 2;
    options.max_gap_bytes = record_size;
    options.max_read_range_bytes = 5 * record_size;
    throwdb::detail::PhysicalReader reader(options, hooks);

    const std::vector<throwdb::detail::PhysicalReadRequest> requests{
        {fd.get(), 7 * record_size, 0},
        {fd.get(), 4 * record_size, 1},
        {fd.get(), 1 * record_size, 2},
        {fd.get(), 2 * record_size, 3},
    };
    std::vector<std::byte> output(requests.size() * record_size);
    reader.execute(record_size, requests, output);
    const std::array<std::size_t, 4> rows{7, 4, 1, 2};
    for (std::size_t output_index = 0;
         output_index != rows.size(); ++output_index) {
        require(
            std::equal(
                output.begin() + output_index * record_size,
                output.begin() + (output_index + 1) * record_size,
                source.begin() + rows[output_index] * record_size),
            "io_uring chunk scatter changed output order or bytes");
    }

    std::vector<std::byte> one(record_size);
    reader.execute(
        record_size,
        std::array<throwdb::detail::PhysicalReadRequest, 1>{
            throwdb::detail::PhysicalReadRequest{
                fd.get(), 3 * record_size, 0}},
        one);
    require(
        std::equal(
            one.begin(), one.end(), source.begin() + 3 * record_size),
        "single-range io_uring coordinator path returned wrong bytes");

    std::lock_guard lock(state->mutex);
    require(
        state->operation_sizes.size() == 2,
        "single-range read bypassed the io_uring coordinator lane");
    require(
        state->operation_sizes.front().size() == 2,
        "merged and direct ranges were not issued in one QD-bounded chunk");
}

void test_real_physical_reader_io_uring_when_runtime_allows() {
    if (!throwdb::detail::IoUringEngine::platform_supported()) {
        return;
    }
    constexpr std::uint64_t record_size = 4;
    TempDirectory temporary;
    const std::array<std::byte, 12> source{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
        std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8},
        std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12}};
    const auto path = temporary.path() / "real-ring-rows.bin";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(
        reinterpret_cast<const char*>(source.data()),
        static_cast<std::streamsize>(source.size()));
    file.close();
    const ScopedFd fd(path);

    throwdb::ReaderOptions options;
    options.io_backend = throwdb::IoBackend::IoUring;
    options.io_uring_workers = 1;
    options.io_uring_queue_depth = 2;
    options.max_read_range_bytes = record_size;
    std::unique_ptr<throwdb::detail::PhysicalReader> reader;
    try {
        reader = std::make_unique<throwdb::detail::PhysicalReader>(options);
    } catch (const throwdb::ReaderBackendError&) {
        // Linux containers commonly disable io_uring at runtime. The fake
        // suite above remains exhaustive; exercise the real adapter only when
        // ring construction is actually permitted.
        return;
    }

    const std::array<throwdb::detail::PhysicalReadRequest, 2> requests{
        throwdb::detail::PhysicalReadRequest{fd.get(), record_size, 1},
        throwdb::detail::PhysicalReadRequest{fd.get(), 0, 0}};
    std::array<std::byte, 8> output{};
    reader->execute(record_size, requests, output);
    require(
        std::equal(output.begin(), output.end(), source.begin()),
        "real io_uring coordinator changed physical-reader bytes");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"validation_empty_and_persistent_lane_reuse",
         test_validation_empty_and_persistent_lane_reuse},
        {"single_lane_caller_driver_is_fair_between_jobs",
         test_single_lane_caller_driver_is_fair_between_jobs},
        {"multi_lane_owners_and_aggregate_credits",
         test_multi_lane_owners_and_aggregate_credits},
        {"ordinary_error_does_not_poison_lane_or_domain",
         test_ordinary_error_does_not_poison_lane_or_domain},
        {"admission_is_bounded_and_fifo",
         test_admission_is_bounded_and_fifo},
        {"poison_propagates_and_waits_for_issued_drain",
         test_poison_propagates_and_waits_for_issued_drain},
        {"shutdown_waits_for_single_caller_driver",
         test_shutdown_waits_for_single_caller_driver},
        {"construction_failure_releases_created_lanes",
         test_construction_failure_releases_created_lanes},
        {"fork_rejects_and_abandons_inherited_domain",
         test_fork_rejects_and_abandons_inherited_domain},
        {"physical_reader_direct_and_merged_scatter",
         test_physical_reader_direct_and_merged_scatter},
        {"real_physical_reader_io_uring_when_runtime_allows",
         test_real_physical_reader_io_uring_when_runtime_allows},
    };

    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " tests passed\n";
    return 0;
}
