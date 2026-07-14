#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <poll.h>
#include <signal.h>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "throwdb/native_collection.hpp"
#include "../src/core/file_lock.hpp"
#include "../src/core/mutation_catalog.hpp"
#include "../src/core/native_collection_test_hooks.hpp"
#include "../src/core/native_format.hpp"

namespace {

using namespace std::chrono_literals;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-native-reader-barrier-test-"
                   + std::to_string(timestamp) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

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

void require_bytes_equal(
    const std::span<const std::byte> actual,
    const std::span<const std::byte> expected,
    const std::string& message) {
    require(
        actual.size() == expected.size()
            && std::equal(actual.begin(), actual.end(), expected.begin()),
        message);
}

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t row_count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> result(row_count * record_size);
    for (std::size_t row = 0; row != row_count; ++row) {
        for (std::size_t column = 0; column != record_size; ++column) {
            result[row * record_size + column] = static_cast<std::byte>(
                salt + static_cast<std::uint8_t>(row * 29 + column * 7));
        }
    }
    return result;
}

void build_store(
    const std::filesystem::path& store,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint32_t dimension) {
    throwdb::NativeBuilderOptions options;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 1ULL << 20;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(ids, embeddings);
    builder->finish();
}

[[nodiscard]] throwdb::detail::MutationCatalog catalog_for(
    const std::filesystem::path& store) {
    const throwdb::detail::NativeManifest manifest =
        throwdb::detail::read_native_manifest(store);
    return throwdb::detail::read_current_mutation_catalog(store, manifest);
}

[[nodiscard]] bool write_byte(const int fd, const char value) noexcept {
    while (true) {
        const ssize_t written = ::write(fd, &value, 1);
        if (written == 1) {
            return true;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
}

[[nodiscard]] bool wait_readable(
    const int fd, const std::chrono::milliseconds timeout) noexcept {
    struct pollfd descriptor {
        fd, POLLIN, 0
    };
    int result = -1;
    do {
        result = ::poll(
            &descriptor, 1, static_cast<int>(timeout.count()));
    } while (result < 0 && errno == EINTR);
    return result == 1 && (descriptor.revents & (POLLIN | POLLHUP)) != 0;
}

[[nodiscard]] bool read_byte(
    const int fd,
    char& value,
    const std::chrono::milliseconds timeout) noexcept {
    if (!wait_readable(fd, timeout)) {
        return false;
    }
    while (true) {
        const ssize_t received = ::read(fd, &value, 1);
        if (received == 1) {
            return true;
        }
        if (received < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
}

struct ReaderOpenObservation {
    int command_fd{-1};
    int status_fd{-1};
    const std::filesystem::path* store{nullptr};
    const std::filesystem::path* frozen_path{nullptr};
    const throwdb::detail::MutationCatalog* before{nullptr};
    throwdb::detail::MutationCatalog while_blocked;
    std::exception_ptr error;
    bool command_sent{false};
    bool cross_process_probe_blocked{false};
    bool publisher_started{false};
    bool publisher_finished_while_blocked{false};
    bool current_unchanged{false};
    bool frozen_existed_while_blocked{false};

    void observe() noexcept {
        try {
            command_sent = write_byte(command_fd, 'G');
            char status = 0;
            cross_process_probe_blocked = command_sent
                && read_byte(status_fd, status, 5s) && status == 'B';
            status = 0;
            publisher_started = cross_process_probe_blocked
                && read_byte(status_fd, status, 5s) && status == 'M';

            // The child first proved that this exact lock inode rejects a
            // cross-process nonblocking EX lock. It now enters the real
            // materialization path, whose publication takes blocking EX.
            // Give an accidentally unlocked publisher time to expose itself;
            // a correct publisher cannot complete until this callback returns
            // and the reader releases SH.
            if (publisher_started && wait_readable(status_fd, 100ms)) {
                char completion = 0;
                publisher_finished_while_blocked =
                    read_byte(status_fd, completion, 0ms)
                    && (completion == 'D' || completion == 'E');
            }

            if (publisher_started) {
                while_blocked = catalog_for(*store);
                current_unchanged = while_blocked == *before;
                frozen_existed_while_blocked =
                    std::filesystem::exists(*frozen_path);
            }
        } catch (...) {
            error = std::current_exception();
        }
    }
};

std::atomic<ReaderOpenObservation*> active_reader_observation{nullptr};

void reader_open_hook(
    const throwdb::detail::NativeCollectionTestHookContext context) noexcept {
    if (context.point
        != throwdb::detail::NativeCollectionTestHookPoint::
               ReaderOpenCommitLockAcquired) {
        return;
    }
    ReaderOpenObservation* const observation =
        active_reader_observation.load(std::memory_order_acquire);
    if (observation != nullptr) {
        observation->observe();
    }
}

class ScopedReaderOpenHook {
public:
    explicit ScopedReaderOpenHook(ReaderOpenObservation& observation) {
        active_reader_observation.store(
            &observation, std::memory_order_release);
        throwdb::detail::set_native_collection_test_hook(reader_open_hook);
    }

    ~ScopedReaderOpenHook() {
        throwdb::detail::set_native_collection_test_hook(nullptr);
        active_reader_observation.store(nullptr, std::memory_order_release);
    }

    ScopedReaderOpenHook(const ScopedReaderOpenHook&) = delete;
    ScopedReaderOpenHook& operator=(const ScopedReaderOpenHook&) = delete;
};

[[nodiscard]] bool wait_for_child(
    const pid_t child,
    int& status,
    const std::chrono::milliseconds timeout) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        const pid_t waited = ::waitpid(child, &status, WNOHANG);
        if (waited == child) {
            return true;
        }
        if (waited < 0 && errno != EINTR) {
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            static_cast<void>(::kill(child, SIGKILL));
            do {
                status = 0;
            } while (::waitpid(child, &status, 0) < 0 && errno == EINTR);
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
}

[[noreturn]] void run_publisher_child(
    const std::filesystem::path& store,
    const int command_fd,
    const int status_fd) noexcept {
    int result = 20;
    try {
        auto writer = throwdb::NativeCollection::open(store, {}, true);
        if (!write_byte(status_fd, 'R')) {
            result = 21;
        } else {
            char command = 0;
            if (!read_byte(command_fd, command, 5s) || command != 'G') {
                result = 22;
            } else {
                bool probe_blocked = false;
                auto probe = throwdb::detail::AdvisoryLockFile::open_existing(
                    store / "locks" / "commit.lock", true);
                try {
                    probe.lock(
                        throwdb::detail::FileLockMode::Exclusive, true);
                    probe.unlock();
                } catch (const throwdb::StorageError&) {
                    probe_blocked = true;
                }
                if (!write_byte(status_fd, probe_blocked ? 'B' : 'U')) {
                    result = 23;
                } else if (!probe_blocked) {
                    result = 24;
                } else if (!write_byte(status_fd, 'M')) {
                    result = 25;
                } else {
                    const auto materialized =
                        writer->materialize_frozen_wals();
                    if (!materialized.materialized
                        || materialized.delta_run_count != 1
                        || materialized.frozen_wal_count != 0) {
                        result = 26;
                    } else if (!write_byte(status_fd, 'D')) {
                        result = 27;
                    } else {
                        result = 0;
                    }
                }
            }
        }
        writer.reset();
    } catch (...) {
        static_cast<void>(write_byte(status_fd, 'E'));
        result = 28;
    }
    static_cast<void>(::close(command_fd));
    static_cast<void>(::close(status_fd));
    ::_exit(result);
}

void test_reader_open_gate_pins_old_catalog_before_materialization_unlink() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 8;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 2> ids{11, 22};
    const std::vector<std::byte> base = rows(2, record_size, 5);
    const std::vector<std::byte> updated = rows(2, record_size, 105);
    build_store(store, ids, base, dimension);

    auto fixture_writer = throwdb::NativeCollection::open(store, {}, true);
    fixture_writer->update(ids, updated);
    const auto rotated = fixture_writer->rotate_wal();
    require(rotated.rotated, "reader-barrier fixture did not rotate its WAL");

    const throwdb::detail::MutationCatalog before = catalog_for(store);
    require_equal(before.wal_entries.size(), std::size_t{2},
                  "reader-barrier fixture has the wrong WAL count");
    const std::filesystem::path frozen_path =
        throwdb::detail::mutation_wal_path(
            store, before.wal_entries.front().wal_id);
    require(std::filesystem::exists(frozen_path),
            "reader-barrier frozen WAL is missing before publication");
    fixture_writer.reset();

    int commands[2]{-1, -1};
    int statuses[2]{-1, -1};
    require(::pipe(commands) == 0, "cannot create publisher command pipe");
    require(::pipe(statuses) == 0, "cannot create publisher status pipe");

    const pid_t child = ::fork();
    require(child >= 0, "cannot fork reader-barrier publisher");
    if (child == 0) {
        static_cast<void>(::close(commands[1]));
        static_cast<void>(::close(statuses[0]));
        run_publisher_child(store, commands[0], statuses[1]);
    }
    static_cast<void>(::close(commands[0]));
    static_cast<void>(::close(statuses[1]));

    char ready = 0;
    const bool publisher_ready = read_byte(statuses[0], ready, 5s)
        && ready == 'R';

    ReaderOpenObservation observation;
    observation.command_fd = commands[1];
    observation.status_fd = statuses[0];
    observation.store = &store;
    observation.frozen_path = &frozen_path;
    observation.before = &before;

    std::unique_ptr<throwdb::NativeCollection> old_reader;
    std::exception_ptr reader_error;
    if (publisher_ready) {
        ScopedReaderOpenHook hook(observation);
        try {
            old_reader = throwdb::NativeCollection::open(store);
        } catch (...) {
            reader_error = std::current_exception();
        }
    }

    static_cast<void>(::close(commands[1]));
    int child_status = 0;
    const bool publisher_reaped = wait_for_child(child, child_status, 10s);
    static_cast<void>(::close(statuses[0]));

    require(publisher_ready, "publisher child did not open its writer");
    require(observation.error == nullptr,
            "cannot observe reader-open barrier state");
    require(observation.command_sent,
            "reader hook did not release the publisher child");
    require(observation.cross_process_probe_blocked,
            "reader SH gate did not block a cross-process EX lock");
    require(observation.publisher_started,
            "materialization publisher did not start");
    require(!observation.publisher_finished_while_blocked,
            "publisher crossed CURRENT/unlink while reader held the open gate");
    require(observation.current_unchanged,
            "CURRENT changed while reader held the open gate");
    require(observation.frozen_existed_while_blocked,
            "publisher unlinked the frozen WAL before reader resource open");
    require(publisher_reaped,
            "materialization publisher did not finish after reader open");
    require(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "materialization publisher child failed");
    if (reader_error != nullptr) {
        std::rethrow_exception(reader_error);
    }
    require(old_reader != nullptr,
            "reader did not finish opening its old view");

    const auto old_stats = old_reader->mutation_log_stats();
    require_equal(old_stats.generation, before.generation,
                  "blocked reader did not capture the old catalog");
    require_equal(old_stats.frozen_wal_count, std::uint64_t{1},
                  "blocked reader lost its frozen WAL view");
    require_equal(old_stats.delta_run_count, std::uint64_t{0},
                  "blocked reader captured the publisher's delta");
    require_bytes_equal(
        old_reader->get_bytes(ids), updated,
        "old reader cannot read its unlinked frozen WAL resource");

    const throwdb::detail::MutationCatalog after = catalog_for(store);
    require_equal(after.generation, before.generation + 2,
                  "publisher produced the wrong catalog generation");
    require_equal(after.delta_entries.size(), std::size_t{1},
                  "publisher did not create one delta run");
    require_equal(after.wal_entries.size(), std::size_t{1},
                  "publisher retained its selected frozen WAL");
    require(!std::filesystem::exists(frozen_path),
            "publisher did not unlink the replaced frozen WAL");

    auto later_reader = throwdb::NativeCollection::open(store);
    const auto later_stats = later_reader->mutation_log_stats();
    require_equal(later_stats.generation, after.generation,
                  "later reader did not select the published catalog");
    require_equal(later_stats.delta_run_count, std::uint64_t{1},
                  "later reader did not open the materialized delta");
    require_equal(later_stats.frozen_wal_count, std::uint64_t{0},
                  "later reader reopened the retired WAL view");
    require_bytes_equal(
        later_reader->get_bytes(ids), updated,
        "later reader did not read the materialized view bit-exactly");
}

struct TestCase {
    const char* name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::array<TestCase, 1> tests{{
        {"reader_open_gate_pins_old_catalog_before_materialization_unlink",
         test_reader_open_gate_pins_old_catalog_before_materialization_unlink},
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
                      << ": unknown non-standard exception\n";
        }
    }
    if (failures != 0) {
        std::cerr << failures << " reader barrier test(s) failed\n";
        return 1;
    }
    return 0;
}
