#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "../src/core/crc32c.hpp"
#include "../src/server/read_workload_log.hpp"

namespace {

using namespace std::chrono_literals;
using throwdb::server::detail::ReadWorkloadCollectionInfo;
using throwdb::server::detail::ReadWorkloadLogger;
using throwdb::server::detail::ReadWorkloadLogOptions;
using throwdb::server::detail::ReadWorkloadMethod;
using throwdb::server::detail::ReadWorkloadSnapshot;
using throwdb::server::detail::kReadWorkloadLogHeaderBytes;
using throwdb::server::detail::kReadWorkloadRecordCrcOffset;
using throwdb::server::detail::kReadWorkloadRecordHeaderBytes;

constexpr std::array<char, 8> kLogMagic{
    'T', 'D', 'B', 'R', 'L', 'O', 'G', '1'};
constexpr std::array<char, 8> kRecordMagic{
    'T', 'D', 'B', 'R', 'R', 'E', 'C', '1'};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Left, class Right>
void require_equal(
    const Left& actual,
    const Right& expected,
    const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message);
    }
}

class TempDirectory {
public:
    explicit TempDirectory(const std::string& label) {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-read-log-" + label + "-"
                   + std::to_string(::getpid()) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::filesystem::remove_all(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (!extra_path_.empty()) {
            std::filesystem::remove_all(extra_path_, error);
        }
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

    void set_extra_path(std::filesystem::path path) {
        extra_path_ = std::move(path);
    }

private:
    std::filesystem::path path_;
    std::filesystem::path extra_path_;
};

[[nodiscard]] ReadWorkloadCollectionInfo collection(
    const std::uint64_t low = 0x1122334455667788ULL,
    const std::uint64_t high = 0x99aabbccddeeff00ULL) {
    return ReadWorkloadCollectionInfo{
        low,
        high,
        7,
        1,
        1'024,
        2'048};
}

[[nodiscard]] ReadWorkloadLogOptions options(
    const std::filesystem::path& directory,
    const std::uint32_t max_ids = 16) {
    ReadWorkloadLogOptions result;
    result.directory = directory;
    result.sample_rate_ppm = 1'000'000;
    result.max_ids_per_event = max_ids;
    const auto maximum_frame =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{max_ids} * sizeof(std::uint64_t);
    result.buffer_bytes = maximum_frame * 16;
    result.segment_bytes =
        kReadWorkloadLogHeaderBytes + maximum_frame * 64;
    result.rotate_seconds = 3'600;
    result.retention_bytes = 1ULL << 30;
    result.retention_seconds = 86'400;
    result.min_free_bytes = 0;
    return result;
}

[[nodiscard]] std::uint32_t get_u32(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    require(offset <= input.size() && 4 <= input.size() - offset,
            "test uint32 read overflow");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index) {
        value |= std::uint32_t(
                     std::to_integer<std::uint8_t>(input[offset + index]))
                 << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint64_t get_u64(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    require(offset <= input.size() && 8 <= input.size() - offset,
            "test uint64 read overflow");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != 8; ++index) {
        value |= std::uint64_t(
                     std::to_integer<std::uint8_t>(input[offset + index]))
                 << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint32_t crc_with_zero(
    const std::span<const std::byte> bytes,
    const std::size_t offset) {
    throwdb::detail::Crc32c crc;
    crc.update(bytes.first(offset));
    crc.update_zeroes(4);
    crc.update(bytes.subspan(offset + 4));
    return crc.value();
}

[[nodiscard]] std::vector<std::byte> read_bytes(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "cannot open test log");
    const auto end = input.tellg();
    require(end >= 0, "cannot size test log");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(
        input.good() || input.eof(),
        "cannot read test log");
    return bytes;
}

[[nodiscard]] std::vector<std::filesystem::path> log_files(
    const std::filesystem::path& directory,
    const std::string& extension = ".tdbrl") {
    std::vector<std::filesystem::path> result;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == extension) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

struct ParsedFrame {
    std::vector<std::byte> bytes;
    std::vector<std::uint64_t> ids;
    std::uint32_t kind = 0;
};

struct ParsedLog {
    std::vector<std::byte> header;
    std::vector<ParsedFrame> frames;
    bool torn_tail = false;
    bool corrupt = false;
};

[[nodiscard]] ParsedLog parse_log(
    const std::span<const std::byte> bytes) {
    ParsedLog result;
    if (bytes.size() < kReadWorkloadLogHeaderBytes) {
        result.torn_tail = true;
        return result;
    }
    result.header.assign(
        bytes.begin(),
        bytes.begin() + kReadWorkloadLogHeaderBytes);
    const auto header = std::span<const std::byte>(result.header);
    if (std::memcmp(header.data(), kLogMagic.data(), kLogMagic.size()) != 0
        || get_u32(header, 8) != 1
        || get_u32(header, 12) != kReadWorkloadLogHeaderBytes
        || get_u32(header, 24) != crc_with_zero(header, 24)) {
        result.corrupt = true;
        return result;
    }
    std::size_t cursor = kReadWorkloadLogHeaderBytes;
    while (cursor != bytes.size()) {
        if (bytes.size() - cursor < kReadWorkloadRecordHeaderBytes) {
            result.torn_tail = true;
            break;
        }
        const auto remaining = bytes.subspan(cursor);
        if (std::memcmp(
                remaining.data(),
                kRecordMagic.data(),
                kRecordMagic.size())
                != 0
            || get_u32(remaining, 8) != 1
            || get_u32(remaining, 12) != kReadWorkloadRecordHeaderBytes) {
            result.corrupt = true;
            break;
        }
        const auto frame_bytes = get_u32(remaining, 16);
        if (frame_bytes < kReadWorkloadRecordHeaderBytes
            || frame_bytes > remaining.size()) {
            result.torn_tail = true;
            break;
        }
        const auto frame = remaining.first(frame_bytes);
        if (get_u32(frame, kReadWorkloadRecordCrcOffset)
            != crc_with_zero(frame, kReadWorkloadRecordCrcOffset)) {
            result.corrupt = true;
            break;
        }
        ParsedFrame parsed;
        parsed.kind = get_u32(frame, 20);
        parsed.bytes.assign(frame.begin(), frame.end());
        if (parsed.kind == 1) {
            const auto captured = get_u32(frame, 112);
            if (frame_bytes
                != kReadWorkloadRecordHeaderBytes
                       + std::uint64_t{captured}
                             * sizeof(std::uint64_t)) {
                result.corrupt = true;
                break;
            }
            for (std::uint32_t index = 0; index != captured; ++index) {
                parsed.ids.push_back(get_u64(
                    frame,
                    kReadWorkloadRecordHeaderBytes
                        + std::size_t{index} * sizeof(std::uint64_t)));
            }
        } else if (parsed.kind != 2
                   || frame_bytes != kReadWorkloadRecordHeaderBytes) {
            result.corrupt = true;
            break;
        }
        result.frames.push_back(std::move(parsed));
        cursor += frame_bytes;
    }
    return result;
}

template <class Predicate>
void wait_until(
    Predicate predicate,
    const std::string& message,
    const std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return;
        }
        std::this_thread::sleep_for(5ms);
    }
    require(predicate(), message);
}

[[nodiscard]] const ParsedFrame& first_event(const ParsedLog& log) {
    const auto iterator = std::find_if(
        log.frames.begin(), log.frames.end(), [](const ParsedFrame& frame) {
            return frame.kind == 1;
        });
    require(iterator != log.frames.end(), "log has no event frame");
    return *iterator;
}

void test_format_crc_checkpoint_and_torn_tail() {
    TempDirectory temporary("format");
    auto logger = ReadWorkloadLogger::open(
        options(temporary.path()), collection());
    require(
        !logger->try_log(ReadWorkloadMethod::Get, {}, {}, 1'024),
        "empty request produced a workload event");
    require_equal(
        logger->stats().observed_events,
        std::uint64_t{0},
        "empty request changed workload counters");
    const std::array<std::uint64_t, 1> invalid_prefix_ids{1};
    require(
        !logger->try_log(
            ReadWorkloadMethod::Get, invalid_prefix_ids, {}, 0)
            && !logger->try_log(
                ReadWorkloadMethod::Get, invalid_prefix_ids, {}, 1'025),
        "invalid prefix produced a workload event");
    require_equal(
        logger->stats().observed_events,
        std::uint64_t{0},
        "invalid prefix changed workload counters");
    const std::array<std::uint64_t, 4> ids{7, 9, 11, 13};
    ReadWorkloadSnapshot snapshot;
    snapshot.catalog_generation = 17;
    snapshot.committed_sequence = 19;
    snapshot.row_count = 23;
    snapshot.stream_id_low = 29;
    snapshot.stream_id_high = 31;
    snapshot.batch_sequence = 37;
    snapshot.exact = true;
    require(
        logger->try_log(
            ReadWorkloadMethod::StreamGet, ids, snapshot, 128),
        "event was not accepted");
    logger->shutdown();

    const auto files = log_files(temporary.path());
    require_equal(files.size(), std::size_t{1}, "unexpected log count");
    const auto bytes = read_bytes(files.front());
    const auto parsed = parse_log(bytes);
    require(!parsed.corrupt && !parsed.torn_tail,
            "fresh log does not parse");
    const auto header = std::span<const std::byte>(parsed.header);
    require_equal(get_u32(header, 16), std::uint32_t{176},
                  "header used bytes changed");
    require_equal(
        get_u64(header, 32),
        collection().collection_id_low,
        "header collection low changed");
    require_equal(
        get_u64(header, 40),
        collection().collection_id_high,
        "header collection high changed");
    require_equal(get_u32(header, 48), std::uint32_t{7},
                  "header format changed");
    require_equal(get_u32(header, 52), std::uint32_t{1},
                  "header dtype changed");
    require_equal(get_u32(header, 56), std::uint32_t{1'024},
                  "header dimension changed");
    require_equal(get_u32(header, 60), std::uint32_t{2'048},
                  "header record size changed");
    require_equal(get_u32(header, 88), std::uint32_t{1'000'000},
                  "header sample ppm changed");
    require_equal(get_u32(header, 92), std::uint32_t{16},
                  "header ID cap changed");
    require_equal(get_u64(header, 152), std::uint64_t{0},
                  "first segment sequence changed");
    require_equal(
        get_u64(header, 160),
        options(temporary.path()).max_write_bytes_per_second,
        "header write-rate bound changed");
    require_equal(
        get_u64(header, 168),
        options(temporary.path()).max_retained_files,
        "header retained-file bound changed");

    require(
        parsed.frames.size() >= 3,
        "segment is missing its opening/event/terminal frames");
    const auto opening =
        std::span<const std::byte>(parsed.frames.front().bytes);
    require_equal(
        parsed.frames.front().kind,
        std::uint32_t{2},
        "segment does not begin with a checkpoint");
    require_equal(
        get_u32(opening, 28),
        std::uint32_t{2},
        "opening checkpoint extension version changed");
    require_equal(
        get_u32(opening, 32),
        std::uint32_t{1},
        "opening checkpoint flag changed");
    require_equal(
        get_u64(opening, 128),
        std::uint64_t{0},
        "first segment opening baseline is not zero");

    const auto& event = first_event(parsed);
    const auto frame = std::span<const std::byte>(event.bytes);
    require_equal(get_u32(frame, 20), std::uint32_t{1},
                  "event kind changed");
    require_equal(
        get_u32(frame, 28),
        static_cast<std::uint32_t>(ReadWorkloadMethod::StreamGet),
        "method changed");
    require_equal(get_u32(frame, 32), std::uint32_t{1},
                  "exact flag changed");
    require_equal(get_u32(frame, 36), std::uint32_t{128},
                  "prefix changed");
    require_equal(get_u64(frame, 56), std::uint64_t{17},
                  "catalog generation changed");
    require_equal(get_u64(frame, 64), std::uint64_t{19},
                  "committed sequence changed");
    require_equal(get_u64(frame, 72), std::uint64_t{23},
                  "live count changed");
    require_equal(get_u64(frame, 80), std::uint64_t{29},
                  "stream low changed");
    require_equal(get_u64(frame, 88), std::uint64_t{31},
                  "stream high changed");
    require_equal(get_u64(frame, 96), std::uint64_t{37},
                  "batch sequence changed");
    require_equal(get_u64(frame, 104), std::uint64_t{4},
                  "original ID count changed");
    require_equal(get_u32(frame, 112), std::uint32_t{4},
                  "captured ID count changed");
    require(event.ids == std::vector<std::uint64_t>(ids.begin(), ids.end()),
            "uncapped IDs changed order");
    require(
        std::any_of(
            parsed.frames.begin(),
            parsed.frames.end(),
            [](const ParsedFrame& candidate) {
                return candidate.kind == 2;
            }),
        "shutdown did not write a counter checkpoint");
    require_equal(
        parsed.frames.back().kind,
        std::uint32_t{2},
        "finalized segment has no terminal checkpoint");
    const auto terminal =
        std::span<const std::byte>(parsed.frames.back().bytes);
    require_equal(
        get_u32(terminal, 28),
        std::uint32_t{2},
        "checkpoint extension version changed");
    require_equal(
        get_u32(terminal, 32),
        std::uint32_t{0},
        "terminal checkpoint was marked as an opening baseline");
    require_equal(
        get_u64(terminal, 56),
        std::uint64_t{4},
        "checkpoint sampled-ID count changed");
    require_equal(
        get_u64(terminal, 64),
        std::uint64_t{4},
        "checkpoint enqueued-ID count changed");
    require_equal(
        get_u64(terminal, 72),
        std::uint64_t{4},
        "checkpoint written-ID count changed");

    auto torn = bytes;
    torn.insert(
        torn.end(),
        event.bytes.begin(),
        event.bytes.begin() + 41);
    const auto torn_parsed = parse_log(torn);
    require(
        torn_parsed.torn_tail && !torn_parsed.corrupt
            && torn_parsed.frames.size() == parsed.frames.size(),
        "torn final frame was not isolated from complete frames");

    auto corrupt = bytes;
    const auto event_offset =
        kReadWorkloadLogHeaderBytes
        + kReadWorkloadRecordHeaderBytes;
    corrupt[event_offset + kReadWorkloadRecordHeaderBytes] ^=
        std::byte{1};
    const auto corrupt_parsed = parse_log(corrupt);
    require(corrupt_parsed.corrupt,
            "payload corruption bypassed frame CRC32C");
}

void test_truncation_is_bounded_and_covers_the_request() {
    TempDirectory temporary("cap");
    auto configured = options(temporary.path(), 4);
    configured.buffer_bytes =
        (kReadWorkloadRecordHeaderBytes + 4 * sizeof(std::uint64_t))
        * 256;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    std::vector<std::uint64_t> ids(1'000);
    for (std::size_t index = 0; index != ids.size(); ++index) {
        ids[index] = index;
    }
    std::size_t accepted = 0;
    while (accepted != 128) {
        if (logger->try_log(
                ReadWorkloadMethod::Get, ids, {}, 1'024)) {
            ++accepted;
        } else {
            std::this_thread::yield();
        }
    }
    logger->shutdown();

    std::array<bool, 4> quarters{};
    std::size_t events = 0;
    for (const auto& path : log_files(temporary.path())) {
        const auto parsed = parse_log(read_bytes(path));
        require(!parsed.corrupt && !parsed.torn_tail,
                "capped log does not parse");
        for (const auto& frame : parsed.frames) {
            if (frame.kind != 1) {
                continue;
            }
            ++events;
            require_equal(frame.ids.size(), std::size_t{4},
                          "ID cap was not enforced");
            const auto bytes = std::span<const std::byte>(frame.bytes);
            require_equal(get_u64(bytes, 104), std::uint64_t{1'000},
                          "original count was not retained");
            require_equal(get_u32(bytes, 32), std::uint32_t{2},
                          "truncated flag was not set");
            auto unique = frame.ids;
            std::sort(unique.begin(), unique.end());
            require(
                std::adjacent_find(unique.begin(), unique.end())
                    == unique.end(),
                "systematic sample repeated an ID");
            for (const auto id : frame.ids) {
                require(id < ids.size(), "sampled ID is outside request");
                quarters[static_cast<std::size_t>(id / 250)] = true;
            }
        }
    }
    require_equal(events, std::size_t{128}, "accepted events were lost");
    require(
        std::all_of(quarters.begin(), quarters.end(), [](const bool value) {
            return value;
        }),
        "systematic sampling did not cover the request");
    const auto stats = logger->stats();
    require_equal(stats.truncated_events, std::uint64_t{128},
                  "truncation counter changed");
    require_equal(stats.captured_ids, std::uint64_t{512},
                  "captured-ID counter changed");
}

void test_queue_saturation_is_all_or_drop() {
    TempDirectory temporary("saturation");
    constexpr std::uint32_t cap = 65'536;
    auto configured = options(temporary.path(), cap);
    const auto maximum_frame =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{cap} * sizeof(std::uint64_t);
    configured.buffer_bytes = maximum_frame * 2;
    configured.segment_bytes =
        kReadWorkloadLogHeaderBytes + maximum_frame * 128;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    std::vector<std::uint64_t> ids(cap);
    std::atomic<bool> start{false};
    std::vector<std::thread> producers;
    for (std::size_t thread = 0; thread != 8; ++thread) {
        producers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (std::size_t call = 0; call != 20; ++call) {
                (void)logger->try_log(
                    ReadWorkloadMethod::Get, ids, {}, 1'024);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& producer : producers) {
        producer.join();
    }
    logger->shutdown();
    const auto stats = logger->stats();
    require_equal(stats.observed_events, std::uint64_t{160},
                  "producer event accounting changed");
    require(
        stats.dropped_busy + stats.dropped_full != 0,
        "bounded queue did not drop under saturation");
    require_equal(
        stats.sampled_events,
        stats.enqueued_events + stats.dropped_busy + stats.dropped_full
            + stats.dropped_runtime_failure + stats.dropped_min_free,
        "sampled event was partially admitted");
    require_equal(
        stats.sampled_ids,
        stats.enqueued_ids + stats.dropped_busy_ids
            + stats.dropped_full_ids
            + stats.dropped_runtime_failure_ids
            + stats.dropped_min_free_ids,
        "sampled request rows were partially admitted");
    require(
        stats.peak_queued_bytes <= stats.buffer_capacity_bytes,
        "queue exceeded its preallocated byte bound");
}

void test_rotation_permissions_and_idle_checkpoint_stability() {
    TempDirectory temporary("rotation");
    auto configured = options(temporary.path(), 8);
    const auto frame_bytes =
        kReadWorkloadRecordHeaderBytes + 8 * sizeof(std::uint64_t);
    configured.buffer_bytes = frame_bytes * 8;
    configured.segment_bytes =
        kReadWorkloadLogHeaderBytes + frame_bytes
        + 2 * kReadWorkloadRecordHeaderBytes;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 8> ids{};
    for (std::size_t event = 0; event != 4; ++event) {
        while (!logger->try_log(
            ReadWorkloadMethod::Get, ids, {}, 1'024)) {
            std::this_thread::yield();
        }
    }
    wait_until(
        [&] { return logger->stats().written_events == 4; },
        "rotation events did not drain");

    std::this_thread::sleep_for(1'300ms);
    auto opens = log_files(temporary.path(), ".open");
    require_equal(opens.size(), std::size_t{1},
                  "logger has no active segment");
    const auto stable_size = std::filesystem::file_size(opens.front());
    std::this_thread::sleep_for(1'300ms);
    require_equal(
        std::filesystem::file_size(opens.front()),
        stable_size,
        "idle logger wrote unbounded counter checkpoints");
    logger->shutdown();

    const auto files = log_files(temporary.path());
    require(files.size() >= 2, "size rotation did not finalize segments");
    struct stat status {};
    require(::stat(temporary.path().c_str(), &status) == 0,
            "cannot stat log directory");
    require_equal(status.st_mode & 0777, mode_t{0700},
                  "log directory permissions changed");
    std::vector<std::uint64_t> segment_sequences;
    std::vector<std::pair<std::uint64_t, ParsedLog>> parsed_segments;
    for (const auto& path : files) {
        require(::stat(path.c_str(), &status) == 0,
                "cannot stat log segment");
        require_equal(status.st_mode & 0777, mode_t{0600},
                      "log segment permissions changed");
        const auto bytes = read_bytes(path);
        const auto segment_sequence = get_u64(
            std::span<const std::byte>(bytes)
                .first(kReadWorkloadLogHeaderBytes),
            152);
        segment_sequences.push_back(segment_sequence);
        auto parsed = parse_log(bytes);
        require(
            !parsed.corrupt && !parsed.torn_tail
                && parsed.frames.size() >= 3,
            "rotated segment is malformed");
        require_equal(
            get_u32(parsed.frames.front().bytes, 32),
            std::uint32_t{1},
            "rotated segment lacks an opening baseline");
        require_equal(
            get_u32(parsed.frames.back().bytes, 32),
            std::uint32_t{0},
            "rotated segment terminal is marked as an opening baseline");
        parsed_segments.emplace_back(
            segment_sequence, std::move(parsed));
    }
    std::sort(segment_sequences.begin(), segment_sequences.end());
    for (std::size_t index = 0; index != segment_sequences.size(); ++index) {
        require_equal(
            segment_sequences[index],
            static_cast<std::uint64_t>(index),
            "segment sequence is not contiguous");
    }
    std::sort(
        parsed_segments.begin(),
        parsed_segments.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    for (std::size_t index = 1; index != parsed_segments.size(); ++index) {
        const auto& previous =
            parsed_segments[index - 1].second.frames.back().bytes;
        const auto& opening =
            parsed_segments[index].second.frames.front().bytes;
        require(
            std::equal(
                previous.begin() + 56,
                previous.end(),
                opening.begin() + 56,
                opening.end()),
            "rotated segment opening baseline differs from the previous "
            "terminal counters");
    }
    require(
        ::stat(
            (temporary.path() / ".read-workload.writer.lock").c_str(),
            &status)
            == 0,
        "writer lock is missing");
    require_equal(status.st_mode & 0777, mode_t{0600},
                  "writer lock permissions changed");
}

void test_idle_rotation_and_unused_segment_cleanup() {
    TempDirectory temporary("idle-rotation");
    auto configured = options(temporary.path(), 4);
    configured.rotate_seconds = 1;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 4> ids{1, 2, 3, 4};
    wait_until(
        [&] {
            return logger->try_log(
                ReadWorkloadMethod::Get, ids, {}, 1'024);
        },
        "idle-rotation event was not accepted");
    wait_until(
        [&] { return logger->stats().rotated_segments >= 1; },
        "idle time bound did not finalize the active segment",
        5s);
    logger->shutdown();

    require_equal(
        log_files(temporary.path()).size(),
        std::size_t{1},
        "idle rotation published an empty successor segment");
    require(
        log_files(temporary.path(), ".open").empty(),
        "clean shutdown left an unused open segment");
}

void test_idle_retention_sweep_expires_raw_ids() {
    TempDirectory temporary("idle-retention");
    auto configured = options(temporary.path(), 4);
    configured.rotate_seconds = 1;
    configured.retention_seconds = 1;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 4> ids{1, 2, 3, 4};
    wait_until(
        [&] {
            return logger->try_log(
                ReadWorkloadMethod::Get, ids, {}, 1'024);
        },
        "idle-retention event was not accepted");
    wait_until(
        [&] {
            return logger->stats().retention_deleted_segments >= 1
                   && log_files(temporary.path()).empty();
        },
        "idle logger did not enforce age-based raw-ID retention",
        6s);
    logger->shutdown();
}

void test_exclusive_writer_lock() {
    TempDirectory temporary("lock");
    auto first = ReadWorkloadLogger::open(
        options(temporary.path()), collection());
    bool rejected = false;
    try {
        auto second = ReadWorkloadLogger::open(
            options(temporary.path()), collection());
        (void)second;
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "second directory writer bypassed advisory lock");
    first->shutdown();
}

void test_directory_marker_is_collection_scoped() {
    TempDirectory temporary("marker");
    {
        auto logger =
            ReadWorkloadLogger::open(
                options(temporary.path()), collection());
        logger->shutdown();
    }
    const auto marker =
        temporary.path() / ".read-workload.owner";
    require(std::filesystem::is_regular_file(marker),
            "ownership marker was not persisted");
    bool rejected = false;
    try {
        auto logger = ReadWorkloadLogger::open(
            options(temporary.path()),
            collection(0x0102030405060708ULL));
        (void)logger;
    } catch (const std::exception&) {
        rejected = true;
    }
    require(
        rejected && std::filesystem::is_regular_file(marker),
        "another collection claimed or damaged the owned log directory");
}

void test_min_free_pauses_without_failing_requests() {
    TempDirectory temporary("min-free");
    auto configured = options(temporary.path());
    configured.min_free_bytes =
        std::numeric_limits<std::uint64_t>::max();
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    require(logger->stats().paused_for_min_free,
            "impossible free-space reserve did not pause logger");
    const std::array<std::uint64_t, 1> ids{1};
    require(
        !logger->try_log(ReadWorkloadMethod::Get, ids, {}, 1'024),
        "paused logger admitted an event");
    const auto stats = logger->stats();
    require_equal(stats.dropped_min_free, std::uint64_t{1},
                  "min-free drop was not accounted");
    require_equal(stats.dropped_min_free_ids, std::uint64_t{1},
                  "min-free row drop was not accounted");
    require(!stats.runtime_failed,
            "min-free pause was treated as a runtime failure");
    logger->shutdown();
}

void test_io_budget_drops_are_bounded_and_accounted() {
    TempDirectory temporary("io-budget");
    constexpr std::uint32_t cap = 65'536;
    auto configured = options(temporary.path(), cap);
    const auto maximum_frame =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{cap} * sizeof(std::uint64_t);
    configured.buffer_bytes = maximum_frame * 16;
    configured.max_write_bytes_per_second =
        kReadWorkloadLogHeaderBytes + maximum_frame
        + 2 * kReadWorkloadRecordHeaderBytes;
    std::vector<std::uint64_t> ids(cap, 7);
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    std::size_t accepted = 0;
    while (accepted != 8) {
        if (logger->try_log(
                ReadWorkloadMethod::Get, ids, {}, 1'024)) {
            ++accepted;
        } else {
            std::this_thread::yield();
        }
    }
    logger->shutdown();
    const auto stats = logger->stats();
    require_equal(stats.enqueued_events, std::uint64_t{8},
                  "I/O-budget test did not enqueue every event");
    require(
        stats.dropped_io_budget != 0
            && stats.written_events + stats.dropped_io_budget
                   == stats.enqueued_events,
        "I/O budget did not account every enqueued event");
    require_equal(
        stats.written_ids + stats.dropped_io_budget_ids,
        stats.enqueued_ids,
        "I/O budget did not account every enqueued row");
    require(
        stats.bytes_written
            <= kReadWorkloadLogHeaderBytes + maximum_frame
                   + 2 * kReadWorkloadRecordHeaderBytes,
        "I/O budget wrote an unbounded number of event frames");
}

void test_runtime_io_failure_disables_only_logging() {
    TempDirectory temporary("runtime");
    auto configured = options(temporary.path(), 1);
    const auto frame_bytes =
        kReadWorkloadRecordHeaderBytes + sizeof(std::uint64_t);
    configured.buffer_bytes = frame_bytes * 4;
    configured.segment_bytes =
        kReadWorkloadLogHeaderBytes + frame_bytes
        + 2 * kReadWorkloadRecordHeaderBytes;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 1> ids{1};
    wait_until(
        [&] {
            return logger->try_log(
                ReadWorkloadMethod::Get, ids, {}, 1'024);
        },
            "first runtime-failure event was not accepted");
    wait_until(
        [&] { return logger->stats().written_events == 1; },
        "first runtime-failure event did not drain");

    const auto active = log_files(temporary.path(), ".open");
    require_equal(active.size(), std::size_t{1},
                  "runtime-failure active segment is missing");
    require(std::filesystem::remove(active.front()),
            "cannot remove active segment for failure injection");
    while (!logger->try_log(
        ReadWorkloadMethod::Get, ids, {}, 1'024)) {
        if (logger->stats().runtime_failed) {
            break;
        }
        std::this_thread::yield();
    }
    wait_until(
        [&] { return logger->stats().runtime_failed; },
        "rotation failure did not disable logger");
    require(
        !logger->try_log(ReadWorkloadMethod::Get, ids, {}, 1'024),
        "failed logger accepted more work");
    const auto stats = logger->stats();
    require(stats.write_errors != 0 && stats.last_error_number != 0,
            "runtime failure diagnostics are missing");
    logger->shutdown();
}

void test_retention_counts_and_removes_stale_open_safely() {
    TempDirectory temporary("retention");
    auto configured = options(temporary.path());
    {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        const std::array<std::uint64_t, 1> ids{1};
        wait_until(
            [&] {
                return logger->try_log(
                    ReadWorkloadMethod::Get, ids, {}, 1'024);
            },
                "retention seed event was not accepted");
        logger->shutdown();
    }
    auto finalized = log_files(temporary.path());
    require_equal(finalized.size(), std::size_t{1},
                  "retention seed segment is missing");
    auto stale = finalized.front();
    stale.replace_extension(".open");
    std::filesystem::copy_file(finalized.front(), stale);
    std::filesystem::last_write_time(
        stale,
        std::filesystem::file_time_type::clock::now() - 10s);

    const auto unknown = temporary.path() / "unknown.open";
    {
        std::ofstream output(unknown);
        output << "not a ThrowDB workload log";
    }
    std::filesystem::last_write_time(
        unknown,
        std::filesystem::file_time_type::clock::now() - 10s);

    const auto invalid_stale =
        temporary.path()
        / ("read-workload-99aabbccddeeff001122334455667788-"
           "invalid.open");
    std::filesystem::copy_file(finalized.front(), invalid_stale);
    {
        std::fstream stream(
            invalid_stale,
            std::ios::binary | std::ios::in | std::ios::out);
        require(stream.good(), "cannot corrupt stale open test segment");
        stream.seekp(0);
        stream.put('X');
    }
    auto invalid_finalized = invalid_stale;
    invalid_finalized.replace_extension(".tdbrl");
    std::filesystem::copy_file(invalid_stale, invalid_finalized);
    std::filesystem::last_write_time(
        invalid_finalized,
        std::filesystem::file_time_type::clock::now() - 10s);

    const auto symlink =
        temporary.path()
        / ("read-workload-99aabbccddeeff001122334455667788-"
           "symlink.tdbrl");
    std::filesystem::create_symlink(finalized.front(), symlink);

    configured.retention_seconds = 1;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const auto stats = logger->stats();
    require(!std::filesystem::exists(stale),
            "aged recognized stale .open was not removed");
    require(std::filesystem::exists(unknown),
            "unknown .open file was removed");
    require(!std::filesystem::exists(invalid_stale),
            "invalid owned stale .open was retained");
    require(!std::filesystem::exists(invalid_finalized),
            "aged corrupt finalized segment bypassed privacy retention");
    require(std::filesystem::is_symlink(symlink),
            "symlink was followed or removed");
    require(stats.retention_deleted_segments >= 1,
            "retention deletion was not accounted");
    require(stats.retained_segments >= 1,
            "finalized retained segment was not accounted");
    logger->shutdown();
}

void test_retained_file_count_survives_restarts() {
    TempDirectory temporary("file-count");
    auto configured = options(temporary.path());
    configured.max_retained_files = 2;
    std::uint64_t deleted = 0;
    const std::array<std::uint64_t, 1> ids{1};
    for (std::size_t run = 0; run != 5; ++run) {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        wait_until(
            [&] {
                return logger->try_log(
                    ReadWorkloadMethod::Get, ids, {}, 1'024);
            },
            "retained-file seed event was not accepted");
        logger->shutdown();
        deleted = std::max(
            deleted,
            logger->stats().retention_deleted_segments);
        require(
            log_files(temporary.path()).size() <= 2,
            "retained-file ceiling was exceeded after restart");
    }
    require(deleted != 0,
            "retained-file pruning was not accounted");
}

void test_invalid_configuration_fails_before_background_start() {
    TempDirectory temporary("invalid");
    std::filesystem::create_directories(temporary.path());
    const auto not_directory = temporary.path() / "file";
    {
        std::ofstream output(not_directory);
        output << "x";
    }
    auto configured = options(not_directory);
    bool rejected = false;
    try {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        (void)logger;
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "non-directory log target was accepted");

    configured = options(temporary.path());
    configured.buffer_bytes =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{configured.max_ids_per_event}
              * sizeof(std::uint64_t);
    rejected = false;
    try {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        (void)logger;
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "one-slot queue bypassed minimum capacity");
}

}  // namespace

int main() {
    try {
        test_format_crc_checkpoint_and_torn_tail();
        test_truncation_is_bounded_and_covers_the_request();
        test_queue_saturation_is_all_or_drop();
        test_rotation_permissions_and_idle_checkpoint_stability();
        test_idle_rotation_and_unused_segment_cleanup();
        test_idle_retention_sweep_expires_raw_ids();
        test_exclusive_writer_lock();
        test_directory_marker_is_collection_scoped();
        test_min_free_pauses_without_failing_requests();
        test_io_budget_drops_are_bounded_and_accounted();
        test_runtime_io_failure_disables_only_logging();
        test_retention_counts_and_removes_stale_open_safely();
        test_retained_file_count_survives_restarts();
        test_invalid_configuration_fails_before_background_start();
        std::cout << "read workload log tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "read workload log test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
