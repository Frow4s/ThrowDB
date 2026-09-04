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
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/core/crc32c.hpp"
#include "../src/server/read_workload_log.hpp"

namespace {

using namespace std::chrono_literals;
using throwdb::server::detail::ReadWorkloadBackend;
using throwdb::server::detail::ReadWorkloadBatchMetrics;
using throwdb::server::detail::ReadWorkloadCollectionInfo;
using throwdb::server::detail::ReadWorkloadIdChunkInfo;
using throwdb::server::detail::ReadWorkloadLogger;
using throwdb::server::detail::ReadWorkloadLogOptions;
using throwdb::server::detail::ReadWorkloadMethod;
using throwdb::server::detail::ReadWorkloadRequestEnd;
using throwdb::server::detail::ReadWorkloadRequestInfo;
using throwdb::server::detail::ReadWorkloadRequestStatus;
using throwdb::server::detail::ReadWorkloadSnapshot;
using throwdb::server::detail::kReadWorkloadBatchRecordBytes;
using throwdb::server::detail::kReadWorkloadLogHeaderBytes;
using throwdb::server::detail::kReadWorkloadMaximumIdChunkRows;
using throwdb::server::detail::kReadWorkloadRecordCrcOffset;
using throwdb::server::detail::kReadWorkloadRecordHeaderBytes;

constexpr std::array<char, 8> kLogMagic{
    'T', 'D', 'B', 'R', 'L', 'O', 'G', '2'};
constexpr std::array<char, 8> kRecordMagic{
    'T', 'D', 'B', 'R', 'R', 'E', 'C', '2'};
constexpr std::uint32_t kBeginKind = 1;
constexpr std::uint32_t kIdChunkKind = 2;
constexpr std::uint32_t kBatchKind = 3;
constexpr std::uint32_t kEndKind = 4;
constexpr std::uint32_t kCheckpointKind = 5;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr std::uint64_t kFnvLowOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvHighOffset = 0x84222325cbf29ce4ULL;

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
                / ("throwdb-read-log-v2-" + label + "-"
                   + std::to_string(::getpid()) + "-"
                   + std::to_string(sequence.fetch_add(1)));
        std::filesystem::remove_all(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[nodiscard]] ReadWorkloadCollectionInfo collection(
    const std::uint64_t low = 0x1122334455667788ULL,
    const std::uint64_t high = 0x99aabbccddeeff00ULL) {
    return ReadWorkloadCollectionInfo{
        .collection_id_low = low,
        .collection_id_high = high,
        .format = 7,
        .dtype = 1,
        .dimension = 1'024,
        .record_size_bytes = 2'048,
    };
}

[[nodiscard]] ReadWorkloadCollectionInfo collection_with_reader() {
    auto result = collection();
    result.reader_backend =
        static_cast<std::uint32_t>(ReadWorkloadBackend::IoUring);
    result.io_workers = 0;
    result.io_queue_capacity = 0;
    result.io_ranges_per_task = 8;
    result.max_read_range_bytes = 1ULL << 20;
    result.max_gap_bytes = 4'096;
    result.io_uring_queue_depth = 32;
    result.io_uring_workers = 2;
    result.metric_page_bytes = 4'096;
    result.metric_chunk_bytes = 1U << 20;
    result.tuning_explicit_overrides = 3;
    return result;
}

[[nodiscard]] ReadWorkloadCollectionInfo
collection_with_sequential_thread_pool_reader() {
    auto result = collection();
    result.reader_backend =
        static_cast<std::uint32_t>(ReadWorkloadBackend::ThreadPool);
    result.io_workers = 0;
    result.io_queue_capacity = 0;
    result.io_ranges_per_task = 8;
    result.max_read_range_bytes = 1ULL << 20;
    result.max_gap_bytes = 0;
    // These are valid configured defaults retained as context even though
    // they are inactive for the thread-pool backend.
    result.io_uring_queue_depth = 32;
    result.io_uring_workers = 1;
    return result;
}

[[nodiscard]] ReadWorkloadLogOptions options(
    const std::filesystem::path& directory,
    const std::uint32_t chunk_rows = 16) {
    ReadWorkloadLogOptions result;
    result.directory = directory;
    result.id_chunk_rows = chunk_rows;
    result.max_active_requests = 64;
    const auto maximum_frame = std::max<std::uint64_t>(
        kReadWorkloadBatchRecordBytes,
        kReadWorkloadRecordHeaderBytes
            + std::uint64_t{chunk_rows} * sizeof(std::uint64_t));
    result.buffer_bytes = maximum_frame * 256;
    result.segment_bytes =
        kReadWorkloadLogHeaderBytes
        + 2 * kReadWorkloadRecordHeaderBytes
        + maximum_frame * 64;
    result.rotate_seconds = 3'600;
    result.retention_bytes = 1ULL << 30;
    result.retention_seconds = 86'400;
    result.min_free_bytes = 0;
    result.max_write_bytes_per_second = 64ULL << 20;
    result.max_retained_files = 4'096;
    return result;
}

[[nodiscard]] ReadWorkloadRequestInfo request_info(
    const ReadWorkloadMethod method,
    const std::uint64_t declared_ids,
    const std::uint64_t low,
    const std::uint64_t high) {
    return ReadWorkloadRequestInfo{
        .method = method,
        .snapshot =
            ReadWorkloadSnapshot{
                .catalog_generation = 17,
                .committed_sequence = 19,
                .row_count = 23,
                .exact = true,
            },
        .prefix_dimensions = 128,
        .declared_ids = declared_ids,
        .request_id_low = low,
        .request_id_high = high,
    };
}

[[nodiscard]] std::uint32_t get_u32(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    require(
        offset <= input.size() && 4 <= input.size() - offset,
        "test uint32 read overflow");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index) {
        value |= std::uint32_t(
                     std::to_integer<std::uint8_t>(
                         input[offset + index]))
                 << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint64_t get_u64(
    const std::span<const std::byte> input,
    const std::size_t offset) {
    require(
        offset <= input.size() && 8 <= input.size() - offset,
        "test uint64 read overflow");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != 8; ++index) {
        value |= std::uint64_t(
                     std::to_integer<std::uint8_t>(
                         input[offset + index]))
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

[[nodiscard]] std::uint64_t fnv1a(
    const std::span<const std::uint64_t> ids,
    std::uint64_t state) {
    for (const auto id : ids) {
        for (unsigned int shift = 0; shift != 64; shift += 8) {
            state ^= (id >> shift) & 0xffU;
            state *= kFnvPrime;
        }
    }
    return state;
}

[[nodiscard]] std::vector<std::byte> read_bytes(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "cannot open workload log");
    const auto end = input.tellg();
    require(end >= 0, "cannot size workload log");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(input.good() || input.eof(), "cannot read workload log");
    return bytes;
}

[[nodiscard]] std::vector<std::filesystem::path> log_files(
    const std::filesystem::path& directory,
    const std::string& extension = ".tdbrl") {
    std::vector<std::filesystem::path> result;
    if (!std::filesystem::exists(directory)) {
        return result;
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == extension) {
            result.push_back(entry.path());
        }
    }
    return result;
}

struct ParsedFrame {
    std::vector<std::byte> bytes;
    std::vector<std::uint64_t> ids;
    std::uint32_t kind = 0;
    std::uint64_t sequence = 0;
    std::uint64_t request_low = 0;
    std::uint64_t request_high = 0;
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
    if (std::memcmp(
            header.data(), kLogMagic.data(), kLogMagic.size())
            != 0
        || get_u32(header, 8) != 2
        || get_u32(header, 12) != kReadWorkloadLogHeaderBytes
        || get_u32(header, 16) != 232
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
            || get_u32(remaining, 8) != 2
            || get_u32(remaining, 12)
                   != kReadWorkloadRecordHeaderBytes) {
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
        parsed.sequence = get_u64(frame, 40);
        parsed.request_low = get_u64(frame, 64);
        parsed.request_high = get_u64(frame, 72);
        parsed.bytes.assign(frame.begin(), frame.end());
        if (parsed.kind == kIdChunkKind) {
            const auto count = get_u32(frame, 32);
            if (count == 0
                || count > kReadWorkloadMaximumIdChunkRows
                || frame_bytes
                       != kReadWorkloadRecordHeaderBytes
                              + std::uint64_t{count}
                                    * sizeof(std::uint64_t)) {
                result.corrupt = true;
                break;
            }
            parsed.ids.reserve(count);
            for (std::uint32_t index = 0; index != count; ++index) {
                parsed.ids.push_back(get_u64(
                    frame,
                    kReadWorkloadRecordHeaderBytes
                        + std::size_t{index}
                              * sizeof(std::uint64_t)));
            }
        } else {
            const auto expected =
                parsed.kind == kBatchKind
                    ? kReadWorkloadBatchRecordBytes
                    : kReadWorkloadRecordHeaderBytes;
            if ((parsed.kind != kBeginKind
                 && parsed.kind != kBatchKind
                 && parsed.kind != kEndKind
                 && parsed.kind != kCheckpointKind)
                || frame_bytes != expected) {
                result.corrupt = true;
                break;
            }
        }
        result.frames.push_back(std::move(parsed));
        cursor += frame_bytes;
    }
    return result;
}

struct ParsedSegment {
    std::uint64_t segment_sequence = 0;
    ParsedLog log;
};

[[nodiscard]] std::vector<ParsedSegment> parse_segments(
    const std::filesystem::path& directory) {
    std::vector<ParsedSegment> result;
    for (const auto& path : log_files(directory)) {
        const auto bytes = read_bytes(path);
        auto parsed = parse_log(bytes);
        require(
            !parsed.corrupt && !parsed.torn_tail,
            "workload segment is malformed");
        result.push_back(ParsedSegment{
            .segment_sequence = get_u64(parsed.header, 152),
            .log = std::move(parsed),
        });
    }
    std::sort(
        result.begin(),
        result.end(),
        [](const auto& left, const auto& right) {
            return left.segment_sequence < right.segment_sequence;
        });
    return result;
}

[[nodiscard]] std::vector<ParsedFrame> lifecycle_frames(
    const std::vector<ParsedSegment>& segments) {
    std::vector<ParsedFrame> result;
    for (const auto& segment : segments) {
        for (const auto& frame : segment.log.frames) {
            if (frame.kind != kCheckpointKind) {
                result.push_back(frame);
            }
        }
    }
    return result;
}

[[nodiscard]] std::vector<ParsedFrame> sequenced_frames(
    const std::vector<ParsedSegment>& segments) {
    std::vector<ParsedFrame> result;
    for (const auto& segment : segments) {
        for (const auto& frame : segment.log.frames) {
            if (frame.sequence != 0) {
                result.push_back(frame);
            }
        }
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

void require_contiguous_physical_sequences(
    const std::vector<ParsedFrame>& frames) {
    if (frames.empty()) {
        return;
    }
    auto expected = frames.front().sequence;
    require(expected != 0, "lifecycle record sequence is zero");
    for (const auto& frame : frames) {
        require_equal(
            frame.sequence,
            expected,
            "physical lifecycle sequence is not contiguous");
        ++expected;
    }
}

void test_v2_format_lifecycle_crc_digest_and_split() {
    TempDirectory temporary("format");
    auto configured = options(temporary.path(), 4);
    auto logger = ReadWorkloadLogger::open(
        configured, collection_with_reader());

    const std::array<std::uint64_t, 10> ids{
        7, 9, 11, 13, 17, 19, 23, 29, 31, 37};
    auto request = logger->begin_request(
        request_info(
            ReadWorkloadMethod::StreamGet,
            ids.size(),
            0x0102030405060708ULL,
            0x1112131415161718ULL));
    require(static_cast<bool>(request), "request begin was rejected");
    require(
        logger->try_append_ids(
            request,
            ids,
            ReadWorkloadIdChunkInfo{
                .logical_offset = 0,
                .batch_sequence = 7,
            }),
        "exact IDs were not accepted");

    ReadWorkloadBatchMetrics metrics;
    metrics.backend = ReadWorkloadBackend::IoUring;
    metrics.batch_sequence = 7;
    metrics.logical_offset = 0;
    metrics.rows = ids.size();
    metrics.response_bytes = 20'480;
    metrics.total_ns = 101;
    metrics.lookup_ns = 102;
    metrics.sort_ns = 103;
    metrics.plan_ns = 104;
    metrics.scheduler_wait_ns = 105;
    metrics.io_ns = 106;
    metrics.scatter_ns = 107;
    metrics.encode_ns = 108;
    metrics.write_ns = 109;
    metrics.logger_queue_ns = 110;
    metrics.unique_storage_segments = 201;
    metrics.physical_requests = 202;
    metrics.merged_ranges = 203;
    metrics.requested_bytes = 204;
    metrics.planned_bytes = 205;
    metrics.completed_bytes = 206;
    metrics.gap_bytes = 207;
    metrics.overread_bytes = 208;
    metrics.pages_touched = 209;
    metrics.page_runs = 210;
    metrics.layout_chunks_touched = 211;
    metrics.layout_chunk_runs = 212;
    metrics.discontinuities = 213;
    metrics.inversions = 214;
    metrics.max_range_bytes = 215;
    metrics.mean_range_bytes = 216;
    metrics.delivery_sequence = 301;
    metrics.useful_read_bytes = 302;
    metrics.backend_tasks = 303;
    require(
        logger->try_append_batch(request, metrics),
        "batch metrics were not accepted");
    logger->end_request(
        request,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
            .delivered_ids = ids.size(),
            .response_bytes = metrics.response_bytes,
        });
    require(!static_cast<bool>(request), "ended handle remained active");
    logger->shutdown();

    const auto segments = parse_segments(temporary.path());
    require_equal(segments.size(), std::size_t{1},
                  "unexpected segment count");
    const auto& header = segments.front().log.header;
    require_equal(get_u64(header, 32), collection().collection_id_low,
                  "collection ID low changed");
    require_equal(get_u64(header, 40), collection().collection_id_high,
                  "collection ID high changed");
    require_equal(get_u32(header, 48), std::uint32_t{7},
                  "format changed");
    require_equal(get_u32(header, 52), std::uint32_t{1},
                  "dtype changed");
    require_equal(get_u32(header, 56), std::uint32_t{1'024},
                  "dimension changed");
    require_equal(get_u32(header, 60), std::uint32_t{2'048},
                  "record size changed");
    require_equal(get_u32(header, 88), std::uint32_t{4},
                  "ID chunk rows changed");
    require_equal(get_u32(header, 92), configured.max_active_requests,
                  "active-request bound changed");
    require_equal(get_u64(header, 96), configured.buffer_bytes,
                  "buffer bound changed");
    require_equal(
        get_u32(header, 176),
        static_cast<std::uint32_t>(ReadWorkloadBackend::IoUring),
        "reader backend changed");
    require_equal(get_u32(header, 180), std::uint32_t{0},
                  "reader worker count changed");
    require_equal(get_u32(header, 184), std::uint32_t{0},
                  "reader queue capacity changed");
    require_equal(get_u32(header, 188), std::uint32_t{8},
                  "reader range task size changed");
    require_equal(get_u64(header, 192), std::uint64_t{1} << 20,
                  "maximum read range changed");
    require_equal(get_u64(header, 200), std::uint64_t{4'096},
                  "maximum gap changed");
    require_equal(get_u32(header, 208), std::uint32_t{32},
                  "io_uring queue depth changed");
    require_equal(get_u32(header, 212), std::uint32_t{2},
                  "io_uring worker count changed");
    require_equal(get_u32(header, 216), std::uint32_t{4'096},
                  "metric page size changed");
    require_equal(get_u32(header, 220), std::uint32_t{1} << 20,
                  "metric chunk size changed");
    require_equal(get_u32(header, 224), std::uint32_t{3},
                  "tuning override mask changed");
    require_equal(get_u32(header, 228), std::uint32_t{0},
                  "header reserved bytes are nonzero");

    const auto frames = lifecycle_frames(segments);
    require_equal(frames.size(), std::size_t{6},
                  "lifecycle frame count changed");
    require_contiguous_physical_sequences(
        sequenced_frames(segments));
    require_equal(frames[0].kind, kBeginKind, "BEGIN is not first");
    require_equal(frames[1].kind, kIdChunkKind, "chunk 0 is missing");
    require_equal(frames[2].kind, kIdChunkKind, "chunk 1 is missing");
    require_equal(frames[3].kind, kIdChunkKind, "chunk 2 is missing");
    require_equal(frames[4].kind, kBatchKind, "BATCH is missing");
    require_equal(frames[5].kind, kEndKind, "END is not last");

    const auto begin = std::span<const std::byte>(frames[0].bytes);
    require_equal(get_u32(begin, 28), std::uint32_t{3},
                  "BEGIN flags changed");
    require_equal(
        get_u32(begin, 32),
        static_cast<std::uint32_t>(ReadWorkloadMethod::StreamGet),
        "BEGIN method changed");
    require_equal(get_u32(begin, 36), std::uint32_t{128},
                  "BEGIN prefix changed");
    require_equal(get_u64(begin, 64), std::uint64_t{0x0102030405060708ULL},
                  "BEGIN request low changed");
    require_equal(get_u64(begin, 72), std::uint64_t{0x1112131415161718ULL},
                  "BEGIN request high changed");
    require_equal(get_u64(begin, 80), std::uint64_t{17},
                  "BEGIN catalog generation changed");
    require_equal(get_u64(begin, 88), std::uint64_t{19},
                  "BEGIN committed sequence changed");
    require_equal(get_u64(begin, 96), std::uint64_t{23},
                  "BEGIN row count changed");
    require_equal(get_u64(begin, 104), std::uint64_t{ids.size()},
                  "BEGIN declaration changed");
    require(get_u64(begin, 112) != 0,
            "BEGIN clock resolution is zero");

    std::vector<std::uint64_t> reconstructed;
    for (std::size_t index = 0; index != 3; ++index) {
        const auto& chunk = frames[index + 1];
        const auto bytes = std::span<const std::byte>(chunk.bytes);
        require_equal(
            get_u64(bytes, 80),
            static_cast<std::uint64_t>(index * 4),
            "chunk logical offset changed");
        require_equal(get_u64(bytes, 88), std::uint64_t{7},
                      "chunk batch sequence changed");
        require_equal(get_u64(bytes, 96), std::uint64_t{0},
                      "pre-read chunk has a delivery sequence");
        require_equal(
            get_u64(bytes, 104),
            static_cast<std::uint64_t>(index),
            "chunk sequence changed");
        reconstructed.insert(
            reconstructed.end(), chunk.ids.begin(), chunk.ids.end());
    }
    require(
        reconstructed
            == std::vector<std::uint64_t>(ids.begin(), ids.end()),
        "chunk split changed original ID order or values");

    const auto batch = std::span<const std::byte>(frames[4].bytes);
    require_equal(
        get_u32(batch, 32),
        static_cast<std::uint32_t>(ReadWorkloadBackend::IoUring),
        "batch backend changed");
    require_equal(get_u64(batch, 80), std::uint64_t{7},
                  "batch sequence changed");
    require_equal(get_u64(batch, 96), std::uint64_t{ids.size()},
                  "batch row count changed");
    require_equal(get_u64(batch, 104), metrics.response_bytes,
                  "batch response bytes changed");
    for (std::size_t index = 0; index != 10; ++index) {
        require_equal(
            get_u64(batch, 112 + index * 8),
            std::uint64_t{101 + index},
            "batch timing field changed");
    }
    for (std::size_t index = 0; index != 16; ++index) {
        const auto actual = get_u64(batch, 256 + index * 8);
        const auto expected = std::uint64_t{201 + index};
        require_equal(
            actual,
            expected,
            "batch physical metric changed at index "
                + std::to_string(index) + ": actual="
                + std::to_string(actual) + ", expected="
                + std::to_string(expected));
    }
    require_equal(get_u64(batch, 224), std::uint64_t{301},
                  "delivery sequence changed");
    require_equal(get_u64(batch, 232), std::uint64_t{302},
                  "useful read bytes changed");
    require_equal(get_u64(batch, 240), std::uint64_t{303},
                  "backend task count changed");
    require_equal(get_u64(batch, 248), std::uint64_t{0},
                  "batch reserved field is nonzero");

    const auto end = std::span<const std::byte>(frames[5].bytes);
    require_equal(get_u32(end, 28), std::uint32_t{1},
                  "successful END is not training-complete");
    require_equal(
        get_u32(end, 32),
        static_cast<std::uint32_t>(
            ReadWorkloadRequestStatus::Success),
        "END status changed");
    require_equal(get_u64(end, 80), std::uint64_t{ids.size()},
                  "END submitted count changed");
    require_equal(get_u64(end, 88), std::uint64_t{ids.size()},
                  "END delivered count changed");
    require_equal(get_u64(end, 96), std::uint64_t{3},
                  "END chunk count changed");
    require_equal(get_u64(end, 104), std::uint64_t{1},
                  "END batch count changed");
    require_equal(get_u64(end, 120), std::uint64_t{0},
                  "END unexpectedly reports dropped records");
    require_equal(get_u64(end, 128), std::uint64_t{0},
                  "END unexpectedly reports dropped IDs");
    require_equal(get_u64(end, 208), fnv1a(ids, kFnvLowOffset),
                  "END low digest changed");
    require_equal(get_u64(end, 216), fnv1a(ids, kFnvHighOffset),
                  "END high digest changed");

    const auto stats = logger->stats();
    require_equal(stats.observed_requests, std::uint64_t{1},
                  "request observation changed");
    require_equal(stats.begun_requests, std::uint64_t{1},
                  "request begin count changed");
    require_equal(stats.completed_requests, std::uint64_t{1},
                  "request completion count changed");
    require_equal(stats.complete_requests, std::uint64_t{1},
                  "training-complete count changed");
    require_equal(stats.incomplete_requests, std::uint64_t{0},
                  "successful request became incomplete");
    require_equal(stats.observed_id_chunks, std::uint64_t{3},
                  "observed chunk count changed");
    require_equal(stats.written_id_chunks, std::uint64_t{3},
                  "written chunk count changed");
    require_equal(stats.written_ids, std::uint64_t{ids.size()},
                  "written ID count changed");
    require_equal(stats.written_batches, std::uint64_t{1},
                  "written batch count changed");

    const auto path = log_files(temporary.path()).front();
    const auto original = read_bytes(path);
    auto torn = original;
    torn.insert(
        torn.end(),
        frames.front().bytes.begin(),
        frames.front().bytes.begin() + 41);
    const auto torn_parsed = parse_log(torn);
    require(
        torn_parsed.torn_tail && !torn_parsed.corrupt,
        "torn final record was not isolated");

    auto corrupt = original;
    std::size_t cursor = kReadWorkloadLogHeaderBytes;
    while (get_u32(
               std::span<const std::byte>(corrupt).subspan(cursor),
               20)
           != kIdChunkKind) {
        cursor += get_u32(
            std::span<const std::byte>(corrupt).subspan(cursor), 16);
    }
    corrupt[cursor + kReadWorkloadRecordHeaderBytes] ^= std::byte{1};
    require(
        parse_log(corrupt).corrupt,
        "ID payload corruption bypassed CRC32C");
}

void test_cancelled_and_abandoned_requests_are_not_training_data() {
    TempDirectory temporary("terminal-status");
    auto logger =
        ReadWorkloadLogger::open(options(temporary.path()), collection());
    const std::array<std::uint64_t, 3> ids{41, 43, 47};

    {
        auto request = logger->begin_request(
            request_info(ReadWorkloadMethod::Get, 3, 1, 2));
        require(static_cast<bool>(request), "cancelled request rejected");
        require(
            logger->try_append_ids(request, ids, {}),
            "cancelled request IDs rejected");
        logger->end_request(
            request,
            ReadWorkloadRequestEnd{
                .status = ReadWorkloadRequestStatus::Cancelled,
                .error_code = 17,
                .delivered_ids = 1,
            });
    }
    {
        auto request = logger->begin_request(
            request_info(ReadWorkloadMethod::Get, 3, 3, 4));
        require(static_cast<bool>(request), "abandoned request rejected");
        require(
            logger->try_append_ids(request, ids, {}),
            "abandoned request IDs rejected");
        // The move-only handle destructor must emit Shutdown and release
        // active admission exactly once.
    }
    logger->shutdown();

    const auto frames =
        lifecycle_frames(parse_segments(temporary.path()));
    std::vector<std::span<const std::byte>> ends;
    for (const auto& frame : frames) {
        if (frame.kind == kEndKind) {
            ends.emplace_back(frame.bytes);
        }
    }
    require_equal(ends.size(), std::size_t{2},
                  "terminal records are missing");
    require_equal(get_u32(ends[0], 28), std::uint32_t{0},
                  "cancelled request became training data");
    require_equal(
        get_u32(ends[0], 32),
        static_cast<std::uint32_t>(
            ReadWorkloadRequestStatus::Cancelled),
        "cancel status changed");
    require_equal(get_u32(ends[0], 36), std::uint32_t{17},
                  "terminal error code changed");
    require_equal(get_u32(ends[1], 28), std::uint32_t{0},
                  "abandoned request became training data");
    require_equal(
        get_u32(ends[1], 32),
        static_cast<std::uint32_t>(
            ReadWorkloadRequestStatus::Shutdown),
        "RAII terminal status changed");
    const auto stats = logger->stats();
    require_equal(stats.active_requests, std::uint64_t{0},
                  "RAII handle leaked active admission");
    require_equal(stats.completed_requests, std::uint64_t{2},
                  "terminal completion accounting changed");
    require_equal(stats.complete_requests, std::uint64_t{0},
                  "failed attempts became training-complete");
    require_equal(stats.incomplete_requests, std::uint64_t{2},
                  "failed attempts were not marked incomplete");
}

void test_concurrent_fifo_lifecycle_and_physical_sequence() {
    TempDirectory temporary("concurrent-fifo");
    auto configured = options(
        temporary.path(), kReadWorkloadMaximumIdChunkRows);
    configured.max_active_requests = 64;
    configured.buffer_bytes = 32ULL << 20;
    configured.segment_bytes = 8ULL << 20;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());

    constexpr std::size_t thread_count = 16;
    constexpr std::size_t requests_per_thread = 8;
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    for (std::size_t thread = 0; thread != thread_count; ++thread) {
        threads.emplace_back([&, thread] {
            std::vector<std::uint64_t> ids(
                kReadWorkloadMaximumIdChunkRows);
            for (std::size_t request_number = 0;
                 request_number != requests_per_thread;
                 ++request_number) {
                for (std::size_t index = 0; index != ids.size(); ++index) {
                    ids[index] =
                        (std::uint64_t{thread} << 56U)
                        | (std::uint64_t{request_number} << 48U)
                        | index;
                }
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                const auto request_low =
                    std::uint64_t{thread} * requests_per_thread
                    + request_number + 1;
                auto request = logger->begin_request(
                    request_info(
                        ReadWorkloadMethod::StreamGet,
                        ids.size(),
                        request_low,
                        0xfeedfaceULL));
                if (!request
                    || !logger->try_append_ids(request, ids, {})) {
                    failed.store(true, std::memory_order_relaxed);
                    if (request) {
                        logger->end_request(
                            request,
                            ReadWorkloadRequestEnd{
                                .status =
                                    ReadWorkloadRequestStatus::Failed,
                            });
                    }
                    continue;
                }
                ReadWorkloadBatchMetrics metrics;
                metrics.batch_sequence = 0;
                metrics.rows = ids.size();
                metrics.delivery_sequence = request_low;
                if (!logger->try_append_batch(request, metrics)) {
                    failed.store(true, std::memory_order_relaxed);
                }
                logger->end_request(
                    request,
                    ReadWorkloadRequestEnd{
                        .status = ReadWorkloadRequestStatus::Success,
                        .delivered_ids = ids.size(),
                    });
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }
    require(!failed.load(std::memory_order_relaxed),
            "lossless concurrency test dropped telemetry");
    logger->shutdown();

    const auto segments = parse_segments(temporary.path());
    const auto frames = lifecycle_frames(segments);
    require_contiguous_physical_sequences(
        sequenced_frames(segments));
    struct Lifecycle {
        bool begun = false;
        bool ended = false;
        std::uint64_t chunks = 0;
        std::uint64_t batches = 0;
    };
    std::unordered_map<std::uint64_t, Lifecycle> requests;
    for (const auto& frame : frames) {
        auto& lifecycle = requests[frame.request_low];
        if (frame.kind == kBeginKind) {
            require(
                !lifecycle.begun && !lifecycle.ended,
                "BEGIN was duplicated or physically reordered");
            lifecycle.begun = true;
        } else if (frame.kind == kIdChunkKind) {
            require(
                lifecycle.begun && !lifecycle.ended,
                "ID_CHUNK preceded BEGIN or followed END");
            require_equal(get_u64(frame.bytes, 104), lifecycle.chunks,
                          "per-request chunk sequence has a gap");
            ++lifecycle.chunks;
        } else if (frame.kind == kBatchKind) {
            require(
                lifecycle.begun && !lifecycle.ended,
                "BATCH preceded BEGIN or followed END");
            ++lifecycle.batches;
        } else if (frame.kind == kEndKind) {
            require(
                lifecycle.begun && !lifecycle.ended,
                "END preceded BEGIN or was duplicated");
            lifecycle.ended = true;
        }
    }
    require_equal(
        requests.size(),
        thread_count * requests_per_thread,
        "concurrent request count changed");
    for (const auto& [request_id, lifecycle] : requests) {
        static_cast<void>(request_id);
        require(
            lifecycle.begun && lifecycle.ended
                && lifecycle.chunks == 1
                && lifecycle.batches == 1,
            "concurrent lifecycle is incomplete");
    }
    const auto stats = logger->stats();
    require_equal(
        stats.complete_requests,
        std::uint64_t{thread_count * requests_per_thread},
        "concurrent complete-request count changed");
    require_equal(stats.active_requests, std::uint64_t{0},
                  "concurrent requests leaked admission");
}

void test_saturation_is_bounded_and_exactly_accounted() {
    TempDirectory temporary("saturation");
    auto configured = options(
        temporary.path(), kReadWorkloadMaximumIdChunkRows);
    const auto maximum_frame =
        kReadWorkloadRecordHeaderBytes
        + std::uint64_t{kReadWorkloadMaximumIdChunkRows}
              * sizeof(std::uint64_t);
    configured.buffer_bytes = maximum_frame * 2;
    configured.max_active_requests = 256;
    configured.max_write_bytes_per_second =
        kReadWorkloadLogHeaderBytes + maximum_frame
        + 2 * kReadWorkloadRecordHeaderBytes;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());

    constexpr std::size_t thread_count = 8;
    constexpr std::size_t calls_per_thread = 40;
    std::vector<std::uint64_t> ids(
        kReadWorkloadMaximumIdChunkRows, 7);
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (std::size_t thread = 0; thread != thread_count; ++thread) {
        threads.emplace_back([&, thread] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (std::size_t call = 0; call != calls_per_thread; ++call) {
                auto request = logger->begin_request(
                    request_info(
                        ReadWorkloadMethod::Get,
                        ids.size(),
                        std::uint64_t{thread} * calls_per_thread
                            + call + 1,
                        9));
                if (!request) {
                    continue;
                }
                const bool accepted =
                    logger->try_append_ids(request, ids, {});
                logger->end_request(
                    request,
                    ReadWorkloadRequestEnd{
                        .status =
                            accepted
                                ? ReadWorkloadRequestStatus::Success
                                : ReadWorkloadRequestStatus::Failed,
                        .delivered_ids =
                            accepted
                                ? static_cast<std::uint64_t>(ids.size())
                                : 0,
                    });
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }
    logger->shutdown();

    const auto stats = logger->stats();
    require_equal(
        stats.observed_requests,
        std::uint64_t{thread_count * calls_per_thread},
        "saturation request accounting changed");
    require(
        stats.dropped_full_records != 0,
        "bounded queue did not report saturation");
    require_equal(
        stats.completed_requests,
        stats.complete_requests + stats.incomplete_requests,
        "terminal request partition changed");
    require_equal(
        stats.observed_records,
        stats.enqueued_records + stats.dropped_full_records
            + stats.dropped_min_free_records
            + stats.dropped_runtime_failure_records,
        "producer record partition changed");
    require_equal(
        stats.observed_ids,
        stats.enqueued_ids + stats.dropped_full_ids
            + stats.dropped_min_free_ids
            + stats.dropped_runtime_failure_ids,
        "producer ID partition changed");
    require_equal(
        stats.enqueued_records,
        stats.written_records + stats.dropped_io_budget_records,
        "writer record partition changed");
    require_equal(
        stats.enqueued_ids,
        stats.written_ids + stats.dropped_io_budget_ids,
        "writer ID partition changed");
    require(
        stats.peak_queued_bytes <= stats.buffer_capacity_bytes,
        "queue exceeded its preallocated memory bound");
    require_equal(stats.active_requests, std::uint64_t{0},
                  "saturation leaked active admission");
}

void test_active_request_admission_and_move_release() {
    TempDirectory temporary("active-admission");
    auto configured = options(temporary.path());
    configured.max_active_requests = 2;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 1> ids{1};

    auto first = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 1, 1));
    auto second = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 2, 1));
    auto rejected = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 3, 1));
    require(first && second && !rejected,
            "active-request admission bound was not enforced");
    require_equal(logger->stats().active_requests, std::uint64_t{2},
                  "active request gauge changed");

    auto moved = std::move(first);
    require(!first && moved, "request handle move changed ownership");
    require(
        logger->try_append_ids(moved, ids, {})
            && logger->try_append_ids(second, ids, {}),
        "admitted request IDs were rejected");
    logger->end_request(
        moved,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
            .delivered_ids = 1,
        });
    logger->end_request(
        second,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
            .delivered_ids = 1,
        });
    require_equal(logger->stats().active_requests, std::uint64_t{0},
                  "end_request did not release admission");

    auto after_release = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 0, 4, 1));
    require(
        static_cast<bool>(after_release),
        "released admission was not reusable");
    logger->end_request(
        after_release,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
        });
    logger->shutdown();

    const auto stats = logger->stats();
    require_equal(stats.observed_requests, std::uint64_t{4},
                  "admission request count changed");
    require_equal(stats.begun_requests, std::uint64_t{3},
                  "admission begin count changed");
    require_equal(stats.completed_requests, std::uint64_t{4},
                  "admission completion count changed");
    require_equal(stats.complete_requests, std::uint64_t{3},
                  "admission complete count changed");
    require_equal(stats.incomplete_requests, std::uint64_t{1},
                  "rejected request was not incomplete");
    require_equal(stats.active_requests, std::uint64_t{0},
                  "active admission was released more or less than once");
}

void test_rotation_absolute_retention_and_permissions() {
    TempDirectory temporary("rotation-retention");
    auto configured = options(temporary.path(), 16);
    constexpr std::uint64_t maximum_frame =
        kReadWorkloadBatchRecordBytes;
    configured.buffer_bytes = maximum_frame * 512;
    configured.segment_bytes =
        kReadWorkloadLogHeaderBytes
        + 2 * kReadWorkloadRecordHeaderBytes
        + maximum_frame * 4;
    configured.retention_bytes = configured.segment_bytes * 2;
    configured.max_retained_files = 32;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 8> ids{
        1, 2, 3, 4, 5, 6, 7, 8};
    for (std::uint64_t request_id = 1; request_id <= 64; ++request_id) {
        auto request = logger->begin_request(
            request_info(ReadWorkloadMethod::Get, ids.size(), request_id, 5));
        require(
            static_cast<bool>(request),
            "retention seed request was rejected");
        require(
            logger->try_append_ids(request, ids, {}),
            "retention seed IDs were rejected");
        logger->end_request(
            request,
            ReadWorkloadRequestEnd{
                .status = ReadWorkloadRequestStatus::Success,
                .delivered_ids = ids.size(),
            });
    }
    logger->shutdown();

    const auto files = log_files(temporary.path());
    require(!files.empty(), "retention removed every segment");
    std::uint64_t actual_bytes = 0;
    struct stat status {};
    require(::stat(temporary.path().c_str(), &status) == 0,
            "cannot stat workload directory");
    require_equal(status.st_mode & 0777, mode_t{0700},
                  "workload directory permissions changed");
    for (const auto& path : files) {
        actual_bytes += std::filesystem::file_size(path);
        require(::stat(path.c_str(), &status) == 0,
                "cannot stat workload segment");
        require_equal(status.st_mode & 0777, mode_t{0600},
                      "workload segment permissions changed");
    }
    require(
        actual_bytes <= configured.retention_bytes,
        "recognized logs exceeded absolute retention bytes");
    const auto stats = logger->stats();
    require_equal(stats.retained_bytes, actual_bytes,
                  "retained byte diagnostics exclude active/finalized data");
    require_equal(
        stats.retained_segments,
        static_cast<std::uint64_t>(files.size()),
        "retained segment diagnostics changed");
    require(stats.rotated_segments != 0,
            "small segment did not rotate");
    require(stats.retention_deleted_segments != 0,
            "absolute retention did not delete old segments");

    const auto retained_segments = parse_segments(temporary.path());
    const auto frames = lifecycle_frames(retained_segments);
    require_contiguous_physical_sequences(
        sequenced_frames(retained_segments));
    require(
        ::stat(
            (temporary.path() / ".read-workload.writer.lock").c_str(),
            &status)
            == 0,
        "writer lock is missing");
    require_equal(status.st_mode & 0777, mode_t{0600},
                  "writer lock permissions changed");
}

void test_idle_rotation_and_retention_progress_without_queue_traffic() {
    TempDirectory temporary("idle-maintenance");
    auto configured = options(temporary.path());
    configured.rotate_seconds = 1;
    configured.retention_seconds = 1;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 1> ids{1};
    auto request = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 1, 7));
    require(
        static_cast<bool>(request),
        "idle-maintenance request rejected");
    require(logger->try_append_ids(request, ids, {}),
            "idle-maintenance IDs rejected");
    logger->end_request(
        request,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
            .delivered_ids = 1,
        });
    wait_until(
        [&] { return logger->stats().rotated_segments >= 1; },
        "idle time rotation did not run",
        6s);
    wait_until(
        [&] {
            return logger->stats().retention_deleted_segments >= 1
                   && log_files(temporary.path()).empty();
        },
        "idle age retention did not run",
        8s);
    logger->shutdown();
    require(
        log_files(temporary.path(), ".open").empty(),
        "clean shutdown left an unused active segment");
}

void test_min_free_is_fail_open_and_accounted() {
    TempDirectory temporary("min-free");
    auto configured = options(temporary.path());
    configured.min_free_bytes =
        std::numeric_limits<std::uint64_t>::max();
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    require(logger->stats().paused_for_min_free,
            "impossible free-space reserve did not pause logger");
    auto request = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 1, 1));
    require(!request, "paused logger admitted a request");
    const auto stats = logger->stats();
    require_equal(stats.observed_requests, std::uint64_t{1},
                  "paused request was not observed");
    require_equal(stats.completed_requests, std::uint64_t{1},
                  "paused request was not resolved");
    require_equal(stats.incomplete_requests, std::uint64_t{1},
                  "paused request was not marked incomplete");
    require_equal(stats.dropped_min_free_records, std::uint64_t{1},
                  "min-free record drop was not accounted");
    require(!stats.runtime_failed,
            "min-free pause became a runtime failure");
    logger->shutdown();
}

void test_exclusive_lock_and_collection_marker() {
    TempDirectory temporary("lock-marker");
    auto first =
        ReadWorkloadLogger::open(options(temporary.path()), collection());
    bool rejected = false;
    try {
        auto second =
            ReadWorkloadLogger::open(options(temporary.path()), collection());
        static_cast<void>(second);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "second writer bypassed directory lock");
    first->shutdown();
    first.reset();

    const auto marker =
        temporary.path() / ".read-workload.owner";
    require(std::filesystem::is_regular_file(marker),
            "collection ownership marker is missing");
    rejected = false;
    try {
        auto foreign = ReadWorkloadLogger::open(
            options(temporary.path()),
            collection(0x0102030405060708ULL));
        static_cast<void>(foreign);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected,
            "another collection claimed the workload directory");
    require(std::filesystem::is_regular_file(marker),
            "foreign open damaged ownership marker");
}

void test_runtime_io_failure_disables_only_logging() {
    TempDirectory temporary("runtime-failure");
    auto configured = options(temporary.path(), 16);
    configured.segment_bytes =
        kReadWorkloadLogHeaderBytes
        + 2 * kReadWorkloadRecordHeaderBytes
        + kReadWorkloadBatchRecordBytes * 3;
    auto logger =
        ReadWorkloadLogger::open(configured, collection());
    const std::array<std::uint64_t, 16> ids{};

    auto first = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, ids.size(), 1, 1));
    require(
        static_cast<bool>(first),
        "runtime-failure seed begin rejected");
    require(logger->try_append_ids(first, ids, {}),
            "runtime-failure seed IDs rejected");
    logger->end_request(
        first,
        ReadWorkloadRequestEnd{
            .status = ReadWorkloadRequestStatus::Success,
            .delivered_ids = ids.size(),
        });
    wait_until(
        [&] { return logger->stats().written_records >= 3; },
        "runtime-failure seed did not drain");

    const auto active = log_files(temporary.path(), ".open");
    require_equal(active.size(), std::size_t{1},
                  "active segment is missing");
    require(std::filesystem::remove(active.front()),
            "cannot unlink active segment for failure injection");

    for (std::uint64_t id = 2;
         id != 32 && !logger->stats().runtime_failed;
         ++id) {
        auto request = logger->begin_request(
            request_info(ReadWorkloadMethod::Get, ids.size(), id, 1));
        if (!request) {
            continue;
        }
        const bool accepted =
            logger->try_append_ids(request, ids, {});
        logger->end_request(
            request,
            ReadWorkloadRequestEnd{
                .status =
                    accepted ? ReadWorkloadRequestStatus::Success
                             : ReadWorkloadRequestStatus::Failed,
                .delivered_ids =
                    accepted
                        ? static_cast<std::uint64_t>(ids.size())
                        : 0,
            });
    }
    wait_until(
        [&] { return logger->stats().runtime_failed; },
        "segment finalization failure did not disable logger");
    auto rejected = logger->begin_request(
        request_info(ReadWorkloadMethod::Get, 1, 99, 1));
    require(!rejected, "failed logger admitted more telemetry");
    const auto stats = logger->stats();
    require(stats.write_errors != 0 && stats.last_error_number != 0,
            "runtime failure diagnostics are missing");
    logger->shutdown();
}

void test_invalid_bounds_are_rejected_before_worker_start() {
    TempDirectory temporary("invalid");
    auto configured = options(temporary.path());
    configured.id_chunk_rows =
        kReadWorkloadMaximumIdChunkRows + 1;
    bool rejected = false;
    try {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        static_cast<void>(logger);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "ID chunk hard maximum was bypassed");

    configured = options(temporary.path());
    configured.max_active_requests = 65'537;
    rejected = false;
    try {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        static_cast<void>(logger);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "active-request hard maximum was bypassed");

    configured = options(temporary.path());
    const auto maximum_frame = std::max<std::uint64_t>(
        kReadWorkloadBatchRecordBytes,
        kReadWorkloadRecordHeaderBytes
            + std::uint64_t{configured.id_chunk_rows}
                  * sizeof(std::uint64_t));
    configured.buffer_bytes = maximum_frame;
    rejected = false;
    try {
        auto logger =
            ReadWorkloadLogger::open(configured, collection());
        static_cast<void>(logger);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "one-slot queue bypassed minimum capacity");
}

void test_default_sequential_reader_context_is_accepted() {
    TempDirectory temporary("sequential-reader");
    auto logger = ReadWorkloadLogger::open(
        options(temporary.path()),
        collection_with_sequential_thread_pool_reader());
    require(
        !logger->stats().runtime_failed,
        "default sequential reader disabled the logger");
    logger->shutdown();
}

void test_impossible_native_reader_context_is_rejected() {
    TempDirectory temporary("invalid-reader");
    const auto require_rejected =
        [&](const ReadWorkloadCollectionInfo& context,
            const std::string& label) {
            bool rejected = false;
            try {
                auto logger = ReadWorkloadLogger::open(
                    options(temporary.path() / label),
                    context);
                static_cast<void>(logger);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, label + " reader context was accepted");
        };

    auto sequential = collection_with_sequential_thread_pool_reader();
    sequential.io_queue_capacity = 1;
    require_rejected(
        sequential,
        "sequential thread-pool queue");

    auto inactive_ring = collection_with_sequential_thread_pool_reader();
    inactive_ring.io_uring_queue_depth = 0;
    require_rejected(
        inactive_ring,
        "invalid inactive io_uring");

    auto ring = collection_with_reader();
    ring.io_workers = 1;
    require_rejected(
        ring,
        "io_uring thread-pool worker");
}

}  // namespace

int main() {
    try {
        test_v2_format_lifecycle_crc_digest_and_split();
        test_cancelled_and_abandoned_requests_are_not_training_data();
        test_concurrent_fifo_lifecycle_and_physical_sequence();
        test_saturation_is_bounded_and_exactly_accounted();
        test_active_request_admission_and_move_release();
        test_rotation_absolute_retention_and_permissions();
        test_idle_rotation_and_retention_progress_without_queue_traffic();
        test_min_free_is_fail_open_and_accounted();
        test_exclusive_lock_and_collection_marker();
        test_runtime_io_failure_disables_only_logging();
        test_invalid_bounds_are_rejected_before_worker_start();
        test_default_sequential_reader_context_is_accepted();
        test_impossible_native_reader_context_is_rejected();
        std::cout << "read workload log v2 tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "read workload log v2 test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
