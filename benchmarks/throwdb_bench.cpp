#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "throwdb/storage.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t parse_positive(
    const std::string_view text,
    const std::string_view name) {
    std::uint64_t result = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result == 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return result;
}

std::vector<std::uint64_t> read_raw_ids(const std::filesystem::path& path) {
    const auto byte_count = std::filesystem::file_size(path);
    if (byte_count == 0 || byte_count % sizeof(std::uint64_t) != 0) {
        throw std::runtime_error("ID file must contain native-endian raw uint64 values");
    }
    if (byte_count
        > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("ID file is too large for this benchmark process");
    }
    std::vector<std::uint64_t> ids(
        static_cast<std::size_t>(byte_count / sizeof(std::uint64_t)));
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open ID file: " + path.string());
    }
    input.read(
        reinterpret_cast<char*>(ids.data()),
        static_cast<std::streamsize>(byte_count));
    if (!input) {
        throw std::runtime_error("could not read complete ID file: " + path.string());
    }
    return ids;
}

double percentile_ms(
    const std::vector<std::uint64_t>& sorted_samples_ns,
    const double percentile) {
    const auto rank = static_cast<std::size_t>(
        percentile / 100.0
        * static_cast<double>(sorted_samples_ns.size() - 1));
    return static_cast<double>(sorted_samples_ns[rank]) / 1'000'000.0;
}

struct Options {
    std::filesystem::path storage;
    std::filesystem::path ids;
    std::size_t batch_size = 1024;
    std::size_t calls = 100;
    std::uint64_t seed = 20260711;
};

Options parse_options(const int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        throw std::invalid_argument(
            "usage: throwdb_bench STORE IDS_RAW [BATCH_SIZE] [CALLS] [SEED]");
    }
    Options options;
    options.storage = argv[1];
    options.ids = argv[2];
    if (argc >= 4) {
        options.batch_size = static_cast<std::size_t>(
            parse_positive(argv[3], "BATCH_SIZE"));
    }
    if (argc >= 5) {
        options.calls =
            static_cast<std::size_t>(parse_positive(argv[4], "CALLS"));
    }
    if (argc >= 6) {
        options.seed = parse_positive(argv[5], "SEED");
    }
    if (options.batch_size
        > std::numeric_limits<std::size_t>::max() / options.calls) {
        throw std::invalid_argument("BATCH_SIZE * CALLS overflows size_t");
    }
    return options;
}

int run(const Options& options) {
    auto storage = throwdb::Storage::open(options.storage);
    const auto population = read_raw_ids(options.ids);

    // Sampling happens before the timed section. Every ID comes from the input
    // population, so strict missing-ID semantics cannot truncate a run.
    std::mt19937_64 random(options.seed);
    std::uniform_int_distribution<std::size_t> choose(
        0, population.size() - 1);
    std::vector<std::uint64_t> queries(options.batch_size * options.calls);
    for (auto& id : queries) {
        id = population[choose(random)];
    }
    if (options.batch_size
        > std::numeric_limits<std::size_t>::max()
              / static_cast<std::size_t>(storage->record_size())) {
        throw std::overflow_error("output buffer size overflows size_t");
    }
    std::vector<std::byte> output(
        options.batch_size * static_cast<std::size_t>(storage->record_size()));

    constexpr std::size_t warmup_calls = 3;
    const auto warmup_query = std::span<const std::uint64_t>(queries).first(
        options.batch_size);
    for (std::size_t call = 0; call < warmup_calls; ++call) {
        storage->get_into(warmup_query, output);
    }

    std::vector<std::uint64_t> latency_ns;
    latency_ns.reserve(options.calls);
    const auto wall_start = Clock::now();
    for (std::size_t call = 0; call < options.calls; ++call) {
        const auto query = std::span<const std::uint64_t>(queries).subspan(
            call * options.batch_size, options.batch_size);
        const auto start = Clock::now();
        storage->get_into(query, output);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start);
        latency_ns.push_back(static_cast<std::uint64_t>(elapsed.count()));
    }
    const auto wall_elapsed =
        std::chrono::duration<double>(Clock::now() - wall_start).count();
    std::sort(latency_ns.begin(), latency_ns.end());

    const auto embedding_count = static_cast<double>(options.batch_size)
                                 * static_cast<double>(options.calls);
    const auto useful_bytes = embedding_count
                              * static_cast<double>(storage->record_size());
    std::cout << std::fixed << std::setprecision(3)
              << "{\n"
              << "  \"storage_count\": " << storage->count() << ",\n"
              << "  \"dimension\": " << storage->dimension() << ",\n"
              << "  \"batch_size\": " << options.batch_size << ",\n"
              << "  \"calls\": " << options.calls << ",\n"
              << "  \"seconds\": " << wall_elapsed << ",\n"
              << "  \"embeddings_per_second\": "
              << embedding_count / wall_elapsed << ",\n"
              << "  \"useful_gib_per_second\": "
              << useful_bytes / wall_elapsed / static_cast<double>(1ULL << 30)
              << ",\n"
              << "  \"latency_p50_ms\": " << percentile_ms(latency_ns, 50)
              << ",\n"
              << "  \"latency_p95_ms\": " << percentile_ms(latency_ns, 95)
              << ",\n"
              << "  \"latency_p99_ms\": " << percentile_ms(latency_ns, 99)
              << ",\n"
              << "  \"latency_max_ms\": "
              << static_cast<double>(latency_ns.back()) / 1'000'000.0 << "\n"
              << "}\n";
    return 0;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "throwdb_bench: " << error.what() << '\n';
        return 1;
    }
}
