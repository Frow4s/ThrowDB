#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/native_collection.hpp"
#include "throwdb/errors.hpp"
#include "throwdb/server/collection_engine.hpp"

namespace {

namespace server = throwdb::server;

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<std::uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-server-service-test-"
                   + std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count())
                   + "-" + std::to_string(sequence.fetch_add(1)));
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

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::vector<std::byte> rows(
    const std::size_t count,
    const std::size_t record_size,
    const std::uint8_t salt) {
    std::vector<std::byte> output(count * record_size);
    for (std::size_t row = 0; row != count; ++row) {
        for (std::size_t column = 0; column != record_size; ++column) {
            output[row * record_size + column] = static_cast<std::byte>(
                salt + static_cast<std::uint8_t>(31 * row + 7 * column));
        }
    }
    return output;
}

void test_engine_crud_get_and_scan() {
    TempDirectory temporary;
    const auto store = temporary.path() / "store";
    constexpr std::uint32_t dimension = 4;
    constexpr std::uint64_t record_size = dimension * 2;
    const std::array<std::uint64_t, 4> base_ids{10, 20, 30, 40};
    const auto base_rows = rows(base_ids.size(), record_size, 3);

    throwdb::NativeBuilderOptions options;
    options.format_version = 7;
    options.dimension = dimension;
    options.dtype = throwdb::DType::Float16;
    options.segment_size_bytes = 2 * record_size;
    auto builder = throwdb::NativeCollectionBuilder::create(store, options);
    builder->append(base_ids, base_rows);
    builder->finish();

    auto collection = throwdb::NativeCollection::open(store, {}, true);
    throwdb::server::SingleCollectionEngine engine(*collection, store);
    const auto description = engine.describe();
    require(description.format_version == 7, "Describe format mismatch");
    require(description.dimension == dimension, "Describe dimension mismatch");
    require(description.dtype == throwdb::DType::Float16, "Describe dtype mismatch");
    require(description.writable, "Describe omitted writable mode");
    require(description.durable_idempotency, "Describe omitted idempotency");
    require(description.record_size == record_size, "Describe record size mismatch");
    require(description.count == base_ids.size(), "Describe count mismatch");

    const std::array<std::uint64_t, 2> get_ids{40, 10};
    const auto get = engine.get(get_ids, 2);
    require(get.dimension == 2, "Get prefix mismatch");
    require(get.dtype == throwdb::DType::Float16, "Get dtype mismatch");
    require(get.row_count == get_ids.size(), "Get row count mismatch");
    const auto& get_bytes = get.embeddings;
    require(get_bytes.size() == 8, "Get response byte count mismatch");
    require(std::equal(get_bytes.begin(), get_bytes.begin() + 4,
                       base_rows.begin() + 3 * record_size),
            "Get first prefix is not bit exact");
    require(std::equal(get_bytes.begin() + 4, get_bytes.end(), base_rows.begin()),
            "Get second prefix is not bit exact");
    const throwdb::server::ProfiledGetResult profiled =
        engine.get_profiled(get_ids, 2);
    require(
        profiled.result.dimension == 2
            && profiled.result.row_count == get_ids.size()
            && std::equal(
                profiled.result.embeddings.begin(),
                profiled.result.embeddings.end(),
                get_bytes.begin(),
                get_bytes.end()),
        "profiled engine Get changed the projected response");
    require(
        profiled.read.requested_rows == get_ids.size()
            && profiled.read.base_rows == get_ids.size()
            && profiled.read.planned_ranges > 0
            && profiled.read.completed_read_bytes > 0,
        "profiled engine Get omitted physical read diagnostics");

    const std::array<std::uint64_t, 2> insert_ids{50, 60};
    const auto inserted_rows = rows(2, record_size, 71);
    const auto inserted = engine.mutate(
        server::MutationKind::Insert, {1, 3}, insert_ids, inserted_rows);
    require(inserted.commit_sequence > 0, "Insert returned zero commit sequence");
    require(inserted.row_count == 2, "Insert returned wrong row count");
    require(engine.get(insert_ids).embeddings == inserted_rows,
            "Insert did not preserve embeddings");

    const std::array<std::uint64_t, 1> update_ids{20};
    const auto updated_row = rows(1, record_size, 111);
    const auto updated = engine.mutate(
        server::MutationKind::Update, {1, 4}, update_ids, updated_row);
    require(updated.commit_sequence > inserted.commit_sequence,
            "Update did not advance commit sequence");
    require(engine.get(update_ids).embeddings == updated_row,
            "Update did not replace embeddings");

    const std::array<std::uint64_t, 2> delete_ids{30, 999};
    const auto deleted = engine.mutate(
        server::MutationKind::Delete, {1, 5}, delete_ids);
    require(deleted.row_count == 2, "Delete count mismatch");
    require(deleted.deleted == std::vector<std::uint8_t>({1, 0}),
            "Delete existing/missing mask mismatch");

    const std::array<std::uint64_t, 2> conflicting_ids{70, 10};
    bool conflict_rejected = false;
    try {
        static_cast<void>(engine.mutate(server::MutationKind::Insert,
            {1, 6}, conflicting_ids, rows(2, record_size, 151)));
    } catch (const throwdb::ExistingIdError&) {
        conflict_rejected = true;
    }
    require(conflict_rejected, "Insert conflict was not rejected");
    bool partial_insert = false;
    try {
        static_cast<void>(engine.get(std::span(conflicting_ids).first(1)));
        partial_insert = true;
    } catch (const throwdb::MissingIdError&) {
    }
    require(!partial_insert, "Conflicting Insert partially committed");

    auto scan = engine.open_scan();
    const auto metadata = scan->metadata();
    require(metadata.collection_id_low != 0 || metadata.collection_id_high != 0,
            "Scan has empty collection ID");
    require(metadata.live_count == 5, "Scan snapshot count mismatch");
    std::array<std::uint64_t, 2> ids{};
    std::array<throwdb::NativeIdLocation, 2> locations{};
    std::set<std::uint64_t> all_ids;
    std::pair<std::uint32_t, std::uint32_t> previous_location{0, 0};
    bool first_location = true;
    while (!scan->done()) {
        const auto count = scan->read_into(ids, locations);
        for (std::size_t index = 0; index < count; ++index) {
            const auto location = std::pair(
                locations[index].segment_id, locations[index].row_index);
            if (!first_location) {
                require(previous_location < location, "Scan locations are not increasing");
            }
            first_location = false;
            previous_location = location;
            require(all_ids.insert(ids[index]).second, "Scan duplicated a live ID");
        }
    }
    require(all_ids == std::set<std::uint64_t>({10, 20, 40, 50, 60}),
            "Scan returned the wrong logical ID set");

}

}  // namespace

int main() {
    try {
        test_engine_crud_get_and_scan();
        std::cout << "server engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "server engine test failure: " << error.what() << '\n';
        return 1;
    }
}
