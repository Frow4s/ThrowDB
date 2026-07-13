#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "throwdb/errors.hpp"
#include "../src/core/crc32c.hpp"
#include "../src/core/mutation_delta.hpp"

namespace {

using throwdb::detail::MutationDeltaBuildEntry;
using throwdb::detail::MutationDeltaBuildSpec;
using throwdb::detail::MutationDeltaDescriptor;
using throwdb::detail::NativeManifest;
using throwdb::detail::RowRef;
using throwdb::detail::Uuid128;
using throwdb::detail::VersionedMutationValue;

constexpr std::size_t kHeaderChecksumOffset = 20;
constexpr std::size_t kDescriptorIndexCrcOffset = 56 + 108;

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t sequence = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("throwdb-mutation-delta-test-"
                   + std::to_string(
                       std::chrono::steady_clock::now()
                           .time_since_epoch().count())
                   + "-" + std::to_string(sequence++));
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

[[nodiscard]] NativeManifest make_manifest() {
    NativeManifest manifest;
    manifest.collection_id = {
        0x0102030405060708ULL, 0x1112131415161718ULL};
    manifest.schema_id = {
        0x2122232425262728ULL, 0x3132333435363738ULL};
    manifest.active_wal_id = {
        0x4142434445464748ULL, 0x5152535455565758ULL};
    manifest.schema.dtype = throwdb::detail::DTypeCode::Float16;
    manifest.schema.dimension = 4;
    manifest.base_record_count = 0;
    manifest.segment_payload_capacity = 4096;
    throwdb::detail::derive_native_manifest_geometry(manifest);
    return manifest;
}

[[nodiscard]] MutationDeltaBuildSpec make_spec(
    const Uuid128 delta_id = {
        0x6162636465666768ULL, 0x7172737475767778ULL}) {
    MutationDeltaBuildSpec spec;
    spec.delta_id = delta_id;
    spec.source_predecessor_wal_id = {
        0x8182838485868788ULL, 0x9192939495969798ULL};
    spec.source_terminal_wal_id = {
        0xa1a2a3a4a5a6a7a8ULL, 0xb1b2b3b4b5b6b7b8ULL};
    spec.first_sequence = 3;
    spec.last_sequence = 9;
    spec.segment_id = 100;
    spec.source_wal_count = 2;
    return spec;
}

[[nodiscard]] std::vector<MutationDeltaBuildEntry> make_entries() {
    return {
        {std::numeric_limits<std::uint64_t>::max(),
         VersionedMutationValue::put(9, RowRef::create(8, 4))},
        {0, VersionedMutationValue::tombstone(4)},
        {42, VersionedMutationValue::put(5, RowRef::create(7, 3))}};
}

void fill_payload(
    const std::span<const RowRef> rows,
    const std::span<std::byte> destination,
    const std::uint64_t record_size) {
    require_equal(
        destination.size(), rows.size() * record_size,
        "payload callback received wrong destination geometry");
    for (std::size_t row = 0; row != rows.size(); ++row) {
        for (std::size_t byte = 0; byte != record_size; ++byte) {
            destination[row * record_size + byte] = static_cast<std::byte>(
                (rows[row].raw() + byte * 17) & 0xffU);
        }
    }
}

[[nodiscard]] std::vector<std::byte> read_bytes(
    const std::filesystem::path& path) {
    const std::uint64_t size = std::filesystem::file_size(path);
    require(
        size <= std::numeric_limits<std::size_t>::max(),
        "test file does not fit memory");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open test file");
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(
        stream.gcount() == static_cast<std::streamsize>(bytes.size()),
        "cannot read complete test file");
    return bytes;
}

void write_bytes(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good(), "cannot open test file for writing");
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    require(stream.good(), "cannot write complete test file");
}

[[nodiscard]] std::uint32_t header_crc(
    const std::span<const std::byte> bytes) {
    throwdb::detail::Crc32c checksum;
    checksum.update(bytes.first(kHeaderChecksumOffset));
    checksum.update_zeroes(sizeof(std::uint32_t));
    checksum.update(bytes.subspan(
        kHeaderChecksumOffset + sizeof(std::uint32_t),
        throwdb::detail::kMutationDeltaHeaderSize
            - kHeaderChecksumOffset - sizeof(std::uint32_t)));
    return checksum.value();
}

void reseal_header(const std::span<std::byte> bytes) {
    require(
        bytes.size() >= throwdb::detail::kMutationDeltaHeaderSize,
        "cannot reseal a truncated test header");
    auto header = bytes.first(throwdb::detail::kMutationDeltaHeaderSize);
    throwdb::detail::put_u32(header, kHeaderChecksumOffset, 0);
    throwdb::detail::put_u32(header, kHeaderChecksumOffset, header_crc(header));
}

void reseal_index_body(
    const std::filesystem::path& data_path,
    const std::filesystem::path& index_path,
    MutationDeltaDescriptor& descriptor) {
    std::vector<std::byte> index = read_bytes(index_path);
    descriptor.index_body_crc32c = throwdb::detail::crc32c(
        std::span<const std::byte>(index).subspan(
            throwdb::detail::kMutationDeltaHeaderSize));
    throwdb::detail::put_u32(
        index, kDescriptorIndexCrcOffset, descriptor.index_body_crc32c);
    throwdb::detail::put_u32(index, 180, descriptor.index_body_crc32c);
    reseal_header(index);
    write_bytes(index_path, index);

    std::vector<std::byte> data = read_bytes(data_path);
    throwdb::detail::put_u32(
        data, kDescriptorIndexCrcOffset, descriptor.index_body_crc32c);
    reseal_header(data);
    write_bytes(data_path, data);
}

[[nodiscard]] MutationDeltaDescriptor build_fixture(
    const std::filesystem::path& path,
    const NativeManifest& manifest,
    const Uuid128 delta_id) {
    const auto entries = make_entries();
    return throwdb::detail::write_mutation_delta(
        path, manifest, make_spec(delta_id), entries,
        [&](const auto rows, const auto destination) {
            fill_payload(rows, destination, manifest.record_size);
        },
        16);
}

void test_roundtrip_streaming_geometry_and_paths() {
    TempDirectory temporary;
    const NativeManifest manifest = make_manifest();
    const MutationDeltaBuildSpec spec = make_spec();
    const auto entries = make_entries();
    std::size_t callback_count = 0;
    std::size_t maximum_batch = 0;
    const MutationDeltaDescriptor descriptor =
        throwdb::detail::write_mutation_delta(
            temporary.path(), manifest, spec, entries,
            [&](const auto rows, const auto destination) {
                ++callback_count;
                maximum_batch = std::max(maximum_batch, destination.size());
                fill_payload(rows, destination, manifest.record_size);
            },
            manifest.record_size);

    require_equal(descriptor.entry_count, std::uint64_t{3},
                  "descriptor entry count is wrong");
    require_equal(descriptor.put_count, std::uint64_t{2},
                  "descriptor PUT count is wrong");
    require_equal(callback_count, std::size_t{2},
                  "payload was not streamed in requested batches");
    require(maximum_batch <= manifest.record_size,
            "payload callback exceeded its batch bound");
    require_equal(
        descriptor.data_file_size,
        throwdb::detail::kMutationDeltaHeaderSize
            + 2 * manifest.record_size,
        "data geometry is wrong");
    require_equal(
        descriptor.index_file_size,
        throwdb::detail::kMutationDeltaIndexAlignment
            * 2 + 2 * throwdb::detail::kMutationDeltaReverseRecordSize,
        "index geometry is wrong");

    const auto data_path = throwdb::detail::mutation_delta_data_path(
        temporary.path(), spec.delta_id);
    const auto index_path = throwdb::detail::mutation_delta_index_path(
        temporary.path(), spec.delta_id);
    require_equal(
        data_path.filename(),
        std::filesystem::path(
            "delta-71727374757677786162636465666768.dat"),
        "data UUID path is unstable");
    require_equal(
        index_path.filename(),
        std::filesystem::path(
            "delta-71727374757677786162636465666768.idx"),
        "index UUID path is unstable");
    require_equal(
        std::filesystem::file_size(data_path), descriptor.data_file_size,
        "published data size is wrong");
    require_equal(
        std::filesystem::file_size(index_path), descriptor.index_file_size,
        "published index size is wrong");

    auto opened = throwdb::detail::open_mutation_delta(
        temporary.path(), manifest, descriptor, true);
    require_equal(opened.descriptor, descriptor,
                  "opened descriptor changed");
    require_equal(opened.records.size(), std::size_t{3},
                  "opened record count is wrong");
    require_equal(opened.records[0].id, std::uint64_t{0},
                  "records are not sorted by ID");
    require(opened.records[0].value.is_tombstone(),
            "first record is not a tombstone");
    require_equal(opened.records[1].id, std::uint64_t{42},
                  "second sorted ID is wrong");
    require_equal(
        opened.records[1].value.row_ref().value(), RowRef::create(100, 0),
        "first PUT did not resolve to delta row zero");
    require_equal(
        opened.records[2].id, std::numeric_limits<std::uint64_t>::max(),
        "maximum uint64 ID was not preserved");
    require_equal(
        opened.records[2].value.row_ref().value(), RowRef::create(100, 1),
        "second PUT did not resolve to delta row one");

    std::vector<std::byte> payload(
        static_cast<std::size_t>(2 * manifest.record_size));
    throwdb::detail::pread_exact(
        opened.data_fd.get(), payload.data(), payload.size(),
        opened.data_header.payload_offset, data_path);
    std::vector<std::byte> expected(payload.size());
    const std::array<RowRef, 2> source_rows{
        RowRef::create(7, 3), RowRef::create(8, 4)};
    fill_payload(source_rows, expected, manifest.record_size);
    require_equal(payload, expected, "streamed payload bytes changed");
}

void test_empty_header_only_run() {
    TempDirectory temporary;
    const NativeManifest manifest = make_manifest();
    const std::vector<MutationDeltaBuildEntry> entries;
    const MutationDeltaDescriptor descriptor =
        throwdb::detail::write_mutation_delta(
            temporary.path(), manifest, make_spec(), entries, {});
    require_equal(descriptor.entry_count, std::uint64_t{0},
                  "empty run has entries");
    require_equal(descriptor.put_count, std::uint64_t{0},
                  "empty run has PUT rows");
    require_equal(
        descriptor.data_file_size,
        throwdb::detail::kMutationDeltaHeaderSize,
        "empty data run is not header-only");
    require_equal(
        descriptor.index_file_size,
        throwdb::detail::kMutationDeltaHeaderSize,
        "empty index run is not header-only");
    require_equal(descriptor.data_payload_crc32c, std::uint32_t{0},
                  "empty payload checksum is not canonical");
    require_equal(descriptor.index_body_crc32c, std::uint32_t{0},
                  "empty index checksum is not canonical");
    const auto opened = throwdb::detail::open_mutation_delta(
        temporary.path(), manifest, descriptor, true);
    require(opened.records.empty(), "empty delta opened with records");
}

void test_writer_rejects_invalid_metadata_before_publication() {
    TempDirectory temporary;
    const NativeManifest manifest = make_manifest();
    auto duplicate = make_entries();
    duplicate.push_back({
        42, VersionedMutationValue::tombstone(6)});
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(throwdb::detail::write_mutation_delta(
                temporary.path(), manifest, make_spec(), duplicate, {}));
        },
        "duplicate IDs were accepted");

    auto out_of_range = make_entries();
    out_of_range[0] = {
        out_of_range[0].id,
        VersionedMutationValue::put(10, RowRef::create(8, 4))};
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(throwdb::detail::write_mutation_delta(
                temporary.path(), manifest, make_spec(), out_of_range,
                [](auto, auto) {}));
        },
        "out-of-range sequence was accepted");

    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            const auto entries = make_entries();
            static_cast<void>(throwdb::detail::write_mutation_delta(
                temporary.path(), manifest, make_spec(), entries, {}));
        },
        "PUT run without payload callback was accepted");
    require(
        !std::filesystem::exists(
            throwdb::detail::mutation_deltas_directory(temporary.path())),
        "invalid build published a delta directory");
}

void test_immutable_destination_and_exact_file_sizes() {
    TempDirectory temporary;
    const NativeManifest manifest = make_manifest();
    const Uuid128 delta_id{1, 2};
    const MutationDeltaDescriptor descriptor =
        build_fixture(temporary.path(), manifest, delta_id);
    require_throws<throwdb::InvalidArgumentError>(
        [&] {
            static_cast<void>(
                build_fixture(temporary.path(), manifest, delta_id));
        },
        "immutable delta UUID was overwritten");

    const auto data_path = throwdb::detail::mutation_delta_data_path(
        temporary.path(), delta_id);
    std::filesystem::resize_file(
        data_path, descriptor.data_file_size - 1);
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::open_mutation_delta(
                temporary.path(), manifest, descriptor));
        },
        "truncated data file was accepted");
}

void test_payload_checksum_is_optional_but_scrubbable() {
    TempDirectory temporary;
    const NativeManifest manifest = make_manifest();
    const Uuid128 delta_id{3, 4};
    const MutationDeltaDescriptor descriptor =
        build_fixture(temporary.path(), manifest, delta_id);
    const auto data_path = throwdb::detail::mutation_delta_data_path(
        temporary.path(), delta_id);
    std::vector<std::byte> data = read_bytes(data_path);
    data[throwdb::detail::kMutationDeltaHeaderSize] ^= std::byte{0x40};
    write_bytes(data_path, data);

    const auto normal = throwdb::detail::open_mutation_delta(
        temporary.path(), manifest, descriptor, false);
    require_equal(normal.records.size(), std::size_t{3},
                  "normal open failed after payload-only damage");
    require_throws<throwdb::CorruptionError>(
        [&] {
            static_cast<void>(throwdb::detail::open_mutation_delta(
                temporary.path(), manifest, descriptor, true));
        },
        "payload scrub did not detect corruption");
}

void test_index_checksum_and_structural_corruption() {
    const NativeManifest manifest = make_manifest();

    {
        TempDirectory temporary;
        const Uuid128 delta_id{5, 6};
        const MutationDeltaDescriptor descriptor =
            build_fixture(temporary.path(), manifest, delta_id);
        const auto index_path = throwdb::detail::mutation_delta_index_path(
            temporary.path(), delta_id);
        std::vector<std::byte> index = read_bytes(index_path);
        index[throwdb::detail::kMutationDeltaHeaderSize + 8] ^=
            std::byte{0x01};
        write_bytes(index_path, index);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_mutation_delta(
                    temporary.path(), manifest, descriptor));
            },
            "index body checksum corruption was accepted");
    }

    {
        TempDirectory temporary;
        const Uuid128 delta_id{7, 8};
        MutationDeltaDescriptor descriptor =
            build_fixture(temporary.path(), manifest, delta_id);
        const auto data_path = throwdb::detail::mutation_delta_data_path(
            temporary.path(), delta_id);
        const auto index_path = throwdb::detail::mutation_delta_index_path(
            temporary.path(), delta_id);
        std::vector<std::byte> index = read_bytes(index_path);
        // The second version record is the first PUT and must name payload row
        // zero. Make it row one, then update every checksum binding so the
        // structural validator, not the CRC, is exercised.
        throwdb::detail::put_u64(
            index,
            throwdb::detail::kMutationDeltaHeaderSize
                + throwdb::detail::kMutationDeltaVersionRecordSize + 16,
            1);
        write_bytes(index_path, index);
        reseal_index_body(data_path, index_path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_mutation_delta(
                    temporary.path(), manifest, descriptor));
            },
            "non-dense payload rows were accepted");
    }

    {
        TempDirectory temporary;
        const Uuid128 delta_id{9, 10};
        MutationDeltaDescriptor descriptor =
            build_fixture(temporary.path(), manifest, delta_id);
        const auto data_path = throwdb::detail::mutation_delta_data_path(
            temporary.path(), delta_id);
        const auto index_path = throwdb::detail::mutation_delta_index_path(
            temporary.path(), delta_id);
        std::vector<std::byte> index = read_bytes(index_path);
        const std::uint64_t reverse_offset =
            throwdb::detail::align_up(
                throwdb::detail::kMutationDeltaHeaderSize
                    + descriptor.entry_count
                          * throwdb::detail::kMutationDeltaVersionRecordSize,
                throwdb::detail::kMutationDeltaIndexAlignment);
        index[static_cast<std::size_t>(reverse_offset)] ^= std::byte{0x20};
        write_bytes(index_path, index);
        reseal_index_body(data_path, index_path, descriptor);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_mutation_delta(
                    temporary.path(), manifest, descriptor));
            },
            "mismatched reverse metadata was accepted");
    }
}

void test_resealed_header_geometry_and_binding_corruption() {
    const NativeManifest manifest = make_manifest();
    {
        TempDirectory temporary;
        const Uuid128 delta_id{11, 12};
        const MutationDeltaDescriptor descriptor =
            build_fixture(temporary.path(), manifest, delta_id);
        const auto data_path = throwdb::detail::mutation_delta_data_path(
            temporary.path(), delta_id);
        std::vector<std::byte> data = read_bytes(data_path);
        throwdb::detail::put_u64(data, 200, 8192);
        reseal_header(data);
        write_bytes(data_path, data);
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_mutation_delta(
                    temporary.path(), manifest, descriptor));
            },
            "resealed invalid payload offset was accepted");
    }

    {
        TempDirectory temporary;
        const Uuid128 delta_id{13, 14};
        const MutationDeltaDescriptor descriptor =
            build_fixture(temporary.path(), manifest, delta_id);
        NativeManifest wrong = manifest;
        wrong.schema_id.low ^= 1;
        require_throws<throwdb::CorruptionError>(
            [&] {
                static_cast<void>(throwdb::detail::open_mutation_delta(
                    temporary.path(), wrong, descriptor));
            },
            "manifest schema mismatch was accepted");
    }
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        std::function<void()> run;
    };
    const std::array tests{
        TestCase{"roundtrip_streaming_geometry_and_paths",
                 test_roundtrip_streaming_geometry_and_paths},
        TestCase{"empty_header_only_run", test_empty_header_only_run},
        TestCase{"writer_rejects_invalid_metadata_before_publication",
                 test_writer_rejects_invalid_metadata_before_publication},
        TestCase{"immutable_destination_and_exact_file_sizes",
                 test_immutable_destination_and_exact_file_sizes},
        TestCase{"payload_checksum_is_optional_but_scrubbable",
                 test_payload_checksum_is_optional_but_scrubbable},
        TestCase{"index_checksum_and_structural_corruption",
                 test_index_checksum_and_structural_corruption},
        TestCase{"resealed_header_geometry_and_binding_corruption",
                 test_resealed_header_geometry_and_binding_corruption},
    };
    std::size_t passed = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            ++passed;
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << test.name << ": " << error.what()
                      << '\n';
            return 1;
        }
    }
    std::cout << "passed " << passed << " mutation delta tests\n";
    return 0;
}
