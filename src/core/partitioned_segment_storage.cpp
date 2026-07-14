#include "partitioned_segment_storage.hpp"

#include "crc32c.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

constexpr std::size_t kChecksumBufferBytes = 1U << 20;
constexpr std::uint64_t kMaximumSequence =
    std::numeric_limits<std::uint64_t>::max() - 1;

template <typename Error>
void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left + right;
}

template <typename Error>
[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    if (right != 0
        && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw Error(std::string(description) + " overflows uint64");
    }
    return left * right;
}

[[nodiscard]] std::string uuid_token(const Uuid128 value) {
    constexpr std::array<char, 16> hex{
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::array<char, 32> result{};
    const auto append = [&result, &hex](
                            const std::uint64_t word,
                            const std::size_t offset) {
        for (std::size_t index = 0; index != 16; ++index) {
            const unsigned shift = static_cast<unsigned>((15 - index) * 4);
            result[offset + index] = hex[(word >> shift) & 0xfU];
        }
    };
    append(value.high, 0);
    append(value.low, 16);
    return std::string(result.begin(), result.end());
}

[[nodiscard]] Uuid128 make_uuid() {
    const SnapshotId value = make_snapshot_id();
    return Uuid128{value.low, value.high};
}

[[nodiscard]] int nofollow_flag() noexcept {
#if defined(O_NOFOLLOW)
    return O_NOFOLLOW;
#else
    return 0;
#endif
}

void validate_segments_directory(
    const std::filesystem::path& path,
    const bool persisted_input) {
    struct stat status {};
    if (::lstat(path.c_str(), &status) < 0) {
        throw StorageError(system_error_message(
            "cannot inspect partitioned segments directory", path));
    }
    if (!S_ISDIR(status.st_mode)) {
        if (persisted_input) {
            throw CorruptionError(
                "partitioned segments path is not a real directory");
        }
        throw InvalidArgumentError(
            "partitioned segments path is not a real directory");
    }
}

[[nodiscard]] FileDescriptor open_regular_nofollow(
    const std::filesystem::path& path) {
    const int fd = ::open(
        path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | nofollow_flag());
    if (fd < 0) {
#if defined(ELOOP)
        if (errno == ELOOP) {
            throw CorruptionError(
                "partitioned segment path is a symlink: '"
                + path.string() + "'");
        }
#endif
        throw StorageError(
            system_error_message("cannot open partitioned segment", path));
    }
    FileDescriptor result(fd);
    struct stat status {};
    if (::fstat(result.get(), &status) < 0) {
        throw StorageError(
            system_error_message("cannot inspect partitioned segment", path));
    }
    require<CorruptionError>(
        S_ISREG(status.st_mode),
        "partitioned segment resource is not a regular file");
    return result;
}

[[nodiscard]] std::array<std::byte, kPartitionedHeaderSize> read_header(
    const int fd,
    const std::filesystem::path& path) {
    std::array<std::byte, kPartitionedHeaderSize> bytes{};
    pread_exact(fd, bytes.data(), bytes.size(), 0, path);
    return bytes;
}

[[nodiscard]] std::uint32_t checksum_payload(
    const int fd,
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::uint64_t size) {
    Crc32c checksum;
    std::vector<std::byte> buffer(kChecksumBufferBytes);
    std::uint64_t cursor = 0;
    while (cursor != size) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), size - cursor));
        pread_exact(fd, buffer.data(), count, offset + cursor, path);
        checksum.update(
            std::span<const std::byte>(buffer.data(), count));
        cursor += count;
    }
    return checksum.value();
}

class ReadOnlyMapping {
public:
    ReadOnlyMapping() = default;
    ~ReadOnlyMapping() { reset(); }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

    void map(
        const int fd,
        const std::uint64_t size,
        const std::filesystem::path& path) {
        require<CorruptionError>(
            size != 0 && size <= std::numeric_limits<std::size_t>::max(),
            "segment IDs sidecar does not fit the process address space");
        address_ = ::mmap(
            nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE,
            fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw StorageError(system_error_message(
                "cannot mmap segment IDs sidecar", path));
        }
        size_ = static_cast<std::size_t>(size);
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {static_cast<const std::byte*>(address_), size_};
    }

    void reset() noexcept {
        if (address_ != nullptr) {
            static_cast<void>(::munmap(address_, size_));
            address_ = nullptr;
            size_ = 0;
        }
    }

private:
    void* address_{nullptr};
    std::size_t size_{0};
};

void validate_supported_kind(
    const PartitionedSegmentKind kind,
    const bool persisted_input) {
    if (kind == PartitionedSegmentKind::Materialized
        || kind == PartitionedSegmentKind::Cleaner) {
        return;
    }
    if (persisted_input) {
        throw CorruptionError(
            "segment-set storage supports only materialized and cleaner "
            "segments");
    }
    throw InvalidArgumentError(
        "segment-set storage supports only materialized and cleaner segments");
}

void validate_sequence(
    const PartitionedSegmentKind kind,
    const std::uint64_t sequence) {
    require<InvalidArgumentError>(
        sequence <= kMaximumSequence,
        "segment row sequence uses the reserved UINT64_MAX value");
    if (kind == PartitionedSegmentKind::Materialized) {
        require<InvalidArgumentError>(
            sequence != 0,
            "materialized segment row sequence must be positive");
    }
}

void validate_manifest_for_build(const PartitionedManifestV2& manifest) {
    static_cast<void>(encode_partitioned_manifest_v2(manifest));
    require<InvalidArgumentError>(
        manifest.record_size <= std::numeric_limits<std::size_t>::max(),
        "segment record size does not fit the process address space");
}

void validate_manifest_for_open(const PartitionedManifestV2& manifest) {
    try {
        validate_manifest_for_build(manifest);
    } catch (const InvalidArgumentError& error) {
        throw CorruptionError(error.what());
    }
}

[[nodiscard]] std::filesystem::path staging_path(
    const std::filesystem::path& final_path) {
    return final_path.parent_path()
           / ("." + final_path.filename().string() + ".staging");
}

void remove_nothrow(const std::filesystem::path& path) noexcept {
    if (path.empty()) {
        return;
    }
    std::error_code ignored;
    static_cast<void>(std::filesystem::remove(path, ignored));
}

void remove_exact(const std::filesystem::path& path) {
    if (::unlink(path.c_str()) < 0 && errno != ENOENT) {
        throw StorageError(
            system_error_message("cannot remove segment staging file", path));
    }
}

void link_exclusive(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
    if (::link(source.c_str(), destination.c_str()) < 0) {
        throw StorageError(
            system_error_message("cannot publish immutable segment", destination));
    }
}

[[nodiscard]] PartitionedSegmentDataHeaderV1 expected_data_header(
    const PartitionedManifestV2& manifest,
    const PartitionedSegmentDescriptor& descriptor) {
    PartitionedSegmentDataHeaderV1 header;
    header.collection_id = manifest.collection_id;
    header.schema_id = manifest.schema_id;
    header.descriptor = descriptor;
    header.schema = manifest.schema;
    header.record_size = manifest.record_size;
    header.payload_size = checked_product<CorruptionError>(
        descriptor.row_count, manifest.record_size,
        "segment data payload size");
    return header;
}

[[nodiscard]] PartitionedSegmentIdsHeaderV1 expected_ids_header(
    const PartitionedManifestV2& manifest,
    const PartitionedSegmentDescriptor& descriptor) {
    PartitionedSegmentIdsHeaderV1 header;
    header.collection_id = manifest.collection_id;
    header.schema_id = manifest.schema_id;
    header.descriptor = descriptor;
    header.payload_size = checked_product<CorruptionError>(
        descriptor.row_count, kPartitionedSegmentIdEntrySize,
        "segment IDs payload size");
    return header;
}

}  // namespace

std::filesystem::path partitioned_segments_directory(
    const std::filesystem::path& store_path) {
    return store_path / "segments";
}

std::filesystem::path partitioned_segment_data_path(
    const std::filesystem::path& store_path,
    const Uuid128 segment_id) {
    return partitioned_segments_directory(store_path)
           / ("segment-" + uuid_token(segment_id) + ".dat");
}

std::filesystem::path partitioned_segment_ids_path(
    const std::filesystem::path& store_path,
    const Uuid128 segment_id) {
    return partitioned_segments_directory(store_path)
           / ("segment-" + uuid_token(segment_id) + ".ids");
}

std::uint64_t partitioned_segment_row_capacity(
    const PartitionedManifestV2& manifest) {
    validate_manifest_for_build(manifest);
    const std::uint64_t target_rows =
        manifest.target_segment_size / manifest.record_size;
    require<InvalidArgumentError>(
        target_rows != 0,
        "manifest segment target cannot fit one record");
    return std::min(
        target_rows, RowRef::kMaximumRowsPerSegment);
}

class PartitionedSegmentSetWriter::Impl {
private:
    struct PendingSegment {
        PartitionedSegmentDescriptor descriptor;
        std::filesystem::path data_staging;
        std::filesystem::path ids_staging;
        std::filesystem::path data_final;
        std::filesystem::path ids_final;
        bool data_staging_owned{true};
        bool ids_staging_owned{true};
        bool data_published{false};
        bool ids_published{false};
    };

public:
    Impl(
        std::filesystem::path store_path,
        PartitionedSegmentSetBuildSpec spec)
        : store_path_(std::move(store_path)),
          segments_path_(partitioned_segments_directory(store_path_)),
          spec_(std::move(spec)),
          rows_per_segment_(partitioned_segment_row_capacity(spec_.manifest)) {
        validate_supported_kind(spec_.kind, false);
        static_cast<void>(SegmentId::create(spec_.first_row_ref_segment_id));
        require<InvalidArgumentError>(
            spec_.creation_epoch != 0,
            "non-initial segment creation epoch must be positive");
        validate_segments_directory(segments_path_, false);
    }

    ~Impl() {
        if (owned_by_current_process() && !finished_) {
            cleanup_outputs();
        } else {
            data_fd_.reset();
            ids_fd_.reset();
        }
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    [[nodiscard]] RowRef append(
        const std::uint64_t id,
        const std::uint64_t sequence,
        const std::span<const std::byte> row) {
        ensure_writable();
        try {
            require<InvalidArgumentError>(
                row.size() == spec_.manifest.record_size,
                "segment append row size does not match manifest record size");
            validate_sequence(spec_.kind, sequence);
            if (!data_fd_ || active_rows_ == rows_per_segment_) {
                close_active_segment();
                open_segment();
            }
            require<StorageError>(
                active_rows_ < rows_per_segment_,
                "segment writer failed to rotate a full segment");

            const RowRef row_ref = RowRef::create(
                active_row_ref_segment_id_, active_rows_);
            std::array<std::byte, kPartitionedSegmentIdEntrySize> sidecar{};
            put_u64(sidecar, 0, id);
            put_u64(sidecar, 8, sequence);
            write_all(
                data_fd_.get(), row.data(), row.size(), data_staging_path_);
            write_all(
                ids_fd_.get(), sidecar.data(), sidecar.size(),
                ids_staging_path_);
            data_crc_.update(row);
            ids_crc_.update(sidecar);
            if (active_rows_ == 0) {
                active_minimum_sequence_ = sequence;
                active_maximum_sequence_ = sequence;
            } else {
                active_minimum_sequence_ = std::min(
                    active_minimum_sequence_, sequence);
                active_maximum_sequence_ = std::max(
                    active_maximum_sequence_, sequence);
            }
            ++active_rows_;
            if (total_rows_ == std::numeric_limits<std::uint64_t>::max()) {
                throw InvalidArgumentError(
                    "segment-set total row count overflows uint64");
            }
            ++total_rows_;
            return row_ref;
        } catch (...) {
            poison_and_cleanup();
            throw;
        }
    }

    [[nodiscard]] const std::vector<PartitionedSegmentDescriptor>& finish() {
        ensure_owner();
        if (finished_) {
            return descriptors_;
        }
        if (poisoned_) {
            throw StorageError(
                "segment-set writer is unusable after an earlier failure");
        }
        try {
            close_active_segment();
            descriptors_.clear();
            descriptors_.reserve(pending_.size());
            for (const PendingSegment& pending : pending_) {
                descriptors_.push_back(pending.descriptor);
            }
            for (PendingSegment& pending : pending_) {
                link_exclusive(pending.data_staging, pending.data_final);
                pending.data_published = true;
                link_exclusive(pending.ids_staging, pending.ids_final);
                pending.ids_published = true;
                remove_exact(pending.data_staging);
                pending.data_staging_owned = false;
                remove_exact(pending.ids_staging);
                pending.ids_staging_owned = false;
            }
            sync_directory(segments_path_);
            finished_ = true;
            return descriptors_;
        } catch (...) {
            poison_and_cleanup();
            throw;
        }
    }

    [[nodiscard]] const std::vector<PartitionedSegmentDescriptor>&
    descriptors() const {
        ensure_owner();
        if (!finished_) {
            throw InvalidArgumentError(
                "segment descriptors are not visible before finish");
        }
        return descriptors_;
    }

    [[nodiscard]] std::uint64_t row_count() const {
        ensure_owner();
        return total_rows_;
    }

    [[nodiscard]] bool finished() const {
        ensure_owner();
        return finished_;
    }

    [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        data_fd_.reset();
        ids_fd_.reset();
        abandoned_ = true;
    }

private:
    void ensure_owner() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "segment-set writer was inherited across fork; create a new "
                "writer in the child process");
        }
        if (abandoned_) {
            throw StorageError("segment-set writer was abandoned");
        }
    }

    void ensure_writable() const {
        ensure_owner();
        if (finished_) {
            throw InvalidArgumentError(
                "cannot append to a finished segment-set writer");
        }
        if (poisoned_) {
            throw StorageError(
                "segment-set writer is unusable after an earlier failure");
        }
    }

    void open_segment() {
        require<StorageError>(
            !data_fd_ && !ids_fd_,
            "segment writer opened a second active segment");
        const std::uint64_t numeric_id = checked_sum<InvalidArgumentError>(
            spec_.first_row_ref_segment_id,
            static_cast<std::uint64_t>(pending_.size()),
            "segment RowRef ID allocation");
        active_row_ref_segment_id_ = SegmentId::create(numeric_id).value();
        active_segment_uuid_ = spec_.segment_id_factory
            ? spec_.segment_id_factory()
            : make_uuid();
        require<InvalidArgumentError>(
            !active_segment_uuid_.empty(),
            "segment UUID factory returned an empty UUID");
        data_final_path_ = partitioned_segment_data_path(
            store_path_, active_segment_uuid_);
        ids_final_path_ = partitioned_segment_ids_path(
            store_path_, active_segment_uuid_);
        data_staging_path_ = staging_path(data_final_path_);
        ids_staging_path_ = staging_path(ids_final_path_);

        try {
            data_fd_ = create_exclusive(data_staging_path_);
            data_staging_owned_ = true;
            ids_fd_ = create_exclusive(ids_staging_path_);
            ids_staging_owned_ = true;
            const std::array<std::byte, kPartitionedHeaderSize> placeholder{};
            write_all(
                data_fd_.get(), placeholder.data(), placeholder.size(),
                data_staging_path_);
            write_all(
                ids_fd_.get(), placeholder.data(), placeholder.size(),
                ids_staging_path_);
        } catch (...) {
            data_fd_.reset();
            ids_fd_.reset();
            if (data_staging_owned_) {
                remove_nothrow(data_staging_path_);
                data_staging_owned_ = false;
            }
            if (ids_staging_owned_) {
                remove_nothrow(ids_staging_path_);
                ids_staging_owned_ = false;
            }
            throw;
        }
        active_rows_ = 0;
        active_minimum_sequence_ = 0;
        active_maximum_sequence_ = 0;
        data_crc_ = Crc32c{};
        ids_crc_ = Crc32c{};
    }

    void close_active_segment() {
        if (!data_fd_) {
            return;
        }
        require<StorageError>(
            ids_fd_ && active_rows_ != 0,
            "segment writer has incomplete active resources");

        PartitionedSegmentDescriptor descriptor;
        descriptor.segment_id = active_segment_uuid_;
        descriptor.row_ref_segment_id = active_row_ref_segment_id_;
        descriptor.kind = spec_.kind;
        descriptor.row_count = active_rows_;
        descriptor.minimum_sequence = active_minimum_sequence_;
        descriptor.maximum_sequence = active_maximum_sequence_;
        descriptor.data_file_size = checked_sum<InvalidArgumentError>(
            kPartitionedHeaderSize,
            checked_product<InvalidArgumentError>(
                active_rows_, spec_.manifest.record_size,
                "segment data payload"),
            "segment data file size");
        descriptor.ids_file_size = checked_sum<InvalidArgumentError>(
            kPartitionedHeaderSize,
            checked_product<InvalidArgumentError>(
                active_rows_, kPartitionedSegmentIdEntrySize,
                "segment IDs payload"),
            "segment IDs file size");
        descriptor.data_payload_crc32c = data_crc_.value();
        descriptor.ids_payload_crc32c = ids_crc_.value();
        descriptor.creation_epoch = spec_.creation_epoch;

        PartitionedSegmentDataHeaderV1 data_header;
        data_header.collection_id = spec_.manifest.collection_id;
        data_header.schema_id = spec_.manifest.schema_id;
        data_header.descriptor = descriptor;
        data_header.schema = spec_.manifest.schema;
        data_header.record_size = spec_.manifest.record_size;
        data_header.payload_size = descriptor.data_file_size
                                   - kPartitionedHeaderSize;
        PartitionedSegmentIdsHeaderV1 ids_header;
        ids_header.collection_id = spec_.manifest.collection_id;
        ids_header.schema_id = spec_.manifest.schema_id;
        ids_header.descriptor = descriptor;
        ids_header.payload_size = descriptor.ids_file_size
                                  - kPartitionedHeaderSize;
        const auto encoded_data =
            encode_partitioned_segment_data_header_v1(data_header);
        const auto encoded_ids =
            encode_partitioned_segment_ids_header_v1(ids_header);
        pwrite_all(
            data_fd_.get(), encoded_data.data(), encoded_data.size(), 0,
            data_staging_path_);
        pwrite_all(
            ids_fd_.get(), encoded_ids.data(), encoded_ids.size(), 0,
            ids_staging_path_);
        require<StorageError>(
            file_size(data_fd_.get(), data_staging_path_)
                    == descriptor.data_file_size
                && file_size(ids_fd_.get(), ids_staging_path_)
                       == descriptor.ids_file_size,
            "segment writer produced unexpected file geometry");
        sync_file(data_fd_.get(), data_staging_path_);
        sync_file(ids_fd_.get(), ids_staging_path_);
        data_fd_.reset();
        ids_fd_.reset();

        pending_.push_back(PendingSegment{
            descriptor,
            data_staging_path_,
            ids_staging_path_,
            data_final_path_,
            ids_final_path_});
        data_staging_owned_ = false;
        ids_staging_owned_ = false;
        clear_active_paths();
    }

    void clear_active_paths() noexcept {
        data_staging_path_.clear();
        ids_staging_path_.clear();
        data_final_path_.clear();
        ids_final_path_.clear();
        active_rows_ = 0;
        active_row_ref_segment_id_ = 0;
        active_segment_uuid_ = {};
    }

    void poison_and_cleanup() noexcept {
        poisoned_ = true;
        cleanup_outputs();
    }

    void cleanup_outputs() noexcept {
        data_fd_.reset();
        ids_fd_.reset();
        if (data_staging_owned_) {
            remove_nothrow(data_staging_path_);
            data_staging_owned_ = false;
        }
        if (ids_staging_owned_) {
            remove_nothrow(ids_staging_path_);
            ids_staging_owned_ = false;
        }
        clear_active_paths();
        for (PendingSegment& pending : pending_) {
            if (pending.data_staging_owned) {
                remove_nothrow(pending.data_staging);
                pending.data_staging_owned = false;
            }
            if (pending.ids_staging_owned) {
                remove_nothrow(pending.ids_staging);
                pending.ids_staging_owned = false;
            }
            if (pending.data_published) {
                remove_nothrow(pending.data_final);
            }
            if (pending.ids_published) {
                remove_nothrow(pending.ids_final);
            }
        }
        try {
            sync_directory(segments_path_);
        } catch (...) {
        }
    }

    std::filesystem::path store_path_;
    std::filesystem::path segments_path_;
    PartitionedSegmentSetBuildSpec spec_;
    const std::uint64_t rows_per_segment_;
    std::vector<PendingSegment> pending_;
    std::vector<PartitionedSegmentDescriptor> descriptors_;
    FileDescriptor data_fd_;
    FileDescriptor ids_fd_;
    Uuid128 active_segment_uuid_;
    std::uint32_t active_row_ref_segment_id_{0};
    std::uint64_t active_rows_{0};
    std::uint64_t active_minimum_sequence_{0};
    std::uint64_t active_maximum_sequence_{0};
    std::uint64_t total_rows_{0};
    Crc32c data_crc_;
    Crc32c ids_crc_;
    std::filesystem::path data_staging_path_;
    std::filesystem::path ids_staging_path_;
    std::filesystem::path data_final_path_;
    std::filesystem::path ids_final_path_;
    bool data_staging_owned_{false};
    bool ids_staging_owned_{false};
    bool finished_{false};
    bool poisoned_{false};
    bool abandoned_{false};
    const pid_t owner_pid_{::getpid()};
};

PartitionedSegmentSetWriter::PartitionedSegmentSetWriter(
    std::filesystem::path store_path,
    PartitionedSegmentSetBuildSpec spec)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), std::move(spec))) {}

PartitionedSegmentSetWriter::~PartitionedSegmentSetWriter() = default;

RowRef PartitionedSegmentSetWriter::append(
    const std::uint64_t id,
    const std::uint64_t sequence,
    const std::span<const std::byte> row) {
    return impl_->append(id, sequence, row);
}

const std::vector<PartitionedSegmentDescriptor>&
PartitionedSegmentSetWriter::finish() {
    return impl_->finish();
}

const std::vector<PartitionedSegmentDescriptor>&
PartitionedSegmentSetWriter::descriptors() const {
    return impl_->descriptors();
}

std::uint64_t PartitionedSegmentSetWriter::row_count() const {
    return impl_->row_count();
}

bool PartitionedSegmentSetWriter::finished() const {
    return impl_->finished();
}

bool PartitionedSegmentSetWriter::poisoned() const noexcept {
    return impl_->poisoned();
}

bool PartitionedSegmentSetWriter::owned_by_current_process() const noexcept {
    return impl_->owned_by_current_process();
}

void PartitionedSegmentSetWriter::abandon_after_fork() noexcept {
    impl_->abandon_after_fork();
}

class PartitionedSegmentReader::Impl {
public:
    Impl(
        std::filesystem::path store_path,
        PartitionedManifestV2 manifest,
        PartitionedSegmentDescriptor expected_descriptor,
        const bool verify_payload_crc)
        : store_path_(std::move(store_path)),
          manifest_(std::move(manifest)),
          descriptor_(std::move(expected_descriptor)) {
        validate_manifest_for_open(manifest_);
        validate_supported_kind(descriptor_.kind, true);
        validate_segments_directory(
            partitioned_segments_directory(store_path_), true);

        PartitionedSegmentDataHeaderV1 expected_data;
        PartitionedSegmentIdsHeaderV1 expected_ids;
        try {
            expected_data = expected_data_header(manifest_, descriptor_);
            expected_ids = expected_ids_header(manifest_, descriptor_);
            static_cast<void>(
                encode_partitioned_segment_data_header_v1(expected_data));
            static_cast<void>(
                encode_partitioned_segment_ids_header_v1(expected_ids));
        } catch (const InvalidArgumentError& error) {
            throw CorruptionError(error.what());
        }

        data_path_ = partitioned_segment_data_path(
            store_path_, descriptor_.segment_id);
        ids_path_ = partitioned_segment_ids_path(
            store_path_, descriptor_.segment_id);
        data_fd_ = open_regular_nofollow(data_path_);
        ids_fd_ = open_regular_nofollow(ids_path_);
        require<CorruptionError>(
            file_size(data_fd_.get(), data_path_)
                    == descriptor_.data_file_size
                && file_size(ids_fd_.get(), ids_path_)
                       == descriptor_.ids_file_size,
            "segment file size does not match catalog geometry");

        const PartitionedSegmentDataHeaderV1 actual_data =
            decode_partitioned_segment_data_header_v1(
                read_header(data_fd_.get(), data_path_));
        const PartitionedSegmentIdsHeaderV1 actual_ids =
            decode_partitioned_segment_ids_header_v1(
                read_header(ids_fd_.get(), ids_path_));
        require<CorruptionError>(
            actual_data == expected_data && actual_ids == expected_ids,
            "segment headers do not match catalog or manifest identity");

        ids_mapping_.map(
            ids_fd_.get(), descriptor_.ids_file_size, ids_path_);

        if (verify_payload_crc) {
            require<CorruptionError>(
                checksum_payload(
                    data_fd_.get(), data_path_, actual_data.payload_offset,
                    actual_data.payload_size)
                        == descriptor_.data_payload_crc32c
                    && crc32c(ids_mapping_.bytes().subspan(
                           static_cast<std::size_t>(actual_ids.payload_offset),
                           static_cast<std::size_t>(actual_ids.payload_size)))
                           == descriptor_.ids_payload_crc32c,
                "segment payload checksum mismatch");
        }
        // The immutable MAP_PRIVATE sidecar remains valid after close. Keep
        // only the data FD that backs future PhysicalReadRequest values.
        ids_fd_.reset();
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    [[nodiscard]] const PartitionedSegmentDescriptor& descriptor() const {
        ensure_owner();
        return descriptor_;
    }

    [[nodiscard]] std::uint64_t row_count() const {
        ensure_owner();
        return descriptor_.row_count;
    }

    [[nodiscard]] PartitionedSegmentIdEntry sidecar_at(
        const std::uint64_t row_index) const {
        ensure_owner();
        require<InvalidArgumentError>(
            row_index < descriptor_.row_count,
            "segment sidecar row index is out of range");
        const std::uint64_t offset = checked_sum<InvalidArgumentError>(
            kPartitionedHeaderSize,
            checked_product<InvalidArgumentError>(
                row_index, kPartitionedSegmentIdEntrySize,
                "segment sidecar row offset"),
            "segment sidecar file offset");
        const std::span<const std::byte> bytes =
            ids_mapping_.bytes().subspan(
                static_cast<std::size_t>(offset),
                static_cast<std::size_t>(kPartitionedSegmentIdEntrySize));
        const PartitionedSegmentIdEntry result{
            get_u64(bytes, 0), get_u64(bytes, 8)};
        require<CorruptionError>(
            result.sequence >= descriptor_.minimum_sequence
                && result.sequence <= descriptor_.maximum_sequence
                && result.sequence <= kMaximumSequence,
            "segment sidecar sequence is outside descriptor bounds");
        if (descriptor_.kind == PartitionedSegmentKind::Materialized) {
            require<CorruptionError>(
                result.sequence != 0,
                "materialized segment sidecar contains sequence zero");
        }
        return result;
    }

    [[nodiscard]] std::span<const std::byte> sidecar_payload() const {
        ensure_owner();
        return ids_mapping_.bytes().subspan(
            static_cast<std::size_t>(kPartitionedHeaderSize),
            static_cast<std::size_t>(
                descriptor_.ids_file_size - kPartitionedHeaderSize));
    }

    [[nodiscard]] PhysicalReadRequest resolve(
        const RowRef row_ref,
        const std::uint64_t output_index) const {
        ensure_owner();
        require<InvalidArgumentError>(
            row_ref.segment_id().value()
                == descriptor_.row_ref_segment_id,
            "RowRef belongs to a different immutable segment");
        require<InvalidArgumentError>(
            row_ref.row_index() < descriptor_.row_count,
            "RowRef row index exceeds immutable segment geometry");
        const std::uint64_t offset = checked_sum<InvalidArgumentError>(
            kPartitionedHeaderSize,
            checked_product<InvalidArgumentError>(
                row_ref.row_index(), manifest_.record_size,
                "segment data row offset"),
            "segment data file offset");
        return PhysicalReadRequest{data_fd_.get(), offset, output_index};
    }

    void abandon_after_fork() noexcept {
        if (owned_by_current_process()) {
            return;
        }
        data_fd_.reset();
        ids_fd_.reset();
        ids_mapping_.reset();
        abandoned_ = true;
    }

private:
    void ensure_owner() const {
        if (!owned_by_current_process()) {
            throw StorageError(
                "segment reader was inherited across fork; reopen the "
                "segment in the child process");
        }
        if (abandoned_) {
            throw StorageError("segment reader was abandoned");
        }
    }

    std::filesystem::path store_path_;
    PartitionedManifestV2 manifest_;
    PartitionedSegmentDescriptor descriptor_;
    FileDescriptor data_fd_;
    FileDescriptor ids_fd_;
    ReadOnlyMapping ids_mapping_;
    std::filesystem::path data_path_;
    std::filesystem::path ids_path_;
    bool abandoned_{false};
    const pid_t owner_pid_{::getpid()};
};

PartitionedSegmentReader::PartitionedSegmentReader(
    std::filesystem::path store_path,
    PartitionedManifestV2 manifest,
    PartitionedSegmentDescriptor expected_descriptor,
    const bool verify_payload_crc)
    : impl_(std::make_unique<Impl>(
          std::move(store_path), std::move(manifest),
          std::move(expected_descriptor), verify_payload_crc)) {}

PartitionedSegmentReader::~PartitionedSegmentReader() = default;
PartitionedSegmentReader::PartitionedSegmentReader(
    PartitionedSegmentReader&&) noexcept = default;
PartitionedSegmentReader& PartitionedSegmentReader::operator=(
    PartitionedSegmentReader&&) noexcept = default;

const PartitionedSegmentDescriptor&
PartitionedSegmentReader::descriptor() const {
    return impl_->descriptor();
}

std::uint64_t PartitionedSegmentReader::row_count() const {
    return impl_->row_count();
}

PartitionedSegmentIdEntry PartitionedSegmentReader::sidecar_at(
    const std::uint64_t row_index) const {
    return impl_->sidecar_at(row_index);
}

std::span<const std::byte> PartitionedSegmentReader::sidecar_payload() const {
    return impl_->sidecar_payload();
}

PhysicalReadRequest PartitionedSegmentReader::resolve(
    const RowRef row_ref,
    const std::uint64_t output_index) const {
    return impl_->resolve(row_ref, output_index);
}

bool PartitionedSegmentReader::owned_by_current_process() const noexcept {
    return impl_->owned_by_current_process();
}

void PartitionedSegmentReader::abandon_after_fork() noexcept {
    impl_->abandon_after_fork();
}

std::unique_ptr<PartitionedSegmentReader> open_partitioned_segment(
    std::filesystem::path store_path,
    PartitionedManifestV2 manifest,
    PartitionedSegmentDescriptor expected_descriptor,
    const bool verify_payload_crc) {
    return std::make_unique<PartitionedSegmentReader>(
        std::move(store_path), std::move(manifest),
        std::move(expected_descriptor), verify_payload_crc);
}

}  // namespace throwdb::detail
