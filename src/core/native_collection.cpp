#include "throwdb/native_collection.hpp"

#include "file_lock.hpp"
#include "internal.hpp"
#include "io_uring_engine.hpp"
#include "io_uring_pool.hpp"
#include "mutation_wal.hpp"
#include "partitioned_index_format.hpp"
#include "partitioned_index_storage.hpp"
#include "partitioned_resource_inventory.hpp"
#include "pending_mutation_fence.hpp"
#include "physical_reader.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <utility>
#include <unistd.h>

namespace throwdb {
namespace {

using PutMode = detail::PartitionedPutMode;

void require_current_store(const std::filesystem::path& path) {
    if (!detail::is_partitioned_index_store(path)) {
        throw InvalidArgumentError(
            "unsupported storage format: only format 7 collections are supported");
    }
}

[[nodiscard]] detail::DTypeCode dtype_code(const DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return detail::DTypeCode::Float16;
        case DType::Float32:
            return detail::DTypeCode::Float32;
    }
    throw InvalidArgumentError("unsupported embedding dtype");
}

[[nodiscard]] std::string uuid_filename_token(
    const detail::Uuid128 value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::array<char, 32> token{};
    std::size_t output = 0;
    const auto append_word = [&token, &output](const std::uint64_t word) {
        for (std::uint32_t shift = 60;; shift -= 4) {
            token[output++] = hex[(word >> shift) & 0xfU];
            if (shift == 0) {
                break;
            }
        }
    };
    append_word(value.high);
    append_word(value.low);
    return std::string(token.data(), token.size());
}

[[nodiscard]] NativeOrphanClass public_orphan_class(
    const detail::PartitionedInventoryClass classification) {
    switch (classification) {
        case detail::PartitionedInventoryClass::Reachable:
            return NativeOrphanClass::Reachable;
        case detail::PartitionedInventoryClass::SupersededCatalog:
            return NativeOrphanClass::SupersededCatalog;
        case detail::PartitionedInventoryClass::TempPublication:
            return NativeOrphanClass::TempPublication;
        case detail::PartitionedInventoryClass::UnreferencedWal:
            return NativeOrphanClass::UnreferencedWal;
        case detail::PartitionedInventoryClass::
            CompleteUnreferencedSegmentPair:
            return NativeOrphanClass::CompleteUnreferencedSegmentPair;
        case detail::PartitionedInventoryClass::
            PartialUnreferencedSegmentPair:
            return NativeOrphanClass::PartialUnreferencedSegmentPair;
        case detail::PartitionedInventoryClass::UnreferencedSnapshot:
            return NativeOrphanClass::UnreferencedSnapshot;
        case detail::PartitionedInventoryClass::UnreferencedRun:
            return NativeOrphanClass::UnreferencedRun;
        case detail::PartitionedInventoryClass::UnknownDoNotTouch:
            return NativeOrphanClass::UnknownDoNotTouch;
    }
    throw StorageError("format-7 inventory contains an unknown class");
}

[[nodiscard]] bool public_orphan_candidate(
    const detail::PartitionedInventoryClass classification) noexcept {
    return classification
               == detail::PartitionedInventoryClass::SupersededCatalog
        || classification
               == detail::PartitionedInventoryClass::TempPublication
        || classification
               == detail::PartitionedInventoryClass::UnreferencedWal
        || classification
               == detail::PartitionedInventoryClass::
                   CompleteUnreferencedSegmentPair
        || classification
               == detail::PartitionedInventoryClass::
                   PartialUnreferencedSegmentPair
        || classification
               == detail::PartitionedInventoryClass::UnreferencedSnapshot
        || classification
               == detail::PartitionedInventoryClass::UnreferencedRun;
}

[[nodiscard]] NativeOrphanInventory public_orphan_inventory(
    detail::PartitionedResourceInventory inventory) {
    NativeOrphanInventory result;
    result.current_generation = inventory.current_generation;
    result.current_catalog_id =
        uuid_filename_token(inventory.current_catalog_id);
    result.entries.reserve(inventory.entries.size());
    for (detail::PartitionedInventoryEntry& entry : inventory.entries) {
        NativeOrphanEntry converted;
        converted.path = std::move(entry.relative_path);
        converted.classification = public_orphan_class(entry.classification);
        converted.bytes = entry.bytes;
        converted.modified_unix_ns = entry.modified_unix_ns;
        converted.candidate = public_orphan_candidate(entry.classification);
        result.entries.push_back(std::move(converted));
    }
    return result;
}

[[nodiscard]] NativeOrphanDisposition public_orphan_disposition(
    const detail::PartitionedOrphanDisposition disposition) {
    switch (disposition) {
        case detail::PartitionedOrphanDisposition::NotCandidate:
            return NativeOrphanDisposition::NotCandidate;
        case detail::PartitionedOrphanDisposition::InvalidModificationTime:
            return NativeOrphanDisposition::InvalidModificationTime;
        case detail::PartitionedOrphanDisposition::MinimumAgeNotReached:
            return NativeOrphanDisposition::MinimumAgeNotReached;
        case detail::PartitionedOrphanDisposition::MaximumFilesReached:
            return NativeOrphanDisposition::MaximumFilesReached;
        case detail::PartitionedOrphanDisposition::MaximumBytesReached:
            return NativeOrphanDisposition::MaximumBytesReached;
        case detail::PartitionedOrphanDisposition::Removed:
            return NativeOrphanDisposition::Removed;
        case detail::PartitionedOrphanDisposition::NoLongerPresent:
            return NativeOrphanDisposition::NoLongerPresent;
        case detail::PartitionedOrphanDisposition::IdentityChanged:
            return NativeOrphanDisposition::IdentityChanged;
        case detail::PartitionedOrphanDisposition::UnlinkFailed:
            return NativeOrphanDisposition::UnlinkFailed;
    }
    throw StorageError("format-7 orphan collector contains an unknown disposition");
}

[[nodiscard]] std::string orphan_area_name(
    const detail::PartitionedInventoryArea area) {
    switch (area) {
        case detail::PartitionedInventoryArea::Root:
            return ".";
        case detail::PartitionedInventoryArea::Catalogs:
            return "catalogs";
        case detail::PartitionedInventoryArea::Segments:
            return "segments";
        case detail::PartitionedInventoryArea::Partitions:
            return "index/partitions";
        case detail::PartitionedInventoryArea::Runs:
            return "index/runs";
        case detail::PartitionedInventoryArea::Wal:
            return "wal";
    }
    throw StorageError("format-7 orphan collector contains an unknown area");
}

[[nodiscard]] NativeOrphanCollectionResult public_orphan_result(
    detail::PartitionedOrphanCollectionResult collected) {
    NativeOrphanCollectionResult result;
    result.current_generation = collected.current_generation;
    result.current_catalog_id =
        uuid_filename_token(collected.current_catalog_id);
    result.snapshot_unix_ns = collected.options.snapshot_unix_ns;
    result.min_age_ns = collected.options.min_age_ns;
    result.max_files = collected.options.max_files;
    result.max_bytes = collected.options.max_bytes;
    result.durability_complete = collected.durability_complete;
    result.synced_directories.reserve(collected.synced_directories.size());
    for (const detail::PartitionedInventoryArea area :
         collected.synced_directories) {
        result.synced_directories.push_back(orphan_area_name(area));
    }
    result.entries.reserve(collected.entries.size());
    for (detail::PartitionedOrphanCollectionEntry& entry :
         collected.entries) {
        NativeOrphanCollectionEntry converted;
        converted.file.path = std::move(entry.file.relative_path);
        converted.file.classification =
            public_orphan_class(entry.file.classification);
        converted.file.candidate =
            public_orphan_candidate(entry.file.classification);
        converted.file.bytes = entry.file.bytes;
        converted.file.modified_unix_ns = entry.file.modified_unix_ns;
        converted.disposition =
            public_orphan_disposition(entry.disposition);
        converted.eligible = entry.eligible;
        converted.selected = entry.selected;
        converted.removed = entry.removed;
        converted.durable = entry.durable;
        converted.error = std::move(entry.error);
        result.entries.push_back(std::move(converted));
    }
    return result;
}

[[nodiscard]] std::uint64_t current_unix_time_ns() noexcept {
    const auto raw = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return raw > 0 ? static_cast<std::uint64_t>(raw) : 0;
}

[[nodiscard]] NativeOrphanCollectionOptions normalized_orphan_options(
    NativeOrphanCollectionOptions options) {
    if (options.max_files == 0 || options.max_bytes == 0) {
        throw InvalidArgumentError(
            "orphan collection file and byte budgets must be positive");
    }
    if (options.snapshot_unix_ns == 0) {
        options.snapshot_unix_ns = current_unix_time_ns();
    }
    return options;
}

[[nodiscard]] std::filesystem::path writer_lock_path(
    const std::filesystem::path& path) {
    return path / "locks" / "writer.lock";
}

[[nodiscard]] std::filesystem::path commit_lock_path(
    const std::filesystem::path& path) {
    // This inode is broader than its historical name: readers hold SH while
    // selecting CURRENT and pinning every mutation resource; writers hold EX
    // for WAL commits and through CURRENT publication, unlink, and directory
    // sync. Never replace it while the collection exists.
    return path / "locks" / "commit.lock";
}

[[nodiscard]] std::filesystem::path view_lock_path(
    const std::filesystem::path& path) {
    return path / "locks" / "view.lock";
}

void ensure_lifecycle_not_fenced(const std::filesystem::path& root) {
    detail::ensure_pending_mutation_fence_absent(
        root, "destructive collection maintenance");
}

}  // namespace

class NativeCollectionBuilder::Impl : public detail::PartitionedIndexBuilder {
public:
    using detail::PartitionedIndexBuilder::PartitionedIndexBuilder;
    [[nodiscard]] std::uint32_t format_version() const {
        static_cast<void>(count());
        return detail::kPartitionedPublicFormatVersion;
    }
};

class NativeReaderTrial::Impl {
public:
    explicit Impl(const ReaderOptions& reader_options)
        : physical_reader_(reader_options) {}

    [[nodiscard]] const detail::PhysicalReader& physical_reader() const {
        return physical_reader_;
    }

    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return owner_pid_ == ::getpid();
    }

    void abandon_after_fork() noexcept {
        physical_reader_.abandon_after_fork();
    }

private:
    detail::PhysicalReader physical_reader_;
    const pid_t owner_pid_{::getpid()};
};

class NativeIdScan::Impl {
public:
    explicit Impl(std::unique_ptr<detail::PartitionedIdScan> scan)
        : scan_(std::move(scan)) {
        if (scan_ == nullptr) {
            throw InvalidArgumentError("native ID scan is null");
        }
    }

    [[nodiscard]] const NativeIdScanMetadata& metadata() const noexcept {
        return scan_->metadata();
    }
    [[nodiscard]] std::uint64_t read_into(
        const std::span<std::uint64_t> ids,
        const std::span<NativeIdLocation> locations) {
        return scan_->read_into(ids, locations);
    }
    [[nodiscard]] std::uint64_t read_into(
        const std::span<std::uint64_t> ids,
        const std::span<NativeIdLocation> locations,
        const NativeIdScan::StopPredicate& stop_requested) {
        return scan_->read_into(ids, locations, stop_requested);
    }
    [[nodiscard]] bool done() const noexcept { return scan_->done(); }

private:
    std::unique_ptr<detail::PartitionedIdScan> scan_;
};

class NativeReadView::Impl {
public:
    explicit Impl(std::unique_ptr<detail::PartitionedReadView> view)
        : view_(std::move(view)) {
        if (view_ == nullptr) {
            throw InvalidArgumentError("native read view is null");
        }
    }

    [[nodiscard]] const NativeReadViewMetadata& metadata() const noexcept {
        return view_->metadata();
    }
    [[nodiscard]] std::uint32_t output_dimension() const noexcept {
        return view_->output_dimension();
    }
    [[nodiscard]] DType dtype() const noexcept { return view_->dtype(); }
    void get_into(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        view_->get_into(ids, output);
    }
    [[nodiscard]] NativeReadStats get_into_profiled(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        return view_->get_into_profiled(ids, output);
    }

private:
    std::unique_ptr<detail::PartitionedReadView> view_;
};

class NativeRecordScan::Impl {
public:
    explicit Impl(std::unique_ptr<detail::PartitionedRecordScan> scan)
        : scan_(std::move(scan)) {
        if (scan_ == nullptr) {
            throw InvalidArgumentError("native record scan is null");
        }
    }

    [[nodiscard]] const NativeReadViewMetadata& metadata() const noexcept {
        return scan_->metadata();
    }
    [[nodiscard]] std::uint32_t output_dimension() const noexcept {
        return scan_->output_dimension();
    }
    [[nodiscard]] DType dtype() const noexcept { return scan_->dtype(); }
    [[nodiscard]] std::uint64_t read_into(
        const std::span<std::uint64_t> ids,
        const std::span<std::byte> embeddings,
        const std::span<NativeIdLocation> locations) {
        return scan_->read_into(ids, embeddings, locations);
    }
    [[nodiscard]] std::uint64_t read_into(
        const std::span<std::uint64_t> ids,
        const std::span<std::byte> embeddings,
        const std::span<NativeIdLocation> locations,
        const NativeRecordScan::StopPredicate& stop_requested) {
        return scan_->read_into(
            ids, embeddings, locations, stop_requested);
    }
    [[nodiscard]] bool done() const noexcept { return scan_->done(); }

private:
    std::unique_ptr<detail::PartitionedRecordScan> scan_;
};

class NativeCollection::Impl {
public:
    Impl(const std::filesystem::path& path, const ReaderOptions& options,
         const bool writable, const bool calibration,
         const NativeOpenOptions& open_options)
        : physical_reader_(std::make_shared<detail::PhysicalReader>(options)),
          reader_(path, options, writable, calibration, open_options) {}

    void get_into(const std::span<const std::uint64_t> ids,
                  const std::span<std::byte> output) const {
        reader_.get_into(*physical_reader_, ids, output);
    }
    [[nodiscard]] NativeReadStats get_into_profiled(
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        return reader_.get_into_profiled(*physical_reader_, ids, output);
    }
    void validate_reader_trial_options(const ReaderOptions& options) const {
        static_cast<void>(reader_.count());
        if (options.max_read_range_bytes < reader_.record_size()) {
            throw InvalidArgumentError(
                "max_read_range_bytes must fit at least one record");
        }
    }
    void get_into_with_reader(const detail::PhysicalReader& reader,
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        reader_.get_into(reader, ids, output);
    }
    [[nodiscard]] NativeReadStats get_into_with_reader_profiled(
        const detail::PhysicalReader& reader,
        const std::span<const std::uint64_t> ids,
        const std::span<std::byte> output) const {
        return reader_.get_into_profiled(reader, ids, output);
    }
    [[nodiscard]] std::vector<std::byte> get_bytes(
        const std::span<const std::uint64_t> ids) const {
        const auto bytes = detail::checked_multiply(
            ids.size(), reader_.record_size(), "get output");
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw InvalidArgumentError("get output does not fit address space");
        }
        std::vector<std::byte> output(static_cast<std::size_t>(bytes));
        get_into(ids, output);
        return output;
    }
    [[nodiscard]] NativeIdSample sample_ids(
        const std::uint64_t requested, const std::uint64_t seed,
        const std::uint64_t max_inspected_slots) const {
        return reader_.sample_ids(requested, seed, max_inspected_slots);
    }
    [[nodiscard]] std::unique_ptr<detail::PartitionedIdScan> scan_ids(
        const std::uint64_t max_snapshot_bytes) const {
        return reader_.scan_ids(max_snapshot_bytes);
    }
    [[nodiscard]] std::unique_ptr<detail::PartitionedReadView> open_read_view(
        const std::uint32_t prefix_dimensions,
        const std::uint64_t max_snapshot_bytes) const {
        return reader_.open_read_view(
            physical_reader_, prefix_dimensions, max_snapshot_bytes);
    }
    [[nodiscard]] std::unique_ptr<detail::PartitionedRecordScan> scan_records(
        const std::uint32_t prefix_dimensions,
        const std::uint64_t max_snapshot_bytes) const {
        return reader_.scan_records(
            physical_reader_, prefix_dimensions, max_snapshot_bytes);
    }
    [[nodiscard]] std::vector<std::uint8_t> contains(
        const std::span<const std::uint64_t> ids) const {
        return reader_.contains(ids);
    }
    [[nodiscard]] NativeCollectionIdentity identity() const {
        return reader_.identity();
    }
    [[nodiscard]] NativeMutationVerification verify_committed_mutation(
        const std::uint64_t sequence,
        const NativeMutationFingerprint& expected) const {
        return reader_.verify_committed_mutation(sequence, expected);
    }
    void put(const PutMode mode, const std::span<const std::uint64_t> ids,
             const std::span<const std::byte> embeddings) {
        reader_.put(mode, ids, embeddings);
    }
    [[nodiscard]] std::vector<std::uint8_t> erase(
        const std::span<const std::uint64_t> ids) {
        return reader_.erase(ids);
    }
    [[nodiscard]] NativeMutationLogStats mutation_log_stats() const {
        return reader_.mutation_log_stats();
    }
    [[nodiscard]] NativeMutationLogStats rotate_wal() {
        return reader_.rotate_wal();
    }
    [[nodiscard]] NativeMutationLogStats materialize_frozen_wals(
        const std::uint64_t max_wals, const std::uint64_t max_input_bytes,
        const std::uint64_t max_input_entries,
        const std::uint64_t max_memory_bytes) {
        return reader_.materialize_frozen_wals(
            *physical_reader_, max_wals, max_input_bytes, max_input_entries,
            max_memory_bytes);
    }
    [[nodiscard]] NativeMutationLogStats fold_partitions(
        const std::uint64_t max_runs, const std::uint64_t max_input_bytes,
        const std::uint64_t max_input_entries) {
        return reader_.fold_partitions(max_runs, max_input_bytes, max_input_entries);
    }
    [[nodiscard]] NativeSegmentCleaningStats clean_segments(
        const NativeSegmentCleaningOptions options) {
        return reader_.clean_segments(*physical_reader_, options);
    }
    [[nodiscard]] NativeOrphanInventory collect_orphans() {
        return public_orphan_inventory(reader_.collect_orphans());
    }
    [[nodiscard]] NativeOrphanCollectionResult reclaim_orphans(
        NativeOrphanCollectionOptions options) {
        options = normalized_orphan_options(options);
        return public_orphan_result(reader_.reclaim_orphans({
            options.snapshot_unix_ns, options.min_age_ns,
            options.max_files, options.max_bytes}));
    }
    [[nodiscard]] std::uint64_t count() const { return reader_.count(); }
    [[nodiscard]] std::uint32_t dimension() const { return reader_.dimension(); }
    [[nodiscard]] DType dtype() const { return reader_.dtype(); }
    [[nodiscard]] std::uint32_t format_version() const {
        static_cast<void>(reader_.dimension());
        return detail::kPartitionedPublicFormatVersion;
    }
    [[nodiscard]] std::uint64_t record_size() const { return reader_.record_size(); }
    [[nodiscard]] bool writable() const { return reader_.writable(); }
    [[nodiscard]] NativeOpenStats open_stats() const { return reader_.open_stats(); }
    [[nodiscard]] bool owned_by_current_process() const noexcept {
        return reader_.owned_by_current_process();
    }
    void abandon_after_fork() noexcept {
        reader_.abandon_after_fork();
        physical_reader_->abandon_after_fork();
    }

private:
    std::shared_ptr<detail::PhysicalReader> physical_reader_;
    detail::PartitionedIndexReader reader_;
};

std::unique_ptr<NativeCollectionBuilder> NativeCollectionBuilder::create(
    const std::filesystem::path& path,
    const NativeBuilderOptions options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection writing requires a little-endian host");
    }
    if (options.dimension == 0) {
        throw InvalidArgumentError("dimension must be greater than zero");
    }
    if (options.format_version != detail::kPartitionedPublicFormatVersion) {
        throw InvalidArgumentError("format_version must be 7");
    }
    if (options.partition_bits > detail::kPartitionedMaximumPartitionBits) {
        throw InvalidArgumentError(
            "format-7 partition_bits must be between 0 and 16");
    }
    const detail::DTypeCode code = dtype_code(options.dtype);
    const std::uint64_t record_size = detail::checked_multiply(
        options.dimension, detail::dtype_item_size(code),
        "embedding record size");
    if (options.segment_size_bytes < record_size) {
        throw InvalidArgumentError(
            "segment_size_bytes must fit at least one record");
    }
    return std::unique_ptr<NativeCollectionBuilder>(
        new NativeCollectionBuilder(std::make_unique<Impl>(path, options)));
}

NativeCollectionBuilder::NativeCollectionBuilder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeCollectionBuilder::~NativeCollectionBuilder() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        // After fork, inherited mutex state is not safe to tear down
        // explicitly. Close the child's descriptor references without an
        // explicit unlock, then leave only process-local heap state for exit.
        impl_->abandon_after_fork();
        static_cast<void>(impl_.release());
    }
}

void NativeCollectionBuilder::append(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->append(ids, embeddings);
}
void NativeCollectionBuilder::finish() { impl_->finish(); }
std::uint64_t NativeCollectionBuilder::count() const {
    return impl_->count();
}
std::uint32_t NativeCollectionBuilder::dimension() const {
    return impl_->dimension();
}
DType NativeCollectionBuilder::dtype() const { return impl_->dtype(); }
std::uint32_t NativeCollectionBuilder::format_version() const {
    return impl_->format_version();
}
std::uint64_t NativeCollectionBuilder::record_size() const {
    return impl_->record_size();
}
bool NativeCollectionBuilder::finished() const {
    return impl_->finished();
}

NativeMutationFingerprint NativeCollection::fingerprint_mutation(
    const NativeMutationOperation operation,
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings,
    const std::uint64_t record_size) {
    if (record_size == 0
        || record_size > std::numeric_limits<std::size_t>::max()) {
        throw InvalidArgumentError("mutation fingerprint record size is invalid");
    }
    if (ids.size() > std::numeric_limits<std::size_t>::max()
                         / static_cast<std::size_t>(record_size)) {
        throw InvalidArgumentError("mutation fingerprint payload size overflows");
    }
    const auto expected_bytes = ids.size()
                                * static_cast<std::size_t>(record_size);
    if ((operation == NativeMutationOperation::Put
         && embeddings.size() != expected_bytes)
        || (operation == NativeMutationOperation::Delete
            && !embeddings.empty())) {
        throw InvalidArgumentError(
            "mutation fingerprint payload does not match its operation");
    }
    if (operation != NativeMutationOperation::Put
        && operation != NativeMutationOperation::Delete) {
        throw InvalidArgumentError("mutation fingerprint operation is invalid");
    }
    std::vector<detail::WalMutation> mutations;
    mutations.reserve(ids.size());
    for (std::size_t index = 0; index != ids.size(); ++index) {
        mutations.push_back(detail::WalMutation{
            ids[index],
            operation == NativeMutationOperation::Put
                ? detail::WalOperationKind::Put
                : detail::WalOperationKind::Delete,
            operation == NativeMutationOperation::Put
                ? embeddings.subspan(
                      index * static_cast<std::size_t>(record_size),
                      static_cast<std::size_t>(record_size))
                : std::span<const std::byte>{}});
    }
    return NativeMutationFingerprint{
        detail::fingerprint_wal_mutations(record_size, mutations)};
}

NativeReaderCapabilities NativeCollection::reader_capabilities(
    const std::uint32_t io_uring_queue_depth,
    const std::uint32_t io_uring_workers) {
    NativeReaderCapabilities capabilities;
    capabilities.io_uring_probe_queue_depth = io_uring_queue_depth;
    capabilities.io_uring_probe_workers = io_uring_workers;
    if (io_uring_queue_depth == 0 || io_uring_queue_depth > 4096) {
        capabilities.io_uring_reason =
            "io_uring_queue_depth must be between 1 and 4096";
        return capabilities;
    }
    if (io_uring_workers == 0 || io_uring_workers > 64) {
        capabilities.io_uring_reason =
            "io_uring_workers must be between 1 and 64";
        return capabilities;
    }
    if (io_uring_workers > io_uring_queue_depth) {
        capabilities.io_uring_reason =
            "io_uring_queue_depth must be at least io_uring_workers";
        return capabilities;
    }
    try {
        detail::FileDescriptor zero_fd(
            ::open("/dev/zero", O_RDONLY | O_CLOEXEC));
        if (!zero_fd) {
            throw StorageError(detail::system_error_message(
                "cannot open capability probe source", "/dev/zero"));
        }
        std::vector<std::byte> buffers(io_uring_workers);
        std::vector<detail::IoUringReadOperation> operations;
        operations.reserve(io_uring_workers);
        for (std::uint32_t index = 0; index != io_uring_workers; ++index) {
            operations.push_back(detail::IoUringReadOperation{
                .fd = zero_fd.get(),
                .buffer = &buffers[index],
                .size = 1,
                .offset = 0,
                .user_index = index,
            });
        }
        if (io_uring_workers == 1) {
            detail::IoUringEngine engine(io_uring_queue_depth);
            engine.read_exact(operations);
            capabilities.io_uring_available = engine.usable();
        } else {
            detail::IoUringPool pool(
                io_uring_workers, io_uring_queue_depth);
            pool.read_exact(operations);
            capabilities.io_uring_available = true;
        }
        if (!capabilities.io_uring_available) {
            capabilities.io_uring_reason =
                "io_uring capability probe returned an unusable ring";
        }
    } catch (const std::exception& error) {
        capabilities.io_uring_reason = error.what();
    }
    return capabilities;
}

NativeCollectionMetadata NativeCollection::inspect_metadata(
    const std::filesystem::path& path) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection access requires a little-endian host");
    }

    require_current_store(path);
    return detail::inspect_partitioned_index_metadata(path);
}

NativeOrphanInventory NativeCollection::inspect_orphans(
    const std::filesystem::path& path) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection access requires a little-endian host");
    }

    require_current_store(path);
    detail::AdvisoryLockFile writer =
        detail::AdvisoryLockFile::open_existing(
            writer_lock_path(path), true);
    detail::ScopedAdvisoryLock writer_lease(
        writer, detail::FileLockMode::Exclusive, true);

    detail::AdvisoryLockFile commit =
        detail::AdvisoryLockFile::open_existing(
            commit_lock_path(path), true);
    detail::ScopedAdvisoryLock stable(
        commit, detail::FileLockMode::Exclusive);
    detail::AdvisoryLockFile view =
        detail::AdvisoryLockFile::open_existing(
            view_lock_path(path), true);
    detail::ScopedAdvisoryLock unlinked_view_barrier(
        view, detail::FileLockMode::Exclusive);
    return public_orphan_inventory(
        detail::inspect_partitioned_resources(path));

}

NativeOrphanCollectionResult NativeCollection::reclaim_orphans(
    const std::filesystem::path& path,
    NativeOrphanCollectionOptions options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection access requires a little-endian host");
    }
    require_current_store(path);
    detail::AdvisoryLockFile writer =
        detail::AdvisoryLockFile::open_existing(
            writer_lock_path(path), true);
    detail::ScopedAdvisoryLock writer_lease(
        writer, detail::FileLockMode::Exclusive, true);
    ensure_lifecycle_not_fenced(path);

    detail::AdvisoryLockFile commit =
        detail::AdvisoryLockFile::open_existing(
            commit_lock_path(path), true);
    detail::ScopedAdvisoryLock stable(
        commit, detail::FileLockMode::Exclusive);
    detail::AdvisoryLockFile view =
        detail::AdvisoryLockFile::open_existing(
            view_lock_path(path), true);
    detail::ScopedAdvisoryLock unlinked_view_barrier(
        view, detail::FileLockMode::Exclusive);
    options = normalized_orphan_options(options);
    return public_orphan_result(detail::collect_partitioned_orphans(
        path,
        detail::PartitionedOrphanCollectionOptions{
            options.snapshot_unix_ns,
            options.min_age_ns,
            options.max_files,
            options.max_bytes}));

}

std::unique_ptr<NativeCollection> NativeCollection::open(
    const std::filesystem::path& path,
    const ReaderOptions reader_options,
    const bool writable,
    const bool calibration,
    const NativeOpenOptions open_options) {
    if (path.empty()) {
        throw InvalidArgumentError("storage path must not be empty");
    }
    if (writable && calibration) {
        throw InvalidArgumentError(
            "calibration handles must be read-only");
    }
    if (open_options.max_wal_bytes == 0
        || open_options.max_wal_operations == 0
        || open_options.max_overlay_entries == 0
        || open_options.max_overlay_bytes == 0) {
        throw InvalidArgumentError(
            "native mutable-open safety limits must be positive");
    }
    if (std::endian::native != std::endian::little) {
        throw InvalidArgumentError(
            "native collection access requires a little-endian host");
    }
    require_current_store(path);
    return std::unique_ptr<NativeCollection>(new NativeCollection(
        std::make_unique<Impl>(
            path, reader_options, writable, calibration, open_options)));
}

NativeReaderTrial::NativeReaderTrial(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeReaderTrial::~NativeReaderTrial() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        impl_->abandon_after_fork();
        static_cast<void>(impl_.release());
    }
}

NativeIdScan::NativeIdScan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeIdScan::~NativeIdScan() = default;
const NativeIdScanMetadata& NativeIdScan::metadata() const noexcept {
    return impl_->metadata();
}
std::uint64_t NativeIdScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<NativeIdLocation> locations) {
    return impl_->read_into(ids, locations);
}
std::uint64_t NativeIdScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<NativeIdLocation> locations,
    const StopPredicate& stop_requested) {
    return impl_->read_into(ids, locations, stop_requested);
}
bool NativeIdScan::done() const noexcept { return impl_->done(); }

NativeReadView::NativeReadView(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeReadView::~NativeReadView() = default;
const NativeReadViewMetadata& NativeReadView::metadata() const noexcept {
    return impl_->metadata();
}
std::uint32_t NativeReadView::output_dimension() const noexcept {
    return impl_->output_dimension();
}
DType NativeReadView::dtype() const noexcept { return impl_->dtype(); }
void NativeReadView::get_into(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    impl_->get_into(ids, output);
}
NativeReadStats NativeReadView::get_into_profiled(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    return impl_->get_into_profiled(ids, output);
}

NativeRecordScan::NativeRecordScan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeRecordScan::~NativeRecordScan() = default;
const NativeReadViewMetadata& NativeRecordScan::metadata() const noexcept {
    return impl_->metadata();
}
std::uint32_t NativeRecordScan::output_dimension() const noexcept {
    return impl_->output_dimension();
}
DType NativeRecordScan::dtype() const noexcept { return impl_->dtype(); }
std::uint64_t NativeRecordScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<std::byte> embeddings,
    const std::span<NativeIdLocation> locations) {
    return impl_->read_into(ids, embeddings, locations);
}
std::uint64_t NativeRecordScan::read_into(
    const std::span<std::uint64_t> ids,
    const std::span<std::byte> embeddings,
    const std::span<NativeIdLocation> locations,
    const StopPredicate& stop_requested) {
    return impl_->read_into(ids, embeddings, locations, stop_requested);
}
bool NativeRecordScan::done() const noexcept { return impl_->done(); }

NativeCollection::NativeCollection(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NativeCollection::~NativeCollection() {
    if (impl_ != nullptr && !impl_->owned_by_current_process()) {
        // See the builder destructor above. In particular, do not explicitly
        // LOCK_UN an inherited flock: it belongs to the same open file
        // description as the still-running parent writer.
        impl_->abandon_after_fork();
        static_cast<void>(impl_.release());
    }
}

void NativeCollection::get_into(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    impl_->get_into(ids, output);
}
NativeReadStats NativeCollection::get_into_profiled(
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    return impl_->get_into_profiled(ids, output);
}
std::unique_ptr<NativeReaderTrial> NativeCollection::create_reader_trial(
    const ReaderOptions reader_options) const {
    impl_->validate_reader_trial_options(reader_options);
    return std::unique_ptr<NativeReaderTrial>(new NativeReaderTrial(
        std::make_unique<NativeReaderTrial::Impl>(reader_options)));
}
void NativeCollection::get_into_with_reader(
    const NativeReaderTrial& reader,
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    impl_->get_into_with_reader(
        reader.impl_->physical_reader(), ids, output);
}
NativeReadStats NativeCollection::get_into_with_reader_profiled(
    const NativeReaderTrial& reader,
    const std::span<const std::uint64_t> ids,
    const std::span<std::byte> output) const {
    return impl_->get_into_with_reader_profiled(
        reader.impl_->physical_reader(), ids, output);
}
NativeIdSample NativeCollection::sample_ids(
    const std::uint64_t requested,
    const std::uint64_t seed,
    const std::uint64_t max_inspected_slots) const {
    return impl_->sample_ids(requested, seed, max_inspected_slots);
}
std::unique_ptr<NativeIdScan> NativeCollection::scan_ids(
    const std::uint64_t max_snapshot_bytes) const {
    return std::unique_ptr<NativeIdScan>(new NativeIdScan(
        std::make_unique<NativeIdScan::Impl>(
            impl_->scan_ids(max_snapshot_bytes))));
}
std::unique_ptr<NativeReadView> NativeCollection::open_read_view(
    const std::uint32_t prefix_dimensions,
    const std::uint64_t max_snapshot_bytes) const {
    return std::unique_ptr<NativeReadView>(new NativeReadView(
        std::make_unique<NativeReadView::Impl>(
            impl_->open_read_view(
                prefix_dimensions, max_snapshot_bytes))));
}
std::unique_ptr<NativeRecordScan> NativeCollection::scan_records(
    const std::uint32_t prefix_dimensions,
    const std::uint64_t max_snapshot_bytes) const {
    return std::unique_ptr<NativeRecordScan>(new NativeRecordScan(
        std::make_unique<NativeRecordScan::Impl>(
            impl_->scan_records(
                prefix_dimensions, max_snapshot_bytes))));
}
std::vector<std::byte> NativeCollection::get_bytes(
    const std::span<const std::uint64_t> ids) const {
    return impl_->get_bytes(ids);
}
std::vector<std::uint8_t> NativeCollection::contains(
    const std::span<const std::uint64_t> ids) const {
    return impl_->contains(ids);
}
NativeCollectionIdentity NativeCollection::identity() const {
    return impl_->identity();
}
NativeMutationVerification NativeCollection::verify_committed_mutation(
    const std::uint64_t sequence,
    const NativeMutationFingerprint& expected) const {
    return impl_->verify_committed_mutation(sequence, expected);
}
void NativeCollection::insert(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Insert, ids, embeddings);
}
void NativeCollection::update(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Update, ids, embeddings);
}
void NativeCollection::upsert(
    const std::span<const std::uint64_t> ids,
    const std::span<const std::byte> embeddings) {
    impl_->put(PutMode::Upsert, ids, embeddings);
}
std::vector<std::uint8_t> NativeCollection::erase(
    const std::span<const std::uint64_t> ids) {
    return impl_->erase(ids);
}
NativeMutationLogStats NativeCollection::mutation_log_stats() const {
    return impl_->mutation_log_stats();
}
NativeMutationLogStats NativeCollection::rotate_wal() {
    return impl_->rotate_wal();
}
NativeMutationLogStats NativeCollection::materialize_frozen_wals(
    const std::uint64_t max_wals,
    const std::uint64_t max_input_bytes,
    const std::uint64_t max_input_entries,
    const std::uint64_t max_memory_bytes) {
    return impl_->materialize_frozen_wals(
        max_wals, max_input_bytes, max_input_entries, max_memory_bytes);
}
NativeMutationLogStats NativeCollection::fold_partitions() {
    return impl_->fold_partitions(4096, 8ULL << 30, 1'000'000);
}
NativeMutationLogStats NativeCollection::fold_partitions(
    const std::uint64_t max_runs,
    const std::uint64_t max_input_bytes,
    const std::uint64_t max_input_entries) {
    return impl_->fold_partitions(
        max_runs, max_input_bytes, max_input_entries);
}
NativeSegmentCleaningStats NativeCollection::clean_segments(
    const NativeSegmentCleaningOptions options) {
    return impl_->clean_segments(options);
}
NativeOrphanInventory NativeCollection::collect_orphans() {
    return impl_->collect_orphans();
}
NativeOrphanCollectionResult NativeCollection::reclaim_orphans(
    NativeOrphanCollectionOptions options) {
    return impl_->reclaim_orphans(options);
}
std::uint64_t NativeCollection::count() const { return impl_->count(); }
std::uint32_t NativeCollection::dimension() const {
    return impl_->dimension();
}
DType NativeCollection::dtype() const { return impl_->dtype(); }
std::uint32_t NativeCollection::format_version() const {
    return impl_->format_version();
}
std::uint64_t NativeCollection::record_size() const {
    return impl_->record_size();
}
bool NativeCollection::writable() const { return impl_->writable(); }
NativeOpenStats NativeCollection::open_stats() const {
    return impl_->open_stats();
}

}  // namespace throwdb
