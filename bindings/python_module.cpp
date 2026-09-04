#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "crc32c.hpp"
#include "throwdb/errors.hpp"
#include "throwdb/native_collection.hpp"
#include "throwdb/storage.hpp"

namespace py = pybind11;

namespace {

constexpr std::uint64_t kDefaultShardSizeBytes = 16ULL << 30;
constexpr std::int64_t kMaxIoWorkers = 1024;
constexpr std::int64_t kMaxIoQueueCapacity = 1 << 20;
constexpr std::int64_t kDefaultIoRangesPerTask = 8;
constexpr std::int64_t kMaxIoRangesPerTask = 1 << 20;
constexpr std::int64_t kDefaultIoUringWorkers = 1;
constexpr std::int64_t kMaxIoUringWorkers = 64;
constexpr std::int64_t kDefaultIoUringQueueDepth = 32;
constexpr std::int64_t kMaxIoUringQueueDepth = 4096;
constexpr std::uint64_t kDefaultMaxReadRangeBytes = 1ULL << 20;
constexpr std::uint64_t kDefaultMaxGapBytes = 0;
constexpr std::uint64_t kMaxIdSampleRows = 1ULL << 22;
constexpr std::uint64_t kMaxIdSampleInspectedSlots = 1ULL << 25;
constexpr std::uint64_t kMaximumDeltaCompactionMaxRuns = 64;
constexpr std::uint64_t kMaximumPartitionedFoldRuns = 4096;
constexpr std::uint64_t kDefaultDeltaCompactionMaxInputBytes = 8ULL << 30;
constexpr std::uint64_t kDefaultDeltaCompactionMaxInputEntries = 1'000'000;
constexpr std::uint64_t kDefaultOrphanMaxFiles = 1024;
constexpr std::uint64_t kDefaultOrphanMaxBytes = 8ULL << 30;
constexpr std::uint64_t kDefaultMaxWalBacklogBytes = 16ULL << 30;
constexpr std::uint64_t kDefaultMaxWalBacklogOperations = 4'000'000;
constexpr std::uint64_t kDefaultMaxOverlayEntries = 4'000'000;
constexpr std::uint64_t kDefaultMaxOverlayBytes = 256ULL << 20;

struct DTypeInfo {
    const char* name;
};

[[nodiscard]] DTypeInfo dtype_info(const throwdb::DType dtype) {
    switch (dtype) {
    case throwdb::DType::Float16:
        return DTypeInfo{"float16"};
    case throwdb::DType::Float32:
        return DTypeInfo{"float32"};
    }
    throw std::runtime_error("storage has an unsupported dtype");
}

[[nodiscard]] throwdb::DType checked_dtype(const std::string& name) {
    if (name == "float16") {
        return throwdb::DType::Float16;
    }
    if (name == "float32") {
        return throwdb::DType::Float32;
    }
    throw py::value_error(
        "dtype must be 'float16' or 'float32', got '" + name + "'");
}

[[nodiscard]] py::dtype numpy_dtype(const throwdb::DType dtype) {
    return py::dtype::from_args(py::str(dtype_info(dtype).name));
}

[[nodiscard]] std::uint64_t embedding_record_size(
    const throwdb::DType dtype, const std::uint32_t dimension) {
    const std::uint64_t item_size =
        dtype == throwdb::DType::Float16 ? 2 : 4;
    if (dimension > std::numeric_limits<std::uint64_t>::max() / item_size) {
        throw py::value_error("embedding record size is too large");
    }
    return static_cast<std::uint64_t>(dimension) * item_size;
}

[[nodiscard]] py::dict read_view_metadata_dict(
    const throwdb::NativeReadViewMetadata& metadata) {
    py::dict result;
    result["collection_id_low"] = metadata.collection_id_low;
    result["collection_id_high"] = metadata.collection_id_high;
    result["catalog_generation"] = metadata.catalog_generation;
    result["committed_sequence"] = metadata.committed_sequence;
    result["live_count"] = metadata.live_count;
    result["snapshot_bytes"] = metadata.snapshot_bytes;
    return result;
}

[[nodiscard]] py::dict mutation_log_stats_dict(
    const throwdb::NativeMutationLogStats& stats) {
    py::dict result;
    result["catalog_managed"] = stats.catalog_managed;
    result["delta_managed"] = stats.delta_managed;
    result["generation"] = stats.generation;
    result["committed_sequence"] = stats.committed_sequence;
    result["materialized_sequence"] = stats.materialized_sequence;
    result["frozen_wal_count"] = stats.frozen_wal_count;
    result["active_wal_bytes"] = stats.active_wal_bytes;
    result["active_wal_put_rows"] = stats.active_wal_put_rows;
    result["active_wal_transaction_count"] =
        stats.active_wal_transaction_count;
    result["total_wal_bytes"] = stats.total_wal_bytes;
    result["overlay_entry_count"] = stats.overlay_entry_count;
    result["estimated_overlay_bytes"] = stats.estimated_overlay_bytes;
    result["active_wal_operation_count"] =
        stats.active_wal_operation_count;
    result["frozen_wal_operation_count"] =
        stats.frozen_wal_operation_count;
    result["total_wal_operation_count"] =
        stats.total_wal_operation_count;
    result["frozen_wal_bytes"] = stats.frozen_wal_bytes;
    result["delta_run_count"] = stats.delta_run_count;
    result["delta_entry_count"] = stats.delta_entry_count;
    result["delta_put_rows"] = stats.delta_put_rows;
    result["total_delta_run_bytes"] = stats.total_delta_run_bytes;
    result["total_delta_segment_bytes"] =
        stats.total_delta_segment_bytes;
    result["total_delta_bytes"] = stats.total_delta_bytes;
    result["reclaimed_wal_count"] = stats.reclaimed_wal_count;
    result["reclaimed_wal_bytes"] = stats.reclaimed_wal_bytes;
    result["materialized_wal_count"] = stats.materialized_wal_count;
    result["materialized_wal_input_bytes"] =
        stats.materialized_wal_input_bytes;
    result["materialized_wal_input_entries"] =
        stats.materialized_wal_input_entries;
    result["materialization_estimated_peak_memory_bytes"] =
        stats.materialization_estimated_peak_memory_bytes;
    result["compacted_delta_run_count"] =
        stats.compacted_delta_run_count;
    result["compacted_delta_input_entries"] =
        stats.compacted_delta_input_entries;
    result["compacted_delta_input_bytes"] =
        stats.compacted_delta_input_bytes;
    result["reclaimed_delta_run_count"] =
        stats.reclaimed_delta_run_count;
    result["reclaimed_delta_bytes"] = stats.reclaimed_delta_bytes;
    result["compacted_partition_count"] =
        stats.compacted_partition_count;
    result["compacted_run_section_count"] =
        stats.compacted_run_section_count;
    result["compacted_snapshot_input_entries"] =
        stats.compacted_snapshot_input_entries;
    result["compacted_snapshot_input_bytes"] =
        stats.compacted_snapshot_input_bytes;
    result["compacted_output_entries"] =
        stats.compacted_output_entries;
    result["compacted_output_bytes"] =
        stats.compacted_output_bytes;
    result["rotated"] = stats.rotated;
    result["materialized"] = stats.materialized;
    result["materialization_budget_blocked"] =
        stats.materialization_budget_blocked;
    result["materialization_entry_budget_blocked"] =
        stats.materialization_entry_budget_blocked;
    result["materialization_memory_budget_blocked"] =
        stats.materialization_memory_budget_blocked;
    result["compacted"] = stats.compacted;
    result["compaction_budget_blocked"] =
        stats.compaction_budget_blocked;
    result["reclamation_complete"] = stats.reclamation_complete;
    return result;
}

[[nodiscard]] py::dict segment_cleaning_stats_dict(
    const throwdb::NativeSegmentCleaningStats& stats) {
    py::dict result;
    result["generation"] = stats.generation;
    result["scanned_segment_count"] = stats.scanned_segment_count;
    result["scanned_row_count"] = stats.scanned_row_count;
    result["scanned_sidecar_bytes"] = stats.scanned_sidecar_bytes;
    result["skipped_budget_segment_count"] =
        stats.skipped_budget_segment_count;
    result["next_start_after_segment_id"] =
        stats.next_start_after_segment_id;
    result["candidate_segment_id"] = stats.candidate_segment_id;
    result["source_rows"] = stats.source_rows;
    result["relocated_rows"] = stats.relocated_rows;
    result["garbage_rows"] = stats.garbage_rows;
    result["source_bytes"] = stats.source_bytes;
    result["output_segment_count"] = stats.output_segment_count;
    result["output_segment_bytes"] = stats.output_segment_bytes;
    result["output_run_bytes"] = stats.output_run_bytes;
    result["reclaimable_bytes"] = stats.reclaimable_bytes;
    result["dry_run"] = stats.dry_run;
    result["candidate_found"] = stats.candidate_found;
    result["cleaned"] = stats.cleaned;
    result["budget_blocked"] = stats.budget_blocked;
    result["cycle_complete"] = stats.cycle_complete;
    result["reclamation_complete"] = stats.reclamation_complete;
    return result;
}

[[nodiscard]] const char* orphan_class_name(
    const throwdb::NativeOrphanClass classification) {
    switch (classification) {
    case throwdb::NativeOrphanClass::Reachable:
        return "reachable";
    case throwdb::NativeOrphanClass::SupersededCatalog:
        return "superseded_catalog";
    case throwdb::NativeOrphanClass::TempPublication:
        return "temp_publication";
    case throwdb::NativeOrphanClass::UnreferencedWal:
        return "unreferenced_wal";
    case throwdb::NativeOrphanClass::CompleteUnreferencedDeltaPair:
        return "complete_unreferenced_delta_pair";
    case throwdb::NativeOrphanClass::PartialUnreferencedDeltaPair:
        return "partial_unreferenced_delta_pair";
    case throwdb::NativeOrphanClass::UnknownDoNotTouch:
        return "unknown_do_not_touch";
    case throwdb::NativeOrphanClass::CompleteUnreferencedSegmentPair:
        return "complete_unreferenced_segment_pair";
    case throwdb::NativeOrphanClass::PartialUnreferencedSegmentPair:
        return "partial_unreferenced_segment_pair";
    case throwdb::NativeOrphanClass::UnreferencedSnapshot:
        return "unreferenced_snapshot";
    case throwdb::NativeOrphanClass::UnreferencedRun:
        return "unreferenced_run";
    }
    throw std::runtime_error("orphan inventory contains an unknown class");
}

[[nodiscard]] py::dict orphan_inventory_dict(
    const throwdb::NativeOrphanInventory& inventory) {
    py::list entries;
    for (const throwdb::NativeOrphanEntry& entry : inventory.entries) {
        py::dict item;
        item["path"] = entry.path;
        item["classification"] = orphan_class_name(entry.classification);
        item["candidate"] = entry.candidate;
        item["bytes"] = entry.bytes;
        item["modified_unix_ns"] = entry.modified_unix_ns;
        entries.append(std::move(item));
    }
    py::dict result;
    result["current_generation"] = inventory.current_generation;
    result["current_catalog_id"] = inventory.current_catalog_id;
    result["entries"] = std::move(entries);
    return result;
}

[[nodiscard]] const char* orphan_disposition_name(
    const throwdb::NativeOrphanDisposition disposition) {
    switch (disposition) {
    case throwdb::NativeOrphanDisposition::NotCandidate:
        return "not_candidate";
    case throwdb::NativeOrphanDisposition::InvalidModificationTime:
        return "invalid_modification_time";
    case throwdb::NativeOrphanDisposition::MinimumAgeNotReached:
        return "minimum_age_not_reached";
    case throwdb::NativeOrphanDisposition::MaximumFilesReached:
        return "max_files_reached";
    case throwdb::NativeOrphanDisposition::MaximumBytesReached:
        return "max_bytes_reached";
    case throwdb::NativeOrphanDisposition::Removed:
        return "removed";
    case throwdb::NativeOrphanDisposition::NoLongerPresent:
        return "no_longer_present";
    case throwdb::NativeOrphanDisposition::IdentityChanged:
        return "identity_changed";
    case throwdb::NativeOrphanDisposition::UnlinkFailed:
        return "unlink_failed";
    }
    throw std::runtime_error("orphan result contains an unknown disposition");
}

[[nodiscard]] py::dict orphan_collection_result_dict(
    const throwdb::NativeOrphanCollectionResult& result) {
    py::list entries;
    for (const throwdb::NativeOrphanCollectionEntry& entry : result.entries) {
        py::dict item;
        item["path"] = entry.file.path;
        item["classification"] =
            orphan_class_name(entry.file.classification);
        item["candidate"] = entry.file.candidate;
        item["bytes"] = entry.file.bytes;
        item["modified_unix_ns"] = entry.file.modified_unix_ns;
        item["disposition"] =
            orphan_disposition_name(entry.disposition);
        item["eligible"] = entry.eligible;
        item["selected"] = entry.selected;
        item["removed"] = entry.removed;
        item["durable"] = entry.durable;
        item["error"] = entry.error.empty()
            ? py::none()
            : py::cast(entry.error);
        entries.append(std::move(item));
    }
    py::list synced_directories;
    for (const std::string& directory : result.synced_directories) {
        synced_directories.append(directory);
    }
    py::dict output;
    output["current_generation"] = result.current_generation;
    output["current_catalog_id"] = result.current_catalog_id;
    output["snapshot_unix_ns"] = result.snapshot_unix_ns;
    output["min_age_ns"] = result.min_age_ns;
    output["max_files"] = result.max_files;
    output["max_bytes"] = result.max_bytes;
    output["synced_directories"] = std::move(synced_directories);
    output["durability_complete"] = result.durability_complete;
    output["entries"] = std::move(entries);
    return output;
}

[[nodiscard]] py::dict native_read_stats_dict(
    const throwdb::NativeReadStats& stats) {
    py::dict result;
    result["total_seconds"] = stats.total_seconds;
    result["lookup_seconds"] = stats.lookup_seconds;
    result["physical_sort_seconds"] = stats.physical_sort_seconds;
    result["physical_plan_seconds"] = stats.physical_plan_seconds;
    result["physical_backend_seconds"] = stats.physical_backend_seconds;
    result["requested_rows"] = stats.requested_rows;
    result["base_rows"] = stats.base_rows;
    result["mutation_rows"] = stats.mutation_rows;
    result["unique_files"] = stats.unique_files;
    result["planned_ranges"] = stats.planned_ranges;
    result["requested_bytes"] = stats.requested_bytes;
    result["planned_read_bytes"] = stats.planned_read_bytes;
    result["completed_read_bytes"] = stats.completed_read_bytes;
    result["useful_read_bytes"] = stats.useful_read_bytes;
    result["gap_bytes"] = stats.gap_bytes;
    result["overread_bytes"] = stats.overread_bytes;
    result["pages_touched"] = stats.pages_touched;
    result["page_runs"] = stats.page_runs;
    result["chunks_touched"] = stats.chunks_touched;
    result["chunk_runs"] = stats.chunk_runs;
    result["input_discontinuities"] = stats.input_discontinuities;
    result["input_inversions"] = stats.input_inversions;
    result["maximum_range_bytes"] = stats.maximum_range_bytes;
    result["mean_range_bytes"] = stats.mean_range_bytes;
    result["backend_tasks"] = stats.backend_tasks;
    result["zero_delta_fast_path"] = stats.zero_delta_fast_path;
    return result;
}

[[nodiscard]] std::filesystem::path filesystem_path(const py::handle& value) {
    py::object path = py::module_::import("os").attr("fspath")(value);
    if (py::isinstance<py::str>(path) || py::isinstance<py::bytes>(path)) {
        return std::filesystem::path(py::cast<std::string>(path));
    }
    throw py::type_error("path must be str, bytes, or os.PathLike");
}

[[nodiscard]] bool dtype_equal(const py::dtype& left, const py::dtype& right) {
    return py::cast<bool>(left.attr("__eq__")(right));
}

[[nodiscard]] std::uint32_t checked_io_option(
    const std::int64_t value,
    const std::int64_t maximum,
    const char* name) {
    if (value < 0 || value > maximum) {
        throw py::value_error(
            std::string(name) + " must be between 0 and "
            + std::to_string(maximum));
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t checked_io_ranges_per_task(
    const std::int64_t value) {
    if (value < 1 || value > kMaxIoRangesPerTask) {
        throw py::value_error(
            "io_ranges_per_task must be between 1 and "
            + std::to_string(kMaxIoRangesPerTask));
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t checked_io_uring_queue_depth(
    const std::int64_t value) {
    if (value < 1 || value > kMaxIoUringQueueDepth) {
        throw py::value_error(
            "io_uring_queue_depth must be between 1 and "
            + std::to_string(kMaxIoUringQueueDepth));
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t checked_io_uring_workers(
    const std::int64_t value) {
    if (value < 1 || value > kMaxIoUringWorkers) {
        throw py::value_error(
            "io_uring_workers must be between 1 and "
            + std::to_string(kMaxIoUringWorkers));
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] bool checked_python_bool(
    const py::handle value, const char* name) {
    if (!py::isinstance<py::bool_>(value)) {
        throw py::type_error(std::string(name) + " must be a bool");
    }
    return py::cast<bool>(value);
}

[[nodiscard]] std::uint64_t checked_uint64_option(
    const py::handle value,
    const char* name,
    const bool allow_zero) {
    if (py::isinstance<py::bool_>(value)) {
        throw py::type_error(std::string(name) + " must be an integer, not bool");
    }

    PyObject* const indexed_raw = PyNumber_Index(value.ptr());
    if (indexed_raw == nullptr) {
        PyErr_Clear();
        throw py::type_error(std::string(name) + " must be an integer");
    }
    const py::object indexed = py::reinterpret_steal<py::object>(indexed_raw);
    const unsigned long long converted =
        PyLong_AsUnsignedLongLong(indexed.ptr());
    if (PyErr_Occurred() != nullptr
        || converted > std::numeric_limits<std::uint64_t>::max()) {
        PyErr_Clear();
        throw py::value_error(
            std::string(name) + " must be between 0 and 2**64-1");
    }
    const auto result = static_cast<std::uint64_t>(converted);
    if (!allow_zero && result == 0) {
        throw py::value_error(
            std::string(name) + " must be greater than zero");
    }
    return result;
}

[[nodiscard]] throwdb::IoBackend checked_io_backend(
    const std::string& value) {
    if (value == "thread_pool") {
        return throwdb::IoBackend::ThreadPool;
    }
    if (value == "io_uring") {
        return throwdb::IoBackend::IoUring;
    }
    throw py::value_error(
        "io_backend must be 'thread_pool' or 'io_uring', got '" + value + "'");
}

[[nodiscard]] throwdb::ReaderOptions checked_reader_options(
    const std::int64_t io_workers,
    const std::int64_t io_queue_capacity,
    const std::int64_t io_ranges_per_task,
    const std::string& io_backend,
    const std::int64_t io_uring_queue_depth,
    const std::int64_t io_uring_workers,
    const bool verify_native_index = false,
    const std::uint64_t max_read_range_bytes = kDefaultMaxReadRangeBytes,
    const std::uint64_t max_gap_bytes = kDefaultMaxGapBytes) {
    throwdb::ReaderOptions options;
    if (max_read_range_bytes == 0) {
        throw py::value_error(
            "max_read_range_bytes must be greater than zero");
    }
    options.max_read_range_bytes = max_read_range_bytes;
    options.max_gap_bytes = max_gap_bytes;
    options.io_workers =
        checked_io_option(io_workers, kMaxIoWorkers, "io_workers");
    options.io_queue_capacity = checked_io_option(
        io_queue_capacity,
        kMaxIoQueueCapacity,
        "io_queue_capacity");
    options.io_ranges_per_task =
        checked_io_ranges_per_task(io_ranges_per_task);
    options.io_backend = checked_io_backend(io_backend);
    options.io_uring_queue_depth =
        checked_io_uring_queue_depth(io_uring_queue_depth);
    options.io_uring_workers =
        checked_io_uring_workers(io_uring_workers);
    options.verify_native_index = verify_native_index;
    if (options.io_backend == throwdb::IoBackend::IoUring
        && (options.io_workers != 0 || options.io_queue_capacity != 0)) {
        throw py::value_error(
            "io_workers and io_queue_capacity must both be 0 when "
            "io_backend is 'io_uring'");
    }
    if (options.io_backend == throwdb::IoBackend::IoUring
        && options.io_uring_workers > options.io_uring_queue_depth) {
        throw py::value_error(
            "io_uring_queue_depth must be at least io_uring_workers");
    }
    if (options.io_workers == 0 && options.io_queue_capacity != 0) {
        throw py::value_error(
            "io_queue_capacity must be 0 when io_workers is 0");
    }
    return options;
}

template <typename Function>
decltype(auto) translate_id_errors(Function&& function) {
    try {
        return std::forward<Function>(function)();
    } catch (const throwdb::ExistingIdError& error) {
        throw py::value_error(error.what());
    } catch (const throwdb::DuplicateIdError& error) {
        throw py::value_error(error.what());
    } catch (const throwdb::MissingIdError& error) {
        throw py::key_error(error.what());
    }
}

void require_native_endian(const py::array& array, const char* name) {
    if (!py::cast<bool>(array.dtype().attr("isnative"))) {
        throw py::value_error(std::string(name) + " must use native byte order");
    }
}

void require_c_contiguous(const py::array& array, const char* name) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

void require_aligned(const py::array& array, const char* name) {
    if (!py::cast<bool>(array.attr("flags").attr("aligned"))) {
        throw py::value_error(std::string(name) + " must be memory-aligned");
    }
}

void require_dtype(const py::array& array,
                   const throwdb::DType expected,
                   const char* name) {
    require_native_endian(array, name);
    const DTypeInfo info = dtype_info(expected);
    if (!dtype_equal(array.dtype(), numpy_dtype(expected))) {
        throw py::value_error(
            std::string(name) + " must have native-endian dtype " + info.name
            + ", got " + py::cast<std::string>(py::str(array.dtype())));
    }
}

[[nodiscard]] std::size_t checked_buffer_size(
    const py::ssize_t rows,
    const std::uint64_t record_size,
    const char* name) {
    if (rows < 0) {
        throw py::value_error(std::string(name) + " has a negative row count");
    }
    const auto row_count = static_cast<std::uint64_t>(rows);
    if (record_size != 0
        && row_count > std::numeric_limits<std::uint64_t>::max() / record_size) {
        throw py::value_error(std::string(name) + " buffer is too large");
    }
    const std::uint64_t bytes = row_count * record_size;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw py::value_error(std::string(name) + " buffer is too large");
    }
    return static_cast<std::size_t>(bytes);
}

[[nodiscard]] std::span<const std::uint64_t> validate_ids(
    const py::array& ids) {
    if (!dtype_equal(ids.dtype(), py::dtype::of<std::uint64_t>())) {
        throw py::value_error(
            "ids must have native-endian dtype uint64, got "
            + py::cast<std::string>(py::str(ids.dtype())));
    }
    require_native_endian(ids, "ids");
    if (ids.ndim() != 1) {
        throw py::value_error("ids must have shape (N,)");
    }
    require_c_contiguous(ids, "ids");
    require_aligned(ids, "ids");
    return {static_cast<const std::uint64_t*>(ids.data()),
            static_cast<std::size_t>(ids.shape(0))};
}

[[nodiscard]] std::span<const std::byte> validate_embeddings(
    const py::array& embeddings,
    const py::ssize_t expected_rows,
    const std::uint32_t dimension,
    const throwdb::DType dtype,
    const std::uint64_t record_size) {
    require_dtype(embeddings, dtype, "embeddings");
    if (embeddings.ndim() != 2) {
        throw py::value_error("embeddings must have shape (N, dimension)");
    }
    if (embeddings.shape(0) != expected_rows) {
        throw py::value_error(
            "ids and embeddings must have the same number of rows");
    }
    if (embeddings.shape(1) != static_cast<py::ssize_t>(dimension)) {
        throw py::value_error(
            "embedding dimension does not match the storage dimension");
    }
    require_c_contiguous(embeddings, "embeddings");
    require_aligned(embeddings, "embeddings");

    const std::size_t bytes = checked_buffer_size(
        expected_rows, record_size, "embeddings");
    if (embeddings.nbytes() < 0
        || static_cast<std::uint64_t>(embeddings.nbytes()) != bytes) {
        throw py::value_error(
            "embeddings byte count does not match rows * record_size");
    }
    return {static_cast<const std::byte*>(embeddings.data()),
            bytes};
}

[[nodiscard]] std::span<std::byte> validate_output(
    py::array& output,
    const py::ssize_t expected_rows,
    const std::uint32_t dimension,
    const throwdb::DType dtype,
    const std::uint64_t record_size) {
    require_dtype(output, dtype, "output");
    if (output.ndim() != 2
        || output.shape(0) != expected_rows
        || output.shape(1) != static_cast<py::ssize_t>(dimension)) {
        throw py::value_error("output must have shape (len(ids), dimension)");
    }
    require_c_contiguous(output, "output");
    require_aligned(output, "output");
    if (!output.writeable()) {
        throw py::value_error("output must be writable");
    }

    const std::size_t bytes = checked_buffer_size(
        expected_rows, record_size, "output");
    if (output.nbytes() < 0
        || static_cast<std::uint64_t>(output.nbytes()) != bytes) {
        throw py::value_error(
            "output byte count does not match rows * record_size");
    }
    return {static_cast<std::byte*>(output.mutable_data()),
            bytes};
}

[[nodiscard]] std::span<std::uint64_t> validate_writable_id_output(
    py::array& ids) {
    static_cast<void>(validate_ids(ids));
    if (!ids.writeable()) {
        throw py::value_error("ids must be writable");
    }
    return {
        static_cast<std::uint64_t*>(ids.mutable_data()),
        static_cast<std::size_t>(ids.shape(0))};
}

[[nodiscard]] std::span<throwdb::NativeIdLocation>
validate_location_output(
    py::array& locations, const py::ssize_t expected_rows) {
    if (!dtype_equal(
            locations.dtype(),
            py::dtype::of<throwdb::NativeIdLocation>())) {
        throw py::value_error(
            "locations must have dtype "
            "[('segment_id', '<u4'), ('row_index', '<u4')]");
    }
    require_native_endian(locations, "locations");
    if (locations.ndim() != 1
        || locations.shape(0) != expected_rows) {
        throw py::value_error(
            "locations must have shape (len(ids),)");
    }
    require_c_contiguous(locations, "locations");
    require_aligned(locations, "locations");
    if (!locations.writeable()) {
        throw py::value_error("locations must be writable");
    }
    return {
        static_cast<throwdb::NativeIdLocation*>(locations.mutable_data()),
        static_cast<std::size_t>(locations.shape(0))};
}

struct ByteAddressRange {
    std::uintptr_t begin{0};
    std::uintptr_t end{0};
};

template <typename Value>
[[nodiscard]] ByteAddressRange checked_byte_address_range(
    const std::span<Value> values,
    const char* const name) {
    if (values.empty()) {
        return {};
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(
        values.data());
    const std::size_t bytes = values.size_bytes();
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw py::value_error(
            std::string(name) + " output address range overflows uintptr_t");
    }
    return ByteAddressRange{begin, begin + bytes};
}

[[nodiscard]] bool byte_address_ranges_overlap(
    const ByteAddressRange left,
    const ByteAddressRange right) noexcept {
    return left.begin != left.end && right.begin != right.end
           && left.begin < right.end && right.begin < left.end;
}

void require_disjoint_id_scan_outputs(
    const std::span<std::uint64_t> ids,
    const std::span<throwdb::NativeIdLocation> locations) {
    if (byte_address_ranges_overlap(
            checked_byte_address_range(ids, "ids"),
            checked_byte_address_range(locations, "locations"))) {
        throw py::value_error(
            "ID scan output buffers must not overlap: ids and locations");
    }
}

void require_disjoint_record_scan_outputs(
    const std::span<std::uint64_t> ids,
    const std::span<std::byte> embeddings,
    const std::span<throwdb::NativeIdLocation> locations) {
    const ByteAddressRange id_range = checked_byte_address_range(ids, "ids");
    const ByteAddressRange embedding_range =
        checked_byte_address_range(embeddings, "embeddings");
    const ByteAddressRange location_range =
        checked_byte_address_range(locations, "locations");
    if (byte_address_ranges_overlap(id_range, embedding_range)) {
        throw py::value_error(
            "record scan output buffers must not overlap: ids and embeddings");
    }
    if (byte_address_ranges_overlap(id_range, location_range)) {
        throw py::value_error(
            "record scan output buffers must not overlap: ids and locations");
    }
    if (byte_address_ranges_overlap(embedding_range, location_range)) {
        throw py::value_error(
            "record scan output buffers must not overlap: embeddings and "
            "locations");
    }
}

class PyWriter {
public:
    static std::unique_ptr<PyWriter> create(
        const py::object& path,
        const std::uint32_t dimension,
        const std::uint64_t shard_size_bytes,
        const std::string& dtype) {
        if (dimension == 0) {
            throw py::value_error("dimension must be greater than zero");
        }
        if (shard_size_bytes == 0) {
            throw py::value_error("shard_size_bytes must be greater than zero");
        }
        throwdb::WriterOptions options;
        options.dimension = dimension;
        options.dtype = checked_dtype(dtype);
        options.shard_size_bytes = shard_size_bytes;
        const auto native_path = filesystem_path(path);
        std::unique_ptr<throwdb::Writer> writer;
        {
            py::gil_scoped_release release;
            writer = throwdb::Writer::create(native_path, options);
        }
        return std::unique_ptr<PyWriter>(new PyWriter(std::move(writer)));
    }

    void add(const py::array& ids, const py::array& embeddings) {
        ensure_open();
        const auto id_span = validate_ids(ids);
        const auto embedding_span = validate_embeddings(
            embeddings, ids.shape(0), dimension(), writer_->dtype(),
            writer_->record_size());
        try {
            py::gil_scoped_release release;
            writer_->append(id_span, embedding_span);
        } catch (const throwdb::DuplicateIdError& error) {
            throw py::value_error(error.what());
        }
    }

    void finish() {
        if (finished_) {
            return;
        }
        try {
            py::gil_scoped_release release;
            writer_->finish();
        } catch (const throwdb::DuplicateIdError& error) {
            throw py::value_error(error.what());
        }
        finished_ = true;
    }

    void abort() noexcept {
        if (writer_ != nullptr && !finished_) {
            // Destroying an unfinished core writer closes the staging files but
            // never publishes the incomplete store.
            writer_.reset();
            finished_ = true;
        }
    }

    [[nodiscard]] std::uint32_t dimension() const {
        ensure_alive();
        return writer_->dimension();
    }

    [[nodiscard]] std::uint64_t count() const {
        ensure_alive();
        return writer_->count();
    }

    [[nodiscard]] std::string dtype() const {
        ensure_alive();
        return dtype_info(writer_->dtype()).name;
    }

    [[nodiscard]] std::uint32_t format_version() const {
        ensure_alive();
        return static_cast<std::uint32_t>(writer_->format_version());
    }

    PyWriter& enter() {
        ensure_open();
        return *this;
    }

    bool exit(const py::object& exception_type,
              const py::object&,
              const py::object&) {
        if (exception_type.is_none()) {
            finish();
        } else {
            py::gil_scoped_release release;
            abort();
        }
        return false;
    }

private:
    explicit PyWriter(std::unique_ptr<throwdb::Writer> writer)
        : writer_(std::move(writer)) {}

    void ensure_open() const {
        ensure_alive();
        if (finished_ || writer_->finished()) {
            throw std::runtime_error("writer is already finished");
        }
    }

    void ensure_alive() const {
        if (writer_ == nullptr) {
            throw std::runtime_error("writer is closed");
        }
    }

    std::unique_ptr<throwdb::Writer> writer_;
    bool finished_{false};
};

class PyStorage {
public:
    static std::unique_ptr<PyStorage> open(
        const py::object& path,
        const std::int64_t io_workers,
        const std::int64_t io_queue_capacity,
        const std::int64_t io_ranges_per_task,
        const std::string& io_backend,
        const std::int64_t io_uring_queue_depth,
        const std::int64_t io_uring_workers,
        const py::object& max_read_range_bytes,
        const py::object& max_gap_bytes) {
        const std::uint64_t checked_max_read_range_bytes =
            checked_uint64_option(
                max_read_range_bytes, "max_read_range_bytes", false);
        const std::uint64_t checked_max_gap_bytes = checked_uint64_option(
            max_gap_bytes, "max_gap_bytes", true);
        const throwdb::ReaderOptions options = checked_reader_options(
            io_workers,
            io_queue_capacity,
            io_ranges_per_task,
            io_backend,
            io_uring_queue_depth,
            io_uring_workers,
            false,
            checked_max_read_range_bytes,
            checked_max_gap_bytes);
        const auto native_path = filesystem_path(path);
        std::unique_ptr<throwdb::Storage> reader;
        {
            py::gil_scoped_release release;
            reader = throwdb::Storage::open(native_path, options);
        }
        return std::unique_ptr<PyStorage>(new PyStorage(std::move(reader)));
    }

    [[nodiscard]] std::uint32_t dimension() const {
        return reader_->dimension();
    }

    [[nodiscard]] std::uint64_t count() const {
        return reader_->count();
    }

    [[nodiscard]] std::string dtype() const {
        return dtype_info(reader_->dtype()).name;
    }

    [[nodiscard]] std::uint32_t format_version() const {
        return static_cast<std::uint32_t>(reader_->format_version());
    }

    py::array get(const py::array& ids) const {
        const auto id_span = validate_ids(ids);
        py::array output(
            numpy_dtype(reader_->dtype()),
            {ids.shape(0), static_cast<py::ssize_t>(dimension())});
        auto output_span = validate_output(
            output, ids.shape(0), dimension(), reader_->dtype(),
            reader_->record_size());
        try {
            py::gil_scoped_release release;
            reader_->get_into(id_span, output_span);
        } catch (const throwdb::MissingIdError& error) {
            throw py::key_error(error.what());
        }
        return output;
    }

    void get_into(const py::array& ids, py::array output) const {
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output, ids.shape(0), dimension(), reader_->dtype(),
            reader_->record_size());
        try {
            py::gil_scoped_release release;
            reader_->get_into(id_span, output_span);
        } catch (const throwdb::MissingIdError& error) {
            throw py::key_error(error.what());
        }
    }

private:
    explicit PyStorage(std::unique_ptr<throwdb::Storage> reader)
        : reader_(std::move(reader)) {}

    std::unique_ptr<throwdb::Storage> reader_;
};

class PyNativeCollectionBuilder {
public:
    static std::unique_ptr<PyNativeCollectionBuilder> create(
        const py::object& path,
        const std::uint32_t dimension,
        const std::uint64_t shard_size_bytes,
        const std::string& dtype,
        const std::uint32_t format_version) {
        if (dimension == 0) {
            throw py::value_error("dimension must be greater than zero");
        }
        if (shard_size_bytes == 0) {
            throw py::value_error("shard_size_bytes must be greater than zero");
        }
        if (format_version != 6 && format_version != 7) {
            throw py::value_error("format_version must be 6 or 7");
        }

        throwdb::NativeBuilderOptions options;
        options.dimension = dimension;
        options.dtype = checked_dtype(dtype);
        options.segment_size_bytes = shard_size_bytes;
        options.format_version = format_version;
        const auto native_path = filesystem_path(path);
        std::unique_ptr<throwdb::NativeCollectionBuilder> builder;
        {
            py::gil_scoped_release release;
            builder = throwdb::NativeCollectionBuilder::create(
                native_path, options);
        }
        return std::unique_ptr<PyNativeCollectionBuilder>(
            new PyNativeCollectionBuilder(std::move(builder)));
    }

    void add(const py::array& ids, const py::array& embeddings) {
        ensure_open();
        const auto id_span = validate_ids(ids);
        const auto embedding_span = validate_embeddings(
            embeddings,
            ids.shape(0),
            builder_->dimension(),
            builder_->dtype(),
            builder_->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            builder_->append(id_span, embedding_span);
        });
    }

    void finish() {
        ensure_alive();
        if (finished_ || builder_->finished()) {
            finished_ = true;
            return;
        }
        translate_id_errors([&] {
            py::gil_scoped_release release;
            builder_->finish();
        });
        finished_ = true;
    }

    void abort() {
        if (builder_ == nullptr || finished_) {
            return;
        }
        {
            py::gil_scoped_release release;
            builder_.reset();
        }
        finished_ = true;
    }

    [[nodiscard]] std::uint32_t dimension() const {
        ensure_alive();
        return builder_->dimension();
    }

    [[nodiscard]] std::uint64_t count() const {
        ensure_alive();
        return builder_->count();
    }

    [[nodiscard]] std::string dtype() const {
        ensure_alive();
        return dtype_info(builder_->dtype()).name;
    }

    [[nodiscard]] std::uint32_t format_version() const {
        ensure_alive();
        return builder_->format_version();
    }

private:
    explicit PyNativeCollectionBuilder(
        std::unique_ptr<throwdb::NativeCollectionBuilder> builder)
        : builder_(std::move(builder)) {}

    void ensure_open() const {
        ensure_alive();
        if (finished_ || builder_->finished()) {
            throw std::runtime_error("collection builder is already finished");
        }
    }

    void ensure_alive() const {
        if (builder_ == nullptr) {
            throw std::runtime_error("collection builder is closed");
        }
    }

    std::unique_ptr<throwdb::NativeCollectionBuilder> builder_;
    bool finished_{false};
};

class PyNativeReaderTrial {
public:
    PyNativeReaderTrial(
        std::shared_ptr<throwdb::NativeCollection> collection,
        std::unique_ptr<throwdb::NativeReaderTrial> trial)
        : collection_(std::move(collection)), trial_(std::move(trial)) {}

    void close() {
        std::shared_ptr<throwdb::NativeReaderTrial> trial;
        std::shared_ptr<throwdb::NativeCollection> collection;
        {
            const std::scoped_lock lock(mutex_);
            trial = std::move(trial_);
            collection = std::move(collection_);
        }
        if (trial == nullptr && collection == nullptr) {
            return;
        }
        py::gil_scoped_release release;
        trial.reset();
        collection.reset();
    }

    void get_into(const py::array& ids, py::array output) const {
        const auto [collection, trial] = acquire();
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->get_into_with_reader(*trial, id_span, output_span);
        });
    }

    [[nodiscard]] py::dict get_into_profiled(
        const py::array& ids, py::array output) const {
        const auto [collection, trial] = acquire();
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        const throwdb::NativeReadStats stats = translate_id_errors([&] {
            py::gil_scoped_release release;
            return collection->get_into_with_reader_profiled(
                *trial, id_span, output_span);
        });
        return native_read_stats_dict(stats);
    }

private:
    using Handles = std::pair<
        std::shared_ptr<throwdb::NativeCollection>,
        std::shared_ptr<throwdb::NativeReaderTrial>>;

    [[nodiscard]] Handles acquire() const {
        const std::scoped_lock lock(mutex_);
        if (collection_ == nullptr || trial_ == nullptr) {
            throw std::runtime_error("native reader trial is closed");
        }
        return {collection_, trial_};
    }

    mutable std::mutex mutex_;
    std::shared_ptr<throwdb::NativeCollection> collection_;
    std::shared_ptr<throwdb::NativeReaderTrial> trial_;
};

class PyNativeIdScan {
public:
    explicit PyNativeIdScan(std::unique_ptr<throwdb::NativeIdScan> scan)
        : scan_(std::move(scan)) {
        if (scan_ == nullptr) {
            throw std::runtime_error("native ID scan is null");
        }
    }

    void close() {
        std::shared_ptr<throwdb::NativeIdScan> scan;
        {
            const std::scoped_lock lock(handle_mutex_);
            scan = std::move(scan_);
        }
        if (scan == nullptr) {
            return;
        }
        py::gil_scoped_release release;
        scan.reset();
    }

    [[nodiscard]] py::dict metadata() const {
        const auto scan = acquire();
        const throwdb::NativeIdScanMetadata& metadata = scan->metadata();
        py::dict result;
        result["collection_id_low"] = metadata.collection_id_low;
        result["collection_id_high"] = metadata.collection_id_high;
        result["catalog_generation"] = metadata.catalog_generation;
        result["committed_sequence"] = metadata.committed_sequence;
        result["live_count"] = metadata.live_count;
        result["snapshot_bytes"] = metadata.snapshot_bytes;
        return result;
    }

    [[nodiscard]] bool done() const {
        const std::unique_lock operation(
            operation_mutex_, std::try_to_lock);
        if (!operation.owns_lock()) {
            throw std::runtime_error(
                "native ID scan is already in use by another thread");
        }
        return acquire()->done();
    }

    [[nodiscard]] std::uint64_t read_into(
        py::array ids,
        const py::object& locations_object) {
        if (!dtype_equal(ids.dtype(), py::dtype::of<std::uint64_t>())) {
            throw py::value_error(
                "ids must have native-endian dtype uint64, got "
                + py::cast<std::string>(py::str(ids.dtype())));
        }
        require_native_endian(ids, "ids");
        if (ids.ndim() != 1) {
            throw py::value_error("ids must have shape (N,)");
        }
        require_c_contiguous(ids, "ids");
        require_aligned(ids, "ids");
        if (!ids.writeable()) {
            throw py::value_error("ids must be writable");
        }
        std::span<std::uint64_t> id_span{
            static_cast<std::uint64_t*>(ids.mutable_data()),
            static_cast<std::size_t>(ids.shape(0))};

        py::array locations;
        std::span<throwdb::NativeIdLocation> location_span;
        if (!locations_object.is_none()) {
            if (!py::isinstance<py::array>(locations_object)) {
                throw py::type_error(
                    "locations must be a NumPy ndarray or None");
            }
            locations = py::cast<py::array>(locations_object);
            if (!dtype_equal(
                    locations.dtype(),
                    py::dtype::of<throwdb::NativeIdLocation>())) {
                throw py::value_error(
                    "locations must have dtype "
                    "[('segment_id', '<u4'), ('row_index', '<u4')]");
            }
            require_native_endian(locations, "locations");
            if (locations.ndim() != 1
                || locations.shape(0) != ids.shape(0)) {
                throw py::value_error(
                    "locations must have shape (len(ids),)");
            }
            require_c_contiguous(locations, "locations");
            require_aligned(locations, "locations");
            if (!locations.writeable()) {
                throw py::value_error("locations must be writable");
            }
            location_span = {
                static_cast<throwdb::NativeIdLocation*>(
                    locations.mutable_data()),
                static_cast<std::size_t>(locations.shape(0))};
        }

        require_disjoint_id_scan_outputs(id_span, location_span);

        const std::unique_lock operation(
            operation_mutex_, std::try_to_lock);
        if (!operation.owns_lock()) {
            throw std::runtime_error(
                "native ID scan is already in use by another thread");
        }
        const auto scan = acquire();
        py::gil_scoped_release release;
        return scan->read_into(id_span, location_span);
    }

private:
    [[nodiscard]] std::shared_ptr<throwdb::NativeIdScan> acquire() const {
        const std::scoped_lock lock(handle_mutex_);
        if (scan_ == nullptr) {
            throw std::runtime_error("native ID scan is closed");
        }
        return scan_;
    }

    mutable std::mutex handle_mutex_;
    mutable std::mutex operation_mutex_;
    std::shared_ptr<throwdb::NativeIdScan> scan_;
};

class PyNativeReadView {
public:
    explicit PyNativeReadView(
        std::unique_ptr<throwdb::NativeReadView> view)
        : view_(std::move(view)) {
        if (view_ == nullptr) {
            throw std::runtime_error("native read view is null");
        }
    }

    void close() {
        std::shared_ptr<throwdb::NativeReadView> view;
        {
            const std::scoped_lock lock(handle_mutex_);
            view = std::move(view_);
        }
        if (view == nullptr) {
            return;
        }
        py::gil_scoped_release release;
        view.reset();
    }

    [[nodiscard]] py::dict metadata() const {
        return read_view_metadata_dict(acquire()->metadata());
    }
    [[nodiscard]] std::uint32_t output_dimension() const {
        return acquire()->output_dimension();
    }
    [[nodiscard]] std::string dtype() const {
        return dtype_info(acquire()->dtype()).name;
    }
    void get_into(const py::array& ids, py::array output) const {
        const auto view = acquire();
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output,
            ids.shape(0),
            view->output_dimension(),
            view->dtype(),
            embedding_record_size(
                view->dtype(), view->output_dimension()));
        translate_id_errors([&] {
            py::gil_scoped_release release;
            view->get_into(id_span, output_span);
        });
    }

private:
    [[nodiscard]] std::shared_ptr<throwdb::NativeReadView> acquire() const {
        const std::scoped_lock lock(handle_mutex_);
        if (view_ == nullptr) {
            throw std::runtime_error("native read view is closed");
        }
        return view_;
    }

    mutable std::mutex handle_mutex_;
    std::shared_ptr<throwdb::NativeReadView> view_;
};

class PyNativeRecordScan {
public:
    explicit PyNativeRecordScan(
        std::unique_ptr<throwdb::NativeRecordScan> scan)
        : scan_(std::move(scan)) {
        if (scan_ == nullptr) {
            throw std::runtime_error("native record scan is null");
        }
    }

    void close() {
        std::shared_ptr<throwdb::NativeRecordScan> scan;
        {
            const std::scoped_lock lock(handle_mutex_);
            scan = std::move(scan_);
        }
        if (scan == nullptr) {
            return;
        }
        py::gil_scoped_release release;
        scan.reset();
    }

    [[nodiscard]] py::dict metadata() const {
        return read_view_metadata_dict(acquire()->metadata());
    }
    [[nodiscard]] std::uint32_t output_dimension() const {
        return acquire()->output_dimension();
    }
    [[nodiscard]] std::string dtype() const {
        return dtype_info(acquire()->dtype()).name;
    }
    [[nodiscard]] bool done() const {
        const std::unique_lock operation(
            operation_mutex_, std::try_to_lock);
        if (!operation.owns_lock()) {
            throw std::runtime_error(
                "native record scan is already in use by another thread");
        }
        return acquire()->done();
    }
    [[nodiscard]] std::uint64_t read_into(
        py::array ids,
        py::array embeddings,
        const py::object& locations_object) {
        auto id_span = validate_writable_id_output(ids);
        const auto scan = acquire();
        auto embedding_span = validate_output(
            embeddings,
            ids.shape(0),
            scan->output_dimension(),
            scan->dtype(),
            embedding_record_size(
                scan->dtype(), scan->output_dimension()));

        py::array locations;
        std::span<throwdb::NativeIdLocation> location_span;
        if (!locations_object.is_none()) {
            if (!py::isinstance<py::array>(locations_object)) {
                throw py::type_error(
                    "locations must be a NumPy ndarray or None");
            }
            locations = py::cast<py::array>(locations_object);
            location_span = validate_location_output(
                locations, ids.shape(0));
        }

        require_disjoint_record_scan_outputs(
            id_span, embedding_span, location_span);

        const std::unique_lock operation(
            operation_mutex_, std::try_to_lock);
        if (!operation.owns_lock()) {
            throw std::runtime_error(
                "native record scan is already in use by another thread");
        }
        py::gil_scoped_release release;
        return scan->read_into(
            id_span, embedding_span, location_span);
    }

private:
    [[nodiscard]] std::shared_ptr<throwdb::NativeRecordScan> acquire() const {
        const std::scoped_lock lock(handle_mutex_);
        if (scan_ == nullptr) {
            throw std::runtime_error("native record scan is closed");
        }
        return scan_;
    }

    mutable std::mutex handle_mutex_;
    mutable std::mutex operation_mutex_;
    std::shared_ptr<throwdb::NativeRecordScan> scan_;
};

class PyNativeCollection {
public:
    [[nodiscard]] static py::dict reader_capabilities(
        const std::int64_t io_uring_queue_depth,
        const std::int64_t io_uring_workers) {
        const std::uint32_t checked_queue_depth =
            checked_io_uring_queue_depth(io_uring_queue_depth);
        const std::uint32_t checked_workers =
            checked_io_uring_workers(io_uring_workers);
        if (checked_workers > checked_queue_depth) {
            throw py::value_error(
                "io_uring_queue_depth must be at least io_uring_workers");
        }
        throwdb::NativeReaderCapabilities capabilities;
        {
            py::gil_scoped_release release;
            capabilities = throwdb::NativeCollection::reader_capabilities(
                checked_queue_depth, checked_workers);
        }
        py::dict result;
        result["io_uring_available"] = capabilities.io_uring_available;
        result["io_uring_probe_queue_depth"] =
            capabilities.io_uring_probe_queue_depth;
        result["io_uring_probe_workers"] =
            capabilities.io_uring_probe_workers;
        result["io_uring_reason"] = capabilities.io_uring_reason.empty()
            ? py::none()
            : py::cast(capabilities.io_uring_reason);
        return result;
    }

    [[nodiscard]] static py::dict inspect_metadata(const py::object& path) {
        const auto native_path = filesystem_path(path);
        throwdb::NativeCollectionMetadata metadata;
        {
            py::gil_scoped_release release;
            metadata = throwdb::NativeCollection::inspect_metadata(native_path);
        }
        py::dict result;
        result["collection_id_low"] = metadata.collection_id_low;
        result["collection_id_high"] = metadata.collection_id_high;
        result["format_version"] = metadata.format_version;
        result["dimension"] = metadata.dimension;
        result["dtype"] = dtype_info(metadata.dtype).name;
        result["record_size"] = metadata.record_size;
        result["segment_size_bytes"] = metadata.segment_size_bytes;
        result["layout"] = metadata.layout;
        result["feature_discriminator"] = metadata.feature_discriminator;
        result["required_feature_bits"] = metadata.required_feature_bits;
        result["optional_feature_bits"] = metadata.optional_feature_bits;
        return result;
    }

    [[nodiscard]] static py::dict inspect_orphans(const py::object& path) {
        const auto native_path = filesystem_path(path);
        throwdb::NativeOrphanInventory inventory;
        {
            py::gil_scoped_release release;
            inventory =
                throwdb::NativeCollection::inspect_orphans(native_path);
        }
        return orphan_inventory_dict(inventory);
    }

    [[nodiscard]] static py::dict reclaim_orphans_at(
        const py::object& path,
        const py::object& snapshot_unix_ns,
        const py::object& min_age_ns,
        const py::object& max_files,
        const py::object& max_bytes) {
        throwdb::NativeOrphanCollectionOptions options;
        options.snapshot_unix_ns = checked_uint64_option(
            snapshot_unix_ns, "snapshot_unix_ns", true);
        options.min_age_ns = checked_uint64_option(
            min_age_ns, "min_age_ns", true);
        options.max_files = checked_uint64_option(
            max_files, "max_files", false);
        options.max_bytes = checked_uint64_option(
            max_bytes, "max_bytes", false);
        const auto native_path = filesystem_path(path);
        throwdb::NativeOrphanCollectionResult result;
        {
            py::gil_scoped_release release;
            result = throwdb::NativeCollection::reclaim_orphans(
                native_path, options);
        }
        return orphan_collection_result_dict(result);
    }

    static std::unique_ptr<PyNativeCollection> open(
        const py::object& path,
        const std::string& mode,
        const std::int64_t io_workers,
        const std::int64_t io_queue_capacity,
        const std::int64_t io_ranges_per_task,
        const std::string& io_backend,
        const std::int64_t io_uring_queue_depth,
        const std::int64_t io_uring_workers,
        const py::object& verify_native_index,
        const py::object& max_read_range_bytes,
        const py::object& max_gap_bytes,
        const py::object& calibration,
        const py::object& max_wal_bytes,
        const py::object& max_wal_operations,
        const py::object& max_overlay_entries,
        const py::object& max_overlay_bytes) {
        bool writable = false;
        if (mode == "r") {
            writable = false;
        } else if (mode == "r+") {
            writable = true;
        } else {
            throw py::value_error("mode must be 'r' or 'r+'");
        }
        const bool checked_calibration = checked_python_bool(
            calibration, "calibration");
        if (writable && checked_calibration) {
            throw py::value_error("calibration handles must be read-only");
        }

        const std::uint64_t checked_max_read_range_bytes =
            checked_uint64_option(
                max_read_range_bytes, "max_read_range_bytes", false);
        const std::uint64_t checked_max_gap_bytes = checked_uint64_option(
            max_gap_bytes, "max_gap_bytes", true);
        const throwdb::ReaderOptions options = checked_reader_options(
            io_workers,
            io_queue_capacity,
            io_ranges_per_task,
            io_backend,
            io_uring_queue_depth,
            io_uring_workers,
            checked_python_bool(
                verify_native_index, "verify_native_index"),
            checked_max_read_range_bytes,
            checked_max_gap_bytes);
        throwdb::NativeOpenOptions open_options;
        open_options.max_wal_bytes = checked_uint64_option(
            max_wal_bytes, "max_wal_bytes", false);
        open_options.max_wal_operations = checked_uint64_option(
            max_wal_operations, "max_wal_operations", false);
        open_options.max_overlay_entries = checked_uint64_option(
            max_overlay_entries, "max_overlay_entries", false);
        open_options.max_overlay_bytes = checked_uint64_option(
            max_overlay_bytes, "max_overlay_bytes", false);
        const auto native_path = filesystem_path(path);
        std::unique_ptr<throwdb::NativeCollection> collection;
        {
            py::gil_scoped_release release;
            collection = throwdb::NativeCollection::open(
                native_path, options, writable, checked_calibration,
                open_options);
        }
        return std::unique_ptr<PyNativeCollection>(
            new PyNativeCollection(std::move(collection)));
    }

    void close() {
        std::shared_ptr<throwdb::NativeCollection> collection;
        {
            const std::scoped_lock lock(collection_mutex_);
            collection = std::move(collection_);
        }
        if (collection == nullptr) {
            return;
        }

        // Native calls release the GIL. Keep an independent shared owner in
        // every operation so close() can safely race an already-started read
        // or mutation without destroying its C++ object underneath it.
        py::gil_scoped_release release;
        collection.reset();
    }

    [[nodiscard]] std::uint32_t dimension() const {
        return acquire_collection()->dimension();
    }

    [[nodiscard]] std::uint64_t count() const {
        return acquire_collection()->count();
    }

    [[nodiscard]] std::string dtype() const {
        return dtype_info(acquire_collection()->dtype()).name;
    }

    [[nodiscard]] std::uint32_t format_version() const {
        return acquire_collection()->format_version();
    }

    [[nodiscard]] py::dict open_stats() const {
        const throwdb::NativeOpenStats stats =
            acquire_collection()->open_stats();
        py::dict result;
        result["total_seconds"] = stats.total_seconds;
        result["manifest_seconds"] = stats.manifest_seconds;
        result["index_seconds"] = stats.index_seconds;
        result["index_header_seconds"] = stats.index_header_seconds;
        result["index_allocation_seconds"] = stats.index_allocation_seconds;
        result["index_controls_seconds"] = stats.index_controls_seconds;
        result["index_slots_seconds"] = stats.index_slots_seconds;
        result["index_checksum_seconds"] = stats.index_checksum_seconds;
        result["index_structural_validation_seconds"] =
            stats.index_structural_validation_seconds;
        result["base_segments_seconds"] = stats.base_segments_seconds;
        result["wal_replay_seconds"] = stats.wal_replay_seconds;
        result["index_file_bytes"] = stats.index_file_bytes;
        result["index_body_checksum_present"] =
            stats.index_body_checksum_present;
        result["index_body_checksum_verified"] =
            stats.index_body_checksum_verified;
        result["index_structural_validation_performed"] =
            stats.index_structural_validation_performed;
        const char* validation = "checksum";
        if (stats.index_structural_validation_performed) {
            validation = stats.index_body_checksum_present
                ? "checksum_and_full"
                : "full_legacy_fallback";
        }
        result["index_validation"] = validation;
        return result;
    }

    [[nodiscard]] py::dict mutation_log_stats() const {
        const auto collection = acquire_collection();
        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->mutation_log_stats();
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict collect_orphans() const {
        const auto collection = acquire_collection();
        throwdb::NativeOrphanInventory inventory;
        {
            py::gil_scoped_release release;
            inventory = collection->collect_orphans();
        }
        return orphan_inventory_dict(inventory);
    }

    [[nodiscard]] py::dict reclaim_orphans(
        const py::object& snapshot_unix_ns,
        const py::object& min_age_ns,
        const py::object& max_files,
        const py::object& max_bytes) const {
        throwdb::NativeOrphanCollectionOptions options;
        options.snapshot_unix_ns = checked_uint64_option(
            snapshot_unix_ns, "snapshot_unix_ns", true);
        options.min_age_ns = checked_uint64_option(
            min_age_ns, "min_age_ns", true);
        options.max_files = checked_uint64_option(
            max_files, "max_files", false);
        options.max_bytes = checked_uint64_option(
            max_bytes, "max_bytes", false);
        const auto collection = acquire_collection();
        throwdb::NativeOrphanCollectionResult result;
        {
            py::gil_scoped_release release;
            result = collection->reclaim_orphans(options);
        }
        return orphan_collection_result_dict(result);
    }

    [[nodiscard]] std::unique_ptr<PyNativeReaderTrial> reader_trial(
        const std::int64_t io_workers,
        const std::int64_t io_queue_capacity,
        const std::int64_t io_ranges_per_task,
        const std::string& io_backend,
        const std::int64_t io_uring_queue_depth,
        const std::int64_t io_uring_workers,
        const py::object& max_read_range_bytes,
        const py::object& max_gap_bytes) const {
        const std::uint64_t checked_max_read_range_bytes =
            checked_uint64_option(
                max_read_range_bytes, "max_read_range_bytes", false);
        const std::uint64_t checked_max_gap_bytes = checked_uint64_option(
            max_gap_bytes, "max_gap_bytes", true);
        const throwdb::ReaderOptions options = checked_reader_options(
            io_workers,
            io_queue_capacity,
            io_ranges_per_task,
            io_backend,
            io_uring_queue_depth,
            io_uring_workers,
            false,
            checked_max_read_range_bytes,
            checked_max_gap_bytes);
        const auto collection = acquire_collection();
        std::unique_ptr<throwdb::NativeReaderTrial> trial;
        {
            py::gil_scoped_release release;
            trial = collection->create_reader_trial(options);
        }
        return std::make_unique<PyNativeReaderTrial>(
            collection, std::move(trial));
    }

    [[nodiscard]] py::dict sample_ids(
        const py::object& requested_count,
        const py::object& seed,
        const py::object& max_inspected_slots) const {
        const std::uint64_t requested = checked_uint64_option(
            requested_count, "requested_count", true);
        const std::uint64_t checked_seed = checked_uint64_option(
            seed, "seed", true);
        const std::uint64_t inspected = checked_uint64_option(
            max_inspected_slots, "max_inspected_slots", true);
        if (requested > kMaxIdSampleRows) {
            throw py::value_error(
                "requested_count must not exceed "
                + std::to_string(kMaxIdSampleRows));
        }
        if (inspected > kMaxIdSampleInspectedSlots) {
            throw py::value_error(
                "max_inspected_slots must not exceed "
                + std::to_string(kMaxIdSampleInspectedSlots));
        }
        const auto collection = acquire_collection();
        throwdb::NativeIdSample sample;
        {
            py::gil_scoped_release release;
            sample = collection->sample_ids(
                requested, checked_seed, inspected);
        }
        py::array_t<std::uint64_t> ids(
            static_cast<py::ssize_t>(sample.ids.size()));
        std::copy(sample.ids.begin(), sample.ids.end(), ids.mutable_data());
        py::dict result;
        result["ids"] = std::move(ids);
        result["requested_count"] = requested;
        result["inspected_slots"] = sample.inspected_slots;
        result["complete"] = sample.complete;
        return result;
    }

    [[nodiscard]] std::unique_ptr<PyNativeIdScan> scan_ids(
        const py::object& max_snapshot_bytes) const {
        const auto collection = acquire_collection();
        const std::uint64_t checked_budget = max_snapshot_bytes.is_none()
            ? std::numeric_limits<std::uint64_t>::max()
            : checked_uint64_option(
                  max_snapshot_bytes, "max_snapshot_bytes", true);
        std::unique_ptr<throwdb::NativeIdScan> scan;
        {
            py::gil_scoped_release release;
            scan = collection->scan_ids(checked_budget);
        }
        return std::make_unique<PyNativeIdScan>(std::move(scan));
    }

    [[nodiscard]] std::unique_ptr<PyNativeReadView> open_read_view(
        const py::object& prefix_dimensions,
        const py::object& max_snapshot_bytes) const {
        const auto collection = acquire_collection();
        const std::uint64_t prefix = prefix_dimensions.is_none()
            ? 0
            : checked_uint64_option(
                  prefix_dimensions, "prefix_dimensions", false);
        if (prefix > std::numeric_limits<std::uint32_t>::max()) {
            throw py::value_error(
                "prefix_dimensions exceeds the uint32 range");
        }
        const std::uint64_t budget = max_snapshot_bytes.is_none()
            ? std::numeric_limits<std::uint64_t>::max()
            : checked_uint64_option(
                  max_snapshot_bytes, "max_snapshot_bytes", true);
        std::unique_ptr<throwdb::NativeReadView> view;
        {
            py::gil_scoped_release release;
            view = collection->open_read_view(
                static_cast<std::uint32_t>(prefix), budget);
        }
        return std::make_unique<PyNativeReadView>(std::move(view));
    }

    [[nodiscard]] std::unique_ptr<PyNativeRecordScan> scan_records(
        const py::object& prefix_dimensions,
        const py::object& max_snapshot_bytes) const {
        const auto collection = acquire_collection();
        const std::uint64_t prefix = prefix_dimensions.is_none()
            ? 0
            : checked_uint64_option(
                  prefix_dimensions, "prefix_dimensions", false);
        if (prefix > std::numeric_limits<std::uint32_t>::max()) {
            throw py::value_error(
                "prefix_dimensions exceeds the uint32 range");
        }
        const std::uint64_t budget = max_snapshot_bytes.is_none()
            ? std::numeric_limits<std::uint64_t>::max()
            : checked_uint64_option(
                  max_snapshot_bytes, "max_snapshot_bytes", true);
        std::unique_ptr<throwdb::NativeRecordScan> scan;
        {
            py::gil_scoped_release release;
            scan = collection->scan_records(
                static_cast<std::uint32_t>(prefix), budget);
        }
        return std::make_unique<PyNativeRecordScan>(std::move(scan));
    }

    [[nodiscard]] py::dict rotate_wal() {
        const auto collection = acquire_collection();
        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->rotate_wal();
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict materialize_frozen_wals(
        const py::object& max_wals,
        const py::object& max_input_bytes,
        const py::object& max_input_entries,
        const py::object& max_memory_bytes) {
        const std::uint64_t checked_max_wals = checked_uint64_option(
            max_wals, "max_wals", false);
        const std::uint64_t checked_max_input_bytes = checked_uint64_option(
            max_input_bytes, "max_input_bytes", false);
        const std::uint64_t checked_max_input_entries = checked_uint64_option(
            max_input_entries, "max_input_entries", false);
        const std::uint64_t checked_max_memory_bytes = checked_uint64_option(
            max_memory_bytes, "max_memory_bytes", false);
        const auto collection = acquire_collection();
        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->materialize_frozen_wals(
                checked_max_wals,
                checked_max_input_bytes,
                checked_max_input_entries,
                checked_max_memory_bytes);
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict compact_delta_runs(
        const py::object& max_runs,
        const py::object& max_input_bytes,
        const py::object& max_input_entries) {
        const auto collection = acquire_collection();
        const std::uint64_t maximum_runs =
            collection->format_version() == 7
                ? kMaximumPartitionedFoldRuns
                : kMaximumDeltaCompactionMaxRuns;
        const std::uint64_t checked_max_runs = max_runs.is_none()
            ? (collection->format_version() == 7
                   ? kMaximumPartitionedFoldRuns
                   : std::uint64_t{8})
            : checked_uint64_option(max_runs, "max_runs", false);
        if (checked_max_runs < 2
            || checked_max_runs > maximum_runs) {
            throw py::value_error(
                "max_runs must be between 2 and "
                + std::to_string(maximum_runs));
        }
        const std::uint64_t checked_max_input_bytes =
            checked_uint64_option(
                max_input_bytes, "max_input_bytes", false);
        const std::uint64_t checked_max_input_entries =
            checked_uint64_option(
                max_input_entries, "max_input_entries", false);

        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->compact_delta_runs(
                checked_max_runs,
                checked_max_input_bytes,
                checked_max_input_entries);
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict clean_segments(
        const bool dry_run,
        const std::uint64_t max_candidate_segments,
        const std::uint64_t max_sidecar_scan_bytes,
        const std::uint64_t max_source_bytes,
        const std::uint64_t max_output_bytes,
        const std::uint64_t max_memory_bytes,
        const std::uint64_t min_garbage_rows,
        const std::uint64_t min_reclaimable_bytes,
        const std::uint32_t min_garbage_ratio_ppm,
        const std::uint32_t start_after_segment_id) {
        throwdb::NativeSegmentCleaningOptions options;
        options.dry_run = dry_run;
        options.max_candidate_segments = max_candidate_segments;
        options.max_sidecar_scan_bytes = max_sidecar_scan_bytes;
        options.max_source_bytes = max_source_bytes;
        options.max_output_bytes = max_output_bytes;
        options.max_memory_bytes = max_memory_bytes;
        options.min_garbage_rows = min_garbage_rows;
        options.min_reclaimable_bytes = min_reclaimable_bytes;
        options.min_garbage_ratio_ppm = min_garbage_ratio_ppm;
        options.start_after_segment_id = start_after_segment_id;
        const auto collection = acquire_collection();
        throwdb::NativeSegmentCleaningStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->clean_segments(options);
        }
        return segment_cleaning_stats_dict(stats);
    }

    py::array get(const py::array& ids) const {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        py::array output(
            numpy_dtype(collection->dtype()),
            {ids.shape(0),
             static_cast<py::ssize_t>(collection->dimension())});
        auto output_span = validate_output(
            output,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->get_into(id_span, output_span);
        });
        return output;
    }

    void get_into(const py::array& ids, py::array output) const {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->get_into(id_span, output_span);
        });
    }

    [[nodiscard]] py::dict get_into_profiled(
        const py::array& ids, py::array output) const {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        auto output_span = validate_output(
            output,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        const throwdb::NativeReadStats stats = translate_id_errors([&] {
            py::gil_scoped_release release;
            return collection->get_into_profiled(id_span, output_span);
        });
        return native_read_stats_dict(stats);
    }

    void insert(const py::array& ids, const py::array& embeddings) {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        const auto embedding_span = validate_embeddings(
            embeddings,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->insert(id_span, embedding_span);
        });
    }

    void update(const py::array& ids, const py::array& embeddings) {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        const auto embedding_span = validate_embeddings(
            embeddings,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->update(id_span, embedding_span);
        });
    }

    void upsert(const py::array& ids, const py::array& embeddings) {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        const auto embedding_span = validate_embeddings(
            embeddings,
            ids.shape(0),
            collection->dimension(),
            collection->dtype(),
            collection->record_size());
        translate_id_errors([&] {
            py::gil_scoped_release release;
            collection->upsert(id_span, embedding_span);
        });
    }

    py::array_t<bool> erase(const py::array& ids) {
        const auto collection = acquire_collection();
        const auto id_span = validate_ids(ids);
        std::vector<std::uint8_t> erased = translate_id_errors([&] {
            py::gil_scoped_release release;
            return collection->erase(id_span);
        });
        if (erased.size() != id_span.size()) {
            throw std::runtime_error(
                "native delete returned an invalid result length");
        }

        py::array_t<bool> output(static_cast<py::ssize_t>(erased.size()));
        bool* output_data = output.mutable_data();
        for (std::size_t index = 0; index < erased.size(); ++index) {
            output_data[index] = erased[index] != 0;
        }
        return output;
    }

private:
    explicit PyNativeCollection(
        std::unique_ptr<throwdb::NativeCollection> collection)
        : collection_(std::move(collection)) {}

    [[nodiscard]] std::shared_ptr<throwdb::NativeCollection>
    acquire_collection() const {
        const std::scoped_lock lock(collection_mutex_);
        if (collection_ == nullptr) {
            throw std::runtime_error("native collection is closed");
        }
        return collection_;
    }

    mutable std::mutex collection_mutex_;
    std::shared_ptr<throwdb::NativeCollection> collection_;
};

}  // namespace

PYBIND11_MODULE(_throwdb, module) {
    module.doc() = "Native bindings for the ThrowDB storage engine";
    module.attr("__version__") = "0.1.0";
    module.def(
        "_crc32c",
        [](const py::buffer& input,
           const std::size_t zero_offset,
           const std::size_t zero_size) {
            const py::buffer_info info = input.request();
            if (info.ndim != 1 || info.itemsize != 1
                || info.strides.empty() || info.strides[0] != 1) {
                throw py::value_error(
                    "CRC32C input must be a contiguous one-byte buffer");
            }
            if (info.size < 0) {
                throw py::value_error("CRC32C input size is invalid");
            }
            const auto size = static_cast<std::size_t>(info.size);
            if (zero_offset > size || zero_size > size - zero_offset) {
                throw py::value_error(
                    "CRC32C zeroed field is outside the input buffer");
            }
            const auto* data = static_cast<const std::byte*>(info.ptr);
            const std::span<const std::byte> bytes(data, size);
            throwdb::detail::Crc32c checksum;
            checksum.update(bytes.first(zero_offset));
            checksum.update_zeroes(zero_size);
            checksum.update(bytes.subspan(zero_offset + zero_size));
            return checksum.value();
        },
        py::arg("input"),
        py::arg("zero_offset") = 0,
        py::arg("zero_size") = 0,
        "Compute CRC32C while treating one byte range as zero.");
    module.def(
        "_fnv1a64",
        [](const py::buffer& input, std::uint64_t state) {
            const py::buffer_info info = input.request();
            if (info.ndim != 1 || info.itemsize != 1
                || info.strides.empty() || info.strides[0] != 1
                || !info.readonly) {
                throw py::value_error(
                    "FNV-1a input must be a contiguous, read-only, one-byte "
                    "buffer");
            }
            if (info.size < 0) {
                throw py::value_error("FNV-1a input size is invalid");
            }
            const auto size = static_cast<std::size_t>(info.size);
            const auto* data =
                static_cast<const std::uint8_t*>(info.ptr);
            constexpr std::uint64_t prime = 0x100000001b3ULL;
            {
                py::gil_scoped_release release;
                for (std::size_t index = 0; index != size; ++index) {
                    state ^= data[index];
                    state *= prime;
                }
            }
            return state;
        },
        py::arg("input"),
        py::arg("state"),
        "Continue a 64-bit FNV-1a digest over a contiguous byte buffer.");
    PYBIND11_NUMPY_DTYPE(
        throwdb::NativeIdLocation, segment_id, row_index);
    py::register_exception<throwdb::ReaderBackendError>(
        module, "ReaderBackendError", PyExc_RuntimeError);

    py::class_<PyWriter>(module, "Writer")
        .def_static(
            "create",
            &PyWriter::create,
            py::arg("path"),
            py::arg("dimension") = 1024,
            py::arg("shard_size_bytes") = kDefaultShardSizeBytes,
            py::arg("dtype") = "float16")
        .def("add", &PyWriter::add, py::arg("ids"), py::arg("embeddings"))
        .def("finish", &PyWriter::finish)
        .def("abort", &PyWriter::abort)
        .def_property_readonly("dimension", &PyWriter::dimension)
        .def_property_readonly("count", &PyWriter::count)
        .def_property_readonly("dtype", &PyWriter::dtype)
        .def_property_readonly("format_version", &PyWriter::format_version)
        .def("__enter__", &PyWriter::enter, py::return_value_policy::reference_internal)
        .def("__exit__", &PyWriter::exit);

    py::class_<PyStorage>(module, "Storage")
        .def_static(
            "open",
            &PyStorage::open,
            py::arg("path"),
            py::arg("io_workers") = 0,
            py::arg("io_queue_capacity") = 0,
            py::arg("io_ranges_per_task") = kDefaultIoRangesPerTask,
            py::arg("io_backend") = "thread_pool",
            py::arg("io_uring_queue_depth") = kDefaultIoUringQueueDepth,
            py::arg("io_uring_workers") = kDefaultIoUringWorkers,
            py::arg("max_read_range_bytes") = kDefaultMaxReadRangeBytes,
            py::arg("max_gap_bytes") = kDefaultMaxGapBytes)
        .def_property_readonly("dimension", &PyStorage::dimension)
        .def_property_readonly("count", &PyStorage::count)
        .def_property_readonly("dtype", &PyStorage::dtype)
        .def_property_readonly("format_version", &PyStorage::format_version)
        .def("get", &PyStorage::get, py::arg("ids"))
        .def("get_into", &PyStorage::get_into, py::arg("ids"), py::arg("output"));

    py::class_<PyNativeCollectionBuilder>(module, "NativeCollectionBuilder")
        .def_static(
            "create",
            &PyNativeCollectionBuilder::create,
            py::arg("path"),
            py::arg("dimension") = 1024,
            py::arg("shard_size_bytes") = kDefaultShardSizeBytes,
            py::arg("dtype") = "float16",
            py::arg("format_version") = 7)
        .def(
            "add",
            &PyNativeCollectionBuilder::add,
            py::arg("ids"),
            py::arg("embeddings"))
        .def("finish", &PyNativeCollectionBuilder::finish)
        .def("abort", &PyNativeCollectionBuilder::abort)
        .def_property_readonly(
            "dimension", &PyNativeCollectionBuilder::dimension)
        .def_property_readonly("count", &PyNativeCollectionBuilder::count)
        .def_property_readonly("dtype", &PyNativeCollectionBuilder::dtype)
        .def_property_readonly(
            "format_version", &PyNativeCollectionBuilder::format_version);

    py::class_<PyNativeReaderTrial>(module, "NativeReaderTrial")
        .def(
            "get_into",
            &PyNativeReaderTrial::get_into,
            py::arg("ids"),
            py::arg("output"))
        .def(
            "get_into_profiled",
            &PyNativeReaderTrial::get_into_profiled,
            py::arg("ids"),
            py::arg("output"))
        .def("close", &PyNativeReaderTrial::close);

    py::class_<PyNativeIdScan>(module, "NativeIdScan")
        .def_property_readonly("metadata", &PyNativeIdScan::metadata)
        .def_property_readonly("done", &PyNativeIdScan::done)
        .def(
            "read_into",
            &PyNativeIdScan::read_into,
            py::arg("ids"),
            py::arg("locations") = py::none())
        .def("close", &PyNativeIdScan::close)
        .def(
            "__enter__",
            [](PyNativeIdScan& scan) -> PyNativeIdScan& { return scan; },
            py::return_value_policy::reference_internal)
        .def(
            "__exit__",
            [](PyNativeIdScan& scan,
               const py::object&,
               const py::object&,
               const py::object&) { scan.close(); });

    py::class_<PyNativeReadView>(module, "NativeReadView")
        .def_property_readonly("metadata", &PyNativeReadView::metadata)
        .def_property_readonly(
            "output_dimension", &PyNativeReadView::output_dimension)
        .def_property_readonly("dtype", &PyNativeReadView::dtype)
        .def(
            "get_into",
            &PyNativeReadView::get_into,
            py::arg("ids"),
            py::arg("output"))
        .def("close", &PyNativeReadView::close)
        .def(
            "__enter__",
            [](PyNativeReadView& view) -> PyNativeReadView& { return view; },
            py::return_value_policy::reference_internal)
        .def(
            "__exit__",
            [](PyNativeReadView& view,
               const py::object&,
               const py::object&,
               const py::object&) { view.close(); });

    py::class_<PyNativeRecordScan>(module, "NativeRecordScan")
        .def_property_readonly("metadata", &PyNativeRecordScan::metadata)
        .def_property_readonly(
            "output_dimension", &PyNativeRecordScan::output_dimension)
        .def_property_readonly("dtype", &PyNativeRecordScan::dtype)
        .def_property_readonly("done", &PyNativeRecordScan::done)
        .def(
            "read_into",
            &PyNativeRecordScan::read_into,
            py::arg("ids"),
            py::arg("embeddings"),
            py::arg("locations") = py::none())
        .def("close", &PyNativeRecordScan::close)
        .def(
            "__enter__",
            [](PyNativeRecordScan& scan) -> PyNativeRecordScan& {
                return scan;
            },
            py::return_value_policy::reference_internal)
        .def(
            "__exit__",
            [](PyNativeRecordScan& scan,
               const py::object&,
               const py::object&,
               const py::object&) { scan.close(); });

    py::class_<PyNativeCollection>(module, "NativeCollection")
        .def_static(
            "reader_capabilities",
            &PyNativeCollection::reader_capabilities,
            py::arg("io_uring_queue_depth") = kDefaultIoUringQueueDepth,
            py::arg("io_uring_workers") = kDefaultIoUringWorkers)
        .def_static(
            "inspect_metadata",
            &PyNativeCollection::inspect_metadata,
            py::arg("path"))
        .def_static(
            "inspect_orphans",
            &PyNativeCollection::inspect_orphans,
            py::arg("path"))
        .def_static(
            "_reclaim_orphans_at",
            &PyNativeCollection::reclaim_orphans_at,
            py::arg("path"),
            py::arg("snapshot_unix_ns") = 0,
            py::arg("min_age_ns") = 0,
            py::arg("max_files") = kDefaultOrphanMaxFiles,
            py::arg("max_bytes") = kDefaultOrphanMaxBytes)
        .def_static(
            "open",
            &PyNativeCollection::open,
            py::arg("path"),
            py::arg("mode") = "r",
            py::arg("io_workers") = 0,
            py::arg("io_queue_capacity") = 0,
            py::arg("io_ranges_per_task") = kDefaultIoRangesPerTask,
            py::arg("io_backend") = "thread_pool",
            py::arg("io_uring_queue_depth") = kDefaultIoUringQueueDepth,
            py::arg("io_uring_workers") = kDefaultIoUringWorkers,
            py::arg("verify_native_index") = false,
            py::arg("max_read_range_bytes") = kDefaultMaxReadRangeBytes,
            py::arg("max_gap_bytes") = kDefaultMaxGapBytes,
            py::arg("calibration") = false,
            py::arg("max_wal_bytes") = kDefaultMaxWalBacklogBytes,
            py::arg("max_wal_operations") =
                kDefaultMaxWalBacklogOperations,
            py::arg("max_overlay_entries") = kDefaultMaxOverlayEntries,
            py::arg("max_overlay_bytes") = kDefaultMaxOverlayBytes)
        .def_property_readonly("dimension", &PyNativeCollection::dimension)
        .def_property_readonly("count", &PyNativeCollection::count)
        .def_property_readonly("dtype", &PyNativeCollection::dtype)
        .def_property_readonly(
            "format_version", &PyNativeCollection::format_version)
        .def_property_readonly("open_stats", &PyNativeCollection::open_stats)
        .def_property_readonly(
            "mutation_log_stats", &PyNativeCollection::mutation_log_stats)
        .def("collect_orphans", &PyNativeCollection::collect_orphans)
        .def(
            "reclaim_orphans",
            &PyNativeCollection::reclaim_orphans,
            py::arg("snapshot_unix_ns") = 0,
            py::arg("min_age_ns") = 0,
            py::arg("max_files") = kDefaultOrphanMaxFiles,
            py::arg("max_bytes") = kDefaultOrphanMaxBytes)
        .def(
            "reader_trial",
            &PyNativeCollection::reader_trial,
            py::arg("io_workers") = 0,
            py::arg("io_queue_capacity") = 0,
            py::arg("io_ranges_per_task") = kDefaultIoRangesPerTask,
            py::arg("io_backend") = "thread_pool",
            py::arg("io_uring_queue_depth") = kDefaultIoUringQueueDepth,
            py::arg("io_uring_workers") = kDefaultIoUringWorkers,
            py::arg("max_read_range_bytes") = kDefaultMaxReadRangeBytes,
            py::arg("max_gap_bytes") = kDefaultMaxGapBytes)
        .def(
            "sample_ids",
            &PyNativeCollection::sample_ids,
            py::arg("requested_count"),
            py::arg("seed"),
            py::arg("max_inspected_slots"))
        .def(
            "scan_ids",
            &PyNativeCollection::scan_ids,
            py::arg("max_snapshot_bytes") = py::none())
        .def(
            "open_read_view",
            &PyNativeCollection::open_read_view,
            py::arg("prefix_dimensions") = py::none(),
            py::arg("max_snapshot_bytes") = py::none())
        .def(
            "scan_records",
            &PyNativeCollection::scan_records,
            py::arg("prefix_dimensions") = py::none(),
            py::arg("max_snapshot_bytes") = py::none())
        .def("get", &PyNativeCollection::get, py::arg("ids"))
        .def(
            "get_into",
            &PyNativeCollection::get_into,
            py::arg("ids"),
            py::arg("output"))
        .def(
            "get_into_profiled",
            &PyNativeCollection::get_into_profiled,
            py::arg("ids"),
            py::arg("output"))
        .def(
            "insert",
            &PyNativeCollection::insert,
            py::arg("ids"),
            py::arg("embeddings"))
        .def(
            "update",
            &PyNativeCollection::update,
            py::arg("ids"),
            py::arg("embeddings"))
        .def(
            "upsert",
            &PyNativeCollection::upsert,
            py::arg("ids"),
            py::arg("embeddings"))
        .def("delete", &PyNativeCollection::erase, py::arg("ids"))
        .def("rotate_wal", &PyNativeCollection::rotate_wal)
        .def(
            "materialize_frozen_wals",
            &PyNativeCollection::materialize_frozen_wals,
            py::arg("max_wals") =
                std::numeric_limits<std::uint64_t>::max(),
            py::arg("max_input_bytes") =
                std::numeric_limits<std::uint64_t>::max(),
            py::arg("max_input_entries") =
                std::numeric_limits<std::uint64_t>::max(),
            py::arg("max_memory_bytes") =
                std::numeric_limits<std::uint64_t>::max())
        .def(
            "compact_delta_runs",
            &PyNativeCollection::compact_delta_runs,
            py::arg("max_runs") = py::none(),
            py::arg("max_input_bytes") =
                kDefaultDeltaCompactionMaxInputBytes,
            py::arg("max_input_entries") =
                kDefaultDeltaCompactionMaxInputEntries)
        .def(
            "clean_segments",
            &PyNativeCollection::clean_segments,
            py::arg("dry_run") = true,
            py::arg("max_candidate_segments") = 8,
            py::arg("max_sidecar_scan_bytes") = 1ULL << 30,
            py::arg("max_source_bytes") = 32ULL << 30,
            py::arg("max_output_bytes") = 32ULL << 30,
            py::arg("max_memory_bytes") = 2ULL << 30,
            py::arg("min_garbage_rows") = 1,
            py::arg("min_reclaimable_bytes") = 512ULL << 20,
            py::arg("min_garbage_ratio_ppm") = 300'000,
            py::arg("start_after_segment_id") = 0)
        .def("close", &PyNativeCollection::close);
}
