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
constexpr std::uint64_t kMaxIdSampleRows = 1ULL << 20;
constexpr std::uint64_t kMaxIdSampleInspectedSlots = 1ULL << 24;
constexpr std::uint64_t kDefaultDeltaCompactionMaxRuns = 8;
constexpr std::uint64_t kMaximumDeltaCompactionMaxRuns = 64;
constexpr std::uint64_t kDefaultDeltaCompactionMaxInputBytes = 8ULL << 30;
constexpr std::uint64_t kDefaultDeltaCompactionMaxInputEntries = 1'000'000;

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
    result["total_wal_bytes"] = stats.total_wal_bytes;
    result["delta_run_count"] = stats.delta_run_count;
    result["delta_entry_count"] = stats.delta_entry_count;
    result["delta_put_rows"] = stats.delta_put_rows;
    result["total_delta_bytes"] = stats.total_delta_bytes;
    result["reclaimed_wal_count"] = stats.reclaimed_wal_count;
    result["reclaimed_wal_bytes"] = stats.reclaimed_wal_bytes;
    result["compacted_delta_run_count"] =
        stats.compacted_delta_run_count;
    result["compacted_delta_input_entries"] =
        stats.compacted_delta_input_entries;
    result["compacted_delta_input_bytes"] =
        stats.compacted_delta_input_bytes;
    result["reclaimed_delta_run_count"] =
        stats.reclaimed_delta_run_count;
    result["reclaimed_delta_bytes"] = stats.reclaimed_delta_bytes;
    result["rotated"] = stats.rotated;
    result["materialized"] = stats.materialized;
    result["compacted"] = stats.compacted;
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

[[nodiscard]] py::dict native_read_stats_dict(
    const throwdb::NativeReadStats& stats) {
    py::dict result;
    result["total_seconds"] = stats.total_seconds;
    result["lookup_seconds"] = stats.lookup_seconds;
    result["physical_plan_seconds"] = stats.physical_plan_seconds;
    result["physical_backend_seconds"] = stats.physical_backend_seconds;
    result["requested_rows"] = stats.requested_rows;
    result["base_rows"] = stats.base_rows;
    result["mutation_rows"] = stats.mutation_rows;
    result["planned_ranges"] = stats.planned_ranges;
    result["planned_read_bytes"] = stats.planned_read_bytes;
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
        const std::string& dtype) {
        if (dimension == 0) {
            throw py::value_error("dimension must be greater than zero");
        }
        if (shard_size_bytes == 0) {
            throw py::value_error("shard_size_bytes must be greater than zero");
        }

        throwdb::NativeBuilderOptions options;
        options.dimension = dimension;
        options.dtype = checked_dtype(dtype);
        options.segment_size_bytes = shard_size_bytes;
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
        const py::object& calibration) {
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
        const auto native_path = filesystem_path(path);
        std::unique_ptr<throwdb::NativeCollection> collection;
        {
            py::gil_scoped_release release;
            collection = throwdb::NativeCollection::open(
                native_path, options, writable, checked_calibration);
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

    [[nodiscard]] py::dict rotate_wal() {
        const auto collection = acquire_collection();
        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->rotate_wal();
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict materialize_frozen_wals() {
        const auto collection = acquire_collection();
        throwdb::NativeMutationLogStats stats;
        {
            py::gil_scoped_release release;
            stats = collection->materialize_frozen_wals();
        }
        return mutation_log_stats_dict(stats);
    }

    [[nodiscard]] py::dict compact_delta_runs(
        const py::object& max_runs,
        const py::object& max_input_bytes,
        const py::object& max_input_entries) {
        const std::uint64_t checked_max_runs = checked_uint64_option(
            max_runs, "max_runs", false);
        if (checked_max_runs < 2
            || checked_max_runs > kMaximumDeltaCompactionMaxRuns) {
            throw py::value_error(
                "max_runs must be between 2 and "
                + std::to_string(kMaximumDeltaCompactionMaxRuns));
        }
        const std::uint64_t checked_max_input_bytes =
            checked_uint64_option(
                max_input_bytes, "max_input_bytes", false);
        const std::uint64_t checked_max_input_entries =
            checked_uint64_option(
                max_input_entries, "max_input_entries", false);

        const auto collection = acquire_collection();
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
            py::arg("dtype") = "float16")
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
            py::arg("calibration") = false)
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
            &PyNativeCollection::materialize_frozen_wals)
        .def(
            "compact_delta_runs",
            &PyNativeCollection::compact_delta_runs,
            py::arg("max_runs") = kDefaultDeltaCompactionMaxRuns,
            py::arg("max_input_bytes") =
                kDefaultDeltaCompactionMaxInputBytes,
            py::arg("max_input_entries") =
                kDefaultDeltaCompactionMaxInputEntries)
        .def("close", &PyNativeCollection::close);
}
