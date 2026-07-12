#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "throwdb/errors.hpp"
#include "throwdb/storage.hpp"

namespace py = pybind11;

namespace {

constexpr std::uint64_t kDefaultShardSizeBytes = 16ULL << 30;
constexpr std::size_t kFloat16ItemSize = 2;
constexpr std::int64_t kMaxIoWorkers = 1024;
constexpr std::int64_t kMaxIoQueueCapacity = 1 << 20;
constexpr std::int64_t kDefaultIoRangesPerTask = 8;
constexpr std::int64_t kMaxIoRangesPerTask = 1 << 20;
constexpr std::int64_t kDefaultIoUringWorkers = 1;
constexpr std::int64_t kMaxIoUringWorkers = 64;
constexpr std::int64_t kDefaultIoUringQueueDepth = 32;
constexpr std::int64_t kMaxIoUringQueueDepth = 4096;

[[nodiscard]] py::dtype float16_dtype() {
    return py::dtype::from_args(py::str("float16"));
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
    const std::uint32_t dimension) {
    if (!dtype_equal(embeddings.dtype(), float16_dtype())) {
        throw py::value_error(
            "embeddings must have native-endian dtype float16, got "
            + py::cast<std::string>(py::str(embeddings.dtype())));
    }
    require_native_endian(embeddings, "embeddings");
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

    const auto rows = static_cast<std::size_t>(expected_rows);
    const auto width = static_cast<std::size_t>(dimension);
    if (width != 0
        && rows > std::numeric_limits<std::size_t>::max()
                       / width / kFloat16ItemSize) {
        throw py::value_error("embeddings buffer is too large");
    }
    return {static_cast<const std::byte*>(embeddings.data()),
            rows * width * kFloat16ItemSize};
}

[[nodiscard]] std::span<std::byte> validate_output(
    py::array& output,
    const py::ssize_t expected_rows,
    const std::uint32_t dimension) {
    if (!dtype_equal(output.dtype(), float16_dtype())) {
        throw py::value_error(
            "output must have native-endian dtype float16, got "
            + py::cast<std::string>(py::str(output.dtype())));
    }
    require_native_endian(output, "output");
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

    const auto rows = static_cast<std::size_t>(expected_rows);
    const auto width = static_cast<std::size_t>(dimension);
    if (width != 0
        && rows > std::numeric_limits<std::size_t>::max()
                       / width / kFloat16ItemSize) {
        throw py::value_error("output buffer is too large");
    }
    return {static_cast<std::byte*>(output.mutable_data()),
            rows * width * kFloat16ItemSize};
}

class PyWriter {
public:
    static std::unique_ptr<PyWriter> create(
        const py::object& path,
        const std::uint32_t dimension,
        const std::uint64_t shard_size_bytes) {
        if (dimension == 0) {
            throw py::value_error("dimension must be greater than zero");
        }
        if (shard_size_bytes == 0) {
            throw py::value_error("shard_size_bytes must be greater than zero");
        }
        throwdb::WriterOptions options;
        options.dimension = dimension;
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
        const auto embedding_span =
            validate_embeddings(embeddings, ids.shape(0), dimension());
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
        const std::int64_t io_uring_workers) {
        throwdb::ReaderOptions options;
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
        if (options.io_backend == throwdb::IoBackend::IoUring
            && (options.io_workers != 0
                || options.io_queue_capacity != 0)) {
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

    py::array get(const py::array& ids) const {
        const auto id_span = validate_ids(ids);
        py::array output(
            float16_dtype(),
            {ids.shape(0), static_cast<py::ssize_t>(dimension())});
        auto output_span = validate_output(output, ids.shape(0), dimension());
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
        auto output_span = validate_output(output, ids.shape(0), dimension());
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

}  // namespace

PYBIND11_MODULE(_throwdb, module) {
    module.doc() = "Native bindings for the ThrowDB storage engine";
    module.attr("__version__") = "0.1.0";

    py::class_<PyWriter>(module, "Writer")
        .def_static(
            "create",
            &PyWriter::create,
            py::arg("path"),
            py::arg("dimension") = 1024,
            py::arg("shard_size_bytes") = kDefaultShardSizeBytes)
        .def("add", &PyWriter::add, py::arg("ids"), py::arg("embeddings"))
        .def("finish", &PyWriter::finish)
        .def_property_readonly("dimension", &PyWriter::dimension)
        .def_property_readonly("count", &PyWriter::count)
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
            py::arg("io_uring_workers") = kDefaultIoUringWorkers)
        .def_property_readonly("dimension", &PyStorage::dimension)
        .def_property_readonly("count", &PyStorage::count)
        .def("get", &PyStorage::get, py::arg("ids"))
        .def("get_into", &PyStorage::get_into, py::arg("ids"), py::arg("output"));
}
