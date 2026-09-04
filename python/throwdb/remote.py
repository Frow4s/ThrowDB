"""Public gRPC client for a single ThrowDB collection."""

from __future__ import annotations

import contextlib
import operator
import threading
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Iterator, Literal, overload

import grpc
import numpy as np

from ._array_validation import (
    _StreamingNumericIdSource,
    _normalize_numeric_ids,
    _require_disjoint_numpy_arrays as _reject_array_overlap,
    _validate_id_scan_output,
    _validate_output,
)
from ._tuning import normalize_reader_options
from .collection import (
    CollectionSchema,
    _normalize_embeddings,
    _validate_prefix_dimensions,
)
from .id_scan import (
    ID_LOCATION_DTYPE,
    IdScanSnapshot,
    LocatedIdBatch,
    LocatedIds,
)
from .record_scan import RecordBatch
from .v1 import throwdb_pb2, throwdb_pb2_grpc


DEFAULT_MAX_MESSAGE_BYTES: Final = 1 << 30
DEFAULT_SCAN_TIMEOUT_SECONDS: Final = 3_600.0
_DEFAULT_SCAN_BATCH_ROWS: Final = 65_536
_DEFAULT_STREAM_BATCH_PAYLOAD_BYTES: Final = 16 << 20
_SCAN_FRAME_ALLOWANCE: Final = 512
_STREAMING_FRAME_ALLOWANCE: Final = 1024
_STREAM_GET_REQUEST_ALLOWANCE: Final = 128
_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_GRPC_MAX_MESSAGE_BYTES: Final = (1 << 31) - 1
_PROTOCOL_MAJOR: Final = 1
_READ_LOG_HEADER_BYTES: Final = 4_096
_READ_LOG_RECORD_HEADER_BYTES: Final = 256
_READ_LOG_BATCH_RECORD_BYTES: Final = 384
_READ_LOG_TERMINAL_CHECKPOINT_BYTES: Final = 256

_DTYPE_TO_PROTO: Final = {
    "float16": throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
    "float32": throwdb_pb2.EMBEDDING_DTYPE_FLOAT32,
}
_PROTO_TO_DTYPE: Final = {
    value: name for name, value in _DTYPE_TO_PROTO.items()
}

CAPABILITY_WRITABLE: Final = 1 << 0
CAPABILITY_PREFIX_GET: Final = 1 << 1
CAPABILITY_SCAN_LOCATIONS: Final = 1 << 2
CAPABILITY_DURABLE_IDEMPOTENCY: Final = 1 << 3
CAPABILITY_STREAM_GET: Final = 1 << 4
CAPABILITY_SCAN_RECORDS: Final = 1 << 5


class RemoteError(Exception):
    """Base class for a typed gRPC failure returned by ThrowDB."""

    def __init__(self, code: str, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(details or code.lower().replace("_", " "))


class RemoteInvalidArgumentError(RemoteError, ValueError):
    pass


class RemoteNotFoundError(RemoteError, KeyError):
    pass


class RemoteAlreadyExistsError(RemoteError, ValueError):
    pass


class RemoteResourceExhaustedError(RemoteError, MemoryError):
    pass


class RemoteFailedPreconditionError(RemoteError, RuntimeError):
    pass


class RemoteDataLossError(RemoteError, OSError):
    pass


class RemoteInternalError(RemoteError, RuntimeError):
    pass


class RemoteUnimplementedError(RemoteError, NotImplementedError):
    pass


class RemoteUnavailableError(RemoteError, ConnectionError):
    pass


class RemoteCancelledError(RemoteError):
    pass


class RemoteDeadlineExceededError(RemoteError, TimeoutError):
    pass


class RemoteProtocolError(RemoteError, RuntimeError):
    """The server returned a successful but internally inconsistent message."""

    def __init__(self, details: str) -> None:
        super().__init__("PROTOCOL_ERROR", details)


_RPC_ERRORS: Final = {
    "INVALID_ARGUMENT": RemoteInvalidArgumentError,
    "NOT_FOUND": RemoteNotFoundError,
    "ALREADY_EXISTS": RemoteAlreadyExistsError,
    "RESOURCE_EXHAUSTED": RemoteResourceExhaustedError,
    "FAILED_PRECONDITION": RemoteFailedPreconditionError,
    "DATA_LOSS": RemoteDataLossError,
    "INTERNAL": RemoteInternalError,
    "UNIMPLEMENTED": RemoteUnimplementedError,
    "UNAVAILABLE": RemoteUnavailableError,
    "CANCELLED": RemoteCancelledError,
    "DEADLINE_EXCEEDED": RemoteDeadlineExceededError,
}

_DEFINITE_MUTATION_REJECTIONS: Final = frozenset(
    {
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "FAILED_PRECONDITION",
        "UNIMPLEMENTED",
    }
)


@dataclass(frozen=True, slots=True)
class RemoteMutationToken:
    """An operation ID bound to its original server replay domain.

    Persist both UUIDs before submitting a mutation when recovery must survive
    client restarts. Pass the complete token as ``operation_id`` on every retry.
    A stale domain is rejected by the server instead of replaying the operation
    as new work after an idempotency-journal checkpoint.
    """

    operation_id: uuid.UUID
    idempotency_domain: uuid.UUID | None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, uuid.UUID):
            raise TypeError("operation_id must be a UUID")
        if self.operation_id.int == 0:
            raise ValueError("operation_id must be nonzero")
        domain = self.idempotency_domain
        if domain is not None:
            if not isinstance(domain, uuid.UUID):
                raise TypeError("idempotency_domain must be a UUID or None")
            if domain.int == 0:
                raise ValueError("idempotency_domain must be nonzero")


class AmbiguousCommitError(RemoteError, ConnectionError):
    """A mutation may have committed although its response was not received."""

    def __init__(
        self,
        operation_id: uuid.UUID,
        operation: str,
        *,
        code: str,
        details: str,
        idempotency_domain: uuid.UUID | None = None,
    ) -> None:
        self.operation_id = operation_id
        self.operation = operation
        self.idempotency_domain = idempotency_domain
        self.retry_token = RemoteMutationToken(operation_id, idempotency_domain)
        super().__init__(
            code,
            f"remote {operation} outcome is unknown; "
            f"operation_id={operation_id}; {details or code}. Do not retry "
            "automatically unless the server advertises durable idempotency. "
            "Retain retry_token and pass it as operation_id on every retry.",
        )


class RemoteAbortedError(AmbiguousCommitError):
    """The server asked the client to retry or reconcile a mutation.

    ThrowDB reserves mutation ``ABORTED`` for rejection before the current
    request entered durable PREPARE, so that request definitely did not
    commit.  A retry must still replay the identical mutation with
    ``operation_id`` and bounded backoff; it must never allocate a new
    operation ID for the same logical mutation.  The class remains an
    :class:`AmbiguousCommitError` subclass so existing handlers keep catching
    the status, while the explicit flag below exposes the stronger wire
    guarantee.
    """

    retryable: Final = True
    same_operation_id_required: Final = True
    definitely_not_committed: Final = True

    def __init__(
        self,
        operation_id: uuid.UUID,
        operation: str,
        *,
        details: str,
        idempotency_domain: uuid.UUID | None = None,
    ) -> None:
        self.operation_id = operation_id
        self.operation = operation
        self.idempotency_domain = idempotency_domain
        self.retry_token = RemoteMutationToken(operation_id, idempotency_domain)
        RemoteError.__init__(
            self,
            "ABORTED",
            f"remote {operation} was rejected before durable PREPARE; "
            f"operation_id={operation_id}; {details or 'ABORTED'}. Retry only "
            f"the identical {operation} request with the same operation_id; "
            "never allocate a new operation_id for this logical mutation. "
            "Pass retry_token as operation_id to preserve its replay domain.",
        )


@dataclass(frozen=True, slots=True)
class RemoteReaderConfiguration:
    max_read_range_bytes: int
    max_gap_bytes: int
    io_workers: int
    io_queue_capacity: int
    io_ranges_per_task: int
    io_backend: str
    io_uring_queue_depth: int
    io_uring_workers: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "max_read_range_bytes": self.max_read_range_bytes,
            "max_gap_bytes": self.max_gap_bytes,
            "io_workers": self.io_workers,
            "io_queue_capacity": self.io_queue_capacity,
            "io_ranges_per_task": self.io_ranges_per_task,
            "io_backend": self.io_backend,
            "io_uring_queue_depth": self.io_uring_queue_depth,
            "io_uring_workers": self.io_uring_workers,
        }


@dataclass(frozen=True, slots=True)
class RemoteTuningSelection:
    mode: str
    source: str
    status: str
    reuse_level: str
    explicit_overrides: bool


@dataclass(frozen=True, slots=True)
class RemoteAvailabilityConfiguration:
    max_concurrent_calls: int
    max_concurrent_scans: int
    max_unary_inflight_bytes: int
    max_scan_inflight_bytes: int
    max_grpc_memory_bytes: int
    scan_max_duration_millis: int
    scan_idle_timeout_millis: int
    shutdown_grace_millis: int
    shutdown_hard_timeout_millis: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_concurrent_scans": self.max_concurrent_scans,
            "max_unary_inflight_bytes": self.max_unary_inflight_bytes,
            "max_scan_inflight_bytes": self.max_scan_inflight_bytes,
            "max_grpc_memory_bytes": self.max_grpc_memory_bytes,
            "scan_max_duration_millis": self.scan_max_duration_millis,
            "scan_idle_timeout_millis": self.scan_idle_timeout_millis,
            "shutdown_grace_millis": self.shutdown_grace_millis,
            "shutdown_hard_timeout_millis": self.shutdown_hard_timeout_millis,
        }


@dataclass(frozen=True, slots=True)
class RemoteAdmissionStatistics:
    active_unary_calls: int
    active_scans: int
    unary_estimated_bytes: int
    scan_estimated_bytes: int
    admitted_unary_calls: int
    rejected_unary_calls: int
    admitted_scans: int
    rejected_scans: int
    admission_closed: bool
    draining: bool
    terminal_faulted: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "active_unary_calls": self.active_unary_calls,
            "active_scans": self.active_scans,
            "unary_estimated_bytes": self.unary_estimated_bytes,
            "scan_estimated_bytes": self.scan_estimated_bytes,
            "admitted_unary_calls": self.admitted_unary_calls,
            "rejected_unary_calls": self.rejected_unary_calls,
            "admitted_scans": self.admitted_scans,
            "rejected_scans": self.rejected_scans,
            "admission_closed": self.admission_closed,
            "draining": self.draining,
            "terminal_faulted": self.terminal_faulted,
        }


@dataclass(frozen=True, slots=True)
class RemoteEngineStatistics:
    catalog_generation: int
    committed_sequence: int
    materialized_sequence: int
    active_wal_bytes: int
    active_wal_transactions: int
    active_wal_operations: int
    frozen_wal_count: int
    frozen_wal_bytes: int
    total_wal_bytes: int
    total_wal_operations: int
    overlay_entries: int
    estimated_overlay_bytes: int
    delta_run_count: int
    total_delta_run_bytes: int
    total_delta_segment_bytes: int
    idempotency_journal_bytes: int
    idempotency_completed_operations: int
    idempotency_aborted_operations: int
    idempotency_max_bytes: int
    idempotency_max_operations: int
    idempotency_generation: int
    filesystem_available_bytes: int
    mutation_disk_reserve_bytes: int
    maintenance_active: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class RemoteReadLoggingConfiguration:
    enabled: bool
    # Protocol-v1.4 compatibility aliases. A v1.5 logger always reports
    # sample_ppm=1_000_000 and max_ids_per_event=id_chunk_rows.
    sample_ppm: int
    max_ids_per_event: int
    buffer_bytes: int
    segment_bytes: int
    rotate_seconds: int
    retention_bytes: int
    retention_seconds: int
    minimum_free_bytes: int
    max_write_bytes_per_second: int
    max_retained_files: int
    id_chunk_rows: int = 0
    format_version: int = 1

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class RemoteReadLoggingStatistics:
    enabled: bool
    considered_events: int
    sampled_events: int
    enqueued_events: int
    written_events: int
    truncated_events: int
    captured_ids: int
    dropped_queue_events: int
    dropped_disk_pressure_events: int
    dropped_error_events: int
    bytes_written: int
    segments_created: int
    retained_bytes: int
    retained_segments: int
    queue_bytes: int
    peak_queue_bytes: int
    retention_deleted_bytes: int
    retention_deleted_segments: int
    writer_errors: int
    observed_ids: int
    unsampled_events: int
    dropped_busy_events: int
    dropped_full_events: int
    rotated_segments: int
    queued_events: int
    buffer_capacity_bytes: int
    last_error_number: int
    paused_for_minimum_free_space: bool
    runtime_failed: bool
    shutting_down: bool
    sampled_ids: int
    enqueued_ids: int
    written_ids: int
    dropped_busy_ids: int
    dropped_full_ids: int
    dropped_error_ids: int
    dropped_disk_pressure_ids: int
    dropped_io_budget_events: int
    dropped_io_budget_ids: int
    # Protocol-v1.5 request-lifecycle and framed-record counters. Defaults
    # preserve source compatibility for callers constructing the v1.4 model.
    observed_requests: int = 0
    begun_requests: int = 0
    completed_requests: int = 0
    complete_requests: int = 0
    incomplete_requests: int = 0
    observed_records: int = 0
    enqueued_records: int = 0
    written_records: int = 0
    observed_id_chunks: int = 0
    enqueued_id_chunks: int = 0
    written_id_chunks: int = 0
    observed_batches: int = 0
    enqueued_batches: int = 0
    written_batches: int = 0
    dropped_full_records: int = 0
    dropped_runtime_failure_records: int = 0
    dropped_minimum_free_space_records: int = 0
    dropped_io_budget_records: int = 0
    queued_records: int = 0
    active_requests: int = 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class RemoteDescription:
    format_version: int
    schema: CollectionSchema
    capabilities: int
    record_size: int
    count: int
    reader: RemoteReaderConfiguration | None
    tuning: RemoteTuningSelection | None
    availability: RemoteAvailabilityConfiguration | None
    admission_stats: RemoteAdmissionStatistics | None
    engine_stats: RemoteEngineStatistics | None
    # A durable mutation is valid only inside this server-advertised replay
    # domain. Journal checkpoint rotates it before retiring old operation
    # results, so a delayed retry fails closed instead of being applied twice.
    idempotency_domain: uuid.UUID | None
    # Added in protocol 1.4. Defaults preserve source compatibility for
    # callers that construct the previously public dataclass themselves.
    read_logging: RemoteReadLoggingConfiguration | None = None
    read_logging_stats: RemoteReadLoggingStatistics | None = None

    @property
    def writable(self) -> bool:
        return bool(self.capabilities & CAPABILITY_WRITABLE)

    @property
    def prefix_get(self) -> bool:
        return bool(self.capabilities & CAPABILITY_PREFIX_GET)

    @property
    def scan_locations(self) -> bool:
        return bool(self.capabilities & CAPABILITY_SCAN_LOCATIONS)

    @property
    def durable_idempotency(self) -> bool:
        return bool(self.capabilities & CAPABILITY_DURABLE_IDEMPOTENCY)

    @property
    def stream_get(self) -> bool:
        return bool(self.capabilities & CAPABILITY_STREAM_GET)

    @property
    def scan_records(self) -> bool:
        return bool(self.capabilities & CAPABILITY_SCAN_RECORDS)


@dataclass(frozen=True, slots=True)
class RemoteEmbeddingBatch:
    """One bounded response from a snapshot-pinned remote lookup stream."""

    sequence: int
    input_offset: int
    ids: np.ndarray
    embeddings: np.ndarray
    snapshot: IdScanSnapshot


def _validate_positive_u32(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if not 1 <= result <= _UINT32_MAX:
        raise ValueError(f"{name} must be between 1 and {_UINT32_MAX}")
    return result


def _validate_optional_positive_u64(value: object | None, *, name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer or None") from error
    if not 1 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must be between 1 and {_UINT64_MAX}")
    return result


def _bounded_stream_parameters(
    *,
    requested_rows: int,
    requested_payload_bytes: int,
    dense_row_bytes: int,
    max_frame_bytes: int,
    include_id_request: bool,
    operation: str,
) -> tuple[int, int]:
    payload_capacity = max_frame_bytes - _STREAMING_FRAME_ALLOWANCE
    if dense_row_bytes <= 0 or payload_capacity < dense_row_bytes:
        raise ValueError(
            f"max_frame_bytes cannot fit one {operation} dense row"
        )
    payload_bytes = min(
        requested_payload_bytes or payload_capacity,
        payload_capacity,
    )
    rows = min(requested_rows, payload_bytes // dense_row_bytes)
    if include_id_request:
        request_capacity = max_frame_bytes - _STREAM_GET_REQUEST_ALLOWANCE
        if request_capacity < 8:
            raise ValueError(
                "max_frame_bytes cannot fit one StreamGet ID request"
            )
        rows = min(rows, request_capacity // 8)
    if rows < 1:
        raise ValueError(
            f"streaming limits cannot fit one {operation} row"
        )
    return int(rows), int(payload_bytes)


@contextlib.contextmanager
def _exclusive_consumer(
    lock: threading.RLock,
    *,
    operation: str,
) -> Iterator[None]:
    if not lock.acquire(blocking=False):
        raise RuntimeError(
            f"{operation} supports exactly one active consumer"
        )
    try:
        yield
    finally:
        lock.release()


def _validate_max_message_bytes(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_frame_bytes must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("max_frame_bytes must be an integer") from error
    if not 1 <= result <= _GRPC_MAX_MESSAGE_BYTES:
        raise ValueError(
            f"max_frame_bytes must be between 1 and {_GRPC_MAX_MESSAGE_BYTES}"
        )
    return result


def _validate_timeout(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _validate_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)


def _operation_id(value: uuid.UUID | str | bytes | None) -> uuid.UUID:
    if value is None:
        return uuid.uuid4()
    if isinstance(value, uuid.UUID):
        result = value
    elif isinstance(value, str):
        try:
            result = uuid.UUID(value)
        except ValueError as error:
            raise ValueError("operation_id must be a UUID string") from error
    elif isinstance(value, bytes):
        if len(value) != 16:
            raise ValueError("operation_id bytes must contain exactly 16 bytes")
        result = uuid.UUID(bytes=value)
    else:
        raise TypeError("operation_id must be UUID, string, 16 bytes, or None")
    if result.int == 0:
        raise ValueError("operation_id must be nonzero")
    return result


def _proto_operation_id(value: uuid.UUID) -> throwdb_pb2.OperationId:
    return throwdb_pb2.OperationId(
        high=value.int >> 64,
        low=value.int & 0xFFFF_FFFF_FFFF_FFFF,
    )


def _operation_id_from_proto(
    value: throwdb_pb2.OperationId,
    *,
    name: str,
) -> uuid.UUID:
    encoded = (int(value.high) << 64) | int(value.low)
    if encoded == 0:
        raise RemoteProtocolError(f"server returned an empty {name}")
    return uuid.UUID(int=encoded)


def _native_little_endian(array: np.ndarray, dtype: str) -> np.ndarray:
    expected = np.dtype(dtype).newbyteorder("<")
    if array.dtype == expected and array.flags.c_contiguous and array.flags.aligned:
        return array
    return np.require(array, dtype=expected, requirements=["C", "A"])


def _proto_ids(ids: np.ndarray) -> throwdb_pb2.IdBatch:
    wire = _native_little_endian(ids, "<u8")
    return throwdb_pb2.IdBatch(
        uint64_ids=throwdb_pb2.UInt64Ids(
            count=ids.size,
            values_le=wire.tobytes(order="C"),
        )
    )


def _remote_read_snapshot(
    snapshot: throwdb_pb2.ReadSnapshot,
    *,
    operation: str,
) -> IdScanSnapshot:
    collection = snapshot.collection_id
    token = snapshot.token
    if collection.high == 0 and collection.low == 0:
        raise RemoteProtocolError(
            f"{operation} returned a zero collection ID"
        )
    if token.high == 0 and token.low == 0:
        raise RemoteProtocolError(f"{operation} returned a zero snapshot token")
    return IdScanSnapshot(
        collection_id=f"{collection.high:016x}{collection.low:016x}",
        catalog_generation=snapshot.catalog_generation,
        committed_sequence=snapshot.committed_sequence,
        count=snapshot.live_count,
        token=f"{token.high:016x}{token.low:016x}",
    )


def _decode_embedding_batch(
    embeddings: throwdb_pb2.EmbeddingBatch,
    *,
    operation: str,
    dtype: str,
    dimension: int,
    row_count: int,
) -> np.ndarray:
    if (
        embeddings.dtype != _DTYPE_TO_PROTO[dtype]
        or embeddings.dimension != dimension
        or embeddings.row_count != row_count
    ):
        raise RemoteProtocolError(
            f"{operation} response metadata does not match its request"
        )
    expected_bytes = row_count * dimension * np.dtype(dtype).itemsize
    if len(embeddings.values_le) != expected_bytes:
        raise RemoteProtocolError(
            f"{operation} response byte length does not match its metadata"
        )
    view = np.frombuffer(
        embeddings.values_le,
        dtype=np.dtype(dtype).newbyteorder("<"),
    ).reshape((row_count, dimension))
    return np.array(view, dtype=np.dtype(dtype), copy=True, order="C")


def _rpc_code_name(error: grpc.RpcError) -> str:
    code = error.code()
    return getattr(code, "name", str(code).rsplit(".", 1)[-1])


def _rpc_details(error: grpc.RpcError) -> str:
    details = error.details()
    return details if isinstance(details, str) else str(details or "")


def _typed_rpc_error(error: grpc.RpcError) -> RemoteError:
    code = _rpc_code_name(error)
    details = _rpc_details(error)
    error_type = _RPC_ERRORS.get(code, RemoteError)
    return error_type(code, details)


class RemoteIdScan(Iterator[np.ndarray | LocatedIdBatch]):
    """One gRPC server-streaming scan pinned to a logical collection view."""

    def __init__(
        self,
        stream: grpc.Call,
        *,
        batch_rows: int,
        with_locations: bool,
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)  # type: ignore[arg-type]
        self._batch_rows = batch_rows
        self._with_locations = with_locations
        self._snapshot: IdScanSnapshot | None = None
        self._received_rows = 0
        self._pending: np.ndarray | LocatedIdBatch | None = None
        self._pending_offset = 0
        self._finished = False
        self._closed = False
        self._failed = False
        self._consumer_lock = threading.RLock()
        try:
            self._read_one()
        except BaseException:
            self._failed = True
            self.close()
            raise

    @property
    def snapshot(self) -> IdScanSnapshot:
        if self._snapshot is None:
            raise RuntimeError("ID scan did not receive snapshot metadata")
        return self._snapshot

    @property
    def snapshot_count(self) -> int:
        return self.snapshot.count

    @property
    def done(self) -> bool:
        return self._finished and self._pending is None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def _read_one(self) -> None:
        try:
            response = next(self._iterator)
        except StopIteration:
            if self._snapshot is None:
                raise RemoteProtocolError(
                    "ScanIds ended without snapshot metadata"
                )
            if self._received_rows != self._snapshot.count:
                raise RemoteProtocolError(
                    "ScanIds ended before its advertised snapshot count"
                )
            self._finished = True
            return
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error

        snapshot = response.snapshot
        collection = snapshot.collection_id
        token = snapshot.token
        received_snapshot = IdScanSnapshot(
            collection_id=f"{collection.high:016x}{collection.low:016x}",
            catalog_generation=snapshot.catalog_generation,
            committed_sequence=snapshot.committed_sequence,
            count=snapshot.count,
            token=f"{token.high:016x}{token.low:016x}",
        )
        if collection.high == 0 and collection.low == 0:
            raise RemoteProtocolError("ScanIds returned a zero collection ID")
        if token.high == 0 and token.low == 0:
            raise RemoteProtocolError("ScanIds returned a zero scan token")
        if self._snapshot is None:
            self._snapshot = received_snapshot
        elif received_snapshot != self._snapshot:
            raise RemoteProtocolError("ScanIds snapshot changed mid-stream")
        if response.emitted_offset != self._received_rows:
            raise RemoteProtocolError("ScanIds emitted offsets are not contiguous")

        if response.ids.WhichOneof("encoding") != "uint64_ids":
            raise RemoteProtocolError("ScanIds returned a non-uint64 ID batch")
        encoded = response.ids.uint64_ids
        count = encoded.count
        if count > self._batch_rows:
            raise RemoteProtocolError("ScanIds chunk exceeds requested batch_rows")
        if len(encoded.values_le) != count * 8:
            raise RemoteProtocolError("ScanIds ID byte length does not match count")
        if self._received_rows + count > self.snapshot.count:
            raise RemoteProtocolError("ScanIds exceeds its advertised snapshot count")
        if count == 0 and self.snapshot.count != 0:
            raise RemoteProtocolError("non-empty ScanIds snapshot emitted an empty chunk")
        ids = np.frombuffer(encoded.values_le, dtype="<u8").astype(
            np.uint64, copy=True
        )

        if self._with_locations:
            if (
                response.location_encoding
                != throwdb_pb2.LOCATION_ENCODING_SEGMENT_ROW_U32_LE
                or len(response.locations_le) != count * ID_LOCATION_DTYPE.itemsize
            ):
                raise RemoteProtocolError(
                    "ScanIds locations do not match the requested encoding"
                )
            locations = np.frombuffer(
                response.locations_le, dtype=ID_LOCATION_DTYPE
            ).copy()
            self._pending = (
                LocatedIdBatch(
                    ids=ids,
                    locations=locations,
                    emitted_offset=response.emitted_offset,
                    snapshot=self.snapshot,
                )
                if count
                else None
            )
        else:
            if (
                response.location_encoding
                != throwdb_pb2.LOCATION_ENCODING_UNSPECIFIED
                or response.locations_le
            ):
                raise RemoteProtocolError(
                    "ScanIds unexpectedly returned physical locations"
                )
            self._pending = ids if count else None
        self._received_rows += count

    def __iter__(self) -> RemoteIdScan:
        return self

    def _take_pending(self) -> np.ndarray | LocatedIdBatch:
        pending = self._pending
        if pending is None:
            raise RuntimeError("ID scan has no pending batch")
        begin = self._pending_offset
        self._pending = None
        self._pending_offset = 0
        if isinstance(pending, LocatedIdBatch):
            if begin == 0:
                return pending
            return LocatedIdBatch(
                ids=pending.ids[begin:],
                locations=pending.locations[begin:],
                emitted_offset=pending.emitted_offset + begin,
                snapshot=pending.snapshot,
            )
        return pending if begin == 0 else pending[begin:]

    def __next__(self) -> np.ndarray | LocatedIdBatch:
        with _exclusive_consumer(
            self._consumer_lock,
            operation="ScanIds",
        ):
            return self._next_unlocked()

    def _next_unlocked(self) -> np.ndarray | LocatedIdBatch:
        if self._closed:
            raise StopIteration
        if self._pending is not None:
            return self._take_pending()
        if self._finished:
            raise StopIteration
        try:
            self._read_one()
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise
        if self._pending is None:
            raise StopIteration
        return self._take_pending()

    def read_into(
        self,
        ids: object,
        locations: object | None = None,
    ) -> int:
        with _exclusive_consumer(
            self._consumer_lock,
            operation="ScanIds",
        ):
            return self._read_into_unlocked(ids, locations)

    def _read_into_unlocked(
        self,
        ids: object,
        locations: object | None,
    ) -> int:
        if self._closed:
            raise RuntimeError("ID scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
            operation="scan_ids",
        )
        try:
            if self._pending is None:
                if self._finished:
                    return 0
                self._read_one()
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise
        pending = self._pending
        if pending is None:
            return 0
        source_ids = pending.ids if isinstance(pending, LocatedIdBatch) else pending
        available = source_ids.size - self._pending_offset
        count = min(output_ids.size, available)
        begin = self._pending_offset
        end = begin + count
        output_ids[:count] = source_ids[begin:end]
        if isinstance(pending, LocatedIdBatch):
            assert output_locations is not None
            output_locations[:count] = pending.locations[begin:end]
        self._pending_offset = end
        if end == source_ids.size:
            self._pending = None
            self._pending_offset = 0
        return count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._pending_offset = 0
        self._stream.cancel()

    def __enter__(self) -> RemoteIdScan:
        if self._closed:
            raise RuntimeError("ID scan is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()


class _IndexedIds(_StreamingNumericIdSource):
    """Remote naming adapter around the shared bounded ID source."""

    @property
    def count(self) -> int:
        return len(self)

    def normalized_slice(self, begin: int, end: int) -> np.ndarray:
        return self.batch(begin, end)

    def reject_output_overlap(self, output: np.ndarray) -> None:
        if self.output_overlaps_unread(output, begin=0):
            raise ValueError("StreamGet buffers ids source and output must not overlap")


class _StreamGetRequestIterator(Iterator[throwdb_pb2.StreamGetRequest]):
    """Application-window-one producer consumed by gRPC's request thread."""

    def __init__(
        self,
        ids: _IndexedIds,
        *,
        prefix_dimensions: int,
        requested_batch_rows: int,
        requested_batch_payload_bytes: int,
    ) -> None:
        self._ids = ids
        self._open = throwdb_pb2.StreamGetRequest(
            open=throwdb_pb2.StreamGetOpen(
                prefix_dimensions=prefix_dimensions,
                max_batch_rows=requested_batch_rows,
                max_batch_payload_bytes=requested_batch_payload_bytes,
            )
        )
        self._condition = threading.Condition()
        self._open_emitted = False
        self._configured = False
        self._closed = False
        self._credit = 0
        self._max_batch_rows = 0
        self._offset = 0
        self._sequence = 0
        self._pending: dict[int, np.ndarray] = {}
        self._failure: BaseException | None = None

    @property
    def count(self) -> int:
        return self._ids.count

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    @property
    def pending(self) -> bool:
        with self._condition:
            return bool(self._pending)

    def reject_output_overlap(self, output: np.ndarray) -> None:
        self._ids.reject_output_overlap(output)

    def configure(self, *, max_batch_rows: int, request_window: int) -> None:
        with self._condition:
            if self._configured:
                raise RemoteProtocolError("StreamGet sent more than one open response")
            self._max_batch_rows = max_batch_rows
            self._credit = request_window
            self._configured = True
            self._condition.notify_all()

    def pending_ids(self, sequence: int) -> np.ndarray:
        with self._condition:
            ids = self._pending.get(sequence)
            if ids is None:
                raise RemoteProtocolError(
                    "StreamGet response has no matching input batch"
                )
            return ids

    def acknowledge(self, sequence: int) -> None:
        with self._condition:
            if self._pending.pop(sequence, None) is None:
                raise RemoteProtocolError(
                    "StreamGet acknowledged an unknown input batch"
                )
            self._credit += 1
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def release_pending(self) -> None:
        with self._condition:
            # A cancelled call must not retain the last normalized ID batch.
            # The consumer lock guarantees that no received response is still
            # resolving its matching IDs while this map is cleared.
            self._pending.clear()

    def close(self) -> None:
        self.cancel()
        self.release_pending()

    def __iter__(self) -> _StreamGetRequestIterator:
        return self

    def __next__(self) -> throwdb_pb2.StreamGetRequest:
        with self._condition:
            if not self._open_emitted:
                self._open_emitted = True
                return self._open
            self._condition.wait_for(
                lambda: self._closed
                or (self._configured and self._credit > 0)
            )
            if self._closed:
                raise StopIteration
            if self._offset == self._ids.count:
                self._closed = True
                raise StopIteration
            begin = self._offset
            end = min(self._ids.count, begin + self._max_batch_rows)
            sequence = self._sequence
            # Reserve the only application credit, then release the condition
            # before calling arbitrary Sequence code, NumPy, or protobuf.
            # close() must remain prompt even if a user Sequence blocks.
            self._credit -= 1
        try:
            ids = self._ids.normalized_slice(begin, end)
            request = throwdb_pb2.StreamGetRequest(
                batch=throwdb_pb2.StreamGetInputBatch(
                    sequence=sequence,
                    ids=_proto_ids(ids),
                )
            )
        except BaseException as error:
            with self._condition:
                if self._closed:
                    raise StopIteration from error
                self._failure = error
                self._closed = True
                self._condition.notify_all()
            raise
        with self._condition:
            if self._closed:
                raise StopIteration
            if ids.size == 0:
                self._failure = RuntimeError(
                    "stream_get produced an unexpected empty ID batch"
                )
                self._closed = True
                raise self._failure
            self._pending[sequence] = ids
            self._offset = end
            self._sequence += 1
            return request


class RemoteEmbeddingStream(Iterator[RemoteEmbeddingBatch]):
    """Snapshot-pinned, bounded remote lookup over an indexed ID source."""

    def __init__(
        self,
        method: object,
        ids: _IndexedIds,
        *,
        dtype: str,
        dimension: int,
        prefix_dimensions: int,
        requested_batch_rows: int,
        requested_batch_payload_bytes: int,
        timeout: float | None,
    ) -> None:
        self._dtype = dtype
        self._dimension = dimension
        self._prefix_dimensions = prefix_dimensions
        self._requested_batch_rows = requested_batch_rows
        self._requested_batch_payload_bytes = requested_batch_payload_bytes
        self._producer = _StreamGetRequestIterator(
            ids,
            prefix_dimensions=prefix_dimensions,
            requested_batch_rows=requested_batch_rows,
            requested_batch_payload_bytes=requested_batch_payload_bytes,
        )
        self._stream = method(  # type: ignore[operator]
            self._producer,
            timeout=timeout,
            wait_for_ready=True,
        )
        self._iterator = iter(self._stream)
        self._snapshot: IdScanSnapshot | None = None
        self._max_batch_rows = 0
        self._max_batch_payload_bytes = 0
        self._next_sequence = 0
        self._rows_completed = 0
        self._done = False
        self._closed = False
        self._failed = False
        self._consumer_lock = threading.RLock()
        try:
            self._read_opened()
        except BaseException:
            self._failed = True
            self.close()
            raise

    @property
    def snapshot(self) -> IdScanSnapshot:
        if self._snapshot is None:
            raise RuntimeError("StreamGet did not receive snapshot metadata")
        return self._snapshot

    @property
    def rows_completed(self) -> int:
        return self._rows_completed

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    @property
    def max_batch_payload_bytes(self) -> int:
        return self._max_batch_payload_bytes

    @property
    def done(self) -> bool:
        return self._done

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def _next_response(self) -> throwdb_pb2.StreamGetResponse:
        try:
            return next(self._iterator)
        except grpc.RpcError as error:
            producer_failure = self._producer.failure
            if producer_failure is not None:
                raise producer_failure
            raise _typed_rpc_error(error) from error

    def _validate_snapshot(
        self,
        response: throwdb_pb2.StreamGetResponse,
    ) -> None:
        received = _remote_read_snapshot(response.snapshot, operation="StreamGet")
        if self._snapshot is None:
            self._snapshot = received
        elif received != self._snapshot:
            raise RemoteProtocolError("StreamGet snapshot changed mid-stream")

    def _read_opened(self) -> None:
        try:
            response = self._next_response()
        except StopIteration as error:
            raise RemoteProtocolError(
                "StreamGet ended before its open response"
            ) from error
        self._validate_snapshot(response)
        if response.WhichOneof("payload") != "opened":
            raise RemoteProtocolError(
                "StreamGet first response is not an open response"
            )
        opened = response.opened
        if (
            opened.dtype != _DTYPE_TO_PROTO[self._dtype]
            or opened.dimension != self._dimension
        ):
            raise RemoteProtocolError(
                "StreamGet open response does not match requested schema"
            )
        if opened.max_batch_rows == 0 or opened.max_batch_payload_bytes == 0:
            raise RemoteProtocolError("StreamGet negotiated an empty batch budget")
        if (
            self._requested_batch_rows != 0
            and opened.max_batch_rows > self._requested_batch_rows
        ):
            raise RemoteProtocolError(
                "StreamGet exceeded the requested batch-row ceiling"
            )
        if (
            self._requested_batch_payload_bytes != 0
            and opened.max_batch_payload_bytes
                > self._requested_batch_payload_bytes
        ):
            raise RemoteProtocolError(
                "StreamGet exceeded the requested batch-byte ceiling"
            )
        if opened.request_window != 1:
            raise RemoteProtocolError(
                "StreamGet protocol v1 requires request_window=1"
            )
        self._max_batch_rows = opened.max_batch_rows
        self._max_batch_payload_bytes = opened.max_batch_payload_bytes
        self._producer.configure(
            max_batch_rows=self._max_batch_rows,
            request_window=opened.request_window,
        )

    def __iter__(self) -> RemoteEmbeddingStream:
        return self

    def __next__(self) -> RemoteEmbeddingBatch:
        with _exclusive_consumer(
            self._consumer_lock,
            operation="StreamGet",
        ):
            return self._next_unlocked()

    def _next_unlocked(self) -> RemoteEmbeddingBatch:
        if self._closed or self._done:
            raise StopIteration
        try:
            response = self._next_response()
        except StopIteration:
            producer_failure = self._producer.failure
            if producer_failure is not None:
                self._failed = True
                self.close()
                raise producer_failure
            if self._producer.pending:
                self._failed = True
                self.close()
                raise RemoteProtocolError(
                    "StreamGet ended before returning every submitted batch"
                )
            if self._rows_completed != self._producer.count:
                self._failed = True
                self.close()
                raise RemoteProtocolError(
                    "StreamGet ended before every requested ID was submitted"
                )
            self._done = True
            self._producer.close()
            raise
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise
        try:
            self._validate_snapshot(response)
            if response.WhichOneof("payload") != "batch":
                raise RemoteProtocolError(
                    "StreamGet returned an unexpected second open response"
                )
            batch = response.batch
            if batch.sequence != self._next_sequence:
                raise RemoteProtocolError(
                    "StreamGet response sequence is not contiguous"
                )
            if batch.input_offset != self._rows_completed:
                raise RemoteProtocolError(
                    "StreamGet response offsets are not contiguous"
                )
            ids = self._producer.pending_ids(batch.sequence)
            row_count = ids.size
            if row_count == 0 or row_count > self._max_batch_rows:
                raise RemoteProtocolError(
                    "StreamGet response exceeds its negotiated row budget"
                )
            if len(batch.embeddings.values_le) > self._max_batch_payload_bytes:
                raise RemoteProtocolError(
                    "StreamGet response exceeds its negotiated byte budget"
                )
            values = _decode_embedding_batch(
                batch.embeddings,
                operation="StreamGet",
                dtype=self._dtype,
                dimension=self._dimension,
                row_count=row_count,
            )
            result = RemoteEmbeddingBatch(
                sequence=batch.sequence,
                input_offset=batch.input_offset,
                ids=ids,
                embeddings=values,
                snapshot=self.snapshot,
            )
            self._rows_completed += row_count
            self._next_sequence += 1
            self._producer.acknowledge(batch.sequence)
            return result
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise

    def get_into(self, output: object) -> None:
        """Incrementally populate output; earlier complete batches survive errors."""

        with _exclusive_consumer(
            self._consumer_lock,
            operation="StreamGet",
        ):
            if self._rows_completed != 0:
                raise RuntimeError(
                    "StreamGet get_into must start before iteration"
                )
            destination = _validate_output(
                output,
                expected_rows=self._producer.count,
                expected_columns=self._dimension,
                dtype=self._dtype,
            )
            self._producer.reject_output_overlap(destination)
            for batch in self:
                begin = batch.input_offset
                end = begin + batch.embeddings.shape[0]
                np.copyto(
                    destination[begin:end],
                    batch.embeddings,
                    casting="no",
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel transport/producer first so a consumer blocked in gRPC or a
        # request source is released promptly. Keep pending IDs alive until
        # the exclusive consumer leaves response decoding: close() can race
        # after _next_response() but before pending_ids().
        self._stream.cancel()
        self._producer.cancel()
        with self._consumer_lock:
            self._producer.release_pending()

    def __enter__(self) -> RemoteEmbeddingStream:
        if self._closed:
            raise RuntimeError("StreamGet is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()


class RemoteRecordScan(Iterator[RecordBatch]):
    """Bounded physical-order record scan pinned to one remote snapshot."""

    def __init__(
        self,
        stream: grpc.Call,
        *,
        requested_batch_rows: int,
        requested_batch_payload_bytes: int,
        dtype: str,
        dimension: int,
        with_locations: bool,
    ) -> None:
        self._stream = stream
        self._iterator = iter(stream)  # type: ignore[arg-type]
        self._requested_batch_rows = requested_batch_rows
        self._requested_batch_payload_bytes = requested_batch_payload_bytes
        self._dtype = dtype
        self._dimension = dimension
        self._with_locations = with_locations
        self._snapshot: IdScanSnapshot | None = None
        self._max_batch_rows = 0
        self._max_batch_payload_bytes = 0
        self._received_rows = 0
        self._delivered_rows = 0
        self._pending: RecordBatch | None = None
        self._pending_offset = 0
        self._done = False
        self._closed = False
        self._failed = False
        self._consumer_lock = threading.RLock()
        try:
            self._read_one()
        except BaseException:
            self._failed = True
            self.close()
            raise

    @property
    def snapshot(self) -> IdScanSnapshot:
        if self._snapshot is None:
            raise RuntimeError("ScanRecords did not receive snapshot metadata")
        return self._snapshot

    @property
    def snapshot_count(self) -> int:
        return self.snapshot.count

    @property
    def rows_completed(self) -> int:
        return self._delivered_rows

    @property
    def max_batch_rows(self) -> int:
        """Server-negotiated maximum rows in one response batch."""

        return self._max_batch_rows

    @property
    def max_batch_payload_bytes(self) -> int:
        """Server-negotiated maximum dense payload bytes per response."""

        return self._max_batch_payload_bytes

    @property
    def done(self) -> bool:
        return self._done

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def _read_one(self) -> None:
        try:
            response = next(self._iterator)
        except StopIteration:
            if self._snapshot is None:
                raise RemoteProtocolError(
                    "ScanRecords ended without snapshot metadata"
                )
            if self._received_rows != self.snapshot.count:
                raise RemoteProtocolError(
                    "ScanRecords ended before its advertised snapshot count"
                )
            self._done = True
            return
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error

        received_snapshot = _remote_read_snapshot(
            response.snapshot,
            operation="ScanRecords",
        )
        if self._snapshot is None:
            self._snapshot = received_snapshot
        elif received_snapshot != self._snapshot:
            raise RemoteProtocolError("ScanRecords snapshot changed mid-stream")
        if response.emitted_offset != self._received_rows:
            raise RemoteProtocolError("ScanRecords offsets are not contiguous")
        if response.max_batch_rows == 0 or response.max_batch_payload_bytes == 0:
            raise RemoteProtocolError("ScanRecords returned an empty batch budget")
        if response.max_batch_rows > self._requested_batch_rows:
            raise RemoteProtocolError(
                "ScanRecords exceeded the requested batch-row ceiling"
            )
        if (
            self._requested_batch_payload_bytes != 0
            and response.max_batch_payload_bytes
                > self._requested_batch_payload_bytes
        ):
            raise RemoteProtocolError(
                "ScanRecords exceeded the requested batch-byte ceiling"
            )
        if self._max_batch_rows == 0:
            self._max_batch_rows = response.max_batch_rows
            self._max_batch_payload_bytes = response.max_batch_payload_bytes
        elif (
            response.max_batch_rows != self._max_batch_rows
            or response.max_batch_payload_bytes
                != self._max_batch_payload_bytes
        ):
            raise RemoteProtocolError(
                "ScanRecords batch negotiation changed mid-stream"
            )

        if response.ids.WhichOneof("encoding") != "uint64_ids":
            raise RemoteProtocolError("ScanRecords returned non-uint64 IDs")
        encoded = response.ids.uint64_ids
        count = int(encoded.count)
        if len(encoded.values_le) != count * 8:
            raise RemoteProtocolError(
                "ScanRecords ID byte length does not match row count"
            )
        if count > self._max_batch_rows:
            raise RemoteProtocolError(
                "ScanRecords response exceeds its negotiated row budget"
            )
        if self._received_rows + count > self.snapshot.count:
            raise RemoteProtocolError(
                "ScanRecords exceeded its advertised snapshot count"
            )
        if count == 0 and self.snapshot.count != 0:
            raise RemoteProtocolError(
                "non-empty ScanRecords snapshot emitted an empty batch"
            )
        payload_bytes = (
            len(encoded.values_le)
            + len(response.embeddings.values_le)
            + len(response.locations_le)
        )
        if payload_bytes > self._max_batch_payload_bytes:
            raise RemoteProtocolError(
                "ScanRecords response exceeds its negotiated byte budget"
            )
        ids = np.frombuffer(encoded.values_le, dtype="<u8").astype(
            np.uint64,
            copy=True,
        )
        embeddings = _decode_embedding_batch(
            response.embeddings,
            operation="ScanRecords",
            dtype=self._dtype,
            dimension=self._dimension,
            row_count=count,
        )
        locations: np.ndarray | None = None
        if self._with_locations:
            if (
                response.location_encoding
                    != throwdb_pb2.LOCATION_ENCODING_SEGMENT_ROW_U32_LE
                or len(response.locations_le)
                    != count * ID_LOCATION_DTYPE.itemsize
            ):
                raise RemoteProtocolError(
                    "ScanRecords locations do not match the requested encoding"
                )
            locations = np.frombuffer(
                response.locations_le,
                dtype=ID_LOCATION_DTYPE,
            ).copy()
        elif (
            response.location_encoding
                != throwdb_pb2.LOCATION_ENCODING_UNSPECIFIED
            or response.locations_le
        ):
            raise RemoteProtocolError(
                "ScanRecords unexpectedly returned physical locations"
            )
        self._pending = (
            RecordBatch(
                ids=ids,
                embeddings=embeddings,
                locations=locations,
                snapshot=self.snapshot,
                emitted_offset=response.emitted_offset,
            )
            if count
            else None
        )
        self._received_rows += count

    def __iter__(self) -> RemoteRecordScan:
        return self

    def _take_pending(self) -> RecordBatch:
        pending = self._pending
        if pending is None:
            raise RuntimeError("record scan has no pending batch")
        begin = self._pending_offset
        self._pending = None
        self._pending_offset = 0
        if begin == 0:
            self._delivered_rows += pending.ids.size
            return pending
        result = RecordBatch(
            ids=pending.ids[begin:],
            embeddings=pending.embeddings[begin:],
            locations=(
                pending.locations[begin:]
                if pending.locations is not None
                else None
            ),
            snapshot=pending.snapshot,
            emitted_offset=pending.emitted_offset + begin,
        )
        self._delivered_rows += result.ids.size
        return result

    def __next__(self) -> RecordBatch:
        with _exclusive_consumer(
            self._consumer_lock,
            operation="ScanRecords",
        ):
            return self._next_unlocked()

    def _next_unlocked(self) -> RecordBatch:
        if self._closed:
            raise StopIteration
        if self._pending is not None:
            return self._take_pending()
        if self._done:
            raise StopIteration
        try:
            self._read_one()
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise
        if self._pending is None:
            raise StopIteration
        return self._take_pending()

    def read_into(
        self,
        ids: object,
        embeddings: object,
        locations: object | None = None,
    ) -> int:
        """Copy at most one bounded transport batch into reusable buffers."""

        with _exclusive_consumer(
            self._consumer_lock,
            operation="ScanRecords",
        ):
            return self._read_into_unlocked(ids, embeddings, locations)

    def _read_into_unlocked(
        self,
        ids: object,
        embeddings: object,
        locations: object | None,
    ) -> int:

        if self._closed:
            raise RuntimeError("record scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
            operation="scan_records",
        )
        output_embeddings = _validate_output(
            embeddings,
            expected_rows=int(output_ids.size),
            expected_columns=self._dimension,
            dtype=self._dtype,
        )
        output_arrays = [
            ("ids", output_ids),
            ("embeddings", output_embeddings),
        ]
        if output_locations is not None:
            output_arrays.append(("locations", output_locations))
        _reject_array_overlap(output_arrays, operation="ScanRecords")
        try:
            if self._pending is None:
                if self._done:
                    return 0
                self._read_one()
            pending = self._pending
            if pending is None:
                return 0
            available = pending.ids.size - self._pending_offset
            count = min(int(output_ids.size), available)
            begin = self._pending_offset
            end = begin + count
            np.copyto(output_ids[:count], pending.ids[begin:end], casting="no")
            np.copyto(
                output_embeddings[:count],
                pending.embeddings[begin:end],
                casting="no",
            )
            if pending.locations is not None:
                assert output_locations is not None
                np.copyto(
                    output_locations[:count],
                    pending.locations[begin:end],
                    casting="no",
                )
            self._pending_offset = end
            if end == pending.ids.size:
                self._pending = None
                self._pending_offset = 0
            self._delivered_rows += count
            return count
        except BaseException:
            if not self._closed:
                self._failed = True
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.cancel()
        self._pending = None
        self._pending_offset = 0

    def __enter__(self) -> RemoteRecordScan:
        if self._closed:
            raise RuntimeError("ScanRecords is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()


class RemoteCollection:
    """Thread-safe synchronous gRPC client for one served collection."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float | None = 30.0,
        scan_timeout: float | None = DEFAULT_SCAN_TIMEOUT_SECONDS,
        max_frame_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        credentials: grpc.ChannelCredentials | None = None,
        use_http_proxy: bool = False,
    ) -> None:
        if not isinstance(endpoint, str):
            raise TypeError("endpoint must be a string")
        if not endpoint.strip():
            raise ValueError("endpoint cannot be empty")
        self._timeout = _validate_timeout(timeout, name="timeout")
        # Full scans can run for many minutes, but a forgotten stream must not
        # retain server resources forever. Passing None explicitly disables the
        # client deadline; the server's independent lifetime/idle limits remain.
        self._scan_timeout = _validate_timeout(scan_timeout, name="scan_timeout")
        maximum_message = _validate_max_message_bytes(max_frame_bytes)
        self._max_frame_bytes = maximum_message
        allow_http_proxy = _validate_bool(
            use_http_proxy,
            name="use_http_proxy",
        )
        options = (
            ("grpc.max_receive_message_length", maximum_message),
            ("grpc.max_send_message_length", maximum_message),
            # No automatic mutation retry policy exists before the durable
            # idempotency capability is advertised.
            ("grpc.enable_retries", 0),
            # Ambient HTTP(S)_PROXY variables must not silently redirect
            # internal embedding traffic. Opting into a proxy is explicit.
            ("grpc.enable_http_proxy", int(allow_http_proxy)),
        )
        if credentials is None:
            channel = grpc.insecure_channel(endpoint, options=options)
        elif isinstance(credentials, grpc.ChannelCredentials):
            channel = grpc.secure_channel(endpoint, credentials, options=options)
        else:
            raise TypeError("credentials must be grpc.ChannelCredentials or None")
        self._endpoint = endpoint
        self._channel = channel
        self._stub = throwdb_pb2_grpc.ThrowDBStub(channel)
        self._state = "open"
        self._protocol_minor: int | None = None
        try:
            health = self.health()
            if not health["serving"]:
                raise RemoteUnavailableError(
                    "UNAVAILABLE", "ThrowDB server is not ready to serve"
                )
            self._description = self._describe()
            # Describe refreshes live diagnostics and may observe a journal
            # rollover. Legacy UUID retries must keep their original domain.
            self._mutation_domain = self._description.idempotency_domain
        except BaseException:
            self.close()
            raise

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        timeout: float | None = 30.0,
        scan_timeout: float | None = DEFAULT_SCAN_TIMEOUT_SECONDS,
        max_frame_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        credentials: grpc.ChannelCredentials | None = None,
        use_http_proxy: bool = False,
    ) -> RemoteCollection:
        """Connect to one gRPC-served collection.

        Ambient ``HTTP_PROXY``/``HTTPS_PROXY`` settings are ignored by
        default so private embedding traffic cannot be redirected
        accidentally. Set ``use_http_proxy=True`` only when the endpoint is
        intentionally reached through the configured HTTP proxy.
        """
        return cls(
            endpoint,
            timeout=timeout,
            scan_timeout=scan_timeout,
            max_frame_bytes=max_frame_bytes,
            credentials=credentials,
            use_http_proxy=use_http_proxy,
        )

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("remote collection is closed")

    def _invoke(self, call: object, request: object) -> object:
        try:
            return call(  # type: ignore[operator]
                request,
                timeout=self._timeout,
                wait_for_ready=True,
            )
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error

    def _invoke_mutation(
        self,
        call: object,
        request: object,
        *,
        operation: str,
        token: RemoteMutationToken,
    ) -> object:
        try:
            return call(  # type: ignore[operator]
                request,
                timeout=self._timeout,
                wait_for_ready=True,
            )
        except grpc.RpcError as error:
            code = _rpc_code_name(error)
            if code == "ABORTED":
                raise RemoteAbortedError(
                    token.operation_id,
                    operation,
                    details=_rpc_details(error),
                    idempotency_domain=token.idempotency_domain,
                ) from error
            if code not in _DEFINITE_MUTATION_REJECTIONS:
                raise AmbiguousCommitError(
                    token.operation_id,
                    operation,
                    code=code,
                    details=_rpc_details(error),
                    idempotency_domain=token.idempotency_domain,
                ) from error
            raise _typed_rpc_error(error) from error

    def _describe(self) -> RemoteDescription:
        response = self._invoke(
            self._stub.Describe, throwdb_pb2.DescribeRequest()
        )
        assert isinstance(response, throwdb_pb2.DescribeResponse)
        if response.format_version != 7:
            raise RemoteProtocolError(
                f"server collection format {response.format_version} is unsupported; "
                "only format 7 is supported"
            )
        if response.schema.id_type != throwdb_pb2.ID_TYPE_UINT64:
            raise RemoteProtocolError("server collection is not a uint64 collection")
        dtype = _PROTO_TO_DTYPE.get(response.schema.embedding_dtype)
        if dtype is None:
            raise RemoteProtocolError("server returned an unknown embedding dtype")
        try:
            schema = CollectionSchema(
                id_type="uint64",
                dimension=response.schema.dimension,
                dtype=dtype,
            )
        except (TypeError, ValueError) as error:
            raise RemoteProtocolError("server returned an invalid schema") from error
        expected_record_size = schema.dimension * np.dtype(schema.dtype).itemsize
        if response.record_size != expected_record_size:
            raise RemoteProtocolError(
                "server record_size does not match dimension and dtype"
            )
        capabilities = 0
        if response.capabilities.writable:
            capabilities |= CAPABILITY_WRITABLE
        if response.capabilities.prefix_get:
            capabilities |= CAPABILITY_PREFIX_GET
        if response.capabilities.scan_locations:
            capabilities |= CAPABILITY_SCAN_LOCATIONS
        if response.capabilities.durable_idempotency:
            capabilities |= CAPABILITY_DURABLE_IDEMPOTENCY
        if response.capabilities.stream_get:
            capabilities |= CAPABILITY_STREAM_GET
        if response.capabilities.scan_records:
            capabilities |= CAPABILITY_SCAN_RECORDS
        reader: RemoteReaderConfiguration | None = None
        if response.HasField("reader"):
            backend = {
                throwdb_pb2.READER_BACKEND_THREAD_POOL: "thread_pool",
                throwdb_pb2.READER_BACKEND_IO_URING: "io_uring",
            }.get(response.reader.io_backend)
            if backend is None:
                raise RemoteProtocolError(
                    "server returned an unknown reader backend"
                )
            raw_reader: dict[str, object] = {
                "max_read_range_bytes": response.reader.max_read_range_bytes,
                "max_gap_bytes": response.reader.max_gap_bytes,
                "io_workers": response.reader.io_workers,
                "io_queue_capacity": response.reader.io_queue_capacity,
                "io_ranges_per_task": response.reader.io_ranges_per_task,
                "io_backend": backend,
                "io_uring_queue_depth": response.reader.io_uring_queue_depth,
                "io_uring_workers": response.reader.io_uring_workers,
            }
            try:
                normalized_reader = normalize_reader_options(
                    raw_reader,
                    persisted=True,
                    record_size=expected_record_size,
                )
            except (TypeError, ValueError) as error:
                raise RemoteProtocolError(
                    "server returned an invalid reader configuration"
                ) from error
            reader = RemoteReaderConfiguration(
                max_read_range_bytes=int(
                    normalized_reader["max_read_range_bytes"]
                ),
                max_gap_bytes=int(normalized_reader["max_gap_bytes"]),
                io_workers=int(normalized_reader["io_workers"]),
                io_queue_capacity=int(
                    normalized_reader["io_queue_capacity"]
                ),
                io_ranges_per_task=int(
                    normalized_reader["io_ranges_per_task"]
                ),
                io_backend=str(normalized_reader["io_backend"]),
                io_uring_queue_depth=int(
                    normalized_reader["io_uring_queue_depth"]
                ),
                io_uring_workers=int(
                    normalized_reader["io_uring_workers"]
                ),
            )
        selection: RemoteTuningSelection | None
        if response.HasField("tuning"):
            selection = RemoteTuningSelection(
                mode=response.tuning.mode,
                source=response.tuning.source,
                status=response.tuning.status,
                reuse_level=response.tuning.reuse_level,
                explicit_overrides=response.tuning.explicit_overrides,
            )
        else:
            selection = None
        if selection is not None and (
            selection.mode not in {
                "auto",
                "persisted",
                "conservative",
                "manual",
            }
            or selection.source not in {"persisted", "fallback", "manual"}
            or not selection.status
            or selection.reuse_level
            not in {"exact", "compatible", "fallback", "manual"}
        ):
            raise RemoteProtocolError(
                "server returned invalid tuning diagnostics"
            )
        availability: RemoteAvailabilityConfiguration | None = None
        if response.HasField("availability"):
            raw = response.availability
            values = {
                "max_concurrent_calls": raw.max_concurrent_calls,
                "max_concurrent_scans": raw.max_concurrent_scans,
                "max_unary_inflight_bytes": raw.max_unary_inflight_bytes,
                "max_scan_inflight_bytes": raw.max_scan_inflight_bytes,
                "max_grpc_memory_bytes": raw.max_grpc_memory_bytes,
                "scan_max_duration_millis": raw.scan_max_duration_millis,
                "scan_idle_timeout_millis": raw.scan_idle_timeout_millis,
                "shutdown_grace_millis": raw.shutdown_grace_millis,
                "shutdown_hard_timeout_millis":
                    raw.shutdown_hard_timeout_millis,
            }
            if any(value <= 0 for value in values.values()):
                raise RemoteProtocolError(
                    "server returned non-positive availability limits"
                )
            if not (
                raw.max_concurrent_scans < raw.max_concurrent_calls
                and raw.shutdown_grace_millis
                    <= raw.shutdown_hard_timeout_millis
            ):
                raise RemoteProtocolError(
                    "server returned inconsistent availability limits"
                )
            availability = RemoteAvailabilityConfiguration(**values)
        admission_stats: RemoteAdmissionStatistics | None = None
        if response.HasField("admission_stats"):
            raw_stats = response.admission_stats
            if (
                raw_stats.draining and raw_stats.terminal_faulted
            ) or (
                (raw_stats.draining or raw_stats.terminal_faulted)
                and not raw_stats.admission_closed
            ):
                raise RemoteProtocolError(
                    "server returned inconsistent admission lifecycle state"
                )
            if availability is not None and (
                raw_stats.active_unary_calls
                    + raw_stats.active_scans
                    > availability.max_concurrent_calls
                or raw_stats.active_scans
                    > availability.max_concurrent_scans
                or raw_stats.unary_estimated_bytes
                    > availability.max_unary_inflight_bytes
                or raw_stats.scan_estimated_bytes
                    > availability.max_scan_inflight_bytes
            ):
                raise RemoteProtocolError(
                    "server returned admission gauges above configured limits"
                )
            admission_stats = RemoteAdmissionStatistics(
                active_unary_calls=raw_stats.active_unary_calls,
                active_scans=raw_stats.active_scans,
                unary_estimated_bytes=raw_stats.unary_estimated_bytes,
                scan_estimated_bytes=raw_stats.scan_estimated_bytes,
                admitted_unary_calls=raw_stats.admitted_unary_calls,
                rejected_unary_calls=raw_stats.rejected_unary_calls,
                admitted_scans=raw_stats.admitted_scans,
                rejected_scans=raw_stats.rejected_scans,
                admission_closed=raw_stats.admission_closed,
                draining=raw_stats.draining,
                terminal_faulted=raw_stats.terminal_faulted,
            )
        idempotency_domain: uuid.UUID | None = None
        if response.capabilities.durable_idempotency:
            if not response.HasField("idempotency_domain"):
                # Protocol v1.0 advertised durable operation-id replay before
                # replay domains existed. Preserve rolling-upgrade and read
                # compatibility, while a v1.1+ contract violation fails closed.
                if self._protocol_minor != 0:
                    raise RemoteProtocolError(
                        "durable server omitted its idempotency replay domain"
                    )
            else:
                idempotency_domain = _operation_id_from_proto(
                    response.idempotency_domain,
                    name="idempotency replay domain",
                )
        elif response.HasField("idempotency_domain") and (
            response.idempotency_domain.high != 0
            or response.idempotency_domain.low != 0
        ):
            raise RemoteProtocolError(
                "non-durable server advertised an idempotency replay domain"
            )
        engine_stats: RemoteEngineStatistics | None = None
        if response.HasField("engine_stats"):
            raw_engine = response.engine_stats
            if (
                raw_engine.materialized_sequence
                    > raw_engine.committed_sequence
                or raw_engine.active_wal_bytes > raw_engine.total_wal_bytes
                or raw_engine.frozen_wal_bytes > raw_engine.total_wal_bytes
                or raw_engine.active_wal_operations
                    > raw_engine.total_wal_operations
                or (
                    raw_engine.idempotency_max_bytes != 0
                    and raw_engine.idempotency_journal_bytes
                        > raw_engine.idempotency_max_bytes
                )
                or (
                    raw_engine.idempotency_max_operations != 0
                    and raw_engine.idempotency_completed_operations
                        + raw_engine.idempotency_aborted_operations
                        > raw_engine.idempotency_max_operations
                )
            ):
                raise RemoteProtocolError(
                    "server returned inconsistent engine statistics"
                )
            engine_stats = RemoteEngineStatistics(
                **{
                    field: getattr(raw_engine, field)
                    for field in RemoteEngineStatistics.__dataclass_fields__
                }
            )
        read_logging: RemoteReadLoggingConfiguration | None = None
        has_read_logging = response.HasField("read_logging")
        has_read_logging_stats = response.HasField("read_logging_stats")
        if (
            self._protocol_minor is not None
            and self._protocol_minor >= 4
            and (not has_read_logging or not has_read_logging_stats)
        ):
            raise RemoteProtocolError(
                "server omitted protocol-v1.4 read-logging diagnostics"
            )
        if has_read_logging:
            raw_logging = response.read_logging
            raw_format_version = int(raw_logging.format_version)
            format_version = (
                raw_format_version
                if raw_format_version != 0
                else (1 if raw_logging.enabled else 0)
            )
            id_chunk_rows = (
                int(raw_logging.id_chunk_rows)
                if format_version >= 2
                else int(raw_logging.max_ids_per_event)
            )
            maximum_record_bytes = max(
                _READ_LOG_BATCH_RECORD_BYTES,
                _READ_LOG_RECORD_HEADER_BYTES + 8 * id_chunk_rows,
            )
            configured_values = (
                raw_logging.sample_ppm,
                raw_logging.max_ids_per_event,
                raw_logging.buffer_bytes,
                raw_logging.segment_bytes,
                raw_logging.rotate_seconds,
                raw_logging.retention_bytes,
                raw_logging.retention_seconds,
                raw_logging.minimum_free_bytes,
                raw_logging.max_write_bytes_per_second,
                raw_logging.max_retained_files,
                raw_logging.id_chunk_rows,
                raw_logging.format_version,
            )
            if (
                raw_logging.enabled
                and (
                    format_version not in {1, 2}
                    or raw_logging.sample_ppm < 1
                    or raw_logging.sample_ppm > 1_000_000
                    or raw_logging.max_ids_per_event < 1
                    or raw_logging.max_ids_per_event > 65_536
                    or id_chunk_rows < 1
                    or id_chunk_rows > (
                        8_192 if format_version >= 2 else 65_536
                    )
                    or (
                        format_version >= 2
                        and (
                            raw_logging.sample_ppm != 1_000_000
                            or raw_logging.max_ids_per_event
                                != id_chunk_rows
                            or raw_logging.id_chunk_rows != id_chunk_rows
                        )
                    )
                    or raw_logging.buffer_bytes
                        < 2 * maximum_record_bytes
                    or raw_logging.segment_bytes
                        < (
                            _READ_LOG_HEADER_BYTES
                            + maximum_record_bytes
                            + 2 * _READ_LOG_TERMINAL_CHECKPOINT_BYTES
                        )
                    or raw_logging.rotate_seconds < 1
                    or raw_logging.rotate_seconds > (1 << 63) - 1
                    or raw_logging.retention_bytes
                        < raw_logging.segment_bytes
                    or raw_logging.retention_seconds < 1
                    or raw_logging.max_write_bytes_per_second
                        < (
                            _READ_LOG_HEADER_BYTES
                            + maximum_record_bytes
                            + 2 * _READ_LOG_TERMINAL_CHECKPOINT_BYTES
                        )
                    or raw_logging.max_retained_files < 1
                    or raw_logging.max_retained_files > 65_533
                )
            ) or (
                not raw_logging.enabled and any(configured_values)
            ):
                raise RemoteProtocolError(
                    "server returned invalid read-logging configuration"
                )
            configuration = {
                field: getattr(raw_logging, field)
                for field in RemoteReadLoggingConfiguration.__dataclass_fields__
            }
            configuration["id_chunk_rows"] = id_chunk_rows
            configuration["format_version"] = format_version
            read_logging = RemoteReadLoggingConfiguration(**configuration)
        read_logging_stats: RemoteReadLoggingStatistics | None = None
        if has_read_logging_stats:
            raw_logging_stats = response.read_logging_stats
            maximum_u64 = (1 << 64) - 1
            expected_queue_drops = min(
                maximum_u64,
                raw_logging_stats.dropped_busy_events
                + raw_logging_stats.dropped_full_events,
            )
            lifecycle_v2 = (
                read_logging is not None
                and read_logging.enabled
                and read_logging.format_version >= 2
            )
            completed_outcomes = min(
                maximum_u64,
                raw_logging_stats.complete_requests
                + raw_logging_stats.incomplete_requests,
            )
            v2_counters_invalid = lifecycle_v2 and (
                raw_logging_stats.begun_requests
                    > raw_logging_stats.observed_requests
                or raw_logging_stats.completed_requests
                    > raw_logging_stats.observed_requests
                or raw_logging_stats.complete_requests
                    > raw_logging_stats.completed_requests
                or raw_logging_stats.completed_requests
                    != completed_outcomes
                or raw_logging_stats.enqueued_records
                    > raw_logging_stats.observed_records
                or raw_logging_stats.written_records
                    > raw_logging_stats.enqueued_records
                or raw_logging_stats.enqueued_id_chunks
                    > raw_logging_stats.observed_id_chunks
                or raw_logging_stats.written_id_chunks
                    > raw_logging_stats.enqueued_id_chunks
                or raw_logging_stats.enqueued_batches
                    > raw_logging_stats.observed_batches
                or raw_logging_stats.written_batches
                    > raw_logging_stats.enqueued_batches
                or raw_logging_stats.enqueued_ids
                    > raw_logging_stats.observed_ids
                or raw_logging_stats.written_ids
                    > raw_logging_stats.enqueued_ids
            )
            numeric_stats = (
                getattr(raw_logging_stats, field)
                for field in
                RemoteReadLoggingStatistics.__dataclass_fields__
                if field not in {
                    "enabled",
                    "paused_for_minimum_free_space",
                    "runtime_failed",
                    "shutting_down",
                }
            )
            if (
                read_logging is None
                or raw_logging_stats.enabled != read_logging.enabled
            ) or (
                raw_logging_stats.dropped_queue_events
                    != expected_queue_drops
            ) or (
                raw_logging_stats.enabled
                and (
                    v2_counters_invalid
                    or raw_logging_stats.buffer_capacity_bytes == 0
                    or raw_logging_stats.buffer_capacity_bytes
                        > read_logging.buffer_bytes
                    or raw_logging_stats.queue_bytes
                        > raw_logging_stats.buffer_capacity_bytes
                    or raw_logging_stats.peak_queue_bytes
                        > raw_logging_stats.buffer_capacity_bytes
                )
            ) or (
                not raw_logging_stats.enabled
                and (
                    any(numeric_stats)
                    or raw_logging_stats.paused_for_minimum_free_space
                    or raw_logging_stats.runtime_failed
                    or raw_logging_stats.shutting_down
                )
            ):
                raise RemoteProtocolError(
                    "server returned inconsistent read-logging statistics"
                )
            statistics = {
                field: getattr(raw_logging_stats, field)
                for field in RemoteReadLoggingStatistics.__dataclass_fields__
            }
            if (
                read_logging is not None
                and read_logging.enabled
                and read_logging.format_version == 1
            ):
                incomplete = min(
                    maximum_u64,
                    raw_logging_stats.truncated_events
                    + raw_logging_stats.dropped_queue_events
                    + raw_logging_stats.dropped_disk_pressure_events
                    + raw_logging_stats.dropped_error_events
                    + raw_logging_stats.dropped_io_budget_events,
                )
                complete = int(raw_logging_stats.written_events)
                statistics.update(
                    observed_requests=int(
                        raw_logging_stats.considered_events
                    ),
                    begun_requests=int(raw_logging_stats.sampled_events),
                    completed_requests=min(
                        maximum_u64, complete + incomplete
                    ),
                    complete_requests=complete,
                    incomplete_requests=incomplete,
                    observed_records=int(
                        raw_logging_stats.sampled_events
                    ),
                    enqueued_records=int(
                        raw_logging_stats.enqueued_events
                    ),
                    written_records=complete,
                    observed_id_chunks=int(
                        raw_logging_stats.sampled_events
                    ),
                    enqueued_id_chunks=int(
                        raw_logging_stats.enqueued_events
                    ),
                    written_id_chunks=complete,
                    dropped_full_records=int(
                        raw_logging_stats.dropped_queue_events
                    ),
                    dropped_runtime_failure_records=int(
                        raw_logging_stats.dropped_error_events
                    ),
                    dropped_minimum_free_space_records=int(
                        raw_logging_stats.dropped_disk_pressure_events
                    ),
                    dropped_io_budget_records=int(
                        raw_logging_stats.dropped_io_budget_events
                    ),
                    queued_records=int(raw_logging_stats.queued_events),
                )
            read_logging_stats = RemoteReadLoggingStatistics(**statistics)
        return RemoteDescription(
            format_version=response.format_version,
            schema=schema,
            capabilities=capabilities,
            record_size=response.record_size,
            count=response.count,
            reader=reader,
            tuning=selection,
            availability=availability,
            admission_stats=admission_stats,
            engine_stats=engine_stats,
            read_logging=read_logging,
            read_logging_stats=read_logging_stats,
            idempotency_domain=idempotency_domain,
        )

    def describe(self) -> RemoteDescription:
        self._ensure_open()
        description = self._describe()
        if description.schema != self._description.schema:
            raise RemoteProtocolError("server schema changed after connect")
        self._description = description
        return description

    def health(self) -> dict[str, int | bool]:
        self._ensure_open()
        response = self._invoke(
            self._stub.Health, throwdb_pb2.HealthRequest()
        )
        assert isinstance(response, throwdb_pb2.HealthResponse)
        if response.status not in {
            throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING,
            throwdb_pb2.HealthResponse.SERVING_STATUS_NOT_SERVING,
        }:
            raise RemoteProtocolError("server returned an invalid health status")
        if response.protocol_major != _PROTOCOL_MAJOR:
            raise RemoteProtocolError(
                "server protocol major is incompatible with this client: "
                f"server={response.protocol_major}, client={_PROTOCOL_MAJOR}"
            )
        self._protocol_minor = response.protocol_minor
        return {
            "serving": response.status
            == throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING,
            "protocol_major": response.protocol_major,
            "protocol_minor": response.protocol_minor,
        }

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def schema(self) -> CollectionSchema:
        return self._description.schema

    @property
    def id_type(self) -> str:
        return self.schema.id_type

    @property
    def dimension(self) -> int:
        return self.schema.dimension

    @property
    def dtype(self) -> str:
        return self.schema.dtype

    @property
    def format_version(self) -> int:
        return self._description.format_version

    @property
    def mode(self) -> str:
        return "read_write" if self._description.writable else "read"

    @property
    def count(self) -> int:
        return self.describe().count

    @property
    def capabilities(self) -> int:
        return self._description.capabilities

    @property
    def durable_idempotency(self) -> bool:
        return self._description.durable_idempotency

    @property
    def supports_stream_get(self) -> bool:
        return self._description.stream_get

    @property
    def supports_scan_records(self) -> bool:
        return self._description.scan_records

    @property
    def idempotency_domain(self) -> uuid.UUID | None:
        return self._description.idempotency_domain

    def mutation_token(
        self,
        operation_id: uuid.UUID | str | bytes | None = None,
    ) -> RemoteMutationToken:
        """Bind a new logical mutation to the currently described domain.

        Call ``describe()`` first when intentionally starting new work after a
        server journal checkpoint. Existing ambiguous operations must retain
        their original token; never create a replacement token for a retry.
        """
        self._ensure_open()
        return RemoteMutationToken(
            _operation_id(operation_id), self._description.idempotency_domain
        )

    def _mutation_token(
        self,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None,
    ) -> RemoteMutationToken:
        if isinstance(operation_id, RemoteMutationToken):
            return operation_id
        # A generated UUID always denotes new work; caller-supplied UUIDs
        # may be retries and therefore retain the connection's original domain.
        domain = (
            self._description.idempotency_domain
            if operation_id is None
            else self._mutation_domain
        )
        return RemoteMutationToken(_operation_id(operation_id), domain)

    @property
    def reader(self) -> RemoteReaderConfiguration | None:
        return self._description.reader

    @property
    def tuning(self) -> RemoteTuningSelection | None:
        return self._description.tuning

    @property
    def availability(self) -> RemoteAvailabilityConfiguration | None:
        return self._description.availability

    @property
    def admission_stats(self) -> RemoteAdmissionStatistics | None:
        # Admission gauges and lifecycle state are live diagnostics rather
        # than immutable collection metadata. Refresh them on each access.
        return self.describe().admission_stats

    @property
    def engine_stats(self) -> RemoteEngineStatistics | None:
        # The mutation backlog and filesystem headroom are live gauges.
        return self.describe().engine_stats

    @property
    def read_logging(self) -> RemoteReadLoggingConfiguration | None:
        return self._description.read_logging

    @property
    def read_logging_stats(self) -> RemoteReadLoggingStatistics | None:
        # Queue pressure, retention and writer failures are live diagnostics.
        return self.describe().read_logging_stats

    def get(
        self,
        ids: object,
        *,
        prefix_dimensions: int | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        normalized_ids = _normalize_numeric_ids(ids)
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        if prefix != self.dimension and not self._description.prefix_get:
            raise NotImplementedError("server does not support prefix reads")
        request = throwdb_pb2.GetRequest(
            ids=_proto_ids(normalized_ids),
            prefix_dimensions=0 if prefix_dimensions is None else prefix,
        )
        response = self._invoke(self._stub.Get, request)
        assert isinstance(response, throwdb_pb2.GetResponse)
        embeddings = response.embeddings
        if (
            embeddings.dtype != _DTYPE_TO_PROTO[self.dtype]
            or embeddings.dimension != prefix
            or embeddings.row_count != normalized_ids.size
        ):
            raise RemoteProtocolError("Get response metadata does not match request")
        expected_bytes = (
            normalized_ids.size * prefix * np.dtype(self.dtype).itemsize
        )
        if len(embeddings.values_le) != expected_bytes:
            raise RemoteProtocolError("Get response byte length does not match metadata")
        view = np.frombuffer(
            embeddings.values_le, dtype=np.dtype(self.dtype).newbyteorder("<")
        ).reshape((normalized_ids.size, prefix))
        return np.array(view, dtype=np.dtype(self.dtype), copy=True, order="C")

    def get_into(
        self,
        ids: object,
        output: object,
        *,
        prefix_dimensions: int | None = None,
    ) -> None:
        self._ensure_open()
        normalized_ids = _normalize_numeric_ids(ids)
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        destination = _validate_output(
            output,
            expected_rows=normalized_ids.size,
            expected_columns=prefix,
            dtype=self.dtype,
        )
        result = self.get(normalized_ids, prefix_dimensions=prefix)
        np.copyto(destination, result, casting="no")

    def stream_get(
        self,
        ids: object,
        *,
        prefix_dimensions: int | None = None,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
        max_batch_payload_bytes: int | None = (
            _DEFAULT_STREAM_BATCH_PAYLOAD_BYTES
        ),
    ) -> RemoteEmbeddingStream:
        """Open a bounded, snapshot-consistent lookup stream.

        IDs are normalized only when their negotiated batch is submitted. The
        protocol uses an application window of one, so the gRPC request thread
        cannot consume the complete source ahead of the response iterator.
        """

        self._ensure_open()
        if not self._description.stream_get:
            raise NotImplementedError("server does not support StreamGet")
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        if prefix != self.dimension and not self._description.prefix_get:
            raise NotImplementedError("server does not support prefix reads")
        requested_rows = _validate_positive_u32(
            batch_rows,
            name="batch_rows",
        )
        requested_bytes = _validate_optional_positive_u64(
            max_batch_payload_bytes,
            name="max_batch_payload_bytes",
        )
        effective_rows, effective_bytes = _bounded_stream_parameters(
            requested_rows=requested_rows,
            requested_payload_bytes=requested_bytes,
            dense_row_bytes=prefix * np.dtype(self.dtype).itemsize,
            max_frame_bytes=self._max_frame_bytes,
            include_id_request=True,
            operation="StreamGet",
        )
        source = _IndexedIds(ids)
        try:
            return RemoteEmbeddingStream(
                self._stub.StreamGet,
                source,
                dtype=self.dtype,
                dimension=prefix,
                prefix_dimensions=(
                    0 if prefix_dimensions is None else prefix
                ),
                requested_batch_rows=effective_rows,
                requested_batch_payload_bytes=effective_bytes,
                timeout=self._scan_timeout,
            )
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error

    def _put(
        self,
        method: object,
        operation_name: str,
        ids: object,
        embeddings: object,
        *,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None,
    ) -> None:
        self._ensure_open()
        if not self._description.writable:
            raise RuntimeError("server collection is read-only")
        normalized_ids = _normalize_numeric_ids(ids)
        normalized_embeddings = _normalize_embeddings(
            embeddings,
            expected_rows=normalized_ids.size,
            schema=self.schema,
        )
        if normalized_ids.size == 0:
            return
        token = self._mutation_token(operation_id)
        identifier = token.operation_id
        wire_embeddings = _native_little_endian(
            normalized_embeddings, self.dtype
        )
        request = throwdb_pb2.MutationRequest(
            operation_id=_proto_operation_id(identifier),
            ids=_proto_ids(normalized_ids),
            embeddings=throwdb_pb2.EmbeddingBatch(
                dtype=_DTYPE_TO_PROTO[self.dtype],
                dimension=self.dimension,
                row_count=normalized_ids.size,
                values_le=wire_embeddings.tobytes(order="C"),
            ),
        )
        if token.idempotency_domain is not None:
            request.idempotency_domain.CopyFrom(
                _proto_operation_id(token.idempotency_domain)
            )
        response = self._invoke_mutation(
            method,
            request,
            operation=operation_name,
            token=token,
        )
        assert isinstance(response, throwdb_pb2.MutationResponse)
        if response.affected_count != normalized_ids.size:
            raise AmbiguousCommitError(
                identifier,
                operation_name,
                idempotency_domain=token.idempotency_domain,
                code="PROTOCOL_ERROR",
                details="mutation response count does not match request",
            )

    def insert(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(
            self._stub.Insert,
            "insert",
            ids,
            embeddings,
            operation_id=operation_id,
        )

    def update(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(
            self._stub.Update,
            "update",
            ids,
            embeddings,
            operation_id=operation_id,
        )

    def upsert(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(
            self._stub.Upsert,
            "upsert",
            ids,
            embeddings,
            operation_id=operation_id,
        )

    def delete(
        self,
        ids: object,
        *,
        operation_id: RemoteMutationToken | uuid.UUID | str | bytes | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        if not self._description.writable:
            raise RuntimeError("server collection is read-only")
        normalized_ids = _normalize_numeric_ids(ids)
        if normalized_ids.size == 0:
            return np.empty(0, dtype=np.bool_)
        token = self._mutation_token(operation_id)
        identifier = token.operation_id
        request = throwdb_pb2.DeleteRequest(
            operation_id=_proto_operation_id(identifier),
            ids=_proto_ids(normalized_ids),
        )
        if token.idempotency_domain is not None:
            request.idempotency_domain.CopyFrom(
                _proto_operation_id(token.idempotency_domain)
            )
        response = self._invoke_mutation(
            self._stub.Delete,
            request,
            operation="delete",
            token=token,
        )
        assert isinstance(response, throwdb_pb2.DeleteResponse)
        if (
            response.affected_count != normalized_ids.size
            or len(response.deleted_mask) != normalized_ids.size
        ):
            raise AmbiguousCommitError(
                identifier,
                "delete",
                idempotency_domain=token.idempotency_domain,
                code="PROTOCOL_ERROR",
                details="delete response size does not match request",
            )
        mask = np.frombuffer(response.deleted_mask, dtype=np.uint8)
        if bool(np.any(mask > 1)):
            raise AmbiguousCommitError(
                identifier,
                "delete",
                idempotency_domain=token.idempotency_domain,
                code="PROTOCOL_ERROR",
                details="delete response contains an invalid bitmap value",
            )
        return mask.astype(np.bool_, copy=True)

    @overload
    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
        with_locations: Literal[False] = False,
    ) -> RemoteIdScan: ...

    @overload
    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
        with_locations: Literal[True],
    ) -> RemoteIdScan: ...

    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
        with_locations: bool = False,
    ) -> RemoteIdScan:
        self._ensure_open()
        rows = _validate_positive_u32(batch_rows, name="batch_rows")
        if not isinstance(with_locations, bool):
            raise TypeError("with_locations must be a bool")
        if with_locations and not self._description.scan_locations:
            raise NotImplementedError("server does not support ID locations")
        row_bytes = 8 + (ID_LOCATION_DTYPE.itemsize if with_locations else 0)
        frame_capacity = self._max_frame_bytes - _SCAN_FRAME_ALLOWANCE
        if frame_capacity < row_bytes:
            raise ValueError("max_frame_bytes cannot fit one ScanIds row")
        rows = min(rows, frame_capacity // row_bytes)
        try:
            stream = self._stub.ScanIds(
                throwdb_pb2.ScanIdsRequest(
                    batch_rows=rows,
                    include_locations=with_locations,
                ),
                timeout=self._scan_timeout,
                wait_for_ready=True,
            )
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error
        return RemoteIdScan(
            stream,
            batch_rows=rows,
            with_locations=with_locations,
        )

    def scan_records(
        self,
        *,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
        max_batch_payload_bytes: int | None = (
            _DEFAULT_STREAM_BATCH_PAYLOAD_BYTES
        ),
        prefix_dimensions: int | None = None,
        with_locations: bool = False,
    ) -> RemoteRecordScan:
        """Stream every live record in physical order from one snapshot."""

        self._ensure_open()
        if not self._description.scan_records:
            raise NotImplementedError("server does not support ScanRecords")
        rows = _validate_positive_u32(batch_rows, name="batch_rows")
        payload_bytes = _validate_optional_positive_u64(
            max_batch_payload_bytes,
            name="max_batch_payload_bytes",
        )
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        if prefix != self.dimension and not self._description.prefix_get:
            raise NotImplementedError("server does not support prefix reads")
        if not isinstance(with_locations, bool):
            raise TypeError("with_locations must be a bool")
        if with_locations and not self._description.scan_locations:
            raise NotImplementedError("server does not support record locations")
        effective_rows, effective_payload_bytes = _bounded_stream_parameters(
            requested_rows=rows,
            requested_payload_bytes=payload_bytes,
            dense_row_bytes=(
                8
                + prefix * np.dtype(self.dtype).itemsize
                + (ID_LOCATION_DTYPE.itemsize if with_locations else 0)
            ),
            max_frame_bytes=self._max_frame_bytes,
            include_id_request=False,
            operation="ScanRecords",
        )
        try:
            stream = self._stub.ScanRecords(
                throwdb_pb2.ScanRecordsRequest(
                    batch_rows=effective_rows,
                    max_batch_payload_bytes=effective_payload_bytes,
                    prefix_dimensions=(
                        0 if prefix_dimensions is None else prefix
                    ),
                    include_locations=with_locations,
                ),
                timeout=self._scan_timeout,
                wait_for_ready=True,
            )
        except grpc.RpcError as error:
            raise _typed_rpc_error(error) from error
        return RemoteRecordScan(
            stream,
            requested_batch_rows=effective_rows,
            requested_batch_payload_bytes=effective_payload_bytes,
            dtype=self.dtype,
            dimension=prefix,
            with_locations=with_locations,
        )

    @overload
    def list_ids(
        self,
        *,
        with_locations: Literal[False] = False,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
    ) -> np.ndarray: ...

    @overload
    def list_ids(
        self,
        *,
        with_locations: Literal[True],
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
    ) -> LocatedIds: ...

    def list_ids(
        self,
        *,
        with_locations: bool = False,
        batch_rows: int = _DEFAULT_SCAN_BATCH_ROWS,
    ) -> np.ndarray | LocatedIds:
        """Materialize every live ID in physical order with no memory guard."""

        with self.scan_ids(
            batch_rows=batch_rows,
            with_locations=with_locations,
        ) as scan:
            ids = np.empty(scan.snapshot_count, dtype=np.uint64)
            locations = (
                np.empty(scan.snapshot_count, dtype=ID_LOCATION_DTYPE)
                if with_locations
                else None
            )
            offset = 0
            for item in scan:
                if isinstance(item, LocatedIdBatch):
                    count = item.ids.size
                    ids[offset : offset + count] = item.ids
                    assert locations is not None
                    locations[offset : offset + count] = item.locations
                else:
                    count = item.size
                    ids[offset : offset + count] = item
                offset += count
            if offset != scan.snapshot_count:
                raise RemoteProtocolError(
                    "ID scan ended before its advertised snapshot count"
                )
            if locations is not None:
                return LocatedIds(
                    ids=ids,
                    locations=locations,
                    snapshot=scan.snapshot,
                )
            return ids

    def close(self) -> None:
        if self._state == "closed":
            return
        self._state = "closed"
        self._channel.close()

    def __enter__(self) -> RemoteCollection:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


__all__ = [
    "AmbiguousCommitError",
    "CAPABILITY_DURABLE_IDEMPOTENCY",
    "CAPABILITY_PREFIX_GET",
    "CAPABILITY_SCAN_RECORDS",
    "CAPABILITY_SCAN_LOCATIONS",
    "CAPABILITY_STREAM_GET",
    "CAPABILITY_WRITABLE",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_SCAN_TIMEOUT_SECONDS",
    "RemoteAlreadyExistsError",
    "RemoteAbortedError",
    "RemoteAvailabilityConfiguration",
    "RemoteCancelledError",
    "RemoteCollection",
    "RemoteDataLossError",
    "RemoteDeadlineExceededError",
    "RemoteDescription",
    "RemoteEngineStatistics",
    "RemoteEmbeddingBatch",
    "RemoteEmbeddingStream",
    "RemoteError",
    "RemoteFailedPreconditionError",
    "RemoteIdScan",
    "RemoteInternalError",
    "RemoteInvalidArgumentError",
    "RemoteMutationToken",
    "RemoteNotFoundError",
    "RemoteProtocolError",
    "RemoteReadLoggingConfiguration",
    "RemoteReadLoggingStatistics",
    "RemoteRecordScan",
    "RemoteReaderConfiguration",
    "RemoteResourceExhaustedError",
    "RemoteUnavailableError",
    "RemoteUnimplementedError",
    "RemoteTuningSelection",
]
