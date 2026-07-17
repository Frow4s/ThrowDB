"""Public gRPC client for a single ThrowDB collection."""

from __future__ import annotations

import contextlib
import operator
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Iterator, Literal, overload

import grpc
import numpy as np

from ._tuning import normalize_reader_options
from .collection import (
    CollectionSchema,
    _normalize_embeddings,
    _normalize_numeric_ids,
    _validate_id_scan_output,
    _validate_output,
    _validate_prefix_dimensions,
)
from .id_scan import (
    ID_LOCATION_DTYPE,
    IdScanSnapshot,
    LocatedIdBatch,
    LocatedIds,
)
from .v1 import throwdb_pb2, throwdb_pb2_grpc


DEFAULT_MAX_MESSAGE_BYTES: Final = 1 << 30
DEFAULT_SCAN_TIMEOUT_SECONDS: Final = 3_600.0
_DEFAULT_SCAN_BATCH_ROWS: Final = 65_536
_UINT32_MAX: Final = (1 << 32) - 1
_GRPC_MAX_MESSAGE_BYTES: Final = (1 << 31) - 1
_PROTOCOL_MAJOR: Final = 1

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


class AmbiguousCommitError(RemoteError, ConnectionError):
    """A mutation may have committed although its response was not received."""

    def __init__(
        self,
        operation_id: uuid.UUID,
        operation: str,
        *,
        code: str,
        details: str,
    ) -> None:
        self.operation_id = operation_id
        self.operation = operation
        super().__init__(
            code,
            f"remote {operation} outcome is unknown; "
            f"operation_id={operation_id}; {details or code}. Do not retry "
            "automatically unless the server advertises durable idempotency.",
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
        try:
            self._read_one()
        except BaseException:
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
        if self._closed:
            raise StopIteration
        if self._pending is not None:
            return self._take_pending()
        if self._finished:
            raise StopIteration
        self._read_one()
        if self._pending is None:
            raise StopIteration
        return self._take_pending()

    def read_into(
        self,
        ids: object,
        locations: object | None = None,
    ) -> int:
        if self._closed:
            raise RuntimeError("ID scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
        )
        if self._pending is None:
            if self._finished:
                return 0
            self._read_one()
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
        self._finished = True
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
        operation_id: uuid.UUID,
    ) -> object:
        try:
            return call(  # type: ignore[operator]
                request,
                timeout=self._timeout,
                wait_for_ready=True,
            )
        except grpc.RpcError as error:
            code = _rpc_code_name(error)
            if code not in _DEFINITE_MUTATION_REJECTIONS:
                raise AmbiguousCommitError(
                    operation_id,
                    operation,
                    code=code,
                    details=_rpc_details(error),
                ) from error
            raise _typed_rpc_error(error) from error

    def _describe(self) -> RemoteDescription:
        response = self._invoke(
            self._stub.Describe, throwdb_pb2.DescribeRequest()
        )
        assert isinstance(response, throwdb_pb2.DescribeResponse)
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
    def idempotency_domain(self) -> uuid.UUID | None:
        return self._description.idempotency_domain

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

    def _put(
        self,
        method: object,
        operation_name: str,
        ids: object,
        embeddings: object,
        *,
        operation_id: uuid.UUID | str | bytes | None,
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
        identifier = _operation_id(operation_id)
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
        if self._description.idempotency_domain is not None:
            request.idempotency_domain.CopyFrom(
                _proto_operation_id(self._description.idempotency_domain)
            )
        response = self._invoke_mutation(
            method,
            request,
            operation=operation_name,
            operation_id=identifier,
        )
        assert isinstance(response, throwdb_pb2.MutationResponse)
        if response.affected_count != normalized_ids.size:
            raise AmbiguousCommitError(
                identifier,
                operation_name,
                code="PROTOCOL_ERROR",
                details="mutation response count does not match request",
            )

    def insert(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: uuid.UUID | str | bytes | None = None,
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
        operation_id: uuid.UUID | str | bytes | None = None,
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
        operation_id: uuid.UUID | str | bytes | None = None,
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
        operation_id: uuid.UUID | str | bytes | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        if not self._description.writable:
            raise RuntimeError("server collection is read-only")
        normalized_ids = _normalize_numeric_ids(ids)
        if normalized_ids.size == 0:
            return np.empty(0, dtype=np.bool_)
        identifier = _operation_id(operation_id)
        request = throwdb_pb2.DeleteRequest(
            operation_id=_proto_operation_id(identifier),
            ids=_proto_ids(normalized_ids),
        )
        if self._description.idempotency_domain is not None:
            request.idempotency_domain.CopyFrom(
                _proto_operation_id(self._description.idempotency_domain)
            )
        response = self._invoke_mutation(
            self._stub.Delete,
            request,
            operation="delete",
            operation_id=identifier,
        )
        assert isinstance(response, throwdb_pb2.DeleteResponse)
        if (
            response.affected_count != normalized_ids.size
            or len(response.deleted_mask) != normalized_ids.size
        ):
            raise AmbiguousCommitError(
                identifier,
                "delete",
                code="PROTOCOL_ERROR",
                details="delete response size does not match request",
            )
        mask = np.frombuffer(response.deleted_mask, dtype=np.uint8)
        if bool(np.any(mask > 1)):
            raise AmbiguousCommitError(
                identifier,
                "delete",
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
    "CAPABILITY_SCAN_LOCATIONS",
    "CAPABILITY_WRITABLE",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_SCAN_TIMEOUT_SECONDS",
    "RemoteAlreadyExistsError",
    "RemoteAvailabilityConfiguration",
    "RemoteCancelledError",
    "RemoteCollection",
    "RemoteDataLossError",
    "RemoteDeadlineExceededError",
    "RemoteDescription",
    "RemoteEngineStatistics",
    "RemoteError",
    "RemoteFailedPreconditionError",
    "RemoteIdScan",
    "RemoteInternalError",
    "RemoteInvalidArgumentError",
    "RemoteNotFoundError",
    "RemoteProtocolError",
    "RemoteReaderConfiguration",
    "RemoteResourceExhaustedError",
    "RemoteUnavailableError",
    "RemoteUnimplementedError",
    "RemoteTuningSelection",
]
