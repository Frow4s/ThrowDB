"""Synchronous Python client for one ThrowDB server collection.

``RemoteCollection`` intentionally mirrors the data-plane portion of
``Collection``.  The transport is private so a future async client or a C++
client can share the same wire protocol without changing the public Python
contract.
"""

from __future__ import annotations

import contextlib
import operator
import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Iterator, Literal, overload

import numpy as np

from ._remote_protocol import (
    DEFAULT_MAX_FRAME_BYTES,
    Frame,
    FrameFlag,
    Opcode,
    RemoteError,
    RemoteProtocolError,
    Status,
    decode_response_payload,
    encode_frame_header,
    receive_frame,
)
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


_DESCRIBE_BODY: Final = struct.Struct("<IIIIQQ")
_HEALTH_BODY: Final = struct.Struct("<IIQ")
_GET_REQUEST: Final = struct.Struct("<IIQ")
_GET_RESPONSE: Final = struct.Struct("<IIQ")
_MUTATION_REQUEST: Final = struct.Struct("<QQIIQ")
_DELETE_REQUEST: Final = struct.Struct("<QQQ")
_MUTATION_RESPONSE: Final = struct.Struct("<QQ")
_SCAN_REQUEST: Final = struct.Struct("<IIQ")
_SCAN_RESPONSE: Final = struct.Struct("<QQQQQQQQQII")

_DTYPE_TO_WIRE: Final = {
    "float16": 1,
    "float32": 2,
}
_WIRE_TO_DTYPE: Final = {value: key for key, value in _DTYPE_TO_WIRE.items()}

CAPABILITY_WRITABLE: Final = 1 << 0
CAPABILITY_PREFIX_GET: Final = 1 << 1
CAPABILITY_SCAN_LOCATIONS: Final = 1 << 2
CAPABILITY_DURABLE_IDEMPOTENCY: Final = 1 << 3
_KNOWN_CAPABILITIES: Final = (
    CAPABILITY_WRITABLE
    | CAPABILITY_PREFIX_GET
    | CAPABILITY_SCAN_LOCATIONS
    | CAPABILITY_DURABLE_IDEMPOTENCY
)

_SCAN_FLAG_LOCATIONS: Final = 1 << 0
_LOCATION_FORMAT_NONE: Final = 0
_LOCATION_FORMAT_SEGMENT_ROW_U32: Final = 1
_DEFAULT_SCAN_BATCH_ROWS: Final = 65_536
_UINT32_MAX: Final = (1 << 32) - 1

class AmbiguousCommitError(ConnectionError):
    """A mutation may have committed although its response was not received."""

    def __init__(self, operation_id: uuid.UUID, operation: str) -> None:
        self.operation_id = operation_id
        self.operation = operation
        super().__init__(
            f"remote {operation} outcome is unknown; operation_id={operation_id}. "
            "Do not retry automatically unless the server advertises durable "
            "idempotency."
        )


@dataclass(frozen=True, slots=True)
class RemoteDescription:
    format_version: int
    schema: CollectionSchema
    capabilities: int
    record_size: int
    count: int

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


def _protocol_error(frame: Frame, message: str) -> RemoteProtocolError:
    return RemoteProtocolError(
        Status.PROTOCOL_ERROR,
        message,
        opcode=frame.opcode,
        request_id=frame.request_id,
    )


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if not isinstance(endpoint, str):
        raise TypeError("endpoint must be a string")
    value = endpoint.strip()
    if not value:
        raise ValueError("endpoint cannot be empty")
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0 or closing + 1 >= len(value) or value[closing + 1] != ":":
            raise ValueError("IPv6 endpoint must use [address]:port syntax")
        host = value[1:closing]
        raw_port = value[closing + 2 :]
    else:
        host, separator, raw_port = value.rpartition(":")
        if not separator or not host:
            raise ValueError("endpoint must use host:port syntax")
    try:
        port = int(raw_port, 10)
    except ValueError as error:
        raise ValueError("endpoint port must be an integer") from error
    if not 1 <= port <= 65_535:
        raise ValueError("endpoint port must be between 1 and 65535")
    return host, port


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


def _uuid_words(value: uuid.UUID) -> tuple[int, int]:
    return value.int >> 64, value.int & 0xFFFF_FFFF_FFFF_FFFF


def _native_little_endian(array: np.ndarray, dtype: str) -> np.ndarray:
    expected = np.dtype(dtype).newbyteorder("<")
    if array.dtype == expected and array.flags.c_contiguous and array.flags.aligned:
        return array
    return np.require(array, dtype=expected, requirements=["C", "A"])


class _Connection:
    def __init__(
        self,
        address: tuple[str, int],
        *,
        timeout: float | None,
        max_frame_bytes: int,
    ) -> None:
        self._socket = socket.create_connection(address, timeout=timeout)
        self._socket.settimeout(timeout)
        self._max_frame_bytes = max_frame_bytes
        self._closed = False

    def send(
        self,
        opcode: Opcode,
        request_id: int,
        payload_parts: tuple[bytes | memoryview, ...],
    ) -> None:
        payload_bytes = sum(len(part) for part in payload_parts)
        self._socket.sendall(
            encode_frame_header(opcode, FrameFlag(0), request_id, payload_bytes)
        )
        for part in payload_parts:
            if part:
                self._socket.sendall(part)

    def receive(self) -> Frame:
        return receive_frame(
            self._socket,
            max_payload_bytes=self._max_frame_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()


class _ConnectionPool:
    def __init__(
        self,
        address: tuple[str, int],
        *,
        timeout: float | None,
        max_connections: int,
        max_frame_bytes: int,
    ) -> None:
        self._address = address
        self._timeout = timeout
        self._max_frame_bytes = max_frame_bytes
        self._slots = threading.BoundedSemaphore(max_connections)
        self._lock = threading.Lock()
        self._idle: list[_Connection] = []
        self._closed = False

    def acquire(self) -> _Connection:
        if not self._slots.acquire(timeout=self._timeout):
            raise TimeoutError("timed out waiting for a free ThrowDB connection")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("remote collection is closed")
                if self._idle:
                    return self._idle.pop()
            return _Connection(
                self._address,
                timeout=self._timeout,
                max_frame_bytes=self._max_frame_bytes,
            )
        except BaseException:
            self._slots.release()
            raise

    def release(self, connection: _Connection, *, reusable: bool) -> None:
        with self._lock:
            keep = reusable and not self._closed
            if keep:
                self._idle.append(connection)
        if not keep:
            connection.close()
        self._slots.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            idle, self._idle = self._idle, []
        for connection in idle:
            connection.close()


class _Transport:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float | None,
        max_connections: int,
        max_frame_bytes: int,
    ) -> None:
        self.endpoint = endpoint
        self._pool = _ConnectionPool(
            _parse_endpoint(endpoint),
            timeout=timeout,
            max_connections=max_connections,
            max_frame_bytes=max_frame_bytes,
        )
        self._request_lock = threading.Lock()
        self._next_request_id = 1

    def next_request_id(self) -> int:
        with self._request_lock:
            result = self._next_request_id
            self._next_request_id = (
                1 if result == 0xFFFF_FFFF_FFFF_FFFF else result + 1
            )
            return result

    def request(
        self,
        opcode: Opcode,
        payload_parts: tuple[bytes | memoryview, ...] = (),
    ) -> memoryview:
        request_id = self.next_request_id()
        connection = self._pool.acquire()
        reusable = False
        try:
            connection.send(opcode, request_id, payload_parts)
            frame = connection.receive()
            if frame.request_id != request_id:
                raise _protocol_error(frame, "response request ID does not match")
            if frame.opcode != opcode:
                raise _protocol_error(frame, "response opcode does not match")
            body = decode_response_payload(frame)
            if frame.flags & (FrameFlag.STREAM | FrameFlag.END_STREAM):
                raise _protocol_error(
                    frame, "unary response unexpectedly carries stream flags"
                )
            reusable = True
            return body
        finally:
            self._pool.release(connection, reusable=reusable)

    def begin_scan(
        self,
        payload_parts: tuple[bytes | memoryview, ...],
    ) -> tuple[_Connection, int]:
        request_id = self.next_request_id()
        connection = self._pool.acquire()
        try:
            connection.send(Opcode.SCAN_IDS, request_id, payload_parts)
        except BaseException:
            self._pool.release(connection, reusable=False)
            raise
        return connection, request_id

    def finish_scan(self, connection: _Connection, *, reusable: bool) -> None:
        self._pool.release(connection, reusable=reusable)

    def close(self) -> None:
        self._pool.close()


class RemoteIdScan(Iterator[np.ndarray | LocatedIdBatch]):
    """One server-side snapshot scan in physical segment/row order."""

    def __init__(
        self,
        transport: _Transport,
        *,
        batch_rows: int,
        with_locations: bool,
    ) -> None:
        self._transport = transport
        self._batch_rows = batch_rows
        self._with_locations = with_locations
        flags = _SCAN_FLAG_LOCATIONS if with_locations else 0
        request = _SCAN_REQUEST.pack(batch_rows, flags, 0)
        self._connection, self._request_id = transport.begin_scan((request,))
        self._released = False
        self._closed = False
        self._finished = False
        self._received_rows = 0
        self._snapshot: IdScanSnapshot | None = None
        self._pending: np.ndarray | LocatedIdBatch | None = None
        self._pending_offset = 0
        try:
            self._read_one()
        except BaseException:
            self.close()
            raise

    @property
    def snapshot(self) -> IdScanSnapshot:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("ID scan did not receive snapshot metadata")
        return snapshot

    @property
    def snapshot_count(self) -> int:
        return self.snapshot.count

    @property
    def done(self) -> bool:
        return self._finished and self._pending is None

    def _release(self, *, reusable: bool) -> None:
        if self._released:
            return
        self._released = True
        self._transport.finish_scan(self._connection, reusable=reusable)

    def _read_one(self) -> None:
        frame = self._connection.receive()
        if frame.request_id != self._request_id:
            raise _protocol_error(frame, "scan response request ID does not match")
        if frame.opcode != Opcode.SCAN_IDS:
            raise _protocol_error(frame, "scan response opcode does not match")
        body = decode_response_payload(frame)
        if not frame.flags & FrameFlag.STREAM:
            raise _protocol_error(frame, "scan response is missing the STREAM flag")
        if len(body) < _SCAN_RESPONSE.size:
            raise _protocol_error(frame, "scan response metadata is truncated")
        (
            collection_id_low,
            collection_id_high,
            catalog_generation,
            committed_sequence,
            snapshot_count,
            token_hi,
            token_lo,
            emitted_offset,
            count,
            location_format,
            reserved,
        ) = _SCAN_RESPONSE.unpack_from(body)
        if reserved != 0:
            raise _protocol_error(frame, "scan response reserved field is nonzero")
        token = f"{token_hi:016x}{token_lo:016x}"
        received_snapshot = IdScanSnapshot(
            collection_id=(
                f"{collection_id_high:016x}{collection_id_low:016x}"
            ),
            catalog_generation=catalog_generation,
            committed_sequence=committed_sequence,
            count=snapshot_count,
            token=token,
        )
        if self._snapshot is None:
            if token_hi == 0 and token_lo == 0:
                raise _protocol_error(frame, "scan response has a zero snapshot token")
            self._snapshot = received_snapshot
        elif received_snapshot != self._snapshot:
            raise _protocol_error(frame, "scan snapshot metadata changed mid-stream")
        if emitted_offset != self._received_rows:
            raise _protocol_error(frame, "scan emitted offset is not contiguous")
        if count > self._batch_rows:
            raise _protocol_error(frame, "scan chunk exceeds requested batch_rows")
        if self._received_rows + count > snapshot_count:
            raise _protocol_error(frame, "scan emits more IDs than snapshot_count")
        expected_location_format = (
            _LOCATION_FORMAT_SEGMENT_ROW_U32
            if self._with_locations
            else _LOCATION_FORMAT_NONE
        )
        if location_format != expected_location_format:
            raise _protocol_error(frame, "scan response location format does not match")

        ids_bytes = count * np.dtype("<u8").itemsize
        location_bytes = count * ID_LOCATION_DTYPE.itemsize if self._with_locations else 0
        expected_bytes = _SCAN_RESPONSE.size + ids_bytes + location_bytes
        if len(body) != expected_bytes:
            raise _protocol_error(frame, "scan response payload size does not match count")
        cursor = _SCAN_RESPONSE.size
        ids_view = np.frombuffer(body[cursor : cursor + ids_bytes], dtype="<u8")
        ids = np.array(ids_view, dtype=np.uint64, copy=True, order="C")
        cursor += ids_bytes
        if self._with_locations:
            locations_view = np.frombuffer(
                body[cursor : cursor + location_bytes], dtype=ID_LOCATION_DTYPE
            )
            locations = np.array(
                locations_view, dtype=ID_LOCATION_DTYPE, copy=True, order="C"
            )
            self._pending = LocatedIdBatch(
                ids=ids,
                locations=locations,
                emitted_offset=emitted_offset,
                snapshot=self.snapshot,
            ) if count else None
        else:
            self._pending = ids if count else None

        self._received_rows += count
        ended = bool(frame.flags & FrameFlag.END_STREAM)
        if ended:
            if self._received_rows != snapshot_count:
                raise _protocol_error(
                    frame, "final scan count does not match snapshot_count"
                )
            self._finished = True
            self._release(reusable=True)
        elif count == 0:
            raise _protocol_error(frame, "non-final scan chunk cannot be empty")

    def __iter__(self) -> RemoteIdScan:
        return self

    def __next__(self) -> np.ndarray | LocatedIdBatch:
        if self._closed:
            raise StopIteration
        if self._pending is not None:
            result = self._take_pending()
            return result
        if self._finished:
            raise StopIteration
        try:
            self._read_one()
        except BaseException:
            self.close()
            raise
        if self._pending is None:
            raise StopIteration
        return self._take_pending()

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

    def read_into(
        self,
        ids: object,
        locations: object | None = None,
    ) -> int:
        """Copy at most one server chunk into caller-owned NumPy buffers."""

        if self._closed:
            raise RuntimeError("ID scan is closed")
        if self._released and self._pending is None:
            # A completely consumed final frame remains a valid EOF cursor.
            if self._finished:
                return 0
            raise RuntimeError("ID scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
        )
        if self._pending is None:
            if self._finished:
                return 0
            try:
                self._read_one()
            except BaseException:
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
        if self._released:
            self._finished = True
            self._pending = None
            self._pending_offset = 0
            return
        self._finished = True
        self._pending = None
        self._pending_offset = 0
        # An incomplete stream has unread frames and cannot return to the pool.
        self._release(reusable=False)

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
    """Thread-safe synchronous client for one collection served by ThrowDB."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float | None = 30.0,
        max_connections: int = 4,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a number or None")
            timeout = float(timeout)
            if timeout <= 0 or not np.isfinite(timeout):
                raise ValueError("timeout must be finite and greater than zero")
        maximum_connections = _validate_positive_u32(
            max_connections, name="max_connections"
        )
        maximum_frame = _validate_positive_u32(
            max_frame_bytes, name="max_frame_bytes"
        )
        self._transport = _Transport(
            endpoint,
            timeout=timeout,
            max_connections=maximum_connections,
            max_frame_bytes=maximum_frame,
        )
        self._endpoint = endpoint
        self._state = "open"
        try:
            self._description = self._describe()
        except BaseException:
            self._transport.close()
            self._state = "closed"
            raise

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        timeout: float | None = 30.0,
        max_connections: int = 4,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> RemoteCollection:
        return cls(
            endpoint,
            timeout=timeout,
            max_connections=max_connections,
            max_frame_bytes=max_frame_bytes,
        )

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("remote collection is closed")

    def _describe(self) -> RemoteDescription:
        body = self._transport.request(Opcode.DESCRIBE)
        if len(body) != _DESCRIBE_BODY.size:
            raise RuntimeError("server returned invalid collection metadata")
        (
            format_version,
            dimension,
            wire_dtype,
            capabilities,
            record_size,
            count,
        ) = _DESCRIBE_BODY.unpack(body)
        dtype = _WIRE_TO_DTYPE.get(wire_dtype)
        if dtype is None:
            raise RuntimeError(f"server returned unknown dtype {wire_dtype}")
        schema = CollectionSchema(
            id_type="uint64",
            dimension=dimension,
            dtype=dtype,
        )
        expected_record_size = dimension * np.dtype(dtype).itemsize
        if record_size != expected_record_size:
            raise RuntimeError(
                "server record_size does not match dimension and dtype"
            )
        if capabilities & ~_KNOWN_CAPABILITIES:
            raise RuntimeError("server returned unknown capability bits")
        return RemoteDescription(
            format_version=format_version,
            schema=schema,
            capabilities=capabilities,
            record_size=record_size,
            count=count,
        )

    def describe(self) -> RemoteDescription:
        self._ensure_open()
        description = self._describe()
        if description.schema != self._description.schema:
            raise RuntimeError("server collection schema changed after connect")
        self._description = description
        return description

    def health(self) -> dict[str, int | bool]:
        self._ensure_open()
        body = self._transport.request(Opcode.HEALTH)
        if len(body) != _HEALTH_BODY.size:
            raise RuntimeError("server returned invalid health metadata")
        serving, protocol_minor, reserved = _HEALTH_BODY.unpack(body)
        if serving not in {0, 1} or reserved != 0:
            raise RuntimeError("server returned invalid health metadata")
        return {"serving": bool(serving), "protocol_minor": protocol_minor}

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
        wire_ids = _native_little_endian(normalized_ids, "<u8")
        request = _GET_REQUEST.pack(prefix, 0, normalized_ids.size)
        body = self._transport.request(
            Opcode.GET, (request, memoryview(wire_ids).cast("B"))
        )
        if len(body) < _GET_RESPONSE.size:
            raise RuntimeError("server returned a truncated get response")
        returned_dimension, wire_dtype, returned_count = _GET_RESPONSE.unpack_from(body)
        if (
            returned_dimension != prefix
            or wire_dtype != _DTYPE_TO_WIRE[self.dtype]
            or returned_count != normalized_ids.size
        ):
            raise RuntimeError("server get metadata does not match the request")
        expected_bytes = returned_count * returned_dimension * np.dtype(self.dtype).itemsize
        if len(body) != _GET_RESPONSE.size + expected_bytes:
            raise RuntimeError("server get payload size does not match its metadata")
        raw = body[_GET_RESPONSE.size :]
        view = np.frombuffer(raw, dtype=np.dtype(self.dtype).newbyteorder("<"))
        view = view.reshape((returned_count, returned_dimension))
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
        # ``get`` completes and validates the entire response before touching
        # caller memory, preserving the embedded API's failure atomicity.
        result = self.get(normalized_ids, prefix_dimensions=prefix)
        np.copyto(destination, result, casting="no")

    def _put(
        self,
        opcode: Opcode,
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
        operation = _operation_id(operation_id)
        operation_hi, operation_lo = _uuid_words(operation)
        request = _MUTATION_REQUEST.pack(
            operation_hi,
            operation_lo,
            self.dimension,
            _DTYPE_TO_WIRE[self.dtype],
            normalized_ids.size,
        )
        wire_ids = _native_little_endian(normalized_ids, "<u8")
        wire_embeddings = _native_little_endian(
            normalized_embeddings, self.dtype
        )
        try:
            body = self._transport.request(
                opcode,
                (
                    request,
                    memoryview(wire_ids).cast("B"),
                    memoryview(wire_embeddings).cast("B"),
                ),
            )
        except RemoteProtocolError as error:
            raise AmbiguousCommitError(operation, opcode.name.lower()) from error
        except RemoteError:
            raise
        except (ConnectionError, OSError, TimeoutError) as error:
            raise AmbiguousCommitError(operation, opcode.name.lower()) from error
        if len(body) != _MUTATION_RESPONSE.size:
            raise AmbiguousCommitError(operation, opcode.name.lower())
        _committed_sequence, returned_count = _MUTATION_RESPONSE.unpack(body)
        if returned_count != normalized_ids.size:
            raise AmbiguousCommitError(operation, opcode.name.lower())

    def insert(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(Opcode.INSERT, ids, embeddings, operation_id=operation_id)

    def update(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(Opcode.UPDATE, ids, embeddings, operation_id=operation_id)

    def upsert(
        self,
        ids: object,
        embeddings: object,
        *,
        operation_id: uuid.UUID | str | bytes | None = None,
    ) -> None:
        self._put(Opcode.UPSERT, ids, embeddings, operation_id=operation_id)

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
        operation = _operation_id(operation_id)
        operation_hi, operation_lo = _uuid_words(operation)
        request = _DELETE_REQUEST.pack(
            operation_hi, operation_lo, normalized_ids.size
        )
        wire_ids = _native_little_endian(normalized_ids, "<u8")
        try:
            body = self._transport.request(
                Opcode.DELETE,
                (request, memoryview(wire_ids).cast("B")),
            )
        except RemoteProtocolError as error:
            raise AmbiguousCommitError(operation, "delete") from error
        except RemoteError:
            raise
        except (ConnectionError, OSError, TimeoutError) as error:
            raise AmbiguousCommitError(operation, "delete") from error
        expected_bytes = _MUTATION_RESPONSE.size + normalized_ids.size
        if len(body) != expected_bytes:
            raise AmbiguousCommitError(operation, "delete")
        _committed_sequence, returned_count = _MUTATION_RESPONSE.unpack_from(body)
        if returned_count != normalized_ids.size:
            raise AmbiguousCommitError(operation, "delete")
        mask = np.frombuffer(body[_MUTATION_RESPONSE.size :], dtype=np.uint8)
        if bool(np.any(mask > 1)):
            raise AmbiguousCommitError(operation, "delete")
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
        return RemoteIdScan(
            self._transport,
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
        """Materialize every live ID in physical storage order.

        This method intentionally has no memory guard.  Callers working with a
        collection too large for one NumPy allocation should use ``scan_ids``.
        """

        with self.scan_ids(
            batch_rows=batch_rows, with_locations=with_locations
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
                raise RuntimeError("ID scan ended before its advertised count")
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
        self._transport.close()

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
    "ID_LOCATION_DTYPE",
    "IdScanSnapshot",
    "LocatedIdBatch",
    "LocatedIds",
    "RemoteCollection",
    "RemoteDescription",
    "RemoteIdScan",
]
