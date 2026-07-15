"""Low-level ThrowDB server protocol primitives.

The protocol deliberately keeps framing independent from the collection API.
This module owns only byte order, frame validation, and status translation; the
higher-level payload codecs live in :mod:`throwdb.remote`.
"""

from __future__ import annotations

import enum
import socket
import struct
from dataclasses import dataclass
from typing import Final, Protocol


MAGIC: Final = b"TDBP"
PROTOCOL_MAJOR: Final = 1
PROTOCOL_MINOR: Final = 0

_HEADER: Final = struct.Struct("<4sHHHHIQQ")
HEADER_SIZE: Final = _HEADER.size
_RESPONSE_PREFIX: Final = struct.Struct("<II")

DEFAULT_MAX_FRAME_BYTES: Final = 1 << 30


class Opcode(enum.IntEnum):
    HEALTH = 1
    DESCRIBE = 2
    GET = 3
    INSERT = 10
    UPDATE = 11
    UPSERT = 12
    DELETE = 13
    SCAN_IDS = 20


class FrameFlag(enum.IntFlag):
    RESPONSE = 1 << 0
    STREAM = 1 << 1
    END_STREAM = 1 << 2
    ERROR = 1 << 3


KNOWN_FRAME_FLAGS: Final = int(
    FrameFlag.RESPONSE
    | FrameFlag.STREAM
    | FrameFlag.END_STREAM
    | FrameFlag.ERROR
)


class Status(enum.IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    ALREADY_EXISTS = 3
    RESOURCE_EXHAUSTED = 4
    FAILED_PRECONDITION = 5
    DATA_LOSS = 6
    INTERNAL = 7
    UNIMPLEMENTED = 8
    UNAVAILABLE = 9
    CANCELLED = 10
    DEADLINE_EXCEEDED = 11
    PROTOCOL_ERROR = 12


class RemoteError(Exception):
    """Base class for an error returned by a ThrowDB server."""

    def __init__(
        self,
        status: Status,
        message: str,
        *,
        opcode: int,
        request_id: int,
    ) -> None:
        self.status = status
        self.remote_message = message
        self.opcode = opcode
        self.request_id = request_id
        super().__init__(message or status.name.lower().replace("_", " "))


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
    """A malformed frame or an explicit remote protocol error."""


_STATUS_ERRORS: Final = {
    Status.INVALID_ARGUMENT: RemoteInvalidArgumentError,
    Status.NOT_FOUND: RemoteNotFoundError,
    Status.ALREADY_EXISTS: RemoteAlreadyExistsError,
    Status.RESOURCE_EXHAUSTED: RemoteResourceExhaustedError,
    Status.FAILED_PRECONDITION: RemoteFailedPreconditionError,
    Status.DATA_LOSS: RemoteDataLossError,
    Status.INTERNAL: RemoteInternalError,
    Status.UNIMPLEMENTED: RemoteUnimplementedError,
    Status.UNAVAILABLE: RemoteUnavailableError,
    Status.CANCELLED: RemoteCancelledError,
    Status.DEADLINE_EXCEEDED: RemoteDeadlineExceededError,
    Status.PROTOCOL_ERROR: RemoteProtocolError,
}


class _SocketLike(Protocol):
    def recv_into(self, buffer: object, nbytes: int = 0) -> int: ...

    def sendall(self, data: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class Frame:
    opcode: int
    flags: FrameFlag
    request_id: int
    payload: bytes | bytearray


def encode_frame(
    opcode: int,
    flags: FrameFlag,
    request_id: int,
    payload: bytes | bytearray | memoryview = b"",
) -> bytes:
    """Encode one complete protocol frame.

    ``bytes(payload)`` is intentional: callers can release or mutate their
    request buffers immediately after this function returns.
    """

    if not 0 <= opcode <= 0xFFFF:
        raise ValueError("opcode must fit uint16")
    if not isinstance(flags, FrameFlag):
        flags = FrameFlag(flags)
    if int(flags) & ~KNOWN_FRAME_FLAGS:
        raise ValueError("frame contains unknown flags")
    if not 1 <= request_id <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("request_id must be a nonzero uint64")
    body = bytes(payload)
    return encode_frame_header(opcode, flags, request_id, len(body)) + body


def encode_frame_header(
    opcode: int,
    flags: FrameFlag,
    request_id: int,
    payload_bytes: int,
) -> bytes:
    """Encode only a frame header for scatter/gather request writes."""

    if not 0 <= opcode <= 0xFFFF:
        raise ValueError("opcode must fit uint16")
    if not isinstance(flags, FrameFlag):
        flags = FrameFlag(flags)
    if int(flags) & ~KNOWN_FRAME_FLAGS:
        raise ValueError("frame contains unknown flags")
    if not 1 <= request_id <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("request_id must be a nonzero uint64")
    if not 0 <= payload_bytes <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("payload_bytes must fit uint64")
    return _HEADER.pack(
        MAGIC,
        PROTOCOL_MAJOR,
        PROTOCOL_MINOR,
        HEADER_SIZE,
        opcode,
        int(flags),
        request_id,
        payload_bytes,
    )


def _recv_exact(stream: _SocketLike, size: int) -> bytearray:
    if size < 0:
        raise ValueError("receive size cannot be negative")
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset != size:
        try:
            received = stream.recv_into(view[offset:], size - offset)
        except socket.timeout as error:
            raise TimeoutError("timed out while receiving a ThrowDB frame") from error
        if received == 0:
            raise ConnectionError("ThrowDB server closed the connection mid-frame")
        offset += received
    return result


def receive_frame(
    stream: _SocketLike,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Frame:
    """Receive and strictly validate one frame from a stream socket."""

    if not 0 <= max_payload_bytes <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("max_payload_bytes must fit uint64")
    header = _recv_exact(stream, HEADER_SIZE)
    (
        magic,
        major,
        minor,
        header_size,
        opcode,
        raw_flags,
        request_id,
        payload_bytes,
    ) = _HEADER.unpack(header)
    if magic != MAGIC:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            f"invalid frame magic {magic!r}",
            opcode=opcode,
            request_id=request_id,
        )
    if major != PROTOCOL_MAJOR or minor > PROTOCOL_MINOR:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            f"unsupported protocol version {major}.{minor}",
            opcode=opcode,
            request_id=request_id,
        )
    if header_size != HEADER_SIZE:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            f"unsupported frame header size {header_size}",
            opcode=opcode,
            request_id=request_id,
        )
    if request_id == 0:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "server returned a zero request ID",
            opcode=opcode,
            request_id=request_id,
        )
    if raw_flags & ~KNOWN_FRAME_FLAGS:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            f"frame contains unknown flags 0x{raw_flags:x}",
            opcode=opcode,
            request_id=request_id,
        )
    if payload_bytes > max_payload_bytes:
        raise RemoteProtocolError(
            Status.RESOURCE_EXHAUSTED,
            "frame payload exceeds the configured client limit",
            opcode=opcode,
            request_id=request_id,
        )
    payload = _recv_exact(stream, payload_bytes)
    return Frame(
        opcode=opcode,
        flags=FrameFlag(raw_flags),
        request_id=request_id,
        payload=payload,
    )


def encode_response_payload(
    status: Status,
    body: bytes | bytearray | memoryview = b"",
    *,
    message: str = "",
) -> bytes:
    """Encode a response payload; primarily useful for protocol tests."""

    encoded_message = message.encode("utf-8")
    if len(encoded_message) > 0xFFFF_FFFF:
        raise ValueError("response message is too large")
    return (
        _RESPONSE_PREFIX.pack(int(status), len(encoded_message))
        + encoded_message
        + bytes(body)
    )


def decode_response_payload(frame: Frame) -> memoryview:
    """Validate a response prefix, raise a typed error, and return its body."""

    if not frame.flags & FrameFlag.RESPONSE:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "server frame is missing the RESPONSE flag",
            opcode=frame.opcode,
            request_id=frame.request_id,
        )
    if len(frame.payload) < _RESPONSE_PREFIX.size:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "response payload is shorter than its status prefix",
            opcode=frame.opcode,
            request_id=frame.request_id,
        )
    raw_status, message_bytes = _RESPONSE_PREFIX.unpack_from(frame.payload)
    message_end = _RESPONSE_PREFIX.size + message_bytes
    if message_end > len(frame.payload):
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "response message exceeds its frame payload",
            opcode=frame.opcode,
            request_id=frame.request_id,
        )
    try:
        message = frame.payload[_RESPONSE_PREFIX.size : message_end].decode(
            "utf-8", errors="strict"
        )
    except UnicodeDecodeError as error:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "response error message is not valid UTF-8",
            opcode=frame.opcode,
            request_id=frame.request_id,
        ) from error
    try:
        status = Status(raw_status)
    except ValueError as error:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            f"server returned unknown status {raw_status}",
            opcode=frame.opcode,
            request_id=frame.request_id,
        ) from error

    error_flag = bool(frame.flags & FrameFlag.ERROR)
    if status == Status.OK:
        if error_flag:
            raise RemoteProtocolError(
                Status.PROTOCOL_ERROR,
                "OK response unexpectedly carries the ERROR flag",
                opcode=frame.opcode,
                request_id=frame.request_id,
            )
        if message:
            raise RemoteProtocolError(
                Status.PROTOCOL_ERROR,
                "OK response unexpectedly carries an error message",
                opcode=frame.opcode,
                request_id=frame.request_id,
            )
        return memoryview(frame.payload)[message_end:]

    if not error_flag:
        raise RemoteProtocolError(
            Status.PROTOCOL_ERROR,
            "non-OK response is missing the ERROR flag",
            opcode=frame.opcode,
            request_id=frame.request_id,
        )
    error_type = _STATUS_ERRORS[status]
    raise error_type(
        status,
        message,
        opcode=frame.opcode,
        request_id=frame.request_id,
    )


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "Frame",
    "FrameFlag",
    "HEADER_SIZE",
    "MAGIC",
    "Opcode",
    "PROTOCOL_MAJOR",
    "PROTOCOL_MINOR",
    "RemoteAlreadyExistsError",
    "RemoteCancelledError",
    "RemoteDataLossError",
    "RemoteDeadlineExceededError",
    "RemoteError",
    "RemoteFailedPreconditionError",
    "RemoteInternalError",
    "RemoteInvalidArgumentError",
    "RemoteNotFoundError",
    "RemoteProtocolError",
    "RemoteResourceExhaustedError",
    "RemoteUnavailableError",
    "RemoteUnimplementedError",
    "Status",
    "decode_response_payload",
    "encode_frame",
    "encode_frame_header",
    "encode_response_payload",
    "receive_frame",
]
