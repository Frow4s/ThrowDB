from __future__ import annotations

import socket
import socketserver
import struct
import threading
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import pytest

from throwdb._remote_protocol import (
    Frame,
    FrameFlag,
    Opcode,
    Status,
    encode_frame,
    encode_response_payload,
    receive_frame,
)
from throwdb._tcp_remote import (
    CAPABILITY_PREFIX_GET,
    CAPABILITY_SCAN_LOCATIONS,
    CAPABILITY_WRITABLE,
    LocatedIdBatch,
    LocatedIds,
    RemoteCollection,
)


_DESCRIBE = struct.Struct("<IIIIQQ")
_HEALTH = struct.Struct("<IIQ")
_GET_REQUEST = struct.Struct("<IIQ")
_GET_RESPONSE = struct.Struct("<IIQ")
_MUTATION_REQUEST = struct.Struct("<QQIIQ")
_DELETE_REQUEST = struct.Struct("<QQQ")
_MUTATION_RESPONSE = struct.Struct("<QQ")
_SCAN_REQUEST = struct.Struct("<IIQ")
_SCAN_RESPONSE = struct.Struct("<QQQQQQQQQII")
_LOCATION_DTYPE = np.dtype(
    [("segment_id", "<u4"), ("row_index", "<u4")], align=False
)


def _vectors(rows: int, dimension: int, *, salt: int = 0) -> np.ndarray:
    bits = (
        np.arange(rows * dimension, dtype=np.uint16) * np.uint16(7_919)
        + np.uint16(17 + salt)
    )
    return bits.reshape(rows, dimension).view(np.float16)


@dataclass
class _Row:
    embedding: np.ndarray
    segment_id: int
    row_index: int


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dimension = 5
        self.commit_sequence = 0
        self.next_row = 0
        base = _vectors(3, self.dimension, salt=1)
        self.rows = {
            identifier: _Row(base[index].copy(), 1, index)
            for index, identifier in enumerate((10, 20, 30))
        }

    def append(self, identifier: int, embedding: np.ndarray) -> None:
        row_index = self.next_row
        self.next_row += 1
        self.rows[identifier] = _Row(embedding.copy(), 2, row_index)


class _Handler(socketserver.BaseRequestHandler):
    server: object

    def _respond(
        self,
        request: Frame,
        *,
        status: Status = Status.OK,
        body: bytes = b"",
        message: str = "",
        stream: bool = False,
        end: bool = False,
    ) -> None:
        flags = FrameFlag.RESPONSE
        if status != Status.OK:
            flags |= FrameFlag.ERROR
        if stream:
            flags |= FrameFlag.STREAM
        if end:
            flags |= FrameFlag.END_STREAM
        payload = encode_response_payload(status, body, message=message)
        self.request.sendall(
            encode_frame(request.opcode, flags, request.request_id, payload)
        )

    def handle(self) -> None:
        while True:
            try:
                request = receive_frame(self.request)
            except ConnectionError:
                return
            if request.flags:
                self._respond(
                    request,
                    status=Status.PROTOCOL_ERROR,
                    message="request flags must be zero",
                )
                continue
            try:
                self._dispatch(request)
            except BaseException as error:  # pragma: no cover - test-server bug
                self._respond(
                    request,
                    status=Status.INTERNAL,
                    message=f"fake server error: {error}",
                )

    def _dispatch(self, request: Frame) -> None:
        state = self.server.state
        opcode = Opcode(request.opcode)
        if opcode == Opcode.HEALTH:
            self._respond(request, body=_HEALTH.pack(1, 0, 0))
            return
        if opcode == Opcode.DESCRIBE:
            with state.lock:
                count = len(state.rows)
            capabilities = (
                CAPABILITY_WRITABLE
                | CAPABILITY_PREFIX_GET
                | CAPABILITY_SCAN_LOCATIONS
            )
            self._respond(
                request,
                body=_DESCRIBE.pack(
                    7,
                    state.dimension,
                    1,
                    capabilities,
                    state.dimension * 2,
                    count,
                ),
            )
            return
        if opcode == Opcode.GET:
            prefix, reserved, count = _GET_REQUEST.unpack_from(request.payload)
            expected = _GET_REQUEST.size + count * 8
            if reserved != 0 or len(request.payload) != expected:
                self._respond(
                    request,
                    status=Status.INVALID_ARGUMENT,
                    message="bad get request",
                )
                return
            ids = np.frombuffer(request.payload, dtype="<u8", offset=_GET_REQUEST.size)
            with state.lock:
                missing = [int(value) for value in ids if int(value) not in state.rows]
                if missing:
                    self._respond(
                        request,
                        status=Status.NOT_FOUND,
                        message=f"missing id {missing[0]}",
                    )
                    return
                output = np.stack(
                    [state.rows[int(value)].embedding[:prefix] for value in ids]
                ) if count else np.empty((0, prefix), dtype=np.float16)
            self._respond(
                request,
                body=_GET_RESPONSE.pack(prefix, 1, count)
                + output.astype("<f2", copy=False).tobytes(),
            )
            return
        if opcode in {Opcode.INSERT, Opcode.UPDATE, Opcode.UPSERT}:
            self._put(request, opcode)
            return
        if opcode == Opcode.DELETE:
            self._delete(request)
            return
        if opcode == Opcode.SCAN_IDS:
            self._scan(request)
            return
        self._respond(
            request, status=Status.UNIMPLEMENTED, message="unsupported opcode"
        )

    def _put(self, request: Frame, opcode: Opcode) -> None:
        state = self.server.state
        op_hi, op_lo, dimension, dtype, count = _MUTATION_REQUEST.unpack_from(
            request.payload
        )
        del op_hi, op_lo
        expected = _MUTATION_REQUEST.size + count * 8 + count * dimension * 2
        if dimension != state.dimension or dtype != 1 or len(request.payload) != expected:
            self._respond(
                request, status=Status.INVALID_ARGUMENT, message="bad mutation"
            )
            return
        ids_begin = _MUTATION_REQUEST.size
        embeddings_begin = ids_begin + count * 8
        ids = np.frombuffer(request.payload, dtype="<u8", count=count, offset=ids_begin)
        embeddings = np.frombuffer(
            request.payload,
            dtype="<f2",
            count=count * dimension,
            offset=embeddings_begin,
        ).reshape((count, dimension))
        numeric = [int(value) for value in ids]
        if len(set(numeric)) != len(numeric):
            self._respond(
                request, status=Status.INVALID_ARGUMENT, message="duplicate ids"
            )
            return
        with state.lock:
            if opcode == Opcode.INSERT:
                existing = next((value for value in numeric if value in state.rows), None)
                if existing is not None:
                    self._respond(
                        request,
                        status=Status.ALREADY_EXISTS,
                        message=f"id {existing} already exists",
                    )
                    return
            if opcode == Opcode.UPDATE:
                missing = next((value for value in numeric if value not in state.rows), None)
                if missing is not None:
                    self._respond(
                        request,
                        status=Status.NOT_FOUND,
                        message=f"id {missing} is missing",
                    )
                    return
            for identifier, embedding in zip(numeric, embeddings, strict=True):
                state.append(identifier, embedding)
            state.commit_sequence += 1
            sequence = state.commit_sequence
        self._respond(request, body=_MUTATION_RESPONSE.pack(sequence, count))

    def _delete(self, request: Frame) -> None:
        state = self.server.state
        op_hi, op_lo, count = _DELETE_REQUEST.unpack_from(request.payload)
        del op_hi, op_lo
        if len(request.payload) != _DELETE_REQUEST.size + count * 8:
            self._respond(
                request, status=Status.INVALID_ARGUMENT, message="bad delete"
            )
            return
        ids = np.frombuffer(
            request.payload, dtype="<u8", count=count, offset=_DELETE_REQUEST.size
        )
        numeric = [int(value) for value in ids]
        if len(set(numeric)) != len(numeric):
            self._respond(
                request, status=Status.INVALID_ARGUMENT, message="duplicate ids"
            )
            return
        with state.lock:
            mask = bytearray(count)
            for index, identifier in enumerate(numeric):
                if identifier in state.rows:
                    mask[index] = 1
                    del state.rows[identifier]
            state.commit_sequence += 1
            sequence = state.commit_sequence
        self._respond(
            request,
            body=_MUTATION_RESPONSE.pack(sequence, count) + bytes(mask),
        )

    def _scan(self, request: Frame) -> None:
        state = self.server.state
        batch_rows, flags, reserved = _SCAN_REQUEST.unpack(request.payload)
        if batch_rows == 0 or flags & ~1 or reserved != 0:
            self._respond(
                request, status=Status.INVALID_ARGUMENT, message="bad scan"
            )
            return
        with_locations = bool(flags & 1)
        with state.lock:
            rows = sorted(
                state.rows.items(),
                key=lambda item: (item[1].segment_id, item[1].row_index),
            )
            sequence = state.commit_sequence
        total = len(rows)
        if not rows:
            chunks = [(0, [])]
        else:
            chunks = [
                (offset, rows[offset : offset + batch_rows])
                for offset in range(0, total, batch_rows)
            ]
        for chunk_index, (offset, chunk) in enumerate(chunks):
            ids = np.array([identifier for identifier, _ in chunk], dtype="<u8")
            locations = np.array(
                [(row.segment_id, row.row_index) for _, row in chunk],
                dtype=_LOCATION_DTYPE,
            )
            metadata = _SCAN_RESPONSE.pack(
                0x0123_4567_89AB_CDEF,
                0xFEDC_BA98_7654_3210,
                17,
                sequence,
                total,
                0x1111_2222_3333_4444,
                0x5555_6666_7777_8888,
                offset,
                len(chunk),
                1 if with_locations else 0,
                0,
            )
            body = metadata + ids.tobytes()
            if with_locations:
                body += locations.tobytes()
            self._respond(
                request,
                body=body,
                stream=True,
                end=chunk_index + 1 == len(chunks),
            )


class _RunningFakeServer:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.server = type("FakeServer", (), {"state": _State()})()
        self.threads: list[threading.Thread] = []

        def connect(
            address: tuple[str, int], timeout: float | None = None
        ) -> socket.socket:
            client, server = socket.socketpair()
            client.settimeout(timeout)
            server.settimeout(timeout)
            thread = threading.Thread(
                target=_Handler,
                args=(server, address, self.server),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
            return client

        monkeypatch.setattr(socket, "create_connection", connect)

    @property
    def endpoint(self) -> str:
        return "fake.throwdb.test:1"

    def close(self) -> None:
        for thread in self.threads:
            thread.join(timeout=2)

    def __enter__(self) -> _RunningFakeServer:
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


def test_remote_get_crud_and_physical_order_id_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _RunningFakeServer(monkeypatch) as server, RemoteCollection.connect(
        server.endpoint, max_connections=2
    ) as collection:
        assert collection.health() == {"serving": True, "protocol_minor": 0}
        assert collection.dimension == 5
        assert collection.dtype == "float16"
        assert collection.format_version == 7
        assert collection.mode == "read_write"
        assert collection.count == 3

        original = _vectors(3, 5, salt=1)
        np.testing.assert_array_equal(
            collection.get([30, 10]).view(np.uint16),
            original[[2, 0]].view(np.uint16),
        )
        prefix = np.empty((2, 2), dtype=np.float16)
        collection.get_into([20, 10], prefix, prefix_dimensions=2)
        np.testing.assert_array_equal(
            prefix.view(np.uint16), original[[1, 0], :2].view(np.uint16)
        )

        inserted = _vectors(2, 5, salt=101)
        updated = _vectors(1, 5, salt=201)
        upserted = _vectors(2, 5, salt=301)
        collection.insert(np.array([40, 50], dtype=np.uint32), inserted)
        collection.update([20], updated)
        collection.upsert([10, 60], upserted)
        np.testing.assert_array_equal(
            collection.delete([30, 999]), np.array([True, False])
        )
        assert collection.count == 5

        with pytest.raises(ValueError, match="already exists"):
            collection.insert([70, 40], inserted)
        with pytest.raises(KeyError, match="missing"):
            collection.update([40, 999], inserted)
        assert collection.count == 5

        batches: list[LocatedIdBatch] = []
        with collection.scan_ids(batch_rows=2, with_locations=True) as scan:
            assert scan.snapshot_count == 5
            assert scan.snapshot.collection_id == "fedcba98765432100123456789abcdef"
            batches.extend(scan)  # type: ignore[arg-type]
        assert [batch.ids.tolist() for batch in batches] == [[40, 50], [20, 10], [60]]
        locations = np.concatenate([batch.locations for batch in batches])
        assert locations["segment_id"].tolist() == [2, 2, 2, 2, 2]
        assert locations["row_index"].tolist() == [0, 1, 2, 3, 4]

        copied_ids = np.empty(1, dtype=np.uint64)
        copied_locations = np.empty(1, dtype=_LOCATION_DTYPE)
        copied: list[tuple[int, int, int]] = []
        with collection.scan_ids(batch_rows=2, with_locations=True) as scan:
            while (count := scan.read_into(copied_ids, copied_locations)) != 0:
                copied.append(
                    (
                        int(copied_ids[0]),
                        int(copied_locations["segment_id"][0]),
                        int(copied_locations["row_index"][0]),
                    )
                )
            assert scan.done is True
        assert copied == [
            (40, 2, 0),
            (50, 2, 1),
            (20, 2, 2),
            (10, 2, 3),
            (60, 2, 4),
        ]

        all_ids = collection.list_ids(batch_rows=3, with_locations=True)
        assert isinstance(all_ids, LocatedIds)
        assert all_ids.ids.tolist() == [40, 50, 20, 10, 60]
        assert all_ids.locations["row_index"].tolist() == [0, 1, 2, 3, 4]
        assert collection.list_ids(batch_rows=2).tolist() == [40, 50, 20, 10, 60]


def test_remote_empty_mutations_do_not_require_a_server_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _RunningFakeServer(monkeypatch) as server, RemoteCollection.connect(
        server.endpoint
    ) as collection:
        empty_ids = np.empty(0, dtype=np.uint64)
        empty_embeddings = np.empty((0, collection.dimension), dtype=np.float16)
        collection.insert(empty_ids, empty_embeddings)
        collection.update(empty_ids, empty_embeddings)
        collection.upsert(empty_ids, empty_embeddings)
        assert collection.delete(empty_ids).shape == (0,)
        assert collection.count == 3
