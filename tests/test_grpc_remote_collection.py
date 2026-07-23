from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import grpc
import numpy as np
import pytest

from throwdb.remote import (
    AmbiguousCommitError,
    LocatedIdBatch,
    LocatedIds,
    RemoteAbortedError,
    RemoteCollection,
    RemoteProtocolError,
    RemoteUnavailableError,
    _IndexedIds,
)
from throwdb.v1 import throwdb_pb2, throwdb_pb2_grpc


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


class _ObservedIds(Sequence[int]):
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.indices: list[int] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        if not isinstance(index, slice):
            self.indices.append(index)
        return self.values[index]


class _BlockingIds(Sequence[int]):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        if isinstance(index, slice):
            start, stop, step = index.indices(1)
            return [self[position] for position in range(start, stop, step)]
        if index in {0, -1}:
            self.entered.set()
            self.release.wait(timeout=2)
            return 10
        raise IndexError(index)


class _FakeGrpcService(throwdb_pb2_grpc.ThrowDBServicer):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dimension = 5
        self.commit_sequence = 0
        self.next_row = 0
        self.last_stream_open: throwdb_pb2.StreamGetOpen | None = None
        self.last_scan_ids_request: throwdb_pb2.ScanIdsRequest | None = None
        self.last_scan_records_request: throwdb_pb2.ScanRecordsRequest | None = None
        base = _vectors(3, self.dimension, salt=1)
        self.rows = {
            identifier: _Row(base[index].copy(), 1, index)
            for index, identifier in enumerate((10, 20, 30))
        }

    @staticmethod
    def _ids(batch: throwdb_pb2.IdBatch) -> np.ndarray:
        if batch.WhichOneof("encoding") != "uint64_ids":
            raise ValueError("only uint64 IDs are supported")
        encoded = batch.uint64_ids
        if len(encoded.values_le) != encoded.count * 8:
            raise ValueError("invalid ID byte length")
        return np.frombuffer(encoded.values_le, dtype="<u8")

    def _append(self, identifier: int, embedding: np.ndarray) -> None:
        row_index = self.next_row
        self.next_row += 1
        self.rows[identifier] = _Row(embedding.copy(), 2, row_index)

    def Health(
        self,
        request: throwdb_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> throwdb_pb2.HealthResponse:
        del request, context
        return throwdb_pb2.HealthResponse(
            status=throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING,
            protocol_major=1,
            protocol_minor=3,
        )

    def Describe(
        self,
        request: throwdb_pb2.DescribeRequest,
        context: grpc.ServicerContext,
    ) -> throwdb_pb2.DescribeResponse:
        del request, context
        with self.lock:
            count = len(self.rows)
        return throwdb_pb2.DescribeResponse(
            format_version=7,
            schema=throwdb_pb2.CollectionSchema(
                id_type=throwdb_pb2.ID_TYPE_UINT64,
                embedding_dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                dimension=self.dimension,
            ),
            record_size=self.dimension * 2,
            count=count,
            capabilities=throwdb_pb2.Capabilities(
                writable=True,
                prefix_get=True,
                scan_locations=True,
                durable_idempotency=False,
                stream_get=True,
                scan_records=True,
            ),
            reader=throwdb_pb2.ReaderConfiguration(
                max_read_range_bytes=1 << 20,
                max_gap_bytes=0,
                io_workers=0,
                io_queue_capacity=0,
                io_ranges_per_task=8,
                io_backend=throwdb_pb2.READER_BACKEND_THREAD_POOL,
                io_uring_queue_depth=32,
                io_uring_workers=1,
            ),
            tuning=throwdb_pb2.TuningSelection(
                mode="auto",
                source="fallback",
                status="registry_missing",
                reuse_level="fallback",
                explicit_overrides=False,
            ),
            availability=throwdb_pb2.AvailabilityConfiguration(
                max_concurrent_calls=32,
                max_concurrent_scans=2,
                max_unary_inflight_bytes=256 << 20,
                max_scan_inflight_bytes=128 << 20,
                max_grpc_memory_bytes=512 << 20,
                scan_max_duration_millis=3_600_000,
                scan_idle_timeout_millis=60_000,
                shutdown_grace_millis=30_000,
                shutdown_hard_timeout_millis=60_000,
            ),
            admission_stats=throwdb_pb2.AdmissionStatistics(
                active_unary_calls=1,
                active_scans=1,
                unary_estimated_bytes=1 << 20,
                scan_estimated_bytes=2 << 20,
                admitted_unary_calls=12,
                rejected_unary_calls=3,
                admitted_scans=4,
                rejected_scans=2,
                admission_closed=False,
                draining=False,
                terminal_faulted=False,
            ),
            engine_stats=throwdb_pb2.EngineStatistics(
                catalog_generation=7,
                committed_sequence=11,
                materialized_sequence=9,
                active_wal_bytes=4096,
                active_wal_transactions=2,
                active_wal_operations=3,
                frozen_wal_count=1,
                frozen_wal_bytes=8192,
                total_wal_bytes=12288,
                total_wal_operations=5,
                overlay_entries=4,
                estimated_overlay_bytes=256,
                delta_run_count=2,
                total_delta_run_bytes=1024,
                total_delta_segment_bytes=2048,
                filesystem_available_bytes=100 << 20,
                mutation_disk_reserve_bytes=10 << 20,
            ),
        )

    def Get(
        self,
        request: throwdb_pb2.GetRequest,
        context: grpc.ServicerContext,
    ) -> throwdb_pb2.GetResponse:
        try:
            ids = self._ids(request.ids)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        prefix = request.prefix_dimensions or self.dimension
        if not 1 <= prefix <= self.dimension:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid prefix")
        with self.lock:
            missing = next((int(value) for value in ids if int(value) not in self.rows), None)
            if missing is not None:
                context.abort(grpc.StatusCode.NOT_FOUND, f"missing id {missing}")
            output = (
                np.stack([self.rows[int(value)].embedding[:prefix] for value in ids])
                if ids.size
                else np.empty((0, prefix), dtype=np.float16)
            )
        return throwdb_pb2.GetResponse(
            embeddings=throwdb_pb2.EmbeddingBatch(
                dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                dimension=prefix,
                row_count=ids.size,
                values_le=output.astype("<f2", copy=False).tobytes(),
            )
        )

    @staticmethod
    def _read_snapshot(sequence: int, count: int) -> throwdb_pb2.ReadSnapshot:
        return throwdb_pb2.ReadSnapshot(
            collection_id=throwdb_pb2.CollectionId(
                low=0x0123_4567_89AB_CDEF,
                high=0xFEDC_BA98_7654_3210,
            ),
            catalog_generation=17,
            committed_sequence=sequence,
            live_count=count,
            token=throwdb_pb2.ScanToken(
                high=0x9999_AAAA_BBBB_CCCC,
                low=0xDDDD_EEEE_FFFF_0001,
            ),
        )

    def StreamGet(self, request_iterator, context):  # type: ignore[no-untyped-def]
        try:
            first = next(request_iterator)
        except StopIteration:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing open frame")
        if first.WhichOneof("payload") != "open":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first frame is not open")
        opened = first.open
        self.last_stream_open = throwdb_pb2.StreamGetOpen()
        self.last_stream_open.CopyFrom(opened)
        prefix = opened.prefix_dimensions or self.dimension
        if not 1 <= prefix <= self.dimension:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid prefix")
        row_bytes = prefix * 2
        payload_bytes = min(opened.max_batch_payload_bytes or (1 << 20), 1 << 20)
        max_rows = min(opened.max_batch_rows or 2, 2, payload_bytes // row_bytes)
        if max_rows == 0:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "batch budget too small")
        with self.lock:
            rows = {
                identifier: _Row(
                    row.embedding.copy(),
                    row.segment_id,
                    row.row_index,
                )
                for identifier, row in self.rows.items()
            }
            sequence = self.commit_sequence
        snapshot = self._read_snapshot(sequence, len(rows))
        yield throwdb_pb2.StreamGetResponse(
            snapshot=snapshot,
            opened=throwdb_pb2.StreamGetOpened(
                dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                dimension=prefix,
                max_batch_rows=max_rows,
                max_batch_payload_bytes=payload_bytes,
                request_window=1,
            ),
        )
        expected_sequence = 0
        offset = 0
        for request in request_iterator:
            if request.WhichOneof("payload") != "batch":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unexpected open frame")
            batch = request.batch
            if batch.sequence != expected_sequence:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "non-contiguous sequence")
            try:
                ids = self._ids(batch.ids)
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            if ids.size == 0 or ids.size > max_rows:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid batch rows")
            missing = next((int(value) for value in ids if int(value) not in rows), None)
            if missing is not None:
                context.abort(grpc.StatusCode.NOT_FOUND, f"missing id {missing}")
            values = np.stack(
                [rows[int(value)].embedding[:prefix] for value in ids]
            ).astype("<f2", copy=False)
            yield throwdb_pb2.StreamGetResponse(
                snapshot=snapshot,
                batch=throwdb_pb2.StreamGetOutputBatch(
                    sequence=expected_sequence,
                    input_offset=offset,
                    embeddings=throwdb_pb2.EmbeddingBatch(
                        dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                        dimension=prefix,
                        row_count=ids.size,
                        values_le=values.tobytes(),
                    ),
                ),
            )
            expected_sequence += 1
            offset += ids.size

    def _put(
        self,
        request: throwdb_pb2.MutationRequest,
        context: grpc.ServicerContext,
        operation: str,
    ) -> throwdb_pb2.MutationResponse:
        try:
            ids = self._ids(request.ids)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        embeddings = request.embeddings
        if (
            embeddings.dtype != throwdb_pb2.EMBEDDING_DTYPE_FLOAT16
            or embeddings.dimension != self.dimension
            or embeddings.row_count != ids.size
            or len(embeddings.values_le) != ids.size * self.dimension * 2
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid embeddings")
        values = np.frombuffer(embeddings.values_le, dtype="<f2").reshape(
            (ids.size, self.dimension)
        )
        numeric = [int(value) for value in ids]
        if len(numeric) != len(set(numeric)):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "duplicate IDs")
        with self.lock:
            if operation == "insert":
                existing = next((value for value in numeric if value in self.rows), None)
                if existing is not None:
                    context.abort(
                        grpc.StatusCode.ALREADY_EXISTS,
                        f"id {existing} already exists",
                    )
            if operation == "update":
                missing = next((value for value in numeric if value not in self.rows), None)
                if missing is not None:
                    context.abort(
                        grpc.StatusCode.NOT_FOUND,
                        f"id {missing} is missing",
                    )
            for identifier, embedding in zip(numeric, values, strict=True):
                self._append(identifier, embedding)
            self.commit_sequence += 1
            sequence = self.commit_sequence
        return throwdb_pb2.MutationResponse(
            committed_sequence=sequence,
            affected_count=ids.size,
        )

    def Insert(self, request, context):  # type: ignore[no-untyped-def]
        return self._put(request, context, "insert")

    def Update(self, request, context):  # type: ignore[no-untyped-def]
        return self._put(request, context, "update")

    def Upsert(self, request, context):  # type: ignore[no-untyped-def]
        return self._put(request, context, "upsert")

    def Delete(
        self,
        request: throwdb_pb2.DeleteRequest,
        context: grpc.ServicerContext,
    ) -> throwdb_pb2.DeleteResponse:
        try:
            ids = self._ids(request.ids)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        numeric = [int(value) for value in ids]
        if len(numeric) != len(set(numeric)):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "duplicate IDs")
        mask = bytearray(ids.size)
        with self.lock:
            for index, identifier in enumerate(numeric):
                if identifier in self.rows:
                    mask[index] = 1
                    del self.rows[identifier]
            self.commit_sequence += 1
            sequence = self.commit_sequence
        return throwdb_pb2.DeleteResponse(
            committed_sequence=sequence,
            affected_count=ids.size,
            deleted_mask=bytes(mask),
        )

    def ScanIds(
        self,
        request: throwdb_pb2.ScanIdsRequest,
        context: grpc.ServicerContext,
    ):  # type: ignore[no-untyped-def]
        self.last_scan_ids_request = throwdb_pb2.ScanIdsRequest()
        self.last_scan_ids_request.CopyFrom(request)
        if request.batch_rows == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "batch_rows is zero")
        with self.lock:
            rows = sorted(
                self.rows.items(),
                key=lambda item: (item[1].segment_id, item[1].row_index),
            )
            sequence = self.commit_sequence
        chunks = (
            [rows[offset : offset + request.batch_rows]
             for offset in range(0, len(rows), request.batch_rows)]
            if rows
            else [[]]
        )
        offset = 0
        for chunk in chunks:
            numeric = np.array([identifier for identifier, _ in chunk], dtype="<u8")
            locations = np.array(
                [(row.segment_id, row.row_index) for _, row in chunk],
                dtype=_LOCATION_DTYPE,
            )
            yield throwdb_pb2.ScanIdsResponse(
                snapshot=throwdb_pb2.IdScanSnapshot(
                    collection_id=throwdb_pb2.CollectionId(
                        low=0x0123_4567_89AB_CDEF,
                        high=0xFEDC_BA98_7654_3210,
                    ),
                    catalog_generation=17,
                    committed_sequence=sequence,
                    count=len(rows),
                    token=throwdb_pb2.ScanToken(
                        high=0x1111_2222_3333_4444,
                        low=0x5555_6666_7777_8888,
                    ),
                ),
                emitted_offset=offset,
                ids=throwdb_pb2.IdBatch(
                    uint64_ids=throwdb_pb2.UInt64Ids(
                        count=len(chunk),
                        values_le=numeric.tobytes(),
                    )
                ),
                location_encoding=(
                    throwdb_pb2.LOCATION_ENCODING_SEGMENT_ROW_U32_LE
                    if request.include_locations
                    else throwdb_pb2.LOCATION_ENCODING_UNSPECIFIED
                ),
                locations_le=(locations.tobytes() if request.include_locations else b""),
            )
            offset += len(chunk)

    def ScanRecords(
        self,
        request: throwdb_pb2.ScanRecordsRequest,
        context: grpc.ServicerContext,
    ):  # type: ignore[no-untyped-def]
        self.last_scan_records_request = throwdb_pb2.ScanRecordsRequest()
        self.last_scan_records_request.CopyFrom(request)
        if request.batch_rows == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "batch_rows is zero")
        prefix = request.prefix_dimensions or self.dimension
        if not 1 <= prefix <= self.dimension:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid prefix")
        dense_row_bytes = 8 + prefix * 2 + (8 if request.include_locations else 0)
        payload_bytes = min(request.max_batch_payload_bytes or (1 << 20), 1 << 20)
        max_rows = min(request.batch_rows, 2, payload_bytes // dense_row_bytes)
        if max_rows == 0:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "batch budget too small")
        with self.lock:
            rows = sorted(
                (
                    identifier,
                    _Row(
                        row.embedding.copy(),
                        row.segment_id,
                        row.row_index,
                    ),
                )
                for identifier, row in self.rows.items()
            )
            rows.sort(key=lambda item: (item[1].segment_id, item[1].row_index))
            sequence = self.commit_sequence
        snapshot = self._read_snapshot(sequence, len(rows))
        chunks = (
            [rows[offset : offset + max_rows]
             for offset in range(0, len(rows), max_rows)]
            if rows
            else [[]]
        )
        offset = 0
        for chunk in chunks:
            ids = np.array([identifier for identifier, _ in chunk], dtype="<u8")
            embeddings = (
                np.stack([row.embedding[:prefix] for _, row in chunk]).astype(
                    "<f2",
                    copy=False,
                )
                if chunk
                else np.empty((0, prefix), dtype="<f2")
            )
            locations = np.array(
                [(row.segment_id, row.row_index) for _, row in chunk],
                dtype=_LOCATION_DTYPE,
            )
            yield throwdb_pb2.ScanRecordsResponse(
                snapshot=snapshot,
                emitted_offset=offset,
                ids=throwdb_pb2.IdBatch(
                    uint64_ids=throwdb_pb2.UInt64Ids(
                        count=len(chunk),
                        values_le=ids.tobytes(),
                    )
                ),
                embeddings=throwdb_pb2.EmbeddingBatch(
                    dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                    dimension=prefix,
                    row_count=len(chunk),
                    values_le=embeddings.tobytes(),
                ),
                location_encoding=(
                    throwdb_pb2.LOCATION_ENCODING_SEGMENT_ROW_U32_LE
                    if request.include_locations
                    else throwdb_pb2.LOCATION_ENCODING_UNSPECIFIED
                ),
                locations_le=(locations.tobytes() if request.include_locations else b""),
                max_batch_rows=max_rows,
                max_batch_payload_bytes=payload_bytes,
            )
            offset += len(chunk)


class _BlockingStreamGetService(_FakeGrpcService):
    def __init__(self) -> None:
        super().__init__()
        self.batch_received = threading.Event()
        self.release = threading.Event()

    def StreamGet(self, request_iterator, context):  # type: ignore[no-untyped-def]
        first = next(request_iterator)
        if first.WhichOneof("payload") != "open":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing open")
        snapshot = self._read_snapshot(self.commit_sequence, len(self.rows))
        yield throwdb_pb2.StreamGetResponse(
            snapshot=snapshot,
            opened=throwdb_pb2.StreamGetOpened(
                dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                dimension=self.dimension,
                max_batch_rows=2,
                max_batch_payload_bytes=1024,
                request_window=1,
            ),
        )
        next(request_iterator)
        self.batch_received.set()
        self.release.wait(timeout=2)
        if context.is_active():
            context.abort(grpc.StatusCode.UNAVAILABLE, "forced disconnect")


class _DisconnectingStreamGetService(_FakeGrpcService):
    def StreamGet(self, request_iterator, context):  # type: ignore[no-untyped-def]
        first = next(request_iterator)
        if first.WhichOneof("payload") != "open":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing open")
        snapshot = self._read_snapshot(self.commit_sequence, len(self.rows))
        yield throwdb_pb2.StreamGetResponse(
            snapshot=snapshot,
            opened=throwdb_pb2.StreamGetOpened(
                dtype=throwdb_pb2.EMBEDDING_DTYPE_FLOAT16,
                dimension=self.dimension,
                max_batch_rows=2,
                max_batch_payload_bytes=1024,
                request_window=1,
            ),
        )
        next(request_iterator)
        context.abort(grpc.StatusCode.UNAVAILABLE, "forced disconnect")


class _DisconnectingRecordScanService(_FakeGrpcService):
    def ScanRecords(self, request, context):  # type: ignore[no-untyped-def]
        responses = super().ScanRecords(request, context)
        yield next(responses)
        context.abort(grpc.StatusCode.UNAVAILABLE, "forced disconnect")


class _BlockingRecordScanService(_FakeGrpcService):
    def __init__(self) -> None:
        super().__init__()
        self.first_batch_sent = threading.Event()
        self.release = threading.Event()

    def ScanRecords(self, request, context):  # type: ignore[no-untyped-def]
        responses = super().ScanRecords(request, context)
        yield next(responses)
        self.first_batch_sent.set()
        self.release.wait(timeout=2)
        if context.is_active():
            yield from responses


class _BlockingIdScanService(_FakeGrpcService):
    def __init__(self) -> None:
        super().__init__()
        self.first_batch_sent = threading.Event()
        self.release = threading.Event()

    def ScanIds(self, request, context):  # type: ignore[no-untyped-def]
        responses = super().ScanIds(request, context)
        yield next(responses)
        self.first_batch_sent.set()
        self.release.wait(timeout=2)
        if context.is_active():
            yield from responses


class _OldGrpcService(_FakeGrpcService):
    """A production-like writable v1.0 server predating replay domains."""

    def Describe(self, request, context):  # type: ignore[no-untyped-def]
        response = super().Describe(request, context)
        response.capabilities.durable_idempotency = True
        response.ClearField("idempotency_domain")
        response.ClearField("reader")
        response.ClearField("tuning")
        response.ClearField("availability")
        response.ClearField("admission_stats")
        response.ClearField("engine_stats")
        return response

    def Health(self, request, context):  # type: ignore[no-untyped-def]
        response = super().Health(request, context)
        response.protocol_minor = 0
        return response


class _MissingReplayDomainService(_FakeGrpcService):
    """A malformed v1.1 server advertising durability without a domain."""

    def Describe(self, request, context):  # type: ignore[no-untyped-def]
        response = super().Describe(request, context)
        response.capabilities.durable_idempotency = True
        response.ClearField("idempotency_domain")
        return response


class _ResourceExhaustedMutationService(_FakeGrpcService):
    """Model an allocation failure whose commit point is not observable."""

    def Insert(self, request, context):  # type: ignore[no-untyped-def]
        del request
        context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "allocation failed after mutation admission",
        )


class _AbortedMutationService(_FakeGrpcService):
    """Model an application-level conflict with an explicit retry contract."""

    def Insert(self, request, context):  # type: ignore[no-untyped-def]
        del request
        context.abort(
            grpc.StatusCode.ABORTED,
            "an ambiguous operation must be resolved before another mutation",
        )


class _IncompatibleProtocolService(_FakeGrpcService):
    def Health(self, request, context):  # type: ignore[no-untyped-def]
        response = super().Health(request, context)
        response.protocol_major = 2
        return response


class _RunningGrpcServer:
    def __init__(self, service: _FakeGrpcService | None = None) -> None:
        self.service = service if service is not None else _FakeGrpcService()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        throwdb_pb2_grpc.add_ThrowDBServicer_to_server(
            self.service, self.server
        )
        self.path = Path("/tmp") / f"throwdb-grpc-{uuid.uuid4().hex}.sock"
        self.endpoint = f"unix:{self.path}"
        assert self.server.add_insecure_port(self.endpoint) != 0
        self.server.start()

    def close(self) -> None:
        self.server.stop(grace=0).wait(timeout=2)
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> _RunningGrpcServer:
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


def test_public_grpc_client_get_crud_scan_and_list(tmp_path: Path) -> None:
    del tmp_path
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        assert collection.health() == {
            "serving": True,
            "protocol_major": 1,
            "protocol_minor": 3,
        }
        assert collection.dimension == 5
        assert collection.dtype == "float16"
        assert collection.reader is not None
        assert collection.reader.as_dict() == {
            "max_read_range_bytes": 1 << 20,
            "max_gap_bytes": 0,
            "io_workers": 0,
            "io_queue_capacity": 0,
            "io_ranges_per_task": 8,
            "io_backend": "thread_pool",
            "io_uring_queue_depth": 32,
            "io_uring_workers": 1,
        }
        assert collection.tuning is not None
        assert collection.tuning.mode == "auto"
        assert collection.tuning.source == "fallback"
        assert collection.tuning.status == "registry_missing"
        assert collection.availability is not None
        assert collection.availability.as_dict() == {
            "max_concurrent_calls": 32,
            "max_concurrent_scans": 2,
            "max_unary_inflight_bytes": 256 << 20,
            "max_scan_inflight_bytes": 128 << 20,
            "max_grpc_memory_bytes": 512 << 20,
            "scan_max_duration_millis": 3_600_000,
            "scan_idle_timeout_millis": 60_000,
            "shutdown_grace_millis": 30_000,
            "shutdown_hard_timeout_millis": 60_000,
        }
        assert collection.admission_stats is not None
        assert collection.admission_stats.as_dict() == {
            "active_unary_calls": 1,
            "active_scans": 1,
            "unary_estimated_bytes": 1 << 20,
            "scan_estimated_bytes": 2 << 20,
            "admitted_unary_calls": 12,
            "rejected_unary_calls": 3,
            "admitted_scans": 4,
            "rejected_scans": 2,
            "admission_closed": False,
            "draining": False,
            "terminal_faulted": False,
        }
        assert collection.engine_stats is not None
        assert collection.engine_stats.catalog_generation == 7
        assert collection.engine_stats.committed_sequence == 11
        assert collection.engine_stats.total_wal_bytes == 12288
        assert collection.engine_stats.filesystem_available_bytes == 100 << 20
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
        collection.insert([40, 50], inserted)
        collection.update([20], updated)
        collection.upsert([10, 60], upserted)
        np.testing.assert_array_equal(collection.delete([30, 999]), [True, False])
        with pytest.raises(ValueError, match="already exists"):
            collection.insert([70, 40], inserted)
        with pytest.raises(KeyError, match="missing"):
            collection.update([40, 999], inserted)

        batches: list[LocatedIdBatch] = []
        with collection.scan_ids(batch_rows=2, with_locations=True) as scan:
            batches.extend(scan)  # type: ignore[arg-type]
        assert [batch.ids.tolist() for batch in batches] == [[40, 50], [20, 10], [60]]

        located = collection.list_ids(with_locations=True, batch_rows=3)
        assert isinstance(located, LocatedIds)
        assert located.ids.tolist() == [40, 50, 20, 10, 60]
        assert located.locations["row_index"].tolist() == [0, 1, 2, 3, 4]
        assert collection.list_ids(batch_rows=2).tolist() == [40, 50, 20, 10, 60]


def test_stream_get_negotiates_batches_and_normalizes_ids_lazily() -> None:
    source = _ObservedIds([30, 10, 30, 20])
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        with collection.stream_get(
            source,
            prefix_dimensions=2,
            batch_rows=4,
            max_batch_payload_bytes=1024,
        ) as stream:
            batches = list(stream)
            assert stream.done
            assert stream.rows_completed == 4
            assert stream.max_batch_rows == 2
            assert stream.max_batch_payload_bytes == 1024
            snapshot = stream.snapshot

    assert source.indices == [0, 1, 2, 3]
    assert [batch.sequence for batch in batches] == [0, 1]
    assert [batch.input_offset for batch in batches] == [0, 2]
    assert [batch.ids.tolist() for batch in batches] == [[30, 10], [30, 20]]
    assert all(batch.snapshot == snapshot for batch in batches)
    expected = _vectors(3, 5, salt=1)[[2, 0, 2, 1], :2]
    actual = np.concatenate([batch.embeddings for batch in batches])
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_stream_get_into_is_bounded_and_reports_partial_progress() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        complete = np.empty((3, 3), dtype=np.float16)
        with collection.stream_get(
            np.array([10, 20, 30], dtype=np.uint32),
            prefix_dimensions=3,
            batch_rows=3,
            max_batch_payload_bytes=1024,
        ) as stream:
            stream.get_into(complete)
            assert stream.rows_completed == 3
        np.testing.assert_array_equal(
            complete.view(np.uint16),
            _vectors(3, 5, salt=1)[:, :3].view(np.uint16),
        )

        sentinel = np.full((3, 3), np.float16(-7), dtype=np.float16)
        with collection.stream_get(
            [10, 20, 999],
            prefix_dimensions=3,
            batch_rows=3,
            max_batch_payload_bytes=1024,
        ) as stream:
            with pytest.raises(KeyError, match="missing id 999"):
                stream.get_into(sentinel)
            assert stream.rows_completed == 2
        np.testing.assert_array_equal(
            sentinel[:2].view(np.uint16),
            _vectors(3, 5, salt=1)[:2, :3].view(np.uint16),
        )
        np.testing.assert_array_equal(
            sentinel[2:],
            np.full((1, 3), np.float16(-7), dtype=np.float16),
        )


def test_stream_get_pins_snapshot_before_later_mutation() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get([10], batch_rows=1)
        replacement = _vectors(1, 5, salt=701)[0]
        with server.service.lock:
            server.service.rows[10].embedding = replacement.copy()
            server.service.commit_sequence += 1
        with stream:
            batch = next(stream)
            with pytest.raises(StopIteration):
                next(stream)
        np.testing.assert_array_equal(
            batch.embeddings.view(np.uint16),
            _vectors(3, 5, salt=1)[[0]].view(np.uint16),
        )


def test_stream_get_cancel_releases_pending_ids_without_claiming_eof() -> None:
    service = _BlockingStreamGetService()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get([10, 20], batch_rows=2)
        try:
            assert service.batch_received.wait(timeout=2)
            assert stream._producer.pending  # type: ignore[attr-defined]
            stream.close()
            assert stream.closed
            assert not stream.done
            assert not stream.failed
            assert not stream._producer.pending  # type: ignore[attr-defined]
        finally:
            service.release.set()
            stream.close()


def test_stream_get_close_does_not_wait_for_blocking_sequence_code() -> None:
    source = _BlockingIds()
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get(source, batch_rows=1)
        assert source.entered.wait(timeout=2)
        close_finished = threading.Event()

        def close_stream() -> None:
            stream.close()
            close_finished.set()

        closer = threading.Thread(target=close_stream)
        closer.start()
        try:
            assert close_finished.wait(timeout=0.5)
            assert stream.closed
            assert not stream.done
            assert not stream.failed
        finally:
            source.release.set()
            closer.join(timeout=2)
            stream.close()


def test_stream_get_rejects_concurrent_consumers_without_corrupting_state() -> None:
    service = _BlockingStreamGetService()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get([10], batch_rows=2)
        consumer_entered = threading.Event()
        original_next_response = stream._next_response  # type: ignore[attr-defined]

        def observed_next_response():  # type: ignore[no-untyped-def]
            consumer_entered.set()
            return original_next_response()

        stream._next_response = observed_next_response  # type: ignore[attr-defined]
        worker_errors: list[BaseException] = []

        def consume() -> None:
            try:
                next(stream)
            except BaseException as error:
                worker_errors.append(error)

        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert consumer_entered.wait(timeout=2)
            with pytest.raises(RuntimeError, match="one active consumer"):
                next(stream)
            assert stream.rows_completed == 0
            assert not stream.done
        finally:
            stream.close()
            service.release.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert worker_errors
        assert stream.closed
        assert not stream.done
        assert not stream.failed


def test_stream_get_rejects_output_overlapping_ndarray_id_source() -> None:
    backing = np.empty(2, dtype=np.uint64)
    ids = backing[:1]
    ids[0] = 10
    output = backing.view(np.float16)[:5].reshape(1, 5)
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get(ids, batch_rows=1)
        with pytest.raises(ValueError, match="must not overlap"):
            stream.get_into(output)
        assert stream.rows_completed == 0
        assert not stream.done
        assert not stream.failed
        stream.close()


def test_stream_get_disconnect_is_failed_not_done() -> None:
    with _RunningGrpcServer(
        _DisconnectingStreamGetService()
    ) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get([10], batch_rows=2)
        with pytest.raises(RemoteUnavailableError, match="forced disconnect"):
            next(stream)
        assert stream.closed
        assert stream.failed
        assert not stream.done


def test_empty_stream_get_reaches_clean_eof_without_a_batch() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        stream = collection.stream_get([], batch_rows=2)
        with stream:
            assert list(stream) == []
            assert stream.done
            assert not stream.closed
            assert not stream.failed
        assert stream.done
        assert stream.closed
        stream.close()


def test_scan_records_streams_physical_snapshot_with_prefix_and_locations() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        with collection.scan_records(
            batch_rows=3,
            max_batch_payload_bytes=1024,
            prefix_dimensions=3,
            with_locations=True,
        ) as scan:
            batches = list(scan)
            assert scan.done
            assert scan.rows_completed == scan.snapshot_count == 3

    assert [batch.ids.tolist() for batch in batches] == [[10, 20], [30]]
    assert [batch.emitted_offset for batch in batches] == [0, 2]
    assert all(batch.locations is not None for batch in batches)
    assert np.concatenate(
        [batch.locations["row_index"] for batch in batches]  # type: ignore[index]
    ).tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        np.concatenate([batch.embeddings for batch in batches]).view(np.uint16),
        _vectors(3, 5, salt=1)[:, :3].view(np.uint16),
    )


def test_scan_records_read_into_reuses_bounded_buffers() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        output_ids = np.empty(1, dtype=np.uint64)
        output_embeddings = np.empty((1, 2), dtype=np.float16)
        output_locations = np.empty(1, dtype=_LOCATION_DTYPE)
        observed_ids: list[int] = []
        observed_bits: list[np.ndarray] = []
        with collection.scan_records(
            batch_rows=3,
            prefix_dimensions=2,
            with_locations=True,
        ) as scan:
            while scan.read_into(
                output_ids,
                output_embeddings,
                output_locations,
            ):
                observed_ids.append(int(output_ids[0]))
                observed_bits.append(output_embeddings[0].view(np.uint16).copy())
                assert scan.rows_completed == len(observed_ids)
            assert scan.done
            assert not scan.failed
        assert scan.closed

    assert observed_ids == [10, 20, 30]
    np.testing.assert_array_equal(
        np.stack(observed_bits),
        _vectors(3, 5, salt=1)[:, :2].view(np.uint16),
    )


def test_scan_records_rejects_concurrent_consumers_without_corrupting_state() -> None:
    service = _BlockingRecordScanService()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_records(batch_rows=2)
        first = next(scan)
        assert first.ids.tolist() == [10, 20]
        consumer_entered = threading.Event()
        original_read_one = scan._read_one  # type: ignore[attr-defined]

        def observed_read_one() -> None:
            consumer_entered.set()
            original_read_one()

        scan._read_one = observed_read_one  # type: ignore[attr-defined]
        worker_errors: list[BaseException] = []

        def consume() -> None:
            try:
                next(scan)
            except BaseException as error:
                worker_errors.append(error)

        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert consumer_entered.wait(timeout=2)
            output_ids = np.empty(1, dtype=np.uint64)
            output_embeddings = np.empty((1, 5), dtype=np.float16)
            with pytest.raises(RuntimeError, match="one active consumer"):
                scan.read_into(output_ids, output_embeddings)
            assert scan.rows_completed == 2
            assert not scan.done
        finally:
            scan.close()
            service.release.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert worker_errors
        assert scan.closed
        assert not scan.done
        assert not scan.failed


@pytest.mark.parametrize(
    "overlap",
    ["ids_embeddings", "ids_locations", "embeddings_locations"],
)
def test_scan_records_rejects_overlapping_output_buffers(overlap: str) -> None:
    output_ids = np.empty(2, dtype=np.uint64)
    output_embeddings = np.empty((2, 5), dtype=np.float16)
    output_locations = np.empty(2, dtype=_LOCATION_DTYPE)
    if overlap == "ids_embeddings":
        backing = np.empty(3, dtype=np.uint64)
        output_ids = backing[:2]
        output_embeddings = backing.view(np.float16)[:10].reshape(2, 5)
    elif overlap == "ids_locations":
        backing = np.empty(2, dtype=np.uint64)
        output_ids = backing
        output_locations = backing.view(_LOCATION_DTYPE)
    else:
        backing = np.empty(3, dtype=np.uint64)
        output_embeddings = backing.view(np.float16)[:10].reshape(2, 5)
        output_locations = backing[:2].view(_LOCATION_DTYPE)

    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_records(batch_rows=2, with_locations=True)
        with pytest.raises(ValueError, match="must not overlap"):
            scan.read_into(
                output_ids,
                output_embeddings,
                output_locations,
            )
        assert scan.rows_completed == 0
        assert not scan.done
        assert not scan.failed
        scan.close()


def test_scan_records_disconnect_is_failed_not_done() -> None:
    with _RunningGrpcServer(
        _DisconnectingRecordScanService()
    ) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_records(batch_rows=2)
        first = next(scan)
        assert first.ids.tolist() == [10, 20]
        with pytest.raises(RemoteUnavailableError, match="forced disconnect"):
            next(scan)
        assert scan.closed
        assert scan.failed
        assert not scan.done


def test_empty_scan_records_reaches_clean_eof_without_a_batch() -> None:
    service = _FakeGrpcService()
    service.rows.clear()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_records(batch_rows=2)
        with scan:
            assert list(scan) == []
            assert scan.done
            assert not scan.closed
            assert not scan.failed
        assert scan.done
        assert scan.closed
        scan.close()


def test_streaming_payload_budget_must_fit_one_dense_row() -> None:
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        with pytest.raises(ValueError, match="cannot fit"):
            collection.stream_get(
                [10],
                batch_rows=1,
                max_batch_payload_bytes=1,
            )
        with pytest.raises(ValueError, match="cannot fit"):
            collection.scan_records(
                batch_rows=1,
                max_batch_payload_bytes=1,
            )


def test_streaming_requests_are_clamped_to_client_frame_limit() -> None:
    service = _FakeGrpcService()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
        max_frame_bytes=2048,
    ) as collection:
        with collection.scan_ids(
            batch_rows=1000,
            with_locations=True,
        ) as scan:
            assert sum(batch.ids.size for batch in scan) == 3
        assert service.last_scan_ids_request is not None
        assert service.last_scan_ids_request.batch_rows <= 96

        with collection.stream_get(
            [10],
            batch_rows=1000,
            max_batch_payload_bytes=1 << 20,
        ) as stream:
            assert len(list(stream)) == 1
        assert service.last_stream_open is not None
        assert service.last_stream_open.max_batch_payload_bytes == 1024
        assert service.last_stream_open.max_batch_rows <= 240

        with collection.scan_records(
            batch_rows=1000,
            max_batch_payload_bytes=1 << 20,
            with_locations=True,
        ) as scan:
            assert sum(batch.ids.size for batch in scan) == 3
        assert service.last_scan_records_request is not None
        assert service.last_scan_records_request.max_batch_payload_bytes == 1024
        assert service.last_scan_records_request.batch_rows <= 56


def test_stream_get_rejects_structurally_changed_ndarray() -> None:
    ids = np.array([10, 20, 30], dtype=np.uint64)
    source = _IndexedIds(ids)
    ids.resize((2,), refcheck=False)
    with pytest.raises(RuntimeError, match="source structure changed"):
        source.normalized_slice(0, 1)


def test_grpc_scan_read_into_and_empty_mutations(tmp_path: Path) -> None:
    del tmp_path
    with _RunningGrpcServer() as server:
        collection = RemoteCollection.connect(server.endpoint, timeout=2)
        empty_ids = np.empty(0, dtype=np.uint64)
        empty_embeddings = np.empty((0, 5), dtype=np.float16)
        collection.insert(empty_ids, empty_embeddings)
        collection.update(empty_ids, empty_embeddings)
        collection.upsert(empty_ids, empty_embeddings)
        assert collection.delete(empty_ids).shape == (0,)

        output_ids = np.empty(1, dtype=np.uint64)
        output_locations = np.empty(1, dtype=_LOCATION_DTYPE)
        copied: list[int] = []
        with collection.scan_ids(batch_rows=2, with_locations=True) as scan:
            while scan.read_into(output_ids, output_locations):
                copied.append(int(output_ids[0]))
            assert scan.done
        assert copied == [10, 20, 30]
        collection.close()


def test_scan_ids_rejects_overlapping_output_buffers() -> None:
    backing = np.empty(2, dtype=np.uint64)
    output_ids = backing
    output_locations = backing.view(_LOCATION_DTYPE)
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_ids(batch_rows=2, with_locations=True)
        with pytest.raises(ValueError, match="must not overlap"):
            scan.read_into(output_ids, output_locations)
        assert not scan.done
        assert not scan.failed
        scan.close()
        assert scan.closed
        assert not scan.done


def test_scan_ids_rejects_concurrent_consumers_and_close_unblocks_reader() -> None:
    service = _BlockingIdScanService()
    with _RunningGrpcServer(service) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        scan = collection.scan_ids(batch_rows=2, with_locations=True)
        first = next(scan)
        assert isinstance(first, LocatedIdBatch)
        assert first.ids.tolist() == [10, 20]
        consumer_entered = threading.Event()
        original_read_one = scan._read_one  # type: ignore[attr-defined]

        def observed_read_one() -> None:
            consumer_entered.set()
            original_read_one()

        scan._read_one = observed_read_one  # type: ignore[attr-defined]
        worker_errors: list[BaseException] = []

        def consume() -> None:
            try:
                next(scan)
            except BaseException as error:
                worker_errors.append(error)

        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert consumer_entered.wait(timeout=2)
            with pytest.raises(RuntimeError, match="one active consumer"):
                next(scan)
            assert not scan.done
        finally:
            scan.close()
            service.release.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert worker_errors
        assert scan.closed
        assert not scan.done
        assert not scan.failed


@pytest.mark.parametrize(
    ("use_http_proxy", "expected"),
    [(False, 0), (True, 1)],
)
def test_http_proxy_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    use_http_proxy: bool,
    expected: int,
) -> None:
    real_insecure_channel = grpc.insecure_channel
    observed: list[tuple[str, int]] = []

    def recording_insecure_channel(  # type: ignore[no-untyped-def]
        endpoint,
        *,
        options,
    ):
        observed.extend(options)
        return real_insecure_channel(endpoint, options=options)

    monkeypatch.setattr(grpc, "insecure_channel", recording_insecure_channel)
    with _RunningGrpcServer() as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
        use_http_proxy=use_http_proxy,
    ):
        pass

    assert ("grpc.enable_http_proxy", expected) in observed


def test_use_http_proxy_must_be_bool() -> None:
    with pytest.raises(TypeError, match="use_http_proxy must be a bool"):
        RemoteCollection.connect("localhost:1", use_http_proxy=1)  # type: ignore[arg-type]


def test_resource_exhausted_mutation_is_ambiguous() -> None:
    operation_id = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with _RunningGrpcServer(
        _ResourceExhaustedMutationService()
    ) as server, RemoteCollection.connect(server.endpoint, timeout=2) as collection:
        with pytest.raises(AmbiguousCommitError) as raised:
            collection.insert(
                [40],
                _vectors(1, 5, salt=401),
                operation_id=operation_id,
            )

    assert raised.value.operation_id == operation_id
    assert raised.value.code == "RESOURCE_EXHAUSTED"
    assert not isinstance(raised.value, RemoteAbortedError)


def test_aborted_mutation_has_typed_same_operation_id_retry_contract() -> None:
    operation_id = uuid.UUID("22345678-1234-5678-9abc-def012345678")
    with _RunningGrpcServer(
        _AbortedMutationService()
    ) as server, RemoteCollection.connect(server.endpoint, timeout=2) as collection:
        durable_idempotency = collection.durable_idempotency
        with pytest.raises(RemoteAbortedError) as raised:
            collection.insert(
                [40],
                _vectors(1, 5, salt=402),
                operation_id=operation_id,
            )

    error = raised.value
    assert isinstance(error, AmbiguousCommitError)
    assert durable_idempotency is False
    assert error.operation_id == operation_id
    assert error.operation == "insert"
    assert error.code == "ABORTED"
    assert error.retryable is True
    assert error.same_operation_id_required is True
    assert error.definitely_not_committed is True
    assert "before durable PREPARE" in str(error)
    assert "an ambiguous operation must be resolved" in str(error)
    assert "same operation_id" in str(error)
    assert "never allocate a new operation_id" in str(error)


def test_protocol_v1_server_without_tuning_diagnostics_is_compatible() -> None:
    with _RunningGrpcServer(_OldGrpcService()) as server, RemoteCollection.connect(
        server.endpoint,
        timeout=2,
    ) as collection:
        assert collection.durable_idempotency is True
        assert collection.idempotency_domain is None
        assert collection.reader is None
        assert collection.tuning is None
        assert collection.availability is None
        assert collection.admission_stats is None
        assert collection.engine_stats is None
        collection.insert([40], _vectors(1, 5, salt=404))
        np.testing.assert_array_equal(
            collection.get([40]),
            _vectors(1, 5, salt=404),
        )


def test_protocol_v1_1_durable_server_without_replay_domain_is_rejected() -> None:
    with _RunningGrpcServer(_MissingReplayDomainService()) as server:
        with pytest.raises(RemoteProtocolError, match="omitted.*replay domain"):
            RemoteCollection.connect(server.endpoint, timeout=2)


def test_protocol_major_mismatch_fails_during_connect() -> None:
    with _RunningGrpcServer(_IncompatibleProtocolService()) as server:
        with pytest.raises(RemoteProtocolError, match="protocol major"):
            RemoteCollection.connect(server.endpoint, timeout=2)
