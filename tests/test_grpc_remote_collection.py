from __future__ import annotations

import threading
import uuid
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import grpc
import numpy as np
import pytest

from throwdb.remote import LocatedIdBatch, LocatedIds, RemoteCollection
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


class _FakeGrpcService(throwdb_pb2_grpc.ThrowDBServicer):
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
            protocol_minor=0,
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


class _RunningGrpcServer:
    def __init__(self) -> None:
        self.service = _FakeGrpcService()
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
            "protocol_minor": 0,
        }
        assert collection.dimension == 5
        assert collection.dtype == "float16"
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
