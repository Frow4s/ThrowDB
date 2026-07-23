from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

import throwdb
from throwdb.collection import (
    EmbeddedIdScan,
    EmbeddedReadStream,
    EmbeddedRecordScan,
    _StreamingNumericIdSource,
)
from throwdb.record_scan import RecordBatch


def _vectors(
    rows: int,
    dimension: int,
    *,
    dtype: str = "float16",
    salt: int = 0,
) -> np.ndarray:
    values = (
        np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension)
        + np.float32(salt * 0.125)
    )
    return values.astype(np.dtype(dtype))


def _assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    width = np.uint16 if actual.dtype == np.dtype("float16") else np.uint32
    np.testing.assert_array_equal(actual.view(width), expected.view(width))


def _build(
    path: Path,
    ids: list[int] | np.ndarray,
    vectors: np.ndarray,
    *,
    format_version: int = 7,
    rows_per_segment: int = 2,
) -> None:
    schema = throwdb.CollectionSchema(
        dimension=vectors.shape[1],
        dtype=vectors.dtype.name,
    )
    with throwdb.Collection.create(
        path,
        schema=schema,
        format_version=format_version,
        shard_size_bytes=max(1, vectors[0].nbytes * rows_per_segment),
    ) as builder:
        builder.add(ids, vectors)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_scan_records_is_bit_exact_bounded_and_in_physical_order(
    tmp_path: Path,
    dtype: str,
) -> None:
    path = tmp_path / dtype
    base_ids = np.array([30, 10, 20, 50], dtype=np.uint64)
    base = _vectors(4, 5, dtype=dtype, salt=3)
    _build(path, base_ids, base)

    replacement = _vectors(1, 5, dtype=dtype, salt=301)
    inserted = _vectors(1, 5, dtype=dtype, salt=601)
    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.update([10], replacement)
        np.testing.assert_array_equal(collection.delete([20]), [True])
        collection.insert([40], inserted)

        prefix = 3
        bytes_per_row = 8 + prefix * np.dtype(dtype).itemsize + 8
        with collection.scan_records(
            batch_rows=100,
            max_batch_bytes=bytes_per_row * 2,
            max_snapshot_bytes=1 << 20,
            prefix_dimensions=prefix,
            with_locations=True,
        ) as scan:
            assert scan.batch_capacity == 2
            assert scan.snapshot_count == collection.count
            assert 0 < scan.snapshot_bytes <= 1 << 20
            batches = list(scan)
            assert scan.done

        assert scan.closed
        assert scan.done
        assert not scan.failed

        assert batches
        assert all(isinstance(batch, RecordBatch) for batch in batches)
        assert all(batch.ids.size <= 2 for batch in batches)
        assert all(batch.locations is not None for batch in batches)
        assert all(batch.snapshot == batches[0].snapshot for batch in batches)
        offsets = [batch.emitted_offset for batch in batches]
        assert offsets == [
            sum(previous.ids.size for previous in batches[:position])
            for position in range(len(batches))
        ]

        ids = np.concatenate([batch.ids for batch in batches])
        embeddings = np.concatenate([batch.embeddings for batch in batches])
        locations = np.concatenate(
            [batch.locations for batch in batches if batch.locations is not None]
        )
        located = collection.list_ids(with_locations=True, batch_rows=1)
        np.testing.assert_array_equal(ids, located.ids)
        np.testing.assert_array_equal(locations, located.locations)
        _assert_same_bits(
            embeddings,
            collection.get(ids)[:, :prefix].copy(order="C"),
        )


def test_record_scan_read_into_validates_buffers_and_reuses_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "read-into"
    vectors = _vectors(3, 4, salt=9)
    _build(path, [1, 2, 3], vectors, rows_per_segment=10)

    with throwdb.Collection.open(path) as collection:
        with collection.scan_records(
            batch_rows=2,
            prefix_dimensions=2,
            with_locations=False,
        ) as scan:
            ids = np.empty(5, dtype=np.uint64)
            embeddings = np.empty((5, 2), dtype=np.float16)
            assert scan.read_into(ids, embeddings) == 2
            assert scan.emitted_count == 2
            with pytest.raises(ValueError, match="with_locations=True"):
                scan.read_into(
                    ids,
                    embeddings,
                    np.empty(5, dtype=throwdb.ID_LOCATION_DTYPE),
                )
            with pytest.raises(TypeError, match="uint64"):
                scan.read_into(
                    np.empty(2, dtype=np.uint32),
                    np.empty((2, 2), dtype=np.float16),
                )
            with pytest.raises(ValueError, match="embeddings output"):
                scan.read_into(ids, np.empty((4, 2), dtype=np.float16))
            assert scan.read_into(ids, embeddings) == 1
            assert scan.done
            assert scan.read_into(ids, embeddings) == 0


def test_record_scan_rejects_every_overlapping_output_pair_before_progress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overlapping-record-outputs"
    vectors = _vectors(2, 4, salt=13)
    _build(path, [11, 22], vectors, rows_per_segment=10)

    with throwdb.Collection.open(path) as collection:
        with collection.scan_records(
            batch_rows=2,
            prefix_dimensions=2,
            with_locations=True,
        ) as scan:
            separate_ids = np.empty(2, dtype=np.uint64)
            separate_embeddings = np.empty((2, 2), dtype=np.float16)
            separate_locations = np.empty(2, dtype=throwdb.ID_LOCATION_DTYPE)

            shared = np.zeros(16, dtype=np.uint8)
            shared_ids = np.ndarray((2,), dtype=np.uint64, buffer=shared)
            shared_embeddings = np.ndarray(
                (2, 2), dtype=np.float16, buffer=shared
            )
            shared_locations = np.ndarray(
                (2,), dtype=throwdb.ID_LOCATION_DTYPE, buffer=shared
            )

            with pytest.raises(ValueError, match="ids and embeddings"):
                scan.read_into(
                    shared_ids,
                    shared_embeddings,
                    separate_locations,
                )
            with pytest.raises(ValueError, match="ids and locations"):
                scan.read_into(
                    shared_ids,
                    separate_embeddings,
                    shared_locations,
                )
            with pytest.raises(ValueError, match="embeddings and locations"):
                scan.read_into(
                    separate_ids,
                    shared_embeddings,
                    shared_locations,
                )
            assert scan.emitted_count == 0
            assert not scan.failed

            assert scan.read_into(
                separate_ids,
                separate_embeddings,
                separate_locations,
            ) == 2
            np.testing.assert_array_equal(separate_ids, [11, 22])
            _assert_same_bits(separate_embeddings, vectors[:, :2])


def test_record_scan_pins_id_and_embedding_versions_across_mutations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot"
    ids = np.array([10, 20, 30], dtype=np.uint64)
    original = _vectors(3, 4, salt=17)
    _build(path, ids, original, rows_per_segment=1)

    with throwdb.Collection.open(path, mode="r+") as collection:
        scan = collection.scan_records(batch_rows=1)
        snapshot = scan.snapshot

        updated = _vectors(1, 4, salt=701)
        inserted = _vectors(1, 4, salt=901)
        collection.update([20], updated)
        np.testing.assert_array_equal(collection.delete([30]), [True])
        collection.insert([40], inserted)

        batches = list(scan)
        scan.close()
        assert all(batch.snapshot == snapshot for batch in batches)
        scanned_ids = np.concatenate([batch.ids for batch in batches])
        scanned_vectors = np.concatenate(
            [batch.embeddings for batch in batches]
        )
        assert set(scanned_ids.tolist()) == {10, 20, 30}
        original_by_id = {
            int(identifier): row
            for identifier, row in zip(ids, original, strict=True)
        }
        expected = np.stack(
            [original_by_id[int(identifier)] for identifier in scanned_ids]
        )
        _assert_same_bits(scanned_vectors, expected)
        assert set(collection.list_ids().tolist()) == {10, 20, 40}
        _assert_same_bits(collection.get([20]), updated)


def test_stream_get_is_lazy_bounded_and_supports_memmap_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream-get"
    ids = np.array([1, 2, 3, 4], dtype=np.uint64)
    vectors = _vectors(4, 6, salt=23)
    _build(path, ids, vectors)

    source_path = tmp_path / "query.bin"
    query = np.memmap(source_path, mode="w+", dtype=np.uint32, shape=(5,))
    query[:] = [3, 1, 3, 4, 2]
    query.flush()
    prefix = 3
    bytes_per_row = 8 + prefix * np.dtype("float16").itemsize

    output_path = tmp_path / "output.bin"
    output = np.memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(5, prefix),
    )
    output[:] = np.float16(-123)

    with throwdb.Collection.open(path) as collection:
        with collection.stream_get(
            query,
            batch_rows=100,
            max_batch_bytes=bytes_per_row * 2,
            max_snapshot_bytes=1 << 20,
            prefix_dimensions=prefix,
        ) as stream:
            assert stream.batch_capacity == 2
            assert stream.requested_count == 5
            assert 0 < stream.snapshot_bytes <= 1 << 20
            produced = stream.read_into(output)
            assert produced == 2
            np.testing.assert_array_equal(
                output[produced:].view(np.uint16),
                np.full((3, prefix), np.float16(-123)).view(np.uint16),
            )
            remaining = list(stream)
            assert stream.done

        assert stream.closed
        assert stream.done
        assert not stream.failed
        actual = np.concatenate([output[:produced], *remaining])
        expected = collection.get(query)[:, :prefix].copy(order="C")
        _assert_same_bits(actual, expected)


def test_stream_get_rejects_output_overlapping_unread_ndarray_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overlapping-stream-source"
    vectors = _vectors(2, 4, salt=27)
    _build(path, [11, 22], vectors)

    shared = np.zeros(16, dtype=np.uint8)
    query = np.ndarray((2,), dtype=np.uint64, buffer=shared)
    query[:] = [11, 22]
    overlapping_output = np.ndarray(
        (1, 4),
        dtype=np.float16,
        buffer=shared,
        offset=np.dtype(np.uint64).itemsize,
    )

    with throwdb.Collection.open(path) as collection:
        with collection.stream_get(query, batch_rows=1) as stream:
            with pytest.raises(ValueError, match="current or future ID source"):
                stream.read_into(overlapping_output)
            np.testing.assert_array_equal(query, [11, 22])
            assert stream.emitted_count == 0
            assert not stream.failed

            first = next(stream)
            second = next(stream)
            _assert_same_bits(np.concatenate([first, second]), vectors)


def test_stream_get_preserves_snapshot_order_duplicates_and_partial_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream-snapshot"
    original = _vectors(3, 4, salt=29)
    _build(path, [1, 2, 3], original)

    with throwdb.Collection.open(path, mode="r+") as collection:
        query = np.array([2, 1, 2], dtype=np.uint64)
        stream = collection.stream_get(query, batch_rows=1)
        replacement = _vectors(1, 4, salt=801)
        collection.update([2], replacement)
        collection.delete([1])
        batches = list(stream)
        stream.close()
        actual = np.concatenate(batches)
        _assert_same_bits(actual, original[[1, 0, 1]])

        failing = collection.stream_get([2, 3, 999], batch_rows=2)
        first = next(failing)
        _assert_same_bits(first, np.concatenate([replacement, original[2:3]]))
        with pytest.raises(KeyError):
            next(failing)
        assert failing.failed
        assert failing.closed
        assert not failing.done
        with pytest.raises(RuntimeError, match="closed"):
            failing.read_into(np.empty((1, 4), dtype=np.float16))


def test_streaming_validation_snapshot_limits_and_format_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "limits"
    vectors = _vectors(2, 4, salt=31)
    _build(path, [1, 2], vectors)

    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.insert([3], _vectors(1, 4, salt=401))
        row_bytes = 8 + 4 * np.dtype("float16").itemsize
        with pytest.raises(ValueError, match="cannot fit one"):
            collection.stream_get([1], max_batch_bytes=row_bytes - 1)
        with pytest.raises(ValueError, match="cannot fit one"):
            collection.scan_records(
                max_batch_bytes=row_bytes - 1,
                with_locations=False,
            )
        with pytest.raises(TypeError, match="with_locations"):
            collection.scan_records(with_locations=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="prefix_dimensions"):
            collection.scan_records(prefix_dimensions=0)
        with pytest.raises(ValueError, match="max_snapshot_bytes"):
            collection.stream_get([1], max_snapshot_bytes=0)
        with pytest.raises((ValueError, RuntimeError), match="snapshot"):
            collection.scan_records(max_snapshot_bytes=1)
        with pytest.raises((ValueError, RuntimeError), match="snapshot"):
            collection.stream_get([1], max_snapshot_bytes=1)
        with pytest.raises((ValueError, RuntimeError), match="snapshot"):
            collection.scan_ids(max_snapshot_bytes=1)
        with collection.scan_ids(max_snapshot_bytes=1 << 20) as scan:
            assert 0 < scan.snapshot_bytes <= 1 << 20

    legacy = tmp_path / "legacy"
    _build(legacy, [1, 2], vectors, format_version=6)
    with throwdb.Collection.open(legacy) as collection:
        with pytest.raises(NotImplementedError, match="format-7"):
            collection.stream_get([1])
        with pytest.raises(NotImplementedError, match="format-7"):
            collection.scan_records()
        with pytest.raises(NotImplementedError, match="format-7"):
            collection.scan_ids()


def test_stream_get_detects_source_length_change_between_batches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source-change"
    vectors = _vectors(3, 3, salt=41)
    _build(path, [1, 2, 3], vectors)
    query = [1, 2, 3]

    with throwdb.Collection.open(path) as collection:
        stream = collection.stream_get(query, batch_rows=1)
        _assert_same_bits(next(stream), vectors[:1])
        query.append(1)
        with pytest.raises(RuntimeError, match="source length changed"):
            next(stream)
        assert stream.failed
        assert stream.closed
        assert not stream.done


def test_stream_get_reports_late_invalid_id_at_absolute_position(
    tmp_path: Path,
) -> None:
    path = tmp_path / "late-invalid-id"
    vectors = _vectors(3, 3, salt=43)
    _build(path, [1, 2, 3], vectors)

    with throwdb.Collection.open(path) as collection:
        stream = collection.stream_get(
            np.array([1, 2, -7], dtype=np.int32),
            batch_rows=2,
        )
        _assert_same_bits(next(stream), vectors[:2])
        with pytest.raises(ValueError, match=r"ids\[2\]=-7 is negative"):
            next(stream)
        assert stream.emitted_count == 2
        assert stream.failed and stream.closed
        assert not stream.done


def test_empty_stream_done_early_close_and_failing_output_contract(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty"
    schema = throwdb.CollectionSchema(dimension=3, dtype="float16")
    with throwdb.Collection.create(
        empty_path,
        schema=schema,
        format_version=7,
    ):
        pass

    with throwdb.Collection.open(empty_path) as collection:
        with collection.stream_get([]) as stream:
            assert stream.done
            output = np.empty((1, 3), dtype=np.float16)
            assert stream.read_into(output) == 0
        assert stream.closed and stream.done and not stream.failed
        with collection.scan_records() as scan:
            assert scan.done
            ids = np.empty(1, dtype=np.uint64)
            embeddings = np.empty((1, 3), dtype=np.float16)
            assert scan.read_into(ids, embeddings) == 0
        assert scan.closed and scan.done and not scan.failed

    path = tmp_path / "failure"
    vectors = _vectors(2, 3, salt=51)
    _build(path, [1, 2], vectors)
    with throwdb.Collection.open(path) as collection:
        stream = collection.stream_get([1, 999], batch_rows=2)
        sentinel = np.full((2, 3), np.float16(-71), dtype=np.float16)
        before = sentinel.copy()
        with pytest.raises(KeyError):
            stream.read_into(sentinel)
        _assert_same_bits(sentinel, before)
        assert stream.failed and stream.closed
        assert not stream.done

        scan = collection.scan_records(batch_rows=1)
        scan.close()
        assert scan.closed
        assert not scan.done
        with pytest.raises(RuntimeError, match="closed"):
            scan.read_into(
                np.empty(1, dtype=np.uint64),
                np.empty((1, 3), dtype=np.float16),
            )


def _blocking_snapshot_metadata() -> dict[str, int]:
    return {
        "collection_id_low": 1,
        "collection_id_high": 2,
        "catalog_generation": 3,
        "committed_sequence": 4,
        "live_count": 2,
        "snapshot_bytes": 4096,
    }


class _BlockingReadView:
    output_dimension = 2
    dtype = "float16"

    def __init__(self) -> None:
        self.metadata = _blocking_snapshot_metadata()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def get_into(self, ids: np.ndarray, output: np.ndarray) -> None:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test read view was not released")
        output[:] = ids[:, None].astype(np.float16)

    def close(self) -> None:
        self.closed = True


class _BlockingIdScan:
    done = False

    def __init__(self) -> None:
        self.metadata = _blocking_snapshot_metadata()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def read_into(
        self,
        ids: np.ndarray,
        locations: np.ndarray | None,
    ) -> int:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test ID scan was not released")
        ids[0] = 11
        if locations is not None:
            locations[0] = (1, 0)
        return 1

    def close(self) -> None:
        self.closed = True


class _BlockingRecordScan:
    output_dimension = 2
    dtype = "float16"
    done = False

    def __init__(self) -> None:
        self.metadata = _blocking_snapshot_metadata()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def read_into(
        self,
        ids: np.ndarray,
        embeddings: np.ndarray,
        locations: np.ndarray | None,
    ) -> int:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test record scan was not released")
        ids[0] = 11
        embeddings[0] = np.array([1, 2], dtype=np.float16)
        if locations is not None:
            locations[0] = (1, 0)
        return 1

    def close(self) -> None:
        self.closed = True


def test_embedded_read_stream_rejects_concurrent_consumer_and_close() -> None:
    native = _BlockingReadView()
    stream = EmbeddedReadStream(
        native,  # type: ignore[arg-type]
        source=_StreamingNumericIdSource(np.array([11, 22], dtype=np.uint64)),
        batch_rows=1,
        prefix_dimensions=2,
        dtype="float16",
    )
    results: list[np.ndarray] = []
    errors: list[BaseException] = []

    def read_first() -> None:
        try:
            results.append(next(stream))
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=read_first)
    reader.start()
    assert native.entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already in use"):
        next(stream)
    with pytest.raises(RuntimeError, match="already in use"):
        stream.close()
    assert not stream.failed
    assert not stream.closed

    native.release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()
    assert errors == []
    assert len(results) == 1
    _assert_same_bits(results[0], np.array([[11, 11]], dtype=np.float16))
    assert stream.emitted_count == 1
    stream.close()
    assert native.closed


def test_embedded_id_scan_rejects_concurrent_consumer_and_close() -> None:
    native = _BlockingIdScan()
    scan = EmbeddedIdScan(
        native,  # type: ignore[arg-type]
        batch_rows=1,
        with_locations=True,
    )
    results: list[throwdb.LocatedIdBatch] = []
    errors: list[BaseException] = []

    def read_first() -> None:
        try:
            batch = next(scan)
            assert isinstance(batch, throwdb.LocatedIdBatch)
            results.append(batch)
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=read_first)
    reader.start()
    assert native.entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already in use"):
        next(scan)
    with pytest.raises(RuntimeError, match="already in use"):
        scan.close()

    native.release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].emitted_offset == 0
    np.testing.assert_array_equal(results[0].ids, [11])
    scan.close()
    assert native.closed


def test_embedded_record_scan_rejects_concurrent_consumer_and_close() -> None:
    native = _BlockingRecordScan()
    scan = EmbeddedRecordScan(
        native,  # type: ignore[arg-type]
        batch_rows=1,
        prefix_dimensions=2,
        dtype="float16",
        with_locations=False,
    )
    results: list[RecordBatch] = []
    errors: list[BaseException] = []

    def read_first() -> None:
        try:
            results.append(next(scan))
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=read_first)
    reader.start()
    assert native.entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already in use"):
        next(scan)
    with pytest.raises(RuntimeError, match="already in use"):
        scan.close()
    assert not scan.failed
    assert not scan.closed

    native.release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].emitted_offset == 0
    np.testing.assert_array_equal(results[0].ids, [11])
    assert scan.emitted_count == 1
    scan.close()
    assert native.closed
