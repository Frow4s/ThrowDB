from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import throwdb


def _vectors(rows: int, dimension: int) -> np.ndarray:
    """Return deterministic float16 payloads, including non-integer values."""
    values = np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension)
    return (values * np.float32(0.125) - np.float32(7.25)).astype(np.float16)


def _assert_same_float16_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.dtype(np.float16)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def _create_store(
    path: Path,
    ids: np.ndarray,
    vectors: np.ndarray,
    *,
    shard_size_bytes: int = 1 << 20,
) -> None:
    with throwdb.Writer.create(
        str(path),
        dimension=vectors.shape[1],
        shard_size_bytes=shard_size_bytes,
    ) as writer:
        writer.add(ids, vectors)


def test_round_trip_preserves_bits_and_query_order(tmp_path: Path) -> None:
    source_ids = np.array(
        [0, np.iinfo(np.uint64).max, 42, 9_000_000_000_000, 7],
        dtype=np.uint64,
    )
    source = _vectors(source_ids.size, 13)
    store_path = tmp_path / "store"
    _create_store(store_path, source_ids, source)

    storage = throwdb.Storage.open(str(store_path))
    assert storage.dimension == source.shape[1]
    assert storage.count == source_ids.size
    assert storage.dtype == "float16"
    assert storage.format_version == 1

    # Query order is unrelated to physical order and a query ID may be repeated.
    positions = np.array([3, 0, 4, 1, 3, 2], dtype=np.intp)
    query_ids = source_ids[positions].copy()
    actual = storage.get(query_ids)
    _assert_same_float16_bits(actual, source[positions])


def test_get_into_writes_the_caller_owned_buffer(tmp_path: Path) -> None:
    ids = np.array([901, 17, 2**63 + 3], dtype=np.uint64)
    vectors = _vectors(ids.size, 8)
    store_path = tmp_path / "store"
    _create_store(store_path, ids, vectors)
    storage = throwdb.Storage.open(str(store_path))

    query_order = np.array([2, 0, 1], dtype=np.intp)
    output = np.full((ids.size, vectors.shape[1]), np.nan, dtype=np.float16)
    result = storage.get_into(ids[query_order].copy(), output)

    # get_into mutates exactly the supplied array and does not allocate a result.
    assert result is None
    _assert_same_float16_bits(output, vectors[query_order])


def test_reads_across_many_small_shards(tmp_path: Path) -> None:
    dimension = 16
    ids = (np.arange(41, dtype=np.uint64) * np.uint64(104_729)) + np.uint64(13)
    vectors = _vectors(ids.size, dimension)
    store_path = tmp_path / "store"

    # Four records of payload per shard. The implementation may account for its
    # header separately; either interpretation still forces multiple shards.
    _create_store(
        store_path,
        ids,
        vectors,
        shard_size_bytes=4 * dimension * np.dtype(np.float16).itemsize,
    )

    order = np.array([40, 0, 20, 4, 39, 5, 17, 1, 31], dtype=np.intp)
    storage = throwdb.Storage.open(str(store_path))
    _assert_same_float16_bits(storage.get(ids[order].copy()), vectors[order])


def test_duplicate_ids_fail_the_build(tmp_path: Path) -> None:
    store_path = tmp_path / "duplicate-store"
    first_ids = np.array([10, 20, 30], dtype=np.uint64)
    second_ids = np.array([40, 20], dtype=np.uint64)

    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        with throwdb.Writer.create(
            str(store_path), dimension=4, shard_size_bytes=1 << 20
        ) as writer:
            writer.add(first_ids, _vectors(first_ids.size, 4))
            writer.add(second_ids, _vectors(second_ids.size, 4))

    with pytest.raises((OSError, RuntimeError, ValueError)):
        throwdb.Storage.open(str(store_path))


def test_missing_id_is_a_strict_error(tmp_path: Path) -> None:
    ids = np.array([11, 22, 33], dtype=np.uint64)
    vectors = _vectors(ids.size, 5)
    store_path = tmp_path / "store"
    _create_store(store_path, ids, vectors)
    storage = throwdb.Storage.open(str(store_path))

    query = np.array([11, 999, 22], dtype=np.uint64)
    with pytest.raises(KeyError):
        storage.get(query)

    output = np.zeros((query.size, vectors.shape[1]), dtype=np.float16)
    with pytest.raises(KeyError):
        storage.get_into(query, output)


@pytest.mark.parametrize(
    ("ids", "vectors", "message"),
    [
        (
            np.array([1, 2], dtype=np.int64),
            _vectors(2, 3),
            "uint64",
        ),
        (
            np.array([[1, 2]], dtype=np.uint64),
            _vectors(2, 3),
            "one-dimensional|shape|ndim",
        ),
        (
            np.array([1, 2], dtype=np.uint64),
            _vectors(2, 3).astype(np.float32),
            "float16",
        ),
        (
            np.array([1, 2], dtype=np.uint64),
            _vectors(2, 3).reshape(2, 3, 1),
            "two-dimensional|shape|ndim",
        ),
        (
            np.array([1, 2], dtype=np.uint64),
            _vectors(3, 3),
            "row|length|same",
        ),
        (
            np.array([1, 2], dtype=np.uint64),
            _vectors(2, 4),
            "dimension|column|shape",
        ),
    ],
)
def test_add_validates_numpy_contract(
    tmp_path: Path, ids: np.ndarray, vectors: np.ndarray, message: str
) -> None:
    writer = throwdb.Writer.create(
        str(tmp_path / "store"), dimension=3, shard_size_bytes=1 << 20
    )
    with pytest.raises(ValueError, match=message):
        writer.add(ids, vectors)


def test_get_into_rejects_an_invalid_output_buffer(tmp_path: Path) -> None:
    ids = np.array([5, 8], dtype=np.uint64)
    vectors = _vectors(ids.size, 3)
    store_path = tmp_path / "store"
    _create_store(store_path, ids, vectors)
    storage = throwdb.Storage.open(str(store_path))

    with pytest.raises(ValueError, match="float16"):
        storage.get_into(ids, np.empty((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape|dimension|column"):
        storage.get_into(ids, np.empty((2, 4), dtype=np.float16))

    non_contiguous = np.empty((2, 6), dtype=np.float16)[:, ::2]
    assert not non_contiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous|stride"):
        storage.get_into(ids, non_contiguous)


def test_exception_in_context_does_not_publish_store(tmp_path: Path) -> None:
    store_path = tmp_path / "aborted-store"
    ids = np.array([1, 2], dtype=np.uint64)

    with pytest.raises(RuntimeError, match="abort build"):
        with throwdb.Writer.create(
            str(store_path), dimension=3, shard_size_bytes=1 << 20
        ) as writer:
            writer.add(ids, _vectors(ids.size, 3))
            raise RuntimeError("abort build")

    with pytest.raises((OSError, RuntimeError, ValueError)):
        throwdb.Storage.open(str(store_path))


def test_explicit_abort_closes_writer_without_publishing_store(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "explicitly-aborted-store"
    ids = np.array([1, 2], dtype=np.uint64)
    writer = throwdb.Writer.create(
        str(store_path), dimension=3, shard_size_bytes=1 << 20
    )
    writer.add(ids, _vectors(ids.size, 3))

    writer.abort()
    writer.abort()  # Aborting an already closed writer is harmless.

    with pytest.raises(RuntimeError, match="closed"):
        writer.add(ids, _vectors(ids.size, 3))
    with pytest.raises((OSError, RuntimeError, ValueError)):
        throwdb.Storage.open(str(store_path))


def test_internal_io_worker_options_preserve_repeated_cross_shard_reads(
    tmp_path: Path,
) -> None:
    dimension = 8
    ids = (np.arange(29, dtype=np.uint64) * np.uint64(65_537)) + np.uint64(19)
    vectors = _vectors(ids.size, dimension)
    store_path = tmp_path / "store"
    _create_store(
        store_path,
        ids,
        vectors,
        shard_size_bytes=3 * dimension * np.dtype(np.float16).itemsize,
    )
    positions = np.array([28, 0, 14, 3, 27, 14, 1, 20, 0], dtype=np.intp)
    query = ids[positions].copy()

    for workers in (0, 1, 2, 4, 8):
        storage = throwdb.Storage.open(
            str(store_path),
            io_workers=workers,
            io_queue_capacity=0 if workers == 0 else 2,
            io_ranges_per_task=8,
        )
        _assert_same_float16_bits(storage.get(query), vectors[positions])

    for ranges_per_task in (1, 4, 8, 16):
        storage = throwdb.Storage.open(
            str(store_path),
            io_workers=4,
            io_queue_capacity=2,
            io_ranges_per_task=ranges_per_task,
        )
        _assert_same_float16_bits(storage.get(query), vectors[positions])

    # Cover adaptive scheduling boundaries, including a single-range request.
    # The implementation does not expose coordinator details through the
    # public API.
    for query_rows in (1, 2, 7, 8, 9, 16, 17):
        small_positions = positions[: min(query_rows, positions.size)]
        if query_rows > positions.size:
            small_positions = np.resize(positions, query_rows)
        storage = throwdb.Storage.open(
            str(store_path),
            io_workers=4,
            io_queue_capacity=1,
            io_ranges_per_task=8,
        )
        _assert_same_float16_bits(
            storage.get(ids[small_positions].copy()),
            vectors[small_positions],
        )

    with pytest.raises(ValueError, match="queue.*workers|workers.*queue"):
        throwdb.Storage.open(
            str(store_path),
            io_workers=0,
            io_queue_capacity=1,
        )

    with pytest.raises(ValueError, match="io_ranges_per_task|ranges.*task"):
        throwdb.Storage.open(
            str(store_path),
            io_ranges_per_task=0,
        )

    with pytest.raises(ValueError, match="io_backend"):
        throwdb.Storage.open(str(store_path), io_backend="unknown")

    for queue_depth in (0, 4097):
        with pytest.raises(ValueError, match="io_uring_queue_depth"):
            throwdb.Storage.open(
                str(store_path),
                io_uring_queue_depth=queue_depth,
            )

    for workers in (0, 65):
        with pytest.raises(ValueError, match="io_uring_workers"):
            throwdb.Storage.open(
                str(store_path),
                io_uring_workers=workers,
            )

    with pytest.raises(ValueError, match="queue_depth.*workers"):
        throwdb.Storage.open(
            str(store_path),
            io_backend="io_uring",
            io_uring_workers=2,
            io_uring_queue_depth=1,
        )

    with pytest.raises(ValueError, match="io_workers.*io_queue_capacity"):
        throwdb.Storage.open(
            str(store_path),
            io_backend="io_uring",
            io_workers=1,
        )


def test_low_level_storage_exposes_physical_range_options(
    tmp_path: Path,
) -> None:
    dimension = 8
    record_size = dimension * np.dtype(np.float16).itemsize
    ids = np.array([1, 2, 3], dtype=np.uint64)
    vectors = _vectors(ids.size, dimension)
    store_path = tmp_path / "store"
    _create_store(store_path, ids, vectors)

    storage = throwdb.Storage.open(
        store_path,
        max_read_range_bytes=record_size,
        max_gap_bytes=(1 << 64) - 1,
    )
    _assert_same_float16_bits(storage.get(ids[::-1].copy()), vectors[::-1])

    with pytest.raises(
        RuntimeError, match="max_read_range_bytes.*at least one record"
    ):
        throwdb.Storage.open(
            store_path,
            max_read_range_bytes=record_size - 1,
        )


@pytest.mark.parametrize(
    ("option", "value", "error", "message"),
    [
        ("max_read_range_bytes", False, TypeError, "not bool"),
        ("max_read_range_bytes", 1.5, TypeError, "must be an integer"),
        ("max_read_range_bytes", -1, ValueError, "between 0 and 2\\*\\*64-1"),
        ("max_read_range_bytes", 0, ValueError, "greater than zero"),
        ("max_read_range_bytes", 1 << 64, ValueError, "2\\*\\*64-1"),
        ("max_gap_bytes", True, TypeError, "not bool"),
        ("max_gap_bytes", 1.5, TypeError, "must be an integer"),
        ("max_gap_bytes", -1, ValueError, "between 0 and 2\\*\\*64-1"),
        ("max_gap_bytes", 1 << 64, ValueError, "2\\*\\*64-1"),
    ],
)
def test_low_level_storage_range_options_validate_uint64(
    tmp_path: Path,
    option: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    store_path = tmp_path / "store"
    _create_store(
        store_path,
        np.array([1], dtype=np.uint64),
        _vectors(1, 2),
    )

    with pytest.raises(error, match=message):
        throwdb.Storage.open(store_path, **{option: value})
