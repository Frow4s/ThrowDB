from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import throwdb


def _vectors(rows: int, dimension: int, dtype: str, *, salt: int = 0) -> np.ndarray:
    element_count = rows * dimension
    if dtype == "float16":
        bits = (
            np.arange(element_count, dtype=np.uint16) * np.uint16(7_919)
            + np.uint16(17 + salt)
        )
        return bits.reshape(rows, dimension).view(np.float16)
    bits = (
        np.arange(element_count, dtype=np.uint32) * np.uint32(2_654_435_761)
        + np.uint32(97 + salt)
    )
    return bits.reshape(rows, dimension).view(np.float32)


def _assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    unsigned = np.uint16 if expected.dtype == np.dtype(np.float16) else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))


def _build(
    path: Path,
    ids: object,
    embeddings: np.ndarray,
    *,
    shard_size_bytes: int = 1 << 20,
) -> None:
    schema = throwdb.CollectionSchema(
        dimension=embeddings.shape[1],
        dtype=embeddings.dtype.name,
    )
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=shard_size_bytes,
    ) as builder:
        builder.add(ids, embeddings)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_crud_is_bit_exact_updates_count_and_survives_reopen(
    tmp_path: Path,
    dtype: str,
) -> None:
    path = tmp_path / dtype
    dimension = 7
    base_ids = np.array([10, 20, 30], dtype=np.uint32)
    base = _vectors(3, dimension, dtype, salt=1)
    _build(path, base_ids, base)

    inserted = _vectors(2, dimension, dtype, salt=101)
    updated = _vectors(2, dimension, dtype, salt=201)
    upserted = _vectors(2, dimension, dtype, salt=301)

    with throwdb.Collection.open(path, mode="r+") as collection:
        assert collection.mode == "read_write"
        assert collection.format_version == 4
        assert collection.count == 3

        collection.insert(np.array([40, 50], dtype=np.int16), inserted)
        assert collection.count == 5

        collection.update(np.array([20, 30], dtype=np.uint8), updated)
        assert collection.count == 5

        # One existing and one absent ID exercise both sides of upsert.
        collection.upsert(np.array([10, 60], dtype=np.uint16), upserted)
        assert collection.count == 6

        query = [60, 10, 40, 30, 20, 50]
        expected = np.stack(
            [upserted[1], upserted[0], inserted[0], updated[1], updated[0], inserted[1]]
        )
        _assert_same_bits(collection.get(query), expected)

        prefix = np.empty((len(query), 3), dtype=np.dtype(dtype))
        collection.get_into(query, prefix, prefix_dimensions=3)
        _assert_same_bits(prefix, expected[:, :3].copy(order="C"))

        deleted = collection.delete(np.array([20, 999, 50], dtype=np.int32))
        assert deleted.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(deleted, [True, False, True])
        assert collection.count == 4

        # Empty mutations validate their complete Python contract but do not
        # require a native transaction or advance logical state.
        empty_ids = np.empty(0, dtype=np.uint32)
        empty_vectors = np.empty((0, dimension), dtype=np.dtype(dtype))
        collection.insert(empty_ids, empty_vectors)
        collection.update(empty_ids, empty_vectors)
        collection.upsert(empty_ids, empty_vectors)
        empty_deleted = collection.delete(empty_ids)
        assert empty_deleted.dtype == np.dtype(np.bool_)
        assert empty_deleted.shape == (0,)
        assert collection.count == 4

    with pytest.raises(RuntimeError, match="closed"):
        _ = collection.count
    with pytest.raises(RuntimeError, match="closed"):
        collection.get([10])
    with pytest.raises(RuntimeError, match="closed"):
        collection.upsert([10], upserted[:1])

    reopened = throwdb.Collection.open(path)
    assert reopened.mode == "read"
    assert reopened.count == 4
    surviving_ids = [10, 30, 40, 60]
    surviving = np.stack([upserted[0], updated[1], inserted[0], upserted[1]])
    _assert_same_bits(reopened.get(surviving_ids), surviving)
    with pytest.raises(KeyError):
        reopened.get([20])
    reopened.close()


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_insert_update_preconditions_and_duplicate_batches_are_atomic(
    tmp_path: Path,
    dtype: str,
) -> None:
    path = tmp_path / dtype
    dimension = 5
    ids = np.array([1, 2], dtype=np.uint64)
    original = _vectors(2, dimension, dtype, salt=3)
    candidate = _vectors(2, dimension, dtype, salt=401)
    _build(path, ids, original)

    with throwdb.Collection.open(path, mode="r+") as collection:
        with pytest.raises(ValueError):
            collection.insert([3, 2], candidate)
        assert collection.count == 2
        _assert_same_bits(collection.get(ids), original)
        with pytest.raises(KeyError):
            collection.get([3])

        with pytest.raises(KeyError):
            collection.update([1, 999], candidate)
        assert collection.count == 2
        _assert_same_bits(collection.get(ids), original)

        for operation in (collection.insert, collection.update, collection.upsert):
            with pytest.raises(ValueError, match="duplicate|Duplicate"):
                operation([7, 7], candidate)
        with pytest.raises(ValueError, match="duplicate|Duplicate"):
            collection.delete([1, 1])
        assert collection.count == 2
        _assert_same_bits(collection.get(ids), original)


def test_delete_is_idempotent_tombstones_base_and_allows_reinsert(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    base = _vectors(2, 6, "float16", salt=5)
    replacement = _vectors(1, 6, "float16", salt=501)
    _build(path, [11, 22], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        np.testing.assert_array_equal(collection.delete([11, 404]), [True, False])
        assert collection.count == 1
        np.testing.assert_array_equal(collection.delete([11, 404]), [False, False])
        assert collection.count == 1

        output = np.full((2, 3), np.float16(-7), dtype=np.float16)
        before = output.copy()
        with pytest.raises(KeyError):
            collection.get_into([22, 11], output, prefix_dimensions=3)
        _assert_same_bits(output, before)

        collection.insert(np.array([11], dtype=np.uint32), replacement)
        assert collection.count == 2
        _assert_same_bits(collection.get([11]), replacement)

    reopened = throwdb.Collection.open(path)
    _assert_same_bits(reopened.get([11, 22]), np.stack([replacement[0], base[1]]))
    reopened.close()


def test_python_open_recovers_torn_wal_tail_and_allows_next_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    base = _vectors(1, 6, "float32", salt=11)
    inserted = _vectors(1, 6, "float32", salt=111)
    replacement = _vectors(2, 6, "float32", salt=211)
    _build(path, [1], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.insert([2], inserted)

    wal = path / "wal" / "active.wal"
    committed_size = wal.stat().st_size
    with wal.open("ab") as stream:
        stream.write(b"torn-wal-tail")
    assert wal.stat().st_size == committed_size + 13

    # Read-only recovery exposes only committed transactions and never mutates
    # collection files as a side effect of opening them.
    with throwdb.Collection.open(path) as reader:
        assert reader.count == 2
        _assert_same_bits(reader.get([1, 2]), np.concatenate([base, inserted]))
        assert wal.stat().st_size == committed_size + 13

    # A writer repairs the ignored tail before accepting a new transaction.
    with throwdb.Collection.open(path, mode="r+") as writer:
        assert writer.count == 2
        assert wal.stat().st_size == committed_size
        writer.upsert([1, 3], replacement)
        assert writer.count == 3

    with throwdb.Collection.open(path) as recovered:
        assert recovered.count == 3
        _assert_same_bits(
            recovered.get([3, 2, 1]),
            np.stack([replacement[1], inserted[0], replacement[0]]),
        )


def test_mutations_apply_public_id_and_embedding_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    dimension = 4
    _build(path, [], np.empty((0, dimension), dtype=np.float32))
    expected = _vectors(3, dimension, "float32", salt=7)

    interleaved_ids = np.array([5, 999, 8, 999, 13, 999], dtype=np.uint32)
    ids = interleaved_ids[::2]
    assert not ids.flags.c_contiguous
    interleaved_vectors = np.empty((3, dimension * 2), dtype=np.float32)
    vectors = interleaved_vectors[:, ::2]
    vectors[:] = expected
    assert not vectors.flags.c_contiguous

    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.insert(ids, vectors)

        foreign_order = ">" if sys.byteorder == "little" else "<"
        foreign_ids = np.array([13, 5, 8], dtype=f"{foreign_order}u8")
        _assert_same_bits(collection.get(foreign_ids), expected[[2, 0, 1]])

        with pytest.raises(ValueError, match=r"ids\[1\]=-1"):
            collection.upsert(
                np.array([5, -1], dtype=np.int32),
                expected[:2],
            )
        with pytest.raises(TypeError, match="float32.*float16"):
            collection.update([5], _vectors(1, dimension, "float16"))
        with pytest.raises(ValueError, match="same number of rows"):
            collection.insert([21, 34], expected[:1])
        assert collection.count == 3
        _assert_same_bits(collection.get([5, 8, 13]), expected)


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("w", ValueError),
        ("rw", ValueError),
        ("", ValueError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_open_mode_is_strict(
    tmp_path: Path,
    mode: object,
    error: type[Exception],
) -> None:
    path = tmp_path / "store"
    _build(path, [1], _vectors(1, 2, "float16"))
    with pytest.raises(error, match="mode"):
        throwdb.Collection.open(path, mode=mode)  # type: ignore[arg-type]


def test_read_only_and_builder_reject_crud_and_context_closes_handles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    vectors = _vectors(1, 3, "float16")
    _build(path, [1], vectors)

    reader = throwdb.Collection.open(path)
    for operation in (reader.insert, reader.update, reader.upsert):
        error = pytest.raises(RuntimeError, match=r"mode='r\+'")
        with error:
            operation([2], vectors)
    error = pytest.raises(RuntimeError, match=r"mode='r\+'")
    with error:
        reader.delete([1])
    reader.close()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.get([1])

    builder = throwdb.Collection.create(
        tmp_path / "builder",
        schema=throwdb.CollectionSchema(dimension=3),
    )
    for operation in (builder.insert, builder.update, builder.upsert):
        with pytest.raises(RuntimeError, match="opened collection"):
            operation([2], vectors)
    with pytest.raises(RuntimeError, match="opened collection"):
        builder.delete([2])
    builder.abort()

    opened: throwdb.Collection
    with throwdb.Collection.open(path) as opened:
        _assert_same_bits(opened.get([1]), vectors)
    with pytest.raises(RuntimeError, match="closed"):
        opened.get([1])


def test_only_one_read_write_handle_holds_the_writer_lease(tmp_path: Path) -> None:
    path = tmp_path / "store"
    _build(path, [1], _vectors(1, 2, "float16"))

    first = throwdb.Collection.open(path, mode="r+")
    # Read-only handles do not contend with the writer lease.
    reader = throwdb.Collection.open(path)
    _assert_same_bits(reader.get([1]), _vectors(1, 2, "float16"))
    reader.close()

    with pytest.raises((OSError, RuntimeError)):
        throwdb.Collection.open(path, mode="r+")
    first.close()

    second = throwdb.Collection.open(path, mode="r+")
    second.close()


def test_read_only_open_accepts_read_only_lock_metadata(tmp_path: Path) -> None:
    path = tmp_path / "store"
    _build(path, [1], _vectors(1, 2, "float16"))
    locks = path / "locks"
    writer_lock = locks / "writer.lock"
    commit_lock = locks / "commit.lock"

    writer_lock.chmod(0o444)
    commit_lock.chmod(0o444)
    locks.chmod(0o555)
    try:
        with throwdb.Collection.open(path) as reader:
            assert reader.count == 1
            _assert_same_bits(reader.get([1]), _vectors(1, 2, "float16"))
    finally:
        locks.chmod(0o755)
        writer_lock.chmod(0o644)
        commit_lock.chmod(0o644)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_writer_is_rejected_and_close_does_not_retain_lease(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    _build(path, [], np.empty((0, 2), dtype=np.float16))
    writer = throwdb.Collection.open(
        path,
        mode="r+",
        tuning={"io_workers": 2},
    )
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(release_write)
        result = 10
        try:
            writer.insert([2], np.array([[2, 2]], dtype=np.float16))
        except RuntimeError as error:
            result = 0 if "fork" in str(error) else 11
        except BaseException:
            result = 12
        writer.close()
        try:
            os.write(ready_write, b"R")
            if os.read(release_read, 1) != b"X":
                result = 13
        except BaseException:
            result = 14
        os._exit(result)

    os.close(ready_write)
    os.close(release_read)
    child_ready = os.read(ready_read, 1) == b"R"
    successor_opened = False
    try:
        assert child_ready
        with pytest.raises(RuntimeError):
            throwdb.Collection.open(path, mode="r+")

        writer.insert([3], np.array([[3, 3]], dtype=np.float16))
        writer.close()
        successor = throwdb.Collection.open(path, mode="r+")
        successor_opened = True
        successor.close()
    finally:
        if child_ready:
            os.write(release_write, b"X")
        os.close(ready_read)
        os.close(release_write)
        waited, status = os.waitpid(pid, 0)

    assert successor_opened
    assert waited == pid
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 0
    with throwdb.Collection.open(path) as reopened:
        assert reopened.count == 1
        _assert_same_bits(
            reopened.get([3]), np.array([[3, 3]], dtype=np.float16)
        )
        with pytest.raises(KeyError):
            reopened.get([2])
