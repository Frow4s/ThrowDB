from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import throwdb
from throwdb.cli import main as cli_main


_FLOAT32_BITS = np.array(
    [
        0x00000000,  # +0
        0x80000000,  # -0
        0x3F800000,  # 1
        0xBF800000,  # -1
        0x00000001,  # smallest positive subnormal
        0x007FFFFF,  # largest positive subnormal
        0x7F7FFFFF,  # largest finite value
        0x7F800000,  # +inf
        0xFF800000,  # -inf
        0x7FC12345,  # quiet NaN with a payload
        0x7FA54321,  # signaling-NaN encoding with a payload
    ],
    dtype=np.uint32,
)


def _float32_vectors(rows: int, dimension: int) -> np.ndarray:
    bits = np.resize(_FLOAT32_BITS, rows * dimension).reshape(rows, dimension)
    return np.ascontiguousarray(bits).view(np.float32)


def _assert_same_float32_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.dtype(np.float32)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_native_float32_round_trip_get_into_and_validation(tmp_path: Path) -> None:
    dimension = 7
    ids = np.array([91, 7, 2**63 + 9, 44, 1_000_003, 0], dtype=np.uint64)
    vectors = _float32_vectors(ids.size, dimension)
    path = tmp_path / "native-float32"

    writer = throwdb.Writer.create(
        path,
        dimension,
        2 * dimension * np.dtype(np.float32).itemsize,
        dtype="float32",
    )
    assert writer.dtype == "float32"
    assert writer.format_version == 2
    writer.add(ids[:3], vectors[:3])
    writer.add(ids[3:], vectors[3:])
    writer.finish()

    storage = throwdb.Storage.open(path)
    assert storage.dtype == "float32"
    assert storage.dimension == dimension
    assert storage.format_version == 2

    positions = np.array([5, 0, 4, 1, 5, 2], dtype=np.intp)
    query = ids[positions].copy()
    expected = vectors[positions]
    _assert_same_float32_bits(storage.get(query), expected)

    output = np.empty(expected.shape, dtype=np.float32)
    assert storage.get_into(query, output) is None
    _assert_same_float32_bits(output, expected)

    empty = storage.get(np.empty(0, dtype=np.uint64))
    assert empty.dtype == np.dtype(np.float32)
    assert empty.shape == (0, dimension)

    with pytest.raises(ValueError, match="float32.*float16"):
        storage.get_into(query, np.empty(expected.shape, dtype=np.float16))

    invalid = throwdb.Writer.create(
        tmp_path / "invalid-payload",
        dimension=dimension,
        shard_size_bytes=1 << 20,
        dtype="float32",
    )
    with pytest.raises(ValueError, match="float32.*float16"):
        invalid.add(ids[:1], np.zeros((1, dimension), dtype=np.float16))
    invalid.abort()

    with pytest.raises(ValueError, match="float16.*float32"):
        throwdb.Writer.create(tmp_path / "invalid-dtype", dtype="float64")


def test_collection_float32_layout_endian_and_prefix_reads(tmp_path: Path) -> None:
    dimension = 9
    ids = np.array([101, 303, 707, 909], dtype=np.uint32)
    expected = _float32_vectors(ids.size, dimension)

    interleaved = np.empty((ids.size, dimension * 2), dtype=np.float32)
    embeddings = interleaved[:, ::2]
    embeddings[:] = expected
    assert not embeddings.flags.c_contiguous

    schema = throwdb.CollectionSchema(dimension=dimension, dtype="float32")
    path = tmp_path / "collection-float32"
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=2 * dimension * np.dtype(np.float32).itemsize,
    ) as builder:
        assert builder.dtype == "float32"
        assert builder.format_version == 5
        builder.add(ids, embeddings)

    collection = throwdb.Collection.open(path)
    assert collection.schema == schema
    assert collection.dtype == "float32"
    assert collection.format_version == 5

    positions = np.array([2, 0, 3, 1, 2], dtype=np.intp)
    query = ids[positions].astype(np.int64)
    prefix = 4
    result = collection.get(query, prefix_dimensions=prefix)
    assert result.flags.c_contiguous
    _assert_same_float32_bits(result, expected[positions, :prefix])

    output = np.empty((query.size, prefix), dtype=np.float32)
    assert collection.get_into(
        query, output, prefix_dimensions=prefix
    ) is None
    _assert_same_float32_bits(output, expected[positions, :prefix])

    untouched = np.full(
        (2, prefix),
        np.array([0xDEADBEEF], dtype=np.uint32).view(np.float32)[0],
        dtype=np.float32,
    )
    before = untouched.view(np.uint32).copy()
    with pytest.raises(KeyError):
        collection.get_into(
            [int(ids[0]), 999_999],
            untouched,
            prefix_dimensions=prefix,
        )
    np.testing.assert_array_equal(untouched.view(np.uint32), before)

    with pytest.raises(TypeError, match="float32.*float16"):
        collection.get_into(
            query,
            np.empty((query.size, prefix), dtype=np.float16),
            prefix_dimensions=prefix,
        )

    finite = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(8)
    foreign_order = ">" if sys.byteorder == "little" else "<"
    foreign = finite.astype(finite.dtype.newbyteorder(foreign_order))
    assert not foreign.dtype.isnative
    foreign_path = tmp_path / "foreign-endian-float32"
    with throwdb.Collection.create(
        foreign_path,
        schema=throwdb.CollectionSchema(dimension=4, dtype="float32"),
    ) as builder:
        builder.add([1, 2, 3], foreign)
    _assert_same_float32_bits(
        throwdb.Collection.open(foreign_path).get([1, 2, 3]), finite
    )


def test_cli_preserves_float32_dtype_and_bits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ids = np.array([8, 3, 99, 42], dtype=np.uint32)
    vectors = _float32_vectors(ids.size, 5)
    query = ids[[3, 0, 2]].copy()
    ids_path = tmp_path / "ids.npy"
    vectors_path = tmp_path / "vectors.npy"
    query_path = tmp_path / "query.npy"
    output_path = tmp_path / "output.npy"
    store_path = tmp_path / "cli-store"
    np.save(ids_path, ids)
    np.save(vectors_path, vectors)
    np.save(query_path, query)

    cli_main(
        [
            "create",
            "--storage",
            str(store_path),
            "--ids",
            str(ids_path),
            "--embeddings",
            str(vectors_path),
            "--batch-rows",
            "2",
        ]
    )
    cli_main(
        [
            "get",
            "--storage",
            str(store_path),
            "--ids",
            str(query_path),
            "--output",
            str(output_path),
            "--batch-rows",
            "2",
        ]
    )
    _assert_same_float32_bits(
        np.load(output_path, allow_pickle=False), vectors[[3, 0, 2]]
    )

    capsys.readouterr()
    cli_main(["inspect", "--storage", str(store_path)])
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["dtype"] == "float32"
    assert metadata["format_version"] == 5
    assert metadata["mutation_log"]["catalog_managed"] is True
    assert metadata["mutation_log"]["generation"] == 1
    assert metadata["open_stats"]["index_validation"] == "checksum"
    assert metadata["open_stats"]["index_body_checksum_verified"] is True

    cli_main(["inspect", "--storage", str(store_path), "--verify-index"])
    verified_metadata = json.loads(capsys.readouterr().out)
    assert (
        verified_metadata["open_stats"]["index_validation"]
        == "checksum_and_full"
    )
    assert (
        verified_metadata["open_stats"][
            "index_structural_validation_performed"
        ]
        is True
    )

    with throwdb.Collection.open(store_path, mode="r+") as mutable:
        mutable.update([int(ids[0])], vectors[:1])
    cli_main(["rotate-wal", "--storage", str(store_path)])
    rotation = json.loads(capsys.readouterr().out)
    assert rotation["rotated"] is True
    assert rotation["generation"] == 2
    assert rotation["committed_sequence"] == 1
    assert rotation["frozen_wal_count"] == 1
