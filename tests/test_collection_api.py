from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import throwdb
import throwdb.collection as collection_module
from throwdb.collection import _normalize_numeric_ids


_INTEGER_DTYPES = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)


def _vectors(rows: int, dimension: int) -> np.ndarray:
    """Return payloads whose exact float16 bit patterns are easy to compare."""

    bits = (
        np.arange(rows * dimension, dtype=np.uint16) * np.uint16(7_919)
        + np.uint16(17)
    )
    return bits.reshape(rows, dimension).view(np.float16)


def _assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.dtype(np.float16)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def _create_collection(
    path: Path,
    ids: object,
    vectors: np.ndarray,
    *,
    segment_size_bytes: int = 1 << 20,
) -> None:
    schema = throwdb.CollectionSchema(
        id_type="uint64",
        dimension=vectors.shape[1],
        dtype="float16",
    )
    with throwdb.Collection.create(
        path,
        schema=schema,
        segment_size_bytes=segment_size_bytes,
        format_version=7,
    ) as collection:
        collection.add(ids, vectors)


def _assert_store_is_unreadable(path: Path) -> None:
    with pytest.raises((OSError, RuntimeError, ValueError)):
        throwdb.Collection.open(path)


@pytest.mark.parametrize("id_dtype", _INTEGER_DTYPES, ids=lambda dtype: dtype.__name__)
def test_collection_accepts_every_supported_numpy_integer_width(
    tmp_path: Path,
    id_dtype: type[np.generic],
) -> None:
    ids = np.array([1, 19, 101], dtype=id_dtype)
    vectors = _vectors(ids.size, 7)
    path = tmp_path / f"store-{id_dtype.__name__}"

    _create_collection(path, ids, vectors)

    collection = throwdb.Collection.open(path)
    # Reversing makes the query non-contiguous as well as preserving its input
    # dtype, so the public path must normalize before entering native code.
    query = ids[::-1]
    assert not query.flags.c_contiguous
    _assert_same_bits(collection.get(query), vectors[::-1])


def test_python_integer_sequences_support_the_complete_uint64_domain(
    tmp_path: Path,
) -> None:
    ids = [0, (1 << 63) + 7, (1 << 64) - 1]
    vectors = _vectors(len(ids), 5)
    path = tmp_path / "store"

    _create_collection(path, ids, vectors)
    collection = throwdb.Collection.open(path)

    _assert_same_bits(collection.get(tuple(reversed(ids))), vectors[::-1])
    assert _normalize_numeric_ids([]).dtype == np.dtype(np.uint64)
    assert _normalize_numeric_ids([]).shape == (0,)


def test_native_aligned_contiguous_uint64_normalization_is_zero_copy() -> None:
    ids = np.array([3, 8, 13], dtype=np.uint64)

    normalized = _normalize_numeric_ids(ids)

    assert normalized is ids


def test_endian_noncontiguous_and_unaligned_ids_are_safely_normalized(
    tmp_path: Path,
) -> None:
    canonical = np.array([5, 21, 89], dtype=np.uint64)
    vectors = _vectors(canonical.size, 4)
    path = tmp_path / "store"
    _create_collection(path, canonical, vectors)
    collection = throwdb.Collection.open(path)

    foreign_prefix = ">" if sys.byteorder == "little" else "<"
    foreign_endian = np.array(canonical, dtype=f"{foreign_prefix}u8")
    assert not foreign_endian.dtype.isnative

    interleaved = np.array([5, 999, 21, 999, 89, 999], dtype=np.uint32)
    noncontiguous = interleaved[::2]
    assert not noncontiguous.flags.c_contiguous

    backing = np.zeros(1 + canonical.nbytes, dtype=np.uint8)
    unaligned = np.ndarray(
        canonical.shape,
        dtype=np.uint64,
        buffer=backing,
        offset=1,
    )
    unaligned[:] = canonical
    assert not unaligned.flags.aligned

    for query in (foreign_endian, noncontiguous, unaligned):
        normalized = _normalize_numeric_ids(query)
        assert normalized.dtype == np.dtype(np.uint64)
        assert normalized.dtype.isnative
        assert normalized.flags.c_contiguous
        assert normalized.flags.aligned
        np.testing.assert_array_equal(normalized, canonical)
        _assert_same_bits(collection.get(query), vectors)


@pytest.mark.parametrize(
    ("ids", "error", "message"),
    [
        (
            np.array([7, -9, 2], dtype=np.int16),
            ValueError,
            r"ids\[1\]=-9.*negative",
        ),
        ([7, -9, 2], ValueError, r"ids\[1\]=-9.*outside|ids\[1\]=-9.*range"),
        ([7, 1 << 64, 2], ValueError, r"ids\[1\]=18446744073709551616"),
        ([7, True, 2], TypeError, r"ids\[1\]=True.*boolean"),
        ([7, 1.0, 2], TypeError, r"ids\[1\]=1\.0.*not an integer"),
        (np.array([1, 2], dtype=np.bool_), TypeError, r"integer dtype.*bool"),
        (np.array([1.0, 2.0]), TypeError, r"integer dtype.*float"),
        (np.array([1, 2], dtype=object), TypeError, r"integer dtype.*object"),
        (np.array([1 + 0j]), TypeError, r"integer dtype.*complex"),
        (np.array([[1, 2]], dtype=np.uint32), ValueError, r"shape \(N,\)"),
        (np.array(1, dtype=np.uint32), ValueError, r"shape \(N,\)"),
        (1, TypeError, r"ordered sequence"),
        ("123", TypeError, r"numeric ids"),
    ],
)
def test_invalid_numeric_ids_are_rejected_with_actionable_errors(
    ids: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _normalize_numeric_ids(ids)


def test_masked_arrays_and_generators_are_rejected_as_ids() -> None:
    masked = np.ma.array([1, 2], mask=[False, True], dtype=np.uint32)
    generator = (value for value in [1, 2])

    with pytest.raises(TypeError, match="masked"):
        _normalize_numeric_ids(masked)
    with pytest.raises(TypeError, match="ordered sequence"):
        _normalize_numeric_ids(generator)


def test_inconsistent_sequence_length_never_exposes_uninitialized_ids() -> None:
    class ShortSequence(Sequence[int]):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, position: int) -> int:
            if position >= 2:
                raise IndexError
            return position + 1

    with pytest.raises(ValueError, match="changed length"):
        _normalize_numeric_ids(ShortSequence())


def test_id_normalization_can_report_an_absolute_batch_position() -> None:
    with pytest.raises(ValueError, match=r"ids\[12\]=-1"):
        _normalize_numeric_ids(
            np.array([7, 8, -1], dtype=np.int32),
            position_offset=10,
        )


def test_collection_create_open_add_get_and_get_into_are_bit_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    schema = throwdb.CollectionSchema(dimension=9)
    ids = np.array([81, 4, 1_000_003, 17], dtype=np.uint32)
    vectors = _vectors(ids.size, schema.dimension)

    with throwdb.Collection.create(
        path,
        schema=schema,
        segment_size_bytes=3 * schema.dimension * np.dtype(np.float16).itemsize,
        format_version=7,
    ) as builder:
        assert builder.mode == "write"
        assert builder.schema is schema
        assert builder.id_type == "uint64"
        assert builder.dtype == "float16"
        assert builder.format_version == 7
        assert builder.dimension == schema.dimension
        assert builder.count == 0
        assert builder.tuning is None
        assert builder.open_stats is None
        builder.add(ids[:2], vectors[:2])
        assert builder.count == 2
        builder.add(ids[2:], vectors[2:])
        assert builder.count == ids.size

    collection = throwdb.Collection.open(path)
    assert collection.mode == "read"
    assert collection.schema == schema
    assert collection.id_type == "uint64"
    assert collection.dtype == "float16"
    assert collection.format_version == 7
    assert collection.dimension == schema.dimension
    assert collection.count == ids.size
    open_stats = collection.open_stats
    assert open_stats is not None
    assert open_stats["index_validation"] == "checksum"
    assert open_stats["index_body_checksum_present"] is True
    assert open_stats["index_body_checksum_verified"] is True
    assert open_stats["index_structural_validation_performed"] is False
    assert open_stats["total_seconds"] > 0
    assert open_stats["base_segments_seconds"] >= 0
    open_stats["index_validation"] = "caller mutation"
    assert collection.open_stats["index_validation"] == "checksum"  # type: ignore[index]

    verified = throwdb.Collection.open(path, verify_index=True)
    verified_stats = verified.open_stats
    assert verified_stats is not None
    assert verified_stats["index_validation"] == "checksum_and_full"
    assert verified_stats["index_structural_validation_performed"] is True
    verified.close()

    with pytest.raises(TypeError, match="verify_index must be a bool"):
        throwdb.Collection.open(path, verify_index=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_wal_backlog_bytes"):
        throwdb.Collection.open(path, max_wal_backlog_bytes=0)
    with pytest.raises(TypeError, match="max_overlay_entries"):
        throwdb.Collection.open(
            path,
            max_overlay_entries=True,  # type: ignore[arg-type]
        )

    order = np.array([3, 0, 3, 2, 1], dtype=np.intp)
    query = ids[order].astype(np.int64)
    _assert_same_bits(collection.get(query), vectors[order])

    output = np.empty((order.size, schema.dimension), dtype=np.float16)
    result = collection.get_into(query, output)
    assert result is None
    _assert_same_bits(output, vectors[order])


def test_collection_create_forwards_default_and_explicit_current_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeBuilder:
        count = 0

        def __init__(self, format_version: int) -> None:
            self.format_version = format_version

        def abort(self) -> None:
            pass

    class FakeNativeCollectionBuilder:
        @staticmethod
        def create(_path: object, **options: object) -> FakeBuilder:
            version = options["format_version"]
            assert isinstance(version, int)
            calls.append(version)
            return FakeBuilder(version)

    monkeypatch.setattr(
        collection_module,
        "NativeCollectionBuilder",
        FakeNativeCollectionBuilder,
    )

    default = throwdb.Collection.create(tmp_path / "default")
    explicit = throwdb.Collection.create(
        tmp_path / "explicit",
        format_version=7,
    )

    assert calls == [7, 7]
    assert default.format_version == 7
    assert explicit.format_version == 7
    default.abort()
    explicit.abort()


@pytest.mark.parametrize(
    ("format_version", "error", "message"),
    [
        (True, TypeError, "not bool"),
        (np.bool_(False), TypeError, "not bool"),
        (6.0, TypeError, "must be an integer"),
        ("7", TypeError, "must be an integer"),
        *((version, ValueError, "must be 7") for version in range(7)),
        (8, ValueError, "must be 7"),
    ],
)
def test_collection_create_rejects_invalid_format_versions_before_native_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_version: object,
    error: type[Exception],
    message: str,
) -> None:
    class UnexpectedNativeCollectionBuilder:
        @staticmethod
        def create(_path: object, **_options: object) -> None:
            raise AssertionError("native create must not run")

    monkeypatch.setattr(
        collection_module,
        "NativeCollectionBuilder",
        UnexpectedNativeCollectionBuilder,
    )

    with pytest.raises(error, match=message):
        throwdb.Collection.create(
            tmp_path / "invalid-format",
            format_version=format_version,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "invalid-format").exists()


def test_legacy_storage_and_writer_are_not_public_api() -> None:
    assert not hasattr(throwdb, "Storage")
    assert not hasattr(throwdb, "Writer")
    assert "Storage" not in throwdb.__all__
    assert "Writer" not in throwdb.__all__


def test_collection_rejects_retired_names_without_changing_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    schema = throwdb.CollectionSchema(dimension=4, dtype="float16")
    with pytest.raises(TypeError, match="shard_size_bytes"):
        throwdb.Collection.create(path, schema=schema, shard_size_bytes=16)
    assert not path.exists()

    vectors = _vectors(2, schema.dimension)
    with throwdb.Collection.create(
        path, schema=schema, segment_size_bytes=vectors[0].nbytes,
    ) as builder:
        builder.add([1, 2], vectors)

    with throwdb.Collection.open(path, mode="r+") as collection:
        assert not hasattr(collection, "compact_deltas")
        assert not hasattr(collection_module.NativeCollection, "compact_delta_runs")
        before = collection.mutation_log_stats
        for keyword in (
            "compact_min_runs", "compact_max_runs",
            "compact_max_input_bytes", "compact_max_input_entries",
        ):
            with pytest.raises(TypeError, match=keyword):
                collection.maintain(**{keyword: 2})
        assert collection.mutation_log_stats == before
        _assert_same_bits(collection.get([2, 1]), vectors[::-1])

        retired_stats = {
            "catalog_managed", "delta_managed", "reclaimed_wal_count",
            "reclaimed_wal_bytes", "compacted_delta_run_count",
            "reclaimed_delta_run_count", "reclaimed_delta_bytes",
            "delta_run_count", "delta_entry_count", "delta_put_rows",
            "total_delta_segment_bytes", "total_delta_run_bytes",
            "total_delta_bytes",
        }
        assert retired_stats.isdisjoint(before)


def test_add_normalizes_layout_but_never_silently_casts_embedding_dtype(
    tmp_path: Path,
) -> None:
    dimension = 6
    ids = np.array([3, 7, 11], dtype=np.uint16)
    expected = _vectors(ids.size, dimension)

    interleaved = np.empty((ids.size, dimension * 2), dtype=np.float16)
    embeddings = interleaved[:, ::2]
    embeddings[:] = expected
    assert not embeddings.flags.c_contiguous

    path = tmp_path / "normalized-layout"
    _create_collection(path, ids, embeddings)
    _assert_same_bits(throwdb.Collection.open(path).get(ids), expected)

    builder = throwdb.Collection.create(
        tmp_path / "wrong-dtype",
        schema=throwdb.CollectionSchema(dimension=dimension),
        format_version=7,
    )
    with pytest.raises(TypeError, match="float16.*float32"):
        builder.add(ids, expected.astype(np.float32))
    builder.abort()


def test_prefix_get_and_get_into_return_bit_exact_contiguous_results(
    tmp_path: Path,
) -> None:
    ids = np.array([101, 303, 707, 909], dtype=np.uint64)
    vectors = _vectors(ids.size, 11)
    path = tmp_path / "store"
    _create_collection(path, ids, vectors)
    collection = throwdb.Collection.open(path)
    order = np.array([2, 0, 3, 1], dtype=np.intp)
    query = ids[order]

    prefix = collection.get(query, prefix_dimensions=4)
    assert prefix.flags.c_contiguous
    _assert_same_bits(prefix, vectors[order, :4])

    full = collection.get(query, prefix_dimensions=collection.dimension)
    _assert_same_bits(full, vectors[order])

    output = np.empty((query.size, 4), dtype=np.float16)
    assert collection.get_into(query, output, prefix_dimensions=4) is None
    _assert_same_bits(output, vectors[order, :4])


@pytest.mark.parametrize(
    ("prefix", "error", "message"),
    [
        (0, ValueError, "between 1"),
        (12, ValueError, "between 1"),
        (-1, ValueError, "between 1"),
        (True, TypeError, "not bool"),
        (3.0, TypeError, "must be an integer"),
    ],
)
def test_prefix_dimensions_are_validated(
    tmp_path: Path,
    prefix: object,
    error: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "store"
    ids = np.array([1], dtype=np.uint64)
    _create_collection(path, ids, _vectors(1, 11))
    collection = throwdb.Collection.open(path)

    with pytest.raises(error, match=message):
        collection.get(ids, prefix_dimensions=prefix)  # type: ignore[arg-type]


def test_prefix_get_into_validates_caller_owned_output(tmp_path: Path) -> None:
    ids = np.array([5, 8], dtype=np.uint64)
    vectors = _vectors(ids.size, 7)
    path = tmp_path / "store"
    _create_collection(path, ids, vectors)
    collection = throwdb.Collection.open(path)

    with pytest.raises(TypeError, match="NumPy ndarray"):
        collection.get_into(ids, [[0.0] * 3] * 2, prefix_dimensions=3)
    with pytest.raises(TypeError, match="float16.*float32"):
        collection.get_into(
            ids,
            np.empty((2, 3), dtype=np.float32),
            prefix_dimensions=3,
        )
    with pytest.raises(ValueError, match="shape"):
        collection.get_into(
            ids,
            np.empty((2, 4), dtype=np.float16),
            prefix_dimensions=3,
        )

    noncontiguous = np.empty((2, 6), dtype=np.float16)[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        collection.get_into(ids, noncontiguous, prefix_dimensions=3)

    readonly = np.empty((2, 3), dtype=np.float16)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        collection.get_into(ids, readonly, prefix_dimensions=3)

    backing = np.zeros(1 + 2 * 3 * np.dtype(np.float16).itemsize, dtype=np.uint8)
    unaligned = np.ndarray((2, 3), dtype=np.float16, buffer=backing, offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        collection.get_into(ids, unaligned, prefix_dimensions=3)

    masked = np.ma.array(np.empty((2, 3), dtype=np.float16), mask=False)
    with pytest.raises(TypeError, match="NumPy ndarray"):
        collection.get_into(ids, masked, prefix_dimensions=3)

    foreign_prefix = ">" if sys.byteorder == "little" else "<"
    foreign = np.empty((2, 3), dtype=f"{foreign_prefix}f2")
    with pytest.raises(TypeError, match="native-endian"):
        collection.get_into(ids, foreign, prefix_dimensions=3)


@pytest.mark.parametrize("prefix", [None, 3])
def test_missing_id_leaves_get_into_output_untouched(
    tmp_path: Path,
    prefix: int | None,
) -> None:
    ids = np.array([11, 22, 33], dtype=np.uint64)
    vectors = _vectors(ids.size, 6)
    path = tmp_path / "store"
    _create_collection(path, ids, vectors)
    collection = throwdb.Collection.open(path)
    columns = collection.dimension if prefix is None else prefix
    output = np.full((3, columns), np.float16(-4.5), dtype=np.float16)
    before = output.copy()

    with pytest.raises(KeyError):
        collection.get_into(
            [11, 999_999, 22],
            output,
            prefix_dimensions=prefix,
        )

    _assert_same_bits(output, before)


def test_builder_context_aborts_on_exception(tmp_path: Path) -> None:
    path = tmp_path / "aborted"
    ids = np.array([1, 2], dtype=np.uint32)

    with pytest.raises(RuntimeError, match="stop build"):
        with throwdb.Collection.create(
            path,
            schema=throwdb.CollectionSchema(dimension=3),
            format_version=7,
        ) as builder:
            builder.add(ids, _vectors(ids.size, 3))
            raise RuntimeError("stop build")

    _assert_store_is_unreadable(path)


def test_explicit_finish_and_abort_are_idempotent_and_close_the_builder(
    tmp_path: Path,
) -> None:
    finished_path = tmp_path / "finished"
    ids = np.array([1, 2], dtype=np.uint64)
    vectors = _vectors(ids.size, 3)
    finished = throwdb.Collection.create(
        finished_path,
        schema=throwdb.CollectionSchema(dimension=3),
        format_version=7,
    )
    finished.add(ids, vectors)
    finished.finish()
    finished.finish()
    finished.abort()  # A completed snapshot must remain published.
    with pytest.raises(RuntimeError, match="already finished"):
        finished.add(ids, vectors)
    with pytest.raises(RuntimeError, match="already finished"):
        finished.__enter__()
    _assert_same_bits(throwdb.Collection.open(finished_path).get(ids), vectors)

    aborted_path = tmp_path / "aborted"
    aborted = throwdb.Collection.create(
        aborted_path,
        schema=throwdb.CollectionSchema(dimension=3),
        format_version=7,
    )
    aborted.add(ids, vectors)
    aborted.abort()
    aborted.abort()
    with pytest.raises(RuntimeError, match="already aborted"):
        aborted.add(ids, vectors)
    with pytest.raises(RuntimeError, match="already aborted"):
        aborted.finish()
    _assert_store_is_unreadable(aborted_path)


def test_collection_operations_reject_the_wrong_mode(tmp_path: Path) -> None:
    path = tmp_path / "store"
    ids = np.array([1], dtype=np.uint64)
    vectors = _vectors(1, 2)
    builder = throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=2),
        format_version=7,
    )

    with pytest.raises(RuntimeError, match="opened collection reader"):
        builder.get(ids)
    with pytest.raises(RuntimeError, match="opened collection reader"):
        builder.get_into(ids, np.empty((1, 2), dtype=np.float16))
    builder.add(ids, vectors)
    builder.finish()

    reader = throwdb.Collection.open(path)
    with pytest.raises(RuntimeError, match="builder"):
        reader.add(ids, vectors)
    with pytest.raises(RuntimeError, match="builder"):
        reader.finish()
    with pytest.raises(RuntimeError, match="builder"):
        reader.abort()


def test_collection_schema_canonicalizes_valid_values() -> None:
    schema = throwdb.CollectionSchema(
        id_type="uint64",
        dimension=np.int32(128),
        dtype=np.dtype(np.float16),  # type: ignore[arg-type]
    )

    assert schema.id_type == "uint64"
    assert schema.dimension == 128
    assert schema.dtype == "float16"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"id_type": 1}, TypeError, "id_type must be a string"),
        ({"id_type": "int32"}, ValueError, "uint64.*utf8"),
        ({"dimension": True}, TypeError, "not bool"),
        ({"dimension": 1.5}, TypeError, "must be an integer"),
        ({"dimension": 0}, ValueError, "between 1"),
        ({"dimension": -1}, ValueError, "between 1"),
        ({"dimension": 1 << 32}, ValueError, "between 1"),
        ({"dtype": "float64"}, ValueError, "float16.*float32"),
        ({"dtype": "not-a-dtype"}, TypeError, "invalid embedding dtype"),
    ],
)
def test_collection_schema_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        throwdb.CollectionSchema(**kwargs)  # type: ignore[arg-type]


def test_current_writer_reports_future_id_schema_features_explicitly(
    tmp_path: Path,
) -> None:
    with pytest.raises(NotImplementedError, match="utf8.*not implemented"):
        throwdb.Collection.create(
            tmp_path / "utf8",
            schema=throwdb.CollectionSchema(id_type="utf8", dimension=4),
        )
    with pytest.raises(TypeError, match="CollectionSchema"):
        throwdb.Collection.create(tmp_path / "invalid", schema=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("segment_size_bytes", "error", "message"),
    [
        (True, TypeError, "not bool"),
        (np.bool_(False), TypeError, "not bool"),
        (1.5, TypeError, "must be an integer"),
        (0, ValueError, "between 1"),
        (-1, ValueError, "between 1"),
        (1 << 64, ValueError, "between 1"),
    ],
)
def test_collection_create_validates_segment_size_before_native_open(
    tmp_path: Path,
    segment_size_bytes: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        throwdb.Collection.create(
            tmp_path / "invalid-segment-size",
            schema=throwdb.CollectionSchema(dimension=4),
            segment_size_bytes=segment_size_bytes,  # type: ignore[arg-type]
        )


def test_auto_and_manual_tuning_profiles_are_observable_and_defensive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    _create_collection(path, [1, 2], _vectors(2, 3))

    automatic = throwdb.Collection.open(path)
    automatic_profile = automatic.tuning
    assert automatic_profile is not None
    assert automatic_profile["mode"] == "auto"
    assert automatic_profile["io_backend"] == "thread_pool"
    assert automatic_profile["io_ranges_per_task"] == 8
    automatic_profile["io_workers"] = 999
    assert automatic.tuning["io_workers"] == 0  # type: ignore[index]

    requested: dict[str, object] = {
        "backend": "thread_pool",
        "workers": 2,
        "queue_capacity": 1,
        "ranges_per_task": 4,
        "queue_depth": 64,
    }
    unchanged = requested.copy()
    manual = throwdb.Collection.open(path, tuning=requested)
    assert requested == unchanged
    profile = manual.tuning
    assert profile is not None
    assert profile["mode"] == "manual"
    assert profile["io_backend"] == "thread_pool"
    assert profile["io_workers"] == 2
    assert profile["io_queue_capacity"] == 1
    assert profile["io_ranges_per_task"] == 4
    assert profile["io_uring_queue_depth"] == 64
    _assert_same_bits(manual.get([2, 1]), _vectors(2, 3)[::-1])


@pytest.mark.parametrize(
    ("tuning", "error", "message"),
    [
        ("manual", ValueError, "must be 'auto'"),
        (None, TypeError, "must be 'auto'"),
        ([], TypeError, "must be 'auto'"),
        ({1: 2}, TypeError, "keys must be strings"),
        ({"unknown": 1}, ValueError, "unknown tuning option"),
        (
            {"backend": "thread_pool", "io_backend": "thread_pool"},
            ValueError,
            "only one.*backend.*io_backend",
        ),
        (
            {"workers": 1, "io_workers": 1},
            ValueError,
            "workers.*backend-specific",
        ),
        (
            {"queue_capacity": 1, "io_queue_capacity": 1},
            ValueError,
            "only one.*queue_capacity.*io_queue_capacity",
        ),
        ({"workers": -1}, ValueError, "io_workers.*between"),
        (
            {"workers": 0, "queue_capacity": 1},
            ValueError,
            "queue_capacity.*workers",
        ),
    ],
)
def test_invalid_tuning_is_rejected(
    tmp_path: Path,
    tuning: object,
    error: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "store"
    _create_collection(path, [1], _vectors(1, 2))

    with pytest.raises(error, match=message):
        throwdb.Collection.open(path, tuning=tuning)  # type: ignore[arg-type]


def test_low_level_native_verify_flag_requires_a_python_bool(
    tmp_path: Path,
) -> None:
    from throwdb._throwdb import NativeCollection

    path = tmp_path / "store"
    _create_collection(path, [1], _vectors(1, 2))

    for invalid in (1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="verify_native_index must be a bool"):
            NativeCollection.open(
                path,
                verify_native_index=invalid,
            )

    reader = NativeCollection.open(path, verify_native_index=True)
    assert reader.open_stats["index_validation"] == "checksum_and_full"
    reader.close()


def test_low_level_native_metadata_and_range_options(
    tmp_path: Path,
) -> None:
    from throwdb._throwdb import NativeCollection

    path = tmp_path / "store"
    dimension = 7
    segment_size_bytes = 1 << 20
    ids = np.array([11, 22, 33], dtype=np.uint64)
    vectors = _vectors(ids.size, dimension)
    _create_collection(
        path,
        ids,
        vectors,
        segment_size_bytes=segment_size_bytes,
    )

    metadata = NativeCollection.inspect_metadata(path)
    collection_id_low = metadata.pop("collection_id_low")
    collection_id_high = metadata.pop("collection_id_high")
    assert collection_id_low != 0 or collection_id_high != 0
    assert metadata == {
        "format_version": 7,
        "dimension": dimension,
        "dtype": "float16",
        "record_size": dimension * np.dtype(np.float16).itemsize,
        "segment_size_bytes": segment_size_bytes,
        "layout": "uint64_dense_row_major_partitioned_rowref_v1",
        "feature_discriminator": "segment_catalog_v1+partitioned_rowref_v1",
        "required_feature_bits": 13,
        "optional_feature_bits": 0,
    }

    reader = NativeCollection.open(
        path,
        max_read_range_bytes=metadata["record_size"],
        max_gap_bytes=(1 << 64) - 1,
    )
    _assert_same_bits(reader.get(ids[::-1].copy()), vectors[::-1])
    reader.close()

    with pytest.raises(ValueError, match="max_read_range_bytes.*greater than zero"):
        NativeCollection.open(path, max_read_range_bytes=0)
    with pytest.raises(
        RuntimeError, match="max_read_range_bytes.*at least one record"
    ):
        NativeCollection.open(
            path,
            max_read_range_bytes=int(metadata["record_size"]) - 1,
        )
    with pytest.raises(ValueError, match="max_gap_bytes.*2\\*\\*64-1"):
        NativeCollection.open(path, max_gap_bytes=-1)
    with pytest.raises(ValueError, match="max_wal_operations.*greater than zero"):
        NativeCollection.open(path, max_wal_operations=0)
    with pytest.raises(ValueError, match="max_overlay_bytes.*2\\*\\*64-1"):
        NativeCollection.open(path, max_overlay_bytes=-1)


def test_low_level_reader_capability_probe_is_bounded_and_observable() -> None:
    from throwdb._throwdb import NativeCollection

    capabilities = NativeCollection.reader_capabilities()
    assert set(capabilities) == {
        "io_uring_available",
        "io_uring_probe_queue_depth",
        "io_uring_probe_workers",
        "io_uring_reason",
    }
    assert isinstance(capabilities["io_uring_available"], bool)
    assert capabilities["io_uring_probe_queue_depth"] == 32
    assert capabilities["io_uring_probe_workers"] == 1
    reason = capabilities["io_uring_reason"]
    assert reason is None or isinstance(reason, str)
    if capabilities["io_uring_available"]:
        assert reason is None
    else:
        assert reason

    exact = NativeCollection.reader_capabilities(
        io_uring_queue_depth=8,
        io_uring_workers=2,
    )
    assert exact["io_uring_probe_queue_depth"] == 8
    assert exact["io_uring_probe_workers"] == 2
    with pytest.raises(ValueError, match="at least io_uring_workers"):
        NativeCollection.reader_capabilities(
            io_uring_queue_depth=1,
            io_uring_workers=2,
        )


def test_low_level_calibration_snapshot_reuses_index_for_reader_trials(
    tmp_path: Path,
) -> None:
    from throwdb._throwdb import NativeCollection

    path = tmp_path / "store"
    ids = np.array([11, 29, 47, 83], dtype=np.uint64)
    vectors = _vectors(ids.size, 6)
    _create_collection(path, ids, vectors)

    calibration = NativeCollection.open(path, calibration=True)
    sampled = calibration.sample_ids(
        requested_count=ids.size,
        seed=123,
        max_inspected_slots=1 << 20,
    )
    assert sampled["complete"] is True
    assert sampled["requested_count"] == ids.size
    assert sampled["inspected_slots"] <= 1 << 20
    sampled_ids = sampled["ids"]
    assert isinstance(sampled_ids, np.ndarray)
    assert sampled_ids.dtype == np.dtype(np.uint64)
    assert set(sampled_ids.tolist()) == set(ids.tolist())

    default_v3_bound = calibration.sample_ids(
        requested_count=1 << 22,
        seed=123,
        max_inspected_slots=1,
    )
    assert default_v3_bound["requested_count"] == 1 << 22
    assert default_v3_bound["inspected_slots"] == 1
    assert default_v3_bound["complete"] is False

    trial = calibration.reader_trial(io_workers=2, io_queue_capacity=4)
    output = np.empty_like(vectors)
    stats = trial.get_into_profiled(ids, output)
    _assert_same_bits(output, vectors)
    assert stats["requested_rows"] == ids.size
    assert stats["planned_read_bytes"] <= vectors.nbytes
    trial.close()
    with pytest.raises(RuntimeError, match="trial is closed"):
        trial.get_into(ids, output)
    calibration.close()

    with pytest.raises(TypeError, match="calibration must be a bool"):
        NativeCollection.open(path, calibration=1)
    with pytest.raises(ValueError, match="must be read-only"):
        NativeCollection.open(path, mode="r+", calibration=True)
