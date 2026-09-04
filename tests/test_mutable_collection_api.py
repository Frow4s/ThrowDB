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
        format_version=6,
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
        assert collection.format_version == 6
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


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_format7_python_read_write_crud_rotation_and_reopen(
    tmp_path: Path,
    dtype: str,
) -> None:
    path = tmp_path / f"partitioned-{dtype}"
    maximum_id = (1 << 64) - 1
    base = _vectors(3, 5, dtype, salt=7)
    schema = throwdb.CollectionSchema(dimension=5, dtype=dtype)
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=base[0].nbytes,
        format_version=7,
    ) as builder:
        builder.add([0, maximum_id, 7], base)

    old_reader = throwdb.Collection.open(path)
    inserted = _vectors(1, 5, dtype, salt=107)
    updated = _vectors(2, 5, dtype, salt=207)
    with throwdb.Collection.open(path, mode="r+") as writer:
        assert writer.format_version == 7
        assert writer.mode == "read_write"
        writer.insert(np.array([9], dtype=np.uint32), inserted)
        writer.update([0, maximum_id], updated)
        np.testing.assert_array_equal(writer.delete([7, 123]), [True, False])
        writer.upsert([7], inserted)
        stats = writer.rotate_wal()
        assert stats["rotated"] is True
        assert stats["generation"] == 2
        assert stats["frozen_wal_count"] == 1
        materialized = writer.materialize_frozen_wals()
        assert materialized["materialized"] is True
        assert materialized["generation"] == 4
        assert materialized["materialized_sequence"] == 4
        assert materialized["frozen_wal_count"] == 0
        assert materialized["delta_run_count"] == 1
        assert materialized["delta_entry_count"] == 4
        assert materialized["delta_put_rows"] == 4
        assert materialized["total_delta_run_bytes"] > 0
        assert materialized["total_delta_segment_bytes"] > 0
        assert materialized["reclamation_complete"] is False
        folded = writer.compact_deltas()
        assert folded["compacted"] is True
        assert folded["generation"] == 6
        assert folded["compacted_partition_count"] >= 1
        assert (
            folded["compacted_run_section_count"]
            == folded["compacted_partition_count"]
        )
        assert folded["compacted_snapshot_input_entries"] == 3
        assert folded["compacted_delta_input_entries"] == 4
        assert folded["compacted_output_entries"] == 4
        assert folded["delta_run_count"] == 0
        assert folded["compaction_budget_blocked"] is False
        assert folded["reclamation_complete"] is False
        expected = np.stack([updated[0], updated[1], inserted[0], inserted[0]])
        _assert_same_bits(writer.get([0, maximum_id, 7, 9]), expected)
        prefix = np.empty((4, 2), dtype=np.dtype(dtype))
        writer.get_into(
            [0, maximum_id, 7, 9], prefix, prefix_dimensions=2
        )
        _assert_same_bits(prefix, expected[:, :2].copy(order="C"))

    assert old_reader.count == 3
    _assert_same_bits(old_reader.get([0, maximum_id, 7]), base)
    with pytest.raises(KeyError):
        old_reader.get([9])
    old_reader.close()

    with throwdb.Collection.open(path) as reopened:
        assert reopened.format_version == 7
        assert reopened.count == 4
        _assert_same_bits(
            reopened.get([0, maximum_id, 7, 9]),
            np.stack([updated[0], updated[1], inserted[0], inserted[0]]),
        )
    reopened.close()


def test_format7_list_and_scan_ids_follow_snapshot_physical_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physical-id-scan"
    dimension = 4
    base = _vectors(4, dimension, "float16", salt=31)
    schema = throwdb.CollectionSchema(dimension=dimension, dtype="float16")
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=base[0].nbytes * 2,
        format_version=7,
    ) as builder:
        builder.add([30, 10, 20, 50], base)

    replacement = _vectors(1, dimension, "float16", salt=331)
    inserted = _vectors(1, dimension, "float16", salt=631)
    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.update([10], replacement)
        np.testing.assert_array_equal(collection.delete([20]), [True])
        collection.insert([40], inserted)

        located = collection.list_ids(with_locations=True, batch_rows=2)
        assert located.ids.dtype == np.dtype(np.uint64)
        assert located.locations.dtype == throwdb.ID_LOCATION_DTYPE
        assert located.ids.flags.c_contiguous
        assert located.locations.flags.c_contiguous
        assert set(located.ids.tolist()) == {10, 30, 40, 50}
        coordinates = [
            (int(segment), int(row))
            for segment, row in zip(
                located.locations["segment_id"],
                located.locations["row_index"],
                strict=True,
            )
        ]
        assert coordinates == sorted(coordinates)
        assert len(set(coordinates)) == len(coordinates)

        plain = collection.list_ids(batch_rows=1)
        np.testing.assert_array_equal(plain, located.ids)

        with collection.scan_ids(
            batch_rows=2,
            with_locations=True,
        ) as scan:
            assert scan.snapshot.count == collection.count
            batches = list(scan)
            assert scan.done
        np.testing.assert_array_equal(
            np.concatenate([batch.ids for batch in batches]),
            located.ids,
        )
        np.testing.assert_array_equal(
            np.concatenate([batch.locations for batch in batches]),
            located.locations,
        )
        offsets: list[int] = []
        emitted = 0
        for batch in batches:
            offsets.append(emitted)
            emitted += batch.ids.size
        assert [batch.emitted_offset for batch in batches] == offsets
        assert all(batch.snapshot == located.snapshot for batch in batches)

        # A cursor pins both its logical winners and physical resources.  A
        # later commit may relocate an ID, but cannot change this scan.
        snapshot_scan = collection.scan_ids(batch_rows=1)
        snapshot_count = snapshot_scan.snapshot_count
        collection.delete([30])
        collection.insert([60], inserted)
        snapshot_ids = np.concatenate(list(snapshot_scan))
        snapshot_scan.close()
        assert snapshot_ids.size == snapshot_count
        assert set(snapshot_ids.tolist()) == {10, 30, 40, 50}

        current = collection.list_ids()
        assert set(current.tolist()) == {10, 40, 50, 60}


def test_scan_ids_validates_buffers_and_empty_collection(tmp_path: Path) -> None:
    path = tmp_path / "empty-scan"
    schema = throwdb.CollectionSchema(dimension=3, dtype="float16")
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=1 << 20,
        format_version=7,
    ):
        pass

    with throwdb.Collection.open(path) as collection:
        assert collection.list_ids().shape == (0,)
        located = collection.list_ids(with_locations=True)
        assert located.ids.shape == (0,)
        assert located.locations.shape == (0,)
        with pytest.raises(ValueError, match="batch_rows"):
            collection.scan_ids(batch_rows=0)
        with pytest.raises(TypeError, match="with_locations"):
            collection.scan_ids(with_locations=1)

        with collection.scan_ids(with_locations=True) as scan:
            ids = np.empty(1, dtype=np.uint64)
            locations = np.empty(1, dtype=throwdb.ID_LOCATION_DTYPE)
            shared = np.zeros(8, dtype=np.uint8)
            overlapping_ids = np.ndarray(
                (1,), dtype=np.uint64, buffer=shared
            )
            overlapping_locations = np.ndarray(
                (1,), dtype=throwdb.ID_LOCATION_DTYPE, buffer=shared
            )
            with pytest.raises(ValueError, match="ids and locations"):
                scan.read_into(overlapping_ids, overlapping_locations)
            assert scan.read_into(ids, locations) == 0
            with pytest.raises(TypeError, match="uint64"):
                scan.read_into(np.empty(1, dtype=np.uint32), locations)


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


def test_profiled_read_reports_zero_delta_and_mutation_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    dimension = 6
    ids = np.array([11, 22], dtype=np.uint64)
    base = _vectors(2, dimension, "float16", salt=17)
    replacement = _vectors(1, dimension, "float16", salt=317)
    _build(path, ids, base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        output = np.empty_like(base)
        clean = collection.get_into_profiled(ids, output)
        _assert_same_bits(output, base)
        assert clean["zero_delta_fast_path"] is True
        assert clean["requested_rows"] == 2
        assert clean["base_rows"] == 2
        assert clean["mutation_rows"] == 0
        assert clean["unique_files"] == 1
        assert clean["planned_ranges"] == 1
        assert clean["requested_bytes"] == base.nbytes
        assert clean["planned_read_bytes"] == base.nbytes
        assert clean["completed_read_bytes"] == base.nbytes
        assert clean["useful_read_bytes"] == base.nbytes
        assert clean["gap_bytes"] == 0
        assert clean["overread_bytes"] == 0
        assert clean["pages_touched"] >= 1
        assert clean["page_runs"] >= 1
        assert clean["chunks_touched"] >= 1
        assert clean["chunk_runs"] >= 1
        assert clean["backend_tasks"] == 1

        collection.update([22], replacement)
        mutated = collection.get_into_profiled(ids, output)
        _assert_same_bits(output, np.stack([base[0], replacement[0]]))
        assert mutated["zero_delta_fast_path"] is False
        assert mutated["requested_rows"] == 2
        assert mutated["base_rows"] == 1
        assert mutated["mutation_rows"] == 1
        assert mutated["planned_ranges"] == 2
        assert mutated["planned_read_bytes"] == base.nbytes
        assert mutated["backend_tasks"] == 2

        for key in (
            "total_seconds",
            "lookup_seconds",
            "physical_sort_seconds",
            "physical_plan_seconds",
            "physical_backend_seconds",
        ):
            assert isinstance(mutated[key], float)
            assert mutated[key] >= 0.0


def test_catalog_wal_rotation_stats_and_recovery(tmp_path: Path) -> None:
    path = tmp_path / "store"
    maximum_id = (1 << 64) - 1
    boundary_id = 1 << 32
    base = _vectors(2, 5, "float16", salt=19)
    first = _vectors(2, 5, "float16", salt=219)
    second = _vectors(2, 5, "float16", salt=419)
    _build(path, [0, maximum_id], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        initial = collection.mutation_log_stats
        assert initial is not None
        assert initial["catalog_managed"] is True
        assert initial["generation"] == 1
        assert initial["committed_sequence"] == 0
        assert initial["frozen_wal_count"] == 0
        assert initial["active_wal_bytes"] == 4096
        assert initial["active_wal_put_rows"] == 0
        assert initial["active_wal_transaction_count"] == 0

        empty = collection.rotate_wal()
        assert empty["rotated"] is False
        assert empty["generation"] == 1

        collection.upsert([0, boundary_id], first)
        before = collection.mutation_log_stats
        assert before is not None
        assert before["committed_sequence"] == 1
        assert before["active_wal_put_rows"] == 2
        assert before["active_wal_transaction_count"] == 1

        rotated = collection.rotate_wal()
        assert rotated["rotated"] is True
        assert rotated["generation"] == 2
        assert rotated["committed_sequence"] == 1
        assert rotated["frozen_wal_count"] == 1
        assert rotated["active_wal_bytes"] == 4096
        assert rotated["active_wal_put_rows"] == 0
        assert rotated["active_wal_transaction_count"] == 0
        assert rotated["total_wal_bytes"] > 8192

        np.testing.assert_array_equal(collection.delete([maximum_id]), [True])
        collection.update([boundary_id], second[:1])
        collection.insert([maximum_id], second[1:])
        assert collection.count == 3
        second_rotation = collection.rotate_wal()
        assert second_rotation["rotated"] is True
        assert second_rotation["generation"] == 3
        assert second_rotation["committed_sequence"] == 4
        assert second_rotation["frozen_wal_count"] == 2

    with throwdb.Collection.open(path) as reopened:
        assert reopened.format_version == 6
        stats = reopened.mutation_log_stats
        assert stats is not None
        assert stats["catalog_managed"] is True
        assert stats["generation"] == 3
        assert stats["committed_sequence"] == 4
        assert stats["frozen_wal_count"] == 2
        _assert_same_bits(
            reopened.get([maximum_id, 0, boundary_id]),
            np.stack([second[1], first[0], second[0]]),
        )
        with pytest.raises(RuntimeError, match=r"mode='r\+'"):
            reopened.rotate_wal()


def test_frozen_wals_materialize_into_reopenable_delta_runs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "materialized"
    maximum_id = (1 << 64) - 1
    boundary_id = 1 << 32
    base = _vectors(2, 6, "float16", salt=31)
    first = _vectors(2, 6, "float16", salt=231)
    second = _vectors(2, 6, "float16", salt=431)
    _build(path, [0, maximum_id], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        initial = collection.materialize_frozen_wals()
        assert initial["materialized"] is False
        assert initial["generation"] == 1
        assert initial["delta_managed"] is True

        collection.upsert([0, boundary_id], first)  # sequence 1
        collection.rotate_wal()
        np.testing.assert_array_equal(collection.delete([maximum_id]), [True])
        collection.update([boundary_id], second[:1])  # sequences 2 and 3
        collection.rotate_wal()
        collection.insert([maximum_id], second[1:])  # sequence 4, active WAL

        materialized = collection.materialize_frozen_wals()
        assert materialized["materialized"] is True
        assert materialized["generation"] == 5  # two rotations + reserve + switch
        assert materialized["committed_sequence"] == 4
        assert materialized["materialized_sequence"] == 3
        assert materialized["frozen_wal_count"] == 0
        assert materialized["delta_run_count"] == 1
        assert materialized["delta_entry_count"] == 3
        assert materialized["delta_put_rows"] == 2
        assert materialized["reclaimed_wal_count"] == 2
        assert materialized["reclamation_complete"] is True
        assert len(list((path / "wal").glob("*.wal"))) == 1
        assert len(list((path / "deltas").glob("*.dat"))) == 1
        assert len(list((path / "deltas").glob("*.idx"))) == 1
        _assert_same_bits(
            collection.get([maximum_id, boundary_id, 0]),
            np.stack([second[1], second[0], first[0]]),
        )

    with throwdb.Collection.open(path, mode="r+") as reopened:
        stats = reopened.mutation_log_stats
        assert stats is not None
        assert stats["materialized_sequence"] == 3
        assert stats["delta_run_count"] == 1
        assert stats["frozen_wal_count"] == 0
        _assert_same_bits(
            reopened.get([0, maximum_id, boundary_id]),
            np.stack([first[0], second[1], second[0]]),
        )

        reopened.rotate_wal()
        second_materialization = reopened.materialize_frozen_wals()
        assert second_materialization["materialized"] is True
        assert second_materialization["materialized_sequence"] == 4
        assert second_materialization["delta_run_count"] == 2
        assert second_materialization["frozen_wal_count"] == 0

    with throwdb.Collection.open(path) as read_only:
        _assert_same_bits(
            read_only.get([maximum_id, 0, boundary_id]),
            np.stack([second[1], first[0], second[0]]),
        )
        with pytest.raises(
            RuntimeError,
            match=r"mode='r\+'",
        ):
            read_only.materialize_frozen_wals()


def test_materialization_selects_a_bounded_oldest_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded-materialization"
    base = _vectors(1, 4, "float16", salt=11)
    first = _vectors(1, 4, "float16", salt=111)
    second = _vectors(1, 4, "float16", salt=211)
    _build(path, [1], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        collection.update([1], first)
        collection.rotate_wal()
        collection.update([1], second)
        collection.rotate_wal()
        before = collection.mutation_log_stats
        assert before is not None
        assert before["frozen_wal_count"] == 2

        blocked = collection.materialize_frozen_wals(
            max_wals=1,
            max_input_bytes=1,
        )
        assert blocked["materialized"] is False
        assert blocked["materialization_budget_blocked"] is True
        assert blocked["generation"] == before["generation"]
        assert blocked["frozen_wal_count"] == 2

        bounded = collection.materialize_frozen_wals(
            max_wals=1,
            max_input_bytes=1 << 20,
        )
        assert bounded["materialized"] is True
        assert bounded["materialization_budget_blocked"] is False
        assert bounded["materialized_wal_count"] == 1
        assert bounded["materialized_wal_input_bytes"] > 0
        assert bounded["frozen_wal_count"] == 1
        assert bounded["delta_run_count"] == 1
        _assert_same_bits(collection.get([1]), second)

        with pytest.raises(ValueError, match="max_wals.*between"):
            collection.materialize_frozen_wals(max_wals=0)
        with pytest.raises(TypeError, match="max_input_bytes.*integer"):
            collection.materialize_frozen_wals(max_input_bytes=1.5)


def test_delta_compaction_public_api_is_bounded_and_reopenable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compacted"
    base = _vectors(2, 6, "float16", salt=41)
    first = _vectors(1, 6, "float16", salt=241)
    second = _vectors(1, 6, "float16", salt=441)
    _build(path, [1, 2], base)

    with throwdb.Collection.open(path, mode="r+") as collection:
        no_op = collection.compact_deltas()
        assert no_op["compacted"] is False
        assert no_op["delta_run_count"] == 0

        collection.update([1], first)
        collection.rotate_wal()
        collection.materialize_frozen_wals()
        collection.update([1], second)
        collection.rotate_wal()
        collection.materialize_frozen_wals()
        assert collection.mutation_log_stats["delta_run_count"] == 2

        compacted = collection.compact_deltas(
            max_runs=np.uint8(2),
            max_input_bytes=np.uint64(1 << 20),
            max_input_entries=np.int64(16),
        )
        assert compacted["compacted"] is True
        assert compacted["compacted_delta_run_count"] == 2
        assert compacted["compacted_delta_input_entries"] == 2
        assert compacted["compacted_delta_input_bytes"] > 0
        assert compacted["delta_run_count"] == 1
        assert compacted["delta_entry_count"] == 1
        assert compacted["reclaimed_delta_run_count"] == 2
        assert compacted["reclaimed_delta_bytes"] > 0
        assert compacted["reclamation_complete"] is True
        _assert_same_bits(
            collection.get([1, 2]),
            np.stack([second[0], base[1]]),
        )

        with pytest.raises(TypeError, match="max_runs.*bool"):
            collection.compact_deltas(max_runs=True)
        with pytest.raises(ValueError, match="max_runs.*between 2 and 64"):
            collection.compact_deltas(max_runs=1)
        with pytest.raises(ValueError, match="max_input_bytes.*between"):
            collection.compact_deltas(max_input_bytes=0)
        with pytest.raises(TypeError, match="max_input_entries.*integer"):
            collection.compact_deltas(max_input_entries=1.5)

    with throwdb.Collection.open(path) as reopened:
        assert reopened.mutation_log_stats["delta_run_count"] == 1
        _assert_same_bits(
            reopened.get([1, 2]),
            np.stack([second[0], base[1]]),
        )
        with pytest.raises(
            RuntimeError,
            match=r"mode='r\+'",
        ):
            reopened.compact_deltas()


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

    wal_paths = list((path / "wal").glob("*.wal"))
    assert len(wal_paths) == 1
    wal = wal_paths[0]
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
        format_version=6,
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
