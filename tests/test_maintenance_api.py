from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import throwdb


def _vectors(salt: int) -> np.ndarray:
    bits = np.arange(8, dtype=np.uint16) * np.uint16(7919)
    bits += np.uint16(salt)
    return bits.reshape(2, 4).view(np.float16)


def _build(path: Path) -> None:
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=4, dtype="float16"),
        shard_size_bytes=1 << 20,
        format_version=6,
    ) as builder:
        builder.add([1, 2], _vectors(1))


def _build_default_v7(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(1, 65, dtype=np.uint64)
    bits = (
        np.arange(ids.size * 1024, dtype=np.uint16) * np.uint16(7919)
        + np.uint16(313)
    )
    vectors = bits.reshape(ids.size, 1024).view(np.float16)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=1024, dtype="float16"),
        shard_size_bytes=vectors.nbytes,
    ) as builder:
        assert builder.format_version == 7
        builder.add(ids, vectors)
    return ids, vectors


def test_maintain_dry_run_is_non_mutating_and_explains_policy(
    tmp_path: Path,
) -> None:
    store = tmp_path / "dry-run"
    _build(store)
    with throwdb.Collection.open(store, mode="r+") as collection:
        collection.update([1], _vectors(101)[:1])
        before = dict(collection.mutation_log_stats or {})
        report = collection.maintain(
            dry_run=True,
            rotate_wal_bytes=1,
            orphan_min_age_seconds=0,
        )
        after = dict(collection.mutation_log_stats or {})

    assert report["format"] == "throwdb-maintenance-report"
    assert report["version"] == 1
    assert report["dry_run"] is True
    assert report["before"] == report["after"] == before == after
    assert report["work_performed"] is False
    assert report["actions_performed"] == 0
    assert [action["name"] for action in report["actions"]] == [
        "rotate_wal",
        "materialize_frozen_wals",
        "compact_deltas",
        "collect_orphans",
    ]
    assert report["actions"][0]["planned"] is True
    assert all(action["attempted"] is False for action in report["actions"])


def test_maintain_executes_rotate_then_bounded_materialization(
    tmp_path: Path,
) -> None:
    store = tmp_path / "execute"
    _build(store)
    expected = _vectors(201)[:1]
    with throwdb.Collection.open(store, mode="r+") as collection:
        collection.update([1], expected)
        report = collection.maintain(
            dry_run=False,
            rotate_wal_bytes=1,
            materialize_max_wals=1,
            materialize_max_input_bytes=1 << 20,
            compact_min_runs=8,
            orphan_min_age_seconds=0,
        )
        np.testing.assert_array_equal(
            collection.get([1]).view(np.uint16), expected.view(np.uint16)
        )

    assert report["dry_run"] is False
    assert report["actions_performed"] == 2
    assert report["work_performed"] is True
    rotate, materialize, compact, collect = report["actions"]
    assert rotate["performed"] is True
    assert materialize["performed"] is True
    assert materialize["result"]["materialized_wal_count"] == 1
    assert compact["attempted"] is False
    assert collect["attempted"] is True
    assert report["after"]["frozen_wal_count"] == 0
    assert report["after"]["delta_run_count"] == 1


def test_format6_maintain_rejects_format7_only_materialization_bounds(
    tmp_path: Path,
) -> None:
    store = tmp_path / "format6-materialization-bounds"
    _build(store)
    with throwdb.Collection.open(store, mode="r+") as collection:
        with pytest.raises(ValueError, match="require a format-7 collection"):
            collection.maintain(materialize_max_input_entries=10)


def test_maintain_hard_action_and_cooperative_time_budgets(
    tmp_path: Path,
) -> None:
    action_store = tmp_path / "action-budget"
    _build(action_store)
    with throwdb.Collection.open(action_store, mode="r+") as collection:
        collection.update([1], _vectors(51)[:1])
        limited = collection.maintain(
            dry_run=False,
            max_actions=1,
            rotate_wal_bytes=1,
        )
        assert limited["actions"][0]["performed"] is True
        assert limited["actions"][1]["reason"] == "maintenance_budget_reached"
        assert limited["action_budget_reached"] is True
        assert collection.mutation_log_stats["frozen_wal_count"] == 1

    time_store = tmp_path / "time-budget"
    _build(time_store)
    with throwdb.Collection.open(time_store, mode="r+") as collection:
        collection.update([1], _vectors(71)[:1])
        stopped = collection.maintain(
            dry_run=False,
            max_seconds=0,
            rotate_wal_bytes=1,
        )
        assert stopped["actions_performed"] == 0
        assert stopped["deadline_reached"] is True
        assert collection.mutation_log_stats["frozen_wal_count"] == 0


def test_format7_maintain_completes_selective_lifecycle_in_two_passes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "format7-lifecycle"
    ids, vectors = _build_default_v7(store)
    cleaner_policy = {
        "clean_max_candidate_segments": 1,
        "clean_max_sidecar_scan_bytes": 1 << 20,
        "clean_max_source_bytes": 2 << 20,
        "clean_max_output_bytes": 2 << 20,
        "clean_max_memory_bytes": 2 << 20,
        "clean_min_garbage_rows": 1,
        "clean_min_reclaimable_bytes": 0,
        "clean_min_garbage_ratio_ppm": 0,
    }

    with throwdb.Collection.open(store, mode="r+") as collection:
        deleted = collection.delete(ids[:-1])
        np.testing.assert_array_equal(deleted, np.ones(ids.size - 1, dtype=bool))
        old_reader = throwdb.Collection.open(store)
        before = dict(collection.mutation_log_stats or {})

        plan = collection.maintain(
            dry_run=True,
            rotate_wal_bytes=1,
            orphan_min_age_seconds=0,
            **cleaner_policy,
        )
        assert plan["version"] == 2
        assert plan["before"] == plan["after"] == before
        assert plan["policy"]["compact_min_runs"] == 1
        assert plan["policy"]["compact_max_runs"] == 4096
        assert [action["name"] for action in plan["actions"]] == [
            "rotate_wal",
            "materialize_frozen_wals",
            "compact_deltas",
            "clean_segments",
            "collect_orphans",
        ]
        assert all(not action["attempted"] for action in plan["actions"])
        assert collection._cleaning_cursor == 0

        first = collection.maintain(
            dry_run=False,
            max_actions=4,
            rotate_wal_bytes=1,
            materialize_max_wals=1,
            materialize_max_input_bytes=1 << 20,
            compact_max_input_bytes=8 << 20,
            compact_max_input_entries=1_000,
            orphan_min_age_seconds=0,
            **cleaner_policy,
        )
        assert [action["name"] for action in first["actions"]] == [
            "rotate_wal",
            "materialize_frozen_wals",
            "compact_deltas",
            "clean_segments",
            "collect_orphans",
        ]
        assert all(
            action["performed"] for action in first["actions"][:4]
        ), first["actions"]
        assert first["actions"][4]["reason"] == "maintenance_budget_reached"
        assert first["actions_performed"] == 4
        assert first["action_budget_reached"] is True
        assert first["after"]["delta_run_count"] == 1
        assert (
            collection._cleaning_cursor
            == first["actions"][3]["result"]["next_start_after_segment_id"]
        )
        np.testing.assert_array_equal(
            collection.get(ids[-1:]).view(np.uint16),
            vectors[-1:].view(np.uint16),
        )

        second = collection.maintain(
            dry_run=False,
            max_actions=5,
            compact_max_input_bytes=8 << 20,
            compact_max_input_entries=1_000,
            orphan_min_age_seconds=0,
            **cleaner_policy,
        )
        assert second["actions"][2]["name"] == "compact_deltas"
        assert second["actions"][2]["performed"] is True
        assert second["after"]["delta_run_count"] == 0
        assert second["actions"][4]["name"] == "collect_orphans"
        assert second["actions"][4]["attempted"] is True
        np.testing.assert_array_equal(
            old_reader.get(ids[-1:]).view(np.uint16),
            vectors[-1:].view(np.uint16),
        )
        old_reader.close()

    with throwdb.Collection.open(store, verify_index=True) as reopened:
        assert reopened.count == 1
        np.testing.assert_array_equal(
            reopened.get(ids[-1:]).view(np.uint16),
            vectors[-1:].view(np.uint16),
        )
