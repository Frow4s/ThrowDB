from __future__ import annotations

from pathlib import Path

import numpy as np

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
    ) as builder:
        builder.add([1, 2], _vectors(1))


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
