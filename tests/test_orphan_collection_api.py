from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import throwdb
import throwdb.collection as collection_module
from throwdb.collection import _orphan_report
from throwdb.cli import main as cli_main


def _build_store(path: Path, *, format_version: int = 7) -> None:
    vectors = np.arange(12, dtype=np.uint16).reshape(2, 6).view(np.float16)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=6, dtype="float16"),
        segment_size_bytes=1 << 20,
        format_version=format_version,
    ) as builder:
        builder.add(np.array([10, 20], dtype=np.uint32), vectors)


def _write_with_mtime(path: Path, payload: bytes, modified_unix_ns: int) -> None:
    path.write_bytes(payload)
    os.utime(path, ns=(modified_unix_ns, modified_unix_ns))


def test_orphan_policy_defers_pre_epoch_and_future_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_unix_ns = 10_000_000_000
    monkeypatch.setattr(
        collection_module,
        "time",
        SimpleNamespace(time_ns=lambda: snapshot_unix_ns),
    )
    report = _orphan_report(
        {
            "current_generation": 7,
            "current_catalog_id": "1" * 32,
            "entries": [
                {
                    "path": "segments/segment-" + "2" * 32 + ".dat",
                    "classification": "partial_unreferenced_segment_pair",
                    "candidate": True,
                    "bytes": 11,
                    "modified_unix_ns": -1,
                },
                {
                    "path": "wal/wal-" + "3" * 32 + ".wal",
                    "classification": "unreferenced_wal",
                    "candidate": True,
                    "bytes": 13,
                    "modified_unix_ns": snapshot_unix_ns + 1,
                },
            ],
        },
        min_age_seconds=0.0,
    )

    assert report["eligible_file_count"] == 0
    assert report["deferred_candidate_count"] == 2
    assert report["reclaimable_bytes"] == 0
    assert [entry["reason"] for entry in report["entries"]] == [
        "invalid_modification_time",
        "invalid_modification_time",
    ]
    assert all(not entry["eligible"] for entry in report["entries"])


def test_orphan_dry_run_classifies_age_and_never_removes_files(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    now_ns = time.time_ns()
    old_ns = now_ns - 60_000_000_000
    current_temp = store / f"CURRENT.{'1' * 32}.tmp"
    old_wal = store / "wal" / f"wal-{'2' * 32}.wal"
    young_wal = store / "wal" / f"wal-{'3' * 32}.wal"
    partial_segment = store / "segments" / f"segment-{'4' * 32}.dat"
    complete_data = store / "segments" / f"segment-{'5' * 32}.dat"
    complete_ids = store / "segments" / f"segment-{'5' * 32}.ids"
    unknown = store / "wal" / "user-notes.txt"

    _write_with_mtime(current_temp, b"tmp", old_ns)
    _write_with_mtime(old_wal, b"old-wal", old_ns)
    _write_with_mtime(young_wal, b"young-wal", now_ns)
    _write_with_mtime(partial_segment, b"partial", old_ns)
    _write_with_mtime(complete_data, b"segment-data", old_ns)
    _write_with_mtime(complete_ids, b"segment-ids", old_ns)
    _write_with_mtime(unknown, b"unknown", old_ns)

    expected_paths = {
        path.relative_to(store).as_posix()
        for path in (
            current_temp,
            old_wal,
            young_wal,
            partial_segment,
            complete_data,
            complete_ids,
            unknown,
        )
    }
    report = throwdb.Collection.inspect_orphans(
        store,
        min_age_seconds=10.0,
    )

    assert report["format"] == "throwdb-orphan-collection-plan"
    assert report["version"] == 1
    assert report["dry_run"] is True
    assert report["min_age_seconds"] == 10.0
    assert report["candidate_file_count"] == 6
    assert report["eligible_file_count"] == 5
    assert report["deferred_candidate_count"] == 1
    assert report["candidate_bytes"] == sum(
        path.stat().st_size
        for path in (
            current_temp,
            old_wal,
            young_wal,
            partial_segment,
            complete_data,
            complete_ids,
        )
    )
    assert report["reclaimable_bytes"] == sum(
        path.stat().st_size
        for path in (
            current_temp,
            old_wal,
            partial_segment,
            complete_data,
            complete_ids,
        )
    )
    assert report["unknown_file_count"] == 1
    assert report["classification_counts"]["temp_publication"] == 1
    assert report["classification_counts"]["unreferenced_wal"] == 2
    assert (
        report["classification_counts"][
            "complete_unreferenced_segment_pair"
        ]
        == 2
    )
    assert (
        report["classification_counts"][
            "partial_unreferenced_segment_pair"
        ]
        == 1
    )
    assert report["classification_counts"]["unknown_do_not_touch"] == 1
    assert report["total_bytes"] == sum(
        report["classification_bytes"].values()
    )
    assert [entry["path"] for entry in report["entries"]] == sorted(
        entry["path"] for entry in report["entries"]
    )

    by_path = {entry["path"]: entry for entry in report["entries"]}
    assert by_path[young_wal.relative_to(store).as_posix()]["eligible"] is False
    assert (
        by_path[young_wal.relative_to(store).as_posix()]["reason"]
        == "minimum_age_not_reached"
    )
    assert by_path[unknown.relative_to(store).as_posix()]["candidate"] is False
    assert (
        by_path[unknown.relative_to(store).as_posix()]["reason"]
        == "unknown_do_not_touch"
    )
    assert expected_paths <= set(by_path)
    assert all((store / relative).exists() for relative in expected_paths)

    with throwdb.Collection.open(store, mode="r+") as writer:
        instance_report = writer.collect_orphans(min_age_seconds=10.0)
        assert instance_report["current_catalog_id"] == report["current_catalog_id"]
        assert [
            (entry["path"], entry["classification"], entry["bytes"])
            for entry in instance_report["entries"]
        ] == [
            (entry["path"], entry["classification"], entry["bytes"])
            for entry in report["entries"]
        ]
        executed = writer.collect_orphans(
            dry_run=False,
            min_age_seconds=10.0,
        )
        assert executed["format"] == "throwdb-orphan-collection-report"
        assert executed["dry_run"] is False
        assert executed["eligible_file_count"] == 5
        assert executed["selected_file_count"] == 5
        assert executed["removed_file_count"] == 5
        assert executed["failed_file_count"] == 0
        assert executed["durability_complete"] is True
        assert executed["policy_complete"] is True
        assert executed["reclamation_complete"] is True
        assert executed["synced_directories"] == [".", "segments", "wal"]
        assert all(
            entry["durable"]
            for entry in executed["entries"]
            if entry["removed"]
        )

    with throwdb.Collection.open(store) as reader:
        with pytest.raises(RuntimeError, match=r"mode='r\+'"):
            reader.collect_orphans()

    assert not current_temp.exists()
    assert not old_wal.exists()
    assert young_wal.exists()
    assert not partial_segment.exists()
    assert not complete_data.exists()
    assert not complete_ids.exists()
    assert unknown.exists()


def test_orphan_execute_enforces_byte_budget_and_skips_oversized_entries(
    tmp_path: Path,
) -> None:
    store = tmp_path / "byte-budget"
    _build_store(store)
    oversized = store / "wal" / f"wal-{'1' * 32}.wal"
    fitting = store / "wal" / f"wal-{'2' * 32}.wal"
    oversized.write_bytes(b"0123456789")
    fitting.write_bytes(b"ok")

    with throwdb.Collection.open(store, mode="r+") as writer:
        report = writer.collect_orphans(
            dry_run=False,
            min_age_seconds=0,
            max_files=10,
            max_bytes=2,
        )

    assert report["eligible_file_count"] == 2
    assert report["selected_file_count"] == 1
    assert report["selected_bytes"] == 2
    assert report["budget_deferred_file_count"] == 1
    assert report["budget_deferred_bytes"] == 10
    assert report["removed_file_count"] == 1
    assert report["durability_complete"] is True
    assert report["policy_complete"] is False
    assert report["reclamation_complete"] is False
    assert oversized.exists()
    assert not fitting.exists()
    by_path = {entry["path"]: entry for entry in report["entries"]}
    assert by_path[oversized.relative_to(store).as_posix()]["reason"] == (
        "max_bytes_reached"
    )


def test_format7_orphan_plan_and_execution_collect_segment_singleton(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "partitioned"
    _build_store(store, format_version=7)
    pair_token = "1" * 32
    partial_token = "2" * 32
    snapshot_token = "3" * 32
    run_token = "4" * 32
    wal_token = "5" * 32
    pair_data = store / "segments" / f"segment-{pair_token}.dat"
    pair_ids = store / "segments" / f"segment-{pair_token}.ids"
    partial = store / "segments" / f"segment-{partial_token}.dat"
    snapshot = (
        store / "index" / "partitions"
        / f"partition-1-{snapshot_token}.idx"
    )
    run = store / "index" / "runs" / f"run-{run_token}.idx"
    wal = store / "wal" / f"wal-{wal_token}.wal"
    for path in (pair_data, pair_ids, partial, snapshot, run, wal):
        path.write_bytes(b"orphan")

    report = throwdb.Collection.inspect_orphans(
        store,
        min_age_seconds=0,
    )
    by_path = {entry["path"]: entry for entry in report["entries"]}
    assert by_path[pair_data.relative_to(store).as_posix()][
        "classification"
    ] == "complete_unreferenced_segment_pair"
    assert by_path[pair_ids.relative_to(store).as_posix()]["candidate"] is True
    assert by_path[partial.relative_to(store).as_posix()] == {
        "path": partial.relative_to(store).as_posix(),
        "classification": "partial_unreferenced_segment_pair",
        "bytes": len(b"orphan"),
        "modified_unix_ns": by_path[partial.relative_to(store).as_posix()][
            "modified_unix_ns"
        ],
        "candidate": True,
        "eligible": True,
        "reason": "minimum_age_reached",
    }
    assert by_path[snapshot.relative_to(store).as_posix()][
        "classification"
    ] == "unreferenced_snapshot"
    assert by_path[run.relative_to(store).as_posix()][
        "classification"
    ] == "unreferenced_run"

    cli_main(
        [
            "collect-orphans",
            "--storage",
            str(store),
            "--dry-run",
            "--min-age-seconds",
            "0",
        ]
    )
    cli_report = json.loads(capsys.readouterr().out)
    cli_by_path = {entry["path"]: entry for entry in cli_report["entries"]}
    assert cli_by_path[partial.relative_to(store).as_posix()][
        "candidate"
    ] is True
    assert cli_by_path[run.relative_to(store).as_posix()][
        "classification"
    ] == "unreferenced_run"

    with throwdb.Collection.open(store, mode="r+") as writer:
        through_writer = writer.collect_orphans(min_age_seconds=0)
        assert through_writer["current_catalog_id"] == report[
            "current_catalog_id"
        ]
        executed = writer.collect_orphans(
            dry_run=False,
            min_age_seconds=0,
        )

    assert executed["removed_file_count"] == 6
    assert executed["failed_file_count"] == 0
    assert executed["synced_directories"] == [
        "segments",
        "index/partitions",
        "index/runs",
        "wal",
    ]
    assert executed["durability_complete"] is True
    assert not pair_data.exists()
    assert not pair_ids.exists()
    assert not snapshot.exists()
    assert not run.exists()
    assert not wal.exists()
    assert not partial.exists()

    second_wal = store / "wal" / f"wal-{'6' * 32}.wal"
    second_wal.write_bytes(b"second")
    cli_main(
        [
            "collect-orphans",
            "--storage",
            str(store),
            "--execute",
            "--min-age-seconds",
            "0",
        ]
    )
    cli_execution = json.loads(capsys.readouterr().out)
    assert cli_execution["removed_file_count"] == 1
    assert cli_execution["synced_directories"] == ["wal"]
    assert not second_wal.exists()
    assert not partial.exists()


def test_format7_collects_superseded_catalogs_without_changing_current(
    tmp_path: Path,
) -> None:
    store = tmp_path / "partitioned-catalogs"
    _build_store(store, format_version=7)
    replacement = (
        np.arange(6, dtype=np.uint16) + np.uint16(701)
    ).reshape(1, 6).view(np.float16)
    with throwdb.Collection.open(store, mode="r+") as writer:
        writer.update([10], replacement)
        assert writer.rotate_wal()["rotated"] is True

    plan = throwdb.Collection.inspect_orphans(store, min_age_seconds=0)
    superseded = [
        entry
        for entry in plan["entries"]
        if entry["classification"] == "superseded_catalog"
    ]
    assert superseded
    assert all(entry["candidate"] and entry["eligible"] for entry in superseded)

    with throwdb.Collection.open(store, mode="r+") as writer:
        result = writer.collect_orphans(
            dry_run=False,
            min_age_seconds=0,
        )
        assert result["removed_file_count"] >= len(superseded)
        np.testing.assert_array_equal(
            writer.get([10]).view(np.uint16),
            replacement.view(np.uint16),
        )

    with throwdb.Collection.open(store, verify_index=True) as reader:
        np.testing.assert_array_equal(
            reader.get([10]).view(np.uint16),
            replacement.view(np.uint16),
        )


def test_format7_complete_segment_pair_uses_group_age_policy() -> None:
    snapshot_unix_ns = 20_000_000_000
    pair_token = "a" * 32
    report = _orphan_report(
        {
            "current_generation": 7,
            "current_catalog_id": "b" * 32,
            "entries": [
                {
                    "path": f"segments/segment-{pair_token}.dat",
                    "classification": "complete_unreferenced_segment_pair",
                    "candidate": True,
                    "bytes": 11,
                    "modified_unix_ns": snapshot_unix_ns - 10_000_000_000,
                },
                {
                    "path": f"segments/segment-{pair_token}.ids",
                    "classification": "complete_unreferenced_segment_pair",
                    "candidate": True,
                    "bytes": 13,
                    "modified_unix_ns": snapshot_unix_ns - 1,
                },
            ],
        },
        min_age_seconds=5,
        snapshot_unix_ns=snapshot_unix_ns,
    )

    assert report["candidate_file_count"] == 2
    assert report["eligible_file_count"] == 0
    assert report["reclaimable_bytes"] == 0
    assert {entry["reason"] for entry in report["entries"]} == {
        "minimum_age_not_reached"
    }


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_orphan_min_age_rejects_boolean_values(
    tmp_path: Path,
    value: object,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    with pytest.raises(TypeError, match="min_age_seconds.*bool"):
        throwdb.Collection.inspect_orphans(
            store,
            min_age_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", ["1", object(), None])
def test_orphan_min_age_requires_a_number(
    tmp_path: Path,
    value: object,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    with pytest.raises(TypeError, match="min_age_seconds must be a number"):
        throwdb.Collection.inspect_orphans(
            store,
            min_age_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, np.nan])
def test_orphan_min_age_requires_finite_nonnegative_value(
    tmp_path: Path,
    value: float,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    with pytest.raises(ValueError, match="finite and zero or greater"):
        throwdb.Collection.inspect_orphans(store, min_age_seconds=value)
