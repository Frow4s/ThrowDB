from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import throwdb
import throwdb.collection as collection_module
from throwdb.collection import _orphan_report


def _build_store(path: Path) -> None:
    vectors = np.arange(12, dtype=np.uint16).reshape(2, 6).view(np.float16)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=6, dtype="float16"),
        shard_size_bytes=1 << 20,
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
                    "path": "deltas/delta-" + "2" * 32 + ".dat",
                    "classification": "partial_unreferenced_delta_pair",
                    "bytes": 11,
                    "modified_unix_ns": -1,
                },
                {
                    "path": "wal/wal-" + "3" * 32 + ".wal",
                    "classification": "unreferenced_wal",
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
    partial_delta = store / "deltas" / f"delta-{'4' * 32}.dat"
    complete_data = store / "deltas" / f"delta-{'5' * 32}.dat"
    complete_index = store / "deltas" / f"delta-{'5' * 32}.idx"
    unknown = store / "wal" / "user-notes.txt"

    _write_with_mtime(current_temp, b"tmp", old_ns)
    _write_with_mtime(old_wal, b"old-wal", old_ns)
    _write_with_mtime(young_wal, b"young-wal", now_ns)
    _write_with_mtime(partial_delta, b"partial", old_ns)
    _write_with_mtime(complete_data, b"delta-data", old_ns)
    _write_with_mtime(complete_index, b"delta-index", old_ns)
    _write_with_mtime(unknown, b"unknown", old_ns)

    expected_paths = {
        path.relative_to(store).as_posix()
        for path in (
            current_temp,
            old_wal,
            young_wal,
            partial_delta,
            complete_data,
            complete_index,
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
            partial_delta,
            complete_data,
            complete_index,
        )
    )
    assert report["reclaimable_bytes"] == sum(
        path.stat().st_size
        for path in (
            current_temp,
            old_wal,
            partial_delta,
            complete_data,
            complete_index,
        )
    )
    assert report["unknown_file_count"] == 1
    assert report["classification_counts"]["temp_publication"] == 1
    assert report["classification_counts"]["unreferenced_wal"] == 2
    assert (
        report["classification_counts"][
            "complete_unreferenced_delta_pair"
        ]
        == 2
    )
    assert (
        report["classification_counts"][
            "partial_unreferenced_delta_pair"
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
        with pytest.raises(NotImplementedError, match="deletion is not implemented"):
            writer.collect_orphans(dry_run=False)

    with throwdb.Collection.open(store) as reader:
        with pytest.raises(RuntimeError, match=r"mode='r\+'"):
            reader.collect_orphans()

    assert all((store / relative).exists() for relative in expected_paths)


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
