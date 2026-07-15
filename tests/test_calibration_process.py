from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import throwdb
import throwdb._calibration_process as calibration_process
import throwdb.collection as collection_module


def _create_store(path: Path) -> tuple[dict[str, object], int]:
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=4),
        shard_size_bytes=1 << 20,
        format_version=6,
    ) as collection:
        collection.add(
            np.arange(64, dtype=np.uint64),
            np.arange(64 * 4, dtype=np.float16).reshape(64, 4),
        )
    return collection_module._layout_identity(path)


def test_isolated_calibration_executes_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    layout, record_size = _create_store(path)
    root_status = path.stat()
    progress: list[str] = []

    report = calibration_process.run_isolated_calibration(
        path,
        record_size=record_size,
        current_options=None,
        seed=17,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        deadline=time.perf_counter() + 30.0,
        expected_dimension=int(layout["dimension"]),
        expected_dtype=str(layout["dtype"]),
        expected_format_version=int(layout["format_version"]),
        expected_root_identity=(root_status.st_dev, root_status.st_ino),
        progress=progress.append,
    )

    process = report["measurement_process"]
    assert process["isolation"] == "subprocess_exec"  # type: ignore[index]
    assert process["status"] == "complete"  # type: ignore[index]
    assert process["pid"] != 0  # type: ignore[index]
    assert process["parent_verified"] is True  # type: ignore[index]
    assert report["format"] == "throwdb-reader-calibration"
    assert report["version"] == 1
    assert progress
    workload = report.get("workload")
    assert isinstance(workload, dict)
    preparations = workload.get("stage_cache_preparation")
    assert isinstance(preparations, dict)
    assert {"screening", "verification"}.issubset(preparations)
    assert all(
        isinstance(value, dict)
        and value.get("status")
        in {
            "performed",
            "unsupported",
            "failed",
            "no_payload_files",
            "deadline_expired",
            "inventory_limit_exceeded",
        }
        for value in preparations.values()
    )
    storage = report.get("storage_stability")
    if isinstance(storage, dict):
        assert storage["telemetry_scope"] == "isolated_calibration_process"
        assert storage["process_isolation"]["passed"] is True  # type: ignore[index]


def test_public_tune_uses_the_isolated_matrix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    _create_store(path)

    report = throwdb.Collection.tune(
        path,
        max_seconds=30.0,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        seed=19,
        dry_run=True,
    )

    process = report["measurement_process"]
    assert process["isolation"] == "subprocess_exec"  # type: ignore[index]
    assert process["status"] == "complete"  # type: ignore[index]
    assert report["status"] != "saved"
    selection = report.get("selection")
    if isinstance(selection, dict):
        assert selection["persisted"] is False


def test_worker_uses_anchored_directory_when_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    replacement = tmp_path / "replacement"
    anchored = tmp_path / "anchored-original"
    layout, record_size = _create_store(path)
    with throwdb.Collection.create(
        replacement,
        schema=throwdb.CollectionSchema(dimension=8),
        shard_size_bytes=1 << 20,
        format_version=6,
    ) as collection:
        collection.add(
            np.arange(64, dtype=np.uint64),
            np.arange(64 * 8, dtype=np.float16).reshape(64, 8),
        )
    root_status = path.stat()
    original_popen = calibration_process.subprocess.Popen

    def start_then_replace(*args: object, **kwargs: object) -> object:
        process = original_popen(*args, **kwargs)
        # The child cannot read its config until this wrapper returns. A
        # string-based reopen would now resolve the incompatible replacement;
        # the inherited directory FD must continue to identify the original.
        path.rename(anchored)
        replacement.rename(path)
        return process

    monkeypatch.setattr(
        calibration_process.subprocess,
        "Popen",
        start_then_replace,
    )

    report = calibration_process.run_isolated_calibration(
        path,
        record_size=record_size,
        current_options=None,
        seed=23,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        deadline=time.perf_counter() + 30.0,
        expected_dimension=int(layout["dimension"]),
        expected_dtype=str(layout["dtype"]),
        expected_format_version=int(layout["format_version"]),
        expected_root_identity=(root_status.st_dev, root_status.st_ino),
    )

    assert report["measurement_process"]["parent_verified"] is True  # type: ignore[index]
    assert anchored.exists()
    assert collection_module._layout_identity(path)[0]["dimension"] == 8


def test_isolated_calibration_refuses_to_start_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_spawn(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("expired calibration attempted to spawn a worker")

    monkeypatch.setattr(calibration_process.subprocess, "Popen", reject_spawn)

    report = calibration_process.run_isolated_calibration(
        b"/unused",
        record_size=8,
        current_options=None,
        seed=1,
        max_read_bytes=1 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        expected_dimension=4,
        expected_dtype="float16",
        expected_format_version=6,
        expected_root_identity=(1, 1),
        clock=lambda: 2.0,
    )

    assert report["status"] == "budget_exhausted"
    assert report["measurement_process"] == {
        "isolation": "subprocess_exec",
        "status": "terminated_at_deadline",
    }


def test_deadline_crossed_while_anchoring_does_not_spawn_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    original_open = calibration_process.os.open

    def delayed_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        now[0] = 2.0
        return descriptor

    def reject_spawn(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("expired root anchor attempted to spawn a worker")

    monkeypatch.setattr(calibration_process.os, "open", delayed_open)
    monkeypatch.setattr(calibration_process.subprocess, "Popen", reject_spawn)
    root_status = tmp_path.stat()

    report = calibration_process.run_isolated_calibration(
        tmp_path,
        record_size=8,
        current_options=None,
        seed=1,
        max_read_bytes=1 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        expected_dimension=4,
        expected_dtype="float16",
        expected_format_version=6,
        expected_root_identity=(root_status.st_dev, root_status.st_ino),
        clock=lambda: now[0],
    )

    assert report["status"] == "budget_exhausted"
    assert "anchoring" in str(report["reason"])


def test_root_descriptor_is_closed_when_identity_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = calibration_process.os.open
    original_fstat = calibration_process.os.fstat
    descriptors: list[int] = []

    def capture_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail_fstat(descriptor: int) -> object:
        if descriptors and descriptor == descriptors[-1]:
            raise OSError("synthetic ESTALE")
        return original_fstat(descriptor)

    monkeypatch.setattr(calibration_process.os, "open", capture_open)
    monkeypatch.setattr(calibration_process.os, "fstat", fail_fstat)

    with pytest.raises(OSError, match="synthetic ESTALE"):
        calibration_process.run_isolated_calibration(
            tmp_path,
            record_size=8,
            current_options=None,
            seed=1,
            max_read_bytes=1 << 20,
            max_memory_bytes=1 << 20,
            deadline=time.perf_counter() + 10.0,
            expected_dimension=4,
            expected_dtype="float16",
            expected_format_version=6,
            expected_root_identity=(1, 1),
        )

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        original_fstat(descriptors[0])


def test_isolated_calibration_propagates_worker_open_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        calibration_process.CalibrationProcessError,
        match="isolated calibration|StorageError|cannot",
    ):
        calibration_process.run_isolated_calibration(
            tmp_path / "missing",
            record_size=8,
            current_options=None,
            seed=1,
            max_read_bytes=1 << 20,
            max_memory_bytes=1 << 20,
            deadline=time.perf_counter() + 10.0,
            expected_dimension=4,
            expected_dtype="float16",
            expected_format_version=6,
            expected_root_identity=(tmp_path.stat().st_dev, -1),
        )
