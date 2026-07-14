from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import throwdb
import throwdb._calibration as calibration_module
import throwdb.collection as collection_module
from throwdb._device import MediaIdentity, RuntimeIdentity, StorageProbeResult
from throwdb._tuning import TuningRegistryError


def _vectors(rows: int, dimension: int) -> np.ndarray:
    return np.arange(rows * dimension, dtype=np.float16).reshape(rows, dimension)


def _create_collection(path: Path) -> np.ndarray:
    vectors = _vectors(3, 4)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=4),
        shard_size_bytes=1 << 20,
    ) as collection:
        collection.add(np.array([11, 22, 33], dtype=np.uint64), vectors)
    return vectors


def _probe(device_byte: str) -> StorageProbeResult:
    return StorageProbeResult(
        media=MediaIdentity(
            fingerprint=device_byte * 64,
            algorithm="sha256-length-framed-v1",
            schema_version=1,
            kind="local_block",
            confidence="high",
            reuse_scope="durable",
            evidence=("wwid",),
            component_count=1,
        ),
        runtime=RuntimeIdentity(
            fingerprint="f" * 64,
            algorithm="sha256-length-framed-v1",
            schema_version=1,
            evidence=("kernel", "reader_abi"),
            affinity_cpu_count=4,
            cpu_quota_limited=False,
            io_uring_disabled=0,
        ),
        probed_path_count=1,
        used_root_fallback=False,
    )


def _unsafe_probe() -> StorageProbeResult:
    base = _probe("a")
    return StorageProbeResult(
        media=MediaIdentity(
            fingerprint="0" * 64,
            algorithm=base.media.algorithm,
            schema_version=base.media.schema_version,
            kind="overlay",
            confidence="low",
            reuse_scope="none",
            evidence=("overlay", "non_reusable"),
            component_count=1,
        ),
        runtime=base.runtime,
        probed_path_count=1,
        used_root_fallback=False,
    )


def _compatible_overlay_probe() -> StorageProbeResult:
    base = _probe("a")
    return StorageProbeResult(
        media=MediaIdentity(
            fingerprint="c" * 64,
            algorithm=base.media.algorithm,
            schema_version=base.media.schema_version,
            kind="overlay",
            confidence="low",
            reuse_scope="compatible",
            evidence=("performance_compatibility", "virtual_filesystem"),
            component_count=1,
        ),
        runtime=base.runtime,
        probed_path_count=1,
        used_root_fallback=False,
    )


def _capability_sensitive_probe(
    *args: object,
    **kwargs: object,
) -> StorageProbeResult:
    policy = kwargs.get("io_uring_policy", "auto")
    runtime_byte = (
        "e"
        if not isinstance(policy, dict) or policy["io_uring_available"]
        else "f"
    )
    base = _probe("a")
    return StorageProbeResult(
        media=base.media,
        runtime=RuntimeIdentity(
            fingerprint=runtime_byte * 64,
            algorithm=base.runtime.algorithm,
            schema_version=base.runtime.schema_version,
            evidence=base.runtime.evidence,
            affinity_cpu_count=base.runtime.affinity_cpu_count,
            cpu_quota_limited=base.runtime.cpu_quota_limited,
            io_uring_disabled=base.runtime.io_uring_disabled,
        ),
        probed_path_count=base.probed_path_count,
        used_root_fallback=base.used_root_fallback,
    )


def _unavailable_io_uring(
    queue_depth: int = 32,
    workers: int = 1,
) -> dict[str, object]:
    return {
        "io_uring_available": False,
        "io_uring_probe_queue_depth": queue_depth,
        "io_uring_probe_workers": workers,
        "io_uring_reason": "disabled by deterministic test fixture",
    }


def _completed_calibration(*args: object, **kwargs: object) -> dict[str, object]:
    del args, kwargs
    return {
        "format": "throwdb-reader-calibration",
        "version": 1,
        "status": "complete",
        "reason": "fixture completed",
        "candidate_discovery": {},
        "sample": {},
        "workload": {},
        "candidates": [],
        "stability": {"passed": True},
        "selection": {
            "winner": "sequential",
            "options": dict(calibration_module.READER_DEFAULTS),
            "score": 1.0,
            "reason": "fixture",
        },
        "planned_read_bytes": 0,
    }


def test_bounded_tune_dry_run_then_persists_and_auto_reuses_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    expected = _create_collection(path)
    current_probe = _probe("a")
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: current_probe,
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )
    monkeypatch.setattr(
        calibration_module,
        "_effective_cpu_count",
        lambda: (
            2,
            {
                "reported": 2,
                "affinity": 2,
                "cgroup_quota": None,
                "effective": 2,
            },
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "_baseline_drift",
        lambda baseline, scenario_names: {
            "passed": True,
            "composite_drift": 0.0,
            "scenario_ratios": [1.0 for _ in scenario_names],
            "unstable_scenarios": [],
            "limit": 0.20,
        },
    )

    dry = throwdb.Collection.tune(
        path,
        max_seconds=30,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        seed=17,
        dry_run=True,
    )
    assert dry["status"] == "dry_run"
    assert dry["selection"]["persisted"] is False  # type: ignore[index]
    assert dry["target"]["profile_reusable"] is True  # type: ignore[index]
    assert dry["target"]["profile_reuse_level"] == "exact"  # type: ignore[index]
    assert dry["target"]["identity_warning"] is None  # type: ignore[index]
    assert not (path / "tuning").exists()
    json.dumps(dry, allow_nan=False)

    saved = throwdb.Collection.tune(
        path,
        max_seconds=30,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        seed=17,
    )
    assert saved["status"] == "saved"
    assert saved["configuration"]["matrix_version"] == 2  # type: ignore[index]
    selection = saved["selection"]
    assert isinstance(selection, dict)
    assert selection["persisted"] is True
    assert selection["profile"]["source"] == "calibrated-v2"
    assert saved["budget"]["planned_read_bytes"] <= 16 << 20  # type: ignore[index]
    assert saved["sample"]["inspected_slots"] <= 8_388_608  # type: ignore[index]
    assert saved["target"]["open_stats"]["index_file_bytes"] > 0  # type: ignore[index]
    json.dumps(saved, allow_nan=False)

    with throwdb.Collection.open(path, tuning="persisted") as reopened:
        assert reopened.tuning is not None
        assert reopened.tuning["source"] == "persisted"
        assert reopened.tuning["match_level"] == "exact"
        assert reopened.tuning["profile_origin"] == "calibrated-v2"
        np.testing.assert_array_equal(reopened.get([33, 11]), expected[[2, 0]])


def test_compatible_overlay_tune_persists_selects_and_removes_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    expected = _create_collection(path)
    probe = _compatible_overlay_probe()
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: probe,
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )
    monkeypatch.setattr(
        calibration_module,
        "run_bounded_calibration",
        _completed_calibration,
    )

    report = throwdb.Collection.tune(
        path,
        max_seconds=30,
        max_read_bytes=16 << 20,
        max_memory_bytes=8 << 20,
        seed=17,
    )

    assert report["status"] == "saved"
    assert report["configuration"]["dry_run"] is False  # type: ignore[index]
    assert report["selection"]["persisted"] is True  # type: ignore[index]
    target = report["target"]
    assert target["device"]["kind"] == "overlay"  # type: ignore[index]
    assert target["reuse_scope"] == "compatible"  # type: ignore[index]
    assert target["profile_reusable"] is True  # type: ignore[index]
    assert target["profile_reuse_level"] == "compatible"  # type: ignore[index]
    assert "performance compatibility signature" in target[  # type: ignore[index]
        "identity_warning"
    ]
    registry_path = path / "tuning" / "reader-profiles.json"
    registry_before_open = registry_path.read_bytes()

    with throwdb.Collection.open(path, tuning="persisted") as reopened:
        profile = reopened.tuning
        assert profile is not None
        assert profile["source"] == "persisted"
        assert profile["status"] == "selected"
        assert profile["match_level"] == "compatible"
        assert profile["profile_reuse_level"] == "compatible"
        assert profile["compatibility"]["match_level"] == "compatible"  # type: ignore[index]
        np.testing.assert_array_equal(reopened.get([33, 11]), expected[[2, 0]])
    assert registry_path.read_bytes() == registry_before_open

    listing = throwdb.Collection.tuning_profiles(path)
    assert listing["status"] == "ok"
    assert listing["current"]["match_level"] == "compatible"  # type: ignore[index]
    assert listing["current"]["selection_status"] == "selected"  # type: ignore[index]
    assert listing["profiles"][0]["current_match"] is True  # type: ignore[index]
    assert (  # type: ignore[index]
        listing["profiles"][0]["current_match_level"] == "compatible"
    )

    removed = throwdb.Collection.remove_tuning_profile(path)
    assert removed["status"] == "removed"
    assert removed["removed"] is True
    with throwdb.Collection.open(path) as fallback:
        assert fallback.tuning is not None
        assert fallback.tuning["source"] == "fallback"
        assert fallback.tuning["status"] == "no_match"
        assert fallback.tuning["match_level"] == "fallback"


def test_tune_allows_non_reusable_identity_only_for_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    calibration_calls = 0

    def complete(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calibration_calls
        calibration_calls += 1
        return _completed_calibration(*args, **kwargs)

    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _unsafe_probe(),
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )
    monkeypatch.setattr(calibration_module, "run_bounded_calibration", complete)

    reports = [
        throwdb.Collection.tune(
            path,
            max_seconds=30,
            max_read_bytes=16 << 20,
            max_memory_bytes=8 << 20,
            seed=seed,
            dry_run=True,
        )
        for seed in (17, 18)
    ]

    assert calibration_calls == 2
    assert not (path / "tuning").exists()
    for report in reports:
        assert report["status"] == "dry_run"
        assert report["selection"]["persisted"] is False  # type: ignore[index]
        target = report["target"]
        assert target["profile_reusable"] is False  # type: ignore[index]
        assert target["reuse_scope"] == "none"  # type: ignore[index]
        warning = target["identity_warning"]  # type: ignore[index]
        assert "cannot be persisted" in warning
        assert target["device"]["kind"] == "overlay"  # type: ignore[index]
        assert target["stable_at_publication"] is True  # type: ignore[index]
        json.dumps(report, allow_nan=False)


def test_tune_rejects_non_reusable_identity_before_open_without_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    original_native = collection_module.NativeCollection
    opened = False

    class NativeProxy:
        inspect_metadata = staticmethod(original_native.inspect_metadata)

        @staticmethod
        def open(*args: object, **kwargs: object) -> object:
            nonlocal opened
            del args, kwargs
            opened = True
            raise AssertionError("unsafe tune must fail before opening")

    monkeypatch.setattr(collection_module, "NativeCollection", NativeProxy)
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _unsafe_probe(),
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )

    with pytest.raises(
        TuningRegistryError,
        match=r"kind=overlay.*reuse_scope=none.*--dry-run",
    ):
        throwdb.Collection.tune(path, max_seconds=30)

    assert opened is False
    assert not (path / "tuning").exists()


def test_tune_does_not_publish_when_target_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    probes = iter((_probe("a"), _probe("a"), _probe("b")))
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: next(probes),
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )
    monkeypatch.setattr(
        calibration_module,
        "run_bounded_calibration",
        lambda *args, **kwargs: {
            "format": "throwdb-reader-calibration",
            "version": 1,
            "status": "complete",
            "reason": "fixture completed",
            "candidate_discovery": {},
            "sample": {},
            "workload": {},
            "candidates": [],
            "stability": {"passed": True},
            "selection": {
                "winner": "sequential",
                "options": dict(calibration_module.READER_DEFAULTS),
                "score": 1.0,
                "reason": "fixture",
            },
            "planned_read_bytes": 0,
        },
    )

    report = throwdb.Collection.tune(path, max_seconds=30)

    assert report["status"] == "target_changed"
    assert report["selection"]["persisted"] is False  # type: ignore[index]
    assert report["target"]["stable_at_publication"] is False  # type: ignore[index]
    assert not (path / "tuning").exists()


def test_tune_skips_final_probes_and_publication_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    now = [0.0]
    probe_calls = 0

    def probe(*args: object, **kwargs: object) -> StorageProbeResult:
        nonlocal probe_calls
        probe_calls += 1
        return _probe("a")

    def complete_then_expire(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        now[0] = 31.0
        return {
            "format": "throwdb-reader-calibration",
            "version": 1,
            "status": "complete",
            "reason": "fixture completed",
            "candidate_discovery": {},
            "sample": {},
            "workload": {},
            "candidates": [],
            "stability": {"passed": True},
            "selection": {
                "winner": "sequential",
                "options": dict(calibration_module.READER_DEFAULTS),
                "score": 1.0,
                "reason": "fixture",
            },
            "planned_read_bytes": 0,
        }

    monkeypatch.setattr(collection_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(collection_module, "probe_collection_storage", probe)
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )
    monkeypatch.setattr(
        calibration_module,
        "run_bounded_calibration",
        complete_then_expire,
    )

    report = throwdb.Collection.tune(path, max_seconds=30)

    assert report["status"] == "budget_exhausted"
    assert report["target"]["stable_at_publication"] is None  # type: ignore[index]
    assert probe_calls == 2
    assert not (path / "tuning").exists()


def test_tune_stops_after_a_collection_open_that_crosses_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    now = [0.0]
    probe_calls = 0
    original_native = collection_module.NativeCollection

    class NativeProxy:
        inspect_metadata = staticmethod(original_native.inspect_metadata)

        @staticmethod
        def open(*args: object, **kwargs: object) -> object:
            native = original_native.open(*args, **kwargs)
            now[0] = 31.0
            return native

    def probe(*args: object, **kwargs: object) -> StorageProbeResult:
        nonlocal probe_calls
        probe_calls += 1
        return _probe("a")

    monkeypatch.setattr(collection_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(collection_module, "NativeCollection", NativeProxy)
    monkeypatch.setattr(collection_module, "probe_collection_storage", probe)
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )

    report = throwdb.Collection.tune(path, max_seconds=30)

    assert report["status"] == "budget_exhausted"
    assert "opening the collection" in str(report["reason"])
    assert report["target"]["open_stats"] is not None  # type: ignore[index]
    assert probe_calls == 1
    assert not (path / "tuning").exists()


def test_tune_rechecks_deadline_after_the_open_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    now = [0.0]
    monkeypatch.setattr(collection_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _probe("a"),
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        _unavailable_io_uring,
    )

    report = throwdb.Collection.tune(
        path,
        max_seconds=30,
        progress=lambda message: now.__setitem__(0, 31.0),
    )

    assert report["status"] == "budget_exhausted"
    assert report["target"]["open_stats"] is None  # type: ignore[index]
    assert "before opening" in str(report["reason"])
    assert not (path / "tuning").exists()


def test_capability_probe_crossing_deadline_does_not_start_storage_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    now = [0.0]
    storage_probe_calls = 0

    def capabilities(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        now[0] = 31.0
        return _unavailable_io_uring()

    def storage_probe(*args: object, **kwargs: object) -> StorageProbeResult:
        nonlocal storage_probe_calls
        storage_probe_calls += 1
        return _probe("a")

    monkeypatch.setattr(collection_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(collection_module, "_reader_capabilities", capabilities)
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        storage_probe,
    )

    report = throwdb.Collection.tune(path, max_seconds=30)

    assert report["status"] == "budget_exhausted"
    assert "capability probe" in str(report["reason"])
    assert storage_probe_calls == 0
    assert report["target"]["open_stats"] is None  # type: ignore[index]
    assert not (path / "tuning").exists()


def test_profile_follows_collection_but_only_matches_the_same_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    expected = _create_collection(path)
    current_probe = _probe("a")
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: current_probe,
    )

    with throwdb.Collection.open(path) as initial:
        assert initial.tuning is not None
        assert initial.tuning["source"] == "fallback"
        assert initial.tuning["status"] == "registry_missing"
        np.testing.assert_array_equal(initial.get([33, 11]), expected[[2, 0]])
    assert not (path / "tuning").exists()

    saved = throwdb.Collection.save_tuning_profile(
        path,
        {
            "io_workers": 2,
            "io_queue_capacity": 4,
            "max_read_range_bytes": 4096,
        },
    )
    assert saved["status"] == "saved"
    registry_path = path / "tuning" / "reader-profiles.json"
    registry_before_open = registry_path.read_bytes()

    with throwdb.Collection.open(path) as selected:
        profile = selected.tuning
        assert profile is not None
        assert profile["source"] == "persisted"
        assert profile["status"] == "selected"
        assert profile["io_workers"] == 2
        assert profile["max_read_range_bytes"] == 4096
        compatibility = profile["compatibility"]
        assert isinstance(compatibility, dict)
        compatibility["device"] = {"kind": "mutated"}
        assert selected.tuning["compatibility"] != compatibility  # type: ignore[index]
    assert registry_path.read_bytes() == registry_before_open

    current_probe = _probe("b")
    with throwdb.Collection.open(path) as moved:
        assert moved.tuning is not None
        assert moved.tuning["source"] == "fallback"
        assert moved.tuning["status"] == "no_match"
    with pytest.raises(TuningRegistryError, match="no saved profile matches"):
        throwdb.Collection.open(path, tuning="persisted")

    current_probe = _probe("a")
    with throwdb.Collection.open(path) as returned:
        assert returned.tuning is not None
        assert returned.tuning["source"] == "persisted"

    listing = throwdb.Collection.tuning_profiles(path)
    assert listing["status"] == "ok"
    assert listing["profile_count"] == 1
    assert listing["profiles"][0]["current_match"] is True  # type: ignore[index]

    removed = throwdb.Collection.remove_tuning_profile(path)
    assert removed["removed"] is True
    with throwdb.Collection.open(path) as reopened:
        assert reopened.tuning is not None
        assert reopened.tuning["status"] == "no_match"


def test_corrupt_optional_registry_falls_back_but_strict_mode_reports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    expected = _create_collection(path)
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _probe("c"),
    )
    throwdb.Collection.save_tuning_profile(path, {"io_workers": 1})
    registry_path = path / "tuning" / "reader-profiles.json"
    registry_path.write_bytes(registry_path.read_bytes()[:-7] + b"broken!")

    with throwdb.Collection.open(path) as automatic:
        assert automatic.tuning is not None
        assert automatic.tuning["source"] == "fallback"
        assert automatic.tuning["status"] == "registry_corrupt"
        np.testing.assert_array_equal(automatic.get([22]), expected[[1]])

    with pytest.raises(TuningRegistryError, match="unusable"):
        throwdb.Collection.open(path, tuning="persisted")

    with throwdb.Collection.open(path, tuning={"io_workers": 0}) as manual:
        assert manual.tuning is not None
        assert manual.tuning["source"] == "manual"
        np.testing.assert_array_equal(manual.get([11]), expected[[0]])

    with pytest.raises(TuningRegistryError):
        throwdb.Collection.reset_tuning_profiles(path)
    reset = throwdb.Collection.reset_tuning_profiles(path, force=True)
    assert reset == {"status": "reset", "removed": True}
    with throwdb.Collection.open(path) as recovered:
        assert recovered.tuning is not None
        assert recovered.tuning["status"] == "registry_missing"


def test_conservative_mode_bypasses_a_matching_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _probe("d"),
    )
    throwdb.Collection.save_tuning_profile(path, {"io_workers": 2})

    with throwdb.Collection.open(path, tuning="conservative") as collection:
        assert collection.tuning is not None
        assert collection.tuning["mode"] == "conservative"
        assert collection.tuning["status"] == "registry_bypassed"
        assert collection.tuning["io_workers"] == 0


def test_io_uring_capability_change_invalidates_auto_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    available = {
        "io_uring_available": True,
        "io_uring_probe_queue_depth": 32,
        "io_uring_probe_workers": 1,
        "io_uring_reason": None,
    }
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        _capability_sensitive_probe,
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        lambda *args, **kwargs: dict(available),
    )
    throwdb.Collection.save_tuning_profile(
        path,
        {
            "io_backend": "io_uring",
            "io_uring_workers": 1,
            "io_uring_queue_depth": 32,
        },
    )

    available.update(
        io_uring_available=False,
        io_uring_reason="io_uring_setup denied",
    )
    with throwdb.Collection.open(path) as fallback:
        assert fallback.tuning is not None
        assert fallback.tuning["source"] == "fallback"
        assert fallback.tuning["status"] == "no_match"
        assert fallback.tuning["io_backend"] == "thread_pool"
    with pytest.raises(TuningRegistryError, match="no saved profile matches"):
        throwdb.Collection.open(path, tuning="persisted")


def test_unavailable_io_uring_profile_is_rejected_without_registry_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        lambda *args, **kwargs: {
            "io_uring_available": False,
            "io_uring_probe_queue_depth": 32,
            "io_uring_probe_workers": 1,
            "io_uring_reason": "io_uring_setup denied",
        },
    )

    with pytest.raises(TuningRegistryError, match="io_uring.*unavailable"):
        throwdb.Collection.save_tuning_profile(
            path,
            {"io_backend": "io_uring"},
        )
    assert not (path / "tuning").exists()


def test_exact_io_uring_topology_is_preflighted_on_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    _create_collection(path)
    exact_available = True

    def capabilities(
        queue_depth: int = 32,
        workers: int = 1,
    ) -> dict[str, object]:
        available = (queue_depth, workers) == (32, 1) or exact_available
        return {
            "io_uring_available": available,
            "io_uring_probe_queue_depth": queue_depth,
            "io_uring_probe_workers": workers,
            "io_uring_reason": None if available else "ring allocation denied",
        }

    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _probe("d"),
    )
    monkeypatch.setattr(
        collection_module,
        "_reader_capabilities",
        capabilities,
    )
    throwdb.Collection.save_tuning_profile(
        path,
        {
            "io_backend": "io_uring",
            "io_uring_workers": 2,
            "io_uring_queue_depth": 64,
        },
    )

    exact_available = False
    with throwdb.Collection.open(path) as fallback:
        assert fallback.tuning is not None
        assert fallback.tuning["source"] == "fallback"
        assert fallback.tuning["status"] == "backend_unavailable"
        assert fallback.tuning["io_backend"] == "thread_pool"
    with pytest.raises(TuningRegistryError, match="topology is unavailable"):
        throwdb.Collection.open(path, tuning="persisted")


def test_auto_retries_only_backend_initialization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    expected = _create_collection(path)
    monkeypatch.setattr(
        collection_module,
        "probe_collection_storage",
        lambda *args, **kwargs: _probe("e"),
    )
    throwdb.Collection.save_tuning_profile(path, {"io_workers": 2})

    native = collection_module.NativeCollection

    class ResourceLimitedNative:
        inspect_metadata = staticmethod(native.inspect_metadata)
        reader_capabilities = staticmethod(native.reader_capabilities)

        @staticmethod
        def open(*args: object, **kwargs: object) -> object:
            if kwargs.get("io_workers") == 2:
                raise throwdb.ReaderBackendError("thread creation denied")
            return native.open(*args, **kwargs)

    monkeypatch.setattr(
        collection_module,
        "NativeCollection",
        ResourceLimitedNative,
    )
    with throwdb.Collection.open(path) as fallback:
        assert fallback.tuning is not None
        assert fallback.tuning["source"] == "fallback"
        assert fallback.tuning["status"] == "backend_initialization_failed"
        assert fallback.tuning["io_workers"] == 0
        np.testing.assert_array_equal(fallback.get([22]), expected[[1]])

    with pytest.raises(throwdb.ReaderBackendError, match="thread creation"):
        throwdb.Collection.open(path, tuning="persisted")
    with pytest.raises(throwdb.ReaderBackendError, match="thread creation"):
        throwdb.Collection.open(path, tuning={"io_workers": 2})
