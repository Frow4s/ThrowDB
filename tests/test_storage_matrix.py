from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_storage_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "throwdb_run_storage_matrix",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _scenario(name: str, throughput: float) -> dict[str, object]:
    return {
        "name": name,
        "summary": {
            "median_throughput_rows_per_second": throughput,
            "median_p95_call_seconds": 0.01,
            "throughput_cv": 0.01,
        },
    }


def _candidate(name: str, throughput: float) -> dict[str, object]:
    return {
        "name": name,
        "status": "measured",
        "score": throughput / 100.0,
        "options": {
            "io_backend": "thread_pool",
            "io_workers": 0 if name == "sequential" else 16,
        },
        "guardrails": {
            "eligible": True,
            "reasons": [],
        },
        "scenarios": [
            _scenario("serial_1024", throughput),
            _scenario("parallel_1024", throughput * 2.0),
            _scenario("serial_10000", throughput * 1.1),
        ],
    }


def _report(kind: str, throughput: float = 1_000.0) -> dict[str, object]:
    winner = _candidate("thread_pool_w16", throughput)
    return {
        "format": "throwdb-reader-calibration",
        "version": 1,
        "status": "dry_run",
        "reason": "bounded matrix completed",
        "selection": {
            "winner": "thread_pool_w16",
            "score": throughput / 100.0,
            "options": winner["options"],
            "persisted": False,
            "profile": None,
        },
        "stability": {
            "passed": True,
            "composite_drift": 0.01,
        },
        "sample": {
            "complete": True,
            "returned_rows": 1_024,
        },
        "candidate_discovery": {"unavailable": []},
        "candidates": [
            _candidate("sequential", 100.0),
            winner,
        ],
        "workload": {
            "cache_control": "ambient_global_disjoint_interleaved",
            "repeats": 3,
            "scenarios": [
                {"name": "serial_1024"},
                {"name": "parallel_1024"},
                {"name": "serial_10000"},
            ],
        },
        "budget": {
            "elapsed_seconds": 10.0,
            "planned_read_bytes": 1 << 20,
            "max_read_bytes": 8 << 30,
            "max_memory_bytes": 512 << 20,
        },
        "target": {
            "stable_at_publication": True,
            "reuse_scope": "compatible" if kind == "overlay" else "durable",
            "profile_reusable": True,
            "profile_reuse_level": (
                "compatible" if kind == "overlay" else "exact"
            ),
            "identity_warning": (
                "opaque compatible target" if kind == "overlay" else None
            ),
            "device": {
                "kind": kind,
                "confidence": "high",
                "evidence": ["fixture"],
            },
            "layout": {
                "format_version": 6,
                "dimension": 1_024,
                "dtype": "float16",
            },
            "mutation_snapshot": {
                "committed_sequence": 0,
                "delta_run_count": 0,
            },
            "open_stats": {
                "total_seconds": 2.0,
                "index_seconds": 1.9,
                "index_file_bytes": 4_563_406_848,
            },
        },
    }


def _stores(tmp_path: Path) -> tuple[Path, Path]:
    local = tmp_path / "local-store"
    ceph = tmp_path / "ceph-store"
    local.mkdir()
    ceph.mkdir()
    return local, ceph


def test_target_parser_accepts_equals_in_path_and_rejects_unsafe_names() -> None:
    target = BENCHMARK._target_spec("ceph=/mnt/a=b/store")
    assert target.name == "ceph"
    assert target.path == Path("/mnt/a=b/store")

    for value in ("missing-separator", "=empty", "../x=/tmp", "a/b=/tmp"):
        with pytest.raises(Exception):
            BENCHMARK._target_spec(value)


def test_storage_matrix_runs_targets_sequentially_with_identical_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    results = tmp_path / "results"
    calls: list[tuple[Path, dict[str, object]]] = []

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append((Path(path), dict(kwargs)))
        return copy.deepcopy(
            _report(
                "overlay" if Path(path) == local else "network",
                1_000.0 if Path(path) == local else 500.0,
            )
        )

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "collect_system_information",
        lambda path: {"path": str(path)},
    )

    exit_code = BENCHMARK.main(
        [
            "--target",
            f"local={local}",
            "--target",
            f"ceph={ceph}",
            "--results",
            str(results),
            "--max-seconds",
            "12.5",
            "--max-read-gib",
            "3",
            "--max-memory-mib",
            "64",
            "--seed",
            "91",
            "--no-monitors",
        ]
    )

    assert exit_code == 0
    assert [path for path, _ in calls] == [local, ceph]
    for _, kwargs in calls:
        assert kwargs["dry_run"] is True
        assert kwargs["max_seconds"] == 12.5
        assert kwargs["max_read_bytes"] == 3 << 30
        assert kwargs["max_memory_bytes"] == 64 << 20
        assert kwargs["seed"] == 91

    local_report = json.loads(
        (results / "local-tune.json").read_text(encoding="utf-8")
    )
    ceph_report = json.loads(
        (results / "ceph-tune.json").read_text(encoding="utf-8")
    )
    assert local_report["target"]["device"]["kind"] == "overlay"
    assert ceph_report["target"]["device"]["kind"] == "network"

    aggregate = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert aggregate["status"] == "complete"
    assert aggregate["configuration"]["target_order"] == ["local", "ceph"]
    assert aggregate["targets"]["local"]["status"] == "complete"
    assert aggregate["targets"]["ceph"]["status"] == "complete"
    assert (
        aggregate["targets"]["local"]["summary"]["profile_reuse_level"]
        == "compatible"
    )
    assert (
        aggregate["targets"]["ceph"]["summary"]["profile_reuse_level"]
        == "exact"
    )
    comparison = aggregate["comparison"]
    assert comparison["comparable"] is True
    candidate = next(
        item
        for item in comparison["candidates"]
        if item["name"] == "thread_pool_w16"
    )
    ratios = candidate["throughput_ratio_to_reference"]["ceph"]
    assert ratios == {
        "parallel_1024": 0.5,
        "serial_10000": 0.5,
        "serial_1024": 0.5,
    }


def test_target_failure_does_not_prevent_the_second_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    results = tmp_path / "results"
    calls: list[Path] = []

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        calls.append(Path(path))
        if Path(path) == local:
            raise RuntimeError("fixture local failure")
        return _report("network")

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "collect_system_information",
        lambda path: {},
    )

    exit_code = BENCHMARK.main(
        [
            "--target",
            f"local={local}",
            "--target",
            f"ceph={ceph}",
            "--results",
            str(results),
            "--no-monitors",
        ]
    )

    assert exit_code == 1
    assert calls == [local, ceph]
    local_error = json.loads(
        (results / "local-tune.json").read_text(encoding="utf-8")
    )
    assert local_error["status"] == "failed"
    assert local_error["error"]["message"] == "fixture local failure"
    aggregate = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert aggregate["status"] == "partial_failure"
    assert aggregate["targets"]["local"]["status"] == "failed"
    assert aggregate["targets"]["ceph"]["status"] == "complete"


@pytest.mark.parametrize(
    "status",
    ["budget_exhausted", "unstable", "target_changed"],
)
def test_incomplete_calibration_status_is_not_success(
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    results = tmp_path / "results"

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        report = _report("local_block" if Path(path) == local else "network")
        if Path(path) == local:
            report["status"] = status
        return report

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "collect_system_information",
        lambda path: {},
    )

    exit_code = BENCHMARK.main(
        [
            "--target",
            f"local={local}",
            "--target",
            f"ceph={ceph}",
            "--results",
            str(results),
            "--no-monitors",
        ]
    )

    assert exit_code == 1
    aggregate = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert aggregate["status"] == "partial_failure"
    assert aggregate["targets"]["local"]["status"] == "inconclusive"


def test_nonserializable_target_report_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    results = tmp_path / "results"
    calls: list[Path] = []

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        calls.append(Path(path))
        report = _report("local_block" if Path(path) == local else "network")
        if Path(path) == local:
            report["not_json"] = math.nan
        return report

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "collect_system_information",
        lambda path: {},
    )

    assert BENCHMARK.main(
        [
            "--target",
            f"local={local}",
            "--target",
            f"ceph={ceph}",
            "--results",
            str(results),
            "--no-monitors",
        ]
    ) == 1
    assert calls == [local, ceph]
    local_error = json.loads(
        (results / "local-tune.json").read_text(encoding="utf-8")
    )
    assert local_error["status"] == "failed"
    assert local_error["error"]["type"] == "ValueError"
    assert not (results / "local-tune.json.tmp").exists()


def test_report_claiming_persistence_violates_read_only_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    results = tmp_path / "results"

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        report = _report("local_block" if Path(path) == local else "network")
        if Path(path) == local:
            selection = report["selection"]
            assert isinstance(selection, dict)
            selection["persisted"] = True
        return report

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "collect_system_information",
        lambda path: {},
    )

    assert BENCHMARK.main(
        [
            "--target",
            f"local={local}",
            "--target",
            f"ceph={ceph}",
            "--results",
            str(results),
            "--no-monitors",
        ]
    ) == 1
    aggregate = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert aggregate["targets"]["local"]["status"] == "inconclusive"
    message = aggregate["targets"]["local"]["error"]["message"]
    assert "selection.persisted" in message


def test_results_directory_must_not_be_inside_a_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, ceph = _stores(tmp_path)
    called = False

    def tune(path: Path, **kwargs: object) -> dict[str, object]:
        nonlocal called
        del path, kwargs
        called = True
        return _report("local_block")

    monkeypatch.setattr(BENCHMARK.throwdb.Collection, "tune", tune)
    with pytest.raises(SystemExit) as stopped:
        BENCHMARK.main(
            [
                "--target",
                f"local={local}",
                "--target",
                f"ceph={ceph}",
                "--results",
                str(local / "benchmark-output"),
                "--no-monitors",
            ]
        )
    assert stopped.value.code == 2
    assert called is False
