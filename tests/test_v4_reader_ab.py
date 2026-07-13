from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_v4_reader_ab.py"
    spec = importlib.util.spec_from_file_location("throwdb_v4_reader_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_default_matrix_is_the_fixed_v4_reader_comparison() -> None:
    parser = BENCHMARK.build_parser()
    args = parser.parse_args(
        ["--store", "/store", "--ids", "/ids.npy", "--results", "/results"]
    )
    scenarios = BENCHMARK.build_scenarios(args.calls_1024, args.calls_10000)

    assert args.cache_mode == "both"
    assert args.repeats == 3
    assert args.require_zero_delta is True
    assert [(scenario.batch_size, scenario.external_reader_workers) for scenario in scenarios] == [
        (1_024, 1),
        (1_024, 8),
        (1_024, 16),
        (10_000, 1),
        (10_000, 8),
        (10_000, 16),
    ]
    assert [scenario.calls_per_worker for scenario in scenarios] == [
        100,
        100,
        100,
        10,
        10,
        10,
    ]
    assert [policy.tuning() for policy in BENCHMARK.READER_POLICIES] == [
        {
            "io_backend": "thread_pool",
            "io_workers": 0,
            "io_queue_capacity": 0,
            "io_ranges_per_task": 8,
            "io_uring_workers": 1,
            "io_uring_queue_depth": 32,
        },
        {
            "io_backend": "thread_pool",
            "io_workers": 16,
            "io_queue_capacity": 32,
            "io_ranges_per_task": 8,
            "io_uring_workers": 1,
            "io_uring_queue_depth": 32,
        },
        {
            "io_backend": "io_uring",
            "io_workers": 0,
            "io_queue_capacity": 0,
            "io_ranges_per_task": 8,
            "io_uring_workers": 2,
            "io_uring_queue_depth": 32,
        },
    ]


def test_zero_delta_gate_has_an_explicit_opt_out() -> None:
    parser = BENCHMARK.build_parser()
    args = parser.parse_args(
        [
            "--store",
            "/store",
            "--ids",
            "/ids.npy",
            "--results",
            "/results",
            "--no-require-zero-delta",
        ]
    )

    assert args.require_zero_delta is False


def test_policy_order_rotates_but_queries_stay_paired() -> None:
    orders = [
        [policy.name for policy in BENCHMARK.rotated_policies(repetition)]
        for repetition in range(3)
    ]
    assert orders == [
        ["thread_pool_w0", "thread_pool_w16_r8", "io_uring_w2_qd32"],
        ["thread_pool_w16_r8", "io_uring_w2_qd32", "thread_pool_w0"],
        ["io_uring_w2_qd32", "thread_pool_w0", "thread_pool_w16_r8"],
    ]

    scenario = BENCHMARK.ReaderScenario(1_024, 8, 100)
    first_policy_seed = BENCHMARK.scenario_seed(100, 0, scenario)
    assert first_policy_seed == BENCHMARK.scenario_seed(100, 0, scenario)
    assert first_policy_seed != BENCHMARK.scenario_seed(100, 1, scenario)
    assert first_policy_seed != BENCHMARK.scenario_seed(
        100, 0, BENCHMARK.ReaderScenario(10_000, 8, 10)
    )


def test_measurement_blocks_are_scenario_first_and_policy_rotated() -> None:
    scenarios = (
        BENCHMARK.ReaderScenario(1_024, 1, 2),
        BENCHMARK.ReaderScenario(10_000, 8, 1),
    )

    blocks = BENCHMARK.build_measurement_blocks(2, scenarios)

    observed = [
        (
            block.repetition,
            block.scenario_index,
            block.scenario.batch_size,
            block.policy_order_position,
            block.policy.name,
        )
        for block in blocks
    ]
    assert observed == [
        (0, 0, 1_024, 0, "thread_pool_w0"),
        (0, 0, 1_024, 1, "thread_pool_w16_r8"),
        (0, 0, 1_024, 2, "io_uring_w2_qd32"),
        (0, 1, 10_000, 0, "thread_pool_w0"),
        (0, 1, 10_000, 1, "thread_pool_w16_r8"),
        (0, 1, 10_000, 2, "io_uring_w2_qd32"),
        (1, 0, 1_024, 0, "thread_pool_w16_r8"),
        (1, 0, 1_024, 1, "io_uring_w2_qd32"),
        (1, 0, 1_024, 2, "thread_pool_w0"),
        (1, 1, 10_000, 0, "thread_pool_w16_r8"),
        (1, 1, 10_000, 1, "io_uring_w2_qd32"),
        (1, 1, 10_000, 2, "thread_pool_w0"),
    ]


class _FakeStorage:
    count = 4
    dimension = 3
    dtype = "float16"
    format_version = 4

    def __init__(
        self,
        *,
        zero_delta: bool = True,
        corrupt: bool = False,
    ) -> None:
        self.zero_delta = zero_delta
        self.corrupt = corrupt

    def get_into_profiled(
        self,
        ids: np.ndarray,
        output: np.ndarray,
    ) -> dict[str, object]:
        output[:] = np.asarray(ids, dtype=np.float16)[:, None] + np.arange(
            self.dimension,
            dtype=np.float16,
        )
        if self.corrupt:
            output[0, 0] += np.float16(1)
        return {
            "total_seconds": 0.01,
            "lookup_seconds": 0.001,
            "physical_plan_seconds": 0.002,
            "physical_backend_seconds": 0.007,
            "requested_rows": len(ids),
            "base_rows": len(ids) if self.zero_delta else len(ids) - 1,
            "mutation_rows": 0 if self.zero_delta else 1,
            "planned_ranges": len(ids),
            "planned_read_bytes": len(ids) * self.dimension * 2,
            "backend_tasks": len(ids),
            "zero_delta_fast_path": self.zero_delta,
        }


def _install_preflight_fakes(
    monkeypatch: pytest.MonkeyPatch,
    storages: dict[str, _FakeStorage | Exception],
) -> tuple[list[str], list[dict[str, object]]]:
    opened: list[str] = []
    persisted: list[dict[str, object]] = []

    def open_policy(
        store: Path,
        policy: object,
    ) -> tuple[_FakeStorage, str, dict[str, object]]:
        del store
        name = str(policy.name)
        opened.append(name)
        value = storages[name]
        if isinstance(value, Exception):
            raise value
        return value, BENCHMARK.load_test.NATIVE_STORE_FORMAT, {
            "policy": name,
            "format_version": value.format_version,
        }

    monkeypatch.setattr(BENCHMARK, "_open_policy", open_policy)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "load_ids",
        lambda path, count: np.arange(count, dtype=np.uint64),
    )
    monkeypatch.setattr(BENCHMARK.load_test, "directory_size", lambda path: 123)
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "correctness_check",
        lambda *args: {"status": "skipped", "reason": "test"},
    )
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "close_benchmark_storage",
        lambda storage: None,
    )
    monkeypatch.setattr(
        BENCHMARK.load_test,
        "atomic_json_dump",
        lambda path, report: persisted.append(copy.deepcopy(report)),
    )
    return opened, persisted


def _preflight(
    tmp_path: Path,
    *,
    require_zero_delta: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    report: dict[str, object] = {}
    population = BENCHMARK.preflight_policies(
        store=tmp_path / "store",
        ids_path=tmp_path / "ids.npy",
        source_embeddings=None,
        source_kind="npy",
        seed=17,
        require_zero_delta=require_zero_delta,
        report=report,
        report_path=tmp_path / "report.json",
    )
    return report, {"population": population}


def test_preflight_checks_every_policy_against_baseline_and_persists_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storages = {
        policy.name: _FakeStorage() for policy in BENCHMARK.READER_POLICIES
    }
    opened, persisted = _install_preflight_fakes(monkeypatch, storages)

    report, result = _preflight(tmp_path, require_zero_delta=True)

    assert opened == [policy.name for policy in BENCHMARK.READER_POLICIES]
    np.testing.assert_array_equal(
        result["population"], np.arange(4, dtype=np.uint64)
    )
    assert report["preflight_status"] == "passed"
    assert [entry["status"] for entry in report["policy_preflight"]] == [
        "passed",
        "passed",
        "passed",
    ]
    assert all(
        entry["correctness"]["bit_exact_vs_baseline"] is True
        for entry in report["policy_preflight"]
    )
    assert all(
        entry["telemetry"]["zero_delta_fast_path"] is True
        for entry in report["policy_preflight"]
    )
    assert persisted[-1]["preflight_status"] == "passed"


def test_preflight_records_candidate_failure_and_still_checks_later_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storages: dict[str, _FakeStorage | Exception] = {
        "thread_pool_w0": _FakeStorage(),
        "thread_pool_w16_r8": _FakeStorage(corrupt=True),
        "io_uring_w2_qd32": _FakeStorage(),
    }
    opened, persisted = _install_preflight_fakes(monkeypatch, storages)
    report: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="not bit-exact"):
        BENCHMARK.preflight_policies(
            store=tmp_path / "store",
            ids_path=tmp_path / "ids.npy",
            source_embeddings=None,
            source_kind="npy",
            seed=17,
            require_zero_delta=True,
            report=report,
            report_path=tmp_path / "report.json",
        )

    assert opened == [policy.name for policy in BENCHMARK.READER_POLICIES]
    assert report["preflight_status"] == "failed"
    assert [entry["status"] for entry in report["policy_preflight"]] == [
        "passed",
        "failed",
        "passed",
    ]
    assert report["policy_preflight"][1]["correctness"][
        "bit_exact_vs_baseline"
    ] is False
    assert "not bit-exact" in report["policy_preflight"][1]["error"]["message"]
    assert persisted[-1]["preflight_status"] == "failed"


def test_preflight_persists_backend_open_failure_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storages: dict[str, _FakeStorage | Exception] = {
        "thread_pool_w0": _FakeStorage(),
        "thread_pool_w16_r8": _FakeStorage(),
        "io_uring_w2_qd32": RuntimeError("io_uring unavailable"),
    }
    opened, persisted = _install_preflight_fakes(monkeypatch, storages)
    report: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="io_uring unavailable"):
        BENCHMARK.preflight_policies(
            store=tmp_path / "store",
            ids_path=tmp_path / "ids.npy",
            source_embeddings=None,
            source_kind="npy",
            seed=17,
            require_zero_delta=True,
            report=report,
            report_path=tmp_path / "report.json",
        )

    assert opened == [policy.name for policy in BENCHMARK.READER_POLICIES]
    failed = report["policy_preflight"][2]
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "builtins.RuntimeError"
    assert failed["error"]["message"] == "io_uring unavailable"
    assert persisted[-1]["preflight_status"] == "failed"


def test_preflight_requires_zero_delta_unless_explicitly_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storages = {
        policy.name: _FakeStorage(zero_delta=False)
        for policy in BENCHMARK.READER_POLICIES
    }
    _install_preflight_fakes(monkeypatch, storages)
    report: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="--no-require-zero-delta"):
        BENCHMARK.preflight_policies(
            store=tmp_path / "store",
            ids_path=tmp_path / "ids.npy",
            source_embeddings=None,
            source_kind="npy",
            seed=17,
            require_zero_delta=True,
            report=report,
            report_path=tmp_path / "report.json",
        )
    assert report["preflight_status"] == "failed"

    report, _ = _preflight(tmp_path, require_zero_delta=False)
    assert report["preflight_status"] == "passed"


def _result(policy: str, repetition: int, throughput: float) -> dict[str, object]:
    return {
        "policy": policy,
        "repetition": repetition,
        "cache_mode": "cold",
        "batch_size": 1_024,
        "external_reader_workers": 8,
        "embeddings_per_second": throughput,
        "useful_gib_per_second": throughput / 100.0,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 2.0,
        "latency_p99_ms": 3.0,
        "latency_max_ms": 4.0,
    }


def test_summary_uses_paired_repetition_speedups() -> None:
    results = [
        _result("thread_pool_w0", 1, 100.0),
        _result("thread_pool_w16_r8", 1, 200.0),
        _result("thread_pool_w0", 2, 200.0),
        _result("thread_pool_w16_r8", 2, 600.0),
    ]

    summary = BENCHMARK.aggregate_results(results)
    candidate = next(
        row for row in summary if row["policy"] == "thread_pool_w16_r8"
    )

    assert candidate["embeddings_per_second"]["samples"] == [200.0, 600.0]
    assert candidate["embeddings_per_second"]["median"] == 400.0
    assert candidate["paired_speedup_vs_thread_pool_w0"]["samples"] == [2.0, 3.0]
    assert candidate["paired_speedup_vs_thread_pool_w0"]["median"] == 2.5


@pytest.mark.parametrize("flag", ["--repeats", "--calls-1024", "--calls-10000"])
def test_positive_counts_are_enforced(flag: str) -> None:
    parser = BENCHMARK.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--store",
                "/store",
                "--ids",
                "/ids.npy",
                "--results",
                "/results",
                flag,
                "0",
            ]
        )
