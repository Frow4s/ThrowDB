from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_load_test.py"
    spec = importlib.util.spec_from_file_location("throwdb_run_load_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_reader_backend_arguments_are_parsed_and_bounded() -> None:
    parser = BENCHMARK.build_parser()
    args = parser.parse_args(
        [
            "--io-backend",
            "io_uring",
            "--io-uring-workers",
            "2",
            "--io-uring-queue-depth",
            "64",
        ]
    )
    assert args.io_backend == "io_uring"
    assert args.io_uring_workers == 2
    assert args.io_uring_queue_depth == 64

    for argv in (
        ["--io-backend", "unknown"],
        ["--io-uring-queue-depth", "0"],
        ["--io-uring-queue-depth", "4097"],
        ["--io-uring-workers", "0"],
        ["--io-uring-workers", "65"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_cgroup_v2_limit_and_counters_are_parsed(tmp_path: Path) -> None:
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/workload\n", encoding="utf-8")
    cgroup_root = tmp_path / "cgroup"
    workload = cgroup_root / "workload"
    workload.mkdir(parents=True)
    (workload / "cpu.max").write_text("1000000 100000\n", encoding="utf-8")
    (workload / "cpu.stat").write_text(
        "usage_usec 123\n"
        "nr_periods 10\n"
        "nr_throttled 3\n"
        "throttled_usec 50000\n",
        encoding="utf-8",
    )

    found = BENCHMARK.find_cgroup_v2_directory(proc_cgroup, cgroup_root)
    assert found == workload
    information = BENCHMARK.collect_cgroup_cpu_information(found)
    assert information["available"] is True
    assert information["effective_cpu_limit"] == 10.0
    assert information["stat"]["nr_throttled"] == 3


def test_cgroup_scenario_delta_has_derived_throttle_metrics() -> None:
    result: dict[str, object] = {}
    BENCHMARK.add_process_metrics(
        result,
        {"max_rss_raw": 1, "user_cpu_seconds": 1.0},
        {"max_rss_raw": 2, "user_cpu_seconds": 1.5},
        {},
        {},
        {
            "usage_usec": 100,
            "nr_periods": 10,
            "nr_throttled": 2,
            "throttled_usec": 1_000,
        },
        {
            "usage_usec": 600,
            "nr_periods": 14,
            "nr_throttled": 4,
            "throttled_usec": 26_000,
        },
        100,
        0.5,
    )

    delta = result["cgroup_cpu_delta"]
    assert isinstance(delta, dict)
    assert delta["nr_periods"] == 4
    assert delta["nr_throttled"] == 2
    assert delta["throttled_seconds"] == 0.025
    assert delta["throttled_period_fraction"] == 0.5
    assert delta["throttled_seconds_per_wall_second"] == 0.05
