from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

import throwdb
import throwdb._calibration as calibration
import throwdb.collection as collection_module
from throwdb._tuning import READER_DEFAULTS


RESOURCE_SCENARIOS = ("serial_1024", "serial_10000")
ALL_SCENARIOS = (*RESOURCE_SCENARIOS, "hot_small_32")


def _epoch(*, cpu_cost: float, rows: int = 1_000) -> dict[str, object]:
    cpu_seconds = cpu_cost * rows / 1_000_000.0
    return {
        "elapsed_seconds": 1.0,
        "rows": rows,
        "useful_bytes": rows * 8,
        "process": {"cpu_seconds": cpu_seconds},
        "io": {},
        "cgroup": {},
        "quality": {},
    }


def _scenario(
    name: str,
    *,
    throughput: float,
    cpu_costs: Sequence[float],
    resource_objective: bool,
) -> dict[str, object]:
    repetitions = [
        {
            "throughput_rows_per_second": throughput,
            "p50_call_seconds": 0.008,
            "p95_call_seconds": 0.010,
            "p99_call_seconds": 0.012,
            "max_call_seconds": 0.013,
            "latency_sample_count": 64,
            "planned_read_bytes": 8_000,
            "resources": _epoch(cpu_cost=cpu_cost),
        }
        for cpu_cost in cpu_costs
    ]
    return {
        "name": name,
        "resource_objective": resource_objective,
        "latency_guardrail": False,
        "repetitions": repetitions,
        "summary": calibration._summarize_repetitions(repetitions),
    }


def _candidate(
    name: str,
    *,
    throughput: float,
    workers: int,
    resource_costs: Sequence[float],
    hot_costs: Sequence[float],
    aggregate_cpu_cost: float,
) -> dict[str, object]:
    return {
        "name": name,
        "options": {
            **READER_DEFAULTS,
            "io_workers": workers,
            "io_queue_capacity": workers * 2,
        },
        "status": "measured",
        "scenarios": [
            *(
                _scenario(
                    scenario,
                    throughput=throughput,
                    cpu_costs=resource_costs,
                    resource_objective=True,
                )
                for scenario in RESOURCE_SCENARIOS
            ),
            _scenario(
                "hot_small_32",
                throughput=throughput,
                cpu_costs=hot_costs,
                resource_objective=False,
            ),
        ],
        # Deliberately contradictory: V3 selection must ignore this aggregate
        # cost once the workload declares comparable per-scenario objectives.
        "resource_summary": {
            "process": {
                "reliable": True,
                "cpu_seconds_per_million_rows": aggregate_cpu_cost,
            }
        },
    }


def test_cpu_efficiency_uses_only_the_same_resource_objective_scenarios() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resource_costs=(100.0, 100.0, 100.0),
        hot_costs=(100.0, 100.0, 100.0),
        aggregate_cpu_cost=100.0,
    )
    fastest_but_expensive = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resource_costs=(120.0, 120.0, 120.0),
        hot_costs=(1.0, 1.0, 1.0),
        aggregate_cpu_cost=1.0,
    )
    efficient = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resource_costs=(80.0, 80.0, 80.0),
        hot_costs=(1_000.0, 1_000.0, 1_000.0),
        aggregate_cpu_cost=1_000.0,
    )

    selected = calibration.select_winner(
        [baseline, fastest_but_expensive, efficient], ALL_SCENARIOS
    )

    assert selected["winner"] == "thread_pool_w10"
    assert selected["objectives_used"] == [
        "performance",
        "cpu_efficiency",
    ]
    assert efficient["cpu_efficiency_score"] == pytest.approx(1.25)
    assert fastest_but_expensive["cpu_efficiency_score"] == pytest.approx(
        100.0 / 120.0
    )
    assert efficient["guardrails"]["cpu_efficiency_ratios"] == pytest.approx(  # type: ignore[index]
        [1.25, 1.25]
    )


def test_unreliable_per_scenario_cpu_cv_disables_cpu_objective() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resource_costs=(100.0, 100.0, 100.0),
        hot_costs=(100.0, 100.0, 100.0),
        aggregate_cpu_cost=100.0,
    )
    unstable_fastest = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resource_costs=(50.0, 150.0, 50.0),
        hot_costs=(50.0, 50.0, 50.0),
        aggregate_cpu_cost=50.0,
    )
    stable_efficient = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resource_costs=(80.0, 80.0, 80.0),
        hot_costs=(80.0, 80.0, 80.0),
        aggregate_cpu_cost=80.0,
    )

    selected = calibration.select_winner(
        [baseline, unstable_fastest, stable_efficient], ALL_SCENARIOS
    )

    first_resource_summary = unstable_fastest["scenarios"][0]["summary"][  # type: ignore[index]
        "resources"
    ]
    assert isinstance(first_resource_summary, Mapping)
    assert first_resource_summary["process"]["reliable"] is False  # type: ignore[index]
    assert first_resource_summary["process"]["cpu_cost_cv"] > 0.15  # type: ignore[index,operator]
    assert unstable_fastest["cpu_efficiency_score"] is None
    assert selected["objectives_used"] == ["performance"]
    assert selected["winner"] == "thread_pool_w16"


def test_cpu_objective_uses_common_reliable_scenario_subset() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resource_costs=(100.0, 100.0, 100.0),
        hot_costs=(100.0, 100.0, 100.0),
        aggregate_cpu_cost=100.0,
    )
    fastest = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resource_costs=(120.0, 120.0, 120.0),
        hot_costs=(120.0, 120.0, 120.0),
        aggregate_cpu_cost=120.0,
    )
    efficient = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resource_costs=(80.0, 80.0, 80.0),
        hot_costs=(80.0, 80.0, 80.0),
        aggregate_cpu_cost=80.0,
    )
    first_resources = fastest["scenarios"][0]["summary"]["resources"]  # type: ignore[index]
    first_resources["process"]["reliable"] = False

    selected = calibration.select_winner(
        [baseline, fastest, efficient], ALL_SCENARIOS
    )

    assert selected["winner"] == "thread_pool_w10"
    assert selected["objectives_used"] == ["performance", "cpu_efficiency"]
    assert selected["cpu_resource_coverage"] == {
        "requested_scenarios": ["serial_1024", "serial_10000"],
        "used_scenarios": ["serial_10000"],
        "dropped_scenarios": ["serial_1024"],
        "mode": "common_reliable_subset",
    }
    assert efficient["cpu_efficiency_score"] == pytest.approx(1.25)


def test_selection_reports_objectives_as_inactive_with_one_eligible_candidate() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resource_costs=(100.0, 100.0, 100.0),
        hot_costs=(100.0, 100.0, 100.0),
        aggregate_cpu_cost=100.0,
    )
    rejected = _candidate(
        "thread_pool_w10",
        throughput=80.0,
        workers=10,
        resource_costs=(100.0, 100.0, 100.0),
        hot_costs=(100.0, 100.0, 100.0),
        aggregate_cpu_cost=100.0,
    )
    for candidate in (baseline, rejected):
        candidate["scenarios"][-1]["latency_guardrail"] = True  # type: ignore[index]
        candidate["resource_summary"]["cgroup"] = {  # type: ignore[index]
            "attribution_confident": True,
            "attributable": True,
            "throttled_period_fraction": 0.0,
            "throttled_seconds_per_wall_second": 0.0,
        }

    selected = calibration.select_winner(
        [baseline, rejected],
        ALL_SCENARIOS,
    )

    assert rejected["guardrails"]["eligible"] is False  # type: ignore[index]
    assert selected["eligible_candidate_count"] == 1
    assert selected["comparable_candidates"] == ["sequential"]
    objective_details = selected["objective_details"]
    assert isinstance(objective_details, Mapping)
    for objective in (
        "performance",
        "cpu_efficiency",
        "tail_latency",
        "cpu_throttling",
    ):
        assert objective_details[objective] == {  # type: ignore[index]
            "telemetry_available": True,
            "decision_active": False,
        }


def test_screening_fallback_reserves_a_low_resource_candidate() -> None:
    ranked = [
        {
            "name": f"thread_pool_w{workers}",
            "score": score,
            "options": {
                **READER_DEFAULTS,
                "io_workers": workers,
                "io_queue_capacity": workers * 2,
            },
        }
        for workers, score in (
            (32, 1.20),
            (24, 1.18),
            (16, 1.16),
            (12, 1.14),
            (10, 1.12),
            (8, 1.09),
            (4, 1.00),
        )
    ]
    provisional = {
        "cpu_resource_coverage": {
            "mode": "performance_only_fallback",
        }
    }

    probe = calibration._screening_resource_probe(ranked, provisional)

    assert probe is not None
    assert probe["name"] == "thread_pool_w8"


def test_screening_resource_band_ignores_an_ineligible_raw_outlier() -> None:
    def candidate(
        workers: int,
        score: float,
        *,
        eligible: bool,
    ) -> dict[str, object]:
        return {
            "name": f"thread_pool_w{workers}",
            "score": score,
            "guardrails": {"eligible": eligible},
            "options": {
                **READER_DEFAULTS,
                "io_workers": workers,
                "io_queue_capacity": workers * 2,
            },
        }

    ranked = [
        candidate(32, 1.50, eligible=False),
        candidate(24, 1.20, eligible=True),
        candidate(16, 1.18, eligible=True),
        candidate(8, 1.09, eligible=True),
        candidate(4, 1.00, eligible=True),
    ]
    provisional = {
        "cpu_resource_coverage": {
            "mode": "performance_only_fallback",
        }
    }

    probe = calibration._screening_resource_probe(ranked, provisional)

    assert probe is not None
    assert probe["name"] == "thread_pool_w8"


def test_heterogeneous_aggregate_resource_summary_is_not_cost_comparable() -> None:
    aggregate = calibration._summarize_resources(
        [
            _epoch(cpu_cost=50.0, rows=1_000),
            _epoch(cpu_cost=500.0, rows=10_000),
        ],
        comparable_work=False,
    )

    process = aggregate["process"]
    assert isinstance(process, Mapping)
    assert process["cost_stability_scope"] == (
        "heterogeneous_work_not_compared"
    )
    assert process["cpu_cost_cv"] is None


def test_short_multicore_resource_summary_is_reliable_from_cpu_time() -> None:
    epochs = [
        _epoch(cpu_cost=cpu_cost)
        for cpu_cost in (190.0, 200.0, 210.0)
    ]
    for epoch in epochs:
        epoch["elapsed_seconds"] = 0.1

    summary = calibration._summarize_resources(
        epochs,
        comparable_work=True,
    )

    process = summary["process"]
    assert isinstance(process, Mapping)
    assert summary["elapsed_seconds"] == pytest.approx(0.3)
    assert process["cpu_seconds"] == pytest.approx(0.6)
    assert process["cpu_cost_cv"] < 0.15  # type: ignore[operator]
    assert process["reliable"] is True


def test_resource_summary_invalidates_reset_counters() -> None:
    epoch = _epoch(cpu_cost=100.0)
    epoch["quality"] = {
        "counter_resets": ["process_usage.user_cpu_seconds"],
        "cgroup_configuration_changed": False,
        "cgroup_attribution_confident": True,
    }

    summary = calibration._summarize_resources([epoch])

    assert summary["process"]["available"] is False  # type: ignore[index]
    assert summary["process"]["reliable"] is False  # type: ignore[index]
    assert summary["quality"]["counter_resets"] == [  # type: ignore[index]
        "process_usage.user_cpu_seconds"
    ]


def test_resource_summary_preserves_psi_as_diagnostic_telemetry() -> None:
    epoch = _epoch(cpu_cost=100.0)
    epoch["cgroup"] = {
        "cpu_seconds": 0.2,
        "cpu_delta": {
            "nr_periods": 2,
            "nr_throttled": 0,
        },
        "throttled_seconds": 0.0,
        "pressure": {
            "io": {
                "some": {
                    "seconds": 0.25,
                    "end_averages": {"avg10": 3.5},
                }
            }
        },
    }
    epoch["quality"] = {
        "counter_resets": [],
        "cgroup_configuration_changed": False,
        "cgroup_attribution_confident": True,
    }

    summary = calibration._summarize_resources([epoch])

    pressure = summary["cgroup"]["pressure"]  # type: ignore[index]
    assert pressure["io"]["some"] == {
        "available_epochs": 1,
        "total_epochs": 1,
        "seconds": 0.25,
        "seconds_per_wall_second": 0.25,
        "max_end_avg10": 3.5,
    }


def test_any_contaminated_finalist_makes_calibration_inconclusive() -> None:
    baseline = {
        "name": "sequential",
        "guardrails": {"eligible": True},
        "resource_summary": {
            "cgroup": {"ambient_cpu_cores": 0.0},
        },
    }
    selected = {
        "name": "thread_pool_w8",
        "resource_summary": {
            "cgroup": {"ambient_cpu_cores": 0.0},
        },
    }
    non_selected = {
        "name": "thread_pool_w16",
        "resource_summary": {
            "cgroup": {"ambient_cpu_cores": 2.0},
        },
    }

    stability = calibration._resource_stability_report(
        [baseline, selected, non_selected],
        baseline=baseline,
        selected_name="thread_pool_w8",
        effective_cpu=10,
    )

    assert stability["passed"] is False
    assert stability["contaminated_candidates"] == ["thread_pool_w16"]
    assert "thread_pool_w16" in str(stability["reason"])


def test_low_process_share_makes_resource_stability_inconclusive() -> None:
    baseline = {
        "name": "sequential",
        "guardrails": {"eligible": True},
        "resource_summary": {
            "cgroup": {
                "ambient_cpu_cores": 0.0,
                "attribution_confident": True,
                "attributable": True,
                "process_cpu_share": 0.95,
            },
        },
    }
    low_share = {
        "name": "thread_pool_w16",
        "guardrails": {"eligible": True},
        "resource_summary": {
            "cgroup": {
                # This is below the current absolute one-core ambient limit,
                # but half of the cgroup CPU time is not attributable to the
                # measured process.
                "ambient_cpu_cores": 0.2,
                "attribution_confident": True,
                "attributable": False,
                "process_cpu_share": 0.5,
            },
        },
    }

    stability = calibration._resource_stability_report(
        [baseline, low_share],
        baseline=baseline,
        selected_name="sequential",
        effective_cpu=10,
    )

    assert stability["passed"] is False
    assert "thread_pool_w16" in str(stability["reason"])
    assert "attribut" in str(stability["reason"]).lower()


@pytest.mark.parametrize(
    "reason",
    [
        "cooperative deadline expired during ID sampling",
        "cooperative deadline expired during screening",
        "max_read_bytes cannot fit the minimum calibration workload",
    ],
)
def test_budget_and_deadline_reports_expose_resource_stability(
    reason: str,
) -> None:
    report = calibration._budget_result(reason)

    assert "resource_stability" in report
    assert report["resource_stability"] is None


def test_public_tune_deadline_report_forwards_resource_stability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "store"
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=4),
        shard_size_bytes=1 << 20,
        format_version=6,
    ) as collection:
        collection.add(
            np.array([1], dtype=np.uint64),
            np.ones((1, 4), dtype=np.float16),
        )

    clock_calls = 0

    def expired_after_start() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 2.0

    monkeypatch.setattr(
        collection_module.time,
        "perf_counter",
        expired_after_start,
    )

    report = throwdb.Collection.tune(path, max_seconds=1.0)

    assert report["status"] == "budget_exhausted"
    assert "resource_stability" in report
    assert report["resource_stability"] is None
