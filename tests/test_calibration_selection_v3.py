from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy

from throwdb._calibration import select_winner
from throwdb._tuning import READER_DEFAULTS


SCENARIOS = ("small", "large")


def _options(*, workers: int) -> dict[str, object]:
    return {
        **READER_DEFAULTS,
        "io_workers": workers,
        "io_queue_capacity": workers * 2,
    }


def _resource_summary(
    *,
    cpu_cost: float,
    reliable: bool = True,
    throttled_period_fraction: float | None = None,
    throttled_wall_ratio: float | None = None,
    severe_throttling: bool = False,
    attributable: bool = True,
    attribution_confident: bool = True,
) -> dict[str, object]:
    return {
        "process": {
            "reliable": reliable,
            "cpu_seconds_per_million_rows": cpu_cost,
        },
        "cgroup": {
            "attribution_confident": attribution_confident,
            "attributable": attributable,
            "throttled_period_fraction": throttled_period_fraction,
            "throttled_seconds_per_wall_second": throttled_wall_ratio,
            "severe_throttling": severe_throttling,
        },
    }


def _candidate(
    name: str,
    *,
    throughput: float | Mapping[str, float],
    workers: int,
    p95: float | Mapping[str, float] = 0.010,
    p99: float | Mapping[str, float] = 0.012,
    latency_samples: int = 64,
    latency_guardrail: bool = False,
    resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    def value(raw: float | Mapping[str, float], scenario: str) -> float:
        if isinstance(raw, Mapping):
            return float(raw[scenario])
        return float(raw)

    report: dict[str, object] = {
        "name": name,
        "options": _options(workers=workers),
        "status": "measured",
        "scenarios": [
            {
                "name": scenario,
                "latency_guardrail": latency_guardrail,
                "repetitions": [
                    {
                        "throughput_rows_per_second": value(
                            throughput, scenario
                        )
                    }
                    for _ in range(3)
                ],
                "summary": {
                    "median_throughput_rows_per_second": value(
                        throughput, scenario
                    ),
                    "median_p95_call_seconds": value(p95, scenario),
                    "median_p99_call_seconds": value(p99, scenario),
                    "latency_sample_count": latency_samples,
                    "throughput_cv": 0.0,
                },
            }
            for scenario in SCENARIOS
        ],
    }
    if resources is not None:
        report["resource_summary"] = dict(resources)
    return report


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            _assert_finite_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_json(nested)


def _with_mixed_workload(
    candidate: dict[str, object],
    *,
    throughput: float,
    small_p99: float,
    large_throughput: float,
) -> dict[str, object]:
    candidate["scenarios"].append(  # type: ignore[union-attr]
        {
            "name": "mixed_small_large",
            "latency_guardrail": False,
            "resource_objective": False,
            "repetitions": [
                {"throughput_rows_per_second": throughput}
                for _ in range(3)
            ],
            "summary": {
                "median_throughput_rows_per_second": throughput,
                "median_p95_call_seconds": small_p99,
                "median_p99_call_seconds": small_p99,
                "latency_sample_count": 64,
                "throughput_cv": 0.0,
                "mixed": {
                    "small": {
                        "median_p99_call_seconds": small_p99,
                        "completed_calls": 32,
                    },
                    "large": {
                        "median_throughput_rows_per_second": large_throughput,
                        "completed_calls": 5,
                    },
                },
            },
        }
    )
    return candidate


def test_v3_legacy_reports_fall_back_to_performance_selection() -> None:
    baseline = _candidate(
        "sequential", throughput=100.0, workers=0
    )
    workers8 = _candidate(
        "thread_pool_w8", throughput=110.0, workers=8
    )
    workers4 = _candidate(
        "thread_pool_w4", throughput=109.0, workers=4
    )

    selected = select_winner(
        [baseline, workers8, workers4], SCENARIOS
    )

    assert selected["winner"] == "thread_pool_w4"
    assert selected["winner"] in selected["pareto_frontier"]
    assert selected["objectives_used"] == ["performance"]
    assert selected["alternatives"]["max_throughput"]["winner"] == (  # type: ignore[index]
        "thread_pool_w8"
    )
    assert selected["alternatives"]["cpu_efficiency"]["winner"] == (  # type: ignore[index]
        "thread_pool_w4"
    )


def test_v3_balanced_policy_prefers_cpu_efficiency_within_performance_band() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(cpu_cost=100.0),
    )
    fastest = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resources=_resource_summary(cpu_cost=120.0),
    )
    efficient = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resources=_resource_summary(cpu_cost=80.0),
    )

    selected = select_winner([baseline, fastest, efficient], SCENARIOS)

    assert selected["winner"] == "thread_pool_w10"
    assert selected["objectives_used"] == [
        "performance",
        "cpu_efficiency",
    ]
    assert selected["alternatives"]["max_throughput"]["winner"] == (  # type: ignore[index]
        "thread_pool_w16"
    )
    assert selected["alternatives"]["cpu_efficiency"]["winner"] == (  # type: ignore[index]
        "thread_pool_w10"
    )


def test_v3_materially_faster_candidate_wins_despite_lower_cpu_efficiency() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(cpu_cost=100.0),
    )
    fastest = _candidate(
        "thread_pool_w16",
        throughput=130.0,
        workers=16,
        resources=_resource_summary(cpu_cost=140.0),
    )
    efficient = _candidate(
        "thread_pool_w10",
        throughput=120.0,
        workers=10,
        resources=_resource_summary(cpu_cost=75.0),
    )

    selected = select_winner([baseline, fastest, efficient], SCENARIOS)

    assert selected["winner"] == "thread_pool_w16"
    assert selected["alternatives"]["cpu_efficiency"]["winner"] == (  # type: ignore[index]
        "thread_pool_w10"
    )


def test_v3_balanced_policy_prefers_clean_candidate_over_moderate_throttling() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )
    throttled = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.20,
            throttled_wall_ratio=0.20,
        ),
    )
    clean = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )

    selected = select_winner([baseline, throttled, clean], SCENARIOS)

    assert selected["winner"] == "thread_pool_w10"
    assert "cpu_throttling" in selected["objectives_used"]


def test_v3_ignores_throttling_that_is_not_attributable() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )
    ambient = _candidate(
        "thread_pool_w16",
        throughput=120.0,
        workers=16,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.90,
            throttled_wall_ratio=0.90,
            attributable=False,
        ),
    )
    clean = _candidate(
        "thread_pool_w10",
        throughput=117.0,
        workers=10,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )

    selected = select_winner([baseline, ambient, clean], SCENARIOS)

    assert selected["winner"] == "thread_pool_w16"
    assert "cpu_throttling" not in selected["objectives_used"]


def test_v3_severe_attributable_throttling_is_a_guardrail_failure() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(cpu_cost=100.0),
    )
    throttled = _candidate(
        "thread_pool_w16",
        throughput=150.0,
        workers=16,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.50,
            throttled_wall_ratio=0.50,
            severe_throttling=True,
        ),
    )

    selected = select_winner([baseline, throttled], SCENARIOS)

    assert selected["winner"] == "sequential"
    assert throttled["guardrails"]["eligible"] is False  # type: ignore[index]
    assert "severe attributable CPU throttling" in (  # type: ignore[operator]
        throttled["guardrails"]["reasons"]  # type: ignore[index]
    )


def test_v3_p99_regression_rejects_latency_guardrail_scenario() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )
    candidate = _candidate(
        "thread_pool_w8",
        throughput=150.0,
        workers=8,
        p99={"small": 0.013, "large": 0.020},
        latency_guardrail=True,
    )

    selected = select_winner([baseline, candidate], SCENARIOS)

    assert selected["winner"] == "sequential"
    assert candidate["guardrails"]["eligible"] is False  # type: ignore[index]
    assert candidate["guardrails"]["reasons"] == [  # type: ignore[index]
        "small p99 latency regressed"
    ]


def test_v3_mixed_workload_rejects_small_request_starvation() -> None:
    baseline = _with_mixed_workload(
        _candidate("sequential", throughput=100.0, workers=0),
        throughput=100.0,
        small_p99=0.012,
        large_throughput=100.0,
    )
    starved = _with_mixed_workload(
        _candidate("thread_pool_w8", throughput=150.0, workers=8),
        throughput=150.0,
        small_p99=0.025,
        large_throughput=150.0,
    )

    selected = select_winner(
        [baseline, starved],
        (*SCENARIOS, "mixed_small_large"),
    )

    assert selected["winner"] == "sequential"
    assert "mixed workload starved small requests" in (  # type: ignore[operator]
        starved["guardrails"]["reasons"]  # type: ignore[index]
    )
    assert (  # type: ignore[index]
        starved["guardrails"]["mixed_fairness"]["enforced"] is True
    )


def test_v3_mixed_own_slowdown_does_not_reject_better_absolute_p99() -> None:
    baseline = _with_mixed_workload(
        _candidate(
            "sequential",
            throughput=100.0,
            workers=0,
            p99={"small": 0.0040, "large": 0.010},
        ),
        throughput=100.0,
        small_p99=0.0046,
        large_throughput=100.0,
    )
    faster = _with_mixed_workload(
        _candidate(
            "io_uring_w1_qd64",
            throughput=150.0,
            workers=1,
            p99={"small": 0.00084, "large": 0.010},
        ),
        throughput=150.0,
        small_p99=0.00431,
        large_throughput=150.0,
    )

    selected = select_winner(
        [baseline, faster],
        (*SCENARIOS, "mixed_small_large"),
    )

    fairness = faster["guardrails"]["mixed_fairness"]  # type: ignore[index]
    assert fairness["small_p99_ratio"] > 2.0  # type: ignore[index]
    assert 0.00431 < 0.0046
    assert faster["guardrails"]["eligible"] is True  # type: ignore[index]
    assert "mixed workload starved small requests" not in (  # type: ignore[operator]
        faster["guardrails"]["reasons"]  # type: ignore[index]
    )
    assert selected["winner"] == "io_uring_w1_qd64"


def test_v3_mixed_absolute_regression_alone_is_not_starvation() -> None:
    baseline = _with_mixed_workload(
        _candidate(
            "sequential",
            throughput=100.0,
            workers=0,
            p99={"small": 0.0040, "large": 0.010},
        ),
        throughput=100.0,
        small_p99=0.0046,
        large_throughput=100.0,
    )
    no_own_slowdown = _with_mixed_workload(
        _candidate(
            "thread_pool_w4",
            throughput=150.0,
            workers=4,
            p99={"small": 0.0040, "large": 0.010},
        ),
        throughput=150.0,
        small_p99=0.0060,
        large_throughput=150.0,
    )

    selected = select_winner(
        [baseline, no_own_slowdown],
        (*SCENARIOS, "mixed_small_large"),
    )

    fairness = no_own_slowdown["guardrails"][  # type: ignore[index]
        "mixed_fairness"
    ]
    assert fairness["small_p99_ratio"] < 2.0  # type: ignore[index]
    assert 0.0060 > 0.0046 * 1.25
    assert no_own_slowdown["guardrails"]["eligible"] is True  # type: ignore[index]
    assert selected["winner"] == "thread_pool_w4"


def test_v3_mixed_starvation_requires_own_and_absolute_regression() -> None:
    baseline = _with_mixed_workload(
        _candidate(
            "sequential",
            throughput=100.0,
            workers=0,
            p99={"small": 0.0040, "large": 0.010},
        ),
        throughput=100.0,
        small_p99=0.0046,
        large_throughput=100.0,
    )
    starved = _with_mixed_workload(
        _candidate(
            "thread_pool_w8",
            throughput=150.0,
            workers=8,
            p99={"small": 0.00084, "large": 0.010},
        ),
        throughput=150.0,
        small_p99=0.0060,
        large_throughput=150.0,
    )

    selected = select_winner(
        [baseline, starved],
        (*SCENARIOS, "mixed_small_large"),
    )

    fairness = starved["guardrails"]["mixed_fairness"]  # type: ignore[index]
    assert fairness["small_p99_ratio"] > 2.0  # type: ignore[index]
    assert 0.0060 > 0.0046 * 1.25
    assert starved["guardrails"]["eligible"] is False  # type: ignore[index]
    assert "mixed workload starved small requests" in (  # type: ignore[operator]
        starved["guardrails"]["reasons"]  # type: ignore[index]
    )
    assert selected["winner"] == "sequential"


def test_v3_mixed_workload_rejects_large_request_collapse() -> None:
    baseline = _with_mixed_workload(
        _candidate("sequential", throughput=100.0, workers=0),
        throughput=100.0,
        small_p99=0.012,
        large_throughput=100.0,
    )
    collapsed = _with_mixed_workload(
        _candidate("thread_pool_w8", throughput=150.0, workers=8),
        throughput=150.0,
        small_p99=0.012,
        large_throughput=100.0,
    )

    selected = select_winner(
        [baseline, collapsed],
        (*SCENARIOS, "mixed_small_large"),
    )

    assert selected["winner"] == "sequential"
    assert "mixed workload regressed large throughput" in (  # type: ignore[operator]
        collapsed["guardrails"]["reasons"]  # type: ignore[index]
    )
    assert (  # type: ignore[index]
        collapsed["guardrails"]["mixed_fairness"]["enforced"] is True
    )


def test_v3_mixed_workload_requires_enough_fairness_samples() -> None:
    baseline = _with_mixed_workload(
        _candidate("sequential", throughput=100.0, workers=0),
        throughput=100.0,
        small_p99=0.012,
        large_throughput=100.0,
    )
    under_sampled = _with_mixed_workload(
        _candidate("thread_pool_w8", throughput=150.0, workers=8),
        throughput=150.0,
        small_p99=0.012,
        large_throughput=150.0,
    )
    mixed = under_sampled["scenarios"][-1]  # type: ignore[index]
    mixed["summary"]["mixed"]["small"]["completed_calls"] = 4

    selected = select_winner(
        [baseline, under_sampled],
        (*SCENARIOS, "mixed_small_large"),
    )

    assert selected["winner"] == "sequential"
    assert under_sampled["guardrails"]["reasons"] == [  # type: ignore[index]
        "mixed workload has insufficient fairness samples"
    ]


def test_v3_pareto_frontier_excludes_dominated_candidate() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(cpu_cost=100.0),
    )
    efficient = _candidate(
        "thread_pool_w4",
        throughput=110.0,
        workers=4,
        resources=_resource_summary(cpu_cost=80.0),
    )
    fastest = _candidate(
        "thread_pool_w8",
        throughput=120.0,
        workers=8,
        resources=_resource_summary(cpu_cost=110.0),
    )
    dominated = _candidate(
        "thread_pool_w12",
        throughput=108.0,
        workers=12,
        resources=_resource_summary(cpu_cost=130.0),
    )

    selected = select_winner(
        [baseline, efficient, fastest, dominated], SCENARIOS
    )

    assert selected["pareto_frontier"] == [
        "sequential",
        "thread_pool_w4",
        "thread_pool_w8",
    ]
    assert dominated["dominated_by"] == [
        "thread_pool_w4",
        "thread_pool_w8",
    ]


def test_v3_reselection_clears_ineligible_candidate_pareto_fields() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )
    candidate = _candidate(
        "thread_pool_w8",
        throughput=120.0,
        workers=8,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )

    first = select_winner([baseline, candidate], SCENARIOS)

    assert candidate["guardrails"]["eligible"] is True  # type: ignore[index]
    assert "dominated_by" in candidate
    assert "topology_rank" in candidate
    assert candidate["name"] in first["pareto_frontier"]

    for scenario in candidate["scenarios"]:  # type: ignore[union-attr]
        scenario["summary"]["median_p99_call_seconds"] = 0.040

    second = select_winner([baseline, candidate], SCENARIOS)

    assert candidate["guardrails"]["eligible"] is False  # type: ignore[index]
    assert "dominated_by" not in candidate
    assert "topology_rank" not in candidate
    assert candidate["name"] not in second["pareto_frontier"]


def test_v3_reports_observed_and_eligible_max_throughput_separately() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )
    observed_fastest = _candidate(
        "thread_pool_w8",
        throughput=150.0,
        workers=8,
        p99={"small": 0.020, "large": 0.020},
        latency_guardrail=True,
    )
    eligible_fastest = _candidate(
        "thread_pool_w4",
        throughput=120.0,
        workers=4,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )

    selected = select_winner(
        [baseline, observed_fastest, eligible_fastest], SCENARIOS
    )

    alternatives = selected["alternatives"]
    assert isinstance(alternatives, Mapping)
    observed = alternatives["observed_max_throughput"]
    eligible = alternatives["eligible_max_throughput"]
    assert isinstance(observed, Mapping)
    assert isinstance(eligible, Mapping)
    assert observed["winner"] == "thread_pool_w8"
    assert observed["eligible"] is False
    assert observed["reasons"] == [
        "small p99 latency regressed"
    ]
    assert eligible["winner"] == "thread_pool_w4"
    assert eligible["eligible"] is True
    assert eligible["reasons"] == []
    assert selected["winner"] == "thread_pool_w4"


def test_v3_zero_eligible_reports_no_eligible_maximum() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        resources=_resource_summary(
            cpu_cost=100.0,
            severe_throttling=True,
        ),
    )
    regressed = _candidate(
        "thread_pool_w8",
        throughput=80.0,
        workers=8,
        resources=_resource_summary(cpu_cost=100.0),
    )

    selected = select_winner([baseline, regressed], SCENARIOS)

    assert selected["eligible_candidate_count"] == 0
    alternatives = selected["alternatives"]
    assert isinstance(alternatives, Mapping)
    eligible = alternatives["eligible_max_throughput"]
    compatibility = alternatives["max_throughput"]
    assert isinstance(eligible, Mapping)
    assert isinstance(compatibility, Mapping)
    assert eligible["available"] is False
    assert eligible["winner"] is None
    assert compatibility["winner"] == "sequential"
    assert compatibility["eligible"] is False


def test_v3_baseline_only_reason_reports_guardrail_rejection() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )
    faster_but_unsafe = _candidate(
        "thread_pool_w8",
        throughput=150.0,
        workers=8,
        p99={"small": 0.020, "large": 0.020},
        latency_guardrail=True,
    )

    selected = select_winner([baseline, faster_but_unsafe], SCENARIOS)

    assert selected["eligible_candidate_count"] == 1
    observed = selected["alternatives"][  # type: ignore[index]
        "observed_max_throughput"
    ]
    assert observed["winner"] == "thread_pool_w8"
    assert observed["score"] >= 1.0 + 0.05
    assert observed["eligible"] is False
    assert "guardrail" in selected["reason"]
    assert "gain" not in selected["reason"]


def test_v3_objectives_are_not_decision_active_with_one_eligible() -> None:
    baseline = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
        resources=_resource_summary(
            cpu_cost=100.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )
    rejected = _candidate(
        "thread_pool_w8",
        throughput=150.0,
        workers=8,
        p99={"small": 0.020, "large": 0.020},
        latency_guardrail=True,
        resources=_resource_summary(
            cpu_cost=80.0,
            throttled_period_fraction=0.0,
            throttled_wall_ratio=0.0,
        ),
    )

    selected = select_winner([baseline, rejected], SCENARIOS)

    details = selected["objective_details"]
    assert isinstance(details, Mapping)
    assert details
    performance = details["performance"]
    assert isinstance(performance, Mapping)
    assert performance["telemetry_available"] is True
    for objective in details.values():
        assert isinstance(objective, Mapping)
        assert objective["decision_active"] is False
    assert rejected["guardrails"]["eligible"] is False  # type: ignore[index]
    assert selected["winner"] == "sequential"


def test_v3_topology_objective_reports_decision_activity() -> None:
    baseline_only = _candidate(
        "sequential",
        throughput=100.0,
        workers=0,
        p99={"small": 0.010, "large": 0.020},
        latency_guardrail=True,
    )
    rejected = _candidate(
        "thread_pool_w8",
        throughput=150.0,
        workers=8,
        p99={"small": 0.020, "large": 0.020},
        latency_guardrail=True,
    )
    one_eligible = select_winner([baseline_only, rejected], SCENARIOS)

    baseline = _candidate(
        "sequential", throughput=100.0, workers=0
    )
    candidate = _candidate(
        "thread_pool_w8", throughput=120.0, workers=8
    )
    two_eligible = select_winner([baseline, candidate], SCENARIOS)

    assert one_eligible["objective_details"]["topology_cost"] == {  # type: ignore[index]
        "telemetry_available": True,
        "decision_active": False,
    }
    assert two_eligible["objective_details"]["topology_cost"] == {  # type: ignore[index]
        "telemetry_available": True,
        "decision_active": True,
    }


def test_v3_selection_is_order_independent_and_strict_json_finite() -> None:
    candidates = [
        _candidate(
            "sequential",
            throughput=100.0,
            workers=0,
            resources=_resource_summary(
                cpu_cost=100.0,
                throttled_period_fraction=0.0,
                throttled_wall_ratio=0.0,
            ),
        ),
        _candidate(
            "thread_pool_w8",
            throughput=120.0,
            workers=8,
            resources=_resource_summary(
                cpu_cost=110.0,
                throttled_period_fraction=0.0,
                throttled_wall_ratio=0.0,
            ),
        ),
        _candidate(
            "thread_pool_w4",
            throughput=118.0,
            workers=4,
            resources=_resource_summary(
                cpu_cost=85.0,
                throttled_period_fraction=0.0,
                throttled_wall_ratio=0.0,
            ),
        ),
    ]

    forward = select_winner(deepcopy(candidates), SCENARIOS)
    reverse = select_winner(deepcopy(list(reversed(candidates))), SCENARIOS)

    assert forward == reverse
    _assert_finite_json(forward)
    json.dumps(forward, allow_nan=False, sort_keys=True)
