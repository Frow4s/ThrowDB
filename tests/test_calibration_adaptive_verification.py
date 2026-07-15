from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

import throwdb._calibration as calibration
from throwdb._tuning import READER_DEFAULTS


class _Trial:
    def __init__(self, record_size: int) -> None:
        self._record_size = record_size

    def get_into_profiled(
        self,
        ids: np.ndarray,
        output: np.ndarray,
    ) -> dict[str, object]:
        output.fill(0)
        return {
            "requested_rows": int(ids.size),
            "planned_read_bytes": int(ids.size) * self._record_size,
        }

    def close(self) -> None:
        pass


class _Native:
    dimension = 1
    dtype = "float16"
    record_size = 2

    def sample_ids(
        self,
        *,
        requested_count: int,
        seed: int,
        max_inspected_slots: int,
    ) -> dict[str, object]:
        del seed
        returned = min(requested_count, 64)
        return {
            "ids": np.arange(returned, dtype=np.uint64),
            "requested_count": requested_count,
            "inspected_slots": min(max_inspected_slots, returned),
            "complete": returned == requested_count,
        }

    def get_into_profiled(
        self,
        ids: np.ndarray,
        output: np.ndarray,
    ) -> dict[str, object]:
        output.fill(0)
        return {
            "requested_rows": int(ids.size),
            "planned_read_bytes": int(ids.size) * self.record_size,
        }

    def reader_trial(self, **options: object) -> _Trial:
        del options
        return _Trial(self.record_size)


def _options(workers: int) -> dict[str, object]:
    return {
        **READER_DEFAULTS,
        "io_workers": workers,
        "io_queue_capacity": workers * 2,
    }


def _candidate(name: str, workers: int) -> calibration._Candidate:
    return calibration._Candidate(name, _options(workers))


def _summary(values: Sequence[float]) -> dict[str, object]:
    mean = statistics.fmean(values)
    return {
        "median_throughput_rows_per_second": statistics.median(values),
        "median_p50_call_seconds": 0.008,
        "median_p95_call_seconds": 0.010,
        "median_p99_call_seconds": 0.012,
        "latency_sample_count": 64,
        "throughput_cv": (
            statistics.stdev(values) / mean if len(values) > 1 else 0.0
        ),
    }


def _stage_report(
    stage: calibration._StagePlan,
    values_by_scenario: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    scenarios = []
    for scenario in stage.scenarios:
        values = list(values_by_scenario[scenario.name])
        scenarios.append(
            {
                "name": scenario.name,
                "cache_mode": scenario.cache_mode,
                "latency_guardrail": scenario.latency_guardrail,
                "resource_objective": scenario.resource_objective,
                "repetitions": [
                    {"throughput_rows_per_second": value}
                    for value in values
                ],
                "summary": _summary(values),
            }
        )
    return {
        "name": stage.name,
        "status": "measured",
        "warmup": {
            "status": "complete",
            "rounds": 1,
            "blocks_total": len(stage.scenarios),
            "blocks_completed": len(stage.scenarios),
            "rows": stage.rows_per_round,
            "planned_read_bytes": stage.rows_per_round * _Native.record_size,
        },
        "scenarios": scenarios,
        "resource_summary": None,
    }


def _patch_small_matrix(
    monkeypatch: pytest.MonkeyPatch,
    candidates: Sequence[calibration._Candidate],
    *,
    verification_repeats: int = 3,
) -> None:
    monkeypatch.setattr(
        calibration,
        "_discover_candidates",
        lambda **kwargs: (
            list(candidates),
            [],
            {
                "reported": 16,
                "affinity": 16,
                "cgroup_quota": None,
                "effective": 14,
            },
        ),
    )
    screening = calibration._StagePlan(
        "screening",
        (calibration._Scenario("bulk", 1, 1, 1),),
        2,
    )
    verification = calibration._StagePlan(
        "verification",
        (calibration._Scenario("bulk", 1, 1, 1),),
        verification_repeats,
    )
    monkeypatch.setattr(
        calibration,
        "_build_stage_plans",
        lambda **kwargs: (screening, verification, 1, 10_000),
    )


def _run(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    return calibration.run_bounded_calibration(
        _Native(),
        record_size=_Native.record_size,
        current_options=None,
        seed=7,
        max_read_bytes=10_000,
        max_memory_bytes=3 << 20,
        deadline=10_000.0,
        clock=lambda: 0.0,
        capability_probe=lambda queue_depth, workers: {
            "io_uring_available": False,
        },
    )


def test_warmup_backend_failure_is_terminal_and_records_phase() -> None:
    class WarmupTrial:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.closed = False

        def get_into(self, ids: np.ndarray, output: np.ndarray) -> None:
            del ids
            if self.fail:
                raise calibration.ReaderBackendError("warm-up read failed")
            output.fill(0)

        def close(self) -> None:
            self.closed = True

    baseline = _candidate("sequential", 0)
    failed = _candidate("warmup_failure", 1)
    failed_trial = WarmupTrial(fail=True)
    trials: dict[str, object] = {
        baseline.name: WarmupTrial(),
        failed.name: failed_trial,
    }
    stage = calibration._StagePlan(
        "screening",
        (calibration._Scenario("warmup_probe", 1, 1, 1),),
        2,
    )

    active, reports, _, exhausted, _, _ = calibration._run_stage(
        stage,
        [baseline, failed],
        trials,
        sample=np.arange(8, dtype=np.uint64),
        replacement_begin=0,
        seed=7,
        dimension=1,
        dtype=np.dtype("float16"),
        record_size=2,
        deadline=10.0,
        clock=lambda: 0.0,
        progress=None,
        resource_probe=None,
    )

    failed_report = reports[failed.name]
    assert [candidate.name for candidate in active] == ["sequential"]
    assert exhausted is False
    assert failed_report["status"] == "failed"
    assert failed_report["warmup"]["status"] == "failed"
    assert failed_report["failure"]["phase"] == "warmup"
    assert failed_report["failure"]["scenario"] == "warmup_probe"
    assert failed_trial.closed is True
    assert failed.name not in trials


def test_last_measured_read_crossing_deadline_is_incomplete() -> None:
    now = [0.0]

    class CrossingTrial:
        def __init__(self) -> None:
            self.calls = 0

        def get_into(self, ids: np.ndarray, output: np.ndarray) -> None:
            del ids
            self.calls += 1
            output.fill(0)
            if self.calls == 2:
                now[0] = 2.0

    baseline = _candidate("sequential", 0)
    trial = CrossingTrial()
    stage = calibration._StagePlan(
        "verification",
        (calibration._Scenario("last_read", 1, 1, 1),),
        1,
    )

    _, reports, _, exhausted, _, _ = calibration._run_stage(
        stage,
        [baseline],
        {baseline.name: trial},
        sample=np.arange(4, dtype=np.uint64),
        replacement_begin=0,
        seed=7,
        dimension=1,
        dtype=np.dtype("float16"),
        record_size=2,
        deadline=1.0,
        clock=lambda: now[0],
        progress=None,
        resource_probe=None,
    )

    assert trial.calls == 2
    assert exhausted is True
    assert reports[baseline.name]["status"] == "incomplete"
    assert reports[baseline.name]["warmup"]["status"] == "complete"
    assert reports[baseline.name]["scenarios"][0]["repetitions"] == []


def test_verification_backfills_best_screening_eligible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate("sequential", 0),
        _candidate("fast_32", 32),
        _candidate("fast_24", 24),
        _candidate("fast_16", 16),
        _candidate("fast_12", 12),
        _candidate("steady_8", 8),
        _candidate("steady_6", 6),
    ]
    _patch_small_matrix(monkeypatch, candidates)
    stage_calls: list[tuple[str, list[str]]] = []

    screening_rates = {
        "sequential": 100.0,
        "fast_32": 150.0,
        "fast_24": 145.0,
        "fast_16": 140.0,
        "fast_12": 135.0,
        "steady_8": 125.0,
        "steady_6": 115.0,
    }

    def run_stage(
        stage: calibration._StagePlan,
        active: list[calibration._Candidate],
        trials: dict[str, object],
        **kwargs: object,
    ) -> tuple[
        list[calibration._Candidate],
        dict[str, dict[str, object]],
        int,
        bool,
        str,
        dict[str, object],
    ]:
        del trials, kwargs
        names = [item.name for item in active]
        stage_calls.append((stage.name, names))
        reports: dict[str, dict[str, object]] = {}
        for item in active:
            if stage.name == "screening":
                rate = screening_rates[item.name]
            elif item.name.startswith("fast_"):
                # All initially selected fast finalists fail a terminal
                # verification guardrail.
                rate = 80.0
            else:
                # The best still-unverified screening-eligible candidate is
                # healthy and materially faster when given its bounded slot.
                rate = screening_rates[item.name]
            reports[item.name] = _stage_report(
                stage,
                {scenario.name: [rate] * stage.repeats for scenario in stage.scenarios},
            )
        return (
            active,
            reports,
            len(active) * 10,
            False,
            "synthetic",
            {"status": "synthetic"},
        )

    monkeypatch.setattr(calibration, "_run_stage", run_stage)

    report = _run(monkeypatch)
    by_name = {
        item["name"]: item for item in report["candidates"]  # type: ignore[index]
    }

    assert "steady_8" not in stage_calls[1][1]
    assert report["status"] == "complete"
    assert report["selection"]["winner"] == "steady_8"  # type: ignore[index]
    assert by_name["steady_8"]["status"] == "measured"
    assert by_name["steady_8"]["verification"]["status"] == "measured"
    assert by_name["steady_6"]["status"] == "screened_out"
    assert [
        names
        for stage_name, names in stage_calls[2:]
        if stage_name.startswith("verification") and "steady_8" in names
    ] == [["sequential", "steady_8"]]
    pairing_reference = by_name["steady_8"]["pairing_reference"]
    assert pairing_reference["source"] == "verification_backfill_sequential"
    assert pairing_reference["candidate"] == "sequential"
    assert pairing_reference["resource_summary"] is None
    assert [
        scenario["name"] for scenario in pairing_reference["scenarios"]
    ] == ["bulk"]
    assert [
        repetition["throughput_rows_per_second"]
        for repetition in pairing_reference["scenarios"][0]["repetitions"]
    ] == [100.0, 100.0, 100.0]
    assert report["planned_read_bytes"] <= 10_000
    assert report["workload"]["adaptive_verification"] == {  # type: ignore[index]
        "max_backfill_candidates": 1,
        "backfill_pairing": "fresh_interleaved_sequential",
        "backfill_stage_candidates_per_slot": 2,
        "stability_recheck_repeats": 1,
        "stability_cv_lower_exclusive": 0.10,
        "stability_cv_upper_inclusive": 0.20,
        "minimum_paired_gain": 0.05,
        "included_in_requested_read_bound": True,
    }


def test_backfill_pairing_reference_stays_aligned_through_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate("sequential", 0),
        _candidate("doomed_32", 32),
        _candidate("doomed_24", 24),
        _candidate("doomed_16", 16),
        _candidate("borderline_12", 12),
        _candidate("steady_8", 8),
        _candidate("steady_6", 6),
    ]
    _patch_small_matrix(monkeypatch, candidates)
    stage_calls: list[tuple[str, list[str]]] = []
    screening_rates = {
        "sequential": 100.0,
        "doomed_32": 150.0,
        "doomed_24": 145.0,
        "doomed_16": 140.0,
        "borderline_12": 135.0,
        "steady_8": 125.0,
        "steady_6": 115.0,
    }

    def run_stage(
        stage: calibration._StagePlan,
        active: list[calibration._Candidate],
        trials: dict[str, object],
        **kwargs: object,
    ) -> tuple[
        list[calibration._Candidate],
        dict[str, dict[str, object]],
        int,
        bool,
        str,
        dict[str, object],
    ]:
        del trials, kwargs
        names = [item.name for item in active]
        stage_calls.append((stage.name, names))
        reports: dict[str, dict[str, object]] = {}

        if stage.name == "screening":
            values = {
                name: [screening_rates[name]] * stage.repeats
                for name in names
            }
            retained = active
        elif stage.name == "verification_recheck":
            values = {
                "sequential": [100.0],
                "borderline_12": [120.0],
                "steady_8": [125.0],
            }
            retained = active
        elif "steady_8" in names:
            # Backfill is paired with a fresh baseline observation whose
            # repetitions intentionally differ from the initial baseline.
            values = {
                "sequential": [90.0] * stage.repeats,
                "steady_8": [120.0] * stage.repeats,
            }
            retained = active
        else:
            values = {
                "sequential": [100.0] * stage.repeats,
                # This profile loses initial eligibility only to the bounded
                # 10-20% stability band, so it triggers one recheck.
                "borderline_12": [108.0, 120.0, 136.0],
                "doomed_32": [80.0] * stage.repeats,
                "doomed_24": [80.0] * stage.repeats,
                "doomed_16": [80.0] * stage.repeats,
            }
            retained = [
                item
                for item in active
                if item.name in {"sequential", "borderline_12"}
            ]

        for item in active:
            stage_report = _stage_report(
                stage,
                {
                    scenario.name: values[item.name]
                    for scenario in stage.scenarios
                },
            )
            if item.name.startswith("doomed_") and stage.name != "screening":
                stage_report["status"] = "failed"
                stage_report["failure"] = {
                    "phase": "measured",
                    "scenario": "bulk",
                }
            reports[item.name] = stage_report

        planned_bytes = (
            len(active)
            * (stage.repeats + calibration._WARMUP_ROUNDS)
            * _Native.record_size
        )
        return (
            retained,
            reports,
            planned_bytes,
            False,
            "synthetic",
            {"status": "synthetic"},
        )

    monkeypatch.setattr(calibration, "_run_stage", run_stage)

    report = _run(monkeypatch)
    by_name = {
        item["name"]: item for item in report["candidates"]  # type: ignore[index]
    }
    steady = by_name["steady_8"]
    verification_repetitions = steady["verification"]["scenarios"][0][
        "repetitions"
    ]
    reference_repetitions = steady["pairing_reference"]["scenarios"][0][
        "repetitions"
    ]

    assert stage_calls[-2:] == [
        ("verification", ["sequential", "steady_8"]),
        (
            "verification_recheck",
            ["sequential", "borderline_12", "steady_8"],
        ),
    ]
    assert [
        item["throughput_rows_per_second"]
        for item in verification_repetitions
    ] == [120.0, 120.0, 120.0, 125.0]
    assert [
        item["throughput_rows_per_second"]
        for item in reference_repetitions
    ] == [90.0, 90.0, 90.0, 100.0]
    assert len(verification_repetitions) == len(reference_repetitions) == 4
    assert steady["guardrails"]["eligible"] is True
    assert "bulk has no paired repetitions" not in steady["guardrails"][
        "reasons"
    ]
    assert report["selection"]["winner"] == "steady_8"  # type: ignore[index]
    assert report["candidate_discovery"]["stability_recheck"] == {  # type: ignore[index]
        "attempted": True,
        "trigger_candidates": ["borderline_12"],
        "scenarios": ["bulk"],
        "additional_repetitions": 1,
        "status": "complete",
    }
    assert report["planned_read_bytes"] <= 10_000


def test_borderline_paired_cv_gets_one_bounded_recheck_before_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_ratios = [1.08, 1.20, 1.36]
    initial_cv = statistics.stdev(initial_ratios) / statistics.fmean(
        initial_ratios
    )
    assert 0.10 < initial_cv < 0.12
    assert min(initial_ratios) >= 1.0 + calibration._MIN_GAIN

    candidates = [
        _candidate("sequential", 0),
        _candidate("fast_8", 8),
    ]
    _patch_small_matrix(monkeypatch, candidates)
    verification_calls = 0

    def run_stage(
        stage: calibration._StagePlan,
        active: list[calibration._Candidate],
        trials: dict[str, object],
        **kwargs: object,
    ) -> tuple[
        list[calibration._Candidate],
        dict[str, dict[str, object]],
        int,
        bool,
        str,
        dict[str, object],
    ]:
        nonlocal verification_calls
        del trials, kwargs
        reports: dict[str, dict[str, object]] = {}
        if stage.name == "screening":
            values = {
                "sequential": [100.0, 100.0],
                "fast_8": [120.0, 120.0],
            }
        else:
            verification_calls += 1
            if verification_calls == 1:
                values = {
                    "sequential": [100.0, 100.0, 100.0],
                    # Paired CV is ~11.58%, just over the 10% limit, while
                    # every observed gain is at least 8%.
                    "fast_8": [108.0, 120.0, 136.0],
                }
            else:
                # One extra paired observation pulls the combined CV below
                # the limit without changing the material positive signal.
                values = {
                    "sequential": [100.0] * stage.repeats,
                    "fast_8": [120.0] * stage.repeats,
                }
        for item in active:
            reports[item.name] = _stage_report(
                stage,
                {
                    scenario.name: values[item.name]
                    for scenario in stage.scenarios
                },
            )
        measured_rows = sum(
            len(values[item.name]) for item in active
        )
        return (
            active,
            reports,
            measured_rows * _Native.record_size,
            False,
            "synthetic",
            {"status": "synthetic"},
        )

    monkeypatch.setattr(calibration, "_run_stage", run_stage)

    report = _run(monkeypatch)
    candidate = next(
        item
        for item in report["candidates"]  # type: ignore[index]
        if item["name"] == "fast_8"
    )
    repetitions = candidate["verification"]["scenarios"][0]["repetitions"]

    assert verification_calls == 2
    assert len(repetitions) == 4
    assert report["selection"]["winner"] == "fast_8"  # type: ignore[index]
    assert candidate["guardrails"]["eligible"] is True
    assert candidate["guardrails"]["paired_throughput_cvs"][0] < 0.10
    assert report["planned_read_bytes"] <= 10_000
