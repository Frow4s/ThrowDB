from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np
import pytest

import throwdb._calibration as calibration
from throwdb._calibration import (
    CalibrationError,
    run_bounded_calibration,
    select_winner,
    validate_limits,
)
from throwdb._tuning import READER_DEFAULTS


_SCENARIOS = ("small", "large")


def _options(*, workers: int = 0) -> dict[str, object]:
    return {
        **READER_DEFAULTS,
        "io_workers": workers,
        "io_queue_capacity": workers * 2,
    }


def _candidate_report(
    name: str,
    *,
    throughput: float | Mapping[str, float],
    p95: float | Mapping[str, float] = 0.01,
    cv: float | Mapping[str, float] = 0.0,
    workers: int = 0,
) -> dict[str, object]:
    def value(raw: float | Mapping[str, float], scenario: str) -> float:
        return float(raw[scenario] if isinstance(raw, Mapping) else raw)

    return {
        "name": name,
        "options": _options(workers=workers),
        "status": "measured",
        "scenarios": [
            {
                "name": scenario,
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
                    "throughput_cv": value(cv, scenario),
                },
            }
            for scenario in _SCENARIOS
        ],
    }


class _StepClock:
    """A deterministic, thread-safe monotonic clock for the fake reads."""

    def __init__(self, *, step: float = 0.0001) -> None:
        self._value = 0.0
        self._step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step
            return value


def _write_embedding_bits(ids: np.ndarray, output: np.ndarray) -> None:
    columns = np.arange(output.shape[1], dtype=np.uint64)
    bits = (
        ids[:, np.newaxis] * np.uint64(7_919)
        + columns[np.newaxis, :] * np.uint64(101)
        + np.uint64(17)
    ).astype(np.uint16)
    output.view(np.uint16)[...] = bits


class _FakeTrial:
    def __init__(
        self,
        owner: _FakeNative,
        options: Mapping[str, object],
    ) -> None:
        self._owner = owner
        self._options = dict(options)
        self._closed = False

    def get_into_profiled(
        self,
        ids: np.ndarray,
        output: np.ndarray,
    ) -> dict[str, object]:
        assert not self._closed
        self._owner.profiled_trial_calls += 1
        workers = int(self._options["io_workers"])
        if (
            workers in self._owner.fail_profiled_workers
            and workers not in self._owner.failed_profiled_workers
        ):
            self._owner.failed_profiled_workers.add(workers)
            raise calibration.ReaderBackendError("fixture correctness failure")
        _write_embedding_bits(ids, output)
        if workers in self._owner.mismatch_workers and ids.size:
            output.view(np.uint16)[0, 0] ^= np.uint16(1)
        requested_bytes = int(ids.size) * self._owner.record_size
        return {
            "requested_rows": int(ids.size),
            "planned_read_bytes": (
                requested_bytes * self._owner.trial_read_amplification
            ),
            "planned_ranges": int(ids.size),
            "backend_tasks": max(1, int(ids.size)),
        }

    def get_into(self, ids: np.ndarray, output: np.ndarray) -> None:
        self._owner.timed_ids.append(ids.copy())
        self.get_into_profiled(ids, output)

    def close(self) -> None:
        self._closed = True
        self._owner.closed_trials += 1


class _FakeNative:
    dimension = 4
    dtype = "float16"
    record_size = 8

    def __init__(
        self,
        *,
        sampled_rows: int = 64,
        mismatch_workers: set[int] | None = None,
        fail_profiled_workers: set[int] | None = None,
        trial_read_amplification: int = 1,
    ) -> None:
        self.sampled_rows = sampled_rows
        self.mismatch_workers = mismatch_workers or set()
        self.fail_profiled_workers = fail_profiled_workers or set()
        self.failed_profiled_workers: set[int] = set()
        self.trial_read_amplification = trial_read_amplification
        self.sample_calls: list[dict[str, int]] = []
        self.trial_options: list[dict[str, object]] = []
        self.timed_ids: list[np.ndarray] = []
        self.profiled_trial_calls = 0
        self.closed_trials = 0

    def sample_ids(
        self,
        *,
        requested_count: int,
        seed: int,
        max_inspected_slots: int,
    ) -> dict[str, object]:
        self.sample_calls.append(
            {
                "requested_count": requested_count,
                "seed": seed,
                "max_inspected_slots": max_inspected_slots,
            }
        )
        returned = min(requested_count, self.sampled_rows)
        offset = np.uint64(seed % 10_000)
        ids = np.arange(1, returned + 1, dtype=np.uint64) + offset
        return {
            "ids": ids,
            "requested_count": requested_count,
            "inspected_slots": min(max_inspected_slots, returned * 2),
            "complete": returned == requested_count,
        }

    def get_into_profiled(
        self,
        ids: np.ndarray,
        output: np.ndarray,
    ) -> dict[str, object]:
        _write_embedding_bits(ids, output)
        return {
            "requested_rows": int(ids.size),
            "planned_read_bytes": int(ids.size) * self.record_size,
            "planned_ranges": int(ids.size),
            "backend_tasks": max(1, int(ids.size)),
        }

    def reader_trial(self, **options: object) -> _FakeTrial:
        self.trial_options.append(dict(options))
        return _FakeTrial(self, options)


def _no_io_uring(
    queue_depth: int,
    workers: int,
) -> dict[str, object]:
    return {
        "io_uring_available": False,
        "io_uring_reason": f"disabled for test: w={workers}, qd={queue_depth}",
    }


def _run_fake(
    native: _FakeNative,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int = 123,
    max_read_bytes: int = 2 << 20,
    deadline: float = 10_000.0,
    clock: object | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (
            4,
            {
                "reported": 8,
                "affinity": 4,
                "cgroup_quota": None,
                "effective": 4,
            },
        ),
    )
    return run_bounded_calibration(
        native,
        record_size=native.record_size,
        current_options=None,
        seed=seed,
        max_read_bytes=max_read_bytes,
        max_memory_bytes=1 << 20,
        deadline=deadline,
        clock=_StepClock() if clock is None else clock,  # type: ignore[arg-type]
        progress=progress,
        capability_probe=_no_io_uring,
    )


def test_validate_limits_accepts_real_numbers_and_numpy_integers() -> None:
    assert validate_limits(
        max_seconds=np.float64(1.5),
        max_read_bytes=np.uint64(4_096),
        max_memory_bytes=np.int64(2_048),
        seed=np.uint64(7),
    ) == (1.5, 4_096, 2_048, 7)


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_validate_limits_rejects_boolean_numbers(value: object) -> None:
    with pytest.raises(TypeError):
        validate_limits(
            max_seconds=value,
            max_read_bytes=1,
            max_memory_bytes=1,
            seed=0,
        )
    for field in ("max_read_bytes", "max_memory_bytes", "seed"):
        values: dict[str, object] = {
            "max_seconds": 1.0,
            "max_read_bytes": 1,
            "max_memory_bytes": 1,
            "seed": 0,
        }
        values[field] = value
        with pytest.raises(TypeError):
            validate_limits(**values)


@pytest.mark.parametrize("value", ["1.5", object(), None])
def test_validate_limits_rejects_non_numeric_seconds(value: object) -> None:
    with pytest.raises(TypeError):
        validate_limits(
            max_seconds=value,
            max_read_bytes=1,
            max_memory_bytes=1,
            seed=0,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_validate_limits_rejects_non_positive_or_non_finite_seconds(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        validate_limits(
            max_seconds=value,
            max_read_bytes=1,
            max_memory_bytes=1,
            seed=0,
        )


@pytest.mark.parametrize("value", [1.0, "1", object(), None])
@pytest.mark.parametrize("field", ["max_read_bytes", "max_memory_bytes", "seed"])
def test_validate_limits_requires_integral_budget_fields(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "max_seconds": 1.0,
        "max_read_bytes": 1,
        "max_memory_bytes": 1,
        "seed": 0,
    }
    values[field] = value
    with pytest.raises(TypeError):
        validate_limits(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_read_bytes", 0),
        ("max_read_bytes", 1 << 64),
        ("max_memory_bytes", 0),
        ("max_memory_bytes", 1 << 63),
        ("seed", -1),
        ("seed", 1 << 64),
    ],
)
def test_validate_limits_enforces_integer_bounds(field: str, value: int) -> None:
    values = {
        "max_seconds": 1.0,
        "max_read_bytes": 1,
        "max_memory_bytes": 1,
        "seed": 0,
    }
    values[field] = value
    with pytest.raises(ValueError):
        validate_limits(**values)


def test_select_winner_rejects_throughput_and_stability_regressions() -> None:
    baseline = _candidate_report("sequential", throughput=100.0)
    unsafe = _candidate_report(
        "thread_pool_w4",
        throughput={"small": 200.0, "large": 89.0},
        p95={"small": 0.005, "large": 0.013},
        cv={"small": 0.0, "large": 0.30},
        workers=4,
    )

    selected = select_winner([baseline, unsafe], _SCENARIOS)

    assert selected["winner"] == "sequential"
    guardrails = unsafe["guardrails"]
    assert guardrails["eligible"] is False  # type: ignore[index]
    assert guardrails["reasons"] == [  # type: ignore[index]
        "large throughput regressed",
        "large is unstable",
    ]
    assert guardrails["p95_ratios"] == pytest.approx([0.5, 1.3])  # type: ignore[index]


def test_measured_late_outlier_remains_ineligible_after_separate_warmup() -> None:
    baseline = _candidate_report("sequential", throughput=100_000.0)
    candidate = _candidate_report(
        "thread_pool_w4",
        throughput=250_000.0,
        workers=4,
    )
    candidate["warmup"] = {
        "status": "complete",
        "rows": 16_144,
        "measured": False,
    }
    for scenario in candidate["scenarios"]:  # type: ignore[assignment]
        scenario["repetitions"] = [
            {"throughput_rows_per_second": value}
            for value in (250_000.0, 250_000.0, 80_000.0)
        ]
        scenario["summary"]["throughput_cv"] = 0.416  # type: ignore[index]

    selected = select_winner([baseline, candidate], _SCENARIOS)

    assert selected["winner"] == "sequential"
    guardrails = candidate["guardrails"]
    assert guardrails["eligible"] is False  # type: ignore[index]
    assert all(  # type: ignore[index]
        f"{scenario} is unstable" in guardrails["reasons"]
        for scenario in _SCENARIOS
    )


def test_select_winner_requires_five_percent_composite_gain() -> None:
    baseline = _candidate_report("sequential", throughput=100.0)
    small_gain = _candidate_report(
        "thread_pool_w4",
        throughput=102.5,
        p95=0.009,
        workers=4,
    )

    selected = select_winner([baseline, small_gain], _SCENARIOS)

    assert selected["winner"] == "sequential"
    assert "minimum 5%" in str(selected["reason"])


def test_select_winner_uses_paired_repetition_ratios() -> None:
    baseline = _candidate_report("sequential", throughput=100.0)
    candidate = _candidate_report(
        "thread_pool_w4",
        throughput=104.0,
        workers=4,
    )
    for baseline_scenario, candidate_scenario in zip(
        baseline["scenarios"],  # type: ignore[index]
        candidate["scenarios"],  # type: ignore[index]
    ):
        baseline_scenario["repetitions"] = [
            {"throughput_rows_per_second": value}
            for value in (90.0, 100.0, 110.0)
        ]
        candidate_scenario["repetitions"] = [
            {"throughput_rows_per_second": value}
            for value in (99.0, 104.0, 121.0)
        ]

    selected = select_winner([baseline, candidate], _SCENARIOS)

    assert selected["winner"] == "thread_pool_w4"
    ratios = candidate["guardrails"]["throughput_ratios"]  # type: ignore[index]
    assert ratios == pytest.approx(
        [1.1, 1.1]
    )


def test_select_winner_uses_deterministic_lower_resource_tie_break() -> None:
    baseline = _candidate_report("sequential", throughput=100.0)
    workers8 = _candidate_report(
        "thread_pool_w8", throughput=110.0, p95=0.009, workers=8
    )
    workers4 = _candidate_report(
        "thread_pool_w4", throughput=109.0, p95=0.009, workers=4
    )

    forward = select_winner(
        deepcopy([baseline, workers8, workers4]), _SCENARIOS
    )
    reversed_candidates = select_winner(
        deepcopy([baseline, workers4, workers8]), _SCENARIOS
    )

    assert forward == reversed_candidates
    assert forward["winner"] == "thread_pool_w4"
    assert forward["options"]["io_workers"] == 4  # type: ignore[index]


def test_tie_band_does_not_replace_a_real_winner_with_subthreshold_gain() -> None:
    baseline = _candidate_report("sequential", throughput=100.0)
    winner = _candidate_report(
        "thread_pool_w8", throughput=106.0, p95=0.009, workers=8
    )
    subthreshold = _candidate_report(
        "thread_pool_w1", throughput=104.0, p95=0.009, workers=1
    )

    selected = select_winner([baseline, winner, subthreshold], _SCENARIOS)

    assert selected["winner"] == "thread_pool_w8"


def test_baseline_middle_repeat_variance_fails_stability_guardrail() -> None:
    baseline = {
        "scenarios": [
            {
                "name": name,
                "repetitions": [
                    {"throughput_rows_per_second": 100.0},
                    {"throughput_rows_per_second": 1_000.0},
                    {"throughput_rows_per_second": 100.0},
                ],
                "summary": {"throughput_cv": 0.50},
            }
            for name in _SCENARIOS
        ]
    }

    stability = calibration._baseline_drift(baseline, _SCENARIOS)

    assert stability["passed"] is False
    assert stability["composite_drift"] == 0.0
    assert stability["unstable_scenarios"] == list(_SCENARIOS)


def test_bounded_calibration_is_deterministic_bit_exact_and_report_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_native = _FakeNative()
    second_native = _FakeNative()

    first = _run_fake(first_native, monkeypatch, seed=987)
    second = _run_fake(second_native, monkeypatch, seed=987)

    reserved_workspace = max(
        2 * first_native.record_size + 8,
        4 * (first_native.record_size + 8),
        32,
    )
    requested_count = ((1 << 20) - reserved_workspace) // 16
    expected_sample_call = {
        "requested_count": requested_count,
        "seed": 987,
        "max_inspected_slots": requested_count * 8,
    }
    assert first_native.sample_calls == [expected_sample_call]
    assert second_native.sample_calls == first_native.sample_calls
    assert first["status"] == "complete"
    assert first["selection"]["winner"] == "sequential"  # type: ignore[index]
    assert first["selection"]["options"] == READER_DEFAULTS  # type: ignore[index]
    assert first["sample"] == second["sample"]
    assert first["workload"] == second["workload"]
    assert [item["name"] for item in first["candidates"]] == [  # type: ignore[index]
        "sequential",
        "thread_pool_w4",
        "thread_pool_w8",
    ]
    assert all(
        item["status"] == "measured" for item in first["candidates"]  # type: ignore[index]
    )
    assert all(
        item["correctness"]["passed"]  # type: ignore[index]
        for item in first["candidates"]  # type: ignore[index]
    )
    assert first["workload"]["warmup"] == {  # type: ignore[index]
        "rounds": 1,
        "measured": False,
        "accounting": "charged_not_measured",
        "blocks_per_candidate": 3,
        "rows_per_candidate": 16_144,
    }
    assert all(
        item["warmup"]
        == {
            "status": "complete",
            "rounds": 1,
            "blocks_total": 3,
            "blocks_completed": 3,
            "rows": 16_144,
            "planned_read_bytes": 129_152,
        }
        for item in first["candidates"]  # type: ignore[index]
    )
    assert all(
        len(scenario["repetitions"]) == 3
        for item in first["candidates"]  # type: ignore[index]
        for scenario in item["scenarios"]
    )
    assert len(first_native.timed_ids) == 84
    assert sum(ids.size for ids in first_native.timed_ids) == 193_728
    assert first["planned_read_bytes"] == 1_551_872
    assert first["planned_read_bytes"] <= 2 << 20
    assert (
        first["planned_read_bytes"]
        == first["workload"]["requested_read_bound_bytes"]  # type: ignore[index]
    )
    assert first_native.closed_trials == len(first_native.trial_options)
    assert len(first_native.trial_options) == len(first["candidates"])
    assert first["workload"]["cache_control"] == (  # type: ignore[index]
        "ambient_replacement_warm_interleaved"
    )
    assert "persisted" not in first
    assert "profile" not in first


def test_measured_candidate_order_rotates_for_three_active_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    report = _run_fake(
        _FakeNative(),
        monkeypatch,
        progress=messages.append,
    )

    assert report["status"] == "complete"
    first_candidate: dict[tuple[int, str], str] = {}
    for message in messages:
        if not message.startswith("repeat "):
            continue
        prefix, block = message.split(": ", 1)
        repetition = int(prefix.split()[1].split("/", 1)[0])
        candidate, scenario = block.split(" / ", 1)
        first_candidate.setdefault((repetition, scenario), candidate)

    for scenario in ("serial_1024", "parallel_1024", "serial_10000"):
        order = [first_candidate[(repetition, scenario)] for repetition in (1, 2, 3)]
        assert len(set(order)) == 3


def test_bounded_calibration_stops_before_sampling_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()

    report = _run_fake(
        native,
        monkeypatch,
        deadline=1.0,
        clock=lambda: 1.0,
    )

    assert report["status"] == "budget_exhausted"
    assert report["sample"] is None
    assert report["selection"] is None
    assert report["planned_read_bytes"] == 0
    assert native.sample_calls == []
    assert native.trial_options == []


def test_bounded_calibration_returns_partial_report_when_matrix_hits_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()
    now = [0.0]
    repeats_started = [0]

    def progress(message: str) -> None:
        if not message.startswith("repeat "):
            return
        repeats_started[0] += 1
        if repeats_started[0] == 2:
            now[0] = 2.0

    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (4, {"effective": 4}),
    )
    report = run_bounded_calibration(
        native,
        record_size=native.record_size,
        current_options=None,
        seed=123,
        max_read_bytes=2 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        clock=lambda: now[0],
        progress=progress,
        capability_probe=_no_io_uring,
    )

    assert report["status"] == "budget_exhausted"
    assert report["sample"] is not None
    assert report["selection"] is None
    assert report["planned_read_bytes"] > 0
    assert all(
        item["status"] == "incomplete" for item in report["candidates"]  # type: ignore[index]
    )
    measured_repetitions = sum(
        len(scenario["repetitions"])
        for item in report["candidates"]  # type: ignore[index]
        for scenario in item["scenarios"]
    )
    assert measured_repetitions == 1
    assert len(native.timed_ids) == 23
    assert sum(ids.size for ids in native.timed_ids) == 50_480
    assert report["planned_read_bytes"] == 405_888
    assert native.closed_trials == len(native.trial_options)


def test_progress_callback_cannot_start_a_measured_read_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()
    now = [0.0]

    def progress(message: str) -> None:
        if message.startswith("repeat "):
            now[0] = 2.0

    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (4, {"effective": 4}),
    )
    report = run_bounded_calibration(
        native,
        record_size=native.record_size,
        current_options=None,
        seed=123,
        max_read_bytes=2 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        clock=lambda: now[0],
        progress=progress,
        capability_probe=_no_io_uring,
    )

    assert report["status"] == "budget_exhausted"
    assert sum(
        len(scenario["repetitions"])
        for item in report["candidates"]  # type: ignore[index]
        for scenario in item["scenarios"]
    ) == 0
    assert len(native.timed_ids) == 21
    assert sum(ids.size for ids in native.timed_ids) == 48_432
    assert report["planned_read_bytes"] == 389_504


def test_deadline_during_warmup_prevents_any_measured_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()
    now = [0.0]

    def progress(message: str) -> None:
        if message.startswith("warm-up "):
            now[0] = 2.0

    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (4, {"effective": 4}),
    )
    report = run_bounded_calibration(
        native,
        record_size=native.record_size,
        current_options=None,
        seed=123,
        max_read_bytes=2 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        clock=lambda: now[0],
        progress=progress,
        capability_probe=_no_io_uring,
    )

    assert report["status"] == "budget_exhausted"
    assert report["reason"] == "cooperative deadline expired during warm-up"
    assert report["selection"] is None
    assert native.timed_ids == []
    assert all(
        not scenario["repetitions"]
        for item in report["candidates"]  # type: ignore[index]
        for scenario in item["scenarios"]
    )


def test_trial_construction_crossing_deadline_skips_correctness_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()
    now = [0.0]
    original_reader_trial = native.reader_trial

    def reader_trial(**options: object) -> _FakeTrial:
        trial = original_reader_trial(**options)
        now[0] = 2.0
        return trial

    native.reader_trial = reader_trial  # type: ignore[method-assign]
    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (4, {"effective": 4}),
    )

    report = run_bounded_calibration(
        native,
        record_size=native.record_size,
        current_options=None,
        seed=123,
        max_read_bytes=2 << 20,
        max_memory_bytes=1 << 20,
        deadline=1.0,
        clock=lambda: now[0],
        capability_probe=_no_io_uring,
    )

    assert report["status"] == "budget_exhausted"
    assert native.profiled_trial_calls == 0
    assert native.closed_trials == 1
    assert all(
        item["warmup"]["status"] == "not_run"
        for item in report["candidates"]  # type: ignore[index]
    )


def test_parallel_start_barrier_is_bounded_when_a_worker_fails_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative()
    trial = native.reader_trial(**READER_DEFAULTS)
    original_empty = calibration.np.empty
    failed = False

    def flaky_empty(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal failed
        if threading.current_thread() is not threading.main_thread() and not failed:
            failed = True
            raise MemoryError("fixture worker allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(calibration.np, "empty", flaky_empty)
    query = np.array([1], dtype=np.uint64)
    started = time.perf_counter()
    with pytest.raises(CalibrationError, match="barrier did not complete"):
        calibration._measure_block(
            trial,
            [query, query],
            calibration._Scenario("parallel", 1, 2, 2),
            dimension=native.dimension,
            dtype=np.dtype(native.dtype),
            record_size=native.record_size,
            deadline=0.05,
            clock=lambda: 0.0,
        )
    assert time.perf_counter() - started < 1.0
    trial.close()


def test_bounded_calibration_aborts_on_any_bit_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative(mismatch_workers={4})

    with pytest.raises(
        CalibrationError,
        match="thread_pool_w4 failed bit-exact correctness",
    ):
        _run_fake(native, monkeypatch)

    assert native.closed_trials == len(native.trial_options)


def test_bounded_calibration_aborts_on_planned_read_amplification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative(trial_read_amplification=2)

    with pytest.raises(CalibrationError, match="zero-gap read bound"):
        _run_fake(native, monkeypatch)

    assert native.closed_trials == len(native.trial_options)


def test_correctness_backend_failure_keeps_conservative_read_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative(fail_profiled_workers={4})

    report = _run_fake(native, monkeypatch)

    failed = next(
        item
        for item in report["candidates"]  # type: ignore[index]
        if item["name"] == "thread_pool_w4"
    )
    assert report["status"] == "complete"
    assert failed["status"] == "unavailable"
    assert failed["warmup"]["status"] == "not_run"
    omitted_warmup_and_measurement = 4 * 16_144 * native.record_size
    assert report["planned_read_bytes"] == (
        report["workload"]["requested_read_bound_bytes"]  # type: ignore[index]
        - omitted_warmup_and_measurement
    )


def test_default_matrix_includes_bounded_io_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (10, {"effective": 10}),
    )

    candidates, unavailable, _ = calibration._discover_candidates(
        record_size=2048,
        current_options=None,
        capability_probe=lambda queue_depth, workers: {
            "io_uring_available": True,
        },
    )

    assert len(candidates) == 8
    assert "thread_pool_w16" in [candidate.name for candidate in candidates]
    assert not [
        item
        for item in unavailable
        if "exceeds the matrix-v2 cap" in str(item["reason"])
    ]


def test_oversized_current_profile_is_not_instantiated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "_effective_cpu_count",
        lambda: (10, {"effective": 10}),
    )
    current = {
        **READER_DEFAULTS,
        "io_workers": 1024,
        "io_queue_capacity": 2048,
    }

    candidates, unavailable, _ = calibration._discover_candidates(
        record_size=2048,
        current_options=current,
        capability_probe=_no_io_uring,
    )

    assert not any(candidate.name.startswith("current") for candidate in candidates)
    assert any(
        item["name"] == "current_w1024"
        and "exceeds the matrix-v2 cap" in str(item["reason"])
        for item in unavailable
    )


def test_read_and_memory_budgets_reduce_correctness_rows_exactly() -> None:
    record_size = 64 << 10
    minimum_rows = 3 + 1 + 3 * (2 + calibration._WARMUP_ROUNDS) * (4 + 2)
    scenarios, repeats, correctness, requested = calibration._build_scenarios(
        candidate_count=3,
        sample_rows=100,
        record_size=record_size,
        effective_cpu=4,
        max_read_bytes=minimum_rows * record_size,
        max_memory_bytes=1 << 20,
    )

    assert repeats == 2
    assert correctness == 1
    assert requested == minimum_rows * record_size
    assert all(scenario.batch_rows == 1 for scenario in scenarios)
    assert 100 * 8 + correctness * (2 * record_size + 8) <= 1 << 20


def test_large_sample_uses_one_globally_disjoint_query_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative(sampled_rows=100_000)

    report = _run_fake(
        native,
        monkeypatch,
        max_read_bytes=200_000,
    )

    assert report["status"] == "complete"
    assert report["workload"]["cache_control"] == (  # type: ignore[index]
        "ambient_global_disjoint_interleaved"
    )
    measured_ids = np.concatenate(native.timed_ids)
    workload = report["workload"]
    expected_rows = (  # type: ignore[index]
        len(report["candidates"])
        * (workload["repeats"] + workload["warmup"]["rounds"])
        * sum(
            scenario["batch_rows"] * scenario["calls_per_repeat"]
            for scenario in workload["scenarios"]
        )
    )
    assert measured_ids.size == expected_rows
    assert np.unique(measured_ids).size == measured_ids.size


def test_replacement_mode_releases_previous_queries_before_allocating_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeNative(sampled_rows=16)
    original = calibration._queries_for_candidate
    previous: weakref.ReferenceType[np.ndarray] | None = None
    calls = 0

    def tracked(*args: object, **kwargs: object) -> list[np.ndarray]:
        nonlocal previous, calls
        if previous is not None:
            assert previous() is None
        result = original(*args, **kwargs)
        selected = result[0].base
        assert isinstance(selected, np.ndarray)
        previous = weakref.ref(selected)
        calls += 1
        return result

    monkeypatch.setattr(calibration, "_queries_for_candidate", tracked)

    report = _run_fake(native, monkeypatch)

    assert report["status"] == "complete"
    assert report["workload"]["cache_control"] == (  # type: ignore[index]
        "ambient_replacement_warm_interleaved"
    )
    assert calls == 36
