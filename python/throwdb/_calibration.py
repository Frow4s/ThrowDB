"""Bounded, deterministic reader calibration for one native collection.

The module deliberately contains no registry writes.  It measures ephemeral
reader trials against the already-open native snapshot and returns a bounded,
JSON-serializable result.  The high-level collection API owns target identity
validation and the single atomic publication step.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import os
import posixpath
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np

from ._throwdb import ReaderBackendError
from ._tuning import READER_DEFAULTS, normalize_reader_options


_REPORT_FORMAT: Final = "throwdb-reader-calibration"
_REPORT_VERSION: Final = 1
_MAX_CANDIDATES: Final = 8
_MAX_SAMPLE_ROWS: Final = 1_048_576
_MAX_INSPECTED_SLOTS: Final = 8_388_608
_CORRECTNESS_ROWS: Final = 256
_DEFAULT_REPEATS: Final = 3
_WARMUP_ROUNDS: Final = 1
_MIN_GAIN: Final = 0.05
_TIE_BAND: Final = 0.02
_MIN_SCENARIO_THROUGHPUT_RATIO: Final = 0.90
_MAX_THROUGHPUT_CV: Final = 0.10
_MAX_BASELINE_DRIFT: Final = 0.20
_MAX_THREAD_POOL_WORKERS: Final = 16
_MAX_THREAD_POOL_QUEUE: Final = 64
_MAX_IO_URING_WORKERS: Final = 2
_MAX_IO_URING_QUEUE_DEPTH: Final = 64
_MAX_RANGES_PER_TASK: Final = 64
_MAX_READ_RANGE_BYTES: Final = 16 << 20


class CalibrationError(RuntimeError):
    """The bounded calibration could not produce a trustworthy result."""


class _DeadlineExpired(RuntimeError):
    """An allocation or harness start crossed the cooperative deadline."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    options: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    batch_rows: int
    calls: int
    clients: int

    @property
    def total_rows(self) -> int:
        return self.batch_rows * self.calls


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        integer = int(value.__index__())  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if not minimum <= integer <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return integer


def _positive_seconds(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_seconds must be a number, not bool")
    if not isinstance(value, (numbers.Real, np.integer, np.floating)):
        raise TypeError("max_seconds must be a number")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("max_seconds must be a number") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("max_seconds must be finite and greater than zero")
    return seconds


def validate_limits(
    *,
    max_seconds: object,
    max_read_bytes: object,
    max_memory_bytes: object,
    seed: object,
) -> tuple[float, int, int, int]:
    """Validate public calibration limits without silently widening them."""

    return (
        _positive_seconds(max_seconds),
        _bounded_int(max_read_bytes, "max_read_bytes", 1, (1 << 64) - 1),
        _bounded_int(max_memory_bytes, "max_memory_bytes", 1, (1 << 63) - 1),
        _bounded_int(seed, "seed", 0, (1 << 64) - 1),
    )


def _cgroup_cpu_quota() -> float | None:
    """Return the tightest bounded cgroup-v2 quota on the common mount."""

    relative = "/"
    try:
        with open("/proc/self/cgroup", encoding="ascii") as stream:
            for raw_line in stream.read(64 << 10).splitlines():
                hierarchy, controllers, path = raw_line.split(":", 2)
                if hierarchy == "0" and controllers == "":
                    relative = posixpath.normpath("/" + path.lstrip("/"))
                    break
    except (OSError, ValueError):
        relative = "/"

    root = "/sys/fs/cgroup"
    current = posixpath.normpath(root + relative)
    if not (current == root or current.startswith(root + "/")):
        current = root
    limits: list[float] = []
    for _ in range(64):
        try:
            with open(
                posixpath.join(current, "cpu.max"), encoding="ascii"
            ) as stream:
                quota_text, period_text = stream.read(128).strip().split()
            if quota_text != "max":
                quota = int(quota_text) / int(period_text)
                if math.isfinite(quota) and quota > 0:
                    limits.append(quota)
        except (OSError, ValueError, ZeroDivisionError):
            pass
        if current == root:
            break
        parent = posixpath.dirname(current)
        current = parent if parent.startswith(root) else root
    return min(limits) if limits else None


def _effective_cpu_count() -> tuple[int, dict[str, object]]:
    reported = os.cpu_count() or 1
    affinity: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except OSError:
            affinity = None

    quota = _cgroup_cpu_quota()

    limits = [reported]
    if affinity is not None and affinity > 0:
        limits.append(affinity)
    if quota is not None:
        # One worker remains useful for a fractional CPU allocation.
        limits.append(max(1, math.floor(quota)))
    effective = max(1, min(limits))
    return effective, {
        "reported": reported,
        "affinity": affinity,
        "cgroup_quota": quota,
        "effective": effective,
    }


def _canonical_key(options: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(options[name] for name in sorted(READER_DEFAULTS))


def _candidate_name(prefix: str, options: Mapping[str, object]) -> str:
    if options["io_backend"] == "io_uring":
        return (
            f"{prefix}_w{options['io_uring_workers']}"
            f"_qd{options['io_uring_queue_depth']}"
        )
    workers = int(options["io_workers"])
    return f"{prefix}_w{workers}"


def _discover_candidates(
    *,
    record_size: int,
    current_options: Mapping[str, object] | None,
    capability_probe: Callable[[int, int], Mapping[str, object]],
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[_Candidate], list[dict[str, object]], dict[str, object]]:
    effective_cpu, cpu = _effective_cpu_count()
    candidates: list[_Candidate] = []
    unavailable: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    baseline_options = {
        **READER_DEFAULTS,
        "max_read_range_bytes": max(
            int(READER_DEFAULTS["max_read_range_bytes"]),
            record_size,
        ),
    }

    def add(name: str, raw: Mapping[str, object]) -> None:
        options = normalize_reader_options(raw, record_size=record_size)
        # Matrix v2 is uniform-random. Keeping gap=0 makes the physical
        # byte bound exact and avoids selecting read amplification by accident.
        if int(options["max_gap_bytes"]) != 0:
            unavailable.append(
                {
                    "name": name,
                    "reason": "max_gap_bytes is outside bounded matrix v2",
                }
            )
            return
        resource_reason: str | None = None
        if int(options["max_read_range_bytes"]) > _MAX_READ_RANGE_BYTES:
            resource_reason = "max_read_range_bytes exceeds the matrix-v2 cap"
        elif int(options["io_ranges_per_task"]) > _MAX_RANGES_PER_TASK:
            resource_reason = "io_ranges_per_task exceeds the matrix-v2 cap"
        elif options["io_backend"] == "thread_pool" and (
            int(options["io_workers"]) > _MAX_THREAD_POOL_WORKERS
            or int(options["io_queue_capacity"]) > _MAX_THREAD_POOL_QUEUE
        ):
            resource_reason = "thread-pool topology exceeds the matrix-v2 cap"
        elif options["io_backend"] == "io_uring" and (
            int(options["io_uring_workers"]) > _MAX_IO_URING_WORKERS
            or int(options["io_uring_queue_depth"]) > _MAX_IO_URING_QUEUE_DEPTH
        ):
            resource_reason = "io_uring topology exceeds the matrix-v2 cap"
        if resource_reason is not None:
            unavailable.append({"name": name, "reason": resource_reason})
            return
        key = _canonical_key(options)
        if key in seen:
            return
        if len(candidates) >= _MAX_CANDIDATES:
            unavailable.append(
                {"name": name, "reason": "bounded matrix v2 is already full"}
            )
            return
        seen.add(key)
        candidates.append(_Candidate(name, options))

    add("sequential", baseline_options)

    thread_targets = {
        min(effective_cpu, 8),
        min(effective_cpu, 16),
        min(16, max(2, 2 * effective_cpu)),
    }
    for workers in sorted(thread_targets):
        add(
            f"thread_pool_w{workers}",
            {
                **baseline_options,
                "io_workers": workers,
                "io_queue_capacity": workers * 2,
            },
        )

    if current_options is not None:
        try:
            current = normalize_reader_options(
                current_options, persisted=True, record_size=record_size
            )
            add(_candidate_name("current", current), current)
        except (TypeError, ValueError) as error:
            unavailable.append(
                {"name": "current", "reason": f"invalid current profile: {error}"}
            )

    for workers, queue_depth in ((1, 16), (1, 32), (1, 64), (2, 32)):
        name = f"io_uring_w{workers}_qd{queue_depth}"
        if deadline is not None and clock() >= deadline:
            unavailable.append(
                {"name": name, "reason": "deadline expired before capability probe"}
            )
            continue
        if workers > effective_cpu:
            unavailable.append(
                {"name": name, "reason": "requires more effective CPUs"}
            )
            continue
        try:
            capability = dict(capability_probe(queue_depth, workers))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            capability = {
                "io_uring_available": False,
                "io_uring_reason": str(error),
            }
        if not bool(capability.get("io_uring_available", False)):
            unavailable.append(
                {
                    "name": name,
                    "reason": str(
                        capability.get("io_uring_reason")
                        or "io_uring topology is unavailable"
                    )[:512],
                }
            )
            continue
        add(
            name,
            {
                **baseline_options,
                "io_backend": "io_uring",
                "io_uring_workers": workers,
                "io_uring_queue_depth": queue_depth,
            },
        )

    return candidates, unavailable, cpu


def _estimated_rows(
    candidates: int,
    correctness_rows: int,
    scenarios: Sequence[_Scenario],
    repeats: int,
) -> int:
    # One baseline truth read plus one bit-exact read per candidate.
    correctness = correctness_rows * (candidates + 1)
    rows_per_round = sum(item.total_rows for item in scenarios)
    warmup = candidates * _WARMUP_ROUNDS * rows_per_round
    measured = candidates * repeats * rows_per_round
    return correctness + warmup + measured


def _build_scenarios(
    *,
    candidate_count: int,
    sample_rows: int,
    record_size: int,
    effective_cpu: int,
    max_read_bytes: int,
    max_memory_bytes: int,
) -> tuple[list[_Scenario], int, int, int]:
    clients = min(4, effective_cpu)
    # The two phases do not overlap: correctness holds the sample plus truth
    # and actual outputs; measurement holds the sample, a global permutation,
    # one selected-ID array and the concurrently active outputs.
    correctness_available = max_memory_bytes - sample_rows * 8
    measurement_available = max_memory_bytes - sample_rows * 16
    correctness_memory_rows = correctness_available // (2 * record_size + 8)
    minimum_measurement = clients * (record_size + 8)
    if correctness_memory_rows < 1 or measurement_available < minimum_measurement:
        raise ValueError(
            "max_memory_bytes is too small for the bounded sample, "
            "correctness buffers, and one parallel read "
            f"({max_memory_bytes} bytes supplied)"
        )

    # serial_1024 initially keeps two query-ID slices plus one output alive;
    # parallel_1024 keeps one query and output per client alive.
    medium_limit = min(
        measurement_available // max(32, record_size + 16),
        measurement_available
        // max(16 * clients, clients * (record_size + 8)),
    )
    large_limit = measurement_available // max(16, record_size + 8)
    medium = max(1, min(1024, medium_limit))
    large = max(1, min(10_000, large_limit))
    repeats = _DEFAULT_REPEATS
    medium_calls = 2

    def make() -> list[_Scenario]:
        return [
            _Scenario("serial_1024", medium, medium_calls, 1),
            _Scenario("parallel_1024", medium, clients, clients),
            _Scenario("serial_10000", large, 1, 1),
        ]

    budget_rows = max_read_bytes // record_size
    minimum_workload_rows = (
        candidate_count * (2 + _WARMUP_ROUNDS) * (clients + 2)
    )
    correctness_read_rows = (
        budget_rows - minimum_workload_rows
    ) // (candidate_count + 1)
    correctness = min(
        _CORRECTNESS_ROWS,
        sample_rows,
        correctness_memory_rows,
        correctness_read_rows,
    )
    if correctness < 1:
        minimum_rows = candidate_count + 1 + minimum_workload_rows
        raise ValueError(
            "max_read_bytes is too small for the minimum calibration matrix; "
            f"at least {minimum_rows * record_size} bytes are required"
        )
    scenarios = make()
    while (
        _estimated_rows(candidate_count, correctness, scenarios, repeats)
        * record_size
        > max_read_bytes
        and repeats > 2
    ):
        repeats -= 1
    if (
        _estimated_rows(candidate_count, correctness, scenarios, repeats)
        * record_size
        > max_read_bytes
        and medium_calls > 1
    ):
        medium_calls = 1
        scenarios = make()

    # If the dtype/dimension is unusually wide, scale row counts while keeping
    # all three workloads represented.  This loop converges geometrically and
    # never drops a scenario below one row/call/repeat.
    for _ in range(16):
        requested = (
            _estimated_rows(candidate_count, correctness, scenarios, repeats)
            * record_size
        )
        if requested <= max_read_bytes:
            return scenarios, repeats, correctness, requested
        ratio = max_read_bytes / requested
        new_medium = max(1, int(medium * ratio * 0.95))
        new_large = max(1, int(large * ratio * 0.95))
        if (new_medium, new_large) == (medium, large):
            break
        medium, large = new_medium, new_large
        scenarios = make()

    minimum = (
        _estimated_rows(
            candidate_count,
            1,
            [
                _Scenario("serial_1024", 1, 1, 1),
                _Scenario("parallel_1024", 1, clients, clients),
                _Scenario("serial_10000", 1, 1, 1),
            ],
            2,
        )
        * record_size
    )
    raise ValueError(
        "max_read_bytes is too small for the minimum calibration matrix; "
        f"at least {minimum} bytes are required"
    )


def _derived_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(b"throwdb-calibration-v1\0")
    digest.update(seed.to_bytes(8, "little"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little")


def _queries_for_candidate(
    sample: np.ndarray,
    *,
    disjoint_indices: np.ndarray | None,
    replacement_begin: int,
    scenario: _Scenario,
    seed: int,
) -> list[np.ndarray]:
    needed = scenario.total_rows
    if disjoint_indices is not None:
        if disjoint_indices.size != needed:
            raise CalibrationError("invalid global disjoint query slice")
        indices = disjoint_indices
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            replacement_begin,
            sample.size,
            size=needed,
            dtype=np.int64,
        )
    selected = np.require(sample[indices], dtype=np.uint64, requirements=["C", "A"])
    return [
        selected[begin : begin + scenario.batch_rows]
        for begin in range(0, selected.size, scenario.batch_rows)
    ]


def _close_trial(trial: object) -> None:
    close = getattr(trial, "close", None)
    if close is not None:
        close()


def _measure_block(
    trial: object,
    queries: Sequence[np.ndarray],
    scenario: _Scenario,
    *,
    dimension: int,
    dtype: np.dtype,
    record_size: int,
    deadline: float,
    clock: Callable[[], float],
) -> dict[str, object]:
    latencies: list[float] = []
    if scenario.clients == 1:
        output = np.empty((scenario.batch_rows, dimension), dtype=dtype)
        block_started = clock()
        if block_started >= deadline:
            raise _DeadlineExpired
        for query in queries:
            if clock() >= deadline:
                raise _DeadlineExpired
            started = clock()
            trial.get_into(query, output)
            latencies.append(clock() - started)
        block_finished = clock()
    else:
        start_box: dict[str, float] = {}
        expired_box = {"value": False}

        def begin_parallel_block() -> None:
            started = clock()
            start_box["value"] = started
            expired_box["value"] = started >= deadline

        barrier = threading.Barrier(
            scenario.clients + 1,
            action=begin_parallel_block,
        )
        barrier_timeout = min(10.0, max(0.001, deadline - clock()))

        def parallel_call(query: np.ndarray) -> float:
            output = np.empty((query.size, dimension), dtype=dtype)
            barrier.wait(timeout=barrier_timeout)
            if expired_box["value"]:
                raise _DeadlineExpired
            started = clock()
            if started >= deadline:
                raise _DeadlineExpired
            trial.get_into(query, output)
            return clock() - started

        try:
            with ThreadPoolExecutor(max_workers=scenario.clients) as executor:
                try:
                    futures = [
                        executor.submit(parallel_call, query)
                        for query in queries
                    ]
                    barrier.wait(timeout=barrier_timeout)
                    for future in futures:
                        latencies.append(future.result())
                    block_finished = clock()
                except BaseException:
                    barrier.abort()
                    raise
        except threading.BrokenBarrierError as error:
            if clock() >= deadline:
                raise _DeadlineExpired from error
            raise CalibrationError(
                "parallel calibration start barrier did not complete"
            ) from error
        block_started = start_box["value"]
    elapsed = max(block_finished - block_started, 1e-12)

    rows = sum(int(query.size) for query in queries)
    planned_bytes = rows * record_size
    return {
        "elapsed_seconds": elapsed,
        "rows": rows,
        "throughput_rows_per_second": rows / elapsed,
        "p95_call_seconds": float(np.percentile(latencies, 95)),
        "planned_read_bytes": planned_bytes,
    }


def _summarize_repetitions(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    throughput = [float(value["throughput_rows_per_second"]) for value in values]
    p95 = [float(value["p95_call_seconds"]) for value in values]
    mean = statistics.fmean(throughput)
    cv = statistics.pstdev(throughput) / mean if len(throughput) > 1 and mean else 0.0
    return {
        "median_throughput_rows_per_second": statistics.median(throughput),
        "median_p95_call_seconds": statistics.median(p95),
        "throughput_cv": cv,
        "planned_read_bytes": sum(int(value["planned_read_bytes"]) for value in values),
    }


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _resource_rank(options: Mapping[str, object]) -> tuple[int, int, int, str]:
    if options["io_backend"] == "thread_pool":
        workers = int(options["io_workers"])
        return (
            0 if workers == 0 else 1,
            workers,
            int(options["io_queue_capacity"]),
            "",
        )
    return (
        2,
        int(options["io_uring_workers"]),
        int(options["io_uring_queue_depth"]),
        "io_uring",
    )


def select_winner(
    candidates: Sequence[Mapping[str, object]],
    scenario_names: Sequence[str],
) -> dict[str, object]:
    """Pure deterministic scoring and guardrails used by tests and runtime."""

    completed = [item for item in candidates if item.get("status") == "measured"]
    baseline = next(
        (item for item in completed if item.get("name") == "sequential"),
        None,
    )
    if baseline is None:
        raise CalibrationError("sequential baseline did not complete")
    baseline_scenarios = {
        str(item["name"]): item for item in baseline["scenarios"]  # type: ignore[index]
    }

    eligible: list[Mapping[str, object]] = []
    for candidate in completed:
        scenario_map = {
            str(item["name"]): item
            for item in candidate["scenarios"]  # type: ignore[index]
        }
        throughput_ratios: list[float] = []
        paired_throughput_cvs: list[float] = []
        p95_ratios: list[float] = []
        reasons: list[str] = []
        for name in scenario_names:
            if name not in scenario_map or name not in baseline_scenarios:
                reasons.append(f"missing scenario {name}")
                continue
            current = scenario_map[name]["summary"]
            reference = baseline_scenarios[name]["summary"]
            current_repetitions = scenario_map[name].get("repetitions", [])
            reference_repetitions = baseline_scenarios[name].get(
                "repetitions", []
            )
            if (
                not isinstance(current_repetitions, Sequence)
                or not isinstance(reference_repetitions, Sequence)
                or len(current_repetitions) != len(reference_repetitions)
                or len(current_repetitions) < 2
            ):
                reasons.append(f"{name} has no paired repetitions")
                continue
            paired = [
                float(current_value["throughput_rows_per_second"])
                / max(
                    float(reference_value["throughput_rows_per_second"]),
                    1e-300,
                )
                for current_value, reference_value in zip(
                    current_repetitions,
                    reference_repetitions,
                )
            ]
            throughput_ratio = statistics.median(paired)
            reference_p95 = float(reference["median_p95_call_seconds"])
            current_p95 = float(current["median_p95_call_seconds"])
            p95_ratio = current_p95 / reference_p95 if reference_p95 > 0 else 1.0
            throughput_ratios.append(throughput_ratio)
            p95_ratios.append(p95_ratio)
            if throughput_ratio < _MIN_SCENARIO_THROUGHPUT_RATIO:
                reasons.append(f"{name} throughput regressed")
            if float(current["throughput_cv"]) > _MAX_THROUGHPUT_CV:
                reasons.append(f"{name} is unstable")
            paired_mean = statistics.fmean(paired)
            paired_cv = (
                statistics.pstdev(paired) / paired_mean
                if paired_mean and len(paired) > 1
                else 0.0
            )
            paired_throughput_cvs.append(paired_cv)
            if paired_cv > _MAX_THROUGHPUT_CV:
                reasons.append(f"{name} paired gain is unstable")
        score = _geometric_mean(throughput_ratios)
        candidate["score"] = score  # type: ignore[index]
        candidate["guardrails"] = {  # type: ignore[index]
            "eligible": not reasons,
            "reasons": reasons,
            "throughput_ratios": throughput_ratios,
            "paired_throughput_cvs": paired_throughput_cvs,
            "p95_ratios": p95_ratios,
        }
        if not reasons:
            eligible.append(candidate)

    improved = [
        item
        for item in eligible
        if item is not baseline
        and float(item.get("score", 0.0))
        >= float(baseline.get("score", 1.0)) * (1.0 + _MIN_GAIN)
    ]
    if not improved:
        return {
            "winner": str(baseline["name"]),
            "options": dict(baseline["options"]),  # type: ignore[arg-type]
            "score": float(baseline.get("score", 1.0)),
            "reason": "no candidate cleared the minimum 5% composite gain",
        }

    best_score = max(float(item.get("score", 0.0)) for item in improved)
    near_best = [
        item
        for item in improved
        if float(item.get("score", 0.0)) >= best_score * (1.0 - _TIE_BAND)
    ]
    selected = min(
        near_best,
        key=lambda item: (
            _resource_rank(item["options"]),  # type: ignore[arg-type]
            str(item["name"]),
        ),
    )
    reason = "best guarded composite score"
    return {
        "winner": str(selected["name"]),
        "options": dict(selected["options"]),  # type: ignore[arg-type]
        "score": float(selected.get("score", 0.0)),
        "reason": reason,
    }


def _baseline_drift(
    baseline: Mapping[str, object],
    scenario_names: Sequence[str],
) -> dict[str, object]:
    ratios: list[float] = []
    unstable: list[str] = []
    by_name = {
        str(item["name"]): item for item in baseline["scenarios"]  # type: ignore[index]
    }
    for name in scenario_names:
        scenario = by_name[name]
        repetitions = scenario["repetitions"]
        first = float(repetitions[0]["throughput_rows_per_second"])
        last = float(repetitions[-1]["throughput_rows_per_second"])
        ratio = last / max(first, 1e-300)
        ratios.append(ratio)
        if abs(ratio - 1.0) > _MAX_BASELINE_DRIFT:
            unstable.append(name)
        if float(scenario["summary"]["throughput_cv"]) > _MAX_THROUGHPUT_CV:
            if name not in unstable:
                unstable.append(name)
    drift = abs(_geometric_mean(ratios) - 1.0)
    return {
        "passed": not unstable and drift <= _MAX_BASELINE_DRIFT,
        "composite_drift": drift,
        "scenario_ratios": ratios,
        "unstable_scenarios": unstable,
        "limit": _MAX_BASELINE_DRIFT,
    }


def _budget_result(
    reason: str,
    *,
    candidate_discovery: Mapping[str, object] | None = None,
    sample: Mapping[str, object] | None = None,
    workload: Mapping[str, object] | None = None,
    candidates: Sequence[Mapping[str, object]] = (),
    planned_read_bytes: int = 0,
) -> dict[str, object]:
    return {
        "format": _REPORT_FORMAT,
        "version": _REPORT_VERSION,
        "status": "budget_exhausted",
        "reason": reason,
        "candidate_discovery": (
            dict(candidate_discovery)
            if candidate_discovery is not None
            else None
        ),
        "sample": dict(sample) if sample is not None else None,
        "workload": dict(workload) if workload is not None else None,
        "candidates": [dict(item) for item in candidates],
        "stability": None,
        "selection": None,
        "planned_read_bytes": planned_read_bytes,
    }


def _sample_report(
    sampled: Mapping[str, object],
    *,
    requested_rows: int,
    returned_rows: int,
    inspected_slots: int,
) -> dict[str, object]:
    return {
        "requested_rows": int(sampled.get("requested_count", requested_rows)),
        "returned_rows": returned_rows,
        "inspected_slots": inspected_slots,
        "complete": bool(sampled.get("complete", False)),
        "source": "bounded_base_index_slots",
    }


def _workload_report(
    scenarios: Sequence[_Scenario],
    *,
    repeats: int,
    correctness_rows: int,
    requested_bound: int,
    cache_control: str,
) -> dict[str, object]:
    return {
        "cache_control": cache_control,
        "repeats": repeats,
        "warmup": {
            "rounds": _WARMUP_ROUNDS,
            "measured": False,
            "accounting": "charged_not_measured",
            "blocks_per_candidate": _WARMUP_ROUNDS * len(scenarios),
            "rows_per_candidate": _WARMUP_ROUNDS
            * sum(scenario.total_rows for scenario in scenarios),
        },
        "correctness_rows": correctness_rows,
        "requested_read_bound_bytes": requested_bound,
        "scenarios": [
            {
                "name": scenario.name,
                "batch_rows": scenario.batch_rows,
                "calls_per_repeat": scenario.calls,
                "clients": scenario.clients,
            }
            for scenario in scenarios
        ],
    }


def run_bounded_calibration(
    native: object,
    *,
    record_size: int,
    current_options: Mapping[str, object] | None,
    seed: int,
    max_read_bytes: int,
    max_memory_bytes: int,
    deadline: float,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    capability_probe: Callable[[int, int], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Measure a bounded matrix against a single stable native snapshot."""

    if capability_probe is None:
        capability_probe = lambda queue_depth, workers: native.reader_capabilities(
            io_uring_queue_depth=queue_depth,
            io_uring_workers=workers,
        )

    dimension = int(native.dimension)
    dtype = np.dtype(str(native.dtype))
    if dimension <= 0 or record_size != dimension * dtype.itemsize:
        raise CalibrationError("native collection schema changed during calibration")

    if clock() >= deadline:
        return _budget_result(
            "cooperative deadline expired while opening the collection"
        )

    candidates, unavailable, cpu = _discover_candidates(
        record_size=record_size,
        current_options=current_options,
        capability_probe=capability_probe,
        deadline=deadline,
        clock=clock,
    )
    if not candidates or candidates[0].name != "sequential":
        raise CalibrationError("bounded candidate matrix has no baseline")

    discovery = {"unavailable": unavailable, "cpu": cpu}
    if clock() >= deadline:
        return _budget_result(
            "cooperative deadline expired during candidate discovery",
            candidate_discovery=discovery,
        )

    effective_cpu = int(cpu["effective"])
    clients = min(4, effective_cpu)
    minimum_measurement_workspace = max(
        max(32, record_size + 16),
        max(16 * clients, clients * (record_size + 8)),
        max(16, record_size + 8),
    )
    reserved_workspace = max(
        2 * record_size + 8,
        minimum_measurement_workspace,
    )
    minimum_memory = reserved_workspace + 16
    if max_memory_bytes < minimum_memory:
        raise ValueError(
            "max_memory_bytes is too small for the minimum calibration buffers; "
            f"at least {minimum_memory} bytes are required"
        )
    sample_rows = min(
        _MAX_SAMPLE_ROWS,
        max(1, (max_memory_bytes - reserved_workspace) // 16),
    )

    sampled = dict(
        native.sample_ids(
            requested_count=sample_rows,
            seed=seed,
            max_inspected_slots=min(_MAX_INSPECTED_SLOTS, sample_rows * 8),
        )
    )
    sample = sampled.get("ids")
    if (
        not isinstance(sample, np.ndarray)
        or sample.ndim != 1
        or sample.dtype != np.dtype(np.uint64)
        or not sample.flags.c_contiguous
        or not sample.flags.aligned
    ):
        raise CalibrationError("native ID sampler returned an invalid array")
    inspected_slots = int(sampled.get("inspected_slots", -1))
    inspection_limit = min(_MAX_INSPECTED_SLOTS, sample_rows * 8)
    if (
        int(sampled.get("requested_count", -1)) != sample_rows
        or sample.size > sample_rows
        or inspected_slots < 0
        or inspected_slots > inspection_limit
    ):
        raise CalibrationError("native ID sampler exceeded its bounded contract")
    sample = np.require(sample, dtype=np.uint64, requirements=["C", "A"])
    if sample.size == 0:
        raise CalibrationError(
            "the bounded base-index sample contains no live IDs; "
            "rebuild or compact a mutation-only collection before tuning"
        )

    sample_report = _sample_report(
        sampled,
        requested_rows=sample_rows,
        returned_rows=int(sample.size),
        inspected_slots=inspected_slots,
    )
    if clock() >= deadline:
        return _budget_result(
            "cooperative deadline expired during ID sampling",
            candidate_discovery=discovery,
            sample=sample_report,
        )

    scenarios, repeats, correctness_rows, requested_bound = _build_scenarios(
        candidate_count=len(candidates),
        sample_rows=sample.size,
        record_size=record_size,
        effective_cpu=effective_cpu,
        max_read_bytes=max_read_bytes,
        max_memory_bytes=max_memory_bytes,
    )
    if progress is not None:
        progress(
            f"sampled {sample.size:,} live IDs; testing {len(candidates)} "
            f"profiles x {len(scenarios)} workloads x {repeats} measured "
            f"repeats after {_WARMUP_ROUNDS} warm-up"
        )

    candidate_reports: list[dict[str, object]] = [
        {
            "name": candidate.name,
            "options": dict(candidate.options),
            "status": "pending",
            "correctness": None,
            "warmup": {
                "status": "pending",
                "rounds": _WARMUP_ROUNDS,
                "blocks_total": _WARMUP_ROUNDS * len(scenarios),
                "blocks_completed": 0,
                "rows": 0,
                "planned_read_bytes": 0,
            },
            "scenarios": [
                {"name": scenario.name, "repetitions": [], "summary": None}
                for scenario in scenarios
            ],
        }
        for candidate in candidates
    ]
    reports_by_name = {str(item["name"]): item for item in candidate_reports}
    planned_read_bytes = 0

    query_stream_rows = (
        len(candidates)
        * (repeats + _WARMUP_ROUNDS)
        * sum(scenario.total_rows for scenario in scenarios)
    )
    replacement_begin = correctness_rows if sample.size > correctness_rows else 0
    globally_disjoint = sample.size - replacement_begin >= query_stream_rows
    cache_control = (
        "ambient_global_disjoint_interleaved"
        if globally_disjoint
        else "ambient_replacement_warm_interleaved"
    )
    workload = _workload_report(
        scenarios,
        repeats=repeats,
        correctness_rows=correctness_rows,
        requested_bound=requested_bound,
        cache_control=cache_control,
    )
    if clock() >= deadline:
        for report in candidate_reports:
            report["status"] = "incomplete"
            report["warmup"]["status"] = "not_run"  # type: ignore[index]
        return _budget_result(
            "cooperative deadline expired before correctness preflight",
            candidate_discovery=discovery,
            sample=sample_report,
            workload=workload,
            candidates=candidate_reports,
        )

    truth_ids = np.require(sample[:correctness_rows], requirements=["C", "A"])
    truth = np.empty((truth_ids.size, dimension), dtype=dtype)
    if clock() >= deadline:
        for report in candidate_reports:
            report["status"] = "incomplete"
            report["warmup"]["status"] = "not_run"  # type: ignore[index]
        return _budget_result(
            "cooperative deadline expired before baseline correctness read",
            candidate_discovery=discovery,
            sample=sample_report,
            workload=workload,
            candidates=candidate_reports,
        )
    truth_stats = dict(native.get_into_profiled(truth_ids, truth))
    truth_planned = int(truth_stats["planned_read_bytes"])
    if (
        int(truth_stats["requested_rows"]) != truth_ids.size
        or truth_planned < 0
        or truth_planned > truth_ids.size * record_size
    ):
        raise CalibrationError("native baseline exceeded the profiled read bound")
    planned_read_bytes += truth_planned

    active: list[_Candidate] = []
    trials: dict[str, object] = {}
    exhausted = clock() >= deadline
    try:
        for candidate in candidates:
            report = reports_by_name[candidate.name]
            if clock() >= deadline:
                exhausted = True
                report["status"] = "not_run"
                continue
            trial: object | None = None
            try:
                trial = native.reader_trial(**candidate.options)
                actual = np.empty_like(truth)
                if clock() >= deadline:
                    exhausted = True
                    report["status"] = "not_run"
                    _close_trial(trial)
                    trial = None
                    del actual
                    continue
                correctness_bound = truth_ids.size * record_size
                planned_read_bytes += correctness_bound
                stats = dict(trial.get_into_profiled(truth_ids, actual))
                planned = int(stats["planned_read_bytes"])
                if (
                    int(stats["requested_rows"]) != truth_ids.size
                    or planned < 0
                    or planned > truth_ids.size * record_size
                ):
                    raise CalibrationError(
                        f"candidate {candidate.name} exceeded the zero-gap read bound"
                    )
                planned_read_bytes -= correctness_bound - planned
                if not np.array_equal(actual.view(np.uint8), truth.view(np.uint8)):
                    raise CalibrationError(
                        f"candidate {candidate.name} failed bit-exact correctness"
                    )
                report["correctness"] = {
                    "passed": True,
                    "rows": int(truth_ids.size),
                    "planned_read_bytes": planned,
                }
                report["status"] = "ready"
                active.append(candidate)
                trials[candidate.name] = trial
                del actual
                if clock() >= deadline:
                    exhausted = True
            except ReaderBackendError as error:
                if trial is not None:
                    _close_trial(trial)
                report["status"] = "unavailable"
                report["error"] = str(error)[:512]
                report["warmup"]["status"] = "not_run"  # type: ignore[index]
            except BaseException:
                if trial is not None and candidate.name not in trials:
                    _close_trial(trial)
                raise

        if exhausted:
            for report in candidate_reports:
                if report["status"] in {"ready", "pending", "not_run"}:
                    report["status"] = "incomplete"
                report["warmup"]["status"] = "not_run"  # type: ignore[index]
            return _budget_result(
                "cooperative deadline expired during correctness preflight",
                candidate_discovery=discovery,
                sample=sample_report,
                workload=workload,
                candidates=candidate_reports,
                planned_read_bytes=planned_read_bytes,
            )

        if not active or active[0].name != "sequential":
            raise CalibrationError("sequential baseline preflight failed")

        # Correctness outputs can be large for wide embeddings.  Release them
        # before allocating the global permutation and measured outputs.
        del truth, truth_ids

        active_query_stream_rows = (
            len(active)
            * (repeats + _WARMUP_ROUNDS)
            * sum(scenario.total_rows for scenario in scenarios)
        )
        globally_disjoint = (
            sample.size - replacement_begin >= active_query_stream_rows
        )
        cache_control = (
            "ambient_global_disjoint_interleaved"
            if globally_disjoint
            else "ambient_replacement_warm_interleaved"
        )
        workload = _workload_report(
            scenarios,
            repeats=repeats,
            correctness_rows=correctness_rows,
            requested_bound=requested_bound,
            cache_control=cache_control,
        )
        permutation: np.ndarray | None = None
        warmup_segment_starts: dict[tuple[int, int, str], int] = {}
        segment_starts: dict[tuple[int, int, str], int] = {}
        if globally_disjoint:
            available = sample.size - replacement_begin
            permutation = np.random.default_rng(
                _derived_seed(seed, "global-query-stream")
            ).permutation(available)
            permutation = permutation.astype(np.int64, copy=False)
            permutation += replacement_begin
            cursor = 0
            for warmup_round in range(_WARMUP_ROUNDS):
                for scenario_index, scenario in enumerate(scenarios):
                    for candidate in active:
                        warmup_segment_starts[
                            (warmup_round, scenario_index, candidate.name)
                        ] = cursor
                        cursor += scenario.total_rows
            for repetition in range(repeats):
                for scenario_index, scenario in enumerate(scenarios):
                    for candidate in active:
                        segment_starts[
                            (repetition, scenario_index, candidate.name)
                        ] = cursor
                        cursor += scenario.total_rows
            if cursor > available:
                raise CalibrationError(
                    "global disjoint query stream exceeded the sampled IDs"
                )

        # Exercise every scenario once before collecting performance samples.
        # This absorbs one-time reader/thread/ring initialization without
        # hiding post-warm-up instability in the measured repetitions.  The
        # complete zero-gap upper bound remains charged to max_read_bytes.
        exhausted = False
        for warmup_round in range(_WARMUP_ROUNDS):
            for scenario_index, scenario in enumerate(scenarios):
                if clock() >= deadline:
                    exhausted = True
                    break
                offset = (warmup_round + scenario_index) % len(active)
                order = active[offset:] + active[:offset]
                for candidate in order:
                    if clock() >= deadline:
                        exhausted = True
                        break
                    report = reports_by_name[candidate.name]
                    disjoint_indices: np.ndarray | None = None
                    if permutation is not None:
                        begin = warmup_segment_starts[
                            (warmup_round, scenario_index, candidate.name)
                        ]
                        disjoint_indices = permutation[
                            begin : begin + scenario.total_rows
                        ]
                    queries = _queries_for_candidate(
                        sample,
                        disjoint_indices=disjoint_indices,
                        replacement_begin=replacement_begin,
                        scenario=scenario,
                        seed=_derived_seed(
                            seed,
                            "warmup-replacement",
                            warmup_round,
                            scenario.name,
                            candidate.name,
                        ),
                    )
                    if progress is not None:
                        progress(
                            f"warm-up {warmup_round + 1}/{_WARMUP_ROUNDS}: "
                            f"{candidate.name} / {scenario.name}"
                        )
                    if clock() >= deadline:
                        exhausted = True
                        del queries
                        break
                    block_rows = sum(int(query.size) for query in queries)
                    block_bytes = block_rows * record_size
                    planned_read_bytes += block_bytes
                    try:
                        _measure_block(
                            trials[candidate.name],
                            queries,
                            scenario,
                            dimension=dimension,
                            dtype=dtype,
                            record_size=record_size,
                            deadline=deadline,
                            clock=clock,
                        )
                    except _DeadlineExpired:
                        exhausted = True
                        del queries
                        break
                    except ReaderBackendError as error:
                        report["status"] = "failed"
                        report["error"] = str(error)[:512]
                        report["warmup"]["status"] = "failed"  # type: ignore[index]
                        _close_trial(trials.pop(candidate.name))
                        active = [
                            item for item in active if item.name != candidate.name
                        ]
                        if candidate.name == "sequential":
                            raise CalibrationError(
                                "sequential baseline warm-up failed"
                            ) from error
                        del queries
                        continue
                    warmup_report = report["warmup"]  # type: ignore[assignment]
                    warmup_report["blocks_completed"] += 1
                    warmup_report["rows"] += block_rows
                    warmup_report["planned_read_bytes"] += block_bytes
                    del queries
                    if clock() >= deadline:
                        exhausted = True
                        break
                if exhausted:
                    break
            if exhausted:
                break

        if exhausted:
            for report in candidate_reports:
                if report["status"] in {"ready", "pending"}:
                    report["status"] = "incomplete"
                if report["warmup"]["status"] == "pending":  # type: ignore[index]
                    report["warmup"]["status"] = "incomplete"  # type: ignore[index]
            return _budget_result(
                "cooperative deadline expired during warm-up",
                candidate_discovery=discovery,
                sample=sample_report,
                workload=workload,
                candidates=candidate_reports,
                planned_read_bytes=planned_read_bytes,
            )

        expected_warmup_blocks = _WARMUP_ROUNDS * len(scenarios)
        for candidate in active:
            warmup_report = reports_by_name[candidate.name]["warmup"]
            if (  # type: ignore[index]
                warmup_report["blocks_completed"] != expected_warmup_blocks
            ):
                raise CalibrationError(
                    f"candidate {candidate.name} did not complete warm-up"
                )
            warmup_report["status"] = "complete"  # type: ignore[index]

        for repetition in range(repeats):
            for scenario_index, scenario in enumerate(scenarios):
                if clock() >= deadline:
                    exhausted = True
                    break
                offset = (
                    _WARMUP_ROUNDS + repetition + scenario_index
                ) % len(active)
                order = active[offset:] + active[:offset]
                for candidate in order:
                    if clock() >= deadline:
                        exhausted = True
                        break
                    report = reports_by_name[candidate.name]
                    disjoint_indices: np.ndarray | None = None
                    if permutation is not None:
                        begin = segment_starts[
                            (repetition, scenario_index, candidate.name)
                        ]
                        disjoint_indices = permutation[
                            begin : begin + scenario.total_rows
                        ]
                    queries = _queries_for_candidate(
                        sample,
                        disjoint_indices=disjoint_indices,
                        replacement_begin=replacement_begin,
                        scenario=scenario,
                        seed=_derived_seed(
                            seed,
                            "replacement",
                            repetition,
                            scenario.name,
                            candidate.name,
                        ),
                    )
                    if progress is not None:
                        progress(
                            f"repeat {repetition + 1}/{repeats}: "
                            f"{candidate.name} / {scenario.name}"
                        )
                    if clock() >= deadline:
                        exhausted = True
                        break
                    # Charge the complete zero-gap upper bound before starting
                    # the block. If one parallel call fails or the deadline is
                    # crossed mid-block, the report remains conservative.
                    planned_read_bytes += (
                        sum(int(query.size) for query in queries) * record_size
                    )
                    try:
                        measured = _measure_block(
                            trials[candidate.name],
                            queries,
                            scenario,
                            dimension=dimension,
                            dtype=dtype,
                            record_size=record_size,
                            deadline=deadline,
                            clock=clock,
                        )
                    except _DeadlineExpired:
                        exhausted = True
                        del queries
                        break
                    except ReaderBackendError as error:
                        report["status"] = "failed"
                        report["error"] = str(error)[:512]
                        _close_trial(trials.pop(candidate.name))
                        active = [
                            item for item in active if item.name != candidate.name
                        ]
                        if candidate.name == "sequential":
                            raise CalibrationError(
                                "sequential baseline read failed"
                            ) from error
                        del queries
                        continue
                    scenario_report = report["scenarios"][  # type: ignore[index]
                        scenario_index
                    ]
                    scenario_report["repetitions"].append(measured)
                    del queries
                    if clock() >= deadline:
                        exhausted = True
                        break
                if exhausted:
                    break
            if exhausted:
                break

        if exhausted:
            for report in candidate_reports:
                if report["status"] in {"ready", "pending"}:
                    report["status"] = "incomplete"
            return _budget_result(
                "cooperative deadline expired before the matrix completed",
                candidate_discovery=discovery,
                sample=sample_report,
                workload=workload,
                candidates=candidate_reports,
                planned_read_bytes=planned_read_bytes,
            )

        scenario_names = [scenario.name for scenario in scenarios]
        for report in candidate_reports:
            if report["status"] != "ready":
                continue
            complete = True
            for scenario_report in report["scenarios"]:  # type: ignore[assignment]
                repetitions_values = scenario_report["repetitions"]
                if len(repetitions_values) != repeats:
                    complete = False
                    break
                scenario_report["summary"] = _summarize_repetitions(
                    repetitions_values
                )
            report["status"] = "measured" if complete else "incomplete"

        selection = select_winner(candidate_reports, scenario_names)
        baseline = reports_by_name["sequential"]
        drift = _baseline_drift(baseline, scenario_names)
        status = "complete" if bool(drift["passed"]) else "unstable"
        if planned_read_bytes > max_read_bytes:
            raise CalibrationError("calibration exceeded max_read_bytes")

        return {
            "format": _REPORT_FORMAT,
            "version": _REPORT_VERSION,
            "status": status,
            "reason": (
                "bounded matrix completed"
                if status == "complete"
                else "baseline drift exceeded the stability guardrail"
            ),
            "candidate_discovery": discovery,
            "sample": sample_report,
            "workload": workload,
            "candidates": candidate_reports,
            "stability": drift,
            "selection": selection,
            "planned_read_bytes": planned_read_bytes,
        }
    finally:
        for trial in trials.values():
            _close_trial(trial)


__all__ = [
    "CalibrationError",
    "run_bounded_calibration",
    "select_winner",
    "validate_limits",
]
