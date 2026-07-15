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
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np

from ._throwdb import ReaderBackendError
from ._calibration_resources import (
    ResourceProbe,
    ResourceSnapshot,
    derive_epoch_metrics,
    effective_cgroup_cpu_quota,
)
from ._tuning import READER_DEFAULTS, normalize_reader_options


_REPORT_FORMAT: Final = "throwdb-reader-calibration"
_REPORT_VERSION: Final = 1
_MAX_CANDIDATES: Final = 12
_MAX_SAMPLE_ROWS: Final = 4_194_304
_MAX_INSPECTED_SLOTS: Final = 33_554_432
_NATIVE_SAMPLE_BYTES_PER_ROW: Final = 96
_NATIVE_SAMPLE_FIXED_BYTES: Final = 2 << 20
_CORRECTNESS_ROWS: Final = 256
_SCREENING_REPEATS: Final = 2
_VERIFICATION_REPEATS: Final = 5
_MAX_FINALISTS: Final = 5
_MAX_VERIFICATION_BACKFILL: Final = 1
_STABILITY_RECHECK_REPEATS: Final = 1
_WARMUP_ROUNDS: Final = 1
_MIN_GAIN: Final = 0.05
_TIE_BAND: Final = 0.02
_BALANCED_PERFORMANCE_BAND: Final = 0.05
_SCREENING_RESOURCE_BAND: Final = 0.10
_CPU_EFFICIENCY_BAND: Final = 0.05
_MIN_SCENARIO_THROUGHPUT_RATIO: Final = 0.90
_MAX_THROUGHPUT_CV: Final = 0.10
_MAX_RECHECK_THROUGHPUT_CV: Final = 0.20
_MAX_CPU_COST_CV: Final = 0.15
_MAX_BASELINE_DRIFT: Final = 0.20
_MAX_P99_RATIO: Final = 1.25
_MIN_LATENCY_SAMPLES: Final = 32
_MAX_MIXED_SMALL_P99_RATIO: Final = 2.0
_MAX_MIXED_SMALL_BASELINE_P99_RATIO: Final = 1.25
_MIN_MIXED_LARGE_THROUGHPUT_RATIO: Final = 0.80
_MIN_MIXED_SMALL_SAMPLES: Final = 10
_SEVERE_THROTTLED_PERIOD_FRACTION: Final = 0.50
_SEVERE_THROTTLED_WALL_RATIO: Final = 0.25
_CLEAN_THROTTLED_PERIOD_FRACTION: Final = 0.05
_CLEAN_THROTTLED_WALL_RATIO: Final = 0.01
_MAX_THREAD_POOL_WORKERS: Final = 32
_MAX_THREAD_POOL_QUEUE: Final = 64
_MAX_IO_URING_WORKERS: Final = 2
_MAX_IO_URING_QUEUE_DEPTH: Final = 64
_MAX_RANGES_PER_TASK: Final = 64
_MAX_READ_RANGE_BYTES: Final = 16 << 20
_PAGE_CACHE_BACKING_RATIO_MAX: Final = 0.10
_BACKING_STORAGE_RATIO_MIN: Final = 0.50
_MIN_EXPECTED_LOGICAL_READ_RATIO: Final = 0.75
_MAX_EXPECTED_LOGICAL_READ_RATIO: Final = 1.25
_MAX_PROFILED_IO_URING_RCHAR_RATIO: Final = 0.10


CachePreparer = Callable[[], Mapping[str, object]]


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
    cache_mode: str = "ambient"
    latency_guardrail: bool = False
    resource_objective: bool = False
    reuse_across_rounds: bool = False
    query_rows: tuple[int, ...] | None = None

    @property
    def total_rows(self) -> int:
        if self.query_rows is not None:
            return sum(self.query_rows)
        return self.batch_rows * self.calls

    @property
    def concurrent_output_rows(self) -> int:
        if self.query_rows is not None:
            return sum(self.query_rows)
        return self.batch_rows * self.clients


@dataclass(frozen=True, slots=True)
class _StagePlan:
    name: str
    scenarios: tuple[_Scenario, ...]
    repeats: int

    @property
    def rows_per_round(self) -> int:
        return sum(item.total_rows for item in self.scenarios)


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
    """Return the tightest visible mountinfo-aware cgroup-v2 quota."""

    return effective_cgroup_cpu_quota()


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
        # Matrix v3 is uniform-random. Keeping gap=0 makes the physical
        # byte bound exact and avoids selecting read amplification by accident.
        if int(options["max_gap_bytes"]) != 0:
            unavailable.append(
                {
                    "name": name,
                    "reason": "max_gap_bytes is outside bounded matrix v3",
                }
            )
            return
        resource_reason: str | None = None
        if int(options["max_read_range_bytes"]) > _MAX_READ_RANGE_BYTES:
            resource_reason = "max_read_range_bytes exceeds the matrix-v3 cap"
        elif int(options["io_ranges_per_task"]) > _MAX_RANGES_PER_TASK:
            resource_reason = "io_ranges_per_task exceeds the matrix-v3 cap"
        elif options["io_backend"] == "thread_pool" and (
            int(options["io_workers"]) > _MAX_THREAD_POOL_WORKERS
            or int(options["io_queue_capacity"]) > _MAX_THREAD_POOL_QUEUE
        ):
            resource_reason = "thread-pool topology exceeds the matrix-v3 cap"
        elif options["io_backend"] == "io_uring" and (
            int(options["io_uring_workers"]) > _MAX_IO_URING_WORKERS
            or int(options["io_uring_queue_depth"]) > _MAX_IO_URING_QUEUE_DEPTH
        ):
            resource_reason = "io_uring topology exceeds the matrix-v3 cap"
        if resource_reason is not None:
            unavailable.append({"name": name, "reason": resource_reason})
            return
        key = _canonical_key(options)
        if key in seen:
            return
        if len(candidates) >= _MAX_CANDIDATES:
            unavailable.append(
                {"name": name, "reason": "bounded matrix v3 is already full"}
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
    # V2 stopped at sixteen workers even when that boundary was still
    # improving materially.  Reserve two bounded oversubscribed points on
    # sufficiently provisioned runtimes so V3 can observe the saturation knee.
    if effective_cpu >= 8:
        thread_targets.update(
            {
                min(24, max(2, 3 * effective_cpu)),
                min(32, max(2, 4 * effective_cpu)),
            }
        )
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


def _estimated_staged_rows(
    candidate_count: int,
    correctness_rows: int,
    screening: _StagePlan,
    verification: _StagePlan,
) -> int:
    correctness = correctness_rows * (candidate_count + 1)
    screening_rows = (
        candidate_count
        * (_WARMUP_ROUNDS + screening.repeats)
        * screening.rows_per_round
    )
    finalist_count = min(_MAX_FINALISTS, candidate_count)
    verification_rows = (
        finalist_count
        * (_WARMUP_ROUNDS + verification.repeats)
        * verification.rows_per_round
    )
    backfill_count = min(
        _MAX_VERIFICATION_BACKFILL,
        max(0, candidate_count - finalist_count),
    )
    # Every backfill candidate is interleaved with a fresh sequential control,
    # so its paired ratios are never compared with an earlier baseline epoch.
    backfill_stage_candidates = backfill_count * 2
    backfill_rows = (
        backfill_stage_candidates
        * (_WARMUP_ROUNDS + verification.repeats)
        * verification.rows_per_round
    )
    # A single bounded recheck adds one paired observation.  In the worst
    # case every initial finalist is needed (baseline plus all borderline
    # candidates), and the recheck may contain the full verification matrix.
    recheck_candidate_count = (
        finalist_count + backfill_count
        if finalist_count + backfill_count > 1
        else 0
    )
    recheck_rows = (
        recheck_candidate_count
        * (_WARMUP_ROUNDS + _STABILITY_RECHECK_REPEATS)
        * verification.rows_per_round
    )
    return (
        correctness
        + screening_rows
        + verification_rows
        + backfill_rows
        + recheck_rows
    )


def _stage_ambient_query_rows(
    stage: _StagePlan,
    candidate_count: int,
) -> int:
    per_round = sum(
        scenario.total_rows
        for scenario in stage.scenarios
        if not scenario.reuse_across_rounds
    )
    return candidate_count * (_WARMUP_ROUNDS + stage.repeats) * per_round


def _minimum_scenario(scenario: _Scenario) -> _Scenario:
    if scenario.query_rows is not None:
        query_rows_list = [1 for _ in scenario.query_rows]
        if len(set(scenario.query_rows)) > 1:
            largest_index = max(
                range(len(scenario.query_rows)),
                key=scenario.query_rows.__getitem__,
            )
            query_rows_list[largest_index] = 2
        query_rows = tuple(query_rows_list)
        return _Scenario(
            scenario.name,
            max(query_rows),
            len(query_rows),
            scenario.clients,
            cache_mode=scenario.cache_mode,
            latency_guardrail=scenario.latency_guardrail,
            resource_objective=scenario.resource_objective,
            reuse_across_rounds=scenario.reuse_across_rounds,
            query_rows=query_rows,
        )
    clients = max(1, scenario.clients)
    calls = clients if clients > 1 else 1
    return _Scenario(
        scenario.name,
        1,
        calls,
        clients,
        cache_mode=scenario.cache_mode,
        latency_guardrail=scenario.latency_guardrail,
        resource_objective=scenario.resource_objective,
        reuse_across_rounds=scenario.reuse_across_rounds,
    )


def _resize_scenario(scenario: _Scenario, ratio: float) -> _Scenario:
    if scenario.query_rows is not None:
        original_largest = max(scenario.query_rows)
        original_smallest = min(scenario.query_rows)
        query_rows = tuple(
            max(1, int(value * ratio)) for value in scenario.query_rows
        )
        if original_largest > original_smallest:
            query_rows = tuple(
                max(2, value) if original == original_largest else value
                for original, value in zip(
                    scenario.query_rows,
                    query_rows,
                    strict=True,
                )
            )
        return _Scenario(
            scenario.name,
            max(query_rows),
            len(query_rows),
            scenario.clients,
            cache_mode=scenario.cache_mode,
            latency_guardrail=scenario.latency_guardrail,
            resource_objective=scenario.resource_objective,
            reuse_across_rounds=scenario.reuse_across_rounds,
            query_rows=query_rows,
        )
    target_rows = max(
        scenario.clients if scenario.clients > 1 else 1,
        int(scenario.total_rows * ratio),
    )
    if scenario.clients > 1:
        calls = scenario.clients
        batch_rows = max(1, target_rows // calls)
    else:
        calls = min(scenario.calls, target_rows)
        batch_rows = max(1, target_rows // calls)
    return _Scenario(
        scenario.name,
        batch_rows,
        calls,
        scenario.clients,
        cache_mode=scenario.cache_mode,
        latency_guardrail=scenario.latency_guardrail,
        resource_objective=scenario.resource_objective,
        reuse_across_rounds=scenario.reuse_across_rounds,
    )


def _minimum_verification_repeats(
    scenarios: Sequence[_Scenario],
) -> int:
    repeats = 2
    for scenario in scenarios:
        if scenario.name != "mixed_small_large" or scenario.query_rows is None:
            continue
        largest = max(scenario.query_rows)
        small_calls = sum(value < largest for value in scenario.query_rows)
        if small_calls < 1:
            raise CalibrationError(
                "mixed calibration workload lost its small/large distinction"
            )
        repeats = max(
            repeats,
            math.ceil(_MIN_MIXED_SMALL_SAMPLES / small_calls),
        )
    return min(_VERIFICATION_REPEATS, repeats)


def _fit_scenario_memory(
    scenario: _Scenario,
    *,
    record_size: int,
    available_bytes: int,
) -> _Scenario:
    current = scenario
    for _ in range(32):
        # Replacement sampling holds both an int64 index array and the
        # fancy-indexed uint64 ID array until the latter has been created.
        # Globally-disjoint sampling needs less, but the public memory limit
        # must remain valid for either cache mode.
        selected_bytes = current.total_rows * 16
        output_bytes = current.concurrent_output_rows * record_size
        if selected_bytes + output_bytes <= available_bytes:
            return current
        resized = _resize_scenario(current, 0.5)
        if resized == current:
            break
        current = resized
    minimum = _minimum_scenario(scenario)
    selected_bytes = minimum.total_rows * 16
    output_bytes = minimum.concurrent_output_rows * record_size
    if selected_bytes + output_bytes > available_bytes:
        raise ValueError(
            "max_memory_bytes is too small for one calibration request"
        )
    return minimum


def _minimum_workspace_bytes(record_size: int, effective_cpu: int) -> int:
    clients = min(8, effective_cpu)
    parallel_rows = clients
    if clients >= 2:
        parallel_rows = 2 + max(2, clients - 1)
    measurement = parallel_rows * (record_size + 16)
    correctness = 2 * record_size + 8
    return max(correctness, measurement, 32)


def _build_stage_plans(
    *,
    candidate_count: int,
    sample_rows: int,
    record_size: int,
    effective_cpu: int,
    max_read_bytes: int,
    max_memory_bytes: int,
) -> tuple[_StagePlan, _StagePlan, int, int]:
    clients = min(8, effective_cpu)
    # Correctness holds the sample plus truth and actual outputs. Measurement
    # holds the sample, one full-size permutation, one selected-ID array, and
    # the concurrently active outputs. Screening and verification never
    # overlap, so they share the same memory envelope.
    correctness_available = max_memory_bytes - sample_rows * 8
    measurement_available = max_memory_bytes - sample_rows * 16
    correctness_memory_rows = correctness_available // (2 * record_size + 8)
    minimum_measurement = _minimum_workspace_bytes(
        record_size,
        effective_cpu,
    )
    if correctness_memory_rows < 1 or measurement_available < minimum_measurement:
        raise ValueError(
            "max_memory_bytes is too small for the bounded sample, "
            "correctness buffers, and one parallel read "
            f"({max_memory_bytes} bytes supplied)"
        )
    screening_scenarios = (
        _Scenario("serial_point", 1, 128, 1, latency_guardrail=True),
        _Scenario("serial_small", 32, 16, 1, latency_guardrail=True),
        _Scenario("serial_1024", 1024, 2, 1, resource_objective=True),
        _Scenario(
            "parallel_1024",
            1024,
            clients,
            clients,
            resource_objective=True,
        ),
        _Scenario("serial_10000", 10_000, 1, 1, resource_objective=True),
    )
    verification_scenario_list = [
        _Scenario("serial_point", 1, 512, 1, latency_guardrail=True),
        _Scenario("serial_small", 32, 128, 1, latency_guardrail=True),
        _Scenario("serial_1024", 1024, 8, 1, resource_objective=True),
        _Scenario(
            "parallel_1024",
            1024,
            clients,
            clients,
            resource_objective=True,
        ),
        _Scenario("serial_10000", 10_000, 3, 1, resource_objective=True),
    ]
    if clients >= 2:
        small_calls = max(2, clients - 1)
        mixed_rows = (10_000, *(32 for _ in range(small_calls)))
        verification_scenario_list.append(
            _Scenario(
                "mixed_small_large",
                max(mixed_rows),
                len(mixed_rows),
                clients,
                cache_mode="mixed",
                query_rows=mixed_rows,
            )
        )
    verification_scenario_list.append(
        _Scenario(
            "hot_small",
            32,
            512,
            1,
            cache_mode="hot",
            latency_guardrail=True,
            reuse_across_rounds=True,
        )
    )
    verification_scenarios = tuple(verification_scenario_list)
    screening = _StagePlan(
        "screening",
        tuple(
            _fit_scenario_memory(
                scenario,
                record_size=record_size,
                available_bytes=measurement_available,
            )
            for scenario in screening_scenarios
        ),
        _SCREENING_REPEATS,
    )
    verification = _StagePlan(
        "verification",
        tuple(
            _fit_scenario_memory(
                scenario,
                record_size=record_size,
                available_bytes=measurement_available,
            )
            for scenario in verification_scenarios
        ),
        _VERIFICATION_REPEATS,
    )
    minimum_verification_repeats = _minimum_verification_repeats(
        verification.scenarios
    )
    budget_rows = max_read_bytes // record_size
    minimum_screening = _StagePlan(
        screening.name,
        tuple(_minimum_scenario(item) for item in screening.scenarios),
        2,
    )
    minimum_verification = _StagePlan(
        verification.name,
        tuple(_minimum_scenario(item) for item in verification.scenarios),
        minimum_verification_repeats,
    )
    minimum_workload_rows = _estimated_staged_rows(
        candidate_count,
        0,
        minimum_screening,
        minimum_verification,
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
    for _ in range(32):
        requested = _estimated_staged_rows(
            candidate_count,
            correctness,
            screening,
            verification,
        ) * record_size
        if requested <= max_read_bytes:
            return screening, verification, correctness, requested
        if verification.repeats > minimum_verification_repeats:
            verification = _StagePlan(
                verification.name,
                verification.scenarios,
                verification.repeats - 1,
            )
            continue
        ratio = max_read_bytes / requested * 0.95
        resized_screening = _StagePlan(
            screening.name,
            tuple(_resize_scenario(item, ratio) for item in screening.scenarios),
            screening.repeats,
        )
        resized_verification = _StagePlan(
            verification.name,
            tuple(
                _resize_scenario(item, ratio) for item in verification.scenarios
            ),
            verification.repeats,
        )
        if resized_screening == screening and resized_verification == verification:
            break
        screening = resized_screening
        verification = resized_verification

    minimum = _estimated_staged_rows(
        candidate_count,
        1,
        minimum_screening,
        minimum_verification,
    ) * record_size
    raise ValueError(
        "max_read_bytes is too small for the minimum calibration matrix; "
        f"at least {minimum} bytes are required"
    )


def _build_scenarios(
    **kwargs: object,
) -> tuple[list[_Scenario], int, int, int]:
    """Compatibility helper returning the V3 verification workload."""

    screening, verification, correctness, requested = _build_stage_plans(
        **kwargs  # type: ignore[arg-type]
    )
    del screening
    return list(verification.scenarios), verification.repeats, correctness, requested


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
    replacement_end: int | None = None,
    scenario: _Scenario,
    seed: int,
) -> list[np.ndarray]:
    needed = scenario.total_rows
    if disjoint_indices is not None:
        if disjoint_indices.size != needed:
            raise CalibrationError("invalid global disjoint query slice")
        indices = disjoint_indices
    else:
        end = sample.size if replacement_end is None else replacement_end
        if replacement_begin < 0 or end <= replacement_begin or end > sample.size:
            raise CalibrationError("invalid replacement query pool")
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            replacement_begin,
            end,
            size=needed,
            dtype=np.int64,
        )
    selected = np.require(sample[indices], dtype=np.uint64, requirements=["C", "A"])
    if scenario.query_rows is not None:
        queries: list[np.ndarray] = []
        begin = 0
        for rows in scenario.query_rows:
            end = begin + rows
            queries.append(selected[begin:end])
            begin = end
        if begin != selected.size:
            raise CalibrationError("mixed query pattern exceeded selected IDs")
        return queries
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
    resource_probe: ResourceProbe | None = None,
) -> dict[str, object]:
    latencies: list[float] = []
    query_sizes: list[int] = []
    resource_before: ResourceSnapshot | None = None
    resource_after: ResourceSnapshot | None = None
    if scenario.clients == 1:
        if clock() >= deadline:
            raise _DeadlineExpired
        output = np.empty((scenario.batch_rows, dimension), dtype=dtype)
        resource_before = (
            resource_probe.snapshot() if resource_probe is not None else None
        )
        block_started = clock()
        if block_started >= deadline:
            raise _DeadlineExpired
        for query in queries:
            if clock() >= deadline:
                raise _DeadlineExpired
            started = clock()
            trial.get_into(query, output)
            latencies.append(clock() - started)
            query_sizes.append(int(query.size))
        block_finished = clock()
        if block_finished >= deadline:
            raise _DeadlineExpired
        resource_after = (
            resource_probe.snapshot() if resource_probe is not None else None
        )
    else:
        start_box: dict[str, float] = {}
        resource_box: dict[str, ResourceSnapshot] = {}
        expired_box = {"value": False}

        def begin_parallel_block() -> None:
            if resource_probe is not None:
                resource_box["before"] = resource_probe.snapshot()
            started = clock()
            start_box["value"] = started
            expired_box["value"] = started >= deadline

        barrier = threading.Barrier(
            scenario.clients + 1,
            action=begin_parallel_block,
        )
        barrier_timeout = min(10.0, max(0.001, deadline - clock()))

        if scenario.query_rows is not None and len(queries) > scenario.clients:
            largest_index = max(
                range(len(queries)),
                key=lambda index: int(queries[index].size),
            )
            query_groups: list[list[np.ndarray]] = [
                [queries[largest_index]],
                *([] for _ in range(scenario.clients - 1)),
            ]
            small_group = 1
            for index, query in enumerate(queries):
                if index == largest_index:
                    continue
                query_groups[small_group].append(query)
                small_group += 1
                if small_group == scenario.clients:
                    small_group = 1
        else:
            query_groups = [[query] for query in queries]
        if (
            len(query_groups) != scenario.clients
            or any(not group for group in query_groups)
        ):
            raise CalibrationError("invalid parallel calibration query groups")

        def parallel_call(
            query_group: Sequence[np.ndarray],
        ) -> list[tuple[int, float]]:
            output = np.empty(
                (max(int(query.size) for query in query_group), dimension),
                dtype=dtype,
            )
            barrier.wait(timeout=barrier_timeout)
            if expired_box["value"]:
                raise _DeadlineExpired
            measured: list[tuple[int, float]] = []
            for query in query_group:
                started = clock()
                if started >= deadline:
                    raise _DeadlineExpired
                trial.get_into(query, output[: query.size])
                measured.append((int(query.size), clock() - started))
            return measured

        try:
            with ThreadPoolExecutor(max_workers=scenario.clients) as executor:
                try:
                    futures = [
                        executor.submit(parallel_call, query_group)
                        for query_group in query_groups
                    ]
                    barrier.wait(timeout=barrier_timeout)
                    for future in futures:
                        for query_size, latency in future.result():
                            query_sizes.append(query_size)
                            latencies.append(latency)
                    block_finished = clock()
                    if block_finished >= deadline:
                        raise _DeadlineExpired
                    if resource_probe is not None:
                        resource_box["after"] = resource_probe.snapshot()
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
        resource_before = resource_box.get("before")
        resource_after = resource_box.get("after")
    elapsed = max(block_finished - block_started, 1e-12)

    rows = sum(int(query.size) for query in queries)
    planned_bytes = rows * record_size
    result: dict[str, object] = {
        "elapsed_seconds": elapsed,
        "rows": rows,
        "throughput_rows_per_second": rows / elapsed,
        "latency_sample_count": len(latencies),
        "p50_call_seconds": float(np.percentile(latencies, 50)),
        "p95_call_seconds": float(np.percentile(latencies, 95)),
        "p99_call_seconds": float(np.percentile(latencies, 99)),
        "max_call_seconds": max(latencies),
        "planned_read_bytes": planned_bytes,
    }
    if len(set(query_sizes)) > 1:
        small_rows = min(query_sizes)
        large_rows = max(query_sizes)
        small_latencies = [
            latency
            for size, latency in zip(query_sizes, latencies, strict=True)
            if size == small_rows
        ]
        large_latencies = [
            latency
            for size, latency in zip(query_sizes, latencies, strict=True)
            if size == large_rows
        ]
        completed_large_rows = sum(
            size for size in query_sizes if size == large_rows
        )
        result["mixed"] = {
            "small": {
                "batch_rows": small_rows,
                "completed_calls": len(small_latencies),
                "p99_call_seconds": float(
                    np.percentile(small_latencies, 99)
                ),
                "max_call_seconds": max(small_latencies),
            },
            "large": {
                "batch_rows": large_rows,
                "completed_calls": len(large_latencies),
                "completed_rows": completed_large_rows,
                "throughput_rows_per_second": (
                    completed_large_rows / max(large_latencies)
                ),
                "max_call_seconds": max(large_latencies),
            },
        }
    if (
        resource_probe is not None
        and resource_before is not None
        and resource_after is not None
    ):
        result["resources"] = derive_epoch_metrics(
            resource_before,
            resource_after,
            rows=rows,
            useful_bytes=planned_bytes,
            elapsed_seconds=elapsed,
        )
    return result


def _nested_number(value: Mapping[str, object], *path: str) -> float | None:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(
        current, (int, float, np.integer, np.floating)
    ):
        return None
    result = float(current)
    return result if math.isfinite(result) else None


def _summarize_resources(
    values: Sequence[Mapping[str, object]],
    *,
    comparable_work: bool = True,
) -> dict[str, object]:
    elapsed = sum(float(value.get("elapsed_seconds", 0.0)) for value in values)
    rows = sum(int(value.get("rows", 0)) for value in values)
    useful_bytes = sum(int(value.get("useful_bytes") or 0) for value in values)
    resets = sorted(
        {
            str(reset)
            for value in values
            for reset in (
                value.get("quality", {}).get("counter_resets", [])
                if isinstance(value.get("quality"), Mapping)
                else []
            )
        }
    )
    configuration_changed = any(
        bool(value.get("quality", {}).get("cgroup_configuration_changed"))
        for value in values
        if isinstance(value.get("quality"), Mapping)
    )
    process_reset = any(item.startswith("process_usage.") for item in resets)
    io_reset = any(item.startswith("process_io.") for item in resets)
    cgroup_reset = any(
        item.startswith("cgroup_cpu.")
        or item.startswith("cgroup_pressure.")
        for item in resets
    )
    cgroup_attribution_confident = all(
        bool(value.get("quality", {}).get("cgroup_attribution_confident"))
        for value in values
        if isinstance(value.get("quality"), Mapping)
    ) and all(isinstance(value.get("quality"), Mapping) for value in values)

    def complete_sum(*path: str) -> int | float | None:
        counters = [
            current
            for value in values
            if (current := _nested_number(value, *path)) is not None
        ]
        if len(counters) != len(values):
            return None
        total = sum(counters)
        return int(total) if all(value.is_integer() for value in counters) else total

    process_cpu_values = [
        current
        for value in values
        if (current := _nested_number(value, "process", "cpu_seconds"))
        is not None
    ]
    process_complete = (
        len(process_cpu_values) == len(values) and not process_reset
    )
    process_cpu = sum(process_cpu_values) if process_complete else None
    max_rss_values = [
        current
        for value in values
        if (current := _nested_number(value, "process", "max_rss_raw"))
        is not None
    ]
    cpu_costs = (
        [
            current * 1_000_000.0 / int(value.get("rows", 0))
            for value, current in zip(
                values,
                process_cpu_values,
                strict=False,
            )
            if int(value.get("rows", 0)) > 0
        ]
        if process_complete and comparable_work
        else []
    )
    cpu_cost_mean = statistics.fmean(cpu_costs) if cpu_costs else 0.0
    cpu_cost_cv = (
        statistics.pstdev(cpu_costs) / cpu_cost_mean
        if len(cpu_costs) > 1 and cpu_cost_mean > 0
        else 0.0
    )

    storage_read_values = [
        current
        for value in values
        if (current := _nested_number(value, "io", "storage_read_bytes"))
        is not None
    ]
    io_complete = len(storage_read_values) == len(values) and not io_reset
    storage_read_bytes = int(sum(storage_read_values)) if io_complete else None

    cgroup_cpu_values = [
        current
        for value in values
        if (current := _nested_number(value, "cgroup", "cpu_seconds"))
        is not None
    ]
    cgroup_complete = (
        len(cgroup_cpu_values) == len(values)
        and not cgroup_reset
        and not configuration_changed
        and cgroup_attribution_confident
    )
    cgroup_cpu = sum(cgroup_cpu_values) if cgroup_complete else None
    periods = sum(
        int(_nested_number(value, "cgroup", "cpu_delta", "nr_periods") or 0)
        for value in values
    )
    throttled_periods = sum(
        int(_nested_number(value, "cgroup", "cpu_delta", "nr_throttled") or 0)
        for value in values
    )
    period_counters_complete = cgroup_complete and all(
        _nested_number(value, "cgroup", "cpu_delta", "nr_periods")
        is not None
        and _nested_number(
            value,
            "cgroup",
            "cpu_delta",
            "nr_throttled",
        )
        is not None
        for value in values
    )
    throttled_seconds_values = [
        current
        for value in values
        if (current := _nested_number(value, "cgroup", "throttled_seconds"))
        is not None
    ]
    throttled_complete = (
        len(throttled_seconds_values) == len(values) and cgroup_complete
    )
    throttled_seconds = (
        sum(throttled_seconds_values) if throttled_complete else None
    )
    process_share = (
        process_cpu / cgroup_cpu
        if process_cpu is not None and cgroup_cpu is not None and cgroup_cpu > 0
        else None
    )
    ambient_cpu_cores = (
        max(0.0, cgroup_cpu - process_cpu) / elapsed
        if process_cpu is not None
        and cgroup_cpu is not None
        and elapsed > 0
        else None
    )
    process_reliable = (
        process_complete
        and process_cpu is not None
        and process_cpu > 0
        # A fast multi-core candidate can accumulate enough process CPU for a
        # stable cost estimate in less than half a wall-clock second.  Requiring
        # wall time alone systematically disabled the CPU objective for the
        # fastest candidates.
        and (elapsed >= 0.5 or process_cpu >= 0.5)
        and (not comparable_work or cpu_cost_cv <= _MAX_CPU_COST_CV)
    )
    throttled_period_fraction = (
        throttled_periods / periods
        if period_counters_complete and periods > 0
        else None
    )
    throttled_wall_ratio = (
        throttled_seconds / elapsed
        if throttled_seconds is not None and elapsed > 0
        else None
    )
    attributable = process_share is not None and process_share >= 0.80
    severe_throttling = bool(
        attributable
        and throttled_period_fraction is not None
        and throttled_period_fraction > _SEVERE_THROTTLED_PERIOD_FRACTION
        and throttled_wall_ratio is not None
        and throttled_wall_ratio > _SEVERE_THROTTLED_WALL_RATIO
    )
    pressure_summary: dict[str, dict[str, dict[str, object]]] = {}
    if not cgroup_reset and not configuration_changed:
        for resource_name in ("cpu", "io", "memory"):
            resource_pressure: dict[str, dict[str, object]] = {}
            for kind in ("some", "full"):
                seconds = [
                    current
                    for value in values
                    if (
                        current := _nested_number(
                            value,
                            "cgroup",
                            "pressure",
                            resource_name,
                            kind,
                            "seconds",
                        )
                    )
                    is not None
                ]
                if not seconds:
                    continue
                end_avg10 = [
                    current
                    for value in values
                    if (
                        current := _nested_number(
                            value,
                            "cgroup",
                            "pressure",
                            resource_name,
                            kind,
                            "end_averages",
                            "avg10",
                        )
                    )
                    is not None
                ]
                total_seconds = sum(seconds)
                resource_pressure[kind] = {
                    "available_epochs": len(seconds),
                    "total_epochs": len(values),
                    "seconds": total_seconds,
                    "seconds_per_wall_second": (
                        total_seconds / elapsed if elapsed > 0 else None
                    ),
                    "max_end_avg10": max(end_avg10) if end_avg10 else None,
                }
            if resource_pressure:
                pressure_summary[resource_name] = resource_pressure
    return {
        "available": bool(process_complete or io_complete or cgroup_complete),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "process": {
            "available": process_complete,
            "reliable": process_reliable,
            "cpu_seconds": process_cpu,
            "average_cpu_cores": (
                process_cpu / elapsed
                if process_cpu is not None and elapsed > 0
                else None
            ),
            "rows_per_cpu_second": (
                rows / process_cpu
                if process_cpu is not None and process_cpu > 0
                else None
            ),
            "cpu_seconds_per_million_rows": (
                process_cpu * 1_000_000.0 / rows
                if process_cpu is not None and rows > 0
                else None
            ),
            "cpu_cost_cv": (
                cpu_cost_cv
                if process_complete and comparable_work
                else None
            ),
            "cost_stability_scope": (
                "repeated_equivalent_work"
                if comparable_work
                else "heterogeneous_work_not_compared"
            ),
            "max_rss_raw_high_water": (
                max(max_rss_values) if max_rss_values else None
            ),
            "minor_page_faults": complete_sum(
                "process",
                "delta",
                "minor_page_faults",
            ),
            "major_page_faults": complete_sum(
                "process",
                "delta",
                "major_page_faults",
            ),
            "voluntary_context_switches": complete_sum(
                "process",
                "delta",
                "voluntary_context_switches",
            ),
            "involuntary_context_switches": complete_sum(
                "process",
                "delta",
                "involuntary_context_switches",
            ),
        },
        "io": {
            "available": io_complete,
            "storage_read_bytes": storage_read_bytes,
            "logical_read_bytes": complete_sum("io", "delta", "rchar"),
            "backing_read_ratio": (
                storage_read_bytes / useful_bytes
                if storage_read_bytes is not None and useful_bytes > 0
                else None
            ),
            "read_syscalls": complete_sum("io", "delta", "syscr"),
        },
        "cgroup": {
            "available": cgroup_complete,
            "attribution_confident": cgroup_attribution_confident,
            "cpu_seconds": cgroup_cpu,
            "process_cpu_share": process_share,
            "ambient_cpu_cores": ambient_cpu_cores,
            "periods": periods,
            "throttled_periods": throttled_periods,
            "throttled_period_fraction": throttled_period_fraction,
            "throttled_seconds": throttled_seconds,
            "throttled_seconds_per_wall_second": throttled_wall_ratio,
            "attributable": attributable,
            "severe_throttling": severe_throttling,
            # PSI is diagnostic only.  Reads can create their own IO pressure,
            # so V3 does not misclassify it as ambient contention without a
            # separate idle-control epoch.
            "pressure": pressure_summary or None,
        },
        "quality": {
            "counter_resets": resets,
            "cgroup_configuration_changed": configuration_changed,
            "cgroup_attribution_confident": cgroup_attribution_confident,
        },
    }


def _summarize_repetitions(
    values: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    throughput = [float(value["throughput_rows_per_second"]) for value in values]
    p95 = [float(value["p95_call_seconds"]) for value in values]
    p50 = [
        float(value.get("p50_call_seconds", value["p95_call_seconds"]))
        for value in values
    ]
    p99 = [
        float(value.get("p99_call_seconds", value["p95_call_seconds"]))
        for value in values
    ]
    maximum = [
        float(value.get("max_call_seconds", value["p95_call_seconds"]))
        for value in values
    ]
    mean = statistics.fmean(throughput)
    cv = statistics.pstdev(throughput) / mean if len(throughput) > 1 and mean else 0.0
    summary: dict[str, object] = {
        "median_throughput_rows_per_second": statistics.median(throughput),
        "median_p50_call_seconds": statistics.median(p50),
        "median_p95_call_seconds": statistics.median(p95),
        "median_p99_call_seconds": statistics.median(p99),
        "max_call_seconds": max(maximum),
        "latency_sample_count": sum(
            int(value.get("latency_sample_count", 0)) for value in values
        ),
        "throughput_cv": cv,
        "planned_read_bytes": sum(int(value["planned_read_bytes"]) for value in values),
    }
    mixed_values = [
        value["mixed"]
        for value in values
        if isinstance(value.get("mixed"), Mapping)
    ]
    if len(mixed_values) == len(values):
        small_p99 = [
            current
            for value in mixed_values
            if (
                current := _nested_number(
                    value,
                    "small",
                    "p99_call_seconds",
                )
            )
            is not None
        ]
        large_throughput = [
            current
            for value in mixed_values
            if (
                current := _nested_number(
                    value,
                    "large",
                    "throughput_rows_per_second",
                )
            )
            is not None
        ]
        if len(small_p99) == len(values) and len(large_throughput) == len(values):
            summary["mixed"] = {
                "small": {
                    "median_p99_call_seconds": statistics.median(small_p99),
                    "completed_calls": sum(
                        int(
                            _nested_number(
                                value,
                                "small",
                                "completed_calls",
                            )
                            or 0
                        )
                        for value in mixed_values
                    ),
                },
                "large": {
                    "median_throughput_rows_per_second": statistics.median(
                        large_throughput
                    ),
                    "completed_calls": sum(
                        int(
                            _nested_number(
                                value,
                                "large",
                                "completed_calls",
                            )
                            or 0
                        )
                        for value in mixed_values
                    ),
                },
            }
    resource_values = [
        value["resources"]
        for value in values
        if isinstance(value.get("resources"), Mapping)
    ]
    if resource_values:
        summary["resources"] = _summarize_resources(resource_values)
    return summary


def _backing_ratio_classification(ratio: float | None) -> str:
    if ratio is None or not math.isfinite(ratio) or ratio < 0:
        return "unknown"
    if ratio <= _PAGE_CACHE_BACKING_RATIO_MAX:
        return "page_cache"
    if ratio >= _BACKING_STORAGE_RATIO_MIN:
        return "backing_storage"
    return "mixed"


def _storage_repetition_observation(
    repetition: Mapping[str, object],
    *,
    ordinal: int,
    io_backend: str,
    isolated_process: bool,
) -> dict[str, object]:
    planned = _nested_number(repetition, "planned_read_bytes")
    resources = repetition.get("resources")
    io = resources.get("io") if isinstance(resources, Mapping) else None
    physical = (
        _nested_number(io, "storage_read_bytes")
        if isinstance(io, Mapping) and io.get("available") is True
        else None
    )
    logical = (
        _nested_number(io, "delta", "rchar")
        if isinstance(io, Mapping) and io.get("available") is True
        else None
    )
    telemetry_available = bool(
        planned is not None
        and planned > 0
        and physical is not None
        and physical >= 0
        and logical is not None
        and logical >= 0
    )
    useful_bytes = int(planned) if telemetry_available else None
    storage_bytes = int(physical) if telemetry_available else None
    logical_bytes = int(logical) if telemetry_available else None
    backing_ratio = (
        storage_bytes / useful_bytes
        if storage_bytes is not None
        and useful_bytes is not None
        and useful_bytes > 0
        else None
    )
    logical_ratio = (
        logical_bytes / useful_bytes
        if logical_bytes is not None
        and useful_bytes is not None
        and useful_bytes > 0
        else None
    )
    procfs_attribution_passed = bool(
        logical_ratio is not None
        and _MIN_EXPECTED_LOGICAL_READ_RATIO
        <= logical_ratio
        <= _MAX_EXPECTED_LOGICAL_READ_RATIO
    )
    # Linux does not consistently charge bytes completed through io_uring to
    # /proc/self/io:rchar.  In the exec-isolated calibration worker the
    # profiled request volume is exact (matrix-v3 always uses max_gap_bytes=0),
    # so an *under-reporting* rchar counter may be replaced by that volume for
    # attribution.  We deliberately do not forgive over-attribution: it can
    # still reveal unrelated buffered reads in the supposedly isolated epoch.
    profiled_io_uring_fallback = bool(
        telemetry_available
        and isolated_process
        and io_backend == "io_uring"
        and logical_ratio is not None
        and logical_ratio <= _MAX_PROFILED_IO_URING_RCHAR_RATIO
    )
    io_attribution_passed = bool(
        procfs_attribution_passed or profiled_io_uring_fallback
    )
    logical_accounting_method = (
        "profiled_planned_bytes_isolated_io_uring"
        if profiled_io_uring_fallback
        else "procfs_rchar"
    )
    attributed_logical_bytes = (
        useful_bytes if profiled_io_uring_fallback else logical_bytes
    )
    attributed_logical_ratio = (
        attributed_logical_bytes / useful_bytes
        if attributed_logical_bytes is not None
        and useful_bytes is not None
        and useful_bytes > 0
        else None
    )
    classification = (
        _backing_ratio_classification(backing_ratio)
        if io_attribution_passed
        else "unknown"
    )
    return {
        "ordinal": ordinal,
        "telemetry_available": telemetry_available,
        "useful_read_bytes": useful_bytes,
        "storage_read_bytes": storage_bytes,
        "backing_read_ratio": backing_ratio,
        "logical_read_bytes": logical_bytes,
        "logical_read_ratio": logical_ratio,
        "logical_accounting_method": logical_accounting_method,
        "logical_counter_compatible": procfs_attribution_passed,
        "attributed_logical_read_bytes": attributed_logical_bytes,
        "attributed_logical_read_ratio": attributed_logical_ratio,
        "io_attribution_passed": io_attribution_passed,
        "classification": classification,
        "passed": classification == "backing_storage",
    }


def _storage_stage_observation(
    stage: Mapping[str, object],
    *,
    candidate: str,
    scope: str,
    source: str,
    io_backend: str,
    isolated_process: bool,
) -> dict[str, object]:
    scenarios_value = stage.get("scenarios")
    scenarios = (
        list(scenarios_value)
        if isinstance(scenarios_value, Sequence)
        and not isinstance(scenarios_value, (str, bytes))
        else []
    )
    relevant = [
        item
        for item in scenarios
        if isinstance(item, Mapping)
        and item.get("cache_mode") != "hot"
    ]
    scenario_names = [str(item.get("name")) for item in relevant]
    useful_bytes = 0
    storage_bytes = 0
    scenario_observations: list[dict[str, object]] = []
    for scenario in relevant:
        repetitions_value = scenario.get("repetitions")
        repetitions = (
            list(repetitions_value)
            if isinstance(repetitions_value, Sequence)
            and not isinstance(repetitions_value, (str, bytes))
            else []
        )
        repetition_observations = [
            _storage_repetition_observation(
                value,
                ordinal=index,
                io_backend=io_backend,
                isolated_process=isolated_process,
            )
            for index, value in enumerate(repetitions)
            if isinstance(value, Mapping)
        ]
        repetition_classifications = [
            str(value["classification"])
            for value in repetition_observations
        ]
        complete_telemetry = bool(repetitions) and (
            len(repetition_observations) == len(repetitions)
            and all(
                value["telemetry_available"] is True
                for value in repetition_observations
            )
        )
        scenario_useful = (
            sum(
                int(value["useful_read_bytes"])
                for value in repetition_observations
            )
            if complete_telemetry
            else None
        )
        scenario_storage = (
            sum(
                int(value["storage_read_bytes"])
                for value in repetition_observations
            )
            if complete_telemetry
            else None
        )
        scenario_logical = (
            sum(
                int(value["logical_read_bytes"])
                for value in repetition_observations
            )
            if complete_telemetry
            else None
        )
        scenario_ratio = (
            scenario_storage / scenario_useful
            if scenario_storage is not None
            and scenario_useful is not None
            and scenario_useful > 0
            else None
        )
        logical_ratio = (
            scenario_logical / scenario_useful
            if scenario_logical is not None
            and scenario_useful is not None
            and scenario_useful > 0
            else None
        )
        io_attribution_passed = bool(
            repetition_observations
            and len(repetition_observations) == len(repetitions)
            and all(
                value["io_attribution_passed"] is True
                for value in repetition_observations
            )
        )
        failed_attribution = [
            {
                "ordinal": int(value["ordinal"]),
                "logical_read_ratio": value.get("logical_read_ratio"),
            }
            for value in repetition_observations
            if value["io_attribution_passed"] is not True
        ]
        if (
            not repetition_observations
            or "unknown" in repetition_classifications
        ):
            scenario_classification = "unknown"
        elif all(
            value == "backing_storage"
            for value in repetition_classifications
        ):
            scenario_classification = "backing_storage"
        elif all(value == "page_cache" for value in repetition_classifications):
            scenario_classification = "page_cache"
        else:
            scenario_classification = "mixed"
        scenario_observations.append(
            {
                "name": str(scenario.get("name")),
                "telemetry_available": complete_telemetry,
                "useful_read_bytes": scenario_useful,
                "storage_read_bytes": scenario_storage,
                "backing_read_ratio": scenario_ratio,
                "logical_read_bytes": scenario_logical,
                "logical_read_ratio": logical_ratio,
                "logical_accounting_methods": sorted(
                    {
                        str(value["logical_accounting_method"])
                        for value in repetition_observations
                    }
                ),
                "io_attribution_passed": io_attribution_passed,
                "io_attribution_failed_repetitions": failed_attribution,
                "classification": scenario_classification,
                "passed": scenario_classification == "backing_storage",
                "repetitions": repetition_observations,
            }
        )
        if complete_telemetry:
            useful_bytes += int(scenario_useful)
            storage_bytes += int(scenario_storage)
    classifications = [
        str(item["classification"]) for item in scenario_observations
    ]
    complete = bool(scenario_observations) and "unknown" not in classifications
    ratio = storage_bytes / useful_bytes if complete and useful_bytes > 0 else None
    if not complete:
        classification = "unknown"
    elif all(item == "backing_storage" for item in classifications):
        classification = "backing_storage"
    elif all(item == "page_cache" for item in classifications):
        classification = "page_cache"
    else:
        classification = "mixed"
    return {
        "candidate": candidate,
        "scope": scope,
        "source": source,
        "io_backend": io_backend,
        "scenario_names": scenario_names,
        "telemetry_available": complete and useful_bytes > 0,
        "useful_read_bytes": useful_bytes if complete else None,
        "storage_read_bytes": storage_bytes if complete else None,
        "backing_read_ratio": ratio,
        "classification": classification,
        "passed": classification == "backing_storage",
        "scenarios": scenario_observations,
    }


def _observation_group(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    classifications = [str(item["classification"]) for item in observations]
    if not observations or "unknown" in classifications:
        classification = "unknown"
    elif all(item == "backing_storage" for item in classifications):
        classification = "backing_storage"
    elif all(item == "page_cache" for item in classifications):
        classification = "page_cache"
    else:
        classification = "mixed"
    useful_values = [
        int(value)
        for item in observations
        if (value := item.get("useful_read_bytes")) is not None
    ]
    storage_values = [
        int(value)
        for item in observations
        if (value := item.get("storage_read_bytes")) is not None
    ]
    complete = (
        bool(observations)
        and len(useful_values) == len(observations)
        and len(storage_values) == len(observations)
    )
    useful_bytes = sum(useful_values) if complete else None
    storage_bytes = sum(storage_values) if complete else None
    ratio = (
        storage_bytes / useful_bytes
        if storage_bytes is not None
        and useful_bytes is not None
        and useful_bytes > 0
        else None
    )
    return {
        "classification": classification,
        "passed": classification == "backing_storage",
        "observation_count": len(observations),
        "useful_read_bytes": useful_bytes,
        "storage_read_bytes": storage_bytes,
        "backing_read_ratio": ratio,
    }


def _storage_stability_report(
    candidate_reports: Sequence[Mapping[str, object]],
    finalist_reports: Sequence[Mapping[str, object]],
    *,
    isolated_process: bool = False,
) -> dict[str, object]:
    """Classify whether decision-bearing measurements reached storage.

    Every decision-bearing scenario participates except the intentionally
    repeated hot workload.  Screening remains part of the guardrail because it
    decides which candidates reach verification; fresh paired baselines
    created by verification backfill are included too.
    """

    observations: list[dict[str, object]] = []
    for report in candidate_reports:
        options = report.get("options")
        io_backend = (
            str(options.get("io_backend"))
            if isinstance(options, Mapping)
            else "unknown"
        )
        screening = report.get("screening")
        if isinstance(screening, Mapping) and screening.get("status") == "measured":
            observations.append(
                _storage_stage_observation(
                    screening,
                    candidate=str(report.get("name")),
                    scope="screening",
                    source="candidate_screening",
                    io_backend=io_backend,
                    isolated_process=isolated_process,
                )
            )
    for report in finalist_reports:
        options = report.get("options")
        io_backend = (
            str(options.get("io_backend"))
            if isinstance(options, Mapping)
            else "unknown"
        )
        verification = report.get("verification")
        if (
            isinstance(verification, Mapping)
            and verification.get("status") == "measured"
        ):
            observations.append(
                _storage_stage_observation(
                    verification,
                    candidate=str(report.get("name")),
                    scope="verification",
                    source="candidate_verification",
                    io_backend=io_backend,
                    isolated_process=isolated_process,
                )
            )
        pairing = report.get("pairing_reference")
        if isinstance(pairing, Mapping):
            observations.append(
                _storage_stage_observation(
                    pairing,
                    candidate=str(pairing.get("candidate", "sequential")),
                    scope="verification",
                    source=str(pairing.get("source", "pairing_reference")),
                    io_backend="thread_pool",
                    isolated_process=isolated_process,
                )
            )

    screening_observations = [
        item for item in observations if item["scope"] == "screening"
    ]
    verification_observations = [
        item for item in observations if item["scope"] == "verification"
    ]
    screening = _observation_group(screening_observations)
    verification = _observation_group(verification_observations)
    overall = _observation_group(observations)
    io_contaminated_scenarios = [
        {
            "candidate": str(observation["candidate"]),
            "scope": str(observation["scope"]),
            "scenario": str(scenario["name"]),
            "logical_read_ratio": scenario[
                "io_attribution_failed_repetitions"
            ][0]["logical_read_ratio"],
            "failed_repetitions": scenario[
                "io_attribution_failed_repetitions"
            ],
        }
        for observation in observations
        for scenario in observation.get("scenarios", [])
        if isinstance(scenario, Mapping)
        and scenario.get("telemetry_available") is True
        and scenario.get("io_attribution_passed") is False
        and scenario.get("io_attribution_failed_repetitions")
    ]
    logical_counter_fallback_scenarios = [
        {
            "candidate": str(observation["candidate"]),
            "scope": str(observation["scope"]),
            "scenario": str(scenario["name"]),
            "io_backend": str(observation["io_backend"]),
            "method": "profiled_planned_bytes_isolated_io_uring",
            "observed_procfs_rchar_ratio": scenario.get(
                "logical_read_ratio"
            ),
            "repetitions": [
                {
                    "ordinal": int(repetition["ordinal"]),
                    "observed_procfs_rchar_ratio": repetition.get(
                        "logical_read_ratio"
                    ),
                }
                for repetition in scenario.get("repetitions", [])
                if isinstance(repetition, Mapping)
                and repetition.get("logical_accounting_method")
                == "profiled_planned_bytes_isolated_io_uring"
            ],
        }
        for observation in observations
        for scenario in observation.get("scenarios", [])
        if isinstance(scenario, Mapping)
        and "profiled_planned_bytes_isolated_io_uring"
        in scenario.get("logical_accounting_methods", [])
    ]
    passed = bool(
        screening["passed"]
        and verification["passed"]
        and observations
    )
    classification = str(overall["classification"])
    if passed:
        reason = (
            "screening and authoritative verification were backed by "
            "storage reads"
        )
    elif classification == "page_cache":
        reason = (
            "decision-bearing reads were dominated by the process page cache"
        )
    elif classification == "mixed":
        reason = (
            "decision-bearing epochs observed inconsistent or partial "
            "backing-storage reads"
        )
    elif io_contaminated_scenarios:
        reason = (
            "process-wide logical I/O counters fall outside the expected "
            "ThrowDB payload range"
        )
    else:
        reason = (
            "reliable /proc/self/io backing-read evidence was unavailable "
            "for every decision-bearing epoch"
        )
    return {
        "classification": classification,
        "passed": passed,
        "persistence_safe": passed,
        "reason": reason,
        "telemetry": "/proc/self/io read_bytes",
        "telemetry_scope": (
            "isolated_calibration_process"
            if isolated_process
            else "process_wide"
        ),
        "measurement_scope": (
            "screening_and_authoritative_verification_non_hot_scenarios"
        ),
        "thresholds": {
            "page_cache_max_backing_read_ratio": (
                _PAGE_CACHE_BACKING_RATIO_MAX
            ),
            "backing_storage_min_backing_read_ratio": (
                _BACKING_STORAGE_RATIO_MIN
            ),
            "logical_read_ratio_min": _MIN_EXPECTED_LOGICAL_READ_RATIO,
            "logical_read_ratio_max": _MAX_EXPECTED_LOGICAL_READ_RATIO,
            "isolated_io_uring_rchar_fallback_max_ratio": (
                _MAX_PROFILED_IO_URING_RCHAR_RATIO
            ),
        },
        "screening": screening,
        "verification": verification,
        "observation_count": len(observations),
        "useful_read_bytes": overall["useful_read_bytes"],
        "storage_read_bytes": overall["storage_read_bytes"],
        "backing_read_ratio": overall["backing_read_ratio"],
        "io_contaminated_scenarios": io_contaminated_scenarios,
        "logical_counter_fallback_scenarios": (
            logical_counter_fallback_scenarios
        ),
        "observations": observations,
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


def _candidate_cpu_cost(candidate: Mapping[str, object]) -> tuple[float, bool]:
    summary = candidate.get("resource_summary")
    if not isinstance(summary, Mapping):
        return 0.0, False
    reliable = bool(
        summary.get("process", {}).get("reliable")
        if isinstance(summary.get("process"), Mapping)
        else False
    )
    process_summary = summary.get("process")
    if (
        isinstance(process_summary, Mapping)
        and process_summary.get("cost_stability_scope")
        == "heterogeneous_work_not_compared"
    ):
        reliable = False
    value = _nested_number(
        summary,
        "process",
        "cpu_seconds_per_million_rows",
    )
    return (value or 0.0), bool(reliable and value is not None and value > 0)


def _scenario_cpu_cost(scenario: Mapping[str, object]) -> tuple[float, bool]:
    summary = scenario.get("summary")
    if not isinstance(summary, Mapping):
        return 0.0, False
    resources = summary.get("resources")
    if not isinstance(resources, Mapping):
        return 0.0, False
    process = resources.get("process")
    if not isinstance(process, Mapping):
        return 0.0, False
    value = _nested_number(
        resources,
        "process",
        "cpu_seconds_per_million_rows",
    )
    return (
        value or 0.0,
        bool(process.get("reliable") and value is not None and value > 0),
    )


def _candidate_throttle(candidate: Mapping[str, object]) -> tuple[float, float] | None:
    summary = candidate.get("resource_summary")
    if not isinstance(summary, Mapping):
        return None
    cgroup = summary.get("cgroup")
    if (
        not isinstance(cgroup, Mapping)
        or not bool(cgroup.get("attribution_confident"))
        or not bool(cgroup.get("attributable"))
    ):
        return None
    period = _nested_number(
        cgroup,
        "throttled_period_fraction",
    )
    wall = _nested_number(
        cgroup,
        "throttled_seconds_per_wall_second",
    )
    if period is None or wall is None:
        return None
    return period, wall


def _throttle_class(candidate: Mapping[str, object]) -> int:
    throttle = _candidate_throttle(candidate)
    if throttle is None:
        return 1
    if (
        throttle[0] <= _CLEAN_THROTTLED_PERIOD_FRACTION
        and throttle[1] <= _CLEAN_THROTTLED_WALL_RATIO
    ):
        return 0
    return 2


def _selection_alternative(candidate: Mapping[str, object]) -> dict[str, object]:
    guardrails = candidate.get("guardrails")
    eligible = bool(
        guardrails.get("eligible")
        if isinstance(guardrails, Mapping)
        else False
    )
    reasons_value = (
        guardrails.get("reasons", [])
        if isinstance(guardrails, Mapping)
        else []
    )
    reasons = (
        [str(item) for item in reasons_value]
        if isinstance(reasons_value, Sequence)
        and not isinstance(reasons_value, (str, bytes))
        else []
    )
    return {
        "available": True,
        "winner": str(candidate["name"]),
        "options": dict(candidate["options"]),  # type: ignore[arg-type]
        "score": float(candidate.get("score", 0.0)),
        "cpu_efficiency_score": candidate.get("cpu_efficiency_score"),
        "tail_score": candidate.get("tail_score"),
        "eligible": eligible,
        "reasons": reasons,
    }


def _unavailable_selection_alternative(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "winner": None,
        "options": None,
        "score": None,
        "cpu_efficiency_score": None,
        "tail_score": None,
        "eligible": False,
        "reasons": [reason],
    }


def _screening_resource_probe(
    ranked: Sequence[Mapping[str, object]],
    provisional: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Retain a cheap near-leader when screening CPU data is unusable.

    A two-repeat screening outlier can fail its own guardrails while still
    carrying the largest raw score.  Such an outlier must not raise the
    near-leader threshold for otherwise viable candidates.  If every measured
    candidate is uncertain, fall back to the full ranked set: screening is too
    short to leave the resource-probe slot empty solely because all variance
    guardrails fired.
    """

    coverage = provisional.get("cpu_resource_coverage")
    if (
        not ranked
        or not isinstance(coverage, Mapping)
        or coverage.get("mode") != "performance_only_fallback"
    ):
        return None
    eligible = [
        item
        for item in ranked
        if isinstance(item.get("guardrails"), Mapping)
        and item["guardrails"].get("eligible") is True  # type: ignore[index,union-attr]
    ]
    reference_pool = eligible or list(ranked)
    best_score = max(
        float(item.get("score", 0.0)) for item in reference_pool
    )
    within_band = [
        item
        for item in reference_pool
        if float(item.get("score", 0.0))
        >= best_score * (1.0 - _SCREENING_RESOURCE_BAND)
    ]
    return min(
        within_band,
        key=lambda item: (
            _resource_rank(item["options"]),  # type: ignore[arg-type]
            -float(item.get("score", 0.0)),
            str(item["name"]),
        ),
    )


def select_winner(
    candidates: Sequence[Mapping[str, object]],
    scenario_names: Sequence[str],
) -> dict[str, object]:
    """Pure deterministic scoring and guardrails used by tests and runtime."""

    completed = [item for item in candidates if item.get("status") == "measured"]
    for candidate in completed:
        # The same report dictionaries are intentionally reused between the
        # screening and verification decisions.  Remove every derived field
        # first so an eligibility transition cannot leave stale Pareto or CPU
        # annotations from the shorter stage.
        for key in (
            "score",
            "cpu_efficiency_score",
            "cpu_efficiency_by_scenario",
            "cpu_efficiency_scenarios_used",
            "tail_score",
            "guardrails",
            "topology_rank",
            "dominated_by",
        ):
            candidate.pop(key, None)  # type: ignore[attr-defined]
    baseline = next(
        (item for item in completed if item.get("name") == "sequential"),
        None,
    )
    if baseline is None:
        raise CalibrationError("sequential baseline did not complete")
    baseline_scenarios = {
        str(item["name"]): item for item in baseline["scenarios"]  # type: ignore[index]
    }

    resource_scenario_names = [
        name
        for name, scenario in baseline_scenarios.items()
        if bool(scenario.get("resource_objective", False))
    ]
    baseline_cpu_cost, baseline_cpu_reliable = _candidate_cpu_cost(baseline)
    eligible: list[Mapping[str, object]] = []
    for candidate in completed:
        scenario_map = {
            str(item["name"]): item
            for item in candidate["scenarios"]  # type: ignore[index]
        }
        reference_scenarios = baseline_scenarios
        pairing_source = "primary_verification_baseline"
        pairing_reference = candidate.get("pairing_reference")
        pairing_reference_invalid = False
        if candidate is not baseline and isinstance(pairing_reference, Mapping):
            reference_value = pairing_reference.get("scenarios")
            if isinstance(reference_value, Sequence) and not isinstance(
                reference_value, (str, bytes)
            ):
                reference_scenarios = {
                    str(item["name"]): item
                    for item in reference_value
                    if isinstance(item, Mapping) and "name" in item
                }
                pairing_source = str(
                    pairing_reference.get(
                        "source",
                        "candidate_specific_baseline",
                    )
                )
            else:
                pairing_reference_invalid = True
        throughput_ratios: list[float] = []
        paired_throughput_cvs: list[float] = []
        p95_ratios: list[float] = []
        p99_ratios: list[float] = []
        reliable_tail_ratios: list[float] = []
        cpu_efficiency_by_scenario: dict[str, float] = {}
        stability_recheck_scenarios: list[str] = []
        stability_overrides: list[str] = []
        reasons: list[str] = []
        if pairing_reference_invalid:
            reasons.append("invalid candidate pairing reference")
        for name in scenario_names:
            if name not in scenario_map or name not in reference_scenarios:
                reasons.append(f"missing scenario {name}")
                continue
            current = scenario_map[name]["summary"]
            reference = reference_scenarios[name]["summary"]
            current_repetitions = scenario_map[name].get("repetitions", [])
            reference_repetitions = reference_scenarios[name].get(
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
            current_p99 = float(
                current.get("median_p99_call_seconds", current_p95)
            )
            reference_p99 = float(
                reference.get("median_p99_call_seconds", reference_p95)
            )
            p99_ratio = (
                current_p99 / reference_p99 if reference_p99 > 0 else 1.0
            )
            p99_ratios.append(p99_ratio)
            if throughput_ratio < _MIN_SCENARIO_THROUGHPUT_RATIO:
                reasons.append(f"{name} throughput regressed")
            current_cv = float(current["throughput_cv"])
            paired_mean = statistics.fmean(paired)
            paired_cv = (
                statistics.pstdev(paired) / paired_mean
                if paired_mean and len(paired) > 1
                else 0.0
            )
            paired_throughput_cvs.append(paired_cv)
            stability_rechecked = bool(
                scenario_map[name].get("stability_rechecked", False)
            )
            stability_cv = max(current_cv, paired_cv)
            positive_lower_bound = min(paired) >= 1.0 + _MIN_GAIN
            borderline_stability = bool(
                candidate is not baseline
                and stability_cv > _MAX_THROUGHPUT_CV
                and stability_cv <= _MAX_RECHECK_THROUGHPUT_CV
                and positive_lower_bound
            )
            if borderline_stability and not stability_rechecked:
                stability_recheck_scenarios.append(name)
            elif borderline_stability:
                # One bounded extra observation has confirmed that every
                # paired sample is still materially positive.  Keep the CV
                # visible, but do not turn a small amount of residual noise
                # into a permanent rejection.
                stability_overrides.append(name)
            if (
                candidate is not baseline
                and current_cv > _MAX_THROUGHPUT_CV
                and not (borderline_stability and stability_rechecked)
            ):
                reasons.append(f"{name} is unstable")
            if (
                candidate is not baseline
                and paired_cv > _MAX_THROUGHPUT_CV
                and not (borderline_stability and stability_rechecked)
            ):
                reasons.append(f"{name} paired gain is unstable")
            latency_samples = min(
                int(current.get("latency_sample_count", 0)),
                int(reference.get("latency_sample_count", 0)),
            )
            latency_guardrail = bool(
                scenario_map[name].get("latency_guardrail", False)
            )
            if latency_guardrail and latency_samples >= _MIN_LATENCY_SAMPLES:
                reliable_tail_ratios.append(p99_ratio)
            if (
                latency_guardrail
                and latency_samples >= _MIN_LATENCY_SAMPLES
                and p99_ratio > _MAX_P99_RATIO
            ):
                reasons.append(f"{name} p99 latency regressed")
            if name in resource_scenario_names:
                reference_cost, reference_reliable = _scenario_cpu_cost(
                    reference_scenarios[name]
                )
                current_cost, current_reliable = _scenario_cpu_cost(
                    scenario_map[name]
                )
                if (
                    reference_reliable
                    and current_reliable
                    and current_cost > 0
                ):
                    cpu_efficiency_by_scenario[name] = (
                        reference_cost / current_cost
                    )
        mixed_guardrail: dict[str, object] | None = None
        reference_mixed = reference_scenarios.get("mixed_small_large")
        reference_mixed_summary = (
            reference_mixed.get("summary")
            if isinstance(reference_mixed, Mapping)
            else None
        )
        reference_mixed_small_p99 = (
            _nested_number(
                reference_mixed_summary,
                "mixed",
                "small",
                "median_p99_call_seconds",
            )
            if isinstance(reference_mixed_summary, Mapping)
            else None
        )
        mixed_scenario = scenario_map.get("mixed_small_large")
        standalone_small = scenario_map.get("serial_small")
        if standalone_small is None:
            standalone_small = scenario_map.get("small")
        standalone_large = scenario_map.get("serial_10000")
        if standalone_large is None:
            standalone_large = scenario_map.get("large")
        if (
            isinstance(mixed_scenario, Mapping)
            and isinstance(standalone_small, Mapping)
            and isinstance(standalone_large, Mapping)
        ):
            mixed_summary = mixed_scenario.get("summary")
            small_summary = standalone_small.get("summary")
            large_summary = standalone_large.get("summary")
            if (
                isinstance(mixed_summary, Mapping)
                and isinstance(small_summary, Mapping)
                and isinstance(large_summary, Mapping)
            ):
                mixed_small_p99 = _nested_number(
                    mixed_summary,
                    "mixed",
                    "small",
                    "median_p99_call_seconds",
                )
                standalone_small_p99 = _nested_number(
                    small_summary,
                    "median_p99_call_seconds",
                )
                small_samples = int(
                    _nested_number(
                        mixed_summary,
                        "mixed",
                        "small",
                        "completed_calls",
                    )
                    or 0
                )
                mixed_large_throughput = _nested_number(
                    mixed_summary,
                    "mixed",
                    "large",
                    "median_throughput_rows_per_second",
                )
                standalone_large_throughput = _nested_number(
                    large_summary,
                    "median_throughput_rows_per_second",
                )
                small_ratio = (
                    mixed_small_p99 / standalone_small_p99
                    if mixed_small_p99 is not None
                    and standalone_small_p99 is not None
                    and standalone_small_p99 > 0
                    else None
                )
                baseline_small_ratio = (
                    mixed_small_p99 / reference_mixed_small_p99
                    if mixed_small_p99 is not None
                    and reference_mixed_small_p99 is not None
                    and reference_mixed_small_p99 > 0
                    else None
                )
                large_ratio = (
                    mixed_large_throughput / standalone_large_throughput
                    if mixed_large_throughput is not None
                    and standalone_large_throughput is not None
                    and standalone_large_throughput > 0
                    else None
                )
                mixed_guardrail = {
                    "evaluated": small_samples >= _MIN_MIXED_SMALL_SAMPLES,
                    "enforced": candidate is not baseline,
                    "small_p99_ratio": small_ratio,
                    "small_p99_baseline_ratio": baseline_small_ratio,
                    "small_completed_calls": small_samples,
                    "small_p99_limit": _MAX_MIXED_SMALL_P99_RATIO,
                    "small_p99_baseline_limit": (
                        _MAX_MIXED_SMALL_BASELINE_P99_RATIO
                    ),
                    "large_throughput_ratio": large_ratio,
                    "large_throughput_minimum": (
                        _MIN_MIXED_LARGE_THROUGHPUT_RATIO
                    ),
                }
                if candidate is not baseline:
                    if small_samples < _MIN_MIXED_SMALL_SAMPLES:
                        reasons.append(
                            "mixed workload has insufficient fairness samples"
                        )
                    else:
                        if small_ratio is None or baseline_small_ratio is None:
                            reasons.append(
                                "mixed workload fairness metrics are unavailable"
                            )
                        elif (
                            small_ratio > _MAX_MIXED_SMALL_P99_RATIO
                            and baseline_small_ratio
                            > _MAX_MIXED_SMALL_BASELINE_P99_RATIO
                        ):
                            reasons.append(
                                "mixed workload starved small requests"
                            )
                        if (
                            large_ratio is None
                            or large_ratio
                            < _MIN_MIXED_LARGE_THROUGHPUT_RATIO
                        ):
                            reasons.append(
                                "mixed workload regressed large throughput"
                            )
        if "mixed_small_large" in scenario_names and mixed_guardrail is None:
            mixed_guardrail = {
                "evaluated": False,
                "enforced": candidate is not baseline,
                "small_p99_ratio": None,
                "small_p99_baseline_ratio": None,
                "small_completed_calls": 0,
                "small_p99_limit": _MAX_MIXED_SMALL_P99_RATIO,
                "small_p99_baseline_limit": (
                    _MAX_MIXED_SMALL_BASELINE_P99_RATIO
                ),
                "large_throughput_ratio": None,
                "large_throughput_minimum": (
                    _MIN_MIXED_LARGE_THROUGHPUT_RATIO
                ),
            }
            if candidate is not baseline:
                reasons.append("mixed workload fairness metrics are unavailable")
        score = _geometric_mean(throughput_ratios)
        cpu_efficiency_score: float | None = None
        if not resource_scenario_names:
            # Backward-compatible path for synthetic and V2-style candidate
            # reports that only carry one aggregate resource summary.
            candidate_cpu_cost, candidate_cpu_reliable = _candidate_cpu_cost(
                candidate
            )
            if (
                baseline_cpu_reliable
                and candidate_cpu_reliable
                and candidate_cpu_cost > 0
            ):
                cpu_efficiency_score = baseline_cpu_cost / candidate_cpu_cost
        tail_score = (
            _geometric_mean([1.0 / value for value in reliable_tail_ratios])
            if reliable_tail_ratios
            and all(value > 0 for value in reliable_tail_ratios)
            else None
        )
        resource_summary = candidate.get("resource_summary")
        severe_throttling = bool(
            resource_summary.get("cgroup", {}).get("severe_throttling")
            if isinstance(resource_summary, Mapping)
            and isinstance(resource_summary.get("cgroup"), Mapping)
            else False
        )
        if severe_throttling:
            reasons.append("severe attributable CPU throttling")
        candidate["score"] = score  # type: ignore[index]
        candidate["cpu_efficiency_score"] = cpu_efficiency_score  # type: ignore[index]
        candidate["cpu_efficiency_by_scenario"] = (  # type: ignore[index]
            cpu_efficiency_by_scenario
        )
        candidate["tail_score"] = tail_score  # type: ignore[index]
        candidate["guardrails"] = {  # type: ignore[index]
            "eligible": not reasons,
            "reasons": reasons,
            "throughput_ratios": throughput_ratios,
            "paired_throughput_cvs": paired_throughput_cvs,
            "p95_ratios": p95_ratios,
            "p99_ratios": p99_ratios,
            "tail_ratios_used": reliable_tail_ratios,
            "cpu_efficiency_ratios": list(
                cpu_efficiency_by_scenario.values()
            ),
            "mixed_fairness": mixed_guardrail,
            "stability_recheck_scenarios": stability_recheck_scenarios,
            "stability_overrides": stability_overrides,
            "pairing_reference": pairing_source,
            "severe_throttling": severe_throttling,
        }
        if not reasons:
            eligible.append(candidate)

    common_cpu_scenarios = [
        name
        for name in resource_scenario_names
        if eligible
        and all(
            isinstance(item.get("cpu_efficiency_by_scenario"), Mapping)
            and isinstance(
                item["cpu_efficiency_by_scenario"].get(name),  # type: ignore[index,union-attr]
                (int, float),
            )
            and float(
                item["cpu_efficiency_by_scenario"][name]  # type: ignore[index]
            )
            > 0
            for item in eligible
        )
    ]
    if resource_scenario_names:
        for candidate in eligible:
            by_scenario = candidate["cpu_efficiency_by_scenario"]
            assert isinstance(by_scenario, Mapping)
            candidate["cpu_efficiency_scenarios_used"] = list(  # type: ignore[index]
                common_cpu_scenarios
            )
            candidate["cpu_efficiency_score"] = (  # type: ignore[index]
                _geometric_mean(
                    [float(by_scenario[name]) for name in common_cpu_scenarios]
                )
                if common_cpu_scenarios
                else None
            )
    cpu_coverage = {
        "requested_scenarios": list(resource_scenario_names),
        "used_scenarios": list(common_cpu_scenarios),
        "dropped_scenarios": [
            name
            for name in resource_scenario_names
            if name not in common_cpu_scenarios
        ],
        "mode": (
            "common_reliable_subset"
            if common_cpu_scenarios
            else (
                "performance_only_fallback"
                if resource_scenario_names
                else "legacy_aggregate"
            )
        ),
    }

    common_cpu = bool(eligible) and all(
        isinstance(item.get("cpu_efficiency_score"), (int, float))
        and float(item["cpu_efficiency_score"]) > 0
        for item in eligible
    )
    common_tail = bool(eligible) and all(
        isinstance(item.get("tail_score"), (int, float))
        and float(item["tail_score"]) > 0
        for item in eligible
    )
    common_throttle = bool(eligible) and all(
        _candidate_throttle(item) is not None for item in eligible
    )
    objective_names = ["performance"]
    if common_cpu:
        objective_names.append("cpu_efficiency")
    if common_tail:
        objective_names.append("tail_latency")
    if common_throttle:
        objective_names.append("cpu_throttling")
    decision_active = len(eligible) >= 2
    objective_details = {
        "performance": {
            "telemetry_available": bool(completed),
            "decision_active": decision_active,
        },
        "cpu_efficiency": {
            "telemetry_available": common_cpu,
            "decision_active": decision_active and common_cpu,
        },
        "tail_latency": {
            "telemetry_available": common_tail,
            "decision_active": decision_active and common_tail,
        },
        "cpu_throttling": {
            "telemetry_available": common_throttle,
            "decision_active": decision_active and common_throttle,
        },
        "topology_cost": {
            "telemetry_available": True,
            "decision_active": decision_active,
        },
    }
    selection_metadata = {
        "selection_schema_revision": 2,
        "selection_valid": baseline in eligible,
        "eligible_candidate_count": len(eligible),
        "comparable_candidates": sorted(str(item["name"]) for item in eligible),
        "objective_details": objective_details,
    }

    best_observed_score = max(
        float(item.get("score", 0.0)) for item in completed
    )
    observed_max_throughput = min(
        [
            item
            for item in completed
            if float(item.get("score", 0.0)) == best_observed_score
        ],
        key=lambda item: (
            _resource_rank(item["options"]),  # type: ignore[arg-type]
            str(item["name"]),
        ),
    )
    if eligible:
        best_eligible_score = max(
            float(item.get("score", 0.0)) for item in eligible
        )
        max_throughput_alternative = min(
            [
                item
                for item in eligible
                if float(item.get("score", 0.0)) == best_eligible_score
            ],
            key=lambda item: (
                _resource_rank(item["options"]),  # type: ignore[arg-type]
                str(item["name"]),
            ),
        )
    else:
        max_throughput_alternative = baseline
    efficiency_candidates = [
        item
        for item in eligible
        if isinstance(item.get("cpu_efficiency_score"), (int, float))
        and float(item["cpu_efficiency_score"]) > 0
    ]
    cpu_efficiency_alternative: Mapping[str, object] | None = None
    if efficiency_candidates:
        best_efficiency = max(
            float(item["cpu_efficiency_score"])
            for item in efficiency_candidates
        )
        cpu_efficiency_alternative = min(
            [
                item
                for item in efficiency_candidates
                if float(item["cpu_efficiency_score"])
                >= best_efficiency * (1.0 - _TIE_BAND)
            ],
            key=lambda item: (
                -float(item.get("score", 0.0)),
                _resource_rank(item["options"]),  # type: ignore[arg-type]
                str(item["name"]),
            ),
        )

    alternatives = {
        # Compatibility alias: before matrix v3.1 this was the only reported
        # maximum and always meant the fastest guardrail-eligible profile.
        "max_throughput": _selection_alternative(max_throughput_alternative),
        "eligible_max_throughput": (
            _selection_alternative(max_throughput_alternative)
            if eligible
            else _unavailable_selection_alternative(
                "no guardrail-eligible candidate was measured"
            )
        ),
        "observed_max_throughput": _selection_alternative(
            observed_max_throughput
        ),
        "cpu_efficiency": _selection_alternative(
            cpu_efficiency_alternative or baseline
        ),
    }

    if baseline not in eligible:
        return {
            "policy": "balanced-v1",
            "winner": str(baseline["name"]),
            "options": dict(baseline["options"]),  # type: ignore[arg-type]
            "score": float(baseline.get("score", 1.0)),
            "reason": "sequential baseline failed a selection guardrail",
            "objectives_used": ["performance"],
            **selection_metadata,
            "objective_details": {
                name: {
                    **details,
                    "decision_active": False,
                }
                for name, details in objective_details.items()
            },
            "cpu_resource_coverage": cpu_coverage,
            "pareto_objectives": ["performance", "topology_cost"],
            "pareto_frontier": [],
            "alternatives": alternatives,
        }

    for candidate in eligible:
        dominated_by: list[str] = []
        candidate_rank = _resource_rank(
            candidate["options"]  # type: ignore[arg-type]
        )
        candidate["topology_rank"] = list(candidate_rank)  # type: ignore[index]
        for other in eligible:
            if other is candidate:
                continue
            comparisons = [
                float(other.get("score", 0.0))
                >= float(candidate.get("score", 0.0))
            ]
            strict = [
                float(other.get("score", 0.0))
                > float(candidate.get("score", 0.0))
            ]
            other_rank = _resource_rank(
                other["options"]  # type: ignore[arg-type]
            )
            comparisons.append(other_rank <= candidate_rank)
            strict.append(other_rank < candidate_rank)
            if common_cpu:
                comparisons.append(
                    float(other["cpu_efficiency_score"])
                    >= float(candidate["cpu_efficiency_score"])
                )
                strict.append(
                    float(other["cpu_efficiency_score"])
                    > float(candidate["cpu_efficiency_score"])
                )
            if common_tail:
                comparisons.append(
                    float(other["tail_score"]) >= float(candidate["tail_score"])
                )
                strict.append(
                    float(other["tail_score"]) > float(candidate["tail_score"])
                )
            if common_throttle:
                other_throttle = _candidate_throttle(other)
                current_throttle = _candidate_throttle(candidate)
                assert other_throttle is not None and current_throttle is not None
                comparisons.extend(
                    [
                        other_throttle[0] <= current_throttle[0],
                        other_throttle[1] <= current_throttle[1],
                    ]
                )
                strict.extend(
                    [
                        other_throttle[0] < current_throttle[0],
                        other_throttle[1] < current_throttle[1],
                    ]
                )
            if all(comparisons) and any(strict):
                dominated_by.append(str(other["name"]))
        candidate["dominated_by"] = sorted(dominated_by)  # type: ignore[index]

    frontier = sorted(
        str(item["name"])
        for item in eligible
        if not item.get("dominated_by")
    )

    improved = [
        item
        for item in eligible
        if item is not baseline
        and float(item.get("score", 0.0))
        >= float(baseline.get("score", 1.0)) * (1.0 + _MIN_GAIN)
    ]
    if not improved:
        faster_observation_rejected = bool(
            observed_max_throughput is not baseline
            and float(observed_max_throughput.get("score", 0.0))
            >= float(baseline.get("score", 1.0)) * (1.0 + _MIN_GAIN)
            and not bool(
                observed_max_throughput.get("guardrails", {}).get("eligible")
                if isinstance(
                    observed_max_throughput.get("guardrails"), Mapping
                )
                else False
            )
        )
        result = {
            "policy": "balanced-v1",
            "winner": str(baseline["name"]),
            "options": dict(baseline["options"]),  # type: ignore[arg-type]
            "score": float(baseline.get("score", 1.0)),
            "reason": (
                "materially faster candidates failed selection guardrails"
                if faster_observation_rejected
                else "no candidate cleared the minimum 5% composite gain"
            ),
            "objectives_used": objective_names,
            **selection_metadata,
            "cpu_resource_coverage": cpu_coverage,
            "pareto_objectives": objective_names + ["topology_cost"],
            "pareto_frontier": frontier,
            "alternatives": alternatives,
        }
        return result

    best_score = max(float(item.get("score", 0.0)) for item in improved)
    near_best = [
        item
        for item in improved
        if float(item.get("score", 0.0))
        >= best_score * (1.0 - _BALANCED_PERFORMANCE_BAND)
    ]
    if common_throttle:
        best_throttle_class = min(_throttle_class(item) for item in near_best)
        near_best = [
            item
            for item in near_best
            if _throttle_class(item) == best_throttle_class
        ]
    if common_cpu:
        best_efficiency = max(
            float(item["cpu_efficiency_score"]) for item in near_best
        )
        near_best = [
            item
            for item in near_best
            if float(item["cpu_efficiency_score"])
            >= best_efficiency * (1.0 - _CPU_EFFICIENCY_BAND)
        ]
    remaining_best_score = max(
        float(item.get("score", 0.0)) for item in near_best
    )
    near_best = [
        item
        for item in near_best
        if float(item.get("score", 0.0))
        >= remaining_best_score * (1.0 - _TIE_BAND)
    ]
    selected = min(
        near_best,
        key=lambda item: (
            _resource_rank(item["options"]),  # type: ignore[arg-type]
            str(item["name"]),
        ),
    )
    return {
        "policy": "balanced-v1",
        "winner": str(selected["name"]),
        "options": dict(selected["options"]),  # type: ignore[arg-type]
        "score": float(selected.get("score", 0.0)),
        "reason": "balanced guarded performance/resource frontier",
        "objectives_used": objective_names,
        **selection_metadata,
        "cpu_resource_coverage": cpu_coverage,
        "pareto_objectives": objective_names + ["topology_cost"],
        "pareto_frontier": frontier,
        "alternatives": {
            **alternatives,
            "cpu_efficiency": _selection_alternative(
                cpu_efficiency_alternative or selected
            ),
        },
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


def _resource_stability_report(
    finalists: Sequence[Mapping[str, object]],
    *,
    baseline: Mapping[str, object],
    selected_name: str,
    effective_cpu: int,
) -> dict[str, object]:
    ambient_limit = max(0.25, 0.10 * effective_cpu)
    resource_measurements: list[Mapping[str, object]] = list(finalists)
    for item in finalists:
        pairing_reference = item.get("pairing_reference")
        if (
            isinstance(pairing_reference, Mapping)
            and isinstance(
                pairing_reference.get("resource_summary"), Mapping
            )
        ):
            resource_measurements.append(
                {
                    "name": f"{item['name']}:pairing_reference",
                    "resource_summary": pairing_reference["resource_summary"],
                }
            )
    contaminated = sorted(
        str(item["name"])
        for item in resource_measurements
        if (
            (ambient := _nested_number(
                item,
                "resource_summary",
                "cgroup",
                "ambient_cpu_cores",
            ))
            is not None
            and ambient > ambient_limit
        )
    )
    unattributable: list[str] = []
    for item in resource_measurements:
        summary = item.get("resource_summary")
        cgroup = (
            summary.get("cgroup")
            if isinstance(summary, Mapping)
            else None
        )
        if (
            not isinstance(cgroup, Mapping)
            or cgroup.get("available") is False
            or not any(
                key in cgroup
                for key in (
                    "attribution_confident",
                    "attributable",
                    "process_cpu_share",
                )
            )
        ):
            # Resource telemetry is optional.  Missing cgroup support disables
            # the resource objective elsewhere; it is not evidence of a
            # contaminated measurement by itself.
            continue
        share = _nested_number(cgroup, "process_cpu_share")
        if (
            cgroup.get("attribution_confident") is not True
            or cgroup.get("attributable") is not True
            or (share is not None and share < 0.80)
        ):
            unattributable.append(str(item["name"]))
    unattributable.sort()
    baseline_guardrails = baseline.get("guardrails")
    baseline_selection_passed = bool(
        baseline_guardrails.get("eligible")
        if isinstance(baseline_guardrails, Mapping)
        else False
    )
    reasons: list[str] = []
    if contaminated:
        reasons.append(
            "ambient cgroup activity affected finalist measurements: "
            + ", ".join(contaminated)
        )
    if unattributable:
        reasons.append(
            "resource attribution was insufficient for finalist measurements: "
            + ", ".join(unattributable)
        )
    if not baseline_selection_passed:
        reasons.append("the sequential baseline failed a selection guardrail")
    return {
        "passed": not reasons,
        "ambient_cpu_core_limit": ambient_limit,
        "contaminated_candidates": contaminated,
        "unattributable_candidates": unattributable,
        "baseline_selection_guardrails_passed": baseline_selection_passed,
        "selected_candidate": selected_name,
        "reason": None if not reasons else "; ".join(reasons),
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
        "resource_stability": None,
        "storage_stability": None,
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
        "source": "bounded_logical_index_locations",
    }


def _new_stage_report(stage: _StagePlan) -> dict[str, object]:
    return {
        "name": stage.name,
        "status": "pending",
        "warmup": {
            "status": "pending",
            "rounds": _WARMUP_ROUNDS,
            "blocks_total": _WARMUP_ROUNDS * len(stage.scenarios),
            "blocks_completed": 0,
            "rows": 0,
            "planned_read_bytes": 0,
        },
        "scenarios": [
            {
                "name": scenario.name,
                "cache_mode": scenario.cache_mode,
                "latency_guardrail": scenario.latency_guardrail,
                "resource_objective": scenario.resource_objective,
                "repetitions": [],
                "summary": None,
            }
            for scenario in stage.scenarios
        ],
        "resource_summary": None,
    }


def _run_stage(
    stage: _StagePlan,
    candidates: list[_Candidate],
    trials: dict[str, object],
    *,
    sample: np.ndarray,
    replacement_begin: int,
    seed: int,
    dimension: int,
    dtype: np.dtype,
    record_size: int,
    deadline: float,
    clock: Callable[[], float],
    progress: Callable[[str], None] | None,
    resource_probe: ResourceProbe | None,
    cache_preparer: CachePreparer | None = None,
    disjoint_pool: tuple[int, int] | None = None,
    hot_pool: tuple[int, int] | None = None,
) -> tuple[
    list[_Candidate],
    dict[str, dict[str, object]],
    int,
    bool,
    str,
    dict[str, object],
]:
    active = list(candidates)
    reports = {candidate.name: _new_stage_report(stage) for candidate in active}
    planned_read_bytes = 0
    query_stream_rows = _stage_ambient_query_rows(stage, len(active))
    if disjoint_pool is None:
        pool_begin, pool_end = replacement_begin, int(sample.size)
    else:
        pool_begin, pool_end = disjoint_pool
        if (
            pool_begin < replacement_begin
            or pool_end < pool_begin
            or pool_end > sample.size
        ):
            raise CalibrationError(f"invalid {stage.name} disjoint query pool")
    globally_disjoint = pool_end - pool_begin >= query_stream_rows
    permutation: np.ndarray | None = None
    starts: dict[tuple[str, int, int, str], int] = {}
    if globally_disjoint and query_stream_rows:
        available = pool_end - pool_begin
        permutation = np.random.default_rng(
            _derived_seed(seed, stage.name, "global-query-stream")
        ).permutation(available)
        permutation = permutation.astype(np.int64, copy=False)
        permutation += pool_begin
        cursor = 0
        for phase, rounds in (
            ("warmup", _WARMUP_ROUNDS),
            ("measured", stage.repeats),
        ):
            for round_index in range(rounds):
                for scenario_index, scenario in enumerate(stage.scenarios):
                    if scenario.reuse_across_rounds:
                        continue
                    for candidate in active:
                        starts[
                            (phase, round_index, scenario_index, candidate.name)
                        ] = cursor
                        cursor += scenario.total_rows
        if cursor > available:
            raise CalibrationError(
                f"{stage.name} disjoint query stream exceeded sampled IDs"
            )

    cache_control = (
        "ambient_global_disjoint_interleaved"
        if globally_disjoint
        else "ambient_replacement_warm_interleaved"
    )
    if any(item.reuse_across_rounds for item in stage.scenarios):
        cache_control += "+explicit_hot_reuse"
    cache_preparation: dict[str, object] = {
        "status": "not_requested",
        "method": None,
        "supported": None,
        "best_effort": True,
        "candidate_file_count": 0,
        "advised_file_count": 0,
        "data_file_count": 0,
        "wal_file_count": 0,
        "advised_bytes": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }

    def queries_for(
        phase: str,
        round_index: int,
        scenario_index: int,
        candidate: _Candidate,
    ) -> list[np.ndarray]:
        scenario = stage.scenarios[scenario_index]
        disjoint_indices: np.ndarray | None = None
        if permutation is not None and not scenario.reuse_across_rounds:
            begin = starts[
                (phase, round_index, scenario_index, candidate.name)
            ]
            disjoint_indices = permutation[begin : begin + scenario.total_rows]
        replacement_parts: tuple[object, ...]
        replacement_end: int | None = None
        if scenario.reuse_across_rounds:
            replacement_parts = (stage.name, "hot", scenario.name)
            if hot_pool is not None:
                replacement_start, replacement_end = hot_pool
            else:
                replacement_start = replacement_begin
        else:
            replacement_parts = (
                stage.name,
                phase,
                round_index,
                scenario.name,
                candidate.name,
            )
            replacement_start = replacement_begin
        return _queries_for_candidate(
            sample,
            disjoint_indices=disjoint_indices,
            replacement_begin=replacement_start,
            replacement_end=replacement_end,
            scenario=scenario,
            seed=_derived_seed(seed, *replacement_parts),
        )

    def fail_candidate(
        candidate: _Candidate,
        error: ReaderBackendError,
        *,
        phase: str,
        scenario: str,
    ) -> None:
        report = reports[candidate.name]
        report["status"] = "failed"
        report["error"] = str(error)[:512]
        report["failure"] = {
            "phase": phase,
            "scenario": scenario,
        }
        warmup = report["warmup"]
        if phase == "warmup" and warmup["status"] == "pending":  # type: ignore[index]
            warmup["status"] = "failed"  # type: ignore[index]
        if candidate.name in trials:
            _close_trial(trials.pop(candidate.name))
        active[:] = [item for item in active if item.name != candidate.name]
        if candidate.name == "sequential":
            raise CalibrationError(
                f"sequential baseline {stage.name} read failed"
            ) from error

    def exhausted_result() -> tuple[
        list[_Candidate],
        dict[str, dict[str, object]],
        int,
        bool,
        str,
        dict[str, object],
    ]:
        for report in reports.values():
            if report["status"] == "pending":
                report["status"] = "incomplete"
            warmup = report["warmup"]
            if warmup["status"] == "pending":  # type: ignore[index]
                warmup["status"] = "incomplete"  # type: ignore[index]
        return (
            active,
            reports,
            planned_read_bytes,
            True,
            cache_control,
            cache_preparation,
        )

    if cache_preparer is not None:
        if progress is not None:
            progress(f"{stage.name}: advising payload pages as unused")
        if clock() >= deadline:
            return exhausted_result()
        prepared = cache_preparer()
        if not isinstance(prepared, Mapping):
            raise CalibrationError(
                "calibration cache preparer returned an invalid report"
            )
        cache_preparation = dict(prepared)
        if clock() >= deadline:
            return exhausted_result()

    for warmup_round in range(_WARMUP_ROUNDS):
        for scenario_index, scenario in enumerate(stage.scenarios):
            if clock() >= deadline:
                return exhausted_result()
            offset = (warmup_round + scenario_index) % len(active)
            order = active[offset:] + active[:offset]
            for candidate in order:
                if candidate not in active:
                    continue
                if clock() >= deadline:
                    return exhausted_result()
                queries = queries_for(
                    "warmup", warmup_round, scenario_index, candidate
                )
                if progress is not None:
                    progress(
                        f"{stage.name} warm-up "
                        f"{warmup_round + 1}/{_WARMUP_ROUNDS}: "
                        f"{candidate.name} / {scenario.name}"
                    )
                if clock() >= deadline:
                    del queries
                    return exhausted_result()
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
                        resource_probe=None,
                    )
                except _DeadlineExpired:
                    del queries
                    return exhausted_result()
                except ReaderBackendError as error:
                    fail_candidate(
                        candidate,
                        error,
                        phase="warmup",
                        scenario=scenario.name,
                    )
                    del queries
                    continue
                warmup = reports[candidate.name]["warmup"]
                warmup["blocks_completed"] += 1  # type: ignore[index]
                warmup["rows"] += block_rows  # type: ignore[index]
                warmup["planned_read_bytes"] += block_bytes  # type: ignore[index]
                del queries

    expected_warmup_blocks = _WARMUP_ROUNDS * len(stage.scenarios)
    for candidate in active:
        warmup = reports[candidate.name]["warmup"]
        if warmup["blocks_completed"] != expected_warmup_blocks:  # type: ignore[index]
            raise CalibrationError(
                f"candidate {candidate.name} did not complete {stage.name} warm-up"
            )
        warmup["status"] = "complete"  # type: ignore[index]

    for repetition in range(stage.repeats):
        for scenario_index, scenario in enumerate(stage.scenarios):
            if clock() >= deadline:
                return exhausted_result()
            offset = (
                _WARMUP_ROUNDS + repetition + scenario_index
            ) % len(active)
            order = active[offset:] + active[:offset]
            for candidate in order:
                if candidate not in active:
                    continue
                queries = queries_for(
                    "measured", repetition, scenario_index, candidate
                )
                if progress is not None:
                    progress(
                        f"{stage.name} repeat {repetition + 1}/{stage.repeats}: "
                        f"{candidate.name} / {scenario.name}"
                    )
                if clock() >= deadline:
                    del queries
                    return exhausted_result()
                block_rows = sum(int(query.size) for query in queries)
                planned_read_bytes += block_rows * record_size
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
                        resource_probe=resource_probe,
                    )
                except _DeadlineExpired:
                    del queries
                    return exhausted_result()
                except ReaderBackendError as error:
                    fail_candidate(
                        candidate,
                        error,
                        phase="measured",
                        scenario=scenario.name,
                    )
                    del queries
                    continue
                scenario_report = reports[candidate.name]["scenarios"][  # type: ignore[index]
                    scenario_index
                ]
                scenario_report["repetitions"].append(measured)
                del queries

    if clock() >= deadline:
        return exhausted_result()
    for candidate in active:
        report = reports[candidate.name]
        resource_values: list[Mapping[str, object]] = []
        for scenario_report in report["scenarios"]:  # type: ignore[assignment]
            repetitions = scenario_report["repetitions"]
            if len(repetitions) != stage.repeats:
                report["status"] = "incomplete"
                break
            scenario_report["summary"] = _summarize_repetitions(repetitions)
            resource_values.extend(
                value["resources"]
                for value in repetitions
                if isinstance(value.get("resources"), Mapping)
            )
        else:
            report["status"] = "measured"
            if resource_values:
                report["resource_summary"] = _summarize_resources(
                    resource_values,
                    comparable_work=False,
                )
    return (
        active,
        reports,
        planned_read_bytes,
        False,
        cache_control,
        cache_preparation,
    )


def _merge_recheck_observations(
    verification_report: dict[str, object],
    recheck_report: Mapping[str, object],
) -> None:
    """Append one paired recheck observation to a verified candidate."""

    target_scenarios_value = verification_report.get("scenarios")
    extra_scenarios_value = recheck_report.get("scenarios")
    if (
        not isinstance(target_scenarios_value, Sequence)
        or not isinstance(extra_scenarios_value, Sequence)
    ):
        raise CalibrationError("invalid stability recheck report")
    target_by_name = {
        str(item["name"]): item
        for item in target_scenarios_value
        if isinstance(item, dict) and "name" in item
    }
    merged_names: list[str] = []
    for extra in extra_scenarios_value:
        if not isinstance(extra, Mapping):
            raise CalibrationError("invalid stability recheck scenario")
        name = str(extra.get("name", ""))
        target = target_by_name.get(name)
        repetitions = extra.get("repetitions")
        if target is None or not isinstance(repetitions, Sequence):
            raise CalibrationError(
                f"stability recheck returned an unexpected scenario {name}"
            )
        target_repetitions = target.get("repetitions")
        if not isinstance(target_repetitions, list):
            raise CalibrationError("invalid verification repetitions")
        target_repetitions.extend(repetitions)
        if all(
            isinstance(value, Mapping)
            and "p95_call_seconds" in value
            and "planned_read_bytes" in value
            for value in target_repetitions
        ):
            target["summary"] = _summarize_repetitions(target_repetitions)
        else:
            # Keep the merge helper usable with minimal synthetic reports used
            # by deterministic selection tests.  Native measurements always
            # take the full path above.
            previous_summary = target.get("summary")
            summary = (
                dict(previous_summary)
                if isinstance(previous_summary, Mapping)
                else {}
            )
            throughput = [
                float(value["throughput_rows_per_second"])
                for value in target_repetitions
                if isinstance(value, Mapping)
            ]
            mean = statistics.fmean(throughput)
            summary["median_throughput_rows_per_second"] = statistics.median(
                throughput
            )
            summary["throughput_cv"] = (
                statistics.pstdev(throughput) / mean
                if len(throughput) > 1 and mean
                else 0.0
            )
            target["summary"] = summary
        target["stability_rechecked"] = True
        merged_names.append(name)

    resource_values: list[Mapping[str, object]] = []
    for scenario in target_scenarios_value:
        if not isinstance(scenario, Mapping):
            continue
        repetitions = scenario.get("repetitions")
        if not isinstance(repetitions, Sequence):
            continue
        resource_values.extend(
            value["resources"]
            for value in repetitions
            if isinstance(value, Mapping)
            and isinstance(value.get("resources"), Mapping)
        )
    if resource_values:
        verification_report["resource_summary"] = _summarize_resources(
            resource_values,
            comparable_work=False,
        )
    verification_report["stability_recheck"] = {
        "name": recheck_report.get("name"),
        "status": recheck_report.get("status"),
        "warmup": recheck_report.get("warmup"),
        "scenarios": merged_names,
        "additional_repetitions": _STABILITY_RECHECK_REPEATS,
    }


def _workload_report(
    screening: _StagePlan,
    verification: _StagePlan,
    *,
    correctness_rows: int,
    requested_bound: int,
    cache_control: Mapping[str, str] | None = None,
    cache_sampling: Mapping[str, object] | None = None,
    cache_preparation: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    def stage_report(stage: _StagePlan) -> dict[str, object]:
        return {
            "name": stage.name,
            "repeats": stage.repeats,
            "warmup_rounds": _WARMUP_ROUNDS,
            "scenarios": [
                {
                    "name": scenario.name,
                    "batch_rows": scenario.batch_rows,
                    "calls_per_repeat": scenario.calls,
                    "clients": scenario.clients,
                    "cache_mode": scenario.cache_mode,
                    "latency_guardrail": scenario.latency_guardrail,
                    "resource_objective": scenario.resource_objective,
                    "reuse_across_rounds": scenario.reuse_across_rounds,
                    "query_rows": (
                        list(scenario.query_rows)
                        if scenario.query_rows is not None
                        else None
                    ),
                }
                for scenario in stage.scenarios
            ],
        }

    stage_cache_control = dict(cache_control or {})
    authoritative_cache_control = str(
        stage_cache_control.get(
            "verification",
            stage_cache_control.get("screening", "unknown"),
        )
    )
    # Preserve the V2 scalar for existing report consumers.  V3 exposes the
    # exact per-stage modes separately, including explicit hot-query reuse.
    legacy_cache_control = authoritative_cache_control.split("+", 1)[0]
    return {
        "matrix": "two_stage_v3",
        "adaptive_verification": {
            "max_backfill_candidates": _MAX_VERIFICATION_BACKFILL,
            "backfill_pairing": "fresh_interleaved_sequential",
            "backfill_stage_candidates_per_slot": 2,
            "stability_recheck_repeats": _STABILITY_RECHECK_REPEATS,
            "stability_cv_lower_exclusive": _MAX_THROUGHPUT_CV,
            "stability_cv_upper_inclusive": _MAX_RECHECK_THROUGHPUT_CV,
            "minimum_paired_gain": _MIN_GAIN,
            "included_in_requested_read_bound": True,
        },
        "cache_control": legacy_cache_control,
        "stage_cache_control": stage_cache_control,
        "cache_sampling": dict(cache_sampling or {}),
        "stage_cache_preparation": {
            str(name): dict(value)
            for name, value in (cache_preparation or {}).items()
        },
        # Legacy summary fields describe the authoritative verification stage.
        "repeats": verification.repeats,
        "warmup": {
            "rounds": _WARMUP_ROUNDS,
            "measured": False,
            "accounting": "charged_not_measured",
            "blocks_per_candidate": _WARMUP_ROUNDS
            * len(verification.scenarios),
            "rows_per_candidate": _WARMUP_ROUNDS
            * verification.rows_per_round,
        },
        "correctness_rows": correctness_rows,
        "requested_read_bound_bytes": requested_bound,
        "stages": [stage_report(screening), stage_report(verification)],
        "scenarios": [
            {
                "name": scenario.name,
                "batch_rows": scenario.batch_rows,
                "calls_per_repeat": scenario.calls,
                "clients": scenario.clients,
                "cache_mode": scenario.cache_mode,
                "latency_guardrail": scenario.latency_guardrail,
                "resource_objective": scenario.resource_objective,
                "query_rows": (
                    list(scenario.query_rows)
                    if scenario.query_rows is not None
                    else None
                ),
            }
            for scenario in verification.scenarios
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
    resource_probe: ResourceProbe | None = None,
    cache_preparer: CachePreparer | None = None,
    isolated_process: bool = False,
) -> dict[str, object]:
    """Measure a bounded V3 matrix against one stable native snapshot."""

    if capability_probe is None:
        capability_probe = lambda queue_depth, workers: native.reader_capabilities(
            io_uring_queue_depth=queue_depth,
            io_uring_workers=workers,
        )
    if resource_probe is None:
        resource_probe = ResourceProbe()

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
    effective_cpu = int(cpu["effective"])
    discovery: dict[str, object] = {
        "unavailable": unavailable,
        "cpu": cpu,
        "thread_pool_boundary_workers": [24, 32]
        if effective_cpu >= 8
        else [],
        "candidate_limit": _MAX_CANDIDATES,
        "finalist_limit": _MAX_FINALISTS,
    }
    if clock() >= deadline:
        return _budget_result(
            "cooperative deadline expired during candidate discovery",
            candidate_discovery=discovery,
        )

    reserved_workspace = _minimum_workspace_bytes(record_size, effective_cpu)
    minimum_memory = max(
        reserved_workspace + 16,
        _NATIVE_SAMPLE_FIXED_BYTES + _NATIVE_SAMPLE_BYTES_PER_ROW,
    )
    if max_memory_bytes < minimum_memory:
        raise ValueError(
            "max_memory_bytes is too small for the minimum calibration buffers; "
            f"at least {minimum_memory} bytes are required"
        )
    sample_rows = min(
        _MAX_SAMPLE_ROWS,
        max(1, (max_memory_bytes - reserved_workspace) // 16),
        max(
            1,
            (max_memory_bytes - _NATIVE_SAMPLE_FIXED_BYTES)
            // _NATIVE_SAMPLE_BYTES_PER_ROW,
        ),
    )
    sampled = dict(
        native.sample_ids(
            requested_count=sample_rows,
            seed=seed,
            max_inspected_slots=min(_MAX_INSPECTED_SLOTS, sample_rows * 8),
        )
    )
    sample_value = sampled.get("ids")
    if (
        not isinstance(sample_value, np.ndarray)
        or sample_value.ndim != 1
        or sample_value.dtype != np.dtype(np.uint64)
        or not sample_value.flags.c_contiguous
        or not sample_value.flags.aligned
    ):
        raise CalibrationError("native ID sampler returned an invalid array")
    inspected_slots = int(sampled.get("inspected_slots", -1))
    inspection_limit = min(_MAX_INSPECTED_SLOTS, sample_rows * 8)
    if (
        int(sampled.get("requested_count", -1)) != sample_rows
        or sample_value.size > sample_rows
        or inspected_slots < 0
        or inspected_slots > inspection_limit
    ):
        raise CalibrationError("native ID sampler exceeded its bounded contract")
    sample = np.require(sample_value, dtype=np.uint64, requirements=["C", "A"])
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

    screening, verification, correctness_rows, requested_bound = (
        _build_stage_plans(
            candidate_count=len(candidates),
            sample_rows=sample.size,
            record_size=record_size,
            effective_cpu=effective_cpu,
            max_read_bytes=max_read_bytes,
            max_memory_bytes=max_memory_bytes,
        )
    )
    workload = _workload_report(
        screening,
        verification,
        correctness_rows=correctness_rows,
        requested_bound=requested_bound,
    )
    if progress is not None:
        progress(
            f"sampled {sample.size:,} live IDs; screening {len(candidates)} "
            f"profiles across {len(screening.scenarios)} workloads, then "
            f"verifying at most {_MAX_FINALISTS} finalists across "
            f"{len(verification.scenarios)} workloads"
        )

    candidate_reports: list[dict[str, object]] = [
        {
            "name": candidate.name,
            "options": dict(candidate.options),
            "status": "pending",
            "correctness": None,
            "warmup": {"status": "not_run"},
            "scenarios": [],
            "screening": None,
            "verification": None,
            "resource_summary": None,
        }
        for candidate in candidates
    ]
    reports_by_name = {str(item["name"]): item for item in candidate_reports}
    planned_read_bytes = 0
    if clock() >= deadline:
        for report in candidate_reports:
            report["status"] = "incomplete"
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
    try:
        exhausted = False
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
                    or planned > correctness_bound
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
            except ReaderBackendError as error:
                if trial is not None:
                    _close_trial(trial)
                report["status"] = "unavailable"
                report["error"] = str(error)[:512]
            except BaseException:
                if trial is not None and candidate.name not in trials:
                    _close_trial(trial)
                raise

        if exhausted:
            for report in candidate_reports:
                if report["status"] in {"ready", "pending", "not_run"}:
                    report["status"] = "incomplete"
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
        del truth, truth_ids
        replacement_begin = (
            correctness_rows if sample.size > correctness_rows else 0
        )
        screen_pool_rows = _stage_ambient_query_rows(screening, len(active))
        initial_finalist_capacity = min(_MAX_FINALISTS, len(active))
        backfill_capacity = min(
            _MAX_VERIFICATION_BACKFILL,
            max(0, len(active) - initial_finalist_capacity),
        )
        verification_pool_rows = _stage_ambient_query_rows(
            verification,
            initial_finalist_capacity,
        )
        backfill_pool_rows = _stage_ambient_query_rows(
            verification,
            backfill_capacity * 2,
        )
        recheck_capacity = (
            initial_finalist_capacity + backfill_capacity
            if initial_finalist_capacity + backfill_capacity > 1
            else 0
        )
        recheck_bound = _StagePlan(
            "verification_recheck",
            verification.scenarios,
            _STABILITY_RECHECK_REPEATS,
        )
        recheck_pool_rows = _stage_ambient_query_rows(
            recheck_bound,
            recheck_capacity,
        )
        hot_pool_rows = sum(
            scenario.total_rows
            for scenario in verification.scenarios
            if scenario.reuse_across_rounds
        )
        available_query_rows = int(sample.size) - replacement_begin
        required_disjoint_rows = (
            screen_pool_rows
            + verification_pool_rows
            + backfill_pool_rows
            + recheck_pool_rows
            + hot_pool_rows
        )
        screen_disjoint_pool: tuple[int, int] | None = None
        verification_disjoint_pool: tuple[int, int] | None = None
        backfill_disjoint_pool: tuple[int, int] | None = None
        recheck_disjoint_pool: tuple[int, int] | None = None
        hot_query_pool: tuple[int, int] | None = None
        cross_stage_disjoint = available_query_rows >= required_disjoint_rows
        if cross_stage_disjoint:
            screen_end = replacement_begin + screen_pool_rows
            verification_end = screen_end + verification_pool_rows
            backfill_end = verification_end + backfill_pool_rows
            recheck_end = backfill_end + recheck_pool_rows
            hot_end = recheck_end + hot_pool_rows
            screen_disjoint_pool = (replacement_begin, screen_end)
            verification_disjoint_pool = (screen_end, verification_end)
            if backfill_pool_rows:
                backfill_disjoint_pool = (verification_end, backfill_end)
            if recheck_pool_rows:
                recheck_disjoint_pool = (backfill_end, recheck_end)
            if hot_pool_rows:
                hot_query_pool = (recheck_end, hot_end)
        cache_sampling = {
            "cross_stage_ambient_disjoint": cross_stage_disjoint,
            "sample_rows_available": available_query_rows,
            "ambient_rows_required": (
                screen_pool_rows
                + verification_pool_rows
                + backfill_pool_rows
                + recheck_pool_rows
            ),
            "adaptive_backfill_rows_reserved": backfill_pool_rows,
            "adaptive_recheck_rows_reserved": recheck_pool_rows,
            "hot_rows_reserved": hot_pool_rows if cross_stage_disjoint else 0,
            "quality": (
                "ambient_cross_stage_disjoint"
                if cross_stage_disjoint
                else "replacement_or_cross_stage_reuse"
            ),
        }
        discovery["cache_sampling"] = cache_sampling

        (
            screen_active,
            screen_reports,
            screen_bytes,
            exhausted,
            screen_cache,
            screen_preparation,
        ) = _run_stage(
            screening,
            active,
            trials,
            sample=sample,
            replacement_begin=replacement_begin,
            seed=seed,
            dimension=dimension,
            dtype=dtype,
            record_size=record_size,
            deadline=deadline,
            clock=clock,
            progress=progress,
            resource_probe=resource_probe,
            cache_preparer=cache_preparer,
            disjoint_pool=screen_disjoint_pool,
        )
        planned_read_bytes += screen_bytes
        for candidate in active:
            report = reports_by_name[candidate.name]
            stage_report = screen_reports.get(candidate.name)
            if stage_report is None:
                continue
            report["screening"] = stage_report
            report["status"] = stage_report["status"]
            report["warmup"] = stage_report["warmup"]
            report["scenarios"] = stage_report["scenarios"]
            report["resource_summary"] = stage_report["resource_summary"]
        workload = _workload_report(
            screening,
            verification,
            correctness_rows=correctness_rows,
            requested_bound=requested_bound,
            cache_control={"screening": screen_cache},
            cache_sampling=cache_sampling,
            cache_preparation={"screening": screen_preparation},
        )
        if exhausted:
            return _budget_result(
                "cooperative deadline expired during screening",
                candidate_discovery=discovery,
                sample=sample_report,
                workload=workload,
                candidates=candidate_reports,
                planned_read_bytes=planned_read_bytes,
            )
        screen_scenarios = [item.name for item in screening.scenarios]
        screen_candidates = [
            reports_by_name[item.name]
            for item in screen_active
            if reports_by_name[item.name]["status"] == "measured"
        ]
        provisional = select_winner(screen_candidates, screen_scenarios)
        ranked = sorted(
            [
                item
                for item in screen_candidates
                if item["name"] != "sequential"
            ],
            key=lambda item: (
                -float(item.get("score", 0.0)),
                _resource_rank(item["options"]),  # type: ignore[arg-type]
                str(item["name"]),
            ),
        )
        finalist_order = ["sequential"]

        def add_finalist(value: object) -> None:
            if (
                isinstance(value, str)
                and value not in finalist_order
                and len(finalist_order) < _MAX_FINALISTS
            ):
                finalist_order.append(value)

        # Screening only has two repeats.  Its variance guardrails describe
        # uncertainty but must not permanently eliminate a potentially fast
        # profile before the longer verification stage.
        if ranked:
            add_finalist(ranked[0]["name"])
        add_finalist(provisional.get("winner"))
        efficient = provisional.get("alternatives", {}).get("cpu_efficiency")
        if isinstance(efficient, Mapping) and isinstance(
            efficient.get("winner"), str
        ):
            add_finalist(efficient["winner"])
        resource_finalist = _screening_resource_probe(ranked, provisional)
        if resource_finalist is not None:
            add_finalist(resource_finalist.get("name"))
        thread_pool_anchors = [
            item
            for item in screen_candidates
            if item["name"] != "sequential"
            and item["options"]["io_backend"] == "thread_pool"  # type: ignore[index]
            and int(item["options"]["io_workers"]) > 0  # type: ignore[index]
        ]
        if thread_pool_anchors:
            resource_anchor = min(
                thread_pool_anchors,
                key=lambda item: (
                    abs(
                        int(item["options"]["io_workers"])  # type: ignore[index]
                        - effective_cpu
                    ),
                    _resource_rank(item["options"]),  # type: ignore[arg-type]
                    str(item["name"]),
                ),
            )
            add_finalist(resource_anchor["name"])
        max_throughput = provisional.get("alternatives", {}).get(
            "max_throughput"
        )
        if isinstance(max_throughput, Mapping) and isinstance(
            max_throughput.get("winner"), str
        ):
            add_finalist(max_throughput["winner"])
        for item in ranked:
            if len(finalist_order) >= _MAX_FINALISTS:
                break
            add_finalist(item["name"])
        finalist_names = set(finalist_order)
        finalists = [
            item for item in screen_active if item.name in finalist_names
        ]
        if not finalists or finalists[0].name != "sequential":
            raise CalibrationError("screening did not retain the baseline")
        discovery["screening_selection"] = provisional
        discovery["screening_resource_probe"] = (
            str(resource_finalist["name"])
            if resource_finalist is not None
            else None
        )
        discovery["screening_uncertain_candidates"] = sorted(
            str(item["name"])
            for item in screen_candidates
            if not bool(item.get("guardrails", {}).get("eligible"))
        )
        discovery["finalists"] = [item.name for item in finalists]
        for candidate in screen_active:
            if candidate.name in finalist_names:
                continue
            if candidate.name in trials:
                _close_trial(trials.pop(candidate.name))
            reports_by_name[candidate.name]["status"] = "screened_out"
            reports_by_name[candidate.name]["verification"] = {
                "name": verification.name,
                "status": "screened_out",
                "reason": "not selected by the bounded screening stage",
            }

        (
            verified_active,
            verification_reports,
            verification_bytes,
            exhausted,
            verification_cache,
            verification_preparation,
        ) = _run_stage(
            verification,
            finalists,
            trials,
            sample=sample,
            replacement_begin=replacement_begin,
            seed=seed,
            dimension=dimension,
            dtype=dtype,
            record_size=record_size,
            deadline=deadline,
            clock=clock,
            progress=progress,
            resource_probe=resource_probe,
            cache_preparer=cache_preparer,
            disjoint_pool=verification_disjoint_pool,
            hot_pool=hot_query_pool,
        )
        planned_read_bytes += verification_bytes
        for candidate in finalists:
            report = reports_by_name[candidate.name]
            stage_report = verification_reports.get(candidate.name)
            if stage_report is None:
                continue
            report["verification"] = stage_report
            report["status"] = stage_report["status"]
            report["warmup"] = stage_report["warmup"]
            report["scenarios"] = stage_report["scenarios"]
            report["resource_summary"] = stage_report["resource_summary"]
        workload = _workload_report(
            screening,
            verification,
            correctness_rows=correctness_rows,
            requested_bound=requested_bound,
            cache_control={
                "screening": screen_cache,
                "verification": verification_cache,
            },
            cache_sampling=cache_sampling,
            cache_preparation={
                "screening": screen_preparation,
                "verification": verification_preparation,
            },
        )
        if exhausted:
            return _budget_result(
                "cooperative deadline expired during finalist verification",
                candidate_discovery=discovery,
                sample=sample_report,
                workload=workload,
                candidates=candidate_reports,
                planned_read_bytes=planned_read_bytes,
            )

        finalist_reports = [
            reports_by_name[item.name]
            for item in verified_active
            if reports_by_name[item.name]["status"] == "measured"
        ]
        scenario_names = [item.name for item in verification.scenarios]
        selection = select_winner(finalist_reports, scenario_names)
        cache_controls = {
            "screening": screen_cache,
            "verification": verification_cache,
        }
        cache_preparations = {
            "screening": screen_preparation,
            "verification": verification_preparation,
        }

        # If initial verification would retain sequential, spend one
        # pre-budgeted slot on the best still-unverified profile that did pass
        # screening.  This prevents an unlucky top-five screening order from
        # collapsing the result to the baseline.
        backfill_report: dict[str, object] = {
            "attempted": False,
            "candidate": None,
            "status": "not_needed",
            "maximum_candidates": _MAX_VERIFICATION_BACKFILL,
        }
        if (
            selection.get("winner") == "sequential"
            and selection.get("selection_valid") is True
            and backfill_capacity > 0
        ):
            backfill_pool = sorted(
                [
                    item
                    for item in candidate_reports
                    if item.get("status") == "screened_out"
                    and isinstance(item.get("guardrails"), Mapping)
                    and item["guardrails"].get("eligible") is True  # type: ignore[index,union-attr]
                ],
                key=lambda item: (
                    -float(item.get("score", 0.0)),
                    _resource_rank(item["options"]),  # type: ignore[arg-type]
                    str(item["name"]),
                ),
            )
            if backfill_pool:
                selected_backfill = backfill_pool[0]
                backfill_name = str(selected_backfill["name"])
                candidate_by_name = {item.name: item for item in candidates}
                backfill_candidate = candidate_by_name[backfill_name]
                backfill_report.update(
                    {
                        "attempted": True,
                        "candidate": backfill_name,
                        "status": "pending",
                    }
                )
                try:
                    trials[backfill_name] = native.reader_trial(
                        **backfill_candidate.options
                    )
                except ReaderBackendError as error:
                    selected_backfill["status"] = "unavailable"
                    selected_backfill["error"] = str(error)[:512]
                    selected_backfill["verification"] = {
                        "name": verification.name,
                        "status": "unavailable",
                        "error": str(error)[:512],
                    }
                    backfill_report["status"] = "unavailable"
                    backfill_report["error"] = str(error)[:512]
                else:
                    (
                        backfill_active,
                        backfill_reports,
                        backfill_bytes,
                        exhausted,
                        backfill_cache,
                        backfill_preparation,
                    ) = _run_stage(
                        verification,
                        [candidate_by_name["sequential"], backfill_candidate],
                        trials,
                        sample=sample,
                        replacement_begin=replacement_begin,
                        seed=seed,
                        dimension=dimension,
                        dtype=dtype,
                        record_size=record_size,
                        deadline=deadline,
                        clock=clock,
                        progress=progress,
                        resource_probe=resource_probe,
                        cache_preparer=cache_preparer,
                        disjoint_pool=backfill_disjoint_pool,
                        hot_pool=hot_query_pool,
                    )
                    planned_read_bytes += backfill_bytes
                    cache_controls["verification_backfill"] = backfill_cache
                    cache_preparations[
                        "verification_backfill"
                    ] = backfill_preparation
                    stage_report = backfill_reports.get(backfill_name)
                    paired_baseline_report = backfill_reports.get("sequential")
                    if stage_report is not None:
                        selected_backfill["verification"] = stage_report
                        selected_backfill["status"] = stage_report["status"]
                        selected_backfill["warmup"] = stage_report["warmup"]
                        selected_backfill["scenarios"] = stage_report["scenarios"]
                        selected_backfill["resource_summary"] = stage_report[
                            "resource_summary"
                        ]
                        backfill_report["status"] = stage_report["status"]
                    if (
                        stage_report is not None
                        and stage_report.get("status") == "measured"
                        and paired_baseline_report is not None
                        and paired_baseline_report.get("status") == "measured"
                    ):
                        selected_backfill["pairing_reference"] = {
                            "source": "verification_backfill_sequential",
                            "candidate": "sequential",
                            "stage": verification.name,
                            "warmup": paired_baseline_report["warmup"],
                            "scenarios": paired_baseline_report["scenarios"],
                            "resource_summary": paired_baseline_report[
                                "resource_summary"
                            ],
                        }
                        backfill_report["pairing_reference"] = {
                            "candidate": "sequential",
                            "source": "same_interleaved_stage",
                            "status": "measured",
                        }
                    workload = _workload_report(
                        screening,
                        verification,
                        correctness_rows=correctness_rows,
                        requested_bound=requested_bound,
                        cache_control=cache_controls,
                        cache_sampling=cache_sampling,
                        cache_preparation=cache_preparations,
                    )
                    if exhausted:
                        discovery["verification_backfill"] = backfill_report
                        return _budget_result(
                            "cooperative deadline expired during verification backfill",
                            candidate_discovery=discovery,
                            sample=sample_report,
                            workload=workload,
                            candidates=candidate_reports,
                            planned_read_bytes=planned_read_bytes,
                        )
                    if (
                        {item.name for item in backfill_active}
                        == {"sequential", backfill_name}
                        and selected_backfill.get("status") == "measured"
                        and isinstance(
                            selected_backfill.get("pairing_reference"), Mapping
                        )
                    ):
                        finalist_reports.append(selected_backfill)
                        selection = select_winner(
                            finalist_reports,
                            scenario_names,
                        )
        discovery["verification_backfill"] = backfill_report

        # A candidate whose only failing guardrails are 10-20% throughput
        # variance, while every paired gain remains at least +5%, gets one
        # additional paired observation.  Recheck all current finalists for
        # those scenarios so repetition vectors stay pairwise aligned.
        recheck_scenarios: set[str] = set()
        recheck_trigger_candidates: list[str] = []
        for report in finalist_reports:
            if report.get("name") == "sequential":
                continue
            guardrails = report.get("guardrails")
            if not isinstance(guardrails, Mapping):
                continue
            scenario_value = guardrails.get("stability_recheck_scenarios")
            reasons_value = guardrails.get("reasons")
            if (
                not isinstance(scenario_value, Sequence)
                or isinstance(scenario_value, (str, bytes))
                or not isinstance(reasons_value, Sequence)
                or isinstance(reasons_value, (str, bytes))
            ):
                continue
            names = {str(item) for item in scenario_value}
            stability_reasons = {
                reason
                for name in names
                for reason in (
                    f"{name} is unstable",
                    f"{name} paired gain is unstable",
                )
            }
            if names and all(
                str(reason) in stability_reasons for reason in reasons_value
            ):
                recheck_scenarios.update(names)
                recheck_trigger_candidates.append(str(report["name"]))

        stability_recheck_report: dict[str, object] = {
            "attempted": False,
            "trigger_candidates": sorted(recheck_trigger_candidates),
            "scenarios": sorted(recheck_scenarios),
            "additional_repetitions": _STABILITY_RECHECK_REPEATS,
            "status": "not_needed",
        }
        if recheck_scenarios:
            recheck_stage = _StagePlan(
                "verification_recheck",
                tuple(
                    item
                    for item in verification.scenarios
                    if item.name in recheck_scenarios
                ),
                _STABILITY_RECHECK_REPEATS,
            )
            candidate_by_name = {item.name: item for item in candidates}
            recheck_candidates = [
                candidate_by_name[str(item["name"])]
                for item in finalist_reports
            ]
            stability_recheck_report["attempted"] = True
            stability_recheck_report["status"] = "pending"
            (
                rechecked_active,
                recheck_reports,
                recheck_bytes,
                exhausted,
                recheck_cache,
                recheck_preparation,
            ) = _run_stage(
                recheck_stage,
                recheck_candidates,
                trials,
                sample=sample,
                replacement_begin=replacement_begin,
                seed=seed,
                dimension=dimension,
                dtype=dtype,
                record_size=record_size,
                deadline=deadline,
                clock=clock,
                progress=progress,
                resource_probe=resource_probe,
                cache_preparer=cache_preparer,
                disjoint_pool=recheck_disjoint_pool,
                hot_pool=hot_query_pool,
            )
            planned_read_bytes += recheck_bytes
            cache_controls["verification_recheck"] = recheck_cache
            cache_preparations[
                "verification_recheck"
            ] = recheck_preparation
            active_recheck_names = {item.name for item in rechecked_active}
            rechecked_baseline = recheck_reports.get("sequential")
            for candidate in recheck_candidates:
                report = reports_by_name[candidate.name]
                extra_report = recheck_reports.get(candidate.name)
                verification_report = report.get("verification")
                if extra_report is None:
                    continue
                if (
                    candidate.name in active_recheck_names
                    and extra_report.get("status") == "measured"
                    and isinstance(verification_report, dict)
                ):
                    _merge_recheck_observations(
                        verification_report,
                        extra_report,
                    )
                    pairing_reference = report.get("pairing_reference")
                    if (
                        candidate.name != "sequential"
                        and isinstance(pairing_reference, dict)
                        and isinstance(rechecked_baseline, Mapping)
                        and rechecked_baseline.get("status") == "measured"
                    ):
                        _merge_recheck_observations(
                            pairing_reference,
                            rechecked_baseline,
                        )
                    report["status"] = "measured"
                    report["warmup"] = verification_report["warmup"]
                    report["scenarios"] = verification_report["scenarios"]
                    report["resource_summary"] = verification_report[
                        "resource_summary"
                    ]
                else:
                    report["status"] = extra_report.get("status", "failed")
                    report["stability_recheck"] = extra_report
            workload = _workload_report(
                screening,
                verification,
                correctness_rows=correctness_rows,
                requested_bound=requested_bound,
                cache_control=cache_controls,
                cache_sampling=cache_sampling,
                cache_preparation=cache_preparations,
            )
            if exhausted:
                stability_recheck_report["status"] = "incomplete"
                discovery["stability_recheck"] = stability_recheck_report
                return _budget_result(
                    "cooperative deadline expired during stability recheck",
                    candidate_discovery=discovery,
                    sample=sample_report,
                    workload=workload,
                    candidates=candidate_reports,
                    planned_read_bytes=planned_read_bytes,
                )
            finalist_reports = [
                reports_by_name[item.name]
                for item in rechecked_active
                if reports_by_name[item.name]["status"] == "measured"
            ]
            selection = select_winner(finalist_reports, scenario_names)
            stability_recheck_report["status"] = "complete"
        discovery["stability_recheck"] = stability_recheck_report
        discovery["verified_candidates"] = [
            str(item["name"]) for item in finalist_reports
        ]
        discovery["final_comparison_candidates"] = list(
            selection.get("comparable_candidates", [])
        )

        baseline = reports_by_name["sequential"]
        drift = _baseline_drift(baseline, scenario_names)

        selected_name = str(selection["winner"])
        resource_stability = _resource_stability_report(
            finalist_reports,
            baseline=baseline,
            selected_name=selected_name,
            effective_cpu=effective_cpu,
        )
        storage_stability = _storage_stability_report(
            candidate_reports,
            finalist_reports,
            isolated_process=isolated_process,
        )
        status = (
            "complete"
            if (
                bool(drift["passed"])
                and bool(resource_stability["passed"])
                and selection.get("selection_valid") is True
            )
            else "unstable"
        )
        if planned_read_bytes > max_read_bytes:
            raise CalibrationError("calibration exceeded max_read_bytes")
        return {
            "format": _REPORT_FORMAT,
            "version": _REPORT_VERSION,
            "status": status,
            "reason": (
                "bounded matrix v3 completed"
                if status == "complete"
                else "calibration stability guardrail failed"
            ),
            "candidate_discovery": discovery,
            "sample": sample_report,
            "workload": workload,
            "candidates": candidate_reports,
            "stability": drift,
            "resource_stability": resource_stability,
            "selection": selection,
            "storage_stability": storage_stability,
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
