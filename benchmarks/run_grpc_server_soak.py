#!/usr/bin/env python3
"""Bounded regression/soak harness for one writable ThrowDB gRPC server.

This is intentionally separate from ``run_grpc_server_benchmark.py``.  The
benchmark measures controlled read performance; this harness exercises a
long-lived mixed workload, durable replay, and process restarts.  Every bound
is checked before starting new work.  The dominant request/response payload is
charged exactly; protobuf and HTTP/2 framing are disclosed as unaccounted
protocol overhead.

Mutation traffic uses IDs that the harness first proves absent.  Each cycle
returns those canaries to the absent state, but WAL/tombstone history remains,
so production acceptance must run on a disposable/rebuildable collection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import throwdb


# Reuse the existing privilege-free /proc, cgroup, atomic-report, and local
# launcher helpers without coupling this soak state machine to the performance
# benchmark's phase orchestration.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_grpc_server_benchmark as benchmark  # noqa: E402


REPORT_FORMAT = "throwdb-grpc-server-soak"
REPORT_REVISION = 2
DEFAULT_SEED = 20260717
GIB = 1 << 30
MIB = 1 << 20
UINT64_MAX = (1 << 64) - 1


class SoakConfigurationError(ValueError):
    pass


class SoakInvariantError(RuntimeError):
    pass


class _ExpectedStreamStop(RuntimeError):
    """Internal control-flow signal for a generation that is being stopped."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _uint64(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > UINT64_MAX:
        raise argparse.ArgumentTypeError("value must fit in uint64")
    return parsed


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a UUID") from error


def parse_sizes(value: str) -> tuple[int, ...]:
    return benchmark.parse_positive_int_list(value)


def deterministic_operation_id(
    seed: int,
    sequence: int,
    name: str,
    *,
    run_id: uuid.UUID = uuid.NAMESPACE_URL,
) -> uuid.UUID:
    """Return a stable UUID inside one globally unique logical soak run."""
    return uuid.uuid5(
        run_id,
        f"throwdb-soak-v1:{seed}:{sequence}:{name}",
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def remote_diagnostics(remote: Any) -> dict[str, Any]:
    description = remote.describe()
    return {
        "format_version": description.format_version,
        "dimension": description.schema.dimension,
        "dtype": description.schema.dtype,
        "count": description.count,
        "writable": description.writable,
        "durable_idempotency": description.durable_idempotency,
        "idempotency_domain": (
            str(description.idempotency_domain)
            if description.idempotency_domain is not None
            else None
        ),
        "reader": _json_value(description.reader),
        "tuning": _json_value(description.tuning),
        "availability": _json_value(description.availability),
        "admission_stats": _json_value(description.admission_stats),
        "engine_stats": _json_value(description.engine_stats),
    }


def process_snapshot(pid: int) -> dict[str, Any]:
    result = benchmark.read_process_snapshot(pid)
    fd_root = Path("/proc") / str(pid) / "fd"
    if fd_root.is_dir():
        try:
            result["fd_count"] = len(list(fd_root.iterdir()))
        except OSError:
            pass
    return result


class MetricsStream:
    """Append-only resource samples plus bounded in-memory aggregates."""

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float,
        pid_provider: Callable[[], tuple[int, int | None]],
    ) -> None:
        self._path = path
        self._interval = interval_seconds
        self._pid_provider = pid_provider
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._peaks: dict[str, int] = {}
        self._first: dict[str, dict[str, Any]] = {}
        self._last: dict[str, dict[str, Any]] = {}
        self._cgroup_first = benchmark.load_test.cgroup_cpu_snapshot()
        self._cgroup_last = dict(self._cgroup_first)
        self._diagnostic_errors = 0
        self._monitor_errors = 0
        self._monitor_error_samples: list[dict[str, str]] = []

    def start(self) -> None:
        self.emit("monitor_start")
        self._thread = threading.Thread(
            target=self._run,
            name="throwdb-soak-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.emit("periodic")
            except Exception as error:
                self._record_monitor_error("periodic", error)
                return

    def _record_monitor_error(self, event: str, error: BaseException) -> None:
        with self._lock:
            self._monitor_errors += 1
            if len(self._monitor_error_samples) < 10:
                self._monitor_error_samples.append(
                    {
                        "event": event,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )

    def emit(self, event: str, remote: Any | None = None) -> None:
        generation, server_pid = self._pid_provider()
        now_ns = time.time_ns()
        client = process_snapshot(os.getpid())
        server = (
            process_snapshot(server_pid)
            if server_pid is not None
            else {"available": False, "pid": None}
        )
        cgroup = benchmark.load_test.cgroup_cpu_snapshot()
        row: dict[str, Any] = {
            "event": event,
            "unix_ns": now_ns,
            "server_generation": generation,
            "client_process": client,
            "server_process": server,
            "cgroup_cpu": cgroup,
        }
        if remote is not None:
            try:
                row["health"] = remote.health()
                row["server"] = remote_diagnostics(remote)
            except Exception as error:  # diagnostics must not hide workload result
                with self._lock:
                    self._diagnostic_errors += 1
                row["diagnostic_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        encoded = json.dumps(_json_value(row), sort_keys=True, allow_nan=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as output:
                output.write(encoded + "\n")
            self._sample_count += 1
            self._cgroup_last = cgroup
            for scope, snapshot in (("client", client), ("server", server)):
                if not snapshot.get("available"):
                    continue
                identity = (
                    scope
                    if scope == "client"
                    else f"server_generation_{generation}"
                )
                point = {
                    "unix_ns": now_ns,
                    "rss_bytes": snapshot.get("rss_bytes"),
                    "fd_count": snapshot.get("fd_count"),
                    "threads": snapshot.get("threads"),
                }
                self._first.setdefault(identity, point)
                self._last[identity] = point
                for field in ("rss_bytes", "fd_count", "threads"):
                    value = snapshot.get(field)
                    if isinstance(value, int):
                        key = f"{identity}_{field}"
                        self._peaks[key] = max(self._peaks.get(key, 0), value)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self._interval))
        try:
            self.emit("monitor_stop")
        except Exception as error:
            self._record_monitor_error("monitor_stop", error)

    def summary(self) -> dict[str, Any]:
        slopes: dict[str, dict[str, float | int | None]] = {}
        for identity, first in self._first.items():
            last = self._last[identity]
            elapsed = (last["unix_ns"] - first["unix_ns"]) / 1_000_000_000
            first_rss = first.get("rss_bytes")
            last_rss = last.get("rss_bytes")
            growth = (
                last_rss - first_rss
                if isinstance(first_rss, int) and isinstance(last_rss, int)
                else None
            )
            slopes[identity] = {
                "elapsed_seconds": elapsed,
                "rss_growth_bytes": growth,
                "rss_growth_bytes_per_second": (
                    growth / elapsed if growth is not None and elapsed > 0 else None
                ),
                "fd_growth": (
                    last["fd_count"] - first["fd_count"]
                    if isinstance(first.get("fd_count"), int)
                    and isinstance(last.get("fd_count"), int)
                    else None
                ),
            }
        cgroup_delta = (
            benchmark.load_test.counter_delta(
                self._cgroup_first, self._cgroup_last
            )
            if self._cgroup_first and self._cgroup_last
            else {}
        )
        if "throttled_usec" in cgroup_delta:
            cgroup_delta["throttled_seconds"] = (
                cgroup_delta["throttled_usec"] / 1_000_000
            )
        return {
            "metrics_jsonl": str(self._path),
            "sample_count": self._sample_count,
            "peaks": dict(self._peaks),
            "growth": slopes,
            "cgroup_cpu_delta": cgroup_delta,
            "diagnostic_error_count": self._diagnostic_errors,
            "monitor_error_count": self._monitor_errors,
            "monitor_error_samples": list(self._monitor_error_samples),
            "cgroup_scope": "shared cgroup; not attributable to ThrowDB alone",
        }


@dataclass(slots=True)
class _OperationAccumulator:
    attempts: int = 0
    successes: int = 0
    errors: int = 0
    rows: int = 0
    useful_bytes: int = 0
    latency_sum_ns: int = 0
    latency_max_ns: int = 0
    samples: dict[int, tuple[int, int, bool]] = field(default_factory=dict)


class OperationMetrics:
    """Thread-safe counters with a deterministic bounded latency sample."""

    def __init__(
        self,
        max_samples_per_operation: int,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._maximum = max_samples_per_operation
        self._seed = seed
        self._lock = threading.Lock()
        self._operations: dict[str, _OperationAccumulator] = {}

    def record(
        self,
        name: str,
        ordinal: int,
        latency_ns: int,
        *,
        rows: int,
        useful_bytes: int,
        success: bool,
    ) -> None:
        with self._lock:
            item = self._operations.setdefault(name, _OperationAccumulator())
            item.attempts += 1
            item.rows += rows
            item.useful_bytes += useful_bytes
            item.latency_sum_ns += latency_ns
            item.latency_max_ns = max(item.latency_max_ns, latency_ns)
            if success:
                item.successes += 1
            else:
                item.errors += 1
            priority = int.from_bytes(
                hashlib.blake2b(
                    f"{self._seed}:{name}:{ordinal}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
                "little",
            )
            item.samples[ordinal] = (priority, latency_ns, success)
            if len(item.samples) > self._maximum:
                evicted = max(
                    item.samples,
                    key=lambda key: (item.samples[key][0], key),
                )
                del item.samples[evicted]

    def summary(self, elapsed_seconds: float) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self._lock:
            snapshot = dict(self._operations)
        for name, item in sorted(snapshot.items()):
            values = np.asarray(
                [item.samples[key][1] for key in sorted(item.samples)],
                dtype=np.int64,
            )
            successful_values = np.asarray(
                [
                    item.samples[key][1]
                    for key in sorted(item.samples)
                    if item.samples[key][2]
                ],
                dtype=np.int64,
            )
            latency = (
                benchmark.load_test.latency_metrics(values) if values.size else {}
            )
            successful_latency = (
                benchmark.load_test.latency_metrics(successful_values)
                if successful_values.size
                else {}
            )
            result[name] = {
                "attempts": item.attempts,
                "successes": item.successes,
                "errors": item.errors,
                "rows": item.rows,
                "useful_bytes": item.useful_bytes,
                "attempts_per_second": (
                    item.attempts / elapsed_seconds if elapsed_seconds else 0.0
                ),
                "mean_latency_ms": (
                    item.latency_sum_ns / item.attempts / 1_000_000
                    if item.attempts
                    else None
                ),
                "latency_max_all_ms": item.latency_max_ns / 1_000_000,
                "latency_sample_count": int(values.size),
                "successful_latency_sample_count": int(
                    successful_values.size
                ),
                "latency_sampling": (
                    f"deterministic hash-priority reservoir, max {self._maximum}"
                ),
                "latency_sample_ordinal_min": (
                    min(item.samples) if item.samples else None
                ),
                "latency_sample_ordinal_max": (
                    max(item.samples) if item.samples else None
                ),
                **latency,
                **{
                    key.replace("latency_", "successful_latency_", 1): value
                    for key, value in successful_latency.items()
                },
            }
        return result


class ErrorMetrics:
    def __init__(self, max_samples: int = 100) -> None:
        self._maximum = max_samples
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._samples: list[dict[str, Any]] = []

    def record(self, operation: str, error: BaseException) -> None:
        code = getattr(error, "code", None)
        key = f"{operation}:{code or type(error).__name__}"
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            if len(self._samples) < self._maximum:
                self._samples.append(
                    {
                        "operation": operation,
                        "type": type(error).__name__,
                        "code": code,
                        "message": str(error),
                    }
                )

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": sum(self._counts.values()),
                "counts": dict(sorted(self._counts.items())),
                "samples": list(self._samples),
                "samples_truncated": sum(self._counts.values())
                > len(self._samples),
            }


class MutationBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0
        self._lock = threading.Lock()

    def can_reserve(self, count: int) -> bool:
        with self._lock:
            return self.used + count <= self.maximum

    def reserve(self, count: int) -> None:
        with self._lock:
            if self.used + count > self.maximum:
                raise benchmark.BenchmarkBudgetExceeded(
                    "maximum unique mutation operation bound would be exceeded"
                )
            self.used += count

    def as_dict(self) -> dict[str, int]:
        return {"maximum_unique_operations": self.maximum, "used": self.used}


def can_schedule_periodic_mutation(
    budget: MutationBudget,
    *,
    remaining_restarts: int,
) -> bool:
    # Four UUIDs for the cycle, two for each not-yet-run restart probe, and
    # one globally available fresh emergency Delete UUID.
    return budget.can_reserve(4 + 2 * remaining_restarts + 1)


def payload_for(
    ids: np.ndarray,
    *,
    version: int,
    dimension: int,
    dtype: str,
    seed: int,
) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    row_bytes = dimension * itemsize
    output = np.empty((ids.size, row_bytes), dtype=np.uint8)
    for index, identifier in enumerate(ids):
        material = (
            int(identifier).to_bytes(8, "little")
            + version.to_bytes(8, "little")
            + seed.to_bytes(8, "little", signed=False)
        )
        digest = hashlib.sha256(material).digest()
        repeated = (digest * ((row_bytes + len(digest) - 1) // len(digest)))[
            :row_bytes
        ]
        output[index] = np.frombuffer(repeated, dtype=np.uint8)
    return output.view(np.dtype(dtype)).reshape(ids.size, dimension).copy()


def _wire_get(rows: int, prefix: int, itemsize: int) -> int:
    return rows * (8 + prefix * itemsize)


def _wire_scan_records(
    rows: int,
    prefix: int,
    itemsize: int,
    *,
    with_locations: bool,
) -> int:
    return rows * (
        8
        + prefix * itemsize
        + (throwdb.ID_LOCATION_DTYPE.itemsize if with_locations else 0)
    )


def _wire_put(rows: int, record_size: int) -> int:
    return rows * (8 + record_size)


def _set_timeout(remote: Any, budget: benchmark.ResourceBudget, timeout: float) -> None:
    remote._timeout = budget.rpc_timeout(timeout)  # type: ignore[attr-defined]


def _assert_idempotency_domain(
    remote: Any,
    expected: uuid.UUID,
    *,
    operation_id: uuid.UUID,
) -> None:
    current = remote.describe().idempotency_domain
    if current != expected:
        raise SoakInvariantError(
            "idempotency replay domain changed before retrying operation "
            f"{operation_id}: expected {expected}, observed {current}; "
            "refusing to reuse the operation UUID"
        )


def _assert_bits(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise SoakInvariantError(f"{context} returned wrong dtype/shape")
    if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
        raise SoakInvariantError(f"{context} returned non-bit-exact data")


def _expect_missing(
    remote: Any,
    ids: np.ndarray,
    *,
    deadline_monotonic: float | None = None,
) -> None:
    # A batched Get returns NOT_FOUND when *any* requested ID is missing, so it
    # cannot prove that every canary is absent. Check each row independently;
    # the dominant transferred bytes are unchanged from one equivalent batch.
    for identifier in np.asarray(ids, dtype=np.uint64):
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("canary absence proof exceeded its deadline")
            configured = getattr(remote, "_timeout", None)
            remote._timeout = max(  # type: ignore[attr-defined]
                0.001,
                min(
                    remaining,
                    configured if configured is not None else remaining,
                ),
            )
        try:
            remote.get(np.asarray([identifier], dtype=np.uint64))
        except throwdb.RemoteNotFoundError:
            continue
        raise SoakInvariantError(
            f"canary ID {int(identifier)} unexpectedly remained readable"
        )


def find_absent_canaries(
    remote: Any,
    *,
    start: int,
    count: int,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
) -> np.ndarray:
    selected: list[int] = []
    candidate = start
    attempts = 0
    while len(selected) < count:
        if attempts >= max(1024, count * 100):
            raise SoakInvariantError("could not find enough absent canary IDs")
        identifier = np.asarray([candidate], dtype=np.uint64)
        budget.reserve(_wire_get(1, remote.dimension, np.dtype(remote.dtype).itemsize))
        _set_timeout(remote, budget, rpc_timeout)
        try:
            remote.get(identifier)
        except throwdb.RemoteNotFoundError:
            selected.append(candidate)
        candidate = (candidate - 1) & UINT64_MAX
        attempts += 1
    return np.asarray(selected, dtype=np.uint64)


def _invoke_mutation_with_replay(
    *,
    remote: Any,
    name: str,
    operation_id: uuid.UUID,
    idempotency_domain: uuid.UUID,
    call: Callable[[], Any],
    rows: int,
    useful_bytes: int,
    ordinal_base: int,
    metrics: OperationMetrics,
    errors: ErrorMetrics,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
    primary_attempts: int,
) -> tuple[Any, Any, int]:
    primary: Any = None
    attempts = 0
    replay_domain = idempotency_domain
    for attempt in range(primary_attempts):
        attempts += 1
        _set_timeout(remote, budget, rpc_timeout)
        started = time.perf_counter_ns()
        try:
            primary = call()
        except Exception as error:
            elapsed = time.perf_counter_ns() - started
            metrics.record(
                f"mutation_{name}", ordinal_base + attempt, elapsed,
                rows=rows, useful_bytes=useful_bytes, success=False,
            )
            if not isinstance(error, throwdb.RemoteAbortedError):
                errors.record(f"mutation_{name}", error)
            if attempt + 1 == primary_attempts:
                raise
            if isinstance(error, throwdb.RemoteAbortedError):
                # ABORTED is the one mutation status guaranteed to have been
                # rejected before durable PREPARE. Refreshing Describe may
                # therefore move this same logical UUID into a newly rotated
                # replay domain without risking a duplicate commit.
                _set_timeout(remote, budget, rpc_timeout)
                refreshed = remote.describe().idempotency_domain
                if refreshed is None:
                    raise SoakInvariantError(
                        "ABORTED retry lost its durable idempotency domain"
                    ) from error
                replay_domain = refreshed
                time.sleep(min(0.05 * (2 ** attempt), 0.5))
                continue
            if isinstance(
                error,
                (throwdb.AmbiguousCommitError, TimeoutError, ConnectionError),
            ):
                _set_timeout(remote, budget, rpc_timeout)
                _assert_idempotency_domain(
                    remote,
                    replay_domain,
                    operation_id=operation_id,
                )
                continue
            # Semantic rejections, protocol/integrity failures, and arbitrary
            # local exceptions are definitive. Retrying them can hide a broken
            # workload or apply a request the caller did not intend.
            raise
        elapsed = time.perf_counter_ns() - started
        metrics.record(
            f"mutation_{name}", ordinal_base + attempt, elapsed,
            rows=rows, useful_bytes=useful_bytes, success=True,
        )
        break
    else:  # pragma: no cover - loop either succeeds or raises
        raise SoakInvariantError(f"mutation {name} had no primary result")

    # Even an immediately successful call is deliberately replayed with the
    # exact same UUID and bytes. This is part of the acceptance workload, not
    # an automatic client retry policy.
    _set_timeout(remote, budget, rpc_timeout)
    _assert_idempotency_domain(
        remote,
        replay_domain,
        operation_id=operation_id,
    )
    started = time.perf_counter_ns()
    try:
        replay = call()
    except Exception as error:
        elapsed = time.perf_counter_ns() - started
        metrics.record(
            f"mutation_{name}", ordinal_base + 50, elapsed,
            rows=rows, useful_bytes=useful_bytes, success=False,
        )
        errors.record(f"mutation_{name}_intentional_replay", error)
        raise
    elapsed = time.perf_counter_ns() - started
    metrics.record(
        f"mutation_{name}", ordinal_base + 50, elapsed,
        rows=rows, useful_bytes=useful_bytes, success=True,
    )
    return primary, replay, attempts + 1


def mutation_cycle_reserved_bytes(
    rows: int,
    record_size: int,
    primary_attempts: int,
) -> int:
    full = _wire_put(rows, record_size)
    get = rows * (8 + record_size)
    delete = rows * 9
    calls_per_operation = primary_attempts + 1  # attempts plus explicit replay
    # Three embedding mutations and one Delete may each use every primary
    # attempt before the explicit successful replay. Three value Gets plus one
    # missing-state Get verify the normal path. Reserve one emergency Delete
    # and another missing-state Get so failure cleanup remains inside the
    # declared transfer ceiling.
    return (
        3 * calls_per_operation * full
        + calls_per_operation * delete
        + 5 * get
        + delete
    )


def run_mutation_cycle(
    remote: Any,
    canaries: np.ndarray,
    *,
    sequence: int,
    seed: int,
    run_id: uuid.UUID = uuid.NAMESPACE_URL,
    budget: benchmark.ResourceBudget,
    mutation_budget: MutationBudget,
    metrics: OperationMetrics,
    errors: ErrorMetrics,
    rpc_timeout: float,
    primary_attempts: int,
) -> dict[str, Any]:
    # Keep one unique-operation slot available for a fresh emergency Delete.
    if not mutation_budget.can_reserve(5):
        return {"status": "skipped", "reason": "mutation_budget"}
    record_size = remote.dimension * np.dtype(remote.dtype).itemsize
    budget.reserve(
        mutation_cycle_reserved_bytes(
            canaries.size,
            record_size,
            primary_attempts,
        ),
        extra_memory_bytes=5 * canaries.size * record_size,
    )
    mutation_budget.reserve(4)
    payloads = {
        version: payload_for(
            canaries,
            version=version,
            dimension=remote.dimension,
            dtype=remote.dtype,
            seed=seed,
        )
        for version in (1, 2, 3)
    }
    _set_timeout(remote, budget, rpc_timeout)
    idempotency_domain = remote.describe().idempotency_domain
    if idempotency_domain is None:
        raise SoakConfigurationError(
            "durable mutation soak requires an idempotency replay domain"
        )
    operations: list[dict[str, Any]] = []
    restored = False
    try:
        insert_id = deterministic_operation_id(
            seed, sequence, "insert", run_id=run_id
        )
        _, _, attempts = _invoke_mutation_with_replay(
            remote=remote, name="insert", operation_id=insert_id,
            idempotency_domain=idempotency_domain,
            call=lambda: remote.insert(
                canaries, payloads[1], operation_id=insert_id
            ),
            rows=canaries.size, useful_bytes=_wire_put(canaries.size, record_size),
            ordinal_base=sequence * 1000 + 100, metrics=metrics, errors=errors,
            budget=budget, rpc_timeout=rpc_timeout,
            primary_attempts=primary_attempts,
        )
        operations.append(
            {
                "name": "insert",
                "operation_id": str(insert_id),
                "attempts": attempts,
            }
        )
        _set_timeout(remote, budget, rpc_timeout)
        _assert_bits(remote.get(canaries), payloads[1], "insert verification")

        update_id = deterministic_operation_id(
            seed, sequence, "update", run_id=run_id
        )
        _, _, attempts = _invoke_mutation_with_replay(
            remote=remote, name="update", operation_id=update_id,
            idempotency_domain=idempotency_domain,
            call=lambda: remote.update(
                canaries, payloads[2], operation_id=update_id
            ),
            rows=canaries.size, useful_bytes=_wire_put(canaries.size, record_size),
            ordinal_base=sequence * 1000 + 200, metrics=metrics, errors=errors,
            budget=budget, rpc_timeout=rpc_timeout,
            primary_attempts=primary_attempts,
        )
        operations.append(
            {
                "name": "update",
                "operation_id": str(update_id),
                "attempts": attempts,
            }
        )
        _set_timeout(remote, budget, rpc_timeout)
        _assert_bits(remote.get(canaries), payloads[2], "update verification")

        upsert_id = deterministic_operation_id(
            seed, sequence, "upsert", run_id=run_id
        )
        _, _, attempts = _invoke_mutation_with_replay(
            remote=remote, name="upsert", operation_id=upsert_id,
            idempotency_domain=idempotency_domain,
            call=lambda: remote.upsert(
                canaries, payloads[3], operation_id=upsert_id
            ),
            rows=canaries.size, useful_bytes=_wire_put(canaries.size, record_size),
            ordinal_base=sequence * 1000 + 300, metrics=metrics, errors=errors,
            budget=budget, rpc_timeout=rpc_timeout,
            primary_attempts=primary_attempts,
        )
        operations.append(
            {
                "name": "upsert",
                "operation_id": str(upsert_id),
                "attempts": attempts,
            }
        )
        _set_timeout(remote, budget, rpc_timeout)
        _assert_bits(remote.get(canaries), payloads[3], "upsert verification")

        delete_id = deterministic_operation_id(
            seed, sequence, "delete", run_id=run_id
        )
        primary, replay, attempts = _invoke_mutation_with_replay(
            remote=remote, name="delete", operation_id=delete_id,
            idempotency_domain=idempotency_domain,
            call=lambda: remote.delete(canaries, operation_id=delete_id),
            rows=canaries.size, useful_bytes=canaries.size * 9,
            ordinal_base=sequence * 1000 + 400, metrics=metrics, errors=errors,
            budget=budget, rpc_timeout=rpc_timeout,
            primary_attempts=primary_attempts,
        )
        if not bool(np.asarray(primary).all()) or not np.array_equal(primary, replay):
            raise SoakInvariantError("Delete replay returned a different result mask")
        operations.append(
            {
                "name": "delete",
                "operation_id": str(delete_id),
                "attempts": attempts,
            }
        )
        _set_timeout(remote, budget, rpc_timeout)
        _expect_missing(
            remote,
            canaries,
            deadline_monotonic=budget.deadline,
        )
        restored = True
    finally:
        if not restored:
            # Cleanup is outside measured acceptance and may use a small grace
            # timeout. It receives a fresh UUID because the preceding outcome
            # may be ambiguous; the final read verifies the absent state.
            cleanup_id = deterministic_operation_id(
                seed, sequence, "cleanup-delete", run_id=run_id
            )
            if mutation_budget.can_reserve(1):
                mutation_budget.reserve(1)
                try:
                    remote._timeout = min(  # type: ignore[attr-defined]
                        10.0, rpc_timeout
                    )
                    remote.delete(canaries, operation_id=cleanup_id)
                    _expect_missing(
                        remote,
                        canaries,
                        deadline_monotonic=(
                            time.monotonic() + min(10.0, rpc_timeout)
                        ),
                    )
                    restored = True
                except Exception as cleanup_error:
                    errors.record("mutation_cleanup", cleanup_error)
    return {
        "status": "passed" if restored else "failed",
        "sequence": sequence,
        "rows": int(canaries.size),
        "operations": operations,
        "logical_absent_state_restored": restored,
    }


@dataclass(frozen=True, slots=True)
class GetPlan:
    ordinal: int
    label: str
    ids: np.ndarray
    prefix: int
    wire_bytes: int
    shadow_expected: np.ndarray
    source_expected: np.ndarray | None = None


def execute_get(
    remote: Any,
    plan: GetPlan,
    *,
    dimension: int,
    itemsize: int,
    retries: int,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
    metrics: OperationMetrics,
    errors: ErrorMetrics,
) -> bool:
    for attempt in range(retries + 1):
        if attempt:
            try:
                budget.reserve(
                    plan.wire_bytes,
                    extra_memory_bytes=3 * plan.ids.size * plan.prefix * itemsize,
                )
            except benchmark.BenchmarkBudgetExceeded:
                return False
        try:
            _set_timeout(remote, budget, rpc_timeout)
        except benchmark.BenchmarkBudgetExceeded:
            return False
        started = time.perf_counter_ns()
        try:
            result = remote.get(
                plan.ids,
                prefix_dimensions=(
                    None if plan.prefix == dimension else plan.prefix
                ),
            )
            if result.shape != (plan.ids.size, plan.prefix):
                raise SoakInvariantError("Get returned an invalid shape")
            _assert_bits(
                result[:1],
                plan.shadow_expected,
                "mixed Get shadow canary",
            )
            if plan.source_expected is not None:
                _assert_bits(
                    result,
                    plan.source_expected,
                    "mixed Get source oracle",
                )
        except Exception as error:
            elapsed = time.perf_counter_ns() - started
            metrics.record(
                plan.label, plan.ordinal * 100 + attempt, elapsed,
                rows=plan.ids.size, useful_bytes=plan.wire_bytes,
                success=False,
            )
            errors.record(plan.label, error)
            # --max-errors is an availability allowance, not a correctness
            # waiver. Wrong bits/shapes, missing sampled IDs, DATA_LOSS,
            # protocol failures, and deterministic semantic rejections must
            # fail the run immediately. Only explicitly transient transport /
            # admission statuses may be retried or counted against the bound.
            retryable = isinstance(
                error,
                (
                    throwdb.RemoteUnavailableError,
                    throwdb.RemoteDeadlineExceededError,
                    throwdb.RemoteResourceExhaustedError,
                    throwdb.RemoteCancelledError,
                ),
            )
            if isinstance(error, SoakInvariantError) or not retryable:
                raise
            if attempt == retries:
                return False
            continue
        elapsed = time.perf_counter_ns() - started
        metrics.record(
            plan.label, plan.ordinal * 100 + attempt, elapsed,
            rows=plan.ids.size, useful_bytes=plan.wire_bytes,
            success=True,
        )
        return True
    return False


def execute_get_on_client(
    lock: threading.Lock,
    remote: Any,
    plan: GetPlan,
    **kwargs: Any,
) -> bool:
    # Each read client owns its mutable RPC deadline. The lock prevents two
    # executor tasks assigned to the same client from racing on `_timeout`
    # without serializing calls assigned to the other client slots.
    with lock:
        return execute_get(remote, plan, **kwargs)


@dataclass(frozen=True, slots=True)
class PopulationSample:
    ids: np.ndarray
    source_rows: np.ndarray | None


class RepeatedIds(Sequence[int]):
    """A fixed-size, lazy repetition of a bounded uint64 population."""

    def __init__(self, values: np.ndarray, count: int) -> None:
        normalized = np.asarray(values, dtype=np.uint64)
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError("repeated ID population must be non-empty and 1-D")
        if count <= 0 or count > sys.maxsize:
            raise ValueError("repeated ID count must fit a positive Py_ssize_t")
        self._values = normalized
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return [int(self._values[position % self._values.size])
                    for position in range(start, stop, step)]
        normalized = int(index)
        if normalized < 0:
            normalized += self._count
        if not 0 <= normalized < self._count:
            raise IndexError(index)
        return int(self._values[normalized % self._values.size])


@dataclass(frozen=True, slots=True)
class StreamingConfiguration:
    stream_get_workers: int
    scan_records_workers: int
    stream_get_rows: int
    scan_records_rows: int
    batch_rows: int
    max_batch_payload_bytes: int
    prefix: int
    attempt_seconds: float
    progress_timeout_seconds: float
    stop_timeout_seconds: float
    rpc_timeout_seconds: float
    mutation_rows: int
    with_locations: bool = True
    source_oracle: bool = False

    @property
    def worker_count(self) -> int:
        return self.stream_get_workers + self.scan_records_workers

    @property
    def enabled_modes(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.stream_get_workers:
            result.append("stream_get")
        if self.scan_records_workers:
            result.append("scan_records")
        return tuple(result)

    def workers_for(self, mode: str) -> int:
        if mode == "stream_get":
            return self.stream_get_workers
        if mode == "scan_records":
            return self.scan_records_workers
        raise ValueError(f"unsupported streaming mode: {mode}")


@dataclass(slots=True)
class _StreamingModeAccumulator:
    configured_workers: int
    workers_started: int = 0
    workers_joined: int = 0
    attempts_started: int = 0
    attempts_completed: int = 0
    attempts_rotated: int = 0
    attempts_cancelled: int = 0
    expected_restart_interruptions: int = 0
    budget_stops: int = 0
    unexpected_failures: int = 0
    batches: int = 0
    rows: int = 0
    useful_bytes: int = 0
    source_verified_rows: int = 0
    shadow_checks: int = 0
    worker_active_seconds: float = 0.0
    open_latency_ns: int = 0
    open_latency_max_ns: int = 0
    max_inter_batch_gap_seconds: float = 0.0
    first_snapshot: dict[str, Any] | None = None
    last_snapshot: dict[str, Any] | None = None
    snapshot_count: int = 0
    first_digest_blake2b_256: str | None = None
    last_digest_blake2b_256: str | None = None


class StreamingMetrics:
    """Bounded, thread-safe aggregate metrics for generation-scoped streams."""

    def __init__(
        self,
        config: StreamingConfiguration,
        operations: OperationMetrics,
    ) -> None:
        self._operations = operations
        self._lock = threading.Lock()
        self._modes = {
            mode: _StreamingModeAccumulator(config.workers_for(mode))
            for mode in config.enabled_modes
        }
        self._last_progress: dict[tuple[int, str, int], float] = {}
        self._ordinals: dict[str, int] = {}
        self._stream_get_reference_digest: str | None = None
        self._generation_events: list[dict[str, Any]] = []

    def worker_started(self, mode: str) -> None:
        with self._lock:
            self._modes[mode].workers_started += 1

    def worker_finished(self, mode: str, elapsed_seconds: float) -> None:
        with self._lock:
            item = self._modes[mode]
            item.workers_joined += 1
            item.worker_active_seconds += elapsed_seconds

    def attempt_opened(
        self,
        mode: str,
        worker_key: tuple[int, str, int],
        *,
        snapshot: Any,
        latency_ns: int,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            item = self._modes[mode]
            item.attempts_started += 1
            item.open_latency_ns += latency_ns
            item.open_latency_max_ns = max(item.open_latency_max_ns, latency_ns)
            encoded = _json_value(snapshot)
            if item.first_snapshot is None:
                item.first_snapshot = encoded
            item.last_snapshot = encoded
            item.snapshot_count += 1
            self._last_progress[worker_key] = now
            ordinal = self._ordinals.get(f"{mode}_open", 0)
            self._ordinals[f"{mode}_open"] = ordinal + 1
        self._operations.record(
            f"{mode}_open",
            ordinal,
            latency_ns,
            rows=0,
            useful_bytes=0,
            success=True,
        )

    def batch(
        self,
        mode: str,
        worker_key: tuple[int, str, int],
        *,
        latency_ns: int,
        rows: int,
        useful_bytes: int,
        source_verified_rows: int,
        shadow_checked: bool,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            item = self._modes[mode]
            previous = self._last_progress.get(worker_key)
            if previous is not None:
                item.max_inter_batch_gap_seconds = max(
                    item.max_inter_batch_gap_seconds,
                    now - previous,
                )
            self._last_progress[worker_key] = now
            item.batches += 1
            item.rows += rows
            item.useful_bytes += useful_bytes
            item.source_verified_rows += source_verified_rows
            item.shadow_checks += int(shadow_checked)
            ordinal = self._ordinals.get(f"{mode}_batch", 0)
            self._ordinals[f"{mode}_batch"] = ordinal + 1
        self._operations.record(
            f"{mode}_batch",
            ordinal,
            latency_ns,
            rows=rows,
            useful_bytes=useful_bytes,
            success=True,
        )

    def attempt_finished(
        self,
        mode: str,
        *,
        status: str,
        digest: str | None,
    ) -> None:
        with self._lock:
            item = self._modes[mode]
            if status == "completed":
                item.attempts_completed += 1
            elif status == "rotated":
                item.attempts_rotated += 1
            elif status == "cancelled":
                item.attempts_cancelled += 1
            elif status == "restart_interrupted":
                item.expected_restart_interruptions += 1
            elif status == "budget":
                item.budget_stops += 1
            else:
                raise ValueError(f"unsupported stream attempt status: {status}")
            if digest is not None:
                if item.first_digest_blake2b_256 is None:
                    item.first_digest_blake2b_256 = digest
                item.last_digest_blake2b_256 = digest
                if mode == "stream_get" and status == "completed":
                    if self._stream_get_reference_digest is None:
                        self._stream_get_reference_digest = digest
                    elif digest != self._stream_get_reference_digest:
                        raise SoakInvariantError(
                            "StreamGet returned different bytes for the stable "
                            "base-ID corpus across completed snapshots"
                        )

    def failure(self, mode: str) -> None:
        with self._lock:
            self._modes[mode].unexpected_failures += 1

    def progress_count(self, generation: int, mode: str) -> int:
        with self._lock:
            return sum(
                1
                for (observed_generation, observed_mode, _), timestamp
                in self._last_progress.items()
                if observed_generation == generation
                and observed_mode == mode
                and timestamp > 0
            )

    def record_generation(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            self._generation_events.append(dict(event))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            modes = {
                mode: _json_value(item)
                for mode, item in self._modes.items()
            }
            events = [dict(item) for item in self._generation_events]
        for item in modes.values():
            active = float(item["worker_active_seconds"])
            rows = int(item["rows"])
            useful = int(item["useful_bytes"])
            attempts = int(item["attempts_started"])
            item["rows_per_worker_active_second"] = (
                rows / active if active > 0 else 0.0
            )
            item["logical_mib_per_worker_active_second"] = (
                useful / MIB / active if active > 0 else 0.0
            )
            item["mean_open_latency_ms"] = (
                item["open_latency_ns"] / attempts / 1_000_000
                if attempts
                else None
            )
            item["max_open_latency_ms"] = (
                item["open_latency_max_ns"] / 1_000_000
            )
        return {"modes": modes, "generations": events}


@dataclass(slots=True)
class _StreamingGeneration:
    generation: int
    endpoint: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    budget_exhausted: threading.Event = field(default_factory=threading.Event)
    release_event: threading.Event = field(default_factory=threading.Event)
    expected_interruption: threading.Event = field(default_factory=threading.Event)
    expected_interruption_workers: set[tuple[str, int]] = field(
        default_factory=set
    )
    interrupted_workers: set[tuple[str, int]] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    clients: dict[tuple[str, int], Any] = field(default_factory=dict)
    streams: dict[tuple[str, int], Any] = field(default_factory=dict)
    threads: list[threading.Thread] = field(default_factory=list)
    failures: list[BaseException] = field(default_factory=list)
    progress_by_worker: dict[tuple[str, int], int] = field(
        default_factory=dict
    )
    last_progress_monotonic: dict[tuple[str, int], float] = field(
        default_factory=dict
    )
    started_monotonic: float = field(default_factory=time.monotonic)


class StreamingWorkload:
    """Own dedicated streaming clients and make restart boundaries explicit."""

    def __init__(
        self,
        *,
        config: StreamingConfiguration,
        population: np.ndarray,
        source_rows: np.ndarray | None,
        source_embeddings: np.ndarray | None,
        shadow_payload: np.ndarray,
        dimension: int,
        dtype: str,
        budget: benchmark.ResourceBudget,
        operations: OperationMetrics,
        errors: ErrorMetrics,
    ) -> None:
        self.config = config
        self._population = np.array(population, dtype=np.uint64, copy=True)
        self._source_rows = (
            np.array(source_rows, dtype=np.int64, copy=True)
            if source_rows is not None
            else None
        )
        self._source_embeddings = source_embeddings
        self._shadow_payload = shadow_payload
        self._dimension = dimension
        self._dtype = np.dtype(dtype)
        self._budget = budget
        self._operations = operations
        self._errors = errors
        self.metrics = StreamingMetrics(config, operations)
        self._current: _StreamingGeneration | None = None

    @property
    def enabled(self) -> bool:
        return self.config.worker_count > 0

    def _connect(self, endpoint: str) -> Any:
        remaining = max(1.0, self._budget.remaining_seconds())
        scan_timeout = min(
            self.config.attempt_seconds + self.config.rpc_timeout_seconds,
            remaining + self.config.stop_timeout_seconds,
        )
        return throwdb.RemoteCollection.connect(
            endpoint,
            timeout=min(self.config.rpc_timeout_seconds, remaining),
            scan_timeout=max(1.0, scan_timeout),
        )

    def _register_client(
        self,
        state: _StreamingGeneration,
        key: tuple[str, int],
        client: Any,
    ) -> None:
        with state.lock:
            if state.stop_event.is_set():
                client.close()
                raise _ExpectedStreamStop(
                    "stream generation stopped during connect"
                )
            state.clients[key] = client
            state.last_progress_monotonic[key] = time.monotonic()

    def _register_stream(
        self,
        state: _StreamingGeneration,
        key: tuple[str, int],
        stream: Any,
    ) -> None:
        with state.lock:
            if state.stop_event.is_set():
                stream.close()
                raise _ExpectedStreamStop(
                    "stream generation stopped during open"
                )
            state.streams[key] = stream
            state.last_progress_monotonic[key] = time.monotonic()

    @staticmethod
    def _unregister_stream(
        state: _StreamingGeneration,
        key: tuple[str, int],
    ) -> None:
        with state.lock:
            state.streams.pop(key, None)

    @staticmethod
    def _raise_if_stopped(state: _StreamingGeneration) -> None:
        if state.stop_event.is_set():
            raise _ExpectedStreamStop("stream generation is stopping")

    def _open_stream(
        self,
        state: _StreamingGeneration,
        key: tuple[str, int],
        factory: Callable[[], Any],
    ) -> tuple[Any, Any, int]:
        """Open one stream, tolerating asynchronous release of a prior lease."""

        started = time.perf_counter_ns()
        deadline = time.monotonic() + min(
            self.config.progress_timeout_seconds,
            max(0.001, self._budget.remaining_seconds()),
        )
        delay = 0.005
        while True:
            self._raise_if_stopped(state)
            stream: Any | None = None
            try:
                stream = factory()
                # Both remote streaming APIs receive and validate their opened
                # response during construction. Accessing snapshot makes that
                # contract explicit for fakes and future implementations.
                snapshot = stream.snapshot
                self._register_stream(state, key, stream)
                return (
                    stream,
                    snapshot,
                    time.perf_counter_ns() - started,
                )
            except throwdb.RemoteResourceExhaustedError:
                if stream is not None:
                    with contextlib.suppress(BaseException):
                        stream.close()
                self._raise_if_stopped(state)
                remaining = min(
                    deadline - time.monotonic(),
                    self._budget.remaining_seconds(),
                )
                if remaining <= 0:
                    raise
                # A cancelled partial stream releases its server-side admission
                # lease asynchronously. Retry only the fresh open; no partial
                # data or snapshot is ever resumed.
                if state.stop_event.wait(min(delay, remaining)):
                    raise _ExpectedStreamStop(
                        "stream generation stopped during admission retry"
                    )
                delay = min(0.1, delay * 2)
            except BaseException:
                if stream is not None:
                    with contextlib.suppress(BaseException):
                        stream.close()
                raise

    @staticmethod
    def _snapshot_digest_part(digest: Any, values: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(values)
        digest.update(memoryview(contiguous).cast("B"))

    def _expected_positions(self, offset: int, rows: int) -> np.ndarray:
        return (
            np.arange(offset, offset + rows, dtype=np.uint64)
            % np.uint64(self._population.size)
        ).astype(np.int64, copy=False)

    def _run_stream_get_attempt(
        self,
        state: _StreamingGeneration,
        key: tuple[str, int],
        client: Any,
        *,
        first_attempt: bool,
    ) -> str:
        self._raise_if_stopped(state)
        rows = self.config.stream_get_rows
        maximum_batch_rows = min(self.config.batch_rows, rows)
        maximum_batch_wire = _wire_get(
            maximum_batch_rows,
            self.config.prefix,
            self._dtype.itemsize,
        )

        def reserve_batch() -> None:
            self._budget.reserve(
                maximum_batch_wire,
                extra_memory_bytes=(
                    3 * self.config.max_batch_payload_bytes
                    * self.config.worker_count
                ),
            )

        # The open response may be followed immediately by a buffered first
        # batch, so reserve one complete negotiated-request batch before open.
        reserve_batch()
        source = RepeatedIds(self._population, rows)
        stream, snapshot, open_latency_ns = self._open_stream(
            state,
            key,
            lambda: client.stream_get(
                source,
                prefix_dimensions=self.config.prefix,
                batch_rows=self.config.batch_rows,
                max_batch_payload_bytes=self.config.max_batch_payload_bytes,
            ),
        )
        worker_key = (state.generation, key[0], key[1])
        self.metrics.attempt_opened(
            "stream_get",
            worker_key,
            snapshot=snapshot,
            latency_ns=open_latency_ns,
        )
        if first_attempt:
            while not state.release_event.wait(0.05):
                if state.stop_event.is_set():
                    return (
                        "restart_interrupted"
                        if state.expected_interruption.is_set()
                        else "cancelled"
                    )
            if state.stop_event.is_set():
                return (
                    "restart_interrupted"
                    if state.expected_interruption.is_set()
                    else "cancelled"
                )
        digest = hashlib.blake2b(digest_size=32)
        emitted = 0
        reserved_batch_available = True
        attempt_started = time.monotonic()
        status = "completed"
        try:
            with stream:
                iterator = iter(stream)
                while True:
                    if state.stop_event.is_set():
                        status = (
                            "restart_interrupted"
                            if state.expected_interruption.is_set()
                            else "cancelled"
                        )
                        break
                    if not reserved_batch_available and emitted < rows:
                        reserve_batch()
                        reserved_batch_available = True
                    next_started = time.perf_counter_ns()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        if state.stop_event.is_set() and emitted < rows:
                            status = (
                                "restart_interrupted"
                                if state.expected_interruption.is_set()
                                else "cancelled"
                            )
                        break
                    reserved_batch_available = False
                    latency_ns = time.perf_counter_ns() - next_started
                    rows_in_batch = int(batch.ids.size)
                    if batch.input_offset != emitted:
                        raise SoakInvariantError(
                            "StreamGet offsets are not contiguous"
                        )
                    positions = self._expected_positions(emitted, rows_in_batch)
                    expected_ids = self._population[positions]
                    if not np.array_equal(batch.ids, expected_ids):
                        raise SoakInvariantError(
                            "StreamGet changed requested ID ordering"
                        )
                    expected_shape = (rows_in_batch, self.config.prefix)
                    if (
                        batch.embeddings.shape != expected_shape
                        or batch.embeddings.dtype != self._dtype
                    ):
                        raise SoakInvariantError(
                            "StreamGet returned an invalid dtype or shape"
                        )
                    shadow_checked = emitted == 0
                    if shadow_checked:
                        _assert_bits(
                            batch.embeddings[:1],
                            self._shadow_payload[:, : self.config.prefix],
                            "StreamGet shadow row",
                        )
                    verified_rows = 0
                    if (
                        self._source_embeddings is not None
                        and self._source_rows is not None
                    ):
                        expected = np.asarray(
                            self._source_embeddings[
                                self._source_rows[positions],
                                : self.config.prefix,
                            ],
                            dtype=self._dtype,
                            order="C",
                        )
                        _assert_bits(
                            batch.embeddings,
                            expected,
                            "StreamGet source oracle",
                        )
                        verified_rows = rows_in_batch
                    self._snapshot_digest_part(digest, batch.ids)
                    self._snapshot_digest_part(digest, batch.embeddings)
                    useful = _wire_get(
                        rows_in_batch,
                        self.config.prefix,
                        self._dtype.itemsize,
                    )
                    emitted += rows_in_batch
                    self.metrics.batch(
                        "stream_get",
                        worker_key,
                        latency_ns=latency_ns,
                        rows=rows_in_batch,
                        useful_bytes=useful,
                        source_verified_rows=verified_rows,
                        shadow_checked=shadow_checked,
                    )
                    with state.lock:
                        state.progress_by_worker[key] = (
                            state.progress_by_worker.get(key, 0)
                            + rows_in_batch
                        )
                        state.last_progress_monotonic[key] = time.monotonic()
                    if (
                        time.monotonic() - attempt_started
                        >= self.config.attempt_seconds
                        and emitted < rows
                    ):
                        status = "rotated"
                        break
            if status == "completed":
                if emitted != rows or not stream.done:
                    raise SoakInvariantError(
                        "StreamGet ended before every requested row"
                    )
                return digest.hexdigest()
            return status
        finally:
            self._unregister_stream(state, key)

    def _run_scan_records_attempt(
        self,
        state: _StreamingGeneration,
        key: tuple[str, int],
        client: Any,
        *,
        first_attempt: bool,
    ) -> tuple[str, str]:
        self._raise_if_stopped(state)
        description = client.describe()
        if description.count <= 0:
            raise SoakInvariantError("ScanRecords requires a non-empty snapshot")
        planned_target_rows = min(
            self.config.scan_records_rows,
            description.count + 2 * self.config.mutation_rows,
        )
        possible_rows = min(
            description.count + 2 * self.config.mutation_rows,
            planned_target_rows + self.config.batch_rows - 1,
        )
        maximum_batch_rows = min(self.config.batch_rows, possible_rows)
        maximum_batch_wire = _wire_scan_records(
            maximum_batch_rows,
            self.config.prefix,
            self._dtype.itemsize,
            with_locations=self.config.with_locations,
        )

        def reserve_batch() -> None:
            self._budget.reserve(
                maximum_batch_wire,
                extra_memory_bytes=(
                    3 * self.config.max_batch_payload_bytes
                    * self.config.worker_count
                ),
            )

        # RemoteRecordScan may eagerly receive its first data frame while
        # opening, hence the first per-batch charge precedes the RPC.
        reserve_batch()
        stream, snapshot, open_latency_ns = self._open_stream(
            state,
            key,
            lambda: client.scan_records(
                prefix_dimensions=self.config.prefix,
                batch_rows=self.config.batch_rows,
                max_batch_payload_bytes=self.config.max_batch_payload_bytes,
                with_locations=self.config.with_locations,
            ),
        )
        target_rows = min(
            self.config.scan_records_rows,
            int(stream.snapshot_count),
        )
        if target_rows <= 0:
            raise SoakInvariantError("ScanRecords opened an empty snapshot")
        worker_key = (state.generation, key[0], key[1])
        self.metrics.attempt_opened(
            "scan_records",
            worker_key,
            snapshot=snapshot,
            latency_ns=open_latency_ns,
        )
        if first_attempt:
            while not state.release_event.wait(0.05):
                if state.stop_event.is_set():
                    return (
                        (
                            "restart_interrupted"
                            if state.expected_interruption.is_set()
                            else "cancelled"
                        ),
                        "",
                    )
            if state.stop_event.is_set():
                return (
                    (
                        "restart_interrupted"
                        if state.expected_interruption.is_set()
                        else "cancelled"
                    ),
                    "",
                )
        digest = hashlib.blake2b(digest_size=32)
        emitted = 0
        reserved_batch_available = True
        previous_location: int | None = None
        attempt_started = time.monotonic()
        status = "completed"
        try:
            with stream:
                iterator = iter(stream)
                while True:
                    if state.stop_event.is_set():
                        status = (
                            "restart_interrupted"
                            if state.expected_interruption.is_set()
                            else "cancelled"
                        )
                        break
                    if (
                        not reserved_batch_available
                        and emitted < target_rows
                    ):
                        reserve_batch()
                        reserved_batch_available = True
                    next_started = time.perf_counter_ns()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        if state.stop_event.is_set() and emitted < target_rows:
                            status = (
                                "restart_interrupted"
                                if state.expected_interruption.is_set()
                                else "cancelled"
                            )
                        break
                    reserved_batch_available = False
                    latency_ns = time.perf_counter_ns() - next_started
                    rows_in_batch = int(batch.ids.size)
                    if batch.emitted_offset != emitted:
                        raise SoakInvariantError(
                            "ScanRecords offsets are not contiguous"
                        )
                    if (
                        batch.embeddings.shape
                        != (rows_in_batch, self.config.prefix)
                        or batch.embeddings.dtype != self._dtype
                    ):
                        raise SoakInvariantError(
                            "ScanRecords returned an invalid dtype or shape"
                        )
                    locations = batch.locations
                    if self.config.with_locations:
                        if (
                            locations is None
                            or locations.dtype != throwdb.ID_LOCATION_DTYPE
                            or locations.shape != (rows_in_batch,)
                        ):
                            raise SoakInvariantError(
                                "ScanRecords omitted requested locations"
                            )
                        encoded_locations = (
                            locations["segment_id"].astype(np.uint64) << 32
                        ) | locations["row_index"].astype(np.uint64)
                        if encoded_locations.size:
                            if (
                                previous_location is not None
                                and int(encoded_locations[0])
                                <= previous_location
                            ) or (
                                encoded_locations.size > 1
                                and bool(np.any(
                                    encoded_locations[1:]
                                    <= encoded_locations[:-1]
                                ))
                            ):
                                raise SoakInvariantError(
                                    "ScanRecords locations are not strictly "
                                    "in physical order"
                                )
                            previous_location = int(encoded_locations[-1])
                    self._snapshot_digest_part(digest, batch.ids)
                    self._snapshot_digest_part(digest, batch.embeddings)
                    if locations is not None:
                        self._snapshot_digest_part(digest, locations)
                    useful = _wire_scan_records(
                        rows_in_batch,
                        self.config.prefix,
                        self._dtype.itemsize,
                        with_locations=self.config.with_locations,
                    )
                    emitted += rows_in_batch
                    self.metrics.batch(
                        "scan_records",
                        worker_key,
                        latency_ns=latency_ns,
                        rows=rows_in_batch,
                        useful_bytes=useful,
                        source_verified_rows=0,
                        shadow_checked=False,
                    )
                    with state.lock:
                        state.progress_by_worker[key] = (
                            state.progress_by_worker.get(key, 0)
                            + rows_in_batch
                        )
                        state.last_progress_monotonic[key] = time.monotonic()
                    if emitted >= target_rows and target_rows < stream.snapshot_count:
                        break
                    if (
                        time.monotonic() - attempt_started
                        >= self.config.attempt_seconds
                        and emitted < target_rows
                    ):
                        status = "rotated"
                        break
            if status == "completed" and emitted < target_rows:
                raise SoakInvariantError(
                    "ScanRecords ended before its bounded target"
                )
            return status, digest.hexdigest()
        finally:
            self._unregister_stream(state, key)

    @staticmethod
    def _expected_controlled_stop_error(error: BaseException) -> bool:
        if isinstance(
            error,
            (
                throwdb.RemoteCancelledError,
                throwdb.RemoteDeadlineExceededError,
                throwdb.RemoteUnavailableError,
            ),
        ):
            return True
        return (
            isinstance(error, RuntimeError)
            and str(error) in {
                "remote collection is closed",
                "StreamGet is closed",
                "ScanRecords is closed",
                "record scan is closed",
            }
        )

    def _worker(
        self,
        state: _StreamingGeneration,
        mode: str,
        worker_index: int,
    ) -> None:
        key = (mode, worker_index)
        started = time.monotonic()
        self.metrics.worker_started(mode)
        client: Any | None = None
        first_attempt = True
        try:
            client = self._connect(state.endpoint)
            self._register_client(state, key, client)
            while not state.stop_event.is_set():
                try:
                    if mode == "stream_get":
                        result = self._run_stream_get_attempt(
                            state,
                            key,
                            client,
                            first_attempt=first_attempt,
                        )
                        if result in {
                            "rotated",
                            "cancelled",
                            "restart_interrupted",
                        }:
                            status = result
                            digest = None
                        else:
                            status = "completed"
                            digest = result
                    else:
                        status, digest = self._run_scan_records_attempt(
                            state,
                            key,
                            client,
                            first_attempt=first_attempt,
                        )
                    self.metrics.attempt_finished(
                        mode,
                        status=status,
                        digest=(
                            digest or None
                            if status == "completed"
                            else None
                        ),
                    )
                    if status == "restart_interrupted":
                        with state.lock:
                            state.interrupted_workers.add(key)
                    first_attempt = False
                    if status in {"cancelled", "restart_interrupted"}:
                        return
                except _ExpectedStreamStop:
                    return
                except benchmark.BenchmarkBudgetExceeded:
                    self.metrics.attempt_finished(
                        mode,
                        status="budget",
                        digest=None,
                    )
                    state.budget_exhausted.set()
                    self._cancel_state(state)
                    return
                except BaseException as error:
                    if (
                        (state.stop_event.is_set()
                         or state.expected_interruption.is_set())
                        and self._expected_controlled_stop_error(error)
                    ):
                        self.metrics.attempt_finished(
                            mode,
                            status=(
                                "restart_interrupted"
                                if state.expected_interruption.is_set()
                                else "cancelled"
                            ),
                            digest=None,
                        )
                        if state.expected_interruption.is_set():
                            with state.lock:
                                state.interrupted_workers.add(key)
                        return
                    self.metrics.failure(mode)
                    self._errors.record(mode, error)
                    with state.lock:
                        state.failures.append(error)
                    self._cancel_state(state)
                    return
        except _ExpectedStreamStop:
            pass
        except benchmark.BenchmarkBudgetExceeded:
            self.metrics.attempt_finished(mode, status="budget", digest=None)
            state.budget_exhausted.set()
            self._cancel_state(state)
        except BaseException as error:
            if (
                (state.stop_event.is_set()
                 or state.expected_interruption.is_set())
                and self._expected_controlled_stop_error(error)
            ):
                self.metrics.attempt_finished(
                    mode,
                    status=(
                        "restart_interrupted"
                        if state.expected_interruption.is_set()
                        else "cancelled"
                    ),
                    digest=None,
                )
                if state.expected_interruption.is_set():
                    with state.lock:
                        state.interrupted_workers.add(key)
            else:
                self.metrics.failure(mode)
                self._errors.record(mode, error)
                with state.lock:
                    state.failures.append(error)
                self._cancel_state(state)
        finally:
            with state.lock:
                state.streams.pop(key, None)
                state.clients.pop(key, None)
            if client is not None:
                with contextlib.suppress(BaseException):
                    client.close()
            self.metrics.worker_finished(mode, time.monotonic() - started)

    def start_generation(self, endpoint: str, generation: int) -> None:
        if not self.enabled:
            return
        if self._current is not None:
            raise RuntimeError("stream generation is already active")
        state = _StreamingGeneration(generation=generation, endpoint=endpoint)
        self._current = state
        for mode in self.config.enabled_modes:
            for worker_index in range(self.config.workers_for(mode)):
                thread = threading.Thread(
                    target=self._worker,
                    args=(state, mode, worker_index),
                    name=f"throwdb-soak-{mode}-g{generation}-w{worker_index}",
                    daemon=True,
                )
                state.threads.append(thread)
                thread.start()
        self.wait_until_open()

    def wait_until_open(self) -> None:
        state = self._current
        if state is None:
            return
        deadline = time.monotonic() + min(
            self.config.progress_timeout_seconds,
            max(0.001, self._budget.remaining_seconds()),
        )
        while time.monotonic() < deadline:
            self.check()
            with state.lock:
                if len(state.streams) == self.config.worker_count:
                    return
                alive = any(thread.is_alive() for thread in state.threads)
            if not alive:
                break
            time.sleep(0.02)
        self._cancel_state(state)
        raise SoakInvariantError(
            "stream workers did not all open before the progress timeout"
        )

    def release_and_wait_for_progress(self) -> None:
        state = self._current
        if state is None:
            return
        state.release_event.set()
        deadline = time.monotonic() + min(
            self.config.progress_timeout_seconds,
            max(0.001, self._budget.remaining_seconds()),
        )
        while time.monotonic() < deadline:
            self.check()
            with state.lock:
                expected_workers = {
                    (mode, worker_index)
                    for mode in self.config.enabled_modes
                    for worker_index in range(
                        self.config.workers_for(mode)
                    )
                }
                progressed = all(
                    state.progress_by_worker.get(key, 0) > 0
                    for key in expected_workers
                )
            if progressed:
                return
            if state.budget_exhausted.is_set():
                raise benchmark.BenchmarkBudgetExceeded(
                    "streaming budget was exhausted before every worker "
                    "made first-batch progress"
                )
            time.sleep(0.02)
        self._cancel_state(state)
        raise SoakInvariantError(
            "stream workers made no first-batch progress before timeout"
        )

    def wait_until_any_active(self) -> None:
        state = self._current
        if state is None:
            raise SoakInvariantError("no active stream generation")
        deadline = time.monotonic() + min(
            self.config.progress_timeout_seconds,
            max(0.001, self._budget.remaining_seconds()),
        )
        while time.monotonic() < deadline:
            self.check()
            with state.lock:
                if state.streams:
                    return
            time.sleep(0.02)
        raise SoakInvariantError(
            "no active stream remained for forced-restart interruption"
        )

    def expect_forced_interruption(self) -> None:
        if self._current is None:
            return
        self.wait_until_any_active()
        self.mark_forced_interruption()

    def mark_forced_interruption(self) -> None:
        """Mark the current generation immediately before supervisor SIGKILL."""

        self.wait_until_any_active()
        state = self._current
        if state is not None:
            with state.lock:
                state.expected_interruption_workers = set(
                    state.streams
                )
            state.expected_interruption.set()

    def check(self) -> None:
        state = self._current
        if state is None:
            return
        with state.lock:
            if state.failures:
                raise state.failures[0]
            alive = any(thread.is_alive() for thread in state.threads)
            now = time.monotonic()
            active_gaps = [
                now - state.last_progress_monotonic.get(key, now)
                for key in state.clients
            ]
        if not alive and not state.stop_event.is_set():
            raise SoakInvariantError(
                "every stream worker stopped before the workload barrier"
            )
        if active_gaps:
            worst_gap = max(active_gaps)
            if worst_gap > self.config.progress_timeout_seconds:
                error = SoakInvariantError(
                    "stream batch progress stalled for "
                    f"{worst_gap:.3f}s (limit "
                    f"{self.config.progress_timeout_seconds:.3f}s)"
                )
                with state.lock:
                    state.failures.append(error)
                self._cancel_state(state)
                raise error

    @property
    def budget_exhausted(self) -> bool:
        return bool(
            self._current is not None
            and self._current.budget_exhausted.is_set()
        )

    @staticmethod
    def _cancel_state(state: _StreamingGeneration) -> None:
        state.stop_event.set()
        state.release_event.set()
        with state.lock:
            streams = list(state.streams.values())
            clients = list(state.clients.values())
        for stream in streams:
            with contextlib.suppress(BaseException):
                stream.close()
        for client in clients:
            with contextlib.suppress(BaseException):
                client.close()

    def stop_generation(self, reason: str) -> dict[str, Any]:
        state = self._current
        if state is None:
            return {
                "generation": None,
                "reason": reason,
                "joined": True,
                "seconds": 0.0,
            }
        started = time.monotonic()
        self._cancel_state(state)
        deadline = started + self.config.stop_timeout_seconds
        for thread in state.threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)
        alive = [thread.name for thread in state.threads if thread.is_alive()]
        with state.lock:
            failures = list(state.failures)
            progress_by_worker = dict(state.progress_by_worker)
            expected_interruption_workers = set(
                state.expected_interruption_workers
            )
            interrupted_workers = set(state.interrupted_workers)
        progress_by_mode = {
            mode: sum(
                rows
                for (worker_mode, _), rows in progress_by_worker.items()
                if worker_mode == mode
            )
            for mode in self.config.enabled_modes
        }
        encoded_worker_progress = [
            {
                "mode": mode,
                "worker_index": worker_index,
                "rows": rows,
            }
            for (mode, worker_index), rows in sorted(
                progress_by_worker.items()
            )
        ]
        event = {
            "generation": state.generation,
            "reason": reason,
            "expected_forced_interruption": (
                state.expected_interruption.is_set()
            ),
            "expected_interruption_workers": [
                {"mode": mode, "worker_index": worker_index}
                for mode, worker_index in sorted(
                    expected_interruption_workers
                )
            ],
            "interrupted_workers": [
                {"mode": mode, "worker_index": worker_index}
                for mode, worker_index in sorted(interrupted_workers)
            ],
            "all_marked_workers_interrupted": (
                not expected_interruption_workers
                or expected_interruption_workers <= interrupted_workers
            ),
            "progress_rows": progress_by_mode,
            "progress_rows_by_worker": encoded_worker_progress,
            "every_worker_progressed": (
                len(progress_by_worker) == self.config.worker_count
                and all(rows > 0 for rows in progress_by_worker.values())
            ),
            "joined": not alive,
            "alive_threads": alive,
            "seconds": time.monotonic() - started,
        }
        if alive:
            raise SoakInvariantError(
                "stream workers did not stop within "
                f"{self.config.stop_timeout_seconds:.3f}s: {alive}"
            )
        self.metrics.record_generation(event)
        self._current = None
        if failures:
            raise failures[0]
        return event

    def summary(self) -> dict[str, Any]:
        return {
            "configuration": _json_value(self.config),
            **self.metrics.summary(),
        }


def load_population(
    path: Path,
    requested: int,
    seed: int,
    budget: benchmark.ResourceBudget | None = None,
) -> PopulationSample:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise SoakConfigurationError("--ids must be a one-dimensional integer .npy")
    if values.size == 0:
        raise SoakConfigurationError("--ids is empty")
    target = min(requested, int(values.size))
    if budget is not None:
        # Python int/list/set bookkeeping dominates the final 16-byte pair of
        # uint64/int64 arrays. Keep this deliberately conservative and check
        # before constructing a large sampler.
        budget.check(extra_memory_bytes=target * 192)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_rows: list[int] = []
    seen: set[int] = set()
    max_attempts = max(10_000, target * 100)
    for _ in range(max_attempts):
        index = int(rng.integers(0, values.size))
        raw = int(values[index])
        if raw < 0 or raw > UINT64_MAX:
            raise SoakConfigurationError("--ids contains a value outside uint64")
        if raw not in seen:
            seen.add(raw)
            selected.append(raw)
            selected_rows.append(int(index))
            if budget is not None and len(selected) % 4096 == 0:
                remaining = target - len(selected)
                budget.check(
                    extra_memory_bytes=remaining * 192 + target * 16
                )
            if len(selected) == target:
                break
    if not selected:
        raise SoakConfigurationError("--ids has no usable IDs")
    return PopulationSample(
        ids=np.asarray(selected, dtype=np.uint64),
        source_rows=np.asarray(selected_rows, dtype=np.int64),
    )


def discover_population(
    remote: Any,
    *,
    requested: int,
    batch_rows: int,
    budget: benchmark.ResourceBudget,
) -> PopulationSample:
    selected: list[np.ndarray] = []
    emitted = 0
    effective_batch_rows = min(batch_rows, requested)
    # RemoteIdScan receives its first response eagerly during construction.
    # Bound that response by the retained target and reserve the complete
    # maximum ID payload before opening the stream.
    budget.reserve(
        requested * 8,
        extra_memory_bytes=(
            requested * 40 + effective_batch_rows * 16
        ),
    )
    remote._scan_timeout = budget.rpc_timeout(  # type: ignore[attr-defined]
        min(300.0, budget.remaining_seconds())
    )
    with remote.scan_ids(batch_rows=effective_batch_rows) as scan:
        for batch in scan:
            ids = np.asarray(batch, dtype=np.uint64)
            take = min(ids.size, requested - emitted)
            if take <= 0:
                break
            selected.append(np.array(ids[:take], copy=True))
            emitted += take
            if emitted == requested:
                break
    if not selected:
        raise SoakConfigurationError("collection scan returned no IDs")
    result = np.concatenate(selected)
    if np.unique(result).size != result.size:
        raise SoakInvariantError("ScanIds returned duplicate IDs in its prefix")
    return PopulationSample(ids=result, source_rows=None)


def scan_acceptance(
    remote: Any,
    *,
    rows: int,
    batch_rows: int,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
) -> dict[str, Any]:
    if rows == 0:
        return {"status": "skipped", "reason": "--scan-rows is zero"}
    planned = min(rows, remote.describe().count)
    effective_batch_rows = min(batch_rows, max(1, planned))
    budget.reserve(
        planned * 16,
        extra_memory_bytes=effective_batch_rows * 32,
    )
    remote._scan_timeout = budget.rpc_timeout(rpc_timeout)  # type: ignore[attr-defined]
    digest = hashlib.sha256()
    emitted = 0
    previous: tuple[int, int] | None = None
    started = time.perf_counter()
    with remote.scan_ids(
        batch_rows=effective_batch_rows,
        with_locations=True,
    ) as scan:
        snapshot = _json_value(scan.snapshot)
        for item in scan:
            if not isinstance(item, throwdb.LocatedIdBatch):
                raise SoakInvariantError("ScanIds omitted requested locations")
            take = min(item.ids.size, planned - emitted)
            if take <= 0:
                break
            locations = item.locations[:take]
            if item.emitted_offset != emitted:
                raise SoakInvariantError("ScanIds offsets are not contiguous")
            for location in locations:
                current = (int(location["segment_id"]), int(location["row_index"]))
                if previous is not None and current <= previous:
                    raise SoakInvariantError(
                        "ScanIds locations are not strictly increasing"
                    )
                previous = current
            digest.update(
                memoryview(np.asarray(item.ids[:take], dtype="<u8")).cast("B")
            )
            digest.update(memoryview(locations).cast("B"))
            emitted += take
            if emitted == planned:
                break
    elapsed = time.perf_counter() - started
    return {
        "status": "passed",
        "rows": emitted,
        "requested_batch_rows": batch_rows,
        "effective_batch_rows": effective_batch_rows,
        "snapshot": snapshot,
        "digest_sha256": digest.hexdigest(),
        "seconds": elapsed,
        "rows_per_second": emitted / elapsed if elapsed else 0.0,
        "physical_locations_strictly_increasing": True,
    }


def capture_stream_get_oracle(
    remote: Any,
    *,
    ids: np.ndarray,
    prefix: int,
    batch_rows: int,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
) -> np.ndarray:
    """Capture a bounded unary oracle used for every repeated StreamGet row."""

    normalized = np.asarray(ids, dtype=np.uint64)
    if normalized.ndim != 1 or normalized.size == 0:
        raise SoakConfigurationError("stream oracle IDs must be non-empty")
    itemsize = np.dtype(remote.dtype).itemsize
    budget.check(
        extra_memory_bytes=normalized.size * prefix * itemsize,
    )
    output = np.empty((normalized.size, prefix), dtype=np.dtype(remote.dtype))
    for begin in range(0, normalized.size, batch_rows):
        end = min(normalized.size, begin + batch_rows)
        rows = end - begin
        budget.reserve(
            _wire_get(rows, prefix, itemsize),
            extra_memory_bytes=3 * rows * prefix * itemsize,
        )
        _set_timeout(remote, budget, rpc_timeout)
        values = remote.get(
            normalized[begin:end],
            prefix_dimensions=(
                None if prefix == remote.dimension else prefix
            ),
        )
        expected_shape = (rows, prefix)
        if values.shape != expected_shape or values.dtype != output.dtype:
            raise SoakInvariantError(
                "unary stream oracle returned an invalid dtype or shape"
            )
        output[begin:end] = values
    return output


@dataclass(frozen=True, slots=True)
class FairnessQuery:
    ids: np.ndarray
    source_expected: np.ndarray | None


def build_fairness_queries(
    remote: Any,
    *,
    population: np.ndarray,
    population_source_rows: np.ndarray | None,
    source_embeddings: np.ndarray | None,
    shadow_id: np.ndarray,
    calls: int,
    batch_size: int,
    prefix: int,
    seed: int,
    budget: benchmark.ResourceBudget,
) -> tuple[list[FairnessQuery], str]:
    """Build one deterministic corpus reused by isolated and mixed windows."""

    rng = np.random.default_rng(seed ^ 0xBACE_1EAF)
    digest = hashlib.sha256()
    itemsize = np.dtype(remote.dtype).itemsize
    budget.check(
        extra_memory_bytes=(
            calls * batch_size * (8 + prefix * itemsize) * 2
        ),
    )
    queries: list[FairnessQuery] = []
    for _ in range(calls):
        positions = rng.integers(
            0,
            population.size,
            size=batch_size,
            dtype=np.int64,
        )
        positions[0] = seed % population.size
        ids = np.asarray(
            population[positions],
            dtype=np.uint64,
            order="C",
        ).copy()
        ids[0] = shadow_id[0]
        digest.update(memoryview(np.asarray(ids, dtype="<u8")).cast("B"))
        queries.append(
            FairnessQuery(
                ids=ids,
                source_expected=(
                    np.array(
                        source_embeddings[
                            population_source_rows[positions],
                            :prefix,
                        ],
                        copy=True,
                        order="C",
                    )
                    if source_embeddings is not None
                    and population_source_rows is not None
                    else None
                ),
            )
        )
    return queries, digest.hexdigest()


def run_get_fairness_window(
    remote: Any,
    *,
    queries: Sequence[FairnessQuery],
    query_digest_sha256: str,
    label: str,
    shadow_payload: np.ndarray,
    prefix: int,
    workers: int,
    max_inflight: int,
    budget: benchmark.ResourceBudget,
    rpc_timeout: float,
    retries: int,
    metrics: OperationMetrics,
    errors: ErrorMetrics,
) -> dict[str, Any]:
    """Run one matched concurrent short-Get responsiveness window."""

    itemsize = np.dtype(remote.dtype).itemsize
    successful = 0
    clients = [
        _connect(remote.endpoint, budget, rpc_timeout)
        for _ in range(workers)
    ]
    locks = [threading.Lock() for _ in clients]
    inflight: set[Future[bool]] = set()

    def consume(done: set[Future[bool]]) -> None:
        nonlocal successful
        for future in done:
            successful += int(future.result())

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ordinal, query in enumerate(queries):
                while len(inflight) >= max_inflight:
                    done, inflight = wait(
                        inflight,
                        return_when=FIRST_COMPLETED,
                    )
                    consume(done)
                wire = _wire_get(query.ids.size, prefix, itemsize)
                budget.reserve(
                    wire,
                    extra_memory_bytes=(
                        3 * query.ids.size * prefix * itemsize
                    ),
                )
                plan = GetPlan(
                    ordinal=ordinal,
                    label=label,
                    ids=query.ids,
                    prefix=prefix,
                    wire_bytes=wire,
                    shadow_expected=np.array(
                        shadow_payload[:, :prefix],
                        copy=True,
                        order="C",
                    ),
                    source_expected=query.source_expected,
                )
                client_index = ordinal % len(clients)
                inflight.add(
                    executor.submit(
                        execute_get_on_client,
                        locks[client_index],
                        clients[client_index],
                        plan,
                        dimension=remote.dimension,
                        itemsize=itemsize,
                        retries=retries,
                        budget=budget,
                        rpc_timeout=rpc_timeout,
                        metrics=metrics,
                        errors=errors,
                    )
                )
            while inflight:
                done, inflight = wait(
                    inflight,
                    return_when=FIRST_COMPLETED,
                )
                consume(done)
    finally:
        for client in clients:
            with contextlib.suppress(BaseException):
                client.close()
    return {
        "label": label,
        "calls": len(queries),
        "successful_calls": successful,
        "batch_size": int(queries[0].ids.size),
        "prefix_dimensions": prefix,
        "query_ids_digest_sha256": query_digest_sha256,
        "source_oracle": all(
            query.source_expected is not None for query in queries
        ),
        "workers": workers,
        "max_inflight": max_inflight,
    }


def wait_for_scan_admission(
    remote: Any,
    *,
    expected_active_scans: int | None = None,
    minimum_admitted_scans: int | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Poll Describe until scan counters/gauges reach the requested state."""

    if expected_active_scans is None and minimum_admitted_scans is None:
        raise ValueError("one scan-admission expectation is required")

    started = time.monotonic()
    deadline = started + timeout_seconds
    last: Any | None = None
    while time.monotonic() < deadline:
        remote._timeout = max(  # type: ignore[attr-defined]
            0.001,
            min(2.0, deadline - time.monotonic()),
        )
        description = remote.describe()
        last = description.admission_stats
        if last is None:
            raise SoakConfigurationError(
                "streaming acceptance requires admission statistics"
        )
        active_matches = (
            True
            if expected_active_scans is None
            else (
                last.active_scans == 0
                and last.scan_estimated_bytes == 0
                if expected_active_scans == 0
                else (
                    last.active_scans >= expected_active_scans
                    and last.scan_estimated_bytes > 0
                )
            )
        )
        admitted_matches = (
            minimum_admitted_scans is None
            or last.admitted_scans >= minimum_admitted_scans
        )
        matches = active_matches and admitted_matches
        if matches:
            return {
                "status": "passed",
                "expected_active_scans": expected_active_scans,
                "minimum_admitted_scans": minimum_admitted_scans,
                "elapsed_seconds": time.monotonic() - started,
                "admission_stats": _json_value(last),
            }
        time.sleep(0.02)
    return {
        "status": "failed",
        "expected_active_scans": expected_active_scans,
        "minimum_admitted_scans": minimum_admitted_scans,
        "elapsed_seconds": time.monotonic() - started,
        "admission_stats": _json_value(last),
    }


def fairness_acceptance(
    operations: Mapping[str, Mapping[str, Any]],
    *,
    baseline_label: str,
    mixed_label: str,
    minimum_mixed_successes: int,
    maximum_p99_ratio: float,
) -> dict[str, Any]:
    baseline = operations.get(baseline_label)
    mixed = operations.get(mixed_label)
    baseline_samples = (
        int(
            baseline.get(
                "successful_latency_sample_count",
                baseline.get("latency_sample_count", 0),
            )
        )
        if baseline is not None
        else 0
    )
    mixed_samples = (
        int(
            mixed.get(
                "successful_latency_sample_count",
                mixed.get("latency_sample_count", 0),
            )
        )
        if mixed is not None
        else 0
    )
    mixed_successes = (
        int(mixed.get("successes", 0)) if mixed is not None else 0
    )
    baseline_p99 = (
        float(
            baseline.get(
                "successful_latency_p99_ms",
                baseline["latency_p99_ms"],
            )
        )
        if baseline is not None and "latency_p99_ms" in baseline
        else None
    )
    mixed_p99 = (
        float(
            mixed.get(
                "successful_latency_p99_ms",
                mixed["latency_p99_ms"],
            )
        )
        if mixed is not None and "latency_p99_ms" in mixed
        else None
    )
    baseline_errors = (
        int(baseline.get("errors", 0)) if baseline is not None else 0
    )
    evaluable = (
        baseline_p99 is not None
        and mixed_p99 is not None
        and baseline_samples > 0
        and mixed_samples >= minimum_mixed_successes
        and mixed_successes >= minimum_mixed_successes
        and baseline_errors == 0
    )
    ratio = (
        mixed_p99 / baseline_p99
        if evaluable and baseline_p99 is not None and baseline_p99 > 0
        else None
    )
    passed = bool(
        evaluable
        and ratio is not None
        and ratio <= maximum_p99_ratio
    )
    return {
        "baseline_label": baseline_label,
        "mixed_label": mixed_label,
        "baseline_sample_count": baseline_samples,
        "mixed_sample_count": mixed_samples,
        "mixed_successes": mixed_successes,
        "baseline_errors": baseline_errors,
        "baseline_p99_ms": baseline_p99,
        "mixed_p99_ms": mixed_p99,
        "observed_p99_ratio": ratio,
        "maximum_p99_ratio": maximum_p99_ratio,
        "minimum_mixed_successes": minimum_mixed_successes,
        "evaluable": evaluable,
        "passed": passed,
        "reason": (
            None
            if passed
            else (
                (
                    "isolated baseline contained RPC errors"
                    if baseline_errors
                    else "insufficient successful latency samples"
                )
                if not evaluable
                else "mixed short-Get P99 exceeded the configured ratio"
            )
        ),
    }


class LocalSupervisor:
    def __init__(self, args: argparse.Namespace, results: Path) -> None:
        self._args = args
        self._results = results
        self.server: benchmark.LocalServer | None = None
        self.generation = -1
        self.generations: list[dict[str, Any]] = []
        self._started_monotonic = 0.0
        self._initial_snapshot: dict[str, Any] | None = None

    @property
    def endpoint(self) -> str:
        if self.server is None:
            raise RuntimeError("local server is not running")
        return self.server.endpoint

    @property
    def pid(self) -> int | None:
        return self.server.pid if self.server is not None else None

    def start(self, timeout: float) -> None:
        self.generation += 1
        directory = self._results / "server-generations" / f"g{self.generation:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        self.server = benchmark.LocalServer.start(
            store=self._args.store,
            results=directory,
            tuning=self._args.tuning,
            timeout=timeout,
            writable=True,
            native_binary=self._args.native_binary,
            max_worker_threads=self._args.server_max_worker_threads,
            max_ids_per_request=self._args.server_max_ids_per_request,
            max_scan_batch_rows=self._args.server_max_scan_batch_rows,
            maintenance_interval_seconds=(
                self._args.server_maintenance_interval_seconds
            ),
            disable_automatic_maintenance=self._args.server_no_auto_maintenance,
            max_concurrent_calls=self._args.server_max_concurrent_calls,
            max_concurrent_scans=self._args.server_max_concurrent_scans,
            scan_max_duration_ms=self._args.server_scan_max_duration_ms,
            scan_idle_timeout_ms=self._args.server_scan_idle_timeout_ms,
            shutdown_grace_ms=self._args.server_shutdown_grace_ms,
            shutdown_hard_timeout_ms=self._args.server_shutdown_hard_timeout_ms,
        )
        self._started_monotonic = started
        self._initial_snapshot = process_snapshot(self.server.pid)

    def stop(self, mode: str = "graceful") -> None:
        if self.server is None:
            return
        if mode not in {"graceful", "sigkill"}:
            raise ValueError("server stop mode must be graceful or sigkill")
        server = self.server
        final_snapshot = process_snapshot(server.pid)
        if mode == "sigkill" and server.process.poll() is None:
            server.process.kill()
            server.process.wait(timeout=10)
        server.close()
        self.generations.append(
            {
                "generation": self.generation,
                "pid": server.pid,
                "endpoint": server.endpoint,
                "command": server.command,
                "ready": server.ready,
                "uptime_seconds": time.monotonic() - self._started_monotonic,
                "returncode": server.process.returncode,
                "stop_mode": mode,
                "expected_forced_exit": mode == "sigkill",
                "process_delta": benchmark.process_delta(
                    self._initial_snapshot or {}, final_snapshot
                ),
                "stdout": str(server.stdout_path),
                "stderr": str(server.stderr_path),
            }
        )
        self.server = None

    def restart(self, timeout: float, mode: str) -> float:
        started = time.perf_counter()
        self.stop(mode)
        self.start(timeout)
        return time.perf_counter() - started


def _connect(endpoint: str, budget: benchmark.ResourceBudget, timeout: float) -> Any:
    return throwdb.RemoteCollection.connect(
        endpoint,
        timeout=budget.rpc_timeout(timeout),
        scan_timeout=budget.rpc_timeout(max(timeout, 60.0)),
    )


def restart_replay_probe(
    *,
    remote: Any,
    supervisor: LocalSupervisor,
    canaries: np.ndarray,
    shadow_id: np.ndarray,
    shadow_payload: np.ndarray,
    sequence: int,
    seed: int,
    run_id: uuid.UUID = uuid.NAMESPACE_URL,
    budget: benchmark.ResourceBudget,
    mutation_budget: MutationBudget,
    metrics: OperationMetrics,
    errors: ErrorMetrics,
    rpc_timeout: float,
    start_timeout: float,
    restart_mode: str,
    before_restart: Callable[[], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    # Two scheduled UUIDs plus one fresh emergency cleanup UUID.
    if not mutation_budget.can_reserve(3):
        raise benchmark.BenchmarkBudgetExceeded(
            "restart replay probe lacks mutation and cleanup headroom"
        )
    record_size = remote.dimension * np.dtype(remote.dtype).itemsize
    payload = payload_for(
        canaries, version=10_000 + sequence,
        dimension=remote.dimension, dtype=remote.dtype, seed=seed,
    )
    reserved = 4 * _wire_put(canaries.size, record_size) + 3 * (
        canaries.size * (8 + record_size)
    ) + 3 * canaries.size * 9 + _wire_get(
        shadow_id.size,
        remote.dimension,
        np.dtype(remote.dtype).itemsize,
    )
    budget.reserve(reserved, extra_memory_bytes=4 * canaries.size * record_size)
    mutation_budget.reserve(2)
    before = remote.describe()
    domain_before = before.idempotency_domain
    insert_id = deterministic_operation_id(
        seed, sequence, "restart-insert", run_id=run_id
    )
    delete_id = deterministic_operation_id(
        seed, sequence, "restart-delete", run_id=run_id
    )
    restored = False
    try:
        _set_timeout(remote, budget, rpc_timeout)
        started = time.perf_counter_ns()
        remote.insert(canaries, payload, operation_id=insert_id)
        metrics.record(
            "restart_insert", sequence * 1000, time.perf_counter_ns() - started,
            rows=canaries.size, useful_bytes=_wire_put(canaries.size, record_size),
            success=True,
        )
        _assert_bits(remote.get(canaries), payload, "pre-restart insert")
        remote.close()

        restart_started = time.perf_counter()
        if before_restart is not None:
            before_restart()
        supervisor.restart(
            min(start_timeout, budget.remaining_seconds()), restart_mode
        )
        remote = _connect(supervisor.endpoint, budget, rpc_timeout)
        after = remote.describe()
        domain_after = after.idempotency_domain
        if domain_after != domain_before:
            raise SoakInvariantError(
                "idempotency replay domain changed across restart; refusing "
                "to replay the acknowledged mutation"
            )
        _set_timeout(remote, budget, rpc_timeout)
        _assert_bits(
            remote.get(shadow_id),
            shadow_payload,
            "post-restart base shadow",
        )
        recovery_seconds = time.perf_counter() - restart_started

        _set_timeout(remote, budget, rpc_timeout)
        started = time.perf_counter_ns()
        remote.insert(canaries, payload, operation_id=insert_id)
        metrics.record(
            "restart_insert_replay", sequence * 1000 + 1,
            time.perf_counter_ns() - started,
            rows=canaries.size, useful_bytes=_wire_put(canaries.size, record_size),
            success=True,
        )
        _assert_bits(remote.get(canaries), payload, "post-restart replay")

        _set_timeout(remote, budget, rpc_timeout)
        deleted = remote.delete(canaries, operation_id=delete_id)
        _set_timeout(remote, budget, rpc_timeout)
        replayed = remote.delete(canaries, operation_id=delete_id)
        if not bool(deleted.all()) or not np.array_equal(deleted, replayed):
            raise SoakInvariantError("post-restart Delete replay mask changed")
        _expect_missing(
            remote,
            canaries,
            deadline_monotonic=budget.deadline,
        )
        restored = True
        return remote, {
            "status": "passed",
            "sequence": sequence,
            "recovery_seconds": recovery_seconds,
            "restart_mode": restart_mode,
            "idempotency_domain_before": str(domain_before),
            "idempotency_domain_after": str(domain_after),
            "acknowledged_insert_replayed_after_restart": True,
            "logical_absent_state_restored": True,
        }
    except Exception as error:
        errors.record("restart_replay_probe", error)
        raise
    finally:
        if not restored and remote is not None:
            if mutation_budget.can_reserve(1):
                mutation_budget.reserve(1)
                try:
                    cleanup = deterministic_operation_id(
                        seed,
                        sequence,
                        "restart-cleanup-delete",
                        run_id=run_id,
                    )
                    remote._timeout = min(  # type: ignore[attr-defined]
                        10.0, rpc_timeout
                    )
                    remote.delete(canaries, operation_id=cleanup)
                    _expect_missing(
                        remote,
                        canaries,
                        deadline_monotonic=(
                            time.monotonic() + min(10.0, rpc_timeout)
                        ),
                    )
                except Exception as cleanup_error:
                    errors.record("restart_cleanup", cleanup_error)


def validate_args(args: argparse.Namespace) -> None:
    if args.endpoint is None and args.store is None:
        raise SoakConfigurationError("--store is required for local launch")
    if args.endpoint is not None and args.store is not None:
        raise SoakConfigurationError("pass either --endpoint or --store, not both")
    if args.endpoint is not None and args.native_binary is not None:
        raise SoakConfigurationError("--native-binary is local-launch only")
    if args.endpoint is not None:
        local_only_overrides = {
            "--tuning": args.tuning != "auto",
            "--server-start-timeout": args.server_start_timeout != 30.0,
            "--server-max-worker-threads": (
                args.server_max_worker_threads is not None
            ),
            "--server-max-concurrent-calls": (
                args.server_max_concurrent_calls is not None
            ),
            "--server-max-concurrent-scans": (
                args.server_max_concurrent_scans is not None
            ),
            "--server-max-ids-per-request": (
                args.server_max_ids_per_request is not None
            ),
            "--server-max-scan-batch-rows": (
                args.server_max_scan_batch_rows is not None
            ),
            "--server-scan-max-duration-ms": (
                args.server_scan_max_duration_ms is not None
            ),
            "--server-scan-idle-timeout-ms": (
                args.server_scan_idle_timeout_ms is not None
            ),
            "--server-shutdown-grace-ms": (
                args.server_shutdown_grace_ms is not None
            ),
            "--server-shutdown-hard-timeout-ms": (
                args.server_shutdown_hard_timeout_ms is not None
            ),
            "--server-maintenance-interval-seconds": (
                args.server_maintenance_interval_seconds is not None
            ),
            "--server-no-auto-maintenance": (
                args.server_no_auto_maintenance
            ),
        }
        ignored = [
            option
            for option, configured in local_only_overrides.items()
            if configured
        ]
        if ignored:
            raise SoakConfigurationError(
                "local-launch-only options cannot be used with --endpoint: "
                + ", ".join(ignored)
            )
    if args.endpoint is None and args.server_pid is not None:
        raise SoakConfigurationError("--server-pid is external-endpoint only")
    if args.endpoint is not None and args.restart_cycles:
        raise SoakConfigurationError("restart cycles require a locally launched server")
    if args.endpoint is not None and not args.allow_external_mutations:
        raise SoakConfigurationError(
            "external mutation soak requires --allow-external-mutations"
        )
    if args.source_embeddings is not None and args.ids is None:
        raise SoakConfigurationError("--source-embeddings requires --ids")
    if args.stream_source_oracle and args.source_embeddings is None:
        raise SoakConfigurationError(
            "--stream-source-oracle requires --source-embeddings"
        )
    if args.require_maintenance_progress and args.server_no_auto_maintenance:
        raise SoakConfigurationError(
            "--require-maintenance-progress conflicts with "
            "--server-no-auto-maintenance"
        )
    if args.max_inflight_gets < args.get_workers:
        raise SoakConfigurationError(
            "--max-inflight-gets must be >= --get-workers"
        )
    if (
        args.fairness_baseline_get_calls
        < args.min_mixed_small_get_successes
    ):
        raise SoakConfigurationError(
            "--fairness-baseline-get-calls must be >= "
            "--min-mixed-small-get-successes"
        )
    if args.max_latency_samples < args.min_mixed_small_get_successes:
        raise SoakConfigurationError(
            "--max-latency-samples must be >= "
            "--min-mixed-small-get-successes"
        )
    configured_get_classes = (
        len(args.get_batch_sizes)
        * len([part for part in args.prefix_dimensions.split(",") if part])
    )
    if (
        args.max_get_calls
        and args.max_get_calls < configured_get_classes
    ):
        raise SoakConfigurationError(
            "--max-get-calls must exercise every configured Get class at "
            "least once"
        )
    if args.get_retries > 99:
        raise SoakConfigurationError("--get-retries must be at most 99")
    if args.mutation_primary_attempts > 50:
        raise SoakConfigurationError(
            "--mutation-primary-attempts must be at most 50"
        )
    stream_workers = args.stream_get_workers + args.scan_records_workers
    sigkill_stream_probe_requested = bool(
        stream_workers
        and args.restart_cycles
        and (
            args.restart_mode == "sigkill"
            or (
                args.restart_mode == "alternating"
                and args.restart_cycles >= 2
            )
        )
    )
    if sigkill_stream_probe_requested and args.stream_get_workers == 0:
        raise SoakConfigurationError(
            "active-stream SIGKILL acceptance requires at least one "
            "--stream-get-worker"
        )
    if (
        sigkill_stream_probe_requested
        and args.stream_get_rows <= args.stream_batch_rows
    ):
        raise SoakConfigurationError(
            "active-stream SIGKILL acceptance requires --stream-get-rows "
            "greater than --stream-batch-rows"
        )
    if (
        args.server_max_concurrent_scans is not None
        and stream_workers > args.server_max_concurrent_scans
    ):
        raise SoakConfigurationError(
            "configured stream workers exceed --server-max-concurrent-scans"
        )
    if (
        args.server_max_concurrent_calls is not None
        and stream_workers + args.get_workers + 1
        > args.server_max_concurrent_calls
    ):
        raise SoakConfigurationError(
            "stream workers, Get workers, and one control/mutation RPC "
            "exceed --server-max-concurrent-calls"
        )
    if (
        args.server_scan_max_duration_ms is not None
        and args.stream_attempt_seconds * 1000
        >= args.server_scan_max_duration_ms
    ):
        raise SoakConfigurationError(
            "--stream-attempt-seconds must be below the server scan watchdog"
        )
    if args.stream_get_rows > sys.maxsize:
        raise SoakConfigurationError("--stream-get-rows must fit Py_ssize_t")
    minimum_mutations = 5 + 2 * args.restart_cycles
    if args.max_mutations < minimum_mutations:
        raise SoakConfigurationError(
            f"--max-mutations must be at least {minimum_mutations} for CRUD, "
            "requested restart probes, and emergency cleanup headroom"
        )
    if args.max_get_calls and args.restart_cycles:
        required = args.restart_every_gets * args.restart_cycles
        if args.max_get_calls < required:
            raise SoakConfigurationError(
                "--max-get-calls cannot reach every requested restart"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded mixed-workload/restart soak for ThrowDB gRPC",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--store", "--storage", dest="store", type=Path)
    target.add_argument("--endpoint")
    parser.add_argument(
        "--server-pid",
        type=_positive_int,
        help=(
            "optional visible server PID for /proc telemetry when --endpoint "
            "is used"
        ),
    )
    parser.add_argument("--ids", type=Path)
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        help="optional row-aligned .npy bit-exact oracle for --ids",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument(
        "--tuning", choices=("auto", "persisted", "conservative"),
        default="auto",
    )
    parser.add_argument("--allow-external-mutations", action="store_true")
    parser.add_argument("--server-start-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--rpc-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--server-max-worker-threads", type=_positive_int)
    parser.add_argument("--server-max-concurrent-calls", type=_positive_int)
    parser.add_argument("--server-max-concurrent-scans", type=_positive_int)
    parser.add_argument("--server-max-ids-per-request", type=_positive_int)
    parser.add_argument("--server-max-scan-batch-rows", type=_positive_int)
    parser.add_argument("--server-scan-max-duration-ms", type=_positive_int)
    parser.add_argument("--server-scan-idle-timeout-ms", type=_positive_int)
    parser.add_argument("--server-shutdown-grace-ms", type=_positive_int)
    parser.add_argument("--server-shutdown-hard-timeout-ms", type=_positive_int)
    parser.add_argument("--server-maintenance-interval-seconds", type=_positive_int)
    parser.add_argument("--server-no-auto-maintenance", action="store_true")
    parser.add_argument(
        "--require-maintenance-progress",
        action="store_true",
        help=(
            "fail unless final engine counters contain maintenance publication "
            "evidence; disabled by default because small runs may not hit policy "
            "thresholds"
        ),
    )

    parser.add_argument("--seed", type=_uint64, default=DEFAULT_SEED)
    parser.add_argument(
        "--run-id",
        type=_uuid,
        help=(
            "globally unique logical-run UUID; generated by default and used "
            "as the namespace for deterministic mutation operation IDs"
        ),
    )
    parser.add_argument("--max-seconds", type=_positive_float, default=3600.0)
    parser.add_argument("--max-transfer-gib", type=_positive_float, default=64.0)
    parser.add_argument("--max-client-memory-mib", type=_positive_float, default=4096.0)
    parser.add_argument("--max-mutations", type=_positive_int, default=400)
    parser.add_argument(
        "--max-get-calls", type=_nonnegative_int, default=10_000,
        help="0 keeps issuing Gets until another run-wide bound is reached",
    )
    parser.add_argument("--max-errors", type=_nonnegative_int, default=0)
    parser.add_argument("--get-workers", type=_positive_int, default=4)
    parser.add_argument("--max-inflight-gets", type=_positive_int, default=8)
    parser.add_argument("--get-retries", type=_nonnegative_int, default=1)
    parser.add_argument("--get-batch-sizes", type=parse_sizes, default=(1, 128, 1024))
    parser.add_argument("--prefix-dimensions", default="full,128")
    parser.add_argument("--stream-get-workers", type=_nonnegative_int, default=1)
    parser.add_argument(
        "--scan-records-workers",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--stream-get-rows",
        type=_positive_int,
        default=131_072,
        help="logical ID rows in one finite StreamGet snapshot attempt",
    )
    parser.add_argument(
        "--scan-records-rows",
        type=_positive_int,
        default=131_072,
        help="minimum physical rows consumed in one ScanRecords attempt",
    )
    parser.add_argument(
        "--stream-batch-rows",
        type=_positive_int,
        default=16_384,
    )
    parser.add_argument(
        "--stream-max-batch-mib",
        type=_positive_float,
        default=32.0,
    )
    parser.add_argument(
        "--stream-prefix-dimensions",
        default="full",
        help="one prefix dimension or 'full' for both streaming modes",
    )
    parser.add_argument(
        "--stream-attempt-seconds",
        type=_positive_float,
        default=300.0,
        help="rotate a partial stream below server/client watchdog deadlines",
    )
    parser.add_argument(
        "--stream-progress-timeout-seconds",
        type=_positive_float,
        default=60.0,
    )
    parser.add_argument(
        "--stream-stop-timeout-seconds",
        type=_positive_float,
        default=15.0,
    )
    parser.add_argument(
        "--stream-locations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request and validate exact physical locations in ScanRecords",
    )
    parser.add_argument(
        "--stream-source-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "verify the bounded unary StreamGet oracle once against source "
            "embeddings before the mixed window"
        ),
    )
    parser.add_argument(
        "--stream-oracle-rows",
        type=_positive_int,
        default=256,
        help=(
            "bounded unique-ID corpus captured through unary Get and then "
            "checked bit-for-bit in every StreamGet batch"
        ),
    )
    parser.add_argument(
        "--fairness-baseline-get-calls",
        type=_positive_int,
        default=32,
    )
    parser.add_argument(
        "--min-mixed-small-get-successes",
        type=_positive_int,
        default=20,
    )
    parser.add_argument(
        "--max-small-get-p99-ratio",
        type=_positive_float,
        default=5.0,
        help="hard mixed/isolated P99 limit for the smallest Get class",
    )
    parser.add_argument(
        "--min-stream-batches-per-mode",
        type=_positive_int,
        default=1,
    )
    parser.add_argument(
        "--min-stream-rows-per-second",
        type=_nonnegative_float,
        default=0.0,
        help="optional absolute per-worker active-time progress gate",
    )
    parser.add_argument("--population-rows", type=_positive_int, default=100_000)
    parser.add_argument(
        "--population-scan-batch-rows",
        type=_positive_int,
        default=65_536,
    )
    parser.add_argument("--mutation-rows", type=_positive_int, default=1)
    parser.add_argument("--mutation-every-gets", type=_positive_int, default=20)
    parser.add_argument("--mutation-primary-attempts", type=_positive_int, default=3)
    parser.add_argument("--canary-id-start", type=_uint64, default=UINT64_MAX)
    parser.add_argument("--restart-cycles", type=_nonnegative_int, default=2)
    parser.add_argument("--restart-every-gets", type=_positive_int, default=250)
    parser.add_argument(
        "--restart-mode",
        choices=("alternating", "graceful", "sigkill"),
        default="alternating",
        help="alternating starts with graceful and then exercises SIGKILL",
    )
    parser.add_argument("--max-recovery-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--scan-rows", type=_nonnegative_int, default=0)
    parser.add_argument("--scan-batch-rows", type=_positive_int, default=65_536)
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=_positive_float,
        default=1.0,
    )
    parser.add_argument("--max-latency-samples", type=_positive_int, default=20_000)
    parser.add_argument(
        "--max-rss-growth-mib",
        type=_nonnegative_float,
        help="optional hard per-process/generation RSS-growth gate",
    )
    parser.add_argument(
        "--max-fd-growth",
        type=_nonnegative_int,
        help="optional hard per-process/generation file-descriptor growth gate",
    )
    parser.add_argument(
        "--max-cgroup-throttled-seconds",
        type=_nonnegative_float,
        help="optional hard shared-cgroup throttled-time delta gate",
    )
    return parser


def _checkpoint(
    path: Path,
    report: dict[str, Any],
    budget: benchmark.ResourceBudget,
) -> None:
    report["budget"] = budget.as_dict()
    report["updated_unix_ns"] = time.time_ns()
    benchmark.load_test.atomic_json_dump(path, _json_value(report))


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    run_id = args.run_id or uuid.uuid4()
    if args.store is not None:
        args.store = args.store.expanduser().resolve()
    if args.ids is not None:
        args.ids = args.ids.expanduser().resolve()
    if args.source_embeddings is not None:
        args.source_embeddings = args.source_embeddings.expanduser().resolve()
    if args.native_binary is not None:
        args.native_binary = args.native_binary.expanduser().resolve()
    args.results = args.results.expanduser().resolve()
    args.results.mkdir(parents=True, exist_ok=False)
    report_path = args.results / "report.json"
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "revision": REPORT_REVISION,
        "run_id": str(run_id),
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "arguments": _json_value(vars(args)),
        "methodology": {
            "workload": (
                "closed-loop bounded short Get, long StreamGet/ScanRecords, "
                "and canary mutations"
            ),
            "mutation_state": (
                "absent -> insert -> update -> upsert -> delete -> absent"
            ),
            "retry": "same deterministic UUID and identical bytes",
            "transfer_accounting": (
                "dominant IDs/embeddings/location payload; protobuf and HTTP/2 "
                "framing excluded; one maximum batch is charged before stream "
                "open and each subsequent receive is charged before requesting "
                "it, so a time-rotated large logical input is bounded without "
                "precharging its unconsumed tail"
            ),
            "latency": (
                "client-observed synchronous RPC round trip; fairness uses "
                "successful samples only and matched concurrent windows over "
                "one exact query corpus"
            ),
            "safety": (
                "canary IDs are proven absent and restored absent; WAL/tombstone "
                "history remains, so use a disposable collection clone"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "throwdb": throwdb.__version__,
            "cpu_count": os.cpu_count(),
        },
        "server": {"generations": []},
        "mutation_cycles": [],
        "mutation_schedule": {
            "periodic_enabled": True,
            "skipped_intervals": 0,
            "skip_reasons": {},
            "disabled_reason": None,
        },
        "restart_cycles": [],
        "streaming_admission": [],
    }
    budget = benchmark.ResourceBudget(
        max_seconds=args.max_seconds,
        max_bytes=int(args.max_transfer_gib * GIB),
        max_client_memory_bytes=int(args.max_client_memory_mib * MIB),
    )
    mutation_budget = MutationBudget(args.max_mutations)
    metrics = OperationMetrics(args.max_latency_samples, args.seed)
    errors = ErrorMetrics()
    supervisor: LocalSupervisor | None = None
    remote: Any | None = None
    read_clients: list[Any] = []
    read_client_locks: list[threading.Lock] = []
    streaming: StreamingWorkload | None = None
    stream_config: StreamingConfiguration | None = None
    fairness_baseline_report: dict[str, Any] | None = None
    generation_holder = {"generation": 0, "pid": args.server_pid}

    _checkpoint(report_path, report, budget)
    try:
        if args.endpoint is None:
            supervisor = LocalSupervisor(args, args.results)
            supervisor.start(
                min(args.server_start_timeout, budget.remaining_seconds())
            )
            endpoint = supervisor.endpoint
            generation_holder.update(
                generation=supervisor.generation,
                pid=supervisor.pid,
            )
        else:
            endpoint = args.endpoint
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "phase": "server_start",
        }
        report["finished_unix_ns"] = time.time_ns()
        if supervisor is not None:
            with contextlib.suppress(BaseException):
                supervisor.stop()
            report["server"]["generations"] = supervisor.generations
        _checkpoint(report_path, report, budget)
        raise
    monitor = MetricsStream(
        args.results / "metrics.jsonl",
        interval_seconds=args.telemetry_interval_seconds,
        pid_provider=lambda: (
            (supervisor.generation, supervisor.pid)
            if supervisor is not None
            else (
                int(generation_holder["generation"]),
                generation_holder["pid"],
            )
        ),
    )
    try:
        monitor.start()
    except BaseException as error:
        if supervisor is not None:
            with contextlib.suppress(BaseException):
                supervisor.stop()
            report["server"]["generations"] = supervisor.generations
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "phase": "monitor_start",
        }
        report["finished_unix_ns"] = time.time_ns()
        _checkpoint(report_path, report, budget)
        raise
    started = time.monotonic()
    stop_reason = "unknown"
    fatal: BaseException | None = None
    canaries: np.ndarray | None = None
    try:
        remote = _connect(endpoint, budget, args.rpc_timeout)
        description = remote.describe()
        if not description.writable or not description.durable_idempotency:
            raise SoakConfigurationError(
                "soak requires writable server with durable idempotency"
            )
        prefixes = benchmark.parse_prefixes(
            args.prefix_dimensions, description.schema.dimension
        )
        stream_prefixes = benchmark.parse_prefixes(
            args.stream_prefix_dimensions,
            description.schema.dimension,
        )
        if len(stream_prefixes) != 1:
            raise SoakConfigurationError(
                "--stream-prefix-dimensions must select exactly one prefix"
            )
        stream_config = StreamingConfiguration(
            stream_get_workers=args.stream_get_workers,
            scan_records_workers=args.scan_records_workers,
            stream_get_rows=args.stream_get_rows,
            scan_records_rows=args.scan_records_rows,
            batch_rows=args.stream_batch_rows,
            max_batch_payload_bytes=int(args.stream_max_batch_mib * MIB),
            prefix=stream_prefixes[0],
            attempt_seconds=args.stream_attempt_seconds,
            progress_timeout_seconds=args.stream_progress_timeout_seconds,
            stop_timeout_seconds=args.stream_stop_timeout_seconds,
            rpc_timeout_seconds=args.rpc_timeout,
            mutation_rows=args.mutation_rows,
            with_locations=args.stream_locations,
            source_oracle=args.stream_source_oracle,
        )
        if stream_config.stream_get_workers and not description.stream_get:
            raise SoakConfigurationError(
                "server does not advertise StreamGet"
            )
        if stream_config.scan_records_workers and not description.scan_records:
            raise SoakConfigurationError(
                "server does not advertise ScanRecords"
            )
        if stream_config.worker_count:
            availability = description.availability
            if availability is None or description.admission_stats is None:
                raise SoakConfigurationError(
                    "streaming acceptance requires availability/admission "
                    "diagnostics"
                )
            if stream_config.worker_count > availability.max_concurrent_scans:
                raise SoakConfigurationError(
                    "configured streaming workers exceed the server's "
                    "max_concurrent_scans"
                )
            if (
                stream_config.worker_count + args.get_workers + 1
                > availability.max_concurrent_calls
            ):
                raise SoakConfigurationError(
                    "streaming plus Get workers and one control/mutation RPC "
                    "exceed max_concurrent_calls"
                )
            if (
                stream_config.attempt_seconds * 1000
                >= availability.scan_max_duration_millis
            ):
                raise SoakConfigurationError(
                    "stream attempt duration must remain below the server "
                    "scan watchdog"
                )
        largest_response = (
            max(args.get_batch_sizes)
            * max(prefixes)
            * np.dtype(description.schema.dtype).itemsize
        )
        budget.check(
            extra_memory_bytes=(
                args.max_inflight_gets * 3 * largest_response
                + stream_config.worker_count
                * 3
                * stream_config.max_batch_payload_bytes
                + stream_config.stream_get_workers
                * stream_config.batch_rows
                * 16
            )
        )
        monitor.emit("connected", remote)
        report["server"]["initial"] = remote_diagnostics(remote)

        population_sample = (
            load_population(
                args.ids,
                args.population_rows,
                args.seed,
                budget,
            )
            if args.ids is not None
            else discover_population(
                remote,
                requested=args.population_rows,
                batch_rows=args.population_scan_batch_rows,
                budget=budget,
            )
        )
        population = population_sample.ids
        source_embeddings: np.ndarray | None = None
        if args.source_embeddings is not None:
            source_embeddings = np.load(
                args.source_embeddings,
                mmap_mode="r",
                allow_pickle=False,
            )
            ids_source = np.load(args.ids, mmap_mode="r", allow_pickle=False)
            expected_dtype = np.dtype(description.schema.dtype)
            if (
                source_embeddings.ndim != 2
                or source_embeddings.shape[0] != ids_source.shape[0]
                or source_embeddings.shape[1] != description.schema.dimension
                or source_embeddings.dtype != expected_dtype
            ):
                raise SoakConfigurationError(
                    "--source-embeddings must be a row-aligned "
                    f"{expected_dtype}[N,{description.schema.dimension}] .npy"
                )
            if population_sample.source_rows is None:
                raise SoakInvariantError("source oracle lost sampled row indices")
        report["population"] = {
            "source": "ids_npy" if args.ids is not None else "bounded_scan",
            "rows": int(population.size),
            "digest_sha256": hashlib.sha256(
                memoryview(np.asarray(population, dtype="<u8")).cast("B")
            ).hexdigest(),
        }
        report["base_read_oracle"] = {
            "mode": (
                "source_npy_bit_exact"
                if source_embeddings is not None
                else "transport_stability_only"
            ),
            "source_embeddings": (
                str(args.source_embeddings)
                if args.source_embeddings is not None
                else None
            ),
            "bit_exact_all_get_rows": source_embeddings is not None,
        }
        shadow_id = np.asarray(
            [population[args.seed % population.size]],
            dtype=np.uint64,
        )
        shadow_wire = _wire_get(
            1,
            description.schema.dimension,
            np.dtype(description.schema.dtype).itemsize,
        )
        budget.reserve(
            shadow_wire,
            extra_memory_bytes=2 * description.schema.dimension
            * np.dtype(description.schema.dtype).itemsize,
        )
        _set_timeout(remote, budget, args.rpc_timeout)
        shadow_payload = remote.get(shadow_id)
        if shadow_payload.shape != (1, description.schema.dimension):
            raise SoakInvariantError("shadow canary returned an invalid shape")
        shadow_payload = np.array(shadow_payload, copy=True, order="C")
        report["shadow_canary"] = {
            "id": int(shadow_id[0]),
            "dimension": description.schema.dimension,
            "dtype": description.schema.dtype,
            "payload_sha256": hashlib.sha256(
                memoryview(shadow_payload).cast("B")
            ).hexdigest(),
            "method": (
                "baseline row captured before mixed workload and checked "
                "bit-for-bit in every Get"
            ),
        }
        canaries = find_absent_canaries(
            remote,
            start=args.canary_id_start,
            count=args.mutation_rows,
            budget=budget,
            rpc_timeout=args.rpc_timeout,
        )
        budget.reserve(
            _wire_get(
                canaries.size,
                description.schema.dimension,
                np.dtype(description.schema.dtype).itemsize,
            ),
            extra_memory_bytes=canaries.size
            * description.schema.dimension
            * np.dtype(description.schema.dtype).itemsize,
        )
        report["canaries"] = {
            "count": int(canaries.size),
            "ids": [int(value) for value in canaries],
            "initial_state": "absent",
        }

        # Guarantee CRUD/replay coverage before a long read-only tail can use
        # the duration or transfer budget.
        initial_cycle = run_mutation_cycle(
            remote, canaries, sequence=1, seed=args.seed,
            run_id=run_id,
            budget=budget, mutation_budget=mutation_budget,
            metrics=metrics, errors=errors, rpc_timeout=args.rpc_timeout,
            primary_attempts=args.mutation_primary_attempts,
        )
        report["mutation_cycles"].append(initial_cycle)
        monitor.emit("initial_mutation_cycle", remote)

        scan_result = scan_acceptance(
            remote, rows=args.scan_rows, batch_rows=args.scan_batch_rows,
            budget=budget, rpc_timeout=args.rpc_timeout,
        )
        report["scan_ids"] = scan_result

        fairness_batch_size = min(args.get_batch_sizes)
        fairness_prefix = min(prefixes)
        fairness_queries, fairness_query_digest = build_fairness_queries(
            remote,
            population=population,
            population_source_rows=population_sample.source_rows,
            source_embeddings=source_embeddings,
            shadow_id=shadow_id,
            calls=args.fairness_baseline_get_calls,
            batch_size=fairness_batch_size,
            prefix=fairness_prefix,
            seed=args.seed,
            budget=budget,
        )
        fairness_baseline_label = (
            f"baseline_get_b{fairness_batch_size}_p{fairness_prefix}"
        )
        fairness_mixed_label = (
            f"mixed_fairness_get_b{fairness_batch_size}_p{fairness_prefix}"
        )
        fairness_baseline_report = run_get_fairness_window(
            remote,
            queries=fairness_queries,
            query_digest_sha256=fairness_query_digest,
            label=fairness_baseline_label,
            shadow_payload=shadow_payload,
            prefix=fairness_prefix,
            workers=args.get_workers,
            max_inflight=args.max_inflight_gets,
            budget=budget,
            rpc_timeout=args.rpc_timeout,
            retries=args.get_retries,
            metrics=metrics,
            errors=errors,
        )
        fairness_baseline_report["mixed_label"] = fairness_mixed_label
        report["fairness_baseline"] = fairness_baseline_report

        shadow_population_index = args.seed % population.size
        stream_population = np.array(population, dtype=np.uint64, copy=True)
        stream_population[
            [0, shadow_population_index]
        ] = stream_population[
            [shadow_population_index, 0]
        ]
        stream_source_rows = (
            np.array(
                population_sample.source_rows,
                dtype=np.int64,
                copy=True,
            )
            if population_sample.source_rows is not None
            else None
        )
        if stream_source_rows is not None:
            stream_source_rows[
                [0, shadow_population_index]
            ] = stream_source_rows[
                [shadow_population_index, 0]
            ]
        stream_oracle_embeddings: np.ndarray | None = None
        stream_oracle_source_rows: np.ndarray | None = None
        if stream_config.stream_get_workers:
            oracle_rows = min(
                args.stream_oracle_rows,
                int(stream_population.size),
            )
            stream_population = np.array(
                stream_population[:oracle_rows],
                dtype=np.uint64,
                copy=True,
            )
            selected_source_rows = (
                np.array(
                    stream_source_rows[:oracle_rows],
                    dtype=np.int64,
                    copy=True,
                )
                if stream_source_rows is not None
                else None
            )
            oracle_batch_rows = min(
                oracle_rows,
                args.server_max_ids_per_request or 64,
                64,
            )
            stream_oracle_embeddings = capture_stream_get_oracle(
                remote,
                ids=stream_population,
                prefix=stream_config.prefix,
                batch_rows=oracle_batch_rows,
                budget=budget,
                rpc_timeout=args.rpc_timeout,
            )
            source_verified = False
            if args.stream_source_oracle:
                assert source_embeddings is not None
                assert selected_source_rows is not None
                expected_oracle = np.asarray(
                    source_embeddings[
                        selected_source_rows,
                        : stream_config.prefix,
                    ],
                    dtype=np.dtype(description.schema.dtype),
                    order="C",
                )
                _assert_bits(
                    stream_oracle_embeddings,
                    expected_oracle,
                    "captured StreamGet unary/source oracle",
                )
                source_verified = True
            stream_oracle_source_rows = np.arange(
                oracle_rows,
                dtype=np.int64,
            )
            report["stream_get_oracle"] = {
                "kind": (
                    "source_npy_verified_unary_capture"
                    if source_verified
                    else "unary_capture"
                ),
                "rows": oracle_rows,
                "batch_rows": oracle_batch_rows,
                "prefix_dimensions": stream_config.prefix,
                "payload_sha256": hashlib.sha256(
                    memoryview(stream_oracle_embeddings).cast("B")
                ).hexdigest(),
                "every_stream_get_row_checked": True,
            }
        else:
            report["stream_get_oracle"] = {
                "kind": "disabled",
                "rows": 0,
                "every_stream_get_row_checked": False,
            }
        streaming = StreamingWorkload(
            config=stream_config,
            population=stream_population,
            source_rows=stream_oracle_source_rows,
            source_embeddings=stream_oracle_embeddings,
            shadow_payload=shadow_payload,
            dimension=description.schema.dimension,
            dtype=description.schema.dtype,
            budget=budget,
            operations=metrics,
            errors=errors,
        )
        if streaming.enabled:
            current_generation = (
                supervisor.generation if supervisor is not None else 0
            )
            admission_before = remote.describe().admission_stats
            if admission_before is None:
                raise SoakConfigurationError(
                    "streaming acceptance requires admission statistics"
                )
            streaming.start_generation(endpoint, current_generation)
            opened_admission = wait_for_scan_admission(
                remote,
                minimum_admitted_scans=(
                    admission_before.admitted_scans
                    + stream_config.worker_count
                ),
                timeout_seconds=stream_config.progress_timeout_seconds,
            )
            opened_admission.update(
                event="generation_streams_open",
                generation=current_generation,
            )
            report["streaming_admission"].append(opened_admission)
            if opened_admission["status"] != "passed":
                raise SoakInvariantError(
                    "server admission did not accept every opened stream"
                )
            monitor.emit("streams_open", remote)
            streaming.release_and_wait_for_progress()
            monitor.emit("streams_progressing", remote)
        fairness_mixed_report = run_get_fairness_window(
            remote,
            queries=fairness_queries,
            query_digest_sha256=fairness_query_digest,
            label=fairness_mixed_label,
            shadow_payload=shadow_payload,
            prefix=fairness_prefix,
            workers=args.get_workers,
            max_inflight=args.max_inflight_gets,
            budget=budget,
            rpc_timeout=args.rpc_timeout,
            retries=args.get_retries,
            metrics=metrics,
            errors=errors,
        )
        report["fairness_mixed"] = fairness_mixed_report
        if (
            fairness_mixed_report["query_ids_digest_sha256"]
            != fairness_baseline_report["query_ids_digest_sha256"]
        ):
            raise SoakInvariantError(
                "fairness windows did not use the same query corpus"
            )
        _checkpoint(report_path, report, budget)

        for _ in range(args.get_workers):
            read_clients.append(_connect(endpoint, budget, args.rpc_timeout))
            read_client_locks.append(threading.Lock())

        rng = np.random.default_rng(args.seed)
        inflight: set[Future[bool]] = set()
        submitted_gets = 0
        completed_gets = 0
        successful_gets = 0
        restart_count = 0
        mutation_sequence = 2
        periodic_mutations_enabled = True
        next_mutation_get = args.mutation_every_gets
        next_restart_get = args.restart_every_gets

        def consume(done: set[Future[bool]]) -> None:
            nonlocal completed_gets, successful_gets
            for future in done:
                successful_gets += int(future.result())
                completed_gets += 1
            if streaming is not None:
                streaming.check()

        def drain() -> None:
            nonlocal inflight
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                consume(done)

        with ThreadPoolExecutor(max_workers=args.get_workers) as executor:
            while True:
                if streaming is not None:
                    streaming.check()
                    if streaming.budget_exhausted:
                        stop_reason = "max_transfer_or_memory"
                        break
                if budget.remaining_seconds() <= 0:
                    stop_reason = "max_seconds"
                    break
                if args.max_get_calls and submitted_gets >= args.max_get_calls:
                    stop_reason = "max_get_calls"
                    break
                if errors.total > args.max_errors:
                    stop_reason = "max_errors"
                    break

                while len(inflight) >= args.max_inflight_gets:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    consume(done)
                batch_size = args.get_batch_sizes[
                    submitted_gets % len(args.get_batch_sizes)
                ]
                prefix = prefixes[
                    (submitted_gets // len(args.get_batch_sizes)) % len(prefixes)
                ]
                positions = rng.integers(
                    0, population.size, size=batch_size, dtype=np.int64
                )
                shadow_population_index = args.seed % population.size
                positions[0] = shadow_population_index
                ids = np.asarray(population[positions], dtype=np.uint64, order="C")
                # The first row is a stable shadow oracle captured before the
                # mixed workload. This keeps validation source-independent and
                # turns ID ordering, prefix, and payload corruption into a hard
                # error on every Get class.
                ids[0] = shadow_id[0]
                wire = _wire_get(
                    batch_size, prefix, np.dtype(description.schema.dtype).itemsize
                )
                try:
                    budget.reserve(
                        wire,
                        extra_memory_bytes=3 * batch_size * prefix
                        * np.dtype(description.schema.dtype).itemsize,
                    )
                except benchmark.BenchmarkBudgetExceeded:
                    stop_reason = (
                        "max_seconds"
                        if budget.remaining_seconds() <= 0
                        else "max_transfer_or_memory"
                    )
                    break
                plan = GetPlan(
                    ordinal=submitted_gets,
                    label=f"get_b{batch_size}_p{prefix}",
                    ids=ids,
                    prefix=prefix,
                    wire_bytes=wire,
                    shadow_expected=np.array(
                        shadow_payload[:, :prefix],
                        copy=True,
                        order="C",
                    ),
                    source_expected=(
                        np.array(
                            source_embeddings[
                                population_sample.source_rows[positions],
                                :prefix,
                            ],
                            copy=True,
                            order="C",
                        )
                        if source_embeddings is not None
                        and population_sample.source_rows is not None
                        else None
                    ),
                )
                assert remote is not None
                inflight.add(
                    executor.submit(
                        execute_get_on_client,
                        read_client_locks[
                            submitted_gets % len(read_client_locks)
                        ],
                        read_clients[submitted_gets % len(read_clients)],
                        plan,
                        dimension=description.schema.dimension,
                        itemsize=np.dtype(description.schema.dtype).itemsize,
                        retries=args.get_retries,
                        budget=budget,
                        rpc_timeout=args.rpc_timeout,
                        metrics=metrics,
                        errors=errors,
                    )
                )
                submitted_gets += 1

                if (
                    periodic_mutations_enabled
                    and submitted_gets >= next_mutation_get
                ):
                    remaining_restarts = args.restart_cycles - restart_count
                    if not can_schedule_periodic_mutation(
                        mutation_budget,
                        remaining_restarts=remaining_restarts,
                    ):
                        reason = "restart_and_cleanup_headroom"
                        schedule = report["mutation_schedule"]
                        schedule["skipped_intervals"] += 1
                        reasons = schedule["skip_reasons"]
                        reasons[reason] = reasons.get(reason, 0) + 1
                        if remaining_restarts == 0:
                            periodic_mutations_enabled = False
                            schedule["periodic_enabled"] = False
                            schedule["disabled_reason"] = (
                                "mutation_ceiling_preserves_cleanup_headroom"
                            )
                    else:
                        try:
                            cycle = run_mutation_cycle(
                                remote, canaries, sequence=mutation_sequence,
                                seed=args.seed, run_id=run_id, budget=budget,
                                mutation_budget=mutation_budget, metrics=metrics,
                                errors=errors, rpc_timeout=args.rpc_timeout,
                                primary_attempts=args.mutation_primary_attempts,
                            )
                        except benchmark.BenchmarkBudgetExceeded:
                            reason = "run_budget"
                            schedule = report["mutation_schedule"]
                            schedule["skipped_intervals"] += 1
                            reasons = schedule["skip_reasons"]
                            reasons[reason] = reasons.get(reason, 0) + 1
                        else:
                            report["mutation_cycles"].append(cycle)
                            mutation_sequence += 1
                    next_mutation_get += args.mutation_every_gets

                if (
                    supervisor is not None
                    and restart_count < args.restart_cycles
                    and submitted_gets >= next_restart_get
                ):
                    drain()
                    for read_client in read_clients:
                        read_client.close()
                    read_clients.clear()
                    read_client_locks.clear()
                    restart_mode = (
                        "graceful" if restart_count % 2 == 0 else "sigkill"
                    ) if args.restart_mode == "alternating" else args.restart_mode
                    if streaming is not None and streaming.enabled:
                        if restart_mode == "graceful":
                            streaming.stop_generation("graceful_restart")
                            drained = wait_for_scan_admission(
                                remote,
                                expected_active_scans=0,
                                timeout_seconds=(
                                    stream_config.stop_timeout_seconds
                                ),
                            )
                            drained.update(
                                event="before_graceful_restart_drained",
                                generation=supervisor.generation,
                            )
                            report["streaming_admission"].append(drained)
                            if drained["status"] != "passed":
                                raise SoakInvariantError(
                                    "stream admission did not drain before "
                                    "graceful restart"
                                )
                        else:
                            streaming.wait_until_any_active()
                            forced_active = wait_for_scan_admission(
                                remote,
                                expected_active_scans=1,
                                timeout_seconds=(
                                    stream_config.progress_timeout_seconds
                                ),
                            )
                            forced_active.update(
                                event="before_sigkill_stream_active",
                                generation=supervisor.generation,
                            )
                            report["streaming_admission"].append(
                                forced_active
                            )
                            if forced_active["status"] != "passed":
                                raise SoakInvariantError(
                                    "SIGKILL restart did not observe an active "
                                    "server-side stream lease"
                                )
                    monitor.emit("before_restart", remote)
                    remote, result = restart_replay_probe(
                        remote=remote, supervisor=supervisor,
                        canaries=canaries,
                        shadow_id=shadow_id,
                        shadow_payload=shadow_payload,
                        sequence=100_000 + restart_count,
                        seed=args.seed, run_id=run_id, budget=budget,
                        mutation_budget=mutation_budget, metrics=metrics,
                        errors=errors, rpc_timeout=args.rpc_timeout,
                        start_timeout=args.server_start_timeout,
                        restart_mode=restart_mode,
                        before_restart=(
                            streaming.mark_forced_interruption
                            if (
                                streaming is not None
                                and streaming.enabled
                                and restart_mode == "sigkill"
                            )
                            else None
                        ),
                    )
                    if (
                        streaming is not None
                        and streaming.enabled
                        and restart_mode == "sigkill"
                    ):
                        streaming.stop_generation("sigkill_restart")
                        replacement_drained = wait_for_scan_admission(
                            remote,
                            expected_active_scans=0,
                            timeout_seconds=(
                                stream_config.stop_timeout_seconds
                            ),
                        )
                        replacement_drained.update(
                            event="after_sigkill_replacement_drained",
                            generation=supervisor.generation,
                        )
                        report["streaming_admission"].append(
                            replacement_drained
                        )
                        if replacement_drained["status"] != "passed":
                            raise SoakInvariantError(
                                "replacement server inherited active scan "
                                "admission state"
                            )
                    report["restart_cycles"].append(result)
                    restart_count += 1
                    next_restart_get += args.restart_every_gets
                    endpoint = supervisor.endpoint
                    for _ in range(args.get_workers):
                        read_clients.append(
                            _connect(endpoint, budget, args.rpc_timeout)
                        )
                        read_client_locks.append(threading.Lock())
                    if streaming is not None and streaming.enabled:
                        admission_before = remote.describe().admission_stats
                        if admission_before is None:
                            raise SoakConfigurationError(
                                "streaming acceptance requires admission "
                                "statistics"
                            )
                        streaming.start_generation(
                            endpoint,
                            supervisor.generation,
                        )
                        opened_admission = wait_for_scan_admission(
                            remote,
                            minimum_admitted_scans=(
                                admission_before.admitted_scans
                                + stream_config.worker_count
                            ),
                            timeout_seconds=(
                                stream_config.progress_timeout_seconds
                            ),
                        )
                        opened_admission.update(
                            event="generation_streams_open",
                            generation=supervisor.generation,
                        )
                        report["streaming_admission"].append(
                            opened_admission
                        )
                        if opened_admission["status"] != "passed":
                            raise SoakInvariantError(
                                "stream admission did not recover after restart"
                            )
                        streaming.release_and_wait_for_progress()
                    monitor.emit("after_restart", remote)
                    _checkpoint(report_path, report, budget)
            drain()

        if streaming is not None and streaming.enabled:
            streaming.stop_generation("workload_complete")
            final_stream_drain = wait_for_scan_admission(
                remote,
                expected_active_scans=0,
                timeout_seconds=stream_config.stop_timeout_seconds,
            )
            final_stream_drain.update(
                event="workload_streams_drained",
                generation=(
                    supervisor.generation if supervisor is not None else 0
                ),
            )
            report["streaming_admission"].append(final_stream_drain)
            if final_stream_drain["status"] != "passed":
                raise SoakInvariantError(
                    "stream admission did not drain at workload completion"
                )
        if streaming is not None:
            report["streaming"] = streaming.summary()
        report["workload"] = {
            "stop_reason": stop_reason,
            "submitted_get_calls": submitted_gets,
            "completed_get_calls": completed_gets,
            "successful_get_calls": successful_gets,
            "shadow_checked_get_calls": successful_gets,
            "source_oracle_checked_get_calls": (
                successful_gets if source_embeddings is not None else 0
            ),
            "configured_get_classes": [
                f"get_b{batch}_p{prefix}"
                for batch in args.get_batch_sizes
                for prefix in prefixes
            ],
        }
        work_deadline_exhausted = budget.remaining_seconds() <= 0
        if not work_deadline_exhausted:
            _set_timeout(remote, budget, args.rpc_timeout)
            _expect_missing(
                remote,
                canaries,
                deadline_monotonic=budget.deadline,
            )
            monitor.emit("workload_complete", remote)
        else:
            # Every completed mutation/restart cycle already proves the
            # canaries absent. Do not turn the documented duration-driven stop
            # into a fatal ResourceBudgetExceeded by issuing new diagnostics
            # after the work deadline. The finally block still attempts the
            # separately disclosed best-effort cleanup grace path.
            remote._timeout = min(5.0, args.rpc_timeout)  # type: ignore[attr-defined]
            monitor.emit("workload_deadline_reached")
        report["server"]["final"] = {
            "health": remote.health(),
            **remote_diagnostics(remote),
            "after_work_deadline": work_deadline_exhausted,
        }
    except BaseException as error:
        fatal = error
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if streaming is not None:
            with contextlib.suppress(BaseException):
                streaming.stop_generation("finally_cleanup")
            report["streaming"] = streaming.summary()
        for read_client in read_clients:
            with contextlib.suppress(Exception):
                read_client.close()
        if remote is not None:
            if canaries is not None:
                present: list[int] = []
                probe_error: BaseException | None = None
                probe_deadline = time.monotonic() + min(args.rpc_timeout, 5.0)
                for identifier in np.asarray(canaries, dtype=np.uint64):
                    remaining = probe_deadline - time.monotonic()
                    if remaining <= 0:
                        probe_error = TimeoutError(
                            "final canary probe exceeded its total grace deadline"
                        )
                        break
                    remote._timeout = max(  # type: ignore[attr-defined]
                        0.001,
                        remaining,
                    )
                    try:
                        remote.get(np.asarray([identifier], dtype=np.uint64))
                    except throwdb.RemoteNotFoundError:
                        continue
                    except BaseException as error:
                        probe_error = error
                        break
                    else:
                        present.append(int(identifier))
                report["final_canary_state"] = {
                    "status": (
                        "unknown"
                        if probe_error is not None
                        else ("present" if present else "absent")
                    ),
                    "present_ids": present[:100],
                    "present_ids_truncated": len(present) > 100,
                    "probe_error": (
                        {
                            "type": type(probe_error).__name__,
                            "message": str(probe_error),
                        }
                        if probe_error is not None
                        else None
                    ),
                }
            with contextlib.suppress(Exception):
                remote.close()
        elif canaries is not None:
            report["final_canary_state"] = {
                "status": "unknown",
                "present_ids": [],
                "present_ids_truncated": False,
                "probe_error": {
                    "type": "RemoteUnavailable",
                    "message": "no connected client remained for final probe",
                },
            }
        if supervisor is not None:
            supervisor.stop()
            report["server"]["generations"] = supervisor.generations
        monitor.stop()

    elapsed = time.monotonic() - started
    operation_summary = metrics.summary(elapsed)
    error_summary = errors.summary()
    resource_summary = monitor.summary()
    report["metrics"] = {
        "elapsed_seconds": elapsed,
        "operations": operation_summary,
        "errors": error_summary,
        "resources": resource_summary,
    }
    if fairness_baseline_report is not None:
        fairness = fairness_acceptance(
            operation_summary,
            baseline_label=fairness_baseline_report["label"],
            mixed_label=fairness_baseline_report["mixed_label"],
            minimum_mixed_successes=args.min_mixed_small_get_successes,
            maximum_p99_ratio=args.max_small_get_p99_ratio,
        )
    else:
        fairness = {
            "evaluable": False,
            "passed": False,
            "reason": "fairness baseline did not complete",
        }
    report["fairness"] = fairness
    streaming_report = report.get("streaming", {})
    streaming_modes = streaming_report.get("modes", {})
    stream_mode_gates: dict[str, Any] = {}
    for mode in ("stream_get", "scan_records"):
        configured_workers = (
            args.stream_get_workers
            if mode == "stream_get"
            else args.scan_records_workers
        )
        values = streaming_modes.get(mode)
        if configured_workers == 0:
            stream_mode_gates[mode] = {
                "enabled": False,
                "passed": True,
                "reason": "disabled by zero workers",
            }
            continue
        if values is None:
            stream_mode_gates[mode] = {
                "enabled": True,
                "passed": False,
                "reason": "no streaming metrics were produced",
            }
            continue
        source_gate = (
            mode != "stream_get"
            or int(values["source_verified_rows"]) == int(values["rows"])
        )
        mode_passed = (
            int(values["attempts_started"]) > 0
            and int(values["attempts_completed"]) > 0
            and int(values["batches"]) >= args.min_stream_batches_per_mode
            and int(values["rows"]) > 0
            and int(values["unexpected_failures"]) == 0
            and int(values["workers_started"])
            == int(values["workers_joined"])
            and float(values["max_inter_batch_gap_seconds"])
            <= args.stream_progress_timeout_seconds
            and float(values["rows_per_worker_active_second"])
            >= args.min_stream_rows_per_second
            and source_gate
        )
        stream_mode_gates[mode] = {
            "enabled": True,
            "passed": mode_passed,
            "minimum_batches": args.min_stream_batches_per_mode,
            "minimum_rows_per_worker_active_second": (
                args.min_stream_rows_per_second
            ),
            "maximum_inter_batch_gap_seconds": (
                args.stream_progress_timeout_seconds
            ),
            "source_oracle_required": (
                mode == "stream_get"
            ),
            "source_oracle_passed": source_gate,
        }
    stream_generation_events = streaming_report.get("generations", [])
    expected_stream_generations = (
        len(report.get("restart_cycles", [])) + 1
        if stream_config is not None and stream_config.worker_count
        else 0
    )
    generation_stream_progress = (
        len(stream_generation_events) == expected_stream_generations
        and all(
            bool(event.get("joined"))
            and bool(event.get("every_worker_progressed"))
            for event in stream_generation_events
        )
    )
    sigkill_restarts = sum(
        item.get("restart_mode") == "sigkill"
        for item in report.get("restart_cycles", [])
    )
    observed_restart_interruptions = sum(
        int(values.get("expected_restart_interruptions", 0))
        for values in streaming_modes.values()
    )
    forced_generation_events = [
        event
        for event in stream_generation_events
        if event.get("expected_forced_interruption")
    ]
    all_marked_workers_interrupted = (
        len(forced_generation_events) == sigkill_restarts
        and all(
            bool(event.get("expected_interruption_workers"))
            and bool(event.get("all_marked_workers_interrupted"))
            for event in forced_generation_events
        )
    )
    stream_admission_events = report.get("streaming_admission", [])
    stream_admission_passed = (
        (
            stream_config is None
            or stream_config.worker_count == 0
            or bool(stream_admission_events)
        )
        and all(
            item.get("status") == "passed"
            for item in stream_admission_events
        )
    )
    report["streaming_gates"] = {
        "modes": stream_mode_gates,
        "expected_generation_count": expected_stream_generations,
        "observed_generation_count": len(stream_generation_events),
        "every_generation_progressed_and_joined": generation_stream_progress,
        "sigkill_restarts": sigkill_restarts,
        "observed_expected_restart_interruptions": (
            observed_restart_interruptions
        ),
        "forced_restart_interruption_passed": (
            stream_config is None
            or stream_config.worker_count == 0
            or (
                observed_restart_interruptions >= sigkill_restarts
                and all_marked_workers_interrupted
            )
        ),
        "admission_events_passed": stream_admission_passed,
    }
    report["mutation_budget"] = mutation_budget.as_dict()
    configured_classes = set(
        report.get("workload", {}).get("configured_get_classes", [])
    )
    successful_classes = {
        name
        for name, values in operation_summary.items()
        if name.startswith("get_") and values["successes"] > 0
    }
    cycles = report.get("mutation_cycles", [])
    mutation_names = {
        operation["name"]
        for cycle in cycles
        for operation in cycle.get("operations", [])
    }
    restarts = report.get("restart_cycles", [])
    generations = report.get("server", {}).get("generations", [])
    expected_restart_modes = {
        (
            "graceful" if index % 2 == 0 else "sigkill"
        ) if args.restart_mode == "alternating" else args.restart_mode
        for index in range(args.restart_cycles)
    }
    observed_restart_modes = {
        str(item.get("restart_mode")) for item in restarts
    }
    generation_exits_expected = all(
        (
            item.get("stop_mode") == "sigkill"
            and item.get("returncode") in {-9, 137}
        )
        or (
            item.get("stop_mode") == "graceful"
            and item.get("returncode") == 0
        )
        for item in generations
    )
    growth_rows = list(resource_summary["growth"].values())
    observed_rss_growth = [
        int(item["rss_growth_bytes"])
        for item in growth_rows
        if isinstance(item.get("rss_growth_bytes"), int)
    ]
    observed_fd_growth = [
        int(item["fd_growth"])
        for item in growth_rows
        if isinstance(item.get("fd_growth"), int)
    ]
    throttled_seconds = resource_summary["cgroup_cpu_delta"].get(
        "throttled_seconds"
    )
    resource_gates = {
        "max_rss_growth_mib": args.max_rss_growth_mib,
        "observed_max_rss_growth_bytes": (
            max(observed_rss_growth) if observed_rss_growth else None
        ),
        "max_fd_growth": args.max_fd_growth,
        "observed_max_fd_growth": (
            max(observed_fd_growth) if observed_fd_growth else None
        ),
        "max_cgroup_throttled_seconds": args.max_cgroup_throttled_seconds,
        "observed_cgroup_throttled_seconds": throttled_seconds,
    }
    report["resource_gates"] = resource_gates
    rss_gate_passed = args.max_rss_growth_mib is None or (
        bool(observed_rss_growth)
        and max(observed_rss_growth) <= args.max_rss_growth_mib * MIB
    )
    fd_gate_passed = args.max_fd_growth is None or (
        bool(observed_fd_growth)
        and max(observed_fd_growth) <= args.max_fd_growth
    )
    cgroup_gate_passed = args.max_cgroup_throttled_seconds is None or (
        isinstance(throttled_seconds, (int, float))
        and throttled_seconds <= args.max_cgroup_throttled_seconds
    )
    final_server = report.get("server", {}).get("final", {})
    final_admission = final_server.get("admission_stats") or {}
    final_server_healthy = bool(
        final_server.get("health", {}).get("serving")
    )
    initial_engine = (
        report.get("server", {}).get("initial", {}).get("engine_stats")
        or {}
    )
    final_engine = final_server.get("engine_stats") or {}
    engine_stats_available = bool(initial_engine) and bool(final_engine)
    engine_invariants_passed = bool(
        engine_stats_available
        and int(final_engine.get("committed_sequence", -1))
        >= int(initial_engine.get("committed_sequence", 0))
        and int(final_engine.get("materialized_sequence", -1))
        >= int(initial_engine.get("materialized_sequence", 0))
        and int(final_engine.get("materialized_sequence", -1))
        <= int(final_engine.get("committed_sequence", -2))
        and int(final_engine.get("filesystem_available_bytes", -1))
        >= int(final_engine.get("mutation_disk_reserve_bytes", 0))
    )
    maintenance_publication_evidence = bool(
        engine_stats_available
        and (
            int(final_engine.get("materialized_sequence", 0))
            > int(initial_engine.get("materialized_sequence", 0))
            or int(final_engine.get("delta_run_count", 0))
            != int(initial_engine.get("delta_run_count", 0))
            or int(final_engine.get("total_delta_run_bytes", 0))
            != int(initial_engine.get("total_delta_run_bytes", 0))
            or int(final_engine.get("frozen_wal_count", 0))
            != int(initial_engine.get("frozen_wal_count", 0))
        )
    )
    maintenance_progress_gate_passed = (
        not args.require_maintenance_progress
        or maintenance_publication_evidence
    )
    report["maintenance"] = {
        "engine_stats_available": engine_stats_available,
        "engine_invariants_passed": engine_invariants_passed,
        "publication_evidence": maintenance_publication_evidence,
        "publication_evidence_required": args.require_maintenance_progress,
        "progress_gate_passed": maintenance_progress_gate_passed,
        "initial": initial_engine,
        "final": final_engine,
        "note": (
            "maintenance_active is instantaneous; progress is inferred from "
            "persistent materialization/run/WAL counters"
        ),
    }
    final_admission_drained = (
        not bool(final_admission.get("terminal_faulted", True))
        and not bool(final_admission.get("draining", True))
        and not bool(final_admission.get("admission_closed", True))
        and int(final_admission.get("active_scans", -1)) == 0
        and int(final_admission.get("active_unary_calls", -1)) == 0
        and int(final_admission.get("scan_estimated_bytes", -1)) == 0
        and int(final_admission.get("unary_estimated_bytes", -1)) == 0
    )
    source_oracle_required = args.source_embeddings is not None
    source_oracle_verified = (
        source_oracle_required
        and report.get("workload", {}).get(
            "source_oracle_checked_get_calls", 0
        ) > 0
        and configured_classes <= successful_classes
    )
    acceptance = {
        "no_fatal_error": fatal is None,
        "errors_within_limit": error_summary["total"] <= args.max_errors,
        "all_get_classes_exercised": configured_classes <= successful_classes,
        "shadow_canary_verified": (
            report.get("workload", {}).get("shadow_checked_get_calls", 0) > 0
        ),
        "base_read_correctness_gate_passed": (
            not source_oracle_required or source_oracle_verified
        ),
        "short_get_fairness_gate_passed": fairness["passed"],
        "stream_modes_passed": all(
            item["passed"] for item in stream_mode_gates.values()
        ),
        "stream_generations_progressed_and_joined": (
            generation_stream_progress
        ),
        "stream_forced_restart_interruption_passed": (
            report["streaming_gates"][
                "forced_restart_interruption_passed"
            ]
        ),
        "stream_admission_lifecycle_passed": stream_admission_passed,
        "maintenance_engine_invariants_passed": engine_invariants_passed,
        "maintenance_progress_gate_passed": (
            maintenance_progress_gate_passed
        ),
        "crud_operations_exercised": mutation_names
        >= {"insert", "update", "upsert", "delete"},
        "all_mutation_cycles_restored": all(
            cycle.get("logical_absent_state_restored", False)
            for cycle in cycles
            if cycle.get("status") != "skipped"
        ),
        "final_canary_state_absent": (
            report.get("final_canary_state", {}).get("status") == "absent"
        ),
        "requested_restarts_completed": len(restarts) == args.restart_cycles,
        "requested_restart_modes_exercised": (
            expected_restart_modes <= observed_restart_modes
        ),
        "restart_replay_verified": all(
            item.get("acknowledged_insert_replayed_after_restart", False)
            for item in restarts
        ),
        "recovery_within_limit": all(
            item.get("recovery_seconds", float("inf"))
            <= args.max_recovery_seconds
            for item in restarts
        ),
        "scan_passed_or_disabled": report.get("scan_ids", {}).get("status")
        in {"passed", "skipped"},
        "no_resource_diagnostic_errors": (
            report["metrics"]["resources"]["diagnostic_error_count"] == 0
        ),
        "resource_monitor_healthy": (
            report["metrics"]["resources"]["monitor_error_count"] == 0
            and report["metrics"]["resources"]["sample_count"] >= 2
        ),
        "rss_growth_gate_passed": rss_gate_passed,
        "fd_growth_gate_passed": fd_gate_passed,
        "cgroup_throttling_gate_passed": cgroup_gate_passed,
        "final_server_health_serving": final_server_healthy,
        "final_admission_drained": final_admission_drained,
        "server_generation_exits_expected": generation_exits_expected,
    }
    acceptance["base_read_correctness_required"] = source_oracle_required
    acceptance["base_read_correctness_verified"] = source_oracle_verified
    informational = {
        "base_read_correctness_required",
        "base_read_correctness_verified",
    }
    acceptance["passed"] = all(
        value for key, value in acceptance.items() if key not in informational
    )
    report["acceptance"] = acceptance
    report["status"] = "complete" if acceptance["passed"] else "failed"
    report["finished_unix_ns"] = time.time_ns()
    _checkpoint(report_path, report, budget)
    if fatal is not None:
        raise fatal
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(f"soak failed: {error}", file=sys.stderr)
        return 1
    log(f"soak report: {args.results / 'report.json'}")
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
