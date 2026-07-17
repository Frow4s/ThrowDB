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
REPORT_REVISION = 1
DEFAULT_SEED = 20260717
GIB = 1 << 30
MIB = 1 << 20
UINT64_MAX = (1 << 64) - 1


class SoakConfigurationError(ValueError):
    pass


class SoakInvariantError(RuntimeError):
    pass


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
    samples: dict[int, tuple[int, int]] = field(default_factory=dict)


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
            item.samples[ordinal] = (priority, latency_ns)
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
            latency = (
                benchmark.load_test.latency_metrics(values) if values.size else {}
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
    for attempt in range(primary_attempts):
        attempts += 1
        if attempt:
            _set_timeout(remote, budget, rpc_timeout)
            _assert_idempotency_domain(
                remote,
                idempotency_domain,
                operation_id=operation_id,
            )
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
            errors.record(f"mutation_{name}", error)
            if attempt + 1 == primary_attempts:
                raise
            continue
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
        idempotency_domain,
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
    if args.max_inflight_gets < args.get_workers:
        raise SoakConfigurationError(
            "--max-inflight-gets must be >= --get-workers"
        )
    if args.get_retries > 99:
        raise SoakConfigurationError("--get-retries must be at most 99")
    if args.mutation_primary_attempts > 50:
        raise SoakConfigurationError(
            "--mutation-primary-attempts must be at most 50"
        )
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
    parser.add_argument("--server-max-ids-per-request", type=_positive_int)
    parser.add_argument("--server-max-scan-batch-rows", type=_positive_int)
    parser.add_argument("--server-maintenance-interval-seconds", type=_positive_int)
    parser.add_argument("--server-no-auto-maintenance", action="store_true")

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
            "workload": "closed-loop bounded mixed Get and canary mutations",
            "mutation_state": (
                "absent -> insert -> update -> upsert -> delete -> absent"
            ),
            "retry": "same deterministic UUID and identical bytes",
            "transfer_accounting": (
                "dominant IDs/embeddings/location payload; protobuf and HTTP/2 "
                "framing excluded"
            ),
            "latency": "client-observed synchronous RPC round trip",
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
        "restart_cycles": [],
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
    generation_holder = {"generation": 0, "pid": args.server_pid}

    if args.endpoint is None:
        supervisor = LocalSupervisor(args, args.results)
        supervisor.start(min(args.server_start_timeout, budget.remaining_seconds()))
        endpoint = supervisor.endpoint
        generation_holder.update(generation=supervisor.generation, pid=supervisor.pid)
    else:
        endpoint = args.endpoint
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
    except Exception:
        if supervisor is not None:
            supervisor.stop()
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
        largest_response = (
            max(args.get_batch_sizes)
            * max(prefixes)
            * np.dtype(description.schema.dtype).itemsize
        )
        budget.check(
            extra_memory_bytes=args.max_inflight_gets * 3 * largest_response
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
        next_mutation_get = args.mutation_every_gets
        next_restart_get = args.restart_every_gets

        def consume(done: set[Future[bool]]) -> None:
            nonlocal completed_gets, successful_gets
            for future in done:
                successful_gets += int(future.result())
                completed_gets += 1

        def drain() -> None:
            nonlocal inflight
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                consume(done)

        with ThreadPoolExecutor(max_workers=args.get_workers) as executor:
            while True:
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

                if submitted_gets >= next_mutation_get:
                    remaining_restarts = args.restart_cycles - restart_count
                    if not can_schedule_periodic_mutation(
                        mutation_budget,
                        remaining_restarts=remaining_restarts,
                    ):
                        cycle = {
                            "status": "skipped",
                            "reason": "restart_and_cleanup_headroom",
                        }
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
                            cycle = {"status": "skipped", "reason": "run_budget"}
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
                    monitor.emit("before_restart", remote)
                    restart_mode = (
                        "graceful" if restart_count % 2 == 0 else "sigkill"
                    ) if args.restart_mode == "alternating" else args.restart_mode
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
                    monitor.emit("after_restart", remote)
                    _checkpoint(report_path, report, budget)
            drain()

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
    final_admission_drained = (
        not bool(final_admission.get("terminal_faulted", True))
        and not bool(final_admission.get("draining", True))
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
