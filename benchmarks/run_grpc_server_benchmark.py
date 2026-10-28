#!/usr/bin/env python3
"""Bounded correctness and load benchmark for the ThrowDB gRPC server.

The default mode launches the public tune-aware server on loopback.  Passing
``--endpoint`` measures an already-running server instead.  The benchmark is
read-only unless ``--crud-rows`` is explicitly non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import queue
import resource
import stat
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

import throwdb
from throwdb._calibration_cache import prepare_cold_payload_cache


# Keep the benchmark independently executable from the source tree while
# reusing the already-tested, privilege-free host telemetry helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_load_test as load_test  # noqa: E402


REPORT_FORMAT = "throwdb-grpc-server-benchmark"
REPORT_REVISION = 3
DEFAULT_SEED = 20260715
GIB = 1 << 30
MIB = 1 << 20
UINT64_MASK = (1 << 64) - 1
CACHE_BACKING_RATIO_MAX = 0.10
CACHE_STORAGE_RATIO_MIN = 0.50
_LEGACY_LOCAL_SERVER_STOP_TIMEOUT_SECONDS = 10.0
_LOCAL_SERVER_SHUTDOWN_TIMEOUT_MARGIN_SECONDS = 5.0
_MAX_LOCAL_SERVER_STOP_TIMEOUT_SECONDS = 15.0 * 60.0
_DEFAULT_SERVER_SHUTDOWN_GRACE_MS = 30_000
_DEFAULT_SERVER_SHUTDOWN_HARD_TIMEOUT_MS = 60_000
_DEFAULT_READ_LOG_MAX_EVENT_DROP_RATIO = 0.05
_DEFAULT_READ_LOG_MAX_ROW_DROP_RATIO = 0.05
_MAX_READ_LOG_ID_CHUNK_ROWS = 8_192
_MAX_READ_LOG_RETAINED_FILES = 65_533
_MAX_READ_LOG_ROTATE_SECONDS = (1 << 63) - 1
_MAX_UINT64_MIB = UINT64_MASK >> 20
_MAX_INSPECTED_READ_LOG_FILES = 65_536
_SERVER_READ_LOG_TUNING_ARGUMENTS = (
    "server_read_log_id_chunk_rows",
    "server_read_log_buffer_mib",
    "server_read_log_segment_mib",
    "server_read_log_rotate_seconds",
    "server_read_log_retention_mib",
    "server_read_log_retention_seconds",
    "server_read_log_min_free_mib",
    "server_read_log_max_write_mib_per_second",
    "server_read_log_max_retained_files",
    "server_read_log_max_event_drop_ratio",
    "server_read_log_max_row_drop_ratio",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class BenchmarkBudgetExceeded(RuntimeError):
    """Raised before work that would cross an explicit benchmark bound."""


class CrudBenchmarkError(RuntimeError):
    """Mutation failure carrying the logical-restoration outcome."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(message)


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


def _uint64_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > np.iinfo(np.uint64).max:
        raise argparse.ArgumentTypeError("value must fit in uint64")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def _read_log_id_chunk_rows(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > _MAX_READ_LOG_ID_CHUNK_ROWS:
        raise argparse.ArgumentTypeError(
            f"value must not exceed {_MAX_READ_LOG_ID_CHUNK_ROWS}"
        )
    return parsed


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated integer list"
        ) from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("every list value must be positive")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("list values must be unique")
    return result


def parse_prefixes(value: str, dimension: int) -> tuple[int, ...]:
    prefixes: list[int] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if token == "full":
            prefix = dimension
        else:
            try:
                prefix = int(token)
            except ValueError as error:
                raise ValueError(
                    "prefix dimensions must be comma-separated integers or 'full'"
                ) from error
        if not 1 <= prefix <= dimension:
            raise ValueError(
                f"prefix dimension {prefix} is outside 1..{dimension}"
            )
        if prefix not in prefixes:
            prefixes.append(prefix)
    if not prefixes:
        raise ValueError("at least one prefix dimension is required")
    return tuple(prefixes)


def resolve_cache_modes(
    requested: str,
    *,
    locally_launched_server: bool,
) -> tuple[str, ...]:
    if requested == "auto":
        return (
            ("cold", "warm")
            if locally_launched_server
            else ("warm",)
        )
    if requested == "both":
        if not locally_launched_server:
            raise ValueError(
                "--cache-mode both requires a server launched by this harness; "
                "an external client cannot safely control the server page cache"
            )
        return ("cold", "warm")
    if requested == "cold" and not locally_launched_server:
        raise ValueError(
            "--cache-mode cold requires a server launched by this harness; "
            "an external client cannot safely control the server page cache"
        )
    if requested not in {"ambient", "cold", "warm"}:
        raise ValueError(f"unsupported cache mode: {requested}")
    return (requested,)


def derived_seed(seed: int, offset: int) -> int:
    return (seed + offset) & UINT64_MASK


def current_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    try:
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    # Linux reports KiB, macOS reports bytes.
    return maximum if platform.system() == "Darwin" else maximum * 1024


@dataclass(slots=True)
class ResourceBudget:
    max_seconds: float
    max_bytes: int
    max_client_memory_bytes: int
    started_monotonic: float = field(default_factory=time.monotonic)
    accounted_bytes: int = 0
    memory_baseline_rss_bytes: int | None = field(default_factory=current_rss_bytes)
    peak_observed_client_rss_bytes: int | None = None
    peak_benchmark_working_bytes: int | None = None
    memory_baselines: list[dict[str, int | str | None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def deadline(self) -> float:
        return self.started_monotonic + self.max_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self, *, extra_memory_bytes: int = 0) -> None:
        if self.remaining_seconds() <= 0:
            raise BenchmarkBudgetExceeded(
                f"maximum wall time ({self.max_seconds:.3f}s) reached"
            )
        rss = current_rss_bytes()
        if rss is not None:
            with self._lock:
                previous = self.peak_observed_client_rss_bytes or 0
                self.peak_observed_client_rss_bytes = max(previous, rss)
                baseline = self.memory_baseline_rss_bytes or rss
                working = max(0, rss - baseline)
                previous_working = self.peak_benchmark_working_bytes or 0
                self.peak_benchmark_working_bytes = max(
                    previous_working, working
                )
            if working + extra_memory_bytes > self.max_client_memory_bytes:
                raise BenchmarkBudgetExceeded(
                    "client memory bound would be exceeded: "
                    f"working={working}, requested={extra_memory_bytes}, "
                    f"limit={self.max_client_memory_bytes}"
                )

    def rebase_memory(self, reason: str) -> None:
        """Exclude an engine/index baseline while still bounding work buffers."""

        rss = current_rss_bytes()
        with self._lock:
            self.memory_baselines.append(
                {
                    "reason": reason,
                    "previous_rss_bytes": self.memory_baseline_rss_bytes,
                    "new_rss_bytes": rss,
                }
            )
            self.memory_baseline_rss_bytes = rss

    def reserve(self, byte_count: int, *, extra_memory_bytes: int = 0) -> None:
        if byte_count < 0 or extra_memory_bytes < 0:
            raise ValueError("budget charges cannot be negative")
        self.check(extra_memory_bytes=extra_memory_bytes)
        with self._lock:
            if self.accounted_bytes + byte_count > self.max_bytes:
                raise BenchmarkBudgetExceeded(
                    "maximum accounted I/O bound would be exceeded: "
                    f"used={self.accounted_bytes}, requested={byte_count}, "
                    f"limit={self.max_bytes}"
                )
            self.accounted_bytes += byte_count

    def rpc_timeout(self, configured_timeout: float) -> float:
        self.check()
        return max(0.001, min(configured_timeout, self.remaining_seconds()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_seconds": self.max_seconds,
            "elapsed_seconds": time.monotonic() - self.started_monotonic,
            "remaining_seconds": self.remaining_seconds(),
            "max_accounted_bytes": self.max_bytes,
            "accounted_bytes": self.accounted_bytes,
            "max_client_memory_bytes": self.max_client_memory_bytes,
            "memory_baseline_rss_bytes": self.memory_baseline_rss_bytes,
            "peak_observed_client_rss_bytes": (
                self.peak_observed_client_rss_bytes
            ),
            "peak_benchmark_working_bytes": self.peak_benchmark_working_bytes,
            "memory_baselines": list(self.memory_baselines),
        }


def cold_cache_preparer(
    store: Path,
    budget: ResourceBudget,
) -> Callable[[], dict[str, object]]:
    """Build a fail-closed, file-scoped cold-cache preparation callback."""

    def prepare() -> dict[str, object]:
        budget.check()
        result = prepare_cold_payload_cache(
            store,
            deadline=budget.deadline,
            clock=time.monotonic,
        )
        if result.get("status") != "performed":
            raise RuntimeError(
                "cold payload cache preparation did not complete: "
                + json.dumps(result, sort_keys=True)
            )
        budget.check()
        return result

    return prepare


def add_cache_observation(
    result: dict[str, Any],
    *,
    record_size: int,
    remote: bool,
) -> None:
    """Classify observed backing reads without claiming guaranteed eviction."""

    planned = int(result.get("rows", 0)) * record_size
    telemetry = result.get("telemetry", {})
    process_key = "server_process" if remote else "client_process"
    process = telemetry.get(process_key, {}) if isinstance(telemetry, dict) else {}
    io_delta = process.get("io_delta", {}) if isinstance(process, dict) else {}
    backing = io_delta.get("read_bytes") if isinstance(io_delta, dict) else None
    if not isinstance(backing, int) or backing < 0:
        result["cache_observation"] = {
            "planned_payload_read_bytes": planned,
            "backing_read_bytes": None,
            "backing_read_ratio": None,
            "classification": "unknown",
            "process_scope": process_key,
        }
        return
    ratio = backing / planned if planned else 0.0
    if ratio <= CACHE_BACKING_RATIO_MAX:
        classification = "page_cache"
    elif ratio >= CACHE_STORAGE_RATIO_MIN:
        classification = "backing_storage"
    else:
        classification = "mixed"
    result["cache_observation"] = {
        "planned_payload_read_bytes": planned,
        "backing_read_bytes": backing,
        "backing_read_ratio": ratio,
        "classification": classification,
        "process_scope": process_key,
        "interpretation": (
            "observed process backing reads; POSIX_FADV_DONTNEED remains advisory"
        ),
    }


def read_process_snapshot(pid: int) -> dict[str, Any]:
    """Read Linux per-process counters without requiring extra tools/root."""

    root = Path("/proc") / str(pid)
    if not root.is_dir():
        return {"available": False, "pid": pid}
    result: dict[str, Any] = {"available": True, "pid": pid}
    try:
        fields = (root / "stat").read_text(encoding="ascii").rstrip()
        end = fields.rfind(")")
        tail = fields[end + 2 :].split()
        ticks = os.sysconf("SC_CLK_TCK")
        pages = os.sysconf("SC_PAGE_SIZE")
        result.update(
            {
                "user_cpu_seconds": int(tail[11]) / ticks,
                "system_cpu_seconds": int(tail[12]) / ticks,
                "minor_page_faults": int(tail[7]),
                "major_page_faults": int(tail[9]),
                "threads": int(tail[17]),
                "rss_bytes": int(tail[21]) * pages,
            }
        )
    except (OSError, ValueError, IndexError):
        result["stat_available"] = False
    io: dict[str, int] = {}
    try:
        for line in (root / "io").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            io[key] = int(value.strip())
    except (OSError, ValueError):
        pass
    result["io"] = io
    return result


def process_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return {"available": False, "pid": after.get("pid", before.get("pid"))}
    result: dict[str, Any] = {
        "available": True,
        "pid": after["pid"],
        "rss_bytes_initial": before.get("rss_bytes"),
        "rss_bytes_final": after.get("rss_bytes"),
        "threads_initial": before.get("threads"),
        "threads_final": after.get("threads"),
    }
    for key in (
        "rss_bytes",
        "threads",
        "minor_page_faults",
        "major_page_faults",
    ):
        if key in before and key in after:
            result[f"{key}_delta"] = after[key] - before[key]
    for key in ("user_cpu_seconds", "system_cpu_seconds"):
        if key in before and key in after:
            result[f"{key}_delta"] = after[key] - before[key]
    before_io = before.get("io", {})
    after_io = after.get("io", {})
    if before_io and after_io:
        result["io_delta"] = load_test.counter_delta(before_io, after_io)
    return result


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    client: dict[str, Any]
    server: dict[str, Any]
    cgroup: dict[str, int]


def telemetry_snapshot(server_pid: int | None) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        client=read_process_snapshot(os.getpid()),
        server=(
            read_process_snapshot(server_pid)
            if server_pid is not None
            else {"available": False, "reason": "server PID not supplied"}
        ),
        cgroup=load_test.cgroup_cpu_snapshot(),
    )


def telemetry_delta(
    before: TelemetrySnapshot, after: TelemetrySnapshot
) -> dict[str, Any]:
    cgroup: dict[str, int | float] | dict[str, Any]
    if before.cgroup and after.cgroup:
        cgroup = load_test.counter_delta(before.cgroup, after.cgroup)
        if "throttled_usec" in cgroup:
            cgroup["throttled_seconds"] = (
                float(cgroup["throttled_usec"]) / 1_000_000
            )
    else:
        cgroup = {"available": False}
    return {
        "client_process": process_delta(before.client, after.client),
        "server_process": process_delta(before.server, after.server),
        "cgroup_cpu_delta": cgroup,
        "cgroup_scope": (
            "shared cgroup counters; not attributable to one process"
        ),
    }


def collect_external_client_information() -> dict[str, Any]:
    """Collect client-side facts when the server storage is not mounted."""

    return {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "throwdb": throwdb.__version__,
        "cpu_count": os.cpu_count(),
        "cgroup_cpu": load_test.collect_cgroup_cpu_information(),
        "storage_probe": {
            "available": False,
            "reason": "external endpoint has no local --store path",
        },
    }


def _local_server_stop_timeout_seconds(
    *,
    shutdown_grace_ms: int | None,
    shutdown_hard_timeout_ms: int | None,
) -> float:
    """Return the bounded parent-side wait for a configured server shutdown.

    Wait through the effective native hard deadline plus a small scheduling
    margin so the harness does not preempt the daemon's own drain/watchdog
    policy. This includes the native defaults when no override is supplied.
    """

    effective_grace_ms = (
        _DEFAULT_SERVER_SHUTDOWN_GRACE_MS
        if shutdown_grace_ms is None
        else shutdown_grace_ms
    )
    effective_hard_timeout_ms = (
        _DEFAULT_SERVER_SHUTDOWN_HARD_TIMEOUT_MS
        if shutdown_hard_timeout_ms is None
        else shutdown_hard_timeout_ms
    )
    configured_seconds = (
        max(effective_grace_ms, effective_hard_timeout_ms) / 1_000.0
        + _LOCAL_SERVER_SHUTDOWN_TIMEOUT_MARGIN_SECONDS
    )
    return min(
        _MAX_LOCAL_SERVER_STOP_TIMEOUT_SECONDS,
        max(_LEGACY_LOCAL_SERVER_STOP_TIMEOUT_SECONDS, configured_seconds),
    )


def _read_log_server_arguments(
    *,
    directory: Path | None,
    id_chunk_rows: int | None,
    buffer_mib: int | None,
    segment_mib: int | None,
    rotate_seconds: int | None,
    retention_mib: int | None,
    retention_seconds: int | None,
    min_free_mib: int | None,
    max_write_mib_per_second: int | None,
    max_retained_files: int | None,
) -> list[str]:
    """Build the explicit public-launcher arguments for read logging."""

    configured = (
        id_chunk_rows,
        buffer_mib,
        segment_mib,
        rotate_seconds,
        retention_mib,
        retention_seconds,
        min_free_mib,
        max_write_mib_per_second,
        max_retained_files,
    )
    if directory is None:
        if any(value is not None for value in configured):
            raise ValueError(
                "server read-log tuning requires a dedicated log directory"
            )
        return []
    result = ["--read-log-dir", str(directory)]
    for option, value in (
        ("--read-log-id-chunk-rows", id_chunk_rows),
        ("--read-log-buffer-mib", buffer_mib),
        ("--read-log-segment-mib", segment_mib),
        ("--read-log-rotate-seconds", rotate_seconds),
        ("--read-log-retention-mib", retention_mib),
        ("--read-log-retention-seconds", retention_seconds),
        ("--read-log-min-free-mib", min_free_mib),
        ("--read-log-max-write-mib-per-second", max_write_mib_per_second),
        ("--read-log-max-retained-files", max_retained_files),
    ):
        if value is not None:
            result.extend((option, str(value)))
    return result


@dataclass(slots=True)
class LocalServer:
    process: subprocess.Popen[str]
    command: list[str]
    ready: dict[str, Any]
    stdout_path: Path
    stderr_path: Path
    _stdout_thread: threading.Thread
    _stderr_thread: threading.Thread
    _stop_timeout_seconds: float = (
        _DEFAULT_SERVER_SHUTDOWN_HARD_TIMEOUT_MS / 1_000.0
        + _LOCAL_SERVER_SHUTDOWN_TIMEOUT_MARGIN_SECONDS
    )

    @property
    def endpoint(self) -> str:
        return str(self.ready["endpoint"])

    @property
    def pid(self) -> int:
        return self.process.pid

    @classmethod
    def start(
        cls,
        *,
        store: Path,
        results: Path,
        tuning: str,
        timeout: float,
        writable: bool,
        native_binary: Path | None,
        max_worker_threads: int | None,
        max_ids_per_request: int | None,
        max_scan_batch_rows: int | None,
        maintenance_interval_seconds: int | None,
        disable_automatic_maintenance: bool,
        max_concurrent_calls: int | None = None,
        max_concurrent_scans: int | None = None,
        scan_max_duration_ms: int | None = None,
        scan_idle_timeout_ms: int | None = None,
        shutdown_grace_ms: int | None = None,
        shutdown_hard_timeout_ms: int | None = None,
        read_log_directory: Path | None = None,
        read_log_id_chunk_rows: int | None = None,
        read_log_buffer_mib: int | None = None,
        read_log_segment_mib: int | None = None,
        read_log_rotate_seconds: int | None = None,
        read_log_retention_mib: int | None = None,
        read_log_retention_seconds: int | None = None,
        read_log_min_free_mib: int | None = None,
        read_log_max_write_mib_per_second: int | None = None,
        read_log_max_retained_files: int | None = None,
    ) -> "LocalServer":
        command = [
            sys.executable,
            "-m",
            "throwdb.server",
            "--storage",
            str(store),
            "--listen",
            "127.0.0.1:0",
            "--tuning",
            tuning,
        ]
        if writable:
            command.append("--writable")
        if disable_automatic_maintenance:
            command.append("--no-auto-maintenance")
        if native_binary is not None:
            command.extend(("--native-binary", str(native_binary)))
        for option, value in (
            ("--max-worker-threads", max_worker_threads),
            ("--max-ids-per-request", max_ids_per_request),
            ("--max-scan-batch-rows", max_scan_batch_rows),
            ("--maintenance-interval-seconds", maintenance_interval_seconds),
            ("--max-concurrent-calls", max_concurrent_calls),
            ("--max-concurrent-scans", max_concurrent_scans),
            ("--scan-max-duration-ms", scan_max_duration_ms),
            ("--scan-idle-timeout-ms", scan_idle_timeout_ms),
            ("--shutdown-grace-ms", shutdown_grace_ms),
            ("--shutdown-hard-timeout-ms", shutdown_hard_timeout_ms),
        ):
            if value is not None:
                command.extend((option, str(value)))
        command.extend(
            _read_log_server_arguments(
                directory=read_log_directory,
                id_chunk_rows=read_log_id_chunk_rows,
                buffer_mib=read_log_buffer_mib,
                segment_mib=read_log_segment_mib,
                rotate_seconds=read_log_rotate_seconds,
                retention_mib=read_log_retention_mib,
                retention_seconds=read_log_retention_seconds,
                min_free_mib=read_log_min_free_mib,
                max_write_mib_per_second=(
                    read_log_max_write_mib_per_second
                ),
                max_retained_files=read_log_max_retained_files,
            )
        )

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        ready_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        stdout_path = results / "server.stdout.log"
        stderr_path = results / "server.stderr.log"

        def read_stdout() -> None:
            with stdout_path.open("w", encoding="utf-8") as output:
                try:
                    for line in process.stdout:
                        output.write(line)
                        output.flush()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and event.get("event") == "ready":
                            ready_queue.put(event)
                finally:
                    ready_queue.put(None)

        def read_stderr() -> None:
            with stderr_path.open("w", encoding="utf-8") as output:
                for line in process.stderr:
                    output.write(line)
                    output.flush()

        stdout_thread = threading.Thread(
            target=read_stdout, name="throwdb-benchmark-server-stdout", daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stderr, name="throwdb-benchmark-server-stderr", daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + timeout
        ready: dict[str, Any] | None = None
        while ready is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                event = ready_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if event is None:
                break
            ready = event
        if ready is None:
            cls._stop_process(process)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                "ThrowDB server did not become ready; "
                f"returncode={process.poll()}; stderr={stderr[-4000:]}"
            )
        if ready.get("protocol") != "grpc/throwdb.v1":
            cls._stop_process(process)
            raise RuntimeError(
                f"server announced unsupported protocol {ready.get('protocol')!r}"
            )
        if not isinstance(ready.get("endpoint"), str):
            cls._stop_process(process)
            raise RuntimeError("server ready event has no endpoint")
        if not isinstance(ready.get("reader"), dict) or not isinstance(
            ready.get("tuning"), dict
        ):
            cls._stop_process(process)
            raise RuntimeError("server ready event has no reader/tuning diagnostics")
        return cls(
            process=process,
            command=command,
            ready=ready,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            _stdout_thread=stdout_thread,
            _stderr_thread=stderr_thread,
            _stop_timeout_seconds=_local_server_stop_timeout_seconds(
                shutdown_grace_ms=shutdown_grace_ms,
                shutdown_hard_timeout_ms=shutdown_hard_timeout_ms,
            ),
        )

    @staticmethod
    def _stop_process(
        process: subprocess.Popen[str],
        *,
        timeout_seconds: float = _LEGACY_LOCAL_SERVER_STOP_TIMEOUT_SECONDS,
    ) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def close(self) -> None:
        self._stop_process(
            self.process,
            timeout_seconds=self._stop_timeout_seconds,
        )
        self._stdout_thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)

    def __enter__(self) -> "LocalServer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def description_dict(description: throwdb.RemoteDescription) -> dict[str, Any]:
    return {
        "format_version": description.format_version,
        "schema": {
            "id_type": description.schema.id_type,
            "dimension": description.schema.dimension,
            "dtype": description.schema.dtype,
        },
        "capabilities": description.capabilities,
        "record_size": description.record_size,
        "count": description.count,
        "writable": description.writable,
        "prefix_get": description.prefix_get,
        "scan_locations": description.scan_locations,
        "durable_idempotency": description.durable_idempotency,
        "reader": (
            description.reader.as_dict() if description.reader is not None else None
        ),
        "tuning": (
            asdict(description.tuning) if description.tuning is not None else None
        ),
        "read_logging": (
            description.read_logging.as_dict()
            if description.read_logging is not None
            else None
        ),
        "read_logging_stats": (
            description.read_logging_stats.as_dict()
            if description.read_logging_stats is not None
            else None
        ),
    }


def validate_ready_description(
    ready: dict[str, Any], description: throwdb.RemoteDescription
) -> None:
    expected_access_mode = "writable" if description.writable else "read_only"
    if ready.get("access_mode") != expected_access_mode:
        raise RuntimeError("ready event has an inconsistent access mode")
    if ready.get("readiness") != {"status": "SERVING"}:
        raise RuntimeError("ready event did not report SERVING readiness")
    if ready.get("security") != {
        "transport": "insecure",
        "client_certificate_required": False,
        "insecure_non_loopback_override": False,
    }:
        raise RuntimeError("local loopback server has unexpected transport security")
    expected_reader = (
        description.reader.as_dict() if description.reader is not None else None
    )
    expected_tuning = (
        asdict(description.tuning) if description.tuning is not None else None
    )
    if ready.get("reader") != expected_reader:
        raise RuntimeError("ready event and Describe disagree about reader settings")
    if ready.get("tuning") != expected_tuning:
        raise RuntimeError("ready event and Describe disagree about tuning selection")
    if ready.get("writable") != description.writable:
        raise RuntimeError("ready event and Describe disagree about writable mode")
    if ready.get("durable_idempotency") != description.durable_idempotency:
        raise RuntimeError(
            "ready event and Describe disagree about durable idempotency"
        )
    ready_read_logging = ready.get("read_logging")
    described_read_logging = description.read_logging
    if described_read_logging is None:
        if ready_read_logging is not None:
            raise RuntimeError(
                "ready event advertised read logging that Describe omitted"
            )
        return
    if not isinstance(ready_read_logging, dict):
        raise RuntimeError(
            "ready event or Describe omitted read-logging configuration"
        )
    if ready_read_logging.get("enabled") != described_read_logging.enabled:
        raise RuntimeError(
            "ready event and Describe disagree about read-logging state"
        )
    if described_read_logging.enabled:
        if not isinstance(ready_read_logging.get("directory"), str):
            raise RuntimeError(
                "ready event omitted the local read-log directory"
            )
        for field_name, expected_value in described_read_logging.as_dict().items():
            if (
                field_name != "enabled"
                and ready_read_logging.get(field_name) != expected_value
            ):
                raise RuntimeError(
                    "ready event and Describe disagree about read-logging "
                    f"field {field_name}"
                )


def _capture_final_describe(
    remote: throwdb.RemoteCollection,
    *,
    budget: ResourceBudget,
    rpc_timeout: float,
) -> dict[str, Any]:
    if budget.remaining_seconds() <= 0:
        return {
            "status": "skipped_budget_exhausted",
            "description": None,
        }
    started = time.perf_counter()
    try:
        _set_remote_timeout(remote, budget, rpc_timeout)
        description = remote.describe()
    except Exception as error:
        return {
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "description": None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    return {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "captured_unix_ns": time.time_ns(),
        "description": description_dict(description),
    }


def _inspect_read_log_after_shutdown(
    *,
    directory: Path,
    storage: Path,
    clean_shutdown: bool,
    budget: ResourceBudget,
    max_event_drop_ratio: float,
    max_row_drop_ratio: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "pending",
        "raw_ids_included": False,
        "max_event_drop_ratio": max_event_drop_ratio,
        "max_row_drop_ratio": max_row_drop_ratio,
    }
    if not clean_shutdown:
        result["status"] = "skipped_unclean_server_shutdown"
        return result
    try:
        budget.check()
    except BenchmarkBudgetExceeded as error:
        result["status"] = "skipped_budget_exhausted"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        return result

    started = time.perf_counter()
    try:
        file_count = 0
        total_bytes = 0
        maximum_file_bytes = 0
        scanned_entries = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                budget.check()
                scanned_entries += 1
                if scanned_entries > _MAX_INSPECTED_READ_LOG_FILES:
                    raise BenchmarkBudgetExceeded(
                        "read-log directory exceeds the bounded post-shutdown "
                        "inventory limit"
                    )
                path = Path(entry.path)
                if path.suffix not in {".tdbrl", ".open"}:
                    continue
                status = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(status.st_mode):
                    continue
                file_count += 1
                if file_count > _MAX_INSPECTED_READ_LOG_FILES:
                    raise BenchmarkBudgetExceeded(
                        "read-log file count exceeds the bounded "
                        "post-shutdown inspection limit"
                    )
                file_bytes = int(status.st_size)
                total_bytes += file_bytes
                maximum_file_bytes = max(maximum_file_bytes, file_bytes)
        # Charge the complete logical input before mmap/CRC traversal. This
        # remains conservative if files shrink and fails before reading if
        # either transfer or worst-case per-file RSS would exceed the harness.
        budget.reserve(
            total_bytes,
            extra_memory_bytes=maximum_file_bytes,
        )
        inspection = throwdb.inspect_workload_logs(
            directory,
            storage=storage,
            max_drop_ratio=max_event_drop_ratio,
            max_row_drop_ratio=max_row_drop_ratio,
            max_files=max(1, file_count),
            max_total_bytes=max(1, total_bytes),
            max_file_bytes=max(1, maximum_file_bytes),
            _deadline_monotonic=budget.deadline,
        )
        budget.check()
    except BenchmarkBudgetExceeded as error:
        result["status"] = "skipped_budget_exhausted"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    except Exception as error:
        # A deadline enforced inside the inspector is still a benchmark-budget
        # outcome, not corrupt telemetry. Re-check the authoritative budget
        # before classifying any other optional-inspection error.
        try:
            budget.check()
        except BenchmarkBudgetExceeded as budget_error:
            result["status"] = "skipped_budget_exhausted"
            result["error"] = {
                "type": type(budget_error).__name__,
                "message": str(budget_error),
            }
        else:
            # Workload logging and its offline qualification are optional.
            # Preserve the benchmark result while making failure explicit.
            result["status"] = "error"
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
    else:
        result["status"] = "complete"
        result["inspection"] = inspection
        result["accounted_input_bytes"] = total_bytes
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def canonical_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.dtype("<u8"), order="C")


def sampled_ids(
    population: np.ndarray,
    rows: int,
    seed: int,
    *,
    budget: ResourceBudget | None = None,
) -> np.ndarray:
    if budget is not None:
        # One int64 position vector, the advanced-index result in the source
        # dtype, and a canonical uint64 result when conversion is required.
        budget.check(
            extra_memory_bytes=rows * (16 + population.dtype.itemsize)
        )
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(population), size=rows, dtype=np.int64)
    return canonical_ids(population[positions])


def update_query_digest(
    digest: Any,
    *,
    label: str,
    rows: int,
    prefix: int,
    seed: int,
    ids: np.ndarray,
) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "little"))
    digest.update(encoded_label)
    digest.update(rows.to_bytes(8, "little"))
    digest.update(prefix.to_bytes(8, "little"))
    digest.update(seed.to_bytes(8, "little", signed=False))
    digest.update(memoryview(ids).cast("B"))


def query_digest(
    population: np.ndarray,
    tasks: Iterable[tuple[str, int, int, int]],
    *,
    budget: ResourceBudget | None = None,
) -> str:
    digest = hashlib.sha256()
    for label, rows, prefix, seed in tasks:
        seed = derived_seed(seed, 0)
        ids = sampled_ids(population, rows, seed, budget=budget)
        update_query_digest(
            digest,
            label=label,
            rows=rows,
            prefix=prefix,
            seed=seed,
            ids=ids,
        )
    return digest.hexdigest()


def _wire_bytes(rows: int, prefix: int, itemsize: int) -> int:
    # Counts the dominant payloads exactly. Protobuf field tags and gRPC frame
    # headers are deliberately reported as unaccounted protocol overhead.
    return rows * (8 + prefix * itemsize)


def _set_remote_timeout(
    remote: throwdb.RemoteCollection,
    budget: ResourceBudget,
    configured_timeout: float,
) -> None:
    # RemoteCollection intentionally has a fixed public timeout. The harness
    # tightens its internal deadline before each call so no new RPC can extend
    # beyond the run-wide wall-clock bound.
    remote._timeout = budget.rpc_timeout(  # type: ignore[attr-defined]
        configured_timeout
    )


def _get_once(
    reader: Any,
    ids: np.ndarray,
    *,
    prefix: int,
    dimension: int,
    budget: ResourceBudget,
    rpc_timeout: float,
    remote: bool,
    itemsize: int,
) -> tuple[int, int]:
    charge = _wire_bytes(ids.size, prefix, itemsize)
    budget.reserve(
        charge,
        extra_memory_bytes=3 * ids.size * prefix * itemsize,
    )
    if remote:
        _set_remote_timeout(reader, budget, rpc_timeout)
    started = time.perf_counter_ns()
    result = reader.get(
        ids,
        prefix_dimensions=None if prefix == dimension else prefix,
    )
    elapsed = time.perf_counter_ns() - started
    if result.shape != (ids.size, prefix):
        raise AssertionError(
            f"Get returned shape {result.shape}, expected {(ids.size, prefix)}"
        )
    return elapsed, charge


def metric_summary(
    latencies_ns: Sequence[int],
    *,
    seconds: float,
    rows: int,
    useful_bytes: int,
) -> dict[str, Any]:
    if not latencies_ns:
        raise RuntimeError("scenario completed no calls")
    values = np.asarray(latencies_ns, dtype=np.int64)
    return {
        "seconds": seconds,
        "calls": len(latencies_ns),
        "rows": rows,
        "useful_embedding_bytes": useful_bytes,
        "rows_per_second": rows / seconds if seconds else 0.0,
        "useful_gib_per_second": useful_bytes / seconds / GIB if seconds else 0.0,
        **load_test.latency_metrics(values),
    }


def run_sequential_get(
    reader: Any,
    population: np.ndarray,
    *,
    label: str,
    batch_size: int,
    prefix: int,
    dimension: int,
    itemsize: int,
    calls: int,
    warmup_calls: int,
    seed: int,
    budget: ResourceBudget,
    rpc_timeout: float,
    remote: bool,
    server_pid: int | None,
    cache_mode: str = "ambient",
    prepare_cold: Callable[[], dict[str, object]] | None = None,
) -> dict[str, Any]:
    if cache_mode not in {"ambient", "cold", "warm"}:
        raise ValueError("sequential cache mode must be ambient, cold, or warm")
    if cache_mode == "cold" and prepare_cold is None:
        raise ValueError("cold sequential measurement requires cache preparation")
    budget.check(
        extra_memory_bytes=(
            (calls + warmup_calls) * batch_size * 8
            + calls * 16
            + batch_size * (16 + 3 * prefix * itemsize)
        )
    )
    warmup_ids = [
        sampled_ids(
            population,
            batch_size,
            derived_seed(seed, call),
            budget=budget,
        )
        for call in range(warmup_calls)
    ]
    measured: list[tuple[int, np.ndarray]] = []
    digest = hashlib.sha256()
    for call in range(calls):
        task_seed = derived_seed(seed, warmup_calls + call)
        ids = sampled_ids(population, batch_size, task_seed, budget=budget)
        update_query_digest(
            digest,
            label=label,
            rows=batch_size,
            prefix=prefix,
            seed=task_seed,
            ids=ids,
        )
        measured.append((task_seed, ids))

    warmup_charge = 0
    for ids in warmup_ids:
        budget.check(
            extra_memory_bytes=batch_size * (16 + 3 * prefix * itemsize)
        )
        _, charge = _get_once(
            reader,
            ids,
            prefix=prefix,
            dimension=dimension,
            budget=budget,
            rpc_timeout=rpc_timeout,
            remote=remote,
            itemsize=itemsize,
        )
        warmup_charge += charge

    cache_preparation: dict[str, object] | None = None
    prewarm_charge = 0
    warm_preparation: dict[str, object] = {
        "kind": "none",
        "calls": 0,
        "accounted_wire_bytes": 0,
    }
    if cache_mode == "cold":
        assert prepare_cold is not None
        cache_preparation = prepare_cold()
    elif cache_mode == "warm":
        for _, ids in measured:
            _, charge = _get_once(
                reader,
                ids,
                prefix=prefix,
                dimension=dimension,
                budget=budget,
                rpc_timeout=rpc_timeout,
                remote=remote,
                itemsize=itemsize,
            )
            prewarm_charge += charge
        warm_preparation = {
            "kind": "exact_query_replay",
            "calls": len(measured),
            "accounted_wire_bytes": prewarm_charge,
            "query_digest_sha256": digest.hexdigest(),
        }
    before = telemetry_snapshot(server_pid)
    started = time.perf_counter()
    latencies: list[int] = []
    charged = 0
    for task_seed, ids in measured:
        budget.check(
            extra_memory_bytes=batch_size * (16 + 3 * prefix * itemsize)
        )
        elapsed, wire_bytes = _get_once(
            reader,
            ids,
            prefix=prefix,
            dimension=dimension,
            budget=budget,
            rpc_timeout=rpc_timeout,
            remote=remote,
            itemsize=itemsize,
        )
        latencies.append(elapsed)
        charged += wire_bytes
    seconds = time.perf_counter() - started
    after = telemetry_snapshot(server_pid)
    useful_bytes = calls * batch_size * prefix * itemsize
    return {
        "label": label,
        "batch_size": batch_size,
        "prefix_dimensions": prefix,
        "warmup_calls": warmup_calls,
        "query_digest_sha256": digest.hexdigest(),
        "accounted_wire_bytes": charged,
        "measured_accounted_wire_bytes": charged,
        "warmup_accounted_wire_bytes": warmup_charge,
        "prewarm_accounted_wire_bytes": prewarm_charge,
        "total_accounted_wire_bytes": (
            charged + warmup_charge + prewarm_charge
        ),
        "cache_mode": cache_mode,
        "cache_preparation": cache_preparation,
        "warm_preparation": warm_preparation,
        "timing_semantics": {
            "wall": "timed Get calls only; query generation is excluded",
            "latency": (
                "client_observed_rpc_round_trip"
                if remote
                else "embedded Get call"
            ),
        },
        **metric_summary(
            latencies,
            seconds=seconds,
            rows=calls * batch_size,
            useful_bytes=useful_bytes,
        ),
        "telemetry": telemetry_delta(before, after),
    }


@dataclass(frozen=True, slots=True)
class GetTask:
    label: str
    rows: int
    prefix: int
    seed: int


@dataclass(frozen=True, slots=True)
class PreparedGetTask:
    ordinal: int
    task: GetTask
    ids: np.ndarray


@dataclass(frozen=True, slots=True)
class ConcurrentSample:
    task_ordinal: int
    label: str
    rows: int
    ready_ns: int
    dequeued_ns: int
    rpc_started_ns: int
    finished_ns: int

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("concurrent sample rows must be positive")
        if not (
            0 <= self.ready_ns <= self.dequeued_ns
            <= self.rpc_started_ns <= self.finished_ns
        ):
            raise ValueError(
                "concurrent sample timestamps must be nonnegative and monotonic"
            )

    @property
    def client_queue_wait_ns(self) -> int:
        return self.dequeued_ns - self.ready_ns

    @property
    def client_prepare_ns(self) -> int:
        return self.rpc_started_ns - self.dequeued_ns

    @property
    def rpc_round_trip_ns(self) -> int:
        return self.finished_ns - self.rpc_started_ns

    @property
    def client_sojourn_ns(self) -> int:
        return self.finished_ns - self.ready_ns


def _duration_distribution(values_ns: Sequence[int]) -> dict[str, float]:
    metrics = load_test.latency_metrics(
        np.asarray(values_ns, dtype=np.int64)
    )
    return {
        "p50_ms": float(metrics["latency_p50_ms"]),
        "p95_ms": float(metrics["latency_p95_ms"]),
        "p99_ms": float(metrics["latency_p99_ms"]),
        "max_ms": float(metrics["latency_max_ms"]),
    }


def summarize_concurrent_samples(
    samples: Sequence[ConcurrentSample],
    *,
    seconds: float,
    itemsize: int,
    prefix: int,
) -> dict[str, Any]:
    """Summarize client admission and RPC timing without overclaiming it.

    ``rpc_round_trip`` is client-observed synchronous RPC time.  It includes
    serialization, transport, server queueing/service, and response copies;
    it is deliberately not named server service time.
    """

    if not samples:
        raise RuntimeError("concurrent scenario completed no calls")
    classes: dict[str, dict[str, Any]] = {}
    for label in sorted({sample.label for sample in samples}):
        selected = [sample for sample in samples if sample.label == label]
        rows = sum(sample.rows for sample in selected)
        useful = rows * prefix * itemsize
        rpc = [sample.rpc_round_trip_ns for sample in selected]
        result = metric_summary(
            rpc,
            seconds=seconds,
            rows=rows,
            useful_bytes=useful,
        )
        result["latency_semantics"] = "client_observed_rpc_round_trip"
        result["throughput_semantics"] = "class_contribution_over_scenario_wall"
        result["timings"] = {
            "client_queue_wait": _duration_distribution(
                [sample.client_queue_wait_ns for sample in selected]
            ),
            "client_prepare": _duration_distribution(
                [sample.client_prepare_ns for sample in selected]
            ),
            "rpc_round_trip": _duration_distribution(rpc),
            "client_sojourn": _duration_distribution(
                [sample.client_sojourn_ns for sample in selected]
            ),
        }
        classes[label] = result
    return {"classes": classes}


def run_concurrent_get(
    endpoint: str,
    population: np.ndarray,
    tasks: Sequence[GetTask],
    *,
    dimension: int,
    itemsize: int,
    workers: int,
    budget: ResourceBudget,
    rpc_timeout: float,
    server_pid: int | None,
    cache_mode: str = "ambient",
    prepare_cold: Callable[[], dict[str, object]] | None = None,
) -> dict[str, Any]:
    if cache_mode not in {"ambient", "cold", "warm"}:
        raise ValueError("concurrent cache mode must be ambient, cold, or warm")
    if cache_mode == "cold" and prepare_cold is None:
        raise ValueError("cold concurrent measurement requires cache preparation")
    if not tasks:
        raise ValueError("concurrent measurement requires at least one task")
    if workers <= 0:
        raise ValueError("concurrent worker count must be positive")
    if any(task.prefix != tasks[0].prefix for task in tasks):
        raise ValueError(
            "one concurrent scenario must use a single prefix dimension"
        )

    # Materialize the complete query corpus before telemetry and the start
    # barrier. NumPy RNG/indexing is benchmark setup, not server throughput.
    planned_id_bytes = sum(task.rows * 8 for task in tasks)
    largest_sampling_memory = max(
        task.rows * (16 + population.dtype.itemsize) for task in tasks
    )
    budget.check(
        extra_memory_bytes=planned_id_bytes + largest_sampling_memory
    )
    prepared: list[PreparedGetTask] = []
    total_id_bytes = 0
    digest = hashlib.sha256()
    for ordinal, task in enumerate(tasks):
        ids = sampled_ids(population, task.rows, task.seed, budget=budget)
        total_id_bytes += int(ids.nbytes)
        update_query_digest(
            digest,
            label=task.label,
            rows=task.rows,
            prefix=task.prefix,
            seed=task.seed,
            ids=ids,
        )
        prepared.append(PreparedGetTask(ordinal, task, ids))
    class_digests: dict[str, str] = {}
    for label in sorted({item.task.label for item in prepared}):
        class_digest = hashlib.sha256()
        for item in prepared:
            if item.task.label != label:
                continue
            update_query_digest(
                class_digest,
                label=item.task.label,
                rows=item.task.rows,
                prefix=item.task.prefix,
                seed=item.task.seed,
                ids=item.ids,
            )
        class_digests[label] = class_digest.hexdigest()

    # Reserve every planned wave atomically before submitting any RPC. This
    # keeps the transfer and peak response-memory bounds deterministic even
    # when workers race, and avoids charging a replay that never ran when the
    # subsequent measured wave would exceed the run-wide budget.
    total_charge = sum(
        _wire_bytes(task.rows, task.prefix, itemsize) for task in tasks
    )
    # Assign exactly one request to every connected client. A shared queue
    # cannot guarantee this: a fast worker may dequeue a second request before
    # another worker has entered its first Get.
    stabilization_prepared = [
        prepared[index % len(prepared)] for index in range(workers)
    ]
    stabilization_charge = sum(
        _wire_bytes(item.task.rows, item.task.prefix, itemsize)
        for item in stabilization_prepared
    )
    largest_task_memory = max(
        task.rows * (16 + 3 * task.prefix * itemsize) for task in tasks
    )
    wave_memory = total_id_bytes + workers * largest_task_memory
    prewarm_charge = total_charge if cache_mode == "warm" else 0
    budget.reserve(
        stabilization_charge + prewarm_charge + total_charge,
        extra_memory_bytes=wave_memory,
    )

    clients: list[throwdb.RemoteCollection] = []
    try:
        for _ in range(workers):
            clients.append(
                throwdb.RemoteCollection.connect(
                    endpoint,
                    timeout=budget.rpc_timeout(rpc_timeout),
                    scan_timeout=budget.rpc_timeout(rpc_timeout),
                )
            )
    except BaseException:
        for client in clients:
            client.close()
        raise
    barrier_timeout = budget.rpc_timeout(rpc_timeout)

    def execute_wave(
        executor: ThreadPoolExecutor,
        *,
        collect_samples: bool,
    ) -> tuple[list[ConcurrentSample], int, int]:
        pending: queue.Queue[PreparedGetTask] = queue.Queue()
        for item in prepared:
            pending.put(item)
        samples: list[ConcurrentSample] = []
        samples_lock = threading.Lock()
        ready_clock = [0]

        def mark_ready() -> None:
            ready_clock[0] = time.perf_counter_ns()

        start_barrier = threading.Barrier(workers + 1, action=mark_ready)

        def worker(remote: throwdb.RemoteCollection) -> None:
            start_barrier.wait(timeout=barrier_timeout)
            while True:
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    return
                dequeued = time.perf_counter_ns()
                task = item.task
                _set_remote_timeout(remote, budget, rpc_timeout)
                rpc_started = time.perf_counter_ns()
                result = remote.get(
                    item.ids,
                    prefix_dimensions=(
                        None if task.prefix == dimension else task.prefix
                    ),
                )
                finished = time.perf_counter_ns()
                if result.shape != (task.rows, task.prefix):
                    raise AssertionError(
                        "concurrent Get returned an invalid shape"
                    )
                if collect_samples:
                    sample = ConcurrentSample(
                        task_ordinal=item.ordinal,
                        label=task.label,
                        rows=task.rows,
                        ready_ns=ready_clock[0],
                        dequeued_ns=dequeued,
                        rpc_started_ns=rpc_started,
                        finished_ns=finished,
                    )
                    with samples_lock:
                        samples.append(sample)

        futures = [executor.submit(worker, client) for client in clients]
        start_barrier.wait(timeout=barrier_timeout)
        for future in futures:
            future.result()
        finished_ns = max(
            (sample.finished_ns for sample in samples),
            default=time.perf_counter_ns(),
        )
        return samples, ready_clock[0], finished_ns

    cache_preparation: dict[str, object] | None = None
    warm_preparation: dict[str, object] = {
        "kind": "none",
        "calls": 0,
        "accounted_wire_bytes": 0,
    }
    stabilization_before = telemetry_snapshot(server_pid)
    stabilization_telemetry: dict[str, Any] | None = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Create every client worker before cache preparation/telemetry so
            # cold and warm process counters have the same runtime topology.
            stabilization = threading.Barrier(workers + 1)
            stabilization_futures = [
                executor.submit(stabilization.wait, barrier_timeout)
                for _ in range(workers)
            ]
            stabilization.wait(timeout=barrier_timeout)
            for future in stabilization_futures:
                future.result()
            rpc_stabilization = threading.Barrier(workers)

            def stabilize_rpc(client: throwdb.RemoteCollection) -> None:
                rpc_stabilization.wait(timeout=barrier_timeout)
                _set_remote_timeout(client, budget, rpc_timeout)
                client.health()

            rpc_futures = [
                executor.submit(stabilize_rpc, client) for client in clients
            ]
            for future in rpc_futures:
                future.result()
            # Exercise one Get per connected worker before either controlled
            # cache preparation. Cold then evicts the embedding payload pages;
            # warm performs its exact full-corpus replay. Thus both modes enter
            # measurement with an initialized client/server Get path and the
            # same worker topology, without hiding this traffic from budgets.
            get_stabilization = threading.Barrier(workers)

            def stabilize_get(
                client: throwdb.RemoteCollection,
                item: PreparedGetTask,
            ) -> None:
                get_stabilization.wait(timeout=barrier_timeout)
                _set_remote_timeout(client, budget, rpc_timeout)
                task = item.task
                result = client.get(
                    item.ids,
                    prefix_dimensions=(
                        None if task.prefix == dimension else task.prefix
                    ),
                )
                if result.shape != (task.rows, task.prefix):
                    raise AssertionError(
                        "concurrent stabilization Get returned an invalid shape"
                    )

            get_futures = [
                executor.submit(stabilize_get, client, item)
                for client, item in zip(
                    clients, stabilization_prepared, strict=True
                )
            ]
            for future in get_futures:
                future.result()
            stabilization_telemetry = telemetry_delta(
                stabilization_before,
                telemetry_snapshot(server_pid),
            )
            if cache_mode == "cold":
                assert prepare_cold is not None
                cache_preparation = prepare_cold()
            elif cache_mode == "warm":
                execute_wave(executor, collect_samples=False)
                warm_preparation = {
                    "kind": "exact_query_replay",
                    "calls": len(prepared),
                    "accounted_wire_bytes": prewarm_charge,
                    "query_digest_sha256": digest.hexdigest(),
                }
            before = telemetry_snapshot(server_pid)
            samples, ready_ns, finished_ns = execute_wave(
                executor, collect_samples=True
            )
            after = telemetry_snapshot(server_pid)
    finally:
        for client in clients:
            client.close()
    seconds = (finished_ns - ready_ns) / 1_000_000_000
    if len(samples) != len(tasks):
        raise RuntimeError("concurrent scenario did not complete every task")
    timing_summary = summarize_concurrent_samples(
        samples,
        seconds=seconds,
        itemsize=itemsize,
        prefix=tasks[0].prefix,
    )
    total_rows = sum(task.rows for task in tasks)
    total_useful = sum(task.rows * task.prefix * itemsize for task in tasks)
    return {
        "workers": workers,
        "seconds": seconds,
        "calls": len(tasks),
        "rows": total_rows,
        "rows_per_second": total_rows / seconds if seconds else 0.0,
        "useful_embedding_bytes": total_useful,
        "useful_gib_per_second": (
            total_useful / seconds / GIB if seconds else 0.0
        ),
        "accounted_wire_bytes": total_charge,
        "measured_accounted_wire_bytes": total_charge,
        "stabilization_accounted_wire_bytes": stabilization_charge,
        "prewarm_accounted_wire_bytes": prewarm_charge,
        "total_accounted_wire_bytes": (
            stabilization_charge + total_charge + prewarm_charge
        ),
        "query_digest_sha256": digest.hexdigest(),
        "class_query_digests_sha256": class_digests,
        "cache_mode": cache_mode,
        "cache_preparation": cache_preparation,
        "warm_preparation": warm_preparation,
        "admission_model": "shared_client_fifo_sync",
        "initial_client_queue_depth": len(tasks),
        "max_in_flight": workers,
        "timing_semantics": {
            "latency": "client_observed_rpc_round_trip",
            "queue_wait": "client FIFO before a worker dequeues the call",
            "sojourn": "wave ready until the RPC response is available",
            "server_internal_timing_available": False,
        },
        "stabilization_telemetry": stabilization_telemetry,
        "stabilization_preparation": {
            "kind": "one_get_per_connected_worker",
            "calls": len(stabilization_prepared),
            "accounted_wire_bytes": stabilization_charge,
        },
        **timing_summary,
        "telemetry": telemetry_delta(before, after),
    }


def correctness_check(
    remote: throwdb.RemoteCollection,
    population: np.ndarray,
    *,
    rows: int,
    seed: int,
    source: np.ndarray | None,
    embedded: throwdb.Collection | None,
    budget: ResourceBudget,
    rpc_timeout: float,
) -> dict[str, Any]:
    sample_rows = min(rows, len(population))
    row_bytes = remote.dimension * np.dtype(remote.dtype).itemsize
    budget.check(extra_memory_bytes=sample_rows * (16 + 4 * row_bytes))
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(population), size=sample_rows, dtype=np.int64)
    ids = canonical_ids(population[positions])
    budget.reserve(
        _wire_bytes(sample_rows, remote.dimension, np.dtype(remote.dtype).itemsize),
        extra_memory_bytes=4 * sample_rows * row_bytes,
    )
    _set_remote_timeout(remote, budget, rpc_timeout)
    actual = remote.get(ids)
    if source is not None:
        expected = np.asarray(source[positions], dtype=remote.dtype, order="C")
        oracle = "source_embeddings"
    elif embedded is not None:
        expected = embedded.get(ids)
        oracle = "embedded_collection"
    else:  # pragma: no cover - guarded by main
        raise RuntimeError("correctness check has no oracle")
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError("remote result metadata differs from correctness oracle")
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    return {
        "status": "passed",
        "oracle": oracle,
        "sample_rows": sample_rows,
        "bit_exact": True,
        "query_digest_sha256": hashlib.sha256(
            memoryview(ids).cast("B")
        ).hexdigest(),
    }


def run_scan(
    remote: throwdb.RemoteCollection,
    *,
    max_rows: int,
    batch_rows: int,
    budget: ResourceBudget,
    rpc_timeout: float,
    server_pid: int | None,
) -> dict[str, Any]:
    if max_rows == 0:
        return {"status": "skipped", "reason": "--scan-rows is zero"}
    _set_remote_timeout(remote, budget, rpc_timeout)
    planned_rows = min(max_rows, remote.describe().count)
    # IDs and two uint32 location fields.
    budget.reserve(planned_rows * 16, extra_memory_bytes=batch_rows * 32)
    remote._scan_timeout = budget.rpc_timeout(rpc_timeout)  # type: ignore[attr-defined]
    digest = hashlib.sha256()
    emitted = 0
    previous_location: tuple[int, int] | None = None
    snapshot: dict[str, Any] | None = None
    before = telemetry_snapshot(server_pid)
    started = time.perf_counter()
    with remote.scan_ids(batch_rows=batch_rows, with_locations=True) as scan:
        snapshot = asdict(scan.snapshot)
        for item in scan:
            budget.check(extra_memory_bytes=batch_rows * 32)
            assert isinstance(item, throwdb.LocatedIdBatch)
            take = min(item.ids.size, planned_rows - emitted)
            if take <= 0:
                break
            ids = canonical_ids(item.ids[:take])
            locations = item.locations[:take]
            if item.emitted_offset != emitted:
                raise AssertionError("ScanIds offsets are not contiguous")
            if locations.size:
                first_location = (
                    int(locations[0]["segment_id"]),
                    int(locations[0]["row_index"]),
                )
                if (
                    previous_location is not None
                    and first_location <= previous_location
                ):
                    raise AssertionError(
                        "ScanIds physical locations are not strictly increasing"
                    )
                segments = locations["segment_id"]
                row_indexes = locations["row_index"]
                invalid = (segments[1:] < segments[:-1]) | (
                    (segments[1:] == segments[:-1])
                    & (row_indexes[1:] <= row_indexes[:-1])
                )
                if bool(np.any(invalid)):
                    raise AssertionError(
                        "ScanIds physical locations are not strictly increasing"
                    )
                previous_location = (
                    int(segments[-1]),
                    int(row_indexes[-1]),
                )
            digest.update(memoryview(ids).cast("B"))
            digest.update(memoryview(locations).cast("B"))
            emitted += take
            if emitted == planned_rows:
                break
    seconds = time.perf_counter() - started
    after = telemetry_snapshot(server_pid)
    return {
        "status": "passed",
        "requested_max_rows": max_rows,
        "batch_rows": batch_rows,
        "rows": emitted,
        "snapshot": snapshot,
        "bounded_prefix_of_snapshot": emitted < int(snapshot["count"]),
        "physical_locations_strictly_increasing": True,
        "digest_sha256": digest.hexdigest(),
        "seconds": seconds,
        "rows_per_second": emitted / seconds if seconds else 0.0,
        "telemetry": telemetry_delta(before, after),
    }


def _different_payload(original: np.ndarray, byte_offset: int) -> np.ndarray:
    changed = np.array(original, copy=True, order="C")
    raw = changed.view(np.uint8).reshape(changed.shape[0], -1)
    raw[:, byte_offset % raw.shape[1]] ^= np.uint8(1)
    return changed


def run_crud(
    remote: throwdb.RemoteCollection,
    population: np.ndarray,
    *,
    rows: int,
    seed: int,
    budget: ResourceBudget,
    rpc_timeout: float,
    allow_nondurable: bool,
) -> dict[str, Any]:
    if rows == 0:
        return {"status": "skipped", "reason": "--crud-rows is zero"}
    _set_remote_timeout(remote, budget, rpc_timeout)
    description = remote.describe()
    if not description.writable:
        raise RuntimeError("CRUD benchmark requires a writable server")
    if not description.durable_idempotency and not allow_nondurable:
        raise RuntimeError(
            "CRUD benchmark requires durable idempotency; pass "
            "--allow-nondurable-crud only on a disposable collection"
        )
    if rows > len(population):
        raise ValueError("--crud-rows cannot exceed the ID population")
    budget.check(
        extra_memory_bytes=(
            rows * 256
            + 6 * rows * remote.dimension * np.dtype(remote.dtype).itemsize
        )
    )
    ids = sampled_ids(population, rows, seed, budget=budget)
    if np.unique(ids).size != ids.size:
        # Pick a small unique sample without allocating a permutation of a
        # potentially 600M-row population.
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        seen: set[int] = set()
        max_attempts = max(1_000, rows * 100)
        for attempt in range(max_attempts):
            if attempt % 256 == 0:
                budget.check()
            value = int(population[int(rng.integers(0, len(population)))])
            if value not in seen:
                seen.add(value)
                selected.append(value)
                if len(selected) == rows:
                    break
        ids = canonical_ids(np.asarray(selected, dtype=np.uint64))
    if np.unique(ids).size != ids.size:
        raise RuntimeError("could not choose unique IDs for CRUD benchmark")
    itemsize = np.dtype(remote.dtype).itemsize
    full_payload = rows * (8 + remote.dimension * itemsize)
    ids_payload = rows * 8
    # Maximum dominant payload for the normal cycle plus one emergency upsert:
    # 8 ID+embedding operations and 2 ID-only operations. Reserving it before
    # the first mutation guarantees cleanup cannot cross the byte budget.
    budget.reserve(
        8 * full_payload + 2 * ids_payload,
        extra_memory_bytes=6 * rows * remote.dimension * itemsize,
    )
    _set_remote_timeout(remote, budget, rpc_timeout)
    original = remote.get(ids)
    changed_update = _different_payload(original, 0)
    changed_insert = _different_payload(original, 1)
    operations: list[dict[str, Any]] = []

    def operation_id(name: str) -> uuid.UUID:
        identifier = uuid.uuid4()
        operations.append({"name": name, "operation_id": str(identifier)})
        return identifier

    restored = False
    failure: BaseException | None = None
    started = time.perf_counter()
    try:
        _set_remote_timeout(remote, budget, rpc_timeout)
        remote.update(
            ids, changed_update, operation_id=operation_id("update_changed")
        )
        _set_remote_timeout(remote, budget, rpc_timeout)
        np.testing.assert_array_equal(
            remote.get(ids).view(np.uint8), changed_update.view(np.uint8)
        )
        _set_remote_timeout(remote, budget, rpc_timeout)
        deleted = remote.delete(ids, operation_id=operation_id("delete"))
        if not bool(deleted.all()):
            raise AssertionError("delete did not remove every selected ID")
        try:
            _set_remote_timeout(remote, budget, rpc_timeout)
            remote.get(ids)
        except throwdb.RemoteNotFoundError:
            pass
        else:
            raise AssertionError("deleted IDs remained readable")
        _set_remote_timeout(remote, budget, rpc_timeout)
        remote.insert(
            ids, changed_insert, operation_id=operation_id("insert_changed")
        )
        _set_remote_timeout(remote, budget, rpc_timeout)
        np.testing.assert_array_equal(
            remote.get(ids).view(np.uint8), changed_insert.view(np.uint8)
        )
        _set_remote_timeout(remote, budget, rpc_timeout)
        remote.upsert(ids, original, operation_id=operation_id("upsert_restore"))
        _set_remote_timeout(remote, budget, rpc_timeout)
        np.testing.assert_array_equal(
            remote.get(ids).view(np.uint8), original.view(np.uint8)
        )
        restored = True
    except Exception as error:
        failure = error
    finally:
        if not restored:
            # Best-effort logical restore. The append-only WAL history remains,
            # which is intentionally disclosed in the report/documentation.
            # Cleanup may use a short grace period after the measured work
            # deadline: leaving user data mutated is worse than a late report.
            with contextlib.suppress(Exception):
                remote._timeout = min(rpc_timeout, 10.0)  # type: ignore[attr-defined]
                remote.upsert(
                    ids,
                    original,
                    operation_id=operation_id("emergency_upsert_restore"),
                )
                restored = True
    elapsed = time.perf_counter() - started
    details = {
        "status": "passed" if failure is None and restored else "failed",
        "rows": rows,
        "seconds": elapsed,
        "logical_contents_restored": restored,
        "wal_history_changed": True,
        "durable_idempotency": description.durable_idempotency,
        "operations": operations,
        "emergency_cleanup_grace_seconds": min(rpc_timeout, 10.0),
    }
    if failure is not None:
        details["error"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
        raise CrudBenchmarkError(
            "CRUD cycle failed; logical restoration "
            + ("succeeded" if restored else "also failed"),
            details,
        ) from failure
    if not restored:
        raise CrudBenchmarkError(
            "CRUD benchmark could not restore original values",
            details,
        )
    return details


def _observed_cache_classification(trial: dict[str, Any]) -> str:
    observation = trial.get("cache_observation") or {}
    return str(observation.get("classification", "unknown"))


def _expected_cache_classification(cache_mode: str) -> str | None:
    return {
        "cold": "backing_storage",
        "warm": "page_cache",
    }.get(cache_mode)


def compare_baseline(
    remote_results: Sequence[dict[str, Any]],
    local_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(result: dict[str, Any]) -> tuple[object, ...]:
        return (
            result["label"],
            result.get("repetition", 1),
            result.get("cache_mode", "ambient"),
            result.get("query_digest_sha256"),
        )

    local_by_key = {key(result): result for result in local_results}
    comparisons: list[dict[str, Any]] = []
    for remote in remote_results:
        local = local_by_key[key(remote)]
        cache_mode = str(remote.get("cache_mode", "ambient"))
        remote_cache = _observed_cache_classification(remote)
        local_cache = _observed_cache_classification(local)
        expected_cache = _expected_cache_classification(cache_mode)
        cache_pair_verified = (
            remote_cache == expected_cache and local_cache == expected_cache
            if expected_cache is not None
            else None
        )
        ratios_are_comparable = cache_pair_verified is not False
        comparisons.append(
            {
                "label": remote["label"],
                "repetition": remote.get("repetition", 1),
                "cache_mode": cache_mode,
                "query_digest_sha256": remote.get("query_digest_sha256"),
                "remote_cache_classification": remote_cache,
                "embedded_cache_classification": local_cache,
                "expected_cache_classification": expected_cache,
                "cache_pair_verified": cache_pair_verified,
                "comparison_status": (
                    "verified_controlled_cache"
                    if cache_pair_verified is True
                    else (
                        "unverified_cache_pair"
                        if cache_pair_verified is False
                        else "uncontrolled_ambient_cache"
                    )
                ),
                "remote_over_embedded_wall_time": (
                    remote["seconds"] / local["seconds"]
                    if ratios_are_comparable and local["seconds"]
                    else None
                ),
                "remote_over_embedded_rows_per_second": (
                    remote["rows_per_second"] / local["rows_per_second"]
                    if ratios_are_comparable and local["rows_per_second"]
                    else None
                ),
                "remote_over_embedded_p95_latency": (
                    remote["latency_p95_ms"] / local["latency_p95_ms"]
                    if ratios_are_comparable and local["latency_p95_ms"]
                    else None
                ),
            }
        )
    return comparisons


def _numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("numeric distribution requires at least one sample")
    mean = statistics.fmean(samples)
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "mean": mean,
        "min": min(samples),
        "max": max(samples),
        "coefficient_of_variation": (
            statistics.pstdev(samples) / mean if len(samples) > 1 and mean else 0.0
        ),
    }


def _cache_verification_summary(
    trials: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Describe observed cache evidence without relabelling requested modes."""

    classifications = [
        _observed_cache_classification(trial) for trial in trials
    ]
    requested_modes = {
        str(trial.get("cache_mode", "ambient")) for trial in trials
    }
    requested = next(iter(requested_modes)) if len(requested_modes) == 1 else None
    expected = (
        _expected_cache_classification(requested)
        if requested is not None
        else None
    )
    verified: bool | None = None
    if expected is not None:
        verified = bool(classifications) and all(
            classification == expected for classification in classifications
        )
    return {
        "observed_cache_classifications": classifications,
        "cache_verified": verified,
        "cache_verification_status": (
            f"{requested}_{'verified' if verified else 'unverified'}"
            if expected is not None
            else "not_applicable"
        ),
        "cold_unverified": requested == "cold" and verified is not True,
    }


def aggregate_trials(
    trials: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate repeated trials while retaining variation explicitly."""

    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for trial in trials:
        groups.setdefault(
            tuple(trial.get(field) for field in group_fields), []
        ).append(trial)
    summaries: list[dict[str, Any]] = []
    metric_fields = (
        "rows_per_second",
        "useful_gib_per_second",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_max_ms",
        *(
            f"{scope}_{percentile}"
            for scope in (
                "client_queue_wait",
                "client_prepare",
                "rpc_round_trip",
                "client_sojourn",
            )
            for percentile in ("p50_ms", "p95_ms", "p99_ms", "max_ms")
        ),
    )
    for identity, selected in groups.items():
        total_seconds = sum(float(item["seconds"]) for item in selected)
        total_rows = sum(int(item["rows"]) for item in selected)
        total_useful = sum(
            int(item["useful_embedding_bytes"]) for item in selected
        )
        row: dict[str, Any] = dict(zip(group_fields, identity, strict=True))
        row.update(
            {
                "repeat_count": len(selected),
                "rows": total_rows,
                "seconds": total_seconds,
                "rows_per_second": (
                    total_rows / total_seconds if total_seconds else 0.0
                ),
                "useful_embedding_bytes": total_useful,
                "useful_gib_per_second": (
                    total_useful / total_seconds / GIB if total_seconds else 0.0
                ),
                "query_digests_sha256": [
                    str(item["query_digest_sha256"]) for item in selected
                ],
                "aggregation": {
                    "throughput": "total work divided by total trial wall time",
                    "latency": "median of per-trial percentile estimates",
                },
                "trial_variation": {},
                **_cache_verification_summary(selected),
            }
        )
        for field in metric_fields:
            values = [
                float(item[field]) for item in selected if field in item
            ]
            if not values:
                continue
            distribution = _numeric_distribution(values)
            row["trial_variation"][field] = distribution
            if field not in {"rows_per_second", "useful_gib_per_second"}:
                row[field] = distribution["median"]
        summaries.append(row)
    return summaries


def concurrent_class_trials(
    trials: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for trial in trials:
        for label, class_result in trial["classes"].items():
            timing_fields = {
                f"{scope}_{percentile}": value
                for scope, distribution in class_result.get(
                    "timings", {}
                ).items()
                for percentile, value in distribution.items()
            }
            flattened.append(
                {
                    "workload": trial.get("workload"),
                    "workers": trial["workers"],
                    "batch_size": trial.get("batch_size"),
                    "cache_mode": trial["cache_mode"],
                    "repetition": trial["repetition"],
                    "class_label": label,
                    "cache_observation": trial.get("cache_observation"),
                    "query_digest_sha256": trial[
                        "class_query_digests_sha256"
                    ][label],
                    **{
                        key: value
                        for key, value in class_result.items()
                        if key != "timings"
                    },
                    **timing_fields,
                }
            )
    return flattened


def aggregate_fairness_trials(
    trials: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        groups.setdefault(str(trial["cache_mode"]), []).append(trial)
    result: list[dict[str, Any]] = []
    for cache_mode, selected in groups.items():
        reference_statuses = [
            trial["fairness"].get("cache_reference_set_verified")
            for trial in selected
        ]
        expected_cache = _expected_cache_classification(cache_mode)
        metric_names = sorted(
            {
                key
                for trial in selected
                for key, value in trial["fairness"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        result.append(
            {
                "cache_mode": cache_mode,
                "repeat_count": len(selected),
                "metrics": {
                    name: _numeric_distribution(
                        [
                            float(trial["fairness"][name])
                            for trial in selected
                            if isinstance(
                                trial["fairness"].get(name), (int, float)
                            )
                            and not isinstance(
                                trial["fairness"].get(name), bool
                            )
                        ]
                    )
                    for name in metric_names
                },
                "interpretation": selected[0]["fairness"]["interpretation"],
                "matched_reference_cache_verified": (
                    all(status is True for status in reference_statuses)
                    if expected_cache is not None
                    else None
                ),
                "matched_reference_cache_statuses": reference_statuses,
                **_cache_verification_summary(selected),
            }
        )
    return result


def matched_fairness_summary(
    mixed: dict[str, Any],
    isolated_small: dict[str, Any],
    isolated_large: dict[str, Any],
) -> dict[str, Any]:
    """Compare mixed classes with exact-corpus, same-inflight references."""

    for reference in (isolated_small, isolated_large):
        for field in ("workers", "cache_mode", "repetition"):
            if reference.get(field) != mixed.get(field):
                raise AssertionError(
                    f"mixed fairness reference differs in {field}"
                )
    mixed_classes = mixed["classes"]
    small = mixed_classes["small"]
    large = mixed_classes["large"]
    small_reference = isolated_small["classes"]["small"]
    large_reference = isolated_large["classes"]["large"]
    for label, reference in (
        ("small", isolated_small),
        ("large", isolated_large),
    ):
        if mixed["class_query_digests_sha256"][label] != reference[
            "class_query_digests_sha256"
        ][label]:
            raise AssertionError(
                f"mixed {label} reference did not replay the exact corpus"
            )

    cache_mode = str(mixed.get("cache_mode", "ambient"))
    expected_cache = _expected_cache_classification(cache_mode)
    cache_classifications = {
        "mixed": _observed_cache_classification(mixed),
        "isolated_small": _observed_cache_classification(isolated_small),
        "isolated_large": _observed_cache_classification(isolated_large),
    }
    cache_reference_set_verified = (
        all(
            classification == expected_cache
            for classification in cache_classifications.values()
        )
        if expected_cache is not None
        else None
    )
    matched_ratios_are_comparable = cache_reference_set_verified is not False

    def ratio(value: float, baseline: float) -> float | None:
        return value / baseline if baseline else None

    def matched_ratio(value: float, baseline: float) -> float | None:
        if not matched_ratios_are_comparable:
            return None
        return ratio(value, baseline)

    small_sojourn = small["timings"]["client_sojourn"]["p95_ms"]
    large_sojourn = large["timings"]["client_sojourn"]["p95_ms"]
    small_reference_sojourn = small_reference["timings"][
        "client_sojourn"
    ]["p95_ms"]
    large_reference_sojourn = large_reference["timings"][
        "client_sojourn"
    ]["p95_ms"]
    return {
        "reference": "exact class corpus with the same synchronous inflight bound",
        "server_fairness_isolated": False,
        "cache_classifications": cache_classifications,
        "expected_cache_classification": expected_cache,
        "cache_reference_set_verified": cache_reference_set_verified,
        "cache_comparison_status": (
            "verified_controlled_cache"
            if cache_reference_set_verified is True
            else (
                "unverified_cache_reference_set"
                if cache_reference_set_verified is False
                else "uncontrolled_ambient_cache"
            )
        ),
        "interpretation": (
            "RPC round trip includes transport and server work; client sojourn "
            "also includes the harness FIFO. Server-internal queue/service "
            "timing is not available in this protocol revision."
        ),
        "small_rpc_p95_slowdown_vs_matched_isolated": matched_ratio(
            small["latency_p95_ms"], small_reference["latency_p95_ms"]
        ),
        "large_rpc_p95_slowdown_vs_matched_isolated": matched_ratio(
            large["latency_p95_ms"], large_reference["latency_p95_ms"]
        ),
        "small_sojourn_p95_slowdown_vs_matched_isolated": matched_ratio(
            small_sojourn, small_reference_sojourn
        ),
        "large_sojourn_p95_slowdown_vs_matched_isolated": matched_ratio(
            large_sojourn, large_reference_sojourn
        ),
        "large_over_small_rpc_p95": ratio(
            large["latency_p95_ms"], small["latency_p95_ms"]
        ),
        "large_over_small_sojourn_p95": ratio(
            large_sojourn, small_sojourn
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded ThrowDB gRPC correctness/load benchmark. By default the "
            "public tune-aware launcher starts a loopback server."
        )
    )
    parser.add_argument(
        "--store",
        "--storage",
        dest="store",
        type=Path,
        help=(
            "collection path for local launch/baseline; optional for an "
            "external endpoint with source-backed correctness"
        ),
    )
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-embeddings", type=Path)
    parser.add_argument("--source-kind", choices=("npy", "raw"), default="npy")
    parser.add_argument("--endpoint", help="measure an existing server instead")
    parser.add_argument("--server-pid", type=_positive_int)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument(
        "--tuning",
        choices=("auto", "persisted", "conservative"),
        default="auto",
    )
    parser.add_argument("--server-start-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--rpc-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--server-max-worker-threads", type=_positive_int)
    parser.add_argument("--server-max-ids-per-request", type=_positive_int)
    parser.add_argument("--server-max-scan-batch-rows", type=_positive_int)
    parser.add_argument("--server-maintenance-interval-seconds", type=_positive_int)
    parser.add_argument("--server-no-auto-maintenance", action="store_true")
    parser.add_argument(
        "--server-read-log-dir",
        type=Path,
        help=(
            "enable bounded point-read logging for a locally launched server "
            "in this dedicated directory"
        ),
    )
    parser.add_argument(
        "--server-read-log-id-chunk-rows",
        type=_read_log_id_chunk_rows,
    )
    parser.add_argument("--server-read-log-buffer-mib", type=_positive_int)
    parser.add_argument("--server-read-log-segment-mib", type=_positive_int)
    parser.add_argument("--server-read-log-rotate-seconds", type=_positive_int)
    parser.add_argument("--server-read-log-retention-mib", type=_positive_int)
    parser.add_argument(
        "--server-read-log-retention-seconds",
        type=_positive_int,
    )
    parser.add_argument(
        "--server-read-log-min-free-mib",
        type=_nonnegative_int,
    )
    parser.add_argument(
        "--server-read-log-max-write-mib-per-second",
        type=_positive_int,
    )
    parser.add_argument(
        "--server-read-log-max-retained-files",
        type=_positive_int,
    )
    parser.add_argument(
        "--server-read-log-max-event-drop-ratio",
        type=_unit_interval,
        help=(
            "inspection suitability threshold for completed event outcomes "
            "lost "
            "(default: 0.05 when read logging is enabled)"
        ),
    )
    parser.add_argument(
        "--server-read-log-max-row-drop-ratio",
        type=_unit_interval,
        help=(
            "inspection suitability threshold for completed row outcomes "
            "lost "
            "(default: 0.05 when read logging is enabled)"
        ),
    )
    parser.add_argument("--max-seconds", type=_positive_float, default=300.0)
    parser.add_argument("--max-transfer-gib", type=_positive_float, default=16.0)
    parser.add_argument(
        "--max-client-memory-mib",
        type=_positive_float,
        default=4096.0,
    )
    parser.add_argument("--seed", type=_uint64_int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cache-mode",
        choices=("auto", "ambient", "cold", "warm", "both"),
        default="auto",
        help=(
            "auto selects paired cold/warm for a locally launched server and "
            "exact-query warm replay for an external endpoint"
        ),
    )
    parser.add_argument(
        "--repeats",
        "--repetitions",
        dest="repeats",
        type=_positive_int,
        default=3,
        help="paired deterministic measurement repetitions (default: 3)",
    )
    parser.add_argument(
        "--batch-sizes", type=parse_positive_int_list, default=(1, 128, 1024)
    )
    parser.add_argument("--prefix-dimensions", default="full,128")
    parser.add_argument("--calls", type=_positive_int, default=100)
    parser.add_argument("--warmup-calls", type=_nonnegative_int, default=2)
    parser.add_argument(
        "--concurrency", type=parse_positive_int_list, default=(1, 2, 4)
    )
    parser.add_argument("--concurrent-batch-size", type=_positive_int, default=1024)
    parser.add_argument(
        "--concurrent-calls-per-worker",
        type=_positive_int,
        default=20,
    )
    parser.add_argument("--mixed-workers", type=_positive_int, default=4)
    parser.add_argument("--mixed-small-batch", type=_positive_int, default=1)
    parser.add_argument("--mixed-large-batch", type=_positive_int, default=1024)
    parser.add_argument("--mixed-small-calls", type=_positive_int, default=200)
    parser.add_argument("--mixed-large-calls", type=_positive_int, default=100)
    parser.add_argument("--correctness-rows", type=_positive_int, default=1024)
    parser.add_argument("--scan-rows", type=_nonnegative_int, default=1_000_000)
    parser.add_argument("--scan-batch-rows", type=_positive_int, default=65_536)
    parser.add_argument("--crud-rows", type=_nonnegative_int, default=0)
    parser.add_argument("--allow-nondurable-crud", action="store_true")
    parser.add_argument("--no-embedded-baseline", action="store_true")
    parser.add_argument("--no-require-id-count-match", action="store_true")
    return parser


def _validate_server_read_log_arguments(args: argparse.Namespace) -> None:
    configured_tuning = [
        name
        for name in _SERVER_READ_LOG_TUNING_ARGUMENTS
        if getattr(args, name) is not None
    ]
    enabled = args.server_read_log_dir is not None
    if args.endpoint is not None and (enabled or configured_tuning):
        raise ValueError(
            "--server-read-log-* options are only valid for a locally "
            "launched server"
        )
    if not enabled and configured_tuning:
        rendered = ", ".join(
            "--" + name.replace("_", "-") for name in configured_tuning
        )
        raise ValueError(
            f"{rendered} require --server-read-log-dir; logging remains "
            "disabled unless its directory is explicit"
        )
    if enabled:
        for name in (
            "server_read_log_buffer_mib",
            "server_read_log_segment_mib",
            "server_read_log_retention_mib",
            "server_read_log_min_free_mib",
            "server_read_log_max_write_mib_per_second",
        ):
            value = getattr(args, name)
            if value is not None and value > _MAX_UINT64_MIB:
                raise ValueError(
                    f"--{name.replace('_', '-')} exceeds the uint64 MiB "
                    "range"
                )
        if (
            args.server_read_log_retention_seconds is not None
            and args.server_read_log_retention_seconds > UINT64_MASK
        ):
            raise ValueError(
                "--server-read-log-retention-seconds exceeds the uint64 "
                "range"
            )
        if (
            args.server_read_log_rotate_seconds is not None
            and args.server_read_log_rotate_seconds
                > _MAX_READ_LOG_ROTATE_SECONDS
        ):
            raise ValueError(
                "--server-read-log-rotate-seconds must fit signed "
                "64-bit seconds"
            )
        if (
            args.server_read_log_max_retained_files is not None
            and args.server_read_log_max_retained_files
                > _MAX_READ_LOG_RETAINED_FILES
        ):
            raise ValueError(
                "--server-read-log-max-retained-files must be in "
                "[1, 65533]"
            )
        if args.server_read_log_max_event_drop_ratio is None:
            args.server_read_log_max_event_drop_ratio = (
                _DEFAULT_READ_LOG_MAX_EVENT_DROP_RATIO
            )
        if args.server_read_log_max_row_drop_ratio is None:
            args.server_read_log_max_row_drop_ratio = (
                _DEFAULT_READ_LOG_MAX_ROW_DROP_RATIO
            )


def _checkpoint(
    report_path: Path,
    report: dict[str, Any],
    budget: ResourceBudget,
) -> None:
    report["budget"] = budget.as_dict()
    report["updated_unix_ns"] = time.time_ns()
    load_test.atomic_json_dump(report_path, report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.store is not None:
        args.store = args.store.resolve()
    args.ids = args.ids.resolve()
    args.results = args.results.resolve()
    if args.source_embeddings is not None:
        args.source_embeddings = args.source_embeddings.resolve()
    if args.native_binary is not None:
        args.native_binary = args.native_binary.resolve()
    if args.server_read_log_dir is not None:
        args.server_read_log_dir = args.server_read_log_dir.resolve()
    _validate_server_read_log_arguments(args)
    if args.endpoint is not None and args.native_binary is not None:
        raise ValueError("--native-binary is only valid for a locally launched server")
    if args.endpoint is None and args.store is None:
        raise ValueError("--store is required when launching a local server")
    if args.server_pid is not None and args.endpoint is None:
        raise ValueError("--server-pid is only valid together with --endpoint")
    if args.store is None and not args.no_embedded_baseline:
        raise ValueError(
            "an external endpoint without --store requires "
            "--no-embedded-baseline"
        )
    if args.store is None and args.source_embeddings is None:
        raise ValueError(
            "an external endpoint without --store requires "
            "--source-embeddings for correctness"
        )
    if (
        args.crud_rows > 0
        and args.endpoint is not None
        and not args.allow_nondurable_crud
    ):
        log("external CRUD will still require advertised durable idempotency")
    cache_modes = resolve_cache_modes(
        args.cache_mode,
        locally_launched_server=args.endpoint is None,
    )
    cold_cache_control_supported = hasattr(os, "posix_fadvise") and hasattr(
        os, "POSIX_FADV_DONTNEED"
    )
    cache_auto_fallback: str | None = None
    if (
        args.cache_mode == "auto"
        and "cold" in cache_modes
        and not cold_cache_control_supported
    ):
        cache_modes = ("warm",)
        cache_auto_fallback = (
            "file-scoped POSIX_FADV_DONTNEED is unavailable; auto resolved to warm"
        )
    if "cold" in cache_modes and not cold_cache_control_supported:
        raise ValueError(
            "cold cache mode requires os.posix_fadvise and "
            "POSIX_FADV_DONTNEED on the benchmark host"
        )
    args.results.mkdir(parents=True, exist_ok=False)
    report_path = args.results / "report.json"
    budget = ResourceBudget(
        max_seconds=args.max_seconds,
        max_bytes=int(args.max_transfer_gib * GIB),
        max_client_memory_bytes=int(args.max_client_memory_mib * MIB),
    )
    prepare_cold = (
        cold_cache_preparer(args.store, budget)
        if "cold" in cache_modes and args.store is not None
        else None
    )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "revision": REPORT_REVISION,
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methodology": {
            "cache_modes": list(cache_modes),
            "cache_auto_fallback": cache_auto_fallback,
            "cache_control": (
                "file-scoped POSIX_FADV_DONTNEED; advisory and validated by "
                "per-process backing-read counters when available"
            ),
            "cache_verification": {
                "cold": "every repetition classified as backing_storage",
                "warm": "every repetition classified as page_cache",
                "unknown_or_mixed": "requested cache mode remains unverified",
            },
            "warm_definition": "untimed exact replay of the measured query corpus",
            "concurrent_stabilization": (
                "one untimed Get per connected client before cold eviction or "
                "warm replay; included in the accounted transfer budget"
            ),
            "query_generation": "completed before measurement telemetry and timers",
            "repetitions": args.repeats,
            "execution_order": (
                "phase-major; repetition outermost within each measurement "
                "matrix with deterministic scenario rotation"
            ),
            "concurrent_admission": "bounded synchronous client FIFO",
            "server_internal_timing_available": False,
        },
        "phases": {},
    }
    _checkpoint(report_path, report, budget)
    diskstats = load_test.ProcDiskstatsMonitor(args.results / "diskstats.jsonl")
    vmstat = load_test.CommandMonitor(("vmstat", "1"), args.results / "vmstat.log")
    local_server: LocalServer | None = None
    remote: throwdb.RemoteCollection | None = None
    embedded: throwdb.Collection | None = None
    caught: BaseException | None = None
    try:
        with diskstats, vmstat:
            if args.endpoint is None:
                assert args.store is not None
                budget.check()
                launch_started = time.perf_counter()
                local_server = LocalServer.start(
                    store=args.store,
                    results=args.results,
                    tuning=args.tuning,
                    timeout=min(args.server_start_timeout, budget.remaining_seconds()),
                    writable=args.crud_rows > 0,
                    native_binary=args.native_binary,
                    max_worker_threads=args.server_max_worker_threads,
                    max_ids_per_request=args.server_max_ids_per_request,
                    max_scan_batch_rows=args.server_max_scan_batch_rows,
                    maintenance_interval_seconds=(
                        args.server_maintenance_interval_seconds
                    ),
                    disable_automatic_maintenance=(
                        args.server_no_auto_maintenance
                    ),
                    read_log_directory=args.server_read_log_dir,
                    read_log_id_chunk_rows=(
                        args.server_read_log_id_chunk_rows
                    ),
                    read_log_buffer_mib=args.server_read_log_buffer_mib,
                    read_log_segment_mib=args.server_read_log_segment_mib,
                    read_log_rotate_seconds=(
                        args.server_read_log_rotate_seconds
                    ),
                    read_log_retention_mib=(
                        args.server_read_log_retention_mib
                    ),
                    read_log_retention_seconds=(
                        args.server_read_log_retention_seconds
                    ),
                    read_log_min_free_mib=(
                        args.server_read_log_min_free_mib
                    ),
                    read_log_max_write_mib_per_second=(
                        args.server_read_log_max_write_mib_per_second
                    ),
                    read_log_max_retained_files=(
                        args.server_read_log_max_retained_files
                    ),
                )
                startup_seconds = time.perf_counter() - launch_started
                endpoint = local_server.endpoint
                server_pid = local_server.pid
                report["server"] = {
                    "mode": "local_loopback",
                    "pid": server_pid,
                    "command": local_server.command,
                    "startup_seconds": startup_seconds,
                    "ready": local_server.ready,
                    "stdout_log": str(local_server.stdout_path),
                    "stderr_log": str(local_server.stderr_path),
                }
            else:
                endpoint = args.endpoint
                server_pid = args.server_pid
                report["server"] = {
                    "mode": "external",
                    "endpoint": endpoint,
                    "pid": server_pid,
                    "ready": None,
                }
            _checkpoint(report_path, report, budget)

            connect_started = time.perf_counter()
            remote = throwdb.RemoteCollection.connect(
                endpoint,
                timeout=budget.rpc_timeout(args.rpc_timeout),
                scan_timeout=budget.rpc_timeout(args.rpc_timeout),
            )
            _set_remote_timeout(remote, budget, args.rpc_timeout)
            description = remote.describe()
            report["server"]["connect_and_describe_seconds"] = (
                time.perf_counter() - connect_started
            )
            if local_server is not None:
                validate_ready_description(local_server.ready, description)
            report["server"]["description"] = description_dict(description)
            _set_remote_timeout(remote, budget, args.rpc_timeout)
            report["server"]["health"] = remote.health()
            report["system"] = (
                load_test.collect_system_information(args.store)
                if args.store is not None
                else collect_external_client_information()
            )
            report["system"]["server_telemetry_scope"] = (
                "local /proc PID" if server_pid is not None else "unavailable"
            )
            _checkpoint(report_path, report, budget)

            expected_id_count = description.count
            if args.no_require_id_count_match:
                probe = np.load(args.ids, mmap_mode="r", allow_pickle=False)
                if probe.ndim == 1:
                    expected_id_count = len(probe)
                del probe
            budget.check(
                extra_memory_bytes=(
                    min(expected_id_count, load_test.ID_VALIDATION_CHUNK_ROWS)
                    * 2
                )
            )
            population = load_test.load_ids(args.ids, expected_id_count)
            if (
                not args.no_require_id_count_match
                and len(population) != description.count
            ):
                raise ValueError(
                    "ID population count differs from server Describe count"
                )
            prefixes = parse_prefixes(
                args.prefix_dimensions,
                description.schema.dimension,
            )
            itemsize = np.dtype(description.schema.dtype).itemsize
            source = None
            if args.source_embeddings is not None:
                source = load_test.open_source_embeddings(
                    args.source_embeddings.resolve(),
                    args.source_kind,
                    len(population),
                    description.schema.dimension,
                    description.schema.dtype,
                )
            if args.no_embedded_baseline and source is None:
                raise ValueError(
                    "--no-embedded-baseline requires --source-embeddings so "
                    "correctness still has an oracle"
                )

            # Source-backed correctness can safely run before timings. Without
            # source data the embedded oracle is deliberately deferred until
            # all remote timings have completed, avoiding an artificial cache warmup.
            if source is not None:
                report["phases"]["correctness"] = correctness_check(
                    remote,
                    population,
                    rows=args.correctness_rows,
                    seed=args.seed,
                    source=source,
                    embedded=None,
                    budget=budget,
                    rpc_timeout=args.rpc_timeout,
                )
                _checkpoint(report_path, report, budget)

            sequential: list[dict[str, Any]] = []
            sequential_scenarios = [
                (scenario_index, prefix, batch_size, f"get_b{batch_size}_p{prefix}")
                for scenario_index, (prefix, batch_size) in enumerate(
                    (prefix, batch_size)
                    for prefix in prefixes
                    for batch_size in args.batch_sizes
                )
            ]
            for repetition in range(args.repeats):
                offset = repetition % len(sequential_scenarios)
                ordered_scenarios = (
                    sequential_scenarios[offset:]
                    + sequential_scenarios[:offset]
                )
                for order_position, (
                    scenario_index,
                    prefix,
                    batch_size,
                    label,
                ) in enumerate(ordered_scenarios):
                    pair: list[dict[str, Any]] = []
                    pair_id = f"remote-sequential-r{repetition + 1}-s{scenario_index}"
                    for cache_mode in cache_modes:
                        log(
                            f"running {label}; repeat={repetition + 1}/"
                            f"{args.repeats}; cache={cache_mode}"
                        )
                        result = run_sequential_get(
                            remote,
                            population,
                            label=label,
                            batch_size=batch_size,
                            prefix=prefix,
                            dimension=description.schema.dimension,
                            itemsize=itemsize,
                            calls=args.calls,
                            warmup_calls=args.warmup_calls,
                            seed=derived_seed(
                                args.seed,
                                repetition * 10_000_000
                                + 10_000 * (scenario_index + 1),
                            ),
                            budget=budget,
                            rpc_timeout=args.rpc_timeout,
                            remote=True,
                            server_pid=server_pid,
                            cache_mode=cache_mode,
                            prepare_cold=(
                                prepare_cold if cache_mode == "cold" else None
                            ),
                        )
                        result.update(
                            {
                                "reader": "remote",
                                "repetition": repetition + 1,
                                "scenario_index": scenario_index,
                                "scenario_order_position": order_position,
                                "pair_id": pair_id,
                            }
                        )
                        add_cache_observation(
                            result,
                            record_size=description.record_size,
                            remote=True,
                        )
                        pair.append(result)
                        sequential.append(result)
                        report["phases"]["sequential_get"] = sequential
                    if len({item["query_digest_sha256"] for item in pair}) != 1:
                        raise AssertionError(
                            "paired sequential cache modes changed the query corpus"
                        )
                    _checkpoint(report_path, report, budget)

            concurrency_results: list[dict[str, Any]] = []
            full_prefix = description.schema.dimension
            for repetition in range(args.repeats):
                worker_order = list(args.concurrency)
                offset = repetition % len(worker_order)
                worker_order = worker_order[offset:] + worker_order[:offset]
                for order_position, worker_count in enumerate(worker_order):
                    task_count = args.concurrent_calls_per_worker * worker_count
                    tasks = [
                        GetTask(
                            label="get",
                            rows=args.concurrent_batch_size,
                            prefix=full_prefix,
                            seed=derived_seed(
                                args.seed,
                                1_000_000
                                + repetition * 10_000_000
                                + call,
                            ),
                        )
                        for call in range(task_count)
                    ]
                    pair: list[dict[str, Any]] = []
                    pair_id = (
                        f"concurrent-r{repetition + 1}-w{worker_count}"
                    )
                    for cache_mode in cache_modes:
                        log(
                            "running concurrent Get; "
                            f"clients={worker_count}; repeat={repetition + 1}/"
                            f"{args.repeats}; cache={cache_mode}"
                        )
                        result = run_concurrent_get(
                            endpoint,
                            population,
                            tasks,
                            dimension=full_prefix,
                            itemsize=itemsize,
                            workers=worker_count,
                            budget=budget,
                            rpc_timeout=args.rpc_timeout,
                            server_pid=server_pid,
                            cache_mode=cache_mode,
                            prepare_cold=(
                                prepare_cold if cache_mode == "cold" else None
                            ),
                        )
                        result.update(
                            {
                                "workload": "homogeneous",
                                "batch_size": args.concurrent_batch_size,
                                "repetition": repetition + 1,
                                "scenario_order_position": order_position,
                                "pair_id": pair_id,
                            }
                        )
                        add_cache_observation(
                            result,
                            record_size=description.record_size,
                            remote=True,
                        )
                        pair.append(result)
                        concurrency_results.append(result)
                        report["phases"]["concurrent_get"] = (
                            concurrency_results
                        )
                    if len({item["query_digest_sha256"] for item in pair}) != 1:
                        raise AssertionError(
                            "paired concurrent cache modes changed the query corpus"
                        )
                    _checkpoint(report_path, report, budget)

            mixed_results: list[dict[str, Any]] = []
            isolated_results: list[dict[str, Any]] = []
            for repetition in range(args.repeats):
                mixed_tasks = [
                    *[
                        GetTask(
                            "small",
                            args.mixed_small_batch,
                            full_prefix,
                            derived_seed(
                                args.seed,
                                2_000_000
                                + repetition * 10_000_000
                                + call,
                            ),
                        )
                        for call in range(args.mixed_small_calls)
                    ],
                    *[
                        GetTask(
                            "large",
                            args.mixed_large_batch,
                            full_prefix,
                            derived_seed(
                                args.seed,
                                3_000_000
                                + repetition * 10_000_000
                                + call,
                            ),
                        )
                        for call in range(args.mixed_large_calls)
                    ],
                ]
                permutation = np.random.default_rng(
                    derived_seed(
                        args.seed,
                        4_000_000 + repetition * 10_000_000,
                    )
                ).permutation(len(mixed_tasks))
                mixed_tasks = [mixed_tasks[int(index)] for index in permutation]
                workloads: list[tuple[str, list[GetTask]]] = [
                    (
                        "isolated_small",
                        [task for task in mixed_tasks if task.label == "small"],
                    ),
                    (
                        "isolated_large",
                        [task for task in mixed_tasks if task.label == "large"],
                    ),
                    ("mixed", mixed_tasks),
                ]
                offset = repetition % len(workloads)
                workloads = workloads[offset:] + workloads[:offset]
                repetition_results: dict[tuple[str, str], dict[str, Any]] = {}
                for order_position, (workload, tasks) in enumerate(workloads):
                    pair: list[dict[str, Any]] = []
                    pair_id = f"{workload}-r{repetition + 1}"
                    for cache_mode in cache_modes:
                        log(
                            f"running {workload}; workers={args.mixed_workers}; "
                            f"repeat={repetition + 1}/{args.repeats}; "
                            f"cache={cache_mode}"
                        )
                        result = run_concurrent_get(
                            endpoint,
                            population,
                            tasks,
                            dimension=full_prefix,
                            itemsize=itemsize,
                            workers=args.mixed_workers,
                            budget=budget,
                            rpc_timeout=args.rpc_timeout,
                            server_pid=server_pid,
                            cache_mode=cache_mode,
                            prepare_cold=(
                                prepare_cold if cache_mode == "cold" else None
                            ),
                        )
                        result.update(
                            {
                                "workload": workload,
                                "repetition": repetition + 1,
                                "scenario_order_position": order_position,
                                "pair_id": pair_id,
                            }
                        )
                        add_cache_observation(
                            result,
                            record_size=description.record_size,
                            remote=True,
                        )
                        pair.append(result)
                        repetition_results[(workload, cache_mode)] = result
                        if workload == "mixed":
                            mixed_results.append(result)
                        else:
                            isolated_results.append(result)
                        report["phases"]["mixed_get"] = mixed_results
                        report["phases"]["mixed_isolated"] = isolated_results
                    if len({item["query_digest_sha256"] for item in pair}) != 1:
                        raise AssertionError(
                            f"paired {workload} cache modes changed the query corpus"
                        )
                    _checkpoint(report_path, report, budget)
                for cache_mode in cache_modes:
                    mixed = repetition_results[("mixed", cache_mode)]
                    mixed["fairness"] = matched_fairness_summary(
                        mixed,
                        repetition_results[("isolated_small", cache_mode)],
                        repetition_results[("isolated_large", cache_mode)],
                    )
                report["phases"]["mixed_get"] = mixed_results
                _checkpoint(report_path, report, budget)

            log("running bounded physical-order ScanIds")
            report["phases"]["scan_ids"] = run_scan(
                remote,
                max_rows=args.scan_rows,
                batch_rows=args.scan_batch_rows,
                budget=budget,
                rpc_timeout=args.rpc_timeout,
                server_pid=server_pid,
            )
            _checkpoint(report_path, report, budget)

            if not args.no_embedded_baseline:
                if description.reader is None:
                    raise RuntimeError(
                        "server did not expose reader settings required for the "
                        "embedded baseline"
                    )
                log("opening embedded reader with the server's effective settings")
                assert args.store is not None
                embedded = throwdb.Collection.open(
                    args.store,
                    mode="r",
                    tuning=description.reader.as_dict(),
                )
                budget.rebase_memory("embedded reader and index opened")
                if source is None:
                    report["phases"]["correctness"] = correctness_check(
                        remote,
                        population,
                        rows=args.correctness_rows,
                        seed=args.seed,
                        source=None,
                        embedded=embedded,
                        budget=budget,
                        rpc_timeout=args.rpc_timeout,
                    )
                    report["phases"]["correctness"][
                        "timing"
                    ] = (
                        "deferred until after remote measurements; controlled "
                        "cache preparation still precedes every timed trial"
                    )
                    _checkpoint(report_path, report, budget)
                baseline: list[dict[str, Any]] = []
                remote_pairs: dict[str, list[dict[str, Any]]] = {}
                for remote_result in sequential:
                    remote_pairs.setdefault(
                        str(remote_result["pair_id"]), []
                    ).append(remote_result)
                for pair_id, remote_pair in remote_pairs.items():
                    baseline_pair: list[dict[str, Any]] = []
                    for remote_result in remote_pair:
                        label = str(remote_result["label"])
                        batch_size = int(remote_result["batch_size"])
                        prefix = int(remote_result["prefix_dimensions"])
                        scenario_index = int(remote_result["scenario_index"])
                        repetition = int(remote_result["repetition"])
                        cache_mode = str(remote_result["cache_mode"])
                        log(
                            f"running embedded {label}; repeat={repetition}/"
                            f"{args.repeats}; cache={cache_mode}"
                        )
                        result = run_sequential_get(
                            embedded,
                            population,
                            label=label,
                            batch_size=batch_size,
                            prefix=prefix,
                            dimension=description.schema.dimension,
                            itemsize=itemsize,
                            calls=args.calls,
                            warmup_calls=args.warmup_calls,
                            seed=derived_seed(
                                args.seed,
                                (repetition - 1) * 10_000_000
                                + 10_000 * (scenario_index + 1),
                            ),
                            budget=budget,
                            rpc_timeout=args.rpc_timeout,
                            remote=False,
                            server_pid=None,
                            cache_mode=cache_mode,
                            prepare_cold=(
                                prepare_cold if cache_mode == "cold" else None
                            ),
                        )
                        result.update(
                            {
                                "reader": "embedded",
                                "repetition": repetition,
                                "scenario_index": scenario_index,
                                "scenario_order_position": remote_result[
                                    "scenario_order_position"
                                ],
                                "pair_id": pair_id,
                            }
                        )
                        add_cache_observation(
                            result,
                            record_size=description.record_size,
                            remote=False,
                        )
                        if result["query_digest_sha256"] != remote_result[
                            "query_digest_sha256"
                        ]:
                            raise AssertionError(
                                "embedded baseline did not replay the remote "
                                "query stream"
                        )
                        baseline_pair.append(result)
                        baseline.append(result)
                        report["phases"]["embedded_baseline"] = baseline
                    _checkpoint(report_path, report, budget)
                report["phases"]["remote_vs_embedded"] = compare_baseline(
                    sequential, baseline
                )
                report["phases"]["remote_vs_embedded_note"] = (
                    "prefix Get currently reads full dense rows in the embedded "
                    "engine, so prefix ratios do not isolate physical prefix I/O"
                )
            else:
                baseline = []

            report["phases"]["summary"] = {
                "sequential_get": aggregate_trials(
                    [*sequential, *baseline],
                    group_fields=(
                        "reader",
                        "label",
                        "batch_size",
                        "prefix_dimensions",
                        "cache_mode",
                    ),
                ),
                "concurrent_get": aggregate_trials(
                    concurrency_results,
                    group_fields=("workers", "batch_size", "cache_mode"),
                ),
                "concurrent_classes": aggregate_trials(
                    concurrent_class_trials(concurrency_results),
                    group_fields=(
                        "workers",
                        "batch_size",
                        "cache_mode",
                        "class_label",
                    ),
                ),
                "mixed_get": aggregate_trials(
                    mixed_results,
                    group_fields=("workload", "workers", "cache_mode"),
                ),
                "mixed_classes": aggregate_trials(
                    concurrent_class_trials(mixed_results),
                    group_fields=(
                        "workload",
                        "workers",
                        "cache_mode",
                        "class_label",
                    ),
                ),
                "mixed_fairness": aggregate_fairness_trials(mixed_results),
                "mixed_isolated": aggregate_trials(
                    isolated_results,
                    group_fields=("workload", "workers", "cache_mode"),
                ),
                "mixed_isolated_classes": aggregate_trials(
                    concurrent_class_trials(isolated_results),
                    group_fields=(
                        "workload",
                        "workers",
                        "cache_mode",
                        "class_label",
                    ),
                ),
            }
            if embedded is not None:
                embedded.close()
                embedded = None

            # Mutations deliberately run after every read comparison. Even a
            # logically restored CRUD cycle appends WAL history and must not
            # change the embedded baseline's storage topology/cache behavior.
            if args.crud_rows:
                log("running opt-in CRUD cycle")
            try:
                report["phases"]["crud"] = run_crud(
                    remote,
                    population,
                    rows=args.crud_rows,
                    seed=derived_seed(args.seed, 5_000_000),
                    budget=budget,
                    rpc_timeout=args.rpc_timeout,
                    allow_nondurable=args.allow_nondurable_crud,
                )
            except CrudBenchmarkError as error:
                report["phases"]["crud"] = error.details
                _checkpoint(report_path, report, budget)
                raise
            _checkpoint(report_path, report, budget)
            report["status"] = "complete"
    except BaseException as error:
        caught = error
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if embedded is not None:
            with contextlib.suppress(Exception):
                embedded.close()
        if remote is not None:
            report.setdefault("server", {})["final_describe"] = (
                _capture_final_describe(
                    remote,
                    budget=budget,
                    rpc_timeout=args.rpc_timeout,
                )
            )
            with contextlib.suppress(Exception):
                remote.close()
        if local_server is not None:
            local_server.close()
            returncode = local_server.process.returncode
            report.setdefault("server", {})["returncode_after_shutdown"] = returncode
            if (
                args.server_read_log_dir is not None
                and args.store is not None
            ):
                report["server"]["read_log_inspection"] = (
                    _inspect_read_log_after_shutdown(
                        directory=args.server_read_log_dir,
                        storage=args.store,
                        clean_shutdown=returncode == 0,
                        budget=budget,
                        max_event_drop_ratio=(
                            args.server_read_log_max_event_drop_ratio
                        ),
                        max_row_drop_ratio=(
                            args.server_read_log_max_row_drop_ratio
                        ),
                    )
                )
        report["diskstats"] = diskstats.summary()
        report["finished_unix_ns"] = time.time_ns()
        _checkpoint(report_path, report, budget)
    if caught is not None:
        raise caught
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    log(f"benchmark complete: {args.results / 'report.json'}")
    if report["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
