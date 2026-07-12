#!/usr/bin/env python3
"""One-command correctness and load test for a local ThrowDB store.

Edit the CONFIGURATION section below, then run:

    python benchmarks/run_load_test.py

Command-line flags can override the paths without editing this file. The script
never drops the system-wide Linux page cache and does not require root
privileges; cold mode advises only ThrowDB shard pages as no longer needed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import throwdb


# ---------------------------------------------------------------------------
# CONFIGURATION: these are the only values most users need to edit.
# ---------------------------------------------------------------------------

STORE_PATH = Path("/tmp/throwdb-prod-test-store")
IDS_PATH = Path("/CHANGE/ME/ids.npy")
RESULTS_DIR = Path("./benchmark-results")

# "quick" is a short smoke/load run. "full" produces more stable percentiles
# and reads substantially more data from the NVMe device.
PROFILE = "full"

# "cold" asks Linux to evict clean shard pages before every scenario, "warm"
# leaves the page cache untouched, and "both" runs the same deterministic query
# stream cold first and warm immediately afterwards.
CACHE_MODE = "both"

# Optional bit-exact correctness check against the original embeddings.
# Leave as None to skip it. Set EMBEDDINGS_KIND to "npy" for a .npy file or
# "raw" for a headerless np.memmap file with shape (storage.count, dimension).
SOURCE_EMBEDDINGS_PATH: Path | None = None
EMBEDDINGS_KIND = "raw"
CORRECTNESS_SAMPLE_ROWS = 10_000

# Seed every workload so repeated runs use the same logical query stream.
RANDOM_SEED = 20260711

# Internal ThrowDB I/O executor. Zero workers is the synchronous pread baseline;
# with workers enabled, zero queue capacity selects 2 * IO_WORKERS. Each queued
# task processes up to IO_RANGES_PER_TASK physical ranges before yielding.
IO_WORKERS = 0
IO_QUEUE_CAPACITY = 0
IO_RANGES_PER_TASK = 8
MAX_IO_RANGES_PER_TASK = 1 << 20
IO_BACKEND = "thread_pool"
IO_URING_QUEUE_DEPTH = 32
MAX_IO_URING_QUEUE_DEPTH = 4096


@dataclass(frozen=True)
class SequentialScenario:
    batch_size: int
    calls: int
    warmups: int


QUICK_SCENARIOS = (
    SequentialScenario(1, 1_000, 20),
    SequentialScenario(128, 200, 3),
    SequentialScenario(1_024, 50, 2),
    SequentialScenario(10_000, 10, 1),
)

FULL_SCENARIOS = (
    SequentialScenario(1, 10_000, 100),
    SequentialScenario(16, 5_000, 20),
    SequentialScenario(128, 2_000, 10),
    SequentialScenario(1_024, 500, 3),
    SequentialScenario(10_000, 100, 2),
    SequentialScenario(100_000, 20, 1),
)

QUICK_CONCURRENCY = {
    "batch_size": 1_024,
    "calls_per_worker": 25,
    "worker_counts": (1, 2, 4, 8),
}

FULL_CONCURRENCY = {
    "batch_size": 1_024,
    "calls_per_worker": 200,
    "worker_counts": (1, 2, 4, 8, 16),
}


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def ranges_per_task_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_IO_RANGES_PER_TASK:
        raise argparse.ArgumentTypeError(
            f"value must be in 1..{MAX_IO_RANGES_PER_TASK}"
        )
    return parsed


def io_uring_queue_depth_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_IO_URING_QUEUE_DEPTH:
        raise argparse.ArgumentTypeError(
            f"value must be in 1..{MAX_IO_URING_QUEUE_DEPTH}"
        )
    return parsed


def run_command(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": list(command)}
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "available": True,
        "command": list(command),
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def find_cgroup_v2_directory(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    """Locate the process CPU controller in a cgroup v2 hierarchy."""
    if not proc_cgroup_path.is_file():
        return None
    try:
        relative_path = None
        for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                relative_path = path.lstrip("/")
                break
        if relative_path is None:
            return None
        # In a cgroup namespace /sys/fs/cgroup may already be mounted at the
        # process cgroup. Try the host-style relative path first, then the
        # namespace root used by containers.
        for candidate in (cgroup_root / relative_path, cgroup_root):
            if (candidate / "cpu.stat").is_file():
                return candidate
    except (OSError, ValueError):
        return None
    return None


def read_integer_counters(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, value = line.split(None, 1)
            result[name] = int(value)
    except (OSError, ValueError):
        return {}
    return result


def cgroup_cpu_snapshot(cgroup_path: Path | None = None) -> dict[str, int]:
    if cgroup_path is None:
        cgroup_path = find_cgroup_v2_directory()
    if cgroup_path is None:
        return {}
    return read_integer_counters(cgroup_path / "cpu.stat")


def collect_cgroup_cpu_information(
    cgroup_path: Path | None = None,
) -> dict[str, Any]:
    if cgroup_path is None:
        cgroup_path = find_cgroup_v2_directory()
    if cgroup_path is None:
        return {"available": False}

    result: dict[str, Any] = {
        "available": True,
        "version": 2,
        "path": str(cgroup_path),
        "stat": cgroup_cpu_snapshot(cgroup_path),
    }
    try:
        raw_cpu_max = (cgroup_path / "cpu.max").read_text(encoding="utf-8").strip()
        quota_text, period_text = raw_cpu_max.split()
        period_usec = int(period_text)
        quota_usec = None if quota_text == "max" else int(quota_text)
        result["cpu_max"] = raw_cpu_max
        result["quota_usec"] = quota_usec
        result["period_usec"] = period_usec
        result["effective_cpu_limit"] = (
            quota_usec / period_usec if quota_usec is not None else None
        )
    except (OSError, ValueError, ZeroDivisionError):
        result["cpu_max"] = None
    return result


def collect_system_information(store_path: Path) -> dict[str, Any]:
    filesystem = os.statvfs(store_path)
    profiling_tools = (
        "vmstat",
        "pidstat",
        "dstat",
        "atop",
        "iotop",
        "perf",
        "strace",
        "blktrace",
        "bpftrace",
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "throwdb": throwdb.__version__,
        "cpu_count": os.cpu_count(),
        "cgroup_cpu": collect_cgroup_cpu_information(),
        "profiling_tools": {
            tool: shutil.which(tool) for tool in profiling_tools
        },
        "filesystem": {
            "block_size": filesystem.f_frsize,
            "total_bytes": filesystem.f_blocks * filesystem.f_frsize,
            "available_bytes": filesystem.f_bavail * filesystem.f_frsize,
        },
        "commands": {
            "uname": run_command(("uname", "-a")),
            "findmnt": run_command(("findmnt", "-T", str(store_path))),
            "df": run_command(("df", "-hT", str(store_path))),
            "lsblk": run_command(
                ("lsblk", "-o", "NAME,MODEL,SIZE,ROTA,FSTYPE,MOUNTPOINTS")
            ),
            "free": run_command(("free", "-h")),
            "lscpu": run_command(("lscpu",)),
        },
    }


def directory_size(path: Path) -> int:
    total = 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            total += (Path(root) / filename).stat().st_size
    return total


def atomic_json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


class IostatMonitor:
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._process: subprocess.Popen[str] | None = None
        self._output = None

    def __enter__(self) -> "IostatMonitor":
        executable = shutil.which("iostat")
        if executable is None:
            log("iostat is unavailable; install the 'sysstat' package for device metrics")
            return self
        self._output = self._output_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [executable, "-xmd", "1"],
            text=True,
            stdout=self._output,
            stderr=subprocess.STDOUT,
        )
        log(f"iostat monitor started: {self._output_path}")
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._output is not None:
            self._output.close()


class CommandMonitor:
    def __init__(self, command: Sequence[str], output_path: Path) -> None:
        self._command = tuple(command)
        self._output_path = output_path
        self._process: subprocess.Popen[str] | None = None
        self._output = None

    def __enter__(self) -> "CommandMonitor":
        executable = shutil.which(self._command[0])
        if executable is None:
            return self
        self._output = self._output_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [executable, *self._command[1:]],
            text=True,
            stdout=self._output,
            stderr=subprocess.STDOUT,
        )
        log(f"{' '.join(self._command)} monitor started: {self._output_path}")
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._output is not None:
            self._output.close()


def read_diskstats() -> dict[str, dict[str, int]]:
    """Read the stable base fields from Linux /proc/diskstats."""
    path = Path("/proc/diskstats")
    if not path.is_file():
        return {}
    result: dict[str, dict[str, int]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 14:
                continue
            result[fields[2]] = {
                "reads_completed": int(fields[3]),
                "reads_merged": int(fields[4]),
                "sectors_read": int(fields[5]),
                "read_time_ms": int(fields[6]),
                "writes_completed": int(fields[7]),
                "writes_merged": int(fields[8]),
                "sectors_written": int(fields[9]),
                "write_time_ms": int(fields[10]),
                "ios_in_progress": int(fields[11]),
                "io_time_ms": int(fields[12]),
                "weighted_io_time_ms": int(fields[13]),
            }
    except (OSError, ValueError):
        return {}
    return result


class ProcDiskstatsMonitor:
    """Sample aggregate block-device counters without root privileges."""

    def __init__(self, output_path: Path, interval_seconds: float = 1.0) -> None:
        self._output_path = output_path
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first: dict[str, dict[str, int]] = {}
        self._last: dict[str, dict[str, int]] = {}
        self._started = 0.0
        self._finished = 0.0

    def __enter__(self) -> "ProcDiskstatsMonitor":
        self._first = read_diskstats()
        if not self._first:
            log("/proc/diskstats is unavailable")
            return self
        self._last = self._first
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._sample,
            name="throwdb-diskstats",
            daemon=True,
        )
        self._thread.start()
        log(f"/proc/diskstats monitor started: {self._output_path}")
        return self

    def _sample(self) -> None:
        with self._output_path.open("w", encoding="utf-8") as output:
            while not self._stop.is_set():
                snapshot = read_diskstats()
                if snapshot:
                    self._last = snapshot
                    output.write(
                        json.dumps(
                            {
                                "monotonic_seconds": time.monotonic(),
                                "devices": snapshot,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    output.flush()
                self._stop.wait(self._interval_seconds)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        final = read_diskstats()
        if final:
            self._last = final
        self._finished = time.monotonic()

    def summary(self) -> dict[str, Any]:
        elapsed = max(0.0, self._finished - self._started)
        devices: dict[str, dict[str, int | float]] = {}
        for device, after in self._last.items():
            before = self._first.get(device)
            if before is None:
                continue
            delta = counter_delta(before, after)
            if not any(
                int(delta.get(field, 0))
                for field in (
                    "reads_completed",
                    "writes_completed",
                    "sectors_read",
                    "sectors_written",
                )
            ):
                continue
            sectors_read = int(delta.get("sectors_read", 0))
            sectors_written = int(delta.get("sectors_written", 0))
            devices[device] = {
                **delta,
                # Linux diskstats sectors are expressed in standard 512-byte units.
                "read_bytes": sectors_read * 512,
                "written_bytes": sectors_written * 512,
                "reads_per_second": (
                    int(delta.get("reads_completed", 0)) / elapsed
                    if elapsed
                    else 0.0
                ),
                "read_mib_per_second": (
                    sectors_read * 512 / elapsed / (1 << 20)
                    if elapsed
                    else 0.0
                ),
            }
        return {"elapsed_seconds": elapsed, "active_devices": devices}


def load_ids(path: Path, expected_count: int) -> np.ndarray:
    ids = np.load(path, mmap_mode="r", allow_pickle=False)
    if ids.dtype != np.dtype(np.uint64):
        raise ValueError(f"IDs must have dtype uint64, got {ids.dtype}")
    if ids.ndim != 1:
        raise ValueError(f"IDs must have shape (N,), got {ids.shape}")
    if not ids.flags.c_contiguous:
        raise ValueError("IDs must be C-contiguous")
    if len(ids) != expected_count:
        raise ValueError(
            f"ID count {len(ids):,} does not match storage count {expected_count:,}"
        )
    if len(ids) == 0:
        raise ValueError("The benchmark requires a non-empty store")
    return ids


def open_source_embeddings(
    path: Path,
    kind: str,
    count: int,
    dimension: int,
) -> np.ndarray:
    if kind == "npy":
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    elif kind == "raw":
        embeddings = np.memmap(
            path,
            mode="r",
            dtype=np.float16,
            shape=(count, dimension),
            order="C",
        )
    else:
        raise ValueError("EMBEDDINGS_KIND must be 'npy' or 'raw'")
    if embeddings.dtype != np.dtype(np.float16):
        raise ValueError(
            f"source embeddings must have dtype float16, got {embeddings.dtype}"
        )
    if embeddings.shape != (count, dimension):
        raise ValueError(
            "source embeddings shape does not match storage: "
            f"{embeddings.shape} != {(count, dimension)}"
        )
    return embeddings


def correctness_check(
    storage: Any,
    population: np.ndarray,
    source_path: Path | None,
    source_kind: str,
) -> dict[str, Any]:
    if source_path is None:
        return {
            "status": "skipped",
            "reason": "SOURCE_EMBEDDINGS_PATH is not configured",
        }
    source = open_source_embeddings(
        source_path,
        source_kind,
        storage.count,
        storage.dimension,
    )
    rows = min(CORRECTNESS_SAMPLE_ROWS, storage.count)
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.integers(0, storage.count, size=rows, dtype=np.int64)
    query_ids = np.asarray(population[positions], dtype=np.uint64, order="C")
    expected = np.asarray(source[positions], dtype=np.float16, order="C")
    actual = storage.get(query_ids)
    np.testing.assert_array_equal(
        actual.view(np.uint16),
        expected.view(np.uint16),
    )
    return {"status": "passed", "sample_rows": rows, "bit_exact": True}


def query_matrix(
    population: np.ndarray,
    rows: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = rng.integers(
        0,
        len(population),
        size=(rows, batch_size),
        dtype=np.int64,
    )
    return np.asarray(population[positions], dtype=np.uint64, order="C")


def latency_metrics(samples_ns: np.ndarray) -> dict[str, float]:
    return {
        "latency_p50_ms": float(np.percentile(samples_ns, 50)) / 1e6,
        "latency_p95_ms": float(np.percentile(samples_ns, 95)) / 1e6,
        "latency_p99_ms": float(np.percentile(samples_ns, 99)) / 1e6,
        "latency_max_ms": float(samples_ns.max()) / 1e6,
    }


def usage_snapshot() -> dict[str, int | float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_raw": usage.ru_maxrss,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "block_input_operations": usage.ru_inblock,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def proc_io_snapshot() -> dict[str, int]:
    """Read Linux per-process I/O counters without requiring privileges."""
    path = Path("/proc/self/io")
    if not path.is_file():
        return {}
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            result[name] = int(value.strip())
    except (OSError, ValueError):
        return {}
    return result


def counter_delta(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> dict[str, int | float]:
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if key != "max_rss_raw"
    }


def add_process_metrics(
    result: dict[str, Any],
    usage_before: dict[str, int | float],
    usage_after: dict[str, int | float],
    io_before: dict[str, int],
    io_after: dict[str, int],
    cgroup_cpu_before: dict[str, int],
    cgroup_cpu_after: dict[str, int],
    useful_bytes: int,
    seconds: float,
) -> None:
    result["process_usage_delta"] = counter_delta(usage_before, usage_after)
    result["process_max_rss_raw"] = usage_after["max_rss_raw"]
    if io_before and io_after:
        io_delta = counter_delta(io_before, io_after)
        result["process_io_delta"] = io_delta
        storage_bytes = int(io_delta.get("read_bytes", 0))
        result["storage_read_gib_per_second"] = (
            storage_bytes / seconds / (1 << 30)
        )
        # This is deliberately called a backing-read ratio rather than read
        # amplification: with page-cache hits it may legitimately be below 1.
        result["backing_read_ratio"] = (
            storage_bytes / useful_bytes if useful_bytes else 0.0
        )
    if cgroup_cpu_before and cgroup_cpu_after:
        cgroup_delta: dict[str, int | float] = {
            key: cgroup_cpu_after[key] - cgroup_cpu_before[key]
            for key in cgroup_cpu_after.keys() & cgroup_cpu_before.keys()
        }
        throttled_usec = int(cgroup_delta.get("throttled_usec", 0))
        nr_periods = int(cgroup_delta.get("nr_periods", 0))
        nr_throttled = int(cgroup_delta.get("nr_throttled", 0))
        cgroup_delta["throttled_seconds"] = throttled_usec / 1_000_000
        cgroup_delta["throttled_period_fraction"] = (
            nr_throttled / nr_periods if nr_periods else 0.0
        )
        cgroup_delta["throttled_seconds_per_wall_second"] = (
            throttled_usec / 1_000_000 / seconds if seconds else 0.0
        )
        result["cgroup_cpu_delta"] = cgroup_delta


def evict_data_cache(store_path: Path) -> dict[str, int | float]:
    """Ask Linux to evict clean pages for every immutable data shard."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError(
            "cold cache mode requires os.posix_fadvise and POSIX_FADV_DONTNEED"
        )
    shard_paths = sorted((store_path / "data").glob("shard-*.dat"))
    if not shard_paths:
        raise RuntimeError(f"no data shards found under {store_path / 'data'}")
    started = time.perf_counter()
    total_bytes = 0
    for shard_path in shard_paths:
        total_bytes += shard_path.stat().st_size
        fd = os.open(shard_path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            raise RuntimeError(
                f"POSIX_FADV_DONTNEED failed for {shard_path}: {error}"
            ) from error
        finally:
            os.close(fd)
    return {
        "shard_count": len(shard_paths),
        "advised_bytes": total_bytes,
        "seconds": time.perf_counter() - started,
    }


def sequential_benchmark(
    storage: Any,
    population: np.ndarray,
    scenario: SequentialScenario,
    seed: int,
) -> dict[str, Any]:
    queries = query_matrix(
        population,
        scenario.calls + scenario.warmups,
        scenario.batch_size,
        seed,
    )
    output = np.empty(
        (scenario.batch_size, storage.dimension),
        dtype=np.float16,
    )
    for query in queries[: scenario.warmups]:
        storage.get_into(query, output)

    samples_ns = np.empty(scenario.calls, dtype=np.int64)
    usage_before = usage_snapshot()
    io_before = proc_io_snapshot()
    cgroup_cpu_before = cgroup_cpu_snapshot()
    wall_started = time.perf_counter()
    for index, query in enumerate(queries[scenario.warmups :]):
        started = time.perf_counter_ns()
        storage.get_into(query, output)
        samples_ns[index] = time.perf_counter_ns() - started
    seconds = time.perf_counter() - wall_started
    usage_after = usage_snapshot()
    io_after = proc_io_snapshot()
    cgroup_cpu_after = cgroup_cpu_snapshot()

    embeddings = scenario.calls * scenario.batch_size
    useful_bytes = embeddings * storage.dimension * 2
    result: dict[str, Any] = {
        **asdict(scenario),
        "workers": 1,
        "embeddings": embeddings,
        "seconds": seconds,
        "embeddings_per_second": embeddings / seconds,
        "useful_gib_per_second": useful_bytes / seconds / (1 << 30),
    }
    result.update(latency_metrics(samples_ns))
    add_process_metrics(
        result,
        usage_before,
        usage_after,
        io_before,
        io_after,
        cgroup_cpu_before,
        cgroup_cpu_after,
        useful_bytes,
        seconds,
    )
    return result


def concurrent_benchmark(
    storage: Any,
    population: np.ndarray,
    workers: int,
    batch_size: int,
    calls_per_worker: int,
    seed: int,
) -> dict[str, Any]:
    prepared = [
        query_matrix(population, calls_per_worker + 1, batch_size, seed + worker)
        for worker in range(workers)
    ]
    ready = threading.Barrier(workers + 1)
    start = threading.Event()

    def run_worker(queries: np.ndarray) -> np.ndarray:
        output = np.empty((batch_size, storage.dimension), dtype=np.float16)
        storage.get_into(queries[0], output)
        samples = np.empty(calls_per_worker, dtype=np.int64)
        ready.wait()
        start.wait()
        for index, query in enumerate(queries[1:]):
            started = time.perf_counter_ns()
            storage.get_into(query, output)
            samples[index] = time.perf_counter_ns() - started
        return samples

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_worker, queries) for queries in prepared]
        ready.wait()
        usage_before = usage_snapshot()
        io_before = proc_io_snapshot()
        cgroup_cpu_before = cgroup_cpu_snapshot()
        wall_started = time.perf_counter()
        start.set()
        samples_ns = np.concatenate([future.result() for future in futures])
        seconds = time.perf_counter() - wall_started
        usage_after = usage_snapshot()
        io_after = proc_io_snapshot()
        cgroup_cpu_after = cgroup_cpu_snapshot()

    embeddings = workers * calls_per_worker * batch_size
    useful_bytes = embeddings * storage.dimension * 2
    result: dict[str, Any] = {
        "workers": workers,
        "batch_size": batch_size,
        "calls_per_worker": calls_per_worker,
        "embeddings": embeddings,
        "seconds": seconds,
        "embeddings_per_second": embeddings / seconds,
        "useful_gib_per_second": useful_bytes / seconds / (1 << 30),
    }
    result.update(latency_metrics(samples_ns))
    add_process_metrics(
        result,
        usage_before,
        usage_after,
        io_before,
        io_after,
        cgroup_cpu_before,
        cgroup_cpu_after,
        useful_bytes,
        seconds,
    )
    return result


def print_result(result: dict[str, Any]) -> None:
    cgroup_delta = result.get("cgroup_cpu_delta", {})
    throttle = ""
    if cgroup_delta:
        throttle = (
            f" throttle={cgroup_delta.get('nr_throttled', 0)}/"
            f"{cgroup_delta.get('nr_periods', 0)} periods"
            f"/{cgroup_delta.get('throttled_seconds', 0.0):.3f}s"
        )
    log(
        f"cache={result.get('cache_mode', 'n/a'):<4} "
        f"clients={result['workers']:>2} "
        f"backend={result.get('io_backend', 'thread_pool'):<11} "
        f"io_workers={result.get('io_workers', 0):>2} "
        f"ranges/task={result.get('io_ranges_per_task', 1):>2} "
        f"batch={result['batch_size']:>7,} "
        f"throughput={result['embeddings_per_second']:>12,.0f} emb/s "
        f"useful={result['useful_gib_per_second']:>7.3f} GiB/s "
        f"p50={result['latency_p50_ms']:>9.3f} ms "
        f"p99={result['latency_p99_ms']:>9.3f} ms"
        f"{throttle}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--ids", type=Path, default=IDS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--profile", choices=("quick", "full"), default=PROFILE)
    parser.add_argument(
        "--io-workers",
        type=non_negative_int,
        default=IO_WORKERS,
        help="internal C++ I/O workers; zero selects the sequential pread baseline",
    )
    parser.add_argument(
        "--io-queue-capacity",
        type=non_negative_int,
        default=IO_QUEUE_CAPACITY,
        help="waiting I/O tasks; zero auto-selects 2 * workers when enabled",
    )
    parser.add_argument(
        "--io-ranges-per-task",
        type=ranges_per_task_int,
        default=IO_RANGES_PER_TASK,
        help=(
            "maximum physical read ranges processed by one executor task "
            f"(1..{MAX_IO_RANGES_PER_TASK}; default: 8)"
        ),
    )
    parser.add_argument(
        "--io-backend",
        choices=("thread_pool", "io_uring"),
        default=IO_BACKEND,
        help="reader backend (default: thread_pool)",
    )
    parser.add_argument(
        "--io-uring-queue-depth",
        type=io_uring_queue_depth_int,
        default=IO_URING_QUEUE_DEPTH,
        help=(
            "maximum in-flight io_uring reads "
            f"(1..{MAX_IO_URING_QUEUE_DEPTH}; default: 32)"
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm", "both"),
        default=CACHE_MODE,
        help="evict shard pages, preserve cache, or run paired cold then warm tests",
    )
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        default=SOURCE_EMBEDDINGS_PATH,
        help="optional original float16 .npy or raw memmap file",
    )
    parser.add_argument(
        "--source-kind",
        choices=("npy", "raw"),
        default=EMBEDDINGS_KIND,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.io_ranges_per_task <= MAX_IO_RANGES_PER_TASK:
        raise ValueError(
            "--io-ranges-per-task must be in "
            f"1..{MAX_IO_RANGES_PER_TASK}"
        )
    if not 1 <= args.io_uring_queue_depth <= MAX_IO_URING_QUEUE_DEPTH:
        raise ValueError(
            "--io-uring-queue-depth must be in "
            f"1..{MAX_IO_URING_QUEUE_DEPTH}"
        )
    if args.io_backend == "io_uring" and (
        args.io_workers != 0 or args.io_queue_capacity != 0
    ):
        raise ValueError(
            "--io-workers and --io-queue-capacity must both be zero when "
            "--io-backend is io_uring"
        )
    if args.io_workers == 0 and args.io_queue_capacity != 0:
        raise ValueError(
            "--io-queue-capacity must be zero when --io-workers is zero"
        )
    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / "report.json"

    if not args.store.is_dir():
        raise FileNotFoundError(f"storage directory does not exist: {args.store}")
    if not args.ids.is_file():
        raise FileNotFoundError(
            f"ID file does not exist: {args.ids}. Edit IDS_PATH at the top "
            "of the script or pass --ids."
        )

    report: dict[str, Any] = {
        "configuration": {
            "store": str(args.store.resolve()),
            "ids": str(args.ids.resolve()),
            "results": str(args.results.resolve()),
            "profile": args.profile,
            "cache_mode": args.cache_mode,
            "source_embeddings": (
                str(args.source_embeddings.resolve())
                if args.source_embeddings is not None
                else None
            ),
            "source_kind": args.source_kind,
            "seed": RANDOM_SEED,
            "io_workers": args.io_workers,
            "io_queue_capacity": args.io_queue_capacity,
            "io_ranges_per_task": args.io_ranges_per_task,
            "io_backend": args.io_backend,
            "io_uring_queue_depth": args.io_uring_queue_depth,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": collect_system_information(args.store),
    }
    atomic_json_dump(report_path, report)

    cgroup_cpu = report["system"]["cgroup_cpu"]
    if cgroup_cpu.get("available"):
        effective_limit = cgroup_cpu.get("effective_cpu_limit")
        limit_text = (
            "unlimited"
            if effective_limit is None
            else f"{effective_limit:.2f} CPUs"
        )
        log(
            f"cgroup v2 CPU limit: {limit_text} "
            f"(cpu.max={cgroup_cpu.get('cpu_max', 'unknown')})"
        )

    log(f"opening storage: {args.store}")
    open_started = time.perf_counter()
    storage = throwdb.Storage.open(
        args.store,
        io_workers=args.io_workers,
        io_queue_capacity=args.io_queue_capacity,
        io_ranges_per_task=args.io_ranges_per_task,
        io_backend=args.io_backend,
        io_uring_queue_depth=args.io_uring_queue_depth,
    )
    open_seconds = time.perf_counter() - open_started
    report["storage"] = {
        "count": storage.count,
        "dimension": storage.dimension,
        "dtype": "float16",
        "directory_bytes": directory_size(args.store),
        "open_seconds": open_seconds,
        "io_workers": args.io_workers,
        "io_queue_capacity": args.io_queue_capacity,
        "io_ranges_per_task": args.io_ranges_per_task,
        "io_backend": args.io_backend,
        "io_uring_queue_depth": args.io_uring_queue_depth,
    }
    log(
        f"storage opened in {open_seconds:.3f}s: "
        f"count={storage.count:,}, dimension={storage.dimension}"
    )

    population = load_ids(args.ids, storage.count)
    log(f"ID population validated: {len(population):,} IDs")

    log("running bit-exact correctness check")
    report["correctness"] = correctness_check(
        storage,
        population,
        args.source_embeddings,
        args.source_kind,
    )
    log(f"correctness: {report['correctness']['status']}")
    atomic_json_dump(report_path, report)

    scenarios = QUICK_SCENARIOS if args.profile == "quick" else FULL_SCENARIOS
    concurrency = (
        QUICK_CONCURRENCY if args.profile == "quick" else FULL_CONCURRENCY
    )
    report["sequential"] = []
    report["concurrent"] = []
    cache_modes = (
        ("cold", "warm") if args.cache_mode == "both" else (args.cache_mode,)
    )

    diskstats_monitor = ProcDiskstatsMonitor(args.results / "diskstats.jsonl")
    with (
        IostatMonitor(args.results / "iostat.log"),
        CommandMonitor(("vmstat", "1"), args.results / "vmstat.log"),
        diskstats_monitor,
    ):
        log("starting sequential benchmarks")
        for index, scenario in enumerate(scenarios):
            for cache_mode in cache_modes:
                eviction = None
                if cache_mode == "cold":
                    log(
                        f"evicting shard pages before sequential "
                        f"batch={scenario.batch_size:,}"
                    )
                    eviction = evict_data_cache(args.store)
                result = sequential_benchmark(
                    storage,
                    population,
                    scenario,
                    RANDOM_SEED + index * 1000,
                )
                result["cache_mode"] = cache_mode
                result["external_reader_workers"] = 1
                result["io_workers"] = args.io_workers
                result["io_queue_capacity"] = args.io_queue_capacity
                result["io_ranges_per_task"] = args.io_ranges_per_task
                result["io_backend"] = args.io_backend
                result["io_uring_queue_depth"] = args.io_uring_queue_depth
                if eviction is not None:
                    result["cache_eviction"] = eviction
                report["sequential"].append(result)
                print_result(result)
                atomic_json_dump(report_path, report)

        log("starting concurrent-reader benchmarks")
        for workers in concurrency["worker_counts"]:
            for cache_mode in cache_modes:
                eviction = None
                if cache_mode == "cold":
                    log(
                        f"evicting shard pages before concurrent "
                        f"workers={workers}"
                    )
                    eviction = evict_data_cache(args.store)
                result = concurrent_benchmark(
                    storage,
                    population,
                    workers,
                    concurrency["batch_size"],
                    concurrency["calls_per_worker"],
                    RANDOM_SEED + workers * 1000,
                )
                result["cache_mode"] = cache_mode
                result["external_reader_workers"] = workers
                result["io_workers"] = args.io_workers
                result["io_queue_capacity"] = args.io_queue_capacity
                result["io_ranges_per_task"] = args.io_ranges_per_task
                result["io_backend"] = args.io_backend
                result["io_uring_queue_depth"] = args.io_uring_queue_depth
                if eviction is not None:
                    result["cache_eviction"] = eviction
                report["concurrent"].append(result)
                print_result(result)
                atomic_json_dump(report_path, report)

    report["diskstats"] = diskstats_monitor.summary()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json_dump(report_path, report)
    log(f"complete report: {report_path.resolve()}")
    log(f"device statistics: {(args.results / 'iostat.log').resolve()}")
    log(f"proc disk statistics: {(args.results / 'diskstats.jsonl').resolve()}")
    log(f"virtual memory statistics: {(args.results / 'vmstat.log').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
