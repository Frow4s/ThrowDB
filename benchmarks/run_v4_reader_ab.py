#!/usr/bin/env python3
"""Run the fixed v4 reader A/B acceptance matrix.

This benchmark compares the conservative synchronous reader with the two
device-saturating profiles that performed best during the first NVMe sweep.
Every policy sees the same deterministic query stream inside one scenario
block. Policy order is rotated between repetitions to reduce time-order bias,
and each policy handle is closed before the next one is opened.

The default matrix is intentionally narrow and directly comparable:

* thread_pool, 0 internal workers (synchronous pread control);
* thread_pool, 16 workers, queue capacity 32, 8 ranges per task;
* io_uring, 2 ring workers, aggregate queue depth 32;
* batches 1,024 and 10,000;
* 1, 8, and 16 external readers;
* 3 repetitions, each measured cold and then warm.

Run from an activated ThrowDB environment, for example:

    python benchmarks/run_v4_reader_ab.py \
      --store /tmp/throwdb-prod-test-store \
      --ids /datasets/ids.npy \
      --results benchmark-results/v4-reader-ab

Before measurement, all policies must open successfully and return a bit-exact
match to the synchronous baseline. The zero-delta read path is required unless
--no-require-zero-delta is explicit. The script never drops the global page
cache. Cold runs advise ThrowDB base segments and mutation WAL payloads as no
longer needed. Warm-only mode explicitly replays the exact measured queries
before starting process and latency counters.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
if str(_BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIRECTORY))

import run_load_test as load_test  # noqa: E402


MINIMUM_FORMAT_VERSION = 4
DEFAULT_REPEATS = 3
DEFAULT_BATCH_CALLS = {1_024: 100, 10_000: 10}
DEFAULT_EXTERNAL_READERS = (1, 8, 16)
BASELINE_POLICY_NAME = "thread_pool_w0"


@dataclass(frozen=True)
class ReaderPolicy:
    name: str
    io_backend: str
    io_workers: int
    io_queue_capacity: int
    io_ranges_per_task: int
    io_uring_workers: int
    io_uring_queue_depth: int

    def tuning(self) -> dict[str, int | str]:
        return {
            "io_backend": self.io_backend,
            "io_workers": self.io_workers,
            "io_queue_capacity": self.io_queue_capacity,
            "io_ranges_per_task": self.io_ranges_per_task,
            "io_uring_workers": self.io_uring_workers,
            "io_uring_queue_depth": self.io_uring_queue_depth,
        }


READER_POLICIES = (
    ReaderPolicy(
        name=BASELINE_POLICY_NAME,
        io_backend="thread_pool",
        io_workers=0,
        io_queue_capacity=0,
        io_ranges_per_task=8,
        io_uring_workers=1,
        io_uring_queue_depth=32,
    ),
    ReaderPolicy(
        name="thread_pool_w16_r8",
        io_backend="thread_pool",
        io_workers=16,
        io_queue_capacity=32,
        io_ranges_per_task=8,
        io_uring_workers=1,
        io_uring_queue_depth=32,
    ),
    ReaderPolicy(
        name="io_uring_w2_qd32",
        io_backend="io_uring",
        io_workers=0,
        io_queue_capacity=0,
        io_ranges_per_task=8,
        io_uring_workers=2,
        io_uring_queue_depth=32,
    ),
)


@dataclass(frozen=True)
class ReaderScenario:
    batch_size: int
    external_reader_workers: int
    calls_per_worker: int


@dataclass(frozen=True)
class MeasurementBlock:
    repetition: int
    scenario_index: int
    scenario: ReaderScenario
    policy_order_position: int
    policy: ReaderPolicy


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_scenarios(
    calls_1024: int,
    calls_10000: int,
    external_readers: Iterable[int] = DEFAULT_EXTERNAL_READERS,
) -> tuple[ReaderScenario, ...]:
    calls_by_batch = {1_024: calls_1024, 10_000: calls_10000}
    return tuple(
        ReaderScenario(
            batch_size=batch_size,
            external_reader_workers=workers,
            calls_per_worker=calls_by_batch[batch_size],
        )
        for batch_size in (1_024, 10_000)
        for workers in external_readers
    )


def rotated_policies(repetition: int) -> tuple[ReaderPolicy, ...]:
    """Return a Latin-rotation order for a zero-based repetition."""

    offset = repetition % len(READER_POLICIES)
    return READER_POLICIES[offset:] + READER_POLICIES[:offset]


def scenario_seed(
    base_seed: int,
    repetition: int,
    scenario: ReaderScenario,
) -> int:
    """Keep streams paired across policies and distinct across scenarios."""

    batch_slot = 0 if scenario.batch_size == 1_024 else 1
    return (
        base_seed
        + repetition * 1_000_000
        + batch_slot * 100_000
        + scenario.external_reader_workers * 1_000
    )


def cache_modes(mode: str) -> tuple[str, ...]:
    return ("cold", "warm") if mode == "both" else (mode,)


def build_measurement_blocks(
    repeats: int,
    scenarios: Sequence[ReaderScenario],
) -> tuple[MeasurementBlock, ...]:
    """Interleave policies inside each paired scenario block."""

    return tuple(
        MeasurementBlock(
            repetition=repetition,
            scenario_index=scenario_index,
            scenario=scenario,
            policy_order_position=policy_position,
            policy=policy,
        )
        for repetition in range(repeats)
        for scenario_index, scenario in enumerate(scenarios)
        for policy_position, policy in enumerate(rotated_policies(repetition))
    )


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "mean": mean,
        "min": min(samples),
        "max": max(samples),
        "coefficient_of_variation": (
            statistics.pstdev(samples) / mean if mean else 0.0
        ),
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repetitions and add paired speedups over synchronous pread."""

    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    by_run: dict[tuple[int, str, int, int, str], dict[str, Any]] = {}
    for result in results:
        policy = str(result["policy"])
        cache_mode = str(result["cache_mode"])
        batch_size = int(result["batch_size"])
        workers = int(result["external_reader_workers"])
        repetition = int(result["repetition"])
        groups.setdefault(
            (policy, cache_mode, batch_size, workers), []
        ).append(result)
        by_run[(repetition, cache_mode, batch_size, workers, policy)] = result

    metrics = (
        "embeddings_per_second",
        "useful_gib_per_second",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_max_ms",
        "backing_read_ratio",
        "storage_read_gib_per_second",
    )
    summary: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: (value[1], value[2], value[3], value[0])):
        policy, mode, batch_size, workers = key
        samples = sorted(groups[key], key=lambda value: int(value["repetition"]))
        row: dict[str, Any] = {
            "policy": policy,
            "cache_mode": mode,
            "batch_size": batch_size,
            "external_reader_workers": workers,
            "repetitions": [int(value["repetition"]) for value in samples],
            "sample_count": len(samples),
        }
        for metric in metrics:
            available = [float(value[metric]) for value in samples if metric in value]
            if available:
                row[metric] = _distribution(available)

        speedups = []
        for sample in samples:
            baseline = by_run.get(
                (
                    int(sample["repetition"]),
                    mode,
                    batch_size,
                    workers,
                    BASELINE_POLICY_NAME,
                )
            )
            if baseline is not None:
                baseline_throughput = float(baseline["embeddings_per_second"])
                speedups.append(
                    float(sample["embeddings_per_second"]) / baseline_throughput
                )
        if speedups:
            row["paired_speedup_vs_thread_pool_w0"] = _distribution(speedups)
        summary.append(row)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm", "both"),
        default="both",
        help="default: paired cold then warm for every scenario",
    )
    parser.add_argument("--repeats", type=positive_int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--calls-1024",
        type=positive_int,
        default=DEFAULT_BATCH_CALLS[1_024],
        help="calls per external reader for batch 1,024 (default: 100)",
    )
    parser.add_argument(
        "--calls-10000",
        type=positive_int,
        default=DEFAULT_BATCH_CALLS[10_000],
        help="calls per external reader for batch 10,000 (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=load_test.RANDOM_SEED)
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        default=None,
        help="optional source used for one bit-exact preflight sample",
    )
    parser.add_argument(
        "--source-kind",
        choices=("npy", "raw"),
        default="npy",
    )
    parser.add_argument(
        "--no-require-zero-delta",
        dest="require_zero_delta",
        action="store_false",
        help=(
            "allow a collection with recovered/current mutations; by default "
            "the regression gate requires the zero-delta base-only read path"
        ),
    )
    parser.set_defaults(require_zero_delta=True)
    return parser


def _annotate_result(
    result: dict[str, Any],
    policy: ReaderPolicy,
    repetition: int,
    cache_mode: str,
    seed: int,
    eviction: dict[str, int | float] | None,
    scenario_index: int,
    policy_order_position: int,
) -> None:
    result.update(policy.tuning())
    result["policy"] = policy.name
    result["repetition"] = repetition + 1
    result["cache_mode"] = cache_mode
    result["external_reader_workers"] = result["workers"]
    result["seed"] = seed
    result["scenario_index"] = scenario_index
    result["policy_order_position"] = policy_order_position
    if policy.io_backend == "io_uring":
        result["io_uring_admission_capacity"] = 2 * policy.io_uring_workers
        result["io_uring_admission_policy"] = "fifo_ticket"
    else:
        result["io_uring_admission_capacity"] = None
        result["io_uring_admission_policy"] = None
    if eviction is not None:
        result["cache_eviction"] = eviction


def _open_policy(
    store: Path,
    policy: ReaderPolicy,
) -> tuple[Any, str, dict[str, Any]]:
    threads_before = load_test.process_thread_count()
    started = time.perf_counter()
    storage, store_format = load_test.open_benchmark_storage(
        store,
        **policy.tuning(),
    )
    open_seconds = time.perf_counter() - started
    metadata: dict[str, Any] = {
        "policy": policy.name,
        "open_seconds": open_seconds,
        "store_format": store_format,
        "format_version": storage.format_version,
        "process_thread_count_before_open": threads_before,
        "process_thread_count_after_open": load_test.process_thread_count(),
    }
    open_stats = load_test.storage_open_stats(storage)
    if open_stats is not None:
        metadata["open_stats"] = open_stats
    return storage, store_format, metadata


def _error_details(error: Exception) -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _require_native_v4(storage: Any, store_format: str) -> None:
    if store_format != load_test.NATIVE_STORE_FORMAT:
        raise ValueError(
            "v4 reader A/B requires a native mutable collection, "
            f"got {store_format}"
        )
    if int(storage.format_version) < MINIMUM_FORMAT_VERSION:
        raise ValueError(
            "v4 reader A/B requires format version >= "
            f"{MINIMUM_FORMAT_VERSION}, got {storage.format_version}"
        )


def _profiled_sample(
    storage: Any,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty(
        (len(query_ids), storage.dimension),
        dtype=np.dtype(storage.dtype),
    )
    stats = storage.get_into_profiled(query_ids, output)
    if not isinstance(stats, dict):
        raise RuntimeError("get_into_profiled returned non-dict telemetry")
    return output, dict(stats)


def _require_bit_exact(
    expected: np.ndarray,
    actual: np.ndarray,
    policy_name: str,
) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"policy {policy_name} returned {actual.dtype}{actual.shape}, "
            f"expected {expected.dtype}{expected.shape}"
        )
    if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
        raise AssertionError(
            f"policy {policy_name} is not bit-exact with {BASELINE_POLICY_NAME}"
        )


def preflight_policies(
    *,
    store: Path,
    ids_path: Path,
    source_embeddings: Path | None,
    source_kind: str,
    seed: int,
    require_zero_delta: bool,
    report: dict[str, Any],
    report_path: Path,
    policies: Sequence[ReaderPolicy] = READER_POLICIES,
) -> np.ndarray:
    """Open and verify every policy before starting the expensive matrix."""

    if not policies or policies[0].name != BASELINE_POLICY_NAME:
        raise ValueError("preflight policies must start with the baseline")
    entries: list[dict[str, Any]] = [
        {
            "policy": policy.name,
            "tuning": policy.tuning(),
            "status": "pending",
        }
        for policy in policies
    ]
    report["preflight_status"] = "running"
    report["policy_preflight"] = entries
    load_test.atomic_json_dump(report_path, report)

    population: np.ndarray | None = None
    query_ids: np.ndarray | None = None
    reference: np.ndarray | None = None
    baseline_schema: tuple[int, int, str] | None = None
    failures: list[str] = []

    for index, policy in enumerate(policies):
        entry = entries[index]
        entry["status"] = "running"
        load_test.atomic_json_dump(report_path, report)
        storage = None
        try:
            storage, store_format, open_metadata = _open_policy(store, policy)
            entry["open"] = open_metadata
            _require_native_v4(storage, store_format)

            schema = (
                int(storage.count),
                int(storage.dimension),
                str(storage.dtype),
            )
            if index == 0:
                population = load_test.load_ids(ids_path, storage.count)
                sample_rows = min(load_test.CORRECTNESS_SAMPLE_ROWS, storage.count)
                query_ids = load_test.query_matrix(
                    population,
                    rows=1,
                    batch_size=sample_rows,
                    seed=seed,
                )[0]
                baseline_schema = schema
            elif baseline_schema is None or schema != baseline_schema:
                raise AssertionError(
                    f"policy {policy.name} opened a different collection schema: "
                    f"{schema} != {baseline_schema}"
                )
            if query_ids is None:
                raise RuntimeError("baseline preflight did not produce a query")

            output, telemetry = _profiled_sample(storage, query_ids)
            entry["telemetry"] = telemetry
            entry["correctness"] = {
                "sample_rows": len(query_ids),
                "bit_exact_vs_baseline": index == 0,
            }
            if index == 0:
                reference = output
                report["storage"] = {
                    **open_metadata,
                    "count": storage.count,
                    "dimension": storage.dimension,
                    "dtype": storage.dtype,
                    "directory_bytes": load_test.directory_size(store),
                }
                report["correctness"] = load_test.correctness_check(
                    storage,
                    population,
                    source_embeddings,
                    source_kind,
                )
            else:
                if reference is None:
                    raise RuntimeError("baseline preflight did not produce output")
                _require_bit_exact(reference, output, policy.name)
                entry["correctness"]["bit_exact_vs_baseline"] = True

            if require_zero_delta and telemetry.get("zero_delta_fast_path") is not True:
                raise RuntimeError(
                    f"policy {policy.name} did not use the zero-delta fast path; "
                    "use a freshly built collection with no committed WAL "
                    "mutations or explicitly pass "
                    "--no-require-zero-delta"
                )
            entry["status"] = "passed"
        except Exception as error:
            entry["status"] = "failed"
            entry["error"] = _error_details(error)
            failures.append(f"{policy.name}: {error}")
        finally:
            if storage is not None:
                load_test.close_benchmark_storage(storage)
            load_test.atomic_json_dump(report_path, report)

        if index == 0 and entry["status"] != "passed" and reference is None:
            for blocked in entries[1:]:
                blocked["status"] = "blocked"
                blocked["error"] = {
                    "type": "preflight.baseline_unavailable",
                    "message": "baseline preflight did not produce reference output",
                }
            break

    if failures:
        report["preflight_status"] = "failed"
        load_test.atomic_json_dump(report_path, report)
        raise RuntimeError("policy preflight failed: " + "; ".join(failures))
    if population is None:
        raise RuntimeError("policy preflight did not load the ID population")
    report["preflight_status"] = "passed"
    load_test.atomic_json_dump(report_path, report)
    return population


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.store.is_dir():
        raise FileNotFoundError(f"storage directory does not exist: {args.store}")
    if not args.ids.is_file():
        raise FileNotFoundError(f"ID file does not exist: {args.ids}")
    if args.source_embeddings is not None and not args.source_embeddings.is_file():
        raise FileNotFoundError(
            f"source embeddings do not exist: {args.source_embeddings}"
        )

    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / "report.json"
    if report_path.exists():
        raise FileExistsError(
            f"result report already exists: {report_path}; use a new --results path"
        )

    scenarios = build_scenarios(args.calls_1024, args.calls_10000)
    measurement_blocks = build_measurement_blocks(args.repeats, scenarios)
    modes = cache_modes(args.cache_mode)
    expected_results = len(measurement_blocks) * len(modes)
    configuration = {
        "profile": "v4_reader_ab_v1",
        "store": str(args.store.resolve()),
        "ids": str(args.ids.resolve()),
        "results": str(args.results.resolve()),
        "cache_mode": args.cache_mode,
        "repeats": args.repeats,
        "seed": args.seed,
        "minimum_format_version": MINIMUM_FORMAT_VERSION,
        "require_zero_delta": args.require_zero_delta,
        "measurement_order": "repetition_scenario_policy_cache",
        "policies": [asdict(policy) for policy in READER_POLICIES],
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "expected_measurement_storage_open_count": len(measurement_blocks),
        "expected_preflight_storage_open_count": len(READER_POLICIES),
        "expected_result_count": expected_results,
        "source_embeddings": (
            str(args.source_embeddings.resolve())
            if args.source_embeddings is not None
            else None
        ),
        "source_kind": args.source_kind,
    }
    report: dict[str, Any] = {
        "configuration": configuration,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": load_test.collect_system_information(args.store),
        "preflight_status": "pending",
        "policy_preflight": [],
        "storage_opens": [],
        "results": [],
        "completed_result_count": 0,
    }
    load_test.atomic_json_dump(report_path, report)

    load_test.log("preflighting every reader policy")
    population = preflight_policies(
        store=args.store,
        ids_path=args.ids,
        source_embeddings=args.source_embeddings,
        source_kind=args.source_kind,
        seed=args.seed,
        require_zero_delta=args.require_zero_delta,
        report=report,
        report_path=report_path,
    )
    load_test.log(
        f"all policy preflights passed: "
        f"format={report['storage']['format_version']}, "
        f"count={report['storage']['count']:,}, "
        f"dimension={report['storage']['dimension']}"
    )

    diskstats_monitor = load_test.ProcDiskstatsMonitor(
        args.results / "diskstats.jsonl"
    )
    with (
        load_test.IostatMonitor(args.results / "iostat.log"),
        load_test.CommandMonitor(("vmstat", "1"), args.results / "vmstat.log"),
        diskstats_monitor,
    ):
        current_scenario_block: tuple[int, int] | None = None
        for block in measurement_blocks:
            repetition = block.repetition
            scenario = block.scenario
            policy = block.policy
            scenario_block = (repetition, block.scenario_index)
            if scenario_block != current_scenario_block:
                current_scenario_block = scenario_block
                order = rotated_policies(repetition)
                load_test.log(
                    f"repetition {repetition + 1}/{args.repeats}; "
                    f"scenario batch={scenario.batch_size:,}, "
                    f"clients={scenario.external_reader_workers}; policy order: "
                    + ", ".join(item.name for item in order)
                )

            seed = scenario_seed(args.seed, repetition, scenario)
            storage, actual_format, open_metadata = _open_policy(
                args.store, policy
            )
            try:
                _require_native_v4(storage, actual_format)
                report["storage_opens"].append(
                    {
                        "repetition": repetition + 1,
                        "scenario_index": block.scenario_index,
                        "batch_size": scenario.batch_size,
                        "external_reader_workers": (
                            scenario.external_reader_workers
                        ),
                        "policy_order_position": block.policy_order_position,
                        **open_metadata,
                    }
                )
                load_test.atomic_json_dump(report_path, report)
                for mode in modes:
                    eviction = None
                    if mode == "cold":
                        load_test.log(
                            "evicting embedding payload pages before "
                            f"policy={policy.name}, "
                            f"batch={scenario.batch_size:,}, "
                            f"clients={scenario.external_reader_workers}"
                        )
                        eviction = load_test.evict_data_cache(args.store)
                    result = load_test.concurrent_benchmark(
                        storage,
                        population,
                        scenario.external_reader_workers,
                        scenario.batch_size,
                        scenario.calls_per_worker,
                        seed,
                        prewarm_measurement=(
                            mode == "warm" and args.cache_mode == "warm"
                        ),
                    )
                    _annotate_result(
                        result,
                        policy,
                        repetition,
                        mode,
                        seed,
                        eviction,
                        block.scenario_index,
                        block.policy_order_position,
                    )
                    report["results"].append(result)
                    report["completed_result_count"] = len(report["results"])
                    load_test.print_result(result)
                    load_test.atomic_json_dump(report_path, report)
            finally:
                load_test.close_benchmark_storage(storage)

    if len(report["results"]) != expected_results:
        raise AssertionError(
            f"incomplete matrix: {len(report['results'])} != {expected_results}"
        )
    report["summary"] = aggregate_results(report["results"])
    report["diskstats"] = diskstats_monitor.summary()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    load_test.atomic_json_dump(report_path, report)
    load_test.log(f"complete A/B report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
