#!/usr/bin/env python3
"""Measure random batch reads through the public Python API.

ID sampling and NumPy output allocation happen before the timed region. Supply
the complete ID array used to build the store so every generated query is valid.
This is a client-visible baseline; kernel-level I/O counters should be collected
separately (for example with iostat/perf) during large NVMe runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import throwdb


DEFAULT_BATCH_SIZES = "1,16,128,1024,10000,100000"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _batch_sizes(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated ints") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("every batch size must be greater than zero")
    return result


def _load_population(path: Path) -> np.ndarray:
    ids = np.load(path, mmap_mode="r", allow_pickle=False)
    if ids.dtype != np.dtype(np.uint64) or ids.ndim != 1:
        raise ValueError(f"expected uint64 IDs with shape (N,), got {ids.dtype} {ids.shape}")
    if ids.size == 0:
        raise ValueError("ID population is empty")
    return ids


def _queries(
    population: np.ndarray,
    *,
    batch_size: int,
    calls: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # Sampling offsets first keeps potentially expensive random generation out of
    # the timed path. copy() guarantees each query row is C-contiguous.
    offsets = rng.integers(0, population.size, size=(calls, batch_size), dtype=np.int64)
    return np.asarray(population[offsets], dtype=np.uint64).copy(order="C")


def _percentile_ms(samples_ns: np.ndarray, percentile: float) -> float:
    return float(np.percentile(samples_ns, percentile)) / 1_000_000.0


def _run_one(
    storage: object,
    population: np.ndarray,
    *,
    batch_size: int,
    calls: int,
    warmups: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    queries = _queries(
        population,
        batch_size=batch_size,
        calls=calls + warmups,
        rng=rng,
    )
    output = np.empty((batch_size, storage.dimension), dtype=np.float16)

    for query in queries[:warmups]:
        storage.get_into(query, output)

    samples = np.empty(calls, dtype=np.int64)
    was_enabled = gc.isenabled()
    gc.disable()
    wall_start = time.perf_counter_ns()
    try:
        for index, query in enumerate(queries[warmups:]):
            start = time.perf_counter_ns()
            storage.get_into(query, output)
            samples[index] = time.perf_counter_ns() - start
    finally:
        wall_ns = time.perf_counter_ns() - wall_start
        if was_enabled:
            gc.enable()

    embeddings = calls * batch_size
    useful_bytes = embeddings * storage.dimension * np.dtype(np.float16).itemsize
    seconds = wall_ns / 1_000_000_000.0
    return {
        "batch_size": batch_size,
        "calls": calls,
        "embeddings": embeddings,
        "seconds": seconds,
        "embeddings_per_second": embeddings / seconds,
        "useful_gib_per_second": useful_bytes / seconds / (1 << 30),
        "latency_p50_ms": _percentile_ms(samples, 50),
        "latency_p95_ms": _percentile_ms(samples, 95),
        "latency_p99_ms": _percentile_ms(samples, 99),
        "latency_max_ms": float(samples.max()) / 1_000_000.0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument(
        "--ids",
        type=Path,
        required=True,
        help="uint64 .npy array containing IDs present in the store",
    )
    parser.add_argument("--batch-sizes", type=_batch_sizes, default=_batch_sizes(DEFAULT_BATCH_SIZES))
    parser.add_argument("--calls", type=_positive_int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")

    population = _load_population(args.ids)
    storage = throwdb.Storage.open(str(args.storage))
    rng = np.random.default_rng(args.seed)
    report = {
        "storage": str(args.storage),
        "population_size": int(population.size),
        "storage_count": int(storage.count),
        "dimension": int(storage.dimension),
        "dtype": "float16",
        "seed": args.seed,
        "results": [
            _run_one(
                storage,
                population,
                batch_size=batch_size,
                calls=args.calls,
                warmups=args.warmups,
                rng=rng,
            )
            for batch_size in args.batch_sizes
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
