#!/usr/bin/env python3
"""Bounded local or gRPC streaming-read benchmark for one collection.

The benchmark deliberately never concatenates result batches.  It hashes each
completed batch, samples process RSS from procfs when available, and records
time-to-first-batch separately from steady-state throughput.  In remote mode
the target server must already be running; this harness never starts or stops
it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

from throwdb import Collection, RemoteCollection


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _proc_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if os.uname().sysname == "Darwin" else value * 1024


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"
    return f"{minutes}m {remaining_seconds:02d}s"


def _format_rss(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / (1 << 30):.2f} GiB"


def _progress_target_rows(
    expected_rows: int,
    max_rows: int | None,
) -> int:
    if expected_rows < 0:
        raise ValueError("expected_rows must not be negative")
    return (
        expected_rows
        if max_rows is None
        else min(expected_rows, max_rows)
    )


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(
            f"[{time.strftime('%H:%M:%S')}] {message}",
            file=sys.stderr,
            flush=True,
        )


class _ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_seconds: float,
        target_rows: int,
        started: float,
        sink: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("progress interval must be positive")
        if target_rows < 0:
            raise ValueError("progress target rows must not be negative")
        self._enabled = enabled
        self._interval_seconds = interval_seconds
        self._target_rows = target_rows
        self._started = started
        self._sink = sink if sink is not None else sys.stderr
        self._clock = clock
        self._next_report = started + interval_seconds
        self._reports = 0

    def report(
        self,
        *,
        rows: int,
        batches: int,
        logical_bytes: int,
        peak_rss_bytes: int | None,
        force: bool = False,
        status: str = "progress",
        now: float | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        observed_at = self._clock() if now is None else now
        first_batch = self._reports == 0 and rows > 0
        if not force and not first_batch and observed_at < self._next_report:
            return False

        elapsed = max(0.0, observed_at - self._started)
        rows_per_second = rows / elapsed if elapsed > 0 else 0.0
        logical_mib_per_second = (
            logical_bytes / (1 << 20) / elapsed if elapsed > 0 else 0.0
        )
        if self._target_rows == 0:
            percent = 100.0
            eta = 0.0
        else:
            bounded_rows = min(rows, self._target_rows)
            percent = bounded_rows / self._target_rows * 100.0
            eta = (
                (self._target_rows - bounded_rows) / rows_per_second
                if rows_per_second > 0
                else float("nan")
            )
        print(
            f"[{time.strftime('%H:%M:%S')}] {status}: "
            f"{rows:,}/{self._target_rows:,} rows ({percent:.2f}%), "
            f"{batches:,} batches, elapsed {_format_duration(elapsed)}, "
            f"{rows_per_second:,.0f} rows/s, "
            f"{logical_mib_per_second:.1f} logical MiB/s, "
            f"ETA {_format_duration(eta)}, "
            f"peak RSS {_format_rss(peak_rss_bytes)}",
            file=self._sink,
            flush=True,
        )
        self._next_report = observed_at + self._interval_seconds
        self._reports += 1
        return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--store",
        type=Path,
        help="open this format-7 collection in the current process",
    )
    target.add_argument(
        "--endpoint",
        help=(
            "connect to an already-running throwdbd gRPC endpoint; the "
            "benchmark never manages the server process"
        ),
    )
    parser.add_argument(
        "--mode", choices=("scan-records", "stream-get"), default="scan-records"
    )
    parser.add_argument(
        "--ids",
        type=Path,
        help="one-dimensional .npy ID array; required by --mode stream-get",
    )
    parser.add_argument("--batch-rows", type=_positive_int, default=65_536)
    parser.add_argument("--max-batch-mib", type=_positive_int, default=64)
    parser.add_argument(
        "--max-snapshot-mib",
        type=_positive_int,
        help="embedded-only snapshot-memory limit (default: 256 MiB)",
    )
    parser.add_argument("--prefix-dimensions", type=_positive_int)
    parser.add_argument("--with-locations", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=_positive_int,
        help=(
            "stop after the first completed batch that reaches this many "
            "rows (benchmark truncation may exceed the value by one batch)"
        ),
    )
    parser.add_argument("--rss-sample-every", type=_positive_int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tuning",
        choices=("auto", "persisted", "conservative"),
        help="embedded-only reader tuning mode (default: auto)",
    )
    parser.add_argument(
        "--rpc-timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="gRPC connect/metadata timeout (default: 30)",
    )
    parser.add_argument(
        "--scan-timeout-seconds",
        type=_positive_float,
        default=3_600.0,
        help="gRPC streaming call deadline (default: 3600)",
    )
    parser.add_argument(
        "--max-frame-mib",
        type=_positive_int,
        default=1024,
        help="gRPC send/receive frame ceiling (default: 1024 MiB)",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress-seconds",
        type=_positive_float,
        default=5.0,
        help=(
            "emit progress and ETA to stderr at most this often; the first "
            "and final batches are always reported (default: 5)"
        ),
    )
    progress.add_argument(
        "--no-progress",
        action="store_true",
        help="disable human-readable stderr progress",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "stream-get" and args.ids is None:
        raise SystemExit("--ids is required by --mode stream-get")
    if args.mode == "scan-records" and args.ids is not None:
        raise SystemExit("--ids is only valid with --mode stream-get")
    if args.mode == "stream-get" and args.with_locations:
        raise SystemExit("--with-locations is only valid with --mode scan-records")
    remote = args.endpoint is not None
    if remote and args.tuning is not None:
        raise SystemExit("--tuning is only valid with --store")
    if remote and args.max_snapshot_mib is not None:
        raise SystemExit("--max-snapshot-mib is only valid with --store")

    max_batch_bytes = args.max_batch_mib << 20
    max_snapshot_bytes = (
        (args.max_snapshot_mib or 256) << 20
        if not remote
        else None
    )
    max_frame_bytes = args.max_frame_mib << 20
    progress_enabled = not args.no_progress
    ids_source = (
        np.load(args.ids, mmap_mode="r", allow_pickle=False)
        if args.ids is not None
        else None
    )

    target_description = (
        str(args.store.resolve())
        if args.store is not None
        else str(args.endpoint)
    )
    _emit_progress(
        progress_enabled,
        f"opening {'gRPC endpoint' if remote else 'embedded collection'} "
        f"{target_description}",
    )
    opened_at = time.monotonic()
    if remote:
        assert args.endpoint is not None
        collection = RemoteCollection.connect(
            args.endpoint,
            timeout=args.rpc_timeout_seconds,
            scan_timeout=args.scan_timeout_seconds,
            max_frame_bytes=max_frame_bytes,
        )
    else:
        assert args.store is not None
        collection = Collection.open(
            args.store,
            tuning=args.tuning or "auto",
        )
    opened_seconds = time.monotonic() - opened_at
    _emit_progress(
        progress_enabled,
        f"collection opened in {_format_duration(opened_seconds)}; "
        f"preparing {args.mode}",
    )
    embedding_digest = hashlib.blake2b(digest_size=32)
    id_digest = hashlib.blake2b(digest_size=32)
    location_digest = hashlib.blake2b(digest_size=32)
    schema_dimension = collection.dimension
    schema_dtype = collection.dtype
    rows = 0
    batches = 0
    logical_bytes = 0
    first_batch_seconds: float | None = None
    peak_sampled_rss = _proc_rss_bytes()
    started = time.monotonic()

    try:
        if args.mode == "scan-records":
            if remote:
                stream = collection.scan_records(
                    batch_rows=args.batch_rows,
                    max_batch_payload_bytes=max_batch_bytes,
                    prefix_dimensions=args.prefix_dimensions,
                    with_locations=args.with_locations,
                )
            else:
                assert max_snapshot_bytes is not None
                stream = collection.scan_records(
                    batch_rows=args.batch_rows,
                    max_batch_bytes=max_batch_bytes,
                    max_snapshot_bytes=max_snapshot_bytes,
                    prefix_dimensions=args.prefix_dimensions,
                    with_locations=args.with_locations,
                )
        else:
            assert ids_source is not None
            if remote:
                stream = collection.stream_get(
                    ids_source,
                    batch_rows=args.batch_rows,
                    max_batch_payload_bytes=max_batch_bytes,
                    prefix_dimensions=args.prefix_dimensions,
                )
            else:
                assert max_snapshot_bytes is not None
                stream = collection.stream_get(
                    ids_source,
                    batch_rows=args.batch_rows,
                    max_batch_bytes=max_batch_bytes,
                    max_snapshot_bytes=max_snapshot_bytes,
                    prefix_dimensions=args.prefix_dimensions,
                )

        snapshot = stream.snapshot
        snapshot_bytes = getattr(stream, "snapshot_bytes", None)
        effective_batch_rows = int(
            getattr(
                stream,
                "batch_capacity",
                getattr(stream, "max_batch_rows", args.batch_rows),
            )
        )
        effective_batch_payload_bytes = getattr(
            stream,
            "max_batch_payload_bytes",
            None,
        )
        expected_rows = (
            int(ids_source.size)
            if ids_source is not None
            else int(snapshot.count)
        )
        progress_target_rows = _progress_target_rows(
            expected_rows,
            args.max_rows,
        )
        progress = _ProgressReporter(
            enabled=progress_enabled,
            interval_seconds=args.progress_seconds,
            target_rows=progress_target_rows,
            started=started,
        )
        stop_after_rows = (
            args.max_rows
            if args.max_rows is not None and args.max_rows < expected_rows
            else None
        )
        _emit_progress(
            progress_enabled,
            f"stream ready: expected_rows={expected_rows:,}, "
            f"target_rows={progress_target_rows:,}, "
            f"effective_batch_rows={effective_batch_rows:,}, "
            f"snapshot_bytes={snapshot_bytes if snapshot_bytes is not None else 'remote'}",
        )
        with stream:
            for item in stream:
                if args.mode == "scan-records":
                    batch_ids = item.ids
                    embeddings = item.embeddings
                    id_digest.update(
                        memoryview(np.ascontiguousarray(batch_ids))
                    )
                    if item.locations is not None:
                        location_digest.update(
                            memoryview(np.ascontiguousarray(item.locations))
                        )
                else:
                    embeddings = item.embeddings if remote else item
                embedding_digest.update(
                    memoryview(np.ascontiguousarray(embeddings))
                )
                count = int(embeddings.shape[0])
                rows += count
                batches += 1
                logical_bytes += int(embeddings.nbytes)
                if first_batch_seconds is None:
                    first_batch_seconds = time.monotonic() - started
                if batches % args.rss_sample_every == 0:
                    sampled = _proc_rss_bytes()
                    if sampled is not None:
                        peak_sampled_rss = max(peak_sampled_rss or 0, sampled)
                progress.report(
                    rows=rows,
                    batches=batches,
                    logical_bytes=logical_bytes,
                    peak_rss_bytes=peak_sampled_rss,
                )
                if stop_after_rows is not None and rows >= stop_after_rows:
                    break
        stream_done = bool(stream.done)
        stream_failed = bool(stream.failed)
        stream_closed = bool(stream.closed)
    finally:
        collection.close()

    if stream_failed:
        raise RuntimeError("stream reported a failed terminal state")
    if not stream_closed:
        raise RuntimeError("stream context exited without closing the cursor")
    if stream_done and rows != expected_rows:
        raise RuntimeError(
            "stream claimed logical EOF at the wrong row count: "
            f"rows={rows}, expected_rows={expected_rows}, done={stream_done}"
        )
    if not stream_done and (
        stop_after_rows is None or rows < stop_after_rows
    ):
        raise RuntimeError(
            "stream ended before logical EOF without reaching the explicit "
            f"truncation bound: rows={rows}, max_rows={stop_after_rows}"
        )

    component_digests = {
        "ids": id_digest.hexdigest() if args.mode == "scan-records" else None,
        "embeddings": embedding_digest.hexdigest(),
        "locations": (
            location_digest.hexdigest()
            if args.mode == "scan-records" and args.with_locations
            else None
        ),
    }
    digest = hashlib.blake2b(digest_size=32)
    for name in ("ids", "embeddings", "locations"):
        value = component_digests[name]
        if value is not None:
            digest.update(name.encode("ascii") + b"\0")
            digest.update(bytes.fromhex(value))

    finished = time.monotonic()
    elapsed = finished - started
    final_rss = _proc_rss_bytes()
    if final_rss is not None:
        peak_sampled_rss = max(peak_sampled_rss or 0, final_rss)
    report: dict[str, Any] = {
        "format": "throwdb-streaming-read-benchmark",
        "version": 2,
        "mode": args.mode,
        "target": {
            "kind": "grpc" if remote else "embedded",
            "store": (
                str(args.store.resolve())
                if args.store is not None
                else None
            ),
            "endpoint": args.endpoint,
        },
        "store": (
            str(args.store.resolve())
            if args.store is not None
            else None
        ),
        "endpoint": args.endpoint,
        "ids": str(args.ids.resolve()) if args.ids is not None else None,
        "schema": {
            "dimension": schema_dimension,
            "dtype": schema_dtype,
            "prefix_dimensions": args.prefix_dimensions,
        },
        "limits": {
            "batch_rows": args.batch_rows,
            "effective_batch_rows": effective_batch_rows,
            "max_batch_bytes": max_batch_bytes,
            "max_snapshot_bytes": max_snapshot_bytes,
            "effective_batch_payload_bytes": (
                int(effective_batch_payload_bytes)
                if effective_batch_payload_bytes is not None
                else None
            ),
            "max_frame_bytes": max_frame_bytes if remote else None,
            "max_rows": args.max_rows,
            "effective_stop_rows": stop_after_rows,
            "progress_seconds": (
                args.progress_seconds if progress_enabled else None
            ),
        },
        "snapshot": {
            "collection_id": snapshot.collection_id,
            "catalog_generation": snapshot.catalog_generation,
            "committed_sequence": snapshot.committed_sequence,
            "count": snapshot.count,
            "snapshot_bytes": snapshot_bytes,
        },
        "result": {
            "rows": rows,
            "expected_rows": expected_rows,
            "batches": batches,
            "logical_eof": stream_done,
            "stream_failed": stream_failed,
            "stream_closed": stream_closed,
            "logical_embedding_bytes": logical_bytes,
            "digest_blake2b_256": digest.hexdigest(),
            "component_digests_blake2b_256": component_digests,
            "open_seconds": opened_seconds,
            "time_to_first_batch_seconds": first_batch_seconds,
            "elapsed_seconds": elapsed,
            "rows_per_second": rows / elapsed if elapsed else None,
            "logical_gib_per_second": (
                logical_bytes / (1 << 30) / elapsed if elapsed else None
            ),
            "peak_sampled_rss_bytes": peak_sampled_rss,
            "process_max_rss_bytes": _maximum_rss_bytes(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress.report(
        rows=rows,
        batches=batches,
        logical_bytes=logical_bytes,
        peak_rss_bytes=peak_sampled_rss,
        force=True,
        status="complete" if stream_done else "stopped_at_limit",
        now=finished,
    )
    print(json.dumps(report["result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
