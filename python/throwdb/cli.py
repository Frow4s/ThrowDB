"""Small command-line interface built on top of the native bindings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ._calibration import CalibrationError
from ._tuning import TuningRegistryError
from .collection import Collection, CollectionSchema, _normalize_numeric_ids


DEFAULT_SHARD_SIZE_GIB = 16
DEFAULT_BATCH_ROWS = 32_768
DEFAULT_DELTA_COMPACTION_MAX_RUNS = 8
MAX_DELTA_COMPACTION_RUNS = 64
DEFAULT_DELTA_COMPACTION_MAX_INPUT_GIB = 8
DEFAULT_DELTA_COMPACTION_MAX_INPUT_ENTRIES = 1_000_000
DEFAULT_ORPHAN_MIN_AGE_SECONDS = 3_600.0
MAX_IO_RANGES_PER_TASK = 1 << 20
MAX_IO_URING_WORKERS = 64
MAX_IO_URING_QUEUE_DEPTH = 4096

_READER_ARGUMENTS = (
    "max_read_range_bytes",
    "max_gap_bytes",
    "io_workers",
    "io_queue_capacity",
    "io_ranges_per_task",
    "io_backend",
    "io_uring_queue_depth",
    "io_uring_workers",
)


def _load_ids(path: Path) -> np.ndarray:
    ids = np.load(path, mmap_mode="r", allow_pickle=False)
    if ids.ndim != 1:
        raise ValueError(f"ids must have shape (N,), got {ids.shape}")
    if ids.dtype.kind not in {"i", "u"} or ids.dtype.itemsize not in {
        1,
        2,
        4,
        8,
    }:
        raise ValueError(
            "ids must have an 8-, 16-, 32-, or 64-bit integer dtype, "
            f"got {ids.dtype}"
        )
    return ids


def _load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    if embeddings.dtype.name not in {"float16", "float32"}:
        raise ValueError(
            "embeddings must have dtype float16 or float32, "
            f"got {embeddings.dtype}"
        )
    if not embeddings.dtype.isnative:
        raise ValueError(
            "embeddings must use native byte order, "
            f"got {embeddings.dtype}"
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must have shape (N, dimension), got {embeddings.shape}"
        )
    if not embeddings.flags.c_contiguous:
        raise ValueError("embeddings must be C-contiguous")
    return embeddings


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
        raise argparse.ArgumentTypeError(
            "value must be finite and zero or greater"
        )
    return parsed


def _io_ranges_per_task(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > MAX_IO_RANGES_PER_TASK:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAX_IO_RANGES_PER_TASK}"
        )
    return parsed


def _io_uring_queue_depth(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > MAX_IO_URING_QUEUE_DEPTH:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAX_IO_URING_QUEUE_DEPTH}"
        )
    return parsed


def _io_uring_workers(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > MAX_IO_URING_WORKERS:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAX_IO_URING_WORKERS}"
        )
    return parsed


def _delta_compaction_runs(value: str) -> int:
    parsed = int(value)
    if parsed < 2 or parsed > MAX_DELTA_COMPACTION_RUNS:
        raise argparse.ArgumentTypeError(
            f"value must be between 2 and {MAX_DELTA_COMPACTION_RUNS}"
        )
    return parsed


def _create(args: argparse.Namespace) -> None:
    ids = _load_ids(args.ids)
    embeddings = _load_embeddings(args.embeddings)
    if ids.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "ids and embeddings must have the same number of rows: "
            f"{ids.shape[0]} != {embeddings.shape[0]}"
        )

    dimension = int(embeddings.shape[1])
    if args.dimension is not None and args.dimension != dimension:
        raise ValueError(
            f"--dimension is {args.dimension}, but embeddings have {dimension} columns"
        )

    shard_size_bytes = args.shard_size_gib * (1 << 30)
    schema = CollectionSchema(
        id_type="uint64",
        dimension=dimension,
        dtype=embeddings.dtype.name,
    )
    with Collection.create(
        args.storage,
        schema=schema,
        shard_size_bytes=shard_size_bytes,
    ) as collection:
        for begin in range(0, ids.shape[0], args.batch_rows):
            end = min(begin + args.batch_rows, ids.shape[0])
            collection.add(
                _normalize_numeric_ids(
                    ids[begin:end], position_offset=begin
                ),
                embeddings[begin:end],
            )


def _get(args: argparse.Namespace) -> None:
    ids = _load_ids(args.ids)
    with Collection.open(
        args.storage,
        tuning=_reader_tuning(args),
    ) as collection:
        output = np.lib.format.open_memmap(
            args.output,
            mode="w+",
            dtype=np.dtype(collection.dtype),
            shape=(ids.shape[0], collection.dimension),
        )
        for begin in range(0, ids.shape[0], args.batch_rows):
            end = min(begin + args.batch_rows, ids.shape[0])
            collection.get_into(
                _normalize_numeric_ids(
                    ids[begin:end], position_offset=begin
                ),
                output[begin:end],
            )
        output.flush()


def _inspect(args: argparse.Namespace) -> None:
    with Collection.open(
        args.storage,
        tuning=_reader_tuning(args),
        verify_index=args.verify_index,
    ) as collection:
        print(
            json.dumps(
                {
                    "path": str(args.storage),
                    "count": collection.count,
                    "dimension": collection.dimension,
                    "dtype": collection.dtype,
                    "format_version": collection.format_version,
                    "tuning": collection.tuning,
                    "open_stats": collection.open_stats,
                    "mutation_log": collection.mutation_log_stats,
                },
                indent=2,
                sort_keys=True,
            )
        )


def _rotate_wal(args: argparse.Namespace) -> None:
    with Collection.open(args.storage, mode="r+") as collection:
        print(
            json.dumps(
                collection.rotate_wal(),
                indent=2,
                sort_keys=True,
            )
        )


def _materialize(args: argparse.Namespace) -> None:
    with Collection.open(args.storage, mode="r+") as collection:
        print(
            json.dumps(
                collection.materialize_frozen_wals(),
                indent=2,
                sort_keys=True,
            )
        )


def _compact_deltas(args: argparse.Namespace) -> None:
    with Collection.open(args.storage, mode="r+") as collection:
        print(
            json.dumps(
                collection.compact_deltas(
                    max_runs=args.max_runs,
                    max_input_bytes=args.max_input_gib * (1 << 30),
                    max_input_entries=args.max_input_entries,
                ),
                indent=2,
                sort_keys=True,
            )
        )


def _collect_orphans(args: argparse.Namespace) -> None:
    report = Collection.inspect_orphans(
        args.storage,
        min_age_seconds=args.min_age_seconds,
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _write_json_atomic(path: Path, serialized: str) -> None:
    """Write an optional calibration report without exposing a partial file."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _calibration_report_error(
    report: object,
    action: str,
    error: BaseException,
) -> CalibrationError:
    selection = report.get("selection") if isinstance(report, dict) else None
    persisted = (
        isinstance(selection, dict) and selection.get("persisted") is True
    )
    persistence = (
        "the winning profile was saved"
        if persisted
        else "no winning profile was reported as saved"
    )
    return CalibrationError(
        f"calibration completed and {persistence}, but failed to {action}: "
        f"{error}"
    )


def _tune(args: argparse.Namespace) -> None:
    def progress(message: str) -> None:
        print(f"[throwdb tune] {message}", file=sys.stderr, flush=True)

    try:
        report = Collection.tune(
            args.storage,
            max_seconds=args.max_seconds,
            max_read_bytes=args.max_read_gib * (1 << 30),
            max_memory_bytes=args.max_memory_mib * (1 << 20),
            seed=args.seed,
            dry_run=args.dry_run,
            progress=progress,
        )
    except CalibrationError:
        raise
    except (OSError, RuntimeError) as error:
        raise CalibrationError(str(error)) from error

    try:
        serialized = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise _calibration_report_error(
            report,
            "serialize the JSON calibration report",
            error,
        ) from error

    if args.output is not None:
        try:
            _write_json_atomic(args.output, serialized)
        except OSError as error:
            raise _calibration_report_error(
                report,
                f"write the JSON calibration report to {args.output}",
                error,
            ) from error

    try:
        print(serialized)
    except OSError as error:
        raise _calibration_report_error(
            report,
            "write the JSON calibration report to stdout",
            error,
        ) from error


def _reader_tuning(
    args: argparse.Namespace,
) -> str | dict[str, object]:
    options = {
        name: value
        for name in _READER_ARGUMENTS
        if (value := getattr(args, name, None)) is not None
    }
    mode = getattr(args, "tuning", "auto")
    if options:
        if mode != "auto":
            raise ValueError(
                "explicit --io-* or range options cannot be combined with "
                f"--tuning {mode}"
            )
        return options
    return mode


def _tuning_show(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            Collection.tuning_profiles(args.storage),
            indent=2,
            sort_keys=True,
        )
    )


def _tuning_set(args: argparse.Namespace) -> None:
    options = _reader_tuning(args)
    if not isinstance(options, dict):
        raise ValueError("tuning set requires at least one reader option")
    print(
        json.dumps(
            Collection.save_tuning_profile(args.storage, options),
            indent=2,
            sort_keys=True,
        )
    )


def _tuning_remove(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            Collection.remove_tuning_profile(args.storage),
            indent=2,
            sort_keys=True,
        )
    )


def _tuning_reset(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            Collection.reset_tuning_profiles(
                args.storage,
                force=args.force,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def _add_reader_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_mode: bool,
) -> None:
    if include_mode:
        parser.add_argument(
            "--tuning",
            choices=("auto", "persisted", "conservative"),
            default="auto",
            help="select a saved profile, require one, or bypass saved profiles",
        )
    parser.add_argument(
        "--max-read-range-bytes",
        type=_positive_int,
        help="maximum merged physical range in bytes (manual default: 1 MiB)",
    )
    parser.add_argument(
        "--max-gap-bytes",
        type=_nonnegative_int,
        help="maximum unrequested gap merged into a range, in bytes",
    )
    parser.add_argument(
        "--io-workers",
        type=_nonnegative_int,
        help="pread worker count; 0 uses sequential reads",
    )
    parser.add_argument(
        "--io-queue-capacity",
        type=_nonnegative_int,
        help="pread task queue bound; 0 uses twice the worker count",
    )
    parser.add_argument(
        "--io-ranges-per-task",
        type=_io_ranges_per_task,
        help="maximum physical ranges in one worker task (manual default: 8)",
    )
    parser.add_argument(
        "--io-backend",
        choices=("thread_pool", "io_uring"),
        help="physical reader backend (manual default: thread_pool)",
    )
    parser.add_argument(
        "--io-uring-workers",
        type=_io_uring_workers,
        help="independent ring count (manual default: 1)",
    )
    parser.add_argument(
        "--io-uring-queue-depth",
        type=_io_uring_queue_depth,
        help="total queue depth across all rings (manual default: 32)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throwdb", description="Build and query a ThrowDB embedding store"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a store from NumPy files")
    create.add_argument("--storage", type=Path, required=True)
    create.add_argument("--ids", type=Path, required=True)
    create.add_argument("--embeddings", type=Path, required=True)
    create.add_argument("--dimension", type=_positive_int)
    create.add_argument(
        "--shard-size-gib", type=_positive_int, default=DEFAULT_SHARD_SIZE_GIB
    )
    create.add_argument(
        "--batch-rows", type=_positive_int, default=DEFAULT_BATCH_ROWS
    )
    create.set_defaults(handler=_create)

    get = subparsers.add_parser(
        "get", help="read IDs into a typed NumPy embedding file"
    )
    get.add_argument("--storage", type=Path, required=True)
    get.add_argument("--ids", type=Path, required=True)
    get.add_argument("--output", type=Path, required=True)
    get.add_argument("--batch-rows", type=_positive_int, default=DEFAULT_BATCH_ROWS)
    _add_reader_arguments(get, include_mode=True)
    get.set_defaults(handler=_get)

    inspect = subparsers.add_parser("inspect", help="print store metadata")
    inspect.add_argument("--storage", type=Path, required=True)
    _add_reader_arguments(inspect, include_mode=True)
    inspect.add_argument(
        "--verify-index",
        action="store_true",
        help="run the complete structural index audit while opening",
    )
    inspect.set_defaults(handler=_inspect)

    rotate_wal = subparsers.add_parser(
        "rotate-wal",
        help="seal a non-empty mutation WAL and publish its successor",
    )
    rotate_wal.add_argument("--storage", type=Path, required=True)
    rotate_wal.set_defaults(handler=_rotate_wal)

    materialize = subparsers.add_parser(
        "materialize",
        help="replace the frozen WAL prefix with an immutable delta run",
    )
    materialize.add_argument("--storage", type=Path, required=True)
    materialize.set_defaults(handler=_materialize)

    compact_deltas = subparsers.add_parser(
        "compact-deltas",
        help="compact a bounded oldest prefix of immutable delta runs",
    )
    compact_deltas.add_argument("--storage", type=Path, required=True)
    compact_deltas.add_argument(
        "--max-runs",
        type=_delta_compaction_runs,
        default=DEFAULT_DELTA_COMPACTION_MAX_RUNS,
        help="maximum input delta runs (default: 8)",
    )
    compact_deltas.add_argument(
        "--max-input-gib",
        type=_positive_int,
        default=DEFAULT_DELTA_COMPACTION_MAX_INPUT_GIB,
        help="maximum total input size in GiB (default: 8)",
    )
    compact_deltas.add_argument(
        "--max-input-entries",
        type=_positive_int,
        default=DEFAULT_DELTA_COMPACTION_MAX_INPUT_ENTRIES,
        help="maximum total input version records (default: 1000000)",
    )
    compact_deltas.set_defaults(handler=_compact_deltas)

    collect_orphans = subparsers.add_parser(
        "collect-orphans",
        help="report orphan mutation files without deleting them",
    )
    collect_orphans.add_argument("--storage", type=Path, required=True)
    collect_orphans.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="required safety acknowledgement; no files are deleted",
    )
    collect_orphans.add_argument(
        "--min-age-seconds",
        type=_nonnegative_float,
        default=DEFAULT_ORPHAN_MIN_AGE_SECONDS,
        help="minimum candidate age in seconds (default: 3600)",
    )
    collect_orphans.set_defaults(handler=_collect_orphans)

    tune = subparsers.add_parser(
        "tune",
        help="benchmark a bounded reader matrix and save the winning profile",
    )
    tune.add_argument("--storage", type=Path, required=True)
    tune.add_argument(
        "--max-seconds",
        type=_positive_float,
        default=300.0,
        help="cooperative wall-clock budget (default: 300)",
    )
    tune.add_argument(
        "--max-read-gib",
        type=_positive_int,
        default=8,
        help="hard planned physical-read budget in GiB (default: 8)",
    )
    tune.add_argument(
        "--max-memory-mib",
        type=_positive_int,
        default=512,
        help="additional query/output memory budget in MiB (default: 512)",
    )
    tune.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=20_260_713,
        help="deterministic workload seed",
    )
    tune.add_argument(
        "--dry-run",
        action="store_true",
        help="select a winner without changing tuning metadata",
    )
    tune.add_argument(
        "--output",
        type=Path,
        help="also save the complete JSON report atomically",
    )
    tune.set_defaults(handler=_tune)

    tuning = subparsers.add_parser(
        "tuning",
        help="inspect or manage persisted reader profiles",
    )
    tuning_commands = tuning.add_subparsers(
        dest="tuning_command",
        required=True,
    )

    tuning_show = tuning_commands.add_parser(
        "show",
        help="show saved profiles and the current target match",
    )
    tuning_show.add_argument("--storage", type=Path, required=True)
    tuning_show.set_defaults(handler=_tuning_show)

    tuning_set = tuning_commands.add_parser(
        "set",
        help="save explicit reader options for the current storage target",
    )
    tuning_set.add_argument("--storage", type=Path, required=True)
    _add_reader_arguments(tuning_set, include_mode=False)
    tuning_set.set_defaults(handler=_tuning_set)

    tuning_remove = tuning_commands.add_parser(
        "remove",
        help="remove the current storage target's saved profile",
    )
    tuning_remove.add_argument("--storage", type=Path, required=True)
    tuning_remove.set_defaults(handler=_tuning_remove)

    tuning_reset = tuning_commands.add_parser(
        "reset",
        help="remove all profiles; --force also recovers corrupt metadata",
    )
    tuning_reset.add_argument("--storage", type=Path, required=True)
    tuning_reset.add_argument(
        "--force",
        action="store_true",
        help="replace corrupt or unsupported tuning metadata",
    )
    tuning_reset.set_defaults(handler=_tuning_reset)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (CalibrationError, TuningRegistryError, TypeError, ValueError) as error:
        parser.error(str(error))
