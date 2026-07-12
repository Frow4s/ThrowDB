"""Small command-line interface built on top of the native bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import Storage, Writer
from .collection import _normalize_numeric_ids


DEFAULT_SHARD_SIZE_GIB = 16
DEFAULT_BATCH_ROWS = 32_768
DEFAULT_IO_RANGES_PER_TASK = 8
MAX_IO_RANGES_PER_TASK = 1 << 20
DEFAULT_IO_BACKEND = "thread_pool"
DEFAULT_IO_URING_WORKERS = 1
MAX_IO_URING_WORKERS = 64
DEFAULT_IO_URING_QUEUE_DEPTH = 32
MAX_IO_URING_QUEUE_DEPTH = 4096


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
    with Writer.create(
        args.storage,
        dimension=dimension,
        shard_size_bytes=shard_size_bytes,
        dtype=embeddings.dtype.name,
    ) as writer:
        for begin in range(0, ids.shape[0], args.batch_rows):
            end = min(begin + args.batch_rows, ids.shape[0])
            writer.add(
                _normalize_numeric_ids(
                    ids[begin:end], position_offset=begin
                ),
                embeddings[begin:end],
            )


def _get(args: argparse.Namespace) -> None:
    ids = _load_ids(args.ids)
    storage = Storage.open(
        args.storage,
        io_workers=args.io_workers,
        io_queue_capacity=args.io_queue_capacity,
        io_ranges_per_task=args.io_ranges_per_task,
        io_backend=args.io_backend,
        io_uring_queue_depth=args.io_uring_queue_depth,
        io_uring_workers=args.io_uring_workers,
    )
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.dtype(storage.dtype),
        shape=(ids.shape[0], storage.dimension),
    )
    for begin in range(0, ids.shape[0], args.batch_rows):
        end = min(begin + args.batch_rows, ids.shape[0])
        storage.get_into(
            _normalize_numeric_ids(
                ids[begin:end], position_offset=begin
            ),
            output[begin:end],
        )
    output.flush()


def _inspect(args: argparse.Namespace) -> None:
    storage = Storage.open(
        args.storage,
        io_workers=args.io_workers,
        io_queue_capacity=args.io_queue_capacity,
        io_ranges_per_task=args.io_ranges_per_task,
        io_backend=args.io_backend,
        io_uring_queue_depth=args.io_uring_queue_depth,
        io_uring_workers=args.io_uring_workers,
    )
    print(
        json.dumps(
            {
                "path": str(args.storage),
                "count": storage.count,
                "dimension": storage.dimension,
                "dtype": storage.dtype,
                "format_version": storage.format_version,
            },
            indent=2,
            sort_keys=True,
        )
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
    get.add_argument("--io-workers", type=_nonnegative_int, default=0)
    get.add_argument("--io-queue-capacity", type=_nonnegative_int, default=0)
    get.add_argument(
        "--io-ranges-per-task",
        type=_io_ranges_per_task,
        default=DEFAULT_IO_RANGES_PER_TASK,
    )
    get.add_argument(
        "--io-backend",
        choices=("thread_pool", "io_uring"),
        default=DEFAULT_IO_BACKEND,
    )
    get.add_argument(
        "--io-uring-workers",
        type=_io_uring_workers,
        default=DEFAULT_IO_URING_WORKERS,
    )
    get.add_argument(
        "--io-uring-queue-depth",
        type=_io_uring_queue_depth,
        default=DEFAULT_IO_URING_QUEUE_DEPTH,
    )
    get.set_defaults(handler=_get)

    inspect = subparsers.add_parser("inspect", help="print store metadata")
    inspect.add_argument("--storage", type=Path, required=True)
    inspect.add_argument("--io-workers", type=_nonnegative_int, default=0)
    inspect.add_argument("--io-queue-capacity", type=_nonnegative_int, default=0)
    inspect.add_argument(
        "--io-ranges-per-task",
        type=_io_ranges_per_task,
        default=DEFAULT_IO_RANGES_PER_TASK,
    )
    inspect.add_argument(
        "--io-backend",
        choices=("thread_pool", "io_uring"),
        default=DEFAULT_IO_BACKEND,
    )
    inspect.add_argument(
        "--io-uring-workers",
        type=_io_uring_workers,
        default=DEFAULT_IO_URING_WORKERS,
    )
    inspect.add_argument(
        "--io-uring-queue-depth",
        type=_io_uring_queue_depth,
        default=DEFAULT_IO_URING_QUEUE_DEPTH,
    )
    inspect.set_defaults(handler=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.handler(args)
