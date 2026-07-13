#!/usr/bin/env python3
"""Destructive-but-reversible smoke test for a native mutable collection.

The test reads only a small sample from the source ``.npy`` files.  It writes
several durable WAL transactions to ``--store``, verifies recovery after a
close/reopen cycle, and restores every sampled base row before reporting
success.  A forced process termination or machine failure can interrupt that
cleanup, so this script is intended for a collection that can be rebuilt from
the supplied NumPy files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

import throwdb


DEFAULT_SAMPLE_ROWS = 64
DEFAULT_RANDOM_SEED = 20260712
SYNTHETIC_ID_COUNT = 3
MAX_SYNTHETIC_PROBES = 4_096
_T = TypeVar("_T")


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _duration_entry(elapsed_seconds: float, rows: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {"elapsed_ms": elapsed_seconds * 1_000.0}
    if rows is not None:
        result["rows"] = rows
        if rows:
            result["microseconds_per_row"] = elapsed_seconds * 1_000_000.0 / rows
    return result


def _timed(
    timings: dict[str, dict[str, Any]],
    name: str,
    operation: Callable[[], _T],
    *,
    rows: int | None = None,
) -> _T:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        timings[name] = _duration_entry(time.perf_counter() - started, rows)


def _load_npy_memmap(path: Path, label: str) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise ValueError(f"cannot open {label} NumPy file {path}: {error}") from error
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{label} must contain one NumPy ndarray")
    return array


def _sample_row_numbers(count: int, sample_rows: int, seed: int) -> np.ndarray:
    """Sample unique row numbers with O(sample_rows) memory.

    ``Generator.choice(count, replace=False)`` is deliberately avoided because
    some NumPy versions allocate work proportional to ``count``.  That is not
    acceptable when the source contains hundreds of millions of rows.
    """

    wanted = min(count, sample_rows)
    generator = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < wanted:
        selected.add(int(generator.integers(0, count)))
    return np.fromiter(sorted(selected), dtype=np.int64, count=wanted)


def _native_sample_ids(source: np.ndarray, rows: np.ndarray) -> np.ndarray:
    sampled = np.asarray(source[rows])
    if sampled.dtype.kind == "i" and bool((sampled < 0).any()):
        raise ValueError("sampled source IDs contain a negative value")
    ids = np.require(sampled, dtype=np.uint64, requirements=["C", "A"])
    if np.unique(ids).size != ids.size:
        raise ValueError("sampled source IDs are not unique")
    return ids


def _native_sample_embeddings(
    source: np.ndarray,
    rows: np.ndarray,
    dtype: str,
) -> np.ndarray:
    return np.require(source[rows], dtype=np.dtype(dtype), requirements=["C", "A"])


def _assert_bit_exact(
    actual: np.ndarray,
    expected: np.ndarray,
    context: str,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{context}: shape mismatch: actual {actual.shape}, expected {expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{context}: dtype mismatch: actual {actual.dtype}, expected {expected.dtype}"
        )
    if actual.tobytes(order="C") != expected.tobytes(order="C"):
        raise AssertionError(f"{context}: embedding payloads are not bit-exact")


def _pattern_vectors(rows: int, dimension: int, dtype: str, salt: int) -> np.ndarray:
    """Create deterministic payloads, including arbitrary valid float bit patterns."""

    float_dtype = np.dtype(dtype)
    unsigned_dtype = np.uint16 if float_dtype.itemsize == 2 else np.uint32
    generator = np.random.default_rng(salt)
    bits = generator.integers(
        0,
        np.iinfo(unsigned_dtype).max,
        size=(rows, dimension),
        dtype=unsigned_dtype,
        endpoint=True,
    )
    return bits.view(float_dtype)


def _assert_missing(collection: throwdb.Collection, identifier: int) -> None:
    try:
        collection.get(np.array([identifier], dtype=np.uint64))
    except KeyError:
        return
    raise AssertionError(f"synthetic ID {identifier} unexpectedly exists")


def _find_absent_ids(
    collection: throwdb.Collection,
    count: int,
    seed: int,
) -> np.ndarray:
    """Find absent uint64 IDs using bounded point lookups, never a full ID scan."""

    generator = np.random.default_rng(seed)
    absent: list[int] = []
    considered: set[int] = set()
    probes = 0
    while len(absent) < count and probes < MAX_SYNTHETIC_PROBES:
        # Generate candidates in small batches, then use authoritative get
        # semantics for each candidate.  At normal collection densities the
        # first batch is overwhelmingly likely to be sufficient.
        candidates = generator.bit_generator.random_raw(max(16, count * 4))
        for raw_candidate in candidates:
            candidate = int(raw_candidate)
            if candidate in considered:
                continue
            considered.add(candidate)
            probes += 1
            try:
                collection.get(np.array([candidate], dtype=np.uint64))
            except KeyError:
                absent.append(candidate)
                if len(absent) == count:
                    break
            if probes >= MAX_SYNTHETIC_PROBES:
                break
    if len(absent) != count:
        raise RuntimeError(
            f"could not find {count} absent synthetic IDs in "
            f"{MAX_SYNTHETIC_PROBES} bounded probes"
        )
    return np.asarray(absent, dtype=np.uint64)


def _verify_sources(
    ids: np.ndarray,
    embeddings: np.ndarray,
    collection: throwdb.Collection,
) -> None:
    if ids.ndim != 1:
        raise ValueError(f"IDs must have shape (N,), got {ids.shape}")
    if ids.dtype.kind not in {"i", "u"} or ids.dtype.itemsize not in {1, 2, 4, 8}:
        raise TypeError(f"IDs must have an integer dtype, got {ids.dtype}")
    if embeddings.ndim != 2:
        raise ValueError(
            f"source embeddings must have shape (N, dimension), got {embeddings.shape}"
        )
    if ids.shape[0] != embeddings.shape[0]:
        raise ValueError("IDs and source embeddings have different row counts")
    if collection.count != ids.shape[0]:
        raise ValueError(
            "collection count does not match the source arrays: "
            f"{collection.count} != {ids.shape[0]}"
        )
    if collection.dimension != embeddings.shape[1]:
        raise ValueError(
            "collection dimension does not match source embeddings: "
            f"{collection.dimension} != {embeddings.shape[1]}"
        )
    source_dtype = embeddings.dtype.newbyteorder("=").name
    if source_dtype != collection.dtype:
        raise TypeError(
            "collection dtype does not match source embeddings: "
            f"{collection.dtype} != {embeddings.dtype}"
        )
    if collection.count == 0:
        raise ValueError("the mutable smoke test requires a non-empty collection")


def _cleanup_and_verify(
    store: Path,
    base_count: int,
    sampled_ids: np.ndarray,
    original_embeddings: np.ndarray,
    synthetic_ids: np.ndarray,
    timings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": True, "status": "running"}
    try:
        def restore() -> list[bool]:
            with throwdb.Collection.open(store, mode="r+") as collection:
                collection.upsert(sampled_ids, original_embeddings)
                deleted = collection.delete(synthetic_ids)
                return [bool(value) for value in deleted]

        delete_results = _timed(
            timings,
            "cleanup_restore",
            restore,
            rows=int(sampled_ids.size + synthetic_ids.size),
        )

        def verify() -> None:
            with throwdb.Collection.open(store) as collection:
                if collection.count != base_count:
                    raise AssertionError(
                        "cleanup count mismatch: "
                        f"actual {collection.count}, expected {base_count}"
                    )
                restored = collection.get(sampled_ids)
                _assert_bit_exact(restored, original_embeddings, "cleanup verification")
                for identifier in synthetic_ids:
                    _assert_missing(collection, int(identifier))

        _timed(
            timings,
            "cleanup_reopen_verify",
            verify,
            rows=int(sampled_ids.size),
        )
        result.update(
            {
                "status": "verified",
                "synthetic_delete_results": delete_results,
                "count_after": base_count,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return result


def run_mutable_test(
    *,
    store: Path,
    ids_path: Path,
    source_embeddings_path: Path,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Run the mutable smoke test and return a JSON-serializable report."""

    report: dict[str, Any] = {
        "status": "running",
        "store": str(store),
        "ids": str(ids_path),
        "source_embeddings": str(source_embeddings_path),
        "sample_rows_requested": sample_rows,
        "seed": seed,
        "warning": (
            "This test durably mutates a rebuildable test store. Normal completion "
            "restores sampled base rows and removes synthetic IDs; a forced process "
            "or machine termination can interrupt cleanup."
        ),
        "timings": {},
        "checks": {},
        "cleanup": {"attempted": False, "status": "not_needed"},
    }
    timings: dict[str, dict[str, Any]] = report["timings"]
    checks: dict[str, Any] = report["checks"]
    writer: throwdb.Collection | None = None
    mutation_cleanup_required = False
    sampled_ids: np.ndarray | None = None
    original_embeddings: np.ndarray | None = None
    synthetic_ids: np.ndarray | None = None
    base_count: int | None = None
    primary_error: Exception | None = None

    try:
        ids = _load_npy_memmap(ids_path, "IDs")
        source_embeddings = _load_npy_memmap(
            source_embeddings_path, "source embeddings"
        )

        reader = _timed(
            timings,
            "initial_open",
            lambda: throwdb.Collection.open(store),
        )
        try:
            _verify_sources(ids, source_embeddings, reader)
            base_count = reader.count
            row_numbers = _sample_row_numbers(base_count, sample_rows, seed)
            sampled_ids = _native_sample_ids(ids, row_numbers)
            original_embeddings = _native_sample_embeddings(
                source_embeddings, row_numbers, reader.dtype
            )
            report.update(
                {
                    "format_version": reader.format_version,
                    "base_count": base_count,
                    "dimension": reader.dimension,
                    "dtype": reader.dtype,
                    "sample_rows": int(sampled_ids.size),
                }
            )
            if reader.format_version not in {3, 4, 5, 6}:
                raise ValueError(
                    "the mutable smoke test requires native mutable format "
                    f"version 3, 4, or 5, got {reader.format_version}"
                )

            actual = _timed(
                timings,
                "base_read",
                lambda: reader.get(sampled_ids),
                rows=int(sampled_ids.size),
            )
            _assert_bit_exact(actual, original_embeddings, "base read")
            checks["base_read_bit_exact"] = True

            prefix_dimensions = min(128, reader.dimension)
            prefix = _timed(
                timings,
                "prefix_read",
                lambda: reader.get(
                    sampled_ids, prefix_dimensions=prefix_dimensions
                ),
                rows=int(sampled_ids.size),
            )
            _assert_bit_exact(
                prefix,
                original_embeddings[:, :prefix_dimensions].copy(order="C"),
                "prefix read",
            )
            checks["prefix_read_bit_exact"] = True
            checks["prefix_dimensions"] = prefix_dimensions

            replacement = _pattern_vectors(
                1, reader.dimension, reader.dtype, seed + 101
            )
            try:
                reader.update(sampled_ids[:1], replacement)
            except RuntimeError:
                checks["read_only_rejected_mutation"] = True
            else:
                raise AssertionError("read-only collection accepted update")
            _assert_bit_exact(
                reader.get(sampled_ids), original_embeddings, "read-only rejection"
            )
        finally:
            reader.close()

        assert base_count is not None
        assert sampled_ids is not None
        assert original_embeddings is not None
        dimension = original_embeddings.shape[1]
        dtype = original_embeddings.dtype.name

        writer = _timed(
            timings,
            "writer_open",
            lambda: throwdb.Collection.open(store, mode="r+"),
        )
        lease_started = time.perf_counter()
        second_writer: throwdb.Collection | None = None
        try:
            second_writer = throwdb.Collection.open(store, mode="r+")
        except (OSError, RuntimeError) as error:
            checks["writer_lease"] = True
            checks["writer_lease_error_type"] = type(error).__name__
        else:
            raise AssertionError("a second read-write handle acquired the writer lease")
        finally:
            timings["writer_lease_rejection"] = _duration_entry(
                time.perf_counter() - lease_started, None
            )
            if second_writer is not None:
                second_writer.close()

        synthetic_ids = _timed(
            timings,
            "synthetic_id_probes",
            lambda: _find_absent_ids(writer, SYNTHETIC_ID_COUNT, seed + 211),
            rows=SYNTHETIC_ID_COUNT,
        )
        report["synthetic_ids"] = [int(value) for value in synthetic_ids]
        # From this point onward, cleanup is attempted even if an invariant
        # fails: a broken atomicity guarantee could have written synth[0].
        mutation_cleanup_required = True

        inserted_vector = _pattern_vectors(1, dimension, dtype, seed + 301)
        collision_vectors = np.concatenate(
            [inserted_vector, original_embeddings[:1]], axis=0
        )
        collision_started = time.perf_counter()
        try:
            writer.insert(
                np.array([synthetic_ids[0], sampled_ids[0]], dtype=np.uint64),
                collision_vectors,
            )
        except ValueError:
            checks["insert_collision_rejected"] = True
        else:
            raise AssertionError("insert accepted a batch containing an existing ID")
        finally:
            timings["insert_collision_rejection"] = _duration_entry(
                time.perf_counter() - collision_started, 2
            )
        _assert_missing(writer, int(synthetic_ids[0]))
        if writer.count != base_count:
            raise AssertionError("failed insert collision changed collection count")
        checks["insert_collision_atomic"] = True

        updated_vectors = _pattern_vectors(
            sampled_ids.size, dimension, dtype, seed + 401
        )
        _timed(
            timings,
            "update",
            lambda: writer.update(sampled_ids, updated_vectors),
            rows=int(sampled_ids.size),
        )
        if writer.count != base_count:
            raise AssertionError("update changed collection count")

        _timed(
            timings,
            "insert",
            lambda: writer.insert(synthetic_ids[:1], inserted_vector),
            rows=1,
        )
        if writer.count != base_count + 1:
            raise AssertionError("insert did not increment collection count")

        upsert_vectors = _pattern_vectors(2, dimension, dtype, seed + 501)
        upsert_ids = np.array(
            [sampled_ids[0], synthetic_ids[1]], dtype=np.uint64
        )
        _timed(
            timings,
            "upsert",
            lambda: writer.upsert(upsert_ids, upsert_vectors),
            rows=2,
        )
        if writer.count != base_count + 2:
            raise AssertionError("mixed upsert produced an incorrect count")

        delete_ids = np.array(
            [synthetic_ids[0], synthetic_ids[2]], dtype=np.uint64
        )
        deleted = _timed(
            timings,
            "delete",
            lambda: writer.delete(delete_ids),
            rows=2,
        )
        if deleted.tolist() != [True, False]:
            raise AssertionError(
                f"delete returned {deleted.tolist()}, expected [True, False]"
            )
        if writer.count != base_count + 1:
            raise AssertionError("delete produced an incorrect count")
        checks["crud_counts"] = True

        writer.close()
        writer = None

        recovered = _timed(
            timings,
            "recovery_open",
            lambda: throwdb.Collection.open(store),
        )
        try:
            expected_existing = updated_vectors.copy(order="C")
            expected_existing[0] = upsert_vectors[0]
            recovered_existing = _timed(
                timings,
                "recovery_read",
                lambda: recovered.get(sampled_ids),
                rows=int(sampled_ids.size),
            )
            _assert_bit_exact(
                recovered_existing, expected_existing, "WAL recovery existing rows"
            )
            _assert_bit_exact(
                recovered.get(synthetic_ids[1:2]),
                upsert_vectors[1:2],
                "WAL recovery inserted row",
            )
            _assert_missing(recovered, int(synthetic_ids[0]))
            if recovered.count != base_count + 1:
                raise AssertionError("recovered collection count is incorrect")
            checks["wal_recovery_bit_exact"] = True
        finally:
            recovered.close()
    except Exception as error:
        primary_error = error
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                report["writer_close_error"] = f"{type(error).__name__}: {error}"
        if (
            mutation_cleanup_required
            and base_count is not None
            and sampled_ids is not None
            and original_embeddings is not None
            and synthetic_ids is not None
        ):
            report["cleanup"] = _cleanup_and_verify(
                store,
                base_count,
                sampled_ids,
                original_embeddings,
                synthetic_ids,
                timings,
            )

    cleanup_ok = report["cleanup"]["status"] in {"not_needed", "verified"}
    if primary_error is None and cleanup_ok:
        report["status"] = "passed"
    else:
        report["status"] = "failed"
        if primary_error is None and not cleanup_ok:
            report["error_type"] = "CleanupError"
            report["error"] = report["cleanup"].get(
                "error", "cleanup verification failed"
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise native mutable CRUD/WAL recovery on a rebuildable ThrowDB store "
            "and restore it on normal completion."
        ),
        epilog=(
            "WARNING: the command durably mutates the store. Do not run it on the "
            "only copy of non-rebuildable data."
        ),
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True, help="source IDs .npy")
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        required=True,
        help="source embeddings .npy",
    )
    parser.add_argument(
        "--sample-rows",
        type=positive_int,
        default=DEFAULT_SAMPLE_ROWS,
        help=f"number of existing rows to exercise (default: {DEFAULT_SAMPLE_ROWS})",
    )
    parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=DEFAULT_RANDOM_SEED,
        help=f"deterministic sampling seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="optional destination JSON file",
    )
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        "WARNING: this test writes durable WAL mutations to a rebuildable test "
        "store. Cleanup is verified on normal completion, but cannot be guaranteed "
        "after SIGKILL, power loss, or machine failure.",
        file=sys.stderr,
        flush=True,
    )
    log(f"opening native mutable store: {arguments.store}")
    report = run_mutable_test(
        store=arguments.store,
        ids_path=arguments.ids,
        source_embeddings_path=arguments.source_embeddings,
        sample_rows=arguments.sample_rows,
        seed=arguments.seed,
    )
    if arguments.results is not None:
        _write_report(arguments.results, report)
        log(f"wrote report: {arguments.results}")
    cleanup = report["cleanup"]
    log(
        f"status={report['status']} cleanup={cleanup['status']} "
        f"sample_rows={report.get('sample_rows', 0)}"
    )
    if report["status"] != "passed":
        print(
            f"ERROR: {report.get('error_type', 'UnknownError')}: "
            f"{report.get('error', 'mutable smoke test failed')}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
