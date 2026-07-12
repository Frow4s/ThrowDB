from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

import throwdb


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_mutable_test.py"
    spec = importlib.util.spec_from_file_location("throwdb_run_mutable_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _vectors(rows: int, dimension: int) -> np.ndarray:
    bits = np.arange(rows * dimension, dtype=np.uint16).reshape(rows, dimension)
    bits = bits * np.uint16(7_919) + np.uint16(31)
    return bits.view(np.float16)


def test_mutable_smoke_test_restores_the_collection_and_writes_report(
    tmp_path: Path,
    capsys: object,
) -> None:
    store = tmp_path / "store"
    ids_path = tmp_path / "ids.npy"
    embeddings_path = tmp_path / "embeddings.npy"
    results_path = tmp_path / "results" / "mutable.json"
    ids = np.array(
        [0, 17, (1 << 32) + 9, (1 << 63) + 7, (1 << 64) - 2, (1 << 64) - 1],
        dtype=np.uint64,
    )
    embeddings = _vectors(ids.size, 7)
    np.save(ids_path, ids)
    np.save(embeddings_path, embeddings)

    with throwdb.Collection.create(
        store,
        schema=throwdb.CollectionSchema(dimension=7, dtype="float16"),
        shard_size_bytes=3 * 7 * np.dtype(np.float16).itemsize,
    ) as builder:
        builder.add(ids[:2], embeddings[:2])
        builder.add(ids[2:], embeddings[2:])

    exit_code = BENCHMARK.main(
        [
            "--store",
            str(store),
            "--ids",
            str(ids_path),
            "--source-embeddings",
            str(embeddings_path),
            "--sample-rows",
            "4",
            "--seed",
            "91",
            "--results",
            str(results_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(results_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["cleanup"]["status"] == "verified"
    assert report["cleanup"]["synthetic_delete_results"] == [False, True, False]
    assert report["checks"] == {
        "base_read_bit_exact": True,
        "crud_counts": True,
        "insert_collision_atomic": True,
        "insert_collision_rejected": True,
        "prefix_dimensions": 7,
        "prefix_read_bit_exact": True,
        "read_only_rejected_mutation": True,
        "wal_recovery_bit_exact": True,
        "writer_lease": True,
        "writer_lease_error_type": report["checks"]["writer_lease_error_type"],
    }
    assert report["sample_rows"] == 4
    assert set(report["timings"]) >= {
        "base_read",
        "prefix_read",
        "update",
        "insert",
        "upsert",
        "delete",
        "recovery_open",
        "cleanup_reopen_verify",
    }

    with throwdb.Collection.open(store) as collection:
        assert collection.count == ids.size
        actual = collection.get(ids)
        assert actual.tobytes(order="C") == embeddings.tobytes(order="C")
        for identifier in report["synthetic_ids"]:
            try:
                collection.get(np.array([identifier], dtype=np.uint64))
            except KeyError:
                pass
            else:
                raise AssertionError(f"synthetic ID {identifier} survived cleanup")

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "durable WAL mutations" in captured.err
    assert "status=passed cleanup=verified" in captured.out

    sampled = BENCHMARK._sample_row_numbers(600_000_000, 64, 91)
    assert sampled.shape == (64,)
    assert np.unique(sampled).size == 64
