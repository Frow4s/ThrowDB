from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_load_test.py"
    spec = importlib.util.spec_from_file_location("throwdb_load_test_formats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (b"THROWDBM", "legacy_immutable"),
        (b"TDBNMNF1", "native_mutable"),
    ],
)
def test_store_format_is_detected_from_manifest_magic(
    tmp_path: Path,
    magic: bytes,
    expected: str,
) -> None:
    (tmp_path / "manifest.bin").write_bytes(magic + bytes(64))
    assert BENCHMARK.detect_store_format(tmp_path) == expected


def test_unknown_and_truncated_manifest_magic_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.bin"
    manifest.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        BENCHMARK.detect_store_format(tmp_path)

    manifest.write_bytes(b"UNKNOWN!")
    with pytest.raises(ValueError, match="unsupported storage manifest magic"):
        BENCHMARK.detect_store_format(tmp_path)


@pytest.mark.parametrize(
    ("magic", "expected_api"),
    [(b"TDBNMNF1", "collection"), (b"THROWDBM", "storage")],
)
def test_storage_open_is_routed_with_identical_reader_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    magic: bytes,
    expected_api: str,
) -> None:
    (tmp_path / "manifest.bin").write_bytes(magic)
    calls: list[tuple[str, Path, dict[str, object]]] = []
    native_handle = object()
    legacy_handle = object()

    class FakeCollection:
        @staticmethod
        def open(path: Path, **kwargs: object) -> object:
            calls.append(("collection", path, kwargs))
            return native_handle

    class FakeStorage:
        @staticmethod
        def open(path: Path, **kwargs: object) -> object:
            calls.append(("storage", path, kwargs))
            return legacy_handle

    monkeypatch.setattr(
        BENCHMARK,
        "throwdb",
        SimpleNamespace(Collection=FakeCollection, Storage=FakeStorage),
    )
    tuning = {
        "io_workers": 4,
        "io_queue_capacity": 8,
        "io_ranges_per_task": 16,
        "io_backend": "thread_pool",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 64,
    }

    handle, _ = BENCHMARK.open_benchmark_storage(tmp_path, **tuning)

    assert calls[0][0] == expected_api
    assert calls[0][1] == tmp_path
    if expected_api == "collection":
        assert handle is native_handle
        assert calls[0][2] == {"mode": "r", "tuning": tuning}
    else:
        assert handle is legacy_handle
        assert calls[0][2] == tuning


def test_float32_correctness_and_throughput_use_the_store_dtype(
    tmp_path: Path,
) -> None:
    source = np.arange(48, dtype=np.float32).reshape(12, 4)
    source_path = tmp_path / "embeddings.npy"
    np.save(source_path, source)
    population = np.arange(len(source), dtype=np.uint64)

    class Float32Storage:
        count = len(source)
        dimension = source.shape[1]
        dtype = "float32"

        @staticmethod
        def get(ids: np.ndarray) -> np.ndarray:
            return np.asarray(source[ids], dtype=np.float32, order="C")

        @staticmethod
        def get_into(ids: np.ndarray, output: np.ndarray) -> None:
            assert output.dtype == np.dtype(np.float32)
            output[:] = source[ids]

    correctness = BENCHMARK.correctness_check(
        Float32Storage,
        population,
        source_path,
        "npy",
    )
    result = BENCHMARK.sequential_benchmark(
        Float32Storage,
        population,
        BENCHMARK.SequentialScenario(batch_size=3, calls=2, warmups=1),
        seed=7,
    )

    assert correctness["status"] == "passed"
    assert correctness["dtype"] == "float32"
    useful_bytes = result["useful_gib_per_second"] * result["seconds"] * (1 << 30)
    assert useful_bytes == pytest.approx(2 * 3 * 4 * np.dtype(np.float32).itemsize)


def test_raw_source_memmap_uses_the_requested_dtype(tmp_path: Path) -> None:
    expected = np.arange(15, dtype=np.float32).reshape(5, 3)
    path = tmp_path / "embeddings.raw"
    expected.tofile(path)

    actual = BENCHMARK.open_source_embeddings(path, "raw", 5, 3, "float32")

    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_id_population_keeps_supported_integer_dtype_and_rejects_negatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsigned_path = tmp_path / "unsigned.npy"
    np.save(unsigned_path, np.array([1, 2, 3], dtype=np.uint32))

    loaded = BENCHMARK.load_ids(unsigned_path, expected_count=3)

    assert loaded.dtype == np.dtype(np.uint32)
    assert isinstance(loaded, np.memmap)

    signed_path = tmp_path / "signed.npy"
    np.save(signed_path, np.array([1, 2, 3, -4], dtype=np.int16))
    monkeypatch.setattr(BENCHMARK, "ID_VALIDATION_CHUNK_ROWS", 2)
    with pytest.raises(ValueError, match=r"ids\[3\]=-4"):
        BENCHMARK.load_ids(signed_path, expected_count=4)


def test_cache_eviction_includes_legacy_shards_and_native_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard-000001.dat").write_bytes(bytes(11))
    (data / "segment-00000001.dat").write_bytes(bytes(13))
    (data / "unrelated.dat").write_bytes(bytes(17))
    advised: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(BENCHMARK.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        BENCHMARK.os,
        "posix_fadvise",
        lambda fd, offset, length, advice: advised.append(
            (fd, offset, length, advice)
        ),
        raising=False,
    )

    result = BENCHMARK.evict_data_cache(tmp_path)

    assert result["shard_count"] == 2
    assert result["data_file_count"] == 2
    assert result["advised_bytes"] == 24
    assert len(advised) == 2
