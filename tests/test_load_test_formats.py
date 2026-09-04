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


def test_storage_open_uses_collection_with_explicit_reader_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []
    native_handle = object()

    class FakeCollection:
        @staticmethod
        def open(path: Path, **kwargs: object) -> object:
            calls.append((path, kwargs))
            return native_handle

    monkeypatch.setattr(
        BENCHMARK, "throwdb", SimpleNamespace(Collection=FakeCollection)
    )
    tuning = {
        "io_workers": 4,
        "io_queue_capacity": 8,
        "io_ranges_per_task": 16,
        "io_backend": "thread_pool",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 64,
    }
    handle, store_format = BENCHMARK.open_benchmark_storage(tmp_path, **tuning)
    assert handle is native_handle
    assert store_format == "native_mutable"
    assert calls == [(tmp_path, {"mode": "r", "tuning": tuning})]


def test_optional_storage_open_stats_are_normalized_to_a_dict() -> None:
    stats = {"total_seconds": 1.25, "index_validation_seconds": 0.75}

    assert BENCHMARK.storage_open_stats(SimpleNamespace(open_stats=stats)) == stats
    assert BENCHMARK.storage_open_stats(SimpleNamespace()) is None


def test_native_benchmark_can_require_persisted_reader_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []
    handle = object()

    class FakeCollection:
        @staticmethod
        def open(path: Path, **kwargs: object) -> object:
            calls.append((path, kwargs))
            return handle

    monkeypatch.setattr(
        BENCHMARK,
        "throwdb",
        SimpleNamespace(Collection=FakeCollection),
    )

    actual, store_format = BENCHMARK.open_benchmark_storage(
        tmp_path,
        io_workers=0,
        io_queue_capacity=0,
        io_ranges_per_task=8,
        io_backend="thread_pool",
        io_uring_workers=1,
        io_uring_queue_depth=32,
        tuning_mode="persisted",
    )

    assert actual is handle
    assert store_format == "native_mutable"
    assert calls == [(tmp_path, {"mode": "r", "tuning": "persisted"})]


def test_actual_storage_tuning_is_normalized_to_a_copy() -> None:
    selected = {
        "source": "persisted",
        "status": "selected",
        "io_backend": "io_uring",
        "io_uring_workers": 1,
        "io_uring_queue_depth": 64,
    }

    observed = BENCHMARK.storage_tuning(SimpleNamespace(tuning=selected))

    assert observed == selected
    assert observed is not selected
    assert BENCHMARK.storage_tuning(SimpleNamespace()) is None


def test_benchmark_parser_accepts_strict_persisted_tuning() -> None:
    args = BENCHMARK.build_parser().parse_args(["--tuning", "persisted"])

    assert args.tuning == "persisted"


def test_registry_tuning_rejects_manual_reader_overrides() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        BENCHMARK.main(
            [
                "--tuning",
                "persisted",
                "--io-uring-queue-depth",
                "64",
            ]
        )


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


def test_cache_eviction_includes_base_segments_and_active_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "segments"
    data.mkdir()
    wal = tmp_path / "wal"
    wal.mkdir()
    (data / "segment-00000000.dat").write_bytes(bytes(11))
    (data / "segment-00000001.dat").write_bytes(bytes(13))
    (data / "unrelated.dat").write_bytes(bytes(17))
    (wal / "active.wal").write_bytes(bytes(19))
    (wal / "unrelated.bin").write_bytes(bytes(23))
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

    assert result["data_file_count"] == 2
    assert result["wal_file_count"] == 1
    assert result["advised_file_count"] == 3
    assert result["advised_bytes"] == 43
    assert len(advised) == 3


def test_cache_eviction_includes_partitioned_format7_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = tmp_path / "segments"
    segments.mkdir()
    wal = tmp_path / "wal"
    wal.mkdir()
    (segments / f"segment-{'1' * 32}.dat").write_bytes(bytes(29))
    (segments / f"segment-{'1' * 32}.ids").write_bytes(bytes(31))
    (wal / f"wal-{'2' * 32}.wal").write_bytes(bytes(37))
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

    assert result["data_file_count"] == 1
    assert result["wal_file_count"] == 1
    assert result["advised_file_count"] == 2
    assert result["advised_bytes"] == 66
    assert len(advised) == 2


def test_sequential_warm_only_replays_measured_queries_before_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = np.arange(32, dtype=np.uint64)
    scenario = BENCHMARK.SequentialScenario(batch_size=2, calls=2, warmups=1)
    expected = BENCHMARK.query_matrix(
        population,
        scenario.calls + scenario.warmups,
        scenario.batch_size,
        seed=17,
    )
    events: list[tuple[str, tuple[int, ...] | None]] = []

    class Storage:
        dimension = 3
        dtype = "float16"

        @staticmethod
        def get_into(ids: np.ndarray, output: np.ndarray) -> None:
            events.append(("get", tuple(int(value) for value in ids)))
            output.fill(0)

    original_usage_snapshot = BENCHMARK.usage_snapshot

    def usage_snapshot() -> dict[str, int | float]:
        events.append(("usage", None))
        return original_usage_snapshot()

    monkeypatch.setattr(BENCHMARK, "usage_snapshot", usage_snapshot)

    result = BENCHMARK.sequential_benchmark(
        Storage,
        population,
        scenario,
        seed=17,
        prewarm_measurement=True,
    )

    expected_gets = [tuple(int(value) for value in query) for query in expected]
    assert events[:4] == [
        ("get", expected_gets[0]),
        ("get", expected_gets[1]),
        ("get", expected_gets[2]),
        ("usage", None),
    ]
    assert [value for event, value in events if event == "get"] == [
        expected_gets[0],
        expected_gets[1],
        expected_gets[2],
        expected_gets[1],
        expected_gets[2],
    ]
    assert result["explicit_measurement_prewarm"] is True


def test_concurrent_warm_only_replays_measured_queries_before_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = np.arange(32, dtype=np.uint64)
    expected = BENCHMARK.query_matrix(
        population,
        rows=3,
        batch_size=2,
        seed=23,
    )
    events: list[tuple[str, tuple[int, ...] | None]] = []

    class Storage:
        dimension = 3
        dtype = "float16"

        @staticmethod
        def get_into(ids: np.ndarray, output: np.ndarray) -> None:
            events.append(("get", tuple(int(value) for value in ids)))
            output.fill(0)

    original_usage_snapshot = BENCHMARK.usage_snapshot

    def usage_snapshot() -> dict[str, int | float]:
        events.append(("usage", None))
        return original_usage_snapshot()

    monkeypatch.setattr(BENCHMARK, "usage_snapshot", usage_snapshot)

    result = BENCHMARK.concurrent_benchmark(
        Storage,
        population,
        workers=1,
        batch_size=2,
        calls_per_worker=2,
        seed=23,
        prewarm_measurement=True,
    )

    expected_gets = [tuple(int(value) for value in query) for query in expected]
    assert events[:4] == [
        ("get", expected_gets[0]),
        ("get", expected_gets[1]),
        ("get", expected_gets[2]),
        ("usage", None),
    ]
    assert [value for event, value in events if event == "get"] == [
        expected_gets[0],
        expected_gets[1],
        expected_gets[2],
        expected_gets[1],
        expected_gets[2],
    ]
    assert result["explicit_measurement_prewarm"] is True


def test_diskstats_summary_reports_current_ios_in_progress_not_delta() -> None:
    fields = {
        "reads_completed": 10,
        "reads_merged": 0,
        "sectors_read": 100,
        "read_time_ms": 20,
        "writes_completed": 0,
        "writes_merged": 0,
        "sectors_written": 0,
        "write_time_ms": 0,
        "ios_in_progress": 5,
        "io_time_ms": 20,
        "weighted_io_time_ms": 20,
    }
    monitor = BENCHMARK.ProcDiskstatsMonitor(Path("unused"))
    monitor._first = {"nvme0n1": fields}
    monitor._last = {
        "nvme0n1": {
            **fields,
            "reads_completed": 11,
            "sectors_read": 108,
            "ios_in_progress": 2,
        }
    }
    monitor._started = 1.0
    monitor._finished = 2.0

    summary = monitor.summary()

    assert summary["active_devices"]["nvme0n1"]["ios_in_progress"] == 2
