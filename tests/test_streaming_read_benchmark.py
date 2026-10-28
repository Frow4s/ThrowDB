from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import throwdb


def _load_benchmark_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "run_streaming_read_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "throwdb_run_streaming_read_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_parser_progress_defaults_and_disable_switch() -> None:
    default = BENCHMARK._parser().parse_args(
        ["--store", "/store", "--output", "/report.json"]
    )
    assert default.progress_seconds == 5.0
    assert default.no_progress is False

    disabled = BENCHMARK._parser().parse_args(
        ["--store", "/store", "--output", "/report.json", "--no-progress"]
    )
    assert disabled.no_progress is True


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_parser_rejects_invalid_progress_interval(value: str) -> None:
    with pytest.raises(SystemExit):
        BENCHMARK._parser().parse_args(
            [
                "--store",
                "/store",
                "--output",
                "/report.json",
                "--progress-seconds",
                value,
            ]
        )


def test_progress_target_is_bounded_by_source_and_explicit_limit() -> None:
    assert BENCHMARK._progress_target_rows(100, None) == 100
    assert BENCHMARK._progress_target_rows(100, 40) == 40
    assert BENCHMARK._progress_target_rows(100, 100) == 100
    assert BENCHMARK._progress_target_rows(100, 200) == 100
    with pytest.raises(ValueError):
        BENCHMARK._progress_target_rows(-1, None)


def test_progress_reporter_throttles_and_always_reports_final_state() -> None:
    now = [100.0]
    output = io.StringIO()
    reporter = BENCHMARK._ProgressReporter(
        enabled=True,
        interval_seconds=5.0,
        target_rows=100,
        started=100.0,
        sink=output,
        clock=lambda: now[0],
    )

    now[0] = 101.0
    assert reporter.report(
        rows=10,
        batches=1,
        logical_bytes=10 << 20,
        peak_rss_bytes=2 << 30,
    )
    now[0] = 103.0
    assert not reporter.report(
        rows=20,
        batches=2,
        logical_bytes=20 << 20,
        peak_rss_bytes=2 << 30,
    )
    now[0] = 106.0
    assert reporter.report(
        rows=60,
        batches=3,
        logical_bytes=60 << 20,
        peak_rss_bytes=3 << 30,
    )
    assert reporter.report(
        rows=104,
        batches=4,
        logical_bytes=104 << 20,
        peak_rss_bytes=3 << 30,
        force=True,
        status="stopped_at_limit",
    )

    lines = output.getvalue().splitlines()
    assert len(lines) == 3
    assert "10/100 rows (10.00%)" in lines[0]
    assert "ETA 9.0s" in lines[0]
    assert "60/100 rows (60.00%)" in lines[1]
    assert "stopped_at_limit" in lines[2]
    assert "104/100 rows (100.00%)" in lines[2]
    assert "ETA 0.0s" in lines[2]


def test_disabled_progress_reporter_is_silent() -> None:
    output = io.StringIO()
    reporter = BENCHMARK._ProgressReporter(
        enabled=False,
        interval_seconds=5.0,
        target_rows=0,
        started=100.0,
        sink=output,
        clock=lambda: 101.0,
    )
    assert not reporter.report(
        rows=0,
        batches=0,
        logical_bytes=0,
        peak_rss_bytes=None,
        force=True,
        status="complete",
    )
    assert output.getvalue() == ""


def _create_collection(
    path: Path,
    *,
    rows: int = 10,
    dimension: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(1, rows + 1, dtype=np.uint32)
    embeddings = np.arange(
        rows * dimension,
        dtype=np.float16,
    ).reshape(rows, dimension)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(
            id_type="uint64",
            dimension=dimension,
            dtype="float16",
        ),
        segment_size_bytes=1 << 20,
    ) as collection:
        collection.add(ids, embeddings)
    return ids, embeddings


def test_embedded_progress_stays_on_stderr_and_stdout_is_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "store"
    ids, _ = _create_collection(store)
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, ids)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_streaming_read_benchmark.py",
            "--store",
            str(store),
            "--mode",
            "stream-get",
            "--ids",
            str(ids_path),
            "--batch-rows",
            "4",
            "--max-rows",
            "3",
            "--progress-seconds",
            "60",
            "--output",
            str(output),
        ],
    )

    assert BENCHMARK.main() == 0
    captured = capsys.readouterr()
    stdout_result = json.loads(captured.out)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert stdout_result == report["result"]
    assert report["result"]["rows"] == 4
    assert report["result"]["logical_eof"] is False
    assert report["limits"]["effective_stop_rows"] == 3
    assert "opening embedded collection" in captured.err
    assert "progress:" in captured.err
    assert "stopped_at_limit:" in captured.err
    assert "4/3 rows (100.00%)" in captured.err


def test_no_progress_keeps_stderr_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "store"
    _create_collection(store, rows=2)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_streaming_read_benchmark.py",
            "--store",
            str(store),
            "--mode",
            "scan-records",
            "--no-progress",
            "--output",
            str(output),
        ],
    )

    assert BENCHMARK.main() == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert captured.err == ""
