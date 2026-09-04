from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import throwdb
import throwdb.workload as workload_module
from throwdb.cli import main
from throwdb._throwdb import NativeCollection
from throwdb.workload import (
    WorkloadLogError,
    _crc32c,
    _v2_fnv_update,
    inspect_workload_logs,
)


HEADER_BYTES = 4096
FRAME_BYTES = 256


def _put_u32(output: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", output, offset, value)


def _put_u64(output: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", output, offset, value)


def _finish_crc(output: bytearray) -> bytes:
    _put_u32(output, 24, _crc32c(memoryview(output), zero_offset=24, zero_size=4))
    return bytes(output)


def _header(
    *,
    collection_low: int = 11,
    collection_high: int = 22,
    instance_low: int = 33,
    instance_high: int = 44,
    created_unix_ns: int = 1_700_000_000_000_000_000,
    segment_sequence: int = 0,
    max_write_bytes_per_second: int = 16 << 20,
    max_retained_files: int = 4096,
    rotate_seconds: int = 3600,
    sample_ppm: int = 10_000,
    max_ids_per_event: int = 4096,
    used_bytes: int = 176,
) -> bytes:
    output = bytearray(HEADER_BYTES)
    output[:8] = b"TDBRLOG1"
    _put_u32(output, 8, 1)
    _put_u32(output, 12, HEADER_BYTES)
    _put_u32(output, 16, used_bytes)
    _put_u64(output, 32, collection_low)
    _put_u64(output, 40, collection_high)
    _put_u32(output, 48, 7)
    _put_u32(output, 52, 1)
    _put_u32(output, 56, 1024)
    _put_u32(output, 60, 2048)
    _put_u64(output, 64, instance_low)
    _put_u64(output, 72, instance_high)
    _put_u64(output, 80, created_unix_ns)
    _put_u32(output, 88, sample_ppm)
    _put_u32(output, 92, max_ids_per_event)
    _put_u64(output, 96, 64 << 20)
    _put_u64(output, 104, 256 << 20)
    _put_u64(output, 112, rotate_seconds)
    _put_u64(output, 120, 10 << 30)
    _put_u64(output, 128, 7 * 24 * 60 * 60)
    _put_u64(output, 136, 2 << 30)
    _put_u64(output, 144, 1234)
    _put_u64(output, 152, segment_sequence)
    _put_u64(output, 160, max_write_bytes_per_second)
    _put_u64(output, 168, max_retained_files)
    return _finish_crc(output)


def _event(
    ids: list[int],
    *,
    original: int | None = None,
    method: int = 1,
    prefix: int = 1024,
    sequence: int = 1,
    counter_sampled: int | None = None,
    sample_ppm: int = 10_000,
) -> bytes:
    original = len(ids) if original is None else original
    truncated = len(ids) < original
    output = bytearray(FRAME_BYTES + len(ids) * 8)
    output[:8] = b"TDBRREC1"
    _put_u32(output, 8, 1)
    _put_u32(output, 12, FRAME_BYTES)
    _put_u32(output, 16, len(output))
    _put_u32(output, 20, 1)
    _put_u32(output, 28, method)
    _put_u32(
        output,
        32,
        (1 if method == 2 else 0) | (2 if truncated else 0),
    )
    _put_u32(output, 36, prefix)
    _put_u64(output, 40, sequence)
    _put_u64(output, 48, 1_700_000_000_000_000_000 + sequence)
    _put_u64(output, 56, 7 if method == 2 else 0)
    _put_u64(output, 64, 9 if method == 2 else 0)
    _put_u64(output, 72, 1_000_000)
    _put_u64(output, 80, 100 if method == 2 else 0)
    _put_u64(output, 88, 200 if method == 2 else 0)
    _put_u64(output, 96, sequence - 1 if method == 2 else 0)
    _put_u64(output, 104, original)
    _put_u32(output, 112, len(ids))
    _put_u32(output, 116, sample_ppm)
    _put_u64(output, 120, 0xABC000 + sequence)
    counter_sampled = (
        sequence if counter_sampled is None else counter_sampled
    )
    counters = (
        counter_sampled,
        original * counter_sampled,
        0,
        counter_sampled,
        counter_sampled,
        max(0, counter_sampled - 1),
        len(ids) * counter_sampled,
        1 if truncated else 0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    )
    for index, value in enumerate(counters):
        _put_u64(output, 128 + index * 8, value)
    for index, value in enumerate(ids):
        _put_u64(output, FRAME_BYTES + index * 8, value)
    return _finish_crc(output)


def _checkpoint(
    *,
    observed: int = 200,
    sampled: int = 2,
    dropped_busy: int = 0,
    dropped_full: int = 0,
    dropped_runtime: int = 0,
    dropped_minfree: int = 0,
    dropped_io_budget: int = 0,
    enqueued: int | None = None,
    written: int | None = None,
    captured_ids: int | None = None,
    sampled_ids: int | None = None,
    enqueued_ids: int | None = None,
    written_ids: int | None = None,
    dropped_busy_ids: int = 0,
    dropped_full_ids: int = 0,
    dropped_runtime_ids: int = 0,
    dropped_minfree_ids: int = 0,
    dropped_io_budget_ids: int = 0,
    extension_version: int = 2,
    opening: bool = False,
) -> bytes:
    output = bytearray(FRAME_BYTES)
    output[:8] = b"TDBRREC1"
    _put_u32(output, 8, 1)
    _put_u32(output, 12, FRAME_BYTES)
    _put_u32(output, 16, FRAME_BYTES)
    _put_u32(output, 20, 2)
    _put_u32(output, 28, extension_version)
    _put_u32(output, 32, 1 if opening else 0)
    _put_u64(
        output,
        48,
        1_700_000_000_100_000_000 + min(observed, 1_000_000_000),
    )
    admission_event_drops = (
        dropped_busy
        + dropped_full
        + dropped_runtime
        + dropped_minfree
    )
    enqueued = (
        sampled - admission_event_drops
        if enqueued is None
        else enqueued
    )
    written = (
        enqueued - dropped_io_budget
        if written is None
        else written
    )
    admission_row_drops = (
        dropped_busy_ids
        + dropped_full_ids
        + dropped_runtime_ids
        + dropped_minfree_ids
    )
    if sampled_ids is None:
        sampled_ids = (
            captured_ids
            if captured_ids is not None
            else (max(5, sampled * 4096) if sampled else 0)
        )
    if written_ids is None:
        written_ids = (
            sampled_ids
            - admission_row_drops
            - dropped_io_budget_ids
        )
    if enqueued_ids is None:
        enqueued_ids = sampled_ids - admission_row_drops
    captured_ids = (
        enqueued_ids
        if captured_ids is None
        else captured_ids
    )
    extension = (
        sampled_ids,
        enqueued_ids,
        written_ids,
        dropped_busy_ids,
        dropped_full_ids,
        dropped_runtime_ids,
        dropped_minfree_ids,
        dropped_io_budget,
        dropped_io_budget_ids,
    )
    for index, value in enumerate(extension):
        _put_u64(output, 56 + index * 8, value)
    values = (
        observed,
        (
            max(20_000, captured_ids, sampled_ids)
            if observed or captured_ids or sampled_ids
            else 0
        ),
        observed - sampled,
        sampled,
        enqueued,
        written,
        captured_ids,
        1 if enqueued else 0,
        dropped_busy,
        dropped_full,
        dropped_runtime,
        dropped_minfree,
        0,
        HEADER_BYTES + FRAME_BYTES if observed or sampled else 0,
        1 if observed or sampled else 0,
        0,
    )
    for index, value in enumerate(values):
        _put_u64(output, 128 + index * 8, value)
    return _finish_crc(output)


def _opening_checkpoint(**overrides: object) -> bytes:
    values: dict[str, object] = {
        "observed": 0,
        "sampled": 0,
        "opening": True,
    }
    values.update(overrides)
    return _checkpoint(**values)  # type: ignore[arg-type]


def _write_log(
    path: Path,
    *,
    header: bytes | None = None,
    frames: tuple[bytes, ...] = (),
    opening: bytes | None = None,
    prepend_opening: bool = True,
) -> None:
    prefix = (
        (opening if opening is not None else _opening_checkpoint())
        if prepend_opening
        else b""
    )
    path.write_bytes((header or _header()) + prefix + b"".join(frames))


def test_inspector_reports_quality_without_exposing_ids(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "sample.tdbrl",
        frames=(
            _event([101, 202, 303], sequence=1),
            _event(
                [404, 505],
                original=10,
                method=2,
                prefix=128,
                sequence=2,
            ),
            _checkpoint(),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is True
    assert report["events"] == {
        "records": 2,
        "empty_events": 0,
        "methods": {"get": 1, "stream_get": 1},
        "prefix_dimensions": {"128": 1, "1024": 1},
        "captured_ids": 5,
        "original_ids": 13,
        "captured_fraction": 5 / 13,
        "truncated_events": 1,
        "first_unix_ns": 1_700_000_000_000_000_001,
        "last_unix_ns": 1_700_000_000_000_000_002,
    }
    assert report["counters"]["drop_ratio"] == 0
    assert report["configuration"]["max_write_bytes_per_second"] == 16 << 20
    assert report["configuration"]["max_retained_files"] == 4096
    serialized = json.dumps(report, sort_keys=True)
    for identifier in ("101", "202", "303", "404", "505"):
        assert identifier not in serialized


def test_open_torn_tail_is_recoverable_but_finalized_tail_is_corrupt(
    tmp_path: Path,
) -> None:
    open_logs = tmp_path / "open"
    open_logs.mkdir()
    valid = _event([1, 2, 3])
    _write_log(
        open_logs / "active.open",
        frames=(valid, _event([4, 5])[:100]),
    )

    open_report = inspect_workload_logs(open_logs)

    assert open_report["files"]["valid"] == 1
    assert open_report["files"]["torn_open"] == 1
    assert open_report["events"]["records"] == 1
    assert open_report["counters"]["sampled_events"] == 1
    assert open_report["counters"]["incomplete_instances"] == 1
    assert open_report["quality"]["suitable"] is False
    assert (
        "incomplete_counter_coverage"
        in open_report["quality"]["reasons"]
    )
    assert "open_file_has_torn_tail" in open_report["quality"]["warnings"]

    finalized_logs = tmp_path / "finalized"
    finalized_logs.mkdir()
    _write_log(
        finalized_logs / "broken.tdbrl",
        frames=(valid, _event([4, 5])[:100]),
    )
    finalized_report = inspect_workload_logs(finalized_logs)

    assert finalized_report["files"]["invalid_finalized"] == 1
    assert finalized_report["quality"]["suitable"] is False
    assert "corrupt_finalized_files" in finalized_report["quality"]["reasons"]


def test_inspector_rejects_mixed_identity_and_excessive_drop_ratio(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "one.tdbrl",
        frames=(_event([1]), _checkpoint(sampled=10, dropped_full=1)),
    )
    _write_log(
        logs / "two.tdbrl",
        header=_header(
            collection_low=99,
            instance_low=55,
            instance_high=66,
        ),
        frames=(_event([2]), _checkpoint()),
    )

    report = inspect_workload_logs(logs, max_drop_ratio=0.01)

    assert report["quality"]["suitable"] is False
    assert report["quality"]["reasons"] == [
        "drop_ratio_exceeds_limit",
        "mixed_collection_or_schema_identity",
    ]


def test_inspector_rejects_mixed_logging_configuration(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "one.tdbrl",
        header=_header(
            instance_low=33,
            instance_high=44,
            sample_ppm=10_000,
        ),
        frames=(
            _event([1], sample_ppm=10_000),
            _checkpoint(),
        ),
    )
    _write_log(
        logs / "two.tdbrl",
        header=_header(
            instance_low=55,
            instance_high=66,
            sample_ppm=20_000,
        ),
        frames=(
            _event([2], sample_ppm=20_000),
            _checkpoint(),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["configuration_variants"] == 2
    assert report["quality"]["suitable"] is False
    assert report["quality"]["reasons"] == [
        "mixed_logging_configuration"
    ]


def test_cli_writes_report_before_enforcing_quality_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "bad.tdbrl",
        frames=(_event([1]), _checkpoint(sampled=10, dropped_full=5)),
    )
    output = tmp_path / "report.json"

    with pytest.raises(SystemExit):
        main(
            [
                "workload",
                "inspect",
                "--logs",
                str(logs),
                "--max-drop-ratio",
                "0.01",
                "--max-row-drop-ratio",
                "0.02",
                "--max-files",
                "2",
                "--max-total-bytes",
                "100000",
                "--max-file-bytes",
                "100000",
                "--max-report-entries",
                "0",
                "--max-prefix-entries",
                "1",
                "--output",
                str(output),
                "--require-suitable",
            ]
        )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["quality"]["suitable"] is False
    assert "drop_ratio_exceeds_limit" in saved["quality"]["reasons"]
    assert saved["quality"]["max_row_drop_ratio"] == 0.02
    assert saved["limits"] == {
        "max_files": 2,
        "max_total_bytes": 100000,
        "max_file_bytes": 100000,
        "max_report_entries": 0,
        "max_prefix_entries": 1,
    }
    assert saved["files"]["reports"] == []
    assert saved["files"]["reports_omitted"] == 1
    assert "workload logs did not pass" in capsys.readouterr().err


def test_finalized_log_without_terminal_checkpoint_is_incomplete(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    _write_log(incomplete / "missing-checkpoint.tdbrl", frames=(_event([1]),))

    incomplete_report = inspect_workload_logs(incomplete)

    assert incomplete_report["files"]["invalid_finalized"] == 0
    assert incomplete_report["files"]["valid"] == 1
    assert incomplete_report["counters"]["incomplete_instances"] == 1
    assert (
        "incomplete_counter_coverage"
        in incomplete_report["quality"]["reasons"]
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    _write_log(
        empty / "empty-get.tdbrl",
        frames=(
            _event([]),
            _checkpoint(observed=1, sampled=1, captured_ids=0),
        ),
    )

    empty_report = inspect_workload_logs(empty)

    assert empty_report["files"]["invalid_finalized"] == 1
    assert empty_report["events"]["records"] == 0
    assert empty_report["events"]["empty_events"] == 0
    assert empty_report["events"]["captured_ids"] == 0
    assert "corrupt_finalized_files" in empty_report["quality"]["reasons"]


def test_open_mid_file_corruption_is_not_treated_as_a_torn_tail(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    corrupt = bytearray(_event([4, 5], sequence=2))
    corrupt[-1] ^= 0xFF
    _write_log(
        logs / "corrupt.open",
        frames=(
            _event([1, 2, 3], sequence=1),
            bytes(corrupt),
            _event([6], sequence=3),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["invalid_open"] == 1
    assert report["files"]["torn_open"] == 0
    assert report["quality"]["suitable"] is False
    assert "corrupt_open_files" in report["quality"]["reasons"]
    assert "open_file_has_torn_tail" not in report["quality"]["warnings"]


def test_inspector_enforces_all_hard_input_and_report_bounds(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    first = logs / "one.tdbrl"
    second = logs / "two.tdbrl"
    _write_log(first, frames=(_event([1]), _checkpoint()))
    _write_log(
        second,
        header=_header(
            instance_low=55,
            instance_high=66,
            created_unix_ns=1_700_000_000_000_000_001,
        ),
        frames=(_event([2]), _checkpoint()),
    )

    with pytest.raises(WorkloadLogError, match="max_files"):
        inspect_workload_logs(logs, max_files=1)
    with pytest.raises(WorkloadLogError, match="max_total_bytes"):
        inspect_workload_logs(
            logs,
            max_total_bytes=first.stat().st_size + second.stat().st_size - 1,
        )
    with pytest.raises(WorkloadLogError, match="max_file_bytes"):
        inspect_workload_logs(logs, max_file_bytes=first.stat().st_size - 1)

    report = inspect_workload_logs(logs, max_report_entries=1)

    assert report["quality"]["suitable"] is True
    assert report["files"]["discovered"] == 2
    assert len(report["files"]["reports"]) == 1
    assert report["files"]["reports_omitted"] == 1
    assert report["limits"]["max_report_entries"] == 1


def test_inspector_bounds_all_directory_entries(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(18):
        (logs / f"unrelated-{index}").write_bytes(b"")

    with pytest.raises(WorkloadLogError, match="directory entry count"):
        inspect_workload_logs(logs, max_files=1)


@pytest.mark.parametrize(
    "header",
    (
        _header(used_bytes=177),
        _header(rotate_seconds=1 << 63),
        _header(max_retained_files=65_534),
    ),
)
def test_inspector_rejects_headers_current_producer_cannot_write(
    tmp_path: Path,
    header: bytes,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "impossible.tdbrl",
        header=header,
        frames=(_event([1]), _checkpoint()),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in report["quality"]["reasons"]


def test_inspector_rejects_impossible_event_and_writer_error_states(
    tmp_path: Path,
) -> None:
    cases: list[tuple[str, bytearray, int, int]] = []
    observed_too_small = bytearray(_event([1, 2, 3]))
    cases.append(("observed", observed_too_small, 136, 2))
    missing_truncation_counter = bytearray(
        _event([1], original=10)
    )
    cases.append(("truncated", missing_truncation_counter, 184, 0))
    writer_error = bytearray(_checkpoint())
    cases.append(("writer-error", writer_error, 224, 1))

    for label, frame, offset, value in cases:
        _put_u64(frame, offset, value)
        finalized = tmp_path / label
        finalized.mkdir()
        _write_log(
            finalized / "impossible.tdbrl",
            frames=(
                _finish_crc(frame),
                *(
                    ()
                    if label == "writer-error"
                    else (_checkpoint(),)
                ),
            ),
        )
        report = inspect_workload_logs(finalized)
        assert report["files"]["invalid_finalized"] == 1
        assert "corrupt_finalized_files" in report["quality"]["reasons"]


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("max_files", 0),
        ("max_files", 65_537),
        ("max_total_bytes", -1),
        ("max_file_bytes", False),
        ("max_report_entries", -1),
        ("max_prefix_entries", 0),
        ("max_active_requests", 0),
        ("max_active_requests", 65_537),
    ),
)
def test_inspector_rejects_invalid_hard_bounds(
    tmp_path: Path,
    argument: str,
    value: object,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()

    with pytest.raises(WorkloadLogError, match=argument):
        inspect_workload_logs(logs, **{argument: value})  # type: ignore[arg-type]


def test_inspector_rejects_duplicate_logical_segments_without_double_counting(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    frames = (_event([101, 202]), _checkpoint())
    _write_log(logs / "one.tdbrl", frames=frames)
    _write_log(
        logs / "copied.tdbrl",
        header=_header(
            created_unix_ns=1_800_000_000_000_000_000,
        ),
        frames=frames,
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["logical_duplicates"] == 1
    assert report["events"]["records"] == 1
    assert report["events"]["captured_ids"] == 2
    assert report["quality"]["suitable"] is False
    assert "duplicate_logical_files" in report["quality"]["reasons"]


def test_inspector_detects_counter_regression_between_segments(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "older.tdbrl",
        header=_header(created_unix_ns=1_700_000_000_000_000_000),
        frames=(
            _event([1], sequence=1),
            _checkpoint(observed=200, sampled=2),
        ),
    )
    _write_log(
        logs / "newer.tdbrl",
        header=_header(
            created_unix_ns=1_699_999_999_999_999_999,
            segment_sequence=1,
        ),
        frames=(
            _event([2], sequence=1),
            _checkpoint(observed=100, sampled=1),
            _checkpoint(observed=200, sampled=2),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["counters"]["regressed_instances"] == 1
    assert report["quality"]["suitable"] is False
    assert "counter_regression" in report["quality"]["reasons"]


def test_inspector_rejects_counter_invariants_and_invalid_prefix(
    tmp_path: Path,
) -> None:
    counters = tmp_path / "counters"
    counters.mkdir()
    _write_log(
        counters / "bad.tdbrl",
        frames=(
            _event([1]),
            _checkpoint(sampled=2, written=3),
        ),
    )

    counter_report = inspect_workload_logs(counters)

    assert counter_report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in counter_report["quality"]["reasons"]

    prefix = tmp_path / "prefix"
    prefix.mkdir()
    _write_log(
        prefix / "bad.tdbrl",
        frames=(
            _event([1], prefix=1025),
            _checkpoint(),
        ),
    )

    prefix_report = inspect_workload_logs(prefix)

    assert prefix_report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in prefix_report["quality"]["reasons"]


def test_storage_identity_native_failure_is_wrapped(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(logs / "valid.tdbrl", frames=(_event([1]), _checkpoint()))

    with pytest.raises(
        WorkloadLogError,
        match="cannot inspect workload-log storage identity",
    ):
        inspect_workload_logs(logs, storage=tmp_path / "missing-store")


def test_storage_identity_matches_real_collection_and_rejects_another(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    with throwdb.Collection.create(
        store,
        schema=throwdb.CollectionSchema(
            dimension=1024,
            dtype="float16",
        ),
        segment_size_bytes=1 << 20,
        format_version=7,
    ):
        pass
    metadata = dict(NativeCollection.inspect_metadata(store))
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "valid.tdbrl",
        header=_header(
            collection_low=int(metadata["collection_id_low"]),
            collection_high=int(metadata["collection_id_high"]),
        ),
        frames=(_event([1]), _checkpoint()),
    )

    matching = inspect_workload_logs(logs, storage=store)

    assert matching["quality"]["suitable"] is True
    assert "storage_identity_mismatch" not in matching["quality"]["reasons"]

    other_store = tmp_path / "other-store"
    with throwdb.Collection.create(
        other_store,
        schema=throwdb.CollectionSchema(
            dimension=1024,
            dtype="float16",
        ),
        segment_size_bytes=1 << 20,
        format_version=7,
    ):
        pass

    mismatched = inspect_workload_logs(logs, storage=other_store)

    assert mismatched["quality"]["suitable"] is False
    assert "storage_identity_mismatch" in mismatched["quality"]["reasons"]


def test_event_and_row_drop_quality_gates_are_independent(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "drops.tdbrl",
        frames=(
            _event([1]),
            _checkpoint(
                sampled=100,
                dropped_io_budget=10,
                sampled_ids=1_000,
                dropped_io_budget_ids=200,
            ),
        ),
    )

    report = inspect_workload_logs(
        logs,
        max_drop_ratio=0.15,
        max_row_drop_ratio=0.15,
    )

    assert report["files"]["invalid_finalized"] == 0
    assert report["counters"]["dropped_events"] == 10
    assert report["counters"]["event_drop_ratio"] == 0.1
    assert report["counters"]["drop_ratio"] == 0.1
    assert report["counters"]["dropped_ids"] == 200
    assert report["counters"]["row_drop_ratio"] == 0.2
    assert report["quality"]["reasons"] == [
        "row_drop_ratio_exceeds_limit"
    ]


def test_rotated_partition_is_allowed_but_latest_must_be_complete(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "segment-0.tdbrl",
        header=_header(segment_sequence=0),
        frames=(
            _event([1]),
            _checkpoint(sampled=10, written=9),
        ),
    )
    _write_log(
        logs / "segment-1.tdbrl",
        header=_header(
            segment_sequence=1,
            created_unix_ns=1_700_000_000_000_000_001,
        ),
        opening=_checkpoint(
            sampled=10,
            written=9,
            opening=True,
        ),
        frames=(
            _event([2]),
            _checkpoint(sampled=10),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is True
    assert report["counters"]["complete_instances"] == 1
    assert all(
        item["fresh_terminal_checkpoint"]
        for item in report["files"]["reports"]
    )

    latest_incomplete = tmp_path / "latest-incomplete"
    latest_incomplete.mkdir()
    _write_log(
        latest_incomplete / "segment-0.tdbrl",
        frames=(
            _event([1]),
            _checkpoint(sampled=10, written=9),
        ),
    )

    incomplete_report = inspect_workload_logs(latest_incomplete)

    assert incomplete_report["files"]["invalid_finalized"] == 0
    assert incomplete_report["files"]["valid"] == 1
    assert incomplete_report["files"]["reports"][0][
        "fresh_terminal_checkpoint"
    ] is True
    assert incomplete_report["counters"]["incomplete_instances"] == 1
    assert (
        "incomplete_counter_coverage"
        in incomplete_report["quality"]["reasons"]
    )


def test_stale_terminal_checkpoint_is_incomplete_not_corrupt(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "stale.tdbrl",
        frames=(
            _event([1], counter_sampled=10),
            _checkpoint(sampled=5),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["invalid_finalized"] == 0
    assert report["files"]["valid"] == 1
    assert report["files"]["reports"][0]["terminal_checkpoint"] is True
    assert report["files"]["reports"][0][
        "fresh_terminal_checkpoint"
    ] is False
    assert report["files"]["reports"][0]["counter_coverage"] == "lower_bound"
    assert "incomplete_counter_coverage" in report["quality"]["reasons"]


def test_open_file_never_has_complete_counter_coverage(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "active.open",
        frames=(_event([1]), _checkpoint()),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["invalid_open"] == 0
    assert report["files"]["reports"][0]["terminal_checkpoint"] is True
    assert report["files"]["reports"][0][
        "fresh_terminal_checkpoint"
    ] is False
    assert report["counters"]["incomplete_instances"] == 1
    assert "incomplete_counter_coverage" in report["quality"]["reasons"]


def test_event_counter_snapshots_may_regress_physically(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "events.tdbrl",
        frames=(
            _event([1], sequence=1, counter_sampled=10),
            _event([2], sequence=2, counter_sampled=2),
            _checkpoint(sampled=10),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is True
    assert report["events"]["records"] == 2
    assert report["counters"]["sampled_events"] == 10
    assert report["counters"]["regressed_instances"] == 0


def test_checkpoint_extension_and_header_resource_fields_are_validated(
    tmp_path: Path,
) -> None:
    bad_extension = tmp_path / "bad-extension"
    bad_extension.mkdir()
    _write_log(
        bad_extension / "bad.tdbrl",
        frames=(
            _event([1]),
            _checkpoint(extension_version=1),
        ),
    )
    extension_report = inspect_workload_logs(bad_extension)
    assert extension_report["files"]["invalid_finalized"] == 1

    old_header = tmp_path / "old-header"
    old_header.mkdir()
    _write_log(
        old_header / "bad.tdbrl",
        header=_header(used_bytes=160),
        frames=(_event([1]), _checkpoint()),
    )
    header_report = inspect_workload_logs(old_header)
    assert header_report["files"]["invalid_finalized"] == 1

    bad_write_cap = tmp_path / "bad-write-cap"
    bad_write_cap.mkdir()
    _write_log(
        bad_write_cap / "bad.tdbrl",
        header=_header(max_write_bytes_per_second=1),
        frames=(_event([1]), _checkpoint()),
    )
    write_cap_report = inspect_workload_logs(bad_write_cap)
    assert write_cap_report["files"]["invalid_finalized"] == 1

    bad_file_cap = tmp_path / "bad-file-cap"
    bad_file_cap.mkdir()
    _write_log(
        bad_file_cap / "bad.tdbrl",
        header=_header(max_retained_files=0),
        frames=(_event([1]), _checkpoint()),
    )
    file_cap_report = inspect_workload_logs(bad_file_cap)
    assert file_cap_report["files"]["invalid_finalized"] == 1


def test_every_segment_requires_one_opening_counter_checkpoint(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-opening"
    missing.mkdir()
    _write_log(
        missing / "bad.tdbrl",
        frames=(_event([1]), _checkpoint()),
        prepend_opening=False,
    )

    missing_report = inspect_workload_logs(missing)

    assert missing_report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in missing_report["quality"]["reasons"]

    repeated = tmp_path / "repeated-opening"
    repeated.mkdir()
    _write_log(
        repeated / "bad.tdbrl",
        frames=(
            _event([1]),
            _opening_checkpoint(),
            _checkpoint(),
        ),
    )

    repeated_report = inspect_workload_logs(repeated)

    assert repeated_report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in repeated_report["quality"]["reasons"]


def test_drop_ratios_cover_only_the_retained_counter_window(
    tmp_path: Path,
) -> None:
    old_drops = tmp_path / "old-drops"
    old_drops.mkdir()
    _write_log(
        old_drops / "segment-7.tdbrl",
        header=_header(segment_sequence=7),
        opening=_checkpoint(
            observed=1_000,
            sampled=1_000,
            dropped_full=900,
            opening=True,
        ),
        frames=(
            _event([1]),
            _checkpoint(
                observed=1_100,
                sampled=1_100,
                dropped_full=900,
            ),
        ),
    )

    old_drop_report = inspect_workload_logs(
        old_drops,
        max_drop_ratio=0.01,
    )

    assert old_drop_report["quality"]["suitable"] is True
    assert old_drop_report["counters"]["sampled_events"] == 100
    assert old_drop_report["counters"]["dropped_events"] == 0
    assert old_drop_report["counters"]["event_drop_ratio"] == 0
    assert old_drop_report["files"]["reports"][0][
        "opening_checkpoint"
    ] is True

    new_drops = tmp_path / "new-drops"
    new_drops.mkdir()
    _write_log(
        new_drops / "segment-9.tdbrl",
        header=_header(segment_sequence=9),
        opening=_checkpoint(
            observed=10,
            sampled=10,
            enqueued=9,
            written=8,
            sampled_ids=100,
            enqueued_ids=90,
            written_ids=80,
            opening=True,
        ),
        frames=(
            _event([1]),
            _checkpoint(
                observed=10,
                sampled=10,
                dropped_full=1,
                sampled_ids=100,
                enqueued_ids=90,
                written_ids=90,
                dropped_full_ids=10,
            ),
        ),
    )

    new_drop_report = inspect_workload_logs(
        new_drops,
        max_drop_ratio=0.4,
        max_row_drop_ratio=1.0,
    )

    assert new_drop_report["quality"]["suitable"] is False
    assert new_drop_report["counters"]["sampled_events"] == 0
    assert new_drop_report["counters"]["dropped_events"] == 1
    assert new_drop_report["counters"]["completed_event_outcomes"] == 2
    assert new_drop_report["counters"]["event_drop_ratio"] == 0.5
    assert "drop_ratio_exceeds_limit" in new_drop_report["quality"]["reasons"]


def test_saturated_retained_window_counters_fail_closed(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    maximum = (1 << 64) - 1
    saturated = {
        "observed": maximum,
        "sampled": maximum,
        "enqueued": maximum,
        "written": maximum,
        "captured_ids": maximum,
        "sampled_ids": maximum,
        "enqueued_ids": maximum,
        "written_ids": maximum,
    }
    _write_log(
        logs / "saturated.tdbrl",
        opening=_checkpoint(opening=True, **saturated),
        frames=(
            _event([1]),
            _checkpoint(**saturated),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is False
    assert report["counters"]["saturated_instances"] == 1
    assert "counter_saturation" in report["quality"]["reasons"]


def test_segment_sequence_gaps_are_unsuitable(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "segment-0.tdbrl",
        header=_header(segment_sequence=0),
        frames=(_event([1]), _checkpoint()),
    )
    _write_log(
        logs / "segment-2.tdbrl",
        header=_header(
            segment_sequence=2,
            created_unix_ns=1_700_000_000_000_000_002,
        ),
        frames=(_event([2]), _checkpoint()),
    )

    report = inspect_workload_logs(logs)

    assert report["counters"]["sequence_gap_instances"] == 1
    assert report["counters"]["incomplete_instances"] == 1
    assert "segment_sequence_gap" in report["quality"]["reasons"]


def test_distinct_prefix_dimensions_have_a_hard_bound(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "one.tdbrl",
        frames=(_event([1], prefix=128), _checkpoint()),
    )
    _write_log(
        logs / "two.tdbrl",
        header=_header(
            instance_low=55,
            instance_high=66,
            created_unix_ns=1_700_000_000_000_000_001,
        ),
        frames=(_event([2], prefix=256), _checkpoint()),
    )

    with pytest.raises(WorkloadLogError, match="max_prefix_entries"):
        inspect_workload_logs(logs, max_prefix_entries=1)


@pytest.mark.parametrize("argument", ("max_drop_ratio", "max_row_drop_ratio"))
@pytest.mark.parametrize("value", (-0.01, 1.01, float("nan")))
def test_inspector_rejects_invalid_drop_ratio_bounds(
    tmp_path: Path,
    argument: str,
    value: float,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()

    with pytest.raises(WorkloadLogError, match=argument):
        inspect_workload_logs(logs, **{argument: value})


V2_TOKEN = (0xA11CE, 0xB0B)
V2_UNIX_NS = 1_800_000_000_000_000_000
V2_COUNTER_NAMES = (
    "observed_requests",
    "begun_requests",
    "completed_requests",
    "complete_requests",
    "incomplete_requests",
    "observed_records",
    "enqueued_records",
    "written_records",
    "observed_ids",
    "enqueued_ids",
    "written_ids",
    "observed_chunks",
    "enqueued_chunks",
    "written_chunks",
    "observed_batches",
    "enqueued_batches",
    "written_batches",
    "dropped_full_records",
    "dropped_full_ids",
    "dropped_runtime_records",
    "dropped_runtime_ids",
    "dropped_minimum_free_records",
    "dropped_minimum_free_ids",
    "dropped_io_budget_records",
    "dropped_io_budget_ids",
)


def _v2_header(
    *,
    instance_low: int = 33,
    instance_high: int = 44,
    segment_sequence: int = 0,
    reader_backend: int = 1,
    id_chunk_rows: int = 8192,
    max_active_requests: int = 64,
    buffer_bytes: int = 64 << 20,
    segment_bytes: int = 256 << 20,
    retention_bytes: int = 10 << 30,
    max_write_bytes_per_second: int = 16 << 20,
    io_workers: int | None = None,
    io_queue_capacity: int | None = None,
    io_ranges_per_task: int = 8,
    max_read_range_bytes: int = 8 << 20,
    max_gap_bytes: int = 64 << 10,
    io_uring_queue_depth: int | None = None,
    io_uring_workers: int | None = None,
) -> bytes:
    output = bytearray(HEADER_BYTES)
    output[:8] = b"TDBRLOG2"
    _put_u32(output, 8, 2)
    _put_u32(output, 12, HEADER_BYTES)
    _put_u32(output, 16, 232)
    _put_u64(output, 32, 11)
    _put_u64(output, 40, 22)
    _put_u32(output, 48, 7)
    _put_u32(output, 52, 1)
    _put_u32(output, 56, 1024)
    _put_u32(output, 60, 2048)
    _put_u64(output, 64, instance_low)
    _put_u64(output, 72, instance_high)
    _put_u64(output, 80, V2_UNIX_NS)
    _put_u32(output, 88, id_chunk_rows)
    _put_u32(output, 92, max_active_requests)
    _put_u64(output, 96, buffer_bytes)
    _put_u64(output, 104, segment_bytes)
    _put_u64(output, 112, 3600)
    _put_u64(output, 120, retention_bytes)
    _put_u64(output, 128, 7 * 24 * 60 * 60)
    _put_u64(output, 136, 2 << 30)
    _put_u64(output, 144, 1234)
    _put_u64(output, 152, segment_sequence)
    _put_u64(output, 160, max_write_bytes_per_second)
    _put_u64(output, 168, 4096)
    _put_u32(output, 176, reader_backend)
    if reader_backend == 1:
        _put_u32(output, 180, 8 if io_workers is None else io_workers)
        _put_u32(
            output,
            184,
            1024 if io_queue_capacity is None else io_queue_capacity,
        )
        _put_u32(output, 188, io_ranges_per_task)
        _put_u64(output, 192, max_read_range_bytes)
        _put_u64(output, 200, max_gap_bytes)
        _put_u32(
            output,
            208,
            32
            if io_uring_queue_depth is None
            else io_uring_queue_depth,
        )
        _put_u32(
            output,
            212,
            1 if io_uring_workers is None else io_uring_workers,
        )
    elif reader_backend == 2:
        _put_u32(output, 180, 0 if io_workers is None else io_workers)
        _put_u32(
            output,
            184,
            0 if io_queue_capacity is None else io_queue_capacity,
        )
        _put_u32(output, 188, io_ranges_per_task)
        _put_u64(output, 192, max_read_range_bytes)
        _put_u64(output, 200, max_gap_bytes)
        _put_u32(
            output,
            208,
            32
            if io_uring_queue_depth is None
            else io_uring_queue_depth,
        )
        _put_u32(
            output,
            212,
            2 if io_uring_workers is None else io_uring_workers,
        )
    _put_u32(output, 216, 4096)
    _put_u32(output, 220, 1 << 20)
    _put_u32(output, 224, 3)
    return _finish_crc(output)


def _v2_record(
    kind: int,
    frame_bytes: int,
    *,
    sequence: int,
    steady_ns: int,
    token: tuple[int, int] = V2_TOKEN,
) -> bytearray:
    output = bytearray(frame_bytes)
    output[:8] = b"TDBRREC2"
    _put_u32(output, 8, 2)
    _put_u32(output, 12, FRAME_BYTES)
    _put_u32(output, 16, frame_bytes)
    _put_u32(output, 20, kind)
    _put_u64(output, 40, sequence)
    _put_u64(output, 48, V2_UNIX_NS + sequence + 1)
    _put_u64(output, 56, steady_ns)
    _put_u64(output, 64, token[0])
    _put_u64(output, 72, token[1])
    return output


def _v2_begin(
    *,
    sequence: int,
    declared_ids: int,
    method: int = 1,
    prefix: int = 1024,
    token: tuple[int, int] = V2_TOKEN,
    steady_ns: int = 1_000,
    exact_snapshot: bool = True,
    catalog_generation: int = 7,
    committed_sequence: int = 9,
) -> bytes:
    output = _v2_record(
        1,
        FRAME_BYTES,
        sequence=sequence,
        steady_ns=steady_ns,
        token=token,
    )
    _put_u32(
        output,
        28,
        (1 if exact_snapshot else 0) | (2 if declared_ids else 0),
    )
    _put_u32(output, 32, method)
    _put_u32(output, 36, prefix)
    _put_u64(output, 80, catalog_generation)
    _put_u64(output, 88, committed_sequence)
    _put_u64(output, 96, 1_000_000)
    _put_u64(output, 104, declared_ids)
    _put_u64(output, 112, 1)
    return _finish_crc(output)


def _v2_chunk(
    ids: list[int],
    *,
    sequence: int,
    logical_offset: int,
    batch_sequence: int,
    chunk_sequence: int,
    delivery_sequence: int,
    token: tuple[int, int] = V2_TOKEN,
) -> bytes:
    output = _v2_record(
        2,
        FRAME_BYTES + len(ids) * 8,
        sequence=sequence,
        steady_ns=1_100 + sequence,
        token=token,
    )
    _put_u32(output, 32, len(ids))
    _put_u64(output, 80, logical_offset)
    _put_u64(output, 88, batch_sequence)
    assert delivery_sequence >= 0
    _put_u64(output, 96, 0)
    _put_u64(output, 104, chunk_sequence)
    for index, identifier in enumerate(ids):
        _put_u64(output, FRAME_BYTES + index * 8, identifier)
    return _finish_crc(output)


def _v2_batch(
    *,
    sequence: int,
    logical_offset: int,
    batch_sequence: int,
    rows: int,
    response_bytes: int,
    token: tuple[int, int] = V2_TOKEN,
    delivery_sequence: int = 0,
) -> bytes:
    output = _v2_record(
        3,
        384,
        sequence=sequence,
        steady_ns=1_200 + sequence,
        token=token,
    )
    _put_u32(output, 32, 1)
    _put_u64(output, 80, batch_sequence)
    _put_u64(output, 88, logical_offset)
    _put_u64(output, 96, rows)
    _put_u64(output, 104, response_bytes)
    for index in range(10):
        _put_u64(output, 112 + index * 8, index + 1)
    for index in range(16):
        _put_u64(output, 256 + index * 8, (index + 1) * 10)
    _put_u64(output, 224, delivery_sequence)
    _put_u64(output, 232, rows * 2048)
    _put_u64(output, 240, 2)
    return _finish_crc(output)


def _v2_digest(ids: list[int]) -> tuple[int, int]:
    low = 0xCBF29CE484222325
    high = 0x84222325CBF29CE4
    for identifier in ids:
        for byte in struct.pack("<Q", identifier):
            low = ((low ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
            high = ((high ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return low, high


def _v2_end(
    ids: list[int],
    *,
    sequence: int,
    chunks: int,
    batches: int,
    response_bytes: int,
    token: tuple[int, int] = V2_TOKEN,
    complete: bool = True,
    status: int = 1,
    error_code: int = 0,
    delivered_ids: int | None = None,
    digest: tuple[int, int] | None = None,
    started_steady_ns: int = 1_000,
) -> bytes:
    output = _v2_record(
        4,
        FRAME_BYTES,
        sequence=sequence,
        steady_ns=2_000,
        token=token,
    )
    _put_u32(output, 28, 1 if complete else 0)
    _put_u32(output, 32, status)
    _put_u32(output, 36, error_code)
    _put_u64(output, 80, len(ids))
    _put_u64(
        output,
        88,
        len(ids) if delivered_ids is None else delivered_ids,
    )
    _put_u64(output, 96, chunks)
    _put_u64(output, 104, batches)
    _put_u64(output, 112, response_bytes)
    _put_u64(output, 136, 2_000 - started_steady_ns)
    low, high = digest or _v2_digest(ids)
    _put_u64(output, 208, low)
    _put_u64(output, 216, high)
    return _finish_crc(output)


def _v2_checkpoint(
    *,
    opening: bool,
    counters: dict[str, int] | None = None,
) -> bytes:
    output = bytearray(FRAME_BYTES)
    output[:8] = b"TDBRREC2"
    _put_u32(output, 8, 2)
    _put_u32(output, 12, FRAME_BYTES)
    _put_u32(output, 16, FRAME_BYTES)
    _put_u32(output, 20, 5)
    _put_u32(output, 28, 1)
    _put_u32(output, 32, 1 if opening else 0)
    _put_u64(output, 48, V2_UNIX_NS + (1 if opening else 10_000))
    values = {name: 0 for name in V2_COUNTER_NAMES}
    values.update(counters or {})
    for index, name in enumerate(V2_COUNTER_NAMES):
        _put_u64(output, 56 + index * 8, values[name])
    return _finish_crc(output)


def _v2_complete_counters(
    *,
    records: int,
    ids: int,
    chunks: int,
    batches: int,
    requests: int = 1,
    complete_requests: int | None = None,
) -> dict[str, int]:
    complete = (
        requests if complete_requests is None else complete_requests
    )
    return {
        "observed_requests": requests,
        "begun_requests": requests,
        "completed_requests": requests,
        "complete_requests": complete,
        "incomplete_requests": requests - complete,
        "observed_records": records,
        "enqueued_records": records,
        "written_records": records,
        "observed_ids": ids,
        "enqueued_ids": ids,
        "written_ids": ids,
        "observed_chunks": chunks,
        "enqueued_chunks": chunks,
        "written_chunks": chunks,
        "observed_batches": batches,
        "enqueued_batches": batches,
        "written_batches": batches,
    }


def _v2_write_log(
    path: Path,
    frames: tuple[bytes, ...],
    *,
    opening_counters: dict[str, int] | None = None,
    terminal_counters: dict[str, int] | None = None,
) -> None:
    terminal = (
        (_v2_checkpoint(opening=False, counters=terminal_counters),)
        if terminal_counters is not None
        else ()
    )
    path.write_bytes(
        _v2_header()
        + _v2_checkpoint(opening=True, counters=opening_counters)
        + b"".join((*frames, *terminal))
    )


def test_v2_reconstructs_exact_get_without_serializing_ids(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [9_182_736_451, 8_172_635_409, 7_162_534_807]
    frames = (
        _v2_begin(sequence=1, declared_ids=len(ids)),
        _v2_chunk(
            ids,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=1,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=len(ids),
            response_bytes=6144,
        ),
        _v2_end(
            ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=6144,
        ),
    )
    _v2_write_log(
        logs / "exact.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=4,
            ids=3,
            chunks=1,
            batches=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["version"] == 2
    assert report["quality"] == {
        "suitable": True,
        "exact_capture": True,
        "reasons": [],
        "warnings": [],
    }
    assert report["requests"]["complete_for_training"] == 1
    assert report["requests"]["methods"] == {"get": 1}
    assert report["id_chunks"]["ids_for_complete_requests"] == 3
    assert report["batches"]["physical_metric_totals"][
        "physical_requests"
    ] == 20
    assert report["batches"]["physical_metric_totals"][
        "useful_read_bytes"
    ] == 6144
    assert report["batches"]["physical_metric_totals"][
        "backend_tasks"
    ] == 2
    assert report["configuration"]["reader_backend"] == "thread_pool"
    serialized = json.dumps(report, sort_keys=True)
    assert all(str(identifier) not in serialized for identifier in ids)
    assert "digest" not in serialized
    assert "request_id" not in serialized


def test_v2_native_and_fallback_digest_match_fragmented_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = memoryview(
        struct.pack("<QQQ", 0, (1 << 64) - 1, 0x123456789ABCDEF0)
    )
    initial = 0xCBF29CE484222325
    native_or_default = _v2_fnv_update(initial, payload)
    monkeypatch.setattr(workload_module, "_native", None)
    fallback = _v2_fnv_update(initial, payload)
    fragmented = initial
    for start, end in ((0, 3), (3, 17), (17, len(payload))):
        fragmented = _v2_fnv_update(
            fragmented,
            payload[start:end],
        )

    assert native_or_default == fallback == fragmented


def test_v2_accepts_unary_inexact_observation_anchor(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [71, 72]
    frames = (
        _v2_begin(
            sequence=1,
            declared_ids=len(ids),
            exact_snapshot=False,
            catalog_generation=17,
            committed_sequence=19,
        ),
        _v2_chunk(
            ids,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=len(ids),
            response_bytes=4096,
        ),
        _v2_end(
            ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=4096,
        ),
    )
    _v2_write_log(
        logs / "unary.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=4,
            ids=2,
            chunks=1,
            batches=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["valid"] == 1
    assert report["quality"]["suitable"] is True
    assert report["requests"]["complete_for_training"] == 1
    assert report["requests"]["reports"][0]["exact_snapshot"] is False


def test_v2_fully_captured_non_success_is_diagnostic_not_telemetry_loss(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    success_ids = [101]
    cancelled_ids = [202, 203]
    success_token = (11, 12)
    cancelled_token = (21, 22)
    frames = (
        _v2_begin(
            sequence=1,
            declared_ids=1,
            token=success_token,
        ),
        _v2_chunk(
            success_ids,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
            token=success_token,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=1,
            response_bytes=2048,
            token=success_token,
        ),
        _v2_end(
            success_ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=2048,
            token=success_token,
        ),
        _v2_begin(
            sequence=5,
            declared_ids=2,
            token=cancelled_token,
        ),
        _v2_chunk(
            cancelled_ids,
            sequence=6,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
            token=cancelled_token,
        ),
        _v2_end(
            cancelled_ids,
            sequence=7,
            chunks=1,
            batches=0,
            response_bytes=0,
            token=cancelled_token,
            complete=False,
            status=2,
            error_code=1,
            delivered_ids=0,
        ),
    )
    _v2_write_log(
        logs / "mixed-outcomes.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=7,
            ids=3,
            chunks=2,
            batches=1,
            requests=2,
            complete_requests=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["exact_capture"] is True
    assert report["quality"]["suitable"] is True
    assert report["quality"]["reasons"] == []
    assert report["quality"]["warnings"] == [
        "non_success_request_outcomes"
    ]
    assert report["requests"]["complete_for_training"] == 1
    assert report["requests"]["non_success_outcomes"] == 1
    assert report["requests"]["incomplete_outcomes"] == 0
    assert report["requests"]["methods"] == {"get": 1}
    assert report["requests"]["statuses"] == {
        "cancelled": 1,
        "success": 1,
    }


def test_v2_reconstructs_multibatch_stream_in_original_order(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    first = [101, 202]
    second = [303, 404, 505]
    ids = first + second
    frames = (
        _v2_begin(
            sequence=1,
            declared_ids=len(ids),
            method=2,
            prefix=128,
        ),
        _v2_chunk(
            first,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=9,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=2,
            response_bytes=512,
            delivery_sequence=9,
        ),
        _v2_chunk(
            second,
            sequence=4,
            logical_offset=2,
            batch_sequence=1,
            chunk_sequence=1,
            delivery_sequence=10,
        ),
        _v2_batch(
            sequence=5,
            logical_offset=2,
            batch_sequence=1,
            rows=3,
            response_bytes=768,
            delivery_sequence=10,
        ),
        _v2_end(
            ids,
            sequence=6,
            chunks=2,
            batches=2,
            response_bytes=1280,
        ),
    )
    _v2_write_log(
        logs / "stream.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=6,
            ids=5,
            chunks=2,
            batches=2,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["exact_capture"] is True
    assert report["requests"]["methods"] == {"stream_get": 1}
    assert report["requests"]["prefix_dimensions"] == {"128": 1}
    assert report["requests"]["reports"][0]["batches"] == 2
    assert report["batches"]["rows_for_complete_requests"] == 5


@pytest.mark.parametrize("failure", ("missing_chunk", "bad_digest"))
def test_v2_fails_closed_on_interior_gap_or_digest_mismatch(
    tmp_path: Path,
    failure: str,
) -> None:
    logs = tmp_path / failure
    logs.mkdir()
    ids = [11, 22, 33]
    chunk = _v2_chunk(
        ids,
        sequence=2,
        logical_offset=0,
        batch_sequence=0,
        chunk_sequence=0,
        delivery_sequence=1,
    )
    frames = (
        _v2_begin(sequence=1, declared_ids=3),
        *(() if failure == "missing_chunk" else (chunk,)),
        _v2_batch(
            sequence=3 if failure == "missing_chunk" else 3,
            logical_offset=0,
            batch_sequence=0,
            rows=3,
            response_bytes=6144,
        ),
        _v2_end(
            ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=6144,
            digest=(
                (1, 2)
                if failure == "bad_digest"
                else _v2_digest(ids)
            ),
        ),
    )
    records = 3 if failure == "missing_chunk" else 4
    _v2_write_log(
        logs / "bad.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=records,
            ids=0 if failure == "missing_chunk" else 3,
            chunks=0 if failure == "missing_chunk" else 1,
            batches=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is False
    assert report["quality"]["exact_capture"] is False
    assert report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in report["quality"]["reasons"]


def test_v2_retention_left_censor_is_excluded_from_training(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    left_ids = [41, 42]
    kept_ids = [51, 52, 53]
    kept_token = (123, 456)
    opening = {
        **_v2_complete_counters(
            records=1,
            ids=0,
            chunks=0,
            batches=0,
        ),
        "completed_requests": 0,
        "complete_requests": 0,
    }
    frames = (
        _v2_chunk(
            left_ids,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=1,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=2,
            response_bytes=4096,
        ),
        _v2_end(
            left_ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=4096,
        ),
        _v2_begin(
            sequence=5,
            declared_ids=3,
            token=kept_token,
        ),
        _v2_chunk(
            kept_ids,
            sequence=6,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=2,
            token=kept_token,
        ),
        _v2_batch(
            sequence=7,
            logical_offset=0,
            batch_sequence=0,
            rows=3,
            response_bytes=6144,
            token=kept_token,
        ),
        _v2_end(
            kept_ids,
            sequence=8,
            chunks=1,
            batches=1,
            response_bytes=6144,
            token=kept_token,
        ),
    )
    terminal = _v2_complete_counters(
        records=8,
        ids=5,
        chunks=2,
        batches=2,
        requests=2,
    )
    _v2_write_log(
        logs / "retained.tdbrl",
        frames,
        opening_counters=opening,
        terminal_counters=terminal,
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is True
    assert report["requests"]["left_censored"] == 1
    assert report["requests"]["begun"] == 1
    assert report["requests"]["complete_for_training"] == 1
    assert report["id_chunks"]["ids_for_complete_requests"] == 3


def test_v2_right_censor_and_active_request_bound_are_fail_closed(
    tmp_path: Path,
) -> None:
    open_logs = tmp_path / "open"
    open_logs.mkdir()
    ids = [1, 2]
    _v2_write_log(
        open_logs / "active.open",
        (
            _v2_begin(sequence=1, declared_ids=2),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=1,
            ),
        ),
    )

    report = inspect_workload_logs(open_logs)

    assert report["requests"]["right_censored"] == 1
    assert report["requests"]["interior_incomplete"] == 0
    assert report["quality"]["exact_capture"] is False
    assert report["quality"]["suitable"] is False
    assert report["quality"]["reasons"] == [
        "incomplete_counter_coverage",
        "no_complete_point_read_requests",
    ]
    assert report["files"]["reports"][0]["counter_coverage"] == (
        "lower_bound"
    )

    bounded = tmp_path / "bounded"
    bounded.mkdir()
    second_token = (777, 888)
    frames = (
        _v2_begin(sequence=1, declared_ids=1),
        _v2_begin(
            sequence=2,
            declared_ids=1,
            token=second_token,
        ),
    )
    _v2_write_log(
        bounded / "too-many.tdbrl",
        frames,
        terminal_counters={
            "observed_requests": 2,
            "begun_requests": 2,
            "observed_records": 2,
            "enqueued_records": 2,
            "written_records": 2,
        },
    )

    with pytest.raises(WorkloadLogError, match="max_active_requests"):
        inspect_workload_logs(bounded, max_active_requests=1)


def test_v2_open_file_keeps_verified_prefix_and_excludes_tail(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [61, 62]
    tail_token = (900, 901)
    verified = _v2_complete_counters(
        records=4,
        ids=2,
        chunks=1,
        batches=1,
    )
    _v2_write_log(
        logs / "active.open",
        (
            _v2_begin(sequence=1, declared_ids=2),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=0,
            ),
            _v2_batch(
                sequence=3,
                logical_offset=0,
                batch_sequence=0,
                rows=2,
                response_bytes=4096,
            ),
            _v2_end(
                ids,
                sequence=4,
                chunks=1,
                batches=1,
                response_bytes=4096,
            ),
            _v2_checkpoint(opening=False, counters=verified),
            _v2_begin(
                sequence=5,
                declared_ids=1,
                token=tail_token,
            ),
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is False
    assert report["quality"]["exact_capture"] is False
    assert report["quality"]["reasons"] == [
        "incomplete_counter_coverage"
    ]
    assert report["requests"]["complete_for_training"] == 1
    assert report["requests"]["right_censored"] == 1
    assert report["requests"]["begun"] == 2
    assert report["counters"]["written_records"] == 4


def test_v2_writer_drops_ignore_legacy_thresholds_and_fail_exactness(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [71]
    frames = (
        _v2_begin(sequence=1, declared_ids=1),
        _v2_chunk(
            ids,
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=1,
            response_bytes=2048,
        ),
        _v2_end(
            ids,
            sequence=4,
            chunks=1,
            batches=1,
            response_bytes=2048,
        ),
    )
    counters = _v2_complete_counters(
        records=4,
        ids=1,
        chunks=1,
        batches=1,
    )
    counters.update(
        {
            "observed_requests": 2,
            "completed_requests": 2,
            "incomplete_requests": 1,
            "observed_records": 5,
            "dropped_full_records": 1,
        }
    )
    _v2_write_log(
        logs / "drops.tdbrl",
        frames,
        terminal_counters=counters,
    )

    report = inspect_workload_logs(
        logs,
        max_drop_ratio=1.0,
        max_row_drop_ratio=1.0,
    )

    assert report["quality"]["exact_capture"] is False
    assert report["quality"]["suitable"] is False
    assert "exact_records_dropped" in report["quality"]["reasons"]
    assert "drop_ratio_exceeds_limit" not in report["quality"]["reasons"]


def test_v2_intermediate_checkpoint_may_have_inflight_pipeline(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [75]
    intermediate = {
        "observed_requests": 1,
        "begun_requests": 1,
        "observed_records": 2,
        "enqueued_records": 2,
        "written_records": 1,
    }
    _v2_write_log(
        logs / "inflight.tdbrl",
        (
            _v2_begin(sequence=1, declared_ids=1),
            _v2_checkpoint(
                opening=False,
                counters=intermediate,
            ),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=0,
            ),
            _v2_batch(
                sequence=3,
                logical_offset=0,
                batch_sequence=0,
                rows=1,
                response_bytes=2048,
            ),
            _v2_end(
                ids,
                sequence=4,
                chunks=1,
                batches=1,
                response_bytes=2048,
            ),
        ),
        terminal_counters=_v2_complete_counters(
            records=4,
            ids=1,
            chunks=1,
            batches=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["valid"] == 1
    assert report["quality"]["suitable"] is True
    assert report["files"]["reports"][0]["checkpoint_records"] == 3


def test_v2_final_terminal_requires_complete_counter_partitions(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [76]
    terminal = _v2_complete_counters(
        records=4,
        ids=1,
        chunks=1,
        batches=1,
    )
    terminal["observed_records"] = 5
    _v2_write_log(
        logs / "incomplete-terminal.tdbrl",
        (
            _v2_begin(sequence=1, declared_ids=1),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=0,
            ),
            _v2_batch(
                sequence=3,
                logical_offset=0,
                batch_sequence=0,
                rows=1,
                response_bytes=2048,
            ),
            _v2_end(
                ids,
                sequence=4,
                chunks=1,
                batches=1,
                response_bytes=2048,
            ),
        ),
        terminal_counters=terminal,
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["invalid_finalized"] == 0
    assert report["quality"]["exact_capture"] is False
    assert "incomplete_terminal_counter_partition" in (
        report["quality"]["reasons"]
    )
    assert (
        report["counters"]["incomplete_terminal_partition_instances"]
        == 1
    )


def test_v2_minimum_chunk_geometry_still_accounts_for_batch_frame(
    tmp_path: Path,
) -> None:
    valid_logs = tmp_path / "valid"
    valid_logs.mkdir()
    minimum_transaction = HEADER_BYTES + 2 * FRAME_BYTES + 384
    frames = (
        _v2_checkpoint(opening=True),
        _v2_checkpoint(opening=False),
    )
    (valid_logs / "minimum.tdbrl").write_bytes(
        _v2_header(
            id_chunk_rows=1,
            buffer_bytes=2 * 384,
            segment_bytes=minimum_transaction,
            retention_bytes=minimum_transaction,
            max_write_bytes_per_second=minimum_transaction,
        )
        + b"".join(frames)
    )

    valid_report = inspect_workload_logs(valid_logs)

    assert valid_report["files"]["valid"] == 1
    assert valid_report["files"]["invalid_finalized"] == 0

    invalid_logs = tmp_path / "invalid"
    invalid_logs.mkdir()
    (invalid_logs / "undersized-buffer.tdbrl").write_bytes(
        _v2_header(
            id_chunk_rows=1,
            buffer_bytes=2 * 384 - 1,
            segment_bytes=minimum_transaction,
            retention_bytes=minimum_transaction,
            max_write_bytes_per_second=minimum_transaction,
        )
        + b"".join(frames)
    )

    invalid_report = inspect_workload_logs(invalid_logs)

    assert invalid_report["files"]["invalid_finalized"] == 1


def test_v2_accepts_default_sequential_thread_pool_reader_context(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    frames = (
        _v2_checkpoint(opening=True),
        _v2_checkpoint(opening=False),
    )
    (logs / "sequential.tdbrl").write_bytes(
        _v2_header(
            reader_backend=1,
            io_workers=0,
            io_queue_capacity=0,
            io_uring_queue_depth=32,
            io_uring_workers=1,
        )
        + b"".join(frames)
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["valid"] == 1
    assert report["files"]["invalid_finalized"] == 0
    assert report["configuration"]["reader_backend"] == "thread_pool"
    assert report["configuration"]["io_workers"] == 0
    assert report["configuration"]["io_queue_capacity"] == 0
    assert report["configuration"]["io_uring_queue_depth"] == 32
    assert report["configuration"]["io_uring_workers"] == 1


@pytest.mark.parametrize(
    "header_overrides",
    [
        {
            "reader_backend": 1,
            "io_workers": 0,
            "io_queue_capacity": 1,
        },
        {
            "reader_backend": 1,
            "io_uring_queue_depth": 0,
            "io_uring_workers": 1,
        },
        {
            "reader_backend": 2,
            "io_workers": 1,
            "io_queue_capacity": 0,
        },
        {
            "reader_backend": 1,
            "max_active_requests": 65_537,
        },
    ],
)
def test_v2_rejects_impossible_native_logger_context(
    tmp_path: Path,
    header_overrides: dict[str, int],
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "invalid-reader.tdbrl").write_bytes(
        _v2_header(**header_overrides)
        + _v2_checkpoint(opening=True)
        + _v2_checkpoint(opening=False)
    )

    report = inspect_workload_logs(logs)

    assert report["files"]["valid"] == 0
    assert report["files"]["invalid_finalized"] == 1
    assert "corrupt_finalized_files" in report["quality"]["reasons"]


def test_v2_pending_metadata_bounds_are_global_across_instances() -> None:
    accumulator = workload_module._V2Accumulator(
        max_active_requests=4,
        max_report_entries=1,
        max_prefix_entries=1,
    )
    first = accumulator._pending((1, 1))
    second = accumulator._pending((2, 2))
    accumulator._record_pending_prefix(first, 128)
    accumulator._reserve_pending_report(first, {"instance": 1})

    with pytest.raises(
        WorkloadLogError,
        match="max_prefix_entries",
    ):
        accumulator._record_pending_prefix(second, 256)
    accumulator._reserve_pending_report(second, {"instance": 2})

    assert accumulator.pending_request_report_entries == 1
    assert len(first.request_reports) == 1
    assert len(second.request_reports) == 0
    assert second.request_reports_omitted == 1

    accumulator.finish_instance((1, 1), complete_terminal=False)
    assert accumulator.pending_request_report_entries == 0
    assert accumulator.pending_prefix_references == {}


def test_v2_file_report_memory_is_bounded_before_return(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(3):
        (logs / f"{index}.tdbrl").write_bytes(
            _v2_header(
                instance_low=index + 1,
                instance_high=index + 10,
            )
            + _v2_checkpoint(opening=True)
            + _v2_checkpoint(opening=False)
        )

    report = inspect_workload_logs(
        logs,
        max_report_entries=1,
    )

    assert len(report["files"]["reports"]) == 1
    assert report["files"]["reports_omitted"] == 2


def test_v2_lifecycle_and_delivery_sequences_fail_closed(
    tmp_path: Path,
) -> None:
    lifecycle = tmp_path / "lifecycle"
    lifecycle.mkdir()
    ids = [81]
    _v2_write_log(
        lifecycle / "gap.tdbrl",
        (
            _v2_begin(sequence=1, declared_ids=1),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=0,
            ),
            _v2_batch(
                sequence=4,
                logical_offset=0,
                batch_sequence=0,
                rows=1,
                response_bytes=2048,
            ),
            _v2_end(
                ids,
                sequence=5,
                chunks=1,
                batches=1,
                response_bytes=2048,
            ),
        ),
        terminal_counters=_v2_complete_counters(
            records=4,
            ids=1,
            chunks=1,
            batches=1,
        ),
    )

    gap_report = inspect_workload_logs(lifecycle)

    assert gap_report["quality"]["exact_capture"] is False
    assert "lifecycle_sequence_gap" in gap_report["quality"]["reasons"]

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    stream_ids = [91, 92]
    frames = (
        _v2_begin(sequence=1, declared_ids=2, method=2),
        _v2_chunk(
            [91],
            sequence=2,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
        ),
        _v2_batch(
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            rows=1,
            response_bytes=2048,
            delivery_sequence=10,
        ),
        _v2_chunk(
            [92],
            sequence=4,
            logical_offset=1,
            batch_sequence=1,
            chunk_sequence=1,
            delivery_sequence=0,
        ),
        _v2_batch(
            sequence=5,
            logical_offset=1,
            batch_sequence=1,
            rows=1,
            response_bytes=2048,
            delivery_sequence=9,
        ),
        _v2_end(
            stream_ids,
            sequence=6,
            chunks=2,
            batches=2,
            response_bytes=4096,
        ),
    )
    _v2_write_log(
        delivery / "reordered.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=6,
            ids=2,
            chunks=2,
            batches=2,
        ),
    )

    delivery_report = inspect_workload_logs(delivery)

    assert delivery_report["quality"]["exact_capture"] is False
    assert "delivery_sequence_order" in (
        delivery_report["quality"]["reasons"]
    )


def test_v2_stream_batch_requires_delivery_sequence(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    ids = [501]
    _v2_write_log(
        logs / "missing-delivery.tdbrl",
        (
            _v2_begin(
                sequence=1,
                declared_ids=1,
                method=2,
            ),
            _v2_chunk(
                ids,
                sequence=2,
                logical_offset=0,
                batch_sequence=0,
                chunk_sequence=0,
                delivery_sequence=0,
            ),
            _v2_batch(
                sequence=3,
                logical_offset=0,
                batch_sequence=0,
                rows=1,
                response_bytes=2048,
                delivery_sequence=0,
            ),
            _v2_end(
                ids,
                sequence=4,
                chunks=1,
                batches=1,
                response_bytes=2048,
            ),
        ),
        terminal_counters=_v2_complete_counters(
            records=4,
            ids=1,
            chunks=1,
            batches=1,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["exact_capture"] is False
    assert "delivery_sequence_order" in report["quality"]["reasons"]


def test_v2_global_delivery_inversion_across_requests_is_diagnostic(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    first_token = (1_001, 1_002)
    second_token = (2_001, 2_002)
    frames = (
        _v2_begin(
            sequence=1,
            declared_ids=1,
            method=2,
            token=first_token,
        ),
        _v2_begin(
            sequence=2,
            declared_ids=1,
            method=2,
            token=second_token,
        ),
        _v2_chunk(
            [101],
            sequence=3,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
            token=first_token,
        ),
        _v2_batch(
            sequence=4,
            logical_offset=0,
            batch_sequence=0,
            rows=1,
            response_bytes=2048,
            delivery_sequence=10,
            token=first_token,
        ),
        _v2_chunk(
            [202],
            sequence=5,
            logical_offset=0,
            batch_sequence=0,
            chunk_sequence=0,
            delivery_sequence=0,
            token=second_token,
        ),
        _v2_batch(
            sequence=6,
            logical_offset=0,
            batch_sequence=0,
            rows=1,
            response_bytes=2048,
            delivery_sequence=9,
            token=second_token,
        ),
        _v2_end(
            [101],
            sequence=7,
            chunks=1,
            batches=1,
            response_bytes=2048,
            token=first_token,
        ),
        _v2_end(
            [202],
            sequence=8,
            chunks=1,
            batches=1,
            response_bytes=2048,
            token=second_token,
        ),
    )
    _v2_write_log(
        logs / "concurrent.tdbrl",
        frames,
        terminal_counters=_v2_complete_counters(
            records=8,
            ids=2,
            chunks=2,
            batches=2,
            requests=2,
        ),
    )

    report = inspect_workload_logs(logs)

    assert report["quality"]["suitable"] is True
    assert report["batches"]["delivery_sequence"] == {
        "nonzero_count": 2,
        "minimum": 9,
        "maximum": 10,
        "span": 2,
        "unobserved_within_span": 0,
        "physical_inversions": 1,
    }
