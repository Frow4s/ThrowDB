"""Bounded inspection of native ThrowDB read-workload logs.

The request logger is intentionally implemented in the C++ server. This
module is an offline control-plane reader: it validates framed files, exposes
aggregate quality signals, and never returns or prints raw IDs.
"""

from __future__ import annotations

import math
import mmap
import os
import stat
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Final

from ._throwdb import NativeCollection

try:
    from . import _throwdb as _native
except ImportError:  # pragma: no cover - a normal install always has it.
    _native = None


_HEADER_MAGIC: Final = b"TDBRLOG1"
_RECORD_MAGIC: Final = b"TDBRREC1"
_VERSION: Final = 1
_HEADER_BYTES: Final = 4096
_RECORD_HEADER_BYTES: Final = 256
_CRC_OFFSET: Final = 24
_CRC_SIZE: Final = 4
_HEADER_USED_BYTES: Final = 176
_MAX_IDS_PER_EVENT: Final = 65_536
_MAX_RETAINED_FILES: Final = 65_533
_INT64_MAX: Final = (1 << 63) - 1
_MAX_FRAME_BYTES: Final = _RECORD_HEADER_BYTES + 8 * _MAX_IDS_PER_EVENT
_UINT64_MAX: Final = (1 << 64) - 1
_EVENT_KIND: Final = 1
_CHECKPOINT_KIND: Final = 2
_CHECKPOINT_EXTENSION_VERSION: Final = 2
_CHECKPOINT_OPENING_BASELINE_FLAG: Final = 1 << 0
_KNOWN_CHECKPOINT_FLAGS: Final = _CHECKPOINT_OPENING_BASELINE_FLAG
_GET_METHOD: Final = 1
_STREAM_GET_METHOD: Final = 2
_EXACT_SNAPSHOT_FLAG: Final = 1 << 0
_TRUNCATED_FLAG: Final = 1 << 1
_KNOWN_EVENT_FLAGS: Final = _EXACT_SNAPSHOT_FLAG | _TRUNCATED_FLAG
_SUFFIXES: Final = frozenset({".tdbrl", ".open"})

# These bounds are deliberately higher than the native server defaults
# (256-MiB segments and 10-GiB retention), but finite.  Operators inspecting a
# deliberately larger retained history must opt in explicitly.
DEFAULT_MAX_FILES: Final = 65_536
DEFAULT_MAX_TOTAL_BYTES: Final = 64 << 30
DEFAULT_MAX_FILE_BYTES: Final = 4 << 30
DEFAULT_MAX_REPORT_ENTRIES: Final = 1_024
DEFAULT_MAX_PREFIX_ENTRIES: Final = 4_096


class WorkloadLogError(ValueError):
    """A workload-log directory or inspection argument is invalid."""


class _WorkloadLogLimitError(WorkloadLogError):
    """An input changed or exceeded a hard inspection bound."""


def _check_deadline(deadline_monotonic: float | None) -> None:
    if (
        deadline_monotonic is not None
        and time.monotonic() >= deadline_monotonic
    ):
        raise _WorkloadLogLimitError(
            "workload-log inspection exceeded its caller deadline"
        )


_CRC32C_TABLE: tuple[int, ...] | None = None

_COUNTER_NAMES: Final = (
    "observed_events",
    "observed_ids",
    "unsampled_events",
    "sampled_events",
    "enqueued_events",
    "written_events",
    "captured_ids",
    "truncated_events",
    "dropped_busy_events",
    "dropped_full_events",
    "dropped_runtime_failure_events",
    "dropped_minimum_free_events",
    "writer_errors",
    "bytes_written",
    "segments_created",
    "rotated_segments",
)

_CHECKPOINT_COUNTER_NAMES: Final = (
    "sampled_ids",
    "enqueued_ids",
    "written_ids",
    "dropped_busy_ids",
    "dropped_full_ids",
    "dropped_runtime_failure_ids",
    "dropped_minimum_free_ids",
    "dropped_io_budget_events",
    "dropped_io_budget_ids",
)

_ALL_COUNTER_NAMES: Final = _COUNTER_NAMES + _CHECKPOINT_COUNTER_NAMES


def _crc32c_table() -> tuple[int, ...]:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        values: list[int] = []
        for byte in range(256):
            value = byte
            for _ in range(8):
                value = (
                    (value >> 1) ^ 0x82F63B78
                    if value & 1
                    else value >> 1
                )
            values.append(value)
        _CRC32C_TABLE = tuple(values)
    return _CRC32C_TABLE


def _crc32c(
    data: memoryview,
    *,
    zero_offset: int = 0,
    zero_size: int = 0,
) -> int:
    native = getattr(_native, "_crc32c", None)
    if native is not None:
        return int(native(data, zero_offset, zero_size))
    if (
        zero_offset < 0
        or zero_size < 0
        or zero_offset > len(data)
        or zero_size > len(data) - zero_offset
    ):
        raise WorkloadLogError("CRC32C zero range is outside the buffer")
    table = _crc32c_table()
    checksum = 0xFFFFFFFF
    zero_end = zero_offset + zero_size
    for index, byte in enumerate(data):
        value = 0 if zero_offset <= index < zero_end else byte
        checksum = table[(checksum ^ value) & 0xFF] ^ (checksum >> 8)
    return checksum ^ 0xFFFFFFFF


def _u32(data: memoryview, offset: int) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _u64(data: memoryview, offset: int) -> int:
    return int(struct.unpack_from("<Q", data, offset)[0])


def _header_identity(header: dict[str, object]) -> tuple[object, ...]:
    return (
        header["collection_id_low"],
        header["collection_id_high"],
        header["format_version"],
        header["dtype_code"],
        header["dimension"],
        header["record_size"],
    )


def _parse_header(data: memoryview) -> dict[str, object]:
    if len(data) != _HEADER_BYTES:
        raise WorkloadLogError("workload-log header is truncated")
    if bytes(data[:8]) != _HEADER_MAGIC:
        raise WorkloadLogError("workload-log header magic is invalid")
    version = _u32(data, 8)
    header_bytes = _u32(data, 12)
    used_bytes = _u32(data, 16)
    flags = _u32(data, 20)
    if version != _VERSION or header_bytes != _HEADER_BYTES:
        raise WorkloadLogError("workload-log header version is unsupported")
    if used_bytes != _HEADER_USED_BYTES:
        raise WorkloadLogError("workload-log header used size is invalid")
    if (
        flags != 0
        or any(data[28:32])
        or any(data[_HEADER_USED_BYTES:])
    ):
        raise WorkloadLogError("workload-log header has unknown flags")
    expected_crc = _u32(data, _CRC_OFFSET)
    if (
        _crc32c(data, zero_offset=_CRC_OFFSET, zero_size=_CRC_SIZE)
        != expected_crc
    ):
        raise WorkloadLogError("workload-log header CRC32C is invalid")

    collection_id_low = _u64(data, 32)
    collection_id_high = _u64(data, 40)
    format_version = _u32(data, 48)
    dtype_code = _u32(data, 52)
    dimension = _u32(data, 56)
    record_size = _u32(data, 60)
    sample_ppm = _u32(data, 88)
    max_ids = _u32(data, 92)
    buffer_bytes = _u64(data, 96)
    segment_bytes = _u64(data, 104)
    rotate_seconds = _u64(data, 112)
    retention_bytes = _u64(data, 120)
    retention_seconds = _u64(data, 128)
    minimum_free_bytes = _u64(data, 136)
    instance_id_low = _u64(data, 64)
    instance_id_high = _u64(data, 72)
    created_unix_ns = _u64(data, 80)
    segment_sequence = _u64(data, 152)
    max_write_bytes_per_second = _u64(data, 160)
    max_retained_files = _u64(data, 168)
    item_size = {1: 2, 2: 4}.get(dtype_code)
    if (
        (collection_id_low == 0 and collection_id_high == 0)
        or (instance_id_low == 0 and instance_id_high == 0)
        or created_unix_ns == 0
        or format_version == 0
        or item_size is None
        or dimension == 0
        or record_size != dimension * item_size
        or sample_ppm < 1
        or sample_ppm > 1_000_000
        or max_ids < 1
        or max_ids > _MAX_IDS_PER_EVENT
        or buffer_bytes < 2 * (_RECORD_HEADER_BYTES + max_ids * 8)
        or segment_bytes
            < (
                _HEADER_BYTES
                + 2 * _RECORD_HEADER_BYTES
                + max_ids * 8
                + _RECORD_HEADER_BYTES
            )
        or retention_bytes < segment_bytes
        or rotate_seconds == 0
        or rotate_seconds > _INT64_MAX
        or retention_seconds == 0
        or max_write_bytes_per_second
            < (
                _HEADER_BYTES
                + 2 * _RECORD_HEADER_BYTES
                + max_ids * 8
                + _RECORD_HEADER_BYTES
            )
        or max_retained_files == 0
        or max_retained_files > _MAX_RETAINED_FILES
    ):
        raise WorkloadLogError("workload-log header configuration is invalid")
    return {
        "collection_id_low": collection_id_low,
        "collection_id_high": collection_id_high,
        "collection_id": (
            f"{collection_id_high:016x}{collection_id_low:016x}"
        ),
        "format_version": format_version,
        "dtype_code": dtype_code,
        "dtype": "float16" if dtype_code == 1 else "float32",
        "dimension": dimension,
        "record_size": record_size,
        "instance_id_low": instance_id_low,
        "instance_id_high": instance_id_high,
        "created_unix_ns": created_unix_ns,
        "segment_sequence": segment_sequence,
        "sample_ppm": sample_ppm,
        "max_ids_per_event": max_ids,
        "buffer_bytes": buffer_bytes,
        "segment_bytes": segment_bytes,
        "rotate_seconds": rotate_seconds,
        "retention_bytes": retention_bytes,
        "retention_seconds": retention_seconds,
        "minimum_free_bytes": minimum_free_bytes,
        "process_id": _u64(data, 144),
        "max_write_bytes_per_second": max_write_bytes_per_second,
        "max_retained_files": max_retained_files,
    }


def _counter_snapshot(frame: memoryview) -> dict[str, int]:
    return {
        name: _u64(frame, 128 + index * 8)
        for index, name in enumerate(_COUNTER_NAMES)
    }


def _checkpoint_counter_snapshot(frame: memoryview) -> dict[str, int]:
    return {
        "sampled_ids": _u64(frame, 56),
        "enqueued_ids": _u64(frame, 64),
        "written_ids": _u64(frame, 72),
        "dropped_busy_ids": _u64(frame, 80),
        "dropped_full_ids": _u64(frame, 88),
        "dropped_runtime_failure_ids": _u64(frame, 96),
        "dropped_minimum_free_ids": _u64(frame, 104),
        "dropped_io_budget_events": _u64(frame, 112),
        "dropped_io_budget_ids": _u64(frame, 120),
    }


def _saturating_sum(*values: int) -> int:
    result = 0
    for value in values:
        result = min(_UINT64_MAX, result + value)
    return result


def _validate_counter_snapshot(counters: dict[str, int]) -> None:
    observed = counters["observed_events"]
    sampled = counters["sampled_events"]
    unsampled = counters["unsampled_events"]
    enqueued = counters["enqueued_events"]
    written = counters["written_events"]
    drops = _saturating_sum(
        counters["dropped_busy_events"],
        counters["dropped_full_events"],
        counters["dropped_runtime_failure_events"],
        counters["dropped_minimum_free_events"],
        counters.get("dropped_io_budget_events", 0),
    )
    if (
        _saturating_sum(sampled, unsampled) > observed
        or enqueued > sampled
        or written > enqueued
        or drops > sampled
        or counters["captured_ids"] > counters["observed_ids"]
        or counters["truncated_events"] > enqueued
        or counters["rotated_segments"] > counters["segments_created"]
        or counters["writer_errors"] != 0
    ):
        raise WorkloadLogError("workload-log counters are inconsistent")


def _validate_checkpoint_counter_snapshot(counters: dict[str, int]) -> None:
    sampled_ids = counters["sampled_ids"]
    enqueued_ids = counters["enqueued_ids"]
    written_ids = counters["written_ids"]
    dropped_ids = _saturating_sum(
        counters["dropped_busy_ids"],
        counters["dropped_full_ids"],
        counters["dropped_runtime_failure_ids"],
        counters["dropped_minimum_free_ids"],
        counters["dropped_io_budget_ids"],
    )
    if (
        enqueued_ids > sampled_ids
        or written_ids > enqueued_ids
        or dropped_ids > sampled_ids
        or sampled_ids > counters["observed_ids"]
        or counters["captured_ids"] > enqueued_ids
        or counters["dropped_io_budget_events"]
            > counters["sampled_events"]
        or counters["dropped_io_budget_events"]
            > counters["enqueued_events"]
        or counters["dropped_io_budget_ids"] > enqueued_ids
    ):
        raise WorkloadLogError(
            "workload-log checkpoint counters are inconsistent"
        )


def _terminal_partitions_are_complete(counters: dict[str, int]) -> bool:
    admission_event_drops = _saturating_sum(
        counters["dropped_busy_events"],
        counters["dropped_full_events"],
        counters["dropped_runtime_failure_events"],
        counters["dropped_minimum_free_events"],
    )
    admission_id_drops = _saturating_sum(
        counters["dropped_busy_ids"],
        counters["dropped_full_ids"],
        counters["dropped_runtime_failure_ids"],
        counters["dropped_minimum_free_ids"],
    )
    return (
        counters["sampled_events"]
        == _saturating_sum(
            counters["enqueued_events"], admission_event_drops
        )
        and counters["enqueued_events"]
        == _saturating_sum(
            counters["written_events"],
            counters["dropped_io_budget_events"],
        )
        and counters["sampled_ids"]
        == _saturating_sum(
            counters["enqueued_ids"], admission_id_drops
        )
        and counters["enqueued_ids"]
        == _saturating_sum(
            counters["written_ids"],
            counters["dropped_io_budget_ids"],
        )
    )


def _counters_dominate(
    newer: dict[str, int],
    older: dict[str, int],
) -> bool:
    return all(newer[name] >= older[name] for name in older)


def _counter_component_max(
    left: dict[str, int],
    right: dict[str, int],
) -> dict[str, int]:
    return {
        name: max(left[name], right[name])
        for name in left
    }


def _counter_delta(
    newer: dict[str, int],
    older: dict[str, int],
) -> dict[str, int]:
    if not _counters_dominate(newer, older):
        raise WorkloadLogError(
            "workload-log counters regress across the retained window"
        )
    return {
        name: newer[name] - older[name]
        for name in newer
    }


def _new_file_report(path: Path, finalized: bool) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": 0,
        "finalized": finalized,
        "valid": False,
        "torn_tail": False,
        "logical_duplicate": False,
        "counter_coverage": "missing",
        "opening_checkpoint": False,
        "terminal_checkpoint": False,
        "fresh_terminal_checkpoint": False,
        "segment_sequence": None,
        "error": None,
        "event_records": 0,
        "empty_events": 0,
        "checkpoint_records": 0,
        "captured_ids": 0,
        "original_ids": 0,
        "truncated_events": 0,
        "first_unix_ns": None,
        "last_unix_ns": None,
    }


def _inspect_file(path: Path) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    Counter[str],
    Counter[int],
    dict[str, int] | None,
    dict[str, int] | None,
    dict[str, int] | None,
]:
    return _inspect_file_bounded(
        path,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_prefix_entries=DEFAULT_MAX_PREFIX_ENTRIES,
    )


def _inspect_file_bounded(
    path: Path,
    *,
    max_file_bytes: int,
    max_prefix_entries: int,
    deadline_monotonic: float | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    Counter[str],
    Counter[int],
    dict[str, int] | None,
    dict[str, int] | None,
    dict[str, int] | None,
]:
    _check_deadline(deadline_monotonic)
    finalized = path.suffix == ".tdbrl"
    report = _new_file_report(path, finalized)
    methods: Counter[str] = Counter()
    prefixes: Counter[int] = Counter()
    aggregate_counters: dict[str, int] | None = None
    first_checkpoint_counters: dict[str, int] | None = None
    terminal_counters: dict[str, int] | None = None
    previous_checkpoint: dict[str, int] | None = None
    last_kind: int | None = None
    last_checkpoint_was_opening = False
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise WorkloadLogError("workload-log path is not a regular file")
        if status.st_size > max_file_bytes:
            raise _WorkloadLogLimitError(
                "workload-log file exceeds max_file_bytes before mmap: "
                f"{path}"
            )
        report["bytes"] = status.st_size
        if status.st_size < _HEADER_BYTES:
            raise WorkloadLogError("workload-log file is shorter than its header")
        with mmap.mmap(
            descriptor,
            status.st_size,
            access=mmap.ACCESS_READ,
        ) as mapping:
            header_view = memoryview(mapping)[:_HEADER_BYTES]
            try:
                header = _parse_header(header_view)
            finally:
                header_view.release()
            report["segment_sequence"] = header["segment_sequence"]
            offset = _HEADER_BYTES
            while offset != status.st_size:
                _check_deadline(deadline_monotonic)
                remaining = status.st_size - offset
                if remaining < _RECORD_HEADER_BYTES:
                    if finalized:
                        raise WorkloadLogError(
                            "finalized workload log has a truncated record header"
                        )
                    report["torn_tail"] = True
                    break
                header_frame = memoryview(mapping)[
                    offset : offset + _RECORD_HEADER_BYTES
                ]
                try:
                    if bytes(header_frame[:8]) != _RECORD_MAGIC:
                        raise WorkloadLogError(
                            "workload-log record magic is invalid"
                        )
                    if (
                        _u32(header_frame, 8) != _VERSION
                        or _u32(header_frame, 12)
                            != _RECORD_HEADER_BYTES
                    ):
                        raise WorkloadLogError(
                            "workload-log record version is unsupported"
                        )
                    frame_bytes = _u32(header_frame, 16)
                    kind = _u32(header_frame, 20)
                    captured = _u32(header_frame, 112)
                    if frame_bytes < _RECORD_HEADER_BYTES:
                        raise WorkloadLogError(
                            "workload-log record size is invalid"
                        )
                    if frame_bytes > _MAX_FRAME_BYTES:
                        raise WorkloadLogError(
                            "workload-log record exceeds the hard size bound"
                        )
                    if frame_bytes > remaining:
                        if finalized:
                            raise WorkloadLogError(
                                "finalized workload log has a truncated record"
                            )
                        report["torn_tail"] = True
                        break
                    if kind == _EVENT_KIND:
                        if (
                            captured > int(header["max_ids_per_event"])
                            or frame_bytes
                                != _RECORD_HEADER_BYTES + captured * 8
                        ):
                            raise WorkloadLogError(
                                "workload-log event geometry is invalid"
                            )
                    elif kind == _CHECKPOINT_KIND:
                        if frame_bytes != _RECORD_HEADER_BYTES:
                            raise WorkloadLogError(
                                "workload-log checkpoint geometry is invalid"
                            )
                    else:
                        raise WorkloadLogError(
                            "workload-log record kind is unknown"
                        )
                finally:
                    header_frame.release()

                frame = memoryview(mapping)[offset : offset + frame_bytes]
                try:
                    expected_crc = _u32(frame, _CRC_OFFSET)
                    actual_crc = _crc32c(
                        frame,
                        zero_offset=_CRC_OFFSET,
                        zero_size=_CRC_SIZE,
                    )
                    if actual_crc != expected_crc:
                        if not finalized and offset + frame_bytes == status.st_size:
                            report["torn_tail"] = True
                            break
                        raise WorkloadLogError(
                            "workload-log record CRC32C is invalid"
                        )
                    counters = _counter_snapshot(frame)
                    _validate_counter_snapshot(counters)
                    last_kind = kind

                    if kind == _CHECKPOINT_KIND:
                        checkpoint_flags = _u32(frame, 32)
                        if (
                            _u32(frame, 28)
                                != _CHECKPOINT_EXTENSION_VERSION
                            or checkpoint_flags
                                & ~_KNOWN_CHECKPOINT_FLAGS
                            or _u32(frame, 36) != 0
                            or _u64(frame, 40) != 0
                        ):
                            raise WorkloadLogError(
                                "workload-log checkpoint extension metadata "
                                "is invalid"
                            )
                        opening_checkpoint = bool(
                            checkpoint_flags
                            & _CHECKPOINT_OPENING_BASELINE_FLAG
                        )
                        first_record = offset == _HEADER_BYTES
                        if opening_checkpoint != first_record:
                            raise WorkloadLogError(
                                "workload-log segment must begin with exactly "
                                "one opening counter checkpoint"
                            )
                        checkpoint_counters = {
                            **counters,
                            **_checkpoint_counter_snapshot(frame),
                        }
                        _validate_counter_snapshot(checkpoint_counters)
                        _validate_checkpoint_counter_snapshot(
                            checkpoint_counters
                        )
                        if (
                            previous_checkpoint is not None
                            and not _counters_dominate(
                                checkpoint_counters, previous_checkpoint
                            )
                        ):
                            raise WorkloadLogError(
                                "workload-log checkpoints regress within "
                                "one file"
                            )
                        previous_checkpoint = checkpoint_counters
                        if first_checkpoint_counters is None:
                            first_checkpoint_counters = checkpoint_counters
                            report["opening_checkpoint"] = True
                        terminal_counters = checkpoint_counters
                        last_checkpoint_was_opening = opening_checkpoint
                        aggregate_counters = (
                            checkpoint_counters
                            if aggregate_counters is None
                            else _counter_component_max(
                                aggregate_counters,
                                checkpoint_counters,
                            )
                        )
                        if _u64(frame, 48) == 0:
                            raise WorkloadLogError(
                                "workload-log checkpoint timestamp is invalid"
                            )
                        report["checkpoint_records"] = (
                            int(report["checkpoint_records"]) + 1
                        )
                    else:
                        if first_checkpoint_counters is None:
                            raise WorkloadLogError(
                                "workload-log event precedes the opening "
                                "counter checkpoint"
                            )
                        last_checkpoint_was_opening = False
                        event_counters = {
                            **counters,
                            **{
                                name: 0
                                for name in _CHECKPOINT_COUNTER_NAMES
                            },
                        }
                        aggregate_counters = (
                            event_counters
                            if aggregate_counters is None
                            else _counter_component_max(
                                aggregate_counters, event_counters
                            )
                        )
                        method = _u32(frame, 28)
                        flags = _u32(frame, 32)
                        prefix = _u32(frame, 36)
                        event_sequence = _u64(frame, 40)
                        timestamp = _u64(frame, 48)
                        catalog_generation = _u64(frame, 56)
                        committed_sequence = _u64(frame, 64)
                        stream_id_low = _u64(frame, 80)
                        stream_id_high = _u64(frame, 88)
                        batch_sequence = _u64(frame, 96)
                        original = _u64(frame, 104)
                        sample_ppm = _u32(frame, 116)
                        exact_snapshot = bool(flags & _EXACT_SNAPSHOT_FLAG)
                        truncated = bool(flags & _TRUNCATED_FLAG)
                        valid_cardinality = (
                            original > 0
                            and captured > 0
                            and captured <= original
                        )
                        valid_method_snapshot = (
                            (
                                method == _GET_METHOD
                                and stream_id_low == 0
                                and stream_id_high == 0
                                and batch_sequence == 0
                                and (
                                    (
                                        exact_snapshot
                                        and catalog_generation > 0
                                    )
                                    or (
                                        not exact_snapshot
                                        and catalog_generation == 0
                                        and committed_sequence == 0
                                    )
                                )
                            )
                            or (
                                method == _STREAM_GET_METHOD
                                and exact_snapshot
                                and catalog_generation > 0
                                and (stream_id_low != 0 or stream_id_high != 0)
                            )
                        )
                        if (
                            method not in {_GET_METHOD, _STREAM_GET_METHOD}
                            or flags & ~_KNOWN_EVENT_FLAGS
                            or prefix == 0
                            or prefix > int(header["dimension"])
                            or event_sequence == 0
                            or timestamp == 0
                            or not valid_cardinality
                            or sample_ppm != header["sample_ppm"]
                            or truncated != (captured < original)
                            or not valid_method_snapshot
                            or counters["sampled_events"] == 0
                            or counters["enqueued_events"] == 0
                            or counters["observed_ids"] < original
                            or counters["captured_ids"] < captured
                            or (
                                truncated
                                and counters["truncated_events"] == 0
                            )
                        ):
                            raise WorkloadLogError(
                                "workload-log event metadata is invalid"
                            )
                        method_name = (
                            "get" if method == _GET_METHOD else "stream_get"
                        )
                        methods[method_name] += 1
                        if (
                            prefix not in prefixes
                            and len(prefixes) >= max_prefix_entries
                        ):
                            raise _WorkloadLogLimitError(
                                "workload-log distinct prefix dimensions "
                                "exceed max_prefix_entries "
                                f"({max_prefix_entries})"
                            )
                        prefixes[prefix] += 1
                        report["event_records"] = (
                            int(report["event_records"]) + 1
                        )
                        report["captured_ids"] = (
                            int(report["captured_ids"]) + captured
                        )
                        report["original_ids"] = (
                            int(report["original_ids"]) + original
                        )
                        if truncated:
                            report["truncated_events"] = (
                                int(report["truncated_events"]) + 1
                            )
                        first = report["first_unix_ns"]
                        last = report["last_unix_ns"]
                        report["first_unix_ns"] = (
                            timestamp
                            if first is None
                            else min(int(first), timestamp)
                        )
                        report["last_unix_ns"] = (
                            timestamp
                            if last is None
                            else max(int(last), timestamp)
                        )
                finally:
                    frame.release()
                offset += frame_bytes
            if first_checkpoint_counters is None:
                if finalized:
                    raise WorkloadLogError(
                        "finalized workload log has no opening counter "
                        "checkpoint"
                    )
                report["torn_tail"] = True
            terminal_checkpoint = last_kind == _CHECKPOINT_KIND
            if (
                finalized
                and terminal_checkpoint
                and last_checkpoint_was_opening
            ):
                raise WorkloadLogError(
                    "finalized workload log ends at its opening checkpoint"
                )
            fresh_terminal_checkpoint = bool(
                finalized
                and terminal_checkpoint
                and not last_checkpoint_was_opening
                and terminal_counters is not None
                and aggregate_counters is not None
                and _counters_dominate(
                    terminal_counters, aggregate_counters
                )
            )
            report["terminal_checkpoint"] = terminal_checkpoint
            report["fresh_terminal_checkpoint"] = fresh_terminal_checkpoint
            report["counter_coverage"] = (
                "complete"
                if fresh_terminal_checkpoint
                else "lower_bound"
                if aggregate_counters is not None
                else "missing"
            )
            report["valid"] = True
            return (
                report,
                header,
                methods,
                prefixes,
                aggregate_counters,
                first_checkpoint_counters,
                terminal_counters,
            )
    finally:
        os.close(descriptor)


def _storage_identity(path: Path) -> tuple[object, ...]:
    try:
        metadata = dict(NativeCollection.inspect_metadata(path))
        dtype = str(metadata["dtype"])
        dtype_code = {"float16": 1, "float32": 2}.get(dtype)
        if dtype_code is None:
            raise WorkloadLogError(
                f"storage has unsupported workload-log dtype {dtype!r}"
            )
        return (
            int(metadata["collection_id_low"]),
            int(metadata["collection_id_high"]),
            int(metadata["format_version"]),
            dtype_code,
            int(metadata["dimension"]),
            int(metadata["record_size"]),
        )
    except WorkloadLogError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise WorkloadLogError(
            f"cannot inspect workload-log storage identity {path}: {error}"
        ) from error


def _positive_bound(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadLogError(f"{name} must be a positive integer")
    return value


def _nonnegative_bound(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadLogError(f"{name} must be a nonnegative integer")
    return value


def _ratio_bound(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadLogError(f"{name} must be finite and in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise WorkloadLogError(f"{name} must be finite and in [0, 1]")
    return parsed


def inspect_workload_logs(
    logs: str | os.PathLike[str],
    *,
    storage: str | os.PathLike[str] | None = None,
    max_drop_ratio: float = 0.05,
    max_row_drop_ratio: float = 0.05,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_report_entries: int = DEFAULT_MAX_REPORT_ENTRIES,
    max_prefix_entries: int = DEFAULT_MAX_PREFIX_ENTRIES,
    _deadline_monotonic: float | None = None,
) -> dict[str, object]:
    """Validate and summarize one dedicated read-workload log directory."""

    max_drop_ratio = _ratio_bound(max_drop_ratio, "max_drop_ratio")
    max_row_drop_ratio = _ratio_bound(
        max_row_drop_ratio, "max_row_drop_ratio"
    )
    max_files = _positive_bound(max_files, "max_files")
    if max_files > DEFAULT_MAX_FILES:
        raise WorkloadLogError(
            f"max_files must not exceed the hard {DEFAULT_MAX_FILES}-entry "
            "inventory ceiling"
        )
    max_total_bytes = _positive_bound(max_total_bytes, "max_total_bytes")
    max_file_bytes = _positive_bound(max_file_bytes, "max_file_bytes")
    max_report_entries = _nonnegative_bound(
        max_report_entries, "max_report_entries"
    )
    max_prefix_entries = _positive_bound(
        max_prefix_entries, "max_prefix_entries"
    )
    _check_deadline(_deadline_monotonic)
    directory = Path(logs).expanduser()
    try:
        directory_status = directory.stat()
    except OSError as error:
        raise WorkloadLogError(
            f"cannot inspect workload-log directory {directory}: {error}"
        ) from error
    if not stat.S_ISDIR(directory_status.st_mode):
        raise WorkloadLogError(f"workload-log path is not a directory: {directory}")

    paths: list[tuple[Path, int]] = []
    preflight_total_bytes = 0
    scanned_entries = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                _check_deadline(_deadline_monotonic)
                scanned_entries += 1
                directory_entry_limit = min(
                    DEFAULT_MAX_FILES, max_files + 16
                )
                if scanned_entries > directory_entry_limit:
                    raise _WorkloadLogLimitError(
                        "workload-log directory entry count exceeds the "
                        f"bounded allowance ({directory_entry_limit})"
                    )
                path = Path(entry.path)
                if path.suffix not in _SUFFIXES:
                    continue
                try:
                    entry_status = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise WorkloadLogError(
                        f"cannot stat workload-log candidate {path}: {error}"
                    ) from error
                if not stat.S_ISREG(entry_status.st_mode):
                    continue
                if len(paths) >= max_files:
                    raise _WorkloadLogLimitError(
                        "workload-log file count exceeds max_files "
                        f"({max_files})"
                    )
                file_bytes = int(entry_status.st_size)
                if file_bytes > max_file_bytes:
                    raise _WorkloadLogLimitError(
                        "workload-log file exceeds max_file_bytes "
                        f"({max_file_bytes}): {path}"
                    )
                preflight_total_bytes += file_bytes
                if preflight_total_bytes > max_total_bytes:
                    raise _WorkloadLogLimitError(
                        "workload-log bytes exceed max_total_bytes "
                        f"({max_total_bytes})"
                    )
                paths.append((path, file_bytes))
    except WorkloadLogError:
        raise
    except OSError as error:
        raise WorkloadLogError(
            f"cannot enumerate workload-log directory {directory}: {error}"
        ) from error
    paths.sort(key=lambda entry: entry[0].name)

    reports: list[dict[str, object]] = []
    methods: Counter[str] = Counter()
    prefixes: Counter[int] = Counter()
    identities: set[tuple[object, ...]] = set()
    configurations: set[tuple[int, ...]] = set()
    logical_files: set[tuple[object, ...]] = set()
    counter_histories: dict[
        tuple[int, int],
        list[
            tuple[
                int,
                dict[str, int],
                dict[str, int] | None,
                dict[str, int] | None,
                dict[str, int] | None,
            ]
        ],
    ] = {}
    canonical_header: dict[str, object] | None = None
    invalid_finalized = 0
    invalid_open = 0
    valid_files = 0
    torn_open = 0
    logical_duplicates = 0
    counter_snapshot_files = 0
    actual_total_bytes = 0
    event_records = 0
    empty_events = 0
    captured_ids = 0
    original_ids = 0
    truncated_events = 0
    first_unix_ns: int | None = None
    last_unix_ns: int | None = None

    for path, preflight_bytes in paths:
        _check_deadline(_deadline_monotonic)
        remaining_bytes = max_total_bytes - actual_total_bytes
        if remaining_bytes <= 0:
            raise _WorkloadLogLimitError(
                "workload-log bytes reached max_total_bytes before all files "
                "were inspected"
            )
        try:
            (
                report,
                header,
                file_methods,
                file_prefixes,
                aggregate_counters,
                first_checkpoint_counters,
                latest_checkpoint_counters,
            ) = _inspect_file_bounded(
                path,
                max_file_bytes=min(max_file_bytes, remaining_bytes),
                max_prefix_entries=max_prefix_entries,
                deadline_monotonic=_deadline_monotonic,
            )
        except _WorkloadLogLimitError:
            raise
        except (OSError, WorkloadLogError) as error:
            report = _new_file_report(path, path.suffix == ".tdbrl")
            try:
                current_status = path.stat(follow_symlinks=False)
                if stat.S_ISREG(current_status.st_mode):
                    report["bytes"] = min(
                        int(current_status.st_size),
                        min(max_file_bytes, remaining_bytes),
                    )
            except OSError:
                report["bytes"] = preflight_bytes
            report["error"] = str(error)
            if path.suffix == ".tdbrl":
                invalid_finalized += 1
            else:
                invalid_open += 1
        else:
            assert header is not None
            valid_files += 1
            if bool(report["torn_tail"]):
                torn_open += 1
            identity = _header_identity(header)
            identities.add(identity)
            configuration = (
                int(header["sample_ppm"]),
                int(header["max_ids_per_event"]),
                int(header["buffer_bytes"]),
                int(header["segment_bytes"]),
                int(header["rotate_seconds"]),
                int(header["retention_bytes"]),
                int(header["retention_seconds"]),
                int(header["minimum_free_bytes"]),
                int(header["max_write_bytes_per_second"]),
                int(header["max_retained_files"]),
            )
            configurations.add(configuration)
            logical_key = (
                int(header["instance_id_low"]),
                int(header["instance_id_high"]),
                int(header["segment_sequence"]),
            )
            if logical_key in logical_files:
                report["logical_duplicate"] = True
                logical_duplicates += 1
            else:
                logical_files.add(logical_key)
                if canonical_header is None:
                    canonical_header = header
                methods.update(file_methods)
                new_prefixes = set(file_prefixes).difference(prefixes)
                if (
                    len(prefixes) + len(new_prefixes)
                    > max_prefix_entries
                ):
                    raise _WorkloadLogLimitError(
                        "workload-log distinct prefix dimensions exceed "
                        f"max_prefix_entries ({max_prefix_entries})"
                    )
                prefixes.update(file_prefixes)
                event_records += int(report["event_records"])
                empty_events += int(report["empty_events"])
                captured_ids += int(report["captured_ids"])
                original_ids += int(report["original_ids"])
                truncated_events += int(report["truncated_events"])
                if report["first_unix_ns"] is not None:
                    value = int(report["first_unix_ns"])
                    first_unix_ns = (
                        value
                        if first_unix_ns is None
                        else min(first_unix_ns, value)
                    )
                if report["last_unix_ns"] is not None:
                    value = int(report["last_unix_ns"])
                    last_unix_ns = (
                        value
                        if last_unix_ns is None
                        else max(last_unix_ns, value)
                    )
                instance = (
                    int(header["instance_id_low"]),
                    int(header["instance_id_high"]),
                )
                if aggregate_counters is not None:
                    counter_snapshot_files += 1
                counter_histories.setdefault(instance, []).append(
                    (
                        int(header["segment_sequence"]),
                        (
                            aggregate_counters
                            if aggregate_counters is not None
                            else {
                                name: 0 for name in _ALL_COUNTER_NAMES
                            }
                        ),
                        first_checkpoint_counters,
                        latest_checkpoint_counters,
                        (
                            latest_checkpoint_counters
                            if bool(report["fresh_terminal_checkpoint"])
                            else None
                        ),
                    )
                )
        actual_total_bytes += int(report["bytes"])
        if actual_total_bytes > max_total_bytes:
            raise _WorkloadLogLimitError(
                "workload-log bytes exceed max_total_bytes after opening a "
                "concurrently changed file"
            )
        if len(reports) < max_report_entries:
            reports.append(report)

    warnings: list[str] = []
    reasons: list[str] = []
    if not paths:
        reasons.append("no_log_files")
    if not identities:
        reasons.append("no_valid_log_headers")
    if len(identities) > 1:
        reasons.append("mixed_collection_or_schema_identity")
    if logical_duplicates:
        reasons.append("duplicate_logical_files")
    if invalid_finalized:
        reasons.append("corrupt_finalized_files")
    if invalid_open:
        reasons.append("corrupt_open_files")
    if torn_open:
        warnings.append("open_file_has_torn_tail")
    if len(configurations) > 1:
        reasons.append("mixed_logging_configuration")

    expected_identity: tuple[object, ...] | None = None
    if storage is not None:
        expected_identity = _storage_identity(Path(storage).expanduser())
        if identities and expected_identity not in identities:
            reasons.append("storage_identity_mismatch")

    counters = Counter[str]()
    incomplete_counter_instances = 0
    regressed_counter_instances = 0
    sequence_gap_instances = 0
    saturated_counter_instances = 0
    for history in counter_histories.values():
        history.sort(key=lambda entry: entry[0])
        previous_checkpoint: dict[str, int] | None = None
        previous_segment_sequence: int | None = None
        latest_complete_terminal: dict[str, int] | None = None
        retained_window_opening: dict[str, int] | None = None
        best: dict[str, int] | None = None
        regressed = False
        sequence_gap = False
        saturated = False
        complete = True
        for (
            _segment_sequence,
            aggregate,
            first_checkpoint,
            latest_checkpoint,
            complete_terminal,
        ) in history:
            if (
                previous_segment_sequence is not None
                and _segment_sequence != previous_segment_sequence + 1
            ):
                sequence_gap = True
            previous_segment_sequence = _segment_sequence
            best = (
                aggregate
                if best is None
                else _counter_component_max(best, aggregate)
            )
            if first_checkpoint is not None:
                if retained_window_opening is None:
                    retained_window_opening = first_checkpoint
                if (
                    previous_checkpoint is not None
                    and first_checkpoint != previous_checkpoint
                ):
                    regressed = True
            if latest_checkpoint is not None:
                previous_checkpoint = latest_checkpoint
            if complete_terminal is None:
                complete = False
            else:
                latest_complete_terminal = complete_terminal
        if (
            latest_complete_terminal is None
            or not _terminal_partitions_are_complete(
                latest_complete_terminal
            )
        ):
            complete = False
        if sequence_gap:
            complete = False
        assert best is not None
        if any(value == _UINT64_MAX for value in best.values()):
            saturated = True
        if (
            retained_window_opening is not None
            and any(
                value == _UINT64_MAX
                for value in retained_window_opening.values()
            )
        ):
            saturated = True
        if (
            retained_window_opening is None
            or not _counters_dominate(best, retained_window_opening)
        ):
            regressed = True
        else:
            retained_window_end = (
                latest_complete_terminal
                if complete and latest_complete_terminal is not None
                else best
            )
            if not _counters_dominate(
                retained_window_end, retained_window_opening
            ):
                regressed = True
            else:
                if any(
                    value == _UINT64_MAX
                    for value in retained_window_end.values()
                ):
                    saturated = True
                counters.update(
                    _counter_delta(
                        retained_window_end, retained_window_opening
                    )
                )
        if regressed:
            regressed_counter_instances += 1
        if sequence_gap:
            sequence_gap_instances += 1
        if saturated:
            saturated_counter_instances += 1
        if not complete:
            incomplete_counter_instances += 1
    if regressed_counter_instances:
        reasons.append("counter_regression")
    if sequence_gap_instances:
        reasons.append("segment_sequence_gap")
    if saturated_counter_instances:
        reasons.append("counter_saturation")
    if incomplete_counter_instances:
        reasons.append("incomplete_counter_coverage")
    if event_records != 0 and counter_snapshot_files == 0:
        reasons.append("missing_counter_coverage")

    dropped_events = sum(
        counters[name]
        for name in (
            "dropped_busy_events",
            "dropped_full_events",
            "dropped_runtime_failure_events",
            "dropped_minimum_free_events",
            "dropped_io_budget_events",
        )
    )
    completed_events = counters["written_events"] + dropped_events
    event_drop_ratio: float | None = (
        dropped_events / completed_events
        if completed_events != 0
        else None
    )
    dropped_ids = sum(
        counters[name]
        for name in (
            "dropped_busy_ids",
            "dropped_full_ids",
            "dropped_runtime_failure_ids",
            "dropped_minimum_free_ids",
            "dropped_io_budget_ids",
        )
    )
    completed_ids = counters["written_ids"] + dropped_ids
    row_drop_ratio: float | None = (
        dropped_ids / completed_ids if completed_ids != 0 else None
    )
    if counter_snapshot_files == 0:
        warnings.append("no_counter_snapshot")
    elif (
        event_drop_ratio is not None
        and event_drop_ratio > max_drop_ratio
    ):
        reasons.append("drop_ratio_exceeds_limit")
    if (
        row_drop_ratio is not None
        and row_drop_ratio > max_row_drop_ratio
    ):
        reasons.append("row_drop_ratio_exceeds_limit")

    if captured_ids == 0:
        reasons.append("no_point_read_events")

    collection = (
        {
            key: canonical_header[key]
            for key in (
                "collection_id",
                "collection_id_low",
                "collection_id_high",
                "format_version",
                "dtype",
                "dimension",
                "record_size",
            )
        }
        if canonical_header is not None
        else None
    )
    configuration = (
        {
            key: canonical_header[key]
            for key in (
                "sample_ppm",
                "max_ids_per_event",
                "buffer_bytes",
                "segment_bytes",
                "rotate_seconds",
                "retention_bytes",
                "retention_seconds",
                "minimum_free_bytes",
                "max_write_bytes_per_second",
                "max_retained_files",
            )
        }
        if canonical_header is not None
        else None
    )
    return {
        "format": "throwdb-workload-log-inspection",
        "version": 1,
        "logs": str(directory.resolve()),
        "limits": {
            "max_files": max_files,
            "max_total_bytes": max_total_bytes,
            "max_file_bytes": max_file_bytes,
            "max_report_entries": max_report_entries,
            "max_prefix_entries": max_prefix_entries,
        },
        "collection": collection,
        "configuration": configuration,
        "files": {
            "discovered": len(paths),
            "valid": valid_files,
            "finalized": sum(path.suffix == ".tdbrl" for path, _ in paths),
            "open": sum(path.suffix == ".open" for path, _ in paths),
            "invalid_finalized": invalid_finalized,
            "invalid_open": invalid_open,
            "torn_open": torn_open,
            "logical_duplicates": logical_duplicates,
            "bytes": actual_total_bytes,
            "reports": reports,
            "reports_omitted": len(paths) - len(reports),
        },
        "events": {
            "records": event_records,
            "empty_events": empty_events,
            "methods": dict(sorted(methods.items())),
            "prefix_dimensions": {
                str(key): value for key, value in sorted(prefixes.items())
            },
            "captured_ids": captured_ids,
            "original_ids": original_ids,
            "captured_fraction": (
                captured_ids / original_ids if original_ids else None
            ),
            "truncated_events": truncated_events,
            "first_unix_ns": first_unix_ns,
            "last_unix_ns": last_unix_ns,
        },
        "counters": {
            **dict(sorted(counters.items())),
            "process_instances": len(counter_histories),
            "complete_instances": (
                len(counter_histories) - incomplete_counter_instances
            ),
            "incomplete_instances": incomplete_counter_instances,
            "regressed_instances": regressed_counter_instances,
            "sequence_gap_instances": sequence_gap_instances,
            "saturated_instances": saturated_counter_instances,
            "dropped_events": dropped_events,
            "completed_event_outcomes": completed_events,
            "event_drop_ratio": event_drop_ratio,
            "drop_ratio": event_drop_ratio,
            "dropped_ids": dropped_ids,
            "completed_row_outcomes": completed_ids,
            "row_drop_ratio": row_drop_ratio,
        },
        "configuration_variants": len(configurations),
        "quality": {
            "suitable": not reasons,
            "reasons": sorted(set(reasons)),
            "warnings": sorted(set(warnings)),
            "max_drop_ratio": max_drop_ratio,
            "max_row_drop_ratio": max_row_drop_ratio,
        },
    }


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_PREFIX_ENTRIES",
    "DEFAULT_MAX_REPORT_ENTRIES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "WorkloadLogError",
    "inspect_workload_logs",
]
