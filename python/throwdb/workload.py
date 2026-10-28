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
from dataclasses import dataclass, field
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

_V2_HEADER_MAGIC: Final = b"TDBRLOG2"
_V2_RECORD_MAGIC: Final = b"TDBRREC2"
_V2_VERSION: Final = 2
_V2_HEADER_USED_BYTES: Final = 232
_V2_REQUEST_BEGIN_KIND: Final = 1
_V2_ID_CHUNK_KIND: Final = 2
_V2_BATCH_KIND: Final = 3
_V2_REQUEST_END_KIND: Final = 4
_V2_CHECKPOINT_KIND: Final = 5
_V2_REQUEST_BEGIN_EXACT_FLAG: Final = 1 << 0
_V2_REQUEST_BEGIN_DECLARED_IDS_FLAG: Final = 1 << 1
_V2_REQUEST_END_COMPLETE_FLAG: Final = 1 << 0
_V2_CHECKPOINT_OPENING_FLAG: Final = 1 << 0
_V2_MAX_IDS_PER_CHUNK: Final = 8_192
_V2_BATCH_FRAME_BYTES: Final = 384
_V2_BACKENDS: Final = {
    0: "unknown",
    1: "thread_pool",
    2: "io_uring",
}
_V2_REQUEST_STATUSES: Final = {
    1: "success",
    2: "cancelled",
    3: "deadline_exceeded",
    4: "failed",
    5: "client_closed",
    6: "shutdown",
}
_V2_BATCH_METRIC_NAMES: Final = (
    "unique_segments",
    "physical_requests",
    "merged_ranges",
    "requested_bytes",
    "planned_bytes",
    "completed_bytes",
    "gap_bytes",
    "overread_bytes",
    "pages_touched",
    "page_runs",
    "chunks_1mib_touched",
    "chunk_runs",
    "physical_discontinuities",
    "physical_inversions",
    "maximum_range_bytes",
    "mean_range_bytes",
)
_V2_BATCH_TIMING_NAMES: Final = (
    "total_ns",
    "lookup_ns",
    "sort_ns",
    "plan_ns",
    "scheduler_ns",
    "io_ns",
    "scatter_ns",
    "encode_ns",
    "write_ns",
    "logger_queue_ns",
)

# These bounds are deliberately higher than the native server defaults
# (256-MiB segments and 50-GiB retention), but finite.  Operators inspecting a
# deliberately larger retained history must opt in explicitly.
DEFAULT_MAX_FILES: Final = 65_536
DEFAULT_MAX_TOTAL_BYTES: Final = 64 << 30
DEFAULT_MAX_FILE_BYTES: Final = 4 << 30
DEFAULT_MAX_REPORT_ENTRIES: Final = 1_024
DEFAULT_MAX_PREFIX_ENTRIES: Final = 4_096
DEFAULT_MAX_ACTIVE_REQUESTS: Final = 4_096


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

_V2_CHECKPOINT_COUNTER_NAMES: Final = (
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
_V2_FNV_PRIME: Final = 0x100000001B3
_V2_FNV_LOW_INITIAL: Final = 0xCBF29CE484222325
_V2_FNV_HIGH_INITIAL: Final = 0x84222325CBF29CE4


@dataclass(slots=True)
class _V2RequestState:
    method: int
    prefix_dimensions: int
    exact_snapshot: bool
    declared_ids_known: bool
    declared_ids: int
    begin_unix_ns: int
    begin_steady_ns: int
    expected_logical_offset: int = 0
    expected_batch_sequence: int = 0
    current_batch_sequence: int | None = None
    current_batch_offset: int = 0
    current_batch_rows: int = 0
    expected_chunk_sequence: int = 0
    previous_delivery_sequence: int | None = None
    id_chunks: int = 0
    batches: int = 0
    response_bytes: int = 0
    batch_rows: int = 0
    batch_size_buckets: Counter[str] = field(default_factory=Counter)
    batch_backends: Counter[str] = field(default_factory=Counter)
    batch_timings: Counter[str] = field(default_factory=Counter)
    batch_metrics: Counter[str] = field(default_factory=Counter)
    digest_low: int = _V2_FNV_LOW_INITIAL
    digest_high: int = _V2_FNV_HIGH_INITIAL


@dataclass(slots=True)
class _V2CensoredState:
    pass


@dataclass(slots=True)
class _V2PendingAggregate:
    complete_requests: int = 0
    non_success_outcomes: int = 0
    incomplete_requests: int = 0
    interior_incomplete_requests: int = 0
    id_chunk_records: int = 0
    ids_in_chunks: int = 0
    batch_records: int = 0
    batch_rows: int = 0
    request_time_ns: int = 0
    methods: Counter[str] = field(default_factory=Counter)
    statuses: Counter[str] = field(default_factory=Counter)
    prefixes: Counter[int] = field(default_factory=Counter)
    batch_size_buckets: Counter[str] = field(default_factory=Counter)
    batch_backends: Counter[str] = field(default_factory=Counter)
    batch_timings: Counter[str] = field(default_factory=Counter)
    batch_metrics: Counter[str] = field(default_factory=Counter)
    request_reports: list[dict[str, object]] = field(default_factory=list)
    request_reports_omitted: int = 0
    delivery_sequence_count: int = 0
    delivery_sequence_min: int | None = None
    delivery_sequence_max: int | None = None
    delivery_physical_inversions: int = 0



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


def _parse_v2_header(data: memoryview) -> dict[str, object]:
    if len(data) != _HEADER_BYTES:
        raise WorkloadLogError("workload-log v2 header is truncated")
    if bytes(data[:8]) != _V2_HEADER_MAGIC:
        raise WorkloadLogError("workload-log v2 header magic is invalid")
    if (
        _u32(data, 8) != _V2_VERSION
        or _u32(data, 12) != _HEADER_BYTES
        or _u32(data, 16) != _V2_HEADER_USED_BYTES
    ):
        raise WorkloadLogError("workload-log v2 header version is unsupported")
    if (
        _u32(data, 20) != 0
        or any(data[28:32])
        or any(data[_V2_HEADER_USED_BYTES:])
    ):
        raise WorkloadLogError("workload-log v2 header has unknown flags")
    if (
        _crc32c(data, zero_offset=_CRC_OFFSET, zero_size=_CRC_SIZE)
        != _u32(data, _CRC_OFFSET)
    ):
        raise WorkloadLogError("workload-log v2 header CRC32C is invalid")

    collection_id_low = _u64(data, 32)
    collection_id_high = _u64(data, 40)
    format_version = _u32(data, 48)
    dtype_code = _u32(data, 52)
    dimension = _u32(data, 56)
    record_size = _u32(data, 60)
    instance_id_low = _u64(data, 64)
    instance_id_high = _u64(data, 72)
    created_unix_ns = _u64(data, 80)
    id_chunk_rows = _u32(data, 88)
    max_active_requests = _u32(data, 92)
    buffer_bytes = _u64(data, 96)
    segment_bytes = _u64(data, 104)
    rotate_seconds = _u64(data, 112)
    retention_bytes = _u64(data, 120)
    retention_seconds = _u64(data, 128)
    minimum_free_bytes = _u64(data, 136)
    segment_sequence = _u64(data, 152)
    max_write_bytes_per_second = _u64(data, 160)
    max_retained_files = _u64(data, 168)
    reader_backend = _u32(data, 176)
    io_workers = _u32(data, 180)
    io_queue_capacity = _u32(data, 184)
    io_ranges_per_task = _u32(data, 188)
    max_read_range_bytes = _u64(data, 192)
    max_gap_bytes = _u64(data, 200)
    io_uring_queue_depth = _u32(data, 208)
    io_uring_workers = _u32(data, 212)
    metric_page_bytes = _u32(data, 216)
    metric_chunk_bytes = _u32(data, 220)
    tuning_explicit_overrides = _u32(data, 224)
    reserved = _u32(data, 228)
    item_size = {1: 2, 2: 4}.get(dtype_code)
    maximum_frame = max(
        _V2_BATCH_FRAME_BYTES,
        _RECORD_HEADER_BYTES + id_chunk_rows * 8,
    )
    maximum_transaction = (
        _HEADER_BYTES + maximum_frame + 2 * _RECORD_HEADER_BYTES
    )
    if (
        (collection_id_low == 0 and collection_id_high == 0)
        or (instance_id_low == 0 and instance_id_high == 0)
        or created_unix_ns == 0
        or format_version == 0
        or item_size is None
        or dimension == 0
        or record_size != dimension * item_size
        or id_chunk_rows < 1
        or id_chunk_rows > _V2_MAX_IDS_PER_CHUNK
        or max_active_requests < 1
        or max_active_requests > 65_536
        or buffer_bytes < 2 * maximum_frame
        or segment_bytes < maximum_transaction
        or rotate_seconds < 1
        or rotate_seconds > _INT64_MAX
        or retention_bytes < segment_bytes
        or retention_seconds < 1
        or max_write_bytes_per_second < maximum_transaction
        or max_retained_files < 1
        or max_retained_files > _MAX_RETAINED_FILES
        or reader_backend not in _V2_BACKENDS
        or metric_page_bytes == 0
        or metric_page_bytes & (metric_page_bytes - 1) != 0
        or metric_chunk_bytes == 0
        or metric_chunk_bytes & (metric_chunk_bytes - 1) != 0
        or metric_chunk_bytes < metric_page_bytes
        or tuning_explicit_overrides & ~0xFF
        or reserved != 0
        or (
            reader_backend in (1, 2)
            and (
                io_ranges_per_task < 1
                or io_ranges_per_task > (1 << 20)
                or max_read_range_bytes < record_size
                or io_uring_queue_depth < 1
                or io_uring_queue_depth > 4_096
                or io_uring_workers < 1
                or io_uring_workers > 64
                or io_uring_queue_depth < io_uring_workers
            )
        )
        or (
            reader_backend == 1
            and (
                io_workers > 1_024
                or io_queue_capacity > (1 << 20)
                or (io_workers == 0 and io_queue_capacity != 0)
            )
        )
        or (
            reader_backend == 2
            and (
                io_workers != 0
                or io_queue_capacity != 0
            )
        )
    ):
        raise WorkloadLogError("workload-log v2 header configuration is invalid")
    return {
        "log_version": _V2_VERSION,
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
        "id_chunk_rows": id_chunk_rows,
        "max_active_requests": max_active_requests,
        "buffer_bytes": buffer_bytes,
        "segment_bytes": segment_bytes,
        "rotate_seconds": rotate_seconds,
        "retention_bytes": retention_bytes,
        "retention_seconds": retention_seconds,
        "minimum_free_bytes": minimum_free_bytes,
        "process_id": _u64(data, 144),
        "segment_sequence": segment_sequence,
        "max_write_bytes_per_second": max_write_bytes_per_second,
        "max_retained_files": max_retained_files,
        "reader_backend_code": reader_backend,
        "reader_backend": _V2_BACKENDS[reader_backend],
        "io_workers": io_workers,
        "io_queue_capacity": io_queue_capacity,
        "io_ranges_per_task": io_ranges_per_task,
        "max_read_range_bytes": max_read_range_bytes,
        "max_gap_bytes": max_gap_bytes,
        "io_uring_queue_depth": io_uring_queue_depth,
        "io_uring_workers": io_uring_workers,
        "metric_page_bytes": metric_page_bytes,
        "metric_chunk_bytes": metric_chunk_bytes,
        "tuning_explicit_overrides": tuning_explicit_overrides,
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


class _V2LifecycleError(WorkloadLogError):
    """A v2 lifecycle cannot be reconstructed exactly."""


def _v2_fnv_update(state: int, data: memoryview) -> int:
    native = getattr(_native, "_fnv1a64", None)
    if native is not None:
        return int(native(data, state))
    value = state
    for byte in data:
        value = ((value ^ byte) * _V2_FNV_PRIME) & _UINT64_MAX
    return value


def _v2_checkpoint_counters(frame: memoryview) -> dict[str, int]:
    if (
        len(frame) != _RECORD_HEADER_BYTES
        or _u32(frame, 28) != 1
        or _u32(frame, 32) & ~_V2_CHECKPOINT_OPENING_FLAG
        or _u32(frame, 36) != 0
        or _u64(frame, 40) != 0
        or _u64(frame, 48) == 0
    ):
        raise WorkloadLogError("workload-log v2 checkpoint metadata is invalid")
    counters = {
        name: _u64(frame, 56 + index * 8)
        for index, name in enumerate(_V2_CHECKPOINT_COUNTER_NAMES)
    }
    if (
        counters["observed_requests"] < counters["begun_requests"]
        or counters["observed_requests"]
            < counters["completed_requests"]
        or counters["completed_requests"]
            != _saturating_sum(
                counters["complete_requests"],
                counters["incomplete_requests"],
            )
        or not (
            counters["observed_records"]
            >= counters["enqueued_records"]
            >= counters["written_records"]
        )
        or not (
            counters["observed_ids"]
            >= counters["enqueued_ids"]
            >= counters["written_ids"]
        )
        or not (
            counters["observed_chunks"]
            >= counters["enqueued_chunks"]
            >= counters["written_chunks"]
        )
        or not (
            counters["observed_batches"]
            >= counters["enqueued_batches"]
            >= counters["written_batches"]
        )
    ):
        raise WorkloadLogError(
            "workload-log v2 checkpoint counters are inconsistent"
        )
    return counters


def _v2_terminal_partitions_complete(
    counters: dict[str, int],
) -> bool:
    admission_record_drops = _saturating_sum(
        counters["dropped_full_records"],
        counters["dropped_runtime_records"],
        counters["dropped_minimum_free_records"],
    )
    admission_id_drops = _saturating_sum(
        counters["dropped_full_ids"],
        counters["dropped_runtime_ids"],
        counters["dropped_minimum_free_ids"],
    )
    return (
        counters["observed_requests"]
        == counters["completed_requests"]
        and counters["observed_records"]
        == _saturating_sum(
            counters["enqueued_records"],
            admission_record_drops,
        )
        and counters["enqueued_records"]
        == _saturating_sum(
            counters["written_records"],
            counters["dropped_io_budget_records"],
        )
        and counters["observed_ids"]
        == _saturating_sum(
            counters["enqueued_ids"],
            admission_id_drops,
        )
        and counters["enqueued_ids"]
        == _saturating_sum(
            counters["written_ids"],
            counters["dropped_io_budget_ids"],
        )
    )


def _v2_batch_size_bucket(rows: int) -> str:
    for boundary in (1, 16, 256, 4_096, 65_536, 1_048_576):
        if rows <= boundary:
            return f"le_{boundary}"
    return "gt_1048576"


class _V2Accumulator:
    def __init__(
        self,
        *,
        max_active_requests: int,
        max_report_entries: int,
        max_prefix_entries: int,
    ) -> None:
        self.max_active_requests = max_active_requests
        self.max_report_entries = max_report_entries
        self.max_prefix_entries = max_prefix_entries
        self.active: dict[
            tuple[int, int, int, int], _V2RequestState
        ] = {}
        self.left_censored_active: dict[
            tuple[int, int, int, int], _V2CensoredState
        ] = {}
        self.left_censored_allowance: dict[tuple[int, int], int] = {}
        self.active_by_instance: Counter[tuple[int, int]] = Counter()
        self.max_parallel_requests = 0
        self.methods: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.prefixes: Counter[int] = Counter()
        self.batch_size_buckets: Counter[str] = Counter()
        self.batch_backends: Counter[str] = Counter()
        self.batch_timings: Counter[str] = Counter()
        self.batch_metrics: Counter[str] = Counter()
        self.request_reports: list[dict[str, object]] = []
        self.request_reports_omitted = 0
        self.begun_requests = 0
        self.complete_requests = 0
        self.non_success_outcomes = 0
        self.incomplete_requests = 0
        self.left_censored_requests = 0
        self.right_censored_requests = 0
        self.interior_incomplete_requests = 0
        self.id_chunk_records = 0
        self.ids_in_chunks = 0
        self.batch_records = 0
        self.batch_rows = 0
        self.request_time_ns = 0
        self.first_unix_ns: int | None = None
        self.last_unix_ns: int | None = None
        self.last_lifecycle_sequence: dict[tuple[int, int], int] = {}
        self.last_delivery_sequence: dict[tuple[int, int], int] = {}
        self.sequence_gap_instances: set[tuple[int, int]] = set()
        self.pending_sequence_gap_instances: set[tuple[int, int]] = set()
        self.delivery_order_instances: set[tuple[int, int]] = set()
        self.pending_delivery_order_instances: set[
            tuple[int, int]
        ] = set()
        self.delivery_sequence_count = 0
        self.delivery_sequence_min: int | None = None
        self.delivery_sequence_max: int | None = None
        self.delivery_physical_inversions = 0
        self.pending: dict[
            tuple[int, int], _V2PendingAggregate
        ] = {}
        # Pending aggregates may span many process instances until their
        # terminal counter histories are qualified. Reserve these two
        # variable-size structures globally, not once per instance.
        self.pending_prefix_references: Counter[int] = Counter()
        self.pending_request_report_entries = 0

    @staticmethod
    def _instance(header: dict[str, object]) -> tuple[int, int]:
        return (
            int(header["instance_id_low"]),
            int(header["instance_id_high"]),
        )

    @staticmethod
    def _request_key(
        header: dict[str, object], frame: memoryview
    ) -> tuple[int, int, int, int]:
        instance = _V2Accumulator._instance(header)
        request_low = _u64(frame, 64)
        request_high = _u64(frame, 72)
        if request_low == 0 and request_high == 0:
            raise _V2LifecycleError(
                "workload-log v2 lifecycle record has a zero request token"
            )
        return (*instance, request_low, request_high)

    def set_left_censored_allowance(
        self,
        instance: tuple[int, int],
        opening: dict[str, int],
    ) -> None:
        if instance not in self.left_censored_allowance:
            allowance = (
                opening["observed_requests"]
                - opening["completed_requests"]
            )
            self.left_censored_allowance[instance] = allowance

    def _check_lifecycle_sequence(
        self, instance: tuple[int, int], sequence: int
    ) -> None:
        if sequence == 0:
            raise _V2LifecycleError(
                "workload-log v2 lifecycle sequence must be nonzero"
            )
        previous = self.last_lifecycle_sequence.get(instance)
        if previous is not None and sequence != previous + 1:
            self.pending_sequence_gap_instances.add(instance)
        self.last_lifecycle_sequence[instance] = sequence

    def commit_checkpoint(self, instance: tuple[int, int]) -> None:
        if instance in self.pending_sequence_gap_instances:
            self.sequence_gap_instances.add(instance)
            self.pending_sequence_gap_instances.discard(instance)
        if instance in self.pending_delivery_order_instances:
            self.delivery_order_instances.add(instance)
            self.pending_delivery_order_instances.discard(instance)
        pending = self.pending.pop(instance, None)
        if pending is None:
            return
        self._release_pending_reservations(pending)
        new_prefixes = set(pending.prefixes).difference(self.prefixes)
        if len(self.prefixes) + len(new_prefixes) > (
            self.max_prefix_entries
        ):
            raise _WorkloadLogLimitError(
                "workload-log distinct prefix dimensions exceed "
                f"max_prefix_entries ({self.max_prefix_entries})"
            )
        self.complete_requests += pending.complete_requests
        self.non_success_outcomes += pending.non_success_outcomes
        self.incomplete_requests += pending.incomplete_requests
        self.interior_incomplete_requests += (
            pending.interior_incomplete_requests
        )
        self.id_chunk_records += pending.id_chunk_records
        self.ids_in_chunks += pending.ids_in_chunks
        self.batch_records += pending.batch_records
        self.batch_rows += pending.batch_rows
        self.request_time_ns += pending.request_time_ns
        self.methods.update(pending.methods)
        self.statuses.update(pending.statuses)
        self.prefixes.update(pending.prefixes)
        self.batch_size_buckets.update(pending.batch_size_buckets)
        self.batch_backends.update(pending.batch_backends)
        self.batch_timings.update(pending.batch_timings)
        self.batch_metrics.update(pending.batch_metrics)
        available = self.max_report_entries - len(self.request_reports)
        if available > 0:
            self.request_reports.extend(
                pending.request_reports[:available]
            )
        self.request_reports_omitted += (
            pending.request_reports_omitted
            + max(0, len(pending.request_reports) - available)
        )
        self.delivery_sequence_count += pending.delivery_sequence_count
        if pending.delivery_sequence_min is not None:
            self.delivery_sequence_min = (
                pending.delivery_sequence_min
                if self.delivery_sequence_min is None
                else min(
                    self.delivery_sequence_min,
                    pending.delivery_sequence_min,
                )
            )
        if pending.delivery_sequence_max is not None:
            self.delivery_sequence_max = (
                pending.delivery_sequence_max
                if self.delivery_sequence_max is None
                else max(
                    self.delivery_sequence_max,
                    pending.delivery_sequence_max,
                )
            )
        self.delivery_physical_inversions += (
            pending.delivery_physical_inversions
        )

    def _record_pending_prefix(
        self,
        pending: _V2PendingAggregate,
        prefix: int,
    ) -> None:
        if prefix not in pending.prefixes:
            if (
                prefix not in self.prefixes
                and prefix not in self.pending_prefix_references
                and (
                    len(self.prefixes)
                    + sum(
                        existing not in self.prefixes
                        for existing in self.pending_prefix_references
                    )
                    >= self.max_prefix_entries
                )
            ):
                raise _WorkloadLogLimitError(
                    "workload-log distinct prefix dimensions exceed "
                    f"max_prefix_entries ({self.max_prefix_entries})"
                )
            self.pending_prefix_references[prefix] += 1
        pending.prefixes[prefix] += 1

    def _reserve_pending_report(
        self,
        pending: _V2PendingAggregate,
        report: dict[str, object],
    ) -> None:
        if (
            len(self.request_reports)
            + self.pending_request_report_entries
            < self.max_report_entries
        ):
            pending.request_reports.append(report)
            self.pending_request_report_entries += 1
        else:
            pending.request_reports_omitted += 1

    def _release_pending_reservations(
        self,
        pending: _V2PendingAggregate,
    ) -> None:
        for prefix in pending.prefixes:
            references = self.pending_prefix_references[prefix] - 1
            if references <= 0:
                self.pending_prefix_references.pop(prefix, None)
            else:
                self.pending_prefix_references[prefix] = references
        self.pending_request_report_entries -= len(
            pending.request_reports
        )
        if self.pending_request_report_entries < 0:
            raise AssertionError(
                "negative workload-log pending report reservation"
            )

    def _pending(
        self, instance: tuple[int, int]
    ) -> _V2PendingAggregate:
        pending = self.pending.get(instance)
        if pending is None:
            pending = _V2PendingAggregate()
            self.pending[instance] = pending
        return pending

    def _track_timestamp(self, unix_ns: int) -> None:
        if unix_ns == 0:
            raise _V2LifecycleError(
                "workload-log v2 lifecycle timestamp is zero"
            )
        self.first_unix_ns = (
            unix_ns
            if self.first_unix_ns is None
            else min(self.first_unix_ns, unix_ns)
        )
        self.last_unix_ns = (
            unix_ns
            if self.last_unix_ns is None
            else max(self.last_unix_ns, unix_ns)
        )

    def _censored_state(
        self,
        key: tuple[int, int, int, int],
    ) -> _V2CensoredState:
        state = self.left_censored_active.get(key)
        if state is not None:
            return state
        instance = (key[0], key[1])
        allowance = self.left_censored_allowance.get(instance, 0)
        if allowance <= 0:
            raise _V2LifecycleError(
                "workload-log v2 lifecycle record has no retained BEGIN"
            )
        if len(self.active) + len(self.left_censored_active) >= (
            self.max_active_requests
        ):
            raise _WorkloadLogLimitError(
                "workload-log v2 active request count exceeds "
                f"max_active_requests ({self.max_active_requests})"
            )
        self.left_censored_allowance[instance] = allowance - 1
        state = _V2CensoredState()
        self.left_censored_active[key] = state
        self.left_censored_requests += 1
        return state

    def consume(
        self,
        kind: int,
        frame: memoryview,
        header: dict[str, object],
    ) -> None:
        instance = self._instance(header)
        self._check_lifecycle_sequence(instance, _u64(frame, 40))
        unix_ns = _u64(frame, 48)
        steady_ns = _u64(frame, 56)
        self._track_timestamp(unix_ns)
        if steady_ns == 0:
            raise _V2LifecycleError(
                "workload-log v2 lifecycle steady timestamp is zero"
            )
        key = self._request_key(header, frame)

        if kind == _V2_REQUEST_BEGIN_KIND:
            self._consume_begin(key, frame, header)
        elif kind == _V2_ID_CHUNK_KIND:
            self._consume_chunk(key, frame)
        elif kind == _V2_BATCH_KIND:
            self._consume_batch(key, frame)
        elif kind == _V2_REQUEST_END_KIND:
            self._consume_end(key, frame)
        else:  # pragma: no cover - geometry rejects unknown kinds first.
            raise _V2LifecycleError("workload-log v2 record kind is unknown")

    def _consume_begin(
        self,
        key: tuple[int, int, int, int],
        frame: memoryview,
        header: dict[str, object],
    ) -> None:
        flags = _u32(frame, 28)
        method = _u32(frame, 32)
        prefix = _u32(frame, 36)
        if (
            flags
            & ~(
                _V2_REQUEST_BEGIN_EXACT_FLAG
                | _V2_REQUEST_BEGIN_DECLARED_IDS_FLAG
            )
            or method not in {_GET_METHOD, _STREAM_GET_METHOD}
            or prefix < 1
            or prefix > int(header["dimension"])
            or _u64(frame, 112) == 0
            or any(frame[120:])
        ):
            raise _V2LifecycleError(
                "workload-log v2 BEGIN metadata is invalid"
            )
        exact_snapshot = bool(flags & _V2_REQUEST_BEGIN_EXACT_FLAG)
        catalog_generation = _u64(frame, 80)
        committed_sequence = _u64(frame, 88)
        # Unary Get cannot expose the exact internally pinned view, but its
        # BEGIN still carries the collection identity observed immediately
        # before the read. The inexact flag qualifies that anchor; it does not
        # require erasing useful generation/sequence context. A committed
        # sequence without any catalog generation remains malformed.
        if (
            exact_snapshot
            and catalog_generation == 0
        ) or (
            catalog_generation == 0
            and committed_sequence != 0
        ):
            raise _V2LifecycleError(
                "workload-log v2 BEGIN snapshot metadata is invalid"
            )
        declared_known = bool(flags & _V2_REQUEST_BEGIN_DECLARED_IDS_FLAG)
        declared_ids = _u64(frame, 104)
        if not declared_known and declared_ids != 0:
            raise _V2LifecycleError(
                "workload-log v2 BEGIN declared-ID metadata is invalid"
            )
        if key in self.active or key in self.left_censored_active:
            raise _V2LifecycleError(
                "workload-log v2 request token has duplicate BEGIN"
            )
        if len(self.active) + len(self.left_censored_active) >= (
            self.max_active_requests
        ):
            raise _WorkloadLogLimitError(
                "workload-log v2 active request count exceeds "
                f"max_active_requests ({self.max_active_requests})"
            )
        state = _V2RequestState(
            method=method,
            prefix_dimensions=prefix,
            exact_snapshot=exact_snapshot,
            declared_ids_known=declared_known,
            declared_ids=declared_ids,
            begin_unix_ns=_u64(frame, 48),
            begin_steady_ns=_u64(frame, 56),
        )
        self.active[key] = state
        instance = (key[0], key[1])
        self.active_by_instance[instance] += 1
        self.max_parallel_requests = max(
            self.max_parallel_requests,
            self.active_by_instance[instance],
        )
        self.begun_requests += 1

    def _consume_chunk(
        self,
        key: tuple[int, int, int, int],
        frame: memoryview,
    ) -> None:
        if _u32(frame, 28) != 0 or _u32(frame, 36) != 0 or any(frame[112:256]):
            raise _V2LifecycleError(
                "workload-log v2 ID_CHUNK metadata is invalid"
            )
        count = _u32(frame, 32)
        logical_offset = _u64(frame, 80)
        batch_sequence = _u64(frame, 88)
        delivery_sequence = _u64(frame, 96)
        chunk_sequence = _u64(frame, 104)
        if delivery_sequence != 0:
            raise _V2LifecycleError(
                "workload-log v2 ID_CHUNK delivery sequence is reserved"
            )
        state = self.active.get(key)
        if state is None:
            self._censored_state(key)
            return

        if state.current_batch_sequence is None:
            if batch_sequence != state.expected_batch_sequence:
                raise _V2LifecycleError(
                    "workload-log v2 batch sequence has a gap"
                )
            state.current_batch_sequence = batch_sequence
            state.current_batch_offset = logical_offset
            state.current_batch_rows = 0
        if (
            batch_sequence != state.current_batch_sequence
            or chunk_sequence != state.expected_chunk_sequence
            or logical_offset != state.expected_logical_offset
        ):
            raise _V2LifecycleError(
                "workload-log v2 ID_CHUNK order or offset is discontinuous"
            )
        payload = frame[_RECORD_HEADER_BYTES:]
        state.digest_low = _v2_fnv_update(state.digest_low, payload)
        state.digest_high = _v2_fnv_update(state.digest_high, payload)
        state.expected_logical_offset += count
        state.current_batch_rows += count
        state.expected_chunk_sequence += 1
        state.id_chunks += 1

    def _consume_batch(
        self,
        key: tuple[int, int, int, int],
        frame: memoryview,
    ) -> None:
        if (
            _u32(frame, 28) != 0
            or _u32(frame, 36) != 0
            or any(frame[192:224])
            or _u64(frame, 248) != 0
        ):
            raise _V2LifecycleError(
                "workload-log v2 BATCH metadata is invalid"
            )
        backend = _u32(frame, 32)
        if backend > 2:
            raise _V2LifecycleError(
                "workload-log v2 BATCH backend is invalid"
            )
        batch_sequence = _u64(frame, 80)
        logical_offset = _u64(frame, 88)
        rows = _u64(frame, 96)
        response_bytes = _u64(frame, 104)
        delivery_sequence = _u64(frame, 224)
        useful_read_bytes = _u64(frame, 232)
        backend_tasks = _u64(frame, 240)
        instance = (key[0], key[1])
        pending = self._pending(instance)
        if delivery_sequence != 0:
            previous_global = self.last_delivery_sequence.get(instance)
            if previous_global is not None and (
                delivery_sequence < previous_global
            ):
                pending.delivery_physical_inversions += 1
            self.last_delivery_sequence[instance] = delivery_sequence
            pending.delivery_sequence_count += 1
            pending.delivery_sequence_min = (
                delivery_sequence
                if pending.delivery_sequence_min is None
                else min(
                    pending.delivery_sequence_min,
                    delivery_sequence,
                )
            )
            pending.delivery_sequence_max = (
                delivery_sequence
                if pending.delivery_sequence_max is None
                else max(
                    pending.delivery_sequence_max,
                    delivery_sequence,
                )
            )
        state = self.active.get(key)
        if state is None:
            self._censored_state(key)
            return
        if (
            batch_sequence != state.expected_batch_sequence
            or logical_offset != state.current_batch_offset
            or rows != state.current_batch_rows
            or (
                rows != 0
                and state.current_batch_sequence != batch_sequence
            )
            or (
                rows == 0
                and state.current_batch_sequence is not None
            )
        ):
            raise _V2LifecycleError(
                "workload-log v2 BATCH boundary is inconsistent"
            )
        if (
            (
                (state.method == _GET_METHOD)
                != (delivery_sequence == 0)
            )
            or (
                delivery_sequence != 0
                and state.previous_delivery_sequence is not None
                and delivery_sequence
                    <= state.previous_delivery_sequence
            )
        ):
            self.pending_delivery_order_instances.add(instance)
        state.batches += 1
        if delivery_sequence != 0:
            state.previous_delivery_sequence = delivery_sequence
        state.response_bytes += response_bytes
        state.batch_rows += rows
        state.expected_batch_sequence += 1
        state.current_batch_sequence = None
        state.current_batch_offset = state.expected_logical_offset
        state.current_batch_rows = 0
        state.batch_size_buckets[_v2_batch_size_bucket(rows)] += 1
        state.batch_backends[_V2_BACKENDS[backend]] += 1
        for index, name in enumerate(_V2_BATCH_TIMING_NAMES):
            state.batch_timings[name] += _u64(frame, 112 + index * 8)
        for index, name in enumerate(_V2_BATCH_METRIC_NAMES):
            state.batch_metrics[name] += _u64(frame, 256 + index * 8)
        state.batch_metrics["useful_read_bytes"] += useful_read_bytes
        state.batch_metrics["backend_tasks"] += backend_tasks

    def _consume_end(
        self,
        key: tuple[int, int, int, int],
        frame: memoryview,
    ) -> None:
        if any(frame[224:]):
            raise _V2LifecycleError(
                "workload-log v2 END reserved bytes are nonzero"
            )
        flags = _u32(frame, 28)
        status = _u32(frame, 32)
        if (
            flags & ~_V2_REQUEST_END_COMPLETE_FLAG
            or status not in _V2_REQUEST_STATUSES
        ):
            raise _V2LifecycleError(
                "workload-log v2 END metadata is invalid"
            )
        state = self.active.pop(key, None)
        if state is None:
            if key not in self.left_censored_active:
                self._censored_state(key)
            self.left_censored_active.pop(key, None)
            return

        instance = (key[0], key[1])
        self.active_by_instance[instance] -= 1
        submitted_ids = _u64(frame, 80)
        delivered_ids = _u64(frame, 88)
        id_chunks = _u64(frame, 96)
        batches = _u64(frame, 104)
        response_bytes = _u64(frame, 112)
        dropped_records = _u64(frame, 120)
        dropped_ids = _u64(frame, 128)
        duration_ns = _u64(frame, 136)
        reason_records = _saturating_sum(
            _u64(frame, 144),
            _u64(frame, 160),
            _u64(frame, 176),
            _u64(frame, 192),
        )
        reason_ids = _saturating_sum(
            _u64(frame, 152),
            _u64(frame, 168),
            _u64(frame, 184),
            _u64(frame, 200),
        )
        complete = bool(flags & _V2_REQUEST_END_COMPLETE_FLAG)
        end_steady_ns = _u64(frame, 56)
        digest_matches = (
            _u64(frame, 208) == state.digest_low
            and _u64(frame, 216) == state.digest_high
        )
        counts_match = (
            submitted_ids == state.expected_logical_offset
            and id_chunks == state.id_chunks
            and batches == state.batches
            and (
                not state.declared_ids_known
                or submitted_ids == state.declared_ids
            )
            and dropped_records == reason_records
            and dropped_ids == reason_ids
            and end_steady_ns >= state.begin_steady_ns
            and duration_ns <= end_steady_ns - state.begin_steady_ns
            and (status != 1 or _u32(frame, 36) == 0)
            and delivered_ids == state.batch_rows
            and response_bytes >= state.response_bytes
        )
        successful_complete = (
            status == 1
            and complete
            and state.current_batch_sequence is None
            and counts_match
            and digest_matches
            and delivered_ids == submitted_ids
            and dropped_records == 0
            and dropped_ids == 0
        )
        diagnostic_non_success_complete = (
            status != 1
            and not complete
            and counts_match
            and digest_matches
            and dropped_records == 0
            and dropped_ids == 0
        )
        if complete and not successful_complete:
            self.interior_incomplete_requests += 1
            raise _V2LifecycleError(
                "workload-log v2 complete request has gaps, drops, or a "
                "digest mismatch"
            )
        pending = self._pending(instance)
        if status == 1 and not successful_complete:
            pending.interior_incomplete_requests += 1
        pending.statuses[_V2_REQUEST_STATUSES[status]] += 1
        if successful_complete:
            method_name = (
                "get" if state.method == _GET_METHOD else "stream_get"
            )
            pending.methods[method_name] += 1
            self._record_pending_prefix(
                pending,
                state.prefix_dimensions,
            )
            pending.complete_requests += 1
            pending.request_time_ns += duration_ns
            pending.id_chunk_records += state.id_chunks
            pending.ids_in_chunks += submitted_ids
            pending.batch_records += state.batches
            pending.batch_rows += state.batch_rows
            pending.batch_size_buckets.update(state.batch_size_buckets)
            pending.batch_backends.update(state.batch_backends)
            pending.batch_timings.update(state.batch_timings)
            pending.batch_metrics.update(state.batch_metrics)
            report = {
                "method": method_name,
                "prefix_dimensions": state.prefix_dimensions,
                "exact_snapshot": state.exact_snapshot,
                "declared_ids": (
                    state.declared_ids
                    if state.declared_ids_known
                    else None
                ),
                "submitted_ids": submitted_ids,
                "delivered_ids": delivered_ids,
                "id_chunks": id_chunks,
                "batches": batches,
                "response_bytes": response_bytes,
                "duration_ns": duration_ns,
                "begin_unix_ns": state.begin_unix_ns,
                "end_unix_ns": _u64(frame, 48),
            }
            self._reserve_pending_report(pending, report)
        elif diagnostic_non_success_complete:
            pending.non_success_outcomes += 1
        else:
            pending.incomplete_requests += 1

    def finish_instance(
        self,
        instance: tuple[int, int],
        *,
        complete_terminal: bool,
    ) -> None:
        active_keys = [
            key for key in self.active if (key[0], key[1]) == instance
        ]
        pending = self.pending.pop(instance, None)
        if pending is not None:
            self._release_pending_reservations(pending)
        if active_keys:
            if complete_terminal:
                self.interior_incomplete_requests += len(active_keys)
            else:
                self.right_censored_requests += len(active_keys)
            for key in active_keys:
                self.active.pop(key, None)
        if pending is not None:
            if complete_terminal:
                self.interior_incomplete_requests += (
                    pending.complete_requests
                    + pending.non_success_outcomes
                    + pending.incomplete_requests
                )
            else:
                self.right_censored_requests += (
                    pending.complete_requests
                    + pending.non_success_outcomes
                    + pending.incomplete_requests
                )
        if complete_terminal:
            if instance in self.pending_sequence_gap_instances:
                self.sequence_gap_instances.add(instance)
            if instance in self.pending_delivery_order_instances:
                self.delivery_order_instances.add(instance)
        self.pending_sequence_gap_instances.discard(instance)
        self.pending_delivery_order_instances.discard(instance)
        censored_keys = [
            key
            for key in self.left_censored_active
            if (key[0], key[1]) == instance
        ]
        for key in censored_keys:
            self.left_censored_active.pop(key, None)
        self.active_by_instance.pop(instance, None)


def _v2_new_file_report(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": 0,
        "finalized": path.suffix == ".tdbrl",
        "valid": False,
        "torn_tail": False,
        "logical_duplicate": False,
        "counter_coverage": "missing",
        "opening_checkpoint": False,
        "terminal_checkpoint": False,
        "fresh_terminal_checkpoint": False,
        "segment_sequence": None,
        "error": None,
        "lifecycle_records": 0,
        "begin_records": 0,
        "id_chunk_records": 0,
        "ids_in_chunks": 0,
        "batch_records": 0,
        "end_records": 0,
        "checkpoint_records": 0,
    }


def _pread_exact(
    descriptor: int,
    size: int,
    offset: int,
    *,
    deadline_monotonic: float | None,
) -> bytes:
    parts: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        _check_deadline(deadline_monotonic)
        part = os.pread(descriptor, remaining, position)
        if not part:
            break
        parts.append(part)
        position += len(part)
        remaining -= len(part)
    return b"".join(parts)


def _read_v2_header_bounded(
    path: Path,
    *,
    max_file_bytes: int,
    deadline_monotonic: float | None,
) -> tuple[dict[str, object], int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise WorkloadLogError(
                "workload-log path is not a regular file"
            )
        size = int(status.st_size)
        if size > max_file_bytes:
            raise _WorkloadLogLimitError(
                "workload-log file exceeds max_file_bytes before reading: "
                f"{path}"
            )
        if size < _HEADER_BYTES:
            raise WorkloadLogError(
                "workload-log v2 file is shorter than its header"
            )
        raw_header = _pread_exact(
            descriptor,
            _HEADER_BYTES,
            0,
            deadline_monotonic=deadline_monotonic,
        )
        if len(raw_header) != _HEADER_BYTES:
            raise _WorkloadLogLimitError(
                "workload-log v2 file changed while reading its header"
            )
        return _parse_v2_header(memoryview(raw_header)), size
    finally:
        os.close(descriptor)


def _v2_checkpoint_matches_records(
    opening: dict[str, int],
    checkpoint: dict[str, int],
    *,
    lifecycle_records: int,
    ids: int,
    chunks: int,
    batches: int,
) -> bool:
    if not _counters_dominate(checkpoint, opening):
        return False
    delta = _counter_delta(checkpoint, opening)
    return (
        delta["written_records"] == lifecycle_records
        and delta["written_ids"] == ids
        and delta["written_chunks"] == chunks
        and delta["written_batches"] == batches
    )


def _inspect_v2_file_bounded(
    path: Path,
    *,
    expected_header: dict[str, object],
    expected_size: int,
    max_file_bytes: int,
    deadline_monotonic: float | None,
    accumulator: _V2Accumulator,
) -> tuple[
    dict[str, object],
    dict[str, int],
    dict[str, int] | None,
]:
    finalized = path.suffix == ".tdbrl"
    report = _v2_new_file_report(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WorkloadLogError(
                "workload-log path is not a regular file"
            )
        file_size = int(before.st_size)
        if file_size > max_file_bytes:
            raise _WorkloadLogLimitError(
                "workload-log file exceeds max_file_bytes before reading: "
                f"{path}"
            )
        if file_size != expected_size:
            raise _WorkloadLogLimitError(
                "workload-log v2 file changed after inventory"
            )
        report["bytes"] = file_size
        raw_header = _pread_exact(
            descriptor,
            _HEADER_BYTES,
            0,
            deadline_monotonic=deadline_monotonic,
        )
        if len(raw_header) != _HEADER_BYTES:
            raise _WorkloadLogLimitError(
                "workload-log v2 file changed while reading"
            )
        header = _parse_v2_header(memoryview(raw_header))
        if header != expected_header:
            raise _WorkloadLogLimitError(
                "workload-log v2 header changed after inventory"
            )
        report["segment_sequence"] = header["segment_sequence"]

        offset = _HEADER_BYTES
        opening: dict[str, int] | None = None
        latest_checkpoint: dict[str, int] | None = None
        previous_checkpoint: dict[str, int] | None = None
        last_kind: int | None = None
        last_checkpoint_opening = False
        lifecycle_records = 0
        ids = 0
        chunks = 0
        batches = 0
        while offset < file_size:
            _check_deadline(deadline_monotonic)
            remaining = file_size - offset
            if remaining < _RECORD_HEADER_BYTES:
                if finalized:
                    raise WorkloadLogError(
                        "finalized workload-log v2 file has a truncated "
                        "record header"
                    )
                report["torn_tail"] = True
                break
            common_raw = _pread_exact(
                descriptor,
                _RECORD_HEADER_BYTES,
                offset,
                deadline_monotonic=deadline_monotonic,
            )
            if len(common_raw) != _RECORD_HEADER_BYTES:
                raise _WorkloadLogLimitError(
                    "workload-log v2 file changed while reading"
                )
            common = memoryview(common_raw)
            if bytes(common[:8]) != _V2_RECORD_MAGIC:
                raise WorkloadLogError(
                    "workload-log v2 record magic is invalid"
                )
            if (
                _u32(common, 8) != _V2_VERSION
                or _u32(common, 12) != _RECORD_HEADER_BYTES
            ):
                raise WorkloadLogError(
                    "workload-log v2 record version is unsupported"
                )
            frame_bytes = _u32(common, 16)
            kind = _u32(common, 20)
            if kind == _V2_ID_CHUNK_KIND:
                count = _u32(common, 32)
                expected_frame_bytes = (
                    _RECORD_HEADER_BYTES + count * 8
                )
                if (
                    count < 1
                    or count > int(header["id_chunk_rows"])
                    or frame_bytes != expected_frame_bytes
                ):
                    raise WorkloadLogError(
                        "workload-log v2 ID_CHUNK geometry is invalid"
                    )
            elif kind in {
                _V2_REQUEST_BEGIN_KIND,
                _V2_REQUEST_END_KIND,
                _V2_CHECKPOINT_KIND,
            }:
                if frame_bytes != _RECORD_HEADER_BYTES:
                    raise WorkloadLogError(
                        "workload-log v2 fixed record geometry is invalid"
                    )
            elif kind == _V2_BATCH_KIND:
                if frame_bytes != _V2_BATCH_FRAME_BYTES:
                    raise WorkloadLogError(
                        "workload-log v2 BATCH geometry is invalid"
                    )
            else:
                raise WorkloadLogError(
                    "workload-log v2 record kind is unknown"
                )
            if frame_bytes > (
                _RECORD_HEADER_BYTES
                + _V2_MAX_IDS_PER_CHUNK * 8
            ):
                raise WorkloadLogError(
                    "workload-log v2 record exceeds its hard size bound"
                )
            if frame_bytes > remaining:
                if finalized:
                    raise WorkloadLogError(
                        "finalized workload-log v2 file has a truncated "
                        "record"
                    )
                report["torn_tail"] = True
                break
            frame_raw = (
                common_raw
                if frame_bytes == _RECORD_HEADER_BYTES
                else _pread_exact(
                    descriptor,
                    frame_bytes,
                    offset,
                    deadline_monotonic=deadline_monotonic,
                )
            )
            if len(frame_raw) != frame_bytes:
                raise _WorkloadLogLimitError(
                    "workload-log v2 file changed while reading a record"
                )
            frame = memoryview(frame_raw)
            if (
                _crc32c(
                    frame,
                    zero_offset=_CRC_OFFSET,
                    zero_size=_CRC_SIZE,
                )
                != _u32(frame, _CRC_OFFSET)
            ):
                if not finalized and offset + frame_bytes == file_size:
                    report["torn_tail"] = True
                    break
                raise WorkloadLogError(
                    "workload-log v2 record CRC32C is invalid"
                )

            first_record = offset == _HEADER_BYTES
            if kind == _V2_CHECKPOINT_KIND:
                counters = _v2_checkpoint_counters(frame)
                opening_flag = bool(
                    _u32(frame, 32) & _V2_CHECKPOINT_OPENING_FLAG
                )
                if opening_flag != first_record:
                    raise WorkloadLogError(
                        "workload-log v2 segment must begin with exactly "
                        "one opening checkpoint"
                    )
                if (
                    previous_checkpoint is not None
                    and not _counters_dominate(
                        counters, previous_checkpoint
                    )
                ):
                    raise WorkloadLogError(
                        "workload-log v2 checkpoints regress within a "
                        "segment"
                    )
                if opening is None:
                    opening = counters
                    report["opening_checkpoint"] = True
                    accumulator.set_left_censored_allowance(
                        _V2Accumulator._instance(header),
                        opening,
                    )
                elif not _v2_checkpoint_matches_records(
                    opening,
                    counters,
                    lifecycle_records=lifecycle_records,
                    ids=ids,
                    chunks=chunks,
                    batches=batches,
                ):
                    raise WorkloadLogError(
                        "workload-log v2 checkpoint does not cover the "
                        "records retained before it"
                    )
                previous_checkpoint = counters
                latest_checkpoint = counters
                last_checkpoint_opening = opening_flag
                accumulator.commit_checkpoint(
                    _V2Accumulator._instance(header)
                )
                report["checkpoint_records"] = (
                    int(report["checkpoint_records"]) + 1
                )
            else:
                if opening is None:
                    raise WorkloadLogError(
                        "workload-log v2 lifecycle record precedes the "
                        "opening checkpoint"
                    )
                accumulator.consume(kind, frame, header)
                lifecycle_records += 1
                report["lifecycle_records"] = lifecycle_records
                if kind == _V2_REQUEST_BEGIN_KIND:
                    report["begin_records"] = (
                        int(report["begin_records"]) + 1
                    )
                elif kind == _V2_ID_CHUNK_KIND:
                    count = _u32(frame, 32)
                    chunks += 1
                    ids += count
                    report["id_chunk_records"] = chunks
                    report["ids_in_chunks"] = ids
                elif kind == _V2_BATCH_KIND:
                    batches += 1
                    report["batch_records"] = batches
                else:
                    report["end_records"] = (
                        int(report["end_records"]) + 1
                    )
                last_checkpoint_opening = False
            last_kind = kind
            offset += frame_bytes

        if opening is None:
            if finalized:
                raise WorkloadLogError(
                    "finalized workload-log v2 file has no opening "
                    "checkpoint"
                )
            report["torn_tail"] = True
            opening = {
                name: 0 for name in _V2_CHECKPOINT_COUNTER_NAMES
            }
        if finalized and last_kind == _V2_CHECKPOINT_KIND and (
            last_checkpoint_opening
        ):
            raise WorkloadLogError(
                "finalized workload-log v2 file ends at its opening "
                "checkpoint"
            )
        finalized_terminal = (
            latest_checkpoint
            if (
                finalized
                and last_kind == _V2_CHECKPOINT_KIND
                and not last_checkpoint_opening
            )
            else None
        )
        report["terminal_checkpoint"] = (
            last_kind == _V2_CHECKPOINT_KIND
        )
        report["fresh_terminal_checkpoint"] = (
            finalized_terminal is not None
        )
        report["counter_coverage"] = (
            "complete"
            if finalized_terminal is not None
            else "lower_bound"
            if latest_checkpoint is not None
            else "missing"
        )
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise _WorkloadLogLimitError(
                "workload-log v2 file changed during inspection"
            )
        report["valid"] = True
        trusted_checkpoint = (
            finalized_terminal if finalized else latest_checkpoint
        )
        return report, opening, trusted_checkpoint
    finally:
        os.close(descriptor)


def _v2_configuration_identity(
    header: dict[str, object],
) -> tuple[object, ...]:
    return tuple(
        header[name]
        for name in (
            "id_chunk_rows",
            "max_active_requests",
            "buffer_bytes",
            "segment_bytes",
            "rotate_seconds",
            "retention_bytes",
            "retention_seconds",
            "minimum_free_bytes",
            "max_write_bytes_per_second",
            "max_retained_files",
            "reader_backend_code",
            "io_workers",
            "io_queue_capacity",
            "io_ranges_per_task",
            "max_read_range_bytes",
            "max_gap_bytes",
            "io_uring_queue_depth",
            "io_uring_workers",
            "metric_page_bytes",
            "metric_chunk_bytes",
            "tuning_explicit_overrides",
        )
    )


def _inspect_workload_logs_v2(
    directory: Path,
    paths: list[tuple[Path, int]],
    *,
    storage: str | os.PathLike[str] | None,
    max_files: int,
    max_total_bytes: int,
    max_file_bytes: int,
    max_report_entries: int,
    max_prefix_entries: int,
    max_active_requests: int,
    deadline_monotonic: float | None,
    mixed_log_versions: bool,
) -> dict[str, object]:
    accumulator = _V2Accumulator(
        max_active_requests=max_active_requests,
        max_report_entries=max_report_entries,
        max_prefix_entries=max_prefix_entries,
    )
    parsed_headers: list[
        tuple[dict[str, object], Path, int]
    ] = []
    reports: list[dict[str, object]] = []
    reports_omitted = 0

    def record_report(report: dict[str, object]) -> None:
        nonlocal reports_omitted
        if len(reports) < max_report_entries:
            reports.append(report)
        else:
            reports_omitted += 1

    invalid_finalized = 0
    invalid_open = 0
    actual_total_bytes = 0
    identities: set[tuple[object, ...]] = set()
    configurations: set[tuple[object, ...]] = set()
    canonical_header: dict[str, object] | None = None

    for path, preflight_bytes in paths:
        _check_deadline(deadline_monotonic)
        try:
            header, file_bytes = _read_v2_header_bounded(
                path,
                max_file_bytes=max_file_bytes,
                deadline_monotonic=deadline_monotonic,
            )
        except _WorkloadLogLimitError:
            raise
        except (OSError, WorkloadLogError) as error:
            report = _v2_new_file_report(path)
            report["bytes"] = preflight_bytes
            report["error"] = str(error)
            record_report(report)
            if path.suffix == ".tdbrl":
                invalid_finalized += 1
            else:
                invalid_open += 1
            actual_total_bytes += preflight_bytes
            continue
        parsed_headers.append((header, path, file_bytes))

    parsed_headers.sort(
        key=lambda item: (
            int(item[0]["instance_id_high"]),
            int(item[0]["instance_id_low"]),
            int(item[0]["segment_sequence"]),
            item[1].name,
        )
    )
    logical_files: set[tuple[int, int, int]] = set()
    histories: dict[
        tuple[int, int],
        list[
            tuple[
                int,
                bool,
                dict[str, int],
                dict[str, int] | None,
            ]
        ],
    ] = {}
    valid_files = 0
    torn_open = 0
    logical_duplicates = 0
    for header, path, file_bytes in parsed_headers:
        _check_deadline(deadline_monotonic)
        identity = _header_identity(header)
        identities.add(identity)
        configurations.add(_v2_configuration_identity(header))
        logical_key = (
            int(header["instance_id_low"]),
            int(header["instance_id_high"]),
            int(header["segment_sequence"]),
        )
        if logical_key in logical_files:
            duplicate = _v2_new_file_report(path)
            duplicate["bytes"] = file_bytes
            duplicate["valid"] = True
            duplicate["logical_duplicate"] = True
            duplicate["segment_sequence"] = header[
                "segment_sequence"
            ]
            record_report(duplicate)
            logical_duplicates += 1
            actual_total_bytes += file_bytes
            continue
        logical_files.add(logical_key)
        try:
            report, opening, terminal = _inspect_v2_file_bounded(
                path,
                expected_header=header,
                expected_size=file_bytes,
                max_file_bytes=max_file_bytes,
                deadline_monotonic=deadline_monotonic,
                accumulator=accumulator,
            )
        except _WorkloadLogLimitError:
            raise
        except (OSError, WorkloadLogError) as error:
            report = _v2_new_file_report(path)
            report["bytes"] = file_bytes
            report["segment_sequence"] = header[
                "segment_sequence"
            ]
            report["error"] = str(error)
            if path.suffix == ".tdbrl":
                invalid_finalized += 1
            else:
                invalid_open += 1
        else:
            valid_files += 1
            if bool(report["torn_tail"]):
                torn_open += 1
            instance = (
                int(header["instance_id_low"]),
                int(header["instance_id_high"]),
            )
            histories.setdefault(instance, []).append(
                (
                    int(header["segment_sequence"]),
                    path.suffix == ".tdbrl",
                    opening,
                    terminal,
                )
            )
            if canonical_header is None:
                canonical_header = header
        record_report(report)
        actual_total_bytes += int(report["bytes"])
        if actual_total_bytes > max_total_bytes:
            raise _WorkloadLogLimitError(
                "workload-log bytes exceed max_total_bytes after opening "
                "a concurrently changed file"
            )

    warnings: list[str] = []
    reasons: list[str] = []
    exact_failures: list[str] = []
    if not paths:
        reasons.append("no_log_files")
    if not identities:
        reasons.append("no_valid_log_headers")
        exact_failures.append("no_valid_log_headers")
    if mixed_log_versions:
        reasons.append("mixed_log_versions")
        exact_failures.append("mixed_log_versions")
    if len(identities) > 1:
        reasons.append("mixed_collection_or_schema_identity")
        exact_failures.append("mixed_collection_or_schema_identity")
    if len(configurations) > 1:
        reasons.append("mixed_logging_configuration")
    if logical_duplicates:
        reasons.append("duplicate_logical_files")
        exact_failures.append("duplicate_logical_files")
    if invalid_finalized:
        reasons.append("corrupt_finalized_files")
        exact_failures.append("corrupt_finalized_files")
    if invalid_open:
        reasons.append("corrupt_open_files")
        exact_failures.append("corrupt_open_files")
    if torn_open:
        warnings.append("open_file_has_torn_tail")

    counters = Counter[str]()
    counter_regressions = 0
    segment_sequence_gaps = 0
    incomplete_counter_instances = 0
    incomplete_terminal_partition_instances = 0
    saturated_counter_instances = 0
    for instance, history in histories.items():
        history.sort(key=lambda item: item[0])
        first_opening: dict[str, int] | None = None
        previous_terminal: dict[str, int] | None = None
        previous_sequence: int | None = None
        latest_terminal: dict[str, int] | None = None
        latest_finalized = False
        regressed = False
        sequence_gap = False
        saturated = False
        complete = True
        for (
            sequence,
            finalized,
            opening,
            terminal,
        ) in history:
            if (
                previous_sequence is not None
                and sequence != previous_sequence + 1
            ):
                sequence_gap = True
            previous_sequence = sequence
            if first_opening is None:
                first_opening = opening
            if (
                previous_terminal is not None
                and opening != previous_terminal
            ):
                regressed = True
            if previous_terminal is None and latest_terminal is not None:
                complete = False
            latest_terminal = terminal
            latest_finalized = finalized
            if terminal is None:
                complete = False
            else:
                previous_terminal = terminal
        if first_opening is None:
            complete = False
        if (
            latest_terminal is None
            or first_opening is None
        ):
            complete = False
        elif not _counters_dominate(latest_terminal, first_opening):
            regressed = True
            complete = False
        else:
            delta = _counter_delta(latest_terminal, first_opening)
            counters.update(delta)
            saturated = any(
                value == _UINT64_MAX
                for value in (
                    *first_opening.values(),
                    *latest_terminal.values(),
                )
            )
            if latest_finalized and not (
                _v2_terminal_partitions_complete(latest_terminal)
            ):
                incomplete_terminal_partition_instances += 1
                complete = False
            if not latest_finalized:
                complete = False
        accumulator.finish_instance(
            instance,
            complete_terminal=complete,
        )
        if regressed:
            counter_regressions += 1
        if sequence_gap:
            segment_sequence_gaps += 1
        if saturated:
            saturated_counter_instances += 1
        if not complete:
            incomplete_counter_instances += 1

    if counter_regressions:
        reasons.append("counter_regression")
        exact_failures.append("counter_regression")
    if segment_sequence_gaps:
        reasons.append("segment_sequence_gap")
        exact_failures.append("segment_sequence_gap")
    if accumulator.sequence_gap_instances:
        reasons.append("lifecycle_sequence_gap")
        exact_failures.append("lifecycle_sequence_gap")
    if accumulator.delivery_order_instances:
        reasons.append("delivery_sequence_order")
        exact_failures.append("delivery_sequence_order")
    if saturated_counter_instances:
        reasons.append("counter_saturation")
        exact_failures.append("counter_saturation")
    if incomplete_counter_instances:
        reasons.append("incomplete_counter_coverage")
        exact_failures.append("incomplete_counter_coverage")
    if incomplete_terminal_partition_instances:
        reasons.append("incomplete_terminal_counter_partition")
        exact_failures.append("incomplete_terminal_counter_partition")
    if accumulator.interior_incomplete_requests:
        reasons.append("interior_incomplete_requests")
        exact_failures.append("interior_incomplete_requests")
    if accumulator.incomplete_requests:
        reasons.append("incomplete_request_outcomes")
        exact_failures.append("incomplete_request_outcomes")
    if accumulator.non_success_outcomes:
        warnings.append("non_success_request_outcomes")

    dropped_records = sum(
        counters[name]
        for name in (
            "dropped_full_records",
            "dropped_runtime_records",
            "dropped_minimum_free_records",
            "dropped_io_budget_records",
        )
    )
    dropped_ids = sum(
        counters[name]
        for name in (
            "dropped_full_ids",
            "dropped_runtime_ids",
            "dropped_minimum_free_ids",
            "dropped_io_budget_ids",
        )
    )
    if dropped_records:
        reasons.append("exact_records_dropped")
        exact_failures.append("exact_records_dropped")
    if dropped_ids:
        reasons.append("exact_ids_dropped")
        exact_failures.append("exact_ids_dropped")
    if accumulator.complete_requests == 0:
        reasons.append("no_complete_point_read_requests")

    expected_identity: tuple[object, ...] | None = None
    if storage is not None:
        expected_identity = _storage_identity(
            Path(storage).expanduser()
        )
        if identities and expected_identity not in identities:
            reasons.append("storage_identity_mismatch")

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
                "id_chunk_rows",
                "max_active_requests",
                "buffer_bytes",
                "segment_bytes",
                "rotate_seconds",
                "retention_bytes",
                "retention_seconds",
                "minimum_free_bytes",
                "max_write_bytes_per_second",
                "max_retained_files",
                "reader_backend",
                "reader_backend_code",
                "io_workers",
                "io_queue_capacity",
                "io_ranges_per_task",
                "max_read_range_bytes",
                "max_gap_bytes",
                "io_uring_queue_depth",
                "io_uring_workers",
                "metric_page_bytes",
                "metric_chunk_bytes",
                "tuning_explicit_overrides",
            )
        }
        if canonical_header is not None
        else None
    )
    exact_capture = not exact_failures
    delivery_span = (
        accumulator.delivery_sequence_max
        - accumulator.delivery_sequence_min
        + 1
        if (
            accumulator.delivery_sequence_min is not None
            and accumulator.delivery_sequence_max is not None
        )
        else 0
    )
    return {
        "format": "throwdb-workload-log-inspection",
        "version": 2,
        "log_version": 2,
        "logs": str(directory.resolve()),
        "limits": {
            "max_files": max_files,
            "max_total_bytes": max_total_bytes,
            "max_file_bytes": max_file_bytes,
            "max_report_entries": max_report_entries,
            "max_prefix_entries": max_prefix_entries,
            "max_active_requests": max_active_requests,
        },
        "collection": collection,
        "configuration": configuration,
        "configuration_variants": len(configurations),
        "files": {
            "discovered": len(paths),
            "valid": valid_files,
            "finalized": sum(
                path.suffix == ".tdbrl" for path, _ in paths
            ),
            "open": sum(path.suffix == ".open" for path, _ in paths),
            "invalid_finalized": invalid_finalized,
            "invalid_open": invalid_open,
            "torn_open": torn_open,
            "logical_duplicates": logical_duplicates,
            "bytes": actual_total_bytes,
            "reports": reports,
            "reports_omitted": reports_omitted,
        },
        "requests": {
            "begun": accumulator.begun_requests,
            "complete_for_training": accumulator.complete_requests,
            "non_success_outcomes": accumulator.non_success_outcomes,
            "incomplete_outcomes": accumulator.incomplete_requests,
            "left_censored": accumulator.left_censored_requests,
            "right_censored": accumulator.right_censored_requests,
            "interior_incomplete": (
                accumulator.interior_incomplete_requests
            ),
            "methods": dict(sorted(accumulator.methods.items())),
            "statuses": dict(sorted(accumulator.statuses.items())),
            "prefix_dimensions": {
                str(key): value
                for key, value in sorted(accumulator.prefixes.items())
            },
            "max_parallel": accumulator.max_parallel_requests,
            "total_duration_ns": accumulator.request_time_ns,
            "first_unix_ns": accumulator.first_unix_ns,
            "last_unix_ns": accumulator.last_unix_ns,
            "reports": accumulator.request_reports,
            "reports_omitted": accumulator.request_reports_omitted,
        },
        "id_chunks": {
            "records_for_complete_requests": (
                accumulator.id_chunk_records
            ),
            "ids_for_complete_requests": accumulator.ids_in_chunks,
            "payload_bytes_for_complete_requests": (
                accumulator.ids_in_chunks * 8
            ),
        },
        "batches": {
            "records_for_complete_requests": accumulator.batch_records,
            "rows_for_complete_requests": accumulator.batch_rows,
            "backends": dict(
                sorted(accumulator.batch_backends.items())
            ),
            "size_buckets": dict(
                sorted(accumulator.batch_size_buckets.items())
            ),
            "timing_totals_ns": dict(
                sorted(accumulator.batch_timings.items())
            ),
            "physical_metric_totals": dict(
                sorted(accumulator.batch_metrics.items())
            ),
            "delivery_sequence": {
                "nonzero_count": accumulator.delivery_sequence_count,
                "minimum": accumulator.delivery_sequence_min,
                "maximum": accumulator.delivery_sequence_max,
                "span": delivery_span,
                "unobserved_within_span": max(
                    0,
                    delivery_span
                    - accumulator.delivery_sequence_count,
                ),
                "physical_inversions": (
                    accumulator.delivery_physical_inversions
                ),
            },
        },
        "counters": {
            **dict(sorted(counters.items())),
            "process_instances": len(histories),
            "complete_instances": (
                len(histories) - incomplete_counter_instances
            ),
            "incomplete_instances": incomplete_counter_instances,
            "incomplete_terminal_partition_instances": (
                incomplete_terminal_partition_instances
            ),
            "regressed_instances": counter_regressions,
            "segment_sequence_gap_instances": (
                segment_sequence_gaps
            ),
            "lifecycle_sequence_gap_instances": len(
                accumulator.sequence_gap_instances
            ),
            "delivery_sequence_order_instances": len(
                accumulator.delivery_order_instances
            ),
            "saturated_instances": saturated_counter_instances,
            "dropped_records": dropped_records,
            "dropped_ids": dropped_ids,
        },
        "quality": {
            "suitable": exact_capture and not reasons,
            "exact_capture": exact_capture,
            "reasons": sorted(set(reasons)),
            "warnings": sorted(set(warnings)),
        },
    }


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
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
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
    max_active_requests = _positive_bound(
        max_active_requests, "max_active_requests"
    )
    if max_active_requests > DEFAULT_MAX_FILES:
        raise WorkloadLogError(
            "max_active_requests must not exceed the hard "
            f"{DEFAULT_MAX_FILES}-request reconstruction ceiling"
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

    detected_versions: set[int] = set()
    for path, _ in paths:
        _check_deadline(_deadline_monotonic)
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                magic = os.pread(descriptor, 8, 0)
            finally:
                os.close(descriptor)
        except OSError:
            continue
        if magic == _HEADER_MAGIC:
            detected_versions.add(_VERSION)
        elif magic == _V2_HEADER_MAGIC:
            detected_versions.add(_V2_VERSION)
    if _V2_VERSION in detected_versions:
        return _inspect_workload_logs_v2(
            directory,
            paths,
            storage=storage,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            max_file_bytes=max_file_bytes,
            max_report_entries=max_report_entries,
            max_prefix_entries=max_prefix_entries,
            max_active_requests=max_active_requests,
            deadline_monotonic=_deadline_monotonic,
            mixed_log_versions=_VERSION in detected_versions,
        )

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
    "DEFAULT_MAX_ACTIVE_REQUESTS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_PREFIX_ENTRIES",
    "DEFAULT_MAX_REPORT_ENTRIES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "WorkloadLogError",
    "inspect_workload_logs",
]
