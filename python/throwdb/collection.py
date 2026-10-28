"""High-level collection API built on top of the strict native bindings."""

from __future__ import annotations

import copy
import math
import numbers
import operator
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from types import TracebackType
from typing import Any, Final, Literal, TypeAlias, TypedDict, cast, overload

import numpy as np

from ._array_validation import (
    _StreamingNumericIdSource,
    _normalize_numeric_ids,
    _require_disjoint_numpy_arrays,
    _validate_id_scan_output,
    _validate_output,
    _validate_stream_output,
)
from ._throwdb import (
    NativeCollection,
    NativeCollectionBuilder,
    NativeIdScan,
    NativeReadView,
    NativeRecordScan,
    ReaderBackendError,
)
from ._device import StorageProbeResult, probe_collection_storage
from ._tuning import (
    READER_DEFAULTS,
    ReaderProfile,
    RegistryKey,
    TuningRegistryError,
    create_registry,
    load_registry,
    normalize_reader_options,
    remove_profile,
    reset_registry,
    upsert_profile,
)
from .id_scan import (
    ID_LOCATION_DTYPE,
    IdScanSnapshot,
    LocatedIdBatch,
    LocatedIds,
)
from .record_scan import RecordBatch


_UINT32_MAX: Final = (1 << 32) - 1
_ROW_REF_SEGMENT_ID_MAX: Final = _UINT32_MAX - 1
_UINT64_MAX: Final = (1 << 64) - 1
_DEFAULT_SEGMENT_SIZE_BYTES: Final = 16 << 30
_DEFAULT_ID_SCAN_BATCH_ROWS: Final = 65_536
_DEFAULT_STREAM_BATCH_BYTES: Final = 64 << 20
_DEFAULT_STREAM_SNAPSHOT_BYTES: Final = 256 << 20
_MAX_PARTITIONED_FOLD_RUNS: Final = 4096
_DEFAULT_FOLD_MAX_INPUT_BYTES: Final = 8 << 30
_DEFAULT_FOLD_MAX_INPUT_ENTRIES: Final = 1_000_000
_DEFAULT_CLEAN_MAX_CANDIDATE_SEGMENTS: Final = 8
_DEFAULT_CLEAN_MAX_SIDECAR_SCAN_BYTES: Final = 1 << 30
_DEFAULT_CLEAN_MAX_SOURCE_BYTES: Final = 32 << 30
_DEFAULT_CLEAN_MAX_OUTPUT_BYTES: Final = 32 << 30
_DEFAULT_CLEAN_MAX_MEMORY_BYTES: Final = 2 << 30
_DEFAULT_CLEAN_MIN_GARBAGE_ROWS: Final = 1
_DEFAULT_CLEAN_MIN_RECLAIMABLE_BYTES: Final = 512 << 20
_DEFAULT_CLEAN_MIN_GARBAGE_RATIO_PPM: Final = 300_000
_DEFAULT_ORPHAN_MIN_AGE_SECONDS: Final = 3_600.0
_DEFAULT_ORPHAN_MAX_FILES: Final = 1_024
_DEFAULT_ORPHAN_MAX_BYTES: Final = 8 << 30
_DEFAULT_MAINTAIN_MAX_SECONDS: Final = 300.0
_DEFAULT_MAINTAIN_MAX_ACTIONS: Final = 5
_DEFAULT_MAINTAIN_ROTATE_WAL_BYTES: Final = 1 << 30
_DEFAULT_MAINTAIN_MATERIALIZE_MAX_WALS: Final = 8
_DEFAULT_MAINTAIN_MATERIALIZE_MAX_INPUT_BYTES: Final = 8 << 30
_DEFAULT_MAINTAIN_MATERIALIZE_MAX_INPUT_ENTRIES: Final = 2_000_000
_DEFAULT_MAINTAIN_MATERIALIZE_MAX_MEMORY_BYTES: Final = 1 << 30
_DEFAULT_MAX_WAL_BACKLOG_BYTES: Final = 16 << 30
_DEFAULT_MAX_WAL_BACKLOG_OPERATIONS: Final = 4_000_000
_DEFAULT_MAX_OVERLAY_ENTRIES: Final = 4_000_000
_DEFAULT_MAX_OVERLAY_MEMORY_BYTES: Final = 256 << 20

_ORPHAN_INVENTORY_CLASS_NAMES: Final = (
    "reachable",
    "superseded_catalog",
    "temp_publication",
    "unreferenced_wal",
    "unknown_do_not_touch",
    "complete_unreferenced_segment_pair",
    "partial_unreferenced_segment_pair",
    "unreferenced_snapshot",
    "unreferenced_run",
)
_ORPHAN_INVENTORY_CLASSES: Final = frozenset(
    _ORPHAN_INVENTORY_CLASS_NAMES
)

_READER_POLICY_ABI: Final = "throwdb-reader-v2"


OrphanClassification: TypeAlias = Literal[
    "reachable",
    "superseded_catalog",
    "temp_publication",
    "unreferenced_wal",
    "unknown_do_not_touch",
    "complete_unreferenced_segment_pair",
    "partial_unreferenced_segment_pair",
    "unreferenced_snapshot",
    "unreferenced_run",
]


class OrphanInventoryEntry(TypedDict):
    """One file in a read-only orphan-collection plan."""

    path: str
    classification: OrphanClassification
    bytes: int
    modified_unix_ns: int
    candidate: bool
    eligible: bool
    reason: str


class OrphanCollectionReport(TypedDict):
    """JSON-serializable report returned by orphan dry-run operations."""

    format: str
    version: int
    dry_run: bool
    snapshot_unix_ns: int
    current_generation: int
    current_catalog_id: str
    min_age_seconds: float
    scanned_file_count: int
    total_bytes: int
    candidate_file_count: int
    candidate_bytes: int
    eligible_file_count: int
    deferred_candidate_count: int
    reclaimable_bytes: int
    unknown_file_count: int
    classification_counts: dict[str, int]
    classification_bytes: dict[str, int]
    entries: list[OrphanInventoryEntry]


class OrphanCollectionExecutionEntry(OrphanInventoryEntry):
    selected: bool
    removed: bool
    durable: bool
    error: str | None


class OrphanCollectionExecutionReport(OrphanCollectionReport):
    max_files: int
    max_bytes: int
    selected_file_count: int
    selected_bytes: int
    budget_deferred_file_count: int
    budget_deferred_bytes: int
    removed_file_count: int
    removed_bytes: int
    failed_file_count: int
    synced_directories: list[str]
    durability_complete: bool
    policy_complete: bool
    reclamation_complete: bool
    entries: list[OrphanCollectionExecutionEntry]  # type: ignore[misc]


class MaintenanceReport(TypedDict):
    format: str
    version: int
    dry_run: bool
    started_unix_ns: int
    finished_unix_ns: int
    elapsed_seconds: float
    policy: dict[str, object]
    before: dict[str, object]
    after: dict[str, object]
    actions: list[dict[str, object]]
    actions_performed: int
    deadline_reached: bool
    action_budget_reached: bool
    work_performed: bool


class SegmentCleaningReport(TypedDict):
    generation: int
    scanned_segment_count: int
    scanned_row_count: int
    scanned_sidecar_bytes: int
    skipped_budget_segment_count: int
    next_start_after_segment_id: int
    candidate_segment_id: int
    source_rows: int
    relocated_rows: int
    garbage_rows: int
    source_bytes: int
    output_segment_count: int
    output_segment_bytes: int
    output_run_bytes: int
    reclaimable_bytes: int
    dry_run: bool
    candidate_found: bool
    cleaned: bool
    budget_blocked: bool
    cycle_complete: bool
    reclamation_complete: bool


def _storage_match_level(probe: StorageProbeResult) -> str:
    if probe.media.reuse_scope in {"durable", "host"}:
        return "exact"
    if probe.media.reuse_scope == "compatible":
        return "compatible"
    return "fallback"


def _storage_identity_is_reusable(probe: StorageProbeResult) -> bool:
    return _storage_match_level(probe) != "fallback"


def _non_reusable_storage_reason(probe: StorageProbeResult) -> str:
    evidence = ",".join(probe.media.evidence) or "none"
    return (
        "the current storage target is not eligible for an automatically "
        "reusable tuning profile "
        f"(kind={probe.media.kind}, confidence={probe.media.confidence}, "
        f"reuse_scope={probe.media.reuse_scope}, evidence={evidence}); "
        "use dry_run=True / --dry-run for a non-persisted measurement, or "
        "tune the collection on an exact or compatibility-classified mount"
    )


def _collection_root_identity(
    path: str | bytes | PathLike[str] | PathLike[bytes],
) -> tuple[int, ...]:
    """Capture both the directory entry and followed collection root."""

    entry = os.stat(path, follow_symlinks=False)
    target = os.stat(path, follow_symlinks=True)
    return (
        int(entry.st_dev),
        int(entry.st_ino),
        int(entry.st_mode),
        int(target.st_dev),
        int(target.st_ino),
        int(target.st_mode),
    )


@dataclass(frozen=True, slots=True)
class CollectionSchema:
    """Immutable logical schema of one collection snapshot.

    Numeric collections support both ``float16`` and ``float32`` embeddings.
    ``utf8`` IDs remain reserved for the future string index and are rejected
    explicitly by ``Collection.create``.
    """

    id_type: str = "uint64"
    dimension: int = 1024
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if not isinstance(self.id_type, str):
            raise TypeError("id_type must be a string")
        if self.id_type not in {"uint64", "utf8"}:
            raise ValueError("id_type must be 'uint64' or 'utf8'")

        if isinstance(self.dimension, (bool, np.bool_)):
            raise TypeError("dimension must be an integer, not bool")
        try:
            dimension = operator.index(self.dimension)
        except TypeError as error:
            raise TypeError("dimension must be an integer") from error
        if not 1 <= dimension <= _UINT32_MAX:
            raise ValueError(f"dimension must be between 1 and {_UINT32_MAX}")

        try:
            dtype = np.dtype(self.dtype).name
        except TypeError as error:
            raise TypeError(f"invalid embedding dtype {self.dtype!r}") from error
        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be 'float16' or 'float32'")

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "dtype", dtype)


def _normalize_embeddings(
    embeddings: object,
    *,
    expected_rows: int,
    schema: CollectionSchema,
) -> np.ndarray:
    if isinstance(embeddings, np.ma.MaskedArray):
        raise TypeError("embeddings must not be a masked array")
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a NumPy ndarray")
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape (N, dimension), "
            f"got {embeddings.shape}"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError("ids and embeddings must have the same number of rows")
    if embeddings.shape[1] != schema.dimension:
        raise ValueError(
            "embedding dimension does not match the collection schema: "
            f"expected {schema.dimension}, got {embeddings.shape[1]}"
        )

    expected_dtype = np.dtype(schema.dtype)
    if embeddings.dtype.newbyteorder("=") != expected_dtype:
        raise TypeError(
            f"embeddings must have dtype {schema.dtype}, got {embeddings.dtype}"
        )
    if (
        embeddings.dtype == expected_dtype
        and embeddings.dtype.isnative
        and embeddings.flags.c_contiguous
        and embeddings.flags.aligned
    ):
        return embeddings
    return np.require(embeddings, dtype=expected_dtype, requirements=["C", "A"])


def _validate_prefix_dimensions(value: object, dimension: int) -> int:
    if value is None:
        return dimension
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("prefix_dimensions must be an integer, not bool")
    try:
        prefix = operator.index(value)
    except TypeError as error:
        raise TypeError("prefix_dimensions must be an integer") from error
    if not 1 <= prefix <= dimension:
        raise ValueError(
            f"prefix_dimensions must be between 1 and {dimension}, got {prefix}"
        )
    return prefix


def _validate_segment_size_bytes(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("segment_size_bytes must be an integer, not bool")
    try:
        size = operator.index(value)
    except TypeError as error:
        raise TypeError("segment_size_bytes must be an integer") from error
    if not 1 <= size <= _UINT64_MAX:
        raise ValueError(
            "segment_size_bytes must be between 1 and 2**64-1"
        )
    return size


def _validate_format_version(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("format_version must be an integer, not bool")
    try:
        version = operator.index(value)
    except TypeError as error:
        raise TypeError("format_version must be an integer") from error
    if version != 7:
        raise ValueError("format_version must be 7")
    return version


def _validate_resource_limit(
    value: object,
    *,
    name: str,
    minimum: int = 1,
    maximum: int = _UINT64_MAX,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        validated = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if not minimum <= validated <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return validated


def _validate_orphan_options(
    *,
    dry_run: object,
    min_age_seconds: object,
    max_files: object = _DEFAULT_ORPHAN_MAX_FILES,
    max_bytes: object = _DEFAULT_ORPHAN_MAX_BYTES,
) -> tuple[float, int, int]:
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a bool")
    if isinstance(min_age_seconds, (bool, np.bool_)):
        raise TypeError("min_age_seconds must be a number, not bool")
    if not isinstance(
        min_age_seconds,
        (numbers.Real, np.integer, np.floating),
    ):
        raise TypeError("min_age_seconds must be a number")
    try:
        seconds = float(min_age_seconds)
    except (TypeError, ValueError) as error:
        raise TypeError("min_age_seconds must be a number") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "min_age_seconds must be finite and zero or greater"
        )
    validated_max_files = _validate_resource_limit(
        max_files, name="max_files"
    )
    validated_max_bytes = _validate_resource_limit(
        max_bytes, name="max_bytes"
    )
    return seconds, validated_max_files, validated_max_bytes


def _seconds_to_nanoseconds_ceil(seconds: float) -> int:
    """Convert even extremely large finite seconds without float overflow."""

    whole_seconds = int(seconds)
    fractional_seconds = seconds - whole_seconds
    return whole_seconds * 1_000_000_000 + math.ceil(
        fractional_seconds * 1_000_000_000
    )


def _native_orphan_min_age_ns(seconds: float) -> int:
    # Native metadata timestamps use uint64 nanoseconds. Saturation preserves
    # the useful semantics for a larger valid Python threshold: no present-day
    # candidate can become eligible merely because the API representation is
    # wider than the wire value.
    return min(_seconds_to_nanoseconds_ceil(seconds), _UINT64_MAX)


def _orphan_report(
    raw_inventory: object,
    *,
    min_age_seconds: float,
    snapshot_unix_ns: int | None = None,
) -> OrphanCollectionReport:
    if not isinstance(raw_inventory, dict):
        raise RuntimeError("native orphan inventory is not a dictionary")

    current_generation = raw_inventory.get("current_generation")
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 0
    ):
        raise RuntimeError("native orphan inventory has an invalid generation")

    current_catalog_id = raw_inventory.get("current_catalog_id")
    if (
        not isinstance(current_catalog_id, str)
        or len(current_catalog_id) != 32
        or any(
            character not in "0123456789abcdef"
            for character in current_catalog_id
        )
    ):
        raise RuntimeError("native orphan inventory has an invalid catalog ID")

    raw_entries = raw_inventory.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError("native orphan inventory has invalid entries")

    if snapshot_unix_ns is None:
        snapshot_unix_ns = time.time_ns()
    elif (
        not isinstance(snapshot_unix_ns, int)
        or isinstance(snapshot_unix_ns, bool)
        or snapshot_unix_ns < 0
    ):
        raise RuntimeError("native orphan result has an invalid snapshot time")
    minimum_age_ns = _seconds_to_nanoseconds_ceil(min_age_seconds)
    entries: list[OrphanInventoryEntry] = []
    total_bytes = 0
    candidate_file_count = 0
    eligible_file_count = 0
    unknown_file_count = 0
    candidate_bytes = 0
    reclaimable_bytes = 0
    classification_counts = {
        name: 0 for name in _ORPHAN_INVENTORY_CLASS_NAMES
    }
    classification_bytes = {
        name: 0 for name in _ORPHAN_INVENTORY_CLASS_NAMES
    }
    previous_path: str | None = None

    # A format-7 segment is published and reclaimed as one .dat/.ids unit.
    # Mirror the native executor's group-age policy in dry-run reports so a
    # young or invalid half defers both files and an execution report cannot
    # disagree with its reconstructed plan.
    complete_segment_pair_states: dict[str, dict[str, int]] = {}
    for raw_entry in raw_entries:
        if (
            isinstance(raw_entry, dict)
            and raw_entry.get("classification")
            == "complete_unreferenced_segment_pair"
        ):
            path = raw_entry.get("path")
            modified_unix_ns = raw_entry.get("modified_unix_ns")
            if (
                isinstance(path, str)
                and path.startswith("segments/segment-")
                and (path.endswith(".dat") or path.endswith(".ids"))
                and isinstance(modified_unix_ns, int)
                and not isinstance(modified_unix_ns, bool)
            ):
                stem = path[:-4]
                if not 0 <= modified_unix_ns <= snapshot_unix_ns:
                    state = 0
                elif snapshot_unix_ns - modified_unix_ns < minimum_age_ns:
                    state = 1
                else:
                    state = 2
                parts = complete_segment_pair_states.setdefault(stem, {})
                suffix = path[-4:]
                if suffix in parts:
                    raise RuntimeError(
                        "native orphan inventory contains a duplicate complete "
                        "segment pair member"
                    )
                parts[suffix] = state

    complete_segment_pair_state: dict[str, int] = {}
    for stem, states in complete_segment_pair_states.items():
        if set(states) != {".dat", ".ids"}:
            raise RuntimeError(
                "native orphan inventory contains an incomplete complete "
                "segment pair"
            )
        complete_segment_pair_state[stem] = min(states.values())

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("native orphan inventory contains an invalid entry")
        path = raw_entry.get("path")
        classification = raw_entry.get("classification")
        native_candidate = raw_entry.get("candidate")
        byte_count = raw_entry.get("bytes")
        modified_unix_ns = raw_entry.get("modified_unix_ns")
        if not isinstance(path, str) or not path:
            raise RuntimeError("native orphan inventory contains an invalid path")
        if previous_path is not None and path <= previous_path:
            raise RuntimeError(
                "native orphan inventory paths are not strictly sorted"
            )
        previous_path = path
        if (
            not isinstance(classification, str)
            or classification not in _ORPHAN_INVENTORY_CLASSES
        ):
            raise RuntimeError(
                "native orphan inventory contains an invalid classification"
            )
        classification = cast(OrphanClassification, classification)
        if not isinstance(native_candidate, bool):
            raise RuntimeError(
                "native orphan inventory contains an invalid candidate policy"
            )
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise RuntimeError(
                "native orphan inventory contains an invalid file size"
            )
        if not isinstance(modified_unix_ns, int) or isinstance(
            modified_unix_ns, bool
        ):
            raise RuntimeError(
                "native orphan inventory contains an invalid modification time"
            )

        # Candidate policy is format-specific. In particular, format 7 can
        # safely retire superseded catalogs and unreferenced segment
        # singletons under view.lock. Never infer this destructive policy
        # from a class name.
        candidate = native_candidate
        valid_modification_time = 0 <= modified_unix_ns <= snapshot_unix_ns
        eligible = (
            candidate
            and valid_modification_time
            and snapshot_unix_ns - modified_unix_ns >= minimum_age_ns
        )
        if classification == "complete_unreferenced_segment_pair":
            state = complete_segment_pair_state.get(path[:-4])
            if state is None:
                raise RuntimeError(
                    "native orphan inventory contains an invalid complete "
                    "segment pair path"
                )
            valid_modification_time = state != 0
            eligible = state == 2
        total_bytes += byte_count
        classification_counts[classification] += 1
        classification_bytes[classification] += byte_count
        if candidate:
            candidate_file_count += 1
            candidate_bytes += byte_count
            if not valid_modification_time:
                reason = "invalid_modification_time"
            elif eligible:
                eligible_file_count += 1
                reclaimable_bytes += byte_count
                reason = "minimum_age_reached"
            else:
                reason = "minimum_age_not_reached"
        elif classification == "reachable":
            reason = "reachable_from_current"
        elif classification == "superseded_catalog":
            reason = "retained_by_policy"
        elif classification == "partial_unreferenced_segment_pair":
            reason = "incomplete_pair_do_not_touch"
        elif classification == "unknown_do_not_touch":
            unknown_file_count += 1
            reason = "unknown_do_not_touch"
        else:
            raise RuntimeError(
                "native orphan inventory class has no collection policy"
            )

        entries.append(
            {
                "path": path,
                "classification": classification,
                "bytes": byte_count,
                "modified_unix_ns": modified_unix_ns,
                "candidate": candidate,
                "eligible": eligible,
                "reason": reason,
            }
        )

    return {
        "format": "throwdb-orphan-collection-plan",
        "version": 1,
        "dry_run": True,
        "snapshot_unix_ns": snapshot_unix_ns,
        "current_generation": current_generation,
        "current_catalog_id": current_catalog_id,
        "min_age_seconds": min_age_seconds,
        "scanned_file_count": len(entries),
        "total_bytes": total_bytes,
        "candidate_file_count": candidate_file_count,
        "candidate_bytes": candidate_bytes,
        "eligible_file_count": eligible_file_count,
        "deferred_candidate_count": (
            candidate_file_count - eligible_file_count
        ),
        "reclaimable_bytes": reclaimable_bytes,
        "unknown_file_count": unknown_file_count,
        "classification_counts": classification_counts,
        "classification_bytes": classification_bytes,
        "entries": entries,
    }


def _orphan_execution_report(
    raw_result: object,
    *,
    min_age_seconds: float,
    max_files: int,
    max_bytes: int,
) -> OrphanCollectionExecutionReport:
    if not isinstance(raw_result, dict):
        raise RuntimeError("native orphan collection result is not a dictionary")
    snapshot_unix_ns = raw_result.get("snapshot_unix_ns")
    min_age_ns = raw_result.get("min_age_ns")
    if (
        not isinstance(min_age_ns, int)
        or isinstance(min_age_ns, bool)
        or min_age_ns != _native_orphan_min_age_ns(min_age_seconds)
    ):
        raise RuntimeError("native orphan collection used an invalid age limit")
    if raw_result.get("max_files") != max_files:
        raise RuntimeError("native orphan collection used an invalid file budget")
    if raw_result.get("max_bytes") != max_bytes:
        raise RuntimeError("native orphan collection used an invalid byte budget")
    raw_entries = raw_result.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError("native orphan collection has invalid entries")

    inventory_entries: list[dict[str, object]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise RuntimeError("native orphan collection contains an invalid entry")
        inventory_entries.append(
            {
                "path": entry.get("path"),
                "classification": entry.get("classification"),
                "candidate": entry.get("candidate"),
                "bytes": entry.get("bytes"),
                "modified_unix_ns": entry.get("modified_unix_ns"),
            }
        )
    plan = _orphan_report(
        {
            "current_generation": raw_result.get("current_generation"),
            "current_catalog_id": raw_result.get("current_catalog_id"),
            "entries": inventory_entries,
        },
        min_age_seconds=min_age_seconds,
        snapshot_unix_ns=snapshot_unix_ns,
    )

    dispositions = {
        "not_candidate",
        "invalid_modification_time",
        "minimum_age_not_reached",
        "max_files_reached",
        "max_bytes_reached",
        "removed",
        "no_longer_present",
        "identity_changed",
        "unlink_failed",
    }
    execution_entries: list[OrphanCollectionExecutionEntry] = []
    selected_file_count = 0
    selected_bytes = 0
    budget_deferred_file_count = 0
    budget_deferred_bytes = 0
    removed_file_count = 0
    removed_bytes = 0
    failed_file_count = 0
    for planned, raw_entry in zip(plan["entries"], raw_entries, strict=True):
        disposition = raw_entry.get("disposition")
        eligible = raw_entry.get("eligible")
        selected = raw_entry.get("selected")
        removed = raw_entry.get("removed")
        durable = raw_entry.get("durable")
        error = raw_entry.get("error")
        if not isinstance(disposition, str) or disposition not in dispositions:
            raise RuntimeError("native orphan collection has an invalid disposition")
        if not all(
            isinstance(value, bool)
            for value in (eligible, selected, removed, durable)
        ):
            raise RuntimeError("native orphan collection has invalid state flags")
        if error is not None and not isinstance(error, str):
            raise RuntimeError("native orphan collection has an invalid error")
        if eligible != planned["eligible"]:
            raise RuntimeError("native orphan collection eligibility disagrees")
        if selected and not eligible:
            raise RuntimeError("native orphan collection selected an ineligible file")
        if removed and (not selected or disposition != "removed"):
            raise RuntimeError("native orphan collection has invalid removal state")
        if durable and not removed:
            raise RuntimeError("native orphan collection has invalid durability state")
        if disposition in {
            "removed",
            "no_longer_present",
            "identity_changed",
            "unlink_failed",
        } and not selected:
            raise RuntimeError("native orphan collection has an invalid outcome")
        if disposition in {
            "max_files_reached",
            "max_bytes_reached",
        } and (selected or not eligible):
            raise RuntimeError("native orphan collection has an invalid budget state")

        byte_count = planned["bytes"]
        if selected:
            selected_file_count += 1
            selected_bytes += byte_count
        if disposition in {"max_files_reached", "max_bytes_reached"}:
            budget_deferred_file_count += 1
            budget_deferred_bytes += byte_count
        if removed:
            removed_file_count += 1
            removed_bytes += byte_count
        elif selected and disposition not in {"no_longer_present"}:
            failed_file_count += 1
        execution_entries.append(
            {
                **planned,
                "selected": selected,
                "removed": removed,
                "durable": durable,
                "reason": (
                    planned["reason"]
                    if disposition == "not_candidate"
                    else disposition
                ),
                "error": error,
            }
        )

    synced_directories = raw_result.get("synced_directories")
    durability_complete = raw_result.get("durability_complete")
    if (
        not isinstance(synced_directories, list)
        or not all(isinstance(value, str) for value in synced_directories)
        or len(synced_directories) != len(set(synced_directories))
        or any(
            value not in {
                ".",
                "catalogs",
                "wal",
                "segments",
                "index/partitions",
                "index/runs",
            }
            for value in synced_directories
        )
    ):
        raise RuntimeError("native orphan collection has invalid synced directories")
    if not isinstance(durability_complete, bool):
        raise RuntimeError("native orphan collection has invalid durability state")
    if selected_file_count > max_files or selected_bytes > max_bytes:
        raise RuntimeError("native orphan collection exceeded a hard budget")
    reclamation_complete = (
        durability_complete
        and failed_file_count == 0
        and budget_deferred_file_count == 0
        and all(not entry["removed"] or entry["durable"] for entry in execution_entries)
    )
    return {
        **plan,
        "format": "throwdb-orphan-collection-report",
        "version": 1,
        "dry_run": False,
        "max_files": max_files,
        "max_bytes": max_bytes,
        "selected_file_count": selected_file_count,
        "selected_bytes": selected_bytes,
        "budget_deferred_file_count": budget_deferred_file_count,
        "budget_deferred_bytes": budget_deferred_bytes,
        "removed_file_count": removed_file_count,
        "removed_bytes": removed_bytes,
        "failed_file_count": failed_file_count,
        "synced_directories": list(synced_directories),
        "durability_complete": durability_complete,
        "policy_complete": reclamation_complete,
        "reclamation_complete": reclamation_complete,
        "entries": execution_entries,
    }


def _layout_identity(
    path: str | bytes | PathLike[str] | PathLike[bytes],
) -> tuple[dict[str, object], int]:
    metadata = dict(NativeCollection.inspect_metadata(path))
    record_size = int(metadata["record_size"])
    layout = {
        "reader_policy_abi": _READER_POLICY_ABI,
        "format_version": int(metadata["format_version"]),
        "dimension": int(metadata["dimension"]),
        "dtype": str(metadata["dtype"]),
        "record_size": record_size,
        "segment_size_bytes": int(metadata["segment_size_bytes"]),
        "layout": str(metadata["layout"]),
        "feature_discriminator": str(metadata["feature_discriminator"]),
        "required_feature_bits": int(metadata["required_feature_bits"]),
        "optional_feature_bits": int(metadata["optional_feature_bits"]),
    }
    return layout, record_size


def _reader_capabilities(
    io_uring_queue_depth: int = 32,
    io_uring_workers: int = 1,
) -> dict[str, object]:
    capabilities = dict(
        NativeCollection.reader_capabilities(
            io_uring_queue_depth=io_uring_queue_depth,
            io_uring_workers=io_uring_workers,
        )
    )
    return {
        "io_uring_available": bool(capabilities["io_uring_available"]),
        "io_uring_probe_queue_depth": int(
            capabilities["io_uring_probe_queue_depth"]
        ),
        "io_uring_probe_workers": int(
            capabilities["io_uring_probe_workers"]
        ),
        "io_uring_reason": capabilities["io_uring_reason"],
    }


def _execute_reader_calibration(
    native: object,
    *,
    path: str | bytes | PathLike[str] | PathLike[bytes],
    record_size: int,
    current_options: Mapping[str, object] | None,
    seed: int,
    max_read_bytes: int,
    max_memory_bytes: int,
    deadline: float,
    expected_root_identity: tuple[int, int],
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    """Execute the authoritative matrix outside the embedding application."""

    from ._calibration_process import run_isolated_calibration

    return run_isolated_calibration(
        path,
        record_size=record_size,
        current_options=current_options,
        seed=seed,
        max_read_bytes=max_read_bytes,
        max_memory_bytes=max_memory_bytes,
        deadline=deadline,
        expected_dimension=int(getattr(native, "dimension")),
        expected_dtype=str(getattr(native, "dtype")),
        expected_format_version=int(getattr(native, "format_version")),
        expected_root_identity=expected_root_identity,
        progress=progress,
    )


def _probe_storage_identity(
    path: str | bytes | PathLike[str] | PathLike[bytes],
    salt: str,
    capabilities: Mapping[str, object],
) -> tuple[
    StorageProbeResult,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    try:
        salt_bytes = bytes.fromhex(salt)
    except ValueError as error:
        raise TuningRegistryError("tuning registry has an invalid salt") from error
    capability_key = {
        "io_uring_available": bool(capabilities["io_uring_available"]),
        "io_uring_probe_queue_depth": int(
            capabilities["io_uring_probe_queue_depth"]
        ),
        "io_uring_probe_workers": int(
            capabilities["io_uring_probe_workers"]
        ),
    }
    probe = probe_collection_storage(
        path,
        collection_salt=salt_bytes,
        reader_abi=_READER_POLICY_ABI,
        io_uring_policy=capability_key,
    )
    return (
        probe,
        probe.media.as_key_dict(),
        probe.runtime.as_key_dict(),
        dict(capabilities),
    )


def _probe_identity(
    path: str | bytes | PathLike[str] | PathLike[bytes],
    salt: str,
) -> tuple[
    StorageProbeResult,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    return _probe_storage_identity(path, salt, _reader_capabilities())


def _fallback_tuning_profile(
    *,
    mode: str,
    status: str,
    reason: str,
    compatibility: Mapping[str, object] | None = None,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "mode": mode,
        "source": "fallback",
        "match_level": "fallback",
        "profile_reuse_level": "fallback",
        "status": status,
        "reason": reason,
        "needs_tuning": mode == "auto",
        "reader_policy_abi": _READER_POLICY_ABI,
        **READER_DEFAULTS,
    }
    if compatibility is not None:
        profile["compatibility"] = copy.deepcopy(dict(compatibility))
    return profile


def _profile_key_dict(key: RegistryKey) -> dict[str, str]:
    return {
        "device_digest": key.device_digest,
        "runtime_digest": key.runtime_digest,
        "layout_digest": key.layout_digest,
    }


def _stored_profile_dict(profile: ReaderProfile) -> dict[str, object]:
    return {
        "key": _profile_key_dict(profile.key),
        "device": copy.deepcopy(dict(profile.device)),
        "runtime": copy.deepcopy(dict(profile.runtime)),
        "layout": copy.deepcopy(dict(profile.layout)),
        "options": copy.deepcopy(dict(profile.options)),
        "source": profile.source,
        "created_at_ns": profile.created_at_ns,
        "updated_at_ns": profile.updated_at_ns,
    }


def _resolve_automatic_tuning(
    path: str | bytes | PathLike[str] | PathLike[bytes],
    *,
    mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    required = mode == "persisted"
    loaded = load_registry(path, strict=False)
    if loaded.status != "ok" or loaded.registry is None:
        reason = (
            "no reader profiles have been saved for this collection"
            if loaded.status == "missing"
            else f"saved reader profiles are unusable: {loaded.error}"
        )
        if required:
            raise TuningRegistryError(reason)
        return {}, _fallback_tuning_profile(
            mode=mode,
            status=f"registry_{loaded.status}",
            reason=reason,
        )

    registry = loaded.registry
    layout, record_size = _layout_identity(path)
    try:
        probe, device, runtime, capabilities = _probe_identity(
            path, registry.salt
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        reason = f"storage identity probe failed: {error}"
        if required:
            raise TuningRegistryError(reason) from error
        return {}, _fallback_tuning_profile(
            mode=mode,
            status="probe_failed",
            reason=reason,
            compatibility={"layout": layout},
        )

    compatibility = {
        "device": probe.media.as_redacted_dict(),
        "runtime": probe.runtime.as_redacted_dict(),
        "layout": layout,
        "match_level": _storage_match_level(probe),
        "probed_path_count": probe.probed_path_count,
        "used_root_fallback": probe.used_root_fallback,
        "reader_capabilities": capabilities,
    }
    if not _storage_identity_is_reusable(probe):
        reason = (
            "the current storage target cannot be identified safely; "
            "a saved profile will not be reused"
        )
        if required:
            raise TuningRegistryError(reason)
        return {}, _fallback_tuning_profile(
            mode=mode,
            status="device_unrecognized",
            reason=reason,
            compatibility=compatibility,
        )

    key = RegistryKey.from_mappings(
        device,
        runtime,
        layout,
        salt=registry.salt,
    )
    stored = next(
        (profile for profile in registry.profiles if profile.key == key),
        None,
    )
    if stored is None:
        reason = "no saved profile matches the current device, runtime, and layout"
        if required:
            raise TuningRegistryError(reason)
        return {}, _fallback_tuning_profile(
            mode=mode,
            status="no_match",
            reason=reason,
            compatibility=compatibility,
        )

    try:
        options = normalize_reader_options(
            stored.options,
            persisted=True,
            record_size=record_size,
        )
    except (TypeError, ValueError) as error:
        reason = f"saved reader profile is invalid for this layout: {error}"
        if required:
            raise TuningRegistryError(reason) from error
        return {}, _fallback_tuning_profile(
            mode=mode,
            status="profile_invalid",
            reason=reason,
            compatibility=compatibility,
        )

    if options["io_backend"] == "io_uring":
        try:
            exact_capabilities = _reader_capabilities(
                int(options["io_uring_queue_depth"]),
                int(options["io_uring_workers"]),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            exact_capabilities = {
                "io_uring_available": False,
                "io_uring_probe_queue_depth": options[
                    "io_uring_queue_depth"
                ],
                "io_uring_probe_workers": options["io_uring_workers"],
                "io_uring_reason": str(error),
            }
        compatibility["selected_reader_capabilities"] = exact_capabilities
        if not exact_capabilities["io_uring_available"]:
            reason = (
                "the saved io_uring topology is unavailable in the current "
                f"runtime: {exact_capabilities['io_uring_reason']}"
            )
            if required:
                raise TuningRegistryError(reason)
            return {}, _fallback_tuning_profile(
                mode=mode,
                status="backend_unavailable",
                reason=reason,
                compatibility=compatibility,
            )

    match_level = _storage_match_level(probe)
    return options, {
        "mode": mode,
        "source": "persisted",
        "match_level": match_level,
        "profile_reuse_level": match_level,
        "status": "selected",
        "reason": (
            "exact device, runtime, and layout profile match"
            if match_level == "exact"
            else "compatible storage, runtime, and layout profile match"
        ),
        "needs_tuning": False,
        "reader_policy_abi": _READER_POLICY_ABI,
        "registry_generation": registry.generation,
        "profile_origin": stored.source,
        "profile_key": _profile_key_dict(key),
        "compatibility": compatibility,
        **options,
    }


def _resolve_tuning(
    path: str | bytes | PathLike[str] | PathLike[bytes],
    tuning: str | Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(tuning, str):
        if tuning == "conservative":
            return {}, _fallback_tuning_profile(
                mode=tuning,
                status="registry_bypassed",
                reason="saved reader profiles were bypassed explicitly",
            )
        if tuning in {"auto", "persisted"}:
            return _resolve_automatic_tuning(path, mode=tuning)
        raise ValueError(
            "tuning must be 'auto', 'persisted', 'conservative', or a "
            "mapping of reader options"
        )
    if not isinstance(tuning, Mapping):
        raise TypeError(
            "tuning must be 'auto', 'persisted', 'conservative', or a "
            "mapping of reader options"
        )

    options = normalize_reader_options(tuning)
    return options, {
        "mode": "manual",
        "source": "manual",
        "match_level": "manual",
        "profile_reuse_level": "manual",
        "status": "explicit",
        "reason": "explicit user configuration",
        "needs_tuning": False,
        "reader_policy_abi": _READER_POLICY_ABI,
        **options,
    }


def _validate_id_scan_batch_rows(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("batch_rows must be an integer, not bool")
    try:
        rows = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_rows must be an integer") from error
    if not 1 <= rows <= _UINT32_MAX:
        raise ValueError(
            f"batch_rows must be between 1 and {_UINT32_MAX}"
        )
    return rows


def _validate_stream_byte_limit(value: object, *, name: str) -> int:
    return _validate_resource_limit(value, name=name)


def _effective_stream_batch_rows(
    batch_rows: int,
    max_batch_bytes: int,
    *,
    bytes_per_row: int,
) -> int:
    if bytes_per_row <= 0:
        raise RuntimeError("stream row byte estimate must be positive")
    if bytes_per_row > max_batch_bytes:
        raise ValueError(
            "max_batch_bytes cannot fit one streamed row: "
            f"requires at least {bytes_per_row} bytes"
        )
    return min(batch_rows, max_batch_bytes // bytes_per_row)


def _scan_snapshot_bytes(metadata: object) -> int:
    if not isinstance(metadata, Mapping):
        raise RuntimeError("native scan returned invalid metadata")
    value = metadata.get("snapshot_bytes")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _UINT64_MAX
    ):
        raise RuntimeError("native scan returned invalid snapshot_bytes")
    return value


class _SingleConsumerOperation:
    """A fork-aware, fail-fast guard for one stateful local cursor."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            # A lock held by a vanished sibling thread must never strand the
            # child after fork. Native handles still reject child-process I/O.
            self._owner_pid = current_pid
            self._lock = threading.Lock()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                f"{self._label} is already in use by another thread"
            )
        return self._lock


def _id_scan_snapshot(metadata: object) -> IdScanSnapshot:
    if not isinstance(metadata, Mapping):
        raise RuntimeError("native ID scan returned invalid metadata")
    names = (
        "collection_id_low",
        "collection_id_high",
        "catalog_generation",
        "committed_sequence",
        "live_count",
    )
    values: dict[str, int] = {}
    for name in names:
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"native ID scan metadata field {name!r} is invalid"
            )
        values[name] = value
    low = values["collection_id_low"]
    high = values["collection_id_high"]
    generation = values["catalog_generation"]
    sequence = values["committed_sequence"]
    count = values["live_count"]
    if low > _UINT64_MAX or high > _UINT64_MAX:
        raise RuntimeError("native ID scan collection identity is invalid")
    if generation > _UINT64_MAX or sequence > _UINT64_MAX:
        raise RuntimeError("native ID scan view identity is invalid")
    if count > _UINT64_MAX:
        raise RuntimeError("native ID scan count is invalid")
    collection_id = f"{high:016x}{low:016x}"
    token = f"local:{collection_id}:{generation:016x}:{sequence:016x}"
    return IdScanSnapshot(
        collection_id=collection_id,
        catalog_generation=generation,
        committed_sequence=sequence,
        count=count,
        token=token,
    )


class EmbeddedIdScan(Iterator[np.ndarray | LocatedIdBatch]):
    """Snapshot cursor over live IDs in physical segment/row order."""

    def __init__(
        self,
        native: NativeIdScan,
        *,
        batch_rows: int,
        with_locations: bool,
    ) -> None:
        self._native = native
        self._batch_rows = batch_rows
        self._with_locations = with_locations
        metadata = native.metadata
        self._snapshot = _id_scan_snapshot(metadata)
        self._snapshot_bytes = _scan_snapshot_bytes(metadata)
        self._emitted = 0
        self._closed = False
        self._finished = self._snapshot.count == 0 or bool(native.done)
        self._operation = _SingleConsumerOperation("ID scan")

    @property
    def snapshot(self) -> IdScanSnapshot:
        return self._snapshot

    @property
    def snapshot_count(self) -> int:
        return self._snapshot.count

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    @property
    def done(self) -> bool:
        return self._finished

    def read_into(
        self,
        ids: object,
        locations: object | None = None,
    ) -> int:
        operation = self._operation.acquire()
        try:
            return self._read_into_locked(ids, locations)
        finally:
            operation.release()

    def _read_into_locked(
        self,
        ids: object,
        locations: object | None,
    ) -> int:
        if self._closed:
            raise RuntimeError("ID scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
        )
        if self._finished:
            return 0
        count = int(self._native.read_into(output_ids, output_locations))
        if not 0 <= count <= output_ids.size:
            raise RuntimeError("native ID scan returned an invalid row count")
        if count == 0 and not self._native.done:
            raise RuntimeError("native ID scan stopped without reaching EOF")
        if self._emitted + count > self._snapshot.count:
            raise RuntimeError("native ID scan exceeded its snapshot count")
        self._emitted += count
        if self._native.done:
            if self._emitted != self._snapshot.count:
                raise RuntimeError(
                    "native ID scan ended before its snapshot count"
                )
            self._finished = True
        return count

    def __iter__(self) -> EmbeddedIdScan:
        return self

    def __next__(self) -> np.ndarray | LocatedIdBatch:
        operation = self._operation.acquire()
        try:
            if self._closed or self._finished:
                raise StopIteration
            ids = np.empty(self._batch_rows, dtype=np.uint64)
            locations = (
                np.empty(self._batch_rows, dtype=ID_LOCATION_DTYPE)
                if self._with_locations
                else None
            )
            emitted_offset = self._emitted
            count = self._read_into_locked(ids, locations)
            if count == 0:
                raise StopIteration
            ids = ids[:count]
            if locations is None:
                return ids
            return LocatedIdBatch(
                ids=ids,
                locations=locations[:count],
                emitted_offset=emitted_offset,
                snapshot=self._snapshot,
            )
        finally:
            operation.release()

    def close(self) -> None:
        operation = self._operation.acquire()
        try:
            self._close_native_locked()
        finally:
            operation.release()

    def _close_native_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        self._native.close()

    def __enter__(self) -> EmbeddedIdScan:
        if self._closed:
            raise RuntimeError("ID scan is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


class EmbeddedReadStream(Iterator[np.ndarray]):
    """Bounded query-order reads from one captured native collection view.

    The ID source is retained rather than normalized eagerly. Its length,
    shape, dtype, strides and data pointer (when applicable) must remain
    unchanged until the stream is closed, and callers must not mutate its ID
    values. Valid integer dtypes are normalized one bounded slice at a time;
    an aligned native ``uint64`` ndarray or memmap slice stays zero-copy.

    Successfully returned batches remain valid if a later batch fails. A
    failing batch is never yielded, the logical position is not advanced, and
    the stream becomes closed. Native missing-ID detection leaves that batch's
    output untouched; other storage errors may have modified caller-provided
    memory before they are reported.
    """

    def __init__(
        self,
        native: NativeReadView,
        *,
        source: _StreamingNumericIdSource,
        batch_rows: int,
        prefix_dimensions: int,
        dtype: str,
    ) -> None:
        if int(native.output_dimension) != prefix_dimensions:
            native.close()
            raise RuntimeError("native read view returned the wrong output dimension")
        if str(native.dtype) != dtype:
            native.close()
            raise RuntimeError("native read view returned the wrong embedding dtype")
        metadata = native.metadata
        self._native = native
        self._source = source
        self._batch_rows = batch_rows
        self._prefix_dimensions = prefix_dimensions
        self._dtype = dtype
        self._snapshot = _id_scan_snapshot(metadata)
        self._snapshot_bytes = _scan_snapshot_bytes(metadata)
        self._position = 0
        self._closed = False
        self._failed = False
        self._operation = _SingleConsumerOperation("embedding read stream")

    @property
    def snapshot(self) -> IdScanSnapshot:
        return self._snapshot

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    @property
    def requested_count(self) -> int:
        return len(self._source)

    @property
    def emitted_count(self) -> int:
        return self._position

    @property
    def batch_capacity(self) -> int:
        return self._batch_rows

    @property
    def done(self) -> bool:
        return self._position == len(self._source)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def read_into(self, output: object) -> int:
        operation = self._operation.acquire()
        try:
            return self._read_into_locked(output)
        finally:
            operation.release()

    def _read_into_locked(self, output: object) -> int:
        if self._closed:
            raise RuntimeError("embedding read stream is closed")
        validated = _validate_stream_output(
            output,
            expected_columns=self._prefix_dimensions,
            dtype=self._dtype,
        )
        remaining = len(self._source) - self._position
        if remaining == 0:
            return 0
        count = min(remaining, self._batch_rows, int(validated.shape[0]))
        begin = self._position
        end = begin + count
        output_slice = validated[:count]
        if self._source.output_overlaps_unread(
            output_slice,
            begin=begin,
        ):
            raise ValueError(
                "stream_get output must not overlap the current or future "
                "ID source"
            )
        try:
            ids = self._source.batch(begin, end)
            self._native.get_into(ids, output_slice)
        except BaseException:
            self._failed = True
            try:
                self._close_native_locked()
            except BaseException:
                pass
            raise
        self._position = end
        return count

    def __iter__(self) -> EmbeddedReadStream:
        return self

    def __next__(self) -> np.ndarray:
        operation = self._operation.acquire()
        try:
            if self._closed or self._position == len(self._source):
                if not self._closed:
                    self._close_native_locked()
                raise StopIteration
            count = min(self._batch_rows, len(self._source) - self._position)
            output = np.empty(
                (count, self._prefix_dimensions),
                dtype=np.dtype(self._dtype),
            )
            produced = self._read_into_locked(output)
            if produced == 0:
                raise RuntimeError("embedding read stream made no progress")
            return output[:produced]
        finally:
            operation.release()

    def close(self) -> None:
        operation = self._operation.acquire()
        try:
            self._close_native_locked()
        finally:
            operation.release()

    def _close_native_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._native.close()

    def __enter__(self) -> EmbeddedReadStream:
        if self._closed:
            raise RuntimeError("embedding read stream is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


class EmbeddedRecordScan(Iterator[RecordBatch]):
    """Snapshot cursor over bit-exact IDs and embeddings in physical order."""

    def __init__(
        self,
        native: NativeRecordScan,
        *,
        batch_rows: int,
        prefix_dimensions: int,
        dtype: str,
        with_locations: bool,
    ) -> None:
        if int(native.output_dimension) != prefix_dimensions:
            native.close()
            raise RuntimeError("native record scan returned the wrong output dimension")
        if str(native.dtype) != dtype:
            native.close()
            raise RuntimeError("native record scan returned the wrong embedding dtype")
        metadata = native.metadata
        self._native = native
        self._batch_rows = batch_rows
        self._prefix_dimensions = prefix_dimensions
        self._dtype = dtype
        self._with_locations = with_locations
        self._snapshot = _id_scan_snapshot(metadata)
        self._snapshot_bytes = _scan_snapshot_bytes(metadata)
        self._emitted = 0
        self._closed = False
        self._failed = False
        self._finished = self._snapshot.count == 0 or bool(native.done)
        self._operation = _SingleConsumerOperation("record scan")

    @property
    def snapshot(self) -> IdScanSnapshot:
        return self._snapshot

    @property
    def snapshot_count(self) -> int:
        return self._snapshot.count

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    @property
    def emitted_count(self) -> int:
        return self._emitted

    @property
    def batch_capacity(self) -> int:
        return self._batch_rows

    @property
    def done(self) -> bool:
        return self._finished

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def read_into(
        self,
        ids: object,
        embeddings: object,
        locations: object | None = None,
    ) -> int:
        operation = self._operation.acquire()
        try:
            return self._read_into_locked(ids, embeddings, locations)
        finally:
            operation.release()

    def _read_into_locked(
        self,
        ids: object,
        embeddings: object,
        locations: object | None,
    ) -> int:
        if self._closed:
            raise RuntimeError("record scan is closed")
        output_ids, output_locations = _validate_id_scan_output(
            ids,
            locations,
            with_locations=self._with_locations,
            operation="scan_records",
        )
        output_embeddings = _validate_stream_output(
            embeddings,
            expected_columns=self._prefix_dimensions,
            dtype=self._dtype,
        )
        if output_embeddings.shape[0] != output_ids.shape[0]:
            raise ValueError(
                "embeddings output must have shape "
                f"({output_ids.size}, {self._prefix_dimensions})"
            )
        named_outputs = [
            ("ids", output_ids),
            ("embeddings", output_embeddings),
        ]
        if output_locations is not None:
            named_outputs.append(("locations", output_locations))
        _require_disjoint_numpy_arrays(named_outputs)
        capacity = min(self._batch_rows, int(output_ids.size))
        id_slice = output_ids[:capacity]
        embedding_slice = output_embeddings[:capacity]
        location_slice = (
            output_locations[:capacity]
            if output_locations is not None
            else None
        )
        if self._finished:
            return 0
        try:
            count = int(
                self._native.read_into(
                    id_slice,
                    embedding_slice,
                    location_slice,
                )
            )
            if not 0 <= count <= capacity:
                raise RuntimeError(
                    "native record scan returned an invalid row count"
                )
            if count == 0 and not self._native.done:
                raise RuntimeError(
                    "native record scan stopped without reaching EOF"
                )
            if self._emitted + count > self._snapshot.count:
                raise RuntimeError(
                    "native record scan exceeded its snapshot count"
                )
            self._emitted += count
            if self._native.done:
                if self._emitted != self._snapshot.count:
                    raise RuntimeError(
                        "native record scan ended before its snapshot count"
                    )
                self._finished = True
            return count
        except BaseException:
            self._failed = True
            try:
                self._close_native_locked()
            except BaseException:
                pass
            raise

    def __iter__(self) -> EmbeddedRecordScan:
        return self

    def __next__(self) -> RecordBatch:
        operation = self._operation.acquire()
        try:
            if self._closed or self._finished:
                raise StopIteration
            ids = np.empty(self._batch_rows, dtype=np.uint64)
            embeddings = np.empty(
                (self._batch_rows, self._prefix_dimensions),
                dtype=np.dtype(self._dtype),
            )
            locations = (
                np.empty(self._batch_rows, dtype=ID_LOCATION_DTYPE)
                if self._with_locations
                else None
            )
            emitted_offset = self._emitted
            count = self._read_into_locked(ids, embeddings, locations)
            if count == 0:
                raise StopIteration
            return RecordBatch(
                ids=ids[:count],
                embeddings=embeddings[:count],
                locations=(
                    locations[:count] if locations is not None else None
                ),
                snapshot=self._snapshot,
                emitted_offset=emitted_offset,
            )
        finally:
            operation.release()

    def close(self) -> None:
        operation = self._operation.acquire()
        try:
            self._close_native_locked()
        finally:
            operation.release()

    def _close_native_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._native.close()

    def __enter__(self) -> EmbeddedRecordScan:
        if self._closed:
            raise RuntimeError("record scan is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


class Collection:
    """User-facing collection wrapper with safe ID normalization."""

    def __init__(
        self,
        *,
        schema: CollectionSchema,
        builder: NativeCollectionBuilder | None = None,
        native: NativeCollection | None = None,
        access_mode: str | None = None,
        tuning_profile: Mapping[str, object] | None = None,
    ) -> None:
        if (builder is None) == (native is None):
            raise RuntimeError("collection must wrap exactly one native handle")
        if builder is not None and access_mode is not None:
            raise RuntimeError("a collection builder cannot have an open mode")
        if native is not None and access_mode not in {"r", "r+"}:
            raise RuntimeError("an opened collection must have mode 'r' or 'r+'")
        self._schema = schema
        self._builder = builder
        self._native = native
        self._access_mode = access_mode
        self._state = "building" if builder is not None else "open"
        if builder is not None:
            self._count = builder.count
            self._format_version = builder.format_version
        else:
            assert native is not None
            self._count = 0
            self._format_version = native.format_version
        self._cleaning_cursor = 0
        self._tuning = dict(tuning_profile) if tuning_profile is not None else None

    @classmethod
    def create(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        schema: CollectionSchema | None = None,
        segment_size_bytes: int = _DEFAULT_SEGMENT_SIZE_BYTES,
        format_version: int = 7,
    ) -> Collection:
        if schema is None:
            schema = CollectionSchema()
        if not isinstance(schema, CollectionSchema):
            raise TypeError("schema must be a CollectionSchema")
        if schema.id_type != "uint64":
            raise NotImplementedError(
                "utf8 IDs are not implemented by the numeric collection format"
            )
        validated_segment_size = _validate_segment_size_bytes(segment_size_bytes)
        validated_format_version = _validate_format_version(format_version)
        builder = NativeCollectionBuilder.create(
            path,
            dimension=schema.dimension,
            segment_size_bytes=validated_segment_size,
            dtype=schema.dtype,
            format_version=validated_format_version,
        )
        return cls(schema=schema, builder=builder)

    @classmethod
    def open(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        mode: str = "r",
        tuning: str | Mapping[str, object] = "auto",
        verify_index: bool = False,
        max_wal_backlog_bytes: int = _DEFAULT_MAX_WAL_BACKLOG_BYTES,
        max_wal_backlog_operations: int = _DEFAULT_MAX_WAL_BACKLOG_OPERATIONS,
        max_overlay_entries: int = _DEFAULT_MAX_OVERLAY_ENTRIES,
        max_overlay_memory_bytes: int = _DEFAULT_MAX_OVERLAY_MEMORY_BYTES,
    ) -> Collection:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in {"r", "r+"}:
            raise ValueError("mode must be 'r' or 'r+'")
        if not isinstance(verify_index, bool):
            raise TypeError("verify_index must be a bool")
        open_bounds = {
            "max_wal_bytes": _validate_resource_limit(
                max_wal_backlog_bytes, name="max_wal_backlog_bytes"
            ),
            "max_wal_operations": _validate_resource_limit(
                max_wal_backlog_operations,
                name="max_wal_backlog_operations",
            ),
            "max_overlay_entries": _validate_resource_limit(
                max_overlay_entries, name="max_overlay_entries"
            ),
            "max_overlay_bytes": _validate_resource_limit(
                max_overlay_memory_bytes, name="max_overlay_memory_bytes"
            ),
        }
        options, profile = _resolve_tuning(path, tuning)
        try:
            native = NativeCollection.open(
                path,
                mode=mode,
                verify_native_index=verify_index,
                **open_bounds,
                **options,
            )
        except ReaderBackendError as error:
            if (
                not isinstance(tuning, str)
                or tuning != "auto"
                or profile.get("source") != "persisted"
            ):
                raise
            compatibility = profile.get("compatibility")
            profile_key = profile.get("profile_key")
            profile = _fallback_tuning_profile(
                mode="auto",
                status="backend_initialization_failed",
                reason=(
                    "the automatically selected reader backend could not be "
                    f"initialized; using conservative defaults: {error}"
                ),
                compatibility=(
                    compatibility
                    if isinstance(compatibility, Mapping)
                    else None
                ),
            )
            if isinstance(profile_key, Mapping):
                profile["failed_profile_key"] = copy.deepcopy(
                    dict(profile_key)
                )
            native = NativeCollection.open(
                path,
                mode=mode,
                verify_native_index=verify_index,
                **open_bounds,
            )
        schema = CollectionSchema(
            id_type="uint64",
            dimension=native.dimension,
            dtype=native.dtype,
        )
        return cls(
            schema=schema,
            native=native,
            access_mode=mode,
            tuning_profile=profile,
        )

    @classmethod
    def inspect_orphans(
        cls,
        path: str | PathLike[str],
        *,
        min_age_seconds: float = _DEFAULT_ORPHAN_MIN_AGE_SECONDS,
    ) -> OrphanCollectionReport:
        """Build a read-only orphan plan without loading the base index.

        The lightweight native scan briefly acquires the collection writer
        lease so it cannot mistake an in-progress lifecycle output for an
        orphan. No files are changed or removed.
        """

        del cls
        validated_min_age, _, _ = _validate_orphan_options(
            dry_run=True,
            min_age_seconds=min_age_seconds,
        )
        return _orphan_report(
            NativeCollection.inspect_orphans(path),
            min_age_seconds=validated_min_age,
        )

    @classmethod
    def _reclaim_orphans_at(
        cls,
        path: str | PathLike[str],
        *,
        min_age_seconds: float = _DEFAULT_ORPHAN_MIN_AGE_SECONDS,
        max_files: int = _DEFAULT_ORPHAN_MAX_FILES,
        max_bytes: int = _DEFAULT_ORPHAN_MAX_BYTES,
    ) -> OrphanCollectionExecutionReport:
        """Private lightweight path used by the CLI execute mode."""

        del cls
        validated_age, validated_files, validated_bytes = (
            _validate_orphan_options(
                dry_run=False,
                min_age_seconds=min_age_seconds,
                max_files=max_files,
                max_bytes=max_bytes,
            )
        )
        raw = NativeCollection._reclaim_orphans_at(
            path,
            snapshot_unix_ns=0,
            min_age_ns=_native_orphan_min_age_ns(validated_age),
            max_files=validated_files,
            max_bytes=validated_bytes,
        )
        return _orphan_execution_report(
            raw,
            min_age_seconds=validated_age,
            max_files=validated_files,
            max_bytes=validated_bytes,
        )

    @classmethod
    def tune(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        max_seconds: float = 300.0,
        max_read_bytes: int = 8 << 30,
        max_memory_bytes: int = 512 << 20,
        seed: int = 20_260_713,
        dry_run: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        """Calibrate and optionally persist a reader profile for this target.

        The command keeps one stable read-only native snapshot in the parent
        and runs the complete matrix in a fresh worker process, so application
        I/O cannot contaminate process-wide storage counters.  The worker
        samples live IDs from one loaded base index and never reloads it for
        individual candidates. ``max_seconds`` is a cooperative deadline: one
        already-started OS read may finish after it expires.
        Each measurement stage uses file-scoped ``POSIX_FADV_DONTNEED`` when
        available and automatic persistence additionally requires observed
        backing-storage reads, so a warm page-cache run cannot publish a disk
        profile.
        """

        from ._calibration import validate_limits

        max_seconds, max_read_bytes, max_memory_bytes, seed = validate_limits(
            max_seconds=max_seconds,
            max_read_bytes=max_read_bytes,
            max_memory_bytes=max_memory_bytes,
            seed=seed,
        )
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")

        started = time.perf_counter()
        deadline = started + max_seconds

        def budget_measurement(reason: str) -> dict[str, object]:
            return {
                "format": "throwdb-reader-calibration",
                "version": 1,
                "status": "budget_exhausted",
                "reason": reason,
                "candidate_discovery": None,
                "sample": None,
                "workload": None,
                "candidates": [],
                "stability": None,
                "resource_stability": None,
                "storage_stability": None,
                "selection": None,
                "planned_read_bytes": 0,
                "measurement_process": None,
            }

        def report(
            measured: Mapping[str, object],
            *,
            status: str | None = None,
            key: RegistryKey | None = None,
            probe: StorageProbeResult | None = None,
            device: Mapping[str, object] | None = None,
            runtime: Mapping[str, object] | None = None,
            layout: Mapping[str, object] | None = None,
            capabilities: Mapping[str, object] | None = None,
            mutation_snapshot: Mapping[str, object] | None = None,
            target_stable: bool | None = None,
            generation: int | None = None,
            open_stats: Mapping[str, object] | None = None,
            selection: Mapping[str, object] | None = None,
        ) -> dict[str, object]:
            profile_reusable = (
                _storage_identity_is_reusable(probe)
                if probe is not None
                else None
            )
            profile_reuse_level = (
                _storage_match_level(probe) if probe is not None else None
            )
            if profile_reuse_level == "compatible":
                identity_warning = (
                    "the backing storage is opaque; automatic selection uses "
                    "a performance compatibility signature rather than an "
                    "exact media identity"
                )
            elif profile_reusable is False:
                identity_warning = (
                    "this storage identity is valid only for this measurement "
                    "and cannot be persisted or selected automatically"
                )
            else:
                identity_warning = None
            return {
                "format": measured["format"],
                "version": measured["version"],
                "status": status or str(measured["status"]),
                "reason": measured.get("reason"),
                "configuration": {
                    "seed": seed,
                    "dry_run": dry_run,
                    "max_seconds": max_seconds,
                    "max_read_bytes": max_read_bytes,
                    "max_memory_bytes": max_memory_bytes,
                    "deadline": "cooperative",
                    "matrix_version": 3,
                },
                "target": {
                    "profile_key": (
                        _profile_key_dict(key) if key is not None else None
                    ),
                    "device": (
                        probe.media.as_redacted_dict()
                        if probe is not None
                        else (dict(device) if device is not None else None)
                    ),
                    "runtime": (
                        probe.runtime.as_redacted_dict()
                        if probe is not None
                        else (dict(runtime) if runtime is not None else None)
                    ),
                    "layout": dict(layout) if layout is not None else None,
                    "reuse_scope": (
                        probe.media.reuse_scope if probe is not None else None
                    ),
                    "profile_reusable": profile_reusable,
                    "profile_reuse_level": profile_reuse_level,
                    "identity_warning": identity_warning,
                    "probed_path_count": (
                        probe.probed_path_count if probe is not None else None
                    ),
                    "used_root_fallback": (
                        probe.used_root_fallback if probe is not None else None
                    ),
                    "reader_capabilities": (
                        dict(capabilities)
                        if capabilities is not None
                        else None
                    ),
                    "mutation_snapshot": (
                        dict(mutation_snapshot)
                        if mutation_snapshot is not None
                        else None
                    ),
                    "stable_at_publication": target_stable,
                    "registry_generation_at_start": generation,
                    "open_stats": (
                        dict(open_stats) if open_stats is not None else None
                    ),
                },
                "budget": {
                    "elapsed_seconds": time.perf_counter() - started,
                    "limit_seconds": max_seconds,
                    "max_read_bytes": max_read_bytes,
                    "planned_read_bytes": int(
                        measured.get("planned_read_bytes", 0)
                    ),
                    "max_memory_bytes": max_memory_bytes,
                },
                "candidate_discovery": measured.get("candidate_discovery"),
                "sample": measured.get("sample"),
                "workload": measured.get("workload"),
                "candidates": measured.get("candidates", []),
                "stability": measured.get("stability"),
                "resource_stability": measured.get("resource_stability"),
                "storage_stability": measured.get("storage_stability"),
                "measurement_process": measured.get("measurement_process"),
                "selection": dict(selection) if selection is not None else None,
            }

        def expired() -> bool:
            return time.perf_counter() >= deadline

        root_initial = _collection_root_identity(path)
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired while identifying the collection"
                )
            )

        loaded = load_registry(path, strict=False)
        if loaded.status == "corrupt":
            raise TuningRegistryError(
                f"cannot calibrate with a corrupt tuning registry: {loaded.error}"
            )
        original_registry = loaded.registry
        salt = (
            original_registry.salt
            if original_registry is not None
            else secrets.token_hex(16)
        )
        generation = (
            original_registry.generation
            if original_registry is not None
            else 0
        )
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired while loading tuning metadata"
                ),
                generation=generation,
            )

        layout_before_open, record_size_before_open = _layout_identity(path)
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired while inspecting collection layout"
                ),
                layout=layout_before_open,
                generation=generation,
            )
        capabilities_before_open = _reader_capabilities()
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired during reader capability probe"
                ),
                layout=layout_before_open,
                capabilities=capabilities_before_open,
                generation=generation,
            )
        (
            probe_before_open,
            device_before_open,
            runtime_before_open,
            capabilities_before_open,
        ) = _probe_storage_identity(
            path,
            salt,
            capabilities_before_open,
        )
        if not _storage_identity_is_reusable(probe_before_open) and not dry_run:
            raise TuningRegistryError(
                _non_reusable_storage_reason(probe_before_open)
            )
        key_before_open = RegistryKey.from_mappings(
            device_before_open,
            runtime_before_open,
            layout_before_open,
            salt=salt,
        )
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired while probing storage identity"
                ),
                key=key_before_open,
                probe=probe_before_open,
                device=device_before_open,
                runtime=runtime_before_open,
                layout=layout_before_open,
                capabilities=capabilities_before_open,
                generation=generation,
            )

        root_before_open = _collection_root_identity(path)
        if root_initial != root_before_open:
            raise TuningRegistryError(
                "collection root changed during calibration preflight"
            )
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired before opening the collection"
                ),
                key=key_before_open,
                probe=probe_before_open,
                device=device_before_open,
                runtime=runtime_before_open,
                layout=layout_before_open,
                capabilities=capabilities_before_open,
                generation=generation,
            )

        if progress is not None:
            progress("opening one stable calibration snapshot")
        if expired():
            return report(
                budget_measurement(
                    "cooperative deadline expired before opening the collection"
                ),
                key=key_before_open,
                probe=probe_before_open,
                device=device_before_open,
                runtime=runtime_before_open,
                layout=layout_before_open,
                capabilities=capabilities_before_open,
                generation=generation,
            )
        native = NativeCollection.open(path, mode="r", calibration=True)
        try:
            open_stats = dict(native.open_stats)
            mutation_before = dict(native.mutation_log_stats)
            common = {
                "key": key_before_open,
                "probe": probe_before_open,
                "device": device_before_open,
                "runtime": runtime_before_open,
                "layout": layout_before_open,
                "capabilities": capabilities_before_open,
                "mutation_snapshot": mutation_before,
                "generation": generation,
                "open_stats": open_stats,
            }
            if expired():
                return report(
                    budget_measurement(
                        "cooperative deadline expired while opening the collection"
                    ),
                    **common,
                )

            root_at_start = _collection_root_identity(path)
            if root_before_open != root_at_start:
                raise TuningRegistryError(
                    "collection root changed while acquiring the calibration lease"
                )
            if expired():
                return report(
                    budget_measurement(
                        "cooperative deadline expired while anchoring the snapshot"
                    ),
                    **common,
                )

            layout, record_size = _layout_identity(path)
            if (
                layout != layout_before_open
                or record_size != record_size_before_open
                or int(layout["dimension"]) != int(native.dimension)
                or str(layout["dtype"]) != str(native.dtype)
                or int(layout["format_version"]) != int(native.format_version)
            ):
                raise TuningRegistryError(
                    "collection layout changed while opening the calibration snapshot"
                )
            if expired():
                return report(
                    budget_measurement(
                        "cooperative deadline expired while verifying collection layout"
                    ),
                    **common,
                )

            capabilities = _reader_capabilities()
            if expired():
                return report(
                    budget_measurement(
                        "cooperative deadline expired during reader capability probe"
                    ),
                    **common,
                )
            probe, device, runtime, capabilities = _probe_storage_identity(
                path,
                salt,
                capabilities,
            )
            if (
                probe.media.reuse_scope != probe_before_open.media.reuse_scope
                or device != device_before_open
                or runtime != runtime_before_open
            ):
                raise TuningRegistryError(
                    "storage identity changed while acquiring the calibration lease"
                )
            key = RegistryKey.from_mappings(
                device,
                runtime,
                layout,
                salt=salt,
            )
            common = {
                "key": key,
                "probe": probe,
                "device": device,
                "runtime": runtime,
                "layout": layout,
                "capabilities": capabilities,
                "mutation_snapshot": mutation_before,
                "generation": generation,
                "open_stats": open_stats,
            }
            if expired():
                return report(
                    budget_measurement(
                        "cooperative deadline expired while verifying storage identity"
                    ),
                    **common,
                )

            existing = (
                original_registry.select(key)
                if original_registry is not None
                else None
            )
            measured = _execute_reader_calibration(
                native,
                path=path,
                record_size=record_size,
                current_options=(
                    dict(existing.options) if existing is not None else None
                ),
                seed=seed,
                max_read_bytes=max_read_bytes,
                max_memory_bytes=max_memory_bytes,
                deadline=deadline,
                expected_root_identity=(
                    root_at_start[3],
                    root_at_start[4],
                ),
                progress=progress,
            )

            selection_value = measured.get("selection")
            selection = (
                dict(selection_value)
                if isinstance(selection_value, Mapping)
                else None
            )
            selection_safe = bool(
                selection is not None
                and selection.get("selection_valid") is True
            )
            stability_value = measured.get("stability")
            resource_stability_value = measured.get("resource_stability")
            matrix_safe = bool(
                isinstance(stability_value, Mapping)
                and stability_value.get("passed") is True
                and isinstance(resource_stability_value, Mapping)
                and resource_stability_value.get("passed") is True
            )
            status = str(measured["status"])
            persisted = False
            saved_profile: dict[str, object] | None = None
            target_stable: bool | None = None

            storage_stability_value = measured.get("storage_stability")
            storage_stability = (
                dict(storage_stability_value)
                if isinstance(storage_stability_value, Mapping)
                else {
                    "classification": "unknown",
                    "passed": False,
                    "persistence_safe": False,
                    "reason": "storage backing-read evidence is missing",
                }
            )
            workload_value = measured.get("workload")
            stage_preparation_value = (
                workload_value.get("stage_cache_preparation")
                if isinstance(workload_value, Mapping)
                else None
            )
            preparation_shape_valid = bool(
                isinstance(stage_preparation_value, Mapping)
                and all(
                    isinstance(value, Mapping)
                    for value in stage_preparation_value.values()
                )
            )
            stage_preparations = (
                {
                    str(name): dict(value)
                    for name, value in stage_preparation_value.items()
                    if isinstance(value, Mapping)
                }
                if isinstance(stage_preparation_value, Mapping)
                else {}
            )
            required_preparation_stages = {"screening", "verification"}
            discovery_value = measured.get("candidate_discovery")
            if isinstance(discovery_value, Mapping):
                backfill_value = discovery_value.get("verification_backfill")
                if (
                    isinstance(backfill_value, Mapping)
                    and backfill_value.get("attempted") is True
                ):
                    required_preparation_stages.add("verification_backfill")
                recheck_value = discovery_value.get("stability_recheck")
                if (
                    isinstance(recheck_value, Mapping)
                    and recheck_value.get("attempted") is True
                ):
                    required_preparation_stages.add("verification_recheck")
            prepared_stages = sorted(stage_preparations)
            preparation_statuses = {
                name: str(value.get("status", "unknown"))
                for name, value in stage_preparations.items()
            }
            preparation_safe = bool(
                preparation_shape_valid
                and required_preparation_stages.issubset(stage_preparations)
                and all(
                    status == "performed"
                    for status in preparation_statuses.values()
                )
            )
            preparation_reason = (
                "every measured stage completed file-scoped cache preparation"
                if preparation_safe
                else (
                    "file-scoped cache preparation was not completed for "
                    "every measured stage"
                )
            )
            measurement_process_value = measured.get("measurement_process")
            measurement_process = (
                dict(measurement_process_value)
                if isinstance(measurement_process_value, Mapping)
                else {}
            )
            worker_pid = measurement_process.get("pid")
            isolation_safe = bool(
                measurement_process.get("isolation") == "subprocess_exec"
                and measurement_process.get("status") == "complete"
                and measurement_process.get("parent_verified") is True
                and not isinstance(worker_pid, bool)
                and isinstance(worker_pid, int)
                and worker_pid > 0
                and worker_pid != os.getpid()
                and storage_stability.get("telemetry_scope")
                == "isolated_calibration_process"
            )
            process_isolation_value = storage_stability.get(
                "process_isolation"
            )
            process_isolation = (
                dict(process_isolation_value)
                if isinstance(process_isolation_value, Mapping)
                else {}
            )
            isolation_safe = bool(
                isolation_safe
                and process_isolation.get("passed") is True
                and process_isolation.get("method") == "subprocess_exec"
                and process_isolation.get("parent_verified") is True
                and process_isolation.get("pid") == worker_pid
            )
            process_isolation.update(
                {
                    "passed": isolation_safe,
                    "required": "subprocess_exec",
                    "measurement_process": measurement_process,
                }
            )
            storage_stability["process_isolation"] = process_isolation
            storage_stability["cache_preparation"] = {
                "passed": preparation_safe,
                "reason": preparation_reason,
                "well_formed": preparation_shape_valid,
                "required_stages": sorted(required_preparation_stages),
                "observed_stages": prepared_stages,
                "statuses": preparation_statuses,
            }
            backing_read_safe = bool(
                storage_stability.get("passed") is True
                and storage_stability.get("persistence_safe") is True
                and storage_stability.get("classification")
                == "backing_storage"
            )
            storage_stability["backing_read_passed"] = backing_read_safe
            storage_stability["passed"] = bool(
                backing_read_safe and preparation_safe and isolation_safe
            )
            storage_stability["persistence_safe"] = bool(
                backing_read_safe and preparation_safe and isolation_safe
            )
            if backing_read_safe and not preparation_safe:
                storage_stability["reason"] = preparation_reason
            elif backing_read_safe and preparation_safe and not isolation_safe:
                storage_stability["reason"] = (
                    "backing-read telemetry was not collected in the "
                    "dedicated calibration process"
                )
            measured["storage_stability"] = storage_stability
            storage_safe = bool(
                storage_stability.get("passed") is True
                and storage_stability.get("persistence_safe") is True
                and storage_stability.get("classification")
                == "backing_storage"
            )
            if status == "complete" and expired():
                status = "budget_exhausted"
                measured["reason"] = (
                    "cooperative deadline expired after the matrix completed"
                )

            if status == "complete":
                mutation_after = dict(native.mutation_log_stats)
                if mutation_before != mutation_after:
                    target_stable = False
                elif expired():
                    status = "budget_exhausted"
                    measured["reason"] = (
                        "cooperative deadline expired during final target "
                        "verification"
                    )
                else:
                    root_after = _collection_root_identity(path)
                    if root_at_start != root_after:
                        target_stable = False
                    elif expired():
                        status = "budget_exhausted"
                        measured["reason"] = (
                            "cooperative deadline expired during final target verification"
                        )
                    else:
                        layout_after, _ = _layout_identity(path)
                        if layout != layout_after:
                            target_stable = False
                        elif expired():
                            status = "budget_exhausted"
                            measured["reason"] = (
                                "cooperative deadline expired during final "
                                "layout verification"
                            )
                        else:
                            final_capabilities = _reader_capabilities()
                            if expired():
                                status = "budget_exhausted"
                                measured["reason"] = (
                                    "cooperative deadline expired during final "
                                    "reader capability probe"
                                )
                            else:
                                (
                                    reprobe,
                                    device_after,
                                    runtime_after,
                                    _,
                                ) = _probe_storage_identity(
                                    path,
                                    salt,
                                    final_capabilities,
                                )
                                if expired():
                                    status = "budget_exhausted"
                                    measured["reason"] = (
                                        "cooperative deadline expired during "
                                        "final storage probe"
                                    )
                                else:
                                    target_stable = (
                                        device == device_after
                                        and runtime == runtime_after
                                        and reprobe.media.reuse_scope
                                        == probe.media.reuse_scope
                                    )

            if status == "complete" and not target_stable:
                status = "target_changed"
                measured["reason"] = (
                    "device, runtime, layout, root, or mutation snapshot changed"
                )
            elif status == "complete" and not matrix_safe:
                status = "calibration_unverified"
                measured["reason"] = (
                    "automatic profile persistence requires passed baseline "
                    "and resource stability guardrails"
                )
            elif status == "complete" and not selection_safe:
                status = "selection_unverified"
                measured["reason"] = (
                    "automatic profile persistence requires an explicitly "
                    "valid calibration selection"
                )
            elif status == "complete" and not storage_safe:
                classification = (
                    str(storage_stability.get("classification"))
                )
                status = (
                    "cache_unrepresentative"
                    if classification in {"page_cache", "mixed"}
                    else "cache_unverified"
                )
                detail = (
                    str(storage_stability.get("reason"))
                )
                measured["reason"] = (
                    "automatic profile persistence requires a "
                    f"backing-storage calibration; {detail}"
                )
            elif status == "complete" and selection_safe:
                if dry_run:
                    status = "dry_run"
                elif not _storage_identity_is_reusable(probe):
                    raise TuningRegistryError(
                        _non_reusable_storage_reason(probe)
                    )
                elif expired():
                    status = "budget_exhausted"
                    measured["reason"] = (
                        "cooperative deadline expired before profile publication"
                    )
                else:
                    profile = upsert_profile(
                        path,
                        device=device,
                        runtime=runtime,
                        layout=layout,
                        options=selection["options"],
                        source="calibrated-v3",
                        record_size=record_size,
                        expected_salt=salt,
                        expected_generation=generation,
                        initial_salt=salt,
                        expected_store_identity=(
                            root_at_start[3],
                            root_at_start[4],
                        ),
                        fail_if_locked=True,
                    )
                    persisted = True
                    saved_profile = _stored_profile_dict(profile)
                    status = "saved"

            if selection is not None:
                selection["persisted"] = persisted
                selection["profile"] = saved_profile
            return report(
                measured,
                status=status,
                target_stable=target_stable,
                selection=selection,
                **common,
            )
        finally:
            native.close()

    @classmethod
    def save_tuning_profile(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        tuning: Mapping[str, object],
    ) -> dict[str, object]:
        """Persist explicit options for the current media/runtime/layout.

        This installs already selected values; it does not run a benchmark.
        Low-confidence media identities are rejected so an unsafe profile can
        never become eligible for automatic selection.
        """

        if not isinstance(tuning, Mapping):
            raise TypeError("tuning must be a mapping of reader options")
        layout, record_size = _layout_identity(path)
        options = normalize_reader_options(tuning, record_size=record_size)
        if options["io_backend"] == "io_uring":
            requested_capabilities = _reader_capabilities(
                int(options["io_uring_queue_depth"]),
                int(options["io_uring_workers"]),
            )
            if not requested_capabilities["io_uring_available"]:
                raise TuningRegistryError(
                    "the requested io_uring topology is unavailable in the "
                    "current runtime: "
                    f"{requested_capabilities['io_uring_reason']}"
                )

        loaded = load_registry(path, strict=False)
        if loaded.status == "corrupt":
            raise TuningRegistryError(
                f"cannot update a corrupt tuning registry: {loaded.error}"
            )
        registry = loaded.registry
        if registry is None:
            # Classify first without creating optional metadata. This keeps a
            # failed attempt on unknown media completely read-only.
            temporary_probe = probe_collection_storage(
                path,
                collection_salt=secrets.token_bytes(16),
                reader_abi=_READER_POLICY_ABI,
            )
            if not _storage_identity_is_reusable(temporary_probe):
                raise TuningRegistryError(
                    "the current storage target cannot be identified safely; "
                    "refusing to persist an automatically reusable profile"
                )
            registry = create_registry(path)

        probe, device, runtime, _ = _probe_identity(path, registry.salt)
        if not _storage_identity_is_reusable(probe):
            raise TuningRegistryError(
                "the current storage target cannot be identified safely; "
                "refusing to persist an automatically reusable profile"
            )
        if options["io_backend"] == "io_uring":
            requested_capabilities = _reader_capabilities(
                int(options["io_uring_queue_depth"]),
                int(options["io_uring_workers"]),
            )
            if not requested_capabilities["io_uring_available"]:
                raise TuningRegistryError(
                    "the requested io_uring topology became unavailable "
                    "before the profile was saved: "
                    f"{requested_capabilities['io_uring_reason']}"
                )
        profile = upsert_profile(
            path,
            device=device,
            runtime=runtime,
            layout=layout,
            options=options,
            source="manual",
            record_size=record_size,
            expected_salt=registry.salt,
        )
        return {
            "status": "saved",
            "profile": _stored_profile_dict(profile),
        }

    @classmethod
    def tuning_profiles(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
    ) -> dict[str, object]:
        """Inspect saved profiles and their match to the current target."""

        layout, _ = _layout_identity(path)
        loaded = load_registry(path, strict=False)
        if loaded.registry is None:
            return {
                "status": loaded.status,
                "error": loaded.error,
                "profile_count": 0,
                "profiles": [],
                "current": {
                    "layout": layout,
                    "match_level": "fallback",
                    "selected_profile_key": None,
                    "selection_status": f"registry_{loaded.status}",
                },
            }

        registry = loaded.registry
        summaries = [_stored_profile_dict(profile) for profile in registry.profiles]
        try:
            probe, device, runtime, capabilities = _probe_identity(
                path, registry.salt
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return {
                "status": "probe_failed",
                "error": str(error),
                "registry_generation": registry.generation,
                "profile_count": len(summaries),
                "profiles": summaries,
                "current": {
                    "layout": layout,
                    "match_level": "fallback",
                    "selected_profile_key": None,
                    "selection_status": "probe_failed",
                },
            }

        key = RegistryKey.from_mappings(
            device,
            runtime,
            layout,
            salt=registry.salt,
        )
        reusable = _storage_identity_is_reusable(probe)
        match_level = _storage_match_level(probe)
        selected_profile_key: dict[str, str] | None = None
        for summary, profile in zip(summaries, registry.profiles, strict=True):
            current_match = reusable and profile.key == key
            summary["current_match"] = current_match
            summary["current_match_level"] = (
                match_level if current_match else "none"
            )
            if current_match:
                selected_profile_key = _profile_key_dict(profile.key)
        selection_status = (
            "selected"
            if selected_profile_key is not None
            else ("no_match" if reusable else "device_unrecognized")
        )
        return {
            "status": "ok" if reusable else "device_unrecognized",
            "error": None,
            "registry_generation": registry.generation,
            "profile_count": len(summaries),
            "profiles": summaries,
            "current": {
                "key": _profile_key_dict(key),
                "device": probe.media.as_redacted_dict(),
                "runtime": probe.runtime.as_redacted_dict(),
                "layout": layout,
                "reusable": reusable,
                "reuse_scope": probe.media.reuse_scope,
                "match_level": (
                    match_level
                    if selected_profile_key is not None
                    else "fallback"
                ),
                "selected_profile_key": selected_profile_key,
                "selection_status": selection_status,
                "probed_path_count": probe.probed_path_count,
                "used_root_fallback": probe.used_root_fallback,
                "reader_capabilities": capabilities,
            },
        }

    @classmethod
    def remove_tuning_profile(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
    ) -> dict[str, object]:
        """Remove the profile matching the current media/runtime/layout."""

        layout, _ = _layout_identity(path)
        loaded = load_registry(path, strict=False)
        if loaded.status == "missing":
            return {"status": "missing", "removed": False}
        if loaded.registry is None:
            raise TuningRegistryError(
                f"cannot update the tuning registry: {loaded.error}"
            )
        registry = loaded.registry
        probe, device, runtime, _ = _probe_identity(path, registry.salt)
        if not _storage_identity_is_reusable(probe):
            return {"status": "device_unrecognized", "removed": False}
        key = RegistryKey.from_mappings(
            device,
            runtime,
            layout,
            salt=registry.salt,
        )
        removed = remove_profile(path, key, strict=True)
        return {
            "status": "removed" if removed else "no_match",
            "removed": removed,
            "key": _profile_key_dict(key),
        }

    @classmethod
    def reset_tuning_profiles(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        force: bool = False,
    ) -> dict[str, object]:
        """Remove every profile, optionally recovering corrupt metadata."""

        if not isinstance(force, bool):
            raise TypeError("force must be a bool")
        # Validate that the target is a collection before removing only its
        # optional tuning metadata.
        _layout_identity(path)
        removed = reset_registry(path, force=force)
        return {
            "status": "reset" if removed else "missing",
            "removed": removed,
        }

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    @property
    def id_type(self) -> str:
        return self._schema.id_type

    @property
    def dimension(self) -> int:
        return self._schema.dimension

    @property
    def dtype(self) -> str:
        return self._schema.dtype

    @property
    def count(self) -> int:
        if self._native is not None:
            return self._ensure_native_open("count").count
        return self._count

    @property
    def format_version(self) -> int:
        return self._format_version

    @property
    def mode(self) -> str:
        if self._builder is not None:
            return "write"
        return "read_write" if self._access_mode == "r+" else "read"

    @property
    def tuning(self) -> dict[str, object] | None:
        """Return the selected reader profile, or ``None`` for a builder."""

        return None if self._tuning is None else copy.deepcopy(self._tuning)

    @property
    def open_stats(self) -> dict[str, object] | None:
        """Return phase timings and index-validation details for this open."""

        if self._native is None:
            return None
        return dict(self._ensure_native_open("inspect open stats").open_stats)

    @property
    def mutation_log_stats(self) -> dict[str, object] | None:
        """Return a stable snapshot of the durable mutation-log lifecycle."""

        if self._native is None:
            return None
        stats = self._ensure_native_open(
            "inspect mutation log stats"
        ).mutation_log_stats
        if not isinstance(stats, dict):
            raise RuntimeError("native mutation-log stats are invalid")
        return dict(stats)

    def _ensure_builder_open(self) -> NativeCollectionBuilder:
        if self._builder is None:
            raise RuntimeError("add is only available on a collection builder")
        if self._state != "building":
            raise RuntimeError(f"collection builder is already {self._state}")
        return self._builder

    def _ensure_native_open(self, operation: str) -> NativeCollection:
        if self._native is None:
            raise RuntimeError("get is only available on an opened collection reader")
        if self._state != "open":
            raise RuntimeError(f"cannot {operation}: collection is closed")
        return self._native

    def _ensure_reader(self) -> NativeCollection:
        return self._ensure_native_open("read")

    def _ensure_mutable(self, operation: str) -> NativeCollection:
        native = self._ensure_native_open(operation)
        if self._access_mode != "r+":
            raise RuntimeError(
                f"{operation} requires Collection.open(..., mode='r+')"
            )
        return native

    def add(self, ids: object, embeddings: object) -> None:
        writer = self._ensure_builder_open()
        normalized_ids = _normalize_numeric_ids(ids)
        normalized_embeddings = _normalize_embeddings(
            embeddings,
            expected_rows=len(normalized_ids),
            schema=self._schema,
        )
        try:
            writer.add(normalized_ids, normalized_embeddings)
        except BaseException:
            # Once the core append starts, an operational failure may have
            # advanced staging files. Never let callers publish that builder.
            self.abort()
            raise
        self._count = writer.count

    def finish(self) -> None:
        if self._builder is None:
            raise RuntimeError("finish is only available on a collection builder")
        if self._state == "finished":
            return
        if self._state != "building":
            raise RuntimeError(f"collection builder is already {self._state}")
        try:
            self._builder.finish()
        except BaseException:
            self.abort()
            raise
        self._state = "finished"
        self._count = self._builder.count

    def abort(self) -> None:
        if self._builder is None:
            raise RuntimeError("abort is only available on a collection builder")
        if self._state in {"finished", "aborted"}:
            return
        self._builder.abort()
        self._state = "aborted"

    def close(self) -> None:
        """Close an opened collection or abort an unfinished builder."""

        if self._builder is not None:
            if self._state == "building":
                self.abort()
            return
        if self._state == "closed":
            return
        assert self._native is not None
        try:
            self._native.close()
        finally:
            self._state = "closed"

    def _put(self, operation: str, ids: object, embeddings: object) -> None:
        native = self._ensure_mutable(operation)
        normalized_ids = _normalize_numeric_ids(ids)
        normalized_embeddings = _normalize_embeddings(
            embeddings,
            expected_rows=len(normalized_ids),
            schema=self._schema,
        )
        if normalized_ids.size == 0:
            return
        getattr(native, operation)(normalized_ids, normalized_embeddings)

    def insert(self, ids: object, embeddings: object) -> None:
        """Insert a batch whose IDs must all be absent."""

        self._put("insert", ids, embeddings)

    def update(self, ids: object, embeddings: object) -> None:
        """Replace a batch whose IDs must all already exist."""

        self._put("update", ids, embeddings)

    def upsert(self, ids: object, embeddings: object) -> None:
        """Insert absent IDs and replace existing IDs atomically."""

        self._put("upsert", ids, embeddings)

    def delete(self, ids: object) -> np.ndarray:
        """Delete IDs and return one boolean existence result per input ID."""

        native = self._ensure_mutable("delete")
        normalized_ids = _normalize_numeric_ids(ids)
        if normalized_ids.size == 0:
            return np.empty(0, dtype=np.bool_)
        deleted = native.delete(normalized_ids)
        if not isinstance(deleted, np.ndarray):
            raise RuntimeError("native delete returned a non-NumPy result")
        if deleted.dtype != np.dtype(np.bool_) or deleted.shape != normalized_ids.shape:
            raise RuntimeError("native delete returned an invalid result array")
        return deleted

    def rotate_wal(self) -> dict[str, object]:
        """Seal a non-empty active WAL and atomically publish its successor.

        Rotation changes only durable mutation resources. It does not rewrite
        immutable base segments or change the logical contents of a collection.
        """

        native = self._ensure_mutable("rotate the mutation WAL")
        stats = native.rotate_wal()
        if not isinstance(stats, dict):
            raise RuntimeError("native WAL rotation returned invalid diagnostics")
        return dict(stats)

    def materialize_frozen_wals(
        self,
        *,
        max_wals: int | None = None,
        max_input_bytes: int | None = None,
        max_input_entries: int | None = None,
        max_memory_bytes: int | None = None,
    ) -> dict[str, object]:
        """Replace the frozen WAL prefix with segments and an immutable index run.

        The durable catalog is switched atomically. Referenced WAL files are
        reclaimed only after the new view is published.
        """

        native = self._ensure_mutable("materialize frozen mutation WALs")
        validated_max_wals = (
            _UINT64_MAX
            if max_wals is None
            else _validate_resource_limit(
                max_wals, name="max_wals"
            )
        )
        validated_max_input_bytes = (
            _UINT64_MAX
            if max_input_bytes is None
            else _validate_resource_limit(
                max_input_bytes, name="max_input_bytes"
            )
        )
        validated_max_input_entries = (
            _UINT64_MAX
            if max_input_entries is None
            else _validate_resource_limit(
                max_input_entries, name="max_input_entries"
            )
        )
        validated_max_memory_bytes = (
            _UINT64_MAX
            if max_memory_bytes is None
            else _validate_resource_limit(
                max_memory_bytes, name="max_memory_bytes"
            )
        )
        stats = native.materialize_frozen_wals(
            validated_max_wals,
            validated_max_input_bytes,
            validated_max_input_entries,
            validated_max_memory_bytes,
        )
        if not isinstance(stats, dict):
            raise RuntimeError(
                "native WAL materialization returned invalid diagnostics"
            )
        return dict(stats)

    def fold_partitions(
        self,
        *,
        max_runs: int | None = None,
        max_input_bytes: int = _DEFAULT_FOLD_MAX_INPUT_BYTES,
        max_input_entries: int = _DEFAULT_FOLD_MAX_INPUT_ENTRIES,
    ) -> dict[str, object]:
        """Fold bounded immutable index state into partition snapshots.

        Fold complete per-partition run sections into authoritative snapshots.
        ``max_runs`` bounds the active run files contributing to each partition;
        its existing snapshot is not counted. The operation never rewrites
        embedding segments.
        """

        native = self._ensure_mutable("fold index partitions")
        validated_max_runs = _validate_resource_limit(
            _MAX_PARTITIONED_FOLD_RUNS
            if max_runs is None
            else max_runs,
            name="max_runs",
            minimum=1,
            maximum=_MAX_PARTITIONED_FOLD_RUNS,
        )
        validated_max_input_bytes = _validate_resource_limit(
            max_input_bytes,
            name="max_input_bytes",
        )
        validated_max_input_entries = _validate_resource_limit(
            max_input_entries,
            name="max_input_entries",
        )
        stats = native.fold_partitions(
            validated_max_runs,
            validated_max_input_bytes,
            validated_max_input_entries,
        )
        if not isinstance(stats, dict):
            raise RuntimeError(
                "native partition folding returned invalid diagnostics"
            )
        return dict(stats)

    def collect_orphans(
        self,
        *,
        dry_run: bool = True,
        min_age_seconds: float = _DEFAULT_ORPHAN_MIN_AGE_SECONDS,
        max_files: int = _DEFAULT_ORPHAN_MAX_FILES,
        max_bytes: int = _DEFAULT_ORPHAN_MAX_BYTES,
    ) -> OrphanCollectionReport | OrphanCollectionExecutionReport:
        """Plan or execute bounded orphan storage-resource collection.

        Execute mode never consumes a previous plan. It performs a fresh scan,
        no-follow identity revalidation, unlink, and directory sync while this
        mutable handle retains the writer lease.
        """

        validated_min_age, validated_files, validated_bytes = (
            _validate_orphan_options(
                dry_run=dry_run,
                min_age_seconds=min_age_seconds,
                max_files=max_files,
                max_bytes=max_bytes,
            )
        )
        native = self._ensure_mutable("collect orphan mutation resources")
        if not dry_run:
            raw = native.reclaim_orphans(
                snapshot_unix_ns=0,
                min_age_ns=_native_orphan_min_age_ns(validated_min_age),
                max_files=validated_files,
                max_bytes=validated_bytes,
            )
            return _orphan_execution_report(
                raw,
                min_age_seconds=validated_min_age,
                max_files=validated_files,
                max_bytes=validated_bytes,
            )
        return _orphan_report(
            native.collect_orphans(),
            min_age_seconds=validated_min_age,
        )

    def maintain(
        self,
        *,
        dry_run: bool = True,
        max_seconds: float = _DEFAULT_MAINTAIN_MAX_SECONDS,
        max_actions: int = _DEFAULT_MAINTAIN_MAX_ACTIONS,
        rotate_wal_bytes: int = _DEFAULT_MAINTAIN_ROTATE_WAL_BYTES,
        materialize_max_wals: int = _DEFAULT_MAINTAIN_MATERIALIZE_MAX_WALS,
        materialize_max_input_bytes: int = (
            _DEFAULT_MAINTAIN_MATERIALIZE_MAX_INPUT_BYTES
        ),
        materialize_max_input_entries: int | None = None,
        materialize_max_memory_bytes: int | None = None,
        fold_min_runs: int | None = None,
        fold_max_runs: int | None = None,
        fold_max_input_bytes: int = (
            _DEFAULT_FOLD_MAX_INPUT_BYTES
        ),
        fold_max_input_entries: int = (
            _DEFAULT_FOLD_MAX_INPUT_ENTRIES
        ),
        clean_max_candidate_segments: int = (
            _DEFAULT_CLEAN_MAX_CANDIDATE_SEGMENTS
        ),
        clean_max_sidecar_scan_bytes: int = (
            _DEFAULT_CLEAN_MAX_SIDECAR_SCAN_BYTES
        ),
        clean_max_source_bytes: int = _DEFAULT_CLEAN_MAX_SOURCE_BYTES,
        clean_max_output_bytes: int = _DEFAULT_CLEAN_MAX_OUTPUT_BYTES,
        clean_max_memory_bytes: int = _DEFAULT_CLEAN_MAX_MEMORY_BYTES,
        clean_min_garbage_rows: int = _DEFAULT_CLEAN_MIN_GARBAGE_ROWS,
        clean_min_reclaimable_bytes: int = (
            _DEFAULT_CLEAN_MIN_RECLAIMABLE_BYTES
        ),
        clean_min_garbage_ratio_ppm: int = (
            _DEFAULT_CLEAN_MIN_GARBAGE_RATIO_PPM
        ),
        clean_start_after_segment_id: int | None = None,
        orphan_min_age_seconds: float = _DEFAULT_ORPHAN_MIN_AGE_SECONDS,
        orphan_max_files: int = _DEFAULT_ORPHAN_MAX_FILES,
        orphan_max_bytes: int = _DEFAULT_ORPHAN_MAX_BYTES,
    ) -> MaintenanceReport:
        """Run one deterministic, bounded lifecycle-maintenance pass.

        The wall-clock limit is cooperative: it is checked between durable
        actions, never by interrupting a publication in the middle.
        """

        native = self._ensure_mutable("maintain mutation resources")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool")
        if isinstance(max_seconds, (bool, np.bool_)) or not isinstance(
            max_seconds, (numbers.Real, np.integer, np.floating)
        ):
            raise TypeError("max_seconds must be a number, not bool")
        seconds = float(max_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("max_seconds must be finite and zero or greater")
        validated_max_actions = _validate_resource_limit(
            max_actions, name="max_actions", maximum=5
        )
        validated_rotate_bytes = _validate_resource_limit(
            rotate_wal_bytes, name="rotate_wal_bytes"
        )
        validated_materialize_wals = _validate_resource_limit(
            materialize_max_wals, name="materialize_max_wals"
        )
        validated_materialize_bytes = _validate_resource_limit(
            materialize_max_input_bytes,
            name="materialize_max_input_bytes",
        )
        validated_materialize_entries = (
            _DEFAULT_MAINTAIN_MATERIALIZE_MAX_INPUT_ENTRIES
            if materialize_max_input_entries is None
            else _validate_resource_limit(
                materialize_max_input_entries,
                name="materialize_max_input_entries",
            )
        )
        validated_materialize_memory = (
            _DEFAULT_MAINTAIN_MATERIALIZE_MAX_MEMORY_BYTES
            if materialize_max_memory_bytes is None
            else _validate_resource_limit(
                materialize_max_memory_bytes,
                name="materialize_max_memory_bytes",
            )
        )
        validated_fold_min_runs = _validate_resource_limit(
            1
            if fold_min_runs is None
            else fold_min_runs,
            name="fold_min_runs",
            minimum=1,
            maximum=_MAX_PARTITIONED_FOLD_RUNS,
        )
        validated_fold_max_runs = _validate_resource_limit(
            _MAX_PARTITIONED_FOLD_RUNS
            if fold_max_runs is None
            else fold_max_runs,
            name="fold_max_runs",
            minimum=1,
            maximum=_MAX_PARTITIONED_FOLD_RUNS,
        )
        if validated_fold_max_runs < validated_fold_min_runs:
            raise ValueError("fold_max_runs must be >= fold_min_runs")
        validated_fold_bytes = _validate_resource_limit(
            fold_max_input_bytes, name="fold_max_input_bytes"
        )
        validated_fold_entries = _validate_resource_limit(
            fold_max_input_entries, name="fold_max_input_entries"
        )
        clean_limits = {
            "clean_max_candidate_segments": clean_max_candidate_segments,
            "clean_max_sidecar_scan_bytes": clean_max_sidecar_scan_bytes,
            "clean_max_source_bytes": clean_max_source_bytes,
            "clean_max_output_bytes": clean_max_output_bytes,
            "clean_max_memory_bytes": clean_max_memory_bytes,
            "clean_min_garbage_rows": clean_min_garbage_rows,
        }
        validated_clean = {
            name: _validate_resource_limit(value, name=name)
            for name, value in clean_limits.items()
        }
        validated_clean_reclaimable = _validate_resource_limit(
            clean_min_reclaimable_bytes,
            name="clean_min_reclaimable_bytes",
            minimum=0,
        )
        validated_clean_ratio = _validate_resource_limit(
            clean_min_garbage_ratio_ppm,
            name="clean_min_garbage_ratio_ppm",
            minimum=0,
            maximum=1_000_000,
        )
        validated_clean_cursor = (
            self._cleaning_cursor
            if clean_start_after_segment_id is None
            else _validate_resource_limit(
                clean_start_after_segment_id,
                name="clean_start_after_segment_id",
                minimum=0,
                maximum=_ROW_REF_SEGMENT_ID_MAX,
            )
        )
        orphan_age, validated_orphan_files, validated_orphan_bytes = (
            _validate_orphan_options(
                dry_run=dry_run,
                min_age_seconds=orphan_min_age_seconds,
                max_files=orphan_max_files,
                max_bytes=orphan_max_bytes,
            )
        )
        policy: dict[str, object] = {
            "max_seconds": seconds,
            "max_actions": validated_max_actions,
            "rotate_wal_bytes": validated_rotate_bytes,
            "materialize_max_wals": validated_materialize_wals,
            "materialize_max_input_bytes": validated_materialize_bytes,
            "materialize_max_input_entries": validated_materialize_entries,
            "materialize_max_memory_bytes": validated_materialize_memory,
            "fold_min_runs": validated_fold_min_runs,
            "fold_max_runs": validated_fold_max_runs,
            "fold_max_input_bytes": validated_fold_bytes,
            "fold_max_input_entries": validated_fold_entries,
            **validated_clean,
            "clean_min_reclaimable_bytes": validated_clean_reclaimable,
            "clean_min_garbage_ratio_ppm": validated_clean_ratio,
            # Record the effective cursor rather than the caller's optional
            # value so the report fully describes the bounded scan it ran.
            "clean_start_after_segment_id": validated_clean_cursor,
            "orphan_min_age_seconds": orphan_age,
            "orphan_max_files": validated_orphan_files,
            "orphan_max_bytes": validated_orphan_bytes,
        }

        def stats_snapshot() -> dict[str, object]:
            value = native.mutation_log_stats
            if not isinstance(value, dict):
                raise RuntimeError("native mutation-log stats are invalid")
            return dict(value)

        def stat_int(stats: Mapping[str, object], name: str) -> int:
            value = stats.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"native mutation-log stat {name!r} is invalid")
            return value

        started_unix_ns = time.time_ns()
        started = time.monotonic()
        before = stats_snapshot()
        current = dict(before)
        actions: list[dict[str, object]] = []
        actions_attempted = 0
        actions_performed = 0
        deadline_reached = False
        action_budget_reached = False

        def may_attempt() -> bool:
            nonlocal deadline_reached, action_budget_reached
            if actions_attempted >= validated_max_actions:
                action_budget_reached = True
                return False
            if time.monotonic() - started >= seconds:
                deadline_reached = True
                return False
            return True

        def record_skipped(name: str, planned: bool, reason: str) -> None:
            actions.append(
                {
                    "name": name,
                    "planned": planned,
                    "attempted": False,
                    "performed": False,
                    "reason": reason,
                    "before": dict(current),
                    "after": dict(current),
                    "result": None,
                }
            )

        rotate_planned = (
            stat_int(current, "active_wal_transaction_count") > 0
            and stat_int(current, "active_wal_bytes") >= validated_rotate_bytes
        )
        if dry_run:
            record_skipped(
                "rotate_wal", rotate_planned,
                "threshold_reached" if rotate_planned else "below_threshold",
            )
        elif not rotate_planned:
            record_skipped("rotate_wal", False, "below_threshold")
        elif not may_attempt():
            record_skipped("rotate_wal", True, "maintenance_budget_reached")
        else:
            actions_attempted += 1
            action_before = dict(current)
            result = self.rotate_wal()
            performed = bool(result.get("rotated"))
            actions_performed += int(performed)
            current = stats_snapshot()
            actions.append({"name": "rotate_wal", "planned": True,
                            "attempted": True, "performed": performed,
                            "reason": "completed", "before": action_before,
                            "after": dict(current), "result": result})

        materialize_planned = stat_int(current, "frozen_wal_count") > 0
        if dry_run:
            record_skipped(
                "materialize_frozen_wals", materialize_planned,
                "frozen_wals_present" if materialize_planned else "no_frozen_wals",
            )
        elif not materialize_planned:
            record_skipped("materialize_frozen_wals", False, "no_frozen_wals")
        elif not may_attempt():
            record_skipped(
                "materialize_frozen_wals", True, "maintenance_budget_reached"
            )
        else:
            actions_attempted += 1
            action_before = dict(current)
            result = self.materialize_frozen_wals(
                max_wals=validated_materialize_wals,
                max_input_bytes=validated_materialize_bytes,
                max_input_entries=validated_materialize_entries,
                max_memory_bytes=validated_materialize_memory,
            )
            performed = bool(result.get("materialized"))
            actions_performed += int(performed)
            reason = (
                "input_exceeds_budget"
                if result.get("materialization_budget_blocked")
                else "completed"
            )
            current = stats_snapshot()
            actions.append({"name": "materialize_frozen_wals", "planned": True,
                            "attempted": True, "performed": performed,
                            "reason": reason, "before": action_before,
                            "after": dict(current), "result": result})

        fold_planned = (
            stat_int(current, "index_run_count")
            >= validated_fold_min_runs
        )
        if dry_run:
            record_skipped(
                "fold_partitions", fold_planned,
                "threshold_reached" if fold_planned else "below_threshold",
            )
        elif not fold_planned:
            record_skipped("fold_partitions", False, "below_threshold")
        elif not may_attempt():
            record_skipped("fold_partitions", True, "maintenance_budget_reached")
        else:
            actions_attempted += 1
            action_before = dict(current)
            result = self.fold_partitions(
                max_runs=validated_fold_max_runs,
                max_input_bytes=validated_fold_bytes,
                max_input_entries=validated_fold_entries,
            )
            performed = bool(result.get("folded"))
            actions_performed += int(performed)
            current = stats_snapshot()
            actions.append({"name": "fold_partitions", "planned": True,
                            "attempted": True, "performed": performed,
                            "reason": (
                                "completed" if performed
                                else (
                                    "input_exceeds_budget"
                                    if result.get("fold_budget_blocked")
                                    else "no_eligible_partition"
                                )
                            ), "before": action_before,
                            "after": dict(current), "result": result})

        clean_kwargs = {
            "max_candidate_segments": validated_clean[
                "clean_max_candidate_segments"
            ],
            "max_sidecar_scan_bytes": validated_clean[
                "clean_max_sidecar_scan_bytes"
            ],
            "max_source_bytes": validated_clean[
                "clean_max_source_bytes"
            ],
            "max_output_bytes": validated_clean[
                "clean_max_output_bytes"
            ],
            "max_memory_bytes": validated_clean[
                "clean_max_memory_bytes"
            ],
            "min_garbage_rows": validated_clean[
                "clean_min_garbage_rows"
            ],
            "min_reclaimable_bytes": validated_clean_reclaimable,
            "min_garbage_ratio_ppm": validated_clean_ratio,
            "start_after_segment_id": validated_clean_cursor,
        }
        if dry_run:
            clean_plan = self.clean_segments(dry_run=True, **clean_kwargs)
            clean_planned = clean_plan["candidate_found"]
            actions.append(
                {
                    "name": "clean_segments",
                    "planned": clean_planned,
                    "attempted": False,
                    "performed": False,
                    "reason": (
                        "candidate_found"
                        if clean_planned
                        else (
                            "budget_blocked"
                            if clean_plan["budget_blocked"]
                            else "no_candidate"
                        )
                    ),
                    "before": dict(current),
                    "after": dict(current),
                    "result": clean_plan,
                }
            )
        elif actions_attempted >= validated_max_actions:
            action_budget_reached = True
            record_skipped(
                "clean_segments", True, "maintenance_budget_reached"
            )
        elif time.monotonic() - started >= seconds:
            deadline_reached = True
            record_skipped(
                "clean_segments", True, "maintenance_budget_reached"
            )
        else:
            # Execute performs its own bounded selection while holding the
            # native lifecycle locks. A separate dry-run here would scan
            # every candidate sidecar twice and could consume twice the
            # advertised scan budget in one maintenance pass.
            actions_attempted += 1
            action_before = dict(current)
            clean_result = self.clean_segments(
                dry_run=False, **clean_kwargs
            )
            performed = clean_result["cleaned"]
            actions_performed += int(performed)
            current = stats_snapshot()
            actions.append(
                {
                    "name": "clean_segments",
                    "planned": clean_result["candidate_found"],
                    "attempted": True,
                    "performed": performed,
                    "reason": (
                        "completed"
                        if performed
                        else (
                            "budget_blocked"
                            if clean_result["budget_blocked"]
                            else "no_candidate"
                        )
                    ),
                    "before": action_before,
                    "after": dict(current),
                    "result": clean_result,
                }
            )

        if dry_run:
            orphan_result = self.collect_orphans(
                dry_run=True,
                min_age_seconds=orphan_age,
                max_files=validated_orphan_files,
                max_bytes=validated_orphan_bytes,
            )
            orphan_planned = orphan_result["eligible_file_count"] > 0
            actions.append({"name": "collect_orphans", "planned": orphan_planned,
                            "attempted": False, "performed": False,
                            "reason": "eligible_files_present" if orphan_planned else "no_eligible_files",
                            "before": dict(current), "after": dict(current),
                            "result": orphan_result})
        elif not may_attempt():
            record_skipped("collect_orphans", True, "maintenance_budget_reached")
        else:
            actions_attempted += 1
            orphan_result = self.collect_orphans(
                dry_run=False,
                min_age_seconds=orphan_age,
                max_files=validated_orphan_files,
                max_bytes=validated_orphan_bytes,
            )
            assert orphan_result["dry_run"] is False
            performed = orphan_result["removed_file_count"] > 0
            actions_performed += int(performed)
            actions.append({"name": "collect_orphans", "planned": True,
                            "attempted": True, "performed": performed,
                            "reason": "completed", "before": dict(current),
                            "after": dict(current), "result": orphan_result})

        after = dict(before) if dry_run else stats_snapshot()
        finished_unix_ns = time.time_ns()
        return {
            "format": "throwdb-maintenance-report",
            "version": 2,
            "dry_run": dry_run,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": finished_unix_ns,
            "elapsed_seconds": time.monotonic() - started,
            "policy": policy,
            "before": before,
            "after": after,
            "actions": actions,
            "actions_performed": actions_performed,
            "deadline_reached": deadline_reached,
            "action_budget_reached": action_budget_reached,
            "work_performed": actions_performed > 0,
        }

    def stream_get(
        self,
        ids: object,
        *,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_STREAM_BATCH_BYTES,
        max_snapshot_bytes: int = _DEFAULT_STREAM_SNAPSHOT_BYTES,
        prefix_dimensions: int | None = None,
    ) -> EmbeddedReadStream:
        """Open a bounded query-order stream against one collection snapshot.

        ``ids`` must be a stable one-dimensional integer ndarray/memmap or an
        ordered ``Sequence`` with a stable length. It is retained and
        normalized lazily, so callers must not mutate its structure or values
        until the returned stream is closed. Previously yielded batches remain
        valid if a later batch contains a missing/invalid ID and fails.
        """

        storage = self._ensure_reader()
        rows = _validate_id_scan_batch_rows(batch_rows)
        batch_bytes = _validate_stream_byte_limit(
            max_batch_bytes,
            name="max_batch_bytes",
        )
        snapshot_bytes = _validate_stream_byte_limit(
            max_snapshot_bytes,
            name="max_snapshot_bytes",
        )
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        # Conservatively include the per-batch normalized uint64 IDs even
        # when a native uint64 ndarray/memmap takes the zero-copy fast path.
        bytes_per_row = 8 + prefix * np.dtype(self.dtype).itemsize
        effective_rows = _effective_stream_batch_rows(
            rows,
            batch_bytes,
            bytes_per_row=bytes_per_row,
        )
        source = _StreamingNumericIdSource(ids)
        native = storage.open_read_view(
            prefix_dimensions=prefix,
            max_snapshot_bytes=snapshot_bytes,
        )
        return EmbeddedReadStream(
            native,
            source=source,
            batch_rows=effective_rows,
            prefix_dimensions=prefix,
            dtype=self.dtype,
        )

    def get(
        self,
        ids: object,
        *,
        prefix_dimensions: int | None = None,
    ) -> np.ndarray:
        storage = self._ensure_reader()
        normalized_ids = _normalize_numeric_ids(ids)
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        full = storage.get(normalized_ids)
        if prefix == self.dimension:
            return full
        # Dense rows are currently read at full width. Copying makes the public
        # prefix result C-contiguous and independent of the temporary matrix.
        return full[:, :prefix].copy(order="C")

    def get_into(
        self,
        ids: object,
        output: object,
        *,
        prefix_dimensions: int | None = None,
    ) -> None:
        storage = self._ensure_reader()
        normalized_ids = _normalize_numeric_ids(ids)
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        validated_output = _validate_output(
            output,
            expected_rows=len(normalized_ids),
            expected_columns=prefix,
            dtype=self.dtype,
        )
        if prefix == self.dimension:
            storage.get_into(normalized_ids, validated_output)
            return

        # Populate caller memory only after the complete read succeeds. A
        # missing ID therefore leaves output untouched, just like native get.
        full = storage.get(normalized_ids)
        np.copyto(validated_output, full[:, :prefix], casting="no")

    def get_into_profiled(
        self,
        ids: object,
        output: object,
    ) -> dict[str, object]:
        """Read full rows and return opt-in native phase diagnostics.

        Unlike :meth:`get_into`, this diagnostic method reads clocks inside
        the native engine. Normal application reads remain uninstrumented.
        """

        storage = self._ensure_reader()
        normalized_ids = _normalize_numeric_ids(ids)
        validated_output = _validate_output(
            output,
            expected_rows=len(normalized_ids),
            expected_columns=self.dimension,
            dtype=self.dtype,
        )
        stats = storage.get_into_profiled(normalized_ids, validated_output)
        if not isinstance(stats, dict):
            raise RuntimeError("native profiled read returned invalid diagnostics")
        return dict(stats)

    def scan_records(
        self,
        *,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_STREAM_BATCH_BYTES,
        max_snapshot_bytes: int = _DEFAULT_STREAM_SNAPSHOT_BYTES,
        prefix_dimensions: int | None = None,
        with_locations: bool = False,
    ) -> EmbeddedRecordScan:
        """Stream every live ID and embedding from one physical-order view.

        Each returned :class:`RecordBatch` contains bit-exact IDs and
        embeddings from the same format-7 snapshot. Batch row count is bounded
        by both ``batch_rows`` and ``max_batch_bytes`` and may be smaller at a
        physical segment boundary. A native failure closes the cursor and the
        failing partial batch is never yielded.
        """

        storage = self._ensure_reader()
        rows = _validate_id_scan_batch_rows(batch_rows)
        if not isinstance(with_locations, bool):
            raise TypeError("with_locations must be a bool")
        batch_bytes = _validate_stream_byte_limit(
            max_batch_bytes,
            name="max_batch_bytes",
        )
        snapshot_bytes = _validate_stream_byte_limit(
            max_snapshot_bytes,
            name="max_snapshot_bytes",
        )
        prefix = _validate_prefix_dimensions(prefix_dimensions, self.dimension)
        bytes_per_row = (
            8
            + prefix * np.dtype(self.dtype).itemsize
            + (ID_LOCATION_DTYPE.itemsize if with_locations else 0)
        )
        effective_rows = _effective_stream_batch_rows(
            rows,
            batch_bytes,
            bytes_per_row=bytes_per_row,
        )
        native = storage.scan_records(
            prefix_dimensions=prefix,
            max_snapshot_bytes=snapshot_bytes,
        )
        return EmbeddedRecordScan(
            native,
            batch_rows=effective_rows,
            prefix_dimensions=prefix,
            dtype=self.dtype,
            with_locations=with_locations,
        )

    @overload
    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
        with_locations: Literal[False] = False,
        max_snapshot_bytes: int | None = None,
    ) -> EmbeddedIdScan: ...

    @overload
    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
        with_locations: Literal[True],
        max_snapshot_bytes: int | None = None,
    ) -> EmbeddedIdScan: ...

    def scan_ids(
        self,
        *,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
        with_locations: bool = False,
        max_snapshot_bytes: int | None = None,
    ) -> EmbeddedIdScan:
        """Open a snapshot cursor in physical ``(segment_id, row_index)`` order.

        Only the current live version of every ID is returned.  Locations are
        exact for the cursor snapshot, but may change after updates or storage
        maintenance and therefore must not be reused with another snapshot.
        """

        rows = _validate_id_scan_batch_rows(batch_rows)
        if not isinstance(with_locations, bool):
            raise TypeError("with_locations must be a bool")
        snapshot_limit = (
            None
            if max_snapshot_bytes is None
            else _validate_stream_byte_limit(
                max_snapshot_bytes,
                name="max_snapshot_bytes",
            )
        )
        native = self._ensure_reader().scan_ids(
            max_snapshot_bytes=snapshot_limit
        )
        return EmbeddedIdScan(
            native,
            batch_rows=rows,
            with_locations=with_locations,
        )

    @overload
    def list_ids(
        self,
        *,
        with_locations: Literal[False] = False,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
    ) -> np.ndarray: ...

    @overload
    def list_ids(
        self,
        *,
        with_locations: Literal[True],
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
    ) -> LocatedIds: ...

    def list_ids(
        self,
        *,
        with_locations: bool = False,
        batch_rows: int = _DEFAULT_ID_SCAN_BATCH_ROWS,
    ) -> np.ndarray | LocatedIds:
        """Materialize all live IDs in physical storage order.

        There is intentionally no memory guard.  This method performs one
        exact-size allocation for IDs (and optionally one for locations).  Use
        :meth:`scan_ids` when the complete result may not fit in memory.
        """

        with self.scan_ids(
            batch_rows=batch_rows,
            with_locations=with_locations,
        ) as scan:
            count = scan.snapshot_count
            ids = np.empty(count, dtype=np.uint64)
            locations = (
                np.empty(count, dtype=ID_LOCATION_DTYPE)
                if with_locations
                else None
            )
            offset = 0
            while offset != count:
                stop = min(count, offset + scan._batch_rows)
                location_slice = (
                    locations[offset:stop]
                    if locations is not None
                    else None
                )
                produced = scan.read_into(ids[offset:stop], location_slice)
                if produced == 0:
                    raise RuntimeError(
                        "ID scan ended before its advertised snapshot count"
                    )
                offset += produced
            if not scan.done:
                probe_ids = np.empty(1, dtype=np.uint64)
                probe_locations = (
                    np.empty(1, dtype=ID_LOCATION_DTYPE)
                    if locations is not None
                    else None
                )
                if scan.read_into(probe_ids, probe_locations) != 0:
                    raise RuntimeError(
                        "ID scan exceeded its advertised snapshot count"
                    )
            if locations is None:
                return ids
            return LocatedIds(
                ids=ids,
                locations=locations,
                snapshot=scan.snapshot,
            )

    def clean_segments(
        self,
        *,
        dry_run: bool = True,
        max_candidate_segments: int = _DEFAULT_CLEAN_MAX_CANDIDATE_SEGMENTS,
        max_sidecar_scan_bytes: int = _DEFAULT_CLEAN_MAX_SIDECAR_SCAN_BYTES,
        max_source_bytes: int = _DEFAULT_CLEAN_MAX_SOURCE_BYTES,
        max_output_bytes: int = _DEFAULT_CLEAN_MAX_OUTPUT_BYTES,
        max_memory_bytes: int = _DEFAULT_CLEAN_MAX_MEMORY_BYTES,
        min_garbage_rows: int = _DEFAULT_CLEAN_MIN_GARBAGE_ROWS,
        min_reclaimable_bytes: int = _DEFAULT_CLEAN_MIN_RECLAIMABLE_BYTES,
        min_garbage_ratio_ppm: int = _DEFAULT_CLEAN_MIN_GARBAGE_RATIO_PPM,
        start_after_segment_id: int | None = None,
    ) -> SegmentCleaningReport:
        """Plan or execute one bounded selective segment rewrite.

        Only format-7 collections support the authoritative Cleaner run used
        by this operation. A call retires at most one immutable segment and
        deliberately leaves its pathnames for :meth:`collect_orphans`.
        Dry runs never advance this handle's in-memory scan cursor.
        """

        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool")
        limits = {
            "max_candidate_segments": max_candidate_segments,
            "max_sidecar_scan_bytes": max_sidecar_scan_bytes,
            "max_source_bytes": max_source_bytes,
            "max_output_bytes": max_output_bytes,
            "max_memory_bytes": max_memory_bytes,
            "min_garbage_rows": min_garbage_rows,
        }
        validated = {
            name: _validate_resource_limit(value, name=name)
            for name, value in limits.items()
        }
        validated_reclaimable = _validate_resource_limit(
            min_reclaimable_bytes,
            name="min_reclaimable_bytes",
            minimum=0,
        )
        validated_ratio = _validate_resource_limit(
            min_garbage_ratio_ppm,
            name="min_garbage_ratio_ppm",
            minimum=0,
            maximum=1_000_000,
        )
        cursor = (
            self._cleaning_cursor
            if start_after_segment_id is None
            else _validate_resource_limit(
                start_after_segment_id,
                name="start_after_segment_id",
                minimum=0,
                maximum=_ROW_REF_SEGMENT_ID_MAX,
            )
        )
        native = self._ensure_mutable("clean immutable segments")
        raw = native.clean_segments(
            dry_run=dry_run,
            max_candidate_segments=validated["max_candidate_segments"],
            max_sidecar_scan_bytes=validated["max_sidecar_scan_bytes"],
            max_source_bytes=validated["max_source_bytes"],
            max_output_bytes=validated["max_output_bytes"],
            max_memory_bytes=validated["max_memory_bytes"],
            min_garbage_rows=validated["min_garbage_rows"],
            min_reclaimable_bytes=validated_reclaimable,
            min_garbage_ratio_ppm=validated_ratio,
            start_after_segment_id=cursor,
        )
        if not isinstance(raw, dict):
            raise RuntimeError(
                "native segment cleaner returned invalid diagnostics"
            )
        next_cursor = raw.get("next_start_after_segment_id")
        if (
            not isinstance(next_cursor, int)
            or isinstance(next_cursor, bool)
            or not 0 <= next_cursor <= _ROW_REF_SEGMENT_ID_MAX
        ):
            raise RuntimeError(
                "native segment cleaner returned an invalid scan cursor"
            )
        if not dry_run:
            self._cleaning_cursor = next_cursor
        return cast(SegmentCleaningReport, dict(raw))

    def __enter__(self) -> Collection:
        if self._builder is not None:
            self._ensure_builder_open()
        else:
            self._ensure_native_open("enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        if self._builder is not None:
            if exc_type is None and self._state == "building":
                self.finish()
            elif exc_type is not None:
                self.abort()
        else:
            self.close()
        return False


__all__ = [
    "Collection",
    "CollectionSchema",
    "EmbeddedIdScan",
    "EmbeddedReadStream",
    "EmbeddedRecordScan",
    "MaintenanceReport",
    "OrphanClassification",
    "OrphanCollectionExecutionEntry",
    "OrphanCollectionExecutionReport",
    "OrphanCollectionReport",
    "OrphanInventoryEntry",
    "SegmentCleaningReport",
]
