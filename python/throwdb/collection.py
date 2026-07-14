"""High-level collection API built on top of the strict native bindings."""

from __future__ import annotations

import copy
import math
import numbers
import operator
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from types import TracebackType
from typing import Final, Literal, TypeAlias, TypedDict, cast

import numpy as np

from ._throwdb import (
    NativeCollection,
    NativeCollectionBuilder,
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


_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_DEFAULT_SHARD_SIZE_BYTES: Final = 16 << 30
_ID_SCAN_CHUNK_ROWS: Final = 1 << 20
_DEFAULT_DELTA_COMPACTION_MAX_RUNS: Final = 8
_MAX_DELTA_COMPACTION_RUNS: Final = 64
_DEFAULT_DELTA_COMPACTION_MAX_INPUT_BYTES: Final = 8 << 30
_DEFAULT_DELTA_COMPACTION_MAX_INPUT_ENTRIES: Final = 1_000_000
_DEFAULT_ORPHAN_MIN_AGE_SECONDS: Final = 3_600.0

_ORPHAN_CANDIDATE_CLASSES: Final = frozenset(
    {
        "temp_publication",
        "unreferenced_wal",
        "complete_unreferenced_delta_pair",
        "partial_unreferenced_delta_pair",
    }
)
_ORPHAN_INVENTORY_CLASS_NAMES: Final = (
    "reachable",
    "superseded_catalog",
    "temp_publication",
    "unreferenced_wal",
    "complete_unreferenced_delta_pair",
    "partial_unreferenced_delta_pair",
    "unknown_do_not_touch",
)
_ORPHAN_INVENTORY_CLASSES: Final = frozenset(
    _ORPHAN_INVENTORY_CLASS_NAMES
)

_READER_POLICY_ABI: Final = "throwdb-reader-v1"


OrphanClassification: TypeAlias = Literal[
    "reachable",
    "superseded_catalog",
    "temp_publication",
    "unreferenced_wal",
    "complete_unreferenced_delta_pair",
    "partial_unreferenced_delta_pair",
    "unknown_do_not_touch",
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


def _first_negative_position(ids: np.ndarray) -> int | None:
    """Return the first negative element without allocating an N-row mask."""

    for begin in range(0, ids.size, _ID_SCAN_CHUNK_ROWS):
        chunk = ids[begin : begin + _ID_SCAN_CHUNK_ROWS]
        negative = chunk < 0
        if bool(negative.any()):
            return begin + int(np.flatnonzero(negative)[0])
    return None


def _normalize_numeric_ids(
    ids: object,
    *,
    position_offset: int = 0,
) -> np.ndarray:
    """Validate numeric IDs and return native, aligned, contiguous uint64.

    A native aligned C-contiguous uint64 ndarray is returned by identity. All
    other accepted inputs receive a safe, exact conversion. This function is
    intentionally kept separate from the native binding: callers that require
    control over every allocation can continue to use ``Writer``/``Storage``.
    """

    if isinstance(ids, np.ma.MaskedArray):
        raise TypeError("ids must not be a masked array")

    if isinstance(ids, np.ndarray):
        if ids.ndim != 1:
            raise ValueError(f"ids must have shape (N,), got {ids.shape}")
        if ids.dtype.kind not in {"i", "u"} or ids.dtype.itemsize not in {
            1,
            2,
            4,
            8,
        }:
            raise TypeError(
                "ids ndarray must have an 8-, 16-, 32-, or 64-bit integer "
                f"dtype, got {ids.dtype}"
            )

        if ids.dtype.kind == "i":
            position = _first_negative_position(ids)
            if position is not None:
                absolute_position = position_offset + position
                raise ValueError(
                    f"ids[{absolute_position}]={int(ids[position])} is negative; "
                    "numeric IDs must be between 0 and 2**64-1"
                )

        native_uint64 = np.dtype(np.uint64)
        if (
            ids.dtype == native_uint64
            and ids.dtype.isnative
            and ids.flags.c_contiguous
            and ids.flags.aligned
        ):
            return ids

        # ascontiguousarray() alone is allowed to retain an unaligned buffer.
        # np.require(..., "A") makes the low-level binding contract explicit.
        return np.require(ids, dtype=native_uint64, requirements=["C", "A"])

    if isinstance(ids, (str, bytes, bytearray, memoryview)):
        raise TypeError(
            "numeric ids must be a one-dimensional integer array or sequence"
        )
    if not isinstance(ids, Sequence):
        raise TypeError(
            "numeric ids must be a one-dimensional integer array or ordered sequence"
        )

    expected_length = len(ids)
    normalized = np.empty(expected_length, dtype=np.uint64)
    actual_length = 0
    for position, value in enumerate(ids):
        absolute_position = position_offset + position
        if position >= expected_length:
            raise ValueError("ids sequence changed length during normalization")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"ids[{absolute_position}]={value!r} is boolean, not an integer ID"
            )
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise TypeError(
                f"ids[{absolute_position}]={value!r} is not an integer ID"
            ) from error
        if not 0 <= integer <= _UINT64_MAX:
            raise ValueError(
                f"ids[{absolute_position}]={integer} is outside the uint64 range "
                "0..2**64-1"
            )
        normalized[position] = integer
        actual_length = position + 1
    if actual_length != expected_length:
        raise ValueError("ids sequence changed length during normalization")
    return normalized


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


def _validate_shard_size_bytes(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("shard_size_bytes must be an integer, not bool")
    try:
        size = operator.index(value)
    except TypeError as error:
        raise TypeError("shard_size_bytes must be an integer") from error
    if not 1 <= size <= _UINT64_MAX:
        raise ValueError(
            "shard_size_bytes must be between 1 and 2**64-1"
        )
    return size


def _validate_delta_compaction_limit(
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


def _validate_orphan_dry_run_options(
    *,
    dry_run: object,
    min_age_seconds: object,
) -> float:
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a bool")
    if not dry_run:
        raise NotImplementedError(
            "orphan deletion is not implemented; use dry_run=True"
        )
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
    return seconds


def _seconds_to_nanoseconds_ceil(seconds: float) -> int:
    """Convert even extremely large finite seconds without float overflow."""

    whole_seconds = int(seconds)
    fractional_seconds = seconds - whole_seconds
    return whole_seconds * 1_000_000_000 + math.ceil(
        fractional_seconds * 1_000_000_000
    )


def _orphan_report(
    raw_inventory: object,
    *,
    min_age_seconds: float,
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

    snapshot_unix_ns = time.time_ns()
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

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("native orphan inventory contains an invalid entry")
        path = raw_entry.get("path")
        classification = raw_entry.get("classification")
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

        candidate = classification in _ORPHAN_CANDIDATE_CLASSES
        valid_modification_time = (
            0 <= modified_unix_ns <= snapshot_unix_ns
        )
        eligible = (
            candidate
            and valid_modification_time
            and snapshot_unix_ns - modified_unix_ns >= minimum_age_ns
        )
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
        else:
            unknown_file_count += 1
            reason = "unknown_do_not_touch"

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


def _validate_output(
    output: object,
    *,
    expected_rows: int,
    expected_columns: int,
    dtype: str,
) -> np.ndarray:
    if isinstance(output, np.ma.MaskedArray) or not isinstance(output, np.ndarray):
        raise TypeError("output must be a NumPy ndarray")
    expected_dtype = np.dtype(dtype)
    if output.dtype != expected_dtype or not output.dtype.isnative:
        raise TypeError(
            f"output must have native-endian dtype {dtype}, got {output.dtype}"
        )
    expected_shape = (expected_rows, expected_columns)
    if output.shape != expected_shape:
        raise ValueError(f"output must have shape {expected_shape}, got {output.shape}")
    if not output.flags.c_contiguous:
        raise ValueError("output must be C-contiguous")
    if not output.flags.aligned:
        raise ValueError("output must be memory-aligned")
    if not output.flags.writeable:
        raise ValueError("output must be writable")
    return output


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
        self._tuning = dict(tuning_profile) if tuning_profile is not None else None

    @classmethod
    def create(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        schema: CollectionSchema | None = None,
        shard_size_bytes: int = _DEFAULT_SHARD_SIZE_BYTES,
    ) -> Collection:
        if schema is None:
            schema = CollectionSchema()
        if not isinstance(schema, CollectionSchema):
            raise TypeError("schema must be a CollectionSchema")
        if schema.id_type != "uint64":
            raise NotImplementedError(
                "utf8 IDs are not implemented by the numeric collection format"
            )
        validated_shard_size = _validate_shard_size_bytes(shard_size_bytes)
        builder = NativeCollectionBuilder.create(
            path,
            dimension=schema.dimension,
            shard_size_bytes=validated_shard_size,
            dtype=schema.dtype,
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
    ) -> Collection:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in {"r", "r+"}:
            raise ValueError("mode must be 'r' or 'r+'")
        if not isinstance(verify_index, bool):
            raise TypeError("verify_index must be a bool")
        options, profile = _resolve_tuning(path, tuning)
        try:
            native = NativeCollection.open(
                path,
                mode=mode,
                verify_native_index=verify_index,
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
        validated_min_age = _validate_orphan_dry_run_options(
            dry_run=True,
            min_age_seconds=min_age_seconds,
        )
        return _orphan_report(
            NativeCollection.inspect_orphans(path),
            min_age_seconds=validated_min_age,
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

        The command opens one stable read-only native snapshot, samples a
        bounded number of live IDs directly from its loaded base index, and
        measures a small executor/backend matrix.  It never reloads the large
        index for individual candidates.  ``max_seconds`` is a cooperative
        deadline: one already-started OS read may finish after it expires.
        """

        from ._calibration import run_bounded_calibration, validate_limits

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
                "selection": None,
                "planned_read_bytes": 0,
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
                    "matrix_version": 2,
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
            measured = run_bounded_calibration(
                native,
                record_size=record_size,
                current_options=(
                    dict(existing.options) if existing is not None else None
                ),
                seed=seed,
                max_read_bytes=max_read_bytes,
                max_memory_bytes=max_memory_bytes,
                deadline=deadline,
                progress=progress,
            )

            selection_value = measured.get("selection")
            selection = (
                dict(selection_value)
                if isinstance(selection_value, Mapping)
                else None
            )
            status = str(measured["status"])
            persisted = False
            saved_profile: dict[str, object] | None = None
            target_stable: bool | None = None

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
            elif status == "complete" and selection is not None:
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
                        source="calibrated-v2",
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

    def materialize_frozen_wals(self) -> dict[str, object]:
        """Replace the frozen WAL prefix with an immutable delta run.

        The durable catalog is switched atomically. Referenced WAL files are
        reclaimed only after the new view is published.
        """

        native = self._ensure_mutable("materialize frozen mutation WALs")
        stats = native.materialize_frozen_wals()
        if not isinstance(stats, dict):
            raise RuntimeError(
                "native WAL materialization returned invalid diagnostics"
            )
        return dict(stats)

    def compact_deltas(
        self,
        *,
        max_runs: int = _DEFAULT_DELTA_COMPACTION_MAX_RUNS,
        max_input_bytes: int = _DEFAULT_DELTA_COMPACTION_MAX_INPUT_BYTES,
        max_input_entries: int = _DEFAULT_DELTA_COMPACTION_MAX_INPUT_ENTRIES,
    ) -> dict[str, object]:
        """Compact a bounded oldest prefix of immutable delta runs.

        At least two complete runs must fit all three limits; otherwise this
        is a successful no-op. The operation never rewrites base segments.
        """

        native = self._ensure_mutable("compact immutable mutation deltas")
        validated_max_runs = _validate_delta_compaction_limit(
            max_runs,
            name="max_runs",
            minimum=2,
            maximum=_MAX_DELTA_COMPACTION_RUNS,
        )
        validated_max_input_bytes = _validate_delta_compaction_limit(
            max_input_bytes,
            name="max_input_bytes",
        )
        validated_max_input_entries = _validate_delta_compaction_limit(
            max_input_entries,
            name="max_input_entries",
        )
        stats = native.compact_delta_runs(
            validated_max_runs,
            validated_max_input_bytes,
            validated_max_input_entries,
        )
        if not isinstance(stats, dict):
            raise RuntimeError(
                "native delta compaction returned invalid diagnostics"
            )
        return dict(stats)

    def collect_orphans(
        self,
        *,
        dry_run: bool = True,
        min_age_seconds: float = _DEFAULT_ORPHAN_MIN_AGE_SECONDS,
    ) -> OrphanCollectionReport:
        """Build an orphan-file collection plan without deleting anything.

        This first lifecycle slice intentionally supports only ``dry_run``.
        A mutable handle is required so the writer lease remains held for the
        complete native inventory snapshot.
        """

        validated_min_age = _validate_orphan_dry_run_options(
            dry_run=dry_run,
            min_age_seconds=min_age_seconds,
        )
        native = self._ensure_mutable("collect orphan mutation resources")
        return _orphan_report(
            native.collect_orphans(),
            min_age_seconds=validated_min_age,
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


__all__ = ["Collection", "CollectionSchema"]
