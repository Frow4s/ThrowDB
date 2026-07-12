"""High-level collection API built on top of the strict native bindings."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from types import TracebackType
from typing import Final

import numpy as np

from ._throwdb import Storage, Writer


_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_DEFAULT_SHARD_SIZE_BYTES: Final = 16 << 30
_ID_SCAN_CHUNK_ROWS: Final = 1 << 20

_READER_DEFAULTS: Final[dict[str, object]] = {
    "io_workers": 0,
    "io_queue_capacity": 0,
    "io_ranges_per_task": 8,
    "io_backend": "thread_pool",
    "io_uring_queue_depth": 32,
    "io_uring_workers": 1,
}
_NATIVE_TUNING_KEYS: Final = frozenset(_READER_DEFAULTS)
_TUNING_ALIAS_KEYS: Final = frozenset(
    {"backend", "workers", "queue_capacity", "ranges_per_task", "queue_depth"}
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


def _resolve_tuning(
    tuning: str | Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(tuning, str):
        if tuning != "auto":
            raise ValueError("tuning must be 'auto' or a mapping of reader options")
        return {}, {
            "mode": "auto",
            "reason": "conservative local-reader default; no calibration was run",
            **_READER_DEFAULTS,
        }
    if not isinstance(tuning, Mapping):
        raise TypeError("tuning must be 'auto' or a mapping of reader options")

    raw = dict(tuning)
    non_string_keys = [key for key in raw if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"tuning keys must be strings, got {non_string_keys[0]!r}")
    unknown = set(raw) - _NATIVE_TUNING_KEYS - _TUNING_ALIAS_KEYS
    if unknown:
        raise ValueError(f"unknown tuning option {sorted(unknown)[0]!r}")

    options = {key: raw[key] for key in _NATIVE_TUNING_KEYS if key in raw}

    if "backend" in raw:
        if "io_backend" in options:
            raise ValueError("use only one of 'backend' and 'io_backend'")
        options["io_backend"] = raw["backend"]
    backend = options.get("io_backend", _READER_DEFAULTS["io_backend"])

    if "workers" in raw:
        if "io_workers" in options or "io_uring_workers" in options:
            raise ValueError(
                "use 'workers' or a backend-specific worker option, not both"
            )
        destination = "io_uring_workers" if backend == "io_uring" else "io_workers"
        options[destination] = raw["workers"]

    aliases = {
        "queue_capacity": "io_queue_capacity",
        "ranges_per_task": "io_ranges_per_task",
        "queue_depth": "io_uring_queue_depth",
    }
    for alias, destination in aliases.items():
        if alias not in raw:
            continue
        if destination in options:
            raise ValueError(f"use only one of {alias!r} and {destination!r}")
        options[destination] = raw[alias]

    effective = {**_READER_DEFAULTS, **options}
    profile = {
        "mode": "manual",
        "reason": "explicit user configuration",
        **effective,
    }
    return options, profile


class Collection:
    """User-facing collection wrapper with safe ID normalization."""

    def __init__(
        self,
        *,
        schema: CollectionSchema,
        writer: Writer | None = None,
        storage: Storage | None = None,
        tuning_profile: Mapping[str, object] | None = None,
    ) -> None:
        if (writer is None) == (storage is None):
            raise RuntimeError("collection must wrap exactly one native handle")
        self._schema = schema
        self._writer = writer
        self._storage = storage
        self._state = "building" if writer is not None else "readable"
        if writer is not None:
            self._count = writer.count
            self._format_version = writer.format_version
        else:
            assert storage is not None
            self._count = storage.count
            self._format_version = storage.format_version
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
                "utf8 IDs require the future format-v2 string index"
            )
        writer = Writer.create(
            path,
            dimension=schema.dimension,
            shard_size_bytes=shard_size_bytes,
            dtype=schema.dtype,
        )
        return cls(schema=schema, writer=writer)

    @classmethod
    def open(
        cls,
        path: str | bytes | PathLike[str] | PathLike[bytes],
        *,
        tuning: str | Mapping[str, object] = "auto",
    ) -> Collection:
        options, profile = _resolve_tuning(tuning)
        storage = Storage.open(path, **options)
        schema = CollectionSchema(
            id_type="uint64",
            dimension=storage.dimension,
            dtype=storage.dtype,
        )
        return cls(schema=schema, storage=storage, tuning_profile=profile)

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
        return self._count

    @property
    def format_version(self) -> int:
        return self._format_version

    @property
    def mode(self) -> str:
        return "write" if self._writer is not None else "read"

    @property
    def tuning(self) -> dict[str, object] | None:
        """Return the selected reader profile, or ``None`` for a builder."""

        return None if self._tuning is None else dict(self._tuning)

    def _ensure_builder_open(self) -> Writer:
        if self._writer is None:
            raise RuntimeError("add is only available on a collection builder")
        if self._state != "building":
            raise RuntimeError(f"collection builder is already {self._state}")
        return self._writer

    def _ensure_reader(self) -> Storage:
        if self._storage is None:
            raise RuntimeError("get is only available on an opened collection reader")
        return self._storage

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
        if self._writer is None:
            raise RuntimeError("finish is only available on a collection builder")
        if self._state == "finished":
            return
        if self._state != "building":
            raise RuntimeError(f"collection builder is already {self._state}")
        try:
            self._writer.finish()
        except BaseException:
            self.abort()
            raise
        self._state = "finished"
        self._count = self._writer.count

    def abort(self) -> None:
        if self._writer is None:
            raise RuntimeError("abort is only available on a collection builder")
        if self._state in {"finished", "aborted"}:
            return
        self._writer.abort()
        self._state = "aborted"

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

    def __enter__(self) -> Collection:
        if self._writer is not None:
            self._ensure_builder_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        if self._writer is not None:
            if exc_type is None and self._state == "building":
                self.finish()
            elif exc_type is not None:
                self.abort()
        return False


__all__ = ["Collection", "CollectionSchema"]
