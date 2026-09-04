"""Shared NumPy buffer contracts for embedded and remote collection APIs."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Final, cast

import numpy as np

from .id_scan import ID_LOCATION_DTYPE


_UINT64_MAX: Final = (1 << 64) - 1
_ID_SCAN_CHUNK_ROWS: Final = 1 << 20


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


def _numpy_arrays_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    """Return an exact overlap answer, with a conservative bounded fallback."""

    # Drop ndarray subclasses before invoking NumPy dispatch. This keeps the
    # check tied to the actual exported address/stride metadata and remains a
    # zero-copy view for ndarray and memmap inputs.
    left_view = np.asarray(left)
    right_view = np.asarray(right)
    if not bool(np.may_share_memory(left_view, right_view)):
        return False
    try:
        return bool(
            np.shares_memory(left_view, right_view, max_work=1024)
        )
    except (RuntimeError, ValueError):
        # Fail closed when NumPy declines bounded exact analysis for an
        # unusually complex retained stream_get source layout.
        return True


def _require_disjoint_numpy_arrays(
    arrays: Sequence[tuple[str, np.ndarray]],
    *,
    operation: str = "record scan",
) -> None:
    for left_index, (left_name, left) in enumerate(arrays):
        for right_name, right in arrays[left_index + 1 :]:
            if _numpy_arrays_overlap(left, right):
                raise ValueError(
                    f"{operation} output buffers must not overlap: "
                    f"{left_name} and {right_name}"
                )


class _BoundedSequenceWindow(Sequence[object]):
    """Non-owning fixed window used to avoid a second per-batch ID list."""

    def __init__(
        self,
        source: Sequence[object],
        begin: int,
        end: int,
    ) -> None:
        self._source = source
        self._begin = begin
        self._length = end - begin

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [
                self._source[self._begin + position]
                for position in range(start, stop, step)
            ]
        normalized = operator.index(index)
        if normalized < 0:
            normalized += self._length
        if not 0 <= normalized < self._length:
            raise IndexError(index)
        return self._source[self._begin + normalized]


class _StreamingNumericIdSource:
    """Lazily normalize a stable ndarray/Sequence in bounded slices."""

    def __init__(self, ids: object) -> None:
        if isinstance(ids, np.ma.MaskedArray):
            raise TypeError("ids must not be a masked array")
        self._array: np.ndarray | None = None
        self._sequence: Sequence[object] | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype[object] | None = None
        self._strides: tuple[int, ...] | None = None
        self._data_pointer: int | None = None

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
                    "ids ndarray must have an 8-, 16-, 32-, or 64-bit "
                    f"integer dtype, got {ids.dtype}"
                )
            self._array = ids
            self._length = int(ids.shape[0])
            self._shape = tuple(ids.shape)
            self._dtype = ids.dtype
            self._strides = tuple(ids.strides)
            self._data_pointer = int(ids.ctypes.data)
            return

        if isinstance(ids, (str, bytes, bytearray, memoryview)):
            raise TypeError(
                "numeric ids must be a one-dimensional integer array or sequence"
            )
        if not isinstance(ids, Sequence):
            raise TypeError(
                "numeric ids must be a one-dimensional integer array or "
                "ordered sequence; streaming iterables are not supported"
            )
        self._sequence = cast(Sequence[object], ids)
        self._length = len(ids)

    def __len__(self) -> int:
        return self._length

    def _require_unchanged(self) -> None:
        if self._array is not None:
            array = self._array
            if (
                array.ndim != 1
                or tuple(array.shape) != self._shape
                or array.dtype != self._dtype
                or tuple(array.strides) != self._strides
                or int(array.ctypes.data) != self._data_pointer
            ):
                raise RuntimeError(
                    "ids source structure changed while stream_get was active"
                )
            return
        assert self._sequence is not None
        if len(self._sequence) != self._length:
            raise RuntimeError(
                "ids source length changed while stream_get was active"
            )

    def batch(self, begin: int, end: int) -> np.ndarray:
        self._require_unchanged()
        if not 0 <= begin <= end <= self._length:
            raise RuntimeError("stream_get requested an invalid ID source slice")
        values: object
        if self._array is not None:
            values = self._array[begin:end]
        else:
            assert self._sequence is not None
            # The minimal Sequence protocol does not require slicing. A
            # non-owning window keeps normalization to one bounded allocation.
            values = _BoundedSequenceWindow(self._sequence, begin, end)
        normalized = _normalize_numeric_ids(values, position_offset=begin)
        if normalized.size != end - begin:
            raise ValueError("ids sequence changed length during streaming")
        # Indexing/conversion can run user code and mutate the retained source.
        # Reject this batch before handing IDs to native code or the transport.
        self._require_unchanged()
        return normalized

    def output_overlaps_unread(
        self,
        output: np.ndarray,
        *,
        begin: int,
    ) -> bool:
        """Detect address overlap with the current or a future array ID."""

        if self._array is None:
            return False
        self._require_unchanged()
        if not 0 <= begin <= self._length:
            raise RuntimeError("stream_get requested an invalid ID source offset")
        return _numpy_arrays_overlap(self._array[begin:], output)


def _validate_stream_output(
    output: object,
    *,
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
    if output.ndim != 2 or output.shape[1] != expected_columns:
        raise ValueError(
            "output must have shape (N, "
            f"{expected_columns}), got {output.shape}"
        )
    if output.shape[0] == 0:
        raise ValueError("output must contain at least one row")
    if not output.flags.c_contiguous:
        raise ValueError("output must be C-contiguous")
    if not output.flags.aligned:
        raise ValueError("output must be memory-aligned")
    if not output.flags.writeable:
        raise ValueError("output must be writable")
    return output


def _validate_id_scan_output(
    ids: object,
    locations: object | None,
    *,
    with_locations: bool,
    operation: str = "scan_ids",
) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(ids, np.ma.MaskedArray) or not isinstance(ids, np.ndarray):
        raise TypeError("ids output must be a NumPy ndarray, not a masked array")
    if ids.dtype != np.dtype(np.uint64) or not ids.dtype.isnative:
        raise TypeError("ids output must have native-endian dtype uint64")
    if ids.ndim != 1:
        raise ValueError("ids output must have shape (N,)")
    if not ids.flags.c_contiguous:
        raise ValueError("ids output must be C-contiguous")
    if not ids.flags.aligned:
        raise ValueError("ids output must be memory-aligned")
    if not ids.flags.writeable:
        raise ValueError("ids output must be writable")
    if ids.size == 0:
        raise ValueError("ids output must contain at least one row")

    if not with_locations:
        if locations is not None:
            raise ValueError(
                "locations output requires "
                f"{operation}(with_locations=True)"
            )
        return ids, None
    if isinstance(locations, np.ma.MaskedArray) or not isinstance(
        locations, np.ndarray
    ):
        raise TypeError("locations output must be a NumPy ndarray, not a masked array")
    if locations.dtype != ID_LOCATION_DTYPE:
        raise TypeError(
            "locations output must have dtype "
            "[('segment_id', '<u4'), ('row_index', '<u4')]"
        )
    if locations.ndim != 1 or locations.shape != ids.shape:
        raise ValueError("locations output must have shape (len(ids),)")
    if not locations.flags.c_contiguous:
        raise ValueError("locations output must be C-contiguous")
    if not locations.flags.aligned:
        raise ValueError("locations output must be memory-aligned")
    if not locations.flags.writeable:
        raise ValueError("locations output must be writable")
    if _numpy_arrays_overlap(ids, locations):
        raise ValueError(
            f"{operation} output buffers must not overlap: ids and locations"
        )
    return ids, locations
