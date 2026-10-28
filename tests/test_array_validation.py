"""Regression coverage for the common embedded/remote NumPy boundary."""

from collections.abc import Sequence

import numpy as np
import pytest

from throwdb._array_validation import _validate_id_scan_output
from throwdb.collection import EmbeddedReadStream, _StreamingNumericIdSource
from throwdb.id_scan import ID_LOCATION_DTYPE
from throwdb.remote import _IndexedIds, _reject_array_overlap


class _GrowingIds(Sequence[int]):
    def __init__(self) -> None:
        self.values = [10]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        if index == 0:
            self.values.append(20)
        return self.values[index]


class _ReadView:
    output_dimension = 2
    dtype = "float16"
    metadata = {
        "collection_id_low": 1,
        "collection_id_high": 2,
        "catalog_generation": 3,
        "committed_sequence": 4,
        "live_count": 2,
        "snapshot_bytes": 4096,
    }

    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    def get_into(self, ids: np.ndarray, output: np.ndarray) -> None:
        self.reads += 1
        output[:] = ids[:, None]

    def close(self) -> None:
        self.closed = True


def test_embedded_stream_rejects_source_growth_during_normalization() -> None:
    native = _ReadView()
    stream = EmbeddedReadStream(
        native,
        source=_StreamingNumericIdSource(_GrowingIds()),
        batch_rows=1,
        prefix_dimensions=2,
        dtype="float16",
    )
    output = np.full((1, 2), -123, dtype=np.float16)
    with pytest.raises(RuntimeError, match="source length changed"):
        stream.read_into(output)
    assert native.reads == 0
    assert native.closed and stream.closed and stream.failed
    assert stream.emitted_count == 0
    np.testing.assert_array_equal(output, [[-123, -123]])


def test_remote_source_rejects_growth_during_normalization() -> None:
    source = _IndexedIds(_GrowingIds())
    with pytest.raises(RuntimeError, match="source length changed"):
        source.normalized_slice(0, 1)


class _CustomArray(np.ndarray):
    """An array subclass whose dispatch cannot be trusted for buffer safety."""

    def __array_function__(self, function, types, args, kwargs):
        if function is np.may_share_memory:
            return False
        return super().__array_function__(function, types, args, kwargs)


def test_remote_stream_overlap_guard_uses_underlying_subclass_memory() -> None:
    backing = np.array([10, 20], dtype=np.uint64)
    source = _IndexedIds(backing.view(_CustomArray))
    output = backing.view(np.float16).reshape(2, 4)
    with pytest.raises(ValueError, match="must not overlap"):
        source.reject_output_overlap(output)
    np.testing.assert_array_equal(backing, [10, 20])


def test_remote_record_overlap_guard_uses_underlying_subclass_memory() -> None:
    ids = np.array([10, 20], dtype=np.uint64)
    embeddings = ids.view(np.float16).reshape(2, 4).view(_CustomArray)
    with pytest.raises(ValueError, match="must not overlap"):
        _reject_array_overlap(
            [("ids", ids), ("embeddings", embeddings)],
            operation="ScanRecords",
        )
    np.testing.assert_array_equal(ids, [10, 20])


def test_remote_stream_allows_disjoint_arrays_with_interleaved_bounds() -> None:
    # The source bounding box includes output, but its actual elements do not.
    backing = np.array([10, 0, 20], dtype=np.uint64)
    source = _IndexedIds(backing[::2])
    output = backing[1:2].view(np.float16).reshape(1, 4)
    source.reject_output_overlap(output)
    np.testing.assert_array_equal(source.normalized_slice(0, 2), [10, 20])


@pytest.mark.parametrize("operation", ["scan_ids", "scan_records"])
@pytest.mark.parametrize("masked_buffer", ["ids", "locations"])
def test_scan_outputs_reject_masked_arrays(operation: str, masked_buffer: str) -> None:
    ids = np.zeros(2, dtype=np.uint64)
    locations = np.zeros(2, dtype=ID_LOCATION_DTYPE)
    if masked_buffer == "ids":
        ids = np.ma.array(ids, mask=True)
    else:
        locations = np.ma.array(locations, mask=True)
    with pytest.raises(TypeError, match=f"{masked_buffer} output.*masked"):
        _validate_id_scan_output(
            ids,
            locations,
            with_locations=True,
            operation=operation,
        )
