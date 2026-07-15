"""Shared result types for embedded and remote physical-order ID scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


ID_LOCATION_DTYPE: Final = np.dtype(
    [("segment_id", "<u4"), ("row_index", "<u4")], align=False
)


@dataclass(frozen=True, slots=True)
class IdScanSnapshot:
    """Identity of the immutable logical view pinned by one ID scan.

    ``token`` is deliberately opaque and is only meaningful together with the
    collection identity and generation.  Location hints must never be applied
    to another snapshot without native validation.
    """

    collection_id: str
    catalog_generation: int
    committed_sequence: int
    count: int
    token: str


@dataclass(frozen=True, slots=True)
class LocatedIdBatch:
    """One physical-order ID batch and its exact format-7 RowRefs."""

    ids: np.ndarray
    locations: np.ndarray
    emitted_offset: int
    snapshot: IdScanSnapshot


@dataclass(frozen=True, slots=True)
class LocatedIds:
    """Fully materialized physical-order IDs and RowRefs for one snapshot."""

    ids: np.ndarray
    locations: np.ndarray
    snapshot: IdScanSnapshot


__all__ = [
    "ID_LOCATION_DTYPE",
    "IdScanSnapshot",
    "LocatedIdBatch",
    "LocatedIds",
]
