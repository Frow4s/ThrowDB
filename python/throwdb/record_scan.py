"""Result types for snapshot-consistent physical embedding scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .id_scan import IdScanSnapshot


@dataclass(frozen=True, slots=True)
class RecordBatch:
    """One physical-order batch from :meth:`Collection.scan_records`.

    Every field belongs to the same captured collection snapshot.  ``ids``
    and ``embeddings`` always contain the same number of rows. ``locations``
    is ``None`` unless the scan was opened with ``with_locations=True``.
    """

    ids: np.ndarray
    embeddings: np.ndarray
    locations: np.ndarray | None
    snapshot: IdScanSnapshot
    emitted_offset: int


__all__ = ["RecordBatch"]
