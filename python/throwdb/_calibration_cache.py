"""Best-effort payload page-cache control for reader calibration."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable
from fnmatch import fnmatchcase
from os import PathLike
from pathlib import Path
from typing import Final


_PAYLOAD_PATTERNS: Final = (
    ("segments", "segment-*.dat", "data"),
    ("wal", "*.wal", "wal"),
)
# Format 7 independently permits 1,000,000 segment resources and 1,000,000
# WAL resources.  Segments are represented by .dat/.ids pairs, and the native
# inventory allows another 65,536 uncatalogued entries.  Keep discovery
# bounded while accepting the worst valid product-owned namespace geometry.
_MAX_DISCOVERY_ENTRIES: Final = 3_065_536


def prepare_cold_payload_cache(
    root: str | bytes | PathLike[str] | PathLike[bytes],
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Advise clean ThrowDB payload pages as unused without global eviction.

    ``POSIX_FADV_DONTNEED`` is deliberately limited to embedding payload and
    WAL files belonging to the collection.  It never writes the collection or
    drops the machine-wide page cache.  The call is best effort, so the tuning
    pipeline still validates its effect with per-process backing-read counters
    before allowing automatic profile persistence.
    """

    started = clock()
    method = "posix_fadvise_dontneed"
    base = Path(os.fsdecode(os.fspath(root)))
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        return {
            "status": "unsupported",
            "method": method,
            "supported": False,
            "best_effort": True,
            "candidate_file_count": 0,
            "advised_file_count": 0,
            "data_file_count": 0,
            "wal_file_count": 0,
            "advised_bytes": 0,
            "elapsed_seconds": max(0.0, clock() - started),
            "error": "POSIX_FADV_DONTNEED is unavailable on this platform",
        }

    patterns_by_directory: dict[str, list[tuple[str, str]]] = {}
    for directory, pattern, kind in _PAYLOAD_PATTERNS:
        patterns_by_directory.setdefault(directory, []).append((pattern, kind))
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    scanned_entries = 0
    candidate_files = 0
    advised_files = 0
    advised_bytes = 0
    files_by_kind = {"data": 0, "wal": 0}

    def result(status: str, error: str | None) -> dict[str, object]:
        return {
            "status": status,
            "method": method,
            "supported": True,
            "best_effort": True,
            "candidate_file_count": candidate_files,
            "advised_file_count": advised_files,
            "data_file_count": files_by_kind["data"],
            "wal_file_count": files_by_kind["wal"],
            "advised_bytes": advised_bytes,
            "scanned_entry_count": scanned_entries,
            "elapsed_seconds": max(0.0, clock() - started),
            "error": error,
        }

    try:
        for directory, patterns in patterns_by_directory.items():
            if deadline is not None and clock() >= deadline:
                return result(
                    "deadline_expired",
                    "cache preparation deadline expired during discovery",
                )
            directory_path = base / directory
            try:
                entries = os.scandir(directory_path)
            except FileNotFoundError:
                continue
            with entries:
                for entry in entries:
                    scanned_entries += 1
                    if scanned_entries > _MAX_DISCOVERY_ENTRIES:
                        return result(
                            "inventory_limit_exceeded",
                            (
                                "payload discovery exceeded its bounded "
                                f"{_MAX_DISCOVERY_ENTRIES}-entry inventory"
                            ),
                        )
                    if deadline is not None and clock() >= deadline:
                        return result(
                            "deadline_expired",
                            (
                                "cache preparation deadline expired during "
                                "discovery"
                            ),
                        )
                    kind = next(
                        (
                            candidate_kind
                            for pattern, candidate_kind in patterns
                            if fnmatchcase(entry.name, pattern)
                        ),
                        None,
                    )
                    if kind is None:
                        continue
                    candidate_files += 1
                    files_by_kind[kind] += 1
                    if deadline is not None and clock() >= deadline:
                        return result(
                            "deadline_expired",
                            "cache preparation deadline expired during advice",
                        )
                    descriptor = os.open(entry.path, open_flags)
                    try:
                        metadata = os.fstat(descriptor)
                        if not stat.S_ISREG(metadata.st_mode):
                            raise OSError("payload path is not a regular file")
                        os.posix_fadvise(
                            descriptor,
                            0,
                            0,
                            os.POSIX_FADV_DONTNEED,
                        )
                        advised_files += 1
                        advised_bytes += int(metadata.st_size)
                    finally:
                        os.close(descriptor)
    except OSError as error:
        return result("failed", f"{type(error).__name__}: {error}")
    if candidate_files == 0:
        return result("no_payload_files", "no ThrowDB payload files were found")
    return result("performed", None)


__all__ = ["prepare_cold_payload_cache"]
