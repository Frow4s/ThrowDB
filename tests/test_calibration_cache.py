from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

import throwdb._calibration_cache as calibration_cache


def test_discovery_bound_covers_maximum_format7_payload_geometry() -> None:
    maximum_segments = 1_000_000
    maximum_wals = 1_000_000
    native_uncatalogued_headroom = 65_536

    assert calibration_cache._MAX_DISCOVERY_ENTRIES >= (
        2 * maximum_segments
        + maximum_wals
        + native_uncatalogued_headroom
    )


def _install_supported_fadvise(
    monkeypatch: pytest.MonkeyPatch,
    implementation: object,
) -> None:
    monkeypatch.setattr(
        calibration_cache.os,
        "POSIX_FADV_DONTNEED",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        calibration_cache.os,
        "posix_fadvise",
        implementation,
        raising=False,
    )


def test_prepare_cold_payload_cache_advises_unique_payload_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    deltas = tmp_path / "deltas"
    segments = tmp_path / "segments"
    wal = tmp_path / "wal"
    data.mkdir()
    deltas.mkdir()
    segments.mkdir()
    wal.mkdir()
    (data / "shard-000001.dat").write_bytes(bytes(11))
    (data / "segment-00000001.dat").write_bytes(bytes(13))
    (deltas / "delta-00000001.dat").write_bytes(bytes(7))
    (segments / "segment-00000000000000000000000000000001.dat").write_bytes(
        bytes(17)
    )
    (wal / "active.wal").write_bytes(bytes(19))
    (segments / "segment-ignored.ids").write_bytes(bytes(23))
    (wal / "ignored.bin").write_bytes(bytes(29))

    # Exercise the set-based path de-duplication explicitly: the extra
    # pattern overlaps both existing data patterns.
    monkeypatch.setattr(
        calibration_cache,
        "_PAYLOAD_PATTERNS",
        (*calibration_cache._PAYLOAD_PATTERNS, ("data", "*.dat", "data")),
    )
    advised: list[tuple[int, int, int]] = []

    def advise(fd: int, offset: int, length: int, advice: int) -> None:
        assert advice == calibration_cache.os.POSIX_FADV_DONTNEED
        advised.append((os.fstat(fd).st_size, offset, length))

    _install_supported_fadvise(monkeypatch, advise)

    report = calibration_cache.prepare_cold_payload_cache(tmp_path)

    assert report["status"] == "performed"
    assert report["method"] == "posix_fadvise_dontneed"
    assert report["supported"] is True
    assert report["best_effort"] is True
    assert report["candidate_file_count"] == 5
    assert report["advised_file_count"] == 5
    assert report["data_file_count"] == 4
    assert report["wal_file_count"] == 1
    assert report["advised_bytes"] == 67
    assert report["error"] is None
    assert report["elapsed_seconds"] >= 0
    assert sorted(advised) == [
        (7, 0, 0),
        (11, 0, 0),
        (13, 0, 0),
        (17, 0, 0),
        (19, 0, 0),
    ]


@pytest.mark.parametrize("missing", ["posix_fadvise", "POSIX_FADV_DONTNEED"])
def test_prepare_cold_payload_cache_reports_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    _install_supported_fadvise(monkeypatch, lambda *args: None)
    monkeypatch.delattr(calibration_cache.os, missing)

    report = calibration_cache.prepare_cold_payload_cache(tmp_path)

    assert report["status"] == "unsupported"
    assert report["supported"] is False
    assert report["best_effort"] is True
    assert report["candidate_file_count"] == 0
    assert report["advised_file_count"] == 0
    assert report["data_file_count"] == 0
    assert report["wal_file_count"] == 0
    assert report["advised_bytes"] == 0
    assert "unavailable" in str(report["error"])
    assert report["elapsed_seconds"] >= 0


def test_prepare_cold_payload_cache_reports_fadvise_failure_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard-000001.dat").write_bytes(bytes(31))
    observed_fds: list[int] = []

    def fail(fd: int, offset: int, length: int, advice: int) -> None:
        del offset, length, advice
        observed_fds.append(fd)
        raise OSError(errno.EIO, "synthetic fadvise failure")

    _install_supported_fadvise(monkeypatch, fail)

    report = calibration_cache.prepare_cold_payload_cache(tmp_path)

    assert report["status"] == "failed"
    assert report["supported"] is True
    assert report["candidate_file_count"] == 1
    assert report["advised_file_count"] == 0
    assert report["data_file_count"] == 1
    assert report["wal_file_count"] == 0
    assert report["advised_bytes"] == 0
    assert "synthetic fadvise failure" in str(report["error"])
    assert len(observed_fds) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(observed_fds[0])
    assert closed.value.errno == errno.EBADF


def test_prepare_cold_payload_cache_never_follows_payload_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose O_NOFOLLOW")
    outside = tmp_path / "outside-payload.dat"
    outside.write_bytes(b"do not advise through this symlink")
    data = tmp_path / "store" / "data"
    data.mkdir(parents=True)
    (data / "shard-000001.dat").symlink_to(outside)
    advised: list[int] = []

    def advise(fd: int, offset: int, length: int, advice: int) -> None:
        del offset, length, advice
        advised.append(fd)

    _install_supported_fadvise(monkeypatch, advise)

    report = calibration_cache.prepare_cold_payload_cache(tmp_path / "store")

    assert report["status"] == "failed"
    assert report["supported"] is True
    assert report["candidate_file_count"] == 1
    assert report["advised_file_count"] == 0
    assert report["advised_bytes"] == 0
    assert advised == []
    assert outside.read_bytes() == b"do not advise through this symlink"


def test_prepare_cold_payload_cache_reports_no_payload_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "unrelated.bin").write_bytes(bytes(7))
    advised: list[int] = []
    _install_supported_fadvise(
        monkeypatch,
        lambda fd, offset, length, advice: advised.append(fd),
    )

    report = calibration_cache.prepare_cold_payload_cache(tmp_path)

    assert report["status"] == "no_payload_files"
    assert report["supported"] is True
    assert report["best_effort"] is True
    assert report["candidate_file_count"] == 0
    assert report["advised_file_count"] == 0
    assert report["data_file_count"] == 0
    assert report["wal_file_count"] == 0
    assert report["advised_bytes"] == 0
    assert "no ThrowDB payload files" in str(report["error"])
    assert advised == []


def test_prepare_cold_payload_cache_honors_discovery_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard-000001.dat").write_bytes(bytes(7))
    (data / "shard-000002.dat").write_bytes(bytes(11))
    advised: list[int] = []
    _install_supported_fadvise(
        monkeypatch,
        lambda fd, offset, length, advice: advised.append(fd),
    )
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls < 4 else 2.0

    report = calibration_cache.prepare_cold_payload_cache(
        tmp_path,
        deadline=1.0,
        clock=clock,
    )

    assert report["status"] == "deadline_expired"
    assert report["advised_file_count"] == 0
    assert report["scanned_entry_count"] == 1
    assert advised == []


def test_prepare_cold_payload_cache_bounds_inventory_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard-000001.dat").write_bytes(bytes(7))
    (data / "shard-000002.dat").write_bytes(bytes(11))
    advised: list[int] = []
    _install_supported_fadvise(
        monkeypatch,
        lambda fd, offset, length, advice: advised.append(fd),
    )
    monkeypatch.setattr(calibration_cache, "_MAX_DISCOVERY_ENTRIES", 1)

    report = calibration_cache.prepare_cold_payload_cache(tmp_path)

    assert report["status"] == "inventory_limit_exceeded"
    # Discovery is deliberately streaming: already validated payloads are
    # advised without retaining an unbounded path inventory in memory.
    assert report["advised_file_count"] == 1
    assert report["scanned_entry_count"] == 2
    assert len(advised) == 1
