from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from throwdb.cli import _build_parser, _reader_tuning, main
from throwdb.collection import Collection


def test_reader_cli_defaults_to_auto_instead_of_manual_defaults() -> None:
    parser = _build_parser()

    inspect = parser.parse_args(["inspect", "--storage", "/tmp/store"])
    assert _reader_tuning(inspect) == "auto"

    get = parser.parse_args(
        [
            "get",
            "--storage",
            "/tmp/store",
            "--ids",
            "/tmp/ids.npy",
            "--output",
            "/tmp/output.npy",
        ]
    )
    assert _reader_tuning(get) == "auto"


def test_reader_cli_builds_only_the_explicit_manual_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "inspect",
            "--storage",
            "/tmp/store",
            "--io-backend",
            "io_uring",
            "--io-uring-workers",
            "2",
            "--io-uring-queue-depth",
            "32",
            "--max-read-range-bytes",
            str(4 << 20),
        ]
    )

    assert _reader_tuning(args) == {
        "max_read_range_bytes": 4 << 20,
        "io_backend": "io_uring",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 32,
    }


def test_persisted_and_conservative_modes_cannot_mix_with_overrides() -> None:
    args = argparse.Namespace(
        tuning="persisted",
        io_workers=4,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _reader_tuning(args)


def test_tuning_commands_are_nested_and_set_requires_options() -> None:
    parser = _build_parser()
    show = parser.parse_args(
        ["tuning", "show", "--storage", "/tmp/store"]
    )
    assert show.tuning_command == "show"

    set_profile = parser.parse_args(
        [
            "tuning",
            "set",
            "--storage",
            "/tmp/store",
            "--io-workers",
            "8",
        ]
    )
    assert _reader_tuning(set_profile) == {"io_workers": 8}

    reset = parser.parse_args(
        ["tuning", "reset", "--storage", "/tmp/store", "--force"]
    )
    assert reset.tuning_command == "reset"
    assert reset.force is True


def test_tune_command_has_bounded_defaults_and_writes_optional_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["tune", "--storage", "/tmp/store"])
    assert parsed.max_seconds == 300.0
    assert parsed.max_read_gib == 8
    assert parsed.max_memory_mib == 512
    assert parsed.seed == 20_260_713
    assert parsed.dry_run is False

    report = {
        "format": "throwdb-reader-calibration",
        "version": 1,
        "status": "dry_run",
        "selection": {"winner": "sequential", "persisted": False},
    }
    observed: dict[str, object] = {}

    def fake_tune(path: Path, **kwargs: object) -> dict[str, object]:
        observed["path"] = path
        observed.update(kwargs)
        return report

    monkeypatch.setattr(Collection, "tune", fake_tune)
    output = tmp_path / "reports" / "tune.json"
    main(
        [
            "tune",
            "--storage",
            "/tmp/store",
            "--max-seconds",
            "12.5",
            "--max-read-gib",
            "3",
            "--max-memory-mib",
            "64",
            "--seed",
            "9",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert observed["path"] == Path("/tmp/store")
    assert observed["max_seconds"] == 12.5
    assert observed["max_read_bytes"] == 3 << 30
    assert observed["max_memory_bytes"] == 64 << 20
    assert observed["seed"] == 9
    assert observed["dry_run"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report


def test_compact_deltas_command_has_bounded_defaults_and_forwards_limits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    defaults = _build_parser().parse_args(
        ["compact-deltas", "--storage", "/tmp/store"]
    )
    assert defaults.max_runs == 8
    assert defaults.max_input_gib == 8
    assert defaults.max_input_entries == 1_000_000

    observed: dict[str, object] = {}
    report = {
        "compacted": True,
        "compacted_delta_run_count": 3,
        "delta_run_count": 1,
    }

    class FakeCollection:
        def __enter__(self) -> FakeCollection:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def compact_deltas(self, **kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return report

    def fake_open(
        cls: type[Collection],
        path: Path,
        **kwargs: object,
    ) -> FakeCollection:
        del cls
        observed["path"] = path
        observed["open"] = kwargs
        return FakeCollection()

    monkeypatch.setattr(Collection, "open", classmethod(fake_open))
    main(
        [
            "compact-deltas",
            "--storage",
            "/tmp/store",
            "--max-runs",
            "3",
            "--max-input-gib",
            "5",
            "--max-input-entries",
            "99",
        ]
    )

    assert observed == {
        "path": Path("/tmp/store"),
        "open": {"mode": "r+"},
        "max_runs": 3,
        "max_input_bytes": 5 << 30,
        "max_input_entries": 99,
    }
    assert json.loads(capsys.readouterr().out) == report


def test_collect_orphans_requires_explicit_mode_and_uses_lightweight_scan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(
        ["collect-orphans", "--storage", "/tmp/store", "--dry-run"]
    )
    assert parsed.dry_run is True
    assert parsed.min_age_seconds == 3_600.0

    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["collect-orphans", "--storage", "/tmp/store"])
    assert stopped.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(
            [
                "collect-orphans", "--storage", "/tmp/store",
                "--dry-run", "--execute",
            ]
        )
    assert stopped.value.code == 2
    capsys.readouterr()

    report = {
        "format": "throwdb-orphan-collection-plan",
        "version": 1,
        "dry_run": True,
        "candidate_file_count": 2,
        "reclaimable_bytes": 17,
        "entries": [],
    }
    observed: dict[str, object] = {}

    def fake_inspect_orphans(
        cls: type[Collection],
        path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del cls
        observed["path"] = path
        observed.update(kwargs)
        return report

    def unexpected_open(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("collect-orphans loaded the collection index")

    monkeypatch.setattr(
        Collection,
        "inspect_orphans",
        classmethod(fake_inspect_orphans),
    )
    monkeypatch.setattr(Collection, "open", unexpected_open)
    main(
        [
            "collect-orphans",
            "--storage",
            "/tmp/store",
            "--dry-run",
            "--min-age-seconds",
            "12.5",
        ]
    )

    assert observed == {
        "path": Path("/tmp/store"),
        "min_age_seconds": 12.5,
    }
    assert json.loads(capsys.readouterr().out) == report


def test_collect_orphans_execute_uses_private_lightweight_binding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "format": "throwdb-orphan-collection-report",
        "version": 1,
        "dry_run": False,
        "removed_file_count": 2,
        "removed_bytes": 17,
        "entries": [],
    }
    observed: dict[str, object] = {}

    def fake_reclaim(
        cls: type[Collection], path: Path, **kwargs: object
    ) -> dict[str, object]:
        del cls
        observed["path"] = path
        observed.update(kwargs)
        return report

    monkeypatch.setattr(
        Collection, "_reclaim_orphans_at", classmethod(fake_reclaim)
    )
    main(
        [
            "collect-orphans", "--storage", "/tmp/store", "--execute",
            "--min-age-seconds", "5", "--max-files", "7",
            "--max-bytes", "99",
        ]
    )
    assert observed == {
        "path": Path("/tmp/store"),
        "min_age_seconds": 5.0,
        "max_files": 7,
        "max_bytes": 99,
    }
    assert json.loads(capsys.readouterr().out) == report


def test_maintain_cli_requires_mode_and_forwards_bounded_policy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["maintain", "--storage", "/tmp/store"])
    capsys.readouterr()
    report = {"format": "throwdb-maintenance-report", "version": 1}
    observed: dict[str, object] = {}

    class FakeCollection:
        def __enter__(self) -> FakeCollection:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def maintain(self, **kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return report

    def fake_open(
        cls: type[Collection], path: Path, **kwargs: object
    ) -> FakeCollection:
        del cls
        observed["path"] = path
        observed["open"] = kwargs
        return FakeCollection()

    monkeypatch.setattr(Collection, "open", classmethod(fake_open))
    main(
        [
            "maintain", "--storage", "/tmp/store", "--execute",
            "--max-seconds", "0", "--max-actions", "2",
            "--rotate-wal-bytes", "10",
            "--materialize-max-wals", "3",
            "--materialize-max-input-bytes", "20",
            "--compact-min-runs", "4", "--compact-max-runs", "5",
            "--compact-max-input-bytes", "30",
            "--compact-max-input-entries", "40",
            "--orphan-min-age-seconds", "6",
            "--orphan-max-files", "7", "--orphan-max-bytes", "50",
        ]
    )
    assert observed == {
        "path": Path("/tmp/store"),
        "open": {"mode": "r+"},
        "dry_run": False,
        "max_seconds": 0.0,
        "max_actions": 2,
        "rotate_wal_bytes": 10,
        "materialize_max_wals": 3,
        "materialize_max_input_bytes": 20,
        "compact_min_runs": 4,
        "compact_max_runs": 5,
        "compact_max_input_bytes": 30,
        "compact_max_input_entries": 40,
        "orphan_min_age_seconds": 6.0,
        "orphan_max_files": 7,
        "orphan_max_bytes": 50,
    }
    assert json.loads(capsys.readouterr().out) == report


def test_tuning_cli_reports_user_errors_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["tuning", "set", "--storage", "/tmp/store"])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "requires at least one reader option" in error
    assert "Traceback" not in error


def test_tune_cli_reports_native_lease_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_tune(path: Path, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        raise RuntimeError("another writer already holds the collection lease")

    monkeypatch.setattr(Collection, "tune", fail_tune)
    with pytest.raises(SystemExit) as stopped:
        main(["tune", "--storage", "/tmp/store"])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "another writer already holds" in error
    assert "Traceback" not in error


def test_tune_cli_reports_serialization_errors_after_profile_was_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "selection": {"persisted": True},
        "not_json_serializable": object(),
    }

    def fake_tune(path: Path, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return report

    monkeypatch.setattr(Collection, "tune", fake_tune)
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "tune",
                "--storage",
                "/tmp/store",
                "--output",
                str(output),
            ]
        )

    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "the winning profile was saved" in captured.err
    assert "failed to serialize the JSON calibration report" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_tune_cli_reports_atomic_output_errors_after_profile_was_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "status": "complete",
        "selection": {"winner": "thread_pool_w8", "persisted": True},
    }

    def fake_tune(path: Path, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return report

    monkeypatch.setattr(Collection, "tune", fake_tune)
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("occupied", encoding="utf-8")
    output = not_a_directory / "report.json"
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "tune",
                "--storage",
                "/tmp/store",
                "--output",
                str(output),
            ]
        )

    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "the winning profile was saved" in captured.err
    assert "failed to write the JSON calibration report" in captured.err
    assert str(output) in captured.err
    assert "Traceback" not in captured.err
