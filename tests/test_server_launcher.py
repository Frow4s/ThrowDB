from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

import throwdb
import throwdb.server as server_module
from throwdb._tuning import READER_DEFAULTS, TuningRegistryError


def _build_store(path: Path) -> None:
    embeddings = np.arange(12, dtype=np.float16).reshape(3, 4)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=4, dtype="float16"),
        format_version=7,
    ) as builder:
        builder.add([1, 2, 3], embeddings)


def _fake_native(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELFtest-native")
    path.chmod(0o755)
    return path


def _args(store: Path, native: Path, *extra: str) -> argparse.Namespace:
    return server_module._build_parser().parse_args(  # type: ignore[attr-defined]
        [
            "--storage",
            str(store),
            "--native-binary",
            str(native),
            *extra,
        ]
    )


def test_auto_falls_back_without_creating_or_tuning_registry(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    resolved = server_module.resolve_server_tuning(store, mode="auto")

    assert resolved.mode == "auto"
    assert resolved.source == "fallback"
    assert resolved.status == "registry_missing"
    assert resolved.reuse_level == "fallback"
    assert resolved.explicit_overrides is False
    assert resolved.options == READER_DEFAULTS
    assert not (store / "tuning").exists()


def test_persisted_mode_fails_closed_when_profile_is_missing(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    with pytest.raises(TuningRegistryError, match="no reader profiles"):
        server_module.resolve_server_tuning(store, mode="persisted")


def test_partial_reader_override_is_manual_and_bypasses_registry(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    resolved = server_module.resolve_server_tuning(
        store,
        mode="auto",
        overrides={"io_workers": 3},
    )

    assert resolved.mode == "manual"
    assert resolved.source == "manual"
    assert resolved.status == "explicit"
    assert resolved.reuse_level == "manual"
    assert resolved.explicit_overrides is True
    assert resolved.options == {**READER_DEFAULTS, "io_workers": 3}
    with pytest.raises(ValueError, match="cannot be combined"):
        server_module.resolve_server_tuning(
            store,
            mode="persisted",
            overrides={"io_workers": 3},
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        server_module.resolve_server_tuning(
            store,
            mode="conservative",
            overrides={"io_workers": 3},
        )


def test_selected_profile_metadata_and_options_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {**READER_DEFAULTS, "io_workers": 6}
    monkeypatch.setattr(
        server_module,
        "_layout_identity",
        lambda path: ({"path": str(path)}, 8),
    )
    monkeypatch.setattr(
        server_module,
        "_resolve_tuning",
        lambda path, mode: (
            options,
            {
                "mode": mode,
                "source": "persisted",
                "status": "selected",
                "profile_reuse_level": "compatible",
            },
        ),
    )

    resolved = server_module.resolve_server_tuning(
        tmp_path / "store",
        mode="auto",
    )

    assert resolved.options == options
    assert resolved.source == "persisted"
    assert resolved.status == "selected"
    assert resolved.reuse_level == "compatible"


def test_launch_plan_passes_complete_effective_configuration(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    plan = server_module.build_server_launch_plan(
        _args(
            store,
            native,
            "--io-workers",
            "4",
            "--max-gap-bytes",
            "16",
            "--max-concurrent-calls",
            "7",
            "--max-concurrent-scans",
            "2",
            "--max-inflight-mib",
            "384",
            "--max-scan-inflight-mib",
            "96",
            "--max-grpc-memory-mib",
            "640",
            "--scan-max-duration-ms",
            "7200000",
            "--scan-idle-timeout-ms",
            "120000",
            "--shutdown-grace-ms",
            "45000",
            "--shutdown-hard-timeout-ms",
            "90000",
            "--writable",
            "--allow-insecure-non-loopback",
            "--tls-cert",
            "server.pem",
            "--tls-key",
            "server.key",
            "--tls-client-ca",
            "clients.pem",
            "--mutation-disk-reserve-mib",
            "2048",
            "--idempotency-journal-max-mib",
            "8192",
            "--idempotency-journal-max-operations",
            "2000000",
            "--max-wal-backlog-mib",
            "12288",
            "--max-wal-backlog-operations",
            "3000000",
            "--max-overlay-entries",
            "2500000",
            "--max-overlay-memory-mib",
            "192",
            "--maintenance-interval-seconds",
            "30",
            "--idempotency-checkpoint-trigger-percent",
            "80",
            "--rotate-wal-mib",
            "512",
            "--materialize-max-input-entries",
            "1500000",
            "--materialize-max-memory-mib",
            "768",
            "--cleaner-min-interval-seconds",
            "900",
            "--cleaner-max-candidate-segments",
            "4",
            "--cleaner-max-memory-mib",
            "512",
            "--cleaner-min-garbage-ratio-ppm",
            "400000",
            "--no-auto-maintenance",
            "--no-idempotency-checkpoint",
            "--no-auto-clean",
            "--no-orphan-reclaim",
        )
    )

    assert plan.executable == native.resolve()
    assert plan.reader == {
        **READER_DEFAULTS,
        "io_workers": 4,
        "max_gap_bytes": 16,
    }
    assert plan.tuning == {
        "mode": "manual",
        "source": "manual",
        "status": "explicit",
        "reuse_level": "manual",
        "explicit_overrides": True,
    }
    assert plan.argv[0] == str(native.resolve())
    assert plan.argv.count("--io-workers") == 1
    assert plan.argv[plan.argv.index("--io-workers") + 1] == "4"
    assert plan.argv[plan.argv.index("--max-concurrent-calls") + 1] == "7"
    assert plan.argv[plan.argv.index("--max-concurrent-scans") + 1] == "2"
    assert plan.argv[plan.argv.index("--max-inflight-mib") + 1] == "384"
    assert plan.argv[plan.argv.index("--max-scan-inflight-mib") + 1] == "96"
    assert plan.argv[plan.argv.index("--max-grpc-memory-mib") + 1] == "640"
    assert plan.argv[plan.argv.index("--scan-max-duration-ms") + 1] == "7200000"
    assert plan.argv[plan.argv.index("--scan-idle-timeout-ms") + 1] == "120000"
    assert plan.argv[plan.argv.index("--shutdown-grace-ms") + 1] == "45000"
    assert (
        plan.argv[plan.argv.index("--shutdown-hard-timeout-ms") + 1]
        == "90000"
    )
    assert "--writable" in plan.argv
    assert "--allow-insecure-non-loopback" in plan.argv
    assert plan.argv[plan.argv.index("--tls-cert") + 1] == "server.pem"
    assert plan.argv[plan.argv.index("--tls-key") + 1] == "server.key"
    assert plan.argv[plan.argv.index("--tls-client-ca") + 1] == "clients.pem"
    assert (
        plan.argv[plan.argv.index("--mutation-disk-reserve-mib") + 1]
        == "2048"
    )
    assert (
        plan.argv[plan.argv.index("--idempotency-journal-max-mib") + 1]
        == "8192"
    )
    assert (
        plan.argv[plan.argv.index("--idempotency-journal-max-operations") + 1]
        == "2000000"
    )
    assert plan.argv[plan.argv.index("--max-wal-backlog-mib") + 1] == "12288"
    assert (
        plan.argv[plan.argv.index("--max-wal-backlog-operations") + 1]
        == "3000000"
    )
    assert plan.argv[plan.argv.index("--max-overlay-entries") + 1] == "2500000"
    assert plan.argv[plan.argv.index("--max-overlay-memory-mib") + 1] == "192"
    assert (
        plan.argv[plan.argv.index("--maintenance-interval-seconds") + 1]
        == "30"
    )
    assert (
        plan.argv[
            plan.argv.index("--idempotency-checkpoint-trigger-percent") + 1
        ]
        == "80"
    )
    assert plan.argv[plan.argv.index("--cleaner-min-interval-seconds") + 1] == "900"
    assert plan.argv[plan.argv.index("--cleaner-max-candidate-segments") + 1] == "4"
    assert plan.argv[plan.argv.index("--cleaner-max-memory-mib") + 1] == "512"
    assert (
        plan.argv[plan.argv.index("--cleaner-min-garbage-ratio-ppm") + 1]
        == "400000"
    )
    assert "--no-idempotency-checkpoint" in plan.argv
    assert "--no-auto-clean" in plan.argv
    assert plan.argv[plan.argv.index("--rotate-wal-mib") + 1] == "512"
    assert (
        plan.argv[plan.argv.index("--materialize-max-input-entries") + 1]
        == "1500000"
    )
    assert (
        plan.argv[plan.argv.index("--materialize-max-memory-mib") + 1]
        == "768"
    )
    assert "--no-auto-maintenance" in plan.argv
    assert "--no-orphan-reclaim" in plan.argv
    assert plan.argv[plan.argv.index("--resolved-tuning-mode") + 1] == "manual"


def test_launcher_defaults_to_read_only_and_rejects_conflicting_modes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    plan = server_module.build_server_launch_plan(_args(store, native))

    assert "--writable" not in plan.argv
    assert "--read-only" in plan.argv
    with pytest.raises(SystemExit):
        _args(store, native, "--writable", "--read-only")


def test_launcher_forwards_explicit_bounded_read_logging(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    logs = tmp_path / "workload-logs"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    plan = server_module.build_server_launch_plan(
        _args(
            store,
            native,
            "--read-log-dir",
            str(logs),
            "--read-log-sample-ppm",
            "25000",
            "--read-log-max-ids-per-event",
            "8192",
            "--read-log-buffer-mib",
            "96",
            "--read-log-segment-mib",
            "128",
            "--read-log-rotate-seconds",
            "900",
            "--read-log-retention-mib",
            "2048",
            "--read-log-retention-seconds",
            "172800",
            "--read-log-min-free-mib",
            "4096",
            "--read-log-max-write-mib-per-second",
            "32",
            "--read-log-max-retained-files",
            "2048",
        )
    )

    expected = {
        "--read-log-dir": str(logs),
        "--read-log-sample-ppm": "25000",
        "--read-log-max-ids-per-event": "8192",
        "--read-log-buffer-mib": "96",
        "--read-log-segment-mib": "128",
        "--read-log-rotate-seconds": "900",
        "--read-log-retention-mib": "2048",
        "--read-log-retention-seconds": "172800",
        "--read-log-min-free-mib": "4096",
        "--read-log-max-write-mib-per-second": "32",
        "--read-log-max-retained-files": "2048",
    }
    for option, value in expected.items():
        assert plan.argv.count(option) == 1
        assert plan.argv[plan.argv.index(option) + 1] == value


def test_launcher_keeps_read_logging_disabled_without_directory(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    plan = server_module.build_server_launch_plan(_args(store, native))

    assert not any(argument.startswith("--read-log-") for argument in plan.argv)
    for option, value in (
        ("--read-log-sample-ppm", "1000"),
        ("--read-log-max-write-mib-per-second", "32"),
        ("--read-log-max-retained-files", "2048"),
    ):
        with pytest.raises(
            server_module.ServerLaunchError,
            match="require --read-log-dir",
        ):
            server_module.build_server_launch_plan(
                _args(store, native, option, value)
            )


def test_launcher_rejects_unsafe_or_unbounded_read_log_geometry(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    for unsafe in (store, store / "telemetry", store.parent):
        with pytest.raises(
            server_module.ServerLaunchError,
            match="dedicated directory disjoint",
        ):
            server_module.build_server_launch_plan(
                _args(
                    store,
                    native,
                    "--read-log-dir",
                    str(unsafe),
                )
            )
    with pytest.raises(
        server_module.ServerLaunchError,
        match="hold two maximum-sized",
    ):
        server_module.build_server_launch_plan(
            _args(
                store,
                native,
                "--read-log-dir",
                str(tmp_path / "logs"),
                "--read-log-max-ids-per-event",
                "65536",
                "--read-log-buffer-mib",
                "1",
            )
        )
    with pytest.raises(
        server_module.ServerLaunchError,
        match="retention-mib.*segment-mib",
    ):
        server_module.build_server_launch_plan(
            _args(
                store,
                native,
                "--read-log-dir",
                str(tmp_path / "logs"),
                "--read-log-segment-mib",
                "512",
                "--read-log-retention-mib",
                "256",
            )
        )
    with pytest.raises(
        server_module.ServerLaunchError,
        match="signed 64-bit",
    ):
        server_module.build_server_launch_plan(
            _args(
                store,
                native,
                "--read-log-dir",
                str(tmp_path / "logs"),
                "--read-log-rotate-seconds",
                str(1 << 63),
            )
        )
    with pytest.raises(
        server_module.ServerLaunchError,
        match="uint64 MiB",
    ):
        server_module.build_server_launch_plan(
            _args(
                store,
                native,
                "--read-log-dir",
                str(tmp_path / "logs"),
                "--read-log-buffer-mib",
                str(1 << 44),
            )
        )
    with pytest.raises(
        server_module.ServerLaunchError,
        match="retention-seconds.*uint64",
    ):
        server_module.build_server_launch_plan(
            _args(
                store,
                native,
                "--read-log-dir",
                str(tmp_path / "logs"),
                "--read-log-retention-seconds",
                str(1 << 64),
            )
        )


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            ("--max-concurrent-calls", "2", "--max-concurrent-scans", "2"),
            "max_concurrent_scans < max_concurrent_calls",
        ),
        (
            ("--max-worker-threads", "4", "--max-concurrent-calls", "4"),
            "max_concurrent_calls < max_worker_threads",
        ),
        (
            ("--shutdown-grace-ms", "60000", "--shutdown-hard-timeout-ms", "1"),
            "hard-timeout-ms.*grace-ms",
        ),
    ],
)
def test_launcher_reserves_capacity_and_bounds_shutdown(
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    with pytest.raises(server_module.ServerLaunchError, match=message):
        server_module.build_server_launch_plan(_args(store, native, *extra))


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--listen", "localhost:50051"), "non-loopback.*mTLS"),
        (("--listen", "[::1]suffix:50051"), "strict HOST:PORT"),
        (("--listen", "127.0.0.1:70000"), "strict HOST:PORT"),
        (("--tls-cert", "server.pem"), "--tls-cert.*--tls-key"),
    ],
)
def test_launcher_dry_run_validates_transport_security(
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    with pytest.raises(server_module.ServerLaunchError, match=message):
        server_module.build_server_launch_plan(_args(store, native, *extra))

    allowed = server_module.build_server_launch_plan(
        _args(
            store,
            native,
            "--listen",
            "service.internal:50051",
            "--tls-cert",
            "server.pem",
            "--tls-key",
            "server.key",
            "--tls-client-ca",
            "clients.pem",
        )
    )
    assert "--tls-client-ca" in allowed.argv


def test_packaged_native_lookup_uses_distribution_not_source_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _fake_native(
        tmp_path / "site-packages" / "throwdb" / "libexec" / "_throwdbd_native"
    )

    class Distribution:
        def locate_file(self, relative: str) -> Path:
            return tmp_path / "site-packages" / relative

    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    monkeypatch.setattr(
        server_module,
        "__file__",
        str(tmp_path / "source" / "python" / "throwdb" / "server.py"),
    )

    assert server_module.find_native_server() == installed.resolve()


def test_launcher_rejects_scripts_and_recursive_console_entry(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "throwdbd"
    launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    launcher.chmod(0o755)

    with pytest.raises(server_module.ServerLaunchError):
        server_module.find_native_server(launcher)


def test_execve_receives_no_shell_and_recursion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")
    observed: dict[str, object] = {}

    def fake_execve(path: str, argv: list[str], environment: dict[str, str]) -> None:
        observed.update(path=path, argv=argv, environment=environment)

    monkeypatch.setattr(os, "execve", fake_execve)
    server_module.run_server(_args(store, native, "--tuning", "conservative"))

    assert observed["path"] == str(native.resolve())
    assert observed["argv"][0] == str(native.resolve())  # type: ignore[index]
    assert observed["environment"][  # type: ignore[index]
        "THROWDB_NATIVE_SERVER_EXEC"
    ] == "1"


def test_launcher_refuses_recursive_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    native = _fake_native(tmp_path / "bin" / "throwdbd")
    monkeypatch.setenv("THROWDB_NATIVE_SERVER_EXEC", "1")

    with pytest.raises(
        server_module.ServerLaunchError,
        match="recursive ThrowDB server launcher",
    ):
        server_module.run_server(_args(store, native))


def test_public_main_reports_startup_errors_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    native = _fake_native(tmp_path / "bin" / "throwdbd")

    with pytest.raises(SystemExit) as raised:
        server_module.main(
            [
                "--storage",
                str(tmp_path / "missing-store"),
                "--native-binary",
                str(native),
            ]
        )

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "cannot prepare native server" in stderr
    assert "Traceback" not in stderr
