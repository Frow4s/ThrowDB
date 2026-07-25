"""One-shot Python control plane for the native single-collection server.

The launcher resolves advisory reader tuning, validates the complete effective
configuration, and then replaces itself with the packaged C++ daemon. Python
does not remain in the request/data path and startup never runs calibration.
"""

from __future__ import annotations

import argparse
import ipaddress
import importlib.metadata
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ._tuning import (
    CANONICAL_READER_OPTIONS,
    READER_DEFAULTS,
    TuningRegistryError,
    normalize_reader_options,
)
from .collection import _layout_identity, _resolve_tuning


_LAUNCH_MARKER: Final = "THROWDB_NATIVE_SERVER_EXEC"
_NATIVE_ENVIRONMENT: Final = "THROWDB_NATIVE_SERVER_BINARY"
_FORMAT: Final = "throwdb-server-launch-plan"
_VERSION: Final = 1
_DEFAULT_MAX_WORKER_THREADS: Final = 64
_DEFAULT_MAX_CONCURRENT_CALLS: Final = 32
_DEFAULT_MAX_CONCURRENT_SCANS: Final = 2
_DEFAULT_SHUTDOWN_GRACE_MS: Final = 30_000
_DEFAULT_SHUTDOWN_HARD_TIMEOUT_MS: Final = 60_000
_DEFAULT_READ_LOG_MAX_IDS_PER_EVENT: Final = 4_096
_DEFAULT_READ_LOG_BUFFER_MIB: Final = 64
_DEFAULT_READ_LOG_SEGMENT_MIB: Final = 256
_DEFAULT_READ_LOG_RETENTION_MIB: Final = 10 * 1_024
_DEFAULT_READ_LOG_MAX_WRITE_MIB_PER_SECOND: Final = 16
_DEFAULT_READ_LOG_MAX_RETAINED_FILES: Final = 4_096
_MAX_READ_LOG_IDS_PER_EVENT: Final = 65_536
_MAX_READ_LOG_RETAINED_FILES: Final = 65_533
_MAX_READ_LOG_ROTATE_SECONDS: Final = (1 << 63) - 1
_MAX_UINT64: Final = (1 << 64) - 1
_MAX_UINT64_MIB: Final = _MAX_UINT64 >> 20
_READ_LOG_HEADER_BYTES: Final = 4_096
_READ_LOG_RECORD_HEADER_BYTES: Final = 256
_READ_LOG_TERMINAL_CHECKPOINT_BYTES: Final = 256
_READ_LOG_ARGUMENTS: Final = (
    "read_log_sample_ppm",
    "read_log_max_ids_per_event",
    "read_log_buffer_mib",
    "read_log_segment_mib",
    "read_log_rotate_seconds",
    "read_log_retention_mib",
    "read_log_retention_seconds",
    "read_log_min_free_mib",
    "read_log_max_write_mib_per_second",
    "read_log_max_retained_files",
)
_NATIVE_MAGICS: Final = frozenset(
    {
        b"\x7fELF",
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }
)

_READER_ARGUMENTS: Final = (
    "max_read_range_bytes",
    "max_gap_bytes",
    "io_workers",
    "io_queue_capacity",
    "io_ranges_per_task",
    "io_backend",
    "io_uring_queue_depth",
    "io_uring_workers",
)

_NATIVE_VALUE_ARGUMENTS: Final = (
    ("max_worker_threads", "--max-worker-threads"),
    ("max_concurrent_calls", "--max-concurrent-calls"),
    ("max_concurrent_scans", "--max-concurrent-scans"),
    ("max_inflight_mib", "--max-inflight-mib"),
    ("max_scan_inflight_mib", "--max-scan-inflight-mib"),
    ("max_grpc_memory_mib", "--max-grpc-memory-mib"),
    ("max_request_mib", "--max-request-mib"),
    ("max_response_mib", "--max-response-mib"),
    ("max_ids_per_request", "--max-ids-per-request"),
    ("max_scan_batch_rows", "--max-scan-batch-rows"),
    ("scan_max_duration_ms", "--scan-max-duration-ms"),
    ("scan_idle_timeout_ms", "--scan-idle-timeout-ms"),
    ("shutdown_grace_ms", "--shutdown-grace-ms"),
    ("shutdown_hard_timeout_ms", "--shutdown-hard-timeout-ms"),
    ("keepalive_time_ms", "--keepalive-time-ms"),
    ("keepalive_timeout_ms", "--keepalive-timeout-ms"),
    ("minimum_ping_interval_ms", "--minimum-ping-interval-ms"),
    ("mutation_disk_reserve_mib", "--mutation-disk-reserve-mib"),
    ("idempotency_journal_max_mib", "--idempotency-journal-max-mib"),
    (
        "idempotency_journal_max_operations",
        "--idempotency-journal-max-operations",
    ),
    ("max_wal_backlog_mib", "--max-wal-backlog-mib"),
    ("max_wal_backlog_operations", "--max-wal-backlog-operations"),
    ("max_overlay_entries", "--max-overlay-entries"),
    ("max_overlay_memory_mib", "--max-overlay-memory-mib"),
    ("maintenance_interval_seconds", "--maintenance-interval-seconds"),
    ("maintenance_max_actions", "--maintenance-max-actions"),
    (
        "idempotency_checkpoint_trigger_percent",
        "--idempotency-checkpoint-trigger-percent",
    ),
    ("rotate_wal_mib", "--rotate-wal-mib"),
    ("rotate_wal_transactions", "--rotate-wal-transactions"),
    ("materialize_max_wals", "--materialize-max-wals"),
    ("materialize_max_input_mib", "--materialize-max-input-mib"),
    (
        "materialize_max_input_entries",
        "--materialize-max-input-entries",
    ),
    ("materialize_max_memory_mib", "--materialize-max-memory-mib"),
    ("fold_min_runs", "--fold-min-runs"),
    ("fold_max_runs", "--fold-max-runs"),
    ("fold_max_input_mib", "--fold-max-input-mib"),
    ("fold_max_input_entries", "--fold-max-input-entries"),
    ("cleaner_min_interval_seconds", "--cleaner-min-interval-seconds"),
    (
        "cleaner_max_candidate_segments",
        "--cleaner-max-candidate-segments",
    ),
    ("cleaner_max_sidecar_scan_mib", "--cleaner-max-sidecar-scan-mib"),
    ("cleaner_max_source_mib", "--cleaner-max-source-mib"),
    ("cleaner_max_output_mib", "--cleaner-max-output-mib"),
    ("cleaner_max_memory_mib", "--cleaner-max-memory-mib"),
    ("cleaner_min_garbage_rows", "--cleaner-min-garbage-rows"),
    (
        "cleaner_min_reclaimable_mib",
        "--cleaner-min-reclaimable-mib",
    ),
    (
        "cleaner_min_garbage_ratio_ppm",
        "--cleaner-min-garbage-ratio-ppm",
    ),
    ("orphan_min_age_seconds", "--orphan-min-age-seconds"),
    ("orphan_max_files", "--orphan-max-files"),
    ("orphan_max_mib", "--orphan-max-mib"),
    ("read_log_dir", "--read-log-dir"),
    ("read_log_sample_ppm", "--read-log-sample-ppm"),
    ("read_log_max_ids_per_event", "--read-log-max-ids-per-event"),
    ("read_log_buffer_mib", "--read-log-buffer-mib"),
    ("read_log_segment_mib", "--read-log-segment-mib"),
    ("read_log_rotate_seconds", "--read-log-rotate-seconds"),
    ("read_log_retention_mib", "--read-log-retention-mib"),
    ("read_log_retention_seconds", "--read-log-retention-seconds"),
    ("read_log_min_free_mib", "--read-log-min-free-mib"),
    (
        "read_log_max_write_mib_per_second",
        "--read-log-max-write-mib-per-second",
    ),
    ("read_log_max_retained_files", "--read-log-max-retained-files"),
)


class ServerLaunchError(ValueError):
    """The native daemon cannot be resolved or launched safely."""


@dataclass(frozen=True, slots=True)
class ServerTuning:
    options: dict[str, object]
    mode: str
    source: str
    status: str
    reuse_level: str
    explicit_overrides: bool

    def diagnostics(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source": self.source,
            "status": self.status,
            "reuse_level": self.reuse_level,
            "explicit_overrides": self.explicit_overrides,
        }


@dataclass(frozen=True, slots=True)
class ServerLaunchPlan:
    executable: Path
    argv: tuple[str, ...]
    reader: dict[str, object]
    tuning: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "native_binary": str(self.executable),
            "argv": list(self.argv),
            "reader": dict(self.reader),
            "tuning": dict(self.tuning),
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _read_log_sample_ppm(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 1_000_000:
        raise argparse.ArgumentTypeError(
            "value must be between 1 and 1000000"
        )
    return parsed


def _read_log_max_ids(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > _MAX_READ_LOG_IDS_PER_EVENT:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {_MAX_READ_LOG_IDS_PER_EVENT}"
        )
    return parsed


def _native_server_candidates() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1] if len(package.parents) > 1 else package
    installed: list[Path] = []
    try:
        distribution = importlib.metadata.distribution("throwdb")
        installed.append(
            Path(
                distribution.locate_file(
                    "throwdb/libexec/_throwdbd_native"
                )
            )
        )
    except importlib.metadata.PackageNotFoundError:
        pass
    return tuple(installed) + (
        package / "libexec" / "_throwdbd_native",
        repository / "build" / "server-release" / "throwdbd",
        repository / "build-grpc-dev" / "throwdbd",
        repository / "build-grpc" / "throwdbd",
    )


def find_native_server(explicit: Path | None = None) -> Path:
    """Find only the internal native daemon, never the public launcher."""

    configured = explicit
    if configured is None:
        environment = os.environ.get(_NATIVE_ENVIRONMENT)
        configured = Path(environment) if environment else None
    candidates = (
        (configured,) if configured is not None else _native_server_candidates()
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            continue
        try:
            with resolved.open("rb") as executable:
                magic = executable.read(4)
        except OSError:
            continue
        if magic in _NATIVE_MAGICS:
            return resolved
        if configured is not None:
            raise ServerLaunchError(
                "native server override is not an ELF/Mach-O executable; "
                "refusing a script or recursive launcher"
            )
    if configured is not None:
        raise ServerLaunchError(
            f"native ThrowDB server is not executable: {configured}"
        )
    raise ServerLaunchError(
        "packaged native ThrowDB server was not found; reinstall a server "
        "wheel or pass --native-binary for a development build"
    )


def resolve_server_tuning(
    storage: Path,
    *,
    mode: str = "auto",
    overrides: Mapping[str, object] | None = None,
) -> ServerTuning:
    """Resolve one startup profile without running or scheduling tuning."""

    if mode not in {"auto", "persisted", "conservative"}:
        raise ValueError(
            "tuning must be 'auto', 'persisted', or 'conservative'"
        )
    manual = dict(overrides or {})
    _, record_size = _layout_identity(storage)
    if manual:
        if mode != "auto":
            raise ValueError(
                "explicit --io-* or range options cannot be combined with "
                f"--tuning {mode}"
            )
        options = normalize_reader_options(
            manual,
            record_size=record_size,
        )
        return ServerTuning(
            options=options,
            mode="manual",
            source="manual",
            status="explicit",
            reuse_level="manual",
            explicit_overrides=True,
        )

    selected, profile = _resolve_tuning(storage, mode)
    if selected:
        raw_options = selected
    else:
        raw_options = {
            name: profile.get(name, READER_DEFAULTS[name])
            for name in CANONICAL_READER_OPTIONS
        }
    options = normalize_reader_options(
        raw_options,
        persisted=True,
        record_size=record_size,
    )
    return ServerTuning(
        options=options,
        mode=str(profile["mode"]),
        source=str(profile["source"]),
        status=str(profile["status"]),
        reuse_level=str(
            profile.get(
                "profile_reuse_level",
                profile.get("match_level", "fallback"),
            )
        ),
        explicit_overrides=False,
    )


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--listen", default="127.0.0.1:0")
    parser.add_argument(
        "--tuning",
        choices=("auto", "persisted", "conservative"),
        default="auto",
        help=(
            "select a compatible saved profile, require one, or bypass the "
            "registry; startup never calibrates"
        ),
    )
    access = parser.add_mutually_exclusive_group()
    access.add_argument(
        "--writable",
        action="store_true",
        help="enable insert/update/upsert/delete; the default is read-only",
    )
    access.add_argument(
        "--read-only",
        action="store_true",
        help="explicitly select the safe default access mode",
    )
    parser.add_argument("--allow-insecure-non-loopback", action="store_true")
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--tls-client-ca", type=Path)
    parser.add_argument("--no-durable-idempotency", action="store_true")
    parser.add_argument(
        "--permit-keepalive-without-calls",
        action="store_true",
    )
    parser.add_argument("--max-worker-threads", type=_positive_int)
    parser.add_argument("--max-concurrent-calls", type=_positive_int)
    parser.add_argument("--max-concurrent-scans", type=_positive_int)
    parser.add_argument("--max-inflight-mib", type=_positive_int)
    parser.add_argument("--max-scan-inflight-mib", type=_positive_int)
    parser.add_argument("--max-grpc-memory-mib", type=_positive_int)
    parser.add_argument("--max-request-mib", type=_positive_int)
    parser.add_argument("--max-response-mib", type=_positive_int)
    parser.add_argument("--max-ids-per-request", type=_positive_int)
    parser.add_argument("--max-scan-batch-rows", type=_positive_int)
    parser.add_argument("--scan-max-duration-ms", type=_positive_int)
    parser.add_argument("--scan-idle-timeout-ms", type=_positive_int)
    parser.add_argument("--shutdown-grace-ms", type=_positive_int)
    parser.add_argument("--shutdown-hard-timeout-ms", type=_positive_int)
    parser.add_argument("--keepalive-time-ms", type=_positive_int)
    parser.add_argument("--keepalive-timeout-ms", type=_positive_int)
    parser.add_argument("--minimum-ping-interval-ms", type=_positive_int)
    parser.add_argument("--mutation-disk-reserve-mib", type=_nonnegative_int)
    parser.add_argument("--idempotency-journal-max-mib", type=_positive_int)
    parser.add_argument(
        "--idempotency-journal-max-operations",
        type=_positive_int,
    )
    parser.add_argument("--max-wal-backlog-mib", type=_positive_int)
    parser.add_argument(
        "--max-wal-backlog-operations", type=_positive_int
    )
    parser.add_argument("--max-overlay-entries", type=_positive_int)
    parser.add_argument("--max-overlay-memory-mib", type=_positive_int)
    parser.add_argument("--maintenance-interval-seconds", type=_positive_int)
    parser.add_argument("--maintenance-max-actions", type=_nonnegative_int)
    parser.add_argument(
        "--idempotency-checkpoint-trigger-percent", type=_positive_int
    )
    parser.add_argument("--rotate-wal-mib", type=_positive_int)
    parser.add_argument("--rotate-wal-transactions", type=_positive_int)
    parser.add_argument("--materialize-max-wals", type=_positive_int)
    parser.add_argument("--materialize-max-input-mib", type=_positive_int)
    parser.add_argument(
        "--materialize-max-input-entries", type=_positive_int
    )
    parser.add_argument("--materialize-max-memory-mib", type=_positive_int)
    parser.add_argument("--fold-min-runs", type=_positive_int)
    parser.add_argument("--fold-max-runs", type=_positive_int)
    parser.add_argument("--fold-max-input-mib", type=_positive_int)
    parser.add_argument("--fold-max-input-entries", type=_positive_int)
    parser.add_argument(
        "--cleaner-min-interval-seconds", type=_nonnegative_int
    )
    parser.add_argument("--cleaner-max-candidate-segments", type=_positive_int)
    parser.add_argument("--cleaner-max-sidecar-scan-mib", type=_positive_int)
    parser.add_argument("--cleaner-max-source-mib", type=_positive_int)
    parser.add_argument("--cleaner-max-output-mib", type=_positive_int)
    parser.add_argument("--cleaner-max-memory-mib", type=_positive_int)
    parser.add_argument("--cleaner-min-garbage-rows", type=_nonnegative_int)
    parser.add_argument("--cleaner-min-reclaimable-mib", type=_nonnegative_int)
    parser.add_argument(
        "--cleaner-min-garbage-ratio-ppm", type=_nonnegative_int
    )
    parser.add_argument("--orphan-min-age-seconds", type=_nonnegative_int)
    parser.add_argument("--orphan-max-files", type=_positive_int)
    parser.add_argument("--orphan-max-mib", type=_positive_int)
    parser.add_argument(
        "--read-log-dir",
        type=Path,
        help=(
            "enable bounded point-read workload logging in this dedicated "
            "directory; logging is disabled when omitted"
        ),
    )
    parser.add_argument(
        "--read-log-sample-ppm",
        type=_read_log_sample_ppm,
        help="sample successful point-read batches in parts per million",
    )
    parser.add_argument(
        "--read-log-max-ids-per-event",
        type=_read_log_max_ids,
        help="hard cap for IDs captured from one sampled batch",
    )
    parser.add_argument("--read-log-buffer-mib", type=_positive_int)
    parser.add_argument("--read-log-segment-mib", type=_positive_int)
    parser.add_argument("--read-log-rotate-seconds", type=_positive_int)
    parser.add_argument("--read-log-retention-mib", type=_positive_int)
    parser.add_argument("--read-log-retention-seconds", type=_positive_int)
    parser.add_argument("--read-log-min-free-mib", type=_nonnegative_int)
    parser.add_argument(
        "--read-log-max-write-mib-per-second",
        type=_positive_int,
    )
    parser.add_argument("--read-log-max-retained-files", type=_positive_int)
    parser.add_argument("--no-auto-maintenance", action="store_true")
    parser.add_argument("--no-idempotency-checkpoint", action="store_true")
    parser.add_argument("--no-auto-clean", action="store_true")
    parser.add_argument("--no-orphan-reclaim", action="store_true")
    parser.add_argument("--max-read-range-bytes", type=_positive_int)
    parser.add_argument("--max-gap-bytes", type=_nonnegative_int)
    parser.add_argument("--io-workers", type=_nonnegative_int)
    parser.add_argument("--io-queue-capacity", type=_nonnegative_int)
    parser.add_argument("--io-ranges-per-task", type=_positive_int)
    parser.add_argument(
        "--io-backend",
        choices=("thread_pool", "io_uring"),
    )
    parser.add_argument("--io-uring-queue-depth", type=_positive_int)
    parser.add_argument("--io-uring-workers", type=_positive_int)
    parser.add_argument(
        "--native-binary",
        type=Path,
        help="development/diagnostic override for the internal C++ daemon",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved launch plan without starting the daemon",
    )


def _reader_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: value
        for name in _READER_ARGUMENTS
        if (value := getattr(args, name, None)) is not None
    }


def _validate_transport_arguments(args: argparse.Namespace) -> None:
    endpoint = str(args.listen)
    host: str
    port_text: str
    if endpoint.startswith("["):
        close = endpoint.find("]")
        if close <= 1 or endpoint[close + 1 : close + 2] != ":":
            raise ServerLaunchError(
                "--listen must be a strict HOST:PORT or [IPv6]:PORT endpoint"
            )
        host = endpoint[1:close]
        port_text = endpoint[close + 2 :]
    else:
        if endpoint.count(":") != 1:
            raise ServerLaunchError(
                "--listen must be a strict HOST:PORT or [IPv6]:PORT endpoint"
            )
        host, port_text = endpoint.rsplit(":", 1)
    if (
        not host
        or not port_text.isascii()
        or not port_text.isdecimal()
        or int(port_text) > 65_535
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in host)
    ):
        raise ServerLaunchError(
            "--listen must be a strict HOST:PORT or [IPv6]:PORT endpoint"
        )

    certificate = args.tls_cert is not None
    private_key = args.tls_key is not None
    client_ca = args.tls_client_ca is not None
    if certificate != private_key:
        raise ServerLaunchError(
            "--tls-cert and --tls-key must be configured together"
        )
    if client_ca and not (certificate and private_key):
        raise ServerLaunchError(
            "--tls-client-ca requires --tls-cert and --tls-key"
        )
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if (
        not loopback
        and not (certificate and private_key and client_ca)
        and not args.allow_insecure_non_loopback
    ):
        raise ServerLaunchError(
            "refusing a non-loopback gRPC endpoint without mTLS; configure "
            "--tls-cert, --tls-key, and --tls-client-ca, or explicitly pass "
            "--allow-insecure-non-loopback for a trusted-network deployment"
        )


def _validate_availability_arguments(args: argparse.Namespace) -> None:
    workers = (
        args.max_worker_threads
        if args.max_worker_threads is not None
        else _DEFAULT_MAX_WORKER_THREADS
    )
    calls = (
        args.max_concurrent_calls
        if args.max_concurrent_calls is not None
        else _DEFAULT_MAX_CONCURRENT_CALLS
    )
    scans = (
        args.max_concurrent_scans
        if args.max_concurrent_scans is not None
        else _DEFAULT_MAX_CONCURRENT_SCANS
    )
    grace = (
        args.shutdown_grace_ms
        if args.shutdown_grace_ms is not None
        else _DEFAULT_SHUTDOWN_GRACE_MS
    )
    hard = (
        args.shutdown_hard_timeout_ms
        if args.shutdown_hard_timeout_ms is not None
        else _DEFAULT_SHUTDOWN_HARD_TIMEOUT_MS
    )
    if not scans < calls < workers:
        raise ServerLaunchError(
            "availability limits must satisfy max_concurrent_scans < "
            "max_concurrent_calls < max_worker_threads"
        )
    if hard < grace:
        raise ServerLaunchError(
            "--shutdown-hard-timeout-ms must not be shorter than "
            "--shutdown-grace-ms"
        )


def _validate_maintenance_arguments(args: argparse.Namespace) -> None:
    if args.maintenance_max_actions is not None and args.maintenance_max_actions > 6:
        raise ServerLaunchError("--maintenance-max-actions must not exceed 6")
    checkpoint = args.idempotency_checkpoint_trigger_percent
    if checkpoint is not None and not 1 <= checkpoint < 100:
        raise ServerLaunchError(
            "--idempotency-checkpoint-trigger-percent must be in [1, 99]"
        )
    garbage_ratio = args.cleaner_min_garbage_ratio_ppm
    if garbage_ratio is not None and garbage_ratio > 1_000_000:
        raise ServerLaunchError(
            "--cleaner-min-garbage-ratio-ppm must not exceed 1000000"
        )


def _validate_read_log_arguments(args: argparse.Namespace) -> None:
    directory = args.read_log_dir
    configured = [
        name
        for name in _READ_LOG_ARGUMENTS
        if getattr(args, name, None) is not None
    ]
    if directory is None:
        if configured:
            options = ", ".join(
                "--" + name.replace("_", "-") for name in configured
            )
            raise ServerLaunchError(
                f"{options} require --read-log-dir; workload logging is "
                "disabled unless its dedicated directory is explicit"
            )
        return

    for name in (
        "read_log_buffer_mib",
        "read_log_segment_mib",
        "read_log_retention_mib",
        "read_log_min_free_mib",
        "read_log_max_write_mib_per_second",
    ):
        value = getattr(args, name)
        if value is not None and value > _MAX_UINT64_MIB:
            raise ServerLaunchError(
                f"--{name.replace('_', '-')} exceeds the uint64 MiB range"
            )
    if (
        args.read_log_retention_seconds is not None
        and args.read_log_retention_seconds > _MAX_UINT64
    ):
        raise ServerLaunchError(
            "--read-log-retention-seconds exceeds the uint64 range"
        )

    storage = args.storage.expanduser().resolve(strict=False)
    log_directory = directory.expanduser().resolve(strict=False)
    if (
        log_directory == storage
        or storage in log_directory.parents
        or log_directory in storage.parents
    ):
        raise ServerLaunchError(
            "--read-log-dir must be a dedicated directory disjoint from the "
            "collection directory so telemetry cannot alter collection "
            "permissions or enter collection backup and restore"
        )

    maximum_ids = (
        args.read_log_max_ids_per_event
        if args.read_log_max_ids_per_event is not None
        else _DEFAULT_READ_LOG_MAX_IDS_PER_EVENT
    )
    buffer_mib = (
        args.read_log_buffer_mib
        if args.read_log_buffer_mib is not None
        else _DEFAULT_READ_LOG_BUFFER_MIB
    )
    segment_mib = (
        args.read_log_segment_mib
        if args.read_log_segment_mib is not None
        else _DEFAULT_READ_LOG_SEGMENT_MIB
    )
    retention_mib = (
        args.read_log_retention_mib
        if args.read_log_retention_mib is not None
        else _DEFAULT_READ_LOG_RETENTION_MIB
    )
    maximum_write_mib_per_second = (
        args.read_log_max_write_mib_per_second
        if args.read_log_max_write_mib_per_second is not None
        else _DEFAULT_READ_LOG_MAX_WRITE_MIB_PER_SECOND
    )
    maximum_retained_files = (
        args.read_log_max_retained_files
        if args.read_log_max_retained_files is not None
        else _DEFAULT_READ_LOG_MAX_RETAINED_FILES
    )
    rotate_seconds = (
        args.read_log_rotate_seconds
        if args.read_log_rotate_seconds is not None
        else 3_600
    )
    # The native queue keeps at least two complete fixed-capacity events.
    # Validate the same lower bound before exec so a dry-run is authoritative.
    maximum_event_bytes = (
        _READ_LOG_RECORD_HEADER_BYTES + maximum_ids * 8
    )
    minimum_buffer_bytes = 2 * maximum_event_bytes
    if buffer_mib * (1 << 20) < minimum_buffer_bytes:
        raise ServerLaunchError(
            "--read-log-buffer-mib cannot hold two maximum-sized log events"
        )
    minimum_segment_bytes = (
        _READ_LOG_HEADER_BYTES
        + maximum_event_bytes
        + 2 * _READ_LOG_TERMINAL_CHECKPOINT_BYTES
    )
    if segment_mib * (1 << 20) < minimum_segment_bytes:
        raise ServerLaunchError(
            "--read-log-segment-mib cannot hold the log header and one "
            "opening checkpoint, maximum-sized event, and terminal "
            "checkpoint"
        )
    if retention_mib < segment_mib:
        raise ServerLaunchError(
            "--read-log-retention-mib must be at least "
            "--read-log-segment-mib"
        )
    maximum_write_transaction = (
        _READ_LOG_HEADER_BYTES
        + maximum_event_bytes
        + 2 * _READ_LOG_TERMINAL_CHECKPOINT_BYTES
    )
    if (
        maximum_write_mib_per_second * (1 << 20)
        < maximum_write_transaction
    ):
        raise ServerLaunchError(
            "--read-log-max-write-mib-per-second must cover at least one "
            "maximum-sized event plus its segment header, opening "
            "checkpoint, and terminal checkpoint per second"
        )
    if not 1 <= maximum_retained_files <= _MAX_READ_LOG_RETAINED_FILES:
        raise ServerLaunchError(
            "--read-log-max-retained-files must be in [1, 65533]"
        )
    if rotate_seconds > _MAX_READ_LOG_ROTATE_SECONDS:
        raise ServerLaunchError(
            "--read-log-rotate-seconds must fit signed 64-bit seconds"
        )


def build_server_launch_plan(args: argparse.Namespace) -> ServerLaunchPlan:
    _validate_transport_arguments(args)
    _validate_availability_arguments(args)
    _validate_maintenance_arguments(args)
    _validate_read_log_arguments(args)
    executable = find_native_server(getattr(args, "native_binary", None))
    tuning = resolve_server_tuning(
        args.storage,
        mode=args.tuning,
        overrides=_reader_overrides(args),
    )
    command = [
        str(executable),
        "--storage",
        str(args.storage),
        "--listen",
        str(args.listen),
    ]
    for attribute, option in _NATIVE_VALUE_ARGUMENTS:
        value = getattr(args, attribute, None)
        if value is not None:
            command.extend((option, str(value)))
    if args.writable:
        command.append("--writable")
    else:
        # Keep the public launcher fail-safe even when --native-binary points
        # at an older development daemon whose historical default was writable.
        command.append("--read-only")
    if args.allow_insecure_non_loopback:
        command.append("--allow-insecure-non-loopback")
    for attribute, option in (
        ("tls_cert", "--tls-cert"),
        ("tls_key", "--tls-key"),
        ("tls_client_ca", "--tls-client-ca"),
    ):
        path = getattr(args, attribute)
        if path is not None:
            command.extend((option, str(path)))
    if args.no_durable_idempotency:
        command.append("--no-durable-idempotency")
    if args.permit_keepalive_without_calls:
        command.append("--permit-keepalive-without-calls")
    if args.no_auto_maintenance:
        command.append("--no-auto-maintenance")
    if args.no_idempotency_checkpoint:
        command.append("--no-idempotency-checkpoint")
    if args.no_auto_clean:
        command.append("--no-auto-clean")
    if args.no_orphan_reclaim:
        command.append("--no-orphan-reclaim")

    reader = tuning.options
    command.extend(
        (
            "--max-read-range-bytes",
            str(reader["max_read_range_bytes"]),
            "--max-gap-bytes",
            str(reader["max_gap_bytes"]),
            "--io-workers",
            str(reader["io_workers"]),
            "--io-queue-capacity",
            str(reader["io_queue_capacity"]),
            "--io-ranges-per-task",
            str(reader["io_ranges_per_task"]),
            "--io-backend",
            str(reader["io_backend"]).replace("_", "-"),
            "--io-uring-queue-depth",
            str(reader["io_uring_queue_depth"]),
            "--io-uring-workers",
            str(reader["io_uring_workers"]),
            "--resolved-tuning-mode",
            tuning.mode,
            "--resolved-tuning-source",
            tuning.source,
            "--resolved-tuning-status",
            tuning.status,
            "--resolved-tuning-reuse-level",
            tuning.reuse_level,
            "--resolved-tuning-explicit-overrides",
            "1" if tuning.explicit_overrides else "0",
        )
    )
    return ServerLaunchPlan(
        executable=executable,
        argv=tuple(command),
        reader=dict(reader),
        tuning=tuning.diagnostics(),
    )


def run_server(args: argparse.Namespace) -> None:
    try:
        plan = build_server_launch_plan(args)
    except (OSError, RuntimeError) as error:
        raise ServerLaunchError(f"cannot prepare native server: {error}") from error
    if args.dry_run:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return
    if os.environ.get(_LAUNCH_MARKER):
        raise ServerLaunchError(
            "refusing recursive ThrowDB server launcher execution"
        )
    environment = dict(os.environ)
    environment[_LAUNCH_MARKER] = "1"
    try:
        os.execve(str(plan.executable), list(plan.argv), environment)
    except OSError as error:
        raise ServerLaunchError(
            f"cannot execute native ThrowDB server: {error}"
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throwdbd",
        description="Resolve tuning once, then exec the native ThrowDB server",
    )
    add_server_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_server(args)
    except (
        OSError,
        RuntimeError,
        ServerLaunchError,
        TuningRegistryError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()


__all__ = [
    "ServerLaunchError",
    "ServerLaunchPlan",
    "ServerTuning",
    "add_server_arguments",
    "build_server_launch_plan",
    "find_native_server",
    "main",
    "resolve_server_tuning",
    "run_server",
]
