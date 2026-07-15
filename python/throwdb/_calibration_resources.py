"""Best-effort resource telemetry for bounded reader calibration.

The calibration harness must keep working when Linux procfs/cgroup files are
not mounted, are hidden by a container runtime, or cannot be read.  This
module therefore treats every operating-system counter as optional.  Invalid
arguments supplied by the caller are still rejected, while unavailable or
reset counters are represented explicitly in the returned epoch report.

The public integration surface is intentionally small::

    probe = ResourceProbe()
    before = probe.snapshot()
    # run one measurement epoch
    after = probe.snapshot()
    metrics = derive_epoch_metrics(
        before,
        after,
        elapsed_seconds=elapsed,
        rows=rows,
        useful_bytes=useful_bytes,
    )

``metrics`` contains only JSON-serializable values.
"""

from __future__ import annotations

import math
import operator
import os
import posixpath
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:  # ``resource`` is absent on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on non-POSIX platforms.
    _resource = None  # type: ignore[assignment]


_REPORT_FORMAT: Final = "throwdb-calibration-resource-epoch"
_REPORT_VERSION: Final = 1
_MAX_TEXT_BYTES: Final = 1 << 20
_PRESSURE_RESOURCES: Final = ("cpu", "io", "memory")

Counter = int | float
ProcessUsageReader = Callable[[], Mapping[str, Counter] | None]


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """One best-effort, process-local resource counter snapshot."""

    monotonic_ns: int
    process_usage: dict[str, Counter] | None
    process_io: dict[str, int] | None
    cgroup_path: str | None
    cgroup_cpu_stat: dict[str, int] | None
    cgroup_cpu_max: dict[str, int | float | str | None] | None
    cgroup_cpu_weight: int | None
    cgroup_pressure: dict[str, dict[str, dict[str, int | float]]] | None
    cgroup_resolution: str | None = None


def _read_text(path: Path) -> str | None:
    try:
        with path.open(encoding="ascii") as stream:
            return stream.read(_MAX_TEXT_BYTES)
    except (OSError, UnicodeError):
        return None


def _default_process_usage() -> Mapping[str, Counter] | None:
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
    except (AttributeError, OSError, ValueError):
        return None
    return {
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        # ru_maxrss is KiB on Linux and bytes on macOS.  Preserve the raw unit
        # rather than publishing a misleading cross-platform byte value.
        "max_rss_raw": int(usage.ru_maxrss),
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "block_input_operations": int(usage.ru_inblock),
        "block_output_operations": int(usage.ru_oublock),
        "signals_received": int(usage.ru_nsignals),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def _read_integer_map(path: Path) -> dict[str, int] | None:
    text = _read_text(path)
    if text is None:
        return None
    values: dict[str, int] = {}
    try:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                key, raw_value = line.split(":", 1)
            else:
                key, raw_value = line.split(None, 1)
            values[key.strip()] = int(raw_value.strip())
    except (TypeError, ValueError):
        return None
    return values


def _parse_cpu_max(path: Path) -> dict[str, int | float | str | None] | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        quota_text, period_text = text.strip().split()
        period_usec = int(period_text)
        if period_usec <= 0:
            return None
        if quota_text == "max":
            quota_usec: int | None = None
            limit_cores: float | None = None
        else:
            quota_usec = int(quota_text)
            if quota_usec <= 0:
                return None
            limit_cores = quota_usec / period_usec
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return {
        "raw": f"{quota_text} {period_usec}",
        "quota_usec": quota_usec,
        "period_usec": period_usec,
        "limit_cores": limit_cores,
    }


def _read_cpu_weight(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        weight = int(text.strip())
    except ValueError:
        return None
    return weight if 1 <= weight <= 10_000 else None


def _parse_pressure(
    path: Path,
) -> dict[str, dict[str, int | float]] | None:
    text = _read_text(path)
    if text is None:
        return None
    result: dict[str, dict[str, int | float]] = {}
    try:
        for raw_line in text.splitlines():
            fields = raw_line.split()
            if len(fields) < 2:
                continue
            kind = fields[0]
            values: dict[str, int | float] = {}
            for field in fields[1:]:
                key, raw_value = field.split("=", 1)
                if key == "total":
                    values["total_usec"] = int(raw_value)
                elif key in {"avg10", "avg60", "avg300"}:
                    average = float(raw_value)
                    if not math.isfinite(average):
                        return None
                    values[key] = average
            if "total_usec" in values:
                result[kind] = values
    except (TypeError, ValueError):
        return None
    return result or None


def _is_cgroup_v2_directory(path: Path) -> bool:
    try:
        return any(
            (path / name).is_file()
            for name in (
                "cgroup.controllers",
                "cpu.stat",
                "cpu.max",
                "cpu.weight",
                "cpu.pressure",
            )
        )
    except OSError:
        return False


def _mountinfo_field(value: str) -> str:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _find_cgroup_v2_directory(
    proc_root: Path,
    cgroup_root: Path,
) -> tuple[Path | None, str | None]:
    text = _read_text(proc_root / "self" / "cgroup")
    if text is None:
        return None, None
    relative: str | None = None
    try:
        for raw_line in text.splitlines():
            hierarchy, controllers, path = raw_line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                normalized = posixpath.normpath("/" + path.lstrip("/"))
                if normalized == "/" or normalized.startswith("/"):
                    relative = normalized
                break
    except ValueError:
        return None, None
    if relative is None:
        return None, None

    mountinfo = _read_text(proc_root / "self" / "mountinfo")
    if mountinfo is not None:
        for raw_line in mountinfo.splitlines():
            try:
                left, right = raw_line.split(" - ", 1)
                left_fields = left.split()
                right_fields = right.split()
                if len(left_fields) < 5 or not right_fields:
                    continue
                if right_fields[0] != "cgroup2":
                    continue
                mount_root = posixpath.normpath(
                    "/" + _mountinfo_field(left_fields[3]).lstrip("/")
                )
                if relative == "/":
                    mapped = "/"
                elif mount_root == "/":
                    mapped = relative
                elif relative == mount_root:
                    mapped = "/"
                elif relative.startswith(mount_root.rstrip("/") + "/"):
                    mapped = relative[len(mount_root) :]
                else:
                    continue
                mount_point = Path(_mountinfo_field(left_fields[4]))
                for mount_base in dict.fromkeys((mount_point, cgroup_root)):
                    candidate = mount_base.joinpath(
                        *mapped.strip("/").split("/")
                    )
                    if _is_cgroup_v2_directory(candidate):
                        return candidate, "mountinfo"
            except (IndexError, ValueError):
                continue

    # The cgroup mount can expose the host hierarchy or only the namespace's
    # current subtree.  Try the procfs-relative path first and the mount root
    # second so both layouts degrade cleanly.
    candidate = cgroup_root.joinpath(*relative.strip("/").split("/"))
    if _is_cgroup_v2_directory(candidate):
        return candidate, "proc_membership"
    if candidate != cgroup_root and _is_cgroup_v2_directory(cgroup_root):
        return cgroup_root, "namespace_root_fallback"
    return None, None


def effective_cgroup_cpu_quota(
    *,
    proc_root: str | os.PathLike[str] = "/proc",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> float | None:
    """Return the tightest visible cgroup-v2 CPU quota in CPU cores.

    The current cgroup is resolved through mountinfo, so this also works when
    a container runtime mounts cgroup2 somewhere other than
    ``/sys/fs/cgroup``.  Ancestor limits matter because a child can advertise
    ``max`` while a parent still constrains the whole pod or job.
    """

    current, _ = _find_cgroup_v2_directory(
        Path(proc_root),
        Path(cgroup_root),
    )
    limits: list[float] = []
    for _ in range(64):
        if current is None or not _is_cgroup_v2_directory(current):
            break
        cpu_max = _parse_cpu_max(current / "cpu.max")
        if cpu_max is not None:
            limit = cpu_max.get("limit_cores")
            if (
                isinstance(limit, (int, float))
                and not isinstance(limit, bool)
                and math.isfinite(float(limit))
                and float(limit) > 0
            ):
                limits.append(float(limit))
        parent = current.parent
        if parent == current or not _is_cgroup_v2_directory(parent):
            break
        current = parent
    return min(limits) if limits else None


class ResourceProbe:
    """Capture optional OS resource counters without raising I/O errors."""

    def __init__(
        self,
        *,
        proc_root: str | os.PathLike[str] = "/proc",
        cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
        process_usage_reader: ProcessUsageReader = _default_process_usage,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._proc_root = Path(proc_root)
        self._process_usage_reader = process_usage_reader
        self._monotonic_ns = monotonic_ns
        self._cgroup_path, self._cgroup_resolution = _find_cgroup_v2_directory(
            self._proc_root,
            Path(cgroup_root),
        )

    def snapshot(self) -> ResourceSnapshot:
        """Return all counters currently available to this process."""

        try:
            raw_usage = self._process_usage_reader()
            process_usage = dict(raw_usage) if raw_usage is not None else None
        except (AttributeError, OSError, TypeError, ValueError):
            process_usage = None

        process_io = _read_integer_map(self._proc_root / "self" / "io")
        cgroup = self._cgroup_path
        if cgroup is None:
            cpu_stat = None
            cpu_max = None
            cpu_weight = None
            pressure = None
        else:
            cpu_stat = _read_integer_map(cgroup / "cpu.stat")
            cpu_max = _parse_cpu_max(cgroup / "cpu.max")
            cpu_weight = _read_cpu_weight(cgroup / "cpu.weight")
            pressure = {}
            for resource_name in _PRESSURE_RESOURCES:
                value = _parse_pressure(cgroup / f"{resource_name}.pressure")
                if value is not None:
                    pressure[resource_name] = value
            if not pressure:
                pressure = None

        return ResourceSnapshot(
            monotonic_ns=int(self._monotonic_ns()),
            process_usage=process_usage,
            process_io=process_io,
            cgroup_path=str(cgroup) if cgroup is not None else None,
            cgroup_cpu_stat=cpu_stat,
            cgroup_cpu_max=cpu_max,
            cgroup_cpu_weight=cpu_weight,
            cgroup_pressure=pressure,
            cgroup_resolution=self._cgroup_resolution,
        )


def _nonnegative_deltas(
    before: Mapping[str, Counter] | None,
    after: Mapping[str, Counter] | None,
    *,
    prefix: str,
    excluded: frozenset[str] = frozenset(),
) -> tuple[dict[str, Counter], list[str]]:
    if before is None or after is None:
        return {}, []
    deltas: dict[str, Counter] = {}
    resets: list[str] = []
    for key in sorted(before.keys() & after.keys()):
        if key in excluded:
            continue
        difference = after[key] - before[key]
        if difference < 0:
            resets.append(f"{prefix}.{key}")
        else:
            deltas[key] = difference
    return deltas, resets


def _pressure_deltas(
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    elapsed_seconds: float,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    resets: list[str] = []
    if before.cgroup_pressure is None or after.cgroup_pressure is None:
        return result, resets
    for resource_name in sorted(
        before.cgroup_pressure.keys() & after.cgroup_pressure.keys()
    ):
        before_resource = before.cgroup_pressure[resource_name]
        after_resource = after.cgroup_pressure[resource_name]
        resource_result: dict[str, dict[str, object]] = {}
        for kind in sorted(before_resource.keys() & after_resource.keys()):
            before_total = before_resource[kind].get("total_usec")
            after_total = after_resource[kind].get("total_usec")
            if not isinstance(before_total, int) or not isinstance(after_total, int):
                continue
            total_usec = after_total - before_total
            if total_usec < 0:
                resets.append(f"cgroup_pressure.{resource_name}.{kind}.total_usec")
                continue
            seconds = total_usec / 1_000_000.0
            end_averages = {
                key: float(after_resource[kind][key])
                for key in ("avg10", "avg60", "avg300")
                if key in after_resource[kind]
            }
            resource_result[kind] = {
                "total_usec": total_usec,
                "seconds": seconds,
                "seconds_per_wall_second": seconds / elapsed_seconds,
                "end_averages": end_averages,
            }
        if resource_result:
            result[resource_name] = resource_result
    return result, resets


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer)


def derive_epoch_metrics(
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    *,
    rows: object,
    useful_bytes: object | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Derive JSON-serializable resource metrics for one measurement epoch."""

    measured_rows = _nonnegative_integer(rows, "rows")
    measured_useful_bytes = (
        None
        if useful_bytes is None
        else _nonnegative_integer(useful_bytes, "useful_bytes")
    )
    if elapsed_seconds is None:
        elapsed = (after.monotonic_ns - before.monotonic_ns) / 1_000_000_000.0
    else:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError) as error:
            raise TypeError("elapsed_seconds must be a number") from error
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed_seconds must be finite and greater than zero")

    process_delta, process_resets = _nonnegative_deltas(
        before.process_usage,
        after.process_usage,
        prefix="process_usage",
        excluded=frozenset({"max_rss_raw"}),
    )
    user_seconds = float(process_delta.get("user_cpu_seconds", 0.0))
    system_seconds = float(process_delta.get("system_cpu_seconds", 0.0))
    process_cpu_seconds = user_seconds + system_seconds
    process_available = before.process_usage is not None and after.process_usage is not None
    process_metrics: dict[str, object] = {
        "available": process_available,
        "delta": process_delta,
        "max_rss_raw": (
            after.process_usage.get("max_rss_raw")
            if after.process_usage is not None
            else None
        ),
        "cpu_seconds": process_cpu_seconds if process_available else None,
        "average_cpu_cores": (
            process_cpu_seconds / elapsed if process_available else None
        ),
        "rows_per_cpu_second": (
            measured_rows / process_cpu_seconds
            if process_available and process_cpu_seconds > 0
            else None
        ),
        "cpu_seconds_per_million_rows": (
            process_cpu_seconds * 1_000_000.0 / measured_rows
            if process_available and measured_rows > 0
            else None
        ),
    }

    io_delta, io_resets = _nonnegative_deltas(
        before.process_io,
        after.process_io,
        prefix="process_io",
    )
    storage_read_bytes = int(io_delta.get("read_bytes", 0))
    read_syscalls = int(io_delta.get("syscr", 0))
    io_available = before.process_io is not None and after.process_io is not None
    io_metrics: dict[str, object] = {
        "available": io_available,
        "delta": io_delta,
        "storage_read_bytes": storage_read_bytes if io_available else None,
        "storage_read_bytes_per_second": (
            storage_read_bytes / elapsed if io_available else None
        ),
        "storage_read_bytes_per_row": (
            storage_read_bytes / measured_rows
            if io_available and measured_rows > 0
            else None
        ),
        "backing_read_ratio": (
            storage_read_bytes / measured_useful_bytes
            if io_available
            and measured_useful_bytes is not None
            and measured_useful_bytes > 0
            else None
        ),
        "read_syscalls_per_million_rows": (
            read_syscalls * 1_000_000.0 / measured_rows
            if io_available and measured_rows > 0
            else None
        ),
    }

    cgroup_delta, cgroup_resets = _nonnegative_deltas(
        before.cgroup_cpu_stat,
        after.cgroup_cpu_stat,
        prefix="cgroup_cpu",
    )
    pressure, pressure_resets = _pressure_deltas(before, after, elapsed)
    cgroup_available = (
        before.cgroup_cpu_stat is not None and after.cgroup_cpu_stat is not None
    )
    cgroup_cpu_seconds = (
        float(cgroup_delta["usage_usec"]) / 1_000_000.0
        if "usage_usec" in cgroup_delta
        else None
    )
    throttled_seconds = (
        float(cgroup_delta["throttled_usec"]) / 1_000_000.0
        if "throttled_usec" in cgroup_delta
        else None
    )
    periods = int(cgroup_delta.get("nr_periods", 0))
    throttled_periods = int(cgroup_delta.get("nr_throttled", 0))
    cpu_limit = (
        after.cgroup_cpu_max.get("limit_cores")
        if after.cgroup_cpu_max is not None
        else None
    )
    cgroup_metrics: dict[str, object] = {
        "available": cgroup_available,
        "path": after.cgroup_path,
        "resolution": after.cgroup_resolution,
        "configuration": {
            "cpu_max": after.cgroup_cpu_max,
            "cpu_weight": after.cgroup_cpu_weight,
        },
        "cpu_delta": cgroup_delta,
        "cpu_seconds": cgroup_cpu_seconds,
        "average_cpu_cores": (
            cgroup_cpu_seconds / elapsed
            if cgroup_cpu_seconds is not None
            else None
        ),
        "cpu_quota_utilization": (
            cgroup_cpu_seconds / (float(cpu_limit) * elapsed)
            if cgroup_cpu_seconds is not None
            and isinstance(cpu_limit, (int, float))
            and cpu_limit > 0
            else None
        ),
        "process_cpu_share": (
            process_cpu_seconds / cgroup_cpu_seconds
            if process_available
            and cgroup_cpu_seconds is not None
            and cgroup_cpu_seconds > 0
            else None
        ),
        "throttled_seconds": throttled_seconds,
        "throttled_period_fraction": (
            throttled_periods / periods if periods > 0 else None
        ),
        "throttled_seconds_per_wall_second": (
            throttled_seconds / elapsed
            if throttled_seconds is not None
            else None
        ),
        "pressure": pressure,
    }

    configuration_changed = (
        before.cgroup_cpu_max != after.cgroup_cpu_max
        or before.cgroup_cpu_weight != after.cgroup_cpu_weight
        or before.cgroup_path != after.cgroup_path
        or before.cgroup_resolution != after.cgroup_resolution
    )
    resets = sorted(
        process_resets + io_resets + cgroup_resets + pressure_resets
    )
    sections = {
        "process": process_available,
        "process_io": io_available,
        "cgroup_cpu": cgroup_available,
        "cgroup_pressure": bool(pressure),
    }
    cgroup_attribution_confident = after.cgroup_resolution in {
        "mountinfo",
        "proc_membership",
    }
    return {
        "format": _REPORT_FORMAT,
        "version": _REPORT_VERSION,
        "elapsed_seconds": elapsed,
        "rows": measured_rows,
        "useful_bytes": measured_useful_bytes,
        "available": any(sections.values()),
        "sections": sections,
        "process": process_metrics,
        "io": io_metrics,
        "cgroup": cgroup_metrics,
        "quality": {
            "complete": (
                all(sections.values())
                and not resets
                and not configuration_changed
                and cgroup_attribution_confident
            ),
            "counter_resets": resets,
            "cgroup_configuration_changed": configuration_changed,
            "cgroup_attribution_confident": cgroup_attribution_confident,
        },
    }


__all__ = [
    "ResourceProbe",
    "ResourceSnapshot",
    "derive_epoch_metrics",
    "effective_cgroup_cpu_quota",
]
