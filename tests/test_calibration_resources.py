from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from throwdb._calibration_resources import (
    ResourceProbe,
    ResourceSnapshot,
    derive_epoch_metrics,
    effective_cgroup_cpu_quota,
)


class _SequenceReader:
    def __init__(self, values: list[Mapping[str, int | float] | None]) -> None:
        self._values = iter(values)

    def __call__(self) -> Mapping[str, int | float] | None:
        return next(self._values)


def _write_counters(path: Path, values: Mapping[str, int]) -> None:
    path.write_text(
        "".join(f"{key} {value}\n" for key, value in values.items()),
        encoding="ascii",
    )


def _write_proc_io(path: Path, values: Mapping[str, int]) -> None:
    path.write_text(
        "".join(f"{key}: {value}\n" for key, value in values.items()),
        encoding="ascii",
    )


def _make_linux_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    proc = tmp_path / "proc"
    proc_self = proc / "self"
    proc_self.mkdir(parents=True)
    (proc_self / "cgroup").write_text("0::/workload\n", encoding="ascii")
    cgroup_root = tmp_path / "cgroup"
    workload = cgroup_root / "workload"
    workload.mkdir(parents=True)
    (workload / "cgroup.controllers").write_text("cpu io memory\n", encoding="ascii")
    return proc, cgroup_root, workload


def test_full_epoch_collects_cpu_io_cgroup_and_pressure(tmp_path: Path) -> None:
    proc, cgroup_root, workload = _make_linux_tree(tmp_path)
    proc_io = proc / "self" / "io"
    _write_proc_io(proc_io, {"rchar": 1_000, "syscr": 10, "read_bytes": 0})
    _write_counters(
        workload / "cpu.stat",
        {
            "usage_usec": 1_000_000,
            "user_usec": 700_000,
            "system_usec": 300_000,
            "nr_periods": 10,
            "nr_throttled": 1,
            "throttled_usec": 50_000,
        },
    )
    (workload / "cpu.max").write_text("1000000 100000\n", encoding="ascii")
    (workload / "cpu.weight").write_text("200\n", encoding="ascii")
    (workload / "cpu.pressure").write_text(
        "some avg10=1.00 avg60=2.00 avg300=3.00 total=100000\n",
        encoding="ascii",
    )
    (workload / "io.pressure").write_text(
        "some avg10=4.00 avg60=5.00 avg300=6.00 total=200000\n"
        "full avg10=0.50 avg60=0.25 avg300=0.10 total=50000\n",
        encoding="ascii",
    )

    usage = _SequenceReader(
        [
            {
                "user_cpu_seconds": 2.0,
                "system_cpu_seconds": 1.0,
                "max_rss_raw": 100,
                "voluntary_context_switches": 10,
            },
            {
                "user_cpu_seconds": 2.5,
                "system_cpu_seconds": 1.25,
                "max_rss_raw": 120,
                "voluntary_context_switches": 14,
            },
        ]
    )
    clock = iter((1_000_000_000, 2_000_000_000))
    probe = ResourceProbe(
        proc_root=proc,
        cgroup_root=cgroup_root,
        process_usage_reader=usage,
        monotonic_ns=lambda: next(clock),
    )
    before = probe.snapshot()

    _write_proc_io(
        proc_io,
        {"rchar": 9_192, "syscr": 18, "read_bytes": 4_096},
    )
    _write_counters(
        workload / "cpu.stat",
        {
            "usage_usec": 2_000_000,
            "user_usec": 1_400_000,
            "system_usec": 600_000,
            "nr_periods": 15,
            "nr_throttled": 2,
            "throttled_usec": 250_000,
        },
    )
    (workload / "cpu.pressure").write_text(
        "some avg10=1.50 avg60=2.50 avg300=3.50 total=250000\n",
        encoding="ascii",
    )
    (workload / "io.pressure").write_text(
        "some avg10=4.50 avg60=5.50 avg300=6.50 total=500000\n"
        "full avg10=0.75 avg60=0.50 avg300=0.20 total=100000\n",
        encoding="ascii",
    )
    after = probe.snapshot()

    result = derive_epoch_metrics(
        before,
        after,
        rows=1_000,
        useful_bytes=2_048,
        elapsed_seconds=1.0,
    )

    assert result["available"] is True
    assert result["sections"] == {
        "process": True,
        "process_io": True,
        "cgroup_cpu": True,
        "cgroup_pressure": True,
    }
    process = result["process"]
    assert isinstance(process, dict)
    assert process["cpu_seconds"] == pytest.approx(0.75)
    assert process["average_cpu_cores"] == pytest.approx(0.75)
    assert process["rows_per_cpu_second"] == pytest.approx(1_333.333333)
    assert process["cpu_seconds_per_million_rows"] == pytest.approx(750.0)
    assert process["max_rss_raw"] == 120

    io = result["io"]
    assert isinstance(io, dict)
    assert io["storage_read_bytes"] == 4_096
    assert io["storage_read_bytes_per_second"] == 4_096
    assert io["backing_read_ratio"] == 2.0
    assert io["read_syscalls_per_million_rows"] == 8_000

    cgroup = result["cgroup"]
    assert isinstance(cgroup, dict)
    assert cgroup["cpu_seconds"] == 1.0
    assert cgroup["average_cpu_cores"] == 1.0
    assert cgroup["cpu_quota_utilization"] == pytest.approx(0.1)
    assert cgroup["process_cpu_share"] == pytest.approx(0.75)
    assert cgroup["throttled_seconds"] == pytest.approx(0.2)
    assert cgroup["throttled_period_fraction"] == pytest.approx(0.2)
    assert cgroup["throttled_seconds_per_wall_second"] == pytest.approx(0.2)
    assert cgroup["configuration"] == {
        "cpu_max": {
            "raw": "1000000 100000",
            "quota_usec": 1_000_000,
            "period_usec": 100_000,
            "limit_cores": 10.0,
        },
        "cpu_weight": 200,
    }
    pressure = cgroup["pressure"]
    assert pressure["cpu"]["some"]["total_usec"] == 150_000
    assert pressure["cpu"]["some"]["seconds_per_wall_second"] == 0.15
    assert pressure["cpu"]["some"]["end_averages"]["avg10"] == 1.5
    assert pressure["io"]["full"]["total_usec"] == 50_000
    assert result["quality"] == {
        "complete": True,
        "counter_resets": [],
        "cgroup_configuration_changed": False,
        "cgroup_attribution_confident": True,
    }
    # The result is suitable for direct inclusion in a calibration JSON file.
    json.dumps(result, allow_nan=False)


def test_missing_platform_sources_degrade_to_unavailable(tmp_path: Path) -> None:
    clock = iter((10, 1_000_000_010))
    probe = ResourceProbe(
        proc_root=tmp_path / "missing-proc",
        cgroup_root=tmp_path / "missing-cgroup",
        process_usage_reader=lambda: None,
        monotonic_ns=lambda: next(clock),
    )

    result = derive_epoch_metrics(probe.snapshot(), probe.snapshot(), rows=0)

    assert result["available"] is False
    assert result["sections"] == {
        "process": False,
        "process_io": False,
        "cgroup_cpu": False,
        "cgroup_pressure": False,
    }
    assert result["process"]["cpu_seconds"] is None
    assert result["io"]["storage_read_bytes"] is None
    assert result["cgroup"]["configuration"] == {
        "cpu_max": None,
        "cpu_weight": None,
    }
    assert result["quality"]["complete"] is False
    json.dumps(result, allow_nan=False)


def test_namespace_mount_root_and_unlimited_cpu_are_supported(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(
        "0::/host/path/hidden/by/namespace\n",
        encoding="ascii",
    )
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "cpu.stat").write_text("usage_usec 1\n", encoding="ascii")
    (cgroup_root / "cpu.max").write_text("max 100000\n", encoding="ascii")
    (cgroup_root / "cpu.weight").write_text("100\n", encoding="ascii")
    clock = iter((1, 1_000_000_001))
    probe = ResourceProbe(
        proc_root=proc,
        cgroup_root=cgroup_root,
        process_usage_reader=lambda: None,
        monotonic_ns=lambda: next(clock),
    )

    before = probe.snapshot()
    (cgroup_root / "cpu.stat").write_text("usage_usec 1000001\n", encoding="ascii")
    after = probe.snapshot()
    result = derive_epoch_metrics(before, after, rows=1)

    assert after.cgroup_path == str(cgroup_root)
    assert result["cgroup"]["configuration"]["cpu_max"] == {
        "raw": "max 100000",
        "quota_usec": None,
        "period_usec": 100_000,
        "limit_cores": None,
    }
    assert result["cgroup"]["cpu_quota_utilization"] is None
    assert result["quality"]["cgroup_attribution_confident"] is False


def test_mountinfo_resolves_a_namespaced_cgroup_root_confidently(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/\n", encoding="ascii")
    cgroup_root = tmp_path / "nonstandard-cgroup-mount"
    cgroup_root.mkdir()
    (proc / "self" / "mountinfo").write_text(
        f"29 23 0:26 /kubepods/pod-1 {cgroup_root} rw - "
        "cgroup2 cgroup rw\n",
        encoding="ascii",
    )
    (cgroup_root / "cpu.stat").write_text(
        "usage_usec 1\nnr_periods 0\nnr_throttled 0\n"
        "throttled_usec 0\n",
        encoding="ascii",
    )
    probe = ResourceProbe(
        proc_root=proc,
        cgroup_root=cgroup_root,
        process_usage_reader=lambda: None,
    )

    snapshot = probe.snapshot()

    assert snapshot.cgroup_path == str(cgroup_root)
    assert snapshot.cgroup_resolution == "mountinfo"


def test_effective_cpu_quota_uses_nonstandard_mount_and_tightest_parent(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(
        "0::/kubepods/pod-1/task\n",
        encoding="ascii",
    )
    mount = tmp_path / "mounted-cgroup2"
    task = mount / "task"
    task.mkdir(parents=True)
    (proc / "self" / "mountinfo").write_text(
        f"29 23 0:26 /kubepods/pod-1 {mount} rw - "
        "cgroup2 cgroup rw\n",
        encoding="ascii",
    )
    (mount / "cpu.stat").write_text("usage_usec 1\n", encoding="ascii")
    (mount / "cpu.max").write_text("400000 100000\n", encoding="ascii")
    (task / "cpu.stat").write_text("usage_usec 1\n", encoding="ascii")
    (task / "cpu.max").write_text("800000 100000\n", encoding="ascii")

    quota = effective_cgroup_cpu_quota(
        proc_root=proc,
        cgroup_root=tmp_path / "unused-default-mount",
    )

    assert quota == 4.0


def test_counter_reset_and_configuration_change_are_reported() -> None:
    before = ResourceSnapshot(
        monotonic_ns=0,
        process_usage={"user_cpu_seconds": 2.0, "max_rss_raw": 100},
        process_io={"read_bytes": 100},
        cgroup_path="/old",
        cgroup_cpu_stat={"usage_usec": 1_000, "nr_periods": 5},
        cgroup_cpu_max={
            "raw": "100000 100000",
            "quota_usec": 100_000,
            "period_usec": 100_000,
            "limit_cores": 1.0,
        },
        cgroup_cpu_weight=100,
        cgroup_pressure=None,
    )
    after = ResourceSnapshot(
        monotonic_ns=1_000_000_000,
        process_usage={"user_cpu_seconds": 1.0, "max_rss_raw": 50},
        process_io={"read_bytes": 10},
        cgroup_path="/new",
        cgroup_cpu_stat={"usage_usec": 100, "nr_periods": 7},
        cgroup_cpu_max={
            "raw": "200000 100000",
            "quota_usec": 200_000,
            "period_usec": 100_000,
            "limit_cores": 2.0,
        },
        cgroup_cpu_weight=200,
        cgroup_pressure=None,
    )

    result = derive_epoch_metrics(before, after, rows=1)

    assert result["quality"] == {
        "complete": False,
        "counter_resets": [
            "cgroup_cpu.usage_usec",
            "process_io.read_bytes",
            "process_usage.user_cpu_seconds",
        ],
        "cgroup_configuration_changed": True,
        "cgroup_attribution_confident": False,
    }
    assert "usage_usec" not in result["cgroup"]["cpu_delta"]
    assert result["io"]["storage_read_bytes"] == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rows": -1}, ValueError),
        ({"rows": True}, TypeError),
        ({"rows": 1, "useful_bytes": -1}, ValueError),
        ({"rows": 1, "elapsed_seconds": 0}, ValueError),
        ({"rows": 1, "elapsed_seconds": float("nan")}, ValueError),
    ],
)
def test_invalid_epoch_arguments_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    snapshot = ResourceSnapshot(
        monotonic_ns=0,
        process_usage=None,
        process_io=None,
        cgroup_path=None,
        cgroup_cpu_stat=None,
        cgroup_cpu_max=None,
        cgroup_cpu_weight=None,
        cgroup_pressure=None,
    )
    with pytest.raises(error):
        derive_epoch_metrics(snapshot, snapshot, **kwargs)
