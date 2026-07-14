from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

from throwdb._device import (
    SystemProbeSource,
    _parse_mountinfo,
    _select_mount,
    probe_collection_storage,
)


SALT = bytes.fromhex("00112233445566778899aabbccddeeff")


class BytesPath:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def __fspath__(self) -> bytes:
        return self.value


class FakeProbeSource:
    def __init__(self, *, platform: str = "linux") -> None:
        self.platform = platform
        self.stats: dict[str, object] = {}
        self.statvfs_values: dict[str, object] = {}
        self.realpaths: dict[str, str] = {}
        self.glob_values: dict[str, Sequence[str]] = {}
        self.glob_errors: dict[str, OSError] = {}
        self.glob_limits: list[tuple[str, int]] = []
        self.texts: dict[str, str] = {}
        self.directories: set[str] = set()
        self.listings: dict[str, Sequence[str]] = {}
        self.uname_value = ("Linux", "6.1.0-test", "x86_64")
        self.affinity_value: Sequence[int] | None = (0, 1, 2, 3)
        self.stat_calls: Counter[str] = Counter()
        self.statvfs_calls: Counter[str] = Counter()
        self.read_calls: Counter[str] = Counter()
        self.is_dir_calls: Counter[str] = Counter()

    def stat(self, path: str) -> object:
        self.stat_calls[path] += 1
        if path not in self.stats:
            raise FileNotFoundError(path)
        return self.stats[path]

    def statvfs(self, path: str) -> object:
        self.statvfs_calls[path] += 1
        if path not in self.statvfs_values:
            raise FileNotFoundError(path)
        return self.statvfs_values[path]

    def realpath(self, path: str) -> str:
        return self.realpaths.get(path, path)

    def glob(self, pattern: str) -> Sequence[str]:
        if pattern in self.glob_errors:
            raise self.glob_errors[pattern]
        return self.glob_values.get(pattern, ())

    def glob_limited(
        self,
        pattern: str,
        *,
        limit: int,
    ) -> tuple[Sequence[str], bool, int]:
        self.glob_limits.append((pattern, limit))
        values = self.glob(pattern)
        scanned = min(len(values), limit)
        return values[:limit], len(values) < limit, scanned

    def read_text(self, path: str) -> str:
        self.read_calls[path] += 1
        if path not in self.texts:
            raise FileNotFoundError(path)
        return self.texts[path]

    def listdir(self, path: str) -> Sequence[str]:
        if path not in self.listings:
            raise FileNotFoundError(path)
        return self.listings[path]

    def is_dir(self, path: str) -> bool:
        self.is_dir_calls[path] += 1
        return path in self.directories

    def uname(self) -> tuple[str, str, str]:
        return self.uname_value

    def cpu_affinity(self) -> Sequence[int] | None:
        return self.affinity_value


def stat_device(major: int, minor: int) -> object:
    # Explicit fields keep Linux mountinfo fixtures portable when pytest runs
    # on a macOS development host whose dev_t encoding differs from Linux.
    return SimpleNamespace(st_dev=0, st_dev_major=major, st_dev_minor=minor)


def statvfs_with_fsid(fsid: int | None) -> object:
    if fsid is None:
        return SimpleNamespace()
    return SimpleNamespace(f_fsid=fsid)


def add_runtime_files(source: FakeProbeSource, *, cpu_max: str = "max 100000") -> None:
    source.texts["/proc/self/cgroup"] = "0::/tenant/job\n"
    source.texts["/sys/fs/cgroup/tenant/job/cpu.max"] = cpu_max + "\n"
    source.texts["/proc/sys/kernel/io_uring_disabled"] = "0\n"


def mount_line(
    mount_id: int,
    major: int,
    minor: int,
    root: str,
    mount_point: str,
    fs_type: str,
    mount_source: str,
    *,
    mount_options: str = "rw,relatime",
    super_options: str = "rw",
) -> str:
    return (
        f"{mount_id} 1 {major}:{minor} {root} {mount_point} "
        f"{mount_options} - {fs_type} {mount_source} {super_options}"
    )


def test_mountinfo_octal_unescape_and_deepest_exact_device_match() -> None:
    entries = _parse_mountinfo(
        "\n".join(
            (
                mount_line(1, 8, 1, "/", "/", "ext4", "/dev/root"),
                mount_line(2, 8, 1, "/bound", "/data\\040set", "ext4", "/dev/root"),
                mount_line(
                    3,
                    8,
                    2,
                    "/wrong",
                    "/data\\040set/nested",
                    "xfs",
                    "/dev/other",
                ),
            )
        )
    )

    assert entries[1].mount_point == "/data set"
    assert entries[1].root == "/bound"
    selected = _select_mount(entries, "/data set/nested/file", 8, 1)
    assert selected is not None
    assert selected.mount_id == 2
    assert _select_mount(entries, "/data set/nested/file", 8, 3) is None


def local_partition_source(
    *,
    root: str = "/collection",
    wwid: str = "nvme-secret-wwid",
) -> FakeProbeSource:
    source = FakeProbeSource()
    data_path = f"{root}/data/segment-1.dat"
    pattern = f"{root}/data/segment-*.dat"
    source.glob_values[pattern] = (data_path,)
    source.stats[root] = stat_device(259, 1)
    source.stats[data_path] = stat_device(259, 1)
    source.statvfs_values[root] = statvfs_with_fsid(0x12345678)
    source.statvfs_values[data_path] = statvfs_with_fsid(0x12345678)
    for directory_name in ("data", "deltas", "wal"):
        directory = f"{root}/{directory_name}"
        source.directories.add(directory)
        source.stats[directory] = stat_device(259, 1)
        source.statvfs_values[directory] = statvfs_with_fsid(0x12345678)
    source.texts["/proc/self/mountinfo"] = "\n".join(
        (
            mount_line(1, 1, 1, "/", "/", "overlay", "overlay"),
            mount_line(
                2,
                259,
                1,
                "/",
                root,
                "ext4",
                "/dev/nvme0n1p1",
            ),
            mount_line(3, 0, 28, "/", "/sys/fs/cgroup", "cgroup2", "cgroup"),
        )
    )

    partition_link = "/sys/dev/block/259:1"
    partition_path = "/sys/devices/pci/block/nvme0n1/nvme0n1p1"
    disk_link = "/sys/dev/block/259:0"
    disk_path = "/sys/devices/pci/block/nvme0n1"
    source.directories.update((partition_link, disk_link))
    source.realpaths.update(
        {
            partition_link: partition_path,
            disk_link: disk_path,
        }
    )
    source.texts.update(
        {
            f"{partition_path}/partition": "1\n",
            f"{partition_path}/start": "2048\n",
            f"{partition_path}/size": "999999\n",
            f"{disk_path}/dev": "259:0\n",
            f"{disk_path}/device/wwid": wwid + "\n",
        }
    )
    add_runtime_files(source)
    return source


def test_probe_uses_data_shard_mount_and_returns_only_redacted_identity() -> None:
    source = local_partition_source()
    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )

    assert result.probed_path_count == 5
    assert result.used_root_fallback is False
    assert result.media.kind == "local_block"
    assert result.media.confidence == "high"
    assert result.media.reuse_scope == "durable"
    assert "wwid" in result.media.evidence
    assert "partition" in result.media.evidence

    serialized = json.dumps(result.as_redacted_dict(), sort_keys=True)
    assert "/collection" not in serialized
    assert "/dev/nvme0n1p1" not in serialized
    assert "nvme-secret-wwid" not in serialized


def test_default_discovery_probes_base_delta_and_wal_payloads() -> None:
    source = local_partition_source()
    delta_path = "/collection/deltas/delta-a.dat"
    wal_path = "/collection/wal/wal-a.wal"
    source.glob_values["/collection/deltas/delta-*.dat"] = (delta_path,)
    source.glob_values["/collection/wal/*.wal"] = (wal_path,)
    for path in (delta_path, wal_path):
        source.stats[path] = stat_device(259, 1)
        source.statvfs_values[path] = statvfs_with_fsid(0x12345678)

    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )
    assert result.probed_path_count == 7
    assert result.media.component_count == 1
    assert result.media.reuse_scope == "durable"
    for path in (
        "/collection",
        "/collection/data",
        "/collection/data/segment-1.dat",
        "/collection/deltas",
        delta_path,
        "/collection/wal",
        wal_path,
    ):
        assert source.stat_calls[path] == 1
    assert sum(source.statvfs_calls.values()) == 1
    assert (
        source.read_calls[
            "/sys/devices/pci/block/nvme0n1/device/wwid"
        ]
        == 1
    )


def test_incomplete_payload_discovery_fails_closed() -> None:
    complete_source = local_partition_source()
    incomplete_source = local_partition_source()
    incomplete_source.glob_errors["/collection/deltas/delta-*.dat"] = PermissionError(
        "not allowed"
    )

    complete = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=complete_source,
    )
    incomplete = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=incomplete_source,
    )
    assert complete.media.reuse_scope == "durable"
    assert incomplete.media.reuse_scope == "none"
    assert incomplete.media.confidence == "low"
    assert "discovery_incomplete" in incomplete.media.evidence
    assert incomplete.media.fingerprint != complete.media.fingerprint


def test_payload_discovery_stops_at_limit_and_fails_closed() -> None:
    source = local_partition_source()
    delta_paths = (
        "/collection/deltas/delta-a.dat",
        "/collection/deltas/delta-b.dat",
    )
    source.glob_values["/collection/deltas/delta-*.dat"] = delta_paths
    for path in delta_paths:
        source.stats[path] = stat_device(259, 1)
        source.statvfs_values[path] = statvfs_with_fsid(0x12345678)

    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        discovery_entry_limit=2,
        source=source,
    )

    assert result.probed_path_count == 5
    assert result.media.reuse_scope == "none"
    assert "discovery_incomplete" in result.media.evidence
    assert source.glob_limits == [
        ("/collection/data/segment-*.dat", 2),
        ("/collection/deltas/delta-*.dat", 1),
    ]
    assert source.stat_calls[delta_paths[0]] == 1
    assert source.stat_calls[delta_paths[1]] == 0
    assert source.is_dir_calls["/collection/wal"] == 0

    for invalid in (0, 65_537, True):
        with pytest.raises(ValueError, match="discovery_entry_limit"):
            probe_collection_storage(
                "/collection",
                collection_salt=SALT,
                discovery_entry_limit=invalid,
                source=source,
            )


def test_system_glob_counts_nonmatching_entries_toward_limit(tmp_path: Path) -> None:
    directory = tmp_path / "data"
    directory.mkdir()
    for index in range(5):
        (directory / f"temporary-{index}.tmp").write_bytes(b"")

    matches, complete, scanned = SystemProbeSource().glob_limited(
        str(directory / "segment-*.dat"),
        limit=2,
    )
    assert matches == ()
    assert complete is False
    assert scanned == 2


def test_local_fingerprint_ignores_mountpoint_but_changes_with_device() -> None:
    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=local_partition_source(),
    )
    moved = probe_collection_storage(
        "/moved",
        collection_salt=SALT,
        source=local_partition_source(root="/moved"),
    )
    replaced = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=local_partition_source(wwid="replacement-device-wwid"),
    )

    assert first.media.fingerprint == moved.media.fingerprint
    assert first.media.fingerprint != replaced.media.fingerprint


def test_placeholder_hardware_identifier_is_not_durable() -> None:
    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=local_partition_source(wwid="0000-0000"),
    )
    assert result.media.confidence == "low"
    assert result.media.reuse_scope == "none"


def test_restricted_local_block_can_use_redacted_host_scoped_identity() -> None:
    machine_id = "0123456789abcdef0123456789abcdef"
    first_source = local_partition_source(wwid="0000-0000")
    first_source.texts["/etc/machine-id"] = machine_id + "\n"
    other_host_source = local_partition_source(wwid="0000-0000")
    other_host_source.texts["/etc/machine-id"] = (
        "fedcba9876543210fedcba9876543210\n"
    )
    other_device_source = local_partition_source(wwid="0000-0000")
    other_device_source.texts["/etc/machine-id"] = machine_id + "\n"
    for path in tuple(other_device_source.stats):
        other_device_source.stats[path] = stat_device(259, 2)
    other_device_source.texts["/proc/self/mountinfo"] = other_device_source.texts[
        "/proc/self/mountinfo"
    ].replace("259:1", "259:2")

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=first_source,
    )
    other_host = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=other_host_source,
    )
    other_device = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=other_device_source,
    )

    assert first.media.kind == "local_block"
    assert first.media.confidence == "medium"
    assert first.media.reuse_scope == "host"
    assert "host_scoped_device" in first.media.evidence
    assert first.media.fingerprint != other_host.media.fingerprint
    assert first.media.fingerprint != other_device.media.fingerprint
    assert first.runtime.fingerprint == other_host.runtime.fingerprint
    assert machine_id not in json.dumps(first.as_redacted_dict())


def test_machine_id_falls_back_to_dbus_location() -> None:
    source = local_partition_source(wwid="0000-0000")
    machine_id = "0123456789abcdef0123456789abcdef"
    source.texts["/var/lib/dbus/machine-id"] = machine_id + "\n"
    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )
    assert result.media.reuse_scope == "host"
    assert "host_identity" in result.media.evidence
    assert machine_id not in json.dumps(result.as_redacted_dict())


def test_block_probe_recursively_includes_device_mapper_slaves() -> None:
    source = FakeProbeSource()
    data_path = "/db/data/segment-1.dat"
    source.stats[data_path] = stat_device(253, 0)
    source.statvfs_values[data_path] = statvfs_with_fsid(42)
    source.texts["/proc/self/mountinfo"] = "\n".join(
        (
            mount_line(5, 253, 0, "/", "/db", "xfs", "/dev/mapper/vectors"),
            mount_line(6, 0, 28, "/", "/sys/fs/cgroup", "cgroup2", "cgroup"),
        )
    )
    dm_link = "/sys/dev/block/253:0"
    dm_path = "/sys/devices/virtual/block/dm-0"
    leaf_link = "/sys/dev/block/259:0"
    leaf_path = "/sys/devices/pci/block/nvme0n1"
    slave_link = f"{dm_path}/slaves/nvme0n1"
    source.directories.update((dm_link, leaf_link))
    source.realpaths.update(
        {
            dm_link: dm_path,
            leaf_link: leaf_path,
            slave_link: leaf_path,
        }
    )
    source.listings[f"{dm_path}/slaves"] = ("nvme0n1",)
    source.texts.update(
        {
            f"{dm_path}/dm/uuid": "LVM-logical-volume\n",
            f"{leaf_path}/dev": "259:0\n",
            f"{leaf_path}/device/wwid": "underlying-leaf-wwid\n",
        }
    )
    add_runtime_files(source)

    result = probe_collection_storage(
        "/db",
        collection_salt=SALT,
        data_paths=(data_path,),
        source=source,
    )
    assert result.media.confidence == "high"
    assert result.media.reuse_scope == "durable"
    assert "dm_uuid" in result.media.evidence
    assert "wwid" in result.media.evidence


def ceph_source(
    *,
    mount_point: str,
    remote_source: str = "client@cluster-uuid.cephfs=/",
    fsid: int | None = 4242,
) -> tuple[FakeProbeSource, str]:
    source = FakeProbeSource()
    data_path = f"{mount_point}/store/data/segment-1.dat"
    source.stats[data_path] = stat_device(0, 42)
    source.statvfs_values[data_path] = statvfs_with_fsid(fsid)
    escaped_mount = mount_point.replace(" ", "\\040")
    source.texts["/proc/self/mountinfo"] = "\n".join(
        (
            mount_line(
                9,
                0,
                42,
                "/volumes/project",
                escaped_mount,
                "ceph",
                remote_source,
                super_options="rw,mds_namespace=vectors,rsize=4194304",
            ),
            mount_line(10, 0, 28, "/", "/sys/fs/cgroup", "cgroup2", "cgroup"),
        )
    )
    add_runtime_files(source)
    return source, data_path


def test_ceph_identity_is_mountpoint_independent_and_redacted() -> None:
    source_a, path_a = ceph_source(mount_point="/mnt/ceph space")
    source_b, path_b = ceph_source(mount_point="/other/ceph")
    source_c, path_c = ceph_source(
        mount_point="/mnt/ceph space",
        remote_source="client@different-cluster.cephfs=/",
    )
    source_d, path_d = ceph_source(
        mount_point="/third/ceph",
        remote_source="another-user@cluster-uuid.cephfs=/",
    )

    first = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(path_a,),
        source=source_a,
    )
    moved = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(path_b,),
        source=source_b,
    )
    changed = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(path_c,),
        source=source_c,
    )
    other_user = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(path_d,),
        source=source_d,
    )

    assert first.media.kind == "network"
    assert first.media.confidence == "medium"
    assert first.media.reuse_scope == "durable"
    assert first.media.fingerprint == moved.media.fingerprint
    assert first.media.fingerprint == other_user.media.fingerprint
    assert first.media.fingerprint != changed.media.fingerprint
    serialized = json.dumps(first.as_redacted_dict())
    assert "cluster-uuid" not in serialized
    assert "/volumes/project" not in serialized
    assert "/mnt/ceph" not in serialized


def test_network_mount_without_fsid_is_not_durably_reused() -> None:
    source, data_path = ceph_source(mount_point="/ceph", fsid=None)
    result = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(data_path,),
        source=source,
    )
    assert result.media.kind == "network"
    assert result.media.confidence == "low"
    assert result.media.reuse_scope == "none"


def test_multiple_data_mounts_form_an_order_independent_composite() -> None:
    source = FakeProbeSource()
    first_path = "/ceph-a/store/data/segment-1.dat"
    second_path = "/ceph-b/store/data/segment-2.dat"
    source.stats[first_path] = stat_device(0, 42)
    source.stats[second_path] = stat_device(0, 43)
    source.statvfs_values[first_path] = statvfs_with_fsid(42)
    source.statvfs_values[second_path] = statvfs_with_fsid(43)
    source.texts["/proc/self/mountinfo"] = "\n".join(
        (
            mount_line(21, 0, 42, "/", "/ceph-a", "ceph", "client@cluster-a.fs=/"),
            mount_line(22, 0, 43, "/", "/ceph-b", "ceph", "client@cluster-b.fs=/"),
            mount_line(23, 0, 28, "/", "/sys/fs/cgroup", "cgroup2", "cgroup"),
        )
    )
    add_runtime_files(source)

    forward = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(first_path, second_path),
        source=source,
    )
    reversed_result = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(second_path, first_path),
        source=source,
    )
    assert forward.media.kind == "heterogeneous"
    assert forward.media.component_count == 2
    assert forward.media.reuse_scope == "durable"
    assert forward.media.fingerprint == reversed_result.media.fingerprint


def overlay_source(
    *,
    mount_point: str,
    mount_id: int,
    minor: int,
    fsid: int,
    machine_id: str,
    affinity: Sequence[int],
    seccomp_filters: int,
) -> tuple[FakeProbeSource, str]:
    source = FakeProbeSource()
    path = f"{mount_point}/store/data/segment-1.dat"
    source.stats[path] = stat_device(0, minor)
    source.statvfs_values[path] = statvfs_with_fsid(fsid)
    source.texts["/proc/self/mountinfo"] = "\n".join(
        (
            mount_line(
                mount_id,
                0,
                minor,
                "/",
                mount_point,
                "overlay",
                "overlay",
                super_options=(
                    f"rw,lowerdir={mount_point}/secret/lower,"
                    f"upperdir={mount_point}/secret/upper"
                ),
            ),
            mount_line(
                mount_id + 1,
                0,
                28,
                "/",
                "/sys/fs/cgroup",
                "cgroup2",
                "cgroup",
            ),
        )
    )
    add_runtime_files(source)
    source.texts["/etc/machine-id"] = machine_id + "\n"
    source.texts["/proc/self/status"] = (
        "Name:\tpython\nSeccomp:\t2\n"
        f"Seccomp_filters:\t{seccomp_filters}\n"
    )
    source.affinity_value = affinity
    return source, path


def test_overlay_identity_is_compatible_and_stable_across_pods() -> None:
    first_source, first_path = overlay_source(
        mount_point="/overlay-a",
        mount_id=11,
        minor=99,
        fsid=99,
        machine_id="0123456789abcdef0123456789abcdef",
        affinity=(0, 1, 2, 3),
        seccomp_filters=1,
    )
    second_source, second_path = overlay_source(
        mount_point="/workspace-b",
        mount_id=41,
        minor=173,
        fsid=8_181,
        machine_id="fedcba9876543210fedcba9876543210",
        affinity=(32, 33, 34, 35),
        seccomp_filters=7,
    )

    first = probe_collection_storage(
        "/overlay-a/store",
        collection_salt=SALT,
        data_paths=(first_path,),
        source=first_source,
    )
    second = probe_collection_storage(
        "/workspace-b/store",
        collection_salt=SALT,
        data_paths=(second_path,),
        source=second_source,
    )

    assert first.media.kind == second.media.kind == "overlay"
    assert first.media.confidence == second.media.confidence == "low"
    assert first.media.reuse_scope == second.media.reuse_scope == "compatible"
    assert first.media.fingerprint == second.media.fingerprint
    assert first.runtime.fingerprint == second.runtime.fingerprint
    serialized = json.dumps((first.as_redacted_dict(), second.as_redacted_dict()))
    for secret in (
        "/overlay-a",
        "/workspace-b",
        "0123456789abcdef0123456789abcdef",
        "fedcba9876543210fedcba9876543210",
    ):
        assert secret not in serialized


def test_local_network_and_overlay_media_domains_are_disjoint() -> None:
    local = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=local_partition_source(),
    )
    network_source, network_path = ceph_source(mount_point="/ceph")
    network = probe_collection_storage(
        "/unused",
        collection_salt=SALT,
        data_paths=(network_path,),
        source=network_source,
    )
    overlay_probe_source, overlay_path = overlay_source(
        mount_point="/overlay",
        mount_id=51,
        minor=199,
        fsid=919,
        machine_id="0123456789abcdef0123456789abcdef",
        affinity=(0, 1, 2, 3),
        seccomp_filters=1,
    )
    overlay = probe_collection_storage(
        "/overlay/store",
        collection_salt=SALT,
        data_paths=(overlay_path,),
        source=overlay_probe_source,
    )

    assert (local.media.kind, local.media.reuse_scope) == (
        "local_block",
        "durable",
    )
    assert (network.media.kind, network.media.reuse_scope) == (
        "network",
        "durable",
    )
    assert (overlay.media.kind, overlay.media.reuse_scope) == (
        "overlay",
        "compatible",
    )
    assert len(
        {
            local.media.fingerprint,
            network.media.fingerprint,
            overlay.media.fingerprint,
        }
    ) == 3


def runtime_source(*, cpu_max: str = "1000000 100000") -> FakeProbeSource:
    source = FakeProbeSource()
    source.texts["/proc/self/mountinfo"] = mount_line(
        20,
        0,
        28,
        "/",
        "/sys/fs/cgroup",
        "cgroup2",
        "cgroup",
    )
    add_runtime_files(source, cpu_max=cpu_max)
    source.stats["/collection"] = stat_device(0, 1)
    source.statvfs_values["/collection"] = statvfs_with_fsid(1)
    return source


def io_uring_policy(*, available: bool) -> dict[str, object]:
    return {
        "io_uring_available": available,
        "io_uring_probe_queue_depth": 32,
        "io_uring_probe_workers": 1,
    }


def test_runtime_digest_ignores_host_cpu_numbers_and_seccomp_filter_count() -> None:
    first_source = runtime_source()
    first_source.texts["/etc/machine-id"] = (
        "0123456789abcdef0123456789abcdef\n"
    )
    first_source.affinity_value = (0, 1, 2, 3)
    first_source.texts["/proc/self/status"] = (
        "Name:\tpython\nSeccomp:\t2\nSeccomp_filters:\t1\n"
    )
    other_pod_source = runtime_source()
    other_pod_source.texts["/etc/machine-id"] = (
        "fedcba9876543210fedcba9876543210\n"
    )
    other_pod_source.affinity_value = (32, 33, 34, 35)
    other_pod_source.texts["/proc/self/status"] = (
        "Name:\tpython\nSeccomp:\t2\nSeccomp_filters:\t9\n"
    )

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=first_source,
    )
    other_pod = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=other_pod_source,
    )

    assert first.runtime.affinity_cpu_count == 4
    assert other_pod.runtime.affinity_cpu_count == 4
    assert first.runtime.fingerprint == other_pod.runtime.fingerprint


def test_runtime_digest_ignores_nonbinding_host_affinity_size() -> None:
    smaller_host = runtime_source(cpu_max="1000000 100000")
    smaller_host.affinity_value = tuple(range(64))
    larger_host = runtime_source(cpu_max="100000 10000")
    larger_host.affinity_value = tuple(range(128))

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=smaller_host,
    )
    second = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=larger_host,
    )

    assert first.runtime.affinity_cpu_count == 64
    assert second.runtime.affinity_cpu_count == 128
    assert first.runtime.cpu_quota_limited is True
    assert first.runtime.fingerprint == second.runtime.fingerprint


def test_runtime_digest_tracks_capacity_capability_abi_and_architecture() -> None:
    first_source = runtime_source()
    first_source.affinity_value = (0, 1, 2, 3)
    fewer_cpus_source = runtime_source()
    fewer_cpus_source.affinity_value = (0, 1, 2)
    architecture_changed_source = runtime_source()
    architecture_changed_source.uname_value = (
        "Linux",
        "6.1.0-test",
        "aarch64",
    )

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=first_source,
    )
    affinity_changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=fewer_cpus_source,
    )
    quota_changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=runtime_source(cpu_max="800000 100000"),
    )
    capability_changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=False),
        source=first_source,
    )
    equivalent_quota = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=runtime_source(cpu_max="100000 10000"),
    )
    abi_changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        reader_abi="throwdb-reader-v3",
        io_uring_policy=io_uring_policy(available=True),
        source=first_source,
    )
    architecture_changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        io_uring_policy=io_uring_policy(available=True),
        source=architecture_changed_source,
    )

    assert first.runtime.affinity_cpu_count == 4
    assert first.runtime.cpu_quota_limited is True
    assert first.runtime.io_uring_disabled == 0
    assert first.runtime.fingerprint != affinity_changed.runtime.fingerprint
    assert first.runtime.fingerprint != quota_changed.runtime.fingerprint
    assert first.runtime.fingerprint != capability_changed.runtime.fingerprint
    assert first.runtime.fingerprint != abi_changed.runtime.fingerprint
    assert first.runtime.fingerprint != architecture_changed.runtime.fingerprint
    assert first.runtime.fingerprint == equivalent_quota.runtime.fingerprint


def test_runtime_digest_uses_effective_parent_cgroup_quota() -> None:
    parent_limited_source = runtime_source(cpu_max="1000000 100000")
    parent_limited_source.texts["/sys/fs/cgroup/tenant/cpu.max"] = "500000 100000\n"
    parent_changed_source = runtime_source(cpu_max="1000000 100000")
    parent_changed_source.texts["/sys/fs/cgroup/tenant/cpu.max"] = "600000 100000\n"

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=parent_limited_source,
    )
    changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=parent_changed_source,
    )
    assert first.runtime.cpu_quota_limited is True
    assert first.runtime.fingerprint != changed.runtime.fingerprint


def test_runtime_digest_ignores_seccomp_filter_count() -> None:
    first_source = runtime_source()
    first_source.texts["/proc/self/status"] = (
        "Name:\tpython\nSeccomp:\t2\nSeccomp_filters:\t1\n"
    )
    changed_source = runtime_source()
    changed_source.texts["/proc/self/status"] = (
        "Name:\tpython\nSeccomp:\t2\nSeccomp_filters:\t2\n"
    )

    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=first_source,
    )
    changed = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=changed_source,
    )
    assert first.runtime.fingerprint == changed.runtime.fingerprint


def test_non_linux_fallback_is_never_durable() -> None:
    source = FakeProbeSource(platform="darwin")
    source.uname_value = ("Darwin", "25.0.0", "arm64")
    source.stats["/collection"] = stat_device(1, 2)
    source.statvfs_values["/collection"] = statvfs_with_fsid(123)
    result = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )
    assert result.media.kind == "local_volume"
    assert result.media.confidence == "low"
    assert result.media.reuse_scope == "none"


def test_bytes_and_pathlike_bytes_are_decoded_for_probe_source() -> None:
    source = local_partition_source()
    expected = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )
    from_bytes = probe_collection_storage(
        BytesPath(b"/collection"),
        collection_salt=SALT,
        data_paths=(BytesPath(b"/collection/data/segment-1.dat"),),
        source=source,
    )
    assert from_bytes.media.fingerprint == expected.media.fingerprint

    from_path = probe_collection_storage(
        Path("/collection"),
        collection_salt=SALT,
        source=source,
    )
    assert from_path.media.fingerprint == expected.media.fingerprint


@pytest.mark.parametrize("salt", [b"", b"too-short"])
def test_collection_salt_must_be_at_least_128_bits(salt: bytes) -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        probe_collection_storage("/collection", collection_salt=salt, source=runtime_source())


def test_collection_salt_must_be_bytes() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        probe_collection_storage(  # type: ignore[arg-type]
            "/collection",
            collection_salt="not-bytes",
            source=runtime_source(),
        )


def test_collection_salt_domains_media_and_runtime_fingerprints() -> None:
    source = local_partition_source()
    first = probe_collection_storage(
        "/collection",
        collection_salt=SALT,
        source=source,
    )
    second = probe_collection_storage(
        "/collection",
        collection_salt=b"fedcba9876543210",
        source=source,
    )
    assert first.media.fingerprint != second.media.fingerprint
    assert first.runtime.fingerprint != second.runtime.fingerprint
