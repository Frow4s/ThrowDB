"""Redacted storage and runtime identity probing.

This module deliberately separates media identity from runtime compatibility.
The raw inputs used to recognize a mount (mount source, filesystem UUIDs,
device serials, and sysfs paths) are only ever fed to a salted digest.  They
are not retained by any returned object.

The implementation is Linux-first.  Other platforms return a conservative,
non-durable identity so that automatic tuning can safely fall back instead of
reusing a profile for an unrecognized device.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import fnmatch
import hashlib
import math
import os
import platform as _platform
import posixpath
import re
import stat as _stat
import sys
from typing import Callable, Mapping, Protocol, Sequence


_FINGERPRINT_ALGORITHM = "sha256-length-framed-v1"
_MEDIA_SCHEMA_VERSION = 2
_RUNTIME_SCHEMA_VERSION = 2
_DEFAULT_READER_ABI = "throwdb-reader-v2"
_MINIMUM_SALT_BYTES = 16
_DEFAULT_DISCOVERY_ENTRY_LIMIT = 4096

_MOUNT_ESCAPE_RE = re.compile(r"\\([0-7]{3})")
_MACHINE_ID_RE = re.compile(r"[0-9a-fA-F]{32}")
_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
    }
)
_VIRTUAL_FILESYSTEMS = frozenset(
    {
        "autofs",
        "bpf",
        "cgroup",
        "cgroup2",
        "configfs",
        "debugfs",
        "devpts",
        "devtmpfs",
        "efivarfs",
        "hugetlbfs",
        "mqueue",
        "overlay",
        "proc",
        "pstore",
        "ramfs",
        "securityfs",
        "sysfs",
        "tmpfs",
        "tracefs",
    }
)
_PHYSICAL_TOKEN_KINDS = frozenset(
    {
        "eui",
        "eui64",
        "nguid",
        "serial",
        "wwid",
    }
)
_NON_UNIQUE_TOKEN_VALUES = frozenset(
    {
        "0",
        "n/a",
        "na",
        "none",
        "not specified",
        "unknown",
    }
)

_SYSFS_TOKEN_FILES: tuple[tuple[str, str], ...] = (
    ("wwid", "wwid"),
    ("wwid", "device/wwid"),
    ("nguid", "nguid"),
    ("nguid", "device/nguid"),
    ("eui", "eui"),
    ("eui", "device/eui"),
    ("eui64", "eui64"),
    ("eui64", "device/eui64"),
    ("serial", "serial"),
    ("serial", "device/serial"),
    ("dm_uuid", "dm/uuid"),
    ("md_uuid", "md/uuid"),
)
_UDEV_TOKEN_KEYS = {
    "ID_FS_UUID": "filesystem_uuid",
    "ID_PART_ENTRY_UUID": "partition_uuid",
    "ID_SERIAL": "serial",
    "ID_SERIAL_SHORT": "serial",
    "ID_WWN": "wwid",
    "ID_WWN_WITH_EXTENSION": "wwid",
}
_NETWORK_OPTION_KEYS = frozenset(
    {
        # Backend identity / namespace.
        "fs",
        "fsid",
        "mds_namespace",
        # Options with a material effect on a measured reader profile.
        "actimeo",
        "cache",
        "hard",
        "ms_mode",
        "multichannel",
        "nconnect",
        "proto",
        "rasize",
        "recover_session",
        "retrans",
        "rsize",
        "soft",
        "timeo",
        "vers",
        "wsize",
    }
)
_LOCAL_OPTION_KEYS = frozenset({"dax", "dirsync", "ro", "rw", "sync"})


class ProbeSource(Protocol):
    """Injectable operating-system facade used by the probe and its tests."""

    platform: str

    def stat(self, path: str) -> object: ...

    def statvfs(self, path: str) -> object: ...

    def realpath(self, path: str) -> str: ...

    def glob(self, pattern: str) -> Sequence[str]: ...

    def glob_limited(
        self, pattern: str, *, limit: int
    ) -> tuple[Sequence[str], bool, int]: ...

    def read_text(self, path: str) -> str: ...

    def listdir(self, path: str) -> Sequence[str]: ...

    def is_dir(self, path: str) -> bool: ...

    def uname(self) -> tuple[str, str, str]: ...

    def cpu_affinity(self) -> Sequence[int] | None: ...


class SystemProbeSource:
    """Default :class:`ProbeSource` backed by unprivileged system calls."""

    platform = sys.platform

    def stat(self, path: str) -> object:
        return os.stat(path)

    def statvfs(self, path: str) -> object:
        return os.statvfs(path)

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def glob(self, pattern: str) -> Sequence[str]:
        directory, name_pattern = os.path.split(pattern)
        try:
            with os.scandir(directory or ".") as entries:
                return tuple(
                    entry.path
                    for entry in entries
                    if fnmatch.fnmatchcase(entry.name, name_pattern)
                )
        except FileNotFoundError:
            return ()

    def glob_limited(
        self,
        pattern: str,
        *,
        limit: int,
    ) -> tuple[Sequence[str], bool, int]:
        if limit == 0:
            return (), False, 0
        directory, name_pattern = os.path.split(pattern)
        try:
            with os.scandir(directory or ".") as entries:
                matches: list[str] = []
                scanned = 0
                iterator = iter(entries)
                while scanned < limit:
                    try:
                        entry = next(iterator)
                    except StopIteration:
                        return tuple(matches), True, scanned
                    scanned += 1
                    if not fnmatch.fnmatchcase(entry.name, name_pattern):
                        continue
                    matches.append(entry.path)
                # Conservatively avoid one extra next() merely to prove EOF.
                return tuple(matches), False, scanned
        except FileNotFoundError:
            return (), True, 0

    def read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="surrogateescape") as handle:
            return handle.read()

    def listdir(self, path: str) -> Sequence[str]:
        return os.listdir(path)

    def is_dir(self, path: str) -> bool:
        return _stat.S_ISDIR(os.stat(path).st_mode)

    def uname(self) -> tuple[str, str, str]:
        value = _platform.uname()
        return value.system, value.release, value.machine

    def cpu_affinity(self) -> Sequence[int] | None:
        getter: Callable[[int], object] | None = getattr(os, "sched_getaffinity", None)
        if getter is None:
            return None
        try:
            return tuple(sorted(int(cpu) for cpu in getter(0)))
        except OSError:
            return None


@dataclass(frozen=True)
class MediaIdentity:
    """A path-free media fingerprint suitable for a tuning-profile key."""

    fingerprint: str
    algorithm: str
    schema_version: int
    kind: str
    confidence: str
    reuse_scope: str
    evidence: tuple[str, ...]
    component_count: int

    def as_key_dict(self) -> dict[str, object]:
        """Return only stable fields that participate in profile selection.

        Diagnostic evidence is intentionally excluded: it may become richer
        after a kernel or probe update without changing the storage target.
        """

        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "reuse_scope": self.reuse_scope,
        }

    def as_redacted_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "confidence": self.confidence,
            "reuse_scope": self.reuse_scope,
            "evidence": list(self.evidence),
            "component_count": self.component_count,
        }


@dataclass(frozen=True)
class RuntimeIdentity:
    """A redacted digest of assumptions that affect reader calibration."""

    fingerprint: str
    algorithm: str
    schema_version: int
    evidence: tuple[str, ...]
    affinity_cpu_count: int | None
    cpu_quota_limited: bool | None
    io_uring_disabled: int | None

    def as_key_dict(self) -> dict[str, object]:
        """Return the pod-independent runtime compatibility key."""

        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
        }

    def as_redacted_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "evidence": list(self.evidence),
            "affinity_cpu_count": self.affinity_cpu_count,
            "cpu_quota_limited": self.cpu_quota_limited,
            "io_uring_disabled": self.io_uring_disabled,
        }


@dataclass(frozen=True)
class StorageProbeResult:
    """Redacted media and runtime identities for one collection open."""

    media: MediaIdentity
    runtime: RuntimeIdentity
    probed_path_count: int
    used_root_fallback: bool

    def as_redacted_dict(self) -> dict[str, object]:
        return {
            "media": self.media.as_redacted_dict(),
            "runtime": self.runtime.as_redacted_dict(),
            "probed_path_count": self.probed_path_count,
            "used_root_fallback": self.used_root_fallback,
        }


@dataclass(frozen=True)
class _MountInfo:
    mount_id: int
    parent_id: int
    major: int
    minor: int
    root: str
    mount_point: str
    mount_options: tuple[str, ...]
    optional_fields: tuple[str, ...]
    fs_type: str
    source: str
    super_options: tuple[str, ...]


@dataclass(frozen=True)
class _ComponentIdentity:
    digest: str
    kind: str
    confidence: str
    reuse_scope: str
    evidence: tuple[str, ...]


def _mountinfo_unescape(value: str) -> str:
    return _MOUNT_ESCAPE_RE.sub(
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _parse_mountinfo(text: str) -> tuple[_MountInfo, ...]:
    """Parse Linux mountinfo without exposing it through the public result."""

    result: list[_MountInfo] = []
    for raw_line in text.splitlines():
        fields = raw_line.split()
        try:
            separator = fields.index("-")
            if separator < 6 or len(fields) < separator + 4:
                continue
            raw_major, raw_minor = fields[2].split(":", 1)
            result.append(
                _MountInfo(
                    mount_id=int(fields[0]),
                    parent_id=int(fields[1]),
                    major=int(raw_major),
                    minor=int(raw_minor),
                    root=_mountinfo_unescape(fields[3]),
                    mount_point=_mountinfo_unescape(fields[4]),
                    mount_options=tuple(fields[5].split(",")),
                    optional_fields=tuple(fields[6:separator]),
                    fs_type=_mountinfo_unescape(fields[separator + 1]),
                    source=_mountinfo_unescape(fields[separator + 2]),
                    super_options=tuple(fields[separator + 3].split(",")),
                )
            )
        except (ValueError, IndexError):
            # A best-effort identity probe must not make Collection.open fail
            # because one unrelated mountinfo entry was malformed.
            continue
    return tuple(result)


def _path_is_beneath(path: str, mount_point: str) -> bool:
    normalized_path = posixpath.normpath(path)
    normalized_mount = posixpath.normpath(mount_point)
    if normalized_mount == "/":
        return normalized_path.startswith("/")
    return normalized_path == normalized_mount or normalized_path.startswith(
        normalized_mount.rstrip("/") + "/"
    )


def _select_mount(
    entries: Sequence[_MountInfo],
    path: str,
    major: int,
    minor: int,
) -> _MountInfo | None:
    """Select the deepest covering mount whose device exactly matches stat."""

    candidates = [
        entry
        for entry in entries
        if entry.major == major
        and entry.minor == minor
        and _path_is_beneath(path, entry.mount_point)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: len(posixpath.normpath(entry.mount_point)))


def _as_bytes(value: str) -> bytes:
    return value.encode("utf-8", errors="surrogateescape")


def _frame(value: bytes) -> bytes:
    if len(value) > 0xFFFFFFFF:
        raise ValueError("identity field is too large")
    return len(value).to_bytes(4, "big") + value


def _canonical_digest(
    salt: bytes,
    domain: str,
    fields: Sequence[tuple[str, bytes]],
) -> str:
    payload = bytearray()
    payload.extend(_frame(_as_bytes(domain)))
    payload.extend(_frame(salt))
    for key, value in sorted(fields, key=lambda item: (item[0], item[1])):
        payload.extend(_frame(_as_bytes(key)))
        payload.extend(_frame(value))
    return hashlib.sha256(payload).hexdigest()


def _validate_salt(value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("collection_salt must be bytes")
    if len(value) < _MINIMUM_SALT_BYTES:
        raise ValueError("collection_salt must contain at least 16 bytes")
    return value


def _read_optional(source: ProbeSource, path: str) -> str | None:
    try:
        value = source.read_text(path).strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _read_host_identity(source: ProbeSource) -> bytes | None:
    """Return a normalized machine-id for hashing, never for publication."""

    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        value = _read_optional(source, path)
        if value is None or _MACHINE_ID_RE.fullmatch(value) is None:
            continue
        normalized = value.lower()
        if any(character != "0" for character in normalized):
            return normalized.encode("ascii")
    return None


def _listdir_optional(source: ProbeSource, path: str) -> tuple[str, ...]:
    try:
        return tuple(source.listdir(path))
    except OSError:
        return ()


def _is_dir_optional(source: ProbeSource, path: str) -> bool:
    try:
        return source.is_dir(path)
    except OSError:
        return False


def _parse_dev(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        raw_major, raw_minor = value.split(":", 1)
        return int(raw_major), int(raw_minor)
    except (TypeError, ValueError):
        return None


def _udev_tokens(source: ProbeSource, major: int, minor: int) -> set[tuple[str, str]]:
    text = _read_optional(source, f"/run/udev/data/b{major}:{minor}")
    if text is None:
        return set()
    result: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if not line.startswith("E:") or "=" not in line:
            continue
        raw_key, raw_value = line[2:].split("=", 1)
        token_kind = _UDEV_TOKEN_KEYS.get(raw_key)
        value = raw_value.strip()
        if token_kind is not None and _token_is_useful(value):
            result.add((token_kind, value))
    return result


def _token_is_useful(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized or normalized in _NON_UNIQUE_TOKEN_VALUES:
        return False
    compact = normalized.replace("-", "").replace(":", "")
    return bool(compact) and any(character != "0" for character in compact)


def _node_tokens(source: ProbeSource, resolved_path: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for token_kind, relative_path in _SYSFS_TOKEN_FILES:
        value = _read_optional(source, posixpath.join(resolved_path, relative_path))
        if value is not None and _token_is_useful(value):
            result.add((token_kind, value))
    return result


def _collect_block_fields(
    source: ProbeSource,
    major: int,
    minor: int,
) -> tuple[list[tuple[str, bytes]], set[str], bool]:
    """Collect stable block IDs while recursively following dm/md slaves."""

    pending = [(major, minor)]
    visited: set[tuple[int, int]] = set()
    tokens: set[tuple[str, str]] = set()
    topology: set[tuple[str, str]] = set()

    while pending and len(visited) < 128:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        raw_path = f"/sys/dev/block/{current[0]}:{current[1]}"
        try:
            if not _is_dir_optional(source, raw_path):
                continue
            resolved = source.realpath(raw_path)
        except OSError:
            continue

        tokens.update(_node_tokens(source, resolved))
        tokens.update(_udev_tokens(source, current[0], current[1]))

        partition = _read_optional(source, posixpath.join(resolved, "partition"))
        if partition is not None:
            topology.add(("partition", partition))
            for name in ("start", "size"):
                value = _read_optional(source, posixpath.join(resolved, name))
                if value is not None:
                    topology.add((f"partition_{name}", value))
            parent = posixpath.dirname(resolved)
            parent_dev = _parse_dev(_read_optional(source, posixpath.join(parent, "dev")))
            if parent_dev is not None and parent_dev not in visited:
                pending.append(parent_dev)
            else:
                # Some restricted sysfs views expose the parent identifiers but
                # omit its dev file. They are still useful with partition geometry.
                tokens.update(_node_tokens(source, parent))

        for child_name in _listdir_optional(source, posixpath.join(resolved, "slaves")):
            child_link = posixpath.join(resolved, "slaves", child_name)
            try:
                child = source.realpath(child_link)
            except OSError:
                continue
            child_dev = _parse_dev(_read_optional(source, posixpath.join(child, "dev")))
            if child_dev is not None and child_dev not in visited:
                pending.append(child_dev)

    fields: list[tuple[str, bytes]] = []
    evidence: set[str] = set()
    for token_kind, token_value in sorted(tokens):
        fields.append((f"block_token:{token_kind}", _as_bytes(token_value)))
        evidence.add(token_kind)
    for topology_kind, topology_value in sorted(topology):
        fields.append((f"block_topology:{topology_kind}", _as_bytes(topology_value)))
        evidence.add(topology_kind)

    durable = any(kind in _PHYSICAL_TOKEN_KINDS for kind, _ in tokens)
    return fields, evidence, durable


def _fsid_bytes(statvfs_value: object) -> bytes | None:
    raw = getattr(statvfs_value, "f_fsid", None)
    if raw is None:
        return None
    try:
        numeric = int(raw) & 0xFFFFFFFFFFFFFFFF
    except (TypeError, ValueError):
        return None
    if numeric == 0:
        return None
    return numeric.to_bytes(8, "big")


def _is_network_filesystem(fs_type: str) -> bool:
    normalized = fs_type.lower()
    return (
        normalized in _NETWORK_FILESYSTEMS
        or normalized.startswith("fuse.ceph")
        or normalized.startswith("fuse.sshfs")
    )


def _network_source_identity(fs_type: str, source: str) -> str:
    """Remove credential identity while preserving the remote namespace."""

    value = source.strip()
    normalized_type = fs_type.lower()
    if (
        normalized_type == "ceph"
        or normalized_type.startswith("fuse.ceph")
    ) and "@" in value:
        # Ceph v2 device syntax is name@fsid.fs_name=/subtree.  ``name`` is
        # the CephX principal, not storage identity; retaining it would split
        # one shared filesystem into a profile per user.
        _, remainder = value.split("@", 1)
        return "@" + remainder
    return value


def _is_virtual_filesystem(fs_type: str) -> bool:
    normalized = fs_type.lower()
    return normalized in _VIRTUAL_FILESYSTEMS or normalized.startswith("fuse.")


def _option_fields(
    mount: _MountInfo,
    *,
    network: bool,
) -> list[tuple[str, bytes]]:
    allowed = _NETWORK_OPTION_KEYS if network else _LOCAL_OPTION_KEYS
    result: list[tuple[str, bytes]] = []
    for raw_option in (*mount.mount_options, *mount.super_options):
        raw_key, separator, raw_value = raw_option.partition("=")
        key = raw_key.strip().lower()
        if key not in allowed:
            continue
        value = raw_value if separator else "present"
        result.append((f"mount_option:{key}", _as_bytes(value)))
    return result


def _safe_statvfs(source: ProbeSource, path: str) -> object | None:
    try:
        return source.statvfs(path)
    except OSError:
        return None


def _stat_device_numbers(stat_value: object) -> tuple[int, int]:
    """Return device numbers, with explicit fields available to test probes."""

    explicit_major = getattr(stat_value, "st_dev_major", None)
    explicit_minor = getattr(stat_value, "st_dev_minor", None)
    if explicit_major is not None and explicit_minor is not None:
        result = int(explicit_major), int(explicit_minor)
    else:
        device = int(getattr(stat_value, "st_dev"))
        result = os.major(device), os.minor(device)
    if result[0] < 0 or result[1] < 0:
        raise ValueError("device numbers cannot be negative")
    return result


def _probe_linux_component(
    path: str,
    salt: bytes,
    source: ProbeSource,
    mounts: Sequence[_MountInfo],
    host_identity: bytes | None,
    cache: dict[tuple[int, int, int | None, str], _ComponentIdentity],
) -> _ComponentIdentity:
    try:
        resolved_path = source.realpath(path)
        stat_value = source.stat(resolved_path)
        major, minor = _stat_device_numbers(stat_value)
    except (OSError, OverflowError, TypeError, ValueError):
        digest = _canonical_digest(
            salt,
            "throwdb-media-component-v2",
            (("kind", b"unknown"), ("probe", b"stat_failed")),
        )
        return _ComponentIdentity(
            digest=digest,
            kind="unknown",
            confidence="none",
            reuse_scope="none",
            evidence=("stat_failed",),
        )

    mount = _select_mount(mounts, resolved_path, major, minor)
    fs_type = mount.fs_type.lower() if mount is not None else "unknown"
    cache_key = (
        major,
        minor,
        mount.mount_id if mount is not None else None,
        fs_type,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    def finish(component: _ComponentIdentity) -> _ComponentIdentity:
        cache[cache_key] = component
        return component

    statvfs_value = _safe_statvfs(source, resolved_path)
    fsid = _fsid_bytes(statvfs_value) if statvfs_value is not None else None
    fields: list[tuple[str, bytes]] = [
        ("kind", b"linux"),
        ("fs_type", _as_bytes(fs_type)),
    ]
    evidence: set[str] = set()
    if fsid is not None:
        fields.append(("filesystem_id", fsid))
        evidence.add("filesystem_id")

    if mount is not None and _is_network_filesystem(fs_type):
        source_value = _network_source_identity(fs_type, mount.source)
        source_is_useful = source_value not in {"", "-", "none"}
        if source_is_useful:
            fields.append(("remote_source", _as_bytes(source_value)))
            # Remote mount root distinguishes exports/subvolumes without
            # exposing their path in the returned value.
            fields.append(("remote_root", _as_bytes(mount.root)))
            evidence.add("remote_identity")
        fields.extend(_option_fields(mount, network=True))
        durable = source_is_useful and fsid is not None
        digest = _canonical_digest(salt, "throwdb-media-component-v2", fields)
        return finish(
            _ComponentIdentity(
                digest=digest,
                kind="network",
                confidence="medium" if durable else "low",
                reuse_scope="durable" if durable else "none",
                evidence=tuple(sorted(evidence | {"network_filesystem"})),
            )
        )

    if mount is not None and _is_virtual_filesystem(fs_type):
        # An overlay exposes neither the backing device nor its remote/local
        # provenance.  fsid, mount ID, mount point and host identity are pod
        # details, so an exact media identity would be both unstable and
        # misleading.  Use a deliberately coarse, disjoint compatibility
        # domain and let the runtime/layout keys guard profile reuse.
        compatible_fields = [
            ("kind", b"virtual"),
            ("fs_type", _as_bytes(fs_type)),
        ]
        digest = _canonical_digest(
            salt,
            "throwdb-media-component-v2",
            compatible_fields,
        )
        return finish(
            _ComponentIdentity(
                digest=digest,
                kind="overlay" if fs_type == "overlay" else "virtual",
                confidence="low",
                reuse_scope="compatible",
                evidence=tuple(sorted(evidence | {"virtual_filesystem"})),
            )
        )

    block_fields, block_evidence, durable = _collect_block_fields(
        source,
        major,
        minor,
    )
    fields.extend(block_fields)
    evidence.update(block_evidence)
    if mount is not None:
        fields.extend(_option_fields(mount, network=False))
    if durable:
        return finish(
            _ComponentIdentity(
                digest=_canonical_digest(
                    salt,
                    "throwdb-media-component-v2",
                    fields,
                ),
                kind="local_block",
                confidence="high",
                reuse_scope="durable",
                evidence=tuple(sorted(evidence | {"stable_block_identity"})),
            )
        )

    host_scoped = (
        host_identity is not None
        and fsid is not None
        and mount is not None
        and fs_type != "unknown"
        and major != 0
    )
    if host_scoped:
        fields.extend(
            (
                ("host_identity", host_identity),
                ("host_device_major", major.to_bytes(8, "big")),
                ("host_device_minor", minor.to_bytes(8, "big")),
            )
        )
        return finish(
            _ComponentIdentity(
                digest=_canonical_digest(
                    salt,
                    "throwdb-media-component-v2",
                    fields,
                ),
                kind="local_block",
                confidence="medium",
                reuse_scope="host",
                evidence=tuple(
                    sorted(
                        evidence
                        | {
                            "host_identity",
                            "host_scoped_device",
                            "stable_block_identity_missing",
                        }
                    )
                ),
            )
        )

    return finish(
        _ComponentIdentity(
            digest=_canonical_digest(
                salt,
                "throwdb-media-component-v2",
                fields,
            ),
            kind=(
                "local_block"
                if _is_dir_optional(source, f"/sys/dev/block/{major}:{minor}")
                else "unknown"
            ),
            confidence="low",
            reuse_scope="none",
            evidence=tuple(sorted(evidence | {"stable_block_identity_missing"})),
        )
    )


def _probe_fallback_component(
    path: str,
    salt: bytes,
    source: ProbeSource,
) -> _ComponentIdentity:
    """Conservative macOS/other-platform fallback with no durable reuse."""

    fields: list[tuple[str, bytes]] = [
        ("platform", _as_bytes(source.platform)),
        ("kind", b"fallback"),
    ]
    evidence: set[str] = {"platform_fallback"}
    try:
        resolved_path = source.realpath(path)
        stat_value = source.stat(resolved_path)
        device = int(getattr(stat_value, "st_dev")) & 0xFFFFFFFFFFFFFFFF
        fields.append(("session_device", device.to_bytes(8, "big")))
        statvfs_value = _safe_statvfs(source, resolved_path)
        fsid = _fsid_bytes(statvfs_value) if statvfs_value is not None else None
        if fsid is not None:
            fields.append(("filesystem_id", fsid))
            evidence.add("filesystem_id")
    except (OSError, TypeError, ValueError):
        evidence.add("stat_failed")
    digest = _canonical_digest(salt, "throwdb-media-component-v2", fields)
    return _ComponentIdentity(
        digest=digest,
        kind="local_volume" if source.platform == "darwin" else "unknown",
        confidence="low",
        reuse_scope="none",
        evidence=tuple(sorted(evidence)),
    )


def _read_mounts(source: ProbeSource) -> tuple[_MountInfo, ...]:
    text = _read_optional(source, "/proc/self/mountinfo")
    return () if text is None else _parse_mountinfo(text)


def _find_cgroup2_cpu_max(
    source: ProbeSource,
    mounts: Sequence[_MountInfo],
) -> str | None:
    cgroup_text = _read_optional(source, "/proc/self/cgroup")
    cgroup_path: str | None = None
    if cgroup_text is not None:
        for line in cgroup_text.splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
                cgroup_path = posixpath.normpath("/" + fields[2].lstrip("/"))
                break

    candidates: list[str] = []
    if cgroup_path is not None:
        for mount in mounts:
            if mount.fs_type != "cgroup2" or not _path_is_beneath(
                cgroup_path,
                mount.root,
            ):
                continue
            relative = posixpath.relpath(cgroup_path, mount.root)
            base = (
                mount.mount_point
                if relative == "."
                else posixpath.join(mount.mount_point, relative)
            )
            while _path_is_beneath(base, mount.mount_point):
                candidates.append(posixpath.join(base, "cpu.max"))
                if posixpath.normpath(base) == posixpath.normpath(mount.mount_point):
                    break
                base = posixpath.dirname(base)
    candidates.append("/sys/fs/cgroup/cpu.max")

    finite_limits: list[Fraction] = []
    saw_unlimited = False
    for candidate in dict.fromkeys(candidates):
        value = _read_optional(source, candidate)
        if value is None:
            continue
        fields = value.split()
        if len(fields) < 2:
            continue
        quota, period = fields[0], fields[1]
        if quota != "max" and not quota.isdigit():
            continue
        if not period.isdigit() or int(period) == 0:
            continue
        if quota == "max":
            saw_unlimited = True
        else:
            finite_limits.append(Fraction(int(quota), int(period)))
    if finite_limits:
        effective = min(finite_limits)
        return f"{effective.numerator}/{effective.denominator}"
    return "max" if saw_unlimited else None


def _policy_bytes(value: object) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("io_uring_policy cannot contain a non-finite float")
        return b"f" + value.hex().encode("ascii")
    if isinstance(value, str):
        return b"s" + _frame(_as_bytes(value))
    if isinstance(value, Mapping):
        framed = bytearray(b"m")
        items: list[tuple[str, object]] = []
        for key, item_value in value.items():
            if not isinstance(key, str):
                raise TypeError("io_uring_policy keys must be strings")
            items.append((key, item_value))
        for key, item_value in sorted(items):
            framed.extend(_frame(_as_bytes(key)))
            framed.extend(_frame(_policy_bytes(item_value)))
        return bytes(framed)
    if isinstance(value, (list, tuple)):
        framed = bytearray(b"l")
        for item in value:
            framed.extend(_frame(_policy_bytes(item)))
        return bytes(framed)
    raise TypeError(f"unsupported io_uring_policy value {type(value).__name__}")


def _linux_security_fields(
    source: ProbeSource,
) -> list[tuple[str, bytes]]:
    status = _read_optional(source, "/proc/self/status")
    if status is None:
        return []
    wanted = {
        "Seccomp": "seccomp_mode",
        "Seccomp_filters": "seccomp_filters",
    }
    result: list[tuple[str, bytes]] = []
    for line in status.splitlines():
        raw_key, separator, raw_value = line.partition(":")
        key = wanted.get(raw_key)
        value = raw_value.strip()
        if separator and key is not None and value.isdigit():
            result.append((key, str(int(value)).encode("ascii")))
    return result


def _runtime_identity(
    salt: bytes,
    source: ProbeSource,
    mounts: Sequence[_MountInfo],
    _host_identity: bytes | None,
    reader_abi: str,
    io_uring_policy: Mapping[str, object] | str,
) -> RuntimeIdentity:
    if not isinstance(reader_abi, str) or not reader_abi:
        raise ValueError("reader_abi must be a non-empty string")
    system, kernel_release, architecture = source.uname()
    fields: list[tuple[str, bytes]] = [
        ("reader_abi", _as_bytes(reader_abi)),
        ("platform", _as_bytes(source.platform)),
        ("system", _as_bytes(system)),
        ("kernel_release", _as_bytes(kernel_release)),
        ("architecture", _as_bytes(architecture)),
        ("io_uring_policy", _policy_bytes(io_uring_policy)),
    ]
    evidence: set[str] = {
        "architecture",
        "io_uring_policy",
        "kernel",
        "reader_abi",
    }
    # Host identity is deliberately ignored here.  It belongs to the
    # media probe's host-scoped fallback, not to runtime compatibility: the
    # same workload can legitimately move between pods and hosts.

    affinity_count: int | None = None
    try:
        raw_affinity = source.cpu_affinity()
    except OSError:
        raw_affinity = None
    if raw_affinity is not None:
        affinity = tuple(sorted({int(cpu) for cpu in raw_affinity}))
        if any(cpu < 0 for cpu in affinity):
            raise ValueError("CPU affinity cannot contain a negative CPU ID")
        affinity_count = len(affinity)
        evidence.add("cpu_affinity")

    cpu_max = (
        _find_cgroup2_cpu_max(source, mounts)
        if source.platform.startswith("linux")
        else None
    )
    cpu_quota_limited: bool | None = None
    if cpu_max is not None:
        fields.append(("cgroup_cpu_max", cpu_max.encode("ascii")))
        cpu_quota_limited = cpu_max != "max"
        evidence.add("cgroup_cpu_max")
    if affinity_count is not None:
        # Affinity is performance-relevant only when it is the binding CPU
        # limit.  Pods with the same quota often inherit every CPU number from
        # different-sized nodes; hashing 64 versus 128 in that case would
        # create needless profiles even though both are capped at (say) 10.
        affinity_is_binding = True
        if cpu_max not in {None, "max"}:
            affinity_is_binding = Fraction(cpu_max) > affinity_count
        if affinity_is_binding:
            fields.append(
                ("cpu_affinity_limit", str(affinity_count).encode("ascii"))
            )

    io_uring_disabled: int | None = None
    if source.platform.startswith("linux"):
        security_fields = _linux_security_fields(source)
        if security_fields:
            # Seccomp filter counts commonly differ between otherwise
            # equivalent pod launches.  The actual io_uring capability probe
            # above is the performance-relevant contract.
            evidence.add("seccomp_policy")
        raw_disabled = _read_optional(source, "/proc/sys/kernel/io_uring_disabled")
        if raw_disabled in {"0", "1", "2"}:
            io_uring_disabled = int(raw_disabled)
            fields.append(("io_uring_disabled", raw_disabled.encode("ascii")))
            evidence.add("io_uring_disabled")

    return RuntimeIdentity(
        fingerprint=_canonical_digest(salt, "throwdb-runtime-v2", fields),
        algorithm=_FINGERPRINT_ALGORITHM,
        schema_version=_RUNTIME_SCHEMA_VERSION,
        evidence=tuple(sorted(evidence)),
        affinity_cpu_count=affinity_count,
        cpu_quota_limited=cpu_quota_limited,
        io_uring_disabled=io_uring_disabled,
    )


def _combine_components(
    salt: bytes,
    components: Sequence[_ComponentIdentity],
    *,
    discovery_complete: bool,
) -> MediaIdentity:
    unique = {component.digest: component for component in components}
    ordered = [unique[digest] for digest in sorted(unique)]
    kinds = {component.kind for component in ordered}
    kind = (
        next(iter(kinds))
        if len(ordered) == 1 and len(kinds) == 1
        else "heterogeneous"
    )

    confidence_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    confidence = min(
        (component.confidence for component in ordered),
        key=lambda value: confidence_order[value],
    )
    scopes = {component.reuse_scope for component in ordered}
    if ordered and scopes == {"durable"}:
        reuse_scope = "durable"
    elif ordered and scopes <= {"durable", "host"} and "host" in scopes:
        reuse_scope = "host"
    elif ordered and "none" not in scopes and "compatible" in scopes:
        reuse_scope = "compatible"
    else:
        reuse_scope = "none"
    evidence_items = {
        item for component in ordered for item in component.evidence
    }
    if not discovery_complete:
        reuse_scope = "none"
        confidence = "low" if confidence in {"medium", "high"} else confidence
        evidence_items.add("discovery_incomplete")
    evidence = tuple(sorted(evidence_items))
    fields: list[tuple[str, bytes]] = [
        ("kind", _as_bytes(kind)),
        ("discovery_complete", b"1" if discovery_complete else b"0"),
    ]
    fields.extend(
        ("component", bytes.fromhex(component.digest)) for component in ordered
    )
    return MediaIdentity(
        fingerprint=_canonical_digest(salt, "throwdb-media-v2", fields),
        algorithm=_FINGERPRINT_ALGORITHM,
        schema_version=_MEDIA_SCHEMA_VERSION,
        kind=kind,
        confidence=confidence,
        reuse_scope=reuse_scope,
        evidence=evidence,
        component_count=len(ordered),
    )


def _glob_limited(
    source: ProbeSource,
    pattern: str,
    *,
    limit: int,
) -> tuple[tuple[str, ...], bool, int]:
    """Enumerate at most ``limit`` matches and report whether that was all.

    The fallback keeps third-party deterministic ProbeSource fixtures working;
    the production source implements early-stop directly on ``scandir``.
    """

    if limit == 0:
        return (), False, 0
    bounded = getattr(source, "glob_limited", None)
    if callable(bounded):
        response = bounded(pattern, limit=limit)
        if len(response) == 3:
            matches, complete, scanned = response
        else:
            # Compatibility for deterministic third-party probe fixtures that
            # implemented the initial two-value experimental hook.
            matches, complete = response
            scanned = len(matches)
        result = tuple(matches)
        if (
            isinstance(scanned, bool)
            or not isinstance(scanned, int)
            or scanned < 0
            or scanned > limit
            or len(result) > scanned
        ):
            return result[:limit], False, limit
        return result, bool(complete), scanned
    matches = tuple(source.glob(pattern))
    if len(matches) <= limit:
        return matches, True, len(matches)
    return matches[:limit], False, limit


def probe_collection_storage(
    collection_path: os.PathLike[str] | os.PathLike[bytes] | str | bytes,
    *,
    collection_salt: bytes,
    data_paths: Sequence[
        os.PathLike[str] | os.PathLike[bytes] | str | bytes
    ]
    | None = None,
    reader_abi: str = _DEFAULT_READER_ABI,
    io_uring_policy: Mapping[str, object] | str = "auto",
    discovery_entry_limit: int = _DEFAULT_DISCOVERY_ENTRY_LIMIT,
    source: ProbeSource | None = None,
) -> StorageProbeResult:
    """Return redacted media/runtime fingerprints for a collection.

    ``collection_salt`` is owned by the collection's tuning registry and must
    travel with it.  Default discovery probes the collection root plus legacy
    base, delta, partitioned-segment, and WAL payloads: any of those can serve
    embedding bytes. Explicit ``data_paths`` replace discovery and are
    primarily useful for native-fd integration and deterministic tests.
    Discovery stops after
    ``discovery_entry_limit`` directory entries and fails closed instead of reusing a
    potentially incomplete media identity.
    """

    salt = _validate_salt(collection_salt)
    if (
        isinstance(discovery_entry_limit, bool)
        or not isinstance(discovery_entry_limit, int)
        or not 1 <= discovery_entry_limit <= 65_536
    ):
        raise ValueError("discovery_entry_limit must be between 1 and 65536")
    active_source: ProbeSource = source if source is not None else SystemProbeSource()
    root = os.fsdecode(os.fspath(collection_path))

    used_root_fallback = False
    discovery_complete = True
    if data_paths is None:
        discovered: set[str] = set()
        scanned_entry_count = 0
        directories_and_patterns = (
            (os.path.join(root, "data"), "segment-*.dat"),
            (os.path.join(root, "deltas"), "delta-*.dat"),
            (os.path.join(root, "segments"), "segment-*.dat"),
            (os.path.join(root, "wal"), "*.wal"),
        )
        for directory, name_pattern in directories_and_patterns:
            try:
                if active_source.is_dir(directory):
                    discovered.add(directory)
            except FileNotFoundError:
                pass
            except OSError:
                discovery_complete = False
            try:
                remaining = discovery_entry_limit - scanned_entry_count
                matches, complete, scanned = _glob_limited(
                    active_source,
                    os.path.join(directory, name_pattern),
                    limit=remaining,
                )
                decoded = tuple(
                    os.fsdecode(os.fspath(item)) for item in matches
                )
                discovered.update(decoded)
                scanned_entry_count += scanned
                if not complete:
                    discovery_complete = False
                    break
            except FileNotFoundError:
                pass
            except OSError:
                # Never select a root-only profile when a nested mount could
                # have been hidden by an unreadable payload directory.
                discovery_complete = False
        used_root_fallback = not discovered
        paths = (root, *sorted(discovered))
    else:
        paths = tuple(os.fsdecode(os.fspath(item)) for item in data_paths)
        if not paths:
            paths = (root,)
            used_root_fallback = True

    is_linux = active_source.platform.startswith("linux")
    mounts = _read_mounts(active_source) if is_linux else ()
    host_identity = _read_host_identity(active_source) if is_linux else None
    if is_linux:
        component_cache: dict[
            tuple[int, int, int | None, str],
            _ComponentIdentity,
        ] = {}
        components = tuple(
            _probe_linux_component(
                path,
                salt,
                active_source,
                mounts,
                host_identity,
                component_cache,
            )
            for path in paths
        )
    else:
        components = tuple(
            _probe_fallback_component(path, salt, active_source) for path in paths
        )

    media = _combine_components(
        salt,
        components,
        discovery_complete=discovery_complete,
    )
    runtime = _runtime_identity(
        salt,
        active_source,
        mounts,
        host_identity,
        reader_abi,
        io_uring_policy,
    )
    return StorageProbeResult(
        media=media,
        runtime=runtime,
        probed_path_count=len(paths),
        used_root_fallback=used_root_fallback,
    )


__all__ = [
    "MediaIdentity",
    "ProbeSource",
    "RuntimeIdentity",
    "StorageProbeResult",
    "SystemProbeSource",
    "probe_collection_storage",
]
