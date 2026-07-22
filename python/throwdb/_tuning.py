"""Crash-safe persisted reader tuning profiles.

The registry is advisory: a missing or damaged file must never make collection
data unreadable.  Ordinary loads are read-only; only the explicit mutation
helpers create ``tuning/`` or its lock file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import operator
import os
import secrets
import stat
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Iterator, Literal


_FORMAT: Final = "throwdb-reader-profiles"
_VERSION: Final = 1
_CHECKSUM_ALGORITHM: Final = "sha256"
_MAX_REGISTRY_BYTES: Final = 1 << 20
_MAX_PROFILES: Final = 64
_MAX_FINGERPRINT_FIELDS: Final = 64
_MAX_FINGERPRINT_BYTES: Final = 64 << 10
_UINT64_MAX: Final = (1 << 64) - 1
_TUNING_DIRECTORY_NAME: Final = "tuning"
_REGISTRY_FILENAME: Final = "reader-profiles.json"
_LOCK_FILENAME: Final = "reader-profiles.lock"
_COLLECTION_LOCK_DIRECTORY: Final = "locks"
_CHECKPOINT_LOCK_FILENAME: Final = "tuning.lock"

READER_DEFAULTS: Final[dict[str, object]] = {
    "max_read_range_bytes": 1 << 20,
    "max_gap_bytes": 0,
    "io_workers": 0,
    "io_queue_capacity": 0,
    "io_ranges_per_task": 8,
    "io_backend": "thread_pool",
    "io_uring_queue_depth": 32,
    "io_uring_workers": 1,
}
CANONICAL_READER_OPTIONS: Final = frozenset(READER_DEFAULTS)
_ALIASES: Final[dict[str, str]] = {
    "backend": "io_backend",
    "queue_capacity": "io_queue_capacity",
    "ranges_per_task": "io_ranges_per_task",
    "queue_depth": "io_uring_queue_depth",
    "read_range_bytes": "max_read_range_bytes",
    "gap_bytes": "max_gap_bytes",
}


class TuningRegistryError(ValueError):
    """Persisted tuning registry is malformed or cannot be published."""


def _integer(
    value: object, name: str, minimum: int, maximum: int
) -> int:
    value_type = type(value)
    if isinstance(value, bool) or (
        value_type.__name__ in {"bool", "bool_"}
        and value_type.__module__.partition(".")[0] == "numpy"
    ):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(
            f"{name} must be an integer, not {type(value).__name__}"
        ) from error
    if not minimum <= integer <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return integer


def normalize_reader_options(
    options: Mapping[str, object],
    *,
    persisted: bool = False,
    record_size: int = 1,
) -> dict[str, object]:
    """Return a complete canonical and validated ReaderOptions mapping.

    Manual input may be partial and use public aliases. Persisted input must
    contain exactly the complete canonical key set, preventing a newer binary
    from silently assigning different defaults to an old profile.
    """

    if not isinstance(options, Mapping):
        raise TypeError("reader options must be a mapping")
    record_size = _integer(record_size, "record_size", 1, _UINT64_MAX)
    raw = dict(options)
    for key in raw:
        if not isinstance(key, str):
            raise TypeError("reader option keys must be strings")

    if persisted:
        if set(raw) != CANONICAL_READER_OPTIONS:
            missing = sorted(CANONICAL_READER_OPTIONS - set(raw))
            unknown = sorted(set(raw) - CANONICAL_READER_OPTIONS)
            details = []
            if missing:
                details.append(f"missing {missing[0]!r}")
            if unknown:
                details.append(f"unknown {unknown[0]!r}")
            raise ValueError(
                "persisted reader options must contain exactly the canonical "
                f"key set ({'; '.join(details)})"
            )
        effective = raw
    else:
        allowed = CANONICAL_READER_OPTIONS | set(_ALIASES) | {"workers"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown tuning option {unknown[0]!r}")
        canonical: dict[str, object] = {
            key: value for key, value in raw.items()
            if key in CANONICAL_READER_OPTIONS
        }
        for alias, destination in _ALIASES.items():
            if alias not in raw:
                continue
            if destination in canonical:
                raise ValueError(
                    f"use only one of {alias!r} and {destination!r}"
                )
            canonical[destination] = raw[alias]
        backend = canonical.get("io_backend", READER_DEFAULTS["io_backend"])
        if "workers" in raw:
            if "io_workers" in canonical or "io_uring_workers" in canonical:
                raise ValueError(
                    "use 'workers' or a backend-specific worker option, not both"
                )
            destination = (
                "io_uring_workers" if backend == "io_uring" else "io_workers"
            )
            canonical[destination] = raw["workers"]
        effective = {**READER_DEFAULTS, **canonical}

    backend = effective["io_backend"]
    if not isinstance(backend, str):
        raise TypeError("io_backend must be a string")
    if backend not in {"thread_pool", "io_uring"}:
        raise ValueError("io_backend must be 'thread_pool' or 'io_uring'")

    normalized: dict[str, object] = {
        "max_read_range_bytes": _integer(
            effective["max_read_range_bytes"],
            "max_read_range_bytes", record_size, _UINT64_MAX,
        ),
        "max_gap_bytes": _integer(
            effective["max_gap_bytes"], "max_gap_bytes", 0, _UINT64_MAX
        ),
        "io_workers": _integer(
            effective["io_workers"], "io_workers", 0, 1024
        ),
        "io_queue_capacity": _integer(
            effective["io_queue_capacity"],
            "io_queue_capacity", 0, 1 << 20,
        ),
        "io_ranges_per_task": _integer(
            effective["io_ranges_per_task"],
            "io_ranges_per_task", 1, 1 << 20,
        ),
        "io_backend": backend,
        "io_uring_queue_depth": _integer(
            effective["io_uring_queue_depth"],
            "io_uring_queue_depth", 1, 4096,
        ),
        "io_uring_workers": _integer(
            effective["io_uring_workers"], "io_uring_workers", 1, 64
        ),
    }
    if backend == "io_uring":
        if normalized["io_workers"] != 0 or normalized["io_queue_capacity"] != 0:
            raise ValueError(
                "io_workers and io_queue_capacity must both be 0 when "
                "io_backend is 'io_uring'"
            )
        if normalized["io_uring_workers"] > normalized["io_uring_queue_depth"]:
            raise ValueError(
                "io_uring_queue_depth must be at least io_uring_workers"
            )
    elif normalized["io_workers"] == 0 and normalized["io_queue_capacity"] != 0:
        raise ValueError("io_queue_capacity must be 0 when io_workers is 0")
    return normalized


def _hex_digest(value: str, name: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _freeze(mapping: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class RegistryKey:
    device_digest: str
    runtime_digest: str
    layout_digest: str

    def __post_init__(self) -> None:
        _hex_digest(self.device_digest, "device_digest")
        _hex_digest(self.runtime_digest, "runtime_digest")
        _hex_digest(self.layout_digest, "layout_digest")

    @classmethod
    def from_mappings(
        cls,
        device: Mapping[str, object],
        runtime: Mapping[str, object],
        layout: Mapping[str, object],
        *,
        salt: str,
    ) -> RegistryKey:
        salt_bytes = bytes.fromhex(_hex_digest(salt, "salt", 32))
        return cls(
            _mapping_digest(device, salt_bytes, "device"),
            _mapping_digest(runtime, salt_bytes, "runtime"),
            _mapping_digest(layout, salt_bytes, "layout"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "device_digest": self.device_digest,
            "runtime_digest": self.runtime_digest,
            "layout_digest": self.layout_digest,
        }


@dataclass(frozen=True, slots=True)
class ReaderProfile:
    key: RegistryKey
    device: Mapping[str, object]
    runtime: Mapping[str, object]
    layout: Mapping[str, object]
    options: Mapping[str, object]
    source: str
    created_at_ns: int
    updated_at_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _freeze(self.device))
        object.__setattr__(self, "runtime", _freeze(self.runtime))
        object.__setattr__(self, "layout", _freeze(self.layout))
        object.__setattr__(self, "options", _freeze(self.options))

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key.as_dict(),
            "device": dict(self.device),
            "runtime": dict(self.runtime),
            "layout": dict(self.layout),
            "options": dict(self.options),
            "source": self.source,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
        }


@dataclass(frozen=True, slots=True)
class TuningRegistry:
    version: int
    generation: int
    salt: str
    profiles: tuple[ReaderProfile, ...]

    def key_for(
        self,
        device: Mapping[str, object],
        runtime: Mapping[str, object],
        layout: Mapping[str, object],
    ) -> RegistryKey:
        return RegistryKey.from_mappings(
            device, runtime, layout, salt=self.salt
        )

    def select(self, key: RegistryKey) -> ReaderProfile | None:
        return next((profile for profile in self.profiles if profile.key == key), None)

    def list_profiles(self) -> list[ReaderProfile]:
        return list(self.profiles)


@dataclass(frozen=True, slots=True)
class RegistryLoadResult:
    status: Literal["ok", "missing", "corrupt"]
    registry: TuningRegistry | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    status: Literal["selected", "missing_registry", "no_match", "corrupt"]
    key: RegistryKey | None = None
    profile: ReaderProfile | None = None
    error: str | None = None


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise TypeError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("ascii")


def _validated_fingerprint_mapping(
    value: Mapping[str, object], name: str
) -> tuple[dict[str, object], bytes]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} fingerprint must be a mapping")
    copied = dict(value)
    if len(copied) > _MAX_FINGERPRINT_FIELDS:
        raise ValueError(f"{name} fingerprint has too many fields")
    for key in copied:
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 128
            or any(ord(character) < 32 for character in key)
        ):
            raise ValueError(f"{name} fingerprint contains an invalid field name")
    encoded = _canonical_bytes(copied)
    if len(encoded) > _MAX_FINGERPRINT_BYTES:
        raise ValueError(f"{name} fingerprint is too large")
    return copied, encoded


def _salted_digest(salt: bytes, domain: str, encoded: bytes) -> str:
    checksum = hashlib.sha256()
    checksum.update(b"throwdb-tuning-v1\0")
    checksum.update(salt)
    checksum.update(domain.encode("ascii"))
    checksum.update(b"\0")
    checksum.update(encoded)
    return checksum.hexdigest()


def _mapping_digest(
    value: Mapping[str, object], salt: bytes, domain: str
) -> str:
    _, encoded = _validated_fingerprint_mapping(value, domain)
    return _salted_digest(salt, domain, encoded)


def _redact_mapping(
    value: Mapping[str, object], salt: bytes, domain: str
) -> dict[str, str]:
    copied, _ = _validated_fingerprint_mapping(value, domain)
    return {
        key: "sha256:"
        + _salted_digest(salt, f"{domain}.{key}", _canonical_bytes(copied[key]))
        for key in sorted(copied)
    }


def _path(value: object) -> Path:
    return Path(os.fsdecode(os.fspath(value)))


def _reject_constant(value: str) -> object:
    raise TuningRegistryError(f"registry contains non-finite number {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TuningRegistryError(f"registry contains duplicate key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: object, expected: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TuningRegistryError(f"{context} must be an object")
    if set(value) != expected:
        raise TuningRegistryError(f"{context} has an unsupported schema")
    return value


def _decode_redacted(value: object, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > _MAX_FINGERPRINT_FIELDS:
        raise TuningRegistryError(f"{context} must be a bounded object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not isinstance(digest, str):
            raise TuningRegistryError(f"{context} fields must be strings")
        if not digest.startswith("sha256:"):
            raise TuningRegistryError(f"{context} contains an invalid digest")
        _hex_digest(digest[7:], f"{context}.{key}")
        result[key] = digest
    return result


def _decode_profile(value: object) -> ReaderProfile:
    profile = _exact_keys(
        value,
        {
            "key", "device", "runtime", "layout", "options", "source",
            "created_at_ns", "updated_at_ns",
        },
        "reader profile",
    )
    key_value = _exact_keys(
        profile["key"],
        {"device_digest", "runtime_digest", "layout_digest"},
        "reader profile key",
    )
    try:
        key = RegistryKey(
            key_value["device_digest"],  # type: ignore[arg-type]
            key_value["runtime_digest"],  # type: ignore[arg-type]
            key_value["layout_digest"],  # type: ignore[arg-type]
        )
        options = normalize_reader_options(
            profile["options"], persisted=True  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise TuningRegistryError(f"invalid reader profile: {error}") from error
    source = profile["source"]
    if (
        not isinstance(source, str)
        or not 1 <= len(source) <= 64
        or any(ord(character) < 32 for character in source)
    ):
        raise TuningRegistryError("reader profile source is invalid")
    try:
        created = _integer(profile["created_at_ns"], "created_at_ns", 0, _UINT64_MAX)
        updated = _integer(profile["updated_at_ns"], "updated_at_ns", created, _UINT64_MAX)
    except (TypeError, ValueError) as error:
        raise TuningRegistryError(f"invalid reader profile timestamp: {error}") from error
    return ReaderProfile(
        key=key,
        device=_decode_redacted(profile["device"], "device fingerprint"),
        runtime=_decode_redacted(profile["runtime"], "runtime fingerprint"),
        layout=_decode_redacted(profile["layout"], "layout fingerprint"),
        options=options,
        source=source,
        created_at_ns=created,
        updated_at_ns=updated,
    )


def _payload(registry: TuningRegistry) -> dict[str, object]:
    return {
        "schema_version": registry.version,
        "generation": registry.generation,
        "salt": registry.salt,
        "profiles": [profile.as_dict() for profile in registry.profiles],
    }


def _encode_registry(registry: TuningRegistry) -> bytes:
    payload = _payload(registry)
    payload_bytes = _canonical_bytes(payload)
    envelope = {
        "format": _FORMAT,
        "version": _VERSION,
        "checksum": {
            "algorithm": _CHECKSUM_ALGORITHM,
            "value": hashlib.sha256(payload_bytes).hexdigest(),
        },
        "payload": payload,
    }
    encoded = _canonical_bytes(envelope) + b"\n"
    if len(encoded) > _MAX_REGISTRY_BYTES:
        raise TuningRegistryError("tuning registry exceeds the 1 MiB limit")
    return encoded


def _decode_registry(encoded: bytes) -> TuningRegistry:
    if not encoded or len(encoded) > _MAX_REGISTRY_BYTES:
        raise TuningRegistryError("tuning registry has an invalid size")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TuningRegistryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TuningRegistryError(f"cannot decode tuning registry: {error}") from error
    envelope = _exact_keys(
        value, {"format", "version", "checksum", "payload"}, "registry envelope"
    )
    if envelope["format"] != _FORMAT or envelope["version"] != _VERSION:
        raise TuningRegistryError("tuning registry has an unsupported format version")
    checksum = _exact_keys(
        envelope["checksum"], {"algorithm", "value"}, "registry checksum"
    )
    if checksum["algorithm"] != _CHECKSUM_ALGORITHM:
        raise TuningRegistryError("tuning registry uses an unsupported checksum")
    checksum_value = checksum["value"]
    try:
        _hex_digest(checksum_value, "registry checksum")  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TuningRegistryError(str(error)) from error
    payload = _exact_keys(
        envelope["payload"],
        {"schema_version", "generation", "salt", "profiles"},
        "registry payload",
    )
    actual_checksum = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if checksum_value != actual_checksum:
        raise TuningRegistryError("tuning registry checksum mismatch")
    if payload["schema_version"] != _VERSION:
        raise TuningRegistryError("registry payload version is unsupported")
    try:
        generation = _integer(payload["generation"], "generation", 1, _UINT64_MAX)
        salt = _hex_digest(payload["salt"], "salt", 32)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TuningRegistryError(str(error)) from error
    profile_values = payload["profiles"]
    if not isinstance(profile_values, list) or len(profile_values) > _MAX_PROFILES:
        raise TuningRegistryError("registry contains an invalid profile list")
    profiles = tuple(_decode_profile(profile) for profile in profile_values)
    keys = [profile.key for profile in profiles]
    if len(set(keys)) != len(keys):
        raise TuningRegistryError("registry contains duplicate profile keys")
    if tuple(sorted(keys, key=_key_tuple)) != tuple(keys):
        raise TuningRegistryError("registry profiles are not in canonical order")
    return TuningRegistry(_VERSION, generation, salt, profiles)


def _key_tuple(key: RegistryKey) -> tuple[str, str, str]:
    return key.device_digest, key.runtime_digest, key.layout_digest


def _read_registry_file(tuning_descriptor: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            _REGISTRY_FILENAME,
            flags,
            dir_fd=tuning_descriptor,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        raise TuningRegistryError(f"cannot open tuning registry: {error}") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise TuningRegistryError("tuning registry is not a regular file")
        if not getattr(os, "O_NOFOLLOW", 0):
            current = os.stat(
                _REGISTRY_FILENAME,
                dir_fd=tuning_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != (status.st_dev, status.st_ino)
            ):
                raise TuningRegistryError("tuning registry is not a regular file")
        if status.st_size <= 0 or status.st_size > _MAX_REGISTRY_BYTES:
            raise TuningRegistryError("tuning registry has an invalid size")
        chunks: list[bytes] = []
        remaining = status.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 << 10))
            if not chunk:
                raise TuningRegistryError("tuning registry is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TuningRegistryError("tuning registry changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _corrupt_load_result(
    error: BaseException,
    *,
    strict: bool,
) -> RegistryLoadResult:
    corruption = (
        error if isinstance(error, TuningRegistryError)
        else TuningRegistryError(str(error))
    )
    if strict:
        raise corruption
    return RegistryLoadResult("corrupt", None, str(corruption))


def _load_registry_from_directory(
    tuning_descriptor: int,
    *,
    strict: bool,
) -> RegistryLoadResult:
    try:
        encoded = _read_registry_file(tuning_descriptor)
        return RegistryLoadResult("ok", _decode_registry(encoded), None)
    except FileNotFoundError:
        return RegistryLoadResult("missing", None, None)
    except (
        OSError,
        TuningRegistryError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        return _corrupt_load_result(error, strict=strict)


def _open_collection_directory(store: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(store, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise TuningRegistryError(
            f"cannot open collection directory {store}: {error}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise TuningRegistryError(
                f"collection path is not a directory: {store}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_lock_directory(store_descriptor: int) -> int:
    """Open the collection lock directory without following a replacement.

    Real collections already contain ``locks/``.  Creating it here retains the
    historical property that the low-level registry helpers also work against
    an otherwise empty directory in tests and tooling.  The coordination file
    lives outside ``tuning/`` so a checkpoint of a collection with no tuning
    metadata does not bootstrap an advisory registry accidentally.
    """

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    created = False
    try:
        os.mkdir(_COLLECTION_LOCK_DIRECTORY, mode=0o755, dir_fd=store_descriptor)
        created = True
    except FileExistsError:
        pass
    except OSError as error:
        raise TuningRegistryError(
            f"cannot create collection lock directory: {error}"
        ) from error
    try:
        descriptor = os.open(
            _COLLECTION_LOCK_DIRECTORY,
            flags,
            dir_fd=store_descriptor,
        )
    except OSError as error:
        try:
            current = os.stat(
                _COLLECTION_LOCK_DIRECTORY,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            current = None
        if current is not None and not stat.S_ISDIR(current.st_mode):
            raise TuningRegistryError(
                "collection lock path is not a real directory"
            ) from error
        raise TuningRegistryError(
            f"cannot open collection lock directory: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise TuningRegistryError(
                "collection lock path is not a real directory"
            )
        if not nofollow:
            current = os.stat(
                _COLLECTION_LOCK_DIRECTORY,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise TuningRegistryError(
                    "collection lock path is not a real directory"
                )
        if created:
            os.fsync(store_descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_lock_file(directory_descriptor: int, name: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | nofollow
    descriptor = -1
    created = False
    open_error: OSError | None = None
    # O_CREAT without O_EXCL can transiently return ENOENT when several
    # threads race to create the same dirfd-relative file on some POSIX
    # filesystems (including APFS). Split create/open and tolerate the narrow
    # winner-removal race instead.
    for _ in range(4):
        try:
            descriptor = os.open(
                name,
                flags | os.O_CREAT | os.O_EXCL,
                0o644,
                dir_fd=directory_descriptor,
            )
            created = True
            break
        except FileExistsError:
            try:
                descriptor = os.open(name, flags, dir_fd=directory_descriptor)
                break
            except FileNotFoundError as error:
                open_error = error
                continue
            except OSError as error:
                open_error = error
                break
        except FileNotFoundError as error:
            open_error = error
            continue
        except OSError as error:
            open_error = error
            break
    if descriptor < 0:
        assert open_error is not None
        raise TuningRegistryError(
            f"cannot open tuning registry lock: {open_error}"
        ) from open_error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise TuningRegistryError(
                "tuning registry lock is not a regular file"
            )
        if not nofollow:
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise TuningRegistryError(
                    "tuning registry lock is not a regular file"
                )
        if created:
            os.fsync(directory_descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def tuning_checkpoint_lock(
    store_path: object,
    *,
    expected_store_identity: tuple[int, int] | None = None,
    fail_if_locked: bool = False,
) -> Iterator[tuple[int, int]]:
    """Serialize registry mutations with complete-collection checkpoints.

    The global order is ``writer.lock`` (when a lifecycle owner has one), then
    ``locks/tuning.lock``, then ``tuning/reader-profiles.lock``.  Registry-only
    callers never acquire ``writer.lock`` after either tuning lock, so a backup
    that already owns the writer lease cannot form a lock cycle.

    The checkpoint barrier is deliberately stored under the existing lock
    namespace rather than ``tuning/``.  Acquiring it therefore does not create
    a registry directory or registry payload when optional tuning metadata is
    absent.
    """

    if expected_store_identity is not None:
        expected_store_identity = _store_identity(expected_store_identity)
    if not isinstance(fail_if_locked, bool):
        raise TypeError("fail_if_locked must be a bool")
    store = _path(store_path)
    store_descriptor = -1
    locks_descriptor = -1
    lock_descriptor = -1
    locked = False
    try:
        store_descriptor = _open_collection_directory(store)
        opened_store = os.fstat(store_descriptor)
        locked_store_identity = (opened_store.st_dev, opened_store.st_ino)
        if (
            expected_store_identity is not None
            and locked_store_identity != expected_store_identity
        ):
            raise TuningRegistryError(
                "collection root changed while preparing the tuning registry; "
                "refusing to publish the profile"
            )
        locks_descriptor = _open_lock_directory(store_descriptor)
        lock_descriptor = _open_or_create_lock_file(
            locks_descriptor, _CHECKPOINT_LOCK_FILENAME
        )
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if fail_if_locked else 0)
        try:
            fcntl.flock(lock_descriptor, operation)
        except BlockingIOError as error:
            raise TuningRegistryError(
                "tuning registry is busy; retry calibration later"
            ) from error
        locked = True
        yield locked_store_identity
    except TuningRegistryError:
        raise
    except OSError as error:
        raise TuningRegistryError(
            f"cannot acquire tuning checkpoint lock: {error}"
        ) from error
    finally:
        try:
            if locked:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            for descriptor in (
                lock_descriptor,
                locks_descriptor,
                store_descriptor,
            ):
                if descriptor >= 0:
                    os.close(descriptor)


def _store_identity(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise TypeError("expected_store_identity must be a (device, inode) tuple")
    return (
        _integer(value[0], "expected_store_identity.device", 0, _UINT64_MAX),
        _integer(value[1], "expected_store_identity.inode", 0, _UINT64_MAX),
    )


def _open_tuning_directory(store_descriptor: int) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    try:
        descriptor = os.open(
            _TUNING_DIRECTORY_NAME,
            flags,
            dir_fd=store_descriptor,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        try:
            current = os.stat(
                _TUNING_DIRECTORY_NAME,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            current = None
        if current is not None and not stat.S_ISDIR(current.st_mode):
            raise TuningRegistryError("tuning path is not a real directory") from error
        raise TuningRegistryError(
            f"cannot open tuning directory: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise TuningRegistryError("tuning path is not a real directory")
        if not nofollow:
            current = os.stat(
                _TUNING_DIRECTORY_NAME,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise TuningRegistryError("tuning path is not a real directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _opened_tuning_directory(
    store_path: object,
    *,
    create: bool,
    expected_store_identity: tuple[int, int] | None = None,
) -> Iterator[int]:
    store = _path(store_path)
    try:
        store_descriptor = _open_collection_directory(store)
    except FileNotFoundError as error:
        if create:
            raise TuningRegistryError(
                f"collection path is not a directory: {store}"
            ) from error
        raise
    tuning_descriptor = -1
    try:
        if expected_store_identity is not None:
            opened_store = os.fstat(store_descriptor)
            if (opened_store.st_dev, opened_store.st_ino) != expected_store_identity:
                raise TuningRegistryError(
                    "collection root changed while preparing the tuning registry; "
                    "refusing to publish the profile"
                )
        if create:
            try:
                os.mkdir(
                    _TUNING_DIRECTORY_NAME,
                    mode=0o755,
                    dir_fd=store_descriptor,
                )
            except FileExistsError:
                pass
            except OSError as error:
                raise TuningRegistryError(
                    f"cannot create tuning directory: {error}"
                ) from error
        try:
            tuning_descriptor = _open_tuning_directory(store_descriptor)
        except FileNotFoundError as error:
            if create:
                raise TuningRegistryError(
                    "tuning directory disappeared while opening it"
                ) from error
            raise
        if create:
            # Always sync the collection directory, not only when this process
            # won mkdir(). Another creator may have crashed before syncing the
            # new directory entry while leaving it visible in the page cache.
            try:
                os.fsync(store_descriptor)
            except OSError as error:
                raise TuningRegistryError(
                    f"cannot sync collection directory: {error}"
                ) from error
        yield tuning_descriptor
    finally:
        if tuning_descriptor >= 0:
            os.close(tuning_descriptor)
        os.close(store_descriptor)


def load_registry(store_path: object, *, strict: bool = False) -> RegistryLoadResult:
    """Load without creating directories, a lock file, or registry metadata."""

    try:
        with _opened_tuning_directory(store_path, create=False) as tuning:
            return _load_registry_from_directory(tuning, strict=strict)
    except FileNotFoundError:
        return RegistryLoadResult("missing", None, None)
    except (
        OSError,
        TuningRegistryError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        return _corrupt_load_result(error, strict=strict)


@contextmanager
def _exclusive_writer(
    store_path: object,
    *,
    expected_store_identity: tuple[int, int] | None = None,
    fail_if_locked: bool = False,
) -> Iterator[int]:
    # Every mutation takes the collection-level checkpoint barrier first.
    # A backup already owns writer.lock before taking that barrier; calibration
    # also owns writer.lock before publication.  No registry-only path takes
    # writer.lock after either tuning lock, preserving the documented order.
    with tuning_checkpoint_lock(
        store_path,
        expected_store_identity=expected_store_identity,
        fail_if_locked=fail_if_locked,
    ) as locked_store_identity:
        with _opened_tuning_directory(
            store_path,
            create=True,
            # Reopening the user-supplied path must still reach the root whose
            # anchored locks/tuning.lock inode we own.  This closes a rename +
            # symlink swap between the two dirfd-based namespaces even when a
            # caller did not provide an optimistic identity explicitly.
            expected_store_identity=locked_store_identity,
        ) as tuning:
            descriptor = _open_or_create_lock_file(tuning, _LOCK_FILENAME)
            locked = False
            try:
                operation = fcntl.LOCK_EX | (
                    fcntl.LOCK_NB if fail_if_locked else 0
                )
                try:
                    fcntl.flock(descriptor, operation)
                except BlockingIOError as error:
                    raise TuningRegistryError(
                        "tuning registry is busy; retry calibration later"
                    ) from error
                locked = True
                yield tuning
            finally:
                try:
                    if locked:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)


def _publish_registry(tuning: int, registry: TuningRegistry) -> None:
    encoded = _encode_registry(registry)
    temporary = (
        f"reader-profiles.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o644, dir_fd=tuning)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TuningRegistryError("cannot write tuning registry")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            destination_status = os.stat(
                _REGISTRY_FILENAME,
                dir_fd=tuning,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_status = None
        if (
            destination_status is not None
            and not stat.S_ISREG(destination_status.st_mode)
        ):
            raise TuningRegistryError("tuning registry path is not a regular file")
        os.replace(
            temporary,
            _REGISTRY_FILENAME,
            src_dir_fd=tuning,
            dst_dir_fd=tuning,
        )
        os.fsync(tuning)
    except OSError as error:
        raise TuningRegistryError(f"cannot publish tuning registry: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=tuning)
        except FileNotFoundError:
            pass


def _empty_registry(*, initial_salt: str | None = None) -> TuningRegistry:
    salt = (
        secrets.token_hex(16)
        if initial_salt is None
        else _hex_digest(initial_salt, "initial_salt", 32)
    )
    return TuningRegistry(_VERSION, 0, salt, ())


def _load_for_mutation(
    tuning: int,
    *,
    initial_salt: str | None = None,
) -> TuningRegistry:
    loaded = _load_registry_from_directory(tuning, strict=True)
    return (
        loaded.registry
        if loaded.registry is not None
        else _empty_registry(initial_salt=initial_salt)
    )


def create_registry(store_path: object) -> TuningRegistry:
    with _exclusive_writer(store_path) as tuning:
        loaded = _load_registry_from_directory(tuning, strict=True)
        if loaded.registry is not None:
            return loaded.registry
        registry = TuningRegistry(_VERSION, 1, secrets.token_hex(16), ())
        _publish_registry(tuning, registry)
        return registry


def select_profile(
    store_path: object,
    *,
    device: Mapping[str, object],
    runtime: Mapping[str, object],
    layout: Mapping[str, object],
    record_size: int = 1,
    strict: bool = False,
) -> ProfileSelection:
    loaded = load_registry(store_path, strict=strict)
    if loaded.status == "missing":
        return ProfileSelection("missing_registry")
    if loaded.status == "corrupt":
        return ProfileSelection("corrupt", error=loaded.error)
    assert loaded.registry is not None
    key = loaded.registry.key_for(device, runtime, layout)
    profile = loaded.registry.select(key)
    if profile is None:
        return ProfileSelection("no_match", key=key)
    try:
        normalize_reader_options(
            profile.options, persisted=True, record_size=record_size
        )
    except (TypeError, ValueError) as error:
        registry_error = TuningRegistryError(
            f"selected tuning profile is invalid for this layout: {error}"
        )
        if strict:
            raise registry_error
        return ProfileSelection("corrupt", key=key, error=str(registry_error))
    return ProfileSelection("selected", key=key, profile=profile)


def _timestamp(value: int | None) -> int:
    return _integer(
        time.time_ns() if value is None else value,
        "timestamp_ns", 0, _UINT64_MAX,
    )


def upsert_profile(
    store_path: object,
    *,
    device: Mapping[str, object],
    runtime: Mapping[str, object],
    layout: Mapping[str, object],
    options: Mapping[str, object],
    source: str = "calibrated",
    timestamp_ns: int | None = None,
    record_size: int = 1,
    expected_salt: str | None = None,
    expected_generation: int | None = None,
    initial_salt: str | None = None,
    expected_store_identity: tuple[int, int] | None = None,
    fail_if_locked: bool = False,
) -> ReaderProfile:
    normalized = normalize_reader_options(options, record_size=record_size)
    if (
        not isinstance(source, str)
        or not 1 <= len(source) <= 64
        or any(ord(character) < 32 for character in source)
    ):
        raise ValueError("source must be a non-empty string of at most 64 characters")
    timestamp = None if timestamp_ns is None else _timestamp(timestamp_ns)
    if expected_salt is not None:
        expected_salt = _hex_digest(expected_salt, "expected_salt", 32)
    if expected_generation is not None:
        expected_generation = _integer(
            expected_generation,
            "expected_generation",
            0,
            _UINT64_MAX,
        )
    if initial_salt is not None:
        initial_salt = _hex_digest(initial_salt, "initial_salt", 32)
    if expected_store_identity is not None:
        expected_store_identity = _store_identity(expected_store_identity)
    if not isinstance(fail_if_locked, bool):
        raise TypeError("fail_if_locked must be a bool")
    with _exclusive_writer(
        store_path,
        expected_store_identity=expected_store_identity,
        fail_if_locked=fail_if_locked,
    ) as tuning:
        registry = _load_for_mutation(tuning, initial_salt=initial_salt)
        if expected_salt is not None and registry.salt != expected_salt:
            raise TuningRegistryError(
                "tuning registry identity changed while preparing the profile; "
                "retry the operation"
            )
        if (
            expected_generation is not None
            and registry.generation != expected_generation
        ):
            raise TuningRegistryError(
                "tuning registry generation changed while preparing the profile; "
                "retry the operation"
            )
        if registry.generation == _UINT64_MAX:
            raise TuningRegistryError("tuning registry generation is exhausted")
        salt_bytes = bytes.fromhex(registry.salt)
        key = registry.key_for(device, runtime, layout)
        existing = registry.select(key)
        if timestamp is None:
            timestamp = _timestamp(None)
            if existing is not None:
                timestamp = max(timestamp, existing.updated_at_ns)
        if existing is None and len(registry.profiles) >= _MAX_PROFILES:
            raise TuningRegistryError("tuning registry profile limit is exhausted")
        created = timestamp if existing is None else existing.created_at_ns
        if timestamp < created:
            raise ValueError("timestamp_ns predates the existing profile")
        profile = ReaderProfile(
            key=key,
            device=_redact_mapping(device, salt_bytes, "device"),
            runtime=_redact_mapping(runtime, salt_bytes, "runtime"),
            layout=_redact_mapping(layout, salt_bytes, "layout"),
            options=normalized,
            source=source,
            created_at_ns=created,
            updated_at_ns=timestamp,
        )
        profiles = [item for item in registry.profiles if item.key != key]
        profiles.append(profile)
        profiles.sort(key=lambda item: _key_tuple(item.key))
        updated = TuningRegistry(
            _VERSION, registry.generation + 1, registry.salt, tuple(profiles)
        )
        _publish_registry(tuning, updated)
        return profile


def remove_profile(
    store_path: object,
    key: RegistryKey,
    *,
    strict: bool = True,
) -> bool:
    if not isinstance(key, RegistryKey):
        raise TypeError("key must be a RegistryKey")
    with _exclusive_writer(store_path) as tuning:
        loaded = _load_registry_from_directory(tuning, strict=strict)
        if loaded.registry is None:
            return False
        registry = loaded.registry
        retained = tuple(profile for profile in registry.profiles if profile.key != key)
        if len(retained) == len(registry.profiles):
            return False
        if registry.generation == _UINT64_MAX:
            raise TuningRegistryError("tuning registry generation is exhausted")
        updated = TuningRegistry(
            _VERSION, registry.generation + 1, registry.salt, retained
        )
        _publish_registry(tuning, updated)
        return True


def list_profiles(
    store_path: object, *, strict: bool = True
) -> list[ReaderProfile]:
    loaded = load_registry(store_path, strict=strict)
    if loaded.registry is None:
        return []
    return loaded.registry.list_profiles()


def reset_registry(store_path: object, *, force: bool = False) -> bool:
    """Remove the selected optional registry, preserving collection data.

    A corrupt registry requires ``force=True`` so a caller cannot erase the
    only diagnostics accidentally. The active pathname is first atomically
    moved aside and the directory is synced; the quarantine file is then
    removed best-effort and is never considered by readers.
    """

    if not isinstance(force, bool):
        raise TypeError("force must be a bool")
    with _exclusive_writer(store_path) as tuning:
        loaded = _load_registry_from_directory(tuning, strict=False)
        if loaded.status == "missing":
            return False
        if loaded.status == "corrupt" and not force:
            raise TuningRegistryError(
                "refusing to reset a corrupt tuning registry without force=True: "
                f"{loaded.error}"
            )
        quarantine = (
            f"reader-profiles.quarantine.{os.getpid()}.{secrets.token_hex(8)}"
        )
        try:
            os.replace(
                _REGISTRY_FILENAME,
                quarantine,
                src_dir_fd=tuning,
                dst_dir_fd=tuning,
            )
            os.fsync(tuning)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise TuningRegistryError(
                f"cannot quarantine tuning registry: {error}"
            ) from error
        try:
            os.unlink(quarantine, dir_fd=tuning)
            os.fsync(tuning)
        except OSError:
            # The unique quarantine pathname is ignored by every load and is
            # safer to retain than to report a failed reset after publication.
            pass
        return True
