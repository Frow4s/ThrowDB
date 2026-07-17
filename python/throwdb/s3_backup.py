"""Crash-safe, lineage-checked ThrowDB checkpoints in versioned Amazon S3.

The module intentionally depends only on the AWS CLI.  Payload objects are
published into alternating slots, while every completion records immutable S3
VersionIds and SHA-256 digests.  ``LATEST.json`` is changed with an S3 compare-
and-swap and all cooperating publishers share a conditional S3 lease.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal, Mapping, Sequence, TypedDict, cast
from urllib.parse import urlsplit


_BACKUP_FORMAT = "throwdb-s3-backup"
_LATEST_FORMAT = "throwdb-s3-backup-latest"
_LINEAGE_FORMAT = "throwdb-s3-lineage"
_LEASE_FORMAT = "throwdb-s3-publisher-lease"
_LOCAL_CURSOR_FORMAT = "throwdb-s3-local-cursor"
_FORMAT_VERSION = 2
_SLOTS = ("slot-a", "slot-b")
_MAX_METADATA_OUTPUT_BYTES = 64 << 20
_OUTPUT_SNIPPET_BYTES = 16 << 10
_MAX_INVENTORY_FILES = 100_000
_HASH_CHUNK_BYTES = 8 << 20
_DEFAULT_MAX_SECONDS = 12 * 60 * 60.0
_LEASE_EXPIRY_MARGIN_SECONDS = 300.0
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{1,61}[A-Za-z0-9]$")
_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_MAGIC = b"TDBNMNF2"
_LOCAL_CURSOR_DIRECTORY = "server/s3-checkpoints"


class S3BackupError(RuntimeError):
    """The checkpoint or restore was rejected or could not finish safely."""


def _is_immutable_version_id(value: object) -> bool:
    # S3 uses the literal string "null" for objects that have no immutable
    # version (for example objects written while Versioning is suspended).
    # Treating it as an ordinary non-empty VersionId would let a long backup
    # publish a completion after the bucket policy changed underneath it.
    return isinstance(value, str) and bool(value) and value != "null"


class S3BackupReport(TypedDict):
    format: str
    version: int
    status: Literal["completed", "dry_run", "skipped"]
    dry_run: bool
    storage: str
    destination: str
    bucket_versioning: str
    previous_slot: str | None
    target_slot: str
    backup_id: str | None
    completion_uri: str | None
    latest_uri: str
    latest_publication_reconciled: bool
    source_inventory: dict[str, object]
    source_state: dict[str, object]
    planned_change_count: int | None
    reason: str | None


class S3RestoreReport(TypedDict):
    format: str
    version: int
    status: Literal["restored"]
    source: str
    storage: str
    backup_id: str
    slot: str
    restored_file_count: int
    restored_bytes: int
    source_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class _S3Target:
    bucket: str
    prefix: str

    def key(self, *parts: str) -> str:
        suffix = "/".join(part.strip("/") for part in parts if part)
        return "/".join(part for part in (self.prefix, suffix) if part)

    def uri(self, *parts: str, directory: bool = False) -> str:
        return self.uri_for_key(self.key(*parts), directory=directory)

    def uri_for_key(self, key: str, *, directory: bool = False) -> str:
        value = f"s3://{self.bucket}"
        if key:
            value += f"/{key}"
        if directory and not value.endswith("/"):
            value += "/"
        return value


@dataclass(frozen=True, slots=True)
class _FileEntry:
    path: str
    size: int
    mode: int
    sha256: str

    def document(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "mode": self.mode,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _DirectoryEntry:
    path: str
    mode: int

    def document(self) -> dict[str, object]:
        return {"path": self.path, "mode": self.mode}


@dataclass(frozen=True, slots=True)
class _Inventory:
    file_count: int
    directory_count: int
    total_bytes: int
    metadata_sha256: str
    stability_sha256: str
    manifest_sha256: str
    manifest_size: int
    format_version: int
    current_sha256: str
    current_size: int
    content_sha256: str
    authoritative_sha256: str
    files: tuple[_FileEntry, ...]
    directories: tuple[_DirectoryEntry, ...]

    def public(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
            "metadata_sha256": self.metadata_sha256,
            "manifest_sha256": self.manifest_sha256,
            "manifest_size": self.manifest_size,
            "format_version": self.format_version,
            "current_sha256": self.current_sha256,
            "current_size": self.current_size,
            "content_sha256": self.content_sha256,
            "authoritative_sha256": self.authoritative_sha256,
        }


@dataclass(frozen=True, slots=True)
class _SourceState:
    collection_id: str
    catalog_generation: int
    committed_sequence: int
    live_count: int
    authoritative_sha256: str
    content_sha256: str

    def document(self) -> dict[str, object]:
        return {
            "collection_id": self.collection_id,
            "catalog_generation": self.catalog_generation,
            "committed_sequence": self.committed_sequence,
            "live_count": self.live_count,
            "authoritative_sha256": self.authoritative_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ObjectSnapshot:
    size: int
    etag: str
    version_id: str


@dataclass(frozen=True, slots=True)
class _ControlSnapshot:
    document: dict[str, object]
    etag: str
    version_id: str


@dataclass(frozen=True, slots=True)
class _RemoteFileEntry:
    path: str
    size: int
    mode: int
    sha256: str
    etag: str
    version_id: str

    def document(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "mode": self.mode,
            "sha256": self.sha256,
            "etag": self.etag,
            "version_id": self.version_id,
        }


@dataclass(frozen=True, slots=True)
class _CompletionRef:
    key: str
    version_id: str

    def document(self) -> dict[str, object]:
        return {"key": self.key, "version_id": self.version_id}


@dataclass(frozen=True, slots=True)
class _LatestInfo:
    slot: str
    backup_id: str
    completion: _CompletionRef
    manifest_sha256: str
    source_state: _SourceState
    completed_at: dt.datetime
    slot_completions: Mapping[str, _CompletionRef]


@dataclass(frozen=True, slots=True)
class _RemoteLease:
    key: str
    holder: str
    snapshot: _ControlSnapshot
    expires_at: dt.datetime


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._finished = time.monotonic() + seconds

    def remaining(self) -> float:
        remaining = self._finished - time.monotonic()
        if remaining <= 0:
            raise S3BackupError("S3 checkpoint deadline expired")
        return remaining


def _parse_target(value: str) -> _S3Target:
    if not isinstance(value, str):
        raise TypeError("S3 target must be a string")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise S3BackupError("S3 target contains an invalid port") from error
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
    ):
        raise S3BackupError(
            "S3 target must be s3://bucket[/prefix] without credentials, "
            "a query, or a fragment"
        )
    bucket = parsed.netloc
    if not _BUCKET_RE.fullmatch(bucket):
        raise S3BackupError(f"invalid S3 bucket name: {bucket!r}")
    prefix = parsed.path.strip("/")
    if "\\" in prefix or any(
        part in {".", ".."} for part in prefix.split("/") if part
    ):
        raise S3BackupError("S3 target prefix contains an unsafe path segment")
    if any(ord(character) < 32 for character in prefix):
        raise S3BackupError("S3 target prefix contains a control character")
    return _S3Target(bucket=bucket, prefix=prefix)


def _resolve_aws_executable(value: str | os.PathLike[str]) -> str:
    executable = os.fspath(value)
    if not executable:
        raise S3BackupError("AWS CLI executable must not be empty")
    if os.sep in executable:
        try:
            path = Path(executable).expanduser().resolve()
        except (OSError, RuntimeError) as error:
            raise S3BackupError(
                f"cannot resolve AWS CLI executable {executable!r}: {error}"
            ) from error
        if not path.is_file() or not os.access(path, os.X_OK):
            raise S3BackupError(f"AWS CLI is not an executable file: {path}")
        return str(path)
    resolved = shutil.which(executable)
    if resolved is None:
        raise S3BackupError(
            f"AWS CLI executable {executable!r} was not found in PATH"
        )
    return resolved


class _AwsCli:
    def __init__(self, executable: str, metadata_timeout_seconds: float) -> None:
        self.executable = executable
        self.metadata_timeout_seconds = metadata_timeout_seconds
        self.environment = os.environ.copy()
        self.environment["AWS_PAGER"] = ""
        self.environment["AWS_CLI_AUTO_PROMPT"] = "off"

    def run_text(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        maximum_output_bytes: int = _MAX_METADATA_OUTPUT_BYTES,
    ) -> str:
        command = [self.executable, *arguments]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=(
                        self.metadata_timeout_seconds if timeout is None else timeout
                    ),
                    env=self.environment,
                    close_fds=True,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise S3BackupError(
                    f"failed to execute AWS CLI command {arguments[0]!r}: {error}"
                ) from error
            stdout.flush()
            stderr.flush()
            stdout_size = stdout.tell()
            stderr_size = stderr.tell()
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read(maximum_output_bytes + 1)
            diagnostic = stderr.read(_OUTPUT_SNIPPET_BYTES).decode(
                "utf-8", errors="replace"
            ).strip()
        if result.returncode != 0:
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} failed with exit code "
                f"{result.returncode}: {diagnostic or 'no diagnostic output'}"
            )
        if stdout_size > maximum_output_bytes:
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} returned oversized metadata"
            )
        if stderr_size:
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} returned a warning: "
                f"{diagnostic or 'no diagnostic output'}"
            )
        try:
            return output.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} returned non-UTF-8 metadata"
            ) from error

    def run_json(
        self, arguments: Sequence[str], *, timeout: float | None = None
    ) -> dict[str, object]:
        text = self.run_text(arguments, timeout=timeout)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} returned invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise S3BackupError(
                f"AWS CLI command {arguments[0]!r} returned a non-object JSON value"
            )
        return cast(dict[str, object], value)

    def copy_file(
        self, source: Path, destination: str, *, timeout: float
    ) -> None:
        self.run_text(
            [
                "s3",
                "cp",
                str(source),
                destination,
                "--checksum-algorithm",
                "SHA256",
                "--only-show-errors",
            ],
            timeout=timeout,
        )

    def sync_directory(
        self, source: Path, destination: str, *, timeout: float
    ) -> None:
        self.run_text(
            [
                "s3",
                "sync",
                str(source),
                destination,
                "--delete",
                "--no-follow-symlinks",
                "--exclude",
                f"{_LOCAL_CURSOR_DIRECTORY}/*",
                "--checksum-algorithm",
                "SHA256",
                "--only-show-errors",
            ],
            timeout=timeout,
        )

    def sync_from_s3(
        self, source: str, destination: Path, *, timeout: float
    ) -> None:
        self.run_text(
            [
                "s3",
                "sync",
                source,
                str(destination),
                "--delete",
                "--checksum-mode",
                "enabled",
                "--only-show-errors",
            ],
            timeout=timeout,
        )

    def get_object(
        self,
        *,
        bucket: str,
        key: str,
        output: Path,
        version_id: str = "",
        etag: str = "",
        timeout: float,
    ) -> None:
        arguments = [
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
        ]
        if version_id:
            arguments.extend(["--version-id", version_id])
        if etag:
            arguments.extend(["--if-match", etag])
        arguments.extend(["--output", "json", str(output)])
        self.run_text(arguments, timeout=timeout)


def _json_document(data: bytes, description: str) -> dict[str, object]:
    if len(data) > _MAX_METADATA_OUTPUT_BYTES:
        raise S3BackupError(f"{description} exceeds the metadata safety limit")
    try:
        value = json.loads(data.decode("utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise S3BackupError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise S3BackupError(f"{description} is not a JSON object")
    return cast(dict[str, object], value)


def _serialize_json(value: Mapping[str, object]) -> bytes:
    try:
        data = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise S3BackupError(f"cannot serialize S3 control metadata: {error}") from error
    if len(data) > _MAX_METADATA_OUTPUT_BYTES:
        raise S3BackupError("S3 control metadata exceeds the safety limit")
    return data


def _object_exists(
    aws: _AwsCli, target: _S3Target, key: str, deadline: _Deadline
) -> bool:
    result = aws.run_json(
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            target.bucket,
            "--prefix",
            key,
            "--max-keys",
            "10",
            "--output",
            "json",
        ],
        timeout=deadline.remaining(),
    )
    contents = result.get("Contents", [])
    if contents is None:
        contents = []
    if not isinstance(contents, list):
        raise S3BackupError("S3 object listing returned invalid Contents metadata")
    return any(
        isinstance(item, dict) and item.get("Key") == key for item in contents
    )


def _destination_has_objects(
    aws: _AwsCli, target: _S3Target, deadline: _Deadline
) -> bool:
    if target.prefix and _object_exists(aws, target, target.prefix, deadline):
        return True
    namespace = f"{target.prefix}/" if target.prefix else ""
    result = aws.run_json(
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            target.bucket,
            "--prefix",
            namespace,
            "--max-keys",
            "1",
            "--output",
            "json",
        ],
        timeout=deadline.remaining(),
    )
    contents = result.get("Contents", [])
    if contents is None:
        contents = []
    if not isinstance(contents, list):
        raise S3BackupError("S3 object listing returned invalid Contents metadata")
    return bool(contents)


def _head_object(
    aws: _AwsCli,
    target: _S3Target,
    key: str,
    deadline: _Deadline,
    *,
    version_id: str = "",
    require_version: bool,
) -> _ObjectSnapshot:
    arguments = [
        "s3api",
        "head-object",
        "--bucket",
        target.bucket,
        "--key",
        key,
    ]
    if version_id:
        arguments.extend(["--version-id", version_id])
    arguments.extend(["--output", "json"])
    result = aws.run_json(arguments, timeout=deadline.remaining())
    size = result.get("ContentLength")
    etag = result.get("ETag")
    observed_version = result.get("VersionId", "")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise S3BackupError("S3 head-object returned an invalid ContentLength")
    if not isinstance(etag, str) or not etag:
        raise S3BackupError("S3 head-object returned an invalid ETag")
    if not isinstance(observed_version, str):
        raise S3BackupError("S3 head-object returned an invalid VersionId")
    if require_version and not _is_immutable_version_id(observed_version):
        raise S3BackupError(
            "S3 object has no immutable VersionId although bucket Versioning "
            "is required"
        )
    if version_id and observed_version != version_id:
        raise S3BackupError("S3 head-object returned a different VersionId")
    return _ObjectSnapshot(size=size, etag=etag, version_id=observed_version)


def _download_control(
    aws: _AwsCli,
    target: _S3Target,
    key: str,
    description: str,
    deadline: _Deadline,
    *,
    version_id: str = "",
    require_version: bool,
) -> _ControlSnapshot:
    head = _head_object(
        aws,
        target,
        key,
        deadline,
        version_id=version_id,
        require_version=require_version,
    )
    with tempfile.TemporaryDirectory(prefix="throwdb-s3-control-") as directory:
        output = Path(directory) / "object.json"
        aws.get_object(
            bucket=target.bucket,
            key=key,
            output=output,
            version_id=head.version_id,
            etag=head.etag,
            timeout=deadline.remaining(),
        )
        try:
            data = output.read_bytes()
        except OSError as error:
            raise S3BackupError(
                f"cannot read downloaded {description}: {error}"
            ) from error
    if len(data) != head.size:
        raise S3BackupError(f"downloaded {description} size differs from S3 metadata")
    return _ControlSnapshot(
        document=_json_document(data, description),
        etag=head.etag,
        version_id=head.version_id,
    )


def _put_json_conditional(
    aws: _AwsCli,
    target: _S3Target,
    key: str,
    document: Mapping[str, object],
    deadline: _Deadline,
    *,
    expected_etag: str | None,
    require_version: bool,
) -> _ControlSnapshot:
    data = _serialize_json(document)
    descriptor = -1
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="throwdb-s3-put-", suffix=".json"
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        arguments = [
            "s3api",
            "put-object",
            "--bucket",
            target.bucket,
            "--key",
            key,
            "--body",
            str(temporary_name),
            "--content-type",
            "application/json",
        ]
        if expected_etag is None:
            arguments.extend(["--if-none-match", "*"])
        else:
            arguments.extend(["--if-match", expected_etag])
        arguments.extend(["--output", "json"])
        result = aws.run_json(arguments, timeout=deadline.remaining())
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(f"cannot prepare S3 control metadata: {error}") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass

    etag = result.get("ETag")
    version_id = result.get("VersionId", "")
    if not isinstance(etag, str) or not etag:
        raise S3BackupError("S3 conditional put returned an invalid ETag")
    if not isinstance(version_id, str):
        raise S3BackupError("S3 conditional put returned an invalid VersionId")
    if require_version and not _is_immutable_version_id(version_id):
        raise S3BackupError(
            "S3 conditional put returned no immutable VersionId despite "
            "required Versioning"
        )
    published = _download_control(
        aws,
        target,
        key,
        "published control object",
        deadline,
        version_id=version_id,
        require_version=require_version,
    )
    if published.etag != etag or published.document != dict(document):
        raise S3BackupError(
            "conditional S3 publication read-back differs from the requested document"
        )
    return published


def _publish_json_reconciled(
    aws: _AwsCli,
    target: _S3Target,
    key: str,
    document: Mapping[str, object],
    deadline: _Deadline,
    *,
    expected_etag: str | None,
    require_version: bool,
    description: str,
) -> tuple[_ControlSnapshot, bool]:
    """Do one conditional put; reconcile only an exact commit-then-error.

    Conditional conflicts, an old AWS CLI, and transport failures are never
    blindly retried.  A successful exact current read-back is the sole safe
    exception because it proves that the requested bytes became current.
    """

    try:
        return (
            _put_json_conditional(
                aws,
                target,
                key,
                document,
                deadline,
                expected_etag=expected_etag,
                require_version=require_version,
            ),
            False,
        )
    except S3BackupError as upload_error:
        try:
            published = _download_control(
                aws,
                target,
                key,
                f"{description} reconciliation read-back",
                deadline,
                require_version=require_version,
            )
        except S3BackupError as readback_error:
            raise S3BackupError(
                f"{description} publication failed and exact read-back could not "
                "establish the outcome; refusing to retry an ambiguous conditional "
                f"write (upload error: {upload_error}; read-back error: {readback_error})"
            ) from upload_error
        if published.document == dict(document):
            return published, True
        raise S3BackupError(
            f"{description} conditional publication failed and current S3 state "
            "contains a different document; concurrent publication or a CAS "
            f"conflict was detected (upload error: {upload_error})"
        ) from upload_error


def _list_objects(
    aws: _AwsCli,
    target: _S3Target,
    prefix: str,
    deadline: _Deadline,
) -> dict[str, int]:
    objects: dict[str, int] = {}
    continuation: str | None = None
    while True:
        arguments = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            target.bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--output",
            "json",
        ]
        if continuation is not None:
            arguments.extend(["--continuation-token", continuation])
        result = aws.run_json(arguments, timeout=deadline.remaining())
        contents = result.get("Contents", [])
        if contents is None:
            contents = []
        if not isinstance(contents, list):
            raise S3BackupError("S3 object listing returned invalid Contents metadata")
        for item in contents:
            if not isinstance(item, dict):
                raise S3BackupError("S3 object listing contains an invalid entry")
            key = item.get("Key")
            size = item.get("Size")
            if (
                not isinstance(key, str)
                or not key.startswith(prefix)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or key in objects
            ):
                raise S3BackupError("S3 object listing contains invalid metadata")
            objects[key] = size
            if len(objects) > _MAX_INVENTORY_FILES:
                raise S3BackupError("S3 slot contains too many objects")
        truncated = result.get("IsTruncated", False)
        if not isinstance(truncated, bool):
            raise S3BackupError("S3 listing returned invalid pagination metadata")
        if not truncated:
            return objects
        next_token = result.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            raise S3BackupError("S3 listing omitted its continuation token")
        continuation = next_token


def _list_current_versions(
    aws: _AwsCli,
    target: _S3Target,
    prefix: str,
    deadline: _Deadline,
) -> dict[str, _ObjectSnapshot]:
    """List current version identities in bulk, including current deletions.

    S3 has no batch HEAD API.  ``list-object-versions`` is the only AWS CLI
    primitive that supplies VersionId and ETag without one process per file.
    Operators should expire noncurrent slot versions after their required
    recovery window so this bounded pagination remains cheap.
    """

    current: dict[str, _ObjectSnapshot] = {}
    deleted: set[str] = set()
    key_marker: str | None = None
    version_marker: str | None = None
    observed_records = 0
    while True:
        arguments = [
            "s3api",
            "list-object-versions",
            "--bucket",
            target.bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--output",
            "json",
        ]
        if key_marker is not None:
            arguments.extend(["--key-marker", key_marker])
        if version_marker is not None:
            arguments.extend(["--version-id-marker", version_marker])
        result = aws.run_json(arguments, timeout=deadline.remaining())
        versions = result.get("Versions", [])
        delete_markers = result.get("DeleteMarkers", [])
        if versions is None:
            versions = []
        if delete_markers is None:
            delete_markers = []
        if not isinstance(versions, list) or not isinstance(delete_markers, list):
            raise S3BackupError("S3 version listing returned invalid arrays")
        observed_records += len(versions) + len(delete_markers)
        if observed_records > _MAX_INVENTORY_FILES * 32:
            raise S3BackupError(
                "S3 slot has too many retained object versions; configure a "
                "NoncurrentVersionExpiration lifecycle policy"
            )
        for item in versions:
            if not isinstance(item, dict) or item.get("IsLatest") is not True:
                continue
            key = item.get("Key")
            size = item.get("Size")
            etag = item.get("ETag")
            version_id = item.get("VersionId")
            if (
                not isinstance(key, str)
                or not key.startswith(prefix)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(etag, str)
                or not etag
                or not _is_immutable_version_id(version_id)
                or key in current
            ):
                raise S3BackupError("S3 current version listing is invalid")
            current[key] = _ObjectSnapshot(size, etag, version_id)
        for item in delete_markers:
            if not isinstance(item, dict) or item.get("IsLatest") is not True:
                continue
            key = item.get("Key")
            if not isinstance(key, str) or not key.startswith(prefix):
                raise S3BackupError("S3 delete-marker listing is invalid")
            deleted.add(key)
        truncated = result.get("IsTruncated", False)
        if not isinstance(truncated, bool):
            raise S3BackupError("S3 version listing has invalid pagination metadata")
        if not truncated:
            for key in deleted:
                current.pop(key, None)
            if len(current) > _MAX_INVENTORY_FILES:
                raise S3BackupError("S3 slot contains too many current objects")
            return current
        next_key = result.get("NextKeyMarker")
        next_version = result.get("NextVersionIdMarker")
        if not isinstance(next_key, str) or not next_key:
            raise S3BackupError("S3 version listing omitted NextKeyMarker")
        if next_version is not None and not isinstance(next_version, str):
            raise S3BackupError("S3 version listing has invalid version marker")
        key_marker = next_key
        version_marker = next_version


def _delete_object(
    aws: _AwsCli, target: _S3Target, key: str, deadline: _Deadline
) -> None:
    aws.run_json(
        [
            "s3api",
            "delete-object",
            "--bucket",
            target.bucket,
            "--key",
            key,
            "--output",
            "json",
        ],
        timeout=deadline.remaining(),
    )


def _excluded_local_path(relative: str) -> bool:
    return relative == _LOCAL_CURSOR_DIRECTORY or relative.startswith(
        f"{_LOCAL_CURSOR_DIRECTORY}/"
    )


def _hash_regular_file(
    path: Path,
    expected: os.stat_result,
    deadline: _Deadline,
    *,
    maximum_bytes: int | None = None,
) -> str:
    if maximum_bytes is not None and expected.st_size > maximum_bytes:
        raise S3BackupError(f"collection anchor is unexpectedly large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_dev != expected.st_dev
                or before.st_ino != expected.st_ino
                or before.st_size != expected.st_size
                or before.st_mtime_ns != expected.st_mtime_ns
                or before.st_ctime_ns != expected.st_ctime_ns
            ):
                raise S3BackupError(f"collection file changed before hashing: {path}")
            total = 0
            while True:
                deadline.remaining()
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(f"cannot hash collection file {path}: {error}") from error
    if (
        total != expected.st_size
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise S3BackupError(f"collection file changed while hashing: {path}")
    return digest.hexdigest()


def _inventory(root: Path, deadline: _Deadline) -> _Inventory:
    metadata = hashlib.sha256()
    stability = hashlib.sha256()
    content = hashlib.sha256()
    files: list[_FileEntry] = []
    directories: list[_DirectoryEntry] = []
    directory_count = 0
    total_bytes = 0
    pending = [root]
    while pending:
        deadline.remaining()
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise S3BackupError(
                f"cannot scan collection directory {directory}: {error}"
            ) from error
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            if _excluded_local_path(relative):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise S3BackupError(
                    f"cannot inspect collection entry {relative}: {error}"
                ) from error
            try:
                encoded = relative.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise S3BackupError(
                    f"collection entry name is not valid UTF-8: {relative!r}"
                ) from error
            if stat.S_ISDIR(info.st_mode):
                directory_count += 1
                mode = stat.S_IMODE(info.st_mode)
                directories.append(_DirectoryEntry(relative, mode))
                metadata.update(b"D\0" + encoded + b"\0" + str(mode).encode() + b"\0")
                stability.update(
                    b"D\0"
                    + encoded
                    + b"\0"
                    + f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}:{info.st_ctime_ns}".encode()
                    + b"\0"
                )
                pending.append(Path(entry.path))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise S3BackupError(
                    "collection contains a symlink or non-regular entry; "
                    f"refusing to checkpoint it: {relative}"
                )
            if len(files) >= _MAX_INVENTORY_FILES:
                raise S3BackupError("collection contains too many files to checkpoint")
            digest = _hash_regular_file(Path(entry.path), info, deadline)
            mode = stat.S_IMODE(info.st_mode)
            file_entry = _FileEntry(relative, info.st_size, mode, digest)
            files.append(file_entry)
            total_bytes += info.st_size
            common = (
                b"F\0"
                + encoded
                + b"\0"
                + f"{info.st_size}:{mode}:{digest}".encode()
                + b"\0"
            )
            metadata.update(common)
            content.update(common)
            stability.update(
                common
                + f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}:{info.st_ctime_ns}".encode()
                + b"\0"
            )

    files.sort(key=lambda entry: entry.path)
    directories.sort(key=lambda entry: entry.path)
    by_path = {entry.path: entry for entry in files}
    manifest = by_path.get("manifest.bin")
    current = by_path.get("CURRENT")
    if manifest is None:
        raise S3BackupError("collection is missing manifest.bin")
    if current is None:
        raise S3BackupError("format-7 collection is missing CURRENT")
    logical_view = hashlib.sha256()
    logical_view.update(b"manifest.bin\0" + manifest.sha256.encode() + b"\0")
    logical_view.update(b"CURRENT\0" + current.sha256.encode() + b"\0")
    try:
        with (root / "manifest.bin").open("rb") as stream:
            magic = stream.read(8)
    except OSError as error:
        raise S3BackupError(f"cannot read collection manifest: {error}") from error
    if magic != _MANIFEST_MAGIC:
        raise S3BackupError(
            "hardened S3 checkpoints require the authoritative partitioned "
            f"format-7 manifest, got {magic!r}"
        )
    return _Inventory(
        file_count=len(files),
        directory_count=directory_count,
        total_bytes=total_bytes,
        metadata_sha256=metadata.hexdigest(),
        stability_sha256=stability.hexdigest(),
        manifest_sha256=manifest.sha256,
        manifest_size=manifest.size,
        format_version=7,
        current_sha256=current.sha256,
        current_size=current.size,
        content_sha256=content.hexdigest(),
        authoritative_sha256=logical_view.hexdigest(),
        files=tuple(files),
        directories=tuple(directories),
    )


def _manifest_collection_id(root: Path) -> str:
    try:
        data = (root / "manifest.bin").read_bytes()
    except OSError as error:
        raise S3BackupError(f"cannot read collection manifest identity: {error}") from error
    if len(data) < 40 or data[:8] != _MANIFEST_MAGIC:
        raise S3BackupError("collection manifest cannot provide a format-7 identity")
    low, high = struct.unpack_from("<QQ", data, 24)
    if low == 0 and high == 0:
        raise S3BackupError("collection manifest contains an empty collection ID")
    return f"{high:016x}{low:016x}"


def _verified_source_state(root: Path, inventory: _Inventory) -> _SourceState:
    pending = root / "server" / "pending-mutation-v1"
    if os.path.lexists(pending):
        raise S3BackupError(
            "collection has a pending server mutation fence; start the writable "
            "server to finish recovery before creating a checkpoint"
        )
    try:
        from .collection import Collection

        with Collection.open(
            root,
            mode="r",
            tuning="conservative",
            verify_index=True,
        ) as collection:
            if collection.format_version != 7:
                raise S3BackupError("S3 checkpoints require collection format 7")
            stats = collection.mutation_log_stats
            live_count = int(collection.count)
    except S3BackupError:
        raise
    except Exception as error:
        raise S3BackupError(
            f"source preflight failed complete index/WAL verification: {error}"
        ) from error
    if not isinstance(stats, dict):
        raise S3BackupError("source preflight returned no mutation state")
    generation = stats.get("generation")
    sequence = stats.get("committed_sequence")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or live_count < 0
    ):
        raise S3BackupError("source preflight returned invalid mutation state")
    return _SourceState(
        collection_id=_manifest_collection_id(root),
        catalog_generation=generation,
        committed_sequence=sequence,
        live_count=live_count,
        authoritative_sha256=inventory.authoritative_sha256,
        content_sha256=inventory.content_sha256,
    )


@contextlib.contextmanager
def _writer_lease(root: Path) -> Iterator[None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_DIRECTORY", 0
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        root_fd = os.open(root, flags | nofollow)
        descriptors.append(root_fd)
        locks_fd = os.open("locks", flags | nofollow, dir_fd=root_fd)
        descriptors.append(locks_fd)
        lock_fd = os.open(
            "writer.lock",
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | nofollow,
            dir_fd=locks_fd,
        )
        descriptors.append(lock_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise S3BackupError("collection writer lock is not a regular file")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise S3BackupError(
                "another writer, server, calibration, maintenance, or backup "
                "holds the collection writer lease; checkpoint a quiesced collection"
            ) from error
        yield
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(
            f"cannot acquire the collection writer lease at {root}: {error}"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _destination_identity(target: _S3Target) -> str:
    return target.uri(directory=True)


def _cursor_path(root: Path, target: _S3Target) -> Path:
    token = hashlib.sha256(_destination_identity(target).encode()).hexdigest()
    return root / _LOCAL_CURSOR_DIRECTORY / f"{token}.json"


def _read_local_cursor(
    root: Path, target: _S3Target, source: _SourceState
) -> dict[str, object] | None:
    path = _cursor_path(root, target)
    if not os.path.lexists(path):
        return None
    try:
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > (1 << 20):
            raise S3BackupError("local S3 lineage cursor is not a small regular file")
        document = _json_document(path.read_bytes(), "local S3 lineage cursor")
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(f"cannot read local S3 lineage cursor: {error}") from error
    if (
        document.get("format") != _LOCAL_CURSOR_FORMAT
        or document.get("version") != _FORMAT_VERSION
        or document.get("destination") != _destination_identity(target)
        or document.get("collection_id") != source.collection_id
        or not isinstance(document.get("backup_id"), str)
        or not _HEX_32_RE.fullmatch(cast(str, document["backup_id"]))
    ):
        raise S3BackupError("local S3 lineage cursor is invalid or belongs elsewhere")
    return document


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_DIRECTORY", 0
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_local_cursor(
    root: Path,
    target: _S3Target,
    source: _SourceState,
    backup_id: str,
) -> None:
    directory = root / _LOCAL_CURSOR_DIRECTORY
    path = _cursor_path(root, target)
    document = {
        "format": _LOCAL_CURSOR_FORMAT,
        "version": _FORMAT_VERSION,
        "destination": _destination_identity(target),
        "collection_id": source.collection_id,
        "backup_id": backup_id,
        "source_state": source.document(),
        "updated_at": _utc_now_text(),
    }
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise S3BackupError("local S3 cursor parent is not a directory")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".cursor-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_serialize_json(document))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
            _fsync_directory(directory)
        finally:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(f"cannot persist local S3 lineage cursor: {error}") from error


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_now_text() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, description: str) -> dt.datetime:
    if not isinstance(value, str):
        raise S3BackupError(f"{description} contains an invalid timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise S3BackupError(f"{description} contains an invalid timestamp") from error
    if parsed.tzinfo is None:
        raise S3BackupError(f"{description} timestamp has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S3BackupError(f"{description} is not a nonnegative integer")
    return value


def _source_state_from_document(value: object, description: str) -> _SourceState:
    if not isinstance(value, dict):
        raise S3BackupError(f"{description} has no source state object")
    collection_id = value.get("collection_id")
    authoritative = value.get("authoritative_sha256")
    content = value.get("content_sha256")
    if not isinstance(collection_id, str) or not _HEX_32_RE.fullmatch(collection_id):
        raise S3BackupError(f"{description} has an invalid collection ID")
    if not isinstance(authoritative, str) or not _HEX_64_RE.fullmatch(authoritative):
        raise S3BackupError(f"{description} has an invalid authoritative digest")
    if not isinstance(content, str) or not _HEX_64_RE.fullmatch(content):
        raise S3BackupError(f"{description} has an invalid content digest")
    return _SourceState(
        collection_id=collection_id,
        catalog_generation=_integer(
            value.get("catalog_generation"), f"{description} catalog generation"
        ),
        committed_sequence=_integer(
            value.get("committed_sequence"), f"{description} committed sequence"
        ),
        live_count=_integer(value.get("live_count"), f"{description} live count"),
        authoritative_sha256=authoritative,
        content_sha256=content,
    )


def _completion_ref(value: object, description: str) -> _CompletionRef:
    if not isinstance(value, dict):
        raise S3BackupError(f"{description} is not an object")
    key = value.get("key")
    version_id = value.get("version_id")
    if not isinstance(key, str) or not key:
        raise S3BackupError(f"{description} has an invalid key")
    if not _is_immutable_version_id(version_id):
        raise S3BackupError(f"{description} has no immutable VersionId")
    return _CompletionRef(key=key, version_id=version_id)


def _validate_lineage(
    document: Mapping[str, object], source: _SourceState, manifest_sha256: str
) -> None:
    if (
        document.get("format") != _LINEAGE_FORMAT
        or document.get("version") != _FORMAT_VERSION
    ):
        raise S3BackupError(
            "S3 LINEAGE has an unsupported format; use a fresh prefix for "
            "hardened version-2 checkpoints"
        )
    if (
        document.get("collection_id") != source.collection_id
        or document.get("manifest_sha256") != manifest_sha256
    ):
        raise S3BackupError(
            "the S3 destination belongs to a different ThrowDB collection"
        )


def _validate_latest(document: Mapping[str, object], target: _S3Target) -> _LatestInfo:
    if (
        document.get("format") != _LATEST_FORMAT
        or document.get("version") != _FORMAT_VERSION
    ):
        raise S3BackupError(
            "LATEST has an unsupported format; version-1 prefixes must be "
            "migrated explicitly or replaced with a fresh prefix"
        )
    slot = document.get("slot")
    backup_id = document.get("backup_id")
    manifest = document.get("manifest_sha256")
    if slot not in _SLOTS:
        raise S3BackupError("LATEST names an invalid backup slot")
    if not isinstance(backup_id, str) or not _HEX_32_RE.fullmatch(backup_id):
        raise S3BackupError("LATEST contains an invalid backup ID")
    if not isinstance(manifest, str) or not _HEX_64_RE.fullmatch(manifest):
        raise S3BackupError("LATEST contains an invalid manifest fingerprint")
    completion = _CompletionRef(
        key=cast(str, document.get("completion_key")),
        version_id=cast(str, document.get("completion_version_id")),
    )
    expected_key = target.key("_throwdb_backup", "completions", f"{backup_id}.json")
    if (
        completion.key != expected_key
        or not _is_immutable_version_id(completion.version_id)
    ):
        raise S3BackupError("LATEST references an invalid completion object")
    refs_value = document.get("slot_completions")
    if not isinstance(refs_value, dict):
        raise S3BackupError("LATEST has no slot completion inventory")
    refs: dict[str, _CompletionRef] = {}
    for named_slot, value in refs_value.items():
        if named_slot not in _SLOTS:
            raise S3BackupError("LATEST has an unexpected slot completion")
        refs[named_slot] = _completion_ref(value, f"LATEST {named_slot} completion")
    if refs.get(cast(str, slot)) != completion:
        raise S3BackupError("LATEST active slot completion disagrees with its pointer")
    return _LatestInfo(
        slot=cast(str, slot),
        backup_id=backup_id,
        completion=completion,
        manifest_sha256=manifest,
        source_state=_source_state_from_document(document.get("source_state"), "LATEST"),
        completed_at=_parse_timestamp(document.get("completed_at"), "LATEST"),
        slot_completions=refs,
    )


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S3BackupError("completion contains an invalid object path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise S3BackupError("completion contains an unsafe object path")
    if any(ord(character) < 32 for character in value):
        raise S3BackupError("completion object path contains a control character")
    if _excluded_local_path(value):
        raise S3BackupError("completion unexpectedly contains a local cursor file")
    return value


def _remote_file_entry(value: object) -> _RemoteFileEntry:
    if not isinstance(value, dict):
        raise S3BackupError("completion object inventory has an invalid entry")
    path = _safe_relative_path(value.get("path"))
    size = _integer(value.get("size"), f"completion size for {path}")
    mode = _integer(value.get("mode"), f"completion mode for {path}")
    digest = value.get("sha256")
    etag = value.get("etag")
    version = value.get("version_id")
    if mode > 0o7777:
        raise S3BackupError(f"completion mode for {path} is invalid")
    if not isinstance(digest, str) or not _HEX_64_RE.fullmatch(digest):
        raise S3BackupError(f"completion digest for {path} is invalid")
    if not isinstance(etag, str) or not etag:
        raise S3BackupError(f"completion ETag for {path} is invalid")
    if not _is_immutable_version_id(version):
        raise S3BackupError(f"completion VersionId for {path} is invalid")
    return _RemoteFileEntry(path, size, mode, digest, etag, version)


def _directory_entry(value: object) -> _DirectoryEntry:
    if not isinstance(value, dict):
        raise S3BackupError("completion directory inventory has an invalid entry")
    path = _safe_relative_path(value.get("path"))
    mode = _integer(value.get("mode"), f"completion directory mode for {path}")
    if mode > 0o7777:
        raise S3BackupError(f"completion directory mode for {path} is invalid")
    return _DirectoryEntry(path, mode)


def _validate_completion(
    document: Mapping[str, object],
    *,
    backup_id: str,
    slot: str,
    expected_state: _SourceState | None = None,
    expected_manifest: str | None = None,
) -> tuple[
    _SourceState,
    tuple[_RemoteFileEntry, ...],
    tuple[_DirectoryEntry, ...],
]:
    if (
        document.get("format") != _BACKUP_FORMAT
        or document.get("version") != _FORMAT_VERSION
        or document.get("backup_id") != backup_id
        or document.get("slot") != slot
    ):
        raise S3BackupError("completion object is inconsistent with its reference")
    state = _source_state_from_document(document.get("source_state"), "completion")
    if expected_state is not None and state != expected_state:
        raise S3BackupError("completion source state disagrees with LATEST")
    source_inventory = document.get("source_inventory")
    if not isinstance(source_inventory, dict):
        raise S3BackupError("completion has no source inventory")
    if (
        expected_manifest is not None
        and source_inventory.get("manifest_sha256") != expected_manifest
    ):
        raise S3BackupError("completion has a different manifest fingerprint")
    objects_value = document.get("objects")
    if not isinstance(objects_value, list) or len(objects_value) > _MAX_INVENTORY_FILES:
        raise S3BackupError("completion has an invalid object inventory")
    entries = tuple(_remote_file_entry(value) for value in objects_value)
    if tuple(entry.path for entry in entries) != tuple(
        sorted(entry.path for entry in entries)
    ) or len({entry.path for entry in entries}) != len(entries):
        raise S3BackupError("completion object inventory is not unique and sorted")
    expected_count = source_inventory.get("file_count")
    expected_bytes = source_inventory.get("total_bytes")
    if expected_count != len(entries) or expected_bytes != sum(
        entry.size for entry in entries
    ):
        raise S3BackupError("completion object inventory totals are inconsistent")
    directories_value = document.get("directories")
    if not isinstance(directories_value, list) or len(directories_value) > _MAX_INVENTORY_FILES:
        raise S3BackupError("completion has an invalid directory inventory")
    directories = tuple(_directory_entry(value) for value in directories_value)
    if tuple(entry.path for entry in directories) != tuple(
        sorted(entry.path for entry in directories)
    ) or len({entry.path for entry in directories}) != len(directories):
        raise S3BackupError("completion directory inventory is not unique and sorted")
    if source_inventory.get("directory_count") != len(directories):
        raise S3BackupError("completion directory inventory total is inconsistent")
    return state, entries, directories


def _ensure_lineage(
    aws: _AwsCli,
    target: _S3Target,
    source: _SourceState,
    manifest_sha256: str,
    deadline: _Deadline,
    *,
    dry_run: bool,
    require_version: bool,
) -> _ControlSnapshot | None:
    key = target.key("_throwdb_backup", "LINEAGE.json")
    if _object_exists(aws, target, key, deadline):
        snapshot = _download_control(
            aws,
            target,
            key,
            "LINEAGE",
            deadline,
            require_version=require_version,
        )
        _validate_lineage(snapshot.document, source, manifest_sha256)
        return snapshot
    if _destination_has_objects(aws, target, deadline):
        raise S3BackupError(
            "the S3 destination prefix is not empty but has no LINEAGE marker; "
            "refusing to adopt or overwrite unverified objects"
        )
    if dry_run:
        return None
    document = {
        "format": _LINEAGE_FORMAT,
        "version": _FORMAT_VERSION,
        "collection_id": source.collection_id,
        "manifest_sha256": manifest_sha256,
        "created_at": _utc_now_text(),
    }
    try:
        snapshot = _put_json_conditional(
            aws,
            target,
            key,
            document,
            deadline,
            expected_etag=None,
            require_version=require_version,
        )
    except S3BackupError as create_error:
        try:
            snapshot = _download_control(
                aws,
                target,
                key,
                "LINEAGE reconciliation read-back",
                deadline,
                require_version=require_version,
            )
        except S3BackupError as readback_error:
            raise S3BackupError(
                "conditional LINEAGE creation failed and could not be reconciled; "
                f"refusing to adopt the prefix ({create_error}; {readback_error})"
            ) from create_error
        _validate_lineage(snapshot.document, source, manifest_sha256)
    return snapshot


def _validate_lease_document(
    document: Mapping[str, object],
) -> tuple[str, str, dt.datetime]:
    if (
        document.get("format") != _LEASE_FORMAT
        or document.get("version") != _FORMAT_VERSION
    ):
        raise S3BackupError("remote publisher lease has an unsupported format")
    state = document.get("state")
    holder = document.get("holder")
    if (
        state not in {"active", "released"}
        or not isinstance(holder, str)
        or not _HEX_32_RE.fullmatch(holder)
    ):
        raise S3BackupError("remote publisher lease contains invalid state")
    expires = _parse_timestamp(document.get("expires_at"), "remote publisher lease")
    return cast(str, state), holder, expires


def _acquire_remote_lease(
    aws: _AwsCli,
    target: _S3Target,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> _RemoteLease:
    key = target.key("_throwdb_backup", "LEASE.json")
    holder = uuid.uuid4().hex
    now = _utc_now()
    expires = now + dt.timedelta(
        seconds=deadline.remaining() + _LEASE_EXPIRY_MARGIN_SECONDS
    )
    document = {
        "format": _LEASE_FORMAT,
        "version": _FORMAT_VERSION,
        "state": "active",
        "holder": holder,
        "acquired_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
    }
    expected_etag: str | None = None
    if _object_exists(aws, target, key, deadline):
        existing = _download_control(
            aws,
            target,
            key,
            "remote publisher lease",
            deadline,
            require_version=require_version,
        )
        state, existing_holder, existing_expiry = _validate_lease_document(
            existing.document
        )
        # A large post-expiry guard absorbs ordinary host clock skew.  Unknown,
        # malformed, or merely near-expired leases remain fail-closed.
        safely_expired = _utc_now() > existing_expiry + dt.timedelta(
            seconds=_LEASE_EXPIRY_MARGIN_SECONDS
        )
        if state == "active" and not safely_expired:
            raise S3BackupError(
                "another publisher holds the remote S3 lease "
                f"({existing_holder}); refusing concurrent slot/LATEST changes"
            )
        expected_etag = existing.etag
    try:
        snapshot = _put_json_conditional(
            aws,
            target,
            key,
            document,
            deadline,
            expected_etag=expected_etag,
            require_version=require_version,
        )
    except S3BackupError as error:
        raise S3BackupError(
            "could not acquire the remote S3 publisher lease with one conditional "
            f"write; refusing to retry after a possible CAS conflict: {error}"
        ) from error
    return _RemoteLease(key, holder, snapshot, expires)


def _assert_remote_lease(
    aws: _AwsCli,
    target: _S3Target,
    lease: _RemoteLease,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> None:
    deadline.remaining()
    if _utc_now() >= lease.expires_at - dt.timedelta(
        seconds=_LEASE_EXPIRY_MARGIN_SECONDS
    ):
        raise S3BackupError("remote S3 publisher lease safety window expired")
    current = _head_object(
        aws,
        target,
        lease.key,
        deadline,
        require_version=require_version,
    )
    if (
        current.etag != lease.snapshot.etag
        or current.version_id != lease.snapshot.version_id
    ):
        raise S3BackupError("remote S3 publisher lease was replaced; aborting")


def _release_remote_lease(
    aws: _AwsCli,
    target: _S3Target,
    lease: _RemoteLease,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> None:
    _assert_remote_lease(
        aws, target, lease, deadline, require_version=require_version
    )
    document = dict(lease.snapshot.document)
    document["state"] = "released"
    document["released_at"] = _utc_now_text()
    try:
        _put_json_conditional(
            aws,
            target,
            lease.key,
            document,
            deadline,
            expected_etag=lease.snapshot.etag,
            require_version=require_version,
        )
    except S3BackupError as release_error:
        try:
            current = _download_control(
                aws,
                target,
                lease.key,
                "released publisher lease",
                deadline,
                require_version=require_version,
            )
            state, holder, _ = _validate_lease_document(current.document)
        except S3BackupError:
            raise release_error
        if state != "released" or holder != lease.holder:
            raise release_error


@contextlib.contextmanager
def _remote_publisher_lease(
    aws: _AwsCli,
    target: _S3Target,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> Iterator[_RemoteLease]:
    lease = _acquire_remote_lease(
        aws, target, deadline, require_version=require_version
    )
    try:
        yield lease
    except BaseException:
        try:
            _release_remote_lease(
                aws, target, lease, deadline, require_version=require_version
            )
        except S3BackupError:
            pass
        raise
    else:
        _release_remote_lease(
            aws, target, lease, deadline, require_version=require_version
        )


def _states_equal(left: _SourceState, right: _SourceState) -> bool:
    return left == right


def _guard_source_lineage(
    root: Path,
    target: _S3Target,
    source: _SourceState,
    latest: _LatestInfo | None,
) -> bool:
    """Validate monotonic state and local ancestry; return cursor-repair need."""

    cursor = _read_local_cursor(root, target, source)
    if latest is None:
        if cursor is not None:
            raise S3BackupError(
                "local collection remembers a published backup but remote LATEST "
                "is absent; refusing a destructive destination reset"
            )
        return False
    previous = latest.source_state
    if previous.collection_id != source.collection_id:
        raise S3BackupError("LATEST belongs to a different ThrowDB collection")
    if source.catalog_generation < previous.catalog_generation:
        raise S3BackupError(
            "source catalog generation is older than LATEST; stale rollback rejected"
        )
    if source.committed_sequence < previous.committed_sequence:
        raise S3BackupError(
            "source committed sequence is older than LATEST; stale rollback rejected"
        )
    if (
        source.catalog_generation == previous.catalog_generation
        and source.committed_sequence == previous.committed_sequence
        and (
            source.live_count != previous.live_count
            or source.authoritative_sha256 != previous.authoritative_sha256
        )
    ):
        raise S3BackupError(
            "source reuses the published generation/sequence with different "
            "authoritative bytes; divergent lineage rejected"
        )
    if cursor is not None and cursor.get("backup_id") == latest.backup_id:
        return False
    # This exact-state exception repairs the narrow crash window after LATEST
    # was committed but before the local cursor was fsynced.  It cannot admit a
    # stale branch that has advanced independently.
    if _states_equal(source, previous):
        return True
    if cursor is None:
        raise S3BackupError(
            "source has advanced beyond LATEST but has no local ancestry cursor; "
            "refusing to infer that a copied/forked collection is authoritative"
        )
    raise S3BackupError(
        "local ancestry cursor does not name current LATEST; stale or forked "
        "source publication rejected"
    )


def _load_latest(
    aws: _AwsCli,
    target: _S3Target,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> tuple[_ControlSnapshot, _LatestInfo] | None:
    key = target.key("_throwdb_backup", "LATEST.json")
    if not _object_exists(aws, target, key, deadline):
        return None
    snapshot = _download_control(
        aws,
        target,
        key,
        "LATEST",
        deadline,
        require_version=require_version,
    )
    return snapshot, _validate_latest(snapshot.document, target)


def _load_completion(
    aws: _AwsCli,
    target: _S3Target,
    reference: _CompletionRef,
    deadline: _Deadline,
    *,
    backup_id: str,
    slot: str,
    require_version: bool,
    expected_state: _SourceState | None = None,
    expected_manifest: str | None = None,
) -> tuple[
    _ControlSnapshot,
    _SourceState,
    tuple[_RemoteFileEntry, ...],
    tuple[_DirectoryEntry, ...],
]:
    snapshot = _download_control(
        aws,
        target,
        reference.key,
        "completion object",
        deadline,
        version_id=reference.version_id,
        require_version=require_version,
    )
    state, entries, directories = _validate_completion(
        snapshot.document,
        backup_id=backup_id,
        slot=slot,
        expected_state=expected_state,
        expected_manifest=expected_manifest,
    )
    return snapshot, state, entries, directories


def _slot_plan(
    aws: _AwsCli,
    target: _S3Target,
    slot: str,
    inventory: _Inventory,
    previous: Sequence[_RemoteFileEntry],
    deadline: _Deadline,
    *,
    require_version: bool,
) -> tuple[
    tuple[_FileEntry, ...],
    tuple[str, ...],
    dict[str, _ObjectSnapshot],
]:
    prefix = target.key(slot) + "/"
    listed = _list_current_versions(aws, target, prefix, deadline)
    previous_by_path = {entry.path: entry for entry in previous}
    changed: list[_FileEntry] = []
    expected_keys: set[str] = set()
    for entry in inventory.files:
        key = target.key(slot, entry.path)
        expected_keys.add(key)
        prior = previous_by_path.get(entry.path)
        if (
            prior is None
            or prior.size != entry.size
            or prior.mode != entry.mode
            or prior.sha256 != entry.sha256
            or key not in listed
        ):
            changed.append(entry)
            continue
        current = listed[key]
        if (
            current.size != entry.size
            or current.etag != prior.etag
            or current.version_id != prior.version_id
        ):
            changed.append(entry)
    extras = tuple(sorted(set(listed) - expected_keys))
    return tuple(changed), extras, listed


def _publish_slot(
    aws: _AwsCli,
    target: _S3Target,
    slot: str,
    root: Path,
    inventory: _Inventory,
    previous: Sequence[_RemoteFileEntry],
    lease: _RemoteLease,
    deadline: _Deadline,
    *,
    require_version: bool,
) -> tuple[_RemoteFileEntry, ...]:
    changed, extras, before = _slot_plan(
        aws,
        target,
        slot,
        inventory,
        previous,
        deadline,
        require_version=require_version,
    )
    _assert_remote_lease(
        aws, target, lease, deadline, require_version=require_version
    )
    aws.sync_directory(
        root,
        target.uri(slot, directory=True),
        timeout=deadline.remaining(),
    )
    # sync is intentionally the bulk transport, not the authority for byte
    # equality.  Force every changed pre-existing key because high-level sync
    # may skip equal-size files with an older timestamp.
    for position, entry in enumerate(changed):
        key = target.key(slot, entry.path)
        if key not in before:
            continue
        if position % 128 == 0:
            _assert_remote_lease(
                aws, target, lease, deadline, require_version=require_version
            )
        aws.copy_file(root / entry.path, target.uri_for_key(key), timeout=deadline.remaining())
    for position, key in enumerate(extras):
        if position % 128 == 0:
            _assert_remote_lease(
                aws, target, lease, deadline, require_version=require_version
            )
        _delete_object(aws, target, key, deadline)

    _assert_remote_lease(
        aws, target, lease, deadline, require_version=require_version
    )
    prefix = target.key(slot) + "/"
    final = _list_current_versions(aws, target, prefix, deadline)
    expected_keys = {target.key(slot, entry.path) for entry in inventory.files}
    if set(final) != expected_keys:
        raise S3BackupError("S3 slot did not converge to the exact local file set")
    result: list[_RemoteFileEntry] = []
    for entry in inventory.files:
        key = target.key(slot, entry.path)
        current = final[key]
        if current.size != entry.size:
            raise S3BackupError(
                f"S3 object size differs after slot publication: {entry.path}"
            )
        result.append(
            _RemoteFileEntry(
                entry.path,
                entry.size,
                entry.mode,
                entry.sha256,
                current.etag,
                current.version_id,
            )
        )
    return tuple(result)


def _validate_float(
    value: object, description: str, *, allow_zero: bool
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        qualifier = "zero or greater" if allow_zero else "greater than zero"
        raise ValueError(f"{description} must be finite and {qualifier}")
    return float(value)


def _bucket_versioning(
    aws: _AwsCli,
    target: _S3Target,
    deadline: _Deadline,
    *,
    unsafe_allow_unversioned: bool,
) -> tuple[str, bool]:
    if unsafe_allow_unversioned:
        return "unsafe_check_skipped", False
    versioning = aws.run_json(
        [
            "s3api",
            "get-bucket-versioning",
            "--bucket",
            target.bucket,
            "--output",
            "json",
        ],
        timeout=deadline.remaining(),
    )
    status = str(versioning.get("Status", "Disabled"))
    if status != "Enabled":
        raise S3BackupError(
            "S3 bucket Versioning must be Enabled before publishing ThrowDB "
            f"checkpoints (reported status: {status})"
        )
    return status, True


def _backup_id_from_reference(reference: _CompletionRef, target: _S3Target) -> str:
    prefix = target.key("_throwdb_backup", "completions") + "/"
    if not reference.key.startswith(prefix) or not reference.key.endswith(".json"):
        raise S3BackupError("slot completion reference has an unexpected key")
    backup_id = reference.key[len(prefix) : -5]
    if not _HEX_32_RE.fullmatch(backup_id):
        raise S3BackupError("slot completion reference has an invalid backup ID")
    return backup_id


def backup_to_s3(
    storage: str | os.PathLike[str],
    destination: str,
    *,
    aws_executable: str | os.PathLike[str] = "aws",
    dry_run: bool = False,
    unsafe_allow_unversioned: bool = False,
    if_older_than_hours: float = 0.0,
    metadata_timeout_seconds: float = 60.0,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> S3BackupReport:
    """Publish one verified, monotonic collection checkpoint.

    The writer lease covers source hashing, complete native verification, slot
    upload, completion publication, the conditional ``LATEST`` write, and the
    local lineage cursor.  A dry-run performs every read-only preflight and an
    exact object plan but creates neither LINEAGE nor the remote lease.
    """

    if not isinstance(dry_run, bool) or not isinstance(
        unsafe_allow_unversioned, bool
    ):
        raise TypeError("dry_run and unsafe_allow_unversioned must be bool values")
    older_hours = _validate_float(
        if_older_than_hours, "if_older_than_hours", allow_zero=True
    )
    metadata_timeout = _validate_float(
        metadata_timeout_seconds, "metadata_timeout_seconds", allow_zero=False
    )
    maximum_seconds = _validate_float(
        max_seconds, "max_seconds", allow_zero=False
    )
    try:
        root = Path(storage).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise S3BackupError(f"cannot resolve collection storage: {error}") from error
    if not root.is_dir():
        raise S3BackupError(f"collection storage is not a directory: {root}")
    target = _parse_target(destination)
    aws = _AwsCli(_resolve_aws_executable(aws_executable), metadata_timeout)
    deadline = _Deadline(maximum_seconds)
    latest_key = target.key("_throwdb_backup", "LATEST.json")

    with _writer_lease(root):
        inventory = _inventory(root, deadline)
        source = _verified_source_state(root, inventory)
        versioning_status, require_version = _bucket_versioning(
            aws,
            target,
            deadline,
            unsafe_allow_unversioned=unsafe_allow_unversioned,
        )
        # The fully hardened format relies on immutable object versions.  Keep
        # the old escape hatch explicit, but do not silently claim v2 safety.
        if not require_version:
            raise S3BackupError(
                "hardened version-2 checkpoints require S3 Versioning; "
                "--unsafe-allow-unversioned is no longer sufficient"
            )
        _ensure_lineage(
            aws,
            target,
            source,
            inventory.manifest_sha256,
            deadline,
            dry_run=dry_run,
            require_version=require_version,
        )

        def execute(lease: _RemoteLease | None) -> S3BackupReport:
            latest_pair = _load_latest(
                aws,
                target,
                deadline,
                require_version=require_version,
            )
            latest_snapshot: _ControlSnapshot | None = None
            latest: _LatestInfo | None = None
            if latest_pair is not None:
                latest_snapshot, latest = latest_pair
                if (
                    latest.manifest_sha256 != inventory.manifest_sha256
                    or latest.source_state.collection_id != source.collection_id
                ):
                    raise S3BackupError(
                        "the S3 destination belongs to a different ThrowDB collection"
                    )
                _load_completion(
                    aws,
                    target,
                    latest.completion,
                    deadline,
                    backup_id=latest.backup_id,
                    slot=latest.slot,
                    require_version=require_version,
                    expected_state=latest.source_state,
                    expected_manifest=latest.manifest_sha256,
                )
                if latest.completed_at > _utc_now():
                    raise S3BackupError(
                        "LATEST completion timestamp is in the future; refusing "
                        "to make a retention decision from an invalid clock value"
                    )

            repair_cursor = _guard_source_lineage(root, target, source, latest)
            previous_slot = latest.slot if latest is not None else None
            target_slot = "slot-a" if previous_slot != "slot-a" else "slot-b"
            report: S3BackupReport = {
                "format": _BACKUP_FORMAT,
                "version": _FORMAT_VERSION,
                "status": "dry_run" if dry_run else "completed",
                "dry_run": dry_run,
                "storage": str(root),
                "destination": target.uri(directory=True),
                "bucket_versioning": versioning_status,
                "previous_slot": previous_slot,
                "target_slot": target_slot,
                "backup_id": None,
                "completion_uri": None,
                "latest_uri": target.uri_for_key(latest_key),
                "latest_publication_reconciled": False,
                "source_inventory": inventory.public(),
                "source_state": source.document(),
                "planned_change_count": None,
                "reason": None,
            }

            if latest is not None and older_hours > 0:
                age = (_utc_now() - latest.completed_at).total_seconds()
                if age < older_hours * 3600.0:
                    report["status"] = "skipped"
                    report["reason"] = (
                        "latest completed backup is younger than "
                        f"{older_hours:g} hours"
                    )
                    if repair_cursor and not dry_run:
                        _write_local_cursor(
                            root, target, source, latest.backup_id
                        )
                    return report

            previous_entries: tuple[_RemoteFileEntry, ...] = ()
            if latest is not None:
                target_reference = latest.slot_completions.get(target_slot)
                if target_reference is not None:
                    target_backup_id = _backup_id_from_reference(
                        target_reference, target
                    )
                    _, _, previous_entries, _ = _load_completion(
                        aws,
                        target,
                        target_reference,
                        deadline,
                        backup_id=target_backup_id,
                        slot=target_slot,
                        require_version=require_version,
                        expected_manifest=inventory.manifest_sha256,
                    )

            if dry_run:
                changed, extras, _ = _slot_plan(
                    aws,
                    target,
                    target_slot,
                    inventory,
                    previous_entries,
                    deadline,
                    require_version=require_version,
                )
                report["planned_change_count"] = len(changed) + len(extras)
                return report

            if lease is None:
                raise AssertionError("actual checkpoint has no remote lease")
            remote_entries = _publish_slot(
                aws,
                target,
                target_slot,
                root,
                inventory,
                previous_entries,
                lease,
                deadline,
                require_version=require_version,
            )
            backup_id = uuid.uuid4().hex
            completed_at = _utc_now_text()
            completion_key = target.key(
                "_throwdb_backup", "completions", f"{backup_id}.json"
            )
            completion_document: dict[str, object] = {
                "format": _BACKUP_FORMAT,
                "version": _FORMAT_VERSION,
                "backup_id": backup_id,
                "parent_backup_id": latest.backup_id if latest is not None else None,
                "slot": target_slot,
                "slot_uri": target.uri(target_slot, directory=True),
                "completed_at": completed_at,
                "source_inventory": inventory.public(),
                "source_state": source.document(),
                "objects": [entry.document() for entry in remote_entries],
                "directories": [
                    entry.document() for entry in inventory.directories
                ],
                "validation": {
                    "native_verify_index": True,
                    "source_stable_under_writer_lease": True,
                    "per_object_sha256": True,
                    "exact_s3_version_ids": True,
                },
            }
            completion_snapshot, _ = _publish_json_reconciled(
                aws,
                target,
                completion_key,
                completion_document,
                deadline,
                expected_etag=None,
                require_version=require_version,
                description="completion object",
            )
            references = (
                dict(latest.slot_completions) if latest is not None else {}
            )
            references[target_slot] = _CompletionRef(
                completion_key, completion_snapshot.version_id
            )
            latest_document: dict[str, object] = {
                "format": _LATEST_FORMAT,
                "version": _FORMAT_VERSION,
                "backup_id": backup_id,
                "parent_backup_id": latest.backup_id if latest is not None else None,
                "slot": target_slot,
                "slot_uri": target.uri(target_slot, directory=True),
                "completed_at": completed_at,
                "completion_key": completion_key,
                "completion_version_id": completion_snapshot.version_id,
                "manifest_sha256": inventory.manifest_sha256,
                "collection_id": source.collection_id,
                "source_state": source.document(),
                "slot_completions": {
                    slot: reference.document()
                    for slot, reference in sorted(references.items())
                },
            }
            _assert_remote_lease(
                aws,
                target,
                lease,
                deadline,
                require_version=require_version,
            )
            _, reconciled = _publish_json_reconciled(
                aws,
                target,
                latest_key,
                latest_document,
                deadline,
                expected_etag=(
                    latest_snapshot.etag if latest_snapshot is not None else None
                ),
                require_version=require_version,
                description="LATEST",
            )
            _write_local_cursor(root, target, source, backup_id)
            report["backup_id"] = backup_id
            report["completion_uri"] = target.uri_for_key(completion_key)
            report["latest_publication_reconciled"] = reconciled
            return report

        if dry_run:
            return execute(None)
        with _remote_publisher_lease(
            aws,
            target,
            deadline,
            require_version=require_version,
        ) as lease:
            return execute(lease)


def _fsync_regular_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise S3BackupError(f"restored path is not a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a racing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    else:
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise S3BackupError(
                "this platform has no atomic no-replace directory rename primitive"
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise S3BackupError(
            f"restore destination appeared during atomic publication: {destination}"
        )
    raise S3BackupError(
        "atomic restore publication failed: "
        f"{os.strerror(error_number)} ({error_number})"
    )


def restore_from_s3(
    source: str,
    storage: str | os.PathLike[str],
    *,
    aws_executable: str | os.PathLike[str] = "aws",
    expected_backup_id: str | None = None,
    metadata_timeout_seconds: float = 60.0,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> S3RestoreReport:
    """Restore a version-pinned current slot into a verified atomic directory."""

    metadata_timeout = _validate_float(
        metadata_timeout_seconds, "metadata_timeout_seconds", allow_zero=False
    )
    maximum_seconds = _validate_float(
        max_seconds, "max_seconds", allow_zero=False
    )
    if expected_backup_id is not None and (
        not isinstance(expected_backup_id, str)
        or not _HEX_32_RE.fullmatch(expected_backup_id)
    ):
        raise ValueError("expected_backup_id must be a 32-character lowercase hex ID")
    requested = Path(storage).expanduser()
    if os.path.lexists(requested):
        raise S3BackupError(f"restore destination already exists: {requested}")
    try:
        parent = requested.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise S3BackupError(f"cannot resolve restore destination parent: {error}") from error
    if not parent.is_dir():
        raise S3BackupError(f"restore destination parent is not a directory: {parent}")
    destination = parent / requested.name
    target = _parse_target(source)
    aws = _AwsCli(_resolve_aws_executable(aws_executable), metadata_timeout)
    deadline = _Deadline(maximum_seconds)
    _, require_version = _bucket_versioning(
        aws,
        target,
        deadline,
        unsafe_allow_unversioned=False,
    )
    latest_pair = _load_latest(
        aws, target, deadline, require_version=require_version
    )
    if latest_pair is None:
        raise S3BackupError("S3 checkpoint prefix has no published LATEST")
    _, latest = latest_pair
    if expected_backup_id is not None and latest.backup_id != expected_backup_id:
        raise S3BackupError(
            f"LATEST backup ID {latest.backup_id} differs from expected "
            f"{expected_backup_id}"
        )
    lineage_key = target.key("_throwdb_backup", "LINEAGE.json")
    if not _object_exists(aws, target, lineage_key, deadline):
        raise S3BackupError("S3 checkpoint prefix has no LINEAGE marker")
    lineage = _download_control(
        aws,
        target,
        lineage_key,
        "LINEAGE",
        deadline,
        require_version=require_version,
    )
    _validate_lineage(
        lineage.document, latest.source_state, latest.manifest_sha256
    )
    _, completion_state, entries, directories = _load_completion(
        aws,
        target,
        latest.completion,
        deadline,
        backup_id=latest.backup_id,
        slot=latest.slot,
        require_version=require_version,
        expected_state=latest.source_state,
        expected_manifest=latest.manifest_sha256,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=parent)
    )
    published = False
    try:
        os.chmod(staging, 0o700)
        slot_prefix = target.key(latest.slot) + "/"
        expected_remote = {
            target.key(latest.slot, entry.path): _ObjectSnapshot(
                entry.size, entry.etag, entry.version_id
            )
            for entry in entries
        }
        before = _list_current_versions(aws, target, slot_prefix, deadline)
        if before != expected_remote:
            raise S3BackupError(
                "published slot current VersionIds differ from its completion; "
                "refusing a mixed-version restore"
            )
        aws.sync_from_s3(
            target.uri(latest.slot, directory=True),
            staging,
            timeout=deadline.remaining(),
        )
        after = _list_current_versions(aws, target, slot_prefix, deadline)
        if after != expected_remote:
            raise S3BackupError(
                "published slot changed during restore; local staging was discarded"
            )
        for directory_entry in sorted(
            directories, key=lambda entry: len(PurePosixPath(entry.path).parts)
        ):
            directory = staging / directory_entry.path
            directory.mkdir(mode=directory_entry.mode, parents=True, exist_ok=True)
            os.chmod(directory, directory_entry.mode)
        for entry in entries:
            deadline.remaining()
            output = staging / entry.path
            try:
                info = output.stat(follow_symlinks=False)
            except OSError as error:
                raise S3BackupError(
                    f"cannot inspect restored object {entry.path}: {error}"
                ) from error
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry.size:
                raise S3BackupError(
                    f"restored object size/type mismatch for {entry.path}"
                )
            digest = _hash_regular_file(output, info, deadline)
            if digest != entry.sha256:
                raise S3BackupError(
                    f"restored object SHA-256 mismatch for {entry.path}"
                )
            os.chmod(output, entry.mode)
            _fsync_regular_file(output)

        restored_inventory = _inventory(staging, deadline)
        expected_files = tuple(
            (entry.path, entry.size, entry.mode, entry.sha256) for entry in entries
        )
        actual_files = tuple(
            (entry.path, entry.size, entry.mode, entry.sha256)
            for entry in restored_inventory.files
        )
        if actual_files != expected_files:
            raise S3BackupError(
                "restored local inventory differs from the checksummed completion inventory"
            )
        expected_directories = tuple(
            (entry.path, entry.mode) for entry in directories
        )
        actual_directories = tuple(
            (entry.path, entry.mode) for entry in restored_inventory.directories
        )
        if actual_directories != expected_directories:
            raise S3BackupError(
                "restored directory inventory differs from the completion inventory"
            )
        restored_state = _verified_source_state(staging, restored_inventory)
        if restored_state != completion_state:
            raise S3BackupError(
                "restored collection state differs from the published source state"
            )
        _write_local_cursor(
            staging, target, restored_state, latest.backup_id
        )
        _fsync_tree_directories(staging)
        if os.path.lexists(destination):
            raise S3BackupError(
                f"restore destination appeared during verification: {destination}"
            )
        _atomic_rename_noreplace(staging, destination)
        published = True
        _fsync_directory(parent)
    except S3BackupError:
        raise
    except OSError as error:
        raise S3BackupError(f"cannot publish restored collection: {error}") from error
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "format": _BACKUP_FORMAT,
        "version": _FORMAT_VERSION,
        "status": "restored",
        "source": target.uri(directory=True),
        "storage": str(destination),
        "backup_id": latest.backup_id,
        "slot": latest.slot,
        "restored_file_count": len(entries),
        "restored_bytes": sum(entry.size for entry in entries),
        "source_state": completion_state.document(),
    }


__all__ = [
    "S3BackupError",
    "S3BackupReport",
    "S3RestoreReport",
    "backup_to_s3",
    "restore_from_s3",
]
