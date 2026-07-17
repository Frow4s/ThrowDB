from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import throwdb.s3_backup as s3_backup_module
from throwdb.cli import _build_parser, main
from throwdb.collection import Collection, CollectionSchema
from throwdb.s3_backup import (
    S3BackupError,
    _writer_lease,
    backup_to_s3,
    restore_from_s3,
)


_FAKE_AWS = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

args = sys.argv[1:]
root = Path(os.environ["FAKE_S3_ROOT"])
root.mkdir(parents=True, exist_ok=True)
log = Path(os.environ["FAKE_AWS_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")
state_path = root / ".state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"objects": {}}

def save():
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True))
    temporary.replace(state_path)

def option(name, default=None):
    try:
        return args[args.index(name) + 1]
    except ValueError:
        return default

def remote(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "s3":
        raise RuntimeError(uri)
    return parsed.netloc, parsed.path.lstrip("/")

def identity(bucket, key):
    return bucket + "/" + key

def current(bucket, key):
    record = state["objects"].get(identity(bucket, key))
    return None if record is None else record.get("current")

def version_path(version):
    return root / ".versions" / version

def visible_path(bucket, key):
    return root / bucket / key

def publish(bucket, key, data, persist=True):
    version = (
        "null"
        if os.environ.get("FAKE_AWS_NULL_VERSION") == "1"
        else uuid.uuid4().hex
    )
    etag = '"' + hashlib.md5(data).hexdigest() + '"'
    path = version_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    record = state["objects"].setdefault(identity(bucket, key), {"versions": {}})
    record["versions"][version] = {"etag": etag, "size": len(data)}
    record["current"] = version
    visible = visible_path(bucket, key)
    visible.parent.mkdir(parents=True, exist_ok=True)
    visible.write_bytes(data)
    if persist:
        save()
    return version, etag

def selected(bucket, key):
    record = state["objects"].get(identity(bucket, key))
    version = option("--version-id") or (record or {}).get("current")
    if record is None or version not in record.get("versions", {}):
        return None
    metadata = record["versions"][version]
    return record, version, metadata, version_path(version)

if args[:2] == ["s3api", "get-bucket-versioning"]:
    status = os.environ.get("FAKE_AWS_VERSIONING", "Enabled")
    print(json.dumps({} if status == "Disabled" else {"Status": status}))
    raise SystemExit(0)

if args[:2] == ["s3api", "list-objects-v2"]:
    bucket = option("--bucket")
    prefix = option("--prefix", "")
    keys = []
    for name, record in state["objects"].items():
        item_bucket, key = name.split("/", 1)
        version = record.get("current")
        if item_bucket == bucket and version and key.startswith(prefix):
            keys.append((key, record["versions"][version]["size"]))
    bucket_root = root / bucket
    if bucket_root.exists():
        for path in bucket_root.rglob("*"):
            if path.is_file():
                key = path.relative_to(bucket_root).as_posix()
                if key.startswith(prefix) and all(existing[0] != key for existing in keys):
                    keys.append((key, path.stat().st_size))
    keys.sort()
    start = int(option("--continuation-token", "0"))
    maximum = int(option("--max-keys", "1000"))
    page = keys[start:start + maximum]
    result = {
        "Contents": [{"Key": key, "Size": size} for key, size in page],
        "KeyCount": len(page),
    }
    if start + maximum < len(keys):
        result["IsTruncated"] = True
        result["NextContinuationToken"] = str(start + maximum)
    else:
        result["IsTruncated"] = False
    print(json.dumps(result))
    raise SystemExit(0)

if args[:2] == ["s3api", "list-object-versions"]:
    bucket = option("--bucket")
    prefix = option("--prefix", "")
    rows = []
    for name, record in state["objects"].items():
        item_bucket, key = name.split("/", 1)
        if item_bucket != bucket or not key.startswith(prefix):
            continue
        for version, metadata in record.get("versions", {}).items():
            rows.append({
                "Key": key,
                "VersionId": version,
                "ETag": metadata["etag"],
                "Size": metadata["size"],
                "IsLatest": record.get("current") == version,
            })
    rows.sort(key=lambda item: (item["Key"], item["VersionId"]))
    start = int(option("--key-marker", "0"))
    maximum = int(option("--max-keys", "1000"))
    page = rows[start:start + maximum]
    result = {"Versions": page, "DeleteMarkers": [], "IsTruncated": start + maximum < len(rows)}
    if result["IsTruncated"]:
        result["NextKeyMarker"] = str(start + maximum)
        result["NextVersionIdMarker"] = "page"
    print(json.dumps(result))
    raise SystemExit(0)

if args[:2] == ["s3api", "head-object"]:
    chosen = selected(option("--bucket"), option("--key"))
    if chosen is None:
        print("not found", file=sys.stderr)
        raise SystemExit(1)
    _, version, metadata, _ = chosen
    print(json.dumps({
        "ContentLength": metadata["size"],
        "ETag": metadata["etag"],
        "VersionId": version,
    }))
    raise SystemExit(0)

if args[:2] == ["s3api", "get-object"]:
    bucket, key = option("--bucket"), option("--key")
    if os.environ.get("FAKE_AWS_FAIL_LATEST_READ") == "1" and key.endswith("LATEST.json"):
        print("injected LATEST read failure", file=sys.stderr)
        raise SystemExit(33)
    chosen = selected(bucket, key)
    if chosen is None:
        print("not found", file=sys.stderr)
        raise SystemExit(1)
    _, version, metadata, path = chosen
    if option("--if-match") not in (None, metadata["etag"]):
        print("precondition failed", file=sys.stderr)
        raise SystemExit(412)
    output = Path(args[-1])
    shutil.copyfile(path, output)
    print(json.dumps({"ETag": metadata["etag"], "VersionId": version}))
    raise SystemExit(0)

if args[:2] == ["s3api", "put-object"]:
    bucket, key = option("--bucket"), option("--key")
    previous = current(bucket, key)
    record = state["objects"].get(identity(bucket, key))
    previous_etag = None if previous is None else record["versions"][previous]["etag"]
    if "--if-none-match" in args and previous is not None:
        print("precondition failed", file=sys.stderr)
        raise SystemExit(412)
    if "--if-match" in args and option("--if-match") != previous_etag:
        print("precondition failed", file=sys.stderr)
        raise SystemExit(412)
    if os.environ.get("FAKE_AWS_CAS_CONFLICT_LATEST") == "1" and key.endswith("LATEST.json"):
        print("injected CAS conflict", file=sys.stderr)
        raise SystemExit(412)
    data = Path(option("--body")).read_bytes()
    version, etag = publish(bucket, key, data)
    if os.environ.get("FAKE_AWS_COMMIT_THEN_FAIL_LATEST") == "1" and key.endswith("LATEST.json"):
        print("injected failure after commit", file=sys.stderr)
        raise SystemExit(32)
    print(json.dumps({"ETag": etag, "VersionId": version}))
    raise SystemExit(0)

if args[:2] == ["s3api", "delete-object"]:
    bucket, key = option("--bucket"), option("--key")
    record = state["objects"].get(identity(bucket, key))
    if record is not None:
        record["current"] = None
    path = visible_path(bucket, key)
    if path.exists():
        path.unlink()
    save()
    print(json.dumps({"DeleteMarker": True, "VersionId": uuid.uuid4().hex}))
    raise SystemExit(0)

if args[:2] == ["s3", "sync"]:
    if args[2].startswith("s3://"):
        bucket, source_prefix = remote(args[2])
        destination = Path(args[3])
        prefix = source_prefix.rstrip("/") + "/"
        source_files = {}
        for name, record in state["objects"].items():
            item_bucket, key = name.split("/", 1)
            version = record.get("current")
            if item_bucket == bucket and version and key.startswith(prefix):
                source_files[key[len(prefix):]] = version_path(version)
        existing = {
            path.relative_to(destination).as_posix(): path
            for path in destination.rglob("*")
            if path.is_file()
        } if destination.exists() else {}
        for relative, path in source_files.items():
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
        for relative, path in existing.items():
            if relative not in source_files:
                path.unlink()
        if (
            os.environ.get("FAKE_AWS_MUTATE_SLOT_DURING_RESTORE") == "1"
            and source_files
        ):
            relative, path = sorted(source_files.items())[0]
            publish(bucket, prefix + relative, path.read_bytes())
        raise SystemExit(0)
    source = Path(args[2])
    bucket, destination_prefix = remote(args[3])
    source_files = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file()
        and not path.relative_to(source).as_posix().startswith(
            "server/s3-checkpoints/"
        )
    }
    existing = {}
    prefix = destination_prefix.rstrip("/") + "/"
    for name, record in state["objects"].items():
        item_bucket, key = name.split("/", 1)
        version = record.get("current")
        if item_bucket == bucket and version and key.startswith(prefix):
            existing[key[len(prefix):]] = record["versions"][version]["size"]
    if os.environ.get("FAKE_AWS_FAIL_PAYLOAD_AFTER_COMMIT") == "1":
        relative, path = sorted(source_files.items())[0]
        publish(bucket, prefix + relative, path.read_bytes())
        print("injected payload failure after partial upload", file=sys.stderr)
        raise SystemExit(24)
    for relative, path in source_files.items():
        if existing.get(relative) != path.stat().st_size:
            publish(bucket, prefix + relative, path.read_bytes(), persist=False)
    for relative in set(existing) - set(source_files):
        record = state["objects"][identity(bucket, prefix + relative)]
        record["current"] = None
        visible = visible_path(bucket, prefix + relative)
        if visible.exists():
            visible.unlink()
    save()
    raise SystemExit(0)

if args[:2] == ["s3", "cp"]:
    source, destination = args[2], args[3]
    if source.startswith("s3://"):
        raise RuntimeError("use s3api get-object in hardened tests")
    bucket, key = remote(destination)
    data = Path(source).read_bytes()
    version, etag = publish(bucket, key, data)
    if os.environ.get("FAKE_AWS_FAIL_PAYLOAD_AFTER_COMMIT") == "1":
        print("injected payload failure after partial upload", file=sys.stderr)
        raise SystemExit(24)
    raise SystemExit(0)

print("unsupported fake aws invocation: " + repr(args), file=sys.stderr)
raise SystemExit(2)
'''


@pytest.fixture
def fake_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    executable = tmp_path / "fake-aws"
    executable.write_text(_FAKE_AWS, encoding="utf-8")
    executable.chmod(0o755)
    remote = tmp_path / "s3"
    log = tmp_path / "aws.jsonl"
    monkeypatch.setenv("FAKE_S3_ROOT", str(remote))
    monkeypatch.setenv("FAKE_AWS_LOG", str(log))
    return executable, remote, log


def _collection(path: Path, *, marker: int = 1) -> Path:
    with Collection.create(
        path,
        schema=CollectionSchema(dimension=2, dtype="float16"),
        shard_size_bytes=1 << 20,
        format_version=7,
    ) as collection:
        collection.add(
            np.asarray([marker], dtype=np.uint64),
            np.asarray([[float(marker), float(marker + 1)]], dtype=np.float16),
        )
    tuning = path / "tuning"
    tuning.mkdir(exist_ok=True)
    (tuning / "opaque.bin").write_bytes(b"tune-one")
    server = path / "server"
    server.mkdir(exist_ok=True)
    (server / "idempotency-v1.log").write_bytes(b"journal-one")
    return path


def _mutate(path: Path, value: float) -> None:
    with Collection.open(path, mode="r+", tuning="conservative") as collection:
        collection.upsert(
            np.asarray([1], dtype=np.uint64),
            np.asarray([[value, value + 1]], dtype=np.float16),
        )


def _latest(remote: Path) -> dict[str, object]:
    return json.loads(
        (remote / "bucket/backups/_throwdb_backup/LATEST.json").read_text()
    )


def _completion(remote: Path, latest: dict[str, object]) -> dict[str, object]:
    return json.loads((remote / "bucket" / str(latest["completion_key"])).read_text())


def _calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_backup_publishes_versioned_checksum_inventory_and_alternates_slots(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, log = fake_aws
    storage = _collection(tmp_path / "collection")
    first = backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert first["target_slot"] == "slot-a"
    latest = _latest(remote)
    completion = _completion(remote, latest)
    assert latest["version"] == 2
    assert latest["completion_version_id"]
    assert completion["objects"]
    assert all(item["sha256"] and item["version_id"] for item in completion["objects"])
    assert not any("s3-checkpoints" in item["path"] for item in completion["objects"])

    _mutate(storage, 10.0)
    second = backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert second["target_slot"] == "slot-b"
    assert _latest(remote)["parent_backup_id"] == first["backup_id"]
    puts = [call for call in _calls(log) if call[:2] == ["s3api", "put-object"]]
    assert any("--if-none-match" in call for call in puts)
    assert any(
        "--if-match" in call
        and call[call.index("--key") + 1].endswith("LATEST.json")
        for call in puts
    )


def test_same_size_older_mtime_change_is_not_skipped(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    source = storage / "tuning/opaque.bin"
    before = source.stat()
    source.write_bytes(b"tune-two")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns - 1_000_000))
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert (remote / "bucket/backups/slot-a/tuning/opaque.bin").read_bytes() == b"tune-two"


def test_server_journal_change_at_same_logical_view_is_not_a_false_fork(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    first = backup_to_s3(
        storage, "s3://bucket/backups", aws_executable=executable
    )
    journal = storage / "server/idempotency-v1.log"
    journal.write_bytes(b"journal-two-with-checkpoint-metadata")
    (storage / "server/orphan-recovery-note").write_bytes(b"safe orphan")
    second = backup_to_s3(
        storage, "s3://bucket/backups", aws_executable=executable
    )
    assert second["source_state"]["catalog_generation"] == first[
        "source_state"
    ]["catalog_generation"]
    assert second["source_state"]["committed_sequence"] == first[
        "source_state"
    ]["committed_sequence"]
    assert second["source_state"]["authoritative_sha256"] == first[
        "source_state"
    ]["authoritative_sha256"]
    assert second["source_state"]["content_sha256"] != first[
        "source_state"
    ]["content_sha256"]
    assert (
        remote / "bucket/backups/slot-b/server/idempotency-v1.log"
    ).read_bytes() == b"journal-two-with-checkpoint-metadata"


def test_first_partial_publication_is_recoverable_but_never_latest(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    monkeypatch.setenv("FAKE_AWS_FAIL_PAYLOAD_AFTER_COMMIT", "1")
    with pytest.raises(S3BackupError, match="partial upload"):
        backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert (remote / "bucket/backups/_throwdb_backup/LINEAGE.json").is_file()
    assert not (remote / "bucket/backups/_throwdb_backup/LATEST.json").exists()
    monkeypatch.delenv("FAKE_AWS_FAIL_PAYLOAD_AFTER_COMMIT")
    report = backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert report["status"] == "completed"


def test_unowned_nonempty_prefix_and_other_collection_are_rejected(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    orphan = remote / "bucket/backups/orphan"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"x")
    with pytest.raises(S3BackupError, match="no LINEAGE"):
        backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    orphan.unlink()
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    other = _collection(tmp_path / "other", marker=2)
    with pytest.raises(S3BackupError, match="different ThrowDB collection"):
        backup_to_s3(other, "s3://bucket/backups", aws_executable=executable)


def test_stale_and_advanced_fork_cannot_roll_back_latest(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, _, _ = fake_aws
    primary = _collection(tmp_path / "primary")
    backup_to_s3(primary, "s3://bucket/backups", aws_executable=executable)
    stale = tmp_path / "stale"
    shutil.copytree(primary, stale)
    _mutate(primary, 20.0)
    backup_to_s3(primary, "s3://bucket/backups", aws_executable=executable)
    with pytest.raises(S3BackupError, match="older than LATEST"):
        backup_to_s3(stale, "s3://bucket/backups", aws_executable=executable)
    _mutate(stale, 30.0)
    with pytest.raises(S3BackupError, match="cursor|forked"):
        backup_to_s3(stale, "s3://bucket/backups", aws_executable=executable)
    _mutate(stale, 40.0)
    with pytest.raises(S3BackupError, match="cursor|forked"):
        backup_to_s3(stale, "s3://bucket/backups", aws_executable=executable)


def test_latest_cas_conflict_and_commit_then_error_are_fail_closed_or_reconciled(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    first = backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    _mutate(storage, 4.0)
    monkeypatch.setenv("FAKE_AWS_CAS_CONFLICT_LATEST", "1")
    with pytest.raises(S3BackupError, match="CAS conflict|different document"):
        backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert _latest(remote)["backup_id"] == first["backup_id"]
    monkeypatch.delenv("FAKE_AWS_CAS_CONFLICT_LATEST")
    monkeypatch.setenv("FAKE_AWS_COMMIT_THEN_FAIL_LATEST", "1")
    report = backup_to_s3(storage, "s3://bucket/backups2", aws_executable=executable)
    assert report["latest_publication_reconciled"] is True


def test_active_remote_lease_rejects_a_second_publisher(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    lease_path = remote / "bucket/backups/_throwdb_backup/LEASE.json"
    lease = json.loads(lease_path.read_text())
    lease["state"] = "active"
    lease["expires_at"] = "2999-01-01T00:00:00Z"
    data = (json.dumps(lease, indent=2, sort_keys=True) + "\n").encode()
    state_path = remote / ".state.json"
    state = json.loads(state_path.read_text())
    record = state["objects"]["bucket/backups/_throwdb_backup/LEASE.json"]
    version = record["current"]
    (remote / ".versions" / version).write_bytes(data)
    lease_path.write_bytes(data)
    record["versions"][version]["size"] = len(data)
    record["versions"][version]["etag"] = (
        '"' + hashlib.md5(data).hexdigest() + '"'
    )
    state_path.write_text(json.dumps(state, sort_keys=True))

    with pytest.raises(S3BackupError, match="another publisher"):
        backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)


def test_native_preflight_corruption_never_creates_remote_lineage(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    current = storage / "CURRENT"
    data = bytearray(current.read_bytes())
    data[-1] ^= 1
    current.write_bytes(data)
    with pytest.raises(S3BackupError, match="source preflight"):
        backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    assert not (remote / "bucket/backups/_throwdb_backup/LINEAGE.json").exists()


def test_restore_downloads_exact_versions_verifies_and_publishes_atomically(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, _, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    report = backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    restored = tmp_path / "restored"
    restore_report = restore_from_s3(
        "s3://bucket/backups",
        restored,
        aws_executable=executable,
        expected_backup_id=report["backup_id"],
    )
    assert restore_report["status"] == "restored"
    with Collection.open(restored, verify_index=True, tuning="conservative") as collection:
        np.testing.assert_array_equal(
            collection.get(np.asarray([1], dtype=np.uint64)),
            np.asarray([[1.0, 2.0]], dtype=np.float16),
        )
    _mutate(restored, 11.0)
    next_report = backup_to_s3(
        restored, "s3://bucket/backups", aws_executable=executable
    )
    assert next_report["status"] == "completed"


def test_restore_detects_checksum_corruption_and_leaves_no_destination(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    latest = _latest(remote)
    completion = _completion(remote, latest)
    victim = completion["objects"][0]
    version = victim["version_id"]
    version_path = remote / ".versions" / version
    data = bytearray(version_path.read_bytes())
    data[0] ^= 1
    version_path.write_bytes(data)
    destination = tmp_path / "corrupt-restore"
    with pytest.raises(S3BackupError, match="SHA-256 mismatch"):
        restore_from_s3(
            "s3://bucket/backups", destination, aws_executable=executable
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".corrupt-restore.restore-*"))


def test_restore_rejects_wrong_expected_backup_id_before_staging(
    tmp_path: Path, fake_aws: tuple[Path, Path, Path]
) -> None:
    executable, _, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    report = backup_to_s3(
        storage, "s3://bucket/backups", aws_executable=executable
    )
    destination = tmp_path / "wrong-backup"
    wrong = "0" * 32 if report["backup_id"] != "0" * 32 else "1" * 32
    with pytest.raises(S3BackupError, match="differs from expected"):
        restore_from_s3(
            "s3://bucket/backups",
            destination,
            aws_executable=executable,
            expected_backup_id=wrong,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".wrong-backup.restore-*"))


def test_restore_rejects_slot_version_race_after_download(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, _, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    destination = tmp_path / "racing-slot"
    monkeypatch.setenv("FAKE_AWS_MUTATE_SLOT_DURING_RESTORE", "1")
    with pytest.raises(S3BackupError, match="changed during restore"):
        restore_from_s3(
            "s3://bucket/backups", destination, aws_executable=executable
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".racing-slot.restore-*"))


def test_restore_destination_race_cannot_replace_other_path(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, _, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    destination = tmp_path / "racing-destination"
    original = s3_backup_module._atomic_rename_noreplace

    def publish_racing_path(source: Path, target: Path) -> None:
        assert target == destination
        target.mkdir()
        original(source, target)

    monkeypatch.setattr(
        s3_backup_module, "_atomic_rename_noreplace", publish_racing_path
    )
    with pytest.raises(
        S3BackupError, match="destination appeared during atomic publication"
    ):
        restore_from_s3(
            "s3://bucket/backups", destination, aws_executable=executable
        )
    assert destination.is_dir()
    assert not any(destination.iterdir())
    assert not list(tmp_path.glob(".racing-destination.restore-*"))


def test_versioning_writer_lease_dry_run_and_daily_skip(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    planned = backup_to_s3(
        storage, "s3://bucket/backups", aws_executable=executable, dry_run=True
    )
    assert planned["planned_change_count"] == planned["source_inventory"]["file_count"]
    assert not (remote / "bucket/backups/_throwdb_backup/LINEAGE.json").exists()
    backup_to_s3(storage, "s3://bucket/backups", aws_executable=executable)
    skipped = backup_to_s3(
        storage,
        "s3://bucket/backups",
        aws_executable=executable,
        if_older_than_hours=24,
    )
    assert skipped["status"] == "skipped"
    monkeypatch.setenv("FAKE_AWS_VERSIONING", "Suspended")
    with pytest.raises(S3BackupError, match="Versioning"):
        backup_to_s3(storage, "s3://bucket/other", aws_executable=executable)

    descriptor = os.open(storage / "locks/writer.lock", os.O_RDWR)
    monkeypatch.setenv("FAKE_AWS_VERSIONING", "Enabled")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(S3BackupError, match="writer lease"):
            backup_to_s3(storage, "s3://bucket/other", aws_executable=executable)
    finally:
        os.close(descriptor)


def test_literal_null_version_id_is_never_treated_as_immutable(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, remote, _ = fake_aws
    storage = _collection(tmp_path / "collection")
    # Models Versioning being suspended after the initial bucket preflight:
    # S3 reports the non-empty sentinel "null", not an immutable version.
    monkeypatch.setenv("FAKE_AWS_NULL_VERSION", "1")
    with pytest.raises(S3BackupError, match="immutable VersionId"):
        backup_to_s3(
            storage,
            "s3://bucket/backups",
            aws_executable=executable,
        )
    assert not (remote / "bucket/backups/_throwdb_backup/LATEST.json").exists()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("if_older_than_hours", float("nan")),
        ("metadata_timeout_seconds", float("inf")),
        ("max_seconds", float("-inf")),
    ],
)
def test_library_rejects_nonfinite_parameters_before_aws(
    tmp_path: Path,
    fake_aws: tuple[Path, Path, Path],
    argument: str,
    value: float,
) -> None:
    executable, _, log = fake_aws
    storage = _collection(tmp_path / "collection")
    with pytest.raises(ValueError, match="finite"):
        backup_to_s3(
            storage,
            "s3://bucket/backups",
            aws_executable=executable,
            **{argument: value},
        )
    assert not log.exists()


def test_cli_backup_and_restore_forward_hardened_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(
        ["backup", "s3", "--storage", "/tmp/store", "--destination", "s3://bucket/prefix"]
    )
    assert parsed.backup_command == "s3"
    observed: dict[str, object] = {}

    def fake_backup(storage: Path, destination: str, **kwargs: object) -> dict[str, object]:
        observed.update(storage=storage, destination=destination, **kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr("throwdb.cli.backup_to_s3", fake_backup)
    main(
        [
            "backup", "s3", "--storage", "/tmp/store", "--destination", "s3://bucket/prefix",
            "--dry-run", "--max-seconds", "90",
        ]
    )
    assert observed["max_seconds"] == 90.0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run"}

    restored: dict[str, object] = {}
    monkeypatch.setattr(
        "throwdb.cli.restore_from_s3",
        lambda source, storage, **kwargs: restored.update(
            source=source, storage=storage, **kwargs
        )
        or {"status": "restored"},
    )
    main(
        [
            "restore",
            "s3",
            "--source",
            "s3://bucket/prefix",
            "--storage",
            "/tmp/new",
            "--expected-backup-id",
            "a" * 32,
        ]
    )
    assert restored["expected_backup_id"] == "a" * 32
    assert json.loads(capsys.readouterr().out) == {"status": "restored"}
