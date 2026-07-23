from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
import sys
import time

import grpc
import numpy as np
import pytest

import throwdb
from throwdb.v1 import throwdb_pb2, throwdb_pb2_grpc
from throwdb.remote import (
    LocatedIds,
    RemoteAlreadyExistsError,
    RemoteCollection,
)

from ._harness import ServerProcess, ServerStartError


def _vectors(rows: int, dimension: int, *, salt: int = 0) -> np.ndarray:
    bits = (
        np.arange(rows * dimension, dtype=np.uint16) * np.uint16(7_919)
        + np.uint16(17 + salt)
    )
    return bits.reshape(rows, dimension).view(np.float16)


def _build_store(path: Path) -> np.ndarray:
    dimension = 7
    base = _vectors(3, dimension, salt=1)
    with throwdb.Collection.create(
        path,
        schema=throwdb.CollectionSchema(dimension=dimension, dtype="float16"),
        shard_size_bytes=base[:2].nbytes,
        format_version=7,
    ) as builder:
        builder.add(np.array([10, 20, 30], dtype=np.uint32), base)
    return base


def _assert_locations_are_in_physical_order(located: LocatedIds) -> None:
    pairs = list(
        zip(
            located.locations["segment_id"].tolist(),
            located.locations["row_index"].tolist(),
            strict=True,
        )
    )
    assert pairs == sorted(pairs)
    assert len(pairs) == len(set(pairs))


def _create_test_mtls_credentials(path: Path) -> dict[str, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for the mTLS server smoke test")
    path.mkdir()
    files = {
        name: path / name
        for name in (
            "ca.key",
            "ca.crt",
            "server.key",
            "server.csr",
            "server.crt",
            "server.ext",
            "client.key",
            "client.csr",
            "client.crt",
            "client.ext",
        )
    }

    def run(*arguments: str) -> None:
        subprocess.run(
            [openssl, *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    run(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "1",
        "-subj",
        "/CN=ThrowDB Test CA",
        "-keyout",
        str(files["ca.key"]),
        "-out",
        str(files["ca.crt"]),
    )
    run(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-subj",
        "/CN=localhost",
        "-keyout",
        str(files["server.key"]),
        "-out",
        str(files["server.csr"]),
    )
    files["server.ext"].write_text(
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=IP:127.0.0.1,DNS:localhost\n",
        encoding="ascii",
    )
    run(
        "x509",
        "-req",
        "-days",
        "1",
        "-in",
        str(files["server.csr"]),
        "-CA",
        str(files["ca.crt"]),
        "-CAkey",
        str(files["ca.key"]),
        "-CAcreateserial",
        "-extfile",
        str(files["server.ext"]),
        "-out",
        str(files["server.crt"]),
    )
    run(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-subj",
        "/CN=throwdb-test-client",
        "-keyout",
        str(files["client.key"]),
        "-out",
        str(files["client.csr"]),
    )
    files["client.ext"].write_text(
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature\n"
        "extendedKeyUsage=clientAuth\n",
        encoding="ascii",
    )
    run(
        "x509",
        "-req",
        "-days",
        "1",
        "-in",
        str(files["client.csr"]),
        "-CA",
        str(files["ca.crt"]),
        "-CAkey",
        str(files["ca.key"]),
        "-CAcreateserial",
        "-extfile",
        str(files["client.ext"]),
        "-out",
        str(files["client.crt"]),
    )
    return files


def test_server_crud_scan_and_restart(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    inserted = _vectors(2, 7, salt=101)
    updated = _vectors(1, 7, salt=201)
    upserted = _vectors(2, 7, salt=301)
    insert_operation = uuid.UUID("f58bb52c-c427-4ba7-bd5a-8d721f74b73c")

    first = server_process_factory(store, extra_args=("--writable",))
    assert first.maintenance["enabled"] is True
    with RemoteCollection.connect(first.endpoint) as remote:
        assert remote.health()["serving"] is True
        assert remote.count == 3
        remote.insert([40, 50], inserted, operation_id=insert_operation)
        remote.update([20], updated)
        remote.upsert([10, 60], upserted)
        np.testing.assert_array_equal(remote.delete([30, 999]), [True, False])
        assert remote.count == 5

        expected_ids = [40, 50, 20, 10, 60]
        located = remote.list_ids(with_locations=True, batch_rows=2)
        assert isinstance(located, LocatedIds)
        assert located.ids.tolist() == expected_ids
        _assert_locations_are_in_physical_order(located)
        assert remote.list_ids(batch_rows=3).tolist() == expected_ids

        expected = np.stack(
            [inserted[0], inserted[1], updated[0], upserted[0], upserted[1]]
        )
        np.testing.assert_array_equal(
            remote.get(expected_ids).view(np.uint16), expected.view(np.uint16)
        )

    first.close()
    assert first.process.returncode == 0, first.diagnostics

    second = server_process_factory(store, extra_args=("--writable",))
    with RemoteCollection.connect(second.endpoint) as reopened:
        assert reopened.count == 5
        # A response may be lost after the WAL commit. Reusing the caller's
        # operation ID after a process restart must replay the durable result,
        # not attempt a second Insert and report that the IDs already exist.
        reopened.insert([40, 50], inserted, operation_id=insert_operation)
        assert reopened.count == 5
        with pytest.raises(RemoteAlreadyExistsError, match="operation_id"):
            reopened.insert(
                [40, 50],
                _vectors(2, 7, salt=401),
                operation_id=insert_operation,
            )
        located = reopened.list_ids(with_locations=True, batch_rows=2)
        assert isinstance(located, LocatedIds)
        assert located.ids.tolist() == [40, 50, 20, 10, 60]
        _assert_locations_are_in_physical_order(located)


def test_native_server_streams_large_reads_from_one_snapshot(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    """Exercise the public Python client against the real C++ streaming RPCs."""

    store = tmp_path / "store"
    base = _build_store(store)
    server = server_process_factory(
        store,
        extra_args=("--writable", "--no-auto-maintenance"),
    )

    query_path = tmp_path / "query-ids.npy"
    np.save(query_path, np.array([30, 10, 30, 20], dtype=np.uint32))
    query = np.load(query_path, mmap_mode="r", allow_pickle=False)
    output = np.memmap(
        tmp_path / "stream-get-output.bin",
        mode="w+",
        dtype=np.float16,
        shape=(query.size, 3),
    )
    output[:] = np.float16(-1)

    with RemoteCollection.connect(server.endpoint, timeout=10) as remote:
        assert remote.supports_stream_get
        assert remote.supports_scan_records

        # Construction receives StreamGetOpened and therefore fixes the
        # native read snapshot before these later mutations are published.
        point_stream = remote.stream_get(
            query,
            prefix_dimensions=3,
            batch_rows=2,
            max_batch_payload_bytes=12,
        )
        replacement = _vectors(1, 7, salt=701)
        remote.update([30], replacement)
        np.testing.assert_array_equal(remote.delete([10]), [True])
        with point_stream:
            point_stream.get_into(output)
            assert point_stream.done
            assert point_stream.rows_completed == query.size
        np.testing.assert_array_equal(
            output.view(np.uint16),
            base[[2, 0, 2, 1], :3].view(np.uint16),
        )

        # ScanRecords prefetches its first bounded response at construction,
        # pinning the post-update/pre-insert view. A later insert must not
        # appear in any subsequent transport batch.
        record_scan = remote.scan_records(
            batch_rows=2,
            max_batch_payload_bytes=40,
            prefix_dimensions=2,
            with_locations=True,
        )
        inserted = _vectors(1, 7, salt=901)
        remote.insert([40], inserted)

        ids_buffer = np.empty(1, dtype=np.uint64)
        embedding_buffer = np.empty((1, 2), dtype=np.float16)
        location_buffer = np.empty(1, dtype=throwdb.ID_LOCATION_DTYPE)
        scanned_ids: list[int] = []
        scanned_embeddings: list[np.ndarray] = []
        scanned_locations: list[tuple[int, int]] = []
        with record_scan:
            while True:
                count = record_scan.read_into(
                    ids_buffer,
                    embedding_buffer,
                    location_buffer,
                )
                if count == 0:
                    break
                assert count == 1
                scanned_ids.append(int(ids_buffer[0]))
                scanned_embeddings.append(embedding_buffer[0].copy())
                scanned_locations.append(
                    (
                        int(location_buffer["segment_id"][0]),
                        int(location_buffer["row_index"][0]),
                    )
                )
            assert record_scan.done
            assert record_scan.rows_completed == 2

        assert scanned_ids == [20, 30]
        assert scanned_locations == sorted(scanned_locations)
        np.testing.assert_array_equal(
            np.stack(scanned_embeddings).view(np.uint16),
            np.stack([base[1, :2], replacement[0, :2]]).view(np.uint16),
        )
        assert set(remote.list_ids().tolist()) == {20, 30, 40}


def test_writable_server_runs_bounded_automatic_maintenance(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    server = server_process_factory(
        store,
        extra_args=(
            "--writable",
            "--maintenance-interval-seconds",
            "1",
            "--rotate-wal-transactions",
            "1",
            "--orphan-min-age-seconds",
            "0",
        ),
    )
    inserted = _vectors(1, 7, salt=701)
    with RemoteCollection.connect(server.endpoint) as remote:
        remote.insert([70], inserted)
        deadline = time.monotonic() + 8
        event: dict[str, object] | None = None
        while time.monotonic() < deadline:
            for line in list(server.stderr_lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(candidate, dict)
                    and candidate.get("event") == "maintenance"
                    and candidate.get("rotated") is True
                ):
                    event = candidate
                    break
            if event is not None:
                break
            time.sleep(0.05)
        assert event is not None, server.diagnostics
        assert event["status"] == "ok"
        assert event["materialized"] is True
        np.testing.assert_array_equal(
            remote.get([70]).view(np.uint16), inserted.view(np.uint16)
        )


def test_read_only_server_keeps_reads_available_and_rejects_mutations(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    base = _build_store(store)
    server = server_process_factory(store)

    assert server.writable is False
    assert server.readiness == {"status": "SERVING"}
    assert server.security == {
        "transport": "insecure",
        "client_certificate_required": False,
        "insecure_non_loopback_override": False,
    }
    assert server.mutation_safety["disk_reserve_bytes"] == 1 << 30
    assert (
        server.mutation_safety["idempotency_journal_max_bytes"] == 4 << 30
    )
    assert (
        server.mutation_safety["idempotency_journal_max_operations"]
        == 1_000_000
    )
    assert server.maintenance["enabled"] is False
    with RemoteCollection.connect(server.endpoint) as remote:
        assert remote.mode == "read"
        np.testing.assert_array_equal(
            remote.get([30, 10]).view(np.uint16),
            base[[2, 0]].view(np.uint16),
        )
        with pytest.raises(RuntimeError, match="read-only"):
            remote.insert([40], _vectors(1, 7, salt=41))


def test_non_loopback_bind_requires_mtls_or_explicit_override(
    tmp_path: Path,
    throwdb_server_binary: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    with pytest.raises(ServerStartError, match="non-loopback.*mTLS"):
        ServerProcess.start(
            throwdb_server_binary,
            store,
            timeout=3,
            extra_args=("--listen", "0.0.0.0:0"),
        )


def test_mtls_server_requires_and_verifies_client_certificate(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    certificates = _create_test_mtls_credentials(tmp_path / "certificates")
    server = server_process_factory(
        store,
        extra_args=(
            "--tls-cert",
            str(certificates["server.crt"]),
            "--tls-key",
            str(certificates["server.key"]),
            "--tls-client-ca",
            str(certificates["ca.crt"]),
        ),
    )
    assert server.security == {
        "transport": "mtls",
        "client_certificate_required": True,
        "insecure_non_loopback_override": False,
    }

    root = certificates["ca.crt"].read_bytes()
    credentials = grpc.ssl_channel_credentials(
        root_certificates=root,
        private_key=certificates["client.key"].read_bytes(),
        certificate_chain=certificates["client.crt"].read_bytes(),
    )
    channel = grpc.secure_channel(
        server.endpoint,
        credentials,
        options=(("grpc.enable_http_proxy", 0),),
    )
    try:
        response = throwdb_pb2_grpc.ThrowDBStub(channel).Health(
            throwdb_pb2.HealthRequest(), timeout=5
        )
        assert (
            response.status
            == throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING
        )
    finally:
        channel.close()

    unauthenticated = grpc.secure_channel(
        server.endpoint,
        grpc.ssl_channel_credentials(root_certificates=root),
        options=(("grpc.enable_http_proxy", 0),),
    )
    try:
        with pytest.raises(grpc.RpcError):
            throwdb_pb2_grpc.ThrowDBStub(unauthenticated).Health(
                throwdb_pb2.HealthRequest(), timeout=2
            )
    finally:
        unauthenticated.close()


def test_grpc_requests_and_responses_larger_than_four_mib(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    # gRPC defaults to 4 MiB. This shape is deliberately larger in both the
    # Update request and the Get response, while staying small enough for CI.
    row_count = 2_100
    dimension = 1_024
    store = tmp_path / "large-store"
    ids = np.arange(1, row_count + 1, dtype=np.uint64)
    original = _vectors(row_count, dimension, salt=11)
    replacement = _vectors(row_count, dimension, salt=37)
    assert original.nbytes > 4 << 20

    with throwdb.Collection.create(
        store,
        schema=throwdb.CollectionSchema(
            dimension=dimension,
            dtype="float16",
        ),
        shard_size_bytes=2 << 20,
        format_version=7,
    ) as builder:
        builder.add(ids, original)

    server = server_process_factory(store, extra_args=("--writable",))
    with RemoteCollection.connect(server.endpoint, timeout=30) as remote:
        received = remote.get(ids)
        np.testing.assert_array_equal(
            received.view(np.uint16), original.view(np.uint16)
        )
        remote.update(
            ids,
            replacement,
            operation_id=uuid.UUID("76979b2a-f88c-466e-9aca-39f94500d664"),
        )
        np.testing.assert_array_equal(
            remote.get([1, row_count]).view(np.uint16),
            replacement[[0, -1]].view(np.uint16),
        )


def test_second_writable_server_is_fenced(
    tmp_path: Path,
    throwdb_server_binary: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    first = server_process_factory(store, extra_args=("--writable",))

    with pytest.raises(ServerStartError, match="(?i)writer|lock|lease|already"):
        ServerProcess.start(
            throwdb_server_binary,
            store,
            timeout=3,
            extra_args=("--writable",),
        )

    assert first.process.poll() is None, first.diagnostics
    with RemoteCollection.connect(first.endpoint, timeout=3) as remote:
        assert remote.health()["serving"] is True


def test_tune_aware_launcher_auto_fallback_and_manual_override(
    tmp_path: Path,
    throwdb_server_binary: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    launcher = (
        "-m",
        "throwdb.server",
        "--native-binary",
        str(throwdb_server_binary),
    )

    with ServerProcess.start(
        Path(sys.executable),
        store,
        command_prefix=launcher,
    ) as fallback:
        assert fallback.tuning == {
            "mode": "auto",
            "source": "fallback",
            "status": "registry_missing",
            "reuse_level": "fallback",
            "explicit_overrides": False,
        }
        with RemoteCollection.connect(fallback.endpoint) as remote:
            description = remote.describe()
            assert description.tuning is not None
            assert description.tuning.status == "registry_missing"
            assert description.reader is not None
            assert description.reader.io_workers == 0

    with ServerProcess.start(
        Path(sys.executable),
        store,
        command_prefix=(*launcher, "--io-workers", "3"),
    ) as manual:
        assert manual.reader["io_workers"] == 3
        assert manual.tuning == {
            "mode": "manual",
            "source": "manual",
            "status": "explicit",
            "reuse_level": "manual",
            "explicit_overrides": True,
        }
        with RemoteCollection.connect(manual.endpoint) as remote:
            description = remote.describe()
            assert description.reader is not None
            assert description.reader.io_workers == 3
            assert description.tuning is not None
            assert description.tuning.mode == "manual"


def test_tune_aware_launcher_persisted_miss_fails_before_bind(
    tmp_path: Path,
    throwdb_server_binary: Path,
) -> None:
    store = tmp_path / "store"
    _build_store(store)

    with pytest.raises(ServerStartError, match="no reader profiles"):
        ServerProcess.start(
            Path(sys.executable),
            store,
            timeout=5,
            command_prefix=(
                "-m",
                "throwdb.server",
                "--native-binary",
                str(throwdb_server_binary),
                "--tuning",
                "persisted",
            ),
        )


def test_selected_tuning_diagnostics_round_trip(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    _build_store(store)
    server = server_process_factory(
        store,
        extra_args=(
            "--io-workers",
            "2",
            "--resolved-tuning-mode",
            "auto",
            "--resolved-tuning-source",
            "persisted",
            "--resolved-tuning-status",
            "selected",
            "--resolved-tuning-reuse-level",
            "exact",
            "--resolved-tuning-explicit-overrides",
            "0",
        ),
    )

    assert server.reader["io_workers"] == 2
    assert server.tuning["source"] == "persisted"
    with RemoteCollection.connect(server.endpoint) as remote:
        description = remote.describe()
        assert description.reader is not None
        assert description.reader.io_workers == 2
        assert description.tuning is not None
        assert description.tuning.source == "persisted"
        assert description.tuning.status == "selected"
        assert description.tuning.reuse_level == "exact"
