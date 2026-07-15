from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import throwdb
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

    first = server_process_factory(store)
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

    second = server_process_factory(store)
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


def test_read_only_server_keeps_reads_available_and_rejects_mutations(
    tmp_path: Path,
    server_process_factory: Callable[..., ServerProcess],
) -> None:
    store = tmp_path / "store"
    base = _build_store(store)
    server = server_process_factory(store, extra_args=("--read-only",))

    assert server.writable is False
    with RemoteCollection.connect(server.endpoint) as remote:
        assert remote.mode == "read"
        np.testing.assert_array_equal(
            remote.get([30, 10]).view(np.uint16),
            base[[2, 0]].view(np.uint16),
        )
        with pytest.raises(RuntimeError, match="read-only"):
            remote.insert([40], _vectors(1, 7, salt=41))


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

    server = server_process_factory(store)
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
    first = server_process_factory(store)

    with pytest.raises(ServerStartError, match="(?i)writer|lock|lease|already"):
        ServerProcess.start(
            throwdb_server_binary,
            store,
            timeout=3,
        )

    assert first.process.poll() is None, first.diagnostics
    with RemoteCollection.connect(first.endpoint, timeout=3) as remote:
        assert remote.health()["serving"] is True
