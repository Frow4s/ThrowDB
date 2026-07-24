from __future__ import annotations

import contextlib
import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import throwdb


def _load_soak() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_grpc_server_soak.py"
    spec = importlib.util.spec_from_file_location(
        "throwdb_grpc_soak_regressions",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOAK = _load_soak()


def _configuration(
    *,
    stream_get_workers: int,
    stream_get_rows: int = 2,
    progress_timeout_seconds: float = 0.15,
) -> Any:
    return SOAK.StreamingConfiguration(
        stream_get_workers=stream_get_workers,
        scan_records_workers=0,
        stream_get_rows=stream_get_rows,
        scan_records_rows=1,
        batch_rows=1,
        max_batch_payload_bytes=64,
        prefix=2,
        attempt_seconds=10.0,
        progress_timeout_seconds=progress_timeout_seconds,
        stop_timeout_seconds=1.0,
        rpc_timeout_seconds=1.0,
        mutation_rows=0,
        with_locations=False,
    )


def _resource_budget() -> Any:
    return SOAK.benchmark.ResourceBudget(
        max_seconds=10.0,
        max_bytes=1 << 20,
        max_client_memory_bytes=1 << 30,
    )


def _workload(
    *,
    config: Any,
    connect: Callable[[str], Any],
    budget: Any | None = None,
) -> tuple[Any, Any]:
    operations = SOAK.OperationMetrics(100, seed=101)
    errors = SOAK.ErrorMetrics()
    workload = SOAK.StreamingWorkload(
        config=config,
        population=np.asarray([10, 20], dtype=np.uint64),
        source_rows=None,
        source_embeddings=None,
        shadow_payload=np.asarray([[10, 11]], dtype=np.float16),
        dimension=2,
        dtype="float16",
        budget=budget or _resource_budget(),
        operations=operations,
        errors=errors,
    )
    workload._connect = connect
    return workload, errors


def _snapshot(token: str) -> Any:
    return throwdb.IdScanSnapshot(
        collection_id="fake-collection",
        catalog_generation=1,
        committed_sequence=1,
        count=2,
        token=token,
    )


def _batch(ids: Any, *, offset: int, snapshot: Any) -> Any:
    normalized = np.asarray(ids, dtype=np.uint64)
    embeddings = np.stack(
        [
            np.asarray([identifier, identifier + 1], dtype=np.float16)
            for identifier in normalized
        ]
    )
    return throwdb.RemoteEmbeddingBatch(
        sequence=offset,
        input_offset=offset,
        ids=normalized,
        embeddings=embeddings,
        snapshot=snapshot,
    )


class _Client:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.closed = False

    def stream_get(self, ids: Any, **_: Any) -> Any:
        self.stream.bind(ids)
        return self.stream

    def close(self) -> None:
        self.closed = True


class _ClosedOnEnterStream:
    """Expose the release/close race without relying on thread scheduling."""

    def __init__(self) -> None:
        self.snapshot = _snapshot("closed-on-enter")
        self.done = False
        self.closed = False
        self.entered = threading.Event()
        self._closed_event = threading.Event()

    def bind(self, ids: Any) -> None:
        self.ids = ids

    def __iter__(self) -> "_ClosedOnEnterStream":
        return self

    def __next__(self) -> Any:
        raise AssertionError("cancelled first attempt must not be consumed")

    def __enter__(self) -> "_ClosedOnEnterStream":
        self.entered.set()
        if not self._closed_event.wait(1.0):
            raise AssertionError("the cancellation path did not close the stream")
        raise RuntimeError("stream is closed")

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True
        self._closed_event.set()


class _BlockingEmbeddingStream:
    def __init__(
        self,
        *,
        token: str,
        emit_first_batch: bool,
        complete_after_first_batch: bool = False,
    ) -> None:
        self.snapshot = _snapshot(token)
        self.done = False
        self.closed = False
        self._emit_first_batch = emit_first_batch
        self._complete_after_first_batch = complete_after_first_batch
        self._emitted = False
        self._close_event = threading.Event()
        self.blocked = threading.Event()

    def bind(self, ids: Any) -> None:
        self.ids = ids

    def __iter__(self) -> "_BlockingEmbeddingStream":
        return self

    def __next__(self) -> Any:
        if self._emit_first_batch and not self._emitted:
            self._emitted = True
            return _batch(
                [self.ids[0]],
                offset=0,
                snapshot=self.snapshot,
            )
        if self._complete_after_first_batch:
            self.done = True
            raise StopIteration
        self.blocked.set()
        if not self._close_event.wait(2.0):
            raise AssertionError("blocked peer was not closed")
        raise throwdb.RemoteCancelledError("CANCELLED", "fake stream closed")

    def __enter__(self) -> "_BlockingEmbeddingStream":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True
        self._close_event.set()


class _SequentialConnector:
    def __init__(self, clients: list[_Client]) -> None:
        self._clients = list(clients)
        self._lock = threading.Lock()

    def __call__(self, endpoint: str) -> _Client:
        assert endpoint == "fake:1"
        with self._lock:
            if not self._clients:
                raise AssertionError("unexpected extra streaming client")
            return self._clients.pop(0)


class _FailAfterInitialReservations:
    """Allow every worker to open once, then exhaust the transfer budget."""

    def __init__(self, initial_reservations: int) -> None:
        self._initial_reservations = initial_reservations
        self._calls = 0
        self._lock = threading.Lock()

    def remaining_seconds(self) -> float:
        return 10.0

    def reserve(self, byte_count: int, *, extra_memory_bytes: int = 0) -> None:
        assert byte_count > 0
        assert extra_memory_bytes >= 0
        with self._lock:
            self._calls += 1
            if self._calls > self._initial_reservations:
                raise SOAK.benchmark.BenchmarkBudgetExceeded(
                    "injected transfer budget exhaustion"
                )


def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.005)


def test_scan_admission_uses_admitted_delta_after_fast_stream_completion() -> None:
    baseline_admitted = 40
    workers = 2
    statistics = throwdb.RemoteAdmissionStatistics(
        active_unary_calls=0,
        active_scans=1,
        unary_estimated_bytes=0,
        scan_estimated_bytes=64,
        admitted_unary_calls=0,
        rejected_unary_calls=0,
        admitted_scans=baseline_admitted + workers,
        rejected_scans=0,
        admission_closed=False,
        draining=False,
        terminal_faulted=False,
    )

    class Remote:
        _timeout = 0.0

        def describe(self) -> Any:
            return SimpleNamespace(admission_stats=statistics)

    result = SOAK.wait_for_scan_admission(
        Remote(),
        minimum_admitted_scans=baseline_admitted + workers,
        timeout_seconds=0.1,
    )

    assert result["status"] == "passed"
    assert result["expected_active_scans"] is None
    assert result["minimum_admitted_scans"] == baseline_admitted + workers


def test_cancel_before_release_never_enters_an_already_closed_stream() -> None:
    stream = _ClosedOnEnterStream()
    client = _Client(stream)
    workload, errors = _workload(
        config=_configuration(stream_get_workers=1, stream_get_rows=1),
        connect=lambda endpoint: client,
    )

    workload.start_generation("fake:1", 1)
    event = workload.stop_generation("cancel_before_release")

    assert event["joined"] is True
    assert stream.closed is True
    assert stream.entered.is_set() is False
    assert errors.total == 0
    assert (
        workload.summary()["modes"]["stream_get"]["attempts_cancelled"]
        == 1
    )


def test_release_requires_a_first_batch_from_every_worker() -> None:
    progressed = _BlockingEmbeddingStream(
        token="progressed",
        emit_first_batch=True,
    )
    blocked = _BlockingEmbeddingStream(
        token="blocked",
        emit_first_batch=False,
    )
    connector = _SequentialConnector([_Client(progressed), _Client(blocked)])
    workload, _ = _workload(
        config=_configuration(
            stream_get_workers=2,
            progress_timeout_seconds=0.1,
        ),
        connect=connector,
    )

    workload.start_generation("fake:1", 2)
    try:
        with pytest.raises(SOAK.SoakInvariantError, match="progress|stalled"):
            workload.release_and_wait_for_progress()
    finally:
        with contextlib.suppress(BaseException):
            workload.stop_generation("unit_test_cleanup")

    assert progressed.blocked.is_set()
    assert blocked.blocked.is_set()


def test_completed_stream_get_digest_must_remain_stable() -> None:
    metrics = SOAK.StreamingMetrics(
        _configuration(stream_get_workers=1),
        SOAK.OperationMetrics(10, seed=7),
    )

    metrics.attempt_finished(
        "stream_get",
        status="completed",
        digest="00" * 32,
    )
    with pytest.raises(
        SOAK.SoakInvariantError,
        match="different bytes",
    ):
        metrics.attempt_finished(
            "stream_get",
            status="completed",
            digest="ff" * 32,
        )


def test_budget_exhaustion_closes_a_blocked_peer() -> None:
    fast = _BlockingEmbeddingStream(
        token="fast",
        emit_first_batch=True,
        complete_after_first_batch=True,
    )
    blocked = _BlockingEmbeddingStream(
        token="budget-peer",
        emit_first_batch=False,
    )
    connector = _SequentialConnector([_Client(fast), _Client(blocked)])
    workload, errors = _workload(
        config=_configuration(
            stream_get_workers=2,
            stream_get_rows=1,
            progress_timeout_seconds=1.0,
        ),
        connect=connector,
        budget=_FailAfterInitialReservations(2),
    )

    workload.start_generation("fake:1", 3)
    assert workload._current is not None
    workload._current.release_event.set()
    _wait_until(
        lambda: (
            workload.summary()["modes"]["stream_get"]["budget_stops"] == 1
        )
    )
    try:
        assert blocked._close_event.wait(0.2), (
            "a worker exhausting the shared budget must cancel and close "
            "its blocked peer immediately"
        )
    finally:
        event = workload.stop_generation("unit_test_cleanup")

    assert event["joined"] is True
    assert errors.total == 0
