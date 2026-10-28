from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import throwdb


def _load_soak() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_grpc_server_soak.py"
    spec = importlib.util.spec_from_file_location("throwdb_grpc_soak", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOAK = _load_soak()


class FakeRemote:
    def __init__(
        self,
        *,
        values: dict[int, np.ndarray] | None = None,
        journal: dict[uuid.UUID, tuple[str, bytes, object]] | None = None,
        domain: uuid.UUID | None = None,
    ) -> None:
        self.dimension = 4
        self.dtype = "float16"
        self._values = values if values is not None else {}
        self._journal = journal if journal is not None else {}
        self._domain = domain or uuid.UUID("11111111-2222-3333-4444-555555555555")
        self.calls: dict[str, int] = {}
        self._timeout = 30.0
        self.closed = False

    def describe(self) -> SimpleNamespace:
        return SimpleNamespace(
            format_version=7,
            schema=SimpleNamespace(dimension=4, dtype="float16"),
            count=len(self._values),
            writable=True,
            durable_idempotency=True,
            idempotency_domain=self._domain,
            reader=None,
            tuning=None,
            availability=None,
            admission_stats=None,
            engine_stats=None,
        )

    def health(self) -> dict[str, int | bool]:
        return {"serving": True, "protocol_major": 1, "protocol_minor": 0}

    def get(self, ids: np.ndarray, **_: object) -> np.ndarray:
        rows: list[np.ndarray] = []
        for identifier in np.asarray(ids, dtype=np.uint64):
            value = self._values.get(int(identifier))
            if value is None:
                raise throwdb.RemoteNotFoundError("NOT_FOUND", "missing")
            rows.append(value)
        return np.asarray(rows, dtype=np.float16)

    def _put(
        self,
        name: str,
        ids: np.ndarray,
        embeddings: np.ndarray,
        operation_id: uuid.UUID,
    ) -> None:
        payload = np.asarray(embeddings).view(np.uint8).tobytes()
        cached = self._journal.get(operation_id)
        if cached is not None:
            if cached[:2] != (name, payload):
                raise AssertionError("conflicting fake replay")
            self.calls[name] = self.calls.get(name, 0) + 1
            return
        normalized = [int(value) for value in np.asarray(ids, dtype=np.uint64)]
        if name == "insert" and any(value in self._values for value in normalized):
            raise throwdb.RemoteAlreadyExistsError("ALREADY_EXISTS", "exists")
        if name == "update" and any(value not in self._values for value in normalized):
            raise throwdb.RemoteNotFoundError("NOT_FOUND", "missing")
        for index, identifier in enumerate(normalized):
            self._values[identifier] = np.array(embeddings[index], copy=True)
        self._journal[operation_id] = (name, payload, None)
        self.calls[name] = self.calls.get(name, 0) + 1

    def insert(
        self, ids: np.ndarray, embeddings: np.ndarray, *, operation_id: uuid.UUID
    ) -> None:
        self._put("insert", ids, embeddings, operation_id)

    def update(
        self, ids: np.ndarray, embeddings: np.ndarray, *, operation_id: uuid.UUID
    ) -> None:
        self._put("update", ids, embeddings, operation_id)

    def upsert(
        self, ids: np.ndarray, embeddings: np.ndarray, *, operation_id: uuid.UUID
    ) -> None:
        self._put("upsert", ids, embeddings, operation_id)

    def delete(
        self, ids: np.ndarray, *, operation_id: uuid.UUID
    ) -> np.ndarray:
        cached = self._journal.get(operation_id)
        if cached is not None:
            assert cached[0] == "delete"
            self.calls["delete"] = self.calls.get("delete", 0) + 1
            return np.array(cached[2], copy=True)
        mask = np.asarray(
            [int(value) in self._values for value in ids], dtype=np.bool_
        )
        for identifier in np.asarray(ids, dtype=np.uint64):
            self._values.pop(int(identifier), None)
        self._journal[operation_id] = ("delete", b"", np.array(mask, copy=True))
        self.calls["delete"] = self.calls.get("delete", 0) + 1
        return mask

    def close(self) -> None:
        self.closed = True


def _budget() -> object:
    return SOAK.benchmark.ResourceBudget(
        max_seconds=100.0,
        max_bytes=1 << 30,
        max_client_memory_bytes=1 << 30,
    )


def test_payload_and_operation_ids_are_deterministic() -> None:
    ids = np.asarray([9, 11], dtype=np.uint64)
    first = SOAK.payload_for(
        ids, version=3, dimension=4, dtype="float16", seed=17
    )
    second = SOAK.payload_for(
        ids, version=3, dimension=4, dtype="float16", seed=17
    )
    changed = SOAK.payload_for(
        ids, version=4, dimension=4, dtype="float16", seed=17
    )

    assert np.array_equal(first.view(np.uint8), second.view(np.uint8))
    assert not np.array_equal(first.view(np.uint8), changed.view(np.uint8))
    assert SOAK.deterministic_operation_id(17, 2, "insert") == (
        SOAK.deterministic_operation_id(17, 2, "insert")
    )
    assert SOAK.deterministic_operation_id(17, 2, "insert") != (
        SOAK.deterministic_operation_id(17, 2, "delete")
    )
    assert SOAK.deterministic_operation_id(
        17,
        2,
        "insert",
        run_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    ) != SOAK.deterministic_operation_id(
        17,
        2,
        "insert",
        run_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    )


def test_bounded_latency_sample_is_deterministic_and_not_early_biased() -> None:
    metrics = SOAK.OperationMetrics(10, seed=91)
    repeated = SOAK.OperationMetrics(10, seed=91)
    for ordinal in range(100):
        latency = (ordinal + 1) * 1_000_000
        metrics.record(
            "get", ordinal, latency,
            rows=1, useful_bytes=8, success=True,
        )
        repeated.record(
            "get", ordinal, latency,
            rows=1, useful_bytes=8, success=True,
        )

    summary = metrics.summary(1.0)["get"]
    repeated_summary = repeated.summary(1.0)["get"]

    assert summary["attempts"] == 100
    assert summary["latency_sample_count"] == 10
    assert summary["latency_sample_ordinal_max"] >= 50
    assert summary == repeated_summary


def test_mutation_cycle_replays_every_uuid_and_restores_absent_state() -> None:
    remote = FakeRemote()
    canaries = np.asarray([UINT64 := np.iinfo(np.uint64).max], dtype=np.uint64)
    metrics = SOAK.OperationMetrics(100)
    errors = SOAK.ErrorMetrics()
    mutation_budget = SOAK.MutationBudget(5)

    result = SOAK.run_mutation_cycle(
        remote,
        canaries,
        sequence=1,
        seed=123,
        budget=_budget(),
        mutation_budget=mutation_budget,
        metrics=metrics,
        errors=errors,
        rpc_timeout=5.0,
        primary_attempts=2,
    )

    assert UINT64 not in remote._values
    assert result["status"] == "passed"
    assert result["logical_absent_state_restored"] is True
    assert {item["name"] for item in result["operations"]} == {
        "insert", "update", "upsert", "delete"
    }
    assert remote.calls == {"insert": 2, "update": 2, "upsert": 2, "delete": 2}
    assert mutation_budget.used == 4
    assert errors.total == 0


def test_missing_check_proves_every_canary_is_absent() -> None:
    present = np.asarray([1, 2, 3, 4], dtype=np.float16)
    remote = FakeRemote(values={22: present})

    with pytest.raises(SOAK.SoakInvariantError, match="22"):
        SOAK._expect_missing(
            remote,
            np.asarray([11, 22], dtype=np.uint64),
        )

    with pytest.raises(TimeoutError, match="deadline"):
        SOAK._expect_missing(
            FakeRemote(),
            np.asarray([11], dtype=np.uint64),
            deadline_monotonic=0.0,
        )


def test_mutation_transfer_reservation_covers_every_primary_attempt() -> None:
    rows = 2
    record_size = 8
    full = rows * (8 + record_size)
    get = rows * (8 + record_size)
    delete = rows * 9

    assert SOAK.mutation_cycle_reserved_bytes(rows, record_size, 3) == (
        3 * 4 * full + 4 * delete + 5 * get + delete
    )


def test_ambiguous_retry_refuses_changed_idempotency_domain() -> None:
    remote = FakeRemote()
    original_domain = remote._domain
    operation_id = uuid.UUID("12345678-1234-1234-1234-123456789abc")
    calls = 0

    def ambiguous_call() -> None:
        nonlocal calls
        calls += 1
        remote._domain = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        raise TimeoutError("ambiguous outcome")

    with pytest.raises(SOAK.SoakInvariantError, match="replay domain changed"):
        SOAK._invoke_mutation_with_replay(
            remote=remote,
            name="insert",
            operation_id=operation_id,
            idempotency_domain=original_domain,
            call=ambiguous_call,
            rows=1,
            useful_bytes=16,
            ordinal_base=1,
            metrics=SOAK.OperationMetrics(10),
            errors=SOAK.ErrorMetrics(),
            budget=_budget(),
            rpc_timeout=5.0,
            primary_attempts=2,
        )

    assert calls == 1


def test_aborted_mutation_retries_same_uuid_in_refreshed_domain() -> None:
    remote = FakeRemote()
    operation_id = uuid.UUID("32345678-1234-1234-1234-123456789abc")
    original_domain = remote._domain
    refreshed_domain = uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
    calls = 0
    errors = SOAK.ErrorMetrics()

    def aborted_once() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            remote._domain = refreshed_domain
            raise throwdb.RemoteAbortedError(
                operation_id,
                "upsert",
                details="maintenance writer handoff",
            )
        return "committed"

    primary, replay, attempts = SOAK._invoke_mutation_with_replay(
        remote=remote,
        name="upsert",
        operation_id=operation_id,
        idempotency_domain=original_domain,
        call=aborted_once,
        rows=1,
        useful_bytes=16,
        ordinal_base=1,
        metrics=SOAK.OperationMetrics(10),
        errors=errors,
        budget=_budget(),
        rpc_timeout=5.0,
        primary_attempts=3,
    )

    assert primary == "committed"
    assert replay == "committed"
    assert attempts == 3
    assert calls == 3
    assert remote._domain == refreshed_domain
    assert errors.total == 0


def test_definitive_mutation_rejection_is_not_retried() -> None:
    remote = FakeRemote()
    operation_id = uuid.UUID("42345678-1234-1234-1234-123456789abc")
    calls = 0
    errors = SOAK.ErrorMetrics()

    def rejected() -> None:
        nonlocal calls
        calls += 1
        raise throwdb.RemoteNotFoundError("NOT_FOUND", "definitive rejection")

    with pytest.raises(throwdb.RemoteNotFoundError):
        SOAK._invoke_mutation_with_replay(
            remote=remote,
            name="update",
            operation_id=operation_id,
            idempotency_domain=remote._domain,
            call=rejected,
            rows=1,
            useful_bytes=16,
            ordinal_base=1,
            metrics=SOAK.OperationMetrics(10),
            errors=errors,
            budget=_budget(),
            rpc_timeout=5.0,
            primary_attempts=3,
        )

    assert calls == 1
    assert errors.total == 1


def test_get_source_oracle_makes_wrong_base_row_bits_immediately_fatal() -> None:
    first = np.asarray([1, 2, 3, 4], dtype=np.float16)
    second = np.asarray([5, 6, 7, 8], dtype=np.float16)
    remote = FakeRemote(values={1: first, 2: second})
    expected = np.stack((first, second))
    expected[1, 0] = np.float16(99)
    plan = SOAK.GetPlan(
        ordinal=0,
        label="get_b2_p4",
        ids=np.asarray([1, 2], dtype=np.uint64),
        prefix=4,
        wire_bytes=48,
        shadow_expected=first.reshape(1, 4),
        source_expected=expected,
    )
    errors = SOAK.ErrorMetrics()

    with pytest.raises(SOAK.SoakInvariantError, match="source oracle"):
        SOAK.execute_get(
            remote,
            plan,
            dimension=4,
            itemsize=2,
            retries=1,
            budget=_budget(),
            rpc_timeout=5.0,
            metrics=SOAK.OperationMetrics(10),
            errors=errors,
        )
    # Integrity failures are recorded once but are never retried/tolerated by
    # --max-errors.
    assert errors.total == 1


def test_get_retries_only_explicitly_transient_remote_failures() -> None:
    expected = np.asarray([1, 2, 3, 4], dtype=np.float16)

    class FlakyRemote(FakeRemote):
        def get(self, ids: np.ndarray, **kwargs: object) -> np.ndarray:
            if self.calls.get("get", 0) == 0:
                self.calls["get"] = 1
                raise throwdb.RemoteUnavailableError(
                    "UNAVAILABLE", "transient test failure"
                )
            return super().get(ids, **kwargs)

    plan = SOAK.GetPlan(
        ordinal=0,
        label="get_b1_p4",
        ids=np.asarray([1], dtype=np.uint64),
        prefix=4,
        wire_bytes=24,
        shadow_expected=expected.reshape(1, 4),
        source_expected=expected.reshape(1, 4),
    )
    errors = SOAK.ErrorMetrics()
    assert SOAK.execute_get(
        FlakyRemote(values={1: expected}),
        plan,
        dimension=4,
        itemsize=2,
        retries=1,
        budget=_budget(),
        rpc_timeout=5.0,
        metrics=SOAK.OperationMetrics(10),
        errors=errors,
    )
    assert errors.total == 1

    with pytest.raises(throwdb.RemoteNotFoundError):
        SOAK.execute_get(
            FakeRemote(),
            plan,
            dimension=4,
            itemsize=2,
            retries=1,
            budget=_budget(),
            rpc_timeout=5.0,
            metrics=SOAK.OperationMetrics(10),
            errors=SOAK.ErrorMetrics(),
        )


def test_periodic_mutations_preserve_documented_restart_headroom() -> None:
    budget = SOAK.MutationBudget(40)
    budget.reserve(4)  # mandatory initial CRUD cycle

    for _ in range(12):  # mutation opportunities before restart at Get 250
        if SOAK.can_schedule_periodic_mutation(budget, remaining_restarts=2):
            budget.reserve(4)
    assert budget.can_reserve(3)  # restart UUIDs plus emergency headroom
    budget.reserve(2)

    for _ in range(12):  # opportunities before the second restart
        if SOAK.can_schedule_periodic_mutation(budget, remaining_restarts=1):
            budget.reserve(4)
    assert budget.can_reserve(3)
    budget.reserve(2)

    assert budget.used <= budget.maximum
    assert budget.can_reserve(1)


def test_population_scan_bounds_eager_first_batch_and_accounting() -> None:
    class Scan:
        def __enter__(self) -> "Scan":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self):
            yield np.asarray([77], dtype=np.uint64)

    class Remote:
        _scan_timeout = 0.0
        requested_batch_rows = 0

        def scan_ids(self, *, batch_rows: int) -> Scan:
            self.requested_batch_rows = batch_rows
            return Scan()

    remote = Remote()
    budget = _budget()

    sample = SOAK.discover_population(
        remote,
        requested=1,
        batch_rows=65_536,
        budget=budget,
    )

    assert remote.requested_batch_rows == 1
    assert budget.accounted_bytes == 8
    np.testing.assert_array_equal(sample.ids, np.asarray([77], dtype=np.uint64))


def test_population_sampler_rejects_memory_bound_before_large_allocation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ids.npy"
    np.save(path, np.arange(10_000, dtype=np.uint64))
    budget = SOAK.benchmark.ResourceBudget(
        max_seconds=10.0,
        max_bytes=1 << 30,
        max_client_memory_bytes=1024,
    )

    with pytest.raises(SOAK.benchmark.BenchmarkBudgetExceeded, match="memory"):
        SOAK.load_population(path, 10_000, 7, budget)


def test_failed_cycle_counts_single_emergency_cleanup_uuid() -> None:
    class FailingUpdateRemote(FakeRemote):
        def update(
            self,
            ids: np.ndarray,
            embeddings: np.ndarray,
            *,
            operation_id: uuid.UUID,
        ) -> None:
            raise TimeoutError("forced ambiguous update")

    remote = FailingUpdateRemote()
    canary = np.asarray([999], dtype=np.uint64)
    mutation_budget = SOAK.MutationBudget(5)

    with pytest.raises(TimeoutError, match="forced ambiguous"):
        SOAK.run_mutation_cycle(
            remote,
            canary,
            sequence=1,
            seed=3,
            budget=_budget(),
            mutation_budget=mutation_budget,
            metrics=SOAK.OperationMetrics(10),
            errors=SOAK.ErrorMetrics(),
            rpc_timeout=5.0,
            primary_attempts=1,
        )

    assert mutation_budget.used == 5
    assert 999 not in remote._values


def test_restart_probe_replays_acknowledged_insert_after_new_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[int, np.ndarray] = {}
    journal: dict[uuid.UUID, tuple[str, bytes, object]] = {}
    domain = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    first = FakeRemote(values=values, journal=journal, domain=domain)
    replacement = FakeRemote(values=values, journal=journal, domain=domain)
    shadow_id = np.asarray([7], dtype=np.uint64)
    shadow_payload = np.asarray([[1, 2, 3, 4]], dtype=np.float16)
    values[7] = shadow_payload[0].copy()

    class Supervisor:
        endpoint = "fake:1"

        def restart(self, timeout: float, mode: str) -> float:
            assert timeout > 0
            assert mode == "sigkill"
            return 0.25

    monkeypatch.setattr(SOAK, "_connect", lambda *args, **kwargs: replacement)
    canaries = np.asarray([123456789], dtype=np.uint64)

    remote, report = SOAK.restart_replay_probe(
        remote=first,
        supervisor=Supervisor(),
        canaries=canaries,
        shadow_id=shadow_id,
        shadow_payload=shadow_payload,
        sequence=7,
        seed=99,
        budget=_budget(),
        mutation_budget=SOAK.MutationBudget(3),
        metrics=SOAK.OperationMetrics(100),
        errors=SOAK.ErrorMetrics(),
        rpc_timeout=5.0,
        start_timeout=10.0,
        restart_mode="sigkill",
    )

    assert remote is replacement
    assert report["acknowledged_insert_replayed_after_restart"] is True
    assert report["logical_absent_state_restored"] is True
    assert set(values) == {7}
    np.testing.assert_array_equal(values[7], shadow_payload[0])
    assert first.closed is True


def test_parser_defaults_are_bounded_and_validation_is_fail_closed() -> None:
    parser = SOAK.build_parser()
    args = parser.parse_args(
        ["--store", "/store", "--results", "/results"]
    )
    SOAK.validate_args(args)

    assert args.max_seconds == 3600.0
    assert args.max_transfer_gib == 64.0
    assert args.max_mutations == 400
    assert args.max_get_calls == 10_000
    assert args.restart_cycles == 2
    assert args.restart_mode == "alternating"
    assert args.get_batch_sizes == (1, 128, 1024)
    assert args.stream_get_workers == 1
    assert args.scan_records_workers == 1
    assert args.stream_get_rows == 131_072
    assert args.scan_records_rows == 131_072
    assert args.stream_source_oracle is False

    external = parser.parse_args(
        ["--endpoint", "host:1", "--results", "/results"]
    )
    with pytest.raises(SOAK.SoakConfigurationError, match="restart"):
        SOAK.validate_args(external)

    unsafe = parser.parse_args(
        [
            "--endpoint", "host:1", "--results", "/results",
            "--restart-cycles", "0",
        ]
    )
    with pytest.raises(SOAK.SoakConfigurationError, match="external mutation"):
        SOAK.validate_args(unsafe)

    local_with_external_pid = parser.parse_args(
        [
            "--store", "/store", "--results", "/results",
            "--server-pid", "123",
        ]
    )
    with pytest.raises(SOAK.SoakConfigurationError, match="external-endpoint"):
        SOAK.validate_args(local_with_external_pid)

    observed_external = parser.parse_args(
        [
            "--endpoint", "host:1", "--results", "/results",
            "--server-pid", "123", "--restart-cycles", "0",
            "--allow-external-mutations",
        ]
    )
    SOAK.validate_args(observed_external)

    colliding_retry_ordinals = parser.parse_args(
        [
            "--store", "/store", "--results", "/results",
            "--get-retries", "100",
        ]
    )
    with pytest.raises(SOAK.SoakConfigurationError, match="at most 99"):
        SOAK.validate_args(colliding_retry_ordinals)

    source_without_ids = parser.parse_args(
        [
            "--store", "/store", "--results", "/results",
            "--source-embeddings", "/embeddings.npy",
        ]
    )
    with pytest.raises(SOAK.SoakConfigurationError, match="requires --ids"):
        SOAK.validate_args(source_without_ids)

    stream_oracle_without_source = parser.parse_args(
        [
            "--store", "/store", "--results", "/results",
            "--stream-source-oracle",
        ]
    )
    with pytest.raises(
        SOAK.SoakConfigurationError,
        match="stream-source-oracle",
    ):
        SOAK.validate_args(stream_oracle_without_source)

    undersized_latency_reservoir = parser.parse_args(
        [
            "--store", "/store", "--results", "/results",
            "--max-latency-samples", "1",
            "--min-mixed-small-get-successes", "2",
        ]
    )
    with pytest.raises(
        SOAK.SoakConfigurationError,
        match="max-latency-samples",
    ):
        SOAK.validate_args(undersized_latency_reservoir)

    ignored_external_server_option = parser.parse_args(
        [
            "--endpoint", "host:1", "--results", "/results",
            "--restart-cycles", "0", "--allow-external-mutations",
            "--server-max-concurrent-calls", "64",
        ]
    )
    with pytest.raises(
        SOAK.SoakConfigurationError,
        match="local-launch-only",
    ):
        SOAK.validate_args(ignored_external_server_option)


def test_operation_metrics_exposes_success_only_latency_distribution() -> None:
    metrics = SOAK.OperationMetrics(100, seed=17)
    metrics.record(
        "get",
        0,
        2_000_000,
        rows=1,
        useful_bytes=10,
        success=True,
    )
    metrics.record(
        "get",
        1,
        2_000_000_000,
        rows=1,
        useful_bytes=10,
        success=False,
    )

    summary = metrics.summary(1.0)["get"]

    assert summary["latency_sample_count"] == 2
    assert summary["successful_latency_sample_count"] == 1
    assert summary["successful_latency_p99_ms"] == 2.0
    assert summary["latency_p99_ms"] > 1000.0


def test_server_start_failure_leaves_atomic_failed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(self: object, timeout: float) -> None:
        del self, timeout
        raise RuntimeError("injected server start failure")

    monkeypatch.setattr(SOAK.LocalSupervisor, "start", fail_start)
    results = tmp_path / "results"
    args = SOAK.build_parser().parse_args(
        [
            "--store", str(tmp_path / "store"),
            "--results", str(results),
            "--restart-cycles", "0",
        ]
    )

    with pytest.raises(RuntimeError, match="injected server start failure"):
        SOAK.run(args)

    report = json.loads(
        (results / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["error"]["phase"] == "server_start"
    assert report["error"]["type"] == "RuntimeError"


def test_population_sampling_is_deterministic_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "ids.npy"
    np.save(path, np.arange(10_000, dtype=np.uint32))

    first = SOAK.load_population(path, 100, 77)
    second = SOAK.load_population(path, 100, 77)

    assert first.ids.dtype == np.uint64
    assert first.ids.size == 100
    assert np.unique(first.ids).size == first.ids.size
    assert first.source_rows is not None
    assert first.source_rows.size == first.ids.size
    np.testing.assert_array_equal(first.ids, second.ids)
    np.testing.assert_array_equal(first.source_rows, second.source_rows)


def test_repeated_ids_is_lazy_and_preserves_cyclic_ordering() -> None:
    population = np.asarray([9, 4, 7], dtype=np.uint64)
    repeated = SOAK.RepeatedIds(population, 10_000_000)

    assert len(repeated) == 10_000_000
    assert repeated._values.size == population.size
    assert [repeated[index] for index in range(8)] == [9, 4, 7, 9, 4, 7, 9, 4]
    assert repeated[-1] == 9
    assert repeated[1:8:2] == [4, 9, 7, 4]
    assert repeated[7:1:-2] == [4, 7, 9]
    with pytest.raises(IndexError):
        _ = repeated[len(repeated)]


@pytest.mark.parametrize(
    ("mixed_p99", "mixed_samples", "mixed_successes", "expected"),
    [
        (6.0, 20, 20, (True, True, None)),
        (
            9.0,
            20,
            20,
            (
                True,
                False,
                "mixed short-Get P99 exceeded the configured ratio",
            ),
        ),
        (
            6.0,
            3,
            3,
            (False, False, "insufficient successful latency samples"),
        ),
    ],
)
def test_fairness_acceptance_reports_pass_failure_and_evaluability(
    mixed_p99: float,
    mixed_samples: int,
    mixed_successes: int,
    expected: tuple[bool, bool, str | None],
) -> None:
    result = SOAK.fairness_acceptance(
        {
            "baseline": {
                "latency_sample_count": 10,
                "latency_p99_ms": 2.0,
                "successes": 10,
            },
            "mixed": {
                "latency_sample_count": mixed_samples,
                "latency_p99_ms": mixed_p99,
                "successes": mixed_successes,
            },
        },
        baseline_label="baseline",
        mixed_label="mixed",
        minimum_mixed_successes=10,
        maximum_p99_ratio=4.0,
    )

    assert (
        result["evaluable"],
        result["passed"],
        result["reason"],
    ) == expected
    if result["evaluable"]:
        assert result["observed_p99_ratio"] == mixed_p99 / 2.0
    else:
        assert result["observed_p99_ratio"] is None


def _streaming_config(
    *,
    stream_get_workers: int = 0,
    scan_records_workers: int = 0,
    stream_get_rows: int = 4,
    scan_records_rows: int = 3,
) -> object:
    return SOAK.StreamingConfiguration(
        stream_get_workers=stream_get_workers,
        scan_records_workers=scan_records_workers,
        stream_get_rows=stream_get_rows,
        scan_records_rows=scan_records_rows,
        batch_rows=2,
        max_batch_payload_bytes=64,
        prefix=2,
        attempt_seconds=10.0,
        progress_timeout_seconds=1.0,
        stop_timeout_seconds=1.0,
        rpc_timeout_seconds=1.0,
        mutation_rows=0,
        with_locations=True,
    )


def _streaming_budget(max_bytes: int = 1 << 20) -> object:
    return SOAK.benchmark.ResourceBudget(
        max_seconds=10.0,
        max_bytes=max_bytes,
        max_client_memory_bytes=1 << 30,
    )


def _wait_until(predicate: object, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():  # type: ignore[operator]
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.005)


class _FakeEmbeddingStream:
    def __init__(
        self,
        ids: object,
        *,
        values: dict[int, np.ndarray],
        prefix: int,
        batch_rows: int,
        block_after_first: bool = False,
        failure: BaseException | None = None,
    ) -> None:
        self._ids = ids
        self._values = values
        self._prefix = prefix
        self._batch_rows = batch_rows
        self._block_after_first = block_after_first
        self._failure = failure
        self._offset = 0
        self._close_event = threading.Event()
        self.second_next_started = threading.Event()
        self.snapshot = throwdb.IdScanSnapshot(
            collection_id="fake-collection",
            catalog_generation=1,
            committed_sequence=1,
            count=len(values),
            token="fake-stream-get-snapshot",
        )
        self.done = False
        self.closed = False
        self.next_calls = 0

    def __iter__(self) -> "_FakeEmbeddingStream":
        return self

    def __next__(self) -> object:
        self.next_calls += 1
        if self._failure is not None:
            failure = self._failure
            self._failure = None
            raise failure
        if self._block_after_first and self._offset:
            self.second_next_started.set()
            if not self._close_event.wait(2.0):
                raise AssertionError("fake StreamGet was not cancelled")
            raise throwdb.RemoteCancelledError("CANCELLED", "fake stream closed")
        if self._offset == len(self._ids):  # type: ignore[arg-type]
            self.done = True
            raise StopIteration
        begin = self._offset
        end = min(begin + self._batch_rows, len(self._ids))  # type: ignore[arg-type]
        ids = np.asarray(self._ids[begin:end], dtype=np.uint64)  # type: ignore[index]
        embeddings = np.stack(
            [self._values[int(identifier)][: self._prefix] for identifier in ids]
        ).astype(np.float16, copy=False)
        self._offset = end
        return throwdb.RemoteEmbeddingBatch(
            sequence=begin // self._batch_rows,
            input_offset=begin,
            ids=ids,
            embeddings=embeddings,
            snapshot=self.snapshot,
        )

    def close(self) -> None:
        self.closed = True
        self._close_event.set()

    def __enter__(self) -> "_FakeEmbeddingStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _FakeRecordStream:
    def __init__(self, *, count: int, prefix: int, batch_rows: int) -> None:
        self.snapshot = throwdb.IdScanSnapshot(
            collection_id="fake-collection",
            catalog_generation=1,
            committed_sequence=1,
            count=count,
            token="fake-record-snapshot",
        )
        self.snapshot_count = count
        self._prefix = prefix
        self._batch_rows = batch_rows
        self._offset = 0
        self.done = False
        self.closed = False
        self.next_calls = 0

    def __iter__(self) -> "_FakeRecordStream":
        return self

    def __next__(self) -> object:
        self.next_calls += 1
        if self._offset == self.snapshot_count:
            self.done = True
            raise StopIteration
        begin = self._offset
        end = min(begin + self._batch_rows, self.snapshot_count)
        ids = np.arange(100 + begin, 100 + end, dtype=np.uint64)
        embeddings = np.stack(
            [
                np.arange(identifier, identifier + self._prefix, dtype=np.float16)
                for identifier in ids
            ]
        )
        locations = np.empty(end - begin, dtype=throwdb.ID_LOCATION_DTYPE)
        locations["segment_id"] = 1
        locations["row_index"] = np.arange(begin, end, dtype=np.uint32)
        self._offset = end
        return throwdb.RecordBatch(
            ids=ids,
            embeddings=embeddings,
            locations=locations,
            snapshot=self.snapshot,
            emitted_offset=begin,
        )

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeRecordStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _FakeStreamingRemote:
    def __init__(
        self,
        *,
        values: dict[int, np.ndarray] | None = None,
        record_count: int = 5,
        block_stream_get: bool = False,
        stream_get_failure: BaseException | None = None,
    ) -> None:
        self._values = values or {}
        self._record_count = record_count
        self._block_stream_get = block_stream_get
        self._stream_get_failure = stream_get_failure
        self.embedding_streams: list[_FakeEmbeddingStream] = []
        self.record_streams: list[_FakeRecordStream] = []
        self.closed = False

    def stream_get(
        self,
        ids: object,
        *,
        prefix_dimensions: int,
        batch_rows: int,
        max_batch_payload_bytes: int,
    ) -> _FakeEmbeddingStream:
        assert max_batch_payload_bytes == 64
        stream = _FakeEmbeddingStream(
            ids,
            values=self._values,
            prefix=prefix_dimensions,
            batch_rows=batch_rows,
            block_after_first=self._block_stream_get,
            failure=self._stream_get_failure,
        )
        self._stream_get_failure = None
        self.embedding_streams.append(stream)
        return stream

    def describe(self) -> SimpleNamespace:
        return SimpleNamespace(count=self._record_count)

    def scan_records(
        self,
        *,
        prefix_dimensions: int,
        batch_rows: int,
        max_batch_payload_bytes: int,
        with_locations: bool,
    ) -> _FakeRecordStream:
        assert max_batch_payload_bytes == 64
        assert with_locations is True
        stream = _FakeRecordStream(
            count=self._record_count,
            prefix=prefix_dimensions,
            batch_rows=batch_rows,
        )
        self.record_streams.append(stream)
        return stream

    def close(self) -> None:
        self.closed = True
        for stream in (*self.embedding_streams, *self.record_streams):
            stream.close()


def _streaming_workload(
    *,
    config: object,
    remote: _FakeStreamingRemote,
    budget: object,
    source_embeddings: np.ndarray | None,
) -> tuple[object, object]:
    population = np.asarray([10, 20], dtype=np.uint64)
    source_rows = (
        np.asarray([1, 0], dtype=np.int64)
        if source_embeddings is not None
        else None
    )
    operations = SOAK.OperationMetrics(100, seed=17)
    errors = SOAK.ErrorMetrics()
    workload = SOAK.StreamingWorkload(
        config=config,
        population=population,
        source_rows=source_rows,
        source_embeddings=source_embeddings,
        shadow_payload=np.asarray(
            [[10, 11, 12]],
            dtype=np.float16,
        ),
        dimension=3,
        dtype="float16",
        budget=budget,
        operations=operations,
        errors=errors,
    )
    workload._connect = lambda endpoint: remote
    return workload, errors


def test_streaming_workload_waits_for_release_checks_source_and_stops_cleanly() -> None:
    source = np.asarray(
        [
            [20, 21, 22],
            [10, 11, 12],
        ],
        dtype=np.float16,
    )
    remote = _FakeStreamingRemote(
        values={10: source[1], 20: source[0]},
        block_stream_get=True,
    )
    workload, errors = _streaming_workload(
        config=_streaming_config(
            stream_get_workers=1,
            stream_get_rows=4,
        ),
        remote=remote,
        budget=_streaming_budget(),
        source_embeddings=source,
    )

    workload.start_generation("fake:1", 3)
    stream = remote.embedding_streams[0]
    assert stream.next_calls == 0

    workload.release_and_wait_for_progress()
    assert stream.second_next_started.wait(1.0)
    event = workload.stop_generation("unit_test")
    summary = workload.summary()
    mode = summary["modes"]["stream_get"]

    assert event["joined"] is True
    assert event["progress_rows"] == {"stream_get": 2}
    assert stream.closed is True
    assert remote.closed is True
    assert errors.total == 0
    assert mode["workers_started"] == 1
    assert mode["workers_joined"] == 1
    assert mode["attempts_cancelled"] == 1
    assert mode["batches"] == 1
    assert mode["rows"] == 2
    assert mode["source_verified_rows"] == 2
    assert mode["shadow_checks"] == 1


def test_streaming_workload_scan_closes_after_bounded_target() -> None:
    config = _streaming_config(
        scan_records_workers=1,
        scan_records_rows=3,
    )
    # The bounded target is three rows, but the second two-row response has
    # already crossed the transport before the client can close the stream.
    reserved_rows = 4
    max_bytes = SOAK._wire_scan_records(
        reserved_rows,
        config.prefix,
        np.dtype("float16").itemsize,
        with_locations=True,
    )
    remote = _FakeStreamingRemote(record_count=5)
    workload, errors = _streaming_workload(
        config=config,
        remote=remote,
        budget=_streaming_budget(max_bytes),
        source_embeddings=None,
    )

    workload.start_generation("fake:1", 4)
    stream = remote.record_streams[0]
    assert stream.next_calls == 0
    workload.release_and_wait_for_progress()
    _wait_until(
        lambda: workload.summary()["modes"]["scan_records"]["budget_stops"] == 1
    )
    event = workload.stop_generation("unit_test")
    mode = workload.summary()["modes"]["scan_records"]

    assert event["joined"] is True
    assert stream.closed is True
    assert stream.done is False
    assert stream.next_calls == 2
    assert errors.total == 0
    assert mode["attempts_started"] == 1
    assert mode["attempts_completed"] == 1
    assert mode["budget_stops"] == 1
    assert mode["batches"] == 2
    assert mode["rows"] == reserved_rows
    assert mode["first_digest_blake2b_256"] is not None


def test_streaming_workload_propagates_unexpected_stream_failure() -> None:
    remote = _FakeStreamingRemote(
        values={
            10: np.asarray([10, 11, 12], dtype=np.float16),
            20: np.asarray([20, 21, 22], dtype=np.float16),
        },
        stream_get_failure=RuntimeError("injected stream failure"),
    )
    workload, errors = _streaming_workload(
        config=_streaming_config(stream_get_workers=1),
        remote=remote,
        budget=_streaming_budget(),
        source_embeddings=None,
    )

    workload.start_generation("fake:1", 5)
    with pytest.raises(RuntimeError, match="injected stream failure"):
        workload.release_and_wait_for_progress()
    with pytest.raises(RuntimeError, match="injected stream failure"):
        workload.stop_generation("unit_test_cleanup")

    mode = workload.summary()["modes"]["stream_get"]
    assert errors.total == 1
    assert remote.closed is True
    assert mode["unexpected_failures"] == 1
    assert mode["workers_joined"] == 1


def test_minimal_native_mixed_streaming_soak(tmp_path: Path) -> None:
    configured = os.environ.get("THROWDB_SERVER_BINARY")
    if not configured:
        pytest.skip("THROWDB_SERVER_BINARY is not configured")
    native = Path(configured)
    if not native.is_file():
        pytest.skip("configured ThrowDB native server is unavailable")

    store = tmp_path / "store"
    ids_path = tmp_path / "ids.npy"
    embeddings_path = tmp_path / "embeddings.npy"
    results = tmp_path / "results"
    ids = np.arange(1, 65, dtype=np.uint64)
    embeddings = (
        np.arange(ids.size * 8, dtype=np.float32).reshape(ids.size, 8)
        % 127
    ).astype(np.float16)
    np.save(ids_path, ids)
    np.save(embeddings_path, embeddings)
    with throwdb.Collection.create(
        store,
        schema=throwdb.CollectionSchema(dimension=8, dtype="float16"),
        segment_size_bytes=1 << 20,
        format_version=7,
    ) as builder:
        builder.add(ids, embeddings)

    exit_code = SOAK.main(
        [
            "--store", str(store),
            "--ids", str(ids_path),
            "--source-embeddings", str(embeddings_path),
            "--results", str(results),
            "--native-binary", str(native),
            "--tuning", "conservative",
            "--max-seconds", "30",
            "--max-transfer-gib", "0.01",
            "--max-client-memory-mib", "128",
            "--max-get-calls", "24",
            "--max-mutations", "9",
            "--get-workers", "2",
            "--max-inflight-gets", "4",
            "--get-batch-sizes", "1,4",
            "--prefix-dimensions", "full,4",
            "--stream-get-rows", "16",
            "--scan-records-rows", "16",
            "--stream-batch-rows", "4",
            "--stream-max-batch-mib", "1",
            "--stream-prefix-dimensions", "4",
            "--stream-attempt-seconds", "10",
            "--stream-progress-timeout-seconds", "5",
            "--stream-stop-timeout-seconds", "5",
            "--stream-source-oracle",
            "--fairness-baseline-get-calls", "4",
            "--min-mixed-small-get-successes", "2",
            "--max-small-get-p99-ratio", "1000",
            "--population-rows", "32",
            "--mutation-every-gets", "100",
            "--restart-cycles", "0",
            "--scan-rows", "4",
            "--scan-batch-rows", "2",
            "--telemetry-interval-seconds", "0.05",
        ]
    )

    assert exit_code == 0
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert report["revision"] == 2
    assert report["status"] == "complete"
    assert report["acceptance"]["passed"] is True
    assert report["streaming_gates"]["modes"]["stream_get"]["passed"] is True
    assert report["streaming_gates"]["modes"]["scan_records"]["passed"] is True
