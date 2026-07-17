from __future__ import annotations

import importlib.util
import sys
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
