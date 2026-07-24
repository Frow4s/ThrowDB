from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import throwdb


def _load_benchmark_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "run_grpc_server_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "throwdb_run_grpc_server_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _vectors(rows: int, dimension: int) -> np.ndarray:
    bits = (
        np.arange(rows * dimension, dtype=np.uint16) * np.uint16(7_919)
        + np.uint16(31)
    )
    return bits.reshape(rows, dimension).view(np.float16)


def _server_binary() -> Path | None:
    configured = os.environ.get("THROWDB_SERVER_BINARY")
    repository = Path(__file__).resolve().parents[1]
    candidates = [
        Path(configured) if configured else None,
        repository / "build-grpc-dev" / "throwdbd",
        repository / "build" / "server-release" / "throwdbd",
        repository / "build-grpc" / "throwdbd",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", (1,)),
        ("1, 128,1024", (1, 128, 1024)),
    ],
)
def test_positive_integer_list_parser(
    value: str,
    expected: tuple[int, ...],
) -> None:
    assert BENCHMARK.parse_positive_int_list(value) == expected


@pytest.mark.parametrize("value", ["", "0", "1,0", "1,1", "1,nope"])
def test_positive_integer_list_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        BENCHMARK.parse_positive_int_list(value)


def test_prefix_parser_resolves_full_deduplicates_and_bounds() -> None:
    assert BENCHMARK.parse_prefixes("full,128,1024,64", 1024) == (
        1024,
        128,
        64,
    )
    with pytest.raises(ValueError, match="outside"):
        BENCHMARK.parse_prefixes("1025", 1024)
    with pytest.raises(ValueError, match="at least one|integers"):
        BENCHMARK.parse_prefixes("", 1024)


def test_parser_has_bounded_reproducible_defaults() -> None:
    args = BENCHMARK.build_parser().parse_args(
        ["--store", "/store", "--ids", "/ids.npy", "--results", "/results"]
    )

    assert args.max_seconds == 300.0
    assert args.max_transfer_gib == 16.0
    assert args.max_client_memory_mib == 4096.0
    assert args.seed == BENCHMARK.DEFAULT_SEED
    assert args.batch_sizes == (1, 128, 1024)
    assert args.calls == 100
    assert args.concurrency == (1, 2, 4)
    assert args.concurrent_calls_per_worker == 20
    assert args.mixed_small_calls == 200
    assert args.mixed_large_calls == 100
    assert args.cache_mode == "auto"
    assert args.repeats == 3
    assert args.crud_rows == 0
    assert args.endpoint is None


def test_parser_accepts_external_endpoint_without_store_and_bounds_seed() -> None:
    parser = BENCHMARK.build_parser()
    args = parser.parse_args(
        [
            "--endpoint",
            "127.0.0.1:50051",
            "--ids",
            "/ids.npy",
            "--results",
            "/results",
            "--seed",
            str(BENCHMARK.UINT64_MASK),
        ]
    )

    assert args.store is None
    assert args.seed == BENCHMARK.UINT64_MASK
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--ids",
                "/ids.npy",
                "--results",
                "/results",
                "--seed",
                str(BENCHMARK.UINT64_MASK + 1),
            ]
        )


@pytest.mark.parametrize(
    (
        "shutdown_grace_ms",
        "shutdown_hard_timeout_ms",
        "expected_seconds",
    ),
    [
        (None, None, 65.0),
        (45_000, None, 65.0),
        (None, 120_000, 125.0),
        (60_000, 180_000, 185.0),
        (1, 1, 10.0),
        (1, 10_000_000, 900.0),
    ],
)
def test_local_server_stop_timeout_tracks_native_shutdown_bounds(
    shutdown_grace_ms: int | None,
    shutdown_hard_timeout_ms: int | None,
    expected_seconds: float,
) -> None:
    assert BENCHMARK._local_server_stop_timeout_seconds(
        shutdown_grace_ms=shutdown_grace_ms,
        shutdown_hard_timeout_ms=shutdown_hard_timeout_ms,
    ) == expected_seconds


def test_local_server_close_waits_for_configured_shutdown_before_kill(
    tmp_path: Path,
) -> None:
    class Process:
        pid = 123

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_timeouts: list[float] = []

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise BENCHMARK.subprocess.TimeoutExpired("throwdbd", timeout)
            return -9

    class Thread:
        def __init__(self) -> None:
            self.join_timeouts: list[float] = []

        def join(self, *, timeout: float) -> None:
            self.join_timeouts.append(timeout)

    process = Process()
    stdout_thread = Thread()
    stderr_thread = Thread()
    server = BENCHMARK.LocalServer(
        process=process,
        command=["throwdbd"],
        ready={"endpoint": "127.0.0.1:1"},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        _stdout_thread=stdout_thread,
        _stderr_thread=stderr_thread,
        _stop_timeout_seconds=185.0,
    )

    server.close()

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_timeouts == [185.0, 2]
    assert stdout_thread.join_timeouts == [1]
    assert stderr_thread.join_timeouts == [1]


@pytest.mark.parametrize(
    ("locally_launched_server", "expected"),
    [
        (True, ("cold", "warm")),
        (False, ("warm",)),
    ],
)
def test_auto_cache_mode_resolves_for_server_ownership(
    locally_launched_server: bool,
    expected: tuple[str, ...],
) -> None:
    assert BENCHMARK.resolve_cache_modes(
        "auto",
        locally_launched_server=locally_launched_server,
    ) == expected


@pytest.mark.parametrize("requested", ["cold", "both"])
def test_external_server_rejects_cache_eviction_modes(requested: str) -> None:
    with pytest.raises(ValueError, match="external|launched by this harness"):
        BENCHMARK.resolve_cache_modes(
            requested,
            locally_launched_server=False,
        )


def test_resource_budget_reservation_is_atomic_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    rss = [1_000]
    monkeypatch.setattr(BENCHMARK.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: rss[0])
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=100,
        max_client_memory_bytes=200,
        started_monotonic=100.0,
        memory_baseline_rss_bytes=1_000,
    )

    budget.reserve(60, extra_memory_bytes=100)
    assert budget.accounted_bytes == 60
    with pytest.raises(BENCHMARK.BenchmarkBudgetExceeded, match="I/O"):
        budget.reserve(41)
    assert budget.accounted_bytes == 60

    rss[0] = 1_150
    with pytest.raises(BENCHMARK.BenchmarkBudgetExceeded, match="memory"):
        budget.check(extra_memory_bytes=51)

    clock[0] = 109.5
    assert budget.rpc_timeout(5.0) == pytest.approx(0.5)
    clock[0] = 110.0
    with pytest.raises(BENCHMARK.BenchmarkBudgetExceeded, match="wall time"):
        budget.check()

    with pytest.raises(ValueError, match="negative"):
        budget.reserve(-1)


def test_resource_budget_concurrent_reservations_do_not_oversubscribe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=100,
        max_client_memory_bytes=1_000,
        memory_baseline_rss_bytes=1_000,
    )

    def reserve() -> bool:
        try:
            budget.reserve(10)
        except BENCHMARK.BenchmarkBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=20) as executor:
        accepted = list(executor.map(lambda _: reserve(), range(20)))

    assert sum(accepted) == 10
    assert budget.accounted_bytes == 100


def test_resource_budget_rebase_is_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 8_192)
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=100,
        max_client_memory_bytes=100,
        memory_baseline_rss_bytes=4_096,
    )

    budget.rebase_memory("embedded index opened")

    assert budget.memory_baseline_rss_bytes == 8_192
    assert budget.as_dict()["memory_baselines"] == [
        {
            "reason": "embedded index opened",
            "previous_rss_bytes": 4_096,
            "new_rss_bytes": 8_192,
        }
    ]


def test_query_digest_is_deterministic_and_order_sensitive() -> None:
    population = np.array([9, 1, 7, 3, 5], dtype=np.uint32)
    tasks = [
        ("small", 3, 4, 101),
        ("large", 5, 8, 102),
    ]

    first = BENCHMARK.query_digest(population, tasks)
    second = BENCHMARK.query_digest(population, tasks)

    assert first == second
    assert len(first) == 64
    assert first != BENCHMARK.query_digest(population, reversed(tasks))
    assert first != BENCHMARK.query_digest(population[::-1], tasks)
    assert first != BENCHMARK.query_digest(
        population,
        [("small", 3, 4, 101), ("large", 5, 8, 103)],
    )
    sampled = BENCHMARK.sampled_ids(population, 4, 55)
    assert sampled.dtype == np.dtype("<u8")
    assert sampled.flags.c_contiguous


def test_query_digest_supports_uint64_seed_wrap_and_checks_memory_first() -> None:
    population = np.array([1, 2, 3], dtype=np.uint64)
    checks: list[int] = []

    class BudgetProbe:
        def check(self, *, extra_memory_bytes: int = 0) -> None:
            checks.append(extra_memory_bytes)

    digest = BENCHMARK.query_digest(
        population,
        [("wrapped", 2, 1, BENCHMARK.UINT64_MASK)],
        budget=BudgetProbe(),
    )

    assert len(digest) == 64
    assert checks == [48]
    assert BENCHMARK.derived_seed(BENCHMARK.UINT64_MASK, 1) == 0


def test_concurrent_sample_decomposes_client_timing() -> None:
    sample = BENCHMARK.ConcurrentSample(
        task_ordinal=7,
        label="small",
        rows=1,
        ready_ns=1_000_000,
        dequeued_ns=3_000_000,
        rpc_started_ns=4_000_000,
        finished_ns=11_000_000,
    )

    assert sample.client_queue_wait_ns == 2_000_000
    assert sample.client_prepare_ns == 1_000_000
    assert sample.rpc_round_trip_ns == 7_000_000
    assert sample.client_sojourn_ns == 10_000_000
    assert sample.client_sojourn_ns == (
        sample.client_queue_wait_ns
        + sample.client_prepare_ns
        + sample.rpc_round_trip_ns
    )


@pytest.mark.parametrize(
    ("ready_ns", "dequeued_ns", "rpc_started_ns", "finished_ns"),
    [
        (2, 1, 3, 4),
        (1, 3, 2, 4),
        (1, 2, 4, 3),
    ],
)
def test_concurrent_sample_rejects_nonmonotonic_timestamps(
    ready_ns: int,
    dequeued_ns: int,
    rpc_started_ns: int,
    finished_ns: int,
) -> None:
    with pytest.raises(ValueError, match="monotonic|timestamp"):
        BENCHMARK.ConcurrentSample(
            task_ordinal=0,
            label="get",
            rows=1,
            ready_ns=ready_ns,
            dequeued_ns=dequeued_ns,
            rpc_started_ns=rpc_started_ns,
            finished_ns=finished_ns,
        )


def test_concurrent_summary_reports_rpc_legacy_and_timing_distributions() -> None:
    samples = [
        BENCHMARK.ConcurrentSample(
            task_ordinal=0,
            label="small",
            rows=1,
            ready_ns=0,
            dequeued_ns=2_000_000,
            rpc_started_ns=3_000_000,
            finished_ns=10_000_000,
        ),
        BENCHMARK.ConcurrentSample(
            task_ordinal=1,
            label="small",
            rows=1,
            ready_ns=0,
            dequeued_ns=4_000_000,
            rpc_started_ns=6_000_000,
            finished_ns=18_000_000,
        ),
        BENCHMARK.ConcurrentSample(
            task_ordinal=2,
            label="large",
            rows=4,
            ready_ns=0,
            dequeued_ns=1_000_000,
            rpc_started_ns=4_000_000,
            finished_ns=24_000_000,
        ),
    ]

    summary = BENCHMARK.summarize_concurrent_samples(
        samples,
        seconds=2.0,
        itemsize=2,
        prefix=8,
    )

    small = summary["classes"]["small"]
    assert small["calls"] == 2
    assert small["rows"] == 2
    assert small["latency_p50_ms"] == pytest.approx(9.5)
    assert small["latency_p95_ms"] == pytest.approx(11.75)
    assert small["latency_p99_ms"] == pytest.approx(11.95)
    assert small["latency_max_ms"] == pytest.approx(12.0)

    timings = small["timings"]
    assert timings["client_queue_wait"] == {
        "p50_ms": pytest.approx(3.0),
        "p95_ms": pytest.approx(3.9),
        "p99_ms": pytest.approx(3.98),
        "max_ms": pytest.approx(4.0),
    }
    assert timings["client_prepare"] == {
        "p50_ms": pytest.approx(1.5),
        "p95_ms": pytest.approx(1.95),
        "p99_ms": pytest.approx(1.99),
        "max_ms": pytest.approx(2.0),
    }
    assert timings["rpc_round_trip"] == {
        "p50_ms": pytest.approx(9.5),
        "p95_ms": pytest.approx(11.75),
        "p99_ms": pytest.approx(11.95),
        "max_ms": pytest.approx(12.0),
    }
    assert timings["client_sojourn"] == {
        "p50_ms": pytest.approx(14.0),
        "p95_ms": pytest.approx(17.6),
        "p99_ms": pytest.approx(17.92),
        "max_ms": pytest.approx(18.0),
    }

    large = summary["classes"]["large"]
    assert large["calls"] == 1
    assert large["rows"] == 4
    assert large["latency_p50_ms"] == pytest.approx(20.0)
    assert large["timings"]["client_queue_wait"]["p95_ms"] == pytest.approx(
        1.0
    )
    assert large["timings"]["client_prepare"]["p95_ms"] == pytest.approx(3.0)
    assert large["timings"]["rpc_round_trip"]["p95_ms"] == pytest.approx(20.0)
    assert large["timings"]["client_sojourn"]["p95_ms"] == pytest.approx(24.0)


def test_concurrent_class_summary_aggregates_queue_rpc_and_sojourn_timings() -> None:
    def trial(repetition: int, queue_p95: float) -> dict[str, object]:
        timings = {
            scope: {
                "p50_ms": queue_p95 - 1.0,
                "p95_ms": queue_p95,
                "p99_ms": queue_p95 + 1.0,
                "max_ms": queue_p95 + 2.0,
            }
            for scope in (
                "client_queue_wait",
                "client_prepare",
                "rpc_round_trip",
                "client_sojourn",
            )
        }
        return {
            "workload": "mixed",
            "workers": 4,
            "cache_mode": "warm",
            "repetition": repetition,
            "cache_observation": {"classification": "page_cache"},
            "class_query_digests_sha256": {"small": f"digest-{repetition}"},
            "classes": {
                "small": {
                    "seconds": 1.0,
                    "calls": 1,
                    "rows": 10,
                    "rows_per_second": 10.0,
                    "useful_embedding_bytes": 20,
                    "useful_gib_per_second": 20 / BENCHMARK.GIB,
                    "latency_p50_ms": 2.0,
                    "latency_p95_ms": 3.0,
                    "latency_p99_ms": 4.0,
                    "latency_max_ms": 5.0,
                    "timings": timings,
                }
            },
        }

    flattened = BENCHMARK.concurrent_class_trials(
        [trial(1, 10.0), trial(2, 30.0)]
    )
    summary = BENCHMARK.aggregate_trials(
        flattened,
        group_fields=("workload", "workers", "cache_mode", "class_label"),
    )[0]

    assert summary["client_queue_wait_p95_ms"] == pytest.approx(20.0)
    assert summary["rpc_round_trip_p95_ms"] == pytest.approx(20.0)
    assert summary["client_sojourn_p95_ms"] == pytest.approx(20.0)
    assert summary["trial_variation"]["client_queue_wait_p95_ms"]["samples"] == [
        10.0,
        30.0,
    ]
    assert summary["trial_variation"]["client_queue_wait_p95_ms"][
        "coefficient_of_variation"
    ] == pytest.approx(0.5)


def test_matched_fairness_uses_exact_corpus_same_workers_and_rpc_sojourn() -> None:
    mixed = {
        "workers": 4,
        "cache_mode": "warm",
        "repetition": 2,
        "cache_observation": {"classification": "page_cache"},
        "class_query_digests_sha256": {
            "small": "small-corpus",
            "large": "large-corpus",
        },
        "classes": {
            "small": {
                "latency_p95_ms": 20.0,
                "timings": {"client_sojourn": {"p95_ms": 40.0}},
            },
            "large": {
                "latency_p95_ms": 80.0,
                "timings": {"client_sojourn": {"p95_ms": 100.0}},
            },
        },
    }
    isolated_small = {
        "workers": 4,
        "cache_mode": "warm",
        "repetition": 2,
        "cache_observation": {"classification": "page_cache"},
        "class_query_digests_sha256": {"small": "small-corpus"},
        "classes": {
            "small": {
                "latency_p95_ms": 10.0,
                "timings": {"client_sojourn": {"p95_ms": 20.0}},
            }
        },
    }
    isolated_large = {
        "workers": 4,
        "cache_mode": "warm",
        "repetition": 2,
        "cache_observation": {"classification": "page_cache"},
        "class_query_digests_sha256": {"large": "large-corpus"},
        "classes": {
            "large": {
                "latency_p95_ms": 40.0,
                "timings": {"client_sojourn": {"p95_ms": 50.0}},
            }
        },
    }

    fairness = BENCHMARK.matched_fairness_summary(
        mixed,
        isolated_small,
        isolated_large,
    )

    assert fairness["server_fairness_isolated"] is False
    assert fairness["small_rpc_p95_slowdown_vs_matched_isolated"] == 2.0
    assert fairness["large_rpc_p95_slowdown_vs_matched_isolated"] == 2.0
    assert fairness["small_sojourn_p95_slowdown_vs_matched_isolated"] == 2.0
    assert fairness["large_sojourn_p95_slowdown_vs_matched_isolated"] == 2.0
    assert fairness["large_over_small_rpc_p95"] == 4.0
    assert fairness["large_over_small_sojourn_p95"] == 2.5
    assert fairness["cache_reference_set_verified"] is True

    with pytest.raises(AssertionError, match="workers"):
        BENCHMARK.matched_fairness_summary(
            mixed,
            {**isolated_small, "workers": 8},
            isolated_large,
        )
    with pytest.raises(AssertionError, match="exact corpus"):
        BENCHMARK.matched_fairness_summary(
            mixed,
            {
                **isolated_small,
                "class_query_digests_sha256": {"small": "different"},
            },
            isolated_large,
        )

    unverified = BENCHMARK.matched_fairness_summary(
        mixed,
        {
            **isolated_small,
            "cache_observation": {"classification": "mixed"},
        },
        isolated_large,
    )
    assert unverified["cache_reference_set_verified"] is False
    assert unverified["small_rpc_p95_slowdown_vs_matched_isolated"] is None
    assert unverified["large_rpc_p95_slowdown_vs_matched_isolated"] is None
    assert unverified["large_over_small_rpc_p95"] == 4.0


def test_aggregate_trials_uses_weighted_throughput_and_reports_variation() -> None:
    trials = [
        {
            "label": "get_b1024_p1024",
            "cache_mode": "warm",
            "seconds": 1.0,
            "rows": 100,
            "rows_per_second": 100.0,
            "useful_embedding_bytes": BENCHMARK.GIB,
            "useful_gib_per_second": 1.0,
            "latency_p50_ms": 5.0,
            "latency_p95_ms": 10.0,
            "latency_p99_ms": 15.0,
            "latency_max_ms": 20.0,
            "query_digest_sha256": "first",
            "cache_observation": {"classification": "page_cache"},
        },
        {
            "label": "get_b1024_p1024",
            "cache_mode": "warm",
            "seconds": 3.0,
            "rows": 600,
            "rows_per_second": 200.0,
            "useful_embedding_bytes": 6 * BENCHMARK.GIB,
            "useful_gib_per_second": 2.0,
            "latency_p50_ms": 9.0,
            "latency_p95_ms": 30.0,
            "latency_p99_ms": 45.0,
            "latency_max_ms": 60.0,
            "query_digest_sha256": "second",
            "cache_observation": {"classification": "page_cache"},
        },
    ]

    summaries = BENCHMARK.aggregate_trials(
        trials,
        group_fields=("label", "cache_mode"),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["repeat_count"] == 2
    assert summary["rows"] == 700
    assert summary["seconds"] == 4.0
    assert summary["rows_per_second"] == pytest.approx(175.0)
    assert summary["useful_gib_per_second"] == pytest.approx(1.75)
    assert summary["latency_p95_ms"] == pytest.approx(20.0)
    assert summary["query_digests_sha256"] == ["first", "second"]
    assert summary["observed_cache_classifications"] == [
        "page_cache",
        "page_cache",
    ]
    assert summary["cache_verified"] is True
    assert summary["cache_verification_status"] == "warm_verified"
    assert summary["cold_unverified"] is False
    assert summary["trial_variation"]["rows_per_second"] == {
        "samples": [100.0, 200.0],
        "median": 150.0,
        "mean": 150.0,
        "min": 100.0,
        "max": 200.0,
        "coefficient_of_variation": pytest.approx(1.0 / 3.0),
    }
    assert summary["trial_variation"]["latency_p95_ms"]["samples"] == [
        10.0,
        30.0,
    ]


def test_aggregate_trials_marks_advisory_cold_result_unverified() -> None:
    trial = {
        "label": "get_b1024_p1024",
        "cache_mode": "cold",
        "seconds": 1.0,
        "rows": 100,
        "rows_per_second": 100.0,
        "useful_embedding_bytes": BENCHMARK.GIB,
        "useful_gib_per_second": 1.0,
        "query_digest_sha256": "digest",
        "cache_observation": {"classification": "page_cache"},
    }

    summary = BENCHMARK.aggregate_trials(
        [trial], group_fields=("label", "cache_mode")
    )[0]

    assert summary["observed_cache_classifications"] == ["page_cache"]
    assert summary["cache_verified"] is False
    assert summary["cache_verification_status"] == "cold_unverified"
    assert summary["cold_unverified"] is True


def test_remote_embedded_ratio_requires_verified_controlled_cache_pair() -> None:
    def trial(
        *,
        reader: str,
        repetition: int,
        classification: str,
    ) -> dict[str, object]:
        remote = reader == "remote"
        return {
            "reader": reader,
            "label": "get_b2_p4",
            "repetition": repetition,
            "cache_mode": "cold",
            "query_digest_sha256": f"digest-{repetition}",
            "seconds": 2.0 if remote else 1.0,
            "rows_per_second": 100.0 if remote else 200.0,
            "latency_p95_ms": 10.0 if remote else 5.0,
            "cache_observation": {"classification": classification},
        }

    comparisons = BENCHMARK.compare_baseline(
        [
            trial(reader="remote", repetition=1, classification="backing_storage"),
            trial(reader="remote", repetition=2, classification="backing_storage"),
        ],
        [
            trial(reader="embedded", repetition=1, classification="page_cache"),
            trial(reader="embedded", repetition=2, classification="backing_storage"),
        ],
    )

    unverified, verified = comparisons
    assert unverified["cache_pair_verified"] is False
    assert unverified["comparison_status"] == "unverified_cache_pair"
    assert unverified["remote_over_embedded_wall_time"] is None
    assert unverified["remote_over_embedded_rows_per_second"] is None
    assert unverified["remote_over_embedded_p95_latency"] is None
    assert verified["cache_pair_verified"] is True
    assert verified["comparison_status"] == "verified_controlled_cache"
    assert verified["remote_over_embedded_wall_time"] == pytest.approx(2.0)
    assert verified["remote_over_embedded_rows_per_second"] == pytest.approx(0.5)
    assert verified["remote_over_embedded_p95_latency"] == pytest.approx(2.0)


def test_fairness_aggregate_retains_cache_verification() -> None:
    result = BENCHMARK.aggregate_fairness_trials(
        [
            {
                "cache_mode": "cold",
                "cache_observation": {"classification": "page_cache"},
                "fairness": {
                    "small_rpc_p95_slowdown_vs_matched_isolated": 1.25,
                    "cache_reference_set_verified": False,
                    "interpretation": "client-observed matched workload",
                },
            }
        ]
    )[0]

    assert result["cache_verified"] is False
    assert result["cold_unverified"] is True
    assert result["observed_cache_classifications"] == ["page_cache"]
    assert result["matched_reference_cache_verified"] is False
    assert result["matched_reference_cache_statuses"] == [False]


@pytest.mark.parametrize(
    ("backing_read_bytes", "classification"),
    [
        (100, "page_cache"),
        (300, "mixed"),
        (500, "backing_storage"),
    ],
)
def test_cache_observation_classifies_thresholds(
    backing_read_bytes: int,
    classification: str,
) -> None:
    result = {
        "rows": 10,
        "telemetry": {
            "server_process": {
                "io_delta": {"read_bytes": backing_read_bytes}
            }
        },
    }

    BENCHMARK.add_cache_observation(result, record_size=100, remote=True)

    observation = result["cache_observation"]
    assert observation["planned_payload_read_bytes"] == 1_000
    assert observation["backing_read_bytes"] == backing_read_bytes
    assert observation["backing_read_ratio"] == pytest.approx(
        backing_read_bytes / 1_000
    )
    assert observation["classification"] == classification
    assert observation["process_scope"] == "server_process"


def test_cache_observation_is_unknown_without_process_read_bytes() -> None:
    result = {"rows": 10, "telemetry": {}}

    BENCHMARK.add_cache_observation(result, record_size=100, remote=True)

    assert result["cache_observation"] == {
        "planned_payload_read_bytes": 1_000,
        "backing_read_bytes": None,
        "backing_read_ratio": None,
        "classification": "unknown",
        "process_scope": "server_process",
    }


def _unavailable_telemetry(_: int | None) -> object:
    return BENCHMARK.TelemetrySnapshot(
        client={"available": False, "pid": None},
        server={"available": False, "pid": None},
        cgroup={},
    )


def test_sequential_warm_replays_exact_corpus_and_charges_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(BENCHMARK, "telemetry_snapshot", _unavailable_telemetry)
    calls: list[np.ndarray] = []

    class Reader:
        def get(
            self,
            ids: np.ndarray,
            *,
            prefix_dimensions: int | None = None,
        ) -> np.ndarray:
            calls.append(ids.copy())
            width = 4 if prefix_dimensions is None else prefix_dimensions
            return np.zeros((ids.size, width), dtype=np.float16)

    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=10_000,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    result = BENCHMARK.run_sequential_get(
        Reader(),
        np.arange(1, 20, dtype=np.uint64),
        label="get_b2_p4",
        batch_size=2,
        prefix=4,
        dimension=4,
        itemsize=2,
        calls=2,
        warmup_calls=1,
        seed=71,
        budget=budget,
        rpc_timeout=1.0,
        remote=False,
        server_pid=None,
        cache_mode="warm",
    )

    assert len(calls) == 5
    np.testing.assert_array_equal(calls[1], calls[3])
    np.testing.assert_array_equal(calls[2], calls[4])
    assert result["warm_preparation"]["kind"] == "exact_query_replay"
    assert result["warm_preparation"]["query_digest_sha256"] == result[
        "query_digest_sha256"
    ]
    assert result["warmup_accounted_wire_bytes"] == 32
    assert result["prewarm_accounted_wire_bytes"] == 64
    assert result["measured_accounted_wire_bytes"] == 64
    assert result["total_accounted_wire_bytes"] == 160
    assert budget.accounted_bytes == 160


def test_sequential_cold_prepares_after_warmup_before_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(BENCHMARK, "telemetry_snapshot", _unavailable_telemetry)
    events: list[str] = []

    class Reader:
        def get(
            self,
            ids: np.ndarray,
            *,
            prefix_dimensions: int | None = None,
        ) -> np.ndarray:
            events.append("get")
            width = 4 if prefix_dimensions is None else prefix_dimensions
            return np.zeros((ids.size, width), dtype=np.float16)

    def prepare() -> dict[str, object]:
        events.append("cold")
        return {"status": "performed"}

    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=10_000,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    result = BENCHMARK.run_sequential_get(
        Reader(),
        np.arange(1, 20, dtype=np.uint64),
        label="get_b2_p4",
        batch_size=2,
        prefix=4,
        dimension=4,
        itemsize=2,
        calls=1,
        warmup_calls=1,
        seed=72,
        budget=budget,
        rpc_timeout=1.0,
        remote=False,
        server_pid=None,
        cache_mode="cold",
        prepare_cold=prepare,
    )

    assert events == ["get", "cold", "get"]
    assert result["cache_preparation"] == {"status": "performed"}
    assert result["prewarm_accounted_wire_bytes"] == 0


def test_concurrent_warm_reuses_clients_and_exact_prepared_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(BENCHMARK, "telemetry_snapshot", _unavailable_telemetry)
    calls: list[np.ndarray] = []

    class Remote:
        def __init__(self) -> None:
            self._timeout = 1.0

        def get(
            self,
            ids: np.ndarray,
            *,
            prefix_dimensions: int | None = None,
        ) -> np.ndarray:
            calls.append(ids.copy())
            width = 4 if prefix_dimensions is None else prefix_dimensions
            return np.zeros((ids.size, width), dtype=np.float16)

        def health(self) -> dict[str, object]:
            return {"serving": True}

        def close(self) -> None:
            pass

    class RemoteCollection:
        @staticmethod
        def connect(*_: object, **__: object) -> Remote:
            return Remote()

    monkeypatch.setattr(
        BENCHMARK.throwdb,
        "RemoteCollection",
        RemoteCollection,
    )
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=10_000,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    tasks = [
        BENCHMARK.GetTask("get", 2, 4, 81),
        BENCHMARK.GetTask("get", 2, 4, 82),
    ]
    result = BENCHMARK.run_concurrent_get(
        "127.0.0.1:1",
        np.arange(1, 20, dtype=np.uint64),
        tasks,
        dimension=4,
        itemsize=2,
        workers=1,
        budget=budget,
        rpc_timeout=1.0,
        server_pid=None,
        cache_mode="warm",
    )

    assert len(calls) == 5
    np.testing.assert_array_equal(calls[1], calls[3])
    np.testing.assert_array_equal(calls[2], calls[4])
    assert result["warm_preparation"]["kind"] == "exact_query_replay"
    assert result["stabilization_preparation"] == {
        "kind": "one_get_per_connected_worker",
        "calls": 1,
        "accounted_wire_bytes": 32,
    }
    assert result["stabilization_accounted_wire_bytes"] == 32
    assert result["prewarm_accounted_wire_bytes"] == 64
    assert result["measured_accounted_wire_bytes"] == 64
    assert result["total_accounted_wire_bytes"] == 160
    assert budget.accounted_bytes == 160


def test_concurrent_warm_reserves_all_waves_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    connected = False

    class RemoteCollection:
        @staticmethod
        def connect(*_: object, **__: object) -> object:
            nonlocal connected
            connected = True
            raise AssertionError("budget failure must precede client creation")

    monkeypatch.setattr(BENCHMARK.throwdb, "RemoteCollection", RemoteCollection)
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=64,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    tasks = [
        BENCHMARK.GetTask("get", 2, 4, 91),
        BENCHMARK.GetTask("get", 2, 4, 92),
    ]

    with pytest.raises(BENCHMARK.BenchmarkBudgetExceeded):
        BENCHMARK.run_concurrent_get(
            "127.0.0.1:1",
            np.arange(1, 20, dtype=np.uint64),
            tasks,
            dimension=4,
            itemsize=2,
            workers=1,
            budget=budget,
            rpc_timeout=1.0,
            server_pid=None,
            cache_mode="warm",
        )

    assert connected is False
    assert budget.accounted_bytes == 0


def test_concurrent_cold_stabilizes_get_path_before_payload_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(BENCHMARK, "telemetry_snapshot", _unavailable_telemetry)
    events: list[str] = []

    class Remote:
        def __init__(self) -> None:
            self._timeout = 1.0

        def get(
            self,
            ids: np.ndarray,
            *,
            prefix_dimensions: int | None = None,
        ) -> np.ndarray:
            events.append("get")
            width = 4 if prefix_dimensions is None else prefix_dimensions
            return np.zeros((ids.size, width), dtype=np.float16)

        def health(self) -> dict[str, object]:
            events.append("health")
            return {"serving": True}

        def close(self) -> None:
            pass

    class RemoteCollection:
        @staticmethod
        def connect(*_: object, **__: object) -> Remote:
            return Remote()

    def prepare() -> dict[str, object]:
        events.append("cold")
        return {"status": "performed"}

    monkeypatch.setattr(BENCHMARK.throwdb, "RemoteCollection", RemoteCollection)
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=10_000,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    result = BENCHMARK.run_concurrent_get(
        "127.0.0.1:1",
        np.arange(1, 20, dtype=np.uint64),
        [
            BENCHMARK.GetTask("get", 2, 4, 101),
            BENCHMARK.GetTask("get", 2, 4, 102),
        ],
        dimension=4,
        itemsize=2,
        workers=1,
        budget=budget,
        rpc_timeout=1.0,
        server_pid=None,
        cache_mode="cold",
        prepare_cold=prepare,
    )

    assert events == ["health", "get", "cold", "get", "get"]
    assert result["stabilization_accounted_wire_bytes"] == 32
    assert result["prewarm_accounted_wire_bytes"] == 0
    assert result["measured_accounted_wire_bytes"] == 64
    assert result["total_accounted_wire_bytes"] == 96
    assert budget.accounted_bytes == 96


def test_concurrent_stabilization_assigns_one_get_to_every_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BENCHMARK, "current_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(BENCHMARK, "telemetry_snapshot", _unavailable_telemetry)
    clients: list[Remote] = []

    class Remote:
        def __init__(self) -> None:
            self._timeout = 1.0
            self.get_calls = 0

        def get(
            self,
            ids: np.ndarray,
            *,
            prefix_dimensions: int | None = None,
        ) -> np.ndarray:
            self.get_calls += 1
            width = 4 if prefix_dimensions is None else prefix_dimensions
            return np.zeros((ids.size, width), dtype=np.float16)

        def health(self) -> dict[str, object]:
            return {"serving": True}

        def close(self) -> None:
            pass

    class RemoteCollection:
        @staticmethod
        def connect(*_: object, **__: object) -> Remote:
            client = Remote()
            clients.append(client)
            return client

    def prepare() -> dict[str, object]:
        assert [client.get_calls for client in clients] == [1, 1, 1]
        return {"status": "performed"}

    monkeypatch.setattr(BENCHMARK.throwdb, "RemoteCollection", RemoteCollection)
    budget = BENCHMARK.ResourceBudget(
        max_seconds=10.0,
        max_bytes=10_000,
        max_client_memory_bytes=10_000,
        memory_baseline_rss_bytes=1_000,
    )
    result = BENCHMARK.run_concurrent_get(
        "127.0.0.1:1",
        np.arange(1, 20, dtype=np.uint64),
        [BENCHMARK.GetTask("get", 2, 4, 111)],
        dimension=4,
        itemsize=2,
        workers=3,
        budget=budget,
        rpc_timeout=1.0,
        server_pid=None,
        cache_mode="cold",
        prepare_cold=prepare,
    )

    assert len(clients) == 3
    assert sum(client.get_calls for client in clients) == 4
    assert result["stabilization_preparation"]["calls"] == 3
    assert result["stabilization_accounted_wire_bytes"] == 96
    assert result["total_accounted_wire_bytes"] == 128
    assert budget.accounted_bytes == 128


def test_process_delta_reports_cpu_io_and_unavailable() -> None:
    before = {
        "available": True,
        "pid": 17,
        "user_cpu_seconds": 2.0,
        "system_cpu_seconds": 1.0,
        "threads": 3,
        "rss_bytes": 1_000,
        "io": {"read_bytes": 100, "write_bytes": 10},
    }
    after = {
        "available": True,
        "pid": 17,
        "user_cpu_seconds": 5.5,
        "system_cpu_seconds": 2.25,
        "threads": 4,
        "rss_bytes": 2_000,
        "io": {"read_bytes": 500, "write_bytes": 40},
    }

    delta = BENCHMARK.process_delta(before, after)

    assert delta == {
        "available": True,
        "pid": 17,
        "rss_bytes_initial": 1_000,
        "rss_bytes_final": 2_000,
        "rss_bytes_delta": 1_000,
        "threads_initial": 3,
        "threads_final": 4,
        "threads_delta": 1,
        "user_cpu_seconds_delta": 3.5,
        "system_cpu_seconds_delta": 1.25,
        "io_delta": {"read_bytes": 400, "write_bytes": 30},
    }
    assert BENCHMARK.process_delta(
        {"available": False, "pid": 17},
        after,
    ) == {"available": False, "pid": 17}


def test_process_snapshot_parses_proc_stat_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 123
    process_root = tmp_path / str(pid)
    process_root.mkdir()
    tail = ["S", *(["0"] * 21)]
    tail[11] = "250"
    tail[12] = "75"
    tail[17] = "7"
    tail[21] = "3"
    (process_root / "stat").write_text(
        f"{pid} (worker with spaces) {' '.join(tail)}\n",
        encoding="ascii",
    )
    (process_root / "io").write_text(
        "read_bytes: 4096\nwrite_bytes: 512\n",
        encoding="ascii",
    )
    real_path = Path
    monkeypatch.setattr(
        BENCHMARK,
        "Path",
        lambda value: tmp_path if str(value) == "/proc" else real_path(value),
    )
    monkeypatch.setattr(
        BENCHMARK.os,
        "sysconf",
        lambda name: 100 if name == "SC_CLK_TCK" else 4096,
    )

    snapshot = BENCHMARK.read_process_snapshot(pid)

    assert snapshot == {
        "available": True,
        "pid": pid,
        "user_cpu_seconds": 2.5,
        "system_cpu_seconds": 0.75,
        "minor_page_faults": 0,
        "major_page_faults": 0,
        "threads": 7,
        "rss_bytes": 12_288,
        "io": {"read_bytes": 4096, "write_bytes": 512},
    }


def test_telemetry_delta_marks_shared_cgroup_scope() -> None:
    before = BENCHMARK.TelemetrySnapshot(
        client={"available": False, "pid": 1},
        server={"available": False, "pid": 2},
        cgroup={"usage_usec": 100, "throttled_usec": 20},
    )
    after = BENCHMARK.TelemetrySnapshot(
        client={"available": False, "pid": 1},
        server={"available": False, "pid": 2},
        cgroup={"usage_usec": 600, "throttled_usec": 120},
    )

    delta = BENCHMARK.telemetry_delta(before, after)

    assert delta["cgroup_cpu_delta"] == {
        "usage_usec": 500,
        "throttled_usec": 100,
        "throttled_seconds": 0.0001,
    }
    assert "not attributable" in delta["cgroup_scope"]


def test_minimal_loopback_benchmark_smoke(tmp_path: Path) -> None:
    native = _server_binary()
    if native is None:
        pytest.skip("throwdbd native test binary is not built")
    store = tmp_path / "store"
    ids_path = tmp_path / "ids.npy"
    embeddings_path = tmp_path / "embeddings.npy"
    results = tmp_path / "results"
    ids = np.array([11, 22, 33, 44], dtype=np.uint32)
    embeddings = _vectors(ids.size, 8)
    np.save(ids_path, ids)
    np.save(embeddings_path, embeddings)
    with throwdb.Collection.create(
        store,
        schema=throwdb.CollectionSchema(dimension=8, dtype="float16"),
        shard_size_bytes=1 << 20,
        format_version=7,
    ) as builder:
        builder.add(ids, embeddings)

    exit_code = BENCHMARK.main(
        [
            "--store",
            str(store),
            "--ids",
            str(ids_path),
            "--source-embeddings",
            str(embeddings_path),
            "--results",
            str(results),
            "--native-binary",
            str(native),
            "--prefix-dimensions",
            "full,4",
            "--batch-sizes",
            "1,2",
            "--calls",
            "1",
            "--warmup-calls",
            "0",
            "--cache-mode",
            "warm",
            "--repeats",
            "1",
            "--concurrency",
            "1",
            "--concurrent-batch-size",
            "2",
            "--concurrent-calls-per-worker",
            "1",
            "--mixed-workers",
            "1",
            "--mixed-small-batch",
            "1",
            "--mixed-large-batch",
            "2",
            "--mixed-small-calls",
            "1",
            "--mixed-large-calls",
            "1",
            "--correctness-rows",
            "2",
            "--scan-rows",
            "2",
            "--scan-batch-rows",
            "2",
            "--max-seconds",
            "30",
            "--max-transfer-gib",
            "0.01",
            "--max-client-memory-mib",
            "64",
        ]
    )

    assert exit_code == 0
    report = BENCHMARK.json.loads(
        (results / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "complete"
    assert report["format"] == BENCHMARK.REPORT_FORMAT
    assert report["revision"] == 2
    assert report["methodology"]["cache_modes"] == ["warm"]
    assert report["server"]["mode"] == "local_loopback"
    assert report["phases"]["correctness"]["bit_exact"] is True
    assert report["phases"]["scan_ids"]["rows"] == 2
    assert report["phases"]["crud"]["status"] == "skipped"
    assert len(report["phases"]["embedded_baseline"]) == 4
    assert len(report["phases"]["remote_vs_embedded"]) == 4
    assert len(report["phases"]["mixed_get"]) == 1
    assert len(report["phases"]["mixed_isolated"]) == 2
    mixed = report["phases"]["mixed_get"][0]
    assert mixed["fairness"]["reference"].startswith("exact class corpus")
    assert mixed["timing_semantics"]["server_internal_timing_available"] is False
    for class_result in mixed["classes"].values():
        assert class_result["timings"]["client_queue_wait"]["p95_ms"] >= 0
        assert class_result["timings"]["rpc_round_trip"]["p95_ms"] >= 0
        assert class_result["timings"]["client_sojourn"]["p95_ms"] >= 0
    assert report["phases"]["summary"]["mixed_get"][0]["repeat_count"] == 1
    with throwdb.Collection.open(store) as collection:
        np.testing.assert_array_equal(collection.get(ids), embeddings)
