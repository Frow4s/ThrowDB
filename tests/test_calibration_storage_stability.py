from __future__ import annotations

import math

import pytest

import throwdb._calibration as calibration


_PLANNED_READ_BYTES = 10_000


def _scenario(
    name: str,
    *,
    backing_read_ratio: float | None,
    resource_objective: bool = True,
    cache_mode: str = "ambient",
    logical_read_ratio: float = 1.0,
) -> dict[str, object]:
    if backing_read_ratio is None:
        resources: dict[str, object] = {"io": {"available": False}}
        repetition_resources: dict[str, object] = {
            "io": {"available": False}
        }
    else:
        resources = {
            "io": {
                "available": True,
                "storage_read_bytes": int(
                    _PLANNED_READ_BYTES * backing_read_ratio
                ),
                "logical_read_bytes": int(
                    _PLANNED_READ_BYTES * logical_read_ratio
                ),
            }
        }
        repetition_resources = {
            "io": {
                "available": True,
                "storage_read_bytes": int(
                    _PLANNED_READ_BYTES * backing_read_ratio
                ),
                "delta": {
                    "rchar": int(
                        _PLANNED_READ_BYTES * logical_read_ratio
                    )
                },
            }
        }
    return {
        "name": name,
        "resource_objective": resource_objective,
        "cache_mode": cache_mode,
        "summary": {
            "planned_read_bytes": _PLANNED_READ_BYTES,
            "resources": resources,
        },
        "repetitions": [
            {
                "planned_read_bytes": _PLANNED_READ_BYTES,
                "resources": repetition_resources,
            }
        ],
    }


def _stage(
    *scenarios: dict[str, object],
    cache_control: str | None = None,
) -> dict[str, object]:
    stage: dict[str, object] = {
        "status": "measured",
        "scenarios": list(scenarios),
    }
    if cache_control is not None:
        stage["cache_control"] = cache_control
    return stage


def _candidate(
    name: str,
    *,
    screening: dict[str, object] | None = None,
    verification: dict[str, object] | None = None,
    status: str = "measured",
    pairing_reference: dict[str, object] | None = None,
    io_backend: str = "thread_pool",
) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "status": status,
        "options": {"io_backend": io_backend},
    }
    if screening is not None:
        report["screening"] = screening
    if verification is not None:
        report["verification"] = verification
    if pairing_reference is not None:
        report["pairing_reference"] = pairing_reference
    return report


def _storage_report(
    *,
    screening_ratio: float | None,
    verification_ratio: float | None,
) -> dict[str, object]:
    candidate = _candidate(
        "sequential",
        screening=_stage(
            _scenario("serial_1024", backing_read_ratio=screening_ratio)
        ),
        verification=_stage(
            _scenario("serial_1024", backing_read_ratio=verification_ratio)
        ),
    )
    return calibration._storage_stability_report([candidate], [candidate])


def test_storage_threshold_boundaries_are_inclusive() -> None:
    page_cache_max = calibration._PAGE_CACHE_BACKING_RATIO_MAX
    backing_storage_min = calibration._BACKING_STORAGE_RATIO_MIN

    assert calibration._backing_ratio_classification(page_cache_max) == "page_cache"
    assert (
        calibration._backing_ratio_classification(
            math.nextafter(page_cache_max, math.inf)
        )
        == "mixed"
    )
    assert (
        calibration._backing_ratio_classification(
            math.nextafter(backing_storage_min, -math.inf)
        )
        == "mixed"
    )
    assert (
        calibration._backing_ratio_classification(backing_storage_min)
        == "backing_storage"
    )
    assert calibration._backing_ratio_classification(None) == "unknown"


def test_backing_storage_is_persistence_safe() -> None:
    report = _storage_report(screening_ratio=0.80, verification_ratio=0.70)

    assert report["classification"] == "backing_storage"
    assert report["passed"] is True
    assert report["persistence_safe"] is True
    assert report["screening"]["classification"] == "backing_storage"
    assert report["verification"]["classification"] == "backing_storage"
    assert report["thresholds"] == {
        "page_cache_max_backing_read_ratio": (
            calibration._PAGE_CACHE_BACKING_RATIO_MAX
        ),
        "backing_storage_min_backing_read_ratio": (
            calibration._BACKING_STORAGE_RATIO_MIN
        ),
        "logical_read_ratio_min": (
            calibration._MIN_EXPECTED_LOGICAL_READ_RATIO
        ),
        "logical_read_ratio_max": (
            calibration._MAX_EXPECTED_LOGICAL_READ_RATIO
        ),
        "isolated_io_uring_rchar_fallback_max_ratio": (
            calibration._MAX_PROFILED_IO_URING_RCHAR_RATIO
        ),
    }
    assert report["backing_read_ratio"] == pytest.approx(0.75)


def test_logically_disjoint_ids_do_not_make_page_cache_reads_safe() -> None:
    logically_disjoint = "ambient_global_disjoint_interleaved"
    candidate = _candidate(
        "sequential",
        screening=_stage(
            _scenario("serial_1024", backing_read_ratio=0.02),
            cache_control=logically_disjoint,
        ),
        verification=_stage(
            _scenario("serial_10000", backing_read_ratio=0.04),
            cache_control=logically_disjoint,
        ),
    )

    report = calibration._storage_stability_report([candidate], [candidate])

    assert report["classification"] == "page_cache"
    assert report["passed"] is False
    assert report["persistence_safe"] is False
    assert all(
        observation["classification"] == "page_cache"
        for observation in report["observations"]
    )


def test_partial_backing_reads_are_classified_as_mixed() -> None:
    report = _storage_report(screening_ratio=0.30, verification_ratio=0.25)

    assert report["classification"] == "mixed"
    assert report["passed"] is False
    assert report["persistence_safe"] is False
    assert "inconsistent or partial" in report["reason"]


def test_missing_io_telemetry_is_unknown() -> None:
    report = _storage_report(screening_ratio=0.80, verification_ratio=None)

    assert report["classification"] == "unknown"
    assert report["passed"] is False
    assert report["persistence_safe"] is False
    assert report["verification"]["classification"] == "unknown"
    assert report["verification"]["backing_read_ratio"] is None
    assert "evidence was unavailable" in report["reason"]


def test_unrelated_process_reads_cannot_certify_cached_payloads() -> None:
    candidate = _candidate(
        "sequential",
        screening=_stage(
            _scenario(
                "serial_1024",
                backing_read_ratio=0.80,
                logical_read_ratio=1.60,
            )
        ),
        verification=_stage(
            _scenario("serial_1024", backing_read_ratio=0.80)
        ),
    )

    report = calibration._storage_stability_report([candidate], [candidate])

    assert report["classification"] == "unknown"
    assert report["persistence_safe"] is False
    assert "outside the expected" in report["reason"]
    assert report["io_contaminated_scenarios"] == [
        {
            "candidate": "sequential",
            "scope": "screening",
            "scenario": "serial_1024",
            "logical_read_ratio": 1.6,
            "failed_repetitions": [
                {"ordinal": 0, "logical_read_ratio": 1.6}
            ],
        }
    ]


def test_isolated_io_uring_uses_profiled_bytes_when_rchar_omits_payload() -> None:
    """Linux rchar does not include bytes completed through io_uring."""

    candidate = _candidate(
        "io_uring_w1_qd64",
        io_backend="io_uring",
        screening=_stage(
            _scenario(
                "serial_1024",
                backing_read_ratio=1.95,
                logical_read_ratio=0.00015,
            )
        ),
        verification=_stage(
            _scenario(
                "serial_10000",
                backing_read_ratio=1.98,
                logical_read_ratio=0.00003,
            )
        ),
    )

    report = calibration._storage_stability_report(
        [candidate],
        [candidate],
        isolated_process=True,
    )

    assert report["classification"] == "backing_storage"
    assert report["persistence_safe"] is True
    assert report["io_contaminated_scenarios"] == []
    assert report["telemetry_scope"] == "isolated_calibration_process"
    assert len(report["logical_counter_fallback_scenarios"]) == 2
    assert {
        item["scope"]
        for item in report["logical_counter_fallback_scenarios"]
    } == {"screening", "verification"}
    for observation in report["observations"]:
        scenario = observation["scenarios"][0]
        assert scenario["classification"] == "backing_storage"
        assert scenario["io_attribution_passed"] is True
        assert scenario["logical_read_ratio"] < 0.01
        repetition = scenario["repetitions"][0]
        assert repetition["logical_accounting_method"] == (
            "profiled_planned_bytes_isolated_io_uring"
        )
        assert repetition["logical_read_ratio"] < 0.01
        assert repetition["io_attribution_passed"] is True


@pytest.mark.parametrize(
    ("io_backend", "isolated_process"),
    [
        ("io_uring", False),
        ("thread_pool", True),
    ],
    ids=["io-uring-not-isolated", "thread-pool-isolated"],
)
def test_profiled_byte_fallback_is_restricted_to_isolated_io_uring(
    io_backend: str,
    isolated_process: bool,
) -> None:
    candidate = _candidate(
        "candidate",
        io_backend=io_backend,
        screening=_stage(
            _scenario(
                "serial_1024",
                backing_read_ratio=1.95,
                logical_read_ratio=0.00015,
            )
        ),
        verification=_stage(
            _scenario(
                "serial_10000",
                backing_read_ratio=1.98,
                logical_read_ratio=0.00003,
            )
        ),
    )

    report = calibration._storage_stability_report(
        [candidate],
        [candidate],
        isolated_process=isolated_process,
    )

    assert report["classification"] == "unknown"
    assert report["persistence_safe"] is False
    assert report["io_contaminated_scenarios"]
    assert report["logical_counter_fallback_scenarios"] == []
    for observation in report["observations"]:
        repetition = observation["scenarios"][0]["repetitions"][0]
        assert repetition["logical_accounting_method"] == "procfs_rchar"
        assert repetition["io_attribution_passed"] is False


def test_isolated_io_uring_does_not_hide_excess_process_reads() -> None:
    candidate = _candidate(
        "io_uring_w1_qd64",
        io_backend="io_uring",
        screening=_stage(
            _scenario(
                "serial_1024",
                backing_read_ratio=1.95,
                logical_read_ratio=1.60,
            )
        ),
        verification=_stage(
            _scenario(
                "serial_10000",
                backing_read_ratio=1.98,
                logical_read_ratio=1.60,
            )
        ),
    )

    report = calibration._storage_stability_report(
        [candidate],
        [candidate],
        isolated_process=True,
    )

    assert report["classification"] == "unknown"
    assert report["persistence_safe"] is False
    assert report["io_contaminated_scenarios"]
    for observation in report["observations"]:
        repetition = observation["scenarios"][0]["repetitions"][0]
        assert repetition["logical_accounting_method"] == "procfs_rchar"
        assert repetition["io_attribution_passed"] is False


@pytest.mark.parametrize("logical_read_ratio", [0.11, 0.50, 0.74])
def test_isolated_io_uring_rejects_ambiguous_partial_rchar_accounting(
    logical_read_ratio: float,
) -> None:
    candidate = _candidate(
        "io_uring_w1_qd64",
        io_backend="io_uring",
        screening=_stage(
            _scenario(
                "serial_1024",
                backing_read_ratio=1.95,
                logical_read_ratio=logical_read_ratio,
            )
        ),
        verification=_stage(
            _scenario(
                "serial_10000",
                backing_read_ratio=1.98,
                logical_read_ratio=logical_read_ratio,
            )
        ),
    )

    report = calibration._storage_stability_report(
        [candidate],
        [candidate],
        isolated_process=True,
    )

    assert report["classification"] == "unknown"
    assert report["persistence_safe"] is False
    assert report["logical_counter_fallback_scenarios"] == []
    assert report["io_contaminated_scenarios"]


def test_every_non_hot_decision_scenario_participates() -> None:
    resource_scenario = _scenario(
        "serial_1024",
        backing_read_ratio=0.80,
        resource_objective=True,
    )
    hot_scenario = _scenario(
        "hot_small_32",
        backing_read_ratio=0.0,
        resource_objective=False,
        cache_mode="hot",
    )
    non_objective_scenario = _scenario(
        "serial_32",
        backing_read_ratio=0.80,
        resource_objective=False,
    )
    candidate = _candidate(
        "sequential",
        screening=_stage(
            resource_scenario,
            hot_scenario,
            non_objective_scenario,
        ),
        verification=_stage(
            resource_scenario,
            hot_scenario,
            non_objective_scenario,
        ),
    )

    report = calibration._storage_stability_report([candidate], [candidate])

    assert report["classification"] == "backing_storage"
    assert report["persistence_safe"] is True
    assert report["observation_count"] == 2
    assert all(
        observation["scenario_names"] == ["serial_1024", "serial_32"]
        for observation in report["observations"]
    )


def test_large_cold_scenario_cannot_mask_a_warm_small_scenario() -> None:
    candidate = _candidate(
        "sequential",
        screening=_stage(
            _scenario("serial_small", backing_read_ratio=0.0),
            _scenario("serial_10000", backing_read_ratio=1.0),
        ),
        verification=_stage(
            _scenario("serial_small", backing_read_ratio=0.0),
            _scenario("serial_10000", backing_read_ratio=1.0),
        ),
    )

    report = calibration._storage_stability_report([candidate], [candidate])

    assert report["classification"] == "mixed"
    assert report["persistence_safe"] is False
    for observation in report["observations"]:
        assert observation["classification"] == "mixed"
        by_name = {item["name"]: item for item in observation["scenarios"]}
        assert by_name["serial_small"]["classification"] == "page_cache"
        assert by_name["serial_10000"]["classification"] == "backing_storage"


def test_cold_repetition_cannot_mask_a_warm_repetition() -> None:
    scenario = _scenario("serial_1024", backing_read_ratio=0.50)
    scenario["summary"] = {
        "planned_read_bytes": 2 * _PLANNED_READ_BYTES,
        "resources": {
            "io": {
                "available": True,
                "storage_read_bytes": _PLANNED_READ_BYTES,
                "logical_read_bytes": 2 * _PLANNED_READ_BYTES,
            }
        },
    }
    scenario["repetitions"] = [
        {
            "planned_read_bytes": _PLANNED_READ_BYTES,
            "resources": {
                "io": {
                    "available": True,
                    "storage_read_bytes": _PLANNED_READ_BYTES,
                    "delta": {"rchar": _PLANNED_READ_BYTES},
                }
            },
        },
        {
            "planned_read_bytes": _PLANNED_READ_BYTES,
            "resources": {
                "io": {
                    "available": True,
                    "storage_read_bytes": 0,
                    "delta": {"rchar": _PLANNED_READ_BYTES},
                }
            },
        },
    ]
    candidate = _candidate(
        "sequential",
        screening=_stage(scenario),
        verification=_stage(scenario),
    )

    report = calibration._storage_stability_report([candidate], [candidate])

    assert report["classification"] == "mixed"
    assert report["persistence_safe"] is False
    for observation in report["observations"]:
        measured = observation["scenarios"][0]
        # The byte-weighted aggregate crosses the backing-storage threshold,
        # but the warm decision epoch must still fail the all-pass gate.
        assert measured["backing_read_ratio"] == pytest.approx(0.50)
        assert measured["classification"] == "mixed"
        assert [
            repetition["classification"]
            for repetition in measured["repetitions"]
        ] == ["backing_storage", "page_cache"]


def test_screened_out_backing_reads_cannot_mask_page_cached_verification() -> None:
    screened_out = _candidate(
        "thread_pool_w32",
        status="screened_out",
        screening=_stage(
            _scenario("serial_1024", backing_read_ratio=0.90)
        ),
    )
    finalist = _candidate(
        "sequential",
        screening=_stage(
            _scenario("serial_1024", backing_read_ratio=0.90)
        ),
        verification=_stage(
            _scenario("serial_1024", backing_read_ratio=0.02)
        ),
    )

    report = calibration._storage_stability_report(
        [screened_out, finalist],
        [finalist],
    )

    assert report["screening"]["classification"] == "backing_storage"
    assert report["verification"]["classification"] == "page_cache"
    assert report["classification"] == "mixed"
    assert report["passed"] is False
    assert report["persistence_safe"] is False


def test_backfill_pairing_reference_is_authoritative_verification_evidence() -> None:
    pairing_reference = {
        "source": "verification_backfill_sequential",
        "candidate": "sequential",
        "scenarios": [
            _scenario("serial_10000", backing_read_ratio=0.75)
        ],
    }
    finalist = _candidate(
        "thread_pool_w8",
        screening=_stage(
            _scenario("serial_1024", backing_read_ratio=0.80)
        ),
        verification=_stage(
            _scenario("serial_10000", backing_read_ratio=0.70)
        ),
        pairing_reference=pairing_reference,
    )

    report = calibration._storage_stability_report([finalist], [finalist])

    assert report["classification"] == "backing_storage"
    assert report["verification"]["observation_count"] == 2
    pairing_observation = next(
        observation
        for observation in report["observations"]
        if observation["source"] == "verification_backfill_sequential"
    )
    assert pairing_observation["candidate"] == "sequential"
    assert pairing_observation["scope"] == "verification"
    assert pairing_observation["classification"] == "backing_storage"
