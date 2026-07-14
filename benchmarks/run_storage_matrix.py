#!/usr/bin/env python3
"""Compare bounded ThrowDB tuning runs across storage targets.

The wrapper deliberately performs measurement-only calibration.  Every target
receives the same seed and per-target time/read/memory limits, targets run
sequentially, and ``Collection.tune`` is always called with ``dry_run=True``.
It therefore cannot publish a reader profile.  Use ``throwdb tune`` separately
after reviewing the comparison if a target has an exact or compatible reusable
identity.  The aggregate target summaries retain ``profile_reuse_level`` so an
opaque-overlay compatibility result is not confused with an exact device or
network-filesystem identity.

Example:

    python benchmarks/run_storage_matrix.py \
      --target local=/data/local/store \
      --target ceph=/mnt/ceph/store \
      --results benchmark-results/storage-matrix

The result directory contains ``<name>-tune.json`` for every target,
``<name>/vmstat.log`` and ``<name>/diskstats.jsonl`` when those monitors are
available, and one aggregate ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import throwdb


_BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
if str(_BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIRECTORY))

import run_load_test as load_test  # noqa: E402


_TARGET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_REPORT_FORMAT = "throwdb-storage-tuning-matrix"
_REPORT_VERSION = 1
_CALIBRATION_FORMAT = "throwdb-reader-calibration"
_CALIBRATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    path: Path


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _target_spec(value: str) -> TargetSpec:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target must use NAME=PATH syntax"
        ) from error
    if not _TARGET_NAME.fullmatch(name) or name in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "target name must contain only letters, digits, '.', '_' or '-', "
            "start with a letter or digit, and be at most 64 characters"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("target path must not be empty")
    return TargetSpec(name=name, path=Path(raw_path).expanduser())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=_target_spec,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named collection target; pass at least two",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--max-seconds",
        type=_positive_float,
        default=300.0,
        help="cooperative time limit per target (default: 300)",
    )
    parser.add_argument(
        "--max-read-gib",
        type=_positive_int,
        default=8,
        help="planned physical-read limit per target in GiB (default: 8)",
    )
    parser.add_argument(
        "--max-memory-mib",
        type=_positive_int,
        default=512,
        help="additional calibration memory per target in MiB (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=20_260_713,
        help="the same deterministic workload seed for every target",
    )
    parser.add_argument(
        "--no-monitors",
        action="store_true",
        help="do not start vmstat, iostat, or /proc/diskstats monitors",
    )
    return parser


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _validated_targets(
    parser: argparse.ArgumentParser,
    raw_targets: Sequence[TargetSpec],
    raw_results: Path,
) -> tuple[tuple[TargetSpec, ...], Path]:
    if len(raw_targets) < 2:
        parser.error("pass at least two --target NAME=PATH arguments")

    names: set[str] = set()
    paths: list[Path] = []
    targets: list[TargetSpec] = []
    for raw in raw_targets:
        folded = raw.name.casefold()
        if folded in names:
            parser.error(f"duplicate target name: {raw.name}")
        names.add(folded)
        path = raw.path.resolve()
        if not path.is_dir():
            parser.error(f"target is not an existing directory: {path}")
        for existing in paths:
            try:
                duplicate = os.path.samefile(path, existing)
            except OSError:
                duplicate = path == existing
            if duplicate:
                parser.error(
                    f"targets must identify different collection directories: {path}"
                )
        paths.append(path)
        targets.append(TargetSpec(raw.name, path))

    results = raw_results.expanduser().resolve()
    for target in targets:
        if _is_within(results, target.path):
            parser.error(
                "--results must not be the target collection or a directory "
                f"inside it: {target.path}"
            )
    if results.exists():
        if not results.is_dir():
            parser.error(f"--results is not a directory: {results}")
        try:
            nonempty = next(results.iterdir(), None) is not None
        except OSError as error:
            parser.error(f"cannot inspect --results {results}: {error}")
        if nonempty:
            parser.error(f"--results must be empty: {results}")
    return tuple(targets), results


def _atomic_json_dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _error_details(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def _safe_system_information(path: Path) -> dict[str, object]:
    try:
        return dict(load_test.collect_system_information(path))
    except Exception as error:  # diagnostic collection must not block tuning
        return {"error": _error_details(error)}


def _validate_tune_report(report: object) -> str | None:
    if not isinstance(report, Mapping):
        return "Collection.tune returned a non-mapping report"
    if report.get("format") != _CALIBRATION_FORMAT:
        return "unexpected calibration report format"
    if report.get("version") != _CALIBRATION_VERSION:
        return "unexpected calibration report version"
    if report.get("status") != "dry_run":
        return f"calibration did not complete: status={report.get('status')!r}"
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        return "completed dry-run has no selection"
    if selection.get("persisted") is not False:
        return "read-only invariant failed: selection.persisted is not false"
    if not isinstance(selection.get("winner"), str):
        return "completed dry-run has no winner"
    stability = report.get("stability")
    if not isinstance(stability, Mapping) or stability.get("passed") is not True:
        return "completed dry-run did not pass the stability guardrail"
    target = report.get("target")
    if not isinstance(target, Mapping):
        return "completed dry-run has no target diagnostics"
    if target.get("stable_at_publication") is not True:
        return "target was not stable at the end of calibration"
    return None


def _scenario_summary(candidate: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    scenarios = candidate.get("scenarios")
    if not isinstance(scenarios, list):
        return result
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            continue
        name = scenario.get("name")
        summary = scenario.get("summary")
        if not isinstance(name, str) or not isinstance(summary, Mapping):
            continue
        result[name] = {
            "median_throughput_rows_per_second": summary.get(
                "median_throughput_rows_per_second"
            ),
            "median_p95_call_seconds": summary.get(
                "median_p95_call_seconds"
            ),
            "throughput_cv": summary.get("throughput_cv"),
        }
    return result


def _candidate_summary(candidate: Mapping[str, object]) -> dict[str, object]:
    guardrails = candidate.get("guardrails")
    guardrails = guardrails if isinstance(guardrails, Mapping) else {}
    return {
        "status": candidate.get("status"),
        "score": candidate.get("score"),
        "eligible": guardrails.get("eligible"),
        "guardrail_reasons": guardrails.get("reasons"),
        "options": candidate.get("options"),
        "scenarios": _scenario_summary(candidate),
    }


def _target_summary(report: Mapping[str, object]) -> dict[str, object]:
    selection = report["selection"]
    assert isinstance(selection, Mapping)
    winner = selection["winner"]
    candidates = report.get("candidates")
    candidate_list = candidates if isinstance(candidates, list) else []
    winner_candidate = next(
        (
            item
            for item in candidate_list
            if isinstance(item, Mapping) and item.get("name") == winner
        ),
        None,
    )
    target = report.get("target")
    target = target if isinstance(target, Mapping) else {}
    device = target.get("device")
    device = device if isinstance(device, Mapping) else {}
    open_stats = target.get("open_stats")
    open_stats = open_stats if isinstance(open_stats, Mapping) else {}
    budget = report.get("budget")
    budget = budget if isinstance(budget, Mapping) else {}
    discovery = report.get("candidate_discovery")
    discovery = discovery if isinstance(discovery, Mapping) else {}
    warning = target.get("identity_warning")
    return {
        "calibration_status": report.get("status"),
        "profile_reuse_level": target.get("profile_reuse_level"),
        "winner": winner,
        "score": selection.get("score"),
        "options": selection.get("options"),
        "winner_scenarios": (
            _scenario_summary(winner_candidate)
            if isinstance(winner_candidate, Mapping)
            else {}
        ),
        "stability": report.get("stability"),
        "sample": report.get("sample"),
        "candidate_count": len(candidate_list),
        "unavailable_candidates": discovery.get("unavailable"),
        "device": {
            "kind": device.get("kind"),
            "confidence": device.get("confidence"),
            "evidence": device.get("evidence"),
            "reuse_scope": target.get("reuse_scope"),
            "profile_reusable": target.get("profile_reusable"),
            "identity_warning": warning,
        },
        "budget": {
            "elapsed_seconds": budget.get("elapsed_seconds"),
            "planned_read_bytes": budget.get("planned_read_bytes"),
            "max_read_bytes": budget.get("max_read_bytes"),
            "max_memory_bytes": budget.get("max_memory_bytes"),
        },
        "open": {
            "total_seconds": open_stats.get("total_seconds"),
            "index_seconds": open_stats.get("index_seconds"),
            "index_file_bytes": open_stats.get("index_file_bytes"),
        },
    }


def _candidate_map(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        if isinstance(candidate, Mapping) and isinstance(
            candidate.get("name"), str
        ):
            result[str(candidate["name"])] = candidate
    return result


def _throughput_ratios(
    summaries: Mapping[str, Mapping[str, object]],
    reference: str,
) -> dict[str, dict[str, float | None]]:
    reference_scenarios = summaries[reference].get("scenarios")
    if not isinstance(reference_scenarios, Mapping):
        return {}
    ratios: dict[str, dict[str, float | None]] = {}
    for target, summary in summaries.items():
        scenarios = summary.get("scenarios")
        if not isinstance(scenarios, Mapping):
            continue
        target_ratios: dict[str, float | None] = {}
        for scenario_name, reference_metrics in reference_scenarios.items():
            metrics = scenarios.get(scenario_name)
            if not isinstance(reference_metrics, Mapping) or not isinstance(
                metrics, Mapping
            ):
                target_ratios[str(scenario_name)] = None
                continue
            baseline = reference_metrics.get(
                "median_throughput_rows_per_second"
            )
            value = metrics.get("median_throughput_rows_per_second")
            if not isinstance(baseline, (int, float)) or not isinstance(
                value, (int, float)
            ) or baseline == 0:
                target_ratios[str(scenario_name)] = None
            else:
                target_ratios[str(scenario_name)] = float(value / baseline)
        ratios[target] = target_ratios
    return ratios


def build_comparison(
    target_order: Sequence[str],
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    first = target_order[0]
    target_sections: dict[str, Mapping[str, object]] = {}
    open_sections: dict[str, Mapping[str, object]] = {}
    for name in target_order:
        target = reports[name].get("target")
        assert isinstance(target, Mapping)
        target_sections[name] = target
        open_stats = target.get("open_stats")
        open_sections[name] = (
            open_stats if isinstance(open_stats, Mapping) else {}
        )
    layouts = {
        name: target_sections[name].get("layout") for name in target_order
    }
    workloads = {name: reports[name].get("workload") for name in target_order}
    mutations = {
        name: target_sections[name].get("mutation_snapshot")
        for name in target_order
    }
    index_sizes = {
        name: open_sections[name].get("index_file_bytes")
        for name in target_order
    }
    layout_compatible = all(layouts[name] == layouts[first] for name in target_order)
    workload_compatible = all(
        workloads[name] == workloads[first] for name in target_order
    )
    mutation_compatible = all(
        mutations[name] == mutations[first] for name in target_order
    )
    index_size_compatible = all(
        index_sizes[name] == index_sizes[first] for name in target_order
    )

    maps = {name: _candidate_map(reports[name]) for name in target_order}
    common_names = [
        name
        for name in maps[first]
        if all(name in maps[target] for target in target_order)
    ]
    candidate_comparison: list[dict[str, object]] = []
    for candidate_name in common_names:
        summaries = {
            target: _candidate_summary(maps[target][candidate_name])
            for target in target_order
        }
        candidate_comparison.append(
            {
                "name": candidate_name,
                "targets": summaries,
                "throughput_ratio_to_reference": _throughput_ratios(
                    summaries,
                    first,
                ),
            }
        )

    comparable = (
        layout_compatible
        and workload_compatible
        and mutation_compatible
        and index_size_compatible
    )
    return {
        "reference_target": first,
        "score_comparison_note": (
            "candidate scores are normalized to each target's sequential "
            "baseline and must not be compared directly across targets"
        ),
        "comparable": comparable,
        "layout_compatible": layout_compatible,
        "workload_compatible": workload_compatible,
        "mutation_snapshot_compatible": mutation_compatible,
        "index_file_size_compatible": index_size_compatible,
        "targets": {
            name: _target_summary(reports[name]) for name in target_order
        },
        "common_candidate_count": len(common_names),
        "candidates": candidate_comparison,
    }


def _monitor_contexts(target_directory: Path, disabled: bool) -> tuple[object, ...]:
    if disabled:
        return (nullcontext(), nullcontext(), None)
    diskstats = load_test.ProcDiskstatsMonitor(
        target_directory / "diskstats.jsonl"
    )
    return (
        load_test.IostatMonitor(target_directory / "iostat.log"),
        load_test.CommandMonitor(
            ("vmstat", "1"), target_directory / "vmstat.log"
        ),
        diskstats,
    )


def _run_target(
    target: TargetSpec,
    target_directory: Path,
    *,
    max_seconds: float,
    max_read_bytes: int,
    max_memory_bytes: int,
    seed: int,
    monitors_disabled: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    contexts = _monitor_contexts(target_directory, monitors_disabled)
    diskstats = contexts[2]

    def progress(message: str) -> None:
        _log(f"[{target.name}] {message}")

    if monitors_disabled:
        tune_report = throwdb.Collection.tune(
            target.path,
            max_seconds=max_seconds,
            max_read_bytes=max_read_bytes,
            max_memory_bytes=max_memory_bytes,
            seed=seed,
            dry_run=True,
            progress=progress,
        )
        return tune_report, {"status": "disabled"}

    first, second, diskstats_monitor = contexts
    assert diskstats_monitor is not None
    with first, second, diskstats_monitor:  # type: ignore[attr-defined]
        tune_report = throwdb.Collection.tune(
            target.path,
            max_seconds=max_seconds,
            max_read_bytes=max_read_bytes,
            max_memory_bytes=max_memory_bytes,
            seed=seed,
            dry_run=True,
            progress=progress,
        )
    return tune_report, diskstats_monitor.summary()  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    targets, results = _validated_targets(parser, args.target, args.results)
    results.mkdir(parents=True, exist_ok=True)
    report_path = results / "report.json"
    max_read_bytes = args.max_read_gib * (1 << 30)
    max_memory_bytes = args.max_memory_mib * (1 << 20)

    aggregate: dict[str, object] = {
        "format": _REPORT_FORMAT,
        "version": _REPORT_VERSION,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "configuration": {
            "dry_run": True,
            "targets_run_sequentially": True,
            "target_order": [target.name for target in targets],
            "max_seconds_per_target": args.max_seconds,
            "max_read_bytes_per_target": max_read_bytes,
            "max_memory_bytes_per_target": max_memory_bytes,
            "seed": args.seed,
            "monitors_enabled": not args.no_monitors,
            "results": str(results),
        },
        "targets": {
            target.name: {
                "path": str(target.path),
                "status": "pending",
                "report": f"{target.name}-tune.json",
            }
            for target in targets
        },
        "comparison": None,
    }
    _atomic_json_dump(report_path, aggregate)
    successful: dict[str, Mapping[str, object]] = {}

    target_entries = aggregate["targets"]
    assert isinstance(target_entries, dict)
    for target in targets:
        entry = target_entries[target.name]
        assert isinstance(entry, dict)
        target_directory = results / target.name
        target_directory.mkdir()
        tune_path = results / f"{target.name}-tune.json"
        entry.update(
            status="running",
            started_at=_now(),
            system=_safe_system_information(target.path),
        )
        _atomic_json_dump(report_path, aggregate)
        _log(f"[{target.name}] calibrating {target.path}")
        started = time.perf_counter()
        try:
            tune_report, diskstats = _run_target(
                target,
                target_directory,
                max_seconds=args.max_seconds,
                max_read_bytes=max_read_bytes,
                max_memory_bytes=max_memory_bytes,
                seed=args.seed,
                monitors_disabled=args.no_monitors,
            )
            _atomic_json_dump(tune_path, tune_report)
            validation_error = _validate_tune_report(tune_report)
            entry["diskstats"] = diskstats
            if validation_error is None:
                entry["status"] = "complete"
                entry["summary"] = _target_summary(tune_report)
                successful[target.name] = tune_report
                selection = tune_report["selection"]
                assert isinstance(selection, Mapping)
                _log(f"[{target.name}] winner={selection['winner']}")
                target_info = tune_report["target"]
                assert isinstance(target_info, Mapping)
                identity_warning = target_info.get("identity_warning")
                if identity_warning:
                    _log(
                        f"[{target.name}] profile reuse level="
                        f"{target_info.get('profile_reuse_level')}: "
                        f"{identity_warning}"
                    )
            else:
                entry["status"] = "inconclusive"
                entry["error"] = {
                    "type": "InvalidCalibrationReport",
                    "message": validation_error,
                }
                _log(f"[{target.name}] inconclusive: {validation_error}")
        except Exception as error:
            error_report = {
                "format": "throwdb-storage-target-error",
                "version": 1,
                "status": "failed",
                "target": target.name,
                "path": str(target.path),
                "error": _error_details(error),
            }
            try:
                _atomic_json_dump(tune_path, error_report)
            except Exception as output_error:
                entry["output_error"] = _error_details(output_error)
            entry["status"] = "failed"
            entry["error"] = _error_details(error)
            _log(f"[{target.name}] failed: {error}")
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - started
            entry["finished_at"] = _now()
            _atomic_json_dump(report_path, aggregate)

    order = [target.name for target in targets if target.name in successful]
    if len(order) >= 2:
        comparison = build_comparison(order, successful)
        aggregate["comparison"] = comparison
    else:
        comparison = None

    statuses = [target_entries[target.name]["status"] for target in targets]
    if all(status == "complete" for status in statuses):
        if comparison is not None and comparison["comparable"] is True:
            aggregate["status"] = "complete"
        else:
            aggregate["status"] = "inconclusive"
    elif any(status == "complete" for status in statuses):
        aggregate["status"] = "partial_failure"
    else:
        aggregate["status"] = "failed"
    aggregate["finished_at"] = _now()
    _atomic_json_dump(report_path, aggregate)

    final = {
        "status": aggregate["status"],
        "report": str(report_path),
        "targets": {
            target.name: target_entries[target.name]["status"]
            for target in targets
        },
    }
    print(json.dumps(final, sort_keys=True, allow_nan=False))
    return 0 if aggregate["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
