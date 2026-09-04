from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import throwdb
from throwdb.cli import main as cli_main


def _vectors(rows: int, dimension: int, *, salt: int = 0) -> np.ndarray:
    bits = (
        np.arange(rows * dimension, dtype=np.uint16) * np.uint16(7_919)
        + np.uint16(17 + salt)
    )
    return bits.reshape(rows, dimension).view(np.float16)


def _build_garbage(
    path: Path,
    *,
    format_version: int = 7,
    deleted_ids: tuple[int, ...] = (1, 2, 3),
) -> None:
    dimension = 128
    ids = np.arange(1, 9, dtype=np.uint64)
    vectors = _vectors(ids.size, dimension, salt=3)
    schema = throwdb.CollectionSchema(dimension=dimension, dtype="float16")
    with throwdb.Collection.create(
        path,
        schema=schema,
        shard_size_bytes=4 * dimension * 2,
        format_version=format_version,
    ) as builder:
        builder.add(ids, vectors)
    with throwdb.Collection.open(path, mode="r+") as writer:
        np.testing.assert_array_equal(
            writer.delete(deleted_ids), [True] * len(deleted_ids)
        )
        assert writer.rotate_wal()["rotated"] is True
        assert writer.materialize_frozen_wals()["materialized"] is True


_PERMISSIVE = {
    "max_candidate_segments": 1,
    "max_sidecar_scan_bytes": 1 << 20,
    "max_source_bytes": 1 << 20,
    "max_output_bytes": 1 << 20,
    "max_memory_bytes": 1 << 20,
    "min_garbage_rows": 1,
    "min_reclaimable_bytes": 0,
    "min_garbage_ratio_ppm": 0,
}


def test_python_clean_segments_is_safe_by_default_and_advances_execute_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    _build_garbage(path)
    with throwdb.Collection.open(path, mode="r+") as writer:
        generation = writer.mutation_log_stats["generation"]
        plan = writer.clean_segments(**_PERMISSIVE)
        assert plan["dry_run"] is True
        assert plan["candidate_found"] is True
        assert plan["cleaned"] is False
        assert plan["candidate_segment_id"] == 1
        assert plan["source_rows"] == 4
        assert plan["relocated_rows"] == 1
        assert plan["garbage_rows"] == 3
        assert plan["next_start_after_segment_id"] == 1
        assert writer.mutation_log_stats["generation"] == generation
        assert writer._cleaning_cursor == 0

        executed = writer.clean_segments(dry_run=False, **_PERMISSIVE)
        assert executed["cleaned"] is True
        assert executed["reclamation_complete"] is False
        assert writer._cleaning_cursor == executed["next_start_after_segment_id"]
        np.testing.assert_array_equal(writer.get([4]), _vectors(8, 128, salt=3)[3:4])
        with pytest.raises(KeyError):
            writer.get([1])

        with pytest.raises(TypeError, match="dry_run"):
            writer.clean_segments(dry_run=1, **_PERMISSIVE)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="min_garbage_ratio_ppm"):
            writer.clean_segments(
                min_garbage_ratio_ppm=1_000_001,
                **{k: v for k, v in _PERMISSIVE.items() if k != "min_garbage_ratio_ppm"},
            )
        with pytest.raises(ValueError, match="start_after_segment_id"):
            writer.clean_segments(
                start_after_segment_id=0xFFFF_FFFF,
                **_PERMISSIVE,
            )

    with throwdb.Collection.open(path, verify_index=True) as reopened:
        assert reopened.count == 5
        np.testing.assert_array_equal(
            reopened.get([4]), _vectors(8, 128, salt=3)[3:4]
        )


def test_clean_segments_cli_dry_run_exposes_every_bound(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cli"
    _build_garbage(path)
    cli_main(
        [
            "clean-segments",
            "--storage",
            str(path),
            "--dry-run",
            "--max-candidate-segments",
            "1",
            "--max-sidecar-scan-bytes",
            str(1 << 20),
            "--max-source-bytes",
            str(1 << 20),
            "--max-output-bytes",
            str(1 << 20),
            "--max-memory-bytes",
            str(1 << 20),
            "--min-garbage-rows",
            "1",
            "--min-reclaimable-bytes",
            "0",
            "--min-garbage-ratio-ppm",
            "0",
            "--start-after-segment-id",
            "0",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["candidate_found"] is True
    assert report["candidate_segment_id"] == 1
    with throwdb.Collection.open(path) as unchanged:
        assert unchanged.count == 5


def test_executing_maintain_advances_past_a_clean_segment_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cursor-maintain"
    _build_garbage(path, deleted_ids=(5, 6, 7))
    with throwdb.Collection.open(path, mode="r+") as writer:
        folded = writer.compact_deltas(
            max_input_bytes=1 << 20,
            max_input_entries=16,
        )
        assert folded["compacted"] is True
        policy = {
            "clean_max_candidate_segments": 1,
            "clean_max_sidecar_scan_bytes": 1 << 20,
            "clean_max_source_bytes": 1 << 20,
            "clean_max_output_bytes": 1 << 20,
            "clean_max_memory_bytes": 1 << 20,
            "clean_min_garbage_rows": 1,
            "clean_min_reclaimable_bytes": 0,
            "clean_min_garbage_ratio_ppm": 0,
            "orphan_min_age_seconds": 3600,
        }

        first = writer.maintain(dry_run=False, **policy)
        first_clean = first["actions"][3]
        assert first_clean["name"] == "clean_segments"
        assert first_clean["planned"] is False
        assert first_clean["attempted"] is True
        assert first["policy"]["clean_start_after_segment_id"] == 0
        assert first_clean["result"]["next_start_after_segment_id"] == 1
        assert writer._cleaning_cursor == 1

        second = writer.maintain(dry_run=False, **policy)
        assert second["policy"]["clean_start_after_segment_id"] == 1
        second_clean = second["actions"][3]
        assert second_clean["performed"] is True
        assert second_clean["result"]["candidate_segment_id"] == 2
