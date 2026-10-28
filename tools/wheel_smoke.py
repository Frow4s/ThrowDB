"""Black-box smoke test for an installed ThrowDB distribution.

This file deliberately imports only the installed package and public API. It is
used by cibuildwheel and the isolated sdist test environment, so a passing run
proves that the extension, Python wrapper, CLI entry point, and format-7
mutation lifecycle were all packaged together correctly.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import throwdb


def _assert_rows(store: Path, expected: np.ndarray) -> None:
    with throwdb.Collection.open(store, verify_index=True) as reader:
        np.testing.assert_array_equal(reader.get([2, 3, 4]), expected)
        np.testing.assert_array_equal(
            reader.get([4, 2], prefix_dimensions=3),
            expected[[2, 0], :3],
        )


def _assert_grpc_server(
    server: Path,
    store: Path,
    expected: np.ndarray,
) -> None:
    process = subprocess.Popen(
        [
            str(server),
            "--storage",
            str(store),
            "--listen",
            "127.0.0.1:0",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_lines: list[str] = []
    ready_lines: queue.Queue[str | None] = queue.Queue()

    def read_stdout() -> None:
        try:
            for line in process.stdout:
                stripped = line.rstrip("\r\n")
                stdout_lines.append(stripped)
                ready_lines.put(stripped)
        finally:
            ready_lines.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    ready: dict[str, object] | None = None
    deadline = time.monotonic() + 15
    try:
        while ready is None and time.monotonic() < deadline:
            try:
                line = ready_lines.get(timeout=0.25)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event") == "ready":
                ready = event

        if ready is None:
            raise AssertionError("packaged throwdbd did not become ready")
        if ready.get("protocol") != "grpc/throwdb.v1":
            raise AssertionError(
                "packaged throwdbd did not announce the public gRPC protocol"
            )
        endpoint = ready.get("endpoint")
        if not isinstance(endpoint, str):
            raise AssertionError("packaged throwdbd announced no endpoint")
        with throwdb.RemoteCollection.connect(endpoint, timeout=5) as remote:
            if not remote.health()["serving"]:
                raise AssertionError("packaged throwdbd is not serving")
            np.testing.assert_array_equal(
                remote.get([2, 3, 4]),
                expected,
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        reader.join(timeout=1)
        stderr = process.stderr.read()
        if process.returncode != 0:
            raise AssertionError(
                "packaged throwdbd exited unsuccessfully\n"
                + "\n".join(stdout_lines)
                + "\n"
                + stderr
            )


def main() -> None:
    assert throwdb.__version__ == importlib.metadata.version("throwdb"), (
        "native extension and installed package versions differ"
    )
    with tempfile.TemporaryDirectory(prefix="throwdb-wheel-smoke-") as raw:
        store = Path(raw) / "store"
        ids = np.array([1, 2, 3], dtype=np.uint32)
        embeddings = np.arange(24, dtype=np.float16).reshape(3, 8)

        with throwdb.Collection.create(
            store,
            schema=throwdb.CollectionSchema(
                id_type="uint64",
                dimension=8,
                dtype="float16",
            ),
            segment_size_bytes=1 << 20,
        ) as builder:
            if builder.format_version != 7:
                raise AssertionError("installed package default is not format 7")
            builder.add(ids, embeddings)

        with throwdb.Collection.open(store) as reader:
            np.testing.assert_array_equal(reader.get(ids), embeddings)
            np.testing.assert_array_equal(
                reader.get([3, 1], prefix_dimensions=3),
                embeddings[[2, 0], :3],
            )

        replacement = np.full((1, 8), 7, dtype=np.float16)
        inserted = np.full((1, 8), 9, dtype=np.float16)
        expected = np.concatenate(
            [replacement, embeddings[[2]], inserted],
            axis=0,
        )
        with throwdb.Collection.open(store, mode="r+") as writer:
            if writer.format_version != 7:
                raise AssertionError("installed package did not create format 7")
            writer.update(np.array([2], dtype=np.int32), replacement)
            writer.insert([4], inserted)
            deleted = writer.delete([1, 999])
            np.testing.assert_array_equal(
                deleted,
                np.array([True, False], dtype=np.bool_),
            )
            rotated = writer.rotate_wal()
            if not rotated["rotated"]:
                raise AssertionError("format-7 WAL did not rotate")
            materialized = writer.materialize_frozen_wals(
                max_wals=8,
                max_input_bytes=8 << 20,
            )
            if not materialized["materialized"]:
                raise AssertionError("format-7 WAL did not materialize")
            folded = writer.fold_partitions(
                max_runs=1,
                max_input_bytes=8 << 20,
                max_input_entries=64,
            )
            if not folded["folded"] or folded["index_run_count"] != 0:
                raise AssertionError("format-7 partitions did not fold")

            cleaner_options = {
                "max_candidate_segments": 8,
                "max_sidecar_scan_bytes": 8 << 20,
                "max_source_bytes": 8 << 20,
                "max_output_bytes": 8 << 20,
                "max_memory_bytes": 16 << 20,
                "min_garbage_rows": 1,
                "min_reclaimable_bytes": 0,
                "min_garbage_ratio_ppm": 0,
                "start_after_segment_id": 0,
            }
            clean_plan = writer.clean_segments(
                dry_run=True,
                **cleaner_options,
            )
            if not clean_plan["candidate_found"] or clean_plan["cleaned"]:
                raise AssertionError("format-7 cleaner returned an invalid plan")
            cleaned = writer.clean_segments(
                dry_run=False,
                **cleaner_options,
            )
            if (
                not cleaned["candidate_found"]
                or not cleaned["cleaned"]
                or cleaned["candidate_segment_id"]
                != clean_plan["candidate_segment_id"]
            ):
                raise AssertionError("format-7 cleaner did not execute its plan")

        _assert_rows(store, expected)

        orphan_report = throwdb.Collection.inspect_orphans(
            store,
            min_age_seconds=0,
        )
        if orphan_report["format"] != "throwdb-orphan-collection-plan":
            raise AssertionError("installed wheel returned an invalid orphan plan")
        if orphan_report["eligible_file_count"] == 0:
            raise AssertionError("format-7 lifecycle produced no collectible files")

        with throwdb.Collection.open(store, mode="r+", verify_index=True) as writer:
            maintenance = writer.maintain(
                dry_run=False,
                max_seconds=30,
                max_actions=5,
                rotate_wal_bytes=1,
                materialize_max_wals=8,
                materialize_max_input_bytes=8 << 20,
                fold_max_input_bytes=8 << 20,
                fold_max_input_entries=64,
                orphan_min_age_seconds=0,
                orphan_max_files=1024,
                orphan_max_bytes=64 << 20,
            )
        if maintenance["format"] != "throwdb-maintenance-report":
            raise AssertionError("installed package returned invalid maintenance")
        fold_action = next(
            action
            for action in maintenance["actions"]
            if action["name"] == "fold_partitions"
        )
        if not fold_action["attempted"] or not fold_action["performed"]:
            raise AssertionError("maintenance did not fold the Cleaner run")
        orphan_action = next(
            action
            for action in maintenance["actions"]
            if action["name"] == "collect_orphans"
        )
        if not orphan_action["attempted"] or not orphan_action["performed"]:
            raise AssertionError("maintenance did not collect lifecycle orphans")

        _assert_rows(store, expected)

        executable = shutil.which("throwdb")
        if executable is None:
            adjacent = Path(sys.executable).with_name("throwdb")
            executable = str(adjacent) if adjacent.is_file() else None
        if executable is None:
            raise AssertionError("installed wheel did not provide the throwdb script")
        completed = subprocess.run(
            [executable, "inspect", "--storage", str(store)],
            check=True,
            capture_output=True,
            text=True,
        )
        if '"dimension": 8' not in completed.stdout:
            raise AssertionError("installed CLI did not inspect the smoke collection")

        require_server = os.environ.get(
            "THROWDB_WHEEL_SMOKE_REQUIRE_SERVER", "1"
        ) not in {"0", "false", "False"}
        if not require_server:
            return

        server_name = shutil.which("throwdbd")
        if server_name is None:
            adjacent = Path(sys.executable).with_name("throwdbd")
            server_name = str(adjacent) if adjacent.is_file() else None
        if server_name is None:
            raise AssertionError(
                "installed wheel did not provide the public gRPC throwdbd command"
            )
        server = Path(server_name)
        server_help = subprocess.run(
            [str(server), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        if "usage: throwdbd" not in server_help.stdout:
            raise AssertionError(
                "packaged throwdbd is not the expected gRPC daemon"
            )
        # An editable install imports the Python package from the source tree,
        # while scikit-build installs compiled artifacts into site-packages.
        # Distribution metadata is therefore the authoritative wheel/editable
        # layout; ``throwdb.__file__`` is only reliable for a regular wheel.
        distribution = importlib.metadata.distribution("throwdb")
        native_server = Path(
            distribution.locate_file(
                "throwdb/libexec/_throwdbd_native"
            )
        ).resolve()
        if not native_server.is_file():
            raise AssertionError(
                "wheel did not include the internal native gRPC daemon"
            )
        if server.resolve() == native_server.resolve():
            raise AssertionError(
                "public throwdbd unexpectedly bypasses the tuning launcher"
            )
        _assert_grpc_server(server, store, expected)


if __name__ == "__main__":
    main()
