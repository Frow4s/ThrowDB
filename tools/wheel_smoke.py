"""Black-box smoke test for an installed ThrowDB wheel.

This file deliberately imports only the installed package and public API. It is
used by cibuildwheel from an isolated test environment, so a passing run proves
that the extension, Python wrapper, CLI entry point, and mutable format were all
packaged together correctly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import throwdb


def main() -> None:
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
            shard_size_bytes=1 << 20,
        ) as builder:
            builder.add(ids, embeddings)

        with throwdb.Collection.open(store) as reader:
            np.testing.assert_array_equal(reader.get(ids), embeddings)
            np.testing.assert_array_equal(
                reader.get([3, 1], prefix_dimensions=3),
                embeddings[[2, 0], :3],
            )

        replacement = np.full((1, 8), 7, dtype=np.float16)
        with throwdb.Collection.open(store, mode="r+") as writer:
            writer.update(np.array([2], dtype=np.int32), replacement)
            writer.insert([4], np.full((1, 8), 9, dtype=np.float16))
            deleted = writer.delete([1, 999])
            np.testing.assert_array_equal(
                deleted,
                np.array([True, False], dtype=np.bool_),
            )

        with throwdb.Collection.open(store) as reader:
            np.testing.assert_array_equal(reader.get([2]), replacement)
            np.testing.assert_array_equal(
                reader.get([4]),
                np.full((1, 8), 9, dtype=np.float16),
            )

        orphan_report = throwdb.Collection.inspect_orphans(
            store,
            min_age_seconds=0,
        )
        if orphan_report["format"] != "throwdb-orphan-collection-plan":
            raise AssertionError("installed wheel returned an invalid orphan plan")

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


if __name__ == "__main__":
    main()
