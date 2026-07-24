from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPDATER = REPOSITORY_ROOT / "tools" / "update_remote.sh"


def _copy_updater(destination: Path) -> Path:
    script = destination / "tools" / "update_remote.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(UPDATER, script)
    return script


def _run_updater(
    script: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *arguments],
        cwd=script.parent.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_updater_treats_stale_gitdir_pointer_as_source_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    script = _copy_updater(snapshot)
    (snapshot / ".git").write_text(
        "gitdir: /source-machine/ThrowDB/.git/worktrees/ThrowDB\n",
        encoding="utf-8",
    )

    completed = _run_updater(script, "--pull", "--check", "none")

    assert completed.returncode == 1
    assert (
        "--pull requires a real Git worktree; this directory is a source snapshot"
        in completed.stderr
    )
    assert "fatal: not a git repository" not in completed.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_updater_does_not_adopt_an_unrelated_parent_repository(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "unrelated"
    subprocess.run(
        ["git", "init", "--quiet", str(parent)],
        check=True,
        capture_output=True,
        text=True,
    )
    script = _copy_updater(parent / "snapshot")

    completed = _run_updater(script, "--pull", "--check", "none")

    assert completed.returncode == 1
    assert (
        "--pull requires a real Git worktree; this directory is a source snapshot"
        in completed.stderr
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_updater_rejects_a_copied_live_worktree_pointer(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=ThrowDB test",
            "-c",
            "user.email=throwdb@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    linked_worktree = tmp_path / "linked-worktree"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(linked_worktree),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = tmp_path / "snapshot"
    script = _copy_updater(snapshot)
    shutil.copy2(linked_worktree / ".git", snapshot / ".git")

    completed = _run_updater(script, "--pull", "--check", "none")

    assert completed.returncode == 1
    assert (
        "--pull requires a real Git worktree; this directory is a source snapshot"
        in completed.stderr
    )


def test_updater_rejects_multiline_source_revision(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    script = _copy_updater(snapshot)
    environment = os.environ.copy()
    environment["THROWDB_SOURCE_REVISION"] = (
        "release-1\nprintf 'unexpected command' > injected"
    )

    completed = _run_updater(
        script,
        "--check",
        "none",
        environment=environment,
    )

    assert completed.returncode == 1
    assert (
        "THROWDB_SOURCE_REVISION must be a 1-128 character safe label"
        in completed.stderr
    )
    assert not (snapshot / "build" / "remote-current" / "environment.sh").exists()
