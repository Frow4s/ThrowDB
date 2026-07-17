from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import throwdb

from ._harness import ServerProcess


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--throwdb-server-binary",
        action="store",
        default=None,
        help="path to the throwdbd binary for out-of-process integration tests",
    )


@pytest.fixture(scope="session")
def throwdb_server_binary(pytestconfig: pytest.Config) -> Path:
    configured = pytestconfig.getoption("--throwdb-server-binary")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    environment = os.environ.get("THROWDB_SERVER_BINARY")
    if environment:
        candidates.append(Path(environment))
    repository = Path(__file__).resolve().parents[2]
    installed_binary = (
        Path(throwdb.__file__).resolve().parent
        / "libexec"
        / "_throwdbd_native"
    )
    candidates.extend(
        [
            installed_binary,
            repository / "build-grpc" / "throwdbd",
            repository / "build-grpc-dev" / "throwdbd",
        ]
    )
    discovered = shutil.which("throwdbd")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    pytest.skip(
        "throwdbd is not built; pass --throwdb-server-binary or set "
        "THROWDB_SERVER_BINARY"
    )


@pytest.fixture
def server_process_factory(
    throwdb_server_binary: Path,
) -> Iterator[Callable[..., ServerProcess]]:
    running: list[ServerProcess] = []

    def start(storage: Path, **options: object) -> ServerProcess:
        process = ServerProcess.start(
            throwdb_server_binary,
            storage,
            **options,  # type: ignore[arg-type]
        )
        running.append(process)
        return process

    yield start
    for process in reversed(running):
        process.close()
