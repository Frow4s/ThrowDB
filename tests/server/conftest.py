from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import throwdb
from throwdb.server import ServerLaunchError, find_native_server

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
    configured = (
        pytestconfig.getoption("--throwdb-server-binary")
        or os.environ.get("THROWDB_SERVER_BINARY")
    )
    if configured:
        try:
            return find_native_server(Path(configured))
        except ServerLaunchError as error:
            pytest.fail(f"configured native server is invalid: {error}", pytrace=False)
    repository = Path(__file__).resolve().parents[2]
    installed_binary = (
        Path(throwdb.__file__).resolve().parent
        / "libexec"
        / "_throwdbd_native"
    )
    candidates = [
        installed_binary,
        repository / "build" / "server-release" / "throwdbd",
        repository / "build-grpc" / "throwdbd",
        repository / "build-grpc-dev" / "throwdbd",
    ]
    discovered = shutil.which("throwdbd")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        try:
            return find_native_server(candidate)
        except ServerLaunchError:
            continue
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
