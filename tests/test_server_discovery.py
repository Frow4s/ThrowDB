from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.server import conftest as discovery


@pytest.fixture
def isolated_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repository = tmp_path / "repository"
    monkeypatch.setattr(
        discovery, "__file__", str(repository / "tests" / "server" / "conftest.py")
    )
    package = tmp_path / "site-packages" / "throwdb" / "__init__.py"
    monkeypatch.setattr(
        discovery, "throwdb", SimpleNamespace(__file__=str(package)),
    )
    monkeypatch.delenv("THROWDB_SERVER_BINARY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    return repository


def _executable(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _resolve(explicit: Path | None = None) -> Path:
    config = SimpleNamespace(
        getoption=lambda option: str(explicit) if explicit else None
    )
    return discovery.throwdb_server_binary.__wrapped__(config)


def test_launcher_only_path_skips_native_server_tests(
    tmp_path: Path, isolated_discovery: Path,
) -> None:
    _executable(tmp_path / "bin" / "throwdbd", b"#!/usr/bin/env python3\n")

    with pytest.raises(pytest.skip.Exception, match="throwdbd is not built"):
        _resolve()


@pytest.mark.parametrize("magic", [b"\x7fELF", b"\xcf\xfa\xed\xfe"])
def test_native_binary_on_path_remains_discoverable(
    tmp_path: Path, isolated_discovery: Path, magic: bytes,
) -> None:
    native = _executable(tmp_path / "bin" / "throwdbd", magic + b"test-native")

    assert _resolve() == native.resolve()


@pytest.mark.parametrize("source", ["option", "environment"])
@pytest.mark.parametrize("kind", ["missing", "launcher"])
def test_invalid_explicit_binary_fails_without_falling_back(
    tmp_path: Path,
    isolated_discovery: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    kind: str,
) -> None:
    _executable(
        isolated_discovery / "build" / "server-release" / "throwdbd",
        b"\x7fELFtest-native",
    )
    configured = tmp_path / "configured" / "throwdbd"
    if kind == "launcher":
        _executable(configured, b"#!/usr/bin/env python3\n")
    if source == "environment":
        monkeypatch.setenv("THROWDB_SERVER_BINARY", str(configured))

    with pytest.raises(
        pytest.fail.Exception, match="configured native server is invalid"
    ):
        _resolve(configured if source == "option" else None)
