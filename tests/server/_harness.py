from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType


class ServerStartError(RuntimeError):
    pass


@dataclass(slots=True)
class ServerProcess:
    """Own a ``throwdbd`` subprocess and its machine-readable ready event."""

    process: subprocess.Popen[str]
    endpoint: str
    writable: bool
    readiness: dict[str, object]
    security: dict[str, object]
    mutation_safety: dict[str, object]
    maintenance: dict[str, object]
    reader: dict[str, object]
    tuning: dict[str, object]
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    _stdout_thread: threading.Thread | None = None
    _stderr_thread: threading.Thread | None = None

    @classmethod
    def start(
        cls,
        executable: Path,
        storage: Path,
        *,
        timeout: float = 15.0,
        extra_args: tuple[str, ...] = (),
        command_prefix: tuple[str, ...] = (),
    ) -> ServerProcess:
        command = [
            str(executable),
            *command_prefix,
            "--storage",
            str(storage),
            "--listen",
            "127.0.0.1:0",
            *extra_args,
        ]
        process = subprocess.Popen(
            command,
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
        ready_lines: queue.Queue[str | None] = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def read_stdout() -> None:
            try:
                for line in process.stdout:  # pragma: no branch - OS pipe loop
                    stripped = line.rstrip("\r\n")
                    stdout_lines.append(stripped)
                    ready_lines.put(stripped)
            finally:
                ready_lines.put(None)

        def read_stderr() -> None:
            for line in process.stderr:  # pragma: no branch - OS pipe loop
                stderr_lines.append(line.rstrip("\r\n"))

        stdout_thread = threading.Thread(
            target=read_stdout,
            name="throwdbd-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_stderr,
            name="throwdbd-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        deadline = time.monotonic() + timeout
        ready: dict[str, object] | None = None
        failure: str | None = None
        while ready is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = f"server did not become ready in {timeout:.1f}s"
                break
            try:
                line = ready_lines.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if process.poll() is not None:
                    failure = f"server exited with code {process.returncode}"
                    break
                continue
            if line is None:
                failure = f"server closed stdout with code {process.poll()}"
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # stdout is reserved for machine-readable events, but keeping
                # the complete line in diagnostics makes an accidental log
                # immediately actionable.
                continue
            if isinstance(event, dict) and event.get("event") == "ready":
                ready = event

        if ready is None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            diagnostics = "\n".join(
                [
                    failure or "unknown startup failure",
                    "stdout:",
                    *stdout_lines,
                    "stderr:",
                    *stderr_lines,
                ]
            )
            raise ServerStartError(diagnostics)

        if ready.get("protocol") != "grpc/throwdb.v1":
            process.terminate()
            process.wait(timeout=2)
            raise ServerStartError(
                f"server announced unsupported protocol {ready.get('protocol')!r}"
            )
        endpoint = ready.get("endpoint")
        writable = ready.get("writable")
        readiness = ready.get("readiness")
        security = ready.get("security")
        mutation_safety = ready.get("mutation_safety")
        maintenance = ready.get("maintenance")
        reader_options = ready.get("reader")
        tuning = ready.get("tuning")
        if (
            not isinstance(endpoint, str)
            or not isinstance(writable, bool)
            or not isinstance(readiness, dict)
            or not isinstance(security, dict)
            or not isinstance(mutation_safety, dict)
            or not isinstance(maintenance, dict)
            or not isinstance(reader_options, dict)
            or not isinstance(tuning, dict)
        ):
            process.terminate()
            process.wait(timeout=2)
            raise ServerStartError(
                "server ready event is missing endpoint/access/security/"
                "mutation/reader/tuning diagnostics"
            )
        return cls(
            process=process,
            endpoint=endpoint,
            writable=writable,
            readiness=readiness,
            security=security,
            mutation_safety=mutation_safety,
            maintenance=maintenance,
            reader=reader_options,
            tuning=tuning,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            _stdout_thread=stdout_thread,
            _stderr_thread=stderr_thread,
        )

    @property
    def diagnostics(self) -> str:
        return "\n".join(
            [
                f"pid={self.process.pid} returncode={self.process.poll()}",
                "stdout:",
                *self.stdout_lines,
                "stderr:",
                *self.stderr_lines,
            ]
        )

    def close(self, *, timeout: float = 10.0) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)

    def __enter__(self) -> ServerProcess:
        if self.process.poll() is not None:
            raise RuntimeError("throwdbd exited before entering the test context")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


__all__ = ["ServerProcess", "ServerStartError"]
