"""Fresh-process execution for authoritative reader calibration.

The calibration resource probe uses Linux ``/proc/self/io`` counters.  Those
counters cover every thread in a process, so an application-owned background
reader could otherwise be mistaken for ThrowDB backing-storage I/O.  The
public tuning path therefore executes the complete decision-bearing matrix in
a fresh interpreter and never falls back to an in-process measurement.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from os import PathLike
from typing import Final, TextIO


_PROTOCOL_FORMAT: Final = "throwdb-calibration-worker"
_PROTOCOL_VERSION: Final = 1
_MAX_CONFIG_BYTES: Final = 1 << 20
_MAX_MESSAGE_CHARACTERS: Final = 64 << 20
_MAX_ERROR_CHARACTERS: Final = 64 << 10
_TERMINATE_GRACE_SECONDS: Final = 0.5


class CalibrationProcessError(RuntimeError):
    """The isolated calibration worker failed or violated its protocol."""


def _budget_result(reason: str) -> dict[str, object]:
    return {
        "format": "throwdb-reader-calibration",
        "version": 1,
        "status": "budget_exhausted",
        "reason": reason,
        "candidate_discovery": None,
        "sample": None,
        "workload": None,
        "candidates": [],
        "stability": None,
        "resource_stability": None,
        "storage_stability": None,
        "selection": None,
        "planned_read_bytes": 0,
        "measurement_process": {
            "isolation": "subprocess_exec",
            "status": "terminated_at_deadline",
        },
    }


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # A task stuck in an uninterruptible kernel read will observe SIGKILL
        # as soon as that syscall returns.  Reap it without extending the
        # caller's bounded deadline synchronously.
        threading.Thread(target=process.wait, daemon=True).start()


def _read_messages(
    stream: TextIO,
    messages: queue.Queue[tuple[str, str]],
) -> None:
    try:
        while True:
            line = stream.readline(_MAX_MESSAGE_CHARACTERS + 1)
            if line == "":
                break
            if len(line) > _MAX_MESSAGE_CHARACTERS or not line.endswith("\n"):
                messages.put(
                    (
                        "protocol_error",
                        "calibration worker message exceeded its bound",
                    )
                )
                break
            messages.put(("message", line))
    except BaseException as error:  # pragma: no cover - defensive pipe guard
        messages.put(
            (
                "protocol_error",
                f"cannot read calibration worker output: {error}",
            )
        )
    finally:
        messages.put(("eof", ""))


def _read_stderr(stream: TextIO, output: list[str]) -> None:
    remaining = _MAX_ERROR_CHARACTERS
    try:
        while True:
            value = stream.read(4096)
            if not value:
                break
            if remaining > 0:
                retained = value[:remaining]
                output.append(retained)
                remaining -= len(retained)
    except BaseException:  # pragma: no cover - diagnostic path only
        return


def _worker_error(
    message: Mapping[str, object],
    stderr: str,
) -> CalibrationProcessError:
    kind = str(message.get("error_type", "worker_error"))
    detail = str(message.get("message", "isolated calibration failed"))
    trace = str(message.get("traceback", ""))
    diagnostics = trace or stderr
    suffix = f"\n{diagnostics[:_MAX_ERROR_CHARACTERS]}" if diagnostics else ""
    return CalibrationProcessError(f"{kind}: {detail}{suffix}")


def run_isolated_calibration(
    path: str | bytes | PathLike[str] | PathLike[bytes],
    *,
    record_size: int,
    current_options: Mapping[str, object] | None,
    seed: int,
    max_read_bytes: int,
    max_memory_bytes: int,
    deadline: float,
    expected_dimension: int,
    expected_dtype: str,
    expected_format_version: int,
    expected_root_identity: tuple[int, int],
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run the full calibration matrix in a fresh interpreter.

    There is intentionally no in-process fallback: without the process
    boundary, block I/O from the embedding application could certify a warm
    matrix and publish the wrong reusable profile.
    """

    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        return _budget_result(
            "cooperative deadline expired before starting the isolated worker"
        )
    if not hasattr(os, "fchdir") or not hasattr(os, "O_DIRECTORY"):
        raise CalibrationProcessError(
            "isolated calibration requires a POSIX directory descriptor"
        )
    native_path = os.fsencode(os.fspath(path))
    open_flags = os.O_RDONLY | os.O_DIRECTORY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open(native_path, open_flags)
    except OSError as error:
        raise CalibrationProcessError(
            f"cannot anchor the collection root: {error}"
        ) from error
    try:
        root_status = os.fstat(root_fd)
        observed_root_identity = (
            int(root_status.st_dev),
            int(root_status.st_ino),
        )
        if observed_root_identity != expected_root_identity:
            raise CalibrationProcessError(
                "collection root changed before starting the isolated worker"
            )
        if clock() >= deadline:
            return _budget_result(
                "cooperative deadline expired while anchoring the collection root"
            )
        configuration = {
            "format": _PROTOCOL_FORMAT,
            "version": _PROTOCOL_VERSION,
            "root_fd": root_fd,
            "expected_root_device": expected_root_identity[0],
            "expected_root_inode": expected_root_identity[1],
            "record_size": int(record_size),
            "current_options": (
                dict(current_options) if current_options is not None else None
            ),
            "seed": int(seed),
            "max_read_bytes": int(max_read_bytes),
            "max_memory_bytes": int(max_memory_bytes),
            "deadline": float(deadline),
            "expected_dimension": int(expected_dimension),
            "expected_dtype": str(expected_dtype),
            "expected_format_version": int(expected_format_version),
        }
        encoded = json.dumps(
            configuration,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > _MAX_CONFIG_BYTES:
            raise CalibrationProcessError(
                "isolated calibration configuration exceeded its bound"
            )
        if clock() >= deadline:
            return _budget_result(
                "cooperative deadline expired before spawning the isolated worker"
            )
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                "throwdb._calibration_process",
                "--worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            close_fds=True,
            pass_fds=(root_fd,),
        )
    finally:
        os.close(root_fd)
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _stop_process(process)
        raise CalibrationProcessError("cannot create calibration worker pipes")

    messages: queue.Queue[tuple[str, str]] = queue.Queue()
    stderr_parts: list[str] = []
    stdout_thread = threading.Thread(
        target=_read_messages,
        args=(process.stdout, messages),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_parts),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        process.stdin.write(encoded + "\n")
        process.stdin.flush()
        process.stdin.close()
    except BaseException:
        _stop_process(process)
        raise

    result: dict[str, object] | None = None
    worker_error: Mapping[str, object] | None = None
    protocol_error: str | None = None
    saw_eof = False
    try:
        while result is None and worker_error is None and protocol_error is None:
            remaining = deadline - clock()
            if remaining <= 0:
                _stop_process(process)
                return _budget_result(
                    "cooperative deadline expired in the isolated worker"
                )
            try:
                kind, raw = messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if process.poll() is not None and saw_eof:
                    break
                continue
            if kind == "eof":
                saw_eof = True
                if process.poll() is not None:
                    break
                continue
            if kind == "protocol_error":
                protocol_error = raw
                break
            try:
                message = json.loads(raw)
            except (TypeError, ValueError) as error:
                protocol_error = f"invalid calibration worker JSON: {error}"
                break
            if not isinstance(message, Mapping):
                protocol_error = "calibration worker message is not an object"
                break
            message_type = message.get("type")
            if message_type == "progress":
                value = str(message.get("message", ""))[:1024]
                if progress is not None:
                    progress(value)
            elif message_type == "result":
                payload = message.get("payload")
                if not isinstance(payload, dict):
                    protocol_error = "calibration worker result is not an object"
                else:
                    result = payload
            elif message_type == "error":
                worker_error = message
            else:
                protocol_error = "calibration worker sent an unknown message"

        if result is not None or worker_error is not None:
            try:
                process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                protocol_error = "calibration worker did not exit after its result"
        elif process.poll() is None:
            _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise

    stderr_thread.join(timeout=0.1)
    stderr = "".join(stderr_parts)
    if protocol_error is not None:
        _stop_process(process)
        raise CalibrationProcessError(
            f"{protocol_error}{(': ' + stderr) if stderr else ''}"
        )
    if worker_error is not None:
        raise _worker_error(worker_error, stderr)
    if result is None:
        return_code = process.poll()
        raise CalibrationProcessError(
            "isolated calibration worker exited without a result "
            f"(status={return_code}){(': ' + stderr) if stderr else ''}"
        )
    if process.returncode != 0:
        raise CalibrationProcessError(
            "isolated calibration worker returned a result with non-zero "
            f"status {process.returncode}{(': ' + stderr) if stderr else ''}"
        )
    if (
        result.get("format") != "throwdb-reader-calibration"
        or result.get("version") != 1
    ):
        raise CalibrationProcessError(
            "isolated calibration worker returned an unsupported report"
        )
    reported_process = result.get("measurement_process")
    reported_pid = (
        reported_process.get("pid")
        if isinstance(reported_process, Mapping)
        else None
    )
    if (
        isinstance(reported_pid, bool)
        or not isinstance(reported_pid, int)
        or reported_pid != process.pid
    ):
        raise CalibrationProcessError(
            "isolated calibration worker returned an invalid process identity"
        )
    result["measurement_process"] = {
        "isolation": "subprocess_exec",
        "status": "complete",
        "pid": process.pid,
        "parent_verified": True,
    }
    storage_value = result.get("storage_stability")
    if isinstance(storage_value, Mapping):
        storage = dict(storage_value)
        worker_isolation = storage.get("process_isolation")
        worker_isolation_pid = (
            worker_isolation.get("pid")
            if isinstance(worker_isolation, Mapping)
            else None
        )
        if (
            storage.get("telemetry_scope")
            != "isolated_calibration_process"
            or not isinstance(worker_isolation, Mapping)
            or worker_isolation.get("passed") is not True
            or worker_isolation.get("method") != "subprocess_exec"
            or isinstance(worker_isolation_pid, bool)
            or not isinstance(worker_isolation_pid, int)
            or worker_isolation_pid != process.pid
        ):
            raise CalibrationProcessError(
                "isolated calibration worker returned invalid storage "
                "attribution"
            )
        storage["process_isolation"] = {
            "passed": True,
            "method": "subprocess_exec",
            "pid": process.pid,
            "parent_verified": True,
        }
        result["storage_stability"] = storage
    return result


def _emit(message: Mapping[str, object]) -> None:
    encoded = json.dumps(
        message,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if len(encoded) > _MAX_MESSAGE_CHARACTERS:
        encoded = json.dumps(
            {
                "type": "error",
                "error_type": "CalibrationProcessError",
                "message": "calibration worker report exceeded its bound",
            },
            separators=(",", ":"),
        )
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _worker_main() -> int:
    try:
        raw = sys.stdin.buffer.readline(_MAX_CONFIG_BYTES + 1)
        if len(raw) > _MAX_CONFIG_BYTES or not raw.endswith(b"\n"):
            raise CalibrationProcessError(
                "calibration worker configuration exceeded its bound"
            )
        configuration = json.loads(raw)
        if (
            not isinstance(configuration, Mapping)
            or configuration.get("format") != _PROTOCOL_FORMAT
            or configuration.get("version") != _PROTOCOL_VERSION
        ):
            raise CalibrationProcessError(
                "unsupported calibration worker configuration"
            )

        from ._calibration import run_bounded_calibration
        from ._calibration_cache import prepare_cold_payload_cache
        from ._throwdb import NativeCollection

        root_fd = int(configuration["root_fd"])
        root_status = os.fstat(root_fd)
        if (
            int(root_status.st_dev)
            != int(configuration["expected_root_device"])
            or int(root_status.st_ino)
            != int(configuration["expected_root_inode"])
        ):
            raise CalibrationProcessError(
                "inherited collection descriptor has the wrong identity"
            )
        os.fchdir(root_fd)
        os.close(root_fd)
        native_path = b"."
        deadline = float(configuration["deadline"])
        # The parent owns the exclusive calibration/writer lease.  The worker
        # deliberately opens a normal read-only snapshot: requesting a second
        # calibration lease would correctly fail against the parent lock.
        native = NativeCollection.open(native_path, mode="r")
        try:
            if (
                int(native.dimension)
                != int(configuration["expected_dimension"])
                or str(native.dtype) != str(configuration["expected_dtype"])
                or int(native.format_version)
                != int(configuration["expected_format_version"])
            ):
                raise CalibrationProcessError(
                    "collection schema changed before isolated calibration"
                )

            def progress(message: str) -> None:
                _emit({"type": "progress", "message": str(message)[:1024]})

            measured = run_bounded_calibration(
                native,
                record_size=int(configuration["record_size"]),
                current_options=configuration.get("current_options"),
                seed=int(configuration["seed"]),
                max_read_bytes=int(configuration["max_read_bytes"]),
                max_memory_bytes=int(configuration["max_memory_bytes"]),
                deadline=deadline,
                progress=progress,
                cache_preparer=lambda: prepare_cold_payload_cache(
                    native_path,
                    deadline=deadline,
                    clock=time.perf_counter,
                ),
                isolated_process=True,
            )
        finally:
            native.close()

        storage_value = measured.get("storage_stability")
        if isinstance(storage_value, Mapping):
            storage = dict(storage_value)
            storage["telemetry_scope"] = "isolated_calibration_process"
            storage["process_isolation"] = {
                "passed": True,
                "method": "subprocess_exec",
                "pid": os.getpid(),
            }
            measured["storage_stability"] = storage
        measured["measurement_process"] = {
            "isolation": "subprocess_exec",
            "status": "complete",
            "pid": os.getpid(),
        }
        _emit({"type": "result", "payload": measured})
        return 0
    except BaseException as error:
        _emit(
            {
                "type": "error",
                "error_type": type(error).__name__,
                "message": str(error)[:4096],
                "traceback": traceback.format_exc(limit=12)[
                    -_MAX_ERROR_CHARACTERS:
                ],
            }
        )
        return 1


def main() -> int:
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("this module is an internal calibration worker")
    return _worker_main()


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "CalibrationProcessError",
    "run_isolated_calibration",
]
