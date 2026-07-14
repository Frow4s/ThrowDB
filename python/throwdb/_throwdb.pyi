from __future__ import annotations

from os import PathLike
from types import TracebackType
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__version__: str


class ReaderBackendError(RuntimeError): ...


_DTypeName: TypeAlias = Literal["float16", "float32"]
_EmbeddingArray: TypeAlias = (
    npt.NDArray[np.float16] | npt.NDArray[np.float32]
)


class Writer:
    @staticmethod
    def create(
        path: str | bytes | PathLike[str] | PathLike[bytes],
        dimension: int = 1024,
        shard_size_bytes: int = 17_179_869_184,
        dtype: _DTypeName = "float16",
    ) -> Writer: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    @property
    def dtype(self) -> _DTypeName: ...

    @property
    def format_version(self) -> int: ...

    def add(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: _EmbeddingArray,
    ) -> None: ...

    def finish(self) -> None: ...

    def abort(self) -> None: ...

    def __enter__(self) -> Writer: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...


class Storage:
    @staticmethod
    def open(
        path: str | bytes | PathLike[str] | PathLike[bytes],
        io_workers: int = 0,
        io_queue_capacity: int = 0,
        io_ranges_per_task: int = 8,
        io_backend: str = "thread_pool",
        io_uring_queue_depth: int = 32,
        io_uring_workers: int = 1,
        max_read_range_bytes: int = 1_048_576,
        max_gap_bytes: int = 0,
    ) -> Storage: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    @property
    def dtype(self) -> _DTypeName: ...

    @property
    def format_version(self) -> int: ...

    def get(self, ids: npt.NDArray[np.uint64]) -> _EmbeddingArray: ...

    def get_into(
        self,
        ids: npt.NDArray[np.uint64],
        output: _EmbeddingArray,
    ) -> None: ...


class NativeCollectionBuilder:
    @staticmethod
    def create(
        path: str | bytes | PathLike[str] | PathLike[bytes],
        dimension: int = 1024,
        shard_size_bytes: int = 17_179_869_184,
        dtype: _DTypeName = "float16",
    ) -> NativeCollectionBuilder: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    @property
    def dtype(self) -> _DTypeName: ...

    @property
    def format_version(self) -> int: ...

    def add(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: _EmbeddingArray,
    ) -> None: ...

    def finish(self) -> None: ...

    def abort(self) -> None: ...


class NativeReaderTrial:
    def get_into(
        self,
        ids: npt.NDArray[np.uint64],
        output: _EmbeddingArray,
    ) -> None: ...

    def get_into_profiled(
        self,
        ids: npt.NDArray[np.uint64],
        output: _EmbeddingArray,
    ) -> dict[str, object]: ...

    def close(self) -> None: ...


class NativeCollection:
    @staticmethod
    def reader_capabilities(
        io_uring_queue_depth: int = 32,
        io_uring_workers: int = 1,
    ) -> dict[str, object]: ...

    @staticmethod
    def inspect_metadata(
        path: str | bytes | PathLike[str] | PathLike[bytes],
    ) -> dict[str, object]: ...

    @staticmethod
    def inspect_orphans(
        path: str | PathLike[str],
    ) -> dict[str, object]: ...

    @staticmethod
    def open(
        path: str | bytes | PathLike[str] | PathLike[bytes],
        mode: Literal["r", "r+"] = "r",
        io_workers: int = 0,
        io_queue_capacity: int = 0,
        io_ranges_per_task: int = 8,
        io_backend: str = "thread_pool",
        io_uring_queue_depth: int = 32,
        io_uring_workers: int = 1,
        verify_native_index: bool = False,
        max_read_range_bytes: int = 1_048_576,
        max_gap_bytes: int = 0,
        calibration: bool = False,
    ) -> NativeCollection: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    @property
    def dtype(self) -> _DTypeName: ...

    @property
    def format_version(self) -> int: ...

    @property
    def open_stats(self) -> dict[str, object]: ...

    @property
    def mutation_log_stats(self) -> dict[str, object]: ...

    def collect_orphans(self) -> dict[str, object]: ...

    def get(self, ids: npt.NDArray[np.uint64]) -> _EmbeddingArray: ...

    def get_into(
        self,
        ids: npt.NDArray[np.uint64],
        output: _EmbeddingArray,
    ) -> None: ...

    def get_into_profiled(
        self,
        ids: npt.NDArray[np.uint64],
        output: _EmbeddingArray,
    ) -> dict[str, object]: ...

    def reader_trial(
        self,
        io_workers: int = 0,
        io_queue_capacity: int = 0,
        io_ranges_per_task: int = 8,
        io_backend: str = "thread_pool",
        io_uring_queue_depth: int = 32,
        io_uring_workers: int = 1,
        max_read_range_bytes: int = 1_048_576,
        max_gap_bytes: int = 0,
    ) -> NativeReaderTrial: ...

    def sample_ids(
        self,
        requested_count: int,
        seed: int,
        max_inspected_slots: int,
    ) -> dict[str, object]: ...

    def insert(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: _EmbeddingArray,
    ) -> None: ...

    def update(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: _EmbeddingArray,
    ) -> None: ...

    def upsert(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: _EmbeddingArray,
    ) -> None: ...

    def delete(
        self,
        ids: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.bool_]: ...

    def rotate_wal(self) -> dict[str, object]: ...
    def materialize_frozen_wals(self) -> dict[str, object]: ...
    def compact_delta_runs(
        self,
        max_runs: int = 8,
        max_input_bytes: int = 8_589_934_592,
        max_input_entries: int = 1_000_000,
    ) -> dict[str, object]: ...

    def close(self) -> None: ...
