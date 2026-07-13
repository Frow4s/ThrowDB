from __future__ import annotations

from os import PathLike
from types import TracebackType
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__version__: str

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


class NativeCollection:
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

    def close(self) -> None: ...
