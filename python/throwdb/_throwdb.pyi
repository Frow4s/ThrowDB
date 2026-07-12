from __future__ import annotations

from os import PathLike
from types import TracebackType

import numpy as np
import numpy.typing as npt

__version__: str


class Writer:
    @staticmethod
    def create(
        path: str | bytes | PathLike[str] | PathLike[bytes],
        dimension: int = 1024,
        shard_size_bytes: int = 17_179_869_184,
    ) -> Writer: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    def add(
        self,
        ids: npt.NDArray[np.uint64],
        embeddings: npt.NDArray[np.float16],
    ) -> None: ...

    def finish(self) -> None: ...

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
    ) -> Storage: ...

    @property
    def dimension(self) -> int: ...

    @property
    def count(self) -> int: ...

    def get(
        self, ids: npt.NDArray[np.uint64]
    ) -> npt.NDArray[np.float16]: ...

    def get_into(
        self,
        ids: npt.NDArray[np.uint64],
        output: npt.NDArray[np.float16],
    ) -> None: ...
