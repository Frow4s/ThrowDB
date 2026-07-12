# ThrowDB

ThrowDB is a read-optimized, disk-backed store for fixed-width embeddings. The
MVP targets arbitrary `uint64` IDs and `float16` NumPy matrices, with immutable
shards on local NVMe and an in-memory immutable hash index.

The implemented format-v1 contract is documented in
[`docs/storage-format.md`](docs/storage-format.md). The target local collection
semantics and staged product plan are documented in
[`docs/collection-contract.md`](docs/collection-contract.md) and
[`docs/product-roadmap.md`](docs/product-roadmap.md).
The proposed version-2 binary representation is specified in
[`docs/storage-format-v2.md`](docs/storage-format-v2.md).

## Install for development

ThrowDB requires a C++20 compiler, CMake, Python 3.10+, NumPy and pybind11.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
pytest -q
```

## Python API

The Python API accepts C-contiguous, native-endian NumPy arrays. Input buffers
are passed directly to the synchronous C++ call and may be reused as soon as
the call returns.

```python
import numpy as np
import throwdb

ids = np.load("ids.npy", mmap_mode="r")
vectors = np.load("embeddings.npy", mmap_mode="r")

with throwdb.Writer.create("/data/example", dimension=1024) as writer:
    writer.add(ids, vectors)

db = throwdb.Storage.open("/data/example")
result = db.get(np.array([42, 100], dtype=np.uint64))
```

The sequential `pread` baseline remains the default. Enable the bounded,
per-storage parallel backend explicitly:

```python
db = throwdb.Storage.open(
    "/data/example",
    io_workers=16,
    io_queue_capacity=32,
    io_ranges_per_task=8,
)
```

Multi-range requests from all simultaneous callers of this `Storage` share the
same executor and queue, so their aggregate I/O concurrency is bounded rather
than multiplied per request. A zero queue capacity with positive workers selects
`2 * workers`. Each executor task processes up to `io_ranges_per_task` physical
ranges; grouping ranges reduces queue and wake-up overhead for page-cache hits.
A request whose planner produces exactly one physical range uses a synchronous
direct-read fast path. This also applies to concurrent callers: those one-range
reads bypass the executor and queue and are not included in the global
multi-range concurrency bound.

An experimental Linux-only backend drives buffered reads through the raw
io_uring kernel ABI and does not require `liburing`:

```python
db = throwdb.Storage.open(
    "/data/example",
    io_backend="io_uring",
    io_uring_workers=2,
    io_uring_queue_depth=32,
)
```

`io_uring_workers=1` preserves the original caller-driven, mutex-serialized
ring. Two or more workers create independent dedicated ring-owner threads and
partition every multi-range call across them. `io_uring_queue_depth` is the
total backend depth, so two workers at depth 32 receive 16 entries each. The
default remains one worker until the multi-ring experiment is benchmarked.
Multi-ring admission allows at most `2 * io_uring_workers` synchronous batches
inside the worker queues; additional callers apply FIFO ticket-based
backpressure before enqueue, so a fast caller cannot repeatedly bypass waiters.
Requests planned as one range retain the direct `pread` fast path. `io_workers`
and `io_queue_capacity` must both be zero in io_uring mode.

Large inputs should be appended in batches; `finish()` writes the immutable
hash index and publishes `manifest.bin` only after the shards and index are
complete. Duplicate build IDs are rejected. A missing query ID raises
`KeyError` before the output buffer is modified.

To avoid allocating the result in C++, use a caller-owned array:

```python
query_ids = np.array([42, 100], dtype=np.uint64)
output = np.empty((len(query_ids), db.dimension), dtype=np.float16)
db.get_into(query_ids, output)
```

## CLI

```bash
throwdb create \
  --storage /data/example \
  --ids ids.npy \
  --embeddings embeddings.npy

throwdb get \
  --storage /data/example \
  --ids query_ids.npy \
  --output result.npy

throwdb inspect --storage /data/example
```

`ids.npy` must have dtype `uint64` and shape `(N,)`. `embeddings.npy` must have
dtype `float16` and shape `(N, dimension)`. The CLI opens both with NumPy memory
mapping and sends them to C++ in bounded batches.

## Current MVP boundaries

- immutable create/open lifecycle;
- local filesystem/NVMe only;
- one `float16` dtype and one fixed dimension per store;
- buffered `pread` backend;
- no update, delete, compaction, checksums or network service yet.

See [the storage format](docs/storage-format.md) and
[the benchmark plan](docs/testing.md) for details. The legacy
`benchmarks/random_read.py` and `throwdb_bench` runners are sequential-baseline
benchmarks: they open `Storage` without enabling the internal executor. For a
complete machine, correctness, sequential-read, and concurrent-reader run, edit
the configuration at the top of `benchmarks/run_load_test.py` and execute it
once. Its
`--cache-mode cold|warm|both` option uses per-file `POSIX_FADV_DONTNEED` for
rootless paired cache measurements on Linux. Use `--io-ranges-per-task` to record
and compare executor task granularity; see the benchmark plan for the remote
`1/4/8/16` sweep. On cgroup v2 systems the report also records the effective CPU
quota and per-scenario `cpu.stat` deltas, including throttled periods and time.

The on-disk format is versioned but still experimental. Do not treat storage
created by the MVP as forward-compatible until format version 1 is declared
stable.
