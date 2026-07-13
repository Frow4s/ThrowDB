# ThrowDB

ThrowDB is a local, disk-backed store for fixed-width embeddings. The native
collection engine supports arbitrary numeric IDs, bit-exact `float16` or
`float32` NumPy matrices, immutable base segments on local NVMe, a fast flat
in-memory index, and durable WAL-backed record CRUD.

The fresh mutable format is documented in
[`docs/native-format-v3.md`](docs/native-format-v3.md). The earlier immutable
format-v1 contract is documented in
[`docs/storage-format.md`](docs/storage-format.md). The target local collection
semantics and staged product plan are documented in
[`docs/collection-contract.md`](docs/collection-contract.md) and
[`docs/product-roadmap.md`](docs/product-roadmap.md).
The numeric version-2 binary representation is specified in
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

`Collection` is the user-facing API. It accepts every NumPy unsigned integer
width, non-negative signed integer arrays, and ordered Python integer
sequences. Numeric IDs are checked exactly and normalized to the canonical
`uint64` domain; an already native, aligned, C-contiguous `uint64` array stays
zero-copy.

```python
import numpy as np
import throwdb

ids = np.load("ids.npy", mmap_mode="r")          # uint32 is accepted
vectors = np.load("embeddings.npy", mmap_mode="r")

schema = throwdb.CollectionSchema(
    id_type="uint64",
    dimension=1024,
    dtype="float32",       # float16 is also supported
)
with throwdb.Collection.create("/data/example", schema=schema) as collection:
    collection.add(ids, vectors)

collection = throwdb.Collection.open("/data/example", tuning="auto")
result = collection.get([42, 100])
prefix = collection.get([42, 100], prefix_dimensions=128)

output = np.empty((2, 128), dtype=np.float32)
collection.get_into([42, 100], output, prefix_dimensions=128)

with throwdb.Collection.open("/data/example", mode="r+") as collection:
    collection.insert([200], vectors[:1])
    collection.update([42], vectors[1:2])
    collection.upsert([100, 300], vectors[:2])
    deleted = collection.delete([200, 999])  # [True, False]
```

`Collection.create()` now builds the checksummed native numeric format reported
as `collection.format_version == 4`. The current reader also opens legacy
checksum-free version-3 collections, using a full structural index validation
as its safe fallback. UTF-8 IDs remain reserved for the next string index slice.
Prefix reads are API-correct today but the dense-row backend still
reads each complete row into a temporary before returning its prefix.
The initial `"auto"` tuning profile deliberately selects the conservative
thread-pool baseline without running a calibration benchmark; the effective
selection and reason are visible through `collection.tuning`.

Normal version-4 open verifies the CRC32C persisted for the complete flat-index
body. Use `Collection.open(path, verify_index=True)` to additionally run the
much more expensive slot, `RowRef`, uniqueness, and probe-path audit. Per-phase
durations and the selected validation path are available in
`collection.open_stats` and are copied into load-test `report.json`.
`throwdb inspect --storage /data/example` prints the same diagnostics as JSON;
add `--verify-index` to use that command as a full scrub-on-open check.

`Writer` and `Storage` remain the legacy immutable low-level API used by the
existing benchmark suite. They require native, aligned, C-contiguous `uint64`
IDs and embedding arrays whose dtype exactly matches the writer (`float16` or
`float32`). They do not open the fresh mutable format; new applications should
use `Collection`.

```python
with throwdb.Writer.create(
    "/data/example-low-level", dimension=1024, dtype="float32"
) as writer:
    writer.add(ids.astype(np.uint64, copy=False), vectors)

db = throwdb.Storage.open("/data/example-low-level")
result = db.get(np.array([42, 100], dtype=np.uint64))
```

The sequential `pread` baseline remains the default. Enable the bounded,
per-storage parallel backend explicitly:

```python
db = throwdb.Collection.open(
    "/data/example",
    tuning={
        "io_workers": 16,
        "io_queue_capacity": 32,
        "io_ranges_per_task": 8,
    },
)
```

Multi-range requests from all simultaneous callers of one collection handle
share the same executor and queue, so their aggregate I/O concurrency is
bounded rather than multiplied per request. A zero queue capacity with positive workers selects
`2 * workers`. Each executor task processes up to `io_ranges_per_task` physical
ranges; grouping ranges reduces queue and wake-up overhead for page-cache hits.
A request whose planner produces exactly one physical range uses a synchronous
direct-read fast path. This also applies to concurrent callers: those one-range
reads bypass the executor and queue and are not included in the global
multi-range concurrency bound.

An experimental Linux-only backend drives buffered reads through the raw
io_uring kernel ABI and does not require `liburing`:

```python
db = throwdb.Collection.open(
    "/data/example",
    tuning={
        "io_backend": "io_uring",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 32,
    },
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
output = np.empty(
    (len(query_ids), db.dimension), dtype=np.dtype(db.dtype)
)
db.get_into(query_ids, output)
```

For an occasional phase-level diagnosis, use the opt-in profiled variant. It
performs the same bit-exact full-row read and returns native timings plus the
physical plan shape; ordinary `get()` and `get_into()` calls do not read clocks
or update telemetry counters.

```python
stats = db.get_into_profiled(query_ids, output)
print(
    stats["lookup_seconds"],
    stats["physical_plan_seconds"],
    stats["physical_backend_seconds"],
    stats["planned_ranges"],
    stats["planned_read_bytes"],
    stats["zero_delta_fast_path"],
)
```

`physical_backend_seconds` currently combines queue/admission wait, physical
I/O, and scatter copies because those operations overlap in parallel backends.
The value is intended for diagnosis, not for instrumentation of every
production request.

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

`ids.npy` may use any 8/16/32/64-bit unsigned integer dtype, or a signed integer
dtype whose values are all non-negative. `embeddings.npy` must use `float16` or
`float32` and have shape `(N, dimension)`. The CLI opens both with NumPy memory
mapping, normalizes IDs safely, and sends them to C++ in bounded batches.

After building a rebuildable test collection, exercise durable CRUD, reopen
recovery, prefix reads, and the writer lease with a small reversible sample:

```bash
python benchmarks/run_mutable_test.py \
  --store /data/example \
  --ids ids.npy \
  --source-embeddings embeddings.npy \
  --sample-rows 64 \
  --results benchmark-results/mutable-smoke.json
```

The command restores the sampled logical rows and removes synthetic IDs when it
finishes normally, but the test transactions remain in WAL history. Do not run
it on the only copy of data that cannot be rebuilt; `SIGKILL` or machine failure
can interrupt cleanup.

## Current MVP boundaries

- atomic bulk creation followed by WAL-backed `insert`, `update`, `upsert`,
  and tombstone `delete`;
- local filesystem/NVMe only;
- one `float16` or `float32` dtype and one fixed dimension per collection;
- canonical `uint64` storage IDs, with safe Python normalization from all
  standard signed/unsigned integer widths;
- one read-write process lease; read-only and read-write handles are snapshots
  captured when opened and must be reopened to observe another process's later
  commits;
- buffered `pread` and experimental Linux `io_uring` backends;
- no WAL rotation/selective cleaning, UTF-8 IDs, payload scrubbing, or network
  service yet. Until rotation lands, mutation history grows in `wal/active.wal`
  and is replayed when the collection opens.

See [the storage format](docs/storage-format.md),
[the mutable-storage design](docs/mutable-storage.md), and
[the benchmark plan](docs/testing.md) for details. The legacy
`benchmarks/random_read.py` and `throwdb_bench` runners are sequential-baseline
benchmarks: they open `Storage` without enabling the internal executor. For a
complete machine, correctness, sequential-read, and concurrent-reader run, edit
the configuration at the top of `benchmarks/run_load_test.py` and execute it
once. The runner auto-detects fresh native and legacy immutable manifests.
`--cache-mode cold` uses per-file `POSIX_FADV_DONTNEED` on immutable base
segments and the active mutation WAL for rootless eviction on Linux.
Standalone `warm` replays the exact measured query set outside the timed
interval before measuring it; `both` measures the same deterministic stream
cold first and warm immediately afterwards. A result's
`explicit_measurement_prewarm` is true only when that standalone replay ran.
Use `--io-ranges-per-task` to record and compare executor task granularity; see
the benchmark plan for the remote `1/4/8/16` sweep. On cgroup v2 systems the
report also records the effective CPU quota and per-scenario `cpu.stat` deltas,
including throttled periods and time.

The on-disk format is versioned but still experimental. Format-v1 compatibility
is covered by a byte-level golden test; format v2 may still evolve before its
first stable declaration.
