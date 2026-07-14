# ThrowDB

ThrowDB is a local, disk-backed store for fixed-width embeddings. The native
collection engine supports arbitrary numeric IDs, bit-exact `float16` or
`float32` NumPy matrices, immutable base segments on local NVMe, a fast flat
in-memory index, durable WAL-backed record CRUD, and immutable mutation-delta
materialization without rewriting the base.

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
    lifecycle = collection.rotate_wal()       # no-op if the WAL is empty
    materialized = collection.materialize_frozen_wals()
    compacted = collection.compact_deltas()   # bounded oldest-prefix merge
    orphan_plan = collection.collect_orphans(
        dry_run=True,
        min_age_seconds=3600,
    )
```

For diagnostics that do not otherwise need an open collection, use
`Collection.inspect_orphans("/data/example", min_age_seconds=3600)`. This
lightweight path validates the selected catalog and scans mutation namespaces
without loading the base index. Both APIs only produce an advisory dry-run
report: they do not delete files, and the report is not authorization for a
later unlink.

`Collection.create()` now builds the checksummed native numeric format reported
as `collection.format_version == 6`. Version 6 adds the required `DeltaV1`
feature to the version-5 checksummed `CURRENT`, immutable catalogs, and
UUID-named active/frozen WAL chain. It can replace a frozen WAL prefix with an
immutable mutation-delta data/index pair. The reader also opens format 5 and
legacy version-3/4 collections; format 5 remains catalog/WAL-only and rejects
materialization. Version 3 uses a full structural index validation as its
checksum-free fallback. UTF-8 IDs remain reserved for the next string index
slice.
Prefix reads are API-correct today but the dense-row backend still
reads each complete row into a temporary before returning its prefix.
`"auto"` never runs a calibration benchmark while opening. It selects the
strongest persisted media/runtime/layout match: `exact` when the storage target
is identifiable, `compatible` when its physical backing is opaque but its
visible performance class is reproducible, and the conservative thread-pool
`fallback` otherwise. The effective `profile_reuse_level`, selection, and
reason are visible through `collection.tuning`. See
[Reader tuning profiles](docs/tuning-profiles.md) for the matching and safety
contract.

Run calibration explicitly after placing or copying a collection onto its
target storage:

```python
report = throwdb.Collection.tune(
    "/data/example",
    max_seconds=300,
    max_read_bytes=8 << 30,
    max_memory_bytes=512 << 20,
)
print(report["status"], report["selection"])
```

The bounded calibrator loads the index once, samples IDs internally, verifies
every candidate bit-for-bit, runs one charged but unmeasured warm-up pass, and
saves only a stable winner. It does not need the original NumPy ID file. Pass
`dry_run=True` (or CLI `--dry-run`) to measure without saving. A stable
OverlayFS access path is reusable at the `compatible` level even though its
local or network backing is hidden. Its report contains
`target.profile_reusable=true`,
`target.profile_reuse_level="compatible"`, and an `identity_warning` describing
that limitation. Visible local-block and CephFS targets use separate exact
identities. Only an incomplete or unclassifiable target remains `fallback` and
requires dry-run because its profile cannot be published.

The MVP performance target is a local SSD/NVMe. Network-filesystem identities
remain isolated from local profiles for correctness and observability, but the
current random-row reader makes no CephFS performance guarantee; a later server
can expose the collection over the network while keeping its payload on local
NVMe.

Compare the same bounded matrix across two or more local storage targets
without publishing any profile:

```bash
python benchmarks/run_storage_matrix.py \
  --target nvme-a=/data/nvme-a/example \
  --target nvme-b=/data/nvme-b/example \
  --results benchmark-results/storage-matrix
```

Targets run sequentially with the same seed. Time, read, and memory limits are
applied independently to each target. The result directory contains each raw
tuning report, optional system-monitor logs, and an aggregate `report.json`.
Each target summary includes `profile_reuse_level`, so an opaque-overlay
compatible result is distinguishable from an exact local identity.

Normal format-6 open verifies the CRC32C persisted for the complete flat-index
body and for every immutable delta-index body. It validates delta version and
reverse-row metadata but deliberately does not rescan potentially large delta
embedding payloads to recompute their CRC. Use
`Collection.open(path, verify_index=True)` to additionally run the much more
expensive base-index slot, `RowRef`, uniqueness, and probe-path audit. Per-phase
durations and the selected validation path are available in
`collection.open_stats` and are copied into load-test `report.json`.
`throwdb inspect --storage /data/example` prints the same diagnostics as JSON;
add `--verify-index` to run the full base-index structural audit. Delta payload
scrubbing is a separate future operation and is not implied by this flag.

`collection.mutation_log_stats` reports the selected catalog generation,
committed/materialized sequence, frozen-WAL count, delta-run count, and
active/total WAL and delta bytes. Lifecycle results additionally report the
selected/reclaimed resources and bytes. Calling `rotate_wal()` seals a non-empty
active WAL and atomically publishes its successor. Calling
`materialize_frozen_wals()` collapses the currently frozen WAL prefix into one
immutable DeltaV1 run, atomically switches `CURRENT`, and then best-effort
unlinks the replaced WAL pathnames. Calling `compact_deltas()` verifies and
merges a bounded oldest contiguous DeltaV1 prefix while preserving original
sequences and tombstones, then best-effort unlinks only the replaced delta
pairs. None of these operations rewrites base embeddings. An empty frozen
prefix or fewer than two eligible delta runs is a successful no-op.

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

The sequential `pread` baseline remains the conservative fallback. A saved
profile may select another backend automatically; pass an explicit mapping to
override it for one open:

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

throwdb tuning show --storage /data/example

throwdb tune \
  --storage /data/example \
  --max-seconds 300 \
  --max-read-gib 8 \
  --max-memory-mib 512 \
  --output tune-report.json

throwdb rotate-wal --storage /data/example

throwdb materialize --storage /data/example

throwdb compact-deltas \
  --storage /data/example \
  --max-runs 8 \
  --max-input-gib 8 \
  --max-input-entries 1000000

throwdb collect-orphans \
  --storage /data/example \
  --dry-run \
  --min-age-seconds 3600
```

`collect-orphans` requires the explicit `--dry-run` acknowledgement. It marks
old temporary publications, unreferenced WALs, and complete or partial
unreferenced delta pairs as eligible candidates after the age threshold.
Reachable files, superseded catalogs retained by policy, and unknown entries
are never candidates. The command does not remove anything.

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
- crash-safe manual WAL rotation through immutable catalogs; automatic
  rotation is currently only the hard safety valve before a WAL would exceed
  the 32-bit `RowRef` row limit;
- manual frozen-WAL-prefix materialization into immutable DeltaV1 `.dat/.idx`
  pairs. The build permits concurrent writes, pauses them for the two short
  catalog publications, preserves newer/active WAL precedence, and never
  rewrites the immutable base;
- manual bounded oldest-prefix DeltaV1 compaction with source-payload
  verification, newest-version collapse, tombstone preservation, atomic
  publication, and best-effort source reclamation;
- public lightweight orphan inspection and a writer-held dry-run collection
  plan with age-based eligibility, but no orphan deletion or automatic
  `maintain()` policy yet;
- no base compaction/selective cleaning, UTF-8 IDs, delta payload scrub command,
  or network service yet. Delta version metadata is still loaded into the RAM
  overlay on open. The lifecycle currently relies on local POSIX rename,
  `fsync`, open-file, and unlink semantics; network filesystems are outside
  this contract.

See [the storage format](docs/storage-format.md),
[the mutable-storage design](docs/mutable-storage.md), and
[the benchmark plan](docs/testing.md) for details. The legacy
`benchmarks/random_read.py` and `throwdb_bench` runners are sequential-baseline
benchmarks: they open `Storage` without enabling the internal executor. For a
complete machine, correctness, sequential-read, and concurrent-reader run, edit
the configuration at the top of `benchmarks/run_load_test.py` and execute it
once. The runner auto-detects fresh native and legacy immutable manifests.
`--cache-mode cold` uses per-file `POSIX_FADV_DONTNEED` on immutable base
segments and every retained mutation WAL for rootless eviction on Linux.
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
