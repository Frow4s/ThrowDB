# ThrowDB

ThrowDB is a local, disk-backed store for fixed-width embeddings. The native
collection engine supports arbitrary numeric IDs, bit-exact `float16` or
`float32` NumPy matrices, immutable segments on local NVMe, durable WAL-backed
record CRUD, and bounded mutation maintenance. Format 7 adds an authoritative
partitioned `ID -> RowRef` index, immutable mutation runs, selective partition
folding, and selective segment cleaning without periodically copying the whole
collection.

The authoritative format-7 lifecycle is documented in
[`docs/mutable-storage.md`](docs/mutable-storage.md); formats 3 through 6 are in
[`docs/native-format-v3.md`](docs/native-format-v3.md). The earlier immutable
format-v1 contract is documented in
[`docs/storage-format.md`](docs/storage-format.md). The target local collection
semantics and staged product plan are documented in
[`docs/collection-contract.md`](docs/collection-contract.md) and
[`docs/product-roadmap.md`](docs/product-roadmap.md).
The numeric version-2 binary representation is specified in
[`docs/storage-format-v2.md`](docs/storage-format-v2.md).

For a complete product handoff, start with the
[repository review](docs/repository-review.md),
[end-user guide](docs/user-guide.md),
[Kubernetes runbook](docs/kubernetes-deployment.md),
[wheel/release guide](docs/release-packaging.md), or the
[documentation index](docs/README.md).

## Install

Release CI is configured to build precompiled CPython 3.10-3.14 wheels for
Linux x86_64 on the `manylinux_2_28` baseline. Each official wheel contains
the embedded C++ extension, the synchronous Python gRPC client, a tune-aware
`throwdbd` launcher, and its private native C++ daemon. The launcher resolves
startup settings once and then replaces itself with the daemon, so Python is
not in the RPC data path. Install a downloaded artifact without a compiler:

```bash
python -m pip install throwdb-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
throwdb --help
throwdbd --help
```

There is currently no PyPI publication, GitHub Release, tag, or checked-in
Linux wheel for this reviewed revision. The only locally verified ready wheel
is a temporary CPython 3.10/macOS ARM64 artifact and cannot be used in Linux or
Kubernetes. Trigger/download the `Linux wheels` Actions artifact or build the
manylinux wheel before deployment; exact locations and commands are in
[`docs/release-packaging.md`](docs/release-packaging.md).

For a complete source build without root privileges, first build the pinned
gRPC/Protobuf toolchain once. The exported ThrowDB-specific prefix is consumed
directly by both the bootstrap and the CMake build launched by pip:

```bash
export THROWDB_GRPC_PREFIX="$HOME/.local/throwdb/grpc-1.81.0"
./tools/bootstrap_grpc.sh
python -m pip install .
```

For an editable complete-server checkout, use the same exported prefix:

```bash
python -m pip install -e '.[test]'
```

On a remote development machine, the repeatable update path is:

```bash
./tools/update_remote.sh
source build/remote-current/environment.sh
```

The updater reuses the pinned gRPC toolchain, performs an incremental complete
editable build without requiring Ninja, and runs a black-box smoke test against
a temporary collection. A copied source snapshot is supported even when it has
no usable Git metadata (including a `.git` file that points back to the source
machine); only `--pull` requires a real Git worktree. The updater never touches
a real collection or restarts a running daemon by default. Use `--pull` in a
real clone to include the fast-forward update,
`--check quick` for focused streaming regressions, or
`--restart-user-service UNIT` for an explicitly managed user-systemd service.
See `./tools/update_remote.sh --help` for paths and environment overrides.
Finish copying a source snapshot before invoking the updater: it fingerprints
the build inputs before and after validation and refuses to publish stable
launcher paths if they changed mid-build.
Restart long-lived Jupyter kernels and Airflow workers after a successful
update: a process that has already imported the C++ extension keeps the old
code in memory.

The bootstrap requires outbound GitHub access, Git, CMake, a C++ compiler,
Make, Perl, and Python 3. It never uses `sudo`. See
[`docs/grpc-server.md`](docs/grpc-server.md) for the reproducible server build
and core-only developer configuration.

The optional C++ SDK is experimental and source-compatible within this MVP,
not a stable binary ABI. Recompile C++ consumers against the exact installed
ThrowDB headers/library after upgrading; pre-1.0 object files are not drop-in
compatible across releases.

## Install for development

ThrowDB requires a C++20 compiler, CMake, Python 3.10+, NumPy and pybind11.
The fast local-only loop deliberately skips the daemon; the dedicated gRPC CI
job builds the complete server with the pinned native dependencies.

```bash
python -m venv .venv
. .venv/bin/activate
CMAKE_ARGS="-DTHROWDB_BUILD_SERVER=OFF -DTHROWDB_BUILD_GRPC_SERVER=OFF" \
  python -m pip install -e '.[test]'
pytest -q
```

The wheel workflow performs a black-box CRUD, prefix-read, orphan-inspection,
and CLI smoke test from an isolated installation. The source CI separately runs
the complete Python suite on CPython 3.10-3.14, native tests with `-Werror`, and
ASan/UBSan. Publishing to a package index remains a release decision; the
workflow only creates downloadable artifacts.

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
with throwdb.Collection.create(
    "/data/example", schema=schema, format_version=7
) as collection:
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
    compacted = collection.compact_deltas()   # format 7: selective partition fold
    cleaner_plan = collection.clean_segments(dry_run=True)
    cleaned = collection.clean_segments(dry_run=False)
    orphan_plan = collection.collect_orphans(
        dry_run=True,
        min_age_seconds=3600,
    )
    maintenance_plan = collection.maintain(dry_run=True)
    maintenance = collection.maintain(dry_run=False)
```

High-level opens fail closed when retained mutable state exceeds safe defaults:
16 GiB or 4,000,000 WAL operations, and 4,000,000 entries or an estimated
256 MiB RAM overlay. The four `max_wal_backlog_*` / `max_overlay_*` arguments
make larger provisioning explicit. Format-7 materialization also has separate
input-entry and estimated-peak-memory budgets, so a lifecycle pass cannot turn
an unexpectedly dense WAL prefix into an unbounded allocation.

## Single-collection gRPC server

`throwdbd` (equivalently `throwdb serve`) resolves a saved reader profile and
then starts the native daemon for exactly one format-7 collection. It binds
loopback by default, shares one in-memory index across clients, and exposes
typed Health, Describe, Get, Insert, Update, Upsert, Delete, and
snapshot-consistent StreamGet, ScanIds, and ScanRecords. Startup never
calibrates: `--tuning auto` reuses a
safe profile or reports a conservative fallback, `persisted` fails closed on a
miss, and `conservative` bypasses the registry. The same Python package
provides `RemoteCollection`:

```bash
throwdbd \
  --storage /data/example \
  --listen 127.0.0.1:50051 \
  --writable
```

Point-read workload logging for future physical-layout optimization is
optional and disabled by default. Enable it only with a dedicated directory
that is neither inside nor above the collection:

```bash
throwdbd \
  --storage /data/example \
  --listen 127.0.0.1:50051 \
  --read-log-dir /data/throwdb-workload/example

throwdb workload inspect \
  --logs /data/throwdb-workload/example \
  --storage /data/example \
  --max-drop-ratio 0 \
  --max-row-drop-ratio 0 \
  --require-suitable \
  --output workload-report.json
```

The dedicated log directory is bound to the collection ID. If the collection
is rebuilt under a new ID, archive or remove that directory before enabling
logging for the rebuilt collection.

The native hot path records every submitted ID from valid `Get`/`StreamGet`
requests in original order. It does not sample, truncate, or infer request
categories. Large logical requests are split into exact 8192-ID records and
written through a preallocated nonblocking MPSC queue. Queue or disk pressure
marks telemetry incomplete rather than failing reads; only a zero-loss trace is
suitable for layout training. The default bounds are a 256-MiB queue,
256-MiB segments, 50-GiB retention, 10-GiB free-space reserve, and 16-MiB/s
writer ceiling. Version 2 additionally records request/batch timing and
aggregate range/page/chunk costs from the profiled native read path. Logs
contain raw numeric IDs, not embeddings, so they are sensitive and deliberately
live outside collection/S3 backup. See
[`docs/workload-aware-layout.md`](docs/workload-aware-layout.md) and the
[server runbook](docs/grpc-server.md#optional-bounded-read-workload-logging).

Both the ready record and `RemoteCollection.describe()` report the complete
effective reader/tuning selection and separate unary/scan availability bounds.
`Describe` additionally reports live per-pool gauges, saturating admit/reject
counters, and drain/terminal state through `admission_stats`.
Scan streams have configurable server lifetime and idle-progress deadlines, so
abandoned scans cannot retain admission capacity indefinitely. A raw source-build
`./build/server-release/throwdbd` bypasses the registry; exercise the normal
startup path with
`throwdbd --native-binary "$PWD/build/server-release/throwdbd" ...`. See
[`docs/grpc-server.md`](docs/grpc-server.md) for manual overrides, dry-run, and
fallback semantics.

```python
import throwdb

with throwdb.RemoteCollection.connect("127.0.0.1:50051") as collection:
    prefixes = collection.get([42, 100], prefix_dimensions=128)
    full_vector = collection.get([42])
    collection.upsert([200], full_vector, operation_id=None)

    # Materializes all IDs and can intentionally exhaust client RAM.
    ids = collection.list_ids()

    # The streaming form is bounded and preserves physical disk order.
    with collection.scan_ids(batch_rows=65_536, with_locations=True) as scan:
        for batch in scan:
            print(batch.ids, batch.locations["segment_id"])

    # Arbitrarily large query arrays/memmaps are normalized one batch at a
    # time. Order and duplicates are preserved by one pinned read snapshot.
    with collection.stream_get(
        query_ids,
        batch_rows=65_536,
        max_batch_payload_bytes=64 << 20,
        prefix_dimensions=128,
    ) as stream:
        for batch in stream:
            consume(batch.ids, batch.embeddings)

    # Full export is also bounded and stays in physical storage order.
    with collection.scan_records(
        batch_rows=65_536,
        prefix_dimensions=128,
        with_locations=True,
    ) as scan:
        for batch in scan:
            consume(batch.ids, batch.embeddings, batch.locations)
```

The local `Collection` exposes the same two operations. Its `stream_get()`
iterator yields embedding arrays in query order, while `scan_records()` yields
`RecordBatch`; both also have reusable-buffer forms for direct NumPy memmap
writes. See [`docs/streaming-reads.md`](docs/streaming-reads.md) for snapshot,
memory, cancellation, and partial-progress semantics.

Public mutations always carry a nonzero UUID operation ID. Supplying one
explicitly lets a caller repeat an ambiguous request after reconnect/restart;
the durable server journal replays the original result only when the complete
request digest matches. `RemoteCollection` creates an ID automatically when
none is supplied and exposes it on `AmbiguousCommitError` if the outcome is
unknown. Ambient HTTP proxies are disabled by default for internal embedding
traffic; opt in explicitly with `use_http_proxy=True`. The MVP transport is
plaintext only on a numeric loopback endpoint. A non-loopback bind requires
server certificate, private key, and client CA (mTLS), unless the operator uses
the explicit unsafe trusted-network override. The daemon is read-only unless
`--writable` is passed and bounds gRPC parser memory, admitted calls, scans, and
estimated in-flight bytes. Application authorization and multi-collection
routing remain later server stages.

For diagnostics that do not otherwise need an open collection, use
`Collection.inspect_orphans("/data/example", min_age_seconds=3600)`. This
lightweight path validates the selected catalog and scans mutation namespaces
without loading the base index. A previous plan is never accepted as deletion
authorization. `collect_orphans(dry_run=False)` on an `r+` handle performs a
fresh locked scan, applies hard file/byte budgets, revalidates every pathname
without following symlinks, unlinks it, and syncs each affected directory.
Format-7 superseded catalogs and interrupted segment-pair singletons converge
through the same age, budget, and identity checks; format 6 keeps its historical
catalog-retention policy.

`Collection.create()` builds format 7 when `format_version` is omitted. The
public boundary accepts versions 6 and 7; `format_version=6` remains available
for legacy compatibility. Version 6 retains a flat base index plus immutable
DeltaV1 runs. Version 7 makes partition snapshots and their active immutable run
sections authoritative, stores every immutable embedding segment with a
reverse `{id, sequence}` sidecar, and supports selective segment replacement.
The reader also opens format 5 and legacy version-3/4 collections; format 5
remains catalog/WAL-only and rejects materialization. Version 3 uses a full
structural index validation as its checksum-free fallback. UTF-8 IDs are
explicitly postponed; the current product accepts numeric IDs only.
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

The bounded matrix-v3.1 calibrator runs its complete decision-bearing matrix in
a fresh Python process, samples IDs internally, verifies every candidate
bit-for-bit, screens the bounded topology matrix, and runs a longer
verification stage for at most five finalists. The parent keeps a stable
read-only calibration handle open while the worker owns all measured I/O; the
worker opens the index once and never reopens it per candidate. If
the initial verification would retain the sequential baseline, one
pre-budgeted backfill verifies the best screening-eligible candidate that was
left out, interleaved with a fresh sequential control. The candidate stores
that control in its top-level `pairing_reference`, so paired ratios never use
an older baseline epoch. A candidate whose only failure is 10-20%
throughput/paired-gain CV, with every paired gain at least 5%, gets one
pre-budgeted paired recheck before the final decision. The read bound reserves
both the backfill candidate and its sequential control. These optional slots
are included in
`workload.requested_read_bound_bytes`; actual
`budget.planned_read_bytes` is lower when they are not used.

The calibrator collects best-effort process CPU, cgroup throttling/pressure,
and storage-I/O telemetry alongside throughput and sample-rich P99 latency.
Before each screening or verification stage on supported POSIX platforms it
uses per-file `POSIX_FADV_DONTNEED` for the collection's embedding payloads and
WALs; it never drops the machine-wide page cache. Separate ambient-disjoint ID
pools avoid candidate and cross-stage warming when the sample is large enough.
The report exposes every attempt under
`workload.stage_cache_preparation`. Mixed-request starvation requires both a
greater than 2x P99
slowdown against the candidate's own standalone small requests and a greater
than 1.25x absolute P99 regression against the baseline mixed workload. The
persisted `balanced-v1` winner must pass correctness, stability, tail,
fairness, attributable-throttling, and physical-storage guardrails. CPU and
cgroup counters may degrade cleanly to the throughput policy, but automatic
profile persistence requires reliable `/proc/self/io:read_bytes` backing-read
evidence from every non-hot scenario in both screening and authoritative
verification, plus successful file-scoped cache preparation for every executed
stage. The fresh-process boundary keeps Jupyter, Airflow, and application
background I/O out of those process-wide counters. Buffered/thread-pool reads
must also keep each repetition's `/proc/self/io:rchar` volume within the
expected payload range. Some kernels do not charge `io_uring` completions to
`rchar`; only inside the exec-isolated worker may a present `rchar` no greater
than 0.10x of the payload use the calibrator's exact profiled request bytes
instead. The parent verifies the worker PID and protocol before persistence.
Missing or ambiguously partial `rchar`, physical `read_bytes` below the backing
threshold, and over-reporting still fail closed. There is deliberately no
in-process fallback. A
page-cache-dominated, mixed, unisolated, or unobservable run remains report-only
with `cache_unrepresentative` or `cache_unverified` status. Present but
low-confidence cgroup attribution also makes the
calibration inconclusive. The selection report
distinguishes observed from guardrail-eligible maximum throughput and explains
which objectives actually participated, including topology cost. Revision 2
selection reports expose `selection_valid`, the verified candidate set, and
the final guardrail-eligible comparison set. The calibrator does not need the
original NumPy ID file. Pass
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
body and every immutable delta index. Normal format-7 open verifies the
catalog-selected partition snapshots and immutable run bodies, plus every
segment header, descriptor, sidecar geometry, and payload size. Large embedding
payloads are not rescanned solely to recompute their CRC during ordinary open.
Use `Collection.open(path, verify_index=True)` to additionally run the much
more expensive format-specific structural and embedding-payload audit. For
format 6 this includes flat-index slots, `RowRef`s, uniqueness, and probe paths;
format 7 verifies complete segment payload CRCs and authoritative snapshot
structure. Per-phase durations and the selected validation path are available in
`collection.open_stats` and are copied into load-test `report.json`.
`throwdb inspect --storage /data/example` prints the same diagnostics as JSON;
add `--verify-index` for the full format-specific audit. A standalone scrub
command and complete format-6 DeltaV1 payload scrubbing remain separate work.

`collection.mutation_log_stats` reports the selected catalog generation,
committed/indexed sequence, frozen-WAL count, immutable-run count, and
active/total WAL, run, and segment bytes. Calling `rotate_wal()` seals a
non-empty active WAL and atomically publishes its successor. On format 7,
`materialize_frozen_wals()` collapses a bounded frozen-WAL prefix, streams its
winning PUT rows into immutable segments, writes one partitioned mutation run,
and atomically advances the indexed sequence. `compact_deltas()` then folds
complete active run sections for selected partitions into new authoritative
partition snapshots; it does not rewrite embedding payloads. Format 6 keeps
the older DeltaV1 materialization and oldest-prefix merge semantics.

`clean_segments()` is format-7-only. It scans a bounded number of immutable ID
sidecars, chooses at most one segment whose garbage and reclaimable-byte
thresholds are met, copies only rows still authoritative at the indexed
sequence, and publishes replacement segments plus a cleaner run. Dry-run uses
the same deterministic candidate policy without writing. Materialization,
folding, and cleaning deliberately leave retired pathnames for the separately
locked orphan collector; already-open readers keep their pinned old inodes.

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

All requests from simultaneous callers of one collection handle share one
thread-pool coordinator when workers are enabled. It applies FIFO bounded
admission and deficit round robin (DRR) between requests, while a global worker
credit bound prevents callers from multiplying device concurrency. Each work
unit contains at most `io_ranges_per_task` physical ranges, and its scheduling
cost accounts for both range count and 4-KiB-rounded physical bytes. Small
requests can therefore complete between units of a large request while the
large request continues to make progress. A one-range request uses the same
coordinator and fairness bound; the synchronous direct-read path exists only
when the thread pool is disabled. A zero queue capacity with positive workers
selects `2 * workers` queued tasks.

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

`io_uring_workers=1` uses one persistent caller-driven ring. Two or more workers
create independent persistent ring lanes with one owner thread per lane.
`io_uring_queue_depth` is the aggregate depth and is divided across lanes. The
same DRR model schedules bounded range/byte work units between concurrent calls
and owns the aggregate ring credits; FIFO ticket admission bounds the number of
jobs waiting inside the coordinator. One-range requests also pass through this
coordinator and its ring lane. `io_workers` and `io_queue_capacity` must both be
zero in io_uring mode.

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

## S3 disaster-recovery checkpoint

The recovery workflow hashes and verifies a quiesced format-7 collection,
mirrors it to alternating slots in a versioned S3 bucket, and publishes
`LATEST` with an S3 compare-and-swap. Every completion records the SHA-256,
ETag, and immutable VersionId of every file:

```bash
throwdb backup s3 \
  --storage /nvme/throwdb/collection \
  --destination s3://my-versioned-bucket/throwdb/collection
```

The command fails while a writable server or embedded writer holds the
collection lease, when a different host owns the remote publisher lease, or
when the local lineage is stale relative to S3. A daily scheduler should force
one run as shown; `--if-older-than-hours` is intended only for more frequent
jobs and must be paired with backup-age monitoring. `--dry-run` performs all
read-only source/lineage checks and computes the exact inactive-slot plan.

A restore always targets a new local path. It pins the current slot to the
completion's VersionId/ETag map before and after transfer, validates every
SHA-256 and the complete native index, then atomically renames the staging
directory:

```bash
throwdb restore s3 \
  --source s3://my-versioned-bucket/throwdb/collection \
  --storage /nvme/throwdb/restored \
  --expected-backup-id <32-character-id-from-monitoring>
```

See [the S3 backup policy and restore drill](docs/grpc-server.md#initial-s3-disaster-recovery-policy).

## CLI

```bash
throwdb create \
  --storage /data/example \
  --ids ids.npy \
  --embeddings embeddings.npy \
  --format-version 7

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

throwdb clean-segments \
  --storage /data/example \
  --dry-run

throwdb clean-segments \
  --storage /data/example \
  --execute

throwdb collect-orphans \
  --storage /data/example \
  --dry-run \
  --min-age-seconds 3600

throwdb collect-orphans \
  --storage /data/example \
  --execute \
  --min-age-seconds 3600 \
  --max-files 1024 \
  --max-bytes 8589934592

throwdb maintain --storage /data/example --dry-run
throwdb maintain --storage /data/example --execute
```

`collect-orphans` requires exactly one explicit mode. `--dry-run` is advisory;
`--execute` always rescans under the writer and commit leases before deleting.
It considers only old temporary publications and catalog-unreferenced mutable
resources. For format 6 these are WALs and delta pairs; superseded catalogs stay
retained. Format 7 additionally collects superseded catalogs, complete segment
pairs, safe partial segment singletons left by interrupted cleanup, partition
snapshots, and index runs. A half still named by CURRENT remains reachable and
is never a candidate. Reachable files, symlinks, and unknown entries are never
removed.

Embedded `Collection.maintain()` and the CLI command below are explicit
one-pass policies, not library-created background threads. Format 7
uses the fixed order rotate WAL, materialize a bounded frozen prefix, fold
selected partitions, clean at most one segment, then collect bounded orphans.
Format 6 omits the cleaner and keeps its oldest-prefix DeltaV1 merge. Its
wall-clock limit is cooperative and checked only between durable actions.

A practical local format-7 lifecycle is:

```bash
# One-time build from rebuildable NumPy sources.
throwdb create --storage /nvme/catalog \
  --ids ids.npy --embeddings embeddings.npy --format-version 7

# Embedded applications use Collection.open(...), CRUD, and get/get_into.
# Run embedded/CLI maintenance explicitly from one operator process.
throwdb maintain --storage /nvme/catalog --dry-run
throwdb maintain --storage /nvme/catalog --execute

# Inspect deferred reclamation independently when needed.
throwdb collect-orphans --storage /nvme/catalog --dry-run
```

The stages solve different costs. WALs make small commits durable with little
write amplification but add recovery/overlay work. Materialization bounds WAL
replay and writes only winning changed rows. Partition folding reduces lookup
layers but writes only index state. Cleaning is the payload-expensive stage: it
reads one candidate and copies its still-live rows, so conservative garbage and
reclaimable-byte thresholds avoid churn. Orphan age and byte/file budgets mean
disk space is intentionally reclaimed after logical publication, not as part of
its critical section.

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
- crash-safe manual WAL rotation through immutable catalogs; the bounded
  writable-server worker also performs replay-journal checkpointing, WAL
  rotation/materialization, partition folding, selective segment cleaning, and
  orphan reclamation. A post-commit byte-threshold valve limits polling
  overshoot; before PREPARE, a commit that would cross the materialization
  budget rotates the prior WAL, while one frame that cannot fit is rejected. A
  pre-existing oversized WAL requires a larger explicit limit;
- format-7 frozen-WAL-prefix materialization into immutable embedding segments
  and a partitioned mutation run, with concurrent commits outside the two short
  catalog publications;
- bounded selective folding of complete active run sections into authoritative
  partition snapshots, without rewriting embedding segments;
- bounded format-7 cleaning of at most one high-garbage segment per call,
  publishing replacement segments and an index fence while preserving old
  reader views;
- public lightweight orphan inspection, fail-closed bounded orphan deletion,
  bounded frozen-WAL materialization, and explicit deterministic `maintain()`;
- fair DRR scheduling and one shared I/O budget for mixed-size concurrent reads
  whenever the thread-pool or io_uring parallel coordinator is enabled;
- a public C++ gRPC service for one existing format-7 collection, including
  numeric CRUD, prefix reads, bounded snapshot point-read streaming,
  physical-order ID/record streaming, health checks, and
  bounded durable mutation replay with explicit replay domains; it starts
  read-only, restricts plaintext to numeric loopback, supports mTLS, closes its
  data plane on drain/terminal storage faults, reserves independent unary/scan
  admission pools, bounds abandoned streams, exports live engine/admission
  state, and enforces a process-level hard shutdown deadline;
- a quiesced, checksummed, version-pinned S3 A/B backup and atomic restore
  workflow with lineage, publisher lease, compare-and-swap publication, and
  complete post-download validation;
- no UTF-8 IDs, payload scrub command, or direct prefix-only physical layout.
  The collection lifecycle and the server's backing store currently rely on
  local POSIX rename, `fsync`, open-file, and unlink semantics; network
  filesystems are outside this durability contract. Automatic cleaning is
  conservative and bounded rather than a full storage optimizer; application
  authorization, multi-collection routing, full payload scrubbing, and
  reduced-copy large RPCs remain later stages.

See [the storage format](docs/storage-format.md),
[the mutable-storage design](docs/mutable-storage.md), and
[the streaming contract](docs/streaming-reads.md), plus
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

For an end-to-end test of the profile selected by `throwdb tune`, pass
`--tuning persisted`. Native collections then require a matching saved profile
instead of silently falling back, and `report.json` records the complete
effective `collection.tuning` selection. The default `--tuning explicit` keeps
the existing `--io-*` comparison workflow.

The on-disk format is versioned but still experimental. Format-v1 compatibility
is covered by a byte-level golden test; format v2 may still evolve before its
first stable declaration.
