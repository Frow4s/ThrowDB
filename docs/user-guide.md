# ThrowDB user guide

This guide is the shortest complete path from NumPy files to either an
embedded local collection or a remotely served single collection. Commands
assume format 7, which is the current default and the only server-supported
mutable format.

## Choose an operating mode

| Mode | Use it when | Process model |
|---|---|---|
| Embedded Python | vectorization and reads happen in one Python application | Python calls the C++ extension directly |
| CLI | building, inspecting, tuning, or manually maintaining a collection | one short-lived Python process calling C++ |
| gRPC service | several applications share one collection | Python client → native C++ `throwdbd` |

All three use the same on-disk collection. Never open the same collection with
two writable processes. An embedded writer, writable server, maintenance
command, calibration, and backup coordinate through collection locks and will
fail rather than silently become concurrent writers.

## Installation

### From a prebuilt wheel

Match all of Python ABI, OS, architecture, and glibc baseline. For example:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  ./throwdb-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
python -m pip check
throwdb --help
throwdbd --help
```

Official Linux wheels are configured for CPython 3.10–3.14, x86_64, and
`manylinux_2_28`. They do not run on Alpine/musl. There is not yet a published
PyPI package or permanent GitHub Release; consult
[`release-packaging.md`](release-packaging.md) before looking for an artifact.

### From source

The complete build includes a pinned native gRPC/Protobuf toolchain:

```bash
git clone git@github.com:Frow4s/ThrowDB.git
cd ThrowDB
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

export THROWDB_GRPC_PREFIX="$HOME/.local/throwdb/grpc-1.81.0"
export THROWDB_GRPC_JOBS=4
./tools/bootstrap_grpc.sh
python -m pip install -e '.[test]'
python -m pytest -q
```

The bootstrap is rootless but requires outbound GitHub access, Git, CMake, a
C++20 compiler, Make, Perl, and Python 3. A local-core-only developer build can
disable both server targets as shown in the main README.

### Reapply a source update on a remote machine

After the one-time clone, use the repository updater instead of repeating the
bootstrap and build commands manually:

```bash
cd ~/ThrowDB
./tools/update_remote.sh --pull
source build/remote-current/environment.sh
```

The default `smoke` check creates only a temporary test collection and starts a
temporary server on an ephemeral loopback port. It does not open, modify, or
maintain the production collection. The gRPC/Protobuf bootstrap is reused when
its pinned installation is complete, while the ThrowDB C++ and Python sources
are rebuilt incrementally with Unix Makefiles.

Useful variants:

```bash
# Do not change Git state; apply files already copied or pulled into the tree.
./tools/update_remote.sh

# Add focused local and remote streaming regression tests.
./tools/update_remote.sh --check quick

# Run every Python test before publishing the stable paths.
./tools/update_remote.sh --check full

# Restart a rootless systemd service only after every selected check passes.
./tools/update_remote.sh \
  --check quick \
  --restart-user-service throwdb.service
```

Successful completion publishes stable ignored paths under
`build/remote-current/`: the public `throwdb` and `throwdbd` launchers, the raw
`throwdbd-native` binary needed by low-level benchmark tools, and an
`environment.sh` file. Existing `throwdbd` processes retain their old mapped
executable until explicitly restarted; this prevents an update from
interrupting active requests unexpectedly. Jupyter kernels, Airflow workers,
and other long-lived Python processes that have already imported ThrowDB must
also be restarted before they use the new C++ extension.

This workflow is intended for a development or benchmark host. An editable
installation is not an atomic production release: Python source files become
visible from the checkout before the new extension has finished building. Use
a versioned wheel/virtual environment or an immutable OCI image for production
rollouts.

## Input contract

`ids.npy`:

- one-dimensional;
- any NumPy signed or unsigned 8/16/32/64-bit integer dtype;
- signed values must be non-negative;
- every ID must be unique during initial creation;
- canonical on-disk representation is `uint64`.

`embeddings.npy`:

- two-dimensional shape `(N, dimension)`;
- dtype exactly `float16` or `float32`;
- the first dimension equals the number of IDs;
- one collection has one fixed dimension and dtype.

Both files may be opened with `numpy.load(..., mmap_mode="r")`; ThrowDB sends
them to native code in bounded batches rather than materializing the complete
1.5 TiB matrix in RAM.

## Create a collection

### CLI

```bash
IDS=/data/source/ids.npy
EMBEDDINGS=/data/source/embeddings.npy
STORE=/nvme/throwdb/collection

throwdb create \
  --storage "$STORE" \
  --ids "$IDS" \
  --embeddings "$EMBEDDINGS" \
  --format-version 7 \
  --shard-size-gib 16 \
  --batch-rows 65536
```

The destination must be new. Do not build into the path currently served by a
daemon. Publication becomes visible only after immutable data and the initial
authoritative index have completed.

### Python

```python
from pathlib import Path

import numpy as np
import throwdb

ids = np.load("/data/source/ids.npy", mmap_mode="r")
vectors = np.load("/data/source/embeddings.npy", mmap_mode="r")

schema = throwdb.CollectionSchema(
    id_type="uint64",
    dimension=vectors.shape[1],
    dtype=str(vectors.dtype),
)

with throwdb.Collection.create(
    Path("/nvme/throwdb/collection"),
    schema=schema,
    format_version=7,
    shard_size_bytes=16 << 30,
) as collection:
    for begin in range(0, len(ids), 65_536):
        end = min(begin + 65_536, len(ids))
        collection.add(ids[begin:end], vectors[begin:end])
```

Exiting the context normally calls `finish()`. An exception aborts the
unpublished builder.

## Validate the new collection

Normal inspection validates manifest/catalog/index structure without reading
every payload byte:

```bash
throwdb inspect --storage "$STORE"
```

The expensive audit additionally verifies complete format-7 segment payload
checksums and authoritative index structure:

```bash
throwdb inspect --storage "$STORE" --verify-index
```

Use the full audit after initial construction, before a release, and after a
restore drill. Do not put it into every pod startup path for a 1.5 TiB
collection.

## Embedded reads

```python
import numpy as np
import throwdb

with throwdb.Collection.open(
    "/nvme/throwdb/collection",
    tuning="auto",
) as collection:
    full = collection.get([42, 100])
    prefix = collection.get([42, 100], prefix_dimensions=128)

    output = np.empty((2, 128), dtype=np.dtype(collection.dtype))
    collection.get_into(
        np.array([42, 100], dtype=np.uint32),
        output,
        prefix_dimensions=128,
    )
```

Output rows preserve query order. A missing ID raises `KeyError` before the
caller-owned output is modified. `get_into()` is preferable when a reusable
output buffer already exists.

## Embedded CRUD

```python
import numpy as np
import throwdb

replacement = np.zeros((1, 1024), dtype=np.float16)

with throwdb.Collection.open(
    "/nvme/throwdb/collection",
    mode="r+",
) as collection:
    collection.insert([200], replacement)
    collection.update([42], replacement)
    collection.upsert([100, 300], np.repeat(replacement, 2, axis=0))
    deleted = collection.delete([200, 999])
    assert deleted.tolist() == [True, False]
```

Each successful mutation is WAL-durable. Another already-open handle is a
snapshot and must be reopened to observe commits made by a different process.
The read-write handle sees its own later commits.

## Enumerate IDs

For a bounded physical-order stream:

```python
with throwdb.Collection.open("/nvme/throwdb/collection") as collection:
    with collection.scan_ids(
        batch_rows=65_536,
        with_locations=True,
    ) as scan:
        print(scan.snapshot_count, scan.snapshot)
        for batch in scan:
            ids = batch.ids
            segment_ids = batch.locations["segment_id"]
            row_indices = batch.locations["row_index"]
```

The cursor pins one logical snapshot and returns every live ID exactly once in
physical `(segment_id, row_index)` order. Locations may be used to group later
queries on the caller side, but they are snapshot-local physical information,
not stable logical IDs.

`collection.list_ids()` and `remote.list_ids()` intentionally allocate arrays
for every live ID. On hundreds of millions of rows, use `scan_ids()` unless
that allocation is deliberate.

## Stream embeddings without materializing the result

Use `stream_get()` for a very large ordered query and `scan_records()` for a
complete physical-order export:

```python
query_ids = np.load("query-ids.npy", mmap_mode="r")

with throwdb.Collection.open("/nvme/throwdb/collection") as collection:
    with collection.stream_get(
        query_ids,
        batch_rows=65_536,
        max_batch_bytes=64 << 20,
        prefix_dimensions=128,
    ) as stream:
        for embeddings in stream:
            consume(embeddings)

    with collection.scan_records(
        batch_rows=65_536,
        max_batch_bytes=64 << 20,
        with_locations=True,
    ) as scan:
        for batch in scan:
            consume(batch.ids, batch.embeddings, batch.locations)
```

For a server, `RemoteCollection.stream_get()` yields `RemoteEmbeddingBatch`
and `RemoteCollection.scan_records()` yields the same `RecordBatch` shape.
Always close a stream (prefer `with`): the snapshot may pin superseded files
until release. Do not mutate the query ID array while `stream_get()` is active.
See [`streaming-reads.md`](streaming-reads.md) for exact failure and memory
semantics.

## Tune the reader

Calibration is explicit and bounded:

```bash
throwdb tune \
  --storage "$STORE" \
  --max-seconds 300 \
  --max-read-gib 8 \
  --max-memory-mib 512 \
  --output tune-report.json

throwdb tuning show --storage "$STORE"
```

Use the final disk, filesystem, kernel, container image, CPU quota, memory
limit, and seccomp policy. The winning profile is stored in the collection.

- `tuning="auto"`: use a matching profile; otherwise conservative fallback.
- `tuning="persisted"`: require a matching reusable profile and fail closed.
- `tuning="conservative"`: bypass profiles.
- mapping: explicitly provide reader settings for one open.

Startup never benchmarks. If a collection is copied to a new device, `auto`
selects a matching profile or fallback; run `tune` again under controlled load
if the target is new.

## Explicit maintenance for embedded use

First inspect a deterministic plan:

```bash
throwdb maintain --storage "$STORE" --dry-run
throwdb maintain --storage "$STORE" --execute
```

The fixed order is:

1. rotate a large active WAL;
2. materialize a bounded frozen WAL prefix;
3. fold eligible partition-run sections into authoritative snapshots;
4. selectively clean at most one qualifying garbage-heavy segment;
5. reclaim bounded old orphan files.

Every action is separately crash-safe. Retired files are intentionally removed
later so already-open readers can continue using pinned inodes. Never delete
WALs, catalogs, runs, partitions, or segment files manually.

Useful diagnostics:

```python
with throwdb.Collection.open("/nvme/throwdb/collection") as collection:
    print(collection.open_stats)
    print(collection.mutation_log_stats)
    print(collection.tuning)

plan = throwdb.Collection.inspect_orphans(
    "/nvme/throwdb/collection",
    min_age_seconds=3600,
)
```

## Start the server locally

Plaintext is allowed only on numeric loopback:

```bash
throwdbd \
  --storage "$STORE" \
  --listen 127.0.0.1:50051 \
  --tuning auto \
  --writable
```

Omit `--writable` for the safe read-only default. The first stdout line is a
JSON ready record containing effective reader, admission, mutation, maintenance
and shutdown settings. Structured maintenance events are written to stderr.

For a non-loopback endpoint, configure mTLS:

```bash
throwdbd \
  --storage "$STORE" \
  --listen 0.0.0.0:50051 \
  --writable \
  --tls-cert server.crt \
  --tls-key server.key \
  --tls-client-ca clients-ca.crt
```

Do not use `--allow-insecure-non-loopback` in production.

## Remote client

```python
import grpc
import numpy as np
import throwdb

credentials = grpc.ssl_channel_credentials(
    root_certificates=open("server-ca.crt", "rb").read(),
    private_key=open("client.key", "rb").read(),
    certificate_chain=open("client.crt", "rb").read(),
)

with throwdb.RemoteCollection.connect(
    "throwdb.throwdb.svc.cluster.local:50051",
    credentials=credentials,
    timeout=30,
) as remote:
    description = remote.describe()
    print(description.schema, description.tuning)
    print(remote.admission_stats)
    print(remote.engine_stats)

    vectors = remote.get([42, 100], prefix_dimensions=128)
    remote.upsert([300], np.zeros((1, remote.dimension), dtype=remote.dtype))
```

The server caps each unary request/response at 64 MiB by default. Batch large
queries below both that byte limit and `max_ids_per_request`.

### Safe mutation retry

`RemoteCollection` generates a UUID operation ID when none is supplied. If the
transport fails after the server may have committed, it raises
`AmbiguousCommitError` and exposes that ID. Retry the identical operation with
the same ID:

```python
import uuid
import throwdb

operation_id = uuid.uuid4()
try:
    remote.upsert(ids, vectors, operation_id=operation_id)
except throwdb.AmbiguousCommitError as error:
    assert error.operation_id == operation_id
    # Reconnect/Describe, then replay the exact same method, IDs and bytes.
    remote.upsert(ids, vectors, operation_id=operation_id)
```

An explicit gRPC mutation `ABORTED` response is exposed as
`RemoteAbortedError`. ThrowDB reserves this status for an admission rejection
before the current request entered durable `PREPARE`, so that request
definitely did not commit. It is safe to retry even when the server does not
advertise durable idempotency, but replay only the identical method, IDs, and
embedding bytes with the error's operation ID and bounded backoff:

```python
try:
    remote.upsert(ids, vectors, operation_id=operation_id)
except throwdb.RemoteAbortedError as error:
    # Apply bounded backoff, then replay the exact logical operation.
    remote.upsert(ids, vectors, operation_id=error.operation_id)
```

`RemoteAbortedError` remains a subclass of `AmbiguousCommitError` for backward
compatibility, despite its stronger definite-rejection guarantee. Catch it
first when admission conflicts and genuinely unknown transport outcomes need
different policies. ThrowDB does not automatically retry mutations. The error
exposes `retryable = True`, `same_operation_id_required = True`, and
`definitely_not_committed = True`; these flags describe this narrow contract
and are not permission to change the request payload.

Never create a new operation ID merely because the previous outcome is
unknown. If the server reports that the replay domain expired, refresh
`Describe`; an old ambiguous operation can no longer be safely attached to the
new domain and requires application reconciliation.

Expected overload errors are typed, including
`RemoteResourceExhaustedError`, `RemoteUnavailableError`, and
`RemoteDeadlineExceededError`. Reads may use bounded exponential backoff.
Mutation retries must additionally obey operation-ID rules.

## S3 checkpoint and restore

The AWS CLI must be installed and the bucket must have Versioning `Enabled`.
Stop the writable server and prevent calibration/tuning-profile changes for the
complete backup window:

```bash
throwdb backup s3 \
  --storage "$STORE" \
  --destination s3://my-versioned-bucket/throwdb/collection \
  --dry-run

throwdb backup s3 \
  --storage "$STORE" \
  --destination s3://my-versioned-bucket/throwdb/collection
```

Record the returned backup ID in monitoring. Restore into a new directory:

```bash
throwdb restore s3 \
  --source s3://my-versioned-bucket/throwdb/collection \
  --storage /nvme/throwdb/restored-2026-07-17 \
  --expected-backup-id REPLACE_WITH_RECORDED_ID

throwdb inspect \
  --storage /nvme/throwdb/restored-2026-07-17 \
  --verify-index
```

Run sampled application reads before switching a service to the restored path.
Restore never overwrites the active directory.

## Benchmark and soak

Embedded full read benchmark:

```bash
python benchmarks/run_load_test.py \
  --store "$STORE" \
  --ids /data/source/ids.npy \
  --source-embeddings /data/source/embeddings.npy \
  --source-kind npy \
  --results benchmark-results/local-full \
  --profile full
```

Remote benchmark and soak:

```bash
python benchmarks/run_grpc_server_benchmark.py --help
python benchmarks/run_grpc_server_soak.py --help
```

Use the exact target disk and cgroup limits. Retain `report.json`, system
samples, daemon logs, software commit, wheel SHA-256, tuning report, kernel,
node model, CPU quota, memory limit, and collection generation with every
result.

## Current user-visible boundaries

- numeric IDs only; UTF-8 is reserved but not implemented;
- no nearest-neighbor search: ThrowDB stores and retrieves embeddings by ID;
- one collection per server process;
- no application roles or method-level ACL;
- no multi-node replication or automatic failover;
- no live writable S3 checkpoint;
- local POSIX/NVMe durability contract, not CephFS/NFS;
- no prefix-only physical read optimization;
- on-disk format is still pre-1.0 and source NumPy files should remain
  rebuildable.

For production deployment, continue with
[`kubernetes-deployment.md`](kubernetes-deployment.md). For internal format and
failure semantics, see [`mutable-storage.md`](mutable-storage.md) and
[`grpc-server.md`](grpc-server.md).
