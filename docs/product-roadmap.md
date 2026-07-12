# ThrowDB product roadmap

The current implementation is a testable local mutable engine for numeric-ID
collections with arbitrary fixed dimension and float16/float32 rows. It has a
fresh segmented base format, a flat RowRef index, checksummed WAL recovery,
atomic batch CRUD, one writer lease, Python bindings, and the existing
pread/io_uring readers. The remaining local-library work is WAL rotation,
selective cleaning, UTF-8 IDs, direct prefix I/O, and adaptive scheduling.
Device-specific settings such as two io_uring rings at total QD32 remain
measured profiles, not product architecture.

## Architectural layers

```text
Python / C++ API / server protocol
                |
        collection + schema
                |
 base index + delta overlay + planner
                |
   fair scheduler + global I/O budget
                |
 pread / io_uring / remote backend
                |
 local shards / network or object storage
```

The network API and network-backed media are independent capabilities. The
first server may expose a network API while continuing to store its one
collection on local NVMe.

## Stage 0: preserve the research result

- Keep the single-ring backend as the conservative reference profile.
- Preserve the measured multi-ring FIFO implementation and benchmark reports.
- Treat backend, ring count, queue depth, range merging, and admission as
  replaceable policy choices.
- Stop adding device-specific branches to the collection read path.

## Stage 1: local collection library

Deliver a stable Python wrapper over the C++ engine. Implemented and remaining
Stage 1 slices are called out inline:

- versioned collection schema;
- arbitrary dimension;
- `float16` and `float32`, followed by `bfloat16`;
- canonical unsigned-integer domain; byte-exact UTF-8 remains next;
- safe Python normalization from `uint8/16/32/64`, non-negative
  `int8/16/32/64`, and Python integers;
- immutable snapshot construction and atomic publication;
- atomic insert, update, upsert, and tombstone delete batches;
- checksummed WAL recovery and one writer lease; WAL rotation and selective
  segment cleaning remain next;
- `get`, `get_into`, and prefix-dimension reads;
- manual tuning and an observable conservative `auto` profile; calibrated and
  request-adaptive tuning remain next;
- explicit source-array rebuild into the fresh format during the experimental
  MVP; backward reading remains available only through the legacy low-level
  API.

The scheduler and backend interfaces are introduced during this stage so the
public API does not depend on `pread`, thread pools, or io_uring.

## Stage 2: server for one collection

Run one long-lived collection per server process and add:

- a binary batched read protocol and Python client;
- request deadlines, cancellation, backpressure, and maximum batch sizes;
- server-side fair scheduling and optional batching;
- network CRUD mapped to the same commit and snapshot-view semantics;
- health checks, metrics, graceful shutdown, and a minimal trusted-network
  authentication model;
- one shared RAM index instead of one copy per client process.

Local NVMe remains the first media backend. Network filesystem or object-store
reads are added through the backend interface, with their own concurrency,
range-size, retry, timeout, and cache policies.

## Stage 3: multi-collection database

Add a catalog and control plane:

- named collections with independent schemas and versions;
- create, list, publish, rollback, and drop lifecycle;
- atomic switch between immutable collection versions;
- per-collection memory, cache, I/O, and concurrency budgets;
- authentication, authorization, quotas, routing, and resource isolation;
- replication, recovery, and background lifecycle operations.

The server and multi-collection catalog preserve the local commit-sequence,
snapshot-view, durability, and selective-cleaning contracts rather than
inventing a second mutation model.

## Stage 4: log-driven collection optimization

Collect bounded aggregate workload signals:

- batch-size and prefix-dimension distributions;
- hot IDs and co-requested IDs;
- planned versus physical bytes;
- cache behaviour, queue delay, concurrency, and backend latency;
- request deadlines and priority classes.

Use these signals to propose or build a new immutable version with improved
sharding, hot tiers, prefix layout, index representation, cache allocation, and
tuning profiles. Validate the new version and atomically publish it; do not
rewrite the active collection in place.

## Adaptive execution strategy

Adaptation is split into three bounded mechanisms.

### Capability detection

Detect kernel/backend support, CPU and cgroup limits, filesystem properties,
and whether media is local or remote. Device names alone do not choose policy.

### Calibration profile

An explicit short `throwdb tune` workflow measures representative range sizes,
queue depths, worker counts, and concurrency. Its versioned result is keyed by
backend, mount/device fingerprint, kernel, CPU quota, ThrowDB version, and
collection layout. A profile is invalidated when those assumptions change.

### Request-level scheduler

The scheduler considers planned range count, useful and physical bytes, prefix
dimension, queue state, active callers, deadlines, and backend capabilities.
It chooses a bounded execution policy rather than embedding hardware constants
in the planner.

Mixed-size requests are divided into I/O quanta and scheduled with a fair
algorithm such as deficit round robin. Small requests can finish without
waiting for an entire large batch, while large requests continue to make
progress. The scheduler owns the global I/O budget; individual requests do not
multiply device concurrency.

The first automatic policy should be deterministic rules plus measured moving
averages. Statistical or learning-based optimization belongs after real server
workloads and explainable baselines exist.

## Immediate implementation order

1. Freeze the local collection and CRUD contracts.
2. Design format/schema v2 and its v1 compatibility path.
3. Introduce `CollectionSchema`, canonical numeric-ID normalization, and the
   Python `Collection` wrapper over the existing numeric store.
4. Implement generic numeric dtype/dimension support and prefix reads.
5. Specify and implement checksummed mutation WAL/deltas, snapshot reads, an
   authoritative partitioned RowRef index, and selective cleaning for numeric
   IDs.
6. Add compact UTF-8 ID storage, collision-verified lookup, and string CRUD.
7. Extract `ReadPlan`, fair scheduler, and storage-backend interfaces.
8. Add an explicit calibration profile and request-policy explanation.
9. Build the single-collection server on the same engine.
