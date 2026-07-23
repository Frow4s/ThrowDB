# ThrowDB product roadmap

The current implementation is a testable local mutable engine plus a C++ gRPC
single-collection server for numeric-ID collections with arbitrary fixed
dimension and float16/float32 rows. Format 7
adds an authoritative partitioned RowRef index, WAL-to-segment materialization,
selective partition folding, selective segment cleaning, deferred orphan
reclamation, and explicit bounded maintenance. When a parallel coordinator is
enabled, both thread-pool and io_uring readers use a fair request scheduler
with one backend budget. The remaining
local-library extensions include UTF-8 IDs, direct prefix I/O, and
workload-aware policy selection; UTF-8 is intentionally postponed until the
numeric lifecycle is stable.
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
- Preserve the measured multi-ring implementation and benchmark reports while
  routing concurrent work through the fair coordinator.
- Treat backend, ring count, queue depth, range merging, and admission as
  replaceable policy choices.
- Stop adding device-specific branches to the collection read path.

## Stage 1: local collection library

Deliver a stable Python wrapper over the C++ engine. Implemented and remaining
Stage 1 slices are called out inline:

- versioned collection schema;
- arbitrary dimension;
- `float16` and `float32`, followed by `bfloat16`;
- canonical unsigned-integer domain; byte-exact UTF-8 is postponed;
- safe Python normalization from `uint8/16/32/64`, non-negative
  `int8/16/32/64`, and Python integers;
- immutable snapshot construction and atomic publication;
- atomic insert, update, upsert, and tombstone delete batches;
- checksummed WAL recovery, one writer lease, WAL rotation, bounded format-7
  WAL-to-segment materialization, selective partition folding, and selective
  cleaning of one segment per pass;
- public lightweight orphan inspection, fresh-scan bounded deletion, bounded
  frozen-WAL selection, and explicit deterministic maintenance;
- `get`, `get_into`, and prefix-dimension reads;
- manual tuning, portable exact/compatible per-device profile persistence, an
  observable conservative fallback, and an explicit bounded `throwdb tune`
  calibrator;
- fair DRR scheduling of mixed request sizes with one shared budget whenever
  the thread-pool or io_uring parallel coordinator is enabled; per-request
  backend/policy selection remains a later optimization;
- explicit source-array rebuild into the fresh format during the experimental
  MVP; high-level read compatibility remains available for formats 3 through
  6, while new collections default to format 7.

The scheduler and backend interfaces keep the public API independent of
`pread`, thread pools, and io_uring.

## Stage 2: server for one collection

Run one long-lived collection per server process. The first functional slice
now includes the versioned dense-bytes gRPC contract, C++ daemon, Python
client, CRUD, prefix Get, snapshot-consistent large-query StreamGet,
physical-order ID/record streaming, bounded messages and threads,
health/reflection, graceful shutdown, writer fencing, and durable
`operation_id` replay. It also includes tune-aware one-shot startup that selects
a compatible persisted profile without implicit calibration, plus a dedicated
bounded local-loopback/external-endpoint benchmark for correctness,
client-visible latency, concurrency, mixed request sizes, ScanIds, and opt-in
CRUD. Launch hardening now also includes read-only default access, numeric-
loopback-only plaintext, non-loopback mTLS, terminal/draining data-plane
closure, process-wide call/scan/estimated-byte admission, a gRPC memory quota,
disk-reserved mutations, durable pending-operation fencing, bounded replay
journal checkpointing with explicit replay domains, bounded automatic
rotation/materialization/folding/cleaning/orphan work, and a quiesced,
version-pinned two-slot S3 backup/restore workflow. Remaining Stage 2 hardening
includes:

- complete queue/admission and lifecycle metrics on top of transport RPC
  deadlines, cancellation checks before Get/CRUD work and between ScanIds
  batches, the bounded worker pool, message limits, and local fair scheduler;
- priorities, deadline-aware scheduling, optional batching, and fewer unary
  payload copies;
- application authorization/certificate rotation, full payload scrubbing,
  long-duration production acceptance, and reduced-copy large responses.

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

An explicit bounded `throwdb tune` workflow compares matrix-v3.1 backend
topologies, queue depths, worker counts, and concurrency. A short screening
stage covers the full bounded matrix and a longer verification stage retains
at most five finalists. One pre-budgeted backfill protects against an unlucky
screening cutoff by interleaving the candidate with a fresh sequential control,
and one pre-budgeted paired recheck handles a 10-20% CV when every observed
gain remains at least 5%. Matrix v3.1 combines paired throughput
with reliable process-CPU efficiency, sample-rich P99 latency, attributable
cgroup throttling, mixed-request fairness, and ambient-load guardrails. Mixed
small-request starvation requires both candidate-local slowdown and an
absolute regression against the baseline. The result exposes the eligible
comparison set, decision-active objectives, and observed versus eligible
maximum-throughput alternatives through a revisioned, validity-marked selection
schema; low-confidence shared-cgroup attribution makes publication
inconclusive. It reports a performance/efficiency/topology
Pareto frontier and persists the balanced winner. It still holds its 1 MiB
range and zero-gap policy fixed; workload-aware range selection is a later
stage. Its versioned result is keyed by media identity, runtime compatibility
including the reader-policy ABI, and collection layout. A profile stops
matching when those assumptions change.

### Request-level scheduler

The implemented scheduler divides each physical plan into bounded work units
and charges both range count and 4-KiB-rounded bytes. Deficit round robin lets a
small request run between units of a large request while preserving progress
for the large request. One coordinator owns global backend credits, and
one-range calls do not bypass it when a parallel backend is enabled.

A later policy layer may additionally consider useful bytes, prefix dimension,
queue state, deadlines, and measured backend capabilities to choose a backend
or task geometry per request. That choice must remain bounded and explainable;
it is separate from the fairness already enforced inside each backend.

The first automatic policy should be deterministic rules plus measured moving
averages. Statistical or learning-based optimization belongs after real server
workloads and explainable baselines exist.

## Delivery order

The local collection/CRUD contract, numeric schema and compatibility design,
Python wrapper and ID normalization, float16/float32 dimensions, semantic
prefix reads, fair scheduler, and format-7 lifecycle are implemented. The next
product sequence is:

1. Stabilize the implemented format-7 mutation lifecycle, authoritative
   partitioned RowRef index, selective cleaner, orphan GC, and maintenance
   reports on production-size local NVMe collections.
2. Preserve explicit calibration profiles and add request-policy explanation
   around the implemented fair scheduler.
3. Continue hardening the tune-aware single-collection gRPC server: metrics
   export, priorities, application authorization, scheduled restore drills,
   payload scrubbing, and fewer unary copies. Admission, mTLS, safe access
   defaults, bounded automatic maintenance and journal checkpointing, live
   engine diagnostics, and lineage-safe S3 backup/restore are already
   implemented.
4. Add compact UTF-8 ID storage, collision-verified lookup, and string CRUD as
   an extension of the stable numeric lifecycle.
