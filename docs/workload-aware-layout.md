# Workload-aware physical layout

Status: implementation slice 1 is complete; planner/rebuild slices remain
future work.

ThrowDB can reduce point-read I/O by placing IDs that are commonly requested
together near one another on disk. This is an offline optimization pipeline,
separate from point reads and routine partition folding:

```text
bounded request log
        |
        v
metadata-only replay of the current physical planner
        |
        v
frequency and bounded co-access layout candidates
        |
        v
chronological validation and holdout replay
        |
        v
new immutable format-7 collection version
        |
        v
read-only canary, explicit cutover, retained rollback source
```

The active collection is never rewritten or automatically replaced by an
unvalidated layout.

## Physical unit being optimized

Format 7 already maps a logical numeric ID to an independent physical
`RowRef(segment_id, row_index)`. A fresh format-7 builder assigns RowRefs in
append order, so changing append order is sufficient to change physical
locality. No format 8 is required for the first implementation.

A format-7 segment is large (16 GiB by default). Merely placing co-requested
IDs in the same segment is not the objective. The physical reader:

1. resolves each ID to a RowRef;
2. sorts resolved rows by file and byte offset;
3. merges neighboring rows according to `max_gap_bytes` and
   `max_read_range_bytes`;
4. reads the planned ranges and scatters rows back into request order.

The layout optimizer must therefore pack useful neighborhoods at the reader's
range scale. With 1024-dimensional float16 embeddings and the default 1-MiB
maximum range, one dense range contains at most 512 complete rows.

Prefix reads need explicit treatment. A 128-dimensional prefix uses 256 useful
bytes but retains the full 2048-byte row stride. With `max_gap_bytes=0`, even
adjacent prefix rows are separate ranges. Layout evaluation must either model
a nonzero gap policy and its read amplification, or report only page/cache
locality gains. A future prefix-major layout would be a separate format
extension.

## Safety and reproducibility contract

Every workload artifact and layout plan is anchored to:

- collection ID;
- schema and layout fingerprint;
- catalog generation and committed sequence of the planning snapshot;
- reader-planner ABI and effective range/gap configuration;
- input-log digest and accepted time window;
- optimizer name, version, configuration, and deterministic seed.

The optimizer rejects mixed collection identities, incompatible schemas,
corrupt frames, mixed logger configurations, unbounded drop rates, and plans
whose source identity has changed. Candidate construction always targets a new
sibling directory.

The current engine has no change stream that can bring a long-running rebuild
forward from snapshot `S` to the latest mutation `T`. The first safe rebuild is
therefore offline: the source must remain read-only/quiesced from the anchored
snapshot through cutover. A later online workflow requires retained WAL/CDC
catch-up followed by a short write quiesce.

After a writable candidate accepts mutations, the old collection is no longer
a lossless rollback target. The first rollout consequently uses:

1. a read-only candidate on a separate endpoint;
2. structural, bit-exact, and workload replay validation;
3. a stopped writer and an explicit storage-path switch;
4. retention of the old version for diagnostic rollback before new writes.

## Read workload log

The server records the workload it actually receives. It does not infer a
category, a broad query, or an export. The only operation label is the real API
method, `Get` or `StreamGet`. For every valid logical request the trace keeps
all submitted numeric IDs, including duplicates, in original client order.
There is no probabilistic sampling and no oversized-request truncation.

A unary `Get` is one logical request. A `StreamGet` is also one logical request,
even when it contains many client batches. Its ID sequence is split into
physical log records of at most 8192 rows while retaining the logical offset,
input-batch sequence, and chunk sequence. A successfully written streaming
response records its global delivery sequence in the corresponding batch
record. This chunking bounds producer work and record size; it does not change
the request or select a subset of IDs. A valid request that later fails, is
cancelled, or loses its client is retained as diagnostic demand but is not
eligible for layout training.

`ScanIds` and `ScanRecords` are not point requests containing user-supplied ID
sets, so this logger does not invent such a set for them. Embeddings and
per-ID physical pointers are never logged. Raw IDs plus catalog generation are
enough to resolve a candidate layout offline; recording old pointers would
almost double the log and make it stale after maintenance.

Logging is disabled by default and enabled only with an explicit directory
outside the collection root. The enabled path uses:

- a preallocated bounded multi-producer/single-consumer queue;
- nonblocking producer admission with no logger backpressure on a read RPC;
- one sequential background writer;
- 256 MiB of queue memory, 256 MiB segments, and 50 GiB retention by default;
- a 16 MiB/s sequential-write ceiling;
- seven-day age, file-count, and 10 GiB minimum-free-space guards;
- exact counters for logical requests, records, ID chunks, batches, IDs, and
  every drop reason.

This is exact workload telemetry when its quality counters report zero loss; it
is not a durable audit WAL. Queue exhaustion, low free space, a write-budget
limit, or a logger I/O failure never changes the RPC result. Instead the
affected request becomes incomplete and the exact-capture quality gate fails.
A blocking lossless mode would feed logger latency and availability back into
the workload being measured and is deliberately not the production default.

Binary format v2 has a 4096-byte segment header and individually CRC32C-framed
little-endian records:

- `REQUEST_BEGIN` stores method, prefix dimension, snapshot identity, declared
  size when known, and monotonic/wall-clock anchors;
- `ID_CHUNK` stores the exact ordered IDs and their logical/batch offsets;
- `BATCH_METRICS` stores timing and aggregate physical-plan telemetry;
- `REQUEST_END` stores status, submitted/delivered rows, response bytes,
  sequence digest, and request-local loss;
- `CHECKPOINT` stores cumulative lifecycle and reason-specific loss counters.

Batch timing separates the phases currently observable without per-range
allocation: lookup, physical sort/plan, backend execution, response
construction, and transport delivery. Physical aggregates include storage-file
fanout, requests, merged ranges, requested/useful/planned/completed bytes,
merge-gap amplification, touched 4 KiB pages and 1 MiB chunks, page/chunk runs,
input-order discontinuities/inversions, range geometry, and backend tasks.
Unavailable sub-phases remain zero rather than being guessed. The segment
header records the exact reader backend, range/gap policy, worker/queue/task
settings, and `io_uring` topology used to obtain these measurements.

The bounded MPSC queue preserves publication order, and the single writer
assigns a strictly increasing physical record sequence. A random request
token, per-request logical offsets and chunk sequences, streaming-delivery
sequences, and monotonic timestamps reconstruct interleaved work from 8–16
parallel clients. File order alone does not imply that IDs from different
requests were semantically related: concurrent requests are not automatically
turned into co-access edges merely because they overlapped. Their timeline is
retained for queue-depth, cache, and device-contention replay.

Every segment carries the collection/schema identity, process-instance ID, and
monotonic segment sequence. It begins with the exact preceding terminal
counter baseline and is renamed from `.open` to `.tdbrl` only after its fresh
terminal checkpoint, file fsync, rename, and directory fsync succeed. A crash
may leave a torn `.open` suffix; only complete CRC-valid frames are readable.
Retention removes an oldest finalized prefix. Requests crossing that oldest
boundary are reported as left-censored and excluded. Requests unfinished at
the newest recoverable tail are right-censored and excluded. A missing chunk
between a retained begin and successful end is an interior gap and invalidates
exact capture.

The retention policy reserves one complete segment from the configured byte
budget for the active writer, so recognized workload-log files remain within
the 50-GiB default rather than overshooting it during rotation. At eight bytes
per numeric ID, 50 GiB contains about 6.7 billion ID occurrences before
framing overhead.
One 25-million-ID request is roughly 192 MiB; one 600-million-ID request is
roughly 4.5 GiB. Sixteen simultaneous 600-million-ID reads cannot all fit in a
50-GiB retained window, and their ID rate can exceed the 16-MiB/s writer bound;
the quality counters make that limitation explicit rather than silently
sampling.

Raw IDs are sensitive application data. The directory is private (`0700`,
files `0600`), must be excluded from collection/S3 backup and routine
diagnostic upload, and should live on an encrypted volume. Unlinking retained
files is not secure erasure on SSD/NVMe media.

## Log quality gate

`throwdb workload inspect` is the first consumer. It validates frames and
reports, without printing raw IDs:

- collection and schema compatibility;
- complete, failed, left/right-censored, and interior-incomplete requests;
- contiguous chunk/batch/logical-offset reconstruction;
- request-level FNV sequence digests and record CRC32C;
- configured reader geometry and covered monotonic interval;
- request method, size, prefix, timing, and physical-plan distributions;
- producer, queue, disk-pressure, write-budget, and runtime loss;
- a machine-readable `exact_capture` and suitability verdict.

For v2, a layout corpus is suitable only when exact capture is true: no
interior request gaps, corrupt frames, counter discontinuities, or retained
window drops. Failed application requests and retention-boundary censoring are
reported but excluded from training. Drop-ratio overrides remain available for
legacy diagnostic logs; they cannot relabel a lossy v2 trace as exact.

Inspection is bounded before planning begins: by default it accepts at most
65536 files, 64 GiB total, 4 GiB per file, 1024 per-file report entries, and
4096 active request states and distinct prefix-dimension buckets. IDs are
hashed and validated through bounded `pread` buffers and are never placed in
the JSON report. Operators may explicitly raise byte and metadata bounds for a
larger retained history; the file inventory remains hard-capped at 65536.

## Scalable layout candidates

Requests are hyperedges, but expanding every request to every pair costs
`O(sum(request_size^2))` and is not viable. The first optimizer is bounded:

1. chronologically split logs into train, validation, and final holdout;
2. count weighted ID frequency through external sort;
3. select a hot set by access-mass target and hard ID/RAM/temp-disk limits;
4. keep unseen and cold IDs in their existing physical order;
5. build a sparse co-access graph for hot IDs:
   - enumerate all pairs only for small requests under an edge budget;
   - project large requests through a bounded number of deterministic anchors;
   - retain only top-K neighbors per hot ID;
6. greedily pack range-sized blocks, starting with the hottest unplaced ID and
   filling from the strongest bounded frontier;
7. build both `hot-frequency-v1` and `hot-coaccess-v1` candidates.

If the workload is effectively uniform, unstable, or too weakly covered, the
valid result is `no_rebuild_recommended`.

Large requests receive request-normalized weights so one very large ID set
cannot dominate the physical layout merely through its cardinality. This is a
mathematical weighting rule over the recorded size and boundaries, not an
inferred semantic classification. Operation method, size, and prefix
distributions remain independently visible in validation.

## Exact metadata-only replay

Candidate selection is based on the same physical planning rules as execution,
not an unrelated composite score. The core range planner will be extracted
into a pure no-I/O surface used by both the reader and replay simulator.

For every validation request, replay records:

- segment fanout;
- planned physical range count;
- planned and useful bytes;
- read amplification;
- backend task estimate;
- fraction of requests served by 1, 2, and 4 ranges;
- P50/P95/P99 by meaningful workload cohort;
- optional block reuse/cache simulation at several bounded cache sizes.

Parameters are selected on validation data. Holdout is read once for the final
decision. Publication requires an explicit, versioned set of guardrails:
meaningful central/tail improvement, no unacceptable P99 or cohort regression,
bounded byte amplification, and stability across time blocks. Exact thresholds
are deployment policy rather than on-disk format.

Metadata replay cannot predict device queueing, readahead, or cache latency
perfectly. A shortlisted candidate must pass a bounded actual-device replay
before cutover.

## Rebuild strategies

The first production-safe implementation builds a complete immutable sibling
collection. A naïve target-order `get()` would turn a roughly 1.5-TB rebuild
into hundreds of millions of random reads. The reorder engine must instead use
a bounded physical strategy.

One conservative implementation is:

1. validate the hot plan and source anchor before payload work;
2. scan source records once in physical order;
3. preserve cold records in source order;
4. route bounded hot records to target-layout buckets on local temporary
   storage;
5. emit hot buckets in target order through the existing format-7 builder;
6. finish and fsync the candidate, then open and validate it.

The plan stores only hot ordering plus `cold_policy=preserve_physical_order`;
it does not need a 600-million-row full permutation. Disk preflight includes
the candidate, bounded hot scratch, index/sidecars, and operational reserve.

A later space-efficient option can recluster complete groups of existing
segments through cleaner-style RowRef runs. Current format-7 invariants require
retiring a whole source segment when any of its rows move, and multi-segment
cleaner runs temporarily increase index cost. This is useful after the clone
pipeline proves the objective, but it is not the first data-movement primitive.

## Implementation slices

1. **Implemented:** exact-ID versioned binary log codec, logical request
   lifecycle, bounded asynchronous C++ logger, profiled point-read path,
   opt-in server flags, remote quality counters, format/security/failure
   tests, and `throwdb workload inspect`.
2. Pure no-I/O physical planner plus exact baseline replay and time-split
   workload reports.
3. Bounded hot-frequency and hot-coaccess planners with deterministic artifacts
   and validation/holdout selection.
4. Offline sibling rebuild, structural and bit-exact verification, and bounded
   actual-device replay.
5. Read-only canary and documented manual cutover/rollback.
6. Later: WAL/CDC catch-up for online rebuild and bounded in-place
   multi-segment reclustering.

The first slice satisfies its functional safety gates: disabled logging does
not allocate logger state, queue saturation and injected I/O failures cannot
fail reads, memory and disk have explicit bounds, and incomplete/corrupt logs
are handled deterministically. A production-device A/B overhead measurement
remains an acceptance run rather than an architectural prerequisite; it must
compare disabled and enabled exact capture before the feature is enabled on a
high-QPS deployment.
