# Workload-aware physical layout

Status: accepted architecture for the first implementation slice.

ThrowDB can reduce point-read I/O by placing IDs that are commonly requested
together near one another on disk. This is an offline optimization pipeline,
not a new read API and not an extension of ordinary mutation compaction:

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
corrupt frames, unbounded drop rates, and plans whose source identity has
changed. Candidate construction always targets a new sibling directory.

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

The server logs logical point-read batches, not embeddings and not physical
locations. Locations are derived from the collection snapshot during replay
and would otherwise become stale after maintenance.

Event boundaries are:

- one successfully completed unary `Get`;
- one successfully completed application batch within `StreamGet`.

`ScanIds` and `ScanRecords` are physical-order administrative scans. They may
contribute aggregate counters, but their IDs are excluded from layout training
because a complete scan would create false uniform demand.

Logging is disabled by default and enabled only by an explicit directory
outside the collection root. Keeping logs outside the root prevents collection
backup/restore from treating workload telemetry as authoritative data.

The enabled hot path has these guarantees:

- request-level Bernoulli sampling before any logging work;
- a configurable hard cap on captured IDs per event;
- unbiased deterministic stratified sampling across oversized batches rather
  than retaining only their prefix;
- a preallocated, byte-bounded producer queue;
- nonblocking all-or-drop enqueue;
- one sequential background writer;
- no read failure or delay caused by queue saturation or a runtime log-write
  error;
- monotonic counters for considered, selected, written, dropped, truncated,
  and failed events and rows.

The first binary log version uses a collection/schema header and individually
framed little-endian records. Headers and frames have strict size limits and
CRC32C checksums. Rotation finalizes
`<instance>.<sequence>.open` to `<instance>.<sequence>.tdbrl`; after a crash,
the analyzer accepts the complete valid prefix of an open file and ignores a
torn tail. Files and directories are private by default. Rotation, retention,
and a minimum-free-space guard bound telemetry's disk impact.

Each point-read record contains at least:

- event sequence and coarse timestamp;
- collection/schema identity from the file header;
- operation kind (`Get` or `StreamGet` batch);
- stream and batch sequence when applicable;
- prefix dimension;
- original row count and captured row count;
- request and within-request sampling metadata;
- captured uint64 IDs.

Raw IDs are sensitive application data. The initial explicit opt-in mode stores
raw uint64 values with private permissions and documented retention. A later
privacy mode may use collision-safe keyed 128-bit tokens plus an external join
against live IDs; a 64-bit hash is not safe enough at hundreds of millions of
IDs.

## Log quality gate

`throwdb workload inspect` is the first consumer. It validates frames and
reports, without printing raw IDs:

- collection and schema compatibility;
- accepted and rejected files;
- complete, torn-tail, and corrupt frame counts;
- covered time interval;
- request-kind, batch-size, and prefix-dimension distributions;
- configured and effective sampling;
- captured/original row coverage;
- producer drops and writer failures;
- a machine-readable suitability verdict.

The planner refuses logs with an incompatible identity or a configured maximum
drop ratio violation. It warns on low time coverage, high truncation, workload
drift, or insufficient hot-set coverage.

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

Large requests are normalized so a single export-like batch cannot dominate
the physical layout. Cohorts by operation kind, batch size, and prefix
dimension are evaluated independently.

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

1. Versioned binary log codec, bounded asynchronous C++ logger, server flags,
   quality counters, format/security/failure tests, and
   `throwdb workload inspect`.
2. Pure no-I/O physical planner plus exact baseline replay and time-split
   workload reports.
3. Bounded hot-frequency and hot-coaccess planners with deterministic artifacts
   and validation/holdout selection.
4. Offline sibling rebuild, structural and bit-exact verification, and bounded
   actual-device replay.
5. Read-only canary and documented manual cutover/rollback.
6. Later: WAL/CDC catch-up for online rebuild and bounded in-place
   multi-segment reclustering.

The first slice is accepted only if disabled logging leaves the existing read
path unchanged; enabled default sampling remains within an explicit overhead
gate; queue saturation and injected I/O failures cannot fail reads; RSS and
disk use remain within configured bounds; and truncated/corrupt logs are
handled deterministically.
