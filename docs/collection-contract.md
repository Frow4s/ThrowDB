# Local collection product contract

This document fixes the contract for the first user-facing ThrowDB product.
The fresh native numeric subset is implemented through the Python `Collection`
API: safe uint64 normalization, float16/float32 rows, arbitrary fixed
dimensions, semantic prefix reads, and durable batch CRUD. Format 7 is the
only supported storage format. It provides
an authoritative partitioned RowRef index, WAL-to-segment materialization,
selective partition folding, selective segment cleaning, and deferred orphan
reclamation. UTF-8 IDs are deliberately postponed.

## Product boundary

A logical collection is a typed mapping:

```text
user ID -> fixed-schema embedding row
```

The local product contract includes record-level
create/read/update/delete operations. The current implementation provides them
for the numeric ID domain.
Approximate nearest-neighbour search, similarity scoring, replication, and a
network protocol are separate product layers.

Physical base segments remain immutable and atomically published:

```text
create -> append batches -> finish -> read-only snapshot
```

`finish()` is the initial publication point. A failed build must not replace a
readable snapshot. Record mutations are represented by committed delta records
over the base, not by overwriting active segment bytes. Routine cleaning rewrites
only selected immutable segments and atomically switches the resource catalog;
it does not copy the complete base. Existing readers retain their previous
view.

## Collection schema

The following fields are immutable for the lifetime of a snapshot:

| Field | Initial contract |
| --- | --- |
| ID domain | unsigned integer (canonical `uint64`) |
| Embedding dtype | `float16` or `float32` |
| Dimension | Positive fixed integer per collection |
| Row layout | Dense row-major embedding values |
| Format version | Explicit and independently readable |

`bfloat16` is the next planned dtype. Quantized types are deliberately deferred
because they require scale/zero-point metadata and should not be represented as
ordinary scalar dtypes.

All numerical payloads are stored bit-exactly. Reading does not perform dtype
conversion unless a future API explicitly requests it.

## ID semantics

Numeric IDs cover every `uint64` value, including zero and `UINT64_MAX`. The
on-disk and C++ index representation is canonical `uint64`; Python input dtype
does not become part of collection schema.

The high-level Python API accepts integer values whenever conversion is exact:

| Python input | Behaviour |
| --- | --- |
| `np.uint8/16/32/64` arrays | Accepted and normalized to `uint64` |
| `np.int8/16/32/64` arrays | Accepted when every value is non-negative |
| Python integer sequences | Accepted when every value is in `0..2**64-1` |
| Existing contiguous native `np.uint64` | Zero-copy fast path |
| Negative or overflowing integer | Rejected with the offending position |
| Float, complex, decimal, or boolean | Rejected; no implicit numeric coercion |

Non-native byte order and non-contiguous valid integer arrays may be copied into
a native contiguous temporary by the high-level wrapper. The low-level binding
remains strict and zero-copy so performance-sensitive callers can avoid hidden
allocation. If negative IDs become a real requirement, `int64` will be added as
a distinct logical ID domain rather than reinterpreting signed bits as uint64.

UTF-8 IDs are not accepted by the current API. Their eventual contract remains
byte-exact storage with no hidden normalization, case folding, or locale rules;
that string arena and collision-verified index are a later extension of the
stable numeric lifecycle, not an implicit coercion in the numeric API.

Duplicate build IDs are an error. Query IDs may repeat and must repeat the
corresponding output rows in the original query order.

## Read contract

The Python surface is intended to converge on:

```python
collection = throwdb.Collection.open(path, tuning="auto")

vectors = collection.get(ids)
collection.get_into(ids, output)

prefix = collection.get(ids, prefix_dimensions=128)
```

Required behaviour:

- every ID is resolved before physical reads begin;
- a missing ID fails the complete request rather than returning partial rows;
- result order is identical to query order;
- output dtype is the collection dtype;
- `get_into` validates dtype, shape, contiguity, and writability before I/O;
- one opened collection is safe to share between simultaneous read callers;
- the C++ owner must keep a handle alive until active methods return; the
  Python binding retains an internal shared owner so `close()` cannot destroy
  an already-running native call.
- opened handles and builders are bound to their creating process; after
  `fork()`, the child must open a new handle rather than use the inherited one.

`prefix_dimensions=N` returns the first `N` values of every row and requires
`1 <= N <= collection.dimension`. The API guarantee does not promise that a
row-major local backend performs only `N * item_size` bytes of physical I/O.
Filesystem page size can dominate random prefix reads. The same API can later
use a prefix-optimized physical layout without changing callers.

### ID enumeration and physical locations

Format 7 exposes the authoritative live ID set through a snapshot cursor:

```python
with collection.scan_ids(
    batch_rows=65_536,
    with_locations=True,
) as scan:
    for batch in scan:
        consume(batch.ids, batch.locations)

all_ids = collection.list_ids()
```

`scan_ids()` is the bounded interface. It pins one collection view for the
cursor lifetime, emits each live ID exactly once, omits tombstones and retired
rows, and preserves physical `(segment_id, row_index)` order within that view.
When requested, locations expose separate native `uint32` `segment_id` and
`row_index` fields. They are snapshot-scoped optimization hints, not stable
logical row numbers: maintenance may relocate a row without changing its ID or
embedding. A caller must not combine locations from one scan snapshot with IDs
or collection metadata from another.

`list_ids()` deliberately materializes the same stream into one NumPy array and
does not impose a client-memory guard. Callers that cannot bound the complete
population in RAM must use `scan_ids()` instead. Enumeration order is physical
storage order rather than insertion order, numeric sort order, or lookup-request
order.

### Bounded embedding streams

Format 7 exposes two snapshot-consistent embedding paths:

```python
# Query order and duplicates are preserved; ids may be a NumPy memmap.
with collection.stream_get(
    query_ids,
    batch_rows=65_536,
    max_batch_bytes=64 << 20,
    max_snapshot_bytes=256 << 20,
    prefix_dimensions=128,
) as stream:
    for embeddings in stream:
        consume(embeddings)

# Authoritative records are emitted in physical storage order.
with collection.scan_records(
    batch_rows=65_536,
    max_batch_bytes=64 << 20,
    max_snapshot_bytes=256 << 20,
    prefix_dimensions=128,
    with_locations=True,
) as scan:
    for batch in scan:
        consume(batch.ids, batch.embeddings, batch.locations)
```

Both operations pin immutable files and copy the mutable overlay once. The
snapshot-memory and batch-payload limits are independent; immutable mmaps,
page cache, and batches retained by the caller are not charged to those Python
limits. Mutations and maintenance may continue, but cannot change later
batches from an already-open stream. A failed batch is never yielded and the
stream cannot be resumed against a newer view. Full semantics are specified in
[`streaming-reads.md`](streaming-reads.md).

## Write contract

The initial Python surface is intended to be:

```python
schema = throwdb.CollectionSchema(
    id_type="uint64",
    dimension=1024,
    dtype="float16",
)

with throwdb.Collection.create(
    path,
    schema=schema,
) as collection:
    collection.add(ids, embeddings)
```

Creation and opening support only format 7. The optional `format_version`
argument accepts only `7`, also reported by `collection.format_version`.

Writers accept bounded batches and may spill build metadata to disk. Successful
context-manager exit calls `finish()`; exceptional exit must not publish the
snapshot. Readers never depend on writer-process memory.

After initial publication, open a writer lease and use the record API:

```python
collection = throwdb.Collection.open(path, mode="r+")
collection.insert(ids, embeddings)   # every ID must be absent
collection.update(ids, embeddings)   # every ID must exist
collection.upsert(ids, embeddings)   # explicit insert-or-replace
deleted = collection.delete(ids)     # boolean result per requested ID
collection.rotate_wal()              # freeze a non-empty active WAL
collection.materialize_frozen_wals() # replace a bounded frozen-WAL prefix
collection.compact_deltas()           # format 7: fold selected partitions
cleaner_plan = collection.clean_segments(dry_run=True)
cleaned = collection.clean_segments(dry_run=False)
plan = collection.collect_orphans(    # advisory report
    dry_run=True,
    min_age_seconds=3600,
)
reclaimed = collection.collect_orphans(
    dry_run=False,                    # fresh locked scan; never reuses plan
    min_age_seconds=3600,
    max_files=1024,
    max_bytes=8 << 30,
)
maintenance = collection.maintain(dry_run=False)
```

Mutation rules:

- schema, dimension, and dtype never change through record CRUD;
- update replaces a complete embedding row; prefix-only mutation is not part of
  the initial contract;
- duplicate IDs inside one mutation batch are rejected as ambiguous;
- insert and update validate the complete batch before commit;
- delete is idempotent and reports `False` for an already absent ID;
- a committed batch is visible completely or not at all;
- one writer lease is allowed per collection, with simultaneous readers;
- the default durability mode acknowledges only after the commit record is
  synced; a future explicit relaxed mode may acknowledge earlier.

`Collection.open()` applies fail-closed startup defaults to retained mutable
state: 16 GiB/4,000,000 WAL operations and a 4,000,000-entry, 256 MiB estimated
overlay. Override `max_wal_backlog_bytes`, `max_wal_backlog_operations`,
`max_overlay_entries`, or `max_overlay_memory_bytes` only after provisioning the
corresponding disk/RAM. Format-7 materialization is independently bounded by
selected WAL count, bytes, committed operations, and estimated peak memory.
Writable handles also check projected total WAL bytes and committed-operation count before each
append, unique overlay growth before publication, and the additional 4096-byte
WAL header before rotation. These bounds reject work before the corresponding
durable write; they never silently truncate durable history.

Each handle currently captures one committed collection view when it opens; a
read-write handle also sees its own later commits. Reopen a different process's
already-open handle to observe later commits. Within a handle, each read
resolves all IDs against one state, so it cannot mix rows across a commit.

## Mutation and segment-cleaning model

```text
immutable embedding segment catalog
          + authoritative index state
          + committed WAL overlay
          = logical collection view
```

Mutation batches receive a monotonic commit sequence and are protected by
checksummed WAL framing and a commit marker. Recovery ignores
incomplete/uncommitted tails. Reads check the current WAL overlay,
then active immutable run sections, then an authoritative partition snapshot;
absence from all three is final and never falls through to a retired segment.

In format 7, `materialize_frozen_wals()` selects a bounded oldest frozen-WAL
prefix and collapses each ID to its newest state in that prefix. Winning PUT
payloads are read in source-offset order and streamed bit-exactly into one or
more immutable `segments/*.dat` plus matching `{id, sequence}` sidecars. One
immutable mutation run stores the resulting per-partition PUT/DELETE mappings.
The final catalog publication advances `indexed_sequence`, removes only the
selected WAL references, and preserves every newer WAL. Commits and rotations
may proceed during the long build; only durable reservation and final rebase
are short serialized metadata transactions.

For format 7, `compact_deltas()` means selective index folding rather than
embedding compaction. Its default admits up to all 4096 active
runs so a partition cannot become permanently ineligible merely because more
than eight runs accumulated; callers may set a smaller explicit bound. A
partition is eligible only when its complete current
snapshot and every active run section for that partition fit the run, byte, and
entry budgets. The merge keeps the greatest original sequence for each ID,
omits authoritative tombstones, publishes a new complete partition snapshot,
and clears only that partition's active bits in its input runs. A run that
still covers another partition remains reachable; a run with no active section
is removed from the catalog and becomes eligible for deferred orphan
collection. No embedding segment is copied by a fold.

`clean_segments()` is the separate format-7 payload reclamation operation. It
scans bounded immutable sidecars at the catalog's indexed sequence, chooses at
most one segment, and copies only rows whose `{id, sequence, RowRef}` is still
authoritative. One cleaner run relocates those rows and fences every discarded
source row, while WAL changes newer than the indexed sequence remain above it.
The source descriptor and replacement descriptors switch atomically. The
cursor returned by a call is only a performance hint for continuing a bounded
scan; losing it cannot change correctness.

For local POSIX storage, `locks/commit.lock` closes publication races and the
format-7 `locks/view.lock` closes the race between opening a selected view and
unlinking retired immutable resources. Readers hold the view lease until every
selected segment descriptor, snapshot mmap, run mmap, and WAL descriptor is
pinned. The orphan collector takes both locks exclusively for a fresh scan,
identity revalidation, unlink, and directory sync. Existing readers continue
through pinned inodes after a catalog switch.

`Collection.inspect_orphans(path, min_age_seconds=3600)` provides the same
read-only plan without first opening the collection or loading its base index.
It briefly acquires the writer lease while validating `CURRENT`, its selected
catalog, and the mutation namespaces. An existing `r+` handle can instead call
`collect_orphans(dry_run=True, min_age_seconds=...)` while retaining its writer
lease. `dry_run=False` executes only from an `r+` handle and always performs a
fresh scan; it never consumes the earlier plan. The CLI requires exactly one of
`--dry-run` and `--execute`; its execute path remains lightweight and does not
load partition snapshots.

The inventory recognizes temporary publications, unreferenced WALs,
superseded catalogs, complete or
partial segment resources, partition snapshots, and index runs. A complete
segment pair is one atomic budget unit; an unreferenced partial singleton is a
candidate so a later pass can finish interrupted pair cleanup. Any remaining
half named by CURRENT is reachable and never a candidate. Symlinks and
unknown/do-not-touch entries are also never deleted. The minimum age controls
eligibility. A report is a potentially stale diagnostic snapshot, not an
authorization token. Execute mode performs a fresh locked scan, revalidates
identity without following symlinks, unlinks only within hard file/byte
budgets, and fsyncs affected directories. Independently, format 7 fails before
unlink if scanned entries exceed the active catalog's reachable-path count by
more than 65,536. This is the local MVP's hard scan-memory/lock-time guard;
resumable streaming with an opaque performance cursor remains a later
extension.

`maintain(dry_run=True)` explains one deterministic embedded pass. For format 7,
explicit execution considers, in order: WAL rotation, bounded materialization,
selective partition folding, one selective segment clean, and orphan
collection.
The embedded library creates no background thread. A writable single-collection
server runs bounded rotate/materialize/fold/clean/orphan calls. `max_actions` is hard and `max_seconds` is cooperative
between complete durable actions.

`max_input_bytes` bounds the selected source-file footprint, not aggregate
physical I/O: integrity scans, copying winning rows, and replacement readback
are additional reads. `max_input_entries` is the explicit in-memory metadata
bound for this first implementation.

Untouched format-7 segments and partitions remain in place throughout this
lifecycle. Retired pathnames are reclaimed by the later orphan action rather
than by materialization, folding, or cleaning itself. A complete collection
rewrite is reserved for explicit schema/layout migration rather than normal
CRUD maintenance. The detailed contract is in `mutable-storage.md`.

## Storage version policy

Format version is a storage contract, not a package version. Only format 7 is
implemented. Unsupported formats are rejected before opening or mutating a
collection. There is no reader, writer, automatic conversion, or in-place
migration for earlier experimental formats. Rebuild needed data into a new
format-7 collection from source NumPy arrays.

The byte layout and selective lifecycle are documented in
[mutable-storage.md](mutable-storage.md).

## Tuning contract

Correctness and collection schema are independent of I/O tuning. Callers may
select:

```python
Collection.open(path, tuning="auto")
Collection.open(path, tuning="persisted")
Collection.open(path, tuning="conservative")
Collection.open(path, tuning={"io_backend": "io_uring",
                              "io_uring_workers": 2,
                              "io_uring_queue_depth": 32})
```

Automatic tuning must be observable, bounded, and overrideable. The product
exposes the selected source and compatibility reason through
`collection.tuning`. Exact media/runtime/layout profiles are optional advisory
metadata inside the collection; a corrupt registry falls back in `"auto"` and
is an error in `"persisted"`. Opening a collection never unexpectedly runs a
destructive or long cold-cache benchmark.

Calibration is an explicit bounded operation:

```python
report = Collection.tune(
    path,
    max_seconds=300,
    max_read_bytes=8 << 30,
    max_memory_bytes=512 << 20,
    dry_run=False,
)
```

It holds the lifecycle writer lease on one read-only snapshot, samples live
IDs from the active index/WAL view without a population scan, compares bit-exact temporary readers that
share the already-loaded index, and publishes only after rechecking the
media/runtime/layout identity, root inode, mutation snapshot, registry salt,
and registry generation. Failure, interruption, instability, a cooperative
deadline, or `dry_run=True` leaves the registry unchanged.
