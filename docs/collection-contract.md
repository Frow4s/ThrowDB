# Local collection product contract

This document fixes the contract for the first user-facing ThrowDB product.
The fresh native numeric subset is implemented through the Python `Collection`
API: safe uint64 normalization, float16/float32 rows, arbitrary fixed
dimensions, semantic prefix reads, and durable batch CRUD. The public format
version produced by new builders is 6. WAL rotation, frozen-WAL-prefix
materialization, and bounded oldest-prefix DeltaV1 compaction are implemented;
UTF-8 IDs and selective base cleaning remain later local-library slices. Format
5 remains a readable and writable catalog/WAL-only format, but it rejects both
materialization and delta compaction.

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
over the base, not by overwriting active shard bytes. Routine cleaning rewrites
only selected immutable segments and atomically switches the resource catalog;
it does not copy the complete base. Existing readers retain their previous
view.

## Collection schema

The following fields are immutable for the lifetime of a snapshot:

| Field | Initial contract |
| --- | --- |
| ID domain | unsigned integer (canonical `uint64`) or byte-exact UTF-8 |
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

String IDs are UTF-8 byte sequences with no hidden normalization, case folding,
or locale rules. Applications that need Unicode normalization must apply it
before writing and querying. A string index may use fingerprints internally,
but every match must be verified against the stored bytes; hash collisions must
never return the wrong embedding.

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

## Write contract

The initial Python surface is intended to be:

```python
schema = throwdb.CollectionSchema(
    id_type="uint64",
    dimension=1024,
    dtype="float16",
)

with throwdb.Collection.create(path, schema=schema) as collection:
    collection.add(ids, embeddings)
```

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
collection.materialize_frozen_wals() # replace frozen prefix with DeltaV1
collection.compact_deltas()           # merge a bounded oldest delta prefix
plan = collection.collect_orphans(    # advisory report; no deletion
    dry_run=True,
    min_age_seconds=3600,
)
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

Each handle currently captures one committed collection view when it opens; a
read-write handle also sees its own later commits. Reopen a different process's
already-open handle to observe later commits. Within a handle, each read
resolves all IDs against one state, so it cannot mix rows across a commit.

## Mutation and segment-cleaning model

```text
immutable base snapshot
          +
committed PUT/DELETE delta segments
          =
logical collection view
```

Lookup checks the newest committed delta state first. A delete is a tombstone;
otherwise lookup falls back to the base index. Mutation batches receive a
monotonic commit sequence and are protected by checksummed WAL framing and a
commit marker. Recovery ignores incomplete/uncommitted tails.

In format 6, `materialize_frozen_wals()` collapses the currently selected
frozen WAL prefix into one immutable DeltaV1 `.dat/.idx` pair. It first
publishes a metadata-only catalog generation that durably reserves a segment
ID, builds the pair without holding the mutation lock, then briefly pauses
writes to rebase and atomically switch `CURRENT`. The active WAL and WALs frozen
after the build began remain in the new catalog and take precedence by their
newer commit sequences. Replaced WAL pathnames are unlinked best-effort only
after the switch. No base segment or base-index entry is copied.

The delta index stores version metadata sorted by numeric ID plus reverse
`{id, original commit sequence}` metadata for every dense PUT row. Today these
records are loaded into the RAM overlay on open. Normal open verifies the whole
delta-index body but does not stream the complete embedding payload to recompute
its stored CRC during ordinary open.

`compact_deltas(max_runs=8, max_input_bytes=8 GiB,
max_input_entries=1_000_000)` merges only the oldest contiguous prefix when at
least two complete runs fit all budgets. It verifies the selected data payload
CRCs before publication, keeps the newest original sequence per ID (including
tombstones), reserves a fresh segment ID, atomically replaces the exact prefix,
verifies the replacement payload, and then reclaims only those source pairs
best-effort. The base and every newer
delta/WAL resource remain unchanged. A public standalone payload scrub and base
compaction are not yet implemented. Publication and old-reader behavior rely
on local POSIX filesystem semantics; network filesystems are outside the
supported durability contract.

For local POSIX storage, `locks/commit.lock` closes the race between opening an
old snapshot and unlinking its files. Readers hold a shared lease until every
mutation resource in the selected catalog is open; publication and reclamation
hold the exclusive lease through `CURRENT`, unlink, and directory sync. This
barrier currently covers WAL/delta resources, not future replaceable base
generations.

`Collection.inspect_orphans(path, min_age_seconds=3600)` provides the same
read-only plan without first opening the collection or loading its base index.
It briefly acquires the writer lease while validating `CURRENT`, its selected
catalog, and the mutation namespaces. An existing `r+` handle can instead call
`collect_orphans(dry_run=True, min_age_seconds=...)` while retaining its writer
lease. `dry_run=False` is deliberately unsupported. The CLI equivalent is
`throwdb collect-orphans --storage PATH --dry-run --min-age-seconds 3600`, and
the `--dry-run` flag is mandatory.

Temporary publications, unreferenced WALs, and complete or partial
unreferenced delta pairs are candidates; the minimum age only decides whether
a candidate is currently eligible. Reachable files, superseded catalogs
retained by policy, and unknown/do-not-touch entries are never candidates. The
report is a potentially stale diagnostic snapshot, not an authorization token
for deletion. There is no public deletion executor or automatic `maintain()`
operation yet.

`max_input_bytes` bounds the selected source-file footprint, not aggregate
physical I/O: integrity scans, copying winning rows, and replacement readback
are additional reads. `max_input_entries` is the explicit in-memory metadata
bound for this first implementation.

After a partitioned `ID -> RowRef` index becomes authoritative, the cleaner
selects segments with enough garbage, copies only rows that are still live in a
stable view, and atomically publishes the replacement segments and affected
index partitions. Untouched base and mutation segments remain in place. Old
files are reclaimed only after no reader retains the old view. A complete
snapshot rewrite is reserved for explicit schema/layout migration rather than
normal CRUD maintenance. The detailed contract is in `mutable-storage.md`.

## Compatibility policy

Format version is a storage contract, not a package version.

- readers must reject unknown versions explicitly;
- the legacy low-level reader remains available for existing benchmarks, but
  the new `Collection` path need not open format-v1/v2 collections during the
  experimental MVP;
- source NumPy arrays are the supported migration path into the current
  DeltaV1 native format 6; the reader retains strict paths for catalog/WAL-only
  format 5 and legacy native formats 3 and 4;
- migration creates a new collection and never rewrites the source arrays;
- reserved fields and feature flags must be validated rather than silently
  interpreted.

The native mutable layout is documented in `native-format-v3.md`.

## Tuning contract

Correctness and collection schema are independent of I/O tuning. Callers may
select:

```python
Collection.open(path, tuning="auto")
Collection.open(path, tuning="persisted")
Collection.open(path, tuning="conservative")
Collection.open(path, tuning={"backend": "io_uring", "workers": 2,
                              "queue_depth": 32})
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
base IDs without a population scan, compares bit-exact temporary readers that
share the already-loaded index, and publishes only after rechecking the
media/runtime/layout identity, root inode, mutation snapshot, registry salt,
and registry generation. Failure, interruption, instability, a cooperative
deadline, or `dry_run=True` leaves the registry unchanged.
