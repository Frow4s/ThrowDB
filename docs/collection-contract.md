# Local collection product contract

This document fixes the intended contract for the first user-facing ThrowDB
product. It is a design target, not a description of the current format-v1
implementation. Format-v1 remains readable while the contract is implemented
through a versioned format-v2 writer and a new Python `Collection` API.

## Product boundary

A logical collection is a typed mapping:

```text
user ID -> fixed-schema embedding row
```

The local product provides record-level create/read/update/delete operations.
Approximate nearest-neighbour search, similarity scoring, replication, and a
network protocol are separate product layers.

Physical base snapshots remain immutable and atomically published:

```text
create -> append batches -> finish -> read-only snapshot
```

`finish()` is the initial publication point. A failed build must not replace a
readable snapshot. Record mutations are represented by committed delta records
over the base, not by overwriting active shard bytes. Compaction builds a new
immutable base and atomically switches the active view while existing readers
retain their previous view.

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
- destroying the collection concurrently with an active method is outside the
  lifetime contract; the owner must keep it alive until callers return.

`prefix_dimensions=N` returns the first `N` values of every row and requires
`1 <= N <= collection.dimension`. The API guarantee does not promise that a
row-major local backend performs only `N * item_size` bytes of physical I/O.
Filesystem page size can dominate random prefix reads. The same API can later
use a prefix-optimized physical layout without changing callers.

## Write contract

The initial Python surface is intended to be:

```python
schema = throwdb.CollectionSchema(
    id_type="utf8",
    dimension=1024,
    dtype="float16",
)

with throwdb.Collection.create(path, schema=schema) as collection:
    collection.add(ids, embeddings)
```

Writers accept bounded batches and may spill build metadata to disk. Successful
context-manager exit calls `finish()`; exceptional exit must not publish the
snapshot. Readers never depend on writer-process memory.

After initial publication, the record API is:

```python
collection.insert(ids, embeddings)   # every ID must be absent
collection.update(ids, embeddings)   # every ID must exist
collection.upsert(ids, embeddings)   # explicit insert-or-replace
deleted = collection.delete(ids)     # boolean result per requested ID
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

Each read captures one committed collection view at request start. Concurrent
mutation or compaction cannot make a single result mix old and new rows.

## Mutation and compaction model

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

Compaction merges the base and committed deltas into a new snapshot, validates
it, and atomically publishes a new view. Old files are reclaimed only after no
reader retains the old view. This keeps the optimized immutable read path while
providing CRUD semantics.

## Compatibility policy

Format version is a storage contract, not a package version.

- readers must reject unknown versions explicitly;
- the format-v2 reader must continue to read valid format-v1 numeric/float16
  snapshots;
- new schema features are written only in v2 or later;
- migration creates a new snapshot and never rewrites the only readable copy;
- reserved fields and feature flags must be validated rather than silently
  interpreted.

The exact format-v2 binary layout is a separate design step. This document fixes
the semantics that layout must represent.

## Tuning contract

Correctness and collection schema are independent of I/O tuning. Callers may
select:

```python
Collection.open(path, tuning="auto")
Collection.open(path, tuning={"backend": "io_uring", "workers": 2,
                              "queue_depth": 32})
```

Automatic tuning must be observable, bounded, and overrideable. The product
will expose the selected profile and the reason for request-level decisions.
Opening a collection must not unexpectedly run a destructive or long cold-cache
benchmark.
