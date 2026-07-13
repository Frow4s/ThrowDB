# Local collection product contract

This document fixes the contract for the first user-facing ThrowDB product.
The fresh native numeric subset is implemented through the Python `Collection`
API: safe uint64 normalization, float16/float32 rows, arbitrary fixed
dimensions, semantic prefix reads, and durable batch CRUD. The public format
version is 4. UTF-8 IDs, WAL rotation, and selective cleaning remain the next
local-library slices.

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
- source NumPy arrays are the supported migration path into the current native
  format 4; the reader retains a strict fallback for legacy native format 3;
- migration creates a new collection and never rewrites the source arrays;
- reserved fields and feature flags must be validated rather than silently
  interpreted.

The native mutable layout is documented in `native-format-v3.md`.

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
