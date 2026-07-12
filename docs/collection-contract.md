# Local collection product contract

This document fixes the intended contract for the first user-facing ThrowDB
product. It is a design target, not a description of the current format-v1
implementation. Format-v1 remains readable while the contract is implemented
through a versioned format-v2 writer and a new Python `Collection` API.

## Product boundary

A collection is an immutable, atomically published mapping:

```text
user ID -> fixed-schema embedding row
```

The local product retrieves embeddings by ID. Approximate nearest-neighbour
search, similarity scoring, online upsert/delete, replication, and a network
protocol are separate product layers and are not part of the local collection
contract.

Collections are built as snapshots:

```text
create -> append batches -> finish -> read-only snapshot
```

`finish()` is the publication point. A failed build must not replace a readable
snapshot. A later mutable or server product may build a new version in the
background and atomically switch the active version; it must not require
in-place mutation of an active snapshot.

## Collection schema

The following fields are immutable for the lifetime of a snapshot:

| Field | Initial contract |
| --- | --- |
| ID type | `uint64` or byte-exact UTF-8 |
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

Numeric IDs cover every `uint64` value, including zero and `UINT64_MAX`.

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
