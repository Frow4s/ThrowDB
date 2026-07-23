# Streaming read contract

ThrowDB exposes two bounded read streams for format-7 numeric collections.
They solve different problems and must not be emulated by composing
`scan_ids()` with ordinary `get()` calls:

- `stream_get()` consumes an arbitrarily large ordered input of IDs in bounded
  batches. Input order and duplicate IDs are preserved.
- `scan_records()` emits every live `(ID, embedding)` pair once in physical
  `(segment_id, row_index)` order. Exact snapshot-scoped locations are
  optional.

Both operations capture one immutable logical read view. A mutation or
maintenance publication that happens after capture cannot change a later
batch. In particular, an update returns the old embedding, a later delete does
not make an ID disappear, and a later insert remains absent. Pinned files may
remain on disk until the stream closes even after maintenance publishes and
unlinks a replacement view.

## Snapshot boundary

Snapshot creation is linearized under the collection state lock. It captures:

- collection ID;
- catalog generation;
- committed sequence;
- live row count;
- immutable index and data resources;
- a resolved copy of the live mutable overlay.

An active WAL remains appendable after the lock is released. Consequently a
snapshot stores the already resolved file descriptor and payload offset for
each captured overlay PUT; it never consults the active WAL's mutable offset
vector later.

The cursor token identifies only this open stream. It is not resumable and is
not accepted by another RPC. Reopening after a transport failure creates a new
snapshot.

## Memory bounds

There are two independent limits:

- `max_snapshot_bytes` bounds retained cursor state, principally the mutable
  overlay copy and segment descriptors. It is checked before an overlay-sized
  allocation. Immutable mmaps, open file descriptors, page cache, and retained
  inode bytes are not heap and are reported separately from this estimate.
- `max_batch_bytes` bounds the user-visible batch payload: IDs, requested
  embedding columns, and optional locations. `batch_rows` is an additional
  upper bound. The effective capacity is the smaller bound.

Native lookup/planning and physical-reader buffers are `O(effective rows)` and
are independently bounded by the reader configuration or server admission
budget. No stream performs an `O(collection live count)` allocation.

Keeping yielded batches keeps their NumPy arrays alive and is therefore caller
owned memory. `read_into()` exists for reusable arrays and direct writes to a
NumPy memmap.

For an ordered lookup whose complete result must become another `.npy` file,
the embedded client can write each native batch directly into its final slice:

```python
query_ids = np.load("query-ids.npy", mmap_mode="r")
partial = "result.npy.partial"
result = np.lib.format.open_memmap(
    partial,
    mode="w+",
    dtype=np.float16,
    shape=(query_ids.size, 128),
)

with collection.stream_get(
    query_ids,
    prefix_dimensions=128,
    batch_rows=65_536,
    max_batch_bytes=64 << 20,
) as stream:
    offset = 0
    while offset != query_ids.size:
        count = stream.read_into(
            result[offset : offset + stream.batch_capacity]
        )
        if count == 0:
            break
        offset += count
    assert stream.done and offset == query_ids.size

result.flush()
del result
os.replace(partial, "result.npy")
```

`RemoteEmbeddingStream.get_into(result)` performs the corresponding bounded
loop for a remote stream. Although `result` has the full logical shape, a
memmap does not materialize the full payload in process RAM. The temporary
name is important: publish it only after clean EOF.

A complete physical-order export uses the same pattern with
`scan_records().read_into(ids, embeddings, locations)`. Allocate file-backed
arrays from `scan.snapshot_count`, pass slices no larger than
`scan.batch_capacity` locally (or `scan.max_batch_rows` remotely), and advance
only by the returned row count because a batch may stop at a segment boundary.

## Batch and failure semantics

For `stream_get()`, all IDs in the current input batch are resolved before its
output buffer is touched. A missing ID fails that batch with `NOT_FOUND`; rows
from earlier completed batches remain valid and are not rolled back. Order and
duplicates match the input exactly.

For `scan_records()`, missing IDs are impossible. Only the authoritative live
version of a physical row is emitted. Stale rows and tombstones are skipped,
and disagreement with the captured live count is corruption.

A yielded batch is complete. An I/O, corruption, cancellation, or protocol
failure never yields a partially valid batch. The stream then becomes
close-only. With the reusable-buffer APIs, a missing point-read ID leaves the
current output slice untouched, and a record scan publishes its IDs and
locations only after embedding I/O succeeds. A lower-level I/O failure may
still have modified bytes in the current embedding slice, so only the returned
row count (or batches yielded before the exception) is committed progress.
Atomic publication of a large result should use a temporary memmap followed by
a rename after successful EOF.

Writable `scan_ids().read_into()` ID and location arrays must not overlap, nor
may the ID, embedding, and location arrays passed to
`scan_records().read_into()`. `stream_get().read_into()` output must not overlap
the current or an unread part of its retained NumPy ID source. Local APIs reject
overlap when the arrays share one address mapping. The caller must also avoid
overlapping file offsets exposed through distinct memmap objects, because NumPy
cannot reliably identify aliasing across independent mappings of the same file.

Prefix dimensions are fixed when the stream opens and are read bit-exactly
from the beginning of each stored row. Invalid prefixes fail before snapshot
publication.

## Concurrency and lifetime

A stream is single-consumer. Local cursors fail fast if another thread tries to
read or close them while a batch is in flight; a rejected concurrent call does
not advance or fail the cursor. Their collection can continue serving ordinary
reads, mutations, and maintenance. Server streams use a separate admission
pool, HTTP/2 flow control, a bounded application window, and duration/idle
watchdogs. Closing a client stream cancels transport work and releases the
snapshot, admission lease, and pinned resources.

The Python input object passed to `stream_get()` must not be mutated until the
stream closes. NumPy arrays and sequences are sliced and normalized per batch;
the complete input is not converted to an internal `uint64` array. This keeps
`uint32`, `int32`, and memmap inputs bounded as well as the output.

## Remote protocol

`StreamGet` is bidirectional rather than server-streaming from one large
request. The first request opens and negotiates a snapshot. Thereafter one
non-empty ID batch produces exactly one embedding batch with contiguous
sequence and row offsets. Protocol v1 uses an application window of one: the
official client does not release the next request batch until it has consumed
the preceding response.

`ScanRecords` is server-streaming because it has no large input. Both RPCs use
the scan admission class and repeat or validate snapshot identity throughout
the stream. Existing unary `Get` remains the all-or-nothing API for results
small enough to fit one response.

Servers that do not advertise the corresponding capability fail explicitly.
Clients must not silently fall back to repeated current-view `Get` calls,
because that would weaken snapshot consistency.

## Compatibility

The first implementation supports numeric `uint64` format-7 collections with
`float16` or `float32` embeddings. Python continues accepting the existing
safe integer input dtypes and normalizes each batch to `uint64`. Older storage
formats fail with a rebuild-required error rather than using a weaker path.

UTF-8 IDs require their own packed offsets/data representation and persistent
string index. They are intentionally not represented by NumPy object arrays or
temporary surrogate integers and remain a separate storage-format extension.
