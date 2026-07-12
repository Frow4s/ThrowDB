# Native mutable numeric format

`Collection.create()` publishes a fresh numeric collection reported through
the public API as format version 3. Its internal wire headers currently use
native-format version 1. This format is intentionally independent of the
earlier immutable format-v1/v2 files: source NumPy arrays are the migration
path while the mutable format is still experimental.

## Layout

```text
collection/
  manifest.bin
  data/segment-N.dat
  ids/segment-N.ids
  index/index.bin
  wal/active.wal
  locks/writer.lock
  locks/commit.lock
```

Every manifest, base-data, ID-sidecar, and index file starts with a 4096-byte
CRC32C-protected header bound to nonzero collection and schema UUIDs. Base data
contains dense row-major embedding bytes. The matching ID sidecar contains one
little-endian `uint64` per row. Stable 64-bit `RowRef` values encode a 32-bit
segment ID and a 32-bit row number.

The base index is an open-addressed flat hash table loaded into RAM. Its values
are `RowRef`s, not global row ordinals, so later storage tiers can resolve the
same logical address without changing the public read path. The table targets
at most 70 percent occupancy. This deliberately favors lookup speed and RAM
over compactness for the first large-index tests.

## Bulk publication

The builder streams embeddings and IDs into immutable base segments. On
`finish()` it syncs every completed segment, builds and syncs the flat index,
creates and syncs an empty active WAL, then atomically renames and syncs
`manifest.bin`. A reader therefore sees either no published collection or the
complete initial collection; an unfinished builder is not readable.

## Reads and snapshot semantics

Opening a collection validates its manifest, base segment headers, ID
sidecars, index structure, and WAL identity. Committed WAL operations rebuild
an in-memory overlay. Lookup checks that overlay first: a PUT resolves to the
active WAL and a tombstone stops lookup; absent overlay entries fall through
to the base index.

All IDs in one `get` are resolved under one shared state lock before physical
I/O starts. The generic physical reader then sorts `{fd, offset}` requests,
merges eligible ranges, and scatters bytes back to caller order. Repeated IDs
and reads spanning base segments and WAL payloads preserve exact output order.

A handle captures a snapshot when it opens. A read-write handle sees its own
subsequent commits. An already-open handle in another process does not refresh
automatically; close and reopen it to observe later commits.

Handles are process-local and must not be reused after `fork()`. An inherited
builder or collection rejects operations in the child; open a fresh handle
there instead. This also prevents a child from sharing the parent's WAL append
offset or accidentally releasing its writer lease.

## Mutations, durability, and recovery

Only one read-write handle may hold `locks/writer.lock`. Every mutation batch
is validated completely before any WAL write: IDs must be unique in the batch,
`insert` requires all IDs to be absent, and `update` requires all IDs to exist.
`upsert` accepts either state. Delete is idempotent and returns one boolean per
requested ID; only existing IDs produce tombstones.

Under `locks/commit.lock`, a commit writes and syncs the checksummed BATCH,
then writes and syncs its matching COMMIT marker, then publishes the new RAM
overlay state. Recovery applies only complete BATCH/COMMIT pairs. A read-only
open ignores one incomplete active-WAL tail. A read-write open, while holding
the writer and commit leases, truncates that tail to the last committed byte
and syncs the repair before accepting new writes.

## Current limits

- Numeric IDs only (`uint64` on disk); the Python layer safely accepts
  `uint8/16/32/64`, non-negative `int8/16/32/64`, and Python integers.
- Dense fixed-width `float16` and `float32` rows.
- Local files only; remote media and a network API are later stages.
- Prefix reads currently perform a full-row physical read and then copy the
  requested prefix.
- One active WAL with no rotation or selective cleaner yet. Its PUT row count
  is bounded by the 32-bit row component of `RowRef`, and open time grows with
  mutation history until rotation is implemented.
- Read-only snapshots do not live-refresh cross-process commits.
