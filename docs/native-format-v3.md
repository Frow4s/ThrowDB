# Native mutable numeric formats 3 through 6

`Collection.create()` publishes a fresh numeric collection reported through
the public API as format version 6. Version 4 added a CRC32C for the complete
flat-index body in fields that were reserved and zero in version 3. Version 5
added the required `CatalogV1` manifest feature, a checksummed `CURRENT`
pointer, immutable catalog generations, and UUID-named active/frozen WALs.
Version 6 additionally requires `DeltaV1`, uses catalog wire version 2, and may
replace a frozen WAL prefix with checksummed immutable mutation-delta data and
index files. The internal base, WAL, and delta headers retain their independent
wire versions; the public version distinguishes the collection-level integrity
and lifecycle contract. This format is intentionally independent of the
earlier immutable format-v1/v2 files: source NumPy arrays remain the migration
path while the mutable format is experimental.

The current reader opens all four versions. A checksum-free version-3 index
falls back to the complete structural scan. Version 4 uses one fixed
`wal/active.wal`. Version 5 is catalog/WAL-only: it supports rotation but
`materialize_frozen_wals()` rejects it because the manifest does not require
DeltaV1. New builders write version 6. Readers that do not implement a required
manifest feature reject formats 5/6 instead of silently opening an incomplete
view.

## Layout

```text
collection/
  manifest.bin
  data/segment-N.dat
  ids/segment-N.ids
  index/index.bin
  CURRENT
  catalogs/catalog-<catalog-uuid>.bin
  wal/wal-<wal-uuid>.wal
  deltas/delta-<delta-uuid>.dat
  deltas/delta-<delta-uuid>.idx
  locks/writer.lock
  locks/commit.lock
```

The `deltas/` directory is created with every new format-6 collection. Format 5
has the catalog/WAL subset of this layout. Legacy version 3/4 collections retain
`wal/active.wal` and have neither `CURRENT` nor catalogs.

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

Before publishing a version-4/5/6 index, the builder performs the complete
structural validation and then records CRC32C over controls, zero alignment
padding, and all serialized slots. A normal open verifies that checksum while
loading the index and does not repeat the expensive per-slot structural audit.
`Collection.open(..., verify_index=True)` requests both checksum verification
and the full slot/`RowRef`/uniqueness/probe-path audit. The lower-level binding
uses the equivalent `verify_native_index=True` argument. This checksum detects
accidental corruption; it is not a cryptographic authenticity mechanism.

## Bulk publication

The builder streams embeddings and IDs into immutable base segments. On
`finish()` it syncs every completed segment, builds and syncs the flat index,
creates and syncs an empty UUID-named active WAL, publishes catalog generation
1 and `CURRENT`, then atomically renames and syncs `manifest.bin`. A reader
therefore sees either no published collection or the complete initial
collection; an unfinished builder is not readable.

`CURRENT` is exactly 4096 bytes and binds collection/schema UUIDs, catalog UUID,
generation, exact catalog size, and CRC32C of the complete catalog. A format-5
catalog uses wire version 1: a 4096-byte checksummed header followed by fixed
128-byte WAL entries. A format-6 catalog uses wire version 2: zero or more
fixed 192-byte delta descriptors precede the same WAL entries. Both record the
predecessor catalog, materialized lineage anchor, monotonic next segment ID,
and an ordered chain with exactly one active WAL at the end. Frozen entries
persist exact final sequence, PUT-row count, and file size; the active entry
derives its changing tail state by scanning the WAL. Catalog, WAL, and delta
file names are derived only from validated UUIDs, never persisted paths.

## Reads and snapshot semantics

Opening a collection validates its manifest, base segment headers, ID
sidecars, index checksum (or the complete index structure for legacy and
explicit-verify opens), `CURRENT`, the selected catalog, every delta descriptor
and index body, and every referenced WAL identity. Frozen WALs are scanned
strictly; only the final active WAL may discard a torn uncommitted tail. Delta
version records are applied oldest run first, then retained WAL commits are
applied in sequence order, so the active and any newer frozen WALs take
precedence. The result is one in-memory versioned overlay. Lookup checks that
overlay first: a PUT resolves through the captured delta/WAL resource set and a
tombstone stops lookup; absent entries fall through to the base index.

Normal open verifies each complete delta `.idx` body, including its reverse
row metadata. It checks the `.dat` header, descriptor binding, and exact file
size but deliberately does not read the potentially large embedding payload to
recompute its stored CRC32C. Payload scrubbing is not yet exposed as a normal
open option or CLI command.

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

## Catalog publication and WAL rotation

`rotate_wal()` is serialized with commits by the mutation mutex and holds the
commit lock exclusively across publication. A header-only WAL is a no-op. For
a non-empty WAL it syncs the predecessor, creates and syncs a successor whose
header names the predecessor UUID and exact previous sequence, publishes an
immutable catalog containing the frozen predecessor plus active successor,
then atomically replaces and syncs `CURRENT`. Only after that durable switch
does it replace the in-memory resource view. Any ambiguous publication error
poisons the writer handle and requires reopen.

Catalog and `CURRENT` temporary names include the new catalog UUID. A crash
before the `CURRENT` rename leaves harmless orphan files; open never scans the
directory for the numerically newest catalog. A crash after publication opens
the complete new view. Readers retain one shared owner of the mutation resource set
through physical I/O, so later catalog switches cannot invalidate an already
planned raw file descriptor.

## DeltaV1 materialization

`materialize_frozen_wals()` operates only on format 6 and is serialized with
another materialization. With no frozen WAL it is a successful no-op. Otherwise
it replaces the complete frozen prefix visible when the operation starts; it
never rewrites a base data segment or the flat base index.

Publication deliberately uses two short metadata transactions:

1. Under the mutation and commit locks, publish a catalog generation that
   advances `next_segment_id`. The logical resource view is unchanged, but the
   selected delta segment ID is now durably reserved and cannot be reused after
   a crash.
2. Release the mutation lock, strictly rescan the captured frozen WAL prefix,
   collapse each ID to its newest version in that prefix, and write/sync the
   immutable delta pair. Normal CRUD and WAL rotation may continue during this
   build.
3. Reacquire the mutation lock, rebase on the latest catalog, and verify that
   the selected frozen prefix and materialized lineage are unchanged. Newer
   frozen WALs and the active WAL are retained.
4. Publish another immutable catalog and atomically switch `CURRENT`. This
   generation appends the DeltaV1 descriptor, advances the materialized
   sequence/terminal-WAL lineage, and removes only the selected WAL prefix.
   The in-memory view switches only after durable publication.
5. Best-effort unlink the replaced WAL pathnames and sync `wal/`. An unlink
   failure does not roll back the published logical view; it leaves an orphan
   that a future resource sweeper may remove.

Both immutable files begin with a checksummed 4096-byte wire-version-1 header
and redundantly bind the catalog descriptor, collection/schema UUIDs, exact
sequence interval, source WAL lineage, segment ID, counts, sizes, and body
checksums:

- `delta-<uuid>.dat` stores dense bit-exact PUT rows after its header. Rows are
  ordered by ascending ID after collapse; tombstones consume no payload row.
- `delta-<uuid>.idx` stores 32-byte version records sorted strictly by ID. Each
  record contains the ID, original commit sequence, PUT payload row or the
  tombstone sentinel, and state. After zero alignment padding, one 16-byte
  reverse record `{id, original sequence}` exists for every PUT in payload-row
  order. The complete index body, including padding and reverse records, is
  covered by CRC32C.

Readers pin open file descriptors while physical reads are in flight. After
the `CURRENT` switch, newer WAL versions continue to shadow the delta, while
already-running readers may finish through their old descriptors even if the
WAL pathnames have been unlinked. This behavior is currently part of the local
POSIX-filesystem contract only.

## Current limits

- Numeric IDs only (`uint64` on disk); the Python layer safely accepts
  `uint8/16/32/64`, non-negative `int8/16/32/64`, and Python integers.
- Dense fixed-width `float16` and `float32` rows.
- Local files only; remote media and a network API are later stages.
- Prefix reads currently perform a full-row physical read and then copy the
  requested prefix.
- Manual crash-safe WAL rotation is implemented, with automatic rotation only
  before the 32-bit per-WAL PUT-row limit. Format 6 can manually materialize a
  frozen prefix and best-effort unlink its WALs; format 5 cannot.
- Delta version metadata is still loaded into the RAM overlay. There is no
  delta-run merge, payload-scrub command, partitioned authoritative index,
  selective base compaction, or general orphan-resource collector yet.
- Materialization permits writes during its data build, but briefly pauses
  mutations for the initial segment-ID reservation and final catalog switch.
- Only local POSIX publication/open-file/unlink semantics are supported;
  network filesystems and object storage are not yet part of the durability
  contract.
- Read-only snapshots do not live-refresh cross-process commits.
