# Mutable local storage architecture

This document defines the target selective-cleaning mutation layer for a
logical ThrowDB collection. It deliberately does **not** require periodically
copying the whole base collection. New writes eventually become smaller
immutable resources, and background cleaning rewrites only selected segments
whose live-data ratio justifies the work.

The fresh native numeric MVP now implements:

- checked `SegmentId`, `RowRef`, tombstone, and partition-routing types;
- CRC32C-protected native headers and WAL frames;
- immutable base segments plus ID sidecars and a flat `ID -> RowRef` index;
- one writer lease, a commit lock, RAM overlay, public batch CRUD, and strict
  active/frozen WAL recovery;
- checksummed `CURRENT`, immutable catalog generations, UUID-named WAL chains,
  and manual plus hard-limit WAL rotation;
- DeltaV1 materialization of a frozen WAL prefix into immutable data/index
  pairs, with durable segment-ID reservation and an atomic catalog switch;
- bounded compaction of an oldest contiguous DeltaV1-run prefix, preserving
  original commit sequences and tombstones while reclaiming only replaced
  delta files;
- generic physical reads spanning base segments, mutation deltas, and multiple
  WAL payloads,
  with one shared resource-set pin retained through physical I/O.

The current format-6 collection can retain immutable mutation-delta runs,
multiple frozen WALs, and exactly one active WAL. Manual materialization removes
a frozen WAL prefix from the current catalog without rewriting the base, and
manual delta compaction can replace several oldest runs with one bounded run;
format 5 remains catalog/WAL-only and rejects both operations. Partitioned
index runs, base-segment selective cleaning, orphan deletion, and automatic
maintenance remain later integration slices. A public orphan dry-run plan is
available today. Materialization and delta compaction reduce retained resources
and replay work, but all surviving delta version metadata is still rebuilt into
a RAM overlay on open. The exact shipped MVP layout and limits are in
`native-format-v3.md`.

## Logical view and resource layout

A reader captures one immutable `CollectionView` at request start:

```text
CollectionView
  collection/schema identity
  visible commit sequence
  segment catalog
  immutable delta and active/frozen WAL resolvers
  RAM overlay captured for this view
```

The current format-6 directory layout extends the fresh native base:

```text
collection/
  manifest.bin, index/, data/, ids/ # fresh immutable native base
  CURRENT                           # points to exactly one immutable catalog
  catalogs/
  wal/
  deltas/                           # immutable delta-UUID.dat/.idx pairs
```

Future partitioned index and cleaner resources will extend this layout without
changing `CURRENT` as the only selected-view pointer.

`CURRENT` is the sole metadata publication point. It names a catalog by an
immutable generation and UUID and records its expected size/checksums. Open
must never search the directory and guess that the numerically newest catalog
is active. Files written before a failed `CURRENT` switch are harmless orphans.

Publication order is:

1. Write every new segment, index run/partition, and WAL resource, then sync it.
2. Write and sync a new immutable catalog.
3. Rename the catalog to its final derived name and sync `catalogs/`.
4. Write and sync a UUID-derived `CURRENT.<catalog-uuid>.tmp`.
5. Atomically rename it to `CURRENT` and sync the collection directory.
6. Publish the equivalent in-memory view.

`locks/commit.lock` is also the mutation reader-open barrier. A reader holds it
shared from immediately before reading `CURRENT` until the complete referenced
catalog, delta indexes, WAL offsets, and data/WAL file descriptors are pinned.
A publisher or reclaimer holds it exclusively across the durable `CURRENT`
switch, in-memory view swap, retired-resource unlink, and directory sync. Thus
an opener either finishes pinning the old view before unlink or waits and opens
the new view. POSIX keeps unlinked inodes alive until older readers release the
final open descriptor.

The global order is `writer.lock -> lifecycle mutex -> mutation mutex ->
commit.lock -> state mutex`. A future collector must own the writer lease before
the commit lock; otherwise it could mistake an unlocked in-progress delta build
for an orphan. The current barrier protects mutation resources only. Future
base generations require a separate base-view barrier so loading a large base
index does not stall every WAL commit.

Segment IDs are durably reserved before a long delta build and are never reused,
which protects against segment-ID ABA. Network filesystem and object-store
semantics are not yet part of this contract.

## Current lifecycle safety slice

The native tests have private lifecycle checkpoints at the reader-open barrier
and at important materialization/compaction transitions: an immutable catalog
is durable before `CURRENT`, `CURRENT` is durable, an output segment ID is
reserved, a delta output pair is durable, and a selected source is immediately
before or after a successful unlink. Checkpoint context identifies the
operation, catalog generation, resource kind, and source ordinal. These hooks
are internal test seams, not an installed header, public failpoint facility, or
supported runtime configuration. With no test callback installed they do not
change the storage state machine.

Fork-based tests terminate a child with `_exit` at those checkpoints. That
models an abrupt **process crash**: destructors do not run, while completed
kernel calls and previously synced files remain visible to the surviving
system. It is not a complete **power-loss** simulation. In particular, an
after-unlink checkpoint occurs after a successful unlink syscall but before the
later directory `fsync`; it does not claim that the pathname removal has reached
stable media. The catalog-before-`CURRENT`, durable-`CURRENT`, and durable-delta
checkpoints are placed after their documented file and directory sync barriers.
Power-loss guarantees still require fault testing on the target local
filesystem and storage stack.

A strict read-only inventory validates `CURRENT` and exactly the immutable
catalog it selects, then inspects entries in the `CURRENT`-like root,
`catalogs/`, `wal/`, and `deltas/` namespaces. Its categories are:

- reachable files in the currently selected operational view;
- superseded catalogs;
- temporary publications;
- unreferenced WALs;
- complete or partial unreferenced delta pairs; and
- unknown/do-not-touch entries.

Operational reachability is deliberately **CURRENT-only**. The selected
catalog, its named WALs, and its named delta pairs are reachable. A
`previous_catalog_id` records lineage but does not make its predecessor a
runtime snapshot, so older catalog files may be reported as superseded. Exact
lowercase UUID filename grammar is required for orphan classification;
malformed names, symlinks, and non-regular entries fail safe to
unknown/do-not-touch.

The inventory powers two public dry-run surfaces. The class method
`Collection.inspect_orphans(path, min_age_seconds=3600)` briefly takes the
writer lease and performs the lightweight scan without loading the base index.
An already-open mutable handle may call
`collection.collect_orphans(dry_run=True, min_age_seconds=...)`. The CLI form
is `throwdb collect-orphans --storage PATH --dry-run --min-age-seconds 3600`;
`--dry-run` is required.

Only temporary publications, unreferenced WALs, and complete or partial
unreferenced delta pairs are candidates. The age threshold separates eligible
from deferred candidates. Reachable files, superseded catalogs retained by
policy, and unknown/do-not-touch entries are never candidates.

An inventory result is diagnostic data, not an authorization token. It can be
stale, and a path can change after inspection. Any future deleter must hold the
writer lease and `commit.lock` exclusively, repeat no-follow type and identity
checks immediately before each unlink, apply age/durability policy, and sync the
affected directory. The current public slice performs no deletion. A deletion
executor and automatic `maintain()` operation are still not implemented;
materialization and delta compaction only reclaim the exact source resources
replaced by their own publication.

## Row references and tombstones

Every physical live row is addressed by one 64-bit value:

```text
bits 63..32  segment_id: 1 .. 0xfffffffe
bits 31..0   zero-based row_index
```

Segment IDs are scoped to one collection, allocated monotonically, and never
reused. Zero and segment ID `0xffffffff` are invalid/reserved. A row index uses
the complete `uint32` range; catalog geometry must still prove that it is below
the referenced segment's row count.

A delete is not a special row-reference bit pattern. Index state is
out-of-band:

```text
empty       -> this index layer has no opinion; continue to an older layer
PUT+RowRef  -> resolve the row in the captured view
DELETE      -> stop lookup and report the ID missing
```

Each catalog segment entry also has a nonzero 128-bit file UUID. A `RowRef`
resolves its numeric segment ID only inside the captured view, which verifies
the expected UUID and file metadata.

Fresh base segments receive IDs while they are built. With
`records_per_segment = R`, logical build row `n` maps to:

```text
segment_id = 1 + n / R
row_index  = n % R
```

Older immutable collections are rebuilt from their source NumPy arrays rather
than adapted with a global-ordinal compatibility layer.

## Index transition and steady state

Two explicit modes avoid pretending that selective cleaning works before its
prerequisites exist.

### Current DeltaV1 catalog-overlay mode

The versioned RAM overlay rebuilt from catalog delta runs followed by the
retained WAL chain shadows the fresh flat base index. Delta `.idx` records are
applied oldest run first; retained frozen/active WAL commits then win by commit
sequence. Both layers return `RowRef`, and reads, CRUD, catalog publication, WAL
rotation, frozen-prefix materialization, and bounded oldest-prefix delta
compaction are implemented. The delta index
body contains durable sorted version metadata and reverse `{id, sequence}` rows,
but those records are still loaded into RAM on open. Selective base cleaning is
disabled because no authoritative partition-level index publication exists yet.

### Authoritative partitioned-RowRef mode

The durable index maps every logical ID to `{sequence, PUT/DELETE, RowRef}`.
Partitioning uses configurable high bits of the existing seeded ID hash; the
local hash table can use disjoint low bits. A starting profile such as 4096
partitions is configuration, not a storage-format constant.

Commits do not rebuild a whole partition. Recent entries live in the RAM/WAL
overlay and then small immutable per-partition runs. Runs are folded into a
larger partition only when their count, lookup cost, or garbage ratio crosses a
budget. This matters for a workload that randomly updates 1% of 600 million
IDs: it touches almost every partition, but should write only the changed
entries rather than thousands of complete hash tables.

## Mutation commit path

One process owns the writer lease. Under its mutation mutex it:

1. Normalizes IDs and validates shapes/dtypes without writing.
2. Resolves `insert`/`update`/`delete` preconditions against one latest view.
3. Rejects duplicate IDs within the batch.
4. Converts API intent to final full-row `PUT` or tombstone `DELETE` records.
5. Appends the complete checksummed BATCH and syncs it.
6. Appends its COMMIT marker immediately after the BATCH and syncs again.
7. Atomically publishes the complete batch in the RAM overlay.
8. Acknowledges the caller.

The two durability barriers are intentional: after recovery sees an intact
COMMIT, the preceding BATCH was already durable. A crash after the first barrier
leaves an uncommitted BATCH that can be discarded. Group commit may amortize
these barriers across several batches later, but it must preserve the same
"all batches durable before any corresponding marker is durable" ordering.

The WAL records final state, not whether the caller named the operation insert,
update, or upsert. If the process fails after the COMMIT sync but before its
response, the outcome is intentionally ambiguous to the caller: recovery must
replay the durable commit.

An active WAL is assigned a catalog segment ID. Its resolver builds a bounded
row-to-payload-offset table during recovery. Its `row_index` is the cumulative
PUT ordinal in that WAL; DELETE consumes no ordinal. The scanner exposes each
payload offset and the resolver increments the ordinal across batches. A WAL
must rotate before it would exceed 2^32 PUT rows, and its catalog entry records
the exact resolved row count. Thus a WAL row can participate in the same
`RowRef` model even though BATCH/COMMIT metadata separates dense payload
extents.

## WAL v1 binary contract

All integers are little-endian. CRC means CRC32C Castagnoli with reflected
polynomial `0x82f63b78`, initial state and final xor `0xffffffff`. The checksum
of `123456789` is `0xe3069283`. Checksum fields are treated as zero while their
own checksum is calculated.

### File header

The file header is exactly 4096 bytes:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | `THROWDBW` |
| 8 | 4 | WAL format version, `1` |
| 12 | 4 | header size, `4096` |
| 16 | 16 | logical collection UUID |
| 32 | 16 | immutable schema UUID |
| 48 | 16 | this WAL file UUID |
| 64 | 16 | predecessor WAL UUID, or zero |
| 80 | 8 | previous committed sequence |
| 88 | 8 | first sequence, exactly previous + 1 |
| 96 | 8 | embedding record size |
| 104 | 4 | dimension |
| 108 | 4 | dtype code |
| 112 | 4 | dtype item size |
| 116 | 4 | ID type, initially canonical uint64 |
| 120 | 4 | dense row-major layout code |
| 124 | 4 | checksum algorithm, CRC32C = `1` |
| 128 | 4 | flags, zero |
| 132 | 4 | CRC32C of the complete header |
| 136 | 8 | required feature bits, zero |
| 144 | 8 | optional feature bits, zero |
| 152 | 8 | creation time in Unix nanoseconds, informational |
| 160 | 3936 | reserved, zero |

Collection, schema, and WAL UUIDs are nonzero. The header schema must satisfy
`record_size == dimension * item_size`; the first implementation supports
uint64 IDs and float16/float32 dense rows. A WAL header is immutable.

### BATCH frame

Every BATCH starts with a 128-byte header:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | `TDBWBAT1` |
| 8 | 4 | frame version, `1` |
| 12 | 4 | header size, `128` |
| 16 | 8 | commit sequence |
| 24 | 8 | operation count |
| 32 | 8 | PUT count |
| 40 | 8 | schema record size |
| 48 | 8 | operation-table offset, exactly `128` |
| 56 | 8 | operation-table bytes, count times `16` |
| 64 | 8 | dense PUT-payload offset |
| 72 | 8 | PUT-payload bytes |
| 80 | 8 | complete BATCH size, aligned to 8 |
| 88 | 16 | owning WAL UUID |
| 104 | 4 | checksum algorithm, CRC32C = `1` |
| 108 | 4 | CRC32C of this 128-byte header |
| 112 | 4 | flags, zero |
| 116 | 12 | reserved, zero |

The operation table contains exactly 16 bytes per operation:

| Entry offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 1 | opcode: PUT `1`, DELETE `2` |
| 1 | 7 | reserved, zero |
| 8 | 8 | exact uint64 ID; zero and UINT64_MAX are ordinary IDs |

After the table, complete PUT rows are packed densely in PUT-table order.
DELETE consumes no payload row. Zero padding extends the BATCH to an 8-byte
boundary and is covered by its checksum. IDs must be unique within the BATCH.

### COMMIT marker

A 64-byte marker immediately follows its BATCH:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | `TDBWCOM1` |
| 8 | 4 | marker version, `1` |
| 12 | 4 | marker size, `64` |
| 16 | 8 | sequence |
| 24 | 8 | absolute BATCH file offset |
| 32 | 8 | BATCH size |
| 40 | 4 | CRC32C of all stored BATCH bytes |
| 44 | 4 | checksum algorithm, CRC32C = `1` |
| 48 | 4 | CRC32C of this marker |
| 52 | 4 | flags, zero |
| 56 | 8 | reserved, zero |

The marker must exactly bind the immediately preceding BATCH. Sequence numbers
are contiguous; zero and UINT64_MAX are reserved so overflow fails explicitly.
The v1 format maxima are 1,000,000 operations and 1 GiB per BATCH. They are
immutable recovery limits, not tuning: a writer may enforce stricter request
limits but cannot raise them. All size arithmetic is checked before allocation.

## Recovery boundary

Recovery fixes file size once and binds the decoded header to catalog-provided
collection, schema, WAL/predecessor UUIDs and sequence metadata. It exposes a
batch only after validating its marker and full BATCH CRC. Its returned
`valid_end` is the safe truncation point before append.

An active WAL may discard one final uncommitted/torn append:

- fewer than 128 bytes remain where a new BATCH header would start;
- a valid complete header declares a frame extending past EOF;
- a complete BATCH has no marker or fewer than 64 marker bytes;
- the final append retained its full file length but has a torn or zero-filled
  BATCH/marker and no intact CRC-valid COMMIT marker at or after the tail.

A frozen WAL must have no ignored tail. An intact COMMIT marker after damaged
data proves that the damage is interior, so even active recovery fails rather
than dropping later commits. A marker that passes its own CRC but has bad
binding, any committed BATCH checksum mismatch, unknown opcode, nonzero
reserved/padding byte, duplicate ID, bad geometry, or sequence gap is likewise
corruption. The scanner may inspect aligned tail positions only to detect an
intact marker; it never resynchronizes and replays past damaged data. Under the
writer lease, an active WAL is truncated to `valid_end` and synced before
appending.

The scanner validates the complete stable image before its first visitor call.
Each operation span is callback-scoped. A rebuilt overlay becomes visible only
after replay completes and the scan returns successfully.

Active-tail repair is a crash-consistency rule, not a complete media-corruption
oracle. Without an independently durable acknowledged high-watermark, a final
marker damaged after acknowledgement is indistinguishable from that marker
being torn before its sync completed and may be truncated. Frozen WALs remain
strict. If the product later promises detection of post-ack marker loss, the
format must add a replicated marker or an independent durable high-watermark.
CRC32C also protects against accidental corruption, not a hostile party that
can deliberately forge valid frame bytes; such a forged tail can make recovery
fail closed but cannot make an unvalidated batch replay.

Recovery of a catalog WAL chain additionally verifies collection/schema IDs,
each predecessor UUID, and exact sequence continuity between files:
`next.previous_sequence == previous.last_sequence`. A missing, duplicated, or
reordered WAL is corruption even when every individual file scans correctly.

## WAL rotation and immutable segments

Rotation does not rewrite the base. It freezes one non-empty active WAL and
publishes its successor in one catalog transaction, so an acknowledged commit
can never live only in an unreferenced WAL:

1. Pause the writer under its metadata lock and freeze the current WAL at a
   known sequence.
2. Create and sync the successor WAL with the frozen WAL's UUID and final
   sequence as predecessor metadata.
3. Publish a catalog/CURRENT generation containing both the frozen WAL and the
   new active WAL. Only then may the writer append or acknowledge new commits.

Format 5 stops there and deliberately retains every frozen WAL. Format 6 adds
`materialize_frozen_wals()` (CLI: `throwdb materialize --storage PATH`) with two
short catalog publications around an unlocked data build:

1. Under the mutation/commit locks, publish a metadata-only generation that
   advances `next_segment_id`. This durably reserves the output segment ID even
   if the process crashes or the later build becomes an orphan.
2. Release the mutation lock. Strictly rescan the captured complete frozen WAL
   prefix and collapse each ID to its newest `{sequence, PUT/DELETE}` in that
   interval. Concurrent commits and rotations continue during this phase.
3. Stream bit-exact live PUT rows into a synced immutable
   `deltas/delta-<uuid>.dat`. Write and sync its matching `.idx`, whose 32-byte
   version records are sorted by ID and whose 16-byte reverse `{id, sequence}`
   records correspond one-for-one with dense payload rows. CRC32C binds both
   bodies through the catalog descriptor.
4. Reacquire the mutation lock, verify the selected prefix and lineage still
   match, and rebase on the latest catalog. Retain the active WAL and every WAL
   frozen after the build started, so their newer sequences continue to shadow
   the delta.
5. Publish a second catalog and atomically replace `CURRENT`, removing only the
   selected WAL prefix and adding the delta descriptor. Switch the in-memory
   view after the durable pointer change.
6. Best-effort unlink the replaced WAL pathnames. Failure leaves harmless
   unreferenced files for a future sweeper; it does not roll back the selected
   catalog. Already-open POSIX descriptors remain valid for older reads.

Thus materialization is not a base compaction: it bounds the selected mutation
history without copying any base embedding or flat-index entry.

Format 6 also exposes `compact_deltas()` (CLI: `throwdb compact-deltas`). It
selects the oldest contiguous prefix of at least two DeltaV1 runs subject to
explicit run-count, input-byte, and input-entry budgets. Before reserving an
output segment, it verifies every selected source payload CRC. It then performs
a bounded k-way merge of the ID-sorted indexes, retaining the greatest original
commit sequence for each ID and retaining tombstones. PUT payload rows are
copied bit-exactly into one new immutable DeltaV1 pair. A metadata-only catalog
generation reserves the output segment ID; a second generation atomically
replaces exactly the selected prefix after revalidation and a complete output
payload checksum pass. Newer unselected
deltas and every retained WAL remain in place and continue to shadow the merged
run. Only after the durable `CURRENT` switch are the selected source pairs
unlinked best-effort. A process that crashes before the switch leaves an orphan
or an unused segment-ID gap; after the switch, recovery selects the compacted
view.

The byte budget is the catalog footprint of selected source files, not a bound
on total physical I/O. Source payloads are scanned for integrity, winning PUT
rows are read again for the copy, and the output payload is read back before
publication. The entry budget bounds retained version metadata; the current
writer still holds bounded input/output vectors rather than implementing a
fully streaming external merge.

This compaction reduces run/file count, duplicated delta payload, and reopen
work without reading or copying the immutable base. It deliberately cannot
discard tombstones while the flat base index remains a fallback. Reclaiming
garbage inside base segments still requires an authoritative partitioned
RowRef index in a later format. There is currently no automatic size/age policy
for invoking lifecycle operations.

### Current DeltaV1 limits

- Delta version records still populate the full mutation overlay in RAM when a
  collection opens; `.idx` is durable metadata, not yet an on-demand or
  partitioned lookup structure.
- Writes and WAL rotation continue while the delta files are built, but the
  initial segment-ID reservation and final rebase/catalog switch briefly pause
  mutations.
- Normal open checks each delta data header, exact size, index geometry, reverse
  rows, and the complete `.idx` body CRC. It intentionally does not stream the
  complete `.dat` embedding payload to verify its stored CRC; no public payload
  scrub operation exists yet.
- Delta compaction is explicit and prefix-only. Its budgets bound one operation,
  but no automatic policy currently prevents repeated materializations from
  accumulating runs between calls. Base segments are not compacted.
- Publication, old-view safety, and post-switch WAL unlink currently rely on
  local POSIX `rename`, `fsync`, open-descriptor, and unlink semantics. CephFS,
  other network filesystems, and object storage require a separately validated
  backend protocol.

## Selective segment cleaner

The cleaner runs only after the partitioned RowRef index is authoritative. It
chooses old segments using measured garbage ratio, age, read heat, and available
I/O/space budgets. For a stable view at sequence `C`, each candidate row is
copied only if the index at `C` still contains exactly the same ID, original
sequence, and old `RowRef`. Newer commits stay in a higher overlay and shadow
any row copied by the cleaner.

The cleaner writes replacement segments and copy-on-write index resources,
then switches them together in one catalog generation. Before publication it
acquires the metadata lock (or performs an equivalent generation CAS), rebases
on the latest catalog, and preserves all WALs/runs committed after `C`. An index
mapping is relocated only if the latest mapping still equals the candidate's
old `{sequence, RowRef}`; a newer mapping is left untouched. If candidate
resource identities changed, that cleaning attempt is discarded or rebuilt.
An all-garbage segment is simply omitted. Unselected segments are untouched.
There is no normal operation that copies every embedding in a 1.5 TB
collection.

Every fresh base segment already has an ID-by-row sidecar because its data
payload contains only embeddings. For 600 million numeric IDs all sidecars are
about 4.8 GB decimal. The initial index build verifies duplicate IDs, and its
reader validates the RowRef bijection (a one-bit-per-row bitmap is about 75 MB
for 600 million rows).

A complete collection rewrite remains an explicit administrative operation for
schema/layout migration or major workload reorganization, not routine garbage
collection.

Tombstone reclamation follows the lookup mode. In the current overlay mode a
DELETE must remain until every older source that could contain the ID is
removed or durably shadowed; dropping it early would resurrect a base row. In
authoritative partitioned mode, partition absence is itself the final
"missing" answer and never falls back to an older base mapping. A DELETE run
can be folded into that absence only when the new authoritative partition is
published atomically and no catalog reachable by the new view can reintroduce
an older mapping.

## Adaptive policy boundary

Correctness does not depend on cleaner thresholds, segment size, partition
count, WAL rotation size, read backend, or queue depth. Those are observable,
bounded policies. Initial defaults should be deterministic; later calibration
and workload logs may tune them per local NVMe, network storage, and request
shape without changing the persisted state machine.
