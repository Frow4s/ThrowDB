# Mutable local storage architecture

This document defines the implemented selective-cleaning mutation layer for a
logical ThrowDB collection. It deliberately does **not** require periodically
copying the whole collection. New writes become bounded immutable resources,
and an explicit cleaner pass rewrites at most one selected segment whose
garbage and reclaimable-byte thresholds justify the work. No background thread
invokes lifecycle maintenance.

The fresh native numeric MVP now implements:

- checked `SegmentId`, `RowRef`, tombstone, and partition-routing types;
- CRC32C-protected native headers and WAL frames;
- immutable embedding segments plus `{id, sequence}` sidecars;
- one writer lease, a commit lock, RAM overlay, public batch CRUD, and strict
  active/frozen WAL recovery;
- checksummed `CURRENT`, immutable catalog generations, UUID-named WAL chains,
  and manual plus hard-limit WAL rotation;
- format-6 DeltaV1 materialization and prefix compaction for compatibility;
- format-7 frozen-WAL materialization into immutable segments and one
  partitioned mutation run;
- an authoritative format-7 index made of one immutable snapshot per partition
  plus small active run sections;
- bounded selective folding of complete partition state without copying
  embedding payloads;
- bounded segment cleaning with same-sequence RowRef relocation and a cleaner
  run that fences every row of the retired source;
- format-aware orphan planning and fail-closed deletion for WALs, delta pairs,
  segment pairs, snapshots, runs, and temporary publications;
- generic physical reads spanning immutable segments and multiple WAL payloads,
  with one shared resource-set pin retained through physical I/O.

Format 7 retains multiple frozen WALs, exactly one active WAL, immutable
materialized segments, partition snapshots, and partitioned mutation/cleaner
runs. Materialization, folding, and cleaning publish a new catalog but do not
unlink retired resources. They become unreachable only after the catalog
switch and are reclaimed by a later fresh, locked orphan pass. This separates
logical publication from namespace reclamation and preserves readers that
pinned the previous view. Format 6 keeps its flat-index/DeltaV1 lifecycle, and
format 5 remains catalog/WAL-only. The historical formats 3 through 6 are
specified in `native-format-v3.md`.

One writable handle serializes materialization, folding, cleaning, and orphan
passes through its lifecycle mutex. Ordinary commits and compatible WAL
rotations may continue during long data-build portions; each operation
revalidates its indexed/catalog anchor before final publication.

## Logical view and resource layout

A reader captures one immutable `CollectionView` at request start:

```text
CollectionView
  collection/schema identity
  visible commit sequence
  segment catalog
  authoritative partition snapshots and active run sections
  active/frozen WAL resolvers and current RAM overlay
```

The format-7 layout is:

```text
collection/
  manifest.bin
  CURRENT                           # points to exactly one immutable catalog
  catalogs/
  segments/                         # segment-UUID.dat + segment-UUID.ids
  index/partitions/                 # one authoritative snapshot per partition
  index/runs/                       # immutable mutation and cleaner runs
  wal/                              # frozen prefix + exactly one active WAL
  locks/writer.lock
  locks/commit.lock
  locks/view.lock
```

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

`locks/commit.lock` serializes mutation publication. Format 7 additionally uses
`locks/view.lock` as the reader-open/reclamation barrier. A reader holds it
shared until every selected segment FD, snapshot mmap, run mmap, and WAL FD is
pinned, then releases the pathname lease. An orphan reclaimer holds both locks
exclusively during its fresh scan, no-follow identity check, unlink, and
directory sync. Thus an opener pins the old view before unlink or waits and
opens the new view. POSIX keeps an unlinked inode alive until the last older
reader releases it.

The global order is `writer.lock -> lifecycle mutex -> mutation mutex ->
commit.lock -> view.lock -> state mutex`. The collector follows the same order,
so it cannot mistake an in-progress lifecycle output for an orphan.

Segment IDs are durably reserved before a long delta build and are never reused,
which protects against segment-ID ABA. Network filesystem and object-store
semantics are not yet part of this contract.

## Current lifecycle safety slice

The native tests have private lifecycle checkpoints at the reader-open barrier
and at important reservation, resource-output, catalog-publication, and orphan
unlink transitions. Checkpoint context identifies the operation, catalog
generation, resource kind, and source ordinal. These hooks are internal test
seams, not an installed header, public failpoint facility, or supported runtime
configuration. With no test callback installed they do not change the storage
state machine.

Fork-based tests terminate a child with `_exit` at those checkpoints. That
models an abrupt **process crash**: destructors do not run, while completed
kernel calls and previously synced files remain visible to the surviving
system. It is not a complete **power-loss** simulation. In particular, an
after-unlink checkpoint occurs after a successful unlink syscall but before the
later directory `fsync`; it does not claim that the pathname removal has reached
stable media. The catalog-before-`CURRENT`, durable-`CURRENT`, and durable
resource-output checkpoints are placed after their documented file and
directory sync barriers.
Power-loss guarantees still require fault testing on the target local
filesystem and storage stack.

A strict read-only inventory validates `CURRENT` and exactly the immutable
catalog it selects, then inspects only product-owned namespaces. The format-7
categories are:

- reachable files in the currently selected operational view;
- superseded catalogs;
- temporary publications;
- unreferenced WALs;
- complete or partial unreferenced segment pairs;
- unreferenced partition snapshots and index runs; and
- unknown/do-not-touch entries.

Operational reachability is deliberately **CURRENT-only**. The selected
catalog and every segment, partition snapshot, run, and WAL it names are
reachable. A
`previous_catalog_id` records lineage but does not make its predecessor a
runtime snapshot, so older catalog files may be reported as superseded. Exact
lowercase UUID filename grammar is required for orphan classification;
malformed names, symlinks, and non-regular entries fail safe to
unknown/do-not-touch.

The inventory powers two public planning surfaces. The class method
`Collection.inspect_orphans(path, min_age_seconds=3600)` briefly takes the
writer lease and performs the lightweight scan without loading the base index.
An already-open mutable handle may call
`collection.collect_orphans(dry_run=True, min_age_seconds=...)`. The CLI
requires exactly one of `--dry-run` and `--execute`.

Superseded catalogs, temporary publications, unreferenced WALs, complete
unreferenced segment pairs, partial unreferenced segment singletons, and
unreferenced snapshots/runs are format-7 candidates. A complete pair is
selected and prevalidated as one budget unit. A partial singleton lets a later
pass converge after an earlier pair deletion stopped between its two unlinks;
if either half is still named by CURRENT it classifies as reachable instead.
The age threshold separates eligible from deferred candidates. Reachable files
and unknown/do-not-touch entries are never candidates.

An inventory result is diagnostic data, not an authorization token. Execute
mode never consumes it: while holding the writer lease and `commit.lock`
exclusively it performs a fresh scan, deterministically selects within hard
file/byte budgets, and compares no-follow `{device,inode,mode,size,mtime,ctime}`
identity immediately before `unlinkat`. Missing or changed entries are skipped;
successful unlinks are followed by an `fsync` of every affected namespace.
`max_files` and `max_bytes` bound selection and unlink work, not the validated
catalog itself. The local MVP therefore has a separate fail-closed inventory
guard: a scan may contain the path count implied by the active catalog plus at
most 65,536 uncataloged entries. Exceeding that bound aborts before any unlink.
This prevents an unexpectedly polluted namespace from driving unbounded scan
memory or lock time; a resumable streaming/cursor collector is a later
extension.
The contract assumes a trusted local POSIX collection directory: POSIX has no
conditional unlink-by-inode primitive that could defeat a hostile process
renaming pathnames outside ThrowDB's locks.

`maintain(dry_run=...)` is an explicit deterministic pass, not a background
thread. Format 7 considers, in order, rotation of a non-empty active WAL,
bounded materialization of the oldest frozen prefix, selective partition
folding, cleaning of at most one segment, and bounded orphan collection. Each
action is attempted at most once. Format 6 uses the older four-action sequence
without cleaning. `max_actions` is hard; `max_seconds` is cooperative and is
checked only between durable primitives. Publication failures remain fail-fast,
while individual orphan unlink/fsync failures are represented in the report.

The normalized thresholds and budgets are repeated in every maintenance report
so a pass is reproducible. The wall-clock limit is not a cancellation deadline:
an already-started durable action always runs to a safe terminal state.

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

## Index modes and format-7 steady state

Format 6 is the compatibility mode: a versioned RAM overlay rebuilt from
DeltaV1 runs and retained WALs shadows one flat base index. It supports CRUD,
rotation, frozen-prefix materialization, and bounded oldest-prefix DeltaV1
compaction, but it cannot safely retire base segments.

Format 7 makes the durable partitioned index authoritative. It maps every live
logical ID to `{sequence, RowRef}`; a tombstone may appear in a run, while
absence from an authoritative snapshot after all active run sections have been
consulted is the final missing answer. Partitioning uses configurable high bits
of the seeded ID hash. The number of partitions is persisted collection
geometry, not a process-local tuning choice.

Commits do not rebuild a partition. Recent entries live in the RAM/WAL overlay
and then in small immutable per-partition run sections. A selective fold reads
one complete partition snapshot and all of that partition's active sections,
retains the greatest original sequence per ID, and writes one replacement
snapshot. Other partitions in the same run stay active. This matters for a
workload that randomly updates 1% of 600 million IDs: materialization writes
only changed rows, and folding touches bounded index partitions rather than
copying a 1.5-TB payload.

### Format-7 catalog invariants

The public format-7 manifest requires both `SegmentCatalogV1` and
`PartitionedRowRefV1`. Its immutable catalog records the generation and UUID,
next monotonic segment ID and index epoch, indexed sequence/live count,
partition geometry, every selected segment/snapshot/run descriptor, terminal
materialized WAL identity, and the remaining WAL chain.

- segment descriptors have unique nonzero file UUIDs and RowRef segment IDs;
  initial rows use sequence zero and derived sidecars remain within the
  descriptor's sequence interval;
- exactly one snapshot descriptor exists for every partition, in partition
  order;
- run epochs are strictly increasing, and each catalog `active_partitions`
  bitmap is a nonempty subset of the immutable file's `covered_partitions`;
- an active run section is newer than the selected snapshot for that partition;
  a section cleared by a fold can no longer participate in lookup;
- a cleaner run additionally binds the exact retired segment UUID, RowRef
  segment ID, row count, and IDs-payload CRC;
- the retained WAL chain begins immediately after `indexed_sequence`, has exact
  predecessor/sequence continuity, and ends in exactly one active WAL.

Open validates these invariants, the `CURRENT` binding, resource identities,
file geometry, and index checksums before exposing a view. It never scans the
directory for a plausible newer generation.

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
   unreferenced files for the explicit orphan collector; it does not roll back
   the selected catalog. Already-open POSIX descriptors remain valid for older
   reads.

Thus materialization is not a base compaction: it bounds the selected mutation
history without copying any base embedding or flat-index entry.

`materialize_frozen_wals(max_wals=..., max_input_bytes=...)` may select only
the oldest contiguous frozen prefix fitting both limits. The legacy no-argument
call remains unlimited. If the oldest frozen WAL does not fit, the operation is
a no-op with `materialization_budget_blocked=true`; it never silently exceeds a
maintenance budget.

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

Format-6 compaction reduces run/file count, duplicated delta payload, and reopen
work without reading or copying the immutable base. It deliberately cannot
discard tombstones while the flat base index remains a fallback and therefore
cannot reclaim garbage inside base segments. Format 7, described below,
provides the authoritative partitioned RowRef index and selective cleaner.
Explicit `maintain()` supplies deterministic triggers; there is intentionally
no background invocation.

### Format-6 DeltaV1 limits

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
- Delta compaction is prefix-only. Explicit `maintain()` applies its bounded
  trigger policy, but no background worker invokes it. Format-6 base segments
  are not compacted.
- Publication, old-view safety, and post-switch WAL unlink currently rely on
  local POSIX `rename`, `fsync`, open-descriptor, and unlink semantics. CephFS,
  other network filesystems, and object storage require a separately validated
  backend protocol.

## Format-7 materialization and selective folding

Format 7 uses the same active/frozen WAL protocol but a different immutable
destination. `materialize_frozen_wals()` selects the oldest contiguous frozen
prefix fitting its WAL-count and input-byte budgets. It collapses each ID to
the greatest sequence in that prefix, orders winning PUT payload reads by their
source `RowRef`, and streams the bit-exact rows into one or more
`segment-<uuid>.dat/.ids` pairs. The sidecar stores `{id, original sequence}`
for every dense row. A single immutable mutation run stores ID-sorted
`{id, sequence, PUT RowRef/DELETE}` records grouped by partition.

The operation first reserves the necessary monotonic segment IDs and index
epoch in a durable metadata-only catalog generation. The long read/copy/write
phase does not hold the mutation mutex, so commits and rotation may continue.
Final publication validates the frozen prefix and indexed anchor, rebases on
the latest WAL chain, then atomically advances `indexed_sequence` and removes
only the selected WAL references. The retired WAL pathnames remain on disk as
unreferenced files for a later orphan pass.

`compact_deltas()` performs a selective fold in format 7. A candidate budget
charges the complete current partition snapshot plus every active run section
for that partition; a partial fold is forbidden. The merge keeps the greatest
sequence, validates equal-sequence cleaner relocations, and omits tombstones
because the new snapshot is authoritative. It writes and syncs a complete
replacement snapshot before atomically clearing the folded partition's active
bits. A partially consumed run remains referenced with its remaining bitmap; a
fully consumed run is removed from the catalog and later collected as an
orphan. Embedding segments are never read by index folding.

## Selective segment cleaner

`clean_segments()` is available only when the format-7 partitioned RowRef index
is authoritative. One call scans complete immutable sidecars within hard
candidate-count, sidecar-byte, source-byte, output-byte, and memory budgets. It
selects at most one candidate satisfying minimum garbage rows, garbage ratio,
and reclaimable bytes. Selection is deterministic: greater reclaimable bytes,
then greater garbage ratio/count, then lower segment ID.

For the indexed view at sequence `C`, a candidate row is copied only when the
index still contains exactly the same ID, original sequence, and source
`RowRef`. A newer indexed winner is recorded as a fence; authoritative absence
becomes a tombstone fence at the source sequence. Commits newer than `C` stay in
the WAL overlay and shadow the cleaner output.

Execution reserves replacement segment IDs and one index epoch, copies live
rows in bounded source-order batches, verifies the complete source payload CRC,
and writes a cleaner run with exactly one record per source row. Its descriptor
binds the retired segment UUID, RowRef segment ID, row count, and sidecar CRC;
this is what permits a same-sequence mapping to move to a replacement RowRef
without accepting an arbitrary conflict. Final publication removes the source
descriptor, adds zero or more replacement segments, and adds the cleaner run in
one catalog switch. An all-garbage segment therefore needs no replacement data
segment. Unselected segments are untouched. There is no normal operation that
copies every embedding in a 1.5-TB collection.

Every fresh base segment already has an ID-by-row sidecar because its data
payload contains only embeddings. For 600 million numeric IDs all sidecars are
about 4.8 GB decimal. The initial index build verifies duplicate IDs, and its
reader validates the RowRef bijection (a one-bit-per-row bitmap is about 75 MB
for 600 million rows).

A complete collection rewrite remains an explicit administrative operation for
schema/layout migration or major workload reorganization, not routine garbage
collection.

Tombstone reclamation follows the lookup mode. In format-6 overlay mode a
DELETE must remain until every older source that could contain the ID is
removed or durably shadowed; dropping it early would resurrect a base row. In
authoritative partitioned mode, partition absence is itself the final
"missing" answer and never falls back to an older base mapping. A DELETE run
can be folded into that absence only when the new authoritative partition is
published atomically and no catalog reachable by the new view can reintroduce
an older mapping.

The dry-run path performs the same bounded scan and reports its candidate
without publishing resources. Each result includes
`next_start_after_segment_id`; reusing it avoids restarting every call at the
lowest segment ID. It is a performance cursor only, not durable correctness
state, and losing it merely repeats earlier scans. An executing `maintain()`
remembers scan progress on its live `Collection` handle even when no candidate
was found. Separate CLI processes can pass the reported cursor back through
`--clean-start-after-segment-id`. Logical publication does not
unlink the retired source. The later orphan collector performs that work under
the view barrier after the configured minimum age.

## Adaptive policy boundary

Correctness does not depend on cleaner thresholds, segment size, partition
count, WAL rotation size, read backend, or queue depth. Those are observable,
bounded policies. The current reader uses saved device/runtime/layout profiles
for backend topology and deterministic request-level fairness inside the chosen
backend. Cleaner and maintenance thresholds remain explicit. Later workload
logs may tune them without changing the persisted state machine.
