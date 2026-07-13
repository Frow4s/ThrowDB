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
- generic physical reads spanning base segments and multiple WAL payloads,
  with one shared resource-set pin retained through physical I/O.

The current collection can retain multiple frozen WALs and exactly one active
WAL. Immutable mutation segments, partitioned index runs, cross-process resource
reclamation, and the selective cleaner below remain the next integration
slices. Rotation alone does not reduce replay, overlay RAM, descriptor count,
or disk usage. The exact shipped MVP layout and limits are in
`native-format-v3.md`.

## Logical view and resource layout

A reader captures one immutable `CollectionView` at request start:

```text
CollectionView
  collection/schema identity
  visible commit sequence
  segment catalog
  active/frozen WAL resolvers
  immutable index partitions and small runs
  RAM overlay captured for this view
```

The eventual directory layout extends the fresh native base:

```text
collection/
  manifest.bin, index/, data/, ids/ # fresh immutable native base
  CURRENT                           # points to exactly one immutable catalog
  catalogs/
  wal/
  segments/
  mutation-index/
  sidecars/                         # future reverse/cleaner metadata
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

Readers own a shared reference to their view and its open files. Resources
removed from a newer catalog are reclaimed only after the final older view is
released. This is both the snapshot-isolation mechanism and the protection
against segment-ID ABA.

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

### Current catalog-WAL overlay mode

The versioned RAM overlay rebuilt from the catalog WAL chain shadows the fresh
flat base index. Both layers return `RowRef`, and reads, CRUD, catalog
publication, plus WAL rotation are implemented. Selective cleaning is still
disabled because there are no immutable mutation segments or authoritative
partition-level durable index publications yet.

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

Rotation does not rewrite the base. It has two publication transactions so an
acknowledged commit can never live only in an unreferenced WAL:

1. Pause the writer under its metadata lock and freeze the current WAL at a
   known sequence.
2. Create and sync the successor WAL with the frozen WAL's UUID and final
   sequence as predecessor metadata.
3. Publish a catalog/CURRENT generation containing both the frozen WAL and the
   new active WAL. Only then may the writer append or acknowledge new commits.
4. Offline, collapse obsolete versions inside the frozen range where safe and
   write immutable mutation segments plus affected partition runs.
5. Under the publication lock, rebase on the latest catalog, preserve the
   active WAL and every post-freeze resource, and atomically publish a second
   generation replacing only the frozen inputs.
6. Retire the frozen WAL after older views release it.

The current format-5 slice implements steps 1–3 and deliberately retains every
frozen WAL. Steps 4–6 are the next lifecycle slice; calling `rotate_wal()` must
not be described as compaction or checkpointing until that materialization and
reclamation path exists.

Every immutable mutation segment carries checksummed reverse metadata for each
dense row: its exact ID and original commit sequence. This may be one compact
row-metadata table or catalog-bound sidecars, but it is part of the segment's
validated identity. Without it the cleaner cannot prove whether a physical row
is still authoritative.

Segment size is configurable and much smaller than the original 16 GiB build
shards; a practical starting range is hundreds of MiB to roughly 1 GiB. The
cleaning policy can therefore recover space in bounded units.

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
