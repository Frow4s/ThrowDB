# Storage format version 2 design

This document defines the implementation target for format version 2. Version 2
represents the local collection contract without changing the dense ordinal data
model. It adds explicit schema, typed IDs, format capabilities, snapshot identity,
and room for integrity metadata. Version 1 remains a supported read format.

All metadata integers and numeric payload values are little-endian. Every file
starts with the existing 16-byte common prefix:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | File-specific ASCII magic |
| 8 | 4 | Format version (`2`) |
| 12 | 4 | Header size (`4096`) |

Unless enabled by a declared feature bit, reserved header bytes must be zero.
Readers reject unknown required feature bits and unknown enum values. Unknown
optional features may be ignored only when the base row and index semantics stay
valid.

## Stable enum values

| Enum | Value | Meaning |
| --- | ---: | --- |
| dtype | 1 | IEEE-754 binary16 (`float16`) |
| dtype | 2 | IEEE-754 binary32 (`float32`) |
| dtype | 3 | bfloat16 (reserved for the next implementation) |
| ID type | 1 | unsigned 64-bit integer |
| ID type | 2 | byte-exact UTF-8 |
| row layout | 1 | dense row-major values |
| index kind | 1 | flat linear-probing uint64 index |
| index kind | 2 | flat fingerprint index with string verification |
| hash algorithm | 1 | seeded SplitMix64 integer hash from format v1 |
| hash algorithm | 2 | SipHash-2-4 over string bytes |
| checksum algorithm | 0 | no checksum |
| checksum algorithm | 1 | CRC32C stored in the low 32 bits of a u64 |

Quantized embeddings require a future schema extension for scale and zero-point
metadata. They must not reuse the scalar dtype enum without that metadata.

## Directory layout

```text
collection/
  manifest.bin
  index/
    index.bin
  ids/
    strings.bin       # UTF-8 collections only
  data/
    shard-000000.dat
    shard-000001.dat
    ...
  staging/            # build-only, empty after publication
```

`manifest.bin` remains the publication point. Every published file carries the
same random 128-bit snapshot ID so readers cannot accidentally assemble a
collection from files belonging to different builds.

## Manifest v2

The manifest keeps all collection-wide schema and file geometry:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 4 | dtype |
| 20 | 4 | embedding dimension |
| 24 | 4 | dtype item size in bytes |
| 28 | 4 | ID type |
| 32 | 4 | row layout |
| 36 | 4 | index kind |
| 40 | 4 | hash algorithm |
| 44 | 4 | checksum algorithm for this header |
| 48 | 8 | bytes per full embedding row |
| 56 | 8 | record count |
| 64 | 8 | configured shard payload capacity |
| 72 | 8 | records per full shard |
| 80 | 8 | shard count |
| 88 | 8 | index slot capacity |
| 96 | 8 | expected index file size |
| 104 | 8 | expected string-arena file size, zero for uint64 IDs |
| 112 | 8 | required feature bits |
| 120 | 8 | optional feature bits |
| 128 | 8 | snapshot ID low word |
| 136 | 8 | snapshot ID high word |
| 144 | 8 | hash seed/key low word |
| 152 | 8 | hash seed/key high word |
| 160 | 8 | index control-array offset |
| 168 | 8 | index slot-array offset |
| 176 | 8 | index slot size |
| 184 | 8 | string-arena payload offset, zero for uint64 IDs |
| 192 | 8 | creation time in Unix nanoseconds (informational) |
| 200 | 8 | manifest header checksum; checksum field is treated as zero |

Validation includes:

- `record_size == dimension * item_size`;
- dtype and item size agree;
- ID type, index kind, and hash algorithm form a supported combination;
- all size/offset arithmetic is overflow-checked;
- string fields are zero for numeric collections;
- shard geometry and index occupancy can represent exactly `record_count` rows.

## Data shard v2

Shard magic remains `THROWDBS`. Payload is still a dense row-major array.

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 8 | snapshot ID low word |
| 24 | 8 | snapshot ID high word |
| 32 | 8 | zero-based shard ID |
| 40 | 8 | first global ordinal |
| 48 | 8 | record count in this shard |
| 56 | 8 | bytes per full row |
| 64 | 4 | embedding dimension |
| 68 | 4 | dtype |
| 72 | 4 | row layout |
| 76 | 4 | payload checksum algorithm |
| 80 | 8 | payload byte count |
| 88 | 8 | payload checksum, or zero when disabled |
| 96 | 8 | header checksum |

Whole-payload checksums are optional because verifying a 16 GiB shard during
every open would be unacceptable. They support explicit scrubbing. A later
required feature may add block-checksum sidecars for verification during reads.

Prefix reads do not change this layout. A row prefix begins at the ordinary row
offset and has `prefix_dimensions * item_size` logical bytes. The planner may
read additional bytes when coalescing ranges or satisfying filesystem alignment.

## Index v2

Index magic remains `THROWDBI`. Controls remain one byte per slot (`0` empty,
`1` occupied), followed by aligned fixed-width slots.

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 8 | snapshot ID low word |
| 24 | 8 | snapshot ID high word |
| 32 | 4 | ID type |
| 36 | 4 | index kind |
| 40 | 4 | hash algorithm |
| 44 | 4 | control encoding (`1` = one-byte empty/occupied) |
| 48 | 8 | occupied slot count |
| 56 | 8 | power-of-two slot capacity |
| 64 | 8 | hash seed/key low word |
| 72 | 8 | hash seed/key high word |
| 80 | 8 | control-array offset |
| 88 | 8 | slot-array offset |
| 96 | 8 | slot size |
| 104 | 8 | total index file size |
| 112 | 8 | expected string payload bytes, zero for uint64 IDs |
| 120 | 8 | header checksum |

### Numeric slots

Numeric slots remain 16 bytes:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | uint64 ID |
| 8 | 8 | ordinal |

This permits direct migration of the format-v1 index representation.

### UTF-8 slots

String slots are 32 bytes:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | 64-bit fingerprint |
| 8 | 8 | ordinal |
| 16 | 8 | byte offset in the string-arena payload |
| 24 | 4 | string byte length |
| 28 | 4 | reserved, zero |

Lookup hashes the query bytes, probes matching fingerprints, and then compares
the complete stored bytes. Fingerprint equality alone is never sufficient.
String length is bounded by both `uint32` and a smaller configurable product
limit enforced by the writer/server.

## String arena v2

UTF-8 collections add `ids/strings.bin` with magic `THROWDBA`.

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 8 | snapshot ID low word |
| 24 | 8 | snapshot ID high word |
| 32 | 8 | payload byte count |
| 40 | 4 | payload checksum algorithm |
| 44 | 4 | reserved, zero |
| 48 | 8 | payload checksum, or zero when disabled |
| 56 | 8 | header checksum |

The payload starts at offset 4096 and is the concatenation of ID byte sequences.
Lengths and offsets live in occupied index slots, so no delimiter or terminator
is stored. Readers validate every referenced interval before exposing the index.
An implementation may memory-map the arena instead of copying all string bytes
into anonymous RAM.

## Mutability boundary

The files specified above form one immutable base snapshot. Record CRUD does not
rewrite these shard or index files in place. A logical mutable collection adds a
separately versioned mutation layer:

```text
collection root/
  CURRENT                 # atomically published collection-view descriptor
  snapshots/<snapshot>/   # one immutable format-v2 base
  deltas/                  # committed PUT/DELETE segments
  wal/                     # active recovery log
```

Each mutation batch has a monotonic commit sequence, checksummed framing, and an
explicit commit marker. Readers capture one `CURRENT` view and consult committed
deltas from newest to oldest before falling back to the base index. DELETE is a
tombstone; PUT carries a complete fixed-schema embedding row.

Compaction creates another complete format-v2 base from a stable committed
sequence, validates it, and atomically replaces `CURRENT`. It never modifies the
only readable snapshot. The exact WAL, delta-segment, and view-descriptor binary
layouts receive their own specification before CRUD implementation; they are
not encoded into unused manifest bytes implicitly.

## Version-1 compatibility

The common-header decoder reads magic, version, and header size before choosing
a version-specific decoder.

Format-v1 manifests are projected into the internal schema as:

```text
id_type       = uint64
dtype         = float16
item_size     = 2
row_layout    = row_major
index_kind    = uint64_flat
hash_algorithm= splitmix64_v1
snapshot_id   = absent
features      = 0
```

The v2 reader continues to validate v1 files with their original v1 rules. It
must not apply v2 offsets to a v1 header.

## Current implementation status

The numeric v2 codec, writer, and reader are implemented for `uint64` IDs and
`float32` rows. Manifest, index, and every shard carry one nonzero snapshot ID;
their schema, geometry, version, snapshot, offsets, sizes, and reserved bytes
are checked when opening the snapshot. Numeric index ordinals must form a
bijection onto the stored rows, and every occupied slot must be reachable by
the declared linear-probing algorithm. Checksum algorithms, feature bits, and
string-arena fields are currently fixed to zero and any unsupported value is
rejected. Exact duplicate-ID verification for externally corrupted index files
is deferred to the explicit checksum/integrity tooling; the supported writer
rejects duplicates before publication.

The `float16` writer deliberately continues to emit the byte-identical v1
layout so existing production workflows do not receive a silent format change.
The version-dispatching reader supports v1 float16 and numeric v2 float16/float32
schema, while the current public writer selects v2 for float32. This rollout can
later move float16 creation to v2 without changing the collection API.

Physical prefix reads, checksums, UTF-8 IDs, and the mutation layer remain later
steps. The current Python prefix API reads full dense rows before copying the
requested columns.

## Incremental implementation order

1. Introduce internal schema/enums and version-dispatching decoders while still
   writing v1; prove all existing v1 tests remain unchanged.
2. Implement v2 numeric/float16 writing and reading with the same payload and
   index algorithms; add cross-version corruption tests.
3. Add the high-level Python `Collection` numeric-ID normalization layer while
   keeping the low-level uint64 zero-copy path.
4. Generalize item-size arithmetic and add bit-exact float32 round trips.
5. Add prefix reads to the row-major planner and Python output validation.
6. Specify mutation WAL, delta segments, view publication, and compaction; then
   implement numeric CRUD.
7. Add string staging, arena construction, collision-verified indexing, UTF-8
   bindings, and string CRUD.
8. Add explicit scrub/checksum tooling separately from the normal open path.
