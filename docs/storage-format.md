# Experimental MVP storage format

This document describes format version 1 as implemented by the MVP. It is an
implementation contract for tests and inspection tools, not yet a compatibility
promise. All integer metadata is unsigned and little-endian. Headers are exactly
4096 bytes; bytes not listed below are reserved and currently zero.

## Directory layout and publication

```text
store/
  manifest.bin
  index/
    index.bin
  data/
    shard-000000.dat
    shard-000001.dat
    ...
  staging/
```

While a writer is active, `staging/id_ordinal.log` holds the ID-to-ordinal log.
The staging directory remains empty after a successful finish. Data shards
and the final index are made durable before `manifest.bin` is atomically renamed
into place. The existence of a valid manifest is therefore the publication
point: readers must not infer a usable store merely from the data directory.

The common 16-byte prefix of every header is:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | File-specific ASCII magic |
| 8 | 4 | Format version (`1`) |
| 12 | 4 | Header size (`4096`) |

## Manifest

`manifest.bin` is exactly 4096 bytes and has magic `THROWDBM`.

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 4 | dtype (`1` = IEEE-754 binary16) |
| 20 | 4 | embedding dimension |
| 24 | 8 | bytes per record (`dimension * 2`) |
| 32 | 8 | total record count |
| 40 | 8 | configured shard payload capacity |
| 48 | 8 | records per full shard |
| 56 | 8 | shard count |
| 64 | 8 | hash-index slot capacity |
| 72 | 8 | hash seed |
| 80 | 8 | index control-array offset |
| 88 | 8 | index slot-array offset |
| 96 | 8 | expected index file size |

## Data shards

Shard files have magic `THROWDBS`. Their header-specific fields are:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 8 | zero-based shard ID |
| 24 | 8 | global ordinal of the first record |
| 32 | 8 | records physically stored in this shard |
| 40 | 8 | bytes per record |
| 48 | 4 | embedding dimension |
| 52 | 4 | dtype |

The payload begins at offset 4096 and is a dense sequence of fixed-width rows.
There is no per-row ID or length. For an ordinal from the index:

```text
shard_id     = ordinal / records_per_shard
row_in_shard = ordinal % records_per_shard
file_offset  = 4096 + row_in_shard * record_size
```

Binary16 payload bits are stored without numerical conversion. The MVP targets
little-endian Linux and requires native-endian NumPy input.

## Immutable hash index

`index/index.bin` has magic `THROWDBI`. Its header-specific fields are:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 16 | 8 | occupied slot/record count |
| 24 | 8 | power-of-two slot capacity |
| 32 | 8 | hash seed |
| 40 | 8 | control-array offset (4096) |
| 48 | 8 | slot-array offset (4096-byte aligned) |
| 56 | 8 | total index file size |
| 64 | 8 | slot size (16) |

The header is followed by one control byte per slot (`0` empty, `1` occupied),
alignment padding, then 16-byte `(id: uint64, ordinal: uint64)` slots. Empty
versus occupied state is carried by the control array, so every possible
`uint64` ID, including zero and `UINT64_MAX`, remains valid. Readers load the
immutable table into RAM at open time.

Capacity is the smallest power of two that keeps load at or below 70%. Lookup
starts at `hash(id, seed) & (capacity - 1)` and probes linearly with wraparound.
The format-v1 hash is the SplitMix64 finalizer after adding the recorded seed and
`0x9e3779b97f4a7c15`. No ABI-dependent C++ structure is written directly.

At 600 million records, the 70% rule selects `2^30` slots. The loaded index is
therefore about 17 GiB (16 GiB of slots plus 1 GiB of controls), excluding small
allocator overhead. The staging log itself is 9.6 GB in decimal units.

## Staging log

The build-only log has magic `THROWDBL`, a 4096-byte header, and dense 16-byte
`(id, ordinal)` entries. Header offset 16 stores entry count and offset 24 stores
entry size (`16`). `finish()` scans this log to construct the flat hash index and
detect duplicate IDs. It is not needed for normal reads and is removed only
after successful publication/cleanup.

## Current integrity guarantees

Readers reject bad magic/version/header sizes, inconsistent metadata, and
truncated files. Version 1 does not yet store payload checksums, so a same-size
bit flip in an embedding may be indistinguishable from valid binary16 data.
Checksums, snapshots, deltas, tombstones, and compaction are intentionally
outside this MVP format.
