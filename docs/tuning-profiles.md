# Reader tuning profiles

ThrowDB keeps reader tuning separate from collection correctness. A profile can
change how physical reads are grouped and submitted, but it cannot change the
logical rows, ID index, mutation catalog, or durability state.

## Selection model

`Collection.open(path, tuning="auto")` performs a bounded, metadata-only
lookup. It does not benchmark the device and does not write to the collection.
Payload discovery stops after examining 4096 directory entries across the
base/delta/WAL directories. Overflow is observable as incomplete discovery and
makes the profile ineligible instead of extending open time with an unbounded
directory walk.

An automatic match requires the same three compatibility dimensions:

- **media**: a salted, redacted identity or compatibility class for the storage
  path serving the immutable embedding segments;
- **runtime**: reader policy ABI, operating system/kernel, architecture,
  effective CPU availability and cgroup CPU quota, and the visible io_uring
  policy;
- **layout**: native format features, ID/row layout, dtype, dimension, record
  size, and immutable segment payload capacity.

The probe assigns one reuse level to the current target:

- **exact**: ThrowDB can recognize the particular local device, host-scoped
  volume, or remote filesystem;
- **compatible**: the physical backing is opaque, but the visible storage
  class and the performance-relevant runtime assumptions are reproducible;
- **fallback**: the probe is incomplete or cannot establish even a safe
  compatibility class, so no saved profile is selected or published.

These levels are deterministic key namespaces, not a fuzzy profile score. The
probe always uses the strongest identity currently visible: an exact target is
never downgraded merely to reuse a broader compatible profile. The media key
does not include the collection path or mount point. Moving a collection inside
one identifiable filesystem therefore preserves an exact match.

An OverlayFS target is a supported compatible target. Its key deliberately
does not contain the overlay mount ID, transient filesystem ID, pod identity,
hostname, concrete CPU numbers, or upper/lower directory paths. It describes
an opaque overlay access path, not the hidden physical disk. Consequently an
overlay profile can be reused by a new process or pod with the same visible
storage/runtime class, but may become suboptimal if the unseen backing storage
changes. This can affect performance only: reader profiles cannot alter row
selection or embedding bytes.

Visible local and network filesystems remain different classes. In particular,
a CephFS mount detected as `ceph` or `fuse.ceph` cannot match a local-block or
opaque-overlay profile. If a container exposes only OverlayFS, ThrowDB cannot
truthfully infer whether its backing is local or remote and reports that
limitation through `identity_warning`. Incomplete payload discovery and failed
probes remain fallback-only.

Runtime matching includes a bounded real `io_uring_setup`/submission probe.
Before an io_uring profile is saved or selected, ThrowDB also creates its exact
queue-depth/worker topology and completes one harmless read through every ring.
If the topology is no longer available, or the target filesystem reports that
io_uring reads are unsupported, calibration skips that candidate and `auto`
falls back instead of failing the collection open. Ordinary I/O and corruption
errors are not classified as backend incompatibility. Runtime compatibility is
based on performance-relevant resources rather than a Jupyter, Airflow, pod,
process, or host identity. A changed effective CPU allocation or io_uring
capability produces a miss; reassignment to different CPU numbers with the same
effective resources does not by itself invalidate a profile. Memory is not yet
part of the key because current reader profiles do not size a memory cache.

Backend construction has a distinct native error type. If thread or ring
resources disappear after selection, only `auto` retries once with sequential
`pread`; explicit mappings and `persisted` mode surface the error. Corruption
and ordinary collection I/O errors are never caught by this retry.

Selection modes are:

- `"auto"`: use an exact or compatible reusable profile, otherwise use
  conservative defaults;
- `"persisted"`: require an exact or compatible saved match and fail on a miss;
- `"conservative"`: ignore the registry and use built-in defaults;
- a mapping: use explicit reader options and do not consult or modify the
  registry.

The selected source, reason, `profile_reuse_level`, compatibility diagnostics,
and complete effective options are exposed as `collection.tuning` and by
`throwdb inspect`.

## Registry

Profiles live in `tuning/reader-profiles.json` inside the collection. This file
is advisory and is not referenced by the data manifest or mutation `CURRENT`.
A missing, unsupported, or corrupt registry can never make `tuning="auto"`
unable to open otherwise valid data; it produces a conservative fallback with
an observable reason.

The registry is versioned, bounded, checksummed, and written under a separate
advisory lock. Publication writes and fsyncs a temporary file, atomically
replaces the registry, then fsyncs its directory. Readers do not take the write
lock and see either the previous or next complete file. Raw device serials,
WWIDs, filesystem UUIDs, remote mount sources, hostnames, paths, and mount
options are never stored. A random salt inside each registry prevents its
fingerprints from correlating devices across collections.

The first implementation stores at most 64 profiles in a 1 MiB registry. Old
profiles have no time-based expiry: they stop matching when a compatibility
input changes and can be removed explicitly.

## Bounded calibration

`throwdb tune` is the explicit operation that measures the current target. It
is never invoked by `Collection.open(..., tuning="auto")`:

```bash
throwdb tune \
  --storage /data/example \
  --max-seconds 300 \
  --max-read-gib 8 \
  --max-memory-mib 512 \
  --output tune-report.json
```

The equivalent library call is:

```python
report = throwdb.Collection.tune(
    "/data/example",
    max_seconds=300,
    max_read_bytes=8 << 30,
    max_memory_bytes=512 << 20,
    seed=20260713,
)
```

The calibrator acquires the collection's exclusive lifecycle lease before it
loads the large flat index. Writers and another calibrator therefore fail fast
for the duration of the run; ordinary immutable readers remain available. One
read-only snapshot, one loaded index, and the same open segment descriptors are
used for the whole matrix. Every candidate owns one bounded reader trial that
remains alive through correctness, warm-up, and the measured matrix. After the
bit-exact preflight, every active candidate runs one explicit unmeasured pass
through all workloads. This absorbs one-time thread-pool/ring initialization;
it is not included in throughput, CV, scoring, or baseline drift. No candidate
reopens the collection or duplicates its multi-gigabyte index.

ID input is not required. The native sampler walks a deterministic permutation
of at most 8,388,608 loaded hash slots and returns at most 1,048,576 currently
live base IDs. It filters base tombstones under the stable mutation snapshot
and never scans the complete population. A collection whose live population
exists only in mutation inserts must be rebuilt or compacted before this
sampler can tune it.

Matrix v2 contains at most eight candidates:

- sequential `pread` as the correctness and performance baseline;
- the current matching profile, exact or compatible, when all of its worker,
  queue, range, and zero-gap settings fit the bounded caps;
- up to three thread-pool sizes derived from effective affinity/cgroup CPUs,
  including a capped oversubscribed size for blocking I/O;
- available one- and two-ring io_uring topologies at bounded queue depths.

Range size stays at 1 MiB (or one complete row when a row is wider) and
`max_gap_bytes` stays zero, with a 16 MiB row/range cap. A uniform-random
sample cannot safely infer useful read amplification, especially on CephFS;
locality/range merging will be a separate workload-aware stage. Matrix v2
covers serial 1,024-row reads, parallel 1,024-row reads, and serial 10,000-row
reads. Rows and repeats are reduced deterministically when dtype, dimension,
or explicit budgets require it. Cache state is ambient. When the sample is
large enough, warm-up and measured calls receive separate slices of one
globally disjoint ID stream and candidate order rotates. The warm-up is
charged to the read and time limits even though it is not measured. Small
collections or tighter memory budgets are reported explicitly as an ambient
warm/replacement mode. ThrowDB does not evict the machine-wide page cache or
disrupt other filesystem users.

Every candidate must first return bit-identical embedding bytes. Timed calls
use the normal production read path. Selection uses the geometric mean of
paired per-repetition throughput ratios, rejects a material per-workload
throughput regression, and requires throughput CV (including paired-ratio CV)
of at most 10%. A candidate needs a 5% composite gain to replace the baseline;
within a 2% tie band the lower-resource topology wins. P95 call latency remains
descriptive because the bounded workload has too few calls for it to be a
sound percentile guardrail. Baseline drift above 20% or CV above 10% makes
the run inconclusive. Saving the measured baseline is intentional when no
candidate wins: it records that this exact or compatible target was calibrated
instead of asking to tune it on every open.

The time limit is cooperative because one already-started filesystem, index,
or payload operation cannot be cancelled safely in the current engine. No new
calibration or publication phase starts after the deadline; registry
publication also takes its advisory lock in fail-fast mode. Candidate count,
sampled slots, query rows, output memory, report size, and planned physical
bytes have hard bounds.
The read-byte limit covers correctness, explicit warm-up, and measured
embedding payload reads made by candidate calls; it does not include the
collection's one normal index load. The memory limit covers calibration
samples, permutations, query arrays, and output arrays; it
does not include the collection index, page cache, native thread stacks, or
kernel ring memory. Those resources are instead bounded by the small candidate
matrix and its capped worker/ring counts. The report includes native
`open_stats`, including index file bytes and phase timings, so the excluded
one-time open cost stays visible.
`--dry-run` returns the same report without creating or changing tuning
metadata. A stable opaque-overlay target is reusable at the compatible level,
so a normal non-dry calibration may publish its winner. The report then has
`target.profile_reusable=true`,
`target.profile_reuse_level="compatible"`, and an `identity_warning` explaining
that the hidden backing storage was not identified. Exact targets report
`profile_reuse_level="exact"` without that warning.

A target remains fallback-only when discovery is incomplete, the storage class
is unknown, or another required compatibility input cannot be probed. Dry-run
may still measure it, but the report has `target.profile_reusable=false` and
`target.profile_reuse_level="fallback"`; a non-dry run rejects it before the
expensive benchmark. On correctness failure, interruption, deadline
exhaustion, unstable load, identity change, or registry-generation conflict,
no profile is published. `--output` writes the JSON report atomically; progress
goes to stderr and the final report to stdout.

On a shared network filesystem, processes and pods with the same exact media,
runtime compatibility, and layout key can reuse one profile. A changed CPU
quota, CPU availability, kernel/reader ABI, or backend capability produces a
separate profile or a conservative fallback; task and pod identifiers do not.
Calibration and writer serialization rely on the filesystem's advisory-lock
semantics, which is the same requirement as mutable collection access.

## Installing an explicit profile

The registry layer accepts an explicitly selected set of options:

```python
import throwdb

throwdb.Collection.save_tuning_profile(
    "/data/example",
    {
        "io_backend": "io_uring",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 32,
        "max_read_range_bytes": 4 << 20,
        "max_gap_bytes": 0,
    },
)

collection = throwdb.Collection.open("/data/example")
assert collection.tuning["source"] == "persisted"
```

The equivalent low-level CLI operation is:

```bash
throwdb tuning set \
  --storage /data/example \
  --io-backend io_uring \
  --io-uring-workers 2 \
  --io-uring-queue-depth 32

throwdb tuning show --storage /data/example
throwdb tuning remove --storage /data/example
throwdb tuning reset --storage /data/example
```

`tuning set` installs already chosen values; it does not claim to have measured
them. `throwdb tune` records its matrix-v2 winner with source `calibrated-v2`.
Both paths publish through the same crash-safe registry. Existing open handles
keep their current policy until reopened.

`tuning remove` removes only the current exact or compatible target profile.
`tuning reset` removes the entire registry and all profiles; add `--force` to
recover an invalid or corrupt optional registry. Neither command touches
collection data or mutation state.

## Current boundary

Profiles currently select one policy when a collection handle opens. They do
not yet choose a different backend per request and do not implement fair
scheduling across mixed batch sizes. Those mechanisms belong in the shared C++
physical-reader scheduler so that they operate on one loaded index and one
global I/O budget.
