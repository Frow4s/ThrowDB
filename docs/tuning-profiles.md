# Reader tuning profiles

ThrowDB keeps reader tuning separate from collection correctness. A profile can
change how physical reads are grouped and submitted, but it cannot change the
logical rows, ID index, mutation catalog, or durability state.

## Selection model

`Collection.open(path, tuning="auto")` performs a bounded, metadata-only
lookup. It does not benchmark the device and does not write to the collection.
Payload discovery stops after examining 4096 directory entries across legacy
base data, format-6 deltas, format-7 `segments/`, and WAL directories. Any of
those files can serve embedding bytes. Overflow is observable as incomplete
discovery and makes the profile ineligible instead of extending open time with
an unbounded directory walk.

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
The current `throwdb-reader-v2` ABI includes the shared fair-coordinator
semantics, so a profile measured against the earlier admission/runtime behavior
cannot match silently after upgrading.

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

The registry is versioned, bounded, checksummed, and written under two advisory
locks. Every create/save/remove/reset first takes `locks/tuning.lock`, then
`tuning/reader-profiles.lock`. Publication writes and fsyncs a temporary file,
atomically replaces the registry, then fsyncs its directory. A complete S3
checkpoint takes `writer.lock` and then `locks/tuning.lock` for its full
inventory/upload/publication window. The global order is therefore
`writer.lock -> locks/tuning.lock -> tuning/reader-profiles.lock`; a
registry-only operation never acquires `writer.lock` after either tuning lock.
If no optional tuning metadata exists, checkpoint locking creates neither the
`tuning/` directory nor a registry payload. Readers do not take either write
lock and see the previous or next complete file. Raw device serials, WWIDs,
filesystem UUIDs, remote mount sources, hostnames, paths, and mount options are
never stored. A random salt inside each registry prevents its fingerprints
from correlating devices across collections.

The barrier is an advisory collection protocol. All processes that can mutate
tuning metadata or create S3 checkpoints must run a version that understands
`locks/tuning.lock`. Concurrent mixed-version backup/tuning operation with an
older ThrowDB build is unsupported; quiesce both operations while upgrading
every worker that can access the collection. This is required because an old
process cannot be made to honor a lock introduced by a newer wheel.

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
remains alive through correctness, screening, and (for finalists) verification.
After the bit-exact preflight, every active candidate runs an explicit
unmeasured pass before each stage in which it participates. This absorbs
one-time thread-pool/ring initialization and stage cache effects; it is not
included in throughput, CV, scoring, or baseline drift. No candidate reopens
the collection or duplicates its multi-gigabyte index.

ID input is not required. The native sampler returns at most 4,194,304 live IDs
after inspecting at most 33,554,432 bounded index locations. The larger cap
lets a 512 MiB calibration keep the default screening and verification streams
disjoint even for 2 KiB records; the actual request still shrinks with the
memory limit. Format 6 samples
its loaded flat base slots and filters them through the stable mutation view.
Format 7 samples the current WAL overlay, active run sections, and authoritative
partition snapshots while deduplicating IDs and rechecking the logical view.
Its exact integer quotas are proportional to raw source capacity, with a
one-slot mutable-layer reservation for very small samples; run quota is then
distributed across every active run instead of favoring the newest run.
Snapshot and run positions are seed-stratified, so sparse hash-table capacity
does not suppress base IDs. Active-entry counts are cached while runs are
opened, and sampling traverses only bounded active-partition bitmaps instead of
rescanning every partition. It therefore includes live format-7 inserts that
do not exist in an initial segment and never needs a complete population scan.

Matrix v3.1 contains at most twelve candidates. The report keeps
`configuration.matrix_version=3`, the `two_stage_v3` workload identifier, and
the persisted `calibrated-v3` source for wire and registry compatibility; v3.1
is a bounded selection refinement, not a new persisted-profile format:

- sequential `pread` as the correctness and performance baseline;
- the current matching profile, exact or compatible, when all of its worker,
  queue, range, and zero-gap settings fit the bounded caps;
- thread-pool sizes derived from effective affinity/cgroup CPUs, including
  bounded saturation probes up to 32 workers on sufficiently provisioned
  runtimes;
- available one-ring io_uring topologies at queue depths 16, 32, and 64 plus a
  bounded two-ring topology at queue depth 32.

Range size stays at 1 MiB (or one complete row when a row is wider) and
`max_gap_bytes` stays zero, with a 16 MiB row/range cap. A bounded
source-stratified pseudo-random sample cannot safely infer useful read
amplification, especially on CephFS;
locality/range merging will be a separate workload-aware stage. The short
screening stage measures every candidate with point, 32-row, serial and
parallel 1,024-row, and serial 10,000-row requests. It retains at most five
finalists for a longer verification stage. When the short screening stage
cannot produce reliable CPU counters, one finalist slot is reserved for the
lowest-resource topology within 10% of the best screening-eligible throughput
score. A raw leader that failed screening guardrails cannot narrow that band;
when every candidate is uncertain, the probe falls back to all measured
candidates. This prevents an efficient near-leader from being removed before
the authoritative CPU-aware stage. Verification increases the number of calls
and adds both a concurrent small/large fairness workload and a repeated-ID hot
32-row workload. If the initial verification would retain the sequential
baseline, matrix v3.1 spends at most one pre-budgeted backfill slot on
the highest-scoring screening-eligible candidate that was not among the
original finalists, using topology cost and name as deterministic tie-breakers.
The backfill candidate and a fresh sequential control run interleaved through
the complete verification workload. The candidate's top-level
`pairing_reference` stores that control's scenarios and resource summary, and
authoritative paired-throughput and latency comparisons use it instead of an
earlier baseline epoch. The backfill is therefore a new paired measurement,
not promotion of a short screening result. Its decision and compact reference
metadata are reported under `candidate_discovery.verification_backfill`.

Rows and verification repeats
are reduced deterministically when dtype, dimension, or explicit budgets
require it. Before every screening, verification, backfill, or recheck stage,
ThrowDB asks the OS to discard clean cached pages for the collection's current
embedding payload and WAL files with per-file `POSIX_FADV_DONTNEED`. This is
file-scoped and never drops the machine-wide page cache. The stage warm-up runs
after that advice, so it still initializes executors and intentionally warms
the dedicated `hot_small` pool. When the sample is large enough, screening,
verification, and hot queries receive separate ID pools; warm-up and measured
ambient calls are globally disjoint both within and across stages, while hot
calls intentionally reuse only their reserved pool. Candidate order rotates.
Warm-up is charged to the read and time limits even though it is not measured.
Small collections or tighter memory budgets are reported explicitly as an
ambient warm/replacement mode. The advice is best effort and can make other
readers of the same collection observe cold payload pages, so tuning remains an
explicit operator action rather than an automatic collection-open side effect.
Every attempt is reported under `workload.stage_cache_preparation`.

The complete authoritative matrix runs in a fresh `exec` worker instead of in
the embedding application's Python process. The parent keeps its calibration
snapshot open as the stable publication anchor, while the child reopens one
read-only index snapshot and performs sampling, correctness, screening,
verification, optional backfill, and optional recheck. This temporary second
index mapping is the cost of strict I/O attribution with the current native
API; candidates still share the worker's single loaded index. There is no
in-process fallback when worker startup, protocol, or isolation fails.

Every candidate must first return bit-identical embedding bytes. Timed calls
use the normal production read path. Performance remains the geometric mean of
paired per-repetition throughput ratios. A candidate is rejected for a
material per-workload regression, throughput or paired-ratio CV above 10%, and
for a P99 regression above 25% on a latency workload with at least 32 samples.
Before every screening, verification, backfill, or recheck selection pass, the
top-level derived score, guardrail, CPU/tail, topology, and Pareto fields are
cleared and recomputed from that pass's authoritative scenarios. The nested
screening report remains available for diagnosis, but its derived eligibility
cannot leak into final scoring.

Matrix v3.1 treats 10-20% candidate or paired-gain CV as borderline when every
paired throughput ratio in the affected scenario is at least 1.05. If those
are the candidate's only failing guardrails, it runs exactly one pre-budgeted
additional paired observation for that scenario, including the baseline and
the other current finalists so repetition vectors remain aligned. The merged
result is scored once more; the report records the attempt under
`candidate_discovery.stability_recheck` and the merged scenario as
`stability_rechecked`. There is no unbounded retry loop. A candidate outside
the 10-20% band, with any paired gain below 5%, or with another guardrail
failure receives no stability override.

The mixed workload rejects small-request starvation only when both tests fail:
small-request P99 is greater than 2x the same candidate's standalone-small P99
and greater than 1.25x the sequential baseline's mixed-workload P99. Either
signal alone is descriptive and does not reject a candidate. Large-request
throughput below 80% of that candidate's standalone-large throughput remains
an independent fairness failure. The minimum small/large distinction and at
least 10 small-request samples are part of the bounded matrix contract; a read
or memory limit that cannot preserve that coverage is rejected instead of
saving an under-tested parallel profile. The sequential baseline is still
measured as the standalone and absolute control, but fairness is enforced on
the parallel candidates it is intended to distinguish.

Matrix v3.1 also records best-effort per-process CPU time, `/proc/self/io`,
cgroup-v2 quota/throttling, and CPU/I/O/memory pressure around each measured
epoch. CPU efficiency participates only when its measurements are complete and
stable. A comparable process-CPU summary is reliable when process counters are
complete, CPU time is positive, CPU-cost CV is at most 15%, and either measured
wall time or accumulated process CPU time is at least 0.5 seconds. The
CPU-time alternative prevents a fast multicore candidate from losing the CPU
objective solely because it completes in under half a wall-clock second.
Missing OS counters do not make an otherwise valid calibration fail.
CPU cost is compared between equivalent medium/large scenarios, not by taking
a variance across inherently different point and bulk requests. Cgroup paths
are resolved through `/proc/self/mountinfo` when available; an ambiguous
namespace-root fallback remains diagnostic and is not used for attribution.
PSI is aggregated for diagnosis but is not treated as ambient contention,
because the measured reads can create their own I/O pressure without an idle
control epoch.
For final selection, every candidate uses the same common subset of reliable
CPU scenarios. A short scenario can therefore be dropped without discarding a
reliable longer scenario, while candidate-specific subsets are never compared.
`selection.cpu_resource_coverage` lists requested, used, and dropped scenarios;
an empty common subset is an explicit performance-only fallback.
Process maximum RSS remains diagnostic because it is a process-lifetime
high-water mark rather than a candidate-local delta. Backing reads are a
separate measurement-validity guardrail: `storage_stability` aggregates
`/proc/self/io read_bytes` for every decision-bearing non-hot scenario in
screening and authoritative verification and includes a fresh backfill
baseline when one is used. Ratios at or below 0.10 are classified as
`page_cache`, ratios at or above 0.50 as `backing_storage`, and the interval
between them as `mixed`. Every individual scenario in every decision-bearing
epoch must be `backing_storage`; a large cold bulk read cannot mask a warm
point, latency, parallel, or fairness workload. Missing/reset counters are
`unknown`. `/proc/self/io` is process-wide, so the full matrix is executed in a
dedicated fresh process. I/O from the parent Jupyter kernel, Airflow task, or
embedding application therefore cannot certify cached ThrowDB reads or bias
which candidates survive screening. Buffered/thread-pool repetitions also
require their `rchar` volume to remain within 0.75-1.25x of the planned ThrowDB
payload, rejecting unexpected reads inside the worker itself. Linux kernels do
not consistently charge `io_uring` payload completions to `rchar`. When a
present counter is no greater than 0.10x of the payload, and only while the
matrix runs in the exec-isolated worker, the guard may use the exact profiled
request bytes because matrix v3 fixes `max_gap_bytes=0`. The parent verifies
the worker PID and protocol before persistence. Missing `rchar`, ambiguous
partial accounting above 0.10x but below the normal range, over-reporting, or
insufficient `/proc/self/io:read_bytes` evidence still fails closed. The report
records the accounting method and raw observed ratio; there is no in-process
fallback. Only an explicitly valid selection, a passed
`backing_storage` report from the isolated worker, and successful
file-scoped cache preparation for every executed stage are eligible for
automatic profile persistence. Memory is enforced as a hard calibration
buffer budget, while
worker/queue/ring counts enter the explicit topology-cost objective instead of
pretending those diagnostics are precise per-candidate costs.
Severe cgroup throttling is a guardrail only when at least 80% of the observed
cgroup CPU time is attributable to the calibration process. Throttling also
participates in the Pareto objective only under that same confident
attribution. When cgroup telemetry is present, low-confidence path attribution,
an explicit non-attributable epoch, or process CPU share below 80% makes the
whole result inconclusive rather than silently teaching a reusable profile
from shared activity. Completely unavailable cgroup telemetry still degrades
to the non-cgroup objectives. Activity from other processes is reported as
ambient load, and affected finalists are listed in
`resource_stability.unattributable_candidates`. Any contaminated or
unattributable finalist makes the result inconclusive, because even a
non-selected measurement can distort the comparison and teach the wrong
reusable profile.

The `balanced-v1` policy builds an explainable Pareto frontier from the
objectives that are reliable for every eligible finalist: performance, CPU
efficiency, tail latency, observed cgroup throttling, and bounded topology cost.
Including topology cost keeps a lower-thread profile on the frontier when it
trades a small amount of speed for fewer resources. The policy chooses within a 5%
performance band, then prefers clean throttling and efficient CPU use, with the
existing 2% performance tie band and lower-resource topology as deterministic
tie-breakers. `selection.eligible_candidate_count` and
`selection.comparable_candidates` identify the guardrail-eligible comparison
set. Each entry in `selection.objective_details` separates
`telemetry_available` from `decision_active`; an objective can be observable
but unable to affect a decision when only one candidate is eligible. The map
also contains `topology_cost`, which is always observable but decision-active
only when the selection is valid and at least two candidates are eligible.
Revision 2 reports set
`selection.selection_schema_revision=2`; `selection.selection_valid` is true
only when the sequential control itself remains eligible.

`candidate_discovery.verified_candidates` lists every candidate with an
authoritative measured verification result, including a successful backfill.
`candidate_discovery.final_comparison_candidates` is the narrower final
guardrail-eligible comparison set and matches
`selection.comparable_candidates`.

The alternatives distinguish raw observations from safe choices.
`selection.alternatives.observed_max_throughput` names the fastest measured
finalist even when it failed a guardrail, while
`selection.alternatives.eligible_max_throughput` names the fastest eligible
finalist. Normal alternative objects set `available=true` and include
`eligible` and `reasons`. If no candidate is eligible,
`eligible_max_throughput` is an explicit unavailable object with
`available=false`, null winner/options/score, and an explanatory reason.
`selection.alternatives.max_throughput` remains the compatibility field: it
aliases the eligible maximum when one exists and retains the baseline
diagnostic when none exists. `selection.alternatives.cpu_efficiency` remains
the reliable CPU alternative. Only the balanced winner is persisted. A candidate
still needs a 5% composite throughput gain to replace the baseline.
Baseline drift above 20% or baseline CV above 10% makes the whole run
inconclusive. Candidate CV above 10% normally makes only that candidate
ineligible; the one explicit borderline-CV recheck described above is the only
exception to immediate candidate rejection. Saving the measured baseline when
no candidate clears the gain threshold is intentional: it records that this
exact or compatible target was calibrated instead of asking to tune it on
every open.

The time limit is cooperative because one already-started filesystem, index,
or payload operation cannot be cancelled safely in the current engine. No new
calibration or publication phase starts after the deadline; registry
publication also takes its advisory lock in fail-fast mode. Candidate count,
sampled slots, query rows, output memory, report size, and planned physical
bytes have hard bounds.
The read-byte limit covers correctness, explicit warm-up, measured embedding
payload reads, one optional full-verification backfill pair (fresh sequential
control plus candidate), and one optional paired stability recheck. Both
adaptive slots are reserved before workload sizing, so
`workload.requested_read_bound_bytes` remains a hard worst-case bound. The
actual `budget.planned_read_bytes` charges only operations that ran and can
therefore be lower when no backfill or recheck was needed. Neither value
includes the collection's one normal index load. The memory limit covers
calibration
samples, the bounded native sampler workspace, permutations, query arrays, and
output arrays. The sampler keeps a single seed-ordered bottom-K across
proportionally bounded WAL-overlay, active-run, and raw base-snapshot scans
instead of copying the complete overlay. `inspected_slots` charges every
ID-bearing run/overlay location and every raw snapshot slot. Run-bitmap metadata
is not an ID slot, but its traversal is separately bounded by the format caps
(at most 4,096 runs times 8,192 bitmap bytes) and reuses a fixed per-run section
buffer. The limit does not include
the collection index, page cache, native thread stacks, or
kernel ring memory. Those resources are instead bounded by the small candidate
matrix and its capped worker/ring counts. The report includes native
`open_stats`, including index file bytes and phase timings, so the excluded
one-time open cost stays visible.
`--dry-run` returns the same report without creating or changing tuning
metadata when the calibration is storage-backed. A page-cache-dominated or
partially cached result instead returns `cache_unrepresentative`; unavailable
backing-read telemetry returns `cache_unverified`. Both retain the complete
selection report but set `selection.persisted=false`. A stable opaque-overlay
target is reusable at the compatible level, so a normal non-dry calibration
may publish its winner only when `storage_stability.persistence_safe=true`.
The report then has
`target.profile_reusable=true`,
`target.profile_reuse_level="compatible"`, and an `identity_warning` explaining
that the hidden backing storage was not identified. Exact targets report
`profile_reuse_level="exact"` without that warning.

A target remains fallback-only when discovery is incomplete, the storage class
is unknown, or another required compatibility input cannot be probed. Dry-run
may still measure it, but the report has `target.profile_reusable=false` and
`target.profile_reuse_level="fallback"`; a non-dry run rejects it before the
expensive benchmark. On correctness failure, interruption, deadline
exhaustion, unstable load, unrepresentative cache state, missing backing-read
evidence, identity change, or registry-generation conflict, no profile is
published. `--output` writes the JSON report atomically; progress goes to
stderr and the final report to stdout.

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
them. `throwdb tune` records its matrix-v3.1 balanced winner with the
compatibility source `calibrated-v3`. The registry schema is unchanged:
existing `calibrated-v2`
profiles remain readable and eligible under the same media/runtime/layout
compatibility rules. Both paths publish through the same crash-safe registry.
Existing open handles keep their current policy until reopened.

`tuning remove` removes only the current exact or compatible target profile.
`tuning reset` removes the entire registry and all profiles; add `--force` to
recover an invalid or corrupt optional registry. Neither command touches
collection data or mutation state.

## Current boundary

Profiles select one backend topology and range/task policy when a collection
handle opens; they do not yet choose a different backend per request. When
that topology enables a parallel thread-pool or io_uring coordinator, the
shared C++ physical reader applies bounded FIFO admission and deficit round
robin across mixed-size concurrent calls. It charges range pressure and
rounded physical bytes under one backend budget, including one-range requests.
Sequential `io_workers=0` reads intentionally bypass the coordinator because
they have no shared parallel backend capacity to schedule.
User-visible priorities, deadlines, and per-request backend selection remain
later policy extensions.
