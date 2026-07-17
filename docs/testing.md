# MVP verification and benchmark plan

The local-storage MVP is considered correct only when the same published
collection can be closed, reopened, queried, mutated, and recovered without
relying on writer-process state. The S3 publication protocol is verified
separately, including an end-to-end version-pinned restore, per-file payload
scrub, native full-index verification, and atomic local publication. Tests use
small dimensions and segment sizes, but exercise the same format, WAL framing,
and address arithmetic as a production `float16[1024]` store.

## io_uring environment probe

The optional Linux probe calls `io_uring_setup` directly through the kernel ABI;
it does not require `liburing`. Run it before building an io_uring reader because
container seccomp policy may reject the syscall even on a supported kernel:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target throwdb_io_uring_probe -j
./build/throwdb_io_uring_probe
```

Exit code 0 means a ring was created successfully. `EPERM` or `EACCES` means the
container policy must be changed before an io_uring backend can be tested.

## Correctness matrix

| Area | Required case | Expected result |
| --- | --- | --- |
| Round trip | Arbitrary `uint64` IDs and bit-exact `float16` vectors | Reopened reader returns identical bits |
| Ordering | Query IDs in a different order, including repeated query IDs | Rows follow query order; repeated IDs repeat rows |
| Sharding | Force several shards with a tiny configured shard size | Reads across shard boundaries are correct |
| Duplicate input | Add the same ID twice, in one or separate batches | Build fails and is not published |
| Missing input | Query at least one ID absent from the index | Entire call fails; MVP does not return a partial result |
| Validation | Wrong ID dtype/rank, vector dtype/rank/dimension, or row count | Call fails before writing the batch |
| Output buffer | Wrong dtype, shape, or non-contiguous output | `get_into` fails before I/O |
| Lifecycle | Explicit `finish` and successful context-manager exit | Store is published and reopenable |
| Lifecycle | Exception inside writer context | No readable store is published |
| CRUD atomicity | Mixed valid/invalid insert or update batch | Complete batch fails with no logical change |
| WAL recovery | Committed PUT/DELETE batches followed by reopen | Exact latest rows and count are restored |
| Torn active WAL | Incomplete final append | Read ignores it; next writer truncates and syncs it |
| Writer lease | Two read-write opens | Exactly one succeeds; readers remain allowed |
| Process boundary | Reuse a builder or handle after `fork()` | Child operation is rejected and cannot release the parent's lease |
| Corruption | Truncated shard, index, or manifest | Open or first affected read fails loudly |
| Format compatibility | New writer emits v1 manifest, index, and every shard at the documented offsets | Existing v1 wire representation stays unchanged |
| Version dispatch | Manifest, segment/shard, index, or WAL advertises an unknown version | Reader rejects it before interpreting version-specific fields |
| Internal I/O | `io_workers` 0, 1, 2, 4, and 8 with a bounded queue | Results remain bit-exact across shards |
| Chunked I/O | `io_ranges_per_task` 1, 4, 8, and 16, including batches smaller than a task | Results remain bit-exact at every partition boundary |
| Shared reader | Simultaneous `get` calls on one `Storage` instance | Calls complete without cross-request corruption |
| Worker error | A queued range hits a truncated shard | Exception reaches the caller; executor remains usable |
| io_uring | Queue depths 1, 2, and 4 with more ranges than SQ entries | Ring wraps and every row remains bit-exact |
| Multi-ring io_uring | One and two workers with total queue depths 2, 3, and 4 | A single request is split across rings without changing bits or row order |
| Multi-ring concurrency | Simultaneous callers share two ring-owner workers | Calls complete without cross-request corruption or deadlock |
| io_uring error | Truncate a shard after open, then issue a healthy request | EOF reaches the caller and the ring remains reusable |
| Mixed-size fairness | A small call arrives behind a multi-unit large call | The small call completes before the large call drains; both make progress |
| One-range fairness | Concurrent one-range calls use an enabled parallel backend | Calls pass through the shared coordinator rather than bypassing its credits |
| Format-7 materialization | Commits and rotation continue while a frozen prefix is built | Final catalog keeps newer WALs and reads remain bit-exact after reopen |
| Selective fold | One partition is active in runs that also cover other partitions | Only that partition is replaced; remaining active sections stay reachable |
| Segment cleaner | Live, stale, deleted, and newer-WAL rows share one source segment | At most one segment is retired; every logical result and old reader view remains correct |
| Format-7 orphan GC | Retired resources, interrupted partial pairs, CURRENT-referenced halves, symlinks, and unknown files coexist | Fresh locked cleanup removes eligible superseded catalogs and unreferenced resources, retries safe partial singletons, and preserves every reachable or unknown entry |
| gRPC transport | Loopback plaintext, non-loopback plaintext, valid mTLS, incomplete mTLS, and malformed endpoints | Only canonical numeric loopback may use plaintext; every non-loopback bind requires a complete mTLS configuration unless the explicit unsafe override is present |
| gRPC admission | Oversized Get/mutation/scan requests, concurrent calls, scans, and mutations | Work is rejected before large decode/native I/O when possible; scan bytes use an independent pool and cannot consume unary Get/CRUD capacity; call, scan, and single-mutation bounds are enforced |
| gRPC admission observability | Admit and reject unary/scan calls, then drain or terminal-fault the server | C++ and Describe snapshots expose zero-leak current gauges, separate monotonic counters, and closed/draining/terminal lifecycle state |
| gRPC stream lifetime | A ScanIds client stops consuming or exceeds the configured total lifetime | The server watchdog cancels the blocked synchronous stream (`CANCELLED` on the raw wire; `DEADLINE_EXCEEDED` remains the client-deadline status), releases its scan lease/bytes, and admits a replacement scan |
| gRPC health closure | Drain or terminal storage fault begins while the server is running | Both Health APIs report `NOT_SERVING`, newly admitted data-plane calls fail with `UNAVAILABLE`, and a terminal fault makes the daemon drain and exit nonzero |
| gRPC shutdown bound | SIGTERM arrives during cooperative RPC/lifecycle work | Graceful drain runs first; the process-level watchdog forces a failed exit at the explicit hard timeout rather than retaining the writer lease indefinitely |
| Mutation safety | Low disk reserve, pending-operation fence, hard WAL/overlay/journal limits, append/fsync ambiguity, checkpoint, and exact operation replay | No new PREPARE is written after a preflight rejection; ambiguous durability poisons the writer; exact existing IDs replay until a no-pending checkpoint atomically publishes a new domain, and stale-domain requests fail before PREPARE |
| Automatic maintenance | Journal/WAL approaches a bound; bounded checkpoint/rotate/materialize/fold/clean/orphan work runs; lifecycle reserve is insufficient | A commit that would cross the materialization bound rotates the prior WAL before PREPARE; one frame that cannot fit is rejected before PREPARE, and crossing the lower polling threshold rotates after commit. Every worker cycle is action-bounded; cleaning rewrites at most one threshold-qualified segment after its minimum interval; insufficient reserve defers payload-copying work without partial publication. A pre-existing oversized WAL still requires a larger explicit budget |
| S3 A/B checkpoint | Fake AWS CLI succeeds, leaves a first-run partial slot, changes equal-size/older-mtime bytes, loses CAS, sees an active remote/local lease, corrupts source state, or starts from a stale/forked copy | Only a native-verified, monotonic source with a SHA-256/VersionId completion may become active; LINEAGE permits safe first-run recovery, remote CAS serializes hosts, and stale rollback is rejected |
| S3 restore | Version-pinned current slot, one corrupt version payload, wrong expected backup ID, or destination race | Restore checks VersionId/ETag before and after transfer, hashes every file, runs full native verification, and publishes only by atomic rename; every failure removes staging |

Floating-point comparisons are deliberately inappropriate here. The storage
engine does not calculate with embeddings: it preserves the original 16-bit
payload. Tests therefore compare `float16` arrays bit for bit.

`tests/test_s3_backup.py` runs the S3 publication protocol without a
network or AWS credentials. Its fake `aws` executable models conditional puts,
ETags, immutable VersionIds, version listing, bulk sync, delete markers, and
commit-then-error outcomes. The suite uses real format-7 collections and covers
full source preflight, checksum inventory, A/B alternation, first-partial
recovery, equal-size/older-mtime changes, local and remote leases, CAS
conflicts, legitimate same-sequence server-journal/orphan changes, stale/fork
rollback, dry-run/age skip, version-pinned restore, checksum corruption
cleanup, native restore verification, and CLI forwarding. A
separate real-S3 drill is still required to validate deployed IAM, multipart
behavior, bucket lifecycle, throughput, and recovery time.

## Large-scale acceptance run

Functional CI should remain small. The NVMe acceptance run is a separate,
explicit job using the production shape and cardinality:

- 500--600 million unique arbitrary `uint64` IDs;
- dimension 1024 and dtype `float16` (2048 useful bytes per row);
- 16 GiB data shards;
- cold and warm random-query phases;
- batch sizes 1, 16, 128, 1K, 10K, and 100K;
- concurrency tested separately after the single-reader baseline is stable.

The read benchmark reports at least median, p95, and p99 call latency,
embeddings/s, useful GiB/s, and query count. A later I/O-instrumented version
must additionally report bytes actually read, read amplification, syscall count,
and queue depth. Random ID generation and output allocation are kept outside
the timed interval.

In `benchmarks/run_load_test.py`, standalone `--cache-mode warm` first replays
the exact deterministic measured query set outside the timed interval. Its
result records `explicit_measurement_prewarm: true`. Mode `both` instead evicts
and measures that stream cold, then immediately measures the same stream warm;
the warm phase needs no additional replay and records the field as false.

For meaningful cold-cache numbers, use an isolated host and an agreed cache
eviction procedure. Running immediately after database construction measures
the page cache rather than NVMe random-read behavior.

The legacy Python benchmark consumes the original `ids.npy` directly:

```bash
python benchmarks/random_read.py \
  --storage /nvme/store \
  --ids /datasets/ids.npy \
  --batch-sizes 1,16,128,1024,10000,100000
```

`throwdb_bench` is the lower-overhead C++ runner built by CMake. It accepts a
headerless, native-endian `uint64` ID population followed by optional batch size,
call count, and seed:

```bash
python -c 'import numpy as np; np.load("ids.npy").tofile("ids.u64")'
./build/throwdb_bench /nvme/store ids.u64 1024 100 20260711
```

Both runners above are legacy sequential `pread` baselines: they open `Storage` with
zero internal I/O workers. They sample only known IDs before timing and reuse an
allocated output buffer. They measure client-visible latency and useful
throughput; neither claims to measure physical bytes read. Use
`run_load_test.py` for executor, chunk-size, and concurrent-reader comparisons;
it auto-detects both fresh native and legacy immutable manifests.

After calibrating and saving a native collection profile, exercise both profile
lookup and the selected backend with the production-shaped reader benchmark:

```bash
python benchmarks/run_load_test.py \
  --store /nvme/store \
  --ids /datasets/ids.npy \
  --source-embeddings /datasets/embeddings.npy \
  --source-kind npy \
  --results benchmark-results/persisted-reader \
  --profile full \
  --cache-mode both \
  --tuning persisted
```

`persisted` is strict: a missing, incompatible, or unavailable saved profile
fails the run rather than benchmarking conservative defaults. The report's
`storage.tuning` object records the actual selected profile and effective reader
options. Use the default `--tuning explicit` for manual `--io-*` sweeps.

Before a large read run, exercise the native mutation path on a rebuildable
collection. Normal completion restores the sampled logical data, although the
test transactions remain in WAL history:

```bash
python benchmarks/run_mutable_test.py \
  --store /nvme/store \
  --ids /datasets/ids.npy \
  --source-embeddings /datasets/embeddings.npy \
  --sample-rows 64 \
  --results benchmark-results/mutable-smoke.json
```

## Single-collection gRPC benchmark

`benchmarks/run_grpc_server_benchmark.py` is the bounded server counterpart to
the embedded reader benchmark. Its default path launches the public Python
control plane on loopback, which selects reader tuning once and `execve()`s the
C++ daemon. Unless `--crud-rows` is nonzero, that daemon is read-only. The run
checks source-backed or embedded bit-exact correctness, measures sequential,
concurrent, and mixed-size Get workloads, streams a bounded physical-order ID
prefix, and replays the identical sequential query stream against an embedded
reader using the server's effective settings.

Report revision 2 adds controlled cache states, three deterministic repetitions
by default, exact-corpus mixed-workload references, and separate client queue,
RPC, and sojourn timings. Quality-oriented defaults use 100 sequential calls,
20 concurrent calls per worker, and 200 small plus 100 large mixed calls in
each repetition. Use strict persisted selection and an explicit paired
cache run for the main local-loopback result:

```bash
STORE=/nvme/store
IDS=/datasets/ids.npy
EMBEDDINGS=/datasets/embeddings.npy
NATIVE="$PWD/build/server-release/throwdbd"

python benchmarks/run_grpc_server_benchmark.py \
  --store "$STORE" \
  --ids "$IDS" \
  --source-embeddings "$EMBEDDINGS" \
  --source-kind npy \
  --results benchmark-results/grpc-local-persisted \
  --native-binary "$NATIVE" \
  --tuning persisted \
  --cache-mode both \
  --repeats 3 \
  --max-seconds 300 \
  --max-transfer-gib 16 \
  --max-client-memory-mib 4096
```

The result directory is created with `exist_ok=False`; use a new directory for
every run. `--tuning persisted` refuses a missing, corrupt, incompatible, or
unavailable saved profile rather than measuring fallback defaults. Omit
`--native-binary` when using an installed server wheel.

### Cache modes and repetitions

`--cache-mode` accepts the following values:

- `auto` is the default. A server launched by this harness resolves to the
  paired `cold,warm` modes when file-scoped cache advice is available. If that
  API is unavailable, the report records the fallback and runs `warm` only. An
  external `--endpoint` always resolves `auto` to `warm`.
- `cold` advises the current collection payload files as unused immediately
  before each timed trial. `both` runs a cold trial followed by a separately
  prepared warm trial for the same query corpus. Explicit `cold` and `both`
  require a server launched by this harness; an external client cannot safely
  control the server's page cache, even when the endpoint is on loopback or a
  local mirror is passed with `--store`.
- `warm` performs an untimed exact replay of the measured query corpus and then
  measures that corpus. Concurrent replay uses the same connected clients,
  worker count, and task topology as the measured wave.
- `ambient` performs neither file advice nor exact-query replay. It describes
  the cache state inherited from surrounding activity and is intended for
  observational deployed-system measurements, not controlled cold/warm
  comparisons.

Cold preparation uses `POSIX_FADV_DONTNEED` only on ThrowDB embedding payload
and WAL files discovered under the selected collection. It neither drops the
machine-wide cache nor modifies the collection. The advice is file-scoped and
advisory, not a guarantee: each trial therefore records available process
`read_bytes` counters and classifies the observation as page-cache,
backing-storage, mixed, or unknown. Unrelated lower-layer or host I/O remains
outside that process-level classification. Aggregate rows retain every
observed classification and expose `cache_verified`,
`cache_verification_status`, and `cold_unverified`; a requested cold run is
verified only when every repetition observes backing-storage reads.
The remote-versus-embedded section records both classifications and leaves its
ratios `null` when a controlled cold/warm pair is not verified on both sides.
Matched mixed-workload slowdowns are likewise `null` unless the mixed,
isolated-small, and isolated-large trials all verify the requested controlled
cache state; within-mixed large/small ratios remain available and are labelled
with the observed state.

The exact warm replay is outside measurement timers and telemetry windows, but
its planned request and response bytes are charged to the run-wide transfer
budget and exposed as `prewarm_accounted_wire_bytes`. Before either cold
eviction or warm replay, concurrent scenarios issue one untimed Get per
connected client so both modes measure an initialized gRPC/Get path. This
traffic is reported as `stabilization_accounted_wire_bytes`. The total budget
includes measured traffic, exact replay, and stabilization, reserved atomically
before the scenario starts. Consequently a paired three-repeat run consumes
materially more than the revision-1 smoke; the recommended full local command
uses 16 GiB of accounted-transfer headroom.

`--repeats N` and its alias `--repetitions N` select the number of deterministic
measurement repetitions; the default is three. Cold and warm trials inside one
pair use the same query digest. Repetitions deliberately use different seeded
corpora, and scenario order rotates deterministically between repetitions to
reduce bias from time-correlated ambient load.

Use this short source-backed smoke after a build. It still exercises the
launcher, Describe/ready agreement, sequential and concurrent Get, mixed-size
traffic, and bounded ScanIds, but omits the embedded throughput comparison:

```bash
python benchmarks/run_grpc_server_benchmark.py \
  --store "$STORE" \
  --ids "$IDS" \
  --source-embeddings "$EMBEDDINGS" \
  --source-kind npy \
  --results benchmark-results/grpc-smoke \
  --native-binary "$NATIVE" \
  --tuning auto \
  --cache-mode warm \
  --repeats 1 \
  --batch-sizes 1,128 \
  --prefix-dimensions full \
  --calls 2 \
  --warmup-calls 1 \
  --concurrency 1,2 \
  --concurrent-batch-size 128 \
  --concurrent-calls-per-worker 1 \
  --mixed-workers 2 \
  --mixed-small-batch 1 \
  --mixed-large-batch 128 \
  --mixed-small-calls 4 \
  --mixed-large-calls 2 \
  --correctness-rows 32 \
  --scan-rows 1024 \
  --scan-batch-rows 256 \
  --no-embedded-baseline \
  --max-seconds 60 \
  --max-transfer-gib 1 \
  --max-client-memory-mib 256
```

Mutation traffic is opt-in and changes durable history. Run it only on a
rebuildable collection with no other writer:

```bash
python benchmarks/run_grpc_server_benchmark.py \
  --store "$STORE" \
  --ids "$IDS" \
  --source-embeddings "$EMBEDDINGS" \
  --source-kind npy \
  --results benchmark-results/grpc-crud-smoke \
  --native-binary "$NATIVE" \
  --tuning persisted \
  --cache-mode warm \
  --repeats 1 \
  --batch-sizes 128 \
  --prefix-dimensions full \
  --calls 2 \
  --warmup-calls 1 \
  --concurrency 1 \
  --concurrent-batch-size 128 \
  --concurrent-calls-per-worker 1 \
  --mixed-workers 2 \
  --mixed-small-calls 2 \
  --mixed-large-calls 2 \
  --correctness-rows 32 \
  --scan-rows 1024 \
  --scan-batch-rows 256 \
  --crud-rows 64 \
  --no-embedded-baseline \
  --max-seconds 120 \
  --max-transfer-gib 1 \
  --max-client-memory-mib 256
```

The cycle reads the selected original rows, writes changed values, deletes and
reinserts them, then upserts the exact originals and verifies them bit for bit.
On failure it attempts an emergency restore. A successful report states
`logical_contents_restored: true`, but the committed mutation WAL and durable
idempotency journal remain changed. This is a logical restoration, not an
on-disk rollback. `--allow-nondurable-crud` is only an escape hatch for a
disposable server that does not advertise durable idempotency.

### Existing endpoint and telemetry scope

Use explicit warm preparation to benchmark a daemon managed outside the
harness:

```bash
python benchmarks/run_grpc_server_benchmark.py \
  --ids "$IDS" \
  --source-embeddings "$EMBEDDINGS" \
  --source-kind npy \
  --results benchmark-results/grpc-existing-server \
  --endpoint "$ENDPOINT" \
  --server-pid "$SERVER_PID" \
  --no-embedded-baseline \
  --cache-mode warm \
  --repeats 3 \
  --max-seconds 300 \
  --max-transfer-gib 16 \
  --max-client-memory-mib 4096
```

In this mode the harness does not start, reconfigure, or stop the daemon; its
effective settings come from `Describe`, not the benchmark's `--tuning`
argument. `--server-pid` is optional and valid only with `--endpoint`. Supply
it only when the server PID is visible in the benchmark process's `/proc`
namespace; it enables per-process CPU, RSS, thread-count, and I/O deltas.
Without it, server process telemetry is explicitly marked unavailable.
`--native-binary` cannot be combined with `--endpoint`. For a genuinely remote
server, `--store` is optional when `--source-embeddings` and
`--no-embedded-baseline` are present; server-storage probes are then marked
unavailable. Supply a local `--store` only when it is an exact mirror and an
embedded comparison against that mirror is meaningful.

Every run writes an atomic, incrementally checkpointed `report.json`. A failed
phase therefore retains completed results, the active budgets, and the error
traceback. Revision 2 records `format=throwdb-grpc-server-benchmark` and
`revision=2`. It includes:

- the arguments, system/cgroup context, server Health and Describe, and for a
  local launch the ready record and exact command;
- the resolved cache methodology, cache preparation and observation, pair IDs,
  and deterministic query SHA-256 digests;
- every raw repetition with per-scenario latency/throughput, client and
  available server process deltas, and shared-cgroup CPU deltas;
- `phases.summary`, where throughput is total work divided by total measured
  trial time, available latency values are medians of the per-trial
  percentiles, and `trial_variation` retains min/median/max/mean and
  coefficient of variation;
- remote-versus-embedded ratios for identical sequential query streams;
- the ScanIds snapshot, bounded row count, physical-order assertion, and
  digest; and
- optional CRUD operation IDs and restoration status.

On Linux, `diskstats.jsonl` samples `/proc/diskstats` once per second and the
final aggregate for active devices is copied into `report.json`. These are
host/block-device counters, not ThrowDB-process attribution; unrelated I/O and
device-mapper layering can affect them. `vmstat.log` is produced only when the
`vmstat` executable exists. Its counters are also system-wide. A locally
launched server additionally writes `server.stdout.log` and
`server.stderr.log`; external mode has no ready/stdout capture.

### Bounds and interpretation

- `--max-seconds` is a run-wide work deadline. New RPC deadlines are clipped
  to the remaining time. Server shutdown and final atomic reporting may add
  cleanup time after that deadline; this is not an operating-system hard kill.
- `--max-transfer-gib` bounds the harness's dominant accounted ID and embedding
  payloads before each phase or concurrent wave, including exact-query warm
  replay and one-Get-per-client stabilization. It does not include protobuf
  tags, gRPC/network framing, allocator overhead, physical read amplification,
  or page-cache traffic, so it must not be reported as exact network or disk
  bytes.
- `--max-client-memory-mib` bounds incremental benchmark working memory using
  planned response buffers plus best-effort client RSS. The embedded index is
  rebased as an engine baseline and the separately running server's index/RSS
  is not charged to this client-memory number.
- `--rpc-timeout` still caps each RPC; server request/message limits can be
  lowered independently with the `--server-max-*` arguments.
- `--scan-rows` is an explicit maximum, and `--scan-batch-rows` bounds each
  retained streaming batch. The harness verifies contiguous offsets and
  strictly increasing `(segment_id, row_index)` locations without materializing
  the complete ID set. Set `--scan-rows 0` to skip this phase.

## Single-collection gRPC regression/soak

`benchmarks/run_grpc_server_soak.py` is the production-oriented stability gate.
It complements, rather than replaces, the controlled performance benchmark.
The harness streams a deterministic closed-loop mix of different Get batch and
prefix sizes while running canary `insert -> update -> upsert -> delete`
cycles. Every mutation has a deterministic UUID and is deliberately replayed
with the same bytes. UUIDs are derived inside a fresh per-invocation `run_id`
namespace, preventing a later soak from accidentally reusing an old journal
entry; the chosen `run_id` is stored in `report.json` and can be supplied
explicitly with `--run-id` when exact orchestration reproduction is required.
The canaries are proven absent before use and verified
absent after every cycle, so logical contents are restored; WAL and tombstone
history still changes. Run it only against a disposable/rebuildable collection
clone, never a live collection shared with another writer.

After an ambiguous primary mutation result, the harness reuses the same UUID
and exact payload only if `Describe` still reports the original idempotency
domain. It fails closed instead of sending an old operation UUID into a new
domain. The deliberate post-success replay uses the same guard.

By default a local run performs two restart cycles and observes three process
generations: the initial process, one after a graceful restart, and one after a
SIGKILL restart. Before each restart it acknowledges a canary insert;
after Health returns `SERVING`, it first verifies that the idempotency replay
domain did not change, then repeats the same insert UUID. This would fail with
`ALREADY_EXISTS` if the result were not durably replayed. A final replayed
delete restores the absent canary state. Each generation has separate stdout
and stderr files under `server-generations/gNNNN/`.

A small collection is sufficient for CI or installation regression testing.
`--ids` is optional: without it, the harness discovers a bounded physical-order
ID prefix through `ScanIds` and retains at most `--population-rows` IDs.
When row-aligned source files are available, pass both `--ids` and
`--source-embeddings`. The embeddings file is opened as a NumPy memmap and only
the sampled query rows/prefixes are copied; every mixed Get is then checked
bit-for-bit. Without it, the report explicitly uses
`transport_stability_only`, sets `base_read_correctness_verified: false`, and
does not claim correctness for every pre-existing base row. It still checks a
stable captured-row shadow on every Get and validates all known mutation
payloads bit-for-bit.
Source-memmap reads are client-side oracle I/O and are not included in the
gRPC transfer-byte ceiling; only sampled rows are touched.

```bash
STORE=/tmp/throwdb-soak-store
NATIVE="$PWD/build-grpc-dev/throwdbd"

python benchmarks/run_grpc_server_soak.py \
  --store "$STORE" \
  --results benchmark-results/grpc-soak-smoke \
  --native-binary "$NATIVE" \
  --tuning conservative \
  --max-seconds 120 \
  --max-transfer-gib 2 \
  --max-client-memory-mib 1024 \
  --max-get-calls 1000 \
  --max-mutations 40 \
  --restart-cycles 2 \
  --restart-every-gets 250 \
  --scan-rows 10000
```

The three independent run-wide ceilings are `--max-seconds`,
`--max-transfer-gib`, and `--max-mutations`; `--max-get-calls` adds a
deterministic smoke-test endpoint and may be `0` for a duration/transfer-driven
soak. Transfer accounting includes dominant ID, embedding, delete-mask, and
scan-location payloads plus conservatively reserved mutation-cleanup headroom.
It excludes protobuf tags and HTTP/2 framing. New RPC deadlines are clipped to
the remaining wall-clock budget. Best-effort canary cleanup and the final
atomic report may extend slightly past that work deadline.
`--max-errors` can tolerate only explicitly transient Get statuses
(`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED`, or `CANCELLED`);
wrong bits/shapes, missing sampled IDs, protocol failures, and `DATA_LOSS`
always fail immediately and are never converted into an availability pass.

`--max-mutations` counts unique operation UUIDs, not retry attempts. The
harness always preserves one emergency-Delete slot and all not-yet-run restart
probes before admitting another periodic four-operation CRUD cycle. This keeps
the documented restart schedule reachable even when periodic mutations would
otherwise consume the ceiling. The final probe is read-only and reports every
canary as `absent`, `present`, or `unknown` without spending another UUID.
Read workers use separate clients, so per-RPC deadline updates cannot race
between concurrent Gets.

`report.json` is checkpointed atomically. `metrics.jsonl` is append-only and
records server generation, client/server `/proc` RSS, CPU, faults, thread/FD
counts, `/proc/PID/io`, and cgroup CPU counters; safe workload milestones also
include Health, admission, engine/WAL/overlay, journal, free-space, and
maintenance diagnostics. The final report contains bounded latency samples,
all-attempt counters, errors by operation/status, per-generation recovery time,
RSS/FD growth, cgroup throttling, and hard acceptance gates. The command exits
nonzero for a wrong shadow read, lost/rejected replay, changed replay domain,
unexpected RPC error, incomplete restart, excessive recovery time, unexpected
server exit, failed scan ordering, or canary restoration failure.
Latency percentiles use a deterministic hash-priority reservoir rather than an
early-run prefix, while mean and maximum cover every attempt. Resource growth
is observational by default. Set `--max-rss-growth-mib`, `--max-fd-growth`,
and/or `--max-cgroup-throttled-seconds` to turn an available `/proc` or cgroup
measurement into a hard gate; requesting a gate when its metric is unavailable
fails closed.

An already-running endpoint can be observed, but process restart ownership is
unavailable and mutations require an explicit safety acknowledgement:

```bash
python benchmarks/run_grpc_server_soak.py \
  --endpoint 127.0.0.1:50051 \
  --server-pid "$THROWDBD_PID" \
  --allow-external-mutations \
  --restart-cycles 0 \
  --ids /datasets/ids.npy \
  --results benchmark-results/grpc-external-soak \
  --max-seconds 600 \
  --max-transfer-gib 8 \
  --max-mutations 100
```

`--server-pid` is optional and does not grant process-control rights; when the
PID is visible in the current PID namespace it only enables the same `/proc`
RSS, FD, thread, CPU, fault, and I/O samples collected for local launches.

Functional CI cannot establish the 500M-row/Linux/NVMe acceptance result. Run
the same local-launch command on the target host with the real collection,
strict `--tuning persisted`, `/proc` and cgroup visibility, hours rather than
minutes, and a fresh output directory, for example:

```bash
python benchmarks/run_grpc_server_soak.py \
  --store /nvme/throwdb-500m-acceptance-clone \
  --ids /datasets/ids.npy \
  --source-embeddings /datasets/embeddings.npy \
  --results benchmark-results/grpc-500m-soak \
  --native-binary "$PWD/build/server-release/throwdbd" \
  --tuning persisted \
  --max-seconds 21600 \
  --max-transfer-gib 512 \
  --max-client-memory-mib 32768 \
  --max-get-calls 0 \
  --max-mutations 2000 \
  --get-batch-sizes 1,128,1024,10000 \
  --restart-cycles 2 \
  --restart-every-gets 50000 \
  --max-recovery-seconds 60 \
  --max-rss-growth-mib 1024 \
  --max-fd-growth 32 \
  --scan-rows 1000000
```

The large-host run is the authoritative acceptance artifact. A passing small
run proves bounded orchestration, CRUD/replay semantics, restart recovery, and
reporting, but cannot predict 1.5-TiB page-cache behavior, NVMe latency tails,
cgroup throttling, free-space pressure, or long-term RSS/FD slopes.

### Concurrent timing and matched mixed-workload references

The concurrent harness materializes query IDs before the measurement window,
places all calls in a shared client FIFO, and allows at most `workers`
synchronous RPCs in flight. Its timing fields have distinct meanings:

- `client_queue_wait` is the interval from release of the common wave barrier
  until a client worker dequeues the call;
- `client_prepare` is the client-side interval between dequeue and starting the
  synchronous RPC;
- `rpc_round_trip` is the client-observed RPC duration. It includes request
  serialization, transport, server admission/queueing/service, and response
  copies, so it must not be described as server service time;
- `client_sojourn` is the complete interval from wave release until the RPC
  response is available: queue wait, client preparation, and RPC round trip.

The legacy class `latency_*` values are RPC-round-trip percentiles. Per-class
throughput is that class's work divided by the complete scenario wall time.
Per-class summaries also flatten every queue, prepare, RPC, and sojourn
percentile across repetitions and retain their sample distributions and
coefficients of variation. The protocol does not yet expose server-internal
queue and service timestamps.

For every mixed small/large repetition, revision 2 also runs isolated-small and
isolated-large waves with the same worker count, cache mode, and exact class ID
corpus. The report verifies class query digests before calculating both RPC and
client-sojourn P95 slowdown ratios. These matched references remove the earlier
query-set, cache-state, and one-client-baseline confounders. They still do not
constitute a server fairness SLA: client FIFO effects, transport, and all
server work remain combined in the observed durations.

## Internal I/O workers versus reader concurrency

These are independent axes. `--io-workers` controls the C++ executor owned by a
single `Storage`; the concurrent-reader scenarios control how many Python client
threads call `get_into` on that instance. Multi-range calls submit work to the
shared executor. Reports record both axes as `io_workers` and
`external_reader_workers` (the legacy `workers` field is the external count).
`io_ranges_per_task` records the maximum number of physical ranges a worker
processes after one queue dispatch.

With workers enabled, every request becomes a scheduler job, including a plan
containing exactly one physical range. Bounded FIFO admission prevents new
callers from repeatedly bypassing waiters, and deficit round robin interleaves
the task units of mixed-size jobs. Only the zero-worker sequential baseline
uses synchronous direct reads without the coordinator.

Zero internal workers is the synchronous `pread` baseline. With workers enabled,
`--io-queue-capacity 0` selects an automatic bound of twice the worker count; an
explicit positive value bounds waiting tasks and applies producer backpressure.
A queue capacity is invalid when the worker count is zero. The ranges-per-task
value must be in `1..1048576`; it defaults to 8 and is validated even for the
sequential baseline.

Run an internal-worker sweep as separate invocations so each configuration has a
fresh executor and can use the same cold-cache procedure:

```bash
for workers in 0 1 2 4 8; do
  python benchmarks/run_load_test.py \
    --store /nvme/store \
    --ids /datasets/ids.npy \
    --io-workers "$workers" \
    --io-queue-capacity 0 \
    --results "benchmark-results/io-workers-$workers"
done
```

Compare sequential scenarios first to isolate internal parallelism. Then compare
the concurrent section to measure interactions between the shared executor and
multiple client calls; do not describe the two worker counts as one combined
concurrency number.

## Remote chunk-size sweep

After selecting the device-saturating worker count, compare task granularity on
the same immutable store. Keep the query seed, worker count, queue capacity, and
cache procedure unchanged. On the current two-device production test host, the
starting point is 16 I/O workers with a queue capacity of 32:

```bash
cd ~/ThrowDB
source .venv/bin/activate

STORE=/tmp/throwdb-prod-test-store-v2
IDS=/mnt/similar-goods/prod/vectors/e5_vectors/e5_embs/test/ids.npy
EMBEDDINGS=/mnt/similar-goods/prod/vectors/e5_vectors/e5_embs/test/embeddings.npy

for ranges in 1 4 8 16; do
  python benchmarks/run_load_test.py \
    --store "$STORE" \
    --ids "$IDS" \
    --source-embeddings "$EMBEDDINGS" \
    --source-kind npy \
    --profile full \
    --cache-mode both \
    --io-workers 16 \
    --io-queue-capacity 32 \
    --io-ranges-per-task "$ranges" \
    --results "benchmark-results-v6-ranges-$ranges"
done
```

Compare cold single-client batch latency first: chunking must retain roughly the
same device-saturating throughput as one-range tasks. Then compare warm external
reader counts, especially 8 and 16 clients; the purpose of chunking is to reduce
queue lock traffic, notifications, and context switches when reads hit the page
cache. Keep the `ranges=1` result as the v5-compatible control. Every result row
and the top-level configuration record `io_ranges_per_task`, so reports can be
combined without relying on directory names.

On Linux with cgroup v2, `system.cgroup_cpu` records `cpu.max`, the effective CPU
limit, and the initial `cpu.stat` counters. Every timed result additionally has
`cgroup_cpu_delta` with `nr_periods`, `nr_throttled`, `throttled_usec`, and
derived throttled-period/time ratios. These are per-scenario deltas rather than
container-lifetime totals, so CPU-quota stalls can be separated from ThrowDB
queue latency.

## Multi-ring io_uring qualification

The raw-ABI backend uses persistent ring lanes. One worker is caller-driven;
two or more workers have one dedicated owner thread per lane. Queue depth is an
aggregate: two workers with depth 32 create two depth-16 rings. The shared
coordinator converts every call into bounded range/byte work units, applies DRR
between concurrent jobs, and never exceeds aggregate dispatch, range, and byte
credits. Admission waiters enter in ticket order. The caller does not return
until every already-issued lane operation has drained, so temporary plan/output
buffers remain valid. The storage section records `/proc/self/task` counts
before and after open to isolate user-space ring workers.

Compare the known single-ring QD32 baseline with two rings at the same aggregate
depth:

```bash
for workers in 1 2; do
  python benchmarks/run_load_test.py \
    --store "$STORE" \
    --ids "$IDS" \
    --source-embeddings "$EMBEDDINGS" \
    --source-kind npy \
    --profile full \
    --cache-mode both \
    --io-backend io_uring \
    --io-workers 0 \
    --io-queue-capacity 0 \
    --io-uring-workers "$workers" \
    --io-uring-queue-depth 32 \
    --results "benchmark-results-io-uring-workers-$workers-qd-32"
done
```

The two-ring success gate is bit-exact correctness, cold single-client
throughput at least 95% of the single-ring QD32 result, higher warm concurrent
throughput, and no return of cgroup throttling. Compare p50/p99, CPU time,
context switches, diskstats, and `/proc/self/task` count. Add an intentionally
mixed small/large-call scenario: the small call must complete before the large
job drains while the large job continues to make bounded progress.

## Fixed v4 reader A/B profile

Use `run_v4_reader_ab.py` as the regression gate before and after changing the
mutable read path or automatic tuning. It runs one report containing the
following fixed comparison:

- synchronous `thread_pool` with zero internal workers;
- `thread_pool` with 16 workers, queue capacity 32, and 8 ranges per task;
- `io_uring` with 2 ring workers and aggregate QD32;
- batches 1,024 and 10,000 with 1, 8, and 16 external readers;
- three repetitions, each paired cold then warm.

```bash
python benchmarks/run_v4_reader_ab.py \
  --store /tmp/throwdb-prod-test-store \
  --ids /mnt/dataset/ids.npy \
  --source-embeddings /mnt/dataset/embeddings.npy \
  --source-kind npy \
  --results benchmark-results/v4-reader-ab
```

The defaults perform 100 calls per reader for batch 1,024 and 10 calls per
reader for batch 10,000, keeping measured embeddings per reader close to
100,000 in both cases. Override `--calls-1024` and `--calls-10000` only when a
shorter smoke run or a longer device qualification is intentional.

The matrix is scenario-blocked and paired. Inside each repetition it selects
one `(batch_size, external_readers)` scenario, then runs all three policies on
the same deterministic query stream before moving to the next scenario. The
policy order uses a three-way Latin rotation across repetitions. This keeps a
candidate close to its synchronous control while balancing first/last policy
position. The generated `summary` reports per-policy distributions and
repetition-paired throughput speedups. With the default `both` cache mode there
are 108 timed results. Standalone `--cache-mode warm` retains the explicit
measured-query replay guarantee used by `run_load_test.py`.

To keep memory bounded, each policy-scenario block opens one collection handle,
runs its adjacent cold/warm pair, and closes it before the next policy. The
default matrix therefore performs 54 measured-policy opens in addition to the
three preflight opens. Open time is recorded in `storage_opens` but is outside
read throughput and latency. This intentionally trades extra index loads for
temporal pairing without retaining three multi-gigabyte indexes at once.

Before monitors or timed work start, all three policies are opened and checked.
Candidates must return a byte-for-byte match to the synchronous baseline even
when `--source-embeddings` is omitted; the optional source additionally checks
that baseline against the original NumPy data. Per-policy open metadata,
correctness status, error details, and one profiled-read telemetry sample are
persisted under `policy_preflight`. An unavailable `io_uring` environment or a
backend mismatch therefore fails before the expensive matrix begins.

By default the preflight also requires
`telemetry.zero_delta_fast_path: true`. A mutation smoke leaves entries in the
active WAL even after restoring their logical values, so use a freshly built
collection with no committed WAL mutations for the empty mutable-read
regression gate. On format 7, materializing every WAL and folding all active run
sections into partition snapshots can restore an authoritative snapshot-only
state; retired run files are reclaimed separately. To intentionally measure a
collection with mutation overlay state, opt
out explicitly:

```bash
python benchmarks/run_v4_reader_ab.py \
  --store /tmp/throwdb-prod-test-store-with-mutations \
  --ids /mnt/dataset/ids.npy \
  --results benchmark-results/v4-reader-ab-mutations \
  --no-require-zero-delta
```

Do not mutate or replace the collection during a run. This profile requires
native format v4 or newer and refuses to overwrite an existing `report.json`.
The report is updated atomically after every preflight and measured result, so
a stopped run leaves its completed raw measurements available for diagnosis;
start a replacement run in a new results directory.

For a short installation and correctness smoke test of the same matrix, reduce
the work without changing its policies or scenarios:

```bash
python benchmarks/run_v4_reader_ab.py \
  --store /tmp/throwdb-prod-test-store \
  --ids /mnt/dataset/ids.npy \
  --source-embeddings /mnt/dataset/embeddings.npy \
  --results benchmark-results/v4-reader-ab-smoke \
  --repeats 1 \
  --calls-1024 2 \
  --calls-10000 1
```

The timed A/B path deliberately uses ordinary uninstrumented `get_into()`.
Preflight uses `Collection.get_into_profiled()` once per policy outside the
acceptance measurement and stores the result in the report. Its
`physical_backend_seconds` combines admission/queue wait, I/O, and scatter;
`planned_ranges`, `planned_read_bytes`, and `backend_tasks` describe the plan.
