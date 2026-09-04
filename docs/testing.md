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
| Segmentation | Force several segments with a tiny configured `shard_size_bytes` | Reads across segment boundaries are correct |
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
| Corruption | Truncated segment, index, or manifest | Open or first affected read fails loudly |
| Storage version | Builder emits a format-7 collection with the documented manifest, catalog, index, and segment structures | Format-7 data remains reopenable; formats 1–6 and unknown versions are rejected |
| Version dispatch | Manifest, segment, index, or WAL advertises an unknown version | Reader rejects it before interpreting version-specific fields |
| Internal I/O | `io_workers` 0, 1, 2, 4, and 8 with a bounded queue | Results remain bit-exact across segments |
| Chunked I/O | `io_ranges_per_task` 1, 4, 8, and 16, including batches smaller than a task | Results remain bit-exact at every partition boundary |
| Shared reader | Simultaneous `get` calls on one `Collection` instance | Calls complete without cross-request corruption |
| Worker error | A queued range hits a truncated segment | Exception reaches the caller; executor remains usable |
| io_uring | Queue depths 1, 2, and 4 with more ranges than SQ entries | Ring wraps and every row remains bit-exact |
| Multi-ring io_uring | One and two workers with total queue depths 2, 3, and 4 | A single request is split across rings without changing bits or row order |
| Multi-ring concurrency | Simultaneous callers share two ring-owner workers | Calls complete without cross-request corruption or deadlock |
| io_uring error | Truncate a segment after open, then issue a healthy request | EOF reaches the caller and the ring remains reusable |
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
- 16 GiB data segments;
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

Use `benchmarks/run_load_test.py` for sequential, executor, chunk-size, and
concurrent-reader comparisons against a format-7 collection. It samples known
IDs before timing and reuses an output buffer; useful throughput does not claim
to measure physical bytes read. The old immutable-only and format-4 A/B runners
have been removed.

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

Report revision 3 retains revision 2's controlled cache states, deterministic
repetitions, exact-corpus mixed-workload references, and separate client queue,
RPC, and sojourn timings. It additionally records optional read-logger
configuration/counters and performs a resource-budgeted post-shutdown quality
inspection. Quality-oriented defaults use 100 sequential calls, 20 concurrent
calls per worker, and 200 small plus 100 large mixed calls in each repetition.
Use strict persisted selection and an explicit paired cache run for the main
local-loopback result:

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

When either gRPC harness owns a local daemon, normal shutdown gives the daemon
time to execute its own drain/watchdog policy. The parent-side timeout is the
larger of the effective grace and hard-shutdown limits, plus five seconds,
bounded to the range 10–900 seconds. With the native defaults of 30 seconds
grace and 60 seconds hard timeout, the harness therefore waits 65 seconds
before forcing a kill. The legacy 10-second path is only a floor and an
early-launch-failure cleanup bound; it is not the normal default shutdown
timeout.

### Embedded and gRPC embedding-stream benchmark

`benchmarks/run_streaming_read_benchmark.py` measures the two format-7
embedding streams without concatenating batches. Exactly one target is
required: `--store` opens an embedded collection, while `--endpoint` connects
to an already-running `throwdbd`. The harness never starts, restarts, or stops
the remote server. It records time to first batch, rows and logical GiB per
second, a content digest, snapshot/terminal state, negotiated batch limits,
and best-effort client-process RSS.

Embedded examples:

```bash
python benchmarks/run_streaming_read_benchmark.py \
  --store "$STORE" \
  --mode scan-records \
  --batch-rows 65536 \
  --max-batch-mib 64 \
  --max-snapshot-mib 256 \
  --prefix-dimensions 128 \
  --with-locations \
  --output benchmark-results/scan-records.json

python benchmarks/run_streaming_read_benchmark.py \
  --store "$STORE" \
  --mode stream-get \
  --ids "$IDS" \
  --batch-rows 65536 \
  --max-batch-mib 64 \
  --max-snapshot-mib 256 \
  --output benchmark-results/stream-get.json
```

The equivalent remote runs use the same modes and bounded batch loop:

```bash
python benchmarks/run_streaming_read_benchmark.py \
  --endpoint 127.0.0.1:50051 \
  --mode scan-records \
  --batch-rows 65536 \
  --max-batch-mib 64 \
  --prefix-dimensions 128 \
  --with-locations \
  --output benchmark-results/grpc-scan-records.json

python benchmarks/run_streaming_read_benchmark.py \
  --endpoint 127.0.0.1:50051 \
  --mode stream-get \
  --ids "$IDS" \
  --batch-rows 65536 \
  --max-batch-mib 64 \
  --output benchmark-results/grpc-stream-get.json
```

`--max-snapshot-mib` and `--tuning` are embedded-only; remote snapshot and
reader budgets belong to the already-running server. Remote client bounds can
be tightened with `--max-frame-mib`, `--rpc-timeout-seconds`, and
`--scan-timeout-seconds`. A full run must observe logical EOF and the exact
expected row count. With `--max-rows`, stopping is allowed only after a
completed bounded batch reaches the requested truncation point. Failed or
unclosed streams always fail the benchmark.

Human-readable progress is emitted to stderr after the first completed batch,
at most every five seconds, and once after the validated report has been
written. It includes completed/target rows, batches, elapsed time, average
rows/s, logical MiB/s, ETA, and sampled peak RSS. Stdout remains the final
machine-readable result JSON and the report file contains no progress text.
Change the cadence with `--progress-seconds SECONDS` or use `--no-progress`.
The rate is end-to-end logical payload throughput, including lookup, I/O,
copies, and hashing; it is not physical device bandwidth.

For a short validation rather than a full collection pass:

```bash
python benchmarks/run_streaming_read_benchmark.py \
  --store "$STORE" \
  --mode stream-get \
  --ids "$IDS" \
  --batch-rows 65536 \
  --max-rows 1048576 \
  --output benchmark-results/stream-get-1m.json
```

Record fields are hashed incrementally with separate ID, embedding, and
location hashers before producing the combined digest. Consequently different
embedded/server batch boundaries do not change the digest for identical
logical records, and no full-result concatenation is required.

The logical-byte rate is not device bandwidth: range coalescing, filesystem
blocks, page cache, and prefix reads can change physical traffic. Retained
inode bytes are likewise not represented by RSS. In gRPC mode the RSS fields
describe only the benchmark client, not `throwdbd`.

## Single-collection gRPC regression/soak

`benchmarks/run_grpc_server_soak.py` is the production-oriented stability gate.
It complements, rather than replaces, the controlled performance benchmark.
Report revision 2 runs three kinds of work concurrently:

- closed-loop unary `Get` calls over the configured batch and prefix matrix;
- dedicated long-lived workers that repeatedly open bounded, snapshot-pinned
  `StreamGet` and `ScanRecords` attempts;
- canary `insert -> update -> upsert -> delete` cycles while a locally launched
  server's automatic maintenance remains enabled unless explicitly disabled.

Every mutation has a deterministic UUID and is deliberately replayed with the
same bytes. UUIDs are derived inside a fresh per-invocation `run_id` namespace,
preventing a later soak from accidentally reusing an old journal entry; the
chosen `run_id` is stored in `report.json` and can be supplied explicitly with
`--run-id` when exact orchestration reproduction is required. The canaries are
proven absent before use and verified absent after every cycle, so logical
contents are restored; WAL and tombstone history still changes. Run it only
against a disposable/rebuildable collection clone, never a live collection
shared with another writer.

After an ambiguous primary mutation result, the harness reuses the same UUID
and exact payload only if `Describe` still reports the original idempotency
domain. It fails closed instead of sending an old operation UUID into a new
domain. A typed `ABORTED` is handled separately: the server guarantees that
the request was rejected before durable PREPARE, so the harness refreshes
`Describe`, applies bounded backoff, and retries the identical request with the
same UUID even if maintenance rotated the replay domain. Semantic rejections
and arbitrary local/protocol failures are never retried. The deliberate
post-success replay uses the current guarded domain.

### Streaming workload and defaults

The default configuration uses one `StreamGet` worker and one `ScanRecords`
worker, exactly filling the server's default two-slot scan-admission pool.
Each worker owns a separate `RemoteCollection`; its scan deadline and mutable
client state therefore cannot race with the unary control client or another
worker. Before mixed traffic starts, all configured workers must have opened
and each individual worker must return its first batch. Opening is accepted
only after the monotonic `Describe.admission_stats.admitted_scans` counter has
increased by at least the number of newly opened workers. The instantaneous
`active_scans` gauge is recorded for observation, but is not the open gate:
a short finite attempt can complete and reopen while `Describe` is sampled.
Zero `active_scans` and zero estimated scan bytes remain hard requirements at
explicit drain boundaries.

The worker itself is long-lived, but one RPC attempt is finite. Completed or
time-rotated attempts are reopened on a new snapshot until the generation is
stopped. This continuously exercises snapshot creation and release without
turning one cursor into an unbounded lifetime. It is not a resumable export:
closing a partial attempt and opening another starts a new snapshot from the
beginning of that attempt's logical input.

The revision-2 defaults are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--stream-get-workers` | 1 | Dedicated `StreamGet` clients; zero disables this mode |
| `--scan-records-workers` | 1 | Dedicated `ScanRecords` clients; zero disables this mode |
| `--stream-get-rows` | 131072 | Logical ID rows in one finite point-stream attempt |
| `--scan-records-rows` | 131072 | Minimum physical rows consumed by one record-scan attempt |
| `--stream-batch-rows` | 16384 | Requested row ceiling for both streaming modes |
| `--stream-max-batch-mib` | 32 | Requested dense-payload ceiling for one batch |
| `--stream-prefix-dimensions` | `full` | One prefix shared by both streaming modes |
| `--stream-attempt-seconds` | 300 | Rotate a still-partial attempt below client/server lifetime bounds |
| `--stream-progress-timeout-seconds` | 60 | Open, first-progress, and maximum inter-batch-stall bound |
| `--stream-stop-timeout-seconds` | 15 | Bound for cancelling and joining one generation's workers |
| `--stream-locations` | enabled | Request and validate physical locations from `ScanRecords` |
| `--stream-oracle-rows` | 256 | Bounded unique-ID corpus captured through unary `Get` and repeated by `StreamGet` |
| `--stream-source-oracle` | disabled | Independently verify the unary-captured corpus against the row-aligned source memmap |

Before long streams start, the harness reads at most `--stream-oracle-rows`
unique IDs through bounded unary `Get` calls and stores their exact embedding
bits. `StreamGet` then receives a lazy repetition of this bounded corpus, so
the complete logical ID input is not allocated. Every returned row is compared
bit-for-bit with the captured unary oracle; every response is also checked for
contiguous offsets, input ordering, dtype, and shape, and the first row of each
attempt is checked against the stable shadow value. A completed attempt
contributes a BLAKE2b digest, and repeated completed attempts over the stable
base-ID corpus must have the same digest.
`ScanRecords` checks offsets, dtype, shape, and, by default, strictly increasing
physical locations. Its bounded target may be exceeded by the final complete
transport batch; a partial batch is never invented merely to hit the target.

When `--stream-source-oracle` is enabled, the bounded unary capture is first
compared bit-for-bit with the matching rows in `--source-embeddings`. The long
stream still compares against the in-memory capture, so source-memmap I/O occurs
only during setup rather than on every streaming batch. Without the flag the
stream remains fully checked against unary results, but that oracle is not
independently grounded in the source file. Neither check is a special mutation
snapshot canary; the test does not claim to prove visibility of a row changed
after that particular stream opened. In the report,
`stream_get_oracle.kind` distinguishes plain unary capture from a
source-verified capture; the historical `source_verified_rows` streaming
counter counts rows checked against the effective captured oracle.

The harness checks the effective `Describe` bounds before starting: streaming
workers may not exceed `max_concurrent_scans`, and
`streaming workers + Get workers + 1` may not exceed
`max_concurrent_calls`. The extra slot is reserved for the control/mutation
RPC used while reads are active. The attempt duration must also remain below
the server's scan-duration watchdog. Local-only pass-through options such as
`--server-max-concurrent-calls`, `--server-max-concurrent-scans`,
`--server-scan-max-duration-ms`, `--server-scan-idle-timeout-ms`, and the two
shutdown bounds make an explicitly overridden server configuration
reproducible in the report.

### Short-Get fairness gate

Before opening the long streams, the harness runs an isolated concurrent
window for the smallest configured Get batch and prefix. It uses 32 calls by
default (`--fairness-baseline-get-calls`) with the configured Get worker and
in-flight limits. After every stream worker has produced a first batch, it
runs a second concurrent window with the same client topology and the exact
same ordered query corpus; matching SHA-256 query digests are a hard
invariant. This second window isolates short-Get responsiveness under active
streaming load. Periodic mutation scheduling begins only after the window, so
the ratio is not presented as a simultaneous mutation-contention result.

The gate is evaluable only when the mixed class has at least
`--min-mixed-small-get-successes` successful latency samples, 20 by default.
The default hard bound is:

```text
mixed smallest-Get P99 / isolated smallest-Get P99 <= 5.0
```

Change it with `--max-small-get-p99-ratio`; a production run can select a
stricter value after establishing a stable host baseline. Insufficient samples
fail closed, as does any error in the isolated baseline. Both P99 values use
successful calls only; failed-call latencies cannot improve or worsen the
ratio, while the ordinary error budget remains a separate acceptance gate.
This deliberately generous default is a critical starvation guard, not a
performance SLA: cache state is not controlled and RPC latency still combines
client, transport, server admission, queueing, I/O, and response copies. Use
`run_grpc_server_benchmark.py` for controlled cache modes and throughput
comparisons.

Each enabled stream mode must also complete at least one attempt and
`--min-stream-batches-per-mode` batches. Every configured worker—not merely
each mode in aggregate—must produce a batch in every process generation, every
worker must join, the largest observed inter-batch gap must remain within the
progress timeout, and no unexpected failure may be reported. The optional
`--min-stream-rows-per-second` adds a hardware-specific absolute progress
floor; its default of zero checks liveness without pretending to define a
portable throughput SLA.

### Correctness population

A small collection is sufficient for CI or installation regression testing.
`--ids` is optional: without it, the harness discovers a bounded physical-order
ID prefix through `ScanIds` and retains at most `--population-rows` IDs.
When row-aligned source files are available, pass both `--ids` and
`--source-embeddings`. The embeddings file is opened as a NumPy memmap and only
the sampled query rows/prefixes are copied; every unary Get is then checked
bit-for-bit. Without it, the report explicitly uses
`transport_stability_only`, sets `base_read_correctness_verified: false`, and
does not claim correctness for every pre-existing base row. It still checks a
stable captured-row shadow on every Get, every `StreamGet` row against its
bounded unary capture, and all known mutation payloads bit-for-bit. This
proves transport stability relative to the capture, not that arbitrary base
rows match an external source. Source-memmap reads are client-side oracle I/O
and are not included in the gRPC transfer-byte ceiling.

### Restarts and admission lifecycle

By default a local run performs two restart cycles and observes three process
generations: the initial process, one after a graceful restart, and one after a
SIGKILL restart. Unary futures are drained and their clients are closed before
either restart, making the stream behavior at the boundary explicit:

- Before a graceful restart, the harness closes every stream, joins its worker,
  and polls `Describe` until `active_scans == 0` and
  `scan_estimated_bytes == 0`. The server is terminated only after this clean
  application-level drain.
- Before SIGKILL, the harness proves that at least one stream is still active,
  marks transport cancellation as expected, and kills the process.
  `CANCELLED`, `DEADLINE_EXCEEDED`, or `UNAVAILABLE` is accepted only for this
  marked interruption. The replacement generation must initially expose no
  inherited scan lease.

After either restart, new dedicated clients open new streams and must make
progress again. Completed batches from the old generation remain valid
observations, but the harness never reuses an opaque token, transparently
resumes a cursor, or combines two generations into one snapshot result.

Before each restart it also acknowledges a canary insert. After Health returns
`SERVING`, it verifies that the idempotency replay domain did not change and
repeats the same insert UUID. This would fail with `ALREADY_EXISTS` if the
result were not durably replayed. A final replayed delete restores the absent
canary state. Each generation has separate stdout and stderr files under
`server-generations/gNNNN/`.

Admission snapshots are recorded when streams open, before graceful restart,
after SIGKILL replacement, and after the workload drains. The final
acceptance gate requires zero active unary/scan calls, zero estimated bytes,
`draining=false`, `admission_closed=false`, no terminal fault, and a serving
server. This verifies the configured stream lifecycle; it is not an
admission-saturation test.

### Short smoke test

```bash
STORE=/tmp/throwdb-soak-store
IDS=/datasets/ids.npy
EMBEDDINGS=/datasets/embeddings.npy
NATIVE="$PWD/build/server-release/throwdbd"

python benchmarks/run_grpc_server_soak.py \
  --store "$STORE" \
  --ids "$IDS" \
  --source-embeddings "$EMBEDDINGS" \
  --results benchmark-results/grpc-soak-smoke \
  --native-binary "$NATIVE" \
  --tuning conservative \
  --max-seconds 180 \
  --max-transfer-gib 4 \
  --max-client-memory-mib 1024 \
  --max-get-calls 120 \
  --max-mutations 40 \
  --restart-cycles 2 \
  --restart-every-gets 40 \
  --stream-get-rows 65536 \
  --scan-records-rows 65536 \
  --stream-batch-rows 4096 \
  --stream-max-batch-mib 8 \
  --stream-attempt-seconds 60 \
  --stream-progress-timeout-seconds 30 \
  --stream-stop-timeout-seconds 10 \
  --stream-oracle-rows 256 \
  --stream-source-oracle \
  --scan-rows 10000
```

This smoke verifies the bounded unary capture against the source file once
during setup. Omitting `--stream-source-oracle` still checks every streamed row
against that capture, but no longer proves that the captured bits match the
external source.

### Resource accounting

`--max-seconds` and `--max-transfer-gib` are run-wide work ceilings.
`--max-mutations` independently caps unique mutation UUIDs without stopping
read/stream work, while `--max-get-calls` adds a deterministic unary-workload
endpoint and may be `0` for a duration/transfer-driven soak. Transfer
accounting includes dominant ID, embedding, delete-mask, and scan-location
payloads plus conservatively reserved mutation-cleanup headroom.

Streaming traffic is charged one conservatively sized batch at a time. Before
opening either stream, the harness reserves one complete maximum requested
batch because the first data frame may already be buffered by the open call.
Before requesting each subsequent batch it reserves the same maximum, even if
the server will return fewer rows because of payload or logical-tail limits.
It therefore may exhaust the accounting ceiling before actual logical bytes
reach the nominal limit, but it does not precharge an unconsumed multi-batch
tail and cannot continue receiving batches after the ceiling is exhausted.
Budget exhaustion cancels the generation's peer streams and ends new workload
submission instead of silently leaving a worker stopped.

The accounting excludes protobuf tags, HTTP/2 framing, physical read
amplification, page cache, and source-oracle memmap reads. New unary RPC
deadlines are clipped to the remaining wall-clock budget. Each dedicated
streaming client instead receives a separate deadline derived when its process
generation starts from the attempt duration, normal RPC timeout, remaining run
time, and stop allowance. Best-effort canary cleanup and the final atomic
report may extend slightly past the work deadline.

`--max-client-memory-mib` is a fourth, independent client-side bound. Before
work begins, the harness checks one aggregate estimate containing every
in-flight Get response, three payload-sized buffers per streaming worker, and
the `StreamGet` ID-batch working arrays. Later checks combine sampled client RSS
growth with the next operation's conservative temporary-memory estimate. This
is benchmark fail-fast accounting rather than an operating-system memory
cgroup; retained NumPy objects, source-memmap residency, and allocator behavior
remain visible in the sampled RSS.

`--max-errors` can tolerate only explicitly transient Get statuses
(`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED`, or `CANCELLED`);
wrong bits/shapes, missing sampled IDs, protocol failures, and `DATA_LOSS`
always fail immediately and are never converted into an availability pass.
Expected stream interruption during the marked SIGKILL window is counted in
the streaming report rather than against this ordinary error allowance.

`--max-mutations` counts unique operation UUIDs, not retry attempts. The
harness always preserves one emergency-Delete slot and all not-yet-run restart
probes before admitting another periodic four-operation CRUD cycle. This keeps
the documented restart schedule reachable even when periodic mutations would
otherwise consume the ceiling. The final probe is read-only and reports every
canary as `absent`, `present`, or `unknown` without spending another UUID.
Intervals skipped for mutation/restart/cleanup headroom or another run budget
are aggregated in the bounded `mutation_schedule.skipped_intervals` and
`skip_reasons` counters rather than appended as one report row per interval.
After all requested restarts, an exhausted mutation allowance disables further
periodic cycles and records `disabled_reason`; Gets and streams continue until
their own stopping bound is reached.
Read workers use separate clients, so per-RPC deadline updates cannot race
between concurrent Gets.

### Revision-2 report

`report.json` is checkpointed atomically. An initial `status=running`
checkpoint is published immediately after the result directory is created,
before server or monitor startup. A startup failure replaces it with
`status=failed`, a traceback, and `error.phase=server_start` or
`monitor_start`; this early artifact is intentionally partial because workload
sections do not exist yet. Later safe milestones and the final acceptance
result are also atomically published.

`metrics.jsonl` is append-only and records server generation, client/server
`/proc` RSS, CPU, faults, thread/FD counts, `/proc/PID/io`, and cgroup CPU
counters; safe workload milestones also include Health, admission,
engine/WAL/overlay, journal, free-space, and maintenance state. Automatic
maintenance is exercised as ambient server behavior. The `maintenance`
section checks persistent engine invariants and reports publication evidence
inferred from materialized-sequence, delta-run, and frozen-WAL transitions.
There is still no completed-cycle counter, so a normal run does not claim that
a particular policy action ran. Use `--require-maintenance-progress` only when
the collection and server thresholds are intentionally configured to make such
a publication observable; it then becomes a hard gate.

Revision 2 adds these top-level sections:

- `fairness_baseline` and `fairness_mixed` record the matched concurrent
  smallest-Get windows, shared deterministic query digest, worker/in-flight
  topology, and successes; `fairness` contains successful-sample counts, both
  success-only P99 values, their ratio, configured limit, evaluability, and
  pass/fail reason.
- `stream_get_oracle` records the bounded unary capture, payload digest, and
  whether it was independently verified against the source file.
- `streaming.configuration` records the effective harness-side settings.
  `streaming.modes` aggregates worker/attempt outcomes, batches, rows, logical
  bytes, captured-oracle/shadow checks, open latency, maximum inter-batch gap,
  active-time rates, snapshots, and digests for `stream_get` and
  `scan_records`.
- `streaming.generations` records why each generation stopped, per-worker and
  per-mode progress, worker join state, and whether a forced interruption was
  expected.
- `streaming_admission` stores the open/drain `Describe` observations.
  `streaming_gates` records per-mode liveness/correctness gates,
  generation/restart coverage, and admission lifecycle results.
- `mutation_schedule` contains bounded aggregate counters for skipped mutation
  intervals and the reason periodic mutations were disabled, if any.
- `maintenance` records initial/final engine counters, sequence/free-space
  invariants, inferred publication evidence, and the optional progress gate.
- `metrics.operations` includes the existing Get/CRUD measurements plus
  `baseline_get_*`, `stream_get_open`, `stream_get_batch`,
  `scan_records_open`, and `scan_records_batch`.

The final report also retains all-attempt counters, errors by operation/status,
per-generation recovery time, RSS/FD growth, cgroup throttling, and a flattened
`acceptance` map for CI. The command exits nonzero for a wrong shadow/source
read, stream ordering/shape/location failure, stalled or leaked stream,
insufficient fairness samples, P99 gate failure, lost/rejected replay, changed
replay domain, unexpected RPC error, incomplete restart, excessive recovery
time, unexpected server exit, failed `ScanIds` ordering, or canary restoration
failure.

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
External mode fails fast rather than silently ignoring local-launch settings.
It requires `--restart-cycles 0` and rejects a non-default `--tuning`,
`--native-binary`, server startup/worker/admission/request/scan/watchdog/
shutdown/maintenance overrides, and `--server-no-auto-maintenance`.
`--server-pid` remains telemetry-only, and `--allow-external-mutations` is
still required because this soak always exercises mutation semantics.

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
  --max-transfer-gib 2048 \
  --max-client-memory-mib 32768 \
  --max-get-calls 0 \
  --max-mutations 2000 \
  --get-batch-sizes 1,128,1024,10000 \
  --stream-get-workers 1 \
  --scan-records-workers 1 \
  --stream-get-rows 131072 \
  --scan-records-rows 131072 \
  --stream-batch-rows 16384 \
  --stream-max-batch-mib 32 \
  --stream-prefix-dimensions full \
  --stream-attempt-seconds 300 \
  --stream-progress-timeout-seconds 60 \
  --stream-stop-timeout-seconds 15 \
  --stream-oracle-rows 256 \
  --stream-source-oracle \
  --fairness-baseline-get-calls 200 \
  --min-mixed-small-get-successes 200 \
  --max-small-get-p99-ratio 2 \
  --min-stream-batches-per-mode 100 \
  --restart-cycles 2 \
  --restart-every-gets 50000 \
  --max-recovery-seconds 60 \
  --max-rss-growth-mib 1024 \
  --max-fd-growth 32 \
  --scan-rows 1000000
```

The production example source-verifies the bounded unary StreamGet capture
once during setup; long streams then compare against memory and do not compete
with repeated source-file reads. It also tightens the P99 ratio only after
moving to a real, stable host. Wall-time or transfer exhaustion ends new work;
the Get-count bound ends the ordinary unary loop, while mutation-ceiling
exhaustion disables periodic mutation cycles but does not stop reads or
streams. Size `--max-transfer-gib` for the requested duration and streaming
prefix.

The large-host run is the authoritative acceptance artifact. A passing small
run proves bounded orchestration, stream liveness/admission cleanup,
CRUD/replay semantics, restart recovery, and reporting, but cannot predict
1.5-TiB page-cache behavior, NVMe latency tails, cgroup throttling, free-space
pressure, or long-term RSS/FD slopes.

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

## Read-workload logger qualification

The native logger has a dedicated deterministic suite:

```bash
cmake --preset server-release
cmake --build --preset server-release
ctest --test-dir build/server-release \
  --output-on-failure \
  -R 'throwdb_(read_workload_log|grpc_server)_tests'
```

It covers binary header/frame CRCs, exact multi-chunk request reconstruction,
original ID order and digests, concurrent producer admission, request
completion/failure/cancellation, queue saturation, crash-torn tails, idle
checkpoint stability, rotation, exact terminal-to-opening baselines,
retained-window loss accounting, private permissions, exclusive ownership,
minimum-free-space pause, runtime fail-open behavior, terminal-checkpoint
finalization, write-rate/file-count/retention bounds, and the actual
unary/streaming gRPC hooks. It also verifies that an unused pre-bind segment is
removed and that minimum-free-space or finalization failure never publishes
`.tdbrl`.

Functional tests do not establish the production overhead of shared atomic
counters, exact ID copying, and profiled native reads. Before enabling
telemetry on a high-QPS
collection, run the existing gRPC benchmark twice on the same isolated host,
reader profile, deterministic corpus, cache mode, and repetition count:

1. server without `--read-log-dir`;
2. server with `--read-log-dir` and the intended chunk/resource bounds.

Use a new result directory for each run. Compare client throughput and
P50/P95/P99 latency by identical query digest, plus server CPU, RSS, cgroup
throttling, and disk telemetry. The enabled run must also satisfy:

- no read correctness or RPC failures;
- `runtime_failed=false` and `writer_errors=0`;
- both completed-event and completed-row outcome drop ratios below their
  deployment quality thresholds;
- RSS within the configured queue plus normal measurement noise;
- finalized/recoverable file count within its configured ceiling;
- retained bytes within the configured retention contract, accounting only
  for the documented active-segment allowance.

Finally stop the server cleanly and run:

```bash
throwdb workload inspect \
  --logs "$READ_LOG_DIR" \
  --storage "$STORE" \
  --max-drop-ratio 0 \
  --max-row-drop-ratio 0 \
  --require-suitable \
  --output "$RESULTS/workload-inspection.json"
```

The optimizer must consume only a suitable, exact-capture report. Interior
request/chunk gaps, stale/missing terminal checkpoints, missing or
discontinuous opening baselines, sequence gaps, counter regressions, any
retained-window record/row loss, and corrupt finalized files fail the
suitability gate. Retention-boundary left/right censoring is reported and
excluded rather than used as a partial request. Inspector inputs are themselves
hard-bounded; use `--max-files`,
`--max-total-bytes`, `--max-file-bytes`, `--max-report-entries`, and
`--max-prefix-entries` (plus the active-request bound) when a deliberate test
corpus exceeds the defaults.
The file inventory has a non-overridable 65536-entry ceiling; split a larger
offline corpus rather than making one inspection walk unbounded.
A logger overhead regression does not justify relaxing fail-open or resource
bounds. Adjust ID chunk size, queue/write/retention capacity, or leave
telemetry disabled while investigating.

## Internal I/O workers versus reader concurrency

These are independent axes. `--io-workers` controls the C++ executor owned by a
single `Collection`; the concurrent-reader scenarios control how many Python client
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
