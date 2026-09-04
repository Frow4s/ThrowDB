# C++ gRPC single-collection server

The public gRPC contract is
[`proto/throwdb/v1/throwdb.proto`](../proto/throwdb/v1/throwdb.proto). It is a
first-class transport over the same transport-neutral collection engine as the
embedded Python API. The earlier internal TDBP/TCP comparison transport has
been removed; gRPC is the supported network API.

## Build contract

The server release preset always requests gRPC:

```bash
cmake --preset server-release
cmake --build --preset server-release
ctest --preset server-release
```

The build requires CMake package configurations that provide all of these
imported targets:

- `protobuf::protoc`
- `protobuf::libprotobuf`
- `gRPC::grpc_cpp_plugin`
- `gRPC::grpc++`

Install Protobuf and gRPC into a system prefix, or point CMake at a private
prefix:

```bash
cmake --preset server-release \
  -DCMAKE_PREFIX_PATH="$HOME/.local/throwdb-grpc"
```

### Ubuntu without root access

[`tools/bootstrap_grpc.sh`](../tools/bootstrap_grpc.sh) clones the pinned
gRPC `v1.81.0` commit (including its pinned submodules), builds only the C++
code generator/runtime needed by ThrowDB, and installs it under the current
user. It does not use `sudo` or a system package manager. The machine needs
outbound GitHub access plus `git`, CMake, a C++ compiler, `make`, Perl, and
Python 3.

```bash
export THROWDB_GRPC_PREFIX="$HOME/.local/throwdb/grpc-1.81.0"
export THROWDB_GRPC_JOBS=4

./tools/bootstrap_grpc.sh
cmake --preset server-release
cmake --build --preset server-release
ctest --preset server-release
```

To build the complete Python distribution from source, reuse the same exported
prefix. `pip` preserves it for the isolated scikit-build/CMake process:

```bash
python -m pip install .

throwdbd --help
```

Editable installation uses the same discovery path:

```bash
python -m pip install -e '.[test]'
```

The canonical protocol is checked in together with its generated Python client
so installing ThrowDB does not require a Python `protoc` step. After changing
`proto/throwdb/v1/throwdb.proto`, regenerate the three checked-in outputs with
the pinned `grpcio-tools` test dependency and review them in the same commit:

```bash
python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=python \
  --pyi_out=python \
  --grpc_python_out=python \
  proto/throwdb/v1/throwdb.proto
```

The CMake cache variable `THROWDB_GRPC_PREFIX` is initialized from the
environment variable with the same name, and an empty older cache entry falls
back to the current environment. An explicit non-empty
`-DTHROWDB_GRPC_PREFIX=/another/prefix` overrides that default. The generic
`CMAKE_PREFIX_PATH` remains supported for system or multi-prefix setups, but is
not required for the bootstrap workflow.

The source/build cache defaults to
`$XDG_CACHE_HOME/throwdb/grpc-1.81.0` (or `~/.cache/throwdb/...`) and can be
moved with `THROWDB_GRPC_WORK_DIR`. Re-running the script reuses the build but
refuses a source checkout whose commit is not the pinned one.

`THROWDB_BUILD_GRPC_SERVER=ON` uses `find_package(... REQUIRED)`. A missing
compiler, plugin, header, or C++ library is therefore a configuration error;
the build fails if its required transport dependencies are missing.

The release preset also enables `THROWDB_GRPC_ENABLE_REFLECTION`. This
requires `gRPC::grpc++_reflection` and exposes the service to `grpcurl` and
similar diagnostics. A minimal distribution may turn reflection off
explicitly without disabling the primary transport.

Official Linux wheels build with the pinned toolchain and install both the
Python gRPC client and the public `throwdbd` launcher. The launcher resolves a
reader profile once and then replaces itself with the wheel's private C++
`throwdb/libexec/_throwdbd_native` daemon. Python is never in the RPC data
path. A missing C++ gRPC toolchain therefore fails the wheel build rather than
silently publishing an incomplete server artifact. Core-only developers
may explicitly configure both
`THROWDB_BUILD_SERVER=OFF` and `THROWDB_BUILD_GRPC_SERVER=OFF`; the Python CI
matrix uses that mode once per Python version, while a dedicated job builds and
tests the complete pinned server.

### Checkpoint verification

The following sequence exercises the complete source checkpoint, including the
native format-7 lifecycle tests, gRPC transport tests, and Python subprocess
tests against the just-built daemon:

```bash
export THROWDB_GRPC_PREFIX="$HOME/.local/throwdb/grpc-1.81.0"
export THROWDB_GRPC_JOBS=4

./tools/bootstrap_grpc.sh
python -m pip install -e '.[test]'
cmake --preset server-release
cmake --build --preset server-release
ctest --preset server-release

THROWDB_SERVER_BINARY="$PWD/build/server-release/throwdbd" \
  python -m pytest -q
```

## Tune-aware startup

The two public entry points are equivalent:

```bash
throwdbd \
  --storage /data/example \
  --listen 127.0.0.1:50051

throwdb serve \
  --storage /data/example \
  --listen 127.0.0.1:50051
```

They perform a bounded control-plane operation: inspect the collection and
saved tuning registry, validate one complete effective reader configuration,
and call `execve()` on the C++ daemon. Startup never runs `throwdb tune` or any
other calibration workload.

`--tuning` has three modes:

| Mode | Behaviour |
| --- | --- |
| `auto` (default) | Reuse the strongest safe persisted match, or start with conservative defaults and report the reason for the fallback. |
| `persisted` | Require a valid reusable persisted profile and fail before binding if none can be selected. |
| `conservative` | Bypass the registry and use the built-in reader defaults. |

Any explicit range or `--io-*` setting with the default `auto` mode selects a
`manual` configuration. Missing fields are filled from built-in defaults, not
from a persisted profile, so the result is deterministic and cannot become a
surprising hybrid. Explicit reader settings are rejected with
`--tuning persisted` or `--tuning conservative`.

Use `--dry-run` to print the complete native argument vector, reader settings,
and selection diagnostics without starting the daemon:

```bash
throwdbd \
  --storage /data/example \
  --tuning auto \
  --dry-run
```

The dry run still validates that the private native executable exists. This
turns it into an installation and profile-resolution check, not just argument
parsing.

An installed wheel keeps the native daemon at the internal package path so it
cannot collide recursively with the PATH-level launcher. A standalone CMake
build intentionally retains a raw native `build/server-release/throwdbd`.
Executing that file directly bypasses the registry and reports
`status=registry_bypassed`; this is useful for low-level diagnostics, not the
normal product entry point. To test a source build through the same public
control plane, select it explicitly:

```bash
throwdbd \
  --native-binary "$PWD/build/server-release/throwdbd" \
  --storage /data/example \
  --listen 127.0.0.1:50051
```

After profile selection, `auto` has one deliberately narrow safety net. If and
only if initialization of a persisted reader backend fails, the native daemon
retries once with conservative defaults and reports
`status=backend_initialization_failed`. It does not mask collection corruption
or ordinary storage-open failures. `persisted` and `manual` configurations
always fail closed.

## Safe access, transport, and resource defaults

The daemon opens the collection read-only unless `--writable` is explicit.
This applies to both the Python launcher and the native binary. A writable
server starts bounded automatic maintenance unless `--no-auto-maintenance` is
passed; `--maintenance-max-actions 0` also disables that worker.

Unauthenticated plaintext is accepted only on a canonical numeric loopback
bind (`127.0.0.0/8` or `[::1]`). Hostnames such as `localhost` are deliberately
not trusted as proof of where a resolver will bind. A non-loopback endpoint
requires mutual TLS:

```bash
throwdbd \
  --storage /data/example \
  --listen 0.0.0.0:50051 \
  --tls-cert server.crt \
  --tls-key server.key \
  --tls-client-ca clients-ca.crt
```

The Python client supplies the corresponding channel credentials:

```python
import grpc
import throwdb

credentials = grpc.ssl_channel_credentials(
    root_certificates=open("server-ca.crt", "rb").read(),
    private_key=open("client.key", "rb").read(),
    certificate_chain=open("client.crt", "rb").read(),
)
remote = throwdb.RemoteCollection.connect(
    "throwdb.internal:50051",
    credentials=credentials,
)
```

`--allow-insecure-non-loopback` is an explicit unsafe escape hatch for a
separately protected trusted network. The override removes the mTLS requirement
and adds no protection by itself. When server certificate/key files are also
supplied, TLS still encrypts the connection and authenticates the server, but
clients are not authenticated without a client CA. ThrowDB has no
application-level users, roles, or per-operation authorization yet.

The launch defaults bound the two memory layers independently:

- 64 MiB maximum parsed request and serialized response;
- 32 admitted data-plane calls and 2 concurrent scans, with the invariant
  `max_concurrent_scans < max_concurrent_calls < max_worker_threads`;
- 256 MiB conservative estimated in-flight memory reserved for unary Get/CRUD;
- a separate 128 MiB streaming cursor/response pool shared by `ScanIds`,
  `StreamGet`, and `ScanRecords`;
- 512 MiB gRPC resource quota for transport/parser buffers;
- 64 gRPC worker threads; Health and Describe bypass application data-plane
  admission, although they still share the gRPC worker pool and
  transport-memory quota.

Data calls fail fast with `RESOURCE_EXHAUSTED` instead of entering an unbounded
application queue. Mutations additionally share one fail-fast server lane,
matching the storage engine's sole commit lane. Any ambiguous mutation error
must be retried with the same `operation_id`. Health and Describe bypass data
admission. Both the typed and standard Health services change to
`NOT_SERVING` during drain and after a terminal storage/corruption fault; the
admission gate then rejects every new Get, CRUD, `ScanIds`, `StreamGet`, and
`ScanRecords` call with
`UNAVAILABLE` while already admitted calls drain.

Every `Describe` response also carries a lock-consistent admission snapshot:
active unary/scan calls, estimated bytes in each independent pool, monotonic
saturating admit/reject counters for each class, and the
`admission_closed`/`draining`/`terminal_faulted` lifecycle flags. The same
snapshot is available to C++ embedders through `GrpcServer::admission_stats()`
and to Python clients as `RemoteDescription.admission_stats`. These counters
cover decisions made by the admission gate; schema validation, health closure,
and other failures that occur before admission are intentionally not counted.

Each streaming read reserves one equal share of the independent scan byte
budget (`max_scan_inflight_bytes / max_concurrent_scans`) for its whole
lifetime. Streaming leases never debit the unary pool, and their concurrency
limit leaves both a data-call slot and a gRPC worker available outside the
streaming pool. The native
layer checks the exact current WAL-overlay cardinality under the same snapshot
lock before allocating the cursor copy. A heavily mutated, unmaterialized
overlay therefore fails with `RESOURCE_EXHAUSTED` instead of escaping
admission; materialize it, reduce scan concurrency, or raise
`--max-scan-inflight-mib`.

The server independently bounds every stream to one hour and requires
transport progress at least once per minute by default. A successful input
frame or output frame counts as progress; merely keeping a bidi RPC open does
not. `--scan-max-duration-ms` and `--scan-idle-timeout-ms` override those
policies. A watchdog calls gRPC `TryCancel`, including while a synchronous
`Read` or `Write` is blocked. Raw gRPC reports that server-side watchdog path
as `CANCELLED`; a client-supplied deadline reports `DEADLINE_EXCEEDED`. In both
cases the handler unwinds and releases its streaming slot and byte reservation
before another stream is admitted. The Python client additionally uses a
one-hour scan deadline by default; explicitly passing `scan_timeout=None`
disables only that client deadline, not the server bounds.

## Optional bounded read-workload logging

The native server can record complete logical point-read traces for the offline
layout optimizer. This is fail-open telemetry, not authoritative collection
state or an audit WAL. It is disabled unless `--read-log-dir` is supplied:

```bash
throwdbd \
  --storage /nvme/collections/products \
  --listen 127.0.0.1:50051 \
  --read-log-dir /nvme/throwdb-workload/products
```

The log directory must be dedicated and disjoint from the collection tree. It
may be a sibling, but it may not equal the collection, be inside it, or contain
it. This prevents raw request IDs from entering collection backup/restore and
prevents the logger from changing permissions on a shared collection
ancestor. The launcher and native daemon both validate this before serving.
The directory is permanently bound to the collection ID by its owner marker.
After rebuilding the collection under a new ID, archive or remove the old
dedicated log directory and start the new collection with a fresh one.

Logging creates one lifecycle for each unary `Get` and one for the complete
bidirectional `StreamGet`. After a valid ID batch is decoded, every submitted
`uint64` ID is retained in original order, including duplicates. A request is
ended with its actual success, cancellation, deadline, client-close, or failure
status. Only successful, fully captured requests are eligible for layout
training. `ScanIds` and `ScanRecords` do not contain a user-supplied ID set and
are excluded. Embeddings and per-ID physical pointers are never logged.

The logger does not infer categories or label a request as broad/full. Its only
operation value is the real `Get` or `StreamGet` method. Larger inputs are split
into contiguous physical ID records, not sampled or truncated.

The defaults after opt-in are:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `--read-log-id-chunk-rows` | `8192` | exact ordered IDs per physical record; hard maximum is 8192 |
| `--read-log-buffer-mib` | `256` | preallocated multi-producer queue |
| `--read-log-segment-mib` | `256` | rotate sequential binary segments |
| `--read-log-rotate-seconds` | `3600` | time-based rotation |
| `--read-log-retention-mib` | `51200` | total recognized-log budget, including the active segment |
| `--read-log-retention-seconds` | `604800` | seven-day age bound |
| `--read-log-min-free-mib` | `10240` | pause before consuming the filesystem reserve |
| `--read-log-max-write-mib-per-second` | `16` | token-bucket cap for all sequential logger bytes |
| `--read-log-max-retained-files` | `4096` | total cap including the active segment; supported range is 1–65533 |

The buffer must fit at least two maximum-sized records. A segment must fit its
header, opening counter baseline, one maximum record, and terminal checkpoint.
Retention must cover at least one segment, and the write-rate cap must permit
that complete bounded transaction in one token-bucket window.
Tuning flags without `--read-log-dir` are rejected instead of silently enabling
telemetry.

Retention prunes recognized finalized files and stale `.open` tails
oldest-first until both the byte and file-count budgets hold. While the logger
is running it reserves one configured file slot for the current writer-owned
`.open` segment, so the total recognized file count never exceeds the
configured ceiling. It also reserves one full configured segment from the byte
budget; finalized/stale files plus the active segment therefore stay within the
configured retention bytes. Age retention is enforced during traffic and by a
periodic idle sweep at least every five minutes.

Disabled mode allocates no logger or queue and adds only an atomic null-pointer
check to point reads. Enabled reads call the explicitly profiled native path and
enqueue fixed-capacity frames through a preallocated bounded MPSC ring. A
producer never waits for the writer. One background thread writes private
`0600` CRC32C-framed files in a `0700` directory. Only one process may own a
log directory.

Every trace contains:

- request begin/end, exact snapshot anchor when available, prefix dimensions,
  status, submitted/delivered rows, and response bytes;
- exact ID chunks with logical offset and input-batch/chunk sequence, plus
  successful streaming-delivery order in the corresponding batch record;
- lookup, physical sort/plan/backend, construction, and transport timings;
- range/task/file fanout, requested/useful/planned/completed bytes,
  merge-gap amplification, 4 KiB page and 1 MiB chunk runs, and input-order
  discontinuities;
- the effective reader backend, merge geometry, workers, queue/task sizing,
  and `io_uring` topology used by that server process.

Explicit startup errors such as invalid geometry, an unsafe/unwritable path, or
a conflicting writer lock fail server startup. After successful startup,
queue pressure, the minimum-free-space guard, and log I/O failures never fail
or delay the read RPC:

- queue pressure increments full-record/ID counters and marks the request
  incomplete;
- low free space pauses admission and is rechecked periodically;
- a runtime filesystem error permanently disables that logger instance until
  server restart and leaves a recoverable `.open` tail;
- shutdown first drains all gRPC handlers, then drains accepted telemetry;
- an active segment becomes `.tdbrl` only after its fresh terminal checkpoint,
  file fsync, rename, and directory fsync succeed;
- every segment starts with the exact terminal-counter baseline carried from
  its predecessor, so deleting an old retained prefix does not turn lifetime
  loss counters into loss attributed to the remaining files;
- minimum-free-space refusal or any terminal-checkpoint/finalization failure
  leaves `.open`; a pre-bind segment that never accepted an event is removed
  instead of becoming an empty finalized file.

`RemoteCollection.describe()` exposes the effective bounded policy and
process-local counters through `read_logging` and `read_logging_stats`, without
disclosing the server-local path. Protocol v1.5 exposes request, record,
ID-chunk, batch, ID, queue, and reason-specific loss counters. Deprecated v1.4
sampling fields remain compatibility aliases; a v1.5 server reports
`sample_ppm=1000000` and never reports truncation. Alert on
`runtime_failed`, `writer_errors`, any incomplete request, or any dropped
record/ID.

Inspect logs offline before using them for optimization:

```bash
throwdb workload inspect \
  --logs /nvme/throwdb-workload/products \
  --storage /nvme/collections/products \
  --max-drop-ratio 0 \
  --max-row-drop-ratio 0 \
  --require-suitable \
  --output workload-report.json
```

The inspector validates collection/schema identity, strict frame geometry,
CRC32C, request/chunk/batch sequence and offsets, the full-sequence digest,
configured resource bounds, cumulative counter consistency, baseline
continuity between adjacent segments, terminal completeness, and
reason-specific loss. It reports aggregates and never prints captured IDs.
Requests crossing an already-pruned oldest prefix are left-censored; requests
unfinished in the newest recoverable tail are right-censored. Both are
excluded. An interior missing chunk, counter/segment discontinuity, corrupt
frame, or any retained-window logger drop makes `exact_capture=false` and the
v2 corpus unsuitable for layout optimization.

Inspection itself is bounded. Defaults are 65536 files, 64 GiB total input,
4 GiB per file, 1024 per-file JSON reports, and 4096 distinct prefix-dimension
buckets/active request states. Override them explicitly with `--max-files`,
`--max-total-bytes`, `--max-file-bytes`, `--max-report-entries`,
`--max-prefix-entries`, and the active-request bound.
The latter is exposed as `--max-active-requests`.
`--max-files` may be lowered but cannot exceed the hard 65536-entry directory
inventory ceiling; the inspector also counts a small bounded allowance of
service files and rejects directories that exceed it. Byte-valued inspector
arguments are raw bytes. Exceeding any hard input or prefix bound aborts
inspection instead of producing a partial suitable report.

Raw IDs make this directory sensitive. Keep it off collection/S3 sync and
routine support bundles, use an encrypted volume, and restrict access to the
server/optimizer identity. Retention unlink is not secure erasure on SSD/NVMe.
The 50-GiB budget stores approximately 6.7 billion raw numeric ID occurrences
before framing overhead, including the reserved active segment.

## Processes and binaries

- `throwdbd` and `throwdb serve`: public one-shot tune-aware launchers; after
  successful startup the process image is the C++ gRPC service.
- `throwdb/libexec/_throwdbd_native`: private daemon inside a wheel.
- `build/server-release/throwdbd`: raw daemon produced by a standalone source
  build; it bypasses the tuning registry unless selected through the launcher
  with `--native-binary`.

The public daemon opens one format-7 collection and enforces its single-writer
lease. gRPC v1 provides `Health`, `Describe`, `Get`, bidirectional `StreamGet`,
`Insert`, `Update`, `Upsert`, `Delete`, and server-streaming `ScanIds` and
`ScanRecords`. `StreamGet` negotiates a stop-and-wait application window of
one; the official client therefore never submits an unbounded ID request.
`ScanRecords` relies on HTTP/2 flow control and the dedicated scan admission
pool. Both operations pin one view, repeat its opaque identity on responses,
and fail instead of falling back to multiple current-view unary reads.

The daemon binds its socket in `NOT_SERVING`, starts the optional maintenance
thread, and only then atomically marks both health services `SERVING` and emits
the newline-delimited ready record. A background-thread startup failure can
therefore never leave a false ready event or transient serving probe. The ready
record on stdout and `Describe` expose the same
effective startup state: reader/tuning selection plus unary/scan admission,
stream lifetime, graceful-drain, and hard process-exit bounds. `Describe` also
exports live admission and engine state, including WAL/overlay backlog,
idempotency-journal use and generation, filesystem headroom, and maintenance
activity. Clients connected to older protocol servers receive `None` for
optional diagnostic objects rather than fabricated defaults.
Protocol-v1.0 writable servers also predate replay domains: the client accepts
their advertised durable operation-ID replay with `idempotency_domain = None`
for rolling upgrades. Such a connection retains the old server's replay
semantics but cannot reject a delayed retry after replay history is retired;
restart the client against a v1.1+ server before relying on domain fencing.

`throwdbd` additionally registers the standard gRPC health service for native
Kubernetes/Airflow gRPC probes; the typed ThrowDB `Health` method remains part
of the public application protocol. The C++ integration test calls
`grpc.health.v1.Health/Check` directly and expects `SERVING`.

Reflection is compiled and tested only when
`THROWDB_GRPC_ENABLE_REFLECTION=ON`. With it enabled, diagnostics can discover
both the ThrowDB and standard health services:

```bash
grpcurl -plaintext 127.0.0.1:50051 list
grpcurl -plaintext -d '{"service":""}' \
  127.0.0.1:50051 grpc.health.v1.Health/Check
```

With reflection disabled the primary RPC and standard health services remain
available; only descriptor discovery is omitted.

Writable servers run one bounded lifecycle cycle every 60 seconds by default.
Every cycle executes at most `--maintenance-max-actions` single-action
decisions, six by default. Before each decision the engine re-evaluates replay-
journal checkpoint, WAL rotation, materialization, partition folding, selective
segment cleaning, then orphan reclamation in that priority order. A backlog in
an earlier stage can consume the cycle, leaving later stages for a subsequent
interval. Crossing the byte rotation threshold also rotates immediately after
a successful mutation. Before PREPARE, a commit that would cross the larger
materialization-input bound first rotates the current WAL; one frame that
cannot fit by itself is rejected and must be split or paired with a larger
limit. A pre-existing oversized WAL still requires a larger explicit
maintenance budget. Materialization, folding, and cleaning preserve the
configured disk reserve before starting. Low space defers the pass and leaves
reads available; an actual lifecycle storage/durability failure makes Health
`NOT_SERVING`, starts daemon drain, and exits nonzero so a process supervisor
can replace the lease-owning instance.

Maintenance does not hold the server mutation lane for a complete lifecycle
build. Journal reconciliation/checkpoint and WAL rotation remain serialized
with `PREPARE -> native commit -> COMPLETE`, because they can change the replay
domain or freeze the WAL required for recovery. Expensive processing of an
already selected frozen WAL, delta-run prefix, or segment runs outside that
lane, so unrelated mutations can continue; native publication is still
serialized internally. Format-7 orphan GC likewise performs its complete
filesystem inventory, classification, and bounded selection without the
mutation lane, then briefly reacquires the native mutation/commit/view locks to
re-read `CURRENT`, revoke candidates that became reachable, revalidate file
identities, unlink, and fsync. If a durable pending-mutation fence appears just
before lifecycle admission, the action makes no destructive progress and the
pass reports a transient `pending_mutation_deferred`. The worker retries on a
later cycle instead of treating that condition as a terminal storage fault.

Automatic cleaning is deliberately conservative. By default it scans at most
eight candidate sidecars once per hour, rewrites at most one segment, requires
at least 512 MiB and a 30% garbage ratio to be reclaimable, and obeys explicit
source/output/memory bounds. `--no-auto-clean` disables it; the `--cleaner-*`
arguments change the bounded policy. Explicit `throwdb clean-segments` and
embedded/CLI `maintain()` remain available for operator-directed work. A native
action already in progress is cooperative rather than cancellable.
Shutdown first prevents another action and gives gRPC its configured 30-second
grace. The standalone daemon then has an absolute 60-second process bound
(`--shutdown-hard-timeout-ms`, which must not be shorter than
`--shutdown-grace-ms`). At that bound it uses `_Exit(EXIT_FAILURE)`; native
lifecycle actions and mutations are crash-safe and recover on the next open.
Choose a larger hard bound for unusually slow media, or disable automatic
maintenance and schedule explicit maintenance if forced process termination is
not an acceptable deployment policy.

`operation_id` is mandatory and nonzero in every public mutation request.
Writable format-7 servers normally report `durable_idempotency = true` and a
nonzero `idempotency_domain`: within that domain, a repeated ID with the same
request digest replays the original result, while a different request is
rejected. `RemoteCollection` pins the domain for caller-supplied UUID/string/bytes IDs when the connection
opens; a subsequent `describe()` or `count` refresh does not silently change
that domain. Automatically generated IDs denote new work and use the current
advertised domain. Refresh metadata with `describe()`, then use
`remote.mutation_token()` to capture a fresh operation ID and
the currently advertised domain for a new logical mutation. Pass that token as
`operation_id=` and retain it with the original request. After an ambiguous
result, `error.retry_token` retains the exact domain and ID sent on the wire,
including across reconnects. A retired domain fails closed; never move an
uncertain operation into a new domain just to make the retry succeed.
If the capability is explicitly disabled, clients must not
automatically retry an ambiguous mutation because its WAL commit may already be
durable.

The server reserves mutation `ABORTED` for an admission failure before the
current request enters durable `PREPARE`; the current request therefore
definitely did not commit. The Python client maps it to `RemoteAbortedError`, a
typed retry signal carrying `operation`, `operation_id`, and `retry_token`, with
`definitely_not_committed = true`. This remains distinct from an unknown
transport outcome and is safe to retry even without durable idempotency. A
caller may apply bounded backoff, but must replay the identical request with
the same operation ID. Allocating a new operation ID would turn one logical
mutation into a second operation. For catch compatibility,
`RemoteAbortedError` is also an `AmbiguousCommitError`; catch the typed subclass
first. The client never retries mutations automatically.

Durable replay metadata lives at `server/idempotency-v1.log` inside the
collection. A backup, filesystem snapshot, or offline copy that is expected to
retain `operation_id` replay history must include the complete `server/`
directory at the same consistency point as the collection catalog and WALs.
Take that copy with the writable daemon stopped or through a storage-level
consistent snapshot. Omitting the journal does not change embedding bytes, but
the copied collection must then be treated as a new replay domain: previously
issued operation IDs are no longer safe retry tokens there.

Before each new mutation, the engine reserves the estimated WAL, journal, and
automatic-rotation writes while keeping 1 GiB of filesystem space free by
default (`--mutation-disk-reserve-mib`). A rejection occurs before durable
PREPARE. After PREPARE, `server/pending-mutation-v1` is a checksummed, fsynced
fence: materialization, folding, cleaning, and orphan deletion fail closed until
restart/recovery proves whether the WAL transaction committed. An ambiguous
journal append poisons the live mutation lane rather than guessing its outcome.

The replay journal has hard defaults of 4 GiB and 1,000,000 unique operation
IDs. Existing IDs remain replayable at the limit; a new ID receives
`RESOURCE_EXHAUSTED` before PREPARE. The maintenance worker checkpoints the
journal at 75% of either bound by default. Checkpoint is allowed only with no
pending PREPARE; it fsyncs and atomically publishes a fresh header, retires the
old replay history, increments the generation, and advertises a new random
domain. A request carrying the old domain is rejected before PREPARE. After a
definitive domain-expired response, call `Describe` before creating a new
operation ID. Never attach an ambiguous operation from the old domain to the
new domain: its original result is no longer provable. The ready record and
`Describe.engine_stats` expose bytes, completed/aborted counts, limits,
generation, and the current domain. Checkpointing can be disabled or its
percentage changed explicitly; doing so makes reaching the hard cap an
operator-managed availability event.

### Initial S3 disaster-recovery policy

The operational target is a latest verified S3 completion age below 24 hours.
A nominal daily schedule alone does not guarantee that bound after jitter or
failure, so schedule with margin and alert before the age reaches 24 hours.
This checkpoint is not replication, high availability, or a bounded
recovery-time guarantee.

Each successful checkpoint represents the complete collection directory,
including `server/`, except the local-only `server/s3-checkpoints/` lineage
cursors. The source must remain unchanged for the whole operation: stop the
writable daemon first, or run against a storage-level snapshot created while
writes were quiesced. Syncing a live writable directory is unsupported because
S3 makes individual object updates atomic, not a multi-file collection update.

The command requires format 7 and S3 bucket Versioning. Under the native writer
lease it hashes every regular file with SHA-256 and opens the collection with
`verify_index=True`; a pending server-mutation fence or any corrupt
manifest/catalog/index/WAL aborts before publication. The bulk transfer uses
`aws s3 sync --delete --checksum-algorithm SHA256`. Local SHA-256, rather than
mtime/size, decides equality: a changed pre-existing key is forcibly uploaded
even when high-level sync would skip it. The final current VersionId, ETag, size,
mode, and local SHA-256 of every file are recorded in a versioned completion.

ThrowDB implements that protocol through the public Python CLI/library. A
manual preflight and backup are:

```bash
STORE=/nvme/throwdb/collection
BACKUP=s3://my-versioned-bucket/throwdb/collection

throwdb backup s3 \
  --storage "$STORE" \
  --destination "$BACKUP" \
  --dry-run

throwdb backup s3 \
  --storage "$STORE" \
  --destination "$BACKUP"
```

The command acquires `locks/writer.lock` nonblockingly and retains it through
local hashing, native verification, sync, completion upload, the conditional
`LATEST` publication, and local-cursor fsync. It therefore fails closed while
a writable `throwdbd`, embedded writer, calibration, maintenance operation, or
another local backup owns the collection. A conditional
`_throwdb_backup/LEASE.json` additionally serializes independent hosts using
one S3 prefix. Stop the writable daemon first; a read-only daemon may remain
online. The AWS subprocess never inherits the writer-lock descriptor.

After `writer.lock`, backup also takes `locks/tuning.lock` for that complete
window. Every tuning-registry create/save/remove/reset takes the same barrier
before `tuning/reader-profiles.lock`, so registry bytes cannot change between
inventory and `LATEST`. Backup fails fast if a registry publication is already
active; a later registry mutation waits. This fixed order prevents a lock
cycle, and a collection with no optional tuning metadata does not gain a
`tuning/` directory merely because it was checkpointed.

This coordination requires every process touching backup or tuning metadata to
understand `locks/tuning.lock`. During a ThrowDB upgrade, quiesce checkpoints
and tuning create/save/remove/reset operations until all such workers use the
new wheel; concurrent mixed-version operation is not supported.

For the simplest daily cron or Airflow task, omit the age guard so scheduler
jitter cannot silently skip a day. If the task is launched every six hours,
redundant attempts can instead be suppressed below the RPO with:

```bash
throwdb backup s3 \
  --storage "$STORE" \
  --destination "$BACKUP" \
  --if-older-than-hours 18
```

The default `0` always runs and is the manual/daily-backup behavior. A future
timestamp in `LATEST` is rejected rather than suppressing backups indefinitely.
In every schedule, alert when the last verified completion approaches 24 hours.
The command requires a current AWS CLI and credentials for
`s3:GetBucketVersioning`, `s3:ListBucket`, `s3:ListBucketVersions`,
`s3:GetObject`, `s3:GetObjectVersion`, `s3:PutObject`, and `s3:DeleteObject`
under the selected bucket/prefix. Versioning must report `Enabled`.
`--unsafe-allow-unversioned` is retained only as a deprecated compatibility
argument and v2 still rejects it: without immutable VersionIds a version-pinned
restore and historical publication evidence cannot be guaranteed. An AWS CLI
too old to support `s3api put-object --if-match/--if-none-match` fails closed.
The S3 sentinel VersionId `"null"` is never accepted as immutable; suspending
Versioning after the initial preflight therefore aborts publication before
`LATEST` can advance.
Large segment uploads use S3 multipart transfer through the AWS CLI. The IAM
policy and bucket lifecycle should therefore also permit/clean incomplete
multipart uploads (`AbortMultipartUpload` and an
`AbortIncompleteMultipartUpload` lifecycle rule).

The remote layout is intentionally separate from the collection mirror:

```text
s3://bucket/prefix/
  slot-a/                                  # alternating payload target
  slot-b/
  _throwdb_backup/LINEAGE.json             # immutable collection ownership
  _throwdb_backup/LEASE.json               # conditional cross-host lease
  _throwdb_backup/LATEST.json              # CAS-published restore pointer
  _throwdb_backup/completions/<id>.json    # checksum + VersionId evidence
```

`LINEAGE.json` is conditionally created before the first payload transfer. A
matching LINEAGE but no LATEST is the recoverable signature of an interrupted
first publication; any nonempty prefix without LINEAGE is rejected. The remote
lease is acquired and released with ETag compare-and-swap. An active or invalid
lease, a lost lease, a CAS conflict, or an old AWS CLI is never retried over an
unknown state. A crashed publisher leaves a bounded lease; takeover is allowed
only after its advertised deadline plus a clock-skew guard.

`LATEST.json` records collection ID, catalog generation, committed sequence,
live count, a logical-view anchor digest over `manifest.bin` and `CURRENT`,
parent backup ID, and exact completion VersionIds for both slots. Generation
and sequence may only advance. Equal generation/sequence with a different
logical-view anchor or live count is a fork. Server-journal checkpoints,
orphan cleanup, tuning metadata, and other non-view bytes may legitimately
change at the same logical sequence; their per-file SHA-256 still participates
in the full-content digest and backup/restore integrity, but not in the
equal-sequence fork predicate. The full-content digest is also required for the
exact-state cursor-repair exception, so two same-sequence WAL forks cannot be
mistaken for the same publication. A destination-specific local cursor proves
that an advanced source descends from
the currently published backup; a stale copy cannot win merely by producing a
higher sequence. The only cursor-repair exception is an exact source-state
match, covering a crash after LATEST committed but before the cursor fsync.

The completion is written under a unique key, then LATEST is conditionally put
with the ETag previously read (or `If-None-Match: *` on first publication).
If the CLI reports an error after a possible commit, exact read-back may
reconcile only the identical document. A missing read-back is ambiguous and a
different document is a conflict; neither causes a retry.

Control documents are checksummed publication metadata, not cryptographic
signatures. Bucket ownership, IAM write permissions, and prefix isolation are
the trust boundary. A principal allowed to rewrite both payload and control
objects can publish a self-consistent, native-valid replacement checkpoint that
restore will accept; checksums detect accidental corruption and partial/stale
publication, not a malicious writer inside that boundary.

`aws s3 sync` does not transfer an in-file binary delta: a changed or appended
file is uploaded again as an object. ThrowDB's immutable segments and index
runs therefore make most daily transfers cheap, while hashing intentionally
reads every source byte so the checksum inventory cannot inherit sync's
mtime/size blind spot. Configure `NoncurrentVersionExpiration` only after the
chosen historical-restore window. `list-object-versions` is used to capture
VersionIds in bulk, so unbounded noncurrent-version retention also increases
checkpoint metadata cost.

### Restore drill

Restore is an offline staging operation, never an in-place update. The command
requires a new destination path, validates LINEAGE/LATEST and the exact
completion VersionId, and checks that every current slot VersionId/ETag matches
the completion both before and after the bulk download. It then verifies every
local SHA-256 and file size, recreates empty directories/modes, opens the
collection with complete native index validation, persists the local lineage
cursor, fsyncs the tree, and atomically renames staging into place:

```bash
throwdb restore s3 \
  --source "$BACKUP" \
  --storage /nvme/throwdb/restored-2026-07-17 \
  --expected-backup-id <id-recorded-by-backup-monitoring>
```

Any metadata race, missing/extra object, checksum mismatch, corrupt native
resource, wrong backup ID, or destination-path race discards the private
staging directory. The command's success is a complete file-level payload scrub
and native structural verification. A production drill should still issue
application-level sampled reads and record elapsed time/RPO evidence, then
delete the restored copy or promote it using the deployment's own service
switch-over procedure. Run this drill regularly; an unexercised checkpoint is
not a recovery guarantee.

`ScanIds` captures one snapshot and returns authoritative live IDs in physical
`(segment_id, row_index)` order. Optional locations are packed as
eight bytes per ID: little-endian `uint32 segment_id` followed by
little-endian `uint32 row_index`. They are valid only for the accompanying
snapshot metadata; clients should expose the two fields separately instead
of treating them as a stable global row number.

The server MVP accepts storage format 7 only. `throwdbd` fails before binding
a port with a clear rebuild message for legacy collections, so `Describe`
never advertises physical scan locations that the opened format cannot
provide.

## Bounded server benchmark

[`benchmarks/run_grpc_server_benchmark.py`](../benchmarks/run_grpc_server_benchmark.py)
is the dedicated correctness and load harness for this service. By default it
starts the public tune-aware launcher on `127.0.0.1:0`, records its ready
diagnostics, and stops it after the run. With `--crud-rows 0` (the default) the
daemon is opened read-only.

Report revision 3 measures controlled cache states, repeats every deterministic
workload three times by default, and adds exact-corpus references for the mixed
small/large workload. Its quality-oriented defaults use 100 sequential calls,
20 concurrent calls per worker, and 200 small plus 100 large mixed calls per
repetition. For the production-shaped read-only loopback run, require
the profile already saved for the collection's current storage target and run
an explicit paired cold/warm measurement:

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

`persisted` is fail-closed: the benchmark does not silently replace a missing,
invalid, or incompatible profile with conservative defaults. An installed
wheel does not need `--native-binary`; the option above deliberately exercises
the source-built daemon through the public launcher.

`--cache-mode` accepts `auto`, `ambient`, `cold`, `warm`, and `both`. The
default `auto` resolves to `cold,warm` for a locally launched server when
`POSIX_FADV_DONTNEED` is available, falls back to `warm` when that API is
unavailable, and always resolves to `warm` for `--endpoint`. Explicit `cold`
and `both` are local-launch-only: an external client cannot safely control the
server cache, even for a loopback endpoint or when given a local mirror.

Cold preparation applies advisory `POSIX_FADV_DONTNEED` only to the selected
collection's embedding payload and WAL files. It never drops the global page
cache or writes the collection, and each trial records available process
backing-read counters because the advice does not guarantee eviction. Summary
rows keep the observed classifications and explicitly mark an advisory cold
trial `cold_unverified` unless all repetitions observed backing-storage reads.
Controlled remote-versus-embedded ratios are omitted when the two observations
do not both verify the requested cache state; mixed-fairness summaries retain
the same cache-verification evidence and omit matched-isolated slowdowns unless
all three participating trials verify that state.
`warm` replays the exact measured query corpus outside the timed and telemetry
windows using the same connected client topology. Before either cold eviction
or warm replay, concurrent scenarios issue one untimed Get per connected client
to initialize an identical gRPC/Get topology. Replay and stabilization are
charged atomically to the transfer budget and appear as
`prewarm_accounted_wire_bytes` and
`stabilization_accounted_wire_bytes`. `ambient` performs neither cache-control
operation and should be treated as an observational cache state.

`--repeats` (alias `--repetitions`) defaults to three. Paired cold and warm
trials have identical query digests; repetitions use deterministic but distinct
corpora and rotate scenario order to reduce ambient-load bias. The 16 GiB
transfer bound in the full command leaves room for measured calls, exact-query
warm replay, stabilization calls, matched mixed references, and the embedded
sequential comparison.

A short read-only smoke uses smaller scenario counts, skips the embedded
throughput baseline, and retains source-backed bit-exact correctness:

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

CRUD is deliberately opt-in. Run it only against a rebuildable collection
with no other writer:

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

The CRUD cycle updates, deletes, reinserts, and finally upserts the original
bit-exact values. Successful completion therefore restores the logical
collection contents, but it intentionally leaves committed mutation and
idempotency history in the WAL/server journal. It is not a byte-for-byte
storage rollback.

To measure an already-running daemon, pass `--endpoint HOST:PORT` and use
explicit warm replay. The harness then neither launches nor stops the server.
Optional `--server-pid PID` enables Linux `/proc/PID` CPU, RSS, thread, and I/O
deltas when that PID is visible from the benchmark process:

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

`--server-pid` is valid only with `--endpoint`; `--native-binary` is valid only
for a locally launched server. A genuinely remote endpoint does not require
`--store` when source-backed correctness and `--no-embedded-baseline` are used;
the report then marks server-storage probing unavailable. Supply `--store`
only when it is an exact local mirror and an embedded comparison is meaningful.

The result directory must not already exist. `report.json` is atomically
checkpointed after every phase. Revision 2 records the resolved cache
methodology, individual repetition trials, cache preparation and observations,
query digests and pair IDs, and a `phases.summary` aggregate. Summary throughput
is total work divided by total trial wall time; available summary latency is the
median of the per-trial percentile estimates, with per-metric trial variation
retained.
The report also contains effective reader/tuning settings, bit-exact
correctness, the bounded physical-order `ScanIds` prefix, optional CRUD,
process/cgroup telemetry, and the same-query embedded comparison.

Concurrent calls enter a shared client FIFO with at most the selected worker
count in flight. `client_queue_wait` measures common-wave release to dequeue,
`client_prepare` measures dequeue to synchronous RPC start, `rpc_round_trip`
measures RPC start to response, and `client_sojourn` spans wave release through
the response. RPC round trip includes serialization, transport, server
admission/queueing/service, and response copies; it is not server service time.
All four percentile families are retained in per-class cross-repeat summaries,
including their sample distributions and coefficients of variation. The
current protocol exposes no server-internal queue/service timestamps.

For each mixed workload, isolated-small and isolated-large references use the
same IDs, cache mode, repetition, worker count, and class query digests as the
mixed wave. Revision 2 reports both RPC-P95 and client-sojourn-P95 slowdown
against those matched references. This removes query-set and one-client
baseline confounders, but it remains an end-to-end observation rather than a
server fairness SLA.

`diskstats.jsonl` samples aggregate Linux block-device counters and
`vmstat.log` is written when `vmstat` is installed; neither source is
process-attributable. Local launches additionally retain `server.stdout.log`
and `server.stderr.log`.

The three primary safety limits are `--max-seconds`,
`--max-transfer-gib`, and `--max-client-memory-mib`. They bound scheduled work,
dominant accounted request/response payloads, and incremental client working
memory respectively. Exact-query warm replay is charged; protobuf framing,
network headers, physical read amplification, page cache, the server index,
and unrelated system I/O are not claimed as part of those byte counters.
Cleanup and final report checkpointing can finish just after the work deadline.
`ScanIds` additionally stops after at most `--scan-rows` and retains only a
`--scan-batch-rows` batch while verifying strictly increasing physical
locations. The full revision-2 cache, timing, aggregation, and interpretation
contract is in [`testing.md`](testing.md).

## Current limitations and deferred hardening

The completed local format-7 lifecycle remains authoritative and is documented
in [`mutable-storage.md`](mutable-storage.md). The server deliberately exposes
that engine rather than introducing a second mutation model, but this
checkpoint has the following boundaries:

- one already-created numeric-ID format-7 collection per process, with one
  fixed `float16` or `float32` schema;
- local POSIX/NVMe backing storage only; CephFS, object storage, and replication
  do not yet share the documented durability contract;
- plaintext is restricted to canonical numeric loopback; non-loopback mTLS is
  implemented, but application users, authorization, certificate rotation, and
  per-tenant quotas are not;
- process-wide call/scan/estimated-byte admission, live admission gauges and
  counters, and a gRPC parser-memory quota are implemented, but priorities,
  deadline-aware scheduling, optional batching, and a time-series metrics
  exporter are not;
- writable servers run bounded journal-checkpoint/rotation/materialization/
  folding/cleaning/orphan work; an already-running native lifecycle action has
  cooperative rather than hard-deadline cancellation;
- journal checkpoint intentionally discards the prior domain's replay history;
  callers must retain the domain with every ambiguous operation and must never
  rebind such an operation after rollover;
- an external soak harness writes JSONL system/engine samples, but a normal
  production time-series exporter, multi-collection routing, managed
  checkpoint scheduling/restore drills, cross-region replication, and
  reduced-copy large unary RPCs are deferred.

The dedicated harness measures client-visible gRPC behaviour and concurrent
remote clients, and `Describe` exposes aggregate admission gauges/counters. It
does not yet decompose admission wait, protobuf copies, collection I/O, or
per-method service time. Device calibration stays explicit and bounded; daemon
startup only selects a compatible saved profile and never launches an implicit
benchmark.
