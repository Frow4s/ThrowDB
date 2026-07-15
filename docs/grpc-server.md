# C++ gRPC single-collection server

The public gRPC contract is
[`proto/throwdb/v1/throwdb.proto`](../proto/throwdb/v1/throwdb.proto). It is a
first-class transport over the same transport-neutral collection engine as the
embedded Python API. The earlier TDBP/TCP transport remains a build-tree-only
comparison tool; it is not the public server contract and is not installed.

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
the build never substitutes the custom TCP server silently.

The release preset also enables `THROWDB_GRPC_ENABLE_REFLECTION`. This
requires `gRPC::grpc++_reflection` and exposes the service to `grpcurl` and
similar diagnostics. A minimal distribution may turn reflection off
explicitly without disabling the primary transport.

Official Linux wheels build with the pinned toolchain and install both the
Python gRPC client and the C++ `throwdbd` command. A missing C++ gRPC toolchain
therefore fails the wheel build rather than silently publishing a client-only
or TCP-only artifact. Core-only developers may explicitly configure both
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

Until tune-aware startup is implemented, start one existing format-7
collection by passing the separately calibrated reader settings explicitly:

```bash
throwdbd \
  --storage /data/example \
  --listen 127.0.0.1:50051 \
  --io-backend io-uring \
  --io-uring-workers 1 \
  --io-uring-queue-depth 64
```

## Binaries

- `throwdbd`: public gRPC service.
- `throwdbd-tcp`: provisional build-tree target retained only for internal
  comparison.

The public daemon opens one format-7 collection and enforces its single-writer
lease. gRPC v1 provides `Health`, `Describe`, `Get`, `Insert`, `Update`,
`Upsert`, `Delete`, and server-streaming `ScanIds`.

`throwdbd` additionally registers the standard gRPC health service for native
Kubernetes/Airflow gRPC probes; the typed ThrowDB `Health` method remains part
of the public application protocol. The C++ integration test calls
`grpc.health.v1.Health/Check` directly and expects `SERVING`.

Reflection is compiled and tested only when
`THROWDB_GRPC_ENABLE_REFLECTION=ON`. With it enabled, diagnostics can discover
both the ThrowDB and standard health services:

```bash
grpcurl -plaintext HOST:PORT list
grpcurl -plaintext -d '{"service":""}' \
  HOST:PORT grpc.health.v1.Health/Check
```

With reflection disabled the primary RPC and standard health services remain
available; only descriptor discovery is omitted.

`operation_id` is mandatory and nonzero in every public mutation request.
Writable format-7 servers normally report `durable_idempotency = true`: a
repeated ID with the same request digest replays the original result, while a
different request is rejected. If the capability is explicitly disabled,
clients must not automatically retry an ambiguous mutation because its WAL
commit may already be durable.

Durable replay metadata lives at `server/idempotency-v1.log` inside the
collection. A backup, filesystem snapshot, or offline copy that is expected to
retain `operation_id` replay history must include the complete `server/`
directory at the same consistency point as the collection catalog and WALs.
Take that copy with the writable daemon stopped or through a storage-level
consistent snapshot. Omitting the journal does not change embedding bytes, but
the copied collection must then be treated as a new replay domain: previously
issued operation IDs are no longer safe retry tokens there.

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

## Current limitations and deferred hardening

The completed local format-7 lifecycle remains authoritative and is documented
in [`mutable-storage.md`](mutable-storage.md). The server deliberately exposes
that engine rather than introducing a second mutation model, but this
checkpoint has the following boundaries:

- one already-created numeric-ID format-7 collection per process, with one
  fixed `float16` or `float32` schema;
- local POSIX/NVMe backing storage only; CephFS, object storage, and replication
  do not yet share the documented durability contract;
- plaintext transport intended for loopback or a trusted network boundary;
- no server RPC or background worker for `maintain()`; stop the writable daemon
  before taking the writer lease for explicit embedded/CLI maintenance;
- `throwdbd` accepts explicit `--io-*` reader settings but does not yet resolve
  and load the collection's saved tuning profile automatically;
- no dedicated network-server load benchmark yet. The existing local reader
  benchmarks do not measure protobuf serialization, gRPC copies, network
  latency, server admission, or concurrent remote clients;
- the durable operation journal is append-only in this slice; checkpointing
  and retention remain future maintenance work;
- TLS/authentication, metrics, server-wide admission/priority policy,
  multi-collection routing, and reduced-copy large unary RPCs are deferred.

The next server performance checkpoint is therefore a dedicated gRPC benchmark
harness followed by tune-aware daemon startup. Device calibration itself stays
explicit and bounded; daemon startup must only select a compatible saved
profile and must not launch an implicit benchmark.
