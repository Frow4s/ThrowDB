# Linux validation of the format-7 API

This runbook uses the existing native suites, backend benchmark and gRPC soak.
It creates a disposable data set and keeps reports separate from the source
checkout. It does not restart a service or open a production collection.
Run commands from an isolated checkout of the revision being evaluated.

## Build and record the environment

Use the pinned gRPC toolchain described in [the server guide](grpc-server.md)
and an installed package matching the checkout. For example, after activating
its Python environment and setting `THROWDB_GRPC_PREFIX`:

```bash
cmake --preset server-release -DCMAKE_CXX_FLAGS=-Werror
cmake --build --preset server-release --parallel 4
python -m pip install '.[test]'

export THROWDB_VALIDATION_DIR="$(mktemp -d /tmp/throwdb-validation.XXXXXX)"
export THROWDB_SERVER_BINARY="$PWD/build/server-release/throwdbd"
{
  git rev-parse HEAD
  uname -a
  lscpu
  free -h
  df -h "$THROWDB_VALIDATION_DIR"
  findmnt -T "$THROWDB_VALIDATION_DIR"
  python --version
  cmake --version
} > "$THROWDB_VALIDATION_DIR/environment.txt"

ctest --preset server-release --no-tests=error
python -m pytest -q
```

Use `mktemp -d /path/on-target-device/throwdb-validation.XXXXXX` instead when
`/tmp` is on tmpfs or a different device. Keep the returned directory for the
whole run. The device and filesystem in `environment.txt` determine what the
measurements actually describe.

Configure a separate Debug tree for ASan/UBSan as in
[CI](../.github/workflows/ci.yml). Native tests include the Linux EINTR close
regression, WAL recovery, lifecycle publication and reader barriers. Successful
fake-lane scheduler tests alone do not prove that real io_uring was available.

## Create a bounded data set

This example creates 1,048,576 float16 rows of dimension 256 (512 MiB payload),
plus a row-aligned source oracle and ID array. The source and collection together
need more than 1 GiB; it requires 4 GiB free before starting. Generation uses
bounded batches and disk-backed NumPy arrays. Increase the row count explicitly
for a representative working set after checking available RAM and disk.

```bash
python - <<'PY'
import os
from pathlib import Path
import shutil
import numpy as np
import throwdb

root = Path(os.environ['THROWDB_VALIDATION_DIR'])
if shutil.disk_usage(root).free < 4 << 30:
    raise RuntimeError('validation requires at least 4 GiB free')
rows, dimension, batch_rows = 1 << 20, 256, 8192
for name in ('ids.npy', 'embeddings.npy', 'store'):
    if (root / name).exists():
        raise RuntimeError(f'refusing to overwrite {root / name}')
ids = np.lib.format.open_memmap(root / 'ids.npy', mode='w+',
                               dtype=np.uint64, shape=(rows,))
vectors = np.lib.format.open_memmap(root / 'embeddings.npy', mode='w+',
                                   dtype=np.float16, shape=(rows, dimension))
rng = np.random.default_rng(20260905)
with throwdb.Collection.create(
    root / 'store',
    schema=throwdb.CollectionSchema(dimension=dimension, dtype='float16'),
    segment_size_bytes=64 << 20,
) as builder:
    for start in range(0, rows, batch_rows):
        end = min(rows, start + batch_rows)
        ids[start:end] = np.arange(start + 1, end + 1, dtype=np.uint64)
        vectors[start:end] = rng.standard_normal(
            (end - start, dimension), dtype=np.float32).astype(np.float16)
        builder.add(ids[start:end], vectors[start:end])
ids.flush()
vectors.flush()
print(root)
PY
```

## Compare actual backends

First require a successful real-ring probe for the exact topology being tested.
A denied `io_uring_setup`, unsupported kernel or failed read is a blocked
io_uring check, not a passed check or permission to substitute another backend.

```bash
(
set -e
python - <<'PY'
import json
import os
from pathlib import Path
from throwdb._throwdb import NativeCollection

capabilities = NativeCollection.reader_capabilities(
    io_uring_queue_depth=32, io_uring_workers=2)
Path(os.environ['THROWDB_VALIDATION_DIR'], 'io-uring-capabilities.json').write_text(
    json.dumps(capabilities, indent=2) + '\n')
if not capabilities['io_uring_available']:
    raise RuntimeError(capabilities['io_uring_reason'])
PY

for backend in pread thread_pool io_uring; do
  case "$backend" in
    pread) options=(--io-backend thread_pool --io-workers 0) ;;
    thread_pool) options=(--io-backend thread_pool --io-workers 4) ;;
    io_uring) options=(--io-backend io_uring --io-workers 0
                      --io-uring-workers 2 --io-uring-queue-depth 32) ;;
  esac
  timeout --signal=TERM --kill-after=30s 10m \
    python benchmarks/run_load_test.py \
      --store "$THROWDB_VALIDATION_DIR/store" \
      --ids "$THROWDB_VALIDATION_DIR/ids.npy" \
      --source-embeddings "$THROWDB_VALIDATION_DIR/embeddings.npy" \
      --source-kind npy --profile quick --cache-mode both --tuning explicit \
      --results "$THROWDB_VALIDATION_DIR/$backend" "${options[@]}"
done
)
```

Use Bash for the array syntax. `quick` fixes query counts and tests concurrent
callers; the outer timeout bounds wall time. A timeout or nonzero exit makes
that backend run incomplete. Review the effective backend and workers/depth in
each report's `storage` section,
bit-exact oracle result, p95/p99, physical bytes, CPU, and memory. The cold path
uses targeted file-cache advice; it is not a global cache flush or a guarantee
that every read reaches NVMe. A 512 MiB data set is a bounded validation sample,
not evidence of behavior on a collection larger than RAM.

## Mixed reads, mutations and restart recovery

Run this against the disposable collection only. The harness launches its own
writable daemon, exercises streaming/unary traffic and durable mutation retries,
and performs two restart cycles if the configured bounds permit enough work.

```bash
timeout --signal=TERM --kill-after=30s 5m \
  python benchmarks/run_grpc_server_soak.py \
    --store "$THROWDB_VALIDATION_DIR/store" \
    --ids "$THROWDB_VALIDATION_DIR/ids.npy" \
    --source-embeddings "$THROWDB_VALIDATION_DIR/embeddings.npy" \
    --results "$THROWDB_VALIDATION_DIR/mixed" \
    --native-binary "$THROWDB_SERVER_BINARY" --tuning conservative \
    --max-seconds 180 --max-transfer-gib 8 --max-client-memory-mib 512 \
    --max-get-calls 0 --max-mutations 400 \
    --get-workers 4 --max-inflight-gets 8 --get-batch-sizes 1,128,1024 \
    --prefix-dimensions full,128 \
    --stream-get-rows 65536 --scan-records-rows 65536 \
    --stream-batch-rows 2048 --stream-max-batch-mib 4 \
    --stream-source-oracle --stream-attempt-seconds 30 \
    --population-rows 100000 --mutation-every-gets 20 \
    --restart-cycles 2 --restart-every-gets 250 --scan-rows 10000
```

Exit status alone is insufficient: inspect `report.json` acceptance and
streaming gates, completed restart cycles, recovery errors and retained
WAL/overlay resource telemetry. A bounds-limited run may not complete the
requested restart cycles or trigger maintenance. Such a run does not validate
those paths. For a longer run, explicitly increase time/transfer limits and
require maintenance progress when the mutation volume reaches its thresholds.

Filesystem exhaustion needs a separate disposable filesystem with a strict
quota or size bound; do not fill the host filesystem to simulate ENOSPC.
Admission-reserve tests and a clean shutdown do not replace that fault test.
The short mixed run also does not replace a multi-hour soak or large-data test.
