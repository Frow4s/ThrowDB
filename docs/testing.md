# MVP verification and benchmark plan

The MVP is considered correct only when the same immutable store can be closed,
reopened, and queried without relying on writer process state. Tests use small
dimensions and shard sizes, but exercise the same format and address arithmetic
as a production `float16[1024]` store.

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
| Corruption | Truncated shard, index, or manifest | Open or first affected read fails loudly |
| Format compatibility | New writer emits v1 manifest, index, and every shard at the documented offsets | Existing v1 wire representation stays unchanged |
| Version dispatch | Manifest, shard, index, or staging advertises v2/unknown version before implementation | Reader rejects it before interpreting v1-specific fields |
| Internal I/O | `io_workers` 0, 1, 2, 4, and 8 with a bounded queue | Results remain bit-exact across shards |
| Chunked I/O | `io_ranges_per_task` 1, 4, 8, and 16, including batches smaller than a task | Results remain bit-exact at every partition boundary |
| Shared reader | Simultaneous `get` calls on one `Storage` instance | Calls complete without cross-request corruption |
| Worker error | A queued range hits a truncated shard | Exception reaches the caller; executor remains usable |
| io_uring | Queue depths 1, 2, and 4 with more ranges than SQ entries | Ring wraps and every row remains bit-exact |
| Multi-ring io_uring | One and two workers with total queue depths 2, 3, and 4 | A single request is split across rings without changing bits or row order |
| Multi-ring concurrency | Simultaneous callers share two ring-owner workers | Calls complete without cross-request corruption or deadlock |
| io_uring error | Truncate a shard after open, then issue a healthy request | EOF reaches the caller and the ring remains reusable |

Floating-point comparisons are deliberately inappropriate here. The storage
engine does not calculate with embeddings: it preserves the original 16-bit
payload. Tests therefore compare `float16` arrays bit for bit.

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

For meaningful cold-cache numbers, use an isolated host and an agreed cache
eviction procedure. Running immediately after database construction measures
the page cache rather than NVMe random-read behavior.

The Python benchmark consumes the original `ids.npy` directly:

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

Both legacy runners are sequential `pread` baselines: they open `Storage` with
zero internal I/O workers. They sample only known IDs before timing and reuse an
allocated output buffer. They measure client-visible latency and useful
throughput; neither claims to measure physical bytes read. Use
`run_load_test.py` for executor, chunk-size, and concurrent-reader comparisons.

## Internal I/O workers versus reader concurrency

These are independent axes. `--io-workers` controls the C++ executor owned by a
single `Storage`; the concurrent-reader scenarios control how many Python client
threads call `get_into` on that instance. Multi-range calls submit work to the
shared executor. Reports record both axes as `io_workers` and
`external_reader_workers` (the legacy `workers` field is the external count).
`io_ranges_per_task` records the maximum number of physical ranges a worker
processes after one queue dispatch.

There is one deliberate exception to the shared bound. If planning produces
exactly one physical range, the calling thread performs a synchronous direct
read without submitting to the executor. One-range calls from concurrent
callers therefore bypass the executor and queue and are not included in the
global concurrency bound that applies to multi-range work. Interpret batch-1
concurrency results with this fast path in mind.

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

## Multi-ring io_uring experiment

The raw-ABI baseline uses one mutex-protected ring per `Storage`. The next
experiment adds dedicated ring-owner workers. One large request is cost-balanced
across all rings, concurrent batches enter every per-worker FIFO in the same
order, and the caller waits for every lane before temporary buffers can expire.
Queue depth is a total: two workers with depth 32 create two depth-16 rings.
Admission is bounded to `2 * io_uring_workers` active/queued batches and is
strictly FIFO between callers. It is recorded as
`io_uring_admission_capacity` and `io_uring_admission_policy` in the benchmark
report. The storage
section also records `/proc/self/task` counts immediately before and after open,
which isolates the dedicated user-space ring workers.

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
context switches, diskstats, and `/proc/self/task` count. Whole-batch FIFO can
still cause head-of-line blocking between very different batch sizes; a fair
quantum scheduler is intentionally deferred until measurements justify it.
