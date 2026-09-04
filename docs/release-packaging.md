# Packaging, wheels, and release artifacts

## Current artifact status

At the reviewed commit there is no PyPI publication, GitHub Release, tag, OCI
image, or checked-in binary artifact. Do not run `pip install throwdb` and
assume it installs this repository revision.

The wheel built and black-box tested during development is:

```text
/private/tmp/throwdb-wheel-final-20260717-2/
  throwdb-0.1.0-cp310-cp310-macosx_14_0_arm64.whl
SHA256 e7268329269c6beeacebca4972e6226f8cbb9e0d56969e7643e8b210239d9000
```

It is a temporary local development artifact for CPython 3.10 on macOS 14
ARM64. It is not compatible with Linux or Kubernetes.

The configured Linux release matrix is:

- CPython 3.10, 3.11, 3.12, 3.13, and 3.14;
- Linux x86_64;
- `manylinux_2_28` / glibc 2.28 or newer;
- no musllinux/Alpine, i686, ARM64, Windows, or official macOS matrix.

Each complete wheel contains:

- the `throwdb` Python package;
- the pybind11 `_throwdb` C++ extension;
- generated Python protobuf/gRPC stubs and the canonical `.proto` file;
- the tune-aware `throwdb` and `throwdbd` console scripts;
- `throwdb/libexec/_throwdbd_native`, the private C++ gRPC daemon.

The public `throwdbd` process selects tuning once and then calls `execve()`;
Python is absent from the live RPC data path.

## Download from GitHub Actions

`.github/workflows/wheels.yml` runs manually or on a `v*` tag. A successful run
uploads two ephemeral, authentication-controlled artifacts:

- `throwdb-manylinux-x86_64`;
- `throwdb-sdist`.

Browser path:

1. Open the `Frow4s/ThrowDB` repository.
2. Open **Actions**.
3. Select **Linux wheels**.
4. Open a successful run for the exact commit/tag.
5. Download `throwdb-manylinux-x86_64` from **Artifacts**.
6. Unzip it into a controlled `wheelhouse/` directory.

With an authenticated GitHub CLI:

```bash
gh run list \
  -R Frow4s/ThrowDB \
  -w wheels.yml

gh run download RUN_ID \
  -R Frow4s/ThrowDB \
  -n throwdb-manylinux-x86_64 \
  -D wheelhouse
```

Actions artifacts expire according to repository policy and are not a stable
release channel. Record the run URL, commit SHA, filename, size, and SHA-256 in
deployment metadata.

## Build portable Linux wheels

Run cibuildwheel on a Linux machine with Docker:

```bash
python3 -m venv .release-venv
. .release-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'cibuildwheel==4.1.0'

# Complete configured CPython matrix.
python -m cibuildwheel \
  --platform linux \
  --output-dir wheelhouse

sha256sum wheelhouse/*.whl > wheelhouse/SHA256SUMS
```

For only the CPython 3.11 deployment artifact:

```bash
CIBW_BUILD='cp311-manylinux_x86_64' \
  python -m cibuildwheel \
    --platform linux \
    --output-dir wheelhouse
```

The Linux cibuildwheel configuration installs build tools inside the
manylinux image and runs `tools/bootstrap_grpc.sh`. That script pins gRPC
1.81.0 to an exact commit and builds gRPC/Protobuf statically in a userspace
prefix. CMake uses `find_package(... REQUIRED)` so a requested server build
cannot silently degrade into a client/core-only wheel.

## Build a same-host wheel without Docker

This is useful for development but is less portable than cibuildwheel:

```bash
python3 -m venv .build-venv
. .build-venv/bin/activate
python -m pip install --upgrade pip build

export THROWDB_GRPC_PREFIX="$HOME/.local/throwdb/grpc-1.81.0"
export THROWDB_GRPC_JOBS=4
./tools/bootstrap_grpc.sh

python -m pip wheel . \
  --no-deps \
  --wheel-dir wheelhouse
```

The resulting platform tag and dynamic-library requirements reflect the build
host. Do not move that wheel into a different Linux baseline without inspecting
it and running an isolated smoke test.

## Verify a wheel

Select a wheel matching the interpreter exactly:

```bash
python3.11 -m venv /tmp/throwdb-wheel-check
/tmp/throwdb-wheel-check/bin/python -m pip install --upgrade pip
/tmp/throwdb-wheel-check/bin/python -m pip install \
  wheelhouse/throwdb-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
/tmp/throwdb-wheel-check/bin/python -m pip check
```

Inspect the installed layout:

```bash
/tmp/throwdb-wheel-check/bin/python - <<'PY'
import importlib.metadata
from pathlib import Path

import throwdb

distribution = importlib.metadata.distribution("throwdb")
daemon = Path(
    distribution.locate_file("throwdb/libexec/_throwdbd_native")
).resolve()
print("version:", throwdb.__version__)
print("native daemon:", daemon)
print("exists:", daemon.is_file())
PY

/tmp/throwdb-wheel-check/bin/throwdb --help
/tmp/throwdb-wheel-check/bin/throwdbd --help
```

From a source checkout, run the black-box product smoke test with that isolated
interpreter:

```bash
/tmp/throwdb-wheel-check/bin/python tools/wheel_smoke.py
```

It validates format-7 create/read/CRUD, prefix reads, rotation,
materialization, folding, cleaning, orphan collection, CLI discovery, public
launcher discovery, native daemon packaging, gRPC Health, and remote Get.

For Linux release candidates also inspect the extension and daemon with
`auditwheel show`, `readelf -d`, or `ldd` inside the minimum supported runtime.
Test on a clean `debian:bookworm-slim`/`python:3.11-slim-bookworm` container,
not only on the build host.

## Offline installation

The ThrowDB wheel does not embed Python dependency wheels. Mirror a complete
wheelhouse for the target CPython ABI:

```bash
python -m pip download \
  --only-binary=:all: \
  --dest offline-wheelhouse \
  ./wheelhouse/throwdb-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl

python -m pip install \
  --no-index \
  --find-links offline-wheelhouse \
  throwdb==0.1.0
```

The resolved runtime dependencies are NumPy, grpcio, and protobuf within the
ranges declared by `pyproject.toml`. Pin and retain exact dependency wheels for
a reproducible production image.

## Build the runtime image

Place the selected Linux wheel under the filename expected by the reviewed
Dockerfile:

```bash
mkdir -p wheelhouse
cp \
  wheelhouse-download/throwdb-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl \
  wheelhouse/throwdb.whl

docker build \
  --file deploy/docker/Dockerfile \
  --tag registry.example/throwdb:0.1.0 \
  .

docker run --rm registry.example/throwdb:0.1.0 --help
docker push registry.example/throwdb:0.1.0
docker inspect \
  --format='{{index .RepoDigests 0}}' \
  registry.example/throwdb:0.1.0
```

Use the immutable digest in Kubernetes, for example:

```text
registry.example/throwdb:0.1.0@sha256:...
```

The runtime image intentionally does not include the AWS CLI. Create a separate
`throwdb-ops` image for backup/restore and pin a current AWS CLI version that
supports conditional `s3api put-object --if-match/--if-none-match`. Validate it
with `throwdb backup s3 --dry-run` against a non-production prefix.

## Recommended release procedure

1. Select the project license and assemble third-party notices.
2. Bump the canonical version in `pyproject.toml`; CMake and the extension derive it.
3. Require all Python, native, gRPC, sanitizer, and protobuf checks.
4. Build at least one manylinux wheel as a PR packaging canary.
5. Tag the exact reviewed commit.
6. Build all configured wheels and the sdist in an isolated workflow.
7. Run `tools/wheel_smoke.py` in each matching clean interpreter image.
8. Generate SHA-256 manifests, SBOM, signatures, and build provenance.
9. Publish to a durable GitHub Release and/or private package index.
10. Build runtime and ops OCI images, then publish them by immutable digest.
11. Retain source commit, toolchain versions, artifacts, checksums, test
    reports, and compatibility notes for rollback.

The current tag workflow only performs steps 6–7 and uploads ephemeral Actions
artifacts. It does not create a GitHub Release or publish to PyPI.

## Known packaging gaps

- The package version is canonical in `pyproject.toml` and shared by CMake
  and the extension, but is not yet validated against a release tag.
- No LICENSE or NOTICE exists, which blocks responsible external distribution.
- Build backend dependencies are lower-bounded rather than fully locked.
- A published sdist would require users to bootstrap native gRPC first.
- Official architecture coverage is Linux x86_64 only.
- There is no packaged doctor/self-test command.
- There is no OCI publishing workflow, signing, SBOM, or provenance yet.

These gaps do not invalidate local engine testing, but they must be resolved
before calling a binary a durable public release.
