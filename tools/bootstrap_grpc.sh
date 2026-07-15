#!/usr/bin/env bash

# Build the exact C++ gRPC/Protobuf toolchain used by ThrowDB into a
# user-owned prefix. No sudo, apt, or mutation outside the work/prefix paths.
set -euo pipefail

readonly GRPC_VERSION="1.81.0"
readonly GRPC_COMMIT="8bdf11e5097abecbe6cffb70866625a0b8044948"
readonly GRPC_REPOSITORY="https://github.com/grpc/grpc.git"

readonly PREFIX="${THROWDB_GRPC_PREFIX:-${HOME}/.local/throwdb/grpc-${GRPC_VERSION}}"
readonly WORK_ROOT="${THROWDB_GRPC_WORK_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/throwdb/grpc-${GRPC_VERSION}}"
readonly SOURCE_DIR="${WORK_ROOT}/source"
readonly BUILD_DIR="${WORK_ROOT}/build"
readonly JOBS="${THROWDB_GRPC_JOBS:-4}"
readonly CMAKE_COMMAND="${CMAKE_COMMAND:-cmake}"

if [[ $# -ne 0 ]]; then
    echo "usage: THROWDB_GRPC_PREFIX=PATH THROWDB_GRPC_JOBS=N $0" >&2
    exit 2
fi
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "THROWDB_GRPC_JOBS must be a positive integer" >&2
    exit 2
fi

for command in git "${CMAKE_COMMAND}" c++ make perl python3; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required build command not found: ${command}" >&2
        exit 1
    fi
done

mkdir -p "${WORK_ROOT}" "${PREFIX}"
if [[ ! -e "${SOURCE_DIR}" ]]; then
    git clone \
        --branch "v${GRPC_VERSION}" \
        --depth 1 \
        --recurse-submodules \
        --shallow-submodules \
        "${GRPC_REPOSITORY}" \
        "${SOURCE_DIR}"
elif [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    echo "refusing to reuse non-git source path: ${SOURCE_DIR}" >&2
    exit 1
fi

actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${GRPC_COMMIT}" ]]; then
    echo "refusing to build an unpinned gRPC checkout" >&2
    echo "expected ${GRPC_COMMIT}, found ${actual_commit}" >&2
    echo "choose an empty THROWDB_GRPC_WORK_DIR and retry" >&2
    exit 1
fi
git -C "${SOURCE_DIR}" submodule update \
    --init --recursive --depth 1 --jobs "${JOBS}"

"${CMAKE_COMMAND}" \
    -S "${SOURCE_DIR}" \
    -B "${BUILD_DIR}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_BUILD_CODEGEN=ON \
    -DgRPC_BUILD_GRPC_CPP_PLUGIN=ON \
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
    -DgRPC_BUILD_GRPCPP_OTEL_PLUGIN=OFF \
    -DgRPC_ABSL_PROVIDER=module \
    -DgRPC_CARES_PROVIDER=module \
    -DgRPC_PROTOBUF_PROVIDER=module \
    -DgRPC_RE2_PROVIDER=module \
    -DgRPC_SSL_PROVIDER=module \
    -DgRPC_ZLIB_PROVIDER=module \
    -Dprotobuf_BUILD_TESTS=OFF

"${CMAKE_COMMAND}" --build "${BUILD_DIR}" \
    --target install --parallel "${JOBS}"

for installed in \
    "${PREFIX}/bin/protoc" \
    "${PREFIX}/bin/grpc_cpp_plugin" \
    "${PREFIX}/lib/cmake/protobuf/protobuf-config.cmake" \
    "${PREFIX}/lib/cmake/grpc/gRPCConfig.cmake"; do
    if [[ ! -e "${installed}" ]]; then
        echo "gRPC bootstrap did not install expected file: ${installed}" >&2
        exit 1
    fi
done

echo
echo "Pinned gRPC ${GRPC_VERSION} toolchain installed in: ${PREFIX}"
echo "Configure ThrowDB with:"
echo "  export THROWDB_GRPC_PREFIX=${PREFIX}"
echo "  cmake --preset server-release"
echo "or build the complete Python package with:"
echo "  python -m pip install ."
echo "or install an editable development checkout with:"
echo "  python -m pip install -e '.[test]'"
