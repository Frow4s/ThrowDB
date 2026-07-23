#!/usr/bin/env bash

# Rebuild and verify a ThrowDB source checkout after it has been updated.
#
# This script is intentionally safe for a remote development host:
# - it never mutates a collection;
# - it never changes Git state unless --pull is requested;
# - it never restarts a live daemon unless --restart-user-service is supplied;
# - the pinned C++ gRPC toolchain is bootstrapped only when it is missing.
set -Eeuo pipefail

readonly SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd -P
)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

VENV="${THROWDB_VENV:-${REPOSITORY_ROOT}/.venv}"
PYTHON_COMMAND="${THROWDB_PYTHON:-python3}"
GRPC_PREFIX="${THROWDB_GRPC_PREFIX:-${HOME}/.local/throwdb/grpc-1.81.0}"
JOBS="${THROWDB_UPDATE_JOBS:-4}"
CHECK_MODE="${THROWDB_UPDATE_CHECK:-smoke}"
BOOTSTRAP_MODE="${THROWDB_UPDATE_BOOTSTRAP_GRPC:-auto}"
RESTART_USER_SERVICE="${THROWDB_UPDATE_SYSTEMD_USER_UNIT:-}"
PULL=0
VENV_CREATED=0
RESTART_ATTEMPTED=0

usage() {
    cat <<'EOF'
Usage: ./tools/update_remote.sh [OPTIONS]

Apply the current ThrowDB checkout to a remote development environment.

Options:
  --pull                       Run "git pull --ff-only" before building.
  --venv PATH                  Virtual environment (default: .venv).
  --python COMMAND             Python used to create .venv (default: python3).
  --grpc-prefix PATH           Pinned gRPC installation prefix.
  --jobs N                     Parallel build jobs (default: 4).
  --check MODE                 none, smoke, quick, or full (default: smoke).
  --bootstrap-grpc MODE        auto, always, or never (default: auto).
  --restart-user-service UNIT  Restart this systemd user unit only after all
                               selected checks pass.
  -h, --help                   Show this help.

Check modes:
  none   Mandatory import, packaging, CLI, and native-daemon checks only.
  smoke  Also run tools/wheel_smoke.py against a temporary collection.
  quick  Smoke plus focused local/remote streaming regression tests.
  full   Smoke plus the complete Python test suite.

Environment equivalents:
  THROWDB_VENV
  THROWDB_PYTHON
  THROWDB_GRPC_PREFIX
  THROWDB_UPDATE_JOBS
  THROWDB_UPDATE_CHECK
  THROWDB_UPDATE_BOOTSTRAP_GRPC
  THROWDB_UPDATE_SYSTEMD_USER_UNIT

The script does not touch a production collection. An already running
throwdbd keeps serving its old executable until it is explicitly restarted.
EOF
}

die() {
    echo "update_remote.sh: $*" >&2
    exit 1
}

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

on_error() {
    local exit_code=$?
    echo >&2
    echo "update_remote.sh: failed at line ${BASH_LINENO[0]} (exit ${exit_code})" >&2
    if [[ "${RESTART_ATTEMPTED}" -eq 1 ]]; then
        echo "The requested service restart was attempted; inspect its current systemd status." >&2
    else
        echo "No running throwdbd process was restarted." >&2
    fi
    echo "The editable environment may be partially updated; fix the error and rerun before starting new processes." >&2
    exit "${exit_code}"
}
trap on_error ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull)
            PULL=1
            shift
            ;;
        --venv)
            [[ $# -ge 2 ]] || die "--venv requires PATH"
            VENV="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires COMMAND"
            PYTHON_COMMAND="$2"
            shift 2
            ;;
        --grpc-prefix)
            [[ $# -ge 2 ]] || die "--grpc-prefix requires PATH"
            GRPC_PREFIX="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || die "--jobs requires N"
            JOBS="$2"
            shift 2
            ;;
        --check)
            [[ $# -ge 2 ]] || die "--check requires MODE"
            CHECK_MODE="$2"
            shift 2
            ;;
        --bootstrap-grpc)
            [[ $# -ge 2 ]] || die "--bootstrap-grpc requires MODE"
            BOOTSTRAP_MODE="$2"
            shift 2
            ;;
        --restart-user-service)
            [[ $# -ge 2 ]] || die "--restart-user-service requires UNIT"
            RESTART_USER_SERVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (use --help)"
            ;;
    esac
done

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] ||
    die "--jobs/THROWDB_UPDATE_JOBS must be a positive integer"
case "${CHECK_MODE}" in
    none|smoke|quick|full) ;;
    *) die "--check must be one of: none, smoke, quick, full" ;;
esac
case "${BOOTSTRAP_MODE}" in
    auto|always|never) ;;
    *) die "--bootstrap-grpc must be one of: auto, always, never" ;;
esac

for command in git cmake c++ make perl; do
    command -v "${command}" >/dev/null 2>&1 ||
        die "required build command not found: ${command}"
done
command -v "${PYTHON_COMMAND}" >/dev/null 2>&1 ||
    die "Python command not found: ${PYTHON_COMMAND}"

cd -- "${REPOSITORY_ROOT}"

mkdir -p -- "${REPOSITORY_ROOT}/build"
if command -v flock >/dev/null 2>&1; then
    exec 9>"${REPOSITORY_ROOT}/build/remote-update.lock"
    flock -n 9 ||
        die "another tools/update_remote.sh process is already running"
fi

if [[ "${PULL}" -eq 1 ]]; then
    git diff --quiet && git diff --cached --quiet ||
        die "--pull requires a clean tracked worktree; commit or stash local changes first"
    log "fast-forwarding the current Git branch"
    run git pull --ff-only
fi

readonly SOURCE_REVISION="$(
    git rev-parse --verify HEAD 2>/dev/null ||
        die "repository HEAD cannot be resolved"
)"
log "updating ThrowDB ${SOURCE_REVISION}"

if [[ ! -x "${VENV}/bin/python" ]]; then
    if [[ -e "${VENV}" ]]; then
        die "virtual-environment path exists but has no bin/python: ${VENV}"
    fi
    log "creating virtual environment: ${VENV}"
    run "${PYTHON_COMMAND}" -m venv "${VENV}"
    VENV_CREATED=1
fi

readonly VENV_PYTHON="${VENV}/bin/python"
run "${VENV_PYTHON}" -c \
    'import sys; assert sys.version_info >= (3, 10), "ThrowDB requires Python 3.10+"'

grpc_toolchain_ready() {
    [[ -x "${GRPC_PREFIX}/bin/protoc" ]] &&
        [[ -x "${GRPC_PREFIX}/bin/grpc_cpp_plugin" ]] &&
        [[ -f "${GRPC_PREFIX}/lib/cmake/protobuf/protobuf-config.cmake" ]] &&
        [[ -f "${GRPC_PREFIX}/lib/cmake/grpc/gRPCConfig.cmake" ]]
}

case "${BOOTSTRAP_MODE}" in
    always)
        log "rebuilding the pinned gRPC/Protobuf toolchain"
        run env \
            THROWDB_GRPC_PREFIX="${GRPC_PREFIX}" \
            THROWDB_GRPC_JOBS="${JOBS}" \
            "${SCRIPT_DIR}/bootstrap_grpc.sh"
        ;;
    auto)
        if grpc_toolchain_ready; then
            log "reusing pinned gRPC/Protobuf toolchain: ${GRPC_PREFIX}"
        else
            log "bootstrapping pinned gRPC/Protobuf toolchain once: ${GRPC_PREFIX}"
            run env \
                THROWDB_GRPC_PREFIX="${GRPC_PREFIX}" \
                THROWDB_GRPC_JOBS="${JOBS}" \
                "${SCRIPT_DIR}/bootstrap_grpc.sh"
        fi
        ;;
    never)
        grpc_toolchain_ready ||
            die "gRPC toolchain is incomplete at ${GRPC_PREFIX}"
        log "using preinstalled pinned gRPC/Protobuf toolchain: ${GRPC_PREFIX}"
        ;;
esac

if [[ "${VENV_CREATED}" -eq 1 ]]; then
    log "upgrading pip in the new virtual environment"
    run "${VENV_PYTHON}" -m pip install --upgrade "pip>=23"
fi
log "ensuring Python build requirements"
run "${VENV_PYTHON}" -m pip install \
    "scikit-build-core>=0.10" \
    "pybind11>=2.12"

readonly PYTHON_TAG="$(
    "${VENV_PYTHON}" -c \
        'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'
)"
readonly EDITABLE_BUILD_DIR="${REPOSITORY_ROOT}/build/remote-editable-${PYTHON_TAG}"

log "building and installing the complete editable package"
run env \
    -u CMAKE_ARGS \
    -u CMAKE_MAKE_PROGRAM \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    CMAKE_GENERATOR="Unix Makefiles" \
    THROWDB_GRPC_PREFIX="${GRPC_PREFIX}" \
    "${VENV_PYTHON}" -m pip install \
    --editable "${REPOSITORY_ROOT}[test]" \
    --no-build-isolation \
    "--config-settings=build-dir=${EDITABLE_BUILD_DIR}" \
    "--config-settings=cmake.define.THROWDB_GRPC_PREFIX=${GRPC_PREFIX}"

log "validating the installed package"
run "${VENV_PYTHON}" -m pip check
run "${VENV_PYTHON}" -c \
    'import throwdb; assert throwdb.__version__; print("throwdb", throwdb.__version__)'
run "${VENV}/bin/throwdb" --help
run "${VENV}/bin/throwdbd" --help

readonly NATIVE_SERVER="$(
    "${VENV_PYTHON}" - <<'PY'
from importlib import metadata
from pathlib import Path

distribution = metadata.distribution("throwdb")
native = Path(
    distribution.locate_file("throwdb/libexec/_throwdbd_native")
).resolve()
if not native.is_file():
    raise SystemExit(f"installed native daemon is missing: {native}")
print(native)
PY
)"
[[ -x "${NATIVE_SERVER}" ]] ||
    die "installed native daemon is not executable: ${NATIVE_SERVER}"
run "${NATIVE_SERVER}" --help

case "${CHECK_MODE}" in
    none)
        ;;
    smoke)
        log "running black-box package and native-server smoke test"
        run env \
            THROWDB_WHEEL_SMOKE_REQUIRE_SERVER=1 \
            "${VENV_PYTHON}" "${REPOSITORY_ROOT}/tools/wheel_smoke.py"
        ;;
    quick)
        log "running black-box smoke and focused streaming regressions"
        run env \
            THROWDB_WHEEL_SMOKE_REQUIRE_SERVER=1 \
            "${VENV_PYTHON}" "${REPOSITORY_ROOT}/tools/wheel_smoke.py"
        run env \
            THROWDB_SERVER_BINARY="${NATIVE_SERVER}" \
            "${VENV_PYTHON}" -m pytest -q \
            "${REPOSITORY_ROOT}/tests/test_record_scan_api.py" \
            "${REPOSITORY_ROOT}/tests/test_grpc_remote_collection.py"
        ;;
    full)
        log "running black-box smoke and complete Python test suite"
        run env \
            THROWDB_WHEEL_SMOKE_REQUIRE_SERVER=1 \
            "${VENV_PYTHON}" "${REPOSITORY_ROOT}/tools/wheel_smoke.py"
        run env \
            THROWDB_SERVER_BINARY="${NATIVE_SERVER}" \
            "${VENV_PYTHON}" -m pytest -q "${REPOSITORY_ROOT}/tests"
        ;;
esac

readonly FINISHED_REVISION="$(git rev-parse --verify HEAD)"
[[ "${FINISHED_REVISION}" == "${SOURCE_REVISION}" ]] ||
    die "Git HEAD changed during the build; run the updater again"

readonly CURRENT_DIR="${REPOSITORY_ROOT}/build/remote-current"
mkdir -p -- "${CURRENT_DIR}"
ln -sfn -- "${NATIVE_SERVER}" "${CURRENT_DIR}/throwdbd-native"
ln -sfn -- "${VENV}/bin/throwdb" "${CURRENT_DIR}/throwdb"
ln -sfn -- "${VENV}/bin/throwdbd" "${CURRENT_DIR}/throwdbd"

environment_tmp="${CURRENT_DIR}/environment.sh.tmp.$$"
{
    printf '# Generated by tools/update_remote.sh for ThrowDB %s\n' \
        "${SOURCE_REVISION}"
    printf 'export THROWDB_REPOSITORY=%q\n' "${REPOSITORY_ROOT}"
    printf 'export THROWDB_VENV=%q\n' "${VENV}"
    printf 'export THROWDB_GRPC_PREFIX=%q\n' "${GRPC_PREFIX}"
    printf 'export THROWDB_NATIVE_SERVER_BINARY=%q\n' \
        "${CURRENT_DIR}/throwdbd-native"
    printf 'export THROWDB_SERVER_BINARY=%q\n' \
        "${CURRENT_DIR}/throwdbd-native"
    printf 'export PATH=%q:"${PATH}"\n' "${VENV}/bin"
} >"${environment_tmp}"
mv -f -- "${environment_tmp}" "${CURRENT_DIR}/environment.sh"

if [[ -n "${RESTART_USER_SERVICE}" ]]; then
    command -v systemctl >/dev/null 2>&1 ||
        die "systemctl is required by --restart-user-service"
    log "restarting systemd user service after successful verification"
    RESTART_ATTEMPTED=1
    run systemctl --user restart "${RESTART_USER_SERVICE}"
    run systemctl --user is-active "${RESTART_USER_SERVICE}"
fi

log "ThrowDB update completed"
printf 'Revision:       %s\n' "${SOURCE_REVISION}"
printf 'Python:         %s\n' "${VENV_PYTHON}"
printf 'CLI:            %s\n' "${CURRENT_DIR}/throwdb"
printf 'Server launcher:%s\n' " ${CURRENT_DIR}/throwdbd"
printf 'Native daemon:  %s\n' "${CURRENT_DIR}/throwdbd-native"
printf 'Environment:    source %q\n' "${CURRENT_DIR}/environment.sh"
if [[ -z "${RESTART_USER_SERVICE}" ]]; then
    printf '\nNo running daemon was restarted. Restart it explicitly before expecting\n'
    printf 'new server code to serve traffic, or pass --restart-user-service UNIT.\n'
fi
printf 'Restart long-lived Python processes (Jupyter kernels and Airflow workers)\n'
printf 'before expecting them to load the new Python wrapper and C++ extension.\n'
