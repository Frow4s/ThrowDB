#!/usr/bin/env python3
"""mTLS-capable Kubernetes exec probe for the ThrowDB Health RPC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import grpc

from throwdb.v1 import throwdb_pb2, throwdb_pb2_grpc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--root-ca", type=Path, required=True)
    parser.add_argument("--client-cert", type=Path, required=True)
    parser.add_argument("--client-key", type=Path, required=True)
    parser.add_argument(
        "--server-name",
        required=True,
        help="DNS name present in the server certificate SAN",
    )
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument(
        "--require-serving",
        action="store_true",
        help="fail when the server responds with NOT_SERVING",
    )
    return parser


def _read(path: Path, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not payload:
        raise RuntimeError(f"{label} {path} is empty")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be greater than zero")

    credentials = grpc.ssl_channel_credentials(
        root_certificates=_read(args.root_ca, "root CA"),
        private_key=_read(args.client_key, "client key"),
        certificate_chain=_read(args.client_cert, "client certificate"),
    )
    options = (
        ("grpc.enable_http_proxy", 0),
        ("grpc.ssl_target_name_override", args.server_name),
        ("grpc.default_authority", args.server_name),
    )
    channel = grpc.secure_channel(args.endpoint, credentials, options=options)
    try:
        response = throwdb_pb2_grpc.ThrowDBStub(channel).Health(
            throwdb_pb2.HealthRequest(),
            timeout=args.timeout_seconds,
            wait_for_ready=False,
        )
    except grpc.RpcError as error:
        print(
            f"ThrowDB health RPC failed: {error.code().name}: "
            f"{error.details()}",
            file=sys.stderr,
        )
        return 1
    finally:
        channel.close()

    valid = {
        throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING,
        throwdb_pb2.HealthResponse.SERVING_STATUS_NOT_SERVING,
    }
    if response.protocol_major != 1 or response.status not in valid:
        print("ThrowDB health response is incompatible", file=sys.stderr)
        return 1
    if (
        args.require_serving
        and response.status
        != throwdb_pb2.HealthResponse.SERVING_STATUS_SERVING
    ):
        print("ThrowDB is NOT_SERVING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
