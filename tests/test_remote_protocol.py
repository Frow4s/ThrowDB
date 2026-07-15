from __future__ import annotations

import socket
import struct
import threading

import pytest

from throwdb._remote_protocol import (
    FrameFlag,
    Opcode,
    RemoteNotFoundError,
    RemoteProtocolError,
    Status,
    decode_response_payload,
    encode_frame,
    encode_response_payload,
    receive_frame,
)


def _send_in_fragments(stream: socket.socket, payload: bytes) -> None:
    try:
        for offset in range(0, len(payload), 3):
            stream.sendall(payload[offset : offset + 3])
    finally:
        stream.close()


def test_frame_receiver_handles_fragmented_stream_reads() -> None:
    receiver, sender = socket.socketpair()
    encoded = encode_frame(Opcode.GET, FrameFlag(0), 97, b"payload")
    thread = threading.Thread(
        target=_send_in_fragments, args=(sender, encoded), daemon=True
    )
    thread.start()
    try:
        frame = receive_frame(receiver)
    finally:
        receiver.close()
        thread.join(timeout=1)

    assert frame.opcode == Opcode.GET
    assert frame.flags == 0
    assert frame.request_id == 97
    assert frame.payload == b"payload"


def test_response_decoder_maps_remote_status_to_python_api_error() -> None:
    receiver, sender = socket.socketpair()
    payload = encode_response_payload(Status.NOT_FOUND, message="missing id 42")
    sender.sendall(
        encode_frame(
            Opcode.GET,
            FrameFlag.RESPONSE | FrameFlag.ERROR,
            11,
            payload,
        )
    )
    sender.close()
    try:
        frame = receive_frame(receiver)
    finally:
        receiver.close()

    with pytest.raises(RemoteNotFoundError, match="missing id 42") as captured:
        decode_response_payload(frame)
    # RemoteNotFoundError preserves embedded Collection's KeyError contract.
    assert isinstance(captured.value, KeyError)
    assert captured.value.request_id == 11


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        struct.pack("<II", Status.OK, 100),
        struct.pack("<II", 999, 0),
    ],
)
def test_malformed_response_prefix_is_rejected(payload: bytes) -> None:
    receiver, sender = socket.socketpair()
    sender.sendall(
        encode_frame(Opcode.DESCRIBE, FrameFlag.RESPONSE, 1, payload)
    )
    sender.close()
    try:
        frame = receive_frame(receiver)
    finally:
        receiver.close()

    with pytest.raises(RemoteProtocolError):
        decode_response_payload(frame)


def test_receiver_rejects_payload_larger_than_client_budget_before_reading_it() -> None:
    receiver, sender = socket.socketpair()
    sender.sendall(encode_frame(Opcode.GET, FrameFlag.RESPONSE, 1, b"0123456789"))
    sender.close()
    try:
        with pytest.raises(RemoteProtocolError, match="client limit"):
            receive_frame(receiver, max_payload_bytes=4)
    finally:
        receiver.close()
