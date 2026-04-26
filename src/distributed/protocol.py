from __future__ import annotations

import json
import socket
from typing import Any


BUFFER_SIZE = 65536


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8") + b"\n"
    sock.sendall(raw)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    chunks = []
    while True:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break

    raw = b"".join(chunks)
    line = raw.split(b"\n", 1)[0]
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))
