import socket

from src.distributed.protocol import recv_json, send_json


def test_protocol_roundtrip_json():
    sock_a, sock_b = socket.socketpair()
    try:
        payload = {"type": "TASK", "task_id": "t1", "meta": {"layer": 1}}
        send_json(sock_a, payload)
        got = recv_json(sock_b)
        assert got == payload
    finally:
        sock_a.close()
        sock_b.close()
