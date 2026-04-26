from __future__ import annotations

import argparse
import json
import socketserver
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.distributed.protocol import recv_json, send_json
from src.utils.logger import get_logger


LOGGER = get_logger("worker")


class WorkerState:
    def __init__(self, worker_id: int, state_dir: str) -> None:
        self.worker_id = worker_id
        self.state_path = Path(state_dir) / f"worker_{worker_id:03d}.json"
        self.lock = threading.Lock()
        self._state = {
            "worker_id": worker_id,
            "tasks_processed": 0,
            "last_task": None,
            "history": [],
        }
        self._flush()

    def record(self, event: dict[str, Any]) -> None:
        with self.lock:
            self._state["tasks_processed"] += 1
            self._state["last_task"] = event
            self._state["history"].append(event)
            self._state["history"] = self._state["history"][-100:]
            self._flush()

    def _flush(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")


class WorkerTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls, state: WorkerState) -> None:
        self.state = state
        super().__init__(server_address, handler_cls)


class WorkerRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        payload = recv_json(self.request)
        msg_type = payload.get("type")

        if msg_type == "HEARTBEAT":
            send_json(self.request, {"type": "HEARTBEAT_ACK", "status": "ok"})
            return

        if msg_type == "TASK":
            result = self._handle_task(payload)
            send_json(self.request, result)
            return

        if msg_type == "SHUTDOWN":
            send_json(self.request, {"type": "SHUTDOWN_ACK", "status": "ok"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        send_json(self.request, {"type": "ERROR", "status": "error", "error_message": "Unknown message type"})

    def _handle_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            a_block = np.asarray(payload["a_block"], dtype=np.float64)
            b_block = np.asarray(payload["b_block"], dtype=np.float64)
            c_block = a_block @ b_block

            event = {
                "task_id": payload.get("task_id"),
                "layer_id": payload.get("layer_id"),
                "phase": payload.get("phase"),
                "op_type": payload.get("op_type"),
                "block_row": payload.get("block_row"),
                "block_col": payload.get("block_col"),
                "status": "ok",
            }
            self.server.state.record(event)

            return {
                "type": "RESULT",
                "status": "ok",
                "task_id": payload["task_id"],
                "layer_id": payload["layer_id"],
                "phase": payload["phase"],
                "block_row": payload["block_row"],
                "block_col": payload["block_col"],
                "output_rows": payload["output_rows"],
                "output_cols": payload["output_cols"],
                "c_block": c_block.tolist(),
            }
        except Exception as exc:
            event = {
                "task_id": payload.get("task_id"),
                "layer_id": payload.get("layer_id"),
                "phase": payload.get("phase"),
                "op_type": payload.get("op_type"),
                "status": "error",
                "error_message": str(exc),
            }
            self.server.state.record(event)
            return {
                "type": "RESULT",
                "status": "error",
                "task_id": payload.get("task_id"),
                "layer_id": payload.get("layer_id"),
                "phase": payload.get("phase"),
                "error_message": str(exc),
            }


def run_worker(host: str, port: int, worker_id: int, state_dir: str) -> None:
    state = WorkerState(worker_id=worker_id, state_dir=state_dir)
    with WorkerTCPServer((host, port), WorkerRequestHandler, state) as server:
        LOGGER.info("Worker %s listening on %s:%s", worker_id, host, port)
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed MLP worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--state-dir", default="runtime/worker_state")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_worker(args.host, args.port, args.worker_id, args.state_dir)
