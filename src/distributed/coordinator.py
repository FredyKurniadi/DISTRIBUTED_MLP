from __future__ import annotations

import socket
import threading
from dataclasses import asdict
from queue import Empty, Queue
from typing import Any

import numpy as np

from src.distributed.assembler import assemble_blocks
from src.distributed.partitioner import MatMulTask, build_matmul_tasks
from src.distributed.protocol import recv_json, send_json


class DistributedCoordinator:
    def __init__(
        self,
        worker_hosts: list[dict[str, Any]],
        partition_rows: int,
        partition_cols: int,
        timeout_sec: int,
        retry_count: int,
    ) -> None:
        self.worker_hosts = worker_hosts
        self.partition_rows = partition_rows
        self.partition_cols = partition_cols
        self.timeout_sec = timeout_sec
        self.retry_count = retry_count

    def distributed_matmul(self, a: np.ndarray, b: np.ndarray, layer_id: int, phase: str, op_type: str) -> np.ndarray:
        tasks = build_matmul_tasks(
            a=a,
            b=b,
            layer_id=layer_id,
            phase=phase,
            op_type=op_type,
            partition_rows=self.partition_rows,
            partition_cols=self.partition_cols,
        )

        results = self._dispatch_tasks(tasks)
        ok_results = [r for r in results if r.get("status") == "ok"]
        if len(ok_results) != len(tasks):
            failed = [r for r in results if r.get("status") != "ok"]
            raise RuntimeError(f"Distributed matmul failed: {failed}")

        return assemble_blocks(shape=(a.shape[0], b.shape[1]), results=ok_results)

    def _dispatch_tasks(self, tasks: list[MatMulTask]) -> list[dict[str, Any]]:
        if not tasks:
            return []

        task_queue: Queue[MatMulTask] = Queue()
        for task in tasks:
            task_queue.put(task)

        results: list[dict[str, Any]] = []
        results_lock = threading.Lock()

        def worker_loop(worker: dict[str, Any]) -> None:
            while True:
                try:
                    task = task_queue.get_nowait()
                except Empty:
                    break

                try:
                    result = self._send_task_with_retry(worker, task)
                finally:
                    task_queue.task_done()

                with results_lock:
                    results.append(result)

        threads: list[threading.Thread] = []
        for index, worker in enumerate(self.worker_hosts):
            thread = threading.Thread(target=worker_loop, args=(worker,), name=f"worker-dispatch-{index}", daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def _send_task_with_retry(self, worker: dict[str, Any], task: MatMulTask) -> dict[str, Any]:
        last_error = None
        for _ in range(self.retry_count + 1):
            try:
                return self._send_task(worker, task)
            except Exception as exc:
                last_error = str(exc)

        return {
            "type": "RESULT",
            "status": "error",
            "task_id": task.task_id,
            "layer_id": task.layer_id,
            "phase": task.phase,
            "error_message": f"Retry exhausted: {last_error}",
        }

    def _send_task(self, worker: dict[str, Any], task: MatMulTask) -> dict[str, Any]:
        payload = asdict(task)
        payload["type"] = "TASK"
        payload["a_block"] = task.a_block.tolist()
        payload["b_block"] = task.b_block.tolist()

        host = worker["host"]
        port = int(worker["port"])
        with socket.create_connection((host, port), timeout=self.timeout_sec) as sock:
            send_json(sock, payload)
            return recv_json(sock)

    def heartbeat(self) -> list[dict[str, Any]]:
        replies = []
        for worker in self.worker_hosts:
            host = worker["host"]
            port = int(worker["port"])
            with socket.create_connection((host, port), timeout=self.timeout_sec) as sock:
                send_json(sock, {"type": "HEARTBEAT"})
                replies.append(recv_json(sock))
        return replies
