from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatMulTask:
    task_id: str
    layer_id: int
    phase: str
    op_type: str
    block_row: int
    block_col: int
    output_rows: tuple[int, int]
    output_cols: tuple[int, int]
    a_block: np.ndarray
    b_block: np.ndarray


def _split_bounds(length: int, chunks: int) -> list[tuple[int, int]]:
    step = (length + chunks - 1) // chunks
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + step)
        bounds.append((start, end))
        start = end
    return bounds


def build_matmul_tasks(
    a: np.ndarray,
    b: np.ndarray,
    layer_id: int,
    phase: str,
    op_type: str,
    partition_rows: int,
    partition_cols: int,
) -> list[MatMulTask]:
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch for matmul: {a.shape} x {b.shape}")

    row_bounds = _split_bounds(a.shape[0], max(1, partition_rows))
    col_bounds = _split_bounds(b.shape[1], max(1, partition_cols))

    tasks: list[MatMulTask] = []
    for i, (r0, r1) in enumerate(row_bounds):
        for j, (c0, c1) in enumerate(col_bounds):
            tasks.append(
                MatMulTask(
                    task_id=f"L{layer_id}_{phase}_{i}_{j}",
                    layer_id=layer_id,
                    phase=phase,
                    op_type=op_type,
                    block_row=i,
                    block_col=j,
                    output_rows=(r0, r1),
                    output_cols=(c0, c1),
                    a_block=a[r0:r1, :],
                    b_block=b[:, c0:c1],
                )
            )
    return tasks
