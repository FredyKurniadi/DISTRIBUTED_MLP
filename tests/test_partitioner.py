import numpy as np

from src.distributed.partitioner import build_matmul_tasks


def test_partitioner_covers_output_grid():
    a = np.random.randn(5, 4)
    b = np.random.randn(4, 6)

    tasks = build_matmul_tasks(
        a=a,
        b=b,
        layer_id=0,
        phase="forward",
        op_type="matmul",
        partition_rows=2,
        partition_cols=3,
    )

    assert len(tasks) == 6
    assert tasks[0].a_block.shape[1] == b.shape[0]
