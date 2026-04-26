import numpy as np

from src.distributed.assembler import assemble_blocks
from src.distributed.partitioner import build_matmul_tasks


def test_assemble_blocks_matches_numpy_matmul():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(7, 5))
    b = rng.normal(size=(5, 9))

    tasks = build_matmul_tasks(
        a=a,
        b=b,
        layer_id=1,
        phase="forward",
        op_type="matmul",
        partition_rows=3,
        partition_cols=2,
    )

    results = []
    for task in tasks:
        results.append(
            {
                "output_rows": task.output_rows,
                "output_cols": task.output_cols,
                "c_block": (task.a_block @ task.b_block).tolist(),
            }
        )

    got = assemble_blocks((a.shape[0], b.shape[1]), results)
    expected = a @ b
    assert np.allclose(got, expected)
