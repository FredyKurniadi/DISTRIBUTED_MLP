from __future__ import annotations

from typing import Any

import numpy as np


def assemble_blocks(shape: tuple[int, int], results: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    for result in results:
        r0, r1 = result["output_rows"]
        c0, c1 = result["output_cols"]
        block = np.asarray(result["c_block"], dtype=np.float64)
        out[r0:r1, c0:c1] = block
    return out
