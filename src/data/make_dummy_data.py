from __future__ import annotations

import numpy as np

from src.mlp.model import MLPParams, forward_local


def generate_teacher_student_data(
    teacher_params: MLPParams,
    n_samples: int,
    input_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, input_dim)).astype(np.float64)
    y = forward_local(x, teacher_params)
    return x, y
