from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLPParams:
    weights: list[np.ndarray]
    biases: list[np.ndarray]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def init_mlp(layer_sizes: list[int], seed: int) -> MLPParams:
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        b = np.zeros((1, fan_out), dtype=np.float64)
        weights.append(w.astype(np.float64))
        biases.append(b)

    return MLPParams(weights=weights, biases=biases)


def forward_local(x: np.ndarray, params: MLPParams) -> np.ndarray:
    a = x
    last_index = len(params.weights) - 1
    for i, (w, b) in enumerate(zip(params.weights, params.biases)):
        z = a @ w + b
        a = relu(z) if i < last_index else z
    return a


def params_l2_distance(a: MLPParams, b: MLPParams) -> float:
    total = 0.0
    for wa, wb in zip(a.weights, b.weights):
        total += float(np.mean((wa - wb) ** 2))
    for ba, bb in zip(a.biases, b.biases):
        total += float(np.mean((ba - bb) ** 2))
    return total
