from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.make_dummy_data import generate_teacher_student_data
from src.distributed.coordinator import DistributedCoordinator
from src.mlp.model import MLPParams, init_mlp, params_l2_distance, relu, relu_grad
from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger


LOGGER = get_logger("train")


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def _forward_distributed(
    x: np.ndarray,
    params: MLPParams,
    coordinator: DistributedCoordinator,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    a = x
    last_idx = len(params.weights) - 1

    for layer_id, (w, b) in enumerate(zip(params.weights, params.biases)):
        z = coordinator.distributed_matmul(a, w, layer_id=layer_id, phase="forward", op_type="matmul") + b
        a_next = relu(z) if layer_id < last_idx else z
        caches.append((a, w, z, a_next))
        a = a_next

    return a, caches


def _backward_distributed(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    params: MLPParams,
    caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    coordinator: DistributedCoordinator,
    lr: float,
) -> None:
    n = y_true.shape[0]
    d_a = (2.0 / n) * (y_pred - y_true)

    for layer_id in reversed(range(len(params.weights))):
        a_prev, w, z, _ = caches[layer_id]
        is_output = layer_id == len(params.weights) - 1

        d_z = d_a if is_output else d_a * relu_grad(z)
        d_w = coordinator.distributed_matmul(
            a_prev.T,
            d_z,
            layer_id=layer_id,
            phase="backward",
            op_type="matmul_transpose_left",
        )
        d_b = np.sum(d_z, axis=0, keepdims=True)

        d_a_prev = coordinator.distributed_matmul(
            d_z,
            w.T,
            layer_id=layer_id,
            phase="backward",
            op_type="matmul_transpose_right",
        )

        params.weights[layer_id] = w - lr * d_w
        params.biases[layer_id] = params.biases[layer_id] - lr * d_b
        d_a = d_a_prev


def train_distributed(distributed_cfg_path: Path, model_cfg_path: Path) -> None:
    dist_cfg = load_yaml(distributed_cfg_path)
    model_cfg = load_yaml(model_cfg_path)

    layer_sizes = [model_cfg["input_dim"], *model_cfg["hidden_dims"], model_cfg["output_dim"]]
    teacher = init_mlp(layer_sizes=layer_sizes, seed=model_cfg["seed_teacher"])
    student = init_mlp(layer_sizes=layer_sizes, seed=model_cfg["seed_student"])

    x, y = generate_teacher_student_data(
        teacher_params=teacher,
        n_samples=model_cfg["n_samples"],
        input_dim=model_cfg["input_dim"],
        seed=model_cfg["seed_teacher"] + 101,
    )

    coordinator = DistributedCoordinator(
        worker_hosts=dist_cfg["worker_hosts"],
        partition_rows=int(dist_cfg["partition_rows"]),
        partition_cols=int(dist_cfg["partition_cols"]),
        timeout_sec=int(dist_cfg["timeout_sec"]),
        retry_count=int(dist_cfg["retry_count"]),
    )

    _ = coordinator.heartbeat()

    batch_size = int(model_cfg["batch_size"])
    epochs = int(model_cfg["epochs"])
    lr = float(model_cfg["learning_rate"])

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for start in range(0, x.shape[0], batch_size):
            end = min(x.shape[0], start + batch_size)
            xb = x[start:end]
            yb = y[start:end]

            y_pred, caches = _forward_distributed(xb, student, coordinator)
            loss = mse_loss(y_pred, yb)
            _backward_distributed(y_pred, yb, student, caches, coordinator, lr)
            epoch_losses.append(loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            dist = params_l2_distance(student, teacher)
            LOGGER.info(
                "epoch=%s loss=%.6f weight_mse_to_teacher=%.6f",
                epoch,
                float(np.mean(epoch_losses)),
                dist,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distributed MLP student model")
    parser.add_argument(
        "--distributed-config",
        default="configs/distributed.yaml",
        help="Path to distributed config YAML",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        help="Path to model config YAML",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_distributed(Path(args.distributed_config), Path(args.model_config))
