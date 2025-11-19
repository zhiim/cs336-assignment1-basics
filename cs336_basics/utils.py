import os
from math import cos, pi
from typing import IO, BinaryIO

import numpy as np
import torch


def cosine_learning_rate_schedule(
    t: int, lr_max: float, lr_min: float, t_w: int, t_c: int
) -> float:
    if t < t_w:
        lr_t = t / t_w * lr_max
    elif t > t_c:
        lr_t = lr_min
    else:
        lr_t = lr_min + 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (
            lr_max - lr_min
        )

    return lr_t


@torch.no_grad()
def gradient_clipping(params, max_norm: float, eps: float = 1e-6) -> None:
    param_grads = []
    for param in params:
        if param.grad is None:
            continue
        param_grads.append(
            param.grad.detach()
        )  # 从计算图分离，防止影响梯度计算
    norm_p_grad = torch.linalg.norm(torch.stack(param_grads))

    if norm_p_grad >= max_norm:
        for param in params:
            if param.grad is None:
                continue
            param.grad.mul_(max_norm / (norm_p_grad + eps))


def data_loading(
    x: np.array, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor]:
    total_len = len(x)
    sample_idxs = np.random.choice(total_len - context_length, batch_size)
    data = np.concatenate(
        [x[idx : idx + context_length].reshape(1, -1) for idx in sample_idxs],
        axis=0,
    )
    label = np.concatenate(
        [
            x[idx + 1 : idx + context_length + 1].reshape(1, -1)
            for idx in sample_idxs
        ],
        axis=0,
    )

    return (torch.Tensor(data).to(device), torch.Tensor(label).to(device))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    return state["iteration"]
