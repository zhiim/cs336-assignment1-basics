import os
from math import cos, pi
from typing import IO, BinaryIO

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.modules import softmax


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


def decode(
    prompt: torch.Tensor,
    model: nn.Module,
    max_num_tokens: int,
    temperature: float,
    top_p: float,
    vocab: dict[int, bytes],
):
    completion = []
    while not (
        len(completion) >= max_num_tokens
        or (len(completion) > 0 and completion[-1] == b"<|endoftext|>")
    ):
        pred = model(prompt)[-1]

        prob = softmax(pred / temperature)

        idx_prob = {idx: p.detach().item() for idx, p in enumerate(prob)}
        idx_prob = sorted(
            idx_prob.items(), key=lambda item: item[1], reverse=True
        )

        accumlate = 0.0
        probs = []
        idxs = []
        for idx, p in idx_prob:
            probs.append(p)
            idxs.append(idx)
            accumlate += p
            if accumlate >= top_p:
                break

        selected = np.random.choice(idxs, size=1, p=probs)[0]

        completion.append[vocab[selected]]

    out = b"".join(completion[:-1])
    out = out.decode("utf-8")

    return out
