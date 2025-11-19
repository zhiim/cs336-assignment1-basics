from math import cos, pi

import torch


def cosine_learning_rate_schedule(
    t: int, lr_max: float, lr_min: float, t_w: int, t_c: int
):
    if t < t_w:
        lr_t = t / t_w * lr_max
    elif t > t_c:
        lr_t = lr_min
    else:
        lr_t = lr_min + 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (
            lr_max - lr_min
        )

    return lr_t


def gradient_clipping(params, max_norm: float, eps: float = 1e-6):
    for param in params:
        if param.grad is None:
            continue

        with torch.no_grad():
            norm_p_grad = torch.linalg.norm(param.grad)
            if norm_p_grad >= max_norm:
                param.grad.mul_(max_norm / (norm_p_grad + eps))
