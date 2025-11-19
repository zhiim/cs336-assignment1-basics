from math import sqrt
from typing import Callable, Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # get current m, v and t
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)

                grad = p.grad.detach()

                # calculate m and v
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2

                # update parameter
                lr_ = lr * sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.add_(m / (torch.sqrt(v) + eps), alpha=-lr_)
                p.mul_(1 - lr * weight_decay)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
