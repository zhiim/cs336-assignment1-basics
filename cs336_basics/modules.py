from math import sqrt

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Construct a linear transformation module. This function should accept
        the following parameters:

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.w = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype)
        )

        self._init_weight(in_features, out_features)

    def _init_weight(self, in_features, out_features):
        sigma = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.w, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "b n d_in, d_out d_in -> b n d_out")
