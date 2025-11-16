from math import sqrt

import torch
import torch.nn as nn
from einops import einsum, reduce


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Construct a linear transformation module.

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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device | None): None Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.embedding_map = nn.Parameter(
            torch.zeros(
                (num_embeddings, embedding_dim), device=device, dtype=dtype
            )
        )

        nn.init.trunc_normal_(self.embedding_map, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.embedding_map[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same
        shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            reduce(
                einsum(x, x, "... d_model, ... d_model -> ... d_model"),
                "... d_model -> ...",
                "mean",
            )
            + self.eps
        )

        normed = einsum(
            einsum(x, 1 / rms, "b seq d_model, b seq -> b seq d_model"),
            self.gain,
            "... d_model, d_model -> ... d_model",
        )

        return normed.to(in_dtype)
