from math import sqrt

import torch
import torch.nn as nn
from einops import einsum, pack, rearrange, reduce


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


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """the SwiGLU feed-forward network, composed of a SiLU activation
        function and a GLU."""
        super().__init__()

        if d_ff is None:
            d_ff = int(((8 / 3) * d_model) // 64 * 64)

        self.w1 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.w2 = Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        w1_x = self.w1(x)
        w1_x = w1_x * torch.sigmoid(w1_x)

        w3_x = self.w3(x)

        out = self.w2(w1_x * w3_x)

        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> None:
        """Construct the RoPE module.

        Args:
            theta (float): theta value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None): Device to store the buffer on
        """
        super().__init__()

        thetas = einsum(
            torch.arange(max_seq_len),
            1 / (theta ** (2 * torch.arange(d_k // 2) / d_k)),
            "max_seq_len, d_k_half -> max_seq_len d_k_half",
        )

        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        rotary_row_1, _ = pack([cos_thetas, -sin_thetas], "seq d *")
        rotary_row_2, _ = pack([sin_thetas, cos_thetas], "seq d *")
        rotary_blocks, _ = pack(
            [
                rearrange(rotary_row_1, "seq d (n1 n2) -> seq d n1 n2", n1=1),
                rearrange(rotary_row_2, "seq d (n1 n2) -> seq d n1 n2", n1=1),
            ],
            "seq d * n",
        )

        rotary_matrix_list = [
            torch.block_diag(*block) for block in rotary_blocks
        ]
        rotary_matrix, _ = pack(rotary_matrix_list, "* d1 d2")

        if dtype is not None:
            rotary_matrix.to(dtype)
        if device is not None:
            rotary_matrix.to(device)
        self.register_buffer("rotary_matrix", rotary_matrix, persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        return einsum(
            x,
            self.rotary_matrix[token_positions],
            "... seq d, ... seq d1 d -> ... seq d1",
        )


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_x

    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    out = exp_x / sum_exp_x

    return out


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
):
    mask = torch.where(mask, 0, -float("inf"))

    d = k.size(-1)  # dim of key
    q_k = einsum(q, k, "... n d, ... m d -> ... n m") / sqrt(d)
    atten_score = softmax(q_k + mask, dim=-1)

    out = einsum(atten_score, v, "... n m, ... m d -> ... n d")

    return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Casusal multi-head self-attention

        Args:
            d_model (int): Dimensionality of the Transformer block inputs
            num_heads (int): Number of heads to use in multi-head self-attention
        """
        super().__init__()

        self.num_heads = num_heads

        dim_head = d_model // num_heads

        self.w_q = Linear(
            in_features=d_model,
            out_features=num_heads * dim_head,
            device=device,
            dtype=dtype,
        )
        self.w_k = Linear(
            in_features=d_model,
            out_features=num_heads * dim_head,
            device=device,
            dtype=dtype,
        )
        self.w_v = Linear(
            in_features=d_model,
            out_features=num_heads * dim_head,
            device=device,
            dtype=dtype,
        )
        self.w_o = Linear(
            in_features=num_heads * dim_head,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len).bool())

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(q, "b seq (h dh) -> b h seq dh", h=self.num_heads)
        k = rearrange(k, "b seq (h dh) -> b h seq dh", h=self.num_heads)
        v = rearrange(v, "b seq (h dh) -> b h seq dh", h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        atten_out = scaled_dot_product_attention(q, k, v, mask)

        atten_out = rearrange(atten_out, "b h seq dh -> b seq (h dh)")

        out = self.w_o(atten_out)

        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """TransformerLayer.

        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention
            d_ff (int): Dimensionality of the position-wise feed-forward inner
                layer
        """
        super().__init__()

        self.atten = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype
        )

        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.norm1 = RMSNorm(
            d_model=d_model, eps=eps, device=device, dtype=dtype
        )
        self.norm2 = RMSNorm(
            d_model=d_model, eps=eps, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        x_norm = self.norm1(x)
        x_atten = self.atten(x_norm, rope, token_positions)
        x = x + x_atten

        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Transformer LM

        Args:
            vocab_size (int): The size of the vocabulary, necessary for
                determining the dimensionality of the token embedding matrix.
            context_length (int): The maximum context length, necessary for
                determining the dimensionality of the position embedding matrix.
            num_layers (int): The number of Transformer blocks to use.
        """
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.trans_layers = nn.Sequential()
        for _ in range(num_layers):
            self.trans_layers.append(
                TransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
            )

        self.norm = RMSNorm(
            d_model=d_model, eps=eps, device=device, dtype=dtype
        )

        self.linear = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        embedd = self.embedding(x)

        trans_out = self.trans_layers(embedd)

        out = self.linear(self.norm(trans_out))

        return softmax(out, dim=-1)
