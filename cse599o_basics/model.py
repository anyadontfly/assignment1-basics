import math

import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Bool


def softmax(x: Tensor, dim: int) -> Tensor:
    assert len(x.shape) > dim, (
        f"dimension {dim} out of range of input dimension {x.shape}"
    )
    normalized_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(normalized_x) / torch.sum(torch.exp(normalized_x), dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Tensor:
    d_k = q.shape[-1]
    qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))
    return torch.matmul(softmax(qk, -1), v)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        r"""
        Construct a linear transformation module.
        
        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameter
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.weight.t())


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None,
    ) -> None:
        r"""
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device: Device to store the parameters on
            dtype: Data type of the parameter
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor)-> Tensor:
        bsz, sqlen = token_ids.shape
        out = torch.empty(bsz, sqlen, self.embedding_dim)
        for i in range(bsz):
            for j in range(sqlen):
                out[i, j] = self.weight[token_ids[i, j]]
        return out


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        r"""
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(d_model, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1.0)

    def forward(self, x: Tensor)-> Tensor:
        r"""
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        assert len(x.shape) == 3, (
            f"Expected input of shape (batch_size, sequence_length, d_model), "
            f"but got tensor with shape {tuple(x.shape)}"
        )
        inv_rms = torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * inv_rms * self.weight).type_as(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ) -> None:
        r"""
        Construct the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int): Dimension of the up-projection
            device: Device to store the parameters on
            dtype: Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(
            torch.empty(d_ff, d_model, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(d_model, d_ff, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.w3_weight = nn.Parameter(
            torch.empty(d_ff, d_model, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.d_model + self.d_ff))
        for w in [self.w1_weight, self.w2_weight, self.w3_weight]:
            nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        w1_prod = torch.matmul(x, self.w1_weight.t())
        silu_w1_prod = w1_prod * torch.sigmoid(w1_prod)
        w3_prod = torch.matmul(x, self.w3_weight.t()) * silu_w1_prod
        return torch.matmul(w3_prod, self.w2_weight.t())


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ) -> None:
        r"""
        Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Theta value for RoPE
            d_k (int): Dimension of query/key vectors (should be even)
            max_seq_len (int): Maximum sequence length that will be inputted
            device: Device to store the buffers on
        """
        super().__init__()
        assert d_k % 2 == 0, "Expected event d_k"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len, device=device)
        angles = theta ** (-(2 * (torch.arange(1, d_k // 2 + 1, device=device) - 1)) / d_k)
        freqs = torch.outer(pos, angles)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    )-> Tensor:
        r"""
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.

        Notes:
            - Accept x with an arbitrary number of batch dimensions.
            - token_positions has shape (..., seq_len) and gives absolute
              positions per token along the sequence dimension.
            - Use token_positions to slice (precomputed) cos/sin tensors
              along the sequence dimension.
        """
        out = torch.empty_like(x).to(x.device)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        r"""
        Construct the multi-head self-attention.

        Args:
            d_model (int): Dimensionality of the Transformer block inputs
            num_heads (int): Number of heads to use in multi-head self-attention
            device: Device to store the buffers on
            dtype: Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device),
            requires_grad=True
        )  # (num_heads * d_head, d_model)
        self.k_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device),
            requires_grad=True
        )  # (num_heads * d_head, d_model)
        self.v_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device),
            requires_grad=True
        )  # (num_heads * d_head, d_model)
        self.o_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device),
            requires_grad=True
        )  # (d_model, num_heads * d_head)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.d_model + self.d_model))
        nn.init.trunc_normal_(self.q_proj_weight, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.k_proj_weight, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.v_proj_weight, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.o_proj_weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
        self,
        x: Tensor,
        rope_layer: RotaryPositionalEmbedding=None,
        token_positions: Tensor=None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        mask = ~torch.triu(
            torch.full((seqlen, seqlen), True, device=x.device), diagonal=1
        ).view(1, 1, seqlen, seqlen)

        # (bsz, num_head, seqlen, d_head)
        xq = torch.matmul(x, self.q_proj_weight.t()).view(bsz, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        # (bsz, num_head, seqlen, d_head)
        xk = torch.matmul(x, self.k_proj_weight.t()).view(bsz, seqlen, self.num_heads, self.d_head).transpose(1, 2)
        # (bsz, num_head, seqlen, d_head)
        xv = torch.matmul(x, self.v_proj_weight.t()).view(bsz, seqlen, self.num_heads, self.d_head).transpose(1, 2)

        if rope_layer:
            assert token_positions is not None, "Expected token_posotions when applying rotary positional embedding"
            xq = rope_layer(xq, token_positions)
            xk = rope_layer(xk, token_positions)

        atten = scaled_dot_product_attention(xq, xk, xv, mask=mask)  # (bsz, num_head, seqlen, d_head)
        return torch.matmul(atten.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model), self.o_proj_weight.t())
    

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.atten = MultiheadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ff_norm = RMSNorm(d_model)
        self.atten_norm = RMSNorm(d_model)

    def reset_parameters(self) -> None:
        self.ff.reset_parameters()
        self.atten.reset_parameters()
        self.ff_norm.reset_parameters()
        self.atten_norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        rope_layer: RotaryPositionalEmbedding=None,
        token_positions: Tensor=None,
    ) -> Tensor:
        x = x + self.atten(self.atten_norm(x), rope_layer=rope_layer, token_positions=token_positions)
        return x + self.ff(self.ff_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.out = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        _, seqlen = x.shape
        token_positions = torch.tensor([i for i in range(seqlen)])

        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, self.rope, token_positions)
        h = self.norm(h)
        return self.out(h)
