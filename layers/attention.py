"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import torch
import torch.nn as nn
import einops


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool,
                 qk_scale: float = None, attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super(Attention, self).__init__()

        self.num_heads = num_heads

        assert (dim % num_heads == 0), "Argument `dim` should be factor of argument `num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        self.q_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x: torch.tensor) \
            -> torch.tensor:
        # Make the dim head
        # Shape of q: (batch_size, num_heads, q_seq_length, head_dim)
        # Shape of k: (batch_size, num_heads, k_seq_length, head_dim)
        # Shape of v: (batch_size, num_heads, v_seq_length, head_dim)
        # NOTE: k_seq_length == v_seq_length
        q = einops.rearrange(self.q_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        k = einops.rearrange(self.k_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        v = einops.rearrange(self.v_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)

        # Compute the attention energy
        # Shape of attn: (batch_size, num_heads, q_seq_length, k_seq_length)
        attn = torch.einsum("bnqd,bnkd->bnqk", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute the final weight on value
        # Shape of x: (batch_size, q_seq_length, emb_size)
        x = torch.einsum("bnqk,bnkd->bnqd", attn, v)
        x = einops.rearrange(x, "b n q d -> b q (n d)")
        x = self.proj(x)

        return x
