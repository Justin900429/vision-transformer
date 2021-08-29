import torch
import torch.nn as nn
import einops
from layers import FeedForward
from utils import StochasticDepth


class ClassAttention(nn.Module):
    """The implementation of of cls_attn is different
    with the xcit implement, but they do the similar things.

    Adapted from
    https://github.com/facebookresearch/deit/blob/e6b10b554d17c25c083eda5d5d7505608c6981f8/cait_models.py#L21-L55
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_scale: float = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of q: (batch_size, num_heads, 1, embed_dim)
        q = self.q_w(x[:, 0:1])
        q = einops.rearrange(q, "b s (n e) -> b n s e", n=self.num_heads) * self.scale

        # Shape of k, v: (batch_size, num_heads, seq_len, embed_dim)
        k = self.k_w(x)
        k = einops.rearrange(k, "b s (n e) -> b n s e", n=self.num_heads)
        v = self.v_w(x)
        v = einops.rearrange(v, "b s (n e) -> b n s e", n=self.num_heads)

        # Shape of attn: (batch_size, num_heads, 1, seq_len)
        attn = torch.einsum("bnqe,bnke->bnqk", q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Shape of x_cls: (batch_size, num_heads, 1, embed_dim)
        x_cls = torch.einsum("bnqk,bnke->bnqe", attn, v)

        # Shape of x_cls: (batch_size, 1, embed_dim)
        x_cls = einops.rearrange(x_cls, "b n q e -> b q (n e)")
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerCABlock(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/deit/blob/e6b10b554d17c25c083eda5d5d7505608c6981f8/cait_models.py#L57-L84
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, qkv_bias: bool = False,
                 qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 attention_block: nn.Module = ClassAttention,
                 feed_forward_block: nn.Module = FeedForward,
                 init_values: float = 1e-4):
        super(LayerCABlock, self).__init__()
        self.norm_one = norm_layer(embed_dim)
        self.attn = attention_block(
            embed_dim=embed_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = StochasticDepth(drop_path)
        self.norm_two = norm_layer(embed_dim)
        self.mlp = feed_forward_block(
            in_features=embed_dim, factor=mlp_ratio,
            act_layer=act_layer, drop=drop
        )
        self.gamma_one = nn.Parameter(init_values * torch.ones(embed_dim),
                                      requires_grad=True)
        self.gamma_two = nn.Parameter(init_values * torch.ones(embed_dim),
                                      requires_grad=True)

    def forward(self, x, x_cls):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of x_cls: (batch_size, 1, embed_dim)
        x = torch.cat([x_cls, x], dim=1)

        x_cls = x_cls + self.drop_path(self.gamma_one * self.attn(self.norm_one(x)))
        x_cls = x_cls + self.drop_path(self.gamma_two * self.mlp(self.norm_two(x_cls)))

        # Shape of x: (batch_size, seq_len + 1, embed_dim)
        return x_cls
