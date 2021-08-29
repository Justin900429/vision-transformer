import torch
import torch.nn as nn
import einops
from layers import FeedForward
from utils import StochasticDepth


class AttnTalkingHead(nn.Module):
    """
    Using talking head attention instead of vanilla multi-head attention.
    See paper: https://arxiv.org/abs/2003.02436

    Adapted from
    https://github.com/facebookresearch/deit/blob/e6b10b554d17c25c083eda5d5d7505608c6981f8/cait_models.py#L87-L128
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_scale: float = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super(AttnTalkingHead, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of qkv: (batch_size, seq_len, embed_dim * 3)
        qkv = self.qkv(x)

        # Shape of qkv: (3, batch_size, num_heads, seq_len, head_dim)
        # Shape of q, k, v: (batch_size, num_heads, seq_len, head_dim)
        qkv = einops.rearrange(qkv, "b s (n h e) -> n b h s e",
                               n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Shape of attn: (batch_size, seq_len, head_dim, num_heads)
        attn = (q @ k.transpose(-2, -1)).permute(0, 2, 3, 1)

        # Shape of attn: (batch_size, num_heads, head_dim, num_heads)
        attn = self.proj_l(attn).permute(0, 3, 1, 2)
        # Shape of attn: (batch_size, seq_len, head_dim, num_heads)
        attn = attn.softmax(dim=-1).permute(0, 2, 3, 1)

        # Shape of attn: (batch_size, num_heads, head_dim, num_heads)
        attn = self.proj_w(attn).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        # Shape of x: (batch_size, seq_len, embed_dim)
        x = einops.rearrange((attn @ v), "b n s e -> b s (n e)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScaleBlock(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/deit/blob/e6b10b554d17c25c083eda5d5d7505608c6981f8/cait_models.py#L130-L150
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, qkv_bias: bool = False,
                 qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 attention_block: nn.Module = AttnTalkingHead,
                 feed_forward_block: nn.Module = FeedForward,
                 init_values: float = 1e-4):
        super().__init__()
        self.norm_one = norm_layer(embed_dim)
        self.attn = attention_block(
            embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
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

    def forward(self, x):
        x = x + self.drop_path(self.gamma_one * self.attn(self.norm_one(x)))
        x = x + self.drop_path(self.gamma_two * self.mlp(self.norm_two(x)))
        return x
