import torch
import torch.nn as nn
from layers import PatchEmbed
import einops
from utils import StochasticDepth
from layers import FeedForward


class PVTAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_scale: float = None,
                 attn_drop: float = 0., drop: float = 0.,
                 sr_ratio: int = 1):
        super(PVTAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_w = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(drop)
        )

        # Spatial reduction can be regarded as inner patch embedding
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = PatchEmbed(
                in_channels=embed_dim, embed_dim=embed_dim,
                patch_size=sr_ratio, stride=sr_ratio
            )

    def forward(self, x, H, W):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of q: (batch_size, num_heads, seq_len, head_dim)
        q = einops.rearrange(self.q_w(x), "b s (n h) -> b n s h",
                             n=self.num_heads)

        if self.sr_ratio > 1:
            # Shape of x: (batch_size, embed_dim, height, width)
            x = einops.rearrange(x, "b (h w) c -> b c h w",
                                 h=H, w=W)

            # Shape of x: (batch_size, new_seq_length, embed_dim)
            #  where new_seq_length = new_height * new_width
            x = self.sr(x)

            # Shape of k, v: (batch_size, num_heads, new_seq_length, head_dim)
            k = einops.rearrange(self.k_w(x), "b s (n h) -> b n s h",
                                 n=self.num_heads)
            v = einops.rearrange(self.v_w(x), "b s (n h) -> b n s h",
                                 n=self.num_heads)
        else:
            # Shape of k, v: (batch_size, num_heads, seq_length, head_dim)
            k = einops.rearrange(self.k_w(x), "b s (n h) -> b n s h",
                                 n=self.num_heads)
            v = einops.rearrange(self.v_w(x), "b s (n h) -> b n s h",
                                 n=self.num_heads)

        # Shape of attn: (batch_size, num_heads, q_length, kv_length)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Shape of x: (batch_size, num_heads, q_length, head_dim)
        x = attn @ v
        # Shape of x: (batch_size, q_length, embed_dim)
        x = einops.rearrange(x, "b n s h -> b s (n h)")
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4., qkv_bias: bool = False,
                 qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 sr_ratio: int = 1):
        super(TransformerBlock, self).__init__()
        self.norm_one = norm_layer(embed_dim)
        self.attn = PVTAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, drop=drop,
            sr_ratio=sr_ratio
        )

        self.drop_path = StochasticDepth(drop_path)
        self.norm_two = norm_layer(embed_dim)
        self.mlp = FeedForward(
            in_features=embed_dim, factor=mlp_ratio,
            act_layer=act_layer, drop=drop
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm_one(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm_two(x)))

        return x
