from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from layers import FeedForward
from utils import StochasticDepth
from .LPI import LPI


class XCAttention(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L221-L261
    """
    def __init__(self, embed_dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super(XCAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of qkv: (batch_size, seq_len, embed_dim * 3)
        qkv = self.qkv(x)

        # Shape of qkv: (3, batch_size, num_heads, seq_len, head_dim)
        # Shape of q, k, v: (batch_size, num_heads, seq_len, head_dim)
        qkv = einops.rearrange(qkv, "b s (n h e) -> n b h s e",
                               n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Shape of q, k, v: (batch_size, num_heads, head_dim, seq_len)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Normalize Q, K to (-1, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Shape of attn: (batch_size, num_heads, head_dim, head_dim)
        attn = torch.einsum("bnqh,bnkh->bnqk", q, k) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Shape of x: (batch_size, num_heads, head_dim, seq_len)
        x = torch.einsum("bnqk,bnks->bnqs", attn, v)
        x = einops.rearrange(x, "b n q s -> b s (n q)")

        # Shape of x: (batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCBlock(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L264-L292
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = False,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: Optional = nn.GELU,
                 norm_layer: Optional = nn.LayerNorm,
                 eta=None):
        super(XCBlock, self).__init__()
        self.norm_one = norm_layer(embed_dim)

        self.attn = XCAttention(embed_dim=embed_dim,
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_drop,
                                proj_drop=drop)
        self.drop_path = StochasticDepth(drop_prob=drop_path)
        self.norm_two = norm_layer(embed_dim)

        self.feed_forward = FeedForward(in_features=embed_dim,
                                        factor=mlp_ratio,
                                        act_layer=act_layer,
                                        drop=drop)
        self.norm_three = norm_layer(embed_dim)
        self.local_map = LPI(in_features=embed_dim,
                             act_layer=act_layer)

        self.gamma_one = nn.Parameter(eta * torch.ones(embed_dim),
                                      requires_grad=True)
        self.gamma_two = nn.Parameter(eta * torch.ones(embed_dim),
                                      requires_grad=True)
        self.gamma_three = nn.Parameter(eta * torch.ones(embed_dim),
                                        requires_grad=True)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.gamma_one * self.attn(self.norm_one(x)))
        x = x + self.drop_path(self.gamma_two * self.local_map(self.norm_two(x), h, w))
        x = x + self.drop_path(self.gamma_three * self.feed_forward(self.norm_three(x)))

        return x
