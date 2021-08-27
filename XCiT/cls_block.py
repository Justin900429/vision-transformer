from typing import Optional
import torch
import torch.nn as nn
import einops
from utils import StochasticDepth
from layers import FeedForward


class ClassAttention(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L144-L173
    """
    def __init__(self, embed_dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

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
                               n=3,
                               h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get only class token and remain the dimension
        # Shape of q_cls: (batch_size, num_heads, 1, head_dim)
        q_cls = q[:, :, 0:1]

        # Shape of attn_cls: (batch_size, num_heads, seq_len)
        attn_cls = (q_cls * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        # Shape of cls_token: (batch_size, 1, embed_dim)
        cls_token = torch.einsum("bns,bnsd->bnd", attn_cls, v)
        cls_token = einops.rearrange(cls_token, "b n d -> b (n d)").unsqueeze(1)
        cls_token = self.proj(cls_token)

        x = torch.cat([self.proj_drop(cls_token), x[:, 1:]], dim=1)

        return x


class ClassAttentionBlock(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L176-L218
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Optional = nn.GELU,
                 norm_layer: Optional = nn.LayerNorm,
                 eta: float = None,
                 tokens_norm: bool = False):
        super(ClassAttentionBlock, self).__init__()
        self.norm_one = norm_layer(embed_dim)

        self.attn = ClassAttention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = StochasticDepth(drop_prob=drop_path)
        self.norm_two = norm_layer(embed_dim)
        self.feed_forward = FeedForward(in_features=embed_dim,
                                        factor=mlp_ratio,
                                        act_layer=act_layer,
                                        drop=drop)
        if eta is not None:
            self.gamma_one = nn.Parameter(eta * torch.ones(embed_dim),
                                          requires_grad=True)
            self.gamma_two = nn.Parameter(eta * torch.ones(embed_dim),
                                          requires_grad=True)
        else:
            self.gamma_one, self.gamma_two = 1.0, 1.0

        # Whether normalized all the tokens or just the cls tokens
        self.tokens_norm = tokens_norm

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        x = x + self.drop_path(self.gamma_one * self.attn(self.norm_one(x)))
        if self.tokens_norm:
            x = self.norm_two(x)
        else:
            x[:, 0:1] = self.norm_two(x[:, 0:1])

        x_res = x
        # Shape of cls_token: (batch_size, 1, embed_dim)
        cls_token = x[:, 0:1]
        cls_token = self.gamma_two * self.feed_forward(cls_token)

        # Shape of x: (batch_size, seq_len, embed_dim)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x
