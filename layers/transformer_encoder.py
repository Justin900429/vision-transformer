"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from typing import Optional
import torch.nn as nn
from layers.feed_forward import FeedForward
from layers.attention import Attention


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 factor: int = 4,
                 qkv_bias: bool = False,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 act_layer: Optional = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        self.attn = nn.Sequential(
            Attention(dim=dim,
                      num_heads=num_heads,
                      qkv_bias=qkv_bias,
                      attn_drop=attn_drop,
                      proj_drop=drop),
            norm_layer(dim),
        )

        self.feed_forward = nn.Sequential(
            FeedForward(in_features=dim,
                        factor=factor,
                        act_layer=act_layer,
                        drop=drop),
            norm_layer(dim)
        )

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_size)
        x = x + self.attn(x)
        x = x + self.feed_forward(x)

        return x
