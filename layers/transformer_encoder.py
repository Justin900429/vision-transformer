"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from typing import Optional
import torch.nn as nn
from .feed_forward import FeedForward
from .attention import Attention
from utils import StochasticDepth


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 factor: int = 4,
                 qkv_bias: bool = False,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 stochastic_drop_prob: float = 0.0):
        super(TransformerEncoder, self).__init__()

        self.attn = nn.Sequential(
            norm_layer(dim),
            Attention(dim=dim,
                      num_heads=num_heads,
                      qkv_bias=qkv_bias,
                      attn_drop=attn_drop,
                      proj_drop=drop),
        )

        self.feed_forward = nn.Sequential(
            norm_layer(dim),
            FeedForward(in_features=dim,
                        factor=factor,
                        act_layer=act_layer,
                        drop=drop),
        )
        self.drop_path = StochasticDepth(stochastic_drop_prob)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_size)
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.feed_forward(x))

        return x
