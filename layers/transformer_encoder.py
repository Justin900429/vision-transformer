"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import torch.nn as nn
from .feed_forward import FeedForward
from .attention import Attention
from utils import StochasticDepth


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 in_dim: int = None,
                 factor: int = 4,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 stochastic_drop_prob: float = 0.0,
                 residual_before: bool = False,
                 attention: nn.Module = Attention):
        super(TransformerEncoder, self).__init__()
        self.residual_before = residual_before
        if in_dim is None:
            in_dim = dim

        self.attn = nn.Sequential(
            norm_layer(dim),
            attention(dim=dim,
                      num_heads=num_heads,
                      in_dim=in_dim,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      attn_drop=attn_drop,
                      proj_drop=drop,
                      residual_before=residual_before),
        )

        self.feed_forward = nn.Sequential(
            norm_layer(in_dim),
            FeedForward(in_features=in_dim,
                        factor=factor,
                        act_layer=act_layer,
                        drop=drop),
        )
        self.drop_path = StochasticDepth(stochastic_drop_prob)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_size)
        # Design for the token2token transformer
        if self.residual_before:
            x = self.drop_path(self.attn(x))
        else:
            x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.feed_forward(x))

        return x
