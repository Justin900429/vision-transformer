import torch.nn as nn
from utils import StochasticDepth
from layers import FeedForward
from CvT import ConvAttention


class CvTBlock(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int,
                 num_heads: int, mlp_ratio: int = 4,
                 qkv_bias: bool = False, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 **kwargs):
        super(CvTBlock, self).__init__()

        self.norm_one = norm_layer(embed_dim)
        self.attn = ConvAttention(
            embed_dim=embed_dim, out_dim=out_dim,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, drop=drop,
            **kwargs
        )

        self.drop_path = StochasticDepth(drop_path)
        self.norm_two = norm_layer(out_dim)
        self.feed_forward = FeedForward(
            in_features=out_dim, factor=mlp_ratio,
            act_layer=act_layer, drop=drop
        )

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm_one(x), h, w))
        x = x + self.drop_path(self.feed_forward(self.norm_two(x)))

        return x
