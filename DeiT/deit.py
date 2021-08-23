"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from typing import Optional
from functools import partial
import torch
import torch.nn as nn
from einops import repeat
from layers.patch_embed import PatchEmbed
from layers.transformer_encoder import TransformerEncoder


class DeiT(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 factor: int = 4,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 embed_layer: nn.Module = PatchEmbed,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: Optional = nn.GELU):
        super(DeiT, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        norm_layer = partial(norm_layer, eps=1e-6)
        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerEncoder(dim=embed_dim,
                               num_heads=num_heads,
                               factor=factor,
                               qkv_bias=qkv_bias,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               norm_layer=norm_layer,
                               act_layer=act_layer)
            for _ in range(depth)
        ])

        # Classifier heads
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head_list = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        # Input embedding
        x = self.patch_embed(x)
        cls_token = repeat(self.cls_token, "b s e -> (b repeat) s e",
                           repeat=x.size(0))
        dist_token = repeat(self.dist_token, "b s e -> (b repeat) s e",
                            repeat=x.size(0))
        x = torch.cat([cls_token, dist_token, x], dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Go through transformer encoder
        # Before: Shape of x: (batch_size, seq_len, embed_dim)
        x = self.blocks(x)

        # Shape of x: (batch_size, num_classes)
        # Shape of x_dist: (batch_size, num_classes)
        x, x_dist = self.head(x[:, 0]), self.head_list(x[:, 1])

        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = DeiT()
    output = model(test_tensor)
    print(output[0].shape, output[1].shape)

    model.eval()
    print(model(test_tensor).shape)
