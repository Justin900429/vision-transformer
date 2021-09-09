"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
"""
from typing import Optional
import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 stride: int = None,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 multi_conv: bool = False,
                 norm_layer: Optional = nn.LayerNorm):
        super(PatchEmbed, self).__init__()

        assert (img_size % patch_size == 0), "Argument `img_size` should be factor of argument `patch_size`"
        self.grid_size = (img_size // patch_size)
        self.patch_size = patch_size
        self.num_patches = self.grid_size ** 2

        if stride is None:
            stride = patch_size

        # For cross variance vision transformer
        if multi_conv:
            if patch_size == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=embed_dim // 4,
                        kernel_size=7, stride=4, padding=3
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=embed_dim // 4, out_channels=embed_dim // 2,
                        kernel_size=3, stride=3
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=embed_dim // 2, out_channels=embed_dim,
                        kernel_size=3, stride=1, padding=1
                    ),
                )
            elif patch_size == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(
                        in_channels, embed_dim // 4,
                        kernel_size=7, stride=4, padding=3
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=embed_dim // 4, out_channels=embed_dim // 2,
                        kernel_size=3, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=embed_dim // 2, out_channels=embed_dim,
                        kernel_size=3, stride=2, padding=1
                    )
                )
        else:
            self.proj = nn.Conv2d(
                in_channels=in_channels, out_channels=embed_dim,
                kernel_size=patch_size, stride=stride
            )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer

    def forward(self, x):
        # Before: Shape of x: (batch_size, channels, width, height)
        # After: Shape of x: (batch_size, embed_dim, img_size // patch_size, img_size // patch_size)
        x = self.proj(x)

        # Shape of x: (batch_size, (img_size // patch_size) ** 2, embed_dim)
        x = x.flatten(start_dim=2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)
        return x


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int,
                 seq_len: int = 5000):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, seq_len, embed_dim)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    embed = PatchEmbed()
    print(embed(test_tensor).shape)

    test_sinusoidal_embed = torch.rand(1, 256, 768)
    sin_embed = SinusoidalEmbedding(768, seq_len=256)
    print(sin_embed(test_sinusoidal_embed).shape)
