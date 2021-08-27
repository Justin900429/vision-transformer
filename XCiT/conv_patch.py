import torch.nn as nn


def to_2tuple(value):
    return value, value


class ConvBlock3x3(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int):
        super(ConvBlock3x3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class ConvPatchEmbed(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L68-L108
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super(ConvPatchEmbed, self).__init__()
        assert patch_size in [8, 16], "Patch size has to be in [8, 15] for convolutional projection"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = nn.Sequential(
                ConvBlock3x3(in_channels, embed_dim // 8, 2),
                nn.GELU(),
                ConvBlock3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                ConvBlock3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                ConvBlock3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = nn.Sequential(
                ConvBlock3x3(in_channels, embed_dim // 4, 2),
                nn.GELU(),
                ConvBlock3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                ConvBlock3x3(embed_dim // 2, embed_dim, 2)
            )

    def forward(self, x):
        # Before: Shape of x: (batch_size, in_channels, height, width)
        # After: Shape of x: (batch_size, embed_dim, out_height, out_width)
        x = self.proj(x)
        height, width = x.shape[-2:]

        # Shape of x: (batch_size, out_height * out_width, embed_dim)
        x = x.flatten(start_dim=2).transpose(-2, -1)

        return x, (height, width)
