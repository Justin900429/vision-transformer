import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 norm_layer: nn.Module = nn.LayerNorm):
        super(PatchEmbed, self).__init__()

        assert (img_size % patch_size == 0), "Argument `img_size` should be factor of argument `patch_size`"
        self.grid_size = (img_size // patch_size)
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # Before: Shape of x: (batch_size, channels, width, height)
        # After: Shape of x: (batch_size, embed_dim, img_size // patch_size, img_size // patch_size)
        x = self.proj(x)

        # Shape of x: (batch_size, (img_size // patch_size) ** 2, embed_dim)
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.norm(x)
        return x
