import math
import torch.nn as nn
import einops
from layers import TransformerEncoder


class T2TModule(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3,
                 embed_dim: int = 768, token_dim: int = 64):
        super(T2TModule, self).__init__()

        # First step: soft split
        self.soft_split_zero = nn.Unfold(
            kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
        )

        # Second step: soft-split + reconstruction
        self.attention_one = TransformerEncoder(
            dim=in_channels * 7 * 7, in_dim=token_dim,
            num_heads=1, factor=1, residual_before=True
        )
        self.soft_split_one = nn.Unfold(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        # Third step: soft-split + reconstruction
        self.attention_two = TransformerEncoder(
            dim=token_dim * 3 * 3, in_dim=token_dim,
            num_heads=1, factor=1, residual_before=True
        )
        self.soft_split_two = nn.Unfold(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        # Final step: projection
        self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

    def forward(self, x):
        # Before: Shape of x: (batch_size, in_channels, width, height)
        # After: Shape of x: (batch_size, new_width_one * new_height_one, in_channels * kernel_size[0] * kernel_size[1])
        x = self.soft_split_zero(x).transpose(1, 2)

        # Shape of x: (batch_size, new_width_two * new_height_two, token_dim)
        x = self.attention_one(x)
        _, new_HW, _ = x.shape
        x = einops.rearrange(x.transpose(1, 2), "b c (nh nw) -> b c nh nw",
                             nh=int(math.sqrt(new_HW)), nw=int(math.sqrt(new_HW)))
        x = self.soft_split_one(x).transpose(1, 2)

        # Shape of x: (batch_size, new_width_three * new_height_three, token_dim)
        x = self.attention_two(x)
        _, new_HW, _ = x.shape
        x = einops.rearrange(x.transpose(1, 2), "b c (nh nw) -> b c nh nw",
                             nh=int(math.sqrt(new_HW)), nw=int(math.sqrt(new_HW)))
        x = self.soft_split_two(x).transpose(1, 2)

        # Shape of x: (batch_size, final_width * final_height, embed_dim)
        x = self.project(x)

        return x
