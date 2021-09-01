from typing import Union
import torch.nn as nn
import einops


class PatchMerging(nn.Module):
    """Adapted from
    https://github.com/microsoft/Swin-Transformer/blob/777f6c66604bb5579086c4447efe3620344d95a9/models/swin_transformer.py#L291-L337
    https://zhuanlan.zhihu.com/p/361366090
    """
    def __init__(self, embed_dim: int,
                 input_resolution: Union[tuple, list],
                 norm_layer: nn.Module = nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.patch_merge = nn.Unfold(
            kernel_size=2, stride=2
        )
        self.linear = nn.Sequential(
            norm_layer(embed_dim * 4),
            nn.Linear(
                in_features=embed_dim * 4,
                out_features=embed_dim * 2,
                bias=False
            )
        )

    def forward(self, x):
        height, width = self.input_resolution
        x = einops.rearrange(x, "b (h w) e -> b e h w",
                             h=height, w=width)
        # Before: Shape of x: (batch_size, in_channels, height, width)
        # After: Shape of x: (batch_size, out_channels, new_h * new_w)
        #  where out_channels = (in_channels * (down_factor)**2), down_factor = 2
        x = self.patch_merge(x)

        # Shape of x: (batch_size, new_h * new_w, out_channels)
        x = x.transpose(-1, -2)

        # If use linear: Shape of x: (batch_size, new_h * new_w, out_channels // 2)
        x = self.linear(x)

        return x
