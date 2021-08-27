from typing import Optional
import torch
import torch.nn as nn
import einops


class LPI(nn.Module):
    """LPI adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L111-L141
    """
    def __init__(self,
                 in_features: int,
                 out_features: int = None,
                 act_layer: Optional = nn.GELU,
                 kernel_size: int = 3):
        super(LPI, self).__init__()
        out_features = out_features or in_features

        # The original norm layer use the `nn.SyncBatchNorm` for multi-GPU training
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=out_features,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=out_features,
                      bias=False),
            act_layer(),
            nn.BatchNorm2d(out_features),
            nn.Conv2d(in_channels=out_features,
                      out_channels=out_features,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=out_features)
        )

    def forward(self, x, h, w):
        x = einops.rearrange(x, "b (h w) c-> b c h w", h=h, w=w)
        x = self.layers(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        return x
