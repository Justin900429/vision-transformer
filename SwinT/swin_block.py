from typing import Union
import torch
import torch.nn as nn
import einops
from window_attn import WindowAttention
from utils import StochasticDepth, to_2tuple
from layers import FeedForward


def window_partition(x, window_size):
    x = einops.rearrange(x, "b (h s1) (w s2) c -> (b h w) s1 s2 c",
                         s1=window_size, s2=window_size)
    return x


def window_reverse(windows, window_size, height, width):
    windows = einops.rearrange(windows, "(b h w) s1 s2 c -> b (h s1) (w s2) c",
                               h=height // window_size, w=width // window_size)
    return windows


class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, input_resolution: Union[tuple, list],
                 num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: int = 4, qkv_bias: bool = True, qk_scale: float = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # If window size is larger than input resolution, we don't partition windows
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "Shift_size must in 0-window_size"

        self.norm_one = norm_layer(embed_dim)

        self.attn = WindowAttention(
            embed_dim=embed_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = StochasticDepth(drop_path)
        self.norm_two = norm_layer(embed_dim)
        self.feed_forward = FeedForward(
            in_features=embed_dim, factor=mlp_ratio,
            act_layer=act_layer, drop=drop
        )

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution

            # (batch_size, height, width, num_heads)
            img_mask = torch.zeros((1, H, W, 1))

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Shape of mask_windows: (batch_size * num_windows, window_size, window_size, 1)
            mask_windows = window_partition(img_mask, self.window_size)
            # Shape of mask_windows: (batch_size * num_windows, window_size * window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        height, width = self.input_resolution
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len == height * width, "Input feature has wrong size"

        shortcut = x
        x = self.norm_one(x)
        x = einops.rearrange(x, "b (h w) e -> b h w e",
                             h=height, w=width)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        # Shape of x_windows: (batch_size * num_windows, window_size, window_size, embed_dim)
        x_windows = window_partition(shifted_x, self.window_size)
        # Shape of x_windows: (batch_size * num_windows, window_size * window_size, embed_dim)
        x_windows = einops.rearrange(x_windows, "b w1 w2 e -> b (w1 w2) e")

        # W-MSA/SW-MSA
        # Shape of x_windows: (batch_size * num_windows, window_size * window_size, embed_dim)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = einops.rearrange(attn_windows, "b (w1 w2) e -> b w1 w2 e",
                                        w1=self.window_size, w2=self.window_size)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width)

        # Reverse cycle shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = einops.rearrange(x, "b w h c -> b (w h) c")

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.feed_forward(self.norm_two(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, embed_dim: int, input_resolution: Union[list, tuple], depth: int,
                 num_heads: int, window_size: int, mlp_ratio: int = 4,
                 qkv_bias: bool = True, qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: Union[list, tuple, float] = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, down_sample: nn.Module = None):
        super(BasicLayer, self).__init__()

        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim=embed_dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]
        )

        # patch merging layer
        if down_sample is not None:
            self.down_sample = down_sample(embed_dim=embed_dim,
                                           input_resolution=input_resolution,
                                           norm_layer=norm_layer)
        else:
            self.down_sample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.down_sample is not None:
            x = self.down_sample(x)

        return x
