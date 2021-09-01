from typing import Union
import torch
import torch.nn as nn
import einops
from utils import to_2tuple


def get_slide_indices(window_size):
    # Shape of coords: (2, window_height * window_width)
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords = torch.flatten(coords, 1)

    # Shape of relative_coords: (2, window_height * window_width, window_height * window_width)
    relative_coords = coords[:, :, None] - coords[:, None, :]

    # Shape of relative_coords: (window_height * window_width, window_height * window_width, 2)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    # Start from 0
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1

    # The line below is quite tricky. The usage here is too
    #  make (1, 2) and (2, 1) different after they are sum up
    # Therefore, it transform its x coordinate by 2 times of y
    # ex. (1, 2): 1 * (2 * 2 - 1) -> (3, 2)
    # ex. (2, 1): 2 * (2 * 1 - 1) -> (2, 1)
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    # Shape of relative_position_index: (window_height * window_width, window_height * window_width)
    relative_position_index = relative_coords.sum(-1)

    return relative_position_index


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        if isinstance(displacement, int):
            self.displacement = to_2tuple(displacement)
        else:
            self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[0]),
                          dims=(1, 2))


class WindowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 window_size: Union[list, tuple, int],
                 qk_scale: float = None, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super(WindowAttention, self).__init__()

        if isinstance(window_size, int):
            window_size = to_2tuple(window_size)

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        relative_indices = get_slide_indices(window_size)
        self.register_buffer("relative_position_index", relative_indices)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table.data, std=.02)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # Shape of x: (batch_size * num_window, seq_len, embed_dim)
        # Shape of qkv: (batch_size * num_window, seq_len, embed_dim * 3)
        qkv = self.qkv(x)

        # Shape of qkv: (3, batch_size * num_window, num_heads, seq_len, head_dim)
        # Shape of q, k, v: (batch_size * num_window, num_heads, seq_len, head_dim)
        qkv = einops.rearrange(qkv, "b s (n h e) -> n b h s e",
                               n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Shape of attn: (batch_size * num_window, seq_len, seq_len)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Shape of relative_position_bias: (window_height * window_width, window_width * window_height, num_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # Shape of relative_position_bias: (num_heads, window_height * window_width, window_width * window_height)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            temp_attn = einops.rearrange(attn, "(b w) h q k -> b w h q k",
                                         w=num_windows)
            attn = temp_attn + mask.unsqueeze(1).unsqueeze(0)
            attn = einops.rearrange(attn, "b w h q k -> (b w) h q k")
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        # Shape of x: (batch_size * num_windows, num_heads, seq_len, embed_dim)
        x = einops.rearrange(attn @ v, "b h s e -> b s (h e)")

        # Shape of x: (batch_size * num_window, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
