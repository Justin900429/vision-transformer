from typing import Union, Optional
import torch
import torch.nn as nn
import einops


class LeAttention(nn.Module):
    """Adapted from
    https://github.com/lucidrains/vit-pytorch/blob/529044c9b3f8981efe27b174123798de5274557c/vit_pytorch/levit.py#L40-L106
    """
    def __init__(self, dim: int, fmap_size: int,
                 num_heads: int = 8, dim_key: int = 32,
                 dim_value: int = 64, dropout: float = 0.,
                 dim_out: int = None, downsample: bool = False):
        super(LeAttention, self).__init__()

        inner_dim_key = dim_key * num_heads
        inner_dim_value = dim_value * num_heads
        if dim_out is None:
            dim_out = dim

        self.num_heads = num_heads
        self.scale = dim_key ** -0.5

        self.q_w = nn.Sequential(
            nn.Conv2d(
                in_channels=dim, out_channels=inner_dim_key,
                kernel_size=1,
                stride=(2 if downsample else 1),
                bias=False),
            nn.BatchNorm2d(inner_dim_key)
        )
        self.k_w = nn.Sequential(
            nn.Conv2d(
                in_channels=dim, out_channels=inner_dim_key,
                kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_dim_key)
        )
        self.v_w = nn.Sequential(
            nn.Conv2d(
                in_channels=dim, out_channels=inner_dim_value,
                kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_dim_value)
        )

        # Set up from
        # https://github.com/facebookresearch/LeViT/blob/9d8eda22b6f6046bedd82d368d4282c82e5530ef/levit.py#L104-L132
        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=inner_dim_value,
                out_channels=dim_out,
                kernel_size=1
            ),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # Create an embedding from length to heads
        # Really awesome implementation !!!
        self.pos_bias = nn.Embedding(fmap_size * fmap_size, num_heads)

        q_range = torch.arange(0, fmap_size, step=(2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: einops.rearrange(t, "i j c -> (i j) c"), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim=-1)
        pos_indices = (x_rel * fmap_size) + y_rel
        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = einops.rearrange(bias, "q k n -> () n q k")
        return fmap + (bias / self.scale)

    def forward(self, x):
        # Shape of q: (batch_size, embed_dim, q_height, q_width)
        q = self.q_w(x)
        height = q.shape[2]
        # Shape of q: (batch_size, num_heads, q_length, head_dim)
        q = einops.rearrange(q, "b (n d) ... -> b n (...) d", n=self.num_heads)

        # Shape of k, v: (batch_size, num_heads, kv_length, head_dim)
        k = einops.rearrange(self.k_w(x), "b (n d) ... -> b n (...) d", n=self.num_heads)
        v = einops.rearrange(self.v_w(x), "b (n d) ... -> b n (...) d", n=self.num_heads)

        # Shape of attn: (batch_size, num_heads, q_length, kv_length)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = self.apply_pos_bias(attn)
        attn = attn.softmax(dim=-1)

        # Shape of x: (batch_size, out_dim, out_height, out_width)
        x = einops.rearrange(attn @ v, "b n (h w) d -> b (n d) h w", h=height)
        x = self.proj(x)

        return x


class LeFeedForward(nn.Module):
    def __init__(self, in_features: int,
                 factor: Union[int, float] = None,
                 act_layer: Optional = nn.Hardswish,
                 drop: float = 0.0):
        super(LeFeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features * factor,
                kernel_size=1
            ),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(
                in_channels=in_features * factor,
                out_channels=in_features,
                kernel_size=1
            ),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, fmap_size: int, depth: int,
                 num_heads: int, dim_key: int, dim_value: int,
                 factor: int = 2, dropout: float = 0.,
                 embed_out: int = None, downsample: bool = False):
        super().__init__()
        if embed_out is None:
            embed_out = embed_dim

        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and embed_out == embed_dim

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    LeAttention(
                        dim=embed_dim, fmap_size=fmap_size, num_heads=num_heads,
                        dim_key=dim_key, dim_value=dim_value, dropout=dropout,
                        dim_out=embed_out, downsample=downsample
                    ),
                    LeFeedForward(
                        in_features=embed_out, factor=factor,
                        drop=dropout
                    )
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x
