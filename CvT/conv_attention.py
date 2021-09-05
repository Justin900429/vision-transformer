import torch
import torch.nn as nn
import einops


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super(DepthWiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, groups=in_channels
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        return self.layers(x)


class ConvAttention(nn.Module):
    """Adapted from
    https://github.com/microsoft/CvT/blob/4cedb05b343e13ab08c0a29c5166b6e94c751112/lib/models/cls_cvt.py#L77-L212
    """
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 8,
                 kernel_size: int = 3, q_stride: int = 1, kv_stride: int = 1,
                 qk_scale: float = None, drop: float = 0., attn_drop: float = 0.0,
                 last_stage: bool = False, qkv_bias: bool = False):
        super(ConvAttention, self).__init__()

        # Whether has cls token
        self.last_stage = last_stage

        self.num_heads = num_heads
        self.scale = qk_scale or out_dim ** -0.5

        self.conv_q = DepthWiseConv(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=kernel_size, stride=q_stride,
            padding=kernel_size // 2
        )
        self.conv_k = DepthWiseConv(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=kernel_size, stride=kv_stride,
            padding=kernel_size // 2
        )
        self.conv_v = DepthWiseConv(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=kernel_size, stride=kv_stride,
            padding=kernel_size // 2
        )
        self.q_w = nn.Linear(embed_dim, out_dim, bias=qkv_bias)
        self.k_w = nn.Linear(embed_dim, out_dim, bias=qkv_bias)
        self.v_w = nn.Linear(embed_dim, out_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Dropout(drop)
        )

    def forward(self, x, h, w):
        # Shape of x: (batch_size, height * width, embed_dim)
        # Shape of cls: (batch_size, 1, embed_dim)
        if self.last_stage:
            cls_token, x = x[:, 0:1], x[:, 1:]

        # Shape of x: (batch_size, embed_dim, height, width)
        x = einops.rearrange(x, 'b (h w) d -> b d h w ',
                             h=h, w=w)

        # Before: Shape of q : (batch_size, out_dim, q_height, q_width)
        # After: Shape of q: (batch_size, q_height * q_width, out_dim)
        q = self.conv_q(x)
        q = einops.rearrange(q, "b d h w -> b (h w) d")
        # Before: Shape of k, v: (batch_size, out_dim, kv_height, kv_width)
        # After: Shape of k, v: (batch_size, kv_height * kv_width, out_dim)
        k = self.conv_k(x)
        k = einops.rearrange(k, "b d h w -> b (h w) d")
        v = self.conv_v(x)
        v = einops.rearrange(v, "b d h w -> b (h w) d")

        # Shape of q, k, v: (batch_size, seq_len + 1, out_dim)
        # where seq_len={q,kv,kv}_length, respectively
        if self.last_stage:
            q = torch.cat([cls_token, q], dim=1)
            k = torch.cat([cls_token, k], dim=1)
            v = torch.cat([cls_token, v], dim=1)

        # Shape of q, k, v: (batch_size, num_heads, {q,kv,kv}_length, head_dim)
        q = einops.rearrange(self.q_w(q), 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(self.k_w(k), 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(self.v_w(v), 'b s (h d) -> b h s d', h=self.num_heads)

        # Shape of attn: (batch_size, num_heads, q_length, kv_length)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Shape of x: (batch_size, num_heads, q_length, head_dim)
        x = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        # Shape of x: (batch_size, q_length, out_dim)
        x = einops.rearrange(x, 'b h s d -> b s (h d)')
        x = self.proj(x)

        return x


