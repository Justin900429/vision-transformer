import math
from functools import partial
from typing import Union
import torch
import torch.nn as nn
import einops
from layers import PatchEmbed, TransformerEncoder
from PiT import ConvHeadPooling


class Transformer(nn.Module):
    def __init__(self, base_dim: int, depth: int, heads: int,
                 mlp_ratio: int, drop_rate: float = .0,
                 attn_drop_rate: float = .0, drop_path_prob: Union[tuple, list] = None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                dim=embed_dim, num_heads=heads,
                factor=mlp_ratio, qkv_bias=True,
                drop=drop_rate, attn_drop=attn_drop_rate,
                stochastic_drop_prob=drop_path_prob[idx],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for idx in range(depth)
        ])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:]

        # Before: Shape of x: (batch_size, in_channels, height, width)
        # After: Shape of x: (batch_size, (height * width), in_channels)
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        token_length = cls_tokens.shape[1]
        # Concat together for the transformer blocks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.blocks(x)

        # Separate the cls_token and others vector for pooling layers
        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x, cls_tokens


class PoolingTransformer(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 stride: int = 8, base_dims: Union[list, tuple] = (48, 48, 48),
                 depth: Union[tuple, list] = (2, 6, 4),
                 heads: Union[tuple, list] = (3, 6, 12),
                 mlp_ratio: int = 4, num_classes: int = 1000,
                 in_channels: int = 3, attn_drop_rate: float = .0,
                 drop_rate: float = .0, drop_path_rate: float = .0):
        super(PoolingTransformer, self).__init__()
        total_block = sum(depth)
        block_idx = 0

        # Compute the width after convolution embedding with no padding
        # Refer to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.width = math.floor(
            (image_size - patch_size) / stride + 1
        )

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], self.width, self.width),
            requires_grad=True
        )
        self.patch_embed = PatchEmbed(
            in_channels=in_channels, embed_dim=base_dims[0] * heads[0],
            patch_size=patch_size, stride=stride, norm_layer=None
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(
                    base_dim=base_dims[stage], depth=depth[stage],
                    heads=heads[stage], mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    drop_path_prob=drop_path_prob
                )
            )

            if stage < len(heads) - 1:
                self.pools.append(
                    ConvHeadPooling(
                        in_feature=base_dims[stage] * heads[stage],
                        out_feature=base_dims[stage + 1] * heads[stage + 1],
                        stride=2
                    )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        self.head = nn.Linear(
            in_features=base_dims[-1] * heads[-1],
            out_features=num_classes
        ) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed.data, std=.02)
        nn.init.trunc_normal_(self.cls_token.data, std=.02)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = einops.rearrange(x, "b (h w) e-> b e h w", h=self.width, w=self.width)

        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)

        # Through the last transformer block
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
        cls_tokens = self.norm(cls_tokens)

        # Use cls token to predict the result
        cls_token = self.head(cls_tokens[:, 0])
        return cls_token


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = PoolingTransformer(num_classes=10)
    print(model(test_tensor).shape)
