from typing import Union
import torch
import torch.nn as nn
from SwinT import BasicLayer
from SwinT import PatchMerging
from layers import PatchEmbed
from utils import to_2tuple


class SwinTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 96, depths: Union[tuple, list] = (2, 2, 6, 2),
                 num_heads: Union[tuple, list] = (3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: int = 4, qkv_bias: bool = True,
                 qk_scale: float = None, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer: nn.Module = nn.LayerNorm,
                 abs_embed: bool = False, patch_norm: bool = True):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.abs_embed = abs_embed
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = to_2tuple(self.patch_embed.grid_size)
        self.patches_resolution = patches_resolution

        # Absolute embedding
        if self.abs_embed:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed.data, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                embed_dim=int(embed_dim * 2 ** idx),
                input_resolution=(patches_resolution[0] // (2 ** idx),
                                  patches_resolution[1] // (2 ** idx)),
                depth=depths[idx],
                num_heads=num_heads[idx],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:idx]):sum(depths[:idx + 1])],
                norm_layer=norm_layer,
                down_sample=PatchMerging if (idx < self.num_layers - 1) else None
            )
            for idx in range(self.num_layers)
        ])

        self.norm = norm_layer(self.num_features)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.abs_embed:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # Shape of x: (batch_size ,seq_len, embed_dim)
        x = self.norm(x)
        # Shape of x: (batch_size ,seq_len, 1)
        x = self.avg_pool(x.transpose(1, 2))
        x = x.flatten(start_dim=1)

        x = self.head(x)
        return x


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = SwinTransformer(num_classes=10)

    print(model(test_tensor).shape)
