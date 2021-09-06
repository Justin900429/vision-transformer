from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PVT import TransformerBlock
from layers import PatchEmbed
from utils import to_2tuple


class PVT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: Union[tuple, list] = (64, 128, 256, 512),
                 num_heads: Union[tuple, list] = (1, 2, 4, 8),
                 mlp_ratios: Union[tuple, list] = (4, 4, 4, 4),
                 qkv_bias: bool = False, qk_scale: float = None,
                 drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 depths: Union[tuple, list] = (3, 4, 6, 3),
                 sr_ratios: Union[tuple, list] = (8, 4, 2, 1),
                 num_stages: int = 4):
        super(PVT, self).__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_channels=in_channels if i == 0 else embed_dim[i - 1],
                embed_dim=embed_dim[i]
            )
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([
                TransformerBlock(
                    embed_dim=embed_dim[i], num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + idx], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i]
                )
                for idx in range(depths[i])
            ])
            cur += depths[i]

            setattr(self, f"patch_embed_{i + 1}", patch_embed)
            setattr(self, f"pos_embed_{i + 1}", pos_embed)
            setattr(self, f"pos_drop_{i + 1}", pos_drop)
            setattr(self, f"block_{i + 1}", block)

        self.norm = norm_layer(embed_dim[3])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[3]))
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()

        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed_{i + 1}")
            nn.init.trunc_normal_(pos_embed.data, std=.02)
        nn.init.trunc_normal_(self.cls_token.data, std=.02)
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

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed_1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.grid_size, patch_embed.grid_size, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False
            ).reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed_{i + 1}")
            pos_embed = getattr(self, f"pos_embed_{i + 1}")
            pos_drop = getattr(self, f"pos_drop_{i + 1}")
            blocks = getattr(self, f"block_{i + 1}")
            x = patch_embed(x)
            H, W = to_2tuple(patch_embed.grid_size)

            # Make the size of the positional embedding same with the patch_size
            #  if the size is different (eg. using pre-trained)
            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                # Take out the cls_token
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for block in blocks:
                x = block(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)[:, 0]
        x = self.head(x)

        return x


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    test_tensor = torch.randn(1, 3, 224, 224)
    model = PVT(num_classes=10)
    print(model(test_tensor).shape)
