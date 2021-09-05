from typing import Union
import torch
import torch.nn as nn
import einops
from utils import to_2tuple
from CvT import CvTBlock


class ConvEmbed(nn.Module):
    """Adapted from
    https://github.com/microsoft/CvT/blob/4cedb05b343e13ab08c0a29c5166b6e94c751112/lib/models/cls_cvt.py#L336-L369
    """
    def __init__(self, patch_size: int = 7, in_channels: int = 3,
                 embed_dim: int = 64, stride: int = 4, padding: int = 2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        h, w = x.shape[-2:]
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x
    
    
class ViTBlock(nn.Module):
    """Adapted from
    https://github.com/microsoft/CvT/blob/4cedb05b343e13ab08c0a29c5166b6e94c751112/lib/models/cls_cvt.py#L372-L488
    """
    def __init__(self, patch_size: int = 16, patch_stride: int = 16,
                 patch_padding: int = 0, in_channels: int = 3, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: int = 4,
                 qkv_bias: bool = False, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 last_stage: bool = False,
                 **kwargs):
        super(ViTBlock, self).__init__()

        self.num_features = embed_dim
        self.embed_dim = embed_dim

        self.rearrange = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size, in_channels=in_channels,
            stride=patch_stride, padding=patch_padding,
            embed_dim=embed_dim, norm_layer=norm_layer
        )

        if last_stage:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            CvTBlock(
                embed_dim=embed_dim, out_dim=embed_dim,
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate_list[idx],
                act_layer=act_layer, norm_layer=norm_layer,
                last_stage=last_stage, **kwargs
            )
            for idx in range(depth)
        ])

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token.data, std=.02)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = x[:, 0:1], x[:, 1:]
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class CvT(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000,
                 num_stages: int = 3, patch_size: Union[tuple, list] = (7, 3, 3),
                 patch_stride: Union[tuple, list] = (4, 2, 2),
                 patch_padding: Union[tuple, list] = (2, 1, 1),
                 embed_dim: Union[tuple, list] = (64, 192, 384),
                 num_heads: Union[tuple, list] = (1, 3, 6),
                 depth: Union[tuple, list] = (1, 2, 10),
                 mlp_ratio: Union[tuple, list] = (4, 4, 4),
                 attn_drop_rate: Union[tuple, list] = (0.0, 0.0, 0.0),
                 drop_rate: Union[tuple, list] = (0.0, 0.0, 0.0),
                 drop_path_rate: Union[tuple, list] = (0.0, 0.0, 0.1),
                 qkv_bias: Union[tuple, list] = (True, True, True),
                 kernel_size: Union[tuple, list] = (3, 3, 3),
                 kv_stride: Union[tuple, list] = (2, 2, 2),
                 q_stride: Union[tuple, list] = (1, 1, 1),
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,):
        super(CvT, self).__init__()
        self.num_stages = num_stages

        self.blocks = nn.ModuleList([])
        for idx in range(self.num_stages):
            self.blocks.append(
                ViTBlock(
                    patch_size=patch_size[idx], patch_padding=patch_padding[idx],
                    patch_stride=patch_stride[idx], in_channels=in_channels, embed_dim=embed_dim[idx],
                    depth=depth[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratio[idx],
                    qkv_bias=qkv_bias[idx], drop_rate=drop_rate[idx], attn_drop_rate=attn_drop_rate[idx],
                    drop_path_rate=drop_path_rate[idx],  act_layer=act_layer,
                    norm_layer=norm_layer, last_stage=(idx == self.num_stages - 1),
                    kernel_size=kernel_size[idx], kv_stride=kv_stride[idx],
                    q_stride=q_stride[idx]
                )
            )
            in_channels = embed_dim[idx]

        dim_embed = embed_dim[-1]
        self.norm = norm_layer(dim_embed)

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.head.weight, std=.02)

    def forward(self, x):
        for block in self.blocks:
            x, cls_tokens = block(x)

        x = self.norm(cls_tokens)
        x = x.squeeze(1)

        x = self.head(x)

        return x


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    test_tensor = torch.rand(1, 3, 224, 224)
    model = CvT(num_classes=10)
    print(model(test_tensor).shape)
