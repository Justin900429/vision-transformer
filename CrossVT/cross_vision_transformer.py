from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from CrossVT import MultiScaleBlock
from layers import PatchEmbed


class CrossVisionTransformer(nn.Module):
    """Adapted from
    https://github.com/IBM/CrossViT/blob/3f7ab77c5728b1c31a2d5cc6185ee9c7e754d952/models/crossvit.py#L205-L311
    """
    def __init__(self, img_size: int = 224, patch_size: Union[tuple, list] = (8, 16),
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: Union[tuple, list] = (192, 384),
                 depth: Union[tuple, list] = ([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads: Union[tuple, list] = (6, 12),
                 mlp_ratio: Union[tuple, list] = (2., 2., 4.),
                 qkv_bias: bool = False, qk_scale: float = None,
                 drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 drop_path_rate: float = 0., multi_conv=False):
        super(CrossVisionTransformer, self).__init__()

        """Depth explain:
        1. ([], [], ..., []) -> len(depth): Number of Multi-stack Transformer block
        2. Depth: [0, ..., i, i + 1]:
            0 ~ i: Depth of each branch
            i + 1: Depth of the fuse
        """
        self.num_classes = num_classes
        self.img_size = img_size

        num_patches = [
            (img_size // p) * (img_size // p)
            for p in patch_size
        ]
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i]))
            for i in range(self.num_branches)
        ])
        for p, d in zip(patch_size, embed_dim):
            self.patch_embed.append(
                PatchEmbed(
                    img_size=img_size, patch_size=p,
                    in_channels=in_channels, embed_dim=d,
                    multi_conv=multi_conv
                )
            )

        self.cls_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim[i]))
            for i in range(self.num_branches)
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()

        # Numbers stack of Transformer Encoder
        for idx, block_depth in enumerate(depth):
            curr_depth = max(block_depth[:-1]) + block_depth[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(
                embed_dim=embed_dim, depth=block_depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_, norm_layer=norm_layer
            )
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([
            norm_layer(embed_dim[i])
            for i in range(self.num_branches)
        ])
        self.head = nn.ModuleList([
            nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
            for i in range(self.num_branches)
        ])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                nn.init.trunc_normal_(self.pos_embed[i].data, std=.02)
            nn.init.trunc_normal_(self.cls_token[i].data, std=.02)

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
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            # Resize the image if the size are different
            x_ = F.interpolate(x, size=(self.img_size, self.img_size), mode="bicubic") \
                if H != self.img_size else x

            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        # Assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        xs = [x[:, 0] for x in xs]

        # Mean up all the head at the final layer
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        return ce_logits


if __name__ == "__main__":
    test_tensor = torch.randn(1, 3, 224, 224)
    model = CrossVisionTransformer(num_classes=10)
    print(model(test_tensor).shape)
