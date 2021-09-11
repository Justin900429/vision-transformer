from typing import Union
import math
import torch
import torch.nn as nn
import einops.layers.torch
from LeViT import TransformerBlock


class LeViT(nn.Module):
    """Adapted from
    https://github.com/lucidrains/vit-pytorch/blob/529044c9b3f8981efe27b174123798de5274557c/vit_pytorch/levit.py#L127-L193
    https://github.com/facebookresearch/LeViT/blob/9d8eda22b6f6046bedd82d368d4282c82e5530ef/levit.py#L357-L494
    """
    def __init__(self, image_size: int = 224, num_classes: int = 1000,
                 dim_key: int = 32, dim_value: int = 64,
                 dropout: float = 0., distill: bool = True,
                 factors: Union[tuple, list] = (2, 2, 2),
                 embed_dims: Union[tuple, list] = (192, 64, 32),
                 depths: Union[tuple, list] = (12, 6, 3),
                 num_heads_list: Union[tuple, list] = (3, 3, 3),
                 ):
        super(LeViT, self).__init__()

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, embed_dims[0], kernel_size=3, stride=2, padding=1)
        )

        fmap_size = image_size // (2 ** 4)
        layers = []

        stages = len(embed_dims)
        for idx, factor, embed_dim, depth, num_heads in zip(range(stages), factors, embed_dims, depths, num_heads_list):
            is_last = idx == (stages - 1)
            layers.append(
                TransformerBlock(
                    embed_dim=embed_dim, fmap_size=fmap_size,
                    depth=depth, num_heads=num_heads, dim_key=dim_key,
                    dim_value=dim_value, factor=factor, dropout=dropout
                )
            )

            if not is_last:
                next_dim = embed_dims[idx + 1]
                layers.append(
                    TransformerBlock(
                        embed_dim=embed_dim, fmap_size=fmap_size,
                        depth=1, num_heads=num_heads * 2, dim_key=dim_key,
                        dim_value=dim_value, embed_out=next_dim, downsample=True
                    )
                )
                fmap_size = math.ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            einops.layers.torch.Rearrange('... () () -> ...')
        )

        self.distill = distill
        self.distill_head = nn.Linear(embed_dims[-1], num_classes) if distill is not None else None
        self.mlp_head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, img):
        x = self.conv_embedding(img)

        x = self.backbone(x)

        x = self.pool(x)

        out = self.mlp_head(x)

        if self.distill is not None:
            distill = self.distill_head(x)
            return out, distill

        return out


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    test_tensor = torch.rand(1, 3, 224, 224)
    model = LeViT(num_classes=10)
    out, distill = model(test_tensor)
    print(out.shape, distill.shape)
