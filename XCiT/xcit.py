from functools import partial
import torch
import torch.nn as nn
from XCiT import ConvPatchEmbed
from XCiT import ClassAttentionBlock
from XCiT import XCBlock
from XCiT import PositionalEncodingFourier


class XCiT(nn.Module):
    """Adapted from
    https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py#L295-L407
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 cls_attn_layers=2, use_pos=True, eta=None, tokens_norm=False):
        super(XCiT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ConvPatchEmbed(img_size=img_size, embed_dim=embed_dim,
                                          in_channels=in_channels, patch_size=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_rate_list = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList([
            XCBlock(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate_list[i],
                    norm_layer=norm_layer, eta=eta)
            for i in range(depth)
        ])

        self.cls_attn_blocks = nn.Sequential(*[
            ClassAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm)
            for _ in range(cls_attn_layers)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.use_pos = use_pos
        if self.use_pos:
            self.pos_embedder = PositionalEncodingFourier(dim=embed_dim)

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
        elif isinstance(layer, nn.Parameter):
            nn.init.trunc_normal_(layer, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # Shape of x: (batch_size, seq_len, embed_dim)
        x, (height, width) = self.patch_embed(x)

        if self.use_pos:
            pos_encoding = self.pos_embedder(B, height, width).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)

        # Shape of x: (batch_size, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x, height, width)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.cls_attn_blocks(x)
        x = self.norm(x)[:, 0]
        x = self.head(x)

        # Return two tensor for cls token and knowledge distillation
        if self.training:
            return x, x
        else:
            return x


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
    test_tensor = torch.randn(1, 3, 224, 224)
    model = XCiT(depth=1, cls_attn_layers=1, num_classes=10, eta=1e-5)
    cls, dis = model(test_tensor)
    print(cls.shape, dis.shape)

    model.eval()
    print(model(test_tensor).shape)
