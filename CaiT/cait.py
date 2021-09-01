import torch
import torch.nn as nn
from CaiT import LayerScaleBlock, AttnTalkingHead
from CaiT import LayerCABlock, ClassAttention
from layers import PatchEmbed, FeedForward


class CaiT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: int = 4, qkv_bias: bool = False,
                 qk_scale: float = None, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 block_layers: nn.Module = LayerScaleBlock,
                 block_layers_token: nn.Module = LayerCABlock,
                 patch_layer: nn.Module = PatchEmbed, act_layer: nn.Module = nn.GELU,
                 attention_block: nn.Module = AttnTalkingHead, mlp_block: nn.Module = FeedForward,
                 attention_block_token_only: nn.Module = ClassAttention,
                 mlp_block_token_only: nn.Module = FeedForward,
                 init_scale: float = 1e-4, depth_token_only: int = 2,
                 mlp_ratio_cls: int = 4):
        super(CaiT, self).__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = patch_layer(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[idx],
                norm_layer=norm_layer, act_layer=act_layer, attention_block=attention_block,
                feed_forward_block=mlp_block, init_values=init_scale)
            for idx in range(depth)]
        )

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_cls, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, attention_block=attention_block_token_only,
                feed_forward_block=mlp_block_token_only, init_values=init_scale)
            for _ in range(depth_token_only)]
        )

        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token.data, std=.02)
        nn.init.trunc_normal_(self.pos_embed.data, std=.02)
        self.init_weights(self.init_weights)

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
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        for block in self.blocks_token_only:
            cls_tokens = block(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)

        x = x[:, 0]
        x = self.head(x)

        return x


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = CaiT(num_classes=10, depth=1, depth_token_only=1)
    print(model(test_tensor).shape)
