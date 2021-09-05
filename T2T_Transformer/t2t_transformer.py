import torch
import torch.nn as nn
import einops
from T2T_Transformer import T2TModule
from layers import SinusoidalEmbedding, TransformerEncoder


class T2TViT(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: int = 4, qkv_bias: bool = False,
                 qk_scale: float = None, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 token_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        self.tokens_to_token = T2TModule(
            img_size=img_size, in_channels=in_channels,
            embed_dim=embed_dim, token_dim=token_dim
        )
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinusoidalEmbedding(embed_dim=embed_dim, seq_len=num_patches + 1)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                dim=embed_dim, num_heads=num_heads, factor=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, stochastic_drop_prob=dpr[idx],
                norm_layer=norm_layer
            )
            for idx in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=.02)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embed(x)
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = T2TViT(num_classes=10)
    print(model(test_tensor).shape)
