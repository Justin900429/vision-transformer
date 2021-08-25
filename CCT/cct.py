import torch
import torch.nn as nn
import torch.nn.functional as F
from CCT import ConvolutionTokenizer
from layers import (
    TransformerEncoder,
    SinusoidalEmbedding
)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 num_classes=1000,
                 drop_rate=0.1,
                 attn_drop=0.1,
                 stochastic_depth_rate=0.1,
                 embed_type='sine',
                 seq_len=None):
        super(TransformerClassifier, self).__init__()

        assert seq_len is not None, "Sequence length should be given"
        assert embed_type in ["sine", "learned", "none"], \
            f"Not supported mode for pos_embed. Should be ['none', 'learnable', 'sine']"

        self.seq_pool = seq_pool
        self.seq_len = seq_len

        # Add sequence pooling or use the vanilla class embedding
        if self.seq_pool:
            self.attention_pool = nn.Linear(
                in_features=embed_dim,
                out_features=1
            )
        else:
            self.class_embed = nn.Parameter(
                torch.zeros(1, 1, self.embedding_dim),
                requires_grad=True
            )

        # Add positional embedding or not
        if embed_type == "none":
            self.pos_embed = None
        elif embed_type == "learnable":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, seq_len, embed_dim),
                requires_grad=True
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.2)
        else:
            self.pos_embed = SinusoidalEmbedding(embed_dim=embed_dim, seq_len=seq_len)

        self.dropout = nn.Dropout(drop_rate)

        stochastic_drop_rate_list = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.transformers_blocks = nn.Sequential(
            *[
                TransformerEncoder(
                    dim=embed_dim,
                    num_heads=num_heads,
                    factor=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    stochastic_drop_prob=stochastic_drop_rate_list[idx],
                )
                for idx in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        # Optional position embedding
        if self.pos_embed is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        # Add class embedding if doesn't use sequence pooling
        if not self.seq_pool:
            cls_token = self.class_embed.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add embedding
        if isinstance(self.pos_embed, nn.Module):
            x = self.pos_embed(x)
        elif self.pos_embed is not None:
            x += self.pos_embed
        x = self.dropout(x)

        # Transformers blocks
        x = self.transformers_blocks(x)
        x = self.norm(x)

        # Take out the features using sequence pooling or class embedding
        if self.seq_pool:
            # Before: Shape of x: (batch_size, seq_len, embed_dim)
            # After: Shape of x: (batch_size, 1, seq_len)
            pool_attn = F.softmax(self.attention_pool(x), dim=-1).transpose(1, 2)

            # Shape of x: (batch_size, 1, embed_dim)
            x = pool_attn @ x
            # Shape of x: (batch_size, embed_dim)
            x = x.squeeze(1)
        else:
            x = x[:, 0]

        x = self.classifier(x)
        return x

    @staticmethod
    def init_weight(layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=.02)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embed_dim=768,
                 in_channels=3,
                 num_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 seq_pool=True,
                 embed_type="sine"):
        super(CCT, self).__init__()

        self.conv_tokenizer = ConvolutionTokenizer(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=embed_dim,
            activation=nn.ReLU,
            conv_bias=False
        )

        self.classifier = TransformerClassifier(
            seq_len=self.conv_tokenizer.sequence_length(n_channels=in_channels,
                                                        height=img_size,
                                                        width=img_size),
            embed_dim=embed_dim,
            seq_pool=seq_pool,
            drop_rate=0.,
            attn_drop=0.1,
            stochastic_depth_rate=0.1,
            embed_type=embed_type
        )

    def forward(self, x):
        x = self.conv_tokenizer(x)
        return self.classifier(x)


if __name__ == "__main__":
    test_tensor = torch.rand(1, 3, 224, 224)
    model = CCT()
    print(model(test_tensor).shape)
