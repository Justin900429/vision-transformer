from typing import Union
import torch
import torch.nn as nn
import einops
from utils import StochasticDepth
from layers import FeedForward


class CrossAttention(nn.Module):
    """Adapted from
    https://github.com/IBM/CrossViT/blob/3f7ab77c5728b1c31a2d5cc6185ee9c7e754d952/models/crossvit.py#L76-L105
    """
    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_scale: float = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of q: (batch_size, num_heads, 1, embed_dim)
        # Shape of k, v: (batch_size, num_heads, seq_len, head_dim)
        q = einops.rearrange(self.q_w(x[:, 0:1, :]), "b s (n h) -> b n s h", n=self.num_heads)
        k = einops.rearrange(self.k_w(x), "b s (n h) -> b n s h", n=self.num_heads)
        v = einops.rearrange(self.v_w(x), "b s (n h) -> b n s h", n=self.num_heads)

        # Shape of attn: (batch_size, 1, seq_len)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Shape of x: (batch_size, 1, embed_dim)
        x = einops.rearrange((attn @ v), "b n s h -> b s (n h)")
        x = self.proj(x)

        return x


class CrossAttentionBlock(nn.Module):
    """Adapted from
    https://github.com/IBM/CrossViT/blob/3f7ab77c5728b1c31a2d5cc6185ee9c7e754d952/models/crossvit.py#L108-L129
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4., qkv_bias: bool = False,
                 qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 has_mlp: bool = True):
        super(CrossAttentionBlock, self).__init__()
        self.norm_one = norm_layer(embed_dim)
        self.attn = CrossAttention(
            dim=embed_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = StochasticDepth(drop_path)
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm_two = norm_layer(embed_dim)
            self.mlp = FeedForward(
                in_features=embed_dim, factor=mlp_ratio,
                act_layer=act_layer, drop=drop
            )

    def forward(self, x):
        # Before: Shape of x: (batch_size, seq_len, embed_dim)
        # After: Shape of x: (batch_size, 1, embed_dim)
        x = x[:, 0:1, :] + self.drop_path(self.attn(self.norm_one(x)))

        # Shape of : (batch_size, 1, embed_dim)
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm_two(x)))

        return x


class MultiScaleBlock(nn.Module):
    """Adapted from
    https://github.com/IBM/CrossViT/blob/3f7ab77c5728b1c31a2d5cc6185ee9c7e754d952/models/crossvit.py#L132-L198
    """
    def __init__(self, embed_dim: Union[tuple, list], depth: Union[tuple, list],
                 num_heads: Union[tuple, list], mlp_ratio: Union[tuple, list],
                 qkv_bias: bool = False, qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: Union[tuple, list] = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        # The length of tuple depth will be: (num_branch + 1)
        # The last tuple is for the fusion blocks

        self.num_branches = len(embed_dim)

        # Create block for each branch
        self.blocks = nn.ModuleList()
        for d in range(self.num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    CrossAttentionBlock(
                        embed_dim=embed_dim[d], num_heads=num_heads[d],
                        mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[i],
                        norm_layer=norm_layer
                    )
                )

            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        # Class projection, (f(x) in paper)
        self.projs = nn.ModuleList()
        for d in range(self.num_branches):
            self.projs.append(nn.Sequential(
                norm_layer(embed_dim[d]),
                act_layer(),
                nn.Linear(embed_dim[d], embed_dim[(d + 1) % self.num_branches])
            ))

        self.fusion = nn.ModuleList()
        for d in range(self.num_branches):
            # Using (d + 1) here is because we change the embed_dim in above section (projs)
            d_ = (d + 1) % self.num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(
                        embed_dim=embed_dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                        attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                        has_mlp=False
                    )
                )
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(
                        CrossAttentionBlock(
                            embed_dim=embed_dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                            attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                            has_mlp=False
                        )
                    )
                self.fusion.append(nn.Sequential(*tmp))

        # Revert back to the original dimension
        self.revert_projs = nn.ModuleList()
        for d in range(self.num_branches):
            if embed_dim[(d + 1) % self.num_branches] == embed_dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(embed_dim[(d + 1) % self.num_branches]),
                    act_layer(),
                    nn.Linear(embed_dim[(d + 1) % self.num_branches], embed_dim[d])
                ]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]

        # Take the cls_token only, (f(x) in paper)
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]

        # Cross attention part
        outs = []
        for i in range(self.num_branches):
            # Concatenate with the next branch patch embedding
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)

            # Go through the attention block
            tmp = self.fusion[i](tmp)

            # Revert back the cls_token
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])

            # Concatenate with the original branch (g(x) in paper)
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)

        return outs
