import math
import torch
import torch.nn as nn
import einops


class Attention(nn.Module):
    """Adapted from
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim: int, num_heads: int, in_dim: int = None,
                 qkv_bias: bool = True, qk_scale: float = None,
                 attn_drop: float = 0., proj_drop: float = 0.,
                 residual_before: bool = False):
        super(Attention, self).__init__()

        self.residual_before = residual_before
        self.num_heads = num_heads

        assert (dim % num_heads == 0), "Argument `dim` should be factor of argument `num_heads"
        if in_dim is None:
            in_dim = dim
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        self.q_w = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.k_w = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.v_w = nn.Linear(dim, in_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x: torch.tensor) \
            -> torch.tensor:
        # Make the dim head
        # Shape of q: (batch_size, num_heads, q_seq_length, head_dim)
        # Shape of k: (batch_size, num_heads, k_seq_length, head_dim)
        # Shape of v: (batch_size, num_heads, v_seq_length, head_dim)
        # NOTE: k_seq_length == v_seq_length
        q = einops.rearrange(self.q_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        k = einops.rearrange(self.k_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        v = einops.rearrange(self.v_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)

        # Compute the attention energy
        # Shape of attn: (batch_size, num_heads, q_seq_length, k_seq_length)
        attn = torch.einsum("bnqd,bnkd->bnqk", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute the final weight on value
        # Shape of x: (batch_size, q_seq_length, in_dim)
        x = torch.einsum("bnqk,bnkd->bnqd", attn, v)
        x = einops.rearrange(x, "b n q d -> b q (n d)")
        x = self.proj(x)

        if self.residual_before:
            v = einops.rearrange(v, "b n v d -> b v (n d)")
            x = x + v

        return x


class PerformerAttention(nn.Module):
    """Adapted from
     https://github.com/yitu-opensource/T2T-ViT/blob/88afb7cf30b603703c02b6e29c06be23c49cf6a2/models/token_performer.py#L8-L59
    """
    def __init__(self, dim: int, in_dim: int = None,
                 qkv_bias: bool = True, kernel_ratio: float = 0.5,
                 attn_drop: float = 0.1, drop_rate: float = 0.1):
        super(PerformerAttention, self).__init__()
        if in_dim is None:
            in_dim = dim

        self.q_w = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.k_w = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.v_w = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.epsilon = 1e-8

        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(drop_rate)
        )
        self.norm = nn.LayerNorm(dim)

        self.m = int(in_dim * kernel_ratio)
        self.w = torch.randn(self.m, in_dim)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m),
                              requires_grad=False)

    def prm_exp(self, x):
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of xd: (batch_size, seq_len, int(in_dim * kernel_ratio))
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2

        # Shape of wtx: (batch_size, seq_len, int(in_dim * kernel_ratio))
        wtx = torch.einsum("bse,me->bsm", x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, dim)
        x = self.norm(x)

        # Shape of q: (batch_size, q_seq_length, in_dim)
        # Shape of k: (batch_size, k_seq_length, in_dim)
        # Shape of v: (batch_size, v_seq_length, in_dim)
        # NOTE: k_seq_length == v_seq_length
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)

        # Shape of kq, qp: (batch_size, seq_len, int(in_dim * kernel_ratio))
        kp, qp = self.prm_exp(k), self.prm_exp(q)

        # Shape of D: (batch_size, seq_len, 1)
        D = torch.einsum('bsm,bm->bs', qp, kp.sum(dim=1)).unsqueeze(dim=2)

        # Shape of kptv: (batch_size, in_dim, int(in_dim * kernel_ratio))
        kptv = torch.einsum("bse,bsm->bem", v.float(), kp)
        kptv = self.attn_drop(kptv)

        # Shape of y: (batch_size, seq_len, in_dim)
        in_dim = kptv.shape[1]
        y = torch.einsum("bsm,bem->bse", qp, kptv) / (D.repeat(1, 1, in_dim) + self.epsilon)
        y = v + self.proj(y)

        return y


if __name__ == "__main__":
    test_tensor = torch.rand(1, 16, 128)
    model = PerformerAttention(dim=128)
    print(model(test_tensor).shape)
