"""
Generic transformer building blocks shared by MotionDiT and GroupDiT (model/dit.py):
timestep/positional embeddings, self- and cross-attention, the feed-forward block,
and the adaLN-zero DiT block that combines them. None of this is motion-specific —
it would look the same in an image DiT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class FramePositionalEmbedding(nn.Module):
    def __init__(self, max_frames: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_frames, dim)

    def forward(self, num_frames: int, device) -> torch.Tensor:
        return self.emb(torch.arange(num_frames, device=device))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q   = nn.Linear(dim, dim, bias=False)
        self.k   = nn.Linear(context_dim, dim, bias=False)
        self.v   = nn.Linear(context_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Shape when stored: (B, heads, N_motion, L_text).
        # N_motion = F for MotionDiT, F*G for GroupDiT (G=7 body-part groups).
        # For Stage 2 mask M1: accumulate these across ALL inversion timesteps and layers,
        # then average over heads, timesteps, and layers before thresholding.
        self.last_attn_map = None

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                store_attn: bool = False) -> torch.Tensor:
        # store_attn=True is passed during Stage 1 inversion and Stage 3 denoising
        # to collect A^{t,l} maps. Must be False during training to avoid memory growth.
        B, N, _ = x.shape
        _, L, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if store_attn:
            # Materialize for attention map capture (inference only — not training).
            attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
            attn = self.dropout(attn)
            self.last_attn_map = attn.detach()
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        else:
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            out = out.transpose(1, 2).reshape(B, N, -1)

        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out  = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # (B, 1, 1, N) bool mask: False positions are padding and get -inf
            attn_mask = mask[:, None, None, :]

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, -1)

        if mask is not None:
            out = out.masked_fill(~mask[:, :, None], 0.0)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.self_attn  = SelfAttention(dim, num_heads, dropout)
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
        self.norm3      = nn.LayerNorm(dim)
        self.ff         = FeedForward(dim, ff_mult, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, store_attn: bool = False,
                mask: torch.Tensor | None = None):
        mods = self.adaLN_modulation(t_emb)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = mods.chunk(9, dim=-1)
        s1, b1, g1 = s1[:, None], b1[:, None], g1[:, None]
        s2, b2, g2 = s2[:, None], b2[:, None], g2[:, None]
        s3, b3, g3 = s3[:, None], b3[:, None], g3[:, None]

        # adaLN-zero: gate (g) is zero-initialised so each block is an identity
        # map at the start of training (DiT §3.2).
        x = x + g1 * self.self_attn(self.norm1(x) * (1 + s1) + b1, mask=mask)
        x = x + g2 * self.cross_attn(self.norm2(x) * (1 + s2) + b2, context, store_attn=store_attn)
        x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        return x
