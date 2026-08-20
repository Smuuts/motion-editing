"""
The building blocks of the GroupCLR temporal U-Net.

Everything here operates on the FRAME axis only — the body-part group axis is carried
through untouched, which is what `to_conv` / `from_conv` fold in and out of the batch
dimension. The CLR block's module order is MotionCLR's: temporal Conv1d (the timestep is
injected here via adaGN, and ONLY here — attention and the feed-forward are timestep-free,
unlike the DiT's adaLN-zero), then self-attention, cross-attention and the feed-forward.
Those three are reused verbatim from model/layers.py, so attn_sink, ctx_pad_mask,
store_attn, the entropy regulariser and the grounding hook behave exactly as in the DiT.
"""

import torch
import torch.nn as nn

from model.layers import CrossAttention, FeedForward, SelfAttention

def pick_groupnorm(out_ch: int, preferred: int = 8) -> int:
    """Largest divisor of out_ch that is <= preferred (>=1). GroupNorm requires
    out_ch % n_groups == 0; latent_dim is not guaranteed divisible by 8."""
    for g in range(min(preferred, out_ch), 0, -1):
        if out_ch % g == 0:
            return g
    return 1


def to_conv(x: torch.Tensor, G: int) -> torch.Tensor:
    """Attention layout (B, F', G, C) → conv layout (B·G, C, F') for per-group
    temporal convs (the frame axis is the last dim conv1d wants)."""
    B, Fp, _, C = x.shape
    return x.permute(0, 2, 3, 1).reshape(B * G, C, Fp)


def from_conv(xr: torch.Tensor, B: int, G: int) -> torch.Tensor:
    """Conv layout (B·G, C, F') → attention layout (B, F', G, C)."""
    _, C, Fp = xr.shape
    return xr.reshape(B, G, C, Fp).permute(0, 3, 1, 2)


class TemporalConvBlock(nn.Module):
    """MotionCLR ResidualTemporalBlock: Conv1d(k) → GroupNorm → adaGN(t) → Mish,
    second (zero-init) Conv1d → GroupNorm → Mish, plus a residual. Operates on
    conv layout (B·G, C, F'); timestep is injected here and nowhere else."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        ng = pick_groupnorm(out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(ng, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.norm2 = nn.GroupNorm(ng, out_ch)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        # adaGN scale/shift from the timestep embedding (per (B·G) sample).
        self.time_mlp = nn.Linear(t_dim, out_ch * 2)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Zero-init the second conv so each block starts as (≈identity residual) +
        # attention — the U-Net analogue of GroupDiT's adaLN-zero calm start.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, G: int) -> torch.Tensor:
        # x: (B·G, in_ch, F'); t_emb: (B, t_dim). The timestep is group-independent, so
        # the adaGN Linear runs once on B rows and its output is broadcast across the G
        # groups (rows are b-major/g-minor: b*G+g) — avoids a G× redundant GEMM.
        h = self.norm1(self.conv1(x))
        ss = self.time_mlp(t_emb)                                   # (B, 2·out_ch)
        ss = ss[:, None, :].expand(-1, G, -1).reshape(x.shape[0], -1)  # (B·G, 2·out_ch)
        scale, shift = ss[:, :, None].chunk(2, dim=1)              # (B·G, out_ch, 1)
        h = self.act(h * (1 + scale) + shift)
        h = self.act(self.norm2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.res(x)


class _FrameResample(nn.Module):
    """Resample the frame axis with a per-group 1-D conv (subclasses set self.conv)."""

    def forward(self, x: torch.Tensor, G: int) -> torch.Tensor:
        B = x.shape[0]
        return from_conv(self.conv(to_conv(x, G)), B, G)


class Downsample(_FrameResample):
    """Strided conv along the frame axis (÷2), applied per group."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)


class Upsample(_FrameResample):
    """Transposed conv along the frame axis (×2), applied per group."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)


class CLRBlock(nn.Module):
    """conv (temporal, adaGN t) → self-attn → cross-attn → FFN, each attention/FFN
    sublayer a pre-norm residual. Reuses model/layers.py attention verbatim, so
    attn_sink / ctx_pad_mask / store_attn / compute_entropy all work unchanged.

    I/O in attention-layout-friendly shape (B, F', G, C_out). `in_ch` may differ
    from `out_ch` only for the first up-block after a skip-concat (2C → C)."""

    def __init__(self, in_ch: int, out_ch: int, context_dim: int, t_dim: int,
                 num_heads: int, ff_mult: int, dropout: float, attn_sink: bool):
        super().__init__()
        self.conv = TemporalConvBlock(in_ch, out_ch, t_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(out_ch)
        self.self_attn = SelfAttention(out_ch, num_heads, dropout)
        self.norm2 = nn.LayerNorm(out_ch)
        self.cross_attn = CrossAttention(out_ch, context_dim, num_heads, dropout,
                                         use_sink=attn_sink)
        self.norm3 = nn.LayerNorm(out_ch)
        self.ff = FeedForward(out_ch, ff_mult, dropout)

    def forward(self, x, t_emb, context, G,
                self_mask=None, store_attn=False, context_mask=None,
                compute_entropy=False, supervise=False):
        B, Fp, _, _ = x.shape

        # ── temporal conv (B·G, C, F'), timestep injected here only ─────────
        x = from_conv(self.conv(to_conv(x, G), t_emb, G), B, G)  # (B, F', G, Cout)
        Cout = x.shape[-1]

        # ── attention layout (B, F'·G, C) ───────────────────────────────────
        h = x.reshape(B, Fp * G, Cout)
        h = h + self.self_attn(self.norm1(h), mask=self_mask)
        h = h + self.cross_attn(self.norm2(h), context, store_attn=store_attn,
                                context_mask=context_mask,
                                compute_entropy=compute_entropy,
                                supervise=supervise)
        h = h + self.ff(self.norm3(h))
        return h.reshape(B, Fp, G, Cout)
