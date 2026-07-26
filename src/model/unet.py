"""
GroupCLR: a MotionCLR-style 1-D temporal U-Net backbone for 3D human motion,
kept a drop-in alternative to GroupDiT (model/dit.py).

Design (see docs/MotionCLR_UNet_Research.md, docs/ARCHITECTURE.md):
  - Same body-part-grouped tokenisation as GroupDiT: each frame → 7 group tokens
    (root, left/right leg, spine, left/right arm, head). The group axis (G=7) is
    preserved through the whole U-Net; only the FRAME axis is down/up-sampled.
  - MotionCLR "CLR block" order: temporal Conv1d (timestep injected here via adaGN,
    and ONLY here — attention/FFN are timestep-free, unlike GroupDiT's adaLN-zero) →
    self-attention → cross-attention → FFN. The attention/FFN modules are reused
    verbatim from model/layers.py, so attn_sink, ctx_pad_mask, store_attn and the
    entropy regulariser all behave exactly as in the DiT.
  - Epsilon prediction (NOT MotionCLR's native x0/"sample"), so NoiseSchedule,
    Min-SNR, x0_confidence_weight, the geometric/FK losses, the sampler and the full
    LEDITS++ inversion/editing stack are reused with zero changes.

Interface contract honoured (identical to GroupDiT):
  forward(motion (B,F,D), t (B,), context (B,L,C)|None,
          store_attn=False, mask=(B,F)|None, entropy_layer=int|None) -> eps (B,F,D)
  attributes: input_dim, group_channels, latent_dim, null_text_emb, blocks
  methods:    get_attn_maps(), get_attn_entropy(idx)

The one U-Net-specific wrinkle: cross-attention lives at multiple frame resolutions,
so each block's stored map has a different (F'·G) token count. get_attn_maps()
temporally upsamples every stored map back to the original F before returning, giving
the uniform (B, heads, F·G, L) list that editing/masking.py expects (LEDITS++ does the
same multi-resolution map aggregation on SD's U-Net).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import (
    TimestepEmbedding, SelfAttention, CrossAttention, FeedForward,
    resolve_context_and_mask,
)
from model.body_groups import (
    group_layout as _group_layout,
)


def _pick_groupnorm(out_ch: int, preferred: int = 8) -> int:
    """Largest divisor of out_ch that is <= preferred (>=1). GroupNorm requires
    out_ch % n_groups == 0; latent_dim is not guaranteed divisible by 8."""
    for g in range(min(preferred, out_ch), 0, -1):
        if out_ch % g == 0:
            return g
    return 1


def _to_conv(x: torch.Tensor, G: int) -> torch.Tensor:
    """Attention layout (B, F', G, C) → conv layout (B·G, C, F') for per-group
    temporal convs (the frame axis is the last dim conv1d wants)."""
    B, Fp, _, C = x.shape
    return x.permute(0, 2, 3, 1).reshape(B * G, C, Fp)


def _from_conv(xr: torch.Tensor, B: int, G: int) -> torch.Tensor:
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
        ng = _pick_groupnorm(out_ch)
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
        return _from_conv(self.conv(_to_conv(x, G)), B, G)


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
                compute_entropy=False):
        B, Fp, _, _ = x.shape

        # ── temporal conv (B·G, C, F'), timestep injected here only ─────────
        x = _from_conv(self.conv(_to_conv(x, G), t_emb, G), B, G)  # (B, F', G, Cout)
        Cout = x.shape[-1]

        # ── attention layout (B, F'·G, C) ───────────────────────────────────
        h = x.reshape(B, Fp * G, Cout)
        h = h + self.self_attn(self.norm1(h), mask=self_mask)
        h = h + self.cross_attn(self.norm2(h), context, store_attn=store_attn,
                                context_mask=context_mask,
                                compute_entropy=compute_entropy)
        h = h + self.ff(self.norm3(h))
        return h.reshape(B, Fp, G, Cout)


class GroupMotionUNet(nn.Module):
    """MotionCLR-style temporal U-Net over body-part group tokens."""

    def __init__(
        self,
        feature_mode: str = "humanml3d",
        latent_dim:   int = 512,
        context_dim:  int = 512,
        num_heads:    int = 8,
        num_layers:   int = 8,
        max_frames:   int = 196,
        ff_mult:      int = 4,
        dropout:      float = 0.1,
        text_seq_len: int = 77,
        ctx_pad_mask: bool = False,
        attn_sink:    bool = False,
        group_mode:   str = "parts",
        unet_levels:  int = 3,
        unet_blocks_per_level: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ctx_pad_mask = ctx_pad_mask
        self.levels = unet_levels

        # ── group tokeniser / detokeniser (identical to GroupDiT) ────────────
        # group_mode picks the token axis: 'parts' (7) or 'joints' (22). G is derived
        # from the partition, never hardcoded — so per-joint checkpoints tokenise right.
        self.group_mode = group_mode
        self.group_channels, group_dims, self.input_dim = _group_layout(feature_mode, group_mode)
        self.G = len(self.group_channels)
        self._group_dims = group_dims
        self.in_projs  = nn.ModuleList([nn.Linear(d, latent_dim) for d in group_dims])
        self.out_projs = nn.ModuleList([nn.Linear(latent_dim, d) for d in group_dims])
        self.group_emb = nn.Embedding(self.G, latent_dim)

        perm = torch.cat([torch.as_tensor(ch, dtype=torch.long) for ch in self.group_channels])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel())
        self.register_buffer("_group_idx", torch.arange(self.G), persistent=False)
        self.register_buffer("_perm_idx", perm, persistent=False)
        self.register_buffer("_inv_perm_idx", inv_perm, persistent=False)

        self.time_emb = TimestepEmbedding(latent_dim)
        # No explicit frame positional embedding: the temporal convs encode frame
        # position (MotionCLR-faithful). group_emb still distinguishes the G tokens
        # (7 parts or 22 joints) for the (shared-weight) attention.

        C = latent_dim
        blk = lambda in_ch, out_ch: CLRBlock(
            in_ch, out_ch, context_dim, latent_dim, num_heads, ff_mult, dropout, attn_sink)

        # encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for _ in range(unet_levels):
            self.down_blocks.append(nn.ModuleList(
                [blk(C, C) for _ in range(unet_blocks_per_level)]))
            self.down_samples.append(Downsample(C))

        # bottleneck
        self.mid_blocks = nn.ModuleList([blk(C, C), blk(C, C)])

        # decoder (skip-concat on the first block of each level: 2C → C)
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(unet_levels):
            self.up_samples.append(Upsample(C))
            self.up_blocks.append(nn.ModuleList(
                [blk(2 * C if b == 0 else C, C) for b in range(unet_blocks_per_level)]))

        self.final_norm = nn.LayerNorm(C)

        # Flat execution-order view of every CLRBlock for get_attn_maps /
        # get_attn_entropy / epoch.py's len(model.blocks). A plain Python list (NOT
        # an nn.ModuleList) so the blocks aren't double-registered — they're already
        # owned by the structured ModuleLists above. deepcopy (EMA) preserves the
        # aliasing via its memo, so the flat view keeps pointing at the copies.
        self.blocks = (
            [b for lvl in self.down_blocks for b in lvl]
            + list(self.mid_blocks)
            + [b for lvl in self.up_blocks for b in lvl]
        )

        self.null_text_emb = nn.Parameter(torch.randn(1, text_seq_len, context_dim) * 0.02)
        self._last_orig_F = max_frames
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # TemporalConvBlock already zero-inits its second conv in __init__.

    @staticmethod
    def _expand_mask(m, G):
        """(B, F') per-frame bool → (B, F'·G) per-token, or None."""
        if m is None:
            return None
        B, Fp = m.shape
        return m[:, :, None].expand(B, Fp, G).reshape(B, Fp * G)

    @staticmethod
    def _downsample_mask(m):
        """(B, F') bool → (B, ceil(F'/2)) bool: a coarse frame is valid if ANY of its
        children is (matches the stride-2 conv's receptive coverage)."""
        if m is None:
            return None
        pooled = F.max_pool1d(m.float()[:, None], kernel_size=2, stride=2, ceil_mode=True)
        return pooled[:, 0] > 0.5

    # ── attention-map / entropy accessors (interface contract) ───────────────
    def get_attn_maps(self) -> list[torch.Tensor]:
        """Per-block cross-attention maps, each temporally upsampled from its own
        F' back to the original F, so all are (B, heads, F·G, L_text) frame-major —
        exactly what editing/masking.py expects. Empty if no forward stored maps."""
        G, Fo = self.G, self._last_orig_F
        out = []
        for b in self.blocks:
            m = b.cross_attn.last_attn_map
            if m is None:
                continue
            B, h, NG, L = m.shape
            Fp = NG // G
            if Fp == Fo:
                out.append(m)
                continue
            # (B,h,Fp*G,L) → interpolate frame axis Fp→Fo
            mm = m.reshape(B, h, Fp, G, L).permute(0, 1, 3, 4, 2).reshape(-1, 1, Fp)
            mm = F.interpolate(mm, size=Fo, mode="linear", align_corners=False)
            mm = mm.reshape(B, h, G, L, Fo).permute(0, 1, 4, 2, 3).reshape(B, h, Fo * G, L)
            out.append(mm)
        return out

    def get_attn_entropy(self, block_idx: int) -> torch.Tensor:
        return self.blocks[block_idx].cross_attn.last_entropy

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, input_dim)
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,   # (B, F) per-frame
        entropy_layer: int | None = None,
    ) -> torch.Tensor:
        B, Fo, _ = motion.shape
        G = self.G
        self._last_orig_F = Fo
        context, ctx_mask = resolve_context_and_mask(
            context, B, self.null_text_emb, self.ctx_pad_mask)

        # ── tokenise → (B, F, G, C) ──────────────────────────────────────────
        group_feats = torch.split(
            motion.index_select(-1, self._perm_idx), self._group_dims, dim=-1)
        x = torch.stack([proj(f) for proj, f in zip(self.in_projs, group_feats)], dim=2)
        x = x + self.group_emb(self._group_idx)[None, None]

        # ── pad frames to a multiple of 2^levels so every ÷2/×2 is exact ─────
        Fpad = math.ceil(Fo / (1 << self.levels)) * (1 << self.levels)
        if Fpad != Fo:
            x = F.pad(x, (0, 0, 0, 0, 0, Fpad - Fo))       # pad frame axis (dim=1)
            if mask is not None:
                mask = F.pad(mask, (0, Fpad - Fo), value=False)

        t_emb = self.time_emb(t)                           # (B, C) — group-independent

        # Per-token self-attention masks, expanded once per resolution (all blocks at a
        # level share it): index l = down level l; index `levels` = mid.
        tok_masks_at = [self._expand_mask(mask, G)]
        cur = mask
        for _ in range(self.levels):
            cur = self._downsample_mask(cur)
            tok_masks_at.append(self._expand_mask(cur, G))

        gidx = 0

        def run(block, tok_mask):
            nonlocal x, gidx
            x = block(x, t_emb, context, G,
                      self_mask=tok_mask,
                      store_attn=store_attn, context_mask=ctx_mask,
                      compute_entropy=(gidx == entropy_layer))
            gidx += 1

        # encoder
        skips = []
        for lvl in range(self.levels):
            for block in self.down_blocks[lvl]:
                run(block, tok_masks_at[lvl])
            skips.append(x)
            x = self.down_samples[lvl](x, G)

        # bottleneck
        for block in self.mid_blocks:
            run(block, tok_masks_at[self.levels])

        # decoder
        for k in range(self.levels):
            x = self.up_samples[k](x, G)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)               # (B, F', G, 2C)
            tok_mask = tok_masks_at[self.levels - 1 - k]
            for block in self.up_blocks[k]:
                run(block, tok_mask)

        # ── detokenise → (B, F, input_dim) epsilon ──────────────────────────
        x = self.final_norm(x)[:, :Fo]                      # crop padding
        group_outs = [self.out_projs[g](x[:, :, g]) for g in range(G)]
        return torch.cat(group_outs, dim=-1).index_select(-1, self._inv_perm_idx)
