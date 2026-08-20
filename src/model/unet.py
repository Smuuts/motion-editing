"""
GroupCLR: a MotionCLR-style 1-D temporal U-Net backbone, a drop-in alternative to
GroupDiT (model/dit.py).

Same body-part-grouped tokenisation as GroupDiT — each frame becomes G group tokens —
and the group axis is preserved through the whole U-Net; only the FRAME axis is
down/up-sampled. Epsilon prediction, NOT MotionCLR's native x0, so the NoiseSchedule,
Min-SNR weighting, the geometric losses, the sampler and the whole LEDITS++
inversion/editing stack are reused with zero changes. Blocks live in unet_blocks.py.

Interface contract, identical to GroupDiT:
  forward(motion (B,F,D), t (B,), context (B,L,C)|None,
          store_attn=False, mask=(B,F)|None, entropy_layer=int|None,
          supervise_layer=int|None) -> eps (B,F,D)
  attributes: input_dim, group_channels, latent_dim, null_text_emb, blocks
  methods:    get_attn_maps(), get_attn_entropy(idx), get_sup_attn(idx)

The one U-Net-specific wrinkle: cross-attention lives at multiple frame resolutions, so
each block's stored map has a different (F'*G) token count. `get_attn_maps` temporally
upsamples every stored map back to the original F before returning, giving the uniform
(B, heads, F*G, L) list the masking code expects — the same multi-resolution aggregation
LEDITS++ does on Stable Diffusion's U-Net.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.body_groups import group_layout as _group_layout
from model.layers import TimestepEmbedding, resolve_context_and_mask
from model.unet_blocks import CLRBlock, Downsample, Upsample

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

    def get_sup_attn(self, block_idx: int) -> torch.Tensor:
        """Head-averaged cross-attention (B, F, G, L_text) with graph, for the
        grounding loss — the GroupDiT.get_sup_attn contract (model/dit.py).

        A block below the top resolution attends over F' < F frames, so its map is
        interpolated back to the original F exactly as get_attn_maps does. That keeps
        the loss's (frame, group) grid the same object the caller's frame mask and
        source-activity reference are indexed on. The interpolation is differentiable,
        so gradient still reaches the coarse block — but the labels' frame axis is
        smeared by the resample, which is one reason run 1 uses arch=dit.

        Reading it clears it, same contract (and same reason) as GroupDiT.get_sup_attn.
        """
        cross = self.blocks[block_idx].cross_attn
        attn, cross.last_sup_attn = cross.last_sup_attn, None       # (B, F'·G, L)
        if attn is None:
            return None
        G, Fo = self.G, self._last_orig_F
        B, NG, L = attn.shape
        Fp = NG // G
        a = attn.reshape(B, Fp, G, L)
        if Fp == Fo:
            return a
        a = a.permute(0, 2, 3, 1).reshape(-1, 1, Fp)                # (B·G·L, 1, F')
        a = F.interpolate(a, size=Fo, mode="linear", align_corners=False)
        return a.reshape(B, G, L, Fo).permute(0, 3, 1, 2)           # (B, F, G, L)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, input_dim)
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,   # (B, F) per-frame
        entropy_layer: int | None = None,
        supervise_layer: int | None = None,
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
                      compute_entropy=(gidx == entropy_layer),
                      supervise=(gidx == supervise_layer))
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
