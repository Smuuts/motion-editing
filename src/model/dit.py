"""
MotionDiT / GroupDiT: epsilon-prediction diffusion transformers for 3D human motion.

Two design choices support LEDITS++ editing:
  - Dedicated cross-attention (not joint attention) exposes A^{t,l} maps
    directly via store_attn=True, enabling mask-guided editing.
  - Epsilon prediction (not velocity) matches the LEDITS++ inversion math.

The generic transformer blocks (attention, feed-forward, adaLN-zero DiT block) live
in model/layers.py; this module only defines how motion is tokenised into those
blocks (per-frame for MotionDiT, per-frame-per-body-part-group for GroupDiT).
"""

import torch
import torch.nn as nn

from model.layers import (
    TimestepEmbedding, FramePositionalEmbedding, DiTBlock, resolve_context_and_mask,
)

# Body-part group definitions live in their own module because they are also
# consumed by the LEDITS++ masking code (Stage 2 M1 + LLM-fallback mask) and by
# analyse_attention.py. GROUP_NAMES is re-exported here for backward compatibility
# with `from model.dit import GROUP_NAMES`.
from model.body_groups import (
    N_GROUPS as _N_GROUPS,
    GROUP_NAMES,
    group_layout as _group_layout,
    is_grouped_mode as _is_grouped_mode,
)


class _MotionDiTBase(nn.Module):
    """Shared transformer scaffolding, weight init, and attention-map accessor
    for MotionDiT / GroupDiT. Subclasses add their own tokeniser layers
    (joint_proj/output_proj vs. in_projs/out_projs/group_emb) and must call
    self._init_weights() once those are in place."""

    def __init__(
        self,
        latent_dim:   int,
        context_dim:  int,
        num_heads:    int,
        num_layers:   int,
        max_frames:   int,
        ff_mult:      int,
        dropout:      float,
        text_seq_len: int,
        ctx_pad_mask: bool = False,
        attn_sink:    bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_emb     = FramePositionalEmbedding(max_frames, latent_dim)
        self.time_emb    = TimestepEmbedding(latent_dim)
        # attn_sink: learnable per-head zero-value sink logit in every cross-attention
        # (see CrossAttention). Config-gated like ctx_pad_mask: it adds parameters, so a
        # checkpoint must be rebuilt exactly as trained.
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, context_dim, num_heads, ff_mult, dropout,
                     attn_sink=attn_sink)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)

        # Mask padding keys in cross-attention. Padding positions are detected as
        # all-zero context columns — the convention T5TextEncoder enforces (CLIP
        # does not zero its pads, so with CLIP this flag is a no-op). Without the
        # mask, the ~L−|words| zero-logit pad columns absorb ~93% of the softmax
        # mass and attenuate the cross-attention output ~14× (docs/FINDINGS.md
        # "padding sink"). Config-gated (default off) because checkpoints trained
        # unmasked equilibrated to the attenuated regime — enabling it on an old
        # checkpoint at inference rescales cross-attention out of distribution.
        self.ctx_pad_mask = ctx_pad_mask

        # null_text_emb length matches the text encoder's fixed output length (77 for CLIP,
        # configurable for T5). For LEDITS++ inversion (Stage 1): pass context=None so the
        # model uses this — inversion is unconditional.
        self.null_text_emb = nn.Parameter(torch.randn(1, text_seq_len, context_dim) * 0.02)

    def _context_and_mask(self, context, B):
        """Thin wrapper over the shared resolver (model.layers) so GroupDiT/MotionDiT
        and GroupMotionUNet resolve context/pad-mask identically."""
        return resolve_context_and_mask(context, B, self.null_text_emb, self.ctx_pad_mask)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # adaLN-zero: zero-init so every block starts as an identity residual (DiT §3.2)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def get_attn_maps(self) -> list[torch.Tensor]:
        """Returns cross-attention maps (B, heads, tokens, L_text) from all layers.

        For LEDITS++ Stage 2: call this after every inversion timestep and stack
        the results as (num_steps, num_layers, B, heads, tokens, L_text).
        Average over steps, layers, heads, then threshold at λ-th percentile to get M1.
        For GroupDiT, tokens=F*G; reshape to (F, G) to get the spatiotemporal map.
        """
        return [b.cross_attn.last_attn_map for b in self.blocks
                if b.cross_attn.last_attn_map is not None]

    def get_attn_entropy(self, block_idx: int) -> torch.Tensor:
        """Entropy stashed by the last forward that passed entropy_layer=block_idx
        (training regulariser; one block per step bounds the materialised-attention
        memory). Kept with graph — consume it in the same step's loss."""
        return self.blocks[block_idx].cross_attn.last_entropy


class MotionDiT(_MotionDiTBase):
    """
    Frame-level motion DiT. Each frame is a single token; input/output (B, F, input_dim).

    LEDITS++ note: cross-attention maps are (B, heads, F, L_text) — one row per frame.
    M1 averaging over edit-token columns produces a per-frame relevance vector of length F,
    which directly gives the temporal dimension of the spatiotemporal mask.
    No body-part spatial resolution: all joints in a frame are masked together.
    Use GroupDiT if joint-group resolution in the mask is needed.
    """

    def __init__(
        self,
        input_dim:    int   = 263,
        latent_dim:   int   = 512,
        context_dim:  int   = 512,
        num_heads:    int   = 8,
        num_layers:   int   = 8,
        max_frames:   int   = 196,
        ff_mult:      int   = 4,
        dropout:      float = 0.1,
        text_seq_len: int   = 77,
        ctx_pad_mask: bool  = False,
        attn_sink:    bool  = False,
    ):
        super().__init__(latent_dim, context_dim, num_heads, num_layers,
                          max_frames, ff_mult, dropout, text_seq_len, ctx_pad_mask,
                          attn_sink)
        self.input_dim = input_dim

        self.joint_proj  = nn.Linear(input_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, input_dim)
        self._init_weights()

    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, input_dim) noisy feature vectors
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None → uses null_text_emb
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,
        entropy_layer: int | None = None,
    ) -> torch.Tensor:
        B, F, _ = motion.shape
        context, ctx_mask = self._context_and_mask(context, B)

        x = self.joint_proj(motion) + self.pos_emb(F, motion.device)
        t_emb = self.time_emb(t)

        for i, block in enumerate(self.blocks):
            x = block(x, t_emb, context, store_attn=store_attn, mask=mask,
                      context_mask=ctx_mask, compute_entropy=(i == entropy_layer))

        return self.output_proj(self.final_norm(x))


class GroupDiT(_MotionDiTBase):
    """
    Body-part-grouped motion DiT.

    Joints are aggregated into 7 body-part tokens per frame (root, left leg,
    right leg, spine, left arm, right arm, head), giving F×7=1,372 tokens
    instead of F=196 (MotionDiT) or F×22=4,312 (full joint-level).

    Each group token carries the full HumanML3D features for its joints:
    positions + 6D rotations + velocities, with foot-contact labels for the
    leg groups. Per-group dims: [7, 50, 50, 36, 48, 48, 24] → 263D total.

    Input/output shape: (B, F, 263).

    LEDITS++ note: cross-attention maps are (B, heads, F*G, L_text) where G=7.
    Reshape to (B, heads, F, G, L_text), then average over heads and edit-token
    columns to get the (F, G) spatiotemporal map for mask M1.
    Each (frame, group) cell answers: "is this body-part group at this frame
    semantically relevant to the edit instruction?"
    This is the primary architecture advantage over MotionDiT for LEDITS++.
    """

    def __init__(
        self,
        feature_mode: str   = "humanml3d",
        latent_dim:   int   = 512,
        context_dim:  int   = 512,
        num_heads:    int   = 8,
        num_layers:   int   = 8,
        max_frames:   int   = 196,
        ff_mult:      int   = 4,
        dropout:      float = 0.1,
        text_seq_len: int   = 77,
        ctx_pad_mask: bool  = False,
        attn_sink:    bool  = False,
    ):
        super().__init__(latent_dim, context_dim, num_heads, num_layers,
                          max_frames, ff_mult, dropout, text_seq_len, ctx_pad_mask,
                          attn_sink)
        # Representation-specific channel partition: 'humanml3d' (263) or 'smplh' (135).
        # The body-part grouping (N_GROUPS, GROUP_NAMES) is shared across both.
        self.group_channels, group_dims, self.input_dim = _group_layout(feature_mode)
        self._group_dims = group_dims

        self.in_projs  = nn.ModuleList([nn.Linear(d, latent_dim) for d in group_dims])
        self.out_projs = nn.ModuleList([nn.Linear(latent_dim, d) for d in group_dims])
        self.group_emb = nn.Embedding(_N_GROUPS, latent_dim)

        # Precomputed index tensors (not learned params, hence persistent=False so
        # they never appear in — or are expected from — a checkpoint's state_dict):
        #   _group_idx: group-embedding lookup, avoids a fresh torch.arange per forward.
        #   _perm_idx:  channel order -> grouped order, so tokenising the feature
        #               vector is one gather (+ free split) instead of G fancy-indexes.
        #   _inv_perm_idx: grouped order -> channel order, for the symmetric regather
        #               on reconstruction instead of a zero-alloc + G scatter writes.
        perm = torch.cat([torch.as_tensor(ch, dtype=torch.long) for ch in self.group_channels])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel())
        self.register_buffer("_group_idx", torch.arange(_N_GROUPS), persistent=False)
        self.register_buffer("_perm_idx", perm, persistent=False)
        self.register_buffer("_inv_perm_idx", inv_perm, persistent=False)

        self._init_weights()

    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, 263)
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,  # (B, F) per-frame
        entropy_layer: int | None = None,
    ) -> torch.Tensor:
        B, F, _ = motion.shape
        G = _N_GROUPS  # 7

        context, ctx_mask = self._context_and_mask(context, B)

        # ── tokenise ──────────────────────────────────────────────────────────
        # Single gather into group order, then a free (view-only) split per group,
        # instead of one fancy-index slice per group.
        group_feats = torch.split(
            motion.index_select(-1, self._perm_idx), self._group_dims, dim=-1
        )

        # project each group → latent_dim, stack → (B, F, G, latent_dim)
        tokens = torch.stack(
            [proj(feat) for proj, feat in zip(self.in_projs, group_feats)], dim=2
        )

        # ── positional embeddings ─────────────────────────────────────────────
        tokens = tokens + self.group_emb(self._group_idx)[None, None]
        tokens = tokens + self.pos_emb(F, motion.device)[None, :, None]

        # ── transformer blocks ────────────────────────────────────────────────
        # Flatten to (B, F*G, latent_dim) for the transformer.
        # For LEDITS++ Stage 2: after collecting last_attn_map (B, heads, F*G, L_text),
        # reshape to (B, heads, F, G, L_text) to recover body-part × frame structure.
        tokens = tokens.reshape(B, F * G, self.latent_dim)

        if mask is not None:
            mask = mask[:, :, None].expand(B, F, G).reshape(B, F * G)

        t_emb = self.time_emb(t)
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, t_emb, context, store_attn=store_attn, mask=mask,
                           context_mask=ctx_mask, compute_entropy=(i == entropy_layer))

        tokens = self.final_norm(tokens).reshape(B, F, G, self.latent_dim)

        # ── reconstruct (B, F, input_dim) ─────────────────────────────────────
        # Concat back into grouped order, then a single gather via the inverse
        # permutation restores original channel order — one op instead of a
        # zero-alloc + one scatter write per group.
        group_outs = [self.out_projs[g](tokens[:, :, g]) for g in range(len(self.group_channels))]
        return torch.cat(group_outs, dim=-1).index_select(-1, self._inv_perm_idx)


def build_model(config: dict, device="cpu") -> nn.Module:
    kwargs = dict(
        latent_dim   = config.get("latent_dim",   512),
        context_dim  = config.get("context_dim",  512),
        num_heads    = config.get("num_heads",     8),
        num_layers   = config.get("num_layers",    8),
        max_frames   = config.get("max_frames",    196),
        ff_mult      = config.get("ff_mult",       4),
        dropout      = config.get("dropout",       0.1),
        text_seq_len = config.get("text_seq_len",  77),
        # Default False: checkpoints saved before 2026-07-15 trained without the
        # pad mask and must be rebuilt exactly as trained (see _MotionDiTBase).
        ctx_pad_mask = config.get("ctx_pad_mask", False),
        attn_sink    = config.get("attn_sink", False),
    )
    feature_mode = config.get("feature_mode", "humanml3d")
    arch = config.get("arch", "dit")
    if arch == "unet":
        # MotionCLR-style temporal U-Net over the same body-part group tokens.
        # Epsilon-prediction and the full config kwarg set are shared with GroupDiT;
        # depth comes from unet_levels/unet_blocks_per_level (num_layers is ignored).
        # Import here to avoid a circular import (unet.py imports from model.layers,
        # not model.dit, but keeping the import local mirrors the optional-backbone
        # intent and avoids importing torch-heavy conv code when arch="dit").
        from model.unet import GroupMotionUNet
        if not _is_grouped_mode(feature_mode):
            raise ValueError(
                f"arch='unet' (GroupMotionUNet) requires a grouped feature_mode "
                f"(humanml3d/smplh); got {feature_mode!r}.")
        model = GroupMotionUNet(
            feature_mode=feature_mode,
            unet_levels=config.get("unet_levels", 3),
            unet_blocks_per_level=config.get("unet_blocks_per_level", 2),
            **kwargs,
        )
    elif _is_grouped_mode(feature_mode):
        model = GroupDiT(feature_mode=feature_mode, **kwargs)
    else:
        # Legacy flat MotionDiT (deprecated) — kept only for loading old flat checkpoints.
        model = MotionDiT(input_dim=config.get("input_dim", 263), **kwargs)
    return model.to(device)
