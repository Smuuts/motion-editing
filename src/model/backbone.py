"""
The scaffolding every attention-capturing backbone shares.

`_MotionDiTBase` holds the parts that are the same whether motion is tokenised per frame
(MotionDiT) or per frame-per-body-part-group (GroupDiT): the timestep and positional
embeddings, the DiT block stack, weight init, and the accessors the editing and training
code read stored attention through. The U-Net backbone reimplements those accessors
rather than inheriting, because its blocks live at several frame resolutions.
"""

import torch
import torch.nn as nn

from model.layers import (DiTBlock, FramePositionalEmbedding, TimestepEmbedding,
                          resolve_context_and_mask)

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
        # mass and attenuate the cross-attention output ~14× (measured;
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

    def get_attn_values(self) -> list[torch.Tensor]:
        """Value vectors (B, heads, L_text, head_dim) from the same stored forward as
        get_attn_maps(), layer for layer and column for column.

        For the norm-weighted M1 read-outs: attention weight ranks keys by
        how hard a query LOOKS at them, not by what it RECEIVES, and those disagree
        whenever value norms differ — maximally so for the zero-value padding columns
        that once held 92.9% of this model's cross-attention mass.
        """
        return [b.cross_attn.last_value for b in self.blocks
                if b.cross_attn.last_value is not None]

    def get_attn_entropy(self, block_idx: int) -> torch.Tensor:
        """Entropy stashed by the last forward that passed entropy_layer=block_idx
        (training regulariser; one block per step bounds the materialised-attention
        memory). Kept with graph — consume it in the same step's loss."""
        return self.blocks[block_idx].cross_attn.last_entropy

    def get_sup_attn(self, block_idx: int) -> torch.Tensor:
        """Head-averaged cross-attention (B, F, G, L_text) stashed by the last forward
        that passed supervise_layer=block_idx, WITH graph — the input to the
        TokenCompose grounding loss (training/grounding.py).

        The F·G → (F, G) split is row-major, matching how GroupDiT.forward flattens its
        token grid (tokens.reshape(B, F*G, latent_dim) after stacking groups on dim=2),
        so cell (f, g) here is body-part group g of frame f. G = 1 for the flat
        MotionDiT, which keeps the loss's shape contract identical across backbones.

        READING IT CLEARS IT. Unlike get_attn_entropy's scalar, this is a full
        (B, F·G, L) tensor with a grad_fn, and a different block is sampled each step —
        so left in place it would pin one stale map (and its graph nodes) per candidate
        block until that block happens to be drawn again. ~32 MB each at batch 45 with
        L = 128, for a value no one will read twice. Dropping the reference here does
        not affect backward: the loss's own graph holds everything it needs.
        """
        cross = self.blocks[block_idx].cross_attn
        attn, cross.last_sup_attn = cross.last_sup_attn, None       # (B, N, L)
        if attn is None:
            return None
        B, N, L = attn.shape
        G = getattr(self, "G", 1)
        return attn.reshape(B, N // G, G, L)
