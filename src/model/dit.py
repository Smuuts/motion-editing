"""
MotionDiT: epsilon-prediction diffusion transformer for 3D human motion.

Two design choices support LEDITS++ editing:
  - Dedicated cross-attention (not joint attention) exposes A^{t,l} maps
    directly via store_attn=True, enabling mask-guided editing.
  - Epsilon prediction (not velocity) matches the LEDITS++ inversion math.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Body-part group definitions live in their own module because they are also
# consumed by the LEDITS++ masking code (Stage 2 M1 + LLM-fallback mask) and by
# analyse_attention.py. GROUP_NAMES is re-exported here for backward compatibility
# with `from model.dit import GROUP_NAMES`.
from model.body_groups import (
    BODY_PART_GROUPS as _BODY_PART_GROUPS,
    N_GROUPS as _N_GROUPS,
    GROUP_CHANNELS as _GROUP_CHANNELS,
    GROUP_DIMS as _GROUP_DIMS,
    GROUP_NAMES,
)


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


class _MotionDiTBase(nn.Module):
    """Shared weight init and attention-map accessor for MotionDiT / GroupDiT."""

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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.input_dim  = input_dim

        self.joint_proj  = nn.Linear(input_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, input_dim)
        self.pos_emb     = FramePositionalEmbedding(max_frames, latent_dim)
        self.time_emb    = TimestepEmbedding(latent_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, context_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)

        # null_text_emb length matches the text encoder's fixed output length (77 for CLIP,
        # configurable for T5). For LEDITS++ inversion (Stage 1): pass context=None so the
        # model uses this — inversion is unconditional.
        self.null_text_emb = nn.Parameter(torch.randn(1, text_seq_len, context_dim) * 0.02)
        self._init_weights()

    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, input_dim) noisy feature vectors
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None → uses null_text_emb
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, F, _ = motion.shape
        if context is None:
            context = self.null_text_emb.expand(B, -1, -1)

        x = self.joint_proj(motion) + self.pos_emb(F, motion.device)
        t_emb = self.time_emb(t)

        for block in self.blocks:
            x = block(x, t_emb, context, store_attn=store_attn, mask=mask)

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
        latent_dim:   int   = 512,
        context_dim:  int   = 512,
        num_heads:    int   = 8,
        num_layers:   int   = 8,
        max_frames:   int   = 196,
        ff_mult:      int   = 4,
        dropout:      float = 0.1,
        text_seq_len: int   = 77,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.input_dim  = 263

        self.in_projs  = nn.ModuleList([nn.Linear(d, latent_dim) for d in _GROUP_DIMS])
        self.out_projs = nn.ModuleList([nn.Linear(latent_dim, d) for d in _GROUP_DIMS])

        self.group_emb = nn.Embedding(_N_GROUPS, latent_dim)
        self.pos_emb   = FramePositionalEmbedding(max_frames, latent_dim)
        self.time_emb  = TimestepEmbedding(latent_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, context_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm    = nn.LayerNorm(latent_dim)
        self.null_text_emb = nn.Parameter(torch.randn(1, text_seq_len, context_dim) * 0.02)

        self._init_weights()

    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, 263)
        t:          torch.Tensor,         # (B,)
        context:    torch.Tensor | None,  # (B, L, context_dim) or None
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,  # (B, F) per-frame
    ) -> torch.Tensor:
        B, F, _ = motion.shape
        G = _N_GROUPS  # 7

        if context is None:
            context = self.null_text_emb.expand(B, -1, -1)

        # ── tokenise ──────────────────────────────────────────────────────────
        # Slice each group's channels from the full 263D vector.
        group_feats = [motion[..., ch] for ch in _GROUP_CHANNELS]

        # project each group → latent_dim, stack → (B, F, G, latent_dim)
        tokens = torch.stack(
            [proj(feat) for proj, feat in zip(self.in_projs, group_feats)], dim=2
        )

        # ── positional embeddings ─────────────────────────────────────────────
        tokens = tokens + self.group_emb(torch.arange(G, device=motion.device))[None, None]
        tokens = tokens + self.pos_emb(F, motion.device)[None, :, None]

        # ── transformer blocks ────────────────────────────────────────────────
        # Flatten to (B, F*G, latent_dim) for the transformer.
        # For LEDITS++ Stage 2: after collecting last_attn_map (B, heads, F*G, L_text),
        # reshape to (B, heads, F, G, L_text) to recover body-part × frame structure.
        tokens = tokens.reshape(B, F * G, self.latent_dim)

        if mask is not None:
            mask = mask[:, :, None].expand(B, F, G).reshape(B, F * G)

        t_emb = self.time_emb(t)
        for block in self.blocks:
            tokens = block(tokens, t_emb, context, store_attn=store_attn, mask=mask)

        tokens = self.final_norm(tokens).reshape(B, F, G, self.latent_dim)

        # ── reconstruct (B, F, 263) ───────────────────────────────────────────
        # Compute projections first so we know the actual dtype (AMP may differ
        # from tokens.dtype when LayerNorm upcasts to float32 internally).
        group_outs = [self.out_projs[g](tokens[:, :, g]) for g in range(len(_GROUP_CHANNELS))]
        out = torch.zeros(B, F, 263, device=tokens.device, dtype=group_outs[0].dtype)
        for g, ch in enumerate(_GROUP_CHANNELS):
            out[..., ch] = group_outs[g]
        return out


def build_model(config: dict, device="cpu") -> _MotionDiTBase:
    kwargs = dict(
        latent_dim   = config.get("latent_dim",   512),
        context_dim  = config.get("context_dim",  512),
        num_heads    = config.get("num_heads",     8),
        num_layers   = config.get("num_layers",    8),
        max_frames   = config.get("max_frames",    196),
        ff_mult      = config.get("ff_mult",       4),
        dropout      = config.get("dropout",       0.1),
        text_seq_len = config.get("text_seq_len",  77),
    )
    if config.get("feature_mode") == "group":
        model = GroupDiT(**kwargs)
    else:
        model = MotionDiT(input_dim=config.get("input_dim", 263), **kwargs)
    return model.to(device)
