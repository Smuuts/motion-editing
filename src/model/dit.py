"""
Cross-Attention Motion Diffusion Transformer (MotionDiT)

Architecture decisions relevant to LEDITS++ transfer:
  - Text conditioning via DEDICATED cross-attention layers, not joint attention.
    This ensures cross-attention maps A^t in R^(|c| x J*F) are directly
    extractable without approximation.
  - DDPM noise prediction (epsilon-prediction), not velocity prediction.
    Required for the LEDITS++ inversion derivation.
  - Attention maps are stored during the forward pass when store_attn=True,
    enabling mask construction during inversion.
"""

import math
import torch
import torch.nn as nn


# ── Sinusoidal timestep embedding ─────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Compute a learnable embedding for diffusion timesteps.

    This module converts integer timesteps into a continuous embedding using
    sinusoidal positional encoding followed by a small MLP. The resulting
    embedding conditions the transformer blocks on the current diffusion step.

    Relevance for LEDITS++:
      - LEDITS++ inversion depends on correctly modeling the forward/backward
        DDPM schedule, so timestep conditioning must be stable and expressive.
      - The produced timestep embedding is used by the block-level adaLN
        conditioning that modulates attention and feed-forward behavior.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


# ── Frame positional embedding ─────────────────────────────────────────────────

class FramePositionalEmbedding(nn.Module):
    """
    Learnable positional embedding over the frame axis.

    This module provides a per-frame positional signal for motion tokens.
    It assigns each frame index a learnable vector so the transformer can
    distinguish temporal order within the motion sequence.

    Relevance for LEDITS++:
      - Strong positional encoding is essential for motion editing because
        frame-aware attention is needed to preserve temporal structure.
      - LEDITS++ uses attention maps over motion frames, so correct frame
        indexing helps make the extracted masks meaningful.
    """
    def __init__(self, max_frames: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_frames, dim)

    def forward(self, num_frames: int, device) -> torch.Tensor:
        # Returns positional embeddings for num_frames frames
        # Shape: (num_frames, dim)
        positions = torch.arange(num_frames, device=device)
        return self.emb(positions)


# ── Cross-attention (text → motion) ───────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention from text context into motion tokens.

    Queries are produced from motion token embeddings, while keys and values
    come from text token embeddings. This dedicated cross-attention layer
    enables the model to condition motion generation on textual input.

    Relevance for LEDITS++:
      - LEDITS++ relies on directly accessible cross-attention maps A^{t,l}
        to construct masks for editing and inversion.
      - By storing `last_attn_map` when `store_attn=True`, the module makes
        these attention maps available for the masking stage without altering
        the forward pass semantics.
    """

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

        self.last_attn_map = None  # (B, heads, N_motion, L_text)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                store_attn: bool = False) -> torch.Tensor:
        """
        x       : (B, N, dim)        motion tokens
        context : (B, L, context_dim) text tokens
        """
        B, N, _ = x.shape
        _, L, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.dropout(attn)

        if store_attn:
            self.last_attn_map = attn.detach()  # (B, heads, N, L)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


# ── Self-attention ─────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Multi-head self-attention over motion tokens.

    This module computes attention between motion tokens themselves, allowing
    the model to capture intra-motion relationships across joints and frames.
    It supports an optional mask for variable-length or partially observed
    motion sequences.

    Relevance for LEDITS++:
      - Self-attention enables the transformer to maintain coherent motion
        structure independently of text conditioning.
      - During inversion or editing, preserving motion consistency is important
        when combining cross-attention-based masks with the underlying motion
        representation.
    """
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if mask is not None:
            out = out.masked_fill(~mask[:, :, None], 0.0)
        return self.out(out)


# ── Feed-forward ───────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP used after attention in each transformer block.

    This module applies a hidden expansion followed by GELU activation and
    a projection back to the original dimension, providing non-linear
    transformation capacity within each block.

    Relevance for LEDITS++:
      - The feed-forward network refines the motion representation after both
        self- and cross-attention, helping the model learn the noise prediction
        needed for accurate inversion and editing.
    """
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


# ── DiT block ─────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Single transformer block combining motion self-attention, text cross-attention,
    and feed-forward transformation with timestep-conditioned adaptive LayerNorm.

    Each block first applies self-attention to model motion dynamics, then
    cross-attention to let text influence motion, and finally a feed-forward
    network for non-linear refinement. Timestep-dependent adaLN modulation
    allows the block to behave differently at different diffusion steps.

    Relevance for LEDITS++:
      - Dedicated cross-attention layers are a core design choice for LEDITS++,
        since they expose the required text-to-motion attention maps directly.
      - The adaptive normalization ensures the diffusion model can be inverted
        correctly at each timestep, which is vital for mask-guided editing.
    """

    def __init__(self, dim: int, context_dim: int, num_heads: int,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()

        self.norm1     = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
        self.ff         = FeedForward(dim, ff_mult, dropout)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, store_attn: bool = False,
                mask: torch.Tensor | None = None):
        """
        x       : (B, F, dim)
        t_emb   : (B, dim)
        context : (B, L, context_dim)
        """
        mods = self.adaLN_modulation(t_emb)
        s1, b1, s2, b2, s3, b3 = mods.chunk(6, dim=-1)
        s1, b1 = s1[:, None], b1[:, None]
        s2, b2 = s2[:, None], b2[:, None]
        s3, b3 = s3[:, None], b3[:, None]

        # 1. temporal self-attention
        x = x + self.self_attn(self.norm1(x) * (1 + s1) + b1, mask=mask)

        # 2. cross-attention (text → motion)
        x = x + self.cross_attn(
            self.norm2(x) * (1 + s2) + b2,
            context,
            store_attn=store_attn,
        )

        # 3. feed-forward
        x = x + self.ff(self.norm3(x) * (1 + s3) + b3)

        return x


# ── Full MotionDiT ─────────────────────────────────────────────────────────────

class MotionDiT(nn.Module):
    """
    Cross-attention Diffusion Transformer for 3D human motion.

    This model encodes motion as a sequence of joint-frame tokens and conditions
    generation on text via dedicated cross-attention layers. It predicts the
    DDPM noise epsilon for each motion frame, enabling diffusion-based
    reconstruction and editing.

    Input:
        motion  : (B, F, 263)  noisy HumanML3D feature vectors
        t       : (B,)         diffusion timesteps
        context : (B, L, C)    CLIP text embeddings (L=77)

    Output:
        eps_pred : (B, F, 263)  predicted noise

    Relevance for LEDITS++:
      - The model is intentionally built for LEDITS++ by using noise prediction
        rather than velocity prediction, matching the inversion math.
      - It stores cross-attention maps from all transformer layers, which are
        later aggregated into LEDITS++ masks for localized motion editing.
      - Classifier-free guidance support via `null_text_emb` enables
        conditional and unconditional inference required in text-guided editing.
    """

    def __init__(
        self,
        input_dim:      int = 263,
        latent_dim:     int = 512,
        context_dim:    int = 512,   # CLIP embedding dim (ViT-L/14 = 768, ViT-B/32 = 512)
        num_heads:      int = 8,
        num_layers:     int = 8,
        max_frames:     int = 196,
        ff_mult:        int = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_frames = max_frames

        self.input_dim   = input_dim
        self.joint_proj  = nn.Linear(input_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, input_dim)

        self.pos_emb  = FramePositionalEmbedding(max_frames, latent_dim)
        self.time_emb = TimestepEmbedding(latent_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, context_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(latent_dim)

        # classifier-free guidance: learnable null text embedding
        # used when context=None (unconditional forward pass)
        # null text embedding for classifier-free guidance unconditional passes
        # must match CLIP sequence length (77) to avoid key distribution mismatch
        self.null_text_emb = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)

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
        # adaLN-zero: zero-init the final linear in each block's modulation so
        # that every DiT block starts as an identity residual (DiT paper §3.2).
        # Without this, random scale/shift values destabilise early training.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(
        self,
        motion:     torch.Tensor,            # (B, F, 263) noisy feature vectors
        t:          torch.Tensor,            # (B,)
        context:    torch.Tensor | None,     # (B, L, context_dim) or None
        store_attn: bool = False,
        mask:       torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, F, _ = motion.shape

        # handle unconditional pass (classifier-free guidance)
        if context is None:
            context = self.null_text_emb.expand(B, -1, -1)  # (B, 1, context_dim)

        # project motion features into latent space
        x = self.joint_proj(motion)  # (B, F, latent_dim)

        # add positional encoding per frame
        x = x + self.pos_emb(F, motion.device)  # (F, latent_dim) broadcasted

        # timestep embedding
        t_emb = self.time_emb(t)  # (B, latent_dim)

        # mask stays per-frame (B, F)
        frame_mask = mask  # (B, F) or None

        # transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, context, store_attn=store_attn, mask=frame_mask)

        x = self.final_norm(x)
        eps_pred = self.output_proj(x)  # (B, F, input_dim)

        return eps_pred

    def get_attn_maps(self):
        """
        Returns cross-attention maps from all layers after a forward pass
        with store_attn=True.

        Returns list of (B, heads, F, L) tensors, one per layer.
        These are A^{t,l} used to build M1 in the LEDITS++ masking stage.
        Per-frame attention means shapes are (batch, heads, num_frames, text_length).
        """
        maps = []
        for block in self.blocks:
            if block.cross_attn.last_attn_map is not None:
                maps.append(block.cross_attn.last_attn_map)
        return maps

def build_model(config: dict, device="cpu") -> MotionDiT:
    model = MotionDiT(
        input_dim  = config.get("input_dim",  263),
        latent_dim = config.get("latent_dim", 512),
        context_dim= config.get("context_dim",512),
        num_heads  = config.get("num_heads",  8),
        num_layers = config.get("num_layers", 8),
        max_frames = config.get("max_frames", 196),
        ff_mult    = config.get("ff_mult",    4),
        dropout    = config.get("dropout",    0.1),
    )
    return model.to(device)