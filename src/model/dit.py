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

        self.last_attn_map = None  # (B, heads, N_motion, L_text) — set when store_attn=True

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                store_attn: bool = False) -> torch.Tensor:
        B, N, _ = x.shape
        _, L, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.dropout(attn)

        if store_attn:
            self.last_attn_map = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
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

        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
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
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, store_attn: bool = False,
                mask: torch.Tensor | None = None):
        mods = self.adaLN_modulation(t_emb)
        s1, b1, s2, b2, s3, b3 = mods.chunk(6, dim=-1)
        s1, b1 = s1[:, None], b1[:, None]
        s2, b2 = s2[:, None], b2[:, None]
        s3, b3 = s3[:, None], b3[:, None]

        x = x + self.self_attn(self.norm1(x) * (1 + s1) + b1, mask=mask)
        x = x + self.cross_attn(self.norm2(x) * (1 + s2) + b2, context, store_attn=store_attn)
        x = x + self.ff(self.norm3(x) * (1 + s3) + b3)
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
        """Returns cross-attention maps (B, heads, tokens, L_text) from all layers."""
        return [b.cross_attn.last_attn_map for b in self.blocks
                if b.cross_attn.last_attn_map is not None]


class MotionDiT(_MotionDiTBase):
    """
    Frame-level motion DiT. Each frame is a single token; input/output (B, F, input_dim).
    """

    def __init__(
        self,
        input_dim:   int   = 263,
        latent_dim:  int   = 512,
        context_dim: int   = 512,
        num_heads:   int   = 8,
        num_layers:  int   = 8,
        max_frames:  int   = 196,
        ff_mult:     int   = 4,
        dropout:     float = 0.1,
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

        # null_text_emb: sequence length 77 to match CLIP, avoiding key distribution mismatch
        self.null_text_emb = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)
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


# Body-part groups: (name, joint_indices into the 21-joint body tensor).
# Joint indices are 0-based, ordered as they appear in the SMPL 130-dim vector:
#   0=L_Hip 1=R_Hip 2=Spine1 3=L_Knee 4=R_Knee 5=Spine2 6=L_Ankle 7=R_Ankle
#   8=Spine3 9=L_Foot 10=R_Foot 11=Neck 12=L_Collar 13=R_Collar 14=Head
#   15=L_Shoulder 16=R_Shoulder 17=L_Elbow 18=R_Elbow 19=L_Wrist 20=R_Wrist
_BODY_PART_GROUPS = [
    ("left_leg",  [0, 3, 6, 9]),      # L_Hip, L_Knee, L_Ankle, L_Foot
    ("right_leg", [1, 4, 7, 10]),     # R_Hip, R_Knee, R_Ankle, R_Foot
    ("spine",     [2, 5, 8]),          # Spine1, Spine2, Spine3
    ("left_arm",  [12, 15, 17, 19]),  # L_Collar, L_Shoulder, L_Elbow, L_Wrist
    ("right_arm", [13, 16, 18, 20]),  # R_Collar, R_Shoulder, R_Elbow, R_Wrist
    ("head",      [11, 14]),           # Neck, Head
]
_N_GROUPS = 1 + len(_BODY_PART_GROUPS)  # 7: root + 6 body-part groups

# Inverse permutation: maps the group-concatenated joint order back to the
# canonical 21-joint order so output can be scattered back correctly.
# Group concat order: left_leg[0,3,6,9] right_leg[1,4,7,10] spine[2,5,8]
#                     left_arm[12,15,17,19] right_arm[13,16,18,20] head[11,14]
_JOINT_INV_PERM = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 19, 11, 15, 20, 12, 16, 13, 17, 14, 18]


class GroupDiT(_MotionDiTBase):
    """
    Body-part-grouped motion DiT.

    Joints are aggregated into 7 body-part tokens per frame (root, left leg,
    right leg, spine, left arm, right arm, head), giving F×7=1,372 tokens
    instead of F=196 (MotionDiT) or F×22=4,312 (full joint-level).

    Each group token is the learned projection of the concatenated 6D rotations
    of that body part (or the 4-dim root features for the root token).
    A group identity embedding and a frame positional embedding are added.

    Input/output shape: (B, F, 130) — identical to SMPL-mode MotionDiT.
    """

    def __init__(
        self,
        latent_dim:  int   = 512,
        context_dim: int   = 512,
        num_heads:   int   = 8,
        num_layers:  int   = 8,
        max_frames:  int   = 196,
        ff_mult:     int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.input_dim  = 130  # fixed: 4 root + 21 × 6 body joints

        # Input/output dims per group: root=4, body parts=n_joints×6
        group_dims = [4] + [len(ids) * 6 for _, ids in _BODY_PART_GROUPS]

        self.in_projs  = nn.ModuleList([nn.Linear(d, latent_dim) for d in group_dims])
        self.out_projs = nn.ModuleList([nn.Linear(latent_dim, d) for d in group_dims])

        self.group_emb = nn.Embedding(_N_GROUPS, latent_dim)
        self.pos_emb   = FramePositionalEmbedding(max_frames, latent_dim)
        self.time_emb  = TimestepEmbedding(latent_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, context_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm    = nn.LayerNorm(latent_dim)
        self.null_text_emb = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)

        self.register_buffer("inv_perm", torch.tensor(_JOINT_INV_PERM))
        self._init_weights()

    def forward(
        self,
        motion:     torch.Tensor,         # (B, F, 130)
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
        root_feats  = motion[..., :4]                            # (B, F, 4)
        body_joints = motion[..., 4:].reshape(B, F, 21, 6)      # (B, F, 21, 6)

        group_feats = [root_feats] + [
            body_joints[:, :, ids, :].reshape(B, F, len(ids) * 6)
            for _, ids in _BODY_PART_GROUPS
        ]

        # project each group → latent_dim, stack → (B, F, G, latent_dim)
        tokens = torch.stack(
            [proj(feat) for proj, feat in zip(self.in_projs, group_feats)], dim=2
        )

        # ── positional embeddings ─────────────────────────────────────────────
        tokens = tokens + self.group_emb(torch.arange(G, device=motion.device))[None, None]
        tokens = tokens + self.pos_emb(F, motion.device)[None, :, None]

        # ── transformer blocks ────────────────────────────────────────────────
        tokens = tokens.reshape(B, F * G, self.latent_dim)

        if mask is not None:
            mask = mask[:, :, None].expand(B, F, G).reshape(B, F * G)

        t_emb = self.time_emb(t)
        for block in self.blocks:
            tokens = block(tokens, t_emb, context, store_attn=store_attn, mask=mask)

        tokens = self.final_norm(tokens).reshape(B, F, G, self.latent_dim)

        # ── reconstruct (B, F, 130) ───────────────────────────────────────────
        root_pred = self.out_projs[0](tokens[:, :, 0])          # (B, F, 4)

        # project each body-part group and cat in group order → (B, F, 21, 6)
        body_pred = torch.cat([
            self.out_projs[i + 1](tokens[:, :, i + 1]).reshape(B, F, len(ids), 6)
            for i, (_, ids) in enumerate(_BODY_PART_GROUPS)
        ], dim=2)

        # reorder from group-concat order back to canonical joint order
        body_pred = body_pred[:, :, self.inv_perm]              # (B, F, 21, 6)

        return torch.cat([root_pred, body_pred.reshape(B, F, 126)], dim=-1)


def build_model(config: dict, device="cpu") -> _MotionDiTBase:
    kwargs = dict(
        latent_dim  = config.get("latent_dim",  512),
        context_dim = config.get("context_dim", 512),
        num_heads   = config.get("num_heads",   8),
        num_layers  = config.get("num_layers",  8),
        max_frames  = config.get("max_frames",  196),
        ff_mult     = config.get("ff_mult",     4),
        dropout     = config.get("dropout",     0.1),
    )
    if config.get("feature_mode") == "group":
        model = GroupDiT(**kwargs)
    else:
        model = MotionDiT(input_dim=config.get("input_dim", 263), **kwargs)
    return model.to(device)
