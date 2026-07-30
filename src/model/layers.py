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


def resolve_context_and_mask(context, B, null_text_emb, ctx_pad_mask):
    """Resolve the (context, context_mask) pair for a forward pass — shared by every
    backbone (GroupDiT/MotionDiT via _MotionDiTBase, and GroupMotionUNet) so the
    padding-sink convention lives in exactly one place and can't drift between them.

    context=None → the learned null embedding, never masked (all its columns are real).
    With ctx_pad_mask, all-zero columns are treated as padding — this also holds
    sample-wise for CFG-dropout batches where torch.where mixed null_text_emb rows
    (nonzero everywhere → unmasked, as intended). Returns ctx_mask=None (keeping the
    fused SDPA fast path) whenever nothing is padded. The pad mask is inference-critical
    (getting it wrong is FID 0.65 → 27.0; see docs/FINDINGS.md 'padding sink')."""
    if context is None:
        return null_text_emb.expand(B, -1, -1), None
    if not ctx_pad_mask:
        return context, None
    ctx_mask = context.abs().sum(dim=-1) > 0                 # (B, L)
    if ctx_mask.all():
        ctx_mask = None
    return context, ctx_mask


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int,
                 dropout: float = 0.0, use_sink: bool = False):
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

        # Learnable attention sink (GPT-OSS-style "sink attention" / softmax-off-by-one
        # with a learnable per-head offset): one extra logit per head participates in the
        # softmax denominator but carries a zero VALUE vector, so mass routed to it simply
        # vanishes from the output. Gives queries that don't need text a principled dump
        # site — instead of hijacking EOS (or, pre-ctx_pad_mask, the zero-value padding
        # columns; see docs/FINDINGS.md "padding sink"). Init 0 = exactly one synthetic
        # pad-like column, learnable from there. Config-gated (attn_sink) because it adds
        # a parameter: old checkpoints don't have it and must load without it.
        self.sink_logit = nn.Parameter(torch.zeros(num_heads)) if use_sink else None

        # Shape when stored: (B, heads, N_motion, L_text) — real-token columns only.
        # With the sink enabled rows sum to <= 1; the missing mass went to the sink.
        # N_motion = F for MotionDiT, F*G for GroupDiT (G=7 body-part groups).
        # For Stage 2 mask M1: accumulate these across ALL inversion timesteps and layers,
        # then average over heads, timesteps, and layers before thresholding.
        self.last_attn_map = None

        # Mean normalised row entropy of the last forward's real-token attention,
        # kept WITH graph when compute_entropy=True (training regulariser); see forward.
        self.last_entropy = None

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                store_attn: bool = False,
                context_mask: torch.Tensor | None = None,
                compute_entropy: bool = False) -> torch.Tensor:
        # store_attn=True is passed during Stage 1 inversion and Stage 3 denoising
        # to collect A^{t,l} maps. Must be False during training to avoid memory growth.
        # context_mask: (B, L) bool, False = padding key. Without it, every padding
        # column competes in the softmax with key logit exactly 0 (zeroed T5 pad
        # embedding × bias-free W_k) and absorbs ~93% of the mass for short texts,
        # attenuating the (zero-value) cross-attention output ~14× — see
        # docs/FINDINGS.md "padding sink". Old checkpoints trained unmasked; the
        # model only passes a mask when built with ctx_pad_mask=True.
        # compute_entropy=True additionally stashes the mean normalised row entropy
        # of the real-token attention in self.last_entropy (graph kept) for the
        # training-time entropy regulariser.
        B, N, _ = x.shape
        _, L, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # The fused SDPA kernel can't express the sink column or return probabilities,
        # so any of these features forces the explicit-softmax path (costs the
        # materialised (B, h, N, L) attention tensor — during training too, when the
        # sink or the entropy regulariser is enabled).
        explicit = store_attn or compute_entropy or self.sink_logit is not None

        if explicit:
            logits = q @ k.transpose(-2, -1) * self.scale        # (B, h, N, L)
            if context_mask is not None:
                logits = logits.masked_fill(~context_mask[:, None, None, :],
                                            torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
            if self.sink_logit is not None:
                # softmax([logits, sink])[..., :L] == softmax(logits) * Z/(Z + e^sink)
                # == softmax(logits) * sigmoid(logsumexp(logits) − sink): same result
                # as concatenating a sink column, without materialising the extra
                # (B, h, N, L+1) tensors in the memory-critical explicit path.
                lse = torch.logsumexp(logits, dim=-1, keepdim=True)
                attn = attn * torch.sigmoid(lse - self.sink_logit.view(1, -1, 1, 1))
                # rows now sum to <= 1; the missing mass went to the sink

            if compute_entropy:
                # Entropy of the REAL-token distribution, renormalised per row: measures
                # how spread attention is over the words themselves, independent of how
                # much total mass the sink/padding absorbed. Normalised by log(#valid
                # keys) so it lives in [0, 1] and the loss weight is scale-free.
                p = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                h_row = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)     # (B, h, N)
                log_n = (context_mask.sum(dim=-1).clamp(min=2).float().log()[:, None, None]
                         if context_mask is not None else math.log(L))
                self.last_entropy = (h_row / log_n).mean()

            attn = self.dropout(attn)
            if store_attn:
                self.last_attn_map = attn.detach()
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        else:
            attn_mask = (context_mask[:, None, None, :]
                         if context_mask is not None else None)
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 dropout_p=dropout_p)
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

        # ── probe-only token↔token attention capture (analyse_self_attention.py) ──
        # Self-attention is the one attention pathway none of the M1/M2 probes ever
        # read (they are all cross-attention); Option 7 in
        # docs/AttentionGrounding_Options.md ports DiffSeg (arXiv 2308.12469) onto it
        # to look for emergent (frame, body-part) segment structure.
        #
        # Enabled by SETTING THE ATTRIBUTE on the module, not by a forward kwarg —
        # deliberately. Capture is needed identically in GroupDiT (model/dit.py) and
        # GroupMotionUNet (model/unet.py), and an attribute avoids threading a new
        # argument through DiTBlock.forward, CLRBlock.forward and both model forwards
        # for an inference-only diagnostic. Defaults keep the fused SDPA path exactly
        # as-is for training and for the editor, and no parameter is added, so
        # checkpoints are completely unaffected.
        #
        # last_attn_map is (B, N, N) with store_attn_head_mean=True (the DiffSeg
        # recipe aggregates over heads anyway) or (B, heads, N, N) without it. The
        # head-mean matters: N = F*G, so a per-head map is heads× the memory of an
        # already-quadratic tensor (G=22, F=196 → N=4312 → ~600 MB per layer per
        # head-full map vs ~74 MB head-meaned).
        self.store_attn = False
        self.store_attn_head_mean = True
        self.last_attn_map = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # (B, 1, 1, N) bool mask: False positions are padding and get -inf
            attn_mask = mask[:, None, None, :]

        if self.store_attn:
            # Explicit softmax: SDPA cannot return the probabilities themselves.
            logits = q @ k.transpose(-2, -1) * self.scale          # (B, h, N, N)
            if attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
            self.last_attn_map = (attn.mean(dim=1) if self.store_attn_head_mean
                                  else attn).detach()
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        else:
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 dropout_p=dropout_p)
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
                 ff_mult: int = 4, dropout: float = 0.0, attn_sink: bool = False):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.self_attn  = SelfAttention(dim, num_heads, dropout)
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout,
                                         use_sink=attn_sink)
        self.norm3      = nn.LayerNorm(dim)
        self.ff         = FeedForward(dim, ff_mult, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor, store_attn: bool = False,
                mask: torch.Tensor | None = None,
                context_mask: torch.Tensor | None = None,
                compute_entropy: bool = False):
        mods = self.adaLN_modulation(t_emb)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = mods.chunk(9, dim=-1)
        s1, b1, g1 = s1[:, None], b1[:, None], g1[:, None]
        s2, b2, g2 = s2[:, None], b2[:, None], g2[:, None]
        s3, b3, g3 = s3[:, None], b3[:, None], g3[:, None]

        # adaLN-zero: gate (g) is zero-initialised so each block is an identity
        # map at the start of training (DiT §3.2).
        x = x + g1 * self.self_attn(self.norm1(x) * (1 + s1) + b1, mask=mask)
        x = x + g2 * self.cross_attn(self.norm2(x) * (1 + s2) + b2, context,
                                     store_attn=store_attn, context_mask=context_mask,
                                     compute_entropy=compute_entropy)
        x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        return x
