"""
LEDITS++ Stage 1 (edit-friendly inversion) and Stage 3 (masked SEGA denoising).

This implements the *DDPM-form* edit-friendly inversion of Huberman-Spiegelglas
et al. (2024) on this project's NoiseSchedule:

  1. Build an independent noisy sequence  x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε_t,
     with ε_t drawn i.i.d. per timestep (the "edit-friendly" noise space).
  2. Extract noise maps  z_t = (x_{t−1} − μ_θ(x_t)) / σ_t,
     where μ_θ is the DDPM posterior mean using the model's *unconditional*
     ε_θ(x_t, ∅) and σ_t = √(posterior_variance_t).

Re-running the reverse process with the stored {z_t} and unchanged conditioning
reproduces x0 exactly (perfect reconstruction). Stage 3 instead replaces the
unconditional ε with the masked multi-edit SEGA estimate (proposal Eq. 1) and
hard-inpaints unedited frames from the stored sequence, so only the targeted
(frame, body-part group) cells move.

Scope note: the proposal targets the sde-dpm-solver++ recurrence for acceleration.
That is a drop-in replacement for the per-step μ/σ recurrence here and is left as
follow-up; the DDPM form already provides the exact-reconstruction guarantee that
Stages 2–3 rely on. Inversion runs on the full timestep grid for that reason.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from editing import masking


@dataclass
class InversionState:
    """Output of invert(): the noisy trajectory and its edit-friendly noise maps."""
    xs: torch.Tensor          # (T, 1, F, D) — x_t for t in [0, T); xs[0] == x0
    zs: torch.Tensor          # (T, 1, F, D) — z_t for t in [1, T); zs[0] unused


class MotionEditor:
    def __init__(self, model, schedule, device, is_group: bool):
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.is_group = is_group
        # Feature layout is representation-specific: 263 (humanml3d) or 135 (smplh).
        # GroupDiT exposes both; fall back to 263 for legacy flat MotionDiT.
        self.feat_dim = getattr(model, "input_dim", 263)
        self.group_channels = getattr(model, "group_channels", None)
        self.model.eval()

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def invert(self, x0: torch.Tensor, show_progress: bool = True) -> InversionState:
        """
        x0 : (1, F, D) normalised source motion (same space the model trained in),
             D = 263 (humanml3d) or 135 (smplh).
        Returns the InversionState consumed by collect_masks() and edit().
        """
        s = self.schedule
        T = s.T
        F = x0.shape[1]
        D = x0.shape[2]
        x0 = x0.to(self.device)

        # 1. independent noisy sequence x_t (xs[0] = x0)
        xs = torch.empty(T, 1, F, D, device=self.device)
        xs[0] = x0
        for t in range(1, T):
            sqrt_acp   = s.sqrt_alphas_cumprod[t]
            sqrt_omacp = s.sqrt_one_minus_alphas_cumprod[t]
            xs[t] = sqrt_acp * x0 + sqrt_omacp * torch.randn_like(x0)

        # 2. edit-friendly noise maps z_t (for t in [1, T))
        zs = torch.zeros(T, 1, F, D, device=self.device)
        it = range(T - 1, 0, -1)
        it = tqdm(it, desc="Inversion", leave=False) if show_progress else it
        for t in it:
            t_b = torch.full((1,), t, device=self.device, dtype=torch.long)
            # to_eps: identity for an eps-head, exact conversion for an x0-head.
            eps = s.to_eps(self.model(xs[t], t_b, context=None), xs[t], t_b)  # uncond
            x0_pred = s.predict_x0_from_eps(xs[t], t_b, eps)
            mu = s.posterior_mean(x0_pred, xs[t], t_b)
            sigma = s.posterior_variance[t].clamp(min=1e-20).sqrt()
            zs[t] = (xs[t - 1] - mu) / sigma

        return InversionState(xs=xs, zs=zs)

    # ── Stage 2 (delegates to masking.py) ──────────────────────────────────────
    @torch.no_grad()
    def collect_masks(self, state, edit_contexts, token_idxs_per_edit, valid_frames,
                      lambda_attn=70.0, lambda_noise=70.0, timesteps=None,
                      mask_mode="m2_only", llm_group_masks=None,
                      context_source=None, m2_group_norm=False,
                      attn_readout="raw", semantic_idxs_per_edit=None):
        """
        Build one mask dict per edit instruction (see masking.build_mask).

        edit_contexts        : list of (1, L, dim) instruction embeddings
        token_idxs_per_edit  : list of content-token index lists (text_encoder.token_info).
                               Only consumed when mask_mode="attn"; may be None otherwise.
        valid_frames         : (F,) bool — real (non-padding) frames
        mask_mode            : "m2_only" (default) | "attn" | "groups" (see build_mask).
        llm_group_masks      : list of (F, G)/(G,) bool group masks, one per edit
                               (required for mask_mode="groups").
        context_source       : (1, L, dim) SOURCE caption embedding used as the ψ
                               reference instead of the null embedding (DiffEdit-style;
                               see masking.collect_statistics). None → null reference.
        m2_group_norm        : normalise ψ per group by source motion energy.
        attn_readout         : M1 per-cell readout of the attention maps — "raw"
                               (original) | "renorm" | "spatial" | "renorm_spatial"
                               (see masking.collect_statistics).
        semantic_idxs_per_edit : stop-word-filtered token index lists (one per edit,
                               masking.semantic_token_subset); used by the non-"raw"
                               readouts. Defaults to token_idxs_per_edit.
        """
        need_attn = mask_mode in ("attn", "m1_only")
        if token_idxs_per_edit is None:
            token_idxs_per_edit = [None] * len(edit_contexts)
        if llm_group_masks is None:
            llm_group_masks = [None] * len(edit_contexts)
        if semantic_idxs_per_edit is None:
            semantic_idxs_per_edit = [None] * len(edit_contexts)

        valid_frames = valid_frames.to(self.device)
        masks = []
        for ctx, tok, llm_m, sem in zip(edit_contexts, token_idxs_per_edit,
                                        llm_group_masks, semantic_idxs_per_edit):
            attn_fg, psi_fg = masking.collect_statistics(
                self.model, self.schedule, state.xs, ctx, tok,
                is_group=self.is_group, timesteps=timesteps, need_attn=need_attn,
                group_channels=self.group_channels,
                context_ref=context_source, psi_group_norm=m2_group_norm,
                valid_frames=valid_frames,
                attn_readout=attn_readout, semantic_idxs=sem,
            )
            masks.append(masking.build_mask(
                attn_fg, psi_fg, valid_frames, self.is_group,
                lambda_attn=lambda_attn, lambda_noise=lambda_noise,
                mask_mode=mask_mode, llm_group_mask=llm_m,
                group_channels=self.group_channels, feat_dim=self.feat_dim,
            ))
        return masks

    # ── Stage 3 ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def edit(self, state, edit_contexts, masks, scales, show_progress=True,
             guidance_alpha_floor=0.03):
        """
        Masked multi-edit SEGA denoising with cell-level hard inpainting (proposal Eq. 1):

            ε̂(x_t, c_e) = ε_θ(x_t, ∅) + Σ_i s_e,i · M_i · [ε_θ(x_t, c_e,i) − ε_θ(x_t, ∅)]

        edit_contexts        : list of (1, L, dim) per instruction
        masks                : list of mask dicts from collect_masks (m_channel / edited)
        scales               : list of guidance scales s_e per instruction
        guidance_alpha_floor : apply edit guidance only at steps where √ᾱ_t ≥ floor.
            predict_x0_from_eps divides by √ᾱ_t, which → 0 at the highest-noise steps;
            adding guidance there amplifies (ε_c − ε_∅) by an unbounded factor and the
            z-reuse cascades it into divergence (abs values reach 1e2–1e4). Skipping the
            ~20 vanishing-α steps (floor 0.03 → 980/1000 active) removes the blow-up
            while leaving edit strength essentially unchanged. floor=0 reproduces the
            old unbounded behaviour. scale=0 reconstructs the source exactly regardless.
        Returns the edited motion x̂0 : (1, F, 263), normalised.
        """
        s = self.schedule
        T = s.T

        m_channels = [m["m_channel"].to(self.device) for m in masks]     # each (F, 263)
        # Union of all edits' per-channel masks → the only cells allowed to change.
        edit_cells = torch.zeros_like(m_channels[0])                     # (F, 263)
        for m_ch in m_channels:
            edit_cells = torch.maximum(edit_cells, m_ch)
        # Hard inpainting is done at the (frame, channel) CELL level, not per frame:
        # every non-edited cell is pinned to the exact source value at each step, so
        # unedited body-part groups inside an otherwise-edited frame stay put. A purely
        # frame-level rule would let the whole 263-vector of an "edited" frame drift —
        # including root velocity, which FK integrates into wild global motion.
        keep_cell = (edit_cells == 0)[None]                             # (1, F, 263) bool

        x = state.xs[T - 1].clone()
        it = range(T - 1, 0, -1)
        it = tqdm(it, desc="Editing", leave=False) if show_progress else it
        for t in it:
            t_b = torch.full((1,), t, device=self.device, dtype=torch.long)

            # to_eps: identity for an eps-head, exact conversion for an x0-head. Note
            # this keeps SEGA in eps space, so the 1/√ᾱ_t amplification (and hence
            # guidance_alpha_floor) still applies under x0 — the x0-native Stage 3 of
            # docs/AttentionGrounding_Options.md §5.3, which removes both, is not this.
            eps_uncond = s.to_eps(self.model(x, t_b, context=None), x, t_b)
            eps_hat = eps_uncond
            # Gate guidance off at vanishing-√ᾱ steps to avoid x0-space divergence.
            if s.sqrt_alphas_cumprod[t] >= guidance_alpha_floor:
                for ctx, m_ch, scale in zip(edit_contexts, m_channels, scales):
                    eps_c   = s.to_eps(self.model(x, t_b, ctx), x, t_b)
                    eps_hat = eps_hat + scale * m_ch * (eps_c - eps_uncond)

            # reverse step reusing the stored edit-friendly noise z_t.
            # NB: x0_pred is NOT clamped — normalised HumanML3D features legitimately
            # reach ~±60 (small-std channels), so any fixed clamp would truncate real
            # values and forfeit the edit-friendly exact-reconstruction guarantee
            # (with scale=0 this loop must reproduce the source). Guidance is already
            # bounded by the mask and per-edit scale.
            x0_pred = s.predict_x0_from_eps(x, t_b, eps_hat)
            mu = s.posterior_mean(x0_pred, x, t_b)
            sigma = s.posterior_variance[t].clamp(min=1e-20).sqrt()
            x = mu + sigma * state.zs[t]

            # hard inpainting: every non-edited cell ← exact source noised to t-1
            x = torch.where(keep_cell, state.xs[t - 1], x)

        return x
