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
unconditional prediction with the masked multi-edit SEGA estimate (proposal Eq. 1) and
hard-inpaints unedited frames from the stored sequence, so only the targeted
(frame, body-part group) cells move.

Edit space (`MotionEditor.edit_space`, resolved from the checkpoint's `predict_type`)
selects which quantity Stages 2–3 do their arithmetic in — see
docs/AttentionGrounding_Options.md §5.3:

  "eps" — the historical path: SEGA composes noise estimates, and the result is mapped
          back with predict_x0_from_eps, i.e. the guidance term is multiplied by
          1/√ᾱ_t (×34 at t=980, unbounded as t→T). Hence `guidance_alpha_floor`.
  "x0"  — the x0-native path: SEGA composes clean-signal predictions directly, so no
          1/√ᾱ_t appears and the highest-noise steps need no gating. Per step this is
          the *same estimate* written one level up (the conversion is affine, so it
          commutes with the guidance sum); what changes is that an x0 head produces it
          without a division by a vanishing √ᾱ_t, and that ψ/M2 is then a difference of
          clean-motion predictions rather than a √SNR_t-weighted one.

Scope note: the proposal targets the sde-dpm-solver++ recurrence for acceleration.
That is a drop-in replacement for the per-step μ/σ recurrence here and is left as
follow-up; the DDPM form already provides the exact-reconstruction guarantee that
Stages 2–3 rely on. Inversion runs on the full timestep grid for that reason.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from editing import masking

# Default `guidance_alpha_floor` for eps-space editing. Only that space needs one: it is
# the 1/√ᾱ_t amplification of the guidance term that diverges, and the x0-native path
# has no such factor (§5.3), so it defaults to 0.0 = guide at every step.
EPS_GUIDANCE_ALPHA_FLOOR = 0.03


@dataclass
class InversionState:
    """Output of invert(): the noisy trajectory and its edit-friendly noise maps."""
    xs: torch.Tensor          # (T, 1, F, D) — x_t for t in [0, T); xs[0] == x0
    zs: torch.Tensor          # (T, 1, F, D) — z_t for t in [1, T); zs[0] unused


class MotionEditor:
    def __init__(self, model, schedule, device, is_group: bool, edit_space="auto",
                 attn_layers=None, psi_readout="abs"):
        """edit_space : "auto" (default) reads the space off the checkpoint's own
        `predict_type` via NoiseSchedule.resolve_space — an x0-trained checkpoint edits
        x0-natively, an ε-trained one keeps the historical ε-space path, and no caller
        has to know which. "eps"/"x0" force it (the A/B control; see the module
        docstring). Forcing "x0" on an ε head is legal but does NOT remove the 1/√ᾱ_t
        amplification — it only moves it inside the difference — so keep a non-zero
        `guidance_alpha_floor` there.

        attn_layers : block indices the M1 read-out averages over, or None for all of
        them. Resolved by the caller from the checkpoint config
        (`training.grounding.resolve_readout_layers`) for the same reason edit_space is:
        the right value is a property of how the checkpoint was TRAINED, not something
        a user should have to remember. A checkpoint trained with the grounding loss
        supervised only 3 of its 8 blocks, so averaging all 8 dilutes what the read-out
        exists to measure; a checkpoint trained without it has no such key and keeps
        the historical all-blocks behaviour.

        psi_readout : what the ψ/M2 contrast measures — "abs" (the LEDITS++ magnitude
        |x̂0^c − x̂0^ref|, the default and the historical behaviour) or "energy" (the
        SIGNED change in per-group motion energy). It sits here rather than on
        `collect_masks` for the same reason `edit_space` does: it changes what ψ *means*,
        so it must be one value for the whole edit, resolved once from a flag. Measured
        effect inside M1 ∩ M2 at matched mask size: alignment 0.452 → 0.583 with recall
        0.659 → 0.843 (docs/FINDINGS.md "ψ is a mixture"). The default stays "abs"
        deliberately — the gain is measured on mask quality, and the standing MotionFix
        negative it is meant to attack is an EDIT result, so the default flips only after
        the end-to-end comparison.
        """
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.is_group = is_group
        self.edit_space = schedule.resolve_space(edit_space)
        self.attn_layers = attn_layers
        self.psi_readout = psi_readout
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
            # `posterior_mean` takes x̂0, so read the unconditional prediction straight
            # into that space: predict_x0_from_eps for an eps head (exactly the line this
            # replaces), the identity for an x0 head. The old code round-tripped an x0
            # head's output x̂0 → ε → x̂0, which is algebraically an identity but loses
            # ~1.5–3 float digits to cancellation at the high-noise end (§5.3 Stage 1).
            # Space-independent: inversion has no guidance term to place.
            x0_pred = s.to_x0(self.model(xs[t], t_b, context=None), xs[t], t_b)  # uncond
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
                      attn_readout="raw", semantic_idxs_per_edit=None,
                      attn_timesteps=None, psi_timesteps=None, per_step_norm=False):
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
        attn_timesteps / psi_timesteps : per-mask sweeps overriding `timesteps` (M1 and
                               M2 carry their signal at different noise levels — see
                               masking.collect_statistics). None → shared `timesteps`.
        per_step_norm        : weight every swept timestep equally instead of by its
                               magnitude (see masking._normalise_step).

        ψ/M2 is computed in this editor's `edit_space`, so an x0 checkpoint contrasts
        clean-signal predictions (ψ_x0) instead of noise estimates (ψ_ε). Same mask at
        any fixed t — the two differ by the positive scalar √SNR_t, which no percentile
        threshold can see — but a different mixture across the sweep; see
        masking.collect_statistics's `psi_space`.
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
                attn_timesteps=attn_timesteps, psi_timesteps=psi_timesteps,
                per_step_norm=per_step_norm, psi_space=self.edit_space,
                attn_layers=self.attn_layers, psi_readout=self.psi_readout,
            )
            masks.append(masking.build_mask(
                attn_fg, psi_fg, valid_frames, self.is_group,
                lambda_attn=lambda_attn, lambda_noise=lambda_noise,
                mask_mode=mask_mode, llm_group_mask=llm_m,
                group_channels=self.group_channels, feat_dim=self.feat_dim,
            ))
        return masks

    # ── Stage 3 ────────────────────────────────────────────────────────────────
    def resolve_alpha_floor(self, guidance_alpha_floor=None) -> float:
        """The `guidance_alpha_floor` `edit()` will actually use — the space-dependent
        default when None (see `edit`). Exposed so callers can log/print the resolved
        value instead of a bare None."""
        if guidance_alpha_floor is not None:
            return float(guidance_alpha_floor)
        return 0.0 if self.edit_space == "x0" else EPS_GUIDANCE_ALPHA_FLOOR

    @torch.no_grad()
    def _guided_x0(self, x, t_b, edit_contexts, m_channels, scales, guide: bool):
        """One reverse step's masked multi-edit SEGA estimate, returned as x̂0.

        In "eps" space this composes noise estimates and converts once at the end
        (the historical arithmetic, unchanged):

            ε̂  = ε_∅ + Σ_i s_i·M_i⊙(ε_{c_i} − ε_∅)   →   x̂0 = predict_x0_from_eps(ε̂)

        In "x0" space it composes the clean-signal predictions themselves (§5.3):

            x̂0 = x̂0^∅ + Σ_i s_i·M_i⊙(x̂0^{c_i} − x̂0^∅)

        Substituting the ε↔x0 shim into either form gives the other exactly — the
        conversion is affine at fixed (x, t), so guidance commutes with it. The
        difference is conditioning, not algebra: the eps form reaches x̂0 by dividing by
        √ᾱ_t, so it amplifies whatever error the head has by up to 1/√ᾱ_t and needs
        `guidance_alpha_floor`; with an x0 head the x0 form never performs that division.

        `guide=False` (floor gate) returns the unconditional prediction, in both spaces.
        """
        s, space = self.schedule, self.edit_space
        base = s.to_space(self.model(x, t_b, context=None), x, t_b, space)
        out = base
        if guide:
            for ctx, m_ch, scale in zip(edit_contexts, m_channels, scales):
                out_c = s.to_space(self.model(x, t_b, ctx), x, t_b, space)
                out = out + scale * m_ch * (out_c - base)
        # NB the result is NOT clamped — normalised HumanML3D features legitimately
        # reach ~±60 (small-std channels), so any fixed clamp would truncate real values
        # and forfeit the edit-friendly exact-reconstruction guarantee (with scale=0 this
        # loop must reproduce the source). Guidance is bounded by the mask and scale.
        return out if space == "x0" else s.predict_x0_from_eps(x, t_b, out)

    @torch.no_grad()
    def edit(self, state, edit_contexts, masks, scales, show_progress=True,
             guidance_alpha_floor=None):
        """
        Masked multi-edit SEGA denoising with cell-level hard inpainting (proposal Eq. 1),
        run in this editor's `edit_space` — see `_guided_x0` for the two forms:

            ε̂(x_t, c_e) = ε_θ(x_t, ∅) + Σ_i s_e,i · M_i · [ε_θ(x_t, c_e,i) − ε_θ(x_t, ∅)]

        edit_contexts        : list of (1, L, dim) per instruction
        masks                : list of mask dicts from collect_masks (m_channel / edited)
        scales               : list of guidance scales s_e per instruction
        guidance_alpha_floor : apply edit guidance only at steps where √ᾱ_t ≥ floor.
            None (default) resolves per space: EPS_GUIDANCE_ALPHA_FLOOR in eps space,
            0.0 (no gate) in x0 space.
            The gate exists for eps space, where reaching x̂0 divides by √ᾱ_t → 0 at the
            highest-noise steps: guidance there amplifies (ε_c − ε_∅) by an unbounded
            factor and the z-reuse cascades it into divergence (abs values reach
            1e2–1e4). Skipping the ~20 vanishing-α steps (floor 0.03 → 980/1000 active)
            removes the blow-up while leaving edit strength essentially unchanged.
            x0-native guidance has no such factor, so those steps are guided too — and
            they are exactly where an x0-trained model's text conditioning is strongest
            (docs/FINDINGS.md). Pass a non-zero floor to gate them anyway (needed if
            edit_space="x0" is forced onto an eps head). scale=0 reconstructs the source
            exactly regardless, in both spaces.
        Returns the edited motion x̂0 : (1, F, 263), normalised.
        """
        s = self.schedule
        T = s.T
        guidance_alpha_floor = self.resolve_alpha_floor(guidance_alpha_floor)

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

            # Gate guidance off at vanishing-√ᾱ steps (eps space only by default — see
            # the docstring), then take the reverse step reusing the stored z_t.
            guide = bool(s.sqrt_alphas_cumprod[t] >= guidance_alpha_floor)
            x0_pred = self._guided_x0(x, t_b, edit_contexts, m_channels, scales, guide)
            mu = s.posterior_mean(x0_pred, x, t_b)
            sigma = s.posterior_variance[t].clamp(min=1e-20).sqrt()
            x = mu + sigma * state.zs[t]

            # hard inpainting: every non-edited cell ← exact source noised to t-1
            x = torch.where(keep_cell, state.xs[t - 1], x)

        return x
