"""
Stage 3's masked multi-edit SEGA estimate, and the guidance gate it needs in eps space.
"""

import torch

# Default `guidance_alpha_floor` for eps-space editing. Only that space needs one: it is
# the 1/sqrt(alpha_bar_t) amplification of the guidance term that diverges, and the
# x0-native path has no such factor, so it defaults to 0.0 = guide at every step.
EPS_GUIDANCE_ALPHA_FLOOR = 0.03


def resolve_alpha_floor(edit_space: str, guidance_alpha_floor=None) -> float:
    """The floor an edit will actually use — the space-dependent default when None."""
    if guidance_alpha_floor is not None:
        return float(guidance_alpha_floor)
    return 0.0 if edit_space == "x0" else EPS_GUIDANCE_ALPHA_FLOOR


@torch.no_grad()
def guided_x0(model, schedule, edit_space, x, t_b, edit_contexts, m_channels, scales,
              guide: bool):
    """One reverse step's masked multi-edit SEGA estimate, returned as x0_hat.

    In "eps" space this composes noise estimates and converts once at the end (the
    historical arithmetic, unchanged):

        eps_hat = eps_null + sum_i s_i * M_i * (eps_{c_i} - eps_null)
        x0_hat  = predict_x0_from_eps(eps_hat)

    In "x0" space it composes the clean-signal predictions themselves:

        x0_hat = x0_null + sum_i s_i * M_i * (x0_{c_i} - x0_null)

    Substituting the eps<->x0 shim into either form gives the other exactly — the
    conversion is affine at fixed (x, t), so guidance commutes with it. The difference is
    conditioning, not algebra: the eps form reaches x0_hat by dividing by
    sqrt(alpha_bar_t), so it amplifies whatever error the head has by up to
    1/sqrt(alpha_bar_t) and needs a guidance floor; with an x0 head the x0 form never
    performs that division.

    `guide=False` (the floor gate) returns the unconditional prediction, in both spaces.
    """
    base = schedule.to_space(model(x, t_b, context=None), x, t_b, edit_space)
    out = base
    if guide:
        for ctx, m_ch, scale in zip(edit_contexts, m_channels, scales):
            # A zero scale contributes 0 * m * (out_c - base) = 0, so the conditional
            # forward that produces out_c is pure waste — and scale 0 is not a corner
            # case here, it is the source-reconstruction calibration every sweep runs.
            # Skipping it saves one model call per step, i.e. ~11 % of a MotionFix run
            # (999 of ~9,100 forwards per clip at four scales). Output is unchanged for
            # any finite out_c; a NaN there would already have poisoned `base`.
            if scale == 0:
                continue
            out_c = schedule.to_space(model(x, t_b, ctx), x, t_b, edit_space)
            out = out + scale * m_ch * (out_c - base)
    # NB the result is NOT clamped — normalised HumanML3D features legitimately reach
    # ~+-60 (small-std channels), so any fixed clamp would truncate real values and
    # forfeit the edit-friendly exact-reconstruction guarantee (with scale=0 this loop
    # must reproduce the source). Guidance is bounded by the mask and the scale.
    return out if edit_space == "x0" else schedule.predict_x0_from_eps(x, t_b, out)
