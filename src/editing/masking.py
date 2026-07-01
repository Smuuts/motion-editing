"""
LEDITS++ Stage 2 — spatiotemporal implicit masking.

Builds, per edit instruction, a binary mask M = M1 ∩ M2 in (frame × group) space:

  M1 (semantic)        — averaged cross-attention over the instruction's content
                         tokens, layers and timesteps; thresholded at a percentile.
                         "Which body-part group at which frame does the edit text
                         attend to?"
  M2 (noise-estimate)  — magnitude of the guidance vector ψ = ε_θ(x_t, c) − ε_θ(x_t, ∅),
                         aggregated per group and thresholded. "Where does the edit
                         actually change the prediction?"

Both masks are accumulated over the stored inversion timesteps x_t, so the mask is
averaged over the whole trajectory rather than read off a single noise level.

Group axis:
  GroupDiT  → G = 7 body-part groups (root + 6), giving true spatiotemporal masks.
  MotionDiT → G = 1 (a frame is masked as a whole; no body-part resolution).

The (F, G) mask is expanded to a per-channel (F, 263) mask via the GROUP_CHANNELS
partition for use in the Eq.-1 guidance, and reduced to a per-frame "edited" flag
for the Stage-3 hard inpainting.
"""

import torch

from model.body_groups import GROUP_CHANNELS, N_GROUPS


def _aggregate_channels_to_groups(per_channel: torch.Tensor, is_group: bool,
                                  group_channels=None) -> torch.Tensor:
    """(F, D) channel quantity → (F, G) per-group mean. G=7 (group) or G=1 (frame).

    group_channels selects the representation's channel partition (263-d humanml3d by
    default, or the 135-d smplh partition passed by the editor); D must match it.
    """
    if not is_group:
        return per_channel.mean(dim=-1, keepdim=True)            # (F, 1)
    if group_channels is None:
        group_channels = GROUP_CHANNELS
    cols = [per_channel[:, ch].mean(dim=-1) for ch in group_channels]
    return torch.stack(cols, dim=-1)                              # (F, G)


def _percentile_threshold(values: torch.Tensor, valid_frames: torch.Tensor,
                          percentile: float) -> torch.Tensor:
    """
    Binarise (F, G) values, keeping entries above the given percentile of the
    distribution over valid (non-padding) frames. percentile=70 keeps the top 30%.
    """
    valid_vals = values[valid_frames].flatten()
    if valid_vals.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    thr = torch.quantile(valid_vals.float(), percentile / 100.0)
    return values >= thr


@torch.no_grad()
def collect_statistics(model, schedule, xs, context_edit, token_idxs,
                       is_group, num_groups=None, timesteps=None, need_attn=True,
                       group_channels=None):
    """
    Sweep the stored inversion sequence and accumulate the raw quantities for M1/M2.

    model         : EMA MotionDiT / GroupDiT (eval mode)
    schedule      : NoiseSchedule (only T is used here)
    xs            : (T, 1, F, 263) stored inversion samples x_t (see inversion.py)
    context_edit  : (1, L, dim) embedding of ONE edit instruction
    token_idxs    : list[int] — content-token columns in [0, L) for this instruction
                    (from text_encoder.token_info). Unused (may be None) when
                    need_attn=False.
    is_group      : True for GroupDiT (G=7), False for MotionDiT (G=1)
    timesteps     : iterable of t to sweep (default: every t in [1, T))
    need_attn     : capture cross-attention for M1. False (e.g. m2_only / llm modes)
                    skips the attention pass entirely and returns attn_fg = zeros.

    Returns (attn_fg, psi_fg), both (F, G):
      attn_fg — mean semantic cross-attention per (frame, group) (zeros if need_attn=False)
      psi_fg  — mean |ε_c − ε_∅| per (frame, group)
    """
    T = schedule.T
    F = xs.shape[2]
    G = (num_groups or N_GROUPS) if is_group else 1
    if timesteps is None:
        timesteps = range(1, T)

    device = context_edit.device
    null_context = None  # model uses its learned null_text_emb
    attn_accum = torch.zeros(F, G, device=device)
    psi_accum  = torch.zeros(F, G, device=device)
    n = 0

    tok = (torch.as_tensor(token_idxs, device=device, dtype=torch.long)
           if need_attn else None)
    for t in timesteps:
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)

        # ε_θ(x_t, c_edit), with attention capture only when M1 is needed.
        eps_c = model(x_t, t_b, context_edit, store_attn=need_attn)
        if need_attn:
            # M1 contribution
            layer_maps = model.get_attn_maps()                # list of (1, h, N, L)
            stacked = torch.stack(layer_maps, dim=0).float()  # (Lyr, 1, h, N, L)
            avg = stacked.mean(dim=(0, 1, 2))                 # (N, L)
            avg = avg[:, tok].mean(dim=-1)                    # (N,) over content tokens
            attn_accum += avg.reshape(F, G)

        # ψ = ε_θ(x_t, c_edit) − ε_θ(x_t, ∅) → M2 contribution
        eps_u = model(x_t, t_b, null_context)
        psi = (eps_c - eps_u)[0].abs()                        # (F, D)
        psi_accum += _aggregate_channels_to_groups(psi, is_group, group_channels)
        n += 1

    return attn_accum / max(n, 1), psi_accum / max(n, 1)


def build_mask(attn_fg, psi_fg, valid_frames, is_group,
               lambda_attn=70.0, lambda_noise=70.0,
               mask_mode="m2_only", llm_group_mask=None,
               group_channels=None, feat_dim=263):
    """
    Build the per-edit (F, G) mask according to `mask_mode`.

    attn_fg, psi_fg : (F, G) accumulated statistics from collect_statistics
    valid_frames    : (F,) bool — True for real frames (excludes padding)
    lambda_*        : percentile thresholds (higher = sparser mask)
    mask_mode       : "none"    — no mask: every (frame, group) cell is editable.
                                  Guidance applies everywhere and nothing is inpainted
                                  (the proposal's "remove the mask entirely" ablation).
                      "m2_only" — M2 alone (no semantic mask). Default while the
                                  implicit attention M1 is not body-part grounded.
                      "m1_only" — M1 alone (semantic cross-attention, no M2 gating).
                                  Useful to test whether attention targets a body part
                                  the source isn't already moving (M2 can't add that).
                      "attn"    — M1 ∩ M2 (implicit cross-attention semantic mask).
                      "llm"     — M_llm ∩ M2, where M_llm is an explicit (F, G) group
                                  mask supplied via `llm_group_mask` (see Phase B).
    llm_group_mask  : (F, G) or (G,) bool — required for mask_mode="llm"; the groups
                      an instruction targets. A (G,) vector is broadcast over frames.

    group_channels  : representation channel partition (263-d default, or 135-d smplh)
    feat_dim        : total feature width D matching group_channels (263 or 135)

    Returns dict with:
      m_group   : (F, G) bool   — final mask, padding frames forced False
      m_channel : (F, D) float  — group mask scattered to feature channels (for Eq. 1)
      edited    : (F,) bool      — frame has any active group (drives hard inpainting)
    """
    valid = valid_frames[:, None]

    # semantic component (M1 / M_llm); absent for "none", "m2_only"
    if mask_mode in ("none", "m2_only"):
        m_sem = None
    elif mask_mode in ("attn", "m1_only"):
        m_sem = _percentile_threshold(attn_fg, valid_frames, lambda_attn)
    elif mask_mode == "llm":
        if llm_group_mask is None:
            raise ValueError("mask_mode='llm' requires llm_group_mask (F, G) or (G,).")
        m_sem = llm_group_mask.to(valid.device, dtype=torch.bool)
        if m_sem.dim() == 1:
            m_sem = m_sem[None, :].expand(valid_frames.shape[0], -1)
    else:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")

    # noise component (M2); absent for "none", "m1_only"
    m2 = (None if mask_mode in ("none", "m1_only")
          else _percentile_threshold(psi_fg, valid_frames, lambda_noise))

    m_group = valid.expand(-1, attn_fg.shape[1]).clone()
    if m_sem is not None:
        m_group = m_group & m_sem
    if m2 is not None:
        m_group = m_group & m2

    return {
        "m_group":   m_group,
        "m_channel": group_mask_to_channels(m_group, is_group, group_channels, feat_dim),
        "edited":    m_group.any(dim=-1),
    }


def group_mask_to_channels(m_group: torch.Tensor, is_group: bool,
                           group_channels=None, feat_dim=263) -> torch.Tensor:
    """(F, G) bool group mask → (F, D) float channel mask via the channel partition.

    Defaults to the 263-d humanml3d partition; the editor passes the model's own
    group_channels + feat_dim (e.g. 135-d smplh).
    """
    F = m_group.shape[0]
    if not is_group:
        return m_group.float().expand(F, feat_dim)           # G=1 → broadcast
    if group_channels is None:
        group_channels = GROUP_CHANNELS
    out = torch.zeros(F, feat_dim, device=m_group.device)
    for g, ch in enumerate(group_channels):
        out[:, ch] = m_group[:, g : g + 1].float()
    return out
