"""
LEDITS++ Stage 2 — spatiotemporal implicit masking.

Builds, per edit instruction, a binary mask M = M1 ∩ M2 in (frame × group) space:

  M1 (semantic)        — averaged cross-attention over the instruction's content
                         tokens, layers and timesteps; thresholded at a percentile.
                         "Which body-part group at which frame does the edit text
                         attend to?" The raw readout is known to be sink-dominated:
                         CrossAttention has no key-padding mask, so the ~L−|words|
                         padding columns (zero T5 embeddings → zero logits) plus EOS
                         absorb most softmax mass, and the per-cell content readout
                         is modulated by that sink denominator. `attn_readout`
                         selects sink-corrected readouts (see collect_statistics):
                         "renorm" (drop sink columns, renormalise — Attend-and-
                         Excite, Chefer et al. 2023) and/or "spatial" (per-token
                         spatial profile — DAAM, Tang et al. 2023).
  M2 (noise-estimate)  — magnitude of the guidance vector ψ = f_θ(x_t, c) − f_θ(x_t, ref),
                         aggregated per group and thresholded. "Where does the edit
                         actually change the prediction?" f_θ is read in the editor's
                         space (`psi_space`): noise estimates ε_θ for an ε checkpoint,
                         clean-signal predictions x̂0 for an x0 one, where ψ becomes the
                         directly interpretable "how differently does the model
                         reconstruct the clean clip when told the instruction"
                         (docs/AttentionGrounding_Options.md §5.3). The reference ref is the
                         learned null embedding by default; passing the SOURCE
                         caption's embedding instead (DiffEdit's "reference text",
                         Couairon et al., ICLR 2023) cancels the part of ψ both
                         conditionings agree on — i.e. the source's own dynamics —
                         which is the known failure mode of the null-referenced mask
                         (it fires on whatever the source already moves). Optionally
                         ψ is additionally normalised per group by the source's own
                         motion energy (psi_group_norm) for the same reason.

Both masks are accumulated over the stored inversion timesteps x_t, so the mask is
averaged over the whole trajectory rather than read off a single noise level. That
average is a raw magnitude sum by default, so an evenly-spaced sweep is NOT an even
average — and M1's and M2's signals do not live at the same noise levels anyway. See
`per_step_norm` / `attn_timesteps` / `psi_timesteps` in collect_statistics.

Group axis:
  GroupDiT  → G = 7 body-part groups (root + 6), giving true spatiotemporal masks.
  MotionDiT → G = 1 (a frame is masked as a whole; no body-part resolution).

The (F, G) mask is expanded to a per-channel (F, 263) mask via the GROUP_CHANNELS
partition for use in the Eq.-1 guidance, and reduced to a per-frame "edited" flag
for the Stage-3 hard inpainting.
"""

import torch

from model.body_groups import GROUP_CHANNELS, N_GROUPS

# Function words + generic caption vocabulary that carry no body-part/action
# semantics for mask purposes.
_STOP_WORDS = {
    "a", "an", "the",
    "person", "man", "woman", "human", "someone",
    "their", "they", "them", "his", "her", "its",
    "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "by", "for", "with",
    "and", "or", "but",
    "up", "down", "forward", "backward", "side", "out",
    "slowly", "quickly", "slightly",
}


def semantic_token_subset(idxs: list[int], labels: list[str]) -> list[int]:
    """Stop-word-filtered subset of content-token positions (falls back to all
    content tokens if the filter would remove everything)."""
    sem = [i for i, lbl in zip(idxs, labels) if lbl.lower() not in _STOP_WORDS]
    return sem or list(idxs)


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


def _group_aggregation_matrix(group_channels, device) -> torch.Tensor:
    """(D, G) matrix such that `per_channel @ matrix == per-group mean` for each
    group's channels — lets collect_statistics's hot loop replace one Python-level
    list comprehension per timestep with a single matmul."""
    D = sum(len(ch) for ch in group_channels)
    mat = torch.zeros(D, len(group_channels), device=device)
    for g, ch in enumerate(group_channels):
        mat[ch, g] = 1.0 / len(ch)
    return mat


def _group_motion_energy(x0: torch.Tensor, valid_frames, agg_matrix: torch.Tensor,
                         floor_frac: float = 0.25) -> torch.Tensor:
    """
    (G,) mean |Δx0| per group over valid frames — how much each body-part group
    already moves in the source. Used to discount ψ for groups whose large noise
    difference is explained by source dynamics rather than by the instruction.

    Floored at floor_frac·mean(energy) so a truly static group doesn't blow up
    to an unbounded ψ boost (dividing by ~0 would hand the mask to *any* static
    group regardless of the instruction).
    """
    if valid_frames is not None:
        x0 = x0[valid_frames]
    diff = (x0[1:] - x0[:-1]).abs().mean(dim=0)      # (D,) per-channel motion energy
    energy = diff @ agg_matrix                        # (G,) per-group mean
    return energy.clamp(min=floor_frac * energy.mean())


def _attn_readout_value(avg: torch.Tensor, tok: torch.Tensor, sem: torch.Tensor,
                        readout: str) -> torch.Tensor:
    """
    (N, L) layer/head-averaged attention map → (N,) per-cell M1 value.

    tok — all content-token columns (words; excludes BOS/EOS/padding).
    sem — stop-word-filtered subset of tok.
    See collect_statistics's docstring for the readout definitions. Note "renorm"
    is only informative when sem ⊊ tok (otherwise the ratio is constant 1 — the
    semantic_token_subset fallback guarantees sem is never empty, not proper).
    """
    if readout == "raw":
        return avg[:, tok].mean(dim=-1)
    if readout == "renorm":
        return avg[:, sem].sum(dim=-1) / avg[:, tok].sum(dim=-1).clamp_min(1e-12)
    if readout == "spatial":
        cols = avg[:, sem]                                       # (N, S)
        cols = cols / cols.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return cols.mean(dim=-1)
    if readout == "renorm_spatial":
        rows = avg[:, sem] / avg[:, tok].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rows = rows / rows.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return rows.mean(dim=-1)
    raise ValueError(f"unknown attn_readout {readout!r}")


def _normalise_step(values: torch.Tensor, valid_frames) -> torch.Tensor:
    """
    Scale one timestep's (F, G) map to unit mean over valid frames.

    The sweep accumulates raw magnitudes, and both quantities' scale varies strongly
    with t (measured on an x0 checkpoint: M1 draws ~48% of its total from t ≥ 750,
    ψ ~68% from t < 250 — see docs/FINDINGS.md). So an evenly-spaced grid still
    produces a magnitude-weighted average, in which a handful of large-scale steps
    decide the mask. Dividing each step by its own mean makes every swept t
    contribute equally, which is what the even grid was meant to express. Relative
    structure within a step — the only thing the percentile threshold reads — is
    untouched.
    """
    cells = values[valid_frames] if valid_frames is not None else values
    scale = cells.mean().clamp_min(1e-12)
    return values / scale


def _resolve_sweep(timesteps, override, T):
    """Timestep list for one mask: its own override if given, else the shared sweep."""
    ts = override if override is not None else timesteps
    return list(range(1, T)) if ts is None else list(ts)


def build_sweep(num_steps, T, lo=1, hi=None):
    """
    `num_steps` evenly-spaced timesteps inside [lo, hi] (default: the whole
    trajectory, i.e. exactly the historical `linspace(1, T-1, num_steps)`).

    Narrowing the window *resamples within it* rather than discarding steps, so a
    10%-wide window is swept as densely as the full range — which matters because the
    windows that carry the most signal are also the ones a uniform grid samples most
    thinly (docs/FINDINGS.md).
    """
    hi = T - 1 if hi is None else min(hi, T - 1)
    lo = max(1, lo)
    if lo > hi:
        raise ValueError(f"empty timestep window [{lo}, {hi}]")
    return torch.linspace(lo, hi, num_steps).long().tolist()


def percentile_threshold(values: torch.Tensor, valid_frames: torch.Tensor,
                         percentile: float) -> torch.Tensor:
    """
    Binarise (F, G) values, keeping entries above the given percentile of the
    distribution over valid (non-padding) frames. percentile=70 keeps the top 30%.

    Public because the probes binarise candidate mask maps of their own (Option 6's
    generation-space divergence, say) and their alignment numbers are only comparable to
    the editor's if the mask is cut the same way.
    """
    valid_vals = values[valid_frames].flatten()
    if valid_vals.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    thr = torch.quantile(valid_vals.float(), percentile / 100.0)
    return values >= thr


@torch.no_grad()
def collect_statistics(model, schedule, xs, context_edit, token_idxs,
                       is_group, timesteps=None, need_attn=True,
                       group_channels=None, context_ref=None, psi_group_norm=False,
                       valid_frames=None, attn_readout="raw", semantic_idxs=None,
                       attn_timesteps=None, psi_timesteps=None, per_step_norm=False,
                       psi_space=None):
    """
    Sweep the stored inversion sequence and accumulate the raw quantities for M1/M2.

    model         : EMA MotionDiT / GroupDiT (eval mode)
    schedule      : NoiseSchedule (T, and the ψ space — see psi_space)
    xs            : (T, 1, F, 263) stored inversion samples x_t (see inversion.py)
    context_edit  : (1, L, dim) embedding of ONE edit instruction
    token_idxs    : list[int] — content-token columns in [0, L) for this instruction
                    (from text_encoder.token_info). Unused (may be None) when
                    need_attn=False.
    is_group      : True for GroupDiT (G=7), False for MotionDiT (G=1)
    timesteps     : iterable of t to sweep (default: every t in [1, T))
    need_attn     : capture cross-attention for M1. False (e.g. m2_only / llm modes)
                    skips the attention pass entirely and returns attn_fg = zeros.
    context_ref   : (1, L, dim) reference embedding for the ψ contrast, or None for
                    the model's learned null embedding (the original LEDITS++/SEGA
                    form). Pass the SOURCE caption's embedding to cancel
                    source-dynamics-driven ψ (DiffEdit-style reference text).
    psi_group_norm: divide ψ per group by the source's per-group motion energy
                    (see _group_motion_energy) before returning. No-op for G=1.
    valid_frames  : (F,) bool — only consulted by psi_group_norm's energy estimate.
    attn_readout  : how the per-cell M1 value is read off the (N, L) attention map
                    (softmax rows include the padding/EOS sink columns — see module
                    docstring):
                    "raw"     — mean attention mass on the content tokens (original).
                    "renorm"  — semantic tokens' share of the *content-token* mass
                                per cell: drops the pad/EOS sink from the denominator
                                (Attend-and-Excite-style re-softmax).
                    "spatial" — each semantic token's map normalised over cells
                                first (its spatial profile, DAAM-style), then
                                averaged: a token holding little total mass still
                                votes with its full spatial distribution.
                    "renorm_spatial" — renorm rows, then spatial-normalise columns.
    semantic_idxs : stop-word-filtered subset of token_idxs (semantic_token_subset).
                    Required for the non-"raw" readouts; defaults to token_idxs.
    attn_timesteps: sweep for M1 only, overriding `timesteps`. M1 and M2 are read off
                    the same trajectory but their signal does not live at the same
                    noise levels — on an x0 checkpoint M1's instruction-sensitivity
                    strengthens monotonically toward high t, while ψ's is flat and its
                    magnitude is concentrated at low t (docs/FINDINGS.md). A single
                    shared sweep cannot serve both; these two arguments let each mask
                    be read where its own signal is. None → use `timesteps`.
    psi_timesteps : sweep for M2 only, overriding `timesteps`. None → use `timesteps`.
    per_step_norm : scale each timestep's map to unit mean before accumulating, so the
                    returned average weights every swept t equally instead of by its
                    magnitude (see _normalise_step). Default False = historical
                    behaviour.
    psi_space     : space the ψ contrast is taken in — "eps" (ψ_ε = |ε_c − ε_ref|),
                    "x0" (ψ_x0 = |x̂0^c − x̂0^ref|), or None/"auto" = the checkpoint's own
                    predict_type, which is how the editor passes it. The two differ by
                    the positive scalar √SNR_t:

                        ψ_ε(t) = √SNR_t · ψ_x0(t)

                    so AT A FIXED t they rank cells identically and give the same
                    percentile mask — reading an ε checkpoint's ψ in x0 space is not a
                    fix, and the measured per-cell instruction-invariance survives any
                    such rescaling. What differs is the MIXTURE across the sweep: those
                    weights run 70.7 → 0.04 over the default linspace(1, 999, 40) grid,
                    where t=1 alone carries 46% of ψ_ε's total and all t ≥ 500 together
                    5.6%. So ψ_ε is effectively a low-noise readout, while ψ_x0 weights
                    every swept t by its own clean-signal displacement — the regime where
                    an x0-trained model's text conditioning is strongest
                    (docs/AttentionGrounding_Options.md §5.3).

    Returns (attn_fg, psi_fg), both (F, G):
      attn_fg — mean semantic cross-attention per (frame, group) (zeros if need_attn=False)
      psi_fg  — mean |f_θ(x_t, c) − f_θ(x_t, ref)| per (frame, group), in psi_space

    Only one forward pass per (t, conditioning) is ever run: a t needed by both masks
    shares its conditional pass, and the reference pass is skipped for t that only M1
    needs — so splitting the sweeps never costs more compute than the union of them.
    """
    T = schedule.T
    F = xs.shape[2]
    # G = number of token-axis cells. The model's own channel partition is the source
    # of truth (7 body-part groups OR 22 per-joint tokens); grouped callers always pass
    # group_channels, so derive G from it. N_GROUPS only backstops a group_channels=None
    # grouped call, which no current caller makes.
    if not is_group:
        G = 1
    else:
        G = len(group_channels) if group_channels is not None else N_GROUPS
    # Per-mask sweeps. They coincide unless a caller overrides one of them, in which
    # case the loop runs over the union and each accumulator only takes the steps it
    # asked for.
    m1_ts = set(_resolve_sweep(timesteps, attn_timesteps, T)) if need_attn else set()
    m2_ts = set(_resolve_sweep(timesteps, psi_timesteps, T))
    sweep = sorted(m1_ts | m2_ts)
    psi_space = schedule.resolve_space(psi_space)

    device = context_edit.device
    # ψ reference: source-caption embedding if given, else None → the model falls
    # back to its learned null_text_emb (original behaviour).
    attn_accum = torch.zeros(F, G, device=device)
    psi_accum  = torch.zeros(F, G, device=device)
    n_attn = n_psi = 0

    tok = (torch.as_tensor(token_idxs, device=device, dtype=torch.long)
           if need_attn else None)
    sem = (torch.as_tensor(semantic_idxs if semantic_idxs else token_idxs,
                           device=device, dtype=torch.long)
           if need_attn else None)
    # Precompute the channel->group aggregation once (it's invariant across the
    # sweep) instead of rebuilding it from group_channels every iteration.
    agg_matrix = (_group_aggregation_matrix(group_channels or GROUP_CHANNELS, device)
                  if is_group else None)
    for t in sweep:
        want_attn, want_psi = t in m1_ts, t in m2_ts
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)

        # f_θ(x_t, c_edit), with attention capture only when M1 wants this step.
        out_c = model(x_t, t_b, context_edit, store_attn=want_attn)
        if want_attn:
            # M1 contribution
            layer_maps = model.get_attn_maps()                # list of (1, h, N, L)
            stacked = torch.stack(layer_maps, dim=0).float()  # (Lyr, 1, h, N, L)
            avg = stacked.mean(dim=(0, 1, 2))                 # (N, L)
            step = _attn_readout_value(avg, tok, sem, attn_readout).reshape(F, G)
            attn_accum += _normalise_step(step, valid_frames) if per_step_norm else step
            n_attn += 1

        if want_psi:
            # ψ = f_θ(x_t, c_edit) − f_θ(x_t, ref) → M2 contribution, read in psi_space:
            # a contrast of noise estimates ("eps") or of clean-motion predictions ("x0").
            f_c = schedule.to_space(out_c, x_t, t_b, psi_space)
            f_r = schedule.to_space(model(x_t, t_b, context_ref), x_t, t_b, psi_space)
            psi = (f_c - f_r)[0].abs()                        # (F, D)
            step = psi @ agg_matrix if is_group else psi.mean(dim=-1, keepdim=True)
            psi_accum += _normalise_step(step, valid_frames) if per_step_norm else step
            n_psi += 1

    psi_fg = psi_accum / max(n_psi, 1)
    if psi_group_norm and is_group:
        energy = _group_motion_energy(xs[0][0].to(device), valid_frames, agg_matrix)
        psi_fg = psi_fg / energy[None, :]
    return attn_accum / max(n_attn, 1), psi_fg


# mask_mode -> (semantic source | None, use_m2). The two mask components are
# independent axes; encoding them as a lookup keeps adding/auditing a mode a
# one-line change instead of touching two parallel if/elif chains.
_MASK_MODE_COMPONENTS = {
    "none":     (None,     False),
    "m2_only":  (None,     True),
    "m1_only":  ("attn",   False),
    "attn":     ("attn",   True),
    "groups":   ("groups", False),
    "temporal": (None,     True),   # frame-level "edit these frames"; see build_mask
}


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
                      "groups"  — M_user alone (no M2 gating): the user-specified
                                  body-part groups (supplied via `llm_group_mask`) are
                                  edited in every valid frame — full temporal coverage
                                  of the named groups rather than restricting to frames
                                  M2 judges as already changing.
                      "temporal"— frame-level "edit these frames": threshold ψ
                                  marginalised over groups to pick the active frames and
                                  edit EVERY body-part group within them (spatially
                                  permissive). Rationale: the group axis is not reliably
                                  instruction-grounded (see docs/FINDINGS.md), so this
                                  trusts only the temporal signal — which IS reliable —
                                  and never freezes the body part that should change (a
                                  wrong (F,G) mask can inpaint the target region back to
                                  source). Optionally ∩ `llm_group_mask` to also restrict
                                  WHICH parts. `lambda_noise` sets the frame percentile.
    llm_group_mask  : (F, G) or (G,) bool — required for mask_mode="groups"; the
                      groups an instruction targets (also optionally intersected in
                      "temporal"). A (G,) vector is broadcast over frames.

    group_channels  : representation channel partition (263-d default, or 135-d smplh)
    feat_dim        : total feature width D matching group_channels (263 or 135)

    Returns dict with:
      m_group   : (F, G) bool   — final mask, padding frames forced False
      m_channel : (F, D) float  — group mask scattered to feature channels (for Eq. 1)
      edited    : (F,) bool      — frame has any active group (drives hard inpainting)
    """
    if mask_mode not in _MASK_MODE_COMPONENTS:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    semantic_source, use_m2 = _MASK_MODE_COMPONENTS[mask_mode]

    valid = valid_frames[:, None]

    # Frame-level "edit these frames" mask (spatially permissive). Handled before the
    # generic (F,G)-cell path because it thresholds a per-FRAME activity score, not
    # per-cell values. ψ summed over groups is the source's "where is there change"
    # signal aggregated across the body; the top (100−lambda_noise)% of frames are
    # edited across ALL groups. See the docstring for why we trust only this axis.
    if mask_mode == "temporal":
        activity = psi_fg.sum(dim=1, keepdim=True)                    # (F, 1)
        active   = percentile_threshold(activity, valid_frames, lambda_noise)  # (F, 1) bool
        m_group  = (valid & active).expand(-1, psi_fg.shape[1]).clone()
        if llm_group_mask is not None:
            gm = llm_group_mask.to(m_group.device, dtype=torch.bool)
            if gm.dim() == 1:
                gm = gm[None, :].expand(m_group.shape[0], -1)
            m_group = m_group & gm
        return {
            "m_group":   m_group,
            "m_channel": group_mask_to_channels(m_group, is_group, group_channels, feat_dim),
            "edited":    m_group.any(dim=-1),
        }

    # semantic component (M1 / M_llm); absent when semantic_source is None
    if semantic_source is None:
        m_sem = None
    elif semantic_source == "attn":
        m_sem = percentile_threshold(attn_fg, valid_frames, lambda_attn)
    else:  # "groups"
        if llm_group_mask is None:
            raise ValueError(f"mask_mode={mask_mode!r} requires llm_group_mask (F, G) or (G,).")
        m_sem = llm_group_mask.to(valid.device, dtype=torch.bool)
        if m_sem.dim() == 1:
            m_sem = m_sem[None, :].expand(valid_frames.shape[0], -1)

    # noise component (M2)
    m2 = percentile_threshold(psi_fg, valid_frames, lambda_noise) if use_m2 else None

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
