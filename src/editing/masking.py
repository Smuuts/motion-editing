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
                         spatial profile — DAAM, Tang et al. 2023). All of those are
                         weight-only; "normw"/"normsum" instead read the CONTRIBUTION
                         α·‖v‖ that actually reaches the residual stream (Kobayashi
                         et al. 2020), which is a different ranking whenever value
                         norms differ across columns — and they differ maximally here,
                         the padding columns having value vectors of exactly zero.
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


def _frame_energy(x: torch.Tensor, agg_matrix, is_group: bool) -> torch.Tensor:
    """(F, D) motion → (F, G) per-(frame, group) motion energy, first frame REPEATED.

    Deliberately the same functional form as `utils.probe.source_activity` (per-channel
    |Δ between consecutive frames|, then the group MEAN), so an energy read off a
    model prediction and the source clip's own activity are the same quantity and can
    be subtracted or correlated without a scale correction. That parity is load-bearing
    — see docs/ARCHITECTURE.md — so the first-frame convention below is changed in BOTH
    functions or neither.

    **Frame 0 repeats frame 1's energy rather than being zeroed (2026-08-16).** Energy is
    defined on the transition into a frame, so frame 0 has no value of its own. Zeroing it
    made ψ[0, :] = E_c[0] − E_ref[0] = 0 − 0 = 0 *exactly*, for every clip and every
    timestep, so with a positive percentile cut the first frame could never enter M2 and
    was pinned to the source in every edit. Repeating is the cheapest convention that
    removes the structural hole without inventing a value: frame 0 inherits the motion it
    is about to undergo, which is also what makes `source_activity`'s "was this group
    already moving" answer non-degenerate at the clip boundary.
    """
    d = (x[1:] - x[:-1]).abs()                                   # (F-1, D)
    e = d @ agg_matrix if is_group else d.mean(dim=-1, keepdim=True)
    return torch.cat([e[:1], e], dim=0)                          # (F, G)


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


# Readouts that need only the attention weights, and those that additionally need the
# value vectors. Kept as data so a caller can ask for "every readout" in one sweep and
# the probe scripts can validate a --m1_readout choice without a second list.
WEIGHT_READOUTS = ("raw", "renorm", "spatial", "renorm_spatial")
VALUE_READOUTS = ("normw", "normsum")
ALL_READOUTS = WEIGHT_READOUTS + VALUE_READOUTS
# ψ read-outs. "abs" is the LEDITS++ form; "energy" keeps the SIGN of the change in
# motion energy, which is what separates "the edit adds motion here" from "the edit
# suppresses the source's motion here" (see collect_statistics).
PSI_READOUTS = ("abs", "energy")


def _value_weighted_map(stacked: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """(Lyr, B, h, N, L) attention → (N, L) map of CONTRIBUTION α·‖v‖, not weight α.

    The per-head value norm is the right scale: attention is computed per head, and a
    head's output contribution from column w is α[h,·,w]·v[h,w], so weighting by a
    head-agnostic norm would mix scales across heads that need not be comparable.
    """
    vnorm = values.norm(dim=-1)                                  # (Lyr, B, h, L)
    return (stacked * vnorm[..., None, :]).mean(dim=(0, 1, 2))   # (N, L)


def _summed_contribution(stacked: torch.Tensor, values: torch.Tensor,
                         sem: torch.Tensor) -> torch.Tensor:
    """‖Σ_w α_w v_w‖ over the semantic columns → (N,) — what each cell actually
    RECEIVED from the body-part words.

    Differs from the per-token form by accounting for CANCELLATION: two words whose
    value vectors point in opposite directions contribute a large α·‖v‖ each while
    summing to nearly nothing. "left" vs "right" is exactly the pair one would expect
    to be near-antiparallel, so this is the form in which a laterality signal could
    survive when the per-token form shows none.
    """
    a = stacked[..., sem]                                        # (Lyr, B, h, N, S)
    v = values[:, :, :, sem, :]                                  # (Lyr, B, h, S, hd)
    contrib = a @ v                                              # (Lyr, B, h, N, hd)
    contrib = contrib.permute(0, 1, 3, 2, 4).flatten(-2)         # (Lyr, B, N, h*hd)
    return contrib.norm(dim=-1).mean(dim=(0, 1))                 # (N,)


def _attn_step_readouts(stacked: torch.Tensor, values, tok: torch.Tensor,
                        sem: torch.Tensor, readouts) -> dict[str, torch.Tensor]:
    """One timestep's (Lyr, B, h, N, L) attention → {readout: (N,) per-cell value}.

    All requested readouts are computed from the SAME stored tensors, so a probe
    comparing them is comparing readouts and nothing else — no second inversion, no
    run-to-run spread between the numbers being contrasted.
    """
    out, avg = {}, None
    for r in readouts:
        if r in WEIGHT_READOUTS:
            if avg is None:
                avg = stacked.mean(dim=(0, 1, 2))                # (N, L)
            out[r] = _attn_readout_value(avg, tok, sem, r)
        elif r == "normw":
            out[r] = _attn_readout_value(_value_weighted_map(stacked, values),
                                         tok, sem, "raw")
        elif r == "normsum":
            out[r] = _summed_contribution(stacked, values, sem)
        else:
            raise ValueError(f"unknown attn_readout {r!r}")
    return out


def _column_class_stats(stacked: torch.Tensor, values: torch.Tensor,
                        tok: torch.Tensor) -> dict[str, float]:
    """Attention mass and mean value norm, split by column class.

    The standing question Option 15 exists to settle: `renorm` drops the EOS/sink column
    from its denominator as a distractor, which is only right if that column contributes
    little. Pads are the calibration point — their value vectors are exactly zero, so
    any mass on them is contribution-free by construction.

    EOS is the column immediately after the last content token (T5 emits
    [tokens..., EOS, pad...]; token_info returns the content columns only).
    """
    avg = stacked.mean(dim=(0, 1, 2))                            # (N, L)
    vnorm = values.norm(dim=-1).mean(dim=(0, 1, 2))              # (L,)
    L = avg.shape[-1]
    eos = int(tok.max().item()) + 1
    pad = torch.arange(eos + 1, L, device=avg.device)
    out = {
        "mass_content": float(avg[:, tok].sum(dim=-1).mean()),
        "vnorm_content": float(vnorm[tok].mean()),
        "row_total": float(avg.sum(dim=-1).mean()),
    }
    if eos < L:
        out["mass_eos"] = float(avg[:, eos].mean())
        out["vnorm_eos"] = float(vnorm[eos])
    if pad.numel():
        out["mass_pad"] = float(avg[:, pad].sum(dim=-1).mean())
        out["vnorm_pad"] = float(vnorm[pad].mean())
    return out


def _normalise_step(values: torch.Tensor, valid_frames, signed: bool = False) -> torch.Tensor:
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

    `signed=True` scales by the mean ABSOLUTE value instead. A signed map (the "energy"
    ψ read-out) has a mean near zero and of either sign, so dividing by it would blow up
    the map and — where the mean is negative — flip it, which is not a normalisation but
    a different map.
    """
    cells = values[valid_frames] if valid_frames is not None else values
    scale = (cells.abs() if signed else cells).mean().clamp_min(1e-12)
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


def percentile_threshold(values: torch.Tensor, scope: torch.Tensor,
                         percentile: float) -> torch.Tensor:
    """
    Binarise (F, G) values, keeping entries above the given percentile of the
    distribution over the cells `scope` selects. percentile=70 keeps the top 30%.

    `scope` is usually the (F,) valid-frame mask — "rank every cell in a real frame" —
    but a (F, G) cell mask works identically and is what `m1_select="rank"` passes to
    take the ψ percentile *within the selected group rows only*. Both shapes index
    `values` down to the same flat set of in-scope cells.

    Public because the probes binarise candidate mask maps of their own (Option 6's
    generation-space divergence, say) and their alignment numbers are only comparable to
    the editor's if the mask is cut the same way — so every cut in this file goes through
    here rather than re-deriving the quantile.
    """
    valid_vals = values[scope].flatten()
    if valid_vals.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    thr = torch.quantile(valid_vals.float(), percentile / 100.0)
    return values >= thr


def rank_group_select(attn_fg: torch.Tensor, valid_frames: torch.Tensor,
                      ratio: float = 0.5, k_max: int = 3) -> torch.Tensor:
    """(G,) bool — the groups M1 actually names, chosen by RANK instead of a percentile.

    `percentile_threshold` is a cell budget, not a selector: at `lambda_attn=70` it must
    hand out `0.30 · G = 2.10` group-rows' worth of cells whatever the map looks like, so
    a one-group instruction spills into the runner-up row and a perfect selector caps at
    `1/2.10 = 0.476` alignment (docs/FINDINGS.md "Two mask defects"). Deriving the
    percentile from a known target count fixes that, but the count is not known — "raise
    the left arm" wants one group and "wave" genuinely wants two.

    So let the map say. Rank the groups by their total mass over valid frames and keep
    every group holding at least `ratio` of the top group's mass:

        keep g  ⟺  w[g] ≥ ratio · max(w)          (then truncate to k_max)

    **Why a ratio and not an elbow.** The largest-gap/elbow rule needs no parameter but is
    unstable when two gaps are close, and it cannot express "keep one" — the biggest gap
    in a flat map is meaningless. The ratio is monotone, scale-free, has one parameter
    with a plain reading ("at least half as much as the winner"), and degrades to the
    top-1 group as ratio → 1.

    **The property that makes this the right rule here**, on `exp_smplh_verbs` span
    read-out, 9 clips: "raise the left arm" gives left 0.94 / right 0.04, a ratio of
    0.04 → one group. A bilateral instruction leaves the pair near-equal → two groups. So
    the same rule reads a confident lateralisation as "one" and a genuinely two-sided
    instruction as "two", without being told which it is. A lopsided-but-close pair (the
    tier-2 split defect, `semantic` read-out: 0.36/0.47 → ratio 0.77) correctly keeps
    BOTH, which is the honest answer when the model is not confident about the side.

    k_max caps the damage when the map is flat — an ungrounded checkpoint has all seven
    groups within a few percent of each other and would otherwise select the whole body.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    w = attn_fg[valid_frames].sum(dim=0)                          # (G,) total mass
    if w.numel() == 0 or float(w.max()) <= 0.0:
        return torch.ones_like(w, dtype=torch.bool)               # degenerate: keep all
    keep = w >= ratio * w.max()
    if int(keep.sum()) > k_max:                                   # truncate by rank
        cut = torch.topk(w, k_max).values[-1]
        keep = keep & (w >= cut)
    return keep


@torch.no_grad()
def collect_statistics(model, schedule, xs, context_edit, token_idxs,
                       is_group, timesteps=None, need_attn=True,
                       group_channels=None, context_ref=None, psi_group_norm=False,
                       valid_frames=None, attn_readout="raw", semantic_idxs=None,
                       attn_timesteps=None, psi_timesteps=None, per_step_norm=False,
                       psi_space=None, stats_out=None, attn_layers=None,
                       psi_readout="energy"):
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
                    "normw"   — CONTRIBUTION α·‖v‖ instead of weight α, then read as
                                "raw". Attention weight ranks keys by how hard a query
                                looks at them; what reaches the residual stream is
                                α·v, and value norms vary by an order of magnitude
                                (Kobayashi et al., EMNLP 2020). This project holds the
                                extreme case: 92.9% of mass once sat on padding columns
                                whose value vectors are exactly zero.
                    "normsum" — ‖Σ_w α_w v_w‖ over the semantic columns: the strictly
                                correct "what this cell received", which additionally
                                accounts for cancellation between tokens.
                    May also be a SEQUENCE of readout names, in which case every one is
                    computed from the same stored attention and the returned attn_fg is
                    a {name: (F, G)} dict instead of a single map. That is the only way
                    to contrast readouts without a second inversion — the inversion is
                    stochastic (±0.02 run-to-run on these metrics), so re-running it per
                    readout would put that spread inside the comparison.
    semantic_idxs : stop-word-filtered subset of token_idxs (semantic_token_subset).
                    Required for the non-"raw" readouts; defaults to token_idxs.
    stats_out     : optional dict, filled in place with the sweep-mean attention mass
                    and value norm per column class (content / EOS / pad) — the
                    diagnostic that says whether dropping the sink column from a readout
                    discards a distractor or the dominant contribution. Only populated
                    when a value-based readout is requested.
    attn_timesteps: sweep for M1 only, overriding `timesteps`. M1 and M2 are read off
                    the same trajectory but their signal does not live at the same
                    noise levels — on an x0 checkpoint M1's instruction-sensitivity
                    strengthens monotonically toward high t, while ψ's is flat and its
                    magnitude is concentrated at low t (docs/FINDINGS.md). A single
                    shared sweep cannot serve both; these two arguments let each mask
                    be read where its own signal is. None → use `timesteps`.
    psi_timesteps : sweep for M2 only, overriding `timesteps`. None → use `timesteps`.
    psi_readout   : what the ψ contrast measures per (frame, group):
                    "abs"    — mean |f_θ(x_t,c) − f_θ(x_t,ref)| (the original; the
                               default until 2026-08-15). Unsigned, so "the edit ADDS
                               motion here" and "the edit REMOVES motion here" are the
                               same large value.
                    "energy" — SIGNED change in motion energy,
                               E[f_θ(x_t,c)] − E[f_θ(x_t,ref)], where E is per-group
                               |Δ between frames| (`_frame_energy`, the same quantity as
                               `utils.probe.source_activity`). Positive = the instruction
                               makes this group move MORE than the reference does,
                               negative = less. **The default since 2026-08-15**, on a
                               size-matched M2-alone comparison (identical 327-cell
                               budget, 9 clips × 3 checkpoints) where it wins every axis
                               on every checkpoint — see MotionEditor.__init__.
                    Why the distinction is load-bearing: captions in HumanML3D describe a
                    whole clip, so conditioning on "raise the right arm" pulls the
                    prediction toward a clip where the right arm moves and the rest is
                    still. Against a source clip that moves other parts, the largest
                    ABSOLUTE differences then appear both at the target group (motion
                    added) and at whatever the source moves (motion suppressed) — which
                    is one mechanism behind ψ's ~+0.5 correlation with source dynamics.
                    "abs" cannot separate those two; "energy" separates them by sign.
                    May also be a SEQUENCE, in which case psi_fg is a {name: (F, G)} dict
                    and every read-out comes from the same two forward passes.
                    NOTE a signed map is not a magnitude: `psi_group_norm` divides it by
                    a positive per-group energy (fine), but code that assumes ψ ≥ 0 —
                    including any percentile threshold reasoning about "the top 30% of
                    mass" — must be read as "the top 30% of values" instead.
    attn_layers   : block indices to average M1 over, or None for ALL blocks (the
                    historical behaviour). Only meaningful for M1 — ψ is a model output,
                    not a per-layer quantity.
                    This matters for a checkpoint trained with the TokenCompose grounding
                    loss: only `attn_ground_layers` (3 of 8 by default) were ever
                    supervised, so averaging all 8 mixes 3 grounded maps into 5
                    ungrounded ones and dilutes the signal ~8/3×. The entry points
                    resolve this from the checkpoint's own config via
                    `training.grounding.resolve_readout_layers`, so a grounded checkpoint
                    reads its own supervised blocks by default and every older checkpoint
                    (no such config key) still averages all of them.
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
      attn_fg — mean semantic cross-attention per (frame, group) (zeros if need_attn=False),
                or {readout: (F, G)} when attn_readout is a sequence
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

    # One readout or many, from one sweep. `single` restores the historical scalar-map
    # return so no existing caller sees a type change.
    single = isinstance(attn_readout, str)
    readouts = (attn_readout,) if single else tuple(attn_readout)
    need_values = need_attn and any(r in VALUE_READOUTS for r in readouts)
    if need_values and not hasattr(model, "get_attn_values"):
        raise ValueError(
            f"readouts {[r for r in readouts if r in VALUE_READOUTS]} need the value "
            f"vectors, which {type(model).__name__} does not expose "
            f"(get_attn_values). Weight-only readouts {WEIGHT_READOUTS} still work.")

    device = context_edit.device
    # ψ reference: source-caption embedding if given, else None → the model falls
    # back to its learned null_text_emb (original behaviour).
    psi_single = isinstance(psi_readout, str)
    psi_readouts = (psi_readout,) if psi_single else tuple(psi_readout)
    unknown = [r for r in psi_readouts if r not in PSI_READOUTS]
    if unknown:
        raise ValueError(f"unknown psi_readout {unknown}, expected {PSI_READOUTS}")

    attn_accum = {r: torch.zeros(F, G, device=device) for r in readouts}
    psi_accum  = {r: torch.zeros(F, G, device=device) for r in psi_readouts}
    col_stats  = {}
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
            layer_vals = model.get_attn_values() if need_values else None
            if attn_layers is not None:
                # Read only the requested blocks. Indices are into the model's block
                # order, which is exactly what get_attn_maps() returns — and the same
                # indexing `--attn_ground_layers` supervised, so the two cannot drift.
                layer_maps = [layer_maps[i] for i in attn_layers]
                if layer_vals is not None:
                    layer_vals = [layer_vals[i] for i in attn_layers]
            stacked = torch.stack(layer_maps, dim=0).float()  # (Lyr, 1, h, N, L)
            values = (torch.stack(layer_vals, dim=0).float()
                      if need_values else None)              # (Lyr, 1, h, L, hd)
            for r, v in _attn_step_readouts(stacked, values, tok, sem, readouts).items():
                step = v.reshape(F, G)
                attn_accum[r] += (_normalise_step(step, valid_frames)
                                  if per_step_norm else step)
            if need_values and stats_out is not None:
                for k, val in _column_class_stats(stacked, values, tok).items():
                    col_stats[k] = col_stats.get(k, 0.0) + val
            n_attn += 1

        if want_psi:
            # ψ = f_θ(x_t, c_edit) − f_θ(x_t, ref) → M2 contribution, read in psi_space:
            # a contrast of noise estimates ("eps") or of clean-motion predictions ("x0").
            f_c = schedule.to_space(out_c, x_t, t_b, psi_space)
            f_r = schedule.to_space(model(x_t, t_b, context_ref), x_t, t_b, psi_space)
            for r in psi_readouts:
                if r == "abs":
                    psi = (f_c - f_r)[0].abs()                # (F, D)
                    step = (psi @ agg_matrix if is_group
                            else psi.mean(dim=-1, keepdim=True))
                else:                                          # "energy" — SIGNED
                    step = (_frame_energy(f_c[0], agg_matrix, is_group)
                            - _frame_energy(f_r[0], agg_matrix, is_group))
                psi_accum[r] += (_normalise_step(step, valid_frames, signed=r != "abs")
                                 if per_step_norm else step)
            n_psi += 1

    psi_fg = {r: p / max(n_psi, 1) for r, p in psi_accum.items()}
    if psi_group_norm and is_group:
        energy = _group_motion_energy(xs[0][0].to(device), valid_frames, agg_matrix)
        psi_fg = {r: p / energy[None, :] for r, p in psi_fg.items()}
    if stats_out is not None and col_stats:
        stats_out.update({k: v / max(n_attn, 1) for k, v in col_stats.items()})
    attn_fg = {r: a / max(n_attn, 1) for r, a in attn_accum.items()}
    return ((attn_fg[readouts[0]] if single else attn_fg),
            (psi_fg[psi_readouts[0]] if psi_single else psi_fg))


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


def mask_mode_components(mask_mode: str) -> tuple[str | None, bool]:
    """(semantic_source, use_m2) for a mask mode — the single source of truth for which
    mask components a mode actually uses.

    Public because the answer was being re-derived as literal tuples by every caller that
    needed only half of it: `"attn" in (...)` to decide whether to capture cross-attention,
    or to decide which flags are live for a given mode. Those copies silently fall out of
    date when a mode is added — which is what the table's "one-line change" comment
    promises they will not — and a stale copy fails open: a new M1-using mode would simply
    not capture attention, or would drop M1's flags from a provenance fingerprint.
    """
    if mask_mode not in _MASK_MODE_COMPONENTS:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    return _MASK_MODE_COMPONENTS[mask_mode]


def build_mask(attn_fg, psi_fg, valid_frames, is_group,
               lambda_attn=70.0, lambda_noise=70.0,
               mask_mode="m2_only", llm_group_mask=None,
               group_channels=None, feat_dim=263,
               m1_select="percentile", m1_rank_ratio=0.5, m1_rank_max=3):
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

    m1_select       : how M1's group component is cut.
                      "percentile" — the historical single global quantile over all
                                     (F, G) cells at `lambda_attn`. Bit-identical to
                                     every result recorded before 2026-08-15.
                      "rank"       — `rank_group_select`: keep the top-ranked groups by
                                     total mass, then threshold M2 **within those rows
                                     only**. This splits a threshold that was doing two
                                     jobs — one global cut cannot both pick groups and
                                     pick frames — into a rank on the group axis and a
                                     percentile on the time axis. `lambda_attn` is unused
                                     in this mode; `lambda_noise` becomes "what fraction
                                     of the SELECTED rows' frames to keep", which is the
                                     reading it should always have had.
    m1_rank_ratio   : keep groups holding ≥ this fraction of the top group's mass.
    m1_rank_max     : hard cap on how many groups "rank" may select.

    group_channels  : representation channel partition (263-d default, or 135-d smplh)
    feat_dim        : total feature width D matching group_channels (263 or 135)

    Returns dict with:
      m_group   : (F, G) bool   — final mask, padding frames forced False
      m_channel : (F, D) float  — group mask scattered to feature channels (for Eq. 1)
      edited    : (F,) bool      — frame has any active group (drives hard inpainting)
    """
    semantic_source, use_m2 = mask_mode_components(mask_mode)

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
        if m1_select == "rank":
            keep = rank_group_select(attn_fg, valid_frames, m1_rank_ratio, m1_rank_max)
            m_sem = keep[None, :].expand(attn_fg.shape[0], -1)
        elif m1_select == "percentile":
            m_sem = percentile_threshold(attn_fg, valid_frames, lambda_attn)
        else:
            raise ValueError(f"m1_select must be 'percentile' or 'rank', got {m1_select!r}")
    else:  # "groups"
        if llm_group_mask is None:
            raise ValueError(f"mask_mode={mask_mode!r} requires llm_group_mask (F, G) or (G,).")
        m_sem = llm_group_mask.to(valid.device, dtype=torch.bool)
        if m_sem.dim() == 1:
            m_sem = m_sem[None, :].expand(valid_frames.shape[0], -1)

    # noise component (M2). Under "rank" the ψ percentile is taken over the SELECTED
    # rows only, which is the other half of the same fix: a global cut spends its budget
    # ranking cells in groups the instruction never named, so the frames it keeps inside
    # the target group depend on how busy the rest of the body is. It also removes the
    # arm/leg recall inversion at source — arm instructions produce a negative ΔE at
    # their own target, so under a global cut fewer arm cells clear it (recall 0.345 vs
    # legs' 0.662); ranked within the selected rows the comparison is limb-relative.
    if not use_m2:
        m2 = None
    elif m1_select == "rank" and m_sem is not None:
        sel = valid & m_sem                                   # (F, G) cells in scope
        m2 = percentile_threshold(psi_fg, sel, lambda_noise) & sel
    else:
        m2 = percentile_threshold(psi_fg, valid_frames, lambda_noise)

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
