"""
The inversion sweep: accumulate the raw M1 (attention) and M2 (psi) statistics.

One forward pass per (timestep, conditioning) is ever run — a timestep both masks want
shares its conditional pass, and the reference pass is skipped for timesteps only M1
needs — so giving the two masks separate sweeps never costs more than their union.
"""

import torch

from model.body_groups import GROUP_CHANNELS, N_GROUPS

from .groups import (frame_energy, group_aggregation_matrix, group_motion_energy,
                     normalise_step)
from .readouts import VALUE_READOUTS, WEIGHT_READOUTS, column_class_stats, step_readouts

# psi read-outs. "abs" is the LEDITS++ form; "energy" keeps the SIGN of the change in
# motion energy, which is what separates "the edit adds motion here" from "the edit
# suppresses the source's motion here".
PSI_READOUTS = ("abs", "energy")


def build_sweep(num_steps: int, T: int, lo: int = 1, hi: int | None = None) -> list[int]:
    """`num_steps` evenly-spaced timesteps inside [lo, hi] (default: the whole
    trajectory, i.e. exactly `linspace(1, T-1, num_steps)`).

    Narrowing the window RESAMPLES within it rather than discarding steps, so a 10 %-wide
    window is swept as densely as the full range — which matters because the windows
    carrying the most signal are also the ones a uniform grid samples most thinly.
    """
    hi = T - 1 if hi is None else min(hi, T - 1)
    lo = max(1, lo)
    if lo > hi:
        raise ValueError(f"empty timestep window [{lo}, {hi}]")
    return torch.linspace(lo, hi, num_steps).long().tolist()


def _resolve_sweep(timesteps, override, T) -> list[int]:
    """Timestep list for one mask: its own override if given, else the shared sweep."""
    ts = override if override is not None else timesteps
    return list(range(1, T)) if ts is None else list(ts)


def _resolve_group_count(is_group: bool, group_channels) -> int:
    """Number of token-axis cells. The model's own channel partition is the source of
    truth (7 body-part groups OR 22 per-joint tokens); grouped callers always pass
    `group_channels`, and N_GROUPS only backstops a grouped call that did not."""
    if not is_group:
        return 1
    return len(group_channels) if group_channels is not None else N_GROUPS


def _validate_readouts(model, readouts, psi_readouts, need_attn):
    """Fail before the sweep on a read-out this model or this module cannot produce."""
    unknown = [r for r in psi_readouts if r not in PSI_READOUTS]
    if unknown:
        raise ValueError(f"unknown psi_readout {unknown}, expected {PSI_READOUTS}")

    need_values = need_attn and any(r in VALUE_READOUTS for r in readouts)
    if need_values and not hasattr(model, "get_attn_values"):
        raise ValueError(
            f"read-outs {[r for r in readouts if r in VALUE_READOUTS]} need the value "
            f"vectors, which {type(model).__name__} does not expose (get_attn_values). "
            f"Weight-only read-outs {WEIGHT_READOUTS} still work.")
    return need_values


def _stack_layers(model, attn_layers, need_values):
    """Stored per-block attention (and values) -> (Lyr, B, h, N, L) / (Lyr, B, h, L, hd).

    `attn_layers` indexes the model's block order, which is exactly what get_attn_maps()
    returns and exactly the indexing `--attn_ground_layers` supervised, so a grounded
    checkpoint's read-out cannot drift from the blocks that were trained.
    """
    maps = model.get_attn_maps()
    vals = model.get_attn_values() if need_values else None
    if attn_layers is not None:
        maps = [maps[i] for i in attn_layers]
        if vals is not None:
            vals = [vals[i] for i in attn_layers]
    stacked = torch.stack(maps, dim=0).float()
    values = torch.stack(vals, dim=0).float() if need_values else None
    return stacked, values


def _psi_step(f_c, f_r, readout, agg_matrix, is_group):
    """One timestep's psi contrast for one read-out, as an (F, G) map."""
    if readout == "abs":
        psi = (f_c - f_r)[0].abs()                               # (F, D)
        return psi @ agg_matrix if is_group else psi.mean(dim=-1, keepdim=True)
    return (frame_energy(f_c[0], agg_matrix, is_group)           # "energy" — SIGNED
            - frame_energy(f_r[0], agg_matrix, is_group))


@torch.no_grad()
def collect_statistics(model, schedule, xs, context_edit, token_idxs,
                       is_group, timesteps=None, need_attn=True,
                       group_channels=None, context_ref=None, psi_group_norm=False,
                       valid_frames=None, attn_readout="raw", semantic_idxs=None,
                       attn_timesteps=None, psi_timesteps=None, per_step_norm=False,
                       psi_space=None, stats_out=None, attn_layers=None,
                       psi_readout="energy"):
    """Sweep the stored inversion sequence and accumulate the raw quantities for M1/M2.

    Returns `(attn_fg, psi_fg)`, both (F, G) — or a `{read-out: (F, G)}` dict for
    whichever of `attn_readout` / `psi_readout` was passed as a SEQUENCE. Asking for
    several read-outs at once is the only way to contrast them without a second
    inversion, and the inversion is stochastic (+-0.02 on these metrics run to run), so
    re-running it per read-out would put that spread inside the comparison.

    model, schedule : EMA denoiser in eval mode, and the NoiseSchedule (T, psi space).
    xs              : (T, 1, F, D) stored inversion samples x_t (see inversion.py).
    context_edit    : (1, L, dim) embedding of ONE edit instruction.
    token_idxs      : content-token columns in [0, L) from `text_encoder.token_info`.
                      May be None when need_attn=False.
    semantic_idxs   : stop-word-filtered subset (`semantic_token_subset`); required by
                      the non-"raw" read-outs, defaults to token_idxs.
    is_group        : True for a grouped backbone (G>1), False for per-frame (G=1).
    need_attn       : capture cross-attention for M1. False skips the attention pass
                      entirely and returns attn_fg = zeros.
    context_ref     : (1, L, dim) reference embedding for the psi contrast, or None for
                      the model's learned null embedding (the original LEDITS++/SEGA
                      form). Passing the SOURCE caption's embedding cancels
                      source-dynamics-driven psi, DiffEdit-style.
    attn_readout    : per-cell M1 read-out, or a sequence of them (see `readouts`).
    psi_readout     : "abs"    — mean |f(x_t,c) - f(x_t,ref)|, the original LEDITS++
                                 form. Unsigned, so "the edit ADDS motion here" and "the
                                 edit REMOVES motion here" are the same large value.
                      "energy" — SIGNED change in per-group motion energy (the default).
                      The distinction is load-bearing because HumanML3D captions describe
                      a WHOLE clip: conditioning on "raise the right arm" pulls the
                      prediction toward a clip where that arm moves and the rest is
                      still. Against a source that moves other parts, the largest
                      absolute differences then appear both at the target group (motion
                      added) and wherever the source moves (motion suppressed) — one
                      mechanism behind psi's ~+0.5 correlation with source dynamics. Only
                      the sign separates the two.
                      A signed map is not a magnitude: `psi_group_norm` divides it by a
                      positive energy (fine), but any reasoning about "the top 30 % of
                      MASS" must be read as "the top 30 % of VALUES" instead.
    psi_space       : "eps", "x0", or None for the checkpoint's own predict_type (how the
                      editor passes it). The two differ by the positive scalar sqrt(SNR_t),
                      so AT A FIXED t they rank cells identically and give the same
                      percentile mask — reading an eps checkpoint's psi in x0 space is not
                      a fix. What differs is the MIXTURE across the sweep: those weights
                      run 70.7 -> 0.04 over linspace(1, 999, 40), where t=1 alone carries
                      46 % of psi_eps's total. So psi_eps is effectively a low-noise
                      read-out, while psi_x0 weights every swept t by its own
                      clean-signal displacement.
    psi_group_norm  : divide psi per group by the source's per-group motion energy
                      (`group_motion_energy`). No-op for G=1.
    valid_frames    : (F,) bool; consulted by psi_group_norm and per_step_norm.
    timesteps       : shared sweep (default: every t in [1, T)).
    attn_timesteps,
    psi_timesteps   : per-mask sweeps overriding `timesteps`. M1 and M2 are read off the
                      same trajectory but their signal does not live at the same noise
                      levels — on an x0 checkpoint M1's instruction-sensitivity
                      strengthens monotonically toward high t while psi's is flat and its
                      magnitude concentrates at low t — so one shared sweep cannot serve
                      both.
    per_step_norm   : scale each timestep's map to unit mean before accumulating, so the
                      average weights every swept t equally instead of by its magnitude
                      (`normalise_step`).
    attn_layers     : block indices to average M1 over, or None for ALL blocks. Only
                      meaningful for M1 — psi is a model output, not a per-layer
                      quantity. It matters for a checkpoint trained with the grounding
                      loss: only the supervised layers were ever grounded, so averaging
                      all of them dilutes the signal. The entry points resolve this from
                      the checkpoint's own config via
                      `training.grounding.resolve_readout_layers`.
    stats_out       : optional dict, filled in place with the sweep-mean attention mass
                      and value norm per column class (content / EOS / pad) — the
                      diagnostic saying whether dropping the sink column discards a
                      distractor or the dominant contribution. Only populated when a
                      value-based read-out is requested.
    """
    T, F = schedule.T, xs.shape[2]
    G = _resolve_group_count(is_group, group_channels)
    device = context_edit.device

    single_attn = isinstance(attn_readout, str)
    readouts = (attn_readout,) if single_attn else tuple(attn_readout)
    single_psi = isinstance(psi_readout, str)
    psi_readouts = (psi_readout,) if single_psi else tuple(psi_readout)
    need_values = _validate_readouts(model, readouts, psi_readouts, need_attn)

    # Per-mask sweeps. They coincide unless a caller overrides one, in which case the
    # loop runs over the union and each accumulator takes only the steps it asked for.
    m1_ts = set(_resolve_sweep(timesteps, attn_timesteps, T)) if need_attn else set()
    m2_ts = set(_resolve_sweep(timesteps, psi_timesteps, T))
    psi_space = schedule.resolve_space(psi_space)

    attn_accum = {r: torch.zeros(F, G, device=device) for r in readouts}
    psi_accum = {r: torch.zeros(F, G, device=device) for r in psi_readouts}
    col_stats: dict[str, float] = {}
    n_attn = n_psi = 0

    tok = sem = None
    if need_attn:
        tok = torch.as_tensor(token_idxs, device=device, dtype=torch.long)
        sem = torch.as_tensor(semantic_idxs or token_idxs, device=device,
                              dtype=torch.long)
    agg_matrix = (group_aggregation_matrix(group_channels or GROUP_CHANNELS, device)
                  if is_group else None)

    for t in sorted(m1_ts | m2_ts):
        want_attn, want_psi = t in m1_ts, t in m2_ts
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        out_c = model(x_t, t_b, context_edit, store_attn=want_attn)

        if want_attn:
            stacked, values = _stack_layers(model, attn_layers, need_values)
            for r, v in step_readouts(stacked, values, tok, sem, readouts).items():
                step = v.reshape(F, G)
                attn_accum[r] += (normalise_step(step, valid_frames)
                                  if per_step_norm else step)
            if need_values and stats_out is not None:
                for k, val in column_class_stats(stacked, values, tok).items():
                    col_stats[k] = col_stats.get(k, 0.0) + val
            n_attn += 1

        if want_psi:
            f_c = schedule.to_space(out_c, x_t, t_b, psi_space)
            f_r = schedule.to_space(model(x_t, t_b, context_ref), x_t, t_b, psi_space)
            for r in psi_readouts:
                step = _psi_step(f_c, f_r, r, agg_matrix, is_group)
                psi_accum[r] += (normalise_step(step, valid_frames, signed=r != "abs")
                                 if per_step_norm else step)
            n_psi += 1

    psi_fg = {r: p / max(n_psi, 1) for r, p in psi_accum.items()}
    if psi_group_norm and is_group:
        energy = group_motion_energy(xs[0][0].to(device), valid_frames, agg_matrix)
        psi_fg = {r: p / energy[None, :] for r, p in psi_fg.items()}
    if stats_out is not None and col_stats:
        stats_out.update({k: v / max(n_attn, 1) for k, v in col_stats.items()})
    attn_fg = {r: a / max(n_attn, 1) for r, a in attn_accum.items()}

    return ((attn_fg[readouts[0]] if single_attn else attn_fg),
            (psi_fg[psi_readouts[0]] if single_psi else psi_fg))
