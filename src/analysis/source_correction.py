"""
Remove the source-motion component from a mask map, and score what is left.

The masks this project reads (M1 cross-attention, M2 noise ψ) correlate strongly with
the source clip's own motion `S = |Δx0|`, which is instruction-INDEPENDENT — so the
obvious repair is to take it out and keep the remainder. Two arithmetics are supported
because they fail differently:

  "sub"  M' = n(M) − λ·n(S)              additive; λ=0 is exactly the untouched map.
  "div"  M' = log(u(M)) − λ·log(u(S))    multiplicative (monotone-equivalent to
                                         u(M)/u(S)^λ); λ=0 is a monotone transform of
                                         the untouched map, so the binary mask and its
                                         alignment are unchanged while Pearson r is not.

Both nest the baseline at λ=0, which is what makes a λ-sweep readable as "does removing
the source help at all?" rather than as a comparison between two different pipelines.

TWO THINGS THAT MAKE OR BREAK A RESULT HERE, both learned the hard way (2026-08-01,
when subtracting the source-motion map from M1/M2 was measured not to help):

1. **A shuffled control is mandatory.** `S` is instruction-independent, so subtracting
   *anything* correlated with the masks' common component moves the instruction-
   invariance r whether or not the remainder means anything. `shuffled_source` supplies
   the two controls: all cells permuted, and group columns permuted (same per-group
   temporal profiles, wrong groups). Whatever the real map does that a shuffled map also
   does is an artifact.
2. **r must never be reported alone.** A percentile-thresholded mask is invariant to any
   monotone rescaling of its map, so an operation that provably cannot change the mask
   can still move r. Alignment of the *binary* mask is the load-bearing number.

Division has a third, specific to it: `S` is ~0 wherever the clip holds still (and
exactly 0 on frame 0 by construction), so dividing amplifies exactly the still regions —
the "bias inversion" that retired the divisive `--m2_group_norm` in 2026-07-15. `floor_q`
clamps `S` from below at its q-th percentile to bound that amplification; report the
floor with any div number, because the result depends on it.
"""

import numpy as np

MODES = ("sub", "div")
NORMS = ("z", "unit", "z_group", "unit_group")
CONTROLS = ("real", "shuffle_cells", "shuffle_groups")


def _valid_mask(fg, valid):
    """(F, G) bool selecting the cells statistics may be computed over."""
    if valid is None:
        return np.ones_like(fg, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    return np.broadcast_to(v[:, None], fg.shape)


def normalise(fg, kind="z", valid=None):
    """Put a map on a scale where λ means the same thing for M and for S.

    "z"    zero mean / unit std over valid cells (global)
    "unit" unit mean over valid cells (global) — positive, required by "div"
    *_group  the same, computed per body-part column.

    The per-group variants are kept because they are the natural way to ask "is this
    group unusually active *for itself*", but note they were measured to HURT: forcing
    every column onto a common scale deletes the "this group is globally more active"
    ranking, which is exactly what a percentile threshold sorts on.
    """
    if kind not in NORMS:
        raise ValueError(f"unknown normalisation {kind!r}, expected one of {NORMS}")
    fg = np.asarray(fg, dtype=np.float64)
    m = _valid_mask(fg, valid)
    axis_wise = kind.endswith("_group")
    base = kind.split("_")[0]

    def _stats(vals):
        return (vals.mean(), vals.std()) if base == "z" else (0.0, vals.mean())

    if not axis_wise:
        centre, scale = _stats(fg[m])
        return (fg - centre) / (scale if scale > 1e-12 else 1.0)

    out = np.empty_like(fg)
    for g in range(fg.shape[1]):
        col, cm = fg[:, g], m[:, g]
        centre, scale = _stats(col[cm]) if cm.any() else (0.0, 1.0)
        out[:, g] = (col - centre) / (scale if scale > 1e-12 else 1.0)
    return out


def shuffled_source(src_act, control, rng):
    """The control version of `S` — see the module docstring on why this is mandatory.

    "shuffle_cells"  every (frame, group) cell permuted: destroys all structure, keeps
                     the marginal distribution of values.
    "shuffle_groups" group columns permuted: keeps each group's temporal profile intact
                     and puts it on the WRONG group. The sharper control of the two —
                     it isolates "is the right group being removed" from "is a
                     source-shaped map being removed".
    """
    S = np.asarray(src_act, dtype=np.float64)
    if control == "real":
        return S
    if control == "shuffle_cells":
        return rng.permutation(S.ravel()).reshape(S.shape)
    if control == "shuffle_groups":
        return S[:, rng.permutation(S.shape[1])]
    raise ValueError(f"unknown control {control!r}, expected one of {CONTROLS}")


def correct(fg, src_act, lam, mode="sub", norm="z", valid=None, floor_q=5.0):
    """(F, G) map with `lam` × the source component removed. λ=0 → the baseline.

    For "div" the normalisation is forced positive ("z" would put negatives inside a
    log), and `floor_q` clamps the source at its q-th percentile over valid cells first:
    without it the still regions (S ≈ 0) divide up to arbitrarily large values and the
    mask inverts onto whatever the clip does NOT move. (Frame 0 used to be an *exact* 0
    here — `source_activity` zeroed it — and so was the worst case for this division; since
    2026-08-16 it repeats frame 1, which removes that particular guaranteed-zero cell but
    not the need for the floor.)
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if mode == "sub":
        return normalise(fg, norm, valid) - lam * normalise(src_act, norm, valid)

    pos_norm = norm.replace("z", "unit")
    M = normalise(fg, pos_norm, valid)
    S = normalise(src_act, pos_norm, valid)
    m = _valid_mask(S, valid)
    floor = np.quantile(S[m], floor_q / 100.0) if m.any() else 0.0
    floor = max(float(floor), 1e-6)
    return np.log(np.maximum(M, 1e-12)) - lam * np.log(np.maximum(S, floor))


def effective_norms(mode, norms):
    """The normalisations a mode will actually run, de-duplicated.

    "div" cannot use a z-normalisation (a log needs a positive map), so a requested "z"
    is COERCED to its unit-mean counterpart rather than dropped — dropping it silently
    produces an empty sweep when someone asks for `--mode div --norm z`, which looks
    like the probe running and finding nothing.
    """
    out = [n.replace("z", "unit") if mode == "div" else n for n in norms]
    return list(dict.fromkeys(out))


def sweep_grid(lambdas, modes=MODES, norms=("z",), controls=CONTROLS):
    """Every (mode, norm, control, λ) combination, baseline first.

    Yielded in an order where the λ=0 rows come first per (mode, norm), so a reader of
    the printed table sees the baseline above the thing being compared to it.
    """
    for mode in modes:
        for norm in effective_norms(mode, norms):
            for control in controls:
                for lam in lambdas:
                    if lam == 0 and control != "real":
                        continue              # λ=0 ignores S, so the controls duplicate it
                    yield mode, norm, control, lam
