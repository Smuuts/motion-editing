"""
Statistics shared by the mask/attention probe scripts.

These are the quantities every probe reports against a baseline: how similar two maps
are (`flat_corr`), what the source clip does on its own (`source_activity` — the
instruction-independent reference the implicit masks are measured against), and where
a map puts its mass across body-part groups (`group_profile`).

`wilson_ci`/`accuracy_block` are the shared *forced-choice* reporting: several probes
ask a question a constant bias cannot win (chance exactly 0.5) and must report the
answer with a CI and with which side of chance it falls on, so those two live here
rather than in whichever script needed them first.
"""

import math

import numpy as np
import torch

from editing import masking


def flat_corr(a, b) -> float:
    """Pearson r between two arrays' flattened values; 0 if either is constant."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pairwise_corr(maps) -> np.ndarray:
    """(n, n) matrix of flat_corr between every pair of maps."""
    n = len(maps)
    return np.array([[flat_corr(maps[i], maps[j]) for j in range(n)] for i in range(n)])


def source_activity(x0, group_channels, is_group=True) -> np.ndarray:
    """(F, G) per-(frame, group) source motion energy |Δx0|, first frame REPEATED.

    This is the reference every mask is compared against: it depends only on the source
    clip, so a mask that correlates with it is a source-dynamics detector.

    Must stay the same functional form as `editing.masking._frame_energy` — that parity is
    what makes "did the edit change this cell" and "was the source already moving here"
    like-for-like. Frame 0 repeats frame 1 rather than being zeroed
    (2026-08-16, changed in both functions together); see `_frame_energy` for why the zero
    was a structural hole rather than a neutral convention.
    """
    diff = (x0[0][1:] - x0[0][:-1]).abs()                        # (F-1, D)
    if is_group:
        act = torch.stack([diff[:, ch].mean(dim=-1) for ch in group_channels], dim=-1)
    else:
        act = diff.mean(dim=-1, keepdim=True)                    # (F-1, 1)
    return torch.cat([act[:1], act], dim=0).cpu().numpy()        # (F, G)


def group_profile(fg) -> np.ndarray:
    """(F, G) map → per-group marginal (mean over frames) normalised to sum 1."""
    v = np.asarray(fg).mean(axis=0)
    s = v.sum()
    return v / s if s > 1e-12 else v


def wilson_ci(k, n, z=1.96):
    """Binomial 95 % CI (Wilson) — sane at small n and near 0/1, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def accuracy_block(wins, label, chance=0.5):
    """A forced-choice result with its CI and *which side of chance* it falls on.

    `below_chance` is a real state, not a failed pass: it means the loser won
    systematically, i.e. the thing being probed has a fixed preference independent of
    the input — the exact bias a forced-choice design is built to expose.
    """
    n = len(wins)
    k = int(np.sum(wins))
    lo, hi = wilson_ci(k, n)
    lo, hi = max(0.0, lo), min(1.0, hi)
    return {"label": label, "n": n, "correct": k,
            "accuracy": k / n if n else 0.0, "ci95": [lo, hi], "chance": chance,
            "beats_chance": lo > chance, "below_chance": hi < chance}


def resolve_sweeps(mask_timesteps, T, m1_window=None, m2_window=None):
    """(shared_ts, m1_ts, m2_ts) timestep sweeps for mask collection.

    `None` on a per-mask window keeps that mask on the shared sweep, so the default run
    is a single even sweep over the whole trajectory. A window is resampled to the same
    number of steps *inside* the window (denser sampling, not fewer points) — M1 and M2
    carry their signal at different noise levels.
    """
    shared = masking.build_sweep(mask_timesteps, T) if mask_timesteps else None
    n = mask_timesteps or T - 1
    m1 = masking.build_sweep(n, T, *m1_window) if m1_window else None
    m2 = masking.build_sweep(n, T, *m2_window) if m2_window else None
    return shared, m1, m2


# ── noise-level bands ────────────────────────────────────────────────────────────

# Default band edges as fractions of T. NOT even in t, on purpose: the two masks carry
# their signal at opposite ends of the trajectory, so an even grid spends most of its
# rows where neither has anything to show.
#   · ψ_ε is a low-t read-out — ψ_ε = √SNR_t·ψ_x0 puts only ~5.6 % of its weight at
#     t ≥ 500 (PROGRESS item 7c) — so the low end needs the resolution.
#   · M1's instruction-sensitivity strengthens monotonically toward high t, and the two
#     boundaries every recorded M1 result is quoted against are t≈250 ("category r 0.899
#     at t < 250 vs 0.746 at t ≥ 750") and t=750 (`--m1_window 750 999`, the best
#     alignment in the project). Both are band edges here, so a row of this figure is
#     directly comparable to a number in FINDINGS.md.
#   · The last edge at 0.90 splits that high-noise end in two, because [750, 900] and
#     [900, 999] are not the same regime: √ᾱ is 0.19 at t=900 and 0.00 at t=999, so the
#     final row is the near-pure-noise limit where there is no clip left to detect and
#     the caption is the only signal in the input at all.
DEFAULT_BAND_FRACTIONS = (0.0, 0.05, 0.15, 0.30, 0.55, 0.75, 0.90, 1.0)


def _edges_to_bands(edges, T) -> list[tuple[int, int]]:
    """Sorted, de-duplicated, clamped edges → contiguous non-overlapping [lo, hi] bands.

    Bands abut rather than share an endpoint (next lo = previous hi + 1) so no timestep
    is averaged into two rows; an edge pair that collapses to nothing is dropped.
    """
    e = sorted({min(max(int(round(float(x))), 1), T - 1) for x in edges})
    if len(e) < 2:
        raise ValueError(f"need at least 2 distinct band edges in [1, {T - 1}], got {e}")
    bands = [(lo if i == 0 else lo + 1, hi)
             for i, (lo, hi) in enumerate(zip(e[:-1], e[1:]))]
    return [(lo, hi) for lo, hi in bands if lo <= hi]


def resolve_bands(spec: str, T: int, sqrt_alpha=None) -> list[tuple[int, int]]:
    """`--bands` spec → [(lo, hi), ...] timestep bands tiling [1, T-1].

    Forms:
      "default"    the `DEFAULT_BAND_FRACTIONS` edges (see there for why they are uneven)
      "linear:N"   N equal-width bands in t
      "log:N"      N bands geometric in t — even in *orders of magnitude* of noise, which
                   is where ψ's magnitude actually varies
      "alpha:N"    N bands even in √ᾱ_t, i.e. even in how much clean signal is left. The
                   most literal reading of "stages of the inversion", and schedule-aware:
                   it lands in the same place on any β-schedule. Needs `sqrt_alpha`.
      "1,50,250"   explicit edges
    """
    spec = (spec or "default").strip()
    if spec == "default":
        return _edges_to_bands([f * T for f in DEFAULT_BAND_FRACTIONS], T)

    kind, _, n_str = spec.partition(":")
    if kind in ("linear", "log", "alpha"):
        n = int(n_str) if n_str else 6
        if n < 1:
            raise ValueError(f"--bands {spec!r}: need at least 1 band")
        if kind == "linear":
            return _edges_to_bands(np.linspace(1, T - 1, n + 1), T)
        if kind == "log":
            return _edges_to_bands(np.geomspace(1, T - 1, n + 1), T)
        if sqrt_alpha is None:
            raise ValueError("--bands alpha:N needs the schedule's sqrt_alphas_cumprod")
        # √ᾱ decreases in t; np.interp needs an ascending x, hence the reversal.
        sa = np.asarray(sqrt_alpha, dtype=np.float64)
        ts = np.arange(len(sa))
        levels = np.linspace(sa[1], sa[T - 1], n + 1)
        return _edges_to_bands(np.interp(levels, sa[::-1], ts[::-1]), T)

    try:
        return _edges_to_bands([float(x) for x in spec.replace(" ", "").split(",")], T)
    except ValueError as e:
        raise ValueError(f"--bands {spec!r}: expected 'default', 'linear:N', 'log:N', "
                         f"'alpha:N' or a comma-separated edge list ({e})")


def band_labels(bands, sqrt_alpha=None) -> list[str]:
    """Row labels for a band figure: the t range, and the √ᾱ range it corresponds to.

    Both, because neither alone is readable — "t 750–999" says where on the CLI grid the
    row sits, "√ᾱ 0.31–0.00" says how much of the clip is still there.
    """
    out = []
    for lo, hi in bands:
        label = f"t {lo}–{hi}"
        if sqrt_alpha is not None:
            sa = np.asarray(sqrt_alpha)
            # The cosine schedule barely moves over the first ~15 % of the trajectory, so
            # 2 dp collapses the low-noise bands to "1.00–1.00". Widen only those.
            dp = 2 if round(float(sa[lo]), 2) != round(float(sa[hi]), 2) else 4
            label += f"\n√ᾱ {sa[lo]:.{dp}f}–{sa[hi]:.{dp}f}"
        out.append(label)
    return out
