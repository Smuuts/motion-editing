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
