"""
Channel <-> body-part-group conversions and the motion-energy quantities built on them.

The representation's channel partition (263-d HumanML3D or 135-d SMPL-H) is the only
thing here that is representation-specific; every caller passes its own
`group_channels`, so nothing below assumes a feature width.
"""

import torch

from model.body_groups import GROUP_CHANNELS


def group_aggregation_matrix(group_channels, device) -> torch.Tensor:
    """(D, G) matrix with `per_channel @ matrix == per-group mean`.

    Precomputed once per sweep so the hot loop replaces a Python-level list
    comprehension per timestep with a single matmul.
    """
    D = sum(len(ch) for ch in group_channels)
    mat = torch.zeros(D, len(group_channels), device=device)
    for g, ch in enumerate(group_channels):
        mat[ch, g] = 1.0 / len(ch)
    return mat


def frame_energy(x: torch.Tensor, agg_matrix, is_group: bool) -> torch.Tensor:
    """(F, D) motion -> (F, G) per-(frame, group) motion energy, first frame REPEATED.

    Deliberately the same functional form as `utils.probe.source_activity` and
    `training.grounding`'s source energy: per-channel |delta between consecutive frames|,
    then the group MEAN. That parity is load-bearing — an energy read off a model
    prediction and the source clip's own activity are then the same quantity and can be
    subtracted or correlated without a scale correction — so the first-frame convention
    below changes in all three or in none.

    Frame 0 repeats frame 1's energy rather than being zeroed. Energy is defined on the
    transition *into* a frame, so frame 0 has no value of its own; zeroing it made
    psi[0, :] = 0 exactly for every clip and timestep, which pinned the first frame to
    the source in every edit. Repeating is the cheapest convention that removes the
    structural hole without inventing a value.
    """
    d = (x[1:] - x[:-1]).abs()                                   # (F-1, D)
    e = d @ agg_matrix if is_group else d.mean(dim=-1, keepdim=True)
    return torch.cat([e[:1], e], dim=0)                          # (F, G)


def group_motion_energy(x0: torch.Tensor, valid_frames, agg_matrix: torch.Tensor,
                        floor_frac: float = 0.25) -> torch.Tensor:
    """(G,) mean |delta x0| per group over valid frames — how much each group already
    moves in the source. Used to discount psi for groups whose large noise difference is
    explained by source dynamics rather than by the instruction.

    Floored at `floor_frac * mean(energy)` so a truly static group cannot blow up to an
    unbounded psi boost: dividing by ~0 would hand the mask to *any* static group
    regardless of the instruction.
    """
    if valid_frames is not None:
        x0 = x0[valid_frames]
    diff = (x0[1:] - x0[:-1]).abs().mean(dim=0)      # (D,) per-channel motion energy
    energy = diff @ agg_matrix                        # (G,) per-group mean
    return energy.clamp(min=floor_frac * energy.mean())


def normalise_step(values: torch.Tensor, valid_frames,
                   signed: bool = False) -> torch.Tensor:
    """Scale one timestep's (F, G) map to unit mean over valid frames.

    A sweep accumulates raw magnitudes, and both M1's and psi's scale vary strongly with
    t (measured on an x0 checkpoint: M1 draws ~48 % of its total from t >= 750, psi ~68 %
    from t < 250). An evenly-spaced grid therefore still produces a magnitude-weighted
    average in which a handful of large-scale steps decide the mask. Dividing each step
    by its own mean makes every swept t contribute equally, which is what the even grid
    was meant to express; relative structure within a step — the only thing a percentile
    threshold reads — is untouched.

    `signed=True` scales by the mean ABSOLUTE value instead. A signed map (the "energy"
    psi read-out) has a mean near zero and of either sign, so dividing by it would blow
    up the map and, where the mean is negative, flip it — a different map, not a
    normalisation.
    """
    cells = values[valid_frames] if valid_frames is not None else values
    scale = (cells.abs() if signed else cells).mean().clamp_min(1e-12)
    return values / scale


def group_mask_to_channels(m_group: torch.Tensor, is_group: bool,
                           group_channels=None, feat_dim: int = 263) -> torch.Tensor:
    """(F, G) bool group mask -> (F, D) float channel mask via the channel partition.

    Defaults to the 263-d HumanML3D partition; the editor passes the model's own
    `group_channels` + `feat_dim` (e.g. 135-d SMPL-H).
    """
    F = m_group.shape[0]
    if not is_group:
        return m_group.float().expand(F, feat_dim)           # G=1 -> broadcast
    if group_channels is None:
        group_channels = GROUP_CHANNELS
    out = torch.zeros(F, feat_dim, device=m_group.device)
    for g, ch in enumerate(group_channels):
        out[:, ch] = m_group[:, g : g + 1].float()
    return out
