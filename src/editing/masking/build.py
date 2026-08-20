"""
Turning the accumulated M1/M2 statistics into the binary (frame x group) edit mask.

Group axis: a grouped backbone gives G body-part groups and therefore true
spatiotemporal masks; a flat one gives G=1, where a frame is masked as a whole. The
(F, G) mask is expanded to a per-channel (F, D) mask for the Eq.-1 guidance and reduced
to a per-frame "edited" flag for Stage-3 hard inpainting.
"""

import torch

from .groups import group_mask_to_channels

# mask_mode -> (semantic source | None, use_m2). The two mask components are independent
# axes; encoding them as a lookup keeps adding or auditing a mode a one-line change
# instead of touching two parallel if/elif chains.
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
    components a mode uses.

    Public because callers needing only half of it were re-deriving the answer as literal
    tuples (`"attn" in (...)` to decide whether to capture cross-attention, say). Those
    copies fall out of date when a mode is added, and they fail OPEN: a new M1-using mode
    would simply not capture attention, or would drop M1's flags from a fingerprint.
    """
    if mask_mode not in _MASK_MODE_COMPONENTS:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    return _MASK_MODE_COMPONENTS[mask_mode]


def percentile_threshold(values: torch.Tensor, scope: torch.Tensor,
                         percentile: float) -> torch.Tensor:
    """Binarise (F, G) values, keeping entries above the given percentile of the
    distribution over the cells `scope` selects. percentile=70 keeps the top 30 %.

    `scope` is usually the (F,) valid-frame mask — "rank every cell in a real frame" —
    but an (F, G) cell mask works identically and is what `m1_select="rank"` passes to
    take the psi percentile within the selected group rows only. Both shapes index
    `values` down to the same flat set of in-scope cells.

    Public because the probes binarise candidate mask maps of their own, and their
    alignment numbers are only comparable to the editor's if the mask is cut the same
    way — so every cut goes through here rather than re-deriving the quantile.
    """
    valid_vals = values[scope].flatten()
    if valid_vals.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    thr = torch.quantile(valid_vals.float(), percentile / 100.0)
    return values >= thr


def rank_group_select(attn_fg: torch.Tensor, valid_frames: torch.Tensor,
                      ratio: float = 0.5, k_max: int = 3) -> torch.Tensor:
    """(G,) bool — the groups M1 actually names, chosen by RANK instead of a percentile.

    `percentile_threshold` is a cell BUDGET, not a selector: at lambda_attn=70 it must
    hand out 0.30*G cells' worth of rows whatever the map looks like, so a one-group
    instruction spills into the runner-up row and a perfect selector caps at 0.476
    alignment. Deriving the percentile from a known target count would fix that, but the
    count is not known — "raise the left arm" wants one group and "wave" wants two.

    So let the map say. Rank groups by total mass over valid frames and keep every group
    holding at least `ratio` of the top group's mass:

        keep g  <=>  w[g] >= ratio * max(w)          (then truncate to k_max)

    A ratio rather than an elbow: the largest-gap rule needs no parameter but is unstable
    when two gaps are close, and it cannot express "keep one" — the biggest gap in a flat
    map is meaningless. The ratio is monotone, scale-free, has one parameter with a plain
    reading ("at least half as much as the winner"), and degrades to top-1 as ratio -> 1.

    The property that makes it right here: on a grounded checkpoint "raise the left arm"
    gives left 0.94 / right 0.04, a ratio of 0.04 -> one group, while a bilateral
    instruction leaves the pair near-equal -> two groups. The same rule reads a confident
    lateralisation as "one" and a genuinely two-sided instruction as "two" without being
    told which it is; a lopsided-but-close pair (0.36/0.47 -> ratio 0.77) correctly keeps
    BOTH, the honest answer when the model is not confident about the side.

    `k_max` caps the damage when the map is flat — an ungrounded checkpoint has all
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


def _as_cell_mask(group_mask: torch.Tensor, n_frames: int, device) -> torch.Tensor:
    """(G,) or (F, G) bool -> (F, G) bool on `device`, broadcasting a per-group vector."""
    m = group_mask.to(device, dtype=torch.bool)
    return m[None, :].expand(n_frames, -1) if m.dim() == 1 else m


def _pack(m_group, is_group, group_channels, feat_dim) -> dict:
    """The three views of one mask every caller downstream needs."""
    return {
        "m_group": m_group,
        "m_channel": group_mask_to_channels(m_group, is_group, group_channels, feat_dim),
        "edited": m_group.any(dim=-1),
    }


def _temporal_mask(psi_fg, valid_frames, lambda_noise, llm_group_mask):
    """Frame-level "edit these frames", spatially permissive.

    Thresholds a per-FRAME activity score rather than per-cell values: psi summed over
    groups is "where is there change" aggregated across the body, and the top
    (100-lambda_noise) % of frames are edited across ALL groups. The rationale is that
    the group axis is not reliably instruction-grounded while the temporal one is, and a
    wrong (F, G) mask can inpaint the target region back to the source — so this trusts
    only the axis that works and never freezes the part that should change.
    """
    activity = psi_fg.sum(dim=1, keepdim=True)                        # (F, 1)
    active = percentile_threshold(activity, valid_frames, lambda_noise)
    m_group = (valid_frames[:, None] & active).expand(-1, psi_fg.shape[1]).clone()
    if llm_group_mask is not None:
        m_group &= _as_cell_mask(llm_group_mask, m_group.shape[0], m_group.device)
    return m_group


def _semantic_mask(semantic_source, attn_fg, valid_frames, llm_group_mask,
                   lambda_attn, m1_select, m1_rank_ratio, m1_rank_max):
    """M1 (cross-attention) or M_user (named groups), or None when the mode uses neither."""
    if semantic_source is None:
        return None
    if semantic_source == "groups":
        if llm_group_mask is None:
            raise ValueError("mask_mode='groups' requires llm_group_mask (F, G) or (G,).")
        return _as_cell_mask(llm_group_mask, valid_frames.shape[0], valid_frames.device)
    if m1_select == "rank":
        keep = rank_group_select(attn_fg, valid_frames, m1_rank_ratio, m1_rank_max)
        return keep[None, :].expand(attn_fg.shape[0], -1)
    if m1_select == "percentile":
        return percentile_threshold(attn_fg, valid_frames, lambda_attn)
    raise ValueError(f"m1_select must be 'percentile' or 'rank', got {m1_select!r}")


def _noise_mask(psi_fg, valid_frames, m_sem, lambda_noise, m1_select):
    """M2, cut globally or within M1's selected rows.

    Restricting the psi percentile to the selected rows is the other half of the rank
    fix: a global cut spends its budget ranking cells in groups the instruction never
    named, so which frames it keeps inside the TARGET group depends on how busy the rest
    of the body is. It also removes the arm/leg recall inversion at source — arm
    instructions produce a negative energy change at their own target, so under a global
    cut fewer arm cells clear it (recall 0.345 vs legs' 0.662).
    """
    if m1_select == "rank" and m_sem is not None:
        sel = valid_frames[:, None] & m_sem                   # (F, G) cells in scope
        return percentile_threshold(psi_fg, sel, lambda_noise) & sel
    return percentile_threshold(psi_fg, valid_frames, lambda_noise)


def build_mask(attn_fg, psi_fg, valid_frames, is_group,
               lambda_attn=70.0, lambda_noise=70.0,
               mask_mode="m2_only", llm_group_mask=None,
               group_channels=None, feat_dim=263,
               m1_select="percentile", m1_rank_ratio=0.5, m1_rank_max=3):
    """Build the per-edit mask according to `mask_mode`.

    attn_fg, psi_fg : (F, G) accumulated statistics from `collect_statistics`.
    valid_frames    : (F,) bool — True for real frames (excludes padding).
    lambda_*        : percentile thresholds; higher = sparser mask.
    mask_mode       : "none"     — no mask: every cell is editable, nothing is inpainted.
                      "m2_only"  — M2 alone, no semantic mask.
                      "m1_only"  — M1 alone, no M2 gating. Tests whether attention
                                   targets a part the source is NOT already moving,
                                   which M2 cannot add.
                      "attn"     — M1 intersect M2.
                      "groups"   — the named groups alone, edited in every valid frame:
                                   full temporal coverage rather than restricting to
                                   frames M2 judges as already changing.
                      "temporal" — frame-level; see `_temporal_mask`.
    llm_group_mask  : (F, G) or (G,) bool — required by "groups", optional in "temporal".
    m1_select       : "percentile" — one global quantile over all (F, G) cells at
                                     `lambda_attn`.
                      "rank"       — `rank_group_select` on the group axis, then a psi
                                     percentile WITHIN those rows. This splits a
                                     threshold that was doing two jobs: one global cut
                                     cannot both pick groups and pick frames.
                                     `lambda_attn` is unused, and `lambda_noise` becomes
                                     "what fraction of the selected rows' frames to
                                     keep" — the reading it should always have had.
    m1_rank_ratio,
    m1_rank_max     : the rank selector's two knobs.
    group_channels,
    feat_dim        : representation channel partition and total width (263 or 135).

    Returns {"m_group": (F, G) bool with padding forced False,
             "m_channel": (F, D) float for the guidance term,
             "edited": (F,) bool driving the hard inpainting}.
    """
    semantic_source, use_m2 = mask_mode_components(mask_mode)
    valid = valid_frames[:, None]

    if mask_mode == "temporal":
        m_group = _temporal_mask(psi_fg, valid_frames, lambda_noise, llm_group_mask)
        return _pack(m_group, is_group, group_channels, feat_dim)

    m_sem = _semantic_mask(semantic_source, attn_fg, valid_frames, llm_group_mask,
                           lambda_attn, m1_select, m1_rank_ratio, m1_rank_max)
    m2 = (_noise_mask(psi_fg, valid_frames, m_sem, lambda_noise, m1_select)
          if use_m2 else None)

    m_group = valid.expand(-1, attn_fg.shape[1]).clone()
    if m_sem is not None:
        m_group &= m_sem
    if m2 is not None:
        m_group &= m2
    return _pack(m_group, is_group, group_channels, feat_dim)
