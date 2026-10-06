"""
Soft variants of a built mask: release groups from the hard inpainting without (or with less)
guidance.

Stage 3 hard-inpaints every cell outside the mask, so a mask that is spotty in time pins the
edited limb back to the source in its gaps (the limb snaps back and forth), and a mask that
stops at the selected group freezes the body parts the edited limb hangs from. Both helpers
take a mask dict from `MotionEditor.collect_masks` and return a new one that `MotionEditor.edit`
understands through its optional `m_release` key (1 = free, r in (0, 1) = pulled back towards
the source by 1 - r every step). Masks without the key behave exactly as before.

  blur_mask       the selected groups are released in every frame and M2's 0/1 frame gating
                  becomes a guidance weight blurred over `sigma` frames (thesis eq:softgate)
  add_neighbours  named groups (e.g. the spine) are released wherever the mask edits, with
                  `weight` of the edit's guidance (0 = released only, they follow the edit)
"""

import torch
from scipy.ndimage import gaussian_filter1d

from .groups import group_mask_to_channels


def add_neighbours(mask, names, weight, gnames, is_group, editor, release=1.0):
    """Release the named groups from inpainting in every frame `mask` edits and give them
    `weight` of the edit's guidance. Groups the mask already selects keep their full weight.
    `release` < 1 only partly frees them (see MotionEditor.edit)."""
    unknown = [n for n in names if n not in gnames]
    if unknown:
        raise SystemExit(f"--neighbour_groups: unknown group(s) {unknown}, expected {gnames}")
    m_group = mask["m_group"]
    nb = torch.zeros_like(m_group)
    nb[:, [gnames.index(n) for n in names]] = mask["edited"][:, None]
    nb &= ~m_group
    ch = group_mask_to_channels(nb, is_group, editor.group_channels, editor.feat_dim)
    ch = ch.to(mask["m_channel"].device)
    if mask.get("m_release") is not None:
        ch_release = torch.maximum(mask["m_release"], release * ch)
    else:
        ch_release = release * ch
    return {**mask, "m_channel": mask["m_channel"] + weight * ch, "m_release": ch_release}


def blur_mask(mask, sigma, is_group, editor, valid_frames):
    """Free the groups `mask` selects in every frame and turn its frame gating into a soft
    guidance weight. The guidance still acts where M2 picked the frames, but the frames in
    between are no longer pinned to the source, so the edit can fade in and out instead of
    jumping. The weight is the 0/1 gating blurred over `sigma` frames (0 = left as 0/1)."""
    m_group = mask["m_group"]
    sel = m_group.any(dim=0, keepdim=True) & valid_frames.to(m_group.device)[:, None]  # (F, G)
    weight = m_group.float()
    if sigma > 0:
        weight = torch.from_numpy(gaussian_filter1d(weight.cpu().numpy(), sigma, axis=0,
                                                    mode="nearest")).to(m_group.device)
        weight = weight * sel
    to_ch = lambda m: group_mask_to_channels(m, is_group, editor.group_channels,
                                             editor.feat_dim).to(mask["m_channel"].device)
    return {**mask, "m_channel": to_ch(weight), "m_release": to_ch(sel), "edited": sel.any(-1)}
