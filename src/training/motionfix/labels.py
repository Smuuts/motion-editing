"""
Grounding labels for the fine-tune: which text columns must attend to which body-part
groups, per triplet.

The TokenCompose loss needs, per item, a set of text columns W and a set of groups S.
There are two sources of S, and they fail differently (measured on 400 train triplets):

  parser  the caption parser reads the groups out of the instruction. Correct by
          construction, but silent on the ~23 % of instructions naming no body part.
  diff    the top groups of |d(velocity_target) - d(velocity_source)|. Covers
          everything and is ~77 % top-1 accurate — but ~55 points of that is reachable
          from a SHUFFLED pair, i.e. the label carries a corpus-level "the target is
          usually an arm" prior. Used alone it partly teaches that prior instead of
          word->group routing, which is the shortcut this loss exists to avoid.

The default `parser_first` uses the parser where it fires and the diff elsewhere: full
coverage, clean labels wherever clean labels exist.
"""

import os

import numpy as np

from data.body_part_labels import to_items
from model.body_groups import GROUP_NAMES
from utils.logger import get_logger

log = get_logger(__name__)


def velocity_diff_map(source, target, group_channels):
    """(F-1, G) per-(frame, group) velocity difference between source and target.

    Velocity, not pose: a constant per-performer rest-pose offset cancels in the
    derivative, and 94.6 % of MotionFix pairs are two different captures. Measured best
    of six read-outs on the GROUP axis: 77-78 % top-1 against 75.4 % for raw pose.
    """
    n = min(len(source), len(target))
    dv = np.abs(np.diff(source[:n], axis=0) - np.diff(target[:n], axis=0))
    if not len(dv):
        return None
    return np.stack([dv[:, ch].mean(1) for ch in group_channels], 1)


def diff_groups(vmap, ratio, kmax) -> list[int]:
    """Group set: everything within `ratio` of the top group's mass, capped at `kmax`."""
    mass = vmap.mean(0)
    if mass.max() <= 0:
        return []
    keep = np.where(mass >= ratio * mass.max())[0]
    return [int(g) for g in keep[np.argsort(-mass[keep])][:kmax]]


def diff_region(vmap, groups, keep_frac) -> np.ndarray:
    """(F, G) BINARY region for the spatiotemporal L_token target.

    Inside the selected group rows, keep the busiest `keep_frac` of frames; everything
    else is 0. Binary on purpose — a soft target caps m below 1 and leaves (1 - m)^2 with
    an irreducible floor, i.e. permanent gradient toward an unreachable optimum.

    THE TEMPORAL AXIS CARRIES ALMOST NO PAIR-SPECIFIC SIGNAL. Share of mass in the
    busiest 20 % of frames: real pair 0.358 against a SHUFFLED pair 0.346 — a gap of
    +0.012, against +0.19..+0.25 for the group axis. Two corrected read-outs were tried
    and neither helped (normalised +0.013, excess +0.021). Both real and shuffled sit
    well above uniform (0.200), so motion differences are genuinely bursty — but a random
    pairing is equally bursty, which means the burstiness tracks "where either clip moves
    fast", not "where the edit happened". `keep_frac=1.0` disables the temporal
    restriction and reproduces the group-set behaviour exactly.
    """
    n_frames, _ = vmap.shape
    region = np.zeros_like(vmap, dtype=np.float32)
    if not groups:
        return region
    if keep_frac >= 1.0:
        region[:, groups] = 1.0
        return region
    activity = vmap[:, groups].sum(1)
    k = max(1, int(round(keep_frac * n_frames)))
    region[np.ix_(np.argsort(-activity)[:k], groups)] = 1.0
    return region


def _diff_item(args, key, cache_dir, text, encoder, group_channels, lateral_names):
    """One velocity-diff-derived label item, or None when the pair yields no groups."""
    from editing.masking import semantic_token_subset   # local: training must not import editing

    source = np.load(os.path.join(cache_dir, f"{key}_s.npy"))
    target = np.load(os.path.join(cache_dir, f"{key}_t.npy"))
    vmap = velocity_diff_map(source, target, group_channels)
    groups = diff_groups(vmap, args.diff_ratio, args.diff_max) if vmap is not None else []
    positions, labels = encoder.token_info(text)
    cols = semantic_token_subset(positions, labels)     # supervise the content words
    if not (groups and cols):
        return None

    tier1 = (args.diff_tier1 and len(groups) == 1
             and GROUP_NAMES[groups[0]] in lateral_names)
    item = {"W": list(cols), "S": list(groups),
            "tier": 1 if tier1 else 2, "lat": bool(tier1)}
    if args.diff_temporal < 1.0:
        item["M"] = diff_region(vmap, groups, args.diff_temporal)
    return item


def build_label_cache(args, keys, cache_dir, texts, encoder, config, group_mode,
                      group_channels):
    """{keyid: [item]} for `grounding_loss`, plus per-source counts.

    Keyed by KEYID, not by text: a diff-derived label depends on the motion pair, and two
    triplets can share an instruction.
    """
    lateral_names = {g for g in GROUP_NAMES if g.startswith(("left_", "right_"))}
    use_verbs = bool(config.get("attn_ground_verbs", False))
    parser_sources = ("parser_first", "parser_only")
    diff_sources = ("parser_first", "diff_only")
    cache, stats = {}, {"parser": 0, "diff": 0, "none": 0}

    for key in log.progress(keys, desc="labels", leave=True):
        text = texts[key]
        items = []
        if args.ground_labels in parser_sources:
            items = to_items(text, encoder.token_spans(text), group_mode,
                             include_verbs=use_verbs)
        if items:
            stats["parser"] += 1
        elif args.ground_labels in diff_sources:
            item = _diff_item(args, key, cache_dir, text, encoder, group_channels,
                              lateral_names)
            items = [item] if item else []
            stats["diff" if item else "none"] += 1
        else:
            stats["none"] += 1
        if items:
            cache[key] = items
    return cache, stats
