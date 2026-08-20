"""
Decompose a mask's instruction-invariance into its LATERALITY and CATEGORY axes.

One mean off-diagonal correlation conflates two very different failures — "left arm"
vs "right arm" (same limb, other side) and "left arm" vs "left leg" (other limb) — and
the project's history predicts these move apart (category is recoverable, laterality
needs supervision). Splitting them is what makes a result readable.

Also computed here:
  align_*             share of the binary mask's active cells landing in the expected
                      group, against the chance rate |target| / G,
  laterality_contrast the named side's mass minus its mirror's, per instruction,
  category_contrast   PAIRED within-clip arm/leg mass shift — an arm-instruction minus
                      a leg-instruction on the SAME clip, so the source's own motion
                      bias cancels and only the instruction-driven part survives.
"""

import numpy as np

from utils.probe import flat_corr, group_profile, pairwise_corr
from .instructions import (
    CAT_PAIRS, DEFAULT_INSTRUCTIONS, DEFAULT_TARGETS, LAT_PAIRS, MIRROR,
)


def _axis_means(corr, pairs):
    return float(np.mean([corr[i][j] for i, j in pairs]))


def axis_stats(maps, glabels, instructions=DEFAULT_INSTRUCTIONS,
               targets=DEFAULT_TARGETS) -> dict:
    """The per-family statistics for ONE set of (F, G) maps, one per instruction.

    Factored out of `decompose` so anything else that produces a per-instruction (F, G)
    map — the generation-space divergence, say — reports the *same* quantities
    computed the *same* way, which is the only reason numbers from different probes can
    be put in one table.

    Keys are unprefixed; `decompose` prefixes them with "m1_"/"m2_".
    """
    corr = pairwise_corr(maps)
    profiles = [group_profile(m) for m in maps]
    arm_instr = (profiles[0] + profiles[1]) / 2
    leg_instr = (profiles[2] + profiles[3]) / 2
    ai = [glabels.index("left_arm"), glabels.index("right_arm")]
    li = [glabels.index("left_leg"), glabels.index("right_leg")]
    return {
        "corr": corr.tolist(),
        "profile": {e: p.tolist() for e, p in zip(instructions, profiles)},
        "r_laterality": _axis_means(corr, LAT_PAIRS),
        "r_category": _axis_means(corr, CAT_PAIRS),
        "r_offdiag": float(np.mean(
            [corr[i][j] for i in range(len(corr)) for j in range(len(corr)) if i != j])),
        "laterality_contrast": [
            float(p[glabels.index(t[0])] - p[glabels.index(MIRROR[t[0]])])
            for p, t in zip(profiles, targets)
        ],
        "category_contrast": {
            "arm_mass_under_arm_instr": float(arm_instr[ai].sum()),
            "arm_mass_under_leg_instr": float(leg_instr[ai].sum()),
            "leg_mass_under_leg_instr": float(leg_instr[li].sum()),
            "leg_mass_under_arm_instr": float(arm_instr[li].sum()),
            "arm_shift": float(arm_instr[ai].sum() - leg_instr[ai].sum()),
            "leg_shift": float(leg_instr[li].sum() - arm_instr[li].sum()),
        },
    }


def alignment(binary_maps, glabels, targets=DEFAULT_TARGETS) -> list[float]:
    """Per instruction: the share of a binary mask's active cells landing in its
    expected group(s). Chance is |target| / G — 0.143 for one group out of 7."""
    hits = []
    for m, tgt in zip(binary_maps, targets):
        idx = [glabels.index(g) for g in tgt if g in glabels]
        total = m.sum()
        hits.append(float(m[:, idx].sum() / total) if total >= 1 and idx else 0.0)
    return hits


def recall(binary_maps, glabels, targets=DEFAULT_TARGETS) -> list[float]:
    """Per instruction: the share of the TARGET group's cells the mask keeps.

    `alignment` is precision, and precision is bought for free by shrinking a mask — so
    the two must always be read together, with the cell count beside them. A mask
    scoring 1.0 alignment on three cells has not solved anything: the editor needs
    enough of the target region to actually change it.
    """
    out = []
    for m, tgt in zip(binary_maps, targets):
        idx = [glabels.index(g) for g in tgt if g in glabels]
        cells = m[:, idx]
        out.append(float(cells.sum() / cells.size) if idx and cells.size else 0.0)
    return out


def decompose(m1_maps, m2_maps, binaries, src_act, glabels,
              instructions=DEFAULT_INSTRUCTIONS, targets=DEFAULT_TARGETS) -> dict:
    """All per-clip statistics for one checkpoint × one clip, as a JSON-ready dict.

    m1_maps / m2_maps : (F, G) arrays, one per instruction (from mask_probe).
    binaries          : {mask_mode: [(F, G) binary maps]} — m1_only isolates what a
                        grounded attention readout would drive, m2_only is the default.
    """
    res = {
        "group_labels": glabels,
        "instructions": list(instructions),
        "m1_src_corr": [flat_corr(m, src_act) for m in m1_maps],
        "m2_src_corr": [flat_corr(m, src_act) for m in m2_maps],
        "src_profile": group_profile(src_act).tolist(),
        "align_chance": 1.0 / len(glabels),
    }
    for key, maps in (("m1", m1_maps), ("m2", m2_maps)):
        res.update({f"{key}_{k}": v
                    for k, v in axis_stats(maps, glabels, instructions, targets).items()})
    for mode, maps in binaries.items():
        res[f"align_{mode}"] = alignment(maps, glabels, targets)
        res[f"recall_{mode}"] = recall(maps, glabels, targets)
        res[f"cells_{mode}"] = [int(np.asarray(m).sum()) for m in maps]
    return res


def summary_row(res: dict) -> list[float]:
    """The comparison table's columns, in header order.

    `align_attn` (M1 ∩ M2 — the composed mask the editor actually uses) is nan on
    results written before it was collected, so old JSONs stay readable.
    """
    return [res["m1_r_category"], res["m1_r_laterality"], res["m1_r_offdiag"],
            res["m2_r_category"], res["m2_r_laterality"], res["m2_r_offdiag"],
            float(np.mean(res["m1_src_corr"])), float(np.mean(res["m2_src_corr"])),
            float(np.mean(res["align_m1_only"])), float(np.mean(res["align_m2_only"])),
            float(np.mean(res["align_attn"])) if "align_attn" in res else float("nan"),
            float(np.mean(res["recall_attn"])) if "recall_attn" in res else float("nan"),
            float(np.mean(res["cells_attn"])) if "cells_attn" in res else float("nan")]
