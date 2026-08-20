"""
Figures for the two mask-CORRECTION experiments: removing the source-motion component
from a map, and reading psi with a sign instead of an absolute value.

Both draw the same four-column shape — raw map, corrected map, raw mask, corrected mask —
because the question in both cases is "did the operation change the BINARY mask, or only
the map underneath it". `plot_correction_sweep` is the lambda-sweep companion.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .heatmaps import (ellipsis, fg_heatmap, heatmap, highlight_rows, save_figure,
                       shared_vmax)

def plot_source_correction(clip_id, caption, instructions, targets, raw_maps,
                           corr_maps, raw_bins, corr_bins, src_act, glabels, subtitle,
                           out_path, align_raw=None, align_corr=None, mask_mode="m2_only"):
    """Raw vs source-corrected map, and the binary mask each one produces.

    The four columns are the whole argument in one picture: if removing the source
    helps, column 2 should look less like the reference row than column 1 does, and
    column 4 should put more of its white inside the red boxes than column 3 does.
    The corrected column uses a DIVERGING scale centred on 0 because a corrected map is
    signed — "below the source's expectation" is a real value, not an absence.
    """
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]
    raw_vmax = shared_vmax(raw_maps)
    lim = float(np.quantile(np.abs(np.concatenate([m.ravel() for m in corr_maps])), 0.99)) or 1.0

    H = 2.2 + 1.15 * n
    fig = plt.figure(figsize=(14, H))
    gs = GridSpec(n + 1, 4, figure=fig, hspace=0.55, wspace=0.2,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 1.05 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    fg_heatmap(ax_ref, src_act, glabels, "cividis", shared_vmax([src_act]), [])
    ax_ref.set_title("SOURCE motion  |Δx0|  (the instruction-INDEPENDENT component being "
                     "removed)", fontsize=8.5, loc="left")

    titles = [f"raw {mask_mode.split('_')[0].upper()} map", "source-corrected map",
              f"mask from raw  ({mask_mode})", f"mask from corrected  ({mask_mode})"]
    for i in range(n):
        panels = [(raw_maps[i], "magma", 0.0, raw_vmax),
                  (corr_maps[i], "RdBu_r", -lim, lim),
                  (raw_bins[i], "gray", 0.0, 1.0),
                  (corr_bins[i], "gray", 0.0, 1.0)]
        for c, (m, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[i + 1, c])
            heatmap(ax, np.asarray(m).T, ylabels=glabels, cmap=cmap, vmin=vmin,
                    vmax=vmax, aspect="auto")
            highlight_rows(ax, tgt_idx[i], np.asarray(m).shape[0])
            if i == 0:
                ax.set_title(titles[c], fontsize=8.5)
            # Alignment belongs on the mask panels only — it is a property of the
            # thresholded mask, not of the map it came from.
            if c == 2 and align_raw is not None:
                ax.set_xlabel(f"align {align_raw[i]:.3f}", fontsize=7)
            if c == 3 and align_corr is not None:
                ax.set_xlabel(f"align {align_corr[i]:.3f}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{instructions[i]}\n(expect: {', '.join(targets[i]) or '—'})",
                              fontsize=7.5, rotation=0, ha="right", va="center", labelpad=38)

    cap = ellipsis(caption, 80)
    fig.suptitle(
        f"Source-corrected mask  ·  {subtitle}\n"
        + (f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}")
        + "\n(red = the group each instruction SHOULD move; chance alignment = "
          f"{1 / len(glabels):.3f})",
        fontsize=10, y=0.995)
    save_figure(fig, out_path)


def plot_psi_sign(clip_id, caption, instructions, targets, psi_abs, psi_energy, m1_maps,
                  energy_masks, src_act, glabels, out_path, align_abs=None,
                  align_energy=None):
    """ψ as a magnitude vs ψ as a SIGNED energy change, and the mask the sign produces.

    Column 2 is the figure's whole argument and is drawn on a diverging scale centred on
    zero: red = the instruction makes this group move MORE than the reference does, blue
    = less. If the mixture hypothesis holds, the red sits on the instructed group and the
    blue sits on whatever the source clip is busy doing — two things column 1 shows as
    one undifferentiated bright region.
    """
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]
    abs_vmax, m1_vmax = shared_vmax(psi_abs), shared_vmax(m1_maps)
    lim = float(np.quantile(np.abs(np.concatenate([m.ravel() for m in psi_energy])),
                            0.99)) or 1.0

    H = 2.2 + 1.15 * n
    fig = plt.figure(figsize=(14, H))
    gs = GridSpec(n + 1, 4, figure=fig, hspace=0.55, wspace=0.2,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 1.05 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    fg_heatmap(ax_ref, src_act, glabels, "cividis", shared_vmax([src_act]), [])
    ax_ref.set_title("SOURCE motion  |Δx0|  (what the edit is being asked to change)",
                     fontsize=8.5, loc="left")

    titles = ["ψ  |x̂0_c − x̂0_ref|   (magnitude)",
              "ΔE  signed energy change\nred = edit ADDS motion, blue = SUPPRESSES",
              "M1  grounded cross-attention", "mask  M1 ∩ ΔE  (size-matched)"]
    for i in range(n):
        panels = [(psi_abs[i], "magma", 0.0, abs_vmax),
                  (psi_energy[i], "RdBu_r", -lim, lim),
                  (m1_maps[i], "magma", 0.0, m1_vmax),
                  (energy_masks[i], "gray", 0.0, 1.0)]
        for c, (m, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[i + 1, c])
            heatmap(ax, np.asarray(m).T, ylabels=glabels, cmap=cmap, vmin=vmin,
                    vmax=vmax, aspect="auto")
            highlight_rows(ax, tgt_idx[i], np.asarray(m).shape[0])
            if i == 0:
                ax.set_title(titles[c], fontsize=8.5)
            # Both alignments sit under the ΔE-mask panel: they are the SAME mask recipe
            # differing only in ψ, so putting them side by side is the comparison.
            if c == 3 and align_energy is not None:
                lbl = f"M1∩ΔE align {align_energy[i]:.3f}"
                if align_abs is not None:
                    lbl += f"   (M1∩ψ {align_abs[i]:.3f})"
                ax.set_xlabel(lbl, fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{instructions[i]}\n(expect: {', '.join(targets[i]) or '—'})",
                              fontsize=7.5, rotation=0, ha="right", va="center", labelpad=38)

    cap = ellipsis(caption, 80)
    fig.suptitle(
        "Is ψ a mixture of 'the edit adds motion' and 'the edit suppresses the source'?\n"
        + (f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}")
        + "\n(red outline = the group each instruction SHOULD move; chance alignment = "
          f"{1 / len(glabels):.3f})",
        fontsize=10, y=0.995)
    save_figure(fig, out_path)


def plot_correction_sweep(curves, chance, out_path, title):
    """λ-sweep curves: alignment and instruction-invariance, real vs shuffled controls.

    Real-vs-control is drawn on the SAME axes on purpose — the question this figure
    exists to answer is not "does the number move" but "does it move more than a
    scrambled source moves it", and that is only readable when the two curves share a
    scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [("align", "binary-mask alignment with the target group", chance),
              ("r_laterality", "instruction-invariance r — laterality", None),
              ("r_category", "instruction-invariance r — category", None)]
    styles = {"real": dict(color="#c44e52", lw=2.0, marker="o", ms=3.5),
              "shuffle_cells": dict(color="#4c72b0", lw=1.2, ls="--", marker="", ms=0),
              "shuffle_groups": dict(color="#55a868", lw=1.2, ls=":", marker="", ms=0)}

    for ax, (key, label, ref) in zip(axes, panels):
        for control, series in curves.items():
            lams = sorted(series)
            ax.plot(lams, [series[l][key] for l in lams], label=control,
                    **styles.get(control, {}))
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8, ls="-.", label="chance")
        ax.set_xlabel("λ  (0 = untouched map)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_path, tight=False)
