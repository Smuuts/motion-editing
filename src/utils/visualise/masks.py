"""
Figures for the implicit LEDITS++ masks (M1 cross-attention, M2 noise ψ).

`plot_mask_problem` / `plot_mask_quant` are the two panels written by
src/visualise_mask_problem.py; `save_mask_heatmap` is the small per-edit mask strip
written alongside every edit_motion.py render. The numbers they display are computed
in analysis/mask_probe.py — these functions only lay them out.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .heatmaps import (
    corr_matrix, ellipsis, heatmap, highlight_rows, mean_off_diagonal, save_figure,
    short_labels,
)


def _fg_heatmap(ax, fg, glabels, cmap, vmax, expected_idx):
    """One (F, G) map drawn as (G, F), with the expected rows outlined in red."""
    heatmap(ax, fg.T, ylabels=glabels, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    highlight_rows(ax, expected_idx, fg.shape[0])


def _shared_vmax(maps):
    """99th-percentile ceiling over ALL instruction rows: a shared colour scale is the
    whole point (identical colours = identical maps), and the percentile keeps one hot
    cell from flattening everything else."""
    return float(np.quantile(np.concatenate([m.ravel() for m in maps]), 0.99)) or 1.0


def plot_mask_problem(clip_id, caption, instructions, targets, m1_maps, m2_maps,
                      bin_maps, src_act, glabels, mask_mode, out_path):
    """Per-instruction grid of raw M1 / raw M2 / final binary mask, above the
    instruction-independent source-motion reference."""
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]
    col_vmax = [_shared_vmax(m1_maps), _shared_vmax(m2_maps), 1.0]
    col_titles = ["M1  raw cross-attention", "M2  raw noise ψ",
                  f"final binary mask  ({mask_mode})"]
    col_cmaps = ["magma", "magma", "gray"]

    H = 2.0 + 1.15 * n
    fig = plt.figure(figsize=(11, H))
    # Reserve ~0.95in at the top for the 3-line suptitle so it never collides with the
    # reference row's own title.
    gs = GridSpec(n + 1, 3, figure=fig, hspace=0.5, wspace=0.18,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 0.95 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    _fg_heatmap(ax_ref, src_act, glabels, "cividis", _shared_vmax([src_act]), [])
    ax_ref.set_title("SOURCE motion  |Δx0|  (instruction-INDEPENDENT reference — what "
                     "the implicit masks actually track)", fontsize=8.5, loc="left")

    for i in range(n):
        for c, maps in enumerate((m1_maps, m2_maps, bin_maps)):
            ax = fig.add_subplot(gs[i + 1, c])
            _fg_heatmap(ax, maps[i], glabels, col_cmaps[c], col_vmax[c], tgt_idx[i])
            if i == 0:
                ax.set_title(col_titles[c], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{instructions[i]}\n(expect: {', '.join(targets[i]) or '—'})",
                              fontsize=7.5, rotation=0, ha="right", va="center", labelpad=38)
            if i == n - 1:
                ax.set_xlabel("frame", fontsize=7)

    cap = ellipsis(caption, 80)
    fig.suptitle(
        "The mask problem — implicit M1/M2 masks are source-dynamics-driven, not "
        "instruction-driven\n"
        + (f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}")
        + "\n(rows barely differ and follow the source reference; red = the group each "
          "instruction SHOULD move)",
        fontsize=10, y=0.995)
    save_figure(fig, out_path)


def plot_mask_quant(clip_id, caption, instructions, m1_corr, m2_corr, m1_src, m2_src,
                    out_path):
    """Instruction×instruction correlation matrices for M1/M2 (off-diagonal ≈ 1 ⇒ the
    mask ignores the instruction) plus each mask's correlation with the source motion."""
    short = short_labels(instructions)
    n = len(instructions)

    fig = plt.figure(figsize=(12.5, 4.8))
    # A dedicated thin column for the colourbar keeps it clear of the bar chart, whose
    # own labels sit on its right edge (tick_right).
    gs = GridSpec(1, 4, figure=fig, wspace=0.45, top=0.78,
                  width_ratios=[1.0, 1.0, 0.07, 1.0])

    corr_matrix(fig.add_subplot(gs[0, 0]), np.asarray(m1_corr), short,
                "M1 map corr across instructions\nmean off-diag r = "
                f"{mean_off_diagonal(m1_corr):.2f}")
    im = corr_matrix(fig.add_subplot(gs[0, 1]), np.asarray(m2_corr), short,
                     "M2 map corr across instructions\nmean off-diag r = "
                     f"{mean_off_diagonal(m2_corr):.2f}")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), label="Pearson r")
    cb.ax.yaxis.set_ticks_position("left")     # numbers face the matrices, not the bars
    cb.ax.yaxis.set_label_position("left")

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(n)
    ax.barh(x - 0.2, m1_src, height=0.38, label="M1", color="#4c72b0")
    ax.barh(x + 0.2, m2_src, height=0.38, label="M2", color="#c44e52")
    ax.set_yticks(x)
    ax.set_yticklabels(short, fontsize=6)
    ax.yaxis.tick_right()                      # labels on the outer edge, clear of cbar
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("corr(mask, source |Δx0|)", fontsize=8)
    ax.set_title("Mask vs source motion\n(high ⇒ source-dynamics detector)", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.invert_yaxis()

    cap = ellipsis(caption, 80)
    fig.suptitle(f"Instruction-invariance of the implicit masks  ·  clip {clip_id}"
                 + (f'   source: "{cap}"' if cap else "")
                 + "\n(off-diagonal r ≈ 1 ⇒ the mask ignores the instruction)",
                 fontsize=10, y=0.99)
    save_figure(fig, out_path)


def save_mask_heatmap(masks, edits, glabels, out_path):
    """One (G, F) binary-mask panel per edit — the companion figure to an edit render."""
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 2.5 * len(glabels)), 3), squeeze=False)
    for ax, m, e in zip(axes[0], masks, edits):
        heatmap(ax, m["m_group"].cpu().numpy().T, title=ellipsis(e, 30),
                ylabels=glabels, cmap="viridis", vmin=0, vmax=1, aspect="auto",
                clear_ticks=False)
        ax.set_xlabel("frame")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=120, tight=False)
