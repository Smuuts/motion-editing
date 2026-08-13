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
                      bin_maps, src_act, glabels, mask_mode, out_path,
                      invariance=None):
    """Per-instruction grid of raw M1 / raw M2 / final binary mask, above the
    instruction-independent source-motion reference.

    `invariance` is the measured (M1, M2) mean off-diagonal r. Pass it: the title is
    then a statement of what THIS figure shows rather than a restatement of the
    project's headline negative, which stopped being true for every checkpoint once the
    grounding loss landed (a grounded M1 gives visibly different rows, and a caption
    reading "rows barely differ" over a figure where they plainly do is worse than no
    caption at all).
    """
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
    if invariance is None:
        head = ("The mask problem — implicit M1/M2 masks are source-dynamics-driven, "
                "not instruction-driven")
        foot = ("(rows barely differ and follow the source reference; red = the group "
                "each instruction SHOULD move)")
    else:
        m1_r, m2_r = invariance
        head = ("Implicit M1/M2 masks across contrasting instructions  ·  "
                f"instruction-invariance r: M1 {m1_r:.2f}, M2 {m2_r:.2f}")
        foot = ("(r → 1 = the mask ignores the instruction; red = the group each "
                "instruction SHOULD move)")
    fig.suptitle(
        head + "\n"
        + (f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}")
        + "\n" + foot,
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
    raw_vmax = _shared_vmax(raw_maps)
    lim = float(np.quantile(np.abs(np.concatenate([m.ravel() for m in corr_maps])), 0.99)) or 1.0

    H = 2.2 + 1.15 * n
    fig = plt.figure(figsize=(14, H))
    gs = GridSpec(n + 1, 4, figure=fig, hspace=0.55, wspace=0.2,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 1.05 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    _fg_heatmap(ax_ref, src_act, glabels, "cividis", _shared_vmax([src_act]), [])
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
    abs_vmax, m1_vmax = _shared_vmax(psi_abs), _shared_vmax(m1_maps)
    lim = float(np.quantile(np.abs(np.concatenate([m.ravel() for m in psi_energy])),
                            0.99)) or 1.0

    H = 2.2 + 1.15 * n
    fig = plt.figure(figsize=(14, H))
    gs = GridSpec(n + 1, 4, figure=fig, hspace=0.55, wspace=0.2,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 1.05 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    _fg_heatmap(ax_ref, src_act, glabels, "cividis", _shared_vmax([src_act]), [])
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


def _profile_bars(ax, profile, glabels, expected_idx, chance):
    """Group profile as horizontal bars, expected group(s) in red, chance line marked.
    Row order matches the heatmaps to its left (group 0 on top)."""
    colors = ["#c44e52" if i in expected_idx else "#9aa0a6" for i in range(len(glabels))]
    ax.barh(np.arange(len(glabels)), profile, color=colors)
    ax.axvline(chance, color="k", lw=0.7, ls="--")
    ax.set_yticks(range(len(glabels)))
    ax.set_yticklabels(glabels, fontsize=6)
    ax.yaxis.tick_right()
    ax.set_xlim(0, max(0.6, float(np.max(profile)) * 1.15))
    ax.tick_params(axis="x", labelsize=6)
    ax.invert_yaxis()


def plot_gen_diff(instructions, targets, maps, profiles, ref_energy, glabels,
                  reference, out_path, title_extra=""):
    """Option 6's per-instruction generation-space divergence D, over the reference
    generation's own motion energy (the instruction-INDEPENDENT row: if D just tracks it,
    the differencing bought nothing)."""
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]
    vmax = _shared_vmax(maps)
    chance = 1.0 / len(glabels)

    H = 2.1 + 1.15 * n
    fig = plt.figure(figsize=(10, H))
    gs = GridSpec(n + 1, 2, figure=fig, hspace=0.5, wspace=0.12,
                  height_ratios=[0.9] + [1.0] * n, width_ratios=[2.6, 1.0],
                  top=1 - 0.95 / H, bottom=0.06)

    ax_ref = fig.add_subplot(gs[0, 0])
    _fg_heatmap(ax_ref, ref_energy, glabels, "cividis", _shared_vmax([ref_energy]), [])
    ax_ref.set_title(f'REFERENCE generation |Δ|  (prompt: "{ellipsis(reference, 46)}") — '
                     "instruction-independent", fontsize=8.5, loc="left")
    ref_marginal = np.asarray(ref_energy).mean(axis=0)
    ax_b = fig.add_subplot(gs[0, 1])
    _profile_bars(ax_b, ref_marginal / max(ref_marginal.sum(), 1e-12), glabels, [], chance)
    ax_b.set_title("group profile\n(dashed = chance)", fontsize=8)

    for i in range(n):
        ax = fig.add_subplot(gs[i + 1, 0])
        _fg_heatmap(ax, maps[i], glabels, "magma", vmax, tgt_idx[i])
        ax.set_ylabel(f"{instructions[i]}\n(expect: {', '.join(targets[i]) or '—'})",
                      fontsize=7.5, rotation=0, ha="right", va="center", labelpad=38)
        if i == 0:
            ax.set_title("D = |g_edit − g_ref|  (shared noise)", fontsize=8.5)
        if i == n - 1:
            ax.set_xlabel("frame", fontsize=7)
        _profile_bars(fig.add_subplot(gs[i + 1, 1]), profiles[i], glabels, tgt_idx[i],
                      chance)

    fig.suptitle("Option 6 — generation-space differential mask\n"
                 "does diffing two shared-noise generations localise the instruction?"
                 + (f"\n{title_extra}" if title_extra else ""), fontsize=10, y=0.995)
    save_figure(fig, out_path)


def plot_gen_diff_summary(instructions, readouts, verdicts, glabels, out_path):
    """The gate, in one panel: pooled forced-choice accuracies per readout against their
    chance lines, the paired readout's instruction-invariance, and its profile matrix."""
    short = short_labels(instructions)
    names = list(readouts)

    fig = plt.figure(figsize=(13, 4.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42, top=0.74, bottom=0.16)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    for k, (key, color) in enumerate((("laterality", "#c44e52"), ("category", "#4c72b0"),
                                      ("top1", "#8172b2"))):
        acc = [verdicts[n][key]["accuracy"] for n in names]
        err = np.array([[a - verdicts[n][key]["ci95"][0] for a, n in zip(acc, names)],
                        [verdicts[n][key]["ci95"][1] - a for a, n in zip(acc, names)]])
        ax.bar(x + (k - 1) * 0.27, acc, width=0.26, color=color, label=key,
               yerr=np.clip(err, 0, None), capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.axhline(1.0 / len(glabels), color="#8172b2", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (95% CI)", fontsize=8)
    ax.set_title("Forced choices vs chance\n(dashed 0.5 = side/limb, dotted = top-1)",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="upper right")

    paired = readouts[names[0]]
    corr_matrix(fig.add_subplot(gs[0, 1]), np.asarray(paired["corr"]), short,
                f"{names[0]} D corr across instructions\nmean off-diag r = "
                f"{mean_off_diagonal(paired['corr']):.2f}")

    ax = fig.add_subplot(gs[0, 2])
    prof = np.array([paired["profile"][e] for e in instructions])
    heatmap(ax, prof, ylabels=short, xlabels=glabels, cmap="magma", vmin=0,
            annotate=True, aspect="auto")
    ax.set_title(f"{names[0]} group profile per instruction", fontsize=8.5)

    fig.suptitle("Option 6 gate — does the generator's own output carry the instruction "
                 "(and its laterality)?", fontsize=10, y=0.97)
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
