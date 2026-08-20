"""
Figures for the generation-space differential mask: per-instruction divergence maps with
their group profiles, and the pooled forced-choice summary.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .heatmaps import (corr_matrix, ellipsis, fg_heatmap, heatmap, mean_off_diagonal,
                       save_figure, shared_vmax, short_labels)

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
    """Per-instruction generation-space divergence D, over the reference
    generation's own motion energy (the instruction-INDEPENDENT row: if D just tracks it,
    the differencing bought nothing)."""
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]
    vmax = shared_vmax(maps)
    chance = 1.0 / len(glabels)

    H = 2.1 + 1.15 * n
    fig = plt.figure(figsize=(10, H))
    gs = GridSpec(n + 1, 2, figure=fig, hspace=0.5, wspace=0.12,
                  height_ratios=[0.9] + [1.0] * n, width_ratios=[2.6, 1.0],
                  top=1 - 0.95 / H, bottom=0.06)

    ax_ref = fig.add_subplot(gs[0, 0])
    fg_heatmap(ax_ref, ref_energy, glabels, "cividis", shared_vmax([ref_energy]), [])
    ax_ref.set_title(f'REFERENCE generation |Δ|  (prompt: "{ellipsis(reference, 46)}") — '
                     "instruction-independent", fontsize=8.5, loc="left")
    ref_marginal = np.asarray(ref_energy).mean(axis=0)
    ax_b = fig.add_subplot(gs[0, 1])
    _profile_bars(ax_b, ref_marginal / max(ref_marginal.sum(), 1e-12), glabels, [], chance)
    ax_b.set_title("group profile\n(dashed = chance)", fontsize=8)

    for i in range(n):
        ax = fig.add_subplot(gs[i + 1, 0])
        fg_heatmap(ax, maps[i], glabels, "magma", vmax, tgt_idx[i])
        ax.set_ylabel(f"{instructions[i]}\n(expect: {', '.join(targets[i]) or '—'})",
                      fontsize=7.5, rotation=0, ha="right", va="center", labelpad=38)
        if i == 0:
            ax.set_title("D = |g_edit − g_ref|  (shared noise)", fontsize=8.5)
        if i == n - 1:
            ax.set_xlabel("frame", fontsize=7)
        _profile_bars(fig.add_subplot(gs[i + 1, 1]), profiles[i], glabels, tgt_idx[i],
                      chance)

    fig.suptitle("Generation-space differential mask\n"
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

    fig.suptitle("Gate — does the generator's own output carry the instruction "
                 "(and its laterality)?", fontsize=10, y=0.97)
    save_figure(fig, out_path)
