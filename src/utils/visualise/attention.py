"""
Figures for the self-attention probe (src/analyse_self_attention.py).

Two panels: the affinity structure (is self-attention body-part / temporally
structured?) and the DiffSeg segmentation with its threshold sweep and the
text-invariance check. All numbers come from analysis/self_attention.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .heatmaps import corr_matrix, ellipsis, heatmap, mean_off_diagonal, save_figure, short_labels


def plot_structure(clip_id, caption, C, R, layer_diag, glabels, anchors, F, G,
                   src_act, diag_group, diag_frame, out_path):
    """Group affinity C, frame affinity R, the per-layer diagonality profile, and two
    example anchor maps against the source's own motion."""
    fig = plt.figure(figsize=(13, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.3, top=0.86, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    im = heatmap(ax, C, f"group affinity C  (row-normalised)\n"
                        f"diagonality {diag_group:.3f}  vs random {1/G:.3f}",
                 ylabels=glabels, xlabels=glabels, annotate=True)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = heatmap(ax, R, f"frame affinity R  (row-normalised)\n"
                        f"diagonality {diag_frame:.4f}  vs random {1/F:.4f}")
    ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("frame", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(range(len(layer_diag)), layer_diag, "o-", color="#4c72b0", label="group")
    ax.axhline(1 / G, color="k", ls="--", lw=0.8, label=f"random 1/G = {1/G:.3f}")
    ax.set_xlabel("block index", fontsize=7)
    ax.set_ylabel("group-affinity diagonality", fontsize=7)
    ax.set_title("Per-layer body-part structure\n(is any single block grounded?)",
                 fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[1, 0])
    heatmap(ax, src_act.T, "SOURCE motion |Δx0|  (reference)",
            ylabels=glabels, cmap="cividis", aspect="auto")
    ax.set_xlabel("frame", fontsize=7)

    for k, (row, name) in enumerate(anchors[:2]):
        ax = fig.add_subplot(gs[1, 1 + k])
        heatmap(ax, row.reshape(F, G).T,
                f"attention FROM cell {name}\n(where does one token look?)",
                ylabels=glabels, aspect="auto")
        ax.set_xlabel("frame", fontsize=7)

    cap = ellipsis(caption, 80)
    fig.suptitle("Self-attention structure (text-free readout)  ·  clip "
                 f"{clip_id}" + (f'  ·  "{cap}"' if cap else "") + "\n"
                 "diagonal C ⇒ body-part structured;  blocked R ⇒ temporally segmented",
                 fontsize=10, y=0.97)
    save_figure(fig, out_path)


def plot_segments(clip_id, labels_fg, glabels, metrics, instr_corr, instructions,
                  out_path):
    """The DiffSeg label map, the merge-threshold sweep, alignment vs the body-part /
    time axes, and the instruction×instruction affinity correlation."""
    sweep = metrics.get("tau_sweep", [])
    fig = plt.figure(figsize=(16, 4.6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.42, top=0.76, bottom=0.2,
                  width_ratios=[1.5, 1.1, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    heatmap(ax, labels_fg.T, f"DiffSeg segmentation  ({metrics['n_segments']} segments)\n"
                             "vertical bands = temporal; horizontal bands = body-part",
            ylabels=glabels, cmap="tab20", aspect="auto", clear_ticks=False)
    ax.set_xlabel("frame", fontsize=7)

    # The threshold sweep: a DiffSeg port lives or dies on tau, so show the whole curve
    # rather than one point. Flat-zero NMI across every segment count is a far stronger
    # negative than a single bad setting.
    ax = fig.add_subplot(gs[0, 1])
    if sweep:
        nseg = [r["n_segments"] for r in sweep]
        ax.plot(nseg, [r["nmi_group_gap"] for r in sweep], "o-", ms=3,
                color="#4c72b0", label="body part")
        ax.plot(nseg, [r["nmi_time_gap"] for r in sweep], "s-", ms=3,
                color="#dd8452", label="time")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(metrics["n_segments"], color="k", ls="--", lw=0.8,
                   label="operating point")
        ax.set_xscale("log")
        ax.set_xlabel("segments (varying merge tau)", fontsize=7)
        ax.set_ylabel("NMI − shuffled baseline", fontsize=7)
        ax.legend(fontsize=6)
    ax.set_title("Threshold sweep\n(is ANY tau informative?)", fontsize=8.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 2])
    names = ["NMI vs\nbody part", "NMI vs\ntime bins"]
    vals  = [metrics["nmi_group"], metrics["nmi_time"]]
    base  = [metrics["nmi_group_shuffled"], metrics["nmi_time_shuffled"]]
    x = np.arange(len(names))
    ax.bar(x - 0.18, vals, width=0.34, color="#4c72b0", label="measured")
    ax.bar(x + 0.18, base, width=0.34, color="#bbbbbb", label="shuffled baseline")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, max(0.35, max(vals + base) * 1.25))
    ax.set_title("Does the segmentation align with\nthe body-part / time axes?",
                 fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 3])
    im = corr_matrix(ax, instr_corr, short_labels(instructions),
                     "Self-attention across instructions\nmean off-diag r = "
                     f"{mean_off_diagonal(instr_corr):.3f} (→1 = text-free)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    lat = metrics.get("segment_laterality", {})
    lat_txt = ("   ·   P(same segment) mirror pair "
               f"{lat['p_same_segment_mirror_pair']:.2f} vs other pairs "
               f"{lat['p_same_segment_other_pairs']:.2f} "
               f"(at {lat.get('n_segments', '?')} segments)") if lat else ""
    fig.suptitle(f"Self-attention segmentation  ·  clip {clip_id}{lat_txt}",
                 fontsize=10, y=0.95)
    save_figure(fig, out_path)
