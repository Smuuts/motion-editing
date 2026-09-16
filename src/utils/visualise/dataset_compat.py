"""
The HumanML3D / MotionFix compatibility figure.

One row per embedding space, three columns that answer the same question at decreasing
levels of hand-waving:

  t-SNE                 what it looks like (neighbourhoods only — gaps are not distances)
  nearest-neighbour     how far each corpus sits from the HumanML3D training manifold
  probe margin          the cross-validated separating direction, i.e. the AUC's own picture

The margin histogram is the one that carries weight. t-SNE renders any consistent offset as
two clean islands, which would overstate a separation that is really two OVERLAPPING clouds
sharing a direction; the margin panel shows that overlap as it actually is.
"""

import matplotlib.pyplot as plt
import numpy as np

from .heatmaps import save_figure

# Grey for the reference corpus, blue for the held-out control, red for the foreign one.
COLORS = {"HumanML3D train": "#9aa0a6",
          "HumanML3D held-out": "#4c72b0",
          "MotionFix": "#c44e52",
          "MotionFix (target)": "#dd8452"}
DEFAULT = "#8172b3"


def _color(name):
    return COLORS.get(name, DEFAULT)


def _scatter(ax, series, coords, title):
    # Draw the big reference cloud first so the small foreign series stays visible on top.
    order = sorted(range(len(series)), key=lambda i: -len(series[i]))
    for i in order:
        s, xy = series[i], coords[i]
        ax.scatter(xy[:, 0], xy[:, 1], s=5, alpha=0.55, linewidths=0,
                   color=_color(s.name), label=f"{s.name} (n={len(s)})")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_alpha(0.3)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.85, markerscale=2.0)


def _hist(ax, groups, title, xlabel, bins=40):
    lo = min(v.min() for v in groups.values())
    hi = max(v.max() for v in groups.values())
    edges = np.linspace(lo, hi, bins + 1)
    for name, v in groups.items():
        ax.hist(v, bins=edges, density=True, alpha=0.55, color=_color(name), label=name)
        ax.axvline(np.median(v), color=_color(name), lw=1.2, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=6.5, framealpha=0.85)


def plot_dataset_compat(panels, out_path, perplexity=30):
    """`panels`: one dict per embedding space with keys
    space, series, coords, nn (dict name->array or None), probe, note."""
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(14.5, 4.3 * n),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
                             squeeze=False)

    for r, p in enumerate(panels):
        _scatter(axes[r][0], p["series"], p["coords"],
                 f"{p['space']} — t-SNE (perplexity {perplexity})")

        if p.get("nn"):
            _hist(axes[r][1], p["nn"],
                  "cosine to NEAREST HumanML3D train sample",
                  "nearest-neighbour cosine")
        else:
            axes[r][1].axis("off")

        pr = p["probe"]
        m, y = pr["margin"], pr["labels"]
        _hist(axes[r][2], {p["control_name"]: m[y == 0], p["foreign_name"]: m[y == 1]},
              "cross-validated probe margin", "logistic decision value")
        axes[r][2].axvline(0.0, color="k", lw=0.8)
        axes[r][2].text(0.02, 0.97,
                        f"AUC {pr['auc']:.4f}\n"
                        f"balanced acc {100 * pr['balanced_acc']:.1f} %  (chance 50 %)\n"
                        f"kNN balanced acc {100 * pr['knn_balanced_acc']:.1f} %\n"
                        f"n = {pr['n_a']} vs {pr['n_b']}",
                        transform=axes[r][2].transAxes, va="top", ha="left", fontsize=7.5,
                        bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=3))

        axes[r][0].set_ylabel(p["space"], fontsize=10, labelpad=8)
        if p.get("note"):
            axes[r][0].text(0.0, -0.06, p["note"], transform=axes[r][0].transAxes,
                            fontsize=7, va="top", ha="left", color="0.35")

    # Deliberately descriptive, not a verdict: the motion panel's t-SNE looks mixed while
    # its probe separates at AUC 0.93, so a title asserting "the motions are compatible"
    # would be contradicted by the panel beside it.
    fig.suptitle("HumanML3D vs MotionFix — corpus separability in the two embedding spaces "
                 "the method uses", fontsize=12, y=0.995)
    # Two lines: as one it overruns the figure width and is clipped at both edges.
    fig.text(0.5, 0.004,
             "t-SNE preserves neighbourhoods, not distances — and it misleads BOTH ways here: "
             "the text panel's clean gap overstates a separation the margins show is partly\n"
             "overlapping, while the motion panel's mixed cloud understates one the probe finds "
             "at AUC 0.93. Only the right-hand panels are quantities.",
             ha="center", va="bottom", fontsize=7.5, color="0.35", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.032, 1, 0.985])
    save_figure(fig, out_path, tight=False)


def plot_perplexity_sweep(space, series, coords_by_perp, out_path):
    """Same data, three perplexities — the honesty check on any t-SNE claim."""
    perps = sorted(coords_by_perp)
    fig, axes = plt.subplots(1, len(perps), figsize=(4.6 * len(perps), 4.6), squeeze=False)
    for ax, p in zip(axes[0], perps):
        _scatter(ax, series, coords_by_perp[p], f"perplexity {p}")
    fig.suptitle(f"{space} — t-SNE perplexity sweep", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path, tight=False)
