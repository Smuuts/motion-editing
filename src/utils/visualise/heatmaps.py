"""
Generic figure primitives shared by the probe/analysis figures.

Everything here is axis-level (it draws into an `ax` you own) so the probe modules
keep control of their own layout; only `heatmap` and `corr_matrix` know how the
project's heatmaps are labelled and annotated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def ellipsis(text: str, width: int) -> str:
    """Trim to `width` characters with a trailing … ('' stays '')."""
    t = (text or "").strip()
    return t if len(t) <= width else t[:width - 1] + "…"


def short_labels(labels, width=18):
    return [ellipsis(l, width) for l in labels]


def _set_ticks(setter, labeller, labels, rotation, clear):
    """Label an axis, or clear its ticks when there are no / too many labels."""
    if labels is not None and len(labels) <= 24:
        setter(range(len(labels)))
        labeller(labels, fontsize=6, rotation=rotation,
                 **({"ha": "right"} if rotation else {}))
    elif clear:
        setter([])


def heatmap(ax, M, title=None, ylabels=None, xlabels=None, cmap="magma",
            vmin=None, vmax=None, annotate=False, aspect="equal", clear_ticks=True):
    """Draw M with independently-labelled axes.

    The two axes are not always the same axis (group×group and frame×frame are square,
    but a per-cell map is group×frame), so x and y labels are passed separately.
    `annotate` writes the cell values — only legible on small matrices.
    `clear_ticks=False` keeps matplotlib's default numeric ticks on unlabelled axes.
    """
    im = ax.imshow(M, cmap=cmap, interpolation="nearest", aspect=aspect,
                   vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=8.5)
    _set_ticks(ax.set_yticks, ax.set_yticklabels, ylabels, 0, clear_ticks)
    _set_ticks(ax.set_xticks, ax.set_xticklabels, xlabels, 45, clear_ticks)
    if annotate and ylabels is not None and xlabels is not None and len(ylabels) <= 8:
        mx = M.max()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] < 0.6 * mx else "black")
    return im


def highlight_rows(ax, rows, width):
    """Outline whole rows in red and bold their y-labels — used to mark the body-part
    group an instruction is *supposed* to move."""
    for r in rows:
        ax.add_patch(Rectangle((-0.5, r - 0.5), width, 1, fill=False,
                               edgecolor="red", lw=1.6))
        ax.get_yticklabels()[r].set_color("red")
        ax.get_yticklabels()[r].set_fontweight("bold")


def corr_matrix(ax, M, labels, title):
    """Symmetric correlation matrix on a fixed [-1, 1] diverging scale, cells annotated.
    The fixed scale is what makes "≈1 everywhere ⇒ invariant" readable at a glance."""
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=40, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    ax.set_title(title, fontsize=8.5)
    return im


def mean_off_diagonal(M) -> float:
    """Mean of a square matrix's off-diagonal entries (nan for a 1×1 matrix)."""
    M = np.asarray(M)
    if len(M) < 2:
        return float("nan")
    return float(M[~np.eye(len(M), dtype=bool)].mean())


def save_figure(fig, out_path, dpi=130, tight=True):
    fig.savefig(out_path, dpi=dpi, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print(f"Wrote {out_path}")
