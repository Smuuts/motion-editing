"""
Figures for the implicit LEDITS++ masks (M1 cross-attention, M2 noise psi).

`plot_mask_problem` / `plot_mask_quant` are the two panels written by
src/visualise_mask_problem.py; `plot_mask_noise_bands` is the noise-level twin of the
first, written by src/visualise_mask_noise_bands.py (rows are stages of the inversion
instead of instructions); `save_mask_heatmap` is the small per-edit mask strip written
alongside every edit_motion.py render. The numbers they display are computed in
analysis/mask_probe.py — these functions only lay them out.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .heatmaps import (corr_matrix, ellipsis, fg_heatmap, heatmap, highlight_rows,
                       mean_off_diagonal, save_figure, shared_vmax, short_labels)

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
    col_vmax = [shared_vmax(m1_maps), shared_vmax(m2_maps), 1.0]
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
    fg_heatmap(ax_ref, src_act, glabels, "cividis", shared_vmax([src_act]), [])
    ax_ref.set_title("SOURCE motion  |Δx0|  (instruction-INDEPENDENT reference — what "
                     "the implicit masks actually track)", fontsize=8.5, loc="left")

    for i in range(n):
        for c, maps in enumerate((m1_maps, m2_maps, bin_maps)):
            ax = fig.add_subplot(gs[i + 2, c])
            fg_heatmap(ax, maps[i], glabels, col_cmaps[c], col_vmax[c], tgt_idx[i])
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


def _panel_vmax(m, signed=False) -> float:
    """Per-panel colour ceiling.

    The unsigned case is exactly `shared_vmax`'s rule applied to one panel (99th
    percentile of the RAW values, 0-anchored), so a band row renders on the same footing
    as a `plot_mask_problem` row and the two figures can be read side by side.

    A SIGNED panel is cut at the 98th percentile of |m| instead: a symmetric scale spends
    half its colour range on each side of zero, so the same ceiling buys half the dynamic
    range per side."""
    if not signed:
        return shared_vmax([np.asarray(m)])
    v = float(np.quantile(np.abs(np.asarray(m)), 0.98))
    return v if v > 1e-12 else (float(np.abs(m).max()) or 1.0)


def _band_panel(ax, fg, glabels, cmap, vmax, tgt_idx, signed):
    """One (F, G) band map drawn as (G, F), `fg_heatmap`-style.

    `signed` draws a symmetric diverging scale centred on 0, and it is ON by default for
    ψ here — unlike `plot_mask_problem`, which renders ψ 0-anchored in magma. The
    difference is measured, not stylistic. `plot_mask_problem`'s rows are FULL-TRAJECTORY
    ψ maps, 16-40 % of whose cells are negative, so a 0-anchored ramp loses a minority of
    them. A narrow BAND's rows are 60-79 % negative at low and mid noise (clip 012698,
    exp_smplh_verbs: 0.79 at t∈[51,150] vs 0.40 full-sweep on "raise the left arm"),
    because the full-trajectory sum is dominated by the t>900 steps where |ψ| is 6-10x
    larger and mostly positive. Blacking those out turns "the edit strongly stills the
    source here" into "nothing here" for four cells in five, and makes a sign flip look
    like a magnitude ramp — which is the one structure this figure exists to show.
    `--psi_magma` restores the 0-anchored rendering when matching the other figure
    matters more."""
    vmin = -vmax if signed else 0.0
    heatmap(ax, np.asarray(fg).T, ylabels=glabels, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto")
    highlight_rows(ax, tgt_idx, np.asarray(fg).shape[0])


def _magnitude_tag(ax, m):
    """The raw mean |value| of a panel, in its top-right corner.

    Load-bearing under the default per-band colour scale: that scale is what makes the
    SHAPES comparable across rows, and it does so by throwing away exactly the thing
    that varies most across the trajectory. ψ's magnitude spans orders of magnitude
    between t=1 and t=999; without this number a reader would take two equally bright
    rows for two equally strong signals."""
    ax.text(0.995, 0.96, f"mean|·| {np.abs(np.asarray(m)).mean():.1e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.0, edgecolor="none"))


def _band_summary(axes, m1_bands, m2_bands, src_act, labels):
    """The two curves under the grid: where along the trajectory each mask's MAGNITUDE
    lives, and how its coupling to the source clip's own motion evolves with noise.

    The magnitude curve is normalised per mask (each to its own maximum) because M1 and
    M2 are not in the same units and never were — the question it answers is "which
    bands does this mask's average actually consist of", which is a within-mask
    question. It is the figure's direct read on the sweep-weighting problem: an
    evenly-spaced sweep is not an even average, so a mask's full-trajectory map is
    dominated by whichever bands peak here."""
    x = np.arange(len(labels))
    series = [("M1", m1_bands, "#4c72b0"), ("M2", m2_bands, "#c44e52")]

    ax = axes[0]
    for name, maps, colour in series:
        mag = np.array([np.abs(m).mean() for m in maps])
        ax.plot(x, mag / (mag.max() or 1.0), marker="o", ms=3.5, lw=1.6, color=colour,
                label=name)
    ax.set_ylim(-0.03, 1.06)
    ax.set_ylabel("band mean |value|\n(÷ that mask's max)", fontsize=7)
    ax.set_title("Where each mask's magnitude sits along the inversion", fontsize=8.5)

    ax = axes[1]
    flat_src = np.asarray(src_act).ravel()
    for name, maps, colour in series:
        r = [0.0 if np.std(m) < 1e-12 or np.std(flat_src) < 1e-12
             else float(np.corrcoef(np.asarray(m).ravel(), flat_src)[0, 1]) for m in maps]
        ax.plot(x, r, marker="o", ms=3.5, lw=1.6, color=colour, label=name)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("corr(band map,\nsource |Δx0|)", fontsize=7)
    ax.set_title("Coupling to the source clip's own motion  (→1 = source-dynamics "
                 "detector)", fontsize=8.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=6, rotation=30,
                           ha="right")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)


def _psi_header(psi_readout, edit_space, signed):
    """(title, subtitle) for the M2 column — the ψ the panels actually show, spelled out.

    The formula is in the figure rather than only the docstring because ψ's definition is
    what makes the panels readable and it is NOT the LEDITS++ one a reader will assume:
    `energy` is a signed difference of per-group MOTION ENERGY, not |Δ prediction|, and it
    is read in the checkpoint's own space (x̂0 or ε), which changes what the symbol means.
    Both are resolved from the run, never hard-coded, so the caption cannot drift from the
    measurement.
    """
    sym = r"\hat{x}_0" if edit_space == "x0" else r"\hat{\epsilon}"
    if psi_readout == "abs":
        formula = (rf"$\psi = \langle\,|{sym}^{{\,c}} - {sym}^{{\,\varnothing}}|\,"
                   r"\rangle_{ch}$   ·   group-mean over channels")
    else:
        formula = (rf"$\psi = E({sym}|c) - E({sym}|\varnothing)$   ·   "
                   r"$E$ = group-mean $|x_f - x_{f-1}|$")
    lines = [formula]
    if signed:
        lines.append("red  ψ > 0: edit ADDS motion    ·    "
                     "blue  ψ < 0: edit STILLS the source")
    return f"M2  raw noise ψ  ({psi_readout},  ∅ = null embedding)", "\n".join(lines)


def _column_header(ax, title, subtitle):
    """Column title with a smaller sub-caption under it. Two `text` calls rather than one
    multi-line title because the formula must be legible at a smaller size than the
    heading, and `set_title` takes a single fontsize."""
    n = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, fontsize=8.5, pad=6 + 9.5 * n)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)


def plot_mask_noise_bands(clip_id, caption, instruction, targets, band_labels,
                          m1_bands, m2_bands, src_act, glabels, out_path,
                          psi_readout="energy", scale="band", summary=True,
                          psi_signed=True, edit_space="x0"):
    """How the raw M1 and M2 maps evolve across the inversion, for ONE instruction.

    The noise-level twin of `plot_mask_problem`: the same source-motion reference on
    top, then one ROW PER NOISE BAND with M1 on the left and M2 on the right, so a
    column reads top-to-bottom as "what this mask looks like at successive stages of the
    inversion". It exists because a mask is accumulated over the WHOLE trajectory and
    then reported as one map — this is the only figure that shows what that average is
    made of, and the project's two live claims about it (M1 sharpens toward high t, ψ_ε
    concentrates at low t) are statements about these rows.

    `scale`:
      "band"   (default) every panel on its own 99th-percentile scale. The right default
               and not a cosmetic one: ψ's magnitude spans orders of magnitude across
               the trajectory, so one shared scale renders every low-noise row as a
               black rectangle and the figure's whole question — does the mask's SHAPE
               change with noise level? — becomes unanswerable. The magnitude that gets
               normalised away is printed per panel and plotted in the summary row.
      "shared" one scale per column, `plot_mask_problem`-style. Answers the other
               question ("which bands dominate the sum?") directly at the cost of the
               first.

    `psi_signed` (default) draws ψ on a diverging scale rather than
    `plot_mask_problem`'s 0-anchored magma; pass False to match that figure instead. See
    `_band_panel` for why the default differs between the two.
    """
    n = len(band_labels)
    tgt_idx = [glabels.index(g) for g in targets if g in glabels]
    signed = psi_signed and psi_readout != "abs"        # "abs" has no sign to draw
    # A diverging map that does not say which way is which is unreadable, so the sign
    # legend rides along with the formula under the column heading.
    psi_title, psi_sub = _psi_header(psi_readout, edit_space, signed)
    cols = [("M1  raw cross-attention", "", m1_bands, "magma", False),
            (psi_title, psi_sub, m2_bands, "RdBu_r" if signed else "magma", signed)]
    shared = {0: shared_vmax(m1_bands),
              1: (_panel_vmax(np.concatenate([m.ravel() for m in m2_bands]), True)
                  if signed else shared_vmax(m2_bands))}

    # Row 1 is an empty SPACER. The M2 column header carries the ψ formula and runs three
    # lines, which would otherwise overlap the reference row's panel; `hspace` alone can
    # only clear it by loosening every other row gap to match.
    n_extra = 2 + (1 if summary else 0)
    H = 3.0 + 1.1 * n + (2.0 if summary else 0.0)
    fig = plt.figure(figsize=(11, H))
    gs = GridSpec(n + n_extra, 2, figure=fig, hspace=0.55, wspace=0.2,
                  height_ratios=[0.9, 0.30] + [1.0] * n + ([1.5] if summary else []),
                  top=1 - 1.05 / H, bottom=0.05)

    ax_ref = fig.add_subplot(gs[0, :])
    fg_heatmap(ax_ref, src_act, glabels, "cividis", shared_vmax([src_act]), tgt_idx)
    ax_ref.set_title("SOURCE motion  |Δx0|  (noise-INDEPENDENT reference — the same in "
                     "every row below)", fontsize=8.5, loc="left")

    for i in range(n):
        for c, (title, subtitle, maps, cmap, is_signed) in enumerate(cols):
            ax = fig.add_subplot(gs[i + 2, c])
            vmax = shared[c] if scale == "shared" else _panel_vmax(maps[i], is_signed)
            _band_panel(ax, maps[i], glabels, cmap, vmax, tgt_idx, is_signed)
            _magnitude_tag(ax, maps[i])
            if i == 0:
                _column_header(ax, title, subtitle)
            if c == 0:
                ax.set_ylabel(band_labels[i], fontsize=7.5, rotation=0, ha="right",
                              va="center", labelpad=30)
            if i == n - 1 and not summary:
                ax.set_xlabel("frame", fontsize=7)

    if summary:
        _band_summary([fig.add_subplot(gs[n + 2, 0]), fig.add_subplot(gs[n + 2, 1])],
                      m1_bands, m2_bands, src_act, band_labels)

    cap = ellipsis(caption, 80)
    fig.suptitle(
        "How the implicit masks change across the inversion  ·  "
        f'edit: "{ellipsis(instruction, 46)}"'
        + (f"  (expect: {', '.join(targets)})" if targets else "")
        + "\n" + (f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}")
        + "\n(rows = noise bands, low noise at the top; colour scale is "
        + ("per panel — compare SHAPES, not brightness; the mean magnitude each panel "
           "drops is printed in its corner" if scale == "band"
           else "shared per column — compare BRIGHTNESS, i.e. which bands dominate the sum")
        + ("; red = the group the edit SHOULD move)" if tgt_idx else ")"),
        fontsize=10, y=0.995)
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
