"""
Plotting for the whole project, split by what is being drawn:

  animation.py    skeleton animations (single, and generated-vs-source comparison)
  heatmaps.py     axis-level primitives shared by every analysis figure
  masks.py        the implicit-mask (M1/M2) figures
  attention.py    the self-attention probe figures
  diagnostics.py  backbone-verification curves

Import the common entry points straight from `utils.visualise`; reach into a submodule
for the primitives (`from utils.visualise.heatmaps import heatmap`).
"""

from .animation import (
    CHAIN_COLORS, KINEMATIC_CHAIN, save_animation, save_comparison_animation,
    show_animation,
)
from .diagnostics import plot_noise_level_sweep
from .heatmaps import mean_off_diagonal
from .masks import (
    plot_correction_sweep, plot_gen_diff, plot_gen_diff_summary, plot_mask_problem,
    plot_mask_quant, plot_source_correction, save_mask_heatmap,
)

__all__ = [
    "CHAIN_COLORS", "KINEMATIC_CHAIN",
    "save_animation", "save_comparison_animation", "show_animation",
    "plot_noise_level_sweep", "mean_off_diagonal",
    "plot_gen_diff", "plot_gen_diff_summary",
    "plot_mask_problem", "plot_mask_quant", "save_mask_heatmap",
    "plot_source_correction", "plot_correction_sweep",
]
