"""
LEDITS++ Stage 2 — spatiotemporal implicit masking.

Builds, per edit instruction, a binary mask M = M1 ∩ M2 over (frame × body-part group):

  M1 (semantic)       averaged cross-attention over the instruction's content tokens,
                      layers and timesteps, thresholded. "Which group at which frame
                      does the edit text attend to?"  -> readouts.py
  M2 (noise-estimate) magnitude (or signed energy) of the guidance vector
                      psi = f(x_t, c) - f(x_t, ref), aggregated per group and
                      thresholded. "Where does the edit actually change the
                      prediction?"  -> collect.py

Both are accumulated over the stored inversion timesteps, so a mask is averaged over the
whole trajectory rather than read off a single noise level. That average is a raw
magnitude sum by default, so an evenly-spaced sweep is NOT an even average — and M1's and
M2's signals do not live at the same noise levels anyway. See `per_step_norm`,
`attn_timesteps` and `psi_timesteps` in `collect_statistics`.

Module map:
  groups.py    channel <-> group aggregation, motion energy, per-step normalisation
  readouts.py  M1 read-outs off a stored attention map, and the sink diagnostics
  collect.py   the inversion sweep that accumulates the raw M1/M2 statistics
  build.py     thresholds, group selection, and the mask modes
"""

from .build import (build_mask, mask_mode_components, percentile_threshold,
                    rank_group_select)
from .collect import PSI_READOUTS, build_sweep, collect_statistics
from .groups import (frame_energy, group_mask_to_channels, group_motion_energy,
                     normalise_step)
from .readouts import (ALL_READOUTS, STOP_WORDS, VALUE_READOUTS, WEIGHT_READOUTS,
                       semantic_token_subset)

__all__ = [
    "ALL_READOUTS", "PSI_READOUTS", "STOP_WORDS", "VALUE_READOUTS", "WEIGHT_READOUTS",
    "build_mask", "build_sweep", "collect_statistics", "frame_energy",
    "group_mask_to_channels", "group_motion_energy", "mask_mode_components",
    "normalise_step", "percentile_threshold", "rank_group_select",
    "semantic_token_subset",
]
