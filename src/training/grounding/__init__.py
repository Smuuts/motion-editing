"""
TokenCompose-style cross-attention grounding loss.

    TokenCompose, Wang et al., CVPR 2024 (arXiv 2312.03626) — the source of L_token

Nothing in the denoising objective ever asks the model to route the word "left" to the
`left_arm` token, so it does not: measured on an ungrounded checkpoint, the M1 masks for
"raise the left arm" and "raise the right arm" correlate at r = 0.985. This package adds
the missing objective.

TokenCompose's second term (L_pixel, a per-pixel BCE against a segmentation mask) has no
analogue here — the group tokens are opaque, with no sub-group resolution to supervise.
Their ablation says L_token carries the load anyway (29.86 -> 49.85 of the 52.15 total),
so this ports the term that matters and drops the one that cannot be expressed. The
MIRROR term is not theirs: it targets the one axis no training-free intervention in this
project has ever moved. L_group alone is satisfied by "put mass on AN arm", which a
source-motion detector can do without reading the word; the mirror margin makes the
left/right distinction explicit.

The labels come from `data/body_part_labels`, offline, from the captions themselves — no
segmenter and no LLM. `S` falls out of the tokeniser's own channel->group partition
(`model/body_groups.py`), which is why this is cheap in motion and expensive in images.

Module map:
  resolve.py  which blocks and which text columns the signal covers
  spec.py     GroundingConfig, and the left/right mirror permutation
  loss.py     L_token, the mirror margin, the evenness term and the monitors
"""

from .loss import batched_source_activity, collect_items, grounding_loss
from .resolve import (resolve_ground_layers, resolve_readout_columns,
                      resolve_readout_layers)
from .spec import GroundingConfig, mirror_matrix

__all__ = [
    "GroundingConfig", "batched_source_activity", "collect_items", "grounding_loss",
    "mirror_matrix", "resolve_ground_layers", "resolve_readout_columns",
    "resolve_readout_layers",
]
