"""
The grounding loss's resolved configuration, and the left/right mirror permutation.
"""

import random
from dataclasses import dataclass, field

import torch

from analysis.instructions import MIRROR
from model.body_groups import group_names

@dataclass
class GroundingConfig:
    """Everything train_one_epoch needs to run the grounding loss, resolved once.

    Bundled rather than passed as eight more keyword arguments because they are only
    ever used together, and because `enabled` then has one obvious home.
    """
    weight:         float = 0.0            # λ; 0 = the whole feature is off
    layers:         list[int] = field(default_factory=list)
    mirror:         float = 1.0            # λ_mirror  — tier 1, "beat your mirror"
    even:           float = 0.1            # λ_even    — tier 2, "do not pick a side"
    margin:         float = 0.1
    warmup_epochs:  int = 20
    window:         tuple[int, int] | None = None   # hard timestep gate, if used
    cache:          dict = field(default_factory=dict)   # caption -> [item, ...]
    group_channels: list = field(default_factory=list)   # for the src_corr monitor
    monitor:        bool = True
    mirror_mat:     torch.Tensor | None = None           # see mirror_matrix()

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 and bool(self.layers)

    def active(self, epoch: int) -> bool:
        """From-scratch attention is random noise at epoch 0 — supervising it just
        teaches the model to satisfy a loss on a signal that carries nothing yet.
        TokenCompose finetuned an already-converged model; we do not, hence the warmup
        (default 20 epochs, higher than the doc's 5)."""
        return self.enabled and epoch >= self.warmup_epochs

    def pick_layer(self) -> int:
        """One block per step. Materialising an explicit (B, h, F·G, L) softmax in every
        block at once does not fit; in expectation each candidate block still receives
        the same pressure. Same argument as the entropy regulariser's ent_layer."""
        return random.choice(self.layers)

    def val_layer(self) -> int:
        """Deterministic block for validation, so the val curve is one quantity across
        epochs instead of a random draw."""
        return self.layers[len(self.layers) // 2]


def _mirror_name(name: str) -> str:
    """Axis name → its left/right mirror, unchanged when it has no side.

    Handles both token axes: 'parts' names are seeded from analysis/instructions.MIRROR
    — the same map the laterality probes score against, so the training signal and the
    metric cannot drift apart by one of them redefining the pairing — and 'joints' names
    (L_Elbow / R_Elbow) fall to the prefix rule.
    """
    if name in MIRROR:
        return MIRROR[name]
    for a, b in (("left_", "right_"), ("right_", "left_"), ("L_", "R_"), ("R_", "L_")):
        if name.startswith(a):
            return b + name[len(a):]
    return name


def mirror_matrix(group_mode: str = "parts") -> torch.Tensor:
    """(G, G) permutation sending each group to its mirror, identity for the
    unlateralised ones. `S @ mirror_matrix` turns a target set into its mirror set.

    A group whose mirror is itself makes the margin term unsatisfiable by construction
    (m_mirror ≡ m_S ⇒ a constant relu(margin) with no gradient), which is exactly why
    the loss applies the margin to tier-1 items only — their S is always one-sided.
    """
    names = group_names(group_mode)
    idx = {n: i for i, n in enumerate(names)}
    M = torch.zeros(len(names), len(names))
    for i, n in enumerate(names):
        M[i, idx.get(_mirror_name(n), i)] = 1.0
    return M
