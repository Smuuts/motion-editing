"""
Option 6 — the generation-space differential mask, and the statistics that judge it.

The idea (docs/AttentionGrounding_Options.md §"Option 6"): this backbone's *generation*
path is measured-good (FID 0.153, R@1 0.554 on the x0 checkpoint) while its *editing*
conditioner on inverted latents is measured-blind to the instruction. So instead of
reading the mask out of the weak pathway (attention M1 / noise-contrast M2), generate
twice from ONE shared noise path — once under a reference prompt, once under the edit
instruction — and localise the edit by differencing the two generations:

    D[f, g] = ‖ g_edit[f, g] − g_ref[f, g] ‖        over the 7 body-part groups

Two properties worth stating up front, because they shape everything here:

* **D never touches the model's token axis.** It is computed from generated *motion*,
  not from attention, so the body-part axis is the channel/joint partition — the same 7
  groups whatever the checkpoint tokenises with (`group_mode` is irrelevant here). That
  is exactly why this route sidesteps the grounding question.
* **D is only trustworthy as a *group selector*.** `g_edit`/`g_ref` are freshly
  generated and therefore not frame-aligned to any real source clip; the frame axis is
  kept because it is meaningful *within* a paired batch (shared noise ⇒ matched
  trajectories) and because the alignment number needs a cell-level mask, but it must
  not be carried over to a source clip. See the option's "Caveats" section.

WHAT COUNTS AS THE READOUT WORKING
---------------------------------
`readout_stats` reports, per readout family:

  forced choices (chance exactly 0.5, and a constant bias cannot win them, because the
  instruction set is symmetric under left↔right and arm↔leg):
    lat_acc   does D put more mass on the named side than on its mirror?
    cat_acc   does D put more mass on the named limb pair than on the other one?
  top1        does D's biggest group equal the instructed one (chance 1/G),
  align       share of the thresholded mask's cells in the instructed group (chance 1/G,
              cut by the editor's own `percentile_threshold` so it is comparable to the
              algM1/algM2 columns of the status board),
  r_lat/r_cat instruction-invariance on the two axes, comparable to every other probe's.

`lat_acc` is the number the whole option is gated on.
"""

import numpy as np
import torch

from editing.masking import percentile_threshold
from model.body_groups import BODY_PART_GROUPS, GROUP_NAMES, group_layout
from utils.probe import accuracy_block, group_profile
from .instructions import DEFAULT_INSTRUCTIONS, DEFAULT_TARGETS, MIRROR
from .mask_axes import alignment, axis_stats

# Joint indices (into the 22-joint SMPL array) owned by each of GROUP_NAMES. Body group
# member `b` is an index into the 21-joint body array, i.e. SMPL joint b+1; the root
# group is the pelvis.
GROUP_JOINTS: list[list[int]] = (
    [[0]] + [[b + 1 for b in joints] for _, joints in BODY_PART_GROUPS])


def part_channels(feature_mode: str) -> list[list[int]]:
    """Per-body-part channel index lists for a feature_mode (263-d or 135-d)."""
    return group_layout(feature_mode, "parts")[0]


# ── the divergence readouts ──────────────────────────────────────────────────────

def feature_divergence(a: np.ndarray, b: np.ndarray, channels) -> np.ndarray:
    """(F, G) mean |a − b| per body-part group, in NORMALISED feature space.

    Normalised (not raw) because the groups own wildly different channel counts and
    scales; the mean-over-channels then matches `utils.probe.source_activity`, which is
    what every other probe's reference row is computed with.
    """
    d = np.abs(np.asarray(a) - np.asarray(b))                    # (F, D)
    return np.stack([d[:, ch].mean(axis=-1) for ch in channels], axis=-1)


def joint_divergence(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """(F, G) mean per-joint displacement between two clips' joint positions.

    pa, pb: (F, 22, 3) world-space joints. Body joints are compared ROOT-RELATIVE, so a
    pure global translation difference — which the root token already carries, and which
    says nothing about which limb the instruction moved — cannot light up every group at
    once. The root group keeps the world displacement, since that IS its content.
    """
    rel = lambda p: p - p[:, :1]                                 # (F, 22, 3)
    d = np.linalg.norm(rel(pa) - rel(pb), axis=-1)               # (F, 22), joint 0 ≡ 0
    d[:, 0] = np.linalg.norm(pa[:, 0] - pb[:, 0], axis=-1)       # root: world motion
    return np.stack([d[:, js].mean(axis=-1) for js in GROUP_JOINTS], axis=-1)


def temporal_activity(divergence_fn, clip) -> np.ndarray:
    """The same divergence applied between consecutive frames of ONE clip → its own
    (F, G) motion energy, first frame REPEATED.

    This is the control that keeps Option 6 honest: if a generation's plain motion energy
    localises the instruction just as well as the paired difference does, then the
    differencing — and the shared noise it needs — buys nothing, and what is really being
    measured is "the generator moves the named limb".

    Frame 0 repeats frame 1 rather than being zeroed (2026-08-16), matching the other three
    copies of this convention — `masking._frame_energy`, `utils.probe.source_activity`,
    `grounding.batched_source_activity`. They are one definition and are changed together, so
    that "this map vs the source's own activity" stays a comparison of like with like
    (docs/ARCHITECTURE.md).
    """
    d = divergence_fn(clip[1:], clip[:-1])
    return np.concatenate([d[:1], d], axis=0)


# ── scoring ──────────────────────────────────────────────────────────────────────

def _binary(m: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold an (F, G) map exactly as the editor thresholds M1/M2 (all frames of a
    generated clip are valid, so there is no padding to exclude)."""
    t = torch.from_numpy(np.asarray(m)).float()
    valid = torch.ones(t.shape[0], dtype=torch.bool)
    return percentile_threshold(t, valid, percentile).numpy()


def forced_choices(profiles, glabels, targets=DEFAULT_TARGETS) -> dict:
    """Per-instruction side/limb choices, plus the top-1 group hit.

    Both choices are between exactly two options and the instruction set covers both
    answers equally often, so chance is 0.5 and a fixed preference ("always left",
    "always the arms") scores exactly chance — it cannot fake a pass.
    """
    arm = [glabels.index("left_arm"), glabels.index("right_arm")]
    leg = [glabels.index("left_leg"), glabels.index("right_leg")]
    lat, cat, top1 = [], [], []
    for p, tgt in zip(profiles, targets):
        named = tgt[0]
        own, other = (arm, leg) if named.endswith("_arm") else (leg, arm)
        lat.append(bool(p[glabels.index(named)] > p[glabels.index(MIRROR[named])]))
        cat.append(bool(p[own].sum() > p[other].sum()))
        top1.append(bool(int(np.argmax(p)) == glabels.index(named)))
    return {"lat_wins": lat, "cat_wins": cat, "top1_wins": top1}


def readout_stats(maps, glabels=GROUP_NAMES, instructions=DEFAULT_INSTRUCTIONS,
                  targets=DEFAULT_TARGETS, percentile: float = 70.0) -> dict:
    """Everything measured for ONE family of per-instruction (F, G) maps.

    `axis_stats`/`alignment` are the editor probes' own definitions, so `r_laterality`,
    `r_category`, the profiles and `align` are directly comparable to the M1/M2 numbers
    in the status board rather than merely similar to them.
    """
    st = axis_stats(maps, glabels, instructions, targets)
    st["align"] = alignment([_binary(m, percentile) for m in maps], glabels, targets)
    st.update(forced_choices([np.array(st["profile"][e]) for e in instructions],
                             glabels, targets))
    st["magnitude"] = [float(np.mean(m)) for m in maps]
    return st


def pool(per_seed: list[dict], key: str) -> list[bool]:
    """Flatten one forced-choice key across seeds → the trials `accuracy_block` scores."""
    return [w for s in per_seed for w in s[key]]


def verdict(readouts: dict) -> dict:
    """Pooled forced-choice blocks per readout family — the summary the gate is read off.

    Chance is 0.5 for the two forced choices; the top-1 group hit is 1/G.
    """
    return {
        name: {
            "laterality": accuracy_block(pool(seeds, "lat_wins"), f"{name}: named side"),
            "category": accuracy_block(pool(seeds, "cat_wins"), f"{name}: named limb"),
            "top1": accuracy_block(pool(seeds, "top1_wins"), f"{name}: top-1 group",
                                   chance=1.0 / len(GROUP_NAMES)),
        }
        for name, seeds in readouts.items()
    }
