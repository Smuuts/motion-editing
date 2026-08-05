"""
The standard contrasting instruction set, and how each instruction maps to the
body-part group it is *supposed* to move.

The set is fixed on purpose: two laterality pairs (same limb, opposite side) and four
category pairs (arm instruction vs leg instruction) across them, so every probe's
"instruction-invariance" number decomposes into the same two axes and stays comparable
across scripts and checkpoints.
"""

from model.body_groups import axis_labels
from utils.cli import parse_group_names

DEFAULT_INSTRUCTIONS = [
    "raise the left arm",
    "raise the right arm",
    "kick with the left leg",
    "kick with the right leg",
]
DEFAULT_TARGETS = [["left_arm"], ["right_arm"], ["left_leg"], ["right_leg"]]

MIRROR = {"left_arm": "right_arm", "right_arm": "left_arm",
          "left_leg": "right_leg", "right_leg": "left_leg"}
LAT_PAIRS = [(0, 1), (2, 3)]                    # same limb, opposite side
CAT_PAIRS = [(0, 2), (0, 3), (1, 2), (1, 3)]    # arm instruction vs leg instruction


# Keyword → group votes for the red "expected group" overlay. Laterality words gate
# left/right; limb words pick the limb. Best-effort and annotation-only: this never
# feeds a mask.
_LIMB_KEYWORDS = {
    "left_arm":  ["arm", "hand", "wrist", "elbow", "shoulder", "reach", "punch", "wave"],
    "right_arm": ["arm", "hand", "wrist", "elbow", "shoulder", "reach", "punch", "wave"],
    "left_leg":  ["leg", "knee", "foot", "feet", "ankle", "kick", "step", "kneel", "stomp"],
    "right_leg": ["leg", "knee", "foot", "feet", "ankle", "kick", "step", "kneel", "stomp"],
    "spine":     ["torso", "spine", "back", "bend", "lean", "twist", "hip", "waist", "chest"],
    "head":      ["head", "neck", "look", "nod", "gaze"],
}


def guess_target_groups(instruction: str) -> list[str]:
    """Best-effort group names the instruction is supposed to move, from keywords."""
    txt = instruction.lower()
    left, right = "left" in txt, "right" in txt
    hits = []
    for group, kws in _LIMB_KEYWORDS.items():
        if not any(k in txt for k in kws):
            continue
        if group.startswith("left_") and right and not left:
            continue
        if group.startswith("right_") and left and not right:
            continue
        hits.append(group)
    return hits


def resolve_targets(instructions, target_groups=None, group_mode="parts"):
    """One expected-group list per instruction, expanded to the model's token axis.

    `target_groups` is the CLI shape: None (guess from the text), one spec applied to
    every instruction, or one spec per instruction. Called from a script, so an
    unusable spec exits cleanly instead of raising (see utils.cli.parse_group_names).
    """
    if target_groups is None:
        # guess_target_groups only ever emits valid body-part names, so this can't raise.
        return [axis_labels(guess_target_groups(e), group_mode) for e in instructions]
    if len(target_groups) == 1:
        return [parse_group_names(target_groups[0], group_mode) for _ in instructions]
    if len(target_groups) == len(instructions):
        return [parse_group_names(s, group_mode) for s in target_groups]
    raise SystemExit(f"--target_groups must be 1 or {len(instructions)} values "
                     f"(got {len(target_groups)}).")
