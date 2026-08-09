"""
Caption → body-part supervision labels for the cross-attention grounding loss
(Option 1, docs/AttentionGrounding_Options.md).

The grounding loss needs to know, for a caption, WHICH text tokens should attend to
WHICH body-part group token: a "left arm" span must route to the `left_arm` group. This
module produces exactly that, as a list of items per caption:

    {"W": [text-token columns], "S": [group indices], "tier": 1|2, "lat": bool}

Nothing here runs during training. `build_cache` writes one JSON keyed by caption string
in a single offline pass over the dataset; the Trainer loads it once and every step is a
dict lookup.

TIERS
-----
tier 1  the caption lateralises the limb ("his left arm")   → S = {left_arm}, mirror term ON
tier 2  the limb is named but not lateralised ("the arms")  → S = {left_arm, right_arm}
        (also "both arms"), so a wrong side is never forced; mirror term OFF

Captions naming no body part produce no items and simply get no grounding loss.

VOCABULARY DECISIONS (measured on the 69,896 train annotations, not guessed)
---------------------------------------------------------------------------
* "back" is EXCLUDED. It occurs in 8,592 annotations and is almost always directional
  ("walks back", "back and forth", "back to where they came from"), not the body part.
  Including it would have been the largest single source of label noise in the set.
* Laterality binds by ADJACENCY, not by a word window: 19,218 of the 19,612 lateral→limb
  co-occurrences within three tokens are directly adjacent. The only other pattern worth
  capturing is a coordinated limb list sharing one laterality ("left hand and arm"), so
  the gap between the two words may contain only limb words and "and". Everything else
  ("...to the left, raises the arm") stays unlateralised rather than guessing a side —
  a tier-2 label is harmless, a wrong side is not.
* "left"/"right" with no adjacent limb word ("turns right", "walks to the left") produce
  nothing, which is the correct reading: those are directions, not body parts.
* Height references ("to chest height", "at waist level") name a LOCATION, not the part
  that moves, and are skipped.
* "shoulder" maps to the ARM group, not the torso — BODY_PART_GROUPS puts L_Collar and
  L_Shoulder in `left_arm` (see model/body_groups.py).
"""

import json
import os
import re
from dataclasses import dataclass

# Laterality markers. "both" is a marker too: it says "not one side", which is exactly
# the tier-2 label, and it must beat a further-away "left"/"right" ("his left hand and
# both feet" → the feet are not left).
LATERAL_LEFT = "left"
LATERAL_RIGHT = "right"
LATERAL_BOTH = "both"
LATERAL_WORDS = {LATERAL_LEFT, LATERAL_RIGHT, LATERAL_BOTH}

# Body-part word → base group name before laterality is applied. "arm"/"leg" are the
# lateralisable bases; "head"/"spine" are single groups whatever the caption says.
LIMB2BASE: dict[str, str] = {
    # arm group — includes the collar/shoulder, per BODY_PART_GROUPS
    "arm": "arm", "arms": "arm",
    "hand": "arm", "hands": "arm",
    "wrist": "arm", "wrists": "arm",
    "elbow": "arm", "elbows": "arm",
    "forearm": "arm", "forearms": "arm",
    "shoulder": "arm", "shoulders": "arm",
    "finger": "arm", "fingers": "arm",
    "palm": "arm", "palms": "arm",
    "thumb": "arm", "thumbs": "arm",
    "fist": "arm", "fists": "arm",
    # leg group
    "leg": "leg", "legs": "leg",
    "foot": "leg", "feet": "leg",
    "knee": "leg", "knees": "leg",
    "ankle": "leg", "ankles": "leg",
    "thigh": "leg", "thighs": "leg",
    "shin": "leg", "shins": "leg",
    "heel": "leg", "heels": "leg",
    "toe": "leg", "toes": "leg",
    "calf": "leg", "calves": "leg",
    # head group
    "head": "head", "neck": "head", "face": "head", "chin": "head",
    # torso / spine group. NB: "back" is deliberately absent — see the module docstring.
    "torso": "spine", "chest": "spine", "waist": "spine", "stomach": "spine",
    "belly": "spine", "abdomen": "spine", "hip": "spine", "hips": "spine",
    "pelvis": "spine", "midsection": "spine",
}

# Bases that carry a side. head/spine are unlateralised by anatomy.
LATERALISABLE = {"arm", "leg"}

# A part word immediately followed by one of these names a height reference, not a
# moving part: "raises their arms to chest height", "at waist level".
HEIGHT_WORDS = {"height", "level", "high"}

# Tokens allowed between a laterality word and the limb it modifies. This is what turns
# "left hand and arm" into two left-arm mentions while leaving "left, raises the arm"
# alone; see the module docstring for the measurement behind it.
_BRIDGE_WORDS = {"and"}

_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Mention:
    """One body-part reference in a caption.

    spans  : character spans of the words that make it up — the laterality word AND the
             limb word, so "left" itself learns to point at left_*.
    groups : target group names, e.g. ["left_arm"] or ["left_arm", "right_arm"].
    lat    : True when the caption named a side (tier 1), False when it did not (tier 2).
    """
    spans: tuple[tuple[int, int], ...]
    groups: tuple[str, ...]
    lat: bool

    @property
    def tier(self) -> int:
        return 1 if self.lat else 2


def _tokenise(text: str) -> list[tuple[str, int, int]]:
    """(lowercased word, char_start, char_end) for every alphabetic word."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _find_laterality(words: list[str], i: int) -> tuple[str | None, int | None]:
    """Scan back from the limb word at `i` for the laterality word modifying it.

    Returns (marker, index) or (None, None). Walks left across limb words and "and"
    only, so a coordinated list shares one side ("left hand and arm") while an
    unrelated earlier "left" is never picked up.
    """
    j = i - 1
    while j >= 0:
        w = words[j]
        if w in LATERAL_WORDS:
            return w, j
        if w in LIMB2BASE or w in _BRIDGE_WORDS:
            j -= 1
            continue
        return None, None
    return None, None


def parse_caption(text: str) -> list[Mention]:
    """Caption → body-part mentions. Pure text in, pure data out — no tokenizer, no
    model, no dataset; this is the unit-testable core of the label set."""
    toks = _tokenise(text)
    words = [w for w, _, _ in toks]
    mentions: list[Mention] = []

    for i, (word, start, end) in enumerate(toks):
        base = LIMB2BASE.get(word)
        if base is None:
            continue
        # "to chest height" names a location, not the part that moves.
        if i + 1 < len(words) and words[i + 1] in HEIGHT_WORDS:
            continue

        spans = [(start, end)]
        if base in LATERALISABLE:
            marker, j = _find_laterality(words, i)
            if marker in (LATERAL_LEFT, LATERAL_RIGHT):
                spans.append((toks[j][1], toks[j][2]))
                mentions.append(Mention(tuple(spans), (f"{marker}_{base}",), True))
                continue
            if marker == LATERAL_BOTH:
                spans.append((toks[j][1], toks[j][2]))
            groups = (f"left_{base}", f"right_{base}")
        else:
            groups = (base,)
        mentions.append(Mention(tuple(spans), groups, False))

    return mentions


def to_items(text: str, spans, group_mode: str = "parts") -> list[dict]:
    """Mentions → cache items with text-token COLUMNS resolved.

    `spans` is the encoder's `token_spans(text)` → [(column, char_start, char_end)].
    A token belongs to a mention when it overlaps one of the mention's character spans,
    which makes sub-word pieces fall out for free (T5 splits "shoulder" into two pieces;
    both overlap the word and both get supervised).
    """
    # Imported here, not at module scope, so `parse_caption` and `self_check` stay free
    # of project imports — the pure-text core runs (and is testable) on its own.
    from model.body_groups import named_token_indices

    items = []
    for m in parse_caption(text):
        cols = sorted({
            col for col, cs, ce in spans
            if any(cs < e and ce > s for s, e in m.spans)
        })
        if not cols:
            continue                       # word fell outside the encoder's truncation
        items.append({
            "W": cols,
            "S": named_token_indices(list(m.groups), group_mode),
            "tier": m.tier,
            "lat": m.lat,
        })
    return items


def build_cache(data_root: str, encoder, splits=("train", "val", "test"),
                group_mode: str = "parts", out_path: str | None = None) -> dict:
    """{caption: items} over every annotation in `splits`, written as JSON.

    One offline pass. The cache is keyed by caption STRING, not by (clip, annotation
    index), because HumanML3DDataset samples a random annotation per epoch and returns
    the text itself — so a string key needs no index bookkeeping and dedupes captions
    shared across clips for free.
    """
    from data.clips import read_captions, split_ids

    seen: set[str] = set()
    for split in splits:
        path = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(path):
            continue
        for clip_id in split_ids(data_root, split):
            seen.update(read_captions(data_root, clip_id))

    cache = {text: to_items(text, encoder.token_spans(text), group_mode)
             for text in sorted(seen)}

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cache, f)
    return cache


def load_cache(path: str) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


# Cases the parser must get right, each one a pattern measured to be frequent in the
# train captions. The negatives matter more than the positives here: "walks back" and
# "turns right" are the two ways a keyword parser silently poisons this label set.
_CASES: list[tuple[str, list[tuple[str, tuple[str, ...]]]]] = [
    ("a person raises his left arm above his head",
     [("left arm", ("left_arm",)), ("head", ("head",))]),
    ("the person walks back to where they came from", []),
    ("a person turns right and walks forward", []),
    ("a person paces from left to right and then back to their origin", []),
    ("a person uses their right hand and arm to throw",
     [("right hand", ("right_arm",)), ("right arm", ("right_arm",))]),
    ("the man raises his arms to chest height",
     [("arms", ("left_arm", "right_arm"))]),
    ("a man rolls his left shoulder and then his right shoulder",
     [("left shoulder", ("left_arm",)), ("right shoulder", ("right_arm",))]),
    ("the figure swings both arms", [("both arms", ("left_arm", "right_arm"))]),
    ("a person kicks with the left leg", [("left leg", ("left_leg",))]),
    ("a person walks in a circle", []),
]


def self_check() -> None:
    """Assert the parser on the cases above; raises on the first disagreement."""
    for text, expected in _CASES:
        got = [(" ".join(text[s:e] for s, e in sorted(m.spans)), m.groups)
               for m in parse_caption(text)]
        assert got == expected, (
            f"\n  caption  {text!r}\n  expected {expected}\n  got      {got}")
    print(f"parser self-check: {len(_CASES)} cases OK")


if __name__ == "__main__":                                  # quick manual check
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    self_check()
