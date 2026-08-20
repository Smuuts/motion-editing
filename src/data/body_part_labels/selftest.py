"""
Parser assertions and corpus coverage checks.

`self_check` asserts the parser against patterns measured to be frequent in the train
captions; the NEGATIVE cases matter more than the positives, since "walks back" and
"turns right" are the two ways a keyword parser silently poisons this label set.
`check_verb_coverage` turns "we think we cover the verbs" into a number.
"""

import os

from utils.logger import get_logger

from .parser import parse_caption
from .vocabulary import LIMB2BASE, VERB2GROUPS, VERB_FORMS, VERBS_EXCLUDED, WORD_RE

log = get_logger(__name__)


# Cases the parser must get right, each one a pattern measured to be frequent in the
# train captions. The negatives matter more than the positives here: "walks back" and
# "turns right" are the two ways a keyword parser silently poisons this label set.
_CASES: list[tuple[str, list[tuple[str, tuple[str, ...]]]]] = [
    ("a person raises his left arm above his head",
     [("left arm", ("left_arm",)), ("head", ("head",))]),
    # "back"/"right"/"left" are still directions, never body parts — the point of these
    # three cases. What they now also show is the verb firing beside that negative.
    ("the person walks back to where they came from",
     [("walks", ("left_leg", "right_leg", "root"))]),
    # Two verbs, two actions, two items: the noun-wins rule is noun-vs-verb ONLY. Verb
    # items are all tier 2, so two of them cannot contradict each other on laterality
    # the way a tier-1 noun and a tier-2 verb can — and suppressing one by the other
    # would make the label set depend on word order.
    ("a person turns right and walks forward",
     [("turns", ("root",)), ("walks", ("left_leg", "right_leg", "root"))]),
    ("a person paces from left to right and then back to their origin",
     [("paces", ("left_leg", "right_leg", "root"))]),
    ("a person uses their right hand and arm to throw",
     [("right hand", ("right_arm",)), ("right arm", ("right_arm",))]),
    ("the man raises his arms to chest height",
     [("arms", ("left_arm", "right_arm"))]),
    ("a man rolls his left shoulder and then his right shoulder",
     [("left shoulder", ("left_arm",)), ("right shoulder", ("right_arm",))]),
    ("the figure swings both arms", [("both arms", ("left_arm", "right_arm"))]),
    ("a person kicks with the left leg", [("left leg", ("left_leg",))]),
    # ── tier 3 (verbs). The negatives are the load-bearing half here too. ──────
    ("a person walks in a circle", [("walks", ("left_leg", "right_leg", "root"))]),
    ("the person jumps up and down", [("jumps", ("left_leg", "right_leg"))]),
    # a noun outranks the verb naming the same groups: ONE tier-1 item, no tier-2 twin
    ("a person kicks with his left foot", [("left foot", ("left_leg",))]),
    # ...but a verb for a DIFFERENT part still fires
    ("a person walks while waving his left arm",
     [("left arm", ("left_arm",)), ("walks", ("left_leg", "right_leg", "root"))]),
    # two groups that are not a limb pair
    ("the person bows", [("bows", ("spine", "head"))]),
    # excluded: all four limbs, ambiguous between limbs, whole-body
    ("a person crawls forward", []),
    ("the person bends down", []),
    ("a person stands still", []),
    # "turns right" is still a direction, and turn routes to the root, not a limb
    ("a person turns right", [("turns", ("root",))]),
]


def self_check() -> None:
    """Assert the parser on the cases above; raises on the first disagreement."""
    for text, expected in _CASES:
        got = [(" ".join(text[s:e] for s, e in sorted(m.spans)), m.groups)
               for m in parse_caption(text)]
        assert got == expected, (
            f"\n  caption  {text!r}\n  expected {expected}\n  got      {got}")
    overlap = set(VERB2GROUPS) & VERBS_EXCLUDED
    assert not overlap, f"verbs both mapped and excluded: {sorted(overlap)}"
    clash = set(VERB2GROUPS) & set(LIMB2BASE)
    assert not clash, f"words used as both a noun and a verb label: {sorted(clash)}"
    log.info("parser self-check: %d cases OK, %d verbs / %d surface forms",
             len(_CASES), len(VERB2GROUPS), len(VERB_FORMS))


def check_verb_coverage(data_root: str, splits=("train", "val", "test")) -> dict:
    """Every VERB-tagged surface form in the corpus vs what `VERB_FORMS` matches.

    HumanML3D annotations carry `word/POS` tags with LEMMAS, and captions carry the
    inflected surface form, so the corpus itself says which spellings exist — including
    irregulars ("threw", "stood") and its own misspellings ("walkes", "squating").
    `_inflect` is a rule generator and rules under-generate; this is the check that turns
    "we think we cover the verbs" into a number, and it is the reason `_IRREGULAR` has
    the entries it has.

    Returns {"missed": {form: count}, ...} — a non-empty `missed` means supervision is
    being silently dropped for those spellings.
    """
    from data.clips import split_ids

    seen, missed, matched = {}, {}, 0
    for split in splits:
        if not os.path.exists(os.path.join(data_root, f"{split}.txt")):
            continue
        for clip_id in split_ids(data_root, split):
            path = os.path.join(data_root, "texts", f"{clip_id}.txt")
            if not os.path.exists(path):
                continue
            for line in open(path):
                parts = line.strip().split("#")
                if len(parts) < 2:
                    continue
                tagged = [t.rsplit("/", 1) for t in parts[1].split() if "/" in t]
                words = WORD_RE.findall(parts[0].lower())
                if len(words) != len(tagged):
                    continue                      # tokenisers disagree; skip the line
                for w, (lemma, pos) in zip(words, tagged):
                    if pos != "VERB" or lemma not in VERB2GROUPS:
                        continue
                    seen[lemma] = seen.get(lemma, 0) + 1
                    if w in VERB_FORMS:
                        matched += 1
                    else:
                        missed[w] = missed.get(w, 0) + 1
    total = sum(seen.values())
    return {"tokens_of_mapped_verbs": total, "matched": matched,
            "coverage": matched / total if total else 1.0,
            "missed": dict(sorted(missed.items(), key=lambda kv: -kv[1]))}

