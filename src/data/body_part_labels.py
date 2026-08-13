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

# ── tier 3: verbs ────────────────────────────────────────────────────────────────
# WHY THIS EXISTS. Measured on the 87,372 HumanML3D annotations: 35,738 captions
# contain a locomotion verb and **86.5 % of them never name a leg** ("a person walks
# forward"), against 21.2 % for arm verbs. Leg semantics live in verbs in this corpus,
# nouns-only supervision misses them, and the consequence was measurable — the M1 mask
# scored 0.476 on arm instructions and 0.420 on leg ones, because the unsupervised verb
# column behaved like the ungrounded attention it came from (docs/FINDINGS.md "The
# arm/leg asymmetry is COLUMN dilution"). This is also the open direction TokenCompose
# names for itself: their supervision is nouns only.
#
# HOW THE ASSIGNMENTS WERE MADE, and what was deliberately left out. Each verb below is
# assigned by motion semantics and then audited against a base-rate-corrected lift —
# P(body part named | verb) / P(body part named) over the 279 verbs with >= 20
# noun-bearing captions. The correction matters: arm nouns are 3.6x more common, so raw
# co-occurrence makes *every* verb look arm-leaning ("walk" co-occurs with an arm noun
# 74 % of the time). Lift finds the real signal — kick 3.52, stomp 3.54, limp 3.33,
# hop 2.83 on the leg axis; pour, stir, clap, flap at the arm ceiling.
#
# Three exclusion rules, applied deliberately:
#   1. **Nothing that moves all four limbs** — crawl, swim, climb, dance, cartwheel,
#      tumble, wrestle, exercise. A mask naming every limb is the mask being off.
#   2. **Nothing ambiguous between limbs.** "bend" is the 15th most common verb and is
#      knees-or-waist; "stretch", "extend", "cross", "open", "spread", "roll" and
#      "shake" (hands or head) are the same problem. A wrong GROUP is worse than a
#      missing label, which is the same principle that keeps "back" out of LIMB2BASE.
#   3. **Nothing whole-body or generic** — stand, sit, fall, rise, get, move, start,
#      stop, do, perform, act, pretend, practise.
# Verbs are always TIER 2 (unlateralised): a side word next to a verb is usually a
# direction ("steps left"), and the noun path already handles a genuinely lateralised
# limb ("kicks with his left foot").
_LEGS = ("left_leg", "right_leg")
_ARMS = ("left_arm", "right_arm")
# LOCOMOTION = a gait: repeated stepping that carries the body across the ground, so the
# person ends up somewhere else. That is exactly what the ROOT channels encode (global
# translation velocity), which is why these verbs get root as a third group and in-place
# leg actions do not. The operational test is "does the body travel?", not "do the legs
# move?" — squatting and kicking move the legs without going anywhere; walking,
# shuffling and sidestepping go somewhere. Single ballistic events (jump/hop/leap) are
# deliberately NOT locomotion: they are usually vertical and in place, and their measured
# energy profile is the weakest case in the whole map (arms 0.329 vs legs 0.288, n=12).
#
# Why root belongs here at all, measured on 98 val clips whose captions name no body
# part: a locomotion clip spends legs 0.429 / arms 0.260 / ROOT 0.153 of its motion
# energy. Legs are the largest single locus, which is what justifies the label — but the
# root term is the one an edit like "walk faster", "walk backwards" or "take bigger
# steps" actually has to change, and without it the mask cannot touch the trajectory.
_LEGS_ROOT = ("left_leg", "right_leg", "root")

VERB2GROUPS: dict[str, tuple[str, ...]] = {
    # ── locomotion: legs + root, because the body travels (3 groups) ───────────
    "walk": _LEGS_ROOT, "run": _LEGS_ROOT, "jog": _LEGS_ROOT, "sprint": _LEGS_ROOT,
    "march": _LEGS_ROOT, "pace": _LEGS_ROOT, "stride": _LEGS_ROOT,
    "saunter": _LEGS_ROOT, "stroll": _LEGS_ROOT, "strut": _LEGS_ROOT,
    "creep": _LEGS_ROOT, "sneak": _LEGS_ROOT, "tiptoe": _LEGS_ROOT,
    "shuffle": _LEGS_ROOT, "sidestep": _LEGS_ROOT, "strafe": _LEGS_ROOT,
    "trot": _LEGS_ROOT, "gallop": _LEGS_ROOT, "waddle": _LEGS_ROOT,
    "limp": _LEGS_ROOT, "stagger": _LEGS_ROOT, "stumble": _LEGS_ROOT,
    "skip": _LEGS_ROOT, "slide": _LEGS_ROOT, "dodge": _LEGS_ROOT,
    # ── in-place leg actions: legs only, the body stays put (2 groups) ─────────
    "step": _LEGS, "stomp": _LEGS, "kick": _LEGS, "squat": _LEGS, "crouch": _LEGS,
    "kneel": _LEGS, "lunge": _LEGS, "trip": _LEGS, "wobble": _LEGS, "pivot": _LEGS,
    "land": _LEGS, "balance": _LEGS, "hurdle": _LEGS,
    # ── ballistic, usually vertical and in place — NOT locomotion (see above) ──
    "jump": _LEGS, "hop": _LEGS, "leap": _LEGS, "bounce": _LEGS,
    # ── arms: reaching, manipulation, striking (2 groups) ──────────────────────
    "wave": _ARMS, "punch": _ARMS, "clap": _ARMS, "throw": _ARMS, "toss": _ARMS,
    "catch": _ARMS, "grab": _ARMS, "grasp": _ARMS, "grip": _ARMS, "reach": _ARMS,
    "push": _ARMS, "pull": _ARMS, "lift": _ARMS, "raise": _ARMS, "lower": _ARMS,
    "drop": _ARMS, "hold": _ARMS, "put": _ARMS, "place": _ARMS, "pick": _ARMS,
    "point": _ARMS, "gesture": _ARMS, "scratch": _ARMS, "rub": _ARMS, "wipe": _ARMS,
    "wash": _ARMS, "clean": _ARMS, "scrub": _ARMS, "stir": _ARMS, "mix": _ARMS,
    "pour": _ARMS, "drink": _ARMS, "eat": _ARMS, "write": _ARMS, "type": _ARMS,
    "dial": _ARMS, "salute": _ARMS, "greet": _ARMS, "knock": _ARMS, "slap": _ARMS,
    "swat": _ARMS, "jab": _ARMS, "chop": _ARMS, "strike": _ARMS, "hit": _ARMS,
    "box": _ARMS, "flap": _ARMS, "shrug": _ARMS, "clasp": _ARMS, "fold": _ARMS,
    "hug": _ARMS, "cradle": _ARMS, "pat": _ARMS, "brush": _ARMS, "sweep": _ARMS,
    "dig": _ARMS, "strum": _ARMS, "press": _ARMS, "tap": _ARMS, "flex": _ARMS,
    # NB no "hand" — it is already a NOUN in LIMB2BASE, which claims the arm groups
    # first, so a verb entry could never fire. self_check asserts the two stay disjoint.
    "swing": _ARMS, "carry": _ARMS, "twirl": _ARMS, "pump": _ARMS,
    "thrust": _ARMS, "shove": _ARMS, "swipe": _ARMS, "cup": _ARMS, "pet": _ARMS,
    "stroke": _ARMS, "dribble": _ARMS, "shoot": _ARMS, "snatch": _ARMS, "wield": _ARMS,
    # ── head (1 group) ─────────────────────────────────────────────────────────
    "look": ("head",), "glance": ("head",), "stare": ("head",), "watch": ("head",),
    "nod": ("head",), "gaze": ("head",),
    # ── spine (1 group) ────────────────────────────────────────────────────────
    "lean": ("spine",), "hunch": ("spine",), "slouch": ("spine",), "stoop": ("spine",),
    "arch": ("spine",),
    # ── two DIFFERENT groups: bowing folds the spine and drops the head ────────
    "bow": ("spine", "head"),
    # ── root: whole-body reorientation, which is what the root channels encode ──
    "turn": ("root",), "spin": ("root",), "swivel": ("root",),
}

# Whole-body / ambiguous verbs, listed so the exclusion is auditable rather than an
# accident of what nobody thought of. Anything here must NOT be added without a reason.
VERBS_EXCLUDED = {
    # all four limbs (rule 1)
    "crawl", "swim", "climb", "dance", "cartwheel", "tumble", "wrestle", "exercise",
    "somersault", "fight",
    # ambiguous between limbs (rule 2)
    "bend", "stretch", "extend", "cross", "uncross", "open", "spread", "close", "roll",
    "shake", "twist", "rotate", "curl", "wiggle", "straighten", "tilt", "circle",
    # whole-body or generic (rule 3)
    "stand", "sit", "fall", "rise", "get", "move", "start", "stop", "do", "go", "come",
    "take", "make", "use", "keep", "begin", "continue", "return", "repeat", "perform",
    "act", "pretend", "practice", "imitate", "mimic", "play", "warm", "try", "appear",
    "seem", "look_like", "rest", "pause", "stay", "wait", "sway", "rock", "shift",
    "face", "lay", "lie", "back", "left", "side", "head", "arm", "shoulder", "elbow",
    "knee", "toe", "torso", "chest",
}

# Surface forms the rule generator below cannot produce: true irregulars and the
# dataset's own misspellings. Both were read off the corpus (every VERB-tagged token
# aligned with its caption word), not guessed — `self_check` fails if a form observed
# in HumanML3D for a mapped verb is not matched.
_IRREGULAR: dict[str, tuple[str, ...]] = {
    "run":   ("ran", "runing"),
    "throw": ("threw", "thrown", "throwed"),
    "step":  ("steping", "steped"),
    # "squat": the doubling rule sees a vowel at [-3] ("u") and declines to double, but
    # the "qu" acts as a consonant here — the corpus spells it "squatting".
    "squat": ("squating", "squated", "squatting", "squatted"),
    "clap":  ("claping", "claped"),
    "walk":  ("walkes",),
    "swing": ("swinge", "swung"),
    "hold":  ("held",),
    "catch": ("caught",),
    "stand": ("stood",),
    "put":   ("putting",),
    "hit":   ("hitting",),
    "shoot": ("shot",),
    "sweep": ("swept",),
    "strike": ("struck",),
    "dig":   ("dug",),
    "spin":  ("spun",),
    "slide": ("slid",),
    "grip":  ("gripe",),
    "write": ("wrote", "written"),
}


def _inflect(lemma: str) -> set[str]:
    """A verb lemma → the surface forms a caption may spell it with.

    Deliberately over-generates (both "hopped" and "hoped"-style variants are cheap;
    a false form simply never occurs in a caption) and is checked for UNDER-generation
    by `self_check`, which is the direction that would silently drop supervision.
    """
    f = {lemma, lemma + "s", lemma + "ing", lemma + "ed", lemma + "es"}
    if lemma.endswith("e"):
        f |= {lemma[:-1] + "ing", lemma + "d"}
    if lemma.endswith("y"):
        f |= {lemma[:-1] + "ies", lemma[:-1] + "ied"}
    if len(lemma) > 2 and lemma[-1] not in "aeiouwxy" and lemma[-2] in "aeiou" \
            and lemma[-3] not in "aeiou":
        f |= {lemma + lemma[-1] + "ing", lemma + lemma[-1] + "ed"}   # hop -> hopping
    return f


VERB_FORMS: dict[str, tuple[str, ...]] = {}
for _lem, _grp in VERB2GROUPS.items():
    for _form in _inflect(_lem) | set(_IRREGULAR.get(_lem, ())):
        VERB_FORMS.setdefault(_form, _grp)

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


def parse_caption(text: str, include_verbs: bool = True) -> list[Mention]:
    """Caption → body-part mentions. Pure text in, pure data out — no tokenizer, no
    model, no dataset; this is the unit-testable core of the label set.

    `include_verbs` adds the tier-3 verb mentions (VERB2GROUPS). Two rules govern them,
    both of which matter more than the vocabulary:

    **A noun always wins.** A verb mention is dropped when the caption already names a
    part in the same group set, so "kicks with his left leg" stays a single tier-1
    `left_leg` item instead of also emitting a tier-2 `{left_leg, right_leg}` item that
    pulls mass onto the wrong side. Without this the two labels would fight, and the
    laterality axis is exactly the one the mirror margin exists to protect.

    **Verbs are never lateralised.** A side word beside a verb is usually a direction
    ("steps left", "turns right"), so verbs are always tier 2 and never carry the mirror
    term. Genuine lateralised limb motion reaches the label set through its noun.

    Set False to reproduce the nouns-only label set (the A/B control).
    """
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

    if include_verbs:
        # A noun already naming one of a verb's groups outranks it — see the docstring.
        claimed = {g for m in mentions for g in m.groups}
        for word, start, end in toks:
            groups = VERB_FORMS.get(word)
            if groups is None or any(g in claimed for g in groups):
                continue
            mentions.append(Mention(((start, end),), groups, False))

    return mentions


def to_items(text: str, spans, group_mode: str = "parts",
             include_verbs: bool = True) -> list[dict]:
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
    for m in parse_caption(text, include_verbs):
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


def route_groups(text: str, group_mode: str = "parts",
                 include_verbs: bool = True) -> list[str]:
    """Instruction → the body-part groups it names, in order of first mention; [] if none.

    This is Option 13's "cheap tier" (docs/MaskOptions.md) with no LLM and no prototype
    embeddings: the parser that labels captions for the grounding loss, pointed at an
    edit instruction instead. It is laterality-correct by construction and rejects the
    two traps a keyword matcher falls into ("turns right", "walks back") because
    `parse_caption` already had to.

    Measured coverage on the 1013 MotionFix test instructions: **58.9 %** resolve to a
    group, of which 28 % name a side. Of the 41 % that do not resolve, three quarters
    name NO body part at all ("do it faster", "make a wider turn") — those are manner and
    trajectory edits with no correct group mask, so an empty list here is the honest
    answer and the caller should fall back to a temporal or unmasked edit rather than
    guess. An LLM would add roughly 11 points (the verb-implied cases), not the rest.

    Returns names in the model's own axis (`named_token_indices` handles the parts →
    joints expansion downstream); the CLI turns them into a (G,) mask via
    `utils.cli.parse_group_mask`.
    """
    seen: list[str] = []
    for m in parse_caption(text, include_verbs):
        for g in m.groups:
            if g not in seen:
                seen.append(g)
    return seen


def build_cache(data_root: str, encoder, splits=("train", "val", "test"),
                group_mode: str = "parts", out_path: str | None = None,
                include_verbs: bool = True) -> dict:
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

    cache = {text: to_items(text, encoder.token_spans(text), group_mode, include_verbs)
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
    print(f"parser self-check: {len(_CASES)} cases OK, "
          f"{len(VERB2GROUPS)} verbs / {len(VERB_FORMS)} surface forms")


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
                words = _WORD_RE.findall(parts[0].lower())
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


if __name__ == "__main__":                                  # quick manual check
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    self_check()
