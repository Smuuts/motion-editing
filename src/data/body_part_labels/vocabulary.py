"""
The label vocabulary: which caption words name which body-part group.

VOCABULARY DECISIONS, measured on the 69,896 train annotations rather than guessed:

* "back" is EXCLUDED from the noun map. It occurs in 8,592 annotations and is almost
  always directional ("walks back", "back and forth"), not the body part — it would
  have been the largest single source of label noise in the set.
* "left"/"right" with no adjacent limb word ("turns right", "walks to the left") name
  a direction, not a body part, and produce nothing.
* Height references ("to chest height", "at waist level") name a LOCATION, not the part
  that moves, and are skipped.
* "shoulder" maps to the ARM group, not the torso — BODY_PART_GROUPS puts L_Collar and
  L_Shoulder in `left_arm` (see model/body_groups.py).
"""

import re


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
# column behaved like the ungrounded attention it came from (measured: "The
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
BRIDGE_WORDS = {"and"}

WORD_RE = re.compile(r"[A-Za-z]+")
