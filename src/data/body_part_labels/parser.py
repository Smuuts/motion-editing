"""
Caption -> body-part mentions, and the two projections of them the callers need.

Pure text in, pure data out: no tokenizer, no model, no dataset. Laterality binds by
ADJACENCY rather than by a word window — 19,218 of the 19,612 lateral->limb
co-occurrences within three tokens are directly adjacent. The only other pattern worth
capturing is a coordinated limb list sharing one laterality ("left hand and arm"), so
the gap between the two words may contain only limb words and "and". Everything else
("...to the left, raises the arm") stays unlateralised rather than guessing a side: a
tier-2 label is harmless, a wrong side is not.
"""

from dataclasses import dataclass

from .vocabulary import (BRIDGE_WORDS, HEIGHT_WORDS, LATERAL_BOTH, LATERAL_LEFT,
                         LATERAL_RIGHT, LATERAL_WORDS, LATERALISABLE, LIMB2BASE,
                         VERB_FORMS, WORD_RE)

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
    return [(m.group(0).lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


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
        if w in LIMB2BASE or w in BRIDGE_WORDS:
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

    This is the "cheap tier" with no LLM and no prototype
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
