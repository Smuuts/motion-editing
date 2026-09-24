"""
Raw LLM mentions -> the `{caption: [items]}` label cache the grounding loss trains on.

`llm_items.py` gets a caption's mentions out of the model. This module is everything
between that and `ground_labels_llm.json`, and it is deliberately the larger half: the
measured risk of an LLM label set is not that it says too little, it is that it asserts a
SIDE the caption does not license (FINDINGS.md 2026-09-22 — 20.7 % of the direction-word
population, and 14 % of mirror-twin pairs get inconsistent sides). Tier-1 items are what
drives the mirror margin, i.e. the term behind the project's one laterality result, so a
wrong side is the most expensive error available here. Everything below is arranged so
that an unverifiable side decays to a tier-2 (bilateral) label rather than a wrong tier-1
one — the same "a tier-2 label is harmless, a wrong side is not" rule `parser.py` already
follows.

FOUR FILTERS, IN ORDER, each catching something the next cannot:

1. **Span resolution.** A `phrase` that is not a substring of the caption buys no
   columns, so the item is dropped (measured 3.2 %). A phrase longer than
   `max_phrase_words` is dropped too — the model occasionally returns half a clause
   ("swings his arms out and in repeatedly"), which would supervise function words the
   regex parser never would and quietly change what the loss means.

2. **Side-evidence veto.** `side="right"` is accepted only when the model quoted a
   substring of the caption that `parse_caption` — the audited adjacency binder — reads
   as a side bound to a limb noun. "waves with his right hand" passes on *"right hand"*;
   "takes a step to the right" has nothing to quote and decays to tier 2. This is the
   filter that lets a verb be lateralised at all, which is the only thing this label set
   does that the regex structurally cannot.

3. **Mirror gate.** Half the train split is mirror clips whose captions are left/right
   swapped, so `label(mirror(c)) == mirror(label(c))` is checkable with no ground truth.
   A pair that fails has its tier-1 items demoted on BOTH sides. Note this gate is blind
   to filter 2's failure mode — a direction misread mirrors correctly — and filter 2 is
   blind to this one's, which is why both exist.

4. **Merge with the regex label set.** The output is `regex ∪ verified LLM`, never a
   replacement: the LLM drops ~30 % of the regex's tier-1 items and disagrees with ~10 %
   of the rest, so using it alone would spend the laterality result to buy +4.4 pp of
   coverage. The one case where the LLM overrides is the one it was brought in for — an
   unlateralised regex item on exactly the same columns, upgraded to a verified tier-1.
"""

import collections
import os

from analysis.instructions import MIRROR
from utils.logger import get_logger

from model.body_groups import group_names, named_token_indices

from .parser import parse_caption, to_items

log = get_logger(__name__)

DEFAULT_MAX_PHRASE_WORDS = 5
# `self_check` asserts no verb in the regex vocabulary maps to more than three groups, and
# the measured regex label set tops out at |S| = 3. The LLM does not know that rule and
# re-introduces exactly what FINDINGS.md's tier-3 design rule 3 excludes on purpose —
# "nothing that moves all four limbs: crawl, swim, climb, dance, cartwheel, tumble,
# wrestle". Those land as |S| = 4..6 targets, i.e. 4-6 of 7 groups, where (1 - m_S)^2 is
# nearly free to satisfy and what it teaches is "spread attention over the whole body" —
# the opposite of the selectivity the mask is read for. 1.7 % of items, dropped on the
# project's own standing rule that a missing label beats a useless one.
DEFAULT_MAX_TARGET_GROUPS = 3


def _bilateral(names):
    """Group names with every one-sided limb replaced by its left/right pair.

    The decay target for any side this module cannot verify: it keeps the category claim
    ("an arm moves") and drops only the claim the evidence does not support.
    """
    out = []
    for g in names:
        for n in (g, MIRROR.get(g)):
            if n and n not in out:
                out.append(n)
    return out


def _occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """Every (start, end) of `phrase` in `text`; exact match first, case-insensitive
    fallback. Case folding is a fallback rather than the default because a caption may
    legitimately contain both "Left" and "left"."""
    for hay, needle in ((text, phrase), (text.lower(), phrase.lower())):
        spans, i = [], hay.find(needle)
        while i != -1:
            spans.append((i, i + len(needle)))
            i = hay.find(needle, i + 1)
        if spans:
            return spans
    return []


def _side_verified(text: str, side: str, evidence: str, groups) -> bool:
    """Does `evidence` independently license `side` for `groups`? (Filter 2.)

    Three things must hold, and the middle one is the one that matters: the quote has to
    be in the caption (the model does not get to invent its own evidence), and
    `parse_caption` — not this module, and not the LLM — has to be the thing that reads a
    side out of it. That keeps the laterality decision on the audited binder even though
    the ATTACHMENT (which verb the limb belongs to) comes from the LLM.
    """
    if side not in ("left", "right") or not evidence.strip():
        return False
    if not _occurrences(text, evidence):
        return False
    bound = {g for m in parse_caption(evidence, include_verbs=False) if m.lat
             for g in m.groups}
    if not bound or any(not g.startswith(f"{side}_") for g in bound):
        return False
    # The mention must point at the same limb the item claims, so evidence naming a hand
    # cannot license a side on a leg group.
    return any(g in bound for g in groups)


def resolve_items(text: str, spans, raw_items, group_mode: str = "parts",
                  max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
                  max_target_groups: int = DEFAULT_MAX_TARGET_GROUPS,
                  stats=None) -> list[dict]:
    """Raw LLM mentions -> cache items with text-token columns resolved (filters 1-2).

    `spans` is the encoder's `token_spans(text)`, exactly as `to_items` takes it, so the
    supervised columns are produced by the same offset-mapping path the regex labels use
    and the two are mergeable without a second convention.
    """
    bump = (lambda k, n=1: stats.__setitem__(k, stats.get(k, 0) + n)) if stats is not None \
        else (lambda k, n=1: None)

    by_phrase = collections.OrderedDict()
    for it in raw_items:
        by_phrase.setdefault(it["phrase"].strip(), []).append(it)

    items = []
    for phrase, emissions in by_phrase.items():
        if len(phrase.split()) > max_phrase_words:
            bump("dropped_phrase_too_long", len(emissions))
            continue
        occ = _occurrences(text, phrase)
        if not occ:
            bump("dropped_phrase_not_in_caption", len(emissions))
            continue

        # One emission for several occurrences means the model did not say WHICH one it
        # meant; supervise them all, but refuse to carry a side across an ambiguity.
        ambiguous = len(emissions) == 1 and len(occ) > 1
        if ambiguous:
            pairs = [(emissions[0], occ)]
            bump("ambiguous_occurrence", 1)
        else:
            if len(emissions) > len(occ):
                bump("dropped_surplus_emission", len(emissions) - len(occ))
            pairs = [(e, [o]) for e, o in zip(emissions, occ)]

        for raw, where in pairs:
            groups, side = list(raw["groups"]), raw["side"]
            lat = (not ambiguous) and _side_verified(text, side, raw["side_evidence"],
                                                     groups)
            if side in ("left", "right") and not lat:
                bump("side_vetoed", 1)
                groups = _bilateral(groups)
            elif side == "none":
                # A one-sided group with no side claimed is the model contradicting
                # itself; take the claim it made explicitly (none) over the one it
                # implied.
                widened = _bilateral(groups)
                if widened != groups:
                    bump("side_implied_without_claim", 1)
                    groups = widened
            elif lat:
                bump("side_verified", 1)

            # After any bilateral widening, so a legitimately two-sided limb pair is
            # measured at its real size rather than its pre-widening one.
            if len(groups) > max_target_groups:
                bump("dropped_target_too_broad", 1)
                continue

            cols = sorted({c for c, cs, ce in spans
                           if any(cs < e and ce > s for s, e in where)})
            if not cols:
                bump("dropped_outside_truncation", 1)
                continue
            items.append({
                "W": cols,
                "S": named_token_indices(groups, group_mode),
                "tier": 1 if lat else 2,
                "lat": bool(lat),
            })
    return items


def mirror_caption_pairs(data_root: str, splits=("train", "val", "test")) -> dict:
    """{caption: {mirrored caption, ...}} from the corpus's own M-prefixed clips.

    HumanML3D ships each clip twice, `X` and `MX`, with the mirrored copy's annotations
    left/right swapped line for line (verified on 000065: "raises his left hand … slides
    to his left … clockwise" vs "right … right … counterclockwise"). Pairing by line
    index therefore needs no string surgery and no extra LLM calls — both captions are
    already in the corpus and already labelled. A caption can appear in more than one
    clip, hence a set of twins rather than one.
    """
    from data.clips import read_captions, split_ids

    ids = set()
    for split in splits:
        if os.path.exists(os.path.join(data_root, f"{split}.txt")):
            ids.update(split_ids(data_root, split))

    pairs: dict[str, set] = collections.defaultdict(set)
    for cid in ids:
        if not cid.startswith("M"):
            continue
        base = cid[1:]
        if base not in ids:
            continue
        for a, b in zip(read_captions(data_root, base), read_captions(data_root, cid)):
            if a and b:
                pairs[a].add(b)
                pairs[b].add(a)
    return dict(pairs)


def tier1_sides(items, group_mode: str = "parts") -> frozenset:
    """The one-sided limb groups a caption's tier-1 items assert.

    Reads `S`, not a names field, so it applies unchanged to a regex cache — which is
    what makes the mirror gate a comparison between the two label sets rather than a
    property only one of them can report.
    """
    names = group_names(group_mode)
    return frozenset(names[s] for it in items if it["lat"] for s in it["S"]
                     if names[s] in MIRROR)


def apply_mirror_gate(cache: dict, pairs: dict, group_mode: str = "parts",
                      max_target_groups: int = DEFAULT_MAX_TARGET_GROUPS,
                      stats=None) -> dict:
    """Demote tier-1 items on any caption whose mirror twin disagrees (filter 3).

    Demotion is applied to BOTH captions of a failing pair, because the gate says the two
    labels are inconsistent, not which one is wrong. In-place on `cache`.
    """
    bump = (lambda k, n=1: stats.__setitem__(k, stats.get(k, 0) + n)) if stats is not None \
        else (lambda k, n=1: None)

    checked = consistent = 0
    demote: set[str] = set()
    for text, twins in pairs.items():
        if text not in cache:
            continue
        a = tier1_sides(cache[text], group_mode)
        for twin in twins:
            if twin not in cache:
                continue
            checked += 1
            b = tier1_sides(cache[twin], group_mode)
            if frozenset(MIRROR.get(g, g) for g in a) == b:
                consistent += 1
            else:
                demote.add(text)
                demote.add(twin)

    # Demotion WIDENS a target ({left_arm, right_leg} -> all four limbs), so it can push an
    # item past max_target_groups after resolve_items already checked it. Re-apply the cap
    # here rather than let the invariant leak: an item that only survives as 4-6 of 7
    # groups is the "spread attention over the whole body" label the cap exists to reject,
    # and the same "a missing label beats a useless one" rule decides it.
    names = group_names(group_mode)
    for text in demote:
        kept = []
        for it in cache[text]:
            if not it["lat"]:
                kept.append(it)
                continue
            groups = _bilateral([names[s] for s in it["S"]])
            if len(groups) > max_target_groups:
                bump("dropped_target_too_broad_after_demotion", 1)
                continue
            it["S"] = named_token_indices(groups, group_mode)
            it["tier"], it["lat"] = 2, False
            bump("demoted_by_mirror_gate", 1)
            kept.append(it)
        cache[text] = kept

    bump("mirror_pairs_checked", checked)
    bump("mirror_pairs_consistent", consistent)
    bump("captions_demoted", len(demote))
    return cache


def merge_with_regex(text: str, spans, llm_items, group_mode: str = "parts",
                     include_verbs: bool = True, stats=None) -> list[dict]:
    """`regex ∪ verified LLM` for one caption (filter 4).

    The regex label set is the base and wins every contested column, with one exception:
    an LLM tier-1 item covering EXACTLY the columns of a single regex tier-2 item
    replaces it. That is the verb-laterality upgrade this whole path exists for
    ("waves" supervised as right_arm instead of {left_arm, right_arm}), and requiring the
    column sets to match exactly keeps it from silently deleting supervision on columns
    the LLM item does not cover.
    """
    bump = (lambda k, n=1: stats.__setitem__(k, stats.get(k, 0) + n)) if stats is not None \
        else (lambda k, n=1: None)

    base = to_items(text, spans, group_mode, include_verbs)
    dropped = set()

    for L in llm_items:
        cols = set(L["W"])
        overlap = [i for i, R in enumerate(base) if i not in dropped and set(R["W"]) & cols]
        if not overlap:
            bump("llm_items_added", 1)
            bump("llm_items_added_tier1", int(L["tier"] == 1))
            base.append(L)
        elif (L["tier"] == 1 and len(overlap) == 1
              and base[overlap[0]]["tier"] == 2
              and set(base[overlap[0]]["W"]) == cols):
            dropped.add(overlap[0])
            bump("llm_items_upgraded", 1)
            base.append(L)
        else:
            bump("llm_items_dropped_conflict", 1)
            # Near-misses: an LLM tier-1 over regex tier-2 rejected only because the
            # column sets differ. A large count here means the exact-match rule is too
            # strict and is throwing away the upgrade this path exists for.
            if L["tier"] == 1 and all(base[i]["tier"] == 2 for i in overlap):
                bump("llm_upgrade_near_miss", 1)

    return [it for i, it in enumerate(base) if i not in dropped]


def build_cache_llm(data_root: str, encoder, raw_by_caption: dict, captions=None,
                    splits=("train", "val", "test"), group_mode: str = "parts",
                    out_path: str | None = None, include_verbs: bool = True,
                    merge_regex: bool = True, mirror_gate: bool = True,
                    max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
                    max_target_groups: int = DEFAULT_MAX_TARGET_GROUPS
                    ) -> tuple[dict, dict]:
    """The whole pipeline: raw LLM mentions -> the label cache, plus its audit counters.

    `raw_by_caption` is `llm_items.route_captions_items_llm`'s output, passed in rather
    than fetched here so that re-deriving the cache under different filter settings costs
    no LLM calls.

    `captions` is the FULL corpus caption list and defaults to the keys of
    `raw_by_caption`. Pass it: a caption the LLM never answered for (a timeout, a refusal)
    is absent from `raw_by_caption`, and keying the output off that dict alone would drop
    it from the label file entirely — so it would lose the REGEX items it would have had,
    silently, and the merge would stop being additive. With the full list such a caption
    still gets its regex labels and simply contributes nothing from the LLM, which keeps
    the guarantee the tier-1 gate depends on: this label set is a superset of the regex
    one, caption for caption.
    """
    import json

    stats: dict = {}
    cache: dict[str, list[dict]] = {}
    # Tokenised once and reused by the merge: token_spans is the expensive call here
    # (51k captions through the HF tokenizer), and both passes want the same answer.
    texts = list(captions) if captions is not None else list(raw_by_caption)
    spans_by_caption = {text: encoder.token_spans(text) for text in texts}
    for text in texts:
        raw = raw_by_caption.get(text)
        if raw is None:
            stats["captions_without_llm_answer"] = \
                stats.get("captions_without_llm_answer", 0) + 1
            raw = []
        cache[text] = resolve_items(text, spans_by_caption[text], raw, group_mode,
                                    max_phrase_words, max_target_groups, stats)
    stats["llm_items_resolved"] = sum(len(v) for v in cache.values())

    if mirror_gate:
        apply_mirror_gate(cache, mirror_caption_pairs(data_root, splits), group_mode,
                          max_target_groups, stats)

    if merge_regex:
        for text in cache:
            cache[text] = merge_with_regex(text, spans_by_caption[text], cache[text],
                                           group_mode, include_verbs, stats)

    if out_path:
        # Atomic, because the label file is now a training PRECONDITION rather than a
        # derived artefact a run can regenerate (see assemble._ground_cache): a build
        # interrupted mid-write would otherwise leave a truncated JSON that fails at
        # load time, hours into a queued overnight run.
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = f"{out_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, out_path)
    return cache, stats


def audit(cache: dict, group_mode: str = "parts") -> dict:
    """Coverage, tier and balance counters for a label cache — the numbers the
    pre-registered gates in MaskOptions.md §13.3 are read against. Shape-compatible with
    a regex cache, so the two can be printed side by side."""
    names = group_names(group_mode)
    n_items = tier1 = 0
    per_group = collections.Counter()
    chance_num = 0.0
    for items in cache.values():
        for it in items:
            n_items += 1
            tier1 += int(it["lat"])
            chance_num += len(it["S"]) / len(names)
            for s in it["S"]:
                per_group[names[s]] += 1
    covered = sum(1 for v in cache.values() if v)
    arm = per_group["left_arm"] + per_group["right_arm"]
    leg = per_group["left_leg"] + per_group["right_leg"]
    return {
        "captions": len(cache),
        "covered": covered,
        "coverage": covered / max(len(cache), 1),
        "items": n_items,
        "tier1_items": tier1,
        "tier2_items": n_items - tier1,
        "chance_m_S": chance_num / max(n_items, 1),
        "per_group": dict(per_group),
        "arm_over_leg": arm / max(leg, 1),
        "left_over_right": ((per_group["left_arm"] + per_group["left_leg"])
                            / max(per_group["right_arm"] + per_group["right_leg"], 1)),
    }
