"""
Caption -> per-MENTION body-part labels from an LLM, the training-side twin of
`llm_router.route_groups_llm`.

WHY THIS IS A SECOND MODULE AND NOT A FLAG ON THE FIRST. `route_groups_llm` answers
"which groups does this text name?" — one list per caption, which is all the inference-
side mask needs (`--group_router llm`). The grounding loss needs something strictly
richer: for every body-part MENTION, which text COLUMNS spell it out, which groups they
point at, and whether a side was named. A caption-level union cannot be turned into that
after the fact, so the prompt itself has to return mentions.

    {"phrase": "right hand", "groups": ["right_arm"], "side": "right",
     "side_evidence": "right hand"}

WHAT `side_evidence` IS FOR — it is the whole point of the exercise. `parser.py` refuses
to lateralise a verb (ARCHITECTURE.md rule 2) because "steps to the right" is a direction,
not a right leg; the price is that "waves with his right hand" supervises the verb column
bilaterally, which is what holds M1 laterality at +0.194 under the default read-out
against +0.871 under `span` (FINDINGS.md 2026-08-15 §3). An LLM can tell those two apart —
but it is also measurably willing to invent a side on a direction word (20.7 % of that
population, FINDINGS.md 2026-09-22). So the model is required to QUOTE the limb phrase
that licenses the side, and `llm_cache.py` accepts the side only when `parse_caption`
independently agrees that quote binds a side to a limb. The LLM proposes; the audited
regex binder disposes.

**The failure contract is the OPPOSITE of `llm_router`'s, on purpose.** There, any failure
returns `[]` so one bad response cannot kill a 1000-caption eval — correct, because `[]` is
a legitimate answer at inference and the fallback is an unmasked edit. Here the answers are
written to a cache that a 20-hour training run then trusts, so a failure that looks like "no
body parts in this caption" is far worse than a crash: kill the Ollama server mid-build and
every remaining caption would be cached, permanently and silently, as having no mentions.
So `_call` returns `None` for "no answer" (transport, JSON or schema failure) as distinct
from `[]` for "the model says there is nothing here", non-answers are never cached, and a
run of consecutive failures aborts the batch. Raw responses are cached to disk keyed by
(model, prompt version, text) — the prompt version
is in the key because changing the prompt changes the answer, and silently reusing
entries from an older prompt is exactly the kind of stale-artefact bug the
`--attn_ground_verbs` cache note warns about.
"""

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from utils.logger import get_logger
from utils.paths import repo_path

from model.body_groups import GROUP_NAMES

log = get_logger(__name__)

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CACHE_PATH = repo_path("data", "llm_item_labels_cache.json")

# Bump whenever _SYSTEM_PROMPT changes — it is part of the cache key, so an edited
# prompt re-asks instead of silently mixing two prompts' answers in one label set.
PROMPT_VERSION = "items-v1"

_VALID = set(GROUP_NAMES)
_SIDES = {"left", "right", "none"}

_SYSTEM_PROMPT = f"""You label a human-motion caption with the body parts its phrases refer to.

Valid groups (use ONLY these names, exactly as spelled): {", ".join(GROUP_NAMES)}

Return one item for EVERY phrase in the caption that refers to a body part or to a body
movement:
  "phrase": the EXACT substring from the caption, copied character for character. Keep it
            SHORT - the body-part words alone ("left hand", "arms"), or the single verb
            ("walks"). NEVER a whole clause.
  "groups": the group(s) that phrase moves.
  "side":   "left" or "right" only when the caption names WHICH LIMB OF THE BODY performs
            it. Otherwise "none".
  "side_evidence": when "side" is "left" or "right", the EXACT substring that names that
            limb (for example "right hand"). Empty string when "side" is "none".

Rules:
- A limb named with no side -> BOTH sides, side "none". Never guess a side.
- A DIRECTION IS NOT A SIDE. "steps to the right", "turns left", "looks to her left",
  "moves it to the left", "slides to his right" all describe where the motion goes, not
  which limb does it: side "none". Only a named limb ("left hand", "his right foot")
  licenses a side, and that limb phrase must be quoted in "side_evidence".
- A VERB with no body-part noun is still an item ("walks", "kicks", "claps"): the phrase
  is the verb and the groups are the parts that perform it.
- A verb DOES take a side when the caption names the limb performing it. "waves with his
  right hand" gives TWO items: the verb "waves" with side "right" and side_evidence
  "right hand", and the noun "right hand" itself.
- Locomotion that carries the body somewhere (walk, run, jog, step, march, shuffle) ->
  left_leg, right_leg AND root. In-place leg actions (squat, kneel, kick, lunge, stomp)
  -> the legs only, no root.
- A trajectory or orientation change with no limb named (turning, spinning, moving to a
  different spot) -> root.
- Nothing that names a body part and nothing that moves the body ("do it faster",
  "again", "more gracefully") -> {{"items": []}}.
- Output ONLY JSON: {{"items": [...]}}. No explanation.

Examples:
"a person raises his left arm" -> {{"items": [{{"phrase": "left arm", "groups": ["left_arm"], "side": "left", "side_evidence": "left arm"}}, {{"phrase": "raises", "groups": ["left_arm"], "side": "left", "side_evidence": "left arm"}}]}}
"a man walks forward and waves" -> {{"items": [{{"phrase": "walks", "groups": ["left_leg", "right_leg", "root"], "side": "none", "side_evidence": ""}}, {{"phrase": "waves", "groups": ["left_arm", "right_arm"], "side": "none", "side_evidence": ""}}]}}
"a person takes a large step to the right" -> {{"items": [{{"phrase": "step", "groups": ["left_leg", "right_leg", "root"], "side": "none", "side_evidence": ""}}]}}
"she kicks a ball with her right foot" -> {{"items": [{{"phrase": "right foot", "groups": ["right_leg"], "side": "right", "side_evidence": "right foot"}}, {{"phrase": "kicks", "groups": ["right_leg"], "side": "right", "side_evidence": "right foot"}}]}}
"the person nods their head" -> {{"items": [{{"phrase": "head", "groups": ["head"], "side": "none", "side_evidence": ""}}, {{"phrase": "nods", "groups": ["head"], "side": "none", "side_evidence": ""}}]}}
"do the same thing faster" -> {{"items": []}}
"""


def cache_key(model: str, text: str) -> str:
    return f"{model}\x1f{PROMPT_VERSION}\x1f{text}"


def load_cache(path: str = DEFAULT_CACHE_PATH) -> dict[str, list[dict]]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str = DEFAULT_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    os.replace(tmp, path)


def _clean(items) -> list[dict]:
    """Schema-level validation only — the parts that are decidable without the caption.

    Span resolution and the side-evidence veto need the caption text and the tokenizer,
    so they live in `llm_cache.py`; keeping them out of here means this module's cache
    holds the model's raw answer and every downstream policy change is a free re-derive
    rather than 51k fresh calls.
    """
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        phrase = it.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        groups, seen = it.get("groups"), []
        if not isinstance(groups, list):
            continue
        for g in groups:
            if g in _VALID and g not in seen:
                seen.append(g)
        if not seen:
            continue
        side = it.get("side", "none")
        side = side if side in _SIDES else "none"
        evidence = it.get("side_evidence", "")
        out.append({"phrase": phrase, "groups": seen, "side": side,
                    "side_evidence": evidence if isinstance(evidence, str) else ""})
    return out


def _call(text: str, *, model: str, host: str, timeout: float,
          temperature: float) -> list[dict] | None:
    """One Ollama chat call -> schema-validated raw items.

    `[]` means the model answered and found nothing; `None` means there was no usable
    answer. The caller must not cache `None` — see the module docstring.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "format": "json",
        "options": {"temperature": temperature},
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        log.warning(f"LLM items: request failed for {text!r}: {e}")
        return None

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(f"LLM items: unparsable output for {text!r}: {content!r} ({e})")
        return None
    return _clean(parsed.get("items", []) if isinstance(parsed, dict) else parsed)


def route_items_llm(text: str, *, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                    cache_path: str = DEFAULT_CACHE_PATH, timeout: float = 60.0,
                    temperature: float = 0.0) -> list[dict]:
    """Caption -> raw per-mention items, cached to disk. Single-caption convenience form."""
    cache = load_cache(cache_path)
    key = cache_key(model, text)
    if key not in cache:
        items = _call(text, model=model, host=host, timeout=timeout,
                      temperature=temperature)
        if items is None:
            raise RuntimeError(
                f"No usable answer from {model} at {host} for {text!r}. Not cached — a "
                f"cached non-answer is indistinguishable from 'this caption names no "
                f"body part' and would silently weaken the label set.")
        cache[key] = items
        save_cache(cache, cache_path)
    return cache[key]


def route_captions_items_llm(texts, *, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                             cache_path: str = DEFAULT_CACHE_PATH, timeout: float = 60.0,
                             temperature: float = 0.0, workers: int = 4,
                             save_every: int = 500, max_consecutive_failures: int = 20,
                             progress=None) -> dict[str, list[dict]]:
    """Batch form: every distinct caption in `texts` -> raw items, resumable.

    `workers` issues that many concurrent requests. Ollama serialises beyond its own
    OLLAMA_NUM_PARALLEL, so oversubscribing is harmless but buys nothing; 4 is the
    server default. The cache dict is written only from the calling thread (results are
    handed back through the executor), so no lock is needed around the file.

    Non-answers are skipped rather than cached, and `max_consecutive_failures` in a row
    aborts: that is the signature of the server having gone away, and the alternative is
    quietly recording tens of thousands of captions as having no body parts. Progress
    already made is on disk, so the run resumes where it stopped.
    """
    cache = load_cache(cache_path)
    distinct = list(dict.fromkeys(texts))
    todo = [t for t in distinct if cache_key(model, t) not in cache]
    log.info(f"LLM item labels: {len(distinct)} distinct captions, {len(todo)} to call "
             f"({len(distinct) - len(todo)} cached), {workers} workers")

    if todo:
        lock = threading.Lock()
        n_done = n_failed = consecutive = 0
        it = progress(total=len(todo)) if progress else None
        run = lambda t: (t, _call(t, model=model, host=host, timeout=timeout,
                                  temperature=temperature))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for text, items in pool.map(run, todo):
                    if items is None:
                        n_failed += 1
                        consecutive += 1
                        if consecutive >= max_consecutive_failures:
                            raise RuntimeError(
                                f"{consecutive} consecutive failures from {model} at "
                                f"{host} — is the server still up? Aborting instead of "
                                f"caching non-answers as empty label sets. "
                                f"{n_done:,} captions are saved; re-run to resume.")
                    else:
                        consecutive = 0
                        cache[cache_key(model, text)] = items
                        with lock:
                            n_done += 1
                        if n_done % save_every == 0:
                            save_cache(cache, cache_path)
                    if it is not None:
                        it.update(1)
        finally:
            if it is not None:
                it.close()
            save_cache(cache, cache_path)
        if n_failed:
            log.warning(f"{n_failed} caption(s) got no usable answer and were NOT cached; "
                        f"re-run to retry just those.")

    return {t: cache[cache_key(model, t)] for t in distinct
            if cache_key(model, t) in cache}
