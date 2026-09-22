"""
The "full tier" of Option 13's group router (docs/MaskOptions.md) — an actual LLM call,
instruction -> body-part group set, as an alternative to `route_groups()`'s regex+verb
"cheap tier" (parser.py). The option always specified both tiers; only the cheap one was
ever built. This module closes that gap so the full tier's coverage can be MEASURED
against the cheap tier's 82.6% (see docs/FINDINGS.md "How much of MotionFix a group mask
can even address") rather than left as the doc's own ~11-point projection.

Talks to a local Ollama server (default `qwen2.5:7b-instruct` on localhost:11434) over
its OpenAI-compatible chat endpoint. No new project dependency: stdlib `urllib` only.

Drop-in contract with `route_groups()` (parser.py), on purpose — a caller should be able
to swap one for the other behind a flag with no other code change:
  - same signature `(text, group_mode="parts") -> list[str]`,
  - names come from the SAME coarse vocabulary (`GROUP_NAMES`) regardless of
    `group_mode` — `named_token_indices()` does the parts -> joints expansion
    downstream, exactly as it does for the regex router,
  - `[]` is a valid, common, non-error answer (manner/timing/trajectory edits with no
    correct group mask) — callers fall back to a temporal or unmasked edit, same as for
    `route_groups()`.

Unlike the regex router, this one can fail at runtime (server down, model missing,
malformed JSON) — every such failure is caught and returns `[]` rather than raising, so
one bad response can't kill a 1000+ caption batch run. Results are cached to disk keyed
by `(model, text)`, since a caption's answer is deterministic-in-intent (temperature 0)
and re-asking costs a network round trip for nothing.
"""

import json
import os
import urllib.error
import urllib.request

from utils.logger import get_logger
from utils.paths import repo_path

from model.body_groups import GROUP_NAMES

log = get_logger(__name__)

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CACHE_PATH = repo_path("data", "llm_group_router_cache.json")

_VALID = set(GROUP_NAMES)  # {root, left_leg, right_leg, spine, left_arm, right_arm, head}

_SYSTEM_PROMPT = f"""You label a motion-editing instruction with the body-part group(s) it refers to.

Valid groups (use ONLY these names, exactly as spelled): {", ".join(GROUP_NAMES)}

Rules:
- If the instruction names a side (left/right) for a limb, return ONLY that side's group.
- If it names a limb with no side, return BOTH sides (e.g. "raise the arms" -> ["left_arm","right_arm"]).
- If it describes locomotion (walking, running, stepping — an actual gait that moves the
  body) return the leg groups AND root.
- "root" alone is for a trajectory/orientation change with NO limb named (turning,
  spinning, moving to a different spot).
- If the instruction has no identifiable body part AND no trajectory/orientation change —
  pure speed, timing, repetition or vague style ("do it faster", "again", "more
  gracefully") — return []. Do not reach for "root" just because the instruction is vague.
- Output ONLY a JSON object: {{"groups": [...]}}. No explanation.

Examples:
"raise the left arm higher" -> {{"groups": ["left_arm"]}}
"kick with the right leg" -> {{"groups": ["right_leg"]}}
"walk faster and turn around" -> {{"groups": ["left_leg", "right_leg", "root"]}}
"make a wider turn" -> {{"groups": ["root"]}}
"do it faster" -> {{"groups": []}}
"do the same thing again" -> {{"groups": []}}
"nod your head" -> {{"groups": ["head"]}}
"""


def _cache_key(model: str, text: str) -> str:
    return f"{model}\x1f{text}"


def load_cache(path: str = DEFAULT_CACHE_PATH) -> dict[str, list[str]]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, list[str]], path: str = DEFAULT_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _call(text: str, *, model: str, host: str, timeout: float,
          temperature: float) -> list[str]:
    """One Ollama chat call -> validated group list, or [] on any failure."""
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
        log.warning(f"LLM router: request failed for {text!r}: {e}")
        return []

    content = data.get("message", {}).get("content", "")
    try:
        groups = json.loads(content)["groups"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning(f"LLM router: unparsable output for {text!r}: {content!r} ({e})")
        return []

    seen: list[str] = []
    for g in groups:
        if g not in _VALID:
            log.warning(f"LLM router: hallucinated group {g!r} for {text!r} — dropped")
        elif g not in seen:
            seen.append(g)
    return seen


def route_groups_llm(text: str, group_mode: str = "parts", *, model: str = DEFAULT_MODEL,
                     host: str = DEFAULT_HOST, cache_path: str = DEFAULT_CACHE_PATH,
                     timeout: float = 60.0, temperature: float = 0.0) -> list[str]:
    """Instruction -> body-part groups, via a local LLM call, cached to disk.

    `group_mode` is accepted only for call-site parity with `route_groups()` — the
    returned names are always the coarse vocabulary (see module docstring); it is not
    read here.

    One call per distinct (model, text): a persistent on-disk cache (default
    `data/llm_group_router_cache.json`) is read and rewritten on every miss, so a
    long batch run is resumable if interrupted — the cost (re-reading a small JSON
    file per new caption) is negligible against a network round trip.
    """
    cache = load_cache(cache_path)
    key = _cache_key(model, text)
    if key in cache:
        return cache[key]
    groups = _call(text, model=model, host=host, timeout=timeout, temperature=temperature)
    cache[key] = groups
    save_cache(cache, cache_path)
    return groups


def route_captions_llm(texts: list[str], *, model: str = DEFAULT_MODEL,
                       host: str = DEFAULT_HOST, cache_path: str = DEFAULT_CACHE_PATH,
                       timeout: float = 60.0, temperature: float = 0.0,
                       save_every: int = 50, progress=None) -> dict[str, list[str]]:
    """Batch form for a measurement run: route every distinct caption in `texts`,
    reusing `route_groups_llm`'s cache, but loading/saving it once (plus every
    `save_every` new calls) instead of once per caption — the throughput path for a
    1000+-caption sweep. Returns {caption: groups} for every text in `texts`
    (duplicates resolved once). `progress` is an optional `tqdm`-like wrapper.
    """
    cache = load_cache(cache_path)
    distinct = list(dict.fromkeys(texts))  # de-dup, keep first-seen order
    iterator = progress(distinct) if progress else distinct
    n_new = 0
    for text in iterator:
        key = _cache_key(model, text)
        if key in cache:
            continue
        cache[key] = _call(text, model=model, host=host, timeout=timeout,
                           temperature=temperature)
        n_new += 1
        if n_new % save_every == 0:
            save_cache(cache, cache_path)
    if n_new % save_every != 0:
        save_cache(cache, cache_path)
    return {text: cache[_cache_key(model, text)] for text in distinct}
