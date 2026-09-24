"""
Caption -> body-part supervision labels for the cross-attention grounding loss.

The grounding loss needs to know, for a caption, WHICH text tokens should attend to
WHICH body-part group token: a "left arm" span must route to the `left_arm` group. This
package produces exactly that, as a list of items per caption:

    {"W": [text-token columns], "S": [group indices], "tier": 1|2, "lat": bool}

Nothing here runs during training. `build_cache` writes one JSON keyed by caption string
in a single offline pass over the dataset; the Trainer loads it once and every step is a
dict lookup.

TIERS
  tier 1  the caption lateralises the limb ("his left arm")  -> S = {left_arm}, mirror ON
  tier 2  the limb is named but not lateralised ("the arms") -> S = {left_arm, right_arm}
          (also "both arms"), so a wrong side is never forced; mirror OFF
  tier 3  a verb implies the limb ("walks") -> always tier 2; see `vocabulary`

Captions naming no body part produce no items and simply get no grounding loss.

Module map:
  vocabulary.py  which words name which groups, and why each inclusion was made
  parser.py      caption -> mentions -> cache items / routed group names
  cache.py       the offline build over a dataset split
  selftest.py    parser assertions and corpus verb coverage
  llm_router.py  the inference-side LLM tier: instruction -> group set (Option 13)
  llm_items.py   the training-side LLM tier: caption -> per-mention labels (§13.3)
  llm_cache.py   those mentions -> a vetted label cache, merged with the regex one
"""

from .cache import build_cache, load_cache
from .llm_cache import audit, build_cache_llm, mirror_caption_pairs
from .llm_items import route_captions_items_llm, route_items_llm
from .llm_router import route_captions_llm, route_groups_llm
from .parser import Mention, parse_caption, route_groups, to_items
from .selftest import check_verb_coverage, self_check
from .vocabulary import (HEIGHT_WORDS, LATERALISABLE, LIMB2BASE, VERB2GROUPS,
                         VERB_FORMS, VERBS_EXCLUDED)

__all__ = [
    "HEIGHT_WORDS", "LATERALISABLE", "LIMB2BASE", "Mention", "VERB2GROUPS",
    "VERBS_EXCLUDED", "VERB_FORMS", "audit", "build_cache", "build_cache_llm",
    "check_verb_coverage", "load_cache", "mirror_caption_pairs", "parse_caption",
    "route_captions_items_llm", "route_captions_llm", "route_groups", "route_groups_llm",
    "route_items_llm", "self_check",
    "to_items",
]
