"""
Building the caption -> label cache offline, once, for the training loop to look up.

Keyed by caption STRING rather than by (clip, annotation index): the dataset samples a
random annotation per epoch and returns the text itself, so a string key needs no index
bookkeeping and dedupes captions shared across clips for free.
"""

import json
import os

from .parser import to_items

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
