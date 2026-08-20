"""
Per-item seeding for the Stage-1 inversion.
"""

import zlib

def derive_seed(base_seed: int, item_id: str) -> int:
    """Per-item inversion seed from a run-level seed and an item's identity.

    Lives beside `invert` because it is a property of how `invert` draws noise, not of any
    one caller. `invert` builds its `x_t` ladder from a fresh `Generator(seed)`, so the draw
    depends only on `(seed, shape)` — which means a batch job passing one seed to every item
    gives **any two items of equal frame count a bit-identical noise realisation**. On the
    MotionFix test set that is 1013 clips over ~60 distinct lengths, i.e. ~17 clips per
    length sharing their whole trajectory noise, and the metric consuming them is retrieval,
    where generations are scored against each other.

    Deriving keeps a run exactly reproducible from one `--seed` (which still moves every item
    together, so seed-spread measurements work unchanged) while making the draws independent
    across items. `crc32` rather than `hash()` because Python string hashing is salted per
    process, so `hash()` would not be reproducible across runs.

    NOTE any multi-item caller wants this — the probes still pass a bare `--seed` across
    their clip loops and therefore still share ladders between equal-length clips.
    """
    return (base_seed * 1_000_003 + zlib.crc32(item_id.encode())) % (2 ** 31 - 1)
