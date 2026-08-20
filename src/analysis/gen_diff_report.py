"""
Rendering the generation-space divergence results: the read-out table, the pooled
forced-choice verdicts, and the one paragraph the gate exists to produce.

Kept beside `analysis/gen_diff.py` rather than in the script so the numbers and the
sentences that interpret them cannot drift apart.
"""

import numpy as np

from model.body_groups import GROUP_NAMES
from utils.logger import get_logger

log = get_logger(__name__)


def family_label(readout, space):
    return f"{readout}·{space}"


def mean_of(per_seed, key):
    return float(np.mean([np.mean(s[key]) for s in per_seed]))


def print_table(families, controls):
    cols = ("lat", "cat", "top1", "align", "r_lat", "r_cat", "r_off", "|D|", "D~act")
    hdr = f"{'readout':16} | " + " ".join(f"{c:>6}" for c in cols)
    log.info("\n" + hdr)
    log.info("-" * len(hdr))
    for name, per_seed in families.items():
        vals = [
            float(np.mean([np.mean(s["lat_wins"]) for s in per_seed])),
            float(np.mean([np.mean(s["cat_wins"]) for s in per_seed])),
            float(np.mean([np.mean(s["top1_wins"]) for s in per_seed])),
            mean_of(per_seed, "align"),
            float(np.mean([s["r_laterality"] for s in per_seed])),
            float(np.mean([s["r_category"] for s in per_seed])),
            float(np.mean([s["r_offdiag"] for s in per_seed])),
            mean_of(per_seed, "magnitude"),
            controls.get(name, float("nan")),
        ]
        log.info(f"{name:16} | " + " ".join(f"{v:6.3f}" for v in vals))
    log.info("-" * len(hdr))
    log.info(f"chance: lat/cat 0.500   top1/align {1/len(GROUP_NAMES):.3f}   "
          "(r → 1 = the read-out ignores the instruction)")


def print_verdicts(verdicts):
    log.section("forced choices, pooled over seeds × instructions")
    for name, blocks in verdicts.items():
        for key in ("laterality", "category", "top1"):
            b = blocks[key]
            flag = ("PASS" if b["beats_chance"]
                    else "BELOW chance" if b["below_chance"] else "at chance")
            log.info(f"  {name:16} {key:10} {b['accuracy']:.3f}  "
                  f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]  n={b['n']:3d}  "
                  f"chance {b['chance']:.3f}  → {flag}")


def print_reading(verdicts):
    """The one thing the gate exists to decide."""
    lat = verdicts[family_label("paired", "joint")]["laterality"]
    cat = verdicts[family_label("paired", "joint")]["category"]
    energy_lat = verdicts[family_label("energy", "joint")]["laterality"]
    log.section("reading")
    if lat["beats_chance"]:
        log.info(f"LATERALITY PASSES on the generator ({lat['accuracy']:.3f}, CI lower "
              f"{lat['ci95'][0]:.3f} > 0.5). The generation-space route can deliver a "
              "group selector on the frozen backbone — wire D into masking.build_mask "
              'as mask_mode="gen_diff" and score it against the LLM router.')
    else:
        log.info(f"LATERALITY FAILS on the generator ({lat['accuracy']:.3f}, CI "
              f"[{lat['ci95'][0]:.3f}, {lat['ci95'][1]:.3f}] straddles/undershoots 0.5). "
              "Left/right is absent from the WEIGHTS, not just from the editing "
              "read-out — the stronger negative, and it makes attention "
              "supervision) the remaining laterality route.")
    log.info(f"category: {cat['accuracy']:.3f} "
          f"({'above' if cat['beats_chance'] else 'at/below'} chance) — whether the "
          "generator resolves arm-vs-leg is a separate, weaker claim.")
    log.info(f"is the differencing needed? paired lat {lat['accuracy']:.3f} vs plain motion "
          f"energy {energy_lat['accuracy']:.3f} — if these match, D adds nothing over "
          "reading the generation itself.")
