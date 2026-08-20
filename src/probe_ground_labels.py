"""
The label-quality gate for the cross-attention grounding supervision.

Before spending a 500-epoch retrain on a supervision signal, measure the signal. This
builds the caption->body-part label set (`data/body_part_labels`) over a whole split and
answers the four questions that decide whether the retrain is worth starting:

1. COVERAGE, per tier. What share of annotations produce a lateralised item (tier 1 — the
   ones carrying the left/right signal the whole exercise exists for), an unlateralised
   one (tier 2), or nothing? This decides whether the loss fires often enough to shape the
   model, whether those clips need oversampling, and whether the optional tier-3
   generator labels are worth 15-24 GPU-hours. The most frequent UNCOVERED caption words
   are printed alongside, because that list is the cheapest possible vocabulary fix.

2. LATERALITY BALANCE. Left-target vs right-target item counts, which should be ~1.000 by
   construction: half the train split is mirror-augmented clips whose captions are
   verified left/right-swapped. A ratio far from 1 means the parser drops one side, which
   would bias the very axis being supervised — a hard stop.

3. AGREEMENT WITH THE GENERATOR (--audit_checkpoint). The independent auditor. A single
   generation's per-group motion energy picks the named group at 0.990 and the named side
   at 1.000, so it is the one signal here known to resolve laterality. Generate from a
   sample of tier-1 captions and check the energy lands on the group the parser claimed.
   High agreement validates the parser at scale; the disagreements are a concrete list of
   vocabulary bugs.

4. COLUMN ALIGNMENT (mandatory). Assert that `token_spans` columns are exactly
   `token_info` columns, i.e. that the supervised columns index the same L axis the
   attention maps use. Supervising the wrong columns trains nonsense and is invisible
   downstream, so this one is an assertion, not a metric.

    # fast: everything except the generator audit (no GPU, seconds)
    python src/probe_ground_labels.py --data_root data/HumanML3D/HumanML3D

    # full gate, with the generator auditing the parser's labels
    python src/probe_ground_labels.py --data_root data/HumanML3D/HumanML3D \\
        --audit_checkpoint runs/exp_hml3d_x0/checkpoint_latest --audit_n 200
"""

import argparse
import collections
import json
import os
import re

from analysis.ground_audit import audit_with_generator
from data.body_part_labels import LIMB2BASE, build_cache, parse_caption
from data.clips import read_captions, split_ids
from model.body_groups import GROUP_NAMES
from utils.cli import add_logging_args, configure_logging
from utils.logger import get_logger

log = get_logger(__name__)

# Words that carry no body-part information and would only clutter the "most frequent
# uncovered word" list the vocabulary fix is read off.
_STOPWORDS = set("""a an the and or but of to in on at for with from by as is are was were
be been being it its this that these those他 he she they them their his her theirs your you
i we our us then than so very just about into over under up down out off again while during
after before around across through toward towards forward backward back front side sides
person man woman someone somebody figure human character subject people
walk walks walked walking run runs running move moves moved moving turn turns turned
turning step steps stepped stepping stand stands stood standing sit sits sat sitting
go goes going get gets got getting take takes took taking put puts putting make makes
made making do does did doing come comes came coming look looks looked looking
appears seems like slowly quickly fast slow forwards backwards
one two three four five six seven eight nine ten times time way
""".split())


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", default="eval_results/ground_labels")
    p.add_argument("--cache_out", default=None,
                   help="Where to write the caption→items cache. "
                        "Default <data_root>/ground_labels.json; pass 'none' to skip.")
    p.add_argument("--cache_splits", nargs="+", default=["train", "val", "test"],
                   help="Splits the written cache covers (coverage is still reported "
                        "for --split only).")
    p.add_argument("--top_uncovered", type=int, default=50,
                   help="How many frequent uncovered words to list.")

    # encoder — only needed for the column assertion and the cache
    p.add_argument("--text_encoder", default="t5", choices=["clip", "t5"])
    p.add_argument("--t5_version", default="t5-base")
    p.add_argument("--t5_max_length", type=int, default=128)
    p.add_argument("--clip_version", default="ViT-B/32")

    # the generator auditor
    p.add_argument("--audit_checkpoint", default=None,
                   help="Checkpoint to audit the labels with. Omitted → audit skipped.")
    p.add_argument("--audit_n", type=int, default=200,
                   help="Tier-1 captions to audit.")
    p.add_argument("--audit_batch", type=int, default=25,
                   help="Captions generated per shared-noise batch.")
    p.add_argument("--audit_length", type=int, default=120)
    p.add_argument("--audit_guidance", type=float, default=4.0)
    p.add_argument("--audit_seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--smplh_model_path",
                   default="data/motionfix/data/body_models/smplh")
    add_logging_args(p)
    return configure_logging(p.parse_args())


# ── 1 + 2: coverage and balance, from the parser alone (no model, no tokenizer) ──────

def collect_annotations(data_root, split):
    """[(clip_id, caption)] for every annotation in the split."""
    out = []
    for clip_id in split_ids(data_root, split):
        for caption in read_captions(data_root, clip_id):
            out.append((clip_id, caption))
    return out


def coverage_report(annotations, top_uncovered):
    """Per-tier coverage, laterality balance, group distribution, uncovered vocabulary."""
    tiers = collections.Counter()          # per ANNOTATION: best tier it reaches
    item_tiers = collections.Counter()     # per ITEM
    side = collections.Counter()
    groups = collections.Counter()
    uncovered_words = collections.Counter()
    clips_with_tier1 = set()

    for clip_id, caption in annotations:
        mentions = parse_caption(caption)
        if not mentions:
            tiers["none"] += 1
            for w in re.findall(r"[a-z]+", caption.lower()):
                if w not in _STOPWORDS and len(w) > 2:
                    uncovered_words[w] += 1
            continue

        for m in mentions:
            item_tiers[m.tier] += 1
            for g in m.groups:
                groups[g] += 1
            if m.lat:
                side["left" if m.groups[0].startswith("left") else "right"] += 1

        if any(m.lat for m in mentions):
            tiers["tier1"] += 1
            clips_with_tier1.add(clip_id)
        else:
            tiers["tier2"] += 1

    n = max(len(annotations), 1)
    left, right = side["left"], side["right"]
    return {
        "n_annotations": len(annotations),
        "n_clips_with_tier1": len(clips_with_tier1),
        "coverage": {k: tiers[k] / n for k in ("tier1", "tier2", "none")},
        "counts": {k: tiers[k] for k in ("tier1", "tier2", "none")},
        "items_per_tier": {str(k): v for k, v in sorted(item_tiers.items())},
        "laterality": {"left_items": left, "right_items": right,
                       "ratio": left / right if right else float("inf")},
        "group_distribution": {g: groups[g] for g in GROUP_NAMES if groups[g]},
        "top_uncovered_words": uncovered_words.most_common(top_uncovered),
    }


def print_coverage(rep, n_clips):
    c, k = rep["coverage"], rep["counts"]
    log.section(f"1. coverage over {rep['n_annotations']} annotations "
                f"({n_clips} clips)")
    log.info(f"  tier 1  lateralised     {k['tier1']:6d}  {c['tier1']:6.1%}   "
          f"← carries the left/right signal")
    log.info(f"  tier 2  limb, no side   {k['tier2']:6d}  {c['tier2']:6.1%}")
    log.info(f"  none    no part word    {k['none']:6d}  {c['none']:6.1%}   "
          f"← tier-3 territory")
    log.info(f"  clips with >=1 tier-1 annotation: {rep['n_clips_with_tier1']} / {n_clips} "
          f"({rep['n_clips_with_tier1'] / max(n_clips, 1):.1%})")
    log.info(f"  items: {rep['items_per_tier']}")

    lat = rep["laterality"]
    log.section("2. laterality balance")
    log.info(f"  left items {lat['left_items']:6d}   right items {lat['right_items']:6d}   "
          f"ratio {lat['ratio']:.4f}")
    verdict = ("OK" if 0.95 <= lat["ratio"] <= 1.05 else
               "SKEWED — the parser is dropping one side; investigate before training")
    log.info(f"  → {verdict}")

    log.info("\n  group distribution over items:")
    total = sum(rep["group_distribution"].values()) or 1
    for g in GROUP_NAMES:
        v = rep["group_distribution"].get(g, 0)
        if v:
            log.info(f"    {g:10s} {v:7d}  {v / total:6.1%}")

    log.info("\n  most frequent words in UNCOVERED annotations "
             "(candidate vocabulary additions):")
    line = "    "
    for w, cnt in rep["top_uncovered_words"]:
        entry = f"{w}({cnt})  "
        if len(line) + len(entry) > 96:
            log.info(line)
            line = "    "
        line += entry
    if line.strip():
        log.info(line)


# ── 4: the mandatory column assertion ────────────────────────────────────────────────

def assert_columns(encoder, annotations, n=200):
    """token_spans columns must be exactly token_info columns, on real captions.

    Both index encode()'s L axis, which is the axis the attention maps' L dimension is.
    A mismatch means the grounding loss would supervise the wrong columns — silently.
    """
    checked = 0
    for _, caption in annotations[:n]:
        spans = encoder.token_spans(caption)
        cols, _ = encoder.token_info(caption)
        span_cols = [c for c, _, _ in spans]
        assert span_cols == cols, (
            f"column mismatch on {caption!r}:\n  token_spans {span_cols}\n"
            f"  token_info  {cols}")
        for _, s, e in spans:
            assert 0 <= s < e <= len(caption), f"bad span on {caption!r}"
        checked += 1
    return checked


# ── main ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    annotations = collect_annotations(args.data_root, args.split)
    n_clips = len(split_ids(args.data_root, args.split))
    log.info(f"data_root {args.data_root}   split {args.split}")

    rep = coverage_report(annotations, args.top_uncovered)
    print_coverage(rep, n_clips)

    # ── 4 (assertion) + the cache, both of which need the encoder ──
    from model.text_encoder import build_text_encoder
    log.section("4. column alignment")
    encoder = build_text_encoder(vars(args), device="cpu")
    checked = assert_columns(encoder, annotations)
    log.info(f"  token_spans columns == token_info columns on {checked} real captions → OK")
    log.info(f"  encoder L axis = {encoder.max_length} "
          f"({args.text_encoder}{'/' + args.t5_version if args.text_encoder == 't5' else ''})")

    cache_out = args.cache_out or os.path.join(args.data_root, "ground_labels.json")
    if cache_out.lower() != "none":
        cache = build_cache(args.data_root, encoder, splits=args.cache_splits,
                            out_path=cache_out)
        n_items = sum(len(v) for v in cache.values())
        log.info(f"\n  wrote {cache_out}: {len(cache)} captions, {n_items} items "
              f"(splits {args.cache_splits})")

    audit = None
    if args.audit_checkpoint:
        from utils.cli import resolve_device
        audit = audit_with_generator(args, annotations, resolve_device(args.device))

    out = os.path.join(args.out_dir, f"ground_labels_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "data_root": args.data_root, "split": args.split,
            "n_clips": n_clips, "encoder_max_length": encoder.max_length,
            "vocabulary_size": len(LIMB2BASE),
            **rep,
            "audit": None if audit is None else {
                "checkpoint": args.audit_checkpoint,
                "blocks": audit["blocks"], "rows": audit["rows"]},
        }, f, indent=2)
    log.info(f"\nWrote {out}")


if __name__ == "__main__":
    main()
