"""
Build the LLM-labelled grounding cache, and audit it against the regex one.

This is the offline half of MaskOptions.md §13.3: it asks a local LLM for the body-part
mentions in every caption of a dataset, turns them into the `{caption: [items]}` file the
TokenCompose grounding loss looks up, and prints the pre-registered label-level gates.
Nothing here touches training — the retrain is the regex recipe with
`--attn_ground_cache` pointed at the file this writes.

    # 1. build (resumable; ~3 h for HumanML3D's 51k captions at 4 workers)
    python src/build_ground_labels_llm.py --data_root data/HumanML3D/HumanML3D_smplh

    # 2. re-derive the cache from the SAME cached LLM answers under different filters
    #    (free — no model calls), e.g. the replacement arm instead of the additive one
    python src/build_ground_labels_llm.py --data_root data/HumanML3D/HumanML3D_smplh \\
        --no-merge_regex --out data/HumanML3D/HumanML3D_smplh/ground_labels_llm_only.json

    # 3. the ablation arms, for reading the gates against
    #    --no-mirror_gate      how much the free self-consistency check is doing
    #    --max_phrase_words 99 how much the long-phrase filter is doing

The gates this prints are the ones registered in MaskOptions.md §13.3 before any GPU time
is spent: tier-1 item count must not fall below the regex set's (the merge is additive,
so it can only rise), coverage must not fall, and mirror-twin consistency must be ~100 %
after the gate. A run that misses them should not proceed to the 20 h retrain.
"""

import argparse
import json
import os

from data.body_part_labels import to_items
from data.body_part_labels.llm_cache import (audit, build_cache_llm,
                                             mirror_caption_pairs, tier1_sides)
from data.body_part_labels.llm_items import (DEFAULT_CACHE_PATH, DEFAULT_HOST,
                                             DEFAULT_MODEL, PROMPT_VERSION,
                                             route_captions_items_llm)
from data.clips import read_captions, split_ids
from analysis.instructions import MIRROR
from utils.cli import add_logging_args, configure_logging
from utils.logger import get_logger
from utils.paths import resolve_repo_path

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D_smplh")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out", default=None,
                   help="Output label cache. Default <data_root>/ground_labels_llm.json. "
                        "Keep it distinct from ground_labels.json: --attn_ground_cache is "
                        "keyed by FILE, so a stale path silently trains the wrong labels.")
    p.add_argument("--group_mode", default="parts", choices=["parts", "joints"])

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--llm_cache", default=DEFAULT_CACHE_PATH,
                   help="Raw LLM responses, keyed by (model, prompt version, caption). "
                        "Separate from the label cache so filter changes re-derive free.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Only label the first N distinct captions — for a smoke run.")

    p.add_argument("--merge_regex", action=argparse.BooleanOptionalAction, default=True,
                   help="Emit regex ∪ verified-LLM (default) rather than LLM alone. The "
                        "additive form is the recommended one: the LLM drops ~30%% of the "
                        "regex's tier-1 items and disagrees on ~10%% of the rest.")
    p.add_argument("--mirror_gate", action=argparse.BooleanOptionalAction, default=True,
                   help="Demote tier-1 items whose mirror twin disagrees (default on). "
                        "Free — the corpus ships the twins.")
    p.add_argument("--include_verbs", action=argparse.BooleanOptionalAction, default=True,
                   help="Tier-3 verb labels in the REGEX half of the merge. Must match "
                        "the --attn_ground_verbs of the run this cache is built for.")
    p.add_argument("--max_phrase_words", type=int, default=5)
    p.add_argument("--max_target_groups", type=int, default=3,
                   help="Drop an item targeting more than this many groups (default 3, "
                        "the regex label set's own maximum). Catches the whole-body verbs "
                        "the regex vocabulary excludes on purpose — crawl, swim, "
                        "cartwheel — which the LLM happily labels with 4-6 of 7 groups.")

    p.add_argument("--text_encoder", default="t5", choices=["clip", "t5"])
    p.add_argument("--t5_version", default="t5-base")
    p.add_argument("--t5_max_length", type=int, default=128)
    p.add_argument("--clip_version", default="ViT-B/32")
    add_logging_args(p)
    return p.parse_args()


def collect_captions(data_root, splits):
    seen = set()
    for split in splits:
        if not os.path.exists(os.path.join(data_root, f"{split}.txt")):
            log.warning(f"no {split}.txt under {data_root} — skipping")
            continue
        for clip_id in split_ids(data_root, split):
            seen.update(read_captions(data_root, clip_id))
    return sorted(seen)


def mirror_consistency(cache, pairs, group_mode="parts"):
    """(checked, consistent) over mirror-twin caption pairs — the free label gate."""
    checked = consistent = 0
    for text, twins in pairs.items():
        if text not in cache:
            continue
        a = tier1_sides(cache[text], group_mode)
        for twin in twins:
            if twin not in cache:
                continue
            checked += 1
            consistent += (frozenset(MIRROR.get(g, g) for g in a)
                           == tier1_sides(cache[twin], group_mode))
    return checked, consistent


def print_comparison(regex_stats, llm_stats):
    rows = [
        ("captions",            "captions",     "{:,}"),
        ("covered (≥1 item)",   "covered",      "{:,}"),
        ("coverage",            "coverage",     "{:.1%}"),
        ("items",               "items",        "{:,}"),
        ("tier-1 (lateralised)", "tier1_items", "{:,}"),
        ("tier-2",              "tier2_items",  "{:,}"),
        ("chance m_S (|S|/G)",  "chance_m_S",   "{:.3f}"),
        ("arm : leg",           "arm_over_leg", "{:.2f}×"),
        ("left : right",        "left_over_right", "{:.3f}"),
    ]
    log.info(f"  {'':<22} {'regex':>14} {'this cache':>14}")
    for label, key, fmt in rows:
        log.info(f"  {label:<22} {fmt.format(regex_stats[key]):>14} "
                 f"{fmt.format(llm_stats[key]):>14}")


def main():
    args = parse_args()
    configure_logging(args)
    # Repo-relative, like every other path in a checkpoint config, so the label cache
    # this writes lands beside the dataset no matter where the script is invoked from.
    args.data_root = resolve_repo_path(args.data_root)

    log.section("1. captions")
    captions = collect_captions(args.data_root, args.splits)
    if args.limit:
        captions = captions[:args.limit]
    log.info(f"  {len(captions):,} distinct captions over splits {args.splits}")

    log.section("2. LLM mentions")
    log.info(f"  {args.model} at {args.host}, prompt {PROMPT_VERSION}, "
             f"cache {args.llm_cache}")
    try:
        from tqdm import tqdm
        # mininterval keeps a multi-hour background run's log readable — the default
        # redraws many times a second, which is fine on a TTY and useless in a file.
        progress = lambda total: tqdm(total=total, unit="cap", smoothing=0.05,
                                      mininterval=30.0)
    except ImportError:
        progress = None
    raw = route_captions_items_llm(captions, model=args.model, host=args.host,
                                   cache_path=args.llm_cache, timeout=args.timeout,
                                   workers=args.workers, progress=progress)
    n_raw = sum(len(v) for v in raw.values())
    empty = sum(1 for v in raw.values() if not v)
    log.info(f"  {n_raw:,} raw mentions; {empty:,} captions returned none "
             f"({empty / max(len(raw), 1):.1%})")

    log.section("3. filters → label cache")
    from model.text_encoder import build_text_encoder
    encoder = build_text_encoder(vars(args), device="cpu")
    out_path = args.out or os.path.join(args.data_root, "ground_labels_llm.json")
    cache, stats = build_cache_llm(
        args.data_root, encoder, raw, captions=captions,
        splits=args.splits, group_mode=args.group_mode,
        out_path=out_path, include_verbs=args.include_verbs,
        merge_regex=args.merge_regex, mirror_gate=args.mirror_gate,
        max_phrase_words=args.max_phrase_words,
        max_target_groups=args.max_target_groups)
    for k in ("dropped_phrase_too_long", "dropped_phrase_not_in_caption",
              "dropped_target_too_broad",
              "dropped_surplus_emission", "dropped_outside_truncation",
              "captions_without_llm_answer",
              "ambiguous_occurrence", "side_verified", "side_vetoed",
              "side_implied_without_claim", "llm_items_resolved",
              "mirror_pairs_checked", "mirror_pairs_consistent", "captions_demoted",
              "demoted_by_mirror_gate", "dropped_target_too_broad_after_demotion",
              "llm_items_added", "llm_items_added_tier1",
              "llm_items_upgraded", "llm_items_dropped_conflict",
              "llm_upgrade_near_miss"):
        if k in stats:
            log.info(f"  {k:<32} {stats[k]:>10,}")
    log.info(f"  wrote {out_path}")

    log.section("4. gates")
    regex_cache = {t: to_items(t, encoder.token_spans(t), args.group_mode,
                               args.include_verbs) for t in cache}
    regex_stats, llm_stats = audit(regex_cache, args.group_mode), audit(cache, args.group_mode)
    print_comparison(regex_stats, llm_stats)

    pairs = mirror_caption_pairs(args.data_root, args.splits)
    checked, consistent = mirror_consistency(cache, pairs, args.group_mode)
    r_checked, r_consistent = mirror_consistency(regex_cache, pairs, args.group_mode)
    log.info("")
    log.info(f"  mirror-twin consistency  regex {r_consistent}/{r_checked} "
             f"= {r_consistent / max(r_checked, 1):.1%}   "
             f"this cache {consistent}/{checked} = {consistent / max(checked, 1):.1%}")

    gates = [
        ("coverage ≥ regex", llm_stats["coverage"] >= regex_stats["coverage"],
         f"{llm_stats['coverage']:.1%} vs {regex_stats['coverage']:.1%}"),
        ("tier-1 items ≥ regex", llm_stats["tier1_items"] >= regex_stats["tier1_items"],
         f"{llm_stats['tier1_items']:,} vs {regex_stats['tier1_items']:,}"),
        ("mirror consistency ≥ 99%", consistent / max(checked, 1) >= 0.99,
         f"{consistent / max(checked, 1):.1%}"),
        ("left:right balance in [0.95, 1.05]",
         0.95 <= llm_stats["left_over_right"] <= 1.05,
         f"{llm_stats['left_over_right']:.3f}"),
    ]
    log.info("")
    for name, ok, detail in gates:
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {name:<36} {detail}")
    if not all(ok for _, ok, _ in gates):
        log.warning("  One or more label gates FAILED — see MaskOptions.md §13.3 before "
                    "spending the retrain.")

    report = os.path.splitext(out_path)[0] + "_audit.json"
    with open(report, "w") as f:
        json.dump({"data_root": args.data_root, "splits": args.splits,
                   "model": args.model, "prompt_version": PROMPT_VERSION,
                   "merge_regex": args.merge_regex, "mirror_gate": args.mirror_gate,
                   "max_phrase_words": args.max_phrase_words,
                   "max_target_groups": args.max_target_groups,
                   "filters": stats, "regex": regex_stats, "llm": llm_stats,
                   "mirror": {"checked": checked, "consistent": consistent,
                              "regex_checked": r_checked,
                              "regex_consistent": r_consistent},
                   "gates": {name: ok for name, ok, _ in gates}}, f, indent=2)
    log.info(f"\n  audit → {report}")


if __name__ == "__main__":
    main()
