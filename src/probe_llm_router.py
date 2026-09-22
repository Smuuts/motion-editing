"""
Option 13's full tier, measured (docs/MaskOptions.md, docs/FINDINGS.md "How much of
MotionFix a group mask can even address").

The regex+verb router (`route_groups`) resolves 82.6% of the 1013 MotionFix test
instructions to a body-part group set, laterality-correct by construction, no model
call. The option always specified a second tier — an actual LLM call — for the rest,
but only ever *projected* its value (originally "~11 points", revised down once verbs
closed most of the gap). This script runs the built LLM tier (`route_groups_llm`,
`data/body_part_labels/llm_router.py`) over the same 1013 captions and reports the
REAL number: how much coverage it adds, where it agrees with the regex router, and
where the two disagree even on captions the regex already resolves (a disagreement
there is a red flag about the LLM, not a coverage gain).

    python src/probe_llm_router.py \\
        --testset data/motionfix/data/motionfix-dataset/motionfix_test.pth.tar \\
        --out eval_results/llm_router_coverage.json
"""

import argparse
import json
import os

from data.body_part_labels import route_captions_llm, route_groups
from data.body_part_labels.llm_router import DEFAULT_HOST, DEFAULT_MODEL
from utils.cli import add_logging_args, configure_logging
from utils.logger import get_logger

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testset", required=True,
                   help="MotionFix joblib dump (keyid -> {'text': ..., ...}), e.g. "
                        "data/motionfix/data/motionfix-dataset/motionfix_test.pth.tar")
    p.add_argument("--group_mode", default="parts", choices=["parts", "joints"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--cache_path", default=None,
                   help="Default: data/llm_group_router_cache.json (resumable).")
    p.add_argument("--out", default=None, help="Write the full report as JSON here.")
    add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(args)

    import joblib
    log.info(f"Loading test set: {args.testset}")
    data = joblib.load(args.testset)
    texts = [data[k]["text"] for k in data]
    n = len(texts)
    log.info(f"{n} instructions ({len(set(texts))} distinct captions)")

    regex = {t: route_groups(t, args.group_mode) for t in set(texts)}

    llm_kwargs = {"model": args.model, "host": args.host}
    if args.cache_path:
        llm_kwargs["cache_path"] = args.cache_path
    llm = route_captions_llm(list(set(texts)), progress=lambda it: log.progress(
        it, desc="LLM routing", leave=True), **llm_kwargs)

    n_regex = sum(1 for t in texts if regex[t])
    n_llm = sum(1 for t in texts if llm[t])
    unresolved_regex = [t for t in set(texts) if not regex[t]]
    n_llm_of_unresolved = sum(1 for t in texts if t in unresolved_regex and llm[t])

    # Confusion over DISTINCT captions (each caption counted once, not weighted by how
    # many clips share it) — the routing decision is a property of the text, not the clip.
    both, regex_only, llm_only, neither = [], [], [], []
    disagree_both_resolved = []
    for t in sorted(set(texts)):
        r, l = set(regex[t]), set(llm[t])
        if r and l:
            both.append(t)
            if r != l:
                disagree_both_resolved.append({"text": t, "regex": sorted(r), "llm": sorted(l)})
        elif r and not l:
            regex_only.append(t)
        elif l and not r:
            llm_only.append(t)
        else:
            neither.append(t)

    report = {
        "n_instructions": n,
        "n_distinct_captions": len(set(texts)),
        "coverage_regex_pct": round(100 * n_regex / n, 1),
        "coverage_llm_pct": round(100 * n_llm / n, 1),
        "llm_pct_of_regex_unresolved": round(
            100 * n_llm_of_unresolved / max(len(unresolved_regex), 1), 1),
        "n_regex_unresolved_captions": len(unresolved_regex),
        "confusion_distinct_captions": {
            "both_resolve": len(both),
            "regex_only": len(regex_only),
            "llm_only": len(llm_only),
            "neither": len(neither),
        },
        "n_disagree_when_both_resolve": len(disagree_both_resolved),
        "disagreements_both_resolve": disagree_both_resolved,
        "examples_llm_only": [{"text": t, "llm": llm[t]} for t in llm_only[:20]],
        "examples_neither": neither[:20],
        "model": args.model,
    }

    log.section("Option 13 full tier — measured")
    log.table(["metric", "value"], [
        ["instructions", n],
        ["distinct captions", len(set(texts))],
        ["regex+verb coverage", f"{report['coverage_regex_pct']}%"],
        ["LLM coverage", f"{report['coverage_llm_pct']}%"],
        ["LLM coverage of regex-unresolved", f"{report['llm_pct_of_regex_unresolved']}%"],
        ["both resolve (distinct)", len(both)],
        ["regex only (distinct)", len(regex_only)],
        ["llm only (distinct)", len(llm_only)],
        ["neither (distinct)", len(neither)],
        ["disagree when both resolve", len(disagree_both_resolved)],
    ])

    if disagree_both_resolved:
        log.info("\nDisagreements where BOTH routers named a group (first 10):")
        for d in disagree_both_resolved[:10]:
            log.info(f"  {d['text']!r}: regex={d['regex']} llm={d['llm']}")

    if llm_only:
        log.info(f"\n{len(llm_only)} captions the LLM resolves that regex does not "
                  f"(first 10):")
        for t in llm_only[:10]:
            log.info(f"  {t!r} -> {llm[t]}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
