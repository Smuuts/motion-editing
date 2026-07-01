"""
Format the MotionFix-comparable TMR retrieval results (from run_motionfix_metrics.py) into a
readable table: one row per edit config (guidance scale), full test set + batches-of-32.

Metrics (MotionFix benchmark standard, higher is better, %):
  R@1/2/3       — target <-> generated : instruction following (edit fidelity)
  R@1/2/3_s2t   — source <-> generated : motion preservation

The `*_s0` (identity) row is the plumbing calibration: R@1_s2t should be ~100.
Rows are sorted by guidance scale; the best instruction-following (R@1) row is flagged.
Runs in any env (stdlib only).

Example:
    python src/eval/aggregate_summary.py \
        --tmr eval_results/motionfix/tmr_metrics.json \
        --out_dir eval_results/motionfix
"""

import os
import re
import json
import argparse


def _scale_key(cfg):
    """Sort configs by their trailing _s<scale> value (e.g. m2_only_s2.5 -> 2.5)."""
    m = re.search(r"_s(-?\d+(?:\.\d+)?)$", cfg)
    return (0, float(m.group(1))) if m else (1, cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmr", required=True, help="JSON from run_motionfix_metrics.py.")
    ap.add_argument("--out_dir", default="eval_results/motionfix")
    args = ap.parse_args()

    tmr = json.load(open(args.tmr))

    rows = []
    for cfg, m in tmr.items():
        full, batch = m.get("full", {}), m.get("batches", {})
        rows.append({
            "config": cfg, "n": m.get("n"),
            "R@1": full.get("R@1"), "R@2": full.get("R@2"), "R@3": full.get("R@3"),
            "R@1_s2t": full.get("R@1_s2t"), "R@2_s2t": full.get("R@2_s2t"),
            "R@3_s2t": full.get("R@3_s2t"),
            "R@1_b": batch.get("R@1"), "R@1_s2t_b": batch.get("R@1_s2t"),
        })
    rows.sort(key=lambda r: _scale_key(r["config"]))

    # flag the best instruction-following (target<->generated R@1, full set)
    best = max((r for r in rows if isinstance(r["R@1"], (int, float))),
               key=lambda r: r["R@1"], default=None)

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    cols = [("config", 16), ("n", 5), ("R@1", 6), ("R@2", 6), ("R@3", 6),
            ("R@1_s2t", 8), ("R@2_s2t", 8), ("R@3_s2t", 8), ("R@1_b", 6), ("R@1_s2t_b", 10)]

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))

    lines = [
        "# MotionFix TMR retrieval (SMPL-H editor)\n",
        "R@k (%, full test set unless `_b` = batches-of-32). "
        "`R@k` = target<->generated (**instruction following**); "
        "`R@k_s2t` = source<->generated (**motion preservation**). "
        "The `_s0` row is the identity calibration (R@1_s2t should be ~100). "
        "Higher is better; there is a preservation/edit trade-off across scales.\n",
        "| " + " | ".join(c for c, _ in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        tag = "  **best R@1**" if best is not None and r["config"] == best["config"] else ""
        lines.append("| " + " | ".join(fmt(r[c]) for c, _ in cols) + " |" + tag)
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out_dir, "summary.md"), "w").write(md)
    print(md)
    print(f"Wrote {args.out_dir}/summary.json and summary.md")


if __name__ == "__main__":
    main()
