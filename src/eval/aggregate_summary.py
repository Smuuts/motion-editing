"""
Aggregate the MotionFix-comparable evaluation into one table: SMPL-space TMR retrieval (from
run_motionfix_metrics.py) next to the HumanML3D-native AED (from native_metrics.py), per edit
config, plus the bridge-ceiling reference rows.

Reads the TMR JSON (run_motionfix_metrics --out) and the per-config native JSONs, and writes a
combined summary.{json,md}. Runs in either env (stdlib only).

Example:
    python src/eval/aggregate_summary.py \
        --tmr eval_results/motionfix/tmr_metrics.json \
        --native eval_results/motionfix/native_m2_only_s2.5.json \
        --native eval_results/motionfix/native_m2_only_s5.json \
        --native eval_results/motionfix/native_m2_only_s7.5.json \
        --out_dir eval_results/motionfix
"""

import os
import json
import argparse

# bridge-ceiling config names (produced by joints2smpl_fit --fit_source / --fit_gt_target) — shown
# as reference rows so the method's TMR numbers are read against what the joints->SMPL bridge allows.
CEILING_KEYS = {"_perfect_source", "_recon_source_fit", "_recon_source_winit", "_gt_target_fit"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmr", required=True, help="JSON from run_motionfix_metrics.py.")
    ap.add_argument("--native", action="append", default=[], dest="natives",
                    help="Per-config native_metrics.py JSON (repeat).")
    ap.add_argument("--out_dir", default="eval_results/motionfix")
    args = ap.parse_args()

    tmr = json.load(open(args.tmr))
    native = {}
    for p in args.natives:
        s = json.load(open(p))["summary"]
        native[s["config"]] = s

    rows = []
    for cfg, m in tmr.items():
        full, batch = m.get("full", {}), m.get("batches", {})
        nat = native.get(cfg, {})
        rows.append({
            "config": cfg,
            "is_ceiling": cfg in CEILING_KEYS,
            "n": m.get("n"),
            # target<->generated (the headline edit-fidelity retrieval) + source<->generated
            "R@1": full.get("R@1"), "R@2": full.get("R@2"), "R@3": full.get("R@3"),
            "R@1_s2t": full.get("R@1_s2t"), "R@2_s2t": full.get("R@2_s2t"), "R@3_s2t": full.get("R@3_s2t"),
            "R@1_batch": batch.get("R@1"), "R@1_s2t_batch": batch.get("R@1_s2t"),
            "AED_target": nat.get("aed_target_mean"), "AED_source": nat.get("aed_source_mean"),
        })
    # method configs first (sorted), ceilings last
    rows.sort(key=lambda r: (r["is_ceiling"], r["config"]))

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    cols = [("config", 22), ("n", 5), ("R@1", 6), ("R@2", 6), ("R@3", 6),
            ("R@1_s2t", 8), ("R@3_s2t", 8), ("AED_target", 11), ("AED_source", 11)]
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["# MotionFix-comparable evaluation\n",
             "TMR retrieval (R@k, %, full test set) is computed through the joints->SMPL bridge and is "
             "**bridge-limited**: HumanML3D's IK-derived joints have lost the bone-twist DOF, so the "
             "`_perfect_source` row (=100) shows the format ceiling while `_recon_source*` / `_gt_target` "
             "rows show the achievable bridge ceiling. AED (metres, HumanML3D joint space) is the "
             "representation-faithful metric. R@1/2/3 = target<->generated; `_s2t` = source<->generated.\n",
             "| " + " | ".join(c for c, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        tag = "  _(ceiling)_" if r["is_ceiling"] else ""
        lines.append("| " + " | ".join(fmt(r[c]) for c, _ in cols) + " |" + tag)
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out_dir, "summary.md"), "w").write(md)
    print(md)
    print(f"Wrote {args.out_dir}/summary.json and summary.md")


if __name__ == "__main__":
    main()
