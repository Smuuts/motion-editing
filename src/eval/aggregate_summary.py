"""
Format the MotionFix-comparable TMR retrieval results (from run_motionfix_metrics.py) into a
readable table: one row per edit config (guidance scale).

Metrics (MotionFix benchmark standard, higher is better, %):
  R@1/2/3       — target <-> generated : instruction following (edit fidelity)
  R@1/2/3_s2t   — source <-> generated : motion preservation

The `*_s0` (identity) row is the plumbing calibration: R@1_s2t should be ~100.

**Which column to lead with depends on coverage.** MotionFix's `retrieval()` restricts its
retrieval set to the keyids you hand it, so the gallery-wide R@k of a SUBSAMPLED run is an
N-way retrieval that reads systematically higher than the published 1013-way protocol, and two
configs with different coverage cannot be compared on it at all — there, batches-of-32 (32-way
whatever N is) is the only comparable column. On a FULL 1013-clip run both are comparable,
because the published MotionFix table reports both protocols, and the gallery column is then
the better one to lead with: the 32-way protocol is nearly saturated by copying the source
unchanged (do-nothing ~73.6 vs a supervised SOTA of 77.3, so ~3.7 pp of range), while the full
protocol has ~14.5 pp. This script flags the subsampled and mixed-coverage cases from `n`.

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

# Duplicated in run_motionfix_metrics.py, which runs in MotionFix's venv and so cannot share
# a module with this one.
FULL_TEST_SET = 1013


def _scale_key(cfg):
    """Sort key for a config dir name: (mask_mode, scale), e.g. m2_only_s2.5 -> ("m2_only", 2.5).

    Grouped by MODE first, then by scale within it. Sorting on scale alone interleaved the
    modes, which became unreadable once each mode got its own scale list — the modes need
    ~24x different scales to reach the same edit magnitude, so their sweeps barely overlap.
    """
    m = re.search(r"^(.*)_s(-?\d+(?:\.\d+)?)$", cfg)
    return (m.group(1), 0, float(m.group(2))) if m else (cfg, 1, 0.0)


def _num(v):
    """Coerce a metric to float, tolerating None.

    run_motionfix_metrics.py now coerces at the evaluator boundary, so a freshly written
    tmr_metrics.json already holds numbers. This stays because MotionFix returns its metrics
    as STRINGS ('71.88' — `line2dict` splits a formatted LaTeX row) and any JSON written
    before that fix still carries them, so reading an older file must keep working."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
            "config": cfg,
            "n": m.get("n"),
            # headline: 32-way, comparable across configs and against the paper
            "R@1_b": _num(batch.get("R@1")), "R@2_b": _num(batch.get("R@2")),
            "R@3_b": _num(batch.get("R@3")), "R@1_s2t_b": _num(batch.get("R@1_s2t")),
            # secondary: N-way, where N is this config's own file count
            "R@1_g": _num(full.get("R@1")), "R@3_g": _num(full.get("R@3")),
            "R@1_s2t_g": _num(full.get("R@1_s2t")),
        })
    rows.sort(key=lambda r: _scale_key(r["config"]))

    galleries = {r["n"] for r in rows if r["n"] is not None}
    subsampled = any(n < FULL_TEST_SET for n in galleries)
    # Under --common_subset every dir is scored on the same keyids, so unequal sizes are
    # already impossible there and this cannot fire alongside the on_subset note below.
    mixed = len(galleries) > 1
    on_subset = any(m.get("common_subset") for m in tmr.values())

    # Flag the best on the 32-way protocol: it is the one column that is comparable no matter
    # how many clips a config generated, so the flag never rewards a config for having skipped
    # clips. On a FULL run the _g column is the more informative one to read (see the header),
    # but it is not safe to RANK on, because coverage differences would confound it.
    best = max((r for r in rows if r["R@1_b"] is not None),
               key=lambda r: r["R@1_b"], default=None)

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    cols = ["config", "n", "R@1_b", "R@2_b", "R@3_b", "R@1_s2t_b",
            "R@1_g", "R@3_g", "R@1_s2t_g"]

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))

    lines = [
        "# MotionFix TMR retrieval (SMPL-H editor)\n",
        "`R@k` = target<->generated (**instruction following**); `R@k_s2t` = "
        "source<->generated (**motion preservation**). Higher is better; there is a "
        "preservation/edit trade-off across scales. The `_s0` row is the identity "
        "calibration (R@1_s2t should be ~100).\n",
        "**`_b` = batches-of-32**, 32-way retrieval whatever the run size. **`_g` = whole "
        "gallery**, an `n`-way retrieval where `n` is that config's own clip count. The published "
        "MotionFix table reports BOTH protocols, so at the full 1013 clips both columns are "
        "comparable — and `_g` is the more informative one, because the 32-way protocol is nearly "
        "saturated by copying the source (do-nothing scores ~73.6 against a supervised SOTA of "
        "77.3, vs ~17.6 against 32.0 on the full set). Lead with `_g` on a full run; `_b` is the "
        "only comparable column on a subsampled one.\n",
    ]
    if subsampled:
        lines.append(
            f"> ⚠ **The `_g` columns are not publication-comparable here.** At least one "
            f"config was scored on fewer than the full {FULL_TEST_SET} test clips, and "
            f"retrieval gets easier as the gallery shrinks, so those values read "
            f"systematically high. Quote the `_b` columns.\n")
    if mixed:
        lines.append(
            f"> ⚠ **Configs have different gallery sizes ({sorted(galleries)}), so the `_g` "
            f"columns cannot be compared to each other either** — the config with fewer "
            f"clips faces an easier retrieval task. Re-score with `--common_subset` for a "
            f"like-for-like table.\n")
    if on_subset:
        lines.append(
            "> Scored with `--common_subset`: every config was restricted to the keyids all "
            "of them share, so the galleries are identical in size and membership.\n")
    lines += [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        tag = "  **best R@1 (32-way)**" if best is not None and r["config"] == best["config"] else ""
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |" + tag)
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out_dir, "summary.md"), "w").write(md)
    print(md)
    print(f"Wrote {args.out_dir}/summary.json and summary.md")


if __name__ == "__main__":
    main()
