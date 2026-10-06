"""
Mean ± std over inversion seeds of the MotionFix scores written by run_motionfix_metrics.py.

Reads one `<prefix>_seed<S>_tmr.json` per seed (eval_motionfix_seeds.sh writes them), and for
every configuration (`<mask_mode>_s<scale>`) present in at least two seeds reports the mean, the
sample standard deviation (ddof = 1) and the number of seeds, for the columns of the thesis'
tab:motionfix plus the paired directional metrics:

    batches-of-32 (32-way)   R@1 R@2 R@3 AvgR            and R@1_s2t (preservation)
    full test set (N-way)    R@1 R@2 R@3 AvgR            and R@1_s2t
    per clip                 PIR (share of clips the edit moved closer to the target), dAvgR

Configurations present in only one seed (scale 0, written by one seed as the do-nothing row) are
listed with their single values. Every seed must have scored the same number of clips, or the
gallery sizes differ and the full-test columns are not comparable — that is checked, not assumed.

Writes <out_dir>/seeds.json, seeds.md and seeds_rows.tex (rows in tab:motionfix's column order).

    python src/eval/aggregate_seeds.py \
        --glob "eval_results/motionfix/<TAG>_seed*_tmr.json" --out_dir eval_results/motionfix/<TAG>_seeds
"""

import argparse
import glob
import json
import math
import os
import re
import sys

COLUMNS = [  # (key in the output, block in the tmr json, metric name)
    ("R@1_b", "batches", "R@1"), ("R@2_b", "batches", "R@2"), ("R@3_b", "batches", "R@3"),
    ("AvgR_b", "batches", "AvgR"),
    ("R@1_g", "full", "R@1"), ("R@2_g", "full", "R@2"), ("R@3_g", "full", "R@3"),
    ("AvgR_g", "full", "AvgR"),
    ("R@1_s2t_b", "batches", "R@1_s2t"), ("R@1_s2t_g", "full", "R@1_s2t"),
]
TABLE = ["R@1_b", "R@2_b", "R@3_b", "AvgR_b", "R@1_g", "R@2_g", "R@3_g", "AvgR_g", "R@1_s2t_b"]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _scale(cfg):
    m = re.search(r"_s([0-9.]+)$", cfg)
    return float(m.group(1)) if m else float("inf")


def _values(entry):
    out = {key: _num(entry.get(block, {}).get(name)) for key, block, name in COLUMNS}
    d = entry.get("directional") or {}
    out["PIR"] = _num(d.get("pir_sim"))
    src, gen = _num(d.get("avgr_src")), _num(d.get("avgr_gen"))
    out["dAvgR"] = None if src is None or gen is None else src - gen
    return out


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    std = (math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
           if len(xs) > 1 else None)
    return {"mean": mean, "std": std, "n": len(xs), "values": xs}


def _fmt(st, digits=2):
    if st is None:
        return "—"
    if st["std"] is None:
        return f"{st['mean']:.{digits}f}"
    return f"{st['mean']:.{digits}f} ± {st['std']:.{digits}f}"


def _tex(st, digits=2):
    if st is None:
        return "---"
    if st["std"] is None:
        return f"${st['mean']:.{digits}f}$"
    return f"${st['mean']:.{digits}f}{{\\scriptstyle\\,\\pm {st['std']:.{digits}f}}}$"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True, help="Pattern matching the per-seed _tmr.json files.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    seeds = {}
    for f in files:
        m = re.search(r"_seed(\d+)_tmr\.json$", f)
        if m:
            seeds[int(m.group(1))] = json.load(open(f))
    if not seeds:
        sys.exit(f"no files match {args.glob!r}")

    # Gallery sizes must agree, or the full-test columns of different seeds are different tasks.
    sizes = {s: {cfg: e.get("n") for cfg, e in d.items()} for s, d in seeds.items()}
    all_n = {n for per in sizes.values() for n in per.values()}
    if len(all_n) > 1:
        print(f"WARNING: seeds scored different clip counts {sorted(all_n)} — the full-test "
              f"columns are not comparable across them. Per seed: {sizes}")

    configs = sorted({cfg for d in seeds.values() for cfg in d}, key=_scale)
    report = {"seeds": sorted(seeds), "files": files, "configs": {}}
    for cfg in configs:
        present = [s for s in sorted(seeds) if cfg in seeds[s]]
        vals = [_values(seeds[s][cfg]) for s in present]
        keys = [k for k, _, _ in COLUMNS] + ["PIR", "dAvgR"]
        report["configs"][cfg] = {
            "seeds": present,
            "n_clips": sorted({seeds[s][cfg].get("n") for s in present}),
            **{k: _stats([v[k] for v in vals]) for k in keys},
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "seeds.json"), "w") as f:
        json.dump(report, f, indent=2)

    head = ["config", "seeds"] + TABLE + ["R@1_s2t_g", "PIR", "dAvgR"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cfg, r in report["configs"].items():
        lines.append("| " + " | ".join(
            [cfg, str(len(r["seeds"]))] + [_fmt(r[k]) for k in TABLE + ["R@1_s2t_g", "PIR", "dAvgR"]]
        ) + " |")
    md = ("# MotionFix over inversion seeds\n\n"
          f"Seeds: {', '.join(map(str, sorted(seeds)))}. Mean ± sample std (ddof = 1) over the "
          "seeds a configuration appears in; a single value means one seed (the scale-0 "
          "do-nothing row).\n`_b` = batches of 32 (32-way), `_g` = the full test set. "
          "AvgR is a rank, lower is better. PIR = share of clips the edit moved closer to "
          "the target than the unedited source (chance 50).\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "seeds.md"), "w") as f:
        f.write(md)

    tex = ["% columns as in tab:motionfix: batch R@1 R@2 R@3 AvgR | test R@1 R@2 R@3 AvgR | "
           "source R@1 (batch)"]
    for cfg, r in report["configs"].items():
        s = _scale(cfg)
        label = "\\emph{Do nothing} (source)" if s == 0 else f"Ours, $s_e{{=}}{s:g}$"
        tex.append(f"{label} & none & " + " & ".join(_tex(r[k]) for k in TABLE) + " \\\\")
    with open(os.path.join(args.out_dir, "seeds_rows.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    print(md)
    print(f"wrote {args.out_dir}/seeds.json, seeds.md, seeds_rows.tex")


if __name__ == "__main__":
    main()
