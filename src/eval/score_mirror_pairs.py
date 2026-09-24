"""
Score mirror-pair edits (from edit_mirror_pairs.py) against their EXACT mirrored targets.

Every number is a joint-position distance in millimetres, after SMPL-H forward kinematics,
frame-aligned (a mirror twin has the source's length). Per edit:

    d_src  = MPJPE(source, target)   the do-nothing baseline: how far the source is from the answer
    d_edit = MPJPE(edit,   target)   how far the edit is from the answer
    move   = MPJPE(edit,   source)   how much the editor changed anything at all

Headline per (mask_mode, scale):
    gap_closed = 1 - Σ d_edit / Σ d_src     share of the source->target distance the edits
                                            removed (0 = did nothing, 1 = exact mirror,
                                            negative = moved AWAY from the target)
    win_rate   = P(d_edit < d_src - 0.01mm)  with a Wilson 95 % CI
    Δ          = mean(d_src - d_edit)       with its paired SE and t

Each is reported twice: `global` (world joints — includes the mirrored root trajectory) and
`local` (joints relative to the pelvis in each frame — pose only). And split by region: joints
of the body-part groups the caption pair names (`in`) vs every other joint (`out`). A part-
local edit can only ever close the `in` gap; the `out` gap is the part of the mirror no
instruction asked for, and a hard-inpainting mask leaves it at the source by construction.

`oracle_floor` = MPJPE(reflect(source joints), target): the residual of an exact geometric
mirror, i.e. the body model's own asymmetry. The target is the ground truth, so this is only a
sanity check that the pairs really are mirrors — it should be a few mm at most.

Usage:
    python src/eval/score_mirror_pairs.py --out_root eval_results/mirror_pairs/run
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import glob
import json
import re

import numpy as np

from data.smplh_features import _MIRROR_PERM
from model.body_groups import BODY_PART_GROUPS, GROUP_NAMES
from utils.cli import add_logging_args, configure_logging
from utils.decode import recover_joints, smplh_body_model
from utils.logger import get_logger
from utils.probe import wilson_ci

log = get_logger(__name__)

# 22-joint indices per group: root is SMPL joint 0; BODY_PART_GROUPS index the 21-joint body.
GROUP_JOINTS = {"root": [0], **{n: [j + 1 for j in js] for n, js in BODY_PART_GROUPS}}
assert set(GROUP_JOINTS) == set(GROUP_NAMES)
WIN_TOL_MM = 0.01


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True, help="edit_mirror_pairs.py --out_root.")
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D_smplh")
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--out", default=None, help="Summary JSON (default <out_root>/scores.json).")
    add_logging_args(p)
    return configure_logging(p.parse_args())


def mpjpe(a, b, joints=None):
    """Mean per-joint position error in mm over frames × joints (optionally a joint subset)."""
    if joints is not None:
        a, b = a[:, joints], b[:, joints]
    return float(np.linalg.norm(a - b, axis=-1).mean() * 1000.0)


def local(j):
    return j - j[:, :1]


def reflect(j):
    """Sagittal mirror of Y-up joints: negate x, swap left/right joints."""
    m = j[:, _MIRROR_PERM].copy()
    m[..., 0] *= -1
    return m


def summarise(d_src, d_edit):
    d_src, d_edit = np.asarray(d_src), np.asarray(d_edit)
    n = len(d_src)
    if n == 0:
        return None
    delta = d_src - d_edit
    # A win must beat the source by more than float noise, or scale 0 (an exact copy)
    # scores ~50 % on rounding alone.
    wins = int((delta > WIN_TOL_MM).sum())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    lo, hi = wilson_ci(wins, n)
    return {
        "n": n,
        "d_src": float(d_src.mean()), "d_edit": float(d_edit.mean()),
        "gap_closed": float(1.0 - d_edit.sum() / d_src.sum()),
        "delta": float(delta.mean()), "delta_se": se,
        "t": float(delta.mean() / se) if se and se > 0 else float("nan"),
        "win_rate": wins / n, "win_ci95": [max(0.0, lo), min(1.0, hi)],
    }


def main():
    args = parse_args()
    smplh_body_model(args.smplh_model_path)
    with open(os.path.join(args.out_root, "pairs.json")) as f:
        meta = json.load(f)

    # source id -> (target id, routed groups, direction)
    edits = {}
    for p in meta["pairs"]:
        edits[p["base"]] = (p["mirror"], p["groups"], "forward")
        edits[p["mirror"]] = (p["base"], p["groups"], "reverse")

    cache = {}

    def joints(clip_id):
        """Dataset clip -> (T, 22, 3) joints, decoded once."""
        if clip_id not in cache:
            feat = np.load(os.path.join(args.data_root, "new_joint_vecs", f"{clip_id}.npy"))
            cache[clip_id] = recover_joints(feat[:args.max_frames].astype(np.float32), "smplh")
        return cache[clip_id]

    # Sanity: are the pairs really mirrors? (computed once, over every scored pair)
    floor = []
    for p in meta["pairs"]:
        floor.append(mpjpe(reflect(joints(p["base"])), joints(p["mirror"])))

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.out_root, "*_s*"))
                      if os.path.isdir(d))
    scale_of = lambda d: float(re.search(r"_s([-\d.]+)$", d).group(1))
    results = {}
    for d in sorted(run_dirs, key=lambda d: (os.path.basename(d).rsplit("_s", 1)[0], scale_of(d))):
        name = os.path.basename(d)
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            src_id = os.path.splitext(os.path.basename(f))[0]
            if src_id not in edits:
                continue
            tgt_id, groups, direction = edits[src_id]
            J_s, J_t = joints(src_id), joints(tgt_id)
            J_e = recover_joints(np.load(f).astype(np.float32), "smplh")
            in_j = sorted({j for g in groups for j in GROUP_JOINTS.get(g, [])})
            out_j = [j for j in range(22) if j not in in_j]
            row = {"id": src_id, "direction": direction, "has_groups": bool(in_j)}
            for space, tf in (("global", lambda x: x), ("local", local)):
                s, t, e = tf(J_s), tf(J_t), tf(J_e)
                row[f"{space}_d_src"] = mpjpe(s, t)
                row[f"{space}_d_edit"] = mpjpe(e, t)
                row[f"{space}_move"] = mpjpe(e, s)
                if in_j:
                    for reg, js in (("in", in_j), ("out", out_j)):
                        row[f"{space}_{reg}_d_src"] = mpjpe(s, t, js)
                        row[f"{space}_{reg}_d_edit"] = mpjpe(e, t, js)
            rows.append(row)
        if not rows:
            continue

        def block(sel):
            out = {}
            for space in ("global", "local"):
                rs = [r for r in rows if sel(r)]
                out[space] = summarise([r[f"{space}_d_src"] for r in rs],
                                       [r[f"{space}_d_edit"] for r in rs])
                if out[space]:
                    out[space]["move"] = float(np.mean([r[f"{space}_move"] for r in rs]))
                rg = [r for r in rs if r["has_groups"]]
                for reg in ("in", "out"):
                    out[f"{space}_{reg}"] = summarise(
                        [r[f"{space}_{reg}_d_src"] for r in rg],
                        [r[f"{space}_{reg}_d_edit"] for r in rg])
            return out

        results[name] = {
            "n": len(rows),
            "all": block(lambda r: True),
            "forward": block(lambda r: r["direction"] == "forward"),
            "reverse": block(lambda r: r["direction"] == "reverse"),
        }
        with open(os.path.join(d, "per_edit.json"), "w") as f:
            json.dump(rows, f)

    summary = {
        "out_root": os.path.abspath(args.out_root),
        "split": meta["split"], "caption_line": meta["caption_line"],
        "n_pairs": len(meta["pairs"]),
        "n_groups_routed": sum(bool(p["groups"]) for p in meta["pairs"]),
        "oracle_floor_mm": {"mean": float(np.mean(floor)), "max": float(np.max(floor))},
        "runs": results,
    }
    out = args.out or os.path.join(args.out_root, "scores.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"{summary['n_pairs']} pairs ({summary['n_groups_routed']} with routed groups); "
             f"mirror sanity floor {summary['oracle_floor_mm']['mean']:.2f} mm mean, "
             f"{summary['oracle_floor_mm']['max']:.2f} max")
    hdr = (f"{'run':<22}{'n':>6}{'space':>8}{'d_src':>8}{'d_edit':>8}{'move':>7}"
           f"{'gap%':>7}{'Δ±SE':>14}{'win%':>7}{'in gap%':>9}{'out gap%':>9}")
    log.info(hdr)
    for name, r in results.items():
        for space in ("global", "local"):
            b = r["all"][space]
            gi, go = r["all"][f"{space}_in"], r["all"][f"{space}_out"]
            log.info(f"{name:<22}{r['n']:>6}{space:>8}{b['d_src']:>8.1f}{b['d_edit']:>8.1f}"
                     f"{b['move']:>7.1f}{100 * b['gap_closed']:>7.2f}"
                     f"{b['delta']:>8.2f}±{b['delta_se']:<5.2f}{100 * b['win_rate']:>7.1f}"
                     f"{(100 * gi['gap_closed'] if gi else float('nan')):>9.2f}"
                     f"{(100 * go['gap_closed'] if go else float('nan')):>9.2f}")
    log.info(f"-> {out}")


if __name__ == "__main__":
    main()
