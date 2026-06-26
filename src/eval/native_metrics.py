"""
Stage E of the MotionFix evaluation: a TMR-free, HumanML3D-native sanity metric, computed
directly in joint space where both the edit and the GT live (no SMPL fit, no learned encoder).

It cross-checks the SMPL-space TMR numbers: if a config has high TMR R@k but poor joint-space
agreement with the target, that gap is fitting/encoder noise rather than genuine edit quality.

For each clip we decode the edited and the GT-target HumanML3D features to 22 joints
(recover_from_ric), align them at the root over XZ (translation is not part of the edit signal),
and report:
  AED-target : mean per-joint Euclidean distance to the GT target  (lower = closer to the edit goal)
  AED-source : mean per-joint Euclidean distance to the source       (edit magnitude / how much moved)

Both motions are at the editor's 20 fps; target is truncated/interpolated to the edited length.
Runs in either env (numpy only). Example:
    python src/eval/native_metrics.py --manifest data/motionfix_hml3d/test.jsonl \
        --edited_dir data/motionfix_edited/m2_only_s5 --out eval_results/motionfix/native_m2_only_s5.json
"""

import os
import sys
import json
import argparse

import numpy as np

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.visualise import recover_from_ric


def _resample_len(arr, L):
    """Linear-interpolate (T, ...) -> (L, ...) along time."""
    T = arr.shape[0]
    if T == L:
        return arr
    src_t, tgt_t = np.linspace(0, 1, T), np.linspace(0, 1, L)
    flat = arr.reshape(T, -1)
    out = np.stack([np.interp(tgt_t, src_t, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape(L, *arr.shape[1:])


def _root_xz_align(joints):
    """Subtract per-frame root XZ so the metric ignores absolute ground translation."""
    j = joints.copy()
    j[..., [0, 2]] -= j[:, 0:1, [0, 2]]
    return j


def aed(a_joints, b_joints):
    """Mean per-joint Euclidean distance (metres) between two (T,22,3) motions, root-XZ aligned."""
    L = min(len(a_joints), len(b_joints))
    a = _root_xz_align(_resample_len(a_joints, L))
    b = _root_xz_align(_resample_len(b_joints, L))
    return float(np.sqrt(((a - b) ** 2).sum(-1)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/motionfix_hml3d/test.jsonl")
    ap.add_argument("--edited_dir", required=True, help="A Stage-B config dir of edited (F,263) .npy.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    per_clip, aed_t, aed_s = {}, [], []
    for row in rows:
        cid = row["id"]
        ep = os.path.join(args.edited_dir, f"{cid}.npy")
        if not os.path.exists(ep):
            continue
        edited = recover_from_ric(np.load(ep).astype(np.float32), 22)
        target = recover_from_ric(np.load(row["target"]).astype(np.float32), 22)
        source = recover_from_ric(np.load(row["source"]).astype(np.float32), 22)
        at, as_ = aed(edited, target), aed(edited, source)
        per_clip[cid] = {"aed_target": at, "aed_source": as_}
        aed_t.append(at)
        aed_s.append(as_)

    summary = {
        "config": os.path.basename(args.edited_dir.rstrip("/")),
        "n": len(aed_t),
        "aed_target_mean": float(np.mean(aed_t)) if aed_t else None,
        "aed_target_median": float(np.median(aed_t)) if aed_t else None,
        "aed_source_mean": float(np.mean(aed_s)) if aed_s else None,
    }
    print(summary)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
