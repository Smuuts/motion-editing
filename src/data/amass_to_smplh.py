"""
Build a HumanML3D-aligned SMPL-H training set from raw AMASS (Path A, steps 1-3).

For every clip in HumanML3D's `index.csv`, slices the matching AMASS sequence's SMPL-H
rotations (the body 22 joints — `poses[:, :66]`) at the same 20 fps downsample HumanML3D used,
then featurizes to the 135-d MotionFix/TMR representation `[trans_delta | body_pose_6d |
global_orient_6d]` (see smplh_features.py, validated to ~5e-7 against MotionFix's loader). Also
emits the `M`-prefixed left/right mirror so the ids match HumanML3D's split files.

HumanAct12 clips are skipped (no SMPL params). Reuse your existing `texts/` and
`train/val/test.txt` unchanged — the clip ids line up. Runs in the `ma` env.

    python src/data/amass_to_smplh.py \
        --hml3d_root /home/smuuts/Documents/MA/implementation/HumanML3D \
        --out_dir data/HumanML3D_smplh \
        --splits data/HumanML3D/train.txt data/HumanML3D/val.txt data/HumanML3D/test.txt
"""

import os
import csv
import argparse

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> src/
from data.smplh_features import smplh_to_features, mirror_smplh

EX_FPS = 20


def wanted_ids(split_files):
    """Union of base ids (M-prefix stripped) referenced by the given split files, or None."""
    if not split_files:
        return None
    ids = set()
    for sf in split_files:
        for line in open(sf):
            s = line.strip()
            if s:
                ids.add(s[1:] if s.startswith("M") else s)
    return ids


def load_downsampled(npz_path, cache):
    """Load AMASS SMPL-H, downsample to ~20 fps the HumanML3D way; cache by path.
    Returns (rots20 (T,66), trans20 (T,3)) or None if unreadable."""
    if npz_path in cache:
        return cache[npz_path]
    try:
        b = np.load(npz_path)
        fps = float(b["mocap_framerate"])
        down = max(1, int(fps / EX_FPS))               # HumanML3D: int(fps/ex_fps), truncated
        rots = np.asarray(b["poses"][::down, :66], dtype=np.float32)   # 22 body joints
        trans = np.asarray(b["trans"][::down], dtype=np.float32)
        out = (rots, trans)
    except Exception:
        out = None
    cache[npz_path] = out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hml3d_root", required=True, help="HumanML3D repo (index.csv + amass_data/).")
    ap.add_argument("--out_dir", default="data/HumanML3D_smplh")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Restrict to ids in these split files (default: all index.csv rows).")
    ap.add_argument("--no_mirror", action="store_true", help="Skip the M-prefixed mirror clips.")
    ap.add_argument("--save_raw", action="store_true",
                    help="Also save raw {rots,trans} alongside the 135-d feature.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_raw:
        os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    keep = wanted_ids(args.splits)

    rows = list(csv.DictReader(open(os.path.join(args.hml3d_root, "index.csv"))))
    if args.limit:
        rows = rows[: args.limit]

    cache = {}
    n_ok = n_skip_ha = n_skip_bad = 0
    for r in tqdm(rows, desc="clips"):
        base = r["new_name"][:-4] if r["new_name"].endswith(".npy") else r["new_name"]
        if keep is not None and base not in keep:
            continue
        src = r["source_path"]
        if src.startswith("./pose_data/humanact12"):       # no SMPL params
            n_skip_ha += 1
            continue
        npz = os.path.join(args.hml3d_root, "amass_data", src[len("./pose_data/"):-4] + ".npz")

        out_p = os.path.join(args.out_dir, f"{base}.npy")
        m_out_p = os.path.join(args.out_dir, f"M{base}.npy")
        if not args.overwrite and os.path.exists(out_p) and (args.no_mirror or os.path.exists(m_out_p)):
            continue

        ds = load_downsampled(npz, cache)
        if ds is None:
            n_skip_bad += 1
            continue
        rots20, trans20 = ds
        s, e = int(r["start_frame"]), int(r["end_frame"])
        e = len(rots20) if e == -1 else e
        rots, trans = rots20[s:e], trans20[s:e]
        if len(rots) < 2:
            n_skip_bad += 1
            continue

        np.save(out_p, smplh_to_features(rots, trans))
        if args.save_raw:
            np.save(os.path.join(args.out_dir, "raw", f"{base}.npy"), {"rots": rots, "trans": trans})
        if not args.no_mirror:
            mr, mt = mirror_smplh(rots, trans)
            np.save(m_out_p, smplh_to_features(mr, mt))
            if args.save_raw:
                np.save(os.path.join(args.out_dir, "raw", f"M{base}.npy"), {"rots": mr, "trans": mt})
        n_ok += 1

    print(f"Done: {n_ok} clips ({'no mirror' if args.no_mirror else '×2 with mirror'}) -> {args.out_dir}")
    print(f"Skipped: {n_skip_ha} HumanAct12 (no SMPL), {n_skip_bad} unreadable/too-short.")
    print("Next: compute stats →  python src/data/smplh_stats.py "
          f"--feat_dir {args.out_dir} --split data/HumanML3D/train.txt")


if __name__ == "__main__":
    main()
