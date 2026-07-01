"""
Step 4: per-channel Mean/Std over the SMPL-H training features (twin of HumanML3D's
cal_mean_variance), for your model's input normalization.

Computes statistics over all frames of all train-split clips, streaming so it doesn't load
everything at once. Writes Mean.npy / Std.npy (135,) into the feature dir.

    python src/data/smplh_stats.py --feat_dir data/HumanML3D_smplh --split data/HumanML3D/train.txt
"""

import os
import argparse

import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True,
                    help="smplh dataset dir from amass_to_smplh.py (contains new_joint_vecs/). "
                         "Mean.npy/Std.npy are written to its root.")
    ap.add_argument("--split", required=True, help="train.txt — ids to compute stats over.")
    ap.add_argument("--eps", type=float, default=1e-8, help="Std floor (avoid divide-by-zero channels).")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.split) if l.strip()]
    # Vectorised running sum / sum-of-squares over frames (float64), one load per clip.
    n = 0
    s = ssq = None
    missing = 0
    for cid in tqdm(ids, desc="stats"):
        p = os.path.join(args.feat_dir, "new_joint_vecs", f"{cid}.npy")
        if not os.path.exists(p):
            missing += 1
            continue
        x = np.load(p).astype(np.float64)              # (T,135)
        if s is None:
            s = np.zeros(x.shape[1]); ssq = np.zeros(x.shape[1])
        n += x.shape[0]
        s += x.sum(0)
        ssq += (x * x).sum(0)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(ssq / max(n, 1) - mean ** 2, 0.0))
    std = np.maximum(std, args.eps)

    np.save(os.path.join(args.feat_dir, "Mean.npy"), mean.astype(np.float32))
    np.save(os.path.join(args.feat_dir, "Std.npy"), std.astype(np.float32))
    print(f"Frames: {n} from {len(ids) - missing}/{len(ids)} clips ({missing} missing).")
    print(f"Wrote {args.feat_dir}/Mean.npy and Std.npy  (shape {mean.shape})")


if __name__ == "__main__":
    main()
