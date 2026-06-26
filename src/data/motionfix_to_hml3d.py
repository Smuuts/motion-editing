"""
Convert the MotionFix dataset (SMPL-H triplets) into HumanML3D 263-dim features so the
GroupDiT/MotionDiT editor can consume it.

MotionFix ships joblib dicts  id -> {motion_source, motion_target, text}, each motion a
dict with `joint_positions` (T, 22, 3) at 30 fps in SMPL-H joint order (== HumanML3D's
22-joint order). For each triplet this script:

    1. resamples joints 30 fps -> 20 fps (HumanML3D's rate; velocity channels depend on it),
    2. runs the HumanML3D forward extraction (see hml3d_features.extract_hml3d_features),
    3. writes RAW (un-normalised) (T-1, 263) .npy for source and target,
    4. writes a manifest .jsonl row {id, source, target, text, ...}.

`text` is the *edit instruction* (the source->target difference), which is what you pass to
edit_motion.py as --instruction. `target` is the ground-truth edited motion, for metrics.

Run it with the MotionFix venv (it has joblib + scipy):
    data/motionfix/mfix-env/bin/python src/data/motionfix_to_hml3d.py \
        --split val --out_dir data/motionfix_hml3d

Then edit a converted source clip with your model (in your normal env):
    python src/edit_motion.py --checkpoint <ckpt> --data_root data/HumanML3D \
        --source data/motionfix_hml3d/val/<id>_source.npy \
        --instruction "<edit text from the manifest>"
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> src/
from data.hml3d_features import extract_hml3d_features, get_tgt_offsets

MOTIONFIX_DIR = "data/motionfix/data/motionfix-dataset"
SPLIT_FILE = {
    "train": "motionfix.pth.tar",
    "val":   "motionfix_val.pth.tar",
    "test":  "motionfix_test.pth.tar",
}
SRC_FPS, TGT_FPS = 30.0, 20.0

# MotionFix stores AMASS-native (Z-up) joints. HumanML3D's process_file assumes Y-up, and
# HumanML3D built its training data from AMASS with exactly this (x, y, z) -> (x, z, y) swap,
# so applying the same swap puts MotionFix in the convention the model was trained on.
# (Verified empirically: the swap brings the joint-position channels to native HumanML3D
# normalised std ~0.7; the proper-rotation alternative leaves them ~5x too large.)
ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]])


def resample(joints, src_fps=SRC_FPS, tgt_fps=TGT_FPS):
    """(T, J, 3) at src_fps -> (T', J, 3) at tgt_fps via per-coordinate linear interp."""
    T = joints.shape[0]
    new_T = max(2, int(round(T * tgt_fps / src_fps)))
    src_t = np.linspace(0.0, 1.0, T)
    tgt_t = np.linspace(0.0, 1.0, new_T)
    J, C = joints.shape[1], joints.shape[2]
    out = np.empty((new_T, J, C), dtype=np.float64)
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(tgt_t, src_t, joints[:, j, c])
    return out


def to_features(joints_30fps, tgt_offsets, return_transform=False):
    joints = np.asarray(joints_30fps, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected (T, 22, 3) joints, got {joints.shape}")
    joints = joints @ ZUP_TO_YUP.T          # Z-up (AMASS) -> Y-up (HumanML3D)
    return extract_hml3d_features(resample(joints), tgt_offsets,
                                  return_transform=return_transform)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=list(SPLIT_FILE))
    ap.add_argument("--out_dir", default="data/motionfix_hml3d")
    ap.add_argument("--data_root", default="data/HumanML3D",
                    help="HumanML3D root, for the reference clip used to size the skeleton.")
    ap.add_argument("--ref_clip", default="000021",
                    help="HumanML3D clip id whose rest pose defines the target skeleton.")
    ap.add_argument("--limit", type=int, default=None, help="Convert only the first N triplets.")
    args = ap.parse_args()

    import joblib  # MotionFix files are joblib dumps despite the .pth.tar extension

    ref = np.load(os.path.join(args.data_root, "new_joints", f"{args.ref_clip}.npy"))
    tgt_offsets = get_tgt_offsets(ref)

    ds_path = os.path.join(MOTIONFIX_DIR, SPLIT_FILE[args.split])
    print(f"Loading {ds_path} …")
    data = joblib.load(ds_path)
    ids = list(data)[: args.limit] if args.limit else list(data)

    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, f"{args.split}.jsonl")

    n_ok, n_skip = 0, 0
    with open(manifest_path, "w") as mf:
        for i, key in enumerate(ids):
            entry = data[key]
            src_jp = np.asarray(entry["motion_source"]["joint_positions"])
            tgt_jp = np.asarray(entry["motion_target"]["joint_positions"])
            try:
                src, src_quat = to_features(src_jp, tgt_offsets, return_transform=True)
                tgt, tgt_quat = to_features(tgt_jp, tgt_offsets, return_transform=True)
            except Exception as e:  # malformed / too-short clip
                print(f"  [skip] {key}: {e}")
                n_skip += 1
                continue
            src_p = os.path.join(out_dir, f"{key}_source.npy")
            tgt_p = os.path.join(out_dir, f"{key}_target.npy")
            np.save(src_p, src)
            np.save(tgt_p, tgt)
            mf.write(json.dumps({
                "id": key,
                "text": entry["text"],
                "source": src_p,
                "target": tgt_p,
                "src_frames": int(src.shape[0]),     # 20 fps (editor rate)
                "tgt_frames": int(tgt.shape[0]),
                # original 30 fps frame counts + the source's canonicalisation quaternion,
                # needed to bring an edited (HumanML3D-canonical) clip back to the dataset
                # frame for SMPL fitting / metrics (see src/eval/joints2smpl_fit.py).
                "src_frames_30fps": int(src_jp.shape[0]),
                "tgt_frames_30fps": int(tgt_jp.shape[0]),
                "src_canon_quat": [float(x) for x in src_quat],
                "tgt_canon_quat": [float(x) for x in tgt_quat],
            }) + "\n")
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ids)} …")

    print(f"Done: {n_ok} converted, {n_skip} skipped.")
    print(f"Features -> {out_dir}/<id>_source.npy, <id>_target.npy")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
