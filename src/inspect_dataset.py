"""
Eyeball one clip straight out of the dataloader: print the sample's fields, decode it
and play it back interactively (blocking window — needs a display), as a skeleton or,
with `--feature_mode smplh --render mesh`, as the posed SMPL body.

A hand-run sanity check, not a test — it was called `test_dataset.py`, which pytest
would have collected.

    python src/inspect_dataset.py [--data_root ...] [--split train] [--index 1]
    python src/inspect_dataset.py --data_root .../HumanML3D_smplh --feature_mode smplh \
        --render mesh
"""

import argparse
import os

import numpy as np

from data.dataset import HumanML3DDataset
from utils.decode import recover_joints, smplh_body_model
from utils.logger import get_logger
from utils.visualise import show_animation
from utils.cli import (add_logging_args, add_render_args, configure_logging,
                       render_vertices)

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./data/HumanML3D/HumanML3D")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--feature_mode", default="humanml3d", choices=["humanml3d", "smplh"],
                   help="Representation stored in --data_root: 'humanml3d' (263-d, the "
                        "default) or 'smplh' (135-d). Must match the data — the decode "
                        "reads the width it expects, so a mismatch is a shape error.")
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh",
                   help="--feature_mode smplh: SMPLHLayer dir (SMPLH_NEUTRAL.npz).")
    p.add_argument("--no_animation", action="store_true",
                   help="Print the sample's fields only (no display needed).")
    add_render_args(p)
    add_logging_args(p)
    args = configure_logging(p.parse_args())

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))

    if args.feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)

    dataset = HumanML3DDataset(args.data_root, split=args.split,
                               feature_mode=args.feature_mode)
    log.info(f"Dataset size: {len(dataset)}")
    sample = dataset[args.index]

    # "context" is present only when a precomputed text_emb/ exists; otherwise "text".
    text_field = ("context shape", tuple(sample["context"].shape)) if "context" in sample \
        else ("text", sample["text"])
    log.kv({"sample keys": list(sample),
            "motion shape": tuple(sample["motion"].shape),
            text_field[0]: text_field[1],
            "length": sample["length"],
            "id": sample["id"]})

    raw = sample["motion"].numpy() * std + mean
    joints = recover_joints(raw, args.feature_mode)
    log.info("joints shape: %s", tuple(joints.shape))
    if not args.no_animation:
        show_animation(joints, title=sample["id"],
                       vertices=render_vertices(raw, args.feature_mode, args),
                       skeleton_overlay=args.mesh_overlay, mesh_size=args.mesh_size)


if __name__ == "__main__":
    main()
