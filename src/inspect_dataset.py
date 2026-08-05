"""
Eyeball one clip straight out of the dataloader: print the sample's fields, decode it
and play the skeleton interactively (blocking window — needs a display).

A hand-run sanity check, not a test — it was called `test_dataset.py`, which pytest
would have collected.

    python src/inspect_dataset.py [--data_root ...] [--split train] [--index 1]
"""

import argparse
import os

import numpy as np

from data.dataset import HumanML3DDataset
from utils.decode import recover_from_ric
from utils.visualise import show_animation


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./data/HumanML3D/HumanML3D")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--no_animation", action="store_true",
                   help="Print the sample's fields only (no display needed).")
    args = p.parse_args()

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))

    dataset = HumanML3DDataset(args.data_root, split=args.split)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[args.index]

    print("Sample keys:", list(sample))
    print("Motion shape:", sample["motion"].shape)
    # "context" is present only when a precomputed text_emb/ exists; otherwise "text".
    print("Context shape:", sample["context"].shape) if "context" in sample \
        else print("Text:", sample["text"])
    print("Length:", sample["length"])
    print("ID:", sample["id"])

    joints = recover_from_ric(sample["motion"].numpy() * std + mean, joints_num=22)
    print("Joints shape:", joints.shape)
    if not args.no_animation:
        show_animation(joints, title=sample["id"])


if __name__ == "__main__":
    main()
