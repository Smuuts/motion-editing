import os
import sys

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import data.dataset as ds
from utils.visualise import recover_from_ric, show_animation


if __name__ == "__main__":
    data_root = "./data/HumanML3D"
    mean = np.load(os.path.join(data_root, "Mean.npy"))
    std  = np.load(os.path.join(data_root, "Std.npy"))

    dataset = ds.HumanML3DDataset(data_root, split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[1]

    print("Sample keys:", sample.keys())
    print("Motion shape:", sample["motion"].shape)
    print("Context shape:", sample["context"].shape)
    print("Length:", sample["length"])
    print("ID:", sample["id"])

    motion_raw = sample["motion"].numpy() * std + mean
    joints = recover_from_ric(motion_raw, joints_num=22)
    print("Joints shape:", joints.shape)

    show_animation(joints, title=sample["id"])
