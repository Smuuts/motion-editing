import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HumanML3DDataset(Dataset):
    """
    Loads HumanML3D motion clips and their text annotations.

    Each item returns:
        motion  : (max_frames, 263) float32 tensor, normalised HumanML3D feature vectors
        text    : one randomly sampled annotation string
        length  : actual number of frames T (before padding)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_frames: int = 196,
        min_frames: int = 16,
    ):
        self.data_root = data_root
        self.max_frames = max_frames
        self.min_frames = min_frames

        # normalisation statistics
        self.mean = np.load(os.path.join(data_root, "Mean.npy"))  # (263,)
        self.std  = np.load(os.path.join(data_root, "Std.npy"))   # (263,)

        # clip IDs for this split
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # filter out clips that are too short after loading lengths
        self.ids = self._filter_by_length()

        self.vec_dir  = os.path.join(data_root, "new_joint_vecs")
        self.text_dir = os.path.join(data_root, "texts")

    def _filter_by_length(self):
        valid = []
        for clip_id in self.ids:
            path = os.path.join(self.data_root, "new_joint_vecs", f"{clip_id}.npy")
            if not os.path.exists(path):
                continue
            T = np.load(path, mmap_mode="r").shape[0]
            if T >= self.min_frames:
                valid.append(clip_id)
        return valid

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        clip_id = self.ids[idx]

        # load 263-dim features and normalise — do NOT reconstruct joints
        vecs = np.load(os.path.join(self.vec_dir, f"{clip_id}.npy"))  # (T, 263)
        vecs = (vecs - self.mean) / self.std                           # (T, 263)

        T = min(len(vecs), self.max_frames)
        vecs = vecs[:T]

        pad = np.zeros((self.max_frames, 263), dtype=np.float32)
        pad[:T] = vecs
        motion = torch.from_numpy(pad)  # (max_frames, 263)

        # text
        text_path = os.path.join(self.text_dir, f"{clip_id}.txt")
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        annotations = [l.split("#")[0].strip() for l in lines]
        text = random.choice(annotations)

        return {"motion": motion, "text": text, "length": T, "id": clip_id}


def build_dataloader(data_root, split="train", batch_size=64,
                     max_frames=196, num_workers=4, shuffle=None):
    dataset = HumanML3DDataset(data_root, split=split, max_frames=max_frames)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
