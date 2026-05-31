import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Channels extracted from the 263-dim HumanML3D vector for SMPL mode (130 dims):
#   [0:4]    root_rot_vel + root_XZ_vel + root_y  (velocity-based root)
#   [67:193] body_pose 6D for 21 non-root joints  (rot_data)
_SMPL_CHANNELS = np.array(list(range(4)) + list(range(67, 193)))


class HumanML3DDataset(Dataset):
    """
    Loads HumanML3D motion clips and their text annotations.

    Each item returns:
        motion  : (max_frames, D) float32 tensor, normalised feature vectors
        text    : one randomly sampled annotation string
        length  : actual number of frames T (before padding)

    feature_mode:
        "humanml3d" — full 263-dim HumanML3D feature vector
        "group"     — 130-dim subset: root velocity/height + body pose 6D (21 joints)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_frames: int = 196,
        min_frames: int = 16,
        feature_mode: str = "humanml3d",
    ):
        assert feature_mode in ("humanml3d", "group"), f"Unknown feature_mode: {feature_mode}"
        self.data_root = data_root
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.feature_mode = feature_mode

        # normalisation statistics — slice to the active channels for SMPL/joint mode
        mean = np.load(os.path.join(data_root, "Mean.npy"))  # (263,)
        std  = np.load(os.path.join(data_root, "Std.npy"))   # (263,)
        if feature_mode == "group":
            self.mean = mean[_SMPL_CHANNELS]  # (130,)
            self.std  = std[_SMPL_CHANNELS]   # (130,)
            self.feature_dim = len(_SMPL_CHANNELS)
        else:
            self.mean = mean
            self.std  = std
            self.feature_dim = 263

        # clip IDs for this split
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # filter out clips that are too short after loading lengths
        self.ids = self._filter_by_length()

        self.vec_dir      = os.path.join(data_root, "new_joint_vecs")
        self.text_dir     = os.path.join(data_root, "texts")
        text_emb_dir      = os.path.join(data_root, "text_emb")
        self.text_emb_dir = text_emb_dir if os.path.isdir(text_emb_dir) else None

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

        vecs = np.load(os.path.join(self.vec_dir, f"{clip_id}.npy"))  # (T, 263)
        if self.feature_mode == "group":
            vecs = vecs[:, _SMPL_CHANNELS]                            # (T, 130)
        # LEDITS++ inversion operates in this normalised space — the model was trained
        # on normalised features, so x0 passed to invert() must also be normalised.
        # Keep the raw (pre-normalisation) version if you need FK evaluation later:
        #   raw = vecs.copy(); vecs = (vecs - self.mean) / self.std
        vecs = (vecs - self.mean) / self.std

        T = min(len(vecs), self.max_frames)
        vecs = vecs[:T]

        pad = np.zeros((self.max_frames, self.feature_dim), dtype=np.float32)
        pad[:T] = vecs
        motion = torch.from_numpy(pad)  # (max_frames, feature_dim)
        # LEDITS++ editing note: `length` (= T) marks the valid frame range.
        # Mask columns in M beyond index T are always zero — those padding frames
        # must never be edited and must be excluded from mask thresholding percentiles.

        # text — use precomputed CLIP embeddings when available
        if self.text_emb_dir is not None:
            emb_path = os.path.join(self.text_emb_dir, f"{clip_id}.npy")
            emb = np.load(emb_path)                          # (num_ann, 77, dim) float16
            idx = random.randrange(len(emb))
            context = torch.from_numpy(emb[idx].astype(np.float32))  # (77, dim)
            return {"motion": motion, "context": context, "length": T, "id": clip_id}

        text_path = os.path.join(self.text_dir, f"{clip_id}.txt")
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        annotations = [l.split("#")[0].strip() for l in lines]
        text = random.choice(annotations)
        return {"motion": motion, "text": text, "length": T, "id": clip_id}


def build_dataloader(data_root, split="train", batch_size=64,
                     max_frames=196, num_workers=4, shuffle=None,
                     feature_mode="humanml3d"):
    dataset = HumanML3DDataset(data_root, split=split, max_frames=max_frames,
                               feature_mode=feature_mode)
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
