import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HumanML3DDataset(Dataset):
    """
    Loads HumanML3D motion clips and their text annotations.

    Each item returns:
        motion  : (max_frames, 263) float32 tensor, normalised feature vectors
        text    : one randomly sampled annotation string
        length  : actual number of frames T (before padding)

    feature_mode:
        "humanml3d" / "group" — 263-dim HumanML3D vectors in `data_root/new_joint_vecs/`,
                      consumed by GroupDiT (partitioned into 7 per-body-part group tokens).
        "smplh"     — 135-dim SMPL-H feature vectors stored flat as `data_root/<id>.npy`
                      (from src/data/amass_to_smplh.py). Texts come from `text_root` (the
                      processed HumanML3D dir), since the SMPL feature dir has none.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_frames: int = 196,
        min_frames: int = 16,
        feature_mode: str = "humanml3d",
        text_root: str | None = None,
    ):
        assert feature_mode in ("humanml3d", "group", "smplh"), \
            f"Unknown feature_mode: {feature_mode}"
        self.data_root = data_root
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.feature_mode = feature_mode
        # texts/text_emb live with the processed dataset; for smplh that differs from data_root.
        text_root = text_root or data_root

        is_smplh = feature_mode == "smplh"
        self.feature_dim = 135 if is_smplh else 263
        # smplh feats are flat <id>.npy at data_root; HumanML3D vecs are in new_joint_vecs/.
        self.vec_dir = data_root if is_smplh else os.path.join(data_root, "new_joint_vecs")

        mean = np.load(os.path.join(data_root, "Mean.npy"))  # (feature_dim,)
        std  = np.load(os.path.join(data_root, "Std.npy"))
        self.mean = mean
        self.std  = std

        # clip IDs for this split
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # filter out clips that are too short after loading lengths
        self.ids = self._filter_by_length()

        self.text_dir     = os.path.join(text_root, "texts")
        text_emb_dir      = os.path.join(text_root, "text_emb")
        self.text_emb_dir = text_emb_dir if os.path.isdir(text_emb_dir) else None

    def _filter_by_length(self):
        valid = []
        for clip_id in self.ids:
            path = os.path.join(self.vec_dir, f"{clip_id}.npy")
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
                     feature_mode="humanml3d", text_root=None):
    dataset = HumanML3DDataset(data_root, split=split, max_frames=max_frames,
                               feature_mode=feature_mode, text_root=text_root)
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
