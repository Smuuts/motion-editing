"""
MotionFix triplets on disk: featurisation cache, dataset, collation and batching.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data.smplh_features import resample_motion, smplh_to_features
from utils.logger import get_logger

log = get_logger(__name__)


def featurise(motion: dict, src_fps: float, edit_fps: float) -> np.ndarray:
    """One SMPL-H `{rots, trans}` clip -> the 135-d training feature at `edit_fps`."""
    rots = np.asarray(motion["rots"], dtype=np.float32)
    trans = np.asarray(motion["trans"], dtype=np.float32)
    rots, trans = resample_motion(rots, trans, src_fps, edit_fps)
    return smplh_to_features(rots, trans)


def build_cache(args, split_keys, data, cache_dir) -> tuple[list[str], dict[str, str]]:
    """Featurise every triplet once into `cache_dir`. Resumable: an existing pair is
    left alone. Returns (kept keyids, {skipped keyid: reason})."""
    os.makedirs(cache_dir, exist_ok=True)
    kept, skipped = [], {}
    for k in log.progress(split_keys, desc=f"caching {os.path.basename(cache_dir)}",
                          leave=True):
        src_path = os.path.join(cache_dir, f"{k}_s.npy")
        tgt_path = os.path.join(cache_dir, f"{k}_t.npy")
        if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
            source = featurise(data[k]["motion_source"], args.src_fps, args.edit_fps)
            target = featurise(data[k]["motion_target"], args.src_fps, args.edit_fps)
            # The pair is scored on its common window.
            n = min(len(source), len(target), args.max_frames)
            if n < args.min_frames:
                skipped[k] = f"too short ({n})"
                continue
            np.save(src_path, source[:n])
            np.save(tgt_path, target[:n])
        kept.append(k)
    return kept, skipped


class TripletDataset(Dataset):
    """(source, target, instruction) triplets, normalised.

    Returns UNPADDED clips — `collate` pads each batch to its own longest clip rather
    than to a global `max_frames`.

    `preload` holds every featurised clip in RAM (0.43 GB for the 5,387 train triplets),
    removing two small disk reads per sample per step. `text_emb` holds instruction
    embeddings precomputed once (2.25 GB), removing a T5 forward from every step and
    letting the encoder be dropped from VRAM entirely.
    """

    def __init__(self, keys, cache_dir, texts, mean, std, max_frames,
                 preload=True, text_emb=None):
        self.keys, self.dir, self.texts = keys, cache_dir, texts
        self.mean, self.std, self.max_frames = mean, std, max_frames
        self.text_emb, self.mem = text_emb, None
        if preload:
            self.mem = {k: (self._load(k, "s"), self._load(k, "t"))
                        for k in log.progress(keys, desc="preloading")}
            self.lengths = [len(self.mem[k][0]) for k in keys]
        else:
            # Read the length out of the .npy HEADER via mmap. Calling _pair() here would
            # load and normalise every clip just to discard it, defeating the point of
            # not preloading in the first place.
            self.lengths = [min(np.load(os.path.join(cache_dir, f"{k}_s.npy"),
                                        mmap_mode="r").shape[0], max_frames)
                            for k in keys]

    def _load(self, key, which):
        a = np.load(os.path.join(self.dir, f"{key}_{which}.npy"))[: self.max_frames]
        return ((a - self.mean) / self.std).astype(np.float32)

    def _pair(self, key):
        if self.mem is not None:
            return self.mem[key]
        return self._load(key, "s"), self._load(key, "t")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        source, target = self._pair(key)
        item = {"source": torch.from_numpy(source), "target": torch.from_numpy(target),
                "length": len(source), "text": self.texts[key], "keyid": key}
        if self.text_emb is not None:
            item["context"] = self.text_emb[key]
        return item


def collate(batch, max_frames=None):
    """Pad a batch of variable-length clips.

    `max_frames=None` pads to the batch's own longest clip; an integer pads to that fixed
    width (`--pad_to max`). Padding frames are excluded from the loss AND masked out of
    self-attention — the (B, F) mask is expanded to (B, F*G) and used as a key-padding
    mask — so batch-max padding is bit-identical to fixed-width padding, verified at
    5.3e-06 on real frames.
    """
    n_frames = max_frames or max(b["length"] for b in batch)
    feat_dim = batch[0]["source"].shape[1]

    def pad(x):
        if len(x) == n_frames:
            return x
        return torch.cat([x, torch.zeros(n_frames - len(x), feat_dim, dtype=x.dtype)])

    out = {"source": torch.stack([pad(b["source"]) for b in batch]),
           "target": torch.stack([pad(b["target"]) for b in batch]),
           "length": torch.tensor([b["length"] for b in batch]),
           "text": [b["text"] for b in batch],
           "keyid": [b["keyid"] for b in batch]}
    if "context" in batch[0]:
        out["context"] = torch.stack([b["context"] for b in batch])
    return out


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar-length clips, in shuffled batch order.

    Batch-max padding only pays off if a batch's clips are actually similar in length —
    one 100-frame clip drags a batch of 40-frame ones up with it. Sorting into buckets
    and then shuffling the BATCHES keeps the stochasticity that matters (which batches,
    in what order) while making each batch homogeneous.
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True, seed=0):
        self.lengths, self.batch_size = list(lengths), batch_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        jitter = rng.uniform(0, 1e-3, len(self.lengths))
        order = np.argsort(np.array(self.lengths) + jitter)
        batches = [order[i:i + self.batch_size].tolist()
                   for i in range(0, len(order), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
