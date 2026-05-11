"""
Pre-compute CLIP text embeddings for every HumanML3D clip.

Saves {data_root}/text_emb/{clip_id}.npy  — shape (num_annotations, 77, dim), float16.
The dataset will automatically use these files during training, eliminating the
CLIP forward pass from the training loop.

Usage:
    python precompute_text.py --data_root ./data/HumanML3D
    python precompute_text.py --data_root ./data/HumanML3D --clip_version ViT-L/14
"""

import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.text_encoder import CLIPTextEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True,
                   help="HumanML3D root directory (must contain texts/ and split .txt files).")
    p.add_argument("--clip_version", default="ViT-B/32",
                   help="CLIP variant — must match the one used in training.")
    p.add_argument("--splits",       nargs="+", default=["train", "val", "test"],
                   help="Which split files to scan for clip IDs.")
    p.add_argument("--batch_size",   type=int, default=512,
                   help="Number of annotation strings encoded per CLIP forward pass.")
    p.add_argument("--device",       default=None,
                   help="'cuda' or 'cpu'. Defaults to cuda if available.")
    p.add_argument("--overwrite",    action="store_true",
                   help="Re-encode clips that already have a cached file.")
    return p.parse_args()


def collect_clip_annotations(data_root: str, splits: list[str]) -> dict[str, list[str]]:
    """Return {clip_id: [annotation, ...]} for every clip found in the given splits."""
    text_dir = os.path.join(data_root, "texts")
    clip_ids: set[str] = set()
    for split in splits:
        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"  Warning: split file not found, skipping: {split_file}")
            continue
        with open(split_file) as f:
            clip_ids.update(line.strip() for line in f if line.strip())

    clip_annotations: dict[str, list[str]] = {}
    missing = 0
    for clip_id in sorted(clip_ids):
        text_path = os.path.join(text_dir, f"{clip_id}.txt")
        if not os.path.exists(text_path):
            missing += 1
            continue
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        annotations = [l.split("#")[0].strip() for l in lines if l.split("#")[0].strip()]
        if annotations:
            clip_annotations[clip_id] = annotations

    print(f"  Found {len(clip_annotations)} clips across splits "
          f"({missing} missing text files skipped).")
    return clip_annotations


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = os.path.join(args.data_root, "text_emb")
    os.makedirs(out_dir, exist_ok=True)

    # ── collect annotations ───────────────────────────────────────────────
    print("Scanning clips...")
    clip_annotations = collect_clip_annotations(args.data_root, args.splits)

    # skip already-cached clips unless --overwrite
    if not args.overwrite:
        clip_annotations = {
            cid: anns for cid, anns in clip_annotations.items()
            if not os.path.exists(os.path.join(out_dir, f"{cid}.npy"))
        }
        print(f"  {len(clip_annotations)} clips need encoding (use --overwrite to redo all).")

    if not clip_annotations:
        print("Nothing to do.")
        return

    # ── load CLIP ─────────────────────────────────────────────────────────
    print(f"Loading CLIP {args.clip_version}...")
    encoder = CLIPTextEncoder(args.clip_version, device=device)

    CHUNK_SIZE = 256
    clip_ids_ordered = list(clip_annotations.keys())
    total_clips = len(clip_ids_ordered)

    dim = None
    saved = 0
    pbar = tqdm(total=total_clips, desc="Encoding & saving")
    for chunk_start in range(0, total_clips, CHUNK_SIZE):
        chunk_ids = clip_ids_ordered[chunk_start : chunk_start + CHUNK_SIZE]

        # flatten annotations for this chunk
        chunk_texts: list[str] = []
        chunk_slices: dict[str, tuple[int, int]] = {}
        for clip_id in chunk_ids:
            anns = clip_annotations[clip_id]
            start = len(chunk_texts)
            chunk_texts.extend(anns)
            chunk_slices[clip_id] = (start, len(chunk_texts))

        # encode in mini-batches
        chunk_embs: list[torch.Tensor] = []
        for i in range(0, len(chunk_texts), args.batch_size):
            mini = chunk_texts[i : i + args.batch_size]
            chunk_embs.append(encoder.encode(mini).cpu())
        chunk_tensor = torch.cat(chunk_embs, dim=0)  # (chunk_annotations, 77, dim)
        dim = chunk_tensor.shape[-1]

        # save each clip and free immediately
        for clip_id in chunk_ids:
            start, end = chunk_slices[clip_id]
            arr = chunk_tensor[start:end].numpy().astype(np.float16)
            np.save(os.path.join(out_dir, f"{clip_id}.npy"), arr)

        del chunk_tensor, chunk_embs
        saved += len(chunk_ids)
        pbar.update(len(chunk_ids))
    pbar.close()

    total_annotations = sum(len(v) for v in clip_annotations.values())
    size_gb = total_annotations * 77 * dim * 2 / 1024 ** 3  # float16 = 2 bytes
    print(f"\nDone. {saved} clips saved to {out_dir}/")
    print(f"  Embedding dim: {dim}  |  Approx disk usage: {size_gb:.2f} GB (float16)")


if __name__ == "__main__":
    main()
