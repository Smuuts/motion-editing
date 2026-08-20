"""
Pre-compute text embeddings for every HumanML3D clip.

Saves {out_dir}/{clip_id}.npy — shape (num_annotations, L, dim), float16,
where L is 77 for CLIP or --t5_max_length for T5.

The dataset loads embeddings from {data_root}/text_emb/ automatically.
Use --out_dir to write to a different directory (e.g. when switching encoders).

Usage:
    python precompute_text.py --data_root ./data/HumanML3D
    python precompute_text.py --data_root ./data/HumanML3D --clip_version ViT-L/14
    python precompute_text.py --data_root ./data/HumanML3D --text_encoder t5 --t5_version t5-base
"""

import os
import argparse

import numpy as np
import torch

from model.text_encoder import build_text_encoder
from utils.logger import get_logger
from utils.cli import add_logging_args, configure_logging

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True,
                   help="HumanML3D root directory (must contain texts/ and split .txt files).")
    p.add_argument("--out_dir",       default=None,
                   help="Output directory for .npy files. Defaults to {data_root}/text_emb.")
    p.add_argument("--text_encoder",  default="clip", choices=["clip", "t5"],
                   help="Text encoder backend — must match the one used in training.")
    p.add_argument("--clip_version",  default="ViT-B/32",
                   help="CLIP variant (used when --text_encoder=clip).")
    p.add_argument("--t5_version",    default="t5-base",
                   help="T5 model name (used when --text_encoder=t5).")
    p.add_argument("--t5_max_length", type=int, default=128,
                   help="Fixed token length for T5 output (used when --text_encoder=t5).")
    p.add_argument("--splits",        nargs="+", default=["train", "val", "test"],
                   help="Which split files to scan for clip IDs.")
    p.add_argument("--batch_size",    type=int, default=512,
                   help="Number of annotation strings encoded per forward pass.")
    p.add_argument("--device",        default=None,
                   help="'cuda' or 'cpu'. Defaults to cuda if available.")
    p.add_argument("--overwrite",     action="store_true",
                   help="Re-encode clips that already have a cached file.")
    add_logging_args(p)
    return configure_logging(p.parse_args())


def collect_clip_annotations(data_root: str, splits: list[str]) -> dict[str, list[str]]:
    """Return {clip_id: [annotation, ...]} for every clip found in the given splits."""
    text_dir = os.path.join(data_root, "texts")
    clip_ids: set[str] = set()
    for split in splits:
        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(split_file):
            log.warning(f"split file not found, skipping: {split_file}")
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

    log.info(f"  Found {len(clip_annotations)} clips across splits "
          f"({missing} missing text files skipped).")
    return clip_annotations


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = args.out_dir or os.path.join(args.data_root, "text_emb")
    os.makedirs(out_dir, exist_ok=True)

    # ── collect annotations ───────────────────────────────────────────────
    log.info("Scanning clips...")
    clip_annotations = collect_clip_annotations(args.data_root, args.splits)

    # skip already-cached clips unless --overwrite
    if not args.overwrite:
        clip_annotations = {
            cid: anns for cid, anns in clip_annotations.items()
            if not os.path.exists(os.path.join(out_dir, f"{cid}.npy"))
        }
        log.info(f"  {len(clip_annotations)} clips need encoding (use --overwrite to redo all).")

    if not clip_annotations:
        log.info("Nothing to do.")
        return

    # ── load encoder ──────────────────────────────────────────────────────
    encoder_label = (f"T5 {args.t5_version} (max_length={args.t5_max_length})"
                     if args.text_encoder == "t5" else f"CLIP {args.clip_version}")
    log.info(f"Loading {encoder_label}...")
    encoder = build_text_encoder(vars(args), device=device)

    CHUNK_SIZE = 256
    clip_ids_ordered = list(clip_annotations.keys())
    total_clips = len(clip_ids_ordered)

    dim = None
    saved = 0
    pbar = log.progress(None, desc="Encoding & saving", total=total_clips)
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
    seq_len = encoder.max_length
    size_gb = total_annotations * seq_len * dim * 2 / 1024 ** 3  # float16 = 2 bytes
    log.info(f"\nDone. {saved} clips saved to {out_dir}/")
    log.info(f"  Encoder: {encoder_label}  |  seq_len: {seq_len}  |  dim: {dim}"
          f"  |  Approx disk: {size_gb:.2f} GB (float16)")


if __name__ == "__main__":
    main()
