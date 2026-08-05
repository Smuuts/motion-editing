"""
generate.py — generate and save motions for a dataset split.

Runs the model over every clip in a split and saves the results to disk.
Metrics are computed separately by evaluate.py.

Each clip is saved as {out_dir}/{clip_id}.npz with keys:
  gen_norm : (T, D) float32  generated motion, normalised
  gt_norm  : (T, D) float32  ground-truth motion, normalised
  ctx      : (77, ctx_dim) float32  CLIP embedding
  T        : int  frame count

A manifest.json is written to out_dir summarising the run args and
listing every successfully generated clip_id.  Passing --resume skips
clips whose .npz already exists.

Usage:
    python src/generate.py \\
        --checkpoint runs/exp_hml3d/checkpoint_latest \\
        --data_root  data/HumanML3D \\
        --split val \\
        --out_dir generated/val \\
        [--max_clips 500] \\
        [--guidance_scale 4.0] \\
        [--num_steps 1000] \\
        [--seed 42] \\
        [--no_ema] \\
        [--resume]
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from data.clips import iter_split_clips
from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from model.sampler import DDPMSampler
from utils.cli import resolve_device
from utils.model_io import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--data_root",      required=True,
                   help="Feature root with new_joint_vecs/, texts/ and Mean/Std. "
                        "smplh: data/HumanML3D/HumanML3D_smplh (135-d); humanml3d: "
                        "data/HumanML3D/HumanML3D (263-d).")
    p.add_argument("--split",          default="val", choices=["train", "val", "test"])
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--max_clips",      type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_steps",      type=int, default=1000,
                   help="Must equal the checkpoint's diffusion timesteps (config.json's "
                        "'timesteps', usually 1000) — DDPMSampler only supports "
                        "full-resolution sampling.")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--no_ema",         action="store_true")
    p.add_argument("--resume",         action="store_true",
                   help="Skip clips whose .npz already exists in out_dir.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    max_frames    = config.get("max_frames", 196)
    feature_mode  = config.get("feature_mode", "humanml3d")
    print(f"Feature mode: {feature_mode}")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))

    # Built lazily: only needed for clips without a precomputed text embedding.
    text_encoder = None
    schedule     = NoiseSchedule.from_config(config, device=device)
    sampler      = DDPMSampler(model, schedule, device)

    print(f"\nLoading '{args.split}' split …")
    clips = iter_split_clips(args.data_root, args.split, max_frames,
                             max_clips=args.max_clips, seed=args.seed,
                             with_text_emb=True)
    print(f"  {len(clips)} clips")

    succeeded = []
    skipped   = 0

    manifest = {
        "split":          args.split,
        "checkpoint":     args.checkpoint,
        "feature_mode":   feature_mode,
        "data_root":      args.data_root,
        "guidance_scale": args.guidance_scale,
        "num_steps":      args.num_steps,
        "seed":           args.seed,
        "n_clips":        0,
        "clip_ids":       [],
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")

    def write_manifest():
        manifest["n_clips"]  = len(succeeded)
        manifest["clip_ids"] = succeeded
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest → {manifest_path}")

    try:
        for clip in tqdm(clips, desc="Generating"):
            cid  = clip["id"]
            T    = clip["T"]
            out_path = os.path.join(args.out_dir, f"{cid}.npz")

            if args.resume and os.path.exists(out_path):
                succeeded.append(cid)
                skipped += 1
                continue

            raw_feat = np.load(clip["vec_path"])[:T]   # (T, 263) or (T, 135) for smplh
            gt_norm  = (raw_feat - mean) / std

            if clip["context_emb"] is not None:
                ctx_np = clip["context_emb"]
                ctx    = torch.from_numpy(ctx_np).unsqueeze(0).to(device)
            else:
                if text_encoder is None:
                    text_encoder = build_text_encoder(config, device=device)
                with torch.no_grad():
                    ctx = text_encoder.encode([clip["text"]])
                ctx_np = ctx[0].cpu().numpy()

            try:
                with torch.no_grad():
                    gen_norm = sampler.sample(
                        ctx,
                        length=T,
                        guidance_scale=args.guidance_scale,
                        num_steps=args.num_steps,
                        show_progress=False,
                    ).cpu().numpy()
            except Exception as exc:
                print(f"  [WARN] {cid}: generation failed — {exc}")
                continue

            np.savez_compressed(out_path,
                                gen_norm=gen_norm.astype(np.float32),
                                gt_norm=gt_norm.astype(np.float32),
                                ctx=ctx_np.astype(np.float32),
                                T=np.array(T, dtype=np.int32))
            succeeded.append(cid)

    except KeyboardInterrupt:
        print(f"\nInterrupted — {len(succeeded)} clips saved so far.")

    finally:
        print(f"\n{len(succeeded)}/{len(clips)} clips saved"
              + (f" ({skipped} resumed)" if skipped else "") + f" → {args.out_dir}")
        write_manifest()


if __name__ == "__main__":
    main()
