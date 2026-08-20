"""
Generate animations from text prompts using a trained checkpoint.

Prompt mode (--num_samples 0):
    python src/sample_model.py --checkpoint ... --data_root ... --num_samples 0 \
        --prompts "a person walks forward" "a person raises their right arm"

Validation-set mode (--num_samples N):
    python src/sample_model.py --checkpoint ... --data_root ... --num_samples 8
    Samples N clips from the split, generates a motion for each annotation, and saves a
    side-by-side MP4 against the ground truth with per-frame MPJPE.
"""

import os
import argparse

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

import matplotlib
from utils.logger import get_logger

log = get_logger(__name__)
matplotlib.use("Agg")

from data.clips import iter_split_clips
from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from model.sampler import DDPMSampler
from utils.cli import resolve_device
from utils.decode import recover_joints
from utils.model_io import load_model
from utils.skeleton import mpjpe_from_joints
from utils.visualise import save_animation, save_comparison_animation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--data_root",      required=True,
                   help="Data root (Mean/Std.npy, <split>.txt, new_joint_vecs/, texts/).")
    p.add_argument("--output_dir",     default="./eval_results")
    p.add_argument("--prompts",        nargs="+", default=None,
                   help="Text prompts (used when --num_samples is 0).")
    p.add_argument("--num_samples",    type=int, default=3,
                   help="Render this many side-by-side comparisons from --split. "
                        "0 = use --prompts instead.")
    p.add_argument("--split",          default="val")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--length",         type=int,   default=196)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_steps",      type=int,   default=1000,
                   help="Must equal the checkpoint's 'timesteps' — DDPMSampler only "
                        "supports full-resolution sampling.")
    p.add_argument("--smooth_sigma",   type=float, default=1.5)
    p.add_argument("--no_ema",         action="store_true",
                   help="Load model.pt instead of ema.pt.")
    return p, p.parse_args()


def load_val_samples(data_root, split, num_samples, max_frames=196, seed=42):
    """`num_samples` random clips from a split as {clip_id, text, raw_feat, length}."""
    samples = []
    for clip in iter_split_clips(data_root, split, max_frames, max_clips=num_samples,
                                 seed=seed):
        raw = np.load(clip["vec_path"])[:clip["T"]]
        samples.append({"clip_id": clip["id"], "text": clip["text"],
                        "raw_feat": raw, "length": clip["T"]})
    return samples


def main():
    parser, args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device()
    log.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode = config.get("feature_mode", "humanml3d")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule     = NoiseSchedule.from_config(config, device=device)
    sampler      = DDPMSampler(model, schedule, device)

    if args.num_samples > 0:
        items = load_val_samples(args.data_root, args.split, args.num_samples,
                                 seed=args.seed)
        log.info(f"\nValidation-set mode: {len(items)} clips from '{args.split}' split\n")
    else:
        if not args.prompts:
            parser.error("--prompts is required when --num_samples is 0.")
        items = [{"clip_id": None, "text": p, "raw_feat": None, "length": None}
                 for p in args.prompts]
        log.info(f"\nPrompt mode: {len(items)} prompts\n")

    for i, item in enumerate(items):
        prompt  = item["text"]
        clip_id = item["clip_id"] or f"prompt_{i:03d}"
        log.info(f"[{i+1}/{len(items)}] '{prompt}'")

        with torch.no_grad():
            context = text_encoder.encode([prompt])
        motion_norm = sampler.sample(
            context, length=args.length, guidance_scale=args.guidance_scale,
            num_steps=args.num_steps).cpu().numpy()          # (length, D) normalised

        joints_gen = recover_joints(motion_norm * std + mean, feature_mode)
        if args.smooth_sigma > 0:
            joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)

        slug = prompt[:40].replace(" ", "_")
        if item["raw_feat"] is None:
            save_animation(joints_gen,
                           os.path.join(args.output_dir, f"{i:03d}_{slug}.mp4"),
                           title=prompt)
            continue

        joints_gt = recover_joints(item["raw_feat"], feature_mode)
        if args.smooth_sigma > 0:
            joints_gt = gaussian_filter1d(joints_gt, sigma=args.smooth_sigma, axis=0)
        per_frame, total_mpjpe, T_common = mpjpe_from_joints(joints_gen, joints_gt)
        log.info(f"   MPJPE: {total_mpjpe*1000:.1f} mm  (over {T_common} frames)")
        save_comparison_animation(
            joints_gen, joints_gt, per_frame, total_mpjpe,
            os.path.join(args.output_dir, f"{i:03d}_{clip_id}_{slug}.mp4"),
            title=prompt, clip_id=clip_id)

    log.info(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
