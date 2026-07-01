"""
Evaluation script: generate animations from text prompts using a trained MotionDiT.

Modes
-----
Prompt mode (requires --num_samples 0):
    python sample_model.py --checkpoint ... --data_root ... --num_samples 0
    --prompts "a person walks forward" "a person raises their right arm"

Validation-set mode  (--num_samples N):
    python sample_model.py --checkpoint ... --data_root ... --num_samples 8
    Samples N clips from the validation split, generates a matching motion for
    each text annotation, and saves a side-by-side MP4 with per-frame MPJPE.
"""

import os
import sys
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")

from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from model.sampler import DDPMSampler
from utils.model_io import load_model
from utils.visualise import (
    recover_from_ric, save_animation, save_comparison_animation, mpjpe_from_joints,
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--data_root",      required=True,
                   help="HumanML3D root (needs Mean.npy, Std.npy, val.txt, …).")
    p.add_argument("--output_dir",     default="./eval_results")

    # Prompt mode
    p.add_argument("--prompts",        nargs="+", default=None,
                   help="Text prompts (used when --num_samples is 0).")

    # Validation-set comparison mode
    p.add_argument("--num_samples",    type=int, default=3,
                   help="Sample this many clips from --split and render "
                        "side-by-side comparison videos with MPJPE overlay. "
                        "Set to 0 to use --prompts mode instead.")
    p.add_argument("--split",          default="val",
                   help="Dataset split to draw clips from (default: val).")
    p.add_argument("--seed",           type=int, default=42)

    # Generation hyper-parameters
    p.add_argument("--length",         type=int,   default=196)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_steps",      type=int,   default=1000,
                   help="Must equal the checkpoint's diffusion timesteps (config.json's "
                        "'timesteps', usually 1000) — DDPMSampler only supports "
                        "full-resolution sampling.")
    p.add_argument("--smooth_sigma",   type=float, default=1.5)
    p.add_argument("--no_ema",         action="store_true",
                   help="Load model.pt instead of ema.pt.")
    return p, p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Joint recovery helpers
# ──────────────────────────────────────────────────────────────────────────────

_SMPLH_BODY_MODEL = None


def _smplh_body_model(model_path: str = "data/motionfix/data/body_models/smplh"):
    """Lazily build + cache a neutral SMPLHLayer for decoding smplh features to joints."""
    global _SMPLH_BODY_MODEL
    if _SMPLH_BODY_MODEL is None:
        import smplx
        _SMPLH_BODY_MODEL = smplx.SMPLHLayer(model_path=model_path, gender="neutral", ext="npz").eval()
    return _SMPLH_BODY_MODEL


def recover_joints(raw_features: np.ndarray, feature_mode: str) -> np.ndarray:
    """Raw (denormalised) features → world-space joint positions (T, 22, 3).

    'humanml3d' (263) via RIC recovery; 'smplh' (135) via SMPL forward kinematics.
    """
    if feature_mode == "smplh":
        from data.smplh_features import smplh_decode_to_joints
        return smplh_decode_to_joints(raw_features, _smplh_body_model())
    return recover_from_ric(raw_features, joints_num=22)


# ──────────────────────────────────────────────────────────────────────────────
# Validation-set sampling
# ──────────────────────────────────────────────────────────────────────────────

def load_val_samples(data_root, split, num_samples, feature_mode, max_frames=196, seed=42):
    """
    Return a list of dicts with keys: clip_id, text, raw_feat, length.
    raw_feat is the (T, D) denormalised feature vector (D=130 or 263).
    """
    rng = np.random.default_rng(seed)

    split_file = os.path.join(data_root, f"{split}.txt")
    with open(split_file) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    vec_dir  = os.path.join(data_root, "new_joint_vecs")
    text_dir = os.path.join(data_root, "texts")

    valid_ids = [
        cid for cid in all_ids
        if os.path.exists(os.path.join(vec_dir, f"{cid}.npy"))
        and np.load(os.path.join(vec_dir, f"{cid}.npy"), mmap_mode="r").shape[0] >= 16
    ]

    chosen = rng.choice(valid_ids, size=min(num_samples, len(valid_ids)),
                        replace=False).tolist()

    samples = []
    for cid in chosen:
        raw = np.load(os.path.join(vec_dir, f"{cid}.npy"))        # (T, 263) raw
        T   = min(len(raw), max_frames)
        raw = raw[:T]
        raw_feat = raw

        with open(os.path.join(text_dir, f"{cid}.txt")) as f:
            lines = [l.strip() for l in f if l.strip()]
        text = lines[0].split("#")[0].strip() if lines else cid

        samples.append({"clip_id": cid, "text": text, "raw_feat": raw_feat, "length": T})

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser, args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode  = config.get("feature_mode", "humanml3d")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder  = build_text_encoder(config, device=device)
    schedule      = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    sampler       = DDPMSampler(model, schedule, device)

    # ── choose mode ────────────────────────────────────────────────────────────
    if args.num_samples > 0:
        samples = load_val_samples(
            args.data_root, args.split, args.num_samples,
            feature_mode, seed=args.seed,
        )
        print(f"\nValidation-set mode: {len(samples)} clips from '{args.split}' split\n")
    else:
        if not args.prompts:
            parser.error("--prompts is required when --num_samples is 0.")
        samples = None
        print(f"\nPrompt mode: {len(args.prompts)} prompts\n")

    # ── generation loop ────────────────────────────────────────────────────────
    items = samples if samples is not None else [
        {"clip_id": None, "text": p, "raw_feat": None, "length": None}
        for p in args.prompts
    ]

    for i, item in enumerate(items):
        prompt   = item["text"]
        clip_id  = item["clip_id"] or f"prompt_{i:03d}"
        raw_feat = item["raw_feat"]   # None in prompt mode

        print(f"[{i+1}/{len(items)}] '{prompt}'")

        with torch.no_grad():
            context = text_encoder.encode([prompt])

        motion_norm = sampler.sample(
            context,
            length=args.length,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
        ).cpu().numpy()                                       # (length, D) normalised

        motion_raw = motion_norm * std + mean                 # (length, D) denormalised
        joints_gen = recover_joints(motion_raw, feature_mode) # (length, 22, 3)

        if args.smooth_sigma > 0:
            joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)

        slug = prompt[:40].replace(" ", "_")

        if raw_feat is not None:
            # ── comparison mode ──────────────────────────────────────────────
            joints_gt = recover_joints(raw_feat, feature_mode)    # (T_gt, 22, 3)

            if args.smooth_sigma > 0:
                joints_gt = gaussian_filter1d(joints_gt, sigma=args.smooth_sigma, axis=0)

            per_frame, total_mpjpe, T_common = mpjpe_from_joints(joints_gen, joints_gt)
            print(f"   MPJPE: {total_mpjpe*1000:.1f} mm  (over {T_common} frames)")

            out_path = os.path.join(args.output_dir, f"{i:03d}_{clip_id}_{slug}.mp4")
            save_comparison_animation(
                joints_gen, joints_gt,
                per_frame, total_mpjpe,
                out_path,
                title=prompt,
                clip_id=clip_id,
            )
        else:
            # ── prompt-only mode ─────────────────────────────────────────────
            out_path = os.path.join(args.output_dir, f"{i:03d}_{slug}.mp4")
            save_animation(joints_gen, out_path, title=prompt)

    print(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
