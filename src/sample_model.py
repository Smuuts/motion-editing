"""
Evaluation script: generate animations from text prompts using a trained MotionDiT.

Usage:
    python evaluate.py \
        --checkpoint ./runs/exp1/checkpoint_latest \
        --data_root  ./data/HumanML3D \
        --prompts    "a person walks forward" "a person raises their right arm"

    # faster sampling during development
    python evaluate.py --checkpoint ... --data_root ... --num_steps 200
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")

from model.dit import build_model
from model.text_encoder import CLIPTextEncoder
from model.schedule import NoiseSchedule
from model.sampler import DDPMSampler
from utils.visualise import recover_from_ric, save_animation


DEFAULT_PROMPTS = [
    "a person walks forward",
    "a person raises their right arm slowly",
    "a person waves their hand",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True,
                   help="Path to checkpoint directory (contains model.pt / ema.pt). "
                        "Pass the checkpoint_latest symlink for the most recent save.")
    p.add_argument("--data_root",      required=True,
                   help="HumanML3D root (needs Mean.npy and Std.npy).")
    p.add_argument("--output_dir",     default="./eval_results")
    p.add_argument("--prompts",        nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--length",         type=int,   default=120,
                   help="Frames to generate. At 20 fps: 120 = 6 s, 196 = ~10 s.")
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="CFG scale. Higher = stronger text adherence, less diversity.")
    p.add_argument("--num_steps",      type=int,   default=1000,
                   help="DDPM sampling steps. 200 is faster for development.")
    p.add_argument("--smooth_sigma",   type=float, default=1.5,
                   help="Gaussian smoothing sigma on recovered joint positions "
                        "to reduce temporal jitter (0 = disabled).")
    return p.parse_args()


def load_model(ckpt_dir: str, device):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    context_dim = 768 if "L/14" in config.get("clip_version", "ViT-B/32") else 512
    model = build_model({
        "input_dim":   263,
        "latent_dim":  config.get("latent_dim",  512),
        "context_dim": context_dim,
        "num_heads":   config.get("num_heads",   8),
        "num_layers":  config.get("num_layers",  8),
        "max_frames":  config.get("max_frames",  196),
        "dropout":     0.0,
    }, device=device)

    weights = os.path.join(ckpt_dir, "ema.pt")
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {weights}")
    return model, config


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))  # (263,)
    std  = np.load(os.path.join(args.data_root, "Std.npy"))   # (263,)

    model, config = load_model(args.checkpoint, device=device)

    clip_version = config.get("clip_version", "ViT-B/32")
    text_encoder = CLIPTextEncoder(clip_version, device=device)

    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    sampler  = DDPMSampler(model, schedule, device)

    print(f"\nGenerating {len(args.prompts)} clips "
          f"({args.length} frames, {args.length / 20:.1f} s at 20 fps)\n")

    for i, prompt in enumerate(args.prompts):
        print(f"[{i+1}/{len(args.prompts)}] '{prompt}'")

        with torch.no_grad():
            context = text_encoder.encode([prompt])  # (1, 77, context_dim)

        motion_norm = sampler.sample(
            context,
            length=args.length,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
        ).cpu().numpy()  # (length, 263), normalised

        # denormalise → raw HumanML3D feature space
        motion_raw = motion_norm * std + mean  # (length, 263)

        # recover world-space joint positions
        joints = recover_from_ric(motion_raw, joints_num=22)  # (length, 22, 3)

        # optional temporal smoothing to reduce frame-to-frame jitter
        if args.smooth_sigma > 0:
            joints = gaussian_filter1d(joints, sigma=args.smooth_sigma, axis=0)

        out_path = os.path.join(args.output_dir, f"{i:03d}_{prompt[:40].replace(' ', '_')}.mp4")
        save_animation(joints, out_path, title=prompt)

    print(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
