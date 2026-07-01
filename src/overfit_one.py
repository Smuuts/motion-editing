"""
Overfit a single training example using the exact same conditions as train.py:
AMP, Min-SNR weighting, CFG dropout, AdamW + cosine-with-warmup schedule.

After training, generates a motion with the full 1000-step DDPM sampler
(identical to real inference in sample_model.py) and saves a side-by-side
MP4 against the ground truth clip.

Usage:
    python src/overfit_one.py --data_root ./data/HumanML3D
    python src/overfit_one.py --data_root ./data/HumanML3D --feature_mode group
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

import matplotlib
matplotlib.use("Agg")

from data.dataset import HumanML3DDataset
from model.dit import build_model
from model.sampler import DDPMSampler
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder, get_encoder_dims
from utils.visualise import recover_from_ric, save_comparison_animation, mpjpe_from_joints


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--feature_mode",  default="humanml3d", choices=["humanml3d", "group"])
    p.add_argument("--clip_version",  default="ViT-B/32")
    p.add_argument("--text_encoder",  default="clip", choices=["clip", "t5"])
    p.add_argument("--t5_version",    default="t5-base")
    p.add_argument("--t5_max_length", type=int, default=128)
    p.add_argument("--max_frames",    type=int,   default=196)

    # model — same defaults as train.py
    p.add_argument("--latent_dim",    type=int,   default=512)
    p.add_argument("--num_layers",    type=int,   default=8)
    p.add_argument("--num_heads",     type=int,   default=8)
    p.add_argument("--dropout",       type=float, default=0.1)

    # diffusion — same defaults as train.py
    p.add_argument("--timesteps",     type=int,   default=1000)
    p.add_argument("--cfg_dropout",   type=float, default=0.1)
    p.add_argument("--snr_gamma",     type=float, default=5.0)

    # optimiser — same defaults as train.py
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_steps",  type=int,   default=500)
    p.add_argument("--no_lr_decay",   action="store_true")

    # training
    p.add_argument("--batch_size",    type=int,   default=64,
                   help="Number of independent (t, ε) samples drawn per step from the "
                        "same motion. Matches real training's gradient variance. "
                        "Reduce if OOM; increase up to 128 to fully match train.py.")
    p.add_argument("--max_steps",     type=int,   default=10000)
    p.add_argument("--target_loss",   type=float, default=0.0,
                   help="Stop early once loss drops below this value (0 = disabled).")
    p.add_argument("--log_every",     type=int,   default=500)
    p.add_argument("--clip_id",       type=str,   default=None,
                   help="Clip ID to overfit on (default: first clip in train split).")

    # video output
    p.add_argument("--output_dir",      default="./eval_results/overfit_one")
    p.add_argument("--guidance_scale",  type=float, default=4.0)
    p.add_argument("--smooth_sigma",    type=float, default=1.5)
    return p.parse_args()


def _recover_joints(raw_feat: np.ndarray, feature_mode: str) -> np.ndarray:
    return recover_from_ric(raw_feat, joints_num=22)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Device: {device}")

    # ── dataset: one example ──────────────────────────────────────────────────
    ds = HumanML3DDataset(
        args.data_root,
        split="train",
        max_frames=args.max_frames,
        feature_mode=args.feature_mode,
    )
    if args.clip_id is not None:
        if args.clip_id not in ds.ids:
            print(f"ERROR: clip '{args.clip_id}' not found in train split.", file=sys.stderr)
            sys.exit(1)
        idx = ds.ids.index(args.clip_id)
    else:
        idx = 0
    sample  = ds[idx]
    clip_id = ds.ids[idx]
    length  = sample["length"]
    print(f"Clip: {clip_id}  (length={length} frames, feature_dim={ds.feature_dim})")

    B = args.batch_size
    # Tile the single motion B times so each step draws B independent (t, ε) pairs,
    # matching the gradient variance of real training with batch_size=B.
    motion    = sample["motion"].unsqueeze(0).expand(B, -1, -1).to(device)  # (B, max_frames, D)
    attn_mask = (torch.arange(args.max_frames, device=device)[None, :] < length).expand(B, -1)  # (B, F)

    # ── text embedding ────────────────────────────────────────────────────────
    encoder_cfg = vars(args)
    context_dim, text_seq_len = get_encoder_dims(encoder_cfg)
    if "context" in sample:
        context_1 = sample["context"].unsqueeze(0).to(device)  # (1, L, dim)
        context_dim = context_1.shape[-1]
        text_seq_len = context_1.shape[1]
        clip_text = f"[precomputed: {clip_id}]"
        print("Using precomputed text embedding.")
    else:
        text_encoder = build_text_encoder(encoder_cfg, device=device)
        clip_text = sample["text"]
        with torch.no_grad():
            context_1 = text_encoder.encode([clip_text])
        print(f"Text: {clip_text!r}")
    context = context_1.expand(B, -1, -1)

    # ── model + schedule ──────────────────────────────────────────────────────
    model = build_model({
        "feature_mode": args.feature_mode,
        "input_dim":    ds.feature_dim,
        "latent_dim":   args.latent_dim,
        "context_dim":  context_dim,
        "text_seq_len": text_seq_len,
        "num_heads":    args.num_heads,
        "num_layers":   args.num_layers,
        "max_frames":   args.max_frames,
        "dropout":      args.dropout,
    }, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.1f}M parameters")

    schedule = NoiseSchedule(timesteps=args.timesteps, device=device)
    scaler   = GradScaler(device=device.type, enabled=device.type == "cuda")

    # ── optimiser + LR schedule (identical to train.py) ──────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))
    if args.no_lr_decay:
        main_sched = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.max_steps - args.warmup_steps),
            eta_min=1e-6,
        )
    if args.warmup_steps > 0:
        warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=args.warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched],
                                 milestones=[args.warmup_steps])
    else:
        scheduler = main_sched

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nMax steps: {args.max_steps}  |  batch_size: {B}"
          f"  |  snr_gamma={args.snr_gamma}  |  cfg_dropout={args.cfg_dropout}\n")

    model.train()
    loss_mask = attn_mask.float().unsqueeze(-1)    # (B, F, 1)

    final_loss = float("inf")
    for step in range(1, args.max_steps + 1):
        t = torch.randint(0, schedule.T, (B,), device=device)
        x_t, noise = schedule.q_sample(motion, t)

        # CFG dropout — same as train.py
        if args.cfg_dropout > 0.0:
            drop_mask  = (torch.rand(B, device=device) < args.cfg_dropout)[:, None, None]
            null_emb   = model.null_text_emb.expand(B, -1, -1)
            context_in = torch.where(drop_mask, null_emb, context)
        else:
            context_in = context

        # loss — same as train.py
        with autocast(device_type=device.type):
            prediction = model(x_t, t, context_in, mask=attn_mask)
            per_elem   = (noise - prediction) ** 2 * loss_mask     # (B, T, D)

            if args.snr_gamma > 0.0:
                valid_elems = (attn_mask.float().sum(dim=1) * noise.shape[-1]).clamp(min=1)
                per_sample  = per_elem.sum(dim=(1, 2)) / valid_elems
                snr_t      = schedule.snr[t]
                snr_weight = snr_t.clamp(max=args.snr_gamma) / snr_t
                loss = (per_sample * snr_weight).mean()
            else:
                loss = per_elem.sum() / (loss_mask.sum() * noise.shape[-1]).clamp(min=1)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        final_loss = loss.item()
        if step % args.log_every == 0 or step == 1:
            print(f"Step {step:6d} | loss {final_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if args.target_loss > 0 and final_loss < args.target_loss:
            print(f"\nReached target loss {args.target_loss} at step {step}.")
            break
    else:
        print(f"\nFinished {args.max_steps} steps. Final loss: {final_loss:.6f}")

    # ── Full DDPM sampling (identical to real inference) ──────────────────────
    print(f"\nSampling with 1000-step DDPM (guidance_scale={args.guidance_scale}) …")
    model.eval()
    sampler = DDPMSampler(model, schedule, device)
    with torch.no_grad():
        motion_norm = sampler.sample(
            context_1,
            length=length,
            guidance_scale=args.guidance_scale,
            num_steps=args.timesteps,
        ).cpu().numpy()    # (length, D) normalised

    mean_np = np.load(os.path.join(args.data_root, "Mean.npy"))
    std_np  = np.load(os.path.join(args.data_root, "Std.npy"))
    motion_raw = motion_norm * std_np + mean_np

    vec_dir  = os.path.join(args.data_root, "new_joint_vecs")
    raw_full = np.load(os.path.join(vec_dir, f"{clip_id}.npy"))
    T_gt  = min(len(raw_full), args.max_frames)
    raw_gt = raw_full[:T_gt]

    joints_gen = _recover_joints(motion_raw, args.feature_mode)
    joints_gt  = _recover_joints(raw_gt,     args.feature_mode)

    if args.smooth_sigma > 0:
        joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)
        joints_gt  = gaussian_filter1d(joints_gt,  sigma=args.smooth_sigma, axis=0)

    per_frame, total_mpjpe, T_common = mpjpe_from_joints(joints_gen, joints_gt)
    print(f"MPJPE: {total_mpjpe * 1000:.1f} mm  (over {T_common} frames)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"overfit_{clip_id}.mp4")
    save_comparison_animation(
        joints_gen, joints_gt,
        per_frame, total_mpjpe,
        out_path,
        title=clip_text,
        clip_id=clip_id,
    )
    print(f"Video saved to: {out_path}")


if __name__ == "__main__":
    main()
