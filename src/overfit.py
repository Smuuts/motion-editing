"""
Overfit sanity check for MotionDiT.

Trains the model on a single motion clip for N steps, then generates
that same clip from noise and compares the result visually and
quantitatively. If the model can overfit one example, the architecture,
loss, and data pipeline are all working correctly.

Usage:
    python overfit_check.py --data_root ./data/HumanML3D

What a passing run looks like:
    - Loss drops from ~1.0 to below 0.01 within 2000–3000 steps (8-layer/512-dim)
    - Reconstructed joint positions closely match the ground truth
    - The generated animation looks like a recognisable human motion

What a failing run looks like:
    - Loss stagnates above 0.1 after 500 steps → architecture or data bug
    - Loss goes to NaN → learning rate too high or normalisation wrong
    - Animation looks random despite low loss → recover_from_ric bug
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── make src/ importable regardless of where the script is run from ──────────
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.dit import build_model
from model.text_encoder import CLIPTextEncoder
from model.schedule import NoiseSchedule
from utils.visualise import recover_from_ric, save_animation


# ── overfit loop ──────────────────────────────────────────────────────────────

def overfit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load Mean/Std ─────────────────────────────────────────────────
    mean = np.load(os.path.join(args.data_root, "Mean.npy"))  # (263,)
    std  = np.load(os.path.join(args.data_root, "Std.npy"))   # (263,)

    # ── pick one clip ─────────────────────────────────────────────────
    split_file = os.path.join(args.data_root, "train.txt")
    with open(split_file) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    clip_id = clip_ids[args.clip_index]
    print(f"Overfitting on clip: {clip_id}")

    # ── load motion ───────────────────────────────────────────────────
    vecs = np.load(os.path.join(args.data_root, "new_joint_vecs", f"{clip_id}.npy"))
    T    = min(len(vecs), args.max_frames)
    vecs = vecs[:T]

    # save ground truth joints for comparison (raw features, before normalisation)
    gt_joints = recover_from_ric(vecs.copy(), joints_num=22)

    # normalise
    vecs_norm = (vecs - mean) / std
    motion_gt = torch.from_numpy(vecs_norm).unsqueeze(0).float().to(device)  # (1, T, 263)

    # ── load text ─────────────────────────────────────────────────────
    text_path = os.path.join(args.data_root, "texts", f"{clip_id}.txt")
    with open(text_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    text = lines[0].split("#")[0].strip()
    print(f"Text annotation: '{text}'")

    # ── build model ───────────────────────────────────────────────────
    context_dim = 768 if "L/14" in args.clip_version else 512
    model_cfg = {
        "input_dim":   263,
        "latent_dim":  args.latent_dim,
        "context_dim": context_dim,
        "num_heads":   args.num_heads,
        "num_layers":  args.num_layers,
        "max_frames":  args.max_frames,
        "dropout":     0.0,
    }
    model        = build_model(model_cfg, device=device)
    text_encoder = CLIPTextEncoder(args.clip_version, device=device)
    schedule     = NoiseSchedule(timesteps=1000, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # ── encode text once — reused every step ─────────────────────────
    with torch.no_grad():
        context = text_encoder.encode([text])  # (1, 77, context_dim)

    # ── optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-5
    )

    # ── overfit loop ──────────────────────────────────────────────────
    losses = []
    model.train()
    pbar = tqdm(range(args.steps), desc="Overfitting")

    cfg_dropout = 0.1

    for step in pbar:
        t = torch.randint(0, 1000, (1,), device=device)
        x_t, noise = schedule.q_sample(motion_gt, t)

        ctx = None if (torch.rand(1).item() < cfg_dropout) else context
        eps_pred = model(x_t, t, ctx)

        loss = ((noise - eps_pred) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    # ── plot loss curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Overfit loss — clip {clip_id}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(args.output_dir, "overfit_loss.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"\nFinal loss: {losses[-1]:.6f}")

    if losses[-1] > 0.05:
        print("WARNING: loss did not converge below 0.05 — something is wrong.")
    else:
        print("Loss converged — model can overfit one sample.")

    # ── generate from noise ───────────────────────────────────────────
    print("\nGenerating motion from noise...")
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, T, 263, device=device)

        for t_val in tqdm(reversed(range(1, 1000)), desc="Denoising", total=999):
            t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)

            eps_cond   = model(x, t_batch, context)
            eps_uncond = model(x, t_batch, context=None)
            eps = eps_uncond + args.guidance_scale * (eps_cond - eps_uncond)

            x = schedule.p_sample(x, t_batch, eps)

    generated_norm = x[0].cpu().numpy()  # (T, 263)

    # ── denormalise and recover joints ────────────────────────────────
    generated = generated_norm * std + mean
    gen_joints = recover_from_ric(generated, joints_num=22)

    # ── compute MPJPE on recovered joints ─────────────────────────────
    mpjpe = np.sqrt(((gt_joints - gen_joints) ** 2).sum(axis=-1)).mean() * 1000
    print(f"MPJPE (generated vs ground truth): {mpjpe:.1f} mm")
    if mpjpe < 150:
        print("MPJPE is low — generated motion closely matches the source.")
    else:
        print("MPJPE is high — the model is not reconstructing the motion well.")

    # ── reconstruct via single-step denoising at t=100 ───────────────
    print("\nRunning reconstruction test (t=100 single-step denoising)...")
    with torch.no_grad():
        t_recon = torch.full((1,), 100, dtype=torch.long, device=device)
        x_noisy, _ = schedule.q_sample(motion_gt, t_recon)
        eps_recon = model(x_noisy, t_recon, context)
        recon_norm = schedule.predict_x0_from_eps(x_noisy, t_recon, eps_recon)
        recon_norm = recon_norm[0].cpu().numpy()

    recon = recon_norm * std + mean
    recon_joints = recover_from_ric(recon, joints_num=22)
    recon_mpjpe  = np.sqrt(((gt_joints - recon_joints) ** 2).sum(axis=-1)).mean() * 1000
    print(f"Reconstruction MPJPE (t=100 single-step): {recon_mpjpe:.1f} mm")

    # ── save animations ───────────────────────────────────────────────
    print("\nRendering animations...")
    save_animation(gt_joints,    os.path.join(args.output_dir, "gt.mp4"),
                   title=f"Ground truth — {text}")
    save_animation(gen_joints,   os.path.join(args.output_dir, "generated.mp4"),
                   title=f"Generated (CFG={args.guidance_scale})")
    save_animation(recon_joints, os.path.join(args.output_dir, "reconstruction.mp4"),
                   title="Reconstruction (t=100)")

    # ── summary ───────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────")
    print(f"  Final overfit loss:         {losses[-1]:.6f}  (target: < 0.01)")
    print(f"  MPJPE generated vs GT:      {mpjpe:.1f} mm    (target: < 150 mm)")
    print(f"  MPJPE reconstruction vs GT: {recon_mpjpe:.1f} mm    (target: < 100 mm at t=100)")
    print(f"  Animations saved to:        {args.output_dir}/")
    print()
    if losses[-1] < 0.01 and recon_mpjpe < 50:
        print("PASS — model overfits correctly. Safe to start full training.")
    elif losses[-1] < 0.01 and recon_mpjpe >= 50:
        print("PARTIAL — loss converged but reconstruction MPJPE is high.")
        print("         Check recover_from_ric — the data pipeline may still be wrong.")
    else:
        print("FAIL — loss did not converge. Check model architecture and data pipeline.")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--output_dir",     default="./overfit_check")
    p.add_argument("--clip_index",     type=int,   default=1,
                   help="Index into train.txt — which clip to overfit on.")
    p.add_argument("--steps",          type=int,   default=3000,
                   help="Gradient steps. An 8-layer/512-dim model typically needs "
                        "~3000 steps to reach loss < 0.01 on one clip.")
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--latent_dim",     type=int,   default=512)
    p.add_argument("--num_layers",     type=int,   default=8)
    p.add_argument("--num_heads",      type=int,   default=8)
    p.add_argument("--max_frames",     type=int,   default=196)
    p.add_argument("--clip_version",   type=str,   default="ViT-B/32")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale. Use 1.0 (conditional only) unless null_text_emb "
                        "has been trained with cfg_dropout > 0.")
    return p.parse_args()


if __name__ == "__main__":
    overfit(parse_args())
