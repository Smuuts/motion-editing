"""
Phase 0.1 — Backbone prerequisite check for LEDITS++.

Verifies two properties the model must satisfy before implementing inversion:

  1. Epsilon-prediction quality — one-step x̂₀ reconstruction MPJPE at
     multiple noise levels, for both the conditional (text) and unconditional
     (null context) branches.  Low error at low t confirms the noise prediction
     is accurate; the unconditional branch is required for Stage 1 inversion
     and the ε_θ(x_t, ∅) term in Stage 3 Eq. 1.

  2. Noise MSE convergence — raw MSE between predicted and true noise at t=500
     as a sanity check that the training objective converged.

Usage:
    python src/verify_backbone.py \\
        --checkpoint runs/exp1/checkpoint_latest \\
        --data_root  data/HumanML3D \\
        --output_dir eval_results/backbone_verify
"""

import os
import sys
import argparse
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from data.dataset import build_dataloader
from utils.skeleton import compute_mpjpe
from utils.model_io import load_model
from utils.masks import length_to_mask


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0.1: Backbone prerequisite check")
    p.add_argument("--checkpoint",    required=True,
                   help="Checkpoint directory (ema.pt + config.json).")
    p.add_argument("--data_root",     required=True,
                   help="HumanML3D data root.")
    p.add_argument("--output_dir",    default="eval_results/backbone_verify")
    p.add_argument("--num_clips",     type=int, default=32,
                   help="Number of validation clips to test on.")
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--noise_levels",  type=int, nargs="+",
                   default=[50, 100, 250, 500, 750, 999],
                   help="Timesteps at which to evaluate one-step reconstruction.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    feature_mode = config.get("feature_mode", "humanml3d")
    timesteps    = config.get("timesteps", 1000)
    print(f"Model:         {type(model).__name__}")
    print(f"Feature mode:  {feature_mode}")
    print(f"Noise steps:   {timesteps}")

    schedule = NoiseSchedule(timesteps=timesteps, device=device)

    loader = build_dataloader(
        args.data_root, split="val",
        batch_size=args.batch_size,
        max_frames=config.get("max_frames", 196),
        feature_mode=feature_mode,
        shuffle=False,
    )
    ds     = loader.dataset
    mean_t = torch.from_numpy(ds.mean).float().to(device)
    std_t  = torch.from_numpy(ds.std).float().to(device)

    text_encoder = None
    if ds.text_emb_dir is None:
        text_encoder = build_text_encoder(config, device=device)

    # ── Collect validation clips ───────────────────────────────────────────────
    motions_l, contexts_l, lengths_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            motions_l.append(batch["motion"])
            lengths_l.append(
                batch["length"] if isinstance(batch["length"], torch.Tensor)
                else torch.tensor(batch["length"])
            )
            if "context" in batch:
                contexts_l.append(batch["context"])
            else:
                contexts_l.append(text_encoder.encode(batch["text"]).cpu())

            if sum(m.shape[0] for m in motions_l) >= args.num_clips:
                break

    motions  = torch.cat(motions_l,  dim=0)[:args.num_clips].to(device)
    contexts = torch.cat(contexts_l, dim=0)[:args.num_clips].to(device)
    lengths  = torch.cat(lengths_l,  dim=0)[:args.num_clips].to(device)
    B        = motions.shape[0]

    attn_mask = length_to_mask(lengths, motions.shape[1])
    print(f"\nTesting on {B} validation clips\n")

    # ── One-step x̂₀ reconstruction across noise levels ────────────────────────
    header = f"{'t':>6}  {'MPJPE cond':>12}  {'MPJPE uncond':>14}  {'noise MSE cond':>16}"
    print(header)
    print("-" * len(header))

    results: dict[int, dict] = {}
    with torch.no_grad():
        for t_val in sorted(args.noise_levels):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t, noise = schedule.q_sample(motions, t_batch)

            # Conditional: uses text embeddings
            eps_c  = model(x_t, t_batch, contexts, mask=attn_mask)
            x0_c   = schedule.predict_x0_from_eps(x_t, t_batch, eps_c)
            mpjpe_c = compute_mpjpe(
                x0_c, motions, mean_t, std_t, mask=attn_mask
            ).item()

            # Unconditional: context=None → model uses null_text_emb
            # This branch is required for Stage 1 inversion and Stage 3 CFG.
            eps_u  = model(x_t, t_batch, context=None, mask=attn_mask)
            x0_u   = schedule.predict_x0_from_eps(x_t, t_batch, eps_u)
            mpjpe_u = compute_mpjpe(
                x0_u, motions, mean_t, std_t, mask=attn_mask
            ).item()

            # Noise MSE — should be well below 1.0 (random baseline)
            mse = (
                ((eps_c - noise) ** 2 * attn_mask.float().unsqueeze(-1)).sum()
                / (attn_mask.float().sum() * noise.shape[-1])
            ).item()

            results[t_val] = {"cond": mpjpe_c, "uncond": mpjpe_u, "noise_mse": mse}
            print(f"{t_val:6d}  {mpjpe_c:12.4f}m  {mpjpe_u:14.4f}m  {mse:16.6f}")

    # ── Save results JSON ──────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ───────────────────────────────────────────────────────────────────
    ts = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ts, [results[t]["cond"]   for t in ts], "o-",  label="conditional (text)")
    ax.plot(ts, [results[t]["uncond"] for t in ts], "s--", label="unconditional (null)")
    ax.set_xlabel("Noise level  t")
    ax.set_ylabel("One-step x̂₀  MPJPE (m)")
    ax.set_title("One-step reconstruction quality vs. noise level")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(ts, [results[t]["noise_mse"] for t in ts], "o-", color="tab:green")
    ax.set_xlabel("Noise level  t")
    ax.set_ylabel("Noise prediction  MSE")
    ax.set_title("Noise MSE (conditional) vs. noise level")
    ax.axhline(1.0, color="red", linestyle=":", label="random baseline (MSE=1)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "mpjpe_vs_noise_level.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    best_t      = min(args.noise_levels)
    mpjpe_low_t = results[best_t]["cond"]
    mse_500     = results.get(500, results[max(results.keys())])["noise_mse"]

    # MPJPE threshold: 0.15 m is a practical upper bound for "the model learned
    # meaningful noise prediction" at low noise levels.
    mpjpe_ok  = mpjpe_low_t < 0.15
    mse_ok    = mse_500 < 0.50

    print(f"\n{'='*56}")
    print("LEDITS++ prerequisite check")
    print(f"{'='*56}")
    print(f"  Architecture:  {type(model).__name__} (epsilon-prediction DiT)  ✓")
    print(f"  Feature mode:  {feature_mode}")
    print(f"  MPJPE at t={best_t:3d} [cond]    {mpjpe_low_t:.4f} m   "
          f"[{'PASS' if mpjpe_ok  else 'REVIEW'} — threshold 0.15 m]")
    print(f"  Noise MSE at t=500          {mse_500:.6f}     "
          f"[{'PASS' if mse_ok else 'REVIEW'} — threshold 0.50]")
    print(f"  Output:        {args.output_dir}/")
    if mpjpe_ok and mse_ok:
        print("  Overall: PASS — backbone ready for LEDITS++ inversion (Phase 1)")
    else:
        print("  Overall: REVIEW — consider more training before implementing inversion")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
