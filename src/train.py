"""
Training script for MotionDiT.

Usage:
    python train.py --data_root /path/to/HumanML3D --output_dir ./runs/exp1

Key design decisions:
    - Epsilon (noise) prediction: required for LEDITS++ inversion
    - Cosine noise schedule: better motion structure preservation
    - Classifier-free guidance training: 10% of batches use null text
    - EMA model: used for inference and evaluation
    - Checkpoints saved every N epochs with full resume support
"""

import os
import argparse
import json
import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

import matplotlib.pyplot as plt

from data.dataset import build_dataloader
from model.dit import build_model
from model.text_encoder import CLIPTextEncoder
from model.schedule import NoiseSchedule
from utils.ema import EMA
from utils.logger import Logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True, default="./data/HumanML3D")
    p.add_argument("--output_dir",   default="./runs/exp1")
    p.add_argument("--config",       default=None,
                   help="Path to a JSON config file. CLI args override it.")

    # model
    p.add_argument("--latent_dim",   type=int,   default=512)
    p.add_argument("--num_layers",   type=int,   default=8)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--clip_version", type=str,   default="ViT-B/32")
    # ViT-B/32 -> context_dim=512, ViT-L/14 -> context_dim=768

    # diffusion
    p.add_argument("--timesteps",    type=int,   default=1000)
    p.add_argument("--cfg_dropout",  type=float, default=0.1,
                   help="Fraction of batch to train unconditionally (CFG).")

    # training
    p.add_argument("--epochs",       type=int,   default=1000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_frames",   type=int,   default=196)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--ema_decay",    type=float, default=0.9999)
    p.add_argument("--save_every",   type=int,   default=100)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--no_lr_decay",  action="store_true",
                   help="Keep learning rate constant (no decay)")
    p.add_argument("--joint_attn",   action="store_true", default=False,
                   help="Add per-frame spatial self-attention over joints inside each DiT block.")

    # resume
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to a checkpoint directory to resume from.")
    return p.parse_args()


def save_checkpoint(output_dir, epoch, model, ema, optimizer, scheduler, config):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(),     os.path.join(ckpt_dir, "model.pt"))
    torch.save(ema.ema_model.state_dict(), os.path.join(ckpt_dir, "ema.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({**config, "epoch": epoch}, f, indent=2)
    # keep a symlink to the latest checkpoint for easy resuming
    latest = os.path.join(output_dir, "checkpoint_latest")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(ckpt_dir, latest)
    print(f"  Saved checkpoint: {ckpt_dir}")


def save_loss_graph(output_dir, epoch_losses, start_epoch=0):
    if not epoch_losses:
        return

    epochs = list(range(start_epoch, start_epoch + len(epoch_losses)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, epoch_losses, marker='o', linestyle='-', color='tab:blue')
    ax.set_title('Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    graph_path = os.path.join(output_dir, 'training_loss.png')
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)


def load_checkpoint(ckpt_dir, model, ema, optimizer, scheduler):
    model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "model.pt"), weights_only=True))
    ema.ema_model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ema.pt"), weights_only=True))
    optimizer.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "optimizer.pt"), weights_only=True))
    scheduler.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "scheduler.pt"), weights_only=True))
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        saved = json.load(f)
    start_epoch = saved["epoch"] + 1
    print(f"  Resumed from epoch {saved['epoch']}")
    return start_epoch


def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer,
    loader, device, cfg_dropout, logger, epoch, log_every,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        motion = batch["motion"].to(device)  # (B, F, 263)
        B = motion.shape[0]

        # ── sample random timesteps ────────────────────────────────────
        t = torch.randint(0, schedule.T, (B,), device=device)

        # ── add noise (forward process) ────────────────────────────────
        x_t, noise = schedule.q_sample(motion, t)  # both (B, F, 263)

        # ── get text context (precomputed or live CLIP encode) ─────────
        # null_text_emb (not all-zeros) must receive gradients so guidance
        # scale > 1 works correctly at inference.
        if "context" in batch:
            context = batch["context"].to(device)   # (B, 77, context_dim)
        else:
            with torch.no_grad():
                context = text_encoder.encode(batch["text"])  # (B, 77, context_dim)
        if cfg_dropout > 0.0:
            drop_mask = (torch.rand(B, device=device) < cfg_dropout)[:, None, None]
            null_emb  = model.null_text_emb.expand(B, -1, -1)  # has gradient
            context   = torch.where(drop_mask, null_emb, context)

        # ── create padding mask for variable-length clips ───────────────
        lengths = batch["length"]
        if isinstance(lengths, torch.Tensor):
            lengths_tensor = lengths.detach().clone().to(device)
        else:
            lengths_tensor = torch.as_tensor(lengths, device=device)
        attn_mask = torch.arange(motion.shape[1], device=device)[None, :] < lengths_tensor[:, None]

        # ── predict noise ──────────────────────────────────────────────
        eps_pred = model(x_t, t, context, mask=attn_mask)  # (B, F, 263)

        # ── loss: masked MSE — only penalise real frames, not padding ──
        loss_mask = attn_mask.float().unsqueeze(-1)  # (B, F, 1)
        loss = ((noise - eps_pred) ** 2 * loss_mask).sum() / (loss_mask.sum() * noise.shape[-1])

        # ── optimise ───────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA must be updated every step, not every epoch.
        # With decay=0.9999, updating once per epoch (100 calls total)
        # leaves the shadow at 0.9999^100 ≈ 0.99× its initial random weights.
        ema.update_from(model)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (step + 1) % log_every == 0:
            logger.log({
                "train/loss": loss.item(),
                "train/step": epoch * len(loader) + step,
            })

    return total_loss / len(loader)


def main():
    args = parse_args()
    print("Training MotionDiT with config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ── config ────────────────────────────────────────────────────────
    config = vars(args)
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
        config.update(file_cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── data ──────────────────────────────────────────────────────────
    train_loader = build_dataloader(
        args.data_root, split="train",
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )
    print(f"Training on {len(train_loader.dataset)} clips")

    # ── text encoder ──────────────────────────────────────────────────
    context_dim = 768 if "L/14" in args.clip_version else 512
    if train_loader.dataset.text_emb_dir is not None:
        print("Precomputed text embeddings found — skipping CLIP model load.")
        text_encoder = None
    else:
        text_encoder = CLIPTextEncoder(args.clip_version, device=device)

    # ── model ─────────────────────────────────────────────────────────
    model_config = {
        "input_dim":      263,
        "latent_dim":     args.latent_dim,
        "context_dim":    context_dim,
        "num_heads":      args.num_heads,
        "num_layers":     args.num_layers,
        "max_frames":     args.max_frames,
        "dropout":        args.dropout,
        "use_joint_attn": args.joint_attn,
    }
    model = build_model(model_config, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # ── EMA ───────────────────────────────────────────────────────────
    ema = EMA(model, decay=args.ema_decay)

    # ── noise schedule ────────────────────────────────────────────────
    schedule = NoiseSchedule(timesteps=args.timesteps, device=device)

    # ── optimizer & scheduler ─────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    if args.no_lr_decay:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── logger ────────────────────────────────────────────────────────
    logger = Logger(args.output_dir)

    # ── resume ────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt_dir = args.resume
        if ckpt_dir == "latest":
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        start_epoch = load_checkpoint(ckpt_dir, model, ema, optimizer, scheduler)

    # ── training loop ─────────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch}")
    epoch_losses = []
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, ema, text_encoder, schedule, optimizer,
            train_loader, device, args.cfg_dropout,
            logger, epoch, args.log_every,
        )
        scheduler.step()

        epoch_losses.append(avg_loss)
        save_loss_graph(args.output_dir, epoch_losses, start_epoch=start_epoch)

        logger.log({
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch,
            "train/lr": scheduler.get_last_lr()[0],
        })
        print(f"Epoch {epoch:4d} | loss {avg_loss:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.output_dir, epoch, model, ema,
                optimizer, scheduler, config,
            )

    # final checkpoint
    save_checkpoint(args.output_dir, args.epochs - 1,
                    model, ema, optimizer, scheduler, config)
    save_loss_graph(args.output_dir, epoch_losses, start_epoch=start_epoch)
    print("Training complete.")


if __name__ == "__main__":
    main()
