import os
# Must be set before any CUDA context is created (i.e. before the first .cuda() call,
# not necessarily before `import torch`). Mitigates the allocator fragmentation that
# PyTorch itself suggests on OOM; only applies if the server env doesn't already set it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import glob
import sys

import numpy as np

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

from data.dataset import build_dataloader
from model.dit import build_model
from model.text_encoder import build_text_encoder, get_encoder_dims
from model.schedule import NoiseSchedule
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ema import EMA
from utils.logger import Logger
from utils.skeleton import build_geo_fn
from training.epoch import train_one_epoch, validate_one_epoch
from training.plotting import save_loss_graph


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True, default="./data/HumanML3D")
    p.add_argument("--output_dir",   default="./runs/exp_1")
    p.add_argument("--config",       default=None,
                   help="Path to a JSON config file. CLI args override it.")

    # model
    p.add_argument("--latent_dim",   type=int,   default=512)
    p.add_argument("--num_layers",   type=int,   default=8)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--clip_version", type=str,   default="ViT-B/32")
    p.add_argument("--text_encoder", type=str,   default="t5",
                   choices=["clip", "t5"],
                   help="Text encoder backend: 'clip' or 't5' (default).")
    p.add_argument("--t5_version",   type=str,   default="t5-base",
                   help="T5 model name (e.g. t5-base, t5-large). Used when --text_encoder=t5.")
    p.add_argument("--t5_max_length", type=int,  default=128,
                   help="Fixed token sequence length for T5 output. Used when --text_encoder=t5.")

    # diffusion
    p.add_argument("--timesteps",    type=int,   default=1000)
    p.add_argument("--cfg_dropout",  type=float, default=0.1,
                   help="Fraction of batch to train unconditionally (CFG).")

    # data
    p.add_argument("--feature_mode", type=str,   default="humanml3d",
                   choices=["humanml3d", "smplh"],
                   help="Both modes are body-part-grouped (GroupDiT). 'humanml3d' = 263-d HumanML3D "
                        "features; 'smplh' = 135-d SMPL-H features (from src/data/amass_to_smplh.py). "
                        "Both read features from data_root/new_joint_vecs/ and texts from "
                        "data_root/texts/.")
    p.add_argument("--smplh_model_path", type=str,
                   default="data/motionfix/data/body_models/smplh",
                   help="SMPLHLayer dir (smplh mode geometric losses). Needs SMPLH_NEUTRAL.npz.")
    p.add_argument("--hml3d_pos_weight",  type=float, default=0.1,
                   help="Weight for MDM L_pos joint position loss (humanml3d: stored joints; "
                        "smplh: SMPL-FK joints). 0 = disabled.")
    p.add_argument("--hml3d_vel_weight",  type=float, default=0.1,
                   help="Weight for MDM L_vel velocity consistency loss. 0 = disabled.")
    p.add_argument("--hml3d_foot_weight", type=float, default=0.01,
                   help="Weight for MDM L_foot contact loss. 0 = disabled.")
    p.add_argument("--snr_gamma",     type=float, default=5.0,
                   help="Min-SNR weighting gamma (Hang et al. 2023). 0 = disabled.")

    # training
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_frames",   type=int,   default=196)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--ema_decay",    type=float, default=0.9999)
    p.add_argument("--save_every",   type=int,   default=100)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--val_every",    type=int,   default=1,
                   help="Compute validation loss every N epochs (0 = disabled).")
    p.add_argument("--warmup_epochs", type=int,  default=5,
                   help="Number of epochs to linearly warm up the LR from 1%% to target.")
    p.add_argument("--no_lr_decay",  action="store_true",
                   help="Keep learning rate constant (no decay)")

    # resume
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to a checkpoint directory to resume from. "
                        "When set, missing args are filled from the saved "
                        "config.json in --output_dir (explicit CLI args still win).")
    return p


def parse_args():
    return build_parser().parse_args()


def explicit_cli_keys(parser):
    """Return the set of arg dest names that were explicitly passed on the CLI."""
    argv = sys.argv[1:]
    return {a.dest for a in parser._actions if any(o in argv for o in a.option_strings)}


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    if args.resume:
        saved_path = os.path.join(args.output_dir, "config.json")
        if os.path.exists(saved_path):
            with open(saved_path) as f:
                saved = json.load(f)
            cli_keys = explicit_cli_keys(parser)
            preserved = cli_keys | {"resume", "output_dir"}
            for k, v in saved.items():
                if k not in preserved:
                    config[k] = v
            print(f"Resume: loaded config from {saved_path} "
                  f"(overridden by CLI args: {sorted(cli_keys - {'resume', 'output_dir'})})")
        else:
            print(f"Resume: no saved config at {saved_path}, using CLI args only.")

    print("Training MotionDiT with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    loader_kwargs = dict(
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        feature_mode=args.feature_mode,
    )
    train_loader = build_dataloader(args.data_root, split="train", **loader_kwargs)
    print(f"Training on {len(train_loader.dataset)} clips")

    val_loader = None
    if args.val_every > 0:
        val_loader = build_dataloader(args.data_root, split="val", **loader_kwargs)
        print(f"Validation on {len(val_loader.dataset)} clips (every {args.val_every} epochs)")

    ds = train_loader.dataset
    mean_t = torch.from_numpy(ds.mean).float().to(device)
    std_t  = torch.from_numpy(ds.std).float().to(device)

    # Geometric losses: route to the rep-specific implementation. geo_fn is a closure
    # (x0_pred, motion, mask) -> {"pos","vel","foot"} so run/validate stay rep-agnostic.
    geo_fn, geo_label = build_geo_fn(
        args.feature_mode, mean_t, std_t, device,
        pos_weight=args.hml3d_pos_weight, vel_weight=args.hml3d_vel_weight,
        foot_weight=args.hml3d_foot_weight, smplh_model_path=args.smplh_model_path,
    )
    if geo_fn is not None:
        weights = (("pos", args.hml3d_pos_weight), ("vel", args.hml3d_vel_weight),
                   ("foot", args.hml3d_foot_weight))
        parts = [f"{name}={w}" for name, w in weights if w]
        print(f"{geo_label} geometric losses enabled ({', '.join(parts)})")

    # text encoder
    context_dim, text_seq_len = get_encoder_dims(config)
    if train_loader.dataset.text_emb_dir is not None:
        sample_file = next(glob.iglob(os.path.join(train_loader.dataset.text_emb_dir, "*.npy")), None)
        if sample_file:
            sample_emb = np.load(sample_file)  # (num_ann, L, dim)
            emb_seq_len, emb_dim = int(sample_emb.shape[1]), int(sample_emb.shape[2])
            if emb_seq_len != text_seq_len or emb_dim != context_dim:
                raise ValueError(
                    f"Precomputed embeddings in '{train_loader.dataset.text_emb_dir}' have shape "
                    f"(*, {emb_seq_len}, {emb_dim}), but the configured encoder "
                    f"(--text_encoder {config['text_encoder']}) expects "
                    f"(*, {text_seq_len}, {context_dim}). "
                    f"Re-run precompute_text.py with matching encoder settings, or "
                    f"point --data_root to a directory without a stale text_emb/ folder."
                )
        print(f"Precomputed text embeddings found — skipping {config['text_encoder'].upper()} model load.")
        text_encoder = None
    else:
        text_encoder = build_text_encoder(config, device=device)

    # model
    model = build_model({
        "feature_mode": args.feature_mode,
        "input_dim":    train_loader.dataset.feature_dim,
        "latent_dim":   args.latent_dim,
        "context_dim":  context_dim,
        "text_seq_len": text_seq_len,
        "num_heads":    args.num_heads,
        "num_layers":   args.num_layers,
        "max_frames":   args.max_frames,
        "dropout":      args.dropout,
    }, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    ema      = EMA(model, decay=args.ema_decay)
    schedule = NoiseSchedule(timesteps=args.timesteps, device=device)
    scaler   = GradScaler(device=device.type, enabled=device.type == "cuda")

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    if args.no_lr_decay:
        main_sched = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        main_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    if args.warmup_epochs > 0:
        warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[args.warmup_epochs])
    else:
        scheduler = main_sched

    logger = Logger(args.output_dir)

    # resume
    start_epoch = 0
    train_losses, val_losses = [], []
    losses_path = os.path.join(args.output_dir, "losses.json")
    if args.resume:
        ckpt_dir = args.resume
        if ckpt_dir == "latest":
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        start_epoch = load_checkpoint(ckpt_dir, model, ema, optimizer, scheduler)
        if os.path.exists(losses_path):
            with open(losses_path) as f:
                saved = json.load(f)
            train_losses = saved.get("train", [])
            val_losses   = saved.get("val",   [])

    # training loop
    print(f"\nStarting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        avg_loss, geo_epoch = train_one_epoch(
            model, ema, text_encoder, schedule, optimizer, scaler,
            train_loader, device, args.cfg_dropout,
            logger, epoch, args.log_every,
            snr_gamma=args.snr_gamma,
            geo_fn=geo_fn,
            hml3d_pos_weight=args.hml3d_pos_weight,
            hml3d_vel_weight=args.hml3d_vel_weight,
            hml3d_foot_weight=args.hml3d_foot_weight,
        )
        # Read the LR used for the epoch that just ran before advancing the
        # scheduler — get_last_lr() after step() would report next epoch's LR.
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        train_losses.append((epoch, avg_loss))

        log_line = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {lr_used:.2e}"
        if geo_epoch:
            # "_raw" = unweighted magnitude in metres-ish units, not what's added to `loss`
            # (loss adds hml3d_{pos,vel,foot}_weight * this value) — see training/epoch.py.
            log_line += (f" | pos_raw {geo_epoch['train/geo_pos_raw_epoch']:.4f}"
                         f" | vel_raw {geo_epoch['train/geo_vel_raw_epoch']:.4f}"
                         f" | foot_raw {geo_epoch['train/geo_foot_raw_epoch']:.4f}")

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                val_loss = validate_one_epoch(
                    ema.ema_model, text_encoder, schedule, val_loader, device, epoch,
                    snr_gamma=args.snr_gamma,
                    geo_fn=geo_fn,
                    hml3d_pos_weight=args.hml3d_pos_weight,
                    hml3d_vel_weight=args.hml3d_vel_weight,
                    hml3d_foot_weight=args.hml3d_foot_weight,
                )
            val_losses.append((epoch, val_loss))
            logger.log({"val/epoch_loss": val_loss, "val/epoch": epoch})
            log_line += f" | val {val_loss:.4f}"

        with open(losses_path, "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f, indent=2)
        save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
        logger.log({"train/epoch_loss": avg_loss, "train/epoch": epoch, "train/lr": lr_used, **geo_epoch})
        print(log_line)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.output_dir, epoch, model, ema, optimizer, scheduler, config)

    save_checkpoint(args.output_dir, args.epochs - 1, model, ema, optimizer, scheduler, config)
    save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
    print("Training complete.")


if __name__ == "__main__":
    main()
