import os
import argparse
import json
import glob

import numpy as np

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import build_dataloader
from model.dit import build_model
from model.text_encoder import build_text_encoder, get_encoder_dims
from model.schedule import NoiseSchedule
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ema import EMA
from utils.logger import Logger
from utils.skeleton import hml3d_geometric_losses


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
                   choices=["humanml3d", "group"],
                   help="'humanml3d' = full 263-dim flat (MotionDiT); "
                        "'group' = 263-dim partitioned into per-body-part group tokens (GroupDiT)")
    p.add_argument("--hml3d_pos_weight",  type=float, default=0.1,
                   help="Weight for MDM L_pos joint position loss (humanml3d mode only). 0 = disabled.")
    p.add_argument("--hml3d_vel_weight",  type=float, default=0.1,
                   help="Weight for MDM L_vel velocity consistency loss (humanml3d mode only). 0 = disabled.")
    p.add_argument("--hml3d_foot_weight", type=float, default=0.01,
                   help="Weight for MDM L_foot contact loss (humanml3d mode only). 0 = disabled.")
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


def explicit_cli_keys():
    """Return the set of arg dest names that were explicitly passed on the CLI."""
    p = build_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    return set(vars(p.parse_args()).keys())


def _save_figure(path, title, ylabel, *series):
    """series: (epochs, values, plot_kwargs) tuples."""
    if not series or not series[0][0]:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for epochs, values, kwargs in series:
        ax.plot(epochs, values, **kwargs)
    if len(series) > 1:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_graph(output_dir, train_losses, val_losses=None):
    series = [
        ([e for e, _ in train_losses], [v for _, v in train_losses],
         dict(marker="o", linestyle="-", color="tab:blue", label="train")),
    ]
    if val_losses:
        series.append(
            ([e for e, _ in val_losses], [v for _, v in val_losses],
             dict(marker="s", linestyle="--", color="tab:orange", label="val"))
        )
    _save_figure(os.path.join(output_dir, "training_loss.png"), "Loss per Epoch", "Average Loss", *series)



def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer, scaler,
    loader, device, cfg_dropout, logger, epoch, log_every,
    snr_gamma=5.0,
    hml3d_stats=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
):
    model.train()
    total_loss = 0.0
    total_geo  = {"pos": 0.0, "vel": 0.0, "foot": 0.0}

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        motion = batch["motion"].to(device)
        B = motion.shape[0]

        t = torch.randint(0, schedule.T, (B,), device=device)
        x_t, noise = schedule.q_sample(motion, t)

        if "context" in batch:
            context = batch["context"].to(device)
        else:
            with torch.no_grad():
                context = text_encoder.encode(batch["text"])
        if cfg_dropout > 0.0:
            drop_mask = (torch.rand(B, device=device) < cfg_dropout)[:, None, None]
            # null_text_emb must keep its gradient so CFG scale > 1 works at inference.
            # CFG dropout also trains the unconditional branch used by LEDITS++ Stage 1
            # inversion (context=None) and the ε_θ(x_t, ∅) term in Stage 3 Eq. 1.
            null_emb  = model.null_text_emb.expand(B, -1, -1)
            context   = torch.where(drop_mask, null_emb, context)

        lengths = batch["length"]
        if isinstance(lengths, torch.Tensor):
            lengths_tensor = lengths.detach().clone().to(device)
        else:
            lengths_tensor = torch.as_tensor(lengths, device=device)
        attn_mask = torch.arange(motion.shape[1], device=device)[None, :] < lengths_tensor[:, None]

        with autocast(device_type=device.type):
            prediction = model(x_t, t, context, mask=attn_mask)
            loss_mask = attn_mask.float().unsqueeze(-1)           # (B, T, 1)
            per_elem  = (noise - prediction) ** 2 * loss_mask    # (B, T, D)

            if snr_gamma > 0.0:
                # Per-sample mean MSE over valid (T, D) elements.
                valid_elems = (attn_mask.float().sum(dim=1) * noise.shape[-1]).clamp(min=1)
                per_sample  = per_elem.sum(dim=(1, 2)) / valid_elems        # (B,)
                # Min-SNR weight: min(SNR(t), γ) / SNR(t)
                snr_t      = schedule.snr[t]                                # (B,)
                snr_weight = snr_t.clamp(max=snr_gamma) / snr_t            # (B,)
                loss = (per_sample * snr_weight).mean()
            else:
                loss = per_elem.sum() / (loss_mask.sum() * noise.shape[-1]).clamp(min=1)

        geo_log = {}
        if hml3d_stats is not None:
            x0_pred = schedule.predict_x0_from_eps(x_t, t, prediction.float())
            mean_h, std_h = hml3d_stats
            geo = hml3d_geometric_losses(x0_pred, motion.float(), mean_h, std_h, mask=attn_mask)
            if hml3d_pos_weight  > 0.0 and torch.isfinite(geo["pos"]):
                loss = loss + hml3d_pos_weight  * geo["pos"]
                geo_log["train/geo_pos"]  = geo["pos"].item()
                total_geo["pos"]  += geo["pos"].item()
            if hml3d_vel_weight  > 0.0 and torch.isfinite(geo["vel"]):
                loss = loss + hml3d_vel_weight  * geo["vel"]
                geo_log["train/geo_vel"]  = geo["vel"].item()
                total_geo["vel"]  += geo["vel"].item()
            if hml3d_foot_weight > 0.0 and torch.isfinite(geo["foot"]):
                loss = loss + hml3d_foot_weight * geo["foot"]
                geo_log["train/geo_foot"] = geo["foot"].item()
                total_geo["foot"] += geo["foot"].item()

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update_from(model)  # per-step update so decay=0.9999 accumulates correctly

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (step + 1) % log_every == 0:
            logger.log({"train/loss": loss.item(), "train/step": epoch * len(loader) + step, **geo_log})

    n = len(loader)
    geo_epoch = {}
    if hml3d_stats is not None:
        geo_epoch = {
            "train/geo_pos_epoch":  total_geo["pos"]  / n,
            "train/geo_vel_epoch":  total_geo["vel"]  / n,
            "train/geo_foot_epoch": total_geo["foot"] / n,
        }
    return total_loss / n, geo_epoch


@torch.no_grad()
def validate_one_epoch(
    ema_model, text_encoder, schedule, loader, device, epoch,
    snr_gamma=5.0,
    hml3d_stats=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
):
    ema_model.eval()
    torch.manual_seed(0)
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Val {epoch}", leave=False)
    for batch in pbar:
        motion = batch["motion"].to(device)
        B = motion.shape[0]

        t = torch.randint(0, schedule.T, (B,), device=device)
        x_t, noise = schedule.q_sample(motion, t)

        if "context" in batch:
            context = batch["context"].to(device)
        else:
            context = text_encoder.encode(batch["text"])

        lengths = batch["length"]
        if isinstance(lengths, torch.Tensor):
            lengths_tensor = lengths.detach().clone().to(device)
        else:
            lengths_tensor = torch.as_tensor(lengths, device=device)
        attn_mask = torch.arange(motion.shape[1], device=device)[None, :] < lengths_tensor[:, None]

        with autocast(device_type=device.type):
            prediction = ema_model(x_t, t, context, mask=attn_mask)
            loss_mask = attn_mask.float().unsqueeze(-1)
            per_elem  = (noise - prediction) ** 2 * loss_mask

            # Mirror train_one_epoch's objective so the train/val curves are the
            # same quantity and can be overlaid in training_loss.png.
            if snr_gamma > 0.0:
                valid_elems = (attn_mask.float().sum(dim=1) * noise.shape[-1]).clamp(min=1)
                per_sample  = per_elem.sum(dim=(1, 2)) / valid_elems
                snr_t       = schedule.snr[t]
                snr_weight  = snr_t.clamp(max=snr_gamma) / snr_t
                loss = (per_sample * snr_weight).mean()
            else:
                loss = per_elem.sum() / (loss_mask.sum() * noise.shape[-1]).clamp(min=1)

        if hml3d_stats is not None:
            x0_pred = schedule.predict_x0_from_eps(x_t, t, prediction.float())
            mean_h, std_h = hml3d_stats
            geo = hml3d_geometric_losses(x0_pred, motion.float(), mean_h, std_h, mask=attn_mask)
            if hml3d_pos_weight  > 0.0 and torch.isfinite(geo["pos"]):
                loss = loss + hml3d_pos_weight  * geo["pos"]
            if hml3d_vel_weight  > 0.0 and torch.isfinite(geo["vel"]):
                loss = loss + hml3d_vel_weight  * geo["vel"]
            if hml3d_foot_weight > 0.0 and torch.isfinite(geo["foot"]):
                loss = loss + hml3d_foot_weight * geo["foot"]

        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    args = parse_args()

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    # On resume, treat the saved config in --output_dir as the base so model
    # architecture etc. need not be re-specified. Explicit CLI args still win,
    # and resume/output_dir are kept so the resume itself isn't clobbered.
    if args.resume:
        saved_path = os.path.join(args.output_dir, "config.json")
        if os.path.exists(saved_path):
            with open(saved_path) as f:
                saved = json.load(f)
            cli_keys = explicit_cli_keys()
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
    feature_stats = (mean_t, std_t)

    hml3d_geo_stats = None
    if any([args.hml3d_pos_weight, args.hml3d_vel_weight, args.hml3d_foot_weight]):
        hml3d_geo_stats = feature_stats
        parts = [f"pos={args.hml3d_pos_weight}", f"vel={args.hml3d_vel_weight}", f"foot={args.hml3d_foot_weight}"]
        print(f"HumanML3D geometric losses enabled ({', '.join(p for p in parts if not p.endswith('=0.0'))})")

    # text encoder
    context_dim, text_seq_len = get_encoder_dims(config)
    if train_loader.dataset.text_emb_dir is not None:
        sample_files = glob.glob(os.path.join(train_loader.dataset.text_emb_dir, "*.npy"))[:1]
        if sample_files:
            sample_emb = np.load(sample_files[0])  # (num_ann, L, dim)
            emb_seq_len, emb_dim = int(sample_emb.shape[1]), int(sample_emb.shape[2])
            if emb_seq_len != text_seq_len or emb_dim != context_dim:
                raise ValueError(
                    f"Precomputed embeddings in '{train_loader.dataset.text_emb_dir}' have shape "
                    f"(*, {emb_seq_len}, {emb_dim}), but the configured encoder "
                    f"(--text_encoder {config.get('text_encoder', 'clip')}) expects "
                    f"(*, {text_seq_len}, {context_dim}). "
                    f"Re-run precompute_text.py with matching encoder settings, or "
                    f"point --data_root to a directory without a stale text_emb/ folder."
                )
        print(f"Precomputed text embeddings found — skipping {config.get('text_encoder', 'clip').upper()} model load.")
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

    # LEDITS++ uses the EMA model for all inference passes (inversion + editing).
    # Load from checkpoint_latest/ema.pt, not model.pt.
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
            hml3d_stats=hml3d_geo_stats,
            hml3d_pos_weight=args.hml3d_pos_weight,
            hml3d_vel_weight=args.hml3d_vel_weight,
            hml3d_foot_weight=args.hml3d_foot_weight,
        )
        scheduler.step()
        train_losses.append((epoch, avg_loss))

        log_line = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        if geo_epoch:
            log_line += f" | pos {geo_epoch['train/geo_pos_epoch']:.4f} | vel {geo_epoch['train/geo_vel_epoch']:.4f} | foot {geo_epoch['train/geo_foot_epoch']:.4f}"

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            rng_cpu = torch.get_rng_state()
            rng_gpu = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            val_loss = validate_one_epoch(
                ema.ema_model, text_encoder, schedule, val_loader, device, epoch,
                snr_gamma=args.snr_gamma,
                hml3d_stats=hml3d_geo_stats,
                hml3d_pos_weight=args.hml3d_pos_weight,
                hml3d_vel_weight=args.hml3d_vel_weight,
                hml3d_foot_weight=args.hml3d_foot_weight,
            )
            torch.set_rng_state(rng_cpu)
            if rng_gpu is not None:
                torch.cuda.set_rng_state(rng_gpu, device)
            val_losses.append((epoch, val_loss))
            logger.log({"val/epoch_loss": val_loss, "val/epoch": epoch})
            log_line += f" | val {val_loss:.4f}"

        with open(losses_path, "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f, indent=2)
        save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
        logger.log({"train/epoch_loss": avg_loss, "train/epoch": epoch, "train/lr": scheduler.get_last_lr()[0], **geo_epoch})
        print(log_line)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.output_dir, epoch, model, ema, optimizer, scheduler, config)

    save_checkpoint(args.output_dir, args.epochs - 1, model, ema, optimizer, scheduler, config)
    save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
    print("Training complete.")


if __name__ == "__main__":
    main()
