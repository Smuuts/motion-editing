import os
import argparse
import json

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
from model.text_encoder import CLIPTextEncoder
from model.schedule import NoiseSchedule
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ema import EMA
from utils.logger import Logger
from utils.skeleton import fk_position_loss, compute_mpjpe


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

    # diffusion
    p.add_argument("--timesteps",    type=int,   default=1000)
    p.add_argument("--cfg_dropout",  type=float, default=0.1,
                   help="Fraction of batch to train unconditionally (CFG).")

    # data
    p.add_argument("--feature_mode", type=str,   default="smpl",
                   choices=["humanml3d", "smpl", "group"],
                   help="'humanml3d' = full 263-dim; 'smpl' = 130-dim (root vel + body pose 6D); "
                        "'joint' = 130-dim split into per-joint tokens (22 tokens × F frames)")
    p.add_argument("--fk_loss_weight", type=float, default=0.1,
                   help="Weight of the FK position loss (SMPL/joint mode only). 0 = disabled.")

    # training
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
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
                   help="Path to a checkpoint directory to resume from.")
    return p.parse_args()


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


def save_mpjpe_graph(output_dir, val_mpjpe_losses):
    if not val_mpjpe_losses:
        return
    _save_figure(
        os.path.join(output_dir, "val_mpjpe.png"),
        "Validation MPJPE per Epoch", "MPJPE (m)",
        ([e for e, _ in val_mpjpe_losses], [v for _, v in val_mpjpe_losses],
         dict(marker="s", linestyle="-", color="tab:green")),
    )


def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer, scaler,
    loader, device, cfg_dropout, logger, epoch, log_every,
    smpl_stats=None, fk_loss_weight=0.0,
):
    model.train()
    total_loss = 0.0

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
            loss_mask = attn_mask.float().unsqueeze(-1)
            loss = ((noise - prediction) ** 2 * loss_mask).sum() / (loss_mask.sum() * noise.shape[-1])

            if smpl_stats is not None and fk_loss_weight > 0.0:
                mean_t, std_t = smpl_stats
                x0_pred = schedule.predict_x0_from_eps(x_t, t, prediction)
                fk_loss = fk_position_loss(x0_pred, motion, mean_t, std_t, mask=attn_mask)
                loss = loss + fk_loss_weight * fk_loss

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
            logger.log({"train/loss": loss.item(), "train/step": epoch * len(loader) + step})

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(ema_model, text_encoder, schedule, loader, device, epoch,
                       feature_stats=None, feature_mode="smpl"):
    ema_model.eval()
    torch.manual_seed(0)
    total_loss = 0.0
    total_mpjpe = 0.0

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
            loss = ((noise - prediction) ** 2 * loss_mask).sum() / (loss_mask.sum() * noise.shape[-1])

        # MPJPE: 10-step deterministic DDIM rollout from T//2.
        # At T//2 the cosine schedule gives alpha_bar ≈ 0.5 (50% signal)
        _n_steps = 10
        _t_start = schedule.T // 2
        _stride  = _t_start // _n_steps
        _steps   = list(range(_t_start, 0, -_stride))

        _tb_start = torch.full((B,), _t_start, device=device, dtype=torch.long)
        x_roll = schedule.q_sample(motion, _tb_start)[0]

        for _t in _steps:
            _t_prev = max(0, _t - _stride)
            _tb = torch.full((B,), _t, device=device, dtype=torch.long)
            with autocast(device_type=device.type):
                _eps = ema_model(x_roll, _tb, context, mask=attn_mask)
            _eps = _eps.float()
            _sqrt_acp   = schedule.sqrt_alphas_cumprod[_t]
            _sqrt_omacp = schedule.sqrt_one_minus_alphas_cumprod[_t]
            _x0 = ((x_roll.float() - _sqrt_omacp * _eps) / _sqrt_acp).clamp(-5, 5)
            if _t_prev > 0:
                _sqrt_acp_p   = schedule.sqrt_alphas_cumprod[_t_prev]
                _sqrt_omacp_p = schedule.sqrt_one_minus_alphas_cumprod[_t_prev]
                x_roll = _sqrt_acp_p * _x0 + _sqrt_omacp_p * _eps
            else:
                x_roll = _x0

        mean_t, std_t = feature_stats
        mpjpe = compute_mpjpe(x_roll, motion, mean_t, std_t, feature_mode, mask=attn_mask)

        total_loss += loss.item()
        total_mpjpe += mpjpe.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}", mpjpe=f"{mpjpe.item():.4f}")

    return total_loss / len(loader), total_mpjpe / len(loader)


def main():
    args = parse_args()
    print("Training MotionDiT with config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

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

    smpl_stats = None
    if args.feature_mode in ("smpl", "group") and args.fk_loss_weight > 0.0:
        smpl_stats = feature_stats
        print(f"FK position loss enabled (weight={args.fk_loss_weight})")

    # text encoder
    context_dim = 768 if "L/14" in args.clip_version else 512
    if train_loader.dataset.text_emb_dir is not None:
        print("Precomputed text embeddings found — skipping CLIP model load.")
        text_encoder = None
    else:
        text_encoder = CLIPTextEncoder(args.clip_version, device=device)

    # model
    model = build_model({
        "feature_mode": args.feature_mode,
        "input_dim":    train_loader.dataset.feature_dim,
        "latent_dim":   args.latent_dim,
        "context_dim":  context_dim,
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
    train_losses, val_losses, val_mpjpe_losses = [], [], []
    losses_path = os.path.join(args.output_dir, "losses.json")
    if args.resume:
        ckpt_dir = args.resume
        if ckpt_dir == "latest":
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        start_epoch = load_checkpoint(ckpt_dir, model, ema, optimizer, scheduler)
        if os.path.exists(losses_path):
            with open(losses_path) as f:
                saved = json.load(f)
            train_losses     = saved.get("train",     [])
            val_losses       = saved.get("val",       [])
            val_mpjpe_losses = saved.get("val_mpjpe", [])

    # training loop
    print(f"\nStarting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, ema, text_encoder, schedule, optimizer, scaler,
            train_loader, device, args.cfg_dropout,
            logger, epoch, args.log_every,
            smpl_stats=smpl_stats, fk_loss_weight=args.fk_loss_weight,
        )
        scheduler.step()
        train_losses.append((epoch, avg_loss))

        log_line = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            rng_cpu = torch.get_rng_state()
            rng_gpu = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            val_loss, val_mpjpe = validate_one_epoch(
                ema.ema_model, text_encoder, schedule, val_loader, device, epoch,
                feature_stats=feature_stats, feature_mode=args.feature_mode,
            )
            torch.set_rng_state(rng_cpu)
            if rng_gpu is not None:
                torch.cuda.set_rng_state(rng_gpu, device)
            val_losses.append((epoch, val_loss))
            val_mpjpe_losses.append((epoch, val_mpjpe))
            logger.log({"val/epoch_loss": val_loss, "val/epoch_mpjpe": val_mpjpe, "val/epoch": epoch})
            log_line += f" | val {val_loss:.4f} | mpjpe {val_mpjpe:.4f}m"

        with open(losses_path, "w") as f:
            json.dump({"train": train_losses, "val": val_losses, "val_mpjpe": val_mpjpe_losses}, f, indent=2)
        save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
        save_mpjpe_graph(args.output_dir, val_mpjpe_losses)
        logger.log({"train/epoch_loss": avg_loss, "train/epoch": epoch, "train/lr": scheduler.get_last_lr()[0]})
        print(log_line)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.output_dir, epoch, model, ema, optimizer, scheduler, config)

    save_checkpoint(args.output_dir, args.epochs - 1, model, ema, optimizer, scheduler, config)
    save_loss_graph(args.output_dir, train_losses, val_losses=val_losses)
    save_mpjpe_graph(args.output_dir, val_mpjpe_losses)
    print("Training complete.")


if __name__ == "__main__":
    main()
