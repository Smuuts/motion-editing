"""
Per-epoch train/validate loops for train.py.

Both loops compute the same Min-SNR-weighted (or plain) diffusion MSE and, when
enabled, the same MDM-style geometric losses — factored into _diffusion_loss and
_add_geo_losses so the train/val objectives can't silently drift apart.
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from utils.masks import length_to_mask


def _batch_lengths(batch, device):
    lengths = batch["length"]
    if isinstance(lengths, torch.Tensor):
        return lengths.detach().clone().to(device)
    return torch.as_tensor(lengths, device=device)


def _diffusion_loss(noise, prediction, attn_mask, schedule, t, snr_gamma):
    """Min-SNR-weighted (or plain) MSE between true and predicted noise, masked to
    valid (non-padding) frames. Shared by train_one_epoch and validate_one_epoch so
    the train/val curves are the same quantity and can be overlaid."""
    loss_mask = attn_mask.float().unsqueeze(-1)           # (B, T, 1)
    per_elem  = (noise - prediction) ** 2 * loss_mask     # (B, T, D)

    if snr_gamma > 0.0:
        # Per-sample mean MSE over valid (T, D) elements.
        valid_elems = (attn_mask.float().sum(dim=1) * noise.shape[-1]).clamp(min=1)
        per_sample  = per_elem.sum(dim=(1, 2)) / valid_elems        # (B,)
        # Min-SNR weight: min(SNR(t), γ) / SNR(t)
        snr_t      = schedule.snr[t]                                # (B,)
        snr_weight = snr_t.clamp(max=snr_gamma) / snr_t             # (B,)
        return (per_sample * snr_weight).mean()
    return per_elem.sum() / (loss_mask.sum() * noise.shape[-1]).clamp(min=1)


def _add_geo_losses(loss, x0_pred, motion, attn_mask, geo_fn, weights):
    """Add each active, finite MDM-style geometric loss term to `loss`.

    weights : {"pos": w, "vel": w, "foot": w} — a term is added only if its weight
              is > 0 and geo_fn returns a finite value for it (guards against NaN/Inf
              spikes at high noise timesteps).
    Returns (loss, contributions), where contributions holds the raw (unweighted)
    scalar values of the terms that were actually added, for epoch-level logging.
    """
    if geo_fn is None:
        return loss, {}
    geo = geo_fn(x0_pred, motion, attn_mask)
    contributions = {}
    for key, weight in weights.items():
        if weight > 0.0 and torch.isfinite(geo[key]):
            loss = loss + weight * geo[key]
            contributions[key] = geo[key].item()
    return loss, contributions


def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer, scaler,
    loader, device, cfg_dropout, logger, epoch, log_every,
    snr_gamma=5.0,
    geo_fn=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
):
    model.train()
    total_loss = 0.0
    total_geo  = {"pos": 0.0, "vel": 0.0, "foot": 0.0}
    geo_weights = {"pos": hml3d_pos_weight, "vel": hml3d_vel_weight, "foot": hml3d_foot_weight}

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

        attn_mask = length_to_mask(_batch_lengths(batch, device), motion.shape[1])

        with autocast(device_type=device.type):
            prediction = model(x_t, t, context, mask=attn_mask)
            loss = _diffusion_loss(noise, prediction, attn_mask, schedule, t, snr_gamma)

        geo_log = {}
        if geo_fn is not None:
            x0_pred = schedule.predict_x0_from_eps(x_t, t, prediction.float())
            loss, geo_contrib = _add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights)
            # geo_contrib holds RAW (unweighted) magnitudes for readability — what
            # actually reaches `loss` is geo_weights[key] * geo_contrib[key], which
            # is why e.g. "pos_raw" can be numerically larger than the total loss.
            for key, value in geo_contrib.items():
                geo_log[f"train/geo_{key}_raw"] = value
                total_geo[key] += value

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
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
    if geo_fn is not None:
        # Raw (unweighted) per-epoch means — see the comment above geo_log for why
        # these aren't directly comparable to `loss`.
        geo_epoch = {
            "train/geo_pos_raw_epoch":  total_geo["pos"]  / n,
            "train/geo_vel_raw_epoch":  total_geo["vel"]  / n,
            "train/geo_foot_raw_epoch": total_geo["foot"] / n,
        }
    return total_loss / n, geo_epoch


@torch.no_grad()
def validate_one_epoch(
    ema_model, text_encoder, schedule, loader, device, epoch,
    snr_gamma=5.0,
    geo_fn=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
):
    ema_model.eval()
    torch.manual_seed(0)
    total_loss = 0.0
    geo_weights = {"pos": hml3d_pos_weight, "vel": hml3d_vel_weight, "foot": hml3d_foot_weight}

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

        attn_mask = length_to_mask(_batch_lengths(batch, device), motion.shape[1])

        with autocast(device_type=device.type):
            prediction = ema_model(x_t, t, context, mask=attn_mask)
            # Mirror train_one_epoch's objective so the train/val curves are the
            # same quantity and can be overlaid in training_loss.png.
            loss = _diffusion_loss(noise, prediction, attn_mask, schedule, t, snr_gamma)

        if geo_fn is not None:
            x0_pred = schedule.predict_x0_from_eps(x_t, t, prediction.float())
            loss, _ = _add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights)

        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(loader)
