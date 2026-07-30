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


def _diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma):
    """Min-SNR-weighted (or plain) MSE between the target and the network output,
    masked to valid (non-padding) frames. Shared by train_one_epoch and
    validate_one_epoch so the train/val curves are the same quantity and can be
    overlaid.

    `target` is whatever the network predicts — ε or the clean signal x0, per
    `schedule.predict_type` (use `schedule.diffusion_target(x0, noise)` to pick it).
    `min_snr_weight` flips form to match, so the caller never has to know.
    """
    loss_mask = attn_mask.float().unsqueeze(-1)           # (B, T, 1)
    per_elem  = (target - prediction) ** 2 * loss_mask    # (B, T, D)

    if snr_gamma > 0.0:
        # Per-sample mean MSE over valid (T, D) elements.
        valid_elems = (attn_mask.float().sum(dim=1) * target.shape[-1]).clamp(min=1)
        per_sample  = per_elem.sum(dim=(1, 2)) / valid_elems        # (B,)
        snr_weight  = schedule.min_snr_weight(t, snr_gamma)         # (B,)
        return (per_sample * snr_weight).mean()
    return per_elem.sum() / (loss_mask.sum() * target.shape[-1]).clamp(min=1)


def _add_geo_losses(loss, x0_pred, motion, attn_mask, geo_fn, weights, sample_weight=None):
    """Add each active, finite MDM-style geometric loss term to `loss`.

    weights : {"pos": w, "vel": w, "foot": w} — a term is added only if its weight
              is > 0 and geo_fn returns a finite value for it (guards against NaN/Inf
              spikes at high noise timesteps).
    sample_weight : (B,) optional — see `NoiseSchedule.x0_confidence_weight`. Down-weights
              samples where x0_pred is an unreliable estimate of the clean signal.
    Returns (loss, contributions), where contributions holds the raw (unweighted)
    scalar values of the terms that were actually added, for epoch-level logging.
    """
    if geo_fn is None:
        return loss, {}
    geo = geo_fn(x0_pred, motion, attn_mask, sample_weight=sample_weight)
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
    attn_entropy_weight=0.0, geo_conf_weight=True,
):
    model.train()
    total_loss = 0.0
    total_geo  = {"pos": 0.0, "vel": 0.0, "foot": 0.0}
    geo_weights = {"pos": hml3d_pos_weight, "vel": hml3d_vel_weight, "foot": hml3d_foot_weight}
    use_entropy = attn_entropy_weight > 0.0

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

        # Entropy regulariser on ONE random block per step: materialising attention
        # in all 8 blocks at once OOMs a 12 GB GPU, and in expectation every layer
        # still receives the same pressure.
        ent_layer = (int(torch.randint(len(model.blocks), (1,)).item())
                     if use_entropy else None)
        with autocast(device_type=device.type):
            prediction = model(x_t, t, context, mask=attn_mask,
                               entropy_layer=ent_layer)
            target = schedule.diffusion_target(motion, noise)
            loss = _diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma)

        entropy_log = {}
        if ent_layer is not None:
            # Normalised real-token attention entropy in [0, 1]; MAXIMISED
            # (subtracted) to discourage queries collapsing onto a single key —
            # see docs/LEDITSpp_Attention_Sink_Research.md §9.
            h_attn = model.get_attn_entropy(ent_layer)
            loss = loss - attn_entropy_weight * h_attn
            entropy_log["train/attn_entropy"] = h_attn.item()

        geo_log = {}
        if geo_fn is not None:
            # to_x0 is a no-op for an x0-head (its output already IS x̂0) and the
            # 1/√ᾱ_t conversion for an eps-head. geo_conf_weight follows: the ᾱ_t
            # damping exists only to fade out that conversion's error amplification,
            # so it is off by default under x0 (see schedule.x0_confidence_weight).
            x0_pred = schedule.to_x0(prediction.float(), x_t, t)
            sample_weight = schedule.x0_confidence_weight(t) if geo_conf_weight else None
            loss, geo_contrib = _add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights,
                sample_weight=sample_weight)
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
            logger.log({"train/loss": loss.item(), "train/step": epoch * len(loader) + step,
                        **geo_log, **entropy_log})

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
    geo_conf_weight=True,
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
            target = schedule.diffusion_target(motion, noise)
            loss = _diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma)

        if geo_fn is not None:
            x0_pred = schedule.to_x0(prediction.float(), x_t, t)
            sample_weight = schedule.x0_confidence_weight(t) if geo_conf_weight else None
            loss, _ = _add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights,
                sample_weight=sample_weight)

        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(loader)
