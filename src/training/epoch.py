"""
Per-epoch train and validate loops for train.py.

Both share their objective with `training/losses.py`, so the two curves stay the same
quantity. The only asymmetries are deliberate: validation uses the EMA weights, a fixed
seed, a deterministic grounding layer, and no CFG dropout.
"""

import torch
import torch.nn as nn
from torch.amp import autocast

from training.losses import (MAX_SKIP_FRACTION, GroundingAccumulator, add_geo_losses,
                             add_grounding_loss, apply_cfg_dropout, batch_lengths,
                             diffusion_loss)
from utils.logger import get_logger
from utils.padding import length_to_mask

log = get_logger(__name__)


def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer, scaler,
    loader, device, cfg_dropout, run_logger, epoch, log_every,
    snr_gamma=5.0,
    geo_fn=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
    attn_entropy_weight=0.0, geo_conf_weight=True, amp_dtype=torch.float16,
    grounding=None,
):
    model.train()
    total_loss = 0.0
    n_counted  = 0          # steps that actually contributed to total_loss
    n_skipped  = 0          # steps dropped for a non-finite loss (see the skip path)
    total_geo  = {"pos": 0.0, "vel": 0.0, "foot": 0.0}
    geo_weights = {"pos": hml3d_pos_weight, "vel": hml3d_vel_weight, "foot": hml3d_foot_weight}
    use_entropy = attn_entropy_weight > 0.0
    use_ground  = grounding is not None and grounding.active(epoch)
    ground_acc  = GroundingAccumulator()

    pbar = log.progress(loader, desc=f"Epoch {epoch}")
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
        context, kept = apply_cfg_dropout(model, context, cfg_dropout)

        attn_mask = length_to_mask(batch_lengths(batch, device), motion.shape[1])

        # Entropy regulariser on ONE random block per step: materialising attention
        # in all 8 blocks at once OOMs a 12 GB GPU, and in expectation every layer
        # still receives the same pressure.
        ent_layer = (int(torch.randint(len(model.blocks), (1,)).item())
                     if use_entropy else None)
        ground_layer = grounding.pick_layer() if use_ground else None
        with autocast(device_type=device.type, dtype=amp_dtype):
            prediction = model(x_t, t, context, mask=attn_mask,
                               entropy_layer=ent_layer,
                               supervise_layer=ground_layer)
            target = schedule.diffusion_target(motion, noise)
            loss = diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma)

        A = l_ground = None
        ground_log = {}
        if ground_layer is not None:
            A = model.get_sup_attn(ground_layer)          # (B, F, G, L), graph kept
            loss, l_ground, gstats = add_grounding_loss(
                loss, A, batch, motion, attn_mask, kept, t, schedule, grounding)
            ground_acc.add(gstats)
            ground_log = {f"train/ground_{k}": v for k, v in gstats.items()}

        entropy_log = {}
        if ent_layer is not None:
            # Normalised real-token attention entropy in [0, 1]; MAXIMISED
            # (subtracted) to discourage queries collapsing onto a single key —
            # see the attention-sink note in model/layers.py.
            h_attn = model.get_attn_entropy(ent_layer)
            loss = loss - attn_entropy_weight * h_attn
            entropy_log["train/attn_entropy"] = h_attn.item()

        geo_log, geo_contrib = {}, {}
        if geo_fn is not None:
            # to_x0 is a no-op for an x0-head (its output already IS x̂0) and the
            # 1/√ᾱ_t conversion for an eps-head. geo_conf_weight follows: the ᾱ_t
            # damping exists only to fade out that conversion's error amplification,
            # so it is off by default under x0 (see schedule.x0_confidence_weight).
            x0_pred = schedule.to_x0(prediction.float(), x_t, t)
            sample_weight = schedule.x0_confidence_weight(t) if geo_conf_weight else None
            loss, geo_contrib = add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights,
                sample_weight=sample_weight)
            # geo_contrib holds RAW (unweighted) magnitudes for readability — what
            # actually reaches `loss` is geo_weights[key] * geo_contrib[key], which
            # is why e.g. "pos_raw" can be numerically larger than the total loss.
            for key, value in geo_contrib.items():
                geo_log[f"train/geo_{key}_raw"] = value

        if not torch.isfinite(loss):
            # Drop this step's autograd graph BEFORE the next iteration's forward
            # allocates its own. `backward()` is what normally frees the saved
            # activations, and it is precisely what this path skips — so leaving `loss`
            # bound keeps the entire graph alive across the next model(...) call and the
            # process holds TWO full graphs at once. That is a hard 2x on peak memory:
            # a diverging run (growing loss, then an fp16 overflow) dies with a CUDA OOM
            # instead of skipping a step. Measured on a 32 GB card at batch 50: steady
            # state 22 GB, OOM at 30.5 GB live partway through the second forward.
            #
            # INVARIANT: every name below holds, or may hold, a tensor with grad_fn from
            # this iteration. Anything new that does must be added here.
            optimizer.zero_grad(set_to_none=True)
            loss = prediction = x0_pred = h_attn = A = l_ground = None
            n_skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update_from(model)  # per-step update so decay=0.9999 accumulates correctly

        total_loss += loss.item()
        n_counted += 1
        # Accumulated here, not where geo_contrib is built, so the epoch means cover the
        # same steps as total_loss (a skipped step contributes to neither).
        for key, value in geo_contrib.items():
            total_geo[key] += value
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (step + 1) % log_every == 0:
            run_logger.metrics({"train/loss": loss.item(),
                                "train/step": epoch * len(loader) + step,
                                **geo_log, **entropy_log, **ground_log})

    # Averaging over the steps that actually contributed, not over len(loader): a
    # skipped step used to be counted as a zero, which DEFLATES the reported epoch loss
    # exactly when the run is diverging — i.e. it hides the problem that caused the skip.
    n = max(n_counted, 1)
    if n_skipped:
        log.warning("epoch %d skipped %d/%d steps with a non-finite loss. This is "
                    "divergence, not noise — the reported loss is a mean over the %d "
                    "surviving steps only.", epoch, n_skipped, len(loader), n_counted)
        run_logger.metrics({"train/skipped_steps": n_skipped, "train/epoch": epoch})
    if n_skipped > MAX_SKIP_FRACTION * len(loader):
        raise RuntimeError(
            f"Epoch {epoch}: {n_skipped}/{len(loader)} steps had a non-finite loss "
            f"(> {MAX_SKIP_FRACTION:.0%}). Almost no gradient is reaching the optimiser, so "
            f"the run cannot recover on its own and every further epoch is wasted compute. "
            f"Most likely cause is an fp16 activation overflow in the forward pass (fp16 "
            f"saturates at 65504): re-run with --amp_dtype bf16 if the GPU supports it, or "
            f"resume from a checkpoint before the first skipped epoch with a lower "
            f"--lr.")
    geo_epoch = {}
    if geo_fn is not None:
        # Raw (unweighted) per-epoch means — see the comment above geo_log for why
        # these aren't directly comparable to `loss`.
        geo_epoch = {
            "train/geo_pos_raw_epoch":  total_geo["pos"]  / n,
            "train/geo_vel_raw_epoch":  total_geo["vel"]  / n,
            "train/geo_foot_raw_epoch": total_geo["foot"] / n,
        }
    return total_loss / n, {**geo_epoch, **ground_acc.epoch_means("train")}


@torch.no_grad()
def validate_one_epoch(
    ema_model, text_encoder, schedule, loader, device, epoch,
    snr_gamma=5.0,
    geo_fn=None, hml3d_pos_weight=0.0, hml3d_vel_weight=0.0, hml3d_foot_weight=0.0,
    geo_conf_weight=True, amp_dtype=torch.float16, grounding_cfg=None,
):
    ema_model.eval()
    torch.manual_seed(0)
    total_loss = 0.0
    geo_weights = {"pos": hml3d_pos_weight, "vel": hml3d_vel_weight, "foot": hml3d_foot_weight}
    # Same term, same layer choice rule, so train/val stay the SAME quantity and the two
    # curves can be overlaid — the reason diffusion_loss and _add_geo_losses are shared
    # functions rather than two copies. Deterministic layer (not a random draw) because
    # a val curve should not carry sampling noise the train curve averages out.
    use_ground = grounding_cfg is not None and grounding_cfg.active(epoch)
    val_layer  = grounding_cfg.val_layer() if use_ground else None
    ground_acc = GroundingAccumulator()

    pbar = log.progress(loader, desc=f"Val {epoch}")
    for batch in pbar:
        motion = batch["motion"].to(device)
        B = motion.shape[0]

        t = torch.randint(0, schedule.T, (B,), device=device)
        x_t, noise = schedule.q_sample(motion, t)

        if "context" in batch:
            context = batch["context"].to(device)
        else:
            context = text_encoder.encode(batch["text"])

        attn_mask = length_to_mask(batch_lengths(batch, device), motion.shape[1])

        with autocast(device_type=device.type, dtype=amp_dtype):
            prediction = ema_model(x_t, t, context, mask=attn_mask,
                                   supervise_layer=val_layer)
            # Mirror train_one_epoch's objective so the train/val curves are the
            # same quantity and can be overlaid in training_loss.png.
            target = schedule.diffusion_target(motion, noise)
            loss = diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma)

        if val_layer is not None:
            # No CFG dropout in validation, so every row's caption is really in the
            # context — kept is all-True.
            kept = torch.ones(B, dtype=torch.bool, device=device)
            loss, _, gstats = add_grounding_loss(
                loss, ema_model.get_sup_attn(val_layer), batch, motion, attn_mask,
                kept, t, schedule, grounding_cfg)
            ground_acc.add(gstats)

        if geo_fn is not None:
            x0_pred = schedule.to_x0(prediction.float(), x_t, t)
            sample_weight = schedule.x0_confidence_weight(t) if geo_conf_weight else None
            loss, _ = add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights,
                sample_weight=sample_weight)

        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(loader), ground_acc.epoch_means("val")
