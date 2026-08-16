"""
Per-epoch train/validate loops for train.py.

Both loops compute the same Min-SNR-weighted (or plain) diffusion MSE and, when
enabled, the same MDM-style geometric losses — factored into diffusion_loss and
_add_geo_losses so the train/val objectives can't silently drift apart.
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from training import grounding
from utils.padding import length_to_mask

# Fraction of an epoch's steps that may be dropped for a non-finite loss before
# train_one_epoch gives up. Past this the run is not "recovering slowly": with almost no
# gradient reaching the optimiser it cannot climb back out, and `ema.update_from` runs so
# rarely that even the validation curve freezes — which is exactly what a run diverging
# into fp16 overflow looks like from the outside. Fail loudly instead of burning the night.
MAX_SKIP_FRACTION = 0.5


def _batch_lengths(batch, device):
    lengths = batch["length"]
    if isinstance(lengths, torch.Tensor):
        return lengths.detach().clone().to(device)
    return torch.as_tensor(lengths, device=device)


def apply_cfg_dropout(model, context, cfg_dropout):
    """Replace a `cfg_dropout` fraction of the batch's conditioning with the learned
    null embedding.

    null_text_emb must keep its gradient so CFG scale > 1 works at inference. This also
    trains the unconditional branch used by LEDITS++ Stage 1 inversion (context=None)
    and the ε_θ(x_t, ∅) term in Stage 3 Eq. 1.

    Returns (context, kept) where `kept` is (B,) bool, False on the rows whose caption
    was replaced. Any loss defined on the CAPTION rather than on the motion — the
    grounding loss is the first — must exclude those rows: their words are not in the
    context at all, so supervising their columns optimises attention toward text the
    forward pass never saw. Silent training noise, not an error anything would catch.
    """
    if cfg_dropout <= 0.0:
        return context, torch.ones(context.shape[0], dtype=torch.bool,
                                   device=context.device)
    B = context.shape[0]
    drop = torch.rand(B, device=context.device) < cfg_dropout
    context = torch.where(drop[:, None, None],
                          model.null_text_emb.expand(B, -1, -1), context)
    return context, ~drop


def diffusion_loss(target, prediction, attn_mask, schedule, t, snr_gamma):
    """Min-SNR-weighted (or plain) MSE between the target and the network output,
    masked to valid (non-padding) frames. Shared by train_one_epoch,
    validate_one_epoch and overfit_one.py, so the three cannot drift apart.

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


def _add_grounding_loss(loss, A, batch, motion, attn_mask, kept, t, schedule, cfg):
    """Add the TokenCompose grounding term (training/grounding.py) to `loss`.

    Returns (loss, l_ground, stats). `l_ground` is handed back so the caller can drop it
    on the non-finite skip path — it holds a graph over an explicit (B, h, F·G, L)
    softmax, which is the largest single tensor in the step.

    TIMESTEP WEIGHTING — the one line most likely to be "fixed" back to the wrong thing.
    `w_t = 1 − ᾱ_t` up-weights HIGH noise. AttentionGrounding_Options.md §1.5 prescribed
    ᾱ_t, i.e. the opposite, and that is backwards twice over:
      1. Under x0 the mask's instruction-sensitivity strengthens with noise (category r
         0.899 at t<250 vs 0.746 at t≥750) and alignment improves 0.205 → 0.238 under
         `--m1_window 750 999`. ᾱ_t would supervise hardest where the mask is never read.
      2. It opens the source-dynamics shortcut. At low t, x_t ≈ x0 and the clip's motion
         is legible, so "attend to whatever moves" satisfies the loss without reading a
         word. At t = 900 there is nothing left to detect and the caption is the only
         signal present.
    `--attn_ground_window LO HI` is the hard-gate alternative kept for ablation: it
    supervises only in-window samples, at ~4× fewer grounded samples per batch.
    """
    if "text" not in batch:
        # Precomputed text_emb/ makes the dataset return `context` and drop the caption,
        # and the labels are keyed by caption string. Trainer._build_grounding refuses
        # the run for this reason; this catches any other caller.
        raise KeyError(
            "the grounding loss needs batch['text'], but this batch carries precomputed "
            "'context' embeddings instead — the caption is the label cache's key.")

    if cfg.window is not None:
        lo, hi = cfg.window
        valid = kept & (t >= lo) & (t <= hi)
        w_t = None                      # a hard gate does not also soft-weight
    else:
        valid = kept
        w_t = 1.0 - schedule.x0_confidence_weight(t)          # (B,)

    src = None
    if cfg.monitor:
        # The shortcut monitor (docs/TokenCompose_Handoff.md §4.6). Computed on the
        # clean motion, detached — it never touches the gradient, it only says whether
        # rising m_S is word routing or a sharpened motion detector.
        with torch.no_grad():
            src = grounding.batched_source_activity(
                motion.detach(), cfg.group_channels, attn_mask)

    l_ground, stats = grounding.grounding_loss(
        A.float(), batch["text"], cfg.cache, attn_mask, valid,
        sample_weight=w_t, lambda_mirror=cfg.mirror, margin=cfg.margin,
        source_act=src, mirror_mat=cfg.mirror_mat, lambda_even=cfg.even)
    return loss + cfg.weight * l_ground, l_ground, stats


class _GroundingAccumulator:
    """Per-epoch means of the grounding diagnostics.

    They are means over STEPS, not over items, so a step with one supervised token
    counts the same as a step with twenty. That is deliberate: `m_S` is read as a level
    ("the word puts X of its attention on its own group") and an item-weighted mean
    would let the handful of caption-dense batches set the epoch's number.
    """

    KEYS = ("m_S", "m_S_tier1", "m_mirror", "split_max", "src_corr")

    def __init__(self):
        self.sums = {k: 0.0 for k in self.KEYS}
        self.counts = {k: 0 for k in self.KEYS}
        self.items = 0

    def add(self, stats):
        self.items += stats.get("n_items", 0)
        for k in self.KEYS:
            if k in stats:
                self.sums[k] += stats[k]
                self.counts[k] += 1

    def epoch_means(self, prefix):
        if not self.items:
            return {}
        out = {f"{prefix}/ground_items_epoch": self.items}
        for k in self.KEYS:
            if self.counts[k]:
                out[f"{prefix}/ground_{k}_epoch"] = self.sums[k] / self.counts[k]
        return out


def train_one_epoch(
    model, ema, text_encoder, schedule, optimizer, scaler,
    loader, device, cfg_dropout, logger, epoch, log_every,
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
    ground_acc  = _GroundingAccumulator()

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
        context, kept = apply_cfg_dropout(model, context, cfg_dropout)

        attn_mask = length_to_mask(_batch_lengths(batch, device), motion.shape[1])

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
            loss, l_ground, gstats = _add_grounding_loss(
                loss, A, batch, motion, attn_mask, kept, t, schedule, grounding)
            ground_acc.add(gstats)
            ground_log = {f"train/ground_{k}": v for k, v in gstats.items()}

        entropy_log = {}
        if ent_layer is not None:
            # Normalised real-token attention entropy in [0, 1]; MAXIMISED
            # (subtracted) to discourage queries collapsing onto a single key —
            # see docs/LEDITSpp_Attention_Sink_Research.md §9.
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
            loss, geo_contrib = _add_geo_losses(
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
            logger.log({"train/loss": loss.item(), "train/step": epoch * len(loader) + step,
                        **geo_log, **entropy_log, **ground_log})

    # Averaging over the steps that actually contributed, not over len(loader): a
    # skipped step used to be counted as a zero, which DEFLATES the reported epoch loss
    # exactly when the run is diverging — i.e. it hides the problem that caused the skip.
    n = max(n_counted, 1)
    if n_skipped:
        print(f"  WARNING: epoch {epoch} skipped {n_skipped}/{len(loader)} steps with a "
              f"non-finite loss. This is divergence, not noise — the reported loss is a "
              f"mean over the {n_counted} surviving steps only.")
        logger.log({"train/skipped_steps": n_skipped, "train/epoch": epoch})
    if n_skipped > MAX_SKIP_FRACTION * len(loader):
        raise RuntimeError(
            f"Epoch {epoch}: {n_skipped}/{len(loader)} steps had a non-finite loss "
            f"(> {MAX_SKIP_FRACTION:.0%}). Almost no gradient is reaching the optimiser, so "
            f"the run cannot recover on its own and every further epoch is wasted compute. "
            f"Most likely cause is an fp16 activation overflow in the forward pass (fp16 "
            f"saturates at 65504): re-run with --amp_dtype bf16 if the GPU supports it, or "
            f"resume from a checkpoint before the first skipped epoch with a lower --lr. "
            f"See docs/FINDINGS.md 'fp16 activation overflow'.")
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
    ground_acc = _GroundingAccumulator()

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
            loss, _, gstats = _add_grounding_loss(
                loss, ema_model.get_sup_attn(val_layer), batch, motion, attn_mask,
                kept, t, schedule, grounding_cfg)
            ground_acc.add(gstats)

        if geo_fn is not None:
            x0_pred = schedule.to_x0(prediction.float(), x_t, t)
            sample_weight = schedule.x0_confidence_weight(t) if geo_conf_weight else None
            loss, _ = _add_geo_losses(
                loss, x0_pred, motion.float(), attn_mask, geo_fn, geo_weights,
                sample_weight=sample_weight)

        total_loss += loss.item()
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(loader), ground_acc.epoch_means("val")
