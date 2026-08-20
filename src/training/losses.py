"""
The loss terms shared by the train loop, the validation loop and overfit_one.py.

Keeping them in one module is what stops the train and validation objectives from
silently drifting apart: both call the same `diffusion_loss` and `add_geo_losses`, so
the two curves are the same quantity and can be overlaid.
"""

import torch

from training import grounding

# Fraction of an epoch's steps that may be dropped for a non-finite loss before
# train_one_epoch gives up. Past this the run is not "recovering slowly": with almost no
# gradient reaching the optimiser it cannot climb back out, and `ema.update_from` runs so
# rarely that even the validation curve freezes — which is exactly what a run diverging
# into fp16 overflow looks like from the outside. Fail loudly instead of burning the night.
MAX_SKIP_FRACTION = 0.5


def batch_lengths(batch, device):
    """`batch["length"]` as a device tensor, whatever the collate produced."""
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


def add_geo_losses(loss, x0_pred, motion, attn_mask, geo_fn, weights, sample_weight=None):
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


def add_grounding_loss(loss, A, batch, motion, attn_mask, kept, t, schedule, cfg):
    """Add the TokenCompose grounding term (training/grounding.py) to `loss`.

    Returns (loss, l_ground, stats). `l_ground` is handed back so the caller can drop it
    on the non-finite skip path — it holds a graph over an explicit (B, h, F·G, L)
    softmax, which is the largest single tensor in the step.

    TIMESTEP WEIGHTING — the one line most likely to be "fixed" back to the wrong thing.
    `w_t = 1 − ᾱ_t` up-weights HIGH noise. The original design prescribed
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
        # The shortcut monitor. Computed on the
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


class GroundingAccumulator:
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
