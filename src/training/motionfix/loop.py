"""
The fine-tune objective and its epoch loop.

Standard diffusion training noises a clip and asks the model to recover THAT clip. This
noises the SOURCE and asks the model to recover the TARGET:

    x_t  = sqrt(a_bar_t) * source + sqrt(1 - a_bar_t) * eps
    loss = || f(x_t, t, edit_instruction) - target ||^2

so the source enters through the input the model already has, and the conditioning is the
edit instruction. Nothing is concatenated and no projection is widened — which is the
whole reason this variant beats a channel-concat: it is the same computation the LEDITS++
editor already performs at inference (Stage 1 inverts the source to x_t, Stage 3
denoises). Training this way teaches the reverse loop to land on the target instead of
reconstructing the source, i.e. it supervises exactly the step the training-free method
was doing by guidance alone.

What it gives up: the DDPM posterior q(x_{t-1} | x_t, x_0) assumes x_t was produced FROM
x_0, which here it was not. The reverse loop is therefore heuristic in the same way
SDEdit's is. In exchange the train/test computation matches exactly.
"""

import torch
from torch.amp import autocast

from training.config import resolve_amp_dtype
from training.epoch import MAX_SKIP_FRACTION
from training.grounding import grounding_loss
from utils.logger import get_logger

log = get_logger(__name__)

_ACCUMULATORS = ("loss", "diff", "ground", "m_S", "m_tok", "n_items")


def _conditioning(batch, model, text_encoder, device, cfg_dropout):
    """(context with CFG dropout applied, keep mask) for one batch."""
    if "context" in batch:                       # precomputed once, no T5 in the loop
        context = batch["context"].to(device, non_blocking=True)
    else:
        with torch.no_grad():
            context = text_encoder.encode(batch["text"])
    keep = torch.rand(context.shape[0], device=device) >= cfg_dropout
    ctx = context.clone()
    if (~keep).any():
        ctx[~keep] = model.null_text_emb.to(ctx.dtype)
    return ctx, keep


def _diffusion_loss(pred, target, x_t, t, schedule, frame_mask):
    """Masked per-channel MSE against the target, in the head's own prediction space.

    An x0 head outputs the clean signal directly; an eps head is asked for the noise that
    would carry x_t to the TARGET — the same statement one level down, which keeps this
    objective usable on either checkpoint.
    """
    goal = (target if schedule.predict_type == "x0"
            else schedule.predict_eps_from_x0(x_t, t, target))
    m = frame_mask[..., None].float()
    return ((pred - goal) ** 2 * m).sum() / m.sum().clamp(min=1) / pred.shape[-1]


def _grounding_term(model, batch, grounding, schedule, t, frame_mask, keep, layer):
    """(loss term, stats) for the TokenCompose grounding supervision."""
    attn = model.get_sup_attn(layer)                        # (B, F, G, L), graph kept
    weight = 1.0 - schedule.alphas_cumprod[t]               # pressure at HIGH noise
    return grounding_loss(
        attn, batch["keyid"], grounding.cache, frame_mask, keep,
        sample_weight=weight, lambda_mirror=grounding.mirror,
        margin=grounding.margin, mirror_mat=grounding.mirror_mat,
        lambda_even=grounding.even)


def _geometric_terms(pred, target, x_t, t, schedule, frame_mask, geo_fn, args):
    """Sum of the enabled MDM-style geometric losses, already weighted."""
    x0_hat = (pred if schedule.predict_type == "x0"
              else schedule.predict_x0_from_eps(x_t, t, pred))
    # x0_confidence_weight down-weights high-noise samples, where x0_hat is a poor
    # clean-signal estimate and SMPL forward kinematics amplifies the error.
    geo = geo_fn(x0_hat, target, frame_mask,
                 sample_weight=schedule.x0_confidence_weight(t))
    total = torch.zeros((), device=pred.device)
    for name, weight in (("pos", args.geo_pos_weight), ("vel", args.geo_vel_weight),
                         ("foot", args.geo_foot_weight)):
        if weight > 0 and torch.isfinite(geo[name]):
            total = total + weight * geo[name]
    return total


def _check_skip_rate(n_skipped, n_batches, epoch, train):
    """Report skipped steps, and abort a run where almost no gradient survives.

    A non-finite loss is divergence, not noise: a run where most steps are skipped must
    stop rather than burn the night. Two overnight runs have been lost to exactly that.
    """
    if n_skipped:
        log.warning("%s epoch %d skipped %d/%d steps with a non-finite loss. The "
                    "reported loss averages the SURVIVORS only, so it understates the "
                    "damage.", "train" if train else "val", epoch, n_skipped, n_batches)
    if train and n_skipped > MAX_SKIP_FRACTION * max(n_batches, 1):
        raise RuntimeError(
            f"epoch {epoch}: {n_skipped}/{n_batches} steps non-finite "
            f"(> {MAX_SKIP_FRACTION:.0%}). Almost no gradient is reaching the optimiser — "
            f"stopping instead of training on nothing. Try --amp_dtype bf16 or a lower --lr.")


def run_epoch(model, ema, schedule, opt, sched, scaler, loader, device, args, grounding,
              text_encoder, epoch, geo_fn, train=True):
    """One pass over `loader`; returns the epoch's mean statistics."""
    model.train(train)
    totals = dict.fromkeys(_ACCUMULATORS, 0.0)
    n_steps = n_skipped = 0
    use_ground = grounding is not None and grounding.active(epoch)
    amp = resolve_amp_dtype(args.amp_dtype)

    pbar = log.progress(loader, desc=f"{'train' if train else 'val'} {epoch}")
    for batch in pbar:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        B, n_frames, _ = source.shape
        lengths = batch["length"].to(device)
        frame_mask = torch.arange(n_frames, device=device)[None] < lengths[:, None]

        # THE OBJECTIVE: noise the SOURCE, regress the TARGET.
        t = torch.randint(args.t_min, args.t_max + 1, (B,), device=device)
        x_t, _ = schedule.q_sample(source, t)
        ctx, keep = _conditioning(batch, model, text_encoder, device, args.cfg_dropout)
        g_layer = grounding.pick_layer() if use_ground else None

        with autocast(device_type=device.type, dtype=amp, enabled=amp != torch.float32):
            pred = model(x_t, t, ctx, mask=frame_mask, supervise_layer=g_layer)
            diff = _diffusion_loss(pred, target, x_t, t, schedule, frame_mask)
            loss = diff

            g_val = torch.zeros((), device=device)
            if use_ground:
                g_val, g_stats = _grounding_term(model, batch, grounding, schedule, t,
                                                 frame_mask, keep, g_layer)
                loss = loss + grounding.weight * g_val
                totals["m_S"] += float(g_stats.get("m_S", 0.0))
                # m_tok is the mass inside the ACTUAL L_token target. It equals m_S
                # unless an item carries a spatiotemporal region, so the gap between the
                # two is the only direct read on what --diff_temporal changed.
                totals["m_tok"] += float(g_stats.get("m_tok", g_stats.get("m_S", 0.0)))
                totals["n_items"] += float(g_stats.get("n_items", 0.0))

            if geo_fn is not None:
                loss = loss + _geometric_terms(pred, target, x_t, t, schedule,
                                               frame_mask, geo_fn, args)

        if not torch.isfinite(loss):
            # INVARIANT: every name below holds, or may hold, a tensor with grad_fn from
            # this iteration. Leaving any of them bound keeps the whole graph alive
            # across the NEXT forward, so the process holds two graphs at once — measured
            # at 1.71x peak on the real loop, which turns a diverging run into a CUDA OOM
            # instead of a skipped step. Anything new that carries a graph goes here.
            # (Graph tensors held only inside the helpers above die with their frames.)
            if train:
                opt.zero_grad(set_to_none=True)
            loss = diff = g_val = pred = None
            n_skipped += 1
            continue

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_from(model)

        totals["loss"] += loss.detach().item()
        totals["diff"] += diff.detach().item()
        totals["ground"] += g_val.detach().item()
        n_steps += 1
        pbar.set_postfix(loss=f"{totals['loss'] / max(n_steps, 1):.4f}")

    _check_skip_rate(n_skipped, len(loader), epoch, train)
    n = max(n_steps, 1)
    return {k: v / n for k, v in totals.items()} | {"steps": n_steps, "skipped": n_skipped}
