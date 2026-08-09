"""
Why is (or will) a checkpoint produce non-finite losses?

Two questions, and the second is the one that works BEFORE a run has diverged:

  1. Is this checkpoint already broken?  One batch, three dtypes (fp32/bf16/fp16).
     fp32 non-finite  -> the WEIGHTS are damaged; resume from earlier.
     fp32/bf16 fine, fp16 inf -> fp16 ACTIVATION OVERFLOW (fp16 saturates at 65504).
     The weights are usable; the arithmetic is not. `--amp_dtype bf16` removes it.

  2. Is it HEADING there?  Pass several checkpoints from the same run (they are cheap
     to compare) and the script reports max|weight| and max|activation| per checkpoint,
     measured in fp32 so nothing saturates, plus the headroom left to fp16's ceiling.
     A run dying of activation growth shows the growth for tens of epochs before the
     cliff — so two checkpoints from BEFORE the divergence are enough to confirm the
     mechanism and to estimate when it will hit. A flat trend means look elsewhere.

The activation numbers come from forward hooks on every leaf module, so the "worst
module" column names where the growth lives (attention vs feed-forward) without
assuming a backbone.

Usage
-----
    # trend across a run (the useful mode before divergence)
    python src/diagnose_divergence.py --data_root data/HumanML3D/HumanML3D_smplh \
        --checkpoint runs/exp_x/checkpoint_epoch_0049 \
        --checkpoint runs/exp_x/checkpoint_epoch_0099 \
        --checkpoint runs/exp_x/checkpoint_latest

    # single checkpoint, post-mortem
    python src/diagnose_divergence.py --checkpoint runs/exp_x/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D_smplh
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.amp import autocast

from data.dataset import build_dataloader
from model.schedule import NoiseSchedule
from training.epoch import diffusion_loss
from utils.cli import add_data_args, add_model_args, resolve_device
from utils.model_io import load_model
from utils.padding import length_to_mask

FP16_MAX = 65504.0
DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, multi_checkpoint=True)
    add_data_args(p, split=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--batches", type=int, default=4,
                   help="Batches per checkpoint. A diverging model overflows on most "
                        "batches, not all, so one batch can mislead.")
    p.add_argument("--t_max", type=int, default=None,
                   help="Sample timesteps from [0, t_max] instead of the full schedule. "
                        "Use to check whether the growth is concentrated at high noise.")
    return p.parse_args()


class ActivationProbe:
    """Records max |output| per leaf module over a set of forward passes."""

    def __init__(self, model):
        self.peaks, self._handles = {}, []
        for name, mod in model.named_modules():
            if list(mod.children()):          # leaves only
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            if not isinstance(out, torch.Tensor) or not out.is_floating_point():
                return
            v = out.detach().abs().max().float().item()
            if v > self.peaks.get(name, 0.0):
                self.peaks[name] = v
        return hook

    def worst(self, k=1):
        return sorted(self.peaks.items(), key=lambda kv: -kv[1])[:k]

    def close(self):
        for h in self._handles:
            h.remove()


def fixed_batches(loader, device, schedule, n, t_max=None):
    """The same n batches, with the same t and noise, reused for every checkpoint."""
    out = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for i, batch in enumerate(loader):
        if i >= n:
            break
        motion = batch["motion"].to(device)
        mask = length_to_mask(torch.as_tensor(batch["length"], device=device),
                              motion.shape[1])
        hi = schedule.T if t_max is None else min(t_max + 1, schedule.T)
        t = torch.randint(0, hi, (motion.shape[0],), generator=g).to(device)
        x_t, noise = schedule.q_sample(motion, t)
        ctx = batch["context"].to(device) if "context" in batch else None
        out.append((motion, mask, t, x_t, noise, ctx))
    return out


def probe_checkpoint(ckpt, args, device, batches_for):
    model, config = load_model(ckpt, device=device, use_ema=not args.no_ema)
    schedule = NoiseSchedule.from_config(config, device=device)
    batches = batches_for(config, schedule)

    max_w = max(p.detach().abs().max().item() for p in model.parameters())
    bad_w = sum(int(not torch.isfinite(p.detach()).all()) for p in model.parameters())

    probe = ActivationProbe(model)
    non_finite = {n: 0 for n, _ in DTYPES}
    with torch.no_grad():
        for (motion, mask, t, x_t, noise, ctx) in batches:
            for name, dt in DTYPES:
                with autocast(device_type=device.type, dtype=dt,
                              enabled=dt is not torch.float32):
                    pred = model(x_t, t, ctx, mask=mask)
                    target = schedule.diffusion_target(motion, noise)
                    loss = diffusion_loss(target, pred, mask, schedule, t, 0.0)
                if not torch.isfinite(loss):
                    non_finite[name] += 1
                if dt is torch.float32:
                    pass          # fp32 pass is the one the hooks should be trusted from
    worst_name, worst_val = probe.worst(1)[0] if probe.peaks else ("-", float("nan"))
    probe.close()
    del model
    torch.cuda.empty_cache()
    return dict(ckpt=ckpt, max_w=max_w, bad_w=bad_w, act=worst_val, mod=worst_name,
                non_finite=non_finite, predict_type=schedule.predict_type,
                epoch=config.get("epoch"),
                amp=config.get("amp_dtype", "fp16 (pre-flag)"))


def main():
    args = parse_args()
    device = resolve_device(args.device)

    cache = {}

    def batches_for(config, schedule):
        key = config.get("feature_mode", "humanml3d")
        if key not in cache:
            loader = build_dataloader(args.data_root, split=args.split,
                                      batch_size=args.batch_size,
                                      max_frames=args.max_frames, num_workers=0,
                                      feature_mode=key)
            cache[key] = fixed_batches(loader, device, schedule, args.batches, args.t_max)
        return cache[key]

    rows = [probe_checkpoint(c, args, device, batches_for) for c in args.checkpoint]

    print(f"\n{args.batches} batches x {args.batch_size}, identical across checkpoints"
          + (f", t sampled from [0, {args.t_max}]" if args.t_max else "")
          + f".  fp16 ceiling = {FP16_MAX:.0f}\n")
    hdr = (f"  {'checkpoint':30} {'epoch':>6} {'max|w|':>9} {'max|act|':>11} "
           f"{'fp16 room':>10}  {'fp32/bf16/fp16 bad':>18}  worst module")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for r in rows:
        room = FP16_MAX / r["act"] if r["act"] > 0 else float("inf")
        nf = "/".join(str(r["non_finite"][n]) for n, _ in DTYPES)
        ep = "?" if r["epoch"] is None else str(r["epoch"])
        print(f"  {os.path.basename(r['ckpt'].rstrip('/')):30} {ep:>6} {r['max_w']:9.2f} "
              f"{r['act']:11.1f} {room:9.1f}x  {nf:>18}  {r['mod']}")

    print("\nverdict:")
    last = rows[-1]
    if last["bad_w"]:
        print("  weights contain non-finite values -> damaged. Resume from earlier; no "
              "dtype change will help.")
    elif last["non_finite"]["fp32"]:
        print("  fp32 forward is non-finite -> the weights are broken even in full "
              "precision. Resume from before the first skipped epoch, with a lower --lr.")
    elif last["non_finite"]["fp16"]:
        print(f"  fp32/bf16 finite but fp16 non-finite on "
              f"{last['non_finite']['fp16']}/{args.batches} batches -> fp16 ACTIVATION "
              f"OVERFLOW. Weights are fine. Resume with --amp_dtype bf16.")
    elif len(rows) > 1:
        first, last_r = rows[0], rows[-1]
        room = FP16_MAX / last_r["act"]
        act_growth = last_r["act"] / max(first["act"], 1e-9)
        w_growth = last_r["max_w"] / max(first["max_w"], 1e-9)
        span = (None if first["epoch"] is None or last_r["epoch"] is None
                else last_r["epoch"] - first["epoch"])
        print(f"  nothing overflows yet. Activations grew {act_growth:.2f}x and weights "
              f"{w_growth:.2f}x across these checkpoints; {room:.1f}x headroom remains to "
              f"the fp16 ceiling.")
        # Reference point measured on a run that did NOT diverge (exp_hml3d_x0, x0 +
        # humanml3d, 500 epochs): activations 225 -> 605 over epochs 99..499, i.e. 1.26x
        # per 100 epochs, still 108x from the ceiling at the end. So GROWTH BY ITSELF IS
        # NORMAL — every transformer here does it. What matters is the rate and how much
        # room is left.
        print("  reference (a run that did NOT diverge — exp_hml3d_x0): 1.26x per 100 "
              "epochs, 108x headroom left at epoch 499.")
        if span and span > 0 and act_growth > 1.0:
            per100 = act_growth ** (100.0 / span)
            import math
            to_ceiling = math.log(room) / math.log(act_growth) * span
            print(f"  this run: {per100:.2f}x per 100 epochs -> at this rate the ceiling "
                  f"arrives about {to_ceiling:.0f} epochs after epoch {last_r['epoch']} "
                  f"(~epoch {last_r['epoch'] + to_ceiling:.0f}).")
            print("  Compare that projection against the epoch of your first "
                  "`skipped N/M steps` warning: if they roughly agree, activation growth "
                  "IS the mechanism. If the projection lands far later, something else "
                  "ends the run and bf16 will only postpone it.")
        else:
            print("  activations are not growing across these checkpoints — either the "
                  "growth starts later, or the divergence is not activation growth. Use "
                  "checkpoints closer to the first skipped epoch, and --t_max to test "
                  "whether it is concentrated at a particular noise level.")
    else:
        print("  nothing overflows on these batches. Pass SEVERAL checkpoints from the "
              "run (--checkpoint repeatedly) to see whether activations are trending "
              "toward the ceiling — that works before the run has diverged.")


if __name__ == "__main__":
    main()
