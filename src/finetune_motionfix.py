"""
Supervised fine-tune of a pretrained SMPL-H checkpoint on MotionFix triplets.

THE OBJECTIVE, and why it needs no architecture change
------------------------------------------------------
Standard diffusion training noises a clip and asks the model to recover *that* clip. This
noises the **source** and asks the model to recover the **target**:

    x_t   = sqrt(a_bar_t) * source + sqrt(1 - a_bar_t) * eps
    loss  = || f_theta(x_t, t, edit_instruction) - target ||^2

so the source enters through the input the model already has, and the conditioning is the
edit instruction. Nothing is concatenated and no projection is widened — the pretrained
weights are used as-is, which is the whole reason this variant is preferable to a
channel-concat: **it is the same computation the LEDITS++ editor already performs at
inference.** Stage 1 inverts the source to x_t; Stage 3 denoises. Training this way teaches
the reverse loop to land on the target instead of reconstructing the source, i.e. it
supervises exactly the step the training-free method was doing by guidance alone.

What it gives up: the DDPM posterior q(x_{t-1} | x_t, x_0) assumes x_t was produced *from*
x_0, which here it was not (x_t comes from the source, x_0 is the target). The reverse loop
is therefore heuristic in the same way SDEdit's is. In exchange the train/test computation
matches exactly, which the concat variant cannot claim.

WHY THE TIMESTEP BAND IS CAPPED (--t_max)
-----------------------------------------
At high t the source is gone — measured on this schedule, sqrt(a_bar) is 0.305 at t=800 and
0.000 at t=999. Past that point "predict the target from noise + a *relative* instruction"
is ill-posed: "raise the arm higher" does not specify a whole motion. Training there teaches
text-to-motion generation from an instruction that was never meant to carry it. The default
t_max=800 keeps the source at least ~30 % present in every training sample.

THE GROUNDING LABEL (--ground_labels)
-------------------------------------
The TokenCompose loss needs, per item, a set of text columns W and a set of body-part groups
S. Two sources of S, measured 2026-08-16 on 400 train triplets (docs/MaskOptions.md §20.1c):

  parser  — the caption parser reads the groups out of the instruction. Correct by
            construction, but silent on the ~23 % of instructions naming no body part.
  diff    — the top groups of |d(velocity_target) - d(velocity_source)|. Covers everything,
            ~77 % top-1 accurate — but ~55 points of that is reachable from a SHUFFLED pair,
            i.e. the label carries a corpus-level "the target is usually an arm" prior. Used
            alone it partly teaches that prior instead of word->group routing, which is the
            shortcut Option 19 exists to avoid. `ground_src_frac` is logged for this reason.

Default `parser_first` uses the parser where it fires and the diff elsewhere: full coverage,
clean labels where clean labels exist.

Example
-------
    python src/finetune_motionfix.py \
        --checkpoint runs/exp_smplh_verbs/checkpoint_latest \
        --smplh_data_root data/HumanML3D/HumanML3D_smplh \
        --output_dir runs/ft_motionfix
"""

import os
import sys
import json
import time
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.body_groups import resolve_group_context, GROUP_NAMES
from training.grounding import (GroundingConfig, grounding_loss, mirror_matrix,
                                resolve_ground_layers)
from training.config import resolve_amp_dtype
from training.epoch import MAX_SKIP_FRACTION
from data.smplh_features import smplh_to_features, resample_motion
from data.body_part_labels import to_items
from utils.checkpoint import save_checkpoint
from utils.ema import EMA
from utils.model_io import load_model


# ── defaults, and where each number comes from ──────────────────────────────────
# 5,387 train triplets / batch 64 = 85 steps/epoch. The pretrain was 500 epochs x 335
# steps = ~167k steps; 100 epochs here is ~8.5k, i.e. ~5 % of it, which is the usual
# order for a fine-tune onto a new objective with 4x less data.
D_EPOCHS      = 100
D_BATCH       = 64
D_LR          = 1e-5      # 10x below the 1e-4 pretrain LR: the objective changes, the weights should not
D_EMA         = 0.999     # horizon ~1/(1-d) = 1,000 steps. 0.9999 needs 10k and this run is ~8.5k,
                          # so the EMA would never catch up and the saved ema.pt would lag the model.
D_TMAX        = 800       # sqrt(a_bar_800) = 0.305 -> source still ~30 % present. See the docstring.


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── what to fine-tune, and on what ──
    p.add_argument("--checkpoint", required=True,
                   help="Pretrained SMPL-H checkpoint dir (config.json + ema.pt/model.pt). "
                        "Fine-tuning starts from its EMA weights unless --no_ema_init.")
    p.add_argument("--smplh_data_root", default=None,
                   help="SMPL-H root holding the 135-d Mean.npy / Std.npy. Default: whatever the "
                        "CHECKPOINT records as its own `data_root`, which is the only value "
                        "guaranteed to be the normalisation the model trained under — a hand-typed "
                        "path that disagrees corrupts every sample silently. Pass one only to "
                        "override deliberately.")
    p.add_argument("--motionfix_root",
                   default=os.path.join(os.path.dirname(src_dir), "data/motionfix/data/motionfix-dataset"),
                   help="Dir with motionfix.pth.tar + splits.json. Defaults to the copy inside this "
                        "repo, resolved from the script's own location so it does not depend on cwd.")
    p.add_argument("--output_dir", default="runs/ft_motionfix")
    p.add_argument("--cache_dir", default=None,
                   help="Where to cache featurised triplets (default <output_dir>/../ft_cache). "
                        "Built once; reused across runs, so an A/B of label sources pays it once.")
    p.add_argument("--no_ema_init", action="store_true",
                   help="Initialise from model.pt instead of ema.pt.")

    # ── the objective ──
    p.add_argument("--t_min", type=int, default=0,
                   help="Lowest diffusion timestep sampled.")
    p.add_argument("--t_max", type=int, default=D_TMAX,
                   help=f"Highest timestep sampled (default {D_TMAX}). Above it the source is "
                        "destroyed and the task degenerates into generating a whole motion from a "
                        "relative instruction. sqrt(alpha_bar): 0.70@500, 0.58@600, 0.45@700, "
                        "0.31@800, 0.00@999.")
    p.add_argument("--cfg_dropout", type=float, default=0.1,
                   help="Probability of replacing the instruction with the null embedding. Keep "
                        "non-zero or classifier-free guidance stops working at inference.")

    # ── grounding (TokenCompose) ──
    p.add_argument("--ground_labels", default="parser_first",
                   choices=["parser_first", "diff_only", "parser_only", "off"],
                   help="Source of the supervised group set S. 'parser_first' (default) = parser "
                        "where it names a body part, velocity-diff elsewhere. 'diff_only' = always "
                        "the motion difference. 'parser_only' = skip unlabelled instructions (the "
                        "control that isolates what the diff adds). 'off' disables the loss.")
    p.add_argument("--attn_ground_weight", type=float, default=5e-3,
                   help="lambda for the TokenCompose term. 5e-3 is the dose measured free of FID "
                        "cost in pretraining; 0.01 cost 31 %% FID (docs/FINDINGS.md).")
    p.add_argument("--attn_ground_layers", default="middle",
                   help="Blocks to supervise. Default matches the pretrained checkpoint's own setting.")
    p.add_argument("--attn_ground_mirror", type=float, default=1.0)
    p.add_argument("--attn_ground_even", type=float, default=0.1)
    p.add_argument("--attn_ground_margin", type=float, default=0.1)
    p.add_argument("--attn_ground_warmup_epochs", type=int, default=0,
                   help="0 by default: unlike pretraining, attention is ALREADY grounded here, so "
                        "there is no noise phase to wait out.")
    p.add_argument("--diff_ratio", type=float, default=0.5,
                   help="Diff labels: keep every group holding >= this share of the top group's "
                        "difference mass. 0.5 gives 1-2 groups on the measured profiles.")
    p.add_argument("--diff_max", type=int, default=2,
                   help="Diff labels: hard cap on |S|. 2, because top-2 was where measured set "
                        "accuracy plateaued (83.7 %%) while the shuffled control kept rising.")
    p.add_argument("--diff_temporal", type=float, default=0.5,
                   help="Diff labels: make the L_token target SPATIOTEMPORAL by keeping only the "
                        "busiest this-fraction of frames inside the selected groups. 1.0 disables "
                        "it and supervises the whole group row (the group-set behaviour, and "
                        "bit-identical to it). ⚠ Measured: the temporal axis of the velocity "
                        "difference carries a real-vs-shuffled gap of only +0.012, against "
                        "+0.19..+0.25 on the group axis — so this mostly sharpens the target "
                        "around frames where EITHER clip moves fast, not where the edit is. "
                        "0.5 is deliberately loose for that reason. docs/MaskOptions.md §20.1d.")
    p.add_argument("--diff_tier1", action="store_true",
                   help="Let diff-derived items be tier 1 (adds the left/right mirror margin). OFF "
                        "by default: a wrong side would actively teach the wrong laterality, and "
                        "the mirror margin is the harshest term in the loss.")

    # ── geometric losses: keep the pretrain recipe so plausibility does not drift ──
    p.add_argument("--geo_pos_weight",  type=float, default=0.1)
    p.add_argument("--geo_vel_weight",  type=float, default=0.1)
    p.add_argument("--geo_foot_weight", type=float, default=0.01)
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh",
                   help="Needed only when a geo weight is non-zero.")

    # ── schedule / optimisation ──
    p.add_argument("--epochs",     type=int,   default=D_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=D_BATCH)
    p.add_argument("--lr",         type=float, default=D_LR)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="0: a short fine-tune from a good initialisation does not need extra pull "
                        "toward the origin, and decay fights the pretrained solution.")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="~2.5 epochs at the default batch. Guards the first steps, where a fresh "
                        "optimiser state on pretrained weights does the most damage.")
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--ema_decay",  type=float, default=D_EMA)
    p.add_argument("--amp_dtype",  default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--pad_to", default="batch", choices=["batch", "max"],
                   help="'batch' (default) pads each batch to its own longest clip; 'max' pads to "
                        "--max_frames, the pre-2026-08-16 behaviour. Bit-identical results (padding "
                        "is masked out of self-attention), ~2.7x less compute per step and up to "
                        "~7x in the quadratic term \u2014 train clips average 74 frames, never exceed 100.")
    p.add_argument("--bucket_by_length", action=argparse.BooleanOptionalAction, default=True,
                   help="Group similar-length clips into the same batch so --pad_to batch actually "
                        "pays off. Batch ORDER stays shuffled.")
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="Hold the featurised clips in RAM (0.43 GB train). Removes two disk reads "
                        "per sample per step.")
    p.add_argument("--precompute_text", action=argparse.BooleanOptionalAction, default=True,
                   help="Encode every instruction once up front (2.25 GB) instead of running T5 on "
                        "every batch of every epoch, and drop the encoder from VRAM afterwards.")
    p.add_argument("--num_workers", type=int, default=2,
                   help="2, not more: each worker forks the parent, and Python refcounting turns "
                        "copy-on-write pages into real copies. The cached .npy reads are cheap "
                        "enough that extra workers buy little and cost GBs.")
    p.add_argument("--seed",       type=int, default=42)

    # ── data shaping ──
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--src_fps",  type=float, default=30.0)
    p.add_argument("--edit_fps", type=float, default=20.0)

    # ── logging ──
    p.add_argument("--val_every",  type=int, default=5)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--log_every",  type=int, default=20)
    p.add_argument("--device", default=None)
    return p


# ── data ────────────────────────────────────────────────────────────────────────

def _featurise(m, src_fps, edit_fps):
    r = np.asarray(m["rots"], dtype=np.float32)
    t = np.asarray(m["trans"], dtype=np.float32)
    r, t = resample_motion(r, t, src_fps, edit_fps)
    return smplh_to_features(r, t)


def build_cache(args, split_keys, data, cache_dir):
    """Featurise every triplet once. Resumable: an existing pair is left alone."""
    os.makedirs(cache_dir, exist_ok=True)
    kept, skipped = [], {}
    for k in tqdm(split_keys, desc=f"caching {os.path.basename(cache_dir)}"):
        ps, pt = os.path.join(cache_dir, f"{k}_s.npy"), os.path.join(cache_dir, f"{k}_t.npy")
        if not (os.path.exists(ps) and os.path.exists(pt)):
            S = _featurise(data[k]["motion_source"], args.src_fps, args.edit_fps)
            T = _featurise(data[k]["motion_target"], args.src_fps, args.edit_fps)
            n = min(len(S), len(T), args.max_frames)      # the pair is scored on its common window
            if n < args.min_frames:
                skipped[k] = f"too short ({n})"
                continue
            np.save(ps, S[:n]); np.save(pt, T[:n])
        kept.append(k)
    return kept, skipped


class TripletDataset(Dataset):
    """(source, target, instruction) triplets, normalised. Returns UNPADDED clips — the
    collate pads each batch to its own longest clip, not to a global `max_frames`.

    `preload` holds every featurised clip in RAM (0.43 GB for the 5,387 train triplets), which
    removes two small disk reads per sample per step. `text_emb` holds the instruction
    embeddings precomputed once (2.25 GB), which removes a T5 forward from every step and lets
    the encoder be dropped from VRAM entirely.
    """

    def __init__(self, keys, cache_dir, texts, mean, std, max_frames,
                 preload=True, text_emb=None):
        self.keys, self.dir, self.texts = keys, cache_dir, texts
        self.mean, self.std, self.max_frames = mean, std, max_frames
        self.text_emb, self.mem = text_emb, None
        if preload:
            self.mem = {}
            for k in tqdm(keys, desc="preloading", leave=False):
                self.mem[k] = (self._load(k, "s"), self._load(k, "t"))
            self.lengths = [len(self.mem[k][0]) for k in keys]
        else:
            # Read the length out of the .npy HEADER via mmap. Calling _pair() here would
            # load and normalise all 5,387 clips just to discard them, which defeats the
            # point of not preloading in the first place.
            self.lengths = [min(np.load(os.path.join(cache_dir, f"{k}_s.npy"),
                                        mmap_mode="r").shape[0], max_frames)
                            for k in keys]

    def _load(self, k, which):
        a = np.load(os.path.join(self.dir, f"{k}_{which}.npy"))[: self.max_frames]
        return ((a - self.mean) / self.std).astype(np.float32)

    def _pair(self, k):
        return self.mem[k] if self.mem is not None else (self._load(k, "s"), self._load(k, "t"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        S, T = self._pair(k)
        item = {"source": torch.from_numpy(S), "target": torch.from_numpy(T),
                "length": len(S), "text": self.texts[k], "keyid": k}
        if self.text_emb is not None:
            item["context"] = self.text_emb[k]
        return item


def collate(batch, max_frames=None):
    """Pad a batch of variable-length clips. `max_frames=None` pads to the batch's own
    longest clip; an integer pads to that fixed width (`--pad_to max`).

    Padding frames are excluded from the loss AND masked out of self-attention (the (B, F)
    mask is expanded to (B, F*G) and used as a key-padding mask), so batch-max padding is
    bit-identical to fixed-width padding — verified at 5.3e-06 on real frames — while
    costing ~2.7x less compute per step, and up to ~7x in the quadratic attention term.
    """
    F = max_frames or max(b["length"] for b in batch)
    D = batch[0]["source"].shape[1]

    def pad(x):
        return x if len(x) == F else torch.cat([x, torch.zeros(F - len(x), D, dtype=x.dtype)])

    out = {"source": torch.stack([pad(b["source"]) for b in batch]),
           "target": torch.stack([pad(b["target"]) for b in batch]),
           "length": torch.tensor([b["length"] for b in batch]),
           "text":   [b["text"] for b in batch],
           "keyid":  [b["keyid"] for b in batch]}
    if "context" in batch[0]:
        out["context"] = torch.stack([b["context"] for b in batch])
    return out


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar-length clips, in shuffled batch order.

    Batch-max padding only pays off if a batch's clips are actually similar in length — one
    100-frame clip drags a batch of 40-frame ones up with it. Sorting into buckets and then
    shuffling the BATCHES keeps the stochasticity that matters (which batches, in what order)
    while making each batch homogeneous.
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True, seed=0):
        self.lengths, self.bs = list(lengths), batch_size
        self.shuffle, self.drop_last, self.seed, self.epoch = shuffle, drop_last, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = np.argsort(np.array(self.lengths) + g.uniform(0, 1e-3, len(self.lengths)))
        batches = [idx[i:i + self.bs].tolist() for i in range(0, len(idx), self.bs)]
        if self.drop_last and batches and len(batches[-1]) < self.bs:
            batches.pop()
        if self.shuffle:
            g.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.lengths)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs


# ── grounding labels ────────────────────────────────────────────────────────────

def velocity_diff_map(S, T, group_channels):
    """(F-1, G) per-(frame, group) velocity difference between source and target.

    Velocity, not pose: a constant per-performer rest-pose offset cancels in the derivative,
    and 94.6 % of MotionFix pairs are two different captures. Measured best of six readouts
    on the GROUP axis (docs/MaskOptions.md §20.1c): 77-78 % top-1 vs 75.4 % for raw pose.
    """
    n = min(len(S), len(T))
    dv = np.abs(np.diff(S[:n], axis=0) - np.diff(T[:n], axis=0))
    if not len(dv):
        return None
    return np.stack([dv[:, ch].mean(1) for ch in group_channels], 1)


def diff_groups(vmap, ratio, kmax):
    """Group set: everything within `ratio` of the top group's mass, capped at `kmax`."""
    mass = vmap.mean(0)
    if mass.max() <= 0:
        return []
    keep = np.where(mass >= ratio * mass.max())[0]
    return [int(g) for g in keep[np.argsort(-mass[keep])][:kmax]]


def diff_region(vmap, groups, keep_frac):
    """(F, G) BINARY region for the spatiotemporal L_token target.

    Inside the selected group rows, keep the busiest `keep_frac` of frames; everything else
    is 0. Binary on purpose — a soft target caps m below 1 and leaves (1 - m)^2 with an
    irreducible floor, i.e. permanent gradient toward an unreachable optimum.

    ⚠ THE TEMPORAL AXIS IS MEASURED TO CARRY ALMOST NO PAIR-SPECIFIC SIGNAL. Share of mass
    in the busiest 20 % of frames: real pair 0.358 vs a SHUFFLED pair 0.346 — a gap of
    +0.012, against +0.19..+0.25 for the group axis. Two corrected read-outs were tried and
    neither helped (normalised +0.013, excess +0.021). Both real and shuffled sit well above
    uniform (0.200), so motion differences are genuinely bursty — but a random pairing is
    equally bursty, which means the burstiness tracks "where either clip moves fast", not
    "where the edit happened". `keep_frac=1.0` disables the temporal restriction and
    reproduces the group-set behaviour exactly.
    """
    F, G = vmap.shape
    M = np.zeros((F, G), dtype=np.float32)
    if not groups:
        return M
    if keep_frac >= 1.0:
        M[:, groups] = 1.0
        return M
    activity = vmap[:, groups].sum(1)
    k = max(1, int(round(keep_frac * F)))
    frames = np.argsort(-activity)[:k]
    M[np.ix_(frames, groups)] = 1.0
    return M


def build_label_cache(args, keys, cache_dir, texts, encoder, config, group_mode, group_channels):
    """{keyid: [item]} for grounding_loss. Keyed by KEYID, not text: a diff-derived label
    depends on the motion pair, and two triplets can share an instruction."""
    from editing.masking import semantic_token_subset     # local: training must not import editing

    lat_names = {g for g in GROUP_NAMES if g.startswith(("left_", "right_"))}
    use_verbs = bool(config.get("attn_ground_verbs", False))
    cache, stats = {}, {"parser": 0, "diff": 0, "none": 0}

    for k in tqdm(keys, desc="labels"):
        text = texts[k]
        items = []
        if args.ground_labels in ("parser_first", "parser_only"):
            items = to_items(text, encoder.token_spans(text), group_mode, include_verbs=use_verbs)
        if items:
            stats["parser"] += 1
        elif args.ground_labels in ("parser_first", "diff_only"):
            S = np.load(os.path.join(cache_dir, f"{k}_s.npy"))
            T = np.load(os.path.join(cache_dir, f"{k}_t.npy"))
            vmap = velocity_diff_map(S, T, group_channels)
            groups = diff_groups(vmap, args.diff_ratio, args.diff_max) if vmap is not None else []
            pos, lab = encoder.token_info(text)
            cols = semantic_token_subset(pos, lab)          # supervise the content words
            if groups and cols:
                tier1 = (args.diff_tier1 and len(groups) == 1
                         and GROUP_NAMES[groups[0]] in lat_names)
                item = {"W": list(cols), "S": list(groups),
                        "tier": 1 if tier1 else 2, "lat": bool(tier1)}
                if args.diff_temporal < 1.0:
                    item["M"] = diff_region(vmap, groups, args.diff_temporal)
                items = [item]
                stats["diff"] += 1
            else:
                stats["none"] += 1
        else:
            stats["none"] += 1
        if items:
            cache[k] = items
    return cache, stats


# ── one epoch ───────────────────────────────────────────────────────────────────

def run_epoch(model, ema, schedule, opt, sched, scaler, loader, device, args, grounding,
              text_encoder, epoch, geo_fn, train=True):
    model.train(train)
    tot = {"loss": 0.0, "diff": 0.0, "ground": 0.0, "m_S": 0.0, "n_items": 0.0}
    n_steps = n_skipped = 0
    use_ground = grounding is not None and grounding.active(epoch)
    amp = resolve_amp_dtype(args.amp_dtype)

    pbar = tqdm(loader, desc=f"{'train' if train else 'val'} {epoch}", leave=False)
    for batch in pbar:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        B, F, _ = source.shape
        lengths = batch["length"].to(device)
        frame_mask = (torch.arange(F, device=device)[None] < lengths[:, None])

        # THE OBJECTIVE: noise the SOURCE, regress the TARGET.
        t = torch.randint(args.t_min, args.t_max + 1, (B,), device=device)
        x_t, _ = schedule.q_sample(source, t)

        if "context" in batch:                       # precomputed once, no T5 in the loop
            context = batch["context"].to(device, non_blocking=True)
        else:
            with torch.no_grad():
                context = text_encoder.encode(batch["text"])
        keep = torch.rand(B, device=device) >= args.cfg_dropout
        ctx = context.clone()
        if (~keep).any():
            ctx[~keep] = model.null_text_emb.to(ctx.dtype)

        g_layer = grounding.pick_layer() if use_ground else None
        with autocast(device_type=device.type, dtype=amp, enabled=amp != torch.float32):
            pred = model(x_t, t, ctx, mask=frame_mask, supervise_layer=g_layer)
            # An x0 head outputs the clean signal directly; an eps head is asked for the
            # noise that would carry x_t to the TARGET, which is the same statement one
            # level down and keeps this script usable on either checkpoint.
            goal = (target if schedule.predict_type == "x0"
                    else schedule.predict_eps_from_x0(x_t, t, target))
            m = frame_mask[..., None].float()
            diff = ((pred - goal) ** 2 * m).sum() / m.sum().clamp(min=1) / pred.shape[-1]
            loss = diff

            g_val, A = torch.zeros((), device=device), None
            if use_ground:
                A = model.get_sup_attn(g_layer)                     # (B, F, G, L), graph kept
                w = 1.0 - schedule.alphas_cumprod[t]                # pressure at HIGH noise
                g_val, g_stats = grounding_loss(
                    A, batch["keyid"], grounding.cache, frame_mask, keep,
                    sample_weight=w, lambda_mirror=grounding.mirror,
                    margin=grounding.margin, mirror_mat=grounding.mirror_mat,
                    lambda_even=grounding.even)
                loss = loss + grounding.weight * g_val
                tot["m_S"] += float(g_stats.get("m_S", 0.0))
                tot["n_items"] += float(g_stats.get("n_items", 0.0))

            if geo_fn is not None:
                x0_hat = pred if schedule.predict_type == "x0" else \
                         schedule.predict_x0_from_eps(x_t, t, pred)
                # x0_confidence_weight down-weights high-noise samples, where x0_hat is a
                # poor clean-signal estimate and SMPL FK amplifies the error.
                geo = geo_fn(x0_hat, target, frame_mask,
                             sample_weight=schedule.x0_confidence_weight(t))
                for name, wgt in (("pos", args.geo_pos_weight), ("vel", args.geo_vel_weight),
                                  ("foot", args.geo_foot_weight)):
                    if wgt > 0 and torch.isfinite(geo[name]):
                        loss = loss + wgt * geo[name]

        if not torch.isfinite(loss):
            # Same skip path as training/epoch.py, and it is counted for the same reason: a
            # non-finite loss is divergence, not noise, and a run where almost no gradient
            # reaches the optimiser must stop rather than burn the night. This project has
            # lost two overnight runs to exactly that (docs/FINDINGS.md, fp16 overflow).
            #
            # INVARIANT: every name below holds, or may hold, a tensor with grad_fn from
            # this iteration. Leaving any of them bound keeps the whole graph alive across
            # the NEXT forward, so the process holds two graphs at once — measured at 1.71x
            # peak on the real loop, which turns a diverging run into a CUDA OOM instead of
            # a skipped step (docs/FINDINGS.md). Anything new that carries a graph goes here.
            if train:
                opt.zero_grad(set_to_none=True)
            loss = diff = g_val = pred = goal = A = None
            n_skipped += 1
            continue
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update_from(model)
        tot["loss"] += loss.detach().item(); tot["diff"] += diff.detach().item()
        tot["ground"] += g_val.detach().item()
        n_steps += 1
        pbar.set_postfix(loss=f"{tot['loss']/max(n_steps,1):.4f}")
    if n_skipped:
        print(f"  WARNING: {'train' if train else 'val'} epoch {epoch} skipped "
              f"{n_skipped}/{len(loader)} steps with a non-finite loss. The reported loss "
              f"averages the SURVIVORS only, so it understates the damage.")
    if train and n_skipped > MAX_SKIP_FRACTION * max(len(loader), 1):
        raise RuntimeError(
            f"epoch {epoch}: {n_skipped}/{len(loader)} steps non-finite "
            f"(> {MAX_SKIP_FRACTION:.0%}). Almost no gradient is reaching the optimiser — "
            f"stopping instead of training on nothing. Try --amp_dtype bf16 or a lower --lr.")
    n = max(n_steps, 1)
    return {k: v / n for k, v in tot.items()} | {"steps": n_steps, "skipped": n_skipped}


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(args.output_dir.rstrip("/")), "ft_cache")

    # The raw triplet dump is ~5 GB. Featurise from it, keep only the instructions, and free
    # it BEFORE the model and text encoder are loaded — otherwise peak RSS is dump + model +
    # T5 + one fork per dataloader worker, which OOMs a 16 GB machine.
    import gc, joblib
    print("loading MotionFix triplets (~5 GB, freed after caching) …")
    data = joblib.load(os.path.join(args.motionfix_root, "motionfix.pth.tar"))
    splits = json.load(open(os.path.join(args.motionfix_root, "splits.json")))
    split_keys, texts = {}, {}
    for split in ("train", "val"):
        keys = [k for k in splits[split] if k in data]
        kept, skipped = build_cache(args, keys, data, os.path.join(cache_dir, split))
        split_keys[split] = (kept, skipped)
        texts.update({k: data[k]["text"] for k in kept})
    del data, splits
    gc.collect()
    print(f"  triplet dump released; {len(texts)} instructions retained")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema_init)
    if config.get("feature_mode") != "smplh":
        raise SystemExit(f"SMPL-H only; checkpoint is {config.get('feature_mode')!r}.")
    feature_mode, is_group, group_mode, gnames = resolve_group_context(config)
    schedule = NoiseSchedule.from_config(config, device=device)
    text_encoder = build_text_encoder(config, device=device)
    print(f"loaded {args.checkpoint}  predict_type={schedule.predict_type}  G={len(gnames)}")
    if not 0 <= args.t_min < args.t_max < schedule.T:
        raise SystemExit(f"need 0 <= t_min < t_max < {schedule.T}")
    print(f"timestep band [{args.t_min}, {args.t_max}]  -> source is "
          f">={float(schedule.sqrt_alphas_cumprod[args.t_max])*100:.0f}% present in every sample")

    # Normalisation must be the checkpoint's own, not a path someone retyped: the wrong
    # Mean/Std is not an error, it is a silent 135-channel rescale of every sample.
    if args.smplh_data_root is None:
        root = config.get("data_root")
        if not root:
            raise SystemExit(
                "checkpoint config.json has no 'data_root'; pass --smplh_data_root explicitly.")
        # The checkpoint stores it as it was typed at training time, i.e. relative to the repo
        # root. Resolve it there rather than against cwd, so the script works from anywhere.
        args.smplh_data_root = (root if os.path.isabs(root)
                                else os.path.join(os.path.dirname(src_dir), root))
        print(f"  normalisation from the checkpoint's own data_root: {args.smplh_data_root}")
    elif config.get("data_root") and os.path.abspath(args.smplh_data_root) != os.path.abspath(config["data_root"]):
        print(f"  ! --smplh_data_root {args.smplh_data_root} differs from the checkpoint's "
              f"{config['data_root']} — using yours, but the stats must still match the weights.")
    for f in ("Mean.npy", "Std.npy"):
        if not os.path.exists(os.path.join(args.smplh_data_root, f)):
            raise SystemExit(f"no {f} in {args.smplh_data_root}")
    mean = np.load(os.path.join(args.smplh_data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.smplh_data_root, "Std.npy"))
    if mean.shape[0] != 135:
        raise SystemExit(f"expected 135-d SMPL-H stats, got {mean.shape[0]} in {args.smplh_data_root}")

    loaders, caches, samplers = {}, {}, {}
    for split in ("train", "val"):
        kept, skipped = split_keys[split]
        cdir = os.path.join(cache_dir, split)
        lc, lstats = ({}, {"parser": 0, "diff": 0, "none": len(kept)})
        if args.ground_labels != "off":
            lc, lstats = build_label_cache(args, kept, cdir, texts, text_encoder, config,
                                           group_mode, model.group_channels)
        caches[split] = lc
        temb = None
        if args.precompute_text:
            temb = {}
            with torch.no_grad():
                for i in tqdm(range(0, len(kept), 128), desc=f"{split} text", leave=False):
                    chunk = kept[i:i + 128]
                    e = text_encoder.encode([texts[k] for k in chunk]).float().cpu()
                    temb.update({k: e[j] for j, k in enumerate(chunk)})
        ds = TripletDataset(kept, cdir, texts, mean, std, args.max_frames,
                            preload=args.preload, text_emb=temb)
        if args.bucket_by_length and args.pad_to == "batch":
            sampler = LengthBucketSampler(ds.lengths, args.batch_size,
                                          shuffle=(split == "train"),
                                          drop_last=(split == "train"), seed=args.seed)
            samplers[split] = sampler
            loaders[split] = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                        num_workers=args.num_workers, pin_memory=True,
                                        persistent_workers=args.num_workers > 0)
        else:
            cf = collate if args.pad_to == "batch" else \
                 partial(collate, max_frames=args.max_frames)
            loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                        shuffle=(split == "train"), collate_fn=cf,
                                        num_workers=args.num_workers, pin_memory=True,
                                        persistent_workers=args.num_workers > 0,
                                        drop_last=(split == "train"))
        print(f"  {split}: {len(kept)} triplets ({len(skipped)} too short) | labels "
              f"parser {lstats['parser']}  diff {lstats['diff']}  none {lstats['none']}")

    if args.precompute_text:
        # Every instruction is embedded and the labels are built, so the encoder has no
        # remaining job. Dropping it frees its VRAM for a larger --batch_size.
        del text_encoder
        text_encoder = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("  text encoder released (embeddings precomputed)")

    grounding = None
    if args.ground_labels != "off" and args.attn_ground_weight > 0:
        grounding = GroundingConfig(
            weight=args.attn_ground_weight,
            layers=resolve_ground_layers(args.attn_ground_layers, config.get("num_layers", 8)),
            mirror=args.attn_ground_mirror, even=args.attn_ground_even,
            margin=args.attn_ground_margin, warmup_epochs=args.attn_ground_warmup_epochs,
            cache=caches["train"], group_channels=model.group_channels,
            mirror_mat=mirror_matrix(group_mode))

    geo_fn = None
    if max(args.geo_pos_weight, args.geo_vel_weight, args.geo_foot_weight) > 0:
        from utils.skeleton import build_geo_fn
        geo_fn, label = build_geo_fn(
            feature_mode,
            torch.from_numpy(mean).float().to(device), torch.from_numpy(std).float().to(device),
            device, pos_weight=args.geo_pos_weight, vel_weight=args.geo_vel_weight,
            foot_weight=args.geo_foot_weight, smplh_model_path=args.smplh_model_path)
        print(f"  geometric losses: {label}")

    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(loaders["train"]), 1)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1))
                       * 0.5 * (1 + np.cos(np.pi * min(s / max(total, 1), 1.0))))
    scaler = GradScaler(device.type, enabled=resolve_amp_dtype(args.amp_dtype) == torch.float16)
    print(f"  {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total} steps  "
          f"(lr {args.lr:g}, warmup {args.warmup_steps}, ema {args.ema_decay})")

    cfg_out = {**config, "finetuned_from": os.path.abspath(args.checkpoint),
               "finetune": {k: v for k, v in vars(args).items()}}
    hist = []
    for epoch in range(args.epochs):
        if "train" in samplers:
            samplers["train"].set_epoch(epoch)
        tr = run_epoch(model, ema, schedule, opt, sched, scaler, loaders["train"], device,
                       args, grounding, text_encoder, epoch, geo_fn, train=True)
        line = (f"epoch {epoch:3d}  loss {tr['loss']:.4f}  diff {tr['diff']:.4f}  "
                f"ground {tr['ground']:.4f}  m_S {tr['m_S']:.3f}  items/step {tr['n_items']:.1f}"
                + (f"  SKIPPED {tr['skipped']}" if tr["skipped"] else ""))
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                if grounding is not None:
                    grounding.cache = caches["val"]
                va = run_epoch(model, ema, schedule, opt, sched, scaler, loaders["val"], device,
                               args, grounding, text_encoder, epoch, geo_fn, train=False)
                if grounding is not None:
                    grounding.cache = caches["train"]
            line += f"  | val {va['loss']:.4f}"
            hist.append({"epoch": epoch, **tr, "val": va})
        else:
            hist.append({"epoch": epoch, **tr})
        print(line)
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(hist[-1]) + "\n")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(args.output_dir, epoch, model, ema, opt, sched, cfg_out)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
