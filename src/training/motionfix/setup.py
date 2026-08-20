"""
Assembling a fine-tune run: triplet cache, backbone, normalisation, loaders, optimiser.
"""

import gc
import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from model.body_groups import resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.config import resolve_amp_dtype
from training.grounding import GroundingConfig, mirror_matrix, resolve_ground_layers
from training.motionfix.data import (LengthBucketSampler, TripletDataset, build_cache,
                                     collate)
from training.motionfix.labels import build_label_cache
from utils.ema import EMA
from utils.logger import get_logger
from utils.model_io import load_model
from utils.paths import resolve_repo_path

log = get_logger(__name__)

FEATURE_DIM = 135
TEXT_BATCH = 128


@dataclass
class FineTuneContext:
    """Everything `run_finetune` needs, built once."""
    model: Any
    ema: Any
    schedule: Any
    text_encoder: Any
    optimiser: Any
    lr_scheduler: Any
    scaler: Any
    loaders: dict
    samplers: dict
    label_caches: dict
    grounding: Any
    geo_fn: Any
    device: torch.device
    config: dict = field(default_factory=dict)


def prepare_triplet_cache(args, cache_dir):
    """Featurise the train/val triplets, then release the raw dump.

    The dump is ~5 GB. It is freed BEFORE the model and text encoder load — otherwise
    peak RSS is dump + model + T5 + one fork per dataloader worker, which OOMs a 16 GB
    machine.
    """
    import joblib

    log.info("loading MotionFix triplets (~5 GB, freed after caching) …")
    data = joblib.load(os.path.join(args.motionfix_root, "motionfix.pth.tar"))
    with open(os.path.join(args.motionfix_root, "splits.json")) as f:
        splits = json.load(f)

    split_keys, texts = {}, {}
    for split in ("train", "val"):
        keys = [k for k in splits[split] if k in data]
        kept, skipped = build_cache(args, keys, data, os.path.join(cache_dir, split))
        split_keys[split] = (kept, skipped)
        texts.update({k: data[k]["text"] for k in kept})

    del data, splits
    gc.collect()
    log.info("  triplet dump released; %d instructions retained", len(texts))
    return split_keys, texts


def load_backbone(args, device):
    """The pretrained SMPL-H checkpoint plus everything derived from its config."""
    model, config = load_model(args.checkpoint, device=device,
                               use_ema=not args.no_ema_init)
    if config.get("feature_mode") != "smplh":
        raise SystemExit(f"SMPL-H only; checkpoint is {config.get('feature_mode')!r}.")

    group_context = resolve_group_context(config)
    schedule = NoiseSchedule.from_config(config, device=device)
    text_encoder = build_text_encoder(config, device=device)
    log.info("loaded %s  predict_type=%s  G=%d", args.checkpoint, schedule.predict_type,
             len(group_context[3]))

    if not 0 <= args.t_min < args.t_max < schedule.T:
        raise SystemExit(f"need 0 <= t_min < t_max < {schedule.T}")
    present = float(schedule.sqrt_alphas_cumprod[args.t_max]) * 100
    log.info("timestep band [%d, %d]  -> source is >=%.0f%% present in every sample",
             args.t_min, args.t_max, present)
    return model, config, schedule, text_encoder, group_context


def resolve_normalisation(args, config):
    """The (Mean, Std) the model was trained under.

    Defaults to the checkpoint's own `data_root`, because the wrong Mean/Std is not an
    error — it is a silent 135-channel rescale of every sample. The checkpoint stores the
    path as it was typed at training time, i.e. relative to the repo root, so it is
    resolved there rather than against the cwd.
    """
    if args.smplh_data_root is None:
        root = config.get("data_root")
        if not root:
            raise SystemExit("checkpoint config.json has no 'data_root'; pass "
                             "--smplh_data_root explicitly.")
        args.smplh_data_root = resolve_repo_path(root)
        log.info("  normalisation from the checkpoint's own data_root: %s",
                 args.smplh_data_root)
    elif config.get("data_root") and os.path.abspath(args.smplh_data_root) != \
            os.path.abspath(config["data_root"]):
        log.warning("--smplh_data_root %s differs from the checkpoint's %s — using "
                    "yours, but the stats must still match the weights.",
                    args.smplh_data_root, config["data_root"])

    for name in ("Mean.npy", "Std.npy"):
        if not os.path.exists(os.path.join(args.smplh_data_root, name)):
            raise SystemExit(f"no {name} in {args.smplh_data_root}")
    mean = np.load(os.path.join(args.smplh_data_root, "Mean.npy"))
    std = np.load(os.path.join(args.smplh_data_root, "Std.npy"))
    if mean.shape[0] != FEATURE_DIM:
        raise SystemExit(f"expected {FEATURE_DIM}-d SMPL-H stats, got {mean.shape[0]} "
                         f"in {args.smplh_data_root}")
    return mean, std


def _precompute_text(keys, texts, encoder, split):
    """{keyid: (L, dim) embedding} for every instruction, encoded once."""
    embeddings = {}
    with torch.no_grad():
        for i in log.progress(range(0, len(keys), TEXT_BATCH), desc=f"{split} text"):
            chunk = keys[i:i + TEXT_BATCH]
            encoded = encoder.encode([texts[k] for k in chunk]).float().cpu()
            embeddings.update({k: encoded[j] for j, k in enumerate(chunk)})
    return embeddings


def _build_loader(args, dataset, split):
    """A length-bucketed batch sampler where it pays off, else a plain shuffled loader."""
    is_train = split == "train"
    common = dict(collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)
    if args.bucket_by_length and args.pad_to == "batch":
        sampler = LengthBucketSampler(dataset.lengths, args.batch_size,
                                      shuffle=is_train, drop_last=is_train,
                                      seed=args.seed)
        return DataLoader(dataset, batch_sampler=sampler, **common), sampler

    common["collate_fn"] = (collate if args.pad_to == "batch"
                            else partial(collate, max_frames=args.max_frames))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=is_train,
                      drop_last=is_train, **common), None


def build_loaders(args, split_keys, texts, cache_dir, mean, std, text_encoder, config,
                  group_mode, group_channels):
    """Dataloaders, label caches and samplers for both splits."""
    loaders, samplers, label_caches = {}, {}, {}
    for split in ("train", "val"):
        kept, skipped = split_keys[split]
        split_cache = os.path.join(cache_dir, split)

        labels, stats = {}, {"parser": 0, "diff": 0, "none": len(kept)}
        if args.ground_labels != "off":
            labels, stats = build_label_cache(args, kept, split_cache, texts,
                                              text_encoder, config, group_mode,
                                              group_channels)
        label_caches[split] = labels

        embeddings = (_precompute_text(kept, texts, text_encoder, split)
                      if args.precompute_text else None)
        dataset = TripletDataset(kept, split_cache, texts, mean, std, args.max_frames,
                                 preload=args.preload, text_emb=embeddings)
        loader, sampler = _build_loader(args, dataset, split)
        loaders[split] = loader
        if sampler is not None:
            samplers[split] = sampler

        log.info("  %s: %d triplets (%d too short) | labels  parser %d  diff %d  none %d",
                 split, len(kept), len(skipped), stats["parser"], stats["diff"],
                 stats["none"])
    return loaders, samplers, label_caches


def release_text_encoder(device):
    """Free the encoder's VRAM once every instruction is embedded and labelled."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("  text encoder released (embeddings precomputed)")


def build_grounding(args, config, label_cache, group_channels, group_mode):
    """The TokenCompose grounding config, or None when the loss is off."""
    if args.ground_labels == "off" or args.attn_ground_weight <= 0:
        return None
    return GroundingConfig(
        weight=args.attn_ground_weight,
        layers=resolve_ground_layers(args.attn_ground_layers,
                                     config.get("num_layers", 8)),
        mirror=args.attn_ground_mirror, even=args.attn_ground_even,
        margin=args.attn_ground_margin,
        warmup_epochs=args.attn_ground_warmup_epochs,
        cache=label_cache, group_channels=group_channels,
        mirror_mat=mirror_matrix(group_mode))


def build_geometric_losses(args, feature_mode, mean, std, device):
    """The MDM-style geometric loss closure, or None when every weight is zero."""
    if max(args.geo_pos_weight, args.geo_vel_weight, args.geo_foot_weight) <= 0:
        return None
    from utils.skeleton import build_geo_fn

    geo_fn, label = build_geo_fn(
        feature_mode, torch.from_numpy(mean).float().to(device),
        torch.from_numpy(std).float().to(device), device,
        pos_weight=args.geo_pos_weight, vel_weight=args.geo_vel_weight,
        foot_weight=args.geo_foot_weight, smplh_model_path=args.smplh_model_path)
    log.info("  geometric losses: %s", label)
    return geo_fn


def build_optimisation(args, model, steps_per_epoch, device):
    """EMA, AdamW, the warmup+cosine LR schedule and the grad scaler."""
    ema = EMA(model, decay=args.ema_decay)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total = steps_per_epoch * args.epochs

    def lr_lambda(step):
        warmup = min(1.0, (step + 1) / max(args.warmup_steps, 1))
        cosine = 0.5 * (1 + np.cos(np.pi * min(step / max(total, 1), 1.0)))
        return warmup * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler = GradScaler(device.type,
                        enabled=resolve_amp_dtype(args.amp_dtype) == torch.float16)
    log.info("  %d steps/epoch x %d epochs = %d steps  (lr %g, warmup %d, ema %g)",
             steps_per_epoch, args.epochs, total, args.lr, args.warmup_steps,
             args.ema_decay)
    return ema, optimiser, lr_scheduler, scaler
