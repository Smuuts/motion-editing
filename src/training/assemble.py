"""
Building a training run's components from one config dict.

Split out of `Trainer` so each piece is a plain function of the config: the wiring stays
readable, and the "why this default" notes sit next to the code that applies them rather
than inside a constructor.
"""

import glob
import os

import numpy as np
import torch
from torch.amp import GradScaler

from data.body_part_labels import build_cache, load_cache
from data.dataset import build_dataloader
from model.dit import build_model
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder, get_encoder_dims
from training.config import resolve_amp_dtype
from training.grounding import GroundingConfig, mirror_matrix, resolve_ground_layers
from training.optim import build_optimizer, build_scheduler
from utils.ema import EMA
from utils.logger import get_logger
from utils.skeleton import build_geo_fn

log = get_logger(__name__)


def build_data(config):
    """Train and (optionally) validation dataloaders, with identical settings."""
    kwargs = dict(batch_size=config["batch_size"], max_frames=config["max_frames"],
                  num_workers=config["num_workers"],
                  feature_mode=config["feature_mode"])
    train_loader = build_dataloader(config["data_root"], split="train", **kwargs)
    log.info("Training on %d clips", len(train_loader.dataset))

    val_loader = None
    if config["val_every"] > 0:
        val_loader = build_dataloader(config["data_root"], split="val", **kwargs)
        log.info("Validation on %d clips (every %d epochs)",
                 len(val_loader.dataset), config["val_every"])
    return train_loader, val_loader


def build_geometric_losses(config, dataset, device):
    """(geo_fn, weights, geo_conf_weight) — the MDM-style geometric loss terms.

    `geo_fn` is a closure (x0_pred, motion, mask) -> {pos, vel, foot}, so the epoch loop
    stays representation-agnostic.
    """
    mean = torch.from_numpy(dataset.mean).float().to(device)
    std = torch.from_numpy(dataset.std).float().to(device)
    weights = {"pos": config["hml3d_pos_weight"], "vel": config["hml3d_vel_weight"],
               "foot": config["hml3d_foot_weight"]}

    geo_fn, label = build_geo_fn(
        config["feature_mode"], mean, std, device,
        pos_weight=weights["pos"], vel_weight=weights["vel"],
        foot_weight=weights["foot"], smplh_model_path=config["smplh_model_path"])
    if geo_fn is not None:
        active = ", ".join(f"{k}={w}" for k, w in weights.items() if w)
        log.info("%s geometric losses enabled (%s)", label, active)

    # AUTO: the alpha_bar_t damping only makes sense when x0 is DERIVED from eps. An x0
    # head outputs x0 directly, so there is no error amplification to damp.
    geo_conf_weight = (config["predict_type"] == "eps"
                       if config["geo_conf_weight"] is None
                       else config["geo_conf_weight"])
    if config["predict_type"] == "x0":
        _report_x0_weighting(config, geo_conf_weight)
    return geo_fn, weights, geo_conf_weight


def _report_x0_weighting(config, geo_conf_weight):
    """x0 runs are not loss-comparable to eps runs; say so once, at startup."""
    gamma = config["snr_gamma"]
    weighting = ("plain (unweighted) — 40% of training weight on t>=600 vs 3% for eps"
                 if gamma == 0.0 else
                 f"min(SNR,{gamma}) x0-form — WARNING: this reproduces the eps "
                 f"baseline's weighting (~3% on t>=600)")
    forced = "ON (forced via --geo_conf_weight)" if geo_conf_weight else "OFF (auto)"
    log.info("Prediction target: x0. Loss weighting: %s. Geometric-loss confidence "
             "weight %s. Loss values are NOT comparable to eps runs — compare x0 runs "
             "only against other x0 runs.", weighting, forced)


def resolve_text_encoder(config, dataset, context_dim, text_seq_len, device):
    """The text encoder, or None when precomputed embeddings make it unnecessary.

    Checks first that the embeddings match the configured encoder — a silent mismatch
    would train against embeddings from a different text model.
    """
    emb_dir = dataset.text_emb_dir
    if emb_dir is None:
        return build_text_encoder(config, device=device)

    sample = next(glob.iglob(os.path.join(emb_dir, "*.npy")), None)
    if sample:
        shape = np.load(sample).shape                 # (num_ann, L, dim)
        if (int(shape[1]), int(shape[2])) != (text_seq_len, context_dim):
            raise ValueError(
                f"Precomputed embeddings in '{emb_dir}' have shape "
                f"(*, {shape[1]}, {shape[2]}), but the configured encoder "
                f"(--text_encoder {config['text_encoder']}) expects "
                f"(*, {text_seq_len}, {context_dim}). Re-run precompute_text.py with "
                f"matching encoder settings, or point --data_root to a directory "
                f"without a stale text_emb/ folder.")
    log.info("Precomputed text embeddings found — skipping %s model load.",
             config["text_encoder"].upper())
    return None


def build_model_stack(config, dataset, device):
    """(text_encoder, model, ema, schedule, amp_dtype, scaler) from the config."""
    context_dim, text_seq_len = get_encoder_dims(config)
    text_encoder = resolve_text_encoder(config, dataset, context_dim, text_seq_len,
                                        device)
    if config["ctx_pad_mask"] and config["text_encoder"] == "clip":
        log.info("NOTE: --ctx_pad_mask is a no-op with the CLIP encoder (CLIP does not "
                 "zero its padding embeddings, so no context column is all-zero).")

    # The FULL config goes through, so new model-defining keys never need hand-copying
    # here — the same pattern as utils/model_io.load_model.
    model = build_model({**config,
                         "input_dim": dataset.feature_dim,
                         "context_dim": context_dim,
                         "text_seq_len": text_seq_len}, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %.1fM", n_params / 1e6)

    ema = EMA(model, decay=config["ema_decay"])
    schedule = NoiseSchedule.from_config(config, device=device)
    # GradScaler exists to keep fp16 GRADIENTS out of the subnormal range. bf16 has
    # fp32's exponent range and fp32 needs no scaling at all, so it is enabled for fp16
    # only — leaving it on under bf16 costs a pointless inf-check per step.
    amp_dtype = resolve_amp_dtype(config.get("amp_dtype", "auto"))
    scaler = GradScaler(device=device.type,
                        enabled=(device.type == "cuda" and amp_dtype is torch.float16))
    log.info("AMP: autocast dtype %s  (GradScaler %s)",
             str(amp_dtype).replace("torch.", ""),
             "on" if scaler.is_enabled() else "off")
    return text_encoder, model, ema, schedule, amp_dtype, scaler


def _ground_cache(config, path, text_encoder):
    """The caption -> (columns, groups) label cache, built offline once.

    PRECOMPUTED, not parsed online, and the reasons are worth writing down because "just
    call the parser in the loop" looks simpler:
      - the map is a pure function of the caption, so recomputing it every epoch is 500x
        the work for the same answer;
      - the text COLUMNS in `W` come from the tokeniser's offset mapping, so an online
        path would have to run the HF tokenizer inside the training step;
      - the coverage/balance gate (src/probe_ground_labels.py) is run against this exact
        file, so what was audited is what trains. An online parser could drift from the
        audited artefact with nothing to notice.
    Built here if absent — a run should not fail on a missing derived artefact — which
    needs the text encoder, i.e. exactly the case the caller already checked.
    """
    if os.path.exists(path):
        return load_cache(path)
    log.info("Grounding labels not found at %s — building (one offline pass over the "
             "captions; see src/probe_ground_labels.py for the audit).", path)
    return build_cache(config["data_root"], text_encoder,
                       group_mode=config.get("group_mode", "parts"), out_path=path,
                       include_verbs=config.get("attn_ground_verbs", True))


def _check_grounding_preconditions(config, model, dataset):
    """Refuse a grounding run that cannot work, before 500 epochs prove it."""
    # A flat backbone has a single token per frame, so S is the whole group axis and
    # L_token = (1 - mass(S))^2 is identically 0 — the loss would run and teach nothing.
    if getattr(model, "G", 1) < 2:
        raise ValueError(
            "--attn_ground_weight needs a grouped backbone (--feature_mode "
            "humanml3d|smplh): the loss supervises WHICH body-part token a word attends "
            "to, and a flat MotionDiT has only one token per frame.")
    # The labels are keyed by CAPTION STRING and looked up in the training loop, so a run
    # against precomputed text_emb/ has nothing to look them up with — the dataset
    # returns `context` and drops the text entirely. Fail here, loudly, rather than
    # silently training with zero supervised items.
    if dataset.text_emb_dir is not None:
        raise ValueError(
            f"--attn_ground_weight needs the raw captions, but {dataset.text_emb_dir} "
            f"holds precomputed text embeddings, so the dataset returns 'context' "
            f"instead of 'text'. Point --data_root at a copy without text_emb/, or move "
            f"that directory aside.")


def build_grounding(config, model, dataset, text_encoder):
    """The TokenCompose grounding config, resolved once.

    Must run after the model exists: the layer set and the mirror matrix depend on the
    backbone's depth and token axis.
    """
    if not config.get("attn_ground_weight", 0.0) > 0.0:
        return GroundingConfig()
    _check_grounding_preconditions(config, model, dataset)

    cache_path = (config.get("attn_ground_cache")
                  or os.path.join(config["data_root"], "ground_labels.json"))
    cache = _ground_cache(config, cache_path, text_encoder)
    window = config.get("attn_ground_window")
    cfg = GroundingConfig(
        weight=config["attn_ground_weight"],
        layers=resolve_ground_layers(config.get("attn_ground_layers", "middle"),
                                     len(model.blocks)),
        mirror=config.get("attn_ground_mirror", 1.0),
        even=config.get("attn_ground_even", 0.1),
        margin=config.get("attn_ground_margin", 0.1),
        warmup_epochs=config.get("attn_ground_warmup_epochs", 20),
        window=tuple(window) if window else None,
        cache=cache,
        group_channels=model.group_channels,
        monitor=config.get("attn_ground_monitor", True),
        mirror_mat=mirror_matrix(config.get("group_mode", "parts")),
    )
    _report_grounding(cfg, cache, cache_path, model.G)
    return cfg


def _report_grounding(cfg, cache, cache_path, n_groups):
    """Announce the grounding setup, including the chance level for THIS label mix.

    Chance m_S is |S|/G averaged over ITEMS, not 1/G. Only a tier-1 item has a single
    target group; a tier-2 limb pair has two and a locomotion verb has three, and a
    uniform attention map scores |S|/G on each. Printing 1/G understates chance by a lot
    once the label set is not all single-group — measured 0.203 for the nouns-only cache
    and 0.262 with verb labels, against the 0.143 that 1/G gives. It is computed from the
    cache that is actually loaded, so it can never drift from the labels again.
    """
    items = [i for v in cache.values() for i in v]
    chance = sum(len(i["S"]) for i in items) / max(len(items), 1) / n_groups
    sizes = sorted({len(i["S"]) for i in items})
    gate = (f"hard timestep gate t in {list(cfg.window)}" if cfg.window
            else "soft 1-alpha_bar_t weighting (pressure at HIGH noise)")
    log.info("Attention grounding ON (TokenCompose L_token): weight %g, layers %s (one "
             "sampled per step), mirror %g @ margin %g (tier 1), even %g (tier 2), "
             "warmup %d epochs, %s.", cfg.weight, cfg.layers, cfg.mirror, cfg.margin,
             cfg.even, cfg.warmup_epochs, gate)
    log.info("  Labels: %d captions / %d items from %s (target sizes |S| = %s).",
             len(cache), len(items), cache_path, sizes)
    log.info("  Watch train/ground_m_S_epoch (chance %.3f for this label mix, not "
             "1/G = %.3f) and train/ground_src_corr_epoch (kill above ~0.5 and rising).",
             chance, 1 / n_groups)


def build_optim(config, model):
    """Optimiser plus the warmup/cosine LR schedule."""
    optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
    scheduler = build_scheduler(optimizer, config["epochs"], config["warmup_epochs"],
                                decay=not config["no_lr_decay"])
    return optimizer, scheduler
