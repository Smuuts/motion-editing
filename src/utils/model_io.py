"""
Shared checkpoint loading for the inference/analysis scripts (generate.py,
sample_model.py, edit_motion.py, analyse_attention.py, verify_backbone.py).

All of them do the same thing: read a checkpoint directory's config.json, build a
matching model via model.dit.build_model, and load either the EMA or raw weights.
Keeping this in one place means the model-building kwargs can't drift between scripts.
"""

import os
import json

import torch

from model.dit import build_model
from model.text_encoder import get_encoder_dims


def load_config(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        return json.load(f)


def load_model(ckpt_dir: str, device, use_ema: bool = True):
    """Build a model from a checkpoint's config.json and load its weights.

    LEDITS++ inference (and all evaluation scripts) use the EMA weights
    (checkpoint_dir/ema.pt); pass use_ema=False for the raw model.pt.

    Returns (model, config).
    """
    config = load_config(ckpt_dir)
    context_dim, text_seq_len = get_encoder_dims(config)
    model = build_model({
        "feature_mode": config.get("feature_mode", "humanml3d"),
        "input_dim":    config.get("input_dim", 263),
        "latent_dim":   config.get("latent_dim", 512),
        "context_dim":  context_dim,
        "text_seq_len": text_seq_len,
        "num_heads":    config.get("num_heads", 8),
        "num_layers":   config.get("num_layers", 8),
        "max_frames":   config.get("max_frames", 196),
        "dropout":      0.0,
        # Attention regime must match training exactly: mask-trained checkpoints
        # (ctx_pad_mask: true in their config) are out-of-distribution without it,
        # and pre-fix checkpoints (no key) are out-of-distribution with it.
        "ctx_pad_mask": config.get("ctx_pad_mask", False),
    }, device=device)

    weights = os.path.join(ckpt_dir, "ema.pt" if use_ema else "model.pt")
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {weights}")
    return model, config
