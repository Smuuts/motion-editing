"""
Shared checkpoint loading for the inference/analysis scripts (generate.py,
sample_model.py, edit_motion.py, verify_backbone.py, the probe scripts).

All of them do the same thing: read a checkpoint directory's config.json, build a
matching model via model.dit.build_model, and load either the EMA or raw weights.
Keeping this in one place means the model-building kwargs can't drift between scripts.
"""

import os
import json

import torch

from model.dit import build_model
from model.text_encoder import get_encoder_dims
from utils.logger import get_logger

log = get_logger(__name__)


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
    # Pass the FULL saved config through (build_model reads what it knows and
    # ignores the rest) so every model-defining key — including attention-regime
    # flags like ctx_pad_mask/attn_sink — travels with the checkpoint
    # automatically. Rebuilding from a hand-copied subset is how a mask-trained
    # checkpoint once ran unmasked (FID 0.65 -> 27.0).
    # Only derived/inference-specific values are overridden.
    model = build_model({
        **config,
        "context_dim":  context_dim,
        "text_seq_len": text_seq_len,
        "dropout":      0.0,
    }, device=device)

    weights = os.path.join(ckpt_dir, "ema.pt" if use_ema else "model.pt")
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    log.info(f"Loaded: {weights}")
    return model, config
