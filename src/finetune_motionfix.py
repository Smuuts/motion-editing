"""
Supervised fine-tune of a pretrained SMPL-H checkpoint on MotionFix triplets.

Noises the SOURCE clip and regresses the TARGET, conditioned on the edit instruction —
the same computation the LEDITS++ editor performs at inference, so no architecture
change is needed. See `training/motionfix/loop.py` for the objective and what it trades
away, and `training/motionfix/labels.py` for where the grounding supervision comes from.

    python src/finetune_motionfix.py \
        --checkpoint runs/exp_smplh_verbs/checkpoint_latest \
        --smplh_data_root data/HumanML3D/HumanML3D_smplh \
        --output_dir runs/ft_motionfix
"""

import os
import random

import numpy as np
import torch

from training.motionfix import setup
from training.motionfix.args import build_parser
from training.motionfix.runner import run_finetune
from utils.cli import add_logging_args, configure_logging, resolve_device
from utils.logger import get_logger

log = get_logger(__name__)


def _default_cache_dir(output_dir: str) -> str:
    """Sibling of the output dir, so an A/B of label sources reuses one featurisation."""
    return os.path.join(os.path.dirname(output_dir.rstrip("/")), "ft_cache")


def build_context(args) -> setup.FineTuneContext:
    """Everything the run needs, in the order the memory budget requires."""
    device = resolve_device(args.device)
    cache_dir = args.cache_dir or _default_cache_dir(args.output_dir)
    split_keys, texts = setup.prepare_triplet_cache(args, cache_dir)

    model, config, schedule, text_encoder, group_context = setup.load_backbone(args, device)
    feature_mode, _, group_mode, _ = group_context
    mean, std = setup.resolve_normalisation(args, config)

    loaders, samplers, label_caches = setup.build_loaders(
        args, split_keys, texts, cache_dir, mean, std, text_encoder, config,
        group_mode, model.group_channels)
    if args.precompute_text:
        # Every instruction is embedded and every label built, so the encoder has no
        # remaining job. Dropping it frees its VRAM for a larger --batch_size.
        text_encoder = None
        setup.release_text_encoder(device)

    ema, optimiser, lr_scheduler, scaler = setup.build_optimisation(
        args, model, max(len(loaders["train"]), 1), device)

    return setup.FineTuneContext(
        model=model, ema=ema, schedule=schedule, text_encoder=text_encoder,
        optimiser=optimiser, lr_scheduler=lr_scheduler, scaler=scaler,
        loaders=loaders, samplers=samplers, label_caches=label_caches,
        grounding=setup.build_grounding(args, config, label_caches["train"],
                                        model.group_channels, group_mode),
        geo_fn=setup.build_geometric_losses(args, feature_mode, mean, std, device),
        device=device, config=config)


def main():
    parser = add_logging_args(build_parser(description=__doc__))
    args = configure_logging(parser.parse_args())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ctx = build_context(args)
    config_out = {**ctx.config, "finetuned_from": os.path.abspath(args.checkpoint),
                  "finetune": dict(vars(args))}
    run_finetune(ctx, args, config_out)


if __name__ == "__main__":
    main()
