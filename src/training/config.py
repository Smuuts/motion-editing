"""
Turning CLI args into the run config that is saved next to the checkpoint.

Three things happen here that are easy to get silently wrong, so they live in one
place: the x0/Min-SNR interaction, resuming (saved config wins over defaults but not
over explicit CLI args), and the compatibility defaults for keys that predate a run.
"""

import json
import os
import sys

import torch
from utils.logger import get_logger

log = get_logger(__name__)

# --amp_dtype name -> torch dtype. fp32 disables autocast's reduced precision entirely
# (the GradScaler is disabled with it, since there is nothing to scale).
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_amp_dtype(name: str) -> torch.dtype:
    """`--amp_dtype` -> the dtype autocast runs the forward in.

    "auto" (the default) picks **bf16 wherever the hardware supports it**, else fp16.
    The two carry the same ~3 decimal digits of precision, but bf16 keeps fp32's
    exponent range (max ~3.4e38) where fp16 saturates at 65504 — so an activation that
    merely grows large produces a finite number instead of an `inf` that turns the loss
    non-finite and the step into a no-op. This project has lost two runs to exactly that
    (fp16 activation overflow), so "auto" deliberately does not
    mean "whatever the previous default was".

    Resumed runs keep whatever they trained with (config.py's compat default is fp16),
    so an existing run's numerics do not change under it.
    """
    if name == "auto":
        return (torch.bfloat16 if torch.cuda.is_available()
                and torch.cuda.is_bf16_supported() else torch.float16)
    if name not in AMP_DTYPES:
        raise ValueError(f"amp_dtype must be 'auto' or one of {sorted(AMP_DTYPES)}, got {name!r}")
    return AMP_DTYPES[name]


def explicit_cli_keys(parser, argv=None) -> set[str]:
    """Arg dest names that were explicitly passed on the command line — the difference
    between "the user asked for this" and "this is just the default"."""
    argv = sys.argv[1:] if argv is None else argv
    return {a.dest for a in parser._actions if any(o in argv for o in a.option_strings)}


def _apply_x0_defaults(config, cli_keys):
    """Under an x0 head, default Min-SNR OFF.

    Min-SNR weights ‖x̂0 − x0‖² by min(SNR_t, γ), which at high t IS the ε-objective's
    weighting — so applying it under an x0 head reproduces the ε baseline almost
    exactly (3.0% of training weight on t >= 600) and cancels the entire point of the
    x0 head, which is that the model must reconstruct the clip at EVERY noise level —
    the regime where the caption is the only information available. Plain unweighted x0
    loss puts 40% of the weight on t >= 600, and is what MDM and MotionCLR do.
    """
    if config["predict_type"] == "x0" and "snr_gamma" not in cli_keys:
        config["snr_gamma"] = 0.0


def _merge_resumed(config, cli_keys, output_dir):
    """Overlay the saved config of the run being resumed, keeping explicit CLI args.

    Keys absent from the saved config must NOT pick up today's CLI defaults: flipping
    the attention regime or the prediction target mid-training changes what the weights
    mean (attn_sink would additionally fail the state_dict load — no sink_logit params).
    """
    saved_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(saved_path):
        log.info(f"Resume: no saved config at {saved_path}, using CLI args only.")
        return

    with open(saved_path) as f:
        saved = json.load(f)
    preserved = cli_keys | {"resume", "output_dir"}
    for k, v in saved.items():
        if k not in preserved:
            config[k] = v

    for key, legacy in (("ctx_pad_mask", False), ("attn_sink", False),
                        ("predict_type", "eps"), ("amp_dtype", "fp16")):
        if key not in saved and key not in cli_keys:
            config[key] = legacy
            log.info(f"Resume: saved config predates {key} — keeping {legacy!r} to match "
                  f"how this run trained (pass --{key} to override).")
    log.info(f"Resume: loaded config from {saved_path} "
          f"(overridden by CLI args: {sorted(cli_keys - {'resume', 'output_dir'})})")


def resolve_config(parser, args) -> dict:
    """Full run config: CLI args, then --config file, then (on resume) the saved config.

    Returns `vars(args)` itself — the live namespace dict — so `args.*` stays in sync
    with every adjustment made here.
    """
    config = vars(args)
    cli_keys = explicit_cli_keys(parser)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    _apply_x0_defaults(config, cli_keys)
    if args.resume:
        _merge_resumed(config, cli_keys, args.output_dir)
    return config


def save_config(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def print_config(config):
    arch = "GroupMotionUNet" if config.get("arch") == "unet" else "MotionDiT"
    log.info(f"Training {arch} with config:")
    for k, v in config.items():
        log.info(f"  {k}: {v}")
