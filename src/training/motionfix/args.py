"""
Command-line surface of the MotionFix fine-tune, and where each default comes from.
"""

import argparse

from utils.paths import repo_path

# 5,387 train triplets / batch 64 = 85 steps/epoch. The pretrain was 500 epochs x 335
# steps = ~167k steps; 100 epochs here is ~8.5k, i.e. ~5 % of it, which is the usual
# order for a fine-tune onto a new objective with 4x less data.
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 64
DEFAULT_LR = 1e-5      # 10x below the 1e-4 pretrain LR: the objective changes, not the weights
DEFAULT_EMA = 0.999    # horizon ~1/(1-d) = 1,000 steps. 0.9999 needs 10k and this run is
                       # ~8.5k, so the EMA would never catch up and ema.pt would lag model.pt.
DEFAULT_T_MAX = 800    # sqrt(alpha_bar_800) = 0.305 -> the source is still ~30 % present


def _add_source_args(p):
    p.add_argument("--checkpoint", required=True,
                   help="Pretrained SMPL-H checkpoint dir (config.json + ema.pt/model.pt). "
                        "Fine-tuning starts from its EMA weights unless --no_ema_init.")
    p.add_argument("--smplh_data_root", default=None,
                   help="SMPL-H root holding the 135-d Mean.npy / Std.npy. Default: whatever "
                        "the CHECKPOINT records as its own `data_root`, the only value "
                        "guaranteed to be the normalisation the model trained under — a "
                        "hand-typed path that disagrees corrupts every sample silently. Pass "
                        "one only to override deliberately.")
    p.add_argument("--motionfix_root",
                   default=repo_path("data/motionfix/data/motionfix-dataset"),
                   help="Dir with motionfix.pth.tar + splits.json. Defaults to the copy "
                        "inside this repo, resolved from the package's own location so it "
                        "does not depend on the cwd.")
    p.add_argument("--output_dir", default="runs/ft_motionfix")
    p.add_argument("--cache_dir", default=None,
                   help="Where to cache featurised triplets (default "
                        "<output_dir>/../ft_cache). Built once and reused across runs, so an "
                        "A/B of label sources pays for it once.")
    p.add_argument("--no_ema_init", action="store_true",
                   help="Initialise from model.pt instead of ema.pt.")


def _add_objective_args(p):
    p.add_argument("--t_min", type=int, default=0,
                   help="Lowest diffusion timestep sampled.")
    p.add_argument("--t_max", type=int, default=DEFAULT_T_MAX,
                   help=f"Highest timestep sampled (default {DEFAULT_T_MAX}). Above it the "
                        "source is destroyed and the task degenerates into generating a whole "
                        "motion from a relative instruction. sqrt(alpha_bar): 0.70@500, "
                        "0.58@600, 0.45@700, 0.31@800, 0.00@999.")
    p.add_argument("--cfg_dropout", type=float, default=0.1,
                   help="Probability of replacing the instruction with the null embedding. "
                        "Keep non-zero or classifier-free guidance stops working at inference.")


def _add_grounding_args(p):
    p.add_argument("--ground_labels", default="parser_first",
                   choices=["parser_first", "diff_only", "parser_only", "off"],
                   help="Source of the supervised group set S. 'parser_first' (default) = "
                        "parser where it names a body part, velocity-diff elsewhere. "
                        "'diff_only' = always the motion difference. 'parser_only' = skip "
                        "unlabelled instructions (the control isolating what the diff adds). "
                        "'off' disables the loss.")
    p.add_argument("--attn_ground_weight", type=float, default=5e-3,
                   help="lambda for the TokenCompose term. 5e-3 is the dose measured free of "
                        "FID cost in pretraining; 0.01 cost 31 %% FID.")
    p.add_argument("--attn_ground_layers", default="middle",
                   help="Blocks to supervise. Default matches the pretrained checkpoint.")
    p.add_argument("--attn_ground_mirror", type=float, default=1.0)
    p.add_argument("--attn_ground_even", type=float, default=0.1)
    p.add_argument("--attn_ground_margin", type=float, default=0.1)
    p.add_argument("--attn_ground_warmup_epochs", type=int, default=0,
                   help="0 by default: unlike pretraining, attention is ALREADY grounded "
                        "here, so there is no noise phase to wait out.")
    p.add_argument("--diff_ratio", type=float, default=0.5,
                   help="Diff labels: keep every group holding >= this share of the top "
                        "group's difference mass. 0.5 gives 1-2 groups on the measured "
                        "profiles.")
    p.add_argument("--diff_max", type=int, default=2,
                   help="Diff labels: hard cap on |S|. 2, because top-2 was where measured "
                        "set accuracy plateaued (83.7 %%) while the shuffled control kept "
                        "rising.")
    p.add_argument("--diff_temporal", type=float, default=0.5,
                   help="Diff labels: make the L_token target SPATIOTEMPORAL by keeping only "
                        "the busiest this-fraction of frames inside the selected groups. 1.0 "
                        "disables it and supervises the whole group row, bit-identically. "
                        "Measured: the temporal axis of the velocity difference carries a "
                        "real-vs-shuffled gap of only +0.012, against +0.19..+0.25 on the "
                        "group axis — so this mostly sharpens the target around frames where "
                        "EITHER clip moves fast, not where the edit is. 0.5 is deliberately "
                        "loose for that reason.")
    p.add_argument("--diff_tier1", action="store_true",
                   help="Let diff-derived items be tier 1 (adds the left/right mirror "
                        "margin). OFF by default: a wrong side would actively teach the wrong "
                        "laterality, and the mirror margin is the harshest term in the loss.")


def _add_loss_args(p):
    """Geometric losses — the pretrain recipe, so plausibility does not drift."""
    p.add_argument("--geo_pos_weight", type=float, default=0.1)
    p.add_argument("--geo_vel_weight", type=float, default=0.1)
    p.add_argument("--geo_foot_weight", type=float, default=0.01)
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh",
                   help="Needed only when a geo weight is non-zero.")


def _add_optim_args(p):
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="0: a short fine-tune from a good initialisation does not need extra "
                        "pull toward the origin, and decay fights the pretrained solution.")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="~2.5 epochs at the default batch. Guards the first steps, where a "
                        "fresh optimiser state on pretrained weights does the most damage.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=DEFAULT_EMA)
    p.add_argument("--amp_dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)


def _add_batching_args(p):
    p.add_argument("--pad_to", default="batch", choices=["batch", "max"],
                   help="'batch' (default) pads each batch to its own longest clip; 'max' "
                        "pads to --max_frames. Bit-identical results — padding is masked out "
                        "of self-attention — at ~2.7x less compute per step and up to ~7x in "
                        "the quadratic term: train clips average 74 frames, never exceed 100.")
    p.add_argument("--bucket_by_length", action=argparse.BooleanOptionalAction, default=True,
                   help="Group similar-length clips into the same batch so --pad_to batch "
                        "actually pays off. Batch ORDER stays shuffled.")
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="Hold the featurised clips in RAM (0.43 GB train). Removes two disk "
                        "reads per sample per step.")
    p.add_argument("--precompute_text", action=argparse.BooleanOptionalAction, default=True,
                   help="Encode every instruction once up front (2.25 GB) instead of running "
                        "T5 on every batch of every epoch, and drop the encoder afterwards.")
    p.add_argument("--num_workers", type=int, default=2,
                   help="2, not more: each worker forks the parent, and Python refcounting "
                        "turns copy-on-write pages into real copies. The cached .npy reads "
                        "are cheap enough that extra workers buy little and cost GBs.")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--src_fps", type=float, default=30.0)
    p.add_argument("--edit_fps", type=float, default=20.0)


def _add_reporting_args(p):
    p.add_argument("--val_every", type=int, default=2,
                   help="2, not 5: the measured validation minimum sits at ~epoch 14, so a "
                        "coarse grid locates it only to +/-5 epochs.")
    p.add_argument("--save_every", type=int, default=10,
                   help="Periodic snapshots. `checkpoint_best` is written independently "
                        "whenever validation improves, so the best model survives regardless.")
    p.add_argument("--early_stop", type=int, default=0,
                   help="Stop after this many consecutive validations without improvement "
                        "(0 = never stop, keep the whole curve).")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--device", default=None)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_source_args(p)
    _add_objective_args(p)
    _add_grounding_args(p)
    _add_loss_args(p)
    _add_optim_args(p)
    _add_batching_args(p)
    _add_reporting_args(p)
    return p
