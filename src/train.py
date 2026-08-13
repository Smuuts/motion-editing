"""
Training entry point.

    python src/train.py --data_root data/HumanML3D/HumanML3D --output_dir runs/exp_1

Argument handling only: the config is resolved in training/config.py and the run
itself lives in training/trainer.py. Every flag here is saved into the checkpoint's
config.json, so generate/evaluate/edit rebuild the same model without repeating flags.
"""

import os
# Must be set before any CUDA context is created (i.e. before the first .cuda() call,
# not necessarily before `import torch`). Mitigates allocator fragmentation; only
# applies if the environment doesn't already set it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

from training.config import print_config, resolve_config, save_config
from training.trainer import Trainer


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True, default="./data/HumanML3D")
    p.add_argument("--output_dir",   default="./runs/exp_1")
    p.add_argument("--config",       default=None,
                   help="JSON config file. CLI args override it.")

    # ── model ──────────────────────────────────────────────────────────────────
    p.add_argument("--arch",         type=str, default="dit", choices=["dit", "unet"],
                   help="Backbone: 'dit' (GroupDiT) or 'unet' (GroupMotionUNet, "
                        "MotionCLR-style temporal U-Net over the same group tokens). "
                        "unet depth comes from --unet_levels/--unet_blocks_per_level; "
                        "--num_layers is ignored.")
    p.add_argument("--group_mode",   type=str, default="parts",
                   choices=["parts", "joints"],
                   help="Token axis: 'parts' (7 body-part tokens per frame) or "
                        "'joints' (22 per-joint tokens — finer query resolution). "
                        "No effect on the legacy flat MotionDiT.")
    p.add_argument("--unet_levels",  type=int, default=3,
                   help="arch=unet: down/up resolution levels (frames are padded to a "
                        "multiple of 2^levels).")
    p.add_argument("--unet_blocks_per_level", type=int, default=2,
                   help="arch=unet: CLR blocks per level (MotionCLR uses 2).")
    p.add_argument("--latent_dim",   type=int,   default=512)
    p.add_argument("--num_layers",   type=int,   default=8)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--text_encoder", type=str, default="t5", choices=["clip", "t5"])
    p.add_argument("--clip_version", type=str, default="ViT-B/32",
                   help="CLIP variant (--text_encoder clip).")
    p.add_argument("--t5_version",   type=str, default="t5-base",
                   help="T5 model name, e.g. t5-base/t5-large (--text_encoder t5).")
    p.add_argument("--t5_max_length", type=int, default=128,
                   help="Fixed T5 token sequence length (--text_encoder t5).")
    p.add_argument("--attn_sink", action=argparse.BooleanOptionalAction, default=True,
                   help="Learnable per-head zero-value sink logit in cross-attention "
                        "(GPT-OSS-style; default on). Gives queries that don't need "
                        "text a dump site instead of hijacking EOS, so stored attention "
                        "maps are sink-free by construction. Config-gated, so old "
                        "checkpoints load with it off. Costs no extra memory: the sink "
                        "is expressed as an SDPA-compatible extra key/value column, so "
                        "training stays on the fused kernel.")
    p.add_argument("--attn_entropy_weight", type=float, default=0.0,
                   help="Cross-attention entropy regulariser: loss -= w * H(attn), "
                        "encouraging queries to spread over words instead of collapsing "
                        "onto one key. 0 disables (default); measured NEGATIVE at 0.01 "
                        "(docs/FINDINGS.md). Also forces the explicit attention path.")
    # ── attention grounding (TokenCompose L_token, Option 1) ───────────────────
    # All default off, so a run that does not pass --attn_ground_weight is byte-identical
    # to one from before these flags existed. See src/training/grounding.py and
    # docs/TokenCompose_Handoff.md.
    p.add_argument("--attn_ground_weight", type=float, default=0.0,
                   help="TokenCompose cross-attention grounding loss: the text columns "
                        "spelling out a body part must put their attention on that "
                        "part's group tokens, loss += w * (1 - mass_on_target)^2. 0 "
                        "disables everything below (default). TokenCompose's released "
                        "config uses 1e-3 summed over ~10 attention maps per step; this "
                        "supervises ONE block per step, so 5e-3 is the starting point.")
    p.add_argument("--attn_ground_layers", type=str, default="middle",
                   help="Blocks eligible for grounding, one sampled per step: 'middle' "
                        "(default, the middle 3/8 = blocks 3-5 of 8 — TokenCompose's own "
                        "mid+decoder choice, and where this project's self-attention "
                        "probe found body-part structure peaking), 'all', or an explicit "
                        "list like '3,4,5'.")
    p.add_argument("--attn_ground_mirror", type=float, default=1.0,
                   help="Weight of the mirror-margin term, relu(mass_mirror - mass_target"
                        " + margin), on lateralised (tier-1) items only. This is the "
                        "term aimed at laterality — the axis no training-free fix has "
                        "ever moved (M1 left/right invariance r=0.985). 0 disables it.")
    p.add_argument("--attn_ground_margin", type=float, default=0.1,
                   help="How far the target group must beat its left/right mirror.")
    p.add_argument("--attn_ground_warmup_epochs", type=int, default=20,
                   help="Epochs before the grounding loss switches on. From scratch, "
                        "attention is random noise at epoch 0 (TokenCompose finetuned a "
                        "converged model); the warmup also means no explicit-softmax "
                        "memory cost until then.")
    p.add_argument("--attn_ground_window", type=int, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Ablation: hard-gate grounding to timesteps in [LO, HI] instead "
                        "of the default soft 1-alpha_bar_t weighting. Costs ~4x the "
                        "grounded-sample rate ([750,999] fires on ~25%% of samples).")
    p.add_argument("--attn_ground_cache", type=str, default=None,
                   help="Caption->body-part label cache built by "
                        "src/probe_ground_labels.py. Default "
                        "<data_root>/ground_labels.json; built automatically if absent.")
    p.add_argument("--attn_ground_verbs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include TIER-3 verb labels (default on): map locomotion and "
                        "manipulation verbs to their body-part groups, not just nouns. "
                        "86.5%% of HumanML3D captions containing a leg verb never name a "
                        "leg, so nouns-only supervision misses where leg semantics live; "
                        "turning this on takes caption coverage 48.0%% -> 92.4%% and leg "
                        "items 10,942 -> 58,218. --no-attn_ground_verbs reproduces the "
                        "nouns-only label set (the A/B control). NOTE the label cache is "
                        "keyed by file, so change --attn_ground_cache too, or delete the "
                        "existing one, when flipping this.")
    p.add_argument("--attn_ground_monitor", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Log corr(supervised attention, source motion energy) each epoch "
                        "(default on). Captions describe their clips, so the loss is "
                        "partly satisfiable by a source-motion detector; this is how you "
                        "see that happening. Kill the run if it passes ~0.5 and rises "
                        "while m_S rises.")
    p.add_argument("--ctx_pad_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Mask padding keys in cross-attention (default on). Without it "
                        "zero-embedding pad columns absorb ~93%% of attention mass and "
                        "attenuate text conditioning ~14x ('padding sink', "
                        "docs/FINDINGS.md). No-op with --text_encoder clip. Old "
                        "checkpoints load with it off automatically.")

    # ── diffusion ──────────────────────────────────────────────────────────────
    p.add_argument("--predict_type", type=str, default="eps", choices=["eps", "x0"],
                   help="What the output head predicts. 'eps' (default) is "
                        "noise-prediction; 'x0' regresses the clean motion directly "
                        "(Option 5, docs/AttentionGrounding_Options.md) — the "
                        "eps-objective discounts high-noise steps, which is exactly "
                        "where the caption is the only information present. Flips "
                        "Min-SNR off and drops the geometric confidence weight "
                        "automatically; saved in the config so inference converts back.")
    p.add_argument("--geo_conf_weight", action=argparse.BooleanOptionalAction, default=None,
                   help="Weight the geometric losses by the x0-confidence factor "
                        "alpha_bar_t. Default AUTO: on for eps, off for x0 (an x0 head "
                        "outputs x0 directly, so there is no 1/sqrt(alpha_bar_t) error "
                        "amplification to damp). Pass to force on.")
    p.add_argument("--timesteps",    type=int,   default=1000)
    p.add_argument("--cfg_dropout",  type=float, default=0.1,
                   help="Fraction of the batch trained unconditionally (CFG).")
    p.add_argument("--snr_gamma",    type=float, default=5.0,
                   help="Min-SNR weighting gamma (Hang et al. 2023). 0 = disabled.")

    # ── data & geometric losses ────────────────────────────────────────────────
    p.add_argument("--feature_mode", type=str, default="humanml3d",
                   choices=["humanml3d", "smplh"],
                   help="Feature representation, both body-part-grouped: 263-d "
                        "HumanML3D or 135-d SMPL-H (src/data/amass_to_smplh.py). Both "
                        "read data_root/new_joint_vecs/ and data_root/texts/.")
    p.add_argument("--smplh_model_path", type=str,
                   default="data/motionfix/data/body_models/smplh",
                   help="SMPLHLayer dir for smplh geometric losses (SMPLH_NEUTRAL.npz).")
    p.add_argument("--hml3d_pos_weight",  type=float, default=0.1,
                   help="MDM L_pos joint-position loss (smplh: on SMPL-FK joints). 0 = off.")
    p.add_argument("--hml3d_vel_weight",  type=float, default=0.1,
                   help="MDM L_vel velocity-consistency loss. 0 = off.")
    p.add_argument("--hml3d_foot_weight", type=float, default=0.01,
                   help="MDM L_foot contact loss. 0 = off.")

    # ── training ───────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_frames",   type=int,   default=196)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--ema_decay",    type=float, default=0.9999)
    p.add_argument("--save_every",   type=int,   default=100)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--val_every",    type=int,   default=1,
                   help="Validate every N epochs (0 = disabled).")
    p.add_argument("--warmup_epochs", type=int,  default=5,
                   help="Epochs spent linearly warming the LR from 1%% to target.")
    p.add_argument("--no_lr_decay",  action="store_true",
                   help="Keep the learning rate constant (no cosine decay).")
    p.add_argument("--amp_dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                   help="Autocast dtype for the forward pass. 'auto' (default) = bf16 "
                        "where the GPU supports it, else fp16. bf16 has fp32's exponent "
                        "range, so a large activation stays finite instead of becoming an "
                        "inf that makes the loss non-finite and the step a no-op. Resumed "
                        "runs keep what they trained with.")
    p.add_argument("--resume",       type=str, default=None,
                   help="Checkpoint dir to resume from, or 'latest'. Missing args are "
                        "filled from --output_dir/config.json (explicit CLI args win).")
    return p


def main():
    parser = build_parser()
    config = resolve_config(parser, parser.parse_args())
    print_config(config)
    save_config(config, config["output_dir"])
    Trainer(config).run()


if __name__ == "__main__":
    main()
