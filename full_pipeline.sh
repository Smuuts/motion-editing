#!/usr/bin/env bash
#
# full_pipeline.sh — train -> generate -> evaluate, run sequentially.
# Configure in the Config block below; don't pass environment variables.

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LEARNING_RATE=1e-4
EXP_NAME="exp_test"
FEATURE_MODE="humanml3d"                     # humanml3d (263-d) | smplh (135-d)
HML3D_ROOT="data/HumanML3D/HumanML3D"        # texts/, new_joints/, 263-d Mean/Std
SMPLH_ROOT="data/HumanML3D/HumanML3D_smplh"  # 135-d smplh feats + their Mean/Std
EVALUATOR_DIR="data/t2m_evaluator"
TEXT_ENCODER="t5"
ARCH="dit"                       # dit (GroupDiT) | unet (GroupMotionUNet, MotionCLR-style)
GROUP_MODE="parts"               # token axis: parts (7 body-part groups) | joints (22 tokens)
NUM_HEADS=8
NUM_LAYERS=8                     # DiT depth; ignored when ARCH=unet
LATENT_DIM=512
UNET_LEVELS=3
UNET_BLOCKS_PER_LEVEL=2
EPOCHS=500
BATCH_SIZE=128
EMA_DECAY=0.9999
SAVE_EVERY=250
VAL_EVERY=1
SPLIT="val"

# What the head predicts: the noise (eps) or the clean motion (x0). Saved in the checkpoint,
# so nothing downstream needs a flag. !! Do NOT add --snr_gamma under x0 — Min-SNR there
# reproduces the eps weighting and cancels the point of an x0 head.
PREDICT_TYPE="x0"                # eps | x0

# MDM-style geometric losses on top of the diffusion MSE. All three at 0 disables the
# machinery entirely (no geo_fn, and smplh never loads the SMPL-H body model).
GEO_POS_WEIGHT=0.1               # joint-position error (smplh: via SMPL FK)
GEO_VEL_WEIGHT=0.1               # frame-to-frame position difference (anti-jitter)
GEO_FOOT_WEIGHT=0.01             # foot velocity on GT-contact frames (anti-skating)
# alpha_bar_t damping of the geo losses. "" = AUTO: on for eps, off for x0, which has no
# 1/sqrt(alpha_bar_t) amplification to damp. "--geo_conf_weight"/"--no-..." force it.
GEO_CONF_WEIGHT=""

# "auto" = bf16 where supported, else fp16. fp16 saturates at 65504, so a large activation
# becomes an inf -> non-finite loss -> skipped step; bf16 has fp32's exponent range and does
# not. fp32 is the safe fallback pre-Ampere.
AMP_DTYPE="auto"                 # auto | bf16 | fp16 | fp32
ATTN_SINK="--attn_sink"          # "--no-attn_sink" to disable
ATTN_ENTROPY_WEIGHT=0.0          # 0 disables; try 0.01

# ── Attention grounding: TokenCompose L_token (src/training/grounding/) ───────────
# Trains a caption's body-part words to attend to that part's group tokens. Labels come
# from the caption parser, no LLM.
ATTN_GROUND_WEIGHT=0.0           # 0 = off (whole block)
ATTN_GROUND_LAYERS="middle"      # "middle" (blocks 3-5 of 8) | "all" | "3,4,5"
ATTN_GROUND_MIRROR=1.0           # weight of the left/right mirror margin (tier-1 items)
ATTN_GROUND_EVEN=0.1             # weight of the evenness term (tier-2 items); 0 = old loss
ATTN_GROUND_MARGIN=0.1           # how far the target group must beat its mirror
ATTN_GROUND_WARMUP_EPOCHS=20     # from-scratch attention is noise before this
ATTN_GROUND_CACHE=""             # "" = <FEAT_ROOT>/ground_labels.json, built on first use
# "" = soft 1-alpha_bar_t weighting, i.e. pressure at HIGH noise — the defence against the
# loss being satisfied by a motion detector; do not flip it to alpha_bar_t.
# "--attn_ground_window 750 999" is the hard-gate ablation.
ATTN_GROUND_WINDOW=""
# "" = log corr(supervised attention, source motion energy) each epoch. KILL CRITERION:
# above ~0.5 and rising while m_S rises. "--no-attn_ground_monitor" turns it off.
ATTN_GROUND_MONITOR=""
# "" = tier-3 verb labels ON: locomotion/manipulation verbs are supervised alongside nouns,
# because a caption with a leg verb usually never names a leg. "--no-attn_ground_verbs" is
# the nouns-only A/B control — point ATTN_GROUND_CACHE elsewhere when you use it, since the
# cache file is not keyed by this setting.
ATTN_GROUND_VERBS=""

# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
# Each dataset dir is self-contained, so the feature root just follows FEATURE_MODE.
# evaluate is the exception: it also needs new_joints/, which only the HumanML3D dir has.
if [[ "${FEATURE_MODE}" == "smplh" ]]; then
  FEAT_ROOT="${SMPLH_ROOT}"
else
  FEAT_ROOT="${HML3D_ROOT}"
fi
EVAL_DATA_ROOT="${HML3D_ROOT}"
OUTPUT_DIR="runs/${EXP_NAME}"
CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest"
GENERATION_DIR="generated/${EXP_NAME}"
EVAL_DIR="eval_results"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ARCH=unet adds the depth flags; dit passes none. Saved in config.json either way, so
# generate/evaluate rebuild the right backbone with no arch flag downstream.
ARCH_ARGS=(--arch "${ARCH}")
if [[ "${ARCH}" == "unet" ]]; then
  ARCH_ARGS+=(--unet_levels "${UNET_LEVELS}" --unet_blocks_per_level "${UNET_BLOCKS_PER_LEVEL}")
fi

# Grounding flags only when the weight is non-zero. Numeric test so 0 / 0.0 / 0e0 match.
GROUND_ARGS=()
if awk "BEGIN{exit !(${ATTN_GROUND_WEIGHT}>0)}"; then
  GROUND_ARGS=(--attn_ground_weight        "${ATTN_GROUND_WEIGHT}"
               --attn_ground_layers        "${ATTN_GROUND_LAYERS}"
               --attn_ground_mirror        "${ATTN_GROUND_MIRROR}"
               --attn_ground_even          "${ATTN_GROUND_EVEN}"
               --attn_ground_margin        "${ATTN_GROUND_MARGIN}"
               --attn_ground_warmup_epochs "${ATTN_GROUND_WARMUP_EPOCHS}")
  [[ -n "${ATTN_GROUND_CACHE}" ]] && GROUND_ARGS+=(--attn_ground_cache "${ATTN_GROUND_CACHE}")
  # Unquoted on purpose: these hold whole flags and must word-split.
  GROUND_ARGS+=(${ATTN_GROUND_WINDOW} ${ATTN_GROUND_MONITOR} ${ATTN_GROUND_VERBS})
fi

if awk "BEGIN{exit !(${GEO_POS_WEIGHT}+${GEO_VEL_WEIGHT}+${GEO_FOOT_WEIGHT}==0)}"; then
  GEO_DESC="geo=OFF"
else
  GEO_DESC="geo=${GEO_POS_WEIGHT}/${GEO_VEL_WEIGHT}/${GEO_FOOT_WEIGHT}"
fi
if awk "BEGIN{exit !(${ATTN_GROUND_WEIGHT}>0)}"; then
  GROUND_DESC="ground=${ATTN_GROUND_WEIGHT}@${ATTN_GROUND_LAYERS}"
else
  GROUND_DESC="ground=OFF"
fi

# ----------------------------------------------------------------------
# 1. Train
# ----------------------------------------------------------------------
log "Training (arch=${ARCH}, predict=${PREDICT_TYPE}, lr=${LEARNING_RATE}, ${GEO_DESC}, ${GROUND_DESC}, out=${OUTPUT_DIR})"
python src/train.py \
  --data_root     "${FEAT_ROOT}" \
  --output_dir    "${OUTPUT_DIR}" \
  --predict_type  "${PREDICT_TYPE}" \
  --lr            "${LEARNING_RATE}" \
  --num_layers    "${NUM_LAYERS}" \
  --num_heads     "${NUM_HEADS}" \
  --latent_dim    "${LATENT_DIM}" \
  --epochs        "${EPOCHS}" \
  --batch_size    "${BATCH_SIZE}" \
  --ema_decay     "${EMA_DECAY}" \
  --save_every    "${SAVE_EVERY}" \
  --val_every     "${VAL_EVERY}" \
  --feature_mode  "${FEATURE_MODE}" \
  --group_mode    "${GROUP_MODE}" \
  --text_encoder  "${TEXT_ENCODER}" \
  --amp_dtype     "${AMP_DTYPE}" \
  --hml3d_pos_weight  "${GEO_POS_WEIGHT}" \
  --hml3d_vel_weight  "${GEO_VEL_WEIGHT}" \
  --hml3d_foot_weight "${GEO_FOOT_WEIGHT}" \
  "${ARCH_ARGS[@]}" \
  "${GROUND_ARGS[@]}" \
  ${ATTN_SINK} \
  ${GEO_CONF_WEIGHT} \
  --attn_entropy_weight "${ATTN_ENTROPY_WEIGHT}"

# ----------------------------------------------------------------------
# 2. Generate
# ----------------------------------------------------------------------
log "Generating samples -> ${GENERATION_DIR}"
python src/generate.py \
  --checkpoint  "${CHECKPOINT}" \
  --data_root   "${FEAT_ROOT}" \
  --split       "${SPLIT}" \
  --out_dir     "${GENERATION_DIR}"

# ----------------------------------------------------------------------
# 3. Evaluate
# ----------------------------------------------------------------------
log "Evaluating -> ${EVAL_DIR}/results_${EXP_NAME}.json"
python src/evaluate.py \
  --generated_dir    "${GENERATION_DIR}" \
  --data_root        "${EVAL_DATA_ROOT}" \
  --evaluator_dir    "${EVALUATOR_DIR}" \
  --experiment_name  "${EXP_NAME}" \
  --output_dir       "${EVAL_DIR}"

log "Pipeline complete. Results in ${EVAL_DIR}/results_${EXP_NAME}.json"
