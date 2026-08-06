#!/usr/bin/env bash
#
# pipeline.sh — train -> generate -> evaluate, run sequentially.

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LEARNING_RATE=1e-4
EXP_NAME="exp_test"
FEATURE_MODE="humanml3d"               # humanml3d (263-d) or smplh (135-d)
HML3D_ROOT="data/HumanML3D/HumanML3D"  # processed HumanML3D (texts/, new_joints/, 263 Mean/Std)
SMPLH_ROOT="data/HumanML3D/HumanML3D_smplh"  # flat 135-d smplh feats + their Mean/Std
EVALUATOR_DIR="data/t2m_evaluator"
TEXT_ENCODER="t5"
ARCH="dit"                       # dit (GroupDiT) or unet (GroupMotionUNet, MotionCLR-style)
GROUP_MODE="parts"               # token axis: parts (7 body-part groups) or joints (22 per-joint tokens)
NUM_HEADS=8
NUM_LAYERS=8                     # DiT depth; ignored when ARCH=unet (see below)
LATENT_DIM=512
UNET_LEVELS=3
UNET_BLOCKS_PER_LEVEL=2
EPOCHS=500
# Output-head parameterisation: eps (default) or x0 (Option 5, see
# docs/AttentionGrounding_Options.md §5 and ARCHITECTURE.md "Prediction target").
# x0 regresses the clean motion directly instead of the noise. Saved into the
# checkpoint config, so generate/evaluate/edit convert back automatically — nothing
# downstream needs a flag.
#   !! Do NOT add --snr_gamma to the train call below when using x0. train.py forces
#   snr_gamma=0 under x0 *only if the flag was not passed explicitly*, and Min-SNR
#   under an x0 head reproduces the eps baseline's weighting almost exactly (3% of
#   training weight on t>=600 vs 40% unweighted), which cancels the whole point.
PREDICT_TYPE="eps"               # eps | x0
# MDM-style geometric losses, added to the diffusion MSE with these weights.
#   pos  — joint-position error. humanml3d reads the positions straight out of channels
#          [4:67]; smplh has none in its features and must run SMPL FK to get them, which
#          composes rotation error down the kinematic chain (the expensive, less stable one).
#   vel  — frame-to-frame position difference (penalises jitter).
#   foot — predicted foot velocity on frames the GT calls "in contact" (anti-skating).
# ALL THREE AT 0 DISABLES THEM COMPLETELY: build_geo_fn returns geo_fn=None, the epoch
# loop skips the whole block, and under smplh the SMPL-H body model is never even loaded.
# Setting only some to 0 keeps the machinery and drops those terms.
#   Worth knowing before changing these: every run in runs/ so far used exactly
#   (0.1, 0.1, 0.01) — they have never been ablated, so there is no measurement in this
#   project of what they buy. Under smplh they are the ONLY positional supervision (the
#   135-d features are rotations); under humanml3d L_pos largely re-weights channels the
#   diffusion MSE already covers. See docs/PROGRESS.md for the run they are implicated in.
GEO_POS_WEIGHT=0.1
GEO_VEL_WEIGHT=0.1
GEO_FOOT_WEIGHT=0.01
# Geometric-loss confidence weight (alpha_bar_t damping). "" = AUTO: on for eps, off
# for x0 (an x0 head outputs x0 directly, so there is no 1/sqrt(alpha_bar_t) error
# amplification to damp). "--geo_conf_weight" forces on — the escape hatch if an x0
# run destabilises near t=T; "--no-geo_conf_weight" forces off. No effect when the
# three weights above are all 0.
GEO_CONF_WEIGHT=""
# NOTE: attn_sink forces the explicit (non-fused) attention path during training —
# costs GPU memory vs SDPA (bs 20 OOMs on a 12 GB card where SDPA fit; bs 16 OK).
# Re-check the batch size on the training machine before a long run.
# Autocast dtype. "auto" = bf16 where the GPU supports it, else fp16. fp16 saturates at
# 65504, so an activation that merely grows large becomes an inf -> non-finite loss ->
# skipped step; bf16 keeps fp32's exponent range and cannot overflow that way. Two runs
# have been lost to this (docs/FINDINGS.md "fp16 activation overflow"). fp32 is the
# slow, definitely-safe fallback for a GPU without bf16 (pre-Ampere, e.g. V100).
AMP_DTYPE="auto"                 # auto | bf16 | fp16 | fp32
BATCH_SIZE=128
ATTN_SINK="--attn_sink"          # "--no-attn_sink" to disable (see ARCHITECTURE.md)
ATTN_ENTROPY_WEIGHT=0.0          # 0 disables; try 0.01 (experiment knob, keep runs attributable)
EMA_DECAY=0.9999
SAVE_EVERY=250
VAL_EVERY=1
SPLIT="val"

# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
# Feature root differs per mode; each dataset dir is self-contained (new_joint_vecs/, texts/,
# Mean/Std, splits). evaluate additionally needs new_joints/ for the smplh tgt-offset reference,
# which only lives in the processed HumanML3D dir. For humanml3d all roots coincide.
if [[ "${FEATURE_MODE}" == "smplh" ]]; then
  FEAT_ROOT="${SMPLH_ROOT}"    # train/generate read 135-d feats + texts + Mean/Std here
else
  FEAT_ROOT="${HML3D_ROOT}"
fi
EVAL_DATA_ROOT="${HML3D_ROOT}" # evaluate needs texts/ + new_joints/ (smplh tgt-offset ref)

OUTPUT_DIR="runs/${EXP_NAME}"
CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest"
GENERATION_DIR="generated/${EXP_NAME}"
EVAL_DIR="eval_results"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ARCH=unet adds the U-Net depth flags; ARCH=dit passes none of them. Either way `arch`
# (and the unet keys) are saved into the checkpoint's config.json, so generate/evaluate
# rebuild the right backbone automatically — no arch flag is needed downstream.
ARCH_ARGS=(--arch "${ARCH}")
if [[ "${ARCH}" == "unet" ]]; then
  ARCH_ARGS+=(--unet_levels "${UNET_LEVELS}" --unet_blocks_per_level "${UNET_BLOCKS_PER_LEVEL}")
fi

# ----------------------------------------------------------------------
# 1. Train
# ----------------------------------------------------------------------
# Numeric test, not a string one, so 0 / 0.0 / 0e0 all report the same thing.
if awk "BEGIN{exit !(${GEO_POS_WEIGHT}+${GEO_VEL_WEIGHT}+${GEO_FOOT_WEIGHT}==0)}"; then
  GEO_DESC="geo=OFF"
else
  GEO_DESC="geo=${GEO_POS_WEIGHT}/${GEO_VEL_WEIGHT}/${GEO_FOOT_WEIGHT}"
fi
log "Training (arch=${ARCH}, predict=${PREDICT_TYPE}, lr=${LEARNING_RATE}, ${GEO_DESC}, out=${OUTPUT_DIR})"
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
