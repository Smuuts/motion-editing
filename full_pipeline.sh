#!/usr/bin/env bash
#
# pipeline.sh — train -> generate -> evaluate, run sequentially.

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LEARNING_RATE=1e-4
EXP_NAME="exp_test"
DATA_ROOT="data/HumanML3D"
EVALUATOR_DIR="data/t2m_evaluator"
FEATURE_MODE="humanml3d"
TEXT_ENCODER="t5"
NUM_HEADS=8
NUM_LAYERS=8
LATENT_DIM=512
EPOCHS=3000
BATCH_SIZE=128
EMA_DECAY=0.9999
SAVE_EVERY=250
VAL_EVERY=1
SPLIT="val"

# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
OUTPUT_DIR="runs/${EXP_NAME}"
CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest"
GENERATION_DIR="generated/${EXP_NAME}"
EVAL_DIR="eval_results/${EXP_NAME}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------
# 1. Train
# ----------------------------------------------------------------------
log "Training (lr=${LEARNING_RATE}, out=${OUTPUT_DIR})"
python src/train.py \
  --data_root     "${DATA_ROOT}" \
  --output_dir    "${OUTPUT_DIR}" \
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
  --text_encoder  "${TEXT_ENCODER}"

# ----------------------------------------------------------------------
# 2. Generate
# ----------------------------------------------------------------------
log "Generating samples -> ${GENERATION_DIR}"
python src/generate.py \
  --checkpoint  "${CHECKPOINT}" \
  --data_root   "${DATA_ROOT}" \
  --split       "${SPLIT}" \
  --out_dir     "${GENERATION_DIR}" \

# ----------------------------------------------------------------------
# 3. Evaluate
# ----------------------------------------------------------------------
log "Evaluating -> ${EVAL_DIR}"
python src/evaluate.py \
  --generated_dir  "${GENERATION_DIR}" \
  --data_root      "${DATA_ROOT}" \
  --evaluator_dir  "${EVALUATOR_DIR}" \
  --output_dir     "${EVAL_DIR}"

log "Pipeline complete. Results in ${EVAL_DIR}"
