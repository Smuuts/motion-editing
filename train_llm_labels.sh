#!/usr/bin/env bash
#
# train_llm_labels.sh — the controlled twin of runs/exp_smplh_verbs, differing in ONE
# thing: the caption->body-part labels the TokenCompose grounding loss is supervised
# from. `exp_smplh_verbs` used the regex parser's ground_labels.json; this run uses
# ground_labels_llm.json, built by src/build_ground_labels_llm.py (MaskOptions.md §13.3).
#
#   ./build_labels.sh is not a thing — build the labels first, then:
#   ./train_llm_labels.sh
#
# Configure in the Config block; don't pass environment variables. Same convention as
# eval_motionfix.sh and finetune_motionfix.sh.
#
# WHAT THIS RUN IS TESTING, stated so the result can be read against it afterwards.
# NOT coverage: the LLM adds ~4.4 pp of captions, and M1 alignment on exp_smplh_verbs is
# already saturated at the metric's ceiling (0.476 +/- 0.000 on all 9 clips x 4
# instructions), so there is no headroom there for a better labeller to take. The target
# is LATERALITY AT THE DEFAULT READ-OUT: laterality contrast is +0.194 under
# `--m1_columns semantic` against +0.871 under `span`, and FINDINGS.md 2026-08-15 §3
# attributes the whole gap to one structural cause — "the verb column is supervised
# BILATERAL ... and cannot be trained away without lateralising verb labels", which is
# exactly what ARCHITECTURE.md rule 2 ("verbs are never lateralised") forbids the regex
# parser from doing. The LLM label set lateralises a verb when, and only when, it can
# quote a limb phrase that the regex binder independently reads as a side (filter 2 in
# data/body_part_labels/llm_cache.py), so "waves with his right hand" gets
# waves -> right_arm while "takes a step to the right" does not.
#
# THE ONE CONFOUND THAT HAD TO BE REMOVED BY HAND. `exp_smplh_verbs` predates
# --attn_ground_even (added 2026-08-15), so it trained with NO evenness term; the flag now
# defaults to 0.1. Left at the default this run would change the labels AND the loss at
# once, which is the exact defect the project has already had to caveat twice ("this run
# changed lambda AND the label set, so it is not a lambda curve"). GROUND_EVEN=0.0 below
# reproduces exp_smplh_verbs' objective exactly. Change it only if you also intend to
# stop comparing against that checkpoint.
#
# PRE-REGISTERED GATES (MaskOptions.md §13.3). Label-level gates are already green from
# the build step. Training-level: src_corr < 0.5 and not rising while m_S rises. Mask:
# laterality contrast under `semantic` +0.194 -> >= +0.6 is the point of the run;
# alignment must hold at 0.476 and laterality invariance must fall. Generation:
# FID <= 0.156 and R@1 >= 0.514, the exp_smplh_verbs numbers.
#
# COMPARE m_S_tier1, NOT m_S. The two label sets have different target-size mixes, so the
# pooled m_S is not comparable across them (the same trap the nouns-vs-verbs comparison
# hit: 0.987 vs 0.967 at chance 0.203 vs 0.262). Tier-1 items are single-group by
# construction, so m_S_tier1 is the one number that compares.
#
# COST. ~20 h for 500 epochs (exp_smplh_verbs ran 2026-08-13 19:19 -> 08-14 15:03).

set -euo pipefail
cd "$(dirname "$0")"

# ── Config ────────────────────────────────────────────────────────────────────────────
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"
OUTPUT_DIR="runs/exp_smplh_llm_labels"
GROUND_CACHE="${DATA_ROOT}/ground_labels_llm.json"

# Everything below is exp_smplh_verbs/config.json, unchanged.
PREDICT_TYPE="x0"
FEATURE_MODE="smplh"
GROUP_MODE="parts"
TEXT_ENCODER="t5"
NUM_LAYERS=8
NUM_HEADS=8
LATENT_DIM=512
EPOCHS=500
BATCH_SIZE=64
LEARNING_RATE=1e-4
EMA_DECAY=0.9999
SAVE_EVERY=100
VAL_EVERY=1

GROUND_WEIGHT=5e-3
GROUND_LAYERS="middle"
GROUND_MIRROR=1.0
GROUND_MARGIN=0.1
GROUND_WARMUP=20
GROUND_EVEN=0.0          # see "THE ONE CONFOUND" above — NOT the flag's 0.1 default
# ──────────────────────────────────────────────────────────────────────────────────────

if [[ ! -f "${GROUND_CACHE}" ]]; then
  echo "ERROR: ${GROUND_CACHE} does not exist." >&2
  echo "Build it first:  python src/build_ground_labels_llm.py --data_root ${DATA_ROOT}" >&2
  exit 1
fi

# --attn_ground_cache is keyed by FILE, and a run that silently falls back to the regex
# labels would look identical in every log line that matters. Say which file, loudly.
echo "Grounding labels: ${GROUND_CACHE}"
python - "${GROUND_CACHE}" <<'PY'
import json, sys
cache = json.load(open(sys.argv[1]))
items = [it for v in cache.values() for it in v]
print(f"  {len(cache):,} captions, {len(items):,} items, "
      f"{sum(it['lat'] for it in items):,} tier-1 "
      f"({sum(1 for v in cache.values() if v) / max(len(cache), 1):.1%} coverage)")
PY

python src/train.py \
  --data_root      "${DATA_ROOT}" \
  --output_dir     "${OUTPUT_DIR}" \
  --predict_type   "${PREDICT_TYPE}" \
  --feature_mode   "${FEATURE_MODE}" \
  --group_mode     "${GROUP_MODE}" \
  --text_encoder   "${TEXT_ENCODER}" \
  --num_layers     "${NUM_LAYERS}" \
  --num_heads      "${NUM_HEADS}" \
  --latent_dim     "${LATENT_DIM}" \
  --epochs         "${EPOCHS}" \
  --batch_size     "${BATCH_SIZE}" \
  --lr             "${LEARNING_RATE}" \
  --ema_decay      "${EMA_DECAY}" \
  --save_every     "${SAVE_EVERY}" \
  --val_every      "${VAL_EVERY}" \
  --attn_sink \
  --ctx_pad_mask \
  --attn_ground_weight         "${GROUND_WEIGHT}" \
  --attn_ground_layers         "${GROUND_LAYERS}" \
  --attn_ground_mirror         "${GROUND_MIRROR}" \
  --attn_ground_margin         "${GROUND_MARGIN}" \
  --attn_ground_even           "${GROUND_EVEN}" \
  --attn_ground_warmup_epochs  "${GROUND_WARMUP}" \
  --attn_ground_verbs \
  --attn_ground_monitor \
  --attn_ground_cache          "${GROUND_CACHE}"
