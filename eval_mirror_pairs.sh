#!/usr/bin/env bash
#
# eval_mirror_pairs.sh — the exact-ground-truth editing eval (docs/EVALUATION.md Option B).
#
#   ./eval_mirror_pairs.sh    # runs whatever the Config block below says
#
# Source = a HumanML3D test clip whose caption names a side, instruction = its mirror twin's
# caption (left/right swapped), target = the mirror twin itself. Both directions are edited.
# Scored as joint-position distance to the true target (mm), so no retrieval model is
# involved. See src/eval/edit_mirror_pairs.py and score_mirror_pairs.py for the details.
#
# THE NUMBER TO QUOTE: `gap%` on the `local` row (pose only, pelvis-relative) — the share of
# the source->target distance the edit removed — with its win rate and Δ±SE. The `global`
# row is dominated by root trajectory on turning/circling clips. Scale 0 must read 0.00 /
# move 0.0: it is the inversion-plumbing check, and doubles as the do-nothing baseline.
#
# COST (measured 2026-09-24): ~40 s per edit for 2 scales while sharing the GPU with a
# training run; expect several times faster on an idle card. Each extra scale adds one
# reverse loop, each extra mask mode re-pays the inversion. CAPTION_LINE=first (766 pairs,
# 1532 edits) is the cheaper full set; LIMIT samples pairs for a quick look. Re-running
# resumes; changed mask settings on the same OUT_ROOT are refused by the fingerprint guard.

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RUN="exp_smplh_llm_labels"
CHECKPOINT="runs/${RUN}/checkpoint_latest"
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"
SPLIT="test"
CAPTION_LINE="any"            # any (1257 pairs) | first (766 pairs)
MASK_MODES=(attn groups)      # groups = routed from both captions, the correct-by-construction control
SCALES=(0 6 16)               # 0 = do-nothing baseline + plumbing check
LIMIT=0                       # pairs; 0 = all
EXTRA=""                      # e.g. "--m1_columns semantic" or "--overwrite"

# ----------------------------------------------------------------------
for MODE in "${MASK_MODES[@]}"; do
    OUT_ROOT="eval_results/mirror_pairs/${RUN}_${SPLIT}_${CAPTION_LINE}_${MODE}"
    echo "=== ${MODE} -> ${OUT_ROOT}"
    python src/eval/edit_mirror_pairs.py \
        --checkpoint "${CHECKPOINT}" --data_root "${DATA_ROOT}" --split "${SPLIT}" \
        --caption_line "${CAPTION_LINE}" --mask_mode "${MODE}" --scales "${SCALES[@]}" \
        --limit "${LIMIT}" --out_root "${OUT_ROOT}" ${EXTRA}
    python src/eval/score_mirror_pairs.py --out_root "${OUT_ROOT}" --data_root "${DATA_ROOT}"
done
