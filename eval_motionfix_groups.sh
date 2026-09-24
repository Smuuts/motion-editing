#!/usr/bin/env bash
#
# eval_motionfix_groups.sh — regex vs. LLM group-router comparison (Option 13, full tier),
# full 1013-clip MotionFix test set, TMR-scored.
#
# Companion to eval_motionfix.sh, not a replacement: that script sweeps {m2_only, attn,
# m1_only} — the M1 ∩ M2 hierarchy, never mask_mode=groups (deliberately excluded there,
# see its own comment). This script exists to answer a different question: for the SAME
# mask_mode=groups, does routing the instruction with an LLM (qwen2.5:7b-instruct via a
# local Ollama server) change the mask, and the edit, versus the regex+verb router
# (route_groups(), 82.6% coverage on this test set)?
#
# Scales are matched-MAGNITUDE, not matched-nominal-scale, because a 2026-09-15 scan found
# the LLM's masks move the body noticeably less per unit guidance scale than the regex's
# (consistent with the coverage measurement's finding that the LLM's mask is a narrower
# subset of the regex's on ~47% of captions where both resolve — see docs/FINDINGS.md and
# docs/MaskOptions.md §13.1). Comparing them at the same nominal scale would conflate mask
# size with guidance strength, exactly the reason the Step-3 run scored {m2_only, attn,
# m1_only} at matched magnitude rather than matched scale.
#
#   scale  regex rot_deg   llm rot_deg
#   0.5    1.20            0.80
#   1      2.18            1.32
#   2      3.91            2.40
#   3      5.32            3.31
#   5      6.83            4.73
#   8      9.28            6.72
#   12     10.57           8.05
#
# regex "0 0.5 1 5" -> 0°, 1.20°, 2.18°, 6.83°
# llm   "0 1   2 8" -> 0°, 1.32°, 2.40°, 6.72°
# Three matched tiers (~1.2-1.3°, ~2.2-2.4°, ~6.7-6.8°), both grids already covered by the
# 2026-09-15 scan (eval_results/motionfix/exp_smplh_verbs_groups_{regex,llm}_scan.json) —
# no extra data needed to justify them. Re-scan (see edit_motionfix_testset.py + scale_scan.py)
# before reusing these on a different checkpoint; they are specific to exp_smplh_verbs.
#
# Scale 0 is kept for BOTH routers despite reconstructing the source almost exactly: it is
# this project's standing plumbing check (scale 0 must score ~100 R@1_s2t or the run's other
# numbers are meaningless) and the "do-nothing" baseline every reported table is read
# against — not a formality, don't drop it to save time.
#
# The two routers skip DIFFERENT clips (regex found no body part in 2/16 scan clips, llm in
# 1/16 — router-specific, not the same set), so scoring uses --common_subset: the keyids
# both routers actually generated, otherwise the galleries differ and R@k isn't comparable
# between rows.
#
#   ./eval_motionfix_groups.sh

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RUN="exp_smplh_verbs"
CHECKPOINT="runs/${RUN}/checkpoint_latest"
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"
MFIX_PY="data/motionfix/mfix-env/bin/python"

declare -A SCALES_BY_ROUTER=(
  [regex]="0 0.5 1 5"
  [llm]="0 1 2 8"
)
ROUTERS="regex llm"

LIMIT=0            # 0 = the full 1013-clip test set — the only publication-comparable gallery
LIMIT_MODE="random"
LIMIT_SEED=0
SEED=42
COMMON_SUBSET=1     # required: the two routers skip different clips (see header)
EXTRA=""            # extra flags for the editor, e.g. "--overwrite"

TAG="${RUN}_groups"
METRICS="eval_results/motionfix/${TAG}_regex_vs_llm_tmr.json"
SUMMARY_DIR="eval_results/motionfix/${TAG}_regex_vs_llm"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------
# Preflight: fail here, not hours in
# ----------------------------------------------------------------------
[[ -f "${CHECKPOINT}/config.json" ]] || { echo "no checkpoint at ${CHECKPOINT}"; exit 1; }
[[ -f "${DATA_ROOT}/Mean.npy" ]]     || { echo "no Mean.npy in ${DATA_ROOT}"; exit 1; }
[[ -x "${MFIX_PY}" ]]                || { echo "no MotionFix venv at ${MFIX_PY}"; exit 1; }
curl -sS -m 5 http://localhost:11434/api/tags -o /dev/null \
  || { echo "no Ollama server at localhost:11434 (needed for --group_router llm)"; exit 1; }
mkdir -p "$(dirname "${METRICS}")"

echo "  checkpoint    ${CHECKPOINT}"
for ROUTER in ${ROUTERS}; do
  printf '  scales        %-6s %s\n' "${ROUTER}" "${SCALES_BY_ROUTER[$ROUTER]}"
done
echo "  clips         all 1013 (full test set)"
echo "  metrics       ${METRICS}"

# ----------------------------------------------------------------------
# 1. Edit: generate the edited motions, one output root per router
# ----------------------------------------------------------------------
# edit_motionfix_testset.py always names its per-scale output dir "{mask_mode}_s{scale}" —
# here that's "groups_s{scale}" for EVERY router, since mask_mode is "groups" regardless of
# --group_router. run_motionfix_metrics.py keys its results dict by
# os.path.basename(smpl_dir), so passing both routers' real dirs straight through silently
# collides on any scale value they share (0 is in both routers' lists) — the later
# --smpl_dir arg overwrites the earlier one's entry and that router's row vanishes with no
# error. Route every --smpl_dir through a staging dir of symlinks named
# "groups_{router}_s{scale}" instead, so the basenames never collide. realpath WOULD resolve
# through the symlink back to the shared basename and reintroduce the bug — use
# `realpath --no-symlinks` (or an equivalent unresolved absolute path) here, always.
STAGE_DIR="data/motionfix/motionfix_smpl/${TAG}_stage"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

SMPL_DIRS=()
for ROUTER in ${ROUTERS}; do
  OUT_ROOT="data/motionfix/motionfix_smpl/${TAG}_${ROUTER}"
  SCALES="${SCALES_BY_ROUTER[$ROUTER]}"
  mkdir -p "${OUT_ROOT}"
  log "Editing: group_router=${ROUTER}  scales=${SCALES}"
  python src/eval/edit_motionfix_testset.py \
    --checkpoint       "${CHECKPOINT}" \
    --smplh_data_root  "${DATA_ROOT}" \
    --out_root         "${OUT_ROOT}" \
    --mask_mode        groups \
    --group_router      "${ROUTER}" \
    --scales            ${SCALES} \
    --seed              "${SEED}" \
    --limit              "${LIMIT}" \
    --limit_mode         "${LIMIT_MODE}" \
    --limit_seed         "${LIMIT_SEED}" \
    ${EXTRA}
  for S in ${SCALES}; do
    LINK="${STAGE_DIR}/groups_${ROUTER}_s${S}"
    ln -sf "$(realpath "${OUT_ROOT}/groups_s${S}")" "${LINK}"
    SMPL_DIRS+=(--smpl_dir "$(realpath --no-symlinks "${LINK}")")
  done
done

# ----------------------------------------------------------------------
# 2. Score with MotionFix's own TMR evaluator (their venv, their code)
# ----------------------------------------------------------------------
FIRST_DIR="${SMPL_DIRS[1]}"
N_GEN=$(find "${FIRST_DIR}" -name '*.npy' 2>/dev/null | wc -l)
if [[ "${N_GEN}" -lt 32 ]]; then
  echo
  echo "Generated ${N_GEN} clips in ${FIRST_DIR}, but MotionFix's TMR evaluator needs >= 32."
  echo "Stage 1 output is on disk and is fine; skipping scoring."
  exit 0
fi
log "Scoring ${#SMPL_DIRS[@]} dirs through TMR"
echo "    ${N_GEN} generations -> $(( N_GEN / 32 )) batches of 32; $(( N_GEN % 32 )) dropped"
SUBSET_FLAG=""
[[ "${COMMON_SUBSET}" == "1" ]] && SUBSET_FLAG="--common_subset"
"${MFIX_PY}" src/eval/run_motionfix_metrics.py "${SMPL_DIRS[@]}" ${SUBSET_FLAG} --out "${METRICS}"

# ----------------------------------------------------------------------
# 3. Render the comparable table
# ----------------------------------------------------------------------
log "Summary -> ${SUMMARY_DIR}/summary.md"
python src/eval/aggregate_summary.py --tmr "${METRICS}" --out_dir "${SUMMARY_DIR}"

log "Complete. ${METRICS} and ${SUMMARY_DIR}/summary.md"
echo "Read it as: *_s2t = retrieval against the SOURCE (preservation), the other against"
echo "the TARGET (instruction-following). Both routers' scale-0 row must score ~100 R@1_s2t —"
echo "if either doesn't, that router's plumbing is broken and its other rows mean nothing."
echo "Quote the _b (batches-of-32) columns: the whole-gallery ones read high on any run."
echo "Remember the two routers were scaled to MATCHED MAGNITUDE, not matched nominal scale —"
echo "compare rows by their rot_deg tier (see the header comment), not by their scale label."
