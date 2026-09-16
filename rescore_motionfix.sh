#!/usr/bin/env bash
#
# rescore_motionfix.sh — re-score generations that ALREADY EXIST on disk with the metrics
# R@1 cannot express. No editing, no inversion, no generation: it loads the .npy files you
# already have and runs TMR over them.
#
#   ./rescore_motionfix.sh                    # list what it can see, then stop
#   ./rescore_motionfix.sh exp_smplh_verbs_signed
#   ./rescore_motionfix.sh --all              # everything under data/motionfix/motionfix_smpl
#
# TWO LAYOUTS ARE HANDLED. Current runs are tagged — <TAG>/<mask_mode>_s<scale>/*.npy — and
# each tag is scored as its own sweep. Older runs predate the tag convention and sit flat as
# <mask_mode>_s<scale>/*.npy directly under the generation root; those are collected and
# scored TOGETHER as one sweep (named by --group, default "legacy"), because they are one,
# and scoring them separately would throw away the cross-scale comparison and make
# --common_subset a no-op.
#
# WHAT IT ADDS over the numbers already in eval_results/ (EVALUATION.md §10, option 1):
#
#   AvgR / MedR   MotionFix's own evaluator computes these and then throws them away before
#                 returning (a hardcoded `names_to_keep`). Recovered here by wrapping their
#                 `line2dict` — no vendored file is touched. Ranks, so LOWER is better, and
#                 unlike R@1 they move when a partial edit lifts the target from rank 8 to 3.
#
#   PIR           Per-clip: did editing land CLOSER to the target than the unedited source
#                 already was? Reported as the share of clips that improved, with an exact
#                 two-sided sign test, plus a per-clip CSV. Chance is exactly 50 % — each clip
#                 is its own control — so this needs no baseline row to be interpretable, and
#                 a paired test over ~1000 clips resolves effects an order of magnitude below
#                 what the current protocol can see.
#
# READ THE RESULT AGAINST 50, NOT AGAINST THE OTHER ROWS. Three outcomes, all informative:
#   PIR > 50, p small   the edit carries directional signal the R@1 table could not see
#   PIR ~ 50, p large   the edit moves the motion at random w.r.t. the target
#   PIR < 50, p small   the edit moves AWAY from the target — which is what this project's
#                       own prior finding ("a magnitude knob, not a semantic one") predicts
#
# COST: one TMR pass per config dir (plus a second one for the per-clip baseline). Minutes,
# not hours — the expensive part of an eval is Stage 3, and that is already done.
#
# ⚠ One scoring call per SWEEP, never one call spanning several tags. run_motionfix_metrics.py
# keys its results by the directory BASENAME, so passing two tags' `attn_s2` dirs to a single
# call would silently overwrite one with the other (this bug was found and fixed once already —
# see MaskOptions.md §13.2). Within a sweep the basenames are unique, so pooling the flat
# config dirs is safe; across tags it is not, which is why each tag gets its own output file.

set -euo pipefail
cd "$(dirname "$0")"

MFIX_PY="data/motionfix/mfix-env/bin/python"
GEN_ROOT="data/motionfix/motionfix_smpl"
OUT_ROOT="eval_results/motionfix"
COMMON_SUBSET=1          # score every dir of a tag on the keyids they all share
MIN_CLIPS=32             # MotionFix's evaluator builds zero batches below this and crashes

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

[[ -x "${MFIX_PY}" ]] || { echo "No MotionFix venv at ${MFIX_PY}"; exit 1; }
[[ -d "${GEN_ROOT}" ]] || { echo "No generations at ${GEN_ROOT}"; exit 1; }

list_tags() { find "${GEN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort; }
# A "flat" entry is a config dir sitting directly under the root: it holds .npy itself
# instead of holding config subdirectories.
is_flat() { [[ -n "$(find "${GEN_ROOT}/$1" -maxdepth 1 -name '*.npy' -print -quit)" ]]; }

GROUP="legacy"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ $# -eq 0 ]]; then
  echo "Generations on disk under ${GEN_ROOT}:"
  while read -r t; do
    if is_flat "$t"; then
      echo "  ${t}  (flat config dir — scored together with the other flat ones)"
    else
      echo "  ${t}"
    fi
  done < <(list_tags)
  echo
  echo "Usage: $0 <TAG> [<TAG> ...] [--group NAME]   |   $0 --all"
  exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
  mapfile -t TAGS < <(list_tags)
else
  TAGS=("$@")
fi

# Split by layout: tagged sweeps are scored one per tag, flat config dirs are pooled into one.
TAGGED=(); FLAT=()
for T in "${TAGS[@]}"; do
  if [[ ! -d "${GEN_ROOT}/${T}" ]]; then
    echo "skipping ${T}: no such directory ${GEN_ROOT}/${T}"
  elif is_flat "${T}"; then
    FLAT+=("${T}")
  else
    TAGGED+=("${T}")
  fi
done

# Collect one --smpl_dir per config dir with enough clips. realpath --no-symlinks, NOT plain
# realpath: resolving through a symlink collapses distinct configs onto one basename, which is
# the collision this script is structured to avoid.
collect() {
  SMPL_DIRS=()
  for d in "$@"; do
    [[ -d "$d" ]] || continue
    n=$(find "$d" -maxdepth 1 -name '*.npy' | wc -l)
    if [[ "$n" -lt "${MIN_CLIPS}" ]]; then
      echo "  skipping $(basename "${d%/}"): ${n} clips < ${MIN_CLIPS}"
      continue
    fi
    SMPL_DIRS+=(--smpl_dir "$(realpath --no-symlinks "$d")")
  done
}

score() {
  local NAME="$1"; shift
  if [[ ${#SMPL_DIRS[@]} -eq 0 ]]; then
    echo "skipping ${NAME}: no config dir has >= ${MIN_CLIPS} clips"
    return
  fi
  local METRICS="${OUT_ROOT}/${NAME}_rescored.json"
  local SUMMARY_DIR="${OUT_ROOT}/${NAME}_rescored"
  mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"
  local SUBSET_FLAG=""
  [[ "${COMMON_SUBSET}" == "1" ]] && SUBSET_FLAG="--common_subset"

  log "${NAME}: scoring $(( ${#SMPL_DIRS[@]} / 2 )) config dirs"
  "${MFIX_PY}" src/eval/run_motionfix_metrics.py "${SMPL_DIRS[@]}" ${SUBSET_FLAG} \
    --per_clip --per_clip_dir "${SUMMARY_DIR}/per_clip" --out "${METRICS}"

  log "${NAME}: table -> ${SUMMARY_DIR}/summary.md"
  python src/eval/aggregate_summary.py --tmr "${METRICS}" --out_dir "${SUMMARY_DIR}"
}

for TAG in "${TAGGED[@]+"${TAGGED[@]}"}"; do
  collect "${GEN_ROOT}/${TAG}"/*/
  score "${TAG}"
done

if [[ ${#FLAT[@]} -gt 0 ]]; then
  FLAT_PATHS=()
  for T in "${FLAT[@]}"; do FLAT_PATHS+=("${GEN_ROOT}/${T}"); done
  collect "${FLAT_PATHS[@]}"
  score "${GROUP}"
fi

log "Done."
echo "Per-tag table:    ${OUT_ROOT}/<TAG>_rescored/summary.md"
echo "Per-clip CSVs:    ${OUT_ROOT}/<TAG>_rescored/per_clip/<config>.csv"
echo
echo "The CSV is the durable asset: keyid, similarity to target for the edit and for the"
echo "unedited source, the per-clip delta, and both ranks. Every later question — subset by"
echo "negation or by body-part coverage, bootstrap a CI, compare two mask modes clip-by-clip"
echo "— is a pandas one-liner over it, with no re-scoring."
