#!/usr/bin/env bash
#
# rescore_motionfix.sh — re-score MotionFix generations that ALREADY EXIST on disk with the
# metrics R@1 cannot express. No editing, no inversion, no generation: it loads the .npy files
# you already have and runs TMR over them.
#
#   ./rescore_motionfix.sh                      # list the sweeps it can find, then stop
#   ./rescore_motionfix.sh exp_smplh_verbs      # prefix match is fine
#   ./rescore_motionfix.sh --all
#   ./rescore_motionfix.sh --gen_root some/dir --all
#
# WHICH FOLDER. It wants the **MotionFix editing** output: directories of `<keyid>.npy`, each a
# (T, 135) SMPL-H feature array, written by `eval_motionfix.sh` (OUT_ROOT, default
# data/motionfix/motionfix_smpl/<TAG>/<mask_mode>_s<scale>/). That is NOT `generated/`, which
# holds `<keyid>.npz` text-to-motion samples from `generate.py` — different task, different
# format, and nothing here can score them. The script checks and says so rather than skipping.
#
# LAYOUT-AGNOSTIC. It assumes no directory depth: it finds every directory that directly
# contains .npy files and groups them by their parent, so <TAG>/<config>/*.npy and flat
# <config>/*.npy both work, at any nesting.
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
# COST: one TMR pass per config dir (plus a second for the per-clip baseline). Minutes, not
# hours — the expensive part of an eval is Stage 3, and that is already done.
#
# ⚠ One scoring call per SWEEP, never one call spanning several. run_motionfix_metrics.py keys
# its results by the directory BASENAME, so passing two sweeps' `attn_s2` dirs to a single call
# would silently overwrite one with the other (this bug was found and fixed once already — see
# MaskOptions.md §13.2). Within a sweep the basenames are unique; across sweeps they are not,
# which is why each sweep gets its own output file.

set -euo pipefail
cd "$(dirname "$0")"

MFIX_PY="data/motionfix/mfix-env/bin/python"
OUT_ROOT="eval_results/motionfix"
COMMON_SUBSET=1          # score every dir of a sweep on the keyids they all share
MIN_CLIPS=32             # MotionFix's evaluator builds zero batches below this and crashes
FLAT_GROUP="legacy"      # sweep name for config dirs sitting directly under a root

# Searched in order unless --gen_root is given. The first is eval_motionfix.sh's own OUT_ROOT.
DEFAULT_ROOTS=("data/motionfix/motionfix_smpl" "generated")

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

ROOTS=(); NAMES=(); ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gen_root) ROOTS+=("${2%/}"); shift 2 ;;
    --all)      ALL=1; shift ;;
    -h|--help)  sed -n '2,48p' "$0"; exit 0 ;;
    *)          NAMES+=("$1"); shift ;;
  esac
done
[[ ${#ROOTS[@]} -eq 0 ]] && ROOTS=("${DEFAULT_ROOTS[@]}")

[[ -x "${MFIX_PY}" ]] || { echo "No MotionFix venv at ${MFIX_PY}"; exit 1; }

# ----------------------------------------------------------------------
# Discovery: a "config dir" is any directory that directly contains .npy files; its parent
# names the sweep. Nothing here depends on how deep the tree is.
# ----------------------------------------------------------------------
SWEEP_NAMES=(); SWEEP_DIRS=(); NPZ_ONLY=()

add_dir() {
  local sweep="$1" dir="$2" i
  for i in "${!SWEEP_NAMES[@]}"; do
    if [[ "${SWEEP_NAMES[$i]}" == "$sweep" ]]; then
      SWEEP_DIRS[$i]="${SWEEP_DIRS[$i]}"$'\n'"$dir"
      return
    fi
  done
  SWEEP_NAMES+=("$sweep"); SWEEP_DIRS+=("$dir")
}

for ROOT in "${ROOTS[@]}"; do
  [[ -d "$ROOT" ]] || continue
  while IFS= read -r d; do
    if [[ -n "$(find "$d" -maxdepth 1 -name '*.npy' -print -quit)" ]]; then
      parent="$(dirname "$d")"
      if [[ "$parent" == "$ROOT" ]]; then
        sweep="${FLAT_GROUP}"
      else
        sweep="${parent#"$ROOT"/}"; sweep="${sweep//\//_}"
      fi
      add_dir "$sweep" "$d"
    elif [[ -n "$(find "$d" -maxdepth 1 -name '*.npz' -print -quit)" ]]; then
      NPZ_ONLY+=("$d")
    fi
  done < <(find "${ROOT}" -mindepth 1 -type d | sort)
done

diagnose() {
  local i n_cfg n_clip
  echo
  echo "Searched: ${ROOTS[*]}"
  if [[ ${#SWEEP_NAMES[@]} -gt 0 ]]; then
    echo
    echo "Sweeps found (MotionFix editing output, <keyid>.npy):"
    for i in "${!SWEEP_NAMES[@]}"; do
      n_cfg=$(printf '%s\n' "${SWEEP_DIRS[$i]}" | wc -l)
      n_clip=$(find "$(printf '%s\n' "${SWEEP_DIRS[$i]}" | head -1)" -maxdepth 1 -name '*.npy' | wc -l)
      printf '  %-44s %2s config dirs, ~%s clips each\n' "${SWEEP_NAMES[$i]}" "$n_cfg" "$n_clip"
    done
  else
    echo "  (no directory containing <keyid>.npy was found)"
  fi
  if [[ ${#NPZ_ONLY[@]} -gt 0 ]]; then
    echo
    echo "Ignored — these hold .npz, not .npy. That is generate.py's text-to-motion output,"
    echo "not MotionFix editing output, and this benchmark cannot score it:"
    printf '  %s\n' "${NPZ_ONLY[@]}" | head -8
    [[ ${#NPZ_ONLY[@]} -gt 8 ]] && echo "  ... and $(( ${#NPZ_ONLY[@]} - 8 )) more"
  fi
  echo
  echo "MotionFix edits are written by eval_motionfix.sh to"
  echo "  data/motionfix/motionfix_smpl/<TAG>/<mask_mode>_s<scale>/<keyid>.npy"
  echo "where TAG is RUN plus the readout suffix — e.g. exp_smplh_verbs_energy, not"
  echo "exp_smplh_verbs. Point elsewhere with --gen_root DIR."
}

if [[ ${#NAMES[@]} -eq 0 && "$ALL" -eq 0 ]]; then
  diagnose
  echo
  echo "Usage: $0 <SWEEP> [<SWEEP> ...] | --all   [--gen_root DIR]"
  exit 0
fi

# ----------------------------------------------------------------------
# Selection: exact name, else prefix, else substring. A miss is an ERROR with a listing —
# never a silent skip that still exits 0, which is how you end up believing a sweep scored.
# ----------------------------------------------------------------------
SELECTED=()
if [[ "$ALL" -eq 1 ]]; then
  SELECTED=("${SWEEP_NAMES[@]+"${SWEEP_NAMES[@]}"}")
else
  for want in "${NAMES[@]}"; do
    hits=()
    for s in "${SWEEP_NAMES[@]+"${SWEEP_NAMES[@]}"}"; do [[ "$s" == "$want" ]] && hits+=("$s"); done
    if [[ ${#hits[@]} -eq 0 ]]; then
      for s in "${SWEEP_NAMES[@]+"${SWEEP_NAMES[@]}"}"; do [[ "$s" == "$want"* ]] && hits+=("$s"); done
      [[ ${#hits[@]} -gt 0 ]] && echo "'${want}' matched by prefix: ${hits[*]}"
    fi
    if [[ ${#hits[@]} -eq 0 ]]; then
      for s in "${SWEEP_NAMES[@]+"${SWEEP_NAMES[@]}"}"; do [[ "$s" == *"$want"* ]] && hits+=("$s"); done
      [[ ${#hits[@]} -gt 0 ]] && echo "'${want}' matched by substring: ${hits[*]}"
    fi
    if [[ ${#hits[@]} -eq 0 ]]; then
      echo "ERROR: nothing on disk matches '${want}'."
      diagnose
      exit 1
    fi
    SELECTED+=("${hits[@]}")
  done
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "ERROR: no sweep selected — there is nothing to score."
  diagnose
  exit 1
fi

# ----------------------------------------------------------------------
# Pre-flight + scoring
# ----------------------------------------------------------------------
check_shape() {
  # A wrong folder is far cheaper to catch here than 40 s into a TMR load with an opaque
  # shape error. MotionFix edits are (T, 135) SMPL-H features.
  local f
  f="$(find "$1" -maxdepth 1 -name '*.npy' -print -quit)"
  python3 - "$f" <<'PY'
import sys, numpy as np
a = np.load(sys.argv[1], allow_pickle=True)
if getattr(a, "ndim", 0) != 2 or a.shape[-1] != 135:
    print(f"      WARNING {sys.argv[1]}: shape {getattr(a, 'shape', type(a))}, expected (T, 135). "
          f"This does not look like MotionFix SMPL-H editing output.")
PY
}

for SWEEP in "${SELECTED[@]}"; do
  for i in "${!SWEEP_NAMES[@]}"; do
    [[ "${SWEEP_NAMES[$i]}" == "$SWEEP" ]] || continue

    SMPL_DIRS=()
    while IFS= read -r d; do
      [[ -n "$d" ]] || continue
      n=$(find "$d" -maxdepth 1 -name '*.npy' | wc -l)
      if [[ "$n" -lt "${MIN_CLIPS}" ]]; then
        echo "  skipping $(basename "$d"): ${n} clips < ${MIN_CLIPS}"
        continue
      fi
      check_shape "$d"
      # realpath --no-symlinks, NOT plain realpath: resolving through a symlink collapses
      # distinct configs onto one basename, which is the collision noted at the top.
      SMPL_DIRS+=(--smpl_dir "$(realpath --no-symlinks "$d")")
    done <<< "${SWEEP_DIRS[$i]}"

    if [[ ${#SMPL_DIRS[@]} -eq 0 ]]; then
      echo "skipping ${SWEEP}: no config dir has >= ${MIN_CLIPS} clips"
      continue
    fi

    METRICS="${OUT_ROOT}/${SWEEP}_rescored.json"
    SUMMARY_DIR="${OUT_ROOT}/${SWEEP}_rescored"
    mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"
    SUBSET_FLAG=""
    [[ "${COMMON_SUBSET}" == "1" ]] && SUBSET_FLAG="--common_subset"

    log "${SWEEP}: scoring $(( ${#SMPL_DIRS[@]} / 2 )) config dirs"
    "${MFIX_PY}" src/eval/run_motionfix_metrics.py "${SMPL_DIRS[@]}" ${SUBSET_FLAG} \
      --per_clip --per_clip_dir "${SUMMARY_DIR}/per_clip" --out "${METRICS}"

    log "${SWEEP}: table -> ${SUMMARY_DIR}/summary.md"
    python src/eval/aggregate_summary.py --tmr "${METRICS}" --out_dir "${SUMMARY_DIR}"
  done
done

log "Done."
echo "Per-sweep table:  ${OUT_ROOT}/<SWEEP>_rescored/summary.md"
echo "Per-clip CSVs:    ${OUT_ROOT}/<SWEEP>_rescored/per_clip/<config>.csv"
echo
echo "The CSV is the durable asset: keyid, similarity to target for the edit and for the"
echo "unedited source, the per-clip delta, and both ranks. Every later question — subset by"
echo "negation or by body-part coverage, bootstrap a CI, compare two mask modes clip-by-clip"
echo "— is a pandas one-liner over it, with no re-scoring."
