#!/usr/bin/env bash
#
# eval_motionfix_seeds.sh — the MotionFix table as mean ± std over inversion seeds, with the
# smoothed mask of the thesis (eq:softgate): M1 n M2 exactly as in tab:motionfix, then the
# selected groups released in every frame, M2 blurred over MASK_BLUR frames into a guidance
# weight, and the spine released without guidance.
#
#   ./eval_motionfix_seeds.sh scan           # 16 clips, many scales, no TMR -> pick SCALES
#   ./eval_motionfix_seeds.sh bench          # 1 vs 2 vs 3 parallel jobs, ~10 min, prints clips/h
#   ./eval_motionfix_seeds.sh smoke 42       # 32 clips through the WHOLE pipeline (edit, TMR,
#                                            # summary, aggregate) — the pre-flight, ~10 min
#   ./eval_motionfix_seeds.sh 42 43 44       # full 1013-clip runs, one per seed, scored
#   ./eval_motionfix_seeds.sh aggregate      # mean ± std over every finished seed
#
# PARALLEL. At batch size 1 the GPU is mostly idle, so run several seed lists as separate
# processes (use `bench` to decide how many), e.g.
#   nohup ./eval_motionfix_seeds.sh 42 44 46 48 50 > logs/mf_seeds_a.log 2>&1 &
#   nohup ./eval_motionfix_seeds.sh 43 45 47 49 51 > logs/mf_seeds_b.log 2>&1 &
# Each seed writes its own folders, so processes never touch each other's files. A killed run
# resumes where it stopped (clips whose .npy exists are skipped), and the editor's fingerprint
# guard refuses to mix settings if the Config block changed in between.
#
# WHAT THE STD MEASURES. The seed sets the inversion noise (derived per clip), i.e. the
# randomness of the edit itself for ONE trained model. It is not training-run variance.
#
# Configure in the Config block; don't pass environment variables. Activate the `ma` env first.

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RUN="exp_smplh_llm_labels"                       # the model behind tab:motionfix
CHECKPOINT="runs/${RUN}/checkpoint_latest"
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"
MFIX_PY="data/motionfix/mfix-env/bin/python"     # MotionFix's own venv, for their evaluator

# Guidance scales of the scored runs, WITHOUT 0. Fill in from `scan`: the smoothed mask edits
# far MORE than the hard M1 n M2 at the same s_e (s=3: most-moved joint 28 deg here, 9.4 deg in
# the hard table), so the old table's 1/2/3/8 do not carry over. Pick three whose rotmax is near
# 3, 9 and 20 deg, the span of the current table.
SCALES=""
MAGNITUDE_SEED=42      # this seed also writes scale 0 (~+1 h): the do-nothing / plumbing row
                       # and the reference for the achieved edit sizes. Other seeds skip it.

# Mask: identical to the run behind tab:motionfix ...
MASK_MODE="attn"
PSI_READOUT="energy"
M1_COLUMNS="span"
M1_SELECT="rank"
M1_RANK_RATIO=0.5
M1_RANK_MAX=3
LAMBDA_NOISE=30           # M2 keeps 70 % of the frames of the selected groups
MASK_TIMESTEPS=40
# ... plus the smoothing of eq:softgate and the released spine
MASK_BLUR=2
NEIGHBOURS="spine"        # released unguided wherever the mask edits, unless selected itself

SCAN_SCALES="0 0.25 0.5 0.75 1 1.5 2 3"
SCAN_LIMIT=16
SCAN_SEED=42
BENCH_LIMIT=16
SMOKE_LIMIT=32            # the TMR evaluator needs >= 32 clips

# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
NB_TAG="${NEIGHBOURS// /-}"
TAG="${RUN}_${PSI_READOUT}_${M1_COLUMNS}_blur${MASK_BLUR}_${NB_TAG:-noneighbours}"
NB_FLAG=()
[[ -n "${NEIGHBOURS}" ]] && NB_FLAG=(--neighbour_groups ${NEIGHBOURS})
OUT_BASE="data/motionfix/motionfix_smpl/${TAG}"   # + _seed<S>/attn_s<scale>/<keyid>.npy
RES_BASE="eval_results/motionfix/${TAG}"         # + _seed<S>_tmr.json, _seed<S>/summary.md

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
norm_scales() { python -c 'import sys; print(" ".join(f"{float(x):g}" for x in sys.argv[1:]))' "$@"; }

preflight() {
  [[ -f "${CHECKPOINT}/config.json" ]] || { echo "no checkpoint at ${CHECKPOINT}"; exit 1; }
  [[ -f "${DATA_ROOT}/Mean.npy" ]]     || { echo "no Mean.npy in ${DATA_ROOT}"; exit 1; }
  [[ -x "${MFIX_PY}" ]]                || { echo "no MotionFix venv at ${MFIX_PY}"; exit 1; }
  PYTHONPATH=src python -c "import editing.masking.release" || \
    { echo "src/editing/masking/release.py missing — pull the latest code"; exit 1; }
  mkdir -p eval_results/motionfix logs
}

# edit <out_root> <seed> <limit> <scales...>
edit() {
  local out="$1" seed="$2" limit="$3"; shift 3
  python src/eval/edit_motionfix_testset.py \
    --checkpoint       "${CHECKPOINT}" \
    --smplh_data_root  "${DATA_ROOT}" \
    --out_root         "${out}" \
    --mask_mode        "${MASK_MODE}" \
    --scales           "$@" \
    --psi_readout      "${PSI_READOUT}" \
    --m1_columns       "${M1_COLUMNS}" \
    --m1_select        "${M1_SELECT}" \
    --m1_rank_ratio    "${M1_RANK_RATIO}" \
    --m1_rank_max      "${M1_RANK_MAX}" \
    --lambda_noise     "${LAMBDA_NOISE}" \
    --mask_timesteps   "${MASK_TIMESTEPS}" \
    --mask_blur        "${MASK_BLUR}" \
    ${NB_FLAG[@]+"${NB_FLAG[@]}"} \
    --seed             "${seed}" \
    --limit            "${limit}"
}

# score <out_root> <res_prefix> <scales...>  -> <res_prefix>_tmr.json and <res_prefix>/summary.md
score() {
  local out="$1" res="$2"; shift 2
  local dirs=()
  for S in $(norm_scales "$@"); do dirs+=(--smpl_dir "$(realpath "${out}/${MASK_MODE}_s${S}")"); done
  local n_gen
  n_gen=$(find "$(realpath "${out}/${MASK_MODE}_s$(norm_scales "$1")")" -name '*.npy' | wc -l)
  if [[ "${n_gen}" -lt 32 ]]; then
    echo "only ${n_gen} generations in ${out}; TMR needs >= 32 — not scoring"; return 0
  fi
  # --per_clip_dir per seed: the default is shared, so parallel seeds would overwrite each
  # other's per-clip CSVs.
  "${MFIX_PY}" src/eval/run_motionfix_metrics.py "${dirs[@]}" --common_subset --per_clip \
    --per_clip_dir "$(realpath -m "${res}/per_clip")" --out "${res}_tmr.json" || return 1
  python src/eval/aggregate_summary.py --tmr "${res}_tmr.json" --out_dir "${res}" || return 1
}

# run_seed <seed> <limit> <suffix>
run_seed() {
  local seed="$1" limit="$2" suffix="$3"
  local scales="${SCALES}"
  [[ "${seed}" == "${MAGNITUDE_SEED}" ]] && scales="0 ${SCALES}"
  local out="${OUT_BASE}${suffix}_seed${seed}" res="${RES_BASE}${suffix}_seed${seed}"
  log "seed ${seed}: editing ${limit/#0/all 1013} clips at scales ${scales} -> ${out}"
  local t0=$SECONDS
  edit "${out}" "${seed}" "${limit}" ${scales} || return 1
  log "seed ${seed}: edited in $(( (SECONDS - t0) / 60 )) min, scoring with TMR"
  score "${out}" "${res}" ${scales} || return 1
  log "seed ${seed}: done -> ${res}/summary.md"
}

aggregate() {
  local suffix="${1:-}"
  python src/eval/aggregate_seeds.py --glob "${RES_BASE}${suffix}_seed*_tmr.json" \
    --out_dir "${RES_BASE}${suffix}_seeds"
  local mag_out="${OUT_BASE}${suffix}_seed${MAGNITUDE_SEED}"
  if [[ -d "${mag_out}/${MASK_MODE}_s0" ]]; then
    log "achieved edit sizes (seed ${MAGNITUDE_SEED}, against its scale 0)"
    python src/eval/scale_scan.py --out_root "${mag_out}" \
      --out "${RES_BASE}${suffix}_seeds/achieved_magnitude.json"
  fi
}

# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------
[[ $# -ge 1 ]] || { sed -n '3,22p' "$0"; exit 1; }
preflight
echo "  tag           ${TAG}"
echo "  mask          ${MASK_MODE}  psi ${PSI_READOUT}  m1 ${M1_COLUMNS}/${M1_SELECT}" \
     "(${M1_RANK_RATIO}, ${M1_RANK_MAX})  lambda_noise ${LAMBDA_NOISE}"
echo "  soft          blur ${MASK_BLUR} frames, released unguided: ${NEIGHBOURS}"

case "$1" in
  scan)
    out="${OUT_BASE}_scan"
    log "scale scan: ${SCAN_LIMIT} clips, scales ${SCAN_SCALES}"
    edit "${out}" "${SCAN_SEED}" "${SCAN_LIMIT}" ${SCAN_SCALES}
    python src/eval/scale_scan.py --out_root "${out}" --out "${RES_BASE}_scan.json"
    echo
    echo "Pick three scales whose rotmax is near 3, 9 and 20 deg and put them into SCALES."
    ;;
  bench)
    [[ -n "${SCALES}" ]] || { echo "set SCALES first (run 'scan')"; exit 1; }
    s1=$(echo ${SCALES} | awk '{print $NF}')     # the largest scale, a full guided loop
    for n in 1 2 3; do
      log "bench: ${n} parallel job(s) x ${BENCH_LIMIT} clips at s=${s1}"
      t0=$SECONDS
      for i in $(seq 1 "${n}"); do
        python src/eval/edit_motionfix_testset.py --checkpoint "${CHECKPOINT}" \
          --smplh_data_root "${DATA_ROOT}" --out_root "${OUT_BASE}_bench_p${n}_${i}" \
          --mask_mode "${MASK_MODE}" --scales "${s1}" --psi_readout "${PSI_READOUT}" \
          --m1_columns "${M1_COLUMNS}" --m1_select "${M1_SELECT}" \
          --m1_rank_ratio "${M1_RANK_RATIO}" --m1_rank_max "${M1_RANK_MAX}" \
          --lambda_noise "${LAMBDA_NOISE}" --mask_timesteps "${MASK_TIMESTEPS}" \
          --mask_blur "${MASK_BLUR}" ${NB_FLAG[@]+"${NB_FLAG[@]}"} \
          --seed $((900 + i)) --limit "${BENCH_LIMIT}" --overwrite > "logs/bench_p${n}_${i}.log" 2>&1 &
      done
      wait
      dt=$(( SECONDS - t0 ))
      echo "  ${n} job(s): ${dt} s for $(( n * BENCH_LIMIT )) clips -> " \
           "$(( 3600 * n * BENCH_LIMIT / dt )) clips/h at one scale (incl. model loading)"
    done
    echo
    echo "Use the largest n that still raises clips/h clearly."
    ;;
  smoke)
    shift
    [[ -n "${SCALES}" ]] || { echo "set SCALES first (run 'scan')"; exit 1; }
    for seed in "${@:-${MAGNITUDE_SEED}}"; do
      run_seed "${seed}" "${SMOKE_LIMIT}" "_smoke" || { echo "!! smoke seed ${seed} FAILED"; exit 1; }
    done
    aggregate "_smoke"
    ;;
  aggregate)
    aggregate ""
    ;;
  *)
    [[ -n "${SCALES}" ]] || { echo "set SCALES first (run 'scan')"; exit 1; }
    for seed in "$@"; do
      [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "not a seed: ${seed}"; exit 1; }
    done
    failed=()
    for seed in "$@"; do
      # A failing seed is reported and skipped, so one bad run cannot cost the rest of the night.
      run_seed "${seed}" 0 "" || { echo "!! seed ${seed} FAILED, continuing"; failed+=("${seed}"); }
    done
    if [[ ${#failed[@]} -gt 0 ]]; then echo "failed seeds: ${failed[*]}"; exit 1; fi
    log "all seeds done: $*  (run './eval_motionfix_seeds.sh aggregate' once every process is)"
    ;;
esac
