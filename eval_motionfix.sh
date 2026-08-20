#!/usr/bin/env bash
#
# eval_motionfix.sh — end-to-end MotionFix evaluation for one checkpoint: edit every test
# clip under its own instruction across a guidance sweep, then score with MotionFix's own
# TMR retrieval evaluator.
#
#   ./eval_motionfix.sh    # runs whatever the Config block below says
#
# Configure in the Config block; don't pass environment variables. The usual edits are
# SCALE_SCAN (1 for the cheap pre-flight, 0 for the scored run), MASK_MODES, SCALES, LIMIT.
#
# START WITH SCALE_SCAN=1. It runs no TMR and tells you which guidance scales actually move
# the motion, which is mask-dependent and worth knowing before committing to a scored run.
#
# COST: the number of SCALES is the main lever, not the number of clips — each scale is its
# own Stage-3 reverse loop, and each extra mask_mode re-pays the inversion on top.
# MASK_TIMESTEPS is ~1 % of the bill, so there is no reason to economise on it. Cut SCALES
# rather than clips. Re-running resumes: clips whose .npy already exists are skipped unless
# EXTRA="--overwrite", and changing a mask setting while re-using the same OUT_ROOT is
# refused by the editor's fingerprint guard rather than silently mixing two configurations.
#
# WHICH NUMBER TO QUOTE. `retrieval()` restricts its gallery to exactly the keyids you
# generated, so a LIMIT=320 run's whole-gallery R@k is a 320-way retrieval and reads
# systematically HIGHER than the published 1013-way protocol. The batches-of-32 protocol is
# 32-way whatever the run size — quote that one. Stage 4 flags this in the rendered table.
#
# MINIMUM 32 CLIPS. MotionFix's evaluator builds batches with `range(len(keyids) // 32)`, so
# fewer than 32 generations produces ZERO batches and crashes on an IndexError; any remainder
# above a multiple of 32 is silently dropped. Stage 3 refuses to run rather than hitting that.

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RUN="exp_smplh_verbs"
CHECKPOINT="runs/${RUN}/checkpoint_latest"
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"       # SMPL-H root with the 135-d Mean/Std
MFIX_PY="data/motionfix/mfix-env/bin/python"     # MotionFix's own venv, for their evaluator

# 1 = the pre-flight: few clips, MANY scales (they share the inversion, so they are cheap),
# no TMR — just "how far does each scale move the motion". Uses the SCAN_* values below and
# stops after stage 1. Do this before choosing SCALES for a scored run.
SCALE_SCAN=0

# ── the scored run ────────────────────────────────────────────────────────────────
# The mask is M = M1 ∩ M2, with M1 owning the GROUP axis and M2 the TEMPORAL one, so these
# three modes are a clean decomposition: each component alone, then the composition.
# m2_only = M2 alone: the right frames, every body-part group. The historical mask.
# attn    = M1 ∩ M2: what the grounding loss was for. THE PRIMARY ARM — listed second so an
#           interrupted run still has the m2_only/attn comparison.
# m1_only = M1 alone: the right groups, EVERY FRAME. M1 is the stronger of the two on the
#           group axis but is flat in time by construction — under M1_SELECT=rank exactly
#           flat, since m_sem is a (G,) vector broadcast over all frames. So this edits the
#           selected 1-3 groups for the whole clip, and neither LAMBDA_ applies (lambda_attn
#           is inert under rank, lambda_noise inert with no M2). Expect it to trade
#           preservation (R@k_s2t) away for edit-following; that is the point of having it.
# groups  = the no-LLM router control. Not in the default list because it SKIPS clips it
#           cannot route (those naming no body part), which both changes the clip set and
#           shrinks its retrieval gallery, inflating its R@k. Stage 3 scores on the common
#           subset (COMMON_SUBSET=1) so a mixed list stays comparable.
MASK_MODES="m2_only attn m1_only"

# Guidance scales, PER MODE — a given scale does very different things depending on how many
# cells the mask lets through, so one list across all three would compare a mode that barely
# moves the body against one that wrecks the pose, and the TMR gap would be a magnitude
# artefact rather than a statement about mask quality. These bands are picked from a scan so
# the three sweeps span a comparable slice of the preservation-vs-edit curve. Scale 0
# reconstructs the source in every mode: the plumbing check.
declare -A SCALES_BY_MODE=(
  [m2_only]="0 2 5 12"
  [attn]="0 6 16 30"          # 16 and 30 are EXTRAPOLATED past the s=12 scan; re-scan to confirm
  [m1_only]="0 0.25 0.7 1.4"  # saturating: its useful band is below 1.5
)
SCALES="0 2.5 5 7.5"             # fallback for any mode not listed above

# 0 = the full 1013-clip test set, the default deliberately. A subsample is not just noisier,
# it is BIASED in a known direction: `retrieval()` sizes its gallery to the clips you
# generate, so a 320-clip run is a 320-way retrieval whose R@k reads high against the
# published protocol and cannot be quoted beside it. Use a limit for a cheap look, never for
# a reported table.
LIMIT=0

# ── the scan (used only when SCALE_SCAN=1) ────────────────────────────────────────
# All three modes, because the useful band is mask-dependent and they differ a lot in mask
# SIZE: attn (M1 ∩ M2) is the smallest and moves the body least at a given s, m1_only edits
# its groups across every frame and so moves it most. A band picked on m2_only alone would be
# wrong for both others. 16 clips, because a magnitude-vs-scale curve is a strong,
# low-variance signal. Do NOT read quality off this many clips — that is the scored run's job.
SCAN_MASK_MODES="m2_only attn m1_only"
SCAN_SCALES="0 0.5 1 2 3 5 8 12"
SCAN_LIMIT=16

# ── mask read-out settings ────────────────────────────────────────────────────────
PSI_READOUT="energy"      # 'energy' = signed change in per-group motion energy; 'abs' = the
                          # LEDITS++ magnitude |x0_c - x0_ref|
M1_COLUMNS="auto"         # used by mask_mode=attn and m1_only. 'auto' (= semantic on a
                          # grounded checkpoint) is the no-parser default. 'span' scores
                          # better but runs the caption parser at inference AND falls back to
                          # the all-token read on instructions naming no body part, so one
                          # score sheet then mixes two M1 read-outs (the editor warns and
                          # lists the affected keyids in its manifest).
M1_SELECT="rank"          # 'rank' picks M1's groups by rank and thresholds psi inside them,
                          # fixing the "lambda_attn is a cell budget, not a selector" defect:
                          # at 70 the percentile must hand out 0.30*G group-rows whatever the
                          # map says, so a one-group instruction always spills into the
                          # runner-up. It is also the parser-free route to what
                          # --m1_columns span buys. 'percentile' is the alternative.
M1_RANK_RATIO=0.5         # keep groups holding >= this share of the top group's M1 mass
M1_RANK_MAX=3             # hard cap, stops a flat map selecting the whole body
LAMBDA_ATTN=70            # M1 percentile — UNUSED when M1_SELECT=rank (the default)
LAMBDA_NOISE=70           # M2/psi percentile (higher = sparser mask)
MASK_TIMESTEPS=40         # 40 is indistinguishable from all 1000 and ~1 % of the run's cost
SEED=42                   # the inversion is stochastic; fix it. Derived per-clip inside the
                          # editor, so equal-length clips don't share a noise realisation.

# ── clip selection and scoring ────────────────────────────────────────────────────
LIMIT_MODE="random"       # 'first' = a contiguous block of sorted keyids
LIMIT_SEED=0              # separate from SEED: the clip SET and the noise vary independently
COMMON_SUBSET=1           # score every dir on the keyids they all share (needed whenever the
                          # mask modes have different coverage, a no-op when they do not)
EXTRA=""                  # extra flags for the editor, e.g. "--overwrite"

# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
if [[ "${SCALE_SCAN}" == "1" ]]; then
  MASK_MODES="${SCAN_MASK_MODES}"
  LIMIT="${SCAN_LIMIT}"
  # The scan is exploratory: one wide net for every mode is the point, since its whole job is
  # to find where each mode's band actually is.
  SCALES="${SCAN_SCALES}"
  SCALES_BY_MODE=()
fi

# Normalise scale strings through Python's %g, which is what names the output dirs
# ({mask_mode}_s{scale:g}). Without this, "0.0 2.50" makes the shell look for _s0.0/_s2.50
# while the editor wrote _s0/_s2.5, and `realpath` does not fail on a missing leaf — so the
# mismatch surfaces much later as a crash inside the scorer.
norm_scales() { python -c 'import sys; print(" ".join(f"{float(x):g}" for x in sys.argv[1:]))' $1; }
scales_for()  { norm_scales "${SCALES_BY_MODE[$1]:-${SCALES}}"; }

# TAG carries the run AND the read-out settings, so a new configuration can never overwrite a
# recorded sweep — the per-scale dirs are named {mask_mode}_s{scale} and would collide
# otherwise. The rule is "tag anything that is not this script's own default", so a default
# run gets a short name and any deviation is visible in the path. The editor's fingerprint
# guard is the backstop when a tag still collides.
TAG="${RUN}_${PSI_READOUT}"
[[ "${M1_COLUMNS}" == "auto" ]] || TAG="${TAG}_${M1_COLUMNS}"
[[ "${M1_SELECT}"  == "rank" ]] || TAG="${TAG}_${M1_SELECT}"
[[ "${SCALE_SCAN}" == "1" ]]    && TAG="${TAG}_scan" || true

OUT_ROOT="data/motionfix/motionfix_smpl/${TAG}"
METRICS="eval_results/motionfix/${TAG}_tmr.json"
SUMMARY_DIR="eval_results/motionfix/${TAG}"
SCAN_JSON="eval_results/motionfix/${TAG}.json"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------
# Preflight: fail here, not hours in
# ----------------------------------------------------------------------
[[ -f "${CHECKPOINT}/config.json" ]] || { echo "no checkpoint at ${CHECKPOINT}"; exit 1; }
[[ -f "${DATA_ROOT}/Mean.npy" ]]     || { echo "no Mean.npy in ${DATA_ROOT}"; exit 1; }
[[ -x "${MFIX_PY}" ]]                || { echo "no MotionFix venv at ${MFIX_PY}"; exit 1; }
python - "$CHECKPOINT" <<'PY'
import json, sys
c = json.load(open(f"{sys.argv[1]}/config.json"))
if c.get("feature_mode") != "smplh":
    sys.exit(f"checkpoint is feature_mode={c.get('feature_mode')!r}; this eval is SMPL-H only")
print(f"  checkpoint    {sys.argv[1]}")
print(f"  trained with  predict_type={c.get('predict_type')} "
      f"lambda_ground={c.get('attn_ground_weight')} verbs={c.get('attn_ground_verbs')}")
PY
mkdir -p "$(dirname "${METRICS}")" "${OUT_ROOT}"

echo "  mask modes    ${MASK_MODES}"
for MODE in ${MASK_MODES}; do
  printf '  scales        %-9s %s\n' "${MODE}" "$(scales_for "${MODE}")"
done
echo "  psi_readout   ${PSI_READOUT}    m1_columns ${M1_COLUMNS}    m1_select ${M1_SELECT}"
if [[ "${M1_SELECT}" == "rank" ]]; then
  echo "  rank          ratio ${M1_RANK_RATIO}  max ${M1_RANK_MAX}   (lambda_attn unused)"
else
  echo "  lambda        attn ${LAMBDA_ATTN}"
fi
echo "  lambda_noise  ${LAMBDA_NOISE}    seed ${SEED}"
if [[ "${LIMIT}" == "0" ]]; then
  echo "  clips         all 1013 (full test set — the only publication-comparable gallery)"
else
  echo "  clips         ${LIMIT}  (${LIMIT_MODE} subsample, limit_seed ${LIMIT_SEED})"
  # Scan mode runs no retrieval, so it has no gallery to warn about.
  [[ "${SCALE_SCAN}" == "1" ]] || \
    echo "                ⚠ ${LIMIT}-way retrieval gallery — reads high, NOT comparable to published numbers"
fi
echo "  out           ${OUT_ROOT}"
if [[ "${SCALE_SCAN}" == "1" ]]; then
  echo "  SCALE SCAN — magnitude only, no TMR. Pick SCALES from this, then run for real."
else
  echo "  metrics       ${METRICS}"
fi

# ----------------------------------------------------------------------
# 1. Edit: generate the edited motions
# ----------------------------------------------------------------------
SMPL_DIRS=()
for MODE in ${MASK_MODES}; do
  MODE_SCALES="$(scales_for "${MODE}")"
  log "Editing: mask_mode=${MODE}  scales=${MODE_SCALES}"
  python src/eval/edit_motionfix_testset.py \
    --checkpoint       "${CHECKPOINT}" \
    --smplh_data_root  "${DATA_ROOT}" \
    --out_root         "${OUT_ROOT}" \
    --mask_mode        "${MODE}" \
    --scales           ${MODE_SCALES} \
    --psi_readout      "${PSI_READOUT}" \
    --m1_columns       "${M1_COLUMNS}" \
    --m1_select        "${M1_SELECT}" \
    --m1_rank_ratio    "${M1_RANK_RATIO}" \
    --m1_rank_max      "${M1_RANK_MAX}" \
    --lambda_attn      "${LAMBDA_ATTN}" \
    --lambda_noise     "${LAMBDA_NOISE}" \
    --mask_timesteps   "${MASK_TIMESTEPS}" \
    --seed             "${SEED}" \
    --limit_mode       "${LIMIT_MODE}" \
    --limit_seed       "${LIMIT_SEED}" \
    --limit            "${LIMIT}" \
    ${EXTRA}
  for S in ${MODE_SCALES}; do
    SMPL_DIRS+=(--smpl_dir "$(realpath "${OUT_ROOT}/${MODE}_s${S}")")
  done
done

# ----------------------------------------------------------------------
# 2. Scan mode stops here: report magnitude per scale, no TMR
# ----------------------------------------------------------------------
if [[ "${SCALE_SCAN}" == "1" ]]; then
  log "Scale scan -> ${SCAN_JSON}"
  python src/eval/scale_scan.py --out_root "${OUT_ROOT}" --out "${SCAN_JSON}"
  echo
  echo "Copy the suggested scales into SCALES in the Config block, set SCALE_SCAN=0,"
  echo "and re-run for the scored evaluation."
  exit 0
fi

# ----------------------------------------------------------------------
# 3. Score with MotionFix's own TMR evaluator (their venv, their code)
# ----------------------------------------------------------------------
# Check the real file count rather than LIMIT — clips can also be dropped by --min_frames,
# and mask_mode=groups skips whatever it cannot route.
# SMPL_DIRS is [--smpl_dir, path, --smpl_dir, path, ...]; index 1 is the first path, already
# built by the loop above. Rebuilding it here would be a second copy of the {mode}_s{scale}
# naming convention that has to stay in sync with the editor.
FIRST_DIR="${SMPL_DIRS[1]}"
N_GEN=$(find "${FIRST_DIR}" -name '*.npy' 2>/dev/null | wc -l)
if [[ "${N_GEN}" -lt 32 ]]; then
  echo
  echo "Generated ${N_GEN} clips in ${FIRST_DIR}, but MotionFix's TMR evaluator needs >= 32"
  echo "(it scores in batches of 32 and drops the remainder). Stage 1 output is on disk and"
  echo "is fine; skipping scoring. Set LIMIT=0 in the Config block, or LIMIT>=32 for a"
  echo "scoreable smoke test."
  exit 0
fi
log "Scoring ${#SMPL_DIRS[@]} dirs through TMR"
echo "    ${N_GEN} generations -> $(( N_GEN / 32 )) batches of 32; $(( N_GEN % 32 )) dropped"
SUBSET_FLAG=""
[[ "${COMMON_SUBSET}" == "1" ]] && SUBSET_FLAG="--common_subset"
"${MFIX_PY}" src/eval/run_motionfix_metrics.py "${SMPL_DIRS[@]}" ${SUBSET_FLAG} --out "${METRICS}"

# ----------------------------------------------------------------------
# 4. Render the comparable table
# ----------------------------------------------------------------------
log "Summary -> ${SUMMARY_DIR}/summary.md"
python src/eval/aggregate_summary.py --tmr "${METRICS}" --out_dir "${SUMMARY_DIR}"

log "Complete. ${METRICS} and ${SUMMARY_DIR}/summary.md"
echo "Read it as: *_s2t = retrieval against the SOURCE (preservation), the other against"
echo "the TARGET (instruction-following). scale 0 must score ~100 R@1_s2t — if it does not,"
echo "the plumbing is broken and nothing else in the table means anything."
echo "Quote the _b (batches-of-32) columns: the whole-gallery ones are an N-way retrieval"
echo "with N = your clip count, so they read high on any subsampled run."
