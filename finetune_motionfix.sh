#!/usr/bin/env bash
#
# finetune_motionfix.sh — supervised fine-tune of a pretrained SMPL-H checkpoint on the
# MotionFix triplets, then a magnitude scan under the mask mode the objective actually
# matches. Runs on the TRAINING SERVER: stage 1 builds a feature cache from the 5.1 GB
# triplet dump, which has OOM'd a 16 GB box before.
#
#   ./finetune_motionfix.sh    # runs whatever the Config block below says
#
# Configure in the Config block; don't pass environment variables. Same convention as
# eval_motionfix.sh.
#
# WHY THIS RUN EXISTS. The recorded fine-tune result (R@1 +0.29 pp, "also a null") rests on a
# measurement with two defects that BOTH bias it downward:
#   (a) it evaluated checkpoint_epoch_0099, but validation bottomed at ~600-700 OPTIMISER STEPS
#       (epoch 14 at batch 128; epoch 7 at batch 64 — epochs are not comparable across batch
#       sizes, steps are) and rose thereafter. `--save_every 25` meant the best model was
#       never written to disk. Confirmed by the 2026-09-03 re-run, which put the optimum at
#       672 steps against the prior run's 630;
#   (b) it scored under mask_mode=attn, which admits ~12.9 % of (frame, group) cells and
#       hard-inpaints the rest back to the source — so ~87 % of the edit the model was
#       trained to make was discarded before scoring.
# The fine-tune objective carries NO spatial mask: it regresses the whole target, every
# channel, every frame. mask_mode=none is the matching arm. Note the asymmetry that made this
# easy to miss: `none` is catastrophic for the PRETRAINED model (R@1 41.94 at s=5, because
# unmasked guidance destroys the motion), but for a fine-tuned model whose conditional
# direction points at the target it is precisely where it should help.
#
# WHAT DEPENDS ON THE ANSWER. "The supervised fine-tune is also null" is load-bearing: it
# appears in the abstract, in the "the bottleneck is the guidance direction, not the data"
# argument, and as a contribution. If this run moves the number, three parts of the thesis
# change. Treat it as a headline experiment, not housekeeping.
#
# COST. The previous 100-epoch runs took ~15 min of GPU (4,200 steps at batch 64, ~8.8 s per
# epoch after the padding fix took throughput 232 -> 697 clips/s). Stage 1 here is SHORTER.
# The one-off expense is the feature cache built from the 5.1 GB dump on the first run;
# afterwards CACHE_DIR is reused and startup is seconds. Stage 2 is 16 clips x 8 scales.
#
# MEMORY. Stage 1 frees the triplet dump before the model and T5 load, but peak RSS during
# cache construction is the risk. On a memory-tight machine set PRELOAD=0 and
# PRECOMPUTE_TEXT=0 (slower, much lower peak) — see the Config block.

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE_RUN="exp_smplh_verbs"                        # the pretrained backbone to fine-tune
BASE_CKPT="runs/${BASE_RUN}/checkpoint_latest"
DATA_ROOT="data/HumanML3D/HumanML3D_smplh"        # SMPL-H root with the 135-d Mean/Std
OUT_DIR="runs/ft_motionfix_best"                  # NOT ft_motionfix — that holds the old,
                                                  # pre-fix run and its epoch-99 checkpoints
CACHE_DIR="runs/ft_cache"                         # featurised triplets, shared across arms

# Which arm to run. See the two blocks below.
#   honest = change NOTHING except checkpoint selection and the epoch budget. This is the arm
#            the thesis claim needs: a trustworthy number under the recorded recipe.
#   tuned  = the one hyperparameter with a MEASURED reason to change (see TUNED_* below).
#            Run this only after the honest arm has a number to compare against.
ARM="honest"

# ── stage control ─────────────────────────────────────────────────────────────────
DO_TRAIN=1
DO_SCAN=1        # magnitude scan under mask_mode=none. Cheap, and it is what tells you which
                 # scales to put in eval_motionfix.sh for a scored run.

# ── the honest arm ────────────────────────────────────────────────────────────────
# Defaults are deliberately untouched: lr 1e-5 (10x below pretrain — the objective changes,
# not the weights), batch 64, EMA 0.999, t_max 800 (sqrt(alpha_bar) = 0.305 there; above it
# the source is gone and the task degenerates into generating a motion from a relative
# instruction), cfg_dropout 0.1, geometric losses at the pretrain weights.
EPOCHS=40             # a CEILING, not a target. The measured optimum is ~600-700 optimiser
                      # steps — at batch 64 that is epoch 7, and the 2026-09-03 run early-stopped
                      # at 19. Early stopping is what ends the run; if checkpoint_best ever lands
                      # at >= EPOCHS-2, raise this rather than trusting the result.
VAL_EVERY=2           # at batch 64 the optimum is ~epoch 7, so the prior run's every-5 grid
                      # bracketed it only to [420, 840] steps. 2 resolves it.
EARLY_STOP=6          # consecutive stale validations => 12 epochs without improvement. Fired at
                      # epoch 19 on the 2026-09-03 run with the best at 7, i.e. it gives the
                      # minimum a wide berth while refusing to burn 85 epochs of overfitting the
                      # way the first run did.
SAVE_EVERY=10         # periodic snapshots; checkpoint_best is written independently on every
                      # validation improvement, so the best model survives regardless
GROUND_LABELS="parser_first"
                      # The default. Do NOT re-run the diff_temporal A/B: it was measured
                      # unpowered — relative L2 between the two fine-tuned weight sets was
                      # 0.00117 against 0.0258 from the pretrained model, i.e. the label
                      # change moved the weights 4.5 % as far as the fine-tune itself, and
                      # validation differed by 0.0005 at every logged epoch. Repeating it
                      # spends GPU to reproduce a null that was never resolvable.
GROUND_WEIGHT=5e-3    # same as pretraining
SEED=42

# ── the tuned arm ─────────────────────────────────────────────────────────────────
# ⚠ THE ORIGINAL JUSTIFICATION FOR THIS ARM WAS FALSIFIED ON 2026-09-03 — read before running.
# It was written on the recorded claim that "m_S falls monotonically 0.757 -> 0.729", i.e. that
# the fine-tune progressively erodes the grounding and 5e-3 cannot hold it. Plotting the full
# series instead of its endpoints kills that: m_S drops fast over the first ~10 epochs and then
# OSCILLATES FLAT (min 0.711 @ ep26, max 0.759 @ ep5, last 0.729), and the 2026-09-03 re-run
# reproduces the same shape. It is a one-time equilibrium shift of ~0.03, not decay.
#
# So there is no runaway to arrest — only an equilibrium to move, and the term is already
# holding the model 0.03 below where pretraining left it rather than losing ground over time.
# This arm is still a legitimate single-factor test ("can more pressure recover the pretrain
# equilibrium?"), but it is NO LONGER the highest-value next step. See PROGRESS.md "Adapting the
# fine-tune so it helps generation AND grounding" — interleaving HumanML3D batches and
# instruction-swap negatives both attack the measured failure (the objective is satisfiable
# without reading the instruction) rather than a symptom of it.
#
# Everything else stays at the honest arm's value, so an honest-vs-tuned comparison is a
# single-factor test in the same style as the rest of the project's checkpoint table.
TUNED_GROUND_WEIGHT=2e-2      # 4x. Tests whether the equilibrium moves back toward 0.75+.
TUNED_OUT_DIR="runs/ft_motionfix_tuned"

# ── memory ────────────────────────────────────────────────────────────────────────
# Both default ON in the script and cost RAM: PRELOAD ~0.43 GB, PRECOMPUTE_TEXT ~2.25 GB (and
# it then drops T5 from VRAM, so batch_size can rise). Set both to 0 on a tight box.
PRELOAD=1
PRECOMPUTE_TEXT=1
NUM_WORKERS=2
BATCH_SIZE=64
AMP_DTYPE="auto"      # bf16 wherever supported — two runs have been lost to fp16 overflow

# ── stage 2: the magnitude scan ───────────────────────────────────────────────────
SCAN_MASK_MODE="none"  # the arm the objective matches (see header)
# Scan LOW. Two independent reasons the band is far below the pretrained model's:
#   1. the fine-tune made guidance ~8x more potent per unit scale (the same rotmax needed
#      s~2.2 where the pretrained model needed s=16, consistent across three tiers);
#   2. removing the mask lets every cell move, which raises magnitude per unit scale again.
# Scale 0 is the plumbing check: it must reconstruct the source exactly in every mode.
SCAN_SCALES="0 0.1 0.25 0.5 1 2 3 5"
SCAN_LIMIT=16          # a magnitude-vs-scale curve is a strong, low-variance signal. Do NOT
                       # read quality off 16 clips — that is the scored run's job.
SCAN_OUT_ROOT="data/motionfix/ft_scan"
PSI_READOUT="energy"
M1_SELECT="rank"
MASK_TIMESTEPS=40
# Target tiers to aim for, so the fine-tuned arm is comparable to the pretrained one at
# MATCHED edit magnitude rather than at matched nominal scale: rotmax ~3 / 8 / 15 degrees.
# Read them off the scan JSON and put them in eval_motionfix.sh's SCALES_BY_MODE.

# ----------------------------------------------------------------------
# Derived
# ----------------------------------------------------------------------
if [[ "${ARM}" == "tuned" ]]; then
  GROUND_WEIGHT="${TUNED_GROUND_WEIGHT}"
  OUT_DIR="${TUNED_OUT_DIR}"
fi

BEST_CKPT="${OUT_DIR}/checkpoint_best"
SCAN_JSON="${SCAN_OUT_ROOT}/scan_$(basename "${OUT_DIR}").json"

bool_flag() {  # $1 = 0/1, $2 = flag stem -> "--stem" or "--no-stem"
  [[ "$1" == "1" ]] && echo "--$2" || echo "--no-$2"
}

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ----------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------
[[ -e "${BASE_CKPT}" ]] || { echo "missing base checkpoint: ${BASE_CKPT}" >&2; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "missing SMPL-H data root: ${DATA_ROOT}" >&2; exit 1; }

log "arm=${ARM}  base=${BASE_RUN}  out=${OUT_DIR}  ground_weight=${GROUND_WEIGHT}"

# ----------------------------------------------------------------------
# Stage 1 — fine-tune
# ----------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
  log "Stage 1: fine-tune -> ${OUT_DIR}"
  mkdir -p "${OUT_DIR}"      # the logger opens train.log before the trainer creates this
  python src/finetune_motionfix.py \
      --checkpoint       "${BASE_CKPT}" \
      --smplh_data_root  "${DATA_ROOT}" \
      --output_dir       "${OUT_DIR}" \
      --cache_dir        "${CACHE_DIR}" \
      --ground_labels    "${GROUND_LABELS}" \
      --attn_ground_weight "${GROUND_WEIGHT}" \
      --epochs           "${EPOCHS}" \
      --batch_size       "${BATCH_SIZE}" \
      --val_every        "${VAL_EVERY}" \
      --save_every       "${SAVE_EVERY}" \
      --early_stop       "${EARLY_STOP}" \
      --amp_dtype        "${AMP_DTYPE}" \
      --num_workers      "${NUM_WORKERS}" \
      --seed             "${SEED}" \
      $(bool_flag "${PRELOAD}" preload) \
      $(bool_flag "${PRECOMPUTE_TEXT}" precompute_text) \
      --log_file         "${OUT_DIR}/train.log"

  log "Stage 1 done. Read these off ${OUT_DIR}/metrics.jsonl before going on:"
  cat <<'CHECKS'
    - the epoch checkpoint_best points at. If it is >= EPOCHS-2 the run was still improving
      and the budget, not overfitting, ended it: raise EPOCHS and re-run.
    - the optimum IN STEPS (epoch x steps_per_epoch), not in epochs — epochs are not comparable
      across batch sizes. Prior run 630, 2026-09-03 run 672. A materially different number means
      something other than checkpoint selection changed; find out what before trusting it.
    - m_S as a SERIES, not endpoints. It settles ~0.03 below pretraining within ~10 epochs and
      then oscillates flat; first-vs-last on that series reads as a decline and is not one.
      An endpoint delta is not a trend.
    - non-finite step count. Both previous runs were clean; anything above zero here wants
      explaining, not ignoring.
CHECKS
fi

# ----------------------------------------------------------------------
# Stage 2 — magnitude scan under mask_mode=none
# ----------------------------------------------------------------------
if [[ "${DO_SCAN}" == "1" ]]; then
  [[ -e "${BEST_CKPT}" ]] || { echo "no ${BEST_CKPT} — did stage 1 run?" >&2; exit 1; }
  log "Stage 2: magnitude scan, mask_mode=${SCAN_MASK_MODE}, ${SCAN_LIMIT} clips"

  python src/eval/edit_motionfix_testset.py \
      --checkpoint       "${BEST_CKPT}" \
      --smplh_data_root  "${DATA_ROOT}" \
      --out_root         "${SCAN_OUT_ROOT}" \
      --mask_mode        "${SCAN_MASK_MODE}" \
      --scales           ${SCAN_SCALES} \
      --limit            "${SCAN_LIMIT}" \
      --psi_readout      "${PSI_READOUT}" \
      --m1_select        "${M1_SELECT}" \
      --mask_timesteps   "${MASK_TIMESTEPS}" \
      --seed             "${SEED}"

  mkdir -p "$(dirname "${SCAN_JSON}")"
  python src/eval/scale_scan.py \
      --out_root "${SCAN_OUT_ROOT}" \
      --mask_mode "${SCAN_MASK_MODE}" \
      --out "${SCAN_JSON}"

  log "Scan written to ${SCAN_JSON}"
  cat <<'NEXT'
    NEXT STEPS (not run here — the scored run is hours of GPU):
      1. Check scale 0 reconstructs exactly. If it does not, stop: the plumbing is wrong and
         every other number is meaningless.
      2. Pick the scales hitting rotmax ~3 / 8 / 15 degrees. Those are the pretrained arm's
         tiers, so matching them is what makes the two comparable at equal edit magnitude
         rather than equal nominal scale. Report `rot deg` (skeleton mean) alongside
         `rotmax`: rotmax rewards spatial concentration and so under-penalises masks of
         different spatial extent, and `none` has the largest extent of any mode.
      3. Put those scales in eval_motionfix.sh:
             RUN / CHECKPOINT -> this fine-tuned checkpoint_best
             MASK_MODES="none"
             SCALES_BY_MODE=( [none]="0 <s1> <s2> <s3>" )
             SCALE_SCAN=0 ; LIMIT=0
         LIMIT=0 matters: retrieval() sizes its gallery to the clips you generate, so a
         subsampled run's whole-gallery R@k is not comparable to the published protocol.
      4. Compare against the PRETRAINED arm at matched magnitude, and against each arm's own
         scale 0 — not against the other arm's raw number.
NEXT
fi

log "done"
