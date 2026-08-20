"""
Batch LEDITS++ editing over the MotionFix test set, producing MotionFix-comparable
generations for TMR retrieval scoring.

Runs the SMPL-H editor on every MotionFix test source clip conditioned on that clip's edit
instruction, across a sweep of guidance scales, and writes one .npy per (scale, keyid) in the
layout MotionFix's `tmr_evaluator.collect_gen_samples` expects:
`[trans(3) | global_orient_6d(6) | body_pose_6d(126)]` at the dataset-native 30 fps.

Pipeline per clip (see docs / plan for the why):
  source rots/trans (30 fps)  --resample-->  20 fps
    --smplh_to_features-->  135-d training feature  --normalise-->  x0
    --invert (once)--> collect_masks (m2_only, once) --edit(scale)--> edited feature
    --denormalise + features_to_smpl-->  raw SMPL (20 fps)  --resample--> 30 fps
    --smpl_to_gen_layout-->  (T,135) gen layout  ->  {out_root}/{mask_mode}_s{scale}/{keyid}.npy

The scale=0 config reconstructs the source and doubles as the plumbing calibration
(expected R@1_s2t ~ 100). This script is SMPL-H only (feature_mode=="smplh").

The Stage-2 mask flags come from `utils.cli.add_mask_args`, i.e. the SAME definitions
`edit_motion.py` and the probes use — so `--m1_select rank`, `--per_step_norm` and the
`--m1_window`/`--m2_window` sweeps are runnable here and cannot drift from their defaults
elsewhere.

⚠ **The TMR gallery is whatever you generate.** MotionFix's `retrieval()` builds its
retrieval set as `MotionFixLoader(keys_to_load=<the keyids you supplied>)`, so a `--limit 320`
run is scored as 320-way retrieval, not 1013-way, and its "full" R@k is NOT comparable to any
published MotionFix number nor to a run that skipped a different number of clips. The
batches-of-32 protocol is fixed at 32-way and is comparable. See run_motionfix_metrics.py.

Score the output with (MotionFix venv):
    data/motionfix/mfix-env/bin/python src/eval/run_motionfix_metrics.py \
        --smpl_dir "$PWD"/data/motionfix/motionfix_smpl/m2_only_s0 \
        --smpl_dir "$PWD"/data/motionfix/motionfix_smpl/m2_only_s5 \
        --out eval_results/motionfix/tmr_metrics.json

Example (smoke test on 16 clips):
    python src/eval/edit_motionfix_testset.py \
        --checkpoint runs/smplh_exp/checkpoint_latest \
        --smplh_data_root data/HumanML3D/HumanML3D_smplh \
        --scales 0 5 --limit 16
"""

import os
import sys

# These scripts live one level below src/, so src/ is not on the path when they are run
# directly. Put it there before any project import.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import json

import numpy as np
import torch

from data.body_part_labels import route_groups
from data.smplh_features import (features_to_smpl, resample_motion, smpl_to_gen_layout,
                                 smplh_to_features)
from editing import MotionEditor, derive_seed
from editing.masking import mask_mode_components
from model.body_groups import resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_columns, resolve_readout_layers
from utils.cli import (add_logging_args, add_mask_args, add_model_args,
                       configure_logging, parse_group_mask, resolve_device)
from utils.logger import get_logger
from eval.provenance import check_resumable, mask_fingerprint, select_keyids
from utils.model_io import load_model
from utils.paths import REPO_ROOT
from utils.probe import resolve_sweeps

log = get_logger(__name__)


def parse_args():
    repo = REPO_ROOT
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --checkpoint / --no_ema / --device, and the whole Stage-2 mask block (thresholds,
    # windows, --m1_select rank, --per_step_norm, --psi_readout, --seed, --edit_space …)
    # shared verbatim with edit_motion.py and the probes.
    add_model_args(p)
    add_mask_args(p, mask_timesteps=40, alpha_floor=True)
    p.add_argument("--smplh_data_root", required=True,
                   help="SMPL-H training root with the 135-d Mean.npy / Std.npy.")
    p.add_argument("--testset",
                   default=os.path.join(repo, "data/motionfix/data/motionfix-dataset/"
                                              "motionfix_test.pth.tar"),
                   help="MotionFix test joblib dump (id -> {motion_source, motion_target, text}).")
    p.add_argument("--out_root", default=os.path.join(repo, "data/motionfix/motionfix_smpl"),
                   help="Root for the per-scale output folders. NOTE the default root "
                        "already holds legacy m2_only_s* dirs from an older, differently "
                        "configured run; the fingerprint guard will refuse to resume on "
                        "top of them. Point this at a per-configuration subdirectory.")
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 2.5, 5.0, 7.5],
                   help="SEGA guidance scales to sweep (one output folder each).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "temporal", "groups"],
                   help="Stage-2 mask source (default m2_only: automated, no LLM/attention). "
                        "'groups' routes the instruction to its body-part groups with the "
                        "caption parser (no LLM) — the correct-by-construction control. "
                        "Clips whose instruction names no body part are SKIPPED and "
                        "listed in the manifest, since a group mask has no answer there.")
    p.add_argument("--src_fps", type=float, default=30.0, help="MotionFix native fps.")
    p.add_argument("--edit_fps", type=float, default=20.0, help="Editor (HumanML3D) fps.")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Only N clips (0 = all). Smoke test.")
    p.add_argument("--limit_mode", default="random", choices=["random", "first"],
                   help="How --limit subsamples. 'random' (default) = a SEEDED random "
                        "sample, reproducible from --limit_seed. 'first' = the first N "
                        "sorted keyids, the pre-2026-08-16 behaviour — note MotionFix "
                        "keyids inherit AMASS/BABEL subject/sequence ordering, so that is "
                        "a contiguous block of the corpus, not a sample of it.")
    p.add_argument("--limit_seed", type=int, default=0,
                   help="Seed for --limit_mode random. Separate from --seed so the clip "
                        "SET and the inversion noise can be varied independently.")
    p.add_argument("--overwrite", action="store_true", help="Recompute clips whose .npy exists.")
    p.add_argument("--ignore_fingerprint", action="store_true",
                   help="Resume even when the existing manifest's settings disagree with "
                        "this invocation's. Produces a score sheet mixing two "
                        "configurations — only for deliberately continuing an interrupted "
                        "run whose flags you have since edited cosmetically.")
    add_logging_args(p)
    return configure_logging(p.parse_args())


def main():
    args = parse_args()
    device = resolve_device(args.device)
    log.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode = config.get("feature_mode", "humanml3d")
    if feature_mode != "smplh":
        raise SystemExit(f"This script is SMPL-H only, but checkpoint feature_mode={feature_mode!r}. "
                         "Train/point to an smplh checkpoint.")

    mean = np.load(os.path.join(args.smplh_data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.smplh_data_root, "Std.npy"))
    if mean.shape[0] != 135:
        raise SystemExit(f"Expected 135-d SMPL-H stats, got {mean.shape[0]} at {args.smplh_data_root}.")

    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)
    _, is_group, group_mode, _ = resolve_group_context(config)
    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space, psi_readout=args.psi_readout,
                          attn_layers=resolve_readout_layers(config, args.m1_layers))
    log.info(f"predict_type={schedule.predict_type}  edit_space={editor.edit_space}  "
          f"psi_readout={editor.psi_readout}  m1_columns={args.m1_columns}  "
          f"m1_select={args.m1_select}")

    # Per-mask sweeps: shared unless --m1_window/--m2_window narrow one of them.
    mask_ts, m1_ts, m2_ts = resolve_sweeps(args.mask_timesteps, schedule.T,
                                           args.m1_window, args.m2_window)

    scale_tag = lambda s: f"{args.mask_mode}_s{s:g}"
    out_dirs = {s: os.path.join(args.out_root, scale_tag(s)) for s in args.scales}
    fingerprint = mask_fingerprint(args, editor, config)
    manifest_path = os.path.join(args.out_root, f"edit_manifest_{args.mask_mode}.json")
    check_resumable(args, out_dirs, fingerprint, manifest_path)
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    # Stamp the fingerprint BEFORE editing anything, so a run killed halfway leaves files
    # that are still identifiable and can be resumed. Writing it only at the end would make
    # every interrupted run look, to the guard, like generations of unknown provenance.
    with open(manifest_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "status": "in-progress"}, f, indent=2)

    import joblib
    log.info(f"Loading test set: {args.testset}")
    data = joblib.load(args.testset)
    keyids = select_keyids(data, args)
    limit_note = ""
    if len(keyids) < len(data):
        seed_note = f", limit_seed={args.limit_seed}" if args.limit_mode == "random" else ""
        limit_note = f" ({args.limit_mode} subsample of {len(data)}{seed_note})"
    log.info(f"{len(keyids)} clips{limit_note} × {len(args.scales)} scales -> {args.out_root}")

    need_attn = mask_mode_components(args.mask_mode)[0] == "attn"
    skipped = {}
    routed, col_modes, col_fallback = {}, {}, []
    n_done = 0
    for k in log.progress(keyids, desc="Editing", leave=True):
        # which scales still need this clip?
        todo = [s for s in args.scales
                if args.overwrite or not os.path.exists(os.path.join(out_dirs[s], f"{k}.npy"))]
        if not todo:
            continue

        src = data[k]["motion_source"]
        rots = np.asarray(src["rots"], dtype=np.float32)      # (T,66) aa @ src_fps
        trans = np.asarray(src["trans"], dtype=np.float32)    # (T,3)
        text = data[k]["text"]

        # 30 -> 20 fps, featurise, normalise
        r20, t20 = resample_motion(rots, trans, args.src_fps, args.edit_fps)
        T = r20.shape[0]
        if T < args.min_frames:
            skipped[k] = f"too short ({T} < {args.min_frames})"
            continue
        if T > args.max_frames:
            r20, t20, T = r20[:args.max_frames], t20[:args.max_frames], args.max_frames
        A = smplh_to_features(r20, t20)                       # (T,135)
        x0 = torch.from_numpy((A - mean) / std).float().unsqueeze(0).to(device)
        valid = torch.ones(T, dtype=torch.bool, device=device)

        # The group router has no answer for an instruction naming no body part (~41% of
        # this test set), so skip rather than fall back to a different mask — mixing two
        # mask sources inside one score sheet would make the comparison unreadable.
        group_masks = None
        if args.mask_mode == "groups":
            names = route_groups(text, group_mode)
            if not names:
                skipped[k] = "router found no body part in the instruction"
                continue
            routed[k] = names
            group_masks = [parse_group_mask(" ".join(names), is_group, group_mode)]

        # invert + collect masks ONCE per clip (both independent of guidance scale)
        state = editor.invert(x0, show_progress=False, seed=derive_seed(args.seed, k))
        with torch.no_grad():
            ctx = text_encoder.encode([text])
            # Only M1 reads text columns; resolving them under m2_only/groups would cost a
            # parse per clip and report a read-out nothing uses.
            if need_attn:
                tok, sem, cmode = resolve_readout_columns(
                    text, text_encoder, config, args.m1_columns, group_mode)
                col_modes[cmode] = col_modes.get(cmode, 0) + 1
                if cmode.startswith("content (no body-part span)"):
                    col_fallback.append(k)
            else:
                tok, sem = None, None
        masks = editor.collect_masks(
            state, [ctx], [tok], valid, semantic_idxs_per_edit=[sem],
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode, llm_group_masks=group_masks, timesteps=mask_ts,
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=args.per_step_norm,
            m1_select=args.m1_select, m1_rank_ratio=args.m1_rank_ratio,
            m1_rank_max=args.m1_rank_max,
        )

        for s in todo:
            x_edit = editor.edit(state, [ctx], masks, scales=[s], show_progress=False,
                                 guidance_alpha_floor=args.guidance_alpha_floor)   # (1,T,135)
            A_edit = x_edit[0].cpu().numpy() * std + mean                          # raw 135 @20fps
            re, te = features_to_smpl(A_edit)                                      # raw SMPL @20fps
            r30, t30 = resample_motion(re, te, args.edit_fps, args.src_fps)        # -> 30 fps
            gen = smpl_to_gen_layout(r30, t30)                                     # (T,135) gen layout
            np.save(os.path.join(out_dirs[s], f"{k}.npy"), gen)
        n_done += 1

    # Files ON DISK, not files written by this invocation — a resumed run edits only the
    # clips it found missing, so n_edited alone under-reports what the scorer will read.
    n_present = {f"{s:g}": len([f for f in os.listdir(out_dirs[s]) if f.endswith(".npy")])
                 for s in args.scales}

    manifest = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "feature_mode": feature_mode,
        "mask_mode": args.mask_mode,
        # Which arithmetic produced these edits — ψ_x0 + ungated high-noise guidance
        # ("x0") or the historical ε-space path. Needed to compare two score sheets.
        "predict_type": schedule.predict_type,
        "edit_space": editor.edit_space,
        "guidance_alpha_floor": editor.resolve_alpha_floor(args.guidance_alpha_floor),
        "scales": args.scales,
        "src_fps": args.src_fps, "edit_fps": args.edit_fps,
        "n_clips": len(keyids), "n_edited_this_run": n_done, "n_present": n_present,
        "n_skipped": len(skipped), "skipped": skipped,
        "limit": args.limit, "limit_mode": args.limit_mode, "limit_seed": args.limit_seed,
        # What actually produced these masks. A score sheet without these is ambiguous
        # now that M1 has a column axis and psi has two read-outs. `fingerprint` is the
        # subset that must match for a resume to be sound (see check_resumable).
        "fingerprint": fingerprint,
        "m1_columns": args.m1_columns, "m1_columns_resolved": col_modes,
        "m1_columns_fallback": col_fallback,
        "psi_readout": editor.psi_readout,
        "seed": args.seed, "seed_mode": "per-clip (seed*1000003 + crc32(keyid))",
        "routed_groups": routed,
        "out_dirs": {f"{s:g}": out_dirs[s] for s in args.scales},
        "status": "complete",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"\nDone: edited {n_done} clips this run, skipped {len(skipped)}; "
          f"{min(n_present.values()) if n_present else 0} on disk per scale.")
    if skipped:
        log.info("Skipped (first few): %s", dict(list(skipped.items())[:5]))
    if col_fallback:
        n_m1 = sum(col_modes.values())
        log.warning(
            "--m1_columns %r fell back to the all-content-token read on %d/%d clips "
            "(%.1f%%) — those instructions name no body part, so there is no supervised "
            "span to restrict M1 to. This score sheet therefore mixes TWO M1 read-outs. "
            "Report it split, or re-run with --m1_columns semantic for one read-out "
            "throughout. The affected keyids are in the manifest as "
            "m1_columns_fallback.",
            args.m1_columns, len(col_fallback), n_m1,
            100 * len(col_fallback) / max(n_m1, 1))
    log.info("Output folders: %s", ", ".join(out_dirs.values()))


if __name__ == "__main__":
    main()
