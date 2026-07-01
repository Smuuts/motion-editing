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
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo/src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from editing import MotionEditor
from data.smplh_features import (
    smplh_to_features, features_to_smpl, smpl_to_gen_layout, resample_motion,
)
from sample_model import load_model


def parse_args():
    repo = os.path.dirname(src_dir)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir with config.json + ema.pt/model.pt (feature_mode=smplh).")
    p.add_argument("--smplh_data_root", required=True,
                   help="SMPL-H training root with the 135-d Mean.npy / Std.npy.")
    p.add_argument("--testset",
                   default=os.path.join(repo, "data/motionfix/data/motionfix-dataset/"
                                              "motionfix_test.pth.tar"),
                   help="MotionFix test joblib dump (id -> {motion_source, motion_target, text}).")
    p.add_argument("--out_root", default=os.path.join(repo, "data/motionfix/motionfix_smpl"),
                   help="Root for the per-scale output folders.")
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 2.5, 5.0, 7.5],
                   help="SEGA guidance scales to sweep (one output folder each).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn"],
                   help="Stage-2 mask source (default m2_only: automated, no LLM/attention).")
    p.add_argument("--lambda_noise", type=float, default=70.0,
                   help="M2 percentile threshold (higher = sparser mask).")
    p.add_argument("--lambda_attn", type=float, default=70.0)
    p.add_argument("--mask_timesteps", type=int, default=None,
                   help="Use this many evenly-spaced timesteps for mask collection "
                        "(default: all). Small values (e.g. 40) speed up the run a lot.")
    p.add_argument("--guidance_alpha_floor", type=float, default=0.03)
    p.add_argument("--src_fps", type=float, default=30.0, help="MotionFix native fps.")
    p.add_argument("--edit_fps", type=float, default=20.0, help="Editor (HumanML3D) fps.")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Only the first N clips (0 = all). Smoke test.")
    p.add_argument("--overwrite", action="store_true", help="Recompute clips whose .npy exists.")
    p.add_argument("--no_ema", action="store_true", help="Load model.pt instead of ema.pt.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

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
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    editor = MotionEditor(model, schedule, device, is_group=True)

    mask_ts = (torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
               if args.mask_timesteps else None)

    scale_tag = lambda s: f"{args.mask_mode}_s{s:g}"
    out_dirs = {s: os.path.join(args.out_root, scale_tag(s)) for s in args.scales}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    import joblib
    print(f"Loading test set: {args.testset}")
    data = joblib.load(args.testset)
    keyids = sorted(data.keys())
    if args.limit:
        keyids = keyids[:args.limit]
    print(f"{len(keyids)} clips × {len(args.scales)} scales -> {args.out_root}")

    skipped = {}
    n_done = 0
    for k in tqdm(keyids, desc="Editing"):
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

        # invert + collect masks ONCE per clip (both independent of guidance scale)
        state = editor.invert(x0, show_progress=False)
        with torch.no_grad():
            ctx = text_encoder.encode([text])
            toks = text_encoder.token_info(text)[0]
        masks = editor.collect_masks(
            state, [ctx], [toks], valid,
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode, timesteps=mask_ts,
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

    manifest = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "feature_mode": feature_mode,
        "mask_mode": args.mask_mode,
        "scales": args.scales,
        "src_fps": args.src_fps, "edit_fps": args.edit_fps,
        "n_clips": len(keyids), "n_edited": n_done,
        "n_skipped": len(skipped), "skipped": skipped,
        "out_dirs": {f"{s:g}": out_dirs[s] for s in args.scales},
    }
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, f"edit_manifest_{args.mask_mode}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: edited {n_done} clips, skipped {len(skipped)}.")
    if skipped:
        print("Skipped (first few):", dict(list(skipped.items())[:5]))
    print("Output folders:", ", ".join(out_dirs.values()))


if __name__ == "__main__":
    main()
