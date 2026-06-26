"""
Stage B of the MotionFix evaluation: batch-edit every converted MotionFix source clip with
the trained GroupDiT editor, sweeping guidance scales.

Reads the manifest from src/data/motionfix_to_hml3d.py (one row per clip, `text` = the edit
instruction, `source` = a raw (T,263) HumanML3D feature file). For each clip it inverts once,
builds the Stage-2 mask once, then runs Stage-3 editing at each requested scale, saving the
edited RAW (de-normalised) (F,263) features to <out_root>/<cfg>/<id>.npy where <cfg> encodes
the (mask_mode, scale). Those feed Stage C (joints2smpl_fit.py).

Runs in the `ma` env (model + torch). Example:
    python src/eval_motionfix_edit.py \
        --checkpoint runs/exp_group_l/checkpoint_latest \
        --data_root data/HumanML3D \
        --manifest  data/motionfix_hml3d/test.jsonl \
        --out_root  data/motionfix_edited \
        --scales 2.5 5.0 7.5
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from editing import MotionEditor
from sample_model import load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True, help="HumanML3D root (Mean.npy, Std.npy).")
    p.add_argument("--manifest", required=True, help="test.jsonl from motionfix_to_hml3d.py.")
    p.add_argument("--out_root", default="data/motionfix_edited")
    p.add_argument("--scales", type=float, nargs="+", default=[2.5, 5.0, 7.5],
                   help="Guidance scales to sweep (one output dir per scale).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "llm"])
    p.add_argument("--lambda_noise", type=float, default=70.0)
    p.add_argument("--lambda_attn", type=float, default=70.0)
    p.add_argument("--mask_timesteps", type=int, default=None,
                   help="Evenly-spaced timesteps for mask collection (default all 1000).")
    p.add_argument("--guidance_alpha_floor", type=float, default=0.03)
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--limit", type=int, default=None, help="Only the first N clips (smoke test).")
    p.add_argument("--overwrite", action="store_true", help="Re-edit clips whose output exists.")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def cfg_name(mask_mode, scale):
    return f"{mask_mode}_s{scale:g}"


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    is_group = config.get("feature_mode", "humanml3d") == "group"
    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    editor = MotionEditor(model, schedule, device, is_group=is_group)
    need_tok = args.mask_mode in ("attn", "m1_only")

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_dirs = {s: os.path.join(args.out_root, cfg_name(args.mask_mode, s)) for s in args.scales}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"{len(rows)} clips | mask_mode={args.mask_mode} | scales={args.scales}")
    print("Output configs:", ", ".join(cfg_name(args.mask_mode, s) for s in args.scales))

    mask_ts = (torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
               if args.mask_timesteps else None)

    n_done = 0
    for i, row in enumerate(rows):
        clip_id, instruction = row["id"], row["text"]
        out_paths = {s: os.path.join(out_dirs[s], f"{clip_id}.npy") for s in args.scales}
        if not args.overwrite and all(os.path.exists(p) for p in out_paths.values()):
            continue

        raw = np.load(row["source"]).astype(np.float32)
        F = min(len(raw), args.max_frames)
        raw = raw[:F]
        x0 = torch.from_numpy((raw - mean) / std).float().unsqueeze(0).to(device)
        valid_frames = torch.ones(F, dtype=torch.bool, device=device)

        with torch.no_grad():
            state = editor.invert(x0, show_progress=False)
            ctx = text_encoder.encode([instruction])
            tok = text_encoder.token_info(instruction)[0] if need_tok else None
            masks = editor.collect_masks(
                state, [ctx], [tok], valid_frames,
                lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
                mask_mode=args.mask_mode, timesteps=mask_ts,
            )
            for s in args.scales:
                x_edit = editor.edit(state, [ctx], masks, scales=[s],
                                     guidance_alpha_floor=args.guidance_alpha_floor,
                                     show_progress=False)
                edited_raw = (x_edit[0].cpu().numpy() * std + mean).astype(np.float32)
                np.save(out_paths[s], edited_raw)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {i + 1}/{len(rows)} clips edited (last: {clip_id}, {F} frames)")

    print(f"Done: edited {n_done} clips into {args.out_root}/<cfg>/<id>.npy")


if __name__ == "__main__":
    main()
