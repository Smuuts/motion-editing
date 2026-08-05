"""
Decompose the implicit masks' instruction-invariance into its LATERALITY and CATEGORY
axes — the quantitative companion to `visualise_mask_problem.py`.

That script reports ONE number per mask: the mean off-diagonal correlation across a
contrasting instruction set ("does the mask change at all when the instruction
changes?"). It conflates two very different failures — laterality ("raise the LEFT arm"
vs "the RIGHT arm") and category ("raise the left ARM" vs "kick with the left LEG") —
which the project's history predicts should move apart, so a single mean can hide a
real effect or manufacture a fake one.

Alongside the split this reports per-instruction group profiles, binary-mask alignment
with the expected group against chance, and a paired within-clip category contrast (the
source's own motion bias cancels). Definitions live in analysis/mask_axes.py; the
machinery is the editor's own, so it runs on any checkpoint the editor runs on. Results
are written as JSON (one file per checkpoint × clip) and printed as a table.

Usage
-----
    # one checkpoint, several clips
    python src/probe_mask_axes.py --checkpoint runs/exp_hml3d_x0/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D --clip 012698 --clip 005742 \
        --out_dir eval_results/mask_problem_x0/quant

    # A/B two checkpoints (repeat --checkpoint); the table groups by clip
    python src/probe_mask_axes.py --checkpoint runs/exp_hml3d_x0/checkpoint_latest \
        --checkpoint runs/exp_hml3d_attn_sink/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D --clip 012698 --out_dir ...
"""

import os
import json
import argparse

import numpy as np
import torch

from analysis.instructions import DEFAULT_INSTRUCTIONS
from analysis.mask_axes import decompose, summary_row
from analysis.mask_probe import collect_instruction_masks
from data.clips import load_clip
from editing import MotionEditor
from model.body_groups import group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from utils.cli import add_data_args, add_mask_args, add_model_args, resolve_device
from utils.model_io import load_model
from utils.probe import resolve_sweeps, source_activity

TABLE_COLUMNS = ["M1 rcat", "M1 rlat", "M1 roff", "M2 rcat", "M2 rlat", "M2 roff",
                 "M1~src", "M2~src", "algM1", "algM2"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, multi_checkpoint=True)
    add_data_args(p, split=False)
    add_mask_args(p)
    p.add_argument("--clip", action="append", required=True,
                   help="Clip id in <data_root>/new_joint_vecs. Repeat for several.")
    p.add_argument("--out_dir", default="eval_results/mask_axes")
    return p.parse_args()


def probe_one(ckpt, clip, args, device) -> dict:
    """All M1/M2 statistics for one checkpoint × one clip."""
    model, config = load_model(ckpt, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    if not is_group:
        raise SystemExit(f"{ckpt}: flat model (G=1) has no body-part axis to decompose.")
    glabels = group_names(group_mode)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    raw, F, _ = load_clip(args.data_root, clip, args.max_frames)
    x0 = torch.from_numpy((raw - mean) / std).float().unsqueeze(0).to(device)
    valid = torch.ones(F, dtype=torch.bool, device=device)

    editor = MotionEditor(model, schedule, device, is_group=is_group)
    state = editor.invert(x0)
    src_act = source_activity(x0, editor.group_channels)

    # m1_only isolates what a grounded attention readout would drive; m2_only is the
    # editor default.
    m1_maps, m2_maps, binaries = collect_instruction_masks(
        model, schedule, editor, state, text_encoder, DEFAULT_INSTRUCTIONS, valid,
        is_group, mask_modes=("m1_only", "m2_only"), lambda_attn=args.lambda_attn,
        lambda_noise=args.lambda_noise,
        sweeps=resolve_sweeps(args.mask_timesteps, schedule.T,
                              args.m1_window, args.m2_window),
        per_step_norm=args.per_step_norm)

    return {
        "checkpoint": ckpt, "clip": clip, "frames": F,
        "predict_type": config.get("predict_type", "eps"),
        "arch": config.get("arch", "dit"), "feature_mode": feature_mode,
        "group_mode": group_mode,
        **decompose(m1_maps, m2_maps, binaries, src_act, glabels),
    }


def _format_row(label, target, values):
    return (f"{label:8} | {target:5} | "
            + "   ".join(f"{v:.3f}" for v in values[:3]) + "   | "
            + "   ".join(f"{v:.3f}" for v in values[3:6]) + "   | "
            + "  ".join(f"{v:+.3f}" for v in values[6:8]) + "  | "
            + "  ".join(f"{v:.3f}" for v in values[8:]))


def print_table(results):
    hdr = (f"{'clip':8} | {'target':5} | "
           + " ".join(f"{c:7}" for c in TABLE_COLUMNS[:3]) + " | "
           + " ".join(f"{c:7}" for c in TABLE_COLUMNS[3:6]) + " | "
           + " ".join(f"{c:7}" for c in TABLE_COLUMNS[6:8]) + " | "
           + " ".join(f"{c:6}" for c in TABLE_COLUMNS[8:]))
    print("\n" + hdr)
    print("-" * len(hdr))
    per_target = {}
    for d in results:
        row = summary_row(d)
        per_target.setdefault(d["predict_type"], []).append(row)
        print(_format_row(d["clip"], d["predict_type"], row))
    print("-" * len(hdr))
    for target, rows in per_target.items():
        print(_format_row("MEAN", target, np.mean(rows, axis=0)))
    print(f"\nchance alignment = {results[0]['align_chance']:.3f}   "
          f"(rcat → 1 = mask ignores arm-vs-leg;  rlat → 1 = mask ignores left-vs-right)")

    print("\npaired category contrast (mass the instruction MOVES onto its own limb):")
    for d in results:
        for key in ("m1", "m2"):
            cc = d[f"{key}_category_contrast"]
            print(f"  {d['clip']} {d['predict_type']:5} {key}: "
                  f"arm {cc['arm_shift']:+.3f}   leg {cc['leg_shift']:+.3f}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    results = []
    for ckpt in args.checkpoint:
        tag = os.path.basename(os.path.dirname(ckpt.rstrip("/"))) or "ckpt"
        for clip in args.clip:
            print(f"\n── {tag}  clip {clip} ──")
            res = probe_one(ckpt, clip, args, device)
            out = os.path.join(args.out_dir, f"{tag}_{clip}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"wrote {out}")
            results.append(res)

    print_table(results)


if __name__ == "__main__":
    main()
