"""
Visualise "the mask problem" — why the implicit LEDITS++ masks fail.

The core negative result of this project (docs/FINDINGS.md, docs/PROGRESS.md "Open
problems") is that both implicit masks are *source-dynamics-driven, not
instruction-driven*:

  M1 (cross-attention)  — meant to answer "which body-part group does the edit TEXT
                          attend to?", but the readout is nearly invariant to the
                          instruction (sink-dominated, no laterality).
  M2 (noise-estimate ψ) — meant to answer "where does the edit change the prediction?",
                          but it fires on whatever the SOURCE clip already moves
                          (r≈0.96 between opposite-laterality instructions).

This script makes that visible: one source clip through a set of deliberately
contrasting instructions (left vs right, arm vs leg), showing that the resulting masks
look like each other and like the source's own motion — not like the instruction.

It reuses the real editing stack, so it works for any checkpoint the editor works for:
humanml3d (263-d) or smplh (135-d), GroupDiT or GroupCLR U-Net, either token axis
(7 body-part groups or 22 per-joint tokens, read from the checkpoint), and the legacy
flat MotionDiT (G=1, where the body-part overlay is omitted).

Two figures per source clip:
  <clip>_mask_problem.png        per-instruction grid of raw M1 / raw M2 / final binary
                                 mask on a shared colour scale, above the
                                 instruction-independent source-|Δx0| reference; each
                                 instruction's expected group outlined in red.
  <clip>_mask_problem_quant.png  instruction×instruction correlation matrices for M1/M2
                                 and, per instruction, corr(mask, source motion).

Usage
-----
    python src/visualise_mask_problem.py \
        --checkpoint runs/exp_smplh_unet/checkpoint_latest \
        --data_root  data/HumanML3D_smplh \
        --source 0 --out_dir eval_results/mask_problem

    # Custom contrasting instructions + explicit expected groups for the overlay:
    python src/visualise_mask_problem.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the left arm"  --target_groups "left_arm" \
        --instruction "raise the right arm" --target_groups "right_arm"

`probe_mask_axes.py` is the quantitative companion: it splits the single off-diagonal
r reported here into its laterality and category components.
"""

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from analysis.instructions import DEFAULT_INSTRUCTIONS, resolve_targets
from analysis.mask_probe import active_cells, collect_instruction_masks
from data.clips import load_source
from editing import MotionEditor
from model.body_groups import GROUP_NAMES, group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from utils.cli import add_data_args, add_mask_args, add_model_args, resolve_device
from utils.decode import smplh_body_model
from utils.model_io import load_model
from utils.probe import flat_corr, pairwise_corr, resolve_sweeps, source_activity
from utils.visualise import mean_off_diagonal, plot_mask_problem, plot_mask_quant


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_data_args(p, source=True, smplh=True)
    add_mask_args(p)
    p.add_argument("--instruction", action="append", dest="instructions", default=None,
                   help=f"Edit instruction; repeat for a contrasting set (default "
                        f"{DEFAULT_INSTRUCTIONS}).")
    p.add_argument("--target_groups", action="append", default=None,
                   help="Expected group(s) for the red overlay, one per --instruction "
                        f"(names from {GROUP_NAMES}). Default: guessed from the text.")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "temporal"],
                   help="Which binary mask fills the 'final mask' column (default "
                        "m2_only, the editor default). Raw M1/M2 are always shown.")
    p.add_argument("--m1_readout", default="raw",
                   choices=["raw", "renorm", "spatial", "renorm_spatial"],
                   help="M1 per-cell attention readout (see masking.collect_statistics).")
    p.add_argument("--out_dir", default="eval_results/mask_problem")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    print(f"feature_mode={feature_mode}  arch={config.get('arch', 'dit')}  "
          f"is_group={is_group}  group_mode={group_mode}")
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    instructions = args.instructions or list(DEFAULT_INSTRUCTIONS)
    # No body-part axis ⇒ the laterality/limb overlay is meaningless.
    targets = (resolve_targets(instructions, args.target_groups, group_mode)
               if is_group else [[] for _ in instructions])

    raw_feat, clip_id, F, caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    print(f"Source: {clip_id}  ({F} frames)   prompt: {caption!r}\n"
          f"instructions: {instructions}")

    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space)
    print(f"predict_type={schedule.predict_type}  edit_space={editor.edit_space} "
          f"(ψ read as {'|x̂0_c − x̂0_ref|' if editor.edit_space == 'x0' else '|ε_c − ε_ref|'})")
    glabels = group_names(group_mode) if is_group else ["all"]
    src_act = source_activity(x0, editor.group_channels, is_group)   # (F, G) reference

    print("Stage 1: inversion …")
    state = editor.invert(x0)

    sweeps = resolve_sweeps(args.mask_timesteps, schedule.T, args.m1_window, args.m2_window)
    if args.m1_window or args.m2_window or args.per_step_norm:
        print(f"sweep: M1 {args.m1_window or 'full'}  M2 {args.m2_window or 'full'}  "
              f"per_step_norm={args.per_step_norm}")

    m1_maps, m2_maps, binaries = collect_instruction_masks(
        model, schedule, editor, state, text_encoder, instructions, valid_frames,
        is_group, mask_modes=(args.mask_mode,), lambda_attn=args.lambda_attn,
        lambda_noise=args.lambda_noise, attn_readout=args.m1_readout, sweeps=sweeps,
        per_step_norm=args.per_step_norm)
    bin_maps = binaries[args.mask_mode]
    for e, b in zip(instructions, bin_maps):
        cells, frames = active_cells(b)
        print(f"  {e!r}: mask {cells} active cells, {frames}/{F} frames")

    m1_src = [flat_corr(m, src_act) for m in m1_maps]
    m2_src = [flat_corr(m, src_act) for m in m2_maps]
    m1_corr, m2_corr = pairwise_corr(m1_maps), pairwise_corr(m2_maps)

    base = os.path.join(args.out_dir, f"{clip_id}_mask_problem")
    plot_mask_problem(clip_id, caption, instructions, targets, m1_maps, m2_maps,
                      bin_maps, src_act, glabels, args.mask_mode, base + ".png")
    plot_mask_quant(clip_id, caption, instructions, m1_corr, m2_corr, m1_src, m2_src,
                    base + "_quant.png")

    print("\n── summary ─────────────────────────────────────────────")
    if len(instructions) > 1:
        print("instruction-invariance (mean off-diagonal r):  "
              f"M1 {mean_off_diagonal(m1_corr):.3f}   M2 {mean_off_diagonal(m2_corr):.3f}"
              "   (→1 = mask ignores the instruction)")
    print(f"mask↔source-motion corr (mean over instr.):    M1 {np.mean(m1_src):.3f}   "
          f"M2 {np.mean(m2_src):.3f}   (→1 = source-dynamics detector)")
    print(f"body-part axis present (G={len(glabels)}): laterality/limb overlay in red."
          if is_group else "flat model (G=1): temporal-only, body-part overlay omitted.")


if __name__ == "__main__":
    main()
