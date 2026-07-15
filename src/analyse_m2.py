"""
M2 (noise-estimate mask) variant analysis for one real edit instruction.

The original negative finding (see docs/FINDINGS.md "Masking is source-dynamics-
driven") is that ψ = |ε(x_t, c_edit) − ε(x_t, ∅)| follows where the source clip
already moves, so the M2 mask cannot target a body part the source holds still.
This script probes the two literature-motivated fixes on a single (source clip,
instruction) pair, sharing one inversion trajectory and one timestep sweep:

  ref  ∈ {null, source} — contrast against the learned null embedding (original
         LEDITS++/SEGA form) vs. against the SOURCE caption's embedding
         (DiffEdit's "reference text", Couairon et al., ICLR 2023): everything
         both conditionings agree on — the source's own dynamics — cancels.
  norm ∈ {raw, energy}  — optionally divide ψ per group by the source's own
         per-group motion energy (masking._group_motion_energy), discounting
         groups whose large ψ is explained by source movement.

For each of the 4 variants it thresholds ψ with the production percentile rule
(masking._percentile_threshold at --lambda_noise) and reports, per body-part
group: the share of ψ mass and the share of active mask cells, plus an
"alignment" score = fraction of active cells that fall in --expect_groups.

Usage (the canonical probe from the original finding — arms-only source,
leg-targeting instruction):
    python src/analyse_m2.py \
        --checkpoint runs/exp_smplh/checkpoint_latest \
        --data_root  data/HumanML3D/HumanML3D_smplh \
        --source data/HumanML3D/HumanML3D_smplh/new_joint_vecs/012698.npy \
        --instruction "kick with the left leg" \
        --expect_groups left_leg \
        --mask_timesteps 40
"""

import os
import sys
import argparse
import json

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.body_groups import GROUP_NAMES, GROUP_CHANNELS
from editing.inversion import MotionEditor
from editing.masking import (_group_aggregation_matrix, _group_motion_energy,
                             _percentile_threshold)
from utils.model_io import load_model
from edit_motion import load_source


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--source", required=True,
                   help="Source clip: an integer index into --split, or a path to a "
                        "raw feature .npy file.")
    p.add_argument("--instruction", required=True)
    p.add_argument("--expect_groups", nargs="+", required=True,
                   help=f"Group name(s) the instruction *should* target, from: "
                        f"{GROUP_NAMES}. Drives the alignment score.")
    p.add_argument("--lambda_noise", type=float, default=70.0,
                   help="M2 percentile threshold (same as edit_motion.py).")
    p.add_argument("--mask_timesteps", type=int, default=40,
                   help="Evenly-spaced timesteps for the sweep (default 40; "
                        "production uses all 999 but the mask is an average).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--out_dir", default="eval_results/m2_analysis")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def sweep_psi(model, xs, ctx_edit, ctx_src, timesteps, agg_matrix, device):
    """One sweep, three ε passes per timestep → ψ against both references.

    Returns (psi_null, psi_src), both (F, G) trajectory means.
    """
    F = xs.shape[2]
    G = agg_matrix.shape[1]
    psi_null = torch.zeros(F, G, device=device)
    psi_src  = torch.zeros(F, G, device=device)
    for t in timesteps:
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        eps_c = model(x_t, t_b, ctx_edit)
        eps_n = model(x_t, t_b, None)          # learned null embedding
        eps_s = model(x_t, t_b, ctx_src)
        psi_null += (eps_c - eps_n)[0].abs() @ agg_matrix
        psi_src  += (eps_c - eps_s)[0].abs() @ agg_matrix
    return psi_null / len(timesteps), psi_src / len(timesteps)


def summarise(psi_fg, valid_frames, lambda_noise, expect_idx):
    """Per-group ψ-mass share, mask-cell share, and alignment for one variant."""
    mask = _percentile_threshold(psi_fg, valid_frames, lambda_noise)
    psi_share  = (psi_fg.mean(dim=0) / psi_fg.mean(dim=0).sum()).cpu().numpy()
    active     = mask.sum().item()
    cell_share = (mask.float().sum(dim=0) / max(active, 1)).cpu().numpy()
    alignment  = float(mask[:, expect_idx].sum().item() / max(active, 1))
    return mask.cpu().numpy(), psi_share, cell_share, alignment


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    expect_idx = []
    for name in args.expect_groups:
        for part in name.replace(",", " ").split():
            if part not in GROUP_NAMES:
                raise SystemExit(f"unknown group {part!r}; choose from {GROUP_NAMES}")
            expect_idx.append(GROUP_NAMES.index(part))

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    if config.get("feature_mode", "humanml3d") not in ("humanml3d", "smplh", "group"):
        raise SystemExit("analyse_m2.py requires a body-part-grouped checkpoint.")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    raw_feat, clip_id, F, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    if not src_caption:
        raise SystemExit(f"clip {clip_id} has no caption in {args.data_root}/texts/ — "
                          "the source-reference variant needs one.")
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    print(f"Source: {clip_id} ({F} frames)  caption: {src_caption!r}")
    print(f"Instruction: {args.instruction!r}   expect: {args.expect_groups}")

    editor = MotionEditor(model, schedule, device, is_group=True)
    print("Inversion …")
    state = editor.invert(x0)

    with torch.no_grad():
        ctx_edit = text_encoder.encode([args.instruction])
        ctx_src  = text_encoder.encode([src_caption])

    timesteps = torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
    group_channels = editor.group_channels or GROUP_CHANNELS
    agg_matrix = _group_aggregation_matrix(group_channels, device)
    print(f"Sweeping {len(timesteps)} timesteps …")
    psi_null, psi_src = sweep_psi(model, state.xs, ctx_edit, ctx_src,
                                  timesteps, agg_matrix, device)

    energy = _group_motion_energy(state.xs[0][0], valid_frames, agg_matrix)
    print("\nSource per-group motion energy (|Δx0| mean, floored):")
    for g, name in enumerate(GROUP_NAMES):
        print(f"  {name:<10} {energy[g].item():.4f}")

    variants = {
        "null_raw":  psi_null,
        "null_norm": psi_null / energy[None, :],
        "src_raw":   psi_src,
        "src_norm":  psi_src / energy[None, :],
    }

    results, masks = {}, {}
    header = "variant     align  " + "".join(f"{n[:9]:>10}" for n in GROUP_NAMES)
    print(f"\nActive-mask-cell share per group (lambda_noise={args.lambda_noise}):")
    print(header)
    for name, psi in variants.items():
        mask, psi_share, cell_share, align = summarise(
            psi, valid_frames, args.lambda_noise, expect_idx)
        masks[name] = mask
        results[name] = {"alignment": align,
                         "psi_share": psi_share.tolist(),
                         "cell_share": cell_share.tolist()}
        print(f"{name:<11} {align:5.2f}  "
              + "".join(f"{s:>10.3f}" for s in cell_share))

    # figure: ψ heatmap + binary mask per variant
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), squeeze=False)
    for row, (name, psi) in enumerate(variants.items()):
        for col, (data, title) in enumerate([
                (psi.cpu().numpy().T, f"{name} — psi"),
                (masks[name].T, f"{name} — mask (align "
                                f"{results[name]['alignment']:.2f})")]):
            ax = axes[row][col]
            ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_yticks(range(len(GROUP_NAMES)))
            ax.set_yticklabels(GROUP_NAMES, fontsize=6)
            ax.set_title(title, fontsize=9)
            if row < 3:
                ax.set_xticks([])
    fig.suptitle(f'M2 variants — src "{src_caption[:60]}" / edit "{args.instruction}"',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    slug = args.instruction[:40].replace(" ", "_").replace("/", "_")
    base = os.path.join(args.out_dir, f"{clip_id}_{slug}")
    fig.savefig(f"{base}_m2_variants.png", dpi=150)
    plt.close(fig)
    np.savez(f"{base}_m2_variants.npz",
             **{f"psi_{k}": v.cpu().numpy() for k, v in variants.items()},
             **{f"mask_{k}": v for k, v in masks.items()},
             energy=energy.cpu().numpy(), group_names=np.array(GROUP_NAMES),
             lambda_noise=args.lambda_noise)
    with open(f"{base}_m2_variants.json", "w") as f:
        json.dump({"clip": clip_id, "caption": src_caption,
                   "instruction": args.instruction,
                   "expect_groups": args.expect_groups,
                   "lambda_noise": args.lambda_noise,
                   "mask_timesteps": args.mask_timesteps,
                   "energy": energy.cpu().numpy().tolist(),
                   "results": results}, f, indent=2)
    print(f"\nWrote {base}_m2_variants.{{png,npz,json}}")


if __name__ == "__main__":
    main()
