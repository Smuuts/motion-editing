"""
Visualise how the implicit LEDITS++ masks change ACROSS THE INVERSION — t=1 to t=T.

The noise-level twin of `visualise_mask_problem.py`. That figure varies the INSTRUCTION
with the timestep sweep held fixed; this one varies the NOISE LEVEL with the instruction
held fixed, and answers the question its rows are averaged over:

  a mask is accumulated over the whole trajectory and then reported as ONE map — what is
  that average actually made of?

Two things the project already believes are statements about these rows, and this is the
figure that shows them rather than quoting them:
  · M1 (cross-attention) sharpens toward HIGH noise — category invariance 0.899 at t<250
    vs 0.746 at t≥750, and the best alignment ever measured came from `--m1_window 750
    999`. At t≈900 there is no clip left in x_t to detect, so the caption is the only
    signal present.
  · M2 (ψ) in ε space is a LOW-noise read-out — ψ_ε = √SNR_t·ψ_x0 puts ~5.6 % of its
    weight at t≥500 — while ψ in x0 space weights every swept t by its own clean-signal
    displacement. The two therefore do NOT average the same trajectory even off one sweep.

Layout: the instruction-independent source-motion reference on top (the same reference
`plot_mask_problem` uses), then one row per noise band with M1 on the left and M2 on the
right, then two summary curves — where each mask's magnitude sits along the inversion,
and how its correlation with the source clip's own motion evolves with noise.

It reuses the real editing stack, so it works for any checkpoint the editor works for:
humanml3d (263-d) or smplh (135-d), GroupDiT or GroupCLR U-Net, either token axis, and
the legacy flat MotionDiT (G=1, where the body-part overlay is omitted). ONE inversion is
shared by every band, so the rows differ only by noise level.

Usage
-----
    python src/visualise_mask_noise_bands.py \
        --checkpoint runs/exp_smplh_unet/checkpoint_latest \
        --data_root  data/HumanML3D_smplh \
        --source 0 --out_dir eval_results/mask_noise_bands

    # Several instructions (one figure each), even bands in "how much clip is left":
    python src/visualise_mask_noise_bands.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the left arm" --instruction "kick with the right leg" \
        --bands alpha:8

    # Which bands DOMINATE the full-trajectory sum, rather than what each one looks like:
    python src/visualise_mask_noise_bands.py ... --scale shared
"""

import os
import argparse
import re

import numpy as np
import torch

import matplotlib
from utils.logger import get_logger

log = get_logger(__name__)
matplotlib.use("Agg")

from analysis.instructions import resolve_targets
from analysis.mask_probe import collect_noise_band_maps
from data.clips import load_source
from editing import MotionEditor
from model.body_groups import GROUP_NAMES, group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_layers
from utils.cli import (add_data_args, add_logging_args, add_mask_args, add_model_args,
                       configure_logging, resolve_device)
from utils.decode import smplh_body_model
from utils.model_io import load_model
from utils.probe import band_labels, flat_corr, resolve_bands, source_activity
from utils.visualise import plot_mask_noise_bands

DEFAULT_INSTRUCTION = "raise the left arm"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_data_args(p, source=True, smplh=True)
    # No binary mask is built here (nothing to threshold — see collect_noise_band_maps),
    # and the sweep is the band grid, so the --lambda_*/--m1_select and --m*_window flags
    # would all be dead. --mask_timesteps survives with a band-local meaning.
    add_mask_args(p, mask_timesteps=16, thresholds=False, windows=False)
    p.add_argument("--instruction", action="append", dest="instructions", default=None,
                   help=f"Edit instruction; repeat for one figure each (default "
                        f"{DEFAULT_INSTRUCTION!r}).")
    p.add_argument("--target_groups", action="append", default=None,
                   help="Expected group(s) for the red overlay, one per --instruction "
                        f"(names from {GROUP_NAMES}). Default: guessed from the text.")
    p.add_argument("--bands", default="default",
                   help="Noise bands: 'default' (uneven on purpose — resolution at both "
                        "ends, where the two masks actually live), 'linear:N', 'log:N', "
                        "'alpha:N' (even in √ᾱ_t, i.e. in how much clean signal is left) "
                        "or explicit edges '1,50,250,750,999'.")
    p.add_argument("--m1_readout", default="raw",
                   choices=["raw", "renorm", "spatial", "renorm_spatial"],
                   help="M1 per-cell attention readout (see masking.collect_statistics).")
    p.add_argument("--per_step_norm", action="store_true",
                   help="Weight every timestep INSIDE a band equally instead of by its "
                        "magnitude. Only affects within-band mixing — the bands "
                        "themselves are already an explicit weighting of the trajectory.")
    p.add_argument("--scale", default="band", choices=["band", "shared"],
                   help="'band' (default): every panel on its own scale — compare the "
                        "mask's SHAPE across noise levels. 'shared': one scale per "
                        "column — compare brightness, i.e. which bands dominate the "
                        "full-trajectory sum.")
    p.add_argument("--psi_magma", action="store_true",
                   help="Render M2 the way visualise_mask_problem.py does — 0-anchored "
                        "magma — so the two figures are visually comparable. Default is "
                        "a diverging red/blue scale (red = the edit adds motion, blue = "
                        "it stills the source): a narrow band's ψ is 60-79 % negative at "
                        "low/mid noise, and magma renders every one of those cells "
                        "identical to zero.")
    p.add_argument("--no_summary", action="store_true",
                   help="Drop the two summary curves under the grid.")
    p.add_argument("--out_dir", default="eval_results/mask_noise_bands")
    add_logging_args(p)
    return configure_logging(p.parse_args())


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "edit"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    log.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    log.info(f"feature_mode={feature_mode}  arch={config.get('arch', 'dit')}  "
             f"is_group={is_group}  group_mode={group_mode}")
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    instructions = args.instructions or [DEFAULT_INSTRUCTION]
    # No body-part axis ⇒ the laterality/limb overlay is meaningless.
    targets = (resolve_targets(instructions, args.target_groups, group_mode)
               if is_group else [[] for _ in instructions])

    raw_feat, clip_id, F, caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    log.info(f"Source: {clip_id}  ({F} frames)   prompt: {caption!r}\n"
             f"instructions: {instructions}")

    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space, psi_readout=args.psi_readout,
                          attn_layers=resolve_readout_layers(config, args.m1_layers))
    log.info(f"predict_type={schedule.predict_type}  edit_space={editor.edit_space} "
             f"(ψ read as {'|x̂0_c − x̂0_ref|' if editor.edit_space == 'x0' else '|ε_c − ε_ref|'})")
    glabels = group_names(group_mode) if is_group else ["all"]
    src_act = source_activity(x0, editor.group_channels, is_group)   # (F, G) reference

    sqrt_alpha = schedule.sqrt_alphas_cumprod.cpu().numpy()
    bands = resolve_bands(args.bands, schedule.T, sqrt_alpha)
    labels = band_labels(bands, sqrt_alpha)
    log.info(f"{len(bands)} bands ({args.bands}), {args.mask_timesteps} timesteps each "
             f"({len(bands) * args.mask_timesteps} of {schedule.T} in total): "
             + "  ".join(f"[{lo},{hi}]" for lo, hi in bands))

    log.info("Stage 1: inversion (one, shared by every band) …")
    state = editor.invert(x0, seed=args.seed)

    log.section("bands")
    for instr, tgt in zip(instructions, targets):
        columns = {}
        m1_bands, m2_bands = collect_noise_band_maps(
            model, schedule, editor, state, text_encoder, instr, valid_frames, is_group,
            bands, band_timesteps=args.mask_timesteps, attn_readout=args.m1_readout,
            per_step_norm=args.per_step_norm, column_mode=args.m1_columns,
            config=config, group_mode=group_mode, columns_out=columns)

        mode, cols = columns[instr]
        log.info(f"{instr!r}  (expect: {', '.join(tgt) or '—'})   M1 columns "
                 f"({args.m1_columns} -> {mode}): {cols}   ψ: {editor.psi_readout}")
        for (lo, hi), m1, m2 in zip(bands, m1_bands, m2_bands):
            log.info(f"  t {lo:>4}–{hi:<4} √ᾱ {sqrt_alpha[lo]:.2f}→{sqrt_alpha[hi]:.2f}  "
                     f"|M1| {np.abs(m1).mean():.3e} r_src {flat_corr(m1, src_act):+.3f}   "
                     f"|M2| {np.abs(m2).mean():.3e} r_src {flat_corr(m2, src_act):+.3f}")

        out = os.path.join(args.out_dir, f"{clip_id}_noise_bands_{slug(instr)}.png")
        plot_mask_noise_bands(clip_id, caption, instr, tgt, labels, m1_bands, m2_bands,
                              src_act, glabels, out, psi_readout=editor.psi_readout,
                              scale=args.scale, summary=not args.no_summary,
                              psi_signed=not args.psi_magma,
                              edit_space=editor.edit_space)

    log.section("summary")
    log.info("Read a COLUMN top-to-bottom: that is one mask at successive stages of the "
             "inversion. Under --scale band the panels are individually normalised, so "
             "compare shapes and read magnitudes off the corner tags / summary curve.")


if __name__ == "__main__":
    main()
