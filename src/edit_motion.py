"""
LEDITS++ motion editing entry-point.

Loads a trained (EMA) checkpoint, runs the three-stage training-free editor on one
source clip + one or more edit instructions, and renders a source-vs-edited comparison
video plus a mask heatmap.

Default mask is **M2-only** (no semantic mask). The implicit attention path
(--mask_mode attn) and the user-supplied path (--mask_mode groups) are one flag away —
see src/editing/masking.py.

Examples
--------
    # Edit val clip #0 with one instruction (M2-only mask):
    python src/edit_motion.py --checkpoint runs/exp_group_l/checkpoint_latest \
        --data_root data/HumanML3D --source 0 --instruction "raise the right arm" \
        --out_dir eval_results/edit_demo

    # Reconstruction check (scale 0 → output must equal the source):
    python src/edit_motion.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the right arm" --scales 0.0

    # Multi-edit composed into one video:
    python src/edit_motion.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the right arm" --instruction "crouch lower" \
        --compose --scales 5.0 7.5

    # User-supplied mask: name the groups each instruction targets.
    python src/edit_motion.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the right arm" --instruction "crouch lower" \
        --mask_mode groups \
        --target_groups "right_arm" --target_groups "left_leg right_leg spine"
"""

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
from scipy.ndimage import gaussian_filter1d

from data.body_part_labels import route_groups
from data.clips import load_source
from editing import MotionEditor
from model.body_groups import GROUP_NAMES, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_columns, resolve_readout_layers
from utils.cli import (
    add_data_args, add_mask_args, add_model_args, parse_group_mask, per_edit_lookup,
    resolve_device,
)
# recover_joints dispatches the feature_mode-correct decode: RIC for humanml3d (263-d),
# SMPL-H forward kinematics for smplh (135-d).
from utils.decode import recover_joints, smplh_body_model
from utils.model_io import load_model
from utils.probe import resolve_sweeps
from utils.visualise import save_comparison_animation, save_mask_heatmap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_data_args(p, source=True, smplh=True)
    add_mask_args(p, mask_timesteps=None)
    p.add_argument("--instruction", action="append", required=True, dest="instructions",
                   help="Edit instruction. Repeat for one video per instruction, or "
                        "with --compose for a single multi-edit.")
    p.add_argument("--compose", action="store_true",
                   help="Combine all --instruction edits into one composed edit.")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="Guidance scale s_e: one value, or one per instruction "
                        "(default 5.0).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "groups", "temporal"],
                   help="Stage-2 mask source (default m2_only). 'none' = guidance "
                        "everywhere, no inpainting. 'groups' = user-supplied M_group "
                        "(see --target_groups), full temporal coverage, no M2 gating.")
    p.add_argument("--target_groups", action="append", default=None,
                   help="--mask_mode groups only: space/comma-separated group names "
                        f"from {GROUP_NAMES}, once per --instruction or once for all. "
                        "A group_mode=joints checkpoint also accepts joint names. "
                        "Pass 'auto' (or omit) to ROUTE the groups from the instruction "
                        "text with the caption parser — no LLM, laterality-correct by "
                        "construction; it errors out rather than guessing when the "
                        "instruction names no body part.")
    p.add_argument("--m2_ref", default="null", choices=["null", "source"],
                   help="Reference for ψ = eps(c_edit) − eps(ref). 'null' = learned "
                        "null embedding (LEDITS++). 'source' = the clip's own caption "
                        "(DiffEdit-style), which cancels source-dynamics-driven ψ.")
    p.add_argument("--m2_group_norm", action="store_true",
                   help="Normalise ψ per group by the source's motion energy, so "
                        "already-moving groups don't monopolise the mask.")
    p.add_argument("--m1_readout", default="raw",
                   choices=["raw", "renorm", "spatial", "renorm_spatial"],
                   help="M1 per-cell readout. 'raw' = content-token mass "
                        "(sink-dominated). 'renorm' = semantic share of it "
                        "(Attend-and-Excite). 'spatial' = per-token spatial profile "
                        "(DAAM). 'renorm_spatial' = both.")
    p.add_argument("--guidance_alpha_floor", type=float, default=None,
                   help="Skip edit guidance where sqrt(alpha_cumprod_t) < this "
                        "(prevents x0-space divergence at high noise). Default: 0.03 "
                        "in eps space, 0 (guide every step) in x0 space, which has no "
                        "1/sqrt(alpha_cumprod_t) amplification to gate.")
    p.add_argument("--smooth_sigma", type=float, default=1.5)
    p.add_argument("--out_dir", default="eval_results/edit_demo")
    return p.parse_args()


def decode(features, feature_mode, smooth_sigma):
    """Raw features → (T, 22, 3) joints, optionally temporally smoothed for rendering."""
    joints = recover_joints(features, feature_mode)
    if smooth_sigma > 0:
        joints = gaussian_filter1d(joints, sigma=smooth_sigma, axis=0)
    return joints


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ── model, schedule, encoder, normalisation stats ───────────────────────────
    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, gnames = resolve_group_context(config)
    print(f"feature_mode={feature_mode}  is_group={is_group}  group_mode={group_mode} "
          f"(G={len(gnames)})")
    # Prime the SMPL-H body model so recover_joints() decodes 135-d features with the
    # configured model (no-op for humanml3d, which uses RIC recovery).
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))   # (D,) = 263 or 135
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    # ── source clip → normalised x0 ─────────────────────────────────────────────
    raw_feat, clip_id, F, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    joints_src = decode(raw_feat, feature_mode, args.smooth_sigma)   # shared by all jobs
    print(f"Source: {clip_id}  ({F} frames)  original prompt: {src_caption!r}")

    # ── invert ONCE (inversion depends only on the source, not the instruction) ──
    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space, psi_readout=args.psi_readout,
                          attn_layers=resolve_readout_layers(config, args.m1_layers))
    print(f"predict_type={schedule.predict_type}  edit_space={editor.edit_space}  "
          f"guidance_alpha_floor={editor.resolve_alpha_floor(args.guidance_alpha_floor):g}"
          + ("  (x0-native ψ/SEGA)" if editor.edit_space == "x0" else ""))
    print("Stage 1: inversion …")
    state = editor.invert(x0)

    ctx_src = None
    if args.m2_ref == "source":
        if src_caption:
            with torch.no_grad():
                ctx_src = text_encoder.encode([src_caption])
        else:
            print("WARNING: --m2_ref source but the clip has no caption; "
                  "falling back to the null reference.")

    edits = args.instructions
    scale_of = per_edit_lookup(args.scales, edits, "scales") or (lambda e: 5.0)
    group_spec_of = None
    if args.mask_mode == "groups":
        if not args.target_groups or args.target_groups == ["auto"]:
            # Option 13's cheap tier: route the instruction to its groups with the same
            # parser the grounding labels use — no LLM, no hand-typed groups, and
            # laterality-correct by construction. Resolves 58.9% of MotionFix test
            # instructions; the rest name no body part and get a clear error rather than
            # a guessed mask (docs/FINDINGS.md "How much of MotionFix a group mask can
            # even address").
            routed = {e: route_groups(e, group_mode) for e in edits}
            for e, g in routed.items():
                print(f"  router {e!r} -> {g or 'NOTHING (no body part named)'}")
            missing = [e for e, g in routed.items() if not g]
            if missing:
                raise SystemExit(
                    f"--mask_mode groups --target_groups auto: the router found no body "
                    f"part in {missing}. Name the groups explicitly with "
                    f"--target_groups, or use --mask_mode temporal, which is the right "
                    f"mask for manner/timing edits.")
            group_spec_of = lambda e: " ".join(routed[e])
        else:
            group_spec_of = per_edit_lookup(args.target_groups, edits, "target_groups")

    mask_ts, m1_ts, m2_ts = resolve_sweeps(args.mask_timesteps, schedule.T,
                                           args.m1_window, args.m2_window)

    # compose → all edits in one video; else → one video per instruction (variety).
    for job in ([edits] if args.compose else [[e] for e in edits]):
        scales = [scale_of(e) for e in job]
        print(f"\nEdit job: {job}  scales={scales}  mask_mode={args.mask_mode}")
        with torch.no_grad():
            # One batched encode() for the whole job: same per-text embeddings (encoder
            # attention is intra-sequence only), fewer forward passes when composing.
            ctxs = list(text_encoder.encode(job).split(1, dim=0))
            # (token_idxs, semantic_idxs) per edit — "content" is the historical read;
            # "span"/auto restricts M1 to the columns the grounding loss supervised.
            cols = [resolve_readout_columns(e, text_encoder, config, args.m1_columns,
                                            group_mode) for e in job]
        if args.mask_mode in ("attn", "m1_only"):
            print("  M1 columns: "
                  + ", ".join(f"{e!r}->{c[2]}" for e, c in zip(job, cols)))

        group_masks = None
        if group_spec_of is not None:
            group_masks = [parse_group_mask(group_spec_of(e), is_group, group_mode)
                           for e in job]
            for e, gm in zip(job, group_masks):
                print(f"  target_groups {e!r}: "
                      f"{[gnames[g] for g in gm.nonzero().flatten().tolist()]}")

        masks = editor.collect_masks(
            state, ctxs, [c[0] for c in cols], valid_frames,
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode, llm_group_masks=group_masks, timesteps=mask_ts,
            context_source=ctx_src, m2_group_norm=args.m2_group_norm,
            attn_readout=args.m1_readout,
            semantic_idxs_per_edit=[c[1] for c in cols],
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=args.per_step_norm,
        )
        x_edit = editor.edit(state, ctxs, masks, scales=scales,
                             guidance_alpha_floor=args.guidance_alpha_floor)  # (1,F,D)
        for i, m in enumerate(masks):
            print(f"  mask[{i}] {job[i]!r}: {int(m['edited'].sum())}/{F} frames, "
                  f"{int(m['m_group'].sum())} active (frame,group) cells")

        joints_edit = decode(x_edit[0].cpu().numpy() * std + mean, feature_mode,
                             args.smooth_sigma)
        per_frame = np.sqrt(((joints_edit - joints_src) ** 2).sum(-1)).mean(-1)   # (F,)
        # A frame is "edited" if ANY of the job's edits touches it.
        edit_mask = np.logical_or.reduce(
            [m["edited"].cpu().numpy() for m in masks]) if masks else None

        edit_text = " + ".join(job)
        base = f"{clip_id}_{args.mask_mode}_{edit_text[:40].replace(' ', '_').replace('/', '_')}"
        save_comparison_animation(
            joints_edit, joints_src, per_frame, float(per_frame.mean()),
            os.path.join(args.out_dir, f"{base}.mp4"),
            title=f"src: {src_caption}   |   edit: {edit_text}",
            clip_id=clip_id,
            gen_label=f"EDIT: {edit_text}",
            gt_label=f"SOURCE: {src_caption}" if src_caption else f"SOURCE [{clip_id}]",
            edit_mask=edit_mask,
        )
        save_mask_heatmap(masks, job, gnames if is_group else ["all"],
                          os.path.join(args.out_dir, f"{base}_mask.png"))


if __name__ == "__main__":
    main()
