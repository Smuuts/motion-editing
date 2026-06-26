"""
LEDITS++ motion editing entry-point.

Loads a trained (EMA) MotionDiT/GroupDiT checkpoint, runs the three-stage
training-free editor on a single source clip + one or more edit instructions, and
renders a source-vs-edited comparison video plus a mask heatmap for inspection.

By default it uses the **M2-only** mask (mask_mode="m2_only"): no semantic M1 and no
LLM mask. The implicit attention M1 path (mask_mode="attn") and the LLM path
(mask_mode="llm") are one flag away — see src/editing/masking.py.

Examples
--------
    # Edit val clip #0 with one instruction (M2-only mask):
    python src/edit_motion.py \
        --checkpoint runs/exp_group_l/checkpoint_latest \
        --data_root  data/HumanML3D \
        --source 0 \
        --instruction "raise the right arm" \
        --out_dir eval_results/edit_demo

    # Reconstruction check (scale 0 → output must equal the source):
    python src/edit_motion.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the right arm" --scales 0.0

    # Multi-edit:
    python src/edit_motion.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the right arm" --instruction "crouch lower" \
        --scales 5.0 7.5
"""

import os
import sys
import argparse

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.body_groups import GROUP_NAMES
from editing import MotionEditor
from utils.visualise import recover_from_ric, save_comparison_animation

# Reuse the checkpoint loader from the generation script (config + EMA weights).
from sample_model import load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir containing config.json + ema.pt/model.pt.")
    p.add_argument("--data_root", required=True,
                   help="HumanML3D root (needs Mean.npy, Std.npy, val.txt, new_joint_vecs/).")
    p.add_argument("--source", required=True,
                   help="Source clip: an integer index into --split, or a path to a "
                        "raw (T, 263) .npy feature file.")
    p.add_argument("--instruction", action="append", required=True, dest="instructions",
                   help="Edit instruction. Repeat to render one video per instruction "
                        "(variety), or with --compose to combine them into one multi-edit.")
    p.add_argument("--compose", action="store_true",
                   help="Combine all --instruction edits into a single composed edit "
                        "(default: each instruction is rendered as its own video).")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="Guidance scale s_e: one value (applied to all) or one per "
                        "instruction (default 5.0 each).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "llm"],
                   help="Stage-2 mask source (default m2_only). 'none' = no mask: "
                        "guidance everywhere, no inpainting (full-edit ablation).")
    p.add_argument("--lambda_noise", type=float, default=70.0,
                   help="M2 percentile threshold (higher = sparser mask).")
    p.add_argument("--lambda_attn", type=float, default=70.0,
                   help="M1 percentile threshold (only used when --mask_mode attn).")
    p.add_argument("--mask_timesteps", type=int, default=None,
                   help="Use this many evenly-spaced timesteps for mask collection "
                        "(default: all 1000). Small values (e.g. 40) greatly speed up "
                        "the attention modes; the mask is a trajectory average.")
    p.add_argument("--guidance_alpha_floor", type=float, default=0.03,
                   help="Skip edit guidance at steps where sqrt(alpha_cumprod_t) < this "
                        "(prevents x0-space divergence at high noise).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--smooth_sigma", type=float, default=1.5)
    p.add_argument("--out_dir", default="eval_results/edit_demo")
    p.add_argument("--no_ema", action="store_true",
                   help="Load model.pt instead of ema.pt.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_source(source, data_root, split, max_frames):
    """
    Return (raw_feat (T,263) float32, clip_id, length, caption).
    `source` is an index into --split or a path to a raw (T,263) .npy file.
    `caption` is the source motion's original HumanML3D annotation ("" if unavailable).
    """
    if source.endswith(".npy") and os.path.exists(source):
        raw = np.load(source)
        clip_id = os.path.splitext(os.path.basename(source))[0]
    else:
        with open(os.path.join(data_root, f"{split}.txt")) as f:
            ids = [l.strip() for l in f if l.strip()]
        clip_id = ids[int(source)]
        raw = np.load(os.path.join(data_root, "new_joint_vecs", f"{clip_id}.npy"))
    # original prompt: first caption line, stripped of HumanML3D's "#pos#tags" suffix
    caption = ""
    text_path = os.path.join(data_root, "texts", f"{clip_id}.txt")
    if os.path.exists(text_path):
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            caption = lines[0].split("#")[0].strip()
    T = min(len(raw), max_frames)
    return raw[:T].astype(np.float32), clip_id, T, caption


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── load model, schedule, encoder, normalisation stats ──────────────────────
    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    is_group = config.get("feature_mode", "humanml3d") == "group"
    print(f"feature_mode={config.get('feature_mode')}  is_group={is_group}")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))   # (263,)
    std  = np.load(os.path.join(args.data_root, "Std.npy"))    # (263,)
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    # ── source clip → normalised x0 + valid_frames ──────────────────────────────
    raw_feat, clip_id, length, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    F = length
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)  # (1,F,263)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    joints_src = recover_from_ric(raw_feat, joints_num=22)         # (F, 22, 3) shared by all
    if args.smooth_sigma > 0:
        joints_src = gaussian_filter1d(joints_src, sigma=args.smooth_sigma, axis=0)
    print(f"Source: {clip_id}  ({F} frames)  original prompt: {src_caption!r}")

    # ── invert ONCE (inversion depends only on the source, not the instruction) ──
    editor = MotionEditor(model, schedule, device, is_group=is_group)
    print("Stage 1: inversion …")
    state = editor.invert(x0)

    # ── group instructions into render jobs ─────────────────────────────────────
    # compose → all edits in one video; else → one video per instruction (variety).
    edits = args.instructions
    if args.scales is None:
        scale_of = lambda e: 5.0
    elif len(args.scales) == 1:
        scale_of = lambda e: args.scales[0]
    elif len(args.scales) == len(edits):
        smap = dict(zip(edits, args.scales)); scale_of = lambda e: smap[e]
    else:
        raise SystemExit(f"--scales must be 1 value or {len(edits)} (got {len(args.scales)}).")
    jobs = [edits] if args.compose else [[e] for e in edits]

    for job in jobs:
        scales = [scale_of(e) for e in job]
        print(f"\nEdit job: {job}  scales={scales}  mask_mode={args.mask_mode}")
        with torch.no_grad():
            ctxs = [text_encoder.encode([e]) for e in job]
            toks = [text_encoder.token_info(e)[0] for e in job]
        mask_ts = (torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
                   if args.mask_timesteps else None)
        masks = editor.collect_masks(
            state, ctxs, toks, valid_frames,
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode, timesteps=mask_ts,
        )
        x_edit = editor.edit(state, ctxs, masks, scales=scales,
                             guidance_alpha_floor=args.guidance_alpha_floor)   # (1,F,263)
        for i, m in enumerate(masks):
            print(f"  mask[{i}] {job[i]!r}: {int(m['edited'].sum())}/{F} frames, "
                  f"{int(m['m_group'].sum())} active (frame,group) cells")

        # decode (denorm → FK → smooth)
        joints_edit = recover_from_ric(x_edit[0].cpu().numpy() * std + mean, joints_num=22)
        if args.smooth_sigma > 0:
            joints_edit = gaussian_filter1d(joints_edit, sigma=args.smooth_sigma, axis=0)
        per_frame = np.sqrt(((joints_edit - joints_src) ** 2).sum(-1)).mean(-1)   # (F,)

        edit_text = " + ".join(job)
        slug = edit_text[:40].replace(" ", "_").replace("/", "_")
        base = f"{clip_id}_{args.mask_mode}_{slug}"
        out_mp4 = os.path.join(args.out_dir, f"{base}.mp4")
        save_comparison_animation(
            joints_edit, joints_src, per_frame, float(per_frame.mean()), out_mp4,
            title=f"src: {src_caption}   |   edit: {edit_text}",
            clip_id=clip_id,
            gen_label=f"EDIT: {edit_text}",
            gt_label=f"SOURCE: {src_caption}" if src_caption else f"SOURCE [{clip_id}]",
        )
        print(f"Wrote {out_mp4}")
        save_mask_heatmap(masks, job, is_group,
                          os.path.join(args.out_dir, f"{base}_mask.png"))


def save_mask_heatmap(masks, edits, is_group, out_path):
    g_labels = GROUP_NAMES if is_group else ["all"]
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 2.5 * len(g_labels)), 3 * 1), squeeze=False)
    for i, (m, e) in enumerate(zip(masks, edits)):
        mg = m["m_group"].cpu().numpy().T          # (G, F)
        ax = axes[0][i]
        ax.imshow(mg, aspect="auto", cmap="viridis", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_yticks(range(len(g_labels)))
        ax.set_yticklabels(g_labels, fontsize=7)
        ax.set_xlabel("frame")
        ax.set_title(e[:30], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
