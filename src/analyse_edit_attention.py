"""
Per-(layer, head) M1 mask analysis for one real edit instruction.

masking.collect_statistics() (the production M1 path used by edit_motion.py)
averages cross-attention over ALL layers and heads before thresholding into a
mask — a single grand mean. This script keeps the layer/head axes so you can
see whether body-part grounding for a *specific* instruction is concentrated in
a few heads that the grand mean washes out, using the same source clip +
inversion trajectory the real editor would build for it (not a synthetic prompt
sweep on random noise, unlike analyse_attention.py).

For each (layer, head) it thresholds that head's own attention (averaged over
the swept inversion timesteps) at --lambda_attn — the identical percentile rule
masking.build_mask uses for M1 — giving a binary (F, G) mask per head. It also
computes, per layer, the mask obtained by first averaging attention over heads
and then thresholding (the "if M1 could only keep per-layer resolution" view).

Usage:
    python src/analyse_edit_attention.py \
        --checkpoint runs/exp_group_l/checkpoint_latest \
        --data_root  data/HumanML3D/HumanML3D \
        --source 0 \
        --instruction "raise the right arm" \
        --out_dir eval_results/edit_attention_lh

    # Faster (subsample the inversion trajectory instead of all 999 steps):
    python src/analyse_edit_attention.py --checkpoint ... --data_root ... \
        --source 0 --instruction "raise the right arm" --mask_timesteps 40
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

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.body_groups import GROUP_NAMES
from editing.inversion import MotionEditor
from editing.masking import _percentile_threshold
from utils.model_io import load_model
from edit_motion import load_source


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
    p.add_argument("--instruction", required=True,
                   help="Edit instruction to analyse.")
    p.add_argument("--lambda_attn", type=float, default=70.0,
                   help="M1 percentile threshold (higher = sparser mask); same "
                        "meaning as edit_motion.py's --lambda_attn.")
    p.add_argument("--mask_timesteps", type=int, default=None,
                   help="Use this many evenly-spaced timesteps for the attention "
                        "sweep (default: all 999). Small values (e.g. 40) greatly "
                        "speed this up — the maps are a trajectory average either way.")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--out_dir", default="eval_results/edit_attention_lh")
    p.add_argument("--no_ema", action="store_true",
                   help="Load model.pt instead of ema.pt.")
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def collect_per_layer_head_attn(model, schedule, xs, context, token_idxs, F, G, timesteps):
    """
    Sweep `timesteps` over the stored inversion trajectory `xs` and accumulate
    mean cross-attention over content tokens, keeping the (layer, head) axes.

    xs           : (T, 1, F, D) stored inversion samples x_t (see editing/inversion.py)
    context      : (1, L, dim) embedding of the instruction
    token_idxs   : content-token column indices (from text_encoder.token_info)
    Returns      : (Lyr, H, F, G) numpy float32 — mean attention over content
                   tokens and the swept timesteps, per (layer, head, frame, group).
    """
    device = context.device
    tok = torch.as_tensor(token_idxs, device=device, dtype=torch.long)
    acc = None
    for t in timesteps:
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        model(x_t, t_b, context, store_attn=True)
        layer_maps = model.get_attn_maps()                     # list of (1, H, N, L_text)
        stacked = torch.stack(layer_maps, dim=0).float()[:, 0]  # (Lyr, H, N, L_text)
        avg = stacked[..., tok].mean(dim=-1)                    # (Lyr, H, N)
        acc = avg if acc is None else acc + avg
    attn_lhn = (acc / len(timesteps)).reshape(acc.shape[0], acc.shape[1], F, G)
    return attn_lhn.cpu().numpy()


def plot_per_head_masks(mask_lh, group_names, instruction, save_path):
    """Grid of L rows x H cols, each panel the binary (G, F) mask for that head."""
    Lyr, H = mask_lh.shape[:2]
    G = len(group_names)
    fig, axes = plt.subplots(Lyr, H, figsize=(max(8, 0.9 * H), max(6, 0.55 * Lyr)),
                             squeeze=False)
    for l in range(Lyr):
        for h in range(H):
            ax = axes[l][h]
            ax.imshow(mask_lh[l, h].T, aspect="auto", cmap="viridis",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=7, rotation=0, ha="right", va="center")
            if l == 0:
                ax.set_title(f"h{h}", fontsize=7)
            if l == Lyr - 1 and h == 0:
                ax.set_yticks(range(G)); ax.set_yticklabels(group_names, fontsize=5)
    fig.suptitle(f'Per-(layer, head) M1 mask — "{instruction}"', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_layer_avg_masks(mask_layer_avg, group_names, instruction, save_path):
    """One row per layer: the mask from head-averaged attention (G, F)."""
    Lyr = mask_layer_avg.shape[0]
    G = len(group_names)
    fig, axes = plt.subplots(Lyr, 1, figsize=(10, max(6, 0.8 * Lyr)), squeeze=False)
    for l in range(Lyr):
        ax = axes[l][0]
        ax.imshow(mask_layer_avg[l].T, aspect="auto", cmap="viridis",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(G)); ax.set_yticklabels(group_names, fontsize=6)
        ax.set_ylabel(f"L{l}", fontsize=8)
        if l < Lyr - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("frame")
    fig.suptitle(f'Per-layer head-averaged M1 mask — "{instruction}"', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    is_group = config.get("feature_mode", "humanml3d") in ("humanml3d", "smplh", "group")
    if not is_group:
        raise SystemExit("analyse_edit_attention.py requires a body-part-grouped "
                          "checkpoint (feature_mode humanml3d/smplh/group).")
    print(f"feature_mode={config.get('feature_mode')}  layers={config.get('num_layers')}  "
          f"heads={config.get('num_heads')}")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    raw_feat, clip_id, length, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    F = length
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    print(f"Source: {clip_id}  ({F} frames)  original prompt: {src_caption!r}")
    print(f"Instruction: {args.instruction!r}")

    editor = MotionEditor(model, schedule, device, is_group=is_group)
    print("Inversion …")
    state = editor.invert(x0)

    with torch.no_grad():
        context = text_encoder.encode([args.instruction])
        token_idxs, token_labels = text_encoder.token_info(args.instruction)
    print(f"Content tokens: {token_labels}")

    timesteps = (torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
                 if args.mask_timesteps else list(range(1, schedule.T)))
    print(f"Sweeping {len(timesteps)} timesteps …")

    G = len(GROUP_NAMES)
    attn_lhfg = collect_per_layer_head_attn(
        model, schedule, state.xs, context, token_idxs, F, G, timesteps)
    Lyr, H = attn_lhfg.shape[:2]
    print(f"Collected attention: {Lyr} layers x {H} heads x {F} frames x {G} groups")

    # per-(layer, head) masks: threshold each head's own attention independently.
    valid_np = valid_frames.cpu()
    mask_lh = np.stack([
        np.stack([
            _percentile_threshold(torch.from_numpy(attn_lhfg[l, h]), valid_np,
                                  args.lambda_attn).numpy()
            for h in range(H)
        ])
        for l in range(Lyr)
    ])  # (Lyr, H, F, G) bool

    # per-layer masks: average over heads first, then threshold.
    attn_layer_avg = attn_lhfg.mean(axis=1)   # (Lyr, F, G)
    mask_layer_avg = np.stack([
        _percentile_threshold(torch.from_numpy(attn_layer_avg[l]), valid_np,
                              args.lambda_attn).numpy()
        for l in range(Lyr)
    ])  # (Lyr, F, G) bool

    slug = args.instruction[:40].replace(" ", "_").replace("/", "_")
    base = f"{clip_id}_{slug}"

    per_head_path = os.path.join(args.out_dir, f"{base}_per_head_masks.png")
    plot_per_head_masks(mask_lh, GROUP_NAMES, args.instruction, per_head_path)
    print(f"Wrote {per_head_path}")

    per_layer_path = os.path.join(args.out_dir, f"{base}_per_layer_avg_masks.png")
    plot_per_layer_avg_masks(mask_layer_avg, GROUP_NAMES, args.instruction, per_layer_path)
    print(f"Wrote {per_layer_path}")

    npz_path = os.path.join(args.out_dir, f"{base}_attn.npz")
    np.savez(npz_path, attn_lhfg=attn_lhfg, mask_lh=mask_lh, mask_layer_avg=mask_layer_avg,
             group_names=np.array(GROUP_NAMES), lambda_attn=args.lambda_attn)
    print(f"Wrote {npz_path}")

    # quick per-(layer, head) summary: active cell count + peak non-root group
    print("\nlayer  head  active_cells  peak_group")
    for l in range(Lyr):
        for h in range(H):
            active = int(mask_lh[l, h].sum())
            peak_g = int(attn_lhfg[l, h, :, 1:].mean(axis=0).argmax()) + 1
            print(f"  {l:>3}  {h:>4}  {active:>12}  {GROUP_NAMES[peak_g]}")


if __name__ == "__main__":
    main()
