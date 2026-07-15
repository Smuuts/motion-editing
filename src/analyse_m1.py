"""
M1 (attention mask) readout-variant analysis for one real edit instruction.

The original negative finding (docs/FINDINGS.md "Cross-attention is not body-part
grounded") measured M1 with the raw readout: mean softmax mass on the content
tokens. That readout is structurally sink-dominated — CrossAttention has no
key-padding mask, so for a short instruction the ~L−|words| padding columns
(zero T5 embeddings → zero logits) plus EOS absorb most of each row's mass, and
the content readout is modulated by that sink denominator, not just by semantics.

This script probes the two literature-motivated readout fixes on a single
(source clip, instruction) pair, sharing one inversion trajectory and sweep:

  renorm  — semantic tokens' share of the content-token mass per cell: drops the
            pad/EOS sink from the denominator (Attend-and-Excite re-softmax,
            Chefer et al., SIGGRAPH 2023 — arXiv 2301.13826).
  spatial — each semantic token's map normalised over cells first (its spatial
            profile), then averaged (DAAM aggregation, Tang et al., ACL 2023 —
            arXiv 2210.04885): a token holding little total mass still votes
            with its full spatial distribution.

plus their composition. For each variant it thresholds with the production
percentile rule at --lambda_attn and reports per-group mask shares and an
alignment score, exactly like analyse_m2.py. It also prints the measured sink
mass (share of attention on non-content columns) — the number that motivates
the whole exercise.

Usage (the canonical probe — arms-only source, leg-targeting instruction):
    python src/analyse_m1.py \
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
from model.body_groups import GROUP_NAMES
from editing.inversion import MotionEditor
from editing.masking import (_attn_readout_value, _percentile_threshold,
                             semantic_token_subset)
from utils.model_io import load_model
from edit_motion import load_source

READOUTS = ["raw", "renorm", "spatial", "renorm_spatial"]


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
    p.add_argument("--lambda_attn", type=float, default=70.0,
                   help="M1 percentile threshold (same as edit_motion.py).")
    p.add_argument("--mask_timesteps", type=int, default=40,
                   help="Evenly-spaced timesteps for the sweep (default 40).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--out_dir", default="eval_results/m1_analysis")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def sweep_attn(model, xs, ctx_edit, tok, sem, timesteps, F, G, device):
    """One sweep, accumulating each readout per timestep exactly as
    collect_statistics does. Also accumulates the mean sink mass (attention on
    non-content columns) and the trajectory-mean (N, L) map for inspection.

    Returns (values: dict readout -> (F, G), sink_mass: float, mean_map: (N, L)).
    """
    acc = {r: torch.zeros(F, G, device=device) for r in READOUTS}
    sink_mass, mean_map = 0.0, None
    for t in timesteps:
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        model(x_t, t_b, ctx_edit, store_attn=True)
        layer_maps = model.get_attn_maps()                # list of (1, h, N, L)
        stacked = torch.stack(layer_maps, dim=0).float()  # (Lyr, 1, h, N, L)
        avg = stacked.mean(dim=(0, 1, 2))                 # (N, L)
        for r in READOUTS:
            acc[r] += _attn_readout_value(avg, tok, sem, r).reshape(F, G)
        sink_mass += float(1.0 - avg[:, tok].sum(dim=-1).mean().item())
        mean_map = avg if mean_map is None else mean_map + avg
    n = len(timesteps)
    return ({r: v / n for r, v in acc.items()}, sink_mass / n,
            (mean_map / n).cpu().numpy())


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
        raise SystemExit("analyse_m1.py requires a body-part-grouped checkpoint.")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    raw_feat, clip_id, F, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    print(f"Source: {clip_id} ({F} frames)  caption: {src_caption!r}")

    tok_idxs, tok_labels = text_encoder.token_info(args.instruction)
    sem_idxs = semantic_token_subset(tok_idxs, tok_labels)
    sem_labels = [tok_labels[tok_idxs.index(i)] for i in sem_idxs]
    print(f"Instruction: {args.instruction!r}   expect: {args.expect_groups}")
    print(f"Content tokens: {tok_labels}   semantic subset: {sem_labels}")
    if sem_idxs == tok_idxs:
        print("NOTE: semantic subset == content tokens; 'renorm' will be constant "
              "(uninformative) for this instruction.")

    editor = MotionEditor(model, schedule, device, is_group=True)
    print("Inversion …")
    state = editor.invert(x0)

    with torch.no_grad():
        ctx_edit = text_encoder.encode([args.instruction])
    tok = torch.as_tensor(tok_idxs, device=device, dtype=torch.long)
    sem = torch.as_tensor(sem_idxs, device=device, dtype=torch.long)

    timesteps = torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
    G = len(GROUP_NAMES)
    print(f"Sweeping {len(timesteps)} timesteps …")
    values, sink_mass, mean_map = sweep_attn(
        model, state.xs, ctx_edit, tok, sem, timesteps, F, G, device)

    L_ctx = mean_map.shape[1]
    print(f"\nSink mass: {sink_mass:.3f} of attention is on the {L_ctx - len(tok_idxs)} "
          f"non-content columns (pad/EOS) vs {len(tok_idxs)} content tokens.")

    results, masks = {}, {}
    print(f"\nActive-mask-cell share per group (lambda_attn={args.lambda_attn}):")
    print("variant         align  " + "".join(f"{n[:9]:>10}" for n in GROUP_NAMES))
    for r in READOUTS:
        mask = _percentile_threshold(values[r], valid_frames, args.lambda_attn)
        active = mask.sum().item()
        cell_share = (mask.float().sum(dim=0) / max(active, 1)).cpu().numpy()
        align = float(mask[:, expect_idx].sum().item() / max(active, 1))
        masks[r] = mask.cpu().numpy()
        results[r] = {"alignment": align, "cell_share": cell_share.tolist()}
        print(f"{r:<15} {align:5.2f}  " + "".join(f"{s:>10.3f}" for s in cell_share))

    fig, axes = plt.subplots(4, 2, figsize=(12, 9), squeeze=False)
    for row, r in enumerate(READOUTS):
        for col, (data, title) in enumerate([
                (values[r].cpu().numpy().T, f"{r} — value"),
                (masks[r].T, f"{r} — mask (align {results[r]['alignment']:.2f})")]):
            ax = axes[row][col]
            ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_yticks(range(G)); ax.set_yticklabels(GROUP_NAMES, fontsize=6)
            ax.set_title(title, fontsize=9)
            if row < 3:
                ax.set_xticks([])
    fig.suptitle(f'M1 readouts — "{args.instruction}" on {clip_id}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    slug = args.instruction[:40].replace(" ", "_").replace("/", "_")
    base = os.path.join(args.out_dir, f"{clip_id}_{slug}")
    fig.savefig(f"{base}_m1_readouts.png", dpi=150)
    plt.close(fig)
    np.savez(f"{base}_m1_readouts.npz",
             **{f"val_{r}": values[r].cpu().numpy() for r in READOUTS},
             **{f"mask_{r}": masks[r] for r in READOUTS},
             mean_map=mean_map, tok_idxs=np.array(tok_idxs), sem_idxs=np.array(sem_idxs),
             group_names=np.array(GROUP_NAMES), lambda_attn=args.lambda_attn)
    with open(f"{base}_m1_readouts.json", "w") as f:
        json.dump({"clip": clip_id, "caption": src_caption,
                   "instruction": args.instruction,
                   "expect_groups": args.expect_groups,
                   "content_tokens": tok_labels, "semantic_tokens": sem_labels,
                   "sink_mass": sink_mass, "lambda_attn": args.lambda_attn,
                   "mask_timesteps": args.mask_timesteps,
                   "results": results}, f, indent=2)
    print(f"\nWrote {base}_m1_readouts.{{png,npz,json}}")


if __name__ == "__main__":
    main()
