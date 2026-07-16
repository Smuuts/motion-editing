"""
Cross-attention head-diversity analysis (mask experiment queue item A.2).

Answers two questions on one (source clip, instruction set) probe, sharing a
single inversion trajectory:

  1. Are the heads collapsed? The research notes' criterion
     (docs/LEDITSpp_Attention_Sink_Research.md §5): if nearly every head pair
     correlates above ~0.98, attention has effectively collapsed and per-head
     readouts are hopeless.
  2. Are there individual grounded heads that the production grand-mean M1
     readout washes out? For each (layer, head) we score alignment (share of
     its top-cells mask on the instruction's expected group) and
     instruction-sensitivity (correlation of its map across instructions —
     low laterality correlation = the head can tell left from right).

Per (layer, head) and instruction it reports:
  align     — percentile-mask share on the expected group (production rule)
  entropy   — normalised entropy of the (F, G) map (1 = uniform, 0 = peaked)
  r_lat     — corr(kick-left map, kick-right map): laterality sensitivity
  r_cat     — mean corr(leg-instruction map, arm-instruction map): category

plus the pairwise head-head correlation summary per instruction (mean
off-diagonal, fraction > 0.9 / > 0.98) and a shortlist of candidate heads
(high alignment on every instruction, low cross-instruction correlation).

Usage (canonical probe):
    python src/analyse_head_diversity.py \
        --checkpoint runs/exp_smplh/checkpoint_latest \
        --data_root  data/HumanML3D/HumanML3D_smplh \
        --source data/HumanML3D/HumanML3D_smplh/new_joint_vecs/012698.npy \
        --mask_timesteps 40
"""

import os
import sys
import json
import argparse
import itertools

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
from editing.masking import _percentile_threshold, semantic_token_subset
from analyse_edit_attention import collect_per_layer_head_attn
from utils.model_io import load_model
from edit_motion import load_source

# (instruction, expected group) probes; first two differ only in laterality,
# the third is the category control — same set as analyse_m1/m2.
DEFAULT_PROBES = [
    ("kick with the left leg",      "left_leg"),
    ("kick with the right leg",     "right_leg"),
    ("raise the right arm higher",  "right_arm"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--lambda_attn", type=float, default=70.0)
    p.add_argument("--mask_timesteps", type=int, default=40)
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--top_k", type=int, default=10,
                   help="How many candidate heads to list/inspect.")
    p.add_argument("--out_dir", default="eval_results/head_diversity")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def head_alignment(attn_fg: np.ndarray, expect_g: int, lambda_attn: float) -> float:
    """Production percentile rule on one head's (F, G) map -> expected-group share."""
    t = torch.from_numpy(attn_fg)
    valid = torch.ones(attn_fg.shape[0], dtype=torch.bool)
    mask = _percentile_threshold(t, valid, lambda_attn)
    active = mask.sum().item()
    return float(mask[:, expect_g].sum().item() / max(active, 1))


def norm_entropy(attn_fg: np.ndarray) -> float:
    p = attn_fg.flatten().astype(np.float64)
    p = p / max(p.sum(), 1e-12)
    h = -(p * np.log(p + 1e-12)).sum()
    return float(h / np.log(len(p)))


def pairwise_corr(maps: np.ndarray) -> np.ndarray:
    """(N, F, G) head maps -> (N, N) Pearson correlation matrix."""
    flat = maps.reshape(maps.shape[0], -1)
    return np.corrcoef(flat)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    raw_feat, clip_id, F, src_caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    print(f"Source: {clip_id} ({F} frames)  caption: {src_caption!r}")

    editor = MotionEditor(model, schedule, device, is_group=True)
    print("Inversion …")
    state = editor.invert(x0)

    timesteps = torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
    G = len(GROUP_NAMES)

    # per-instruction per-(layer, head) maps, semantic-token readout
    maps = {}     # name -> (Lyr, H, F, G)
    expect = {}   # name -> group index
    for instr, grp in DEFAULT_PROBES:
        with torch.no_grad():
            ctx = text_encoder.encode([instr])
        tok_idxs, tok_labels = text_encoder.token_info(instr)
        sem = semantic_token_subset(tok_idxs, tok_labels)
        print(f"Sweeping {len(timesteps)} timesteps for {instr!r} …")
        maps[instr] = collect_per_layer_head_attn(
            model, schedule, state.xs, ctx, sem, F, G, timesteps)
        expect[instr] = GROUP_NAMES.index(grp)
    Lyr, H = next(iter(maps.values())).shape[:2]
    names = list(maps.keys())
    kickL, kickR, arm = names

    # ── 1. head-collapse check: pairwise head correlations per instruction ──
    print(f"\nHead-collapse check ({Lyr}x{H} = {Lyr*H} heads):")
    collapse = {}
    for n in names:
        c = pairwise_corr(maps[n].reshape(Lyr * H, F, G))
        off = c[~np.eye(len(c), dtype=bool)]
        collapse[n] = {"mean_offdiag": float(off.mean()),
                       "frac_gt_0.9": float((off > 0.9).mean()),
                       "frac_gt_0.98": float((off > 0.98).mean())}
        print(f"  {n[:32]:<34} mean r={off.mean():.3f}   "
              f">0.9: {100*(off>0.9).mean():.1f}%   >0.98: {100*(off>0.98).mean():.1f}%")

    # ── 2. per-head scores ──
    rows = []
    for l in range(Lyr):
        for h in range(H):
            aligns = {n: head_alignment(maps[n][l, h], expect[n], args.lambda_attn)
                      for n in names}
            flatL = maps[kickL][l, h].flatten()
            flatR = maps[kickR][l, h].flatten()
            flatA = maps[arm][l, h].flatten()
            r_lat = float(np.corrcoef(flatL, flatR)[0, 1])
            r_cat = float(np.mean([np.corrcoef(flatL, flatA)[0, 1],
                                   np.corrcoef(flatR, flatA)[0, 1]]))
            rows.append({
                "layer": l, "head": h,
                "align_kickL": aligns[kickL], "align_kickR": aligns[kickR],
                "align_arm": aligns[arm],
                "align_min": min(aligns.values()),
                "align_mean": float(np.mean(list(aligns.values()))),
                "entropy": float(np.mean([norm_entropy(maps[n][l, h]) for n in names])),
                "r_lat": r_lat, "r_cat": r_cat,
            })

    # candidate heads: grounded on every probe AND instruction-sensitive.
    # Rank by worst-case alignment, tie-broken by category sensitivity (low r_cat).
    ranked = sorted(rows, key=lambda r: (-r["align_min"], r["r_cat"]))
    print(f"\nTop {args.top_k} heads by worst-case alignment "
          f"(random-mask baseline ≈ group share ≈ {1/G:.2f}):")
    print("layer head  alignL alignR alignArm   r_lat  r_cat  entropy")
    for r in ranked[:args.top_k]:
        print(f"  {r['layer']:>3}  {r['head']:>3}   {r['align_kickL']:.2f}   "
              f"{r['align_kickR']:.2f}    {r['align_arm']:.2f}    "
              f"{r['r_lat']:+.2f}  {r['r_cat']:+.2f}   {r['entropy']:.3f}")

    grand = {n: float(np.mean([r[f"align_{k}"] for r in rows]))
             for n, k in zip(names, ["kickL", "kickR", "arm"])}
    print("\nMean per-head alignment vs production grand-mean readout "
          "(analyse_m1 raw): see summary json.")

    lat_sensitive = [r for r in rows if r["r_lat"] < 0.8]
    print(f"Laterality-sensitive heads (r_lat < 0.8): {len(lat_sensitive)}/{Lyr*H}")
    cat_sensitive = [r for r in rows if r["r_cat"] < 0.5]
    print(f"Category-sensitive heads (r_cat < 0.5):   {len(cat_sensitive)}/{Lyr*H}")

    # ── figures: correlation heatmap (first probe) + alignment/sensitivity scatter ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    c = pairwise_corr(maps[kickL].reshape(Lyr * H, F, G))
    im = axes[0].imshow(c, cmap="viridis", vmin=-1, vmax=1)
    axes[0].set_title(f"head-head corr — '{kickL[:28]}'", fontsize=9)
    axes[0].set_xlabel("head idx (layer-major)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    am = np.array([r["align_mean"] for r in rows])
    rl = np.array([r["r_lat"] for r in rows])
    rc = np.array([r["r_cat"] for r in rows])
    lyr_of = np.array([r["layer"] for r in rows])
    sc = axes[1].scatter(rl, am, c=lyr_of, cmap="plasma", s=22)
    axes[1].axhline(1 / G, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("r_lat (kickL~kickR)  — lower = laterality-sensitive")
    axes[1].set_ylabel("mean alignment")
    axes[1].set_title("alignment vs laterality sensitivity", fontsize=9)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, label="layer")
    sc2 = axes[2].scatter(rc, am, c=lyr_of, cmap="plasma", s=22)
    axes[2].axhline(1 / G, color="grey", ls="--", lw=0.8)
    axes[2].set_xlabel("r_cat (leg~arm)  — lower = category-sensitive")
    axes[2].set_title("alignment vs category sensitivity", fontsize=9)
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, label="layer")
    fig.tight_layout()

    base = os.path.join(args.out_dir, f"{clip_id}_head_diversity")
    fig.savefig(f"{base}.png", dpi=150)
    plt.close(fig)
    np.savez(f"{base}.npz",
             **{f"maps_{i}": maps[n] for i, n in enumerate(names)},
             instructions=np.array(names), group_names=np.array(GROUP_NAMES))
    with open(f"{base}.json", "w") as f:
        json.dump({"clip": clip_id, "caption": src_caption,
                   "probes": DEFAULT_PROBES, "lambda_attn": args.lambda_attn,
                   "collapse": collapse, "mean_alignment_per_head": grand,
                   "n_lat_sensitive": len(lat_sensitive),
                   "n_cat_sensitive": len(cat_sensitive),
                   "heads": ranked}, f, indent=2)
    print(f"\nWrote {base}.{{png,npz,json}}")


if __name__ == "__main__":
    main()
