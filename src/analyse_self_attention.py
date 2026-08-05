"""
Probe the SELF-attention pathway for emergent (frame, body-part) structure — Option 7
of docs/AttentionGrounding_Options.md, Family A of docs/ImplicitMask_Research.md.

Why this probe exists
---------------------
Every mask probe so far read *cross*-attention (M1) or the noise contrast ψ (M2), and
both are measured to be instruction-invariant and source-dynamics-driven
(docs/FINDINGS.md). Self-attention is the one pathway never examined. DiffSeg
("Diffuse, Attend, and Segment", arXiv 2308.12469) segments images from
**self-attention only, no text and no labels**, and MotionCLR (arXiv 2410.18977) shows
motion self-attention carries repetition/segment structure — so there is plausibly a
signal here. A positive result buys a training-free, data-driven spatial partition of
the clip to intersect with an instruction-driven selector; it cannot buy instruction
selectivity, because self-attention is text-free. A clean negative is equally useful:
it closes "is there ANY usable emergent structure in this backbone without a retrain?".

What is measured (implementations + rationale in analysis/self_attention.py)
  1. Affinity structure — group×group C (body-part structured? left vs right?) and
     frame×frame R (temporally blocked?), each against its random baseline.
  2. DiffSeg segmentation — KL-threshold merging over a swept tau, scored by NMI
     against the body-part axis and a coarse temporal binning, minus a shuffled floor.
  3. Laterality — on the affinity (self−mirror AND mirror−other) and on the
     segmentation (do the two sides land in different segments).
  4. Text-invariance — self-attention is text-free by construction only in layer 0;
     later layers see the cross-attention residual. Running contrasting instructions
     measures whether any instruction signal survives, i.e. whether Family A's premise
     holds.

Outputs per source clip in --out_dir: `<clip>_selfattn_structure.png`,
`<clip>_selfattn_segments.png`, `<clip>_selfattn.json` (every number in both figures).

Usage
-----
    python src/analyse_self_attention.py --checkpoint runs/exp_smplh/checkpoint_latest \
        --data_root data/HumanML3D_smplh --source 0 --out_dir eval_results/self_attention

    # Restrict the timestep window (cf. PROGRESS item B.7c):
    python src/analyse_self_attention.py --checkpoint ... --data_root ... --source 0 \
        --t_min 250 --t_max 750 --mask_timesteps 40

Notes
-----
* Backbone-agnostic like visualise_mask_problem.py: it reuses the real editing stack,
  so it runs on any checkpoint the editor runs on. For the GroupCLR U-Net only blocks
  already at F·G tokens are used; the number skipped is reported (see read_maps).
* Memory: capture is head-meaned (model/layers.py). Peak is one (N, N) float32 map per
  block, N = F·G — trim with --max_frames on a 22-token axis.
"""

import os
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from analysis import self_attention as sa
from analysis.instructions import DEFAULT_INSTRUCTIONS
from data.clips import load_source
from editing import MotionEditor
from model.body_groups import group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from utils.cli import add_data_args, add_model_args, resolve_device
from utils.decode import smplh_body_model
from utils.model_io import load_model
from utils.probe import flat_corr, source_activity
from utils.visualise import mean_off_diagonal
from utils.visualise.attention import plot_segments, plot_structure


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_data_args(p, source=True, smplh=True)
    p.add_argument("--instruction", action="append", dest="instructions", default=None,
                   help="Instruction for the text-invariance check; repeat for a "
                        f"contrasting set (default {DEFAULT_INSTRUCTIONS}). The primary "
                        "analysis always uses the null (text-free) context.")
    p.add_argument("--mask_timesteps", type=int, default=40,
                   help="Evenly-spaced inversion timesteps swept and averaged.")
    p.add_argument("--t_min", type=int, default=1, help="Lowest timestep swept.")
    p.add_argument("--t_max", type=int, default=None,
                   help="Highest timestep swept (default T-1). Motion-Adapter reports "
                        "attention stops aligning above t≈750; --t_max 750 tests that.")
    p.add_argument("--layers", default=None,
                   help="Comma-separated block indices to read (default: all).")
    p.add_argument("--anchor_stride", type=int, default=4,
                   help="DiffSeg anchors: every Nth frame, all groups.")
    p.add_argument("--merge_iters", type=int, default=8, help="DiffSeg merge rounds.")
    p.add_argument("--kl_tau", type=float, default=None,
                   help="Fixed KL merge threshold. Default: sweep percentiles of the "
                        "anchors' own pairwise KL and report the best operating point.")
    p.add_argument("--time_bins", type=int, default=8,
                   help="Coarse temporal bins for the temporal-NMI reference.")
    p.add_argument("--out_dir", default="eval_results/self_attention")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the (stochastic) inversion and the shuffled NMI baselines.")
    return p.parse_args()


def text_invariance(model, state, timesteps, N, keep_layers, text_encoder,
                    instructions, A_null):
    """(instruction×instruction correlation matrix, r of each vs the null context).

    Near-identical maps confirm the text-free premise Family A rests on.
    """
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
    per_instr = []
    for e, ctx in zip(instructions, ctxs):
        A_e, _, _ = sa.aggregate(model, state.xs, ctx, timesteps, N, keep_layers)
        per_instr.append(A_e)
        print(f"  {e!r}: r vs null context = {flat_corr(A_e, A_null):.4f}")
    corr = np.array([[flat_corr(a, b) for b in per_instr] for a in per_instr])
    return corr, [flat_corr(a, A_null) for a in per_instr]


def print_summary(metrics, F, G, n_seg, tau, n_instructions):
    print("\n── summary ─────────────────────────────────────────────")
    print(f"group-affinity diagonality : {metrics['group_diagonality']:.3f}   "
          f"(random {1/G:.3f})   → is self-attention body-part structured?")
    print(f"frame-affinity diagonality : {metrics['frame_diagonality']:.4f}   "
          f"(random {1/F:.4f})  → is it temporally blocked?")
    print(f"incoming attn vs source |Δx0|: "
          f"{metrics['corr_incoming_attention_vs_source_motion']:+.3f}   "
          f"(→1 ⇒ another source-dynamics detector)")
    print(f"DiffSeg segments           : {n_seg}  (tau {tau:.4f})")
    for axis in ("group", "time"):
        label = "body-part axis" if axis == "group" else "time bins"
        print(f"NMI vs {label:<19}: {metrics[f'nmi_{axis}']:.3f}   "
              f"(shuffled {metrics[f'nmi_{axis}_shuffled']:.3f}  → gap "
              f"{metrics[f'nmi_{axis}'] - metrics[f'nmi_{axis}_shuffled']:+.3f}; "
              f"best over sweep {metrics[f'best_nmi_{axis}_gap_over_sweep']:+.3f})")
    for k, v in metrics["affinity_laterality"].items():
        print(f"laterality [{k:>4}]           : self−mirror {v['self_vs_mirror']:+.3f} "
              f"(>0 ⇒ sides distinguished)   mirror−other {v['mirror_vs_other']:+.3f} "
              f"(>0 ⇒ sides LINKED, i.e. bilateral symmetry)")
    if n_instructions > 1:
        print(f"instruction-invariance r   : "
              f"{metrics['instruction_invariance_mean_off_diag_r']:.4f}   "
              f"(→1 = text-free, as Family A assumes)")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Edit-friendly DDPM inversion draws fresh noise per step, so the whole probe is
    # stochastic; seed torch too or the segment count drifts between runs.
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    arch = config.get("arch", "dit")
    print(f"feature_mode={feature_mode}  arch={arch}  is_group={is_group}  "
          f"group_mode={group_mode}")
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    raw_feat, clip_id, F, caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    print(f"Source: {clip_id}  ({F} frames)   prompt: {caption!r}")

    editor = MotionEditor(model, schedule, device, is_group=is_group)
    glabels = group_names(group_mode) if is_group else ["all"]
    G = len(glabels)
    N = F * G
    src_act = source_activity(x0, editor.group_channels, is_group)

    keep_layers = {int(s) for s in args.layers.split(",")} if args.layers else None
    t_max = args.t_max if args.t_max is not None else schedule.T - 1
    timesteps = torch.linspace(args.t_min, t_max, args.mask_timesteps).long().tolist()
    print(f"Sweeping {len(timesteps)} timesteps in [{args.t_min}, {t_max}], "
          f"N = F*G = {N}")

    print("Stage 1: inversion …")
    state = editor.invert(x0)

    # Primary readout: NULL context — self-attention with no text at all, the honest
    # "text-free structure" measurement Family A is about.
    print("Aggregating self-attention (null context) …")
    try:
        A_null, per_layer, skipped = sa.aggregate(
            model, state.xs, None, timesteps, N, keep_layers)
    except RuntimeError as e:
        raise SystemExit(str(e))
    if skipped:
        print(f"  note: skipped {skipped} block(s) at reduced temporal resolution "
              f"(U-Net levels); using the {per_layer.shape[0]} full-resolution block(s).")

    C = sa.group_affinity(A_null, F, G)
    R = sa.frame_affinity(A_null, F, G)
    layer_diag = [sa.diagonality(sa.group_affinity(per_layer[i], F, G))
                  for i in range(per_layer.shape[0])]

    group_ref = sa.group_reference(F, G)
    time_ref = sa.time_reference(F, G, args.time_bins)

    print("DiffSeg merging …")
    if args.kl_tau is not None:
        labels, n_seg = sa.diffseg(A_null, F, G, args.anchor_stride, args.merge_iters,
                                   args.kl_tau)
        sweep, kl_stats, tau = [], {}, args.kl_tau
    else:
        sweep, best, kl_stats = sa.diffseg_tau_sweep(
            A_null, F, G, args.anchor_stride, args.merge_iters, group_ref, time_ref, rng)
        labels, n_seg, tau = best["_labels"], best["n_segments"], best["tau"]
        print("  tau sweep: " + ", ".join(
            f"p{r['percentile']}→{r['n_segments']}seg/gap{r['nmi_group_gap']:+.2f}"
            for r in sweep))
        print(f"  operating point (max body-part gap): tau {tau:.4f} → {n_seg} "
              f"segments; anchor KL median {kl_stats['kl_median']:.3f}")
    labels_fg = labels.reshape(F, G)

    metrics = {
        "clip": clip_id, "caption": caption, "frames": F, "groups": G,
        "arch": arch, "feature_mode": feature_mode, "group_mode": group_mode,
        "timesteps": [args.t_min, t_max], "n_timesteps": len(timesteps),
        "blocks_used": int(per_layer.shape[0]), "blocks_skipped": int(skipped),
        "group_diagonality": sa.diagonality(C), "group_diagonality_random": 1.0 / G,
        "frame_diagonality": sa.diagonality(R), "frame_diagonality_random": 1.0 / F,
        "per_layer_group_diagonality": [float(v) for v in layer_diag],
        "affinity_laterality": sa.affinity_laterality(C, glabels) if is_group else {},
        "n_segments": int(n_seg), "kl_tau": tau,
        "anchor_kl_stats": kl_stats,
        "tau_sweep": [{k: v for k, v in r.items() if k != "_labels"} for r in sweep],
        # Granularity-fair headline: NMI minus the same-granularity shuffled baseline,
        # maximised over the sweep — an upper bound by construction.
        "best_nmi_group_gap_over_sweep": max((r["nmi_group_gap"] for r in sweep
                                              if r["n_segments"] > 1), default=0.0),
        "best_nmi_time_gap_over_sweep": max((r["nmi_time_gap"] for r in sweep
                                             if r["n_segments"] > 1), default=0.0),
        "nmi_group": sa.nmi(labels, group_ref),
        "nmi_group_shuffled": sa.shuffled_nmi(labels, group_ref, rng),
        "nmi_time": sa.nmi(labels, time_ref),
        "nmi_time_shuffled": sa.shuffled_nmi(labels, time_ref, rng),
        # Read at the granularity where the statistic has dynamic range (see
        # pick_segment_laterality); affinity_laterality is the threshold-free version.
        "segment_laterality": (sa.pick_segment_laterality(sweep, labels_fg, glabels, F, G)
                               if is_group else {}),
        # The self-attention analogue of M1/M2's "is this just a source-dynamics
        # detector?" number.
        "corr_incoming_attention_vs_source_motion": flat_corr(
            sa.incoming_attention(A_null, F, G), src_act),
    }

    instructions = args.instructions or list(DEFAULT_INSTRUCTIONS)
    print(f"Text-invariance check over {len(instructions)} instructions …")
    instr_corr, vs_null = text_invariance(
        model, state, timesteps, N, keep_layers, text_encoder, instructions, A_null)
    metrics["instruction_invariance_mean_off_diag_r"] = mean_off_diagonal(instr_corr)
    metrics["corr_vs_null_context"] = vs_null

    base = os.path.join(args.out_dir, f"{clip_id}_selfattn")
    anchors = [(A_null[(F // 2) * G + g], f"(frame {F//2}, {glabels[g]})")
               for g in range(min(2, G))]
    plot_structure(clip_id, caption, C, R, layer_diag, glabels, anchors, F, G, src_act,
                   metrics["group_diagonality"], metrics["frame_diagonality"],
                   base + "_structure.png")
    plot_segments(clip_id, labels_fg, glabels, metrics, instr_corr, instructions,
                  base + "_segments.png")
    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {base}.json")

    print_summary(metrics, F, G, n_seg, tau, len(instructions))


if __name__ == "__main__":
    main()
