"""
Is ψ a mixture of "the edit ADDS motion here" and "the edit SUPPRESSES the source here"?

The hypothesis (user, 2026-08-11). HumanML3D captions describe a whole clip, so
conditioning on "raise the right arm" pulls the prediction toward a clip in which the
right arm moves AND the rest of the body is still. Run against a source clip that moves
other parts, the conditional branch therefore differs from the reference in two places
at once: at the target group (motion added) and wherever the source moves (motion
removed). LEDITS++ reads ψ = |f_θ(x_t,c) − f_θ(x_t,ref)|, whose absolute value maps both
onto "large" — which would explain ψ's ~+0.5 correlation with source dynamics without
ψ being a mere source-motion detector.

The test. Read the same two forward passes as a SIGNED change in motion energy,
E[f_θ(x_t,c)] − E[f_θ(x_t,ref)] per (frame, group) (`psi_readout="energy"`). The
hypothesis predicts a specific sign pattern:

  H1  the target group's energy change is POSITIVE,
  H2  non-target groups' energy change is NEGATIVE in proportion to how much the SOURCE
      moves them — i.e. corr(source energy, ΔE) < 0 across non-target groups.

A pure scale/magnitude explanation — that both branches simply predict larger values
where the motion is larger, so their absolute difference is larger there — predicts H2's
correlation to be ≈ 0 or positive. The two are only distinguishable with the sign, which
is why `.abs()` hides this.

The fix, if the hypothesis holds. Threshold the signed map instead of the absolute one:
the suppression term is then negative and falls below the percentile cut, so the mask
keeps only what the instruction ADDS. This scores that mask against the ψ_abs baseline,
against grounded M1 alone, and against both intersections.

A caveat about rectification, measured rather than assumed. `relu(ΔE)` looks like a
no-op for a percentile threshold — relu is monotone, and the mask-rescaling theorem says
a monotone rescaling cannot change a percentile ranking. It is not, and the reason is
instructive: relu is monotone but not STRICTLY monotone, and it maps every negative cell
to exactly 0. Here ~72 % of cells are negative, so after relu the 70th percentile IS 0
and `values >= thr` admits the entire map. The theorem needs strict monotonicity; ties at
the cut break it. So this probe thresholds the SIGNED map, and `relu` is not offered.

Usage
-----
    python src/probe_psi_sign.py \
        --checkpoint runs/exp_hml3d_masked/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D \
        --clip 012698 --clip 005742 --clip 008433 --clip 006516 \
        --out_dir eval_results/psi_sign
"""

import os
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from analysis.instructions import DEFAULT_INSTRUCTIONS, resolve_targets
from analysis.mask_axes import alignment, axis_stats
from data.clips import load_clip
from editing import MotionEditor, masking
from editing.masking import semantic_token_subset
from model.body_groups import group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_layers
from utils.cli import add_data_args, add_mask_args, add_model_args, resolve_device
from utils.model_io import load_model
from utils.probe import flat_corr, group_profile, resolve_sweeps, source_activity
from utils.visualise.masks import plot_psi_sign

PSI_READOUTS = ("abs", "energy")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, multi_checkpoint=True)
    add_data_args(p, split=False)
    add_mask_args(p)
    p.add_argument("--clip", action="append", required=True,
                   help="Clip id in <data_root>/new_joint_vecs. Repeat for several.")
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--out_dir", default="eval_results/psi_sign")
    return p.parse_args()


def sign_stats(energy_maps, src_act, glabels, targets, valid) -> list[dict]:
    """H1/H2 per instruction, from the signed (F, G) energy-change maps.

    Everything is computed on per-GROUP marginals (mean over valid frames), because the
    hypothesis is about which body part the instruction moves and which it stills — a
    per-group claim. The source profile is normalised to sum 1; the energy change is
    NOT (its sign and relative scale are the whole point).
    """
    src_prof = group_profile(src_act)
    out = []
    for emap, tgt in zip(energy_maps, targets):
        de = np.asarray(emap)[valid].mean(axis=0)                  # (G,) signed
        ti = [glabels.index(g) for g in tgt if g in glabels]
        oi = [i for i in range(len(glabels)) if i not in ti]
        # H2 needs variation in the source profile to correlate against; a clip whose
        # non-target groups all move equally cannot answer it either way.
        corr = (float(np.corrcoef(src_prof[oi], de[oi])[0, 1])
                if np.std(src_prof[oi]) > 1e-12 and np.std(de[oi]) > 1e-12 else float("nan"))
        w = src_prof[oi] / max(src_prof[oi].sum(), 1e-12)
        out.append({
            "delta_energy_target": float(de[ti].mean()) if ti else float("nan"),
            "delta_energy_other": float(de[oi].mean()),
            # source-weighted: what happens to the groups the SOURCE actually moves
            "delta_energy_other_src_weighted": float((de[oi] * w).sum()),
            "corr_src_vs_delta_nontarget": corr,
            "profile": de.tolist(),
        })
    return out


def recall(binary_maps, glabels, targets) -> list[float]:
    """Per instruction: the share of the TARGET group's cells the mask keeps.

    Alignment alone is precision, and precision is trivially raised by shrinking the
    mask — so a sparser mask's higher alignment means nothing without this number next
    to it. A mask that scores 1.0 alignment on three cells has not solved anything: the
    editor needs enough of the target region to actually edit it.
    """
    out = []
    for m, tgt in zip(binary_maps, targets):
        idx = [glabels.index(g) for g in tgt if g in glabels]
        cells = m[:, idx]
        out.append(float(cells.sum() / cells.size) if idx and cells.size else 0.0)
    return out


def match_size(attn_maps, psi_map_list, target_counts, valid, is_group, editor, args):
    """M1 ∩ threshold(ΔE, p) with p chosen per instruction so the mask has as close to
    `target_counts` active cells as the grid allows.

    The size-matched control for the headline comparison: the ΔE intersection is sparser
    than the ψ one at the same percentile, and a sparser mask scores higher alignment for
    free. Equalising the size is what separates "ΔE ranks cells better" from "ΔE happens
    to cut more of them".
    """
    out, used = [], []
    for a, p, want in zip(attn_maps, psi_map_list, target_counts):
        best, best_pct = None, None
        for pct in np.arange(0.0, 99.5, 0.5):
            m = masking.build_mask(a, p, valid, is_group, lambda_attn=args.lambda_attn,
                                   lambda_noise=float(pct), mask_mode="attn",
                                   group_channels=editor.group_channels,
                                   feat_dim=editor.feat_dim)["m_group"]
            n = int(m.sum())
            if best is None or abs(n - want) < abs(int(best.sum()) - want):
                best, best_pct = m, float(pct)
        out.append(best.float().cpu().numpy())
        used.append(best_pct)
    return out, used


def probe_one(ckpt, clip, args, device) -> dict:
    model, config = load_model(ckpt, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    if not is_group:
        raise SystemExit(f"{ckpt}: flat model (G=1) has no body-part axis.")
    glabels = group_names(group_mode)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    raw, F, caption = load_clip(args.data_root, clip, args.max_frames)
    x0 = torch.from_numpy((raw - mean) / std).float().unsqueeze(0).to(device)
    valid = torch.ones(F, dtype=torch.bool, device=device)

    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space,
                          attn_layers=resolve_readout_layers(config, args.m1_layers))
    state = editor.invert(x0)
    src_act = source_activity(x0, editor.group_channels)
    valid_np = valid.cpu().numpy()

    instructions = list(DEFAULT_INSTRUCTIONS)
    targets = resolve_targets(instructions, None, group_mode)
    shared, m1_ts, m2_ts = resolve_sweeps(args.mask_timesteps, schedule.T,
                                          args.m1_window, args.m2_window)

    # Both ψ read-outs come off the SAME two forward passes per timestep, so the
    # abs-vs-energy contrast carries no inversion or sampling noise at all.
    m1_maps, psi = [], {r: [] for r in PSI_READOUTS}
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
        tok_info = [text_encoder.token_info(e) for e in instructions]
    for ctx, ti in zip(ctxs, tok_info):
        attn_fg, psi_fg = masking.collect_statistics(
            model, schedule, state.xs, ctx, ti[0], is_group=is_group,
            timesteps=shared, need_attn=True, group_channels=editor.group_channels,
            valid_frames=valid, semantic_idxs=semantic_token_subset(*ti),
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=args.per_step_norm,
            psi_space=editor.edit_space, attn_layers=editor.attn_layers,
            psi_readout=PSI_READOUTS)
        m1_maps.append(attn_fg)
        for r in PSI_READOUTS:
            psi[r].append(psi_fg[r])

    np_m1 = [m.cpu().numpy() for m in m1_maps]
    np_psi = {r: [m.cpu().numpy() for m in maps] for r, maps in psi.items()}

    # Masks: ψ alone under each read-out, M1 alone, and each intersection — scored
    # through the editor's own build_mask so the numbers sit in the same table as
    # every other alignment in this project.
    masks = {}
    for r in PSI_READOUTS:
        for mode in ("m2_only", "attn"):
            masks[f"{mode}_{r}"] = [
                masking.build_mask(a, p, valid, is_group, lambda_attn=args.lambda_attn,
                                   lambda_noise=args.lambda_noise, mask_mode=mode,
                                   group_channels=editor.group_channels,
                                   feat_dim=editor.feat_dim)["m_group"].float().cpu().numpy()
                for a, p in zip(m1_maps, psi[r])]
    masks["m1_only"] = [
        masking.build_mask(a, psi["abs"][i], valid, is_group, lambda_attn=args.lambda_attn,
                           lambda_noise=args.lambda_noise, mask_mode="m1_only",
                           group_channels=editor.group_channels,
                           feat_dim=editor.feat_dim)["m_group"].float().cpu().numpy()
        for i, a in enumerate(m1_maps)]
    # The control that decides whether the ΔE intersection is better or merely smaller.
    matched, matched_pct = match_size(
        m1_maps, psi["energy"], [int(m.sum()) for m in masks["attn_abs"]],
        valid, is_group, editor, args)
    masks["attn_energy_matched"] = matched

    return {
        "checkpoint": ckpt, "clip": clip, "frames": F, "caption": caption,
        "edit_space": editor.edit_space, "attn_layers": editor.attn_layers,
        "instructions": instructions, "targets": [list(t) for t in targets],
        "group_labels": glabels, "align_chance": 1.0 / len(glabels),
        "sign": sign_stats(np_psi["energy"], src_act, glabels, targets, valid_np),
        "align": {k: alignment(v, glabels, targets) for k, v in masks.items()},
        "recall": {k: recall(v, glabels, targets) for k, v in masks.items()},
        "cells": {k: float(np.mean([m.sum() for m in v])) for k, v in masks.items()},
        "matched_percentile": matched_pct,
        "r_laterality": {r: axis_stats(np_psi[r], glabels, instructions, targets)["r_laterality"]
                         for r in PSI_READOUTS},
        "r_category": {r: axis_stats(np_psi[r], glabels, instructions, targets)["r_category"]
                       for r in PSI_READOUTS},
        "src_corr": {r: float(np.mean([flat_corr(m, src_act) for m in np_psi[r]]))
                     for r in PSI_READOUTS},
        "_maps": np_psi, "_m1": np_m1, "_src": src_act, "_masks": masks,
    }


def print_report(results):
    print("\n── H1/H2: the sign of the energy change ────────────────────────────")
    print("  H1: ΔE at the TARGET group > 0        H2: corr(source energy, ΔE) < 0 off-target")
    print(f"\n  {'clip':8} {'instruction':24} {'ΔE target':>10} {'ΔE other':>10} "
          f"{'ΔE other (src-w)':>17} {'corr H2':>9}")
    h1 = h2 = n = 0
    for d in results:
        for instr, s in zip(d["instructions"], d["sign"]):
            h1 += s["delta_energy_target"] > 0
            h2 += s["corr_src_vs_delta_nontarget"] < 0
            n += 1
            print(f"  {d['clip']:8} {instr:24} {s['delta_energy_target']:+10.4f} "
                  f"{s['delta_energy_other']:+10.4f} "
                  f"{s['delta_energy_other_src_weighted']:+17.4f} "
                  f"{s['corr_src_vs_delta_nontarget']:+9.3f}")
    print(f"\n  H1 holds in {h1}/{n} cases;  H2 holds in {h2}/{n} cases")
    corrs = [s["corr_src_vs_delta_nontarget"] for d in results for s in d["sign"]
             if np.isfinite(s["corr_src_vs_delta_nontarget"])]
    print(f"  mean corr(source energy, ΔE) over non-target groups = {np.mean(corrs):+.3f}")

    print("\n── does the signed read-out fix the mask? (alignment, chance "
          f"{results[0]['align_chance']:.3f}) ──")
    keys = ["m2_only_abs", "m2_only_energy", "m1_only", "attn_abs", "attn_energy",
            "attn_energy_matched"]
    print(f"\n  {'clip':8} " + " ".join(f"{k:>19}" for k in keys))
    for d in results:
        print(f"  {d['clip']:8} "
              + " ".join(f"{np.mean(d['align'][k]):19.3f}" for k in keys))
    print(f"  {'MEAN':8} "
          + " ".join(f"{np.mean([np.mean(d['align'][k]) for d in results]):19.3f}"
                     for k in keys))
    print(f"  {'recall':8} "
          + " ".join(f"{np.mean([np.mean(d['recall'][k]) for d in results]):19.3f}"
                     for k in keys))
    print(f"  {'cells':8} "
          + " ".join(f"{np.mean([d['cells'][k] for d in results]):19.0f}" for k in keys))
    print("\n  alignment = precision (share of mask cells inside the target group);\n"
          "  recall    = share of the target group's cells kept. A sparser mask buys\n"
          "  precision for free, which is what `attn_energy_matched` controls for.")
    print("\n  ψ map statistics (r → 1 = ignores the instruction):")
    for r in PSI_READOUTS:
        print(f"    {r:7} r_lat {np.mean([d['r_laterality'][r] for d in results]):.3f}   "
              f"r_cat {np.mean([d['r_category'][r] for d in results]):.3f}   "
              f"corr with source motion {np.mean([d['src_corr'][r] for d in results]):+.3f}")


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
            if not args.no_figures:
                # The last panel shows the mask this probe actually recommends — the
                # size-matched M1 ∩ ΔE — not ΔE alone, which is no better than ψ alone
                # as a standalone mask (the gain is in the combination).
                plot_psi_sign(
                    res["clip"], res["caption"], res["instructions"], res["targets"],
                    res["_maps"]["abs"], res["_maps"]["energy"], res["_m1"],
                    res["_masks"]["attn_energy_matched"], res["_src"], res["group_labels"],
                    os.path.join(args.out_dir, f"{tag}_{clip}_psi_sign.png"),
                    align_abs=res["align"]["attn_abs"],
                    align_energy=res["align"]["attn_energy_matched"])
            out = os.path.join(args.out_dir, f"{tag}_{clip}_psi_sign.json")
            with open(out, "w") as f:
                json.dump({k: v for k, v in res.items() if not k.startswith("_")},
                          f, indent=2)
            print(f"  wrote {out}")
            results.append(res)

    print_report(results)


if __name__ == "__main__":
    main()
