"""
Can the source-motion component be removed from M2 (or M1), leaving a usable mask?

The implicit masks track the source clip's own motion `S = |Δx0|`, which is
instruction-independent. This sweeps λ over two ways of taking it out —

    sub   M' = n(M) − λ·n(S)                 (additive)
    div   M' = log(u(M)) − λ·log(u(S))       (multiplicative; ~ u(M)/u(S)^λ)

— and scores the mask each remainder produces, against SHUFFLED-source controls.

Read the alignment column, not the r columns. `S` is instruction-independent, so
subtracting anything correlated with the masks' common component moves the
instruction-invariance r whether or not the remainder means anything; and a percentile
threshold is invariant to monotone rescaling, so r can move when the mask provably
cannot. The controls (all cells permuted / group columns permuted) are what separates
"the source was removed" from "a source-shaped map was removed".

History: this was run once before (2026-08-01, script since lost) and came back
negative — alignment fell monotonically and lost to its own shuffled control at every
λ > 0. Two things have changed since, which is why it is worth re-running rather than
citing: ψ/M2 is now read x0-natively (the ε-space shim was rewritten 2026-08-05, and M2
numbers are NOT comparable across spaces), and the grounded checkpoint exists. The `div`
arithmetic is new — the retired divisive variant was per-group energy normalisation
(`--m2_group_norm`), not per-cell division.

Outputs per checkpoint × clip: a JSON of the full grid, a λ-sweep figure, and — at the
best λ — the four-column raw-vs-corrected map/mask figure in the requested mask mode.

Usage
-----
    python src/probe_source_subtraction.py \
        --checkpoint runs/exp_hml3d_masked/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D \
        --clip 012698 --clip 005742 --clip 008433 --clip 006516 \
        --out_dir eval_results/source_correction
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
from analysis.mask_probe import collect_instruction_masks
from analysis.source_correction import (
    CONTROLS, MODES, NORMS, correct, effective_norms, shuffled_source, sweep_grid,
)
from data.clips import load_clip
from editing import MotionEditor, masking
from model.body_groups import group_names, resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_layers
from utils.cli import add_data_args, add_mask_args, add_model_args, resolve_device
from utils.model_io import load_model
from utils.probe import flat_corr, resolve_sweeps, source_activity
from utils.visualise import plot_correction_sweep, plot_source_correction


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, multi_checkpoint=True)
    add_data_args(p, split=False)
    add_mask_args(p)
    p.add_argument("--clip", action="append", required=True,
                   help="Clip id in <data_root>/new_joint_vecs. Repeat for several.")
    p.add_argument("--mask", default="m2", choices=["m2", "m1"],
                   help="Which raw map to correct (default m2 — the noise ψ).")
    p.add_argument("--mode", action="append", choices=list(MODES), default=None,
                   help="Correction arithmetic; repeat for several (default: both).")
    p.add_argument("--norm", action="append", choices=list(NORMS), default=None,
                   help="Normalisation applied to both maps before correcting "
                        "(default: z. 'div' silently uses the unit-mean variant, since "
                        "a log needs a positive map).")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                   help="λ grid. 0 must be present — it is the baseline every other "
                        "value is judged against.")
    p.add_argument("--floor_q", type=float, default=5.0,
                   help="Percentile the source is clamped at from below before a "
                        "division (bounds the amplification of still regions, which is "
                        "div's specific failure mode). Only affects --mode div.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the shuffled controls.")
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--out_dir", default="eval_results/source_correction")
    return p.parse_args()


def score(maps, src_act, valid_np, glabels, targets, instructions, lambda_pct,
          is_group, editor, m1_maps):
    """Every statistic for ONE set of corrected (F, G) maps, scored exactly as the
    editor would: the binary mask goes through `masking.build_mask` in m2_only, so an
    alignment here is comparable to one from probe_mask_axes.py."""
    valid_t = torch.from_numpy(valid_np).to(torch.bool)
    bins = []
    for corrected, m1 in zip(maps, m1_maps):
        psi = torch.from_numpy(np.ascontiguousarray(corrected)).float()
        attn = torch.from_numpy(np.ascontiguousarray(m1)).float()
        bins.append(masking.build_mask(
            attn, psi, valid_t, is_group, lambda_noise=lambda_pct, mask_mode="m2_only",
            group_channels=editor.group_channels, feat_dim=editor.feat_dim,
        )["m_group"].float().numpy())

    stats = axis_stats(maps, glabels, instructions, targets)
    aligns = alignment(bins, glabels, targets)
    return {
        "align": float(np.mean(aligns)),
        "align_per_instruction": aligns,
        "r_laterality": stats["r_laterality"],
        "r_category": stats["r_category"],
        "r_offdiag": stats["r_offdiag"],
        "src_corr": float(np.mean([flat_corr(m, src_act) for m in maps])),
    }, bins


def probe_one(ckpt, clip, args, device) -> dict:
    model, config = load_model(ckpt, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    if not is_group:
        raise SystemExit(f"{ckpt}: flat model (G=1) has no body-part axis to correct.")
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
    print(f"  predict_type={schedule.predict_type}  edit_space={editor.edit_space}  "
          f"attn_layers={editor.attn_layers or 'all'}")
    state = editor.invert(x0)
    src_act = source_activity(x0, editor.group_channels)
    valid_np = valid.cpu().numpy()

    instructions = list(DEFAULT_INSTRUCTIONS)
    targets = resolve_targets(instructions, None, group_mode)
    m1_maps, m2_maps, _ = collect_instruction_masks(
        model, schedule, editor, state, text_encoder, instructions, valid, is_group,
        mask_modes=("m2_only",), lambda_attn=args.lambda_attn,
        lambda_noise=args.lambda_noise,
        sweeps=resolve_sweeps(args.mask_timesteps, schedule.T,
                              args.m1_window, args.m2_window),
        per_step_norm=args.per_step_norm)
    base_maps = m2_maps if args.mask == "m2" else m1_maps

    rng = np.random.default_rng(args.seed)
    modes = tuple(args.mode or MODES)
    norms = tuple(args.norm or ("z",))
    grid, cached_maps = {}, {}
    for mode, norm, control, lam in sweep_grid(args.lambdas, modes, norms, CONTROLS):
        S = shuffled_source(src_act, control, rng)
        maps = [correct(m, S, lam, mode, norm, valid_np, args.floor_q)
                for m in base_maps]
        res, bins = score(maps, src_act, valid_np, glabels, targets, instructions,
                          args.lambda_noise, is_group, editor, m1_maps)
        key = f"{mode}|{norm}|{control}|{lam:g}"
        grid[key] = res
        if control == "real":
            cached_maps[key] = (maps, bins)

    return {
        "checkpoint": ckpt, "clip": clip, "frames": F, "caption": caption,
        "mask": args.mask, "predict_type": config.get("predict_type", "eps"),
        "edit_space": editor.edit_space, "feature_mode": feature_mode,
        "attn_layers": editor.attn_layers, "lambda_noise": args.lambda_noise,
        "floor_q": args.floor_q, "lambdas": list(args.lambdas),
        "instructions": instructions, "targets": [list(t) for t in targets],
        "group_labels": glabels, "align_chance": 1.0 / len(glabels),
        "grid": grid,
        # not JSON — carried out for the figures, dropped before writing
        "_maps": cached_maps, "_src_act": src_act, "_glabels": glabels,
        "_base_maps": base_maps,
    }


def best_key(res, mode, norm, positive_only=True) -> str:
    """The λ that maximises alignment for one (mode, norm) on the REAL source.

    `positive_only` excludes λ=0 — for the FIGURE, where showing "the best correction is
    no correction" as an unchanged pair of panels hides what the correction does to the
    map. The verdict table has no such exclusion, so a negative result stays visible as
    a number even when the picture shows the best-case correction.
    """
    keys = [k for k in res["grid"] if k.startswith(f"{mode}|{norm}|real|")
            and (not positive_only or float(k.split("|")[-1]) > 0)]
    return max(keys, key=lambda k: res["grid"][k]["align"])


def make_figures(res, args, out_dir, tag):
    """The sweep curves per (mode, norm), and the four-column map/mask figure at the
    best λ — the picture the sweep table is an index into."""
    modes = tuple(args.mode or MODES)
    norms = tuple(args.norm or ("z",))
    clip = res["clip"]
    for mode in modes:
        for norm in effective_norms(mode, norms):
            curves = {}
            for control in CONTROLS:
                series = {}
                for lam in args.lambdas:
                    k = f"{mode}|{norm}|{control}|{lam:g}"
                    if k not in res["grid"] and lam == 0:
                        k = f"{mode}|{norm}|real|0"        # controls share the baseline
                    if k in res["grid"]:
                        series[lam] = res["grid"][k]
                if series:
                    curves[control] = series
            stem = f"{tag}_{clip}_{mode}_{norm}"
            plot_correction_sweep(
                curves, res["align_chance"], os.path.join(out_dir, f"{stem}_sweep.png"),
                f"{tag} · clip {clip} · {res['mask'].upper()} corrected by "
                f"'{mode}' ({norm}-normalised), ψ read in {res['edit_space']} space")

            bk = best_key(res, mode, norm)
            lam = float(bk.split("|")[-1])
            maps, bins = res["_maps"][bk]
            _, base_bins = res["_maps"][f"{mode}|{norm}|real|0"]
            plot_source_correction(
                clip, res["caption"], res["instructions"], res["targets"],
                res["_base_maps"], maps, base_bins, bins, res["_src_act"],
                res["_glabels"],
                f"{tag} · {res['mask'].upper()} · {mode} λ={lam:g} ({norm}) · "
                f"align {res['grid'][f'{mode}|{norm}|real|0']['align']:.3f} → "
                f"{res['grid'][bk]['align']:.3f}",
                os.path.join(out_dir, f"{stem}_maps.png"),
                align_raw=res["grid"][f"{mode}|{norm}|real|0"]["align_per_instruction"],
                align_corr=res["grid"][bk]["align_per_instruction"])


def print_table(res, tag):
    print(f"\n{tag}  clip {res['clip']}   ({res['mask'].upper()} corrected; "
          f"chance align = {res['align_chance']:.3f})")
    hdr = (f"  {'mode':5} {'norm':11} {'control':14} {'lambda':>7} | {'align':>7} "
           f"{'r_lat':>7} {'r_cat':>7} {'M~S':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, v in res["grid"].items():
        mode, norm, control, lam = key.split("|")
        print(f"  {mode:5} {norm:11} {control:14} {float(lam):7.2f} | "
              f"{v['align']:7.3f} {v['r_laterality']:7.3f} {v['r_category']:7.3f} "
              f"{v['src_corr']:+7.3f}")


def verdict(results, args):
    """The two questions the sweep exists to answer, stated as pass/fail.

    (1) does any λ > 0 beat the untouched map, averaged over clips, and
    (2) does it beat its own shuffled controls at that λ.
    A correction that fails (2) is removing a source-SHAPED map, not the source.
    """
    modes = tuple(args.mode or MODES)
    norms = tuple(args.norm or ("z",))
    print("\n── verdict (mean over clips) ───────────────────────────")
    for mode in modes:
        for norm in effective_norms(mode, norms):
            base = np.mean([r["grid"][f"{mode}|{norm}|real|0"]["align"] for r in results])
            rows = []
            for lam in args.lambdas:
                if lam == 0:
                    continue
                got = {c: [r["grid"].get(f"{mode}|{norm}|{c}|{lam:g}", {}).get("align")
                           for r in results] for c in CONTROLS}
                if any(v is None for v in got["real"]):
                    continue
                rows.append((lam, {c: float(np.mean(v)) for c, v in got.items()
                                   if all(x is not None for x in v)}))
            if not rows:
                continue
            lam, best = max(rows, key=lambda r: r[1]["real"])
            ctrl = max(best[c] for c in CONTROLS if c != "real" and c in best)
            print(f"  {mode:5} {norm:11} baseline λ=0 align {base:.3f}   "
                  f"best λ={lam:g} align {best['real']:.3f}   "
                  f"best shuffled control {ctrl:.3f}")
            gained = best["real"] > base + 1e-9
            beats = best["real"] > ctrl
            print(f"        → {'IMPROVES' if gained else 'does NOT improve'} on the "
                  f"untouched map; {'beats' if beats else 'LOSES TO'} its shuffled "
                  f"control{'' if gained and beats else '  ⇒ negative'}")


def main():
    args = parse_args()
    if 0.0 not in args.lambdas:
        raise SystemExit("--lambdas must include 0 (the baseline every λ is judged against)")
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
                make_figures(res, args, args.out_dir, tag)
            print_table(res, tag)
            out = os.path.join(args.out_dir, f"{tag}_{clip}_{args.mask}.json")
            with open(out, "w") as f:
                json.dump({k: v for k, v in res.items() if not k.startswith("_")},
                          f, indent=2)
            print(f"  wrote {out}")
            results.append(res)

    verdict(results, args)


if __name__ == "__main__":
    main()
