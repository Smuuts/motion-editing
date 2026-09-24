"""
Mirror-pair editing eval: the one protocol in this project with an EXACT ground-truth target.

HumanML3D ships every clip twice, `X` and its left/right mirror `MX`, with the mirror's
captions left/right-swapped line for line. When a caption names a side, the pair is an edit
with a known answer: source = `X`, instruction = the mirrored caption, target = `MX` (the
exact reflection, same length, frame-aligned). Both directions are edited (`X -> MX` and
`MX -> X`), so a model that simply prefers one side cannot score by bias.

Why it exists (docs/EVALUATION.md Option B, §6D): the text is exactly in-distribution — it
IS a HumanML3D caption — so the MotionFix register gap is removed, and the score is a
distance to the true target rather than a retrieval proxy. If the editor works here and not
on MotionFix, the register is isolated as the cause.

What it does not test: a mirror reflects the WHOLE body, so a part-local edit cannot close
the gap completely. score_mirror_pairs.py therefore reports the error inside and outside the
named body parts separately.

Pair selection: the pair qualifies when some caption line differs between `X` and `MX`
(HumanML3D copies non-lateral captions verbatim to the twin, so "differs" == "names a
side"); the first such line is used. `--caption_line first` keeps only pairs whose FIRST
line differs (the stricter, smaller set).

`--mask_mode groups` routes the UNION of the source and target captions: mirroring moves the
named limb AND its twin (stop kicking with the left leg, start with the right), so routing
the target caption alone would leave half the edit outside the mask.

Output, one raw (T, 135) feature file per (scale, direction, clip):
    {out_root}/{mask_mode}_s{scale}/{source_id}.npy   (source_id is X or MX)
plus {out_root}/pairs.json (the pair list, captions, routed groups) and a manifest.

Score with (CPU, no MotionFix venv needed):
    python src/eval/score_mirror_pairs.py --out_root <out_root>

Example (smoke test on 4 pairs):
    python src/eval/edit_mirror_pairs.py --checkpoint runs/exp_smplh_llm_labels/checkpoint_latest \
        --data_root data/HumanML3D/HumanML3D_smplh --mask_mode attn --scales 0 6 --limit 4
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import json
import random

import numpy as np
import torch

from data.body_part_labels import route_groups
from data.clips import read_captions, split_ids
from editing import MotionEditor, derive_seed
from editing.masking import mask_mode_components
from model.body_groups import resolve_group_context
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from training.grounding import resolve_readout_columns, resolve_readout_layers
from utils.cli import (add_logging_args, add_mask_args, add_model_args,
                       configure_logging, parse_group_mask, resolve_device)
from utils.logger import get_logger
from eval.provenance import check_resumable, mask_fingerprint
from utils.model_io import load_model

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_mask_args(p, mask_timesteps=40, alpha_floor=True)
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D_smplh",
                   help="SMPL-H root with new_joint_vecs/, texts/, <split>.txt, Mean/Std.")
    p.add_argument("--split", default="test",
                   help="Default test: val was used for probe development in this project.")
    p.add_argument("--caption_line", default="any", choices=["any", "first"],
                   help="'any' = use the first caption line that differs between X and MX "
                        "(test: 1257 pairs). 'first' = only pairs whose first line differs "
                        "(test: 766).")
    p.add_argument("--directions", default="both", choices=["both", "forward", "reverse"],
                   help="forward = X -> MX, reverse = MX -> X.")
    p.add_argument("--out_root", default="eval_results/mirror_pairs/run",
                   help="Per-configuration output root (one subdir per scale).")
    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 5.0])
    p.add_argument("--mask_mode", default="attn",
                   choices=["none", "m2_only", "m1_only", "attn", "temporal", "groups"])
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="Only N PAIRS (seeded random sample; 0 = all). Smoke test.")
    p.add_argument("--limit_seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ignore_fingerprint", action="store_true")
    add_logging_args(p)
    args = p.parse_args()
    # mask_fingerprint reads these; the clips are native 20 fps, no resampling happens.
    args.src_fps = args.edit_fps = 20.0
    return configure_logging(args)


def mirror_pairs(data_root, split, caption_line="any", min_frames=16):
    """[{base, mirror, line, cap_base, cap_mirror, T}] for lateralised mirror pairs."""
    ids = set(split_ids(data_root, split))
    pairs = []
    for cid in sorted(ids):
        if cid.startswith("M") or "M" + cid not in ids:
            continue
        a, b = read_captions(data_root, cid), read_captions(data_root, "M" + cid)
        lines = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if not lines or (caption_line == "first" and lines[0] != 0):
            continue
        T = int(np.load(os.path.join(data_root, "new_joint_vecs", f"{cid}.npy"),
                        mmap_mode="r").shape[0])
        if T < min_frames:
            continue
        i = lines[0]
        pairs.append({"base": cid, "mirror": "M" + cid, "line": i,
                      "cap_base": a[i], "cap_mirror": b[i], "T": T})
    return pairs


def edit_jobs(pairs, directions):
    """(source_id, target_id, source caption, instruction) per edit."""
    jobs = []
    for p in pairs:
        if directions in ("both", "forward"):
            jobs.append((p["base"], p["mirror"], p["cap_base"], p["cap_mirror"]))
        if directions in ("both", "reverse"):
            jobs.append((p["mirror"], p["base"], p["cap_mirror"], p["cap_base"]))
    return jobs


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    if config.get("feature_mode", "humanml3d") != "smplh":
        raise SystemExit("SMPL-H checkpoints only (the mirror twins are built in SMPL-H).")
    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))

    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)
    _, is_group, group_mode, _ = resolve_group_context(config)
    editor = MotionEditor(model, schedule, device, is_group=is_group,
                          edit_space=args.edit_space, psi_readout=args.psi_readout,
                          attn_layers=resolve_readout_layers(config, args.m1_layers))

    from utils.probe import resolve_sweeps
    mask_ts, m1_ts, m2_ts = resolve_sweeps(args.mask_timesteps, schedule.T,
                                           args.m1_window, args.m2_window)

    pairs = mirror_pairs(args.data_root, args.split, args.caption_line, args.min_frames)
    n_all = len(pairs)
    if args.limit and args.limit < n_all:
        pairs = sorted(random.Random(args.limit_seed).sample(pairs, args.limit),
                       key=lambda p: p["base"])
    for p in pairs:
        # Union of both captions: the mirror moves the named limb and its twin.
        p["groups"] = sorted(set(route_groups(p["cap_base"], group_mode))
                             | set(route_groups(p["cap_mirror"], group_mode)))
    jobs = edit_jobs(pairs, args.directions)
    log.info(f"{len(pairs)}/{n_all} lateralised pairs in {args.split} "
             f"(caption_line={args.caption_line}) -> {len(jobs)} edits × {len(args.scales)} scales")

    out_dirs = {s: os.path.join(args.out_root, f"{args.mask_mode}_s{s:g}") for s in args.scales}
    fingerprint = mask_fingerprint(args, editor, config)
    fingerprint.update({"data_root": os.path.abspath(args.data_root), "split": args.split,
                        "caption_line": args.caption_line})
    manifest_path = os.path.join(args.out_root, f"edit_manifest_{args.mask_mode}.json")
    check_resumable(args, out_dirs, fingerprint, manifest_path)
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "status": "in-progress"}, f, indent=2)
    # The pair list is what the scorer reads; the manifest records how edits were made.
    with open(os.path.join(args.out_root, "pairs.json"), "w") as f:
        json.dump({"split": args.split, "caption_line": args.caption_line,
                   "directions": args.directions, "n_lateralised_in_split": n_all,
                   "pairs": pairs}, f, indent=2)

    groups_of = {p["base"]: p["groups"] for p in pairs}
    groups_of.update({p["mirror"]: p["groups"] for p in pairs})
    need_attn = mask_mode_components(args.mask_mode)[0] == "attn"
    skipped, n_done = {}, 0
    for src_id, _tgt_id, _src_cap, text in log.progress(jobs, desc="Editing", leave=True):
        todo = [s for s in args.scales if args.overwrite
                or not os.path.exists(os.path.join(out_dirs[s], f"{src_id}.npy"))]
        if not todo:
            continue
        group_masks = None
        if args.mask_mode == "groups":
            if not groups_of[src_id]:
                skipped[src_id] = "router found no body part in either caption"
                continue
            group_masks = [parse_group_mask(" ".join(groups_of[src_id]), is_group, group_mode)]

        raw = np.load(os.path.join(args.data_root, "new_joint_vecs", f"{src_id}.npy"))
        raw = raw[:args.max_frames].astype(np.float32)
        T = raw.shape[0]
        x0 = torch.from_numpy((raw - mean) / std).float().unsqueeze(0).to(device)
        valid = torch.ones(T, dtype=torch.bool, device=device)

        state = editor.invert(x0, show_progress=False, seed=derive_seed(args.seed, src_id))
        with torch.no_grad():
            ctx = text_encoder.encode([text])
            tok = sem = None
            if need_attn:
                tok, sem, _ = resolve_readout_columns(text, text_encoder, config,
                                                      args.m1_columns, group_mode)
        masks = editor.collect_masks(
            state, [ctx], [tok], valid, semantic_idxs_per_edit=[sem],
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode, llm_group_masks=group_masks, timesteps=mask_ts,
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=args.per_step_norm,
            m1_select=args.m1_select, m1_rank_ratio=args.m1_rank_ratio,
            m1_rank_max=args.m1_rank_max,
        )
        for s in todo:
            x_edit = editor.edit(state, [ctx], masks, scales=[s], show_progress=False,
                                 guidance_alpha_floor=args.guidance_alpha_floor)
            np.save(os.path.join(out_dirs[s], f"{src_id}.npy"),
                    (x_edit[0].cpu().numpy() * std + mean).astype(np.float32))
        n_done += 1

    manifest = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "mask_mode": args.mask_mode, "scales": args.scales,
        "predict_type": schedule.predict_type, "edit_space": editor.edit_space,
        "guidance_alpha_floor": editor.resolve_alpha_floor(args.guidance_alpha_floor),
        "n_pairs": len(pairs), "n_edits": len(jobs), "n_edited_this_run": n_done,
        "n_skipped": len(skipped), "skipped": skipped,
        "limit": args.limit, "limit_seed": args.limit_seed,
        "seed": args.seed, "seed_mode": "per-clip derive_seed(seed, source_id)",
        "fingerprint": fingerprint,
        "out_dirs": {f"{s:g}": out_dirs[s] for s in args.scales},
        "status": "complete",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Done: edited {n_done} sources this run, skipped {len(skipped)}. "
             f"Score with: python src/eval/score_mirror_pairs.py --out_root {args.out_root}")


if __name__ == "__main__":
    main()
