"""
What settings produced a batch of generations, and whether this run may add to them.

The resume rule is "skip a clip whose .npy already exists", which is only sound while
those files mean the same thing. Otherwise one score sheet silently mixes two
configurations — and the manifest, rewritten at the end, describes only the newer one.
"""

import json
import os
import random

from editing.masking import mask_mode_components
from training.grounding import resolve_readout_layers
from utils.logger import get_logger

log = get_logger(__name__)



# Settings that change the CONTENT of a generated .npy, hence what may not silently differ
# between the files already in an output dir and the ones about to join them. Conditioned on
# mask_mode so that changing an inert knob (lambda_attn under m2_only, say) is not a false
# mismatch. --scales and --limit are deliberately absent: adding a scale or more clips is a
# legitimate resume, since each lands in its own file.
def mask_fingerprint(args, editor, config) -> dict:
    fp = {
        "checkpoint":   os.path.abspath(args.checkpoint),
        "weights":      "model" if args.no_ema else "ema",
        "mask_mode":    args.mask_mode,
        "edit_space":   editor.edit_space,
        "alpha_floor":  editor.resolve_alpha_floor(args.guidance_alpha_floor),
        "seed":         args.seed,
        "mask_timesteps": args.mask_timesteps,
        "src_fps": args.src_fps, "edit_fps": args.edit_fps,
        "max_frames": args.max_frames, "min_frames": args.min_frames,
    }
    # Which components this mode actually uses comes from masking's own table, so adding a
    # mask mode cannot leave the fingerprint silently omitting that mode's live flags —
    # which would let the guard pass a genuinely mixed run.
    semantic_source, uses_m2 = mask_mode_components(args.mask_mode)
    uses_m1 = semantic_source == "attn"
    if uses_m1:
        fp.update({
            "m1_columns": args.m1_columns,
            "m1_layers":  resolve_readout_layers(config, args.m1_layers),
            "m1_select":  args.m1_select,
            "m1_window":  args.m1_window,
        })
        if args.m1_select == "rank":
            fp.update({"m1_rank_ratio": args.m1_rank_ratio,
                       "m1_rank_max": args.m1_rank_max})
        else:
            fp["lambda_attn"] = args.lambda_attn
    if uses_m2:
        fp.update({"psi_readout": editor.psi_readout,
                   "lambda_noise": args.lambda_noise,
                   "m2_window": args.m2_window,
                   "per_step_norm": args.per_step_norm})
    return fp


def check_resumable(args, out_dirs, fingerprint, manifest_path):
    """Refuse to add files to an output dir whose existing generations came from different
    settings. The resume rule is "skip a clip whose .npy exists", which is only sound while
    those files mean the same thing — otherwise one score sheet silently mixes two
    configurations and the manifest, rewritten at the end, describes only the newer one."""
    if args.overwrite:                    # every file is recomputed; nothing stale survives
        return
    present = [d for d in out_dirs.values()
               if os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d))]
    if not present:
        return
    old = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f).get("fingerprint", {})
    if old == fingerprint:
        return
    diff = [f"    {k}: on disk {old.get(k, '<absent>')!r} -> now {fingerprint[k]!r}"
            for k in sorted(fingerprint) if old.get(k) != fingerprint[k]]
    msg = (f"\nRefusing to resume: {len(present)} output dir(s) under {args.out_root} already "
           f"hold generations made with different settings.\n"
           + ("\n".join(diff) if diff else "    (no fingerprint recorded — a pre-2026-08-16 run)")
           + "\n\nThose .npy files would be kept as-is and mixed with new ones in the same "
             "score sheet.\nPick one:\n"
             "  --out_root <a fresh dir>   keep both runs (recommended)\n"
             "  --overwrite                 recompute everything with the new settings\n"
             "  --ignore_fingerprint        proceed anyway, accepting the mixture\n")
    if args.ignore_fingerprint:
        log.info(msg + "\n--ignore_fingerprint given: proceeding with a MIXED output set.\n")
        return
    raise SystemExit(msg)


def select_keyids(data, args):
    """The clips to edit. `--limit_mode random` samples with its own seeded RNG and then
    restores sorted order, so the SET is a sample while the iteration order stays stable."""
    keyids = sorted(data.keys())
    if not args.limit or args.limit >= len(keyids):
        return keyids
    if args.limit_mode == "first":
        return keyids[:args.limit]
    return sorted(random.Random(args.limit_seed).sample(keyids, args.limit))
