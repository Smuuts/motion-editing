"""
How hard does each guidance scale actually bite? — the cheap pre-flight for a MotionFix sweep.

The full benchmark costs hours and answers "is the edit BETTER". This answers the much
cheaper question you need first: "does the edit do ANYTHING, and has it already gone off a
cliff". Run it before committing to a scale list, because the recorded scales
(0/2.5/5/7.5) were tuned in eps space, x0 bites harder at the same s (0.014 -> 0.041 m
mean joint displacement at s=2.5), and psi changed to the signed energy
read-out on top of that. If 2.5 is already past the useful range the whole sweep measures
damage and every number will look like the standing negative regardless of mask quality.

WHAT IT MEASURES. Scale 0 reconstructs the source, so it is the natural reference: every
other scale is compared against it, per clip, in the units the edit actually moves.

    rot     mean geodesic angle between the s=0 and s>0 joint rotations, in DEGREES,
            over all frames x 22 joints.
    rotmax  the same, but MAX over the 22 joints before averaging over frames. Read this
            one first for a body-part editor: the mean is over the whole skeleton, so an
            edit that correctly moves one arm and nothing else is divided by ~22 and looks
            like a no-op. `rot` says "how much of the body moved", `rotmax` says "did
            anything move at all".
    trans   mean per-frame root translation change, in METRES.

Both are magnitudes, not quality: a large number means the edit moved a lot, not that it
moved correctly. What you are looking for is the band between "indistinguishable from no
edit" and "the pose has been destroyed", and only then is it worth spending TMR on it.

Reads the MotionFix GEN layout written by edit_motionfix_testset.py —
[trans(3) | global_orient_6d(6) | body_pose_6d(126)] — which is NOT the training feature
layout (that one is [trans_delta(3) | body_pose(126) | global_orient(6)]; mixing them up
silently scrambles the joints).

    python src/eval/scale_scan.py --out_root data/motionfix/motionfix_smpl/exp_smplh_verbs_scan
"""

import os
import sys

# These scripts live one level below src/, so src/ is not on the path when they are run
# directly. Put it there before any project import.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import glob
import json
import re

import numpy as np
import torch

from data.smplh_features import gen_layout_to_rotmats
from utils.logger import add_logging_args, configure_logging, get_logger

log = get_logger(__name__)


def load_clip(path: str):
    """One gen-layout .npy -> (rotmats (T,22,3,3), trans (T,3))."""
    feat = np.load(path)
    return gen_layout_to_rotmats(feat), feat[:, :3]


def load_dir(d: str) -> dict:
    """{keyid: (rotmats, trans)} for every .npy in a scale dir."""
    return {os.path.basename(p)[:-4]: load_clip(p)
            for p in glob.glob(os.path.join(d, "*.npy"))}


def geodesic_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-rotation angle between two (..., 3, 3) batches, in degrees."""
    rel = torch.matmul(a.transpose(-1, -2), b)
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1.0) / 2.0
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def scale_dirs(out_root: str, mask_mode: str | None):
    """{scale: dir} for every '<mask_mode>_s<scale>' folder under out_root."""
    found = {}
    for d in sorted(glob.glob(os.path.join(out_root, "*_s*"))):
        if not os.path.isdir(d):
            continue
        m = re.match(r"^(.*)_s(-?[\d.]+)$", os.path.basename(d))
        if not m:
            continue
        mode, s = m.group(1), float(m.group(2))
        if mask_mode and mode != mask_mode:
            continue
        found.setdefault(mode, {})[s] = d
    return found


def compare(ref: dict, dir_s: str):
    """(rot_deg, rot_max_deg, trans_m) per clip, over the clips both sets share.

    `ref` is the already-loaded s0 set: it is the same reference for every scale, so loading
    and 6D-decoding it per scale would repeat that work (n_scales - 1) times.
    """
    cur = load_dir(dir_s)
    keys = sorted(set(ref) & set(cur))
    rot, rmax, tr = [], [], []
    for k in keys:
        (rot_a, tr_a), (rot_b, tr_b) = ref[k], cur[k]
        if rot_a.shape != rot_b.shape:  # should not happen; skip rather than crash a scan
            continue
        d = geodesic_deg(rot_a, rot_b)                                   # (T, 22)
        rot.append(float(d.mean()))
        # max over joints, then mean over frames: "the most-moved joint, typically".
        rmax.append(float(d.max(dim=-1).values.mean()))
        tr.append(float(np.linalg.norm(tr_a - tr_b, axis=-1).mean()))
    return np.array(rot), np.array(rmax), np.array(tr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True,
                   help="Directory holding the <mask_mode>_s<scale> folders.")
    p.add_argument("--mask_mode", default=None, help="Only this mode (default: all found).")
    p.add_argument("--noop_deg", type=float, default=1.0,
                   help="Below this mean rotation change, call the edit a no-op.")
    p.add_argument("--out", default=None, help="Also write the table as JSON here.")
    add_logging_args(p)
    args = configure_logging(p.parse_args())

    modes = scale_dirs(args.out_root, args.mask_mode)
    if not modes:
        raise SystemExit(f"no <mask_mode>_s<scale> folders under {args.out_root}")

    report = {}
    for mode, by_scale in modes.items():
        if 0.0 not in by_scale:
            log.info(f"[{mode}] no s0 folder — scale 0 is the reference, skipping.")
            continue
        ref = load_dir(by_scale[0.0])          # loaded once, reused across every scale
        n_ref = len(ref)
        log.info(f"\n=== {mode}  ({n_ref} clips, reference = s0 = source reconstruction) ===")
        log.info(f"{'scale':>7} {'rot deg':>9} {'rotmax':>9} {'p90 max':>9} "
              f"{'trans m':>9} {'no-op':>7}")
        log.info("-" * 56)
        rows = {}
        for s in sorted(by_scale):
            if s == 0.0:
                continue
            rot, rmax, tr = compare(ref, by_scale[s])
            if not len(rot):
                log.info(f"{s:>7g}   (no shared clips)"); continue
            # "no-op" is judged on the MOST-MOVED joint, not the skeleton mean — a
            # body-part edit is supposed to leave 21 of 22 joints alone.
            noop = float((rmax < args.noop_deg).mean())
            log.info(f"{s:>7g} {rot.mean():>9.2f} {rmax.mean():>9.2f} "
                  f"{np.percentile(rmax, 90):>9.2f} {tr.mean():>9.4f} {noop:>6.0%}")
            rows[s] = {"rot_deg_mean": float(rot.mean()),
                       "rot_max_deg_mean": float(rmax.mean()),
                       "rot_max_deg_p90": float(np.percentile(rmax, 90)),
                       "trans_m_mean": float(tr.mean()),
                       "frac_noop": noop, "n_clips": int(len(rot))}
        report[mode] = rows

    # Data-derived suggestion, because a hardcoded one is wrong as often as it is right.
    for mode, rows in report.items():
        if not rows:
            continue
        live = [s for s in sorted(rows) if rows[s]["frac_noop"] < 0.5]
        top = max(rows)
        log.info()
        if not live:
            log.info(f"[{mode}] NOTHING here is doing an edit — even s={top:g} leaves the "
                  f"most-moved joint under {args.noop_deg:g} deg on most clips.")
            log.info(f"        Scan HIGHER before spending TMR: set SCAN_SCALES=\"0 {top:g} "
                  f"{top*2:g} {top*4:g}\" in eval_motionfix.sh's Config block and re-run.")
        else:
            lo = live[0]
            log.info(f"[{mode}] the edit starts moving the body around s={lo:g} "
                  f"(most-moved joint clears {args.noop_deg:g} deg on >50 % of clips).")
            hi = [s for s in sorted(rows) if s >= lo][:3]
            log.info(f"        Suggested SCALES for the scored run: \"0 "
                  f"{' '.join(f'{s:g}' for s in hi)}\"")

    log.info("\nRead `rotmax` first: it is the most-moved joint, so it survives a "
          "body-part edit that\nleaves the rest of the skeleton alone. `rot deg` is the "
          "skeleton mean and divides that\nby ~22. Both measure MAGNITUDE, never "
          "correctness — a big number is not a good edit.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
