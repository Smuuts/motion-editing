"""
Stage D of the MotionFix evaluation: run MotionFix's *own* TMR retrieval evaluator on our
SMPL-fitted generations, so the numbers are identical to the paper's.

MUST be run with the MotionFix venv interpreter, because `retrieval()` pulls in their full
hydra/einops/TMR stack:

    data/motionfix/mfix-env/bin/python src/eval/run_motionfix_metrics.py \
        --smpl_dir <abs path>/data/motionfix_smpl/m2_only_s5 \
        --smpl_dir <abs path>/data/motionfix_smpl/m2_only_s2.5 \
        --out eval_results/motionfix/tmr_metrics.json

`retrieval(samples_dir)` returns (metrs_batches, metrs_full) — R@1/2/3 for source↔generated
(`*_s2t`) and target↔generated, on batches-of-32 and the full test set. It expects to run from
the MotionFix repo root (uses hydra.utils.get_original_cwd() + relative paths to eval-deps/ and
the test .pth.tar); we chdir there and stub get_original_cwd so it works outside a hydra app.
"""

import os
import sys
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_dir", action="append", required=True, dest="smpl_dirs",
                    help="Absolute path to a fitted-SMPL config dir (repeat for several).")
    ap.add_argument("--motionfix_dir", default=None,
                    help="MotionFix repo root (default: <repo>/data/motionfix).")
    ap.add_argument("--out", default=None, help="Write the collected metrics as JSON here.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
    mfix = os.path.abspath(args.motionfix_dir or os.path.join(here, "data", "motionfix"))
    smpl_dirs = [os.path.abspath(d) for d in args.smpl_dirs]
    out_path = os.path.abspath(args.out) if args.out else None

    os.chdir(mfix)
    sys.path.insert(0, mfix)
    # retrieval() + MotionFixLoader call hydra.utils.get_original_cwd(); outside a hydra app that
    # raises, so point it at the repo root we just chdir'd into (where eval-deps/ + data/ live).
    import hydra.utils
    hydra.utils.get_original_cwd = lambda: mfix

    # NOTE: src.utils.file_io imports moviepy at module load, which needs pkg_resources — if this
    # import fails, run once: data/motionfix/mfix-env/bin/python -m pip install "setuptools<81".
    from tmr_evaluator.motion2motion_retr import retrieval
    import numpy as np
    import torch

    results = {}
    for d in smpl_dirs:
        cfg = os.path.basename(d.rstrip("/"))
        files = [f for f in os.listdir(d) if f.endswith(".npy")]
        # Load the plain (T,135) arrays into the {keyid: tensor} dict retrieval() also accepts
        # (collect_gen_samples else-branch) — avoids the evaluator's own np.load, which chokes on
        # cross-numpy-version pickles.
        gen = {f[:-4]: torch.from_numpy(np.load(os.path.join(d, f))).float() for f in files}
        print(f"\n===== {cfg}  ({len(gen)} generations) =====", flush=True)
        metrs_batches, metrs_full = retrieval(gen)
        n = len(gen)
        print("  batches-of-32:", metrs_batches)
        print("  full test set:", metrs_full)
        results[cfg] = {"n": n, "batches": metrs_batches, "full": metrs_full}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
