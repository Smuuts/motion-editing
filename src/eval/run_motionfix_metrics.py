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
(`*_s2t`) and target↔generated, on batches-of-32 and on the whole gallery. It expects to run
from the MotionFix repo root (uses hydra.utils.get_original_cwd() + relative paths to
eval-deps/ and the test .pth.tar); we chdir there and stub get_original_cwd so it works
outside a hydra app.

⚠ **READ THIS BEFORE QUOTING A NUMBER — the gallery is whatever you generated.**
`retrieval()` builds its retrieval set as `MotionFixLoader(sets=['test'],
keys_to_load=<the keyids present in your dict>)` and scores against `dataset.keyids`
(tmr_evaluator/motion2motion_retr.py:441-463). So the second return value — what this project
used to call "the full test set" — is an **N-way retrieval where N is your file count**, not
1013-way. Retrieval gets monotonically easier as N falls, so:

  * a `--limit 320` run's R@k is inflated relative to the published MotionFix protocol and is
    NOT comparable to any number in that literature;
  * two configs with different file counts (e.g. `groups`, which skips the ~17 % of
    instructions it cannot route) are NOT comparable to each other — the one that skipped
    more gets a smaller gallery and a free boost.

The batches-of-32 protocol is fixed at 32-way whatever N is, so it IS comparable across
configs and against the paper. **Prefer it as the headline.** For a like-for-like comparison
of configs with different coverage, pass `--common_subset`, which restricts every directory
to the keyids all of them share before scoring.
"""

import os
import sys

# These scripts live one level below src/, so src/ is not on the path when they are run
# directly. Put it there before any project import.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import json

from utils.logger import add_logging_args, configure_logging, get_logger

log = get_logger(__name__)

# The published protocol's size, and the evaluator's batch size. Duplicated in
# aggregate_summary.py (which renders the warning) because that module runs in the PROJECT
# venv while this one runs in MotionFix's — they cannot import each other.
FULL_TEST_SET = 1013
MIN_BATCH = 32


def _numeric(metrics: dict) -> dict:
    """MotionFix returns its metrics as STRINGS ('71.88') — `all_contrastive_metrics_m2m`
    builds them by splitting a formatted LaTeX row. Coerce here, at the boundary where the
    foreign data enters, so tmr_metrics.json holds numbers and every downstream consumer is
    correct by default. Leaving it to the consumer already cost one silent failure: an
    `isinstance(v, float)` guard in the summary renderer matched nothing and its "best row"
    flag never fired."""
    out = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smpl_dir", action="append", required=True, dest="smpl_dirs",
                    help="Absolute path to a fitted-SMPL config dir (repeat for several).")
    ap.add_argument("--motionfix_dir", default=None,
                    help="MotionFix repo root (default: <repo>/data/motionfix).")
    ap.add_argument("--common_subset", action="store_true",
                    help="Score every --smpl_dir on the intersection of their keyids, so all "
                         "configs face an identically-sized gallery. Required for a fair "
                         "comparison whenever the configs skip different clips.")
    ap.add_argument("--out", default=None, help="Write the collected metrics as JSON here.")
    add_logging_args(ap)
    args = configure_logging(ap.parse_args())

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
    mfix = os.path.abspath(args.motionfix_dir or os.path.join(here, "data", "motionfix"))
    smpl_dirs = [os.path.abspath(d) for d in args.smpl_dirs]
    out_path = os.path.abspath(args.out) if args.out else None

    # Keyid sets first: whether the galleries match decides whether the numbers below can be
    # compared at all, and it is free to check before loading a GPU model.
    keysets = {}
    for d in smpl_dirs:
        keysets[d] = {f[:-4] for f in os.listdir(d) if f.endswith(".npy")}
    sizes = {len(v) for v in keysets.values()}
    common = set.intersection(*keysets.values())
    if len(sizes) > 1:
        action = ("Scoring on it (--common_subset given)." if args.common_subset
                  else "Pass --common_subset to score on it.")
        log.warning("the --smpl_dir sets have DIFFERENT clip counts (%s). Retrieval gets "
                    "easier as the gallery shrinks, so their 'full' R@k are not "
                    "comparable.\n         Intersection = %d clips. %s",
                    sorted(sizes), len(common), action)

    # Guard before the heavy import below: retrieval() batches with `range(len(keyids) // 32)`,
    # so under 32 it builds ZERO batches and dies on `result[0]` with an opaque IndexError from
    # inside their code. eval_motionfix.sh guards its own file count, but --common_subset can
    # drop the EFFECTIVE count below 32 after that check has already passed. Runs here, not in
    # the scoring loop, so dir 2's problem cannot surface only after dir 1 has been scored.
    for d in smpl_dirs:
        n_eff = len(common if args.common_subset else keysets[d])
        if n_eff < MIN_BATCH:
            raise SystemExit(
                f"\n{os.path.basename(d.rstrip('/'))} would be scored on {n_eff} clips, but "
                f"MotionFix's evaluator needs >= {MIN_BATCH}\n(it scores in batches of 32 and "
                f"crashes with IndexError on fewer). "
                + (f"The intersection of your --smpl_dir sets is only {len(common)} clips; "
                   f"drop --common_subset or\nregenerate the short config.\n"
                   if args.common_subset else "Generate more clips.\n"))

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
        keep = common if args.common_subset else keysets[d]
        # Load the plain (T,135) arrays into the {keyid: tensor} dict retrieval() also accepts
        # (collect_gen_samples else-branch) — avoids the evaluator's own np.load, which chokes on
        # cross-numpy-version pickles.
        gen = {k: torch.from_numpy(np.load(os.path.join(d, f"{k}.npy"))).float()
               for k in sorted(keep)}
        n = len(gen)
        log.section(f"{cfg}  ({n} generations -> {n}-way gallery)")
        metrs_batches, metrs_full = retrieval(gen)
        log.info("  batches-of-32 (32-way, COMPARABLE): %s", metrs_batches)
        log.info("  whole gallery (%d-way): %s", n, metrs_full)
        if n < FULL_TEST_SET:
            log.info(f"  NOTE: {n} < {FULL_TEST_SET} clips, so the gallery row is an {n}-way "
                  f"retrieval and reads HIGHER than the published 1013-way protocol. "
                  f"Quote the batches row.")
        # `n` IS the gallery size — there is no second notion of run size to distinguish it
        # from, and whether it is the full test set is one comparison the renderer can make.
        results[cfg] = {
            "n": n,
            "common_subset": bool(args.common_subset),
            "batches": _numeric(metrs_batches),
            "full": _numeric(metrs_full),
        }
        del gen        # ~80-160 MB; otherwise it stays alive while the next dir loads

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
