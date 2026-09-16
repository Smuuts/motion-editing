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

TWO METRIC SETS BEYOND R@1/2/3 (added 2026-09-16, EVALUATION.md §10 option 1)
----------------------------------------------------------------------------
R@1 has ~3.7 pp of usable range on the 32-way protocol (73.6 for copying the source vs 77.3
supervised SOTA), so an edit that moves the target from rank 8 to rank 3 scores identically to
one that does nothing. Two additions, neither of which can saturate:

1. **AvgR / MedR, recovered.** MotionFix's own evaluator computes R@1/2/3/5/10 + MedR + AvgR
   for both directions and then throws eight of them away in `retrieval()` via a hardcoded
   `names_to_keep` (motion2motion_retr.py:595). We wrap their `line2dict` to stash the full
   dict before that filter, so `batches`/`full` now carry all 14 metrics. **No vendored file is
   modified.** Average rank moves continuously as a partial edit shifts the target up the
   ranking, which is the regime this method operates in.

2. **Per-clip directional metrics (`--per_clip`).** The paired question R@k cannot ask: *for
   this clip, did editing move the motion CLOSER to its target than doing nothing did?*
   Baseline = the unedited source, obtained from the evaluator's own encoder by calling
   `compute_sim_matrix(..., gen_samples=None)`, whose `s_t` matrix is exactly
   sim(source_i, target_j). Reported as PIR (the share of clips that improved) with an exact
   two-sided sign test, plus a per-clip CSV so subsets and paired statistics can be computed
   later without re-running anything. Chance is 50 %, known without a baseline run, and each
   clip is its own control, so performer/length/difficulty cancel.

   ⚠ The two rank columns use DIFFERENT galleries by design — `rank_gen` ranks the true target
   among the *edited* motions, `rank_src` among the *unedited sources*. That is the honest
   editor-vs-do-nothing comparison (each system scored on its own outputs), and it is the same
   contrast the scale-0 row makes. `avgr_gen` here should agree with the recovered AvgR above;
   they are computed by different code paths, so a mismatch means one of them is wrong.
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


def _sign_test(k, n):
    """Exact two-sided binomial p-value for k successes in n trials under p = 0.5.

    Exact rather than normal-approximated because the interesting outcomes here sit near
    50 % on n ~ 1000, where the two agree — but also get run on subsets of a few dozen clips,
    where they do not. Ties are excluded by the caller, which is the standard sign test.
    """
    from math import comb
    if n == 0:
        return None
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _capture_all_metrics(module):
    """Context manager: stash every dict `line2dict` builds, before `retrieval()` filters it.

    `retrieval()` computes all 14 metrics, calls `line2dict` twice (batches, then full) and
    then keeps 6. Wrapping the function is enough to recover the rest and leaves the vendored
    file untouched, so a MotionFix update does not silently revert this.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        captured = []
        original = module.line2dict

        def wrapper(line):
            d = original(line)
            captured.append(d)
            return d

        module.line2dict = wrapper
        try:
            yield captured
        finally:
            module.line2dict = original

    return _cm()


def _directional(gen, mfix, device, log):
    """Per-clip paired metrics: did the edit move each clip toward its target, vs doing nothing?

    Uses the evaluator's OWN encoder and normalisation path (`collect_gen_samples` ->
    `compute_sim_matrix`) so these numbers sit in the same embedding space as the R@k above.
    The do-nothing baseline comes free: `compute_sim_matrix(..., gen_samples=None)` encodes
    source against target, so its `sim_matrix_s_t` is sim(source_i, target_j).
    """
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from pathlib import Path

    from tmr_evaluator.motion2motion_retr import (collect_gen_samples,
                                                  compute_sim_matrix, read_config)
    from src.tmr.data.motionfix_loader import MotionFixLoader, Normalizer
    from src.tmr.load_model import load_model_from_cfg

    # Explicit, rather than going through the get_original_cwd() stub main() installs — this
    # function has the path already and should not depend on that patch still being in place.
    curdir, run_dir = Path(mfix), "eval-deps"
    cfg = read_config(curdir / run_dir)
    pl.seed_everything(cfg.seed)
    model = load_model_from_cfg(cfg, "last", eval_mode=True, device=device)
    normalizer = Normalizer(curdir / run_dir / "stats/humanml3d/amass_feats")

    gen_samples, _ = collect_gen_samples(gen, normalizer, model.device)
    dataset = MotionFixLoader(sets=["test"], keys_to_load=list(gen_samples.keys()))
    gen_samples = {k: v for k, v in gen_samples.items() if k in dataset.motions.keys()}
    keyids = sorted(dataset.keyids)

    # Same keyid list to both calls, so row i means the same clip in every matrix below.
    res_gen, keys_gen = compute_sim_matrix(model, dataset, keyids,
                                           gen_samples=gen_samples, progress=False)
    res_non, keys_non = compute_sim_matrix(model, dataset, keyids,
                                           gen_samples=None, progress=False)
    if keys_gen["t_t"] != keys_non["s_t"]:
        raise RuntimeError("keyid order differs between the edited and baseline passes; "
                           "the per-clip pairing would be wrong")
    order = keys_gen["t_t"]

    S_tg = np.asarray(res_gen["sim_matrix_t_t"])   # [i,j] = sim(target_i, gen_j)
    S_sg = np.asarray(res_gen["sim_matrix_s_t"])   # [i,j] = sim(source_i, gen_j)
    S_ts = np.asarray(res_non["sim_matrix_s_t"])   # [i,j] = sim(source_i, target_j)
    n = len(order)

    d = np.diag_indices(n)
    sim_gen_target, sim_src_target, sim_gen_source = S_tg[d], np.diag(S_ts), S_sg[d]
    delta_sim = sim_gen_target - sim_src_target

    # Competition rank (1-based) of the true match. Query = target in both cases; the gallery
    # is the edited set for one and the unedited sources for the other.
    rank_gen = (S_tg > sim_gen_target[:, None]).sum(1) + 1
    rank_src = (S_ts.T > sim_src_target[:, None]).sum(1) + 1
    rank_delta = rank_src - rank_gen

    rows = [{"keyid": k,
             "sim_gen_target": float(sim_gen_target[i]),
             "sim_src_target": float(sim_src_target[i]),
             "delta_sim": float(delta_sim[i]),
             "improved": int(delta_sim[i] > 0),
             "sim_gen_source": float(sim_gen_source[i]),
             "rank_gen": int(rank_gen[i]),
             "rank_src": int(rank_src[i]),
             "rank_delta": int(rank_delta[i])} for i, k in enumerate(order)]

    n_up, n_tied = int((delta_sim > 0).sum()), int((delta_sim == 0).sum())
    r_up, r_tied = int((rank_delta > 0).sum()), int((rank_delta == 0).sum())
    agg = {
        "n": n,
        # PIR: the headline. Share of clips the edit moved closer to the target than the
        # source already was. 50 % is chance, and needs no baseline run to establish.
        "pir_sim": 100.0 * n_up / n,
        "pir_sim_p": _sign_test(n_up, n - n_tied),
        "n_tied_sim": n_tied,
        "mean_delta_sim": float(delta_sim.mean()),
        "median_delta_sim": float(np.median(delta_sim)),
        "se_delta_sim": float(delta_sim.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        # Ranks are discrete, so ties are common and MEANINGFUL — a tie says the edit did not
        # change the retrieval outcome for that clip. Counting them in the denominator would
        # report a perfect do-nothing reconstruction as "0 % improved" when the honest answer
        # is "no clip moved". So this ratio is over clips that moved, with the tie count beside
        # it. (pir_sim keeps the all-clips denominator: similarities are continuous, ties there
        # are float noise, and that is the published definition.) Expect many rank ties on a
        # 32-way gallery and few on the 1013-way one.
        "pir_rank": (100.0 * r_up / (n - r_tied)) if n - r_tied else None,
        "pir_rank_p": _sign_test(r_up, n - r_tied),
        "n_tied_rank": r_tied,
        "n_moved_rank": n - r_tied,
        "avgr_gen": float(rank_gen.mean()), "medr_gen": float(np.median(rank_gen)),
        "avgr_src": float(rank_src.mean()), "medr_src": float(np.median(rank_src)),
        "mean_sim_gen_source": float(sim_gen_source.mean()),
        "mean_sim_gen_target": float(sim_gen_target.mean()),
        "mean_sim_src_target": float(sim_src_target.mean()),
    }
    log.info("  PIR (similarity): %.2f %% of %d clips improved  (p = %s, chance 50 %%)",
             agg["pir_sim"], n, f"{agg['pir_sim_p']:.3g}" if agg["pir_sim_p"] is not None else "n/a")
    log.info("  PIR (rank):       %s of the %d clips whose rank moved (%d tied)  |  "
             "AvgR edited %.2f vs do-nothing %.2f",
             f"{agg['pir_rank']:.2f} %" if agg["pir_rank"] is not None else "n/a",
             n - r_tied, r_tied, agg["avgr_gen"], agg["avgr_src"])
    del model
    torch.cuda.empty_cache()   # a second TMR sits beside retrieval()'s for the next config
    return agg, rows


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
    ap.add_argument("--per_clip", action="store_true",
                    help="Also compute the per-clip directional metrics (PIR + rank change vs "
                         "the unedited source) and write a CSV per config. Costs one extra "
                         "encoding pass and a second TMR model load.")
    ap.add_argument("--per_clip_dir", default=None,
                    help="Where the per-clip CSVs go (default: alongside --out, in per_clip/).")
    ap.add_argument("--no_retrieval", action="store_true",
                    help="Skip MotionFix's retrieval() entirely and compute only --per_clip. "
                         "Use when re-scoring purely for the directional metrics.")
    ap.add_argument("--out", default=None, help="Write the collected metrics as JSON here.")
    add_logging_args(ap)
    args = configure_logging(ap.parse_args())

    if args.no_retrieval and not args.per_clip:
        raise SystemExit("--no_retrieval with no --per_clip would compute nothing.")

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
    import tmr_evaluator.motion2motion_retr as M
    from tmr_evaluator.motion2motion_retr import retrieval
    import numpy as np
    import torch

    per_clip_dir = None
    if args.per_clip:
        per_clip_dir = os.path.abspath(
            args.per_clip_dir
            or (os.path.join(os.path.dirname(out_path), "per_clip") if out_path
                else os.path.join(here, "eval_results", "motionfix", "per_clip")))
        os.makedirs(per_clip_dir, exist_ok=True)

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
        metrs_batches, metrs_full = {}, {}
        if not args.no_retrieval:
            # The capture gives us AvgR/MedR/R@5/R@10 that retrieval() computes and discards.
            # line2dict is called twice per retrieval(): batches first, then the full gallery.
            with _capture_all_metrics(M) as captured:
                metrs_batches, metrs_full = retrieval(gen)
            if len(captured) == 2:
                metrs_batches, metrs_full = captured[0], captured[1]
            else:
                log.warning("expected 2 line2dict calls, saw %d — keeping the 6 filtered "
                            "metrics only (AvgR/MedR unavailable for %s)", len(captured), cfg)
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

        if args.per_clip:
            import csv
            agg, rows = _directional(gen, mfix, "cuda", log)
            results[cfg]["directional"] = agg
            csv_path = os.path.join(per_clip_dir, f"{cfg}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            log.info("  per-clip -> %s", csv_path)
            # Cross-check: the same quantity from two independent code paths. Only possible
            # when the gallery row was computed, and only meaningful on an N-way gallery.
            avgr = _numeric(metrs_full).get("AvgR")
            if isinstance(avgr, float) and abs(avgr - agg["avgr_gen"]) > 0.5:
                log.warning("AvgR disagrees between the evaluator (%.2f) and the per-clip "
                            "pass (%.2f) for %s — one of them is wrong, do not quote either "
                            "until it is resolved.", avgr, agg["avgr_gen"], cfg)

        del gen        # ~80-160 MB; otherwise it stays alive while the next dir loads

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
