"""
Are MotionFix and HumanML3D compatible — as motion, and as language?

Produces the two-panel compatibility figure plus the numbers behind it.

  MOTION panel   TMR motion embeddings (the space MotionFix scores retrieval in).
  TEXT panel     pooled T5 embeddings (the space this backbone actually conditions on).

Both panels carry three series: HumanML3D train, HumanML3D HELD-OUT, and MotionFix test.
The held-out series is the control that makes the result falsifiable — a classifier
two-sample test has to be shown failing on two samples of the same distribution before its
success on MotionFix means anything.

Clip length is matched on the motion panel (nearest frame count, without replacement).
TMR pools over time, MotionFix test clips run 2-5 s and HumanML3D reaches ~10 s, and an
unmatched comparison would report "incompatible" about duration rather than about motion.

Usage
-----
    python src/probe_dataset_compat.py --panels both --out eval_results/dataset_compat
    python src/probe_dataset_compat.py --panels text --n_train 3000     # text only, ~1 min
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from analysis.dataset_compat import (Series, block_decomposition, embed_2d,
                                     encode_motions,
                                     load_hml3d_captions, load_hml3d_motions,
                                     load_motionfix_instructions,
                                     load_motionfix_motions,
                                     match_length_distribution, nn_cosine,
                                     pool_text, two_sample)
from utils.cli import add_logging_args, configure_logging, resolve_device
from utils.logger import get_logger
from utils.paths import resolve_repo_path
from utils.visualise import plot_dataset_compat, plot_perplexity_sweep

log = get_logger(__name__)

TRAIN, CONTROL, FOREIGN = "HumanML3D train", "HumanML3D held-out", "MotionFix"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D_smplh",
                   help="SMPL-H dataset root (new_joint_vecs/ + texts/ + split .txt).")
    p.add_argument("--motionfix_root", default="data/motionfix")
    p.add_argument("--panels", default="both", choices=["both", "text", "motion"])
    p.add_argument("--n_train", type=int, default=3000,
                   help="HumanML3D train samples (3000 matches the recorded T5 measurement).")
    p.add_argument("--n_control", type=int, default=0,
                   help="Held-out HumanML3D samples. 0 = match the MotionFix count.")
    p.add_argument("--control_split", default="val+test",
                   help="Held-out split(s), '+'-joined. val+test together, because test\n                        alone is too small to cover MotionFix's length range.")
    p.add_argument("--keep_mirrored", action="store_true",
                   help="Keep HumanML3D's 'M'-prefixed mirror copies (near-duplicates).")
    p.add_argument("--no_match_lengths", action="store_true",
                   help="Motion panel: skip clip-length matching (ablation — see module doc).")
    p.add_argument("--raw_heading", action="store_true",
                   help="Motion panel: keep each clip's capture heading instead of "
                        "canonicalising it (ablation — heading is a property of the mocap "
                        "session, not of the motion).")
    # text encoder — must be the pair the backbone was conditioned with
    p.add_argument("--t5_version", default="t5-base")
    p.add_argument("--t5_max_length", type=int, default=128)
    p.add_argument("--text_batch", type=int, default=64)
    # motion / TMR
    p.add_argument("--src_fps", type=float, default=20.0, help="HumanML3D dataset fps.")
    p.add_argument("--tmr_fps", type=float, default=30.0, help="TMR's native fps.")
    p.add_argument("--motion_batch", type=int, default=32)
    # t-SNE
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--sweep", default="5,30,50",
                   help="Perplexity sweep for the companion figure; '' to skip.")
    p.add_argument("--tsne_equal_n", action="store_true", default=True,
                   help="Subsample every series to the smallest for the SCATTER only, so "
                        "cloud density is comparable. Statistics always use the full N.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval_results/dataset_compat")
    p.add_argument("--device", default=None)
    add_logging_args(p)
    return configure_logging(p.parse_args())


def _report(name, res):
    log.info("  %-22s AUC %.4f | acc %5.1f %% | kNN %5.1f %% (chance %.0f %%) | "
             "centroid cos %.3f, sep %.3f vs spreads %.3f / %.3f",
             name, res["auc"], 100 * res["acc"], 100 * res["knn_acc"],
             100 * res["chance"], res["centroid_cos"], res["centroid_sep"],
             res["spread_a"], res["spread_b"])


def _panel(space, series, args, note=""):
    """Shared tail of both panels: NN-cosine, two-sample test, t-SNE."""
    train, control, foreign = series
    nn = {CONTROL: nn_cosine(control.X, train.X),
          FOREIGN: nn_cosine(foreign.X, train.X)}
    for k, v in nn.items():
        log.info("  NN cosine to %s: %-20s mean %.3f  median %.3f",
                 TRAIN, k, float(v.mean()), float(np.median(v)))
    # 5th percentile of the control is the "a held-out caption always has a close
    # neighbour" reference line the foreign corpus is scored against.
    p5 = float(np.percentile(nn[CONTROL], 5))
    below = float((nn[FOREIGN] < p5).mean())
    log.info("  %.1f %% of %s below the 5th percentile of %s (%.3f)",
             100 * below, FOREIGN, CONTROL, p5)

    probe = two_sample(control.X, foreign.X, seed=args.seed)
    _report(f"{CONTROL} vs {FOREIGN}", probe)
    # Control-vs-control: the same test on two samples that ARE the same distribution.
    half = len(train) // 2
    null = two_sample(train.X[:half], train.X[half:2 * half], seed=args.seed)
    _report("train vs train (null)", null)

    coords = embed_2d(series, perplexity=args.perplexity, seed=args.seed)
    return {"space": space, "series": series, "coords": coords, "nn": nn,
            "probe": probe, "null": null, "note": note,
            "control_name": CONTROL, "foreign_name": FOREIGN,
            "nn_p5_control": p5, "frac_foreign_below_p5": below}


def _equal_n(series, coords, seed):
    """Subsample every series to the smallest, for the scatter only."""
    rng = np.random.default_rng(seed)
    m = min(len(s) for s in series)
    out_s, out_c = [], []
    for s, xy in zip(series, coords):
        idx = np.sort(rng.choice(len(s), m, replace=False)) if len(s) > m else np.arange(len(s))
        out_s.append(Series(s.name, s.X[idx], [s.ids[i] for i in idx] if s.ids else []))
        out_c.append(xy[idx])
    return out_s, out_c


def text_panel(args, device):
    log.info("TEXT panel — pooled %s, the space the backbone conditions on", args.t5_version)
    root = resolve_repo_path(args.data_root)
    mfix = load_motionfix_instructions(resolve_repo_path(args.motionfix_root))
    n_control = args.n_control or len(mfix)

    train = load_hml3d_captions(root, "train", args.n_train, args.seed,
                                exclude_mirrored=not args.keep_mirrored)
    control = load_hml3d_captions(root, args.control_split, n_control, args.seed + 1,
                                  exclude_mirrored=not args.keep_mirrored)
    log.info("  %d train / %d held-out captions, %d MotionFix instructions",
             len(train), len(control), len(mfix))
    log.info('  example caption:     "%s"', train[0][1])
    log.info('  example instruction: "%s"', mfix[0][1])

    from model.text_encoder import build_text_encoder
    encoder = build_text_encoder(
        {"text_encoder": "t5", "t5_version": args.t5_version,
         "t5_max_length": args.t5_max_length}, device)

    series = []
    for name, items in ((TRAIN, train), (CONTROL, control), (FOREIGN, mfix)):
        X = pool_text(encoder, [t for _, t in items], args.text_batch, device)
        series.append(Series(name, X, [i for i, _ in items]))
    del encoder
    torch.cuda.empty_cache()
    return _panel(f"text — pooled {args.t5_version}", series, args,
                  note="captions vs edit instructions: different linguistic register")


def motion_panel(args, device):
    log.info("MOTION panel — TMR motion encoder (eval-deps/last_weights)")
    root = resolve_repo_path(args.data_root)
    mfix = load_motionfix_motions(resolve_repo_path(args.motionfix_root))
    n_control = args.n_control or len(mfix)

    # MotionFix is stored at TMR's own 30 fps; HumanML3D at 20. Matching happens in
    # HumanML3D's units so a pool clip is compared against what it will become.
    tgt = np.array([len(f) for _, f in mfix]) * args.src_fps / args.tmr_fps

    def pick(split, n, seed):
        """n HumanML3D clips whose length distribution matches MotionFix's."""
        if args.no_match_lengths:
            return load_hml3d_motions(root, split, n, seed,
                                      exclude_mirrored=not args.keep_mirrored), None
        # 0 = read the whole split; stratified sampling needs the full pool to draw from.
        pool = load_hml3d_motions(root, split, 10 ** 9, seed,
                                  exclude_mirrored=not args.keep_mirrored)
        return match_length_distribution(pool, tgt, n, seed)

    train, r_train = pick("train", args.n_train, args.seed)
    control, r_ctrl = pick(args.control_split, n_control, args.seed + 1)
    log.info("  %d train / %d held-out clips, %d MotionFix sources",
             len(train), len(control), len(mfix))
    for tag, d in (("train", r_train), ("held-out", r_ctrl)):
        if d is not None:
            log.info("  %-9s length match @%.0f fps: mean %.1f vs target %.1f, "
                     "median %.0f vs %.0f (shortfall %d)", tag, args.src_fps,
                     d["got_mean"], d["target_mean"], d["got_median"], d["target_median"],
                     d["shortfall"])

    from probe_tmr_laterality import load_tmr        # the plain-torch TMR rebuild
    motion_enc, _text_enc, stats, _t2e = load_tmr(device)

    series, prepared = [], {}
    for name, clips, fps in ((TRAIN, train, args.src_fps),
                             (CONTROL, control, args.src_fps),
                             (FOREIGN, mfix, args.tmr_fps)):
        X, lengths, feats = encode_motions(clips, motion_enc, stats, device, fps,
                                           args.tmr_fps, args.motion_batch,
                                           canonical=not args.raw_heading)
        log.info("  %-20s frames @%.0f fps: mean %.1f, median %.0f",
                 name, args.tmr_fps, lengths.mean(), np.median(lengths))
        series.append(Series(name, X, [i for i, _ in clips], lengths))
        prepared[name] = feats
    del motion_enc, _text_enc, _t2e
    torch.cuda.empty_cache()

    # Where does the separation actually live? Read it off the interpretable features,
    # at the fps and heading convention TMR was given, before any embedding is involved.
    blocks = block_decomposition(prepared[CONTROL], prepared[FOREIGN], seed=args.seed)
    log.info("  feature-block decomposition (%s vs %s):", CONTROL, FOREIGN)
    for k, v in blocks.items():
        log.info("    %-14s AUC %.4f | acc %5.1f %%", k, v["auc"], 100 * v["acc"])

    note = ("length-matched; heading canonicalised" if not args.raw_heading
            else "⚠ raw capture heading")
    if args.no_match_lengths:
        note = "⚠ lengths NOT matched"
    panel = _panel("motion — TMR", series, args, note=note)
    panel["blocks"] = blocks
    return panel


def main():
    args = parse_args()
    out = resolve_repo_path(args.out)
    os.makedirs(out, exist_ok=True)
    device = resolve_device(args.device)
    log.info("Device: %s\n", device)

    panels = []
    if args.panels in ("both", "motion"):
        panels.append(motion_panel(args, device))
    if args.panels in ("both", "text"):
        panels.append(text_panel(args, device))

    fig_series = [p["series"] for p in panels]
    fig_coords = [p["coords"] for p in panels]
    if args.tsne_equal_n:
        for i, p in enumerate(panels):
            fig_series[i], fig_coords[i] = _equal_n(p["series"], p["coords"], args.seed)
    fig_panels = [dict(p, series=s, coords=c)
                  for p, s, c in zip(panels, fig_series, fig_coords)]
    plot_dataset_compat(fig_panels, os.path.join(out, "dataset_compat.png"),
                        perplexity=args.perplexity)

    if args.sweep:
        for p in panels:
            perps = [float(x) for x in args.sweep.split(",") if x.strip()]
            coords = {q: embed_2d(p["series"], perplexity=q, seed=args.seed) for q in perps}
            if args.tsne_equal_n:
                coords = {q: _equal_n(p["series"], c, args.seed)[1] for q, c in coords.items()}
            series = _equal_n(p["series"], p["coords"], args.seed)[0] if args.tsne_equal_n \
                else p["series"]
            tag = p["space"].split()[0]
            plot_perplexity_sweep(p["space"], series, coords,
                                  os.path.join(out, f"tsne_perplexity_{tag}.png"))

    summary = {"config": {k: v for k, v in vars(args).items() if k != "log_level"},
               "panels": []}
    for p in panels:
        d = {"space": p["space"],
             "series": {s.name: len(s) for s in p["series"]},
             "nn_cosine": {k: {"mean": float(v.mean()), "median": float(np.median(v))}
                           for k, v in p["nn"].items()},
             "nn_p5_control": p["nn_p5_control"],
             "frac_foreign_below_p5": p["frac_foreign_below_p5"]}
        for key in ("probe", "null"):
            d[key] = {k: v for k, v in p[key].items() if k not in ("margin", "labels")}
        if "blocks" in p:
            d["feature_blocks"] = p["blocks"]
        summary["panels"].append(d)
    path = os.path.join(out, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
