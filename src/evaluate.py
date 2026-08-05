"""
evaluate.py — FID, R-Precision, MM-Dist and Diversity with the T2M evaluator.

These use the pretrained T2M evaluator (Guo et al., 2022), so they are directly
comparable to MDM, MLD, MotionDiffuse and other HumanML3D papers. MPJPE is also
reported for internal tracking but is not a standard benchmark metric. Metric
definitions and the rep-aware decode live in src/eval/t2m_metrics.py.

Required files in --evaluator_dir:
  checkpoint/finest.tar              — text_mot_match/model/finest.tar
  glove/our_vab_{data.npy,words.pkl,idx.pkl}
  t2m/Comp_v6_KLD01/meta/{mean,std}.npy   — the evaluator's own normalisation

Usage:
    python src/evaluate.py --generated_dir generated/val --data_root data/HumanML3D \\
        --evaluator_dir data/t2m_evaluator --experiment_name val
"""

import os
import json
import argparse

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

from eval.t2m_metrics import (
    build_decoders, compute_diversity, compute_fid, compute_mm_dist,
    compute_r_precision, load_generated,
)
from model.t2m_eval import T2MEvaluator
from utils.cli import resolve_device
from utils.skeleton import mpjpe_from_joints


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_dir", required=True,
                   help="Directory written by generate.py (.npz files + manifest.json).")
    p.add_argument("--data_root",     required=True,
                   help="Processed HumanML3D root — texts/ (R-Precision), 263 Mean/Std, "
                        "and new_joints/ (smplh tgt-offset reference).")
    p.add_argument("--evaluator_dir", required=True,
                   help="Root of the T2M evaluator files (e.g. data/t2m_evaluator).")
    p.add_argument("--experiment_name", required=True,
                   help="Results go to <output_dir>/results_<experiment_name>.json.")
    p.add_argument("--smplh_feat_root", default=None,
                   help="smplh only: dir with the 135-d Mean/Std used to denormalise "
                        "(defaults to manifest['data_root']).")
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh",
                   help="smplh only: SMPLHLayer dir (needs SMPLH_NEUTRAL.npz).")
    p.add_argument("--eval_meta_dir", default=None,
                   help="Dir with the evaluator's own Mean/std.npy "
                        "(default: <evaluator_dir>/t2m/Comp_v6_KLD01/meta).")
    p.add_argument("--output_dir",    default="./eval_results")
    p.add_argument("--pool_size",     type=int,   default=32)
    p.add_argument("--diversity_times", type=int, default=300,
                   help="Number of random pairs for the Diversity metric.")
    p.add_argument("--smooth_sigma",  type=float, default=1.5)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def mpjpe_over_clips(clips, to_joints, smooth_sigma):
    """Per-clip root-relative MPJPE between the generated and GT decodes."""
    values = []
    for clip in tqdm(clips, desc="MPJPE"):
        joints = [to_joints(clip[k]) for k in ("gen_norm", "gt_norm")]
        if smooth_sigma > 0:
            joints = [gaussian_filter1d(j, sigma=smooth_sigma, axis=0) for j in joints]
        values.append(mpjpe_from_joints(*joints)[1])
    return np.array(values)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")

    print(f"\nLoading generated clips from {args.generated_dir} …")
    clips, manifest = load_generated(args.generated_dir, args.data_root)
    N = len(clips)
    print(f"  {N} clips")
    if N == 0:
        print("No clips found — nothing to report.")
        return

    print(f"\nLoading T2M evaluator from {args.evaluator_dir} …")
    evaluator = T2MEvaluator(
        checkpoint_path=os.path.join(args.evaluator_dir, "checkpoint", "finest.tar"),
        glove_dir=os.path.join(args.evaluator_dir, "glove"),
        device=device,
    )

    # Manifests from older runs lack feature_mode — infer it from the stored dim.
    feature_mode = ("smplh" if clips[0]["gen_norm"].shape[-1] == 135
                    else manifest.get("feature_mode", "humanml3d"))
    print(f"Feature mode: {feature_mode}")

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    to_raw263, to_joints = build_decoders(
        feature_mode, mean, std, args.data_root,
        smplh_feat_root=args.smplh_feat_root or manifest.get("data_root"),
        smplh_model_path=args.smplh_model_path)

    # The evaluator was trained with its OWN normalisation. Motions must be
    # denormalised to raw and renormalised with its stats before encoding, or
    # cross-modal alignment (R-Precision) collapses.
    eval_meta = args.eval_meta_dir or os.path.join(
        args.evaluator_dir, "t2m", "Comp_v6_KLD01", "meta")
    eval_mean = np.load(os.path.join(eval_meta, "mean.npy"))
    eval_std  = np.load(os.path.join(eval_meta, "std.npy"))

    def to_eval_norm(motion_norm):
        return ((to_raw263(motion_norm) - eval_mean) / eval_std).astype(np.float32)

    print("\n── MPJPE (internal, not comparable to literature) ──────────────────────────")
    mpjpe = mpjpe_over_clips(clips, to_joints, args.smooth_sigma)
    print(f"MPJPE   {mpjpe.mean()*1000:.2f} ± {mpjpe.std()*1000:.2f} mm  (N={N})")

    print("\n── Encoding motions and texts …")
    gen_embs = evaluator.encode_motion(
        tqdm([to_eval_norm(c["gen_norm"]) for c in clips], desc="  gen motions"))
    gt_embs = evaluator.encode_motion(
        tqdm([to_eval_norm(c["gt_norm"]) for c in clips], desc="  gt  motions"))
    text_embs = evaluator.encode_text([c["tokens"] for c in clips])
    print(f"  embedding dim: {gen_embs.shape[1]}")

    fid = compute_fid(gt_embs, gen_embs)
    print(f"\nFID     {fid:.4f}  (T2M feature space — comparable to MDM / MLD / MotionDiffuse)")

    effective_pool = min(args.pool_size, N)
    r_prec = compute_r_precision(gen_embs, text_embs, pool_size=effective_pool,
                                 seed=args.seed)
    for k, v in sorted(r_prec.items()):
        print(f"R-Prec@{k} {v:.4f}")

    mm_dist = compute_mm_dist(gen_embs, text_embs)
    print(f"MM-Dist {mm_dist:.4f}  (lower = better; distance to own caption)")

    div_gen = compute_diversity(gen_embs, args.diversity_times, seed=args.seed)
    div_gt  = compute_diversity(gt_embs,  args.diversity_times, seed=args.seed)
    print(f"Diversity (gen) {div_gen:.4f}   (real motions: {div_gt:.4f} — closer is better)")

    summary = {
        "generated_dir": args.generated_dir,
        "evaluator_dir": args.evaluator_dir,
        "feature_mode":  feature_mode,
        "n_clips":       N,
        "mpjpe_mean_mm": round(float(mpjpe.mean() * 1000), 4),
        "mpjpe_std_mm":  round(float(mpjpe.std() * 1000), 4),
        "fid":           round(fid, 6),
        "r_precision":   {f"top_{k}": round(v, 6) for k, v in r_prec.items()},
        "mm_dist":       round(mm_dist, 6),
        "diversity":     round(div_gen, 6),
        "diversity_gt":  round(div_gt, 6),
        "pool_size":     effective_pool,
        "diversity_times": min(args.diversity_times, N),
        "generation_args": {k: v for k, v in manifest.items() if k != "clip_ids"},
    }
    out_json = os.path.join(args.output_dir, f"results_{args.experiment_name}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults → {out_json}")


if __name__ == "__main__":
    main()
