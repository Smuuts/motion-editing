"""
evaluate.py — compute FID and R-Precision using the T2M evaluator.

FID and R-Precision use the pretrained T2M evaluator (Guo et al., 2022) and
are directly comparable to MDM, MLD, MotionDiffuse, and other HumanML3D papers.
MPJPE is also reported for internal tracking but is not a standard benchmark metric.

Required files in --evaluator_dir:
  checkpoint/finest.tar              — text_mot_match/model/finest.tar
  glove/our_vab_data.npy             — GloVe word vectors
  glove/our_vab_words.pkl
  glove/our_vab_idx.pkl
  t2m/Comp_v6_KLD01/meta/mean.npy    — evaluator's own normalisation
  t2m/Comp_v6_KLD01/meta/std.npy

Text is read from the HumanML3D text files' pre-tagged `word/POS` tokens —
no spaCy / re-tokenisation needed (that would shift the text embeddings).

Usage:
    python src/evaluate.py \\
        --generated_dir generated/val \\
        --data_root     data/HumanML3D \\
        --evaluator_dir data/t2m_evaluator \\
        --experiment_name val \\
        [--output_dir   eval_results] \\
        [--pool_size 32] \\
        [--smooth_sigma 1.5] \\
        [--seed 42]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")

from model.t2m_eval import T2MEvaluator
from utils.visualise import recover_from_ric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_dir", required=True,
                   help="Directory written by generate.py (contains .npz files + manifest.json).")
    p.add_argument("--data_root",     required=True,
                   help="HumanML3D root — for Mean/Std (MPJPE) and texts/ (R-Precision).")
    p.add_argument("--evaluator_dir", required=True,
                   help="Root of T2M evaluator files (e.g. data/t2m_evaluator).")
    p.add_argument("--experiment_name", required=True,
                   help="Experiment name — results are written to "
                        "<output_dir>/results_<experiment_name>.json.")
    p.add_argument("--eval_meta_dir", default=None,
                   help="Dir with the T2M evaluator's own Mean/std.npy "
                        "(default: <evaluator_dir>/t2m/Comp_v6_KLD01/meta).")
    p.add_argument("--output_dir",    default="./eval_results")
    p.add_argument("--pool_size",     type=int,   default=32)
    p.add_argument("--diversity_times", type=int, default=300,
                   help="Number of random pairs for the Diversity metric.")
    p.add_argument("--smooth_sigma",  type=float, default=1.5)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def load_generated(generated_dir, data_root):
    manifest_path = os.path.join(generated_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    text_dir = os.path.join(data_root, "texts")
    clips = []
    for cid in manifest["clip_ids"]:
        npz_path  = os.path.join(generated_dir, f"{cid}.npz")
        text_path = os.path.join(text_dir, f"{cid}.txt")
        if not os.path.exists(npz_path):
            print(f"  [WARN] missing {npz_path}, skipping")
            continue
        if not os.path.exists(text_path):
            print(f"  [WARN] missing text for {cid}, skipping")
            continue

        data = np.load(npz_path)
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        # HumanML3D format: caption#word/POS word/POS ...#start#end
        # Use the pre-tagged tokens (field 1) — same preprocessing the T2M
        # evaluator was trained on. Re-tokenising would shift the embeddings.
        parts  = lines[0].split("#")
        text   = parts[0].strip()
        tokens = parts[1].strip().split() if len(parts) > 1 and parts[1].strip() else []

        clips.append({
            "id":       cid,
            "gen_norm": data["gen_norm"],   # (T, 263)
            "gt_norm":  data["gt_norm"],    # (T, 263)
            "text":     text,
            "tokens":   tokens,
            "T":        int(data["T"]),
        })
    return clips, manifest


# ── MPJPE ─────────────────────────────────────────────────────────────────────

def mpjpe_from_joints(joints_gen, joints_gt):
    T = min(len(joints_gen), len(joints_gt))
    gen = joints_gen[:T] - joints_gen[:T, 0:1]
    gt  = joints_gt[:T]  - joints_gt[:T,  0:1]
    return float(np.sqrt(((gen - gt) ** 2).sum(axis=-1)).mean(axis=-1).mean())


# ── FID ───────────────────────────────────────────────────────────────────────

def compute_fid(real_feats, gen_feats):
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sig_r = np.cov(real_feats.T)
    sig_g = np.cov(gen_feats.T)
    diff    = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


# ── R-Precision ───────────────────────────────────────────────────────────────

def compute_r_precision(motion_embs, text_embs, pool_size=32, top_k=(1, 2, 3), seed=0):
    N = len(motion_embs)
    counts = {k: 0 for k in top_k}
    rng = np.random.default_rng(seed)

    for i in tqdm(range(N), desc="R-Precision", leave=False):
        neg_pool = [j for j in range(N) if j != i]
        n_neg    = min(pool_size - 1, len(neg_pool))
        neg_idx  = rng.choice(neg_pool, size=n_neg, replace=False)
        pool_idx = np.concatenate([[i], neg_idx])

        m_emb  = motion_embs[i]        # (512,)
        t_embs = text_embs[pool_idx]   # (pool, 512)
        dists  = np.sqrt(((m_emb - t_embs) ** 2).sum(axis=-1))
        rank   = int(np.where(np.argsort(dists) == 0)[0][0]) + 1

        for k in top_k:
            if rank <= k:
                counts[k] += 1

    return {k: counts[k] / N for k in top_k}


# ── Multimodal Distance ─────────────────────────────────────────────────────────

def compute_mm_dist(motion_embs, text_embs):
    """Mean Euclidean distance between each motion and its own caption embedding.
    Lower = better text-motion alignment (R-Precision's continuous counterpart)."""
    return float(np.sqrt(((motion_embs - text_embs) ** 2).sum(axis=-1)).mean())


# ── Diversity ───────────────────────────────────────────────────────────────────

def compute_diversity(motion_embs, diversity_times=300, seed=0):
    """Mean distance between randomly paired motion embeddings.
    Detects mode collapse — should be close to the ground-truth diversity."""
    n     = len(motion_embs)
    times = min(diversity_times, n)
    rng   = np.random.default_rng(seed)
    first  = rng.choice(n, times, replace=False)
    second = rng.choice(n, times, replace=False)
    return float(np.sqrt(((motion_embs[first] - motion_embs[second]) ** 2).sum(axis=-1)).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))

    # The T2M evaluator was trained with its OWN normalisation, which differs
    # from the HumanML3D training Mean/Std. Motions must be denormalised to raw
    # then renormalised with the evaluator's stats before encoding, otherwise
    # cross-modal alignment (R-Precision) collapses.
    eval_meta = args.eval_meta_dir or os.path.join(
        args.evaluator_dir, "t2m", "Comp_v6_KLD01", "meta")
    eval_mean = np.load(os.path.join(eval_meta, "mean.npy"))
    eval_std  = np.load(os.path.join(eval_meta, "std.npy"))

    def to_eval_norm(motion_hml_norm):
        raw = motion_hml_norm * std + mean
        return ((raw - eval_mean) / eval_std).astype(np.float32)

    # ── MPJPE (non-standard, internal tracking only) ───────────────────────────
    print("\n── MPJPE (internal, not comparable to literature) ──────────────────────────")
    mpjpe_values = []
    for clip in tqdm(clips, desc="MPJPE"):
        T       = clip["T"]
        gen_raw = clip["gen_norm"] * std + mean
        gt_raw  = clip["gt_norm"]  * std + mean
        joints_gen = recover_from_ric(gen_raw, joints_num=22)
        joints_gt  = recover_from_ric(gt_raw,  joints_num=22)
        if args.smooth_sigma > 0:
            joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)
            joints_gt  = gaussian_filter1d(joints_gt,  sigma=args.smooth_sigma, axis=0)
        mpjpe_values.append(mpjpe_from_joints(joints_gen, joints_gt))

    mpjpe_arr = np.array(mpjpe_values)
    print(f"MPJPE   {mpjpe_arr.mean()*1000:.2f} ± {mpjpe_arr.std()*1000:.2f} mm  (N={N})")

    # ── T2M embeddings ────────────────────────────────────────────────────────
    print("\n── Encoding motions and texts …")
    gen_motions = [to_eval_norm(c["gen_norm"]) for c in clips]
    gt_motions  = [to_eval_norm(c["gt_norm"])  for c in clips]
    token_lists = [c["tokens"] for c in clips]

    gen_embs  = evaluator.encode_motion(tqdm(gen_motions, desc="  gen motions"))
    gt_embs   = evaluator.encode_motion(tqdm(gt_motions,  desc="  gt  motions"))
    text_embs = evaluator.encode_text(token_lists)
    print(f"  embedding dim: {gen_embs.shape[1]}")

    # ── FID ───────────────────────────────────────────────────────────────────
    print("\n── FID ─────────────────────────────────────────────────────────────────────")
    fid = compute_fid(gt_embs, gen_embs)
    print(f"FID     {fid:.4f}  (T2M feature space — comparable to MDM / MLD / MotionDiffuse)")

    # ── R-Precision ───────────────────────────────────────────────────────────
    print("\n── R-Precision ─────────────────────────────────────────────────────────────")
    effective_pool = min(args.pool_size, N)
    r_prec = compute_r_precision(gen_embs, text_embs,
                                  pool_size=effective_pool, seed=args.seed)
    for k, v in sorted(r_prec.items()):
        print(f"R-Prec@{k} {v:.4f}")

    # ── Multimodal Distance ────────────────────────────────────────────────────
    print("\n── Multimodal Distance ─────────────────────────────────────────────────────")
    mm_dist = compute_mm_dist(gen_embs, text_embs)
    print(f"MM-Dist {mm_dist:.4f}  (lower = better; distance to own caption)")

    # ── Diversity ──────────────────────────────────────────────────────────────
    print("\n── Diversity ───────────────────────────────────────────────────────────────")
    div_gen = compute_diversity(gen_embs, args.diversity_times, seed=args.seed)
    div_gt  = compute_diversity(gt_embs,  args.diversity_times, seed=args.seed)
    print(f"Diversity (gen) {div_gen:.4f}   (real motions: {div_gt:.4f} — closer is better)")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "generated_dir": args.generated_dir,
        "evaluator_dir": args.evaluator_dir,
        "n_clips":       N,
        "mpjpe_mean_mm": round(float(mpjpe_arr.mean() * 1000), 4),
        "mpjpe_std_mm":  round(float(mpjpe_arr.std()  * 1000), 4),
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
