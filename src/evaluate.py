"""
evaluate.py — compute MPJPE, FID, and R-Precision from saved generations.

Reads the .npz files written by generate.py and computes three metrics:

  MPJPE        Root-relative mean per-joint position error (mm).
  FID          Fréchet distance in mean-pooled normalised motion feature space.
  R-Precision  NLL-based ranking of 1 true text vs (pool_size-1) random negatives,
               scored against ground-truth motions to avoid self-referential bias.

Usage:
    python src/evaluate.py \\
        --generated_dir generated/val \\
        --checkpoint    runs/exp_hml3d/checkpoint_latest \\
        --data_root     data/HumanML3D \\
        [--output_dir   eval_results/val] \\
        [--pool_size 32] \\
        [--n_eval_t 4] \\
        [--smooth_sigma 1.5] \\
        [--seed 42] \\
        [--no_ema]
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

from model.dit import build_model
from model.schedule import NoiseSchedule
from utils.visualise import recover_from_ric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_dir", required=True,
                   help="Directory written by generate.py (contains .npz files + manifest.json).")
    p.add_argument("--checkpoint",    required=True,
                   help="Checkpoint directory — used to load the model for R-Precision scoring.")
    p.add_argument("--data_root",     required=True,
                   help="HumanML3D root — used for Mean.npy / Std.npy (MPJPE denormalisation).")
    p.add_argument("--output_dir",    default="./eval_results/evaluate")
    p.add_argument("--pool_size",     type=int, default=32)
    p.add_argument("--n_eval_t",      type=int, default=4)
    p.add_argument("--smooth_sigma",  type=float, default=1.5)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_ema",        action="store_true")
    return p.parse_args()


def load_model(ckpt_dir, device, use_ema=True):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)

    context_dim = 768 if "L/14" in config.get("clip_version", "ViT-B/32") else 512
    model = build_model({
        "feature_mode": config.get("feature_mode", "humanml3d"),
        "input_dim":    263,
        "latent_dim":   config.get("latent_dim",  512),
        "context_dim":  context_dim,
        "num_heads":    config.get("num_heads",   8),
        "num_layers":   config.get("num_layers",  8),
        "max_frames":   config.get("max_frames",  196),
        "dropout":      0.0,
    }, device=device)

    weights = os.path.join(ckpt_dir, "ema.pt" if use_ema else "model.pt")
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {weights}")
    return model, config


def load_generated(generated_dir):
    """Load all .npz files listed in manifest.json. Returns list of clip dicts."""
    manifest_path = os.path.join(generated_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    clips = []
    for cid in manifest["clip_ids"]:
        path = os.path.join(generated_dir, f"{cid}.npz")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}, skipping")
            continue
        data = np.load(path)
        clips.append({
            "id":       cid,
            "gen_norm": data["gen_norm"],   # (T, D)
            "gt_norm":  data["gt_norm"],    # (T, D)
            "ctx":      data["ctx"],        # (77, ctx_dim)
            "T":        int(data["T"]),
        })
    return clips, manifest


# ── MPJPE ────────────────────────────────────────────────────────────────────

def mpjpe_from_joints(joints_gen, joints_gt):
    T = min(len(joints_gen), len(joints_gt))
    gen = joints_gen[:T] - joints_gen[:T, 0:1]
    gt  = joints_gt[:T]  - joints_gt[:T,  0:1]
    per_frame = np.sqrt(((gen - gt) ** 2).sum(axis=-1)).mean(axis=-1)
    return float(per_frame.mean())


# ── FID ──────────────────────────────────────────────────────────────────────

def motion_to_feat(motion_norm, T):
    return motion_norm[:T].mean(axis=0).astype(np.float64)


def _pca_reduce(arr, n_components):
    mu = arr.mean(axis=0)
    centered = arr - mu
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


def compute_fid(real_feats, gen_feats):
    N, D = real_feats.shape
    if N < D:
        n_comp = max(2, N // 4)
        combined = _pca_reduce(np.concatenate([real_feats, gen_feats], axis=0), n_comp)
        real_feats = combined[:N]
        gen_feats  = combined[N:]

    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r     = np.cov(real_feats.T)
    sigma_g     = np.cov(gen_feats.T)

    diff    = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


# ── R-Precision ───────────────────────────────────────────────────────────────

@torch.no_grad()
def nll_scores_for_clip(model, schedule, motion_norm_np, text_ctxs_np, device, n_eval_t):
    """
    Score K candidate texts against one motion via average denoising MSE.
    motion_norm_np : (T, D)       — ground-truth normalised motion
    text_ctxs_np   : (K, 77, ctx) — candidate CLIP embeddings
    Returns        : (K,) float32 — lower = better match
    """
    K    = len(text_ctxs_np)
    T, D = motion_norm_np.shape

    x0   = torch.from_numpy(motion_norm_np.astype(np.float32)).unsqueeze(0).to(device)
    ctxs = torch.from_numpy(text_ctxs_np).to(device)

    t_vals = torch.linspace(50, 950, n_eval_t, dtype=torch.long, device=device)
    scores = torch.zeros(K, device=device)

    for t_val in t_vals:
        t1         = t_val.unsqueeze(0)
        noise      = torch.randn(1, T, D, device=device)
        sqrt_acp   = schedule.sqrt_alphas_cumprod[t1][:, None, None]
        sqrt_omacp = schedule.sqrt_one_minus_alphas_cumprod[t1][:, None, None]
        x_t        = (sqrt_acp * x0 + sqrt_omacp * noise).expand(K, -1, -1).contiguous()
        t_rep      = t_val.unsqueeze(0).expand(K)
        noise_rep  = noise.expand(K, -1, -1).contiguous()

        eps_pred = model(x_t, t_rep, ctxs)
        scores  += ((eps_pred - noise_rep) ** 2).mean(dim=(1, 2))

    return (scores / n_eval_t).cpu().numpy()


def compute_r_precision(model, schedule, gt_motions, all_ctxs, device,
                        pool_size=32, top_k=(1, 2, 3), n_eval_t=4, seed=0):
    """
    R-Precision@k scored against ground-truth motions (not generated ones)
    to avoid self-referential bias from using the same model for generation
    and scoring.

    gt_motions : list[np.ndarray] — each (T_i, D) normalised gt motion
    all_ctxs   : (N, 77, ctx_dim)
    """
    N      = len(gt_motions)
    counts = {k: 0 for k in top_k}
    rng    = np.random.default_rng(seed)

    for i, motion_norm in enumerate(tqdm(gt_motions, desc="R-Precision", leave=False)):
        neg_pool = [j for j in range(N) if j != i]
        n_neg    = min(pool_size - 1, len(neg_pool))
        neg_idx  = rng.choice(neg_pool, size=n_neg, replace=False)
        pool_idx = np.concatenate([[i], neg_idx])

        pool_ctxs = all_ctxs[pool_idx]
        scores    = nll_scores_for_clip(model, schedule, motion_norm, pool_ctxs, device, n_eval_t)

        rank = int(np.where(np.argsort(scores) == 0)[0][0]) + 1
        for k in top_k:
            if rank <= k:
                counts[k] += 1

    return {k: counts[k] / N for k in top_k}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading generated clips from {args.generated_dir} …")
    clips, manifest = load_generated(args.generated_dir)
    N = len(clips)
    print(f"  {N} clips")

    if N == 0:
        print("No clips found — nothing to report.")
        return

    model, config = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    schedule      = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))

    # ── MPJPE ─────────────────────────────────────────────────────────────────
    print("\n── MPJPE ───────────────────────────────────────────────────────────────")
    mpjpe_values = []
    real_feats   = []
    gen_feats    = []
    gt_motions   = []
    all_ctxs     = []

    for clip in tqdm(clips, desc="MPJPE"):
        T        = clip["T"]
        gen_norm = clip["gen_norm"]
        gt_norm  = clip["gt_norm"]

        gen_raw = gen_norm * std + mean
        gt_raw  = gt_norm  * std + mean

        joints_gen = recover_from_ric(gen_raw, joints_num=22)
        joints_gt  = recover_from_ric(gt_raw,  joints_num=22)

        if args.smooth_sigma > 0:
            joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)
            joints_gt  = gaussian_filter1d(joints_gt,  sigma=args.smooth_sigma, axis=0)

        mpjpe_values.append(mpjpe_from_joints(joints_gen, joints_gt))
        real_feats.append(motion_to_feat(gt_norm,  T))
        gen_feats.append(motion_to_feat(gen_norm, T))
        gt_motions.append(gt_norm)
        all_ctxs.append(clip["ctx"])

    mpjpe_arr  = np.array(mpjpe_values)
    mean_mpjpe = float(mpjpe_arr.mean())
    std_mpjpe  = float(mpjpe_arr.std())
    print(f"MPJPE   {mean_mpjpe*1000:.2f} ± {std_mpjpe*1000:.2f} mm  (N={N})")

    # ── FID ───────────────────────────────────────────────────────────────────
    print("\n── FID ─────────────────────────────────────────────────────────────────")
    real_arr = np.stack(real_feats)
    gen_arr  = np.stack(gen_feats)
    fid      = compute_fid(real_arr, gen_arr)
    D        = real_arr.shape[1]
    fid_note = "raw feature space" if N >= D else f"PCA-reduced ({max(2, N//4)}-d, N={N}<D={D})"
    print(f"FID     {fid:.4f}  ({fid_note})")

    # ── R-Precision ───────────────────────────────────────────────────────────
    print("\n── R-Precision ─────────────────────────────────────────────────────────")
    effective_pool = min(args.pool_size, N)
    all_ctxs_arr   = np.stack(all_ctxs)
    r_prec = compute_r_precision(
        model, schedule, gt_motions, all_ctxs_arr, device,
        pool_size=effective_pool,
        n_eval_t=args.n_eval_t,
        seed=args.seed,
    )
    for k, v in sorted(r_prec.items()):
        print(f"R-Prec@{k} {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "generated_dir":  args.generated_dir,
        "checkpoint":     args.checkpoint,
        "n_clips":        N,
        "mpjpe_mean_mm":  round(mean_mpjpe * 1000, 4),
        "mpjpe_std_mm":   round(std_mpjpe  * 1000, 4),
        "fid":            round(fid, 6),
        "fid_note":       f"mean-pooled {D}-dim normalised motion features ({fid_note}); not standard t2m FID",
        "r_precision":    {f"top_{k}": round(v, 6) for k, v in r_prec.items()},
        "r_prec_note":    (
            f"NLL-based denoising MSE over {args.n_eval_t} timesteps "
            f"(linspace 50–950); pool_size={effective_pool}; "
            f"scored against ground-truth motions"
        ),
        "generation_args": {k: v for k, v in manifest.items() if k != "clip_ids"},
    }

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults → {out_json}")


if __name__ == "__main__":
    main()
