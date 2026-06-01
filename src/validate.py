"""
validate.py — full-split evaluation: MPJPE, FID, R-Precision.

Iterates over every clip in a split, generates a motion from the clip's text,
and computes three metrics vs. the ground-truth motion:

  MPJPE          Root-relative mean per-joint position error (mm).
  FID            Fréchet distance in mean-pooled normalised motion feature space.
                 NOTE: this is NOT the standard t2m FID (which needs a pre-trained
                 motion encoder). It is a distribution-level proxy; use it to
                 compare runs against each other, not against published numbers.
  R-Precision@{1,2,3}
                 For each generated motion, rank a pool of `pool_size` texts
                 (1 true + pool_size-1 random negatives) by their average
                 denoising NLL and report the fraction where the true text
                 appears in top-k.

Usage:
    python validate.py \\
        --checkpoint runs/exp_hml3d/checkpoint_latest \\
        --data_root  data/HumanML3D \\
        --split val \\
        [--output_dir eval_results/validate_val] \\
        [--max_clips 500]          # omit to eval entire split \\
        [--pool_size 32]           # R-Precision pool size \\
        [--n_eval_t 4]             # NLL timesteps per clip \\
        [--guidance_scale 4.0] \\
        [--num_steps 1000] \\
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
from model.text_encoder import CLIPTextEncoder
from model.schedule import NoiseSchedule
from model.sampler import DDPMSampler
from utils.visualise import recover_from_ric
from utils.skeleton import recover_world_positions_smpl
from data.dataset import _SMPL_CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="Checkpoint directory (contains config.json + ema.pt).")
    p.add_argument("--data_root",     required=True,
                   help="HumanML3D root (needs Mean.npy, Std.npy, {split}.txt, …).")
    p.add_argument("--split",         default="val",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate.")
    p.add_argument("--output_dir",    default="./eval_results/validate")
    p.add_argument("--max_clips",     type=int, default=None,
                   help="Cap number of clips evaluated (random subset). None = full split.")
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_steps",     type=int, default=1000,
                   help="DDPM denoising steps (reduce for faster but lower-quality samples).")
    p.add_argument("--smooth_sigma",  type=float, default=1.5,
                   help="Gaussian smoothing on joint trajectories before MPJPE (0 = off).")
    p.add_argument("--pool_size",     type=int, default=32,
                   help="R-Precision pool: 1 true text + (pool_size-1) random negatives.")
    p.add_argument("--n_eval_t",      type=int, default=4,
                   help="Number of denoising timesteps sampled per clip for NLL scoring.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_ema",        action="store_true",
                   help="Load model.pt instead of ema.pt.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_dir, device, use_ema=True):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)

    feature_mode = config.get("feature_mode", "humanml3d")
    input_dim    = len(_SMPL_CHANNELS) if feature_mode == "group" else 263
    context_dim  = 768 if "L/14" in config.get("clip_version", "ViT-B/32") else 512

    model = build_model({
        "feature_mode": feature_mode,
        "input_dim":    input_dim,
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
    print(f"Loaded: {weights}  (feature_mode={feature_mode})")
    return model, config


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(data_root, split, feature_mode, max_frames=196, min_frames=16,
               max_clips=None, seed=42):
    """Return a list of clip dicts for the requested split."""
    with open(os.path.join(data_root, f"{split}.txt")) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    vec_dir      = os.path.join(data_root, "new_joint_vecs")
    text_dir     = os.path.join(data_root, "texts")
    text_emb_dir = os.path.join(data_root, "text_emb")
    has_emb      = os.path.isdir(text_emb_dir)

    clips = []
    for cid in all_ids:
        vec_path = os.path.join(vec_dir, f"{cid}.npy")
        if not os.path.exists(vec_path):
            continue
        T_raw = int(np.load(vec_path, mmap_mode="r").shape[0])
        if T_raw < min_frames:
            continue

        text_path = os.path.join(text_dir, f"{cid}.txt")
        if not os.path.exists(text_path):
            continue
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        text = lines[0].split("#")[0].strip()

        # precomputed CLIP embedding (first annotation, float32)
        context_emb = None
        if has_emb:
            emb_path = os.path.join(text_emb_dir, f"{cid}.npy")
            if os.path.exists(emb_path):
                context_emb = np.load(emb_path)[0].astype(np.float32)  # (77, ctx_dim)

        clips.append({
            "id":          cid,
            "text":        text,
            "vec_path":    vec_path,
            "T":           min(T_raw, max_frames),
            "context_emb": context_emb,   # (77, ctx_dim) or None
        })

    rng = np.random.default_rng(seed)
    rng.shuffle(clips)
    if max_clips is not None:
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Joint recovery
# ─────────────────────────────────────────────────────────────────────────────

def recover_joints(raw_features, feature_mode):
    """Denormalised (T, D) features → (T, 22, 3) world-space joint positions."""
    if feature_mode == "group":
        t = torch.from_numpy(raw_features.astype(np.float32)).unsqueeze(0)
        return recover_world_positions_smpl(t)[0].numpy()
    return recover_from_ric(raw_features, joints_num=22)


# ─────────────────────────────────────────────────────────────────────────────
# MPJPE
# ─────────────────────────────────────────────────────────────────────────────

def mpjpe_from_joints(joints_gen, joints_gt):
    """
    Root-relative MPJPE (metres) over the common frame range.
    Returns (mpjpe_scalar, T_common).
    """
    T = min(len(joints_gen), len(joints_gt))
    gen = joints_gen[:T] - joints_gen[:T, 0:1]  # root-relative
    gt  = joints_gt[:T]  - joints_gt[:T,  0:1]
    per_frame = np.sqrt(((gen - gt) ** 2).sum(axis=-1)).mean(axis=-1)  # (T,)
    return float(per_frame.mean()), T


# ─────────────────────────────────────────────────────────────────────────────
# FID helpers
# ─────────────────────────────────────────────────────────────────────────────

def motion_to_feat(motion_norm, T):
    """Mean-pool normalised motion over the T valid frames → (D,) float64."""
    return motion_norm[:T].mean(axis=0).astype(np.float64)


def _pca_reduce(arr, n_components):
    """Simple PCA via SVD. arr: (N, D) → (N, n_components)."""
    mu = arr.mean(axis=0)
    centered = arr - mu
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


def compute_fid(real_feats, gen_feats):
    """
    Fréchet distance between two (N, D) feature arrays.
    Applies PCA reduction when N < D to keep the covariance well-conditioned.
    """
    N, D = real_feats.shape
    if N < D:
        n_comp = max(2, N // 4)
        combined = _pca_reduce(
            np.concatenate([real_feats, gen_feats], axis=0), n_comp
        )
        real_feats = combined[:N]
        gen_feats  = combined[N:]

    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r     = np.cov(real_feats.T)
    sigma_g     = np.cov(gen_feats.T)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


# ─────────────────────────────────────────────────────────────────────────────
# R-Precision — NLL scoring
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def nll_scores_for_clip(model, schedule, motion_norm_np, text_ctxs_np, device, n_eval_t):
    """
    Score K candidate texts against one generated motion via average denoising MSE.

    For each of n_eval_t uniformly-spaced timesteps t:
      1. Sample a single noise ε.
      2. Form x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε  (same for all K texts).
      3. Run the model with each of the K text contexts → K noise predictions.
      4. Accumulate MSE(ε̂_k, ε) per text.

    motion_norm_np : (T, D) normalised generated motion (numpy)
    text_ctxs_np   : (K, 77, ctx_dim) CLIP embeddings (numpy)
    Returns        : (K,) float32 — lower score = better text-motion match.
    """
    K     = len(text_ctxs_np)
    T, D  = motion_norm_np.shape

    x0   = torch.from_numpy(motion_norm_np.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, D)
    ctxs = torch.from_numpy(text_ctxs_np).to(device)                                    # (K, 77, ctx_dim)

    # linearly-spaced evaluation timesteps in [50, 950]
    t_vals = torch.linspace(50, 950, n_eval_t, dtype=torch.long, device=device)
    scores = torch.zeros(K, device=device)

    for t_val in t_vals:
        t1 = t_val.unsqueeze(0)  # (1,)

        noise = torch.randn(1, T, D, device=device)
        sqrt_acp   = schedule.sqrt_alphas_cumprod[t1][:, None, None]        # (1, 1, 1)
        sqrt_omacp = schedule.sqrt_one_minus_alphas_cumprod[t1][:, None, None]
        x_t = (sqrt_acp * x0 + sqrt_omacp * noise).expand(K, -1, -1).contiguous()  # (K, T, D)

        t_rep      = t_val.unsqueeze(0).expand(K)                           # (K,)
        noise_rep  = noise.expand(K, -1, -1).contiguous()                   # (K, T, D)

        eps_pred = model(x_t, t_rep, ctxs)                                  # (K, T, D)
        mse = ((eps_pred - noise_rep) ** 2).mean(dim=(1, 2))                # (K,)
        scores += mse

    return (scores / n_eval_t).cpu().numpy()


def compute_r_precision(model, schedule, gen_motions, all_ctxs, device,
                        pool_size=32, top_k=(1, 2, 3), n_eval_t=4, seed=0):
    """
    R-Precision@k for a list of generated motions.

    gen_motions : list[np.ndarray]  — each (T_i, D) normalised
    all_ctxs    : (N, 77, ctx_dim) — one embedding per clip, matching gen_motions order
    Returns     : dict  {1: float, 2: float, 3: float}
    """
    N      = len(gen_motions)
    counts = {k: 0 for k in top_k}
    rng    = np.random.default_rng(seed)

    for i, motion_norm in enumerate(tqdm(gen_motions, desc="R-Precision", leave=False)):
        neg_pool = [j for j in range(N) if j != i]
        n_neg    = min(pool_size - 1, len(neg_pool))
        neg_idx  = rng.choice(neg_pool, size=n_neg, replace=False)
        pool_idx = np.concatenate([[i], neg_idx])

        pool_ctxs = all_ctxs[pool_idx]                           # (P, 77, ctx_dim)
        scores    = nll_scores_for_clip(model, schedule, motion_norm, pool_ctxs, device, n_eval_t)

        # rank ascending (lower NLL = better match); position of true text (index 0 in pool)
        rank = int(np.where(np.argsort(scores) == 0)[0][0]) + 1  # 1-indexed
        for k in top_k:
            if rank <= k:
                counts[k] += 1

    return {k: counts[k] / N for k in top_k}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    feature_mode  = config.get("feature_mode", "humanml3d")
    max_frames    = config.get("max_frames", 196)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    if feature_mode == "group":
        mean = mean[_SMPL_CHANNELS]
        std  = std[_SMPL_CHANNELS]

    text_encoder = CLIPTextEncoder(config.get("clip_version", "ViT-B/32"), device=device)
    schedule     = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    sampler      = DDPMSampler(model, schedule, device)

    # ── Load split ─────────────────────────────────────────────────────────────
    print(f"\nLoading '{args.split}' split …")
    clips = load_split(
        args.data_root, args.split, feature_mode,
        max_frames=max_frames, max_clips=args.max_clips, seed=args.seed,
    )
    print(f"  {len(clips)} clips")

    # ── Phase 1: generate motions ──────────────────────────────────────────────
    print(f"\n── Phase 1 / {len(clips)}: generating motions ───────────────────────")

    per_clip_ids  = []
    mpjpe_values  = []
    real_feats    = []   # for FID
    gen_feats     = []   # for FID
    gen_motions   = []   # for R-Precision: list of (T, D) normalised
    all_ctxs      = []   # for R-Precision: list of (77, ctx_dim)

    for clip in tqdm(clips, desc="Generating"):
        cid  = clip["id"]
        T    = clip["T"]
        text = clip["text"]

        # ground-truth raw features
        raw_263  = np.load(clip["vec_path"])[:T]                               # (T, 263)
        raw_feat = raw_263[:, _SMPL_CHANNELS] if feature_mode == "group" else raw_263  # (T, D)
        gt_norm  = (raw_feat - mean) / std                                     # (T, D)

        # text context
        if clip["context_emb"] is not None:
            ctx_np = clip["context_emb"]                                       # (77, ctx_dim)
            ctx    = torch.from_numpy(ctx_np).unsqueeze(0).to(device)         # (1, 77, ctx_dim)
        else:
            with torch.no_grad():
                ctx = text_encoder.encode([text])                              # (1, 77, ctx_dim)
            ctx_np = ctx[0].cpu().numpy()

        # generate
        try:
            with torch.no_grad():
                gen_norm = sampler.sample(
                    ctx,
                    length=T,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.num_steps,
                    show_progress=False,
                ).cpu().numpy()                                                # (T, D) normalised
        except Exception as exc:
            print(f"  [WARN] {cid}: generation failed — {exc}")
            continue

        gen_raw = gen_norm * std + mean                                        # (T, D) denormalised

        # MPJPE
        joints_gen = recover_joints(gen_raw, feature_mode)                    # (T, 22, 3)
        joints_gt  = recover_joints(raw_feat, feature_mode)
        if args.smooth_sigma > 0:
            joints_gen = gaussian_filter1d(joints_gen, sigma=args.smooth_sigma, axis=0)
            joints_gt  = gaussian_filter1d(joints_gt,  sigma=args.smooth_sigma, axis=0)
        mpjpe, _ = mpjpe_from_joints(joints_gen, joints_gt)

        per_clip_ids.append(cid)
        mpjpe_values.append(mpjpe)
        real_feats.append(motion_to_feat(gt_norm, T))
        gen_feats.append(motion_to_feat(gen_norm, T))
        gen_motions.append(gen_norm)
        all_ctxs.append(ctx_np)

    N = len(per_clip_ids)
    print(f"  {N}/{len(clips)} clips succeeded.")

    if N == 0:
        print("No clips succeeded — nothing to report.")
        return

    # ── Phase 2: MPJPE ────────────────────────────────────────────────────────
    mpjpe_arr   = np.array(mpjpe_values)
    mean_mpjpe  = float(mpjpe_arr.mean())
    std_mpjpe   = float(mpjpe_arr.std())
    print(f"\nMPJPE   {mean_mpjpe*1000:.2f} ± {std_mpjpe*1000:.2f} mm  (N={N})")

    # ── Phase 3: FID ──────────────────────────────────────────────────────────
    real_arr = np.stack(real_feats)   # (N, D)
    gen_arr  = np.stack(gen_feats)
    fid      = compute_fid(real_arr, gen_arr)
    D        = real_arr.shape[1]
    fid_note = "raw feature space"
    if N < D:
        fid_note = f"PCA-reduced ({max(2, N//4)}-d, N={N}<D={D})"
    print(f"FID     {fid:.4f}  ({fid_note})")

    # ── Phase 4: R-Precision ──────────────────────────────────────────────────
    effective_pool = min(args.pool_size, N)
    all_ctxs_arr   = np.stack(all_ctxs)                                       # (N, 77, ctx_dim)
    r_prec = compute_r_precision(
        model, schedule, gen_motions, all_ctxs_arr, device,
        pool_size=effective_pool,
        n_eval_t=args.n_eval_t,
        seed=args.seed,
    )
    for k, v in sorted(r_prec.items()):
        print(f"R-Prec@{k} {v:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "split":          args.split,
        "checkpoint":     args.checkpoint,
        "feature_mode":   feature_mode,
        "n_clips":        N,
        "guidance_scale": args.guidance_scale,
        "num_steps":      args.num_steps,
        "mpjpe_mean_mm":  round(mean_mpjpe * 1000, 4),
        "mpjpe_std_mm":   round(std_mpjpe  * 1000, 4),
        "fid":            round(fid, 6),
        "fid_note":       f"mean-pooled {D}-dim normalised motion features ({fid_note}); not standard t2m FID",
        "r_precision":    {f"top_{k}": round(v, 6) for k, v in r_prec.items()},
        "r_prec_note":    (
            f"NLL-based: average denoising MSE over {args.n_eval_t} timesteps "
            f"(linspace 50–950); pool_size={effective_pool}"
        ),
    }

    out_json = os.path.join(args.output_dir, f"results_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    out_csv = os.path.join(args.output_dir, f"per_clip_{args.split}.csv")
    with open(out_csv, "w") as f:
        f.write("clip_id,mpjpe_mm\n")
        for cid, mpjpe in zip(per_clip_ids, mpjpe_values):
            f.write(f"{cid},{mpjpe * 1000:.4f}\n")

    print(f"\nResults → {out_json}")
    print(f"Per-clip → {out_csv}")


if __name__ == "__main__":
    main()
