"""
Generation metrics in the T2M evaluator's feature space (Guo et al., 2022) — FID,
R-Precision, Multimodal Distance, Diversity — plus the loading/decoding they need.

The evaluator only understands 263-d HumanML3D features, so smplh-generated clips
(135-d) are decoded to joints via SMPL forward kinematics and re-extracted into 263-d
features. Both gen AND gt go through the identical path, so FID stays a self-consistent
comparison: raw135 → Y-up joints → extract_hml3d_features → raw263.
"""

import json
import os

import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

from data.clips import read_tagged_caption
from data.hml3d_features import extract_hml3d_features, get_tgt_offsets
from utils.decode import recover_from_ric


def load_generated(generated_dir: str, data_root: str):
    """(clips, manifest) for a directory written by generate.py.

    Clips missing their .npz or their annotation are skipped with a warning. Text comes
    from the pre-tagged `word/POS` tokens the evaluator was trained on — re-tokenising
    would shift its text embeddings.
    """
    with open(os.path.join(generated_dir, "manifest.json")) as f:
        manifest = json.load(f)

    clips = []
    for cid in manifest["clip_ids"]:
        npz_path = os.path.join(generated_dir, f"{cid}.npz")
        if not os.path.exists(npz_path):
            print(f"  [WARN] missing {npz_path}, skipping")
            continue
        text, tokens = read_tagged_caption(data_root, cid)
        if not text:
            print(f"  [WARN] missing text for {cid}, skipping")
            continue
        data = np.load(npz_path)
        clips.append({
            "id": cid,
            "gen_norm": data["gen_norm"],   # (T, D)
            "gt_norm":  data["gt_norm"],    # (T, D)
            "text": text, "tokens": tokens, "T": int(data["T"]),
        })
    return clips, manifest


def build_decoders(feature_mode, mean, std, data_root, smplh_feat_root=None,
                   smplh_model_path=None):
    """(to_raw263, to_joints): closures mapping a *normalised* motion, as stored in the
    .npz, to raw 263-d HumanML3D features and to (T, 22, 3) Y-up joints."""
    if feature_mode != "smplh":
        def to_raw263(m_norm):
            return (m_norm * std + mean).astype(np.float32)

        def to_joints(m_norm):
            return recover_from_ric((m_norm * std + mean).astype(np.float32), joints_num=22)
        return to_raw263, to_joints

    import smplx
    from data.smplh_features import smplh_decode_to_joints

    if smplh_feat_root is None:
        raise ValueError("smplh eval needs the 135-d feature Mean/Std: pass "
                         "--smplh_feat_root or regenerate so manifest['data_root'] is set.")
    smplh_mean = np.load(os.path.join(smplh_feat_root, "Mean.npy"))
    smplh_std  = np.load(os.path.join(smplh_feat_root, "Std.npy"))
    body_model = smplx.SMPLHLayer(model_path=smplh_model_path, gender="neutral",
                                  ext="npz").eval()

    # Rest-pose skeleton to retarget onto: taken once from a reference HumanML3D clip.
    ref_dir = os.path.join(data_root, "new_joints")
    ref_files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".npy"))
    tgt_offsets = get_tgt_offsets(np.load(os.path.join(ref_dir, ref_files[0])))

    def to_joints(m_norm):
        raw135 = (m_norm * smplh_std + smplh_mean).astype(np.float32)
        return smplh_decode_to_joints(raw135, body_model)      # (T, 22, 3) Y-up

    def to_raw263(m_norm):
        return extract_hml3d_features(to_joints(m_norm), tgt_offsets)   # (T-1, 263)

    return to_raw263, to_joints


def compute_fid(real_feats, gen_feats) -> float:
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sig_r, sig_g = np.cov(real_feats.T), np.cov(gen_feats.T)
    diff = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


def compute_r_precision(motion_embs, text_embs, pool_size=32, top_k=(1, 2, 3), seed=0):
    """Fraction of motions whose own caption ranks in the top k of a random pool."""
    N = len(motion_embs)
    counts = {k: 0 for k in top_k}
    rng = np.random.default_rng(seed)

    for i in tqdm(range(N), desc="R-Precision", leave=False):
        neg_pool = [j for j in range(N) if j != i]
        neg_idx = rng.choice(neg_pool, size=min(pool_size - 1, len(neg_pool)),
                             replace=False)
        pool_idx = np.concatenate([[i], neg_idx])
        dists = np.sqrt(((motion_embs[i] - text_embs[pool_idx]) ** 2).sum(axis=-1))
        rank = int(np.where(np.argsort(dists) == 0)[0][0]) + 1
        for k in top_k:
            counts[k] += rank <= k
    return {k: counts[k] / N for k in top_k}


def compute_mm_dist(motion_embs, text_embs) -> float:
    """Mean Euclidean distance between each motion and its own caption embedding.
    Lower = better text-motion alignment (R-Precision's continuous counterpart)."""
    return float(np.sqrt(((motion_embs - text_embs) ** 2).sum(axis=-1)).mean())


def compute_diversity(motion_embs, diversity_times=300, seed=0) -> float:
    """Mean distance between randomly paired motion embeddings — detects mode collapse;
    should land close to the ground-truth diversity."""
    n = len(motion_embs)
    times = min(diversity_times, n)
    rng = np.random.default_rng(seed)
    first, second = rng.choice(n, times, replace=False), rng.choice(n, times, replace=False)
    return float(np.sqrt(((motion_embs[first] - motion_embs[second]) ** 2).sum(axis=-1)).mean())
