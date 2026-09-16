"""
How compatible is MotionFix with HumanML3D — as motion, and as language?

The two questions live in DIFFERENT embedding spaces, on purpose:

  motion  TMR's motion encoder (`data/motionfix/eval-deps/last_weights`), the space the
          MotionFix benchmark itself scores retrieval in.
  text    the pooled T5 embedding, i.e. the space the GroupDiT backbone actually
          conditions on. "Compatible" for text means *this model can read it*, which is a
          claim about the conditioning space and nowhere else.

They are NOT merged into one plot. TMR has a modality gap — motions and texts occupy
separate regions of its latent space regardless of which corpus they came from — so a joint
scatter would show a modality split and say nothing about datasets.

Every panel carries a HELD-OUT HumanML3D series alongside the MotionFix one. Without it a
separation is unfalsifiable: t-SNE separates most things, and a classifier two-sample test
needs to be shown failing on data that genuinely is the same distribution before its
success on MotionFix means anything.

The headline statistic is the same on both panels — the accuracy of a classifier trying to
recover which corpus a point came from (a classifier two-sample test). Near chance = the
corpora are interchangeable in that space; near 100 % = they are not.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from data.smplh_features import rotation_6d_to_matrix
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Series:
    """One corpus in one embedding space."""
    name: str
    X: np.ndarray                      # (N, D) embeddings
    ids: list = field(default_factory=list)
    lengths: np.ndarray | None = None  # frame counts, motion panel only

    def __len__(self):
        return len(self.X)


# ── corpora ─────────────────────────────────────────────────────────────────────
def _split_ids(data_root, split, exclude_mirrored=True):
    """`split` may name several splits joined by '+' ("val+test") — the test split alone
    holds too few clips inside MotionFix's length range to build a control series."""
    ids = []
    for part in split.split("+"):
        with open(os.path.join(data_root, f"{part}.txt")) as f:
            ids += [l.strip() for l in f if l.strip()]
    # HumanML3D ships every clip twice, the 'M'-prefixed copy being the left/right mirror
    # with a correspondingly swapped caption. Keeping both would fill the plot with
    # near-duplicate pairs and make every density estimate (t-SNE, kNN) count each motion
    # twice.
    return [i for i in ids if not i.startswith("M")] if exclude_mirrored else ids


def load_hml3d_captions(data_root, split, n, seed, exclude_mirrored=True,
                        per_clip=1):
    """One caption per clip by default — HumanML3D's ~3 annotations of the same clip are
    paraphrases of each other, so taking all of them inflates local density with samples
    that are not independent."""
    rng = np.random.default_rng(seed)
    ids = _split_ids(data_root, split, exclude_mirrored)
    rng.shuffle(ids)
    out = []
    for cid in ids:
        path = os.path.join(data_root, "texts", f"{cid}.txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            caps = [l.split("#")[0].strip() for l in f if l.strip()]
        caps = [c for c in caps if c]
        for c in caps[:per_clip]:
            out.append((cid, c))
        if len(out) >= n:
            break
    return out[:n]


def load_motionfix_instructions(mfix_root, split="test"):
    """From the 6 MB annotation JSON, not the 5 GB triplet dump.

    Verified byte-identical to the dump's `text` field on all 1013 test keys, which is what
    the editor is conditioned on — so the small file is not an approximation here.
    """
    ds = os.path.join(mfix_root, "data", "motionfix-dataset")
    with open(os.path.join(ds, "amt_motionfix_latest.json")) as f:
        amt = json.load(f)
    with open(os.path.join(ds, "splits.json")) as f:
        keys = json.load(f)[split]
    return [(k, amt[k]["annotation"]) for k in keys if k in amt]


# ── text embeddings ─────────────────────────────────────────────────────────────
@torch.no_grad()
def pool_text(encoder, texts, batch=128, device="cpu"):
    """Mean-pool an encoder's (B, L, D) output over its REAL tokens.

    Padding columns are already zeroed by both encoders in `model/text_encoder.py`, so the
    real-token count is the number of non-zero rows; dividing by L instead would rescale
    every embedding by its own caption length and turn a length difference between the two
    corpora into an apparent semantic one.
    """
    out = []
    for i in log.progress(range(0, len(texts), batch), desc="encoding text"):
        ctx = encoder.encode(list(texts[i:i + batch])).to(device)     # (B, L, D)
        real = (ctx.abs().sum(-1) > 0).float()                        # (B, L)
        pooled = (ctx * real[..., None]).sum(1) / real.sum(1).clamp(min=1)[:, None]
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, 0)


# ── motion embeddings ───────────────────────────────────────────────────────────
def load_hml3d_motions(data_root, split, n, seed, min_frames=20, exclude_mirrored=True):
    """Raw (T, 135) SMPL-H features at the dataset's 20 fps, plus their ids."""
    rng = np.random.default_rng(seed)
    ids = _split_ids(data_root, split, exclude_mirrored)
    rng.shuffle(ids)
    out = []
    for cid in ids:
        path = os.path.join(data_root, "new_joint_vecs", f"{cid}.npy")
        if not os.path.exists(path):
            continue
        feats = np.load(path)
        if len(feats) < min_frames:
            continue
        out.append((cid, feats))
        if len(out) >= n:
            break
    return out


def load_motionfix_motions(mfix_root, split="test", which="source"):
    """MotionFix clips as raw (T, 135) features at their native 30 fps.

    Reads `motionfix_{split}.pth.tar` (805 MB for test) rather than the 5 GB combined dump,
    and releases it before returning.
    """
    import gc
    import joblib

    from data.smplh_features import smplh_to_features

    ds = os.path.join(mfix_root, "data", "motionfix-dataset")
    log.info("loading motionfix_%s.pth.tar (released after featurising) …", split)
    data = joblib.load(os.path.join(ds, f"motionfix_{split}.pth.tar"))
    key = f"motion_{which}"
    out = []
    for k in log.progress(sorted(data), desc=f"featurising {which}"):
        m = data[k][key]
        rots = np.asarray(m["rots"], dtype=np.float32)
        trans = np.asarray(m["trans"], dtype=np.float32)
        out.append((k, smplh_to_features(rots, trans)))   # already 30 fps == TMR's fps
    del data
    gc.collect()
    return out


def match_length_distribution(pool, targets, n, seed=0, bins=10):
    """Draw `n` clips from `pool` whose length distribution matches `targets`.

    TMR pools over time, so its embedding geometry tracks clip duration: MotionFix test
    clips run 2-5 s while HumanML3D reaches ~10 s, and left uncontrolled that difference
    alone lets a classifier separate the corpora — it would report "incompatible" about
    duration rather than about motion.

    Matching the DISTRIBUTION (stratified sampling over the target's own length deciles)
    rather than pairing clip-to-clip matters: without-replacement 1-1 pairing exhausts the
    pool in the tails and degrades badly once `n` approaches the pool size — measured at a
    31.8-frame mean residual on the held-out split. The diagnostics returned here report
    the achieved mean/median against the target's, and `log.warning` fires if the pool
    cannot cover the target range; check them rather than assuming the match worked.

    Returns (picked, diagnostics).
    """
    rng = np.random.default_rng(seed)
    lens = np.array([len(f) for _, f in pool], dtype=float)
    # Bin edges are the target's own quantiles. The outer edges must NOT be opened to
    # +/-inf: doing so makes the top bin swallow every pool clip longer than the target
    # maximum, and since the pool skews long the "matched" sample comes back ~2x too long
    # (measured: mean 131.5 frames against a 73.3 target). Clips outside the target range
    # are simply ineligible.
    edges = np.unique(np.quantile(targets, np.linspace(0, 1, bins + 1)))
    edges[-1] = edges[-1] + 1e-6            # half-open top bin must include the longest
    picked, short = [], 0
    for i in range(len(edges) - 1):
        want = int(round(n * float(((targets >= edges[i]) & (targets < edges[i + 1])).mean())))
        cand = np.flatnonzero((lens >= edges[i]) & (lens < edges[i + 1]))
        if len(cand) < want:
            short += want - len(cand)
            take = cand
        else:
            take = rng.choice(cand, want, replace=False)
        picked.extend(pool[j] for j in take)
    rng.shuffle(picked)
    got = np.array([len(f) for _, f in picked], dtype=float)
    if len(got) and abs(got.mean() - np.mean(targets)) > 0.1 * np.mean(targets):
        log.warning("length matching is OFF by >10 %%: got mean %.1f vs target %.1f — "
                    "the pool cannot cover the target range, so any corpus difference "
                    "measured downstream is confounded by clip duration",
                    got.mean(), np.mean(targets))
    return picked, {"n": len(picked), "shortfall": short,
                    "target_mean": float(np.mean(targets)),
                    "got_mean": float(got.mean()) if len(got) else float("nan"),
                    "target_median": float(np.median(targets)),
                    "got_median": float(np.median(got)) if len(got) else float("nan")}


def canonicalise_heading(feats, up=2):
    """Rotate a clip about the world up axis so its first frame has a canonical heading.

    Which way the performer happened to face in the capture rig is a property of the MOCAP
    SESSION, not of the motion, and the two corpora do not share that convention. Left in,
    it dominates the comparison: heading alone separates HumanML3D from MotionFix at
    AUC 0.96, against 0.65 once removed.

    Only `global_orient` changes. `body_pose` is expressed in local joint frames and
    `trans_delta` in the pelvis frame — for a global pre-rotation Q,
    R_{i-1}^T Q^T Q (t_i - t_{i-1}) = R_{i-1}^T (t_i - t_{i-1}) — so both are invariant by
    construction, which the block decomposition confirms numerically (their AUCs are
    unchanged to 4 decimal places with and without this).
    """
    f = np.array(feats, copy=True)
    R = rotation_6d_to_matrix(torch.as_tensor(np.ascontiguousarray(f[:, 129:135])))
    horiz = np.delete(np.arange(3), up)                      # the two horizontal axes
    u = (R[0] @ torch.eye(3)[horiz[0]]).numpy()
    th = np.arctan2(u[horiz[1]], u[horiz[0]])
    c, s_ = float(np.cos(-th)), float(np.sin(-th))
    Q = torch.eye(3)
    Q[horiz[0], horiz[0]] = c;  Q[horiz[0], horiz[1]] = -s_
    Q[horiz[1], horiz[0]] = s_; Q[horiz[1], horiz[1]] = c
    f[:, 129:135] = (Q @ R)[:, :2, :].reshape(-1, 6).numpy()
    return f


FEATURE_BLOCKS = {"trans_delta": slice(0, 3), "body_pose": slice(3, 129),
                  "global_orient": slice(129, 135), "all": slice(0, 135)}


def block_decomposition(A, B, seed=0):
    """Which part of the 135-d feature separates the corpora, before TMR sees any of it.

    Each clip is summarised by the mean and std over time of the block, so this compares
    distributions of interpretable features rather than of a learned embedding. Both sides
    must already be at the SAME fps — `trans_delta` is a per-frame displacement, so an fps
    mismatch alone would separate the corpora through that block and through no other.
    """
    out = {}
    # Clip length on its own, as the confound baseline. Length matching is never exact —
    # HumanML3D has few clips as short as MotionFix's — so this says how much of any
    # separation below could be residual duration rather than motion content.
    la = np.array([[len(a)] for a in A], float)
    lb = np.array([[len(b)] for b in B], float)
    r = two_sample(la, lb, seed=seed)
    out["clip_length_only"] = {"auc": r["auc"], "acc": r["acc"], "knn_acc": r["knn_acc"]}
    for name, sl in FEATURE_BLOCKS.items():
        fa = np.stack([np.concatenate([a[:, sl].mean(0), a[:, sl].std(0)]) for a in A])
        fb = np.stack([np.concatenate([b[:, sl].mean(0), b[:, sl].std(0)]) for b in B])
        r = two_sample(fa, fb, seed=seed)
        out[name] = {"auc": r["auc"], "acc": r["acc"], "knn_acc": r["knn_acc"]}
    return out


@torch.no_grad()
def encode_motions(clips, tmr, stats, device, src_fps, tmr_fps, batch=32,
                   canonical=True):
    """Raw (T, 135) features -> TMR motion embeddings (N, 256).

    `src_fps != tmr_fps` triggers a resample through raw SMPL space, the same round trip
    `probe_tmr_laterality.py` uses. Returns the prepared (resampled, canonicalised,
    UN-normalised) features too, so the block decomposition reads exactly what TMR saw.
    """
    from data.smplh_features import features_to_smpl, resample_motion, smplh_to_features

    mean, std = stats
    mean_np, std_np = mean.cpu().numpy(), std.cpu().numpy()
    prepared, raw, lengths = [], [], []
    for _, feats in clips:
        if src_fps != tmr_fps:
            rots, trans = features_to_smpl(feats)
            rots, trans = resample_motion(rots, trans, src_fps, tmr_fps)
            feats = smplh_to_features(rots, trans)
        if canonical:
            feats = canonicalise_heading(feats)
        raw.append(feats)
        prepared.append((feats - mean_np) / (std_np + 1e-12))
        lengths.append(len(feats))

    out = []
    for i in log.progress(range(0, len(prepared), batch), desc="encoding motion"):
        chunk = prepared[i:i + batch]
        T = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), T, chunk[0].shape[-1], device=device)
        for j, c in enumerate(chunk):
            x[j, :len(c)] = torch.as_tensor(c, dtype=torch.float32, device=device)
        ln = [len(c) for c in chunk]
        mask = (torch.arange(T, device=device)[None]
                < torch.tensor(ln, device=device)[:, None])
        emb = tmr({"x": x, "mask": mask})[:, 0]           # mu of the VAE head
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0), np.array(lengths), raw


# ── statistics ──────────────────────────────────────────────────────────────────
def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def nn_cosine(query, bank, block=512):
    """Cosine to the NEAREST bank vector, per query row."""
    q, b = _unit(query), _unit(bank)
    best = np.empty(len(q), np.float32)
    for i in range(0, len(q), block):
        best[i:i + block] = (q[i:i + block] @ b.T).max(1)
    return best


def two_sample(A, B, seed=0, folds=5, k=5):
    """Classifier two-sample test: can a model recover which corpus a point came from?

    Decision values come from `cross_val_predict`, so every point is scored by a fold that
    never saw it. Fitting one probe on everything and plotting its margins would draw a
    separation the probe partly memorised.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([A, B], 0)
    y = np.concatenate([np.zeros(len(A), int), np.ones(len(B), int)])
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    margin = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
    pred = (margin > 0).astype(int)

    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, metric="cosine"))
    knn_pred = cross_val_predict(knn, X, y, cv=cv)

    ca, cb = A.mean(0), B.mean(0)
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    return {
        "n_a": len(A), "n_b": len(B),
        "auc": float(roc_auc_score(y, margin)),
        "acc": float((pred == y).mean()),
        # The series are not always the same size (the HumanML3D held-out pool is thin
        # inside MotionFix's length range), so raw accuracy has to be read against a
        # shifted chance baseline. Balanced accuracy is chance-0.5 whatever the prevalence
        # and is what the figure shows.
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "knn_balanced_acc": float(balanced_accuracy_score(y, knn_pred)),
        "knn_acc": float((knn_pred == y).mean()),
        "chance": float(max(len(A), len(B)) / (len(A) + len(B))),
        "centroid_cos": cos,
        "centroid_sep": float(np.linalg.norm(ca - cb)),
        "spread_a": float(np.linalg.norm(A - ca, axis=1).mean()),
        "spread_b": float(np.linalg.norm(B - cb, axis=1).mean()),
        "margin": margin, "labels": y,
    }


def embed_2d(series, perplexity=30, seed=0, pca_dim=50):
    """One shared t-SNE over every series, PCA-reduced first.

    ⚠ Read the distances with care: t-SNE preserves neighbourhoods, not gaps. Cluster
    separation and cluster size on this map are NOT quantities — the two-sample numbers
    beside it are.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.concatenate([s.X for s in series], 0)
    d = min(pca_dim, X.shape[1], X.shape[0] - 1)
    Xp = PCA(n_components=d, random_state=seed).fit_transform(X)
    Y = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity,
             random_state=seed, max_iter=1000).fit_transform(Xp)
    bounds, i = [], 0
    for s in series:
        bounds.append((i, i + len(s)))
        i += len(s)
    return [Y[a:b] for a, b in bounds]
