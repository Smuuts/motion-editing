"""
Self-attention structure: capture, affinity marginals, DiffSeg segmentation, metrics.

The measurement side of src/analyse_self_attention.py — see that script's docstring for
what the probe is for and how to read the result.

Layout: tokens are (frame, group) cells flattened as f*G + g, so an (N, N) token↔token
map marginalises two ways — over frame pairs into a group×group affinity, and over
group pairs into a frame×frame affinity. Everything is reported against an explicit
random baseline (1/G, 1/F, or a shuffled-label NMI floor), because these numbers mean
nothing on their own.
"""

import numpy as np
import torch

from model.layers import SelfAttention


# ── capture ──────────────────────────────────────────────────────────────────────
# The model API exposes get_attn_maps() for CROSS-attention only; self-attention
# capture is an inference-only diagnostic flag on the SelfAttention modules themselves
# (see model/layers.py). Walking model.modules() yields them
# in registration order, which is block order for both GroupDiT and GroupMotionUNet.

def self_attn_modules(model) -> list[SelfAttention]:
    return [m for m in model.modules() if isinstance(m, SelfAttention)]


def set_capture(model, on: bool):
    for m in self_attn_modules(model):
        m.store_attn = on
        if not on:
            m.last_attn_map = None


def read_maps(model, expect_n: int, keep_layers=None):
    """Stored (N, N) maps whose token count matches expect_n, cleared as we go.

    Blocks at a reduced temporal resolution (GroupCLR U-Net levels) are skipped rather
    than resampled: unet.get_attn_maps() can upsample a rectangular (tokens × text) map
    along one axis, but a square token↔token affinity has no honest one-axis
    resampling — interpolating both axes would manufacture exactly the structure this
    probe is trying to measure. Returns (maps, n_skipped).
    """
    kept, skipped = [], 0
    for i, m in enumerate(self_attn_modules(model)):
        a = m.last_attn_map
        m.last_attn_map = None
        if a is None:
            continue
        if a.shape[-1] != expect_n:
            skipped += 1
            continue
        if keep_layers is not None and i not in keep_layers:
            continue
        kept.append(a[0])                                  # (N, N), batch of 1
    return kept, skipped


@torch.no_grad()
def aggregate(model, xs, context, timesteps, expect_n, keep_layers=None):
    """Mean self-attention over {heads (already), layers, timesteps} → (N, N), plus a
    per-layer stack (n_layers, N, N) for the layer-profile panel and the skip count.

    Raises RuntimeError if no block ran at the expected token count, so a caller
    sweeping several checkpoints can skip that one; the CLI turns it into an exit.
    """
    set_capture(model, True)
    total, per_layer, layer_n, skipped_total = None, {}, 0, 0
    device = context.device if context is not None else xs.device
    for t in timesteps:
        x_t = xs[t].to(device)
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        model(x_t, t_b, context)
        maps, skipped = read_maps(model, expect_n, keep_layers)
        skipped_total = max(skipped_total, skipped)
        if not maps:
            continue
        layer_n = len(maps)
        for li, a in enumerate(maps):
            a = a.float()
            per_layer[li] = a.clone() if li not in per_layer else per_layer[li] + a
            total = a.clone() if total is None else total + a
    set_capture(model, False)
    if total is None:
        raise RuntimeError(
            "No self-attention maps captured. If this is a GroupCLR U-Net, no block "
            "runs at full temporal resolution for this clip length.")
    n_steps = len(list(timesteps))
    stack = torch.stack([per_layer[li] / n_steps for li in sorted(per_layer)], dim=0)
    return (total / (n_steps * max(layer_n, 1))).cpu().numpy(), stack.cpu().numpy(), skipped_total


# ── marginals ────────────────────────────────────────────────────────────────────

def group_affinity(A, F, G):
    """C[g,g'] = mean over (f,f') of A[(f,g),(f',g')] — 'how much does group g attend to
    group g' anywhere in time'. Rows renormalised, so the random baseline is exactly 1/G."""
    C = A.reshape(F, G, F, G).mean(axis=(0, 2))
    return C / np.clip(C.sum(axis=1, keepdims=True), 1e-12, None)


def frame_affinity(A, F, G):
    """R[f,f'] = mean over (g,g') of A[(f,g),(f',g')], rows renormalised (baseline 1/F)."""
    R = A.reshape(F, G, F, G).mean(axis=(1, 3))
    return R / np.clip(R.sum(axis=1, keepdims=True), 1e-12, None)


def diagonality(M):
    """Fraction of row-normalised mass on the diagonal. Random = 1/len(M)."""
    return float(np.trace(M) / max(M.sum(), 1e-12))


def incoming_attention(A, F, G):
    """(F, G) attention received per cell (mean over the querying axes) — the quantity
    compared against the source's own motion. Outgoing attention is uninformative:
    rows are normalised distributions, so their mean is the constant 1/N."""
    return A.reshape(F, G, F, G).mean(axis=(0, 1))


# ── laterality ───────────────────────────────────────────────────────────────────

def laterality_pairs(labels):
    """[(i_left, i_right, base_name)] for either axis naming — 'left_arm'/'right_arm'
    ('parts') and 'L_Wrist'/'R_Wrist' ('joints')."""
    pairs = []
    for i, n in enumerate(labels):
        for lp, rp in (("left_", "right_"), ("L_", "R_")):
            if n.startswith(lp):
                mirror = rp + n[len(lp):]
                if mirror in labels:
                    pairs.append((i, labels.index(mirror), n[len(lp):]))
    return pairs


def affinity_laterality(C, labels):
    """Disentangle the two things a naive 'self minus cross-side' score conflates.

    Per L/R pair, on the row-normalised group affinity:
      self    C[l,l]  — the diagonal, inflated for EVERY group whenever the matrix is
                        diagonal at all, so it is not laterality evidence on its own
      mirror  C[l,r]  — affinity to the opposite side of the same limb
      other   mean C[l,k], k ∉ {l, r}
    Hence `self_vs_mirror` > 0 ⇒ the sides ARE distinguished, while
    `mirror_vs_other` > 0 ⇒ the sides are specially LINKED (bilateral symmetry — the
    opposite, and the more likely outcome given every other probe's result). Reporting
    only the first would let plain diagonality masquerade as laterality.
    """
    G = len(labels)
    out = {}
    for il, ir, base in laterality_pairs(labels):
        others = [k for k in range(G) if k not in (il, ir)]
        selfa  = 0.5 * (C[il, il] + C[ir, ir])
        mirror = 0.5 * (C[il, ir] + C[ir, il])
        other  = 0.5 * (C[il, others].mean() + C[ir, others].mean()) if others else 0.0
        out[base] = {
            "self": float(selfa), "mirror": float(mirror), "other": float(other),
            "self_vs_mirror": float(selfa - mirror),
            "mirror_vs_other": float(mirror - other),
        }
    return out


def segment_laterality(labels_fg, glabels):
    """Does the segmentation separate a limb's two sides?

    P(same segment) for a mirror pair at the SAME frame, against the same quantity for
    non-mirror pairs at the same frame. Same-frame throughout, so the comparison isn't
    confounded by the temporal structure that dominates this segmentation (an earlier
    version compared a group against its own next frame, which measures temporal
    persistence). mirror ≈ other ⇒ no laterality.
    """
    pairs = laterality_pairs(glabels)
    if not pairs:
        return {}
    F, G = labels_fg.shape
    mirror, other = [], []
    for il, ir, _ in pairs:
        for f in range(F):
            mirror.append(float(labels_fg[f, il] == labels_fg[f, ir]))
            other += [float(labels_fg[f, il] == labels_fg[f, k])
                      for k in range(G) if k not in (il, ir)]
    return {"p_same_segment_mirror_pair": float(np.mean(mirror)),
            "p_same_segment_other_pairs": float(np.mean(other))}


def pick_segment_laterality(sweep, labels_fg, glabels, F, G):
    """Segment-level laterality read at the granularity where it is meaningful.

    At a very fine segmentation every same-frame pair lands in a different segment, so
    both probabilities collapse to 0 and the comparison is vacuous. Evaluate across the
    whole tau sweep and report the row with the most dynamic range (largest
    `other_pairs`) — the coarsest granularity that still separates anything.
    """
    if not sweep:
        return segment_laterality(labels_fg, glabels)
    cands = []
    for r in sweep:
        if r["n_segments"] <= 1:
            continue
        s = segment_laterality(r["_labels"].reshape(F, G), glabels)
        if s:
            cands.append({**s, "n_segments": r["n_segments"], "tau": r["tau"]})
    return max(cands, key=lambda s: s["p_same_segment_other_pairs"]) if cands else {}


# ── DiffSeg: iterative KL-threshold merging (no cluster count) ───────────────────

def _sym_kl(P):
    """Pairwise symmetric KL between rows of P (K, N) — the DiffSeg merge distance."""
    Q = np.clip(P, 1e-12, None)
    logQ = np.log(Q)
    # KL(i||j) = sum_n P_i,n (log P_i,n - log P_j,n)
    cross = Q @ logQ.T                       # (K, K): sum_n P_i,n log P_j,n
    self_e = np.sum(Q * logQ, axis=1)        # (K,)
    kl = self_e[:, None] - cross
    return 0.5 * (kl + kl.T)


def _anchor_maps(A, F, G, anchor_stride):
    anchors = [f * G + g for f in range(0, F, anchor_stride) for g in range(G)]
    P = A[anchors].astype(np.float64)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)


def diffseg(A, F, G, anchor_stride=4, iters=8, tau=1.0):
    """DiffSeg (arXiv 2308.12469) ported to the (F·G) token grid.

    Anchors are sampled on a stride over frames × all groups (the 1-D analogue of
    DiffSeg's image grid). Each round merges every proposal with all proposals within
    symmetric-KL `tau` of it, deduplicates and renormalises, until it converges. Labels
    are the argmax over proposals. Returns (labels (N,), n_segments).
    """
    P = _anchor_maps(A, F, G, anchor_stride)
    for _ in range(iters):
        if len(P) <= 1:
            break
        D = _sym_kl(P)
        merged, taken = [], np.zeros(len(P), dtype=bool)
        for i in range(len(P)):
            if taken[i]:
                continue
            grp = np.where((D[i] <= tau) & (~taken))[0]
            taken[grp] = True
            m = P[grp].mean(axis=0)
            merged.append(m / max(m.sum(), 1e-12))
        new = np.stack(merged)
        converged = len(new) == len(P)
        P = new
        if converged:
            break
    # P is (K, N): each cell takes the proposal claiming the most mass there.
    return P.argmax(axis=0), len(P)


def diffseg_tau_sweep(A, F, G, anchor_stride, iters, group_ref, time_ref, rng):
    """Run DiffSeg across a range of merge thresholds instead of trusting one.

    A single tau is the whole fragility of a DiffSeg port: too small and every anchor
    stays its own segment, too large and everything collapses into one — neither says
    anything about the model. Sweeping percentiles of the anchors' own pairwise-KL
    distribution turns the hyperparameter into a measurement. Two things keep the
    comparison fair:
    * raw NMI is not comparable across segment counts (a finer segmentation scores
      higher against any reference), so every row also gets a *shuffled* baseline — same
      labels permuted, hence the same segment-size distribution — and the decisive
      quantity is the gap, measured − shuffled;
    * the operating point is the most favourable non-degenerate tau (max group gap), so
      a flat result cannot be blamed on a badly chosen threshold.

    Returns (rows, best_row, kl_stats).
    """
    P0 = _anchor_maps(A, F, G, anchor_stride)
    D0 = _sym_kl(P0)
    off = D0[~np.eye(len(D0), dtype=bool)]
    rows = []
    for p in [0.1, 0.5, 1, 2, 5, 10, 25, 50]:
        tau = float(np.percentile(off, p)) if off.size else 0.0
        labels, n_seg = diffseg(A, F, G, anchor_stride, iters, tau)
        g, t = nmi(labels, group_ref), nmi(labels, time_ref)
        gs, ts = shuffled_nmi(labels, group_ref, rng), shuffled_nmi(labels, time_ref, rng)
        rows.append({
            "percentile": p, "tau": tau, "n_segments": int(n_seg),
            "nmi_group": g, "nmi_group_shuffled": gs, "nmi_group_gap": g - gs,
            "nmi_time": t, "nmi_time_shuffled": ts, "nmi_time_gap": t - ts,
            "_labels": labels,
        })
    usable = [r for r in rows if r["n_segments"] > 1] or rows
    best = max(usable, key=lambda r: r["nmi_group_gap"])
    kl_stats = {
        "kl_min": float(off.min()) if off.size else 0.0,
        "kl_median": float(np.median(off)) if off.size else 0.0,
        "kl_max": float(off.max()) if off.size else 0.0,
        "n_anchors": int(len(P0)),
    }
    return rows, best, kl_stats


# ── reference labelings + NMI ────────────────────────────────────────────────────

def group_reference(F, G):
    """Body-part reference labeling over the (F·G) token grid."""
    return np.tile(np.arange(G), F)


def time_reference(F, G, n_bins):
    """Coarse temporal-binning reference labeling over the (F·G) token grid."""
    return np.repeat(np.minimum((np.arange(F) * n_bins) // F, n_bins - 1), G)


def nmi(a, b):
    """Normalised mutual information (symmetric, sqrt-normalised) between two integer
    labelings. Implemented locally — sklearn is not a project dependency."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), 1.0)
    joint /= n
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / max(np.sqrt(ha * hb), 1e-12)


def shuffled_nmi(labels, ref, rng, trials=20):
    """Mean NMI of a randomly permuted labeling against the same reference — the 'what
    would nothing look like' floor these scores must beat."""
    return float(np.mean([nmi(rng.permutation(labels), ref) for _ in range(trials)]))
