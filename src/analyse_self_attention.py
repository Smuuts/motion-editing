"""
Probe the SELF-attention pathway for emergent (frame, body-part) structure — Option 7
of docs/AttentionGrounding_Options.md, Family A of docs/ImplicitMask_Research.md.

Why this probe exists
---------------------
Every mask probe in this project so far read *cross*-attention (M1) or the noise
contrast ψ (M2), and both are measured to be instruction-invariant and
source-dynamics-driven (docs/FINDINGS.md). Self-attention is the one attention
pathway that was never examined. DiffSeg ("Diffuse, Attend, and Segment",
arXiv 2308.12469) produces a segmentation from **self-attention only, no text and no
labels**, and MotionCLR (arXiv 2410.18977) shows motion self-attention carries
repetition/segment structure — so there is a plausible signal here.

What a positive result would buy us: an automatic, training-free spatial partition of
the clip (data-driven regions instead of the fixed 7-group axis) to intersect with an
instruction-driven selector. What it cannot buy us: instruction selectivity — self-
attention is text-free, so it can find coherent segments but not decide which one the
edit text means. A clean negative is equally useful: it closes "is there ANY usable
emergent structure in this backbone without a retrain?".

What is measured
----------------
1. **Affinity structure** (no clustering, the robust part). Aggregate the token↔token
   self-attention over heads, layers and inversion timesteps into `A` (N×N, N=F·G),
   then marginalise it two ways:
     - group affinity  C[g,g'] = mean_{f,f'} A[(f,g),(f',g')]   → is attention
       body-part structured (diagonal) and does it separate left from right?
     - frame affinity  R[f,f'] = mean_{g,g'} A[(f,g),(f',g')]   → is it temporally
       blocked (the structure MotionCLR reports)?
   Reported against explicit random baselines (1/G, 1/F), because "0.2" means nothing
   on its own.
2. **DiffSeg segmentation.** Iterative KL-threshold merging of anchor attention maps
   (no cluster count, deterministic), then argmax → a label per (frame, group) cell.
   Scored by normalised mutual information against the body-part axis and against a
   coarse temporal binning, each with a shuffled baseline.
3. **Laterality**, the axis every other probe failed: does self-attention distinguish
   left_arm from right_arm? Measured both on the affinity (within-side minus
   cross-side) and on the segmentation (do the two sides land in different segments).
4. **Text-invariance check.** Self-attention is text-free *by construction* only in
   layer 0 — every later layer sees the cross-attention residual, so its self-attention
   is indirectly text-conditioned. Running the same clip under contrasting instructions
   measures whether any instruction signal survives into the self-attention structure.
   Near-identical maps confirm the "instruction-agnostic" premise of Family A.

Outputs (per source clip, in --out_dir)
---------------------------------------
  <clip>_selfattn_structure.png   group affinity C, frame affinity R, per-layer
                                  diagonality profile, example anchor maps.
  <clip>_selfattn_segments.png    the DiffSeg label map over (frame × group), the
                                  alignment/laterality metrics vs their baselines, and
                                  the instruction×instruction affinity correlation.
  <clip>_selfattn.json            every number in both figures.

Usage
-----
    python src/analyse_self_attention.py \
        --checkpoint runs/exp_smplh/checkpoint_latest \
        --data_root  data/HumanML3D_smplh \
        --source 0 \
        --out_dir eval_results/self_attention

    # Restrict the timestep window (cf. PROGRESS item B.7c / Motion-Adapter's
    # observation that attention stops aligning with motion at high noise):
    python src/analyse_self_attention.py --checkpoint ... --data_root ... --source 0 \
        --t_min 250 --t_max 750 --mask_timesteps 40

Notes
-----
* Backbone-agnostic in the same way as visualise_mask_problem.py: it reuses the real
  editing stack (MotionEditor inversion), so it runs on any checkpoint the editor runs
  on. For the GroupCLR U-Net, blocks live at several temporal resolutions and a square
  token↔token map cannot be honestly resampled to full resolution the way
  unet.get_attn_maps() resamples the rectangular cross-attention maps — so only blocks
  whose token count is already F·G are used, and the number skipped is reported.
* Memory: capture is head-meaned (see model/layers.py SelfAttention). Peak is one
  (N, N) float32 map per block; N = F·G, so trim with --max_frames on a 22-token axis.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.layers import SelfAttention
from model.body_groups import group_names, resolve_group_context
from editing import MotionEditor
from utils.model_io import load_model
from sample_model import _smplh_body_model

from visualise_mask_problem import load_source, source_activity, flat_corr


# Same contrasting set as visualise_mask_problem.py, so the text-invariance number
# here is directly comparable to the M1/M2 instruction-invariance numbers there.
DEFAULT_INSTRUCTIONS = [
    "raise the left arm",
    "raise the right arm",
    "kick with the left leg",
    "kick with the right leg",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir (config.json + ema.pt/model.pt).")
    p.add_argument("--data_root", required=True,
                   help="Data root (needs Mean.npy, Std.npy, <split>.txt, new_joint_vecs/).")
    p.add_argument("--source", required=True,
                   help="Source clip: integer index into --split, or a path to a raw "
                        "(T, D) .npy feature file.")
    p.add_argument("--instruction", action="append", dest="instructions", default=None,
                   help="Instruction for the text-invariance check. Repeat for the "
                        f"contrasting set (default: {DEFAULT_INSTRUCTIONS}). The "
                        "primary analysis always uses the null (text-free) context.")
    p.add_argument("--mask_timesteps", type=int, default=40,
                   help="Evenly-spaced inversion timesteps swept and averaged (default 40).")
    p.add_argument("--t_min", type=int, default=1,
                   help="Lowest timestep included in the sweep (default 1).")
    p.add_argument("--t_max", type=int, default=None,
                   help="Highest timestep included (default: T-1). Motion-Adapter "
                        "(arXiv 2604.16135) reports attention stops aligning with the "
                        "motion above t≈750; --t_max 750 tests that here.")
    p.add_argument("--layers", default=None,
                   help="Comma-separated block indices to read (default: all). Middle "
                        "layers carry semantic binding in the image literature.")
    p.add_argument("--anchor_stride", type=int, default=4,
                   help="DiffSeg anchor sampling: every Nth frame, all groups (default 4).")
    p.add_argument("--merge_iters", type=int, default=8,
                   help="DiffSeg iterative-merge rounds (default 8).")
    p.add_argument("--kl_tau", type=float, default=None,
                   help="KL merge threshold. Default: data-adaptive (10th percentile of "
                        "the initial pairwise KL), avoiding a magic constant tuned on SD.")
    p.add_argument("--time_bins", type=int, default=8,
                   help="Coarse temporal bins for the temporal-NMI baseline (default 8).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh")
    p.add_argument("--out_dir", default="eval_results/self_attention")
    p.add_argument("--no_ema", action="store_true", help="Load model.pt instead of ema.pt.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the inversion noise and the shuffled NMI baselines "
                        "(the inversion is stochastic, so this makes runs comparable).")
    return p.parse_args()


# ── self-attention capture ───────────────────────────────────────────────────────
# The model API exposes get_attn_maps() for CROSS-attention only; self-attention
# capture is an inference-only diagnostic flag on the SelfAttention modules
# themselves (see model/layers.py). Walking model.modules() yields them in
# registration order, which is block order for both GroupDiT and GroupMotionUNet.
def self_attn_modules(model) -> list[SelfAttention]:
    return [m for m in model.modules() if isinstance(m, SelfAttention)]


def set_capture(model, on: bool):
    for m in self_attn_modules(model):
        m.store_attn = on
        if not on:
            m.last_attn_map = None


def read_maps(model, expect_n: int, keep_layers=None):
    """Stored (N, N) maps whose token count matches expect_n, cleared as we go.

    Blocks at a reduced temporal resolution (GroupCLR U-Net levels) are skipped
    rather than resampled: unet.get_attn_maps() can upsample a rectangular
    (tokens × text) map along one axis, but a square token↔token affinity has no
    honest one-axis resampling — interpolating both axes would manufacture
    structure, which is exactly what this probe is trying to measure.
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
def aggregate_self_attention(model, xs, context, timesteps, expect_n, keep_layers=None):
    """Mean self-attention over {heads (already), layers, timesteps} → (N, N) plus a
    per-layer stack (n_layers, N, N) for the layer-profile panel."""
    set_capture(model, True)
    total = None
    per_layer, layer_n = {}, 0
    skipped_total = 0
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
        raise SystemExit(
            "No self-attention maps captured. If this is a GroupCLR U-Net, no block "
            "runs at full temporal resolution for this clip length.")
    n_steps = len(list(timesteps))
    stack = torch.stack([per_layer[li] / n_steps for li in sorted(per_layer)], dim=0)
    return (total / (n_steps * max(layer_n, 1))).cpu().numpy(), stack.cpu().numpy(), skipped_total


# ── marginals ────────────────────────────────────────────────────────────────────
def group_affinity(A, F, G):
    """C[g,g'] = mean over (f,f') of A[(f,g),(f',g')] — 'how much does group g attend
    to group g' anywhere in time'. Rows renormalised so each sums to 1, making the
    random baseline exactly 1/G."""
    C = A.reshape(F, G, F, G).mean(axis=(0, 2))
    return C / np.clip(C.sum(axis=1, keepdims=True), 1e-12, None)


def frame_affinity(A, F, G):
    """R[f,f'] = mean over (g,g') of A[(f,g),(f',g')], rows renormalised (baseline 1/F)."""
    R = A.reshape(F, G, F, G).mean(axis=(1, 3))
    return R / np.clip(R.sum(axis=1, keepdims=True), 1e-12, None)


def diagonality(M):
    """Fraction of row-normalised mass on the diagonal. Random = 1/len(M)."""
    return float(np.trace(M) / max(M.sum(), 1e-12))


def laterality_pairs(labels):
    """[(i_left, i_right, base_name)] for the axis names — 'left_arm'/'right_arm'
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

    For each L/R pair, on the row-normalised group affinity:
      self         C[l,l]  — the diagonal (inflated for EVERY group whenever the
                             matrix is diagonal at all, so it is not evidence of
                             laterality on its own)
      mirror       C[l,r]  — affinity to the opposite side of the same limb
      other        mean C[l,k] over k ∉ {l, r} — affinity to unrelated groups
    Then:
      self_vs_mirror  = self − mirror   >0 ⇒ the sides ARE distinguished
      mirror_vs_other = mirror − other  >0 ⇒ the sides are specially LINKED
                                        (bilateral symmetry — the opposite of
                                        distinguishing them, and the more likely
                                        outcome given every other probe's result)
    Reporting only the first number would let plain diagonality masquerade as
    laterality, which is exactly the mistake this project's findings warn about.
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
    DiffSeg's image grid). Each round: merge every proposal with all proposals within
    symmetric-KL `tau` of it, deduplicate, renormalise, until it converges. Labels =
    argmax over proposals. Returns (labels (N,), n_segments).
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
    # P is (K, N): for each cell, the proposal that claims the most mass there.
    return P.argmax(axis=0), len(P)


def diffseg_tau_sweep(A, F, G, anchor_stride, iters, group_ref, time_ref, rng):
    """Run DiffSeg across a range of merge thresholds instead of trusting one.

    A single tau is the whole fragility of a DiffSeg port: too small and every anchor
    stays its own segment, too large and everything collapses into one. Neither says
    anything about the model. Sweeping percentiles of the anchors' own pairwise-KL
    distribution turns the hyperparameter into a measurement.

    Two things make the comparison fair:
    * **Raw NMI is not comparable across segment counts** — a fine segmentation scores
      higher against any reference just by being fine. So every row also gets a
      *shuffled* baseline (the same labels permuted, hence the same segment-size
      distribution), and the decisive quantity is the **gap** measured − shuffled.
    * **The operating point is the most favourable non-degenerate tau** (max group
      gap), not a fixed guess. If the headline is still ≈0 there, the negative cannot
      be blamed on a badly chosen threshold — which is the whole point of running a
      sweep in a probe whose likely outcome is negative.
    """
    P0 = _anchor_maps(A, F, G, anchor_stride)
    D0 = _sym_kl(P0)
    off = D0[~np.eye(len(D0), dtype=bool)]
    rows = []
    for p in [0.1, 0.5, 1, 2, 5, 10, 25, 50]:
        tau = float(np.percentile(off, p)) if off.size else 0.0
        labels, n_seg = diffseg(A, F, G, anchor_stride, iters, tau)
        g, t = nmi(labels, group_ref), nmi(labels, time_ref)
        gs = shuffled_nmi(labels, group_ref, rng)
        ts = shuffled_nmi(labels, time_ref, rng)
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


# ── metrics ──────────────────────────────────────────────────────────────────────
def nmi(a, b):
    """Normalised mutual information (symmetric, sqrt-normalised) between two
    integer labelings. Implemented locally — sklearn is not a project dependency."""
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
    """Mean NMI of a randomly permuted labeling against the same reference — the
    'what would nothing look like' floor these scores must beat."""
    return float(np.mean([nmi(rng.permutation(labels), ref) for _ in range(trials)]))


def segment_laterality(labels_fg, glabels):
    """Does the segmentation separate a limb's two sides?

    P(same segment) for a mirror pair (left_arm, right_arm) at the SAME frame, against
    the same quantity for non-mirror pairs (left_arm, spine / left_leg / …) at the same
    frame. Same-frame throughout, so the comparison is not confounded by the temporal
    structure that dominates this segmentation — an earlier version compared against a
    group's own next frame, which measures temporal persistence, not laterality.

    mirror ≈ other ⇒ the two sides of a limb are no more distinguishable from each
    other than any two unrelated groups, i.e. no laterality.
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

    At a very fine segmentation every same-frame pair lands in different segments, so
    both probabilities collapse to 0 and the mirror-vs-other comparison is vacuous.
    Evaluate across the whole tau sweep and report the row with the most dynamic range
    (largest `other_pairs`), which is the coarsest granularity that still separates
    anything. Falls back to the operating point if there is no sweep.
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
    if not cands:
        return {}
    return max(cands, key=lambda s: s["p_same_segment_other_pairs"])


# ── figures ──────────────────────────────────────────────────────────────────────
def _matshow(ax, M, title, ylabels=None, xlabels=None, cmap="magma", annotate=False,
             aspect="equal"):
    """Heatmap with independently-labelled axes — the two axes are NOT always the
    same axis here (group×group and frame×frame are square, but the per-cell maps are
    group×frame), so x and y labels must be passed separately."""
    im = ax.imshow(M, cmap=cmap, interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=8.5)
    for setter, ticks, labs, rot in (
            (ax.set_yticks, ax.set_yticklabels, ylabels, 0),
            (ax.set_xticks, ax.set_xticklabels, xlabels, 45)):
        if labs is not None and len(labs) <= 24:
            setter(range(len(labs)))
            ticks(labs, fontsize=5.5, rotation=rot, **({"ha": "right"} if rot else {}))
        else:
            setter([])
    if annotate and ylabels is not None and xlabels is not None and len(ylabels) <= 8:
        mx = M.max()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] < 0.6 * mx else "black")
    return im


def plot_structure(clip_id, caption, C, R, layer_diag, glabels, anchors, F, G,
                   src_act, out_path):
    fig = plt.figure(figsize=(13, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.3, top=0.86, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    im = _matshow(ax, C, f"group affinity C  (row-normalised)\n"
                         f"diagonality {diagonality(C):.3f}  vs random {1/G:.3f}",
                  ylabels=glabels, xlabels=glabels, annotate=True)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = _matshow(ax, R, f"frame affinity R  (row-normalised)\n"
                         f"diagonality {diagonality(R):.4f}  vs random {1/F:.4f}")
    ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("frame", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(range(len(layer_diag)), layer_diag, "o-", color="#4c72b0", label="group")
    ax.axhline(1 / G, color="k", ls="--", lw=0.8, label=f"random 1/G = {1/G:.3f}")
    ax.set_xlabel("block index", fontsize=7)
    ax.set_ylabel("group-affinity diagonality", fontsize=7)
    ax.set_title("Per-layer body-part structure\n(is any single block grounded?)",
                 fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[1, 0])
    _matshow(ax, src_act.T, "SOURCE motion |Δx0|  (reference)",
             ylabels=glabels, cmap="cividis", aspect="auto")
    ax.set_xlabel("frame", fontsize=7)

    for k, (row, name) in enumerate(anchors[:2]):
        ax = fig.add_subplot(gs[1, 1 + k])
        _matshow(ax, row.reshape(F, G).T,
                 f"attention FROM cell {name}\n(where does one token look?)",
                 ylabels=glabels, aspect="auto")
        ax.set_xlabel("frame", fontsize=7)

    cap = (caption or "").strip()
    cap = cap if len(cap) <= 80 else cap[:79] + "…"
    fig.suptitle(
        "Self-attention structure (text-free readout)  ·  clip "
        f"{clip_id}" + (f'  ·  "{cap}"' if cap else "") + "\n"
        "diagonal C ⇒ body-part structured;  blocked R ⇒ temporally segmented",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_segments(clip_id, labels_fg, glabels, metrics, instr_corr, instructions,
                  out_path):
    sweep = metrics.get("tau_sweep", [])
    fig = plt.figure(figsize=(16, 4.6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.42, top=0.76, bottom=0.2,
                  width_ratios=[1.5, 1.1, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(labels_fg.T, aspect="auto", cmap="tab20", interpolation="nearest")
    ax.set_yticks(range(len(glabels)))
    ax.set_yticklabels(glabels, fontsize=6)
    ax.set_xlabel("frame", fontsize=7)
    ax.set_title(f"DiffSeg segmentation  ({metrics['n_segments']} segments)\n"
                 "vertical bands = temporal; horizontal bands = body-part",
                 fontsize=8.5)

    # The threshold sweep: a DiffSeg port lives or dies on tau, so show the whole
    # curve rather than one point. A flat-zero NMI across every segment count is a
    # far stronger negative than a single bad setting.
    ax = fig.add_subplot(gs[0, 1])
    if sweep:
        nseg = [r["n_segments"] for r in sweep]
        ax.plot(nseg, [r["nmi_group_gap"] for r in sweep], "o-", ms=3,
                color="#4c72b0", label="body part")
        ax.plot(nseg, [r["nmi_time_gap"] for r in sweep], "s-", ms=3,
                color="#dd8452", label="time")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(metrics["n_segments"], color="k", ls="--", lw=0.8,
                   label="operating point")
        ax.set_xscale("log")
        ax.set_xlabel("segments (varying merge tau)", fontsize=7)
        ax.set_ylabel("NMI − shuffled baseline", fontsize=7)
        ax.legend(fontsize=6)
    ax.set_title("Threshold sweep\n(is ANY tau informative?)", fontsize=8.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 2])
    names = ["NMI vs\nbody part", "NMI vs\ntime bins"]
    vals  = [metrics["nmi_group"], metrics["nmi_time"]]
    base  = [metrics["nmi_group_shuffled"], metrics["nmi_time_shuffled"]]
    x = np.arange(len(names))
    ax.bar(x - 0.18, vals, width=0.34, color="#4c72b0", label="measured")
    ax.bar(x + 0.18, base, width=0.34, color="#bbbbbb", label="shuffled baseline")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, max(0.35, max(vals + base) * 1.25))
    ax.set_title("Does the segmentation align with\nthe body-part / time axes?",
                 fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 3])
    short = [e if len(e) <= 18 else e[:16] + "…" for e in instructions]
    im = ax.imshow(instr_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, fontsize=6, rotation=40, ha="right")
    ax.set_yticklabels(short, fontsize=6)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{instr_corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(instr_corr[i, j]) > 0.5 else "black")
    off = ~np.eye(len(short), dtype=bool)
    mean_off = instr_corr[off].mean() if len(short) > 1 else float("nan")
    ax.set_title(f"Self-attention across instructions\nmean off-diag r = {mean_off:.3f}"
                 " (→1 = text-free)", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    lat = metrics.get("segment_laterality", {})
    lat_txt = ""
    if lat:
        lat_txt = ("   ·   P(same segment) mirror pair "
                   f"{lat['p_same_segment_mirror_pair']:.2f} vs other pairs "
                   f"{lat['p_same_segment_other_pairs']:.2f} "
                   f"(at {lat.get('n_segments', '?')} segments)")
    fig.suptitle(f"Self-attention segmentation  ·  clip {clip_id}{lat_txt}",
                 fontsize=10, y=0.95)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Edit-friendly DDPM inversion draws fresh noise per step, so the whole probe is
    # stochastic; seed torch too or the segment count and metrics drift between runs.
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    arch = config.get("arch", "dit")
    print(f"feature_mode={feature_mode}  arch={arch}  is_group={is_group}  "
          f"group_mode={group_mode}")
    if feature_mode == "smplh":
        _smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    raw_feat, clip_id, F, caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)
    print(f"Source: {clip_id}  ({F} frames)   prompt: {caption!r}")

    editor = MotionEditor(model, schedule, device, is_group=is_group)
    glabels = group_names(group_mode) if is_group else ["all"]
    G = len(glabels) if is_group else 1
    N = F * G
    src_act = source_activity(x0, editor.group_channels, is_group)

    keep_layers = ({int(s) for s in args.layers.split(",")} if args.layers else None)
    t_max = args.t_max if args.t_max is not None else schedule.T - 1
    timesteps = torch.linspace(args.t_min, t_max, args.mask_timesteps).long().tolist()
    print(f"Sweeping {len(timesteps)} timesteps in [{args.t_min}, {t_max}], "
          f"N = F*G = {N}")

    print("Stage 1: inversion …")
    state = editor.invert(x0)

    # Primary readout: NULL context — self-attention with no text at all, the honest
    # "text-free structure" measurement Family A is about.
    print("Aggregating self-attention (null context) …")
    A_null, per_layer, skipped = aggregate_self_attention(
        model, state.xs, None, timesteps, N, keep_layers)
    if skipped:
        print(f"  note: skipped {skipped} block(s) at reduced temporal resolution "
              f"(U-Net levels); using the {per_layer.shape[0]} full-resolution block(s).")

    C = group_affinity(A_null, F, G)
    R = frame_affinity(A_null, F, G)
    layer_diag = [diagonality(group_affinity(per_layer[i], F, G))
                  for i in range(per_layer.shape[0])]

    group_ref = np.tile(np.arange(G), F)
    time_ref = np.repeat(np.minimum((np.arange(F) * args.time_bins) // F,
                                    args.time_bins - 1), G)

    print("DiffSeg merging …")
    if args.kl_tau is not None:
        labels, n_seg = diffseg(A_null, F, G, args.anchor_stride, args.merge_iters,
                                args.kl_tau)
        sweep, kl_stats, tau = [], {}, args.kl_tau
    else:
        sweep, best, kl_stats = diffseg_tau_sweep(
            A_null, F, G, args.anchor_stride, args.merge_iters, group_ref, time_ref, rng)
        labels, n_seg, tau = best["_labels"], best["n_segments"], best["tau"]
        print("  tau sweep: " + ", ".join(
            f"p{r['percentile']}→{r['n_segments']}seg/gap{r['nmi_group_gap']:+.2f}"
            for r in sweep))
        print(f"  operating point (max body-part gap): tau {tau:.4f} → {n_seg} "
              f"segments; anchor KL median {kl_stats['kl_median']:.3f}")
    labels_fg = labels.reshape(F, G)
    seg_lat = (pick_segment_laterality(sweep, labels_fg, glabels, F, G)
               if is_group else {})

    metrics = {
        "clip": clip_id, "caption": caption, "frames": F, "groups": G,
        "arch": arch, "feature_mode": feature_mode, "group_mode": group_mode,
        "timesteps": [args.t_min, t_max], "n_timesteps": len(timesteps),
        "blocks_used": int(per_layer.shape[0]), "blocks_skipped": int(skipped),
        "group_diagonality": diagonality(C), "group_diagonality_random": 1.0 / G,
        "frame_diagonality": diagonality(R), "frame_diagonality_random": 1.0 / F,
        "per_layer_group_diagonality": [float(v) for v in layer_diag],
        "affinity_laterality": affinity_laterality(C, glabels) if is_group else {},
        "n_segments": int(n_seg), "kl_tau": tau,
        "anchor_kl_stats": kl_stats,
        "tau_sweep": [{k: v for k, v in r.items() if k != "_labels"} for r in sweep],
        # Granularity-fair headline: NMI minus the same-granularity shuffled baseline,
        # maximised over the threshold sweep. This is an upper bound by construction.
        "best_nmi_group_gap_over_sweep": (max((r["nmi_group_gap"] for r in sweep
                                               if r["n_segments"] > 1), default=0.0)),
        "best_nmi_time_gap_over_sweep": (max((r["nmi_time_gap"] for r in sweep
                                              if r["n_segments"] > 1), default=0.0)),
        "nmi_group": nmi(labels, group_ref),
        "nmi_group_shuffled": shuffled_nmi(labels, group_ref, rng),
        "nmi_time": nmi(labels, time_ref),
        "nmi_time_shuffled": shuffled_nmi(labels, time_ref, rng),
        # Evaluated at the granularity where the statistic actually has dynamic range
        # (see pick_segment_laterality) — at the very fine operating point both
        # probabilities collapse to 0 and the comparison says nothing. The
        # threshold-free affinity_laterality above is the robust version of this.
        "segment_laterality": seg_lat,
        # Incoming attention per cell (mean over the querying axes) vs the source's
        # own motion: the self-attention analogue of the M1/M2 "is this really just a
        # source-dynamics detector?" number. Outgoing attention is uninformative here
        # (rows are normalised distributions, so their mean is constant 1/N).
        "corr_incoming_attention_vs_source_motion": flat_corr(
            A_null.reshape(F, G, F, G).mean(axis=(0, 1)), src_act),
    }

    # Text-invariance: is the self-attention structure moved at all by the instruction?
    instructions = args.instructions or list(DEFAULT_INSTRUCTIONS)
    print(f"Text-invariance check over {len(instructions)} instructions …")
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
    per_instr = []
    for e, ctx in zip(instructions, ctxs):
        A_e, _, _ = aggregate_self_attention(model, state.xs, ctx, timesteps, N,
                                             keep_layers)
        per_instr.append(A_e)
        print(f"  {e!r}: r vs null context = {flat_corr(A_e, A_null):.4f}")
    n = len(per_instr)
    instr_corr = np.eye(n)
    for i in range(n):
        for j in range(n):
            instr_corr[i, j] = flat_corr(per_instr[i], per_instr[j])
    off = ~np.eye(n, dtype=bool)
    metrics["instruction_invariance_mean_off_diag_r"] = (
        float(instr_corr[off].mean()) if n > 1 else float("nan"))
    metrics["corr_vs_null_context"] = [flat_corr(a, A_null) for a in per_instr]

    base = os.path.join(args.out_dir, f"{clip_id}_selfattn")
    anchors = [(A_null[(F // 2) * G + g], f"(frame {F//2}, {glabels[g]})")
               for g in range(min(2, G))]
    plot_structure(clip_id, caption, C, R, layer_diag, glabels, anchors, F, G,
                   src_act, base + "_structure.png")
    plot_segments(clip_id, labels_fg, glabels, metrics, instr_corr, instructions,
                  base + "_segments.png")
    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {base}.json")

    print("\n── summary ─────────────────────────────────────────────")
    print(f"group-affinity diagonality : {metrics['group_diagonality']:.3f}   "
          f"(random {1/G:.3f})   → is self-attention body-part structured?")
    print(f"frame-affinity diagonality : {metrics['frame_diagonality']:.4f}   "
          f"(random {1/F:.4f})  → is it temporally blocked?")
    print(f"incoming attn vs source |Δx0|: "
          f"{metrics['corr_incoming_attention_vs_source_motion']:+.3f}   "
          f"(→1 ⇒ another source-dynamics detector)")
    print(f"DiffSeg segments           : {n_seg}  (tau {tau:.4f})")
    print(f"NMI vs body-part axis      : {metrics['nmi_group']:.3f}   "
          f"(shuffled {metrics['nmi_group_shuffled']:.3f}  → gap "
          f"{metrics['nmi_group'] - metrics['nmi_group_shuffled']:+.3f}; "
          f"best gap over tau sweep {metrics['best_nmi_group_gap_over_sweep']:+.3f})")
    print(f"NMI vs time bins           : {metrics['nmi_time']:.3f}   "
          f"(shuffled {metrics['nmi_time_shuffled']:.3f}  → gap "
          f"{metrics['nmi_time'] - metrics['nmi_time_shuffled']:+.3f}; "
          f"best gap over tau sweep {metrics['best_nmi_time_gap_over_sweep']:+.3f})")
    for k, v in metrics["affinity_laterality"].items():
        print(f"laterality [{k:>4}]           : self−mirror {v['self_vs_mirror']:+.3f} "
              f"(>0 ⇒ sides distinguished)   "
              f"mirror−other {v['mirror_vs_other']:+.3f} "
              f"(>0 ⇒ sides LINKED, i.e. bilateral symmetry)")
    if n > 1:
        print(f"instruction-invariance r   : "
              f"{metrics['instruction_invariance_mean_off_diag_r']:.4f}   "
              f"(→1 = text-free, as Family A assumes)")


if __name__ == "__main__":
    main()
