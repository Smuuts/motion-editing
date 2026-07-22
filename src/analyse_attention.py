"""
Phase 0.2 — Cross-attention structure analysis for LEDITS++.

Answers the secondary research question from the proposal:
  "Whether the cross-attention maps of a text-conditioned motion Diffusion
   Transformer exhibit body-part- and temporally-grounded structure analogous
   to the spatial structure that LEDITS++ exploits in image diffusion models."

The result determines which masking strategy is used in Stage 2:
  - If maps are body-part grounded  → use implicit attention-derived M1
  - If maps are diffuse             → use LLM-derived explicit joint-group mask

For each test prompt the script produces three panels:
  (a) Spatiotemporal heatmap (frame × body-part group), content tokens averaged.
  (b) Body-part group profile bar chart.
  (c) Token × group attention matrix (which text tokens drive which groups).

An alignment score is computed: for prompts with a clear expected body-part
(e.g., "right arm" → right_arm group), it checks whether the expected group
achieves the highest attention among non-root groups.

In addition to the conditional readout, a null-subtracted ("differential")
readout is scored side by side: clip(A_cond − A_null, 0, 1), where A_null is the
attention under the learned null_text_emb (context=None). Subtracting the
prompt-independent baseline cancels the common-mode sink and clamping to [0, 1]
keeps only the positive word-driven excess (so the differential stays a valid
attention magnitude); the test is whether that isolates body-part grounding the
conditional grand mean washes out. The null baseline is
prompt-independent, so it is computed once and reused for every prompt. Both the
grand-mean and per-(layer, head) alignment are reported for the differential
readout alongside the conditional one.

Usage:
    python src/analyse_attention.py \\
        --checkpoint runs/exp1/checkpoint_latest \\
        --data_root  data/HumanML3D \\
        --output_dir eval_results/attention_analysis

    # Use more motions for a stable average, or a higher noise level:
    python src/analyse_attention.py --checkpoint ... --num_motions 8 --noise_level 400
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.dit import GroupDiT, GROUP_NAMES
from model.text_encoder import build_text_encoder
from model.schedule import NoiseSchedule
from data.dataset import build_dataloader
from utils.model_io import load_model
from utils.masks import length_to_mask

# G = 7: ["root", "left_leg", "right_leg", "spine", "left_arm", "right_arm", "head"]
_G = len(GROUP_NAMES)

# Test prompts (prompt, expected_peak_group).
# expected_peak_group=None means whole-body — no single group expected.
# These cover each non-root group at least once, and include a whole-body control.
_TEST_PROMPTS = [
    ("a person raises their right arm",             "right_arm"),
    ("a person lifts their left arm up",            "left_arm"),
    ("a person kicks forward with their right leg", "right_leg"),
    ("a person kicks forward with their left leg",  "left_leg"),
    ("a person nods their head",                    "head"),
    ("a person bends their back forward",           "spine"),
    ("a person walks forward",                      None),   # whole-body control
    ("a person sits down",                          None),   # whole-body control
]

# Tokens that dominate HumanML3D attention due to dataset frequency ("a person ...")
# rather than semantic content.  The M1 mask (Phase 2) must exclude these — otherwise
# "a" / "person" absorb most of the attention weight and make the mask uniform.
_STOP_WORDS = {
    "a", "an", "the",
    "person", "man", "woman", "human", "someone",
    "their", "they", "them", "his", "her", "its",
    "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "by", "for", "with",
    "and", "or", "but",
    "up", "down", "forward", "backward", "side", "out",
    "slowly", "quickly", "slightly",
}

def get_semantic_token_info(all_idxs: list[int], all_labels: list[str]):
    """
    Filter content tokens to remove stop words, keeping only semantically meaningful
    tokens (body-part nouns, action verbs, directional adjectives like 'left'/'right').
    Falls back to all content tokens if everything would be filtered.
    This is the token selection strategy that will be used for M1 in Phase 2.
    """
    sem_idxs   = [idx for idx, lbl in zip(all_idxs, all_labels)
                  if lbl.lower() not in _STOP_WORDS]
    sem_labels = [lbl for lbl in all_labels if lbl.lower() not in _STOP_WORDS]
    if not sem_idxs:
        return all_idxs, all_labels   # fallback: nothing left after filtering
    return sem_idxs, sem_labels


def extract_per_layer_head(model, x_t, t_batch, context, mask, F: int, G: int | None):
    """
    Run one forward pass with store_attn=True, collect per-layer maps, and average
    over the batch only — keeping the layer and head axes so the grand mean over
    (layer, head) can be decomposed.

    The default grand-mean view averages over all layers AND heads, which washes
    out grounding that, if present, tends to live in a few mid-stack heads
    (cf. prompt-to-prompt / LEDITS++). Returning the (L, H, ...) tensor lets the
    caller score each (layer, head) separately before collapsing.

    Returns:
      GroupDiT  → (L, H, F, G, L_text)  numpy float32
      MotionDiT → (L, H, F, L_text)     numpy float32
    """
    with torch.no_grad():
        model(x_t, t_batch, context, store_attn=True, mask=mask)

    layer_maps = model.get_attn_maps()  # list of (B, heads, N_tokens, L_text)
    if not layer_maps:
        raise RuntimeError(
            "get_attn_maps() returned empty list — "
            "store_attn=True must be passed to the model."
        )

    stacked = torch.stack(layer_maps, dim=0).float()  # (L, B, H, N, L_text)
    avg_b   = stacked.mean(dim=1)                      # (L, H, N, L_text)
    L, H, _, _ = avg_b.shape

    if G is not None:
        return avg_b.reshape(L, H, F, G, -1).cpu().numpy()  # (L, H, F, G, L_text)
    else:
        return avg_b.reshape(L, H, F, -1).cpu().numpy()     # (L, H, F, L_text)


def extract_avg_per_layer_head(model, schedule, motions, timesteps, context, mask,
                               F: int, G: int | None):
    """
    Per-(layer, head) attention averaged over multiple denoise steps.

    For each t in `timesteps` the motions are re-noised to t (q_sample) and one
    forward pass is run; the maps are averaged over the sweep. This mirrors how the
    editing pipeline builds its mask — by averaging attention over the inversion
    trajectory — whereas extract_per_layer_head uses a single fixed noise level.

    Returns the same shape as extract_per_layer_head:
      GroupDiT  → (L, H, F, G, L_text);  MotionDiT → (L, H, F, L_text)
    """
    B = motions.shape[0]
    acc = None
    for t in timesteps:
        t_batch = torch.full((B,), int(t), device=motions.device, dtype=torch.long)
        x_t, _ = schedule.q_sample(motions, t_batch)
        maps = extract_per_layer_head(model, x_t, t_batch, context, mask, F=F, G=G)
        acc = maps if acc is None else acc + maps
    return acc / len(timesteps)


def _plot_group_analysis(
    attn_single: np.ndarray,        # (F, G, L_text) — single noise level
    attn_avg: np.ndarray,           # (F, G, L_text) — averaged over denoise steps
    content_idxs: list[int],        # all content token positions
    content_labels: list[str],
    prompt: str,
    expected_group: str | None,
    save_path: str,
    subsample: int,
    single_lbl: str = "single step",
    avg_lbl: str = "averaged",
):
    """One figure per prompt comparing the single-step and averaged attention.

    All panels use every content token (BOS/EOS/padding already excluded), not the
    stop-word-filtered semantic subset. The heatmap panels (a)/(b) are min-max
    normalised per body-part group over all frames (independently per panel);
    panels (c)/(d) and the alignment scoring use raw attention.
    """
    G = attn_single.shape[1]

    def _derive(attn):
        spatio = attn[:, :, content_idxs].mean(axis=2)           # (F, G)
        # Per-group min-max over all F frames → each group's temporal profile
        # spans [0, 1]. Heatmap-only: group profile / token×group / alignment
        # scoring stay on raw attention so cross-group magnitudes remain valid.
        lo, hi = spatio.min(axis=0), spatio.max(axis=0)          # (G,), (G,)
        norm = (spatio - lo) / np.where(hi > lo, hi - lo, 1.0)
        return norm[::subsample], spatio.mean(axis=0)            # (F//sub, G), (G,)

    spatio_s, gprof_s = _derive(attn_single)
    spatio_a, gprof_a = _derive(attn_avg)
    # averaged token × group matrix
    tok_group = attn_avg[:, :, content_idxs].mean(axis=0).T      # (num_tok, G)

    frame_ticks = list(range(0, spatio_s.shape[0], max(1, spatio_s.shape[0] // 10)))

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 2, height_ratios=[1.3, 1.3, 1.1], hspace=0.5, wspace=0.32)

    def _heatmap(ax, data, title):
        im = ax.imshow(data.T, aspect="auto", cmap="hot", origin="upper", vmin=0, vmax=1.0)
        ax.set_yticks(range(G)); ax.set_yticklabels(GROUP_NAMES, fontsize=8)
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels([str(i * subsample) for i in frame_ticks], fontsize=7)
        ax.set_xlabel("Frame", fontsize=8); ax.set_ylabel("Group", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                     label="attention (per-group min-max)")

    # (a) single-step heatmap, (b) averaged heatmap — each min-max normalised
    # per group over all frames, independently per panel
    _heatmap(fig.add_subplot(gs[0, :]),
             spatio_s, f'(a) Spatiotemporal attention (per-group norm.) — {single_lbl}  —  "{prompt}"')
    _heatmap(fig.add_subplot(gs[1, :]),
             spatio_a, f'(b) Spatiotemporal attention (per-group norm.) — {avg_lbl}  —  "{prompt}"')

    # (c) group profile: single vs averaged overlaid
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(G); w = 0.38
    ax.barh(x - w/2, gprof_s, w, color="steelblue",  alpha=0.85, label=single_lbl)
    ax.barh(x + w/2, gprof_a, w, color="tab:orange", alpha=0.85, label=avg_lbl)
    for xi, name in enumerate(GROUP_NAMES):
        if name == expected_group:
            ax.axhline(xi, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(range(G)); ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_xlabel("Mean attention", fontsize=8)
    title_c = "(c) Group profile (single vs. averaged)"
    if expected_group:
        title_c += f"\n[expected: {expected_group} — dashed red]"
    ax.set_title(title_c, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # (d) token × group matrix (averaged)
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(tok_group, aspect="auto", cmap="Blues", origin="upper",
                   vmin=0, vmax=tok_group.max() or 1.0)
    ax.set_xticks(range(G))
    ax.set_xticklabels(GROUP_NAMES, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(content_labels)))
    ax.set_yticklabels(content_labels, fontsize=8)
    ax.set_xlabel("Group", fontsize=8); ax.set_ylabel("Token", fontsize=8)
    ax.set_title(f"(d) Token × group matrix ({avg_lbl})", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _peak_group_per_lh(attn_lhfgl: np.ndarray, idxs: list[int]) -> np.ndarray:
    """
    Per (layer, head) peak body-part group (excluding root), using the mean over
    the given text-token columns and over frames.

    attn_lhfgl : (L, H, F, G, L_text)
    Returns    : (L, H) int array of group indices into GROUP_NAMES (>= 1, root excluded).
    """
    gp = attn_lhfgl[:, :, :, 1:, :][..., idxs].mean(axis=(2, 4))  # (L, H, G-1)
    return gp.argmax(axis=-1) + 1                                  # (L, H)


def _peak_group(attn_fgl: np.ndarray, idxs: list[int]) -> str:
    """Peak body-part group (root excluded) from a grand-mean (F, G, L_text) map,
    averaging over frames and the given text-token columns. The differential map is
    clamped to [0, 1] before this is called, so argmax picks the group with the
    largest positive word-driven excess attention."""
    gp = attn_fgl[:, 1:, idxs].mean(axis=(0, 2))   # (G-1,)
    return GROUP_NAMES[int(gp.argmax()) + 1]


def _plot_diff_analysis(
    attn_cond: np.ndarray,          # (F, G, L_text) conditional, grand mean
    attn_null: np.ndarray,          # (F, G, L_text) null baseline, grand mean
    content_idxs: list[int],
    prompt: str,
    expected_group: str | None,
    save_path: str,
    subsample: int,
):
    """Null-subtracted ("differential") cross-attention:  A_cond − A_null.

    Isolates the attention that real words at the content positions produce *over
    and above* the learned null_text_emb baseline at the same positions — a
    common-mode / sink subtraction in the spirit of Differential Transformer
    (arXiv 2410.05258) and DiffEdit's reference-text contrast (arXiv 2210.11427).
    If body-part grounding is present but buried under a prompt-independent sink,
    it should surface here; if the conditional signal is near-invariant to the
    instruction (as the M2/ε probe found, r=0.96), the residual is noise and this
    will not ground either.

    Panels: (a) differential spatiotemporal heatmap (clamped to [0, 1]);
            (b) group profiles — conditional vs null vs differential.
    """
    G = attn_cond.shape[1]
    diff     = np.clip(attn_cond - attn_null, 0.0, 1.0)         # keep positive excess only
    spatio_d = diff[:, :, content_idxs].mean(axis=2)            # (F, G) in [0, 1]
    cond_g   = attn_cond[:, :, content_idxs].mean(axis=(0, 2))  # (G,)
    null_g   = attn_null[:, :, content_idxs].mean(axis=(0, 2))  # (G,)
    diff_g   = spatio_d.mean(axis=0)                            # (G,)

    vmax = float(spatio_d.max()) or 1.0
    sub  = spatio_d[::subsample]
    frame_ticks = list(range(0, sub.shape[0], max(1, sub.shape[0] // 10)))

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.3, 1.0], hspace=0.5)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(sub.T, aspect="auto", cmap="hot", origin="upper",
                   vmin=0.0, vmax=vmax)
    ax.set_yticks(range(G)); ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_xticks(frame_ticks)
    ax.set_xticklabels([str(i * subsample) for i in frame_ticks], fontsize=7)
    ax.set_xlabel("Frame", fontsize=8); ax.set_ylabel("Group", fontsize=8)
    ax.set_title(f'(a) Differential attention  clip(A_cond − A_null, 0, 1)  —  "{prompt}"', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Δ attention (clamped ≥ 0)")

    ax = fig.add_subplot(gs[1])
    x = np.arange(G); w = 0.27
    ax.barh(x - w, cond_g, w, color="steelblue", alpha=0.85, label="conditional")
    ax.barh(x,     null_g, w, color="gray",       alpha=0.70, label="null baseline")
    ax.barh(x + w, diff_g, w, color="tab:green",  alpha=0.85, label="differential")
    for xi, name in enumerate(GROUP_NAMES):
        if name == expected_group:
            ax.axhline(xi, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_yticks(range(G)); ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_xlabel("Mean attention", fontsize=8)
    title = "(b) Group profile: conditional vs null vs differential"
    if expected_group:
        title += f"\n[expected: {expected_group} — dashed red]"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layerhead_alignment(
    acc_lh: np.ndarray,        # (L, H) alignment accuracy across evaluable prompts
    grand_score: float,        # grand-mean (all layers+heads) semantic-token score
    save_path: str,
):
    """Heatmap of per-(layer, head) alignment accuracy, to reveal whether grounding
    is concentrated in specific heads that the grand mean washes out."""
    L, H = acc_lh.shape
    fig, ax = plt.subplots(figsize=(1.1 * H + 3, 0.5 * L + 2))
    im = ax.imshow(acc_lh, aspect="auto", cmap="viridis", origin="upper",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(H)); ax.set_xticklabels([f"h{h}" for h in range(H)], fontsize=8)
    ax.set_yticks(range(L)); ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=8)
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    for l in range(L):
        for h in range(H):
            ax.text(h, l, f"{acc_lh[l, h]:.2f}", ha="center", va="center",
                    color="white" if acc_lh[l, h] < 0.6 else "black", fontsize=7)
    best = np.unravel_index(int(acc_lh.argmax()), acc_lh.shape)
    ax.set_title(
        f"Per-(layer, head) alignment accuracy\n"
        f"grand-mean (all L+H) = {grand_score:.0%}   "
        f"best = L{best[0]}·h{best[1]} ({acc_lh[best]:.0%})",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="alignment accuracy")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_frame_analysis(
    attn_single: np.ndarray,   # (F, L_text) — single noise level
    attn_avg: np.ndarray,      # (F, L_text) — averaged over denoise steps
    content_idxs: list[int],
    content_labels: list[str],
    prompt: str,
    save_path: str,
    subsample: int,
    single_lbl: str = "single step",
    avg_lbl: str = "averaged",
):
    """MotionDiT fallback (no body-part group dimension): single vs averaged in one figure."""
    sub_s = attn_single[::subsample][:, content_idxs]            # (F//sub, num_tok)
    sub_a = attn_avg[::subsample][:, content_idxs]
    hmax = max(sub_s.max(), sub_a.max()) or 1.0                  # shared scale
    xticks = range(0, sub_s.shape[0], max(1, sub_s.shape[0] // 8))
    xlabels = [str(i * subsample) for i in xticks]

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1.0], hspace=0.4, wspace=0.25)

    def _tokframe(ax, data, title):
        im = ax.imshow(data.T, aspect="auto", cmap="hot", origin="upper", vmin=0, vmax=hmax)
        ax.set_xticks(list(xticks)); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_yticks(range(len(content_labels)))
        ax.set_yticklabels(content_labels, fontsize=8)
        ax.set_xlabel("Frame")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax)

    _tokframe(fig.add_subplot(gs[0, 0]), sub_s, f'(a) Token × frame — {single_lbl}')
    _tokframe(fig.add_subplot(gs[0, 1]), sub_a, f'(b) Token × frame — {avg_lbl}')

    # (c) temporal attention profile, single vs averaged overlaid
    ax = fig.add_subplot(gs[1, :])
    ax.plot(sub_s.mean(axis=1), color="steelblue",  label=single_lbl)
    ax.plot(sub_a.mean(axis=1), color="tab:orange", label=avg_lbl)
    ax.set_xlabel("Frame (subsampled)")
    ax.set_ylabel("Mean attention over content tokens")
    ax.set_title(f'(c) Temporal attention profile  —  "{prompt}"', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0.2: Cross-attention structure analysis")
    p.add_argument("--checkpoint",   required=True,
                   help="Checkpoint directory (ema.pt + config.json).")
    p.add_argument("--data_root",    required=True,
                   help="HumanML3D data root.")
    p.add_argument("--output_dir",   default="eval_results/attention_analysis")
    p.add_argument("--num_motions",  type=int, default=5,
                   help="Number of validation motions to average maps over per prompt. "
                        "Higher = more stable but slower.")
    p.add_argument("--noise_level",  type=int, default=200,
                   help="Timestep t at which attention is extracted. "
                        "During editing, maps are averaged over inversion steps; "
                        "here we use a single t as a proxy. "
                        "200–400 shows clearer body-part structure than t≈0 or t≈999.")
    p.add_argument("--subsample",    type=int, default=4,
                   help="Frame subsampling factor for heatmap x-axis readability.")
    p.add_argument("--avg_num_steps", type=int, default=20,
                   help="Number of evenly spaced denoise steps in [1, T) to average "
                        "attention over for the averaged plot. This approximates how the "
                        "editing pipeline averages attention over the inversion trajectory, "
                        "alongside the single-step plot at --noise_level.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    feature_mode = config.get("feature_mode", "humanml3d")
    is_group     = isinstance(model, GroupDiT)
    G            = _G if is_group else None

    print(f"Model:         {type(model).__name__}")
    print(f"Feature mode:  {feature_mode}")
    if not is_group:
        print(
            "NOTE: checkpoint uses MotionDiT — body-part group analysis unavailable.\n"
            "      Temporal-only analysis will run instead.\n"
            "      For full spatiotemporal masking, re-train with feature_mode=group."
        )

    schedule     = NoiseSchedule(timesteps=config.get("timesteps", 1000), device=device)
    text_encoder = build_text_encoder(config, device=device)

    loader = build_dataloader(
        args.data_root, split="val",
        batch_size=args.num_motions,
        max_frames=config.get("max_frames", 196),
        feature_mode=feature_mode,
        shuffle=False,
    )

    # Use the first batch of validation clips as the fixed motion set.
    # Using real (but different-content) motions tests whether the text attention
    # is grounded by the instruction alone, independent of motion content —
    # the same regime as editing an unannotated source clip.
    batch   = next(iter(loader))
    motions = batch["motion"][:args.num_motions].to(device)
    lengths = torch.as_tensor(
        batch["length"][:args.num_motions] if not isinstance(batch["length"], torch.Tensor)
        else batch["length"][:args.num_motions],
        device=device,
    )
    B, F, _ = motions.shape
    attn_mask = length_to_mask(lengths, F)

    t_batch = torch.full((B,), args.noise_level, device=device, dtype=torch.long)
    x_t, _ = schedule.q_sample(motions, t_batch)

    # Timesteps for the averaged-attention plot: evenly spaced over the diffusion range.
    avg_timesteps = np.unique(
        np.linspace(1, schedule.T - 1, args.avg_num_steps).round().astype(int)
    )

    print(f"\nMotions:       {B} validation clips, {F} frames each")
    print(f"Noise level:   t = {args.noise_level} (single-step plot)")
    print(f"Averaged plot: {len(avg_timesteps)} steps t ∈ [{avg_timesteps[0]}, {avg_timesteps[-1]}]")
    print(f"Prompts:       {len(_TEST_PROMPTS)}\n")

    # ── Null-text baseline for the differential readout ─────────────────────────
    # null_text_emb (context=None) is prompt-independent, so with the same motions
    # and timesteps the null attention is identical for every prompt — compute it
    # once and reuse. Subtracting it isolates what real words add over the learned
    # placeholder at the same content positions (common-mode / sink cancellation).
    attn_null_np: np.ndarray | None = None
    attn_lh_null: np.ndarray | None = None
    if is_group:
        print("Computing null-text baseline (context=None) for differential readout ...\n")
        attn_lh_null = extract_per_layer_head(
            model, x_t, t_batch, None, attn_mask, F=F, G=G
        )                                              # (L, H, F, G, L_text)
        attn_null_np = attn_lh_null.mean(axis=(0, 1))  # (F, G, L_text)

    alignment_results = []
    n_correct = 0
    n_correct_diff = 0
    n_evaluable = 0

    # Per-(layer, head) accumulators: how often each head's semantic-token peak group
    # matches the expected group, across the evaluable prompts — for the conditional
    # and the null-subtracted (differential) readout respectively.
    lh_correct: np.ndarray | None = None        # (L, H) initialised lazily
    lh_correct_diff: np.ndarray | None = None   # (L, H) initialised lazily

    for i, (prompt, expected_group) in enumerate(_TEST_PROMPTS):
        print(f"[{i+1}/{len(_TEST_PROMPTS)}] \"{prompt}\"")

        context = text_encoder.encode([prompt] * B)     # (B, L_text, dim)
        # token_info() comes from the encoder itself, so the selected column
        # positions match this encoder's L_text dimension (CLIP=77 or T5=max_length).
        content_idxs, content_labels = text_encoder.token_info(prompt)
        sem_idxs, sem_labels = get_semantic_token_info(content_idxs, content_labels)
        print(f"  All tokens:      {content_labels}")
        print(f"  Semantic tokens: {sem_labels}")

        try:
            attn_lh = extract_per_layer_head(
                model, x_t, t_batch, context, attn_mask, F=F, G=G
            )
            attn_lh_avg = extract_avg_per_layer_head(
                model, schedule, motions, avg_timesteps, context, attn_mask, F=F, G=G
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        # Grand mean over (layer, head). Two views:
        #   single — attention at the fixed noise level args.noise_level
        #   avg    — averaged over avg_timesteps (matches the editing pipeline)
        # The alignment scoring below uses the single-step view, unchanged.
        attn_np     = attn_lh.mean(axis=(0, 1))       # (F, G, L_text) or (F, L_text)
        attn_np_avg = attn_lh_avg.mean(axis=(0, 1))

        slug       = prompt.replace(" ", "_")[:45]
        save_path  = os.path.join(args.output_dir, f"{i:02d}_{slug}.png")
        single_lbl = f"single step t={args.noise_level}"
        avg_lbl    = f"avg over {len(avg_timesteps)} steps"

        peak_diff_sem = aligned_diff_sem = None
        if is_group:
            # One figure per prompt: single-step and averaged views combined.
            _plot_group_analysis(attn_np, attn_np_avg, content_idxs, content_labels,
                                 prompt, expected_group, save_path, args.subsample,
                                 single_lbl, avg_lbl)

            peak_all = _peak_group(attn_np, content_idxs)
            peak_sem = _peak_group(attn_np, sem_idxs)

            # Null-subtracted (differential) readout on the same single-step view.
            # Requires matching text-column counts between the conditional context
            # and null_text_emb (they should — see dit.GroupDiT.null_text_emb).
            if attn_null_np.shape[-1] != attn_np.shape[-1]:
                raise RuntimeError(
                    f"null baseline L_text={attn_null_np.shape[-1]} != conditional "
                    f"L_text={attn_np.shape[-1]}; cannot form the differential map. "
                    "null_text_emb length must match the text encoder output length."
                )
            attn_np_diff = np.clip(attn_np - attn_null_np, 0.0, 1.0)  # (F, G, L_text) in [0, 1]
            peak_diff_all = _peak_group(attn_np_diff, content_idxs)
            peak_diff_sem = _peak_group(attn_np_diff, sem_idxs)

            diff_path = os.path.join(args.output_dir, f"{i:02d}_{slug}_diff.png")
            _plot_diff_analysis(attn_np, attn_null_np, content_idxs, prompt,
                                expected_group, diff_path, args.subsample)

            if expected_group is not None:
                aligned_all      = (peak_all == expected_group)
                aligned_sem      = (peak_sem == expected_group)
                aligned_diff_sem = (peak_diff_sem == expected_group)
                n_evaluable += 1
                n_correct      += int(aligned_sem)        # conditional, semantic tokens
                n_correct_diff += int(aligned_diff_sem)   # differential, semantic tokens
                m_all  = "✓" if aligned_all      else "✗"
                m_sem  = "✓" if aligned_sem      else "✗"
                m_diff = "✓" if aligned_diff_sem else "✗"

                # Per-(layer, head): does each head's semantic-token peak match?
                expected_idx = GROUP_NAMES.index(expected_group)
                peak_lh      = _peak_group_per_lh(attn_lh, sem_idxs)                 # (L, H)
                peak_lh_diff = _peak_group_per_lh(
                    np.clip(attn_lh - attn_lh_null, 0.0, 1.0), sem_idxs)            # (L, H)
                if lh_correct is None:
                    lh_correct      = np.zeros(peak_lh.shape, dtype=np.int64)
                    lh_correct_diff = np.zeros(peak_lh.shape, dtype=np.int64)
                lh_correct      += (peak_lh      == expected_idx).astype(np.int64)
                lh_correct_diff += (peak_lh_diff == expected_idx).astype(np.int64)
            else:
                aligned_all = aligned_sem = None
                m_all = m_sem = m_diff = "—"

            aligned = aligned_sem
            print(f"  Peak (all tokens):  {peak_all:<12}  {m_all}")
            print(f"  Peak (semantic):    {peak_sem:<12}  expected: {expected_group or 'N/A':<12}  {m_sem}")
            print(f"  Peak (differential):{peak_diff_sem:<12}  null-subtracted           {m_diff}")

        else:
            _plot_frame_analysis(attn_np, attn_np_avg, content_idxs, content_labels,
                                 prompt, save_path, args.subsample, single_lbl, avg_lbl)
            peak_group = None
            aligned    = None
            print("  (MotionDiT: temporal analysis only — no group alignment score)")

        print(f"  Saved → {save_path}")
        alignment_results.append({
            "prompt":            prompt,
            "expected_group":    expected_group,
            "peak_group_all":    peak_all      if is_group else None,
            "peak_group_sem":    peak_sem      if is_group else None,
            "aligned_all_toks":  aligned_all   if is_group else None,
            "aligned_sem_toks":  aligned_sem   if is_group else None,
            "peak_group_diff":   peak_diff_sem if is_group else None,
            "aligned_diff_toks": aligned_diff_sem if is_group else None,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    score_sem = n_correct / n_evaluable if n_evaluable > 0 else 0.0
    n_correct_all = sum(
        1 for r in alignment_results
        if r.get("aligned_all_toks") is True
    )
    score_all  = n_correct_all  / n_evaluable if n_evaluable > 0 else 0.0
    score_diff = n_correct_diff / n_evaluable if n_evaluable > 0 else 0.0

    # ── Per-(layer, head) breakdown ─────────────────────────────────────────────
    # The grand mean averages over all layers and heads; if grounding exists in only
    # a few heads it is invisible there. This finds the best single (layer, head)
    # and reports its accuracy, so the "diffuse" verdict isn't an averaging artifact.
    best_layerhead = None
    if is_group and lh_correct is not None and n_evaluable > 0:
        acc_lh = lh_correct / n_evaluable          # (L, H)
        L_n, H_n = acc_lh.shape
        bl, bh = np.unravel_index(int(acc_lh.argmax()), acc_lh.shape)
        lh_path = os.path.join(args.output_dir, "per_layer_head_alignment.png")
        _plot_layerhead_alignment(acc_lh, score_sem, lh_path)
        best_layerhead = {
            "best_layer":           int(bl),
            "best_head":            int(bh),
            "best_layerhead_score": float(acc_lh[bl, bh]),
            "best_layer_mean":      float(acc_lh.mean(axis=1).max()),  # best layer, heads avgd
            "best_head_mean":       float(acc_lh.mean(axis=0).max()),  # best head, layers avgd
            "accuracy_matrix":      acc_lh.round(4).tolist(),
            "plot":                 lh_path,
        }

    # Same per-(layer, head) breakdown for the null-subtracted (differential) readout.
    best_layerhead_diff = None
    if is_group and lh_correct_diff is not None and n_evaluable > 0:
        acc_lh_d = lh_correct_diff / n_evaluable   # (L, H)
        bl, bh   = np.unravel_index(int(acc_lh_d.argmax()), acc_lh_d.shape)
        lh_path_d = os.path.join(args.output_dir, "per_layer_head_alignment_diff.png")
        _plot_layerhead_alignment(acc_lh_d, score_diff, lh_path_d)
        best_layerhead_diff = {
            "best_layer":           int(bl),
            "best_head":            int(bh),
            "best_layerhead_score": float(acc_lh_d[bl, bh]),
            "best_layer_mean":      float(acc_lh_d.mean(axis=1).max()),
            "best_head_mean":       float(acc_lh_d.mean(axis=0).max()),
            "accuracy_matrix":      acc_lh_d.round(4).tolist(),
            "plot":                 lh_path_d,
        }

    summary = {
        "model_type":              type(model).__name__,
        "feature_mode":            feature_mode,
        "noise_level_t":           args.noise_level,
        "num_motions":             args.num_motions,
        "alignment_score_all_tok":  score_all,
        "alignment_score_sem_tok":  score_sem,
        "alignment_score_diff_tok": score_diff,
        "n_evaluable":              n_evaluable,
        "n_correct_all_tok":        n_correct_all,
        "n_correct_sem_tok":        n_correct,
        "n_correct_diff_tok":       n_correct_diff,
        "per_layer_head":           best_layerhead,
        "per_layer_head_diff":      best_layerhead_diff,
        "results":                  alignment_results,
    }
    summary_path = os.path.join(args.output_dir, "alignment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Decision uses semantic-token score; all-token score is shown for comparison.
    if is_group and n_evaluable > 0:
        if score_sem >= 0.75:
            verdict = "IMPLICIT M1 VIABLE   — semantic-token maps are body-part grounded"
        elif score_sem >= 0.50:
            verdict = "BORDERLINE           — test on more prompts / a stronger checkpoint"
        else:
            verdict = "USE LLM FALLBACK     — maps not sufficiently grounded even with semantic tokens"
    else:
        verdict = "N/A (MotionDiT: retrain with feature_mode=group for group analysis)"

    print(f"\n{'='*60}")
    print("Attention structure analysis summary")
    print(f"{'='*60}")
    print(f"  Model:                    {type(model).__name__}")
    print(f"  Noise level:              t = {args.noise_level}")
    if is_group and n_evaluable > 0:
        print(f"  Alignment (all tokens):   {score_all:.0%}  ({n_correct_all}/{n_evaluable})")
        print(f"  Alignment (semantic tok): {score_sem:.0%}  ({n_correct}/{n_evaluable})  ← decision basis")
        print(f"  Alignment (differential): {score_diff:.0%}  ({n_correct_diff}/{n_evaluable})  "
              f"← null-subtracted (A_cond − A_null)")
        if best_layerhead is not None:
            bl, bh = best_layerhead["best_layer"], best_layerhead["best_head"]
            print(f"  Best single (layer,head): L{bl}·h{bh} = "
                  f"{best_layerhead['best_layerhead_score']:.0%}  "
                  f"(vs {score_sem:.0%} grand-mean — gap = grounding hidden by averaging)")
        if best_layerhead_diff is not None:
            bl, bh = best_layerhead_diff["best_layer"], best_layerhead_diff["best_head"]
            print(f"  Best (layer,head) diff:   L{bl}·h{bh} = "
                  f"{best_layerhead_diff['best_layerhead_score']:.0%}  (differential readout)")
        print(f"  Stop words removed:       {sorted(_STOP_WORDS)}")
    print(f"  Decision:                 {verdict}")
    print(f"  Output:                   {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
