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

from model.dit import build_model, GroupDiT, GROUP_NAMES
from model.text_encoder import build_text_encoder, get_encoder_dims
from model.schedule import NoiseSchedule
from data.dataset import build_dataloader

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

def _load_model(ckpt_dir: str, device):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    context_dim, text_seq_len = get_encoder_dims(config)
    model = build_model({
        "feature_mode": config.get("feature_mode", "humanml3d"),
        "input_dim":    config.get("input_dim", 263),
        "latent_dim":   config.get("latent_dim", 512),
        "context_dim":  context_dim,
        "text_seq_len": text_seq_len,
        "num_heads":    config.get("num_heads", 8),
        "num_layers":   config.get("num_layers", 8),
        "max_frames":   config.get("max_frames", 196),
        "dropout":      0.0,
    }, device=device)

    ema_path = os.path.join(ckpt_dir, "ema.pt")
    model.load_state_dict(
        torch.load(ema_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, config


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


def extract_and_aggregate(model, x_t, t_batch, context, mask, F: int, G: int | None):
    """
    Run one forward pass with store_attn=True, collect per-layer maps,
    and return a single averaged array.

    Returns:
      GroupDiT  → (F, G, L_text)  numpy float32
      MotionDiT → (F, L_text)     numpy float32
    """
    with torch.no_grad():
        model(x_t, t_batch, context, store_attn=True, mask=mask)

    layer_maps = model.get_attn_maps()  # list of (B, heads, N_tokens, L_text)
    if not layer_maps:
        raise RuntimeError(
            "get_attn_maps() returned empty list — "
            "store_attn=True must be passed to the model."
        )

    # Stack and average over layers, batch, heads → (N_tokens, L_text)
    stacked = torch.stack(layer_maps, dim=0).float()  # (L, B, H, N, L_text)
    avg = stacked.mean(dim=(0, 1, 2))                 # (N_tokens, L_text)

    if G is not None:
        return avg.reshape(F, G, -1).cpu().numpy()   # (F, G, L_text)
    else:
        return avg.reshape(F, -1).cpu().numpy()      # (F, L_text)


def _plot_group_analysis(
    attn_fgl: np.ndarray,           # (F, G, L_text)
    content_idxs: list[int],        # all content token positions
    content_labels: list[str],
    sem_idxs: list[int],            # semantic-only token positions (stop words removed)
    sem_labels: list[str],
    prompt: str,
    expected_group: str | None,
    save_path: str,
    subsample: int,
):
    F, G, _ = attn_fgl.shape

    def _derive(idxs):
        spatio     = attn_fgl[:, :, idxs].mean(axis=2)   # (F, G)
        spatio_sub = spatio[::subsample]
        group_prof = spatio.mean(axis=0)                  # (G,)
        tok_group  = attn_fgl[:, :, idxs].mean(axis=0).T # (num_tok, G)
        return spatio_sub, group_prof, tok_group

    spatio_all, gprof_all, tg_all = _derive(content_idxs)
    spatio_sem, gprof_sem, tg_sem = _derive(sem_idxs)

    bar_colors = [
        "tab:gray" if n == "root" else
        ("tab:red" if n == expected_group else "steelblue")
        for n in GROUP_NAMES
    ]
    frame_ticks = list(range(0, spatio_all.shape[0], max(1, spatio_all.shape[0] // 10)))

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(3, 2, height_ratios=[1.3, 1.3, 1.0], hspace=0.55, wspace=0.32)

    def _heatmap(ax, data, title):
        im = ax.imshow(data.T, aspect="auto", cmap="hot", origin="upper",
                       vmin=0, vmax=data.max() or 1.0)
        ax.set_yticks(range(G))
        ax.set_yticklabels(GROUP_NAMES, fontsize=8)
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels([str(i * subsample) for i in frame_ticks], fontsize=7)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Group", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    def _bar(ax, gprof, title):
        ax.barh(range(G), gprof, color=bar_colors)
        ax.set_yticks(range(G))
        ax.set_yticklabels(GROUP_NAMES, fontsize=8)
        ax.set_xlabel("Mean attention", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    def _tokmat(ax, tg, labels, title):
        im = ax.imshow(tg, aspect="auto", cmap="Blues", origin="upper",
                       vmin=0, vmax=tg.max() or 1.0)
        ax.set_xticks(range(G))
        ax.set_xticklabels(GROUP_NAMES, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Group", fontsize=8)
        ax.set_ylabel("Token", fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)

    # Row 0: all-token spatiotemporal heatmap
    _heatmap(fig.add_subplot(gs[0, :]),
             spatio_all, f'(a) All content tokens  —  "{prompt}"')

    # Row 1: semantic-only spatiotemporal heatmap
    sem_title = f'(b) Semantic tokens only  {sem_labels}  —  "{prompt}"'
    _heatmap(fig.add_subplot(gs[1, :]), spatio_sem, sem_title)

    # Row 2 left: group profiles overlaid
    ax_bar = fig.add_subplot(gs[2, 0])
    x = np.arange(G)
    w = 0.38
    ax_bar.barh(x - w/2, gprof_all, w, color="steelblue",  alpha=0.8, label="all tokens")
    ax_bar.barh(x + w/2, gprof_sem, w, color="tab:orange", alpha=0.8, label="semantic only")
    for xi, name in enumerate(GROUP_NAMES):
        if name == expected_group:
            ax_bar.axhline(xi, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax_bar.set_yticks(range(G))
    ax_bar.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax_bar.set_xlabel("Mean attention", fontsize=8)
    title_bar = "(c) Group profiles (all vs. semantic)"
    if expected_group:
        title_bar += f"\n[expected: {expected_group} — dashed red]"
    ax_bar.set_title(title_bar, fontsize=9)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Row 2 right: semantic token × group matrix
    _tokmat(fig.add_subplot(gs[2, 1]), tg_sem, sem_labels,
            "(d) Semantic-token × group matrix")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_frame_analysis(
    attn_fl: np.ndarray,       # (F, L_text)
    content_idxs: list[int],
    content_labels: list[str],
    prompt: str,
    save_path: str,
    subsample: int,
):
    """Fallback visualisation for MotionDiT (no body-part group dimension)."""
    spatio_sub = attn_fl[::subsample][:, content_idxs]        # (F//sub, num_tok)
    temporal   = spatio_sub.mean(axis=1)                       # (F//sub,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(spatio_sub.T, aspect="auto", cmap="hot", origin="upper")
    ax.set_xticks(range(0, spatio_sub.shape[0], max(1, spatio_sub.shape[0] // 8)))
    ax.set_xticklabels(
        [str(i * subsample) for i in range(0, spatio_sub.shape[0], max(1, spatio_sub.shape[0] // 8))],
        fontsize=8,
    )
    ax.set_yticks(range(len(content_labels)))
    ax.set_yticklabels(content_labels, fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_title(f'Token × frame attention\n"{prompt}"', fontsize=9)
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    ax.plot(temporal)
    ax.set_xlabel("Frame (subsampled)")
    ax.set_ylabel("Mean attention over content tokens")
    ax.set_title("Temporal attention profile", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
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
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = _load_model(args.checkpoint, device)
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
    attn_mask = torch.arange(F, device=device)[None, :] < lengths[:, None]

    t_batch = torch.full((B,), args.noise_level, device=device, dtype=torch.long)
    x_t, _ = schedule.q_sample(motions, t_batch)

    print(f"\nMotions:       {B} validation clips, {F} frames each")
    print(f"Noise level:   t = {args.noise_level}")
    print(f"Prompts:       {len(_TEST_PROMPTS)}\n")

    alignment_results = []
    n_correct = 0
    n_evaluable = 0

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
            attn_np = extract_and_aggregate(
                model, x_t, t_batch, context, attn_mask, F=F, G=G
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        # Save visualisation
        slug      = prompt.replace(" ", "_")[:45]
        save_path = os.path.join(args.output_dir, f"{i:02d}_{slug}.png")

        if is_group:
            _plot_group_analysis(
                attn_np, content_idxs, content_labels,
                sem_idxs, sem_labels,
                prompt, expected_group, save_path, args.subsample,
            )

            def _peak(idxs):
                gp = attn_np[:, 1:, idxs].mean(axis=(0, 2))  # (G-1,)
                return GROUP_NAMES[int(gp.argmax()) + 1]

            peak_all = _peak(content_idxs)
            peak_sem = _peak(sem_idxs)

            if expected_group is not None:
                aligned_all = (peak_all == expected_group)
                aligned_sem = (peak_sem == expected_group)
                n_evaluable += 1
                n_correct   += int(aligned_sem)   # score on semantic tokens
                m_all = "✓" if aligned_all else "✗"
                m_sem = "✓" if aligned_sem else "✗"
            else:
                aligned_all = aligned_sem = None
                m_all = m_sem = "—"

            aligned = aligned_sem
            print(f"  Peak (all tokens):  {peak_all:<12}  {m_all}")
            print(f"  Peak (semantic):    {peak_sem:<12}  expected: {expected_group or 'N/A':<12}  {m_sem}")

        else:
            _plot_frame_analysis(
                attn_np, content_idxs, content_labels,
                prompt, save_path, args.subsample,
            )
            peak_group = None
            aligned    = None
            print("  (MotionDiT: temporal analysis only — no group alignment score)")

        print(f"  Saved → {save_path}")
        alignment_results.append({
            "prompt":            prompt,
            "expected_group":    expected_group,
            "peak_group_all":    peak_all    if is_group else None,
            "peak_group_sem":    peak_sem    if is_group else None,
            "aligned_all_toks":  aligned_all if is_group else None,
            "aligned_sem_toks":  aligned_sem if is_group else None,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    score_sem = n_correct / n_evaluable if n_evaluable > 0 else 0.0
    n_correct_all = sum(
        1 for r in alignment_results
        if r.get("aligned_all_toks") is True
    )
    score_all = n_correct_all / n_evaluable if n_evaluable > 0 else 0.0

    summary = {
        "model_type":              type(model).__name__,
        "feature_mode":            feature_mode,
        "noise_level_t":           args.noise_level,
        "num_motions":             args.num_motions,
        "alignment_score_all_tok": score_all,
        "alignment_score_sem_tok": score_sem,
        "n_evaluable":             n_evaluable,
        "n_correct_all_tok":       n_correct_all,
        "n_correct_sem_tok":       n_correct,
        "results":                 alignment_results,
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
        print(f"  Stop words removed:       {sorted(_STOP_WORDS)}")
    print(f"  Decision:                 {verdict}")
    print(f"  Output:                   {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
