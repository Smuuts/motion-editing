"""
Visualise "the mask problem" — why the implicit LEDITS++ masks fail.

The core negative result of this project (see docs/FINDINGS.md, docs/PROGRESS.md
"Open problems") is that the two implicit masks are *source-dynamics-driven, not
instruction-driven*:

  M1 (cross-attention)   — meant to answer "which body-part group does the edit
                           TEXT attend to?", but the readout is nearly invariant
                           to the instruction (sink-dominated, no laterality).
  M2 (noise-estimate ψ)  — meant to answer "where does the edit change the
                           prediction?", but it fires on whatever the SOURCE clip
                           already moves, essentially independent of the text
                           (r≈0.96 between opposite-laterality instructions).

Neither can target a body part the source holds still, and neither distinguishes
left from right. This script makes that visible: it runs ONE source clip through a
SET of deliberately contrasting instructions (e.g. left- vs right-arm, arm vs leg)
and shows that the resulting masks look the same as each other and track the
source's own motion — not the word in the instruction.

It is training-mode agnostic: it reuses the real editing stack (MotionEditor +
masking.collect_statistics/build_mask), so it works for any checkpoint the editor
works for — humanml3d (263-d) or smplh (135-d) features, GroupDiT or GroupCLR
(U-Net) backbones, either token axis (group_mode: 7 body-part groups or 22 per-joint
tokens — the group axis is read from the checkpoint, so the heatmaps scale to
whichever it is), and the legacy flat MotionDiT (G=1, in which case only the
temporal story is shown — the body-part axis does not exist).

Two figures are written per source clip:

  <clip>_mask_problem.png   The main panel. A per-instruction grid of the raw M1,
                            raw M2 and final binary mask (frame × group), all on a
                            shared colour scale, above the instruction-independent
                            source-motion reference. If the maps were
                            instruction-driven the rows would differ and each would
                            light up its own target group (outlined in red); instead
                            they look alike and follow the reference.
  <clip>_mask_problem_quant.png
                            The quantification. Instruction×instruction correlation
                            matrices for M1 and M2 (≈1 off-diagonal ⇒ the mask barely
                            changes with the instruction) and, per instruction, the
                            correlation of each mask with the source's own motion
                            (high ⇒ the mask is really just a source-dynamics
                            detector).

Usage
-----
    python src/visualise_mask_problem.py \
        --checkpoint runs/exp_smplh_unet/checkpoint_latest \
        --data_root  data/HumanML3D_smplh \
        --source 0 \
        --out_dir eval_results/mask_problem

    # Custom contrasting instructions + explicit expected groups for the overlay:
    python src/visualise_mask_problem.py --checkpoint ... --data_root ... --source 0 \
        --instruction "raise the left arm"  --target_groups "left_arm" \
        --instruction "raise the right arm" --target_groups "right_arm" \
        --mask_timesteps 40
"""

import os
import sys
import argparse

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from model.body_groups import (
    GROUP_NAMES, GROUP_CHANNELS, group_names, named_token_indices, resolve_group_context,
)
from editing import MotionEditor
from editing import masking
from editing.masking import semantic_token_subset
from utils.model_io import load_model
from sample_model import recover_joints, _smplh_body_model  # noqa: F401 (parity w/ edit_motion)


# Default contrasting set: laterality (left vs right) and limb (arm vs leg) pairs —
# the two axes the implicit masks are known to miss. Overridable via --instruction.
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
                   help="Edit instruction. Repeat for the contrasting set (default: "
                        f"{DEFAULT_INSTRUCTIONS}).")
    p.add_argument("--target_groups", action="append", default=None,
                   help="Expected body-part group(s) for the red overlay, one per "
                        "--instruction (space/comma separated names from "
                        f"{GROUP_NAMES}). If omitted, they are guessed from the "
                        "instruction text (best-effort keyword match).")
    p.add_argument("--mask_mode", default="m2_only",
                   choices=["none", "m2_only", "m1_only", "attn", "temporal"],
                   help="Which binary mask to show in the 'final mask' column "
                        "(default m2_only, the editor default). The raw M1/M2 maps are "
                        "always shown regardless.")
    p.add_argument("--lambda_attn", type=float, default=70.0,
                   help="M1 percentile threshold for the binary mask.")
    p.add_argument("--lambda_noise", type=float, default=70.0,
                   help="M2 percentile threshold for the binary mask.")
    p.add_argument("--m1_readout", default="raw",
                   choices=["raw", "renorm", "spatial", "renorm_spatial"],
                   help="M1 per-cell attention readout (see masking.collect_statistics).")
    p.add_argument("--mask_timesteps", type=int, default=40,
                   help="Evenly-spaced timesteps swept for mask collection "
                        "(default 40; None-equivalent full 1000 is much slower).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--smplh_model_path", default="data/motionfix/data/body_models/smplh",
                   help="smplh checkpoints only: SMPLHLayer dir for decoding (kept for "
                        "parity; this script does not render skeletons).")
    p.add_argument("--out_dir", default="eval_results/mask_problem")
    p.add_argument("--no_ema", action="store_true", help="Load model.pt instead of ema.pt.")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ── expected-group inference (for the red overlay only) ──────────────────────────
# Keyword → group-name votes. Laterality words gate left/right; limb words pick the
# limb. Best-effort: this only annotates what the mask SHOULD hit, it never feeds
# the mask itself.
_LIMB_KEYWORDS = {
    "left_arm":  ["arm", "hand", "wrist", "elbow", "shoulder", "reach", "punch", "wave"],
    "right_arm": ["arm", "hand", "wrist", "elbow", "shoulder", "reach", "punch", "wave"],
    "left_leg":  ["leg", "knee", "foot", "feet", "ankle", "kick", "step", "kneel", "stomp"],
    "right_leg": ["leg", "knee", "foot", "feet", "ankle", "kick", "step", "kneel", "stomp"],
    "spine":     ["torso", "spine", "back", "bend", "lean", "twist", "hip", "waist", "chest"],
    "head":      ["head", "neck", "look", "nod", "gaze"],
}


def guess_target_groups(instruction: str) -> list[str]:
    """Best-effort {group name} the instruction is *supposed* to move, from keywords."""
    txt = instruction.lower()
    left, right = "left" in txt, "right" in txt
    hits = []
    for group, kws in _LIMB_KEYWORDS.items():
        if not any(k in txt for k in kws):
            continue
        if group.startswith("left_") and right and not left:
            continue
        if group.startswith("right_") and left and not right:
            continue
        hits.append(group)
    return hits


def _expand_to_axis(names: list[str], group_mode: str) -> list[str]:
    """Resolve part/joint names to the model's token axis (names valid for the red
    overlay). In 'joints' mode a coarse part name expands to its joint tokens."""
    axis = group_names(group_mode)
    try:
        return [axis[i] for i in named_token_indices(names, group_mode)]
    except ValueError as e:
        raise SystemExit(str(e))


def parse_target_groups(spec: str, group_mode: str) -> list[str]:
    names = [n.strip() for n in spec.replace(",", " ").split() if n.strip()]
    return _expand_to_axis(names, group_mode)


def resolve_targets(instructions, target_groups, group_mode="parts"):
    """One list of expected token-axis-name lists, one per instruction (already
    expanded to the model's group_mode axis: part names in 'parts', joint names in
    'joints')."""
    if target_groups is None:
        return [_expand_to_axis(guess_target_groups(e), group_mode) for e in instructions]
    if len(target_groups) == 1:
        g = parse_target_groups(target_groups[0], group_mode)
        return [g for _ in instructions]
    if len(target_groups) == len(instructions):
        return [parse_target_groups(s, group_mode) for s in target_groups]
    raise SystemExit(f"--target_groups must be 1 or {len(instructions)} values "
                     f"(got {len(target_groups)}).")


# ── source clip loading (mirrors edit_motion.load_source) ────────────────────────
def load_source(source, data_root, split, max_frames):
    if source.endswith(".npy") and os.path.exists(source):
        raw = np.load(source)
        clip_id = os.path.splitext(os.path.basename(source))[0]
    else:
        with open(os.path.join(data_root, f"{split}.txt")) as f:
            ids = [l.strip() for l in f if l.strip()]
        clip_id = ids[int(source)]
        raw = np.load(os.path.join(data_root, "new_joint_vecs", f"{clip_id}.npy"))
    # original prompt: first caption line, stripped of HumanML3D's "#pos#tags" suffix
    caption = ""
    text_path = os.path.join(data_root, "texts", f"{clip_id}.txt")
    if os.path.exists(text_path):
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            caption = lines[0].split("#")[0].strip()
    T = min(len(raw), max_frames)
    return raw[:T].astype(np.float32), clip_id, T, caption


def source_activity(x0, group_channels, is_group):
    """(F, G) per-(frame, group) source motion energy |Δx0|"""
    d = x0[0]                                            # (F, D)
    diff = (d[1:] - d[:-1]).abs()                        # (F-1, D)
    if not is_group:
        act = diff.mean(dim=-1, keepdim=True)            # (F-1, 1)
    else:
        gc = group_channels or GROUP_CHANNELS
        cols = [diff[:, ch].mean(dim=-1) for ch in gc]
        act = torch.stack(cols, dim=-1)                  # (F-1, G)
    act = torch.cat([act[:1] * 0, act], dim=0)           # pad first frame → (F, G)
    return act.cpu().numpy()


def flat_corr(a, b):
    """Pearson r between two arrays' flattened values; 0 if either is constant."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ── figure 1: the main per-instruction grid ──────────────────────────────────────
def _heatmap(ax, fg, glabels, cmap, vmin, vmax, expected_idx):
    """fg is (F, G); drawn as (G, F). expected_idx rows get a red outline."""
    F, G = fg.shape
    ax.imshow(fg.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_yticks(range(G))
    ax.set_yticklabels(glabels, fontsize=6)
    ax.set_xticks([])
    for g in expected_idx:
        ax.add_patch(Rectangle((-0.5, g - 0.5), F, 1, fill=False,
                                edgecolor="red", lw=1.6))
        ax.get_yticklabels()[g].set_color("red")
        ax.get_yticklabels()[g].set_fontweight("bold")


def _short_caption(caption, width=80):
    """Source prompt trimmed to fit a figure headline ('' if the clip has none)."""
    c = (caption or "").strip()
    return c if len(c) <= width else c[:width - 1] + "…"


def plot_main(clip_id, caption, instructions, targets, m1_maps, m2_maps, bin_maps,
              src_act, glabels, mask_mode, out_path):
    n = len(instructions)
    tgt_idx = [[glabels.index(g) for g in t if g in glabels] for t in targets]

    # Shared colour scales across instruction rows are the whole point: identical
    # colours = identical maps. 99th percentile ceiling keeps one hot cell from
    # flattening everything else.
    def vmax_of(maps):
        allv = np.concatenate([m.ravel() for m in maps])
        return float(np.quantile(allv, 0.99)) or 1.0
    m1_vmax, m2_vmax = vmax_of(m1_maps), vmax_of(m2_maps)
    act_vmax = float(np.quantile(src_act.ravel(), 0.99)) or 1.0

    H = 2.0 + 1.15 * n
    fig = plt.figure(figsize=(11, H))
    # Reserve ~0.95in at the top for the 3-line suptitle (source prompt included) so
    # it never collides with the reference-row title below it.
    gs = GridSpec(n + 1, 3, figure=fig, hspace=0.5, wspace=0.18,
                  height_ratios=[0.9] + [1.0] * n, top=1 - 0.95 / H, bottom=0.05)

    # top reference row: instruction-independent source dynamics, full width
    ax_ref = fig.add_subplot(gs[0, :])
    _heatmap(ax_ref, src_act, glabels, "cividis", 0.0, act_vmax, [])
    ax_ref.set_title(
        "SOURCE motion  |Δx0|  (instruction-INDEPENDENT reference — what the "
        "implicit masks actually track)",
        fontsize=8.5, loc="left")

    col_titles = ["M1  raw cross-attention", "M2  raw noise ψ",
                  f"final binary mask  ({mask_mode})"]
    col_cmaps = ["magma", "magma", "gray"]
    col_vmax = [m1_vmax, m2_vmax, 1.0]

    for i in range(n):
        for c in range(3):
            ax = fig.add_subplot(gs[i + 1, c])
            maps = (m1_maps, m2_maps, bin_maps)[c]
            _heatmap(ax, maps[i], glabels, col_cmaps[c], 0.0, col_vmax[c], tgt_idx[i])
            if i == 0:
                ax.set_title(col_titles[c], fontsize=8.5)
            if c == 0:
                exp = ", ".join(targets[i]) or "—"
                ax.set_ylabel(f"{instructions[i]}\n(expect: {exp})",
                              fontsize=7.5, rotation=0, ha="right", va="center",
                              labelpad=38)
            if i == n - 1:
                ax.set_xlabel("frame", fontsize=7)

    head = ("The mask problem — implicit M1/M2 masks are source-dynamics-driven, "
            "not instruction-driven")
    sub = ("rows barely differ and follow the source reference; "
           "red = the group each instruction SHOULD move")
    cap = _short_caption(caption)
    src_line = f'source clip {clip_id}: "{cap}"' if cap else f"source clip {clip_id}"
    fig.suptitle(f"{head}\n{src_line}\n({sub})", fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── figure 2: quantification ──────────────────────────────────────────────────────
def _corr_matrix(maps):
    n = len(maps)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = flat_corr(maps[i], maps[j])
    return M


def _plot_corr(ax, M, labels, title):
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=40, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    ax.set_title(title, fontsize=8.5)
    return im


def plot_quant(clip_id, caption, instructions, m1_maps, m2_maps, src_act, out_path):
    short = [e if len(e) <= 18 else e[:16] + "…" for e in instructions]
    n = len(instructions)

    M1c, M2c = _corr_matrix(m1_maps), _corr_matrix(m2_maps)
    off = ~np.eye(n, dtype=bool)
    m1_mean = M1c[off].mean() if n > 1 else float("nan")
    m2_mean = M2c[off].mean() if n > 1 else float("nan")

    # each mask's correlation with the instruction-independent source motion
    m1_src = [flat_corr(m, src_act) for m in m1_maps]
    m2_src = [flat_corr(m, src_act) for m in m2_maps]

    fig = plt.figure(figsize=(12.5, 4.8))
    # Dedicated thin column for the colourbar so it never overlaps the bar chart's
    # labels; the bar chart's own y-labels sit on its right edge (tick_right), well
    # clear of the colourbar.
    gs = GridSpec(1, 4, figure=fig, wspace=0.45, top=0.78,
                  width_ratios=[1.0, 1.0, 0.07, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    _plot_corr(ax0, M1c, short,
               f"M1 map corr across instructions\nmean off-diag r = {m1_mean:.2f}")
    ax1 = fig.add_subplot(gs[0, 1])
    im = _plot_corr(ax1, M2c, short,
                    f"M2 map corr across instructions\nmean off-diag r = {m2_mean:.2f}")
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, label="Pearson r")
    cb.ax.yaxis.set_ticks_position("left")     # numbers face the matrices, not the bars
    cb.ax.yaxis.set_label_position("left")

    ax2 = fig.add_subplot(gs[0, 3])
    x = np.arange(n)
    ax2.barh(x - 0.2, m1_src, height=0.38, label="M1", color="#4c72b0")
    ax2.barh(x + 0.2, m2_src, height=0.38, label="M2", color="#c44e52")
    ax2.set_yticks(x)
    ax2.set_yticklabels(short, fontsize=6)
    ax2.yaxis.tick_right()                      # labels on the outer edge, clear of cbar
    ax2.set_xlim(-1, 1)
    ax2.axvline(0, color="k", lw=0.6)
    ax2.set_xlabel("corr(mask, source |Δx0|)", fontsize=8)
    ax2.set_title("Mask vs source motion\n(high ⇒ source-dynamics detector)",
                  fontsize=8.5)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.invert_yaxis()

    cap = _short_caption(caption)
    src = f'   source: "{cap}"' if cap else ""
    fig.suptitle(
        f"Instruction-invariance of the implicit masks  ·  clip {clip_id}{src}\n"
        f"(off-diagonal r ≈ 1 ⇒ the mask ignores the instruction)",
        fontsize=10, y=0.99)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return m1_mean, m2_mean, m1_src, m2_src


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode, is_group, group_mode, _ = resolve_group_context(config)
    arch = config.get("arch", "dit")
    print(f"feature_mode={feature_mode}  arch={arch}  is_group={is_group}  group_mode={group_mode}")
    if feature_mode == "smplh":
        _smplh_body_model(args.smplh_model_path)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std  = np.load(os.path.join(args.data_root, "Std.npy"))
    text_encoder = build_text_encoder(config, device=device)
    schedule = NoiseSchedule.from_config(config, device=device)

    instructions = args.instructions or list(DEFAULT_INSTRUCTIONS)
    targets = resolve_targets(instructions, args.target_groups, group_mode)
    if not is_group:
        # No body-part axis: the laterality/limb overlay is meaningless.
        targets = [[] for _ in instructions]

    raw_feat, clip_id, length, caption = load_source(
        args.source, args.data_root, args.split, args.max_frames)
    F = length
    x0 = torch.from_numpy((raw_feat - mean) / std).float().unsqueeze(0).to(device)  # (1,F,D)
    valid_frames = torch.ones(F, dtype=torch.bool, device=device)
    print(f"Source: {clip_id}  ({F} frames)   prompt: {caption!r}\n"
          f"instructions: {instructions}")

    editor = MotionEditor(model, schedule, device, is_group=is_group)
    glabels = group_names(group_mode) if is_group else ["all"]

    src_act = source_activity(x0, editor.group_channels, is_group)   # (F, G) reference

    print("Stage 1: inversion …")
    state = editor.invert(x0)

    mask_ts = (torch.linspace(1, schedule.T - 1, args.mask_timesteps).long().tolist()
               if args.mask_timesteps else None)

    m1_maps, m2_maps, bin_maps = [], [], []
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
        tok_info = [text_encoder.token_info(e) for e in instructions]

    for e, ctx, ti in zip(instructions, ctxs, tok_info):
        toks = ti[0]
        sem = semantic_token_subset(*ti)
        # need_attn=True unconditionally so the raw M1 map is always available for the
        # figure, regardless of which binary mask_mode the reader asked to display.
        attn_fg, psi_fg = masking.collect_statistics(
            model, schedule, state.xs, ctx, toks,
            is_group=is_group, timesteps=mask_ts, need_attn=True,
            group_channels=editor.group_channels, valid_frames=valid_frames,
            attn_readout=args.m1_readout, semantic_idxs=sem,
        )
        mdict = masking.build_mask(
            attn_fg, psi_fg, valid_frames, is_group,
            lambda_attn=args.lambda_attn, lambda_noise=args.lambda_noise,
            mask_mode=args.mask_mode,
            group_channels=editor.group_channels, feat_dim=editor.feat_dim,
        )
        m1_maps.append(attn_fg.cpu().numpy())
        m2_maps.append(psi_fg.cpu().numpy())
        bin_maps.append(mdict["m_group"].float().cpu().numpy())
        print(f"  {e!r}: mask {int(mdict['m_group'].sum())} active cells, "
              f"{int(mdict['edited'].sum())}/{F} frames")

    base = os.path.join(args.out_dir, f"{clip_id}_mask_problem")
    plot_main(clip_id, caption, instructions, targets, m1_maps, m2_maps, bin_maps,
              src_act, glabels, args.mask_mode, base + ".png")
    m1_mean, m2_mean, m1_src, m2_src = plot_quant(
        clip_id, caption, instructions, m1_maps, m2_maps, src_act, base + "_quant.png")

    # text summary — the headline numbers the figures visualise
    print("\n── summary ─────────────────────────────────────────────")
    if len(instructions) > 1:
        print(f"instruction-invariance (mean off-diagonal r):  M1 {m1_mean:.3f}   "
              f"M2 {m2_mean:.3f}   (→1 = mask ignores the instruction)")
    print(f"mask↔source-motion corr (mean over instr.):    M1 {np.mean(m1_src):.3f}   "
          f"M2 {np.mean(m2_src):.3f}   (→1 = source-dynamics detector)")
    if is_group:
        print("body-part axis present (G=%d): laterality/limb overlay in red." % len(glabels))
    else:
        print("flat model (G=1): temporal-only, body-part overlay omitted.")


if __name__ == "__main__":
    main()
