"""
Does MotionFix's TMR retrieval model encode LEFT vs RIGHT?

Gate test for queue item F.14 (frozen-detector classifier guidance). Every mask signal read
out of the diffusion backbone has been measured to be laterality-blind — attention, ψ,
self-attention, every readout variant, every noise window, disattenuated r ≈ 1.0 throughout
(docs/FINDINGS.md). F.14 proposes using a frozen external model instead, with TMR the obvious
candidate since it is already in the eval stack. That only makes sense if TMR itself can tell
left from right, and this test answers that in forward passes only — no differentiable decode
path, no gradients, no SMPL-H x0 checkpoint needed, all of which F.14 proper would require.

DESIGN
------
For each clip with a lateralised caption:

  M   the motion            T    its caption
  M'  the mirrored motion   T'   the caption with left<->right swapped
                            T''  the caption with the LIMB swapped (arm<->leg, ...)

  d_lat  = s(M, T) − s(M, T')    how much the score drops when the SIDE word is wrong
  d_lat' = s(M', T') − s(M', T)  same on the mirrored clip (its correct caption is T')
  d_cat  = s(M, T) − s(M, T'')   how much it drops when the LIMB word is wrong

d_cat is the calibration: it is the contrast the diffusion backbone *can* partly do, so it
says how big a "TMR clearly noticed" effect looks on this scale. d_lat is the question. The
mirror is what makes d_lat fair — M and M' are the same motion up to reflection, so nothing
but laterality distinguishes them, and T/T' differ in one word.

Reported alongside:
  * a RETRIEVAL SANITY check (do clips retrieve their own captions?) — if this fails the
    feature layout / normalisation / fps handling is wrong and no other number means anything;
  * cos(M, M') — whether the motion embedding separates a clip from its mirror at all, which
    is the ceiling on any laterality behaviour downstream.

Everything is a frozen forward pass. TMR is rebuilt from the pre-extracted state dicts in
`data/motionfix/eval-deps/last_weights/` (plain torch — no hydra/lightning), so this runs in
the `ma` env; only `einops` is needed beyond the training stack.

Usage
-----
    python src/probe_tmr_laterality.py \
        --data_root data/HumanML3D/HumanML3D_smplh \
        --num_clips 64 --out eval_results/tmr_laterality
"""

import os
import re
import sys
import json
import argparse

import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.smplh_features import (
    features_to_smpl, smplh_to_features, resample_motion, mirror_smplh,
)

REPO = os.path.dirname(src_dir)
MFIX = os.path.join(REPO, "data", "motionfix")
EVAL_DEPS = os.path.join(MFIX, "eval-deps")

# From eval-deps/config.json — both encoders are ACTORStyleEncoder with these dims.
ENC_KW = dict(vae=True, latent_dim=256, ff_size=1024, num_layers=6, num_heads=4)

_LAT = [("left", "right")]
_LIMB = [("arm", "leg"), ("arms", "legs"), ("hand", "foot"), ("hands", "feet"),
         ("wrist", "ankle"), ("elbow", "knee"), ("shoulder", "hip")]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data/HumanML3D/HumanML3D_smplh",
                   help="SMPL-H dataset root (new_joint_vecs/ + texts/ + split .txt).")
    p.add_argument("--split", default="val")
    p.add_argument("--num_clips", type=int, default=64)
    p.add_argument("--min_frames", type=int, default=40)
    p.add_argument("--src_fps", type=float, default=20.0, help="Dataset fps.")
    p.add_argument("--tmr_fps", type=float, default=30.0, help="TMR's native fps.")
    p.add_argument("--no_resample", action="store_true",
                   help="Feed the clips at --src_fps (ablation for the fps handling).")
    p.add_argument("--out", default="eval_results/tmr_laterality")
    p.add_argument("--device", default=None)
    return p.parse_args()


def swap_words(text, pairs):
    """Whole-word swap of each (a, b) pair, simultaneously in both directions."""
    def repl(m):
        w = m.group(0)
        low = w.lower()
        for a, b in pairs:
            if low == a:
                out = b
            elif low == b:
                out = a
            else:
                continue
            return out.capitalize() if w[0].isupper() else out
        return w
    words = "|".join(sorted({w for pair in pairs for w in pair}, key=len, reverse=True))
    return re.sub(rf"\b({words})\b", repl, text, flags=re.IGNORECASE)


def has_any(text, pairs):
    words = "|".join({w for pair in pairs for w in pair})
    return re.search(rf"\b({words})\b", text, flags=re.IGNORECASE) is not None


# ── TMR, rebuilt from the extracted state dicts ─────────────────────────────────
def _load_module(path, name):
    """Import a MotionFix module by file path, bypassing its package __init__.

    `src.tmr.__init__` imports temos.py, which needs pytorch_lightning — a dependency that
    lives only in `mfix-env`. actor.py and text_encoder.py themselves are plain
    torch/einops/transformers, so loading them directly keeps this probe in the `ma` env.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tmr(device):
    tmr_dir = os.path.join(MFIX, "src", "tmr")
    actor = _load_module(os.path.join(tmr_dir, "actor.py"), "_mfix_actor")
    txt = _load_module(os.path.join(tmr_dir, "text_encoder.py"), "_mfix_text_encoder")
    ACTORStyleEncoder, TextToEmb = actor.ACTORStyleEncoder, txt.TextToEmb

    motion_enc = ACTORStyleEncoder(nfeats=135, **ENC_KW)
    text_enc = ACTORStyleEncoder(nfeats=768, **ENC_KW)
    for enc, name in ((motion_enc, "motion_encoder"), (text_enc, "text_encoder")):
        sd = torch.load(os.path.join(EVAL_DEPS, "last_weights", f"{name}.pt"),
                        map_location="cpu", weights_only=True)
        enc.load_state_dict(sd, strict=True)        # strict: a silent mismatch would
        enc.eval().to(device)                       # invalidate every number below
        for p in enc.parameters():
            p.requires_grad_(False)
    stats = os.path.join(EVAL_DEPS, "stats", "humanml3d", "amass_feats")
    mean = torch.load(os.path.join(stats, "mean.pt"), map_location="cpu",
                      weights_only=True).float().to(device)
    std = torch.load(os.path.join(stats, "std.pt"), map_location="cpu",
                     weights_only=True).float().to(device)
    text_to_emb = TextToEmb("distilbert-base-uncased", device=str(device))
    return motion_enc, text_enc, (mean, std), text_to_emb


def encode_seq(encoder, x, lengths, device):
    """x: (B, T, D) padded, lengths: list[int] -> (B, latent) mu of the VAE head."""
    B, T, _ = x.shape
    mask = (torch.arange(T, device=device)[None] < torch.tensor(lengths, device=device)[:, None])
    with torch.no_grad():
        out = encoder({"x": x.to(device), "mask": mask})
    return out[:, 0]                                      # mu (vae=True -> [mu, logvar])


def pad_stack(seqs, device):
    T = max(len(s) for s in seqs)
    D = seqs[0].shape[-1]
    out = torch.zeros(len(seqs), T, D, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.float32, device=device)
    return out, [len(s) for s in seqs]


def cos_sim(a, b):
    return torch.nn.functional.normalize(a, dim=-1) @ \
           torch.nn.functional.normalize(b, dim=-1).T


# ── motion preparation: dataset features -> TMR input ───────────────────────────
def to_tmr_features(raw_feats, args, mirror=False):
    """Dataset (T,135) raw features @src_fps -> TMR-layout (T',135) @tmr_fps.

    Our `smplh_to_features` layout IS TMR's encoder layout
    ([trans_delta | body_pose_6d | global_orient_6d]); only the fps and the normalisation
    stats differ. Mirroring and resampling both have to happen in raw SMPL space, hence the
    round trip through `features_to_smpl`.
    """
    rots, trans = features_to_smpl(raw_feats)
    if mirror:
        rots, trans = mirror_smplh(rots, trans)
    if not args.no_resample and args.src_fps != args.tmr_fps:
        rots, trans = resample_motion(rots, trans, args.src_fps, args.tmr_fps)
    return smplh_to_features(rots, trans)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    with open(os.path.join(args.data_root, f"{args.split}.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]

    # Keep clips whose caption names a side AND a limb, so all three texts are well-defined.
    clips = []
    for cid in ids:
        tp = os.path.join(args.data_root, "texts", f"{cid}.txt")
        fp = os.path.join(args.data_root, "new_joint_vecs", f"{cid}.npy")
        if not (os.path.exists(tp) and os.path.exists(fp)):
            continue
        with open(tp) as f:
            lines = [l.split("#")[0].strip() for l in f if l.strip()]
        cap = lines[0] if lines else ""
        if not cap or not has_any(cap, _LAT) or not has_any(cap, _LIMB):
            continue
        feats = np.load(fp)
        if len(feats) < args.min_frames:
            continue
        clips.append((cid, cap, feats))
        if len(clips) >= args.num_clips:
            break
    if len(clips) < 4:
        raise SystemExit(f"only {len(clips)} usable clips found — loosen the filters")
    print(f"{len(clips)} clips with a side word + a limb word in the caption\n")

    motion_enc, text_enc, (mean, std), text_to_emb = load_tmr(device)
    print("TMR loaded from eval-deps/last_weights (motion 135->256, text 768->256)\n")

    ids_, caps, feats_o, feats_m = [], [], [], []
    for cid, cap, feats in clips:
        ids_.append(cid)
        caps.append(cap)
        feats_o.append((to_tmr_features(feats, args, mirror=False) - mean.cpu().numpy())
                       / (std.cpu().numpy() + 1e-12))
        feats_m.append((to_tmr_features(feats, args, mirror=True) - mean.cpu().numpy())
                       / (std.cpu().numpy() + 1e-12))

    caps_lat = [swap_words(c, _LAT) for c in caps]          # side flipped
    caps_cat = [swap_words(c, _LIMB) for c in caps]         # limb flipped
    print(f"example:\n  T   {caps[0]!r}\n  T'  {caps_lat[0]!r}\n  T'' {caps_cat[0]!r}\n")

    emb_o = encode_seq(motion_enc, *pad_stack(feats_o, device)[:2], device)
    emb_m = encode_seq(motion_enc, *pad_stack(feats_m, device)[:2], device)

    def enc_text(texts):
        d = text_to_emb(texts, device=str(device))
        return encode_seq(text_enc, d["x"].float(), d["length"].tolist(), device)

    t_o, t_lat, t_cat = enc_text(caps), enc_text(caps_lat), enc_text(caps_cat)

    # ── 1. retrieval sanity ────────────────────────────────────────────────────
    S = cos_sim(emb_o, t_o).cpu().numpy()                   # (N motions, N texts)
    n = len(ids_)
    rank_m2t = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    rank_t2m = (S.T > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    r_at = lambda r, k: float((r < k).mean())
    print("── retrieval sanity (does a clip retrieve its own caption?) ───────────")
    print(f"  motion->text  R@1 {r_at(rank_m2t,1):.3f}  R@3 {r_at(rank_m2t,3):.3f}  "
          f"R@10 {r_at(rank_m2t,10):.3f}   (chance R@1 = {1/n:.3f})")
    print(f"  text->motion  R@1 {r_at(rank_t2m,1):.3f}  R@3 {r_at(rank_t2m,3):.3f}  "
          f"R@10 {r_at(rank_t2m,10):.3f}")

    # ── 2. does the motion embedding separate a clip from its mirror? ──────────
    cos_mm = torch.nn.functional.cosine_similarity(emb_o, emb_m, dim=-1).cpu().numpy()
    # baseline: similarity to an unrelated clip, to read cos_mm against
    off = cos_sim(emb_o, emb_o).cpu().numpy()
    other = off[~np.eye(n, dtype=bool)].mean()
    print("\n── motion embedding: clip vs its own mirror ───────────────────────────")
    print(f"  cos(M, M')          {cos_mm.mean():.4f}  (min {cos_mm.min():.4f}, "
          f"max {cos_mm.max():.4f})")
    print(f"  cos(M, other clip)  {other:.4f}   <- the scale: 1.0 would mean 'identical'")

    # ── 3. the paired contrasts ───────────────────────────────────────────────
    d = lambda a, b: (torch.nn.functional.cosine_similarity(a, b, dim=-1)).cpu().numpy()
    s_o_o, s_o_lat, s_o_cat = d(emb_o, t_o), d(emb_o, t_lat), d(emb_o, t_cat)
    s_m_o, s_m_lat = d(emb_m, t_o), d(emb_m, t_lat)
    d_lat = s_o_o - s_o_lat                 # original clip prefers its own side word?
    d_lat_m = s_m_lat - s_m_o               # mirrored clip prefers the swapped word?
    d_cat = s_o_o - s_o_cat                 # calibration: limb word
    print("\n── paired contrasts (>0 = TMR prefers the CORRECT word) ───────────────")
    print(f"  d_cat  limb  swap, original : {d_cat.mean():+.4f}  "
          f"(positive on {int((d_cat>0).sum())}/{n})")
    print(f"  d_lat  side  swap, original : {d_lat.mean():+.4f}  "
          f"(positive on {int((d_lat>0).sum())}/{n})")
    print(f"  d_lat' side  swap, mirrored : {d_lat_m.mean():+.4f}  "
          f"(positive on {int((d_lat_m>0).sum())}/{n})")
    combined = (d_lat + d_lat_m) / 2
    print(f"  laterality, mirror-paired   : {combined.mean():+.4f}  "
          f"(positive on {int((combined>0).sum())}/{n})")
    ratio = combined.mean() / d_cat.mean() if abs(d_cat.mean()) > 1e-9 else float("nan")
    print(f"\n  laterality / category effect ratio: {ratio:.2f}   "
          f"(≈1 ⇒ TMR reads sides as well as limbs; ≈0 ⇒ laterality-blind)")

    res = {
        "n_clips": n, "clip_ids": ids_, "resampled": not args.no_resample,
        "retrieval": {"m2t_R@1": r_at(rank_m2t, 1), "m2t_R@3": r_at(rank_m2t, 3),
                      "t2m_R@1": r_at(rank_t2m, 1), "chance_R@1": 1 / n},
        "cos_motion_vs_mirror": float(cos_mm.mean()),
        "cos_motion_vs_other": float(other),
        "d_cat": float(d_cat.mean()), "d_cat_pos": int((d_cat > 0).sum()),
        "d_lat": float(d_lat.mean()), "d_lat_pos": int((d_lat > 0).sum()),
        "d_lat_mirror": float(d_lat_m.mean()), "d_lat_mirror_pos": int((d_lat_m > 0).sum()),
        "laterality_paired": float(combined.mean()),
        "laterality_paired_pos": int((combined > 0).sum()),
        "laterality_over_category": float(ratio),
        "per_clip": [{"id": ids_[i], "caption": caps[i], "d_lat": float(d_lat[i]),
                      "d_lat_mirror": float(d_lat_m[i]), "d_cat": float(d_cat[i]),
                      "cos_mirror": float(cos_mm[i])} for i in range(n)],
    }
    path = os.path.join(args.out, f"tmr_laterality_{args.split}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
