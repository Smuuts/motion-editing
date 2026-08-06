"""
Can a frozen text↔motion scorer tell LEFT from RIGHT?

The pre-gate for Option 10 ([AttentionGrounding_Options.md](../docs/AttentionGrounding_Options.md)):
that route takes the edit's localisation from an external frozen scorer instead of from
this backbone, whose masks are measured to have no laterality at all (instruction-
invariance r 0.985, net laterality response ≈ 0, on the converged x0 checkpoint). The
whole family rests on one assumption nobody has tested — that the borrowed scorer knows
something our denoiser does not. If it does not, Option 10 dies here for half a day's
compute, and "laterality is absent from motion-language embeddings, not just from our
model" becomes a stronger claim than the in-house negative it replaces.

No checkpoint of ours is involved: this measures the scorer, not the editor.

WHY THE OBVIOUS EXPERIMENT IS A TRAP
------------------------------------
Scoring one clip against "…left leg" vs "…right leg" and seeing "left" win proves
nothing: a scorer with a constant preference for the word "left" does that on every clip,
including clips that kick right. Both parts below are therefore built as *forced choices
that a constant bias cannot win* (chance is exactly 50 %), and every accuracy is reported
with a binomial CI and split by side.

WHAT IS MEASURED
----------------
Part A — mirrored caption × mirrored motion, fully in-distribution and tagger-free.
  HumanML3D ships an `M`-prefixed mirror of every clip whose caption is the left/right
  word-swap of the original ("drying his right arm" ↔ "…left arm"), and leaves captions
  without a laterality word byte-identical — which both supplies the minimal pairs and
  filters them. For motions {x, Mx} × captions {c, Mc}, count two forced choices:
      s(x, c)  > s(x, Mc)      the real motion prefers its own caption
      s(Mx, Mc) > s(Mx, c)     the mirrored motion prefers the mirrored caption

Part B — instruction antisymmetry, on the short imperative strings the editor sends.
  Needs no ground-truth side label: mirroring the motion must flip the preference.
      Δ(x)  = s(x,  "…left …") − s(x,  "…right …")
      Δ(Mx) = s(Mx, "…left …") − s(Mx, "…right …")
  A laterality-capable scorer gives sign Δ(x) ≠ sign Δ(Mx). Reported as the sign-flip
  rate plus the effect size |Δ| against the scorer's own score spread.

Controls — an unrelated-caption arm (must be near-ceiling, else the plumbing is broken
and nothing else in the run means anything), a per-side split (catches a constant side
bias), a tagger check against the tokens on disk, and an embedding-level check that our
tagging of a caption lands where the disk tokens land.

Usage
-----
    python src/probe_scorer_laterality.py --data_root data/HumanML3D/HumanML3D

    # quick smoke run
    python src/probe_scorer_laterality.py --data_root data/HumanML3D/HumanML3D \\
        --max_pairs 20 --out_dir /tmp/scorer_gate
"""

import os
import json
import argparse

import numpy as np

from analysis.instructions import DEFAULT_INSTRUCTIONS, LAT_PAIRS
from analysis.scorers import build_scorer
from data.clips import load_clip, read_tagged_caption, split_ids
from data.text_tags import verify_against_dataset
from utils.cli import resolve_device
from utils.probe import accuracy_block

# Pairs of DEFAULT_INSTRUCTIONS indices that differ ONLY in the laterality word.
INSTRUCTION_PAIRS = LAT_PAIRS


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="HumanML3D root (new_joint_vecs/, texts/, <split>.txt).")
    p.add_argument("--split", default="val")
    p.add_argument("--max_frames", type=int, default=196)
    p.add_argument("--scorer", default="t2m", help="Frozen scorer to gate (t2m | tmr).")
    p.add_argument("--evaluator_dir", default="data/t2m_evaluator",
                   help="T2M evaluator dir (checkpoint/, glove/, t2m/.../meta/).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap the number of mirrored pairs (default: all in the split).")
    p.add_argument("--tagger_check", type=int, default=200,
                   help="Captions to re-tag for the tagger control (0 to skip).")
    p.add_argument("--device", default=None, help="'cuda'/'cpu' (default: cuda if available).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="eval_results/scorer_laterality")
    return p.parse_args()


# ── the pair set ────────────────────────────────────────────────────────────────

def lateralised_pairs(data_root, split, max_pairs=None):
    """[(clip_id, caption, mirror_caption, tokens, mirror_tokens)] for lateralised clips.

    A clip qualifies when its mirror's caption *differs* from its own — which happens
    exactly when the caption contains a laterality word, since HumanML3D copies
    non-lateralised captions to the `M` twin verbatim. That is a stricter filter than a
    keyword list and needs no vocabulary of our own.
    """
    pairs = []
    for cid in split_ids(data_root, split):
        if cid.startswith("M"):
            continue                                   # the mirror is reached via its base
        text, tokens = read_tagged_caption(data_root, cid)
        mtext, mtokens = read_tagged_caption(data_root, "M" + cid)
        if not text or not mtext or not tokens or not mtokens:
            continue
        if text == mtext:
            continue                                   # no laterality word in the caption
        pairs.append((cid, text, mtext, tokens, mtokens))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def caption_side(text):
    """'left' / 'right' / 'both' — which side the caption names first (for the split)."""
    t = text.lower()
    li, ri = t.find("left"), t.find("right")
    if li < 0 and ri < 0:
        return "none"
    if li < 0:
        return "right"
    if ri < 0:
        return "left"
    return "left" if li < ri else "right"


# ── the three measurements ──────────────────────────────────────────────────────
# `wilson_ci` / `accuracy_block` (the forced-choice reporting) live in utils/probe.py,
# shared with the other probes that ask a question a constant bias cannot win.

def part_a(scorer, embs, pairs):
    """Mirrored caption × mirrored motion: does each motion prefer its own caption?"""
    s = {(m, t): scorer.similarity(embs["motion"][m], embs["text"][t])
         for m in ("x", "mx") for t in ("c", "mc")}
    win_x = s[("x", "c")] > s[("x", "mc")]              # real motion → real caption
    win_mx = s[("mx", "mc")] > s[("mx", "c")]           # mirrored motion → mirrored caption
    sides = np.array([caption_side(p[1]) for p in pairs])

    # If the motion plays no part in the choice, then s(x,c) > s(x,mc) holds exactly when
    # s(mx,c) > s(mx,mc) — the two arms disagree by construction and their accuracies sum
    # to 1. This measures that directly: the rate at which swapping the motion for its
    # mirror does NOT change which caption wins, i.e. the caption alone decided.
    caption_decides = win_x != win_mx

    out = {"overall": accuracy_block(np.concatenate([win_x, win_mx]), "Part A (both arms)"),
           "real_motion": accuracy_block(win_x, "  real motion → own caption"),
           "mirror_motion": accuracy_block(win_mx, "  mirror motion → mirror caption"),
           "caption_decides_rate": float(np.mean(caption_decides)),
           "margin_mean": float(np.mean(np.concatenate(
               [s[("x", "c")] - s[("x", "mc")], s[("mx", "mc")] - s[("mx", "c")]]))),
           "by_side": {}}
    for side in ("left", "right"):
        m = sides == side
        if m.any():
            out["by_side"][side] = accuracy_block(
                np.concatenate([win_x[m], win_mx[m]]), f"  captions naming '{side}' first")
    return out


def part_b(scorer, embs, instr_embs, instructions):
    """Instruction antisymmetry: mirroring the motion must flip the preferred side."""
    results = {}
    flips_all = []
    for i, j in INSTRUCTION_PAIRS:
        d_x = (scorer.similarity(embs["motion"]["x"], instr_embs[i])
               - scorer.similarity(embs["motion"]["x"], instr_embs[j]))
        d_mx = (scorer.similarity(embs["motion"]["mx"], instr_embs[i])
                - scorer.similarity(embs["motion"]["mx"], instr_embs[j]))
        flips = np.sign(d_x) != np.sign(d_mx)
        flips_all.append(flips)
        spread = float(np.std(np.concatenate([d_x, d_mx])))
        key = f"{instructions[i]} vs {instructions[j]}"
        results[key] = {
            **accuracy_block(flips, f"  {key}"),
            # |Δ| relative to the scorer's own score scale: a sign flip on numerical
            # noise is not a laterality response, so report the size too.
            "abs_delta_mean": float(np.mean(np.abs(np.concatenate([d_x, d_mx])))),
            "delta_spread": spread,
            # A constant side preference shows up as both means having the same sign.
            "mean_delta_real": float(np.mean(d_x)),
            "mean_delta_mirror": float(np.mean(d_mx)),
        }
    results["overall"] = accuracy_block(np.concatenate(flips_all), "Part B (both pairs)")
    return results


def controls(scorer, embs, pairs, rng):
    """Plumbing calibration: the same forced choice against an unrelated clip's caption.

    If the scorer cannot prefer a clip's own caption over a randomly chosen one, it is
    not working (or the normalisation/token path is wrong) and the laterality numbers
    above are meaningless — the same calibrate-before-you-conclude step the MotionFix
    bridge got (see FINDINGS.md).
    """
    n = len(pairs)
    perm = rng.permutation(n)
    perm[perm == np.arange(n)] = (perm[perm == np.arange(n)] + 1) % n   # never self
    own = scorer.similarity(embs["motion"]["x"], embs["text"]["c"])
    other = scorer.similarity(embs["motion"]["x"], embs["text"]["c"][perm])
    return {"unrelated_caption": accuracy_block(
        own > other, "Control: own caption > unrelated caption")}


def encoder_diagnostics(embs):
    """Which encoder loses laterality — the motion side, the text side, or neither?

    A forced choice can only work if mirroring actually MOVES the embeddings. Comparing
    the mirror distance against the ordinary between-clip distance in the same space
    localises the failure: a motion encoder that is mirror-invariant (ratio ≈ 0) cannot
    be rescued by any text, and a text encoder that ignores the left/right word has the
    same effect from the other side. This turns a bare "fails" into a statement about
    where in the pipeline laterality disappears.
    """
    def ratio(a, b):
        mirror = np.sqrt(((a - b) ** 2).sum(axis=-1)).mean()
        # Scale reference: distance between two unrelated rows of the same kind.
        between = np.sqrt(((a[:-1] - a[1:]) ** 2).sum(axis=-1)).mean()
        return {"mirror_distance": float(mirror), "between_distance": float(between),
                "relative": float(mirror / between) if between else float("nan")}

    return {"motion": ratio(embs["motion"]["x"], embs["motion"]["mx"]),
            "text": ratio(embs["text"]["c"], embs["text"]["mc"])}


def tagger_control(scorer, data_root, pairs, n_check):
    """Does our tagging of a caption land where the caption's disk tokens land?

    `verify_against_dataset` compares token *strings*; what actually matters is whether
    the embedding moves, since the evaluator's VIP lists override POS for every
    body-part and laterality word. This measures that directly.
    """
    out = {}
    if n_check:
        out["token_match"] = verify_against_dataset(data_root, n=n_check)
    texts = [p[1] for p in pairs]
    from_disk = scorer.embed_texts(tokens=[p[3] for p in pairs])
    from_tagger = scorer.embed_texts(texts=texts)
    dist = np.sqrt(((from_disk - from_tagger) ** 2).sum(axis=-1))
    # Scale reference: how far apart two *different* captions are in the same space.
    ref = np.sqrt(((from_disk[:-1] - from_disk[1:]) ** 2).sum(axis=-1)).mean()
    out["embedding_drift"] = {
        "mean_distance": float(dist.mean()),
        "between_caption_distance": float(ref),
        "relative": float(dist.mean() / ref) if ref else float("nan"),
    }
    return out


# ── reporting ───────────────────────────────────────────────────────────────────

def _line(block):
    lo, hi = block["ci95"]
    flag = ("PASS" if block["beats_chance"]
            else "BELOW chance" if block["below_chance"] else "at chance")
    return (f"{block['label']:50s} {block['accuracy']:.3f}  "
            f"[{lo:.3f}, {hi:.3f}]  n={block['n']:<5d} {flag}")


def print_report(res):
    print("\n" + "=" * 78)
    print(f"Scorer laterality pre-gate — {res['scorer']}   "
          f"({res['n_pairs']} mirrored pairs, chance = 0.500)")
    print("=" * 78)

    print("\nCONTROL (must pass, else nothing below is interpretable)")
    print(_line(res["controls"]["unrelated_caption"]))
    drift = res["tagger"]["embedding_drift"]
    print(f"{'  tagged-vs-disk caption embedding drift':50s} {drift['mean_distance']:.3f}  "
          f"({drift['relative']:.2f}× the gap between two different captions)")
    if "token_match" in res["tagger"]:
        tm = res["tagger"]["token_match"]
        print(f"{'  tagger vs disk tokens (exact/words/vip)':50s} "
              f"{tm['exact_match_rate']:.3f} / {tm['word_match_rate']:.3f} / "
              f"{tm['vip_match_rate']:.3f}  n={tm['checked']}")

    print("\nDIAGNOSTIC — does mirroring move the embedding at all? "
          "(distance relative to two unrelated clips/captions)")
    for side in ("motion", "text"):
        d = res["encoders"][side]
        print(f"{'  ' + side + ' encoder: mirror vs between':50s} "
              f"{d['mirror_distance']:.3f} / {d['between_distance']:.3f} = "
              f"{d['relative']:.3f}")

    print("\nPART A — mirrored caption × mirrored motion (dataset captions, no tagger)")
    for key in ("overall", "real_motion", "mirror_motion"):
        print(_line(res["part_a"][key]))
    for block in res["part_a"]["by_side"].values():
        print(_line(block))
    print(f"{'  caption alone decided the choice':50s} "
          f"{res['part_a']['caption_decides_rate']:.3f}   "
          f"(1.000 = swapping the motion for its mirror never changes the winner)")

    print("\nPART B — instruction antisymmetry (sign of the left−right preference must flip)")
    print(_line(res["part_b"]["overall"]))
    for key, block in res["part_b"].items():
        if key == "overall":
            continue
        print(_line(block))
        print(f"{'':46s} mean Δ real {block['mean_delta_real']:+.4f}, "
              f"mirror {block['mean_delta_mirror']:+.4f}, "
              f"|Δ| {block['abs_delta_mean']:.4f} (spread {block['delta_spread']:.4f})")

    a, b = res["part_a"]["overall"], res["part_b"]["overall"]
    print("\n" + "-" * 78)
    print("VERDICT: ", end="")
    if not res["controls"]["unrelated_caption"]["beats_chance"]:
        print("INVALID — the scorer fails the plumbing control; fix that first.")
    elif a["beats_chance"] and b["beats_chance"]:
        print(f"PASS — the scorer is laterality-capable "
              f"(A {a['accuracy']:.3f}, B {b['accuracy']:.3f}). Option 10 is live.")
    elif a["beats_chance"]:
        print(f"PARTIAL — laterality on dataset captions ({a['accuracy']:.3f}) but not on "
              f"instruction-style text ({b['accuracy']:.3f}).")
    else:
        print(f"FAIL — at chance on dataset captions ({a['accuracy']:.3f}); the scorer "
              f"cannot tell left from right. Option 10's premise does not hold.")
        if b["below_chance"]:
            print("         Part B is BELOW chance: the preferred side does not flip when "
                  "the motion is\n         mirrored, i.e. a fixed preference for one "
                  "instruction, independent of the motion.")
    print("-" * 78)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")

    pairs = lateralised_pairs(args.data_root, args.split, args.max_pairs)
    if len(pairs) < 8:
        raise SystemExit(f"only {len(pairs)} lateralised pairs in {args.split} — too few.")
    print(f"Lateralised mirrored pairs in {args.split}: {len(pairs)}")

    scorer = build_scorer(args.scorer, evaluator_dir=args.evaluator_dir, device=device)

    print("Embedding motions …")
    motions_x, motions_mx = [], []
    for cid, *_ in pairs:
        motions_x.append(load_clip(args.data_root, cid, args.max_frames)[0])
        motions_mx.append(load_clip(args.data_root, "M" + cid, args.max_frames)[0])
    embs = {
        "motion": {"x": scorer.embed_motions(motions_x),
                   "mx": scorer.embed_motions(motions_mx)},
        "text": {"c": scorer.embed_texts(tokens=[p[3] for p in pairs]),
                 "mc": scorer.embed_texts(tokens=[p[4] for p in pairs])},
    }
    instr_embs = scorer.embed_texts(texts=DEFAULT_INSTRUCTIONS)

    res = {
        "scorer": args.scorer,
        "data_root": args.data_root,
        "split": args.split,
        "n_pairs": len(pairs),
        "instructions": DEFAULT_INSTRUCTIONS,
        "seed": args.seed,
        "controls": controls(scorer, embs, pairs, rng),
        "encoders": encoder_diagnostics(embs),
        "tagger": tagger_control(scorer, args.data_root, pairs, args.tagger_check),
        "part_a": part_a(scorer, embs, pairs),
        "part_b": part_b(scorer, embs, instr_embs, DEFAULT_INSTRUCTIONS),
        "clips": [p[0] for p in pairs],
    }

    out = os.path.join(args.out_dir, f"{args.scorer}_laterality.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print_report(res)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
