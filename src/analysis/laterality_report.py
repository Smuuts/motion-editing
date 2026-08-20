"""
Rendering the frozen-scorer laterality gate: one line per forced-choice block, then the
verdict.

Every block is reported with its binomial CI and its own chance level, because the whole
design rests on forced choices a constant bias cannot win — a bare accuracy would hide
exactly the failure the CI exposes.
"""

from utils.logger import get_logger

log = get_logger(__name__)

def _line(block):
    lo, hi = block["ci95"]
    flag = ("PASS" if block["beats_chance"]
            else "BELOW chance" if block["below_chance"] else "at chance")
    return (f"{block['label']:50s} {block['accuracy']:.3f}  "
            f"[{lo:.3f}, {hi:.3f}]  n={block['n']:<5d} {flag}")


def print_report(res):
    log.section(f"Scorer laterality pre-gate — {res['scorer']}   "
                f"({res['n_pairs']} mirrored pairs, chance = 0.500)")

    log.info("\nCONTROL (must pass, else nothing below is interpretable)")
    log.info(_line(res["controls"]["unrelated_caption"]))
    drift = res["tagger"]["embedding_drift"]
    log.info(f"{'  tagged-vs-disk caption embedding drift':50s} {drift['mean_distance']:.3f}  "
          f"({drift['relative']:.2f}× the gap between two different captions)")
    if "token_match" in res["tagger"]:
        tm = res["tagger"]["token_match"]
        log.info(f"{'  tagger vs disk tokens (exact/words/vip)':50s} "
              f"{tm['exact_match_rate']:.3f} / {tm['word_match_rate']:.3f} / "
              f"{tm['vip_match_rate']:.3f}  n={tm['checked']}")

    log.info("\nDIAGNOSTIC — does mirroring move the embedding at all? "
          "(distance relative to two unrelated clips/captions)")
    for side in ("motion", "text"):
        d = res["encoders"][side]
        log.info(f"{'  ' + side + ' encoder: mirror vs between':50s} "
              f"{d['mirror_distance']:.3f} / {d['between_distance']:.3f} = "
              f"{d['relative']:.3f}")

    log.info("\nPART A — mirrored caption × mirrored motion (dataset captions, no tagger)")
    for key in ("overall", "real_motion", "mirror_motion"):
        log.info(_line(res["part_a"][key]))
    for block in res["part_a"]["by_side"].values():
        log.info(_line(block))
    log.info(f"{'  caption alone decided the choice':50s} "
          f"{res['part_a']['caption_decides_rate']:.3f}   "
          f"(1.000 = swapping the motion for its mirror never changes the winner)")

    log.info("\nPART B — instruction antisymmetry (sign of the left−right preference must flip)")
    log.info(_line(res["part_b"]["overall"]))
    for key, block in res["part_b"].items():
        if key == "overall":
            continue
        log.info(_line(block))
        log.info(f"{'':46s} mean Δ real {block['mean_delta_real']:+.4f}, "
              f"mirror {block['mean_delta_mirror']:+.4f}, "
              f"|Δ| {block['abs_delta_mean']:.4f} (spread {block['delta_spread']:.4f})")

    a, b = res["part_a"]["overall"], res["part_b"]["overall"]
    log.rule()
    if not res["controls"]["unrelated_caption"]["beats_chance"]:
        log.info("VERDICT: INVALID — the scorer fails the plumbing control; fix that "
                 "first.")
    elif a["beats_chance"] and b["beats_chance"]:
        log.info(f"PASS — the scorer is laterality-capable "
              f"(A {a['accuracy']:.3f}, B {b['accuracy']:.3f}). The route is live.")
    elif a["beats_chance"]:
        log.info(f"PARTIAL — laterality on dataset captions ({a['accuracy']:.3f}) but not on "
              f"instruction-style text ({b['accuracy']:.3f}).")
    else:
        log.info(f"FAIL — at chance on dataset captions ({a['accuracy']:.3f}); the scorer "
              f"cannot tell left from right. The route's premise does not hold.")
        if b["below_chance"]:
            log.info("         Part B is BELOW chance: the preferred side does not flip when "
                  "the motion is\n         mirrored, i.e. a fixed preference for one "
                  "instruction, independent of the motion.")
    log.info("-" * 78)
