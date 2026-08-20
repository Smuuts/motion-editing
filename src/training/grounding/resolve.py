"""
Resolving WHICH blocks and WHICH text columns the grounding signal covers.

Both answers are needed twice — once at training time to decide what to supervise, and
once at read-out time to decide what to average — and a mismatch between the two is
invisible downstream, so the two callers share these functions rather than each
re-deriving the set.
"""

from utils.logger import get_logger

log = get_logger(__name__)

def resolve_ground_layers(spec, num_layers: int) -> list[int]:
    """`--attn_ground_layers` → block indices to sample from.

    "middle" (the default) is TokenCompose's own layer choice ported: they supervise the
    middle block and the 16/32 decoder blocks, explicitly NOT the encoder, and this
    project's own self-attention probe independently found body-part structure peaking
    in blocks 4–5 of 8 (measured). Concretely the middle 3/8 of the stack:
    {3, 4, 5} at the default depth 8. "all" supervises every block; an explicit
    "3,4,5" (or "3 4 5") overrides both.
    """
    if isinstance(spec, (list, tuple)):
        layers = [int(x) for x in spec]
    elif spec == "all":
        layers = list(range(num_layers))
    elif spec == "middle":
        lo, hi = int(num_layers * 0.375), int(num_layers * 0.75)
        layers = list(range(lo, max(hi, lo + 1)))
    else:
        layers = [int(x) for x in str(spec).replace(",", " ").split()]

    bad = [l for l in layers if not 0 <= l < num_layers]
    if bad or not layers:
        raise ValueError(
            f"--attn_ground_layers {spec!r} resolved to {layers}, which is empty or out "
            f"of range for a {num_layers}-block backbone (valid: 0..{num_layers - 1}).")
    return layers


def resolve_readout_layers(config: dict, override=None, num_layers: int | None = None):
    """Which blocks the M1 read-out should average — the inference-side counterpart of
    `resolve_ground_layers`. Returns a list of block indices, or None for "all blocks".

    WHY THIS EXISTS. `masking.collect_statistics` averages cross-attention over every
    block. Under a grounding run only `attn_ground_layers` (3 of 8 by default) were ever
    supervised, so averaging all 8 mixes 3 grounded maps into 5 ungrounded ones and
    dilutes the signal roughly 8/3× — the read-out understates the effect it is being
    used to measure. Resolving the default from the checkpoint's OWN
    `attn_ground_layers` makes the measurement match the training.

    `override`: None/"auto" → read the checkpoint config; "all" → every block (the
    historical behaviour, and what any pre-grounding checkpoint gets since its config
    has no such key); an explicit "3,4,5" → exactly those, for the A/B. Same
    auto|explicit precedent as `--edit_space`.
    """
    if override not in (None, "auto"):
        if override == "all":
            return None
        return resolve_ground_layers(override, num_layers or config.get("num_layers", 8))
    spec = config.get("attn_ground_layers")
    if not spec or not config.get("attn_ground_weight", 0.0) > 0.0:
        return None                      # never grounded → all blocks, as before
    return resolve_ground_layers(spec, num_layers or config.get("num_layers", 8))


def resolve_readout_columns(text, encoder, config=None, override=None,
                            group_mode: str = "parts"):
    """(token_idxs, semantic_idxs, mode) — which TEXT COLUMNS the M1 read-out reads.

    The column-axis twin of `resolve_readout_layers`, and it exists for exactly the same
    reason. The grounding loss supervises the body-part SPAN of a caption ("left leg")
    and nothing else, but `collect_statistics` reads M1 from every content token —
    including the verb, which was never supervised and still behaves like the ungrounded
    attention it came from, i.e. a source-motion detector. Averaging the two dilutes the
    grounded columns with ungrounded ones.

    Measured on `exp_hml3d_masked` ("The arm/leg asymmetry is COLUMN
    dilution"): restricting to the span leaves ARM instructions unchanged (0.475 → 0.476)
    and lifts LEG instructions 0.420 → 0.476, i.e. it removes the limb asymmetry entirely
    and puts both at the metric's ceiling. The asymmetry is a HumanML3D property, not a
    model one — 86.5 % of captions containing a locomotion verb never name a leg — so the
    verb column is where leg semantics live and it is precisely what supervision missed.

    NOT self-fulfilling, which is the objection to answer first: the parser only says
    WHICH COLUMNS to read, never which group they should point at. On the ungrounded twin
    the same restriction makes the mask WORSE (leg alignment 0.205 → 0.147 = chance), so
    the span columns carry the answer only because they were trained to.

    THREE MODES, and they differ in how much they assume:
      "content"  every content token — the historical read. NOTE this includes the
                 function words: the default "raw" readout reads `tok`, not `sem`, so
                 the `_STOP_WORDS` filter that masking.py has always defined was never
                 applied on the default path.
      "semantic" content tokens minus `_STOP_WORDS` ("a", "person", "the", …). Removes
                 only words that cannot carry body-part information, and **keeps the
                 verb**. The conservative middle: no parser knowledge beyond a fixed
                 stop list, so it does not edge toward "build the mask from the text".
      "span"     only the body-part words the grounding loss actually supervised. The
                 strongest and the best-measuring, but it uses the caption parser at
                 inference, which is close to what an explicit group router does — read
                 the trade-off recorded for this flag before making it a default.

    `override`: None/"auto" → "semantic" when the checkpoint was trained with the
    grounding loss, "content" otherwise (so no pre-grounding checkpoint changes
    behaviour). "span" falls back to "content" when the instruction names no body part
    at all ("move faster") — there is no supervised column to read, and the returned
    mode string says so.
    """
    # Local imports: `training` must not depend on `editing`, and body_part_labels keeps
    # its pure-text core import-free (see its module docstring).
    from data.body_part_labels import to_items
    from editing.masking import semantic_token_subset

    positions, labels = encoder.token_info(text)
    content = (positions, semantic_token_subset(positions, labels))

    mode = override if override not in (None, "auto") else (
        "semantic" if (config or {}).get("attn_ground_weight", 0.0) > 0.0 else "content")
    if mode == "content":
        return content[0], content[1], "content"
    if mode == "semantic":
        # Read over the stop-word-filtered set. `sem` is already that subset, and
        # semantic_token_subset falls back to the full list rather than returning
        # empty, so this can never read zero columns.
        return list(content[1]), list(content[1]), "semantic"
    if mode != "span":
        raise ValueError(
            f"--m1_columns must be auto|span|semantic|content, got {override!r}")

    # The supervised span depends on whether the checkpoint was trained with tier-3 verb
    # labels: under nouns-only supervision the verb column is NOT supervised and reading
    # it is the dilution this mode exists to avoid, while a verb-trained checkpoint
    # supervised it and excluding it would throw signal away. Default False for any
    # config predating the flag — those runs are nouns-only by definition.
    cols = sorted({c for item in to_items(
        text, encoder.token_spans(text), group_mode,
        include_verbs=bool((config or {}).get("attn_ground_verbs", False)))
        for c in item["W"]})
    if not cols:
        return content[0], content[1], "content (no body-part span)"
    return cols, cols, "span"
