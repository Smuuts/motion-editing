"""
TokenCompose-style cross-attention grounding loss (Option 1).

    docs/TokenCompose_Handoff.md — the design, the measured gate, the risks
    docs/AttentionGrounding_Options.md §1 — where this sits among the alternatives
    TokenCompose, Wang et al., CVPR 2024 (arXiv 2312.03626) — the source of L_token

WHAT IT DOES
------------
Nothing in the denoising objective ever asks the model to route the word "left" to the
`left_arm` token, so it does not: measured on runs/exp_hml3d_x0, the M1 masks for "raise
the left arm" and "raise the right arm" correlate at r = 0.985. This module adds the
missing objective. For each body-part mention in a caption, the text columns that spell
it out must place their cross-attention mass on that mention's body-part group tokens:

    Â_w      = A[b, :, :, w] / Σ_{f valid, g} A[b, f, g, w]     distribution over cells
    m_S      = Σ_{f valid, g ∈ S} Â_w[f, g]                     mass on the target groups
    L_group  = (1 − m_S)²                                       TokenCompose's L_token
    L_mirror = relu(m_S' − m_S + margin)                        S' = MIRROR[S], tier 1 only

TokenCompose's second term (L_pixel, a per-pixel BCE against the segmentation mask) has
no analogue here: our 7 group tokens are opaque, there is no sub-group resolution to
supervise. Their ablation says L_token carries the load anyway (29.86 → 49.85 of the
52.15 total), so this ports the term that matters and drops the one we cannot express.

The mirror term is ours, not theirs. It targets the one axis no training-free
intervention in this project has ever moved: laterality. L_group alone is satisfied by
"put mass on *an* arm", which a source-motion detector can do without reading the word —
the mirror margin makes the left/right distinction explicit.

WHERE THE LABELS COME FROM
--------------------------
data/body_part_labels.py, offline, from the captions themselves — no segmenter and no
LLM. `S` falls out of the tokeniser's own channel→group partition (model/body_groups.py),
which is why this is cheap in motion and expensive in images.

THE FAILURE MODE THIS CODE IS BUILT TO EXPOSE
---------------------------------------------
Captions describe their clips, so a label correlates with the source clip's dynamics: on
tier-1 items, argmax(source energy) already equals the label 69.5 % of the time. A model
that learns "attend to whatever moves" scores well on L_group without learning any
word→group routing at all. Three things push back — 36.9 % of captions carry ≥2 items
with *different* targets (a token-invariant detector cannot fit those), the mirror term,
and `1 − ᾱ_t` timestep weighting that concentrates the pressure at high noise where
there is no motion left to detect — and `src_corr` in the returned stats *measures*
whether they worked. Kill criterion: src_corr above ~0.5 and rising while m_S rises.
"""

import random
from dataclasses import dataclass, field

import torch

from analysis.instructions import MIRROR
from model.body_groups import group_names


def resolve_ground_layers(spec, num_layers: int) -> list[int]:
    """`--attn_ground_layers` → block indices to sample from.

    "middle" (the default) is TokenCompose's own layer choice ported: they supervise the
    middle block and the 16/32 decoder blocks, explicitly NOT the encoder, and this
    project's own self-attention probe independently found body-part structure peaking
    in blocks 4–5 of 8 (docs/FINDINGS.md). Concretely the middle 3/8 of the stack:
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

    Measured on `exp_hml3d_masked` (docs/FINDINGS.md "The arm/leg asymmetry is COLUMN
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
                 the trade-off in docs/ARCHITECTURE.md before making it a default.

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


@dataclass
class GroundingConfig:
    """Everything train_one_epoch needs to run the grounding loss, resolved once.

    Bundled rather than passed as eight more keyword arguments because they are only
    ever used together, and because `enabled` then has one obvious home.
    """
    weight:         float = 0.0            # λ; 0 = the whole feature is off
    layers:         list[int] = field(default_factory=list)
    mirror:         float = 1.0            # λ_mirror  — tier 1, "beat your mirror"
    even:           float = 0.1            # λ_even    — tier 2, "do not pick a side"
    margin:         float = 0.1
    warmup_epochs:  int = 20
    window:         tuple[int, int] | None = None   # hard timestep gate, if used
    cache:          dict = field(default_factory=dict)   # caption -> [item, ...]
    group_channels: list = field(default_factory=list)   # for the src_corr monitor
    monitor:        bool = True
    mirror_mat:     torch.Tensor | None = None           # see mirror_matrix()

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 and bool(self.layers)

    def active(self, epoch: int) -> bool:
        """From-scratch attention is random noise at epoch 0 — supervising it just
        teaches the model to satisfy a loss on a signal that carries nothing yet.
        TokenCompose finetuned an already-converged model; we do not, hence the warmup
        (default 20 epochs, higher than the doc's 5)."""
        return self.enabled and epoch >= self.warmup_epochs

    def pick_layer(self) -> int:
        """One block per step. Materialising an explicit (B, h, F·G, L) softmax in every
        block at once does not fit; in expectation each candidate block still receives
        the same pressure. Same argument as the entropy regulariser's ent_layer."""
        return random.choice(self.layers)

    def val_layer(self) -> int:
        """Deterministic block for validation, so the val curve is one quantity across
        epochs instead of a random draw."""
        return self.layers[len(self.layers) // 2]


def _mirror_name(name: str) -> str:
    """Axis name → its left/right mirror, unchanged when it has no side.

    Handles both token axes: 'parts' names are seeded from analysis/instructions.MIRROR
    — the same map the laterality probes score against, so the training signal and the
    metric cannot drift apart by one of them redefining the pairing — and 'joints' names
    (L_Elbow / R_Elbow) fall to the prefix rule.
    """
    if name in MIRROR:
        return MIRROR[name]
    for a, b in (("left_", "right_"), ("right_", "left_"), ("L_", "R_"), ("R_", "L_")):
        if name.startswith(a):
            return b + name[len(a):]
    return name


def mirror_matrix(group_mode: str = "parts") -> torch.Tensor:
    """(G, G) permutation sending each group to its mirror, identity for the
    unlateralised ones. `S @ mirror_matrix` turns a target set into its mirror set.

    A group whose mirror is itself makes the margin term unsatisfiable by construction
    (m_mirror ≡ m_S ⇒ a constant relu(margin) with no gradient), which is exactly why
    the loss applies the margin to tier-1 items only — their S is always one-sided.
    """
    names = group_names(group_mode)
    idx = {n: i for i, n in enumerate(names)}
    M = torch.zeros(len(names), len(names))
    for i, n in enumerate(names):
        M[i, idx.get(_mirror_name(n), i)] = 1.0
    return M


def batched_source_activity(motion: torch.Tensor, group_channels,
                            frame_mask: torch.Tensor) -> torch.Tensor:
    """(B, F, G) per-(frame, group) |Δx0| of the SOURCE clip — the batched twin of
    utils.probe.source_activity, which is single-clip and numpy.

    This is the reference the shortcut monitor correlates against: it depends only on
    the clip, never on the caption, so an attention map that tracks it is a motion
    detector rather than a word→group router.

    Frame 0 repeats frame 1 rather than being zeroed (2026-08-16), matching
    `utils.probe.source_activity` and `editing.masking._frame_energy` — the three are one
    definition and are changed together (docs/ARCHITECTURE.md). This shifts the logged
    `src_corr` slightly: it removes one artificially-zero cell per clip that was shared by
    both correlands and therefore inflated their agreement.
    """
    diff = (motion[:, 1:] - motion[:, :-1]).abs()                   # (B, F-1, D)
    act = torch.stack([diff[:, :, ch].mean(dim=-1) for ch in group_channels], dim=-1)
    act = torch.cat([act[:, :1], act], dim=1)                       # (B, F, G)
    return act * frame_mask[:, :, None].to(act.dtype)


def _pearson(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson r over the entries `valid` selects; (n,) out for (n, k) in."""
    w = valid.to(a.dtype)
    n = w.sum(dim=-1).clamp_min(1)
    am = (a * w).sum(-1, keepdim=True) / n[:, None]
    bm = (b * w).sum(-1, keepdim=True) / n[:, None]
    da, db = (a - am) * w, (b - bm) * w
    denom = (da.pow(2).sum(-1) * db.pow(2).sum(-1)).sqrt().clamp_min(1e-12)
    return (da * db).sum(-1) / denom


def collect_items(texts, cache, valid_sample) -> list[tuple[int, dict]]:
    """[(batch row, item)] for every supervised body-part mention in the batch.

    `valid_sample` excludes the CFG-dropout rows: their context was replaced by the
    learned null embedding, so the caption's columns are not in the context at all and
    supervising them pushes attention toward text that is not there — silent noise
    rather than a visible error (docs/TokenCompose_Handoff.md §4.4).
    """
    out = []
    for b, text in enumerate(texts):
        if not valid_sample[b]:
            continue
        for item in cache.get(text, ()):
            if item["W"] and item["S"]:
                out.append((b, item))
    return out


def grounding_loss(A, texts, cache, frame_mask, valid_sample,
                   sample_weight=None, lambda_mirror=1.0, margin=0.1,
                   source_act=None, mirror_mat=None, lambda_even=0.1):
    """TokenCompose L_token (+ the mirror margin) on the supervised text columns.

    A            : (B, F, G, L) head-averaged cross-attention, GRAPH KEPT.
    texts        : the batch's raw captions, the cache's key.
    cache        : {caption: [{"W": cols, "S": groups, "tier": 1|2, "lat": bool}]}.
    frame_mask   : (B, F) bool — real frames. Padding frames are excluded from BOTH the
                   numerator and the normalising denominator, so a short clip is not
                   penalised for the mass its padding never received.
    valid_sample : (B,) bool — False on CFG-dropout rows.
    sample_weight: (B,) per-sample weight, normally 1 − ᾱ_t (see the epoch loop for why
                   NOT ᾱ_t). Weights each item by its row's value; it does not
                   normalise, so the loss stays comparable as the weight distribution
                   shifts.
    source_act   : (B, F, G) optional; enables the src_corr shortcut monitor.
    lambda_even  : weight of the tier-2 evenness term (see the block that computes it).
                   0 reproduces the pre-2026-08-15 loss exactly, which is the A/B control.

    Returns (loss, stats). Averaging is over supervised TOKENS, not samples: a caption
    naming two parts exerts twice the pressure of one naming a single part, which is
    the intent — those two-target captions are precisely the 36.9 % that a
    token-invariant motion detector cannot fit.

    `stats["m_S"]` is the number to watch, not the loss: it reads directly as "the word
    puts m_S of its attention on its own body-part group(s)".

    **Chance is |S|/G averaged over items, NOT 1/G.** A tier-1 item has one target group
    (chance 1/7 = 0.143), a tier-2 limb pair has two (0.286) and a locomotion verb has
    three — legs + root (0.429). Measured on the real caches: **0.203** nouns-only,
    **0.262** with tier-3 verb labels. The Trainer prints the value for the cache it
    actually loaded; do not compare an m_S against 1/G, and do not compare m_S across two
    runs whose label sets differ in target-size mix. `m_S_tier1` is the one number that
    is always comparable, because tier-1 items are single-group by construction.
    """
    picks = collect_items(texts, cache, valid_sample)
    if not picks:
        # Still return a tensor tied to the graph: a Python 0.0 here would detach the
        # step's loss on empty batches and make `loss.backward()` inconsistent between
        # steps. A no-op with the right type, not a special case for the caller.
        return A.sum() * 0.0, {"n_items": 0, "n_tokens": 0}

    device, G = A.device, A.shape[2]
    b_idx, w_idx, s_rows, tier1 = [], [], [], []
    for b, item in picks:
        for w in item["W"]:
            b_idx.append(b)
            w_idx.append(w)
            row = torch.zeros(G)
            row[list(item["S"])] = 1.0
            s_rows.append(row)
            tier1.append(bool(item["lat"]))

    b_idx = torch.as_tensor(b_idx, device=device)
    w_idx = torch.as_tensor(w_idx, device=device)
    S = torch.stack(s_rows).to(device=device, dtype=A.dtype)         # (n, G)
    is_t1 = torch.as_tensor(tier1, device=device, dtype=A.dtype)     # (n,)

    # (B, F, G, L) -> (n, F, G): one map per supervised (sample, text column) pair.
    a = A.permute(0, 3, 1, 2)[b_idx, w_idx]
    fm = frame_mask[b_idx].to(a.dtype)                               # (n, F)
    a = a * fm[:, :, None]

    denom = a.sum(dim=(1, 2)).clamp_min(1e-8)                        # (n,)
    mass_g = a.sum(dim=1) / denom[:, None]                           # (n, G), sums to 1
    m_S = (mass_g * S).sum(-1)                                       # (n,)

    loss_tok = (1.0 - m_S).pow(2)                                    # TokenCompose L_token

    m_mirror = torch.zeros_like(m_S)
    if lambda_mirror > 0.0:
        Mx = mirror_matrix() if mirror_mat is None else mirror_mat
        S_mirror = S @ Mx.to(device=device, dtype=A.dtype)           # (n, G)
        m_mirror = (mass_g * S_mirror).sum(-1)
        # Tier-1 only: a tier-2 item's S is already {left_X, right_X}, so its mirror is
        # itself and the margin would ask the map to beat its own mass — unsatisfiable.
        loss_tok = loss_tok + lambda_mirror * is_t1 * torch.relu(
            m_mirror - m_S + margin)

    # ── the tier-2 evenness term (the mirror term's twin) ─────────────────────────
    # L_token constrains the SUM over S and nothing else, so for a tier-2 item
    # S = {left_X, right_X} every split of that sum is an exact global optimum: 50/50
    # and 100/0 score identically. That is not a weak constraint, it is no constraint,
    # and gradient descent parks the split wherever the token's initialisation happened
    # to point and never corrects it. Measured consequence on `exp_smplh_verbs`: `raise`
    # leans left-arm and `kick` leans right-leg, in the SAME direction on 9/9 clips,
    # absent from the source clips' own energy and gone when the verb column is dropped
    # (docs/FINDINGS.md "Two mask defects with different causes").
    #
    # This does NOT teach a verb a side — that would break "verbs never lateralise" and
    # put supervision on both halves of the axis the mirror margin exists to sharpen. It
    # teaches it NO side, which is what that rule always meant; the rule was only ever
    # enforced by *omitting* a laterality term, and omitting a constraint on a free
    # parameter yields an arbitrary value, not a neutral one. The two terms are disjoint
    # by construction: mirror runs on tier 1, this runs on tier 2.
    split_max = torch.zeros_like(m_S)
    if lambda_even > 0.0:
        n_S = S.sum(-1).clamp_min(1.0)                               # (n,) = |S|
        p = mass_g * S                                               # (n, G), 0 off S
        # Deviation from an even split of whatever mass has arrived. Unnormalised on
        # purpose: dividing by m_S would blow up early in training when m_S ≈ 0, whereas
        # this scales WITH m_S — no pressure before the mass is there, full pressure once
        # L_token has done its job. Free annealing, no schedule.
        dev = (p - (m_S / n_S)[:, None]) * S                         # (n, G)
        # |S| = 1 makes this identically zero, so the tier-1 gate is belt-and-braces —
        # but it is the gate that states the intent, and it is what stops a future
        # multi-group LATERALISED item from being forced flat.
        loss_tok = loss_tok + lambda_even * (1.0 - is_t1) * dev.pow(2).sum(-1)
        # Monitor: the largest share of the on-target mass held by any one group.
        # 1/|S| is perfect (0.5 for a limb pair), 1.0 is "all of it on one side".
        split_max = p.max(dim=-1).values / m_S.clamp_min(1e-8)

    w = (torch.ones_like(m_S) if sample_weight is None
         else sample_weight[b_idx].to(a.dtype))
    loss = (w * loss_tok).mean()

    with torch.no_grad():
        n1 = is_t1.sum()
        stats = {
            "n_items":  len(picks),
            "n_tokens": int(m_S.numel()),
            "m_S":      m_S.mean().item(),
            "m_mirror": m_mirror.mean().item(),
        }
        if n1 > 0:
            # The tier-1 split is the one that answers the laterality question; tier 2
            # ({left_X, right_X}) is satisfiable without ever reading the side word.
            stats["m_S_tier1"] = (m_S * is_t1).sum().item() / n1.item()
        n2 = (1.0 - is_t1).sum()
        if n2 > 0 and lambda_even > 0.0:
            # Tier-2 items only — a tier-1 item's |S| is 1, so its split_max is 1.0 by
            # definition and averaging it in would hide the number this stat exists for.
            stats["split_max"] = (split_max * (1.0 - is_t1)).sum().item() / n2.item()
        if source_act is not None:
            valid_cells = frame_mask[b_idx][:, :, None].expand(-1, -1, G).reshape(
                len(b_idx), -1)
            r = _pearson(a.reshape(len(b_idx), -1).float(),
                         source_act[b_idx].reshape(len(b_idx), -1).float(),
                         valid_cells)
            stats["src_corr"] = r.mean().item()
    return loss, stats
