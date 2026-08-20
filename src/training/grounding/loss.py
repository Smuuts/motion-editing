"""
TokenCompose's L_token, the mirror margin, the tier-2 evenness term, and the monitors.

For each body-part mention in a caption, the text columns that spell it out must place
their cross-attention mass on that mention's body-part group tokens:

    A_w      = A[b, :, :, w] / sum_{f valid, g} A[b, f, g, w]   distribution over cells
    m_S      = sum_{f valid, g in S} A_w[f, g]                  mass on the target groups
    L_group  = (1 - m_S)^2                                      TokenCompose's L_token
    L_mirror = relu(m_S' - m_S + margin)      S' = MIRROR[S], tier 1 only

THE FAILURE MODE THIS CODE IS BUILT TO EXPOSE. Captions describe their clips, so a label
correlates with the source clip's dynamics: on tier-1 items, argmax(source energy)
already equals the label 69.5 % of the time. A model that learns "attend to whatever
moves" scores well on L_group without learning any word->group routing at all. Three
things push back — 36.9 % of captions carry >=2 items with DIFFERENT targets (a
token-invariant detector cannot fit those), the mirror term, and the `1 - alpha_bar_t`
timestep weighting that concentrates pressure at high noise where there is no motion left
to detect — and `src_corr` in the returned stats MEASURES whether they worked. Kill
criterion: src_corr above ~0.5 and rising while m_S rises.
"""

import torch

from .spec import mirror_matrix

def batched_source_activity(motion: torch.Tensor, group_channels,
                            frame_mask: torch.Tensor) -> torch.Tensor:
    """(B, F, G) per-(frame, group) |Δx0| of the SOURCE clip — the batched twin of
    utils.probe.source_activity, which is single-clip and numpy.

    This is the reference the shortcut monitor correlates against: it depends only on
    the clip, never on the caption, so an attention map that tracks it is a motion
    detector rather than a word→group router.

    Frame 0 repeats frame 1 rather than being zeroed (2026-08-16), matching
    `utils.probe.source_activity` and `editing.masking._frame_energy` — the three are one
    definition and are changed together. This shifts the logged
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
    rather than a visible error.
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
    b_idx, w_idx, s_rows, tier1, st_rows = [], [], [], [], []
    any_st = False
    for b, item in picks:
        st = item.get("M")          # optional (F, G) spatiotemporal target; see below
        any_st = any_st or st is not None
        for w in item["W"]:
            b_idx.append(b)
            w_idx.append(w)
            row = torch.zeros(G)
            row[list(item["S"])] = 1.0
            s_rows.append(row)
            tier1.append(bool(item["lat"]))
            st_rows.append(st)

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

    # L_token's target is normally the group set S, applied at every frame. An item may
    # instead carry "M", a binary (F, G) REGION — then the token is asked for its mass
    # inside that region rather than inside those group rows. The group form is exactly
    # the special case M[f, g] = S[g], so this generalises rather than branches:
    #
    #     m = sum_{f,g} (a[f,g] / denom) * M[f,g]        with M = S  =>  m == m_S
    #
    # verified bit-identical when no item supplies M, which is why the fast path below is
    # kept: it avoids building an (n, F, G) tensor for the overwhelmingly common case.
    # NOTE M must be BINARY. A soft target caps m below 1 and leaves (1 - m)^2 with an
    # irreducible floor, i.e. permanent gradient toward an unreachable optimum.
    if any_st:
        Fdim = a.shape[1]
        M = S[:, None, :].expand(-1, Fdim, -1).clone()               # (n, F, G) default
        for i, st in enumerate(st_rows):
            if st is None:
                continue
            t = torch.as_tensor(st, device=device, dtype=A.dtype)
            f = min(t.shape[0], Fdim)
            M[i, :f], M[i, f:] = t[:f], 0.0
        m_tok = ((a / denom[:, None, None]) * M).sum(dim=(1, 2))     # (n,)
    else:
        m_tok = m_S

    loss_tok = (1.0 - m_tok).pow(2)                                  # TokenCompose L_token

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
    # (the same cell-budget defect that motivated `rank_group_select`).
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
            "m_tok":    m_tok.mean().item(),
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
