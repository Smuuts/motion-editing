"""
M1 read-outs: turning a stored cross-attention map into one value per (frame, group).

The raw read-out is sink-dominated. CrossAttention has no key-padding mask, so the
~L-|words| padding columns (zero T5 embeddings -> zero logits) plus EOS absorb most of
the softmax mass, and a per-cell content read-out is modulated by that sink denominator.
The alternatives here correct for it in two different ways:

  weight-only   "renorm" drops the sink columns and renormalises (Attend-and-Excite,
                Chefer et al. 2023); "spatial" reads each token's spatial profile
                instead of its mass (DAAM, Tang et al. 2023).
  contribution  "normw"/"normsum" read alpha*||v||, what actually reaches the residual
                stream (Kobayashi et al. 2020). That is a different ranking whenever
                value norms differ across columns — and they differ maximally here, the
                padding columns having value vectors of exactly zero.
"""

import torch

# Function words and generic caption vocabulary that carry no body-part/action
# semantics for mask purposes.
STOP_WORDS = {
    "a", "an", "the",
    "person", "man", "woman", "human", "someone",
    "their", "they", "them", "his", "her", "its",
    "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "by", "for", "with",
    "and", "or", "but",
    "up", "down", "forward", "backward", "side", "out",
    "slowly", "quickly", "slightly",
}

# Read-outs needing only the attention weights, and those additionally needing the value
# vectors. Kept as data so a caller can ask for "every read-out" in one sweep and the
# probe scripts can validate a --m1_readout choice without a second list.
WEIGHT_READOUTS = ("raw", "renorm", "spatial", "renorm_spatial")
VALUE_READOUTS = ("normw", "normsum")
ALL_READOUTS = WEIGHT_READOUTS + VALUE_READOUTS


def semantic_token_subset(idxs: list[int], labels: list[str]) -> list[int]:
    """Stop-word-filtered subset of content-token positions.

    Falls back to all content tokens when the filter would empty the set.
    """
    sem = [i for i, lbl in zip(idxs, labels) if lbl.lower() not in STOP_WORDS]
    return sem or list(idxs)


def _weight_readout(avg: torch.Tensor, tok: torch.Tensor, sem: torch.Tensor,
                    readout: str) -> torch.Tensor:
    """(N, L) layer/head-averaged attention -> (N,) per-cell M1 value.

    tok — all content-token columns (words; excludes BOS/EOS/padding).
    sem — stop-word-filtered subset of tok.

    "renorm" is only informative when sem is a PROPER subset of tok; otherwise the ratio
    is constant 1. `semantic_token_subset`'s fallback guarantees sem is non-empty, not
    that it is proper.
    """
    if readout == "raw":
        return avg[:, tok].mean(dim=-1)
    if readout == "renorm":
        return avg[:, sem].sum(dim=-1) / avg[:, tok].sum(dim=-1).clamp_min(1e-12)
    if readout == "spatial":
        cols = avg[:, sem]                                       # (N, S)
        cols = cols / cols.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return cols.mean(dim=-1)
    if readout == "renorm_spatial":
        rows = avg[:, sem] / avg[:, tok].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rows = rows / rows.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return rows.mean(dim=-1)
    raise ValueError(f"unknown attn_readout {readout!r}")


def _value_weighted_map(stacked: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """(Lyr, B, h, N, L) attention -> (N, L) map of CONTRIBUTION alpha*||v||.

    The PER-HEAD value norm is the right scale: attention is computed per head, and a
    head's output contribution from column w is alpha[h,.,w]*v[h,w], so weighting by a
    head-agnostic norm would mix scales across heads that need not be comparable.
    """
    vnorm = values.norm(dim=-1)                                  # (Lyr, B, h, L)
    return (stacked * vnorm[..., None, :]).mean(dim=(0, 1, 2))   # (N, L)


def _summed_contribution(stacked: torch.Tensor, values: torch.Tensor,
                         sem: torch.Tensor) -> torch.Tensor:
    """||sum_w alpha_w v_w|| over the semantic columns -> (N,): what each cell actually
    RECEIVED from the body-part words.

    Differs from the per-token form by accounting for CANCELLATION: two words whose value
    vectors point in opposite directions each contribute a large alpha*||v|| while summing
    to nearly nothing. "left" vs "right" is exactly the pair one expects to be
    near-antiparallel, so this is the form in which a laterality signal could survive when
    the per-token form shows none.
    """
    a = stacked[..., sem]                                        # (Lyr, B, h, N, S)
    v = values[:, :, :, sem, :]                                  # (Lyr, B, h, S, hd)
    contrib = a @ v                                              # (Lyr, B, h, N, hd)
    contrib = contrib.permute(0, 1, 3, 2, 4).flatten(-2)         # (Lyr, B, N, h*hd)
    return contrib.norm(dim=-1).mean(dim=(0, 1))                 # (N,)


def step_readouts(stacked: torch.Tensor, values, tok: torch.Tensor, sem: torch.Tensor,
                  readouts) -> dict[str, torch.Tensor]:
    """One timestep's (Lyr, B, h, N, L) attention -> {readout: (N,) per-cell value}.

    All requested read-outs come from the SAME stored tensors, so a probe comparing them
    is comparing read-outs and nothing else — no second inversion, no run-to-run spread
    inside the comparison.
    """
    out, avg = {}, None
    for r in readouts:
        if r in WEIGHT_READOUTS:
            if avg is None:
                avg = stacked.mean(dim=(0, 1, 2))                # (N, L)
            out[r] = _weight_readout(avg, tok, sem, r)
        elif r == "normw":
            out[r] = _weight_readout(_value_weighted_map(stacked, values),
                                     tok, sem, "raw")
        elif r == "normsum":
            out[r] = _summed_contribution(stacked, values, sem)
        else:
            raise ValueError(f"unknown attn_readout {r!r}")
    return out


def column_class_stats(stacked: torch.Tensor, values: torch.Tensor,
                       tok: torch.Tensor) -> dict[str, float]:
    """Attention mass and mean value norm, split by column class.

    Settles whether `renorm` is right to drop the EOS/sink column as a distractor, which
    it only is if that column contributes little. Pads are the calibration point: their
    value vectors are exactly zero, so any mass on them is contribution-free by
    construction.

    EOS is the column immediately after the last content token — T5 emits
    [tokens..., EOS, pad...] and `token_info` returns the content columns only.
    """
    avg = stacked.mean(dim=(0, 1, 2))                            # (N, L)
    vnorm = values.norm(dim=-1).mean(dim=(0, 1, 2))              # (L,)
    L = avg.shape[-1]
    eos = int(tok.max().item()) + 1
    pad = torch.arange(eos + 1, L, device=avg.device)
    out = {
        "mass_content": float(avg[:, tok].sum(dim=-1).mean()),
        "vnorm_content": float(vnorm[tok].mean()),
        "row_total": float(avg.sum(dim=-1).mean()),
    }
    if eos < L:
        out["mass_eos"] = float(avg[:, eos].mean())
        out["vnorm_eos"] = float(vnorm[eos])
    if pad.numel():
        out["mass_pad"] = float(avg[:, pad].sum(dim=-1).mean())
        out["vnorm_pad"] = float(vnorm[pad].mean())
    return out
