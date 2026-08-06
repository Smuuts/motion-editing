"""
Run one source clip through a set of contrasting instructions and collect the implicit
masks — the measurement behind visualise_mask_problem.py and probe_mask_axes.py.

It goes through the real editing stack (MotionEditor inversion +
masking.collect_statistics/build_mask), so whatever it reports is what the editor would
actually use, on any checkpoint the editor runs on.
"""

import numpy as np
import torch

from editing import masking
from editing.masking import semantic_token_subset


def collect_instruction_masks(
    model, schedule, editor, state, text_encoder, instructions, valid_frames,
    is_group, *, mask_modes=("m2_only",), lambda_attn=70.0, lambda_noise=70.0,
    attn_readout="raw", sweeps=(None, None, None), per_step_norm=False,
    context_ref=None, psi_group_norm=False,
):
    """-> (m1_maps, m2_maps, {mode: [binary (F, G) maps]}), all numpy, one per instruction.

    `sweeps` is (shared_ts, m1_ts, m2_ts) from utils.probe.resolve_sweeps.
    M1 is always collected (need_attn=True) so the raw map is available for the figures
    regardless of which mask_mode drives the binary mask.

    ψ/M2 is read in `editor.edit_space` — so on an x0 checkpoint the probe measures the
    x0-native ψ the editor now actually uses, not the ε-space shim. M2 numbers are
    therefore only comparable across checkpoints trained with the same predict_type;
    to reproduce a pre-2026-08-05 M2 measurement on an x0 checkpoint, build the editor
    with edit_space="eps".
    """
    shared_ts, m1_ts, m2_ts = sweeps
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
        tok_info = [text_encoder.token_info(e) for e in instructions]

    m1_maps, m2_maps = [], []
    for ctx, ti in zip(ctxs, tok_info):
        attn_fg, psi_fg = masking.collect_statistics(
            model, schedule, state.xs, ctx, ti[0],
            is_group=is_group, timesteps=shared_ts, need_attn=True,
            group_channels=editor.group_channels, valid_frames=valid_frames,
            attn_readout=attn_readout, semantic_idxs=semantic_token_subset(*ti),
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=per_step_norm,
            context_ref=context_ref, psi_group_norm=psi_group_norm,
            psi_space=editor.edit_space,
        )
        m1_maps.append(attn_fg)
        m2_maps.append(psi_fg)

    binaries = {
        mode: [
            masking.build_mask(
                a, p, valid_frames, is_group,
                lambda_attn=lambda_attn, lambda_noise=lambda_noise, mask_mode=mode,
                group_channels=editor.group_channels, feat_dim=editor.feat_dim,
            )["m_group"].float().cpu().numpy()
            for a, p in zip(m1_maps, m2_maps)
        ]
        for mode in mask_modes
    }
    return ([m.cpu().numpy() for m in m1_maps],
            [m.cpu().numpy() for m in m2_maps],
            binaries)


def active_cells(binary_map) -> tuple[int, int]:
    """(active (frame, group) cells, frames touched) of a binary mask — the one-line
    summary printed per instruction."""
    m = np.asarray(binary_map)
    return int(m.sum()), int((m.sum(axis=1) > 0).sum())
