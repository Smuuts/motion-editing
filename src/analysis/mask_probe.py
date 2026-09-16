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
from training.grounding import resolve_readout_columns


def collect_instruction_masks(
    model, schedule, editor, state, text_encoder, instructions, valid_frames,
    is_group, *, mask_modes=("m2_only",), lambda_attn=70.0, lambda_noise=70.0,
    attn_readout="raw", sweeps=(None, None, None), per_step_norm=False,
    context_ref=None, psi_group_norm=False, stats_out=None, psi_readout=None,
    column_mode="content", config=None, group_mode="parts", columns_out=None,
    m1_select="percentile", m1_rank_ratio=0.5, m1_rank_max=3,
):
    """-> (m1_maps, m2_maps, {mode: [binary (F, G) maps]}), all numpy, one per instruction.

    `sweeps` is (shared_ts, m1_ts, m2_ts) from utils.probe.resolve_sweeps.
    M1 is always collected (need_attn=True) so the raw map is available for the figures
    regardless of which mask_mode drives the binary mask.

    `attn_readout` may be a SEQUENCE of readout names, in which case m1_maps and
    `binaries` become {readout: ...} dicts and every readout is computed from the one
    inversion already in `state` — the comparison is then between readouts and nothing
    else. `stats_out`, if given, receives the per-column-class attention mass / value
    norm diagnostic keyed by instruction.

    `psi_readout` selects what the ψ contrast measures ("abs" or the signed "energy";
    see masking.collect_statistics). ONE name only here, and None means "whatever the
    editor was built with" — to compare ψ read-outs off a single sweep, call
    `collect_statistics` directly with a sequence, the way `probe_psi_sign.py` does.

    `column_mode` / `config` / `group_mode` choose WHICH TEXT COLUMNS M1 reads, via
    `training.grounding.resolve_readout_columns` — "content" (every content token, the
    historical read) or "span" (only the supervised body-part words). "auto" resolves it
    from the checkpoint like `--m1_layers` does. `columns_out`, if given, is filled with
    {instruction: (mode, columns)} so a caller can report what was actually read.

    ψ/M2 is read in `editor.edit_space` — so on an x0 checkpoint the probe measures the
    x0-native ψ the editor now actually uses, not the ε-space shim. M2 numbers are
    therefore only comparable across checkpoints trained with the same predict_type;
    to reproduce a pre-2026-08-05 M2 measurement on an x0 checkpoint, build the editor
    with edit_space="eps".
    """
    shared_ts, m1_ts, m2_ts = sweeps
    single = isinstance(attn_readout, str)
    readouts = (attn_readout,) if single else tuple(attn_readout)
    with torch.no_grad():
        ctxs = list(text_encoder.encode(instructions).split(1, dim=0))
        # (token_idxs, semantic_idxs) per instruction — "content" reproduces the
        # historical read exactly, so this is a no-op unless a caller asks for "span".
        cols = [resolve_readout_columns(e, text_encoder, config, column_mode, group_mode)
                for e in instructions]
    if columns_out is not None:
        columns_out.update({e: (c[2], c[0]) for e, c in zip(instructions, cols)})

    m1_maps = {r: [] for r in readouts}
    m2_maps = []
    for instr, ctx, (tok, sem, _) in zip(instructions, ctxs, cols):
        per_instr_stats = {} if stats_out is not None else None
        attn_fg, psi_fg = masking.collect_statistics(
            model, schedule, state.xs, ctx, tok,
            is_group=is_group, timesteps=shared_ts, need_attn=True,
            group_channels=editor.group_channels, valid_frames=valid_frames,
            attn_readout=readouts, semantic_idxs=sem,
            attn_timesteps=m1_ts, psi_timesteps=m2_ts, per_step_norm=per_step_norm,
            context_ref=context_ref, psi_group_norm=psi_group_norm,
            psi_space=editor.edit_space, stats_out=per_instr_stats,
            attn_layers=editor.attn_layers,
            psi_readout=psi_readout or getattr(editor, "psi_readout", "energy"),
        )
        for r in readouts:
            m1_maps[r].append(attn_fg[r])
        m2_maps.append(psi_fg)
        if per_instr_stats:
            stats_out[instr] = per_instr_stats

    def _binaries(maps):
        return {
            mode: [
                masking.build_mask(
                    a, p, valid_frames, is_group,
                    lambda_attn=lambda_attn, lambda_noise=lambda_noise, mask_mode=mode,
                    group_channels=editor.group_channels, feat_dim=editor.feat_dim,
                    m1_select=m1_select, m1_rank_ratio=m1_rank_ratio,
                    m1_rank_max=m1_rank_max,
                )["m_group"].float().cpu().numpy()
                for a, p in zip(maps, m2_maps)
            ]
            for mode in mask_modes
        }

    binaries = {r: _binaries(m1_maps[r]) for r in readouts}
    np_m1 = {r: [m.cpu().numpy() for m in maps] for r, maps in m1_maps.items()}
    np_m2 = [m.cpu().numpy() for m in m2_maps]
    if single:
        return np_m1[readouts[0]], np_m2, binaries[readouts[0]]
    return np_m1, np_m2, binaries


def active_cells(binary_map) -> tuple[int, int]:
    """(active (frame, group) cells, frames touched) of a binary mask — the one-line
    summary printed per instruction."""
    m = np.asarray(binary_map)
    return int(m.sum()), int((m.sum(axis=1) > 0).sum())


def collect_noise_band_maps(model, schedule, editor, state, text_encoder, instruction,
                            valid_frames, is_group, bands, *, band_timesteps=16,
                            attn_readout="raw", per_step_norm=False,
                            column_mode="content", config=None, group_mode="parts",
                            psi_readout=None, columns_out=None):
    """-> (m1_bands, m2_bands): one raw (F, G) numpy map per noise band, for ONE instruction.

    The band twin of `collect_instruction_masks`: that one holds the timestep sweep fixed
    and varies the instruction, this one holds the instruction fixed and varies the
    *noise level*. Nothing is thresholded — a binary mask is a property of the whole
    sweep, and asking what one band's mask would look like on its own would answer a
    question nobody is going to run.

    `bands` is [(lo, hi), ...] from `utils.probe.resolve_bands`; each is swept with
    `band_timesteps` steps RESAMPLED INSIDE it (`masking.build_sweep`), so a narrow
    low-t band gets the same number of samples as a wide one and the rows are averages
    over an equal number of forward passes rather than over an equal width of t.

    Every band reads the SAME stored inversion in `state`, so the rows differ only by
    noise level. Re-inverting per band would put the inversion's own ±0.02 run-to-run
    spread inside the comparison the figure exists to make.
    """
    with torch.no_grad():
        ctx = text_encoder.encode([instruction])
        tok, sem, mode = resolve_readout_columns(
            instruction, text_encoder, config, column_mode, group_mode)
    if columns_out is not None:
        columns_out[instruction] = (mode, tok)

    m1_bands, m2_bands = [], []
    for lo, hi in bands:
        attn_fg, psi_fg = masking.collect_statistics(
            model, schedule, state.xs, ctx, tok, is_group=is_group,
            timesteps=masking.build_sweep(band_timesteps, schedule.T, lo, hi),
            need_attn=True, group_channels=editor.group_channels,
            valid_frames=valid_frames, attn_readout=attn_readout, semantic_idxs=sem,
            per_step_norm=per_step_norm, psi_space=editor.edit_space,
            attn_layers=editor.attn_layers,
            psi_readout=psi_readout or getattr(editor, "psi_readout", "energy"))
        m1_bands.append(attn_fg.cpu().numpy())
        m2_bands.append(psi_fg.cpu().numpy())
    return m1_bands, m2_bands
