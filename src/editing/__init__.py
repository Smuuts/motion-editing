"""
LEDITS++ motion-editing pipeline.

  inversion.MotionEditor  Stage 1 (edit-friendly inversion) + Stage 3 (masked SEGA
                          denoising with hard inpainting)
  masking/                Stage 2 (spatiotemporal M1 ∩ M2 mask construction)
  sega.py                 Stage 3's guidance estimate and its eps-space gate
  seeding.py              per-item inversion seeds

Typical flow (one source motion x0, one or more edit instructions):

    editor = MotionEditor(ema_model, schedule, device, is_group=True)
    state  = editor.invert(x0)                                     # Stage 1
    ctxs   = [text_encoder.encode([e]) for e in edits]
    toks   = [text_encoder.token_info(e)[0] for e in edits]
    masks  = editor.collect_masks(state, ctxs, toks, valid_frames)  # Stage 2
    x_edit = editor.edit(state, ctxs, masks, scales=[5.0])          # Stage 3
"""

from editing import masking
from editing.inversion import InversionState, MotionEditor
from editing.seeding import derive_seed

__all__ = ["InversionState", "MotionEditor", "derive_seed", "masking"]
