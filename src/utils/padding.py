"""Padding-mask helpers: sequence lengths -> per-frame boolean masks."""

import torch


def length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) integer sequence lengths -> (B, max_len) bool mask, True = real (non-padding) frame.

    lengths must already be on the target device; the returned mask lives on that
    same device.
    """
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
