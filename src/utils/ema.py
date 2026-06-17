"""
Exponential Moving Average of model weights.

The EMA model is used for inference and evaluation. It is smoother than
the online model and typically gives better sample quality.

Usage:
    ema = EMA(model, decay=0.9999)
    # after each optimizer.step():
    ema.update_from(model)
    # for inference / evaluation, use the maintained shadow model directly:
    output = ema.ema_model(...)

LEDITS++ note: all inference passes (inversion + editing) use ema_model, loaded
from checkpoint_latest/ema.pt rather than model.pt.
"""

from copy import deepcopy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_from(self, online_model: nn.Module):
        for ema_p, online_p in zip(
            self.ema_model.parameters(), online_model.parameters()
        ):
            ema_p.data.lerp_(online_p.data, 1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)
