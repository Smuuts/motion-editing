"""Optimiser and LR schedule, shared by train.py and overfit_one.py so the two can't
drift apart (overfit_one exists precisely to reproduce training conditions)."""

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def build_optimizer(model, lr, weight_decay):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer, total, warmup, decay=True):
    """Linear warmup (1% → 100% of lr over `warmup` steps) then cosine decay to 1e-6.

    `total` and `warmup` are epochs in train.py and optimiser steps in overfit_one.py;
    the schedule only cares that .step() is called once per unit.
    """
    main = (CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-6)
            if decay else LambdaLR(optimizer, lr_lambda=lambda _: 1.0))
    if warmup <= 0:
        return main
    warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])
