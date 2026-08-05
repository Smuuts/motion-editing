"""Diagnostic curves for the backbone prerequisite check (src/verify_backbone.py)."""

import matplotlib.pyplot as plt

from .heatmaps import save_figure


def plot_noise_level_sweep(results: dict, out_path: str):
    """results: {t: {"cond", "uncond", "noise_mse"}} → one-step reconstruction quality
    and noise MSE against the noise level."""
    ts = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ts, [results[t]["cond"]   for t in ts], "o-",  label="conditional (text)")
    ax.plot(ts, [results[t]["uncond"] for t in ts], "s--", label="unconditional (null)")
    ax.set_xlabel("Noise level  t")
    ax.set_ylabel("One-step x̂₀  MPJPE (m)")
    ax.set_title("One-step reconstruction quality vs. noise level")

    ax = axes[1]
    ax.plot(ts, [results[t]["noise_mse"] for t in ts], "o-", color="tab:green")
    ax.set_xlabel("Noise level  t")
    ax.set_ylabel("Noise prediction  MSE")
    ax.set_title("Noise MSE (conditional) vs. noise level")
    ax.axhline(1.0, color="red", linestyle=":", label="random baseline (MSE=1)")

    for ax in axes:
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=150, tight=False)
