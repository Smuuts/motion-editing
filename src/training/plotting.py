"""Loss-curve plotting for train.py."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(path, title, ylabel, *series):
    """series: (epochs, values, plot_kwargs) tuples."""
    if not series or not series[0][0]:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for epochs, values, kwargs in series:
        ax.plot(epochs, values, **kwargs)
    if len(series) > 1:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_graph(output_dir, train_losses, val_losses=None):
    series = [
        ([e for e, _ in train_losses], [v for _, v in train_losses],
         dict(marker="o", linestyle="-", color="tab:blue", label="train")),
    ]
    if val_losses:
        series.append(
            ([e for e, _ in val_losses], [v for _, v in val_losses],
             dict(marker="s", linestyle="--", color="tab:orange", label="val"))
        )
    _save_figure(os.path.join(output_dir, "training_loss.png"), "Loss per Epoch", "Average Loss", *series)
