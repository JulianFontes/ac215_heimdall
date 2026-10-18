"""figures for a trained Forecaster: training history, scores by lead time next to persistence, and
one test example as cloud-top-height maps."""
import matplotlib.pyplot as plt
import numpy as np

from utilities.plot_style import SKY_COLOR, styled

TOP_CMAP = plt.get_cmap("viridis").with_extremes(under=SKY_COLOR)


@styled
def plot_history(history):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(history["step"], history["train_loss"], "o-", label="Train")
    ax.plot(history["step"], history["val_loss"], "o-", label="Validation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (binary cross-entropy)")
    ax.legend()
    ax.grid(alpha=0.3, lw=0.5)
    return fig


@styled
def plot_scores(table):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (column, title) in zip(axes, [("csi", "Cloud voxel CSI (higher is better)"),
                                          ("prob_mae", "P(cloud) mean absolute error (lower is better)")]):
        for method, group in table.groupby("method"):
            ax.plot(group["lead_minutes"], group[column], "o-", label=method.capitalize())
        ax.set_xlabel("Lead time (min)")
        ax.set_title(title)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend()
    return fig


def cloud_top(frame, heights):
    """highest level with P(cloud) >= 0.5 (km), -1 where clear; frame (levels, H, W)."""
    top = np.where(frame >= 0.5, np.asarray(heights)[:, None, None], -1.0).max(axis=0)
    return top


@styled
def plot_example(x, target, forecast, heights, title=""):
    """rows: C3DIR (truth), ConvLSTM, persistence; columns: the last input, then each lead time."""
    rows = [("C3DIR", [x[-1], *target]), ("ConvLSTM", [x[-1], *forecast]), ("Persistence", [x[-1]] * (len(target) + 1))]
    fig, axes = plt.subplots(3, len(target) + 1, figsize=(3 * (len(target) + 1), 8.5), layout="constrained",
                             squeeze=False)
    for r, (name, frames) in enumerate(rows):
        for c, frame in enumerate(frames):
            image = axes[r, c].imshow(cloud_top(frame, heights), origin="lower", cmap=TOP_CMAP, vmin=0,
                                      vmax=max(heights))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title("Last input (t)" if c == 0 else f"t + {c * 10} min")
            if c == 0:
                axes[r, c].set_ylabel(name)
    fig.colorbar(image, ax=axes, label="Cloud top (km); blue = clear", shrink=0.7)
    fig.suptitle(title)
    return fig
