"""figures for a ConvLSTM forecaster: training history, scores by lead time next to persistence,
and an example forecast as maps of cloud-top height."""
import matplotlib.pyplot as plt
import numpy as np

from heimdall.c3dir.plots.model_plots import plot_history  # noqa: F401 - same figure, re-exported
from heimdall.plot_style import NO_DATA_COLOR, SKY_COLOR, styled

TOP_CMAP = plt.get_cmap("viridis").with_extremes(under=SKY_COLOR, bad=NO_DATA_COLOR)


@styled
def plot_scores(table):
    """CSI, cloud-cover accuracy and probability error against lead time, model vs persistence."""
    panels = [("csi", "Cloud voxel CSI"), ("cover_accuracy", "Cloud cover accuracy"), ("prob_mae", "P(cloud) MAE")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), layout="constrained")
    for ax, (column, title) in zip(axes, panels):
        for method, group in table.groupby("method"):
            ax.plot(group["lead_minutes"], group[column], "o-", label=method.capitalize())
        ax.set_xlabel("Lead time (min)")
        ax.set_title(title)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend()
    return fig


def cloud_top(frame, heights):
    """highest level with P(cloud) >= 0.5 (km), nan where clear; frame (C, H, W)."""
    prob = frame[: len(heights)]
    cloudy = prob >= 0.5
    top = np.where(cloudy, np.asarray(heights)[:, None, None], -np.inf).max(axis=0)
    return np.where(np.isfinite(top), top, np.nan)


@styled
def plot_example(x, target, forecast, heights, valid, times=None, title=""):
    """rows: truth, ConvLSTM, persistence; columns: last input then each lead time. shows cloud-top
    height; x (T, C, H, W), target / forecast (S, C, H, W)."""
    steps = len(target)
    rows = [("C3DIR (truth)", [x[-1], *target]), ("ConvLSTM", [x[-1], *forecast]),
            ("Persistence", [x[-1]] * (steps + 1))]
    fig, axes = plt.subplots(3, steps + 1, figsize=(3 * (steps + 1), 8.5), layout="constrained", squeeze=False)
    for r, (name, frames) in enumerate(rows):
        for c, frame in enumerate(frames):
            top = cloud_top(frame, heights)
            top = np.where(valid, np.nan_to_num(top, nan=-1.0), np.nan)
            image = axes[r, c].imshow(top, origin="lower", cmap=TOP_CMAP, vmin=0, vmax=max(heights))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                label = "Last input" if c == 0 else f"t+{c}"
                if times is not None:
                    label += f"\n{times[c]:%m-%d %H:%M}"
                axes[r, c].set_title(label)
            if c == 0:
                axes[r, c].set_ylabel(name)
    fig.colorbar(image, ax=axes, label="Cloud top (km); blue = clear", shrink=0.7)
    fig.suptitle(title)
    return fig
