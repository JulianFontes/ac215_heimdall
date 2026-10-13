"""Training curves, best model vs persistence, and the height-longitude forecast comparison."""
import matplotlib
matplotlib.use("Agg")                     # write PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from config import CLOUD_PROPERTIES

INK, MUTED, GRID = "#1f1f1e", "#6f6e69", "#e4e3dd"
COLORS = dict(zip(CLOUD_PROPERTIES, ["#2a78d6", "#eb6834", "#1baf7a"]))      # one colour per cloud property
CLOUD_CMAP = LinearSegmentedColormap.from_list("cloud", ["#a1d1e4", "#ffffff"])

# one look for every plot, applied once when this file is imported
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "grid.color": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

def plot_curves(hist, base, path):
    """Left: train / validation loss. Right: validation CSI per cloud property, persistence as dashed lines."""
    fig, (loss, csi) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    loss.plot(hist.epoch, hist.train_loss, color=MUTED, lw=2, label="train")
    loss.plot(hist.epoch, hist.val_loss, color=INK, lw=2, label="validation")
    loss.set(title="Train / validation loss", xlabel="epoch", ylabel="weighted BCE")
    loss.legend(frameon=False)

    for prop in CLOUD_PROPERTIES:
        csi.plot(hist.epoch, hist[f"val_CSI_{prop}"], color=COLORS[prop], lw=2, label=prop)
        csi.axhline(base[prop]["CSI"], color=COLORS[prop], lw=1.2, ls="--")   # persistence does not change with training
    csi.plot([], [], color=MUTED, lw=1.2, ls="--", label="persistence")       # legend entry for the dashed lines
    csi.set(title="Validation CSI vs persistence", xlabel="epoch", ylabel="CSI", ylim=(0, 1))
    csi.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)

    best = hist.loc[hist.val_loss.idxmin(), "epoch"]                           # epoch of the saved checkpoint
    for ax in (loss, csi):
        ax.grid(axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(best, color=MUTED, lw=1, ls=":", zorder=0)
        ax.text(best, 1, " best", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_persistence(final, base, path):
    """Validation CSI per cloud property: best model next to persistence."""
    x, width = np.arange(len(CLOUD_PROPERTIES)), 0.38
    model = [final[prop]["CSI"] for prop in CLOUD_PROPERTIES]
    persistence = [base[prop]["CSI"] for prop in CLOUD_PROPERTIES]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for offset, values, color, label in ((-width / 2, model, "#2a78d6", "ConvLSTM3d (best)"),
                                         (width / 2, persistence, "#b4b2a9", "persistence")):
        bars = ax.bar(x + offset, values, width, color=color, edgecolor="white", lw=2, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK)
    ax.set_xticks(x, CLOUD_PROPERTIES)
    ax.set(title="Validation CSI: best model vs persistence", ylabel="CSI", ylim=(0, 1.08))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=150)
    plt.close(fig)

def lat_mean(a, valid):
    """Average over latitude (axis -2), counting only bins inside the ERA5 column. a: (..., alt, lat, lon)."""
    n = valid.sum(axis=-2)
    return np.where(n > 0, (a * valid).sum(axis=-2) / np.maximum(n, 1), np.nan)

def plot_height_lon_compare(obs, prob, valid, alt, lon, times, path):
    """Height-longitude sections over time (latitude averaged): actual, ConvLSTM and persistence.

    obs:   (N, 3, alt, lat, lon) observed masks for N consecutive hours
    prob:  (K, 3, alt, lat, lon) one-hour-ahead forecast probability for the last K of those hours
    valid: (N, alt, lat, lon)    1 where the bin is inside the ERA5 column

    Three rows per cloud property: actual (all N hours), then ConvLSTM and persistence (only the last K hours, boxed).
    Persistence for an hour is the observed mask of the hour before it. Every panel is a share between 0 and 1:
    the share of latitudes with occurrence (actual, persistence) or the averaged probability (ConvLSTM).
    """
    n, k = len(obs), len(prob)
    actual = lat_mean(obs, valid[:, None])                  # (N, 3, alt, lon)
    model = lat_mean(prob, valid[n - k:, None])             # (K, 3, alt, lon)
    persistence = actual[n - k - 1:n - 1]                   # (K, 3, alt, lon): the hour before each forecast hour
    first = n - k                                           # first forecast column

    fig, axes = plt.subplots(3 * len(CLOUD_PROPERTIES), n, figsize=(1.8 * n + 1, 4.5 * len(CLOUD_PROPERTIES) + 2), layout="constrained", squeeze=False)

    def draw(ax, field):
        return ax.pcolormesh(lon, alt, field, cmap=CLOUD_CMAP, vmin=0, vmax=1, shading="auto")

    for r, prop in enumerate(CLOUD_PROPERTIES):
        top = 3 * r                                         # rows: top = actual, top + 1 = ConvLSTM, top + 2 = persistence
        for t in range(n):
            im = draw(axes[top, t], actual[t, r])
        for j in range(k):
            draw(axes[top + 1, first + j], model[j, r])
            draw(axes[top + 2, first + j], persistence[j, r])
        for ax in axes[top + 1:top + 3, :first].ravel():
            ax.set_axis_off()                               # ConvLSTM and persistence only exist for forecast hours

        axes[top, 0].set_ylabel(f"{prop}\naltitude (km)", fontsize=10)
        for row, name in enumerate(("actual", "ConvLSTM", "persistence")):   # name the rows in the boxed columns
            axes[top + row, first].set_ylabel(f"{prop} {name}", fontsize=9)
            axes[top + row, first].tick_params(labelleft=True)

    for t in range(n):
        axes[0, t].set_title(times[t].strftime("%m-%d %H:%M"), fontsize=9)
        bottom = axes[-1 if t >= first else -3, t]          # lowest panel that exists in this column
        bottom.set_xlabel("longitude", fontsize=8)
        bottom.tick_params(labelbottom=True)

    fig.colorbar(im, ax=axes, location="bottom", shrink=0.4, aspect=40, label="Proportion of latitudes with clouds / forecast probability")
    fig.suptitle(f"{first} observed hours, then {k} forecasted hours (boxed)", fontsize=14)

    # dashed box around the forecast columns (their positions are only final after a draw)
    fig.canvas.draw()
    box = Bbox.union([ax.get_tightbbox() for ax in axes[:, first:].ravel()]).transformed(fig.transFigure.inverted())
    pad = 0.004
    fig.add_artist(Rectangle((box.x0 - pad, box.y0 - pad), box.width + 2 * pad, box.height + 2 * pad,
                             transform=fig.transFigure, fill=False, ec=INK, lw=1.5, ls="--"))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1)   # tight: keeps the colorbar label from clipping
    plt.close(fig)