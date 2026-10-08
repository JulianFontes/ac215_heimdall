"""Training curves, the best-model-vs-persistence bar chart, and height-longitude forecast sections."""
import matplotlib
matplotlib.use("Agg")                     # write PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from config import CLOUD_PROPERTIES

COLORS = dict(zip(CLOUD_PROPERTIES, ["#2a78d6", "#eb6834", "#1baf7a"]))   # one fixed colour per species
MODEL_C, BASE_C = "#2a78d6", "#b4b2a9"                             # model vs persistence bars
INK, MUTED, GRID = "#1f1f1e", "#6f6e69", "#e4e3dd"

# forecast sections: one blue ramp for occurrence and probability (0 = surface, 1 = dark blue)
SURFACE = "#f4f3ee"
PROB_CMAP = LinearSegmentedColormap.from_list("prob", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    if xlabel == "epoch":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_curves(hist, base, path):
    """Left: train / validation loss. Right: validation CSI per species, persistence as dashed lines."""
    best = hist.loc[hist.val_loss.idxmin(), "epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    a1.plot(hist.epoch, hist.train_loss, color=MUTED, lw=2, label="train")
    a1.plot(hist.epoch, hist.val_loss, color=INK, lw=2, label="validation")
    _style(a1, "Train / validation loss", "epoch", "weighted BCE")
    a1.legend(frameon=False)

    for s in CLOUD_PROPERTIES:
        a2.plot(hist.epoch, hist[f"val_CSI_{s}"], color=COLORS[s], lw=2, label=s)
        a2.axhline(base[s]["CSI"], color=COLORS[s], lw=1.2, ls="--")          # persistence does not change with training
    a2.plot([], [], color=MUTED, lw=1.2, ls="--", label="persistence")        # legend entry for the dashed lines
    _style(a2, "Validation CSI vs persistence", "epoch", "CSI")
    a2.set_ylim(0, 1)
    a2.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)

    for ax in (a1, a2):   # epoch of the saved checkpoint
        ax.axvline(best, color=MUTED, lw=1, ls=":", zorder=0)
        ax.text(best, 1, " best", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_vs_persistence(final, base, path):
    """Validation CSI per species: best model next to persistence."""
    xs, w = np.arange(len(CLOUD_PROPERTIES)), 0.38
    m = np.array([final[s]["CSI"] for s in CLOUD_PROPERTIES])
    p = np.array([base[s]["CSI"] for s in CLOUD_PROPERTIES])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for off, vals, c, lab in ((-w / 2, m, MODEL_C, "ConvLSTM3d (best)"), (w / 2, p, BASE_C, "persistence")):
        bars = ax.bar(xs + off, vals, w, color=c, edgecolor="white", lw=2, label=lab)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK)
    ax.set_xticks(xs, CLOUD_PROPERTIES)
    ax.set_ylim(0, 1.08)
    _style(ax, "Validation CSI: best model vs persistence", "", "CSI")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=150)
    plt.close(fig)

def lat_mean(a, valid):
    """Average over latitude (axis -2), counting only bins inside the ERA5 column. a: (..., alt, lat, lon)."""
    n = valid.sum(axis=-2)
    return np.where(n > 0, (a * valid).sum(axis=-2) / np.maximum(n, 1), np.nan)

def plot_height_lon_sequence(obs, prob, valid, alt, lon, times, path, level=0.3):
    """Height-longitude sections over time (latitude averaged). Two rows per species: model, then persistence.

    obs:   (N, 3, alt, lat, lon) observed masks for N consecutive hours
    prob:  (K, 3, alt, lat, lon) one-hour-ahead forecast probability for the last K of those hours
    valid: (N, alt, lat, lon)    1 where the bin is inside the ERA5 column
    The first N - K columns show the observed hours. The last K (boxed) show the model forecast (top row) and the
    persistence forecast (bottom row, = the previous observed hour), with the observed occurrence outlined where
    the share of latitudes reaches `level`.
    """
    n, k = len(obs), len(prob)
    obs_c, fc_c = lat_mean(obs, valid[:, None]), lat_mean(prob, valid[n - k:, None])    # (hours, 3, alt, lon)
    base_c = obs_c[n - k - 1:n - 1]                                      # persistence: previous observed hour

    rows = 2 * len(CLOUD_PROPERTIES)
    fig, axes = plt.subplots(rows, n, figsize=(1.8 * n + 1, 1.5 * rows + 1.6), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for r, s in enumerate(CLOUD_PROPERTIES):
        for t in range(n):
            top, bottom = axes[2 * r, t], axes[2 * r + 1, t]
            if t < n - k:
                im = top.pcolormesh(lon, alt, obs_c[t, r], cmap=PROB_CMAP, vmin=0, vmax=1, shading="auto")
                bottom.set_axis_off()                                    # persistence only exists for forecast hours
            else:
                j = t - (n - k)
                for ax, field in ((top, fc_c[j, r]), (bottom, base_c[j, r])):
                    ax.pcolormesh(lon, alt, field, cmap=PROB_CMAP, vmin=0, vmax=1, shading="auto")
                    ax.contour(lon, alt, np.nan_to_num(obs_c[t, r]), levels=[level], colors=INK, linewidths=0.9)
            for ax in (top, bottom):
                ax.set_facecolor("white")                                # NaN (outside data) shows as white
                ax.tick_params(colors=MUTED, labelsize=7)
                for side in ax.spines.values():
                    side.set_color(GRID)
        top, bottom = axes[2 * r, n - k], axes[2 * r + 1, n - k]         # label the rows inside the boxed columns
        for ax, name in ((top, "forecast"), (bottom, "persistence")):
            ax.text(0.04, 0.95, name, transform=ax.transAxes, va="top", fontsize=8, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        axes[2 * r, 0].set_ylabel(f"{s}\naltitude (km)", color=INK, fontsize=10)
        bottom.set_ylabel(f"{s}\naltitude (km)", color=INK, fontsize=10)
        bottom.tick_params(labelleft=True)
    for t in range(n):
        axes[0, t].set_title(times[t].strftime("%m-%d %H:%M"), fontsize=9, color=INK)
        last = axes[-1, t] if t >= n - k else axes[-2, t]                # lowest panel that exists in this column
        last.set_xlabel("longitude", color=MUTED, fontsize=8)
        last.tick_params(labelbottom=True)

    fig.colorbar(im, ax=axes, location="bottom", shrink=0.4, aspect=40,
                 label="share of latitudes with occurrence / forecast probability")
    fig.suptitle(f"ConvLSTM3d vs persistence, one-hour-ahead: {n - k} observed hours, then {k} forecasted hours "
                 "(boxed; black line = observed occurrence)", x=0.01, ha="left", color=INK)

    # dashed box around the forecasted columns (positions are final only after a draw)
    fig.canvas.draw()
    renderer, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    boxes = [inv.transform(axes[r, t].get_tightbbox(renderer)) for r in range(rows) for t in range(n - k, n)]
    (x0, y0), (x1, y1) = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    pad = 0.004
    fig.add_artist(Rectangle((x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, transform=fig.transFigure,
                             fill=False, ec=INK, lw=1.5, ls="--"))
    fig.savefig(path, dpi=150)
    plt.close(fig)