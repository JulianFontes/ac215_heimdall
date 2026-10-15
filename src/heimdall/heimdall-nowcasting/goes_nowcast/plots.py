"""Training curves and forecast plots."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .config import PLOT_EVERY_MIN, STEP_MIN, TARGET_BANDS, T_OUT
from .data import split_batch
from .evaluation import persistence

GRAY, BLUE = "#898781", "#2a78d6"

def unit_label(u):
    return "unitless" if u == "1" else u # "1" is the CF convention for dimensionless (reflectance factor)

def plot_leads():
    """Indices of the lead times to plot: every PLOT_EVERY_MIN minutes (all lead times if none match)."""
    return [k for k in range(T_OUT) if (k + 1) * STEP_MIN % PLOT_EVERY_MIN == 0] or list(range(T_OUT))

def by_unit(units):
    """Target bands grouped by unit, in band order: [(unit, [bands]), ...]. Bands with different units cannot share an axis.
    units: {band: unit} as read from the data files."""
    groups = {}
    for b in TARGET_BANDS:
        groups.setdefault(units[b], []).append(b)
    return list(groups.items())

def plot_curves(history, units, path):
    """Left: train / validation loss. Right: validation RMSE of every target band on one axis, ConvLSTM (solid) next to
    persistence (dashed) in the same colour, in the bands' own unit (bands with different units get their own axis).
    The dotted vertical line marks the best epoch (lowest validation loss), i.e. the saved checkpoint."""
    ep = [h["epoch"] for h in history]
    best = int(min(history, key=lambda h: h["val_loss"])["epoch"])
    i_best = ep.index(best)
    groups = by_unit(units)
    palette = (plt.cm.tab10 if len(TARGET_BANDS) <= 10 else plt.cm.tab20).colors
    color = {b: palette[i % len(palette)] for i, b in enumerate(TARGET_BANDS)}

    fig, axes = plt.subplots(1, 1 + len(groups), figsize=(5.2 * (1 + len(groups)), 5.0), squeeze=False, layout="constrained")
    axes = axes[0]
    axes[0].plot(ep, [h["train_loss"] for h in history], color=GRAY, label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], color="k", label="validation")
    axes[0].set(ylabel="masked MSE (normalized)", title="Loss")
    axes[0].legend(fontsize=8)
    for ax, (u, bands) in zip(axes[1:], groups):
        for b in bands:
            model = [h[f"val_RMSE_C{b:02d}"] for h in history]
            base = [h[f"base_RMSE_C{b:02d}"] for h in history]
            ax.plot(ep, base, "--", color=color[b])
            ax.plot(ep, model, color=color[b])
            ax.plot(ep[i_best], model[i_best], "o", color=color[b], ms=4)
        ax.set(ylabel=f"RMSE ({unit_label(u)})", title=f"Validation RMSE ({unit_label(u)})")
    for ax in axes:
        ax.axvline(best, color="k", ls=":", lw=1.2)
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([0], [0], color=color[b], label=f"C{b:02d}") for b in TARGET_BANDS]
    handles += [Line2D([0], [0], color="k", label="ConvLSTM (solid)"), Line2D([0], [0], color="k", ls="--", label="persistence (dashed)"),
                Line2D([0], [0], color="k", ls=":", label=f"best model (epoch {best})")]
    fig.suptitle("Validation error: ConvLSTM vs persistence (lower is better)", fontsize=11)
    # below the axes, so it never meets the titles; "outside" reserves room for however many rows it needs
    fig.legend(handles=handles, ncol=min(len(handles), 10), loc="outside lower center", frameon=False, fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)

def bars(ax, labels, base, model, fmt):
    """Persistence next to the model for each label, value written on every bar."""
    x = np.arange(len(labels))
    for off, vals, color, name in ((-0.2, base, GRAY, "persistence"), (0.2, model, BLUE, "ConvLSTM")):
        ax.bar_label(ax.bar(x + off, vals, 0.38, color=color, label=name), fmt=fmt, fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.spines[["top", "right"]].set_visible(False)

def plot_vs_persistence(r, base, units, path, leads=None):
    """RMSE of persistence next to the ConvLSTM for every band in config.TARGET_BANDS, every PLOT_EVERY_MIN minutes of
    lead time (or the lead indices in `leads`), at most 3 panels per row. Bands are grouped by unit since they cannot share an axis."""
    leads = leads or plot_leads()
    groups = by_unit(units)
    ncols = min(3, len(leads))
    rows_per_group = -(-len(leads) // ncols) # ceil
    rows = rows_per_group * len(groups)
    width = max(3.6, 0.5 * max(len(bands) for _, bands in groups)) # wide enough for every band label
    fig = plt.figure(figsize=(width * ncols, 3.0 * rows + 0.6))
    gs = fig.add_gridspec(rows, ncols)
    for g, (u, bands) in enumerate(groups):
        for j, k in enumerate(leads):
            row, col = g * rows_per_group + j // ncols, j % ncols
            ax = fig.add_subplot(gs[row, col])
            bars(ax, [f"C{b:02d}" for b in bands], [base[f"t{k}_C{b:02d}"]["RMSE"] for b in bands],
                 [r[f"t{k}_C{b:02d}"]["RMSE"] for b in bands], "%.1f" if u == "K" else "%.3f")
            ax.set_title(f"+{(k + 1) * STEP_MIN} min", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"RMSE ({unit_label(u)})")
    handles, names = fig.axes[0].get_legend_handles_labels()
    fig.suptitle("RMSE by band, lower is better", y=0.99, fontsize=11)
    fig.legend(handles, names, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=120)
    plt.close(fig)

@torch.no_grad()
def plot_forecast(model, loader, stats, units, device, out_dir):
    """One figure per target band, forecast_CXX.png in out_dir (created if missing): truth, ConvLSTM forecast, persistence and the error of
    both forecasts for the first window of the loader. Each error panel shows its RMSE over the valid pixels. The error
    colour scale is set per band (99th percentile of the persistence error), so bands with small errors stay readable."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    mean, std = (a.cpu() for a in stats)
    x, y, valid = split_batch(next(iter(loader)))
    pred, pers = model(x.to(device)).cpu(), persistence(x)
    ok = valid[0, :, 0].bool()
    for i, b in enumerate(TARGET_BANDS):
        phys = lambda a: a[0, :, i] * std[i] + mean[i] # first window, band i, physical units
        truth, pr, pe = phys(y), phys(pred), phys(pers)
        vmin, vmax = float(truth.min()), float(truth.max())
        emax = float((pe - truth)[ok].abs().quantile(0.99))
        leads = plot_leads()
        fig, ax = plt.subplots(5, len(leads), figsize=(2.2 * len(leads) + 1.0, 10.6), squeeze=False, layout="constrained")
        for c, k in enumerate(leads):
            for row, img in enumerate((truth, pr, pe)):
                im = ax[row, c].imshow(img[k], vmin=vmin, vmax=vmax, cmap="gray_r")
            for row, f in ((3, pr), (4, pe)):
                e = f[k] - truth[k]
                err = ax[row, c].imshow(e, vmin=-emax, vmax=emax, cmap="RdBu_r")
                rmse = float(e[ok[k]].pow(2).mean().sqrt())
                ax[row, c].text(0.03, 0.97, f"RMSE {rmse:.3g}", transform=ax[row, c].transAxes, va="top", fontsize=8,
                                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            ax[0, c].set_title(f"+{(k + 1) * STEP_MIN} min", fontsize=9)
        for a, label in zip(ax[:, 0], ("truth", "ConvLSTM", "persistence", "ConvLSTM - truth", "persistence - truth")):
            a.set_ylabel(label)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        u = unit_label(units[b])
        fig.colorbar(im, ax=ax[:3, :], shrink=0.9).set_label(f"C{b:02d} ({u})", fontsize=8)
        fig.colorbar(err, ax=ax[3:, :], shrink=0.9).set_label(f"forecast - truth ({u})\nred: (overshoot) ), blue: (undershoot)", fontsize=8)
        fig.suptitle(f"Band C{b:02d}: ConvLSTM vs persistence")
        fig.savefig(os.path.join(out_dir, f"forecast_C{b:02d}.png"), dpi=100)
        plt.close(fig)
