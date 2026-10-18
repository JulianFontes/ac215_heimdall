"""figure style shared by every heimdall plot: arial bold text, 12 pt tick labels and
legends, 2 pt lines and 8 pt markers (the look of afl_utils.ags), plus the shared colors.

set as matplotlib defaults rather than applied to finished figures, so lines drawn thin
on purpose (pixel grids, gridlines) stay thin.

  @styled                  wrap a function that makes figures
  use()                    turn the style on for a whole script
"""
import functools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

STYLE = {
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

RAY_COLOR = "tab:orange"  # goes lines of sight, matched voxels and C3DIR predictions
SKY_COLOR = "#a9cbea"  # clear air
NO_DATA_COLOR = "0.25"  # no valid retrieval or no data
# clouds gray (thin) to white (thick) on a blue sky; values below the colormap range show as sky
CLOUD_CMAP = LinearSegmentedColormap.from_list("cloud", ["#5f6b78", "#b8c0c8", "#ffffff"]).with_extremes(
    under=SKY_COLOR, bad=NO_DATA_COLOR)
PROBABILITY_CMAP = LinearSegmentedColormap.from_list("probability", [SKY_COLOR, "#ffffff"]).with_extremes(
    bad=NO_DATA_COLOR)


def true_color(ds):
    """rough true color from goes bands 1 (blue), 2 (red) and 3 (veggie)."""
    r, b, veg = (np.clip(ds[f"CMI_C0{i}"].values, 0, 1) for i in (2, 1, 3))
    g = 0.45 * r + 0.10 * veg + 0.45 * b  # abi has no green band, so one is synthesized
    return np.nan_to_num(np.dstack([r, g, b])) ** (1 / 2.2)  # gamma brightening


def use():
    """apply the style to every figure made from now on."""
    plt.rcParams.update(STYLE)


def styled(func):
    """make the figures inside func with the style, leaving matplotlib's defaults untouched afterwards."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return func(*args, **kwargs)
    return wrapper
