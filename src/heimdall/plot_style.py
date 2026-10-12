"""figure style shared by every heimdall plot: arial bold text, 12 pt tick labels and
legends, 2 pt lines and 8 pt markers (the look of afl_utils.ags).

set as matplotlib defaults rather than applied to finished figures, so lines drawn thin
on purpose (pixel grids, gridlines) stay thin.

  @styled                  wrap a function that makes figures
  use()                    turn the style on for a whole script
"""
import functools

import matplotlib.pyplot as plt

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
