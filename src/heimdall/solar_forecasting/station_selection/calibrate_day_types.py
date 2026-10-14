"""calibrate the daily Stein VI thresholds behind classify_days (CLEAR_DAY_VI, OVERCAST_DAY_VI).

for every station's 2025 days: daily corrected k, daily VI and the share of daylight minutes pvlib's
detect_clearsky marks clear. prints VI quantiles for reference groups (mostly-clear days, high-k days,
low-k days) and the day-type counts with the thresholds in common.py, and plots
  - the daily VI distribution per reference group, with the thresholds
  - random days per VI bin near each threshold (measured vs clear-sky GHI), to see by eye where
    smooth days turn broken
  - random days of each day type per station, to check the classes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting import common
from heimdall.solar_forecasting.common import (
    STATION_NAMES,
    STATIONS,
    classify_days,
    daylight,
    figure_dir,
    load_all_stations,
    solar_date,
)

# settings
MOSTLY_CLEAR = 0.9  # share of daylight minutes is_clear for a reference clear day
DAYS_PER_PANEL_ROW = 4  # random example days per row
SEED = 0
PLOT_DIR = figure_dir("selection", "day_type_check")
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script
DAY_TYPES = ["clear", "variable", "overcast"]

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


def daily_table(df):
    """classify_days plus the daily share of daylight minutes flagged clear by detect_clearsky."""
    days = classify_days(df)
    date = solar_date(df.index, df.attrs["longitude"])
    days["clear_share"] = df["is_clear"].where(daylight(df)).groupby(date).mean().reindex(days.index)
    return days


def quantiles(vi):
    q = vi.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return f"n={len(vi):4d}  " + "  ".join(f"q{int(p * 100):02d}={v:5.2f}" for p, v in q.items())


def plot_day(ax, df, d, title):
    """measured and alpha-scaled clear-sky GHI of local solar date d against local solar time."""
    g = df[(solar_date(df.index, df.attrs["longitude"]) == d) & daylight(df)]
    hours = (g.index.tz_localize(None) - d).total_seconds() / 3600 + df.attrs["longitude"] / 15
    ax.plot(hours, g["ghi_cs"] * df.attrs["clearsky_alpha"], color="0.45", lw=1.2)
    ax.plot(hours, g["ghi"], color="tab:orange", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(5, 19)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)


def legend_handles():
    return [plt.Line2D([], [], color="tab:orange", lw=1.5, label="measured GHI"),
            plt.Line2D([], [], color="0.45", lw=1.5, label="clear-sky GHI × station alpha")]


def plot_vi_histogram(all_days):
    groups = [(f"detect_clearsky marks ≥ {MOSTLY_CLEAR:.0%} of minutes clear",
               all_days["clear_share"] >= MOSTLY_CLEAR, "tab:blue"),
              (f"daily k ≥ {common.CLEAR_DAY_K}", all_days["k"] >= common.CLEAR_DAY_K, "tab:green"),
              (f"{common.OVERCAST_DAY_K} ≤ daily k < {common.CLEAR_DAY_K}",
               all_days["k"].between(common.OVERCAST_DAY_K, common.CLEAR_DAY_K, inclusive="left"), "tab:orange"),
              (f"daily k < {common.OVERCAST_DAY_K}", all_days["k"] < common.OVERCAST_DAY_K, "0.3")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(0.5, 15.01, 0.125)
    for name, mask, color in groups:
        ax.hist(all_days.loc[mask, "vi"].clip(upper=15), bins=bins, histtype="step", lw=1.8, color=color,
                label=f"{name} (n = {mask.sum()})")
    ax.axvline(common.CLEAR_DAY_VI, color="k", ls="--", lw=1.2, label=f"CLEAR_DAY_VI = {common.CLEAR_DAY_VI}")
    ax.axvline(common.OVERCAST_DAY_VI, color="k", ls=":", lw=1.5,
               label=f"OVERCAST_DAY_VI = {common.OVERCAST_DAY_VI}")
    ref = all_days.loc[all_days["clear_share"] >= MOSTLY_CLEAR, "vi"]
    ax.annotate(f"clear-day peak: median VI {ref.median():.2f},\n95th percentile {ref.quantile(0.95):.2f}",
                xy=(ref.median(), 230), xytext=(3.5, 150), fontsize=11, arrowprops={"arrowstyle": "->"})
    ax.set_yscale("log")
    ax.set_xlabel("daily Stein variability index (1 = as smooth as clear sky; last bin = VI ≥ 15)")
    ax.set_ylabel("days, all 7 stations, 2025 (log scale)")
    ax.set_title("Genuinely clear days have VI ≈ 1.0–1.4; everything above 1.5 has cloud passages")
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.savefig(PLOT_DIR / "daily_vi_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boundary(stations, all_days, name, mask, vi_bins, title, rng):
    """random days (any station) per VI bin among days matching mask, to see where smooth turns broken."""
    fig, axes = plt.subplots(len(vi_bins), DAYS_PER_PANEL_ROW, figsize=(4 * DAYS_PER_PANEL_ROW, 2.6 * len(vi_bins)),
                             squeeze=False, sharex=True)
    for row, (lo, hi) in zip(axes, vi_bins):
        pool = all_days[mask & all_days["vi"].between(lo, hi, inclusive="left")].index
        pick = [pool[i] for i in sorted(rng.choice(len(pool), size=min(DAYS_PER_PANEL_ROW, len(pool)), replace=False))]
        for ax, (code, d) in zip(row, pick):
            r = all_days.loc[(code, d)]
            plot_day(ax, stations[code][1], d, f"{code.upper()} {d:%Y-%m-%d}  k={r['k']:.2f}  VI={r['vi']:.2f}")
        for ax in row[len(pick):]:
            ax.axis("off")
        row[0].set_ylabel(f"VI {lo}–{hi}\nGHI (W m$^{{-2}}$)", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("local solar time (h)", fontsize=11)
    fig.legend(handles=legend_handles(), loc="upper right", ncol=2, fontsize=11, bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_DIR / f"boundary_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_examples(code, df, days, rng):
    """random days of each day type for one station."""
    fig, axes = plt.subplots(len(DAY_TYPES), DAYS_PER_PANEL_ROW, figsize=(4 * DAYS_PER_PANEL_ROW, 2.8 * len(DAY_TYPES)),
                             sharex=True, squeeze=False)
    counts = days["day_type"].value_counts()
    for row, t in zip(axes, DAY_TYPES):
        pool = days.index[days["day_type"] == t]
        pick = sorted(pool[i] for i in rng.choice(len(pool), size=min(DAYS_PER_PANEL_ROW, len(pool)), replace=False))
        for ax, d in zip(row, pick):
            r = days.loc[d]
            plot_day(ax, df, d, f"{d:%Y-%m-%d}  k={r['k']:.2f}  VI={r['vi']:.2f}")
        for ax in row[len(pick):]:
            ax.axis("off")
        row[0].set_ylabel(f"{t} ({counts.get(t, 0)} days)\nGHI (W m$^{{-2}}$)", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("local solar time (h)", fontsize=11)
    fig.legend(handles=legend_handles(), loc="upper right", ncol=2, fontsize=11, bbox_to_anchor=(1, 1.0))
    fig.suptitle(f"{label(code)}: clear days are smooth, variable days broken; overcast days are dim but can "
                 f"have brief sun breaks\n(random days per type; clear: k ≥ {common.CLEAR_DAY_K} and "
                 f"VI < {common.CLEAR_DAY_VI}, overcast: k < {common.OVERCAST_DAY_K} and VI < {common.OVERCAST_DAY_VI})",
                 x=0.01, ha="left", y=1.0, fontsize=13, color=COLORS[code])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PLOT_DIR / f"day_types_{code}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    stations = load_all_stations()
    tables = {code: daily_table(df) for code, (_, df) in stations.items()}
    all_days = pd.concat(tables, names=["station", "date"])

    high_k = all_days["k"] >= common.CLEAR_DAY_K
    mostly_clear = all_days["clear_share"] >= MOSTLY_CLEAR
    print("daily VI, all stations")
    for name, mask in {f"is_clear share >= {MOSTLY_CLEAR}": mostly_clear,
                       f"k >= {common.CLEAR_DAY_K}": high_k,
                       f"k >= {common.CLEAR_DAY_K} and is_clear share < 0.5": high_k & (all_days["clear_share"] < 0.5),
                       f"k < {common.OVERCAST_DAY_K}": all_days["k"] < common.OVERCAST_DAY_K}.items():
        print(f"  {name:45s} {quantiles(all_days.loc[mask, 'vi'])}")
    edges = [1, 1.1, 1.2, 1.3, 1.5, 1.75, 2, 2.5, 3, 4, 6, 10, np.inf]
    print(f"\nk >= {common.CLEAR_DAY_K} days by VI bin: count and median is_clear share")
    print(all_days[high_k].groupby(pd.cut(all_days.loc[high_k, "vi"], edges, right=False), observed=False)
          ["clear_share"].agg(["count", "median"]).round(2))

    print(f"\nday types with CLEAR_DAY_VI={common.CLEAR_DAY_VI}, OVERCAST_DAY_VI={common.OVERCAST_DAY_VI}")
    counts = all_days.groupby("station")["day_type"].value_counts().unstack(fill_value=0)[DAY_TYPES]
    counts["variable share"] = (counts["variable"] / counts.sum(axis=1)).round(3)
    print(counts)

    rng = np.random.default_rng(SEED)
    plot_vi_histogram(all_days)
    plot_boundary(stations, all_days, "high_k", high_k,
                  [(1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)],
                  f"Days with daily k ≥ {common.CLEAR_DAY_K}: smooth below VI 1.5, visible cloud passages above",
                  rng)
    plot_boundary(stations, all_days, "low_k", all_days["k"] < common.OVERCAST_DAY_K,
                  [(0.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0)],
                  f"Days with daily k < {common.OVERCAST_DAY_K}: unbroken overcast below VI 3, sun breaks become common above",
                  rng)
    for code, (_, df) in stations.items():
        plot_examples(code, df, tables[code], rng)
    print(f"\nfigures in {PLOT_DIR}")


if __name__ == "__main__":
    main()
