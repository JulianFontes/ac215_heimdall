"""monthly sky climate of every SURFRAD station: which stations are seasonal vs steady, and when
variability peaks.

per station and month (local solar dates):
  - median alpha-corrected 10-min k
  - shares of clear / broken / overcast 10-min intervals (classify_intervals)
  - energy ratio: sum power / sum power_cs over minutes with a valid power value
  - median daily Stein variability index (daily_variability_index)

figures go to figure_dir("eda"), the table to table_dir("eda").
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    STATION_NAMES,
    STATIONS,
    classify_intervals,
    daily_variability_index,
    daylight,
    figure_dir,
    load_all_stations,
    solar_date,
    table_dir,
    to_10min,
)

# settings
FIG_DIR = figure_dir("eda")
TAB_DIR = table_dir("eda")
SKY_COLORS = {"clear": "#f2c94c", "broken": "#7aa6d6", "overcast": "#3b4a6b"}
use_plot_style()

stations = load_all_stations()
colors = dict(zip(STATIONS, plt.get_cmap("tab10").colors))
labels = {c: f"{c.upper()} – {STATION_NAMES[c]}" for c in STATIONS}
month_names = list(pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%b"))

rows = []
for code in STATIONS:
    meta, df = stations[code]
    alpha = df.attrs["clearsky_alpha"]
    k10 = to_10min(df).dropna(subset=["k"])
    k10["kc"] = k10["k"] / alpha
    k10["sky"] = classify_intervals(k10, alpha)
    k10["month"] = solar_date(k10.index, meta["longitude"]).month
    ok = daylight(df) & df["power"].notna()
    month_1min = solar_date(df.index, meta["longitude"]).month
    energy = df[["power", "power_cs"]].where(ok, axis=0).groupby(month_1min).sum()
    vi = daily_variability_index(df, meta["longitude"])
    vi_month = vi.groupby(vi.index.month)
    sky = k10.groupby("month")["sky"].value_counts(normalize=True).unstack().reindex(
        columns=["clear", "broken", "overcast"]).fillna(0)
    for m in range(1, 13):
        rows.append({"station": code, "month": m, "median_k": k10.loc[k10.month == m, "kc"].median(),
                     "clear": sky.loc[m, "clear"], "broken": sky.loc[m, "broken"],
                     "overcast": sky.loc[m, "overcast"],
                     "energy_ratio": energy.loc[m, "power"] / energy.loc[m, "power_cs"],
                     "median_vi": vi_month.median().get(m, np.nan), "vi_days": vi_month.count().get(m, 0),
                     "intervals": int((k10.month == m).sum())})
monthly = pd.DataFrame(rows)
monthly.to_csv(TAB_DIR / "monthly_climate.csv", index=False)


def pivot(col):
    return monthly.pivot(index="station", columns="month", values=col).reindex(STATIONS)


# seasonal vs steady: range of the monthly median k and of the broken share over the year
season = pd.DataFrame({
    "k_range": pivot("median_k").max(axis=1) - pivot("median_k").min(axis=1),
    "k_min_month": pivot("median_k").idxmin(axis=1), "k_max_month": pivot("median_k").idxmax(axis=1),
    "broken_range": pivot("broken").max(axis=1) - pivot("broken").min(axis=1),
    "broken_peak_month": pivot("broken").idxmax(axis=1),
    "vi_peak_month": pivot("median_vi").idxmax(axis=1), "vi_peak": pivot("median_vi").max(axis=1),
    "vi_min": pivot("median_vi").min(axis=1),
    "energy_ratio_min": pivot("energy_ratio").min(axis=1), "energy_ratio_max": pivot("energy_ratio").max(axis=1),
})
season.to_csv(TAB_DIR / "monthly_climate_seasonality.csv")
pd.set_option("display.width", 200)
print("monthly table (alpha-corrected median k, sky shares, energy ratio, median daily VI):")
print(monthly.round(2).to_string(index=False))
print("\nseasonality (range over the 12 months; months as numbers):")
print(season.round(2).to_string())
print("\nall-station median daily VI by month: " + ", ".join(
    f"{month_names[m - 1]} {v:.2f}" for m, v in pivot("median_vi").median().items()))


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIG_DIR / name}")


print("\nfigures:")
order = season["k_range"].sort_values(ascending=False).index  # most seasonal first

# 1. small multiples: sky-class shares (stacked bars) with median k and energy ratio on top
fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True, sharey=True, layout="constrained")
x = np.arange(1, 13)
for ax, code in zip(axes.flat, order):
    m = monthly[monthly.station == code].set_index("month")
    bottom = np.zeros(12)
    for cls, c in SKY_COLORS.items():
        ax.bar(x, m[cls], bottom=bottom, color=c, width=0.85, label=f"{cls} intervals")
        bottom += m[cls].to_numpy()
    ax.plot(x, m["median_k"], "o-", color="k", lw=2, ms=5, label="median 10-min k / alpha")
    ax.plot(x, m["energy_ratio"], "s--", color=colors[code], lw=2, ms=5, mec="k", mew=0.5,
            label="PV energy / clear-sky energy")
    ax.set_title(f"{labels[code]}  (median k range {season.loc[code, 'k_range']:.2f})", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.3, lw=0.5, axis="y")
axes.flat[-1].set_visible(False)
for ax in axes[-1]:
    ax.set_xticks(x, month_names, fontsize=10)
axes[-2, 1].tick_params(labelbottom=True)
axes[-2, 1].set_xticks(x, month_names, fontsize=10)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="center", bbox_to_anchor=(0.75, 0.12), fontsize=12, frameon=False)
fig.supylabel("share of daylight 10-min intervals  /  k  /  energy ratio")
most, least = order[0], order[-1]
fig.suptitle(f"{STATION_NAMES[most]} is the most seasonal station (median k {pivot('median_k').loc[most].min():.2f} "
             f"in {month_names[season.loc[most, 'k_min_month'] - 1]} to "
             f"{pivot('median_k').loc[most].max():.2f} in {month_names[season.loc[most, 'k_max_month'] - 1]});\n"
             f"{STATION_NAMES[least]} is the steadiest (range {season.loc[least, 'k_range']:.2f})."
             " Panels ordered most to least seasonal", fontsize=14)
save(fig, "monthly_climate_panels.png")

# 2. heatmaps (station x month): broken share and median daily VI
fig, axes = plt.subplots(2, 1, figsize=(13, 11), layout="constrained")
for ax, col, cmap, fmt, title, cbar in [
        (axes[0], "broken", "viridis", "{:.0%}", "share of broken-cloud 10-min intervals", "broken share"),
        (axes[1], "median_vi", "cividis", "{:.1f}", "median daily variability index (Stein VI)", "median daily VI")]:
    data = pivot(col).to_numpy()
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    lo, hi = np.nanmin(data), np.nanmax(data)
    for (i, j), v in np.ndenumerate(data):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10,
                    color="k" if (v - lo) / (hi - lo) > 0.55 else "white")
    ax.set_xticks(range(12), month_names)
    ax.set_yticks(range(len(STATIONS)), [labels[c] for c in STATIONS])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar, shrink=0.9)
vi_all = pivot("median_vi").median()
broken_all = pivot("broken").median()
fig.suptitle(f"Daily variability (VI) peaks in summer (all-station median {vi_all.max():.1f} in "
             f"{month_names[vi_all.idxmax() - 1]} vs {vi_all.min():.1f} in {month_names[vi_all.idxmin() - 1]}),\n"
             f"but the broken-interval share peaks in winter ({broken_all.max():.0%} in "
             f"{month_names[broken_all.idxmax() - 1]}): VI scales with sun strength, the share does not",
             fontsize=14)
save(fig, "monthly_climate_heatmaps.png")
