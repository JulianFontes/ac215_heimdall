"""which SURFRAD stations move together: correlation of the clear-sky index between every station pair,
against great-circle distance.

  daily k:   sum ghi / sum ghi_cs / alpha per local solar date (energy-weighted, daylight minutes only)
  anomaly:   daily k minus its centered ANOMALY_WINDOW-day rolling median, so a shared seasonal cycle
             doesn't count as moving together
  10-min k:  to_10min k / alpha at the UTC times both stations have a value

figures go to figure_dir("eda"), the pair table to table_dir("eda").
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    MIN_DAY_MINUTES,
    STATION_NAMES,
    STATIONS,
    daylight,
    figure_dir,
    load_all_stations,
    solar_date,
    table_dir,
    to_10min,
)

# settings
ANOMALY_WINDOW = 31  # days
MIN_COMMON = 30  # common days / 10-min intervals needed for a pair correlation
EARTH_RADIUS_KM = 6371.0
FIG_DIR = figure_dir("eda")
TAB_DIR = table_dir("eda")
use_plot_style()

stations = load_all_stations()
colors = dict(zip(STATIONS, plt.get_cmap("tab10").colors))
labels = {c: f"{c.upper()} – {STATION_NAMES[c]}" for c in STATIONS}

daily, k10 = {}, {}
for code in STATIONS:
    meta, df = stations[code]
    alpha = df.attrs["clearsky_alpha"]
    ok = daylight(df) & df["ghi"].notna()
    date = solar_date(df.index, meta["longitude"])
    sums = df[["ghi", "ghi_cs"]].where(ok, axis=0).groupby(date).sum()
    n = ok.groupby(date).sum()
    daily[code] = (sums["ghi"] / sums["ghi_cs"] / alpha).where(n >= MIN_DAY_MINUTES)
    t = to_10min(df)
    k10[code] = t["k"] / alpha
daily = pd.DataFrame(daily)
anomaly = daily - daily.rolling(ANOMALY_WINDOW, center=True, min_periods=ANOMALY_WINDOW // 3).median()
k10 = pd.DataFrame(k10)


def great_circle_km(a, b):
    lat1, lon1, lat2, lon2 = np.radians([a["latitude"], a["longitude"], b["latitude"], b["longitude"]])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


pairs = []
for a, b in combinations(STATIONS, 2):
    row = {"a": a, "b": b, "distance_km": great_circle_km(stations[a][0], stations[b][0])}
    for name, table in [("daily", daily), ("anomaly", anomaly), ("10min", k10)]:
        both = table[[a, b]].dropna()
        row[f"r_{name}"] = both[a].corr(both[b]) if len(both) >= MIN_COMMON else np.nan
        row[f"n_{name}"] = len(both)
    # next-day skill: yesterday's anomaly at one station vs today's at the other
    row["r_lag_a_to_b"] = anomaly[a].shift(1).corr(anomaly[b])
    row["r_lag_b_to_a"] = anomaly[b].shift(1).corr(anomaly[a])
    pairs.append(row)
pairs = pd.DataFrame(pairs).sort_values("distance_km")
pairs.to_csv(TAB_DIR / "station_correlation_pairs.csv", index=False)


def efold_km(d, r):
    """L in r = exp(-d / L), least squares on log r through the origin (positive r only)."""
    keep = r > 0
    return -1 / (np.sum(d[keep] * np.log(r[keep])) / np.sum(d[keep] ** 2))


efold = {name: efold_km(pairs["distance_km"].to_numpy(), pairs[f"r_{name}"].to_numpy())
         for name in ["daily", "anomaly", "10min"]}

pd.set_option("display.width", 200)
print("station pairs, nearest first (r = Pearson correlation of alpha-corrected k):")
print(pairs.round(2).to_string(index=False))
print("\ne-folding distance of r = exp(-d / L): " + ", ".join(f"{k} {v:.0f} km" for k, v in efold.items()))
for name in ["daily", "10min"]:
    best = pairs.loc[pairs[f"r_{name}"].idxmax()]
    print(f"most correlated pair ({name}): {best['a']}-{best['b']} r = {best[f'r_{name}']:.2f} "
          f"at {best['distance_km']:.0f} km")
lag = pd.concat([pairs.assign(src=pairs.a, dst=pairs.b, r_lag=pairs.r_lag_a_to_b),
                 pairs.assign(src=pairs.b, dst=pairs.a, r_lag=pairs.r_lag_b_to_a)])
lag["src_lon"] = lag["src"].map(lambda c: stations[c][0]["longitude"])
lag["dst_lon"] = lag["dst"].map(lambda c: stations[c][0]["longitude"])
lag["west_to_east"] = lag["src_lon"] < lag["dst_lon"]
print("\nnext-day anomaly correlation (src yesterday -> dst today), strongest first:")
print(lag.sort_values("r_lag", ascending=False)[["src", "dst", "distance_km", "r_anomaly", "r_lag", "west_to_east"]]
      .head(10).round(2).to_string(index=False))
print("mean next-day r, west -> east vs east -> west: "
      f"{lag.loc[lag.west_to_east, 'r_lag'].mean():.3f} vs {lag.loc[~lag.west_to_east, 'r_lag'].mean():.3f}")
for code in STATIONS:
    others = pairs[(pairs.a == code) | (pairs.b == code)]
    print(f"  {code}: mean daily r with the others {others['r_daily'].mean():.2f}, "
          f"max {others['r_daily'].max():.2f}")


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIG_DIR / name}")


def matrix(col):
    m = pd.DataFrame(np.eye(len(STATIONS)), index=STATIONS, columns=STATIONS)
    for _, r in pairs.iterrows():
        m.loc[r["a"], r["b"]] = m.loc[r["b"], r["a"]] = r[col]
    return m


print("\nfigures:")

# 1. heatmaps: daily k, daily anomaly and 10-min k correlation for every pair
R_LIM = 0.5  # color limit; every pair is well inside it
fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), layout="constrained")
for ax, (col, title) in zip(axes, [("r_daily", "daily k"), ("r_anomaly", f"daily k anomaly ({ANOMALY_WINDOW}-day)"),
                                   ("r_10min", "10-min k, common UTC times")]):
    m = matrix(col).to_numpy()
    shown = np.where(np.eye(len(STATIONS), dtype=bool), np.nan, m)
    im = ax.imshow(shown, cmap="RdBu_r", vmin=-R_LIM, vmax=R_LIM)
    for (i, j), v in np.ndenumerate(shown):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > 0.6 * R_LIM else "k")
    ax.set_xticks(range(len(STATIONS)), [c.upper() for c in STATIONS])
    ax.set_yticks(range(len(STATIONS)), [labels[c] for c in STATIONS] if ax is axes[0] else
                  [c.upper() for c in STATIONS])
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="Pearson r of alpha-corrected k", shrink=0.8, extend="both")
best = pairs.loc[pairs["r_daily"].idxmax()]
fig.suptitle(f"Stations mostly move independently: the strongest daily pair is {best['a'].upper()}–{best['b'].upper()} "
             f"(r = {best['r_daily']:.2f}, {best['distance_km']:.0f} km apart);\nmedian daily r over all 21 pairs "
             f"is {pairs['r_daily'].median():.2f}, and {pairs['r_10min'].median():.2f} at 10 min", fontsize=14)
save(fig, "station_correlation_heatmap.png")

# 2. correlation vs distance with exponential fits, and same-day vs next-day for the closest pairs
fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
d_min, d_max = pairs["distance_km"].min(), pairs["distance_km"].max()
d_fit, d_extra = np.linspace(d_min, d_max, 200), np.linspace(0, d_min, 50)
for col, name, marker, color in [("r_daily", "daily", "o", "tab:blue"),
                                 ("r_anomaly", "anomaly", "s", "tab:orange"),
                                 ("r_10min", "10min", "^", "tab:green")]:
    label = {"daily": "daily k", "anomaly": "daily k anomaly", "10min": "10-min k"}[name]
    ax.scatter(pairs["distance_km"], pairs[col], marker=marker, s=70, color=color, edgecolor="k", lw=0.5,
               label=f"{label} (e-folding {efold[name]:.0f} km)", zorder=3)
    ax.plot(d_fit, np.exp(-d_fit / efold[name]), color=color, lw=1.8, ls="--")
    ax.plot(d_extra, np.exp(-d_extra / efold[name]), color=color, lw=1, ls=":")
notable = pairs[pairs["r_daily"] >= 0.19].sort_values("r_daily", ascending=False)
for i, (_, r) in enumerate(notable.iterrows()):  # only the notable pairs, each on its own text row
    ax.annotate(f"{r['a'].upper()}–{r['b'].upper()} {r['r_daily']:.2f}", (r["distance_km"], r["r_daily"]),
                xytext=(1500, 0.62 - 0.07 * i), textcoords="data", fontsize=11, color="0.2", va="center",
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8})
ax.axhline(0, color="0.3", lw=0.8)
ax.axvspan(0, d_min, color="0.92", zorder=0)
ax.text(d_min / 2, 0.5, "no station pair\ncloser than\n"
        f"{d_min:.0f} km\n(fit dotted)", ha="center", va="center", fontsize=11, color="0.35")
ax.set_xlabel("great-circle distance between stations (km)")
ax.set_ylabel("Pearson r of alpha-corrected k")
ax.set_xlim(0, d_max * 1.03)
ax.set_ylim(-0.2, 1.02)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(loc="upper right")
ax.set_title("same-day correlation vs distance (labels: daily r)")

ax = axes[1]
west_first = lag[lag.west_to_east].set_index(["src", "dst"])
east_first = lag[~lag.west_to_east].set_index(["dst", "src"])  # same pair, opposite direction
near = west_first.assign(r_back=east_first["r_lag"].reindex(west_first.index)).sort_values("r_lag").tail(8)
y = np.arange(len(near))
for off, col, color, label in [(0.27, "r_anomaly", "0.6", "same day"),
                               (0, "r_lag", "tab:purple", "western yesterday → eastern today"),
                               (-0.27, "r_back", "tab:cyan", "eastern yesterday → western today")]:
    ax.barh(y + off, near[col], 0.27, color=color, label=label)
    for yi, v in zip(y, near[col]):
        ax.annotate(f"{v:.2f}", (v, yi + off), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=10)
ax.set_yticks(y, [f"{s.upper()} → {d.upper()}  ({r['distance_km']:.0f} km)" for (s, d), r in near.iterrows()])
ax.axvline(0, color="0.3", lw=0.8)
ax.set_xlim(-0.2, 0.55)
ax.set_xlabel("Pearson r of daily k anomaly")
ax.legend(loc="lower right", fontsize=10)
ax.set_title("8 strongest west → east next-day pairs")
w2e = lag.loc[lag.west_to_east, "r_lag"].mean()
e2w = lag.loc[~lag.west_to_east, "r_lag"].mean()
top = lag.sort_values("r_lag").iloc[-1]
fig.suptitle(f"Same-day correlation is gone beyond ~1000 km (e-folding ~{efold['daily']:.0f} km), but weather "
             f"moves west to east:\nyesterday upwind predicts today downwind (mean r {w2e:.2f} vs {e2w:.2f} the "
             f"other way; best {top['src'].upper()} → {top['dst'].upper()} r = {top['r_lag']:.2f})", fontsize=14)
save(fig, "station_correlation_distance.png")
