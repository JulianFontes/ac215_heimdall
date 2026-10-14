"""how much of each station's "above clear sky" share is clear-sky model bias vs real cloud enhancement.

pvlib's Ineichen clear sky (monthly Linke turbidity climatology) runs low at every SURFRAD site, so raw
k = GHI / GHI_cs sits above 1 under a clear sky. detect_clearsky fits one scale factor alpha per station;
here we compare k before and after dividing by alpha, and check whether one alpha is enough:
  1. 10-min shares of clear / k > 1.05 / overcast and mean k, raw vs alpha-corrected
  2. the clear-sky level by month (median k on detected clear minutes) -> is the bias seasonal?
  3. the clear-sky level by solar zenith bin -> does the bias depend on sun angle?
  4. of the corrected k > 1.05 intervals, how many are smooth (residual bias) vs variable (enhancement)

figures go to figure_dir("eda"), tables to table_dir("eda").
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    CLEAR_K_STD,
    MAX_ZENITH,
    STATION_NAMES,
    STATIONS,
    classify_intervals,
    figure_dir,
    load_all_stations,
    table_dir,
    to_10min,
)

# settings
ABOVE_K = 1.05  # "above clear sky" threshold on 10-min k
ZENITH_BINS = np.arange(10, MAX_ZENITH + 1, 10)  # degrees; few clear minutes below 10° anywhere
MIN_CLEAR_MINUTES = 300  # clear 1-min samples needed for a monthly / zenith-bin clear-sky level
FIG_DIR = figure_dir("eda")
TAB_DIR = table_dir("eda")
use_plot_style()

stations = load_all_stations()
colors = dict(zip(STATIONS, plt.get_cmap("tab10").colors))
month_names = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%b")


def shares(k, sky):
    """10-min summary for one k column (raw or corrected) and its sky classes."""
    return {"clear": (sky == "clear").mean(), "above_1.05": (k > ABOVE_K).mean(),
            "overcast": (sky == "overcast").mean(), "mean_k": k.mean(), "median_k": k.median()}


summary, monthly, by_zenith = [], [], []
for code in STATIONS:
    meta, df = stations[code]
    alpha = df.attrs["clearsky_alpha"]
    k10 = to_10min(df).dropna(subset=["k"])
    raw = shares(k10["k"], classify_intervals(k10, 1.0))
    corr = shares(k10["k"] / alpha, classify_intervals(k10, alpha))

    # the clear-sky level by month and by zenith: median 1-min k on detected clear minutes
    clear = df.loc[df["is_clear"], ["k", "zenith"]].dropna()
    by_month = clear.groupby(clear.index.month)["k"].agg(["median", "count"])
    zbin = pd.cut(clear["zenith"], np.r_[0, ZENITH_BINS])
    by_z = clear.groupby(zbin, observed=False)["k"].agg(["median", "count"])
    for m, row in by_month.iterrows():
        monthly.append({"station": code, "month": m, "clear_k_median": row["median"],
                        "clear_minutes": int(row["count"]), "alpha": alpha})
    for b, row in by_z.iterrows():
        by_zenith.append({"station": code, "zenith_lo": b.left, "zenith_hi": b.right,
                          "clear_k_median": row["median"], "clear_minutes": int(row["count"]), "alpha": alpha})

    # residual after a monthly (instead of annual) correction, and the smooth / variable split
    month_level = by_month["median"].where(by_month["count"] >= MIN_CLEAR_MINUTES).reindex(range(1, 13))
    month_level = month_level.fillna(alpha)
    k_month = k10["k"] / k10.index.month.map(month_level).to_numpy()
    smooth_10 = k10["k_std_1min"] <= CLEAR_K_STD
    above = k10["k"] / alpha > ABOVE_K
    above_month = k_month > ABOVE_K
    ok_month = by_month["count"] >= MIN_CLEAR_MINUTES
    z_ok = by_z[by_z["count"] >= MIN_CLEAR_MINUTES]
    summary.append({
        "station": code, "name": meta["name"], "alpha": alpha, "intervals": len(k10),
        **{f"{c}_raw": v for c, v in raw.items()}, **{f"{c}_corr": v for c, v in corr.items()},
        "above_1.05_monthly_corr": above_month.mean(),
        # smooth = 1-min std like a clear sky: leftover model bias; variable = cloud enhancement
        "above_corr_smooth_share": (above & smooth_10).sum() / above.sum(),
        "above_monthly_smooth_share": (above_month & smooth_10).sum() / above_month.sum(),
        "enhancement_variable_corr": (above & ~smooth_10).mean(),
        "clear_k_month_min": by_month.loc[ok_month, "median"].min(),
        "clear_k_month_max": by_month.loc[ok_month, "median"].max(),
        "clear_k_month_argmin": by_month.loc[ok_month, "median"].idxmin(),
        "clear_k_month_argmax": by_month.loc[ok_month, "median"].idxmax(),
        "clear_k_z_first_bin": z_ok.index[0].left,  # lowest zenith bin the sun reaches with enough data
        "clear_k_z_low": z_ok["median"].iloc[0],
        "clear_k_z_high": z_ok["median"].iloc[-1],  # 70-80°
    })

summary = pd.DataFrame(summary).set_index("station")
summary["bias_share_of_above"] = 1 - summary["above_1.05_corr"] / summary["above_1.05_raw"]
summary["clear_k_month_range"] = summary["clear_k_month_max"] - summary["clear_k_month_min"]
monthly = pd.DataFrame(monthly)
by_zenith = pd.DataFrame(by_zenith)
summary.to_csv(TAB_DIR / "clearsky_bias_summary.csv")
monthly.to_csv(TAB_DIR / "clearsky_bias_monthly.csv", index=False)
by_zenith.to_csv(TAB_DIR / "clearsky_bias_zenith.csv", index=False)

print("10-min shares, raw k vs k / alpha (sky classes from classify_intervals):")
cols = ["alpha", "clear_raw", "clear_corr", "above_1.05_raw", "above_1.05_corr", "above_1.05_monthly_corr",
        "overcast_raw", "overcast_corr", "mean_k_raw", "mean_k_corr"]
print(summary[cols].round(3).to_string())
print("\nhow much of the raw k > 1.05 share the annual alpha removes (= model bias), and what's left:")
print(summary[["bias_share_of_above", "above_corr_smooth_share", "above_monthly_smooth_share",
               "enhancement_variable_corr"]].round(3).to_string())
print("  (smooth share: of the corrected k > 1.05 intervals, fraction with 1-min std <= "
      f"{CLEAR_K_STD}, i.e. smooth like clear sky rather than broken-cloud enhancement;")
print("   enhancement_variable_corr: share of all intervals with k / alpha > 1.05 and std > that)")
print("\nclear-sky level (median clear-minute k) by month and by zenith:")
print(summary[["clear_k_month_min", "clear_k_month_argmin", "clear_k_month_max", "clear_k_month_argmax",
               "clear_k_month_range", "clear_k_z_first_bin", "clear_k_z_low", "clear_k_z_high"]]
      .round(3).to_string())


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIG_DIR / name}")


print("\nfigures:")

labels = [f"{c.upper()} – {STATION_NAMES[c]}" for c in STATIONS]

# 1. the k > 1.05 share before and after correction, and the part that is real (variable) enhancement
bars = [("above_1.05_raw", "raw k (pvlib Ineichen)", "0.75", None),
        ("above_1.05_corr", "k / annual alpha", None, None),
        ("above_1.05_monthly_corr", "k / monthly clear-sky level", None, "//"),
        ("enhancement_variable_corr", "k / alpha > 1.05 and variable (cloud enhancement)", "k", None)]
fig, ax = plt.subplots(figsize=(12, 8.5), layout="constrained")
y = np.arange(len(STATIONS))
h = 0.2
for i, (col, label, color, hatch) in enumerate(bars):
    pos = y + (i - 1.5) * h
    vals = summary[col].to_numpy() * 100
    face = [color or colors[c] for c in STATIONS]
    ax.barh(pos, vals, h, color=face, hatch=hatch, edgecolor="white" if hatch else None, lw=0)
    for p, v in zip(pos, vals):
        ax.annotate(f"{v:.1f}", (v, p), xytext=(3, 0), textcoords="offset points", va="center", fontsize=10)
ax.set_yticks(y, labels, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel(f"% of daylight 10-min intervals with k > {ABOVE_K}")
ax.set_xlim(0, summary["above_1.05_raw"].max() * 100 * 1.12)
ax.grid(alpha=0.3, lw=0.5, axis="x")
handles = [plt.Rectangle((0, 0), 1, 1, color="0.75"), plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
           plt.Rectangle((0, 0), 1, 1, color="tab:blue", hatch="//", ec="white", lw=0),
           plt.Rectangle((0, 0), 1, 1, color="k")]
ax.legend(handles, [b[1] for b in bars], loc="lower right", title="bar (station color = corrected)")
lo, hi = summary["bias_share_of_above"].min(), summary["bias_share_of_above"].max()
enh_lo, enh_hi = summary["enhancement_variable_corr"].min() * 100, summary["enhancement_variable_corr"].max() * 100
ax.set_title(f"Most above-clear-sky intervals are clear-sky model bias: the station alpha removes "
             f"{lo:.0%}–{hi:.0%} of them,\nand only {enh_lo:.1f}–{enh_hi:.1f}% of intervals are "
             "variable cloud enhancement", fontsize=13)
save(fig, "clearsky_bias_shares.png")

# 2. clear-sky level relative to the annual alpha by month (heatmap) and by sun angle
ratio = (monthly[monthly.clear_minutes >= MIN_CLEAR_MINUTES]
         .assign(r=lambda d: d["clear_k_median"] / d["alpha"])
         .pivot(index="station", columns="month", values="r").reindex(index=STATIONS, columns=range(1, 13)))
spread = (ratio.max(axis=1) - ratio.min(axis=1)).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(17, 6), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
lim = np.nanmax(np.abs(ratio.to_numpy() - 1))
im = axes[0].imshow(ratio.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(1, 1 - lim, 1 + lim), aspect="auto")
for (i, j), v in np.ndenumerate(ratio.to_numpy()):
    if np.isfinite(v):
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                     color="white" if abs(v - 1) > 0.6 * lim else "k")
axes[0].set_xticks(range(12), month_names)
axes[0].set_yticks(range(len(STATIONS)), labels)
axes[0].set_title("by month: median clear-minute k / annual alpha")
fig.colorbar(im, ax=axes[0], label="clear-sky level / annual alpha (1 = alpha fits)", shrink=0.9)
for code, label in zip(STATIONS, labels):
    z = by_zenith[(by_zenith.station == code) & (by_zenith.clear_minutes >= MIN_CLEAR_MINUTES)]
    axes[1].plot((z["zenith_lo"] + z["zenith_hi"]) / 2, z["clear_k_median"] / z["alpha"], "o-",
                 color=colors[code], lw=1.8, ms=5, label=label)
axes[1].axhline(1, color="0.3", lw=0.8, ls=":")
axes[1].set_xlabel("solar zenith angle (°)")
axes[1].set_ylabel("median clear-minute k / annual alpha")
axes[1].set_title("by sun angle")
axes[1].grid(alpha=0.3, lw=0.5)
axes[1].legend(fontsize=10, loc="upper left")
fig.suptitle(f"One alpha per station is not enough: the clear-sky bias swings by {spread.iloc[-1]:.0%} over the "
             f"year at {STATION_NAMES[spread.index[-1]]}\n(high in winter / low sun) but only "
             f"{spread.iloc[0]:.0%} at {STATION_NAMES[spread.index[0]]}", fontsize=14)
save(fig, "clearsky_bias_month_zenith.png")
