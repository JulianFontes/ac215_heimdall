"""when each selected station is variable: by month and by local solar hour.

per station (all from the 2025 1-min data, k corrected by the station's clearsky_alpha):
  - month x local solar hour heatmaps of median corrected 10-min k and of median 1-min k std inside
    each 10-min interval (the variability GOES can't resolve)
  - monthly shares of clear / broken / overcast 10-min intervals (classify_intervals) and monthly PV
    energy of the reference system against its clear-sky plan
plus one cross-station figure of monthly broken share and monthly median daily VI.

    python seasonal_diurnal.py [--station CODE]   (default: every code in selected_stations.csv)

tables go to <DATA_DIR>/stations/deep_dive/<code>/, figures to <FIGURES_DIR>/solar_forecasting/deep_dive/<code>/
(comparison figure under deep_dive/comparison/, only when every selected station is run).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    STATION_NAMES,
    STATIONS,
    alpha_corrected_k,
    classify_days,
    classify_intervals,
    figure_dir,
    load_station_1min,
    solar_date,
    table_dir,
    to_10min,
)

# settings
SELECTED = table_dir("selection") / "selected_stations.csv"
HOURS = np.arange(6, 18)  # local solar hour bins [h, h + 1) shown in the heatmaps
MIN_CELL = 20  # 10-min intervals needed for a heatmap cell
SKY_CLASSES = ["clear", "broken", "overcast"]
SKY_COLORS = {"clear": "#F0C419", "broken": "#2A9D8F", "overcast": "0.45"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_NAME = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


def selected_codes():
    return list(pd.read_csv(SELECTED)["code"])


def time_of_day(hour):
    return "mornings" if hour < 11 else "around solar noon" if hour < 13 else "afternoons"


def interval_table(df):
    """10-min rows with corrected k, corrected k_std_1min, sky class, solar month and solar hour bin."""
    alpha, lon = df.attrs["clearsky_alpha"], df.attrs["longitude"]
    k10 = to_10min(df)
    centre = k10.index + pd.Timedelta("5min")
    solar_hour = (centre.hour + centre.minute / 60 + lon / 15) % 24
    return pd.DataFrame({"k": alpha_corrected_k(k10, alpha), "k_std": k10["k_std_1min"] / alpha,
                         "sky": classify_intervals(k10, alpha),
                         "month": solar_date(k10.index, lon).month, "hour": np.floor(solar_hour).astype(int)},
                        index=k10.index)


def diurnal_grid(k10):
    """month x hour medians of k and k_std (nan where fewer than MIN_CELL intervals), plus long table."""
    g = k10[k10["hour"].isin(HOURS)].groupby(["month", "hour"])
    long = pd.DataFrame({"median_k": g["k"].median(), "median_k_std_1min": g["k_std"].median(), "n": g.size()})
    long = long.where(long["n"] >= MIN_CELL)
    full = pd.MultiIndex.from_product([range(1, 13), HOURS], names=["month", "hour"])
    long = long.reindex(full)
    return long, long["median_k"].unstack(), long["median_k_std_1min"].unstack()


def monthly_table(df, k10, days):
    """per solar month: sky-class shares, PV energy vs clear-sky plan (alpha-scaled), median daily VI."""
    alpha, lon = df.attrs["clearsky_alpha"], df.attrs["longitude"]
    shares = pd.crosstab(k10["month"], k10["sky"], normalize="index").reindex(columns=SKY_CLASSES, fill_value=0)
    shares.columns = [f"{c}_share" for c in SKY_CLASSES]
    ok = df["power"].notna() & df["power_cs"].notna()
    month = solar_date(df.index, lon).month
    energy = df.loc[ok, ["power", "power_cs"]].groupby(month[ok]).sum() / 60 / 1000  # W min -> kWh
    out = shares.join(pd.DataFrame({"energy_kwh": energy["power"], "plan_kwh": energy["power_cs"] * alpha}))
    out["energy_ratio"] = out["energy_kwh"] / out["plan_kwh"]
    out["median_daily_vi"] = days["vi"].groupby(days.index.month).median()
    out["n_intervals"] = k10.groupby("month").size()
    return out.rename_axis("month")


def annotate_grid(ax, values, fmt, threshold):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10,
                        color="black" if v > threshold else "white")


def plot_heatmaps(code, k_grid, std_grid, path):
    month_std, hour_std = std_grid.mean(axis=1), std_grid.mean(axis=0)  # mean of cell medians
    peak_m, peak_h = int(month_std.idxmax()), int(hour_std.idxmax())
    calm_m = int(month_std.idxmin())
    cell = std_grid.stack().idxmax()
    dim_m = int(k_grid.median(axis=1).idxmin())
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.8), sharey=True)
    panels = [(axes[0], k_grid, "cividis", (0, 1.05), "{:.2f}", 0.55, "median corrected 10-min k (–, 1 = clear sky)",
               (f"Median clear-sky index: dimmest month {MONTHS[dim_m - 1]} "
                f"(monthly median k {k_grid.loc[dim_m].median():.2f})")),
              (axes[1], std_grid, "viridis", (0, np.nanmax(std_grid.to_numpy())), "{:.3f}",
               0.6 * np.nanmax(std_grid.to_numpy()), "median std of 1-min corrected k in 10-min interval (–)",
               (f"Sub-10-min variability: peak cell {MONTHS[cell[0] - 1]} {cell[1]}–{cell[1] + 1} h "
                f"({std_grid.loc[cell]:.3f})"))]
    for ax, grid, cmap, (vmin, vmax), fmt, thr, clabel, title in panels:
        im = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        annotate_grid(ax, grid.to_numpy(), fmt, thr)
        ax.set_xticks(range(len(HOURS)), [f"{h}" for h in HOURS])
        ax.set_xlabel("local solar hour (h, bin start; 12 = solar noon)", fontsize=12)
        ax.set_title(title, fontsize=13)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(clabel, fontsize=11)
    axes[0].set_yticks(range(12), MONTHS)
    axes[0].set_ylabel("month (2025)", fontsize=12)
    fig.suptitle(f"{STATION_NAMES[code]}'s variability peaks in {SEASON_NAME[peak_m]} {time_of_day(peak_h)}: "
                 f"1-min k std averages {month_std[peak_m]:.3f} over {MONTHS[peak_m - 1]}'s hours vs "
                 f"{month_std[calm_m]:.3f} in {MONTHS[calm_m - 1]}, highest at {peak_h}–{peak_h + 1} h solar time"
                 f"\n{label(code)}, 2025; blank cells: "
                 f"< {MIN_CELL} intervals or sun lower than 10°", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"peak_month": MONTHS[peak_m - 1], "peak_hour": peak_h,
            "peak_cell": f"{MONTHS[cell[0] - 1]} {cell[1]} h ({std_grid.loc[cell]:.3f})",
            "peak_month_std": month_std[peak_m], "calm_month": MONTHS[calm_m - 1], "calm_month_std": month_std[calm_m]}


def plot_monthly(code, monthly, path):
    x = np.arange(12)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True, gridspec_kw={"hspace": 0.32})
    bottom = np.zeros(12)
    for sky in SKY_CLASSES:
        v = monthly[f"{sky}_share"].to_numpy() * 100
        ax1.bar(x, v, bottom=bottom, color=SKY_COLORS[sky], edgecolor="white", lw=0.8, label=sky)
        for xi, (b, vi) in enumerate(zip(bottom, v)):
            if vi >= 6:
                ax1.text(xi, b + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=11,
                         color="white" if sky != "clear" else "black")
        bottom += v
    broken = monthly["broken_share"] * 100
    top_m = int(broken.idxmax())
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("share of daylight 10-min intervals (%)", fontsize=12)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=12)
    ax1.set_title(f"Broken cloud is the most common sky in {(broken > monthly[['clear_share', 'overcast_share']].max(axis=1) * 100).sum()} "
                  f"of 12 months (peak {broken[top_m]:.0f}% in {MONTHS[top_m - 1]}; year "
                  f"{monthly['broken_share'].mul(monthly['n_intervals']).sum() / monthly['n_intervals'].sum():.1%})",
                  fontsize=13)

    w = 0.38
    ax2.bar(x - w / 2, monthly["plan_kwh"], w, color="0.75", edgecolor="0.4", label="clear-sky plan (× station alpha)")
    ax2.bar(x + w / 2, monthly["energy_kwh"], w, color=COLORS[code], edgecolor="0.3", label="produced (measured sky)")
    for xi, (p, e, r) in enumerate(zip(monthly["plan_kwh"], monthly["energy_kwh"], monthly["energy_ratio"])):
        ax2.text(xi, max(p, e) * 1.02, f"{r:.0%}", ha="center", va="bottom", fontsize=11)
    ratio_year = monthly["energy_kwh"].sum() / monthly["plan_kwh"].sum()
    low_m = int(monthly["energy_ratio"].idxmin())
    ax2.set_ylim(0, monthly[["plan_kwh", "energy_kwh"]].max().max() * 1.18)
    ax2.set_ylabel("PV energy, 5 kW reference system (kWh)", fontsize=12)
    ax2.set_xticks(x, MONTHS)
    ax2.set_xlabel("month (2025; % = produced / plan)", fontsize=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=11, ncol=2)
    ax2.set_title(f"Clouds cost {1 - ratio_year:.0%} of the clear-sky energy over the year "
                  f"(worst month {MONTHS[low_m - 1]}: {monthly.loc[low_m, 'energy_ratio']:.0%} delivered)", fontsize=13)
    fig.suptitle(f"{label(code)}: sky mix and PV energy by month", fontsize=15, x=0.01, ha="left", y=0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"broken_peak_month": MONTHS[top_m - 1], "broken_peak": broken[top_m] / 100, "energy_ratio_year": ratio_year,
            "worst_energy_month": MONTHS[low_m - 1], "worst_energy_ratio": monthly.loc[low_m, "energy_ratio"]}


def plot_comparison(monthlies, path):
    x = np.arange(12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6.8))
    broken = pd.DataFrame({c: m["broken_share"] * 100 for c, m in monthlies.items()})
    vi = pd.DataFrame({c: m["median_daily_vi"] for c, m in monthlies.items()})
    for ax, values, fmt, gap in ((ax1, broken, "{:.0f}%", 4), (ax2, vi, "{:.1f}", 1)):
        for code in values:
            ax.plot(x, values[code], "-o", color=COLORS[code], label=label(code))
            i = int(values[code].to_numpy().argmax())
            v = values[code].iloc[i]
            others = values.drop(columns=code).iloc[i]
            below = ((others > v) & (others <= v + gap)).any()  # label would sit on another station's point
            ax.annotate(fmt.format(v), (i, v), xytext=(0, -18 if below else 9), textcoords="offset points",
                        ha="center", fontsize=11, color=COLORS[code])
    ax1.set_ylabel("10-min intervals classed broken (%)", fontsize=12)
    ax1.set_title("Broken-cloud share by month (peak labelled)", fontsize=13)
    ax2.set_ylabel("median daily Stein VI (–, 1 = smooth as clear sky)", fontsize=12)
    ax2.set_title("Median daily variability index by month (peak labelled)", fontsize=13)
    ax2.axhline(1.5, color="0.4", ls="--", lw=1.2)
    ax2.text(11.3, 1.5, "clear-day limit 1.5", ha="right", va="bottom", fontsize=10, color="0.3")
    for ax in (ax1, ax2):
        ax.set_xticks(x, MONTHS)
        ax.set_xlabel("month (2025)", fontsize=12)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.08)
        ax.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(monthlies), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.06))
    vi_peak = vi.idxmax()
    same = vi_peak.nunique() == 1
    vi_text = (f"Daily variability peaks in {MONTHS[vi_peak.iloc[0] - 1]} at every station "
               f"(median daily VI {vi.max().min():.1f}–{vi.max().max():.1f})" if same else
               "Daily variability peaks in " + ", ".join(f"{MONTHS[m - 1]} at {c.upper()}" for c, m in vi_peak.items()))
    fig.suptitle(f"{vi_text};\nbroken skies cover ≥ {broken.min().min():.0f}% of daylight in every month, "
                 f"peaking in " + ", ".join(f"{MONTHS[m - 1]} ({c.upper()})" for c, m in broken.idxmax().items()),
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_station(code):
    _, df = load_station_1min(code)
    k10 = interval_table(df)
    days = classify_days(df)
    long, k_grid, std_grid = diurnal_grid(k10)
    monthly = monthly_table(df, k10, days)
    fig_dir, tab_dir = figure_dir("deep_dive", code), table_dir("deep_dive", code)
    long.round(4).to_csv(tab_dir / "diurnal_month_hour.csv")
    monthly.round(4).to_csv(tab_dir / "monthly_sky_energy.csv")
    paths = [fig_dir / "diurnal_heatmaps.png", fig_dir / "monthly_sky_energy.png"]
    facts = plot_heatmaps(code, k_grid, std_grid, paths[0]) | plot_monthly(code, monthly, paths[1])
    year_shares = k10["sky"].value_counts(normalize=True)
    print(f"\n{label(code)}  (alpha {df.attrs['clearsky_alpha']:.3f})")
    print("  year sky shares:", ", ".join(f"{s} {year_shares.get(s, 0):.1%}" for s in SKY_CLASSES))
    for key, v in facts.items():
        print(f"  {key}: {v:.3f}" if isinstance(v, float) else f"  {key}: {v}")
    print(monthly.round(3).to_string())
    print("  figures:", *paths, sep="\n    ")
    return monthly


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--station", choices=STATIONS, help="one station code (default: every selected station)")
    args = parser.parse_args()
    codes = [args.station] if args.station else selected_codes()
    monthlies = {code: run_station(code) for code in codes}
    if not args.station:
        path = figure_dir("deep_dive", "comparison") / "monthly_broken_vi.png"
        plot_comparison(monthlies, path)
        print("\ncomparison figure:", path)


if __name__ == "__main__":
    main()
