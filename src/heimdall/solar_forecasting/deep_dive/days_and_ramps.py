"""each selected station day by day: daily variability calendar, PV ramps and six case days.

per station (2025, k corrected by the station's clearsky_alpha, dates are local solar dates):
  - calendar (month x day of month) of the daily Stein VI with the day type marked, and day-type
    counts by month
  - ramp exceedance: share of daylight minute pairs (t, t + lag) where the reference PV power moves by
    more than a given % of capacity (PV_PDC0), lags 1 / 10 / 60 min, by season (DJF/MAM/JJA/SON), against
    the same curve for clear-sky power (sun angle only)
  - six case days, chosen by the rule shared with the satellite analysis (case_days):
      days from classify_days with n_minutes >= 0.9 x that day's daylight minutes;
      warm season = Apr-Sep, cold season = Oct-Mar; per season
      clear: lowest-VI clear day, overcast: lowest-k overcast day, variable: highest-VI variable day;
      a season with no day of a type takes the next best day of that type from the other season.

    python days_and_ramps.py [--station CODE]   (default: every code in selected_stations.csv)

tables go to <DATA_DIR>/stations/deep_dive/<code>/, figures to <FIGURES_DIR>/solar_forecasting/deep_dive/<code>/.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    PV_PDC0,
    STATION_NAMES,
    STATIONS,
    alpha_corrected_k,
    classify_days,
    daylight,
    figure_dir,
    load_station_1min,
    pv_power,
    solar_date,
    table_dir,
)

# settings
SELECTED = table_dir("selection") / "selected_stations.csv"
MIN_COVERAGE = 0.9  # case days: valid minutes / daylight minutes
WARM_MONTHS = range(4, 10)  # Apr-Sep; cold = Oct-Mar
CASE_RULES = {  # day type: (column, ascending) to rank candidates, best first
    "clear": ("vi", True), "variable": ("vi", False), "overcast": ("k", True)}
DAY_TYPES = ["clear", "variable", "overcast"]
DAY_COLORS = {"clear": "#F0C419", "variable": "#2A9D8F", "overcast": "0.45"}
DAY_MARK = {"clear": "C", "overcast": "O"}
RAMP_LAGS = [1, 10, 60]  # minutes
RAMP_THRESHOLDS = np.arange(0, 100.01, 0.25)  # % of capacity
RAMP_TABLE_THRESHOLDS = [5, 10, 20, 30, 50]
RAMP_HEADLINE = 20  # % of capacity, the selection metric's threshold
CS_TEMP_AIR, CS_WIND = 20.0, 1.0  # °C, m s^-1: fixed weather for the sun-angle-only power curve
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_COLORS = {"DJF": "#0072B2", "MAM": "#009E73", "JJA": "#D55E00", "SON": "#CC79A7"}  # Okabe-Ito
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


def selected_codes():
    return list(pd.read_csv(SELECTED)["code"])


def half_year(month):
    return "warm" if month in WARM_MONTHS else "cold"


def meteo_season(month):
    return next(s for s, months in SEASONS.items() if month in months)


def day_table(df):
    """classify_days plus daylight minutes, coverage, warm/cold season and meteorological season."""
    days = classify_days(df)
    date = solar_date(df.index, df.attrs["longitude"])
    days["daylight_minutes"] = daylight(df).groupby(date).sum().reindex(days.index)
    days["coverage"] = days["n_minutes"] / days["daylight_minutes"]
    days["season"] = [half_year(m) for m in days.index.month]
    days["meteo_season"] = [meteo_season(m) for m in days.index.month]
    return days


def case_days(days):
    """the six case days (see module docstring), one row per (day_type, season slot)."""
    pool = days[days["n_minutes"] >= MIN_COVERAGE * days["daylight_minutes"]]
    rows, notes = [], []
    for t, (col, ascending) in CASE_RULES.items():
        ranked = pool[pool["day_type"] == t].sort_values(col, ascending=ascending)
        own = {s: ranked[ranked["season"] == s] for s in ("warm", "cold")}
        chosen = {s: own[s].index[0] for s in own if len(own[s])}
        for s in ("warm", "cold"):
            if s not in chosen:
                other = ranked.drop(index=list(chosen.values()))
                if len(other):
                    chosen[s] = other.index[0]
                    notes.append(f"no {s}-season {t} day; took {chosen[s]:%Y-%m-%d} from the other season")
        for s in ("warm", "cold"):
            if s in chosen:
                d = chosen[s]
                rows.append({"date": d, "season": days.loc[d, "season"], "slot": s, "day_type": t,
                             "k": days.loc[d, "k"], "vi": days.loc[d, "vi"]})
    return pd.DataFrame(rows), notes


def ramp_changes(power, lag):
    """|P(t + lag) - P(t)| / PV_PDC0 in % over daylight minute pairs, indexed by t."""
    later = power.shift(-lag, freq="min").reindex(power.index)
    return ((later - power).abs() / PV_PDC0 * 100).dropna()


def exceedance(values):
    """share of values > each RAMP_THRESHOLDS entry."""
    v = np.sort(values.to_numpy())
    return 1 - np.searchsorted(v, RAMP_THRESHOLDS, side="right") / len(v)


def sun_only_power(meta, df):
    """clear-sky power with fixed air temperature and wind: df["power_cs"] uses the measured weather,
    whose gusts and gap-filled temperatures move cell temperature by several % of capacity per minute."""
    location = pvlib.location.Location(meta["latitude"], meta["longitude"], altitude=meta["altitude"])
    solpos = location.get_solarposition(df.index)
    const = pd.Series(1.0, index=df.index)
    return pv_power(location, solpos, df["ghi_cs"], df["dni_cs"], df["dhi_cs"], const * CS_TEMP_AIR, const * CS_WIND)


def ramp_tables(meta, df):
    """exceedance curves {lag: {season or "year" or "clear sky": array}} plus a long table."""
    lon = df.attrs["longitude"]
    day = daylight(df)
    power_sun = sun_only_power(meta, df).where(day)
    curves, rows = {}, []
    for lag in RAMP_LAGS:
        dp = ramp_changes(df["power"].where(day), lag)
        dp_cs = ramp_changes(power_sun, lag)
        season = pd.Series([meteo_season(m) for m in solar_date(dp.index, lon).month], index=dp.index)
        groups = {s: dp[season == s] for s in SEASONS} | {"year": dp, "clear sky": dp_cs}
        curves[lag] = {name: exceedance(v) for name, v in groups.items()}
        for name, v in groups.items():
            for thr in RAMP_TABLE_THRESHOLDS:
                rows.append({"lag_min": lag, "group": name, "threshold_pct": thr,
                             "exceedance": (v > thr).mean(), "max_pct": v.max(), "n_pairs": len(v)})
    return curves, pd.DataFrame(rows)


def plot_calendar(code, days, path):
    grid = np.full((12, 31), np.nan)
    for d, vi in days["vi"].items():
        grid[d.month - 1, d.day - 1] = vi
    counts = pd.crosstab(days.index.month, days["day_type"]).reindex(index=range(1, 13), columns=DAY_TYPES,
                                                                      fill_value=0)
    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[31, 7.5], height_ratios=[12, 0.45], wspace=0.03, hspace=0.32)
    ax, bx, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    im = ax.imshow(np.clip(grid, 1, 30), cmap="viridis", norm=LogNorm(1, 30), aspect="auto")
    for d in pd.date_range(days.index.min(), "2025-12-31"):
        if d not in days.index:  # real date with too little data
            ax.text(d.day - 1, d.month - 1, "–", ha="center", va="center", fontsize=11, color="0.5")
    for d, t in days["day_type"].items():
        if t in DAY_MARK:
            ax.text(d.day - 1, d.month - 1, DAY_MARK[t], ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
    ax.set_xticks(range(31), [str(i) if i % 2 else "" for i in range(1, 32)])
    ax.set_xticks(np.arange(-0.5, 31), minor=True)
    ax.set_yticks(np.arange(-0.5, 12), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(12), MONTHS)
    ax.set_xlabel("day of month (local solar date)", fontsize=12)
    ax.set_ylabel("month (2025)", fontsize=12)
    ax.set_title("Daily Stein VI (colour) — C = clear day, O = overcast day, unmarked = variable, – = too little data",
                 fontsize=12)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_minor_locator(NullLocator())
    cbar.set_ticks([1, 1.5, 2, 3, 5, 10, 20, 30], labels=["1", "1.5", "2", "3", "5", "10", "20", "≥30"])
    cbar.set_label("daily Stein variability index (–, log scale; 1 = smooth as clear sky, clear days < 1.5)",
                   fontsize=12)

    left = np.zeros(12)
    y = np.arange(12)
    for t in DAY_TYPES:
        v = counts[t].to_numpy()
        bx.barh(y, v, left=left, color=DAY_COLORS[t], edgecolor="white", lw=0.8, label=t)
        for yi, (lft, n) in enumerate(zip(left, v)):
            if n >= 2:
                bx.text(lft + n / 2, yi, f"{n}", ha="center", va="center", fontsize=10,
                        color="black" if t == "clear" else "white")
        left += v
    bx.set_ylim(11.5, -0.5)
    bx.set_yticks(y, [])
    bx.set_xlim(0, 31)
    bx.set_xlabel("days per month", fontsize=12)
    bx.set_title("Day types by month", fontsize=12)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=11,
              handlelength=1.2, columnspacing=0.8)

    n = days["day_type"].value_counts().reindex(DAY_TYPES, fill_value=0)
    clear_months = counts["clear"].sort_values(ascending=False)
    top = [MONTHS[m - 1] for m, c in clear_months.items() if c > 0][:2]
    fig.suptitle(f"{STATION_NAMES[code]} has variable sky on {n['variable']} of {len(days)} days; only {n['clear']} "
                 f"days are clear (most in {' and '.join(top)}) and {n['overcast']} overcast "
                 f"(median daily VI {days['vi'].median():.2f})\n{label(code)}, 2025", fontsize=15, x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ramps(code, curves, table, path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    xmax = {1: 50, 10: 80, 60: 100}
    for ax, lag in zip(axes, RAMP_LAGS):
        c = curves[lag]
        for s in SEASONS:
            ax.plot(RAMP_THRESHOLDS, c[s] * 100, color=SEASON_COLORS[s], lw=2, label=s)
        ax.plot(RAMP_THRESHOLDS, c["year"] * 100, color="black", lw=2.6, label="whole year")
        cs = c["clear sky"]
        ax.plot(RAMP_THRESHOLDS, np.where(cs > 0, cs * 100, np.nan), color="0.4", lw=2.4, ls="--",
                label="clear sky (sun angle only)")
        t = table[(table["lag_min"] == lag)].set_index(["group", "threshold_pct"])
        year = t.loc[("year", RAMP_HEADLINE), "exceedance"]
        cs_max = t.loc[("clear sky", RAMP_HEADLINE), "max_pct"]
        ax.axvline(RAMP_HEADLINE, color="0.6", lw=1, ls=":")
        ax.plot(RAMP_HEADLINE, year * 100, "o", color="black", ms=9, zorder=5)
        ax.annotate(f"{year:.1%} of minutes\n> {RAMP_HEADLINE}% of capacity", (RAMP_HEADLINE, year * 100),
                    xytext=(12, 6), textcoords="offset points", fontsize=11, va="bottom")
        ax.annotate(f"sun angle alone:\nmax {cs_max:.1f}% in {lag} min", (cs_max, 1e-3),
                    xytext=(8, 0), textcoords="offset points", fontsize=11, color="0.3", va="center")
        ax.axvline(cs_max, ymax=0.15, color="0.4", lw=1.5, ls="--")
        peak = max(SEASONS, key=lambda s: t.loc[(s, RAMP_HEADLINE), "exceedance"])
        ax.set_title(f"{lag}-min ramps: most in {peak} "
                     f"({t.loc[(peak, RAMP_HEADLINE), 'exceedance']:.1%} > {RAMP_HEADLINE}%)", fontsize=13)
        ax.set_xlim(0, xmax[lag])
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 300)
        ax.set_xlabel(f"|ΔP| over {lag} min (% of {PV_PDC0 / 1000:.0f} kW capacity)", fontsize=12)
        ax.grid(alpha=0.3, which="major")
    axes[0].set_ylabel("daylight minutes with a larger ramp (%, log scale)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.07))
    t10 = table[table["lag_min"] == 10].set_index(["group", "threshold_pct"])
    fig.suptitle(f"At {STATION_NAMES[code]}, clouds swing PV output by > {RAMP_HEADLINE}% of capacity within "
                 f"10 min in {t10.loc[('year', RAMP_HEADLINE), 'exceedance']:.1%} of daylight minutes; "
                 f"the sun's motion alone never exceeds {t10.loc[('clear sky', RAMP_HEADLINE), 'max_pct']:.1f}%"
                 f"\n{label(code)}, 2025, 5 kW fixed-tilt reference system; ramp share by season (DJF/MAM/JJA/SON)",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def day_slice(df, d):
    g = df[solar_date(df.index, df.attrs["longitude"]) == d]
    hours = (g.index.tz_localize(None) - d).total_seconds() / 3600 + df.attrs["longitude"] / 15
    return g, np.asarray(hours)


def plot_case_days(code, df, cases, path):
    alpha = df.attrs["clearsky_alpha"]
    fig = plt.figure(figsize=(20, 12.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[2.2, 1, 2.2, 1], hspace=0.12, wspace=0.08)
    ghi_max = 0
    ghi_axes, swing = [], {}
    for j, t in enumerate(DAY_TYPES):
        for i, slot in enumerate(("warm", "cold")):
            row = cases[(cases["day_type"] == t) & (cases["slot"] == slot)]
            ax = fig.add_subplot(gs[2 * i, j])
            kx = fig.add_subplot(gs[2 * i + 1, j], sharex=ax)
            ghi_axes.append(ax)
            if row.empty:
                ax.axis("off")
                kx.axis("off")
                continue
            r = row.iloc[0]
            g, h = day_slice(df, r["date"])
            day = daylight(g).to_numpy()
            ax.plot(h, g["ghi_cs"] * alpha, color="0.45", lw=1.8, label="clear-sky GHI × station alpha")
            ax.plot(h, g["ghi"], color=COLORS[code], lw=1.1, label="measured GHI (1 min)")
            ghi_max = max(ghi_max, np.nanmax(g["ghi"]), np.nanmax(g["ghi_cs"] * alpha))
            k = alpha_corrected_k(g["k"], alpha)
            kx.plot(h[day], k.to_numpy()[day], color=COLORS[code], lw=1.1)
            kx.axhline(1, color="0.45", lw=1.2, ls="--")
            jump = g["ghi"].where(daylight(g)).diff().abs().max()
            swing[(t, slot)] = jump
            ax.text(0.02, 0.95, f"daily k {r['k']:.2f}\nVI {r['vi']:.2f}\nmax 1-min ΔGHI {jump:.0f} W m$^{{-2}}$",
                    transform=ax.transAxes, va="top", fontsize=11,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
            fallback = " (other-season fallback)" if r["season"] != slot else ""
            ax.set_title(f"{t} – {r['date']:%a %d %b %Y}, {r['season']} season{fallback}", fontsize=12)
            ax.set_xlim(4, 20)
            kx.set_ylim(0, 1.6)
            kx.set_yticks([0, 0.5, 1, 1.5])
            plt.setp(ax.get_xticklabels(), visible=False)
            if j == 0:
                ax.set_ylabel("GHI (W m$^{-2}$)", fontsize=12)
                kx.set_ylabel("corrected k (–)", fontsize=12)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(kx.get_yticklabels(), visible=False)
            if i == 1:
                kx.set_xlabel("local solar time (h)", fontsize=12)
            else:
                plt.setp(kx.get_xticklabels(), visible=False)
    for ax in ghi_axes:
        ax.set_ylim(0, ghi_max * 1.32)  # headroom for the info box
    handles, labels = ghi_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.01))
    var = cases[cases["day_type"] == "variable"].set_index("slot")
    clear = cases[cases["day_type"] == "clear"].set_index("slot")
    vw = var.loc["warm"] if "warm" in var.index else var.iloc[0]
    cw = clear.loc["warm"] if "warm" in clear.index else clear.iloc[0]
    fig.suptitle(f"{STATION_NAMES[code]}'s most variable warm-season day ({vw['date']:%d %b}, VI {vw['vi']:.1f}) "
                 f"jumps by up to {swing.get(('variable', 'warm'), np.nan):.0f} W m$^{{-2}}$ in a minute; its clearest "
                 f"({cw['date']:%d %b}, VI {cw['vi']:.2f}) tracks the clear sky\n{label(code)}, 2025 case days: "
                 f"columns clear / variable / overcast, rows warm (Apr–Sep) / cold (Oct–Mar) season",
                 fontsize=15, x=0.01, ha="left", y=0.97)
    fig.subplots_adjust(top=0.885)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_station(code):
    meta, df = load_station_1min(code)
    days = day_table(df)
    cases, notes = case_days(days)
    curves, ramps = ramp_tables(meta, df)
    fig_dir, tab_dir = figure_dir("deep_dive", code), table_dir("deep_dive", code)
    days.round(4).to_csv(tab_dir / "daily_day_types.csv")
    ramps.to_csv(tab_dir / "ramp_exceedance.csv", index=False)
    out = cases[["date", "season", "day_type", "k", "vi"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.round(4).to_csv(tab_dir / "case_days.csv", index=False)
    paths = [fig_dir / "daily_vi_calendar.png", fig_dir / "ramp_exceedance.png", fig_dir / "case_days.png"]
    plot_calendar(code, days, paths[0])
    plot_ramps(code, curves, ramps, paths[1])
    plot_case_days(code, df, cases, paths[2])

    print(f"\n{label(code)}")
    print("  day types:", days["day_type"].value_counts().reindex(DAY_TYPES).to_dict(),
          f"median daily VI {days['vi'].median():.2f}")
    for col in ("season", "meteo_season"):
        print(pd.crosstab(days[col], days["day_type"]).reindex(columns=DAY_TYPES, fill_value=0).to_string())
    show = ramps[ramps["threshold_pct"].isin([10, 20, 30])].pivot_table(
        index=["lag_min", "group"], columns="threshold_pct", values="exceedance")
    print((show * 100).round(2).to_string())
    print("  clear-sky max ramp (% capacity):",
          ramps[ramps["group"] == "clear sky"].groupby("lag_min")["max_pct"].first().round(2).to_dict())
    print("  case days" + (" (" + "; ".join(notes) + ")" if notes else "") + ":")
    print(out.round(3).to_string(index=False))
    print("  figures:", *paths, sep="\n    ")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--station", choices=STATIONS, help="one station code (default: every selected station)")
    args = parser.parse_args()
    for code in [args.station] if args.station else selected_codes():
        run_station(code)


if __name__ == "__main__":
    main()
