"""rank the 7 SURFRAD stations by cloud variability and pick three for the in-depth study.

six metrics per station over the 2025 daylight data, all with alpha-corrected k:
  1. median daily Stein variability index
  2. median std of 1-min k inside each 10-min interval (variability GOES can't resolve)
  3. share of 10-min intervals classed "broken" (classify_intervals)
  4. share of days classed "variable" (classify_days)
  5. share of daylight minutes where reference PV power moves > RAMP_THRESHOLD of capacity in 10 min
  6. RMSE of 10-min k persistence at 1 h
each metric is ranked (1 = most variable), composite = mean rank, ties broken by the broken share.
the top three are picked, skipping any station whose daily k correlates above MAX_DAILY_K_CORR with
one already picked (near-duplicate weather), and each pick gets a one-line reason.

tables go to <DATA_DIR>/stations/selection/, figures to <FIGURES_DIR>/solar_forecasting/selection/.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    CLEAR_DAY_VI,
    PV_PDC0,
    STATION_NAMES,
    STATIONS,
    alpha_corrected_k,
    classify_days,
    classify_intervals,
    daylight,
    figure_dir,
    load_all_stations,
    table_dir,
    to_10min,
)

# settings
N_PICK = 3
RAMP_MINUTES = 10
RAMP_THRESHOLD = 0.2  # |dP| / PV_PDC0 over RAMP_MINUTES
PERSISTENCE_LAG = pd.Timedelta("1h")
MAX_DAILY_K_CORR = 0.7  # picks correlating above this are treated as near-duplicates
TABLE_DIR = table_dir("selection")
PLOT_DIR = figure_dir("selection")
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script
GREY = "0.82"
CLIMATE = {  # short climate description for the reasons
    "bon": "humid continental Midwest cropland, frontal and summer convective cloud",
    "dra": "Mojave desert, mostly clear skies",
    "fpk": "semi-arid northern Great Plains",
    "gwn": "humid subtropical Deep South, frequent summer convection",
    "psu": "humid continental Appalachian valley, persistent stratocumulus",
    "sxf": "continental northern plains",
    "tbl": "semi-arid Front Range foothills, frequent afternoon orographic convection",
}
# key: (column, axis label, value format)
METRICS = {
    "vi": ("median_daily_vi", "median daily Stein VI (–)", "{:.2f}"),
    "kstd": ("median_k_std_1min", "median 1-min k std in 10-min interval (–)", "{:.3f}"),
    "broken": ("broken_share", "10-min intervals classed broken (%)", "{:.1%}"),
    "variable": ("variable_day_share", "days classed variable (%)", "{:.1%}"),
    "ramps": ("ramp_share", f"daylight minutes with 10-min PV ramp > {RAMP_THRESHOLD:.0%} of capacity (%)", "{:.2%}"),
    "persistence": ("persistence_rmse_1h", "1-h persistence RMSE of 10-min k (–)", "{:.3f}"),
}
METRIC_TITLES = {
    "vi": "Daily Stein variability index",
    "kstd": "Sub-10-min variability of k",
    "broken": "Broken-cloud 10-min intervals",
    "variable": "Variable days",
    "ramps": "Large PV power ramps",
    "persistence": "1-h persistence forecast error",
}
METRIC_SHORT = {"vi": "daily VI", "kstd": "1-min k std", "broken": "broken share", "variable": "variable days",
                "ramps": "PV ramps", "persistence": "persistence error"}

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


def ramp_share(df):
    """share of 1-min daylight pairs (t, t + RAMP_MINUTES) with |dP| / PV_PDC0 > RAMP_THRESHOLD."""
    power = df["power"].where(daylight(df))
    dp = (power.shift(-RAMP_MINUTES, freq="min").reindex(power.index) - power).dropna()
    return (dp.abs() / PV_PDC0 > RAMP_THRESHOLD).mean()


def persistence_rmse(k):
    """RMSE of k(t + PERSISTENCE_LAG) - k(t) over pairs where both 10-min times are valid daylight rows."""
    later = k.reindex(k.index + PERSISTENCE_LAG).to_numpy()
    err = later - k.to_numpy()
    return np.sqrt(np.nanmean(err ** 2))


def station_metrics(code, meta, df):
    alpha = df.attrs["clearsky_alpha"]
    k10 = to_10min(df)
    days = classify_days(df)
    k = alpha_corrected_k(k10, alpha)
    row = {"code": code, "name": STATION_NAMES[code], "latitude": meta["latitude"], "longitude": meta["longitude"],
           "clearsky_alpha": alpha, "n_days": len(days), "n_intervals": len(k10),
           "median_daily_vi": days["vi"].median(),
           "median_k_std_1min": (k10["k_std_1min"] / alpha).median(),
           "broken_share": (classify_intervals(k10, alpha) == "broken").mean(),
           "variable_day_share": (days["day_type"] == "variable").mean(),
           "ramp_share": ramp_share(df),
           "persistence_rmse_1h": persistence_rmse(k)}
    return row, days["k"].rename(code)


def rank(table):
    """per-metric ranks (1 = most variable), composite mean rank and composite position."""
    out = table.copy()
    for key, (col, _, _) in METRICS.items():
        out[f"rank_{key}"] = out[col].rank(ascending=False, method="min").astype(int)
    out["composite"] = out[[f"rank_{key}" for key in METRICS]].mean(axis=1)
    out = out.sort_values(["composite", "broken_share"], ascending=[True, False])
    out["composite_rank"] = np.arange(1, len(out) + 1)
    return out


def select(ranking, corr):
    """walk down the composite ranking, skipping stations correlated > MAX_DAILY_K_CORR with a pick."""
    picks, notes = [], []
    for code in ranking.index:
        clash = [p for p in picks if corr.loc[code, p] > MAX_DAILY_K_CORR]
        if clash:
            notes.append(f"{label(code)} (composite #{ranking.loc[code, 'composite_rank']}) skipped: daily k "
                         f"correlation {corr.loc[code, clash[0]]:.2f} with {label(clash[0])} > {MAX_DAILY_K_CORR}")
            continue
        picks.append(code)
        if len(picks) == N_PICK:
            break
    return picks, notes


def reason(code, r, corr, picks):
    top = [METRIC_SHORT[key] for key in METRICS if r[f"rank_{key}"] == 1]
    others = [p for p in picks if p != code]
    max_corr = max(corr.loc[code, p] for p in others) if others else np.nan
    parts = [f"composite #{r['composite_rank']} (mean rank {r['composite']:.2f})"
             + (f", #1 on {', '.join(top)}" if top else "")]
    parts.append(f"{r['broken_share']:.0%} broken 10-min intervals, median daily VI {r['median_daily_vi']:.1f}, "
                 f"{r['variable_day_share']:.0%} variable days, 1-h persistence RMSE {r['persistence_rmse_1h']:.2f}")
    parts.append(f"daily k correlation with the other picks ≤ {max_corr:.2f}")
    return f"{CLIMATE[code][0].upper()}{CLIMATE[code][1:]}; " + "; ".join(parts) + "."


def leave_one_out(table):
    """top N_PICK by composite with each metric dropped in turn, to show how stable the picks are."""
    out = {}
    for drop in METRICS:
        ranks = table[[f"rank_{k}" for k in METRICS if k != drop]].mean(axis=1)
        order = pd.DataFrame({"c": ranks, "b": table["broken_share"]}).sort_values(["c", "b"], ascending=[True, False])
        out[drop] = list(order.index[:N_PICK])
    return out


def bar_colors(codes, picks):
    return [COLORS[c] if c in picks else GREY for c in codes]


def pick_legend(picks):
    return [Patch(color=COLORS[c], label=f"{label(c)} (selected)") for c in picks] + \
           [Patch(color=GREY, label="not selected (grey)")]


def plot_metric_panels(ranking, picks):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    for ax, (key, (col, xlabel, fmt)) in zip(axes.flat, METRICS.items()):
        order = ranking.sort_values(col, ascending=True)  # most variable at the top
        y = np.arange(len(order))
        values = order[col].to_numpy()
        scale = 100 if "%" in xlabel else 1
        ax.barh(y, values * scale, color=bar_colors(order.index, picks), edgecolor="0.3", lw=0.6)
        for yi, (code, v) in zip(y, order[col].items()):
            ax.text(v * scale, yi, f"  {fmt.format(v)}  (#{order.loc[code, f'rank_{key}']})", va="center", fontsize=11)
        ax.set_yticks(y, [label(c) for c in order.index], fontsize=11)
        for tick, code in zip(ax.get_yticklabels(), order.index):
            tick.set_color(COLORS[code] if code in picks else "0.35")
        ax.set_xlim(0, values.max() * scale * 1.45)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f"{METRIC_TITLES[key]}: {order.index[-1].upper()} highest", fontsize=13)
        ax.tick_params(axis="x", labelsize=10)
    names = ", ".join(STATION_NAMES[c] for c in picks)
    fig.suptitle(f"{names} are the most cloud-variable SURFRAD stations across the six metrics "
                 "(bars: value and rank, #1 = most variable)", fontsize=15, x=0.01, ha="left")
    fig.legend(handles=pick_legend(picks), loc="lower center", ncol=N_PICK + 1, fontsize=12,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = PLOT_DIR / "metric_panels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rank_heatmap(ranking, picks):
    keys = list(METRICS)
    ranks = ranking[[f"rank_{k}" for k in keys] + ["composite"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(ranks, cmap="cividis_r", vmin=1, vmax=len(ranking), aspect="auto")
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            text = f"{ranks[i, j]:.2f}" if j == len(keys) else f"{ranks[i, j]:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    color="white" if ranks[i, j] > 4.5 else "black")
    ax.set_xticks(range(len(keys) + 1), [textwrap.fill(METRIC_TITLES[k], 13) for k in keys]
                  + ["composite\n(mean rank)"], fontsize=10)
    ax.set_yticks(range(len(ranking)), [f"#{r}  {label(c)}" for c, r in ranking["composite_rank"].items()], fontsize=11)
    for tick, code in zip(ax.get_yticklabels(), ranking.index):
        tick.set_color(COLORS[code] if code in picks else "0.35")
    ax.axvline(len(keys) - 0.5, color="white", lw=3)
    ax.axhline(N_PICK - 0.5, color="red", lw=2, ls="--")
    cbar = fig.colorbar(im, ax=ax, ticks=range(1, len(ranking) + 1))
    cbar.set_label("rank (1 = most variable)", fontsize=11)
    comp = ranking["composite"]
    ax.set_title(f"{STATION_NAMES[ranking.index[0]]} ranks near the top on every metric; #2–#4 are within "
                 f"{comp.iloc[3] - comp.iloc[1]:.2f} mean rank\n(rank per metric, rows sorted by composite, "
                 f"dashed line = top {N_PICK})", fontsize=13)
    path = PLOT_DIR / "rank_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation(corr, ranking, picks):
    order = list(ranking.index)
    c = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > 0.6 else "black",
                    fontweight="bold" if (order[i] in picks and order[j] in picks and i != j) else "normal")
    ax.set_xticks(range(len(order)), [code.upper() for code in order], fontsize=11)
    ax.set_yticks(range(len(order)), [label(code) for code in order], fontsize=11)
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, code in zip(ticks, order):
            tick.set_color(COLORS[code] if code in picks else "0.35")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson correlation of daily corrected k (–)", fontsize=11)
    off = c.where(~np.eye(len(order), dtype=bool))
    pick_max = off.loc[picks, picks].max().max()
    verdict = "no pair of picks is a near-duplicate" if pick_max <= MAX_DAILY_K_CORR else "near-duplicate picks"
    ax.set_title(f"Daily cloudiness is largely independent between stations\n(max off-diagonal {off.max().max():.2f}; "
                 f"picks ≤ {pick_max:.2f} < {MAX_DAILY_K_CORR}: {verdict})", fontsize=13)
    path = PLOT_DIR / "daily_k_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(ranking, picks, reasons, notes):
    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.08)
    ax = fig.add_subplot(grid[0])
    order = ranking.iloc[::-1]  # composite #1 at the top
    y = np.arange(len(order))
    ax.barh(y, order["composite"], color=bar_colors(order.index, picks), edgecolor="0.3", lw=0.6)
    for yi, (code, r) in zip(y, order.iterrows()):
        ax.text(r["composite"], yi, f"  {r['composite']:.2f}  (#{r['composite_rank']})", va="center", fontsize=12)
    ax.set_yticks(y, [label(c) for c in order.index], fontsize=12)
    for tick, code in zip(ax.get_yticklabels(), order.index):
        tick.set_color(COLORS[code] if code in picks else "0.35")
        tick.set_fontweight("bold" if code in picks else "normal")
    ax.set_xlim(0, len(ranking) + 1.2)
    ax.set_xlabel("composite = mean rank over 6 variability metrics\n(1 = most variable on every metric)", fontsize=12)
    ax.legend(handles=pick_legend(picks), loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, fontsize=11,
              frameon=False)

    text_ax = fig.add_subplot(grid[1])
    text_ax.axis("off")
    yy = 1.0
    for code in picks:
        lines = textwrap.wrap(reasons[code], 66)
        text_ax.text(0.0, yy, f"{label(code)}", color=COLORS[code], fontsize=14, fontweight="bold",
                     va="top", transform=text_ax.transAxes)
        text_ax.text(0.0, yy - 0.06, "\n".join(lines), fontsize=12, va="top", transform=text_ax.transAxes)
        yy -= 0.11 + 0.047 * len(lines)
    comp, broken = ranking["composite"], ranking["broken_share"]
    footnote = (f"Ties in composite are broken by the broken-cloud share. #{N_PICK} "
                f"{ranking.index[N_PICK - 1].upper()} ({comp.iloc[N_PICK - 1]:.2f}, {broken.iloc[N_PICK - 1]:.1%} "
                f"broken) is only {comp.iloc[N_PICK] - comp.iloc[N_PICK - 1]:.2f} mean rank ahead of #{N_PICK + 1} "
                f"{ranking.index[N_PICK].upper()} ({comp.iloc[N_PICK]:.2f}). "
                + ("; ".join(notes) if notes else f"No near-duplicate swap: all picks have daily k correlation "
                   f"≤ {MAX_DAILY_K_CORR}."))
    text_ax.text(0.0, yy - 0.02, "\n".join(textwrap.wrap(footnote, 78)), fontsize=11, color="0.3", va="top",
                 style="italic", transform=text_ax.transAxes)
    names = ", ".join(STATION_NAMES[c] for c in picks[:-1]) + f" and {STATION_NAMES[picks[-1]]}"
    fig.suptitle(f"{names} have the most variable cloud cover of the 7 SURFRAD stations (2025)",
                 fontsize=16, x=0.01, ha="left", y=0.99)
    path = PLOT_DIR / "selection_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    stations = load_all_stations()
    rows, daily_k = zip(*(station_metrics(code, meta, df) for code, (meta, df) in stations.items()))
    ranking = rank(pd.DataFrame(rows).set_index("code"))
    corr = pd.concat(daily_k, axis=1, sort=True).corr()
    picks, notes = select(ranking, corr)
    reasons = {code: reason(code, ranking.loc[code], corr, picks) for code in picks}

    ranking.to_csv(TABLE_DIR / "station_ranking.csv")
    corr.round(3).to_csv(TABLE_DIR / "daily_k_correlation.csv")
    rank_cols = [f"rank_{key}" for key in METRICS]
    selected = ranking.loc[picks, ["name", "latitude", "longitude", "composite", "composite_rank"] + rank_cols].copy()
    selected["reason"] = [reasons[c] for c in picks]
    selected.to_csv(TABLE_DIR / "selected_stations.csv")

    pd.set_option("display.width", 200)
    print(f"day types use CLEAR_DAY_VI = {CLEAR_DAY_VI}")
    show = ranking[[col for col, _, _ in METRICS.values()] + rank_cols + ["composite", "composite_rank"]]
    print(show.round(3).to_string())
    print("\ndaily corrected k correlation\n", corr.round(2).to_string())
    print("\nnear-duplicate check:", "; ".join(notes) if notes else
          f"no swap, every pair of picks correlates <= {MAX_DAILY_K_CORR}")
    print("\ntop 3 with one metric dropped")
    for drop, top in leave_one_out(ranking).items():
        print(f"  without {drop:12s} {', '.join(c.upper() for c in top)}")
    print("\nselected")
    for code in picks:
        print(f"  {label(code)}: {reasons[code]}")
    paths = [plot_metric_panels(ranking, picks), plot_rank_heatmap(ranking, picks),
             plot_correlation(corr, ranking, picks), plot_summary(ranking, picks, reasons, notes)]
    print("\ntables in", TABLE_DIR)
    print("figures:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
