"""reference forecasts of 10-min GHI at the selected stations, and how much worse they get on cloudy days.

on the 10-min series (to_10min, daylight only, alpha-corrected k), issue times t and targets t + h for
h = 10 min ... 3 h, both in daylight on the same solar day. baselines for GHI(t + h):
  persistence        GHI(t)
  smart persistence  k(t) GHI_cs(t + h)                  (k persists, the clear-sky diurnal cycle does not)
  climatology        k_clim(month, solar hour) GHI_cs(t + h), k_clim = median corrected k at that solar hour
                     in the two neighbouring months (target month left out, so it is out of sample)
  blend              k_clim + w_h (k(t) - k_clim), w_h fitted per horizon on the other 11 months
                     (least squares in GHI, leave-one-month-out)
metrics: RMSE, MAE (W m^-2), nRMSE (% of mean measured GHI in the same subset) and skill
1 - RMSE / RMSE_climatology, split by the day type of the day (classify_days) and the sky class of the
target interval (classify_intervals). every baseline is scored on the same issue/target pairs.

per-station tables go to <DATA_DIR>/stations/deep_dive/<code>/, figures to
<FIGURES_DIR>/solar_forecasting/deep_dive/<code>/; cross-station outputs to .../deep_dive/comparison/.
Desert Rock (dra) is run as a clear-site reference for the comparison only.

    uv run python src/heimdall/solar_forecasting/deep_dive/forecast_baselines.py [--station CODE]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
STEP = pd.Timedelta("10min")
HORIZONS = range(1, 19)  # in STEPs: 10 min ... 3 h
KEY_HORIZONS = {6: "1 h", 18: "3 h"}
REFERENCE = "dra"  # clear-sky site, comparison only
BASELINES = {  # key: (label, line style, bar hatch, bar alpha)
    "persistence": ("persistence", ":", "..", 0.35),
    "smart": ("smart persistence", "-", "", 1.0),
    "clim": ("climatology", "--", "//", 0.55),
    "blend": ("blend", "-.", "xx", 0.75),
}
DAY_TYPES = ["clear", "variable", "overcast"]
SKY_CLASSES = ["clear", "broken", "overcast"]
EXAMPLE_MONTHS = range(4, 10)  # long days for the example plots
MIN_EXAMPLE_COVERAGE = 0.9  # share of the month's fullest day's 1-h pairs
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script
SELECTED = table_dir("selection") / "selected_stations.csv"

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


def hours(steps):
    return steps * STEP / pd.Timedelta("1h")


def series_10min(df):
    """10-min daylight rows with corrected k, sky class, day type, solar date / hour and the
    out-of-sample climatological corrected k."""
    alpha, lon = df.attrs["clearsky_alpha"], df.attrs["longitude"]
    k10 = to_10min(df)
    k10 = k10[k10["ghi"].notna() & (k10["ghi_cs"] > 0)].copy()
    k10["kc"] = alpha_corrected_k(k10, alpha)
    k10["sky"] = classify_intervals(k10, alpha)
    k10["date"] = solar_date(k10.index, lon)
    k10["day_type"] = k10["date"].map(classify_days(df)["day_type"])
    k10["month"] = k10["date"].dt.month
    k10["solar_hour"] = ((k10.index + pd.to_timedelta(lon / 15, unit="h")).hour
                         + (k10.index + pd.to_timedelta(lon / 15, unit="h")).minute / 60)
    k10["hour_bin"] = np.floor(k10["solar_hour"]).astype(int)
    k10["kc_clim"] = np.nan
    for m in range(1, 13):
        near = k10[k10["month"].isin([(m - 2) % 12 + 1, m % 12 + 1])]
        clim = near.groupby("hour_bin")["kc"].median()
        rows = k10["month"] == m
        k10.loc[rows, "kc_clim"] = k10.loc[rows, "hour_bin"].map(clim).to_numpy()
    return k10


def forecast_pairs(k10, alpha):
    """one row per (issue time, horizon) with the observed target and all baseline forecasts (W m^-2)."""
    frames = []
    for h in HORIZONS:
        tgt = k10.reindex(k10.index + h * STEP)
        ok = (tgt["kc"].notna().to_numpy() & tgt["kc_clim"].notna().to_numpy()
              & (tgt["date"].to_numpy() == k10["date"].to_numpy()))
        src, tgt = k10[ok], tgt[ok]
        cs = tgt["ghi_cs"].to_numpy()
        frames.append(pd.DataFrame({
            "issue": src.index, "target": tgt.index, "h": h, "date": tgt["date"].to_numpy(),
            "month": tgt["month"].to_numpy(), "day_type": tgt["day_type"].to_numpy(), "sky": tgt["sky"].to_numpy(),
            "solar_hour": tgt["solar_hour"].to_numpy(), "ghi": tgt["ghi"].to_numpy(), "ghi_cs": cs,
            "k_issue": src["k"].to_numpy(), "k_target": tgt["k"].to_numpy(),
            "x": src["kc"].to_numpy() - tgt["kc_clim"].to_numpy(),  # anomaly at issue vs target climatology
            "y": tgt["kc"].to_numpy() - tgt["kc_clim"].to_numpy(),
            "persistence": src["ghi"].to_numpy(),
            "smart": src["kc"].to_numpy() * alpha * cs,
            "clim": tgt["kc_clim"].to_numpy() * alpha * cs,
        }))
    pairs = pd.concat(frames, ignore_index=True)
    # blend weight per horizon, leave-one-month-out least squares in GHI units (weights ghi_cs^2)
    wcs = (alpha * pairs["ghi_cs"]) ** 2
    sums = pd.DataFrame({"h": pairs["h"], "month": pairs["month"],
                         "xy": wcs * pairs["x"] * pairs["y"], "xx": wcs * pairs["x"] ** 2})
    by_month = sums.groupby(["h", "month"])[["xy", "xx"]].sum()
    total = by_month.groupby(level="h").transform("sum")
    weight = ((total["xy"] - by_month["xy"]) / (total["xx"] - by_month["xx"])).clip(0, 1).rename("w")
    pairs = pairs.join(weight, on=["h", "month"])
    pairs["blend"] = pairs["clim"] + pairs["w"] * pairs["x"] * alpha * pairs["ghi_cs"]
    return pairs


def scores(pairs, by):
    """RMSE / MAE / nRMSE / skill per horizon and group in by (None = all pairs), long format."""
    keys = ["h"] + ([by] if by else [])
    rows = []
    for key, g in pairs.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        mean_obs = g["ghi"].mean()
        rmse = {b: np.sqrt(((g[b] - g["ghi"]) ** 2).mean()) for b in BASELINES}
        for b in BASELINES:
            rows.append({"h": key[0], "horizon_min": key[0] * 10, "split": by or "all", "group": key[1] if by else "all",
                         "baseline": b, "n": len(g), "n_days": g["date"].nunique(), "mean_ghi": mean_obs,
                         "rmse": rmse[b], "mae": (g[b] - g["ghi"]).abs().mean(),
                         "nrmse": 100 * rmse[b] / mean_obs, "skill_vs_clim": 1 - rmse[b] / rmse["clim"]})
    return pd.DataFrame(rows)


def station_scores(code):
    _, df = load_station_1min(code)
    alpha = df.attrs["clearsky_alpha"]
    k10 = series_10min(df)
    pairs = forecast_pairs(k10, alpha)
    table = pd.concat([scores(pairs, None), scores(pairs, "day_type"), scores(pairs, "sky")], ignore_index=True)
    table.insert(0, "station", code)
    one_h = pairs[pairs["h"] == 6]
    k_rmse = np.sqrt(((one_h["k_target"] - one_h["k_issue"]) ** 2).mean()) / alpha
    print(f"{label(code)}: alpha {alpha:.3f}, {len(k10)} 10-min rows, {pairs['date'].nunique()} days; "
          f"1-h persistence RMSE of corrected k {k_rmse:.3f}; blend weight at 1 h / 3 h "
          f"{pairs.loc[pairs['h'] == 6, 'w'].mean():.2f} / {pairs.loc[pairs['h'] == 18, 'w'].mean():.2f}")
    return table, pairs, k10


def pick(table, split, group, baseline, h, col="nrmse"):
    row = table[(table["split"] == split) & (table["group"] == group) & (table["baseline"] == baseline)
                & (table["h"] == h)]
    return row[col].iloc[0] if len(row) else np.nan


def spread(values, gap):
    """nudge sorted label positions apart by at least gap, keeping their order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]] + gap)
    return out


# ---------- per-station figures ----------

def plot_horizon(code, table, out_dir):
    groups = [("all", "all"), ("day_type", "clear"), ("day_type", "variable"), ("day_type", "overcast")]
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    color = COLORS[code]
    for ax, (split, group) in zip(axes, groups):
        sub = table[(table["split"] == split) & (table["group"] == group)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for b, (name, ls, _, a) in BASELINES.items():
            s = sub[sub["baseline"] == b]
            ax.plot(s["horizon_min"] / 60, s["nrmse"], ls=ls, color=color, alpha=max(a, 0.5), label=name)
        smart = sub[sub["baseline"] == "smart"].set_index("h")["nrmse"]
        for h in KEY_HORIZONS:
            ax.plot(hours(h), smart[h], "o", color=color, ms=8)
            ax.annotate(f"{smart[h]:.0f}%", (hours(h), smart[h]), xytext=(-6, 9), textcoords="offset points",
                        ha="right", fontsize=12, color="black")
        n_days = sub["n_days"].iloc[0]
        ax.set_title(f"{'all days' if group == 'all' else group + ' days'} ({n_days} days)", fontsize=13)
        ax.set_xlabel("forecast horizon (h)", fontsize=12)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nRMSE of 10-min GHI (% of mean measured GHI)", fontsize=12)
    handles = [Line2D([], [], color=color, ls=ls, alpha=max(a, 0.5), label=name)
               for name, ls, _, a in BASELINES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.06), frameon=False)
    clear, var = pick(table, "day_type", "clear", "smart", 6), pick(table, "day_type", "variable", "smart", 6)
    fig.suptitle(f"{label(code)}: 1-h smart-persistence nRMSE is {clear:.0f}% on clear days and {var:.0f}% on "
                 f"variable days ({var / clear:.1f}×)  (dots: 1 h and 3 h)", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = out_dir / f"error_vs_horizon_{code}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def grouped_bars(ax, table, split, groups, h, color):
    width = 0.2
    x = np.arange(len(groups))
    vals = {b: [pick(table, split, g, b, h) for g in groups] for b in BASELINES}
    # overcast climatology nRMSE is huge (tiny mean GHI); cap the axis so the other bars stay readable
    top = 1.25 * np.nanmax([v for b, vs in vals.items() for g, v in zip(groups, vs)
                            if g != "overcast" or b in ("persistence", "smart")])
    for i, (b, (name, _, hatch, a)) in enumerate(BASELINES.items()):
        pos = x + (i - 1.5) * width
        ax.bar(pos, np.minimum(vals[b], top), width, color=color, alpha=a, hatch=hatch, edgecolor="0.25", lw=0.6)
        for p, v in zip(pos, vals[b]):
            if np.isfinite(v):
                clipped = v > top
                ax.text(p, min(v, top) * (0.97 if clipped else 1), f"{v:.0f}{'↑' if clipped else ''}",
                        ha="center", va="top" if clipped else "bottom", fontsize=10, rotation=90 if clipped else 0,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1} if clipped else None)
    n_days = [table[(table["split"] == split) & (table["group"] == g)]["n_days"].max() for g in groups]
    n = [table[(table["split"] == split) & (table["group"] == g) & (table["h"] == h)]["n"].max() for g in groups]
    ticks = [f"{g}\n({nd:.0f} days)" if split == "day_type" else f"{g}\n({nn:,.0f} targets)"
             for g, nd, nn in zip(groups, n_days, n)]
    ax.set_xticks(x, ticks, fontsize=12)
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=0.3)


def baseline_legend(color):
    return [Patch(facecolor=color, alpha=a, hatch=hatch, edgecolor="0.25", label=name)
            for name, _, hatch, a in BASELINES.values()]


def plot_splits(code, table, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    splits = [("day_type", DAY_TYPES, "day type of the day"), ("sky", SKY_CLASSES, "sky class of the target interval")]
    for r, (split, groups, what) in enumerate(splits):
        for c, (h, txt) in enumerate(KEY_HORIZONS.items()):
            ax = axes[r, c]
            grouped_bars(ax, table, split, groups, h, COLORS[code])
            ax.set_title(f"{txt} ahead, by {what}", fontsize=13)
            ax.set_ylabel("nRMSE of GHI (%)", fontsize=12)
    fig.legend(handles=baseline_legend(COLORS[code]), loc="lower center", ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    broken, clear = pick(table, "sky", "broken", "smart", 6), pick(table, "sky", "clear", "smart", 6)
    fig.suptitle(f"{label(code)}: 1-h smart persistence errs {broken:.0f}% on broken-cloud targets vs "
                 f"{clear:.0f}% on clear ones\n(nRMSE = RMSE / mean measured GHI within each group; ↑ = bar cut "
                 "off, overcast means are small so their nRMSE is inflated)",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = out_dir / f"error_by_day_type_and_sky_{code}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def example_days(pairs):
    """the clear and variable days (Apr-Sep, near-complete data) whose 1-h smart-persistence nRMSE is the
    median of their day type, so the examples are typical rather than extreme."""
    one = pairs[pairs["h"] == 6].copy()
    one["err2"] = (one["smart"] - one["ghi"]) ** 2
    daily = one.groupby("date").agg(n=("ghi", "size"), mse=("err2", "mean"), mean=("ghi", "mean"),
                                    day_type=("day_type", "first"), month=("month", "first"))
    daily["nrmse"] = 100 * np.sqrt(daily["mse"]) / daily["mean"]
    daily["coverage"] = daily["n"] / daily.groupby("month")["n"].transform("max")
    ok = daily[(daily["coverage"] >= MIN_EXAMPLE_COVERAGE)]
    out = {}
    for t in ["clear", "variable"]:
        cand = ok[(ok["day_type"] == t) & ok["month"].isin(EXAMPLE_MONTHS)]
        cand = cand if len(cand) else ok[ok["day_type"] == t]
        if len(cand):
            out[t] = (cand["nrmse"] - cand["nrmse"].median()).abs().idxmin(), cand.loc[
                (cand["nrmse"] - cand["nrmse"].median()).abs().idxmin(), "nrmse"]
    return out


def plot_examples(code, pairs, k10, out_dir):
    days = example_days(pairs)
    fig, axes = plt.subplots(1, len(days), figsize=(17, 6.2), sharey=True, squeeze=False)
    color = COLORS[code]
    alpha = k10["k"].iloc[0] / k10["kc"].iloc[0]
    for ax, (t, (date, nrmse)) in zip(axes[0], days.items()):
        obs = k10[k10["date"] == date]
        f = pairs[(pairs["h"] == 6) & (pairs["date"] == date)]
        ax.plot(obs["solar_hour"], alpha * obs["ghi_cs"], color="0.55", ls="--", lw=1.5,
                label=f"station clear-sky GHI (Ineichen × {alpha:.2f})")
        ax.plot(obs["solar_hour"], obs["ghi"], color="black", lw=2, label="measured GHI (10-min)")
        ax.plot(f["solar_hour"], f["smart"], color=color, lw=2.2, label="1-h smart-persistence forecast")
        ax.fill_between(f["solar_hour"], f["ghi"], f["smart"], color=color, alpha=0.2, lw=0, label="forecast error")
        rmse = np.sqrt(((f["smart"] - f["ghi"]) ** 2).mean())
        ax.set_title(f"typical {t} day, {date:%Y-%m-%d}: RMSE {rmse:.0f} W m$^{{-2}}$, nRMSE {nrmse:.0f}%",
                     fontsize=13)
        ax.set_xlabel("local solar time (h)", fontsize=12)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("GHI (W m$^{-2}$)", fontsize=12)
    axes[0, 0].set_ylim(0, None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"{label(code)}: 1-h smart persistence tracks a clear day but lags every cloud passage on a "
                 "variable day", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = out_dir / f"example_days_{code}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, days


# ---------- cross-station figures ----------

def plot_compare_horizon(table, codes, out_dir):
    groups = [("all", "all"), ("day_type", "clear"), ("day_type", "variable")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), sharey=True)
    for ax, (split, group) in zip(axes, groups):
        ends = {}
        for code in codes:
            sub = table[(table["station"] == code) & (table["split"] == split) & (table["group"] == group)]
            for b in ["smart", "clim"]:
                s = sub[sub["baseline"] == b]
                ax.plot(s["horizon_min"] / 60, s["nrmse"], ls=BASELINES[b][1], color=COLORS[code],
                        lw=2.4 if b == "smart" else 1.6)
            smart = sub[sub["baseline"] == "smart"].set_index("h")["nrmse"]
            ax.plot(hours(6), smart[6], "o", color=COLORS[code])
            ends[code] = smart[18]
        ys = spread(list(ends.values()), gap=2.6)
        for (code, v), y in zip(ends.items(), ys):
            ax.plot(3, v, "o", color=COLORS[code])
            ax.annotate(f"{code.upper()} {v:.0f}%", (3, v), xytext=(3.07, y), textcoords="data", fontsize=11,
                        color=COLORS[code], va="center")
        ax.set_xlim(0, 3.55)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
        ax.set_title("all days" if group == "all" else f"{group} days", fontsize=13)
        ax.set_xlabel("forecast horizon (h)", fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("nRMSE of 10-min GHI (%)", fontsize=12)
    handles = [Line2D([], [], color=COLORS[c], lw=2.4, label=label(c)) for c in codes]
    handles += [Line2D([], [], color="0.3", ls=BASELINES[b][1], label=BASELINES[b][0]) for b in ["smart", "clim"]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12, bbox_to_anchor=(0.5, -0.07), frameon=False)
    sel = [c for c in codes if c != REFERENCE]
    var = [pick(table[table["station"] == c], "day_type", "variable", "smart", 6) for c in sel]
    clear = [pick(table[table["station"] == c], "day_type", "clear", "smart", 6) for c in sel]
    fig.suptitle(f"Smart-persistence error grows with horizon; at 1 h it is {min(clear):.0f}–{max(clear):.0f}% on "
                 f"clear days vs {min(var):.0f}–{max(var):.0f}% on variable days at PSU, GWN, TBL\n"
                 "(solid: smart persistence, dashed: climatology; labels: smart persistence at 3 h, dot at 1 h)",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    path = out_dir / "error_vs_horizon_stations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_skill(table, codes, out_dir):
    groups = [("all", "all"), ("day_type", "clear"), ("day_type", "variable")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    for ax, (split, group) in zip(axes, groups):
        for code in codes:
            sub = table[(table["station"] == code) & (table["split"] == split) & (table["group"] == group)]
            for b in ["smart", "blend"]:
                s = sub[sub["baseline"] == b]
                ax.plot(s["horizon_min"] / 60, s["skill_vs_clim"], ls=BASELINES[b][1], color=COLORS[code],
                        lw=2.4 if b == "smart" else 1.6)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("all days" if group == "all" else f"{group} days", fontsize=13)
        ax.set_xlabel("forecast horizon (h)", fontsize=12)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("skill vs climatology, 1 − RMSE / RMSE$_{clim}$ (–)", fontsize=12)
    handles = [Line2D([], [], color=COLORS[c], lw=2.4, label=label(c)) for c in codes]
    handles += [Line2D([], [], color="0.3", ls=BASELINES[b][1], label=BASELINES[b][0]) for b in ["smart", "blend"]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12, bbox_to_anchor=(0.5, -0.07), frameon=False)
    sel = [c for c in codes if c != REFERENCE]
    s1 = [pick(table[table["station"] == c], "day_type", "variable", "smart", 6, "skill_vs_clim") for c in sel]
    s3 = [pick(table[table["station"] == c], "day_type", "variable", "smart", 18, "skill_vs_clim") for c in sel]
    fig.suptitle(f"On variable days smart persistence has a skill of only {min(s1):.2f} to {max(s1):.2f} over climatology "
                 f"at 1 h and {min(s3):.2f} to {max(s3):.2f} at 3 h at PSU, GWN, TBL\n(skill > 0: better than climatology; "
                 "solid: smart persistence, dash-dot: blend)", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    path = out_dir / "skill_vs_horizon_stations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(table, codes, out_dir):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    width = 0.8 / len(codes)
    x = np.arange(len(DAY_TYPES))
    vals = {c: [pick(table[table["station"] == c], "day_type", t, "smart", 6) for t in DAY_TYPES] for c in codes}
    top = np.nanmax(list(vals.values()))
    for i, code in enumerate(codes):
        sub = table[(table["station"] == code) & (table["h"] == 6)]
        days = [sub[(sub["split"] == "day_type") & (sub["group"] == t)]["n_days"].max() for t in DAY_TYPES]
        pos = x + (i - (len(codes) - 1) / 2) * width
        ax.bar(pos, vals[code], width * 0.95, color=COLORS[code], edgecolor="0.25", lw=0.6, label=label(code),
               hatch="//" if code == REFERENCE else "")
        for p, v, d in zip(pos, vals[code], days):
            if np.isfinite(v):
                ax.text(p, v + top * 0.045, f"{v:.1f}%" if v < 10 else f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=13)
                ax.text(p, v + top * 0.01, f"{d:.0f} d", ha="center", va="bottom", fontsize=10, color="0.35")
    ax.set_xticks(x, [f"{t} days" for t in DAY_TYPES], fontsize=14)
    ax.set_ylabel("1-h smart-persistence nRMSE of 10-min GHI (%)", fontsize=13)
    ax.set_ylim(0, top * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    sel = [c for c in codes if c != REFERENCE]
    ratios = [pick(table[table["station"] == c], "day_type", "variable", "smart", 6)
              / pick(table[table["station"] == c], "day_type", "clear", "smart", 6) for c in sel]
    ax.set_title(f"A 1-hour-ahead forecast is {min(ratios):.0f}–{max(ratios):.0f}× worse on variable days than "
                 f"clear days at PSU, GWN and TBL\n(smart persistence k(t)·GHI$_{{cs}}$(t+1 h); DRA = clear "
                 "desert reference; grey: days in group)", fontsize=14, loc="left")
    path = out_dir / "summary_1h_nrmse_by_day_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def key_table(table):
    """nRMSE (and RMSE, W m^-2) at 1 h and 3 h by day type, station and baseline, wide."""
    sub = table[table["h"].isin(KEY_HORIZONS) & (table["split"].isin(["all", "day_type"]))].copy()
    sub["lead"] = sub["h"].map(KEY_HORIZONS)
    wide = sub.pivot_table(index=["station", "group"], columns=["lead", "baseline"],
                           values=["nrmse", "rmse"]).round(1)
    wide.columns = [f"{v}_{lead}_{b}" for v, lead, b in wide.columns]
    wide.insert(0, "n_days", sub[sub["h"] == 6].groupby(["station", "group"])["n_days"].max())
    order = {"all": 0, "clear": 1, "variable": 2, "overcast": 3}
    wide = wide.reset_index().sort_values(["station", "group"], key=lambda s: s.map(order) if s.name == "group" else s)
    return wide.rename(columns={"group": "day_type"})


def run_station(code, figures=True):
    table, pairs, k10 = station_scores(code)
    if figures:
        out_fig = figure_dir("deep_dive", code)
        table.round(4).to_csv(table_dir("deep_dive", code) / f"forecast_scores_{code}.csv", index=False)
        paths = [plot_horizon(code, table, out_fig), plot_splits(code, table, out_fig)]
        path, days = plot_examples(code, pairs, k10, out_fig)
        paths.append(path)
        print("  example days:", ", ".join(f"{t} {d:%Y-%m-%d} (1-h nRMSE {n:.0f}%)" for t, (d, n) in days.items()))
        print("  figures:", *paths, sep="\n    ")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--station", help="one station code; default: the selected stations plus dra")
    args = parser.parse_args()
    if args.station:
        run_station(args.station.lower())
        return
    selected = list(pd.read_csv(SELECTED)["code"])
    codes = selected + [REFERENCE]
    table = pd.concat([run_station(c, figures=c != REFERENCE) for c in codes], ignore_index=True)
    out_tab, out_fig = table_dir("deep_dive", "comparison"), figure_dir("deep_dive", "comparison")
    table.round(4).to_csv(out_tab / "forecast_scores_all.csv", index=False)
    key = key_table(table)
    key.to_csv(out_tab / "forecast_key_nrmse_1h_3h.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for lead in KEY_HORIZONS.values():
        cols = ["station", "day_type", "n_days"] + [f"nrmse_{lead}_{b}" for b in BASELINES] \
               + [f"rmse_{lead}_smart"]
        print(f"\nnRMSE (%) at {lead} by day type (rmse in W m^-2)\n", key[cols].to_string(index=False))
    skill = table[table["h"].isin(KEY_HORIZONS) & table["split"].isin(["all", "day_type"])
                  & table["baseline"].isin(["smart", "blend"])]
    print("\nskill vs climatology\n", skill.pivot_table(index=["station", "group"], columns=["baseline", "horizon_min"],
                                                         values="skill_vs_clim").round(3).to_string())
    paths = [plot_summary(table, codes, out_fig), plot_compare_horizon(table, codes, out_fig),
             plot_skill(table, codes, out_fig)]
    print("\ntables in", out_tab)
    print("figures:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
