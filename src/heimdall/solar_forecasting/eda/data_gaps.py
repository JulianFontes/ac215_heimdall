"""missing and quality-flagged 1-min GHI / DNI / DHI at every SURFRAD station, and which months they hurt.

status of every minute of the year, from the raw daily files:
  missing  = no row in the file, or the value is the -9999.9 fill (nan after read_surfrad)
  flagged  = a value is present but its SURFRAD quality flag is not 0
common.py turns both into nan, so both are lost for k, energy and VI.
split into sun up (zenith < 90°), the k window (daylight(): zenith < 80° and GHI_cs > 50) and night.
a month is "affected" if more than AFFECTED_SHARE of its k-window minutes are unusable.

the parsed status is cached to table_dir("eda") / raw_status_2025.pkl (reading 2555 files takes minutes).
figures go to figure_dir("eda"), tables to table_dir("eda").
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    STATION_NAMES,
    STATIONS,
    SURFRAD_DIR,
    YEAR,
    daylight,
    figure_dir,
    load_all_stations,
    table_dir,
)

# settings
VARS = ["ghi", "dni", "dhi"]
AFFECTED_SHARE = 0.05  # unusable share of k-window minutes that makes a month's statistics suspect
TOP_GAPS = 3  # longest gaps listed per station
FIG_DIR = figure_dir("eda")
TAB_DIR = table_dir("eda")
STATUS_CACHE = TAB_DIR / f"raw_status_{YEAR}.pkl"
use_plot_style()
GOOD, FLAGGED, MISSING = 0, 1, 2

stations = load_all_stations()
labels = {c: f"{c.upper()} – {STATION_NAMES[c]}" for c in STATIONS}
month_names = list(pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%b"))
grid = pd.date_range(f"{YEAR}-01-01", f"{YEAR + 1}-01-01", freq="1min", inclusive="left", tz="UTC")


def raw_status(code):
    """GOOD / FLAGGED / MISSING per minute of the year for each of VARS, from the raw daily files."""
    frames = []
    for day in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D"):
        data, _ = pvlib.iotools.read_surfrad(str(SURFRAD_DIR / code / f"{code}{day:%y}{day.dayofyear:03d}.dat"))
        status = pd.DataFrame(GOOD, index=data.index, columns=VARS, dtype="int8")
        for v in VARS:
            status.loc[data[f"{v}_flag"] != 0, v] = FLAGGED
            status.loc[data[v].isna(), v] = MISSING
        frames.append(status)
    status = pd.concat(frames)
    status = status[~status.index.duplicated()]
    return status.reindex(grid, fill_value=MISSING)  # absent rows count as missing


if STATUS_CACHE.exists():
    status = pd.read_pickle(STATUS_CACHE)
else:
    status = {}
    for code in STATIONS:
        print(f"reading raw flags: {code}")
        status[code] = raw_status(code)
    pd.to_pickle(status, STATUS_CACHE)

# ---- monthly counts by variable and period ----
rows, gaps, daily_bad = [], [], {}
for code in STATIONS:
    _, df = stations[code]
    s = status[code].reindex(df.index)
    periods = {"sun_up": df["zenith"] < 90, "k_window": daylight(df), "night": df["zenith"] >= 90}
    month = df.index.month
    for v in VARS:
        for p, mask in periods.items():
            g = pd.DataFrame({"missing": s[v] == MISSING, "flagged": s[v] == FLAGGED})[mask].groupby(month[mask])
            counts = g.sum()
            counts["minutes"] = g.size()
            for m, r in counts.iterrows():
                rows.append({"station": code, "month": m, "var": v, "period": p, "minutes": int(r["minutes"]),
                             "missing": int(r["missing"]), "flagged": int(r["flagged"])})
    # power (and the 10-min table) needs all three components; k and VI only need GHI
    kwin = periods["k_window"]
    any_bad = (s[VARS] != GOOD).any(axis=1)
    for name, bad in [("ghi", s["ghi"] != GOOD), ("any", any_bad)]:
        daily_bad[(code, name)] = bad[kwin].groupby(df.index[kwin].normalize()).mean()

    # longest runs of unusable data (any time of day), with the k-window minutes they cost
    for name, bad in [("ghi", s["ghi"] != GOOD), ("any", any_bad)]:
        edges = np.flatnonzero(np.diff(np.r_[0, bad.to_numpy().astype(int), 0]))
        starts, ends = edges[::2], edges[1::2]
        for i in np.argsort(ends - starts)[::-1][:TOP_GAPS]:
            a, b = starts[i], ends[i]
            gaps.append({"station": code, "unusable": name, "start": df.index[a],
                         "end": df.index[b - 1] + pd.Timedelta("1min"), "hours": round((b - a) / 60, 1),
                         "k_window_minutes_lost": int(kwin.iloc[a:b].sum())})

counts = pd.DataFrame(rows)
counts["bad"] = counts["missing"] + counts["flagged"]
counts["bad_share"] = counts["bad"] / counts["minutes"]
counts.to_csv(TAB_DIR / "data_gaps_monthly.csv", index=False)
gaps = pd.DataFrame(gaps)
gaps.to_csv(TAB_DIR / "data_gaps_longest.csv", index=False)

# per station x month: unusable share of k-window minutes, for GHI alone and for any component
kw = []
for code in STATIONS:
    _, df = stations[code]
    s = status[code].reindex(df.index)
    kwin = daylight(df)
    month = df.index.month[kwin]
    kw.append(pd.DataFrame({"station": code, "month": month, "ghi": (s.loc[kwin, "ghi"] != GOOD).to_numpy(),
                            "any": (s.loc[kwin, VARS] != GOOD).any(axis=1).to_numpy()})
              .groupby(["station", "month"]).mean())
kw = pd.concat(kw)
kw["ghi_affected"] = kw["ghi"] > AFFECTED_SHARE
kw["any_affected"] = kw["any"] > AFFECTED_SHARE
kw.to_csv(TAB_DIR / "data_gaps_k_window.csv")

pd.set_option("display.width", 200)
totals = counts.groupby(["station", "var", "period"])[["minutes", "missing", "flagged"]].sum()
totals["missing_%"] = totals["missing"] / totals["minutes"] * 100
totals["flagged_%"] = totals["flagged"] / totals["minutes"] * 100
print("whole-year missing / flagged share (%) by station, variable and period:")
print(totals[["missing_%", "flagged_%"]].unstack("period").round(2).to_string())
print(f"\nunusable share of k-window minutes by month (GHI only; any of GHI/DNI/DHI), > {AFFECTED_SHARE:.0%} marked *:")
for col in ["ghi", "any"]:
    t = kw[col].unstack("month").reindex(STATIONS) * 100
    print(f"\n{col}:")
    print(t.round(1).to_string())
affected = kw[kw["ghi_affected"] | kw["any_affected"]]
print("\naffected station-months:")
print((affected[["ghi", "any"]] * 100).round(1).to_string() if len(affected) else "  none")
print(f"\nlongest unusable runs (top {TOP_GAPS} per station; ghi = GHI alone, any = any component):")
print(gaps.to_string(index=False))
whole_days = (gaps[gaps.hours >= 23.9].assign(day=lambda d: d["start"].dt.strftime("%Y-%m-%d"))
              .drop_duplicates(["station", "day"]).groupby("day")["station"].apply(lambda s: ", ".join(s)))
print("\nwhole days (>= 24 h runs) lost, and at which stations:")
print(whole_days.to_string())


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIG_DIR / name}")


print("\nfigures:")

# 1. heatmaps: unusable k-window share by station x month, GHI only (k, VI) and any component (power)
fig, axes = plt.subplots(2, 1, figsize=(13, 10.5), layout="constrained")
vmax = max(kw["any"].max() * 100, AFFECTED_SHARE * 100 * 2)
for ax, col, title in [(axes[0], "ghi", "GHI unusable (affects k, daily VI, sky classes)"),
                       (axes[1], "any", "any of GHI / DNI / DHI unusable (affects PV power and energy)")]:
    data = kw[col].unstack("month").reindex(index=STATIONS, columns=range(1, 13)).to_numpy() * 100
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        text = "0" if v == 0 else f"{v:.1f}" if v < 10 else f"{v:.0f}"
        ax.text(j, i, text + ("*" if v > AFFECTED_SHARE * 100 else ""), ha="center", va="center", fontsize=10,
                color="k" if v > 0.6 * vmax else "white")
    ax.set_xticks(range(12), month_names)
    ax.set_yticks(range(len(STATIONS)), [labels[c] for c in STATIONS])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% of k-window minutes", shrink=0.9)
n_aff = int(kw["any_affected"].sum())
n_aff_ghi = int(kw["ghi_affected"].sum())
worst = kw["any"].idxmax()
fig.suptitle(f"Data gaps are rare: {n_aff_ghi} of 84 station-months lose > {AFFECTED_SHARE:.0%} of daylight GHI and "
             f"{n_aff} lose > {AFFECTED_SHARE:.0%} of some component\n(worst: {labels[worst[0]]}, "
             f"{month_names[worst[1] - 1]}, {kw['any'].max():.0%}; * = above {AFFECTED_SHARE:.0%})", fontsize=14)
save(fig, "data_gaps_heatmap.png")

# 2. when the gaps happen: daily unusable share of k-window GHI, with each station's longest gap
days = pd.date_range(f"{YEAR}-01-01", f"{YEAR + 1}-01-01", freq="D")
table = np.vstack([daily_bad[(c, "any")].tz_localize(None).reindex(days[:-1]).to_numpy() * 100 for c in STATIONS])
fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
mesh = ax.pcolormesh(days, np.arange(len(STATIONS) + 1), table, cmap="viridis", vmin=0, vmax=100)
ax.hlines(np.arange(1, len(STATIONS)), days[0], days[-1], color="white", lw=2)
for i, code in enumerate(STATIONS):
    g = gaps[(gaps.station == code) & (gaps.unusable == "any")].iloc[0]
    mid = g["start"].tz_localize(None) + (g["end"] - g["start"]) / 2
    right = mid.month >= 9  # keep labels for late-year gaps inside the axes
    length = f"{g['hours'] / 24:.1f} days" if g["hours"] >= 48 else f"{g['hours']:.0f} h"
    ax.annotate(f"longest {length} ({g['start']:%b %d})", (mid, i + 0.5), xytext=(-8 if right else 8, 0),
                textcoords="offset points", va="center", ha="right" if right else "left", fontsize=10, color="w",
                bbox={"boxstyle": "round,pad=0.15", "fc": "0.2", "ec": "none", "alpha": 0.8})
ax.set_yticks(np.arange(len(STATIONS)) + 0.5, [labels[c] for c in STATIONS])
ax.invert_yaxis()
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
ax.set_xlabel(f"{YEAR}, one cell per UTC day")
fig.colorbar(mesh, ax=ax, label="% of k-window minutes with any of\nGHI / DNI / DHI unusable", pad=0.01)
shared = whole_days[whole_days.str.count(",") >= 2]
shared_text = "; ".join(f"{pd.Timestamp(d):%b %d} at {s.count(',') + 1} stations" for d, s in shared.items())
longest = gaps[gaps.unusable == "any"].sort_values("hours").iloc[-1]
ax.set_title(f"Gaps are mostly single lost days, some network-wide ({shared_text});\nthe longest is "
             f"{longest['hours'] / 24:.1f} days at {labels[longest['station']]} from {longest['start']:%b %d} "
             "(labels: each station's longest run with any component unusable)", fontsize=13)
save(fig, "data_gaps_timeline.png")
