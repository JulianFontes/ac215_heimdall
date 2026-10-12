"""explore ground irradiance stations and the clear-sky index (k) with pvlib.

for every station and day in [START, END]:
  1. download 1-min GHI / DNI / DHI from NOAA SURFRAD (public, no account needed)
  2. model clear-sky GHI / DNI / DHI with pvlib (Ineichen, Linke turbidity climatology)
  3. k = GHI_measured / GHI_clearsky, at 1 min and averaged to 10 min (the GOES-19
     full-disk cadence, so k lines up with GOES / C3DIR scenes later)
  4. PV power for a reference system from measured vs clear-sky irradiance,
     giving a power-based k_power = P / P_clearsky
  5. figures and a 10-min table for later modeling

change the settings below and run the file. downloads are cached in
<DATA_DIR>/stations/, figures go to <PLOT_DIR>/, and the 10-min table to
<DATA_DIR>/stations/k_10min_<START>_<END>.csv.
"""
import warnings
from concurrent.futures import ThreadPoolExecutor

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import requests
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from heimdall.c3dir.constants import GOES19_LON, VIEW_RADIUS_DEG
from heimdall.c3dir.geometry import view_circle
from heimdall.paths import DATA_DIR, FIGURES_DIR
from heimdall.plot_style import use as use_plot_style

# settings: change these
START = "2025-01-01"  # UTC days, inclusive; a full year covers every season
END = "2025-12-31"
SURFRAD_STATIONS = ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]  # all 7 SURFRAD sites
EXAMPLE_STATION = "bon"  # station for the detailed one-day figure
EXAMPLE_DAY = "2025-07-09"
AVERAGE = "10min"  # GOES-19 full-disk cadence
MAX_ZENITH = 80  # k gets unreliable near sunrise/sunset (small GHI, cosine errors)
PV_TILT, PV_AZIMUTH = 30, 180  # degrees; reference fixed-tilt, south-facing system
PV_PDC0, PV_GAMMA = 5000, -0.004  # W at 1000 W m^-2 and 25 °C; power temperature coefficient (1/°C)
STATION_DIR = DATA_DIR / "stations"
PLOT_DIR = FIGURES_DIR / "solar_forecasting"

SURFRAD_URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad"
use_plot_style()  # arial bold, 12 pt ticks and legends, as in every heimdall figure


def download(url, path):
    """download url to path once. requests is used instead of pvlib's urlopen,
    which fails certificate checks on python.org installs of python."""
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def load_surfrad(station, start, end):
    """1-min SURFRAD data for whole days in [start, end]; values with a quality flag set become nan."""
    def fetch(day):
        name = f"{station}{day:%y}{day.dayofyear:03d}.dat"
        try:
            return download(f"{SURFRAD_URL}/{station}/{day.year}/{name}", STATION_DIR / "surfrad" / station / name)
        except requests.HTTPError:
            warnings.warn(f"SURFRAD {station} {day:%Y-%m-%d} missing, skipped")
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:  # one file per day, so download several at once
        paths = list(pool.map(fetch, pd.date_range(start, end, freq="D")))
    frames, meta = [], None
    for path in paths:
        if path is None:
            continue
        data, meta = pvlib.iotools.read_surfrad(str(path))
        for v in ["ghi", "dni", "dhi", "temp_air", "wind_speed"]:
            data.loc[data[f"{v}_flag"] != 0, v] = np.nan  # SURFRAD flag 0 = good
        frames.append(data[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])
    if not frames:
        return None, None
    return pd.concat(frames), {"name": meta["name"], "latitude": meta["latitude"],
                               "longitude": meta["longitude"], "altitude": meta["elevation"]}


def pv_power(location, solpos, ghi, dni, dhi, temp_air, wind_speed):
    """dc power of the reference system: transpose to the panel plane (Hay-Davies),
    SAPM cell temperature, then PVWatts."""
    poa = pvlib.irradiance.get_total_irradiance(
        PV_TILT, PV_AZIMUTH, solpos["apparent_zenith"], solpos["azimuth"], dni, ghi, dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(ghi.index), model="haydavies")["poa_global"]
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    temp_cell = pvlib.temperature.sapm_cell(poa, temp_air, wind_speed, **params)
    return pvlib.pvsystem.pvwatts_dc(poa, temp_cell, PV_PDC0, PV_GAMMA)


def analyze(data, meta):
    """add clear-sky irradiance, k, clear-sky flags and PV power to one station's 1-min data."""
    location = pvlib.location.Location(meta["latitude"], meta["longitude"], altitude=meta["altitude"])
    times = pd.date_range(data.index.min().floor("D"), data.index.max().ceil("D"),
                          freq="1min", inclusive="left", tz="UTC")
    df = data.reindex(times)  # regular 1-min grid, gaps as nan
    solpos = location.get_solarposition(times)
    clear = location.get_clearsky(times, solar_position=solpos)  # ghi, dni, dhi
    df["zenith"] = solpos["apparent_zenith"]
    for v in ["ghi", "dni", "dhi"]:
        df[f"{v}_cs"] = clear[v]

    day = (df["zenith"] < MAX_ZENITH) & (df["ghi_cs"] > 50)
    df["k"] = (df["ghi"] / df["ghi_cs"]).where(day)
    # closure: GHI should equal DNI cos(zenith) + DHI within a few percent (standard radiometer QC check)
    df["closure"] = ((df["dni"] * np.cos(np.radians(df["zenith"])) + df["dhi"]) / df["ghi"]).where(day)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        is_clear, _, alpha = pvlib.clearsky.detect_clearsky(df["ghi"].fillna(0), df["ghi_cs"], times,
                                                            return_components=True)
    df["is_clear"] = is_clear & day
    # alpha: factor that best scales the modeled clear sky onto the measured clear periods;
    # alpha > 1 means the Ineichen clear sky (monthly turbidity climatology) runs low at this site
    df.attrs["clearsky_alpha"] = alpha

    temp_air = df["temp_air"].fillna(20.0)
    wind = df["wind_speed"].fillna(1.0)
    df["power"] = pv_power(location, solpos, df["ghi"], df["dni"], df["dhi"], temp_air, wind)
    df["power_cs"] = pv_power(location, solpos, df["ghi_cs"], df["dni_cs"], df["dhi_cs"], temp_air, wind)
    df["k_power"] = (df["power"] / df["power_cs"]).where(day)
    return df


def to_10min(df):
    """average to the GOES cadence. k is the ratio of the averages (not the average of
    ratios), so it stays consistent with energy over the interval."""
    cols = ["ghi", "ghi_cs", "dni", "dni_cs", "dhi", "dhi_cs", "power", "power_cs", "zenith"]
    out = df[cols].resample(AVERAGE).mean()
    out["k"] = out["ghi"] / out["ghi_cs"]
    out["k_power"] = out["power"] / out["power_cs"]
    out["k_std_1min"] = df["k"].resample(AVERAGE).std()  # variability GOES can't resolve
    out["clear_fraction"] = df["is_clear"].resample(AVERAGE).mean()
    out["n_valid"] = df["k"].resample(AVERAGE).count()
    return out[(out["zenith"] < MAX_ZENITH) & (out["n_valid"] >= 8)]


# ---- load and analyze every station ----
stations = {}  # code -> (meta, 1-min df)
for code in SURFRAD_STATIONS:
    try:
        data, meta = load_surfrad(code, START, END)
    except Exception as error:  # a bad code or server hiccup shouldn't stop the others
        print(f"{code}: failed ({error})")
        continue
    if data is None:
        print(f"{code}: skipped (no data)")
        continue
    stations[code] = (meta, analyze(data, meta))
    print(f"{code} ({meta['name']}, {meta['latitude']:.2f}, {meta['longitude']:.2f}): "
          f"{stations[code][1]['ghi'].notna().sum()} 1-min GHI values")
if not stations:
    raise SystemExit("no station data loaded")

# ---- 10-min table and summary ----
tables = []
for code, (meta, df) in stations.items():
    t = to_10min(df)
    t.insert(0, "station", code)
    t.insert(1, "latitude", meta["latitude"])
    t.insert(2, "longitude", meta["longitude"])
    tables.append(t)
k10 = pd.concat(tables).rename_axis("time")
STATION_DIR.mkdir(parents=True, exist_ok=True)
k10.to_csv(STATION_DIR / f"k_10min_{START}_{END}.csv")

summary = k10.groupby("station").agg(
    intervals=("k", "count"), mean_k=("k", "mean"), median_k=("k", "median"),
    clear=("clear_fraction", lambda s: (s > 0.9).mean()),
    overcast=("k", lambda s: (s < 0.3).mean()),
    above_1=("k", lambda s: (s > 1.05).mean()), mean_k_power=("k_power", "mean"))
summary["k_when_clear"] = k10[k10["clear_fraction"] > 0.9].groupby("station")["k"].median()
energy_totals = k10.groupby("station")[["power", "power_cs"]].sum()
summary["energy_vs_clearsky"] = energy_totals["power"] / energy_totals["power_cs"]
summary["clearsky_alpha"] = [stations[c][1].attrs["clearsky_alpha"] for c in summary.index]
print(f"\n10-min intervals with sun > {90 - MAX_ZENITH}° up, {START} to {END}")
print("(clear / overcast / above_1: fraction of intervals that are clear, k < 0.3, k > 1.05;")
print(" k_when_clear and clearsky_alpha above 1 mean the pvlib clear sky runs low at that site)")
print(summary.round(2).to_string())

# ---- figures ----
PLOT_DIR.mkdir(parents=True, exist_ok=True)
codes = list(stations)
colors = dict(zip(codes, plt.get_cmap("tab10").colors))


def save(fig, name):
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {PLOT_DIR / name}")


print("\nfigures:")

# 1. where the stations are relative to what GOES-19 sees
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection=ccrs.Orthographic(-75.2, 0))
ax.set_global()
ax.add_feature(cfeature.LAND.with_scale("50m"), color="0.92")
ax.coastlines("50m", lw=0.5, color="0.4")
ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="0.6")
view_lat, view_lon = view_circle(GOES19_LON, VIEW_RADIUS_DEG)
ax.plot(np.r_[view_lon, view_lon[:1]], np.r_[view_lat, view_lat[:1]], color="tab:blue", lw=1.5, transform=ccrs.Geodetic(),
        label="GOES-19 view used for C3DIR (65°)")
for code, (meta, _) in stations.items():
    ax.plot(meta["longitude"], meta["latitude"], "o", color=colors[code], ms=7, mec="k", mew=0.5,
            transform=ccrs.PlateCarree())
    ax.annotate(code.upper(), (meta["longitude"], meta["latitude"]), xytext=(5, 3),
                textcoords="offset points", fontsize=10, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))
ax.plot([], [], "o", color="0.5", mec="k", label="SURFRAD stations")
ax.legend(loc="lower left")
ax.set_title("SURFRAD stations in the GOES-19 view")
save(fig, "station_map.png")

# 2. k over the whole period for every station in one plot: one row per station, color = k.
# short windows show every 10-min interval; long ones one cell per day (total GHI / total clear-sky GHI)
long_window = len(pd.date_range(START, END, freq="D")) > 31
cell = "D" if long_window else AVERAGE
grid = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1),
                     freq=cell).tz_localize(None)  # cell edges
k_cells = k10.groupby("station")[["ghi", "ghi_cs"]].resample(cell, level="time").sum()
k_cells = (k_cells["ghi"] / k_cells["ghi_cs"]).where(k_cells["ghi_cs"] > 0).rename("k").reset_index()
table = (k_cells.assign(time=lambda d: d["time"].dt.tz_localize(None))
         .pivot(index="station", columns="time", values="k").reindex(index=codes, columns=grid[:-1]))
# blue = cloud attenuation, pale yellow = clear (k = 1), orange = cloud enhancement; gray = night
k_cmap = LinearSegmentedColormap.from_list(
    "k", [(0, "#1f3b73"), (0.3, "#6b8fc7"), (0.5, "#fff4c2"), (1, "#e8892b")]).with_extremes(bad="0.85")
# TwoSlopeNorm puts k = 1 at the middle of the colormap, so the pale yellow sits at 0.5
fig, ax = plt.subplots(figsize=(14, 0.55 * len(codes) + 1.6), layout="constrained")
mesh = ax.pcolormesh(grid, np.arange(len(codes) + 1), table.values, cmap=k_cmap,
                     norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=1.3))
ax.hlines(np.arange(1, len(codes)), grid[0], grid[-1], color="white", lw=2)  # gaps between rows
ax.set_yticks(np.arange(len(codes)) + 0.5,
              [f"{c.upper()} – {stations[c][0]['name']}" for c in codes])
ax.invert_yaxis()
if len(pd.date_range(START, END, freq="D")) <= 31:  # day lines only help on short windows
    for day_edge in pd.date_range(START, END, freq="D")[1:]:
        ax.axvline(day_edge, color="k", lw=0.4, alpha=0.5)
ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
ax.set_xlabel("UTC, one cell per day" if long_window else "UTC (gray = night)")
fig.colorbar(mesh, ax=ax, extend="max", pad=0.01,
             label=f"k = GHI / GHI$_{{cs}}$ ({'daily' if long_window else AVERAGE})").set_ticks([0, 0.25, 0.5, 0.75, 1, 1.15, 1.3])
ax.set_title(f"Clear-sky index, {START} to {END}")
save(fig, f"k_all_stations_{START}_{END}.png")

# 3. measured vs clear-sky GHI on the example day, one panel per station
day_start = pd.Timestamp(EXAMPLE_DAY, tz="UTC")
day = slice(day_start, day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1))
n = len(codes)
fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(12, 2.6 * np.ceil(n / 2)),
                         sharey=True, layout="constrained", squeeze=False)
for ax, code in zip(axes.flat, codes):
    df = stations[code][1].loc[day]
    ax.plot(df.index, df["ghi_cs"], color="0.5", ls="--", lw=1.2, label="Clear-sky (pvlib Ineichen)")
    ax.plot(df.index, df["ghi"], color=colors[code], lw=0.8, label="Measured")
    ax.set_title(f"{code.upper()} – {stations[code][0]['name']}")
    ax.grid(alpha=0.3, lw=0.5)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
for ax in axes.flat[n:]:
    ax.set_visible(False)
axes[0, 0].legend(loc="upper left")
fig.supylabel("GHI (W m$^{-2}$)")
fig.supxlabel("UTC")
fig.suptitle(f"Measured vs clear-sky GHI, {EXAMPLE_DAY}")
save(fig, f"ghi_vs_clearsky_{EXAMPLE_DAY}.png")

# 4. one station in detail: irradiance components, k, and PV power
if EXAMPLE_STATION in stations:
    df = stations[EXAMPLE_STATION][1].loc[day]
    # full 10-min grid so gaps (night, bad data) break the line instead of being bridged
    t10 = k10[k10.station == EXAMPLE_STATION].reindex(
        pd.date_range(day.start, day.stop, freq=AVERAGE)).loc[day]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, layout="constrained")
    for v, c in [("ghi", "k"), ("dni", "tab:orange"), ("dhi", "tab:blue")]:
        axes[0].plot(df.index, df[v], color=c, lw=0.8, label=f"{v.upper()} measured")
        axes[0].plot(df.index, df[f"{v}_cs"], color=c, lw=1, ls="--", alpha=0.6, label=f"{v.upper()} clear-sky")
    axes[0].set_ylabel("W m$^{-2}$")
    axes[0].legend(ncols=3, loc="upper left")
    axes[0].set_title("Irradiance")

    axes[1].plot(df.index, df["k"], color="0.6", lw=0.6, label="1-min")
    axes[1].step(t10.index, t10["k"], where="post", color="k", lw=1.5, label=f"{AVERAGE} (GOES cadence)")
    axes[1].fill_between(df.index, 0, 1.4, where=df["is_clear"], color="gold", alpha=0.25,
                         step="mid", label="pvlib detect_clearsky")
    axes[1].axhline(1, color="0.3", lw=0.6, ls=":")
    axes[1].set_ylim(0, 1.4)
    axes[1].set_ylabel("k = GHI / GHI$_{cs}$")
    axes[1].legend(ncols=3, loc="lower left")
    axes[1].set_title("Clear-sky index k")

    axes[2].plot(df.index, df["power_cs"] / 1000, color="0.5", ls="--", label="Clear-sky")
    axes[2].plot(df.index, df["power"] / 1000, color="tab:green", lw=0.8, label="From measured irradiance")
    axes[2].set_ylabel("DC power (kW)")
    axes[2].legend(loc="upper left")
    axes[2].set_title(f"PV power ({PV_PDC0 / 1000:.0f} kW, {PV_TILT}° tilt)")

    axes[3].plot(df.index, df["k_power"], color="tab:green", lw=0.6, alpha=0.7, label="k_power = P / P$_{cs}$")
    axes[3].plot(df.index, df["k"], color="k", lw=0.6, alpha=0.7, label="k (GHI)")
    axes[3].axhline(1, color="0.3", lw=0.6, ls=":")
    axes[3].set_ylim(0, 1.4)
    axes[3].set_ylabel("index")
    axes[3].legend(loc="lower left")
    axes[3].set_title("k from GHI vs from power")
    axes[3].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    axes[3].set_xlabel("UTC")
    fig.suptitle(f"{EXAMPLE_STATION.upper()} – {stations[EXAMPLE_STATION][0]['name']}, {EXAMPLE_DAY}")
    save(fig, f"{EXAMPLE_STATION}_detail_{EXAMPLE_DAY}.png")

# 5. distribution of 10-min k per station over the whole period
fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(11, 2.2 * np.ceil(n / 2)),
                         sharex=True, layout="constrained", squeeze=False)
bins = np.linspace(0, 1.4, 57)
for ax, code in zip(axes.flat, codes):
    k = k10.loc[k10.station == code, "k"].dropna()
    ax.hist(k, bins=bins, color=colors[code], alpha=0.85)
    ax.axvline(1, color="0.3", lw=0.6, ls=":")
    ax.set_title(f"{code.upper()}  (median {k.median():.2f}, n = {len(k)})")
for ax in axes.flat[n:]:
    ax.set_visible(False)
fig.supxlabel(f"k ({AVERAGE} mean)")
fig.supylabel("count")
fig.suptitle(f"Clear-sky index distribution, {START} to {END}")
save(fig, f"k_distribution_{START}_{END}.png")

# 6. k vs solar zenith: where k stops being trustworthy
all_1min = pd.concat([df[["k", "zenith"]] for _, df in stations.values()]).dropna()
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
hb = ax.hexbin(all_1min["zenith"], all_1min["k"], gridsize=60, extent=(0, MAX_ZENITH, 0, 1.4),
               bins="log", cmap="Blues", mincnt=1)
ax.axhline(1, color="0.3", lw=0.6, ls=":")
ax.set_xlabel("Solar zenith angle (°)")
ax.set_ylabel("k (1-min)")
ax.set_title("Clear-sky index vs sun angle, all stations")
fig.colorbar(hb, label="1-min samples")
save(fig, f"k_vs_zenith_{START}_{END}.png")

# 7. data quality: measured GHI vs DNI·cos(z) + DHI
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
for code, (_, df) in stations.items():
    d = df[df["k"].notna()]
    d = d.sample(min(len(d), 20000), random_state=0)  # enough points to judge, fast to draw
    ax.scatter(d["ghi"], d["dni"] * np.cos(np.radians(d["zenith"])) + d["dhi"],
               s=1, alpha=0.2, color=colors[code], label=code.upper(), rasterized=True)
lim = [0, 1300]
ax.plot(lim, lim, color="k", lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Measured GHI (W m$^{-2}$)")
ax.set_ylabel("DNI·cos z + DHI (W m$^{-2}$)")
ax.set_title("Component closure")
ax.legend(markerscale=8)
save(fig, f"closure_{START}_{END}.png")

# 8. persistence skill: how well k now predicts k at t + h (the baseline any forecast must beat)
horizons = np.arange(1, 37)  # steps of AVERAGE: 10 min to 6 h
step_h = pd.Timedelta(AVERAGE) / pd.Timedelta(hours=1)
persistence = {}
for code in codes:
    t = k10.loc[k10.station == code, "k"]
    t = t.reindex(pd.date_range(t.index.min(), t.index.max(), freq=AVERAGE))  # nights become gaps
    rmse, corr = [], []
    for h in horizons:
        pair = pd.concat([t, t.shift(-h)], axis=1).dropna()  # only pairs with sun at both times
        rmse.append(np.sqrt(((pair.iloc[:, 1] - pair.iloc[:, 0]) ** 2).mean()))
        corr.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    persistence[code] = (np.array(rmse), np.array(corr))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
for code, (rmse, corr) in persistence.items():
    axes[0].plot(horizons * step_h, rmse, color=colors[code], lw=2, label=code.upper())
    axes[1].plot(horizons * step_h, corr, color=colors[code], lw=2, label=code.upper())
axes[0].set_ylabel("RMSE of k")
axes[0].set_title("Persistence error")
axes[1].set_ylabel("correlation of k(t) and k(t + h)")
axes[1].axhline(0, color="0.3", lw=0.6, ls=":")
axes[1].set_title("Persistence correlation")
for ax in axes:
    ax.set_xlabel("forecast horizon h (hours)")
    ax.grid(alpha=0.3, lw=0.5)
axes[1].legend(ncols=2)
fig.suptitle(f"Persistence of k, {START} to {END}")
save(fig, f"persistence_{START}_{END}.png")
print("    persistence RMSE of k at 1 h / 3 h: " + ", ".join(
    f"{c.upper()} {r[5]:.2f}/{r[17]:.2f}" for c, (r, _) in persistence.items()))

# 9. power ramps: how often output jumps by more than x % of capacity
ramp_minutes = [1, 10, 60]
thresholds = np.linspace(0, 80, 161)  # % of capacity
fig, axes = plt.subplots(1, len(ramp_minutes), figsize=(15, 4.6), sharey=True, layout="constrained")
for ax, minutes in zip(axes, ramp_minutes):
    clear_ramps = []
    for code, (_, df) in stations.items():
        sun = df["zenith"] < MAX_ZENITH
        ramp = (df["power"].shift(-minutes) - df["power"]).abs()[sun].dropna() / PV_PDC0 * 100
        exceed = [(ramp > x).mean() for x in thresholds]
        ax.plot(thresholds, exceed, color=colors[code], lw=1.5, label=code.upper())
        clear_ramps.append((df["power_cs"].shift(-minutes) - df["power_cs"]).abs()[sun].dropna() / PV_PDC0 * 100)
    clear_ramps = pd.concat(clear_ramps)
    ax.plot(thresholds, [(clear_ramps > x).mean() for x in thresholds], color="0.4", ls="--", lw=1.5,
            label="clear sky (sun angle only)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("|ΔP| (% of capacity)")
    ax.set_title(f"over {minutes} min")
    ax.grid(alpha=0.3, lw=0.5, which="both")
axes[0].set_ylabel("fraction of daytime with a larger ramp")
axes[-1].legend(ncols=2)
fig.suptitle("PV power ramps (above the dashed line = caused by clouds)")
save(fig, f"power_ramps_{START}_{END}.png")

# 10. energy: what a clear-sky schedule would promise vs what was produced (daily, or monthly on long windows)
monthly = len(pd.date_range(START, END, freq="D")) > 31
period, period_name, date_fmt = ("MS", "monthly", "%b") if monthly else ("D", "daily", "%b %d")
fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(12, 2.4 * np.ceil(n / 2)),
                         sharey=True, layout="constrained", squeeze=False)
for ax, code in zip(axes.flat, codes):
    df = stations[code][1]
    ok = df["power"].notna()  # only minutes with a measurement count on both sides
    energy = (df["power"].where(ok).resample(period).sum() / 60 / 1000)  # W per minute -> kWh
    energy_cs = (df["power_cs"].where(ok).resample(period).sum() / 60 / 1000)
    x = np.arange(len(energy))
    ax.bar(x, energy_cs, width=0.8, color="0.85", label="clear-sky plan")
    ax.bar(x, energy, width=0.5, color=colors[code], label="produced")
    for xi, e, ecs in zip(x, energy, energy_cs):
        ax.annotate(f"{e / ecs:.0%}", (xi, max(e, ecs)), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color="0.3")
    ax.set_xticks(x, [d.strftime(date_fmt) for d in energy.index], fontsize=10)
    ax.set_title(f"{code.upper()} – {stations[code][0]['name']}")
for ax in axes.flat[n:]:
    ax.set_visible(False)
axes[0, 0].legend(loc="lower left")
axes[0, 0].set_ylim(top=axes[0, 0].get_ylim()[1] * 1.1)  # room for the labels (y is shared)
fig.supylabel(f"{period_name} energy (kWh)")
fig.suptitle(f"{period_name.capitalize()} PV energy (labels: produced / clear-sky)")
save(fig, f"{period_name}_energy_{START}_{END}.png")

# 11. typical k by local solar time: when in the day clouds hit
fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(12, 2.4 * np.ceil(n / 2)),
                         sharex=True, sharey=True, layout="constrained", squeeze=False)
k10_solar = k10.assign(solar_hour=(k10.index.hour + k10.index.minute / 60 + k10["longitude"] / 15) % 24)
k10_solar["hour_bin"] = k10_solar["solar_hour"].round()
everyone = k10_solar.groupby("hour_bin")["k"].median()
for ax, code in zip(axes.flat, codes):
    by_hour = k10_solar[k10_solar.station == code].groupby("hour_bin")["k"]
    q = by_hour.quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(q.index, q[0.25], q[0.75], color=colors[code], alpha=0.25, label="middle 50%")
    ax.plot(q.index, q[0.5], color=colors[code], lw=2, label="median")
    ax.plot(everyone.index, everyone.values, color="0.4", lw=1, ls="--", label="all stations median")
    ax.axhline(1, color="0.3", lw=0.6, ls=":")
    ax.set_title(f"{code.upper()} – {stations[code][0]['name']}")
    ax.grid(alpha=0.3, lw=0.5)
for ax in axes.flat[n:]:
    ax.set_visible(False)
axes[0, 0].legend(loc="lower left")
axes[0, 0].set_ylim(0, 1.3)
fig.supxlabel("local solar time (h, 12 = solar noon)")
fig.supylabel(f"k ({AVERAGE})")
fig.suptitle(f"Clear-sky index by time of day, {START} to {END}")
save(fig, f"k_by_solar_hour_{START}_{END}.png")

# 12. diffuse fraction vs clearness index, against the Erbs split pvlib would use to get DNI/DHI from GHI
pooled = []
for code, (_, df) in stations.items():
    d = df[(df["zenith"] < MAX_ZENITH) & (df["ghi"] > 20)]
    erbs = pvlib.irradiance.erbs(d["ghi"], d["zenith"], d.index)
    pooled.append(pd.DataFrame({"kt": erbs["kt"], "measured": d["dhi"] / d["ghi"],
                                "erbs": erbs["dhi"] / d["ghi"], "dhi": d["dhi"], "dhi_erbs": erbs["dhi"]}))
pooled = pd.concat(pooled).dropna()
rmse_dhi = np.sqrt(((pooled["dhi_erbs"] - pooled["dhi"]) ** 2).mean())
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
hb = ax.hexbin(pooled["kt"], pooled["measured"].clip(0, 1.2), gridsize=70, extent=(0, 1, 0, 1.2),
               bins="log", cmap="Blues", mincnt=1)
curve = pooled.sort_values("kt")
ax.plot(curve["kt"], curve["erbs"], color="tab:orange", lw=2, label="Erbs model (pvlib)")
ax.set_xlabel("clearness index kt = GHI / (top-of-atmosphere irradiance · cos z)")
ax.set_ylabel("diffuse fraction DHI / GHI")
ax.set_title(f"Diffuse fraction vs Erbs (DHI RMSE {rmse_dhi:.0f} W m$^{{-2}}$)")
ax.legend()
fig.colorbar(hb, label="1-min samples")
save(fig, f"diffuse_fraction_{START}_{END}.png")

# 13. variability inside each 10-min interval: what a GOES snapshot can't resolve
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
v = k10.dropna(subset=["k", "k_std_1min"])
hb = ax.hexbin(v["k"], v["k_std_1min"], gridsize=50, extent=(0, 1.3, 0, 0.5), bins="log", cmap="Blues", mincnt=1)
med = v.groupby(pd.cut(v["k"], np.linspace(0, 1.3, 27)), observed=True)["k_std_1min"].median()
ax.plot([b.mid for b in med.index], med.values, color="tab:orange", lw=2, label="median")
ax.set_xlabel(f"k ({AVERAGE} mean)")
ax.set_ylabel(f"std of 1-min k inside the {AVERAGE} interval")
ax.set_title(f"Variability within {AVERAGE}")
ax.legend()
fig.colorbar(hb, label=f"{AVERAGE} intervals")
save(fig, f"k_variability_{START}_{END}.png")

# 14. cloud enhancement events: bursts of sunlight above clear sky from light reflected off cloud edges
# threshold per station: ENHANCEMENT above that station's typical clear-sky k (the median k during its
# detected clear periods), so the clear-sky model's 0-7% bias isn't counted as enhancement
ENHANCEMENT = 1.10
durations, share, peaks, limits = {}, {}, {}, {}
for code, (_, df) in stations.items():
    clear_k = df.loc[df["is_clear"], "k"].dropna()
    clear_level = clear_k.median() if len(clear_k) > 60 else df.attrs["clearsky_alpha"]
    limits[code] = ENHANCEMENT * clear_level
    sun = df["k"].notna()
    event = (df["k"] > limits[code]) & ~df["is_clear"]
    run_id = (event != event.shift()).cumsum()
    durations[code] = event.groupby(run_id).sum()[event.groupby(run_id).first()].values  # minutes per event
    share[code] = event[sun].mean()
    peaks[code] = df["k"].max()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})
bins = np.arange(1, 32)
for code in codes:
    axes[0].hist(durations[code], bins=bins, histtype="step", lw=1.5, color=colors[code], label=code.upper())
axes[0].set_yscale("log")
axes[0].set_xlabel("event length (minutes)")
axes[0].set_ylabel("number of events")
axes[0].set_title(f"Event length (k > {ENHANCEMENT:.2f} × clear-sky k)")
axes[0].legend(ncols=2)
x = np.arange(len(codes))
axes[1].bar(x, [share[c] * 100 for c in codes], color=[colors[c] for c in codes])
for xi, c in zip(x, codes):
    axes[1].annotate(f"limit {limits[c]:.2f}\nmax {peaks[c]:.2f}", (xi, share[c] * 100), xytext=(0, 2),
                     textcoords="offset points",
                     ha="center", fontsize=7, color="0.3")
axes[1].set_xticks(x, [c.upper() for c in codes])
axes[1].set_ylabel("% of daytime minutes")
axes[1].set_ylim(top=max(share.values()) * 100 * 1.2)  # room for the labels
axes[1].set_title("Time spent above clear sky")
fig.suptitle(f"Cloud enhancement, {START} to {END}")
save(fig, f"cloud_enhancement_{START}_{END}.png")
