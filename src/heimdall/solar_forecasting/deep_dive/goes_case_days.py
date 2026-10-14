"""what GOES-19 sees over PSU, GWN and TBL on their case days, next to the ground clear-sky index k.

case days per station (same rule as the ground-data deep dive): from classify_days, days with
n_minutes >= 90% of that date's daylight minutes; per season (warm Apr-Sep, cold Oct-Mar) the
lowest-VI clear day, the lowest-k overcast day and the highest-VI variable day. a season with no
day of a type borrows the next best day of that type from the other season. 6 days per station.

for each daylight CONUS scan every 10 min (every other 5-min scan) it keeps, from ABI-L2-CMIPC
band 13 (10.3 um brightness temperature) and ABI-L2-ACMC (clear-sky mask):
  - band 13 at the station pixel, and its mean and std in a 50 km box centred on the station
  - box cloud fraction: share of good-quality ACM pixels flagged cloudy or probably cloudy (BCM)
  - the band 13 crop (150 km) for image strips
then joins the 10-min alpha-corrected k of the interval containing the time GOES imaged the
station's row. per station-day extractions are cached under table_dir("deep_dive", code, "goes"),
so reruns only download scans that are missing or failed. raw files go to a short temporary
folder (Windows path limits under OneDrive) and are deleted right after extraction.

figures: a 6-panel case-day figure per station, a cross-station scatter, and a 3-frame band 13
strip on each station's most variable case day.

run with PYTHONUTF8=1 (goes2go prints emoji that crash the default Windows console).
"""
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import s3fs
from goes2go import GOES

from heimdall.c3dir.goes19 import goes_proj
from heimdall.plot_style import use as use_plot_style
from heimdall.solar_forecasting.common import (
    STATION_NAMES,
    STATIONS,
    alpha_corrected_k,
    classify_days,
    daylight,
    figure_dir,
    load_station_1min,
    solar_date,
    table_dir,
    to_10min,
)

# settings
CODES = ["psu", "gwn", "tbl"]
WARM_MONTHS = range(4, 10)  # Apr-Sep; Oct-Mar is the cold season
MIN_DAY_SHARE = 0.9  # valid minutes / daylight minutes for a case day
DAY_TYPES = ["clear", "variable", "overcast"]
SEASONS = ["warm", "cold"]
BOX_KM = 50  # cloud fraction / band 13 std box width, centred on the station
CROP_KM = 150  # band 13 image strip width
CROP_PX = 60  # half-width of the pixel window cut around the station (covers CROP_KM at 2-4 km pixels)
SCAN_STEP = "10min"
PRODUCTS = {"bt": ("ABI-L2-CMIP", 13), "acm": ("ABI-L2-ACM", None)}
MAX_FILE_MB = 20
WORKERS = 8
TMP_DIR = Path("C:/goes_tmp")
COLORS = dict(zip(STATIONS, plt.get_cmap("tab10").colors))  # fixed station colors, every script
BT_RANGE = (200, 310)  # K, band 13 color scale of the case-day strips
STRIP_HOURS = (9, 15)  # local solar hours searched for the image strip's k swing (k is noisy near sunrise/set)

use_plot_style()


def label(code):
    return f"{code.upper()} – {STATION_NAMES[code]}"


# --- case days ---------------------------------------------------------------------------------

def case_days(df):
    """the 6 case days of one station: DataFrame with date, season, day_type, k, vi, n_minutes."""
    days = classify_days(df)
    date = solar_date(df.index, df.attrs["longitude"])
    daylight_minutes = daylight(df).groupby(date).sum()
    days = days[days["n_minutes"] >= MIN_DAY_SHARE * daylight_minutes.reindex(days.index)].copy()
    days["season"] = np.where(days.index.month.isin(WARM_MONTHS), "warm", "cold")
    # best first: clear = lowest VI, overcast = lowest k, variable = highest VI
    order = {"clear": ("vi", True), "overcast": ("k", True), "variable": ("vi", False)}
    rows = []
    for t in DAY_TYPES:
        col, ascending = order[t]
        pool = days[days["day_type"] == t].sort_values(col, ascending=ascending)
        used = set()
        for season in SEASONS:
            own = pool[(pool["season"] == season) & ~pool.index.isin(used)]
            other = pool[(pool["season"] != season) & ~pool.index.isin(used)]
            pick = own.index[0] if len(own) else other.index[0]
            used.add(pick)
            rows.append({"date": pick, "season": season, "day_type": t, **days.loc[pick, ["k", "vi", "n_minutes"]]})
    return pd.DataFrame(rows)


# --- scans -------------------------------------------------------------------------------------

def daylight_window(df, d):
    """first and last daylight minute (naive UTC) of local solar date d."""
    idx = df.index[(solar_date(df.index, df.attrs["longitude"]) == d) & daylight(df)]
    return idx.min().tz_localize(None), idx.max().tz_localize(None)


def list_scans(clients, start, end):
    """one CMIPC band 13 + ACMC scan pair per 10 min in [start, end]: DataFrame with scan_start, bt, acm."""
    pad = pd.Timedelta(minutes=10)  # goes2go keeps only files entirely inside the window
    tables = []
    for name, client in clients.items():
        files = client.df(start - pad, end + pad, ignore_missing=True)  # some hourly folders don't exist
        files = files[(files["start"] >= start) & (files["start"] <= end)]
        tables.append(files.assign(scan_start=files["start"].dt.floor("min"))
                      .set_index("scan_start")["file"].rename(name))
    scans = pd.concat(tables, axis=1).dropna().sort_index()
    scans = scans[~scans.index.floor(SCAN_STEP).duplicated()]  # every other 5-min scan
    slots = pd.date_range(start.floor(SCAN_STEP), end.floor(SCAN_STEP), freq=SCAN_STEP)
    scans = scans.rename_axis("scan_start").reset_index()
    scans.attrs["missing_slots"] = slots.difference(scans["scan_start"].dt.floor(SCAN_STEP))
    return scans


def download(fs, key, retries=3):
    """copy one s3 object into TMP_DIR; returns the local path."""
    local = TMP_DIR / Path(key).name
    for attempt in range(retries):
        try:
            fs.get(key, str(local))
            return local
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def station_window(nc, lat, lon):
    """pixel window around the station: (rows, cols slices, station row/col inside it, dx/dy km)."""
    p = nc["goes_imager_projection"].__dict__
    h = p["perspective_point_height"]
    proj = goes_proj({"goes_imager_projection": SimpleNamespace(attrs=p)})
    sx, sy = proj(lon, lat)
    x, y = nc["x"][:].data, nc["y"][:].data  # scan angles, radians
    col, row = int(np.argmin(np.abs(x - sx / h))), int(np.argmin(np.abs(y - sy / h)))
    rows = slice(max(row - CROP_PX, 0), min(row + CROP_PX + 1, len(y)))
    cols = slice(max(col - CROP_PX, 0), min(col + CROP_PX + 1, len(x)))
    xx, yy = np.meshgrid(x[cols] * h, y[rows] * h)
    plon, plat = proj(xx, yy, inverse=True)
    dx = (plon - lon) * 111.32 * np.cos(np.radians(lat))  # km east of the station
    dy = (plat - lat) * 110.57  # km north
    return rows, cols, row - rows.start, col - cols.start, dx, dy


def row_time(nc, row):
    """time GOES imaged one row: rows spread evenly north to south over the scan (as in GOES19.row_times)."""
    start = pd.Timestamp(nc.time_coverage_start).tz_localize(None)
    end = pd.Timestamp(nc.time_coverage_end).tz_localize(None)
    y = nc["y"][:].data
    return start + (y.max() - y[row]) / (y.max() - y.min()) * (end - start)


def read(nc, name, rows, cols):
    """scaled window of one variable, fill values as nan."""
    return np.ma.filled(nc[name][rows, cols].astype(np.float32), np.nan)


def extract(bt_path, acm_path, lat, lon):
    """station / box values of one scan pair, plus the band 13 crop and its km grid."""
    # netCDF4 directly: xarray builds all ~90 variables of each file on open, ~1.5 s per file
    with netCDF4.Dataset(bt_path) as bt_nc, netCDF4.Dataset(acm_path) as acm_nc:
        if not (np.array_equal(bt_nc["x"][:], acm_nc["x"][:]) and np.array_equal(bt_nc["y"][:], acm_nc["y"][:])):
            raise ValueError("CMIPC and ACMC grids differ")
        rows, cols, r, c, dx, dy = station_window(bt_nc, lat, lon)
        bt = read(bt_nc, "CMI", rows, cols)
        bt = np.where(read(bt_nc, "DQF", rows, cols) <= 1, bt, np.nan)  # good or conditionally usable
        cloudy = read(acm_nc, "BCM", rows, cols)
        cloudy = np.where(read(acm_nc, "DQF", rows, cols) == 0, cloudy, np.nan)  # good quality only
        pixel_time = row_time(bt_nc, rows.start + r)
    box = (np.abs(dx) <= BOX_KM / 2) & (np.abs(dy) <= BOX_KM / 2)
    values = {"pixel_time": pixel_time, "bt13_station": bt[r, c], "bt13_box_mean": np.nanmean(bt[box]),
              "bt13_box_std": np.nanstd(bt[box]), "cloud_fraction_box": np.nanmean(cloudy[box]),
              "cloudy_station": cloudy[r, c],
              "n_box_pixels": int(box.sum()), "n_box_valid_acm": int(np.isfinite(cloudy[box]).sum())}
    return values, bt, dx, dy


def cache_paths(code, d):
    folder = table_dir("deep_dive", code, "goes")
    return folder / f"{code}_goes_{d:%Y-%m-%d}.csv", folder / f"{code}_goes_{d:%Y-%m-%d}_bt13_crops.npz"


def station_day(code, d, scans, meta, fs):
    """extract every scan of one station-day not yet cached (or failed before); returns the table."""
    csv, npz = cache_paths(code, d)
    old = pd.read_csv(csv, parse_dates=["scan_start", "pixel_time"]) if csv.exists() else pd.DataFrame()
    crops = dict(np.load(npz)) if npz.exists() else {}
    done = set(old.loc[old["status"] == "ok", "scan_start"]) if len(old) else set()
    todo = scans[~scans["scan_start"].isin(done)]
    if todo.empty:
        return old
    print(f"  {code} {d:%Y-%m-%d}: {len(todo)} scans to download ({len(done)} cached)", flush=True)
    rows, sizes = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        jobs = [(s, pool.submit(download, fs, s.bt), pool.submit(download, fs, s.acm)) for s in todo.itertuples()]
        # netCDF/HDF5 reads aren't thread-safe: download in parallel, extract one at a time
        for s, bt_job, acm_job in jobs:
            row = {"scan_start": s.scan_start, "bt_file": Path(s.bt).name, "acm_file": Path(s.acm).name}
            paths = []
            try:
                paths = [bt_job.result(), acm_job.result()]
                sizes += [p.stat().st_size for p in paths]
                values, bt, dx, dy = extract(*paths, meta["latitude"], meta["longitude"])
                row.update(values, status="ok")
                crops[f"{s.scan_start:%Y%m%dT%H%M}"] = bt.astype(np.float32)
                crops["dx_km"], crops["dy_km"] = dx.astype(np.float32), dy.astype(np.float32)
            except (OSError, RuntimeError, ValueError, IndexError, KeyError) as error:  # s3 / netCDF errors
                row["status"] = f"failed: {type(error).__name__}: {error}"[:200]
                print(f"    {s.scan_start} {row['status']}", flush=True)
            finally:
                for p in paths:
                    p.unlink(missing_ok=True)
                for job in (bt_job, acm_job):  # a download that finished after the other failed
                    if job.done() and job.exception() is None:
                        job.result().unlink(missing_ok=True)
            rows.append(row)
    new = pd.DataFrame(rows)
    keep = old[~old["scan_start"].isin(new["scan_start"])] if len(old) else old
    out = pd.concat([keep, new], ignore_index=True).sort_values("scan_start").reset_index(drop=True)
    out.to_csv(csv, index=False)
    np.savez_compressed(npz, **crops)
    station_day.downloaded_bytes += sum(sizes)
    station_day.downloaded_files += len(sizes)
    return out


station_day.downloaded_bytes = 0
station_day.downloaded_files = 0


def probe_sizes(clients, fs, start):
    """download one file of each product, print its size; stop if either is over MAX_FILE_MB."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    scans = list_scans(clients, start, start + pd.Timedelta(minutes=30))
    sizes, seconds = {}, {}
    for name in PRODUCTS:
        t0 = time.time()
        path = download(fs, scans[name].iloc[0])
        seconds[name] = time.time() - t0
        sizes[name] = path.stat().st_size / 1e6
        path.unlink()
        print(f"probe {name}: {path.name}  {sizes[name]:.2f} MB  {seconds[name]:.1f} s")
    if max(sizes.values()) > MAX_FILE_MB:
        sys.exit(f"a probe file is over {MAX_FILE_MB} MB ({sizes}); stopping before the full run")
    return sizes, seconds


# --- ground join -------------------------------------------------------------------------------

def join_ground(goes, k10, alpha, longitude, d):
    """10-min corrected k (and within-interval k std) of the interval containing each scan's station time."""
    goes = goes[goes["status"] == "ok"].copy()
    goes["interval"] = pd.to_datetime(goes["pixel_time"]).dt.floor(SCAN_STEP).dt.tz_localize("UTC")
    ground = pd.DataFrame({"k": alpha_corrected_k(k10, alpha), "k_std": k10["k_std_1min"] / alpha})
    goes = goes.join(ground, on="interval")
    goes["solar_hour"] = ((goes["interval"].dt.tz_localize(None) + pd.Timedelta(minutes=5) - d)
                          .dt.total_seconds() / 3600 + longitude / 15)
    return goes


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fmt_r(r):
    return "r = n/a" if np.isnan(r) else f"r = {r:+.2f}"


# --- figures -----------------------------------------------------------------------------------

def case_day_title(code, df, cases, joined):
    """the finding: agreement on clear / overcast days, mismatch on variable days."""
    date = solar_date(df.index, df.attrs["longitude"])
    mean_cf = {t: np.mean([joined[d]["cloud_fraction_box"].mean() for d in cases.loc[cases["day_type"] == t, "date"]])
               for t in ["clear", "overcast"]}
    parts = []
    for _, case in cases[cases["day_type"] == "variable"].iterrows():
        d, g = case["date"], joined[case["date"]]
        k = alpha_corrected_k(df[(date == d) & daylight(df)], df.attrs["clearsky_alpha"])
        lo, hi = k.quantile([0.05, 0.95])
        parts.append(f"{d:%d %b}: median box cloud fraction {g['cloud_fraction_box'].median():.2f}, "
                     f"1-min k {lo:.1f}–{hi:.1f}, {fmt_r(corr(g['cloud_fraction_box'].values, g['k'].values))}")
    return (f"{label(code)}: GOES-19 agrees with the ground on clear and overcast days (mean box cloud fraction "
            f"{mean_cf['clear']:.2f} and {mean_cf['overcast']:.2f}), but on variable days box cloud fraction\n"
            f"moves slowly while 1-min k swings between overcast and clear levels ({'; '.join(parts)})")


def plot_case_days(code, meta, df, cases, joined):
    """6 panels (rows: clear, variable, overcast; columns: warm, cold): 1-min and 10-min corrected k,
    GOES box cloud fraction on a twin axis, and a band 13 strip above each panel."""
    alpha, lon = df.attrs["clearsky_alpha"], meta["longitude"]
    title = case_day_title(code, df, cases, joined)
    fig = plt.figure(figsize=(17, 14.5))
    outer = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.28, top=0.9, bottom=0.15)
    date = solar_date(df.index, lon)
    color = COLORS[code]
    mesh = None
    for i, t in enumerate(DAY_TYPES):
        for j, season in enumerate(SEASONS):
            case = cases[(cases["day_type"] == t) & (cases["season"] == season)].iloc[0]
            d = case["date"]
            g = joined[d]
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[0.45, 4], hspace=0.06)
            strip = fig.add_subplot(inner[0])
            ax = fig.add_subplot(inner[1], sharex=strip)
            day = df[(date == d) & daylight(df)]
            hours = (day.index.tz_localize(None) - d).total_seconds() / 3600 + lon / 15
            ax.plot(hours, alpha_corrected_k(day, alpha), color=color, lw=0.6, alpha=0.45)
            ax.plot(g["solar_hour"], g["k"], color=color, lw=2.2, marker="o", ms=3.5)
            ax.set_ylim(0, 1.5)
            ax.set_ylabel("clear-sky index k", fontsize=12)
            tw = ax.twinx()
            tw.plot(g["solar_hour"], g["cloud_fraction_box"], color="k", lw=1.6, ls="--", marker="s", ms=3)
            tw.set_ylim(-0.03, 1.08)
            tw.set_ylabel("GOES box cloud fraction", fontsize=12)
            # band 13 as a gray strip, cold (high cloud tops) bright as in IR imagery
            edges = np.r_[g["solar_hour"].values - 5 / 60, g["solar_hour"].values[-1] + 5 / 60]
            mesh = strip.pcolormesh(edges, [0, 1], g["bt13_station"].values[None, :], cmap="Greys",
                                    vmin=BT_RANGE[0], vmax=BT_RANGE[1])
            strip.set_yticks([])
            strip.tick_params(labelbottom=False)
            r = corr(g["cloud_fraction_box"].values, g["k"].values)
            strip.set_title(f"{season} {t}: {d:%d %b %Y}   day k = {case['k']:.2f}, VI = {case['vi']:.2f}   "
                            f"cloud fraction vs k {fmt_r(r)}", fontsize=12, loc="left")
            ax.set_xlim(hours.min() - 0.2, hours.max() + 0.2)
            ax.tick_params(labelsize=11)
            tw.tick_params(labelsize=11)
            if i == 2:
                ax.set_xlabel("local solar time (h)", fontsize=12)
    handles = [plt.Line2D([], [], color=color, lw=0.8, alpha=0.6, label="k, 1 min"),
               plt.Line2D([], [], color=color, lw=2.2, marker="o", ms=4, label="k, 10 min (GOES scan times)"),
               plt.Line2D([], [], color="k", lw=1.6, ls="--", marker="s", ms=4,
                          label=f"GOES-19 cloud fraction, {BOX_KM} km box (right axis)")]
    fig.legend(handles=handles, loc="upper left", ncol=3, fontsize=12, bbox_to_anchor=(0.1, 0.075), frameon=False)
    cax = fig.add_axes([0.13, 0.03, 0.3, 0.01])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("gray strips above panels: band 13 at the station pixel (K); bright = cold cloud top", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.suptitle(title, x=0.01, ha="left", y=0.955, fontsize=15, color=color)
    path = figure_dir("deep_dive", code) / f"goes_case_days_{code}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(pooled, stats):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [("cloud_fraction_box", "k", f"GOES-19 cloud fraction in {BOX_KM} km box",
               "clear-sky index k (10 min, alpha-corrected)", "r_cf_k"),
              ("bt13_box_std", "k_std", f"band 13 std in {BOX_KM} km box (K)",
               "std of 1-min k within the 10 min", "r_std")]
    for ax, (xcol, ycol, xlabel, ylabel, key) in zip(axes, panels):
        for code in CODES:
            g = pooled[pooled["station"] == code]
            ax.scatter(g[xcol], g[ycol], s=14, alpha=0.45, color=COLORS[code], edgecolor="none",
                       label=f"{label(code)}: {fmt_r(stats.loc[code, key])}, "
                             f"variable days {fmt_r(stats.loc[code, key + '_var'])}")
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"pooled {fmt_r(stats.loc['pooled', key])}; variable days only "
                     f"{fmt_r(stats.loc['pooled', key + '_var'])}", fontsize=13)
        ax.legend(fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False,
                  markerscale=2)
    v = pooled[pooled["day_type"] == "variable"]
    sunny_cloud = ((v["cloud_fraction_box"] >= 0.95) & (v["k"] >= 0.8)).sum()
    dim_clear = ((v["cloud_fraction_box"] <= 0.05) & (v["k"] < 0.6)).sum()
    axes[0].text(0.5, 1.45, f"variable-day scans ({len(v)}): {sunny_cloud} with box cloud fraction ≥ 0.95 but k ≥ 0.8,\n"
                 f"{dim_clear} with cloud fraction ≤ 0.05 but k < 0.6", ha="center", va="top", fontsize=11,
                 bbox={"boxstyle": "round", "fc": "white", "ec": "0.6"})
    axes[0].set_ylim(0, 1.55)
    axes[1].set_ylim(bottom=0)
    fig.suptitle(stats.attrs["title"], x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figure_dir("deep_dive", "comparison") / "goes_vs_ground_k.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strip(code, case, g):
    """3 band 13 frames 20 min apart around the biggest 10-min k swing between STRIP_HOURS, plus the k series."""
    _, npz = cache_paths(code, case["date"])
    crops = np.load(npz)
    dx, dy = crops["dx_km"], crops["dy_km"]
    g = g.dropna(subset=["k"]).reset_index(drop=True)
    swing = pd.Series(np.abs(np.diff(g["k"].values, prepend=np.nan)), index=g.index)
    swing = swing[g["solar_hour"].between(*STRIP_HOURS)]
    i = int(np.clip(swing.idxmax(), 2, len(g) - 3))
    picks = [i - 2, i, i + 2]
    frames = [crops[f"{pd.Timestamp(g['scan_start'].iloc[p]):%Y%m%dT%H%M}"] for p in picks]
    vmin, vmax = np.nanpercentile(np.stack(frames), [1, 99.5])  # stretch so low cloud shows
    s = g.iloc[picks]
    title = (f"{label(code)}, {case['date']:%d %b %Y} (day VI = {case['vi']:.1f}): in 40 min k goes "
             f"{', '.join(f'{v:.2f}' for v in s['k'])} while GOES-19 box cloud fraction reads "
             f"{', '.join(f'{v:.2f}' for v in s['cloud_fraction_box'])}")
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.6], hspace=0.3, wspace=0.08)
    mesh = None
    for n, (p, frame) in enumerate(zip(picks, frames)):
        s = g.iloc[p]
        ax = fig.add_subplot(gs[0, n])
        mesh = ax.pcolormesh(dx, dy, frame, cmap="Greys", vmin=vmin, vmax=vmax, shading="nearest")
        half = BOX_KM / 2
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="tab:red", lw=1.5)
        ax.plot(0, 0, marker="*", color=COLORS[code], ms=16, mec="k")
        ax.set_xlim(-CROP_KM / 2, CROP_KM / 2)
        ax.set_ylim(-CROP_KM / 2, CROP_KM / 2)
        ax.set_aspect("equal")
        ax.set_xlabel("km east of station", fontsize=12)
        if n == 0:
            ax.set_ylabel("km north of station", fontsize=12)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{s['solar_hour']:.2f} h solar ({pd.Timestamp(s['scan_start']):%H:%M} UTC)\n"
                     f"k = {s['k']:.2f}, cloud fraction = {s['cloud_fraction_box']:.2f}", fontsize=12)
    cb = fig.colorbar(mesh, ax=fig.axes, orientation="vertical", fraction=0.02, pad=0.01)
    cb.set_label("band 13 brightness temperature (K); bright = cold cloud top", fontsize=12)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(g["solar_hour"], g["k"], color=COLORS[code], marker="o", ms=4, label="k, 10 min")
    tw = ax.twinx()
    tw.plot(g["solar_hour"], g["cloud_fraction_box"], color="k", ls="--", lw=1.5, label="box cloud fraction")
    tw.set_ylim(-0.03, 1.08)
    tw.set_ylabel("cloud fraction", fontsize=12)
    for p in picks:
        ax.axvline(g["solar_hour"].iloc[p], color="tab:red", lw=1.2, ls=":")
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("local solar time (h)", fontsize=12)
    ax.set_ylabel("clear-sky index k", fontsize=12)
    ax.legend(handles=ax.get_lines()[:1] + tw.get_lines() + [plt.Line2D([], [], color="tab:red", ls=":",
                                                                       label="frames above")],
              loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=11, ncol=3, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, color=COLORS[code])
    path = figure_dir("deep_dive", code) / f"goes_bt13_strip_{code}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- main --------------------------------------------------------------------------------------

def main():
    clients = {name: GOES(satellite=19, product=product, domain="C", bands=band)
               for name, (product, band) in PRODUCTS.items()}
    fs = s3fs.S3FileSystem(anon=True)
    stations = {code: load_station_1min(code) for code in CODES}
    cases = {code: case_days(df) for code, (_, df) in stations.items()}
    for code, c in cases.items():
        print(f"\n{label(code)} case days\n{c.to_string(index=False)}")

    first = cases[CODES[0]].iloc[0]
    sizes, seconds = probe_sizes(clients, fs, daylight_window(stations[CODES[0]][1], first["date"])[0])

    scans = {}
    for code, c in cases.items():
        for d in c["date"]:
            scans[code, d] = list_scans(clients, *daylight_window(stations[code][1], d))
    n_pairs = sum(len(s) for s in scans.values())
    n_todo = 0
    for (code, d), s in scans.items():
        csv, _ = cache_paths(code, d)
        done = set(pd.read_csv(csv, parse_dates=["scan_start"]).query("status == 'ok'")["scan_start"]) \
            if csv.exists() else set()
        n_todo += (~s["scan_start"].isin(done)).sum()
    volume_gb = n_todo * sum(sizes.values()) / 1e3
    serial_min = n_todo * sum(seconds.values()) / 60  # parallel downloads share bandwidth, so the real time is in between
    print(f"\n{n_pairs} scan pairs over {len(scans)} station-days, {n_todo} not cached: ~{volume_gb:.1f} GB to "
          f"download, ~{serial_min / WORKERS:.0f}-{serial_min:.0f} min ({WORKERS} parallel downloads vs serial)",
          flush=True)

    try:
        tables = {key: station_day(*key, s, stations[key[0]][0], fs) for key, s in scans.items()}
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\ndownloaded {station_day.downloaded_files} files, {station_day.downloaded_bytes / 1e9:.2f} GB")

    failed = pd.concat([t.assign(station=k[0]) for k, t in tables.items()])
    failed = failed[failed["status"] != "ok"]
    print(f"failed scans: {len(failed)}")
    if len(failed):
        print(failed[["station", "scan_start", "status"]].to_string(index=False))
    for (code, d), s in scans.items():
        missing = s.attrs["missing_slots"]
        if len(missing):
            print(f"{code} {d:%Y-%m-%d}: {len(missing)} daylight 10-min slots without a CMIPC+ACMC scan pair "
                  f"({', '.join(f'{t:%H:%M}' for t in missing[:12])}{' ...' if len(missing) > 12 else ''} UTC)")

    joined, frames = {}, []
    for code, (meta, df) in stations.items():
        k10 = to_10min(df)
        joined[code] = {}
        for _, case in cases[code].iterrows():
            g = join_ground(tables[code, case["date"]], k10, df.attrs["clearsky_alpha"], meta["longitude"],
                            case["date"])
            joined[code][case["date"]] = g
            frames.append(g.assign(station=code, day_type=case["day_type"], season=case["season"],
                                   date=case["date"]))
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(table_dir("deep_dive", "comparison") / "goes_vs_ground_k.csv", index=False)

    stats = {}
    for name, g in [*((code, pooled[pooled["station"] == code]) for code in CODES), ("pooled", pooled)]:
        v = g[g["day_type"] == "variable"]
        stats[name] = {"n": len(g), "r_cf_k": corr(g["cloud_fraction_box"].values, g["k"].values),
                       "r_std": corr(g["bt13_box_std"].values, g["k_std"].values),
                       "r_cf_k_var": corr(v["cloud_fraction_box"].values, v["k"].values),
                       "r_std_var": corr(v["bt13_box_std"].values, v["k_std"].values)}
    stats = pd.DataFrame(stats).T
    stats.to_csv(table_dir("deep_dive", "comparison") / "goes_vs_ground_k_correlations.csv")
    print(f"\ncorrelations (Pearson), all case days and variable days only\n{stats.round(3).to_string()}")

    # per-day correlations, for the figure titles
    per_day = pooled.groupby(["station", "date", "day_type", "season"]).apply(
        lambda g: pd.Series({"r_cf_k": corr(g["cloud_fraction_box"].values, g["k"].values),
                             "cf_mean": g["cloud_fraction_box"].mean(), "k_mean": g["k"].mean()}),
        include_groups=False).reset_index()
    print(f"\nper case day\n{per_day.to_string(index=False, float_format='%.2f')}")

    stats.attrs["title"] = (
        f"GOES-19 box cloud fraction separates clear from overcast scans (pooled {fmt_r(stats.loc['pooled', 'r_cf_k'])}), "
        f"but on variable days it explains\nlittle of k ({fmt_r(stats.loc['pooled', 'r_cf_k_var'])}), and band 13 "
        f"texture says almost nothing about minute-scale k swings ({fmt_r(stats.loc['pooled', 'r_std_var'])})")
    paths = [plot_comparison(pooled, stats)]
    for code, (meta, df) in stations.items():
        paths.append(plot_case_days(code, meta, df, cases[code], joined[code]))
        var = cases[code][cases[code]["day_type"] == "variable"].sort_values("vi").iloc[-1]
        paths.append(plot_strip(code, var, joined[code][var["date"]]))
    print("\nfigures:\n" + "\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
