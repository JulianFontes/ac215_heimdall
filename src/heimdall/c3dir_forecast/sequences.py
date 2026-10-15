"""build time series of C3DIR output to forecast from: one 3D cloud field every 10 minutes over a
region (e.g. around a SURFRAD station), for every daytime GOES-19 scan in a set of days.

  days = training_days(model)                     the days the C3DIR model was trained on
  build_frames(model, "psu", days)                one .nc per time in <DATA_DIR>/c3dir_forecast/frames/psu/
  sequence = stack_frames("psu")                  (time, field, level, lat, lon), saved next to them

each frame is predict_region (heimdall.c3dir.model.inference) on that scan, averaged onto
coarser levels so the forecaster stays small. the raw GOES scans are deleted after use
unless keep_raw is set.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from heimdall.c3dir.constants import LEVELS_KM
from heimdall.c3dir.model.inference import SURFRAD, goes_scene, predict_region, scan_time, sun_zenith
from heimdall.paths import DATA_DIR

FORECAST_DIR = DATA_DIR / "c3dir_forecast"
FIELDS = ["prob_any", "ice_water_g_m3", "liquid_water_g_m3", "rain_water_g_m3"]  # what gets forecast


def training_days(model, splits=("train", "val"), fill=True):
    """the days a C3DIR model was trained on, from the sample files saved with it (each name holds
    the GOES scan start, _sYYYYJJJ...). fill=True returns every day from the first to the last, so
    the forecaster sees whole consecutive days, not only the ones with an EarthCARE pass."""
    names = [name for split in splits for name in model.data.get(split, [])]
    days = sorted({pd.to_datetime(m.group(1), format="%Y%j") for name in names
                   if (m := re.search(r"_s(\d{7})", name))})
    if not days:
        raise ValueError(f"C3DIR {model.version} has no training files with scan times; pass the days yourself")
    return list(pd.date_range(days[0], days[-1], freq="D")) if fill else days


def site_location(site=None, lat=None, lon=None):
    """(name, lat, lon) for a SURFRAD code or an explicit point."""
    if site:
        return SURFRAD[site]
    return f"{lat:.2f}, {lon:.2f}", lat, lon


def _zenith(time, lat, lon):
    return sun_zenith(xr.Dataset(attrs={"time_coverage_start": str(time)}), lat, lon)


def frame_times(days, lat, lon, cadence="10min", max_zenith=80.0):
    """every cadence step of the given days when the sun is within max_zenith of overhead at the
    point (C3DIR was trained on daytime passes only)."""
    times = pd.DatetimeIndex(np.concatenate([pd.date_range(d, d + pd.Timedelta("1D"), freq=cadence,
                                                           inclusive="left") for d in pd.to_datetime(days)]))
    return times[[_zenith(t, lat, lon) <= max_zenith for t in times]]


def coarsen_levels(region, every=4):
    """average every `every` levels (80 -> 20 levels of 1 km by default), keeping only FIELDS."""
    ds = region[FIELDS].coarsen(height_km=every, boundary="trim").mean()
    return ds.assign_coords(height_km=LEVELS_KM[: len(ds.height_km) * every].reshape(-1, every).mean(1))


def _remove_raw(scene):
    """delete the downloaded GOES file(s) of this scan."""
    stamp = f"_s{scan_time(scene):%Y%j%H%M}"
    for path in (DATA_DIR / "goes").rglob(f"*{stamp}*.nc"):
        path.unlink(missing_ok=True)


def frame_dir(slug):
    return FORECAST_DIR / "frames" / slug


def build_frames(model, site=None, days=None, lat=None, lon=None, cadence="10min", domain="C",
                 half_width_deg=1.28, step_deg=0.04, levels_every=4, max_zenith=80.0, max_offset="4min",
                 keep_raw=False):
    """run C3DIR on every daytime scan of the days and save one frame per time. cached: times
    already saved are skipped, so an interrupted run picks up where it stopped. returns the paths.

    half_width_deg / step_deg: region around the point (default 1.28° / 0.04°, 65 cells north-south,
    ~4 km, about the 2 km GOES pixels averaged 2x2). max_offset: skip a time if the nearest scan
    is further away than this (missing scans)."""
    name, lat, lon = site_location(site, lat, lon)
    slug = site or "".join(c if c.isalnum() else "_" for c in name).strip("_")
    days = training_days(model) if days is None else days
    out_dir = frame_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = frame_times(days, lat, lon, cadence, max_zenith)
    print(f"{len(times)} daytime times over {len(days)} days for {name}, saving to {out_dir}")
    paths = []
    for k, time in enumerate(times):
        path = out_dir / f"{time:%Y%m%dT%H%M}.nc"
        if not path.exists():
            try:
                scene = goes_scene(time.strftime("%Y-%m-%dT%H:%M"), domain)
            except Exception as error:  # noqa: BLE001 - missing scans happen; skip the time, keep going
                warnings.warn(f"{time}: no GOES scan ({error})")
                continue
            start = scan_time(scene)
            if abs(start - time) > pd.Timedelta(max_offset):
                warnings.warn(f"{time}: nearest scan is {start}, skipped")
            else:
                region = predict_region(model, scene, lat, lon, half_width_deg, step_deg,
                                        size=_crop_size(half_width_deg, lat))
                frame = coarsen_levels(region, levels_every).astype(np.float32)
                frame.attrs = {**region.attrs, "time": str(time), "site": name}
                frame.to_netcdf(path, encoding={v: {"zlib": True} for v in frame.data_vars})
            if not keep_raw:
                _remove_raw(scene)
            if not path.exists():
                continue
        paths.append(path)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} / {len(times)}")
    return paths


def _crop_size(half_width_deg, lat):
    """GOES crop (pixels, a multiple of 64) wide enough for the region plus slanted lines of sight
    up to 20 km (~2 km pixels, wider at high latitude)."""
    km = 2 * half_width_deg * 111 / np.cos(np.radians(lat)) + 2 * 40
    return int(np.ceil(km / 2 / 64) * 64) + 64


def stack_frames(slug, save=True):
    """all saved frames of a site as one dataset (time, level, lat, lon) per field."""
    paths = sorted(frame_dir(slug).glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frame_dir(slug)}; run build_frames first")
    frames = [frame.expand_dims(time=[pd.Timestamp(frame.attrs["time"])]) for frame in map(xr.load_dataset, paths)]
    sequence = xr.concat(frames, dim="time", combine_attrs="drop_conflicts")
    if save:
        sequence.to_netcdf(sequence_path(slug), encoding={v: {"zlib": True} for v in sequence.data_vars})
    return sequence


def sequence_path(slug):
    return FORECAST_DIR / f"{slug}_sequence.nc"


def load_sequence(slug):
    path = sequence_path(slug)
    return xr.load_dataset(path) if Path(path).exists() else stack_frames(slug)
