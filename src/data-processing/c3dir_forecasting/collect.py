"""step 1 of the C3DIR forecasting data: GOES-19 full-disk scans over a month, cropped around
several sites. step 2, c3dir_forecasting.predict in model-deploy, runs C3DIR on the crops.

for every scan time (default: every 10 min from 17:40 to 20:00 UTC, the span of C3DIR's training
scans, each day of August 2025):
  1. download the full-disk scan (~340 MB) to CACHE_DIR/goes/, once for all sites
  2. for each site, save the GOES crop around it (all 16 bands) to
     PROCESSED_DIR/c3dir_forecast/goes/<site>/<YYYYmmddTHHMM>.nc
  3. delete the full-disk scan and log the time to PROCESSED_DIR/c3dir_forecast/manifest.csv

cached: a time whose crops exist for every site is skipped without downloading, so an interrupted
run picks up where it stopped. a time with no scan within MAX_OFFSET is logged and skipped.
GOES downloads need no login. change the settings below and run from the repo root:

  uv run --project src/data-processing python -m c3dir_forecasting.collect
"""
import time as clock
import warnings

import numpy as np
import pandas as pd
from goes2go import GOES

from c3dir.constants import BANDS
from c3dir.geometry import scan_angles
from c3dir.paths import CACHE_DIR, PROCESSED_DIR

# settings: change these
SITES = {  # slug: (name, lat, lon)
    "psu": ("Penn State, PA", 40.72, -77.93),
    "costa_rica": ("San José, Costa Rica", 9.93, -84.08),
    "manaus": ("Manaus, Brazil (Amazon)", -3.12, -60.02),
    "buenos_aires": ("Buenos Aires, Argentina", -34.60, -58.38),
    "chile_coast": ("Northern Chile coast, off Antofagasta", -23.50, -71.50),
}
DAYS = pd.date_range("2025-08-01", "2025-08-31")
FIRST, LAST = "17:40", "20:00"  # UTC scan times each day, inclusive: C3DIR's training scans span 17:40 - 20:00
CADENCE = "10min"  # full-disk scans start every 10 min
DOMAIN = "F"  # full disk: covers every site, and is what C3DIR was trained on
MAX_OFFSET = "3min"  # skip a time if the nearest scan starts further away than this
CROP_PIXELS = 320  # GOES crop (pixels, a multiple of 8): +/- 2° around the site plus slanted lines of sight
OUT_DIR = PROCESSED_DIR / "c3dir_forecast"


def slot_times(days=DAYS, first=FIRST, last=LAST, cadence=CADENCE):
    """every scan time to collect: each day, first to last inclusive."""
    return pd.DatetimeIndex([t for day in days
                             for t in pd.date_range(f"{day:%Y-%m-%d} {first}", f"{day:%Y-%m-%d} {last}", freq=cadence)])


def crop_path(site, slot):
    return OUT_DIR / "goes" / site / f"{slot:%Y%m%dT%H%M}.nc"


def done(slot):
    return all(crop_path(site, slot).exists() for site in SITES)


def scan_time(scene):
    """start of the scan in UTC (goes2go keeps it as an attribute or a coordinate)."""
    value = scene.attrs.get("time_coverage_start", scene.get("time_coverage_start"))
    return pd.Timestamp(str(np.asarray(value).item() if hasattr(value, "values") else value)).tz_localize(None)


def crop(scene, lat, lon, size=CROP_PIXELS):
    """size x size goes pixels centered on the point, all 16 bands (the same cut as model-deploy's
    c3dir.inference.crop, so C3DIR sees exactly what it would on the whole scan)."""
    p = scene["goes_imager_projection"].attrs
    x, y = scene.x.values, scene.y.values
    vx, vy = scan_angles(lat, lon, 0.0, p)
    if not np.isfinite(vx):
        raise ValueError(f"{lat}, {lon} is not visible from GOES-19")
    row, col = int(np.rint((vy - y[0]) / (y[1] - y[0]))), int(np.rint((vx - x[0]) / (x[1] - x[0])))
    if not (0 <= row < len(y) and 0 <= col < len(x)):
        raise ValueError(f"{lat}, {lon} is outside this scan; use the full disk (domain 'F')")
    r0 = int(np.clip(row - size // 2, 0, len(y) - size))
    c0 = int(np.clip(col - size // 2, 0, len(x) - size))
    return scene[BANDS + ["goes_imager_projection"]].isel(y=slice(r0, r0 + size), x=slice(c0, c0 + size)).load()


def save(ds, path):
    """write atomically (a stopped run never leaves a half-written file that looks done)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    ds.to_netcdf(partial, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars
                                    if ds[v].dtype.kind == "f"})
    partial.replace(path)


def _remove_scan(scan):
    """delete the downloaded full-disk file(s) of this scan."""
    for path in (CACHE_DIR / "goes").rglob(f"*MCMIP{DOMAIN}*_s{scan:%Y%j%H%M}*.nc"):
        path.unlink(missing_ok=True)


def collect_slot(slot):
    """download the scan nearest slot and save every site's crop. returns a manifest row."""
    row = {"slot": slot, "scan_start": None, "status": "ok", "seconds": None}
    start = clock.time()
    client = GOES(satellite=19, product="ABI-L2-MCMIP", domain=DOMAIN)
    try:
        scene = client.nearesttime(slot.strftime("%Y-%m-%dT%H:%M"), download=True, return_as="xarray",
                                   save_dir=CACHE_DIR / "goes", verbose=False).squeeze()
    except Exception as error:  # noqa: BLE001 - missing scans happen; log the time, keep going
        return row | {"status": f"no scan: {error}"}
    scan = scan_time(scene)
    try:
        row["scan_start"] = scan
        if abs(scan - slot) > pd.Timedelta(MAX_OFFSET):
            return row | {"status": f"nearest scan {scan} too far"}
        for site, (name, lat, lon) in SITES.items():
            path = crop_path(site, slot)
            if not path.exists():
                patch = crop(scene, lat, lon)
                patch.attrs = {"site": site, "site_name": name, "site_lat": lat, "site_lon": lon, "slot": str(slot),
                               "scan_start": str(scan), "source": scene.attrs.get("dataset_name", "")}
                save(patch, path)
    finally:
        scene.close()
        _remove_scan(scan)
    return row | {"seconds": round(clock.time() - start, 1)}


def main():
    slots = slot_times()
    todo = [slot for slot in slots if not done(slot)]
    print(f"{len(SITES)} sites, {len(slots)} scan times ({slots[0]:%Y-%m-%d} to {slots[-1]:%Y-%m-%d}, "
          f"{FIRST}-{LAST} UTC), {len(todo)} still to collect")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.csv"
    for k, slot in enumerate(todo, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = collect_slot(slot)
        pd.DataFrame([row]).to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
        print(f"  {k:>4} / {len(todo)}  {slot:%Y-%m-%d %H:%M}  {row['status']}"
              + (f"  {row['seconds']} s" if row["seconds"] else ""), flush=True)
    missing = [slot for slot in slots if not done(slot)]
    print(f"done: {len(slots) - len(missing)} / {len(slots)} times cropped for all sites"
          + (f"; missing {len(missing)} (see {manifest})" if missing else "")
          + "\nnext: uv run --project src/model-deploy python -m c3dir_forecasting.predict")


if __name__ == "__main__":
    main()
