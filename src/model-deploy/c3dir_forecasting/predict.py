"""step 2 of the C3DIR forecasting data: run a trained C3DIR model on every GOES crop saved by
step 1 (c3dir_forecasting.collect in data-processing) and save the 3D cloud field around each site.

  PROCESSED_DIR/c3dir_forecast/goes/<site>/<YYYYmmddTHHMM>.nc    in:  GOES crop, 16 bands
  PROCESSED_DIR/c3dir_forecast/c3dir/<site>/<YYYYmmddTHHMM>.nc   out: C3DIR field (height_km, lat, lon)

the site's position comes from each crop's attributes, so this needs no site list. cached: crops
that already have a field are skipped, so run it again after collecting more days. change the
settings below and run from the repo root:

  uv run --project src/model-deploy python -m c3dir_forecasting.predict
"""
import time as clock

import xarray as xr

from c3dir.inference import predict_region, sun_zenith
from c3dir.model import C3DIR
from c3dir.paths import PROCESSED_DIR

# settings: change these
MODEL_VERSION = "v002"
HALF_WIDTH_DEG = 2.0  # C3DIR field: +/- this many degrees of latitude (wider in longitude by 1 / cos lat)
STEP_DEG = 0.04  # grid spacing (~4 km)
SITES = None  # None: every site with crops; or e.g. ["psu", "manaus"]
DATA_DIR = PROCESSED_DIR / "c3dir_forecast"


def field_path(crop_path):
    """the C3DIR field for a GOES crop: goes/<site>/<time>.nc -> c3dir/<site>/<time>.nc."""
    return DATA_DIR / "c3dir" / crop_path.parent.name / crop_path.name


def predict_crop(model, crop_path):
    """C3DIR's 3D field around the crop's site, on the regular lat / lon grid."""
    patch = xr.load_dataset(crop_path)
    patch.attrs["time_coverage_start"] = patch.attrs["scan_start"]  # where c3dir.inference looks for the scan time
    lat, lon = patch.attrs["site_lat"], patch.attrs["site_lon"]
    region = predict_region(model, patch, lat, lon, HALF_WIDTH_DEG, STEP_DEG, size=patch.sizes["y"])
    keep = ("site", "site_name", "site_lat", "site_lon", "slot", "scan_start")
    region.attrs |= {k: patch.attrs[k] for k in keep} | {"sun_zenith_deg": round(sun_zenith(patch, lat, lon), 2)}
    return region


def save(ds, path):
    """write atomically (a stopped run never leaves a half-written file that looks done)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    ds.to_netcdf(partial, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    partial.replace(path)


def main():
    model = C3DIR.load(MODEL_VERSION)
    sites = SITES or sorted(p.name for p in (DATA_DIR / "goes").iterdir() if p.is_dir())
    crops = sorted(path for site in sites for path in (DATA_DIR / "goes" / site).glob("*.nc"))
    todo = [path for path in crops if not field_path(path).exists()]
    print(f"C3DIR {model.version}: {len(crops)} GOES crops over {len(sites)} sites, {len(todo)} still to predict")
    start = clock.time()
    for k, path in enumerate(todo, 1):
        save(predict_crop(model, path), field_path(path))
        if k % 25 == 0 or k == len(todo):
            print(f"  {k:>5} / {len(todo)}  {(clock.time() - start) / k:.1f} s each", flush=True)
    print(f"done: fields in {DATA_DIR / 'c3dir'}")


if __name__ == "__main__":
    main()
