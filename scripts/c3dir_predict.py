"""predict 3D clouds at a point (e.g. a SURFRAD station) from one GOES-19 scan, with a trained
C3DIR model: the column above the point, and optionally the whole 3D field around it.

change the settings below and run the file. the scan is downloaded to data/goes/; figures and
the column CSV go to artifacts/figures/c3dir_predict/, the 3D field to data/c3dir_predictions/.
"""
import matplotlib.pyplot as plt

from heimdall.c3dir.model.c3dir import C3DIR
from heimdall.c3dir.plots.model_plots import plot_column, plot_region
from heimdall.c3dir.model.inference import (SURFRAD, check_daylight, goes_scene, predict_column, predict_region,
                                      scan_time, summary)
from heimdall.paths import DATA_DIR, FIGURES_DIR

# settings: change these
SITE = "psu"  # a SURFRAD code (bon, dra, fpk, gwn, psu, sxf, tbl), or None to use LAT / LON
LAT, LON = 40.72, -77.93  # used when SITE is None
NAME = None  # label and file name for LAT / LON, e.g. "Miami, FL"
TIME = "latest"  # "latest", or a UTC time like "2025-08-15T18:30"
DOMAIN = "C"  # GOES "C" CONUS (smaller download) or "F" full disk
MODEL_VERSION = "latest"  # or e.g. "v002"
REGION = True  # also predict the 3D field around the point
HALF_WIDTH_DEG = 1.0  # region size: +/- this many degrees of latitude (~110 km)
PLOT_DIR = FIGURES_DIR / "c3dir_predict"

name, lat, lon = SURFRAD[SITE] if SITE else (NAME or f"{LAT:.2f}, {LON:.2f}", LAT, LON)
slug = SITE or "".join(c if c.isalnum() else "_" for c in name.split(",")[0].lower()).strip("_")
model = C3DIR.load(MODEL_VERSION)
scene = goes_scene(TIME, DOMAIN)
zenith = check_daylight(scene, lat, lon)

column = predict_column(model, scene, lat, lon)
print(f"C3DIR {model.version} above {name}, GOES-19 scan {scan_time(scene):%Y-%m-%d %H:%M} UTC, "
      f"sun {zenith:.0f}° from overhead")
for key, value in summary(column).items():
    print(f"  {key}: {value if value is None else round(value, 2)}")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
fig = plot_column(column, scene, lat, lon, name, model.version)
out = PLOT_DIR / f"{slug}_{scan_time(scene):%Y%m%dT%H%M}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
column.to_csv(out.with_suffix(".csv"), index=False)
print(f"saved {out} and {out.with_suffix('.csv').name}")

if REGION:
    region = predict_region(model, scene, lat, lon, HALF_WIDTH_DEG)
    cube = DATA_DIR / "c3dir_predictions" / f"{out.stem}_region.nc"
    cube.parent.mkdir(parents=True, exist_ok=True)
    region.to_netcdf(cube, encoding={v: {"zlib": True} for v in region.data_vars})
    fig = plot_region(region, name)
    fig.savefig(out.with_name(f"{out.stem}_region.png"), dpi=150, bbox_inches="tight")
    print(f"saved {cube} {dict(region.sizes)} and {out.stem}_region.png")
plt.show()
