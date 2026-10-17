"""run a trained C3DIR model on a GOES-19 scan and read out the cloud column above one point.

  scene = goes_scene("latest") or goes_scene("2025-08-15T18:30")
  column = predict_column(model, scene, lat, lon)      DataFrame over the 80 levels
  region = predict_region(model, scene, lat, lon)      xarray (height, lat, lon)

figures for these are in c3dir.plots. scans are downloaded to CACHE_DIR/goes/

the column is built normal to the surface, as the paper does for its profile comparisons
(section 5.3): at each height, the voxel of the goes pixel whose line of sight crosses that
height above the point.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from goes2go import GOES

from c3dir.constants import BANDS, LEVELS_KM, PHASES, SPECIES
from c3dir.geometry import ray_latlon, scan_angles
from c3dir.paths import CACHE_DIR
from c3dir.model import decode_numpy

# SURFRAD stations (lat, lon)
SURFRAD = {
    "bon": ("Bondville, IL", 40.05, -88.37),
    "dra": ("Desert Rock, NV", 36.62, -116.02),
    "fpk": ("Fort Peck, MT", 48.31, -105.10),
    "gwn": ("Goodwin Creek, MS", 34.25, -89.87),
    "psu": ("Penn State, PA", 40.72, -77.93),
    "sxf": ("Sioux Falls, SD", 43.73, -96.62),
    "tbl": ("Table Mountain, CO", 40.12, -105.24),
}


def goes_scene(time="latest", domain="C"):
    """a goes-19 MCMIP scan (all 16 bands, 2 km): the latest, or the one nearest a UTC time.
    "C" (CONUS, every 5 min) downloads ~10x less than "F" (full disk)."""
    client = GOES(satellite=19, product="ABI-L2-MCMIP", domain=domain)
    kwargs = {"download": True, "return_as": "xarray", "save_dir": CACHE_DIR / "goes", "verbose": False}
    scene = client.latest(**kwargs) if time == "latest" else client.nearesttime(time, **kwargs)
    return scene.squeeze()


def crop(scene, lat, lon, size=128):
    """size x size goes pixels centered on the point (size a multiple of 8)."""
    p = scene["goes_imager_projection"].attrs
    x, y = scene.x.values, scene.y.values
    vx, vy = scan_angles(lat, lon, 0.0, p)
    if not np.isfinite(vx):
        raise ValueError(f"{lat}, {lon} is not visible from GOES-19")
    row, col = int(np.rint((vy - y[0]) / (y[1] - y[0]))), int(np.rint((vx - x[0]) / (x[1] - x[0])))
    if not (0 <= row < len(y) and 0 <= col < len(x)):
        raise ValueError(f"{lat}, {lon} is outside this scan; try the full disk (domain 'F')")
    r0 = int(np.clip(row - size // 2, 0, len(y) - size))
    c0 = int(np.clip(col - size // 2, 0, len(x) - size))
    return scene[BANDS + ["goes_imager_projection"]].isel(y=slice(r0, r0 + size), x=slice(c0, c0 + size)).load()


def predict_column(model, scene, lat, lon, size=128):
    """the predicted column above lat/lon as a DataFrame over the 80 levels: probability of any /
    ice / liquid / rain, water content (g m^-3) and effective radius (um), with each voxel's
    lat / lon."""
    patch = crop(scene, lat, lon, size)
    p = patch["goes_imager_projection"].attrs
    x, y = patch.x.values, patch.y.values
    vx, vy = scan_angles(np.full(len(LEVELS_KM), lat), np.full(len(LEVELS_KM), lon), LEVELS_KM * 1000.0, p)
    rows = np.clip(np.rint((vy - y[0]) / (y[1] - y[0])), 0, len(y) - 1).astype(int)
    cols = np.clip(np.rint((vx - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(int)
    sample = xr.Dataset({"goes": (("band", "y", "x"), np.stack([patch[b].values for b in BANDS]))})
    raw = model.predict_voxels(sample, (np.arange(len(LEVELS_KM)), rows, cols), window=size, margin=0)
    # where each voxel sits: the chosen pixel's line of sight at that height
    voxel_lat, voxel_lon = ray_latlon(x[cols], y[rows], LEVELS_KM * 1000.0, p)
    table = pd.DataFrame({"height_km": LEVELS_KM, "lat": voxel_lat, "lon": voxel_lon})
    return pd.concat([table, _decode(raw)], axis=1)


def _decode(raw):
    """raw outputs (80, 8) -> named columns, units in the names."""
    probability, water, radius = decode_numpy(raw)
    columns = {f"prob_{s}": probability[:, k] for k, s in enumerate(SPECIES)}
    columns |= {f"{w}_water_g_m3": water[:, k] for k, w in enumerate(PHASES)}
    columns |= {"ice_radius_um": radius[:, 0], "liquid_radius_um": radius[:, 1]}
    return pd.DataFrame(columns)


def summary(column):
    """cloud top (highest level with P(any) >= 0.5) and ice / liquid / rain water path (g m^-2)."""
    cloudy = column.loc[column["prob_any"] >= 0.5, "height_km"]
    paths = {f"{w}_water_path_g_m2": float((column[f"{w}_water_g_m3"] * 250.0).sum()) for w in PHASES}
    return {"cloud_top_km": float(cloudy.max()) if len(cloudy) else None, **paths}


def scan_time(scene):
    """start of the scan in UTC (goes2go keeps it as an attribute or a coordinate)."""
    value = scene.attrs.get("time_coverage_start", scene.get("time_coverage_start"))
    return pd.Timestamp(str(np.asarray(value).item() if hasattr(value, "values") else value)).tz_localize(None)


def sun_zenith(scene, lat, lon):
    """angle of the sun from overhead (degrees) at the scan time, NOAA's approximate equations
    (within ~0.5°, plenty for a day / night check)."""
    time = scan_time(scene)
    gamma = 2 * np.pi / 365 * (time.dayofyear - 1 + (time.hour - 12) / 24)  # fractional year, rad
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))  # minutes
    solar_minutes = time.hour * 60 + time.minute + time.second / 60 + eqtime + 4 * lon
    hour_angle = np.radians(solar_minutes / 4 - 180)
    lat = np.radians(lat)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return float(np.degrees(np.arccos(np.clip(cos_zenith, -1, 1))))


def check_daylight(scene, lat, lon):
    zenith = sun_zenith(scene, lat, lon)
    if zenith > 85:
        warnings.warn(f"sun is {zenith:.0f}° from overhead: the visible bands are dark, and the model was "
                      "trained on daytime passes only, so this prediction is unreliable")
    return zenith


def predict_region(model, scene, lat, lon, half_width_deg=1.0, step_deg=0.02, size=192):
    """the predicted 3D cloud field around lat/lon on a regular grid: (height, lat, lon), with
    columns normal to the surface (paper section 3.1). each grid cell at each height takes the
    voxel of the goes pixel whose line of sight crosses that height above it. half_width_deg sets
    the north-south half width; east-west is widened by 1 / cos(lat) so the area is about square."""
    patch = crop(scene, lat, lon, size)
    p = patch["goes_imager_projection"].attrs
    x, y = patch.x.values, patch.y.values
    lats = np.arange(lat - half_width_deg, lat + half_width_deg + step_deg / 2, step_deg)
    lon_half = half_width_deg / np.cos(np.radians(lat))
    lons = np.arange(lon - lon_half, lon + lon_half + step_deg / 2, step_deg)
    z, la, lo = np.meshgrid(LEVELS_KM * 1000.0, lats, lons, indexing="ij")
    vx, vy = scan_angles(la.ravel(), lo.ravel(), z.ravel(), p)
    rows, cols = np.rint((vy - y[0]) / (y[1] - y[0])), np.rint((vx - x[0]) / (x[1] - x[0]))
    inside = (rows >= 0) & (rows < len(y)) & (cols >= 0) & (cols < len(x))
    level = np.broadcast_to(np.arange(len(LEVELS_KM))[:, None, None], z.shape).ravel()
    sample = xr.Dataset({"goes": (("band", "y", "x"), np.stack([patch[b].values for b in BANDS]))})
    raw = model.predict_voxels(sample, (level[inside], rows[inside].astype(int), cols[inside].astype(int)),
                               window=size, margin=0)
    probability, water, _ = decode_numpy(raw)
    shape = z.shape

    def field(values):
        out = np.full(len(inside), np.nan, np.float32)
        out[inside] = values
        return (("height_km", "lat", "lon"), out.reshape(shape))

    variables = {f"prob_{s}": field(probability[:, k]) for k, s in enumerate(SPECIES)}
    variables |= {f"{w}_water_g_m3": field(water[:, k]) for k, w in enumerate(PHASES)}
    # the goes image on the same grid, from each cell's ground position
    gx, gy = scan_angles(la[0].ravel(), lo[0].ravel(), 0.0, p)
    g_row = np.clip(np.rint((gy - y[0]) / (y[1] - y[0])), 0, len(y) - 1).astype(int)
    g_col = np.clip(np.rint((gx - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(int)
    for band in ["CMI_C01", "CMI_C02", "CMI_C03", "CMI_C13"]:
        variables[band] = (("lat", "lon"), patch[band].values[g_row, g_col].reshape(shape[1:]))
    return xr.Dataset(variables, coords={"height_km": LEVELS_KM, "lat": lats, "lon": lons},
                      attrs={"model": model.version, "scan_start": str(scan_time(scene)),
                             "center_lat": lat, "center_lon": lon})
