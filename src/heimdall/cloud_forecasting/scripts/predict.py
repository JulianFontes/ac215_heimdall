# Forecast Export for heimdall-frontend
"""
Run a trained ConvLSTM checkpoint for K forecast hours and export the result for heimdall-frontend (Kepler.gl),
in the same format as the frontend's sample data plus a cloud_type label.

Each forecast hour is a one-hour-ahead prediction from the T_IN observed hours before it (the same
convention as scripts/plot_forecasts.py), so the volume must cover T_IN hours before --start and
the K forecasted hours themselves. The --history observed hours before --start (default T_IN, the model's inputs)
are exported too, so the frontend timeline plays the observations into the forecast.

Every valid (hour, alt, lat, lon) bin is exported, with no thresholding (bins outside the ERA5 column are skipped):
    cloud_type   the most likely cloud type (ice / liquid / rain), or none where all three are 0 (observed clear air)
    confidence   probability of that cloud type
    density      100 * P(any cloud) = 100 * (1 - (1 - p_ice)(1 - p_liquid)(1 - p_rain)), in [0, 100]
    source       'observed' (ERA5 masks, so density is 0 or 100) or 'forecast' (model probabilities)

Outputs (in the --out folder, default heimdall-frontend/public/data/, overwriting the sample data):
    points.csv      lat, lon, alt (m, bin centre), density, confidence, timestamp, cloud_type, source
    voxels.geojson  one extruded square per bin, z = base altitude (m),
                    properties density, confidence, points, base_alt_m, height_m, timestamp, cloud_type, source
    meta.json       time range (observed + forecast hours), first forecast hour and map centre, read by the frontend,
                    plus a "run" record of how the forecast was made (checkpoint, data, settings; ignored by the frontend)

Example: python -m heimdall.cloud_forecasting.scripts.predict -p data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc \
             -c training_runs/testing/convlstm3d_best.pt -k 6 --start "2026-06-20 00:00"
(-k defaults to K in config.py, --start defaults to the latest start the volume allows)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
import xarray as xr

from heimdall.cloud_forecasting.config import CLOUD_PROPERTIES, K, N_CH, VAL_FRAC
from heimdall.cloud_forecasting.data import load_channels
from heimdall.cloud_forecasting.model import ConvLSTM3d

FRONTEND_DATA = Path(__file__).resolve().parents[2] / "heimdall-frontend" / "public" / "data"

def to_table(prob, valid, forecast_start):
    """prob: DataArray (time, cloud_type, alt, latitude, longitude) -> DataFrame with one row per valid bin.
    Hours before forecast_start are labelled 'observed', the rest 'forecast'."""
    df = prob.to_dataset(dim="cloud_type").to_dataframe().reset_index()[valid.values.ravel().astype(bool)]
    p = df[CLOUD_PROPERTIES].to_numpy("float64") # float64 so the rounded values print as e.g. 0.33, not 0.33000001
    return pd.DataFrame({
        "lat": df.latitude, "lon": df.longitude, "alt_km": df.alt,
        "density": (100 * (1 - np.prod(1 - p, axis=1))).round(2), "confidence": p.max(axis=1).round(3),
        "timestamp": df.time.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cloud_type": np.where(p.max(axis=1) > 0, np.array(CLOUD_PROPERTIES)[p.argmax(axis=1)], "none"),
        "source": np.where(df.time < forecast_start, "observed", "forecast"),
    })

def to_voxels(rows, d_lat, d_lon, d_alt_m):
    """One square per bin at its base altitude: Kepler extrudes it upwards by height_m."""
    z = (rows.alt_km * 1000).round().astype(int)
    x0, x1, y0, y1 = rows.lon - d_lon / 2, rows.lon + d_lon / 2, rows.lat - d_lat / 2, rows.lat + d_lat / 2
    ring = np.stack([np.c_[x0, y0, z], np.c_[x1, y0, z], np.c_[x1, y1, z], np.c_[x0, y1, z], np.c_[x0, y0, z]], axis=1) # (n, 5, xyz)
    props = rows.assign(points=1, base_alt_m=z, height_m=d_alt_m)[
        ["density", "confidence", "points", "base_alt_m", "height_m", "timestamp", "cloud_type", "source"]]
    return gpd.GeoDataFrame(props, geometry=shapely.polygons(ring), crs="EPSG:4326")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--path", required=True, help="c3dir_like_*.nc")
    p.add_argument("-c", "--ckpt", required=True, help="convlstm3d_best.pt from scripts/train.py")
    p.add_argument("-k", "--k", type=int, default=K, help=f"forecast hours (default: K={K} from config.py)")
    p.add_argument("--start", default=None, help="first forecasted hour, e.g. '2026-06-20 00:00' (default: latest possible)")
    p.add_argument("--history", type=int, default=None,
                   help="observed hours to export before the forecast (default: T_IN from the checkpoint, 0 = none)")
    p.add_argument("--out", default=str(FRONTEND_DATA), help="output folder (default: heimdall-frontend/public/data)")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #* Model: rebuilt from the settings stored in the checkpoint
    ckpt = torch.load(a.ckpt, map_location=device)
    t_in = ckpt["t_in"]
    model = ConvLSTM3d(N_CH, ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #* Data: the T_IN hours before --start plus the K forecasted hours
    x = load_channels(a.path, ckpt["alt_max"])
    times = pd.to_datetime(x.time.values)
    n, k = len(times), a.k
    s = times.get_loc(pd.Timestamp(a.start)) if a.start else n - k # index of the first forecasted hour
    if s < t_in or s + k > n:
        raise SystemExit(f"need T_IN={t_in} hours before --start and K={k} after it; "
                         f"forecast starts can run from {times[t_in]} to {times[n - k]}")
    h = t_in if a.history is None else min(a.history, s) # observed hours to export before the forecast
    in_training = s < int(n * (1 - VAL_FRAC))
    if in_training:
        print(f"note: {times[s]} is in the training period of this volume, so the model has seen these hours")

    w = torch.from_numpy(x.isel(time=slice(s - t_in, s + k)).values) # (T_IN + K, channel, alt, lat, lon)
    x_in = torch.stack([w[j:j + t_in, :N_CH] for j in range(k)]) # window j forecasts hour s + j
    with torch.no_grad():
        prob = torch.sigmoid(model(x_in.to(device))).cpu().numpy()
    print(f"device {device} | forecast {times[s]} -> {times[s + k - 1]} ({k} h) from {t_in} observed hours")

    #* Export: every valid bin of the h observed hours (their 0/1 masks) and the k forecasted hours (probabilities),
    #* as points and voxels, plus a small meta file for the frontend
    observed = x.isel(time=slice(s - h, s), channel=slice(3, N_CH)).values # (h, cloud_type, alt, lat, lon)
    prob = xr.DataArray(np.concatenate([observed, prob]), dims=("time", "cloud_type", "alt", "latitude", "longitude"),
                        coords={"time": times[s - h:s + k], "cloud_type": CLOUD_PROPERTIES,
                                "alt": x.alt, "latitude": x.latitude, "longitude": x.longitude})
    rows = to_table(prob, x.isel(time=slice(s - h, s + k), channel=N_CH), times[s])

    d_lat, d_lon = abs(float(x.latitude.diff("latitude").mean())), abs(float(x.longitude.diff("longitude").mean()))
    d_alt_m = round(float(x.alt.diff("alt").mean()) * 1000)
    (out / "voxels.geojson").write_text(to_voxels(rows, d_lat, d_lon, d_alt_m).to_json(drop_id=True, separators=(",", ":")))

    pts = rows.assign(alt=rows.alt_km * 1000 + d_alt_m / 2) # bin centre in metres
    pts[["lat", "lon", "alt", "density", "confidence", "timestamp", "cloud_type", "source"]].to_csv(out / "points.csv", index=False)

    ms = lambda t: int(t.value // 10**6)
    meta = {"t0_ms": ms(times[s - h]), "step_ms": 3600 * 1000, "frames": h + k, "forecast_t0_ms": ms(times[s]),
            "center": {"latitude": float(x.latitude.mean()), "longitude": float(x.longitude.mean())},
            "run": {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ckpt": a.ckpt, "data": a.path, "k": k, "start": a.start, "history": h, # start null = latest possible
                "forecast_hours": [str(times[s]), str(times[s + k - 1])],
                "in_training_period": in_training, # forecast hours fall in the 80% of the volume the model trained on
                "t_in": t_in, "hidden": list(ckpt["hidden"]), "alt_max": ckpt["alt_max"],
                "train_args": ckpt.get("args"), "device": device}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"saved {len(rows)} points / voxels ({rows.cloud_type.value_counts().to_dict()}) to {out}/")

if __name__ == "__main__":
    main()
