# GOES Nowcast Export
"""
Run a trained ConvLSTM checkpoint on GOES-19 MCMIP frames and write the forecast to a netCDF file.

The model reads the T_IN frames ending at --start (default: the last frame in the data) and forecasts the next T_OUT
frames of config.TARGET_BANDS, STEP_MIN minutes apart. Inputs are normalized with the training mean / std stored in
the checkpoint, and the forecast is converted back to physical units (K, or reflectance factor for visible bands).

Output (--out, default goes_forecast.nc):
    CMI_CXX                  (time, y, x) forecast of every target band, NaN where the last input frame was invalid
    goes_imager_projection   copied from the input file, so the x / y scan angles can be mapped as in the input
    attributes               checkpoint, data, last input frame and when the forecast was made

Example: python -m goes_nowcasting.predict -p data/noaa-goes19/aug2025/MCMIPC_G19_20250801T0000_20250808T0000_psu.nc \
             -c training_runs/pcu_week1/convlstm_best.pt --start "2025-08-07 18:00"
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import xarray as xr

from goes_nowcasting.config import N_CH, STEP_MIN, TARGET_BANDS, TARGET_IDX, T_IN, T_OUT
from goes_nowcasting.model import ConvLSTM
from goes_nowcasting.preprocess import find_files, load_channels, scale

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--path", required=True, help="MCMIP .nc file, folder of .nc files, or glob pattern")
    p.add_argument("-c", "--ckpt", required=True, help="convlstm_best.pt from model-train's goes_nowcasting.train")
    p.add_argument("--start", default=None, help="last input frame, e.g. '2025-08-07 18:00' (default: the last frame in the data)")
    p.add_argument("--out", default="goes_forecast.nc", help="output netCDF file (default goes_forecast.nc)")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #* Model: weights and the training normalization statistics from the checkpoint
    ckpt = torch.load(a.ckpt, map_location=device)
    model = ConvLSTM().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["mean"].cpu().numpy(), ckpt["std"].cpu().numpy() # per band, BANDS order

    #* Data: the T_IN frames ending at --start, one scan apart
    x = load_channels(a.path)
    times = pd.to_datetime(x.time.values)
    s = times.get_indexer([pd.Timestamp(a.start)], method="nearest")[0] if a.start else len(times) - 1 # last input frame
    if s < T_IN - 1:
        raise SystemExit(f"need T_IN={T_IN} frames up to --start; the first possible start is {times[T_IN - 1]}")
    window = x.isel(time=slice(s - T_IN + 1, s + 1)).copy() # copy: scale works in place
    gaps = np.diff(window.time.values) / np.timedelta64(1, "m")
    if (gaps > 1.5 * STEP_MIN).any():
        raise SystemExit(f"the {T_IN} input frames up to {times[s]} are not consecutive (gap of {gaps.max():.0f} min)")
    print(f"input frames {times[s - T_IN + 1]} -> {times[s]}")

    #* Forecast, back in physical units
    inp = torch.from_numpy(scale(window, mean, std).values[None, :, :N_CH]).to(device) # (1, T_IN, N_CH, H, W)
    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy() # (T_OUT, n_out, H, W), normalized
    pred = pred * std[TARGET_IDX, None, None] + mean[TARGET_IDX, None, None]
    pred[:, :, ~window.sel(channel="valid").isel(time=-1).values.astype(bool)] = np.nan

    #* Export
    units = x["units"].values
    forecast_times = times[s] + pd.to_timedelta(STEP_MIN * np.arange(1, T_OUT + 1), unit="min")
    ds = xr.Dataset(
        {f"CMI_C{b:02d}": (("time", "y", "x"), pred[:, i], {"units": units[TARGET_IDX[i]]}) for i, b in enumerate(TARGET_BANDS)},
        coords={"time": forecast_times, "y": x.y.values, "x": x.x.values},
        attrs={"checkpoint": a.ckpt, "data": a.path, "last_input_frame": str(times[s]),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    with xr.open_dataset(find_files(a.path)[0]) as src:
        if "goes_imager_projection" in src:
            proj = src["goes_imager_projection"]
            ds["goes_imager_projection"] = (proj.isel(t=0) if "t" in proj.dims else proj).reset_coords(drop=True).load() # the attributes are what matter
    ds.to_netcdf(a.out)
    print(f"forecast {forecast_times[0]} -> {forecast_times[-1]} ({T_OUT} frames, bands {list(TARGET_BANDS)}) written to {a.out}")

if __name__ == "__main__":
    main()
