# Forecast Plotting
"""
Height-longitude forecast plots from a trained ConvLSTM checkpoint, for a chosen K.

Loads the checkpoint and the C3DIR-like volume, takes the same chronological validation split as training,
and on every validation stretch makes K one-hour-ahead forecasts in a row, saving one
plot_height_lon_compare figure per stretch (T_IN observed hours, then K forecasted hours).

Outputs (in the --out folder, default runs/forecast_height_lon_k{K}/):
    {YYYY-mm-dd_HH}h.png  one plot per stretch, named after its first forecasted hour

Example: python -m heimdall.cloud_forecasting.scripts.plot_forecasts -p c3dir_like_era5_20260601_20260615_N45.5_W74_S39.5_E68.nc \
             -c training_runs/15day_epoch50/convlstm3d_best.pt -k 4
(-k defaults to K in config.py)
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from heimdall.cloud_forecasting.config import K, N_CH, VAL_FRAC
from heimdall.cloud_forecasting.data import load_channels
from heimdall.cloud_forecasting.model import ConvLSTM3d
from heimdall.cloud_forecasting.plots import plot_height_lon_compare

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--path", required=True, help="c3dir_like_*.nc")
    p.add_argument("-c", "--ckpt", required=True, help="convlstm3d_best.pt from scripts/train.py")
    p.add_argument("-k", "--k", type=int, default=K, help=f"forecasted hours per plot (default: K={K} from config.py)")
    p.add_argument("--stride", type=int, default=None, help="hours between stretch starts (default: k, no overlap)")
    p.add_argument("--out", default=None, help="output folder (default runs/forecast_height_lon_k{K}, created if missing)")
    a = p.parse_args()

    k, stride = a.k, a.stride or a.k
    out = Path(a.out or f"runs/forecast_height_lon_k{k}")
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #* Model: rebuilt from the settings stored in the checkpoint
    ckpt = torch.load(a.ckpt, map_location=device)
    t_in = ckpt["t_in"]
    model = ConvLSTM3d(N_CH, ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #* Data: same chronological split as training, so only unseen hours are forecasted
    x = load_channels(a.path, ckpt["alt_max"])
    split = int(x.sizes["time"] * (1 - VAL_FRAC))
    val = x.isel(time=slice(split, None))
    w = torch.from_numpy(val.values) # (hours, channel, alt, lat, lon)

    starts = range(0, len(w) - t_in - k + 1, stride)
    if not starts:
        raise SystemExit(f"validation set has {len(w)} hours, too few for T_IN={t_in} + K={k}")
    print(f"device {device} | validation hours {len(w)} | {len(starts)} plots with K={k}")

    for start in tqdm(starts, desc="forecast plots"):
        seq = w[start:start + t_in + k] # T_IN input hours + k forecasted hours
        x_in = torch.stack([seq[j:j + t_in, :N_CH] for j in range(k)]) # window j forecasts hour T_IN + j
        with torch.no_grad():
            prob = torch.sigmoid(model(x_in.to(device))).cpu().numpy()
        times = pd.to_datetime(val.time.values[start:start + t_in + k])
        plot_height_lon_compare(seq[:, 3:N_CH].numpy(), prob, seq[:, N_CH].numpy(), x.alt.values, x.longitude.values,
                                times, out / f"{times[t_in]:%Y-%m-%d_%H}h.png")

    print(f"\nsaved {len(starts)} plots to {out}/")

if __name__ == "__main__":
    main()
