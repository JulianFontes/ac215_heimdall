# Forecast vs Observation Video
"""
MP4 of side-by-side forecast cases: observed ERA5 clouds (left) next to the ConvLSTM forecast (right).

Takes the same chronological validation split as training (so every case is unseen), picks --cases evenly spaced
stretches of T_IN observed hours + K forecast hours, and animates the K forecast hours of each case. Each forecast
hour is a one-hour-ahead prediction from the T_IN observed hours before it (as in scripts/plot_forecasts.py).

Every frame shows any cloud (ice, liquid or rain) in two views, on one 0-100 % colour scale:
    map view    % of the altitude levels above each point that are cloudy (observed) / expected % (forecast)
    side view   height-longitude section: % of latitudes that are cloudy (observed) / expected % (forecast)

Example: python -m cloud_forecasting.scripts.forecast_video -p data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc \
             -c training_runs/training_runs/june2026_epoch100/convlstm3d_best.pt -k 6 --cases 4
(output defaults to forecast_vs_observed_k{K}.mp4 next to the checkpoint)
"""
import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from cloud_forecasting.config import K, N_CH, VAL_FRAC
from cloud_forecasting.data import load_channels
from cloud_forecasting.model import ConvLSTM3d
from cloud_forecasting.plots import lat_mean

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe() # bundled ffmpeg, no system install needed
SKY = LinearSegmentedColormap.from_list("sky", ["#0b1d33", "#3d5a7a", "#ffffff"]) # clear sky -> cloud
BG, FG, MUTED = "#0e1116", "#f2f2f2", "#9aa3ab"

def any_cloud(p):
    """(..., 3, alt, lat, lon) per-type probabilities or masks -> P(any type) (..., alt, lat, lon)."""
    return 1 - np.prod(1 - p, axis=-4)

def alt_mean(a, valid):
    """Average over altitude (axis -3), counting only bins inside the ERA5 column."""
    n = valid.sum(axis=-3)
    return np.where(n > 0, (a * valid).sum(axis=-3) / np.maximum(n, 1), np.nan)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--path", required=True, help="c3dir_like_*.nc")
    p.add_argument("-c", "--ckpt", required=True, help="convlstm3d_best.pt from scripts/train.py")
    p.add_argument("-k", "--k", type=int, default=K, help=f"forecast hours per case (default: K={K} from config.py)")
    p.add_argument("--cases", type=int, default=4, help="number of validation cases, evenly spaced (default 4)")
    p.add_argument("--fps", type=float, default=1.5, help="frames (forecast hours) per second (default 1.5)")
    p.add_argument("--out", default=None, help="output .mp4 (default: forecast_vs_observed_k{K}.mp4 next to the checkpoint)")
    a = p.parse_args()
    k = a.k
    out = Path(a.out or Path(a.ckpt).parent / f"forecast_vs_observed_k{k}.mp4")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #* Model: rebuilt from the settings stored in the checkpoint
    ckpt = torch.load(a.ckpt, map_location=device)
    t_in = ckpt["t_in"]
    model = ConvLSTM3d(N_CH, ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #* Data: same chronological split as training, so only unseen hours are used
    x = load_channels(a.path, ckpt["alt_max"])
    val = x.isel(time=slice(int(x.sizes["time"] * (1 - VAL_FRAC)), None))
    w = torch.from_numpy(val.values) # (hours, channel, alt, lat, lon)
    last = len(w) - t_in - k # latest case start
    if last < 0:
        raise SystemExit(f"validation set has {len(w)} hours, too few for T_IN={t_in} + K={k}")
    starts = np.unique(np.linspace(0, last, a.cases).round().astype(int))
    vtimes = pd.to_datetime(val.time.values)
    alt, lat, lon = x.alt.values, x.latitude.values, x.longitude.values
    print(f"device {device} | {len(starts)} cases of {t_in} observed + {k} forecast hours")

    #* Forecasts: one frame per forecast hour of every case
    frames = []
    for c, start in enumerate(starts):
        seq = w[start:start + t_in + k]
        x_in = torch.stack([seq[j:j + t_in, :N_CH] for j in range(k)]) # window j forecasts hour T_IN + j
        with torch.no_grad():
            prob = torch.sigmoid(model(x_in.to(device))).cpu().numpy() # (k, 3, alt, lat, lon)
        obs, valid = seq[t_in:, 3:N_CH].numpy(), seq[t_in:, N_CH].numpy() # (k, 3, ...), (k, alt, lat, lon)
        p_any, o_any = any_cloud(prob), any_cloud(obs)
        for j in range(k):
            frames.append(dict(case=c, hour=j, time=vtimes[start + t_in + j],
                               obs_map=alt_mean(o_any[j], valid[j]), fc_map=alt_mean(p_any[j], valid[j]),
                               obs_sec=lat_mean(o_any[j], valid[j]), fc_sec=lat_mean(p_any[j], valid[j])))

    #* Figure: observed (left) vs forecast (right), map view on top, side view (height-longitude) below
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), height_ratios=[1.6, 1], facecolor=BG, layout="constrained")
    (om, fm), (os_, fs) = axes
    ext = (lon.min(), lon.max(), lat.min(), lat.max())
    ims = {}
    for ax, key, title in ((om, "obs_map", "Actual (ERA5)"), (fm, "fc_map", "Predicted (ConvLSTM)")):
        ims[key] = ax.imshow(frames[0][key], origin="lower" if lat[0] < lat[-1] else "upper", extent=ext,
                             cmap=SKY, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
        ax.set_title(title, color=FG, fontsize=14, loc="left", fontweight="bold")
        ax.set_title("Map view · averaged over height", color=MUTED, fontsize=9, loc="right")
        ax.plot(-71.06, 42.36, marker="o", ms=5, mfc="none", mec="#ff6b5b", mew=1.5) # Boston
        ax.text(-71.0, 42.45, "Boston", color="#ff6b5b", fontsize=8)
        ax.set_xlabel("longitude", color=MUTED, fontsize=9); ax.set_ylabel("latitude", color=MUTED, fontsize=9)
    for ax, key in ((os_, "obs_sec"), (fs, "fc_sec")):
        ims[key] = ax.pcolormesh(lon, alt, frames[0][key], cmap=SKY, vmin=0, vmax=1, shading="auto")
        ax.set_xlabel("longitude", color=MUTED, fontsize=9); ax.set_ylabel("altitude (km)", color=MUTED, fontsize=9)
        ax.set_title("Side view · averaged north–south", color=MUTED, fontsize=9, loc="right")
    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2a313a")
    cb = fig.colorbar(ims["obs_map"], ax=axes, location="right", shrink=0.6, aspect=30, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cb.set_label("Cloudiness: % of grid cells with cloud\n(predicted = expected %)", color=FG, fontsize=10)
    cb.ax.tick_params(colors=MUTED, labelsize=9); cb.outline.set_edgecolor("#2a313a")
    cb.ax.text(0.5, 1.02, "Fully\ncloudy", transform=cb.ax.transAxes, ha="center", va="bottom", color=FG, fontsize=8)
    cb.ax.text(0.5, -0.02, "Clear", transform=cb.ax.transAxes, ha="center", va="top", color=FG, fontsize=8)
    header = fig.suptitle("", color=FG, fontsize=15, x=0.02, ha="left", fontweight="bold")
    # supxlabel, not fig.text, so the constrained layout leaves room for it under the panels
    fig.supxlabel(f"Each predicted hour uses the {t_in} actual hours before it (one hour ahead). "
             f"Validation data only: the model never trained on these hours.",
             color=MUTED, fontsize=8, x=0.02, ha="left")

    writer = FFMpegWriter(fps=a.fps, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=120):
        for f in frames:
            for key in ("obs_map", "fc_map"):
                ims[key].set_data(f[key])
            for key in ("obs_sec", "fc_sec"):
                ims[key].set_array(f[key].ravel())
            header.set_text(f"Case {f['case'] + 1}/{len(starts)}  ·  {f['time']:%b %d, %H:%M} UTC  ·  "
                            f"hour {f['hour'] + 1} of {k}")
            writer.grab_frame(facecolor=BG)
    plt.close(fig)
    print(f"saved {len(frames)} frames ({len(frames) / a.fps:.0f} s) to {out}")

if __name__ == "__main__":
    main()
