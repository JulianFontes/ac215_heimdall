# GOES True Colour + C13 Video
"""
MP4 of GOES ABI imagery in two stacked panels, with US state lines, for one or more MCMIPC_*_psu.nc files.

    top     true colour: C01 blue, C02 red, C03 near-IR as a stand-in for green (ABI has no green band); dark at night
    bottom  C13 clean IR window brightness temperature in grayscale, cold (cloud tops) white and warm (ground) dark

Files are concatenated in the order given, so passing the weekly files of a month plays the month through.

Example (quick preview, 12-20 UTC of the first day of each week, every 15 min):
    python -m goes_nowcasting.goes_video data/noaa-goes19/aug2025/MCMIPC_*.nc --start-hour 12 --hours 8 --stride 3
Example (everything, every 5-min scan):
    python -m goes_nowcasting.goes_video data/noaa-goes19/aug2025/MCMIPC_*.nc --out aug2025_bands.mp4
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import patheffects
from matplotlib.animation import FFMpegWriter

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe() # bundled ffmpeg, no system install needed
BG, FG, MUTED = "#0e1116", "#f2f2f2", "#9aa3ab"
OUTLINE = [patheffects.Stroke(linewidth=2.2, foreground="black"), patheffects.Normal()] # white lines readable on any colour
BANDS = ["CMI_C01", "CMI_C02", "CMI_C03", "CMI_C13"]

def true_color(ds):
    """(t, y, x, 3) uint8 RGB from C01/C02/C03 reflectance, using the CIMSS synthetic green and a 1/2.2 gamma."""
    r, b, nir = (np.clip(ds[v].values, 0, 1) for v in ("CMI_C02", "CMI_C01", "CMI_C03"))
    g = np.clip(0.45 * r + 0.1 * nir + 0.45 * b, 0, 1)
    rgb = np.power(np.stack([r, g, b], axis=-1), 1 / 2.2)
    return (np.nan_to_num(rgb) * 255).astype(np.uint8)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="MCMIPC_*_psu.nc files, played in the order given")
    p.add_argument("--start-hour", type=float, default=0, help="UTC hour of each file's first day to start at (default 0)")
    p.add_argument("--hours", type=float, default=None, help="hours to keep from each file (default: all)")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th scan (default 1 = every 5 min)")
    p.add_argument("--fps", type=float, default=12, help="frames per second (default 12)")
    p.add_argument("--out", default="goes_bands.mp4", help="output .mp4 (default goes_bands.mp4)")
    a = p.parse_args()

    #* Data: the requested slice of the four bands from every file
    clips = []
    for path in a.paths:
        ds = xr.open_dataset(path)
        t0 = pd.Timestamp(ds.t.values[0]).normalize() + pd.Timedelta(hours=a.start_hour)
        t1 = t0 + pd.Timedelta(hours=a.hours) if a.hours else None
        clips.append(ds[BANDS].sel(t=slice(t0, t1)).isel(t=slice(None, None, a.stride)).load())
    data = xr.concat(clips, dim="t")
    print(f"{len(data.t)} frames from {len(a.paths)} files")
    rgb = true_color(data)
    ir = data["CMI_C13"].values
    ir_lo, ir_hi = np.nanpercentile(ir[::10], [2, 98])

    #* Geometry: ABI fixed grid (scan angle in radians) -> geostationary projection metres
    proj = ds.goes_imager_projection.attrs
    h = proj["perspective_point_height"]
    crs = ccrs.Geostationary(central_longitude=proj["longitude_of_projection_origin"], satellite_height=h,
                             sweep_axis=proj["sweep_angle_axis"],
                             globe=ccrs.Globe(semimajor_axis=proj["semi_major_axis"], semiminor_axis=proj["semi_minor_axis"]))
    x, y = data.x.values * h, data.y.values * h
    ext = (x.min(), x.max(), y.min(), y.max())
    origin = "upper" if data.y.values[0] > data.y.values[-1] else "lower"
    lat, lon = ds.attrs.get("site_lat"), ds.attrs.get("site_lon")

    #* Figure: true colour on top, C13 grayscale below, each with state lines and site marker
    fig, (top, bot) = plt.subplots(2, 1, figsize=(6, 11), facecolor=BG, subplot_kw={"projection": crs}, layout="constrained")
    rgb_im = top.imshow(rgb[0], extent=ext, origin=origin, transform=crs, interpolation="nearest")
    ir_im = bot.imshow(ir[0], extent=ext, origin=origin, transform=crs, cmap="gray_r", vmin=ir_lo, vmax=ir_hi,
                       interpolation="nearest")
    for ax, title in ((top, "True colour"),
                                (bot, "Band 13")):
        ax.set_extent(ext, crs=crs)
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="white", facecolor="none", linewidth=0.9, path_effects=OUTLINE)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="white", linewidth=0.8, path_effects=OUTLINE)
        if lat is not None:
            ax.plot(lon, lat, marker="o", ms=6, mfc="none", mec="white", mew=1.5, path_effects=OUTLINE, transform=ccrs.PlateCarree())
        ax.set_title(title, color=FG, fontsize=12, fontweight="bold", loc="left")
        cb = fig.colorbar(ir_im, ax=ax, location="right", shrink=0.85, aspect=30)
        if ax is top: # RGB has no colour scale; keep the space so both maps stay the same size
            cb.ax.set_visible(False)
        else:
            cb.set_label("K", color=MUTED, fontsize=9)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
    title = fig.suptitle("", color=FG, fontsize=13, fontweight="bold")

    #* Video
    out = Path(a.out)
    times = pd.to_datetime(data.t.values)
    writer = FFMpegWriter(fps=a.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=120):
        for i, t in enumerate(times):
            rgb_im.set_data(rgb[i])
            ir_im.set_data(ir[i])
            title.set_text(f"GOES-19  {t:%Y-%m-%d %H:%M} UTC")
            writer.grab_frame(facecolor=BG)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
