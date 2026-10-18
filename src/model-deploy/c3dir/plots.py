"""figures for C3DIR predictions on a GOES-19 scan: the column above a point and the 3D field
around it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from c3dir.constants import LEVELS_KM, PHASES, SPECIES
from c3dir.geometry import ray_latlon, scan_angles
from c3dir.inference import crop, scan_time
from utilities.plot_style import CLOUD_CMAP, SKY_COLOR, styled, true_color


@styled
def plot_column(column, scene, lat, lon, name, version):
    """band 13 around the point (left) and the predicted column above it (right)."""
    patch = crop(scene, lat, lon, 64)
    p = patch["goes_imager_projection"].attrs
    vx, vy = scan_angles(lat, lon, 0.0, p)
    x, y = patch.x.values, patch.y.values
    fig, (ax_img, ax_p, ax_w) = plt.subplots(1, 3, figsize=(17, 6.5), layout="constrained",
                                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    img = ax_img.imshow(patch["CMI_C13"].values, cmap="Greys", origin="upper")
    to_px = lambda ax_, ay_: ((ax_ - x[0]) / (x[1] - x[0]), (ay_ - y[0]) / (y[1] - y[0]))  # noqa: E731
    # latitude / longitude lines from each pixel's ground position
    xx, yy = np.meshgrid(x, y)
    grid_lat, grid_lon = ray_latlon(xx, yy, 0.0, p)
    for values in (grid_lat, grid_lon):
        step = _nice_step(np.nanmax(values) - np.nanmin(values))
        levels = np.arange(np.floor(np.nanmin(values) / step) * step, np.nanmax(values) + step, step)
        lines = ax_img.contour(values, levels=levels, colors="tab:orange", linewidths=0.8, alpha=0.8)
        ax_img.clabel(lines, fmt=lambda v: f"{v:g}°", fontsize=9)
    # the pixels the column uses, from the ground (red +) up to 18 km
    vx_col, vy_col = scan_angles(np.full(len(LEVELS_KM), lat), np.full(len(LEVELS_KM), lon), LEVELS_KM * 1000.0, p)
    ax_img.plot(*to_px(vx_col, vy_col), color="tab:red", lw=1.5, label="Column pixels, 0-20 km")
    ax_img.plot(*to_px(vx, vy), "+", color="tab:red", ms=16, mew=3)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.legend(loc="lower left", fontsize=10)
    ax_img.set_title("GOES-19 band 13 (orange: latitude / longitude)")
    fig.colorbar(img, ax=ax_img, shrink=0.8, label="Band 13 (K)")

    z = column["height_km"]
    for s, color in zip(SPECIES, ["k", "tab:blue", "tab:green", "tab:purple"]):
        key = f"prob_{s}"
        ax_p.plot(column[key], z, color=color, label=s.capitalize())
    ax_p.axvline(0.5, color="0.6", ls=":", lw=1)
    ax_p.set_xlim(0, 1)
    ax_p.set_xlabel("Probability")
    ax_p.set_ylabel("Height (km)")
    ax_p.set_title("Hydrometeor probability")

    ax_p.legend(loc="upper right")

    for w, color in zip(PHASES, ["tab:blue", "tab:green", "tab:purple"]):
        key = f"{w}_water_g_m3"
        ax_w.plot(np.maximum(column[key], 1e-5), z, color=color, label=w.capitalize())
    ax_w.set_xscale("log")
    ax_w.set_xlim(1e-5, 2)
    ax_w.set_xlabel("Water content (g m$^{-3}$)")
    ax_w.set_title("Water content")
    ax_w.legend(loc="upper right")
    for ax in (ax_p, ax_w):
        ax.set_ylim(0, 18)
        ax.grid(alpha=0.3, lw=0.5)

    time = scan_time(scene)
    fig.suptitle(f"C3DIR {version} above {name} ({lat:.2f}, {lon:.2f}), GOES-19 {time:%Y-%m-%d %H:%M} UTC")
    return fig


def _nice_step(span):
    return next(step for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 5) if span / step <= 5)


@styled
def plot_region(region, name):
    """the predicted cloud field around a point. top: goes true color and band 13, and C3DIR
    cloud-top height, all on the same lat / lon grid. bottom: C3DIR 3D view of cloudy voxels, and
    the west-east cross section through the point."""
    lat0, lon0 = region.attrs["center_lat"], region.attrs["center_lon"]
    lats, lons, z = region["lat"].values, region["lon"].values, region["height_km"].values
    cloudy = (region["prob_any"] >= 0.5).values
    total = sum(region[f"{w}_water_g_m3"] for w in PHASES).values
    total = np.where(cloudy | np.isnan(total), total, 0.0)  # water only where cloud is likely, as in the 3D view
    top = np.where(cloudy.any(0), z[cloudy.shape[0] - 1 - np.argmax(cloudy[::-1], 0)], np.nan)
    sky = SKY_COLOR

    fig = plt.figure(figsize=(22, 13), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    ax_rgb, ax_ir, ax_top = (fig.add_subplot(grid[0, k]) for k in range(3))
    bottom = grid[1, :].subgridspec(1, 2, width_ratios=[1, 1.3])
    ax3 = fig.add_subplot(bottom[0], projection="3d")
    ax_we = fig.add_subplot(bottom[1])

    # goes, on the same grid as the prediction
    ax_rgb.imshow(true_color(region), origin="lower", extent=(lons[0], lons[-1], lats[0], lats[-1]))
    ax_rgb.set_title("GOES-19 true color")
    ir = ax_ir.pcolormesh(lons, lats, region["CMI_C13"], cmap="Greys", shading="nearest")
    ax_ir.set_title("GOES-19 band 13 (10.3 µm IR)")
    fig.colorbar(ir, ax=ax_ir, label="Band 13 (K)", shrink=0.8)
    for ax in (ax_rgb, ax_ir, ax_top):
        ax.plot(lon0, lat0, "+", color="tab:red", ms=18, mew=3)
        ax.set_aspect(1 / np.cos(np.radians(lat0)))
        ax.set_xlabel("Longitude (°)")
    ax_rgb.set_ylabel("Latitude (°)")

    # 3D: every cloudy voxel, gray (thin) to white (thick), on a blue sky
    h, i, j = np.nonzero(cloudy)
    keep = np.random.default_rng(0).permutation(len(h))[:60000]
    h, i, j = h[keep], i[keep], j[keep]
    norm = LogNorm(vmin=1e-5, vmax=1.0)
    ax3.scatter(lons[j], lats[i], z[h], c=np.clip(total[h, i, j], 1e-5, 1), cmap=CLOUD_CMAP, norm=norm,
                s=4, marker="s", alpha=0.5, linewidths=0, depthshade=False)
    ax3.plot([lon0, lon0], [lat0, lat0], [0, 18], color="tab:red", lw=2)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor(sky)
        pane.set_alpha(1)
    ax3.set_xlim(lons[0], lons[-1])  # fixed, so a cloud-free region still shows the right area
    ax3.set_ylim(lats[0], lats[-1])
    ax3.set_zlim(0, 18)
    ax3.set_xlabel("Longitude (°)", labelpad=12)
    ax3.set_ylabel("Latitude (°)", labelpad=12)
    ax3.set_zlabel("Height (km)", labelpad=4)
    ax3.tick_params(labelsize=10)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.view_init(elev=22, azim=-60)
    ax3.set_title("C3DIR cloudy voxels (P ≥ 0.5)")

    # top down: cloud-top height
    mesh = ax_top.pcolormesh(lons, lats, top, cmap="viridis", vmin=0, vmax=16, shading="nearest")
    ax_top.set_facecolor(sky)
    ax_top.axhline(lat0, color="tab:red", lw=1, ls="--")
    ax_top.set_title("C3DIR cloud-top height (blue = clear)")
    fig.colorbar(mesh, ax=ax_top, label="Cloud top (km)", shrink=0.8)

    # side view along the dashed line through the point
    values = total[:, np.argmin(abs(lats - lat0)), :]
    shown = np.where(values > 1e-5, values, np.where(np.isnan(values), np.nan, 1e-6))
    side = ax_we.pcolormesh(lons, z, shown, cmap=CLOUD_CMAP, norm=LogNorm(vmin=1e-5, vmax=1.0, clip=False),
                            shading="nearest")
    ax_we.axvline(lon0, color="tab:red", lw=1, ls="--")
    ax_we.set_facecolor("0.25")
    ax_we.set_ylim(0, 18)
    ax_we.set_xlabel("Longitude (°)")
    ax_we.set_ylabel("Height (km)")
    ax_we.set_title("C3DIR west-east section (along dashed line)")
    fig.colorbar(side, ax=ax_we, label="Water content (g m$^{-3}$)", shrink=0.9)
    fig.suptitle(f"C3DIR {region.attrs['model']} around {name}, GOES-19 "
                 f"{pd.Timestamp(region.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")
    return fig
