"""figures for a trained C3DIR model: training history, test scores, EarthCARE vs C3DIR along
the track (curtain, species, profiles), and predictions around a point (column, region).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from heimdall.c3dir.constants import LEVELS_KM, MASK_THRESHOLD, PHASES, SPECIES, WATER
from heimdall.c3dir.model.evaluation import _water
from heimdall.c3dir.geometry import ray_latlon, scan_angles
from heimdall.c3dir.model.inference import crop, scan_time
from heimdall.plot_style import CLOUD_CMAP, PROBABILITY_CMAP, RAY_COLOR, SKY_COLOR, styled, true_color


@styled
def plot_history(history):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(history["step"], history["train_loss"], label="Train")
    ax.plot(history["step"], history["val_loss"], label="Validation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    ax.grid(alpha=0.3, lw=0.5)
    return fig

@styled
def plot_detection(scores_by_level):
    """detection scores against height for each species (paper figure 6a-d)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True, layout="constrained")
    for ax, name in zip(axes, SPECIES):
        s = scores_by_level[scores_by_level.species == name]
        for metric in ["BACC", "F1", "TPR", "TNR"]:
            ax.plot(s[metric], s["level_km"], label=metric)
        ax.set_title({"any": "Any hydrometeor"}.get(name, name.capitalize()))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Height (km)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Voxel detection")
    return fig

@styled
def plot_water(table, scores):
    """predicted vs EarthCARE water content (paper figure 7)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    bins = np.logspace(-5, 0.5, 60)
    for ax, (_, row) in zip(axes, scores.iterrows()):
        truth, pred = _water(table, row["water"])
        ok = (truth > MASK_THRESHOLD) & (pred > MASK_THRESHOLD)
        ax.hist2d(truth[ok], pred[ok], bins=[bins, bins], cmap="Blues", norm=LogNorm(), cmin=1)
        ax.plot(bins[[0, -1]], bins[[0, -1]], color="k", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(row["water"].upper())
        ax.set_xlabel("EarthCARE (g m$^{-3}$)")
        ax.text(0.04, 0.96, f"r = {row['pearson_r']:.2f}\nMALE = {row['MALE']:.2f}", transform=ax.transAxes,
                va="top", fontsize=12)
    axes[0].set_ylabel("C3DIR (g m$^{-3}$)")
    fig.suptitle("Water content")
    return fig

@styled
def plot_curtain(pair, curtain):
    """EarthCARE vs C3DIR along the track (paper section 5.3): water content and hydrometeor
    probability. curtain: output of predict_curtain."""
    probability, water = curtain
    truth = sum(pair[w].values for w in WATER)
    panels = [("EarthCARE", truth, True), ("C3DIR", water.sum(-1), True),
              ("C3DIR hydrometeor probability", probability[..., 0], False)]
    lat = pair["lat"].values
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, (title, values, is_water) in zip(axes, panels):
        if is_water:
            shown = np.where(values > MASK_THRESHOLD, values, np.where(np.isnan(values), np.nan, MASK_THRESHOLD / 10))
            mesh = ax.pcolormesh(lat, LEVELS_KM, shown.T, cmap=CLOUD_CMAP, shading="nearest",
                                 norm=LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False))
            fig.colorbar(mesh, ax=ax, label="Water content (g m$^{-3}$)")
        else:
            mesh = ax.pcolormesh(lat, LEVELS_KM, values.T, cmap="Greys_r", vmin=0, vmax=1, shading="nearest")
            fig.colorbar(mesh, ax=ax, label="Probability")
        ax.set_facecolor("0.25")
        ax.set_ylim(0, 20)
        ax.set_ylabel("Height (km)")
        ax.set_title(title)
    axes[0].set_xlim(lat[0], lat[-1])
    axes[-1].set_xlabel("Latitude (°)")
    fig.suptitle(f"EarthCARE {pair.attrs['orbit_and_frame']} × GOES-19, "
                 f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")
    return fig

@styled
def plot_profiles(pair, curtain, n=6):
    """EarthCARE vs C3DIR vertical profiles at n cloudy points spread along the track:
    hydrometeor mask vs probability, then ice, liquid and rain water content."""
    probability, water = curtain
    cloudy = (pair["hydrometeor_mask"].values == 1).sum(1)
    candidates = np.flatnonzero(cloudy >= max(3, np.quantile(cloudy[cloudy > 0], 0.75)))
    picks = candidates[np.linspace(0, len(candidates) - 1, min(n, len(candidates))).astype(int)]
    fig, axes = plt.subplots(4, len(picks), figsize=(3 * len(picks), 13), sharey=True,
                             layout="constrained", squeeze=False)
    floor = MASK_THRESHOLD
    for j, p in enumerate(picks):
        ax = axes[0, j]
        ax.fill_betweenx(LEVELS_KM, 0, pair["hydrometeor_mask"].values[p] == 1, step="mid", color="0.8",
                         label="EarthCARE")
        ax.plot(probability[p, :, 0], LEVELS_KM, color=RAY_COLOR, label="C3DIR")
        ax.set_xlim(-0.05, 1.05)
        lat, lon = pair["lat"].values[p], pair["lon"].values[p]
        ax.set_title(f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'W' if lon < 0 else 'E'}",
                     fontsize=11)
        for row, (k, name) in enumerate(zip(range(3), ["Ice", "Liquid", "Rain"]), start=1):
            ax = axes[row, j]
            ax.plot(np.maximum(pair[WATER[k]].values[p], floor), LEVELS_KM, "k", label="EarthCARE")
            ax.plot(np.maximum(water[p, :, k], floor), LEVELS_KM, color=RAY_COLOR, label="C3DIR")
            ax.set_xscale("log")
            ax.set_xlim(floor * 0.8, 2)
            ax.set_xlabel(f"{name} (g m$^{{-3}}$)", fontsize=11)
    axes[0, 0].set_xlabel("Hydrometeor probability", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Height (km)")
    axes[0, 0].set_ylim(0, 18)
    axes[1, 0].legend(loc="upper right", fontsize=10)
    fig.suptitle(f"EarthCARE {pair.attrs['orbit_and_frame']} × GOES-19, "
                 f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")
    return fig

@styled
def plot_species_curtain(pair, curtain):
    """EarthCARE (left) vs C3DIR (right) along the track: hydrometeor mask vs probability, and
    ice / liquid water content where C3DIR's ice / liquid probability is at least 0.5."""
    probability, water = curtain
    rows = [("Hydrometeor", pair["hydrometeor_mask"].values, probability[..., 0], False),
            ("IWC", pair["iwc"].values, np.where(probability[..., 1] >= 0.5, water[..., 0], 0), True),
            ("LWC", pair["lwc"].values, np.where(probability[..., 2] >= 0.5, water[..., 1], 0), True)]
    lat = pair["lat"].values
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True, layout="constrained")
    for (name, truth, pred, is_water), row in zip(rows, axes):
        for ax, values, who in [(row[0], truth, "EarthCARE"), (row[1], pred, "C3DIR")]:
            if is_water:
                shown = np.where(values > MASK_THRESHOLD, values, np.where(np.isnan(values), np.nan, MASK_THRESHOLD / 10))
                mesh = ax.pcolormesh(lat, LEVELS_KM, shown.T, cmap=CLOUD_CMAP, shading="nearest",
                                     norm=LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False))
                title = f"{who} {name}" if who == "EarthCARE" else f"C3DIR {name} where P({'ice' if name == 'IWC' else 'liquid'}) ≥ 0.5"
            else:
                mesh = ax.pcolormesh(lat, LEVELS_KM, values.T, cmap=PROBABILITY_CMAP, vmin=0, vmax=1, shading="nearest")
                title = "EarthCARE hydrometeor mask" if who == "EarthCARE" else "C3DIR P(hydrometeor)"
            ax.set_facecolor("0.25")
            ax.set_title(title)
        fig.colorbar(mesh, ax=row, label="g m$^{-3}$" if is_water else "Probability", shrink=0.9)
        row[0].set_ylabel("Height (km)")
    axes[0, 0].set_ylim(0, 18)
    axes[0, 0].set_xlim(lat[0], lat[-1])
    for ax in axes[-1]:
        ax.set_xlabel("Latitude (°)")
    fig.suptitle(f"EarthCARE {pair.attrs['orbit_and_frame']} × GOES-19, "
                 f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")
    return fig


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
        if f"{key}_p10" in column:
            ax_p.fill_betweenx(z, column[f"{key}_p10"], column[f"{key}_p90"], color=color, alpha=0.15, lw=0)
    ax_p.axvline(0.5, color="0.6", ls=":", lw=1)
    ax_p.set_xlim(0, 1)
    ax_p.set_xlabel("Probability")
    ax_p.set_ylabel("Height (km)")
    ax_p.set_title("Hydrometeor probability")

    ax_p.legend(loc="upper right", title="median, band 10-90%", title_fontsize=10)

    for w, color in zip(PHASES, ["tab:blue", "tab:green", "tab:purple"]):
        key = f"{w}_water_g_m3"
        ax_w.plot(np.maximum(column[key], 1e-5), z, color=color, label=w.capitalize())
        if f"{key}_p10" in column:
            ax_w.fill_betweenx(z, np.maximum(column[f"{key}_p10"], 1e-5), np.maximum(column[f"{key}_p90"], 1e-5),
                               color=color, alpha=0.15, lw=0)
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
