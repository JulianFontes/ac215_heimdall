"""figures for a trained C3DIR model: training history, test scores, where the matches are, and
EarthCARE vs C3DIR along the track (curtain, species, profiles).
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from c3dir.constants import GOES19_LON, LEVELS_KM, MASK_THRESHOLD, SPECIES, VIEW_RADIUS_DEG, WATER
from c3dir.geometry import view_circle
from c3dir.evaluation import _water
from utilities.plot_style import CLOUD_CMAP, PROBABILITY_CMAP, RAY_COLOR, styled


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


FULL_DISK_COLOR, SCRAPING_COLOR = "#8fb8de", "#eaf3fb"  # pastel blues for the GOES-19 view areas


def _coverage(splits):
    """matched profile positions and dates for each split of collocated pair files."""
    out = {}
    for name, pairs in splits.items():
        lat, lon, days = [], [], set()
        for path in pairs:
            with xr.open_dataset(path) as pair:
                lat.append(pair["lat"].values)  # one entry per matched profile
                lon.append(pair["lon"].values)
                days.add(pd.Timestamp(pair.attrs["scan_start"]).date())
        out[name] = (np.concatenate(lat), np.concatenate(lon), sorted(days))
    return out


def _density(lat, lon, lat_edges, lon_edges, smooth_deg=1.0):
    """matched profiles per square degree on a fine grid, smoothed with a gaussian of smooth_deg so
    the heatmap has no visible boxes; below 1 per square degree is masked to show the earth."""
    step = lat_edges[1] - lat_edges[0]
    counts, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    density = gaussian_filter(counts, smooth_deg / step) / step ** 2
    return np.ma.masked_less(density, 1.0)


def _view_areas(ax):
    """the GOES-19 full-disk footprint (where lines of sight graze the earth, arccos(R / satellite
    distance) ~81° from the point under it) and the scraping view limit, as pastel fills."""
    footprint_deg = np.degrees(np.arccos(6378137.0 / (6378137.0 + 35786023.0)))
    for radius, color, alpha, label in [(footprint_deg, FULL_DISK_COLOR, 0.35, "GOES-19 full disk"),
                                        (VIEW_RADIUS_DEG, SCRAPING_COLOR, 0.6,
                                         f"Scraping limit ({VIEW_RADIUS_DEG:.0f}°)")]:
        ring_lat, ring_lon = view_circle(GOES19_LON, radius, n=360)
        ax.fill(ring_lon, ring_lat, facecolor=color, edgecolor=FULL_DISK_COLOR, alpha=alpha, lw=1,
                transform=ccrs.Geodetic(), label=label)


def _earth(ax):
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.9")  # 110m isn't cached, so ask for 50m
    ax.coastlines("50m", color="0.4", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="0.4", linewidth=0.3)


@styled
def plot_coverage(splits, grid_deg=0.25):
    """where EarthCARE was matched to GOES-19, one global map per split (like paper figure 2a,d,e):
    a heatmap of matched EarthCARE profiles over the GOES-19 full-disk footprint and the scraping
    view limit.

    splits: {name: collocated pair files (data/processed/c3dir/collocated/)}, e.g. {"Train": [...], "Test": [...]}.
    """
    lat_edges, lon_edges = np.arange(-90, 90 + grid_deg, grid_deg), np.arange(-180, 180 + grid_deg, grid_deg)
    fig, axes = plt.subplots(len(splits), 1, figsize=(8, 4.2 * len(splits)), layout="constrained",
                             subplot_kw={"projection": ccrs.Robinson(central_longitude=GOES19_LON)}, squeeze=False)
    for ax, (name, (lat, lon, days)) in zip(axes[:, 0], _coverage(splits).items()):
        ax.set_global()
        _earth(ax)
        _view_areas(ax)
        mesh = ax.pcolormesh(lon_edges, lat_edges, _density(lat, lon, lat_edges, lon_edges), cmap="viridis",
                             transform=ccrs.PlateCarree(), rasterized=True)
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="Matched profiles\nper square degree")
        ax.set_title(f"{name}: {len(days)} days, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")
    axes[-1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    return fig


@styled
def plot_coverage_disk(splits, grid_deg=0.25):
    """a heatmap of every matched EarthCARE profile (all splits together) on the hemisphere GOES-19
    sees, over the full disk and the scraping view limit."""
    lat, lon, days = (np.concatenate(v) for v in zip(*[(la, lo, d) for la, lo, d in _coverage(splits).values()]))
    # grid only around the data, so every cell corner stays on the visible disk
    lat_edges = np.arange(np.floor(lat.min()) - 5, lat.max() + 5, grid_deg)
    lon_edges = np.arange(np.floor(lon.min()) - 5, lon.max() + 5, grid_deg)
    fig, ax = plt.subplots(figsize=(8, 8.5), layout="constrained",
                           subplot_kw={"projection": ccrs.Geostationary(central_longitude=GOES19_LON)})
    ax.set_global()
    _earth(ax)
    _view_areas(ax)
    mesh = ax.pcolormesh(lon_edges, lat_edges, _density(lat, lon, lat_edges, lon_edges), cmap="viridis",
                         transform=ccrs.PlateCarree(), rasterized=True)
    fig.colorbar(mesh, ax=ax, shrink=0.6, label="Matched profiles per square degree")
    ax.set_title(f"EarthCARE profiles matched to GOES-19 ({len(set(days))} days)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
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
