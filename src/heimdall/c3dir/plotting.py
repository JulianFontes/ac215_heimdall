"""figures for the collocated goes-19 / earthcare pairs (see heimdall.c3dir.goes19)."""
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from heimdall.c3dir.earthcare_labels import MASK_THRESHOLD
from heimdall.c3dir.goes19 import goes_proj


class Plotting:
    """
    out_dir: folder to save figures in; None returns the figures without saving.
    """

    def __init__(self, out_dir=None):
        self.out_dir = Path(out_dir) if out_dir else None

    @staticmethod
    def true_color(ds):
        """rough true color from goes bands 1 (blue), 2 (red) and 3 (veggie)."""
        r, b, veg = (np.clip(ds[f"CMI_C0{i}"].values, 0, 1) for i in (2, 1, 3))
        g = 0.45 * r + 0.10 * veg + 0.45 * b  # abi has no green band, so one is synthesized
        return np.nan_to_num(np.dstack([r, g, b])) ** (1 / 2.2)  # gamma brightening

    @staticmethod
    def geos_crs(ds):
        # cartopy version of the goes projection so imshow lines up with the coastlines
        p = ds["goes_imager_projection"].attrs
        globe = ccrs.Globe(semimajor_axis=p["semi_major_axis"], semiminor_axis=p["semi_minor_axis"])
        return ccrs.Geostationary(central_longitude=p["longitude_of_projection_origin"],
                                  satellite_height=p["perspective_point_height"],
                                  sweep_axis=p["sweep_angle_axis"], globe=globe)

    @staticmethod
    def cloud_top_km(pair):
        """highest 0.25 km level with something in the hydrometeor mask, nan if the profile is clear."""
        cloudy = pair["hydrometeor_mask"].values == 1
        # flip the levels so argmax finds the highest cloudy one
        top = pair["level"].values[cloudy.shape[1] - 1 - np.argmax(cloudy[:, ::-1], axis=1)]
        return np.where(cloudy.any(axis=1), top, np.nan)

    def track_on_goes(self, pair, frame=None):
        """true color and band 13 (10.3 µm IR) panels with the earthcare track drawn on top.

        pair: dataset from GOES19.collocate(). frame: optional labels dataset for
        the whole frame, drawn as a dashed line for context. matched profiles
        are colored by ACM-CAP cloud-top height; clear ones show in red.
        """
        # x/y are scan angles, multiply by the satellite height to get meters for cartopy
        crs = self.geos_crs(pair)
        h = pair["goes_imager_projection"].attrs["perspective_point_height"]
        x, y = pair.x.values * h, pair.y.values * h
        dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
        extent = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)
        top = self.cloud_top_km(pair)

        # left panel true color, right panel the 10.3 µm infrared band
        panel_w = float(np.clip(7 * len(x) / len(y), 3, 8))  # follow the crop's aspect ratio
        fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 2.5, 8), subplot_kw={"projection": crs},
                                 layout="constrained")
        axes[0].imshow(self.true_color(pair), extent=extent, origin="upper", transform=crs)
        axes[0].set_title("True color (ABI bands 1–3)", fontsize=10)
        bt = axes[1].imshow(pair["CMI_C13"].values, extent=extent, origin="upper",
                            transform=crs, cmap="Greys")
        axes[1].set_title("ABI band 13 (10.3 µm IR)", fontsize=10)
        fig.colorbar(bt, ax=axes[1], label="Brightness temperature (K)", shrink=0.6)

        # shared map features and track overlay on both panels
        for ax in axes:
            ax.set_extent(extent, crs=crs)
            ax.coastlines(color="goldenrod", linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, edgecolor="goldenrod", linewidth=0.4)
            grid = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            grid.top_labels = grid.right_labels = False
            grid.xlocator = MaxNLocator(4)
            if frame is not None:
                ax.plot(frame["lon"], frame["lat"], color="white", lw=1, ls="--",
                        transform=ccrs.Geodetic(), label="EarthCARE frame")
            # every matched profile in red, cloudy ones colored by cloud-top height on top
            ax.scatter(pair["lon"], pair["lat"], s=2, color="tab:red", transform=ccrs.PlateCarree(),
                       label="Profiles within 2.5 min of their pixel")
            points = ax.scatter(pair["lon"], pair["lat"], c=top, s=8, cmap="viridis",
                                vmin=0, vmax=15, transform=ccrs.PlateCarree(), zorder=3)
        axes[0].legend(loc="lower left", fontsize=8)
        fig.colorbar(points, ax=axes, location="bottom", shrink=0.4,
                     label="ACM-CAP cloud-top height (km)")
        fig.suptitle(f"GOES-19 {pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M:%S} UTC scan + EarthCARE "
                     f"{pair.attrs['orbit_and_frame']} ({pair.sizes['profile']} profiles)")

        self._save(fig, pair)
        return fig

    @staticmethod
    def track_on_grid(pair):
        """where each profile lands on the goes crop, as fractional (row, col).

        rounding gives the goes pixel the profile falls in (nearest pixel to the
        ground lat/lon, no parallax correction, same as GOES19.collocate()).
        """
        h = pair["goes_imager_projection"].attrs["perspective_point_height"]
        px, py = goes_proj(pair)(pair["lon"].values, pair["lat"].values)
        x, y = pair.x.values, pair.y.values
        # pixel centers are evenly spaced, so position / spacing is the index
        return (np.asarray(py) / h - y[0]) / (y[1] - y[0]), (np.asarray(px) / h - x[0]) / (x[1] - x[0])

    def goes_along_track(self, pair):
        """the goes pixel each matched profile falls in, as a dataset over profile."""
        row, col = (xr.DataArray(np.rint(i).astype(int), dims="profile") for i in self.track_on_grid(pair))
        return pair[[v for v in pair.data_vars if v.startswith("CMI_")]].isel(y=row, x=col)

    def voxels(self, pair, n_columns=40):
        """zoom on a stretch of track to show the collocated voxels (goes pixel x 0.25 km level).

        left: band 13 (10.3 µm IR) at native goes pixels with the earthcare
        profiles; matched pixels are outlined and numbered. right: those pixels
        as voxel columns, each the mean of the profiles inside it, with band 13
        along the top.

        pair: dataset from GOES19.collocate(). n_columns: number of goes pixels
        to show; the stretch with the most cloudy voxels is chosen.
        """
        row_f, col_f = self.track_on_grid(pair)
        row, col = np.rint(row_f).astype(int), np.rint(col_f).astype(int)
        # consecutive profiles in the same goes pixel make one voxel column
        new = np.r_[True, (row[1:] != row[:-1]) | (col[1:] != col[:-1])]
        column = np.cumsum(new) - 1
        n = column[-1] + 1
        water = pair["iwc"].values + pair["lwc"].values + pair["rwc"].values  # total condensate [profile, level]
        total = np.full((n, water.shape[1]), np.nan)
        for c in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # levels with no valid retrieval stay nan
                total[c] = np.nanmean(water[column == c], axis=0)
        first = np.flatnonzero(new)  # first profile in each column, gives its pixel
        c_row, c_col = row[first], col[first]
        c13 = pair["CMI_C13"].values[c_row, c_col]

        # the window of n_columns with the most cloudy voxels
        n_columns = min(n_columns, n)
        cloudy = np.convolve((total > MASK_THRESHOLD).sum(1), np.ones(n_columns), "valid")
        start = int(np.argmax(cloudy))
        shown = slice(start, start + n_columns)
        ids = np.arange(start, start + n_columns)

        fig, (ax_map, ax_curt) = plt.subplots(1, 2, figsize=(16, 7.5), layout="constrained",
                                              gridspec_kw={"width_ratios": [1, 1.6]})

        # left: goes pixels around the window, in row/col units so each square is one pixel
        pad = 4
        r0, r1 = c_row[shown].min() - pad, c_row[shown].max() + pad + 1
        q0, q1 = c_col[shown].min() - pad, c_col[shown].max() + pad + 1
        r0, q0 = max(r0, 0), max(q0, 0)
        bt = pair["CMI_C13"].values[r0:r1, q0:q1]
        vmin, vmax = np.nanpercentile(bt, [2, 98])
        img = ax_map.imshow(bt, cmap="Greys", vmin=vmin, vmax=vmax, origin="upper",
                            extent=(q0 - 0.5, q0 + bt.shape[1] - 0.5, r0 + bt.shape[0] - 0.5, r0 - 0.5))
        # pixel edges show each footprint
        ax_map.set_xticks(np.arange(q0, q0 + bt.shape[1] + 1) - 0.5, minor=True)
        ax_map.set_yticks(np.arange(r0, r0 + bt.shape[0] + 1) - 0.5, minor=True)
        ax_map.grid(which="minor", color="0.5", lw=0.3)
        ax_map.tick_params(which="minor", length=0)
        # outline the matched pixels; number every 5th to match the right panel
        for i, (r, q) in zip(ids, zip(c_row[shown], c_col[shown])):
            ax_map.add_patch(plt.Rectangle((q - 0.5, r - 0.5), 1, 1, fill=False, ec="tab:orange", lw=1.2))
            if i % 5 == 0:
                ax_map.annotate(str(i), (q, r), xytext=(8, 0), textcoords="offset points",
                                color="tab:orange", fontsize=8, fontweight="bold", va="center")
        in_window = (column >= start) & (column < start + n_columns)
        ax_map.plot(col_f[in_window], row_f[in_window], ".", ms=2.5, color="tab:red",
                    label="EarthCARE profiles")
        ax_map.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, ec="tab:orange", label="Matched GOES pixels"))
        ax_map.set_xlim(q0 - 0.5, q0 + bt.shape[1] - 0.5)
        ax_map.set_ylim(r0 + bt.shape[0] - 0.5, r0 - 0.5)
        ax_map.set_xlabel("GOES column in crop")
        ax_map.set_ylabel("GOES row in crop")
        ax_map.set_title("GOES-19 ABI band 13 (10.3 µm IR), zoomed", fontsize=10)
        ax_map.legend(loc="lower left", fontsize=8)
        fig.colorbar(img, ax=ax_map, location="bottom", shrink=0.7, label="Band 13 brightness temperature (K)")

        # right: one column per collocated goes pixel, one row per 0.25 km level, voxel edges drawn
        z = pair["level"].values
        edges_x = np.arange(start, start + n_columns + 1) - 0.5
        edges_z = np.r_[z - 0.125, z[-1] + 0.125]
        v = total[shown].T
        v = np.where(v > MASK_THRESHOLD, v, MASK_THRESHOLD / 10)  # clear -> "under" -> white
        v[np.isnan(total[shown].T)] = np.nan  # no retrieval -> gray
        cmap = plt.get_cmap("Blues").with_extremes(under="white", bad="0.8")
        mesh = ax_curt.pcolormesh(edges_x, edges_z, v, cmap=cmap, edgecolors="0.85", linewidth=0.2,
                                  norm=LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False))
        top = np.nanmax(np.where(total[shown] > MASK_THRESHOLD, z, np.nan), initial=2.0)
        ax_curt.set_ylim(0, min(20, top + 2))
        ax_curt.set_xlabel("Matched GOES pixel #")
        ax_curt.set_ylabel("Height (km)")
        ax_curt.set_xticks(ids[ids % 5 == 0])
        fig.colorbar(mesh, ax=ax_curt, location="right", shrink=0.8,
                     label="Ice + liquid + rain water content (g m$^{-3}$)")
        # what goes saw in each of those pixels, same gray scale as the map
        strip = ax_curt.inset_axes([0, 1.02, 1, 0.06], sharex=ax_curt)
        strip.pcolormesh(edges_x, [0, 1], c13[shown][None], cmap="Greys", vmin=vmin, vmax=vmax,
                         edgecolors="0.5", linewidth=0.3)
        strip.set_yticks([0.5], ["Band 13"])
        strip.tick_params(labelbottom=False, bottom=False)
        ax_curt.set_title(f"{n_columns} pixels × 80 levels ({in_window.sum()} profiles averaged)",
                          fontsize=10, pad=34)

        fig.suptitle(f"EarthCARE {pair.attrs['orbit_and_frame']} × GOES-19, "
                     f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")
        self._save(fig, pair, suffix="_voxels")
        return fig

    def curtain(self, pair):
        """along-track view of a collocated pair: what goes saw above each
        profile (true color strip and band 13) over the ACM-CAP ice, liquid and
        rain water content curtains, sharing latitude on the x axis.

        pair: dataset from GOES19.collocate(). white = clear (below the 1e-5
        g m^-3 mask threshold), gray = no valid retrieval.
        """
        lat, z = pair["lat"].values, pair["level"].values
        goes = self.goes_along_track(pair)
        water = {"Ice": "iwc", "Liquid": "lwc", "Rain": "rwc"}

        fig, axes = plt.subplots(2 + len(water), 1, figsize=(11, 10), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [0.35, 1] + [1.6] * len(water)})
        # x runs in flight order, so the overpass starts on the left
        axes[-1].set_xlim(lat[0], lat[-1])

        # what goes saw directly over each profile
        axes[0].imshow(self.true_color(goes), extent=(lat[0], lat[-1], 0, 1), aspect="auto")
        axes[0].set_yticks([])
        axes[0].set_ylabel("True\ncolor", rotation=0, ha="right", va="center", fontsize=9)
        axes[1].plot(lat, goes["CMI_C13"].values, color="0.2", lw=1.2)
        axes[1].invert_yaxis()  # colder (higher) cloud tops point up, like the curtains
        axes[1].set_ylabel("Band 13 IR (K)\ncold ↑", fontsize=9)
        axes[1].grid(alpha=0.3, lw=0.5)

        # earthcare curtains, one shared log scale so the panels compare directly
        cmap = plt.get_cmap("Blues").with_extremes(under="white", bad=(0, 0, 0, 0))
        norm = LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False)
        for ax, (name, var) in zip(axes[2:], water.items()):
            w = pair[var].values.T  # [level, profile]
            w = np.where(w > MASK_THRESHOLD, w, MASK_THRESHOLD / 10)  # clear -> "under" -> white
            w[np.isnan(pair[var].values.T)] = np.nan  # no retrieval -> transparent -> gray facecolor
            mesh = ax.pcolormesh(lat, z, w, cmap=cmap, norm=norm, shading="nearest")
            ax.set_facecolor("0.8")
            ax.set_ylim(0, 20)
            ax.set_ylabel(f"{name}\nheight (km)", fontsize=9)
        fig.colorbar(mesh, ax=axes[2:], shrink=0.6, label="Water content (g m$^{-3}$)")
        axes[-1].set_xlabel("Latitude (°)")

        # longitude on the top axis, to match locations on the track_on_goes map;
        # the track is nearly straight in lat/lon within one crop, so a linear fit suffices
        slope, offset = np.polyfit(lat, pair["lon"].values, 1)
        to_lon = lambda v: slope * np.asarray(v) + offset
        to_lat = lambda v: (np.asarray(v) - offset) / slope
        top = axes[0].secondary_xaxis("top", functions=(to_lon, to_lat))
        top.xaxis.set_major_locator(MaxNLocator(6))
        top.set_xlabel("Longitude (°)", fontsize=9)

        fig.suptitle(f"EarthCARE {pair.attrs['orbit_and_frame']} under GOES-19 "
                     f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M:%S} UTC scan "
                     f"({pair.sizes['profile']} profiles, |Δt| ≤ {pair.attrs['max_seconds']:.0f} s)")
        self._save(fig, pair, suffix="_curtain")
        return fig

    def _save(self, fig, pair, suffix=""):
        """save the figure if out_dir is set."""
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            name = Path(pair.attrs["scan_file"]).stem
            fig.savefig(self.out_dir / f"{pair.attrs['orbit_and_frame']}_{name}{suffix}.png",
                        dpi=150, bbox_inches="tight")
