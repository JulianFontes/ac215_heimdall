"""figures for the collocated goes-19 / earthcare pairs (see c3dir.collocation)."""
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from c3dir.constants import BANDS, MASK_THRESHOLD, TARGETS
from c3dir.geometry import ray_point, satellite_frame, scan_angles
from utilities.plot_style import CLOUD_CMAP, NO_DATA_COLOR, RAY_COLOR, styled, true_color




class Plotting:
    """
    out_dir: folder to save figures in; None returns the figures without saving.
    """

    def __init__(self, out_dir=None):
        self.out_dir = Path(out_dir) if out_dir else None

    true_color = staticmethod(true_color)

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

    @styled
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
        axes[0].set_title("True color")
        bt = axes[1].imshow(pair["CMI_C13"].values, extent=extent, origin="upper",
                            transform=crs, cmap="Greys")
        axes[1].set_title("Band 13 (10.3 µm IR)")
        fig.colorbar(bt, ax=axes[1], label="Band 13 (K)", shrink=0.6)

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
                       label="EarthCARE profiles")
            points = ax.scatter(pair["lon"], pair["lat"], c=top, s=8, cmap="viridis",
                                vmin=0, vmax=15, transform=ccrs.PlateCarree(), zorder=3)
        axes[0].legend(loc="lower left")
        fig.colorbar(points, ax=axes, location="bottom", shrink=0.4, label="Cloud-top height (km)")
        fig.suptitle(self._title(pair))

        self._save(fig, pair)
        return fig

    @staticmethod
    def voxel_table(pair):
        """the matched voxels of a pair as a table: level, row, col, profile, distance_km and the 9 targets."""
        table = pd.DataFrame({k: pair[f"voxel_{k}"].values for k in ["level", "row", "col", "profile", "distance_km"]})
        return table.join(pd.DataFrame(pair["voxel_labels"].values, columns=TARGETS))

    def profile_pixels(self, pair):
        """(row, col) of the goes pixel that sees each matched profile: the pixel its highest
        cloudy voxel was matched to, or its lowest matched voxel when the profile is clear."""
        v = self.voxel_table(pair).sort_values("level")
        pixels = v.groupby("profile")[["row", "col"]].first()
        cloud_top = v[v["hydrometeor_mask"] == 1].groupby("profile")[["row", "col"]].last()
        pixels.loc[cloud_top.index] = cloud_top
        return pixels.reindex(range(pair.sizes["profile"]))  # every matched profile has a voxel

    @staticmethod
    def track_on_grid(pair, height_m=0.0):
        """where each profile, at height_m, appears on the goes crop, as fractional (row, col)."""
        vx, vy = scan_angles(pair["lat"].values, pair["lon"].values, height_m,
                             pair["goes_imager_projection"].attrs)
        x, y = pair.x.values, pair.y.values
        # pixel centers are evenly spaced, so position / spacing is the index
        return (vy - y[0]) / (y[1] - y[0]), (vx - x[0]) / (x[1] - x[0])

    @staticmethod
    def along_track_km(lat, lon):
        """distance along the track from the first profile (km), great circle on a 6371 km sphere."""
        lat, lon = np.radians(lat), np.radians(lon)
        a = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
        return np.r_[0, np.cumsum(2 * 6371.0 * np.arcsin(np.sqrt(a)))]

    def goes_along_track(self, pair):
        """bands of the goes pixel that sees each matched profile (see profile_pixels), over profile."""
        pixels = self.profile_pixels(pair)
        return pair[BANDS].isel(y=xr.DataArray(pixels["row"].values, dims="profile"),
                                x=xr.DataArray(pixels["col"].values, dims="profile"))

    @styled
    def voxels(self, pair, n_columns=40, rays=True):
        """zoom on a stretch of track to show the collocated voxels.

        a goes voxel is one pixel's line of sight at one of the 80 C3DIR heights (0.25 km
        apart); each is matched to the earthcare voxel within max_km of it (paper section 3).
        left: band 13 at native goes pixels with the earthcare ground track; matched pixels
        are outlined (and numbered in the grid view).
        right, rays=True: the earthcare curtain with those pixels' lines of sight drawn
        through it where they pass within max_km of the track, and the earthcare voxels
        matched to them. rays=False: one column per pixel and one row per level instead,
        filled from the matched voxels. top: the viewing geometry drawn to scale.

        pair: dataset from GOES19.collocate(). n_columns: number of matched goes pixels
        to show; the stretch with the most cloudy voxels is chosen.
        """
        x, y = pair.x.values, pair.y.values
        max_km = pair.attrs["max_km"]
        v = self.voxel_table(pair)
        v["total"] = v[["iwc", "lwc", "rwc"]].sum(axis=1)  # total condensate
        v["cloudy"] = v["hydrometeor_mask"] == 1
        # matched pixels in the order the track reaches them
        pix = (v.groupby(["row", "col"]).agg(pos=("profile", "mean"), cloudy=("cloudy", "sum"))
               .sort_values("pos").reset_index())
        n_columns = min(n_columns, len(pix))
        start = int(np.argmax(np.convolve(pix["cloudy"], np.ones(n_columns), "valid")))  # cloudiest stretch
        shown = pix.iloc[start:start + n_columns].reset_index(drop=True)
        ids = np.arange(start, start + n_columns)
        mine = [v[(v["row"] == r.row) & (v["col"] == r.col)] for r in shown.itertuples()]  # voxels per pixel
        p0 = min(m["profile"].min() for m in mine)
        p1 = max(m["profile"].max() for m in mine)
        # first / last matched pixel in flight order, by where the pixel sits on the ground: the
        # track profile nearest to its center on the map
        g_row, g_col = self.track_on_grid(pair)
        base = [np.nanargmin(np.hypot(g_row - r.row, g_col - r.col)) for r in shown.itertuples()]
        ends = {"first": int(np.argmin(base)), "last": int(np.argmax(base))}

        axes = plt.figure(figsize=(16, 11.5), layout="constrained").subplot_mosaic(
            [["geometry", "geometry"], ["map", "curtain"]], width_ratios=[1, 1.6], height_ratios=[1.1, 3])
        fig, ax_map, ax_right = axes["map"].figure, axes["map"], axes["curtain"]

        # left: goes pixels around the window, in row/col units so each square is one pixel
        pad = 4
        r0, r1 = max(shown["row"].min() - pad, 0), min(shown["row"].max() + pad + 1, len(y))
        q0, q1 = max(shown["col"].min() - pad, 0), min(shown["col"].max() + pad + 1, len(x))
        bt = pair["CMI_C13"].values[r0:r1, q0:q1]
        vmin, vmax = np.nanpercentile(bt, [2, 98])
        img = ax_map.imshow(bt, cmap="Greys", vmin=vmin, vmax=vmax, origin="upper",
                            extent=(q0 - 0.5, q1 - 0.5, r1 - 0.5, r0 - 0.5))
        ax_map.set_xticks(np.arange(q0, q1 + 1) - 0.5, minor=True)  # pixel edges show each footprint
        ax_map.set_yticks(np.arange(r0, r1 + 1) - 0.5, minor=True)
        ax_map.grid(which="minor", color="0.5", lw=0.3)
        ax_map.tick_params(which="minor", length=0)
        near = slice(max(p0 - 30, 0), p1 + 31)
        row_f, col_f = self.track_on_grid(pair)
        ax_map.plot(col_f[near], row_f[near], "-", color="tab:red", label="EarthCARE track")
        # arrowheads along the track: the direction earthcare flies (time increasing)
        tc, tr = col_f[near], row_f[near]
        visible = np.flatnonzero((tc > q0) & (tc < q1 - 1) & (tr > r0) & (tr < r1 - 1))[:-3]
        for k in visible[np.linspace(0, len(visible) - 1, 5)[1:-1].astype(int)] if len(visible) > 4 else []:
            ax_map.annotate("", (tc[k + 3], tr[k + 3]), (tc[k], tr[k]),
                            arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=0, mutation_scale=22))
        # outline the matched pixels
        for i, r in zip(ids, shown.itertuples()):
            ax_map.add_patch(plt.Rectangle((r.col - 0.5, r.row - 0.5), 1, 1, fill=False, ec=RAY_COLOR, lw=1.2))
            if not rays and i % 5 == 0:  # numbers match the grid view's x axis
                ax_map.annotate(str(i), (r.col, r.row), xytext=(8, 0), textcoords="offset points",
                                color=RAY_COLOR, fontsize=10, va="center")
        ax_map.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, ec=RAY_COLOR, label="Matched pixel"))
        # first and last matched pixel along the track, the same two lines of sight are tagged on the right
        for tag, j in ends.items():
            r = shown.iloc[j]
            ax_map.add_patch(plt.Rectangle((r["col"] - 0.5, r["row"] - 0.5), 1, 1, fill=False, ec="k", lw=2))
            ax_map.annotate(tag, (r["col"] + 0.5, r["row"]), xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=11, bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))
        ax_map.set_xlim(q0 - 0.5, q1 - 0.5)
        ax_map.set_ylim(r1 - 0.5, r0 - 0.5)
        ax_map.set_xlabel("GOES column")
        ax_map.set_ylabel("GOES row")
        ax_map.set_title("GOES-19 band 13")
        ax_map.legend(loc="lower left")
        fig.colorbar(img, ax=ax_map, location="bottom", shrink=0.7, label="Band 13 (K)")

        cmap = plt.get_cmap("Blues").with_extremes(under="white", bad="0.85")
        norm = LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False)
        cloud_top = v.loc[v["profile"].between(p0, p1) & v["cloudy"], "level"].max()
        z_top = min(20, (pair["level"].values[int(cloud_top)] if np.isfinite(cloud_top) else 2) + 3)
        if rays:
            mesh = self._rays_panel(ax_right, pair, shown, mine, p0, p1, max_km, cmap, norm, ends)
            ax_right.set_ylim(0, z_top)
        else:
            mesh = self._grid_panel(ax_right, pair, shown, ids, mine, max_km, cmap, norm, vmin, vmax)
            ax_right.set_ylim(0, z_top)
        fig.colorbar(mesh, ax=ax_right, location="right", shrink=0.8,
                     label="Water content (g m$^{-3}$)")
        self._geometry_panel(axes["geometry"], pair, (p0 + p1) // 2)
        fig.suptitle(self._title(pair))
        self._save(fig, pair, suffix="_voxels")
        return fig

    def _rays_panel(self, ax, pair, shown, mine, p0, p1, max_km, cmap, norm, ends):
        """earthcare curtain with the shown pixels' lines of sight drawn through it."""
        proj = pair["goes_imager_projection"].attrs
        x, y = pair.x.values, pair.y.values
        span = slice(max(p0 - 40, 0), min(p1 + 41, pair.sizes["profile"]))  # profiles around the window
        lat, lon = pair["lat"].values[span], pair["lon"].values[span]
        s = self.along_track_km(lat, lon)
        z = pair["level"].values

        # curtain of total water content on the 0.25 km levels: white clear, gray no retrieval
        total = (pair["iwc"] + pair["lwc"] + pair["rwc"]).values[span].T  # [level, profile]
        w = np.where(total > MASK_THRESHOLD, total, MASK_THRESHOLD / 10)
        w[np.isnan(total)] = np.nan
        mesh = ax.pcolormesh(s, z, w, cmap=cmap, norm=norm, shading="nearest", edgecolors="0.9", linewidth=0.05)

        # each line of sight, placed at its exact position along the track at every height:
        # the nearest profile plus the offset along the local track direction
        heights = np.arange(0, 20.001, 0.125) * 1000
        track = np.stack(satellite_frame(lat[:, None], lon[:, None], heights[None, :], proj))  # [3, profile, height]
        h_idx = np.arange(len(heights))
        for j, (r, m) in enumerate(zip(shown.itertuples(), mine)):
            ray = np.stack(ray_point(x[r.col], y[r.row], heights, proj))  # [3, height]
            d = np.linalg.norm(track - ray[:, None, :], axis=0) / 1000  # [profile, height], km
            k = d.argmin(0)
            along = track[:, np.minimum(k + 1, len(s) - 1), h_idx] - track[:, np.maximum(k - 1, 0), h_idx]
            along /= np.linalg.norm(along, axis=0)
            pos = s[k] + np.sum((ray - track[:, k, h_idx]) * along, axis=0) / 1000
            close = d.min(0) <= max_km
            ax.plot(np.where(close, pos, np.nan), heights / 1000, color=RAY_COLOR, lw=1.5)  # within max_km
            if j in ends.values() and close.any():  # tag the first and last line of sight
                ax.plot(np.where(close, pos, np.nan), heights / 1000, color="k", lw=2.5, zorder=4)
                mid = np.flatnonzero(close)[len(np.flatnonzero(close)) // 2]
                ax.annotate("first" if j == ends["first"] else "last", (pos[mid], heights[mid] / 1000), xytext=(6, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=11, zorder=5,
                            bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
            ax.scatter(s[m["profile"] - span.start], z[m["level"]], s=11, color=RAY_COLOR, edgecolor="k",
                       linewidth=0.4, zorder=3)
        ax.set_xlim(s[p0 - span.start] - 3, s[p1 - span.start] + 3)
        ax.set_xlabel("Along-track distance (km)")
        ax.set_ylabel("Height (km)")
        ax.set_title("GOES-19 lines of sight")
        # earthcare flies left to right in this panel (time increasing)
        ax.annotate("", (1.0, -0.085), (0.84, -0.085), xycoords="axes fraction", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.5, mutation_scale=20))
        ax.annotate("EarthCARE flight", (0.83, -0.085), xycoords="axes fraction", ha="right", va="center",
                    color="tab:red", fontsize=11, annotation_clip=False)
        ax.legend(handles=[Line2D([], [], color=RAY_COLOR, lw=1.5, label="Line of sight"),
                           Line2D([], [], color=RAY_COLOR, marker="o", mec="k", mew=0.4, ls="", ms=5,
                                  label="Matched voxel")],
                  loc="upper left")
        return mesh

    @staticmethod
    def _geometry_panel(ax, pair, profile):
        """to-scale side view of the viewing geometry with north up: earth, goes-19 over the
        equator on the right, the line of sight to one earthcare profile and earthcare's flight
        direction. points are drawn without their east-west offset (looking from the side), so
        straight lines stay straight; the viewing angle and range in the label are exact."""
        proj = pair["goes_imager_projection"].attrs
        lat, lon = float(pair["lat"].values[profile]), float(pair["lon"].values[profile])
        a_km, b_km = proj["semi_major_axis"] / 1000, proj["semi_minor_axis"] / 1000
        H = (proj["perspective_point_height"] + proj["semi_major_axis"]) / 1000  # satellite to earth's center

        def side(position):
            """satellite-frame position (m) -> drawing (x toward goes-19 from earth's center, y north), km."""
            sx, _, sz = position
            return np.array([H - sx / 1000, sz / 1000])

        s = np.array(satellite_frame(lat, lon, 0.0, proj))
        up = np.array(satellite_frame(lat, lon, 1.0, proj)) - s
        vza = np.degrees(np.arccos(np.dot(-s, up) / np.linalg.norm(s) / np.linalg.norm(up)))
        distance = np.linalg.norm(s) / 1000
        sat, point = np.array([H, 0.0]), side(s)

        ax.add_patch(Ellipse((0, 0), 2 * a_km, 2 * b_km, facecolor="#dce6f0", edgecolor="0.4", lw=0.8))
        ax.plot([-a_km, a_km], [0, 0], color="0.6", lw=0.8, ls="--")  # equator
        ax.annotate("N", (0, b_km), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color="0.4")
        ax.annotate("S", (0, -b_km), xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                    fontsize=11, color="0.4")
        ax.annotate("Equator", (-a_km, 0), xytext=(-4, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=10, color="0.4")
        ax.plot(*zip(sat, point), color=RAY_COLOR)
        ax.plot(*point, "o", color="tab:red", ms=7, zorder=3)

        # earthcare's flight direction, seen from the same side
        lats, lons = pair["lat"].values, pair["lon"].values
        i0, i1 = max(profile - 5, 0), min(profile + 5, len(lats) - 1)
        p1 = np.array(satellite_frame(lats[i0], lons[i0], 0.0, proj))
        p2 = np.array(satellite_frame(lats[i1], lons[i1], 0.0, proj))
        motion = (p2 - p1) / np.linalg.norm(p2 - p1)
        in_page = np.array([-motion[0], motion[2]])  # toward goes-19, north
        ax.annotate("", point + 0.45 * a_km * in_page, point,
                    arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.5, mutation_scale=18), zorder=4)
        heading = "north" if motion[2] > 0 else "south"
        ax.plot(*sat, "s", color="k", ms=7)
        ax.annotate(f"GOES-19 ({abs(proj['longitude_of_projection_origin']):.1f}°W)", sat, xytext=(0, -10),
                    textcoords="offset points", ha="center", va="top")
        ns = "N" if lat >= 0 else "S"
        # label toward the equator side of the dot, clear of the N / S pole letters
        ax.annotate(f"EarthCARE, {abs(lat):.0f}°{ns}\nflying {heading}", point,
                    xytext=(-12, -10 if lat >= 0 else 10), textcoords="offset points", ha="right",
                    va="top" if lat >= 0 else "bottom", color="tab:red")
        ax.annotate(f"{vza:.0f}° viewing angle, {distance:,.0f} km", (sat + point) / 2, xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom")
        ax.set_aspect("equal")
        ax.set_xlim(-1.45 * a_km, H + 0.05 * a_km)
        ax.set_ylim(-1.2 * b_km, 1.25 * b_km)
        ax.axis("off")
        ax.set_title("Viewing geometry (to scale, north up)")

    @staticmethod
    def _grid_panel(ax, pair, shown, ids, mine, max_km, cmap, norm, vmin, vmax):
        """one column per shown pixel, one row per level along its line of sight."""
        z = pair["level"].values
        grid = np.full((len(z), len(shown)), np.nan)  # [level, pixel]
        for j, m in enumerate(mine):
            grid[m["level"], j] = m["total"]
        w = np.where(grid > MASK_THRESHOLD, grid, MASK_THRESHOLD / 10)  # clear -> "under" -> white
        w[np.isnan(grid)] = np.nan  # no earthcare voxel within max_km -> gray
        edges_x = np.arange(ids[0], ids[-1] + 2) - 0.5
        edges_z = np.r_[z - 0.125, z[-1] + 0.125]
        mesh = ax.pcolormesh(edges_x, edges_z, w, cmap=cmap, norm=norm, edgecolors="0.85", linewidth=0.2)
        ax.set_xlabel("Matched pixel #")
        ax.set_ylabel("Height along line of sight (km)")
        ax.set_xticks(ids[ids % 5 == 0])
        # what goes saw in each of those pixels, same gray scale as the map
        strip = ax.inset_axes([0, 1.02, 1, 0.06], sharex=ax)
        strip.pcolormesh(edges_x, [0, 1], pair["CMI_C13"].values[shown["row"], shown["col"]][None],
                         cmap="Greys", vmin=vmin, vmax=vmax, edgecolors="0.5", linewidth=0.3)
        strip.set_yticks([0.5], ["Band 13"])
        strip.tick_params(labelbottom=False, bottom=False)
        ax.set_title("Matched voxels (gray = no match)", pad=34)
        return mesh

    @styled
    def curtain(self, pair):
        """along-track view of a collocated pair: what goes saw of each profile (true
        color strip and band 13, from the pixel matched to its cloud top, see
        profile_pixels) over the ACM-CAP ice, liquid and rain water content curtains,
        sharing latitude on the x axis.

        pair: dataset from GOES19.collocate(). clouds are gray (thin) to white (thick),
        blue = clear (below the 1e-5 g m^-3 mask threshold), dark gray = no valid retrieval.
        """
        lat, z = pair["lat"].values, pair["level"].values
        goes = self.goes_along_track(pair)
        water = {"Ice": "iwc", "Liquid": "lwc", "Rain": "rwc"}

        fig, axes = plt.subplots(2 + len(water), 1, figsize=(11, 10), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [0.35, 1] + [1.6] * len(water)})
        # x runs in flight order, so the overpass starts on the left
        axes[-1].set_xlim(lat[0], lat[-1])

        # what goes saw of each profile
        axes[0].imshow(self.true_color(goes), extent=(lat[0], lat[-1], 0, 1), aspect="auto")
        axes[0].set_yticks([])
        axes[0].set_ylabel("True\ncolor", rotation=0, ha="right", va="center")
        axes[1].plot(lat, goes["CMI_C13"].values, color="0.2", lw=1.2)
        axes[1].invert_yaxis()  # colder (higher) cloud tops point up, like the curtains
        axes[1].set_ylabel("Band 13 (K)\ncold ↑")
        axes[1].grid(alpha=0.3, lw=0.5)

        # earthcare curtains, one shared log scale so the panels compare directly
        # clouds gray (thin) to white (thick) on a blue sky, like looking at clouds
        cmap = CLOUD_CMAP.with_extremes(bad=(0, 0, 0, 0))  # transparent where no retrieval, so the facecolor shows
        norm = LogNorm(vmin=MASK_THRESHOLD, vmax=1.0, clip=False)
        for ax, (name, var) in zip(axes[2:], water.items()):
            w = pair[var].values.T  # [level, profile]
            w = np.where(w > MASK_THRESHOLD, w, MASK_THRESHOLD / 10)  # clear -> "under" -> sky blue
            w[np.isnan(pair[var].values.T)] = np.nan  # no retrieval -> transparent -> dark facecolor
            mesh = ax.pcolormesh(lat, z, w, cmap=cmap, norm=norm, shading="nearest")
            ax.set_facecolor(NO_DATA_COLOR)
            ax.set_ylim(0, 20)
            ax.set_ylabel(f"{name}\nheight (km)")
        fig.colorbar(mesh, ax=axes[2:], shrink=0.6, label="Water content (g m$^{-3}$)")
        axes[-1].set_xlabel("Latitude (°)")

        # longitude on the top axis, to match locations on the track_on_goes map;
        # the track is nearly straight in lat/lon within one crop, so a linear fit suffices
        slope, offset = np.polyfit(lat, pair["lon"].values, 1)
        to_lon = lambda v: slope * np.asarray(v) + offset
        to_lat = lambda v: (np.asarray(v) - offset) / slope
        top = axes[0].secondary_xaxis("top", functions=(to_lon, to_lat))
        top.xaxis.set_major_locator(MaxNLocator(6))
        top.set_xlabel("Longitude (°)")

        fig.suptitle(self._title(pair))
        self._save(fig, pair, suffix="_curtain")
        return fig

    @staticmethod
    def _title(pair):
        """figure title shared by every plot of a pair."""
        return (f"EarthCARE {pair.attrs['orbit_and_frame']} × GOES-19, "
                f"{pd.Timestamp(pair.attrs['scan_start']):%Y-%m-%d %H:%M} UTC")

    def _save(self, fig, pair, suffix=""):
        """save the figure if out_dir is set."""
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            name = Path(pair.attrs["scan_file"]).stem
            fig.savefig(self.out_dir / f"{pair.attrs['orbit_and_frame']}_{name}{suffix}.png",
                        dpi=150, bbox_inches="tight")
