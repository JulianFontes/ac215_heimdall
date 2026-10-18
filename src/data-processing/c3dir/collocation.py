"""goes-19 strips collocated voxel by voxel with the earthcare labels.

follows the C3DIR paper's 3-D collocation (sections 3.1-3.2). goes sees through the
atmosphere along a slanted line of sight, so each goes pixel is a column of voxels at
the 80 C3DIR heights along that line, not straight up. every labeled earthcare voxel
(profile, height) is matched to the goes pixel whose line of sight passes closest at
that height, if the two are within 5 km horizontally and within 2.5 min of when goes
imaged that pixel (the north-to-south scan included). each goes voxel keeps a single
earthcare voxel, the closest. goes revisits each pixel every 10 min (full disk) or
5 min (CONUS), so an earthcare voxel matches at most one scan. the goes image is cropped
around the matched pixels.

reads the strips from c3dir.goes (RAW_DIR/goes/) and the labels; each pair is cached as
PROCESSED_DIR/c3dir/collocated/<orbit_and_frame>_<scan start>_<domain>.nc, holding the goes
crop (y, x), the full label columns of the matched profiles (profile, level), and the matched
voxels (voxel). strips without a match get a .skip file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from c3dir.labels import EarthCareLabels
from c3dir.constants import BANDS, TARGETS
from c3dir.geometry import closest_ray, satellite_frame, scan_angles
from utilities.paths import PROCESSED_DIR, RAW_DIR


class Collocation:
    """
    strip_dir: the collector's goes strips. out_dir: where the collocated pairs go.
    max_seconds: largest allowed time gap between an earthcare voxel and its goes pixel
        (150 s = the paper's 2.5 min, section 3.2).
    max_km: largest allowed horizontal distance between an earthcare voxel and the goes
        line of sight at the same height (5 km, paper section 3.2).
    pad_px: goes pixels kept around the matched pixels, as context for the model;
        an assumption, not directly from the paper.
    """

    def __init__(self, labels=None, strip_dir=RAW_DIR / "goes", out_dir=PROCESSED_DIR / "c3dir" / "collocated",
                 max_seconds=150.0, max_km=5.0, pad_px=64):
        self.labels = labels or EarthCareLabels()
        self.strip_dir = Path(strip_dir)
        self.out_dir = Path(out_dir)
        self.max_seconds = pd.Timedelta(seconds=max_seconds)
        self.max_km = max_km
        self.pad_px = pad_px

    def run(self, start=None, end=None):
        """collocate every strip (or those whose scan starts in [start, end]) not done yet;
        returns the paths of the collocated pairs in that window, new or cached."""
        out = []
        for strip_path in sorted(self.strip_dir.glob("*.nc")):
            with xr.open_dataset(strip_path) as s:
                scan_start = pd.Timestamp(s.attrs["scan_start"])
            if (start and scan_start < pd.Timestamp(start)) or (end and scan_start > pd.Timestamp(end)):
                continue
            path = self.out_dir / strip_path.name
            skip = path.with_suffix(".skip")
            if skip.exists():
                continue
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    pair = self.collocate(xr.load_dataset(strip_path))
                except ValueError as error:
                    skip.write_text(f"{error}\n")  # no match: don't try again
                    continue
                pair.to_netcdf(path)
            out.append(path)
        return out

    def collocate(self, strip):
        """the collocated pair for one goes strip, with its frame's labels."""
        frame = strip.attrs["orbit_and_frame"]
        labels = self.labels.load(frame)
        times = pd.to_datetime(labels["time"].values)
        scan_start, scan_end = pd.Timestamp(strip.attrs["scan_start"]), pd.Timestamp(strip.attrs["scan_end"])
        # cheap prefilter on the whole-scan window; the per-pixel test is in _match
        in_scan = (times >= scan_start - self.max_seconds) & (times <= scan_end + self.max_seconds)
        if labels.sizes["profile"] == 0 or not in_scan.any():
            raise ValueError("no earthcare profiles within the scan's time window")
        pair = self._match(strip, labels.isel(profile=in_scan))
        pair.attrs = {"orbit_and_frame": frame, "scan_file": strip.attrs["scan_file"],
                      "scan_start": strip.attrs["scan_start"], "scan_end": strip.attrs["scan_end"],
                      "max_seconds": self.max_seconds.total_seconds(), "max_km": self.max_km}
        return pair

    def _match(self, ds, labels):
        """match earthcare voxels to goes line-of-sight voxels and crop goes around them."""
        p = ds["goes_imager_projection"].attrs
        x, y = ds.x.values, ds.y.values
        heights = labels["level"].values * 1000.0  # level centers, m

        # every labeled earthcare voxel: all three water contents valid
        labeled = np.all([np.isfinite(labels[v].values) for v in TARGETS[:3]], axis=0)
        prof, lev = np.nonzero(labeled)
        lat, lon, z = labels["lat"].values[prof], labels["lon"].values[prof], heights[lev]

        # first guess: the pixel nearest in scan angle to where goes sees each voxel
        vx, vy = scan_angles(lat, lon, z, p)
        col = np.rint((vx - x[0]) / (x[1] - x[0]))
        row = np.rint((vy - y[0]) / (y[1] - y[0]))
        on_grid = np.isfinite(col) & np.isfinite(row) & (col >= -2) & (col < len(x) + 2) & \
                  (row >= -2) & (row < len(y) + 2)
        if not on_grid.any():
            raise ValueError("no earthcare voxels inside this scan")
        prof, lev, z, col, row = prof[on_grid], lev[on_grid], z[on_grid], \
            col[on_grid].astype(int), row[on_grid].astype(int)

        # the pixel whose line of sight passes closest at the voxel's height, and that horizontal
        # distance (paper: <= 5 km). at oblique views a footprint is stretched diagonally across
        # the scan-angle grid, so the rounded guess can be a pixel or two off: check its 5x5 block
        voxel = np.stack(satellite_frame(labels["lat"].values[prof], labels["lon"].values[prof], z, p))
        row, col, distance_km = closest_ray(voxel, z, row, col, x, y, p)
        # band 13 checks the pixel actually has data (not space or a gap)
        has_data = np.isfinite(ds["CMI_C13"].isel(y=xr.DataArray(row), x=xr.DataArray(col)).values)
        # per-pixel time check, the 2.5 min rule from the C3DIR paper (section 3.2)
        pixel_time = ds["pixel_time"].values
        dt = pd.to_datetime(labels["time"].values[prof]) - pd.to_datetime(pixel_time[row])
        keep = (distance_km <= self.max_km) & has_data & (np.abs(dt) <= self.max_seconds)
        if not keep.any():
            raise ValueError(f"no earthcare voxels within {self.max_km} km and {self.max_seconds} "
                             "of a goes line of sight")

        # one earthcare voxel per goes voxel (paper section 3.2): the closest
        order = np.flatnonzero(keep)[np.argsort(distance_km[keep], kind="stable")]
        key = (lev[order] * len(y) + row[order]) * len(x) + col[order]
        chosen = np.sort(order[np.unique(key, return_index=True)[1]])

        # crop around every matched pixel, at any height, with pad_px of context
        r0, r1 = max(row[chosen].min() - self.pad_px, 0), min(row[chosen].max() + self.pad_px + 1, len(y))
        c0, c1 = max(col[chosen].min() - self.pad_px, 0), min(col[chosen].max() + self.pad_px + 1, len(x))
        crop = ds[BANDS + ["goes_imager_projection", "pixel_time"]].isel(y=slice(r0, r1), x=slice(c0, c1))

        # full label columns of the matched profiles, for plots
        used, voxel_profile = np.unique(prof[chosen], return_inverse=True)
        matched = labels.isel(profile=used)
        # the matched voxels: the training labels
        values = np.stack([labels[v].values[prof[chosen], lev[chosen]] for v in TARGETS], axis=1)
        voxels = xr.Dataset({
            "voxel_level": ("voxel", lev[chosen].astype(np.int16), {"description": "index into level"}),
            "voxel_row": ("voxel", (row[chosen] - r0).astype(np.int32), {"description": "goes row in the crop"}),
            "voxel_col": ("voxel", (col[chosen] - c0).astype(np.int32), {"description": "goes column in the crop"}),
            "voxel_profile": ("voxel", voxel_profile.astype(np.int32), {"description": "index into profile"}),
            "voxel_distance_km": ("voxel", distance_km[chosen].astype(np.float32),
                                  {"description": "horizontal distance to the goes line of sight"}),
            "voxel_dt_seconds": ("voxel", dt[chosen].total_seconds().values.astype(np.float32),
                                 {"description": "EarthCARE time minus GOES pixel time"}),
            "voxel_labels": (("voxel", "target"), values.astype(np.float32),
                             {"description": "C3DIR targets of the matched earthcare voxel"}),
        }, coords={"target": TARGETS})
        # goes crop (y, x), profiles (profile, level) and voxels (voxel) side by side in one file
        return xr.merge([crop, matched, voxels], combine_attrs="override")
