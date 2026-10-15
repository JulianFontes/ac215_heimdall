"""goes-19 ABI scans collocated voxel by voxel with the earthcare labels.

follows the C3DIR paper's 3-D collocation (sections 3.1-3.2). goes sees through the
atmosphere along a slanted line of sight, so each goes pixel is a column of voxels at
the 80 C3DIR heights along that line, not straight up. every labeled earthcare voxel
(profile, height) is matched to the goes pixel whose line of sight passes closest at
that height, if the two are within 5 km horizontally and within 2.5 min of when goes
imaged that pixel (the north-to-south scan included). each goes voxel keeps a single
earthcare voxel, the closest. goes revisits each pixel every 10 min (full disk) or
5 min (CONUS), so an earthcare voxel matches at most one scan. each scan (goes2go,
MCMIP: all 16 bands on the 2 km grid) is cropped around the matched pixels.

each pair is cached as data/collocated/<orbit_and_frame>_<scan start>_<domain>.nc,
holding the goes crop (y, x), the full label columns of the matched profiles
(profile, level), and the matched voxels (voxel).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from goes2go import GOES

from heimdall.c3dir.constants import BANDS, TARGETS
from heimdall.c3dir.geometry import closest_ray, ray_point, satellite_frame, scan_angles
from heimdall.paths import DATA_DIR

SCAN_MINUTES = {"F": 10, "C": 5}  # how often each domain gets scanned


class GOES19:
    """
    data_dir: raw scans go in <data_dir>/goes (goes2go's folder layout),
        collocated pairs in <data_dir>/collocated.
    domain: "F" full disk (every 10 min, what the C3DIR paper uses) or "C" CONUS (every 5 min).
    max_seconds: largest allowed time gap between an earthcare voxel and its goes pixel
        (150 s = the paper's 2.5 min, section 3.2).
    max_km: largest allowed horizontal distance between an earthcare voxel and the goes
        line of sight at the same height (5 km, paper section 3.2).
    pad_px: goes pixels kept around the matched pixels, as context for the model;
        an assumption, not directly from the paper.
    keep_raw: keep the downloaded scan (~250 MB for full disk) after cropping.
    """

    def __init__(self, data_dir=DATA_DIR, domain="F", max_seconds=150.0, max_km=5.0, pad_px=64,
                 keep_raw=False):
        self.raw_dir = Path(data_dir) / "goes"
        self.collocated_dir = Path(data_dir) / "collocated"
        self.domain = domain
        self.max_seconds = pd.Timedelta(seconds=max_seconds)
        self.max_km = max_km
        self.pad_px = pad_px
        self.keep_raw = keep_raw
        # mcmip = cloud and moisture imagery, every band in one file
        self.client = GOES(satellite=19, product="ABI-L2-MCMIP", domain=domain)

    def scans(self, start, end):
        """list the scans that overlap [start, end] (no download)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # goes2go keeps only files entirely inside the window, so pad by a scan
        pad = pd.Timedelta(minutes=SCAN_MINUTES[self.domain])
        df = self.client.df(start - pad, end + pad)
        return df[(df.end >= start) & (df.start <= end)].reset_index(drop=True)

    def collocate(self, labels):
        """collocate one frame of labels with every goes scan it overlaps.

        labels: dataset from EarthCareLabels.load(). returns one dataset (goes crop +
        matched voxels) per scan with at least one matched voxel.
        """
        if labels.sizes["profile"] == 0:
            return []
        frame = labels.attrs["orbit_and_frame"]
        times = pd.to_datetime(labels["time"].values)
        # every scan that could overlap the overpass
        scans = self.scans(times.min() - self.max_seconds, times.max() + self.max_seconds)
        out = []
        for scan in scans.itertuples():
            # cheap prefilter on the whole-scan window; the per-pixel test is in _build
            in_scan = (times >= scan.start - self.max_seconds) & (times <= scan.end + self.max_seconds)
            cache = self.collocated_dir / f"{frame}_{scan.start:%Y%m%dT%H%M%S}_{self.domain}.nc"
            skip = cache.with_suffix(".skip")  # scans already checked and found to have no match
            if not in_scan.any() or skip.exists():
                continue
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                try:
                    pair = self._build(scan, labels.isel(profile=in_scan))
                except ValueError as error:
                    # mark the miss so this scan isn't downloaded again
                    skip.write_text(f"{error}\n")
                    continue
                pair.to_netcdf(cache)
            out.append(xr.load_dataset(cache))
        return out

    def _build(self, scan, labels):
        # selects exactly this file: goes2go keeps start >= scan.start and end <= scan.end
        self.client.timerange(scan.start, scan.end, return_as="filelist", download=True,
                              save_dir=self.raw_dir, verbose=False)
        raw = self.raw_dir / scan.file
        try:
            with xr.open_dataset(raw) as ds:
                pair = self._match(ds, labels)
        finally:
            # delete the large raw scan once it's cropped
            if not self.keep_raw:
                raw.unlink(missing_ok=True)
        pair.attrs = {"orbit_and_frame": labels.attrs["orbit_and_frame"], "scan_file": scan.file,
                      "scan_start": str(scan.start), "scan_end": str(scan.end),
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
        pixel_time = self.row_times(ds)
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
        crop = ds[BANDS + ["goes_imager_projection"]].isel(y=slice(r0, r1), x=slice(c0, c1)).load()
        crop["pixel_time"] = ("y", pixel_time[r0:r1], {"description": "estimated time each row was imaged"})

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

    @staticmethod
    def row_times(ds):
        """approximate time each row of the scan was imaged.

        goes scans north to south (C3DIR paper section 3.2), so rows are spread
        evenly between time_coverage_start (top) and time_coverage_end (bottom).
        """
        start = pd.Timestamp(ds.attrs["time_coverage_start"]).tz_localize(None)
        end = pd.Timestamp(ds.attrs["time_coverage_end"]).tz_localize(None)
        y = ds.y.values
        # 0 at the top row, 1 at the bottom row
        fraction = (y.max() - y) / (y.max() - y.min())
        return start.to_datetime64() + (fraction * (end - start).value).astype("timedelta64[ns]")
