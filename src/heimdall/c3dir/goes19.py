"""goes-19 ABI scans collocated in time and space with the earthcare labels.

time matching follows the C3DIR paper (section 3.2): a profile is kept only if
earthcare observed it within 2.5 min of when goes imaged that pixel, with pixel
times accounting for the north-to-south scan. goes revisits each pixel every
10 min (full disk) or 5 min (CONUS), so a profile matches at most one scan.
each scan (goes2go, MCMIP: all 16 bands on the 2 km grid) is cropped around the
matched profiles, padded for parallax.

each pair is cached as data/collocated/<orbit_and_frame>_<scan start>_<domain>.nc,
holding the goes crop (y, x) and the matched labels (profile, level).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from goes2go import GOES
from pyproj import Proj

from heimdall.paths import DATA_DIR

BANDS = [f"CMI_C{band:02d}" for band in range(1, 17)]  # all 16 abi bands
SCAN_MINUTES = {"F": 10, "C": 5}  # how often each domain gets scanned


def goes_proj(ds):
    """pyproj projection for the goes grid (x/y in meters = scan angle * satellite height)."""
    p = ds["goes_imager_projection"].attrs
    return Proj(proj="geos", h=p["perspective_point_height"],
                lon_0=p["longitude_of_projection_origin"], sweep=p["sweep_angle_axis"],
                a=p["semi_major_axis"], b=p["semi_minor_axis"])


class GOES19:
    """
    data_dir: raw scans go in <data_dir>/goes (goes2go's folder layout),
        collocated pairs in <data_dir>/collocated.
    domain: "F" full disk (every 10 min, what the C3DIR paper uses) or "C" CONUS (every 5 min).
    max_seconds: largest allowed time gap between a profile and its goes pixel
        (150 s = the paper's 2.5 min, section 3.2).
    pad_px: goes pixels kept on each side of the track. parallax shifts a 20 km
        cloud top by at most ~9 px at the edge of the view, so 64 leaves a wide
        margin; an assumption, not directly from the paper.
    keep_raw: keep the downloaded scan (~250 MB for full disk) after cropping.
    """

    def __init__(self, data_dir=DATA_DIR, domain="F", max_seconds=150.0, pad_px=64, keep_raw=False):
        self.raw_dir = Path(data_dir) / "goes"
        self.collocated_dir = Path(data_dir) / "collocated"
        self.domain = domain
        self.max_seconds = pd.Timedelta(seconds=max_seconds)
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

        labels: dataset from EarthCareLabels.load(). returns one dataset
        (goes crop + matched labels) per scan with at least one time-matched profile.
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
                      "max_seconds": self.max_seconds.total_seconds()}
        return pair

    def _match(self, ds, labels):
        """keep the profiles within max_seconds of their pixel's time and crop goes around them."""
        # profile positions in scan angles (rad); off-disk points come back as 1e30
        h = ds["goes_imager_projection"].attrs["perspective_point_height"]
        px, py = goes_proj(ds)(labels["lon"].values, labels["lat"].values)
        px, py = np.asarray(px) / h, np.asarray(py) / h
        step = abs(float(ds.x[1] - ds.x[0]))
        x, y = ds.x.values, ds.y.values
        on_grid = (px >= x.min() - step / 2) & (px <= x.max() + step / 2) & \
                  (py >= y.min() - step / 2) & (py <= y.max() + step / 2)
        if not on_grid.any():
            raise ValueError("no profiles inside this scan")
        # nearest pixel (centers are evenly spaced; clip before casting the 1e30s)
        col = np.clip(np.rint((px - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(int)
        row = np.clip(np.rint((py - y[0]) / (y[1] - y[0])), 0, len(y) - 1).astype(int)
        # use band 13 to check the pixel actually has data (not space or a gap)
        c13 = ds["CMI_C13"].isel(y=xr.DataArray(row), x=xr.DataArray(col)).values
        has_data = on_grid & np.isfinite(c13)

        # per-pixel time check, the 2.5 min rule from the C3DIR paper (section 3.2)
        pixel_time = self.row_times(ds)
        dt = pd.to_datetime(labels["time"].values) - pd.to_datetime(pixel_time[row])
        keep = has_data & (np.abs(dt) <= self.max_seconds)
        if not keep.any():
            raise ValueError(f"no profiles within {self.max_seconds} of their GOES pixel")

        # crop a box around the matched track with some padding for parallax
        pad = self.pad_px * step
        crop_x = slice(px[keep].min() - pad, px[keep].max() + pad)
        crop_y = slice(py[keep].max() + pad, py[keep].min() - pad)  # y runs north to south
        crop = ds[BANDS + ["goes_imager_projection"]].sel(x=crop_x, y=crop_y).load()
        crop["pixel_time"] = ("y", pixel_time[np.isin(y, crop.y.values)],
                              {"description": "estimated time each row was imaged"})
        matched = labels.isel(profile=keep)
        matched["dt_seconds"] = ("profile", dt[keep].total_seconds().values.astype(np.float32),
                                 {"description": "EarthCARE time minus GOES pixel time"})
        # goes crop (y, x) and labels (profile, level) live side by side in one file
        return xr.merge([crop, matched], combine_attrs="override")

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
