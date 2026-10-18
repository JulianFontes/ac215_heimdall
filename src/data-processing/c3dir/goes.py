"""goes-19 ABI scans around each earthcare overpass, for C3DIR collocation.

c3dir.collocation matches earthcare voxels to goes pixels whose line of sight passes within
5 km at the same height, imaged within 2.5 min (C3DIR paper sections 3.1-3.2). only a strip
along the track can ever match, so each full-disk scan (~250 MB, goes2go MCMIP: all 16 bands on
the 2 km grid) is downloaded to CACHE_DIR, cut to that strip, and deleted. the strip covers
wherever the track appears at any height from 0 to 20 km (goes sees at a slant), plus the
context kept around matched pixels and the matching search reach.

each strip is saved as RAW_DIR/goes/<orbit_and_frame>_<scan start>_<domain>.nc with the 16
bands, the projection, and pixel_time (when each row was imaged, needed for the 2.5 min rule).
scans where the track falls outside the image get an empty .skip file so they aren't downloaded
again.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from goes2go import GOES

from c3dir.constants import BANDS, LEVELS_KM
from c3dir.geometry import scan_angles
from utilities.paths import CACHE_DIR, RAW_DIR

SCAN_MINUTES = {"F": 10, "C": 5}  # how often each domain gets scanned


class GOESStrips:
    """
    raw_dir: where the strips are saved. cache_dir: full scans while they are cut (goes2go's layout).
    domain: "F" full disk (every 10 min, what the C3DIR paper uses) or "C" CONUS (every 5 min).
    max_seconds: largest time gap between an earthcare profile and a scan for the scan to be
        used (150 s = the paper's 2.5 min, section 3.2).
    pad_px: goes pixels of context the processor keeps around matched pixels; an assumption, not
        directly from the paper. reach: how far the matching search looks around its first
        guess (a 5 x 5 block). the strip keeps both around the track.
    keep_full: keep the downloaded full scan after cutting the strip.
    """

    def __init__(self, raw_dir=RAW_DIR / "goes", cache_dir=CACHE_DIR / "goes", domain="F",
                 max_seconds=150.0, pad_px=64, reach=2, keep_full=False):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.domain = domain
        self.max_seconds = pd.Timedelta(seconds=max_seconds)
        self.margin = pad_px + reach
        self.keep_full = keep_full
        # mcmip = cloud and moisture imagery, every band in one file
        self.client = GOES(satellite=19, product="ABI-L2-MCMIP", domain=domain)

    def scans(self, start, end):
        """list the scans that overlap [start, end] (no download)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # goes2go keeps only files entirely inside the window, so pad by a scan
        pad = pd.Timedelta(minutes=SCAN_MINUTES[self.domain])
        df = self.client.df(start - pad, end + pad)
        return df[(df.end >= start) & (df.start <= end)].reset_index(drop=True)

    def collect(self, orbit_and_frame, profiles):
        """the strips of every scan within max_seconds of the overpass; returns their paths.

        profiles: time / lat / lon of the frame's visible profiles (EarthCareFrames.profiles).
        """
        if profiles.empty:
            return []
        times = pd.to_datetime(profiles["time"].values)
        scans = self.scans(times.min() - self.max_seconds, times.max() + self.max_seconds)
        out = []
        for scan in scans.itertuples():
            # profiles close enough in time to this scan; the per-pixel test is in the processor
            in_scan = (times >= scan.start - self.max_seconds) & (times <= scan.end + self.max_seconds)
            path = self.raw_dir / f"{orbit_and_frame}_{scan.start:%Y%m%dT%H%M%S}_{self.domain}.nc"
            skip = path.with_suffix(".skip")
            if not in_scan.any() or skip.exists():
                continue
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                strip = self._strip(scan, profiles[in_scan])
                if strip is None:
                    skip.write_text("the track falls outside this scan\n")  # don't download it again
                    continue
                strip.attrs = {"orbit_and_frame": orbit_and_frame, "scan_file": scan.file,
                               "scan_start": str(scan.start), "scan_end": str(scan.end),
                               "domain": self.domain}
                strip.to_netcdf(path)
            out.append(path)
        return out

    def _strip(self, scan, profiles):
        # selects exactly this file: goes2go keeps start >= scan.start and end <= scan.end
        self.client.timerange(scan.start, scan.end, return_as="filelist", download=True,
                              save_dir=self.cache_dir, verbose=False)
        full = self.cache_dir / scan.file
        try:
            with xr.open_dataset(full) as ds:
                return self._cut(ds, profiles)
        finally:
            if not self.keep_full:
                full.unlink(missing_ok=True)  # the full scan is ~250 MB

    def _cut(self, ds, profiles):
        """the rows and columns where the track can appear at any C3DIR height, plus the margin."""
        p = ds["goes_imager_projection"].attrs
        x, y = ds.x.values, ds.y.values
        n = len(profiles)
        lat = np.repeat(profiles["lat"].values, len(LEVELS_KM))
        lon = np.repeat(profiles["lon"].values, len(LEVELS_KM))
        z = np.tile(LEVELS_KM * 1000.0, n)
        vx, vy = scan_angles(lat, lon, z, p)
        col, row = np.rint((vx - x[0]) / (x[1] - x[0])), np.rint((vy - y[0]) / (y[1] - y[0]))
        # same first-guess window as the matching: up to 2 pixels outside the image still counts
        ok = np.isfinite(col) & np.isfinite(row) & (col >= -2) & (col < len(x) + 2) & \
            (row >= -2) & (row < len(y) + 2)
        if not ok.any():
            return None
        r0 = int(max(row[ok].min() - self.margin, 0))
        r1 = int(min(row[ok].max() + self.margin + 1, len(y)))
        c0 = int(max(col[ok].min() - self.margin, 0))
        c1 = int(min(col[ok].max() + self.margin + 1, len(x)))
        strip = ds[BANDS + ["goes_imager_projection"]].isel(y=slice(r0, r1), x=slice(c0, c1)).load()
        strip["pixel_time"] = ("y", self.row_times(ds)[r0:r1],
                               {"description": "estimated time each row was imaged"})
        return strip

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
