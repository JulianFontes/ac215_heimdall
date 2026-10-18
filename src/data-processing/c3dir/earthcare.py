"""earthcare ACM-CAP frames over the goes-19 view, for C3DIR (White et al., 2026).

finds the ACM_CAP_2B frames that goes-19 can see and downloads them with earthcarekit,
as they come (~50 MB .h5 each), to RAW_DIR/earthcare/. turning them into C3DIR labels is
c3dir.labels' job.

downloads require the one-time ESA MAAP login setup:
https://tropos-rsd.github.io/earthcarekit/setup/
"""
from pathlib import Path

import earthcarekit as eck
import pandas as pd

from c3dir.constants import GOES19_LON, VIEW_RADIUS_DEG
from c3dir.geometry import view_circle, within_view
from utilities.paths import RAW_DIR


class EarthCareFrames:
    """
    raw_dir: where the .h5 frames are saved.
    view_radius_deg: angular distance (from earth's center) from the point under goes-19 that
        still counts as visible. 65° corresponds to a ~73° viewing angle from goes; an
        assumption to limit the data, not directly from the paper.
    """

    file_type = "ACM_CAP_2B"  # the combined cloud/aerosol/precip product C3DIR trains on

    def __init__(self, raw_dir=RAW_DIR / "earthcare", lon0=GOES19_LON, view_radius_deg=VIEW_RADIUS_DEG):
        self.raw_dir = Path(raw_dir)
        self.lon0 = lon0
        self.view_radius_deg = view_radius_deg

    def view_polygon(self, n=72):
        """geojson circle covering what goes-19 can see, used as the search area.
        the MAAP search api currently rejects earthcarekit's `bounding_box`, so a
        polygon is used instead."""
        lat, lon = view_circle(self.lon0, self.view_radius_deg, n)
        ring = [[float(x), float(y)] for x, y in zip(lon, lat)]
        return {"type": "Polygon", "coordinates": [ring + ring[:1]]}  # geojson wants the ring closed

    def search(self, start, end):
        """list the ACM_CAP_2B frames in [start, end] that cross the goes-19 view (doesn't download)."""
        df = eck.ecdownload(file_type=self.file_type, start_time=str(start), end_time=str(end),
                            geometry=self.view_polygon(), is_download=False,
                            return_results=True, verbose=False)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return pd.DataFrame(df).sort_values("start_sensing_time").reset_index(drop=True)

    def download(self, orbit_and_frame):
        """the frame's .h5 file, downloaded only if it isn't already on disk."""
        pattern = f"ECA_*_{self.file_type}_*_{orbit_and_frame}.h5"
        if not any(self.raw_dir.rglob(pattern)):
            self.raw_dir.mkdir(parents=True, exist_ok=True)  # earthcarekit won't create it
            eck.ecdownload(file_type=self.file_type, orbit_and_frame=orbit_and_frame,
                           path_to_data=str(self.raw_dir), is_create_subdirs=False, verbose=False)
        hits = sorted(self.raw_dir.rglob(pattern))  # latest processing time sorts last
        if not hits:
            raise FileNotFoundError(f"no {self.file_type} file for {orbit_and_frame}")
        return hits[-1]

    def profiles(self, path):
        """time, lat and lon of the frame's profiles that goes-19 can see (for finding goes scans)."""
        with eck.read_any(str(path)) as ds:
            lat, lon, time = ds["latitude"].values, ds["longitude"].values, ds["time"].values
        keep = within_view(lat, lon, self.lon0, self.view_radius_deg)
        return pd.DataFrame({"time": time[keep], "lat": lat[keep], "lon": lon[keep]})
