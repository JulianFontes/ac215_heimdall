"""earthcare ACM-CAP labels for C3DIR (White et al., 2026).

finds the ACM_CAP_2B frames that goes-19 can see, downloads them with
earthcarekit and converts them to the C3DIR targets (paper sections 2.2-2.3):
ice, liquid and rain water content, ice and liquid effective radius, and four
detection masks (water content > 1e-5 g m^-3), averaged from ~100 m onto 80
levels of 0.25 km.

levels: the paper gives "-0.125 to 19.875 km", but its figure 2 levels
(0.125, 14.875 km) and the 80-level output only fit 0-20 km bins, so the level
centers here are 0.125 ... 19.875 km.

each frame is cached as data/earthcare/labels/<orbit_and_frame>.nc. downloads
require the one-time ESA MAAP login setup:
https://tropos-rsd.github.io/earthcarekit/setup/
"""
from pathlib import Path

import earthcarekit as eck
import numpy as np
import pandas as pd
import xarray as xr

from heimdall.paths import DATA_DIR

GOES19_LON = -75.2  # goes-19 sub-satellite longitude
LEVELS_KM = np.arange(80) * 0.25 + 0.125  # 80 levels every 0.25 km, from the C3DIR paper (section 2.3)
MASK_THRESHOLD = 1e-5  # g m^-3, cloud/no-cloud cutoff from the C3DIR paper (section 2.2)


class EarthCareLabels:
    """
    data_dir: raw frames go in <data_dir>/earthcare/raw, labels in <data_dir>/earthcare/labels.
    view_radius_deg: angular distance (from earth's center) from the point under
        goes-19 that still counts as visible. 65° corresponds to a ~73° viewing
        angle from goes; an assumption to limit the data, not directly from the paper.
    keep_raw: keep the ~50 MB .h5 frame after its labels are cached.
    """

    file_type = "ACM_CAP_2B"  # the combined cloud/aerosol/precip product C3DIR trains on

    def __init__(self, data_dir=DATA_DIR, lon0=GOES19_LON, view_radius_deg=65.0, keep_raw=True):
        self.raw_dir = Path(data_dir) / "earthcare" / "raw"
        self.label_dir = Path(data_dir) / "earthcare" / "labels"
        self.lon0 = lon0
        self.view_radius_deg = view_radius_deg
        self.keep_raw = keep_raw

    def view_polygon(self, n=72):
        """geojson circle covering what goes-19 can see, used as the search area.
        the MAAP search api currently rejects earthcarekit's `bounding_box`, so a
        polygon is used instead."""
        # points on a circle of radius view_radius_deg around the sub-satellite point
        azimuth = np.deg2rad(np.linspace(360, 0, n, endpoint=False))  # counter-clockwise
        d = np.deg2rad(self.view_radius_deg)
        lat = np.rad2deg(np.arcsin(np.sin(d) * np.cos(azimuth)))
        lon = self.lon0 + np.rad2deg(np.arctan2(np.sin(azimuth) * np.sin(d), np.cos(d)))
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

    def visible(self, lat, lon):
        """which profiles are within view_radius_deg of the point under goes-19."""
        # cosine of the angular distance from the point under the satellite
        cos_d = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon - self.lon0))
        return np.isfinite(cos_d) & (cos_d >= np.cos(np.deg2rad(self.view_radius_deg)))

    def load(self, orbit_and_frame):
        """labels for one frame as an xarray dataset; built and cached on first use.

        dims: profile (along track, ~1 km apart) x level (0.25 km).
        variables: time/lat/lon per profile; iwc/lwc/rwc [g m^-3] and the four
        masks per (profile, level), nan where ACM-CAP has no valid retrieval;
        ice_radius/liquid_radius [µm], nan where that phase is absent.
        """
        cache = self.label_dir / f"{orbit_and_frame}.nc"
        if not cache.exists():
            # not cached yet: download, convert to labels, save
            raw = self._download(orbit_and_frame)
            labels = self._read(raw)
            labels.attrs["orbit_and_frame"] = orbit_and_frame
            cache.parent.mkdir(parents=True, exist_ok=True)
            labels.to_netcdf(cache)
            if not self.keep_raw:
                for file in raw.parent.glob(f"{raw.stem}.*"):  # the .h5 and its .HDR
                    file.unlink()
        return xr.load_dataset(cache)

    def _download(self, orbit_and_frame):
        pattern = f"ECA_*_{self.file_type}_*_{orbit_and_frame}.h5"
        # download only if the file isn't already on disk
        if not any(self.raw_dir.rglob(pattern)):
            self.raw_dir.mkdir(parents=True, exist_ok=True)  # earthcarekit won't create it
            eck.ecdownload(file_type=self.file_type, orbit_and_frame=orbit_and_frame,
                           path_to_data=str(self.raw_dir), is_create_subdirs=False, verbose=False)
        hits = sorted(self.raw_dir.rglob(pattern))  # latest processing time sorts last
        if not hits:
            raise FileNotFoundError(f"no {self.file_type} file for {orbit_and_frame}")
        return hits[-1]

    def _read(self, path):
        """convert one ACM-CAP frame into labels for its visible profiles on LEVELS_KM.

        lwc/rwc are fill values (nan) wherever there is no liquid/rain, so they
        become 0. voxels outside the retrieval (liquid_classification < 0, e.g.
        below ground) and failed profiles (quality_status >= 8) are set to nan.
        the quality cutoff is an assumption, not directly from the paper.
        """
        with eck.read_any(str(path)) as ds:
            # drop profiles goes-19 can't see
            ds = ds.isel(along_track=self.visible(ds["latitude"].values, ds["longitude"].values))
            # ice/liquid/rain water content and effective radii are the C3DIR targets (paper section 2.2)
            water = np.stack([ds[v].values for v in
                              ("ice_water_content", "liquid_water_content", "rain_water_content")],
                             axis=-1) * 1000.0  # kg m^-3 -> g m^-3
            radius = np.stack([ds["ice_effective_radius"].values,
                               ds["liquid_effective_radius"].values], axis=-1) * 1e6  # m -> µm
            water[..., 1:] = np.nan_to_num(water[..., 1:])  # absent liquid/rain is zero, not missing
            # below ground or failed retrieval -> nan
            invalid = ~(ds["liquid_classification"].values >= 0) | \
                (ds["quality_status"].values >= 8)[:, None]
            water[invalid] = radius[invalid] = np.nan
            z_km = np.broadcast_to(ds["height"].values / 1000.0, invalid.shape)
            profiles = {"time": ds["time"].values, "lat": ds["latitude"].values,
                        "lon": ds["longitude"].values}

        # average the native ~100 m levels down to the 0.25 km C3DIR grid (paper section 2.3)
        level = np.floor(z_km / 0.25).astype(int)  # bin k spans [0.25k, 0.25k + 0.25) km
        water = self._bin_mean(water, level, np.isfinite(water).all(-1))
        radius = self._bin_mean(radius, level, np.isfinite(radius))
        # detection masks from the water content threshold, C3DIR paper section 2.2
        masks = np.where(np.isnan(water), np.nan, water > MASK_THRESHOLD)
        hydrometeor = np.where(np.isnan(water).any(-1), np.nan, masks.any(-1))  # anything at all

        # assemble the dataset
        dims = ("profile", "level")
        variables = {
            "iwc": water[..., 0], "lwc": water[..., 1], "rwc": water[..., 2],
            "ice_radius": radius[..., 0], "liquid_radius": radius[..., 1],
            "ice_mask": masks[..., 0], "liquid_mask": masks[..., 1], "rain_mask": masks[..., 2],
            "hydrometeor_mask": hydrometeor,
        }
        units = {"iwc": "g m-3", "lwc": "g m-3", "rwc": "g m-3",
                 "ice_radius": "um", "liquid_radius": "um"}
        return xr.Dataset(
            {k: (dims, v.astype(np.float32), {"units": units.get(k, "1")}) for k, v in variables.items()}
            | {k: ("profile", v) for k, v in profiles.items()},
            coords={"level": ("level", LEVELS_KM, {"units": "km"})},
        )

    @staticmethod
    def _bin_mean(values, level, valid):
        """average [P, Z, n] native samples into [P, len(LEVELS_KM), n]; empty bins come out nan.

        valid: either [P, Z] (all n fields valid together) or [P, Z, n] (each field on its own).
        """
        valid = np.broadcast_to(valid if valid.ndim == 3 else valid[..., None], values.shape)
        inside = (level >= 0) & (level < len(LEVELS_KM))  # drop anything below 0 or above 20 km
        total = np.zeros((len(level), len(LEVELS_KM), values.shape[-1]))
        count = np.zeros_like(total)
        # sum values and counts per bin, then divide for the mean
        for i in range(values.shape[-1]):
            ok = valid[..., i] & inside
            p = np.broadcast_to(np.arange(len(level))[:, None], level.shape)[ok]
            np.add.at(total[..., i], (p, level[ok]), values[..., i][ok])
            np.add.at(count[..., i], (p, level[ok]), 1)
        with np.errstate(invalid="ignore"):  # empty bins are 0/0 -> nan, as intended
            return total / count
