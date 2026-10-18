"""C3DIR labels from earthcare ACM-CAP frames (White et al., 2026, sections 2.2-2.3).

converts each raw frame (RAW_DIR/earthcare/, from c3dir.earthcare) to the C3DIR targets:
ice, liquid and rain water content, ice and liquid effective radius, and four detection masks
(water content > 1e-5 g m^-3), averaged from ~100 m onto 80 levels of 0.25 km.

levels: the paper gives "-0.125 to 19.875 km", but its figure 2 levels (0.125, 14.875 km) and
the 80-level output only fit 0-20 km bins, so the level centers here are 0.125 ... 19.875 km.

each frame is cached as PROCESSED_DIR/c3dir/labels/<orbit_and_frame>.nc.
"""
from pathlib import Path

import earthcarekit as eck
import numpy as np
import xarray as xr

from c3dir.constants import GOES19_LON, LEVELS_KM, MASK_THRESHOLD, VIEW_RADIUS_DEG
from c3dir.geometry import within_view
from utilities.paths import PROCESSED_DIR, RAW_DIR


class EarthCareLabels:
    """
    raw_dir: the collector's .h5 frames. label_dir: where the labels are cached.
    view_radius_deg: profiles beyond this earth-center angle from the point under goes-19 are
        dropped (~73° viewing angle); an assumption to limit the data, not directly from the paper.
    """

    file_type = "ACM_CAP_2B"

    def __init__(self, raw_dir=RAW_DIR / "earthcare", label_dir=PROCESSED_DIR / "c3dir" / "labels",
                 lon0=GOES19_LON, view_radius_deg=VIEW_RADIUS_DEG):
        self.raw_dir = Path(raw_dir)
        self.label_dir = Path(label_dir)
        self.lon0 = lon0
        self.view_radius_deg = view_radius_deg

    def load(self, orbit_and_frame):
        """labels for one frame as an xarray dataset; built and cached on first use.

        dims: profile (along track, ~1 km apart) x level (0.25 km).
        variables: time/lat/lon per profile; iwc/lwc/rwc [g m^-3] and the four
        masks per (profile, level), nan where ACM-CAP has no valid retrieval;
        ice_radius/liquid_radius [µm], nan where that phase is absent.
        """
        cache = self.label_dir / f"{orbit_and_frame}.nc"
        if not cache.exists():
            hits = sorted(self.raw_dir.rglob(f"ECA_*_{self.file_type}_*_{orbit_and_frame}.h5"))
            if not hits:
                raise FileNotFoundError(f"no raw {self.file_type} frame for {orbit_and_frame} in {self.raw_dir}; "
                                        "collect it with c3dir.collect first")
            labels = self._read(hits[-1])  # latest processing time sorts last
            labels.attrs["orbit_and_frame"] = orbit_and_frame
            cache.parent.mkdir(parents=True, exist_ok=True)
            labels.to_netcdf(cache)
        return xr.load_dataset(cache)

    def _read(self, path):
        """convert one ACM-CAP frame into labels for its visible profiles on LEVELS_KM.

        lwc/rwc are fill values (nan) wherever there is no liquid/rain, so they
        become 0. voxels outside the retrieval (liquid_classification < 0, e.g.
        below ground) and failed profiles (quality_status >= 8) are set to nan.
        the quality cutoff is an assumption, not directly from the paper.
        """
        with eck.read_any(str(path)) as ds:
            # drop profiles goes-19 can't see
            ds = ds.isel(along_track=within_view(ds["latitude"].values, ds["longitude"].values,
                                                 self.lon0, self.view_radius_deg))
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
