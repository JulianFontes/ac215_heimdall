"""C3DIR training samples, built from the collocated goes-19 / earthcare pairs, cached in
PROCESSED_DIR/c3dir/samples/:

  Samples.build()   one sample per collocated pair: the goes crop as a (band, y, x)
                    image plus the matched voxels (level, row, col and 9 targets)
                    -> <pair>.nc

each sample stands alone, so processing only adds files. model-train splits the samples into
train / val / test, computes the band normalization (c3dir.splits) and cuts patches
(c3dir.dataset). see docs/c3dir.md for the formats, sizes and assumptions.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from c3dir.constants import BANDS
from utilities.paths import PROCESSED_DIR

VOXEL_VARS = ["voxel_level", "voxel_row", "voxel_col", "voxel_profile", "voxel_distance_km",
              "voxel_dt_seconds", "voxel_labels"]


class Samples:
    """
    collocated_dir: the collocated pairs. samples_dir: where the samples go.
    """

    def __init__(self, collocated_dir=PROCESSED_DIR / "c3dir" / "collocated",
                 samples_dir=PROCESSED_DIR / "c3dir" / "samples"):
        self.collocated_dir = Path(collocated_dir)
        self.samples_dir = Path(samples_dir)

    def build(self, overwrite=False):
        """turn every collocated pair into a sample file; skips ones already built.

        each <pair>.nc has the goes image as goes (band, y, x) plus the matched voxels
        (voxel). returns the list of sample files.
        """
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        built = []
        for path in sorted(self.collocated_dir.glob("*.nc")):
            out = self.samples_dir / path.name
            if overwrite or not out.exists():
                pair = xr.load_dataset(path)
                image = np.stack([pair[b].values for b in BANDS]).astype(np.float32)
                sample = pair[VOXEL_VARS + ["goes_imager_projection"]].drop_vars("level", errors="ignore")
                sample["goes"] = (("band", "y", "x"), image,
                                  {"description": "ABI bands 1-6 reflectance (1), bands 7-16 brightness temperature (K)"})
                sample = sample.assign_coords(band=BANDS, y=pair.y, x=pair.x)
                sample.attrs = pair.attrs
                sample.to_netcdf(out, encoding={"goes": {"zlib": True, "complevel": 4}})  # roughly halves the file size
            built.append(out)
        print(f"{len(built)} samples in {self.samples_dir}")
        return built
