"""C3DIR training samples, built from the collocated goes-19 / earthcare pairs.

two steps, both cached in PROCESSED_DIR/c3dir/samples/:

  Samples.build()   one sample per collocated pair: the goes crop as a (band, y, x)
                    image plus the matched voxels (level, row, col and 9 targets)
                    -> <pair>.nc
  Samples.split()   assigns every sample to train / val / test by date and computes
                    per-band normalization from train -> splits.csv and band_stats.csv

model-train cuts patches from these (c3dir.dataset). see docs/c3dir.md
for the formats, sizes and assumptions.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from c3dir.constants import BANDS
from utilities.paths import PROCESSED_DIR

VOXEL_VARS = ["voxel_level", "voxel_row", "voxel_col", "voxel_profile", "voxel_distance_km",
              "voxel_dt_seconds", "voxel_labels"]


class Samples:
    """
    collocated_dir: the collocated pairs. samples_dir: where the samples, splits.csv and
    band_stats.csv go.
    """

    def __init__(self, collocated_dir=PROCESSED_DIR / "c3dir" / "collocated",
                 samples_dir=PROCESSED_DIR / "c3dir" / "samples"):
        self.collocated_dir = Path(collocated_dir)
        self.samples_dir = Path(samples_dir)
        self.splits_file = self.samples_dir / "splits.csv"
        self.stats_file = self.samples_dir / "band_stats.csv"

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

    def split(self, val_fraction=0.1, test_fraction=0.1, val_after=None, test_after=None):
        """assign every sample to train / val / test by the date of its goes scan.

        all samples from one day share a split, so neighboring pixels on a
        track never land on both sides. the split comes from a hash of the date,
        so a day never moves when more data is scraped.
        val_after / test_after: optional dates for held-out later periods, as in the paper:
        days from val_after go to val, days from test_after to test, and the hash only
        splits the days before the first of them.

        also writes the mean/std of each band over the train samples to
        band_stats.csv for training to normalize with.
        """
        rows = []
        for path in sorted(self.samples_dir.glob("*.nc")):
            with xr.open_dataset(path) as s:
                start = pd.Timestamp(s.attrs["scan_start"])
                rows.append({"file": path.name, "date": start.date().isoformat(),
                             "frame": s.attrs["orbit_and_frame"], "scan_start": start,
                             "n_voxels": s.sizes["voxel"]})
        table = pd.DataFrame(rows)
        if table.empty:
            raise FileNotFoundError(f"no samples in {self.samples_dir}; run build() first")

        def which(date):
            if test_after is not None and pd.Timestamp(date) >= pd.Timestamp(test_after):
                return "test"
            if val_after is not None and pd.Timestamp(date) >= pd.Timestamp(val_after):
                return "val"
            # md5 instead of hash() since python randomizes hash() between runs
            u = int(hashlib.md5(date.encode()).hexdigest(), 16) % 10_000 / 10_000
            return "test" if u < test_fraction else "val" if u < test_fraction + val_fraction else "train"

        table["split"] = table["date"].map(which)
        table.to_csv(self.splits_file, index=False)

        # per-band mean/std over the train images, for normalizing the inputs
        train = table[table.split == "train"]
        if len(train):
            total = np.zeros(len(BANDS))
            total_sq = np.zeros(len(BANDS))
            count = np.zeros(len(BANDS))
            for name in train["file"]:
                with xr.open_dataset(self.samples_dir / name) as s:
                    image = s["goes"].values.reshape(len(BANDS), -1).astype(np.float64)
                ok = np.isfinite(image)
                total += np.where(ok, image, 0).sum(1)
                total_sq += np.where(ok, image ** 2, 0).sum(1)
                count += ok.sum(1)
            mean = total / count
            std = np.sqrt(total_sq / count - mean ** 2)
            pd.DataFrame({"band": BANDS, "mean": mean, "std": std}).to_csv(self.stats_file, index=False)

        summary = table.groupby("split").agg(days=("date", "nunique"), samples=("file", "size"),
                                             voxels=("n_voxels", "sum"))
        print(summary.to_string())
        return table
