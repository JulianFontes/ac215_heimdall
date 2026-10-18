"""the train / val / test split and the input normalization, decided at training time from every
sample present (data-processing only builds the per-day samples, c3dir.samples):

  table = split_samples(val_after=None, test_after=None)   one row per sample, with its split
  stats = band_stats(table)                                 per-band mean / std over the train samples

all samples from one day share a split, so neighboring pixels on a track never land on both sides.
the split comes from a hash of the date, so a day keeps its split however much data is added and
whichever run computes it. c3dir.train saves both tables with each model version (splits.csv,
band_stats.csv), as the record of what it trained on.
"""
import hashlib

import numpy as np
import pandas as pd
import xarray as xr

from c3dir.constants import BANDS
from utilities.paths import PROCESSED_DIR

SAMPLES_DIR = PROCESSED_DIR / "c3dir" / "samples"


def split_samples(samples_dir=SAMPLES_DIR, val_fraction=0.1, test_fraction=0.1, val_after=None, test_after=None):
    """assign every sample to train / val / test by the date of its goes scan.

    val_after / test_after: optional dates for held-out later periods, as in the paper: days from
    val_after go to val, days from test_after to test, and the hash only splits the days before
    the first of them.
    """
    rows = []
    for path in sorted(samples_dir.glob("*.nc")):
        with xr.open_dataset(path) as s:
            start = pd.Timestamp(s.attrs["scan_start"])
            rows.append({"file": path.name, "date": start.date().isoformat(),
                         "frame": s.attrs["orbit_and_frame"], "scan_start": start,
                         "n_voxels": s.sizes["voxel"]})
    table = pd.DataFrame(rows)
    if table.empty:
        raise FileNotFoundError(f"no samples in {samples_dir}; run data-processing's c3dir.process first")

    def which(date):
        if test_after is not None and pd.Timestamp(date) >= pd.Timestamp(test_after):
            return "test"
        if val_after is not None and pd.Timestamp(date) >= pd.Timestamp(val_after):
            return "val"
        # md5 instead of hash() since python randomizes hash() between runs
        u = int(hashlib.md5(date.encode()).hexdigest(), 16) % 10_000 / 10_000
        return "test" if u < test_fraction else "val" if u < test_fraction + val_fraction else "train"

    table["split"] = table["date"].map(which)
    return table


def band_stats(table, samples_dir=SAMPLES_DIR):
    """per-band mean and std over the train samples, for normalizing the inputs."""
    total = np.zeros(len(BANDS))
    total_sq = np.zeros(len(BANDS))
    count = np.zeros(len(BANDS))
    for name in table.loc[table.split == "train", "file"]:
        with xr.open_dataset(samples_dir / name) as s:
            image = s["goes"].values.reshape(len(BANDS), -1).astype(np.float64)
        ok = np.isfinite(image)
        total += np.where(ok, image, 0).sum(1)
        total_sq += np.where(ok, image ** 2, 0).sum(1)
        count += ok.sum(1)
    if not count.any():
        raise ValueError("no train samples to compute band statistics from")
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return pd.DataFrame({"band": BANDS, "mean": mean, "std": std})


def summary(table):
    """days, samples and labeled voxels per split."""
    return table.groupby("split").agg(days=("date", "nunique"), samples=("file", "size"),
                                      voxels=("n_voxels", "sum"))
