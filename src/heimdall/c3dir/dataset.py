"""training samples for C3DIR, built from the collocated goes-19 / earthcare pairs.

two steps, both cached on disk next to the collocated pairs:

  Samples.build()   one sample per collocated pair: the goes crop as a (band, y, x)
                    image plus the earthcare labels averaged into one column per
                    goes pixel the track crosses -> <data_dir>/samples/<pair>.nc
  Samples.split()   assigns every sample to train / val / test by date and computes
                    per-band normalization from train -> <data_dir>/samples/splits.csv
                    and band_stats.csv

C3DIRDataset then cuts fixed-size patches along the track for one split. it needs
only numpy and works with torch.utils.data.DataLoader as-is (pass collate_fn=collate).

see README.md in this folder for the formats, sizes and assumptions.
"""
import hashlib
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from heimdall.c3dir.goes19 import BANDS, goes_proj
from heimdall.paths import DATA_DIR

# the label columns in the order the model sees them: Y[..., i, level] is TARGETS[i]
WATER = ["iwc", "lwc", "rwc"]  # g m^-3
RADII = ["ice_radius", "liquid_radius"]  # µm
MASKS = ["ice_mask", "liquid_mask", "rain_mask", "hydrometeor_mask"]  # fraction of profiles, 0-1
TARGETS = WATER + RADII + MASKS


def track_pixels(pair):
    """(row, col) of the goes pixel each profile in a collocated pair falls in.

    nearest pixel to the profile's ground lat/lon, no parallax correction, same
    rule as GOES19.collocate().
    """
    h = pair["goes_imager_projection"].attrs["perspective_point_height"]
    px, py = goes_proj(pair)(pair["lon"].values, pair["lat"].values)
    x, y = pair.x.values, pair.y.values
    # pixel centers are evenly spaced, so position / spacing is the index
    row = np.rint((np.asarray(py) / h - y[0]) / (y[1] - y[0])).astype(int)
    col = np.rint((np.asarray(px) / h - x[0]) / (x[1] - x[0])).astype(int)
    return row, col


def pixel_columns(pair):
    """average the earthcare profiles that land in the same goes pixel.

    returns a dataset over (pixel, level), pixels in the order the track
    crosses them:
      row, col          where the pixel sits in the goes crop
      n_profiles        how many profiles went into it (~2 in the tropics, ~5 near 60°)
      lat, lon, time, dt_seconds   averaged over those profiles
      iwc, lwc, rwc, ice_radius, liquid_radius   mean over the profiles with a value
      <those>_std       spread over the profiles (nan if fewer than 2 had a value);
                        measured sub-pixel variability, for checking predicted uncertainty
      *_mask            fraction of profiles with that hydrometeor, a soft 0-1 label
    """
    row, col = track_pixels(pair)
    key = row * pair.sizes["x"] + col
    # np.unique sorts by key, so reorder the pixels by when the track first hits them
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    pixel = rank[inverse]  # pixel index of every profile, 0 = first along the track
    n = len(order)

    # profiles sorted by pixel, so each pixel's profiles are one contiguous slice
    by_pixel = np.argsort(pixel, kind="stable")
    counts = np.bincount(pixel, minlength=n)
    bounds = np.r_[0, np.cumsum(counts)]

    def group(values):
        """mean and std over the profiles in each pixel, ignoring nans: [profile, ...] -> [pixel, ...]."""
        values = values[by_pixel].astype(np.float64)
        mean = np.full((n,) + values.shape[1:], np.nan)
        std = np.full_like(mean, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-nan levels stay nan
            for p in range(n):
                chunk = values[bounds[p]:bounds[p + 1]]
                mean[p] = np.nanmean(chunk, axis=0)
                std[p] = np.where(np.isfinite(chunk).sum(0) >= 2, np.nanstd(chunk, axis=0), np.nan)
        return mean.astype(np.float32), std.astype(np.float32)

    profile_vars = {}
    for v in WATER + RADII + MASKS:
        mean, std = group(pair[v].values)
        profile_vars[v] = (("pixel", "level"), mean, pair[v].attrs)
        if v not in MASKS:
            profile_vars[f"{v}_std"] = (("pixel", "level"), std,
                                        {**pair[v].attrs, "description": "std over the profiles in this pixel"})

    seconds = (pair["time"].values - pair["time"].values[0]) / np.timedelta64(1, "s")
    mean_time = pair["time"].values[0] + (group(seconds)[0].astype(np.float64) * 1e9).astype("timedelta64[ns]")
    per_pixel = {
        "row": ("pixel", row[first[order]].astype(np.int32)),
        "col": ("pixel", col[first[order]].astype(np.int32)),
        "n_profiles": ("pixel", counts.astype(np.int32)),
        "lat": ("pixel", group(pair["lat"].values)[0]),
        "lon": ("pixel", group(pair["lon"].values)[0]),
        "time": ("pixel", mean_time),
        "dt_seconds": ("pixel", group(pair["dt_seconds"].values)[0]),
    }
    return xr.Dataset(per_pixel | profile_vars, coords={"level": pair["level"]})


class Samples:
    """
    data_dir: reads <data_dir>/collocated, writes <data_dir>/samples.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.collocated_dir = Path(data_dir) / "collocated"
        self.samples_dir = Path(data_dir) / "samples"
        self.splits_file = self.samples_dir / "splits.csv"
        self.stats_file = self.samples_dir / "band_stats.csv"

    def build(self, overwrite=False):
        """turn every collocated pair into a sample file; skips ones already built.

        each <pair>.nc has the goes image as goes (band, y, x) plus everything from
        pixel_columns() over (pixel, level). returns the list of sample files.
        """
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        built = []
        for path in sorted(self.collocated_dir.glob("*.nc")):
            out = self.samples_dir / path.name
            if overwrite or not out.exists():
                pair = xr.load_dataset(path)
                image = np.stack([pair[b].values for b in BANDS]).astype(np.float32)
                sample = pixel_columns(pair)
                sample["goes"] = (("band", "y", "x"), image,
                                  {"description": "ABI bands 1-6 reflectance (1), bands 7-16 brightness temperature (K)"})
                sample = sample.assign_coords(band=BANDS, y=pair.y, x=pair.x)
                sample["goes_imager_projection"] = pair["goes_imager_projection"]
                sample.attrs = pair.attrs
                sample.to_netcdf(out, encoding={"goes": {"zlib": True, "complevel": 4}})  # roughly halves the file size
            built.append(out)
        print(f"{len(built)} samples in {self.samples_dir}")
        return built

    def split(self, val_fraction=0.1, test_fraction=0.1, test_after=None):
        """assign every sample to train / val / test by the date of its goes scan.

        all samples from one day share a split, so neighbouring pixels on a
        track never land on both sides. the split comes from a hash of the date,
        so a day never moves when more data is scraped.
        test_after: optional date; every day from then on goes to test (a held-out
        later period), and the hash splits only the days before it.

        also writes the mean/std of each band over the train samples to
        band_stats.csv for C3DIRDataset to normalize with.
        """
        rows = []
        for path in sorted(self.samples_dir.glob("*.nc")):
            with xr.open_dataset(path) as s:
                start = pd.Timestamp(s.attrs["scan_start"])
                rows.append({"file": path.name, "date": start.date().isoformat(),
                             "frame": s.attrs["orbit_and_frame"], "scan_start": start,
                             "n_pixels": s.sizes["pixel"]})
        table = pd.DataFrame(rows)
        if table.empty:
            raise FileNotFoundError(f"no samples in {self.samples_dir}; run build() first")

        def which(date):
            if test_after is not None and pd.Timestamp(date) >= pd.Timestamp(test_after):
                return "test"
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
                                             pixels=("n_pixels", "sum"))
        print(summary.to_string())
        return table


@lru_cache(maxsize=32)
def _load(path):
    """cache recent sample files in memory; consecutive patches usually come from the same file."""
    return xr.load_dataset(path)


class C3DIRDataset:
    """patches cut along the track for one split.

    split: "train", "val" or "test" (from splits.csv).
    patch: patch height and width in goes pixels. crops are only ~130-300 px wide
        (64 px of padding each side of the track), so 128 is about the largest
        that stays inside the crop; patches that don't fit are padded.
    stride: how many track pixels to move between patch centers; default patch // 2,
        so each labelled pixel shows up in ~2 patches.
    normalize: scale each band by the train mean/std from band_stats.csv.

    each item is a dict:
      x         (16, patch, patch) float32, normalized; nan (space / padding) -> 0
      valid     (patch, patch) bool, where x has real data
      track_rc  (n, 2) int, row/col inside the patch of each labelled pixel
      y         (n, len(TARGETS), 80) float32 labels in physical units, nan = no label
      y_std     (n, 5, 80) float32 spread over the profiles in each pixel (water + radii)
      n_profiles (n,) int
    n changes from patch to patch, so batch them with collate().
    """

    def __init__(self, split, data_dir=DATA_DIR, patch=128, stride=None, normalize=True):
        samples = Samples(data_dir)
        table = pd.read_csv(samples.splits_file)
        self.files = [samples.samples_dir / f for f in table.loc[table.split == split, "file"]]
        self.patch = patch
        stride = stride or patch // 2

        self.mean = self.std = None
        if normalize:
            stats = pd.read_csv(samples.stats_file).set_index("band").loc[BANDS]
            self.mean = stats["mean"].values.astype(np.float32)[:, None, None]
            self.std = stats["std"].values.astype(np.float32)[:, None, None]

        # one patch centered on every stride-th labelled pixel of every sample
        self.index = []
        for i, path in enumerate(self.files):
            with xr.open_dataset(path) as s:
                n = s.sizes["pixel"]
            self.index += [(i, p) for p in range(stride // 2, n, stride)] or [(i, n // 2)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, p = self.index[k]
        s = _load(str(self.files[i]))
        rows, cols = s["row"].values, s["col"].values
        H, W, P = s.sizes["y"], s.sizes["x"], self.patch

        # top-left corner: centered on pixel p, pushed back inside the crop where possible
        r0 = int(np.clip(rows[p] - P // 2, 0, max(H - P, 0)))
        c0 = int(np.clip(cols[p] - P // 2, 0, max(W - P, 0)))
        image = s["goes"].values[:, r0:r0 + P, c0:c0 + P]
        x = np.full((len(BANDS), P, P), np.nan, dtype=np.float32)  # pad if the crop is smaller than a patch
        x[:, :image.shape[1], :image.shape[2]] = image
        valid = np.isfinite(x).all(0)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0)  # 0 = the band mean after normalizing

        # every labelled pixel that lands inside this patch, not just the center one
        inside = (rows >= r0) & (rows < r0 + P) & (cols >= c0) & (cols < c0 + P)
        y = np.stack([s[v].values[inside] for v in TARGETS], axis=1)
        y_std = np.stack([s[f"{v}_std"].values[inside] for v in WATER + RADII], axis=1)
        return {
            "x": x, "valid": valid,
            "track_rc": np.stack([rows[inside] - r0, cols[inside] - c0], axis=1).astype(np.int64),
            "y": y.astype(np.float32), "y_std": y_std.astype(np.float32),
            "n_profiles": s["n_profiles"].values[inside],
        }


def collate(items):
    """batch dataset items for a DataLoader: pads the labelled pixels to the most in
    the batch (padding has y = nan, track_rc = 0, n_profiles = 0) and stacks everything.

    gives numpy arrays: x (B, 16, P, P), valid (B, P, P), track_rc (B, n, 2),
    y (B, n, 9, 80), y_std (B, n, 5, 80), n_profiles (B, n). wrap them with
    torch.as_tensor in the training loop.
    """
    n = max(len(item["track_rc"]) for item in items)

    def pad(a, fill):
        out = np.full((n,) + a.shape[1:], fill, dtype=a.dtype)
        out[:len(a)] = a
        return out

    return {
        "x": np.stack([item["x"] for item in items]),
        "valid": np.stack([item["valid"] for item in items]),
        "track_rc": np.stack([pad(item["track_rc"], 0) for item in items]),
        "y": np.stack([pad(item["y"], np.nan) for item in items]),
        "y_std": np.stack([pad(item["y_std"], np.nan) for item in items]),
        "n_profiles": np.stack([pad(item["n_profiles"], 0) for item in items]),
    }
