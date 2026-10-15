"""training samples for C3DIR, built from the collocated goes-19 / earthcare pairs.

two steps, both cached on disk next to the collocated pairs:

  Samples.build()   one sample per collocated pair: the goes crop as a (band, y, x)
                    image plus the matched voxels (level, row, col and 9 targets)
                    -> <data_dir>/samples/<pair>.nc
  Samples.split()   assigns every sample to train / val / test by date and computes
                    per-band normalization from train -> <data_dir>/samples/splits.csv
                    and band_stats.csv

C3DIRDataset then cuts fixed-size patches along the track for one split. labels are a
list of voxels with their coordinates in the model's output volume, as in the C3DIR
paper (section 4): the model predicts every voxel, and the loss uses only the labeled
ones. the vertical axis of that volume follows each pixel's slanted line of sight
(paper section 3.1). it needs only numpy and works with torch.utils.data.DataLoader
as-is (pass collate_fn=collate).

see README.md in this folder for the formats, sizes and assumptions.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from heimdall.c3dir.constants import BANDS, TARGETS
from heimdall.paths import DATA_DIR

VOXEL_VARS = ["voxel_level", "voxel_row", "voxel_col", "voxel_profile", "voxel_distance_km",
              "voxel_dt_seconds", "voxel_labels"]


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
        band_stats.csv for C3DIRDataset to normalize with.
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


@lru_cache(maxsize=None)
def _load(path):
    """the arrays of a sample file, kept in memory after the first read (~10-15 MB per sample),
    since patches are drawn from all files at random."""
    with xr.open_dataset(path) as s:
        return {k: s[k].values for k in ["goes", "voxel_level", "voxel_row", "voxel_col", "voxel_labels"]}


class C3DIRDataset:
    """patches cut along the track for one split.

    split: "train", "val" or "test" (from splits.csv).
    patch: patch height and width in goes pixels; 64 as in the C3DIR paper (section 2.3).
        patches that don't fit inside the crop are padded.
    stride: how many matched goes pixels to move along the track between patch
        centers; default patch // 2, so most voxels show up in ~2 patches.
    normalize: scale each band by the train mean/std from band_stats.csv.

    each item is a dict:
      x       (16, patch, patch) float32, normalized; nan (space / padding) -> 0
      valid   (patch, patch) bool, where x has real data
      index   (n, 3) int, (level, row, col) of each labeled voxel in the output volume
      y       (n, 9) float32 labels in physical units, in TARGETS order (radii nan
              where that phase is absent)
    n changes from patch to patch, so batch them with collate().
    """

    def __init__(self, split, data_dir=DATA_DIR, patch=64, stride=None, normalize=True):
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

        # patch centers: every stride-th matched pixel, in the order the track reaches them
        self.index = []
        for i, path in enumerate(self.files):
            with xr.open_dataset(path) as s:
                v = pd.DataFrame({k: s[f"voxel_{k}"].values for k in ["row", "col", "profile"]})
            pixels = v.groupby(["row", "col"])["profile"].mean().sort_values().index
            centers = pixels[stride // 2::stride] if len(pixels) > stride // 2 else pixels[:1]
            self.index += [(i, row, col) for row, col in centers]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, row, col = self.index[k]
        s = _load(str(self.files[i]))
        (_, H, W), P = s["goes"].shape, self.patch

        # top-left corner: centered on the pixel, pushed back inside the crop where possible
        r0 = int(np.clip(row - P // 2, 0, max(H - P, 0)))
        c0 = int(np.clip(col - P // 2, 0, max(W - P, 0)))
        image = s["goes"][:, r0:r0 + P, c0:c0 + P]
        x = np.full((len(BANDS), P, P), np.nan, dtype=np.float32)  # pad if the crop is smaller than a patch
        x[:, :image.shape[1], :image.shape[2]] = image
        valid = np.isfinite(x).all(0)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0)  # 0 = the band mean after normalizing

        # every labeled voxel inside this patch
        rows, cols = s["voxel_row"], s["voxel_col"]
        inside = (rows >= r0) & (rows < r0 + P) & (cols >= c0) & (cols < c0 + P)
        index = np.stack([s["voxel_level"][inside], rows[inside] - r0, cols[inside] - c0], axis=1)
        return {"x": x, "valid": valid, "index": index.astype(np.int64),
                "y": s["voxel_labels"][inside].astype(np.float32)}


def collate(items):
    """batch dataset items for a DataLoader: pads the labeled voxels to the most in the
    batch and stacks everything.

    gives numpy arrays: x (B, 16, P, P), valid (B, P, P), index (B, n, 3), y (B, n, 9)
    and mask (B, n), False for padding (padding has index 0 and y nan). wrap them with
    torch.as_tensor in the training loop.
    """
    n = max(len(item["index"]) for item in items)

    def pad(a, fill):
        out = np.full((n,) + a.shape[1:], fill, dtype=a.dtype)
        out[:len(a)] = a
        return out

    return {
        "x": np.stack([item["x"] for item in items]),
        "valid": np.stack([item["valid"] for item in items]),
        "index": np.stack([pad(item["index"], 0) for item in items]),
        "y": np.stack([pad(item["y"], np.nan) for item in items]),
        "mask": np.stack([np.arange(n) < len(item["index"]) for item in items]),
    }
