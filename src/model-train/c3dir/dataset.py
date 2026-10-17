"""C3DIR training patches, cut from data-processing's samples (c3dir.samples).

the samples (PROCESSED_DIR/c3dir/samples/<pair>.nc), their train / val / test split
(splits.csv) and per-band normalization (band_stats.csv) come from
c3dir.samples. C3DIRDataset cuts fixed-size patches along the track for one
split. labels are a list of voxels with their coordinates in the model's output volume, as in
the C3DIR paper (section 4): the model predicts every voxel, and the loss uses only the labeled
ones. the vertical axis of that volume follows each pixel's slanted line of sight (paper
section 3.1). it needs only numpy and works with torch.utils.data.DataLoader as-is (pass
collate_fn=collate).

see docs/c3dir.md for the formats, sizes and assumptions.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from c3dir.constants import BANDS
from c3dir.paths import PROCESSED_DIR

SAMPLES_DIR = PROCESSED_DIR / "c3dir" / "samples"


@lru_cache(maxsize=None)
def _load(path):
    """the arrays of a sample file, kept in memory after the first read (~10-15 MB per sample),
    since patches are drawn from all files at random."""
    with xr.open_dataset(path) as s:
        return {k: s[k].values for k in ["goes", "voxel_level", "voxel_row", "voxel_col", "voxel_labels"]}


class C3DIRDataset:
    """patches cut along the track for one split.

    split: "train", "val" or "test" (from splits.csv in samples_dir).
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

    def __init__(self, split, samples_dir=SAMPLES_DIR, patch=64, stride=None, normalize=True):
        samples_dir = Path(samples_dir)
        table = pd.read_csv(samples_dir / "splits.csv")
        self.files = [samples_dir / f for f in table.loc[table.split == split, "file"]]
        self.patch = patch
        stride = stride or patch // 2

        self.mean = self.std = None
        if normalize:
            stats = pd.read_csv(samples_dir / "band_stats.csv").set_index("band").loc[BANDS]
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
